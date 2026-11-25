#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Split TVM's single generated C file into one folder per layer.

``relay.build(target="c")`` emits every kernel into one translation unit
(``<stem>.c``: 4.5k lines, 22 kernels for the zoo default). That is awkward for per-layer work --
offloading one conv to AIE, diffing a single kernel, compiling layers
separately -- so this splits it into ``layers/``, **one folder per layer, in
graph execution order**:

    layers/
      layers_common.h                 includes + every forward declaration
      00_conv2d_add_relu/
        00_conv2d_add_relu.c
      01_max_pool2d/
        01_max_pool2d.c
      02_conv2d_add_relu/
      03_conv2d_add_add_relu/
      ...
      21_batch_flatten/
      manifest.json                   folder <-> TVM symbol <-> ops <-> calls
      Makefile                        builds every layer to a .o
      README.md

**One folder per layer, not per op kind.** The folder is a *per-layer working
directory*, so reading ``layers/`` top to bottom is reading the network in the
order it executes. It holds a single ``.c`` today, but it is the place the
AIE artifacts for that layer will live alongside it -- ``.bcf``, kernel
sources, the generated ``host.cc``/``routing.cc``, a per-layer Makefile
fragment. Those are per-layer, not per-op-kind, which is why the folder is keyed
by execution index.

Ordering comes from ``<stem>_graph.json``; without it the source order is
used and every file says so. Pass ``group=False`` (CLI ``--flat``) to get one
flat directory of ``.c`` files with no per-layer folders -- useful only when
nothing else will ever sit beside them.

The name is the fused op list: ``fused_nn_conv2d_expand_dims_add_add_nn_relu``
becomes ``conv2d_add_add_relu`` -- ``nn_`` stripped, ``expand_dims`` dropped as
a bias-broadcast artifact, the two ``add``s being the bias and the residual.
The full TVM symbol is never lost: it stays in a header comment and in the
manifest.

**There is no standalone ``relu.c``, and that is not a bug.** TVM's ``FuseOps``
welds elementwise ops into their producer, so the compiled unit really is
"conv2d+bias+residual+relu" -- one loop nest, one function. Splitting cannot
undo that. To get true per-op files, build with fusion off:

    python src/frontend/tvmrelay/deploy_flow.py --no-fuse

Each emitted file is a complete translation unit: it includes
``layers_common.h`` and compiles on its own.

Standalone use, against an already-generated tree:

    python src/frontend/tvmrelay/split_layers.py worklocal/tvmrelay_deploy
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
import sys
from pathlib import Path

__all__ = ["split_layers", "short_layer_name", "iter_c_functions", "split_module"]

#: The 3-line guard TVM puts before every declaration and every definition.
_EXTERN_C = '#ifdef __cplusplus\nextern "C"\n#endif\n'

#: Start of a function *definition* (a declaration ends in ``;`` instead).
_DEF_RE = re.compile(
    re.escape(_EXTERN_C) + r"(TVM_DLL\s+\w[\w\s\*]*?\s(\w+)\s*\([^;{]*?\)\s*\{)"
)

_PREFIX = "tvmgen_default_fused_"

#: Dropped from layer names: a broadcast of the conv bias to NCHW, not a layer
#: anyone thinks in terms of.
_NOISE_OPS = ("expand_dims",)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


_BRACE_SCAN = re.compile(r"[\"'{}]|//|/\*")


def _match_brace(source: str, open_idx: int) -> int:
    """Index just past the ``}`` closing the ``{`` at *open_idx*.

    String and char literals are skipped: generated code calls
    ``TVMAPISetLastError("...")`` with messages that can contain braces, and a
    naive counter would end the function in the middle of one.

    Jumps between the interesting tokens with a regex instead of walking every
    character in Python (the char loop cost ~32 s per warm UNet build).
    """
    depth = 0
    i, n = open_idx, len(source)
    search = _BRACE_SCAN.search
    while True:
        m = search(source, i)
        if m is None:
            break
        i = m.start()
        tok = m.group()
        if tok in ('"', "'"):
            i += 1
            while i < n and source[i] != tok:
                i += 2 if source[i] == "\\" else 1
            i += 1
            if i > n:
                break
            continue
        if tok == "//":
            i = source.find("\n", i)
            if i < 0:
                return n
            continue
        if tok == "/*":
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if tok == "{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"unbalanced braces from offset {open_idx}")


def iter_c_functions(source: str):
    """Yield ``(name, start, end)`` for each function definition in *source*.

    ``start`` points at the ``#ifdef __cplusplus`` guard rather than the
    signature, so ``source[start:end]`` is a self-contained, correctly-guarded
    definition.
    """
    for m in _DEF_RE.finditer(source):
        open_idx = source.index("{", m.start(1))
        yield m.group(2), m.start(), _match_brace(source, open_idx)


@functools.lru_cache(maxsize=4)
def function_spans(source: str) -> dict:
    """``{name: (start, end)}`` of :func:`iter_c_functions`, first definition wins.

    Cached per source text: ``network_md.kernel_param_usage`` looks up one
    symbol per graph node, and rescanning the 1.5 MB module from the top each
    time was quadratic. Indexing stops at the first unbalanced function, which
    is where a per-symbol scan would also have given up.
    """
    spans = {}
    try:
        for name, start, end in iter_c_functions(source):
            spans.setdefault(name, (start, end))
    except ValueError:
        pass
    return spans


def split_module(source: str) -> tuple:
    """Return ``(prologue, declarations, interstitial, definitions)``.

    *prologue* is everything before the first guard block (the ``#include``s),
    *declarations* the forward-declaration run, *definitions* the list of
    ``(name, start, end)`` from :func:`iter_c_functions`.

    *interstitial* is the file-scope text that sits BETWEEN function
    definitions, concatenated in source order. TVM emits none, but a BYOC
    codegen appends its own module after TVM's kernels, so its prelude --
    ``byoc/aie_codegen.py``'s declaration of ``conv2d_stem_prepadded`` -- lands
    mid-file, past the first ``extern "C"`` guard. Dropping it is what made a
    split layer compile with an *implicit declaration* of that function: still
    linkable on aarch64, since the pointer arguments pass in registers either
    way, but unprototyped and a hard error under ``-Werror``.

    It is hoisted into ``layers_common.h``, which every layer includes, so this
    carries **declarations**, not definitions. That is the standing contract --
    a BYOC codegen must keep its weights as *function-scope* statics (see the
    note in ``aie_codegen._FUNC_TMPL``). File-scope data here would be
    duplicated into all 65 translation units and collide at link, loudly.
    """
    if _EXTERN_C not in source:
        raise ValueError('no \'extern "C"\' guard found -- not a TVM C module?')
    defs = list(iter_c_functions(source))
    if not defs:
        raise ValueError("no TVM_DLL function definitions found")
    first_guard = source.index(_EXTERN_C)
    prologue = source[:first_guard].rstrip() + "\n"
    declarations = source[first_guard:defs[0][1]].rstrip() + "\n"

    gaps, prev_end = [], defs[0][1]
    for _, start, end in defs:
        chunk = source[prev_end:start].strip()
        if chunk:
            gaps.append(chunk)
        prev_end = end
    tail = source[prev_end:].strip()
    if tail:
        gaps.append(tail)
    interstitial = ("\n\n".join(gaps) + "\n") if gaps else ""

    # Declare anything DEFINED here that TVM's own declaration run did not
    # already cover. TVM pre-declares its packed-ABI kernels, but a BYOC
    # codegen's helper is only ever defined -- in the single file that was
    # fine, because the definition preceded its caller. Split apart, the
    # caller is a different translation unit (the AIE shim is layer 1, the
    # wrapper it calls is layer 64), so without this the shim compiles with
    # an implicit declaration.
    synthesized = []
    for name, start, end in defs:
        if re.search(rf"\b{re.escape(name)}\b\s*\(", declarations):
            continue
        brace = source.index("{", start, end)
        synthesized.append(source[start:brace].strip() + ";")
    if synthesized:
        interstitial += "\n" + "\n".join(synthesized) + "\n"

    return prologue, declarations, interstitial, defs


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #


def layer_ops(func_name: str) -> list:
    """The fused op sequence behind a generated symbol, in order.

    ``tvmgen_default_fused_nn_conv2d_expand_dims_add_add_nn_relu_1`` ->
    ``['conv2d', 'expand_dims', 'add', 'add', 'relu']``.
    """
    name = func_name[len(_PREFIX):] if func_name.startswith(_PREFIX) else func_name
    name = re.sub(r"_\d+$", "", name)              # trailing _1 / _2 disambiguator
    # Glue the multi-word op names back together so a plain split works.
    multiword = ("global_avg_pool2d", "adaptive_avg_pool2d", "batch_flatten",
                 "batch_norm", "max_pool2d", "avg_pool2d", "expand_dims",
                 "bias_add", "batch_matmul", "layer_norm", "log_softmax",
                 "leaky_relu", "contrib_dense_pack")
    holes = {}
    for i, op in enumerate(multiword):
        token = f"OP{i}X"
        holes[token] = op
        name = re.sub(rf"(^|_)(nn_)?{op}(_|$)", rf"\1{token}\3", name)
    ops = [holes.get(p, p) for p in name.split("_") if p and p != "nn"]
    return ops


def short_layer_name(func_name: str) -> str:
    """Readable file stem for a generated symbol: ``conv2d_add_relu``.

    Keeps the whole fused sequence in order -- so a residual block reads
    ``conv2d_add_add_relu``: conv, bias, residual, activation -- minus the
    ``expand_dims`` bias artifact. Two kernels can share a stem (the zoo
    default has eight ``conv2d_add_relu``s); the execution-order prefix disambiguates them.
    """
    ops = [op for op in layer_ops(func_name) if op not in _NOISE_OPS]
    stem = "_".join(ops)
    return re.sub(r"[^0-9A-Za-z_]", "_", stem) or "layer"


def _graph_schedule(graph_path) -> tuple:
    """``(order, call_counts)`` from a graph JSON, or ``([], {})`` without one.

    *order* is each kernel symbol at its first invocation; *call_counts* is how
    many graph nodes invoke it (the zoo default reuses two kernels, so 24 nodes map onto
    22 functions).
    """
    if not graph_path or not Path(graph_path).is_file():
        return [], {}
    graph = json.loads(Path(graph_path).read_text())
    order, counts = [], {}
    for node in graph.get("nodes", []):
        if node.get("op") == "null":
            continue
        fn = (node.get("attrs") or {}).get("func_name")
        if not fn:
            continue
        if fn not in counts:
            order.append(fn)
        counts[fn] = counts.get(fn, 0) + 1
    return order, counts


# --------------------------------------------------------------------------- #
# emitting
# --------------------------------------------------------------------------- #


_HEADER_TMPL = """\
/* Generated by src/frontend/tvmrelay/split_layers.py -- do not edit.
 *
 * Shared prologue for the per-layer files split out of {src}.
 * Every layer file includes this and is otherwise self-contained.
 */
#ifndef TVMRELAY_LAYERS_COMMON_H
#define TVMRELAY_LAYERS_COMMON_H

{prologue}
/* Forward declarations for every kernel in the module, so any layer may call
 * any other (and so each file compiles without seeing its siblings). */
{declarations}
{interstitial}
#endif  /* TVMRELAY_LAYERS_COMMON_H */
"""

#: Heading for the hoisted mid-file material. Only emitted when there is any,
#: so a plain TVM module's header is byte-identical to before.
_INTERSTITIAL_HDR = """\
/* File-scope material that sat BETWEEN function definitions in the original
 * module -- a BYOC codegen's prelude, appended after TVM's own kernels. Hoisted
 * here because a layer that uses it would otherwise compile with an implicit
 * declaration. Declarations only: anything defining storage would be duplicated
 * into every layer and collide at link. */
"""

_LAYER_TMPL = """\
/* Generated by src/frontend/tvmrelay/split_layers.py -- do not edit.
 *
 * Layer {index} of {total}{order_note}
 * ops     : {ops}
 * symbol  : {symbol}
 * source  : {src}
 * invoked : {calls}
 */
#include "{header}"

{body}
"""

_MAKEFILE_TMPL = """\
# Generated by src/frontend/tvmrelay/split_layers.py -- do not edit.
#
#   make            compile every layer to a .o
#   make TVM_HOME=/path/to/tvm
TVM_HOME ?= {tvm_home}
# No -Wall: this is generated code and it warns freely (unused arg_type_ids
# locals in every kernel). Warnings here are TVM's, not yours.
CFLAGS   ?= -O2 -fPIC -I$(TVM_HOME)/include \\
            -I$(TVM_HOME)/3rdparty/dlpack/include

SRCS := $(sort $(wildcard {srcs}))
OBJS := $(SRCS:.c=.o)

all: $(OBJS)

# Every layer includes layers_common.h, so rebuild all of them when it moves.
%.o: %.c layers_common.h
\t$(CC) $(CFLAGS) -c $< -o $@

# One layer at a time:  make 00_conv2d_add_relu
{kind_targets}
clean:
\trm -f $(OBJS)

.PHONY: all clean{kind_phony}
"""


def _write_readme(out_dir: Path, src_name: str, entries: list, group: bool) -> None:
    lines = [
        f"# `layers/` — {src_name} split one file per layer",
        "",
        "Generated by `src/frontend/tvmrelay/split_layers.py`. Do not edit by",
        "hand; re-run the splitter instead.",
        "",
        "Each file is a complete translation unit — it includes",
        "`layers_common.h` and compiles alone. `make` builds them all;",
        "`make <kind>` builds one folder.",
        "",
    ]
    if group:
        lines += [
            "**One folder per layer, in graph execution order** — reading",
            "`layers/` top to bottom is reading the network in the order it",
            "runs. Each folder is that layer's working directory: it holds a",
            "single `.c` today, and is where the AIE artifacts for the layer",
            "(`.bcf`, kernel sources, generated `host.cc`/`routing.cc`) belong",
            "next to it.",
            "",
        ]
    else:
        lines += ["Flat layout (`--flat`), numbered in graph execution order.",
                  ""]
    lines += [
        "There is no standalone `relu` layer: TVM's `FuseOps` welds elementwise",
        "ops into their producer, so the compiled unit really is",
        "conv2d+bias+residual+relu — one loop nest, one function. For true",
        "per-op layers build with `deploy_flow.py --no-fuse`.",
        "",
        "| # | layer | ops | TVM symbol | calls |",
        "|---|-------|-----|------------|-------|",
    ]
    for e in entries:
        lines.append(
            "| {index} | `{path}` | {ops} | `{symbol}` | {calls} |".format(
                index=e["index"], path=e["path"], ops=", ".join(e["ops"]),
                symbol=e["symbol"], calls=e["calls"],
            )
        )
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines))


def _tvm_include_dirs():
    """``(tvm_home, [include dirs])`` for the importable TVM, or ``(None, [])``."""
    try:
        import tvm  # noqa: F401
    except Exception:
        return None, []
    home = Path(tvm.__file__).resolve().parents[2]
    dirs = [home / "include", home / "3rdparty" / "dlpack" / "include"]
    return home, [d for d in dirs if d.is_dir()]


def _syntax_check(files: list, include_dirs: list) -> list:
    """``gcc -fsyntax-only`` each file; return a list of failure strings."""
    failures = []
    flags = [a for d in include_dirs for a in ("-I", str(d))]
    for path in files:
        proc = subprocess.run(
            ["gcc", "-fsyntax-only", *flags, str(path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            head = (proc.stderr.strip().splitlines() or ["(no output)"])[0]
            failures.append(f"{path.name}: {head}")
    return failures


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def _clear_stale(out_dir: Path) -> None:
    """Drop previously-generated layer files so a shrinking model leaves none.

    Only ``.c``/``.o`` under *out_dir* and its immediate subdirectories, then
    any subdirectory that this leaves empty -- so a folder someone added by
    hand, or one holding an AIE kernel next to the generated C, survives.
    """
    for stale in list(out_dir.glob("*.c")) + list(out_dir.glob("*/*.c")) \
            + list(out_dir.glob("*.o")) + list(out_dir.glob("*/*.o")):
        stale.unlink()
    for sub in out_dir.iterdir():
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()


def split_layers(c_path, graph_path=None, out_dir=None, verify=True,
                 group=True, verbose=True) -> dict:
    """Split *c_path* into one folder per kernel under *out_dir*.

    With *group* (the default) each layer gets its own folder, named for its
    execution step -- ``layers/02_conv2d_add_relu/02_conv2d_add_relu.c`` -- so
    the directory listing reads in execution order and each folder can hold
    that layer's AIE artifacts (``.bcf``, kernel sources, ``host.cc``) beside
    the C. ``group=False`` writes one flat directory of ``.c`` files.

    *graph_path* (default ``<stem>_graph.json`` beside the source) supplies the
    execution order; without it the files keep source order. *out_dir* defaults
    to ``layers/`` beside the source, and is cleared of previously-generated
    files first (see :func:`_clear_stale`).

    Returns the manifest dict, which is also written to
    ``<out_dir>/manifest.json``.
    """
    c_path = Path(c_path)
    source = c_path.read_text()
    prologue, declarations, interstitial, defs = split_module(source)

    if graph_path is None:
        cand = c_path.with_name(c_path.stem + "_graph.json")
        graph_path = cand if cand.is_file() else None
    order, counts = _graph_schedule(graph_path)

    # Execution order first, then anything the graph never calls (kept, not
    # dropped -- an uncalled kernel is worth seeing, not hiding).
    by_name = {name: (start, end) for name, start, end in defs}
    ordered = [n for n in order if n in by_name]
    ordered += [name for name, _, _ in defs if name not in set(ordered)]

    out_dir = Path(out_dir) if out_dir else c_path.with_name("layers")
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale(out_dir)

    (out_dir / "layers_common.h").write_text(_HEADER_TMPL.format(
        src=c_path.name, prologue=prologue, declarations=declarations,
        interstitial=(_INTERSTITIAL_HDR + interstitial) if interstitial else ""))

    width = max(2, len(str(len(ordered) - 1)))
    entries, written, kinds = [], [], []
    for i, name in enumerate(ordered):
        start, end = by_name[name]
        kind = short_layer_name(name)
        stem = f"{i:0{width}d}_{kind}"
        calls = counts.get(name, 0)
        if not counts:
            calls_note = "unknown (no graph JSON)"
        elif calls == 0:
            # Emitted but unreachable -- e.g. batch_flatten, which the graph
            # elides into a __nop because the reshape is layout-free.
            calls_note = "never -- not reached by the graph (dead kernel)"
        else:
            calls_note = f"{calls} graph node(s)"

        if group:
            # One folder per layer, named for its execution step. This is the
            # per-layer working directory the AIE artifacts (.bcf, kernel
            # sources, host.cc/routing.cc) will land in next to the C.
            (out_dir / stem).mkdir(exist_ok=True)
            path = out_dir / stem / f"{stem}.c"
            # layers_common.h stays at the root: one copy, one source of truth.
            header = "../layers_common.h"
            kinds.append(stem)
        else:
            path = out_dir / f"{stem}.c"
            header = "layers_common.h"

        path.write_text(_LAYER_TMPL.format(
            index=i, total=len(ordered),
            order_note="" if order else "  (source order -- no graph JSON)",
            ops=", ".join(layer_ops(name)), symbol=name, src=c_path.name,
            calls=calls_note, header=header, body=source[start:end].strip(),
        ))
        written.append(path)
        entries.append({"index": i, "file": path.name,
                        "dir": stem if group else "", "kind": kind,
                        "path": str(path.relative_to(out_dir)), "symbol": name,
                        "ops": layer_ops(name), "calls": calls,
                        "lines": source[start:end].count("\n") + 1})

    tvm_home, include_dirs = _tvm_include_dirs()
    (out_dir / "Makefile").write_text(_MAKEFILE_TMPL.format(
        tvm_home=tvm_home or "/path/to/tvm",
        srcs="*/*.c" if group else "*.c",
        kind_targets="".join(
            f"{k}: $(patsubst %.c,%.o,$(wildcard {k}/*.c))\n" for k in kinds),
        kind_phony=("".join(f" {k}" for k in kinds)) if kinds else "",
    ))
    _write_readme(out_dir, c_path.name, entries, group)

    manifest = {
        "source": str(c_path),
        "graph": str(graph_path) if graph_path else None,
        "layout": "grouped" if group else "flat",
        "layer_count": len(entries),
        "kind_count": len(kinds) if group else None,
        "kernel_invocations": sum(counts.values()) if counts else None,
        "layers": entries,
    }

    if verify and include_dirs:
        failures = _syntax_check(written, include_dirs)
        manifest["syntax_check"] = "pass" if not failures else failures
        if verbose:
            if failures:
                print(f"  [split] syntax check FAILED on {len(failures)} file(s):")
                for f in failures[:5]:
                    print(f"      {f}")
            else:
                print(f"  [split] syntax check: {len(written)} file(s) compile clean")
    elif verify and verbose:
        manifest["syntax_check"] = "skipped (TVM headers not found)"
        print("  [split] syntax check skipped -- TVM headers not found")

    # Directories this run did NOT produce. `_clear_stale` deliberately keeps
    # a non-empty folder -- that is how an AIE-offloaded layer's `aie/`
    # artifacts survive a re-split -- but the side effect is that a folder
    # from an older, differently-shaped graph lingers forever once anything
    # else drops a file into it. Two builds then leave two `01_*` folders side
    # by side, and `ls layers/` silently disagrees with the manifest. Deleting
    # them is not safe (that is exactly the AIE case); naming them is.
    live = {e["dir"] for e in entries if e.get("dir")} if group else set()
    manifest["stale_dirs"] = sorted(
        p.name for p in out_dir.iterdir()
        if p.is_dir() and p.name not in live) if group else []

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if manifest["stale_dirs"] and verbose:
        n = len(manifest["stale_dirs"])
        print(f"  [split] {n} directory(ies) left over from an earlier build, "
              f"NOT part of this graph (manifest.stale_dirs):")
        for name in manifest["stale_dirs"][:3]:
            print(f"      {name}/")
        if n > 3:
            print(f"      ... and {n - 3} more")

    if verbose:
        if group:
            print(f"  [split] {len(entries)} layer folders -> {out_dir}")
            for e in entries[:3]:
                print(f"      {e['dir']}/")
            if len(entries) > 3:
                print(f"      ... and {len(entries) - 3} more, in execution order")
        else:
            print(f"  [split] {len(entries)} layers -> {out_dir}")
            for e in entries[:3]:
                print(f"      {e['path']}  ({e['lines']} lines)")
            if len(entries) > 3:
                print(f"      ... and {len(entries) - 3} more")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Split a TVM-generated C module into one file per layer.")
    ap.add_argument("target", nargs="?", default="worklocal/tvmrelay_deploy",
                    help="the generated .c file, or a directory holding one "
                         "(default: worklocal/tvmrelay_deploy)")
    ap.add_argument("--graph", help="graph JSON for execution order "
                                    "(default: <stem>_graph.json beside it)")
    ap.add_argument("--out-dir", help="output directory (default: layers/ "
                                      "beside the source)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the per-file gcc -fsyntax-only check")
    ap.add_argument("--flat", action="store_true",
                    help="one flat directory of .c files instead of a folder "
                         "per layer")
    args = ap.parse_args(argv)

    target = Path(args.target)
    if target.is_dir():
        candidates = sorted(p for p in target.glob("*.c") if p.is_file())
        if not candidates:
            print(f"error: no .c file in {target}", file=sys.stderr)
            return 2
        if len(candidates) > 1:
            print(f"error: {len(candidates)} .c files in {target}; name one",
                  file=sys.stderr)
            return 2
        target = candidates[0]
    if not target.is_file():
        print(f"error: {target} not found", file=sys.stderr)
        return 2

    try:
        split_layers(target, graph_path=args.graph, out_dir=args.out_dir,
                     verify=not args.no_verify, group=not args.flat)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
