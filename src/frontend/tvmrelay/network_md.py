#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Describe the compiled graph as markdown: shapes, dtypes, and who feeds whom.

``<stem>_graph.json`` holds everything about the network that survives
compilation -- every tensor's shape and dtype, every kernel's arguments, and
the storage plan TVM's liveness analysis produced. It is also 400 KB of
flat JSON with the connectivity expressed as ``node_row_ptr`` arithmetic, so
reading it by eye is not practical.

This module turns it into ``network.md`` next to the generated C:

    Summary        counts, the input and output tensor, total buffer bytes
    Dataflow       a Mermaid flowchart -- the actual topology, skips included
    Layers         one row per kernel call: kind, inputs, output shape/dtype
    Layer detail   every argument of every call, with its role and storage id
    Buffer reuse   which tensors share a storage_id, and what that saves

Three things it is careful about, because each is a real source of confusion:

**Entry indices are not node indices.** A node's k-th output is tensor
``node_row_ptr[node] + k``, and the ``shape``/``dltype``/``storage_id`` arrays
are indexed by *that*. The tables print the entry index, because that is what
``graph_driver.c``'s ``tensors[]`` is indexed by -- so a row here lines up with
a `GRAPH_TRACE` dump line without any translation.

**Parameters are not activations.** 137 of this graph's 143 buffers are
weights, and drawing them would bury the topology under a fan of constant
edges. The flowchart carries activation edges only; the per-layer tables mark
each argument ``in`` / ``param`` / ``out`` so the weights are still accounted
for.

**Storage ids alias.** TVM reuses one buffer for several tensors whose live
ranges do not overlap, so "143 tensors" and "143 buffers" are different claims
and the bytes do not add up unless you deduplicate. The buffer-reuse section
reports what the aliasing actually saves.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

__all__ = ["build_network", "write_network_md", "kernel_param_usage",
           "param_role"]

# ─────────────────────────────────────────────────────────────────────────────
# What each parameter IS -- read out of the generated C, not guessed
#
# TVM names constants `p0, p1, ...` by *argument slot*, which says nothing
# about whether a tensor is a weight, a bias or a requantization shift. The
# roles below come from how the kernel actually uses each one, so they cannot
# drift from the code:
#
#     conv:  ... * ((int32_t)((int16_t*)p1_1)[...])      -> multiplied  -> weight
#     epilogue:
#       v_ = (int32_t)(((((int64_t)(acc + p2_1[c] - p3_1[c]))   -> add  -> bias
#                                                               -> sub  -> zero-point
#              * ((int64_t)p4_1[c]))                            -> mul  -> multiplier
#              + ((int64_t)1 << (p6_1[c] + 31 - 1)))
#             >> ((int64_t)(p6_1[c] + 31)));                    -> shift
#
# A slot the body never references is reported `unused` rather than given a
# plausible name -- `p5` above really is dead, a left-shift operand that
# constant-folded to zero.
# ─────────────────────────────────────────────────────────────────────────────

#: `((int32_t*)p4_1)[` -- a parameter dereference, capturing the slot number.
_PARAM_REF = re.compile(r"\(\((?:u?int\d+_t|float|double)\s*\*\)\s*p(\d+)_1\)\s*\[")

#: A cast sitting immediately before a reference, e.g. the `(int64_t)` in
#: `* ((int64_t)((int32_t*)p4_1)[...])`. Skipped when looking for the operator,
#: or every operand reads as `)` and nothing can be classified.
_TRAILING_CAST = re.compile(r"\(\s*(?:u?int\d+_t|float|double)\s*\)\s*$")

#: usage -> the name shown in the tables.
_ROLE_NAMES = {"mul": "weight", "add": "bias", "sub": "zero-point",
               "shift": "requant shift", "copy": "copied"}


def _op_before(text: str, pos: int) -> tuple:
    """``(operator, innermost_cast)`` governing the reference at *pos*.

    The cast is what separates the two multiplies, and it is the only
    *reliable* separator. Rank does not work: a conv weight is packed
    ``[16,1,7,7,3,4]`` (rank 6) but the dense weight is ``[125,512,8]``
    (rank 3), which a "rank >= 4 means weight" rule silently calls a
    requantization multiplier. The emitted C is unambiguous --

        conv MAC    ... * ((int32_t)((int16_t*)p1_1)[...])
        requantize  ... * ((int64_t)((int32_t*)p4_1)[...])

    -- because the accumulator multiply must widen to int64 and the MAC
    must not.
    """
    pre, cast = text[:pos], None
    while True:
        stripped = pre.rstrip(" \t\n")
        if stripped.endswith("("):
            pre = stripped[:-1]
            continue
        m = _TRAILING_CAST.search(stripped)
        if m:
            if cast is None:
                cast = stripped[m.start():].strip().strip("()").strip()
            pre = stripped[:m.start()]
            continue
        pre = stripped
        break
    if not pre:
        return "?", cast
    if pre.endswith(">>") or pre.endswith("<<"):
        return "shift", cast
    return pre[-1], cast


def kernel_param_usage(source: str, symbol: str = None) -> dict:
    """``{slot: {usage, ...}}`` for one kernel, from its generated C.

    *source* may be a whole module or a single per-layer file; *symbol* picks
    the function out of a module. Slots are the kernel's own ``pN`` numbering
    (argument position), **not** the graph's global constant names -- layer 29
    takes ``p131..p136`` from the graph and calls them ``p1..p6`` internally,
    so keying on the name instead of the slot mislabels every later layer.
    """
    body = source
    if symbol and symbol in source:
        try:
            from frontend.tvmrelay.split_layers import function_spans

            span = function_spans(source).get(symbol)
            if span:
                body = source[span[0]:span[1]]
        except Exception:
            pass

    usage: dict = {}
    for m in _PARAM_REF.finditer(body):
        slot = int(m.group(1))
        op, cast = _op_before(body, m.start())
        kind = {"+": "add", "-": "sub", "*": "mul", "shift": "shift",
                # `out[...] = p1_1[...]` -- the kernel only re-lays-out the
                # constant (a repeat/layout_transform of the per-channel
                # requant values). Real use, but no arithmetic role.
                "=": "copy"}.get(op, "other")
        if kind == "mul" and cast == "int64_t":
            kind = "mul64"        # widened: the requantization multiply
        usage.setdefault(slot, set()).add(kind)
    return usage


def param_role(usage: set, shape=None) -> str:
    """Turn one slot's usage into a role name.

    Order matters: a slot used in several ways is named for the most specific
    use. ``mul`` is checked before ``mul64`` only in the sense that the two are
    disjoint -- the int32 MAC multiply is the weight, the widened int64
    multiply is the requantization multiplier (see :func:`_op_before`).
    """
    if not usage:
        return "unused"
    if "mul" in usage:
        return "weight"
    if "mul64" in usage:
        return "requant multiplier"
    for key in ("shift", "add", "sub", "copy"):
        if key in usage:
            return _ROLE_NAMES[key]
    return "param"

#: Edge labels carry the dtype and shape; past this many characters they are
#: truncated so Mermaid lays the graph out vertically instead of sprawling.
_EDGE_LABEL_MAX = 34

#: Mermaid gets unreadable long before this, and a graph nobody can read is
#: worse than a link to the tables. Past it, the section says so and stops.
_MERMAID_MAX_NODES = 120


def _bytes_of(dltype: str) -> int:
    """Bytes per element for a graph ``dltype`` string (``float32`` -> 4)."""
    for suffix, size in (("64", 8), ("32", 4), ("16", 2), ("8", 1)):
        if dltype.endswith(suffix):
            return size
    return 0                     # unknown width: counted as 0 rather than guessed


def _fmt_shape(shape) -> str:
    return "[" + ",".join(str(d) for d in shape) + "]"


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def build_network(graph_path, params_path=None, manifest_path=None) -> dict:
    """Parse the graph JSON into the structure the markdown is rendered from.

    Returns ``{"calls", "tensors", "input", "output", "storage", ...}`` where
    ``calls`` is one entry per ``tvm_op`` node in execution order, each with
    its inputs split into activations and parameters.

    Kept separate from the rendering so the same walk can be reused -- it is
    the only place entry-index arithmetic happens.
    """
    graph = json.loads(Path(graph_path).read_text())
    nodes = graph["nodes"]
    row_ptr = graph["node_row_ptr"]
    shapes = graph["attrs"]["shape"][1]
    dltypes = graph["attrs"]["dltype"][1]
    sids = graph["attrs"]["storage_id"][1]

    # The blob is what separates a constant from an activation. Without it
    # EVERY parameter silently reclassifies as an activation -- the document
    # then reports `parameters | 0`, no `param` rows at all, and reads exactly
    # like a model that has no weights. So the failure is recorded and
    # reported rather than swallowed.
    param_names, params_status = set(), "ok"
    if not params_path or not Path(params_path).is_file():
        params_status = f"missing {Path(params_path).name if params_path else '<none>'}"
    else:
        try:
            from frontend.tvmrelay.arm_build import read_param_names

            param_names = set(read_param_names(Path(params_path)))
        except Exception as exc:
            params_status = f"unreadable ({type(exc).__name__}: {exc})"

    # Keyed by SYMBOL, not by index. Stage 5 writes one folder per *distinct
    # kernel* (28 here) while the graph makes 30 calls, because two fused
    # groups are invoked twice -- so layer index and call index are different
    # numbers and matching them up positionally silently mislabels every row
    # after the first repeat.
    layer_dirs = {}
    layers_root = Path(manifest_path).parent if manifest_path else None
    if manifest_path and Path(manifest_path).is_file():
        try:
            man = json.loads(Path(manifest_path).read_text())
            for entry in man.get("layers", []):
                sym = entry.get("symbol")
                if sym:
                    layer_dirs[sym] = entry.get("dir") or entry.get("file")
        except Exception:
            layer_dirs = {}

    # The monolithic module, read once: the role fallback when stage 5 was
    # skipped. `<stem>_graph.json` sits next to `<stem>.c`.
    mono_source = ""
    mono_path = Path(graph_path)
    mono_path = mono_path.with_name(
        mono_path.name.replace("_graph.json", ".c"))
    if mono_path.is_file():
        try:
            mono_source = mono_path.read_text()
        except OSError:
            mono_source = ""

    def tensor(eid: int) -> dict:
        return {"eid": eid, "shape": shapes[eid], "dtype": dltypes[eid],
                "sid": sids[eid],
                "bytes": _numel(shapes[eid]) * _bytes_of(dltypes[eid])}

    # Which node produced each entry, so an input can be traced to its layer.
    producer = {}
    for idx, node in enumerate(nodes):
        if node.get("op") == "tvm_op":
            n_out = int(node["attrs"].get("num_outputs", 1))
            for k in range(n_out):
                producer[row_ptr[idx] + k] = idx

    calls, call_of_node = [], {}
    for idx, node in enumerate(nodes):
        if node.get("op") != "tvm_op":
            continue
        fn = node["attrs"]["func_name"]
        n_out = int(node["attrs"].get("num_outputs", 1))
        acts, params = [], []
        for arg, (src, slot, _) in enumerate(node["inputs"]):
            eid = row_ptr[src] + slot
            rec = tensor(eid)
            rec["from"] = producer.get(eid)
            # The kernel's own `pN` numbering, which is what the C references.
            rec["slot"] = arg
            if nodes[src].get("op") == "null":
                rec["name"] = nodes[src]["name"]
                (params if rec["name"] in param_names else acts).append(rec)
            else:
                acts.append(rec)
        call = {"call": len(calls), "node": idx, "func": fn,
                "elided": fn == "__nop",
                "attrs": {k: v for k, v in node["attrs"].items()
                          if k in ("src_layout", "dst_layout", "hash")},
                "inputs": acts, "params": params,
                "outputs": [tensor(row_ptr[idx] + k) for k in range(n_out)],
                "dir": layer_dirs.get(fn)}
        # Roles from the kernel's own C: the per-layer split first, then the
        # monolithic module (--no-split leaves only that).
        usage = {}
        src_c = layers_root / call["dir"] if (layers_root and call["dir"]) else None
        if src_c is not None and src_c.is_dir():
            for cand in sorted(src_c.glob("*.c")):
                usage = kernel_param_usage(cand.read_text(), fn)
                if usage:
                    break
        if not usage and mono_source:
            usage = kernel_param_usage(mono_source, fn)
        # With no C to read, a role is genuinely unknown -- and `unused` is
        # the one answer that must not be guessed, since it asserts the kernel
        # never touches the tensor. Say `param` instead.
        for rec in params:
            rec["role"] = (param_role(usage.get(rec["slot"]), rec["shape"])
                           if usage else "param")
        call["roles_known"] = bool(usage)
        call_of_node[idx] = len(calls)
        calls.append(call)

    # The graph input: an arg node with no entry in the params blob.
    in_idx = next((i for i, n in enumerate(nodes)
                   if n.get("op") == "null" and n["name"] not in param_names),
                  None)
    out_node, out_slot, _ = graph["heads"][0]

    storage: dict = {}
    for eid, sid in enumerate(sids):
        storage.setdefault(sid, []).append(eid)

    from frontend.tvmrelay.arm_build import graph_sha

    return {
        "graph_sha": graph_sha(graph_path),
        "params_status": params_status,
        "calls": calls, "call_of_node": call_of_node, "producer": producer,
        "n_tensors": len(sids), "n_params": len(param_names),
        "shapes": shapes, "dltypes": dltypes, "sids": sids,
        "input": (dict(tensor(row_ptr[in_idx]), name=nodes[in_idx]["name"])
                  if in_idx is not None else None),
        "output": tensor(row_ptr[out_node] + out_slot),
        "storage": storage,
        "param_names": param_names,
    }


def _short(fn: str) -> str:
    """Kernel name without the ``tvmgen_default_fused_`` boilerplate."""
    return fn.replace("tvmgen_default_fused_", "").rstrip("_") or fn


#: Ordered most-specific-first: the first hit is the op that gives a fused
#: group its character. ``contrib_conv2d_NCHWc`` must precede ``conv2d``, and
#: ``global_avg_pool2d`` must precede ``avg_pool2d``, or the shorter name wins
#: and every conv in the network gets labelled the same way.
_PRIMARY_OPS = (
    ("contrib_dense_pack", "dense"), ("nn_dense", "dense"), ("dense", "dense"),
    ("contrib_conv2d_NCHWc", "conv2d"), ("conv2d_NCHWc", "conv2d"),
    ("conv2d", "conv2d"),
    ("max_pool2d", "max_pool"), ("global_avg_pool2d", "global_avg_pool"),
    ("avg_pool2d", "avg_pool"), ("batch_flatten", "flatten"),
    # Before layout_transform: layer 0 is `divide_round_add_clip_cast_
    # subtract_layout_transform`, and calling that "layout" buries the fact
    # that it is the input quantizer.
    ("divide", "quantize"),
    ("layout_transform", "layout"), ("fixed_point_multiply", "requantize"),
    ("subtract", "subtract"), ("add", "add"),
    ("clip", "clip"), ("cast", "cast"), ("pad", "pad"), ("sum", "sum"),
)

#: Counted to show how much got fused into one kernel. Not the same list as
#: above: these are the ops that may appear anywhere in a fused name.
_OP_WORDS = ("conv2d", "dense", "pool2d", "flatten", "layout_transform",
             "fixed_point_multiply", "divide", "round", "add", "subtract",
             "clip", "cast", "pad", "sum", "multiply", "relu", "requantize")


def _primary_op(fn: str) -> tuple:
    """``(label, n_fused)`` -- what this kernel mainly does, and how fused.

    The raw names run to 80 characters of underscore-joined op list plus a
    16-hex-digit hash, which truncate to an identical prefix for all 20 convs.
    A diagram node needs the op, not the provenance.
    """
    name = _short(fn)
    for needle, label in _PRIMARY_OPS:
        if needle in name:
            n = sum(1 for w in _OP_WORDS if w in name)
            return label, n
    return (name[:20] or "op"), 0


def _mermaid(net: dict) -> list:
    """The dataflow flowchart: activation edges only, labelled dtype + shape."""
    calls = net["calls"]
    lines = ["## Dataflow", ""]
    if len(calls) > _MERMAID_MAX_NODES:
        lines += [f"{len(calls)} kernel calls -- too many to draw legibly. "
                  f"See [Layers](#layers) for the same connectivity as a "
                  f"table.", ""]
        return lines
    lines += [
        "Activation edges only -- the "
        f"{net['n_params']} parameter tensors are left out, or the topology "
        "would disappear under a fan of constant edges. Each edge carries the "
        "dtype and shape actually passed; **dotted edges are skips** (the "
        "producer is not the preceding call), which is where a residual network's skip "
        "structure shows up. Node labels give the primary op and `+N` for how "
        "many further ops were fused into that kernel -- the full name is in "
        "[Layers](#layers).",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    if net["input"]:
        t = net["input"]
        lines.append(f'  IN(["{t["name"]}<br/>{t["dtype"]} '
                     f'{_fmt_shape(t["shape"])}"])')
    for c in calls:
        op, nfused = _primary_op(c["func"])
        out = c["outputs"][0] if c["outputs"] else None
        fused = f" +{nfused - 1}" if nfused > 1 else ""
        second = (f'<br/>{out["dtype"]} {_fmt_shape(out["shape"])}'
                  if out else "")
        lines.append(f'  L{c["call"]:02d}["<b>{c["call"]:02d}</b> {op}{fused}'
                     f'{second}"]')
    lines.append(f'  OUT(["output<br/>{net["output"]["dtype"]} '
                 f'{_fmt_shape(net["output"]["shape"])}"])')
    lines.append("")

    for c in calls:
        for rec in c["inputs"]:
            label = f'{rec["dtype"]} {_fmt_shape(rec["shape"])}'
            if len(label) > _EDGE_LABEL_MAX:
                label = rec["dtype"]
            src = rec.get("from")
            if src is None:
                lines.append(f'  IN -- "{label}" --> L{c["call"]:02d}')
                continue
            src_call = net["call_of_node"][src]
            # Dotted when the producer is not the immediately preceding call:
            # that is exactly a residual skip, and it is the structure worth
            # seeing at a glance.
            arrow = "-->" if src_call == c["call"] - 1 else "-.->"
            lines.append(f'  L{src_call:02d} -- "{label}" {arrow} '
                         f'L{c["call"]:02d}')
    last = net["producer"].get(net["output"]["eid"])
    if last is not None:
        lines.append(f'  L{net["call_of_node"][last]:02d} --> OUT')
    lines += ["```", ""]
    return lines


def _params_brief(params: list) -> str:
    """``int16[16,1,7,7,3,4], int32[1,16,1,1,4]×2`` -- one layer's constants.

    Runs of the same ``(dtype, shape)`` are collapsed with ``×N`` because the
    requantization constants come in identical groups -- a conv's six params
    are one weight tensor and five same-shaped int32 vectors, and listing them
    separately makes the one that matters harder to find, not easier.
    """
    if not params:
        return "—"
    out, run_key, run_n = [], None, 0

    def flush():
        if run_key:
            role, body = run_key
            out.append(f"**{role}** `{body}`" + (f"×{run_n}" if run_n > 1 else ""))

    for p in params:
        key = (p.get("role", "param"),
               f'{p["dtype"]}{_fmt_shape(p["shape"])}')
        if key == run_key:
            run_n += 1
            continue
        flush()
        run_key, run_n = key, 1
    flush()
    return ", ".join(out)


def _params_section(net: dict) -> list:
    """Where the parameter bytes actually go: by dtype, and the biggest ones."""
    seen, by_dtype, rows = set(), {}, []
    for c in net["calls"]:
        for p in c["params"]:
            if p["eid"] in seen:        # a constant feeding two calls
                continue
            seen.add(p["eid"])
            agg = by_dtype.setdefault(p["dtype"], [0, 0])
            agg[0] += 1
            agg[1] += p["bytes"]
            rows.append((p, c))
    total = sum(b for _, b in by_dtype.values())
    lines = [
        "## Parameters", "",
        f"{len(seen)} constant tensors, **{total:,} bytes** -- this is what "
        f"`weights.bin` holds, and `graph_init()` memcpy's into place. Shapes "
        "and dtypes per layer are in [Layer detail](#layer-detail); this is "
        "where the bytes go.", "",
        "| dtype | tensors | bytes | share |", "|---|---|---|---|",
    ]
    for dt, (n, b) in sorted(by_dtype.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"| `{dt}` | {n} | {b:,} | "
                     f"{(100.0 * b / total) if total else 0:.1f}% |")
    lines += ["", "Largest:", "",
              "| parameter | layer | dtype | shape | bytes |",
              "|---|---|---|---|---|"]
    for p, c in sorted(rows, key=lambda r: -r[0]["bytes"])[:10]:
        op, _ = _primary_op(c["func"])
        lines.append(f'| `{p.get("name", "?")}` | {c["call"]:02d} {op} | '
                     f'`{p["dtype"]}` | `{_fmt_shape(p["shape"])}` | '
                     f'{p["bytes"]:,} |')
    lines.append("")
    return lines


def _layer_table(net: dict) -> list:
    """One row per kernel call: inputs, parameter count, output shape/dtype."""
    lines = ["## Layers", "",
             "In execution order. **Tensor** is the entry index -- the same "
             "index `graph_driver.c` uses for `tensors[]`, so a row here lines "
             "up with a `make TRACE=1` dump line directly.", "",
             "`layer` is matched to the call by **kernel symbol**, not by "
             "position: stage 5 writes one folder per *distinct* kernel, so a "
             "kernel invoked twice gives two rows pointing at the same "
             "folder, and the numbering drifts from the call index after the "
             "first repeat.", "",
             "Activations carry their entry index; **parameters** are listed "
             "as `dtype[shape]` with `×N` collapsing identical consecutive "
             "ones. Their names, entry indices and byte sizes are in "
             "[Layer detail](#layer-detail), and the rollup is in "
             "[Parameters](#parameters).", "",
             "| # | layer | kernel | from | in dtype[shape] | out dtype[shape] "
             "| parameters dtype[shape] |",
             "|---|---|---|---|---|---|---|"]
    for c in net["calls"]:
        src = ", ".join(
            "input" if r.get("from") is None
            else f'{net["call_of_node"][r["from"]]:02d}' for r in c["inputs"]
        ) or "—"
        ins = ", ".join(f'{r["eid"]}&nbsp;`{r["dtype"]}{_fmt_shape(r["shape"])}`'
                        for r in c["inputs"]) or "—"
        outs = c["outputs"][0] if c["outputs"] else None
        folder = f'`{c["dir"]}`' if c["dir"] else "—"
        out_cell = (f'{outs["eid"]}&nbsp;`{outs["dtype"]}'
                    f'{_fmt_shape(outs["shape"])}`' if outs else "—")
        lines.append(
            f'| {c["call"]:02d} | {folder} | `{_short(c["func"])}` | {src} | '
            f'{ins} | {out_cell} | '
            f'{len(c["params"])}: {_params_brief(c["params"])} |')
    lines.append("")
    return lines


def _layer_detail(net: dict) -> list:
    """Every argument of every call, with its role, storage id and size."""
    lines = ["## Layer detail", "",
             "Every argument of every call, **in kernel argument order** -- "
             "the order `graph_driver.c` fills `args[]`, outputs last. "
             "Activations and constants interleave (a conv takes "
             "`args[0]` activation, `args[1]` weight, `args[2]` activation, "
             "then its constants), so the rows are sorted by slot rather than "
             "grouped by kind.", "",
             "**what** is read out of the generated C, not guessed: `weight` "
             "is multiplied against the data, `bias` is added to the "
             "accumulator, `zero-point` is subtracted from it, "
             "`requant multiplier` / `requant shift` are the `(acc * m + "
             "2^(s+30)) >> (s+31)` pair, `copied` means the kernel only "
             "re-lays-out the constant (a repeat into NCHWc) without doing "
             "arithmetic on it, `unused` means the body never reads that "
             "argument at all (TVM still passes it), and a bare `param` means "
             "the kernel C was not available to read. **arg** is the "
             "kernel's own `pN` slot -- the name TVM uses *inside* the "
             "function, which differs from the graph-level **name** in the "
             "next column.", ""]
    for c in net["calls"]:
        title = f'### {c["call"]:02d} — {_short(c["func"])}'
        lines.append(title)
        if c["dir"]:
            lines.append(f'`layers/{c["dir"]}/`')
        layouts = [f'{k} `{v}`' for k, v in c["attrs"].items()
                   if k in ("src_layout", "dst_layout")]
        if layouts:
            lines.append("")
            lines.append("Layout: " + " → ".join(layouts))
        lines += ["", "| kind | what | arg | name | tensor | dtype | shape | "
                  "storage | bytes |",
                  "|---|---|---|---|---|---|---|---|---|"]
        # Sorted by kernel arg slot, NOT grouped activations-then-params.
        # Under int8 legalization the two interleave -- a conv takes
        # args[0]=activation, args[1]=weight, args[2]=activation,
        # args[3..]=constants -- so grouping would print an order that is not
        # the order `graph_driver.c` fills `args[]`, which is the whole point
        # of showing the slot.
        rows = sorted(
            [("in", r) for r in c["inputs"]] + [("param", r) for r in c["params"]],
            key=lambda kr: kr[1].get("slot", 0))
        rows += [("out", r) for r in c["outputs"]]
        for kind, r in rows:
            what = r.get("role", "activation" if kind != "out" else "result")
            slot = r.get("slot")
            arg = f"`p{slot}`" if slot is not None else "—"
            name = f'`{r["name"]}`' if r.get("name") else "—"
            lines.append(f'| {kind} | {what} | {arg} | {name} | {r["eid"]} | '
                         f'`{r["dtype"]}` | `{_fmt_shape(r["shape"])}` | '
                         f'{r["sid"]} | {r["bytes"]:,} |')
        lines.append("")
    return lines


def _buffer_reuse(net: dict) -> list:
    """What TVM's storage aliasing actually saves, and where it is sharpest."""
    shapes, dltypes = net["shapes"], net["dltypes"]
    naive = sum(_numel(s) * _bytes_of(d) for s, d in zip(shapes, dltypes))
    per_sid = {sid: max(_numel(shapes[e]) * _bytes_of(dltypes[e]) for e in eids)
               for sid, eids in net["storage"].items()}
    actual = sum(per_sid.values())
    shared = {sid: eids for sid, eids in net["storage"].items() if len(eids) > 1}
    lines = [
        "## Buffer reuse", "",
        f"TVM's liveness analysis assigns the {net['n_tensors']} tensors to "
        f"**{len(net['storage'])} storage ids**; tensors whose live ranges do "
        f"not overlap share one. So tensor count and buffer count are "
        f"different claims, and the per-tensor bytes above do **not** sum to "
        f"the program's footprint.", "",
        f"| | bytes |",
        f"|---|---|",
        f"| one buffer per tensor | {naive:,} |",
        f"| after aliasing (what `graph_driver.c` allocates) | **{actual:,}** |",
        f"| saved | {naive - actual:,} "
        f"({(100.0 * (naive - actual) / naive) if naive else 0:.1f}%) |",
        "",
    ]
    if shared:
        lines += [f"{len(shared)} storage ids carry more than one tensor:", "",
                  "| storage | tensors | buffer bytes |", "|---|---|---|"]
        for sid in sorted(shared, key=lambda s: -per_sid[s])[:12]:
            eids = shared[sid]
            lines.append(f'| {sid} | {", ".join(str(e) for e in eids)} | '
                         f'{per_sid[sid]:,} |')
        if len(shared) > 12:
            lines.append(f'| … | {len(shared) - 12} more | |')
        lines.append("")
    return lines


def render(net: dict, graph_path, c_name: str) -> str:
    """The whole markdown document, as a string."""
    calls = net["calls"]
    kinds = {}
    for c in calls:
        kinds[c["func"]] = kinds.get(c["func"], 0) + 1
    inp, outp = net["input"], net["output"]
    lines = [
        "# Network — layers, shapes, dtypes and connectivity", "",
        "Generated by `src/frontend/tvmrelay/network_md.py` from "
        f"`{Path(graph_path).name}`. **Do not edit by hand** — it is rewritten "
        "on every `deploy_flow.py` run.", "",
        f"    graph-sha: {net['graph_sha']}   ({len(calls)} kernel calls)", "",
        "The same line is in `arm_build/graph_driver.c`. If they differ, one "
        "of the two is from an earlier build — a tree can hold several graph "
        "shapes in one session (int16 vs int8 operands, with and without "
        "`--aie-offload`), and both files are rewritten every run:", "",
        "```bash",
        "grep -m1 graph-sha network.md arm_build/graph_driver.c",
        "```", "",
    ]
    if net.get("params_status", "ok") != "ok":
        lines += [
            "> **WARNING — parameters are NOT described in this document.**",
            f"> The parameter blob is {net['params_status']}, and it is the "
            "only thing that tells a constant from an activation. Without it "
            "every weight is reported as an activation, so the counts below "
            "read as a model with no parameters. Re-run `deploy_flow.py` (or "
            "point `--c-name` at the right stem) and regenerate.",
            "",
        ]
    lines += [
        "This is the graph *after* compilation, so it is what actually runs: "
        "layouts are the transformed ones (`NCHWc`), quantized ops appear as "
        "the integer arithmetic they lowered to, and fused groups are single "
        "kernels rather than the ops you wrote.", "",
        "## Summary", "",
        "| | |", "|---|---|",
        f"| kernel calls | {len(calls)} |",
        f"| distinct kernels | {len(kinds)} |",
        f"| tensors (entries) | {net['n_tensors']} |",
        f"| storage buffers | {len(net['storage'])} |",
        f"| parameters | {net['n_params']} |",
    ]
    if inp:
        lines.append(f"| input | `{inp['name']}` `{inp['dtype']}` "
                     f"`{_fmt_shape(inp['shape'])}` |")
    lines += [
        f"| output | `{outp['dtype']}` `{_fmt_shape(outp['shape'])}` |",
        f"| kernels source | `{c_name}` |",
        "",
    ]
    repeated = {k: v for k, v in kinds.items() if v > 1}
    if repeated:
        lines += [f"{len(repeated)} kernel(s) are invoked more than once "
                  f"(identical fused groups share one function):", ""]
        for fn, n in sorted(repeated.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{_short(fn)}` × {n}")
        lines.append("")
    return "\n".join(lines + _mermaid(net) + _layer_table(net)
                     + _layer_detail(net) + _params_section(net)
                     + _buffer_reuse(net))


def write_network_md(out_dir, c_name: str = "model.c",
                     verbose: bool = True):
    """Write ``<out_dir>/network.md``. Returns its path, or ``None``.

    Returns ``None`` rather than raising when the graph JSON is absent -- a
    flow that stopped before stage 4 has nothing to describe, and that is not
    an error worth failing the run over.
    """
    out_dir = Path(out_dir)
    stem = Path(c_name).stem
    graph_path = out_dir / f"{stem}_graph.json"
    if not graph_path.is_file():
        if verbose:
            print(f"  [net] skipped -- {graph_path.name} not found")
        return None

    net = build_network(graph_path, out_dir / f"{stem}_params.bin",
                        out_dir / "layers" / "manifest.json")
    path = out_dir / "network.md"
    path.write_text(render(net, graph_path, c_name))
    if verbose:
        print(f"  [net] {path} ({len(net['calls'])} layers, "
              f"{net['n_tensors']} tensors, {len(net['storage'])} buffers)")
        if net.get("params_status", "ok") != "ok":
            print(f"  [net] WARNING: parameter blob {net['params_status']} -- "
                  f"every weight is reported as an activation, so the document "
                  f"says 0 parameters. It is not describing the model's "
                  f"constants at all.")
    return path


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", nargs="?", default="worklocal/tvmrelay_deploy",
                    help="directory holding the generated graph + params")
    ap.add_argument("--c-name", default="model.c",
                    help="kernel source name, used to find <stem>_graph.json")
    args = ap.parse_args(argv)
    return 0 if write_network_md(args.out_dir, args.c_name) else 1


if __name__ == "__main__":
    raise SystemExit(main())
