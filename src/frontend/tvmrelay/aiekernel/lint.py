###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Static checks on a user kernel source, before any compiler runs.

Every rule here is a failure this repo has already paid for, and almost all of
them are SILENT: the aiehlc build still exits 0, the pipeline still "succeeds",
and the symptom shows up minutes later as an empty module, an undeclared
identifier in a translation unit the user never wrote, or wrong pixels on the
board. The rule text names the skill that documents each one.

==== ============================================================== =========
code what                                                          level
==== ============================================================== =========
L001 no ``AIEK_SPEC``                                              error
L002 malformed ``AIEK_SPEC``                                       error
L003 the spec's entry is not defined                               error
L004 ``aie_user_kernel.h`` not included                            error
L005 entry name unusable (not an identifier / ``aie_`` prefix)     error
L010 no AIE kernel -- reference-only (detect/calibrate only)       info
L011 the kernel keyword inside a comment or string                 error
L012 the launch token inside a comment or string                   error
L014 something that reaches the launch is named above the kernel   error
L015 launch without ``.partition(...)`` in the same function       error
L016 a ``main()`` -- library mode would link an ELF instead        error
L017 kernel body calls a function (helpers must be ``#define``)    error
L018 kernel body uses a macro not ``#define``-d in this source     warning
L019 quoted header include from a subdirectory (flat copy)         error
L020 quoted ``.cc`` include that does not resolve                  error
L021 control-plane pragma without ``control_plan_op_control_packet`` error
L022 ``control_plan_shim_bd_ctrl`` without the dedicated-shim flag  warning
L023 ``control_plan_shim_bd_ctrl`` without high-throughput enable   error
L025 int32 read through a cast of a byte array (A78 fault)          warning
L027 symbolic dims in a spec that also has an AIE kernel           warning
L028 no/missing ``reference`` file                                 warn/err
L029 the reference file does not define ``<entry>_ref``            error
L031 int16 accumulator in the kernel body                          warning
==== ============================================================== =========
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .spec import (SourceText, SpecError, load_source, parse_specs,
                   strip_comments_and_strings)

__all__ = ["Finding", "LintReport", "lint_source", "KERNEL_KEYWORD", "LAUNCH_TOKEN"]

# Spelled by concatenation so that grepping THIS file for the raw keyword
# finds nothing a careless copy-paste could carry into a kernel source.
KERNEL_KEYWORD = "__" + "global" + "__"
LAUNCH_TOKEN = "<" * 3


@dataclass
class Finding:
    level: str          # error | warning | info
    code: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7s} {self.code} {self.where}: {self.message}"


@dataclass
class LintReport:
    source: Path
    findings: list = field(default_factory=list)
    specs: list = field(default_factory=list)
    kernels: list = field(default_factory=list)       # AIE kernel function names
    launches: list = field(default_factory=list)      # (function, kernel) pairs
    included: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [f for f in self.findings if f.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def has_aie_kernel(self) -> bool:
        return bool(self.kernels)

    def add(self, level, code, where, message):
        self.findings.append(Finding(level, code, where, message))

    def text(self) -> str:
        if not self.findings:
            return "  lint: clean"
        return "\n".join(f"  {f}" for f in self.findings)


# ═══════════════════════════════════════════════════════════════════════════
#  Structure: top-level brace blocks and the functions they belong to
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _Func:
    name: str
    head: int       # offset of the name
    open: int       # offset of '{'
    close: int      # offset of the matching '}'


_TRANSPARENT = re.compile(r'(extern\s*"\s*"|namespace(\s+\w+)?)\s*$')


def _match_back(clean: str, close: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(close, -1, -1):
        if clean[i] == close_ch:
            depth += 1
        elif clean[i] == open_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _func_name(clean: str, brace: int):
    """Name of the function whose body opens at *brace*, else None.

    ``... name(params) [const|noexcept] {`` -> name; ``AIEK_ENTRY(x) {`` and
    ``AIEK_REFERENCE(x) {`` -> x. An initializer (``= {``) is not a function.
    """
    head = clean[:brace].rstrip()
    head = re.sub(r"(\b(const|noexcept|override|final)\s*)+$", "", head).rstrip()
    if not head.endswith(")"):
        return None, -1
    lp = _match_back(head, len(head) - 1, "(", ")")
    if lp < 0:
        return None, -1
    m = re.search(r"([A-Za-z_]\w*)\s*$", head[:lp])
    if not m or m.group(1) in ("if", "for", "while", "switch", "sizeof", "return"):
        return None, -1
    if m.group(1) in ("AIEK_ENTRY", "AIEK_REFERENCE"):
        inner = head[lp + 1:-1].strip()
        suffix = "_ref" if m.group(1) == "AIEK_REFERENCE" else ""
        return inner + suffix, m.start(1)
    return m.group(1), m.start(1)


def _functions(clean: str) -> list:
    """Every function body at file scope (``extern "C" {`` / namespaces are
    transparent). Brace matching runs on comment- and string-free text."""
    funcs, stack = [], []          # stack entries: (open_offset, transparent)
    for i, c in enumerate(clean):
        if c == "{":
            transparent = bool(_TRANSPARENT.search(clean[max(0, i - 40):i]))
            stack.append((i, transparent))
        elif c == "}" and stack:
            start, transparent = stack.pop()
            depth = sum(1 for _, t in stack if not t)
            if transparent or depth:
                continue
            name, head = _func_name(clean, start)
            if name:
                funcs.append(_Func(name, head, start, i))
    return funcs


def _kernels(clean: str) -> list:
    """``(name, offset)`` of each AIE kernel definition."""
    pat = re.compile(re.escape(KERNEL_KEYWORD)
                     + r"\s*(\([^)]*\))?\s*void\s+([A-Za-z_]\w*)\s*\(")
    return [(m.group(2), m.start()) for m in pat.finditer(clean)]


def _enclosing(funcs: list, offset: int):
    for f in funcs:
        if f.open < offset < f.close:
            return f
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Rules
# ═══════════════════════════════════════════════════════════════════════════

def _rule_tokens_in_comments(src: SourceText, clean: str, rep: LintReport):
    """L011/L012: the aiehlc rewriter matches TEXT (skill aiesourcetextrewrite)."""
    for token, code, what in ((KERNEL_KEYWORD, "L011", "the kernel keyword"),
                              (LAUNCH_TOKEN, "L012", "the launch token")):
        for m in re.finditer(re.escape(token), src.text):
            if clean[m.start():m.end()] != token:
                rep.add("error", code, src.where(m.start()),
                        f"{what} appears inside a comment or string; aiehlc's "
                        f"source rewriter matches text, so it rewrites the wrong "
                        f"code and the build still succeeds (skill: "
                        f"aiesourcetextrewrite). Reword it.")


def _rule_launch_structure(src, clean, funcs, kernels, rep: LintReport):
    """L014/L015: kernel-first ordering and the mesh VarDecl rule."""
    if not kernels:
        return
    first_kernel = min(off for _, off in kernels)
    launch_re = re.compile(r"([A-Za-z_]\w*)\s*" + re.escape(LAUNCH_TOKEN)
                           + r"\s*\w+\s*>>>")
    reaching = set()
    for m in launch_re.finditer(clean):
        fn = _enclosing(funcs, m.start())
        rep.launches.append((fn.name if fn else None, m.group(1)))
        if fn is None:
            rep.add("error", "L015", src.where(m.start()),
                    "launch outside any function")
            continue
        reaching.add(fn.name)
        if ".partition(" not in clean[fn.open:fn.close]:
            rep.add("error", "L015", src.where(m.start()),
                    f"{fn.name}() launches {m.group(1)} but has no "
                    f"`device.partition(...)` of its own: aiehlc recovers the "
                    f"mesh from that function-local VarDecl, in the SAME "
                    f"function as the launch")
    # Callers of a launching function reach the launch too (one level is what
    # the conv2dstem layout needs: entry -> stem_run -> launch).
    for f in funcs:
        body = clean[f.open:f.close]
        if any(re.search(rf"\b{re.escape(r)}\s*\(", body) for r in reaching):
            reaching.add(f.name)
    for name in sorted(reaching):
        m = re.search(rf"\b{re.escape(name)}\s*\(", clean[:first_kernel])
        if m:
            rep.add("error", "L014", src.where(m.start()),
                    f"`{name}` reaches the launch and is named ABOVE the kernel "
                    f"definition; the aiehlc frontend then visits the launch "
                    f"before the kernel is registered and emits an EMPTY module "
                    f"(skill: aiesourcetextrewrite). Declare/define it after the "
                    f"kernel, and include headers that declare it after it too.")


def _rule_kernel_body(src, clean, funcs, kernels, rep: LintReport):
    """L017/L018/L031: what the extracted kernel translation unit can see."""
    defines = set(re.findall(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)", src.text, re.M))
    host_funcs = {f.name for f in funcs}
    for kname, off in kernels:
        body_fn = next((f for f in funcs if f.name == kname and f.head > off), None)
        if body_fn is None:
            continue
        body = clean[body_fn.open:body_fn.close]
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body):
            callee = m.group(1)
            if callee in host_funcs and callee != kname:
                rep.add("error", "L017", src.where(body_fn.open + m.start()),
                        f"kernel {kname} calls {callee}(), a function: the kernel "
                        f"extractor copies only #define lines of this file into "
                        f"kernel.cc, so it fails with 'use of undeclared "
                        f"identifier'. Make {callee} a macro.")
        unknown = sorted({t for t in re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", body)
                          if t not in defines})
        if unknown:
            rep.add("warning", "L018", src.where(body_fn.open),
                    f"kernel {kname} uses {', '.join(unknown[:6])}"
                    f"{' ...' if len(unknown) > 6 else ''}, not #define-d in this "
                    f"source; a macro from a header does not reach kernel.cc")
        m = re.search(r"\bint16_t\s+(sum|acc)\w*", body)
        if m:
            rep.add("warning", "L031", src.where(body_fn.open + m.start()),
                    "int16 accumulator: with real int8 weights a conv "
                    "accumulator wraps silently (stem peak |acc| = 1,023,225); "
                    "use int32")


def _rule_includes(src: SourceText, rep: LintReport):
    """L019/L020: aiehlc.sh copies the source dir's *.h FLAT (skill
    conv2dstemfixture) and splices quoted .cc includes."""
    for name, path, line in src.missing:
        rep.add("error", "L020", f"{path.name}:{line}",
                f'#include "{name}" does not resolve against {src.path.parent} '
                f"(aiehlc resolves every spliced .cc against the ENTRY dir)")
    for m in re.finditer(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+\.h(pp)?)"', src.text, re.M):
        if "/" in m.group(1):
            rep.add("error", "L019", src.where(m.start()),
                    f'"{m.group(1)}": aiehlc.sh copies headers with a flat, '
                    f"non-recursive *.h glob, so it parses here and then fails "
                    f"in the cross host compile. Put it next to the entry file.")


def _rule_pragmas(src: SourceText, clean: str, rep: LintReport):
    """L021-L023: the control-plane pragma family (see conv2dstem.cc)."""
    prag = set(re.findall(r"^[ \t]*#[ \t]*pragma[ \t]+(\w+)", src.text, re.M))
    plan = "control_plan_op_control_packet" in prag
    for p in ("CONTROL_PLAN_GROUP_REG_WRITE", "control_plan_shim_bd_ctrl",
              "control_plan_dedicated_shim"):
        if p in prag and not plan:
            rep.add("error", "L021", src.path.name,
                    f"#pragma {p} needs #pragma control_plan_op_control_packet")
    if "control_plan_shim_bd_ctrl" in prag:
        if "control_plan_dedicated_shim" not in prag:
            rep.add("warning", "L022", src.path.name,
                    "control_plan_shim_bd_ctrl alone only WARNs and falls back to "
                    "MMIO when no spare column exists; add "
                    "control_plan_dedicated_shim to make that a build error")
        if not re.search(r"__Runtime_ctrl_high_throughput_enable\s*\(\s*1\s*\)", clean):
            rep.add("error", "L023", src.path.name,
                    "control_plan_shim_bd_ctrl requires "
                    "__Runtime_ctrl_high_throughput_enable(1) before device init")


def _rule_unaligned(src: SourceText, clean: str, rep: LintReport):
    """L025: skill conv2dstemtvmlayer -- `unsigned char _data[]` read as int32."""
    for m in re.finditer(r"\(\s*(const\s+)?u?int32_t\s*\*\s*\)\s*\(?\s*\w*_data\b", clean):
        rep.add("warning", "L025", src.where(m.start()),
                "int32 read through a cast of a byte array: an unaligned load "
                "that faults on the A78. Use aiek_rd_i32().")


def _rule_spec(src, clean, funcs, kernels, rep: LintReport):
    """L001-L005, L027-L029."""
    try:
        rep.specs = parse_specs(src)
    except SpecError as exc:
        rep.add("error", "L002", exc.where or src.path.name, str(exc))
        return
    if not rep.specs:
        rep.add("error", "L001", src.path.name,
                "no AIEK_SPEC(entry, \"...\") -- nothing says which layers this "
                "kernel replaces")
        return
    if not re.search(r'#\s*include\s*[<"]aie_user_kernel\.h[>"]', src.text):
        rep.add("error", "L004", src.path.name,
                '#include "aie_user_kernel.h" is missing (AIEK_* macros, aiek_tensor)')
    names = {f.name for f in funcs}
    for spec in rep.specs:
        if not re.fullmatch(r"[A-Za-z_]\w*", spec.entry) or spec.entry.startswith("aie_"):
            rep.add("error", "L005", spec.where,
                    f"entry {spec.entry!r}: must be a C identifier not starting "
                    f"with aie_ (the generated adapters own that prefix)")
        if kernels and spec.entry not in names:
            rep.add("error", "L003", spec.where,
                    f"AIEK_SPEC names entry {spec.entry} but no "
                    f"AIEK_ENTRY({spec.entry}) {{...}} is defined")
        if kernels and spec.symbolic:
            rep.add("warning", "L027", spec.where,
                    "symbolic/`*` dims with an AIE kernel: aiehlc bakes ONE "
                    "geometry into the build, so the entry must itself handle "
                    "every shape the spec accepts")
        # A reference-only source may BE the reference: spec and
        # AIEK_REFERENCE in one .c, no `reference` line needed.
        if spec.reference is None and f"{spec.entry}_ref" in names:
            spec.reference = src.path
        _rule_reference(spec, rep)


def _rule_reference(spec, rep: LintReport):
    if spec.reference is None:
        rep.add("warning", "L028", spec.where,
                "no `reference <file.c>`: calibrate and emulate need an x86 "
                "reference, so only detection is possible")
        return
    if not spec.reference.is_file():
        rep.add("error", "L028", spec.where, f"reference {spec.reference} not found")
        return
    ref = strip_comments_and_strings(spec.reference.read_text(errors="replace"))
    if not re.search(rf"AIEK_REFERENCE\s*\(\s*{spec.entry}\s*\)\s*\{{|"
                     rf"\b{spec.entry}_ref\s*\([^;]*\)\s*\{{", ref):
        rep.add("error", "L029", spec.reference.name,
                f"does not define AIEK_REFERENCE({spec.entry}) "
                f"(i.e. {spec.entry}_ref)")


_PREPROC = re.compile(r"^[ \t]*#(?:[^\n]*\\\n)*[^\n]*", re.M)


def _blank_preprocessor(clean: str) -> str:
    """Blank directive lines (with continuations), offsets preserved, so a
    ``#define F(x) { ... }`` is not mistaken for a function body."""
    return _PREPROC.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), clean)


def lint_source(path) -> LintReport:
    """Run every rule on *path* (after splicing its ``.cc`` includes)."""
    src = load_source(path)
    clean = strip_comments_and_strings(src.text)
    rep = LintReport(source=src.path, included=src.included)
    code = _blank_preprocessor(clean)
    funcs = _functions(code)
    kernels = _kernels(code)
    rep.kernels = [k for k, _ in kernels]

    _rule_spec(src, clean, funcs, kernels, rep)
    _rule_tokens_in_comments(src, clean, rep)
    _rule_includes(src, rep)
    if not kernels:
        rep.add("info", "L010", src.path.name,
                "no AIE kernel in this source: reference-only -- lint/detect/"
                "calibrate/emulate work, build/offload do not")
        return rep
    _rule_launch_structure(src, code, funcs, kernels, rep)
    _rule_kernel_body(src, code, funcs, kernels, rep)
    _rule_pragmas(src, clean, rep)
    _rule_unaligned(src, code, rep)
    if re.search(r"\b(int|void)\s+main\s*\(", code):
        rep.add("error", "L016", src.path.name,
                "defines main(): hostcompile.sh would link an ELF instead of the "
                "lib<app>.a a layer links against (skill: hostlibrarymode)")
    return rep
