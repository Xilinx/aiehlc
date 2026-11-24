###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiednn -- the AIE kernel / host-glue template library aiegraph offloads with.

Templates are plain source files, one directory per operator category:

    src/aiednn/conv2d/       conv as bf16 gemm (im2col / NCHWc host glue, 64x64 tile and K-in-core kernels)
    src/aiednn/gemm/         gemm_kacc kernel, dense (linear) host glue, control-packet pragmas, benches
    src/aiednn/attention/    flash attention (fused QK^T -> softmax -> PV) and bmm host glue
    src/aiednn/norm/         normalisation steps (online softmax + 1/l rescale of flash attention)
    src/aiednn/elementwise/  fused epilogues (bias / per-channel add / residual add)
    src/aiednn/layout/       layout helpers (NEON 4x4 transposes)

Board-side runtime helpers that are not kernels (compare tool, profiling glue) live in
``src/aiegraph/runtime/`` and are read with :func:`runtime_source`.

A template is returned VERBATIM (bytes as on disk, no newline translation) minus its leading AMD
license header block (:func:`strip_license`): the header documents the file in the repo and must not
ride along into every generated source. Parameters are spelled ``@NAME@`` inside the text;
:func:`params` lists them and :func:`render` fills them.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / "aiegraph" / "runtime"
CATEGORIES = ("conv2d", "gemm", "attention", "norm", "elementwise", "layout")
SUFFIXES = (".cc", ".c", ".h")
_PARAM = re.compile(r"@([A-Z][A-Z0-9_]*)@")

__all__ = ["ROOT", "RUNTIME", "CATEGORIES", "TemplateError", "path", "template", "params", "render", "runtime_path",
           "runtime_source", "templates", "strip_license", "GEMM_KACC_TPARAMS", "gemm_kacc_targs",
           "gemm_kacc_launch"]

# The AMD Apache-2.0 header a template / runtime source carries on its first lines (C block comment).
_LICENSE = re.compile(r"\A/\*{20,}\r?\n"
                      r" \* Copyright \(C\) \d{4}(?:-\d{4})? Advanced Micro Devices, Inc\. All Rights Reserved\.\r?\n"
                      r" \* SPDX-License-Identifier: [A-Za-z0-9.\-+]+\r?\n"
                      r" \*{20,}/\r?\n")


class TemplateError(LookupError):
    pass


def strip_license(text: str) -> str:
    """*text* without its leading license header block (unchanged when it has none)."""
    return _LICENSE.sub("", text, count=1)


def _read(p: Path) -> str:
    with open(p, encoding="utf-8", newline="") as f:
        return strip_license(f.read())


def path(name: str) -> Path:
    """``"<category>/<file>"`` (an optional ``aiednn/`` prefix is accepted) -> the file under src/aiednn."""
    rel = name[len("aiednn/"):] if name.startswith("aiednn/") else name
    parts = rel.split("/")
    if len(parts) != 2 or not parts[1]:
        raise TemplateError(f"template name {name!r}: expected '<category>/<file>'")
    cat, fn = parts
    if cat not in CATEGORIES:
        raise TemplateError(f"template {name!r}: unknown category {cat!r} (one of {', '.join(CATEGORIES)})")
    if not fn.endswith(SUFFIXES):
        raise TemplateError(f"template {name!r}: not a C/C++ source ({', '.join(SUFFIXES)})")
    p = ROOT / cat / fn
    if not p.is_file():
        raise TemplateError(f"template {name!r}: no file {p}")
    return p


def template(name: str) -> str:
    """The template text, verbatim after the license header."""
    return _read(path(name))


def params(name: str) -> list:
    """Sorted ``@NAME@`` parameters of a template (its parameter header)."""
    return sorted(set(_PARAM.findall(template(name))))


def render(name: str, **kv) -> str:
    """Fill every ``@NAME@`` of a template; missing or unknown parameters are errors."""
    t = template(name)
    want = set(_PARAM.findall(t))
    if set(kv) != want:
        raise TemplateError(f"template {name!r}: missing {sorted(want - set(kv))} unknown {sorted(set(kv) - want)}")
    for k, v in kv.items():
        t = t.replace(f"@{k}@", str(v))
    return t


def templates(category: str | None = None) -> list:
    """Every template name (``"<category>/<file>"``), optionally of one category."""
    cats = [category] if category else list(CATEGORIES)
    out = []
    for c in cats:
        if c not in CATEGORIES:
            raise TemplateError(f"unknown category {c!r}")
        d = ROOT / c
        out += sorted(f"{c}/{p.name}" for p in d.glob("*") if p.is_file() and p.suffix in SUFFIXES) if d.is_dir() else []
    return out


def runtime_path(name: str) -> Path:
    """A board-side runtime helper in src/aiegraph/runtime (e.g. ``cmp_cos.c``)."""
    p = RUNTIME / name
    if "/" in name or not p.is_file():
        raise TemplateError(f"runtime source {name!r}: no file {p}")
    return p


def runtime_source(name: str) -> str:
    return _read(runtime_path(name))


# gemm/gemm_kacc_kernel.cc is a C++ template __global__ (aiehlc instantiates one kernel per distinct argument list
# in the same source file; a single instance keeps the bare name gemm_kacc). Its template parameters, in order:
GEMM_KACC_TPARAMS = ("CH", "KDIM", "PX", "KW", "AROWS", "BROWS", "TKW", "KCH", "NKC", "NT", "S", "TR")


def gemm_kacc_targs(ch: int, k: int, px: int, g: dict) -> str:
    """``<...>`` template argument list of the gemm_kacc instance (ch, k, px) with walk geometry *g* (a kacc_geom
    dict: ntile, S, tkw, kch, nkc, kp, optional tr / arows / brows; default = not transposed)."""
    tr = int(g.get("tr", 0))
    vals = (ch, k, px, g["kp"] // 2, g.get("arows", px if tr else ch), g.get("brows", ch if tr else px), g["tkw"],
            g["kch"], g["nkc"], g["ntile"], g["S"], tr)
    return "<" + ", ".join(str(int(v)) for v in vals) + ">"


def gemm_kacc_launch(ch: int, k: int, px: int, g: dict) -> str:
    """The launch callee ``gemm_kacc<...>`` (write ``<callee><<<mesh>>>(a, b, c);`` in the TU that holds
    ``template("gemm/gemm_kacc_kernel.cc")``)."""
    return "gemm_kacc" + gemm_kacc_targs(ch, k, px, g)
