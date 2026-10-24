###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Offload specs in every spelling a user might type -> ``[(layer_ref, source)]``.

    "layer1=aie/stem.cc"                    one pair
    "layer1=a.cc; layer4-9=b.cc"            several
    "aie/conv3x3.cc"                        every layer the kernel fits (auto)
    ("layer1", "aie/stem.cc")               a tuple
    {"layer1": "a.cc", "layer4": "b.cc"}    a dict
    [ ...any of the above... ]              a list
"""

from __future__ import annotations

from pathlib import Path

from .errors import AiegraphError

__all__ = ["parse_offload", "group_by_source"]


def _one(item) -> list:
    if isinstance(item, dict):
        return [(k, v) for k, v in item.items()]
    if isinstance(item, tuple) and len(item) == 2:
        return [(item[0], item[1])]
    if isinstance(item, (str, Path)):
        out = []
        for part in str(item).split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                ref, src = part.split("=", 1)
                out.append((ref.strip(), src.strip()))
            else:
                out.append(("auto", part))
        return out
    raise AiegraphError("offload", f"cannot read offload spec {item!r}",
                        hint='use "layer1=kernel.cc", ("layer1", "kernel.cc") or '
                             '{"layer1": "kernel.cc"}')


def parse_offload(spec) -> list:
    """Normalize *spec* to ``[(layer_ref, Path)]``; sources must exist."""
    if spec is None:
        return []
    items = spec if isinstance(spec, list) else [spec]
    pairs = [p for item in items for p in _one(item)]
    for _ref, src in pairs:
        if not Path(src).is_file():
            raise AiegraphError("offload", f"kernel source {src} does not exist",
                                hint="paths are relative to the current directory")
    return [(ref, Path(src).resolve()) for ref, src in pairs]


def group_by_source(pairs, resolve) -> dict:
    """``{source: [layer indices] | None}`` -- None means "every layer it fits"."""
    out = {}
    for ref, src in pairs:
        if ref in (None, "auto", "*"):
            out[src] = None
            continue
        if out.get(src, []) is None:
            continue
        out.setdefault(src, [])
        out[src] += [i for i in resolve(ref) if i not in out[src]]
    return out
