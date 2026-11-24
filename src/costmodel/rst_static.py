###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Static ``core_reset`` count per launch (M1.2.2 P0): a pure function of a build directory.

The measured side is ``LAUNCH split_detail calls ... core_reset=<n>`` divided by the launch count
(launch_split.py).  It is read from the build products only -- no board counters:

1. Which main prints split_detail and what it prints for ``core_reset``:
   * aiegraph boardmain (UNet, ``<build>/ag_work/arm_build/main.c``, template
     src/aiegraph/runtime/main_launch_line.c:56,61) prints ``core_reset=%u`` with ``krstn`` =
     runtime counter ``g_kl_rst_n`` (__Runtime_kload_split_cycles, src/mlir/runtime/aie_runtime.c:334-344).
   * the P0 kernel host (``<build>/kacc_gemm.cc``, from script/perf/p0_kacc_host.cc:173) prints a
     literal ``core_reset=0``: the resets still happen on the board (its split_words rst_n is 33.0, which
     is why launch_l.py does not use rst_n), but split_detail reports 0.
2. ``g_kl_rst_n`` only counts under ``-DAIEHLC_PROFILING=1`` (RT_PROF_ADD, aie_runtime.c:107-125; the
   define is in the build log: UNet ``<build>.build.log`` line ``[build_unet] ... defs=``, kernels
   ``<build>/build.log``).
3. Which kernel-load entry point the generated host code calls once per launch
   (``.../worklocal/build/host_fixed.cc``):
   * ``__Runtime_load_kernel_group_{4,8,16}t(..., n)`` -> ``__Runtime_load_kernel_group_nt``
     (aie_runtime.c:7636-7658, 7590-7633), which adds to ``g_kl_rst_n``
       - 1 per call for the DMA channel reset+unreset of all tiles once a group is loaded
         (rt_relaunch_dma_reset, aie_runtime.c:7575-7586; skipped on the very first load, :7576), and
       - 2 per AIE core tile (CoreDisable+CoreReset at :7614-7617, CoreUnreset at :7622-7623); a core tile
         has row >= XAIE_AIE_TILE_ROW_START = 3 (aie_runtime.c:2419, alib xaie_lite_hwcfg.h:101).
     So steady state is 2 * n_core + 1 per launch: 16 core tiles -> 33.0.  This is the UNet warm opts
     ``core_reset_unreset_per_launch`` (boardmain.py:33): UNet switches kernels on the same 16 tiles, so
     every launch reloads.  (Board: 258 launches -> 33*258 - 1 = 8513, the -1 is the first load.)
   * ``__Runtime_load_kernel_group_{4,8,16}t_ctrl`` (control-packet ELF load, aot / aot-kco) ->
     rt_ctrl_load_kernel_group_nt (aie_runtime.c:8410-8440): no RT_PROF_ADD on g_kl_rst_n -> 0.

rst_per_launch(build_dir) -> float (steady state, per launch).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

AIE_TILE_ROW_START = 3  # alib xaie_lite_hwcfg.h:101 (AIE2PS)

# where the split_detail printing main lives in each build layout
_MAINS = ("ag_work/arm_build/main.c", "kacc_gemm.cc")
# generated host code per build layout (kernel: aout/; UNet: one dir per aiek kernel / per AIE layer)
_HOSTS = ("aout/worklocal/build/host_fixed.cc", "ag_work/aiek/*/worklocal/build/host_fixed.cc",
          "ag_work/layers/*/aie/build/host_fixed.cc")

_DETAIL_RST = re.compile(r'LAUNCH split_detail calls [^"]*?core_reset=(%u|\d+)')
_LOAD = re.compile(r"__Runtime_load_kernel_group_(\d+)t(_ctrl)?\(([^;]*)\);")
_TILELOC = re.compile(r"XAie_LocType\s+(v\d+)\s*=\s*XAie_TileLoc\((\d+),\s*(\d+)\)")
_PROF = re.compile(r"-DAIEHLC_PROFILING=1\b")


def _main_src(b: Path) -> Optional[Path]:
    for rel in _MAINS:
        p = b / rel
        if p.is_file() and "LAUNCH split_detail" in p.read_text(errors="replace"):
            return p
    return None


def _build_logs(b: Path) -> List[Path]:
    return [p for p in (b / "build.log", b.parent / (b.name + ".build.log")) if p.is_file()]


def _sites(text: str) -> List[Dict[str, Any]]:
    """Every kernel-group load call in one host_fixed.cc with its core-tile count."""
    out = []
    # tile variables are function-local; resolve each call against the TileLoc lines before it
    for m in _LOAD.finditer(text):
        width, ctrl, args = int(m.group(1)), bool(m.group(2)), [a.strip() for a in m.group(3).split(",")]
        if ctrl:  # (dev, &fabric, t0..t{w-1}, n, bd_id, mm2s_ch)
            tiles, n = args[2:2 + width], int(args[2 + width])
        else:  # (dev, t0..t{w-1}, n)
            tiles, n = args[1:1 + width], int(args[1 + width])
        fn_start = text.rfind("\nvoid ", 0, m.start())
        locs = {v: (int(c), int(r)) for v, c, r in _TILELOC.findall(text, max(fn_start, 0), m.start())}
        rows = [locs[t][1] for t in tiles[:n] if t in locs]
        n_core = sum(1 for r in rows if r >= AIE_TILE_ROW_START) if len(rows) == n else n
        out.append({"ctrl": ctrl, "n": n, "n_core": n_core})
    return out


def rst_static(build_dir) -> Dict[str, Any]:
    """Per-launch split_detail core_reset from the build products, with the reason."""
    b = Path(build_dir)
    main = _main_src(b)
    if main is None:
        raise FileNotFoundError(f"{b}: no main printing 'LAUNCH split_detail' ({', '.join(_MAINS)})")
    m = _DETAIL_RST.search(main.read_text(errors="replace"))
    if m is None:
        raise ValueError(f"{main}: split_detail printf has no core_reset field")
    if m.group(1) != "%u":  # literal (p0_kacc_host.cc:173): the counter is not reported
        return {"rst": float(m.group(1)), "why": f"literal core_reset={m.group(1)} in {main.name}",
                "main": str(main), "sites": 0}
    logs = _build_logs(b)
    if not any(_PROF.search(p.read_text(errors="replace")) for p in logs):
        return {"rst": 0.0, "why": "no -DAIEHLC_PROFILING=1 (RT_PROF_ADD compiled out)", "main": str(main),
                "sites": 0}
    sites: List[Dict[str, Any]] = []
    for pat in _HOSTS:
        for h in sorted(b.glob(pat)):
            sites += _sites(h.read_text(errors="replace"))
    if not sites:
        raise FileNotFoundError(f"{b}: no __Runtime_load_kernel_group_* call in {', '.join(_HOSTS)}")
    # one kernel-group load per launch: steady state 2*n_core + 1 (MMIO path), 0 (control-packet path)
    per = [0.0 if s["ctrl"] else 2.0 * s["n_core"] + 1.0 for s in sites]
    rst = sum(per) / len(per)
    kinds = sorted({("ctrl" if s["ctrl"] else f"mmio n_core={s['n_core']}") for s in sites})
    return {"rst": rst, "why": f"{len(sites)} load sites: {', '.join(kinds)}", "main": str(main),
            "sites": len(sites)}


def rst_per_launch(build_dir) -> float:
    """core_reset calls per launch that split_detail reports for the ELF built in *build_dir*."""
    return rst_static(build_dir)["rst"]
