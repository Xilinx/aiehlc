#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Static L-segment features per launch from a kernel BUILD directory only (M1.2.2 P1, UNet P3).

  static_feat.py --build-root <root with k_<cfg>/<kernel>/ and unet_<cfg>/>
                 [--kernel-logs <dir with <cfg>/<kernel>.log>]
                 [--unet cfg=LOG [cfg=LOG ...]]      UNet ELF builds <build-root>/unet_<cfg>, LOG = board log
                 [--params src/costmodel/params/palmyra213.json] [--kcal src/costmodel/params/palmyra213_kcal.json]
                 [--cycles <aie2ps_kernel_cycles.json>]

UNet (--unet, P3): the launch sequence (258 launches: 48 uc_conv, 78 uc_linear, 132 fa_run_o over 44 kernels) and
site geometry come from graph_driver.c + aiek_unetconv.cc via unet_seq.py (copied from maxwell kswitch.py); every
launch takes its kernel's single-launch value from the same formulas below applied to host_canonicalized_<kernel>
(host.cc) and host_ctrlpkt_<kernel>.h; per launch = sum / launches. Two terms the 27 kernel points never exercise:
  - FA kernels issue 8 plain __Runtime_dma_bd_config on shim tiles (row 0): PH_BDCFG calls, 0 MMIO words
    (shim BD RunOp path), so tile BD words = 6 x non-shim __Runtime_dma_bd_config only.
  - rt_sb_begin (aie_runtime.c:6714) re-prepares the shim-BD window when the control fabric changes
    (kernel switch, 119 of 258 launches): that window also resends the shim-row return-route prefix (54 words
    for every UNet kernel) -> +24.9 cp_w (and bdcfg_w) per launch in aot / aot-kco.
  aot vs off, +2.0 bdcfg calls per launch: every aot kernel's host function has one __Runtime_ctrl_shim_bd_begin
  and one __Runtime_ctrl_shim_bd_commit -> one effective window per launch, charged as 2 PH_BDCFG phases
  (rt_sb_kick + rt_sb_ack_wait); the BD calls themselves are the same 369.4 as off.

Predicted per launch (no board counter, no log enters the prediction):
  bdcfg_calls  PH_BDCFG phases (aie_runtime.c RT_PROF_PHASE(PH_BDCFG, ...)):
               one per executed __Runtime_dma_bd_config[_multidim[_ooo]] call (aie_runtime.c:3841/6843/6949),
               + 2 per effective __Runtime_ctrl_shim_bd_commit (rt_sb_kick :6419 + rt_sb_ack_wait :6319);
                 rt_sb_commit is a no-op unless a __Runtime_ctrl_shim_bd_begin opened the window, so the
                 effective count is min(executed begins, executed commits),
               + 3 per __Runtime_ctrl_aot_window (window :7202 + kick + ack wait).
  startio      executed __Runtime_startio calls (aie_runtime.c:7092); aot-kco folds them into ioevent_make.
  bdcfg_w      words charged to PH_BDCFG:
               6 MMIO words per tile-DMA __Runtime_dma_bd_config (AIE2PS tile BD = 6 regs, Write32 backend op);
               shim multidim BDs go through XAIE_BACKEND_OP_CONFIG_SHIMDMABD (RunOp, not a counted
               Write32/BlockWrite32 op: aie_runtime.c rt_wc_install) = 0 words;
               per shim-BD control-packet window: cp_win words + 6 MMIO (4 XAie_DmaUpdateBdLen + 2
               XAie_DmaChannelPushBdToQueue, armed chain, rt_sb_kick).
  cp_w         per window cp_win = sum over the shim segments in worklocal/host_ctrlpkt.h of the segment words
               minus the leading return-route words (set up once by __Runtime_ctrl_shim_bd_begin, not resent).
  Execution counts: calls in host_canonicalized (aout/worklocal/host.cc) x the trip count of every enclosing
  `for (size_t v = vA; v < vB; v += vC)` loop (constants resolved from the function's size_t definitions).

Measured (only for the comparison): LAUNCH split_detail calls / launches and LAUNCH split_words of
<kernel-logs>/<cfg>/<kernel>.log (launch_split.lines() rejoin).  Rule: rel = |pred-meas|/meas, 0/0 -> 0;
word features with meas <= 16 use the absolute tolerance |pred-meas| <= 16 (rel recorded 0 if inside).
Any missing build/log or parse failure prints FAIL for the point; the ALL line needs n=27 and maxrel <= 0.05
(src=kernel), n=4 UNet cfgs, 258 launches each and maxrel <= 0.05 (src=unet; launches from 'LAUNCH warm|cold').

SFEAT_CCYC (report only, no PASS/FAIL): static compute cycles = des.compute_cycles() with the instance taken from
the build's gemm_kacc<...> template arguments (kacc_gemm.cc), core_cyc = the ISS cycle count of that kernel
(--cycles, static simulation, not a board counter) and the frozen kcal chunk/round overheads; measured = P0 line
compute_cyc= (active - lock_stall - stream_stall). Measured <= 0 -> rel=SKIP_NEG, not counted.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from costmodel import des  # noqa: E402
from costmodel.launch_split import lines, parse  # noqa: E402
from costmodel.unet_seq import unet_launches  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CYCLES = "/scratch/staff/huaj/aiengineers/maxwell/tasks/2026-11-19_kernel-cycles/aie2ps_kernel_cycles.json"
TILE_BD_WORDS = 6        # AIE2PS tile DMA BD: 6 x Write32
SB_KICK_MMIO = 6         # 4 XAie_DmaUpdateBdLen + 2 XAie_DmaChannelPushBdToQueue per armed shim-window kick
COMMIT_PHASES = 2        # rt_sb_kick + rt_sb_ack_wait
WINDOW_PHASES = 3        # __Runtime_ctrl_aot_window + rt_sb_kick + rt_sb_ack_wait
WORD_ABS_TOL = 16
REL_GATE = 0.05
N_EXPECT = 27
N_UNET = 4
UNET_LAUNCHES = 258
UNET_WL = os.path.join("ag_work", "aiek", "aiek_unetconv", "worklocal")

_FOR = re.compile(r"for \(size_t (v\d+) = (\w+); \1 < (\w+); \1 \+= (\w+)\)")
_CONST = re.compile(r"^\s*size_t (v\d+) = (\d+);")
_CALL = re.compile(r"\b(__Runtime_[A-Za-z0-9_]+)\(")
_LOC = re.compile(r"XAie_LocType (v\d+) = XAie_TileLoc\((\d+), (\d+)\);")
_BDCFG_TILE = re.compile(r"\b__Runtime_dma_bd_config\(\w+, (v\d+),")
SHIM_BD = "__Runtime_dma_bd_config@shim"   # host_counts key: __Runtime_dma_bd_config on a row-0 (shim) tile
_TARGS = re.compile(r"gemm_kacc<\s*(\d+(?:\s*,\s*\d+){11})\s*>")


def _walk(host_cc, fn, txt=None):
    """yield (raw line, code without comments, executions per launch, {tile var: row}) of host function fn:
    executions = product of the trip counts of the enclosing `for (size_t v = vA; v < vB; v += vC)` loops
    (constants resolved from the function's size_t definitions); loop header lines are not yielded."""
    if txt is None:
        txt = open(host_cc).read()
    m = re.search(r"^void %s\(.*?\{\n(.*?)^\}" % re.escape(fn), txt, re.S | re.M)
    if not m:
        raise ValueError("no %s in %s" % (fn, host_cc))
    consts, locs = {}, {}
    stack = []  # [(depth at loop open, trip)]
    depth = 0
    for line in m.group(1).split("\n"):
        code = re.sub(r"/\*.*?\*/", "", line)
        cm = _CONST.match(code)
        if cm:
            consts[cm.group(1)] = int(cm.group(2))
        for v, _col, row in _LOC.findall(code):
            locs[v] = int(row)
        mult = 1
        for _d, t in stack:
            mult *= t
        fm = _FOR.search(code)
        if fm:
            def val(x):
                return int(x) if x.isdigit() else consts[x]
            lo, hi, st = val(fm.group(2)), val(fm.group(3)), val(fm.group(4))
            trip = max(0, (hi - lo + st - 1) // st)
            code_after = code[fm.end():]
            stack.append((depth, trip))
            depth += code_after.count("{") - code_after.count("}")
            continue
        yield line, code, mult, locs
        depth += code.count("{") - code.count("}")
        while stack and depth <= stack[-1][0]:
            stack.pop()


def host_counts(host_cc, fn="host_canonicalized", txt=None):
    """{runtime API name: executed calls per launch} of host function fn (host_canonicalized[_<kernel>])."""
    counts = {}
    for _line, code, mult, locs in _walk(host_cc, fn, txt):
        for c in _CALL.findall(code):
            counts[c] = counts.get(c, 0) + mult
        # a plain __Runtime_dma_bd_config on a shim tile (row 0) takes the XAIE_BACKEND_OP_CONFIG_SHIMDMABD RunOp
        # path like the multidim shim BDs: a PH_BDCFG call but 0 counted MMIO words (UNet FA kernels, 8/launch)
        for v in _BDCFG_TILE.findall(code):
            if locs.get(v) == 0:
                counts[SHIM_BD] = counts.get(SHIM_BD, 0) + mult
    return counts


_BDC = re.compile(r"/\* DMA BD Config: (.*?) \*/")
_CIO = re.compile(r"/\* Create IO: channel_id=(\d+), bd_id=(\d+), tile=\((\d+),(\d+)\), direction=(MM2S|S2MM) \*/")


def bd_run_b(bd):
    """contiguous byte run of one BD: plain BD = len; multidim: d0 is contiguous when d0_stride == 4 (32-bit
    element), then d1.. merge while d<i>_stride == bytes covered so far; the run never exceeds len"""
    nd = bd.get("num_dims", 0)
    if not nd:
        return bd["len"]
    if bd["d0_stride"] != 4:
        return 4
    run = 4 * bd["d0_wrap"]
    for i in range(1, nd):
        if bd.get("d%d_stride" % i) != run:
            break
        run *= bd["d%d_wrap" % i]
    return min(run, bd["len"])


def shim_bd_shapes(host_cc, fn, txt=None):
    """shim (row 0) BDs of host function fn, one dict per BD comment of the codegen
    ('/* DMA BD Config: bd_id=.., len=<bytes>, .., num_dims=.., d<i>_stride=<bytes>, d<i>_wrap=.. */', present in
    the off and the aot/aot-kco host.cc alike: aot drops the calls but keeps the comments). Each BD belongs to the
    next '/* Create IO: .. tile=(c,r), direction=.. */' (a chained / out-of-order BD group is created once, on its
    first BD). Adds tile, dir, mult (executions per launch, _walk) and run_b (bd_run_b).
    Raises on a BD comment with no following Create IO."""
    out, pend = [], []
    for line, _code, mult, _locs in _walk(host_cc, fn, txt):
        bm = _BDC.search(line)
        if bm:
            d = {}
            for kv in bm.group(1).split(", "):
                k, _, v = kv.partition("=")
                d[k] = v if v in ("true", "false") else int(v)
            d["mult"] = mult
            pend.append(d)
            continue
        cm = _CIO.search(line)
        if cm:
            col, row, dr = int(cm.group(3)), int(cm.group(4)), cm.group(5)
            for d in pend:
                if row == 0:
                    d.update(tile=(col, row), dir=dr, run_b=bd_run_b(d))
                    out.append(d)
            pend = []
    if pend:
        raise ValueError("%s: %d BD comments without a following Create IO" % (fn, len(pend)))
    return out


def window_cp_words(ctrlpkt_h, return_route=False):
    """cp words of one shim-BD window: segment words minus the leading return-route words
    (return_route=True: the return-route words alone)."""
    txt = open(ctrlpkt_h).read()
    segs = re.findall(r"segment, stream id 0x[0-9a-f]+, (\d+) words \*/\nstatic const uint32_t \w+\[\] = \{(.*?)\n\};",
                      txt, re.S)
    if not segs:
        raise ValueError("no shim-BD segments in %s" % ctrlpkt_h)
    total = rr_total = 0
    for n, body in segs:
        seg_w = rr_w = 0
        for blk in re.split(r"\n\s*\n", body):
            w = len(re.findall(r"0x[0-9a-fA-F]+u", re.sub(r"/\*.*?\*/", "", blk, flags=re.S)))
            seg_w += w
            if "return route" in blk:
                rr_w += w
        if seg_w != int(n):
            raise ValueError("segment word count %d != header %s in %s" % (seg_w, n, ctrlpkt_h))
        total += seg_w - rr_w
        rr_total += rr_w
    return rr_total if return_route else total


def static_feat(kdir):
    wl = os.path.join(kdir, "aout", "worklocal")
    return feat_of_counts(host_counts(os.path.join(wl, "host.cc")), os.path.join(wl, "host_ctrlpkt.h"))


def feat_of_counts(c, ctrlpkt_h):
    """single-launch features of one kernel from its host function's executed call counts (P1 formulas)."""
    tile_bd = c.get("__Runtime_dma_bd_config", 0) - c.get(SHIM_BD, 0)
    bd = c.get("__Runtime_dma_bd_config", 0) + c.get("__Runtime_dma_bd_config_multidim", 0) + c.get("__Runtime_dma_bd_config_multidim_ooo", 0)
    # rt_sb_commit only kicks while a window is active (set by __Runtime_ctrl_shim_bd_begin, cleared by the commit):
    # a begin before the round loop + a commit inside it sends ONE window per launch
    commits = min(c.get("__Runtime_ctrl_shim_bd_begin", 0), c.get("__Runtime_ctrl_shim_bd_commit", 0))
    windows = c.get("__Runtime_ctrl_aot_window", 0)
    kicks = commits + windows
    cp_win = window_cp_words(ctrlpkt_h) if kicks else 0
    cp_w = kicks * cp_win
    return {
        "bdcfg_calls": float(bd + COMMIT_PHASES * commits + WINDOW_PHASES * windows),
        "startio_calls": float(c.get("__Runtime_startio", 0)),
        "bdcfg_w": TILE_BD_WORDS * tile_bd + cp_w + SB_KICK_MMIO * kicks,
        "cp_w": cp_w,
        "kicks": kicks,
    }


UNET_TOT = ("bdcfg_calls", "startio_calls", "bdcfg_w", "cp_w")


def window_rr_tiles(ctrlpkt_h):
    """{shim tile (col, row): return-route words of its segment} of one shim-BD window (window_cp_words split by tile)."""
    txt = open(ctrlpkt_h).read()
    out = {}
    for c, r, body in re.findall(r"/\* shim \((\d+),(\d+)\) segment, stream id 0x[0-9a-f]+, \d+ words \*/\n"
                                 r"static const uint32_t \w+\[\] = \{(.*?)\n\};", txt, re.S):
        rr_w = 0
        for blk in re.split(r"\n\s*\n", body):
            if "return route" in blk:
                rr_w += len(re.findall(r"0x[0-9a-fA-F]+u", re.sub(r"/\*.*?\*/", "", blk, flags=re.S)))
        out[(int(c), int(r))] = out.get((int(c), int(r)), 0) + rr_w
    return out


def unet_site_feat(build):
    """per-launch features of a UNet aiegraph build dir in graph_run launch order (unet_seq): one dict per launch
    with kernel, site_fn, desc, the kernel's single-launch P1 values (bdcfg_calls, startio_calls, bdcfg_w, cp_w,
    kicks, shim_bd, tile_bd), kswitch (kernel != previous launch's), rr (return-route words resent on this
    launch: the kernel's rr only when kswitch and it kicks a window), rr_tiles ({shim tile: rr words}) and prev.
    unet_static_feat = (sum of these + rr into cp_w/bdcfg_w) / launches."""
    wl = os.path.join(build, UNET_WL)
    host = os.path.join(wl, "host.cc")
    txt = open(host).read()
    seq = unet_launches(build)
    if not seq:
        raise ValueError("no launches in %s" % build)
    per = {}
    out = []
    prev = None
    for k, site, desc in seq:
        if k not in per:
            c = host_counts(host, "host_canonicalized_" + k, txt)
            h = os.path.join(wl, "host_ctrlpkt_%s.h" % k)
            f = feat_of_counts(c, h)
            f["rr"] = window_cp_words(h, return_route=True) if f["kicks"] else 0
            f["rr_tiles"] = window_rr_tiles(h) if f["kicks"] else {}
            f["shim_bd"] = c.get(SHIM_BD, 0)
            f["tile_bd"] = c.get("__Runtime_dma_bd_config", 0) - f["shim_bd"]
            # shim multidim BDs (RunOp path, 0 counted Write32 words in bdcfg_w; n_feat charges them for N)
            f["shim_md"] = c.get("__Runtime_dma_bd_config_multidim", 0) + c.get("__Runtime_dma_bd_config_multidim_ooo", 0)
            per[k] = f
        f = per[k]
        sw = k != prev
        d = {x: f[x] for x in UNET_TOT + ("kicks", "shim_bd", "tile_bd", "shim_md")}
        d.update(kernel=k, site_fn=site, desc=desc, kswitch=int(sw), rr=f["rr"] if sw and f["kicks"] else 0,
                 rr_tiles=f["rr_tiles"], prev=prev)
        out.append(d)
        prev = k
    return out


def unet_static_feat(build):
    """per-launch features of a UNet aiegraph build dir: each site's single-launch value (P1 kernel formulas on
    host_canonicalized_<kernel> + host_ctrlpkt_<kernel>.h) summed over the graph_run launch sequence
    (graph_driver.c x aiek_unetconv.cc, unet_seq / maxwell kswitch.py) and divided by the launch count.

    One UNet-only term: rt_sb_begin (aie_runtime.c:6714) re-runs rt_sb_prepare when the control fabric differs
    from the last armed one (g_sb.f != f), i.e. on every kernel switch, and that window resends the shim-row
    return-route prefix that the kernel benchmarks (one kernel, re-armed once) amortise to 0."""
    seq = unet_site_feat(build)
    tot = {x: 0.0 for x in UNET_TOT}
    info = {"launches": len(seq), "kernels": len({d["kernel"] for d in seq}), "switches": 0, "rr_w": 0, "kicks": 0,
            "shim_bd": 0, "tile_bd": 0}
    for d in seq:
        for x in tot:
            tot[x] += d[x]
        if d["rr"]:
            tot["cp_w"] += d["rr"]
            tot["bdcfg_w"] += d["rr"]
            info["rr_w"] += d["rr"]
        info["switches"] += d["kswitch"]
        info["kicks"] += d["kicks"]
        info["shim_bd"] += d["shim_bd"]
        info["tile_bd"] += d["tile_bd"]
    n = float(len(seq))
    return {x: v / n for x, v in tot.items()}, info


def measured(log):
    r = parse(log)
    sd = r["split_detail"]
    n = sd["other"]["launches"]
    words = None
    ccyc = None
    for ln in lines(log):
        if "LAUNCH split_words " in ln:
            words = {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)", ln.split("split_words", 1)[1])}
        if ln.startswith("P0 "):
            m = re.search(r"\bcompute_cyc=(-?\d+)", ln)
            if m:
                ccyc = int(m.group(1))
    if not n or words is None:
        raise ValueError("no launches/split_words in %s" % log)
    return {"bdcfg_calls": sd["calls"]["bdcfg"] / n, "startio_calls": sd["calls"]["startio"] / n,
            "bdcfg_w": words["bdcfg_w"], "cp_w": words["cp_w"], "compute_cyc": ccyc}


def unet_measured(log):
    """UNet board log: split_detail calls / launches of the 'LAUNCH warm|cold ... launches=N' line, split_words."""
    sd = parse(log)["split_detail"]
    n = words = None
    for ln in lines(log):
        lm = re.match(r"LAUNCH (?:warm|cold) .*?\blaunches=(\d+)", ln)
        if lm:
            n = int(lm.group(1))
        if "LAUNCH split_words " in ln:
            words = {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)", ln.split("split_words", 1)[1])}
    if not n or words is None:
        raise ValueError("no LAUNCH warm|cold launches= / split_words in %s" % log)
    return {"bdcfg_calls": sd["calls"]["bdcfg"] / n, "startio_calls": sd["calls"]["startio"] / n,
            "bdcfg_w": words["bdcfg_w"], "cp_w": words["cp_w"], "launches": n}


def rel_of(p, m, word):
    if m == 0 and p == 0:
        return 0.0
    if word and m <= WORD_ABS_TOL:
        return 0.0 if abs(p - m) <= WORD_ABS_TOL else float("inf")
    return abs(p - m) / m if m else float("inf")


def static_ccyc(kdir, kname, params, cyc_by_kernel):
    src = open(os.path.join(kdir, "kacc_gemm.cc")).read()
    m = _TARGS.search(src)
    if not m:
        raise ValueError("no gemm_kacc<...> in %s/kacc_gemm.cc" % kdir)
    ch, kd, px, _kw, _ar, _br, tkw, kch, nkc, nt, s, tr = [int(x) for x in m.group(1).split(",")]
    inst = {"ch": ch, "k": kd, "px": px, "ntile": nt, "S": s, "tkw": tkw, "kch": kch, "nkc": nkc, "tr": tr}
    kc = cyc_by_kernel[kname]
    return des.compute_cycles({"instance": inst}, params, kc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--kernel-logs")
    ap.add_argument("--params", default=str(ROOT / "src/costmodel/params/palmyra213.json"))
    ap.add_argument("--kcal", default=str(ROOT / "src/costmodel/params/palmyra213_kcal.json"))
    ap.add_argument("--cycles", default=CYCLES)
    ap.add_argument("--unet", action="extend", nargs="+", default=[], metavar="cfg=PATH",
                    help="UNet board log of cfg; build = <build-root>/unet_<cfg> (P3)")
    a = ap.parse_args()
    if not a.kernel_logs and not a.unet:
        ap.error("need --kernel-logs and/or --unet")
    rc = 0
    if a.unet:
        rc |= unet_main(a)
    if a.kernel_logs:
        rc |= kernel_main(a)
    return rc


def unet_main(a):
    n = fails = 0
    maxrel = 0.0
    seen = set()
    for spec in a.unet:
        cfg, _, log = spec.partition("=")
        if not log or cfg in seen:
            print("SFEAT cfg=%s target=unet FAIL error=bad or duplicate --unet %s" % (cfg, spec))
            fails += 1
            continue
        seen.add(cfg)
        build = os.path.join(a.build_root, "unet_" + cfg)
        try:
            p, info = unet_static_feat(build)
            m = unet_measured(log)
            mlaunch = m["launches"]
        except Exception as e:  # noqa: BLE001 -- a point that cannot be computed is a FAIL, never skipped
            print("SFEAT cfg=%s target=unet FAIL error=%s" % (cfg, e))
            fails += 1
            continue
        rels = {f: rel_of(p[f], m[f], f.endswith("_w")) for f in ("bdcfg_calls", "startio_calls", "bdcfg_w", "cp_w")}
        r = max(rels.values())
        ok = r <= REL_GATE and info["launches"] == mlaunch == UNET_LAUNCHES
        n += 1
        fails += not ok
        maxrel = max(maxrel, r)
        print("SFEAT_UNET_INFO cfg=%s launches=%d/%d kernels=%d switches=%d tile_bd=%.1f shim_bd=%.1f kicks=%.1f "
              "rr_w=%.1f (per launch)" % (cfg, info["launches"], mlaunch, info["kernels"], info["switches"],
                                          info["tile_bd"] / info["launches"], info["shim_bd"] / info["launches"],
                                          info["kicks"] / info["launches"], info["rr_w"] / info["launches"]))
        print("SFEAT cfg=%s target=unet bdcfg_calls=%.1f/%.1f startio_calls=%.1f/%.1f bdcfg_w=%d/%d cp_w=%d/%d "
              "rel_bdcfg=%.3f rel_startio=%.3f rel_bdcfg_w=%.3f rel_cp_w=%.3f maxrel=%.3f %s"
              % (cfg, p["bdcfg_calls"], m["bdcfg_calls"], p["startio_calls"], m["startio_calls"],
                 round(p["bdcfg_w"]), round(m["bdcfg_w"]), round(p["cp_w"]), round(m["cp_w"]),
                 rels["bdcfg_calls"], rels["startio_calls"], rels["bdcfg_w"], rels["cp_w"], r,
                 "PASS" if ok else "FAIL"))
    allok = fails == 0 and n == N_UNET and maxrel <= REL_GATE
    print("SFEAT ALL src=unet n=%d maxrel=%.3f %s" % (n, maxrel, "PASS" if allok else "FAIL"))
    return 0 if allok else 1


def kernel_main(a):

    params = dict(json.load(open(a.params)))
    params.update({k: v for k, v in json.load(open(a.kcal)).items() if k in des.CAL_DEFAULTS})
    cyc_by_kernel = {k["kernel"]: k for k in json.load(open(a.cycles))["kernels"]}

    logs = sorted(glob.glob(os.path.join(a.kernel_logs, "*", "*.log")))
    n = 0
    fails = 0
    maxrel = 0.0
    cc_n = cc_neg = 0
    cc_max = 0.0
    for log in logs:
        cfg = os.path.basename(os.path.dirname(log))
        k = os.path.basename(log)[:-4]
        kdir = os.path.join(a.build_root, "k_" + cfg, k)
        try:
            p = static_feat(kdir)
            m = measured(log)
        except Exception as e:  # noqa: BLE001 -- a point that cannot be computed is a FAIL, never skipped
            print("SFEAT cfg=%s k=%s FAIL error=%s" % (cfg, k, e))
            fails += 1
            continue
        rels = {f: rel_of(p[f], m[f], f.endswith("_w")) for f in ("bdcfg_calls", "startio_calls", "bdcfg_w", "cp_w")}
        r = max(rels.values())
        ok = r <= REL_GATE
        n += 1
        fails += not ok
        maxrel = max(maxrel, r)
        print("SFEAT cfg=%s k=%s bdcfg_calls=%.1f/%.1f startio_calls=%.1f/%.1f bdcfg_w=%d/%d cp_w=%d/%d "
              "rel_bdcfg=%.3f rel_startio=%.3f rel_bdcfg_w=%.3f rel_cp_w=%.3f maxrel=%.3f %s"
              % (cfg, k, p["bdcfg_calls"], m["bdcfg_calls"], p["startio_calls"], m["startio_calls"],
                 round(p["bdcfg_w"]), round(m["bdcfg_w"]), round(p["cp_w"]), round(m["cp_w"]),
                 rels["bdcfg_calls"], rels["startio_calls"], rels["bdcfg_w"], rels["cp_w"], r,
                 "PASS" if ok else "FAIL"))
        # compute cycles: report only
        try:
            sc = static_ccyc(kdir, k, params, cyc_by_kernel)
            mc = m["compute_cyc"]
            if mc is None:
                raise ValueError("no P0 compute_cyc=")
        except Exception as e:  # noqa: BLE001
            print("SFEAT_CCYC cfg=%s k=%s ERROR %s" % (cfg, k, e))
            continue
        if mc <= 0:
            cc_neg += 1
            print("SFEAT_CCYC cfg=%s k=%s ccyc=%d/%d rel=SKIP_NEG" % (cfg, k, round(sc), mc))
        else:
            cr = abs(sc / mc - 1.0)
            cc_n += 1
            cc_max = max(cc_max, cr)
            print("SFEAT_CCYC cfg=%s k=%s ccyc=%d/%d rel=%.3f" % (cfg, k, round(sc), mc, cr))
    print("SFEAT_CCYC ALL n=%d skipped_neg=%d maxrel=%.3f REPORT" % (cc_n, cc_neg, cc_max))
    allok = fails == 0 and n == N_EXPECT and maxrel <= REL_GATE
    print("SFEAT ALL src=kernel n=%d maxrel=%.3f %s" % (n, maxrel, "PASS" if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
