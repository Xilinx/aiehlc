#!/usr/bin/env python3
"""Static L-segment features per launch from a kernel BUILD directory only (M1.2.2 P1).

  static_feat.py --build-root <root with k_<cfg>/<kernel>/> --kernel-logs <dir with <cfg>/<kernel>.log>
                 [--params script/perf/params/palmyra213.json] [--kcal script/perf/params/palmyra213_kcal.json]
                 [--cycles <aie2ps_kernel_cycles.json>] [--unet cfg=PATH ...]

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
Any missing build/log or parse failure prints FAIL for the point; the ALL line needs n=27 and maxrel <= 0.05.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aiegraph.costmodel import des  # noqa: E402
from aiegraph.costmodel.launch_split import lines, parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CYCLES = "/scratch/staff/huaj/aiengineers/maxwell/tasks/2026-11-19_kernel-cycles/aie2ps_kernel_cycles.json"
TILE_BD_WORDS = 6        # AIE2PS tile DMA BD: 6 x Write32
SB_KICK_MMIO = 6         # 4 XAie_DmaUpdateBdLen + 2 XAie_DmaChannelPushBdToQueue per armed shim-window kick
COMMIT_PHASES = 2        # rt_sb_kick + rt_sb_ack_wait
WINDOW_PHASES = 3        # __Runtime_ctrl_aot_window + rt_sb_kick + rt_sb_ack_wait
WORD_ABS_TOL = 16
REL_GATE = 0.05
N_EXPECT = 27

_FOR = re.compile(r"for \(size_t (v\d+) = (\w+); \1 < (\w+); \1 \+= (\w+)\)")
_CONST = re.compile(r"^\s*size_t (v\d+) = (\d+);")
_CALL = re.compile(r"\b(__Runtime_[A-Za-z0-9_]+)\(")
_TARGS = re.compile(r"gemm_kacc<\s*(\d+(?:\s*,\s*\d+){11})\s*>")


def host_counts(host_cc):
    """{runtime API name: executed calls per launch} of host_canonicalized."""
    txt = open(host_cc).read()
    m = re.search(r"^void host_canonicalized\(.*?\{\n(.*?)^\}", txt, re.S | re.M)
    if not m:
        raise ValueError("no host_canonicalized in %s" % host_cc)
    consts, counts = {}, {}
    stack = []  # [(depth at loop open, trip)]
    depth = 0
    for line in m.group(1).split("\n"):
        code = re.sub(r"/\*.*?\*/", "", line)
        cm = _CONST.match(code)
        if cm:
            consts[cm.group(1)] = int(cm.group(2))
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
        for c in _CALL.findall(code):
            counts[c] = counts.get(c, 0) + mult
        depth += code.count("{") - code.count("}")
        while stack and depth <= stack[-1][0]:
            stack.pop()
    return counts


def window_cp_words(ctrlpkt_h):
    """cp words of one shim-BD window: segment words minus the leading return-route words."""
    txt = open(ctrlpkt_h).read()
    segs = re.findall(r"segment, stream id 0x[0-9a-f]+, (\d+) words \*/\nstatic const uint32_t \w+\[\] = \{(.*?)\n\};",
                      txt, re.S)
    if not segs:
        raise ValueError("no shim-BD segments in %s" % ctrlpkt_h)
    total = 0
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
    return total


def static_feat(kdir):
    wl = os.path.join(kdir, "aout", "worklocal")
    c = host_counts(os.path.join(wl, "host.cc"))
    tile_bd = c.get("__Runtime_dma_bd_config", 0)
    bd = tile_bd + c.get("__Runtime_dma_bd_config_multidim", 0) + c.get("__Runtime_dma_bd_config_multidim_ooo", 0)
    # rt_sb_commit only kicks while a window is active (set by __Runtime_ctrl_shim_bd_begin, cleared by the commit):
    # a begin before the round loop + a commit inside it sends ONE window per launch
    commits = min(c.get("__Runtime_ctrl_shim_bd_begin", 0), c.get("__Runtime_ctrl_shim_bd_commit", 0))
    windows = c.get("__Runtime_ctrl_aot_window", 0)
    kicks = commits + windows
    cp_win = window_cp_words(os.path.join(wl, "host_ctrlpkt.h")) if kicks else 0
    cp_w = kicks * cp_win
    return {
        "bdcfg_calls": float(bd + COMMIT_PHASES * commits + WINDOW_PHASES * windows),
        "startio_calls": float(c.get("__Runtime_startio", 0)),
        "bdcfg_w": TILE_BD_WORDS * tile_bd + cp_w + SB_KICK_MMIO * kicks,
        "cp_w": cp_w,
    }


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
    ap.add_argument("--kernel-logs", required=True)
    ap.add_argument("--params", default=str(ROOT / "script/perf/params/palmyra213.json"))
    ap.add_argument("--kcal", default=str(ROOT / "script/perf/params/palmyra213_kcal.json"))
    ap.add_argument("--cycles", default=CYCLES)
    ap.add_argument("--unet", action="append", default=[], metavar="cfg=PATH",
                    help="reserved for P3 (UNet ELF builds)")
    a = ap.parse_args()

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
    for u in a.unet:
        print("SFEAT_UNET %s reserved (P3), not evaluated" % u)
    print("SFEAT_CCYC ALL n=%d skipped_neg=%d maxrel=%.3f REPORT" % (cc_n, cc_neg, cc_max))
    allok = fails == 0 and n == N_EXPECT and maxrel <= REL_GATE
    print("SFEAT ALL src=kernel n=%d maxrel=%.3f %s" % (n, maxrel, "PASS" if allok else "FAIL"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
