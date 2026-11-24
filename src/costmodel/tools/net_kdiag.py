#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""net_kdiag -- costmodel M4.1 P4: per-site core- vs DMA-bound from the core counters + static DMA physical
features, for the CNN nets (resnet18; aiek_cnn gemm_kacc instances). Diagnosis only: no fit, no parameters.

Counter evidence (board log, 'PROF aie_site_perf' of run a, k_diag.parse_perf):
  compute = compute_cyc  (= dACTIVE - dLOCK_STALL - dSTREAM_STALL of the launch window, aie_runtime.c:621)
  wait    = lock_stall_cyc + stream_stall_cyc
  bound   = core if compute >= wait else dma
  gemm_kacc has no core-to-core lock or cascade: every lock it stalls on is a DMA window lock (input a/b ports
  filled by tile DMA from shim/memtile, output c drained by tile DMA), so a lock/stream stall is a data-movement
  wait -> the two classes partition the sites. A site without a full counter reading (perf_n < launches or no
  compute_cyc tail) gets bound=none and is in neither class (KDIAG dma_bound + core_bound < sites).
  KDSITE net cfg part kind launches perf_n compute_cyc wait_cyc lock_cyc stream_cyc ratio=compute/wait bound
  KDIAG net=<n> cfg=<c> sites=<log sites> dma_bound=<n> core_bound=<n>

Static DMA physical features (build dir only, never board timing):
  site -> instance: arm_build/graph_driver.c 'aiednn_conv|dense(cid, .., px, kg); /* [part] group ..' gives
  px (pixels per launch) and kg (padded K); network.md L<part> output shape gives N (channels) and M (pixels);
  instance = gemm_kacc__<ceil64(N)>_<kg>_<px>_* in ag_work/aiek/aiek_cnn/worklocal/host.cc;
  launches_static = ceil(M / px) (must equal the log's launches= of the site).
  Per launch, over the shim row-0 'Create IO' channels (col, dir, channel_id) of the instance, each with the
  'DMA BD Config' comments preceding it (k_dmafeat pairing rule; loop trip counts by static_feat._walk):
    run_b_min / run_b_max  min / max contiguous run of a BD (static_feat.bd_run_b, per BD -- no uniform 16 B)
    num_dims               max BD dims (0 = plain BD)
    burst_b                min(run_b_min, BURST_MAX_B=256)  (site_feat.BURST_MAX_B, ASSUMED max AXI burst)
    shim_ch_n              shim channels moving bytes in one launch (concurrently active)
    shim_ch_mm2s/s2mm      of those, read / write channels
    shim_cols              shim columns used
    rw_ratio               rd_b / (rd_b + wr_b)   (MM2S = DDR read, S2MM = DDR write)
    ddr_b, bd_n            bytes / BDs per launch over all shim channels
  KDWSITE net cfg part kind inst launches <features>
  KDWFEAT net=<n> cfg=<c> sites=<log sites> missing=<sites with no instance / launch mismatch / no bytes / nan>
Bandwidth references (not used by any number here): palmyra213 whole-array shim<->DDR R 9.9 GB/s, R+W 7.4 GB/s,
  measured 2026-11-14: amit/tasks/2026-11-14_palmyra-ddr-cap/board_palmyra213/run_ddrbw_0.log:42 (cfg=R4
  B/cyc=9.910) and :50 (cfg=RW4 B/cyc=7.429); README.md:24 table. Printed as KBWREF.

  yolo26n (M4.2): site -> instance from the build log (net_sites.sites: 'aie conv .. -> call P .. -> AxBxC
  launches=L'), instance = gemm_kacc__A_B_C_* in ag_work/aiek/aiek_cnnconv/worklocal/host.cc
  (net_sites.aiek_dir), launches_static = L.

usage: net_kdiag.py --net resnet18|yolo26n --build-root ROOT off=<log_a> aot-kco|aot=<log_a>   (rc 0 iff every cfg has
       dma+core == sites and missing == 0)
       net_kdiag.py --selftest
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import net_sites  # noqa: E402


def cnn_wl(net):
    """worklocal dir of the net's aiek (net_sites.aiek_dir: aiek_cnn for resnet18/unet, aiek_cnnconv for yolo26n)"""
    return os.path.join("ag_work", "aiek", net_sites.aiek_dir(net), "worklocal")


BURST_MAX_B = 256   # = site_feat.BURST_MAX_B (ASSUMED)
BWREF = ("KBWREF R=9.9GBps src=amit/tasks/2026-11-14_palmyra-ddr-cap/board_palmyra213/run_ddrbw_0.log:42 "
         "RW=7.4GBps src=amit/tasks/2026-11-14_palmyra-ddr-cap/board_palmyra213/run_ddrbw_0.log:50 used=0")
_CALL = re.compile(r"^\s*aiednn_(conv|dense)\((\d+), .*, (\d+), (\d+)\);\s*/\* \[(\d+)\] group", re.M)
_SHAPE = re.compile(r'^\s*L(\d+)\["<b>\d+</b> [^<]*<br/>\w+ \[([\d,]+)\]"\]', re.M)
FEATS = ("run_b_min", "run_b_max", "num_dims", "burst_b", "shim_ch_n", "shim_ch_mm2s", "shim_ch_s2mm",
         "shim_cols", "rw_ratio", "ddr_b", "bd_n")


# ------------------------------------------------------------------ counters
def bound_of(p):
    """(bound, compute, wait) of one parse_perf row: core | dma | none (no full counter reading)"""
    if not p or not p.get("has_ovl") or p["perf_n"] < p["launches"] or p["perf_n"] <= 0:
        return "none", 0, 0
    c, w = p["compute_cyc"], p["lock"] + p["stream"]
    return ("core" if c >= w else "dma"), c, w


# ------------------------------------------------------------------ static site table
def ceil64(n):
    return (n + 63) // 64 * 64


def site_inst(build):
    """{part: dict(kind, px, kg, N, M, inst_prefix, launches_static)} from graph_driver.c + network.md"""
    gd = open(os.path.join(build, "ag_work", "arm_build", "graph_driver.c")).read()
    md = open(os.path.join(build, "ag_work", "network.md")).read()
    shp = {int(p): [int(x) for x in s.split(",")] for p, s in _SHAPE.findall(md)}
    out = {}
    for kind, _cid, px, kg, part in _CALL.findall(gd):
        part, px, kg = int(part), int(px), int(kg)
        s = shp.get(part)
        if not s:
            continue
        if kind == "conv" and len(s) == 5:
            n, m = s[1] * s[4], s[2] * s[3]
        elif kind == "dense" and len(s) == 2:
            n, m = s[1], s[0]
        else:
            continue
        out[part] = dict(kind="conv" if kind == "conv" else "linear", px=px, kg=kg, N=n, M=m,
                         inst="gemm_kacc__%d_%d_%d_" % (ceil64(n), kg, px), launches_static=-(-m // px))
    return out


def site_inst_log(build_root, net, cfg):
    """yolo26n: {part: dict(kind, inst_prefix, launches_static)} from the build log's 'aie conv .. -> call P ..
    -> AxBxC launches=L' (net_sites.sites); instance = gemm_kacc__A_B_C_*"""
    out = {}
    for part, kind, _cid, inst, ln in net_sites.sites(build_root, net, cfg):
        out[part] = dict(kind=kind, inst="gemm_kacc__%d_%d_%d_" % tuple(inst), launches_static=ln)
    return out


_HOSTMSG = re.compile(r"(?<=[^\n])\[\d{2}:\d{2}:\d{2} [A-Z]+\] [^\n]*\n")


def unsplice(txt):
    """drop a host-side '[HH:MM:SS LEVEL] ...' message the log capture spliced into the middle of a board line
    (yolo26n_off.a.log:386 'pre_io_act_c[07:35:23 DEBUG] main: program_done ...' + :387 'yc=1414904 ...'):
    rejoins the board line, which parse_perf would otherwise miss (bound=none)"""
    return _HOSTMSG.sub("", txt)


_STARTIO = re.compile(r"__Runtime_startio\(\w+, \w+, -?\d+, (\d+)\)")
_OOO = re.compile(r"Enable out-of-order BD on tile\(\d+,0\)")
_HSTART = re.compile(r"/\* \[call (\d+)\] host\.cc:\d+ __Runtime_startio\(tile\((\d+),(\d+)\) ch (\d+) (MM2S|S2MM), "
                     r"bd \d+, repeat (\d+)\)")


def aot_repeats(wl, kernel):
    """{(col, dir, channel_id): repeat} of the row-0 startio calls the AOT control-packet window folded into
    __Runtime_ioevent_make (aot-kco: host_ctrlpkt_<kernel>.h lists them call by call); {} if no header"""
    p = os.path.join(wl, "host_ctrlpkt_%s.h" % kernel)
    if not os.path.isfile(p):
        return {}
    out = {}
    for _call, col, row, ch, d, rep in set(_HSTART.findall(open(p).read())):
        if int(row) == 0:
            key = (int(col), d, int(ch))
            if out.get(key, int(rep)) != int(rep):
                raise ValueError("%s: %s has repeats %s and %s" % (p, key, out[key], rep))
            out[key] = int(rep)
    return out


def kernel_bds(host):
    """{kernel: {(col, dir, channel_id): dict(bytes, bd_n, run_min, run_max, nd)}} per launch of the shim (row 0)
    Create IO of host_canonicalized_<kernel>: the BD comments preceding each Create IO (k_dmafeat pairing rule)
    form the task, started by __Runtime_startio(dev, io, bd, repeat) (aie_runtime.c:7277; the start queue runs the
    task repeat times) -- the next startio after the Create IO (off), or, when the AOT window folded the startio
    into __Runtime_ioevent_make (aot-kco), the repeat of that (col, dir, ch) in host_ctrlpkt_<kernel>.h.
    bytes = repeat x (chain bytes = sum of the task's BD len; an out-of-order channel ('Enable out-of-order BD')
    moves one BD per task -> max BD len) x loop trips (static_feat._walk); bd_n = repeat x (chain length | 1)
    x trips. A row-0 task with no repeat from either source is dropped (-> no bytes -> missing)."""
    from costmodel.static_feat import _BDC, _CIO, _walk, bd_run_b
    txt = open(host).read()
    wl = os.path.dirname(host)
    out = {}
    for k in sorted(set(re.findall(r"^void host_canonicalized_(\w+)\(", txt, re.M))):
        tasks, pend, cur = [], [], None
        for line, _c, mult, _l in _walk(host, "host_canonicalized_" + k, txt):
            b = _BDC.search(line)
            if b:
                d = {}
                for kv in b.group(1).split(", "):
                    kk, _, v = kv.partition("=")
                    d[kk] = v if v in ("true", "false") else int(v)
                pend.append(d)
                continue
            m = _CIO.search(line)
            if m:
                cur = None
                if int(m.group(4)) == 0 and pend:
                    cur = dict(key=(int(m.group(3)), m.group(5), int(m.group(1))), bds=pend, ooo=False, rep=None,
                               mult=mult)
                    tasks.append(cur)
                pend = []
                continue
            if cur is not None and _OOO.search(line):
                cur["ooo"] = True
                continue
            s = _STARTIO.search(line)
            if s and cur is not None:
                cur["rep"] = int(s.group(1))
                cur = None
        hrep = aot_repeats(wl, k) if any(t["rep"] is None for t in tasks) else {}
        by = {}
        for t in tasks:
            rep = t["rep"] if t["rep"] is not None else hrep.get(t["key"])
            if rep is None:
                continue
            bds = t["bds"]
            task_b = max(d["len"] for d in bds) if t["ooo"] else sum(d["len"] for d in bds)
            task_n = 1 if t["ooo"] else len(bds)
            c = by.setdefault(t["key"], dict(bytes=0, bd_n=0, run_min=0, run_max=0, nd=0))
            runs = [bd_run_b(d) for d in bds]
            c["bytes"] += task_b * rep * t["mult"]
            c["bd_n"] += task_n * rep * t["mult"]
            c["run_min"] = min(runs) if c["run_min"] <= 0 else min(c["run_min"], min(runs))
            c["run_max"] = max([c["run_max"]] + runs)
            c["nd"] = max([c["nd"]] + [d.get("num_dims", 0) for d in bds])
        out[k] = by
    return out


def feats_of(ch):
    act = {k: c for k, c in ch.items() if c["bytes"] > 0}
    rd = sum(c["bytes"] for k, c in act.items() if k[1] == "MM2S")
    wr = sum(c["bytes"] for k, c in act.items() if k[1] == "S2MM")
    rmin = min([c["run_min"] for c in act.values()] or [0])
    return dict(run_b_min=rmin, run_b_max=max([c["run_max"] for c in act.values()] or [0]),
                num_dims=max([c["nd"] for c in act.values()] or [0]), burst_b=min(rmin, BURST_MAX_B),
                shim_ch_n=len(act), shim_ch_mm2s=sum(1 for k in act if k[1] == "MM2S"),
                shim_ch_s2mm=sum(1 for k in act if k[1] == "S2MM"), shim_cols=len({k[0] for k in act}),
                rw_ratio=rd / (rd + wr) if rd + wr else float("nan"), ddr_b=rd + wr,
                bd_n=sum(c["bd_n"] for c in act.values()))


# ------------------------------------------------------------------ per cfg
def run_cfg(net, cfg, log, build_root):
    from k_diag import parse_perf
    txt = unsplice(Path(log).read_text(errors="replace"))
    cen = net_sites.census(log, net)
    perf = parse_perf(txt)
    split = {int(p): (k, int(n)) for p, k, n in net_sites._SPLIT.findall(txt)}
    nb = {"core": 0, "dma": 0, "none": 0}
    for part in sorted(split):
        kind, n = split[part]
        b, c, w = bound_of(perf.get(part))
        nb[b] += 1
        p = perf.get(part) or {}
        print("KDSITE net=%s cfg=%s part=%d kind=%s launches=%d perf_n=%d compute_cyc=%d wait_cyc=%d lock_cyc=%d "
              "stream_cyc=%d ratio=%s bound=%s" % (net, cfg, part, kind, n, p.get("perf_n", 0), c, w,
                                                    p.get("lock", 0), p.get("stream", 0),
                                                    ("%.3f" % (c / w)) if w else "inf", b))
    print("KDIAG net=%s cfg=%s sites=%d dma_bound=%d core_bound=%d" % (net, cfg, len(split), nb["dma"], nb["core"]))
    build = net_sites.build_dir(build_root, net, cfg)
    si = site_inst(build) if net in ("resnet18", "unet") else site_inst_log(build_root, net, cfg)
    kb = kernel_bds(os.path.join(build, cnn_wl(net), "host.cc"))
    miss = 0
    for part in sorted(split):
        kind, n = split[part]
        s = si.get(part)
        ks = [k for k in kb if s and k.startswith(s["inst"])]
        why = ""
        if not s:
            why = "no_call"
        elif len(ks) != 1:
            why = "inst_n=%d" % len(ks)
        elif s["launches_static"] != n:
            why = "launches_static=%d" % s["launches_static"]
        f = feats_of(kb[ks[0]]) if not why else {}
        if not why and (f["ddr_b"] <= 0 or not all(math.isfinite(float(f[x])) for x in FEATS)):
            why = "no_bytes"
        if why:
            miss += 1
            print("KDWSITE net=%s cfg=%s part=%d kind=%s launches=%d MISSING %s" % (net, cfg, part, kind, n, why))
            continue
        print("KDWSITE net=%s cfg=%s part=%d kind=%s inst=%s launches=%d %s" % (
            net, cfg, part, kind, ks[0], n,
            " ".join("%s=%s" % (x, ("%.4f" % f[x]) if isinstance(f[x], float) else f[x]) for x in FEATS)))
    print("KDWFEAT net=%s cfg=%s sites=%d missing=%d" % (net, cfg, len(split), miss))
    ok = cen["sum_ok"] == 1 and nb["dma"] + nb["core"] == len(split) and len(split) > 0 and miss == 0
    return ok


def selftest():
    n = bad = 0

    def chk(name, cond):
        nonlocal n, bad
        n += 1
        bad += 0 if cond else 1
        print("NETKDIAG SELFTEST %-16s %s" % (name, "ok" if cond else "FAIL"))

    base = dict(has_ovl=True, perf_n=4, launches=4, stream=0)
    chk("core", bound_of(dict(base, compute_cyc=10, lock=5))[0] == "core")
    chk("dma", bound_of(dict(base, compute_cyc=10, lock=11))[0] == "dma")
    chk("short", bound_of(dict(base, perf_n=3, compute_cyc=10, lock=1))[0] == "none")
    chk("no_ovl", bound_of(dict(base, has_ovl=False, compute_cyc=10, lock=1))[0] == "none")
    chk("missing_row", bound_of(None)[0] == "none")
    f = feats_of({(0, "MM2S", 0): dict(bytes=300, bd_n=2, run_min=64, run_max=4096, nd=3),
                  (1, "S2MM", 0): dict(bytes=100, bd_n=1, run_min=1024, run_max=1024, nd=0),
                  (2, "MM2S", 1): dict(bytes=0, bd_n=0, run_min=0, run_max=0, nd=0)})
    chk("feats", f["rw_ratio"] == 0.75 and f["shim_ch_n"] == 2 and f["run_b_min"] == 64 and f["burst_b"] == 64
        and f["num_dims"] == 3 and f["shim_cols"] == 2 and f["ddr_b"] == 400 and f["run_b_max"] == 4096)
    chk("unsplice", unsplice("a=1 b_c[07:35:23 DEBUG] main: x\nyc=2\n[07:35:24 DEBUG] keep\n")
        == "a=1 b_cyc=2\n[07:35:24 DEBUG] keep\n")
    chk("ceil64", ceil64(1000) == 1024 and ceil64(64) == 64)
    print("NETKDIAG SELFTEST %s (%d/%d)" % ("PASS" if bad == 0 else "FAIL", n - bad, n))
    return bad == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--net", default="resnet18", choices=net_sites.NETS)
    ap.add_argument("--build-root")
    ap.add_argument("runs", nargs="*", help="cfg=log (run a)")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    if not a.build_root or not a.runs:
        ap.error("need --build-root and cfg=log")
    print(BWREF)
    ok = True
    for r in a.runs:
        cfg, _, log = r.partition("=")
        ok = run_cfg(a.net, cfg, log, a.build_root) and ok
    print("NETKDIAG ALL %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
