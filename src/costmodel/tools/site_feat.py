#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Per-site static N-segment features of a UNet build (costmodel M2, N = sum coef x bytes/words + c*launches + d).

  site_feat.py --build <unet build dir> --log <board log> [--build-log <build>.build.log]
               [--pairs kindcid/kindcid,...]  (e.g. conv16/conv70,linear8/linear132)

Sites: n_feat.sites (one aiednn_aie_* call of graph_run, graph_driver.c order). Their launches are taken in order
from static_feat.unet_site_feat (one record per launch, same launch sequence) and summed per site:
  cp_w, bdcfg_w   P1 single-launch words of the kernel, + rr on a kernel-switch launch (as unet_static_feat)
  shim_bd/tile_bd shim-row / tile __Runtime_dma_bd_config calls
  wbytes          K*N*2 per gemm_kacc__N_K_M_... launch (bf16 weights), 0 for fa
  kswitch         launches whose kernel != previous launch's kernel
  rr              return-route words resent (only on kswitch, kernel kicks a shim-BD window)
  rr_pair         on kswitch launches: cur's return-route words on shim tiles that prev's window does not route
                  (window_rr_tiles of cur minus the tiles of prev)
Check (no board number enters the features): sum over sites of cp_w / bdcfg_w vs the log's
'LAUNCH split_words' x 'LAUNCH warm|cold launches='; rel_max over columns with nonzero measured, PASS <= 0.020.

  SITEFEAT cfg=<cfg> run=<a|b> sites=.. launch_sites=.. cp_w=.. split_cp_w=.. bdcfg_w=.. split_bdcfg_w=.. rel_max=.. PASS|FAIL
  SITEPAIR cfg=.. pair=<a>/<b> cp_w=a/b bdcfg_w=a/b wbytes=a/b kswitch=a/b rr=a/b rr_pair=a/b

--bd-shape --cfg <c> (costmodel M3.1 P2): static shim-BD shape and DDR bytes of every launches>0 site
(m3_site.site_rows: 72 linear + 48 conv + 18 fa per cfg), build dir only (the --log only aligns the sites).
Per kernel: static_feat.shim_bd_shapes of host_canonicalized_<kernel> (the codegen's '/* DMA BD Config */' comments,
present in off and aot host.cc alike) x the enclosing loop trips (static_feat._walk, same counting as host_counts);
summed over the site's launches (graph_run launch order, as site_feats):
  bd_n       shim (row 0) BD configs executed
  num_dims   max num_dims of the site's MM2S (DDR read) shim BDs (plain BD = 1)
  run_b      min contiguous byte run of the MM2S shim BDs (static_feat.bd_run_b)
  des_run_b  kch*2 of the site's gemm_kacc instance (des.geometry run_b; kernel-name decode = kcyc_unet.kacc_inst,
             the aie2ps_kernel_cycles instance), '-' for fa
  burst_b    min(run_b, BURST_MAX_B): bytes per shim AXI burst (ASSUMED 256 B max burst, not a board number)
  ddr_in_b   MM2S shim BD bytes (len x executions), ddr_out_b S2MM
Check: kacc sites, ddr_in_b and ddr_out_b vs des.geometry (R x ddr_rd_round_b / ddr_wr_round_b per launch);
kacc_bytes_vs_des_maxrel = max rel over both (GATE_DES = 0.05, ASSUMED). missing = sites without shim BDs or
(kacc) without a decodable instance.
  BDSITE  cfg=<c> <kind><cid> bd_n=.. num_dims=.. run_b=.. des_run_b=.. burst_b=.. ddr_in_b=.. ddr_out_b=..
  BDSHAPE cfg=<c> build=<--build> elf_mtime=<main.elf mtime> sites=.. kacc=.. fa=.. missing=..
          kacc_bytes_vs_des_maxrel=%.3f            (rc=1 unless sites=138 kacc=120 fa=18 missing=0 maxrel<=GATE_DES)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # python -I: no implicit src on sys.path
from costmodel.n_feat import base_sites as n_sites  # noqa: E402
from costmodel.static_feat import unet_site_feat  # noqa: E402
from costmodel.unet_seq import read  # noqa: E402

GATE = 0.020
GATE_DES = 0.05      # ASSUMED; kacc shim-BD bytes vs des.geometry
BURST_MAX_B = 256    # ASSUMED shim AXI max burst
FEATS = ("cp_w", "bdcfg_w", "shim_bd", "tile_bd", "wbytes", "kswitch", "rr", "rr_pair")
_GEMM = re.compile(r"^gemm_kacc__(\d+)_(\d+)_(\d+)_")
_LAUNCH = re.compile(r"^LAUNCH (?:warm|cold) .*?\blaunches=(\d+)", re.M)
_WORDS = re.compile(r"^LAUNCH split_words (.*)$", re.M)
_CTRL = re.compile(r"^LAUNCH (?:warm|cold) .*\bctrlpkt=(\w+)", re.M)


def fmt(v):
    return "%d" % v if float(v).is_integer() else "%.1f" % v


def wbytes(kernel):
    m = _GEMM.match(kernel)
    if not m:
        return 0
    n, k = int(m.group(1)), int(m.group(2))
    return k * n * 2


def site_feats(build, build_log):
    st = n_sites(build, build_log)
    seq = unet_site_feat(build)
    i = 0
    tiles = {}  # kernel -> rr_tiles (window tiles of that kernel)
    for d in seq:
        tiles.setdefault(d["kernel"], d["rr_tiles"])
    for s in st:
        f = dict.fromkeys(FEATS, 0)
        ks = set()
        for d in seq[i:i + s["launches"]]:
            ks.add(d["kernel"])
            f["cp_w"] += d["cp_w"] + d["rr"]
            f["bdcfg_w"] += d["bdcfg_w"] + d["rr"]
            f["shim_bd"] += d["shim_bd"]
            f["tile_bd"] += d["tile_bd"]
            f["wbytes"] += wbytes(d["kernel"])
            f["kswitch"] += d["kswitch"]
            f["rr"] += d["rr"]
            if d["kswitch"]:
                prev_t = set(tiles.get(d["prev"], {})) if d["prev"] else set()
                f["rr_pair"] += sum(w for t, w in d["rr_tiles"].items() if t not in prev_t)
        if len(seq[i:i + s["launches"]]) != s["launches"] or sorted(ks) != s["kernels"]:
            raise ValueError("site %s%d: launches/kernels do not match the launch sequence" % (s["kind"], s["cid"]))
        i += s["launches"]
        s.update(f)
    if i != len(seq):
        raise ValueError("%d site launches vs %d in the launch sequence" % (i, len(seq)))
    return st, len(seq)


def kernel_shapes(build):
    """{kernel: per-launch shim-BD shape} from the build's worklocal host.cc"""
    from costmodel.static_feat import UNET_WL, shim_bd_shapes
    host = os.path.join(build, UNET_WL, "host.cc")
    txt = open(host).read()
    out = {}
    for k in sorted(set(re.findall(r"^void host_canonicalized_(\w+)\(", txt, re.M))):
        bds = shim_bd_shapes(host, "host_canonicalized_" + k, txt)
        rd = [d for d in bds if d["dir"] == "MM2S"]
        out[k] = {"bd_n": sum(d["mult"] for d in bds),
                  "num_dims": max([d.get("num_dims", 1) or 1 for d in rd] or [0]),
                  "run_b": min([d["run_b"] for d in rd] or [0]),
                  "ddr_in_b": sum(d["mult"] * d["len"] for d in rd),
                  "ddr_out_b": sum(d["mult"] * d["len"] for d in bds if d["dir"] == "S2MM")}
    return out


def des_bytes(kernel):
    """(in, out, run_b) per launch of a gemm_kacc kernel from des.geometry, None if not a kacc kernel"""
    from costmodel import des
    from kcyc_unet import kacc_inst
    if not kernel.startswith("gemm_kacc__"):
        return None
    g = des.geometry({"instance": kacc_inst(kernel)})
    return g["R"] * g["ddr_rd_round_b"], g["R"] * g["ddr_wr_round_b"], g["run_b"]


def bd_shape_rows(build, rows, build_log):
    """per-site static shim-BD shape of the launches>0 sites in rows (m3_site.site_rows):
    ([dict(kind, cid, bd_n, num_dims, run_b, des_run_b, burst_b, ddr_in_b, ddr_out_b, ok)], kacc bytes-vs-des rels)"""
    want = {(r["kind"], r["cid"]) for r in rows}
    st = n_sites(build, build_log)
    seq = unet_site_feat(build)
    ks = kernel_shapes(build)
    i = 0
    out = []
    rels = []
    for s in st:
        lk = [d["kernel"] for d in seq[i:i + s["launches"]]]
        i += s["launches"]
        if (s["kind"], s["cid"]) not in want:
            continue
        f = dict(bd_n=0, num_dims=0, run_b=None, ddr_in_b=0, ddr_out_b=0)
        din = dout = 0
        drun = None
        ok = bool(lk)
        for k in lk:
            sh = ks.get(k)
            if not sh or not sh["bd_n"]:
                ok = False
                continue
            for x in ("bd_n", "ddr_in_b", "ddr_out_b"):
                f[x] += sh[x]
            f["num_dims"] = max(f["num_dims"], sh["num_dims"])
            f["run_b"] = sh["run_b"] if f["run_b"] is None else min(f["run_b"], sh["run_b"])
            if s["kind"] != "fa":
                db = des_bytes(k)
                if db is None:
                    ok = False
                    continue
                din += db[0]
                dout += db[1]
                drun = db[2] if drun is None else min(drun, db[2])
        if ok and s["kind"] != "fa":
            rels.append(max(abs(f["ddr_in_b"] - din) / din, abs(f["ddr_out_b"] - dout) / dout))
        f["run_b"] = f["run_b"] or 0
        f["burst_b"] = min(f["run_b"], BURST_MAX_B)
        out.append(dict(f, kind=s["kind"], cid=s["cid"], des_run_b=drun, ok=ok))
    if i != len(seq):
        raise ValueError("%d site launches vs %d in the launch sequence" % (i, len(seq)))
    return out, rels


def bd_shape(build, log, build_log, cfg):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m3_site import site_rows
    rows, _ = site_rows(build, log)
    out, rels = bd_shape_rows(build, rows, build_log)
    missing = sum(not f["ok"] for f in out)
    for f in out:
        print("BDSITE cfg=%s %s%d bd_n=%d num_dims=%d run_b=%d des_run_b=%s burst_b=%d ddr_in_b=%d ddr_out_b=%d"
              % (cfg, f["kind"], f["cid"], f["bd_n"], f["num_dims"], f["run_b"],
                 "-" if f["des_run_b"] is None else "%d" % f["des_run_b"], f["burst_b"], f["ddr_in_b"],
                 f["ddr_out_b"]))
    nk = sum(s["kind"] in ("conv", "linear") for s in out)
    nf = sum(s["kind"] == "fa" for s in out)
    rel = max(rels) if rels else float("inf")
    elf = os.path.join(build, "main.elf")
    mt = int(os.stat(elf).st_mtime) if os.path.exists(elf) else 0
    print("BDSHAPE cfg=%s build=%s elf_mtime=%d sites=%d kacc=%d fa=%d missing=%d kacc_bytes_vs_des_maxrel=%.3f"
          % (cfg, build, mt, len(out), nk, nf, missing, rel))
    return len(out) == 138 and nk == 120 and nf == 18 and missing == 0 and mt > 0 and rel <= GATE_DES


def measured(log):
    txt = read(log)
    lm = _LAUNCH.findall(txt)
    wm = _WORDS.findall(txt)
    if not lm or not wm:
        raise ValueError("no LAUNCH warm|cold launches= / split_words in " + log)
    w = {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)", wm[-1])}
    cm = _CTRL.findall(txt)
    return int(lm[-1]), w, cm[-1] if cm else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--build-log")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--bd-shape", action="store_true", help="static shim-BD shape / DDR bytes per site (M3.1 P2)")
    ap.add_argument("--cfg", help="cfg label (default: build dir name without unet_)")
    a = ap.parse_args(argv)
    build = a.build.rstrip("/")
    cfg = os.path.basename(build)
    cfg = a.cfg or (cfg[5:] if cfg.startswith("unet_") else cfg)
    if a.bd_shape:
        return 0 if bd_shape(build, a.log, a.build_log or build + ".build.log", cfg) else 1
    m = re.search(r"\.([ab])\.log$", a.log)
    run = m.group(1) if m else "-"
    st, nl = site_feats(build, a.build_log or build + ".build.log")
    n, w, _ctrl = measured(a.log)
    ok = n == nl
    rels = []
    out = {}
    for x in ("cp_w", "bdcfg_w"):
        p = float(sum(s[x] for s in st))
        meas = w[x] * n
        out[x] = (p, meas)
        if meas != 0:
            rels.append(abs(p - meas) / meas)
        elif p != 0:
            ok = False
    rel = max(rels) if rels else 0.0
    ok = ok and bool(rels) and rel <= GATE
    for s in st:
        print("SITEFEAT_SITE cfg=%s %s%d launches=%d %s" % (cfg, s["kind"], s["cid"], s["launches"],
                                                           " ".join("%s=%s" % (x, fmt(s[x])) for x in FEATS)))
    print("SITEFEAT cfg=%s run=%s sites=%d launch_sites=%d cp_w=%.1f split_cp_w=%.1f bdcfg_w=%.1f split_bdcfg_w=%.1f "
          "launches=%d/%d rel_max=%.3f %s" % (cfg, run, len(st), sum(1 for s in st if s["launches"]),
                                              out["cp_w"][0], out["cp_w"][1], out["bdcfg_w"][0], out["bdcfg_w"][1],
                                              nl, n, rel, "PASS" if ok else "FAIL"))
    by = {"%s%d" % (s["kind"], s["cid"]): s for s in st}
    for pr in filter(None, a.pairs.split(",")):
        x, _, y = pr.partition("/")
        if x not in by or y not in by:
            print("SITEPAIR cfg=%s pair=%s ERROR unknown site" % (cfg, pr))
            ok = False
            continue
        print("SITEPAIR cfg=%s pair=%s/%s %s" % (cfg, x, y, " ".join(
            "%s=%s/%s" % (f, fmt(by[x][f]), fmt(by[y][f])) for f in ("cp_w", "bdcfg_w", "wbytes", "kswitch", "rr", "rr_pair"))))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
