#!/usr/bin/env python3
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
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # python -I: no implicit src on sys.path
from aiegraph.costmodel.n_feat import sites as n_sites  # noqa: E402
from aiegraph.costmodel.static_feat import unet_site_feat  # noqa: E402
from aiegraph.costmodel.unet_seq import read  # noqa: E402

GATE = 0.020
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
    a = ap.parse_args(argv)
    build = a.build.rstrip("/")
    cfg = os.path.basename(build)
    cfg = cfg[5:] if cfg.startswith("unet_") else cfg
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
