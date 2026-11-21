#!/usr/bin/env python3
"""k_dmafeat -- M3.1.2.2 (.2a evidence): 5 static DMA physical features per UNet K site vs the dma residual.

Offline only: features come from the build directories (worklocal host.cc shim BD / Create IO comments,
same site table as k_diag KDMAP = m3_site.site_rows + site_feat.bd_shape_rows); the residual comes from the
KSITE lines of a k_diag run (--resid, e.g. arthur's pr_real.out). No board timing is read for the features.

Features per site (launches>0 sites, all launches of the site):
  run_b     min contiguous shim MM2S run (bytes)                     site_feat.bd_shape_rows
  num_dims  max MM2S BD dims                                         site_feat.bd_shape_rows
  burst_b   bytes of one AXI burst per BD = min(run_b, BURST_MAX_B)  site_feat.bd_shape_rows
  shim_ch   concurrently active shim MM2S/S2MM channels, byte-weighted: sum(ch bytes)/max(ch bytes) over the
            row-0 (col, dir, channel_id) Create IO of the site's kernels (shim_ch_n = raw active count; it is 12
            for every UNet kernel, so the raw count carries no information)
  rw_ratio  rd_b / (rd_b + wr_b)  (ddr_in_b / (ddr_in_b + ddr_out_b))
Residual: dma_resid_ms = mean over the KSITE runs of the site of (K_hat - K_meas); dma_ms (model dma term) is
printed alongside.

Output:
  KDMAF cfg=<c> part=<p> kind=<k> site=<s> run_b= num_dims= burst_b= shim_ch= rw_ratio= dma_resid_ms= dma_ms= runs=
  KDMAF CORR cfg=<c> feat=<f> spearman=<x> n=<n>          (per cfg x 5 feats)
  KDMAF ALL sites=<n> feats=5 PASS|FAIL                   (last line; rc=1 on FAIL)
PASS iff every site has bd ok, a residual from >=1 KSITE run, and every spearman is finite.

usage: k_dmafeat.py --build-root DIR --resid KSITE.out off=<log> aot-kco=<log>
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
sys.path.insert(0, str(HERE.parents[1] / "src"))

FEATS = ("run_b", "num_dims", "burst_b", "shim_ch", "rw_ratio")
_KSITE = re.compile(r"^KSITE cfg=(\S+) run=(\S+) (\S+) part=(\d+) .*?K_meas=([-\d.]+) .*?dma=([-\d.]+) .*?"
                    r"K_hat=([-\d.naif]+) ", re.M)


def kernel_shim_ch(build):
    """{kernel: {(col, dir, channel_id): bytes per launch}} of the shim (row 0) Create IO of
    host_canonicalized_<kernel>; bytes = sum(len * mult) of the BD comments preceding each Create IO
    (same pairing rule as static_feat.shim_bd_shapes)"""
    from aiegraph.costmodel.static_feat import UNET_WL, _BDC, _CIO, _walk
    host = os.path.join(build, UNET_WL, "host.cc")
    txt = open(host).read()
    out = {}
    for k in sorted(set(re.findall(r"^void host_canonicalized_(\w+)\(", txt, re.M))):
        by, pend = {}, []
        for line, _c, mult, _l in _walk(host, "host_canonicalized_" + k, txt):
            b = _BDC.search(line)
            if b:
                d = dict(kv.partition("=")[::2] for kv in b.group(1).split(", "))
                pend.append(int(d["len"]) * mult)
                continue
            m = _CIO.search(line)
            if m:
                if int(m.group(4)) == 0:
                    key = (int(m.group(3)), m.group(5), int(m.group(1)))
                    by[key] = by.get(key, 0) + sum(pend)
                pend = []
        out[k] = by
    return out


def shim_ch_eff(chs):
    """concurrently active shim channels of a site, byte-weighted: sum(bytes) / max(channel bytes)
    (= raw count when balanced; a channel that finishes early counts by its share), and raw active count"""
    act = [v for v in chs.values() if v > 0]
    return (sum(act) / max(act) if act else 0.0), len(act)


def site_table(cfg, log, build_root):
    from m3_site import site_rows
    from site_feat import bd_shape_rows
    build = str(Path(build_root) / ("unet_" + cfg))
    rows, _ = site_rows(build, log)
    bds, _ = bd_shape_rows(build, rows, build + ".build.log")
    bd = {(f["kind"], f["cid"]): f for f in bds}
    kch = kernel_shim_ch(build)
    out = []
    for r in rows:
        f = bd.get((r["kind"], r["cid"]))
        ks = r.get("kernels") or []
        tot = (f["ddr_in_b"] + f["ddr_out_b"]) if f else 0
        out.append(dict(cfg=cfg, site="%s%d" % (r["kind"], r["cid"]), part=r["part"], kind=r["kind"],
                        ok=bool(f and f["ok"]) and all(k in kch for k in ks) and tot > 0,
                        run_b=f["run_b"] if f else 0, num_dims=f["num_dims"] if f else 0,
                        burst_b=f["burst_b"] if f else 0,
                        shim_ch_b=_site_ch(kch, r, ks),
                        rw_ratio=f["ddr_in_b"] / tot if tot else float("nan")))
        out[-1]["shim_ch"], out[-1]["shim_ch_n"] = shim_ch_eff(out[-1]["shim_ch_b"])
    return out


def _site_ch(kch, r, ks):
    """{channel: bytes per launch} summed over the site's distinct kernels (one launch shape each)"""
    tot = {}
    for k in ks:
        for c, v in kch.get(k, {}).items():
            tot[c] = tot.get(c, 0) + v
    return tot


def load_resid(path):
    """{(cfg, site, part): [(K_hat - K_meas, dma)] over runs}"""
    out = {}
    for cfg, _run, site, part, km, dma, kh in _KSITE.findall(open(path).read()):
        out.setdefault((cfg, site, int(part)), []).append((float(kh) - float(km), float(dma)))
    return out


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(x, y):
    """Spearman rho with average ranks for ties; nan if either side is constant or n<3"""
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--resid", required=True, help="k_diag output with KSITE lines")
    ap.add_argument("runs", nargs="+", help="cfg=log")
    a = ap.parse_args(argv)
    res = load_resid(a.resid)
    ok = True
    tot = 0
    for spec in a.runs:
        cfg, log = spec.split("=", 1)
        sites = site_table(cfg, log, a.build_root)
        used = []
        for s in sites:
            rr = res.get((cfg, s["site"], s["part"]), [])
            s["runs"] = len(rr)
            s["resid"] = sum(x for x, _ in rr) / len(rr) if rr else float("nan")
            s["dma"] = sum(d for _, d in rr) / len(rr) if rr else float("nan")
            good = s["ok"] and rr and math.isfinite(s["resid"])
            ok &= bool(good)
            if good:
                used.append(s)
            print("KDMAF cfg=%s part=%d kind=%s site=%s run_b=%d num_dims=%d burst_b=%d shim_ch=%.3f shim_ch_n=%d rw_ratio=%.4f "
                  "dma_resid_ms=%.4f dma_ms=%.4f runs=%d%s"
                  % (cfg, s["part"], s["kind"], s["site"], s["run_b"], s["num_dims"], s["burst_b"], s["shim_ch"],
                     s["shim_ch_n"], s["rw_ratio"], s["resid"], s["dma"], s["runs"], "" if good else " MISSING"))
        tot += len(sites)
        for f in FEATS:
            rho = spearman([float(s[f]) for s in used], [s["resid"] for s in used])
            ok &= math.isfinite(rho)
            print("KDMAF CORR cfg=%s feat=%s spearman=%.4f n=%d" % (cfg, f, rho, len(used)))
    print("KDMAF ALL sites=%d feats=%d %s" % (tot, len(FEATS), "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
