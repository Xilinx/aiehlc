#!/usr/bin/env python3
"""k_dmafeat -- M3.1.2.2/M3.1.2.3: 7 static DMA physical features per UNet K site vs the dma residual.

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
  bd_n      shim BDs per launch = sum over the site's launches of kernel bd_n / launches   site_feat.bd_shape_rows
  ddr_b     DDR bytes per launch = (ddr_in_b + ddr_out_b) / launches                      site_feat.bd_shape_rows
Bandwidth references (notes only, not parameters): R+W 7.4 GB/s and read-only 9.9 GB/s are ASSUMED (not re-measured
here; no log line cited).
Public API: site_dma_feats(cfg, log, build_root) -> {part: {feat: value}} (build dir + the log's site table only,
no board timing; used by k_diag --dmafeat).
Residual: dma_resid_ms = mean over the KSITE runs of the site of (K_hat - K_meas); dma_ms (model dma term) is
printed alongside.

Output:
  KDMAF cfg=<c> part=<p> kind=<k> site=<s> run_b= num_dims= burst_b= shim_ch= rw_ratio= dma_resid_ms= dma_ms= runs=
  KDMAF CORR cfg=<c> feat=<f> spearman=<x> n=<n>          (per cfg x 7 feats)
  KDMAF CORRCLS cfg=<c> cls=<kind> feat=<f> n=<n> spearman=<x>   (per cfg x class with n>=3 x 7 feats; within-class
            rank correlation: a feature whose CORR sign is only a class proxy shows ~0 / flipped / nan (constant) here)
  KDMAF ALL sites=<n> feats=7 PASS|FAIL                   (last line; rc=1 on FAIL)
PASS iff every site has bd ok, a residual from >=1 KSITE run, and every (pooled) spearman is finite.
CORRCLS nan (feature constant inside a class) is reported, not a failure.

usage: k_dmafeat.py --build-root DIR --resid KSITE.out off=<log> aot-kco=<log>
       k_dmafeat.py --selftest      (last line: KDMAF SELFTEST PASS (k/n))
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

FEATS = ("run_b", "num_dims", "burst_b", "shim_ch", "rw_ratio", "bd_n", "ddr_b")
CLS_MIN_N = 3
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
        nl = int(r.get("launches") or 0)
        out.append(dict(cfg=cfg, site="%s%d" % (r["kind"], r["cid"]), part=r["part"], kind=r["kind"],
                        ok=bool(f and f["ok"]) and all(k in kch for k in ks) and tot > 0,
                        run_b=f["run_b"] if f else 0, num_dims=f["num_dims"] if f else 0,
                        burst_b=f["burst_b"] if f else 0,
                        shim_ch_b=_site_ch(kch, r, ks),
                        rw_ratio=f["ddr_in_b"] / tot if tot else float("nan"),
                        bd_n=per_launch(f["bd_n"] if f else 0, nl), ddr_b=per_launch(tot, nl)))
        out[-1]["ok"] &= nl > 0
        out[-1]["shim_ch"], out[-1]["shim_ch_n"] = shim_ch_eff(out[-1]["shim_ch_b"])
    return out


def per_launch(total, launches):
    """site total over its launches -> per launch (nan if no launch)"""
    return total / launches if launches > 0 else float("nan")


def site_dma_feats(cfg, log, build_root):
    """{part: {feat: value}} of the 7 FEATS (+ ok) for the launches>0 sites of build_root/unet_<cfg>; reads the
    build dir (and the log's site/launch table) only, never board timing"""
    return {s["part"]: dict({f: s[f] for f in FEATS}, ok=s["ok"], kind=s["kind"], site=s["site"])
            for s in site_table(cfg, log, build_root)}


def corr_by_cls(used, feats=FEATS, min_n=CLS_MIN_N):
    """[(cls, feat, n, spearman)] within each kind with n >= min_n"""
    by = {}
    for s in used:
        by.setdefault(s["kind"], []).append(s)
    out = []
    for c in sorted(by):
        ss = by[c]
        if len(ss) < min_n:
            continue
        for f in feats:
            out.append((c, f, len(ss), spearman([float(s[f]) for s in ss], [s["resid"] for s in ss])))
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


def selftest():
    cases = []

    def chk(name, cond):
        cases.append(bool(cond))
        print("KDMAF SELFTEST %s %s" % (name, "ok" if cond else "FAIL"))

    chk("ranks_ties", ranks([3, 1, 3, 2]) == [3.5, 1.0, 3.5, 2.0])
    chk("spearman_mono", abs(spearman([1, 2, 3, 4], [10, 20, 25, 99]) - 1.0) < 1e-12)
    chk("spearman_anti", abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12)
    chk("spearman_const_nan", math.isnan(spearman([1, 1, 1], [1, 2, 3])))
    chk("spearman_short_nan", math.isnan(spearman([1, 2], [1, 2])))
    chk("shim_ch_eff", shim_ch_eff({"a": 100, "b": 50, "c": 0}) == (1.5, 2))
    chk("per_launch", per_launch(120, 4) == 30 and math.isnan(per_launch(5, 0)))
    # class proxy: pooled rho = +1 but inside each class the feature runs against the residual
    used = [dict(kind="a", x=x, resid=r) for x, r in ((1, 3), (2, 2), (3, 1))] + \
           [dict(kind="b", x=x, resid=r) for x, r in ((10, 13), (11, 12), (12, 11))] + \
           [dict(kind="c", x=0, resid=0), dict(kind="c", x=1, resid=1)]
    pooled = spearman([s["x"] for s in used], [s["resid"] for s in used])
    cc = corr_by_cls(used, feats=("x",))
    chk("corrcls_proxy", pooled > 0.5 and [(c, n) for c, _f, n, _r in cc] == [("a", 3), ("b", 3)]
        and all(abs(r + 1.0) < 1e-12 for _c, _f, _n, r in cc))
    chk("feats7", len(FEATS) == 7 and FEATS[-2:] == ("bd_n", "ddr_b"))
    k = sum(cases)
    print("KDMAF SELFTEST %s (%d/%d)" % ("PASS" if k == len(cases) else "FAIL", k, len(cases)))
    return 0 if k == len(cases) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build-root")
    ap.add_argument("--resid", help="k_diag output with KSITE lines")
    ap.add_argument("runs", nargs="*", help="cfg=log")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.build_root and a.resid and a.runs):
        ap.error("--build-root, --resid and cfg=log runs are required")
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
                  "bd_n=%.3f ddr_b=%.1f dma_resid_ms=%.4f dma_ms=%.4f runs=%d%s"
                  % (cfg, s["part"], s["kind"], s["site"], s["run_b"], s["num_dims"], s["burst_b"], s["shim_ch"],
                     s["shim_ch_n"], s["rw_ratio"], s["bd_n"], s["ddr_b"], s["resid"], s["dma"], s["runs"],
                     "" if good else " MISSING"))
        tot += len(sites)
        for f in FEATS:
            rho = spearman([float(s[f]) for s in used], [s["resid"] for s in used])
            ok &= math.isfinite(rho)
            print("KDMAF CORR cfg=%s feat=%s spearman=%.4f n=%d" % (cfg, f, rho, len(used)))
        for c, f, n, rho in corr_by_cls(used):
            print("KDMAF CORRCLS cfg=%s cls=%s feat=%s n=%d spearman=%.4f" % (cfg, c, f, n, rho))
    print("KDMAF ALL sites=%d feats=%d %s" % (tot, len(FEATS), "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
