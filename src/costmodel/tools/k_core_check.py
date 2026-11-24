#!/usr/bin/env python3
"""UNet static core-cycle estimate vs measured per-site compute_cyc (costmodel M3.1.2.3 QM probe).

  k_core_check.py --build-root DIR [--iss JSON] [--max-rel X] CFG=LOG_A,LOG_B[,LOG_C] ...

Per cfg the static estimate comes from kcyc_unet.py run on DIR/unet_<cfg> (its own DIR/unet_<cfg>.build.log):
  conv/linear: kcyc_unet src iss | iss_unet | formula (gemm_kacc instance from kcyc_unet.kacc_inst)
  fa:          kcyc_unet fa: src iss (RF0 ISS of the build's own FA TU, md5-checked) | formula_noloo
  pred_cyc(site) = core_cyc (one core, one launch) * launches
Measured (per log): 'PROF aie_site_perf part=.. compute_cyc=N' (k_diag._PERF; sum over perf_n launches), joined
  to the site by k_diag.load (m3_site rows, (kind, cid)); meas_cyc(site) = median over the cfg's logs of
  compute_cyc / perf_n * launches (perf_n == launches on a full reading). rel = |pred - meas| / meas.
Nothing is fitted to the logged compute_cyc: the prediction is the ISS (or kacc formula) value x launches.
--max-rel X: the ALL line FAILs (rc=1) when the max per-site rel over all cfgs exceeds X (default: no limit).
Output:
  KCORE SRC cfg=<c> log=<path> board=<b> launches=<n> run_rc=<rc> sites=<n> ovl_sites=<n>   (one per log)
  KCORE_SITE cfg site cls src launches pred_cyc meas_cyc rel
  KCORE cfg=<c> cls=<cls> sites=<n> meas=<n> pred_Mcyc=<x> meas_Mcyc=<x> sum_ratio=<x> maxrel=<x> med_rel=<x>
        src=<iss|formula|mixed>   (cls=ALL last per cfg; iss = every site iss/iss_unet, formula = none, else mixed)
  KCORE ALL sites=<n> cover=<n> finite=<0|1> maxrel=<x> med_rel=<x> worst_cls=<cfg>:<cls> fa_src=<iss|formula|mixed>
        PASS|FAIL   (last line; fa_src over the FA sites of every cfg)
PASS iff every site of every cfg has a finite prediction (cover == sites == 138 * cfgs) and there are 6 SRC lines,
each board=palmyra213 launches=258 run_rc=0 sites=138, kcyc_unet rc=0 for every cfg, fa_src=iss, and
maxrel <= --max-rel (when given). FAIL -> rc=1.
"""
import argparse
import contextlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

ISS = "/scratch/staff/huaj/aiengineers/maxwell/tasks/2026-11-19_kernel-cycles/aie2ps_kernel_cycles.json"
SITES_PER_CFG = 138
N_LOGS = 6
BOARD, LAUNCHES = "palmyra213", "258"


def kcyc(build, iss):
    """kcyc_unet on one build -> (rc, {(kind, cid): site rec})"""
    import kcyc_unet
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "kcyc.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = kcyc_unet.main(["--build-root", build, "--iss", iss, "--out", out])
        doc = json.load(open(out)) if os.path.exists(out) else dict(sites=[])
    for ln in buf.getvalue().splitlines():
        if ln.startswith(("KCYC ", "KCYC_SUM", "KCYC_ERR")):
            print("KCORE_KCYC " + ln)
    return rc, {(r["kind"], r["cid"]): r for r in doc["sites"]}


def src_cls(srcs):
    """site srcs -> iss | formula | mixed (iss_unet counts as iss, formula_noloo as formula)"""
    k = {"iss" if x in ("iss", "iss_unet") else "formula" for x in srcs}
    return k.pop() if len(k) == 1 else "mixed"


def fnum(x):
    return "%.4f" % x if isinstance(x, float) and math.isfinite(x) else "nan"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--iss", default=ISS)
    ap.add_argument("--max-rel", type=float, default=None, help="FAIL when maxrel over all sites exceeds this")
    ap.add_argument("cfgs", nargs="+", help="CFG=LOG_A,LOG_B[,LOG_C]")
    a = ap.parse_args(argv)
    from k_diag import _CMP, load
    ok = True
    n_src = n_sites = cover = 0
    finite = True
    all_rel, cls_stat, fa_srcs = [], {}, []
    for spec in a.cfgs:
        cfg, logs = spec.split("=", 1)
        logs = logs.split(",")
        rc, pred = kcyc(str(Path(a.build_root) / ("unet_" + cfg)), a.iss)
        if rc != 0:
            print("KCORE_ERR cfg=%s kcyc_unet rc=%d" % (cfg, rc))
            ok = False
        runs = []
        for lg in logs:
            ss, tx = load(cfg, lg, a.build_root)
            m = _CMP.findall(tx)
            b, n, rr = m[-1] if m else ("?", "0", "-1")
            print("KCORE SRC cfg=%s log=%s board=%s launches=%s run_rc=%s sites=%d ovl_sites=%d"
                  % (cfg, lg, b, n, rr, len(ss), sum(bool(s["perf"] and s["perf"].get("has_ovl")) for s in ss)))
            n_src += 1
            if (b, n, rr, len(ss)) != (BOARD, LAUNCHES, "0", SITES_PER_CFG):
                ok = False
            runs.append({(s["kind"], int(s["site"][len(s["kind"]):])): s for s in ss})
        keys = list(runs[0])
        if any(set(r) != set(keys) for r in runs):
            print("KCORE_ERR cfg=%s site sets differ between logs" % cfg)
            ok = False
        per_cls = {}
        for key in keys:
            s0 = runs[0][key]
            n_sites += 1
            r = pred.get(key)
            pc = float(r["core_cyc"]) * s0["launches"] if r else float("nan")
            if r:
                cover += 1
            if not math.isfinite(pc) or pc <= 0:
                finite = False
            ms = [rr[key]["perf"]["compute_cyc"] / rr[key]["perf"]["perf_n"] * rr[key]["launches"]
                  for rr in runs if key in rr and rr[key]["perf"] and rr[key]["perf"].get("has_ovl")
                  and rr[key]["perf"]["perf_n"] > 0]
            mc = float(statistics.median(ms)) if ms else float("nan")
            rel = abs(pc - mc) / mc if ms and mc > 0 and math.isfinite(pc) else float("nan")
            print("KCORE_SITE cfg=%s site=%s cls=%s src=%s launches=%d pred_cyc=%.0f meas_cyc=%.0f rel=%s"
                  % (cfg, s0["site"], s0["cls"], r["src"] if r else "-", s0["launches"], pc, mc, fnum(rel)))
            src = r["src"] if r else "-"
            if s0["kind"] == "fa":
                fa_srcs.append(src)
            for c in (s0["cls"], "ALL"):
                per_cls.setdefault(c, []).append((pc, mc, rel, src))
        for c in sorted(per_cls, key=lambda c: (c == "ALL", c)):
            v = per_cls[c]
            rl = [x[2] for x in v if math.isfinite(x[2])]
            ps = sum(x[0] for x in v if math.isfinite(x[0]))
            mm = sum(x[1] for x in v if math.isfinite(x[1]))
            mx = max(rl) if rl else float("nan")
            md = statistics.median(rl) if rl else float("nan")
            print("KCORE cfg=%s cls=%s sites=%d meas=%d pred_Mcyc=%.3f meas_Mcyc=%.3f sum_ratio=%s maxrel=%s med_rel=%s src=%s"
                  % (cfg, c, len(v), len(rl), ps / 1e6, mm / 1e6, fnum(ps / mm if mm > 0 else float("nan")),
                     fnum(mx), fnum(md), src_cls(x[3] for x in v)))
            if c != "ALL":
                cls_stat["%s:%s" % (cfg, c)] = mx
                all_rel += rl
    if n_src != N_LOGS or n_sites != SITES_PER_CFG * len(a.cfgs) or cover != n_sites:
        ok = False
    ok = ok and finite
    fa_src = src_cls(fa_srcs) if fa_srcs else "none"
    if fa_src != "iss":
        ok = False
    mx_all = max(all_rel) if all_rel else float("nan")
    if a.max_rel is not None and not (mx_all <= a.max_rel):
        ok = False
    fin = [(v, k) for k, v in cls_stat.items() if math.isfinite(v)]
    worst = max(fin)[1] if fin else "-"
    print("KCORE ALL sites=%d cover=%d finite=%d maxrel=%s med_rel=%s worst_cls=%s fa_src=%s %s"
          % (n_sites, cover, int(finite), fnum(mx_all),
             fnum(statistics.median(all_rel)) if all_rel else "nan", worst, fa_src, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
