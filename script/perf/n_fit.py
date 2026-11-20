#!/usr/bin/env python3
"""Fit / check the N (host glue) segment of UNet AIE sites (costmodel M2).

  n_fit.py --build-root DIR --fit CFG=LOG [...] --out JSON --check CFG=LOG [...]
  n_fit.py --build-root DIR --params JSON --check CFG=LOG [...]          (no fit)
  n_fit.py --build-root DIR --lower-bound --check CFG=LOG [...]          (fit-free lower bound, no fit)
  n_fit.py --build-root DIR --ab CFG=A,B [...] [--ab-ref CFG=A,B [...] --ref-build-root R0]   (run-to-run glue)
  n_fit.py --build-root DIR --floor --fit CFG=LOG [...] --check CFG=LOG [...]   (all-FEATS NNLS floor)
--ab: NSRC + NSUM for every log, then per cfg
  NAB cfg=.. sites=.. glue_ab_max=.. glue_ab_p95=.. ref_p95=.. smaller=0|1   (|glue_a-glue_b|/mean, launches>0)
  --ab-metric abs: NAB cfg=.. metric=abs_ms sites=.. ... (|glue_a-glue_b| in ms, ref_p95 same metric)
  NAB ALL cfgs=.. nsum_pass=.. smaller=.. PASS|FAIL                    (PASS = all NSRC ok and all NSUM PASS)
--floor: NNLS with all FEATS, no subset cap, fitted on the --fit log, max relative error on the --check log:
  NFLOOR cfg=.. kind=.. n=.. floor=..
  NFLOOR ALL cfgs=.. maxfloor=.. PASS|FAIL                             (maxfloor<=0.150)

Per site (n_feat.sites of DIR/unet_<cfg>, aligned with the log's 'PROF part=<i> op=aie_host_*' lines):
  N = (site_ms - aie_ms) + glue_ms + flush_ms + inval_ms
  site_ms - aie_ms is the PROF part 'ms'; glue/flush/inval come from 'PROF aie_host_split part=<i>'.
  aie_ms (in launch: kload, rt_*) belongs to K+L, not N.
Features (all static, n_feat.sites = base sites + static_feat launch features + n_layout layout features):
  FEATS below (= the b6450ac9 list, incl. kfirst); n_feat also yields Mw/Kw/Nw/ucs_b/... which are not used here.
Model per (cfg, kind in conv|linear|fa), sites with launches>0 only:
  N = sum_f coef_f * f + d    (NNLS on relative error, weight 1/N, coef >= 0, d >= 0; <= MAX_FEATS[kind] active features,
                               subset = exhaustive search, see fit_kind)
  NLOO  cfg=.. run=a|b kind=.. loo_maxrel=..                           (LOO with the subset frozen from run a;
                                                                       run b = the --check log of that cfg)
  NLOO  ALL cfgs=.. runs=.. maxloo=.. PASS|FAIL                        (max over cfg x run, <=0.200)
  NTOTAL cfg=.. total_rel=..                                           (printed only, not a gate)
--lower-bound: sites with identical (kind, launches, all FEATS) get the same prediction p from ANY such model, so
per-site maxrel >= (hi-lo)/(hi+lo) over each group; prints the worst group per cfg and the max over cfgs:
  NLB cfg=.. groups=.. top=<kind cid>/<kind cid> maxlb=..
  NLB ALL cfgs=.. maxlb=.. PASS|FAIL                                   (maxlb<=0.150)
Host-only sites (attn roles with 0 launches) are not fitted: their summed fit-run ms is frozen as attn_host_ms
and only enters total_rel.
  NSRC  cfg=.. log=.. board=.. launches=.. run_rc=..                  (from 'CMP unet board=..')
  NSUM  cfg=.. site_glue_ms=.. split_glue_ms=.. rel=.. PASS|FAIL       (sum site glue vs split_detail
                                                                       host_glue_bdcfg*launches, rel<=0.02)
  NFIT  cfg=.. kind=.. n=.. a=.. b=.. c=.. d=..
  NGATE cfg=.. sites=.. r2=.. maxrel=.. total_rel=.. attn_host_ms=.. PASS|FAIL   (maxrel<=0.200)
  NGATE ALL src=board cfgs=.. sites=.. r2_min=.. maxrel=.. PASS|FAIL
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aiegraph.costmodel.n_feat import sites as n_sites  # noqa: E402
from aiegraph.costmodel.unet_seq import read  # noqa: E402

KINDS = ("conv", "linear", "fa")
FEATS = ("MK", "MN", "launches", "wbytes", "cp_w", "bdcfg_w", "rr", "rr_pair", "kswitch", "kfirst",
         "in_pack_bytes", "out_scatter_bytes", "relayout")  # = b6450ac9
MAX_FEATS = {"conv": 7, "linear": 5, "fa": 5}  # active features per kind (fa has n=18; off conv 7 feats -> 0.164)
TOP_LOO = 40   # in-sample best subsets re-ranked by leave-one-out max relative error
GATE = 0.200
LB_GATE = 0.150
FLOOR_GATE = 0.150
SUM_TOL = 0.02

_PROF = re.compile(r"^PROF part=(\d+) op=aie_host_(conv|linear|attn)(\d+) ms=([\d.]+) .*site_ms=([\d.]+) "
                   r"aie_ms=([\d.]+)", re.M)
_SPLIT = re.compile(r"^PROF aie_host_split part=(\d+) kind=(\w+) .*glue_ms=([\d.]+) flush_ms=([\d.]+) "
                    r"inval_ms=([\d.]+) launches=(\d+)$", re.M)
_CMP = re.compile(r"^CMP unet board=(\S+) .*aie_launches=(\d+) .*run_rc=(-?\d+)", re.M)
_DETAIL = re.compile(r"^LAUNCH split_detail .*host_glue_bdcfg=([\d.]+)", re.M)
_LAUNCH = re.compile(r"^LAUNCH (?:warm|cold) .*?\blaunches=(\d+)", re.M)


def sig(x, n=9):
    return float("%.*g" % (n, x))


def derive(r):
    """static features derived from n_feat.sites: krep (launches that rerun the previous launch's kernel),
    in_arm / out_arm (pack / scatter bytes whose producer / reader is an ARM call)"""
    r["krep"] = r["launches"] - r["kswitch"]
    r["in_arm"] = r["in_pack_bytes"] if r["up"] == "arm" else 0
    r["out_arm"] = r["out_scatter_bytes"] if r["down"] == "arm" else 0
    return r


def measure(build, log):
    """sites of build with N measured from log; raises on misalignment"""
    st = n_sites(build, build + ".build.log")
    txt = read(log)
    prof = _PROF.findall(txt)
    split = {int(p): (k, float(g), float(f), float(i), int(n)) for p, k, g, f, i, n in _SPLIT.findall(txt)}
    if len(prof) != len(st):
        raise ValueError("%s: %d PROF aie_host sites vs %d build sites" % (log, len(prof), len(st)))
    out = []
    for s, (part, op, cid, ms, site_ms, aie_ms) in zip(st, prof):
        part, cid, ms, aie_ms = int(part), int(cid), float(ms), float(aie_ms)
        kind_ok = op == s["kind"] or (op == "attn" and s["kind"] in ("fa", "attn"))
        if not kind_ok or cid != s["cid"] or (s["launches"] > 0) != (aie_ms > 0):
            raise ValueError("misaligned site part=%d op=%s%d vs %s cid=%d launches=%d"
                             % (part, op, cid, s["kind"], s["cid"], s["launches"]))
        g = f = i = 0.0
        if part in split:
            k, g, f, i, n = split[part]
            if n != s["launches"]:
                raise ValueError("part=%d split launches=%d vs build %d" % (part, n, s["launches"]))
        elif s["launches"] > 0:
            raise ValueError("part=%d: no aie_host_split line" % part)
        out.append(derive(dict(s, part=part, ms=ms, aie_ms=aie_ms, glue=g, N=ms + g + f + i)))
    return out, txt


def src_line(cfg, log, txt):
    m = _CMP.findall(txt)
    board, launches, rc = m[-1] if m else ("-", "-1", "-1")
    print("NSRC cfg=%s log=%s board=%s launches=%s run_rc=%s" % (cfg, log, board, launches, rc))
    return rc == "0" and int(launches) > 0


def sum_line(cfg, rows, txt):
    site = sum(r["glue"] for r in rows)
    d = _DETAIL.findall(txt)
    lm = _LAUNCH.findall(txt)
    split = float(d[-1]) * int(lm[-1]) if d and lm else float("nan")
    rel = abs(site - split) / split if split > 0 else float("inf")
    ok = rel <= SUM_TOL
    print("NSUM cfg=%s site_glue_ms=%.3f split_glue_ms=%.3f rel=%.3f %s" % (cfg, site, split, rel,
                                                                           "PASS" if ok else "FAIL"))
    return ok


def _nnls(X, y):
    """non-negative least squares on relative error (weight 1/y); X has the intercept column last"""
    sc = np.maximum(np.abs(X).max(axis=0), 1e-300)
    c, _ = nnls(X / sc / y[:, None], np.ones(len(y)))
    return c / sc


def _params(sub, c):
    return {"coef": {f: sig(float(v)) for f, v in zip(sub, c[:-1]) if v > 0}, "d": sig(float(c[-1]))}


def maxrel(p, rows):
    return max(abs(predict(p, r) - r["N"]) / r["N"] for r in rows)


def fit_kind(rows, sub=None, kind="conv"):
    """{'coef': {feat: v>0}, 'd': v>=0} by NNLS on relative error. sub=None: every subset of <= MAX_FEATS
    non-constant FEATS; the TOP_LOO with the smallest in-sample max relative error are re-ranked by
    max(in-sample, leave-one-out) max relative error on the same rows (ties: fewer features, FEATS order);
    'feats' = the chosen subset (frozen for LOO). sub given: refit on that subset."""
    y = np.array([r["N"] for r in rows])
    X = np.array([[float(r[f]) for f in FEATS] + [1.0] for r in rows])
    if sub is not None:
        cols = [FEATS.index(f) for f in sub] + [len(FEATS)]
        return _params(sub, _nnls(X[:, cols], y))
    var = [j for j in range(len(FEATS)) if np.ptp(X[:, j]) > 0]
    cand = []
    for k in range(MAX_FEATS[kind] + 1):
        for cmb in itertools.combinations(var, k):
            cols = list(cmb) + [len(FEATS)]
            c = _nnls(X[:, cols], y)
            cand.append((float(np.max(np.abs(X[:, cols] @ c - y) / y)), k, cmb, c))
    cand.sort(key=lambda t: (t[0], t[1], t[2]))
    best = None
    for m, k, cmb, c in cand[:TOP_LOO]:
        cols = list(cmb) + [len(FEATS)]
        lv = 0.0
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            ci = _nnls(X[keep][:, cols], y[keep])
            lv = max(lv, abs(float(X[i, cols] @ ci) - y[i]) / y[i])
        if best is None or max(lv, m) < best[0] - 1e-12:
            best = (max(lv, m), [FEATS[j] for j in cmb], c)
    p = _params(best[1], best[2])
    p["feats"] = [f for f in best[1] if f in p["coef"]]
    return p


def predict(p, r):
    if "coef" not in p:  # M2-P2 params: N = a*MK + b*MN + c*launches + d
        return p["a"] * r["MK"] + p["b"] * r["MN"] + p["c"] * r["launches"] + p["d"]
    return sum(v * r[f] for f, v in p["coef"].items()) + p["d"]


def lower_bound(cfg, rows):
    """worst identical-feature group of one cfg: (maxlb, groups, lo row, hi row)"""
    g = {}
    for r in rows:
        if r["launches"] > 0:
            g.setdefault((r["kind"], r["launches"]) + tuple(r[f] for f in FEATS), []).append(r)
    best = (0.0, None, None)
    for rs in g.values():
        lo = min(rs, key=lambda r: r["N"])
        hi = max(rs, key=lambda r: r["N"])
        lb = (hi["N"] - lo["N"]) / (hi["N"] + lo["N"])
        if lb > best[0] or best[1] is None:
            best = (lb, lo, hi)
    lb, lo, hi = best
    top = "%s%d/%s%d" % (lo["kind"], lo["cid"], hi["kind"], hi["cid"]) if lo else "-"
    print("NLB cfg=%s groups=%d top=%s maxlb=%.3f" % (cfg, len(g), top, lb))
    if lo and lb > LB_GATE:
        for r in (lo, hi):
            print("NLB_WORST cfg=%s %s%d part=%d N=%.3f %s" % (cfg, r["kind"], r["cid"], r["part"], r["N"],
                                                              " ".join("%s=%s" % (f, r[f]) for f in FEATS)))
    return lb


def loo(rows, sub):
    """leave-one-out max relative error with the frozen feature subset sub (coefficients refit)"""
    worst = 0.0
    for i in range(len(rows)):
        p = fit_kind(rows[:i] + rows[i + 1:], sub)
        worst = max(worst, abs(predict(p, rows[i]) - rows[i]["N"]) / rows[i]["N"])
    return worst


def pairs(items):
    out = []
    for it in items or []:
        c, _, lg = it.partition("=")
        out.append((c, lg))
    return out


def _glue_ab(rows_a, rows_b, metric="rel"):
    """over sites with launches>0 (rows aligned by part): rel = |glue_a - glue_b| / mean, abs = |glue_a - glue_b| in ms"""
    out = []
    for ra, rb in zip(rows_a, rows_b):
        if ra["launches"] > 0 and ra["part"] == rb["part"]:
            d = abs(ra["glue"] - rb["glue"])
            if metric == "abs":
                out.append(d)
                continue
            m = 0.5 * (ra["glue"] + rb["glue"])
            out.append(d / m if m > 0 else 0.0)
    return np.array(out)


def ab_mode(a, root):
    """--ab: per cfg NSRC/NSUM for both logs, then NAB cfg=.. and NAB ALL (no NGATE)"""
    rroot = (a.ref_build_root or a.build_root).rstrip("/")
    ref = {}
    for cfg, lg in pairs(a.ab_ref):
        la, _, lb = lg.partition(",")
        ra = measure("%s/unet_%s" % (rroot, cfg), la)[0]
        rb = measure("%s/unet_%s" % (rroot, cfg), lb)[0]
        g = _glue_ab(ra, rb, a.ab_metric)
        ref[cfg] = float(np.percentile(g, 95)) if len(g) else float("nan")
    ncfg, npass, nsmall, ok = 0, 0, 0, True
    for cfg, lg in pairs(a.ab):
        la, _, lb = lg.partition(",")
        rows = []
        for log in (la, lb):
            r, txt = measure("%s/unet_%s" % (root, cfg), log)
            ok &= src_line(cfg, log, txt)
            s = sum_line(cfg, r, txt)
            npass += int(s)
            rows.append(r)
        g = _glue_ab(rows[0], rows[1], a.ab_metric)
        mx = float(g.max()) if len(g) else float("nan")
        p95 = float(np.percentile(g, 95)) if len(g) else float("nan")
        rp = ref.get(cfg, float("nan"))
        small = int(p95 < rp)
        nsmall += small
        if a.ab_metric == "abs":
            print("NAB cfg=%s metric=abs_ms sites=%d glue_ab_max=%.4f glue_ab_p95=%.4f ref_p95=%.4f smaller=%d"
                  % (cfg, len(g), mx, p95, rp, small))
        else:
            print("NAB cfg=%s sites=%d glue_ab_max=%.3f glue_ab_p95=%.3f ref_p95=%.3f smaller=%d"
                  % (cfg, len(g), mx, p95, rp, small))
        ncfg += 1
    ok &= ncfg > 0 and npass == 2 * ncfg
    print("NAB ALL cfgs=%d nsum_pass=%d smaller=%d %s" % (ncfg, npass, nsmall, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def floor_mode(a, root):
    """--floor: NNLS on all FEATS (no cap) per (cfg, kind) fitted on --fit, max relative error on --check"""
    checks = dict(pairs(a.check))
    worst, ncfg, ok = 0.0, 0, True
    for cfg, log in pairs(a.fit):
        if cfg not in checks:
            print("NFLOOR cfg=%s no --check log FAIL" % cfg)
            ok = False
            continue
        rows, txt = measure("%s/unet_%s" % (root, cfg), log)
        ok &= src_line(cfg, log, txt)
        rows_b, txt_b = measure("%s/unet_%s" % (root, cfg), checks[cfg])
        ok &= src_line(cfg, checks[cfg], txt_b)
        for kind in KINDS:
            ka = [r for r in rows if r["kind"] == kind and r["launches"] > 0]
            kb = [r for r in rows_b if r["kind"] == kind and r["launches"] > 0]
            p = fit_kind(ka, list(FEATS))
            fl = maxrel(p, kb) if kb else 0.0
            worst = max(worst, fl)
            print("NFLOOR cfg=%s kind=%s n=%d floor=%.3f" % (cfg, kind, len(kb), fl))
        ncfg += 1
    ok &= ncfg > 0 and worst <= FLOOR_GATE
    print("NFLOOR ALL cfgs=%d maxfloor=%.3f %s" % (ncfg, worst, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--fit", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--params")
    ap.add_argument("--check", nargs="+")
    ap.add_argument("--lower-bound", action="store_true")
    ap.add_argument("--ab", nargs="+", metavar="CFG=A.log,B.log", help="run-to-run glue spread of two logs per cfg")
    ap.add_argument("--ab-ref", nargs="+", metavar="CFG=A.log,B.log", help="reference pairs for --ab (ref_p95)")
    ap.add_argument("--ab-metric", choices=("rel", "abs"), default="rel",
                    help="--ab spread: rel = |a-b|/mean (default), abs = |a-b| in ms")
    ap.add_argument("--ref-build-root", help="build root of the --ab-ref logs (default --build-root)")
    ap.add_argument("--floor", action="store_true", help="NNLS on all FEATS, no cap: fit on --fit, error on --check")
    a = ap.parse_args(argv)
    root = a.build_root.rstrip("/")
    ok = True
    if a.ab:
        return ab_mode(a, root)
    if a.floor:
        if not a.fit or not a.check:
            ap.error("--floor needs --fit and --check")
        return floor_mode(a, root)
    if not a.check:
        ap.error("--check is required with --fit/--params/--lower-bound")
    if a.lower_bound:
        worst, ncfg = 0.0, 0
        for cfg, log in pairs(a.check):
            rows, txt = measure("%s/unet_%s" % (root, cfg), log)
            ok &= src_line(cfg, log, txt)
            worst = max(worst, lower_bound(cfg, rows))
            ncfg += 1
        ok &= ncfg > 0 and worst <= LB_GATE
        print("NLB ALL cfgs=%d maxlb=%.3f %s" % (ncfg, worst, "PASS" if ok else "FAIL"))
        return 0 if ok else 1
    if a.params:
        prm = json.load(open(a.params))
    else:
        if not a.fit:
            ap.error("--fit is required without --params")
        prm, looruns = {}, []
        checks = dict(pairs(a.check))
        for cfg, log in pairs(a.fit):
            rows, txt = measure("%s/unet_%s" % (root, cfg), log)
            ok &= src_line(cfg, log, txt)
            ok &= sum_line(cfg, rows, txt)
            kinds = {}
            for kind in KINDS:
                kr = [r for r in rows if r["kind"] == kind and r["launches"] > 0]
                p = fit_kind(kr, kind=kind)
                p["n"] = len(kr)
                kinds[kind] = p
                print("NFIT cfg=%s kind=%s n=%d %s d=%.6g" % (cfg, kind, len(kr), " ".join(
                    "%s=%.6g" % (f, v) for f, v in sorted(p["coef"].items())), p["d"]))
            # LOO with the subset chosen on the fit run, once on the fit log (a) and once on the check log (b)
            runs = [("a", rows)]
            if cfg in checks:
                runs.append(("b", measure("%s/unet_%s" % (root, cfg), checks[cfg])[0]))
            for run, rr in runs:
                worst_run = 0.0
                for kind in KINDS:
                    kr = [r for r in rr if r["kind"] == kind and r["launches"] > 0]
                    lv = loo(kr, kinds[kind]["feats"])
                    worst_run = max(worst_run, lv)
                    print("NLOO cfg=%s run=%s kind=%s loo_maxrel=%.3f" % (cfg, run, kind, lv))
                looruns.append(worst_run)
            host = sig(sum(r["N"] for r in rows if r["launches"] == 0))
            prm[cfg] = {"kinds": kinds, "attn_host_ms": host,
                        "model": "N=sum(coef[f]*f)+d (ms), nnls on relative error, <=%s features per kind"
                                 % json.dumps(MAX_FEATS, sort_keys=True)}
        mloo = max(looruns) if looruns else float("inf")
        print("NLOO ALL cfgs=%d runs=%d maxloo=%.3f %s" % (len(prm), len(looruns) // max(len(prm), 1), mloo,
                                                           "PASS" if looruns and mloo <= GATE else "FAIL"))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(prm, f, sort_keys=True, indent=1)
                f.write("\n")
    allsites, r2s, worst, ncfg = 0, [], 0.0, 0
    for cfg, log in pairs(a.check):
        rows, txt = measure("%s/unet_%s" % (root, cfg), log)
        ok &= src_line(cfg, log, txt)
        ok &= sum_line(cfg, rows, txt)
        if cfg not in prm:
            print("NGATE cfg=%s no params FAIL" % cfg)
            ok = False
            continue
        p = prm[cfg]
        fr = [r for r in rows if r["launches"] > 0]
        meas = np.array([r["N"] for r in fr])
        pred = np.array([predict(p["kinds"][r["kind"]], r) for r in fr])
        rel = np.abs(pred - meas) / meas
        for r, pv, rv in zip(fr, pred, rel):
            if rv > GATE:
                print("NSITE_BAD cfg=%s part=%d kind=%s cid=%d pred=%.3f meas=%.3f rel=%.3f"
                      % (cfg, r["part"], r["kind"], r["cid"], pv, r["N"], rv))
        r2 = 1.0 - float(np.sum((pred - meas) ** 2)) / float(np.sum((meas - meas.mean()) ** 2))
        host_meas = sum(r["N"] for r in rows if r["launches"] == 0)
        tot_meas = float(meas.sum()) + host_meas
        tot_pred = float(pred.sum()) + p["attn_host_ms"]
        trel = abs(tot_pred - tot_meas) / tot_meas
        mx = float(rel.max())
        good = mx <= GATE
        ok &= good
        print("NTOTAL cfg=%s total_rel=%.3f meas_ms=%.3f pred_ms=%.3f" % (cfg, trel, tot_meas, tot_pred))
        print("NGATE cfg=%s sites=%d r2=%.3f maxrel=%.3f total_rel=%.3f attn_host_ms=%.3f %s"
              % (cfg, len(fr), r2, mx, trel, host_meas, "PASS" if good else "FAIL"))
        allsites += len(fr)
        r2s.append(r2)
        worst = max(worst, mx)
        ncfg += 1
    ok &= ncfg > 0
    print("NGATE ALL src=board cfgs=%d sites=%d r2_min=%.3f maxrel=%.3f %s"
          % (ncfg, allsites, min(r2s) if r2s else float("nan"), worst, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
