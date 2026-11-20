#!/usr/bin/env python3
"""Fit / check the N (host glue) segment of UNet AIE sites (costmodel M2).

  n_fit.py --build-root DIR --fit CFG=LOG [...] --out JSON --check CFG=LOG [...]
  n_fit.py --build-root DIR --params JSON --check CFG=LOG [...]          (no fit)

Per site (n_feat.sites of DIR/unet_<cfg>, aligned with the log's 'PROF part=<i> op=aie_host_*' lines):
  N = (site_ms - aie_ms) + glue_ms + flush_ms + inval_ms
  site_ms - aie_ms is the PROF part 'ms'; glue/flush/inval come from 'PROF aie_host_split part=<i>'.
  aie_ms (in launch: kload, rt_*) belongs to K+L, not N.
Model per (cfg, kind in conv|linear|fa), sites with launches>0 only:
  N = a*MK + b*MN + c*launches + d       (least squares; a feature constant over the kind's sites -> coef 0)
Host-only sites (attn roles with 0 launches) are not fitted: their summed fit-run ms is frozen as attn_host_ms
and only enters total_rel.
  NSRC  cfg=.. log=.. board=.. launches=.. run_rc=..                  (from 'CMP unet board=..')
  NSUM  cfg=.. site_glue_ms=.. split_glue_ms=.. rel=.. PASS|FAIL       (sum site glue vs split_detail
                                                                       host_glue_bdcfg*launches, rel<=0.02)
  NFIT  cfg=.. kind=.. n=.. a=.. b=.. c=.. d=..
  NLOO  cfg=.. kind=.. loo_maxrel=..                                   (leave-one-out, printed only)
  NGATE cfg=.. sites=.. r2=.. maxrel=.. total_rel=.. attn_host_ms=.. PASS|FAIL   (maxrel<=0.200)
  NGATE ALL src=board cfgs=.. sites=.. r2_min=.. maxrel=.. PASS|FAIL
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aiegraph.costmodel.n_feat import sites as n_sites  # noqa: E402
from aiegraph.costmodel.unet_seq import read  # noqa: E402

KINDS = ("conv", "linear", "fa")
FEATS = ("MK", "MN", "launches")
GATE = 0.200
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
        out.append(dict(s, part=part, ms=ms, aie_ms=aie_ms, glue=g, N=ms + g + f + i))
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


def design(rows, active):
    return np.array([[float(r[f]) for f, on in zip(FEATS, active) if on] + [1.0] for r in rows])


def fit_kind(rows):
    """coef dict a,b,c,d; features constant over rows get 0"""
    X = np.array([[float(r[f]) for f in FEATS] for r in rows])
    active = [bool(np.ptp(X[:, j]) > 0) for j in range(len(FEATS))]
    y = np.array([r["N"] for r in rows])
    coef, *_ = np.linalg.lstsq(design(rows, active), y, rcond=None)
    full, k = [], 0
    for on in active:
        full.append(float(coef[k]) if on else 0.0)
        k += on
    full.append(float(coef[-1]))
    return {n: sig(v) for n, v in zip("abcd", full)}


def predict(p, r):
    return p["a"] * r["MK"] + p["b"] * r["MN"] + p["c"] * r["launches"] + p["d"]


def loo(rows):
    worst = 0.0
    for i in range(len(rows)):
        p = fit_kind(rows[:i] + rows[i + 1:])
        worst = max(worst, abs(predict(p, rows[i]) - rows[i]["N"]) / rows[i]["N"])
    return worst


def pairs(items):
    out = []
    for it in items or []:
        c, _, lg = it.partition("=")
        out.append((c, lg))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--fit", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--params")
    ap.add_argument("--check", nargs="+", required=True)
    a = ap.parse_args(argv)
    root = a.build_root.rstrip("/")
    ok = True
    if a.params:
        prm = json.load(open(a.params))
    else:
        if not a.fit or not a.out:
            ap.error("--fit and --out are required without --params")
        prm = {}
        for cfg, log in pairs(a.fit):
            rows, txt = measure("%s/unet_%s" % (root, cfg), log)
            ok &= src_line(cfg, log, txt)
            ok &= sum_line(cfg, rows, txt)
            kinds = {}
            for kind in KINDS:
                kr = [r for r in rows if r["kind"] == kind and r["launches"] > 0]
                p = fit_kind(kr)
                p["n"] = len(kr)
                kinds[kind] = p
                print("NFIT cfg=%s kind=%s n=%d a=%.6g b=%.6g c=%.6g d=%.6g" % (cfg, kind, len(kr), p["a"], p["b"],
                                                                                  p["c"], p["d"]))
                print("NLOO cfg=%s kind=%s loo_maxrel=%.3f" % (cfg, kind, loo(kr)))
            host = sig(sum(r["N"] for r in rows if r["launches"] == 0))
            prm[cfg] = {"kinds": kinds, "attn_host_ms": host, "model": "N=a*MK+b*MN+c*launches+d (ms)"}
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
