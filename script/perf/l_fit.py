#!/usr/bin/env python3
"""Fit / check the L (launch overhead) model.

  l_fit.py --kernels DIR [--board palmyra213] --out FIT.json
      DIR/<cfg>/*.log for cfg in off, aot, aot-kco; per-cfg least squares of
      L_meas = const + a*bdcfg_calls + b*startio_calls + c*bdcfg_w + d*cp_w (per launch).
      Prints  LFIT cfg=<cfg> board=<b> n=<n> r2=%.3f maxrel=%.3f loo_maxrel=%.3f
      and writes a reproducible JSON (sort_keys, fixed decimals, no paths/timestamps).
  l_fit.py --check-unet LOG:CFG [...] [--params JSON]
      Prints  LCHK cfg=.. rel=%.3f  and  LGATE ALL n=<n> maxrel=%.3f PASS|FAIL (<=0.200).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aiegraph.costmodel.launch_l import CFGS, FEATURES, feature_row, point, predict_l, load_params  # noqa: E402

ND = 6


def _fit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.round(coef, ND)


def fit_cfg(pts):
    X = np.array([feature_row(p["calls"], p["words"]) for p in pts])
    y = np.array([p["l_meas"] for p in pts])
    coef = _fit(X, y)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    maxrel = float(np.max(np.abs(pred - y) / y))
    loo = 0.0
    for i in range(len(pts)):
        m = np.ones(len(pts), bool)
        m[i] = False
        c = _fit(X[m], y[m])
        loo = max(loo, abs(float(X[i] @ c) - y[i]) / y[i])
    return coef, r2, maxrel, loo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels")
    ap.add_argument("--board", default="palmyra213")
    ap.add_argument("--out")
    ap.add_argument("--check-unet", nargs="+", metavar="LOG:CFG")
    ap.add_argument("--params")
    a = ap.parse_args()
    rc = 0
    if a.kernels:
        out = {"board": a.board, "features": list(FEATURES),
               "l_def": "kload+core_reset+coreen+rt_bdcfg+rt_startio (ms/launch)", "cfg": {}}
        for cfg in CFGS:
            logs = sorted((Path(a.kernels) / cfg).glob("*.log"))
            pts = [point(str(p)) for p in logs]
            if len(pts) < 2:
                print(f"LFIT cfg={cfg} board={a.board} n={len(pts)} too few points")
                rc = 1
                continue
            coef, r2, maxrel, loo = fit_cfg(pts)
            for p, lg in zip(pts, logs):
                pr = float(np.dot(feature_row(p["calls"], p["words"]), coef))
                print(f"  {cfg:8s} {lg.stem:24s} meas={p['l_meas']:.4f} pred={pr:.4f} rel={abs(pr - p['l_meas']) / p['l_meas']:.3f}")
            print(f"LFIT cfg={cfg} board={a.board} n={len(pts)} r2={r2:.3f} maxrel={maxrel:.3f} loo_maxrel={loo:.3f}")
            out["cfg"][cfg] = {"coef": {k: float(f"{v:.{ND}f}") for k, v in zip(FEATURES, coef)},
                               "n": len(pts), "r2": float(f"{r2:.3f}"), "maxrel": float(f"{maxrel:.3f}"),
                               "loo_maxrel": float(f"{loo:.3f}"),
                               "kernels": [lg.stem for lg in logs]}
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(out, f, sort_keys=True, indent=1)
                f.write("\n")
    if a.check_unet:
        prm = load_params(a.params) if a.params else load_params()
        worst = 0.0
        for item in a.check_unet:
            log, cfg = item.rsplit(":", 1)
            p = point(log)
            pr = predict_l(p["calls"], p["words"], cfg, prm)
            rel = abs(pr - p["l_meas"]) / p["l_meas"]
            worst = max(worst, rel)
            print(f"LCHK cfg={cfg} meas={p['l_meas']:.4f} pred={pr:.4f} rel={rel:.3f}")
        ok = worst <= 0.200
        print(f"LGATE ALL n={len(a.check_unet)} maxrel={worst:.3f} {'PASS' if ok else 'FAIL'}")
        rc = rc or (0 if ok else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
