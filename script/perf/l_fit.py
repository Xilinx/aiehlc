#!/usr/bin/env python3
"""Fit / check the L (launch overhead) model.

  l_fit.py --kernels DIR [--board palmyra213] --out FIT.json
      DIR/<cfg>/*.log for cfg in off, aot, aot-kco; per-cfg least squares of
      L_meas = const + a*bdcfg_calls + b*startio_calls + c*bdcfg_w + d*cp_w (per launch).
      Prints  LFIT cfg=<cfg> board=<b> n=<n> r2=%.3f maxrel=%.3f loo_maxrel=%.3f
      and writes a reproducible JSON (sort_keys, fixed decimals, no paths/timestamps).
  l_fit.py --check-unet CFG=LOG [...] [--params JSON]
      apppaltest UNet logs (no BOARD line): board/launches from 'CMP unet board=.. aie_launches=.. run_rc=0',
      ctrlpkt from 'LAUNCH warm' (aot-kco prints aot), active [CTRLPKT_AOT] (table!=0) iff cfg!=off, split_words required.
      Prediction uses only per-launch call counts and word counts.  Prints
      LGATE cfg=.. board=.. launches=.. pred_ms= meas_ms= abs_us= rel=%.3f  and
      LGATE ALL n=<n> maxrel=%.3f PASS|FAIL (<=0.200).
"""
import argparse
import json
import re
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

_CMP = re.compile(r"CMP unet board=(\S+) .*aie_launches=(\d+) .*run_rc=(-?\d+)")
_WARM = re.compile(r"LAUNCH warm .*launches=(\d+).* ctrlpkt=(\S+)")


def unet_meta(log):
    """apppaltest UNet log: board/launches from the CMP line, ctrlpkt from LAUNCH warm."""
    u = {"board": None, "launches": None, "run_rc": None, "ctrlpkt": None, "warm_launches": None,
         "cp_lines": 0, "split_words": False}
    with open(log, errors="replace") as f:
        for line in f:
            m = _CMP.search(line)
            if m:
                u["board"], u["launches"], u["run_rc"] = m.group(1), int(m.group(2)), int(m.group(3))
            m = _WARM.search(line)
            if m:
                u["warm_launches"], u["ctrlpkt"] = int(m.group(1)), m.group(2)
            if "[CTRLPKT_AOT]" in line and not re.search(r"\btable=0\b", line):
                u["cp_lines"] += 1  # off prints an all-zero 'table=0' summary; count active tables only
            if "LAUNCH split_words" in line:
                u["split_words"] = True
    return u


def unet_check(u, cfg):
    if u["board"] is None:
        return "no 'CMP unet board=... aie_launches=... run_rc=' line"
    if u["run_rc"] != 0:
        return f"run_rc={u['run_rc']}"
    if u["ctrlpkt"] is None:
        return "no 'LAUNCH warm ... ctrlpkt=' line"
    want = "off" if cfg == "off" else "aot"  # aot-kco prints ctrlpkt=aot (build_unet.py)
    if u["ctrlpkt"] != want:
        return f"ctrlpkt={u['ctrlpkt']} but cfg={cfg}"
    if (cfg == "off") != (u["cp_lines"] == 0):
        return f"[CTRLPKT_AOT] lines={u['cp_lines']} inconsistent with cfg={cfg}"
    if not u["split_words"]:
        return "no 'LAUNCH split_words' line"
    if u["warm_launches"] is not None and u["warm_launches"] != u["launches"]:
        return f"LAUNCH warm launches={u['warm_launches']} != aie_launches={u['launches']}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels")
    ap.add_argument("--board", default="palmyra213")
    ap.add_argument("--out")
    ap.add_argument("--check-unet", nargs="+", metavar="CFG=LOG")
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
        worst, n_ok = 0.0, 0
        for item in a.check_unet:
            if "=" in item and item.split("=", 1)[0] in CFGS:
                cfg, log = item.split("=", 1)
            else:  # legacy LOG:CFG
                log, cfg = item.rsplit(":", 1)
            err = None
            try:
                u = unet_meta(log)
                p = point(log)
            except Exception as e:  # noqa: BLE001
                err, u = f"parse error {e!r}", {}
            if err is None:
                err = unet_check(u, cfg)
            if err:
                print(f"LGATE cfg={cfg} FAIL {err}")
                worst = max(worst, 9.999)
                continue
            pr = predict_l(p["calls"], p["words"], cfg, prm)
            meas = p["l_meas"]
            rel = abs(pr - meas) / meas
            worst = max(worst, rel)
            n_ok += 1
            print(f"  LDET cfg={cfg} calls/launch bdcfg={p['calls']['bdcfg']:.3f} startio={p['calls']['startio']:.3f} "
                  f"kload={p['calls']['kload']:.3f} words bdcfg_w={p['words']['bdcfg_w']:.1f} cp_w={p['words']['cp_w']:.1f} "
                  f"ctrlpkt={u['ctrlpkt']} cp_aot_lines={u['cp_lines']}")
            print(f"LGATE cfg={cfg} board={u['board']} launches={u['launches']} pred_ms={pr:.4f} meas_ms={meas:.4f} "
                  f"abs_us={abs(pr - meas) * 1000:.1f} rel={rel:.3f}")
        ok = worst <= 0.200 and n_ok == len(a.check_unet)
        print(f"LGATE ALL n={len(a.check_unet)} maxrel={worst:.3f} {'PASS' if ok else 'FAIL'}")
        rc = rc or (0 if ok else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
