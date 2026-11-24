#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Example C prototype -- the offline tuning loop of PROPOSAL.md on the candidates that exist today.

    python3 -I src/costmodel/example/offline_tuning_proposal/tune_loop.py [--top-k 1] [--margin 0.05] [--no-refit]

Stages (see PROPOSAL.md for what is implemented vs proposed):
  1 enumerate  every compiled candidate build found on disk: <net>_<cfg>/ under the M6 build roots
               (6 candidates: unet / resnet18 / yolo26n x 2 control-path cfgs).  The knob enumerated is the build cfg
               (off = MMIO, aot-kco / aot = control packets), the only knob with compiled builds AND board logs today.
  2 estimate   estimate({"net", "build", "cfg"}) per candidate -> total_ms (frozen palmyra213 params), unpriced terms
  3 rank       per net by predicted total_ms; a candidate goes to the board if it is in the top-k OR its predicted
               total is within --margin of the best one (the model cannot separate such candidates).
  4 "board"    REPLAY ONLY: this script never touches a board.  It reads the archived palmyra213 logs of the selected
               candidates (the M7 logs) as if they were the fresh runs, and -- for scoring only -- the logs of the
               rejected ones too, to tell whether the board-best candidate was kept.
  5 refit      K-segment kernel calibration (costmodel_fit.calibrate, form m5) refit from board logs:
               the 17 archived M7 P3 kernel logs (/scratch/staff/huaj/tmp/costmodel_m7/kernels/palmyra213), compared
               with the frozen params/palmyra213_kcal.m5.json.  Written to --refit-out only if given (never to params/).
Last line: TUNE nets=<n> candidates=<n> board_runs=<n> saved=<n> best_kept=<n>/<n> refit_maxrel=<x> frozen_maxrel=<x>
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "example" / "aiegraph"))

import costmodel_cfg_decision as DEC  # noqa: E402
from costmodel import des, estimate, tools  # noqa: E402
from costmodel.estimate import CYCLES, PARAMS_DIR, board_params  # noqa: E402

KLOGS = "/scratch/staff/huaj/tmp/costmodel_m7/kernels/palmyra213"


def enumerate_candidates():
    """[(net, cfg, build_root, log)] for every <net>_<cfg> build dir that exists"""
    out = []
    for net, (ca, cb, _L, logs) in sorted(DEC.NETS.items()):
        root = DEC.UNET_BUILD if net == "unet" else DEC.NET_BUILD
        for cfg, log in zip((ca, cb), logs["main"]):
            if os.path.isdir(os.path.join(root, "%s_%s" % (net, cfg))):
                out.append((net, cfg, root, log))
    return out


def select(rows, top_k, margin):
    """rows of one net sorted by pred -> rows sent to the board"""
    best = rows[0]["pred"]
    return [r for i, r in enumerate(rows) if i < top_k or r["pred"] <= best * (1.0 + margin)]


def tune(top_k, margin):
    by = {}
    for net, cfg, root, log in enumerate_candidates():
        r = estimate({"net": net, "build": root, "cfg": cfg}, board="palmyra213")
        by.setdefault(net, []).append(dict(net=net, cfg=cfg, log=log, pred=r["total_ms"], unpriced=r["unpriced"]))
        print("CAND net=%-8s cfg=%-7s pred_ms=%9.1f sites=%d unpriced=%s" % (net, cfg, r["total_ms"], len(r["sites"]),
                                                                         ",".join(r["unpriced"]) or "-"))
    runs = kept = 0
    for net, rows in sorted(by.items()):
        rows.sort(key=lambda r: r["pred"])
        sel = select(rows, top_k, margin)
        for r in rows:                                    # replay: measured total of the archived log
            r["meas"] = DEC.read_log(r["log"], net, DEC.NETS[net][2])["total"]
        true_best = min(rows, key=lambda r: r["meas"])
        pick = min(sel, key=lambda r: r["meas"])          # the board decides among the selected ones
        runs += len(sel)
        kept += true_best in sel
        print("RANK net=%-8s order=%s board=%s pick=%s(meas %.1f ms) board_best=%s(meas %.1f ms) %s"
              % (net, ">".join("%s:%.1f" % (r["cfg"], r["pred"]) for r in rows), ",".join(r["cfg"] for r in sel),
                 pick["cfg"], pick["meas"], true_best["cfg"], true_best["meas"],
                 "kept" if true_best in sel else "MISSED"))
    ncand = sum(len(v) for v in by.values())
    return len(by), ncand, runs, kept


def maxrel(points, params):
    return max(abs(des.predict_launch(p["op"], params, p["cycles"])["wait_ms"] / p["meas_ms"] - 1.0) for p in points)


def refit(out):
    fit = tools.load("costmodel_fit")
    base = board_params("palmyra213")
    kc = {k["kernel"]: k for k in json.loads(Path(CYCLES).read_text())["kernels"]}
    pts = [fit.load_point(p, kc[os.path.basename(p)[:-4]], "stored", form="m5")
           for p in sorted(glob.glob(os.path.join(KLOGS, "*.log")))]
    frozen = json.loads((PARAMS_DIR / "palmyra213_kcal.m5.json").read_text())
    keep = lambda c: {k: v for k, v in c.items() if k in des.CAL_DEFAULTS or k in ("ddr_model", "k_form")}  # noqa
    new = {k: (float(v) if isinstance(v, float) else v) for k, v in fit.calibrate(pts, base, "m5").items()}
    mf, mn = maxrel(pts, dict(base, **keep(frozen))), maxrel(pts, dict(base, **keep(new)))
    for k in sorted(keep(new)):
        print("REFIT %-16s frozen=%-12s refit=%s" % (k, frozen.get(k), new[k]))
    print("REFIT kernels=%d logs=%s maxrel frozen=%.3f refit=%.3f (in-sample)" % (len(pts), KLOGS, mf, mn))
    if out:
        Path(out).write_text(json.dumps(dict(new, fitted_on=sorted(p["kernel"] for p in pts)), indent=1,
                                        sort_keys=True))
        print("REFIT written %s (params/ untouched)" % out)
    return mn, mf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--margin", type=float, default=0.05, help="also run candidates within this of the best pred")
    ap.add_argument("--no-refit", action="store_true")
    ap.add_argument("--refit-out", help="write the refit K calibration json here (never into params/)")
    a = ap.parse_args(argv)
    nets, ncand, runs, kept = tune(a.top_k, a.margin)
    rn = rf = float("nan")
    if not a.no_refit:
        rn, rf = refit(a.refit_out)
    print("TUNE nets=%d candidates=%d board_runs=%d saved=%d best_kept=%d/%d refit_maxrel=%.3f frozen_maxrel=%.3f"
          % (nets, ncand, runs, ncand - runs, kept, nets, rn, rf))
    return 0 if kept == nets else 1


if __name__ == "__main__":
    sys.exit(main())
