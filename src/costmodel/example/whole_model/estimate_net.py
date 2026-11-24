#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Example B -- estimate a whole network per site + total and compare with the measured board run.

    python3 -I src/costmodel/example/whole_model/estimate_net.py [--net resnet18|unet|yolo26n] [--rows N]

Prediction: costmodel.estimate({"net", "build", "cfg"}) (net mode, frozen per-board params, board
palmyra213), per site site_ms = T_hat x launches, for both build cfgs of the net.  Inputs are the build dirs only.
Measurement: the archived palmyra213 board logs of the M7 report (unet M34 *.a.log, resnet18 M41 *.d.log, yolo26n
M42 *.d.log), read with example/aiegraph/costmodel_cfg_decision.read_log -- the same build dirs, logs and reader as
tools/report_scatter.py, so the numbers are the M7 pred_vs_meas.tsv rows of this net.
Cross-checks (when the files exist): T_hat == the committed M6 golden (src/costmodel/test/data/costmodel/m6_golden_*.tsv)
and site rows == the M7 report's pred_vs_meas.tsv.
Data used that is NOT in the repository (archived on /scratch, read only): the build dirs (UNET_BUILD / NET_BUILD of
costmodel_cfg_decision.py) and the board logs it names.
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "example" / "aiegraph"))

import costmodel_cfg_decision as DEC  # noqa: E402
from costmodel import estimate  # noqa: E402

GOLDEN = {"resnet18": "m6_golden_r18.tsv", "yolo26n": "m6_golden_yolo26n.tsv"}
M7_TSV = Path("/scratch/staff/huaj/aiengineers/arthur/clones/cost/tasks/2026-11-23_M7/scatter/pred_vs_meas.tsv")


def golden_mismatch(net, cfg, sites):
    """sites whose T_hat differs from the committed M6 golden (None when there is no golden for the net)"""
    if net not in GOLDEN:
        return None
    g = {(r["cfg"], r["site"]): float(r["T_hat"]) for r in csv.DictReader(
        open(ROOT / "src/costmodel/test/data/costmodel" / GOLDEN[net]), delimiter="\t")}
    return [s["site"] for s in sites if abs(g.get((cfg, s["site"]), -1.0) - s["T_hat"]) > 5e-5]


def m7_mismatch(net, cfg, pred, meas):
    if not M7_TSV.exists():
        return None
    rows = [r for r in csv.DictReader(open(M7_TSV), delimiter="\t") if r["net"] == net and r["cfg"] == cfg]
    bad = [r["part"] for r in rows if abs(float(r["pred_ms"]) - pred.get(int(r["part"]), -1)) > 5e-4
           or abs(float(r["meas_ms"]) - meas.get(int(r["part"]), -1)) > 5e-4]
    return bad + (["count %d != %d" % (len(rows), len(pred))] if len(rows) != len(pred) else [])


def r2(x, y):
    my = sum(y) / len(y)
    return 1.0 - sum((b - a) ** 2 for a, b in zip(x, y)) / sum((v - my) ** 2 for v in y)


def one_cfg(net, cfg, log, build, nrows):
    r = estimate({"net": net, "build": build, "cfg": cfg}, board="palmyra213")
    m = DEC.read_log(log, net, DEC.NETS[net][2])
    key = lambda s: int(s["part"])  # noqa: E731  (board log PROF part= of the site, every net)
    pred = {key(s): s["T_hat"] * s["launches"] for s in r["sites"]}
    meas = dict(m["sites"])
    print("== %s cfg=%s board=%s  build=%s/%s_%s" % (net, cfg, ",".join(m["boards"]), build, net, cfg))
    print("   log=%s" % log)
    print("   %-10s %8s %10s %10s %10s %7s" % ("site", "launches", "T_hat_ms", "pred_ms", "meas_ms", "rel"))
    rows = sorted(r["sites"], key=lambda s: -pred[key(s)])
    for s in rows[:nrows]:
        p, q = pred[key(s)], meas.get(key(s), float("nan"))
        print("   %-10s %8d %10.4f %10.4f %10.4f %+7.3f" % (s["site"], s["launches"], s["T_hat"], p, q, p / q - 1))
    if len(rows) > nrows:
        print("   ... %d more sites (--rows to show)" % (len(rows) - nrows))
    common = sorted(set(pred) & set(meas))
    tp, tm = sum(pred[k] for k in common), sum(meas[k] for k in common)
    print("   TOTAL sites=%d pred_ms=%.1f meas_ms=%.1f rel=%+.3f (estimate total_ms=%.4f)"
          % (len(common), tp, tm, tp / tm - 1, r["total_ms"]))
    gm, mm = golden_mismatch(net, cfg, r["sites"]), m7_mismatch(net, cfg, pred, meas)
    print("   check M6 golden T_hat: %s; M7 pred_vs_meas.tsv rows: %s"
          % ("n/a" if gm is None else ("identical" if not gm else "DIFF %s" % gm[:5]),
             "n/a" if mm is None else ("identical" if not mm else "DIFF %s" % mm[:5])))
    ok = len(common) == len(pred) == len(meas) and not gm and not mm
    return ok, [pred[k] for k in common], [meas[k] for k in common]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="resnet18", choices=sorted(DEC.NETS))
    ap.add_argument("--rows", type=int, default=25, help="sites printed per cfg (largest predicted first)")
    a = ap.parse_args(argv)
    ca, cb, _L, logs = DEC.NETS[a.net]
    build = DEC.UNET_BUILD if a.net == "unet" else DEC.NET_BUILD
    ok, xs, ys = True, [], []
    for cfg, log in zip((ca, cb), logs["main"]):
        o, x, y = one_cfg(a.net, cfg, log, build, a.rows)
        ok, xs, ys = ok and o, xs + x, ys + y
    q, mr = r2(xs, ys), max(abs(x / y - 1) for x, y in zip(xs, ys))
    print("EXAMPLE_B net=%s pts=%d r2=%.3f maxrel=%.3f %s" % (a.net, len(xs), q, mr, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
