#!/usr/bin/env python3
"""Cost-model fit, stage K (AIE side): DES (src/aiegraph/costmodel/des.py) prediction of rt_wait_io per launch vs the
board (palmyra213) per-kernel logs of script/perf/p0_one_kernel.sh.

  costmodel_fit.py --stage K --params script/perf/params/<board>.json --cycles <aie2ps_kernel_cycles.json>
                   --kboard <board_compute.json> --logs <dir of <kernel>.log> [--rerun-kernel first|<name>] [--board <ip>] --out k_pred_vs_meas.tsv
                   [--cal-out <json>] [--no-calibrate --cal <json>]

* every log must carry the board marker "BOARD machine=aarch64 midr=0x... board=<name>" -- logs without it are
  refused (printed) and do not enter the fit; prints logs=<N> board_logs=<N>.
* measured value: rt_wait_io of the LAUNCH split_detail line (per launch, ms), NOT wait_io (that adds host_inval).
* --rerun-kernel first re-runs the first kernels.tsv row live on the board (p0_one_kernel.sh) and adds that fresh
  point (kind=rerun) to the table; it is predicted, not used for calibration.
* calibration (CAL_DEFAULTS keys of des.py), 6 constants (cal_params=), DDR model bytes-only (cal ddr_model=bytes):
  - chunk_ovh_cyc: least squares (relative) of DES compute (ISS core_cyc + ovh * chunks) against the board compute
    counters (--kboard board_compute.json: median over launch tags of max-tile active - lock_stall - stream_stall);
  - ddr_bw_scale, ddr_run0_b (achieved DDR bandwidth and per-run overhead of the shim BD run), b_prefetch_b (b-port
    bytes streamed while the host still issues the round), round_fixed_cyc (fixed per-round latency) and
    compute_exposed (share of core compute on the wait-window critical path) by minimax of |log(pred/meas)| over
    the stored kernels.
  Written to --cal-out (default params/<board>_kcal.json).  Overfit guard: leave-one-kernel-out refit of all 6
  constants, loo_maxrel = max held-out |pred/meas - 1| (printed on the COSTMODEL line, also with --cal).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from aiegraph.costmodel import des  # noqa: E402
from aiegraph.costmodel.launch_split import parse  # noqa: E402
from aiegraph.costmodel.roofline import roofline  # noqa: E402

BOARD_RE = re.compile(r"^BOARD machine=(\S+) midr=(0x[0-9a-f]+) board=(\S+)$")
KTSV = "/scratch/staff/huaj/aiengineers/arthur/tasks/2026-11-19_costmodel/kernels.tsv"
CAL_KEYS = ("chunk_ovh_cyc", "ddr_bw_scale", "ddr_run0_b", "b_prefetch_b", "round_fixed_cyc", "compute_exposed")
DDR_MODEL = "bytes"


def board_marker(path):
    with open(path, errors="replace") as f:
        for line in f:
            m = BOARD_RE.match(line.rstrip("\r\n"))
            if m:
                return m
    return None


def load_point(path, kern, kind, name=None):
    r = parse(path)
    sd = r["split_detail"]
    if sd is None or "wait_io" not in sd["rt"]:
        return None
    perf = [p for p in r["perf"] if p["rc_bad"] == 0 and p["compute_max"] is not None]
    inst = kern["instance"]
    op = {"instance": inst, "host": des.host_from_split(sd, inst["ntile"])}
    n = len(perf)
    return {
        "name": name or kern["kernel"], "kernel": kern["kernel"], "kind": kind, "recipe": kern["recipe"], "op": op,
        "cycles": kern, "meas_ms": float(sd["rt"]["wait_io"]), "launches": int(sd["other"].get("launches", 0)),
        "board_compute_cyc": (sum(p["compute_max"] for p in perf) / n) if n else None,
        # lock+stream stall of the slowest tile, mean over launches
        "board_stall_cyc": (sum(max(t["lock_stall"] + t["stream_stall"] for t in p["per_tile"] if t["rc"] == 0)
                                for p in perf) / n) if n else None,
        "perf_n": n, "log": path,
    }


def nnls_compute(points, params):
    """chunk_ovh_cyc >= 0 minimising the relative error of DES compute vs board compute."""
    import numpy as np
    from scipy.optimize import nnls
    pts = [p for p in points if p["board_compute_cyc"]]
    X, y = [], []
    for p in pts:
        g = des.geometry(p["op"])
        w = 1.0 / p["board_compute_cyc"]
        X.append([g["chunks"] * w])
        y.append((p["board_compute_cyc"] - p["cycles"]["core_cyc"]) * w)
    coef, _ = nnls(np.array(X), np.array(y))
    return {"chunk_ovh_cyc": round(float(coef[0]), 2)}


def fit_wait(points, params, cal):
    """ddr_bw_scale, ddr_run0_b, b_prefetch_b, round_fixed_cyc, compute_exposed by minimax |log(pred/meas)|
    (+ 0.5 mean square), bytes-only DDR model."""
    import numpy as np
    from scipy.optimize import minimize

    def unpack(x):
        c = dict(cal)
        c.update({"round_fixed_cyc": abs(x[0]) * 1e3, "ddr_bw_scale": abs(x[1]), "ddr_run0_b": abs(x[2]),
                  "compute_exposed": min(1.0, abs(x[3])), "b_prefetch_b": abs(x[4]) * 1e3})
        return c

    def obj(x):
        p2 = dict(params, ddr_model=DDR_MODEL)
        p2.update(unpack(x))
        e = np.array([math.log(max(des.predict_launch(p["op"], p2, p["cycles"])["wait_ms"], 1e-9) / p["meas_ms"])
                      for p in points])
        return float(np.max(np.abs(e)) + 0.5 * np.mean(e ** 2))

    best = None
    for x0 in ([17.0, 1.5, 45.0, 0.01, 178.0], [20.0, 1.5, 40.0, 0.3, 100.0], [40.0, 1.0, 20.0, 0.7, 0.0]):
        x = list(x0)
        for _rs in range(6):  # Nelder-Mead restarts from the incumbent (the minimax objective is not smooth)
            o = minimize(obj, x, method="Nelder-Mead", options={"maxiter": 3000, "maxfev": 3000, "xatol": 1e-6,
                                                                "fatol": 1e-10, "adaptive": True})
            improved = best is None or o.fun < best.fun - 1e-9
            if improved:
                best = o
            x = list(best.x)
            if not improved:
                break
    c = unpack(best.x)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}


def calibrate(points, params):
    return dict(fit_wait(points, params, nnls_compute(points, params)), ddr_model=DDR_MODEL)


def des_terms(op, params, d):
    """Split the predicted wait (ms/launch) into the DES terms of des.predict_launch:
    compute = core compute + locks on the round's critical path (aie - stall), dma = DMA/DDR-paced idle + c drain
    (stall), fixed = round_fixed_cyc + wait_call_cyc*waits, overlap = -(overlap_bd*bd + b_overlap_cyc*b share)."""
    g = des.geometry(op)
    clk_mhz = float(params["aie_clk_mhz"])
    host = op.get("host") or {}
    bd_cyc = float(host.get("bd_us", params.get("host_bd_us", 5.0))) * clk_mhz
    R = g["R"]
    wpr = float(host.get("waits_per_launch", 12 * R)) / R
    to_ms = 1.0 / (clk_mhz * 1e3)
    fixed = des._p(params, "round_fixed_cyc") + des._p(params, "wait_call_cyc") * wpr
    overlap = des._p(params, "overlap_bd") * bd_cyc + des._p(params, "b_overlap_cyc") * g["b_b"] / (g["a_b"] + g["b_b"])
    return {"compute": d["aie_ms"] - d["stall_ms"], "dma": d["stall_ms"], "fixed": R * fixed * to_ms,
            "overlap": -R * overlap * to_ms}


def loo_maxrel(points, params, detail=None):
    """Leave-one-kernel-out: refit all calibrated constants without the kernel, predict it.
    detail = full-fit cal dict -> also print one KLOO line per held-out kernel (signed rel, the calibrated constant
    with the largest relative change vs the full fit, the largest |DES term| of the held-out prediction)."""
    worst = 0.0
    for i, q in enumerate(points):
        rest = [p for p in points if p["kernel"] != q["kernel"]]
        p2 = dict(params, ddr_model=DDR_MODEL)
        cl = calibrate(rest, params)
        p2.update(cl)
        d = des.predict_launch(q["op"], p2, q["cycles"])
        rel = abs(d["wait_ms"] / q["meas_ms"] - 1.0)
        print("[fit] loo %-26s rel=%.3f" % (q["kernel"], rel))
        worst = max(worst, rel)
        if detail is not None:
            dc = {k: (cl[k] - float(detail[k])) / max(abs(float(detail[k])), 1e-9)
                  for k in CAL_KEYS if k in cl and k in detail}
            kmax = max(dc, key=lambda k: abs(dc[k]))
            t = des_terms(q["op"], p2, d)
            tmax = max(t, key=lambda k: abs(t[k]))
            print("KLOO kernel=%s rel=%+.3f pred_ms=%.4f meas_ms=%.4f dcal=%s:%s->%s(%+.2f) term=%s %s"
                  % (q["kernel"], d["wait_ms"] / q["meas_ms"] - 1.0, d["wait_ms"], q["meas_ms"], kmax,
                     detail[kmax], cl[kmax], dc[kmax], tmax,
                     " ".join("%s_ms=%.4f" % (k, v) for k, v in t.items())))
    return worst


def rerun(kernel, board, outdir):
    os.makedirs(outdir, exist_ok=True)
    log = os.path.join(outdir, kernel + ".log")
    cmd = ["bash", os.path.join(HERE, "p0_one_kernel.sh"), "--board", board, "--kernel", kernel]
    print("[fit] rerun: %s > %s" % (" ".join(cmd), log), flush=True)
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    p0 = [l.rstrip() for l in open(log, errors="replace") if l.startswith("P0 ")]
    print("[fit] rerun rc=%d %s" % (rc, p0[-1] if p0 else "(no P0 line)"), flush=True)
    return log, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["K"])
    ap.add_argument("--params", required=True)
    ap.add_argument("--cycles", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--rerun-kernel", default=None)
    ap.add_argument("--board", default=None)
    ap.add_argument("--kernels", default=KTSV)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cal-out", default=None)
    ap.add_argument("--cal", default=None, help="use this calibration json instead of fitting")
    ap.add_argument("--kboard", default=None,
                    help="board_compute.json (kboard_from_logs): board compute counters per kernel for the compute "
                         "calibration and the DES compute check (else the logs' PERF lines, mean over launches)")
    ap.add_argument("--rerun-dir", default=None)
    ap.add_argument("--loo-detail", action="store_true",
                    help="print a KLOO line per held-out kernel (rel, dcal = largest relative change of a calibrated "
                         "constant vs the full fit, term = largest DES term of the held-out prediction)")
    a = ap.parse_args()

    params = json.load(open(a.params))
    board_name = params.get("board", "")
    kc = json.load(open(a.cycles))["kernels"]
    kmap = {k["kernel"]: k for k in kc}
    kbmap = {}
    if a.kboard:
        kb = json.load(open(a.kboard))
        if kb.get("board") not in (None, board_name):
            sys.exit("--kboard board=%s != params board=%s" % (kb.get("board"), board_name))
        kbmap = {k["name"]: float(k["board_compute_cyc"]) for k in kb["kernels"]}
        print("[fit] kboard=%s (%d kernels, %s)" % (a.kboard, len(kbmap), kb.get("board_compute", "")[:60]))
    print("[fit] stage=K params=%s (%d keys, board=%s aie_clk_mhz=%s) cycles=%s (%d kernels, src=%s)"
          % (a.params, len(params), board_name, params.get("aie_clk_mhz"), a.cycles, len(kc), "ISS"))

    logs = sorted(glob.glob(os.path.join(a.logs, "*.log")))
    n_logs = n_board = 0
    first_marker = None
    points = []
    for path in logs:
        n_logs += 1
        m = board_marker(path)
        if not m or m.group(1) != "aarch64" or m.group(3) != board_name:
            print("[fit] refused %s: no 'BOARD machine=aarch64 ... board=%s' marker" % (path, board_name))
            continue
        n_board += 1
        first_marker = first_marker or m.group(0)
        name = os.path.basename(path)[:-4]
        if name not in kmap:
            print("[fit] skip %s: kernel not in %s" % (path, a.cycles))
            continue
        p = load_point(path, kmap[name], "stored")
        if p is None:
            print("[fit] skip %s: no LAUNCH split_detail rt_wait_io" % path)
            continue
        if kbmap:
            if name not in kbmap:
                print("[fit] skip %s: kernel not in %s" % (path, a.kboard))
                continue
            p["board_compute_cyc"] = kbmap[name]
        points.append(p)
    if first_marker:
        print(first_marker)

    rerun_pt = None
    if a.rerun_kernel:
        if not a.board:
            sys.exit("--rerun-kernel needs --board")
        kern = a.rerun_kernel
        if kern == "first":
            with open(a.kernels) as f:
                kern = next(csv.DictReader(f, delimiter="\t"))["kernel"]
        rdir = a.rerun_dir or os.path.join(os.path.dirname(os.path.abspath(a.out)), "rerun")
        rlog, rc = rerun(kern, a.board, rdir)
        n_logs += 1
        m = board_marker(rlog)
        if rc != 0 or not m or m.group(1) != "aarch64" or m.group(3) != board_name:
            print("[fit] refused rerun log %s (rc=%d, marker=%s)" % (rlog, rc, m.group(0) if m else None))
        else:
            n_board += 1
            print(m.group(0))
            rerun_pt = load_point(rlog, kmap[kern], "rerun", name=kern + "_rerun")
            if rerun_pt and kern in kbmap and not rerun_pt["board_compute_cyc"]:
                rerun_pt["board_compute_cyc"] = kbmap[kern]
            if rerun_pt:
                print("rerun=%s board=%s machine=%s launches=%d rt_wait_io_ms=%.4f log=%s"
                      % (kern, m.group(3), m.group(1), rerun_pt["launches"], rerun_pt["meas_ms"], rlog))

    # calibration on the stored kernels only
    if a.cal:
        cal = json.load(open(a.cal))
    else:
        cal = calibrate(points, params)
        cal_out = a.cal_out or os.path.join(os.path.dirname(os.path.abspath(a.params)), board_name + "_kcal.json")
        json.dump(dict(cal, fitted_on=sorted(p["kernel"] for p in points), n=len(points),
                       meas="rt_wait_io", src_params=os.path.abspath(a.params)),
                  open(cal_out, "w"), indent=1, sort_keys=True)
        print("[fit] calibration -> %s" % cal_out)
    print("[fit] cal " + " ".join("%s=%s" % (k, cal[k]) for k in sorted(cal) if k in des.CAL_DEFAULTS))
    p2 = dict(params)
    p2.update({k: v for k, v in cal.items() if k in des.CAL_DEFAULTS})
    if "ddr_model" in cal:
        p2["ddr_model"] = cal["ddr_model"]
    cal_params = len([k for k in cal if k in des.CAL_DEFAULTS])
    print("[fit] ddr model=%s (ddr_burst_cyc0=%s ddr_burst_b_per_cyc=%s)" % (
        "per-burst" if des.burst_model(p2) else "bytes-only", p2.get("ddr_burst_cyc0"), p2.get("ddr_burst_b_per_cyc")))

    allp = points + ([rerun_pt] if rerun_pt else [])
    rows = []
    for p in allp:
        d = des.predict_launch(p["op"], p2, p["cycles"])
        rf = roofline(p["op"], p2, p["cycles"])
        p["pred"] = d
        p["roof"] = rf
        rows.append(p)
        print("[K] %-26s %-6s pred_ms=%.4f meas_ms=%.4f rel=%+.3f bound=%s roof=%s compute_ms=%.3f dma_ms=%.3f "
              "lock_ms=%.3f stall_ms=%.3f" % (p["name"], p["kind"], d["wait_ms"], p["meas_ms"],
                                              (d["wait_ms"] - p["meas_ms"]) / p["meas_ms"], d["bound"], rf["bound"],
                                              d["compute_ms"], d["dma_ms"], d["lock_ms"], d["stall_ms"]))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["stage", "name", "kind", "pred_ms", "meas_ms", "rel", "bound", "compute_ms", "dma_ms", "lock_ms",
                    "stall_ms", "board_compute_ms", "board_stall_ms", "iss_over_board", "log"])
        clk_khz = float(params["aie_clk_mhz"]) * 1e3
        for p in rows:
            d = p["pred"]
            bc = p["board_compute_cyc"]
            w.writerow(["K", p["name"], p["kind"], "%.4f" % d["wait_ms"], "%.4f" % p["meas_ms"],
                        "%.4f" % ((d["wait_ms"] - p["meas_ms"]) / p["meas_ms"]), d["bound"],
                        "%.4f" % d["compute_ms"], "%.4f" % d["dma_ms"], "%.4f" % d["lock_ms"], "%.4f" % d["stall_ms"],
                        "%.4f" % (bc / clk_khz) if bc else "", "%.4f" % (p["board_stall_cyc"] / clk_khz)
                        if p["board_stall_cyc"] else "", "%.4f" % (p["cycles"]["core_cyc"] / bc) if bc else "",
                        p["log"]])
    print("[fit] wrote %s (%d rows)" % (a.out, len(rows)))

    pr = [p["pred"]["wait_ms"] for p in rows]
    me = [p["meas_ms"] for p in rows]
    mu = sum(me) / len(me)
    r2 = 1 - sum((x - y) ** 2 for x, y in zip(pr, me)) / sum((y - mu) ** 2 for y in me)
    mr = max(abs(x - y) / y for x, y in zip(pr, me))
    lme = [math.log(y) for y in me]
    lmu = sum(lme) / len(lme)
    lr2 = 1 - sum((math.log(x) - y) ** 2 for x, y in zip(pr, lme)) / sum((y - lmu) ** 2 for y in lme)
    kernels = len({p["kernel"] for p in rows})
    print("[fit] report log_R2=%.3f" % lr2)
    comp = [(p["pred"]["compute_ms"] * clk_khz, p["board_compute_cyc"]) for p in rows if p["board_compute_cyc"]]
    cmr = max(abs(x - y) / y for x, y in comp)
    iss = [p["cycles"]["core_cyc"] / p["board_compute_cyc"] for p in rows if p["board_compute_cyc"]]
    print("[fit] report ISS/board compute ratio: min=%.3f max=%.3f mean=%.3f n=%d"
          % (min(iss), max(iss), sum(iss) / len(iss), len(iss)))
    st = [abs(p["pred"]["stall_ms"] - p["board_stall_cyc"] / clk_khz) * 1e3 for p in rows if p["board_stall_cyc"]]
    print("[fit] report DES stall vs board lock+stream stall: abs err max=%.1f us mean=%.1f us n=%d "
          "(board stall includes core idle during host issue; profiling on)" % (max(st), sum(st) / len(st), len(st)))
    loo = loo_maxrel(points, params, detail=cal if a.loo_detail else None)
    print("COSTMODEL K fit: R2=%.3f maxrel=%.3f n=%d kernels=%d logs=%d board_logs=%d meas=rt_wait_io"
          " cal_params=%d loo_maxrel=%.3f" % (r2, mr, len(rows), kernels, n_logs, n_board, cal_params, loo))
    print("DES compute: maxrel=%.3f n=%d src=perfctr" % (cmr, len(comp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
