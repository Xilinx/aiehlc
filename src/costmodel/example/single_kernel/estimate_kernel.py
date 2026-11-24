#!/usr/bin/env python3
"""Example A -- estimate ONE kacc GEMM kernel (segment K, the AIE-side wait) and compare with its board log.

    python3 -I src/costmodel/example/single_kernel/estimate_kernel.py [--log LOG]

Kernel: u_lin_1024x320x320 (kacc_gemm_bf16, M=1024 K=320 N=320), one of the 17 calibration kernels.
Measured: the committed palmyra213 board log src/costmodel/test/data/costmodel/u_lin_1024x320x320.log
  (rt_wait_io of its 'LAUNCH split_detail' line, ms per launch -- the quantity segment K predicts).
Predicted, three ways, all with the committed frozen params (nothing is fitted here):
  1. estimate() public API, no host input        -> host BD cost defaults to 5.0 us (listed as unpriced)
  2. estimate() public API, host BD cost from log -> host_from_split() of the same log (an INPUT, not the answer)
  3. the M5 kernel calibration (params/palmyra213_kcal.m5.json, the one behind REPORT 3b) through des.predict_launch
The ISS row (instance geometry + core_cyc) is inlined from aie2ps_kernel_cycles.json, as test_costmodel_estimate.py
does, so the example needs no file outside the repository.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2]                                   # .../src
sys.path.insert(0, str(SRC))                            # python -I drops PYTHONPATH

from costmodel import des, estimate  # noqa: E402
from costmodel import tools  # noqa: E402
from costmodel.estimate import PARAMS_DIR, board_params  # noqa: E402

KERNEL = "u_lin_1024x320x320"
ROW = {"kernel": KERNEL, "recipe": "kacc_gemm_bf16", "M": 1024, "K": 320, "N": 320,
       "instance": {"ch": 320, "k": 320, "px": 1024, "ntile": 16, "S": 5, "tkw": 8, "kch": 16, "nkc": 20, "tr": 1},
       "core_cyc": 694124}
LOG = SRC / "costmodel" / "test" / "data" / "costmodel" / (KERNEL + ".log")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(LOG))
    a = ap.parse_args(argv)
    fit = tools.load("costmodel_fit")                   # the K fitting tool's own log parser
    marker = fit.board_marker(a.log)
    pt = fit.load_point(a.log, ROW, "stored", form="m5")
    meas = pt["meas_ms"]
    print(marker.group(0) if marker else "(no BOARD marker)")
    print("measured  rt_wait_io = %.4f ms/launch  (launches=%d, %s)" % (meas, pt["launches"],
                                                                       Path(a.log).relative_to(SRC.parent)))
    rows = []
    op = {"kernel": KERNEL, "recipe": "kacc_gemm_bf16", "M": 1024, "K": 320, "N": 320}
    with tempfile.TemporaryDirectory() as td:          # estimate() reads a cycles table file: give it the one row
        cyc = os.path.join(td, "cycles.json")
        Path(cyc).write_text(json.dumps({"kernels": [ROW]}))
        r1 = estimate([op], board="palmyra213", cycles=cyc)
        host = pt["op"]["host"]
        r2 = estimate([dict(op, host={"bd_us": host["bd_us"]})], board="palmyra213", cycles=cyc)
    rows.append(("estimate() default host (5.0 us/BD)", r1["wait"], r1["bound"], r1["unpriced"]))
    rows.append(("estimate() host bd_us=%.3f from log" % host["bd_us"], r2["wait"], r2["bound"], r2["unpriced"]))
    m5 = dict(board_params("palmyra213"))
    cal = json.loads((PARAMS_DIR / "palmyra213_kcal.m5.json").read_text())
    m5.update({k: v for k, v in cal.items() if k in des.CAL_DEFAULTS or k in ("ddr_model", "k_form")})
    d = des.predict_launch(pt["op"], m5, ROW)
    rows.append(("des.predict_launch, M5 cal (kcal.m5)", d["wait_ms"], d["bound"], []))
    for name, pred, bound, unp in rows:
        print("predicted %-40s = %.4f ms  rel=%+.3f  bound=%s%s" % (name, pred, pred / meas - 1.0, bound,
                                                                    "  unpriced=" + ",".join(unp) if unp else ""))
    print("DES terms (M5): compute_ms=%.4f ddr_ms=%.4f dma_ms=%.4f stall_ms=%.4f rounds=%d"
          % (d["compute_ms"], d["ddr_ms"], d["dma_ms"], d["stall_ms"], d["rounds"]))
    ok = abs(rows[-1][1] / meas - 1.0) <= 0.20
    print("EXAMPLE_A kernel=%s meas_ms=%.4f pred_m5_ms=%.4f rel=%+.3f %s"
          % (KERNEL, meas, rows[-1][1], rows[-1][1] / meas - 1.0, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
