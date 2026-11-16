#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Flow 4 -- the offline model compiler: dnn_torch -> aiednn MLIR pipeline -> ctypes run or C app.

    python example/aiednn/flows/4_model_compiler/run.py [--model resnet18] [--mode f32|int8] [--no-mlir]
    python example/aiednn/flows/4_model_compiler/run.py --app [--target vek280 --board <host> --vitis-settings <...>]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import model_and_input, model_arg        # noqa: E402
from aiegraph import dnn_torch as dt                # noqa: E402
from aiegraph.dnn_ctypes import AiednnLib           # noqa: E402

OUT = REPO / "worklocal" / "aiednn" / "flows"


def main() -> int:
    ap = argparse.ArgumentParser()
    model_arg(ap)
    ap.add_argument("--mode", default="int8", choices=["f32", "int8"])
    ap.add_argument("--no-mlir", action="store_true", help="the Python epilogue fusion instead of the MLIR pipeline")
    ap.add_argument("--app", action="store_true", help="emit the C app and build it with ag.dnn_build")
    ap.add_argument("--target", default="x86", choices=["x86", "vek280", "vek385"])
    ap.add_argument("--board", default=None, help="run the built ELF on this board (a non-x86 --target)")
    ap.add_argument("--vitis-settings", default=None)
    a = ap.parse_args()
    m, x = model_and_input(a.model)
    mlir = not a.no_mlir
    if not a.app:                                            # workstation emulation
        lib = AiednnLib.build(OUT / "lib")
        r = dt.model_test(m, x, a.mode, lib, a.model, mlir=mlir)
        print(f"{r.ops} primitives, {r.on_aie} on the AIE; integer GEMMs bit-exact: "
              f"{r.gemm_checked - len(r.gemm_bad)}/{r.gemm_checked}")
        for impl, n in sorted(r.impls.items(), key=lambda kv: -kv[1]) if isinstance(r.impls, dict) else []:
            print(f"  {n:3d}  {impl}")
        print(f"{'PASS' if r.ok else 'FAIL'}: flow 4 model compiler [{a.model}] ({a.mode}{', mlir' if mlir else ''}), "
              f"rel err vs PyTorch {r.relerr:.2e}, top-1 {r.top1} (reference {r.ref_top1})")
        return 0 if r.ok else 1
    import aiegraph as ag
    d = OUT / f"flow4_{a.model}_{a.mode}"
    g, ref, yf = dt.references(m, x, a.mode, a.model, mlir=mlir)
    app = dt.emit_app(g, d, ref, dt.TOL[a.mode], dt._top1(yf, g.top1_range)[0])
    run = a.target == "x86" or bool(a.board)
    rep = ag.dnn_build(app, target=a.target, out=d / a.target, run=run, board=a.board,
                       vitis_settings=a.vitis_settings)
    log = rep.run_log.read_text(errors="replace") if run and rep.run_log and rep.run_log.exists() else ""
    print(f"app {app}\nelf {rep.elf}")
    print("\n".join(ln for ln in log.splitlines() if ln.startswith(("MODEL ", "PASS", "FAIL"))))
    ok = rep.ok and (rep.run_ok if run else True)
    print(f"{'PASS' if ok else 'FAIL'}: flow 4 model compiler, C app on {a.target}{'' if run else ' (built, not run)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
