#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""A PyTorch model in aiednn's EAGER mode: one templated, shape-specialized AIE kernel instance per layer shape,
loaded right before its layer runs (README.md here explains the layers of code).

    python example/aiednn/eager/models/run.py --model gpt2_small                      # workstation emulation
    python example/aiednn/eager/models/run.py --model gpt2_small --show               # + the plan and instance code
    python example/aiednn/eager/models/run.py --model gpt2_small --mesh 8x4 --target vek280 --board galerina13 \\
        --vitis-settings <.../Vitis/settings64.sh>                                    # build the ELF, run it on the board

Models: the benchmark set of example/aiednn/models/zoo.py -- gpt2_small, tiny_llama, bert_base, resnet18, resnet50,
mobilenet_v2, yolov8n (and the rest of the zoo). The steps are the five below; run_models.py --eager does the same.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "example" / "aiednn" / "models"))

import zoo                                          # noqa: E402
from aiegraph import dnn_torch as dt                # noqa: E402
from aiegraph import dnn_eager                      # noqa: E402

OUT = REPO / "worklocal" / "aiednn" / "eager_examples"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2_small")
    ap.add_argument("--mesh", default="4x4", choices=["4x4", "6x4", "8x4"], help="the instances' mesh (rows x 4)")
    ap.add_argument("--library", action="store_true",
                    help="the premade template library (dnn_eager.LIBRARY) instead of instances made for this model")
    ap.add_argument("--target", default="x86", choices=["x86", "vek280"])
    ap.add_argument("--board", default=None, help="run the built ELF on this VEK280 board")
    ap.add_argument("--vitis-settings", default=None)
    ap.add_argument("--show", action="store_true", help="print the plan and the generated instance code")
    a = ap.parse_args()
    rows = int(a.mesh.split("x")[0])
    d = OUT / f"{a.model}_{a.mesh}{'_lib' if a.library else ''}"

    # 1. the PyTorch model and its input (zoo.py: torchvision with ImageNet weights, or the transformer classes)
    m, x = zoo.build(a.model)

    # 2. lower to aiednn primitives: int8 (calibrated on x), the aiednn MLIR pipeline (fusion, requant folds,
    #    memory plan); g is the primitive graph, ref / yf the PyTorch references the app checks against
    g, ref, yf = dt.references(m, x, "int8", a.model, mlir=True, keep_ir=d / "ir")

    # 3. the eager plan: every layer gets a template instance whose shapes / tiling / types fit it
    plan = dnn_eager.library_plan(a.model, rows) if a.library else dnn_eager.plan(g, a.model, rows)
    print(plan.summary(), flush=True)
    if a.show:
        print("\n----- generated instance entry (eager_instances.cc) -----")
        print(dnn_eager.entry_source(plan))

    # 4. the C app: plain include/aiednn.h calls, one *_create + execute per primitive
    app = dt.emit_app(g, d, ref, dt.TOL["int8"], dt._top1(yf, g.top1_range)[0])

    # 5. build in eager mode -- the instances in one aiehlc entry, each launch loads its kernel -- and run
    import aiegraph as ag
    run = a.target == "x86" or bool(a.board)
    rep = ag.dnn_build(app, target=a.target, out=d / a.target, run=run, board=a.board, timeout=3600,
                       vitis_settings=a.vitis_settings, mode="eager", eager=plan)
    log = rep.run_log.read_text(errors="replace") if run and rep.run_log and rep.run_log.exists() else ""
    print(f"app {app}\nelf {rep.elf}")
    print("\n".join(ln for ln in log.replace("\r", "").splitlines()
                    if ln.startswith(("MODEL ", "PROF eager", "PASS", "FAIL"))))
    ok = rep.ok and (rep.run_ok if run else True)
    print(f"{'PASS' if ok else 'FAIL'}: eager mode [{a.model}, {a.mesh}] on {a.target}{'' if run else ' (built, not run)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
