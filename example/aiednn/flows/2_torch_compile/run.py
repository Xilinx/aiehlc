#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Flow 2 -- torch.compile(model, backend="aiednn"), checked against eager (workstation emulation).

    python example/aiednn/flows/2_torch_compile/run.py [--model resnet18]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                   # noqa: E402

from model import model_and_input, model_arg   # noqa: E402
from aiegraph import dnn_torch_backend as be   # noqa: E402
from aiegraph.backends import aiednn as flags  # noqa: E402  (registers backend="aiednn")


def main() -> int:
    ap = argparse.ArgumentParser()
    model_arg(ap)
    a = ap.parse_args()
    m, x = model_and_input(a.model)
    flags.graph = True
    flags.benchmark = False
    torch._dynamo.reset()
    with torch.no_grad():
        got = torch.compile(m, backend="aiednn")(x)
        want = m(x)
    fg = be.aiednn_backend.last.aiednn_graph
    print(fg.report())
    rel = float((got - want).abs().max() / want.abs().max())
    top, wtop = int(got.flatten().argmax()), int(want.flatten().argmax())
    ok = rel < 1e-3 and top == wtop
    print(f"{'PASS' if ok else 'FAIL'}: flow 2 torch.compile [{a.model}], max rel diff vs eager {rel:.2e}, "
          f"top-1 {top} (eager {wtop})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
