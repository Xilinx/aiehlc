#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph quickstart, PyTorch: a torch.nn.Module through the same flow.

    pip install -e ".[torch]"              # torchvision for the model
    python example/aiegraph/quickstart_torch.py

`ag.from_torch` exports the module to ONNX (opset 13) and joins the ONNX path,
so from step 2 on this is identical to quickstart_onnx.py. Here the whole
thing is ONE call: ag.compile(model, target, offload).

Expected outcome with torchvision's IMAGENET1K_V1 weights: layer1 is REFUSED
at detect -- `arg 7 (shift): require=range(-30,32), but values span [7, 50]`.
Seven stem channels quantize to a requant shift of 36..50, i.e. an int64 shift
by 67..81 bits: undefined behaviour in C, in TVM's own layer code as much as
in the AIE epilogue, so nothing can be bit-exact with it. strict=False keeps
the layer on the A78 and still links main.elf; that is the gate doing its job.
(The ONNX zoo ResNet-18 in quickstart_onnx.py has every shift in range.)
"""

import sys
from pathlib import Path

import torch
import torchvision

import aiegraph as ag

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "src" / "aietensorop" / "conv2dstem" / "conv2dstem_uk.cc"


def main() -> int:
    net = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    m = ag.from_torch(net, example_inputs=(torch.randn(1, 3, 224, 224),), name="resnet18_torch")
    print(m)

    # convert + offload_check + offload + build in one call. With
    # strict=False a layer that fails a gate stays on the A78 instead of raising.
    target = "aie" if ag.doctor().ok_for("offload") else "x86"
    emod = ag.compile(m, target, f"layer1={KERNEL}" if target == "aie" else None,
                      strict=False)
    print(emod)
    return 0


if __name__ == "__main__":
    sys.exit(main())
