#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph quickstart, ONNX: ResNet-18 -> int8 C on the A78, layer 1 on the AIE.

    pip install -e .                       # once, from the repo root
    python example/aiegraph/quickstart_onnx.py

Every step prints what it did and what to do next. Nothing runs on a board;
step 6 only links main.elf (and needs Vitis -- `aiegraph doctor` tells you).
"""

import sys
from pathlib import Path

import aiegraph as ag

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "src" / "aietensorop" / "conv2dstem" / "conv2dstem_uk.cc"


def main() -> int:
    # 0. Is this machine ready?  (fix list printed if not)
    doc = ag.doctor()
    print(doc)
    if not doc.ok_for("convert"):
        return 2

    # 1. A model: zoo name, an .onnx path, or ag.from_torch(module)
    m = ag.load("resnet18")
    print(m)

    # 2. int8 + one C file per layer + the A78 driver
    emod = ag.embedded_convert(m, target="a78")
    print(emod)

    # 3. Which layers can my kernel take?
    print(emod.detect(KERNEL))

    # 4. Prove it on x86: lint, signature+values, bit-exact vs TVM, whole network
    rep = emod.offload_check("layer1", KERNEL)
    print(rep)
    if not rep.ok:
        return 1

    # 5. aiehlc build + swap the call (needs Vitis for the build)
    if not doc.ok_for("offload"):
        print("Vitis not available: stopping after the x86 checks (see doctor above)")
        return 0
    print(emod.offload(("layer1", KERNEL)))

    # 6. Link main.elf -- copy it to a board yourself
    print(emod.build())
    ag.guide()
    return 0


if __name__ == "__main__":
    sys.exit(main())
