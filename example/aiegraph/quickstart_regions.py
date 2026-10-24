#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph quickstart, by region: ResNet-18's stem conv on the AIE, the rest on the A78.

    pip install -e .                       # once, from the repo root
    python example/aiegraph/quickstart_regions.py [region]    # default: conv1

Try ``layer1.0.conv1``: the stem kernel does not fit a 3x3 conv, so
``check_offload`` says why and ``partition`` keeps it on the A78. Nothing runs
on a board; ``build`` only links main.elf (and needs Vitis).
"""

import sys
from pathlib import Path

import aiegraph as ag

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "src" / "aietensorop" / "conv2dstem" / "conv2dstem_uk.cc"


def main(region: str = "conv1") -> int:
    model = ag.load("resnet18")
    module = ag.convert(model, target="a78")
    print(module.regions)

    # Register or describe the kernel contract.
    module.register_kernel(
        name="conv2dstem",
        source=KERNEL,
        op="conv2d",
        target="a78",
        constraints={
            "input_layout": "NCHW",
            "weight_layout": "OIHW",
            "dtype": "int8",
            "quantization": "per_channel",
        },
    )

    # Inspect whether the named graph region is compatible.
    check = module.check_offload(region=region, kernel="conv2dstem")
    print(check.summary())

    # Explicit assignment for bring-up; later this can be automatic.
    module.assign(region=region, backend="aie:conv2dstem")

    # Make fallback policy explicit.
    plan = module.partition(fallback="cpu", optimize_for="latency")
    print(plan.report())

    artifact = module.build(target="a78", output="resnet18_a78")
    print(artifact)
    return 0 if artifact.ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
