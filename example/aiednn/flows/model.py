###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The models flows 2-4 run. Default "tiny": a small net defined here (conv-bn-relu -> conv -> residual add ->
relu -> global pool -> fc; random weights and input) so the flows finish in seconds. Any other name comes from
example/aiednn/models/zoo.py: torchvision models with their ImageNet weights on a real photo (resnet18,
mobilenet_v2, resnet50, vit_b_16, ...), and the benchmark transformers (bert_base, gpt2_small, tiny_llama)
and yolov8n."""
import sys
from pathlib import Path

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.bn = nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        y = torch.relu(self.bn(self.c1(x)))
        y = torch.relu(self.c2(y) + y)
        return self.fc(y.mean((2, 3)))


def model_and_input(name: str = "tiny", seed: int = 0):
    """(eval model, input)"""
    if name == "tiny":
        torch.manual_seed(seed)
        return Net().eval(), torch.randn(1, 8, 16, 16)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    import zoo
    return zoo.build(name)


def model_arg(ap):
    ap.add_argument("--model", default="tiny", help="tiny (default) or a zoo model: resnet18, mobilenet_v2, vit_b_16, ...")
