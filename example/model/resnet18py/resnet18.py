###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""ResNet-18 (v2 / pre-activation) implemented from scratch in PyTorch.

This matches the ONNX Model Zoo `resnet18-v2-7` model (MXNet Gluon `resnetv22`):
the pre-activation variant where each block is BN -> ReLU -> Conv, with an extra
BatchNorm applied directly to the input image and a final BN -> ReLU before the
global average pool.

    input
      -> BN0 (3 ch, input normalisation)
      -> Conv 7x7 / stride 2       (3 -> 64)
      -> BN -> ReLU -> MaxPool 3x3 / stride 2
      -> stage1: 2 x BasicBlockV2  (64,  stride 1)
      -> stage2: 2 x BasicBlockV2  (128, stride 2 in first block)
      -> stage3: 2 x BasicBlockV2  (256, stride 2 in first block)
      -> stage4: 2 x BasicBlockV2  (512, stride 2 in first block)
      -> BN -> ReLU -> GlobalAvgPool -> Linear(512 -> 1000)

Weights are loaded from the ONNX file by name (see `load_onnx_weights`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlockV2(nn.Module):
    """Pre-activation basic residual block.

    forward:
        preact = relu(bn1(x))
        out    = conv2(relu(bn2(conv1(preact))))
        skip   = downsample(preact) if downsample else x
        return out + skip
    """

    def __init__(self, in_ch, out_ch, stride=1, downsample=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-5)
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        # 1x1 projection on the *pre-activated* input, used when shape changes.
        self.downsample = conv1x1(in_ch, out_ch, stride) if downsample else None

    def forward(self, x):
        preact = F.relu(self.bn1(x))
        out = self.conv1(preact)
        out = self.conv2(F.relu(self.bn2(out)))
        skip = self.downsample(preact) if self.downsample is not None else x
        return out + skip


class ResNet18V2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # BN applied to the raw 3-channel input (learned input normalisation).
        self.bn0 = nn.BatchNorm2d(3, eps=1e-5)
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(64, 64, stride=1)
        self.stage2 = self._make_stage(64, 128, stride=2)
        self.stage3 = self._make_stage(128, 256, stride=2)
        self.stage4 = self._make_stage(256, 512, stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, stride):
        # Two blocks per stage; the first may downsample / change channels.
        downsample = (stride != 1) or (in_ch != out_ch)
        return nn.ModuleList([
            BasicBlockV2(in_ch, out_ch, stride=stride, downsample=downsample),
            BasicBlockV2(out_ch, out_ch, stride=1, downsample=False),
        ])

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv0(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                x = block(x)
        x = F.relu(self.bn2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _onnx_initializers(onnx_path):
    import onnx
    from onnx import numpy_helper
    model = onnx.load(onnx_path)
    return {t.name: torch.from_numpy(numpy_helper.to_array(t).copy())
            for t in model.graph.initializer}


def _bn_names(prefix):
    return {
        "weight": f"{prefix}_gamma",
        "bias": f"{prefix}_beta",
        "running_mean": f"{prefix}_running_mean",
        "running_var": f"{prefix}_running_var",
    }


def load_onnx_weights(model, onnx_path):
    """Load resnet18-v2 ONNX initializers into `model` (matched by name)."""
    W = _onnx_initializers(onnx_path)
    sd = {}

    # Stem.
    for k, v in _bn_names("resnetv22_batchnorm0").items():
        sd[f"bn0.{k}"] = W[v]
    sd["conv0.weight"] = W["resnetv22_conv0_weight"]
    for k, v in _bn_names("resnetv22_batchnorm1").items():
        sd[f"bn1.{k}"] = W[v]

    # Stages. ONNX uses per-stage running counters for bn/conv; the first block
    # of stages 2-4 carries an extra 1x1 downsample conv.
    for s, stage_name in enumerate(["stage1", "stage2", "stage3", "stage4"], start=1):
        stage = getattr(model, stage_name)
        p = f"resnetv22_stage{s}"
        conv_i = 0
        bn_i = 0
        for b, block in enumerate(stage):
            for k, v in _bn_names(f"{p}_batchnorm{bn_i}").items():
                sd[f"{stage_name}.{b}.bn1.{k}"] = W[v]
            bn_i += 1
            sd[f"{stage_name}.{b}.conv1.weight"] = W[f"{p}_conv{conv_i}_weight"]
            conv_i += 1
            for k, v in _bn_names(f"{p}_batchnorm{bn_i}").items():
                sd[f"{stage_name}.{b}.bn2.{k}"] = W[v]
            bn_i += 1
            sd[f"{stage_name}.{b}.conv2.weight"] = W[f"{p}_conv{conv_i}_weight"]
            conv_i += 1
            if block.downsample is not None:
                sd[f"{stage_name}.{b}.downsample.weight"] = W[f"{p}_conv{conv_i}_weight"]
                conv_i += 1

    # Head.
    for k, v in _bn_names("resnetv22_batchnorm2").items():
        sd[f"bn2.{k}"] = W[v]
    sd["fc.weight"] = W["resnetv22_dense0_weight"]
    sd["fc.bias"] = W["resnetv22_dense0_bias"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The only acceptable "missing" keys are BN batch counters (not in ONNX).
    bad = [m for m in missing if not m.endswith("num_batches_tracked")]
    if bad:
        raise RuntimeError(f"Unfilled parameters: {bad}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys built from ONNX: {unexpected}")
    return model


def resnet18(onnx_path=None, num_classes=1000):
    """Build ResNet18-v2; if `onnx_path` given, load pretrained ImageNet weights."""
    model = ResNet18V2(num_classes=num_classes)
    if onnx_path is not None:
        load_onnx_weights(model, onnx_path)
    return model.eval()
