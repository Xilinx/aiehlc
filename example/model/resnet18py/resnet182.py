###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=strice, padding=1, bias=False)

def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)

class BasicBlockV2(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch, eps=1e-5)
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=1e-5)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
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
        downsample = stride != 1 or in_ch != out_ch
        return nn.Sequential(
            BasicBlockV2(in_ch, out_ch, stride=stride, downsample=downsample),
            BasicBlockV2(out_ch, out_ch, stride=1, downsample=False),   
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool(F.relu(self.bn1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x