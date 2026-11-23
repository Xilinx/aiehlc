###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Unit tests for aiegraph_partition._aiegraph_ops_for (no TVM needed)."""
import os
import sys
import unittest

os.environ.setdefault("AIEHLC_TVM_AUTO_INSTALL", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from frontend.tvmrelay.aiegraph_partition import _aiegraph_ops_for  # noqa: E402


class AiegraphOpsForTest(unittest.TestCase):
    # ResNet-18 path: BN folded into a plain `add` bias.
    def test_resnet_conv_bias_relu(self):
        self.assertEqual(_aiegraph_ops_for("", ["conv2d", "add", "relu"]),
                         ["conv_bn_relu"])

    def test_resnet_conv_bias_only(self):
        self.assertEqual(_aiegraph_ops_for("", ["conv2d", "add"]), ["conv_bn"])

    def test_resnet_residual_tail(self):
        self.assertEqual(
            _aiegraph_ops_for("", ["conv2d", "add", "add", "relu"]),
            ["conv_bn", "residual_add_relu"])

    # ONNX/PyTorch import path: bias arrives as `bias_add`.
    def test_bias_add_conv(self):
        self.assertEqual(_aiegraph_ops_for("", ["conv2d", "bias_add"]),
                         ["conv_bn"])

    def test_bias_add_conv_relu(self):
        self.assertEqual(_aiegraph_ops_for("", ["conv2d", "bias_add", "relu"]),
                         ["conv_bn_relu"])

    def test_bias_add_plus_join_without_relu_is_conv_bn_epilogue(self):
        # UNet skip / timestep / attention residual: conv_bn with an fp32
        # epilogue add (aiegraph.partition.match_conv); partition_layers keeps
        # it off a backend without an epilogue, so the join is never dropped.
        self.assertEqual(
            _aiegraph_ops_for("", ["conv2d", "bias_add", "add"]), ["conv_bn"])

    def test_bias_add_plus_join_with_relu(self):
        self.assertEqual(
            _aiegraph_ops_for("", ["conv2d", "bias_add", "add", "relu"]),
            ["conv_bn", "residual_add_relu"])

    def test_qnn_resnet_residual_keeps_mapping(self):
        # Real deploy-manifest tokens: ReLU is a (truncated) requant clip.
        ops = ["contrib", "conv2d", "NCHWc", "add", "subtract", "fixed",
               "point", "multiply", "per", "axis", "add", "cli", "18f7ea1f"]
        self.assertEqual(_aiegraph_ops_for("", ops),
                         ["conv_bn", "residual_add_relu"])

    def test_qnn_resnet_conv_unchanged(self):
        ops = ["contrib", "conv2d", "NCHWc", "add", "subtract", "fixed",
               "point", "multiply", "per", "axis", "clip", "ca", "a7e8b96f"]
        self.assertEqual(_aiegraph_ops_for("", ops), ["conv_bn"])

    # Name-token form of bias_add: "nn_conv2d_nn_bias_add_add".
    def test_name_tokens_bias_add_plus_join_is_conv_bn_epilogue(self):
        ops = ["nn", "conv2d", "nn", "bias", "add", "add"]
        self.assertEqual(_aiegraph_ops_for("", ops), ["conv_bn"])

    def test_name_tokens_bias_add_only(self):
        ops = ["nn", "conv2d", "nn", "bias", "add"]
        self.assertEqual(_aiegraph_ops_for("", ops), ["conv_bn"])

    def test_two_joins_unmapped(self):
        self.assertEqual(
            _aiegraph_ops_for("", ["conv2d", "bias_add", "add", "add", "relu"]),
            [])

    def test_no_conv(self):
        self.assertEqual(_aiegraph_ops_for("", ["dense", "add"]), [])

    def test_bare_conv(self):
        self.assertEqual(_aiegraph_ops_for("", ["conv2d"]), ["conv_bn"])


if __name__ == "__main__":
    unittest.main()
