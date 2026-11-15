###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GAP-aiegraph-partition-9759be: the UNet conv family conv(+bias)(+add), no ReLU.

x86 only: no TVM build, no board. ``aiegraph.partition.match_conv`` must map the
UNet conv forms (bias, + timestep add, + skip / shortcut / attention residual)
onto aiegraph ops without dropping the add or inventing a ReLU, keep the
ResNet-18 (QNN and float) mapping, and the epilogue must run in fp32 before the
output cast.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph.partition import (AIEGRAPH_OP_KINDS, ConvPattern, conv_epilogue_ref,  # noqa: E402
                                match_conv)

UNET_ONNX = Path("/scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx")


# ── pattern family ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ops,aiegraph,adds,relu", [
    (["conv2d"], ("conv_bn",), (), False),
    (["conv2d", "bias_add"], ("conv_bn",), (), False),
    (["conv2d", "bias_add", "relu"], ("conv_bn_relu",), (), True),
    (["conv2d", "bias_add", "add"], ("conv_bn",), ("join",), False),
    (["conv2d", "bias_add", "add", "add"], ("conv_bn",), ("join", "join"), False),
    (["conv2d", "bias_add", "add", "relu"], ("conv_bn", "residual_add_relu"), ("join",), True),
    # TVM fused-name tokens: NCHWc legalizes bias_add to a plain add
    (["contrib", "conv2d", "NCHWc", "add"], ("conv_bn",), (), False),
    (["contrib", "conv2d", "NCHWc", "add", "add"], ("conv_bn",), ("join",), False),
    (["nn", "conv2d", "nn", "bias", "add", "add"], ("conv_bn",), ("join",), False),
    # ResNet-18 float (BN folded into add)
    (["conv2d", "add", "relu"], ("conv_bn_relu",), (), True),
    (["conv2d", "add", "add", "relu"], ("conv_bn", "residual_add_relu"), ("join",), True),
])
def test_family(ops, aiegraph, adds, relu):
    p = match_conv(ops)
    assert p is not None
    assert p.aiegraph == aiegraph and p.adds == adds and p.relu == relu
    assert all(k in AIEGRAPH_OP_KINDS for k in p.aiegraph)
    assert p.epilogue_fp32


def test_unet_join_never_gets_a_relu():
    # 1c7316a bug class: a no-relu join must not become residual_add_relu
    for ops in (["conv2d", "bias_add", "add"], ["contrib", "conv2d", "NCHWc", "add", "add"]):
        assert "residual_add_relu" not in match_conv(ops).aiegraph


@pytest.mark.parametrize("ops", [
    ["dense", "add"],
    ["conv2d", "bias_add", "multiply"],          # foreign op fused in: never dropped
    ["conv2d", "bias_add", "sigmoid", "multiply"],
    ["conv2d", "bias_add", "add", "add", "relu"],  # relu after two joins: ambiguous
    ["conv2d", "bias_add", "add", "add", "add"],   # > MAX_EPILOGUE_ADDS
])
def test_unmapped(ops):
    assert match_conv(ops) is None


def test_qnn_resnet_unchanged():
    tail = ["contrib", "conv2d", "NCHWc", "add", "subtract", "fixed", "point", "multiply",
            "per", "axis", "add", "cli", "18f7ea1f"]
    assert match_conv(tail).aiegraph == ("conv_bn", "residual_add_relu")
    conv = ["contrib", "conv2d", "NCHWc", "add", "subtract", "fixed", "point", "multiply",
            "per", "axis", "clip", "ca", "a7e8b96f"]
    assert match_conv(conv).aiegraph == ("conv_bn",)


def test_add_shapes_typed():
    out = [1, 320, 32, 32]
    p = match_conv(["conv2d", "bias_add", "add"], [[1, 320, 1, 1]], out)
    assert p.adds == ("channel",) and p.family == "conv+bias+add:channel"
    p = match_conv(["conv2d", "bias_add", "add"], [out], out)
    assert p.adds == ("tensor",)
    # NCHWc output, per-channel operand in the same blocking
    p = match_conv(["conv2d", "add", "add"], [[1, 20, 1, 1, 16]], [1, 20, 32, 32, 16])
    assert p.adds == ("channel",)
    # spatial broadcast / wrong channel count: not this family
    assert match_conv(["conv2d", "bias_add", "add"], [[1, 1, 32, 32]], out) is None
    assert match_conv(["conv2d", "bias_add", "add"], [[1, 640, 1, 1]], out) is None
    # operand count must match the joins
    assert match_conv(["conv2d", "bias_add", "add"], [], out) is None


# ── fp32 epilogue before the cast ────────────────────────────────────────────
def test_epilogue_fp32_before_bf16_cast():
    rng = np.random.default_rng(0)
    acc = rng.standard_normal((1, 8, 4, 4)).astype(np.float32) * 3
    bias = rng.standard_normal(8).astype(np.float32)
    temb = rng.standard_normal(8).astype(np.float32)
    skip = rng.standard_normal((1, 8, 4, 4)).astype(np.float32) * 7
    want = acc + bias.reshape(1, 8, 1, 1) + temb.reshape(1, 8, 1, 1) + skip
    y32 = conv_epilogue_ref(acc, bias, (temb, skip))
    np.testing.assert_array_equal(y32, want)
    yb = conv_epilogue_ref(acc, bias, (temb, skip), out_dtype="bf16")
    # exactly one RNE rounding of the fp32 sum ...
    u = yb.view(np.uint32)
    assert np.all((u & 0xFFFF) == 0)
    assert np.max(np.abs(yb - want) / np.maximum(np.abs(want), 1e-30)) <= 2.0 ** -8
    # ... which differs from casting the conv first and adding after
    cast_first = conv_epilogue_ref(conv_epilogue_ref(acc, bias, out_dtype="bf16"),
                                   None, (temb, skip), out_dtype="bf16")
    assert not np.array_equal(cast_first, yb)
    cos = float(np.dot(yb.ravel(), want.ravel())
                / np.linalg.norm(yb) / np.linalg.norm(want))
    assert cos > 0.99999


# ── the tvmrelay partition keeps the add (never offloaded without epilogue) ──
def test_tvmrelay_partition_keeps_epilogue_on_cpu():
    from frontend.tvmrelay.aiegraph_partition import partition_layers

    recs = [dict(index=1, aiegraph=["conv_bn"], epilogue=["join"]),
            dict(index=2, aiegraph=["conv_bn"]),
            dict(index=3, aiegraph=["conv_bn", "residual_add_relu"])]
    v = partition_layers(recs)
    assert [x["target"] for x in v] == ["cpu", "aie", "cpu"]
    assert "fp32 epilogue add" in v[0]["verdict_reason"]
    v = partition_layers(recs, epilogue=True)
    assert v[0]["target"] == "aie"


# ── ONNX walk (synthetic UNet resnet block, then the real UNet if present) ────
def _toy_resnet_onnx():
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    C = 4
    w = lambda n, *s: numpy_helper.from_array(np.zeros(s, np.float32), n)  # noqa: E731
    nodes = [
        helper.make_node("Conv", ["x", "w1", "b1"], ["c1"], name="conv1", pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["c1", "temb"], ["h1"], name="add_temb"),
        helper.make_node("Relu", ["h1"], ["r1"], name="act"),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], name="conv2", pads=[1, 1, 1, 1]),
        helper.make_node("Add", ["x", "c2"], ["y"], name="add_skip"),
        helper.make_node("Conv", ["y", "w3", "b3"], ["c3"], name="conv3"),
        helper.make_node("Sigmoid", ["c3"], ["z"], name="sig"),
    ]
    g = helper.make_graph(
        nodes, "toy",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, 8, 8]),
         helper.make_tensor_value_info("temb", TensorProto.FLOAT, [1, C, 1, 1])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, C, 8, 8])],
        [w("w1", C, C, 3, 3), w("b1", C), w("w2", C, C, 3, 3), w("b2", C),
         w("w3", C, C, 1, 1), w("b3", C)])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def test_onnx_walk_toy():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probe_unet_patterns import conv_patterns

    rows = {r["name"]: r for r in conv_patterns(_toy_resnet_onnx())}
    # conv1 + per-channel add + relu: residual_add_relu needs a same-shape join,
    # so it stays unmapped instead of being claimed as a residual tail
    assert rows["conv1"]["pattern"] is None and not rows["conv1"]["mapped"]
    assert rows["conv2"]["pattern"].family == "conv+bias+add:tensor"
    assert rows["conv2"]["pattern"].aiegraph == ("conv_bn",)
    assert rows["conv3"]["pattern"].family == "conv+bias"   # the sigmoid is not fused
    assert all(r["mapped"] for r in rows.values() if r["name"] != "conv1")


@pytest.mark.skipif(not UNET_ONNX.is_file(), reason="bk-sdm-tiny unet_sim.onnx not present")
def test_real_unet_all_convs_mapped():
    import onnx
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probe_unet_patterns import conv_patterns

    rows = conv_patterns(onnx.load(str(UNET_ONNX)))
    assert len(rows) == 50
    assert [r["name"] for r in rows if not r["mapped"]] == []
    adds = [a for r in rows for a in r["pattern"].adds]
    assert adds.count("channel") > 0 and adds.count("tensor") > 0
    assert not any(r["pattern"].relu for r in rows)
