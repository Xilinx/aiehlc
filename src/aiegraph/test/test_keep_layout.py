###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GAP-B: partition.json ``keep_layout`` (aiegraph.partition.parse_keep_layout / plan_keep_layout).

x86 only, small synthetic ONNX graphs: an AIE conv output kept in the AIE layout reaches its AIE consumer through
Transpose / Reshape that are labels only; a host consumer, a graph output or a reshape that would move data
is rejected (never silently ignored).
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph.errors import AiegraphError  # noqa: E402
from aiegraph.partition import parse_keep_layout, plan_keep_layout  # noqa: E402


def _graph(tail, C=8, H=4, W=4):
    """x -> Conv c0 [1,C,H,W] -> tail nodes ... ; tail(list of NodeProto, vinfos, outputs)."""
    nodes, vi, outs = tail
    w = helper.make_tensor("w", TensorProto.FLOAT, [C, C, 1, 1], np.zeros(C * C, np.float32))
    g = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y0"], name="c0")] + nodes, "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, C, H, W])],
        [helper.make_tensor_value_info(o, TensorProto.FLOAT, s) for o, s in outs],
        [w], value_info=[helper.make_tensor_value_info(n, TensorProto.FLOAT, s) for n, s in vi])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])


def _shape(name, v):
    return helper.make_tensor(name, TensorProto.INT64, [len(v)], np.array(v, np.int64))


def _attn_like(consumer_on_aie=True, perm=(0, 2, 3, 1)):
    """UNet proj_in pattern: conv -> Transpose(0,2,3,1) -> Reshape [1,HW,C] -> MatMul (to_q)."""
    C, HW = 8, 16
    t_shape = [1, 8, 4, 4] if list(perm) == [0, 1, 2, 3] else [[1, 8, 4, 4][p] for p in perm]
    nodes = [helper.make_node("Transpose", ["y0"], ["t"], name="tr", perm=list(perm)),
             helper.make_node("Constant", [], ["s"], name="cs", value=_shape("s", [1, HW, C])),
             helper.make_node("Reshape", ["t", "s"], ["r"], name="rs"),
             helper.make_node("Constant", [], ["wq"], name="cq",
                              value=helper.make_tensor("wq", TensorProto.FLOAT, [C, C], np.zeros(C * C))),
             helper.make_node("MatMul", ["r", "wq"], ["q"], name="to_q"),
             helper.make_node("Relu", ["q"], ["z"], name="act")]
    vi = [("y0", [1, C, 4, 4]), ("t", t_shape), ("r", [1, HW, C]), ("q", [1, HW, C])]
    m = _graph((nodes, vi, [("z", [1, HW, C])]))
    rows = [{"node": "c0", "cls": "conv", "keep_layout": {"layout": "nhwc"}}]
    if consumer_on_aie:
        rows.append({"node": "to_q", "cls": "linear"})
    return m, {"layers": rows}


def test_parse():
    assert parse_keep_layout({"node": "a"}) is None
    assert parse_keep_layout({"node": "a", "keep_layout": False}) is None
    assert parse_keep_layout({"node": "a", "keep_layout": True}) == {"layout": "nchw"}
    assert parse_keep_layout({"node": "a", "keep_layout": "nhwc"}) == {"layout": "nhwc"}
    assert parse_keep_layout({"node": "a", "keep_layout": {"layout": "nhwc"}}) == {"layout": "nhwc"}
    for bad in ({"layout": "nc4hw"}, {"layout": "nhwc", "x": 1}, 3):
        with pytest.raises(AiegraphError):
            parse_keep_layout({"node": "a", "keep_layout": bad})


def test_no_field_is_noop():
    # no row asks for it -> nothing to plan, the model is not even loaded
    assert plan_keep_layout(None, {"layers": [{"node": "c0"}]}) == dict(edges=[], label_ops=0, kept=0)


def test_transpose_reshape_are_labels():
    m, part = _attn_like()
    rep = plan_keep_layout(m, part)
    assert rep["kept"] == 1 and rep["label_ops"] == 2
    e = rep["edges"][0]
    assert e["node"] == "c0" and e["layout"] == "nhwc" and e["label_ops"] == ["tr", "rs"]
    assert e["consumers"] == ["to_q"] and e["views"] == [[1, 16, 8]]


def test_label_proof_numpy():
    """The bytes of the AIE buffer in nhwc ARE the row-major [HW][C] the MatMul reads (no data moves)."""
    y = np.arange(8 * 16, dtype=np.float32).reshape(1, 8, 4, 4)
    aie_bytes = np.ascontiguousarray(y.transpose(0, 2, 3, 1)).reshape(-1)    # kept nhwc image
    host = y.transpose(0, 2, 3, 1).reshape(1, 16, 8)                          # Transpose + Reshape on the host
    assert np.array_equal(aie_bytes.reshape(1, 16, 8), host)
    # in nchw the same reinterpretation is wrong -> plan_keep_layout must reject it (next test)
    assert not np.array_equal(np.ascontiguousarray(y).reshape(1, 16, 8), host)


def test_nchw_needs_real_transpose():
    m, part = _attn_like()
    part["layers"][0]["keep_layout"] = True        # nchw: the Reshape merges H,W of a [N,H,W,C] view -> moves data
    with pytest.raises(AiegraphError, match="moves data|real transpose"):
        plan_keep_layout(m, part)


def test_host_consumer_rejected():
    m, part = _attn_like(consumer_on_aie=False)
    with pytest.raises(AiegraphError, match="MatMul.*host"):
        plan_keep_layout(m, part)


def test_graph_output_rejected():
    m = _graph(([], [], [("y0", [1, 8, 4, 4])]))
    with pytest.raises(AiegraphError, match="graph output"):
        plan_keep_layout(m, {"layers": [{"node": "c0", "keep_layout": True}]})


def test_direct_conv_chain_nchw():
    """conv -> conv, both on the AIE, nchw kept: zero label ops, the consumer reads [1,C,H,W] as is."""
    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [8, 8, 1, 1], np.zeros(64, np.float32))
    nodes = [helper.make_node("Constant", [], ["w1c"], name="cw", value=w1),
             helper.make_node("Conv", ["y0", "w1c"], ["y1"], name="c1")]
    m = _graph((nodes, [("y0", [1, 8, 4, 4])], [("y1", [1, 8, 4, 4])]))
    rep = plan_keep_layout(m, {"layers": [{"node": "c0", "keep_layout": True}, {"node": "c1"}]})
    assert rep["edges"][0]["label_ops"] == [] and rep["edges"][0]["consumers"] == ["c1"]
    # nhwc kept but c1 reads NCHW -> real transpose
    with pytest.raises(AiegraphError, match="real transpose"):
        plan_keep_layout(m, {"layers": [{"node": "c0", "keep_layout": "nhwc"}, {"node": "c1"}]})


def test_unknown_node():
    m, part = _attn_like()
    part["layers"][0]["node"] = "nope"
    with pytest.raises(AiegraphError, match="not in the model"):
        plan_keep_layout(m, part)
