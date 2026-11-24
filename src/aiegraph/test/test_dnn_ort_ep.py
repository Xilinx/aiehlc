###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiednn as an ONNX Runtime execution provider (aiegraph.dnn_ort_ep) vs ORT's CPU EP, workstation emulation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from aiegraph import _env  # noqa: E402

REPO = _env.repo_root()
if tuple(int(v) for v in ort.__version__.split(".")[:2]) < (1, 23):
    pytest.skip("plugin execution providers need onnxruntime >= 1.23", allow_module_level=True)
if not (REPO / "include" / "aiednn_graph.h").is_file():
    pytest.skip("partition graph API (aiednn_graph.h) not in this tree", allow_module_level=True)

from aiegraph import dnn_ort_ep  # noqa: E402


@pytest.fixture(scope="module")
def ep():
    try:
        dnn_ort_ep.register()
    except Exception as exc:  # noqa: BLE001 -- no headers offline
        pytest.skip(f"EP library not buildable here: {exc}")
    return True


def _small_cnn(path: Path, dynamic_batch: bool) -> None:
    rs = np.random.RandomState(0)
    w1 = rs.randn(8, 3, 3, 3).astype(np.float32) * 0.3
    b1 = rs.randn(8).astype(np.float32) * 0.1
    w2 = rs.randn(10, 8 * 4 * 4).astype(np.float32) * 0.1
    b2 = rs.randn(10).astype(np.float32) * 0.1
    nodes = [helper.make_node("Conv", ["x", "w1", "b1"], ["c"], pads=[1, 1, 1, 1]),
             helper.make_node("Relu", ["c"], ["r"]),
             helper.make_node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
             helper.make_node("CumSum", ["p", "ax"], ["cs"]),           # not mapped: stays on the CPU EP
             helper.make_node("Flatten", ["cs"], ["f"]),
             helper.make_node("Gemm", ["f", "w2", "b2"], ["y"], transB=1)]
    inits = [numpy_helper.from_array(a, n) for a, n in
             ((w1, "w1"), (b1, "b1"), (w2, "w2"), (b2, "b2"), (np.array(3, np.int64), "ax"))]
    xin = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N" if dynamic_batch else 1, 3, 8, 8])
    yout = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N" if dynamic_batch else 1, 10])
    m = helper.make_model(helper.make_graph(nodes, "small", [xin], [yout], inits),
                          opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9   # the onnx package writes its newest IR; onnxruntime reads up to its own
    onnx.save(m, str(path))


@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_small_cnn_matches_cpu_ep(ep, tmp_path, capfd, monkeypatch, dynamic_batch):
    monkeypatch.setenv("AIEDNN_ORT_EP_LOG", "1")
    path = tmp_path / "small.onnx"
    _small_cnn(path, dynamic_batch)
    x = np.random.RandomState(1).rand(1, 3, 8, 8).astype(np.float32)
    want = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    sess = dnn_ort_ep.session(path)
    assert sess.get_providers()[0] == "aiednn"
    got = sess.run(None, {"x": x})[0]
    err = capfd.readouterr().err
    claimed = [ln for ln in err.splitlines() if ln.startswith("[aiednn-ep]   + ")]
    assert any("conv+relu" in ln for ln in claimed), err          # the dynamic batch is pinned, so fused
    assert any("wildcard" in ln for ln in err.splitlines() if "CumSum" in ln or "unsupported" in ln)
    assert float(np.abs(got - want).max() / np.abs(want).max()) < 1e-5


def test_resnet18_onnx(ep):
    path = REPO / "worklocal" / "tvmrelay_deploy" / "resnet18-v1-7.onnx"
    if not path.is_file():
        pytest.skip("resnet18-v1-7.onnx not downloaded (deploy_flow.py)")
    x = np.random.RandomState(0).rand(1, 3, 224, 224).astype(np.float32)
    ref = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = ref.get_inputs()[0].name
    want = ref.run(None, {name: x})[0]
    got = dnn_ort_ep.session(path, benchmark=True).run(None, {name: x})[0]
    assert int(got.argmax()) == int(want.argmax())
    assert float(np.abs(got - want).max() / np.abs(want).max()) < 1e-4
