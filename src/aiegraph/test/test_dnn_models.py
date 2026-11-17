###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Whole models through aiednn (``aiegraph.dnn_torch``) against PyTorch, on x86.

    python3 -m pytest -q src/aiegraph/test/test_dnn_models.py

Random weights / inputs (no download): the custom networks of
example/aiednn/models/zoo.py and a few torchvision architectures. f32 must match
the float model; int8 must match the same model with the same fake quantization,
with every integer GEMM layer recomputed exactly from its own inputs. The
pretrained / board runs are example/aiednn/models/run_models.py.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "example" / "aiednn" / "models"))

import zoo  # noqa: E402

from aiegraph import dnn_torch as dt  # noqa: E402
from aiegraph.dnn_ctypes import AiednnLib  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
MODELS = list(zoo.CUSTOM) + ["mobilenet_v2", "squeezenet1_1", "regnet_x_400mf", "mobilenet_v3_small"]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn"))


@pytest.mark.parametrize("mode", ["f32", "int8"])
@pytest.mark.parametrize("name", MODELS)
def test_model_matches_pytorch(lib, name, mode):
    m, x = zoo.build(name, pretrained=False)
    r = dt.model_test(m, x, mode, lib, name)
    assert r.ok, r
    if mode == "int8":
        assert r.gemm_checked > 0 and r.on_aie > 0, r       # the AIE path was used and verified
    else:  # f32: its matmuls take the AIE bf16 mode as bf16x3 (near f32), the rest ref:
        assert r.relerr < 1e-4, r


def test_conv_bn_folded_and_views_are_free(lib):
    m, x = zoo.build("tiny_unet", pretrained=False)
    g = dt.lower(m, x, "f32")
    fns = [o.fn for o in g.ops]
    assert "aiednn_batch_normalization_forward_inference_create" not in fns   # Conv+BN folded
    assert fns.count("aiednn_deconvolution_forward_create") == 1
    v = dt.lower(*zoo.build("tiny_vit", pretrained=False), "f32")
    assert any(t.base is not None for t in v.tensors)                      # reshape / flatten: aliases


def test_unsupported_node_is_named():
    class Bad(torch.nn.Module):
        def forward(self, x):
            return torch.cumsum(x, 1)
    with pytest.raises(Exception, match="cumsum"):
        dt.lower(Bad(), torch.randn(1, 4), "f32")


def test_layer_diff_points_at_layers(lib):
    m, x = zoo.build("mixer_head", pretrained=False)
    d = dt.layer_diff(m, x, "f32", lib)
    assert d and all(rel < 1e-4 for _, rel, _ in d)


def test_emitted_app_runs_on_x86(tmp_path):
    import aiegraph as ag
    m, x = zoo.build("tiny_unet", pretrained=False)
    g, ref, yf = dt.references(m, x, "int8", "tiny_unet")
    app = dt.emit_app(g, tmp_path, ref, dt.TOL["int8"], int(yf.argmax()))
    rep = ag.dnn_build(app, target="x86", out=tmp_path / "x86", run=True)
    assert rep.ok and rep.run_ok, rep
    text = rep.run_log.read_text()
    assert "gemm_checked=7 gemm_bad=0" in text and "PASS: aiednn model tiny_unet [int8]" in text


def test_emitted_mlir_app_with_int8_residuals_runs_on_x86(tmp_path):
    """the MLIR pipeline's int8 skip streams (--aiednn-quantize-residual) survive the C app:
    each binary post-op carries its s8 src1's scale / zero point, and the app matches"""
    import aiegraph as ag
    m, x = zoo.build("resnet18", pretrained=False)
    g, ref, yf = dt.references(m, x, "int8", "resnet18", mlir=True)
    app = dt.emit_app(g, tmp_path, ref, dt.TOL["int8"], int(yf.argmax()))
    src = app.read_text() if app.is_file() else "".join(p.read_text() for p in app.glob("*.cc"))
    assert "aiednn_post_op_binary, aiednn_binary_add, 0.0f" not in src.replace("0.0f, 0.0f, {4", "")  # skips scaled
    rep = ag.dnn_build(app, target="x86", out=tmp_path / "x86", run=True)
    assert rep.ok and rep.run_ok, rep
    assert "PASS: aiednn model resnet18 [int8]" in rep.run_log.read_text()
