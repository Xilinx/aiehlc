###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The aiednn MLIR dialect as the model IR (``aiegraph.dnn_mlir``): bit-identical round trip."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "example" / "aiednn" / "models"))

import zoo  # noqa: E402

from aiegraph import dnn_mlir as dm  # noqa: E402
from aiegraph import dnn_torch as dt  # noqa: E402
from aiegraph.dnn_ctypes import AiednnLib  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402

pytestmark = [pytest.mark.skipif(not shutil.which("g++"), reason="needs g++"),
              pytest.mark.skipif(dm.aiednn_opt_path() is None, reason="aiednn-opt not built")]
MODELS = list(zoo.CUSTOM) + ["mobilenet_v3_small", "squeezenet1_1"]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn"))


def _lower(name, mode):
    m, x = zoo.build(name, pretrained=False)
    if mode == "f32":
        return dt.lower(m, x, mode), x
    gm = dt.prepare(m, x)
    return dt.lower(m, x, mode, gm=gm, qscale=dt.calibrate(gm, x)), x


@pytest.mark.parametrize("mode", ["f32", "int8"])
@pytest.mark.parametrize("name", MODELS)
def test_round_trip_is_bit_exact(lib, name, mode):
    g, x = _lower(name, mode)
    text, store = dm.to_mlir(g)
    g2 = dm.from_mlir(text, store, g.name, mode)
    assert [(o.name, o.fn) for o in g.ops] == [(o.name, o.fn) for o in g2.ops]
    y1, _ = dt.run_ctypes(g, lib, x.numpy())
    y2, _ = dt.run_ctypes(g2, lib, x.numpy())
    assert np.array_equal(y1.view(np.uint8), y2.view(np.uint8))


def test_ir_names_ops_and_fusion():
    g, _ = _lower("mixer_head", "f32")
    text, store = dm.to_mlir(g)
    assert '"aiednn.conv"' in text and 'loc("c2")' in text and "#aiednn.post_op<" in text
    assert all(isinstance(a, np.ndarray) for a in store)
    printed = dm.opt(text)                                  # parse + verify + print (custom syntax)
    assert "aiednn.conv" in printed and "post_ops = [#aiednn.post_op<" in printed


def test_verifier_rejects_bad_conv():
    bad = """
module {
  func.func @m(%x: tensor<1x8x4x4xf32>) -> tensor<1x6x4x4xf32> {
    %w = aiednn.const <{index = 0 : i64}> : tensor<6x3x3x3xf32>
    %y = aiednn.conv %x, %w <{strides = array<i64: 1, 1>, pad_l = array<i64: 1, 1>, pad_r = array<i64: 1, 1>,
           dilations = array<i64: 0, 0>, groups = 1 : i64}>
         : (tensor<1x8x4x4xf32>, tensor<6x3x3x3xf32>) -> tensor<1x6x4x4xf32>
    func.return %y : tensor<1x6x4x4xf32>
  }
}
"""
    with pytest.raises(AiegraphError, match=r"weights must be \[O, C/groups"):
        dm.opt(bad)


@pytest.mark.parametrize("name", ["tiny_vit", "tiny_unet", "mobilenet_v3_small"])
def test_fuse_epilogue_pass_matches_python_fusion(lib, name):
    """lowering unfused + --aiednn-fuse-epilogue == the Python epilogue fusion"""
    m, x = zoo.build(name, pretrained=False)
    gm = dt.prepare(m, x)
    q = dt.calibrate(gm, x)
    gf = dt.lower(m, x, "int8", gm=gm, qscale=q, fuse=True)
    gu = dt.lower(m, x, "int8", gm=gm, qscale=q, fuse=False)
    text, store = dm.to_mlir(gu)
    g2 = dm.from_mlir(text, store, gu.name, "int8", passes=["--aiednn-fuse-epilogue"])
    assert len(g2.ops) <= len(gf.ops) < len(gu.ops)          # MLIR may fuse more (tiny_vit), never less
    if len(g2.ops) == len(gf.ops):
        assert [(o.name, o.fn) for o in gf.ops] == [(o.name, o.fn) for o in g2.ops]
        y1, _ = dt.run_ctypes(gf, lib, x.numpy())
        y2, _ = dt.run_ctypes(g2, lib, x.numpy())
        assert np.array_equal(y1.view(np.uint8), y2.view(np.uint8))


@pytest.mark.parametrize("name", ["mobilenet_v3_small", "tiny_unet", "tiny_vit"])
def test_pipeline_int8(lib, name):
    """The default pipeline: fewer primitives, exact integer GEMMs, a smaller bit-identical arena."""
    m, x = zoo.build(name, pretrained=False)
    rp = dt.model_test(m, x, "int8", lib, name)
    r = dt.model_test(m, x, "int8", lib, name, mlir=True)
    assert r.ok and not r.gemm_bad and r.gemm_checked == rp.gemm_checked, r
    nre = lambda g: sum(1 for o in g.ops if "reorder" in o.fn)  # noqa: E731
    assert r.ops < rp.ops and nre(r.graph) < nre(rp.graph)
    g = r.graph
    assert g.plan is not None and 0 < g.arena < g.naive_bytes
    int_dst = [o for o in g.ops if o.fn in dt._GEMM_FNS and any(c[0] == "attr" and c[1].get("dst_scale")
                                                               for c in o.create)]
    assert int_dst, "no GEMM writes int8 directly (--aiednn-fold-requant)"


def test_cse_merges_duplicate_quantization():
    """tiny_vit quantizes the same LayerNorm output for q, k and v: one reorder after cse"""
    g, _ = _lower("tiny_vit", "int8")
    text, store = dm.to_mlir(g)
    g2 = dm.from_mlir(text, store, g.name, "int8", passes=["--cse"])
    nre = lambda gg: sum(1 for o in gg.ops if "reorder" in o.fn)  # noqa: E731
    assert nre(g2) < nre(g)


def test_canonicalize_folds_quant_round_trip():
    ir = """
module {
  func.func @m(%x: tensor<1x8xsi8>) -> tensor<1x8xsi8> {
    %f = aiednn.reorder %x <{scale = 0.5 : f32, zero_point = 3 : i64}> : (tensor<1x8xsi8>) -> tensor<1x8xf32>
    %q = aiednn.reorder %f <{scale = 0.5 : f32, zero_point = 3 : i64}> : (tensor<1x8xf32>) -> tensor<1x8xsi8>
    func.return %q : tensor<1x8xsi8>
  }
}
"""
    out = dm.opt(ir, "--canonicalize")
    assert "aiednn.reorder" not in out and "return %arg0" in out


def test_quantize_before_maxpool_puts_pooling_on_the_aie(lib):
    """--aiednn-quantize-before-maxpool puts squeezenet's max-pools on the int8 pool mode."""
    m, x = zoo.build("squeezenet1_1", pretrained=False)
    r = dt.model_test(m, x, "int8", lib, "squeezenet1_1", mlir=True)
    assert r.ok and not r.gemm_bad, r
    assert r.impls.get("poolblk:aie:pool_s8s32", 0) == 3, r.impls


def test_quantize_residual_makes_skip_streams_int8(lib):
    """--aiednn-quantize-residual makes resnet18's skips s8; matches the quantized-skip reference."""
    m, x = zoo.build("resnet18", pretrained=False)
    r = dt.model_test(m, x, "int8", lib, "resnet18", mlir=True)
    assert r.ok and not r.gemm_bad, r
    g = r.graph
    adds = [(o, p) for o in g.ops for c in o.create if c[0] == "attr" for p in c[1]["post"] if p[0] == "binary"]
    q = [p for _, p in adds if g.tensors[p[2]].dt == "s8"]
    assert len(q) == 8 and len(q) == len(adds) and all(len(p) > 3 and p[3] > 0 for p in q), adds


def test_sink_permute_fuses_convnext_layer_scale(lib):
    """--aiednn-sink-permute + --aiednn-fuse-epilogue fold convnext's layer scale into the matmul."""
    m, x = zoo.build("convnext_tiny", pretrained=False)
    r = dt.model_test(m, x, "int8", lib, "convnext_tiny", mlir=True)
    assert r.ok and not r.gemm_bad, r
    muls = [o for o in r.graph.ops for c in o.create if c[0] == "attr" for p in c[1]["post"]
            if p[0] == "binary" and p[1] == "mul"]
    assert len(muls) == 18 and sum(1 for o in r.graph.ops if o.fn == "aiednn_binary_create") == 18


def test_sink_permute_moves_a_channel_constant_but_not_a_transposed_matrix():
    """A [1,C,1,1] constant above a permute is a view; a [1,4,4] above a transpose is not."""
    moved = dm.opt("""
module {
  func.func @m(%x: tensor<1x2x3x4xf32>) -> tensor<1x4x2x3xf32> {
    %p = aiednn.reorder %x <{perm = array<i32: 0, 3, 1, 2>, scale = 1.0 : f32, zero_point = 0 : i64}> : (tensor<1x2x3x4xf32>) -> tensor<1x4x2x3xf32>
    %c = aiednn.const <{index = 0 : i64}> : tensor<1x4x1x1xf32>
    %y = aiednn.binary %c, %p <{alg = "mul"}> : (tensor<1x4x1x1xf32>, tensor<1x4x2x3xf32>) -> tensor<1x4x2x3xf32>
    func.return %y : tensor<1x4x2x3xf32>
  }
}
""", "--aiednn-sink-permute")
    assert moved.index("aiednn.binary") < moved.index("aiednn.reorder") and "to tensor<1x1x1x4xf32>" in moved
    kept = dm.opt("""
module {
  func.func @m(%x: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
    %p = aiednn.reorder %x <{perm = array<i32: 0, 2, 1>, scale = 1.0 : f32, zero_point = 0 : i64}> : (tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %c = aiednn.const <{index = 0 : i64}> : tensor<1x4x4xf32>
    %y = aiednn.binary %p, %c <{alg = "add"}> : (tensor<1x4x4xf32>, tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    func.return %y : tensor<1x4x4xf32>
  }
}
""", "--aiednn-sink-permute")
    assert kept.index("aiednn.reorder") < kept.index("aiednn.binary") and "aiednn.view" not in kept


def test_fuse_norm_relu_folds_the_relu_into_batch_norm(lib):
    """--aiednn-fuse-norm-relu: BN -> ReLU is one batch_norm, same result."""
    m, x = zoo.build("densenet121", pretrained=False)
    r = dt.model_test(m, x, "int8", lib, "densenet121", mlir=True)
    assert r.ok and not r.gemm_bad, r
    fns = [o.fn for o in r.graph.ops]
    bns = [f for f in fns if f.startswith("aiednn_batch_normalization_forward_inference_create")]
    assert fns.count("aiednn_eltwise_forward_create") == 0 and len(bns) == 62


def test_fold_norm_quant_drops_the_reorders_bit_exactly(lib):
    """--aiednn-fold-norm-quant: batch_norm -> quantize pairs fold, logits bit-identical."""
    m, x = zoo.build("densenet121", pretrained=False)
    out = {}
    for tag, pipe in (("on", dm.PIPELINE), ("off", tuple(p for p in dm.PIPELINE if p != "--aiednn-fold-norm-quant"))):
        g = dt.lower(m, x, "int8", "densenet121", fuse=False, qscale=dt.calibrate(dt.prepare(m, x), x))
        g = dm.optimize(g, passes=pipe)
        y = dt.run_ctypes(g, lib, x.numpy())[0]
        out[tag] = (np.asarray(y).copy(), [o.fn for o in g.ops])
    (y1, f1), (y0, f0) = out["on"], out["off"]
    assert np.array_equal(y1.view(np.uint8), y0.view(np.uint8))
    folded = f1.count("aiednn_batch_normalization_forward_inference_create_attr")
    assert folded >= 60 and f0.count("aiednn_reorder_create") - f1.count("aiednn_reorder_create") == folded


def test_transformer_folds_are_bit_exact(lib):
    """tiny_vit: head-permute and norm-quant folds give fewer primitives, bit-identical logits."""
    import os
    m, x = zoo.build("tiny_vit", pretrained=False)
    out = {}
    drop = ("--aiednn-fold-head-permute", "--aiednn-fold-norm-quant")
    # compared on ref:sdpa: the folded sdpa would take sdpablk:, which is not bit-exact
    os.environ["AIEDNN_NO_SDPABLK"] = "1"
    try:
        for tag, pipe in (("on", dm.PIPELINE), ("off", tuple(p for p in dm.PIPELINE if p not in drop))):
            g = dt.lower(m, x, "int8", "tiny_vit", fuse=False, qscale=dt.calibrate(dt.prepare(m, x), x))
            g = dm.optimize(g, passes=pipe)
            y = dt.run_ctypes(g, lib, x.numpy())[0]
            out[tag] = (np.asarray(y).copy(), [o.fn for o in g.ops])
    finally:
        del os.environ["AIEDNN_NO_SDPABLK"]
    (y1, f1), (y0, f0) = out["on"], out["off"]
    assert np.array_equal(y1.view(np.uint8), y0.view(np.uint8))
    assert f0.count("aiednn_reorder_create") - f1.count("aiednn_reorder_create") >= 14   # 8 permutes + 6 quantizes
    assert f1.count("aiednn_sdpa_forward_create_attr") == 2
    assert f1.count("aiednn_layer_normalization_forward_inference_create_attr") >= 4


def test_llm_rewrites_fold_rope_rmsnorm_and_qkv_split(lib):
    """tiny_llama RoPE / RMSNorm rewrites and gpt2's c_attn split are bit-identical."""
    m, x = zoo.build("tiny_llama", pretrained=False)
    qs = dt.calibrate(dt.prepare(m, x), x)
    out = {}
    for tag, rw in (("on", dm.REWRITES), ("off", ("fuse_rmsnorm",))):
        g = dm.optimize(dt.lower(m, x, "int8", "tiny_llama", fuse=False, qscale=qs), rewrites=rw)
        out[tag] = (np.asarray(dt.run_ctypes(g, lib, x.numpy())[0]).copy(), [o.fn for o in g.ops])
    (y1, f1), (y0, f0) = out["on"], out["off"]
    assert np.array_equal(y1.view(np.uint8), y0.view(np.uint8))
    assert f1.count("aiednn_rope_create") == 8
    assert not {"aiednn_slice_create", "aiednn_concat_create"} & set(f1)
    assert not any(fn == "aiednn_reorder_create" for fn in f1)          # quantizes folded, permutes into sdpa
    assert f1.count("aiednn_reduction_create") == 0 and f0.count("aiednn_reduction_create") == 0   # rmsnorm in both
    m, x = zoo.build("gpt2_small", pretrained=False)
    qs = dt.calibrate(dt.prepare(m, x), x)
    out = {}
    for tag, rw in (("on", ("split_gemm_slices",)), ("off", ())):
        g = dm.optimize(dt.lower(m, x, "int8", "gpt2_small", fuse=False, qscale=qs), rewrites=rw)
        out[tag] = (np.asarray(dt.run_ctypes(g, lib, x.numpy())[0]).copy(), [o.fn for o in g.ops])
    (y1, f1), (y0, f0) = out["on"], out["off"]
    assert np.array_equal(y1.view(np.uint8), y0.view(np.uint8))
    assert f1.count("aiednn_slice_create") == 0 and f0.count("aiednn_slice_create") == 36


def test_direct_conv_runs_resnet_convs_exactly(lib):
    """Direct conv (aie_modes_conv.h) takes eligible 3x3 convs; every GEMM layer matches ref:."""
    from aiegraph import dnn_viz
    m, x = zoo.build("resnet18", pretrained=False)
    r = dt.model_test(m, x, "int8", lib, "resnet18", mlir=True)
    assert r.ok and not r.gemm_bad, r
    _, _, _, _, calls = dnn_viz.run_timed(r.graph, lib, x.numpy(), calls=True)
    used = {k for c in calls for k, v in c.items() if v}
    assert {"conv_kc_s8s32", "conv_q_s8s8"} <= used, used
