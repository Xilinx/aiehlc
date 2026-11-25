###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.dnn_eager: eager mode's per-layer GEMM template instances (no board, no aiehlc)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "example" / "aiednn" / "models"))

import zoo  # noqa: E402

from aiegraph import dnn_eager as de  # noqa: E402
from aiegraph import dnn_mlir as dm  # noqa: E402
from aiegraph import dnn_torch as dt  # noqa: E402
from aiegraph.dnn_ctypes import AiednnLib  # noqa: E402


def test_pick_fits_core_memory_with_the_most_work_per_round():
    assert de.pick(64, 64, 256) == ((64, 64, 256, 256, 16), None)
    assert de.pick(64, 64, 64) == ((64, 64, 64, 64, 16), None)        # TK never pads a small K up to 256
    assert de.pick(50, 64, 64)[0][0] == 64                             # M padded to 4 x TM rows
    # a wide N is capped at 4 TM (W's block <= A's: VEK280's allocator sizes input slots from A); slices at run time
    assert de.pick(128, 768, 768) == ((128, 64, 768, 256, 16), None)
    assert de.pick(197, 3072, 768) == ((256, 64, 768, 256, 16), None)
    for M, N, K in ((128, 768, 768), (128, 128, 512), (6400, 32, 160)):
        Mp, Np, Kp, TK, TM = de.pick(M, N, K)[0]
        assert Np // de.MESH_COLS <= TM                                  # W block <= A block
    assert de.pick(12544, 64, 147)[0][2:4] == (160, 32)                # resnet stem K: padding within K / 8
    assert de.pick(3136, 64, 576)[0][2:4] == (640, 128)                # not TK 256 -> 768
    for args in (de.pick(128, 768, 768)[0], de.pick(197, 3072, 768)[0]):
        ports, acc = de.core_bytes(args[4], args[3], args[1] // de.MESH_COLS)
        assert ports <= de.CORE_BUDGET and acc <= de.ACC_BUDGET
    # an 8-bit C needs whole 32-bit words per core row: gemm_q / gemm_qr pad N to a multiple of 16
    assert de.pick(3136, 24, 96, "gemm_q")[0][1] == 32 and de.pick(3136, 24, 144, "gemm_qr")[0][1] == 32
    assert de.pick(3136, 24, 96)[0][1] == 24                            # gemm_s8 writes int32: no padding
    args, why = de.pick(64, 6, 64)
    assert args is None and "multiple of 4" in why                     # each mesh column owns N / 4


#: plans that use gemm_q need the MLIR pipeline (fold-requant makes a GEMM write int8 itself)
needs_opt = pytest.mark.skipif(dm.aiednn_opt_path() is None, reason="aiednn-opt not built ($AIEDNN_OPT)")


@pytest.fixture(scope="module")
def mlp_mlir():
    m, x = zoo.build("mlp", pretrained=False)
    return dt.references(m, x, "int8", "mlp", mlir=True)[0]


@pytest.fixture(scope="module")
def mlp_graph():
    m, x = zoo.build("mlp", pretrained=False)
    return m, x, dt.references(m, x, "int8", "mlp")[0]


@needs_opt
def test_mlp_plan_splits_gemm_q_and_gemm_s8(mlp_mlir, tmp_path):
    """3 Linear layers: l1 is 64 x 64 x 256, l2 and l3 share the 64 x 64 x 64 instance"""
    p = de.plan(mlp_mlir, "mlp")
    assert p.gemm_layers == 3 and not p.refused
    # l1 / l2 end in ReLU -> int8: gemm_q requantizes on the core; l3 writes f32 (int32 + host epilogue)
    assert sorted(i.name for i in p.instances) == ["gemm_q__64_64_256_256_16", "gemm_q__64_64_64_64_16",
                                                   "gemm_s8__64_64_64_64_16", "quant__128_64_0_16"]  # + input quantize
    assert {i.name: [o["name"] for o in i.ops] for i in p.instances}["gemm_s8__64_64_64_64_16"] == ["l3"]
    j = json.loads(p.write(tmp_path / "eager_instances.json").read_text())
    assert j["constraints"]["MESH_COLS"] == 4 and len(j["instances"]) == 4


@needs_opt
def test_conv_maps_through_im2col_and_every_resnet18_layer_plans():
    """resnet18: a conv is M = images x OH x OW, N = output channels, K = Cin x kh x kw; with the template's
    N loop all 21 GEMM layers get an instance (equal shapes share one)"""
    m, x = zoo.build("resnet18", pretrained=False)
    p = de.plan(dt.references(m, x, "int8", "resnet18", mlir=True)[0], "resnet18")
    names = {i.name: [o["name"] for o in i.ops] for i in p.instances}
    assert names["gemm_q__12544_64_160_32_16"] == ["conv1"]        # 112 x 112, 3 x 7 x 7 = 147 -> 160; conv+ReLU -> s8
    assert len(names["gemm_q__3136_64_640_128_16"]) == 2           # layer1 conv1s: 56 x 56, 64 x 3 x 3 = 576
    assert len(names["gemm_qr__3136_64_640_128_16"]) == 2          # layer1 conv2s: + the s8 skip, ReLU (gemm_qr)
    assert sum(len(i.ops) for i in p.instances if i.template == "gemm_qr") == 7   # every residual block
    # the stem max pool: 3 x 3 taps -> KK 12 + 4, MW capped at 4096 (the runtime loops over chunks), max, s8
    assert names["pool__4096_16_16_0_0"] == ["maxpool"]
    assert names["quant__128_64_0_16"] == ["conv1_q", "fc_q"]          # the f32 -> s8 input / fc quantizes
    assert names["frow__128_64_8_4_16"] == ["avgpool"]            # the global average pool: a row op
    assert p.other_layers == 4
    assert names["gemm_s8__64_64_512_256_16"] == ["fc"]            # f32 logits; N 1000 in 64-wide steps (W <= A)
    assert p.gemm_layers == 21 and not p.refused


@needs_opt
def test_depthwise_convs_plan_on_dwconv_q_one_instance_per_tap_count():
    """mobilenet_v2: all 17 depthwise convs (3 x 3, clip 0..6 -> s8) share one dwconv_q instance (TP 12) with the
    largest M / C of its layers; every other layer is a GEMM"""
    m, x = zoo.build("mobilenet_v2", pretrained=False)
    p = de.plan(dt.references(m, x, "int8", "mobilenet_v2", mlir=True)[0], "mobilenet_v2")
    dw = [i for i in p.instances if i.template == "dwconv_q"]
    names = {i.name: [op["name"] for op in i.ops] for i in p.instances}
    assert len(dw) == 1 and dw[0].params == (1024, 64, 12, 16) and len(dw[0].ops) == 17
    assert names["frow__128_64_8_4_16"] == ["adaptive_avg_pool2d"]   # the global average pool: a row op
    assert p.other_layers == 20 and p.gemm_layers == 36 and not p.refused   # 17 depthwise + 2 quantizes + the pool
    assert de.dw_core_bytes(*dw[0].params[:3])[0] <= de.CORE_BUDGET


@needs_opt
def test_rms_norms_and_ropes_plan_on_one_instance_each():
    """tiny_llama: its 9 RMS norms (f32 rows of 512; the 8 int8-dst ones carry AIEDNN_ATTR_TOLERANCE_1CODE from
    fold-norm-quant) share one lnorm instance, RMS kind, L 512, TR 2"""
    m, x = zoo.build("tiny_llama", pretrained=False)
    p = de.plan(dt.references(m, x, "int8", "tiny_llama", mlir=True)[0], "tiny_llama")
    ln = [i for i in p.instances if i.template == "lnorm"]
    assert len(ln) == 1 and ln[0].params == (128, 512, 2, 1) and len(ln[0].ops) == 9
    assert de.lnorm_core_bytes(512, 2) <= de.CORE_BUDGET and not p.refused
    # and its 8 RoPEs (f32 x / cos / sin -> s8, half-split, head dim 64): one rope instance
    rp = [i for i in p.instances if i.template == "rope"]
    assert len(rp) == 1 and rp[0].params == (128, 64, 8, 0) and len(rp[0].ops) == 8


def test_mlp_runs_on_the_fixed_megakernel_path(mlp_graph, tmp_path):
    """the model itself: int8 through today's path matches PyTorch (workstation emulation)"""
    m, x, _ = mlp_graph
    r = dt.model_test(m, x, "int8", AiednnLib.build(tmp_path), "mlp")
    assert r.ok and not r.gemm_bad, r


@needs_opt
def test_entry_source_follows_the_instance_contract(mlp_mlir):
    """one launch + AIEK_ENTRY / AIEK_SPEC (exact padded dims) / AIEK_REFERENCE per instance, the template header first"""
    p = de.plan(mlp_mlir, "mlp")
    src = de.entry_source(p)
    assert src.index('#include "aie_user_kernel.h"') < src.index('#include "gemm_s8_eager_kernel.cc"') < src.index("gemm_s8<")
    assert src.index('#include "gemm_q_eager_kernel.cc"') < src.index("gemm_q<64, 64, 256, 256, 16><<<mesh>>>(A, W, C);")
    assert "out   int8  [64,64]  dst" in src and "in    int8  [64,512]  src" in src   # gemm_q: K + TK chunk
    assert "AIEK_ENTRY(eager_gemm_s8__64_64_64_64_16)" in src and src.count("AIEK_SPEC(") == 4
    # AIEK_SPEC defines a static array: a missing ";" only fails in the board compile (the x86 build compiles
    # the references, never this entry)
    assert all(ln.rstrip().endswith(";") or "#" in ln for ln in src.splitlines() if ln.startswith("AIEK_SPEC("))
    assert "in    int8  [64,64]  src\\nparam int8  [64,64]  weight\\nout   int32 [64,64]  dst" in src


def test_mlp_eager_build_runs_on_the_workstation(mlp_graph, tmp_path):
    """dnn_build(mode="eager", target="x86"): the instances' references stand in for the kernels; every GEMM
    goes through eager: and matches PyTorch"""
    from aiegraph import dnn
    m, x, g = mlp_graph
    ref, yf = dt.references(m, x, "int8", "mlp")[1:]
    app = dt.emit_app(g, tmp_path, ref, dt.TOL["int8"], dt._top1(yf, g.top1_range)[0])
    rep = dnn.dnn_build(app, target="x86", out=tmp_path / "x86", run=True, mode="eager", eager=de.plan(g, "mlp"))
    log = rep.run_log.read_text()
    assert rep.run_ok, log[-2000:]
    assert "eager:mm" in log and "gemm_bad=0" in log
