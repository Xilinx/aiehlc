###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.kincore: one gemm_kacc launch per conv / linear layer (unit tests + a test_<fn>_rejects* counter-example
per public planner). x86 only, no TVM build: the kernel source is checked as text."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # this checkout's src/
import aiednn  # noqa: E402
from aiegraph import cnn, kincore  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402

SPEC = '"in  x\\n"\n        "out y"'
SHAPES = [(128, 576, 1024), (320, 2880, 256), (640, 5760, 64)]
LIN = [(320, 768, 128, 128), (1280, 320, 1024, 512)]


# ----------------------------------------------------------------- kplan
def test_kplan_one_launch():
    kp = kincore.kplan(1024, 2880, 320)
    assert kp["legal"] and kp["launches"] == 1 and kp["why"] == []
    assert (kp["mt"], kp["nt"], kp["k_chunks"]) == (16, 5, 45)
    assert kp["acc"] == 1024 and kp["a_port"] == kp["b_port"] == 8192


def test_kplan_more_macro_rows_than_bds_take_more_launches():
    assert kincore.kplan(4096, 2880, 320)["launches"] == 4     # 64 macro rows / 16 shim BDs


def test_kplan_rejects_unaligned_dims():
    kp = kincore.kplan(100, 2880, 330)
    assert not kp["legal"] and kp["why"] == ["M%64", "N%64"]
    assert kincore.kplan(64, 100, 64)["why"] == ["K%64"]


# ----------------------------------------------------------------- kacc_split
def test_kacc_split_whole_gemm_when_it_fits():
    assert kincore.kacc_split(1280, 320, 128) == (1, 128)
    assert kincore.kacc_split(320, 768, 128) == (1, 128)


def test_kacc_split_splits_px_when_one_instance_does_not_fit():
    S, px = kincore.kacc_split(8192, 64, 2048)
    assert S > 1 and S * px == 2048 and px % 64 == 0
    cnn.kacc_geom(8192, 64, px)                     # the launch shape has a walk


def test_kacc_split_rejects_shape_without_split():
    with pytest.raises(AiegraphError):
        kincore.kacc_split(8192, 64, 64 * 33)       # no walk at 2112 px and 2112 is not a multiple of 128
    with pytest.raises(AiegraphError):
        kincore.kacc_split(100, 64, 64)             # ch not a multiple of 64


# ----------------------------------------------------------------- c_window (GAP-aiehlc-tilinglinalg-a45e5b)
@pytest.mark.parametrize("shape", SHAPES + [(320, 2880, 1024), (1280, 2560, 64)])
def test_c_window_is_the_core_whole_b_range(shape):
    w = cnn.kacc_geom(*shape)
    win = kincore.c_window(*shape)
    assert win["rows"] == 16 and win["cols"] == w["brows"] // 4 == win["b_rows"]   # tileN == tileCols
    assert win["bytes"] == 16 * win["cols"] * 4
    assert 2 * (16 * win["tkw"] * 4 + win["cols"] * win["tkw"] * 4 + win["bytes"]) <= kincore.PORT_BYTES


def test_c_window_transposes_when_the_wide_window_does_not_fit():
    assert kincore.c_window(320, 2880, 256)["tr"] == 0
    win = kincore.c_window(320, 2880, 1024)          # c {16, 256} fp32 ping-pong > 29696 B
    assert win["tr"] == 1 and win["cols"] == 320 // 4


def test_c_window_rejects_shape_without_walk():
    with pytest.raises(AiegraphError):
        kincore.c_window(8192, 64, 2048)            # transposed: 2048 ch / 4 needs tkw < 4
    with pytest.raises(AiegraphError):
        kincore.c_window(64, 96, 64)                # K not a multiple of 64


def test_kacc_geom_wide_transposed_refused_not_zero_division():
    with pytest.raises(AiegraphError):
        cnn.kacc_geom(8192, 128, 4096)


# ----------------------------------------------------------------- kacc_name
def test_kacc_name_is_the_aiednn_launch_text():
    for s in SHAPES:
        n = kincore.kacc_name(*s)
        assert n == aiednn.gemm_kacc_launch(*s, cnn.kacc_geom(*s)) and n.startswith("gemm_kacc<")


def test_kacc_name_rejects_unaligned_instance():
    with pytest.raises(AiegraphError):
        kincore.kacc_name(100, 576, 64)


# ----------------------------------------------------------------- plan_kincore
def _conv(M, K, N, cid=0):
    return dict(cid=cid, M=M, K=K, N=N)


def test_plan_kincore_conv_and_linear():
    convs = [_conv(1024, 576, 128), _conv(256, 2880, 320, 1), _conv(256, 2880, 320, 2)]
    lins = [dict(cid=3, M=77, K=768, N=320), dict(cid=4, M=1024, K=320, N=1280)]
    shapes, lin = kincore.plan_kincore(convs, lins)
    assert all(c["launches"] == 1 for c in convs)
    assert [(c["S"], c["pxs"], c["launches"]) for c in lins] == [(1, 128, 1), (2, 512, 2)]
    assert shapes == sorted({(128, 576, 1024), (320, 2880, 256), (320, 768, 128), (1280, 320, 512)})
    assert lin == [(320, 768, 128, 128), (1280, 320, 1024, 512)]


def test_plan_kincore_keeps_a_given_split():
    lins = [dict(M=64, K=320, N=1280, S=2, pxs=32)]
    kincore.plan_kincore([], lins)
    assert lins[0]["launches"] == 2


def test_plan_kincore_rejects_conv_without_one_launch():
    with pytest.raises(AiegraphError):
        kincore.plan_kincore([_conv(100, 576, 128)])          # M not a multiple of 64
    with pytest.raises(AiegraphError):
        kincore.plan_kincore([_conv(4096, 576, 128)])         # 64 macro rows: 4 launches


# ----------------------------------------------------------------- kernel_source / write_source
def test_kernel_source_dispatch_exports_and_parameters():
    t = kincore.kernel_source(SPEC, SHAPES, entry="aiek_x", warm=True, ctrlpkt="aot-kco", lin=LIN)
    assert not re.search(r"@[A-Z][A-Z0-9_]*@", t)
    for ch, k, px in SHAPES:
        assert t.count(f"if (N == {ch} && K == {k} && M == {px}) {{") == 1
        assert f"{kincore.kacc_name(ch, k, px)}<<<mesh>>>(" in t
    ex = re.findall(r'"export    (\w+)\\n"', t)
    assert ex == ["aiednn_aie_conv", "aiednn_aie_stats", "aiednn_aie_linear", "aiednn_aie_launch_stats",
                  "aiednn_aie_host_split_site", "aiednn_aie_host_split_line", "aiednn_aie_split_stats"]
    assert t.startswith(aiednn.template("gemm/ctrlpkt_aot_pragmas.h") + aiednn.template("gemm/ctrlpkt_kco_pragma.h"))
    assert t.index('extern "C" int aiednn_aie_linear(') < t.index('extern "C" int aiednn_aie_conv(int cid')
    assert "gemm_bf16<<<" not in t and "static int uc_alloc(void) {" not in t


def test_kernel_source_extra_parts_and_no_linear():
    t = kincore.kernel_source(SPEC, SHAPES, entry="aiek_x", exports=("x_extra",), prologue="/*P*/\n",
                              kernels="\n/*K*/\n", host_entries="/*H*/\n", bufs=[(10 ** 7, 1, 1)])
    assert 'extern "C" int aiednn_aie_linear(' not in t and "aiednn_aie_linear" not in t
    assert re.findall(r'"export    (\w+)\\n"', t)[2] == "x_extra"
    assert t.index("/*P*/") < t.index("/*K*/") < t.index("/*H*/") < t.index('extern "C" int aiednn_aie_conv(int cid')
    assert "10000000" in t                                     # MAXW from the extra buffer
    off = kincore.kernel_source(SPEC, SHAPES, entry="aiek_x")
    assert "#pragma" not in off.split("gemm_kacc")[0] or "KERNELCONFIGOFFLOAD" not in off


def test_kernel_source_rejects_bad_inputs():
    with pytest.raises(AiegraphError):
        kincore.kernel_source(SPEC, SHAPES, entry="aiek_x", ctrlpkt="kco")
    with pytest.raises(AiegraphError):
        kincore.kernel_source('"in  x"', SHAPES, entry="aiek_x")          # no out line
    with pytest.raises(AiegraphError):
        kincore.kernel_source(SPEC, [], entry="aiek_x")                     # nothing to launch
    with pytest.raises(AiegraphError):
        kincore.kernel_source(SPEC, [(100, 576, 64)], entry="aiek_x")      # instance without a walk


def test_write_source_writes_kernel_and_reference(tmp_path):
    kcc = kincore.write_source(tmp_path / "k", SPEC, SHAPES, entry="aiek_x", lin=LIN)
    assert kcc == tmp_path / "k" / "aiek_x.cc"
    assert kcc.read_text() == kincore.kernel_source(SPEC, SHAPES, entry="aiek_x", lin=LIN)
    ref = (tmp_path / "k" / "aiek_x_ref.c").read_text()
    assert ref == aiednn.template("conv2d/conv_bf16_carrier_ref.c").replace("@E@", "aiek_x") and "@E@" not in ref


def test_write_source_rejects_non_identifier_entry(tmp_path):
    with pytest.raises(AiegraphError):
        kincore.write_source(tmp_path, SPEC, SHAPES, entry="aiek-x")
    assert not list(tmp_path.iterdir())


# ── ctrlpkt_head / ctrlpkt_npragmas (unet-decl P4: the --ctrlpkt pragmas, was the example's CTRLPKT_PRAGMAS) ──
def test_ctrlpkt_head_modes():
    aot, kco = aiednn.template("gemm/ctrlpkt_aot_pragmas.h"), aiednn.template("gemm/ctrlpkt_kco_pragma.h")
    assert kincore.ctrlpkt_head("off") == ""
    assert kincore.ctrlpkt_head("aot") == aot
    assert kincore.ctrlpkt_head("aot-kco") == aot + kco
    t = kincore.kernel_source(SPEC, SHAPES, entry="aiek_x", ctrlpkt="aot")
    assert t.startswith(kincore.ctrlpkt_head("aot")) and "KERNELCONFIGOFFLOAD" not in t
    assert kincore.KACC_KERNEL == aiednn.template("gemm/gemm_kacc_kernel.cc")


def test_ctrlpkt_head_rejects_unknown_mode():
    for bad in ("kco", "AOT", "", None):
        with pytest.raises(AiegraphError):
            kincore.ctrlpkt_head(bad)


def test_ctrlpkt_npragmas_counts_pragma_lines():
    assert kincore.ctrlpkt_npragmas("off") == 0
    assert kincore.ctrlpkt_npragmas("aot") == 5
    assert kincore.ctrlpkt_npragmas("aot-kco") == 6


def test_ctrlpkt_npragmas_rejects_unknown_mode():
    with pytest.raises(AiegraphError):
        kincore.ctrlpkt_npragmas("aot+kco")
