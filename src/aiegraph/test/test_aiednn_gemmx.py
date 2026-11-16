###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""gemm+ composites (aiednn_rt_gemmx.h) against ref:: s8 bit-exact, float allclose."""

from __future__ import annotations

import ctypes as C
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, conv_nd_params, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(41)
A = dict(ARG, diff_src=20, diff_weights=21, diff_bias=22, diff_dst=23)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_gemmx"))


@pytest.fixture
def any_inner():
    os.environ["AIEDNN_GEMMX_ANY_INNER"] = "1"
    yield
    del os.environ["AIEDNN_GEMMX_ANY_INNER"]


def m(a, dt=None):
    return C.byref(md(a.shape, dt or {np.float32: "f32", np.int8: "s8", np.int32: "s32"}[a.dtype.type]))


def both(lib, fn, args, bufs, outs):
    """(default dispatch impl, its outputs), (ref: outputs) on copies of the same buffers"""
    res = []
    for filt in (None, "ref:"):
        b = {k: v.copy() for k, v in bufs.items()}
        lib.set_filter(filt)
        try:
            _, impl = lib.call(fn, args, b, outs[0])
        finally:
            lib.set_filter(None)
        res.append((impl, [b[o] for o in outs]))
    return res[0][0], res[0][1], res[1][1]


def fwd_args(x, w, y, cp, nd):
    return [m(x), m(w), None, m(y), C.byref(cp), None]


# (spatial, kernel, stride, pad_l, pad_r, dilation-1, groups)
CASES = [((6, 7, 5), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0), 1),
         ((5, 6, 6), (3, 1, 3), (2, 1, 2), (1, 0, 0), (0, 0, 1), (0, 0, 1), 2),
         ((9, 8), (3, 3), (2, 1), (1, 1), (1, 1), (1, 0), 2),
         ((17,), (5,), (2,), (2,), (1,), (0,), 1)]


def conv_out(sp, k, s, pl, pr, d):
    return tuple((sp[i] + pl[i] + pr[i] - (k[i] - 1) * (d[i] + 1) - 1) // s[i] + 1 for i in range(len(sp)))


def test_conv3d_s8_runs_on_the_aie_bit_exact(lib):
    sp, k, s, pl, pr, d, g = CASES[0]
    N, Cc, O = 2, 8, 12
    x = RNG.integers(-128, 128, (N, Cc, *sp)).astype(np.int8)
    w = RNG.integers(-128, 128, (O, Cc // g, *k)).astype(np.int8)
    y = np.zeros((N, O, *conv_out(sp, k, s, pl, pr, d)), np.int32)
    cp = conv_nd_params(s, pl, pr, d, g)
    impl, got, want = both(lib, "aiednn_convolution_forward_create_nd", fwd_args(x, w, y, cp, 3),
                           {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: y}, [ARG["dst"]])
    assert impl.startswith("gemm+conv+") and "aie:" in impl, impl
    np.testing.assert_array_equal(got[0], want[0])


def test_float_conv_on_the_bf16x3_gemm(lib):
    """with the floating GEMM mode in the library, a float conv is gemm+conv over bf16x3 (no override)"""
    sp, k, s, pl, pr, d, g = CASES[2]
    x = RNG.uniform(-1, 1, (1, 4, *sp)).astype(np.float32)
    w = RNG.uniform(-1, 1, (6, 4 // g, *k)).astype(np.float32)
    y = np.zeros((1, 6, *conv_out(sp, k, s, pl, pr, d)), np.float32)
    cp = conv_nd_params(s, pl, pr, d, g)
    impl, got, want = both(lib, "aiednn_convolution_forward_create_nd", fwd_args(x, w, y, cp, 2),
                           {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: y}, [ARG["dst"]])
    assert impl == "gemm+conv+bf16x3:aie:bf16_f32", impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("case", CASES)
def test_float_conv_forward_and_backward(lib, any_inner, case):
    sp, k, s, pl, pr, d, g = case
    N, Cc, O = 2, 4, 6
    x = RNG.uniform(-1, 1, (N, Cc, *sp)).astype(np.float32)
    w = RNG.uniform(-1, 1, (O, Cc // g, *k)).astype(np.float32)
    ys = (N, O, *conv_out(sp, k, s, pl, pr, d))
    dy = RNG.uniform(-1, 1, ys).astype(np.float32)
    cp = conv_nd_params(s, pl, pr, d, g)
    tol = dict(rtol=1e-5, atol=1e-5)
    impl, got, want = both(lib, "aiednn_convolution_forward_create_nd", fwd_args(x, w, dy, cp, len(sp)),
                           {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: np.zeros(ys, np.float32)}, [ARG["dst"]])
    assert impl.startswith("gemm+conv"), impl
    np.testing.assert_allclose(got[0], want[0], **tol)
    dx = np.zeros(x.shape, np.float32)
    impl, got, want = both(lib, "aiednn_convolution_backward_data_create", [m(dx), m(w), m(dy), C.byref(cp)],
                           {A["diff_src"]: dx, A["weights"]: w, A["diff_dst"]: dy}, [A["diff_src"]])
    assert impl.startswith("gemm+conv"), impl
    np.testing.assert_allclose(got[0], want[0], **tol)
    dw, db = np.zeros(w.shape, np.float32), np.zeros((O,), np.float32)
    impl, got, want = both(lib, "aiednn_convolution_backward_weights_create", [m(x), m(dw), m(db), m(dy), C.byref(cp)],
                           {A["src"]: x, A["diff_weights"]: dw, A["diff_bias"]: db, A["diff_dst"]: dy},
                           [A["diff_weights"], A["diff_bias"]])
    assert impl.startswith("gemm+conv"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got[1], want[1], **tol)


RNN_DIRS = {"left2right": 0, "right2left": 1, "bi_concat": 2, "bi_sum": 3}


@pytest.mark.parametrize("cell", ["rnn_lstm", "rnn_gru", "rnn_lbr_gru", "rnn_vanilla_tanh", "rnn_vanilla_relu"])
@pytest.mark.parametrize("dirn,L", [("left2right", 1), ("right2left", 1), ("bi_concat", 2), ("bi_sum", 2)])
def test_rnn_as_gemms(lib, any_inner, cell, dirn, L):
    T, N, H = 5, 3, 6
    D = 2 if dirn.startswith("bi") else 1
    DLC = 2 * H if dirn == "bi_concat" else H
    SLC = DLC if L > 1 else 4
    G = {"rnn_lstm": 4, "rnn_gru": 3, "rnn_lbr_gru": 3}.get(cell, 1)
    Gb = G + (cell == "rnn_lbr_gru")
    u = lambda *s: RNG.uniform(-0.6, 0.6, s).astype(np.float32)
    x, W, U, B = u(T, N, SLC), u(L, D, SLC, G, H), u(L, D, H, G, H), u(L, D, Gb, H)
    h0, c0 = u(L, D, N, H), u(L, D, N, H)
    lstm = cell == "rnn_lstm"
    y, hT = np.zeros((T, N, DLC), np.float32), np.zeros((L, D, N, H), np.float32)
    bufs = {ARG["src"]: x, ARG["weights"]: W, 14: U, ARG["bias"]: B, ARG["dst"]: y, 12: h0, 15: hT}
    if lstm:
        bufs.update({13: c0, 16: np.zeros_like(hT)})
    args = [C.c_int(lib.ALG[cell]), C.c_int(RNN_DIRS[dirn]), C.c_float(0.1), m(x), m(h0), m(c0) if lstm else None,
            m(W), m(U), m(B), m(y), m(hT), m(hT) if lstm else None]
    outs = [ARG["dst"], 15] + ([16] if lstm else [])
    impl, got, want = both(lib, "aiednn_rnn_forward_create", args, bufs, outs)
    assert impl.startswith("gemm+rnn"), impl
    for g_, w_ in zip(got, want):
        np.testing.assert_allclose(g_, w_, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind,wdt,sdt,zp", [("matmul", "s4", "f32", True), ("matmul", "f4_e2m1", "e8m0", False),
                                             ("matmul", "s8", "bf16", False), ("ip", "u4", "f32", True)])
def test_dequant_composite(lib, any_inner, kind, wdt, sdt, zp):
    """group-quantized weights: dequantize, then the same problem (bias + ReLU epilogue) in f32"""
    from aiegraph.dnn_ctypes import ARG_WEI_SCALES, ARG_WEI_ZPS, attr, pack4
    M, K, N, g = 3, 64, 10, 32 if sdt == "e8m0" else 16
    x = RNG.uniform(-1, 1, (M, K)).astype(np.float32)
    lo, hi = {"s4": (-8, 7), "u4": (0, 15), "s8": (-128, 127), "f4_e2m1": (0, 15)}[wdt]
    codes = RNG.integers(lo, hi + 1, (N, K) if kind == "ip" else (K, N))
    wbuf = pack4(codes) if wdt != "s8" else codes.astype(np.int8)
    sbuf = {"f32": RNG.uniform(0.01, 0.1, (K // g, N)).astype(np.float32),
            "bf16": (RNG.uniform(0.01, 0.1, (K // g, N)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16),
            "e8m0": RNG.integers(122, 130, (K // g, N)).astype(np.uint8)}[sdt]
    b = RNG.uniform(-1, 1, N).astype(np.float32)
    bufs = {ARG["src"]: x, ARG["weights"]: wbuf, ARG["bias"]: b, ARG["dst"]: np.zeros((M, N), np.float32),
            ARG_WEI_SCALES: sbuf}
    if zp:
        bufs[ARG_WEI_ZPS] = RNG.integers(-2, 3, (K // g, N)).astype(np.int8)
    a, _k = attr(lib.ALG, post_ops=[("eltwise", "relu", 0.0, 0.0)], wei_scales=((K // g, N), sdt),
                 wei_zps=((K // g, N), "s8") if zp else None)
    wshape = (N, K) if kind == "ip" else (K, N)
    if kind == "ip":
        args = [m(x), C.byref(md(wshape, wdt)), C.byref(md((N,), "f32")), C.byref(md((M, N), "f32")), C.byref(a)]
        fn = "aiednn_inner_product_forward_create_attr"
    else:
        args = [m(x), C.byref(md(wshape, wdt)), C.byref(md((1, N), "f32")), C.byref(md((M, N), "f32")), C.c_int(0),
                C.byref(a)]
        bufs[ARG["bias"]] = b.reshape(1, N)
        fn = "aiednn_matmul_forward_create_attr"
    impl, got, want = both(lib, fn, args, bufs, [ARG["dst"]])
    assert impl.startswith("dequant+"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sdt,wdt", [("f16", "f16"), ("bf16", "f16"), ("f32", "s4"), ("f8_e4m3", "f8_e4m3")])
def test_cvt_composite(lib, any_inner, sdt, wdt):
    """operand types no GEMM mode takes: widened to f32, then the same matmul (output scales kept)"""
    from aiegraph.dnn_ctypes import attr, pack4
    M, K, N = 4, 24, 9
    xf = RNG.uniform(-1, 1, (M, K)).astype(np.float32)
    raw = {"f16": lambda a: a.astype(np.float16), "f32": lambda a: a,
           "bf16": lambda a: (a.view(np.uint32) >> 16).astype(np.uint16),
           "f8_e4m3": lambda a: RNG.integers(0, 0x7E, a.shape).astype(np.uint8)}
    x = raw[sdt](xf)
    if wdt == "s4":
        w = pack4(RNG.integers(-8, 8, (K, N)))
    else:
        w = raw[wdt](RNG.uniform(-1, 1, (K, N)).astype(np.float32))
    sc = RNG.uniform(0.5, 1.5, N).astype(np.float32)
    a, _k = attr(lib.ALG, scales=sc)
    args = [C.byref(md((M, K), sdt)), C.byref(md((K, N), wdt)), None, C.byref(md((M, N), "f32")), C.c_int(0), C.byref(a)]
    impl, got, want = both(lib, "aiednn_matmul_forward_create_attr", args,
                           {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: np.zeros((M, N), np.float32)}, [ARG["dst"]])
    assert impl.startswith("cvt+"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)


def deconv_out(sp, k, s, pl, pr, d):
    return tuple((sp[i] - 1) * s[i] - pl[i] - pr[i] + (k[i] - 1) * (d[i] + 1) + 1 for i in range(len(sp)))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dt", ["f32", "s8"])
def test_deconv_as_gemm(lib, any_inner, case, dt):
    sp, k, s, pl, pr, d, g = case
    if dt == "s8" and len(sp) != 3:
        pytest.skip("2-D / 1-D s8 deconvolution is col2im's")
    N, Cc, O = 2, 4, 6
    if dt == "s8":
        x = RNG.integers(-128, 128, (N, Cc, *sp)).astype(np.int8)
        w = RNG.integers(-128, 128, (O, Cc // g, *k)).astype(np.int8)
        y = np.zeros((N, O, *deconv_out(sp, k, s, pl, pr, d)), np.int32)
    else:
        x = RNG.uniform(-1, 1, (N, Cc, *sp)).astype(np.float32)
        w = RNG.uniform(-1, 1, (O, Cc // g, *k)).astype(np.float32)
        y = np.zeros((N, O, *deconv_out(sp, k, s, pl, pr, d)), np.float32)
    cp = conv_nd_params(s, pl, pr, d, g)
    impl, got, want = both(lib, "aiednn_deconvolution_forward_create_nd", fwd_args(x, w, y, cp, len(sp)),
                           {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: y}, [ARG["dst"]])
    assert impl.startswith("gemm+deconv"), impl
    if dt == "s8":
        assert "aie:" in impl
        np.testing.assert_array_equal(got[0], want[0])
    else:
        np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)


def test_inner_product_backward_as_gemms(lib, any_inner):
    x = RNG.uniform(-1, 1, (5, 3, 2, 2)).astype(np.float32)
    w = RNG.uniform(-1, 1, (7, 3, 2, 2)).astype(np.float32)
    dy = RNG.uniform(-1, 1, (5, 7)).astype(np.float32)
    dx = np.zeros_like(x)
    impl, got, want = both(lib, "aiednn_inner_product_backward_data_create", [m(dx), m(w), m(dy)],
                           {A["diff_src"]: dx, A["weights"]: w, A["diff_dst"]: dy}, [A["diff_src"]])
    assert impl.startswith("gemm+ip"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-6)
    dw, db = np.zeros_like(w), np.zeros(7, np.float32)
    impl, got, want = both(lib, "aiednn_inner_product_backward_weights_create", [m(x), m(dw), m(db), m(dy)],
                           {A["src"]: x, A["diff_weights"]: dw, A["diff_bias"]: db, A["diff_dst"]: dy},
                           [A["diff_weights"], A["diff_bias"]])
    assert impl.startswith("gemm+ip"), impl
    for g_, w_ in zip(got, want):
        np.testing.assert_allclose(g_, w_, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mask,causal,sq,skv,layout", [(False, False, 7, 9, 0), (True, False, 5, 5, 0),
                                                        (False, True, 6, 8, 0), (False, True, 6, 6, 2 | 4)])
def test_sdpa_as_gemms(lib, any_inner, mask, causal, sq, skv, layout):
    B, H, D, Dv = 2, 3, 8, 5
    shp = (lambda S, W: (B, S, H, W)) if layout & 2 else (lambda S, W: (B, H, S, W))
    q = RNG.uniform(-1, 1, shp(sq, D)).astype(np.float32)
    k = RNG.uniform(-1, 1, shp(skv, D)).astype(np.float32)
    v = RNG.uniform(-1, 1, shp(skv, Dv)).astype(np.float32)
    dshape = (B, sq, H, Dv) if layout & 4 else (B, H, sq, Dv)
    mk = RNG.uniform(-1, 1, (1, H, sq, skv)).astype(np.float32) if mask else None
    bufs = {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v, ARG["dst"]: np.zeros(dshape, np.float32)}
    if mask:
        bufs[ARG["attn_mask"]] = mk
    # the descriptors are the logical [B, H, S, D] shapes; the layout bits say the memory is [B, S, H, D]
    lg = lambda S, W: (B, H, S, W)
    args = [C.byref(md(lg(sq, D), "f32")), C.byref(md(lg(skv, D), "f32")), C.byref(md(lg(skv, Dv), "f32")),
            m(mk) if mask else None, C.byref(md(lg(sq, Dv), "f32")), C.c_float(0.35), C.c_int(int(causal) | layout)]
    impl, got, want = both(lib, "aiednn_sdpa_forward_create", args, bufs, [ARG["dst"]])
    assert impl.startswith("gemm+sdpa"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mask,causal", [(False, False), (True, False), (False, True)])
def test_sdpa_float_softmax_rows(lib, mask, causal):
    """gemm+sdpa's f32 softmax rows are as close to ref: as the double rows."""
    B, H, S, D = 1, 3, 197, 64
    q, k, v = (RNG.uniform(-2, 2, (B, H, S, D)).astype(np.float32) for _ in range(3))
    lg = (B, H, S, D)
    mk = np.where(RNG.uniform(0, 1, (1, 1, S, S)) < 0.2, -np.inf, RNG.uniform(-1, 1, (1, 1, S, S))).astype(np.float32)
    args = [C.byref(md(lg, "f32")), C.byref(md(lg, "f32")), C.byref(md(lg, "f32")), m(mk) if mask else None,
            C.byref(md(lg, "f32")), C.c_float(0.125), C.c_int(int(causal))]
    bufs = {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v, ARG["dst"]: np.zeros(lg, np.float32)}
    if mask:
        bufs[ARG["attn_mask"]] = mk
    impl, got, want = both(lib, "aiednn_sdpa_forward_create", args, bufs, [ARG["dst"]])
    assert impl.startswith("gemm+sdpa"), impl
    os.environ["AIEDNN_SDPA_DOUBLE_SOFTMAX"] = "1"
    try:
        _, dbl, _ = both(lib, "aiednn_sdpa_forward_create", args, bufs, [ARG["dst"]])
    finally:
        del os.environ["AIEDNN_SDPA_DOUBLE_SOFTMAX"]
    # the bf16x3 GEMMs set the error; the float rows must not add to it
    e_f, e_d = np.abs(got[0] - want[0]).max(), np.abs(dbl[0] - want[0]).max()
    assert e_f <= 5e-6 and e_d <= 5e-6, (e_f, e_d)
    assert e_f <= 1.25 * e_d + 2e-7, (e_f, e_d)


def test_switching_execute_tries_ref_once(lib):
    """aiednn_prefer_fastest after an ELF switch trials ref: once and keeps the faster; outputs stay exact."""
    a = RNG.integers(-128, 128, (32, 64)).astype(np.int8)
    b = RNG.integers(-128, 128, (64, 48)).astype(np.int8)
    want = a.astype(np.int32) @ b.astype(np.int32)
    lib.so.aiednn_set_preference(1)
    os.environ["AIEDNN_TEST_MK_SWITCH"] = "1"
    try:
        prim, st = lib.create("aiednn_matmul_forward_create", m(a), m(b), None, C.byref(md((32, 48), "s32")),
                              C.c_int(0))
        assert st == 0
        before = lib.impl(prim)
        outs, impls = [], []
        for _ in range(3):
            c = np.zeros((32, 48), np.int32)
            assert lib.execute(prim, {ARG["src"]: a, ARG["weights"]: b, ARG["dst"]: c}) == 0
            outs.append(c), impls.append(lib.impl(prim))
        lib.destroy(prim)
    finally:
        del os.environ["AIEDNN_TEST_MK_SWITCH"]
        lib.so.aiednn_set_preference(0)
    assert "aie:" in before, before
    assert impls[0].startswith("ref:") and impls[1] == impls[2] == impls[0], (before, impls)
    for c in outs:
        np.testing.assert_array_equal(c, want)


# ── asymmetric int8: u8 src / src zero points (frameworks' QDQ models) ───────────

@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sdt,zp", [("u8", 113), ("u8", 0), ("s8", -7)])
def test_conv_src_zero_point_on_the_aie(lib, case, sdt, zp):
    """sum (x - zp) w with padding contributing 0 -- the GEMM on the AIE block, the correction exact"""
    from aiegraph.dnn_ctypes import attr
    sp, k, s, pl, pr, d, g = case
    N, Cc, O = 2, 8, 12
    x = (RNG.integers(0, 256, (N, Cc, *sp)).astype(np.uint8) if sdt == "u8"
         else RNG.integers(-128, 128, (N, Cc, *sp)).astype(np.int8))
    w = RNG.integers(-128, 128, (O, Cc // g, *k)).astype(np.int8)
    y = np.zeros((N, O, *conv_out(sp, k, s, pl, pr, d)), np.int32)
    cp = conv_nd_params(s, pl, pr, d, g)
    a, _k = attr(lib.ALG, src_zp=zp)
    impl, got, want = both(lib, "aiednn_convolution_forward_create_nd",
                           [C.byref(md(x.shape, sdt)), m(w), None, m(y), C.byref(cp), C.byref(a)],
                           {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: y}, [ARG["dst"]])
    # 1-D / 2-D: zp+ (the symmetric s8 problem on the tuned paths); 3-D, and narrow groups the tuned paths
    # leave to ref: (< RT_MIN_GROUP_N channels per group): gemm+conv, its GEMMs on the AIE
    assert impl.startswith("zp+") or (impl.startswith("gemm+conv+") and "aie:" in impl), impl
    assert len(sp) < 3 or impl.startswith("gemm+conv+"), impl
    np.testing.assert_array_equal(got[0], want[0])


@pytest.mark.parametrize("kind", ["matmul", "matmul_t", "ip"])
@pytest.mark.parametrize("sdt,zp", [("u8", 113), ("u8", 0), ("s8", 5)])
def test_matmul_src_zero_point_on_the_aie(lib, kind, sdt, zp):
    """Asymmetric activations x symmetric s8 weights with a quantizing epilogue: bit-exact vs ref:."""
    from aiegraph.dnn_ctypes import attr
    M, K, N = 9, 200, 70
    x = (RNG.integers(0, 256, (M, K)).astype(np.uint8) if sdt == "u8" else RNG.integers(-128, 128, (M, K)).astype(np.int8))
    w = RNG.integers(-128, 128, (N, K) if kind in ("ip", "matmul_t") else (K, N)).astype(np.int8)   # matmul_t: ONNX Gemm transB
    b = RNG.uniform(-50, 50, N).astype(np.float32)
    sc = RNG.uniform(1e-3, 3e-3, N).astype(np.float32)
    a, _k = attr(lib.ALG, scales=sc, dst_scale=0.05, post_ops=[("eltwise", "relu", 0.0, 0.0)], src_zp=zp)
    if kind == "ip":
        fn, args = "aiednn_inner_product_forward_create_attr", [C.byref(md(x.shape, sdt)), m(w), m(b), C.byref(md((M, N), "s8")), C.byref(a)]
        bias = b
    else:
        fn, args = "aiednn_matmul_forward_create_attr", [C.byref(md(x.shape, sdt)), m(w), C.byref(md((1, N), "f32")),
                                                         C.byref(md((M, N), "s8")), C.c_int(kind == "matmul_t"), C.byref(a)]
        bias = b.reshape(1, N)
    impl, got, want = both(lib, fn, args, {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: bias,
                                           ARG["dst"]: np.zeros((M, N), np.int8)}, [ARG["dst"]])
    assert impl.startswith("zp+"), impl   # the symmetric s8 problem: blocked / ip with the correction
    np.testing.assert_array_equal(got[0], want[0])


@pytest.mark.parametrize("const_w", [0, 1])
def test_composites_with_constant_weights_take_fresh_inputs(lib, const_w):
    """Constant weights: the composite's inner temporaries must never key the constant-weight cache."""
    from aiegraph.dnn_ctypes import attr
    x0 = RNG.integers(0, 256, (1, 3, 40, 40)).astype(np.uint8)
    w = RNG.integers(-128, 128, (16, 3, 7, 7)).astype(np.int8)
    cp = conv_nd_params((2, 2), (3, 3), (3, 3), (0, 0), 1)
    a, _k = attr(lib.ALG, src_zp=113)
    args = [C.byref(md(x0.shape, "u8")), m(w), None, C.byref(md((1, 16, 20, 20), "s32")), C.byref(cp), C.byref(a)]
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            prim, st = lib.create("aiednn_convolution_forward_create_nd", *args)
            assert st == 0
            lib.so.aiednn_primitive_set_constant_weights(prim, const_w)
            impl = lib.impl(prim)
            res = []
            for seed in (1, 2, 3):
                x = np.random.default_rng(seed).integers(0, 256, x0.shape).astype(np.uint8)
                y = np.zeros((1, 16, 20, 20), np.int32)
                assert lib.execute(prim, {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: y}) == 0
                res.append(y.copy())
            lib.so.aiednn_primitive_destroy(prim)
        finally:
            lib.set_filter(None)
        outs[filt] = (impl, res)
    assert outs[None][0].startswith("zp+"), outs[None][0]
    for got, want in zip(outs[None][1], outs["ref:"][1]):
        np.testing.assert_array_equal(got, want)


def test_rnn_constant_weights_cache_the_weights_only(lib):
    """gemm+rnn under set_constant_weights: W / U cached, h never; equal to ref:."""
    T, N, H = 4, 2, 16
    u = lambda *s: RNG.uniform(-0.6, 0.6, s).astype(np.float32)
    W, U, B = u(1, 1, H, 4, H), u(1, 1, H, 4, H), u(1, 1, 4, H)
    xs = m(np.zeros((T, N, H), np.float32))
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            prim, st = lib.create("aiednn_rnn_forward_create", C.c_int(lib.ALG["rnn_lstm"]), C.c_int(0), C.c_float(0.0),
                                  xs, None, None, m(W), m(U), m(B), xs, None, None)
            assert st == 0
            lib.so.aiednn_primitive_set_constant_weights(prim, 1)
            impl, res = lib.impl(prim), []
            for seed in (1, 2, 3):
                x = np.random.default_rng(seed).uniform(-1, 1, (T, N, H)).astype(np.float32)
                y = np.zeros((T, N, H), np.float32)
                assert lib.execute(prim, {ARG["src"]: x, ARG["weights"]: W, 14: U, ARG["bias"]: B, ARG["dst"]: y}) == 0
                res.append(y.copy())
            lib.so.aiednn_primitive_destroy(prim)
        finally:
            lib.set_filter(None)
        outs[filt] = (impl, res)
    assert outs[None][0].startswith("gemm+rnn+") and "aie:" in outs[None][0], outs[None][0]
    for got, want in zip(outs[None][1], outs["ref:"][1]):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_zp_weight_sums_cached_for_constant_weights(lib):
    """zp+ with constant weights caches the weight sums; results byte for byte equal."""
    x = RNG.integers(0, 256, (1, 16, 10, 10)).astype(np.uint8)
    w = RNG.integers(-60, 60, (24, 16, 3, 3)).astype(np.int8)
    from aiegraph.dnn_ctypes import attr
    a, keep = attr(lib.ALG, src_zp=117)
    cp = conv_nd_params((1, 1), (1, 1), (1, 1), (0, 0), 1)
    outs = []
    for const in (0, 1, 1):
        prim, st = lib.create("aiednn_convolution_forward_create_nd", C.byref(md(x.shape, "u8")), C.byref(md(w.shape, "s8")),
                              None, C.byref(md((1, 24, 10, 10), "s32")), C.byref(cp), C.byref(a))
        assert st == 0
        assert lib.impl(prim).startswith("zp+"), lib.impl(prim)
        lib.so.aiednn_primitive_set_constant_weights(prim, const)
        for _ in range(2 if const else 1):
            y = np.zeros((1, 24, 10, 10), np.int32)
            assert lib.execute(prim, {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: y}) == 0
            outs.append(y)
        lib.destroy(prim)
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])


@pytest.mark.parametrize("sdt,zp", [("s8", 5), ("u8", 113)])
@pytest.mark.parametrize("tr", [0, 1])
def test_batched_linear_with_src_zero_point_on_the_aie(lib, sdt, zp, tr):
    """An int8 nn.Linear on [B, S, D]: zp+ with the batch folded into M, bit-identical to ref:."""
    from aiegraph.dnn_ctypes import attr
    B, S_, K, N = 2, 70, 96, 40
    x = (RNG.integers(0, 256, (B, S_, K)).astype(np.uint8) if sdt == "u8"
         else RNG.integers(-128, 128, (B, S_, K)).astype(np.int8))
    w = RNG.integers(-128, 128, (1, N, K) if tr else (1, K, N)).astype(np.int8)
    bias = RNG.uniform(-1, 1, (1, 1, N)).astype(np.float32)
    a, keep = attr(lib.ALG, scales=RNG.uniform(1e-3, 2e-3, N).astype(np.float32), src_zp=zp)
    args = [C.byref(md(x.shape, sdt)), C.byref(md(w.shape, "s8")), C.byref(md(bias.shape, "f32")),
            C.byref(md((B, S_, N), "f32")), C.c_int(tr), C.byref(a)]
    bufs = {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: bias, ARG["dst"]: np.zeros((B, S_, N), np.float32)}
    impl, got, want = both(lib, "aiednn_matmul_forward_create_attr", args, bufs, [ARG["dst"]])
    assert impl.startswith("zp+") and "aie:" in impl, impl
    np.testing.assert_array_equal(got[0], want[0])


@pytest.mark.parametrize("case", CASES)
def test_deconv_backward_as_convolutions(lib, any_inner, case):
    """conv+deconvbwd: deconvolution backward data / weights as convolutions, ref:'s gradients."""
    sp, k, s, pl, pr, d, g = case
    N, Cc, O = 2, 4, 6
    ys = tuple((sp[i] - 1) * s[i] - pl[i] - pr[i] + (k[i] - 1) * (d[i] + 1) + 1 for i in range(len(sp)))
    x = RNG.uniform(-1, 1, (N, Cc, *sp)).astype(np.float32)
    w = RNG.uniform(-1, 1, (O, Cc // g, *k)).astype(np.float32)
    dy = RNG.uniform(-1, 1, (N, O, *ys)).astype(np.float32)
    cp = conv_nd_params(s, pl, pr, d, g)
    dx = np.zeros(x.shape, np.float32)
    impl, got, want = both(lib, "aiednn_deconvolution_backward_data_create", [m(dx), m(w), m(dy), C.byref(cp)],
                           {A["diff_src"]: dx, A["weights"]: w, A["diff_dst"]: dy}, [A["diff_src"]])
    assert impl.startswith("conv+deconvbwd+gemm+conv"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-4, atol=1e-4)
    dw, db = np.zeros(w.shape, np.float32), np.zeros((O,), np.float32)
    impl, got, want = both(lib, "aiednn_deconvolution_backward_weights_create", [m(x), m(dw), m(db), m(dy), C.byref(cp)],
                           {A["src"]: x, A["diff_weights"]: dw, A["diff_bias"]: db, A["diff_dst"]: dy},
                           [A["diff_weights"], A["diff_bias"]])
    assert impl.startswith("conv+deconvbwd+gemm+conv"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got[1], want[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sp,k,s,pl,pr,alg", [
    ((9, 8), (3, 3), (2, 2), (1, 1), (1, 1), "pooling_avg_include_padding"),
    ((10, 7), (2, 3), (2, 1), (0, 1), (0, 0), "pooling_avg_include_padding"),   # a partial last window dropped
    ((8, 8), (2, 2), (2, 2), (0, 0), (0, 0), "pooling_avg_exclude_padding"),    # no padding: one divisor
    ((6, 5, 7), (2, 2, 3), (1, 2, 2), (0, 1, 1), (1, 0, 1), "pooling_avg_include_padding"),
    ((9, 8), (3, 3), (2, 2), (1, 1), (1, 1), "pooling_avg_exclude_padding"),    # per-window divisors: dy scaled
    ((6, 5, 7), (2, 2, 3), (1, 2, 2), (0, 1, 1), (1, 0, 1), "pooling_avg_exclude_padding"),
])
def test_avg_pool_backward_as_depthwise_deconv(lib, any_inner, sp, k, s, pl, pr, alg):
    """deconv+avgpoolbwd: dx = depthwise deconvolution of dy with 1 / |window| -- ref:pooling's gradient"""
    from aiegraph.dnn_ctypes import pool_nd_params
    N, Cc = 2, 3
    ys = tuple((sp[i] + pl[i] + pr[i] - k[i]) // s[i] + 1 for i in range(len(sp)))
    dy = RNG.uniform(-1, 1, (N, Cc, *ys)).astype(np.float32)
    dx = np.zeros((N, Cc, *sp), np.float32)
    pp = pool_nd_params(k, s, pl, pr)
    impl, got, want = both(lib, "aiednn_pooling_backward_create", [C.c_int(lib.ALG[alg]), m(dx), m(dy), None, C.byref(pp)],
                           {A["diff_src"]: dx, A["diff_dst"]: dy}, [A["diff_src"]])
    assert impl.startswith("deconv+avgpoolbwd+"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("sp,f", [((5, 7), (2, 2)), ((4, 6), (3, 1)), ((3, 4, 5), (2, 2, 2)), ((9,), (4,))])
def test_nearest_resampling_backward_as_depthwise_conv(lib, any_inner, sp, f):
    """conv+resampbwd: nearest up-sampling by integer factors -- dx sums dy over f-blocks (a ones kernel, stride f)"""
    N, Cc = 2, 3
    dy = RNG.uniform(-1, 1, (N, Cc, *[sp[i] * f[i] for i in range(len(sp))])).astype(np.float32)
    dx = np.zeros((N, Cc, *sp), np.float32)
    impl, got, want = both(lib, "aiednn_resampling_backward_create", [C.c_int(lib.ALG["resampling_nearest"]), m(dx), m(dy)],
                           {A["diff_src"]: dx, A["diff_dst"]: dy}, [A["diff_src"]])
    assert impl.startswith("conv+resampbwd+"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("ish,osh", [((2, 3, 5, 7), (2, 3, 10, 14)), ((1, 2, 6, 6), (1, 2, 9, 4)), ((2, 2, 9), (2, 2, 20)),
                                     ((1, 2, 3, 4, 5), (1, 2, 6, 7, 10)), ((1, 2, 9, 3), (1, 2, 4, 8))])   # down x up
def test_linear_resampling_backward_as_axis_gemms(lib, any_inner, ish, osh):
    """gemm+resampbwd: the separable linear map's transpose, one axis at a time as a GEMM -- ref:'s gradient"""
    dy = RNG.uniform(-1, 1, osh).astype(np.float32)
    dx = np.zeros(ish, np.float32)
    impl, got, want = both(lib, "aiednn_resampling_backward_create", [C.c_int(lib.ALG["resampling_linear"]), m(dx), m(dy)],
                           {A["diff_src"]: dx, A["diff_dst"]: dy}, [A["diff_src"]])
    assert impl.startswith("gemm+resampbwd"), impl
    np.testing.assert_allclose(got[0], want[0], rtol=1e-5, atol=1e-5)
