###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Every aiednn primitive against an independent oracle (PyTorch, else numpy).

    python3 -m pytest -q src/aiegraph/test/test_aiednn_ops.py

The runtime is built as a shared library whose AIE implementation list holds
the gemm_s8s32 block kernel with its x86 reference as the entry, so the int8
GEMM family runs through the same blocked / im2col / col2im / ip composites as
on the board; the test also checks which implementation dispatch picked.
Needs g++ and torch; no Vitis, no board.
"""

from __future__ import annotations

import ctypes as C
import math
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import (ARG, DT, MULTIPLE_SRC, AiednnLib, conv_params, f, md,  # noqa: E402
                                 pool_params)

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(7)
AIE_GEMM = "blocked:aie:gemm_s8s32"
AIE_SDPA = "aie:sdpa_s8s32"


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn"))


def i8(shape, lo=-128, hi=127):
    return RNG.integers(lo, hi + 1, size=shape).astype(np.int8)


def f32(shape, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, size=shape).astype(np.float32)


def out(shape, dt):
    return np.zeros(shape, DT[dt][1])


def T(x):
    return torch.from_numpy(np.asarray(x, np.float64))


# ── convolution / deconvolution ───────────────────────────────────────────

CONV_CASES = [  # N, C, H, W, O, K, stride, pad_l, pad_r, dil, groups
    (1, 3, 9, 8, 8, 3, (1, 1), (1, 1), (1, 1), (0, 0), 1),
    (2, 6, 11, 10, 8, 3, (2, 1), (1, 0), (2, 1), (1, 0), 2),
    (1, 8, 7, 7, 8, 3, (1, 1), (1, 1), (1, 1), (0, 0), 8),     # depthwise
    (2, 5, 12, 9, 7, 5, (2, 2), (2, 1), (0, 2), (0, 1), 1),
    (1, 16, 6, 6, 20, 1, (2, 2), (0, 0), (0, 0), (0, 0), 1),   # 1x1 s2
]


def conv_out(H, K, s, pl, pr, d):
    return (H + pl + pr - ((K - 1) * (d + 1) + 1)) // s + 1


def torch_conv(x, w, b, s, pl, pr, d, g):
    xp = F.pad(T(x), (pl[1], pr[1], pl[0], pr[0]))
    return F.conv2d(xp, T(w), None if b is None else T(b), stride=s, dilation=(d[0] + 1, d[1] + 1),
                    groups=g).numpy()


def run_conv(lib, fn, x, w, b, oshape, cp, dt_s, dt_d):
    bufs = {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: out(oshape, dt_d)}
    bmd = None
    if b is not None:
        bufs[ARG["bias"]] = b
        bmd = C.byref(md(b.shape, "s32" if dt_d == "s32" else "f32"))
    return lib.call(fn, [C.byref(md(x.shape, dt_s)), C.byref(md(w.shape, "s8" if dt_s != "f32" else "f32")), bmd,
                         C.byref(md(oshape, dt_d)), C.byref(cp)], bufs, ARG["dst"])


@pytest.mark.parametrize("case", CONV_CASES)
@pytest.mark.parametrize("dt", ["s8", "f32", "u8"])
def test_convolution(lib, case, dt):
    N, Cc, H, W, O, K, s, pl, pr, d, g = case
    x = i8((N, Cc, H, W)) if dt == "s8" else (RNG.integers(0, 256, (N, Cc, H, W)).astype(np.uint8)
                                              if dt == "u8" else f32((N, Cc, H, W)))
    w = i8((O, Cc // g, K, K)) if dt != "f32" else f32((O, Cc // g, K, K))
    b = RNG.integers(-1000, 1000, O).astype(np.int32) if dt != "f32" else f32((O,))
    oshape = (N, O, conv_out(H, K, s[0], pl[0], pr[0], d[0]), conv_out(W, K, s[1], pl[1], pr[1], d[1]))
    got, impl = run_conv(lib, "aiednn_convolution_forward_create", x, w, b, oshape, conv_params(s, pl, pr, d, g),
                         dt, "s32" if dt != "f32" else "f32")
    want = torch_conv(x, w, b, s, pl, pr, d, g)
    if dt == "f32":
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
        assert impl.startswith("gemm+conv+") and "aie:" in impl, impl   # GEMMs on the AIE bf16x3 mode
    else:
        np.testing.assert_array_equal(got, want.astype(np.int64))
        dw = dt == "s8" and g == Cc == O and K <= 5 and s[0] == s[1] and s[0] <= 2 and d == (0, 0)
        on_aie = dt == "s8" and (g == 1 or O // g >= 8)     # other narrow groups -> ref:
        if dt == "u8":   # u8 src: zp+ (the shifted s8 problem on the tuned paths), or gemm+conv for narrow groups
            assert impl.startswith("zp+") or (impl.startswith("gemm+conv+") and "aie:" in impl), impl
        else:
            assert impl == ("dwblk:aie:dw_s8s32" if dw else "im2col+" + AIE_GEMM if on_aie else "ref:convolution")


DECONV_CASES = [  # N, C, H, W, O, K, stride, pad_l, pad_r, dil, groups
    (1, 4, 5, 6, 6, 3, (2, 2), (1, 1), (1, 1), (0, 0), 1),
    (2, 6, 4, 4, 4, 3, (1, 2), (0, 1), (1, 0), (1, 0), 2),
    (1, 3, 3, 5, 2, 4, (3, 1), (0, 0), (0, 0), (0, 1), 1),
]


@pytest.mark.parametrize("case", DECONV_CASES)
@pytest.mark.parametrize("dt", ["s8", "f32"])
def test_deconvolution(lib, case, dt):
    N, Cc, H, W, O, K, s, pl, pr, d, g = case
    x = i8((N, Cc, H, W)) if dt == "s8" else f32((N, Cc, H, W))
    w = i8((O, Cc // g, K, K)) if dt == "s8" else f32((O, Cc // g, K, K))
    b = RNG.integers(-500, 500, O).astype(np.int32) if dt == "s8" else f32((O,))
    Og, Cg = O // g, Cc // g
    wt = np.zeros((Cc, Og, K, K), np.float64)          # torch: [in, out/groups, kh, kw]
    for gi in range(g):
        wt[gi * Cg:(gi + 1) * Cg] = w[gi * Og:(gi + 1) * Og].astype(np.float64).transpose(1, 0, 2, 3)
    full = F.conv_transpose2d(T(x), torch.from_numpy(wt), T(b), stride=s, dilation=(d[0] + 1, d[1] + 1),
                              groups=g).numpy()
    Ho, Wo = full.shape[2] - pl[0] - pr[0], full.shape[3] - pl[1] - pr[1]
    want = full[:, :, pl[0]:pl[0] + Ho, pl[1]:pl[1] + Wo]
    got, impl = run_conv(lib, "aiednn_deconvolution_forward_create", x, w, b, (N, O, Ho, Wo),
                         conv_params(s, pl, pr, d, g), dt, "s32" if dt == "s8" else "f32")
    if dt == "f32":
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
        assert impl.startswith("gemm+deconv+") and "aie:" in impl, impl
    else:
        np.testing.assert_array_equal(got, want.astype(np.int64))
        assert impl == ("col2im+" + AIE_GEMM if g == 1 or Cc // g >= 8 else "ref:deconvolution")


# ── matmul / inner product / sdpa ─────────────────────────────────────────

@pytest.mark.parametrize("sa,sw,transposed,bias,dt,ddt", [
    ((70, 300), (300, 150), 0, None, "s8", "s32"),
    ((3, 20, 64), (1, 64, 30), 0, (1, 1, 30), "s8", "s32"),     # batch broadcast + bias
    ((2, 1, 17, 40), (1, 3, 25, 40), 1, (2, 3, 17, 25), "s8", "s8"),  # transposed, saturating
    ((5, 33, 12), (5, 12, 7), 0, (1, 33, 7), "f32", "f32"),
])
def test_matmul(lib, sa, sw, transposed, bias, dt, ddt):
    a = i8(sa, -20, 20) if dt == "s8" else f32(sa)
    w = i8(sw, -20, 20) if dt == "s8" else f32(sw)
    wm = np.swapaxes(w, -1, -2) if transposed else w
    want = np.matmul(a.astype(np.float64), wm.astype(np.float64))
    bufs = {ARG["src"]: a, ARG["weights"]: w}
    bmd = None
    if bias:
        b = RNG.integers(-3000, 3000, bias).astype(np.int32) if dt == "s8" else f32(bias)
        want = want + b
        bufs[ARG["bias"]] = b
        bmd = C.byref(md(bias, "s32" if dt == "s8" else "f32"))
    oshape = want.shape
    bufs[ARG["dst"]] = out(oshape, ddt)
    got, impl = lib.call("aiednn_matmul_forward_create",
                         [C.byref(md(sa, dt)), C.byref(md(sw, dt)), bmd, C.byref(md(oshape, ddt)), C.c_int(transposed)],
                         bufs, ARG["dst"])
    if dt == "f32":   # the AIE bf16 mode as a bf16x3 split (test_aiednn_bf16.py): near f32
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
        assert impl == "bf16x3:aie:bf16_f32"
    else:
        exp = np.clip(want, -128, 127) if ddt == "s8" else want
        np.testing.assert_array_equal(got, exp.astype(np.int64))
        assert impl == AIE_GEMM


@pytest.mark.parametrize("dt", ["s8", "f32"])
def test_inner_product(lib, dt):
    x = i8((3, 8, 4, 4)) if dt == "s8" else f32((3, 8, 4, 4))
    w = i8((50, 8, 4, 4)) if dt == "s8" else f32((50, 8, 4, 4))
    b = RNG.integers(-100, 100, 50).astype(np.int32) if dt == "s8" else f32((50,))
    want = x.reshape(3, -1).astype(np.float64) @ w.reshape(50, -1).astype(np.float64).T + b
    ddt = "s32" if dt == "s8" else "f32"
    got, impl = lib.call("aiednn_inner_product_forward_create",
                         [C.byref(md(x.shape, dt)), C.byref(md(w.shape, dt)), C.byref(md(b.shape, ddt)),
                          C.byref(md((3, 50), ddt))],
                         {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: out((3, 50), ddt)}, ARG["dst"])
    if dt == "f32":
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
        assert impl == "ip+bf16x3:aie:bf16_f32"   # f32 inner product through the AIE bf16x3 matmul
    else:
        np.testing.assert_array_equal(got, want.astype(np.int64))
        assert impl == "ip+" + AIE_GEMM


@pytest.mark.parametrize("mask,causal,sq,skv", [(False, False, 7, 9), (True, False, 5, 5), (False, True, 6, 6)])
def test_sdpa(lib, mask, causal, sq, skv):
    B, H, D, Dv = 2, 3, 8, 5
    q, k, v = f32((B, H, sq, D)), f32((B, H, skv, D)), f32((B, H, skv, Dv))
    scale = 1.0 / math.sqrt(D)
    m = f32((1, H, sq, skv)) if mask else None
    want = F.scaled_dot_product_attention(T(q), T(k), T(v), attn_mask=None if m is None else T(m),
                                          is_causal=causal, scale=scale).numpy()
    bufs = {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v, ARG["dst"]: out(want.shape, "f32")}
    if mask:
        bufs[ARG["attn_mask"]] = m
    got, impl = lib.call("aiednn_sdpa_forward_create",
                         [C.byref(md(q.shape, "f32")), C.byref(md(k.shape, "f32")), C.byref(md(v.shape, "f32")),
                          C.byref(md(m.shape, "f32")) if mask else None, C.byref(md(want.shape, "f32")), f(scale),
                          C.c_int(causal)], bufs, ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    assert impl.startswith("gemm+sdpa+") and "aie:" in impl, impl   # Q K^T and P V on the AIE bf16x3 mode


@pytest.mark.parametrize("causal,sq,skv,layout", [(False, 64, 64, 0), (True, 64, 64, 0), (False, 37, 50, 0),
                                                   (False, 64, 64, 2 | 4), (True, 200, 120, 0)])
def test_sdpa_on_aie_block(lib, causal, sq, skv, layout):
    """sdpablk: (the GEMM ELF's SDPA mode, aie_modes_sdpa.h) for an sdpa into a quantized s8 destination:
    Q / K / V quantized per tensor, integer scores, table softmax -- within RT_SDPA_TOL (4) codes of
    ref:sdpa; the f32-destination sdpa above stays on ref:. Query blocks (sq 200), a short last block (37)
    and the folded head-permute layouts (AIEDNN_SDPA_QKV_BSHD | DST_BSHD) included."""
    B, H, D, Dv = 2, 4, 16, 16
    bshd = layout & 2
    shp = (lambda s, d: (B, s, H, d)) if bshd else (lambda s, d: (B, H, s, d))
    q, k, v = f32(shp(sq, D)), f32(shp(skv, D)), f32(shp(skv, Dv))
    lg = shp                                # descriptors as the memory is: create() reads the flags
    a, keep = attr(lib.ALG, dst_scale=0.01, dst_zp=3)
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_sdpa_forward_create_attr",
                                 [C.byref(md(lg(sq, D), "f32")), C.byref(md(lg(skv, D), "f32")),
                                  C.byref(md(lg(skv, Dv), "f32")), None, C.byref(md(lg(sq, Dv), "s8")),
                                  f(1.0 / math.sqrt(D)), C.c_int(causal | layout), C.byref(a)],
                                 {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v,
                                  ARG["dst"]: out(shp(sq, Dv), "s8")}, ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    aie, ref = outs.pop("sdpablk:" + AIE_SDPA), outs.pop("ref:sdpa")
    diff = np.abs(aie.astype(np.int64) - ref)
    assert diff.max() <= 4, diff.max()   # RT_SDPA_TOL: Q / K quantized to int8 move the softmax a little
    assert (diff > 0).mean() < 0.3   # V's quantization step (max|v| / 127) is coarser than dst_scale here


# ── eltwise / binary / prelu / reduction ──────────────────────────────────

ELTWISE = [  # name, alpha, beta, oracle, input domain
    ("relu", 0.0, 0.0, F.relu, None), ("relu", 0.1, 0.0, lambda x: F.leaky_relu(x, 0.1), None),
    ("elu", 1.0, 0.0, F.elu, None), ("gelu_erf", 0, 0, F.gelu, None),
    ("gelu_tanh", 0, 0, lambda x: F.gelu(x, approximate="tanh"), None), ("logistic", 0, 0, torch.sigmoid, None),
    ("tanh", 0, 0, torch.tanh, None), ("swish", 1.0, 0, F.silu, None), ("soft_relu", 1.0, 0, F.softplus, None),
    ("mish", 0, 0, F.mish, None), ("hardswish", 1 / 6, 0.5, F.hardswish, None),
    ("hardsigmoid", 1 / 6, 0.5, F.hardsigmoid, None), ("clip", -0.5, 0.7, lambda x: x.clamp(-0.5, 0.7), None),
    ("linear", 2.0, 0.5, lambda x: 2 * x + 0.5, None), ("abs", 0, 0, torch.abs, None), ("exp", 0, 0, torch.exp, None),
    ("log", 0, 0, torch.log, "pos"), ("sqrt", 0, 0, torch.sqrt, "pos"), ("square", 0, 0, torch.square, None),
    ("pow", 2.0, 3.0, lambda x: 2 * x ** 3, None), ("round", 0, 0, torch.round, None), ("neg", 0, 0, torch.neg, None),
    ("reciprocal", 0, 0, torch.reciprocal, "pos"), ("rsqrt", 0, 0, torch.rsqrt, "pos"), ("erf", 0, 0, torch.erf, None),
    ("floor", 0, 0, torch.floor, None), ("ceil", 0, 0, torch.ceil, None), ("sin", 0, 0, torch.sin, None),
    ("cos", 0, 0, torch.cos, None),
]


# eltwise algorithms the AIE FVEC mode serves for f32 (aiednn_rt_fvec.h)
# the FVEC mode is registered when gemm_uk.cc carries its entry (aie_modes_fvec.h / aiednn_rt_fvec.h)
HAS_FVEC = "AIEK_SPEC(fvec_blk_f32" in (Path(__file__).resolve().parents[3] / "src" / "aietensorop" / "gemm" / "gemm_uk.cc").read_text()
HAS_FPOOL = "AIEK_SPEC(fpool_blk_f32" in (Path(__file__).resolve().parents[3] / "src" / "aietensorop" / "gemm" / "gemm_uk.cc").read_text()
FVEC_ELT = {"relu", "elu", "gelu_tanh", "logistic", "tanh", "swish", "clip", "linear", "abs", "exp", "sqrt", "square",
            "neg", "identity", "gelu_erf", "selu", "soft_relu", "mish", "hardswish", "hardsigmoid", "log", "reciprocal",
            "rsqrt", "erf", "sign", "softsign", "log_sigmoid",
            "round", "floor", "ceil", "logical_not", "pow", "sin", "cos", "tan"}   # vector programs (aie_fvec_prog); pow: integer beta in [-2, 4]


@pytest.mark.parametrize("name,alpha,beta,oracle,dom", ELTWISE, ids=[f"{e[0]}{e[1]}" for e in ELTWISE])
def test_eltwise(lib, name, alpha, beta, oracle, dom):
    x = f32((2, 3, 5, 7), 0.1, 3.0) if dom == "pos" else f32((2, 3, 5, 7), -3.0, 3.0)
    if name == "round":
        x[0, 0, 0, :4] = [0.5, 1.5, 2.5, -0.5]          # ties to even
    got, impl = lib.call("aiednn_eltwise_forward_create",
                         [C.c_int(lib.ALG["eltwise_" + name]), f(alpha), f(beta), C.byref(md(x.shape, "f32")),
                          C.byref(md(x.shape, "f32"))], {ARG["src"]: x, ARG["dst"]: out(x.shape, "f32")}, ARG["dst"])
    np.testing.assert_allclose(got, oracle(T(x)).numpy(), rtol=1e-5, atol=1e-6)
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC and name in FVEC_ELT else "ref:eltwise"), impl   # f32: the AIE FVEC mode


def test_eltwise_int8_relu_saturates(lib):
    x = i8((4, 9))
    got, _ = lib.call("aiednn_eltwise_forward_create",
                      [C.c_int(lib.ALG["eltwise_linear"]), f(3.0), f(1.0), C.byref(md(x.shape, "s8")),
                       C.byref(md(x.shape, "s8"))], {ARG["src"]: x, ARG["dst"]: out(x.shape, "s8")}, ARG["dst"])
    np.testing.assert_array_equal(got, np.clip(3 * x.astype(np.int64) + 1, -128, 127))


BINARY = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "max": np.maximum,
          "min": np.minimum, "pow": np.power, "eq": np.equal, "ne": np.not_equal, "gt": np.greater,
          "ge": np.greater_equal, "lt": np.less, "le": np.less_equal}


@pytest.mark.parametrize("name", list(BINARY) + ["select"])
def test_binary(lib, name):
    a, b = f32((2, 3, 4, 5), 0.5, 2.0), f32((1, 3, 1, 5), 0.5, 2.0)
    if name in ("eq", "ne", "ge", "le"):
        b[0, 0, 0, :] = a[0, 0, 0, :]                   # some equal pairs
    bufs = {ARG["src"]: a, ARG["src_1"]: b, ARG["dst"]: out(a.shape, "f32")}
    c_md = None
    if name == "select":
        c = RNG.integers(0, 2, (2, 1, 4, 1)).astype(np.int8)
        bufs[ARG["src_2"]] = c
        c_md = C.byref(md(c.shape, "s8"))
        want = np.where(c != 0, a, b)
    else:
        want = BINARY[name](a.astype(np.float64), b.astype(np.float64))
    got, impl = lib.call("aiednn_binary_create",
                         [C.c_int(lib.ALG["binary_" + name]), C.byref(md(a.shape, "f32")), C.byref(md(b.shape, "f32")),
                          c_md, C.byref(md(a.shape, "f32"))], bufs, ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-6)
    fvec = ("add", "sub", "mul", "div", "max", "min", "eq", "ne", "gt", "ge", "lt", "le", "select")   # pow: ref:
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC and name in fvec else "ref:binary"), impl


def test_binary_int_add_saturates(lib):
    a, b = i8((3, 4)), i8((1, 4))
    got, _ = lib.call("aiednn_binary_create",
                      [C.c_int(lib.ALG["binary_add"]), C.byref(md(a.shape, "s8")), C.byref(md(b.shape, "s8")), None,
                       C.byref(md(a.shape, "s8"))], {ARG["src"]: a, ARG["src_1"]: b, ARG["dst"]: out(a.shape, "s8")},
                      ARG["dst"])
    np.testing.assert_array_equal(got, np.clip(a.astype(np.int64) + b, -128, 127))


def test_prelu(lib):
    x, w = f32((2, 4, 3, 3)), f32((1, 4, 1, 1), 0.0, 0.5)
    got, impl = lib.call("aiednn_prelu_forward_create",
                         [C.byref(md(x.shape, "f32")), C.byref(md(w.shape, "f32")), C.byref(md(x.shape, "f32"))],
                         {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: out(x.shape, "f32")}, ARG["dst"])
    np.testing.assert_allclose(got, F.prelu(T(x), T(w.reshape(4))).numpy(), rtol=1e-6)
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC else "ref:prelu"), impl   # f32: FVEC (ROWPROG / program)


REDUCTIONS = [("sum", 0, 0, lambda x, ax: x.sum(ax, keepdims=True)),
              ("mean", 0, 0, lambda x, ax: x.mean(ax, keepdims=True)),
              ("max", 0, 0, lambda x, ax: x.max(ax, keepdims=True)), ("min", 0, 0, lambda x, ax: x.min(ax, keepdims=True)),
              ("mul", 0, 0, lambda x, ax: x.prod(ax, keepdims=True)),
              ("amax", 0, 0, lambda x, ax: np.abs(x).max(ax, keepdims=True)),
              ("norm_lp_sum", 2.0, 0.0, lambda x, ax: np.sqrt((x ** 2).sum(ax, keepdims=True))),
              ("norm_lp_power_p_sum", 3.0, 0.25, lambda x, ax: (np.abs(x) ** 3).sum(ax, keepdims=True) + 0.25)]


@pytest.mark.parametrize("name,p,eps,oracle", REDUCTIONS, ids=[r[0] for r in REDUCTIONS])
@pytest.mark.parametrize("axes", [(1,), (0, 2), (3,)])
def test_reduction(lib, name, p, eps, oracle, axes):
    x = f32((2, 3, 4, 5), 0.5, 1.5)
    want = oracle(x.astype(np.float64), axes)
    got, impl = lib.call("aiednn_reduction_create",
                         [C.c_int(lib.ALG["reduction_" + name]), f(p), f(eps), C.byref(md(x.shape, "f32")),
                          C.byref(md(want.shape, "f32"))], {ARG["src"]: x, ARG["dst"]: out(want.shape, "f32")},
                         ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-5)
    trailing = axes == (3,) and name in ("sum", "mean", "max", "min", "amax")   # rows of the last axis: the AIE FVEC mode
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC and trailing else "ref:reduction"), impl


# ── pooling / resampling / lrn ────────────────────────────────────────────

@pytest.mark.parametrize("alg", ["max", "avg_include_padding", "avg_exclude_padding"])
@pytest.mark.parametrize("dt", ["f32", "s8"])
def test_pooling(lib, alg, dt):
    x = f32((2, 3, 9, 8)) if dt == "f32" else i8((2, 3, 9, 8))
    k, s, p = (3, 2), (2, 1), (1, 1)
    dil = (1, 0) if alg == "max" else (0, 0)            # torch avg_pool has no dilation
    Ho = conv_out(9, k[0], s[0], p[0], p[0], dil[0])
    Wo = conv_out(8, k[1], s[1], p[1], p[1], dil[1])
    if alg == "max":
        want = F.max_pool2d(T(x), k, s, p, dilation=(dil[0] + 1, dil[1] + 1)).numpy()
    else:
        want = F.avg_pool2d(T(x), k, s, p, count_include_pad=alg == "avg_include_padding").numpy()
    got, impl = lib.call("aiednn_pooling_forward_create",
                         [C.c_int(lib.ALG["pooling_" + alg]), C.byref(md(x.shape, dt)), C.byref(md((2, 3, Ho, Wo), dt)),
                          C.byref(pool_params(k, s, p, p, dil))],
                         {ARG["src"]: x, ARG["dst"]: out((2, 3, Ho, Wo), dt)}, ARG["dst"])
    if dt == "f32":   # atol: fpool:'s f32 window sum vs ref:'s double one, for averages near 0
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_array_equal(got, np.rint(want))
    # f32 windows: the FPOOL mode (aiednn_rt_fpool.h) when the GEMM ELF carries it; dilated max stays ref:
    on_fpool = HAS_FPOOL and dt == "f32" and alg != "max"
    assert impl == ("fpool:aie:fpool_f32" if on_fpool else "ref:pooling"), impl


@pytest.mark.parametrize("alg,shape,oshape,mode", [
    ("nearest", (2, 3, 5, 7), (2, 3, 9, 4), "nearest-exact"),
    ("linear", (2, 3, 5, 7), (2, 3, 11, 3), "bilinear"),
    ("linear", (1, 2, 9), (1, 2, 4), "linear"),
    ("linear", (1, 2, 3, 4, 5), (1, 2, 5, 3, 8), "trilinear"),
    ("nearest", (1, 2, 4, 4, 4), (1, 2, 6, 2, 5), "nearest-exact"),
])
def test_resampling(lib, alg, shape, oshape, mode):
    x = f32(shape)
    kw = {} if mode.startswith("nearest") else {"align_corners": False}
    want = F.interpolate(T(x), size=oshape[2:], mode=mode, **kw).numpy()
    got, impl = lib.call("aiednn_resampling_forward_create",
                         [C.c_int(lib.ALG["resampling_" + alg]), C.byref(md(shape, "f32")), C.byref(md(oshape, "f32"))],
                         {ARG["src"]: x, ARG["dst"]: out(oshape, "f32")}, ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert impl == "ref:resampling"


def lrn_within(x, n, alpha, beta, k):
    half = (n - 1) // 2
    sq = np.pad(x.astype(np.float64) ** 2, ((0, 0), (0, 0), (half, half), (half, half)))
    acc = sum(sq[:, :, i:i + x.shape[2], j:j + x.shape[3]] for i in range(n) for j in range(n))
    return x / (k + alpha / (n * n) * acc) ** beta


@pytest.mark.parametrize("alg", ["across_channels", "within_channel"])
def test_lrn(lib, alg):
    x = f32((2, 7, 5, 6))
    n, alpha, beta, k = 5, 1e-2, 0.75, 1.5
    want = (F.local_response_norm(T(x), n, alpha, beta, k).numpy() if alg == "across_channels"
            else lrn_within(x, n, alpha, beta, k))
    got, impl = lib.call("aiednn_lrn_forward_create",
                         [C.c_int(lib.ALG["lrn_" + alg]), C.c_int64(n), f(alpha), f(beta), f(k),
                          C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "f32"))],
                         {ARG["src"]: x, ARG["dst"]: out(x.shape, "f32")}, ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-5)
    assert impl == "ref:lrn"


# ── softmax / normalization ───────────────────────────────────────────────

@pytest.mark.parametrize("alg", ["accurate", "log"])
@pytest.mark.parametrize("axis", [0, 1, 3])
def test_softmax(lib, alg, axis):
    x = f32((2, 3, 4, 5), -5, 5)
    want = (F.softmax if alg == "accurate" else F.log_softmax)(T(x), dim=axis).numpy()
    got, impl = lib.call("aiednn_softmax_forward_create",
                         [C.c_int(lib.ALG["softmax_" + alg]), C.c_int(axis), C.byref(md(x.shape, "f32")),
                          C.byref(md(x.shape, "f32"))], {ARG["src"]: x, ARG["dst"]: out(x.shape, "f32")}, ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC else "ref:softmax"), impl   # FVEC: rows of any axis (strided gather)


USE_SCALE, USE_SHIFT, FUSE_RELU, RMS = 1, 2, 4, 8


@pytest.mark.parametrize("flags", [0, USE_SCALE | USE_SHIFT, USE_SCALE | USE_SHIFT | FUSE_RELU])
def test_batch_normalization(lib, flags):
    x, mean, var = f32((2, 4, 3, 5)), f32((4,)), f32((4,), 0.5, 2.0)
    g, b = f32((4,)), f32((4,))
    want = F.batch_norm(T(x), T(mean), T(var), T(g) if flags & USE_SCALE else None,
                        T(b) if flags & USE_SHIFT else None, training=False, eps=1e-3)
    want = (F.relu(want) if flags & FUSE_RELU else want).numpy()
    bufs = {ARG["src"]: x, ARG["dst"]: out(x.shape, "f32"), ARG["mean"]: mean, ARG["variance"]: var}
    if flags & USE_SCALE:
        bufs[ARG["scale"]] = g
    if flags & USE_SHIFT:
        bufs[ARG["shift"]] = b
    got, impl = lib.call("aiednn_batch_normalization_forward_inference_create",
                         [C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "f32")), f(1e-3), C.c_uint(flags)], bufs,
                         ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC else "ref:batch_normalization"), impl   # f32: FVEC ROWPROG


@pytest.mark.parametrize("rms", [False, True])
def test_layer_normalization(lib, rms):
    x, g, b = f32((2, 3, 16)), f32((16,)), f32((16,))
    if rms:
        xd = T(x)
        want = (xd / torch.sqrt((xd ** 2).mean(-1, keepdim=True) + 1e-5) * T(g)).numpy()
        flags = USE_SCALE | RMS
    else:
        want = F.layer_norm(T(x), (16,), T(g), T(b), eps=1e-5).numpy()
        flags = USE_SCALE | USE_SHIFT
    bufs = {ARG["src"]: x, ARG["dst"]: out(x.shape, "f32"), ARG["scale"]: g}
    if not rms:
        bufs[ARG["shift"]] = b
    got, impl = lib.call("aiednn_layer_normalization_forward_inference_create",
                         [C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "f32")), f(1e-5), C.c_uint(flags)], bufs,
                         ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC else "ref:layer_normalization"), impl   # rows of 16: FVEC


@pytest.mark.parametrize("groups", [1, 3, 6])                  # 6 = C: instance norm
def test_group_normalization(lib, groups):
    x, g, b = f32((2, 6, 4, 5)), f32((6,)), f32((6,))
    want = F.group_norm(T(x), groups, T(g), T(b), eps=1e-5).numpy()
    got, impl = lib.call("aiednn_group_normalization_forward_inference_create",
                         [C.c_int64(groups), C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "f32")), f(1e-5),
                          C.c_uint(USE_SCALE | USE_SHIFT)],
                         {ARG["src"]: x, ARG["dst"]: out(x.shape, "f32"), ARG["scale"]: g, ARG["shift"]: b}, ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    if groups == 6:
        np.testing.assert_allclose(got, F.instance_norm(T(x), weight=T(g), bias=T(b), eps=1e-5).numpy(),
                                   rtol=1e-5, atol=1e-5)
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC else "ref:group_normalization"), impl   # f32: FVEC block-norm rows


# ── data movement ─────────────────────────────────────────────────────────

def test_concat(lib):
    xs = [i8((2, n, 3)) for n in (1, 4, 2)]
    want = np.concatenate(xs, axis=1)
    mds = (md((2, 1, 3), "s8"), md((2, 4, 3), "s8"), md((2, 2, 3), "s8"))
    arr = (type(mds[0]) * 3)(*mds)
    bufs = {MULTIPLE_SRC + i: x for i, x in enumerate(xs)}
    bufs[ARG["dst"]] = out(want.shape, "s8")
    got, impl = lib.call("aiednn_concat_create", [C.c_int(3), arr, C.c_int(1), C.byref(md(want.shape, "s8"))], bufs,
                         ARG["dst"])
    np.testing.assert_array_equal(got, want)
    assert impl == "ref:concat"


def test_sum(lib):
    xs = [f32((3, 4)) for _ in range(3)]
    sc = (C.c_float * 3)(0.5, -1.0, 2.0)
    arr = (type(md((3, 4), "f32")) * 3)(*[md((3, 4), "f32") for _ in xs])
    bufs = {MULTIPLE_SRC + i: x for i, x in enumerate(xs)}
    bufs[ARG["dst"]] = out((3, 4), "f32")
    got, impl = lib.call("aiednn_sum_create", [C.c_int(3), sc, arr, C.byref(md((3, 4), "f32"))], bufs, ARG["dst"])
    np.testing.assert_allclose(got, 0.5 * xs[0] - xs[1] + 2 * xs[2], rtol=1e-6, atol=1e-6)   # f32 rounding near 0
    assert impl == "ref:sum"


@pytest.mark.parametrize("src_dt,dst_dt,perm", [("f32", "s8", (0, 2, 3, 1)), ("s8", "f32", (3, 2, 1, 0)),
                                                ("s32", "u8", None), ("f32", "f32", (1, 0, 2, 3)),
                                                # same type, a batched transpose (reorder_transpose)
                                                ("f32", "f32", (0, 2, 3, 1)), ("f32", "f32", (0, 3, 1, 2)),
                                                ("f32", "f32", (2, 3, 0, 1)), ("s8", "s8", (0, 2, 3, 1))])
def test_reorder(lib, src_dt, dst_dt, perm):
    shape = (2, 3, 4, 5)
    x = {"f32": f32(shape, -20, 20), "s8": i8(shape), "s32": RNG.integers(-500, 500, shape).astype(np.int32)}[src_dt]
    scale, zp = 0.25, 3
    y = np.transpose(x, perm) if perm else x
    if src_dt == dst_dt:
        want = y
    elif dst_dt == "s8":
        want = np.clip(np.rint(y.astype(np.float64) / scale) + zp, -128, 127)
    elif src_dt == "s8" and dst_dt == "f32":
        want = (y.astype(np.float64) - zp) * scale
    elif dst_dt == "u8":
        want = np.clip(y, 0, 255)
    else:
        want = y
    cperm = (C.c_int * 4)(*perm) if perm else None
    got, impl = lib.call("aiednn_reorder_create",
                         [C.byref(md(x.shape, src_dt)), C.byref(md(want.shape, dst_dt)), cperm, f(scale), C.c_int32(zp)],
                         {ARG["src"]: x, ARG["dst"]: out(want.shape, dst_dt)}, ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-6)
    # the library carries every megakernel entry (the board's dispatch): an integer -> 8-bit
    # reorder may run on the AIE step-table mode (exact, aiednn_rt_steps.h)
    assert impl == "ref:reorder" or impl.startswith("steps:"), impl


def test_shuffle(lib):
    x = i8((2, 12, 3, 3))
    want = torch.nn.ChannelShuffle(3)(torch.from_numpy(x)).numpy()
    got, impl = lib.call("aiednn_shuffle_forward_create",
                         [C.c_int(1), C.c_int64(3), C.byref(md(x.shape, "s8")), C.byref(md(x.shape, "s8"))],
                         {ARG["src"]: x, ARG["dst"]: out(x.shape, "s8")}, ARG["dst"])
    np.testing.assert_array_equal(got, want)
    assert impl == "ref:shuffle"


# ── dispatch rules ────────────────────────────────────────────────────────

def test_impl_filter_forces_ref(lib):
    x, w = i8((1, 4, 6, 6)), i8((5, 4, 3, 3))
    lib.set_filter("ref:")
    try:
        got, impl = run_conv(lib, "aiednn_convolution_forward_create", x, w, None, (1, 5, 4, 4), conv_params(), "s8",
                             "s32")
    finally:
        lib.set_filter(None)
    assert impl == "ref:convolution"
    np.testing.assert_array_equal(got, torch_conv(x, w, None, (1, 1), (0, 0), (0, 0), (0, 0), 1).astype(np.int64))


def test_without_aie_kernels_everything_is_ref(tmp_path):
    ref = AiednnLib.build(tmp_path, with_gemm=False)
    x, w = i8((1, 4, 6, 6)), i8((5, 4, 3, 3))
    _, impl = run_conv(ref, "aiednn_convolution_forward_create", x, w, None, (1, 5, 4, 4), conv_params(), "s8", "s32")
    assert impl == "ref:convolution"


@pytest.mark.parametrize("fn,args", [
    ("aiednn_convolution_forward_create", "badspatial"),
    ("aiednn_eltwise_forward_create", "badalg"),
    ("aiednn_softmax_forward_create", "int"),
])
def test_invalid_arguments(lib, fn, args):
    if args == "badspatial":
        prim, st = lib.create(fn, C.byref(md((1, 4, 6, 6), "s8")), C.byref(md((5, 4, 3, 3), "s8")), None,
                              C.byref(md((1, 5, 5, 5), "s32")), C.byref(conv_params()))
    elif args == "badalg":
        prim, st = lib.create(fn, C.c_int(999), f(0), f(0), C.byref(md((2, 2), "f32")), C.byref(md((2, 2), "f32")))
    else:
        prim, st = lib.create(fn, C.c_int(lib.ALG["softmax_accurate"]), C.c_int(0), C.byref(md((2, 2), "s8")),
                              C.byref(md((2, 2), "s8")))
    assert st == 2                                       # aiednn_invalid_arguments


def test_execute_missing_argument(lib):
    x = f32((2, 3))
    prim, st = lib.create("aiednn_eltwise_forward_create", C.c_int(lib.ALG["eltwise_relu"]), f(0), f(0),
                          C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "f32")))
    assert st == 0
    assert lib.execute(prim, {ARG["src"]: x}) == 2       # no DST
    lib.destroy(prim)


# ── fused epilogue (aiednn_attr_t) ────────────────────────────────────────

from aiegraph.dnn_ctypes import POST_OP_SRC, attr  # noqa: E402


def epilogue_ref(acc, scales, bias, ch_axis, post, dst_dt, dst_scale=1.0, zp=0):
    shape = [1] * acc.ndim
    shape[ch_axis] = -1
    v = acc.astype(np.float64) * np.asarray(scales, np.float64).reshape(shape) + np.asarray(bias, np.float64).reshape(shape)
    for kind, arg in post:
        v = np.maximum(v, 0) if kind == "relu" else v + arg if kind == "add" else np.clip(v, 0, 6)
    if dst_dt == "f32":
        return v
    return np.clip(np.rint(v / dst_scale) + zp, -128 if dst_dt == "s8" else 0, 127 if dst_dt == "s8" else 255)


@pytest.mark.parametrize("dst_dt", ["f32", "s8", "u8"])
@pytest.mark.parametrize("groups", [1, 2])
def test_conv_fused_epilogue(lib, dst_dt, groups):
    N, Cc, H, W, O = 2, 16, 9, 9, 32
    x, w = i8((N, Cc, H, W)), i8((O, Cc // groups, 3, 3))
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    res = f32((N, O, 9, 9))
    acc = torch_conv(x, w, None, (1, 1), (1, 1), (1, 1), (0, 0), groups)
    want = epilogue_ref(acc, sc, b, 1, [("add", res), ("relu", None)], dst_dt, 0.05, 3 if dst_dt == "u8" else 0)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=3 if dst_dt == "u8" else 0,
                   post_ops=[("binary", "add", res.shape, "f32"), ("eltwise", "relu", 0.0, 0.0)])
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((N, O, 9, 9), dst_dt)), C.byref(conv_params((1, 1), (1, 1), (1, 1), (0, 0), groups)),
                                  C.byref(a)],
                                 {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, POST_OP_SRC: res,
                                  ARG["dst"]: out((N, O, 9, 9), dst_dt)}, ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    aie = outs.pop("im2col+" + AIE_GEMM)                  # the fused epilogue kept the AIE path
    ref = outs.pop("ref:convolution")
    np.testing.assert_array_equal(aie, ref)                # one epilogue: AIE == ref: bit for bit
    if dst_dt == "f32":
        np.testing.assert_allclose(aie, want, rtol=1e-5, atol=1e-6)
    else:
        assert np.abs(aie.astype(np.int64) - want).max() <= 1   # float rounding at .5 ties only
        assert (aie.astype(np.int64) != want).mean() < 1e-3



@pytest.mark.parametrize("dst_dt", ["s8", "u8"])
def test_conv_int8_residual_fixed_point(lib, dst_dt):
    """a fused residual add whose src1 is s8 (scale / zero point in the post-op's alpha / beta):
    the fixed-point epilogue carries it as (r - zr) * R; AIE == ref: bit for bit, within one
    code of the double formula with the dequantized residual"""
    N, Cc, O = 1, 16, 32
    x, w = i8((N, Cc, 9, 9)), i8((O, Cc, 3, 3))
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    r8, rs, rz = i8((N, O, 9, 9)), 0.03, 5
    zp = 3 if dst_dt == "u8" else 0
    acc = torch_conv(x, w, None, (1, 1), (1, 1), (1, 1), (0, 0), 1)
    want = epilogue_ref(acc, sc, b, 1, [("add", (r8.astype(np.float64) - rz) * rs), ("relu", None)], dst_dt, 0.05, zp)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=zp,
                   post_ops=[("binary", "add", r8.shape, "s8", rs, rz), ("eltwise", "relu", 0.0, 0.0)])
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((N, O, 9, 9), dst_dt)), C.byref(conv_params((1, 1), (1, 1), (1, 1), (0, 0), 1)),
                                  C.byref(a)],
                                 {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, POST_OP_SRC: r8,
                                  ARG["dst"]: out((N, O, 9, 9), dst_dt)}, ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    aie, ref = outs.pop("im2col+" + AIE_GEMM), outs.pop("ref:convolution")
    np.testing.assert_array_equal(aie, ref)
    assert np.abs(aie.astype(np.int64) - want).max() <= 1
    assert (aie.astype(np.int64) != want).mean() < 1e-2

@pytest.mark.parametrize("k", [1, 3])
def test_conv_batch_in_m_residual(lib, k):
    """a batch through the im2col pipes is ONE call with the images stacked along the GEMM's M
    (conv_geo.nimg): rows of several images share M blocks, and the int8 output and the s8 residual
    are mapped back per image -- AIE == ref: bit for bit"""
    N, Cc, O, H = 3, 16, 32, 7       # 3 * 49 rows: images share the first M block
    x, w = i8((N, Cc, H, H)), i8((O, Cc, k, k))
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    r8 = i8((N, O, H, H))
    pad = (k // 2, k // 2)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=0,
                   post_ops=[("binary", "add", r8.shape, "s8", 0.03, 5), ("eltwise", "relu", 0.0, 0.0)])
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((N, O, H, H), "s8")), C.byref(conv_params((1, 1), pad, pad, (0, 0), 1)),
                                  C.byref(a)],
                                 {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, POST_OP_SRC: r8,
                                  ARG["dst"]: out((N, O, H, H), "s8")}, ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    aie, ref = outs.pop("im2col+" + AIE_GEMM), outs.pop("ref:convolution")
    np.testing.assert_array_equal(aie, ref)


def test_matmul_fused_per_tensor_scale_and_clip(lib):
    a_, w = i8((5, 33, 200), -30, 30), i8((5, 200, 70), -30, 30)
    b = RNG.integers(-1000, 1000, (1, 1, 70)).astype(np.int32)
    acc = np.matmul(a_.astype(np.float64), w.astype(np.float64))
    want = np.clip((acc * 0.002 + b), 0, 6)
    at, keep = attr(lib.ALG, scales=[0.002], post_ops=[("eltwise", "clip", 0.0, 6.0)])
    got, impl = lib.call("aiednn_matmul_forward_create_attr",
                         [C.byref(md(a_.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "s32")),
                          C.byref(md((5, 33, 70), "f32")), C.c_int(0), C.byref(at)],
                         {ARG["src"]: a_, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: out((5, 33, 70), "f32")}, ARG["dst"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert impl == AIE_GEMM


def test_attr_validation(lib):
    a, keep = attr(lib.ALG, scales=np.ones(7, np.float32))             # 7 scales for 8 channels
    prim, st = lib.create("aiednn_convolution_forward_create_attr", C.byref(md((1, 4, 6, 6), "s8")),
                          C.byref(md((8, 4, 3, 3), "s8")), None, C.byref(md((1, 8, 4, 4), "f32")),
                          C.byref(conv_params()), C.byref(a))
    assert st == 2
    prim, st = lib.create("aiednn_convolution_forward_create", C.byref(md((1, 4, 6, 6), "s8")),
                          C.byref(md((8, 4, 3, 3), "s8")), None, C.byref(md((1, 8, 4, 4), "f32")), C.byref(conv_params()))
    assert st == 2                                                      # f32 dst from int8 needs an attr


# ── depthwise convolution on the AIE block (dwblk:aie:dw_s8s32) ──────────

@pytest.mark.parametrize("k,s,hw", [(3, 1, 13), (3, 2, 14), (5, 1, 11), (5, 2, 17), (3, 1, 112), (7, 1, 56), (7, 2, 15)])
def test_depthwise_on_aie_block(lib, k, s, hw):
    """the depthwise overlay of the GEMM ELF: banded items, exact int32 accumulators,
    the shared epilogue (scales, bias, ReLU6) -> bit-identical to ref:convolution"""
    N, Cc, p = 2, 24, k // 2
    Ho = (hw + 2 * p - k) // s + 1
    x, w = i8((N, Cc, hw, hw)), i8((Cc, 1, k, k))
    sc, b = f32((Cc,), 1e-3, 1e-2), f32((Cc,), -1, 1)
    acc = torch_conv(x, w, None, (s, s), (p, p), (p, p), (0, 0), Cc)
    a, keep = attr(lib.ALG, scales=sc, post_ops=[("eltwise", "clip", 0.0, 6.0)])
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((N, Cc, Ho, Ho), "f32")),
                                  C.byref(conv_params((s, s), (p, p), (p, p), (0, 0), Cc)), C.byref(a)],
                                 {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: out((N, Cc, Ho, Ho), "f32")},
                                 ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    assert "dwblk:aie:dw_s8s32" in outs, outs.keys()       # depthwise now runs on the AIE block
    np.testing.assert_array_equal(outs["dwblk:aie:dw_s8s32"], outs["ref:convolution"])
    want = epilogue_ref(acc, sc, b, 1, [("clip6", None)], "f32")
    np.testing.assert_allclose(outs["dwblk:aie:dw_s8s32"], want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("alg,k,s,p,hw", [("pooling_max", 3, 2, 1, 56), ("pooling_max", 2, 2, 0, 64),
                                          ("pooling_avg_include_padding", 3, 1, 1, 13),
                                          ("pooling_avg_exclude_padding", 3, 2, 1, 63)])
def test_pooling_on_aie_block(lib, alg, k, s, p, hw):
    """the pool overlay of the GEMM ELF (poolblk:aie:pool_s8s32) == ref:pooling bit for bit (problems with
    enough window work per item: RT_PL_MIN_WORK leaves tiny ones on ref:)"""
    N, Cc = 2, 8
    x = i8((N, Cc, hw, hw))
    Ho = (hw + 2 * p - k) // s + 1
    from aiegraph.dnn_ctypes import PoolParams
    pp = PoolParams()
    for i in range(2):
        pp.kernel[i], pp.strides[i], pp.padding_l[i], pp.padding_r[i], pp.dilates[i] = k, s, p, p, 0
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_pooling_forward_create",
                                 [C.c_int(lib.ALG[alg]), C.byref(md(x.shape, "s8")), C.byref(md((N, Cc, Ho, Ho), "s8")),
                                  C.byref(pp)], {ARG["src"]: x, ARG["dst"]: out((N, Cc, Ho, Ho), "s8")}, ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    assert "poolblk:aie:pool_s8s32" in outs, outs.keys()
    np.testing.assert_array_equal(outs["poolblk:aie:pool_s8s32"], outs["ref:pooling"])


def test_global_pool_placement_follows_the_preference(lib):
    """a global 7x7 pool has ~49 window ops per item: the AIE launches lose to ref: (board: 5.1 vs
    0.23 ms on 512 channels). The default preference still puts it on the AIE; aiednn_prefer_fastest
    lets the pool block refuse it (RT_PL_MIN_WORK)"""
    x = i8((1, 64, 7, 7))
    from aiegraph.dnn_ctypes import PoolParams
    pp = PoolParams()
    for i in range(2):
        pp.kernel[i], pp.strides[i] = 7, 1
    impls = []
    for pref in (0, 1):   # aiednn_prefer_aie, aiednn_prefer_fastest
        lib.so.aiednn_set_preference(pref)
        try:
            got, impl = lib.call("aiednn_pooling_forward_create",
                                 [C.c_int(lib.ALG["pooling_avg_exclude_padding"]), C.byref(md(x.shape, "s8")),
                                  C.byref(md((1, 64, 1, 1), "s8")), C.byref(pp)],
                                 {ARG["src"]: x, ARG["dst"]: out((1, 64, 1, 1), "s8")}, ARG["dst"])
        finally:
            lib.so.aiednn_set_preference(0)
        impls.append((impl, got.copy()))
    assert impls[0][0] != "ref:pooling" and impls[1][0] == "ref:pooling", impls
    np.testing.assert_array_equal(impls[0][1], impls[1][1])


@pytest.mark.parametrize("k,s,hw", [(3, 1, 14), (3, 2, 15), (5, 1, 9), (7, 1, 28)])
def test_depthwise_int8_requant_on_core(lib, k, s, hw):
    """s8 dst + ReLU6: the depthwise mode requantizes on the core (0xD8, fixed point); ref:
    computes the same integer epilogue -> bit-identical, within one code of the double formula"""
    N, Cc, p = 1, 16, k // 2
    Ho = (hw + 2 * p - k) // s + 1
    x, w = i8((N, Cc, hw, hw)), i8((Cc, 1, k, k))
    sc, b = f32((Cc,), 1e-3, 1e-2), f32((Cc,), -1, 1)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, post_ops=[("eltwise", "clip", 0.0, 6.0)])
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((N, Cc, Ho, Ho), "s8")),
                                  C.byref(conv_params((s, s), (p, p), (p, p), (0, 0), Cc)), C.byref(a)],
                                 {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: out((N, Cc, Ho, Ho), "s8")},
                                 ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    np.testing.assert_array_equal(outs["dwblk:aie:dw_s8s32"], outs["ref:convolution"])
    acc = torch_conv(x, w, None, (s, s), (p, p), (p, p), (0, 0), Cc)
    want = epilogue_ref(acc, sc, b, 1, [("clip6", None)], "s8", 0.05, 0)
    assert np.abs(outs["dwblk:aie:dw_s8s32"].astype(np.int64) - want).max() <= 1


@pytest.mark.parametrize("cin,dst_dt", [(96, "f32"), (96, "s8"), (64 * 25 // 25 * 25, "s8")])
def test_k_split_conv_chains_on_the_core(lib, cin, dst_dt):
    """K = cin*9 > 512: the K blocks chain on the core (gemm_kc: the core keeps the partial sum);
    an s8 dst with an integer epilogue also requantizes there (0xF1 continuing the chain).
    AIE == ref: bit for bit"""
    N, H, O = 1, 10, 40
    x, w = i8((N, cin, H, H)), i8((O, cin, 3, 3))
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05 if dst_dt == "s8" else 0.0,
                   post_ops=[("eltwise", "relu", 0.0, 0.0)])
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((N, O, H, H), dst_dt)), C.byref(conv_params((1, 1), (1, 1), (1, 1), (0, 0), 1)),
                                  C.byref(a)],
                                 {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: out((N, O, H, H), dst_dt)},
                                 ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    assert cin * 9 > 512
    np.testing.assert_array_equal(outs["im2col+" + AIE_GEMM], outs["ref:convolution"])


@pytest.mark.parametrize("dt,zp,relu", [("s8", 0, 0), ("s8", 3, 0), ("u8", 113, 1), ("s8", -5, 1)])
def test_batch_norm_create_attr_equals_batch_norm_then_quantize(lib, dt, zp, relu):
    """--aiednn-fold-norm-quant's runtime half: batch norm f32 -> s8 / u8 with dst_scale / zp is
    bit-identical to the f32 batch norm followed by the quantizing reorder -- including ties
    (identity BN, inputs on a 0.25 grid, dst_scale 0.5) and odd zero points, where
    round(x) + zp and round(x + zp) differ"""
    rng = np.random.default_rng(7)
    x = (rng.integers(-300, 300, size=(2, 4, 3, 5)) * 0.25).astype(np.float32)
    mean, var = np.zeros(4, np.float32), np.ones(4, np.float32)
    flags = FUSE_RELU if relu else 0
    base = {ARG["src"]: x, ARG["mean"]: mean, ARG["variance"]: var}
    bn, _ = lib.call("aiednn_batch_normalization_forward_inference_create",
                     [C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "f32")), f(0.0), C.c_uint(flags)],
                     {**base, ARG["dst"]: out(x.shape, "f32")}, ARG["dst"])
    want, _ = lib.call("aiednn_reorder_create",
                       [C.byref(md(x.shape, "f32")), C.byref(md(x.shape, dt)), None, f(0.5), C.c_int32(zp)],
                       {ARG["src"]: np.ascontiguousarray(bn, np.float32), ARG["dst"]: out(x.shape, dt)}, ARG["dst"])
    a, held = attr(lib.ALG, dst_scale=0.5, dst_zp=zp)
    got, impl = lib.call("aiednn_batch_normalization_forward_inference_create_attr",
                         [C.byref(md(x.shape, "f32")), C.byref(md(x.shape, dt)), f(0.0), C.c_uint(flags), C.byref(a)],
                         {**base, ARG["dst"]: out(x.shape, dt)}, ARG["dst"])
    assert np.array_equal(got, want) and impl.startswith("ref:")


def test_dead_channel_keeps_the_integer_epilogue(lib):
    """a conv output channel whose weights are all zero (output scale 0) has multiplier 0: the
    fixed-point epilogue says that exactly (M = 0) instead of falling back to the host double path
    for the whole layer (resnet18's stem: 802,816 outputs). Checked against the double epilogue"""
    rng = np.random.default_rng(3)
    x = rng.integers(-60, 60, size=(1, 8, 6, 6)).astype(np.int8)
    w = rng.integers(-60, 60, size=(4, 8, 3, 3)).astype(np.int8)
    w[2] = 0
    sc = np.array([1e-3, 2e-3, 0.0, 1.5e-3], np.float32)
    bias = np.array([0.1, -0.2, 0.3, 0.0], np.float32)
    a, held = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=3, post_ops=[("eltwise", "relu", 0.0, 0.0)])
    got, impl = lib.call("aiednn_convolution_forward_create_attr",
                         [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(bias.shape, "f32")),
                          C.byref(md((1, 4, 4, 4), "s8")), C.byref(conv_params()), C.byref(a)],
                         {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: bias, ARG["dst"]: out((1, 4, 4, 4), "s8")},
                         ARG["dst"])
    acc = np.zeros((1, 4, 4, 4), np.int64)
    for o in range(4):
        for i in range(4):
            for j in range(4):
                acc[0, o, i, j] = int((x[0, :, i:i + 3, j:j + 3].astype(np.int64) * w[o].astype(np.int64)).sum())
    v = np.maximum(acc * sc[None, :, None, None].astype(np.float64) + bias[None, :, None, None], 0) / 0.05 + 3
    want = np.clip(np.floor(v + 0.5), -128, 127).astype(np.int8)    # fixed point rounds half up
    assert (np.abs(got.astype(int) - want.astype(int)) <= 1).all()
    assert (got[0, 2] == want[0, 2]).all()                          # the dead channel: rounded bias exactly


def test_autotune_times_every_acceptor_and_keeps_one(lib, capfd):
    """aiednn_set_autotune(1): the first execute of a signature runs every accepting implementation
    (here the AIE im2col composite and ref:), keeps one, and the result is unchanged; a second
    primitive with the same signature takes the recorded choice; dump / load round-trip it"""
    x, w = i8((1, 8, 12, 12)), i8((16, 8, 3, 3))
    b = RNG.integers(-1000, 1000, 16).astype(np.int32)
    oshape = (1, 16, 10, 10)
    cp = conv_params((1, 1), (0, 0), (0, 0), (0, 0), 1)
    want = torch_conv(x, w, b, (1, 1), (0, 0), (0, 0), (0, 0), 1).astype(np.int64)
    lib.so.aiednn_set_autotune(1)
    lib.so.aiednn_set_verbose(1)
    lib.so.aiednn_set_preference(1)   # aiednn_prefer_fastest: ref: is a candidate too
    try:
        for _ in range(2):
            got, impl = run_conv(lib, "aiednn_convolution_forward_create", x, w, b, oshape, cp, "s8", "s32")
            np.testing.assert_array_equal(got, want)
        lib.so.aiednn_autotune_dump()
        C.CDLL(None).fflush(None)   # the C runtime's stdout buffer, before capfd reads
    finally:
        lib.so.aiednn_set_preference(0)
        lib.so.aiednn_set_autotune(0)
        lib.so.aiednn_set_verbose(0)
    out = capfd.readouterr().out
    assert out.count("aiednn_verbose,autotune,convolution") == 1, out   # timed once per signature
    assert "| ref:convolution" in out and "| im2col" in out
    lines = [l for l in out.splitlines() if l.startswith("AIEDNN_TUNE")]
    assert len(lines) >= 1 and lines[-1].split()[3] in ("im2col", "ref:convolution")
    assert lib.so.aiednn_autotune_load("\n".join(lines).encode()) == len(lines)


@pytest.mark.parametrize("groups", [1, 16])   # im2col GEMM (0xF1) and depthwise (0xD8)
def test_clamp_chain_stays_on_the_core_requant(lib, groups):
    """graph fusion level 2 keeps a folded requantize's saturation as a clip before the activation
    (clip(sat), clip(0, 6)); a run of clamps is one clamp, so the integer / on-core epilogue still takes
    it: the chain equals the single clip(0, 5.9) byte for byte, and the AIE equals ref:"""
    Cc = 16
    x, w = i8((1, Cc, 12, 12)), i8((Cc, Cc // groups, 3, 3))
    sc, b = f32((Cc,), 1e-3, 1e-2), f32((Cc,), -1, 1)
    res = {}
    for name, pos in (("chain", [("eltwise", "clip", -0.4, 5.9), ("eltwise", "clip", 0.0, 6.0)]),
                      ("one", [("eltwise", "clip", 0.0, 5.9)])):
        for filt in (None, "ref:"):
            a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, post_ops=pos)
            lib.set_filter(filt)
            try:
                got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                     [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                      C.byref(md((1, Cc, 12, 12), "s8")),
                                      C.byref(conv_params((1, 1), (1, 1), (1, 1), (0, 0), groups)), C.byref(a)],
                                     {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: out((1, Cc, 12, 12), "s8")},
                                     ARG["dst"])
            finally:
                lib.set_filter(None)
            res[(name, filt)] = (got, impl)
    assert "aie:" in res[("chain", None)][1], res[("chain", None)][1]
    np.testing.assert_array_equal(res[("chain", None)][0], res[("one", None)][0])
    np.testing.assert_array_equal(res[("chain", None)][0], res[("chain", "ref:")][0])


@pytest.mark.parametrize("kind", ["conv", "matmul"])
@pytest.mark.parametrize("filt", [None, "ref:"])
@pytest.mark.parametrize("ddt,post", [("s8", ()), ("u8", ("relu",)), ("s8", ("clip",)), ("f32", ("relu",))])
def test_requant_binary_post_op_equals_the_unfused_chain(lib, kind, filt, ddt, post):
    """aiednn_post_op_requant_binary (PT2E's conv -> quantize -> dequantize -> add -> [relu] -> quantize as one
    primitive): byte-identical to the GEMM with its int8 intermediate as dst followed by
    aiednn_binary_create_attr -- on the AIE path and on ref:"""
    from aiegraph.dnn_ctypes import Attr
    q_sc, q_zp, r_sc, r_zp, d_sc, d_zp = 0.047, -3, 0.031, 4, 0.052, (2 if ddt == "s8" else 120 if ddt == "u8" else 0)
    if kind == "conv":
        x, w = i8((1, 16, 10, 10)), i8((24, 16, 3, 3))
        oshape, O = (1, 24, 10, 10), 24
        args = lambda dt, a: [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md((O,), "f32")),
                              C.byref(md(oshape, dt)), C.byref(conv_params((1, 1), (1, 1), (1, 1), (0, 0), 1)), C.byref(a)]
        fn = "aiednn_convolution_forward_create_attr"
    else:
        x, w = i8((32, 64)), i8((64, 48))
        oshape, O = (32, 48), 48
        args = lambda dt, a: [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md((1, O), "f32")),
                              C.byref(md(oshape, dt)), C.c_int(0), C.byref(a)]
        fn = "aiednn_matmul_forward_create_attr"
    sc, b = f32((O,), 1e-3, 1e-2), f32((O,) if kind == "conv" else (1, O), -1, 1)
    r = i8(oshape)
    tail_po = [("eltwise", "relu", 0.0, 0.0)] if post == ("relu",) else [("eltwise", "clip", -0.5, 1.5)] if post else []
    odt = {"s8": np.int8, "u8": np.uint8, "f32": np.float32}[ddt]
    lib.set_filter(filt)
    try:
        # fused
        af, kf = attr(lib.ALG, scales=sc, dst_scale=d_sc if ddt != "f32" else 0.0, dst_zp=d_zp,
                      post_ops=[("requant_binary", "add", oshape, "s8", r_sc, r_zp, q_sc, q_zp, "s8")] + tail_po)
        got, impl = lib.call(fn, args(ddt, af), {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b,
                                                 POST_OP_SRC + 0: r, ARG["dst"]: out(oshape, ddt)}, ARG["dst"])
        # unfused: the GEMM to the int8 intermediate, then the quantized binary
        a1, k1 = attr(lib.ALG, scales=sc, dst_scale=q_sc, dst_zp=q_zp)
        mid, impl1 = lib.call(fn, args("s8", a1), {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b,
                                                   ARG["dst"]: out(oshape, "s8")}, ARG["dst"])
        a2, k2 = attr(lib.ALG, dst_scale=d_sc if ddt != "f32" else 0.0, dst_zp=d_zp, post_ops=tail_po)
        a2.src_scales[0], a2.src_scales[1], a2.src_zero_points[0], a2.src_zero_points[1] = q_sc, r_sc, q_zp, r_zp
        want, _ = lib.call("aiednn_binary_create_attr", [C.c_int(lib.ALG["binary_add"]), C.byref(md(oshape, "s8")),
                                                         C.byref(md(oshape, "s8")), C.byref(md(oshape, ddt)), C.byref(a2)],
                           {ARG["src"]: np.ascontiguousarray(mid), ARG["src_1"]: r, ARG["dst"]: out(oshape, ddt)}, ARG["dst"])
    finally:
        lib.set_filter(None)
    assert impl == impl1 + "+qres", (impl, impl1)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("groups", [1, 16])   # GEMM and depthwise
@pytest.mark.parametrize("filt", [None, "ref:"])
@pytest.mark.parametrize("alg,a,b,ddt,qdt", [("clip", 0.0, 6.0, "s8", "s8"), ("hardswish", 1 / 6, 0.5, "u8", "u8"),
                                              ("logistic", 0, 0, "s8", "s8"), ("gelu_erf", 0, 0, "f32", "s8")])
def test_requant_eltwise_post_op_equals_the_unfused_chain(lib, groups, filt, alg, a, b, ddt, qdt):
    """aiednn_post_op_requant_eltwise (conv -> Q -> DQ -> hardtanh / hardswish / sigmoid -> Q as one primitive):
    byte-identical to the conv with its int8 intermediate as dst followed by aiednn_eltwise_forward_create_attr"""
    Cc, H = 16, 10
    x, w = i8((1, Cc, H, H)), i8((Cc, Cc // groups, 3, 3))
    sc, bias = f32((Cc,), 1e-3, 1e-2), f32((Cc,), -1, 1)
    q_sc, q_zp = 0.047, (-3 if qdt == "s8" else 110)
    d_sc, d_zp = 0.031, (2 if ddt == "s8" else 0)
    odt = {"s8": np.int8, "u8": np.uint8, "f32": np.float32}[ddt]
    oshape = (1, Cc, H, H)
    cargs = lambda dt, at: [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md((Cc,), "f32")),
                            C.byref(md(oshape, dt)), C.byref(conv_params((1, 1), (1, 1), (1, 1), (0, 0), groups)), C.byref(at)]
    bufs = lambda dt: {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: bias, ARG["dst"]: out(oshape, dt)}
    lib.set_filter(filt)
    try:
        af, kf = attr(lib.ALG, scales=sc, dst_scale=d_sc if ddt != "f32" else 0.0, dst_zp=d_zp,
                      post_ops=[("requant_eltwise", alg, a, b, q_sc, q_zp, qdt)])
        got, impl = lib.call("aiednn_convolution_forward_create_attr", cargs(ddt, af), bufs(ddt), ARG["dst"])
        a1, k1 = attr(lib.ALG, scales=sc, dst_scale=q_sc, dst_zp=q_zp)
        mid, impl1 = lib.call("aiednn_convolution_forward_create_attr", cargs(qdt, a1), bufs(qdt), ARG["dst"])
        a2, k2 = attr(lib.ALG, dst_scale=d_sc if ddt != "f32" else 0.0, dst_zp=d_zp)
        a2.src_scales[0], a2.src_zero_points[0] = q_sc, q_zp
        want, _ = lib.call("aiednn_eltwise_forward_create_attr",
                           [C.c_int(lib.ALG["eltwise_" + alg]), C.c_float(a), C.c_float(b), C.byref(md(oshape, qdt)),
                            C.byref(md(oshape, ddt)), C.byref(a2)],
                           {ARG["src"]: np.ascontiguousarray(mid), ARG["dst"]: out(oshape, ddt)}, ARG["dst"])
    finally:
        lib.set_filter(None)
    assert impl == impl1 + "+qres", (impl, impl1)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("hk", [1, 2])           # MQA, GQA (H = 4)
@pytest.mark.parametrize("bshd", [0, 2 | 4])
@pytest.mark.parametrize("filt", [None, "ref:"])
def test_sdpa_gqa(lib, hk, bshd, filt):
    """grouped-query / multi-query attention: K, V with Hk heads, query head h reads head h // (H // Hk) --
    PyTorch's enable_gqa (= K, V repeat_interleave'd over the head axis)"""
    B, H, Sq, Skv, D, Dv = 2, 4, 9, 13, 8, 6
    q, k, v = f32((B, H, Sq, D)), f32((B, hk, Skv, D)), f32((B, hk, Skv, Dv))
    scale = 1.0 / math.sqrt(D)
    rep = lambda t: T(t).repeat_interleave(H // hk, dim=1)
    # non-causal: PyTorch's is_causal aligns the mask top-left when Sq != Skv, aiednn bottom-right (as oneDNN)
    want = F.scaled_dot_product_attention(T(q), rep(k), rep(v), scale=scale).numpy()
    tr = (lambda a: np.ascontiguousarray(a.transpose(0, 2, 1, 3))) if bshd else (lambda a: a)
    lib.set_filter(filt)
    try:
        got, impl = lib.call("aiednn_sdpa_forward_create",
                             [C.byref(md(tr(q).shape, "f32")), C.byref(md(tr(k).shape, "f32")),
                              C.byref(md(tr(v).shape, "f32")), None, C.byref(md(tr(want).shape, "f32")), f(scale),
                              C.c_int(bshd)],
                             {ARG["src"]: tr(q), ARG["src_1"]: tr(k), ARG["src_2"]: tr(v),
                              ARG["dst"]: out(tr(want).shape, "f32")}, ARG["dst"])
    finally:
        lib.set_filter(None)
    np.testing.assert_allclose(got, tr(want), rtol=1e-4, atol=1e-5)
    assert impl == "ref:sdpa" if filt else (impl.startswith("gemm+sdpa+") and "aie:" in impl), impl


def test_sdpa_heads_queued_across_launch_boundaries(lib):
    """gemm+sdpa defers each GEMM pass over all heads (b16q_defer): 36 score tiles and 48 P V tiles cross
    the 32-tile queue mid K chain, GQA, several N blocks -- the same result as PyTorch (bf16x3 tolerance)"""
    B, H, hk, Sq, Skv, D, Dv = 1, 6, 3, 40, 300, 64, 24
    q, k, v = f32((B, H, Sq, D)), f32((B, hk, Skv, D)), f32((B, hk, Skv, Dv))
    scale = 1.0 / math.sqrt(D)
    rep = lambda t: T(t).repeat_interleave(H // hk, dim=1)
    want = F.scaled_dot_product_attention(T(q), rep(k), rep(v), scale=scale).numpy()
    got, impl = lib.call("aiednn_sdpa_forward_create",
                         [C.byref(md(q.shape, "f32")), C.byref(md(k.shape, "f32")), C.byref(md(v.shape, "f32")), None,
                          C.byref(md(want.shape, "f32")), f(scale), C.c_int(0)],
                         {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v, ARG["dst"]: out(want.shape, "f32")}, ARG["dst"])
    assert impl.startswith("gemm+sdpa+") and "aie:" in impl, impl
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_sdpa_gqa_on_aie_block(lib):
    """GQA on sdpablk (int8 scores, quantized dst): within RT_SDPA_TOL codes of ref:sdpa, as the MHA case"""
    B, H, hk, Sq, Skv, D = 1, 8, 2, 40, 40, 16
    q, k, v = f32((B, H, Sq, D)), f32((B, hk, Skv, D)), f32((B, hk, Skv, D))
    a, keep = attr(lib.ALG, dst_scale=0.01, dst_zp=3)
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_sdpa_forward_create_attr",
                                 [C.byref(md(q.shape, "f32")), C.byref(md(k.shape, "f32")), C.byref(md(v.shape, "f32")),
                                  None, C.byref(md(q.shape, "s8")), f(1.0 / math.sqrt(D)), C.c_int(0), C.byref(a)],
                                 {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v, ARG["dst"]: out(q.shape, "s8")}, ARG["dst"])
        finally:
            lib.set_filter(None)
        outs[impl] = got
    aie, ref = outs.pop("sdpablk:" + AIE_SDPA), outs.pop("ref:sdpa")
    assert np.abs(aie.astype(int) - ref.astype(int)).max() <= 4


@pytest.mark.parametrize("hk,mask,causal,bshd", [(4, False, False, 0), (2, True, False, 0), (1, False, True, 0),
                                                  (2, False, True, 2)])
@pytest.mark.parametrize("filt", [None, "ref:"])
def test_sdpa_backward_vs_autograd(lib, hk, mask, causal, bshd, filt):
    """sdpa backward (dQ, dK, dV) against torch autograd: MHA / GQA / MQA, mask, causal (Sq = Skv), [B,S,H,D];
    gemm+sdpabwd puts its GEMMs on the AIE float GEMM"""
    B, H, S, D, Dv = 2, 4, 12, 8, 6
    q, k, v = f32((B, H, S, D)), f32((B, hk, S, D)), f32((B, hk, S, Dv))
    m = f32((1, H, S, S)) if mask else None
    dy = f32((B, H, S, Dv))
    qt, kt, vt = (torch.from_numpy(a.astype(np.float64)).requires_grad_(True) for a in (q, k, v))
    rep = lambda t: t.repeat_interleave(H // hk, dim=1)
    y = F.scaled_dot_product_attention(qt, rep(kt), rep(vt), attn_mask=None if m is None else T(m), is_causal=causal,
                                       scale=1 / math.sqrt(D))
    y.backward(T(dy))
    tr = (lambda a: np.ascontiguousarray(a.transpose(0, 2, 1, 3))) if bshd else (lambda a: a)
    bufs = {ARG["src"]: tr(q), ARG["src_1"]: tr(k), ARG["src_2"]: tr(v), 23: tr(dy),
            20: np.zeros(tr(q).shape, np.float32), 37: np.zeros(tr(k).shape, np.float32), 38: np.zeros(tr(v).shape, np.float32)}
    if mask:
        bufs[ARG["attn_mask"]] = m
    lib.set_filter(filt)
    try:
        prim, st = lib.create("aiednn_sdpa_backward_create", C.byref(md(tr(q).shape, "f32")), C.byref(md(tr(k).shape, "f32")),
                              C.byref(md(tr(v).shape, "f32")), C.byref(md(m.shape, "f32")) if mask else None,
                              C.byref(md(tr(dy).shape, "f32")), C.byref(md(tr(q).shape, "f32")), C.byref(md(tr(k).shape, "f32")),
                              C.byref(md(tr(v).shape, "f32")), f(1 / math.sqrt(D)), C.c_int(int(causal) | bshd))
        assert st == 0, lib.status(st)
        impl = lib.impl(prim)
        assert lib.execute(prim, bufs) == 0
        lib.destroy(prim)
    finally:
        lib.set_filter(None)
    assert impl == "ref:sdpa" if filt else (impl.startswith("gemm+sdpabwd+") and "aie:" in impl), impl
    for got, want in ((bufs[20], qt.grad), (bufs[37], kt.grad), (bufs[38], vt.grad)):
        np.testing.assert_allclose(got, tr(want.detach().numpy().astype(np.float32)), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ["sin", "cos", "tan"])
def test_eltwise_sin_cos_wide_range_on_fvec(lib, name):
    """FVEC sin / cos: Cody-Waite reduction by 2 pi, a fold into [-pi/2, pi/2] and the odd polynomial: |error|
    within 2e-6 over [-200, 200] (incl. the fold boundaries)"""
    x = f32((4, 1000), -200.0, 200.0)
    x[0, :8] = [np.pi / 2, -np.pi / 2, np.pi, -np.pi, 3 * np.pi / 2, 0.0, 1e-7, -1e-7]
    got, impl = lib.call("aiednn_eltwise_forward_create",
                         [C.c_int(lib.ALG["eltwise_" + name]), f(0.0), f(0.0), C.byref(md(x.shape, "f32")),
                          C.byref(md(x.shape, "f32"))], {ARG["src"]: x, ARG["dst"]: out(x.shape, "f32")}, ARG["dst"])
    want = {"sin": np.sin, "cos": np.cos, "tan": np.tan}[name](x.astype(np.float64))
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC else "ref:eltwise"), impl
    if name == "tan":  # relative near the poles (cos -> 0); x[0, :8] holds pi/2 -- skip those
        ok = np.abs(np.cos(x.astype(np.float64))) > 1e-3
        assert np.all(np.abs(got - want)[ok] <= 2e-5 * (1 + np.abs(want[ok]))), np.abs(got - want)[ok].max()
    else:
        assert np.abs(got - want).max() < 2e-6, np.abs(got - want).max()


@pytest.mark.parametrize("cdt", ["s8", "u8", "f32"])
def test_binary_select_on_fvec(lib, cdt):
    """select: three FVEC element passes ([c] a, [!c] b, add) -- each output exactly a or b (a -0.0 may come back as
    +0.0); broadcast operands are expanded on the host first (test_binary covers that)"""
    s = (2, 3, 4, 50)
    a, b = f32(s, -2, 2), f32(s, -2, 2)
    c = RNG.integers(0, 2, s).astype({"s8": np.int8, "u8": np.uint8, "f32": np.float32}[cdt])
    got, impl = lib.call("aiednn_binary_create",
                         [C.c_int(lib.ALG["binary_select"]), C.byref(md(s, "f32")), C.byref(md(s, "f32")),
                          C.byref(md(s, cdt)), C.byref(md(s, "f32"))],
                         {ARG["src"]: a, ARG["src_1"]: b, ARG["src_2"]: c, ARG["dst"]: out(s, "f32")}, ARG["dst"])
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC else "ref:binary"), impl
    np.testing.assert_array_equal(got, np.where(c != 0, a, b))
