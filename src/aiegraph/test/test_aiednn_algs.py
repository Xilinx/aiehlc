###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The pointwise / binary / reduction / softmax algorithms added for parity, against PyTorch.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_algs.py
"""

from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, f, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(21)
# the FVEC mode is registered when gemm_uk.cc carries its entry (aie_modes_fvec.h / aiednn_rt_fvec.h)
HAS_FVEC = "AIEK_SPEC(fvec_blk_f32" in (Path(__file__).resolve().parents[3] / "src" / "aietensorop" / "gemm" / "gemm_uk.cc").read_text()


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_algs"))


def m(a, dt="f32"):
    return C.byref(md(a.shape, dt))


def T(x):
    return torch.from_numpy(np.asarray(x, np.float64))


ELT = {"eltwise_tan": torch.tan, "eltwise_identity": lambda x: x, "eltwise_logical_not": torch.logical_not,
       "eltwise_sign": torch.sign, "eltwise_softsign": F.softsign, "eltwise_log_sigmoid": F.logsigmoid,
       "eltwise_selu": F.selu}


@pytest.mark.parametrize("alg", list(ELT))
def test_new_eltwise(lib, alg):
    x = RNG.uniform(-1.3, 1.3, (5, 31)).astype(np.float32)
    x[0, :3] = 0
    got, impl = lib.call("aiednn_eltwise_forward_create", [C.c_int(lib.ALG[alg]), f(0), f(0), m(x), m(x)],
                         {ARG["src"]: x, ARG["dst"]: np.zeros_like(x)}, ARG["dst"])
    on_fvec = alg in ("eltwise_identity", "eltwise_sign", "eltwise_softsign", "eltwise_log_sigmoid", "eltwise_selu",
                      "eltwise_logical_not", "eltwise_tan")
    assert impl == ("fvec:aie:fvec_f32" if HAS_FVEC and on_fvec else "ref:eltwise"), impl
    np.testing.assert_allclose(got, ELT[alg](T(x)).double().numpy(), rtol=1e-6, atol=1e-6)


def test_new_eltwise_int8_on_the_lut(lib):
    """an 8-bit eltwise of any algorithm runs on the AIE table (it is ref: on all 256 inputs)"""
    x = RNG.integers(-128, 128, (64, 64)).astype(np.int8)
    got, impl = lib.call("aiednn_eltwise_forward_create",
                         [C.c_int(lib.ALG["eltwise_softsign"]), f(0), f(0), m(x, "s8"), m(x, "f32")],
                         {ARG["src"]: x, ARG["dst"]: np.zeros(x.shape, np.float32)}, ARG["dst"])
    np.testing.assert_allclose(got, F.softsign(T(x)).float().numpy(), rtol=1e-6)


BIN = {"binary_fmod": torch.fmod, "binary_logical_and": torch.logical_and, "binary_logical_or": torch.logical_or,
       "binary_logical_xor": torch.logical_xor, "binary_add_square": lambda a, b: a + b * b}


@pytest.mark.parametrize("alg", list(BIN))
def test_new_binary(lib, alg):
    a = RNG.uniform(-3, 3, (2, 3, 4)).astype(np.float32)
    b = RNG.uniform(-2, 2, (1, 3, 1)).astype(np.float32)
    a[0, 0, :2] = 0
    got, impl = lib.call("aiednn_binary_create", [C.c_int(lib.ALG[alg]), m(a), m(b), None, m(a)],
                         {ARG["src"]: a, ARG["src_1"]: b, ARG["dst"]: np.zeros_like(a)}, ARG["dst"])
    np.testing.assert_allclose(got, BIN[alg](T(a), T(b)).double().numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("alg", ["reduction_norm_lp_max", "reduction_norm_lp_power_p_max", "reduction_mul_no_zeros"])
def test_new_reduction(lib, alg):
    x = RNG.uniform(-1.5, 1.5, (3, 4, 5)).astype(np.float32)
    x[0, 0, :2] = 0
    x[1] = 0     # a slice of only zeros: lp max takes eps, mul_no_zeros 1
    p, eps = 2.0, 1e-3
    got, impl = lib.call("aiednn_reduction_create", [C.c_int(lib.ALG[alg]), f(p), f(eps), m(x), C.byref(md((3, 1, 5), "f32"))],
                         {ARG["src"]: x, ARG["dst"]: np.zeros((3, 1, 5), np.float32)}, ARG["dst"])
    t = T(x)
    s = (t.abs() ** p).sum(1, keepdim=True)
    want = {"reduction_norm_lp_max": torch.clamp(s, min=eps) ** (1 / p),
            "reduction_norm_lp_power_p_max": torch.clamp(s, min=eps),
            "reduction_mul_no_zeros": torch.where(t == 0, torch.ones_like(t), t).prod(1, keepdim=True)}[alg]
    np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-6)


def test_softmax_inf_as_zero(lib):
    x = RNG.uniform(-2, 2, (3, 6)).astype(np.float32)
    x[1] = -np.inf
    got, _ = lib.call("aiednn_softmax_forward_create",
                      [C.c_int(lib.ALG["softmax_accurate_inf_as_zero"]), C.c_int(1), m(x), m(x)],
                      {ARG["src"]: x, ARG["dst"]: np.zeros_like(x)}, ARG["dst"])
    want = torch.softmax(T(x), 1).nan_to_num(0.0)
    np.testing.assert_allclose(got, want.numpy(), rtol=1e-6)


@pytest.mark.parametrize("dt,transposed", [("f32", 0), ("f32", 1), ("s8", 0)])
def test_grouped_matmul(lib, dt, transposed):
    """MoE: tokens sorted by expert, rows per expert at execute; int8 groups run on the AIE GEMM"""
    T_, K, N, E = 40, 24, 16, 4
    sizes = np.array([7, 0, 20, 9], np.int32)          # 36 of 40 rows: the last 4 untouched
    if dt == "f32":
        x = RNG.uniform(-1, 1, (T_, K)).astype(np.float32)
        w = RNG.uniform(-1, 1, (E, N, K) if transposed else (E, K, N)).astype(np.float32)
        b, odt = RNG.uniform(-1, 1, (E, N)).astype(np.float32), "f32"
        dst = np.full((T_, N), 7.0, np.float32)
    else:
        x = RNG.integers(-128, 128, (T_, K)).astype(np.int8)
        w = RNG.integers(-128, 128, (E, K, N)).astype(np.int8)
        b, odt = RNG.integers(-500, 500, (E, N)).astype(np.int32), "s32"
        dst = np.full((T_, N), 7, np.int32)
    mf = lambda a, t: C.byref(md(a.shape, t))
    got, impl = lib.call("aiednn_grouped_matmul_create",
                         [mf(x, dt), mf(w, dt), mf(b, odt), mf(dst, odt), C.c_int(transposed), None],
                         {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: dst, 26: sizes}, ARG["dst"])
    assert impl == "grouped"
    want, off = np.full((T_, N), 7.0), 0
    for e, n in enumerate(sizes):
        we = w[e].T if transposed else w[e]
        want[off:off + n] = x[off:off + n].astype(np.float64) @ we.astype(np.float64) + b[e]
        off += n
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_int4_weights_matmul_and_reorder(lib):
    """s4 / u4 weights decompressed in a float matmul (value x per-channel output scale), and reorder"""
    from aiegraph.dnn_ctypes import attr, pack4, unpack4
    M, K, N = 5, 9, 6
    x = RNG.uniform(-1, 1, (M, K)).astype(np.float32)
    for dt, lo, hi in (("s4", -8, 7), ("u4", 0, 15)):
        wv = RNG.integers(lo, hi + 1, (K, N))
        wb = pack4(wv)
        sc = RNG.uniform(0.01, 0.1, N).astype(np.float32)
        a, _keep = attr(lib.ALG, scales=sc)
        got, impl = lib.call("aiednn_matmul_forward_create_attr",
                             [C.byref(md(x.shape, "f32")), C.byref(md((K, N), dt)), None, C.byref(md((M, N), "f32")),
                              C.c_int(0), C.byref(a)],
                             {ARG["src"]: x, ARG["weights"]: wb, ARG["dst"]: np.zeros((M, N), np.float32)}, ARG["dst"])
        assert impl.startswith("cvt+") and "aie:" in impl, impl   # weights widened to f32, bf16x3 on the AIE
        np.testing.assert_allclose(got, (x.astype(np.float64) @ wv) * sc, rtol=1e-5, atol=1e-6)
        # reorder f32 -> 4-bit rounds to nearest even and saturates, and back
        fv = np.array([-9.0, -8.4, -0.5, 0.5, 1.5, 2.5, 6.6, 7.5, 16.0, 3.0, 4.2], np.float32)
        q = np.zeros((fv.size + 1) // 2, np.uint8)
        got, _ = lib.call("aiednn_reorder_create",
                          [C.byref(md(fv.shape, "f32")), C.byref(md(fv.shape, dt)), None, C.c_float(1.0), C.c_int32(0)],
                          {ARG["src"]: fv, ARG["dst"]: q}, ARG["dst"])
        want = np.clip(np.round(fv), lo, hi)      # numpy rounds half to even, as aiednn
        np.testing.assert_array_equal(unpack4(got, fv.size, dt == "s4"), want)


def test_src_and_weights_zero_points(lib):
    """asymmetric int8: acc = sum (src - zs) (w - zw); padding contributes 0; ref: only"""
    from aiegraph.dnn_ctypes import attr, conv_params
    zs, zw = 17, -3
    a, _keep = attr(lib.ALG, src_zp=zs, wei_zp=zw)
    x = RNG.integers(0, 256, (3, 5, 24)).astype(np.uint8)
    w = RNG.integers(-128, 128, (3, 24, 7)).astype(np.int8)
    got, impl = lib.call("aiednn_matmul_forward_create_attr",
                         [C.byref(md(x.shape, "u8")), C.byref(md(w.shape, "s8")), None, C.byref(md((3, 5, 7), "s32")),
                          C.c_int(0), C.byref(a)],
                         {ARG["src"]: x, ARG["weights"]: w, ARG["dst"]: np.zeros((3, 5, 7), np.int32)}, ARG["dst"])
    assert impl == "ref:matmul"
    np.testing.assert_array_equal(got, (x.astype(np.int64) - zs) @ (w.astype(np.int64) - zw))
    xc = RNG.integers(0, 256, (1, 4, 7, 6)).astype(np.uint8)
    wc = RNG.integers(-128, 128, (8, 4, 3, 3)).astype(np.int8)
    got, impl = lib.call("aiednn_convolution_forward_create_attr",
                         [C.byref(md(xc.shape, "u8")), C.byref(md(wc.shape, "s8")), None, C.byref(md((1, 8, 7, 6), "s32")),
                          C.byref(conv_params((1, 1), (1, 1), (1, 1))), C.byref(a)],
                         {ARG["src"]: xc, ARG["weights"]: wc, ARG["dst"]: np.zeros((1, 8, 7, 6), np.int32)}, ARG["dst"])
    assert impl == "ref:convolution"
    want = F.conv2d(F.pad(T(xc) - zs, (1, 1, 1, 1)), T(wc) - zw).numpy()
    np.testing.assert_array_equal(got, want.astype(np.int64))
