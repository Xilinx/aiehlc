###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The narrow floating types (bf16, f16, f8 e5m2 / e4m3) against PyTorch."""

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
from aiegraph.dnn_ctypes import ARG, AiednnLib, conv_params, f, md, pool_params  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(11)
TT = {"bf16": torch.bfloat16, "f16": torch.float16, "f8_e5m2": torch.float8_e5m2, "f8_e4m3": torch.float8_e4m3fn}
RAW = {"bf16": (np.uint16, torch.int16), "f16": (np.float16, None), "f8_e5m2": (np.uint8, torch.uint8),
       "f8_e4m3": (np.uint8, torch.uint8)}
ULP = {"bf16": 2.0 ** -7, "f16": 2.0 ** -10, "f8_e5m2": 2.0 ** -2, "f8_e4m3": 2.0 ** -3}
FLOATS = ["bf16", "f16"]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_dt"))


def to_raw(t, dt: str) -> np.ndarray:
    """a torch tensor -> the dtype's raw buffer (numpy)"""
    q = t.to(TT[dt])
    if dt == "f16":
        return q.numpy()
    npt, view = RAW[dt]
    return q.view(view).numpy().view(npt).copy()


def from_raw(a: np.ndarray, dt: str):
    """a raw buffer -> torch float64"""
    if dt == "f16":
        return torch.from_numpy(a.astype(np.float32)).double()
    npt, view = RAW[dt]
    return torch.from_numpy(a.view(np.int16 if view == torch.int16 else np.uint8).copy()).view(TT[dt]).double()


def rnd(shape, lo=-2.0, hi=2.0):
    return torch.from_numpy(RNG.uniform(lo, hi, size=shape)).double()


def buf(shape, dt):
    return np.zeros(shape, RAW[dt][0] if dt in RAW else np.float32)


def close(got: np.ndarray, want, dt: str, k=2.0, atol=0.0):
    g = from_raw(got, dt)
    w = want.to(TT[dt]).double()     # torch's own rounding of the float64 result
    tol = k * ULP[dt] * w.abs() + atol + 1e-30
    bad = ((g - w).abs() > tol) & ~(torch.isnan(g) & torch.isnan(w))
    assert int(bad.sum()) == 0, f"{int(bad.sum())} of {bad.numel()} differ, e.g. {g[bad][:4]} vs {w[bad][:4]}"


# ── conversions: bit-exact with torch, special values included ─────────────

@pytest.mark.parametrize("dt", ["bf16", "f16", "f8_e5m2", "f8_e4m3"])
def test_reorder_f32_to_narrow_bit_exact(lib, dt):
    specials = [0.0, -0.0, 1.0, -1.5, 1e-8, -1e-8, 3.0e38, -3.0e38, 65504.0, 70000.0, 448.0, 464.0, 500.0,
                57344.0, 61440.0, 1e-5, 6e-8, 2.0 ** -9, 2.0 ** -16, float("inf"), float("-inf"), float("nan")]
    x = np.concatenate([np.array(specials, np.float32), RNG.normal(0, 30, 4000).astype(np.float32),
                        RNG.normal(0, 1e-3, 1000).astype(np.float32)])
    for src_dt, dst_dt, src, want in ((("f32", dt, x, to_raw(torch.from_numpy(x), dt))),):
        got, impl = lib.call("aiednn_reorder_create",
                             [C.byref(md(x.shape, src_dt)), C.byref(md(x.shape, dst_dt)), None, f(1.0), C.c_int32(0)],
                             {ARG["src"]: src, ARG["dst"]: buf(x.shape, dst_dt)}, ARG["dst"])
        assert impl == "ref:reorder"
        nan = torch.isnan(from_raw(want, dst_dt)).numpy()   # NaN payloads differ between torch's own kernels
        assert torch.isnan(from_raw(got, dst_dt)).numpy()[nan].all()
        np.testing.assert_array_equal(got[~nan].view(np.uint8), want[~nan].view(np.uint8))
    # and back: narrow -> f32 is exact (every narrow value is an f32)
    raw = to_raw(torch.from_numpy(x), dt)
    got, _ = lib.call("aiednn_reorder_create",
                      [C.byref(md(x.shape, dt)), C.byref(md(x.shape, "f32")), None, f(1.0), C.c_int32(0)],
                      {ARG["src"]: raw, ARG["dst"]: np.zeros(x.shape, np.float32)}, ARG["dst"])
    np.testing.assert_array_equal(got.view(np.uint32), from_raw(raw, dt).float().numpy().view(np.uint32))


# ── primitives in each type ────────────────────────────────────────────────

@pytest.mark.parametrize("dt", FLOATS)
def test_conv_and_matmul(lib, dt):
    x, w, b = rnd((2, 6, 9, 8)), rnd((8, 3, 3, 3), -0.5, 0.5), rnd((8,))
    xr, wr, br = to_raw(x, dt), to_raw(w, dt), to_raw(b, dt)
    want = F.conv2d(F.pad(from_raw(xr, dt), (1, 0, 1, 1)), from_raw(wr, dt), from_raw(br, dt), stride=(1, 2), groups=2)
    got, impl = lib.call("aiednn_convolution_forward_create",
                         [C.byref(md(x.shape, dt)), C.byref(md(w.shape, dt)), C.byref(md(b.shape, dt)),
                          C.byref(md(tuple(want.shape), dt)), C.byref(conv_params((1, 2), (1, 1), (1, 0), (0, 0), 2))],
                         {ARG["src"]: xr, ARG["weights"]: wr, ARG["bias"]: br, ARG["dst"]: buf(tuple(want.shape), dt)},
                         ARG["dst"])
    assert impl.startswith("gemm+conv+") and "aie:" in impl, impl   # bf16: bf16blk, f16: widened, bf16x3
    close(got, want, dt, atol=1e-3)
    a, bm = rnd((3, 5, 7)), rnd((1, 7, 4))
    ar, bmr = to_raw(a, dt), to_raw(bm, dt)
    want = torch.matmul(from_raw(ar, dt), from_raw(bmr, dt))
    got, _ = lib.call("aiednn_matmul_forward_create",
                      [C.byref(md(a.shape, dt)), C.byref(md(bm.shape, dt)), None, C.byref(md((3, 5, 4), "f32")), C.c_int(0)],
                      {ARG["src"]: ar, ARG["weights"]: bmr, ARG["dst"]: np.zeros((3, 5, 4), np.float32)}, ARG["dst"])
    np.testing.assert_allclose(got, want.float().numpy(), rtol=1e-6, atol=1e-6)   # bf16 x bf16 -> f32 dst


@pytest.mark.parametrize("dt", FLOATS)
@pytest.mark.parametrize("alg", ["eltwise_gelu_erf", "eltwise_logistic", "eltwise_exp", "eltwise_relu"])
def test_eltwise(lib, dt, alg):
    x = rnd((4, 33))
    xr = to_raw(x, dt)
    xs = from_raw(xr, dt)
    want = {"eltwise_gelu_erf": F.gelu(xs), "eltwise_logistic": torch.sigmoid(xs), "eltwise_exp": torch.exp(xs),
            "eltwise_relu": F.relu(xs)}[alg]
    got, impl = lib.call("aiednn_eltwise_forward_create",
                         [C.c_int(lib.ALG[alg]), f(0.0), f(0.0), C.byref(md(x.shape, dt)), C.byref(md(x.shape, dt))],
                         {ARG["src"]: xr, ARG["dst"]: buf(x.shape, dt)}, ARG["dst"])
    assert impl == "ref:eltwise"
    close(got, want, dt, atol=1e-6)


@pytest.mark.parametrize("dt", FLOATS)
def test_binary_softmax_norm_pool(lib, dt):
    a, b = rnd((2, 3, 4, 5)), rnd((1, 3, 1, 5))
    ar, br = to_raw(a, dt), to_raw(b, dt)
    want = from_raw(ar, dt) * from_raw(br, dt)
    got, _ = lib.call("aiednn_binary_create",
                      [C.c_int(lib.ALG["binary_mul"]), C.byref(md(a.shape, dt)), C.byref(md(b.shape, dt)), None,
                       C.byref(md(a.shape, dt))],
                      {ARG["src"]: ar, ARG["src_1"]: br, ARG["dst"]: buf(a.shape, dt)}, ARG["dst"])
    close(got, want, dt, atol=1e-6)
    x = rnd((3, 4, 10))
    xr = to_raw(x, dt)
    want = torch.softmax(from_raw(xr, dt), dim=2)
    got, _ = lib.call("aiednn_softmax_forward_create",
                      [C.c_int(lib.ALG["softmax_accurate"]), C.c_int(2), C.byref(md(x.shape, dt)), C.byref(md(x.shape, dt))],
                      {ARG["src"]: xr, ARG["dst"]: buf(x.shape, dt)}, ARG["dst"])
    close(got, want, dt, atol=1e-4)
    want = F.layer_norm(from_raw(xr, dt), (10,), eps=1e-5)
    got, _ = lib.call("aiednn_layer_normalization_forward_inference_create",
                      [C.byref(md(x.shape, dt)), C.byref(md(x.shape, dt)), f(1e-5), C.c_uint(0)],
                      {ARG["src"]: xr, ARG["dst"]: buf(x.shape, dt)}, ARG["dst"])
    close(got, want, dt, atol=1e-2)
    p = rnd((1, 3, 8, 8))
    pr = to_raw(p, dt)
    want = F.max_pool2d(from_raw(pr, dt), 2, 2)
    got, _ = lib.call("aiednn_pooling_forward_create",
                      [C.c_int(lib.ALG["pooling_max"]), C.byref(md(p.shape, dt)), C.byref(md((1, 3, 4, 4), dt)),
                       C.byref(pool_params((2, 2), (2, 2), (0, 0), (0, 0)))],
                      {ARG["src"]: pr, ARG["dst"]: buf((1, 3, 4, 4), dt)}, ARG["dst"])
    np.testing.assert_array_equal(got.view(np.uint8), to_raw(want, dt).view(np.uint8))   # max is exact


def test_f8_eltwise_and_reorder_dequant(lib):
    """f8 in, f32 out: the stored f8 value through relu, bit for bit"""
    for dt in ("f8_e5m2", "f8_e4m3"):
        x = rnd((64,), -40, 40)
        xr = to_raw(x, dt)
        got, _ = lib.call("aiednn_eltwise_forward_create",
                          [C.c_int(lib.ALG["eltwise_relu"]), f(0.0), f(0.0), C.byref(md(x.shape, dt)),
                           C.byref(md(x.shape, "f32"))],
                          {ARG["src"]: xr, ARG["dst"]: np.zeros(x.shape, np.float32)}, ARG["dst"])
        np.testing.assert_array_equal(got, F.relu(from_raw(xr, dt)).float().numpy())
