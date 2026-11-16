###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Weight-only quantization (group scales / zero points) and MX types e8m0 / f4_e2m1 against numpy."""

from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, ARG_WEI_SCALES, ARG_WEI_ZPS, AiednnLib, attr, md, pack4, unpack4  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(31)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_mx"))


def m(shape, dt="f32"):
    return C.byref(md(shape, dt))


def reorder(lib, x, dt, nbytes):
    out = np.zeros(nbytes, np.uint8)
    got, _ = lib.call("aiednn_reorder_create", [m(x.shape), m(x.shape, dt), None, C.c_float(1.0), C.c_int32(0)],
                      {ARG["src"]: x, ARG["dst"]: out}, ARG["dst"])
    back = np.zeros(x.shape, np.float32)
    lib.call("aiednn_reorder_create", [m(x.shape, dt), m(x.shape), None, C.c_float(1.0), C.c_int32(0)],
             {ARG["src"]: got, ARG["dst"]: back}, ARG["dst"])
    return got, back


def e8m0_model(x):
    """nearest power of two, ties to the even code, clamped; NaN / inf -> 0xff"""
    out = np.zeros(x.size, np.uint8)
    for i, v in enumerate(np.asarray(x, np.float64).reshape(-1)):
        if not np.isfinite(v):
            out[i] = 0xFF
            continue
        if v <= 0:
            continue
        e = int(np.floor(np.log2(v)))
        lo, hi = 2.0 ** e, 2.0 ** (e + 1)
        if v - lo > hi - v or (v - lo == hi - v and (e + 127) % 2):
            e += 1
        out[i] = min(max(e + 127, 0), 0xFE)
    return out


def e2m1_model(x):
    codes = []
    for v in np.asarray(x, np.float64).reshape(-1):
        if np.isnan(v):
            codes.append(0)
            continue
        a, sign = abs(v), 8 if np.signbit(v) else 0
        d = np.abs(E2M1 - a)
        best = np.flatnonzero(d == d.min())
        codes.append(sign | int(best[0] if len(best) == 1 else best[best % 2 == 0][0]))
    return np.array(codes)


def decode_e2m1(codes):
    return np.where(codes & 8, -1.0, 1.0) * E2M1[codes & 7]


def test_e8m0_conversion(lib):
    x = np.array([1.0, 1.5, 1.4999, 1.5001, 3.0, 0.75, 6.0, 5.0, 2.0 ** 100, 3 * 2.0 ** 126, 2.0 ** -127,
                  1.5 * 2.0 ** -127, 1.6 * 2.0 ** -127, 2.0 ** -140, 0.0, -2.0, np.inf, np.nan], np.float32)
    got, back = reorder(lib, x, "e8m0", x.size)
    np.testing.assert_array_equal(got, e8m0_model(x))
    want = np.where(got == 0xFF, np.nan, 2.0 ** (got.astype(np.float64) - 127))
    np.testing.assert_array_equal(back, want.astype(np.float32))


def test_f4_e2m1_conversion(lib):
    x = np.array([0, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 6, 7, 100, -0.25, -0.3, -2.5, -9, np.nan,
                  -0.0], np.float32)
    got, back = reorder(lib, x, "f4_e2m1", (x.size + 1) // 2)
    codes = unpack4(got, x.size, signed=False)
    np.testing.assert_array_equal(codes, e2m1_model(x))
    np.testing.assert_array_equal(back, decode_e2m1(codes).astype(np.float32))


def grouped_ref(x, wv, s, z, g):
    """x [M, K] @ dequant(w [K, N]) with per (k // g, n) scale / zero point"""
    deq = (wv - np.repeat(z, g, axis=0)) * np.repeat(s, g, axis=0)
    return x.astype(np.float64) @ deq


@pytest.mark.parametrize("wdt,zdt,sdt", [("s4", None, "f32"), ("u4", "u4", "f32"), ("s8", "s8", "f32"),
                                         ("u4", "s32", "bf16"), ("s4", None, "f16")])
def test_grouped_int_matmul(lib, wdt, zdt, sdt):
    """int4 / int8 weight-only quantization, group 32 along K (the LLM decode GEMM)"""
    M, K, N, g = 3, 96, 10, 32
    lo, hi = {"s4": (-8, 7), "u4": (0, 15), "s8": (-128, 127)}[wdt]
    x = RNG.uniform(-1, 1, (M, K)).astype(np.float32)
    wv = RNG.integers(lo, hi + 1, (K, N))
    wbuf = pack4(wv) if wdt in ("s4", "u4") else wv.astype(np.int8)
    s32 = RNG.uniform(0.01, 0.1, (K // g, N)).astype(np.float32)
    if sdt == "f32":
        sbuf, s = s32, s32.astype(np.float64)
    elif sdt == "f16":
        sbuf = s32.astype(np.float16)
        s = sbuf.astype(np.float64)
    else:                                   # bf16: the top half of the f32 bits (truncated, exact here)
        sbuf = (s32.view(np.uint32) >> 16).astype(np.uint16)
        s = (sbuf.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    z = np.zeros((K // g, N))
    bufs = {ARG["src"]: x, ARG["weights"]: wbuf, ARG["dst"]: np.zeros((M, N), np.float32), ARG_WEI_SCALES: sbuf}
    if zdt:
        zlo, zhi = {"u4": (0, 15), "s8": (-128, 127), "s32": (-3, 9)}[zdt]
        z = RNG.integers(zlo, zhi + 1, (K // g, N))
        bufs[ARG_WEI_ZPS] = pack4(z) if zdt == "u4" else z.astype(np.int8 if zdt == "s8" else np.int32)
    a, _k = attr(lib.ALG, wei_scales=((K // g, N), sdt), wei_zps=((K // g, N), zdt) if zdt else None)
    got, impl = lib.call("aiednn_matmul_forward_create_attr",
                         [m((M, K)), m((K, N), wdt), None, m((M, N)), C.c_int(0), C.byref(a)], bufs, ARG["dst"])
    assert impl.startswith("dequant+") and "aie:" in impl, impl   # dequantized to f32, bf16x3 on the AIE
    np.testing.assert_allclose(got, grouped_ref(x, wv, s, z, g), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("wdt", ["f4_e2m1", "f8_e4m3"])
def test_mx_matmul(lib, wdt):
    """MXFP4 / MXFP8: f4 / f8 elements, one e8m0 scale per 32 along K"""
    M, K, N, g = 4, 64, 6, 32
    x = RNG.uniform(-1, 1, (M, K)).astype(np.float32)
    scodes = RNG.integers(120, 132, (K // g, N)).astype(np.uint8)
    if wdt == "f4_e2m1":
        codes = RNG.integers(0, 16, (K, N))
        wbuf, wv = pack4(codes), decode_e2m1(codes)
    else:                                   # e4m3 codes away from NaN, decoded by the library itself
        wbuf = RNG.integers(0, 0x7E, (K, N)).astype(np.uint8) | (RNG.integers(0, 2, (K, N)) << 7).astype(np.uint8)
        wv = np.zeros((K, N), np.float32)
        lib.call("aiednn_reorder_create", [m((K, N), wdt), m((K, N)), None, C.c_float(1.0), C.c_int32(0)],
                 {ARG["src"]: wbuf, ARG["dst"]: wv}, ARG["dst"])
        wv = wv.astype(np.float64)
    a, _k = attr(lib.ALG, wei_scales=((K // g, N), "e8m0"))
    got, impl = lib.call("aiednn_matmul_forward_create_attr",
                         [m((M, K)), m((K, N), wdt), None, m((M, N)), C.c_int(0), C.byref(a)],
                         {ARG["src"]: x, ARG["weights"]: wbuf, ARG["dst"]: np.zeros((M, N), np.float32),
                          ARG_WEI_SCALES: scodes}, ARG["dst"])
    assert impl.startswith("dequant+") and "aie:" in impl, impl
    s = 2.0 ** (scodes.astype(np.float64) - 127)
    np.testing.assert_allclose(got, grouped_ref(x, wv, s, np.zeros_like(s), g), rtol=1e-5, atol=1e-6)


def test_grouped_inner_product(lib):
    """inner product weights are [OC, IC]: the group runs along IC, the scales are [IC / g, OC]"""
    N, IC, OC, g = 2, 64, 5, 16
    x = RNG.uniform(-1, 1, (N, IC)).astype(np.float32)
    wv = RNG.integers(-128, 128, (OC, IC))
    s = RNG.uniform(0.01, 0.1, (IC // g, OC)).astype(np.float32)
    b = RNG.uniform(-1, 1, OC).astype(np.float32)
    a, _k = attr(lib.ALG, wei_scales=((IC // g, OC), "f32"))
    got, impl = lib.call("aiednn_inner_product_forward_create_attr",
                         [m((N, IC)), m((OC, IC), "s8"), m((OC,)), m((N, OC)), C.byref(a)],
                         {ARG["src"]: x, ARG["weights"]: wv.astype(np.int8), ARG["bias"]: b,
                          ARG["dst"]: np.zeros((N, OC), np.float32), ARG_WEI_SCALES: s}, ARG["dst"])
    assert impl.startswith("dequant+") and "aie:" in impl, impl
    want = grouped_ref(x, wv.T, s.astype(np.float64), np.zeros(s.shape), g) + b
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_grouped_int8_src(lib):
    """s8 src x s4 weights with group scales and zero points -> f32: the exact reference."""
    M, K, N, g = 4, 128, 6, 64
    x = RNG.integers(-128, 128, (M, K))
    wv = RNG.integers(-8, 8, (K, N))
    s = RNG.uniform(0.01, 0.1, (K // g, N)).astype(np.float32)
    z = RNG.integers(-2, 3, (K // g, N))
    a, _k = attr(lib.ALG, wei_scales=((K // g, N), "f32"), wei_zps=((K // g, N), "s8"))
    got, impl = lib.call("aiednn_matmul_forward_create_attr",
                         [m((M, K), "s8"), m((K, N), "s4"), None, m((M, N)), C.c_int(0), C.byref(a)],
                         {ARG["src"]: x.astype(np.int8), ARG["weights"]: pack4(wv), ARG["dst"]: np.zeros((M, N), np.float32),
                          ARG_WEI_SCALES: s, ARG_WEI_ZPS: z.astype(np.int8)}, ARG["dst"])
    assert impl == "wq:aie:wq_s8f32"   # the WQ mode (aiednn_rt_wq.h; test_aiednn_wq.py has the rest)
    np.testing.assert_allclose(got, grouped_ref(x, wv, s.astype(np.float64), z, g), rtol=1e-5, atol=1e-4)


REASON = {"group": "the group dividing K", "kind": "matmul / inner product only", "s32": "types:",   # s32 weights never reach the group check: the type check refuses them
          "zp_only": "need weight scales", "int_src": "types:"}   # s8 src needs a floating dst


@pytest.mark.parametrize("case", list(REASON))
def test_grouped_refusals(lib, case, capfd):
    K, N = 64, 8
    sc = ((K // 3 if case == "group" else K // 32, N), "f32")
    a, _k = attr(lib.ALG, wei_scales=None if case == "zp_only" else sc, wei_zps=((2, N), "s8") if case == "zp_only" else None)
    wdt = "s32" if case == "s32" else "s8"
    sdt = "s8" if case == "int_src" else "f32"
    lib.so.aiednn_set_verbose(1)
    if case == "kind":
        from aiegraph.dnn_ctypes import conv_params
        cp = conv_params()
        _, st = lib.create("aiednn_convolution_forward_create_attr", m((1, K, 4, 4)), m((N, K, 1, 1), "s8"), None,
                           m((1, N, 4, 4)), C.byref(cp), C.byref(a))
    else:
        _, st = lib.create("aiednn_matmul_forward_create_attr", m((2, K), sdt), m((K, N), wdt), None,
                           m((2, N), "s32" if case == "int_src" else "f32"), C.c_int(0), C.byref(a))
    lib.so.aiednn_set_verbose(0)
    C.CDLL(None).fflush(None)   # the C runtime's stdout buffer, before capfd reads
    assert st != 0
    assert REASON[case] in capfd.readouterr().out


@pytest.mark.parametrize("et,emax,vmax", [("f8_e4m3", 8, 448.0), ("f8_e5m2", 15, 57344.0), ("f4_e2m1", 2, 6.0)])
def test_block_quantize_and_dequantize(lib, et, emax, vmax):
    """MX quantize / dequantize per block of 32 against the numpy model."""
    rng = np.random.default_rng(5)
    R, K, blk = 3, 128, 32
    x = (rng.standard_normal((R, K)) * rng.choice([1e-3, 1.0, 50.0, 3e4], (R, K))).astype(np.float32)
    x[1, 32:64] = 0.0                                     # a zero block
    x[2, :32] *= 1e-30                                    # tiny values
    nb = K // blk
    xb = x.reshape(R, nb, blk).astype(np.float64)
    amax = np.abs(xb).max(-1)
    se = np.where(amax > 0, np.frexp(amax)[1] - 1 - emax, -127).clip(-127, 127)
    want_sc = (se + 127).astype(np.uint8)
    scaled = np.clip(xb / np.exp2(se)[..., None], -vmax, vmax).reshape(R, K).astype(np.float32)
    nbytes = R * K if et != "f4_e2m1" else R * K // 2
    want_el, _ = reorder(lib, scaled, et, nbytes)
    el = np.zeros(nbytes, np.uint8)
    sc = np.zeros((R, nb), np.uint8)
    got, impl = lib.call("aiednn_block_quantize_create", [m((R, K)), m((R, K), et), m((R, nb), "e8m0"), C.c_int64(blk)],
                         {ARG["src"]: x, ARG["dst"]: el, ARG["scale"]: sc}, ARG["dst"])
    assert impl == "ref:block_quantize"
    np.testing.assert_array_equal(sc, want_sc)
    np.testing.assert_array_equal(got, want_el)
    # dequantize: decode (reorder back) times 2^s
    _, dec = reorder(lib, scaled, et, nbytes)
    back = np.zeros((R, K), np.float32)
    got2, _ = lib.call("aiednn_block_dequantize_create", [m((R, K), et), m((R, nb), "e8m0"), m((R, K)), C.c_int64(blk)],
                       {ARG["src"]: got, ARG["scale"]: sc, ARG["dst"]: back}, ARG["dst"])
    want2 = (dec.reshape(R, nb, blk).astype(np.float64) * np.exp2(se)[..., None]).reshape(R, K).astype(np.float32)
    np.testing.assert_array_equal(got2, want2)
