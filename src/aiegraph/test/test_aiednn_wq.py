###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Grouped weight-only quantization on the AIE (impl wq:, the GEMM ELF's WQ mode, aiednn_rt_wq.h): an
8-bit src times 8- / 4-bit weights with per-group scales and zero points, against the exact double
reference, through ctypes on the workstation build (the mode's C reference stands in for the array).

    python3 -m pytest -q src/aiegraph/test/test_aiednn_wq.py
"""
from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, ARG_WEI_SCALES, ARG_WEI_ZPS, AiednnLib, attr, md, pack4  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(57)
WQ = "wq:aie:wq_s8f32"
RANGE = {"s8": (-128, 128), "u8": (0, 256), "s4": (-8, 8), "u4": (0, 16)}
NPDT = {"s8": np.int8, "u8": np.uint8}


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_wq"))


def m(shape, dt="f32"):
    return C.byref(md(shape, dt))


def run(lib, M, K, N, g, sdt="s8", wdt="s8", zdt=None, src_zp=0, transposed=0):
    x = RNG.integers(*RANGE[sdt], (M, K))
    wv = RNG.integers(*RANGE[wdt], (K, N))
    s = RNG.uniform(0.005, 0.05, (K // g, N)).astype(np.float32)
    zr = {"s4": RANGE["s4"], "u4": RANGE["u4"], "u8": (120, 136), "s8": (-3, 4), "s32": (-3, 4)}
    z = RNG.integers(*(zr[zdt] if zdt else (0, 1)), (K // g, N))
    a, _keep = attr(lib.ALG, src_zp=src_zp, wei_scales=((K // g, N), "f32"),
                    wei_zps=((K // g, N), zdt) if zdt else None)
    wstore = wv.T if transposed else wv
    wbuf = pack4(np.ascontiguousarray(wstore)) if wdt in ("s4", "u4") else np.ascontiguousarray(wstore).astype(NPDT[wdt])
    bufs = {ARG["src"]: x.astype(NPDT[sdt]), ARG["weights"]: wbuf, ARG["dst"]: np.zeros((M, N), np.float32),
            ARG_WEI_SCALES: s}
    if zdt:
        bufs[ARG_WEI_ZPS] = pack4(z) if zdt in ("s4", "u4") else z.astype({"s8": np.int8, "u8": np.uint8, "s32": np.int32}[zdt])
    got, impl = lib.call("aiednn_matmul_forward_create_attr",
                         [m((M, K), sdt), m(wstore.shape, wdt), None, m((M, N)), C.c_int(transposed), C.byref(a)],
                         bufs, ARG["dst"])
    zr = z if zdt else np.zeros(s.shape)
    deq = (wv - np.repeat(zr, g, axis=0)) * np.repeat(s.astype(np.float64), g, axis=0)
    want = (x.astype(np.float64) - src_zp) @ deq
    mag = np.abs(x.astype(np.float64) - src_zp) @ np.abs(deq)
    return got, want, mag, impl


@pytest.mark.parametrize("M,K,N,g", [(4, 128, 6, 64),          # the library's s8-src reference case
                                     (37, 2048, 70, 128),      # four launches, groups inside launches
                                     (130, 1536, 33, 512),     # two M blocks, group = launch
                                     (5, 3072, 140, 1024),     # groups spanning launches, two N blocks
                                     (3, 640, 8, 320)])        # K not a multiple of 512, group of 5 rounds
def test_wq_shapes(lib, M, K, N, g):
    got, want, mag, impl = run(lib, M, K, N, g)
    assert impl == WQ
    assert np.all(np.abs(got - want) <= 2e-6 * (mag + 1e-30)), np.max(np.abs(got - want) / (mag + 1e-30))


@pytest.mark.parametrize("sdt,wdt,zdt,src_zp", [("s8", "s4", "s8", 0), ("u8", "u4", "u4", 113), ("u8", "u8", None, 128),
                                                ("s8", "u8", "u8", -5), ("u8", "s8", "s32", 0)])
def test_wq_types_and_zero_points(lib, sdt, wdt, zdt, src_zp):
    """u8 operands shifted by 128 and every zero point corrected per group on the host, exactly
    (the create API takes group scales with [K, N] weights only)"""
    got, want, mag, impl = run(lib, 9, 1024, 40, 128, sdt, wdt, zdt, src_zp)
    assert impl == WQ
    assert np.all(np.abs(got - want) <= 2e-6 * (mag + 1e-30)), np.max(np.abs(got - want) / (mag + 1e-30))


def test_wq_refuses_groups_off_the_round(lib, capfd):
    """a group that is not a multiple of the core's 64-K round stays on ref:"""
    got, want, mag, impl = run(lib, 4, 128, 8, 32)
    assert impl == "ref:matmul"
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)


def test_wq_inner_product(lib):
    """inner product [N, IC] x [OC, IC] with group scales [IC / g, OC]: the same WQ problem"""
    N, IC, OC, g = 7, 1024, 48, 128
    x = RNG.integers(-128, 128, (N, IC))
    wv = RNG.integers(-8, 8, (OC, IC))
    s = RNG.uniform(0.005, 0.05, (IC // g, OC)).astype(np.float32)
    z = RNG.integers(-3, 4, (IC // g, OC))
    a, _keep = attr(lib.ALG, wei_scales=((IC // g, OC), "f32"), wei_zps=((IC // g, OC), "s8"))
    got, impl = lib.call("aiednn_inner_product_forward_create_attr",
                         [m((N, IC), "s8"), m((OC, IC), "s4"), None, m((N, OC)), C.byref(a)],
                         {ARG["src"]: x.astype(np.int8), ARG["weights"]: pack4(wv), ARG["dst"]: np.zeros((N, OC), np.float32),
                          ARG_WEI_SCALES: s, ARG_WEI_ZPS: z.astype(np.int8)}, ARG["dst"])
    assert impl == WQ
    deq = (wv.T - np.repeat(z, g, axis=0)) * np.repeat(s.astype(np.float64), g, axis=0)
    want, mag = x.astype(np.float64) @ deq, np.abs(x.astype(np.float64)) @ np.abs(deq)
    assert np.all(np.abs(got - want) <= 2e-6 * (mag + 1e-30))


@pytest.mark.parametrize("dynq", [False, True])
def test_wq_dynamic_f32_src_opt_in(lib, monkeypatch, dynq):
    """a floating src stays dequant+ unless AIEDNN_WQ_DYNQ=1: then rows quantize to s8 (scale max|x| / 127)
    and take wq: -- within the int8 rounding of the activations"""
    if dynq:
        monkeypatch.setenv("AIEDNN_WQ_DYNQ", "1")
    else:
        monkeypatch.delenv("AIEDNN_WQ_DYNQ", raising=False)
    M, K, N, g = 5, 512, 40, 128
    x = RNG.uniform(-1, 1, (M, K)).astype(np.float32)
    wv = RNG.integers(-8, 8, (K, N))
    s = RNG.uniform(0.005, 0.05, (K // g, N)).astype(np.float32)
    a, _keep = attr(lib.ALG, wei_scales=((K // g, N), "f32"))
    got, impl = lib.call("aiednn_matmul_forward_create_attr",
                         [m((M, K)), m((K, N), "s4"), None, m((M, N)), C.c_int(0), C.byref(a)],
                         {ARG["src"]: x, ARG["weights"]: pack4(wv), ARG["dst"]: np.zeros((M, N), np.float32),
                          ARG_WEI_SCALES: s}, ARG["dst"])
    want = x.astype(np.float64) @ (wv * np.repeat(s.astype(np.float64), g, axis=0))
    if dynq:
        assert impl == WQ
        assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-2   # activations rounded to int8
    else:
        assert impl.startswith("dequant")
        assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-5
