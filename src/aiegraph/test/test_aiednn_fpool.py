###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""f32 windowed pooling on the AIE (fpool:) against ref:, through ctypes on the workstation build."""
from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, md, pool_nd_params, pool_params  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(61)
FP = "fpool:aie:fpool_f32"
A = dict(ARG, diff_src=20, diff_dst=23)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_fpool"))


def run2(lib, shape, alg, k, s, pad, filt=None, x=None):
    N, Cc, H, W = shape
    Ho, Wo = (H + 2 * pad[0] - k[0]) // s[0] + 1, (W + 2 * pad[1] - k[1]) // s[1] + 1
    x = RNG.uniform(-3, 3, shape).astype(np.float32) if x is None else x
    pp = pool_params(k, s, pad, pad)
    lib.so.aiednn_set_impl_filter(filt.encode() if filt else None)
    try:
        return lib.call("aiednn_pooling_forward_create", [C.c_int(lib.ALG[alg]), C.byref(md(shape, "f32")),
                        C.byref(md((N, Cc, Ho, Wo), "f32")), C.byref(pp)],
                        {ARG["src"]: x, ARG["dst"]: np.zeros((N, Cc, Ho, Wo), np.float32)}, ARG["dst"]), x
    finally:
        lib.so.aiednn_set_impl_filter(None)


@pytest.mark.parametrize("alg", ["pooling_max", "pooling_avg_include_padding", "pooling_avg_exclude_padding"])
@pytest.mark.parametrize("shape,k,s,pad", [((1, 16, 28, 28), (2, 2), (2, 2), (0, 0)),
                                           ((2, 8, 56, 56), (3, 3), (2, 2), (1, 1)),     # resnet's stem pool
                                           ((1, 4, 17, 45), (3, 5), (1, 1), (1, 2))])    # stride 1, odd sizes, tails
def test_fpool_2d(lib, alg, shape, k, s, pad):
    (got, impl), x = run2(lib, shape, alg, k, s, pad)
    assert impl == FP
    (want, rimpl), _ = run2(lib, shape, alg, k, s, pad, "ref:", x)
    assert rimpl.startswith("ref:")
    if alg == "pooling_max":
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("alg", ["pooling_max", "pooling_avg_exclude_padding"])
def test_fpool_3d(lib, alg):
    """[1, 32, 8, 28, 28], 2x2x2 stride 2 -- the parity bench's pool3d -- and a padded 3x3x3 stride 1"""
    for shape, k, s, pad in [((1, 32, 8, 28, 28), (2, 2, 2), (2, 2, 2), (0, 0, 0)),
                             ((1, 4, 5, 9, 11), (3, 3, 3), (1, 1, 1), (1, 1, 1))]:
        N, Cc, D, H, W = shape
        out = tuple((n + 2 * p - kk) // ss + 1 for n, kk, ss, p in zip((D, H, W), k, s, pad))
        x = RNG.uniform(-3, 3, shape).astype(np.float32)
        pp = pool_nd_params(k, s, pad, pad)
        res = []
        for filt in (None, b"ref:"):
            lib.so.aiednn_set_impl_filter(filt)
            try:
                res.append(lib.call("aiednn_pooling_forward_create_nd",
                                    [C.c_int(lib.ALG[alg]), C.byref(md(shape, "f32")), C.byref(md((N, Cc) + out, "f32")),
                                     C.byref(pp)],
                                    {ARG["src"]: x, ARG["dst"]: np.zeros((N, Cc) + out, np.float32)}, ARG["dst"]))
            finally:
                lib.so.aiednn_set_impl_filter(None)
        (got, impl), (want, _) = res
        assert impl == FP
        if alg == "pooling_max":
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


def test_fpool_refuses_dilation_and_wide_stride(lib):
    (got, impl), _ = run2(lib, (1, 4, 20, 20), "pooling_max", (2, 2), (3, 3), (0, 0))
    assert impl.startswith("ref:")


@pytest.mark.parametrize("shape,k,s,pad", [((2, 8, 28, 28), (3, 3), (2, 2), (1, 1)),      # overlapping windows
                                           ((1, 4, 17, 45), (3, 5), (1, 1), (1, 2)),      # stride 1, heavy overlap
                                           ((1, 32, 8, 28, 28), (2, 2, 2), (2, 2, 2), (0, 0, 0)),
                                           ((1, 4, 5, 9, 11), (3, 3, 3), (1, 1, 1), (1, 1, 1))])
@pytest.mark.parametrize("ties", [False, True])
def test_fpool_max_backward_bit_identical(lib, shape, k, s, pad, ties):
    """Max pooling backward on the AIE: bit-identical to ref:, also with ties."""
    nd = len(shape)
    out = tuple((n + 2 * p - kk) // ss + 1 for n, kk, ss, p in zip(shape[2:], k, s, pad))
    yshape = shape[:2] + out
    x = (RNG.integers(-3, 4, shape) if ties else RNG.uniform(-3, 3, shape)).astype(np.float32)
    dy = RNG.uniform(-1, 1, yshape).astype(np.float32)
    pp = pool_nd_params(k, s, pad, pad)
    res = []
    for filt in (None, b"ref:"):
        lib.so.aiednn_set_impl_filter(filt)
        try:
            res.append(lib.call("aiednn_pooling_backward_create",
                                [C.c_int(lib.ALG["pooling_max"]), C.byref(md(shape, "f32")), C.byref(md(yshape, "f32")),
                                 C.byref(md(shape, "f32")), C.byref(pp)],
                                {A["diff_src"]: np.zeros(shape, np.float32), A["diff_dst"]: dy, A["src"]: x},
                                A["diff_src"]))
        finally:
            lib.so.aiednn_set_impl_filter(None)
    (got, impl), (want, rimpl) = res
    assert impl == FP and rimpl.startswith("ref:"), (impl, rimpl)
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


def test_fpool_avg_backward_is_a_depthwise_deconv(lib):
    """avg pooling backward is not an FPOOL mode: it runs as a depthwise deconvolution (deconv+avgpoolbwd)"""
    shape, yshape = (1, 4, 8, 8), (1, 4, 4, 4)
    pp = pool_nd_params((2, 2), (2, 2), (0, 0), (0, 0))
    got, impl = lib.call("aiednn_pooling_backward_create",
                         [C.c_int(lib.ALG["pooling_avg_include_padding"]), C.byref(md(shape, "f32")),
                          C.byref(md(yshape, "f32")), None, C.byref(pp)],
                         {A["diff_src"]: np.zeros(shape, np.float32),
                          A["diff_dst"]: RNG.uniform(-1, 1, yshape).astype(np.float32)}, A["diff_src"])
    assert impl.startswith("deconv+avgpoolbwd"), impl
