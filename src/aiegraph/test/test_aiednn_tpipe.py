###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The transposed conv tiling (run_t_pipe): same bytes as the plain tiling, fewer launches."""

from __future__ import annotations

import ctypes as C
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, DT, POST_OP_SRC, AiednnLib, attr, conv_params, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(31)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_tpipe"), mesh=(8, 4))   # BM 256, BN 128


def i8(shape):
    return RNG.integers(-128, 128, size=shape).astype(np.int8)


def conv(lib, x, w, b, oshape, cp, a, extra, fmt, ddt):
    m = lambda s, dt, f="dense": C.byref(md(s, dt, fmt=f))
    bufs = {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: np.zeros(oshape, DT[ddt][1]), **extra}
    if fmt == "nhwc":
        bufs[ARG["src"]] = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
        bufs[ARG["dst"]] = np.zeros((oshape[0], oshape[2], oshape[3], oshape[1]), DT[ddt][1])
    lib.entry_calls()
    got, impl = lib.call("aiednn_convolution_forward_create_attr",
                         [m(x.shape, "s8", fmt), m(w.shape, "s8"), m(b.shape, "f32"), m(oshape, ddt, fmt), C.byref(cp),
                          C.byref(a)], bufs, ARG["dst"])
    calls = sum(lib.entry_calls().values())
    return (got.transpose(0, 3, 1, 2) if fmt == "nhwc" else got), impl, calls


@pytest.mark.parametrize("fmt", ["dense", "nhwc"])
@pytest.mark.parametrize("res", [False, True], ids=["plain", "residual"])
@pytest.mark.parametrize("k,cin", [(3, 64), (1, 256)], ids=["3x3", "1x1"])
def test_layer4_shape_transposed(lib, fmt, res, k, cin):
    N, O, H = 1, 512, 7
    x, w = i8((N, cin, H, H)), i8((O, cin, k, k))
    sc, b = RNG.uniform(1e-4, 1e-3, O).astype(np.float32), RNG.uniform(-1, 1, O).astype(np.float32)
    r8 = i8((N, O, H, H))
    po = [("binary", "add", r8.shape, "s8", 0.03, 5, fmt)] if res else []
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, post_ops=po + [("eltwise", "relu", 0.0, 0.0)])
    extra = {POST_OP_SRC: np.ascontiguousarray(r8.transpose(0, 2, 3, 1)) if fmt == "nhwc" else r8} if res else {}
    cp = conv_params((1, 1), (k // 2, k // 2), (k // 2, k // 2), (0, 0), 1)
    os.environ["AIEDNN_NO_TPIPE"] = "1"
    try:
        want, i0, n0 = conv(lib, x, w, b, (N, O, H, H), cp, a, extra, fmt, "s8")
    finally:
        del os.environ["AIEDNN_NO_TPIPE"]
    got, i1, n1 = conv(lib, x, w, b, (N, O, H, H), cp, a, extra, fmt, "s8")
    np.testing.assert_array_equal(got, want)       # the int32 sums are exact; the host epilogue is the core's
    assert i0 == i1 and i1.startswith("im2col+"), (i0, i1)
    if res:   # a fused residual keeps the plain tiling off the n-tile launches: there the swap is fewer launches
        assert n1 < n0, (n1, n0)
    else:     # plain: the n-tile path batches every tile already, the swap is not taken
        assert n1 == n0, (n1, n0)


def test_large_m_keeps_the_plain_tiling(lib):
    """layer1-like (3136 pixels x 64 channels): the transposed tiling would take more launches -- not used"""
    x, w = i8((1, 64, 56, 56)), i8((64, 64, 3, 3))
    sc, b = RNG.uniform(1e-4, 1e-3, 64).astype(np.float32), RNG.uniform(-1, 1, 64).astype(np.float32)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05)
    cp = conv_params((1, 1), (1, 1), (1, 1), (0, 0), 1)
    os.environ["AIEDNN_NO_TPIPE"] = "1"
    try:
        want, _, n0 = conv(lib, x, w, b, (1, 64, 56, 56), cp, a, {}, "dense", "s8")
    finally:
        del os.environ["AIEDNN_NO_TPIPE"]
    got, _, n1 = conv(lib, x, w, b, (1, 64, 56, 56), cp, a, {}, "dense", "s8")
    np.testing.assert_array_equal(got, want)
    assert n1 == n0
