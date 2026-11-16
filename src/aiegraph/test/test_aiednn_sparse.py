###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""CSR sparse-src matmul (aiednn_sparse_matmul_create) against the dense product."""

from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, DT, AiednnLib, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
IDX, PTR = 43, 44
RNG = np.random.default_rng(21)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_sparse"))


def csr(a):
    rows, cols = np.nonzero(a)
    ptr = np.zeros(a.shape[0] + 1, np.int32)
    np.add.at(ptr, rows + 1, 1)
    return a[rows, cols], cols.astype(np.int32), np.cumsum(ptr).astype(np.int32)


def run(lib, a, w, b, ddt, path, values=None, idx=None, ptr=None):
    M, K = a.shape
    vals, ix, rp = csr(a) if values is None else (values, idx, ptr)
    dst = np.zeros((M, w.shape[1]), DT[ddt][1])
    bufs = {ARG["src"]: np.ascontiguousarray(vals), ARG["weights"]: w, ARG["dst"]: dst, IDX: ix, PTR: rp}
    if b is not None:
        bufs[ARG["bias"]] = b
    sdt = "s8" if a.dtype == np.int8 else "f32"
    lib.set_filter(None if path == "densify" else path)
    try:
        got, impl = lib.call("aiednn_sparse_matmul_create",
                             [C.byref(md(vals.shape, sdt)), C.c_int64(M), C.c_int64(K), C.byref(md(w.shape, sdt)),
                              C.byref(md(b.shape, "s32" if sdt == "s8" else "f32")) if b is not None else None,
                              C.byref(md(dst.shape, ddt))], bufs, ARG["dst"])
    finally:
        lib.set_filter(None)
    assert impl.startswith(path), impl
    return got


@pytest.mark.parametrize("path", ["densify", "ref:"])
@pytest.mark.parametrize("density", [0.02, 0.3])
def test_int8(lib, path, density):
    M, K, N = 37, 300, 70
    a = np.where(RNG.uniform(size=(M, K)) < density, RNG.integers(-128, 128, (M, K)), 0).astype(np.int8)
    a[5] = 0                                                    # an empty row
    w = RNG.integers(-128, 128, (K, N)).astype(np.int8)
    b = RNG.integers(-1000, 1000, (1, N)).astype(np.int32)
    got = run(lib, a, w, b, "s32", path)
    np.testing.assert_array_equal(got, a.astype(np.int64) @ w.astype(np.int64) + b)   # exact on every path


@pytest.mark.parametrize("path", ["densify", "ref:"])
def test_f32(lib, path):
    M, K, N = 20, 64, 33
    a = np.where(RNG.uniform(size=(M, K)) < 0.2, RNG.normal(size=(M, K)), 0).astype(np.float32)
    w = RNG.normal(size=(K, N)).astype(np.float32)
    got = run(lib, a, w, None, "f32", path)
    np.testing.assert_allclose(got, a.astype(np.float64) @ w, rtol=1e-4, atol=1e-4)


def test_malformed_csr_is_a_runtime_error(lib):
    a = np.eye(4, dtype=np.float32)
    w = np.ones((4, 3), np.float32)
    vals, ix, rp = csr(a)
    ix = ix.copy()
    ix[2] = 9                                                   # column out of range
    with pytest.raises(RuntimeError, match="execute"):
        run(lib, a, w, None, "f32", "ref:", vals, ix, rp)


def test_prefer_fastest_keeps_very_sparse_on_the_host(lib):
    M, K, N = 64, 512, 64
    a = np.where(RNG.uniform(size=(M, K)) < 0.01, 1, 0).astype(np.int8)
    w = RNG.integers(-128, 128, (K, N)).astype(np.int8)
    lib.so.aiednn_set_preference(C.c_int(1))                  # aiednn_prefer_fastest
    try:
        got = run(lib, a, w, None, "s32", "ref:sparse_matmul")
    finally:
        lib.so.aiednn_set_preference(C.c_int(0))
    np.testing.assert_array_equal(got, a.astype(np.int64) @ w.astype(np.int64))
