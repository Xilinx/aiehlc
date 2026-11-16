###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Typed f32 resampling and LRN paths are byte-identical to the generic loops, on ref:."""
from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, f, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(23)
A = dict(ARG, diff_src=20, diff_dst=23)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_ref_fast"))


def both(lib, monkeypatch, fn, args, bufs, out):
    res = []
    for generic in (False, True):
        if generic:
            monkeypatch.setenv("AIEDNN_REF_GENERIC", "1")
        else:
            monkeypatch.delenv("AIEDNN_REF_GENERIC", raising=False)
        b = {k: v.copy() for k, v in bufs.items()}
        lib.so.aiednn_set_impl_filter(b"ref:")
        try:
            prim, st = lib.create(fn, *args)
            assert st == 0, lib.status(st)
            assert lib.execute(prim, b) == 0
            lib.destroy(prim)
        finally:
            lib.so.aiednn_set_impl_filter(None)
        res.append(b[out])
    monkeypatch.delenv("AIEDNN_REF_GENERIC", raising=False)
    np.testing.assert_array_equal(res[0].view(np.uint32), res[1].view(np.uint32))


RS = [((2, 3, 5, 7), (2, 3, 9, 4)), ((2, 3, 5, 7), (2, 3, 11, 3)), ((1, 2, 9), (1, 2, 4)), ((1, 2, 3, 4, 5), (1, 2, 5, 3, 8)),
      ((1, 4, 16, 16), (1, 4, 32, 32))]


@pytest.mark.parametrize("alg", ["resampling_nearest", "resampling_linear"])
@pytest.mark.parametrize("shape,oshape", RS)
def test_resampling_typed_equals_generic(lib, monkeypatch, alg, shape, oshape):
    x = RNG.uniform(-2, 2, shape).astype(np.float32)
    both(lib, monkeypatch, "aiednn_resampling_forward_create",
         [C.c_int(lib.ALG[alg]), C.byref(md(shape, "f32")), C.byref(md(oshape, "f32"))],
         {A["src"]: x, A["dst"]: np.zeros(oshape, np.float32)}, A["dst"])
    dy = RNG.uniform(-2, 2, oshape).astype(np.float32)
    both(lib, monkeypatch, "aiednn_resampling_backward_create",
         [C.c_int(lib.ALG[alg]), C.byref(md(shape, "f32")), C.byref(md(oshape, "f32"))],
         {A["diff_dst"]: dy, A["diff_src"]: np.zeros(shape, np.float32)}, A["diff_src"])


@pytest.mark.parametrize("alg", ["lrn_across_channels", "lrn_within_channel"])
@pytest.mark.parametrize("shape,ls", [((2, 6, 3, 4), 5), ((1, 16, 7, 9), 3), ((2, 5, 4, 4), 1)])
def test_lrn_typed_equals_generic(lib, monkeypatch, alg, shape, ls):
    x = RNG.uniform(-2, 2, shape).astype(np.float32)
    args = [C.c_int(lib.ALG[alg]), C.c_int64(ls), f(1e-1), f(0.75), f(2.0)]
    both(lib, monkeypatch, "aiednn_lrn_forward_create", args + [C.byref(md(shape, "f32")), C.byref(md(shape, "f32"))],
         {A["src"]: x, A["dst"]: np.zeros(shape, np.float32)}, A["dst"])
    dy = RNG.uniform(-1, 1, shape).astype(np.float32)
    both(lib, monkeypatch, "aiednn_lrn_backward_create",
         args + [C.byref(md(shape, "f32")), C.byref(md(shape, "f32")), C.byref(md(shape, "f32"))],
         {A["src"]: x, A["diff_dst"]: dy, A["diff_src"]: np.zeros(shape, np.float32)}, A["diff_src"])
