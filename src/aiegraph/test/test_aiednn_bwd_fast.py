###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The typed f32 host paths of average-pooling backward and layer / RMS-norm backward (aiednn_rt_bwd.h)
are byte-identical to the generic loops they bypass (AIEDNN_BWD_GENERIC=1 forces those), on ref:.
test_aiednn_train.py checks both against PyTorch.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_bwd_fast.py
"""
from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, f, md, pool_nd_params  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(17)
A = dict(ARG, diff_src=20, diff_dst=23, diff_scale=24, diff_shift=25)
SCALE, SHIFT, RMS = 1, 2, 8


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_bwd_fast"))


def both(lib, monkeypatch, fn, args, bufs, outs):
    """[typed outputs, generic outputs], on ref:"""
    res = []
    for generic in (False, True):
        if generic:
            monkeypatch.setenv("AIEDNN_BWD_GENERIC", "1")
        else:
            monkeypatch.delenv("AIEDNN_BWD_GENERIC", raising=False)
        b = {k: v.copy() for k, v in bufs.items()}
        lib.so.aiednn_set_impl_filter(b"ref:")
        try:
            prim, st = lib.create(fn, *args)
            assert st == 0, lib.status(st)
            assert lib.execute(prim, b) == 0
            lib.destroy(prim)
        finally:
            lib.so.aiednn_set_impl_filter(None)
        res.append([b[o] for o in outs])
    monkeypatch.delenv("AIEDNN_BWD_GENERIC", raising=False)
    return res


@pytest.mark.parametrize("alg", ["pooling_avg_include_padding", "pooling_avg_exclude_padding"])
@pytest.mark.parametrize("sp,k,s,p", [((28, 28), (3, 3), (2, 2), (1, 1)), ((9, 10), (3, 2), (1, 1), (1, 0)),
                                      ((5, 6, 6), (2, 3, 3), (2, 2, 1), (0, 1, 1)), ((13,), (3,), (2,), (1,))])
def test_avg_pool_backward_typed_equals_generic(lib, monkeypatch, alg, sp, k, s, p):
    shape = (2, 5) + sp
    out = tuple((n + 2 * pp - kk) // ss + 1 for n, kk, ss, pp in zip(sp, k, s, p))
    dy = RNG.uniform(-2, 2, (2, 5) + out).astype(np.float32)
    dx = np.zeros(shape, np.float32)
    typed, generic = both(lib, monkeypatch, "aiednn_pooling_backward_create",
                          [C.c_int(lib.ALG[alg]), C.byref(md(shape, "f32")), C.byref(md(dy.shape, "f32")), None,
                           C.byref(pool_nd_params(k, s, p, p))],
                          {A["diff_src"]: dx, A["diff_dst"]: dy}, [A["diff_src"]])
    np.testing.assert_array_equal(typed[0].view(np.uint32), generic[0].view(np.uint32))


@pytest.mark.parametrize("shape,flags", [((3, 4, 64), SCALE | SHIFT), ((7, 384), SCALE | SHIFT), ((5, 8), SCALE | RMS),
                                         ((6, 33), 0), ((2, 3, 17), RMS)])
def test_layer_norm_backward_typed_equals_generic(lib, monkeypatch, shape, flags):
    L = shape[-1]
    rows = int(np.prod(shape[:-1]))
    x = RNG.uniform(-2, 2, shape).astype(np.float32)
    mean = x.reshape(rows, L).mean(-1).astype(np.float32) if not flags & RMS else np.zeros(rows, np.float32)
    var = (x.reshape(rows, L).var(-1) if not flags & RMS else (x.reshape(rows, L) ** 2).mean(-1)).astype(np.float32)
    dy = RNG.uniform(-1, 1, shape).astype(np.float32)
    bufs = {A["src"]: x, A["mean"]: mean, A["variance"]: var, A["diff_dst"]: dy, A["diff_src"]: np.zeros(shape, np.float32)}
    outs = [A["diff_src"]]
    if flags & SCALE:
        bufs[A["scale"]] = RNG.uniform(0.5, 1.5, L).astype(np.float32)
        bufs[A["diff_scale"]] = np.zeros(L, np.float32)
        outs.append(A["diff_scale"])
    if flags & SHIFT:
        bufs[A["diff_shift"]] = np.zeros(L, np.float32)
        outs.append(A["diff_shift"])
    typed, generic = both(lib, monkeypatch, "aiednn_layer_normalization_backward_create",
                          [C.byref(md(shape, "f32")), C.byref(md(shape, "f32")), C.byref(md(shape, "f32")), f(1e-5),
                           C.c_uint(flags)], bufs, outs)
    for t, g in zip(typed, generic):
        np.testing.assert_array_equal(t.view(np.uint32), g.view(np.uint32))
