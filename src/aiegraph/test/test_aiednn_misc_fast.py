###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Typed f32 dropout and grid sample paths are bit-identical to a numpy replica of ref:'s arithmetic."""
from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(71)
A = dict(ARG, mask=27, grid=28)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_misc_fast"))


def run(lib, fn, args, bufs, outs):
    prim, st = lib.create(fn, *args)
    assert st == 0, f"{fn}: {lib.status(st)}"
    assert lib.execute(prim, bufs) == 0
    lib.destroy(prim)
    return [bufs[o] for o in outs]


def mix64(z):
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def test_dropout_f32_bit_identical(lib):
    n, p, seed = 100_003, 0.3, 0xDEADBEEF12345
    x = RNG.uniform(-4, 4, n).astype(np.float32)
    y, mask = run(lib, "aiednn_dropout_forward_create", [C.byref(md((n,), "f32")), C.byref(md((n,), "f32")),
                                                          C.c_float(p), C.c_uint64(seed)],
                  {A["src"]: x, A["dst"]: np.zeros(n, np.float32), A["mask"]: np.zeros(n, np.uint8)}, [A["dst"], A["mask"]])
    keep = 1.0 - float(np.float32(p))
    s = np.uint64(((seed >> 32) << 32) ^ (seed & 0xFFFFFFFF))
    with np.errstate(over="ignore"):
        h = mix64(s * np.uint64(0x2545F4914F6CDD1D) + np.arange(n, dtype=np.uint64))
    want_mask = (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0) < keep
    want = np.where(want_mask, (x.astype(np.float64) / keep).astype(np.float32), np.float32(0))
    np.testing.assert_array_equal(mask.astype(bool), want_mask)
    np.testing.assert_array_equal(y.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("align,nearest", [(0, 0), (1, 0), (0, 1)])
def test_grid_sample_f32_bit_identical(lib, align, nearest):
    N, Cc, H, W, Ho, Wo = 2, 5, 13, 17, 9, 11
    x = RNG.uniform(-2, 2, (N, Cc, H, W)).astype(np.float32)
    g = RNG.uniform(-1.2, 1.2, (N, Ho, Wo, 2)).astype(np.float32)
    flags = (1 if nearest else 0) | (2 if align else 0)
    (y,) = run(lib, "aiednn_grid_sample_forward_create",
               [C.byref(md(x.shape, "f32")), C.byref(md(g.shape, "f32")), C.byref(md((N, Cc, Ho, Wo), "f32")), C.c_uint(flags)],
               {A["src"]: x, A["grid"]: g, A["dst"]: np.zeros((N, Cc, Ho, Wo), np.float32)}, [A["dst"]])
    def unnorm(v, size):
        return (v + 1) / 2 * (size - 1) if align else ((v + 1) * size - 1) / 2
    want = np.zeros((N, Cc, Ho, Wo), np.float32)
    for n in range(N):
        for h in range(Ho):
            for w in range(Wo):
                gx, gy = unnorm(float(g[n, h, w, 0]), W), unnorm(float(g[n, h, w, 1]), H)
                x0, y0 = np.floor(gx), np.floor(gy)
                wx, wy, ix, iy = gx - x0, gy - y0, int(x0), int(y0)
                def at(c, yy, xx):
                    return 0.0 if yy < 0 or yy >= H or xx < 0 or xx >= W else float(x[n, c, yy, xx])
                for c in range(Cc):
                    if nearest:
                        v = at(c, int(np.rint(gy)), int(np.rint(gx)))
                    else:
                        v = (at(c, iy, ix) * (1 - wx) * (1 - wy) + at(c, iy, ix + 1) * wx * (1 - wy) +
                             at(c, iy + 1, ix) * (1 - wx) * wy + at(c, iy + 1, ix + 1) * wx * wy)
                    want[n, c, h, w] = np.float32(v)
    np.testing.assert_array_equal(y.view(np.uint32), want.view(np.uint32))
