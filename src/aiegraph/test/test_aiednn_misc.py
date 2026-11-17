###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Dropout, the spatial transformer (affine grid, grid sample) and CTC loss against PyTorch.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_misc.py
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
RNG = np.random.default_rng(31)
A = dict(ARG, diff_src=20, diff_weights=21, diff_dst=23, mask=27, grid=28, labels=29, label_lengths=30,
         input_lengths=31)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_misc"))


def m(a, dt="f32"):
    return C.byref(md(a.shape, dt))


def run(lib, fn, args, bufs, outs):
    prim, st = lib.create(fn, *args)
    assert st == 0, f"{fn}: {lib.status(st)}"
    st = lib.execute(prim, bufs)
    lib.destroy(prim)
    assert st == 0, f"{fn} execute: {lib.status(st)}"
    return [bufs[o] for o in outs]


def test_dropout(lib):
    x = RNG.uniform(-2, 2, (64, 257)).astype(np.float32)
    p = 0.3
    y, mask = np.zeros_like(x), np.zeros(x.shape, np.uint8)
    y, mask = run(lib, "aiednn_dropout_forward_create", [m(x), m(y), f(p), C.c_uint64(1234)],
                  {A["src"]: x, A["dst"]: y, A["mask"]: mask}, [A["dst"], A["mask"]])
    assert abs(mask.mean() - (1 - p)) < 0.01                         # keep rate
    np.testing.assert_allclose(y, np.where(mask, x / (1 - p), 0), rtol=1e-6)
    y2, mask2 = run(lib, "aiednn_dropout_forward_create", [m(x), m(y), f(p), C.c_uint64(1234)],
                    {A["src"]: x, A["dst"]: np.zeros_like(x), A["mask"]: np.zeros(x.shape, np.uint8)},
                    [A["dst"], A["mask"]])
    np.testing.assert_array_equal(mask, mask2)                       # reproducible for a seed
    dy = RNG.uniform(-1, 1, x.shape).astype(np.float32)
    (dx,) = run(lib, "aiednn_dropout_backward_create", [m(x), m(dy), f(p)],
                {A["diff_src"]: np.zeros_like(x), A["diff_dst"]: dy, A["mask"]: mask}, [A["diff_src"]])
    np.testing.assert_allclose(dx, np.where(mask, dy / (1 - p), 0), rtol=1e-6)


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("nearest", [False, True])
def test_spatial_transformer(lib, align, nearest):
    N, Cc, H, W, Ho, Wo = 2, 3, 7, 9, 5, 6
    theta = torch.tensor(RNG.uniform(-1, 1, (N, 2, 3)), dtype=torch.float64)
    grid = F.affine_grid(theta, (N, Cc, Ho, Wo), align_corners=align)
    flags = (1 if nearest else 0) | (2 if align else 0)
    th = theta.float().numpy()
    (g,) = run(lib, "aiednn_affine_grid_create", [m(th), C.byref(md((N, Ho, Wo, 2), "f32")), C.c_uint(flags)],
               {A["src"]: th, A["dst"]: np.zeros((N, Ho, Wo, 2), np.float32)}, [A["dst"]])
    np.testing.assert_allclose(g, grid.numpy(), rtol=1e-5, atol=1e-6)
    x = torch.tensor(RNG.uniform(-1, 1, (N, Cc, H, W)), dtype=torch.float64, requires_grad=True)
    gr = torch.tensor(g, dtype=torch.float64, requires_grad=True)
    y = F.grid_sample(x, gr, mode="nearest" if nearest else "bilinear", padding_mode="zeros", align_corners=align)
    xv = x.detach().float().numpy()
    (yv,) = run(lib, "aiednn_grid_sample_forward_create", [m(xv), m(g), C.byref(md(tuple(y.shape), "f32")), C.c_uint(flags)],
                {A["src"]: xv, A["grid"]: g, A["dst"]: np.zeros(tuple(y.shape), np.float32)}, [A["dst"]])
    np.testing.assert_allclose(yv, y.detach().numpy(), rtol=1e-5, atol=1e-5)
    if nearest:
        return
    dy = RNG.uniform(-1, 1, tuple(y.shape)).astype(np.float32)
    y.backward(torch.tensor(dy, dtype=torch.float64))
    dx, dg = run(lib, "aiednn_grid_sample_backward_create",
                 [m(xv), m(g), m(dy), m(xv), m(g), C.c_uint(flags)],
                 {A["diff_src"]: np.zeros_like(xv), A["diff_weights"]: np.zeros_like(g), A["diff_dst"]: dy,
                  A["src"]: xv, A["grid"]: g}, [A["diff_src"], A["diff_weights"]])
    np.testing.assert_allclose(dx, x.grad.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dg, gr.grad.numpy(), rtol=1e-4, atol=1e-4)


def test_ctc_loss(lib):
    T_, N, Cc, Lmax = 12, 3, 6, 4
    logits = torch.tensor(RNG.normal(0, 1, (T_, N, Cc)), dtype=torch.float64)
    lp = torch.log_softmax(logits, 2).float().double().requires_grad_(True)
    labels = np.array([[1, 2, 2, 3], [4, 4, 0, 0], [5, 1, 3, 0]], np.int32)
    lab_len = np.array([4, 2, 3], np.int32)
    in_len = np.array([12, 9, 10], np.int32)
    loss = F.ctc_loss(lp, torch.tensor(labels, dtype=torch.long), torch.tensor(in_len, dtype=torch.long),
                      torch.tensor(lab_len, dtype=torch.long), blank=0, reduction="none")
    loss.sum().backward()
    lpv = lp.detach().float().numpy()
    out, dx = run(lib, "aiednn_ctc_loss_create", [m(lpv), C.c_int64(Lmax), C.byref(md((N,), "f32")), m(lpv)],
                  {A["src"]: lpv, A["dst"]: np.zeros(N, np.float32), A["labels"]: labels, A["label_lengths"]: lab_len,
                   A["input_lengths"]: in_len, A["diff_src"]: np.zeros_like(lpv)}, [A["dst"], A["diff_src"]])
    np.testing.assert_allclose(out, loss.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(dx, lp.grad.numpy(), rtol=1e-4, atol=1e-5)
