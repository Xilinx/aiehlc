###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Training primitives (backward passes, normalization training forward) against torch autograd.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_train.py
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
from aiegraph.dnn_ctypes import ARG, AiednnLib, conv_nd_params, f, md, pool_nd_params  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(13)
A = dict(ARG, diff_src=20, diff_weights=21, diff_bias=22, diff_dst=23, diff_scale=24, diff_shift=25)
SCALE, SHIFT, RMS = 1, 2, 8


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_train"))


def r(*shape, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, shape).astype(np.float32)


def T(x, grad=False):
    return torch.from_numpy(np.asarray(x, np.float64)).requires_grad_(grad)


def m(a):
    return C.byref(md(a.shape, "f32"))


def run(lib, fn, args, bufs, outs):
    prim, st = lib.create(fn, *args)
    assert st == 0, f"{fn}: {lib.status(st)}"
    impl = lib.impl(prim)   # ref:, or an AIE path: gemm+ composites (conv / ip backward), fvec: (eltwise backward)
    assert impl.startswith("ref:") or "aie:" in impl, impl
    st = lib.execute(prim, bufs)
    lib.destroy(prim)
    assert st == 0, f"{fn} execute: {lib.status(st)}"
    return [bufs[o] for o in outs]


def close(got, want, tol=1e-4):
    np.testing.assert_allclose(got, want.detach().numpy(), rtol=tol, atol=tol)


def torch_pad_conv(x, w, s, pl, pr, d, g, deconv):
    nd = len(s)
    dil = tuple(v + 1 for v in d)
    if not deconv:
        pad = []
        for i in reversed(range(nd)):
            pad += [pl[i], pr[i]]
        return (F.conv1d, F.conv2d, F.conv3d)[nd - 1](F.pad(x, pad), w, stride=s, dilation=dil, groups=g)
    O, Cg = w.shape[0], w.shape[1]
    wt = w.reshape(g, O // g, Cg, *w.shape[2:]).transpose(1, 2).reshape(g * Cg, O // g, *w.shape[2:])
    y = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[nd - 1](x, wt, stride=s, dilation=dil, groups=g)
    return y[(slice(None), slice(None)) + tuple(slice(pl[i], y.shape[2 + i] - pr[i]) for i in range(nd))]


CONV = [((11,), (3,), (2,), (1,), (0,), (1,), 2), ((7, 6), (3, 3), (1, 2), (1, 0), (1, 1), (0, 1), 1),
        ((4, 5, 6), (2, 3, 3), (1, 2, 1), (0, 1, 1), (1, 0, 1), (0, 0, 1), 2)]


@pytest.mark.parametrize("case", CONV)
@pytest.mark.parametrize("deconv", [0, 1])
def test_conv_backward(lib, case, deconv):
    sp, k, s, pl, pr, d, g = case
    N, Cc, O = 2, 4, 6
    x, w = T(r(N, Cc, *sp), True), T(r(O, Cc // g, *k), True)
    y = torch_pad_conv(x, w, s, pl, pr, d, g, deconv)
    dy = r(*y.shape)
    y.backward(T(dy))
    cp = conv_nd_params(s, pl, pr, d, g)
    name = "deconvolution" if deconv else "convolution"
    dx = np.zeros(x.shape, np.float32)
    (got,) = run(lib, f"aiednn_{name}_backward_data_create", [m(dx), m(w.detach().numpy()), m(dy), C.byref(cp)],
                 {A["diff_src"]: dx, A["weights"]: w.detach().numpy().astype(np.float32), A["diff_dst"]: dy}, [A["diff_src"]])
    close(got, x.grad)
    dw, db = np.zeros(w.shape, np.float32), np.zeros((O,), np.float32)
    gw, gb = run(lib, f"aiednn_{name}_backward_weights_create", [m(x.detach().numpy()), m(dw), m(db), m(dy), C.byref(cp)],
                 {A["src"]: x.detach().numpy().astype(np.float32), A["diff_weights"]: dw, A["diff_bias"]: db, A["diff_dst"]: dy},
                 [A["diff_weights"], A["diff_bias"]])
    close(gw, w.grad, 1e-3)
    close(gb, T(dy).sum(dim=[0] + list(range(2, dy.ndim))))


def test_inner_product_backward(lib):
    x, w = T(r(3, 4, 2, 2), True), T(r(5, 4, 2, 2), True)
    y = x.reshape(3, -1) @ w.reshape(5, -1).T
    dy = r(3, 5)
    y.backward(T(dy))
    dx = np.zeros(x.shape, np.float32)
    (got,) = run(lib, "aiednn_inner_product_backward_data_create", [m(dx), m(w.detach().numpy()), m(dy)],
                 {A["diff_src"]: dx, A["weights"]: w.detach().numpy().astype(np.float32), A["diff_dst"]: dy}, [A["diff_src"]])
    close(got, x.grad)
    dw, db = np.zeros(w.shape, np.float32), np.zeros((5,), np.float32)
    gw, gb = run(lib, "aiednn_inner_product_backward_weights_create", [m(x.detach().numpy()), m(dw), m(db), m(dy)],
                 {A["src"]: x.detach().numpy().astype(np.float32), A["diff_weights"]: dw, A["diff_bias"]: db,
                  A["diff_dst"]: dy}, [A["diff_weights"], A["diff_bias"]])
    close(gw, w.grad)
    close(gb, T(dy).sum(0))


ELT = {
    "eltwise_relu": (0.1, 0.0, lambda x: F.leaky_relu(x, 0.1)),
    "eltwise_elu": (0.7, 0.0, lambda x: F.elu(x, 0.7)),
    "eltwise_gelu_erf": (0.0, 0.0, F.gelu),
    "eltwise_gelu_tanh": (0.0, 0.0, lambda x: F.gelu(x, approximate="tanh")),
    "eltwise_logistic": (0.0, 0.0, torch.sigmoid),
    "eltwise_tanh": (0.0, 0.0, torch.tanh),
    "eltwise_swish": (1.0, 0.0, F.silu),
    "eltwise_soft_relu": (1.0, 0.0, F.softplus),
    "eltwise_mish": (0.0, 0.0, F.mish),
    "eltwise_hardswish": (1 / 6, 0.5, F.hardswish),
    "eltwise_hardsigmoid": (1 / 6, 0.5, F.hardsigmoid),
    "eltwise_clip": (-0.5, 0.8, lambda x: torch.clamp(x, -0.5, 0.8)),
    "eltwise_linear": (1.5, 0.3, lambda x: 1.5 * x + 0.3),
    "eltwise_abs": (0.0, 0.0, torch.abs),
    "eltwise_exp": (0.0, 0.0, torch.exp),
    "eltwise_square": (0.0, 0.0, torch.square),
    "eltwise_neg": (0.0, 0.0, torch.neg),
    "eltwise_erf": (0.0, 0.0, torch.erf),
    "eltwise_sin": (0.0, 0.0, torch.sin),
    "eltwise_cos": (0.0, 0.0, torch.cos),
    "eltwise_floor": (0.0, 0.0, torch.floor),
    "eltwise_tan": (0.0, 0.0, torch.tan),
    "eltwise_identity": (0.0, 0.0, lambda x: x * 1.0),
    "eltwise_softsign": (0.0, 0.0, F.softsign),
    "eltwise_log_sigmoid": (0.0, 0.0, F.logsigmoid),
    "eltwise_selu": (0.0, 0.0, F.selu),
}
POS = {"eltwise_log": torch.log, "eltwise_sqrt": torch.sqrt, "eltwise_reciprocal": torch.reciprocal,
       "eltwise_rsqrt": torch.rsqrt, "eltwise_pow": None}


@pytest.mark.parametrize("alg", list(ELT) + list(POS))
def test_eltwise_backward(lib, alg):
    if alg in POS:
        x = r(4, 37, lo=0.2, hi=3.0)
        a, b, fn = (1.3, 2.5, lambda t: 1.3 * t ** 2.5) if alg == "eltwise_pow" else (0.0, 0.0, POS[alg])
    else:
        x = r(4, 37)
        a, b, fn = ELT[alg]
    xt = T(x, True)
    dy = r(*x.shape)
    fn(xt).backward(T(dy))
    dx = np.zeros(x.shape, np.float32)
    (got,) = run(lib, "aiednn_eltwise_backward_create", [C.c_int(lib.ALG[alg]), f(a), f(b), m(dx), m(dy), m(x)],
                 {A["diff_src"]: dx, A["diff_dst"]: dy, A["src"]: x}, [A["diff_src"]])
    close(got, xt.grad, 2e-4)


@pytest.mark.parametrize("alg", ["softmax_accurate", "softmax_log"])
def test_softmax_backward(lib, alg):
    x = T(r(3, 7, 5), True)
    y = (torch.softmax if alg == "softmax_accurate" else torch.log_softmax)(x, dim=1)
    dy = r(*y.shape)
    y.backward(T(dy))
    yv = y.detach().numpy().astype(np.float32)
    dx = np.zeros(yv.shape, np.float32)
    (got,) = run(lib, "aiednn_softmax_backward_create", [C.c_int(lib.ALG[alg]), C.c_int(1), m(dx), m(dy), m(yv)],
                 {A["diff_src"]: dx, A["diff_dst"]: dy, A["dst"]: yv}, [A["diff_src"]])
    close(got, x.grad)


@pytest.mark.parametrize("alg", ["pooling_max", "pooling_avg_include_padding", "pooling_avg_exclude_padding"])
@pytest.mark.parametrize("sp,k,s,p", [((9, 10), (3, 2), (2, 2), (1, 0)), ((5, 6, 6), (2, 3, 3), (2, 2, 1), (0, 1, 1)),
                                      ((13,), (3,), (2,), (1,))])
def test_pooling_backward(lib, alg, sp, k, s, p):
    x = T(r(2, 3, *sp), True)
    nd = len(sp)
    if alg == "pooling_max":
        y = (F.max_pool1d, F.max_pool2d, F.max_pool3d)[nd - 1](x, k, s, p)
    else:
        y = (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)[nd - 1](x, k, s, p,
                                                             count_include_pad=alg == "pooling_avg_include_padding")
    dy = r(*y.shape)
    y.backward(T(dy))
    xv = x.detach().numpy().astype(np.float32)
    dx = np.zeros(xv.shape, np.float32)
    (got,) = run(lib, "aiednn_pooling_backward_create",
                 [C.c_int(lib.ALG[alg]), m(dx), m(dy), m(xv), C.byref(pool_nd_params(k, s, p, p))],
                 {A["diff_src"]: dx, A["diff_dst"]: dy, A["src"]: xv}, [A["diff_src"]])
    close(got, x.grad)


NORMS = [("batch", (3, 4, 5, 2), 0, SCALE | SHIFT), ("batch", (4, 3), 0, 0), ("layer", (3, 4, 6), 0, SCALE | SHIFT),
         ("layer", (5, 8), 0, SCALE | RMS), ("group", (2, 6, 3, 2), 3, SCALE | SHIFT), ("group", (2, 4, 5), 2, 0)]


def torch_norm(kind, x, g, b, groups, flags, eps):
    C_ = x.shape[-1] if kind == "layer" else x.shape[1]
    if kind == "batch":
        dims = [0] + list(range(2, x.ndim))
        mean, var = x.mean(dims, keepdim=True), x.var(dims, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
        sh = (1, C_) + (1,) * (x.ndim - 2)
    elif kind == "layer":
        if flags & RMS:
            mean, var = torch.zeros_like(x[..., :1]), (x * x).mean(-1, keepdim=True)
        else:
            mean, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + eps)
        sh = (C_,)
    else:
        xg = x.reshape(x.shape[0], groups, -1)
        mean, var = xg.mean(-1, keepdim=True), xg.var(-1, unbiased=False, keepdim=True)
        y = ((xg - mean) / torch.sqrt(var + eps)).reshape(x.shape)
        sh = (1, C_) + (1,) * (x.ndim - 2)
    if flags & SCALE:
        y = y * g.reshape(sh)
    if flags & SHIFT and not flags & RMS:
        y = y + b.reshape(sh)
    return y, mean.reshape(-1), var.reshape(-1)


@pytest.mark.parametrize("kind,shape,groups,flags", NORMS)
def test_normalization_training(lib, kind, shape, groups, flags):
    eps = 1e-5
    x = T(r(*shape), True)
    C_ = shape[-1] if kind == "layer" else shape[1]
    g, b = T(r(C_, lo=0.5, hi=1.5), True), T(r(C_), True)
    y, mean, var = torch_norm(kind, x, g, b, groups, flags, eps)
    dy = r(*shape)
    y.backward(T(dy))
    xv = x.detach().numpy().astype(np.float32)
    gv, bv = g.detach().numpy().astype(np.float32), b.detach().numpy().astype(np.float32)
    nst = mean.numel()
    yv, mv, vv = np.zeros(shape, np.float32), np.zeros(nst, np.float32), np.zeros(nst, np.float32)
    pre = f"aiednn_{kind}_normalization"
    gargs = [C.c_int64(groups)] if kind == "group" else []
    bufs = {A["src"]: xv, A["dst"]: yv, A["mean"]: mv, A["variance"]: vv}
    if flags & SCALE:
        bufs[A["scale"]] = gv
    if flags & SHIFT:
        bufs[A["shift"]] = bv
    gy, gm, gvv = run(lib, pre + "_forward_training_create", gargs + [m(xv), m(yv), f(eps), C.c_uint(flags)], bufs,
                      [A["dst"], A["mean"], A["variance"]])
    close(gy, y)
    close(gm, mean)
    close(gvv, var)
    dx, dsc, dsh = np.zeros(shape, np.float32), np.zeros(C_, np.float32), np.zeros(C_, np.float32)
    bufs = {A["src"]: xv, A["mean"]: mv, A["variance"]: vv, A["diff_dst"]: dy, A["diff_src"]: dx}
    if flags & SCALE:
        bufs[A["scale"]], bufs[A["diff_scale"]] = gv, dsc
    if flags & SHIFT:
        bufs[A["diff_shift"]] = dsh
    run(lib, pre + "_backward_create", gargs + [m(dx), m(dy), m(xv), f(eps), C.c_uint(flags)], bufs, [A["diff_src"]])
    close(dx, x.grad, 1e-3)
    if flags & SCALE:
        close(dsc, g.grad, 1e-3)
    if flags & SHIFT and not flags & RMS:
        close(dsh, b.grad, 1e-3)


def test_prelu_lrn_resampling_shuffle_backward(lib):
    x, w = T(r(2, 3, 4, 5), True), T(r(1, 3, 1, 1, lo=0.1, hi=0.5), True)
    y = torch.where(x > 0, x, w * x)
    dy = r(*y.shape)
    y.backward(T(dy))
    dx, dw = np.zeros(x.shape, np.float32), np.zeros(w.shape, np.float32)
    gx, gw = run(lib, "aiednn_prelu_backward_create",
                 [m(dx), m(dw), m(dy), m(x.detach().numpy()), m(w.detach().numpy())],
                 {A["diff_src"]: dx, A["diff_weights"]: dw, A["diff_dst"]: dy,
                  A["src"]: x.detach().numpy().astype(np.float32), A["weights"]: w.detach().numpy().astype(np.float32)},
                 [A["diff_src"], A["diff_weights"]])
    close(gx, x.grad)
    close(gw, w.grad)
    x = T(r(2, 6, 3, 4), True)
    y = F.local_response_norm(x, 5, alpha=1e-1, beta=0.75, k=2.0)
    dy = r(*y.shape)
    y.backward(T(dy))
    xv, dx = x.detach().numpy().astype(np.float32), np.zeros(x.shape, np.float32)
    (got,) = run(lib, "aiednn_lrn_backward_create",
                 [C.c_int(lib.ALG["lrn_across_channels"]), C.c_int64(5), f(1e-1), f(0.75), f(2.0), m(dx), m(dy), m(xv)],
                 {A["diff_src"]: dx, A["diff_dst"]: dy, A["src"]: xv}, [A["diff_src"]])
    close(got, x.grad)
    for alg, mode in (("resampling_nearest", "nearest-exact"), ("resampling_linear", "bilinear")):
        x = T(r(2, 3, 5, 7), True)
        y = F.interpolate(x, size=(8, 4), mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))
        dy = r(*y.shape)
        y.backward(T(dy))
        dx = np.zeros(x.shape, np.float32)
        (got,) = run(lib, "aiednn_resampling_backward_create", [C.c_int(lib.ALG[alg]), m(dx), m(dy)],
                     {A["diff_src"]: dx, A["diff_dst"]: dy}, [A["diff_src"]])
        close(got, x.grad)
    x = T(r(2, 6, 3), True)
    y = x.reshape(2, 2, 3, 3).transpose(1, 2).reshape(2, 6, 3)   # channel shuffle, group_size 2
    dy = r(*y.shape)
    y.backward(T(dy))
    dx = np.zeros(x.shape, np.float32)
    (got,) = run(lib, "aiednn_shuffle_backward_create", [C.c_int(1), C.c_int64(2), m(dx), m(dy)],
                 {A["diff_src"]: dx, A["diff_dst"]: dy}, [A["diff_src"]])
    close(got, x.grad)
