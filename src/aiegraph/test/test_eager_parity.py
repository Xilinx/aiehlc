###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Operator-support parity between aiednn's two build modes, in workstation emulation (g++ only).

FIXED: the fixed-megakernel library (AiednnLib.build: gemm_uk.cc's entries, x86 references).
EAGER: the premade eager template library (dnn_eager.library_plan(), built with AIEDNN_EAGER).

Per problem: the implementation each library picks, whether it reached an AIE entry (entry call counts,
or an aie: / eager: name), and its result against ref: under the family's contract. The scoreboard goes
to stdout and worklocal/eager_parity.md. test_parity[*] asserts "fixed on the AIE => eager on the AIE";
known gaps are xfail(strict=False), so they turn XPASS as eager closes them. test_contract[*] is never
xfailed: an AIE result outside its family's contract is a correctness bug.

    PYTHONPATH=src python -m pytest -q -s src/aiegraph/test/test_eager_parity.py
"""

from __future__ import annotations

import ctypes as C
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import (ARG, ARG_WEI_SCALES, ARG_WEI_ZPS, DT, AiednnLib, _alg_table, attr,  # noqa: E402
                                 conv_nd_params, conv_params, md, pack4, pool_params)

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
REPO = Path(__file__).resolve().parents[3]
ALG = _alg_table()                      # aiednn_alg_kind_t values (aiednn.h); the same in both libraries
S, S1, S2, W, B, D = ARG["src"], ARG["src_1"], ARG["src_2"], ARG["weights"], ARG["bias"], ARG["dst"]
MEAN, VAR, SC, SH = ARG["mean"], ARG["variance"], ARG["scale"], ARG["shift"]
UP, DOWN, GROUP_SIZES, SCALES, ZPS, IDX, PTR = 46, 47, 26, 50, 51, 43, 44
RNN = dict(src_iter=12, src_iter_c=13, weights_iter=14, dst_iter=15, dst_iter_c=16)
LA = dict(gate=52, beta=53, state=54, state_out=55)


# ── problem description ───────────────────────────────────────────────────

@dataclass
class P:
    """fn(*args) creates; bufs bound at execute; out the checked arg; tol the family contract:
    ("exact",) | ("codes", n) | ("close", rtol, atol) | ("scaled", t) (max err <= t * max(1, max|ref|)) |
    ("relmax", t) (max err <= t * max|ref|)"""
    fn: str
    args: list
    bufs: dict
    tol: tuple = ("exact",)
    out: int = D
    keep: list = field(default_factory=list)
    want: object = None     # numpy reference when no ref: implementation exists (grouped matmul)


RNG = np.random.default_rng(1234)
FV = ("close", 1e-5, 1e-5)        # fvec:'s contract (test_aiednn_fvec.py)
FG = ("scaled", 1e-4)             # float GEMM families: bf16x3 / bf16blk / composites over them


def i8(s, lo=-128, hi=127):
    return RNG.integers(lo, hi + 1, s).astype(np.int8)


def u8(s):
    return RNG.integers(0, 256, s).astype(np.uint8)


def f32(s, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, s).astype(np.float32)


def m(shape, dt="f32"):
    return C.byref(md(shape, dt))


def z(shape, dt):
    return np.zeros(shape, DT[dt][1])


def dt_of(a):
    return {np.float32: "f32", np.int8: "s8", np.uint8: "u8", np.int32: "s32", np.float16: "f16"}[a.dtype.type]


def cout(n, k, s, pl, pr, d=0):
    return (n + pl + pr - (k - 1) * (d + 1) - 1) // s + 1


# ── GEMM family: blocked / im2col / ip / col2im / zp+ / gemm+ composites ──

def matmul(sa, sw, dt, ddt, tol, a=None, bias=None):
    x, w = (i8(sa, -20, 20), i8(sw, -20, 20)) if dt == "s8" else (f32(sa), f32(sw))
    M, N = sa[-2], sw[-1]
    bufs = {S: x, W: w, D: z(sa[:-2] + (M, N), ddt)}
    bmd = None
    if bias is not None:
        bufs[B], bmd = bias, m(bias.shape, dt_of(bias))
    args = [m(sa, dt), m(sw, dt), bmd, m(bufs[D].shape, ddt), C.c_int(0)]
    if a is None:
        return P("aiednn_matmul_forward_create", args, bufs, tol)
    return P("aiednn_matmul_forward_create_attr", args + [C.byref(a[0])], bufs, tol, keep=[a])


def bf16_matmul():
    to = lambda a: (np.ascontiguousarray(a).view(np.uint32) >> 16).astype(np.uint16)
    a, w = to(f32((37, 300))), to(f32((300, 70)))
    return P("aiednn_matmul_forward_create", [m(a.shape, "bf16"), m(w.shape, "bf16"), None, m((37, 70)), C.c_int(0)],
             {S: a, W: w, D: z((37, 70), "f32")}, FG)


def conv(xs, ws, dt, ddt, cp, tol, bias=None, a=None, deconv=False, nd=False):
    x = {"s8": lambda: i8(xs), "u8": lambda: u8(xs), "f32": lambda: f32(xs)}[dt]()
    w = f32(ws) if dt == "f32" else i8(ws)
    sp, k, n = xs[2:], ws[2:], len(xs) - 2
    if deconv:
        os_ = tuple((sp[i] - 1) * cp.strides[i] - cp.padding_l[i] - cp.padding_r[i] + (k[i] - 1) * (cp.dilates[i] + 1) + 1
                    for i in range(n))
    else:
        os_ = tuple(cout(sp[i], k[i], cp.strides[i], cp.padding_l[i], cp.padding_r[i], cp.dilates[i]) for i in range(n))
    oshape = (xs[0], ws[0]) + os_
    bufs = {S: x, W: w, D: z(oshape, ddt)}
    bmd = None
    if bias is not None:
        bufs[B], bmd = bias, m(bias.shape, dt_of(bias))
    kind = "deconvolution" if deconv else "convolution"
    args = [m(xs, dt), m(ws, "f32" if dt == "f32" else "s8"), bmd, m(oshape, ddt), C.byref(cp)]
    if nd:
        return P(f"aiednn_{kind}_forward_create_nd", args + [None], bufs, tol)   # attr NULL
    if a is None:
        return P(f"aiednn_{kind}_forward_create", args, bufs, tol)
    return P(f"aiednn_{kind}_forward_create_attr", args + [C.byref(a[0])], bufs, tol, keep=[a])


def inner_product(dt):
    x, w = (i8((3, 8, 4, 4)), i8((50, 8, 4, 4))) if dt == "s8" else (f32((3, 8, 4, 4)), f32((50, 8, 4, 4)))
    ddt = "s32" if dt == "s8" else "f32"
    b = RNG.integers(-100, 100, 50).astype(np.int32) if dt == "s8" else f32((50,))
    return P("aiednn_inner_product_forward_create", [m(x.shape, dt), m(w.shape, dt), m(b.shape, ddt), m((3, 50), ddt)],
             {S: x, W: w, B: b, D: z((3, 50), ddt)}, ("exact",) if dt == "s8" else FG)


def requant_attr(n, ddt="s8", post=(("eltwise", "relu", 0.0, 0.0),)):
    sc = (RNG.random(n) * 1e-3 + 1e-4).astype(np.float32)
    return attr(ALG, scales=sc, dst_scale=0.05, dst_zp=3 if ddt == "u8" else 0, post_ops=list(post))


def grouped_matmul():
    T_, K, N, E = 40, 24, 16, 4
    x, w = i8((T_, K)), i8((E, K, N))
    b = RNG.integers(-500, 500, (E, N)).astype(np.int32)
    sizes = np.array([7, 0, 20, 9], np.int32)
    want, off = np.full((T_, N), 7, np.int64), 0
    for e, n in enumerate(sizes):            # "grouped" is the only implementation: no ref: to compare with
        want[off:off + n] = x[off:off + n].astype(np.int64) @ w[e].astype(np.int64) + b[e]
        off += n
    return P("aiednn_grouped_matmul_create", [m(x.shape, "s8"), m(w.shape, "s8"), m(b.shape, "s32"),
                                              m((T_, N), "s32"), C.c_int(0), None],
             {S: x, W: w, B: b, D: np.full((T_, N), 7, np.int32), GROUP_SIZES: sizes}, want=want.astype(np.int32))


def sparse_matmul():
    M, K, N = 37, 300, 70
    a = np.where(RNG.uniform(size=(M, K)) < 0.3, RNG.integers(-128, 128, (M, K)), 0).astype(np.int8)
    rows, cols = np.nonzero(a)
    ptr = np.zeros(M + 1, np.int32)
    np.add.at(ptr, rows + 1, 1)
    vals, w = np.ascontiguousarray(a[rows, cols]), i8((K, N))
    return P("aiednn_sparse_matmul_create", [m(vals.shape, "s8"), C.c_int64(M), C.c_int64(K), m(w.shape, "s8"), None,
                                             m((M, N), "s32")],
             {S: vals, W: w, D: z((M, N), "s32"), IDX: cols.astype(np.int32), PTR: np.cumsum(ptr).astype(np.int32)})


# ── weight-only quantization / type conversion ────────────────────────────

def wq_s4():
    M, K, N, g = 9, 1024, 40, 128
    x, wv = i8((M, K)), RNG.integers(-8, 8, (K, N))
    s = RNG.uniform(0.005, 0.05, (K // g, N)).astype(np.float32)
    zp = RNG.integers(-3, 4, (K // g, N)).astype(np.int8)
    a = attr(ALG, wei_scales=((K // g, N), "f32"), wei_zps=((K // g, N), "s8"))
    return P("aiednn_matmul_forward_create_attr", [m((M, K), "s8"), m((K, N), "s4"), None, m((M, N)), C.c_int(0),
                                                   C.byref(a[0])],
             {S: x, W: pack4(wv), D: z((M, N), "f32"), ARG_WEI_SCALES: s, ARG_WEI_ZPS: zp}, FG, keep=[a])


def dequant(wdt):
    M, K, N = 3, 96, 10
    g = 32
    x = f32((M, K), -1, 1)
    if wdt == "s4":
        wbuf, sdt, sbuf = pack4(RNG.integers(-8, 8, (K, N))), "f32", RNG.uniform(0.01, 0.1, (K // g, N)).astype(np.float32)
    else:   # MXFP4: f4_e2m1 codes, one e8m0 scale per 32 along K
        wbuf, sdt, sbuf = pack4(RNG.integers(0, 16, (K, N))), "e8m0", RNG.integers(120, 132, (K // g, N)).astype(np.uint8)
    a = attr(ALG, wei_scales=((K // g, N), sdt))
    return P("aiednn_matmul_forward_create_attr", [m((M, K)), m((K, N), wdt), None, m((M, N)), C.c_int(0), C.byref(a[0])],
             {S: x, W: wbuf, D: z((M, N), "f32"), ARG_WEI_SCALES: sbuf}, FG, keep=[a])


def cvt_f16():
    M, K, N = 4, 24, 9
    x, w = f32((M, K), -1, 1).astype(np.float16), f32((K, N), -1, 1).astype(np.float16)
    a = attr(ALG, scales=RNG.uniform(0.5, 1.5, N).astype(np.float32))
    return P("aiednn_matmul_forward_create_attr", [m((M, K), "f16"), m((K, N), "f16"), None, m((M, N)), C.c_int(0),
                                                   C.byref(a[0])], {S: x, W: w, D: z((M, N), "f32")}, FG, keep=[a])


def dyn_quantize():
    shape = (2, 16, 32, 32)
    return P("aiednn_dynamic_quantize_create", [m(shape), m(shape, "s8"), m((1,)), m((1,), "s32"), C.c_int(-1)],
             {S: f32(shape, -3, 3), D: z(shape, "s8"), SCALES: np.array([0.031], np.float32),
              ZPS: np.array([3], np.int32)})


# ── pointwise: eltwise / binary / prelu / batch norm / reorder / softmax / norms ──

def eltwise(alg, a, b, sdt, ddt, shape=(1, 64, 28, 28), tol=("exact",), q=None):
    x = {"s8": lambda: i8(shape), "u8": lambda: u8(shape), "f32": lambda: f32(shape, -6, 6)}[sdt]()
    args = [C.c_int(ALG[alg]), C.c_float(a), C.c_float(b), m(shape, sdt), m(shape, ddt)]
    if q is None:
        return P("aiednn_eltwise_forward_create", args, {S: x, D: z(shape, ddt)}, tol)
    return P("aiednn_eltwise_forward_create_attr", args + [C.byref(q[0])], {S: x, D: z(shape, ddt)}, tol, keep=[q])


def qattr(sc, zp, dsc, dzp):
    """src_scales / src_zero_points of a quantized eltwise / binary"""
    a, keep = attr(ALG, dst_scale=dsc, dst_zp=dzp)
    a.src_scales[0], a.src_scales[1] = sc
    a.src_zero_points[0], a.src_zero_points[1] = zp
    return a, keep


def binary(alg, s0, d0, s1, d1, ddt, tol=("exact",), q=None):
    gen = {"s8": i8, "u8": u8, "f32": lambda s: f32(s, -4, 4)}
    x, y = gen[d0](s0), gen[d1](s1)
    if q is not None:
        return P("aiednn_binary_create_attr", [C.c_int(ALG[alg]), m(s0, d0), m(s1, d1), m(s0, ddt), C.byref(q[0])],
                 {S: x, S1: y, D: z(s0, ddt)}, tol, keep=[q])
    return P("aiednn_binary_create", [C.c_int(ALG[alg]), m(s0, d0), m(s1, d1), None, m(s0, ddt)],
             {S: x, S1: y, D: z(s0, ddt)}, tol)


def prelu(sdt, ddt, tol):
    shape = (1, 48, 20, 20)
    x = u8(shape) if sdt == "u8" else f32(shape, -4, 4)
    w = f32((1, 48, 1, 1), -0.5, 0.5)
    return P("aiednn_prelu_forward_create", [m(shape, sdt), m(w.shape), m(shape, ddt)], {S: x, W: w, D: z(shape, ddt)}, tol)


def batch_norm(sdt, ddt, tol, flags=3, q=None):
    Cc, shape = 64, (1, 64, 28, 28)
    x = {"s8": lambda: i8(shape), "f32": lambda: f32(shape, -4, 4)}[sdt]()
    bufs = {S: x, D: z(shape, ddt), MEAN: f32((Cc,), -1, 1), VAR: f32((Cc,), 0.5, 3), SC: f32((Cc,), 0.5, 1.5),
            SH: f32((Cc,), -1, 1)}
    if sdt == "s8":
        bufs[MEAN], bufs[VAR] = f32((Cc,), -20, 20), f32((Cc,), 50, 900)
    args = [m(shape, sdt), m(shape, ddt), C.c_float(1e-5), C.c_uint(flags)]
    if q is None:
        return P("aiednn_batch_normalization_forward_inference_create", args, bufs, tol)
    return P("aiednn_batch_normalization_forward_inference_create_attr", args + [C.byref(q[0])], bufs, tol, keep=[q])


def reorder(sdt, ddt, scale, zp, shape=(1, 64, 28, 28)):
    x = {"s8": lambda: i8(shape), "f32": lambda: f32(shape, -4, 4), "s32": lambda: RNG.integers(-2048, 2048, shape).astype(np.int32)}[sdt]()
    return P("aiednn_reorder_create", [m(shape, sdt), m(shape, ddt), None, C.c_float(scale), C.c_int32(zp)],
             {S: x, D: z(shape, ddt)})


def softmax(shape=(3, 4, 64)):
    return P("aiednn_softmax_forward_create", [C.c_int(ALG["softmax_accurate"]), C.c_int(len(shape) - 1), m(shape), m(shape)],
             {S: f32(shape, -8, 8), D: z(shape, "f32")}, FV)


def layer_norm(shape=(6, 256)):
    Cc = shape[-1]
    return P("aiednn_layer_normalization_forward_inference_create", [m(shape), m(shape), C.c_float(1e-5), C.c_uint(3)],
             {S: f32(shape, -3, 3), D: z(shape, "f32"), SC: f32((Cc,), 0.5, 1.5), SH: f32((Cc,), -1, 1)}, FV)


def rope():
    x = f32((2, 6, 30, 64))
    ang = f32((1, 1, 30, 64), -3, 3)
    return P("aiednn_rope_create", [m(x.shape), m(ang.shape), m(ang.shape), m(x.shape), C.c_int(0)],
             {S: x, ARG["rope_cos"]: np.cos(ang), ARG["rope_sin"]: np.sin(ang), D: z(x.shape, "f32")})


# ── spatial: pooling / reduction / resampling ─────────────────────────────

def pooling(alg, dt, hw, k, s, p, tol=("exact",)):
    xs = (2, 8, hw, hw)
    x = {"s8": lambda: i8(xs), "f32": lambda: f32(xs)}[dt]()
    o = cout(hw, k, s, p, p)
    return P("aiednn_pooling_forward_create", [C.c_int(ALG[alg]), m(xs, dt), m((2, 8, o, o), dt),
                                               C.byref(pool_params((k, k), (s, s), (p, p), (p, p)))],
             {S: x, D: z((2, 8, o, o), dt)}, tol)


def reduction(alg, xs, sdt, ddt, a=0.0, b=0.0, tol=("exact",)):
    x = {"s8": lambda: i8(xs), "u8": lambda: u8(xs), "f32": lambda: f32(xs)}[sdt]()
    os_ = xs[:2] + (1,) * (len(xs) - 2)
    return P("aiednn_reduction_create", [C.c_int(ALG[alg]), C.c_float(a), C.c_float(b), m(xs, sdt), m(os_, ddt)],
             {S: x, D: z(os_, ddt)}, tol)


def resampling():
    return P("aiednn_resampling_forward_create", [C.c_int(ALG["resampling_linear"]), m((2, 3, 5, 7)), m((2, 3, 11, 3))],
             {S: f32((2, 3, 5, 7)), D: z((2, 3, 11, 3), "f32")}, FV)


# ── attention / recurrent / fused ─────────────────────────────────────────

def sdpa(ddt, sq=7, skv=9):
    Bt, H, Dd = 2, 3, 16
    q, k, v = f32((Bt, H, sq, Dd)), f32((Bt, H, skv, Dd)), f32((Bt, H, skv, Dd))
    args = [m(q.shape), m(k.shape), m(v.shape), None, m((Bt, H, sq, Dd), ddt), C.c_float(1 / math.sqrt(Dd)), C.c_int(0)]
    bufs = {S: q, S1: k, S2: v, D: z((Bt, H, sq, Dd), ddt)}
    if ddt == "f32":
        return P("aiednn_sdpa_forward_create", args, bufs, FG)
    a = attr(ALG, dst_scale=0.01, dst_zp=3)
    return P("aiednn_sdpa_forward_create_attr", args + [C.byref(a[0])], bufs, ("codes", 4), keep=[a])


def sdpa_int8():
    """int8 Q / K / V ([B, S, H, D]) -> f32: gemm+sdpa8 with AIEDNN_ATTR_TOLERANCE_INT8_ATTN (flag 4); contract
    as test_aiednn_sdpa8.py: within 1 % of the largest output"""
    Bt, Sq, H, Dd = 1, 33, 4, 16
    sh = (Bt, Sq, H, Dd)
    q, k, v = i8(sh, -127), i8(sh, -127), i8(sh, -127)
    a = attr(ALG)
    a[0].src_scales[0], a[0].src_scales[1], a[0].src_scales_23[0] = 0.03, 0.03, 0.05
    a[0].flags = 4
    return P("aiednn_sdpa_forward_create_attr", [m(sh, "s8"), m(sh, "s8"), m(sh, "s8"), None, m(sh),
                                                 C.c_float(1 / math.sqrt(Dd)), C.c_int(2 | 4), C.byref(a[0])],
             {S: q, S1: k, S2: v, D: z(sh, "f32")}, ("relmax", 0.01), keep=[a])


def lstm():
    T, N, H, SLC, G = 5, 3, 6, 4, 4
    u = lambda *s: RNG.uniform(-0.6, 0.6, s).astype(np.float32)
    x, Wl, U, Bb, h0, c0 = u(T, N, SLC), u(1, 1, SLC, G, H), u(1, 1, H, G, H), u(1, 1, G, H), u(1, 1, N, H), u(1, 1, N, H)
    y, hT, cT = z((T, N, H), "f32"), z((1, 1, N, H), "f32"), z((1, 1, N, H), "f32")
    mm = lambda a: m(a.shape)
    return P("aiednn_rnn_forward_create", [C.c_int(ALG["rnn_lstm"]), C.c_int(0), C.c_float(0.0), mm(x), mm(h0), mm(c0),
                                           mm(Wl), mm(U), mm(Bb), mm(y), mm(hT), mm(cT)],
             {S: x, W: Wl, RNN["weights_iter"]: U, B: Bb, D: y, RNN["src_iter"]: h0, RNN["src_iter_c"]: c0,
              RNN["dst_iter"]: hT, RNN["dst_iter_c"]: cT}, FG)


def gated_mlp():
    Mm, Dm, Fd = 37, 64, 96
    x = f32((1, Mm, Dm), -1, 1)
    wg, wu, wd = (RNG.normal(0, 0.1, s).astype(np.float32) for s in ((Dm, Fd), (Dm, Fd), (Fd, Dm)))
    return P("aiednn_gated_mlp_create", [m(x.shape), m(wg.shape), m(wu.shape), m(wd.shape), m(x.shape),
                                         C.c_int(ALG["eltwise_swish"]), C.c_float(1.0), C.c_float(0.0), C.c_uint(0)],
             {S: x, W: wg, UP: wu, DOWN: wd, D: z(x.shape, "f32")}, FG)


def linear_attention():
    Bt, H, T, Dk, Dv = 1, 2, 70, 32, 24
    q = RNG.normal(size=(Bt, H, T, Dk)).astype(np.float32)
    k = RNG.normal(size=(Bt, H, T, Dk))
    k = (k / np.linalg.norm(k, axis=-1, keepdims=True)).astype(np.float32)
    v = RNG.normal(size=(Bt, H, T, Dv)).astype(np.float32)
    g, beta = f32((Bt, H, T), -0.3, 0.0), f32((Bt, H, T), 0.1, 0.9)
    return P("aiednn_linear_attention_create", [m(q.shape), m(k.shape), m(v.shape), m(g.shape), m(beta.shape), None,
                                                m(v.shape), None, C.c_float(0.17), C.c_uint(0)],
             {S: q, S1: k, S2: v, LA["gate"]: g, LA["beta"]: beta, D: z(v.shape, "f32")}, ("scaled", 2e-4))


# ── the problem list: (name, family, builder) ─────────────────────────────

def _problems():
    cp1 = conv_params((1, 1), (1, 1), (1, 1))
    dwp = lambda s, p, g: conv_params((s, s), (p, p), (p, p), (0, 0), g)
    b32 = lambda n: RNG.integers(-1000, 1000, n).astype(np.int32)
    q8 = lambda sc, zp: qattr((sc, 0.0), (zp, 0), 0.0459, 5)
    return [
        ("matmul_s8_s32", "blocked:", lambda: matmul((70, 300), (300, 150), "s8", "s32", ("exact",))),
        ("matmul_s8_s8_requant_relu", "blocked:", lambda: matmul((50, 250), (250, 64), "s8", "s8", ("exact",),
                                                                  a=requant_attr(64), bias=f32((1, 64), -1, 1))),
        ("matmul_f32", "bf16x3:", lambda: matmul((5, 33, 12), (5, 12, 7), "f32", "f32", FG)),
        ("matmul_bf16", "bf16blk:", bf16_matmul),
        ("conv3x3_s8_s32", "im2col", lambda: conv((1, 20, 10, 18), (64, 20, 3, 3), "s8", "s32", conv_params(), ("exact",))),
        ("conv3x3_s8_s8_requant", "im2col", lambda: conv((1, 20, 10, 18), (64, 20, 3, 3), "s8", "s8", conv_params(),
                                                          ("exact",), bias=f32((64,), -1, 1), a=requant_attr(64))),
        ("conv3x3_u8_s32_zp", "zp+", lambda: conv((1, 3, 9, 8), (8, 3, 3, 3), "u8", "s32", cp1, ("exact",), bias=b32(8))),
        ("conv3x3_f32", "gemm+conv", lambda: conv((1, 8, 12, 12), (16, 8, 3, 3), "f32", "f32", conv_params(), FG,
                                                   bias=f32((16,)))),
        ("conv3d_s8_s32", "gemm+conv", lambda: conv((2, 8, 6, 7, 5), (12, 8, 3, 3, 3), "s8", "s32",
                                                    conv_nd_params((1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
                                                    ("exact",), nd=True)),
        ("inner_product_s8", "ip", lambda: inner_product("s8")),
        ("inner_product_f32", "ip", lambda: inner_product("f32")),
        ("deconv_s8_s32", "col2im", lambda: conv((1, 4, 5, 6), (6, 4, 3, 3), "s8", "s32",
                                                 conv_params((2, 2), (1, 1), (1, 1)), ("exact",), bias=b32(6), deconv=True)),
        ("deconv_f32", "gemm+deconv", lambda: conv((1, 4, 5, 6), (6, 4, 3, 3), "f32", "f32",
                                                   conv_params((2, 2), (1, 1), (1, 1)), FG, bias=f32((6,)), deconv=True)),
        ("deconv3d_s8_s32", "gemm+deconv", lambda: conv((2, 4, 6, 7, 5), (6, 4, 3, 3, 3), "s8", "s32",
                                                        conv_nd_params((1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
                                                        ("exact",), deconv=True, nd=True)),
        ("depthwise_s8_s32", "dwblk:", lambda: conv((1, 8, 7, 7), (8, 1, 3, 3), "s8", "s32", dwp(1, 1, 8), ("exact",),
                                                     bias=b32(8))),
        ("depthwise_s8_f32_scales", "dwblk:", lambda: conv((2, 24, 13, 13), (24, 1, 3, 3), "s8", "f32", dwp(1, 1, 24),
                                                           ("exact",), bias=f32((24,), -1, 1),
                                                           a=requant_attr(24, post=(("eltwise", "clip", 0.0, 6.0),)))),
        ("depthwise_s8_s8_requant", "dwblk:", lambda: conv((1, 16, 14, 14), (16, 1, 3, 3), "s8", "s8", dwp(1, 1, 16),
                                                           ("exact",), bias=f32((16,), -1, 1),
                                                           a=requant_attr(16, post=(("eltwise", "clip", 0.0, 6.0),)))),
        ("grouped_matmul_s8", "grouped", grouped_matmul),
        ("sparse_matmul_s8", "densify", sparse_matmul),
        ("wq_s8_x_s4", "wq:", wq_s4),
        ("dequant_s4_f32", "dequant+", lambda: dequant("s4")),
        ("dequant_mxfp4", "dequant+", lambda: dequant("f4_e2m1")),
        ("cvt_f16_matmul", "cvt+", cvt_f16),
        ("dynamic_quantize_f32_s8", "dynq", dyn_quantize),
        ("eltwise_gelu_s8_s8", "lut:", lambda: eltwise("eltwise_gelu_erf", 0, 0, "s8", "s8")),
        ("eltwise_swish_u8_f32", "lut:", lambda: eltwise("eltwise_swish", 1.0, 0, "u8", "f32")),
        ("eltwise_q_hardswish_s8_s8", "lut:", lambda: eltwise("eltwise_hardswish", 1 / 6, 0.5, "s8", "s8",
                                                               q=q8(0.0391, 3))),
        ("eltwise_q_gelu_s8_f32", "lut:", lambda: eltwise("eltwise_gelu_erf", 0, 0, "s8", "f32", q=q8(0.0391, 3))),
        ("eltwise_gelu_tanh_f32", "fvec:", lambda: eltwise("eltwise_gelu_tanh", 0, 0, "f32", "f32", tol=FV)),
        ("eltwise_clip_f32_u8", "steps:", lambda: eltwise("eltwise_clip", 0.0, 6.0, "f32", "u8")),
        ("softmax_f32", "fvec:", softmax),
        ("layer_norm_f32", "fvec:", layer_norm),
        ("binary_add_s8_u8_s8", "eltblk:", lambda: binary("binary_add", (1, 64, 28, 28), "s8", (1, 64, 28, 28), "u8", "s8")),
        ("binary_mul_s8_s8_f32", "eltblk:", lambda: binary("binary_mul", (1, 32, 28, 28), "s8", (1, 32, 28, 28), "s8", "f32")),
        ("binary_mul_s8_perchannel", "lut:", lambda: binary("binary_mul", (1, 64, 28, 28), "s8", (1, 64, 1, 1), "f32", "s8")),
        ("binary_add_f32_bcast", "fvec:", lambda: binary("binary_add", (4, 8, 33), "f32", (1, 8, 1), "f32", "f32", FV)),
        ("binary_q_add_s8_s8", "fvec:", lambda: binary("binary_add", (2, 16, 9, 9), "s8", (2, 16, 9, 9), "s8", "s8",
                                                        q=qattr((0.0213, 0.0371), (3, -7), 0.0459, 5))),
        ("prelu_f32", "fvec:", lambda: prelu("f32", "f32", FV)),
        ("prelu_u8_f32", "lut:", lambda: prelu("u8", "f32", ("exact",))),
        ("batch_norm_f32", "fvec:", lambda: batch_norm("f32", "f32", FV)),
        ("batch_norm_relu_f32_s8", "steps:", lambda: batch_norm("f32", "s8", ("exact",), flags=3 | 4,
                                                                q=attr(ALG, dst_scale=0.031))),
        ("batch_norm_relu_s8_s8", "lut:", lambda: batch_norm("s8", "s8", ("exact",), flags=3 | 4)),
        ("reorder_f32_s8", "steps:", lambda: reorder("f32", "s8", 0.031, 3)),
        ("reorder_s32_s8", "steps:", lambda: reorder("s32", "s8", 1.0, 0)),
        ("rope_f32", "(none)", rope),
        ("pool_max3x3s2_s8", "poolblk:", lambda: pooling("pooling_max", "s8", 56, 3, 2, 1)),
        ("pool_avg3x3_s8", "poolblk:", lambda: pooling("pooling_avg_exclude_padding", "s8", 63, 3, 2, 1)),
        ("pool_avg3x3_f32", "fpool:", lambda: pooling("pooling_avg_include_padding", "f32", 16, 3, 2, 1,
                                                      ("close", 1e-5, 1e-6))),
        ("pool_max2x2_f32", "fpool:", lambda: pooling("pooling_max", "f32", 16, 2, 2, 0, ("close", 1e-5, 1e-6))),
        ("reduction_mean_s8_f32", "redblk:", lambda: reduction("reduction_mean", (1, 256, 7, 7), "s8", "f32")),
        ("reduction_max_u8", "redblk:", lambda: reduction("reduction_max", (2, 32, 56, 56), "u8", "u8")),
        ("reduction_sum_f32", "fvec:", lambda: reduction("reduction_sum", (2, 32, 14, 14), "f32", "f32", tol=FV)),
        ("resampling_linear_f32", "(none)", resampling),
        ("sdpa_f32", "gemm+sdpa", lambda: sdpa("f32")),
        ("sdpa_f32_to_s8", "sdpablk:", lambda: sdpa("s8", 64, 64)),
        ("sdpa_int8_qkv", "gemm+sdpa8", sdpa_int8),
        ("rnn_lstm_f32", "gemm+rnn", lstm),
        ("gated_mlp_f32", "gemm+gmlp", gated_mlp),
        ("linear_attention_f32", "gemm+linattn", linear_attention),
    ]


PROBLEMS = _problems()
NAMES = [p[0] for p in PROBLEMS]
#: fixed has an AIE implementation, eager does not (yet): xfail(strict=False) -> XPASS when eager closes it
KNOWN_GAPS = set("""
""".split())  # none left: every fixed AIE family has an eager template


# ── building the two libraries ─────────────────────────────────────────────

def build_eager(out: Path) -> AiednnLib:
    """dnn_eager.library_plan() -> write_sources -> kernel_lib -> impl list (counted) -> g++ with AIEDNN_EAGER,
    as test_aiednn_eager.py's fixture and dnn._dnn_build_eager do for x86"""
    from aiegraph import dnn_eager
    from aiegraph.dnn import EAGER_DEFINES, impl_list_source
    from aiegraph.kernel_lib import kernel_lib
    src = out / "src"
    dnn_eager.write_sources(dnn_eager.library_plan("parity"), src)
    picked = list(kernel_lib(src, quiet=True))
    assert picked, "kernel_lib found no eager instances"
    (out / "impl_list.cc").write_text(impl_list_source(picked, x86=True, count=True))
    so = out / "libaiednn_eager.so"
    cmd = ["g++", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-std=gnu++17",
           *[f"-D{d}" for d in EAGER_DEFINES], f"-I{REPO / 'include'}", f"-I{REPO / 'src' / 'aietensorop' / 'ops'}",
           f"-I{REPO / 'src' / 'aiednn'}", f"-I{src}", str(REPO / "src" / "aiednn" / "aiednn_rt.cc"),
           str(out / "impl_list.cc"), "-x", "c++", str(src / dnn_eager.REF_FILE), "-x", "none", "-o", str(so), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    return AiednnLib(so)


# ── running and judging ────────────────────────────────────────────────────

@dataclass
class Run:
    impl: str
    out: "np.ndarray | None" = None
    calls: int = 0
    err: str = ""

    @property
    def aie(self) -> bool:
        return (self.out is not None and not self.impl.startswith("ref:")
                and (self.calls > 0 or "aie:" in self.impl or "eager:" in self.impl))


def run(lib: AiednnLib, p: P, filt=None) -> Run:
    bufs = {k: v.copy() for k, v in p.bufs.items()}
    lib.set_filter(filt)
    try:
        prim, st = lib.create(p.fn, *p.args)
        if st != 0:
            return Run(f"(create: {lib.status(st)})", err=lib.status(st))
        impl = lib.impl(prim)
        lib.entry_calls(reset=True)
        st = lib.execute(prim, bufs)
        calls = sum(lib.entry_calls(reset=True).values())
        lib.destroy(prim)
    finally:
        lib.set_filter(None)
    if st != 0:
        return Run(impl, err=f"execute: {lib.status(st)}")
    return Run(impl, bufs[p.out], calls)


def within(tol, got, want) -> "tuple[bool, str]":
    g, w = got.astype(np.float64), want.astype(np.float64)
    d = np.abs(g - w)
    fin = np.isfinite(w)
    if not np.array_equal(np.isfinite(g), fin):
        return False, "non-finite mismatch"
    d = np.where(fin, d, 0.0)
    if tol[0] == "exact":
        bad = int(np.count_nonzero(got != want))
        return bad == 0, f"{bad} differ" if bad else "exact"
    if tol[0] == "codes":
        mx = float(d.max())
        return mx <= tol[1], f"max {mx:.0f} codes"
    if tol[0] == "close":
        r = float(np.max(d / (tol[2] + tol[1] * np.abs(np.where(fin, w, 0)))))
        return r <= 1.0, f"{r:.2f} of tol"
    ref = float(np.abs(np.where(fin, w, 0)).max())
    ref = ref if tol[0] == "relmax" else max(1.0, ref)
    return float(d.max()) <= tol[1] * ref, f"max err {d.max() / ref:.1e} rel"


def status(fx: Run, eg: Run) -> str:
    if fx.aie and eg.aie:
        return "parity"
    if fx.aie:
        return "GAP"
    return "eager-only" if eg.aie else "both ref"


@dataclass
class Row:
    name: str
    family: str
    fixed: Run
    eager: Run
    ref: Run
    fixed_ok: "tuple[bool, str]"
    eager_ok: "tuple[bool, str]"

    @property
    def status(self) -> str:
        return status(self.fixed, self.eager)


def judge(r: Run, ref: Run, tol) -> "tuple[bool, str]":
    if not r.aie:
        return True, "-"
    return within(tol, r.out, ref.out)


def scoreboard_md(rows) -> str:
    gaps = sum(r.status == "GAP" for r in rows)
    bad = [r for r in rows if not r.eager_ok[0] or not r.fixed_ok[0]]
    out = ["# aiednn operator-support parity: fixed megakernel vs eager template library", "",
           "Workstation emulation (`src/aiegraph/test/test_eager_parity.py`). An implementation counts as AIE when it",
           "is not `ref:` and called an AIE entry (or names one). Contract: the family's tolerance against `ref:`.", "",
           f"**{len(rows)} problems: {sum(r.status == 'parity' for r in rows)} parity, {gaps} gaps, "
           f"{sum(r.status == 'both ref' for r in rows)} both ref, {sum(r.status == 'eager-only' for r in rows)} "
           f"eager-only; {len(bad)} contract violations.**", "",
           "| problem | family | fixed impl | eager impl | status | fixed vs ref | eager vs ref |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.name} | {r.family} | `{r.fixed.impl}` | `{r.eager.impl}` | {r.status} | "
                   f"{r.fixed_ok[1]} | {r.eager_ok[1]}{'' if r.eager_ok[0] else ' **OUT OF CONTRACT**'} |")
    return "\n".join(out) + "\n"


@pytest.fixture(scope="module")
def board(tmp_path_factory):
    fixed = AiednnLib.build(tmp_path_factory.mktemp("parity_fixed"))
    eager = build_eager(tmp_path_factory.mktemp("parity_eager"))
    rows = {}
    for name, family, make in PROBLEMS:
        p = make()
        ref = run(fixed, p, "ref:") if p.want is None else Run("numpy", p.want)
        fx, eg = run(fixed, p), run(eager, p)
        rows[name] = Row(name, family, fx, eg, ref, judge(fx, ref, p.tol), judge(eg, ref, p.tol))
    text = scoreboard_md(list(rows.values()))
    (REPO / "worklocal").mkdir(exist_ok=True)
    (REPO / "worklocal" / "eager_parity.md").write_text(text)
    w = max(len(n) for n in NAMES)
    print("\n" + "\n".join(f"{r.name:<{w}}  {r.family:<12} {r.fixed.impl:<44} {r.eager.impl:<40} {r.status}"
                           f"{'' if r.eager_ok[0] else '  EAGER OUT OF CONTRACT: ' + r.eager_ok[1]}"
                           for r in rows.values()))
    return rows


def test_every_problem_has_a_ref(board):
    """every problem creates and runs on ref: (else it is malformed, not a parity finding)"""
    bad = {n: r.ref.impl + " " + r.ref.err for n, r in board.items() if r.ref.out is None}
    assert not bad, bad


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.xfail(strict=False, reason="eager gap"))
                                  if n in KNOWN_GAPS else n for n in NAMES])
def test_parity(board, name):
    r = board[name]
    if r.fixed.aie:
        assert r.eager.aie, f"fixed {r.fixed.impl} on the AIE, eager {r.eager.impl}"


@pytest.mark.parametrize("name", NAMES)
def test_contract(board, name):
    """an AIE result (either library) within the family's contract against ref: -- never xfailed"""
    r = board[name]
    assert r.fixed_ok[0], f"fixed {r.fixed.impl}: {r.fixed_ok[1]}"
    assert r.eager_ok[0], f"eager {r.eager.impl}: {r.eager_ok[1]}"
