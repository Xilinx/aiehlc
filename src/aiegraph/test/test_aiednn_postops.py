###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Post-ops on the non-GEMM primitives (aiednn_rt_ptail.h) and slice: fused == unfused chain."""

from __future__ import annotations

import ctypes as C
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, DT, MULTIPLE_SRC as MULTIPLE_SRC_ARG, POST_OP_SRC, AiednnLib, attr, conv_params, f, md, pool_params  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(41)
S, D = ARG["src"], ARG["dst"]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_postops"))


def f32(shape, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, size=shape).astype(np.float32)


def chain(y, post, r=None, dst_dt="f32", dscale=1.0, dzp=0):
    """the reference: y (the op's f32 result) through relu / add(r) / tanh, then the dst quantization"""
    v = y.astype(np.float64)
    for p in post:
        if p == "relu":
            v = np.maximum(v, 0)
        elif p == "add":
            v = v + r.astype(np.float64)
        elif p == "tanh":
            v = np.tanh(v)
    if dst_dt == "f32":
        return v.astype(np.float32)
    lo, hi = (-128, 127) if dst_dt == "s8" else (0, 255)
    return np.clip(np.rint(v / dscale + dzp), lo, hi).astype(DT[dst_dt][1])


def post_attr(lib, post, rshape, dst_dt):
    ops = []
    for p in post:
        ops.append(("binary", "add", rshape, "f32") if p == "add" else ("eltwise", p, 0.0, 0.0))
    return attr(lib.ALG, dst_scale=0.02 if dst_dt != "f32" else 0.0, dst_zp=3 if dst_dt == "u8" else 0,
                post_ops=ops)


def run_both(lib, fn, args_plain, args_attr_tail, bufs, dshape, dst_dt, post, rshape=None):
    """the plain op into f32 (the reference input), then the fused op"""
    y = np.zeros(dshape, np.float32)
    got_plain, _ = lib.call(fn, args_plain(md(dshape, "f32")), {**bufs, D: y}, D)
    r = f32(rshape) if rshape else None
    a, keep = post_attr(lib, post, rshape, dst_dt)
    out = np.zeros(dshape, DT[dst_dt][1])
    extra = {POST_OP_SRC + post.index("add"): r} if "add" in post else {}
    got, impl = lib.call(fn + "_attr", args_attr_tail(md(dshape, dst_dt), a), {**bufs, **extra, D: out}, D)
    want = chain(got_plain, post, r, dst_dt, 0.02, 3 if dst_dt == "u8" else 0)
    if dst_dt == "f32":
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    else:
        assert np.abs(got.astype(np.int64) - want.astype(np.int64)).max() <= 1   # .5 ties in f32 vs double
    assert impl.endswith("+pt"), impl
    return impl


POSTS = [(["relu"], "f32"), (["add", "relu"], "s8"), (["tanh"], "u8")]


@pytest.mark.parametrize("post,ddt", POSTS, ids=["relu", "add_relu_s8", "tanh_u8"])
def test_pooling_post_ops(lib, post, ddt):
    x = f32((2, 8, 9, 9))
    pp = pool_params((3, 3), (2, 2), (1, 1), (1, 1))
    alg = C.c_int(lib.ALG["pooling_avg_exclude_padding"])
    run_both(lib, "aiednn_pooling_forward_create",
             lambda dm: [alg, C.byref(md(x.shape, "f32")), C.byref(dm), C.byref(pp)],
             lambda dm, a: [alg, C.byref(md(x.shape, "f32")), C.byref(dm), C.byref(pp), C.byref(a)],
             {S: x}, (2, 8, 5, 5), ddt, post, (2, 8, 5, 5) if "add" in post else None)


@pytest.mark.parametrize("post,ddt", POSTS, ids=["relu", "add_relu_s8", "tanh_u8"])
def test_softmax_post_ops(lib, post, ddt):
    x = f32((4, 7, 33))
    alg = C.c_int(lib.ALG["softmax_accurate"])
    run_both(lib, "aiednn_softmax_forward_create",
             lambda dm: [alg, C.c_int(2), C.byref(md(x.shape, "f32")), C.byref(dm)],
             lambda dm, a: [alg, C.c_int(2), C.byref(md(x.shape, "f32")), C.byref(dm), C.byref(a)],
             {S: x}, x.shape, ddt, post, x.shape if "add" in post else None)


@pytest.mark.parametrize("post,ddt", POSTS, ids=["relu", "add_relu_s8", "tanh_u8"])
def test_reduction_post_ops(lib, post, ddt):
    x = f32((2, 16, 6, 6))
    alg = C.c_int(lib.ALG["reduction_mean"])
    run_both(lib, "aiednn_reduction_create",
             lambda dm: [alg, f(0.0), f(0.0), C.byref(md(x.shape, "f32")), C.byref(dm)],
             lambda dm, a: [alg, f(0.0), f(0.0), C.byref(md(x.shape, "f32")), C.byref(dm), C.byref(a)],
             {S: x}, (2, 16, 1, 1), ddt, post, (1, 16, 1, 1) if "add" in post else None)


@pytest.mark.parametrize("post,ddt", POSTS, ids=["relu", "add_relu_s8", "tanh_u8"])
def test_resampling_post_ops(lib, post, ddt):
    x = f32((1, 4, 5, 6))
    alg = C.c_int(lib.ALG["resampling_linear"])
    run_both(lib, "aiednn_resampling_forward_create",
             lambda dm: [alg, C.byref(md(x.shape, "f32")), C.byref(dm)],
             lambda dm, a: [alg, C.byref(md(x.shape, "f32")), C.byref(dm), C.byref(a)],
             {S: x}, (1, 4, 10, 12), ddt, post, (1, 4, 10, 12) if "add" in post else None)


@pytest.mark.parametrize("post,ddt", POSTS, ids=["relu", "add_relu_s8", "tanh_u8"])
def test_group_norm_post_ops(lib, post, ddt):
    x = f32((2, 8, 4, 4))
    run_both(lib, "aiednn_group_normalization_forward_inference_create",
             lambda dm: [C.c_int64(4), C.byref(md(x.shape, "f32")), C.byref(dm), f(1e-5), C.c_uint(0)],
             lambda dm, a: [C.c_int64(4), C.byref(md(x.shape, "f32")), C.byref(dm), f(1e-5), C.c_uint(0), C.byref(a)],
             {S: x}, x.shape, ddt, post, x.shape if "add" in post else None)


@pytest.mark.parametrize("post,ddt", POSTS, ids=["relu", "add_relu_s8", "tanh_u8"])
def test_layer_norm_post_ops(lib, post, ddt):
    x = f32((3, 10, 64))
    run_both(lib, "aiednn_layer_normalization_forward_inference_create",
             lambda dm: [C.byref(md(x.shape, "f32")), C.byref(dm), f(1e-5), C.c_uint(0)],
             lambda dm, a: [C.byref(md(x.shape, "f32")), C.byref(dm), f(1e-5), C.c_uint(0), C.byref(a)],
             {S: x}, x.shape, ddt, post, x.shape if "add" in post else None)


def test_batch_norm_relu(lib):
    """oneDNN's fp_bnorm_relu pattern"""
    x = f32((2, 6, 4, 5))
    stats = {ARG["mean"]: f32((6,)), ARG["variance"]: f32((6,), 0.5, 2.0)}
    run_both(lib, "aiednn_batch_normalization_forward_inference_create",
             lambda dm: [C.byref(md(x.shape, "f32")), C.byref(dm), f(1e-5), C.c_uint(0)],
             lambda dm, a: [C.byref(md(x.shape, "f32")), C.byref(dm), f(1e-5), C.c_uint(0), C.byref(a)],
             {S: x, **stats}, x.shape, "f32", ["relu"])


def test_per_channel_scales_refused(lib):
    x = f32((1, 4, 4, 4))
    sc = np.ones(4, np.float32)
    a, keep = attr(lib.ALG, scales=sc)
    prim, st = lib.create("aiednn_softmax_forward_create_attr", C.c_int(lib.ALG["softmax_accurate"]), C.c_int(1),
                          C.byref(md(x.shape, "f32")), C.byref(md(x.shape, "f32")), C.byref(a))
    assert st != 0


# ── slice ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("off,steps,dshape", [((0, 0, 768), None, (1, 128, 768)),          # gpt2's k out of qkv
                                              ((0, 0, 0), (1, 1, 2), (1, 5, 4)),           # every other column
                                              ((0, 0, 0, 32), None, (1, 4, 6, 32))])       # rope's second half
def test_slice(lib, off, steps, dshape):
    sshape = {3: (1, 128, 2304) if dshape[-1] == 768 else (1, 5, 8), 4: (1, 4, 6, 64)}[len(dshape)]
    x = f32(sshape)
    off_a = (C.c_int64 * len(off))(*off)
    st_a = (C.c_int64 * len(off))(*steps) if steps else None
    got, impl = lib.call("aiednn_slice_create", [C.byref(md(x.shape, "f32")), C.byref(md(dshape, "f32")), off_a, st_a],
                         {S: x, D: np.zeros(dshape, np.float32)}, D)
    st = steps or (1,) * len(off)
    want = x[tuple(slice(o, o + n * s, s) for o, n, s in zip(off, dshape, st))]
    np.testing.assert_array_equal(got, want)
    assert impl == "ref:slice"


def test_slice_nhwc_channels(lib):
    """channels 8..15 of an nhwc tensor: the canon path (offsets follow the physical axes)"""
    x = f32((1, 16, 3, 4))
    off = (C.c_int64 * 4)(0, 8, 0, 0)
    xs = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    out = np.zeros((1, 3, 4, 8), np.float32)
    got, _ = lib.call("aiednn_slice_create", [C.byref(md(x.shape, "f32", fmt="nhwc")),
                                              C.byref(md((1, 8, 3, 4), "f32", fmt="nhwc")), off, None],
                      {S: xs, D: out}, D)
    np.testing.assert_array_equal(got.transpose(0, 3, 1, 2), x[:, 8:16])


@pytest.mark.parametrize("ddt,zp", [("s8", 0), ("s8", -3), ("u8", 113), ("u8", 0)])
def test_slice_quantize_is_slice_then_quantize(lib, ddt, zp):
    """aiednn_slice_create_attr with dst quantization: bit-identical to slice + quantizing reorder."""
    sc = 0.25
    x = f32((1, 32, 6, 5))
    x.reshape(-1)[:64] = (np.arange(64) - 32 + 0.5) * sc   # exact ties
    off = (C.c_int64 * 4)(0, 16, 0, 0)
    dshape = (1, 16, 6, 5)
    a, _keep = attr(lib.ALG, dst_scale=sc, dst_zp=zp)
    got, impl = lib.call("aiednn_slice_create_attr", [C.byref(md(x.shape, "f32")), C.byref(md(dshape, ddt)), off, None,
                                                      C.byref(a)],
                         {S: x, D: np.zeros(dshape, DT[ddt][1])}, D)
    assert impl == "ref:slice", impl
    piece = np.ascontiguousarray(x[:, 16:32])
    want, _ = lib.call("aiednn_reorder_create", [C.byref(md(dshape, "f32")), C.byref(md(dshape, ddt)), None,
                                                 C.c_float(sc), C.c_int32(zp)],
                       {S: piece, D: np.zeros(dshape, DT[ddt][1])}, D)
    np.testing.assert_array_equal(got, want)
    x2 = x.copy()
    x2[:, 16:32].reshape(-1)[:64] = (np.arange(64) - 32 + 0.5) * sc   # the ties inside the slice too
    got2, _ = lib.call("aiednn_slice_create_attr", [C.byref(md(x.shape, "f32")), C.byref(md(dshape, ddt)), off, None,
                                                    C.byref(a)],
                       {S: x2, D: np.zeros(dshape, DT[ddt][1])}, D)
    want2, _ = lib.call("aiednn_reorder_create", [C.byref(md(dshape, "f32")), C.byref(md(dshape, ddt)), None,
                                                  C.c_float(sc), C.c_int32(zp)],
                        {S: np.ascontiguousarray(x2[:, 16:32]), D: np.zeros(dshape, DT[ddt][1])}, D)
    np.testing.assert_array_equal(got2, want2)


def test_slice_relu_quantize_takes_the_tail(lib):
    """post-ops on a slice: the +pt tail (relu, then the dst quantization)"""
    x = f32((1, 8, 4, 4))
    off = (C.c_int64 * 4)(0, 4, 0, 0)
    dshape = (1, 4, 4, 4)
    a, _keep = attr(lib.ALG, dst_scale=0.02, post_ops=[("eltwise", "relu", 0.0, 0.0)])
    got, impl = lib.call("aiednn_slice_create_attr", [C.byref(md(x.shape, "f32")), C.byref(md(dshape, "s8")), off, None,
                                                      C.byref(a)],
                         {S: x, D: np.zeros(dshape, np.int8)}, D)
    assert impl.endswith("+pt"), impl
    want = np.clip(np.rint(np.maximum(x[:, 4:8], 0).astype(np.float64) / np.float64(np.float32(0.02))), -128, 127)
    np.testing.assert_array_equal(got, want.astype(np.int8))


def test_slice_attr_refusals(lib):
    a, _keep = attr(lib.ALG, dst_scale=0.1)
    off = (C.c_int64 * 2)(0, 1)
    _, st = lib.create("aiednn_slice_create_attr", C.byref(md((2, 4), "s8")), C.byref(md((2, 2), "s8")), off, None,
                       C.byref(a))
    assert st != 0   # 8-bit src: requantize through a reorder, not a slice
    _, st = lib.create("aiednn_slice_create_attr", C.byref(md((2, 4), "f32")), C.byref(md((2, 2), "f32")), off, None,
                       C.byref(a))
    assert st != 0   # f32 dst and no post-ops: the attr would be ignored


def test_slice_out_of_range_refused(lib):
    off = (C.c_int64 * 2)(0, 3)
    prim, st = lib.create("aiednn_slice_create", C.byref(md((2, 4), "f32")), C.byref(md((2, 2), "f32")), off, None)
    assert st != 0


@pytest.mark.parametrize("kind", ["pool", "softmax", "reduction"])
def test_dst_quantization_alone_is_applied(lib, kind):
    """An attr with only dst_scale / dst_zero_point must quantize."""
    x = f32((1, 16, 7, 7))
    a, keep = attr(lib.ALG, dst_scale=0.01, dst_zp=7)
    if kind == "pool":
        pp = pool_params((7, 7), (1, 1))
        dshape, fn = (1, 16, 1, 1), "aiednn_pooling_forward_create"
        head = [C.c_int(lib.ALG["pooling_avg_exclude_padding"])]
        tail_args = [C.byref(pp)]
    elif kind == "softmax":
        dshape, fn = x.shape, "aiednn_softmax_forward_create"
        head, tail_args = [C.c_int(lib.ALG["softmax_accurate"]), C.c_int(1)], []
    else:
        dshape, fn = (1, 16, 1, 1), "aiednn_reduction_create"
        head, tail_args = [C.c_int(lib.ALG["reduction_mean"]), f(0.0), f(0.0)], []
    y, _ = lib.call(fn, head + [C.byref(md(x.shape, "f32")), C.byref(md(dshape, "f32"))] + tail_args,
                    {S: x, D: np.zeros(dshape, np.float32)}, D)
    got, impl = lib.call(fn + "_attr", head + [C.byref(md(x.shape, "f32")), C.byref(md(dshape, "u8"))] + tail_args +
                         [C.byref(a)], {S: x, D: np.zeros(dshape, np.uint8)}, D)
    want = np.clip(np.rint(y.astype(np.float64) / 0.01 + 7), 0, 255)
    assert np.abs(got.astype(np.int64) - want).max() <= 1 and impl.endswith("+pt"), impl


# ── int8 concat, reorder + post-ops ───────────────────────────────────────

def test_int8_concat_requantizes_each_source(lib):
    """oneDNN's int8_concat_fusion: dequantize each source -> concat -> quantize, exact"""
    a = RNG.integers(-128, 128, (1, 3, 4, 4)).astype(np.int8)
    b = RNG.integers(0, 256, (1, 5, 4, 4)).astype(np.uint8)
    c = RNG.integers(-128, 128, (1, 2, 4, 4)).astype(np.int8)
    MDS = type(md((1,), "s8")) * 3
    srcs = MDS(md(a.shape, "s8"), md(b.shape, "u8"), md(c.shape, "s8"))
    scales = (C.c_float * 3)(0.05, 0.02, 0.1)      # c already in dst's quantization: a plain copy
    zps = (C.c_int32 * 3)(2, 128, -1)
    out = np.zeros((1, 10, 4, 4), np.int8)
    got, impl = lib.call("aiednn_concat_create_q", [C.c_int(3), srcs, scales, zps, C.c_int(1),
                                                      C.byref(md(out.shape, "s8")), f(0.1), C.c_int32(-1)],
                         {MULTIPLE_SRC_ARG + 0: a, MULTIPLE_SRC_ARG + 1: b, MULTIPLE_SRC_ARG + 2: c, D: out}, D)
    deq = lambda x, s, z: (x.astype(np.float64) - z) * np.float32(s)
    v = np.concatenate([deq(a, 0.05, 2).astype(np.float32), deq(b, 0.02, 128).astype(np.float32),
                        deq(c, 0.1, -1).astype(np.float32)], axis=1)
    want = np.clip(np.rint(v.astype(np.float64) / np.float64(np.float32(0.1))) - 1, -128, 127).astype(np.int8)
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(got[:, 8:], c)


def test_reorder_dequant_add_quantize(lib):
    """s8 -> dequantize -> add a f32 tensor -> relu -> u8 as one reorder (x8 reorder + binary post-op chain)"""
    x = RNG.integers(-128, 128, (2, 6, 5)).astype(np.int8)
    r = f32((2, 6, 5))
    a, keep = attr(lib.ALG, dst_scale=0.03, dst_zp=4, post_ops=[("binary", "add", r.shape, "f32"),
                                                                ("eltwise", "relu", 0.0, 0.0)])
    got, impl = lib.call("aiednn_reorder_create_attr", [C.byref(md(x.shape, "s8")), C.byref(md(x.shape, "u8")), None,
                                                         f(0.05), C.c_int32(3), C.byref(a)],
                         {S: x, POST_OP_SRC: r, D: np.zeros(x.shape, np.uint8)}, D)
    v = np.float32((x.astype(np.float64) - 3) * np.float64(np.float32(0.05))).astype(np.float64) + r
    want = np.clip(np.rint(np.maximum(v, 0) / np.float64(np.float32(0.03)) + 4), 0, 255)
    assert np.abs(got.astype(np.int64) - want).max() <= 1 and impl.endswith("+pt"), impl


@pytest.mark.parametrize("perm", [(0, 2, 1, 3), (2, 0, 1, 3), (1, 0, 2)], ids=["heads", "outer", "3d"])
def test_reorder_permute_keeping_last_axis(lib, perm):
    """head split / merge permutes take the row-copy path; same bytes as numpy's transpose"""
    shape = (2, 5, 3, 8) if len(perm) == 4 else (4, 6, 8)
    x = f32(shape)
    dshape = tuple(shape[i] for i in perm)
    pa = (C.c_int * len(perm))(*perm)
    got, _ = lib.call("aiednn_reorder_create", [C.byref(md(shape, "f32")), C.byref(md(dshape, "f32")), pa, f(1.0),
                                                C.c_int32(0)], {S: x, D: np.zeros(dshape, np.float32)}, D)
    np.testing.assert_array_equal(got, x.transpose(perm))


@pytest.mark.parametrize("bias,res", [(False, None), (True, None), (False, "f32"), (True, "f32"), (True, "s8"),
                                      (False, "u8")],
                         ids=["plain", "bias", "residual", "bias_residual", "bias_s8_skip", "u8_skip"])
def test_int8_linear_f32_epilogue_matches_ref(lib, bias, res):
    """The int8 linear-layer epilogue fast paths are bit-identical to ref:matmul's generic epilogue."""
    M, K, N = 33, 96, 200
    x = RNG.integers(-128, 128, (M, K)).astype(np.int8)
    w = RNG.integers(-128, 128, (K, N)).astype(np.int8)
    sc = RNG.uniform(1e-4, 1e-3, N).astype(np.float32)
    b = f32((1, N)) if bias else None
    r = {"s8": RNG.integers(-128, 128, (M, N)).astype(np.int8), "u8": RNG.integers(0, 256, (M, N)).astype(np.uint8)
         }.get(res, f32((M, N)))
    po = [("binary", "add", r.shape, res) + ((0.031, 3 if res == "s8" else 121) if res in ("s8", "u8") else ())]
    a, keep = attr(lib.ALG, scales=sc, post_ops=po if res else [])
    outs = []
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            bufs = {S: x, ARG["weights"]: w, D: np.zeros((M, N), np.float32)}
            if bias:
                bufs[ARG["bias"]] = b
            if res:
                bufs[POST_OP_SRC] = r
            got, impl = lib.call("aiednn_matmul_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")),
                                  C.byref(md(b.shape, "f32")) if bias else None, C.byref(md((M, N), "f32")), C.c_int(0),
                                  C.byref(a)], bufs, D)
        finally:
            lib.set_filter(None)
        outs.append((got, impl))
    assert "aie:" in outs[0][1] and outs[1][1].startswith("ref:"), (outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][0], outs[1][0])


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("ddt", ["s8", "u8"])
def test_conv_u8_residual_requantized_on_the_core(lib, k, ddt):
    """A fused residual with a u8 src1 takes the on-core requant, bit-identical to ref:."""
    N, Cc, O, H = 1, 32, 64, 9
    x = RNG.integers(-128, 128, (N, Cc, H, H)).astype(np.int8)
    w = RNG.integers(-128, 128, (O, Cc, k, k)).astype(np.int8)
    r = RNG.integers(0, 256, (N, O, H, H)).astype(np.uint8)
    sc, b = RNG.uniform(1e-4, 1e-3, O).astype(np.float32), f32((O,), -1, 1)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=4 if ddt == "u8" else 0,
                   post_ops=[("binary", "add", r.shape, "u8", 0.03, 117), ("eltwise", "relu", 0.0, 0.0)])
    p = k // 2
    cp = conv_params((1, 1), (p, p), (p, p), (0, 0), 1)
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        lib.entry_calls()
        try:
            got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((N, O, H, H), ddt)), C.byref(cp), C.byref(a)],
                                 {S: x, ARG["weights"]: w, ARG["bias"]: b, POST_OP_SRC: r,
                                  D: np.zeros((N, O, H, H), DT[ddt][1])}, D)
        finally:
            lib.set_filter(None)
        outs[filt] = (got, {e for e, n in lib.entry_calls().items() if n})
    np.testing.assert_array_equal(outs[None][0], outs["ref:"][0])
    assert any("_q_" in e for e in outs[None][1]), outs[None][1]   # the requantizing core entry ran


@pytest.mark.parametrize("alg", ["gelu_tanh", "gelu_erf"])
@pytest.mark.parametrize("ddt", ["s8", "u8"])
def test_int8_linear_gelu_to_int8_matches_the_formula(lib, alg, ddt):
    """GELU int8 epilogue step table returns the double-precision codes."""
    import math
    M, K, N = 40, 128, 96
    x = RNG.integers(-128, 128, (M, K)).astype(np.int8)
    w = RNG.integers(-128, 128, (K, N)).astype(np.int8)
    sc, b = RNG.uniform(5e-5, 2e-4, N).astype(np.float32), f32((1, N), -1, 1)
    ds, zp = 0.0219, (0 if ddt == "s8" else 20)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=ds, dst_zp=zp, post_ops=[("eltwise", alg, 0.0, 0.0)])
    odt = np.int8 if ddt == "s8" else np.uint8
    got, impl = lib.call("aiednn_matmul_forward_create_attr",
                         [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                          C.byref(md((M, N), ddt)), C.c_int(0), C.byref(a)],
                         {S: x, ARG["weights"]: w, ARG["bias"]: b, D: np.zeros((M, N), odt)}, D)
    assert "aie:" in impl, impl
    acc = x.astype(np.int64) @ w.astype(np.int64)
    v = acc * sc.astype(np.float64) + b.astype(np.float64)
    if alg == "gelu_tanh":
        f = 0.5 * v * (1.0 + np.tanh(0.79788456080286535588 * (v + 0.044715 * v * v * v)))
    else:
        f = v * 0.5 * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))
    lo, hi = (-128, 127) if ddt == "s8" else (0, 255)
    want = np.clip(np.rint(f / ds + zp), lo, hi).astype(odt)
    # erf: aiednn's Phi is a polynomial, not libm -- a code may sit on a rounding edge
    assert np.count_nonzero(got != want) <= (0 if alg == "gelu_tanh" else 2), np.argwhere(got != want)[:5]


@pytest.fixture
def ntile_emu(monkeypatch):
    """the n-tile launches emulated on the workstation (aiednn_rt_ntile.h, AIEDNN_NTILE_EMU)"""
    monkeypatch.setenv("AIEDNN_NTILE_EMU", "1")


@pytest.mark.parametrize("tr", ["1", "0"], ids=["either", "untransposed"])
@pytest.mark.parametrize("M,N,K", [(128, 768, 768), (300, 300, 1100), (3000, 130, 1100), (100, 3000, 700)],
                         ids=["bert_proj", "multi_block", "chunked", "transposed_chunked"])
def test_int8_matmul_ntile_matches_per_block_launches(lib, ntile_emu, monkeypatch, M, N, K, tr):
    """The int8 matmul's K-chained blocks as n-tile launches give the per-block result bit for bit."""
    monkeypatch.setenv("AIEDNN_NTILE_MM_T", tr)
    x = RNG.integers(-128, 128, (M, K)).astype(np.int8)
    w = RNG.integers(-128, 128, (K, N)).astype(np.int8)
    sc, b = RNG.uniform(1e-4, 1e-3, N).astype(np.float32), f32((1, N))
    a, keep = attr(lib.ALG, scales=sc)
    outs = []
    for on in ("0", "1"):
        monkeypatch.setenv("AIEDNN_NO_NTILE_MM", "0" if on == "1" else "1")
        got, impl = lib.call("aiednn_matmul_forward_create_attr",
                             [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                              C.byref(md((M, N), "f32")), C.c_int(0), C.byref(a)],
                             {S: x, ARG["weights"]: w, ARG["bias"]: b, D: np.zeros((M, N), np.float32)}, D)
        assert "aie:" in impl, impl
        outs.append(got)
    np.testing.assert_array_equal(outs[0], outs[1])


def test_sdpa_bf16_queue_as_ntile_launches(lib, ntile_emu, monkeypatch):
    """AIEDNN_B16N: the queued attention tiles as n-tile launches -- the same output as the per-tile sequence"""
    import math
    B, H, Sq, Skv, Dh, Dv = 1, 6, 40, 300, 64, 24
    q, k, v = f32((B, H, Sq, Dh)), f32((B, 3, Skv, Dh)), f32((B, 3, Skv, Dv))
    outs = []
    for on in ("0", "1"):
        monkeypatch.setenv("AIEDNN_B16N", on)
        got, impl = lib.call("aiednn_sdpa_forward_create",
                             [C.byref(md(q.shape, "f32")), C.byref(md(k.shape, "f32")), C.byref(md(v.shape, "f32")),
                              None, C.byref(md((B, H, Sq, Dv), "f32")), C.c_float(1 / math.sqrt(Dh)), C.c_int(1)],
                             {S: q, ARG["src_1"]: k, ARG["src_2"]: v, D: np.zeros((B, H, Sq, Dv), np.float32)}, D)
        assert impl.startswith("gemm+sdpa+") and "aie:" in impl, impl
        outs.append(got)
    np.testing.assert_array_equal(outs[0], outs[1])


def _dq(x, sc, zp):
    """the dequantize reorder: f32((code - zp) * scale), in double as the runtime computes it"""
    return ((x.astype(np.float64) - zp) * np.float64(np.float32(sc))).astype(np.float32)


@pytest.mark.parametrize("sdt,zp", [("s8", -3), ("u8", 120)])
@pytest.mark.parametrize("op", ["layer_norm", "layer_norm_q", "softmax", "softmax_q"])
def test_dequantized_source_equals_dequantize_then_op(lib, sdt, zp, op):
    """attr.src_scales on layer norm / softmax: byte-identical to a dequantize reorder in front."""
    shape, sc = (3, 7, 40), 0.037
    x = (RNG.integers(0, 256, shape).astype(np.uint8) if sdt == "u8" else RNG.integers(-128, 128, shape).astype(np.int8))
    q = op.endswith("_q")
    ddt, odt = ("s8", np.int8) if q else ("f32", np.float32)
    a1, k1 = attr(lib.ALG, dst_scale=0.02 if q else 0.0, dst_zp=1 if q else 0)
    a1.src_scales[0], a1.src_zero_points[0] = sc, zp
    a2, k2 = attr(lib.ALG, dst_scale=0.02 if q else 0.0, dst_zp=1 if q else 0)
    xf = _dq(x, sc, zp)
    if op.startswith("layer_norm"):
        g, b = f32((shape[-1],)), f32((shape[-1],))
        fl = 1 | 2  # AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT
        call = lambda src, dt, at: lib.call("aiednn_layer_normalization_forward_inference_create_attr",
                                            [C.byref(md(shape, dt)), C.byref(md(shape, ddt)), C.c_float(1e-5),
                                             C.c_uint(fl), C.byref(at)],
                                            {S: src, D: np.zeros(shape, odt), ARG["scale"]: g, ARG["shift"]: b}, D)
    else:
        call = lambda src, dt, at: lib.call("aiednn_softmax_forward_create_attr",
                                            [C.c_int(lib.ALG["softmax_accurate"]), C.c_int(2), C.byref(md(shape, dt)),
                                             C.byref(md(shape, ddt)), C.byref(at)], {S: src, D: np.zeros(shape, odt)}, D)
    got, impl = call(x, sdt, a1)
    want, _ = (call(xf, "f32", a2) if q else
               (lib.call("aiednn_layer_normalization_forward_inference_create",
                         [C.byref(md(shape, "f32")), C.byref(md(shape, "f32")), C.c_float(1e-5), C.c_uint(1 | 2)],
                         {S: xf, D: np.zeros(shape, np.float32), ARG["scale"]: g, ARG["shift"]: b}, D)
                if op == "layer_norm" else
                lib.call("aiednn_softmax_forward_create", [C.c_int(lib.ALG["softmax_accurate"]), C.c_int(2),
                                                           C.byref(md(shape, "f32")), C.byref(md(shape, "f32"))],
                         {S: xf, D: np.zeros(shape, np.float32)}, D)))
    assert impl.endswith("+dq"), impl
    np.testing.assert_array_equal(got, want)


def test_dequantized_qkv_sdpa_equals_dequantize_then_sdpa(lib):
    """sdpa with 8-bit Q / K / V read through src scales: byte-identical to the f32 sdpa."""
    B, H, Sq, Skv, Dh = 1, 2, 9, 11, 16
    q = RNG.integers(-128, 128, (B, H, Sq, Dh)).astype(np.int8)
    k = RNG.integers(-128, 128, (B, H, Skv, Dh)).astype(np.int8)
    v = RNG.integers(0, 256, (B, H, Skv, Dh)).astype(np.uint8)
    a, keep = attr(lib.ALG)
    a.src_scales[0], a.src_zero_points[0] = 0.02, 0
    a.src_scales[1], a.src_zero_points[1] = 0.03, -2
    a.src_scales_23[0], a.src_zero_points_23[0] = 0.01, 128
    od = (B, H, Sq, Dh)
    got, impl = lib.call("aiednn_sdpa_forward_create_attr",
                         [C.byref(md(q.shape, "s8")), C.byref(md(k.shape, "s8")), C.byref(md(v.shape, "u8")), None,
                          C.byref(md(od, "f32")), C.c_float(0.25), C.c_int(0), C.byref(a)],
                         {S: q, ARG["src_1"]: k, ARG["src_2"]: v, D: np.zeros(od, np.float32)}, D)
    qf, kf, vf = _dq(q, 0.02, 0), _dq(k, 0.03, -2), _dq(v, 0.01, 128)
    want, impl2 = lib.call("aiednn_sdpa_forward_create",
                           [C.byref(md(q.shape, "f32")), C.byref(md(k.shape, "f32")), C.byref(md(v.shape, "f32")), None,
                            C.byref(md(od, "f32")), C.c_float(0.25), C.c_int(0)],
                           {S: qf, ARG["src_1"]: kf, ARG["src_2"]: vf, D: np.zeros(od, np.float32)}, D)
    assert impl == impl2 + "+dq", (impl, impl2)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dt", ["f32", "s8"])
@pytest.mark.parametrize("dst_bshd", [0, 4])
def test_sdpa_reads_a_packed_qkv(lib, dt, dst_bshd):
    """AIEDNN_SDPA_QKV_PACKED: byte-identical to the same sdpa on split, permuted tensors."""
    B, S_, H, Dh = 2, 13, 3, 16
    qkv = (RNG.integers(-128, 128, (B, S_, 3, H, Dh)).astype(np.int8) if dt == "s8"
           else f32((B, S_, 3, H, Dh), -1, 1))
    sc = (0.02, 0.03, 0.01)
    split = [np.ascontiguousarray(qkv[:, :, t].transpose(0, 2, 1, 3)) for t in range(3)]  # [B, H, S, D]
    if dt == "s8":
        split = [_dq(x, sc[t], 0) for t, x in enumerate(split)]
    od = (B, S_, H, Dh) if dst_bshd else (B, H, S_, Dh)
    lg = (B, H, S_, Dh)
    a, keep = attr(lib.ALG)
    if dt == "s8":
        a.src_scales[0], a.src_scales[1], a.src_scales_23[0] = sc
    got, impl = lib.call("aiednn_sdpa_forward_create_attr",
                         [C.byref(md(lg, dt)), C.byref(md(lg, dt)), C.byref(md(lg, dt)), None, C.byref(md(od, "f32")),
                          C.c_float(0.25), C.c_int(8 | dst_bshd | 1), C.byref(a)],
                         {S: qkv, ARG["src_1"]: qkv, ARG["src_2"]: qkv, D: np.zeros(od, np.float32)}, D)
    want, _ = lib.call("aiednn_sdpa_forward_create",
                       [C.byref(md(lg, "f32")), C.byref(md(lg, "f32")), C.byref(md(lg, "f32")), None,
                        C.byref(md(od, "f32")), C.c_float(0.25), C.c_int(dst_bshd | 1)],
                       {S: split[0], ARG["src_1"]: split[1], ARG["src_2"]: split[2], D: np.zeros(od, np.float32)}, D)
    assert impl.endswith("+dq"), impl
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("case", ["tiny_step_big_bias", "shift_spread", "tiny_step_residual"])
def test_requant_with_extreme_scales_stays_on_the_core(lib, case):
    """A tiny accumulator step lowers the shift and stays on the core's requant, bit-identical to ref:."""
    N, Cc, O, H = 1, 32, 64, 9
    x = RNG.integers(-128, 128, (N, Cc, H, H)).astype(np.int8)
    w = RNG.integers(-128, 128, (O, Cc, 1, 1)).astype(np.int8)
    sc = RNG.uniform(2e-14, 4e-14, O).astype(np.float32)
    if case == "shift_spread":
        sc[::2] *= 2.0 ** 18  # half the channels 18 octaves coarser
    ds = 1e-9
    b = (RNG.uniform(-300, 300, O) * ds).astype(np.float32)  # |bias / ds| up to 300 codes
    post = [("eltwise", "relu", 0.0, 0.0)]
    r = RNG.integers(-128, 128, (N, O, H, H)).astype(np.int8)
    if case == "tiny_step_residual":
        post = [("binary", "add", r.shape, "s8", 5e-9, 3)] + post
    a, keep = attr(lib.ALG, scales=sc, dst_scale=ds, post_ops=post)
    cp = conv_params((1, 1), (0, 0), (0, 0), (0, 0), 1)
    outs = {}
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        lib.entry_calls()
        try:
            bufs = {S: x, ARG["weights"]: w, ARG["bias"]: b, D: np.zeros((N, O, H, H), np.int8)}
            if case == "tiny_step_residual":
                bufs[POST_OP_SRC] = r
            got, impl = lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((N, O, H, H), "s8")), C.byref(cp), C.byref(a)], bufs, D)
        finally:
            lib.set_filter(None)
        outs[filt] = (got, {e for e, n in lib.entry_calls().items() if n})
    np.testing.assert_array_equal(outs[None][0], outs["ref:"][0])
    # the requantizing core entry ran (gemm_q, or MMT's vector requant for this short K)
    assert any("_q_" in e or e.startswith("mmt") for e in outs[None][1]), outs[None][1]
    acc = np.einsum("nchw,oc->nohw", x.astype(np.int64), w[:, :, 0, 0].astype(np.int64)).astype(np.float64)
    v = acc * sc.astype(np.float64)[None, :, None, None] + b.astype(np.float64)[None, :, None, None]
    if case == "tiny_step_residual":
        v = v + (r.astype(np.float64) - 3) * 5e-9
    want = np.clip(np.rint(np.maximum(v, 0) / ds), -128, 127)
    assert np.abs(outs[None][0].astype(np.int64) - want).max() <= 1


@pytest.mark.parametrize("ash,bsh", [((1, 128, 512), (1, 1, 512)), ((1, 128, 512), (1, 128, 1)), ((1, 128, 1), (1, 1, 1)),
                                     ((1, 8, 128, 64), (1, 1, 128, 64)), ((2, 3, 5), (2, 1, 1)), ((2, 3, 5), (1, 3, 1))])
@pytest.mark.parametrize("alg", ["binary_mul", "binary_add", "binary_sub", "binary_max"])
def test_f32_binary_common_broadcasts(lib, ash, bsh, alg):
    """The f32 binary's broadcast fast path equals the per-element loop."""
    x, y = f32(ash), f32(bsh)
    lib.set_filter("ref:")
    try:
        got, impl = lib.call("aiednn_binary_create", [C.c_int(lib.ALG[alg]), C.byref(md(ash, "f32")),
                                                      C.byref(md(bsh, "f32")), None, C.byref(md(ash, "f32"))],
                             {S: x, ARG["src_1"]: y, D: np.zeros(ash, np.float32)}, D)
    finally:
        lib.set_filter(None)
    f = {"binary_mul": np.multiply, "binary_add": np.add, "binary_sub": np.subtract, "binary_max": np.maximum}[alg]
    want = f(x.astype(np.float64), np.broadcast_to(y, ash).astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(got, want)


def test_gen_index(lib):
    """gen_index: the s32 position along an axis"""
    shape = (2, 3, 5)
    for axis in range(3):
        got, impl = lib.call("aiednn_gen_index_create", [C.byref(md(shape, "s32")), C.c_int(axis)],
                             {D: np.zeros(shape, np.int32)}, D)
        want = np.broadcast_to(np.arange(shape[axis]).reshape([-1 if i == axis else 1 for i in range(3)]), shape)
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("direction", ["quantize", "dequantize"])
@pytest.mark.parametrize("axis", [-1, 1])
@pytest.mark.parametrize("qdt", ["s8", "u8"])
def test_dynamic_quantize(lib, direction, axis, qdt):
    """Dynamic quantize / dequantize with execute-time scales equal the static reorder."""
    shape = (2, 6, 7)
    nc = 1 if axis < 0 else shape[axis]
    sc = RNG.uniform(0.01, 0.05, nc).astype(np.float32)
    zp = RNG.integers(-5, 6, nc).astype(np.int32) + (0 if qdt == "s8" else 120)
    lo, hi = (-128, 127) if qdt == "s8" else (0, 255)
    bshape = [-1 if i == axis else 1 for i in range(3)] if axis >= 0 else [1, 1, 1]
    s_, z_ = sc.reshape(bshape).astype(np.float64), zp.reshape(bshape).astype(np.float64)
    odt = np.int8 if qdt == "s8" else np.uint8
    if direction == "quantize":
        x = f32(shape, -3, 3)
        sdt, ddt, dz = "f32", qdt, np.zeros(shape, odt)
        want = np.clip(np.rint(x.astype(np.float64) / s_) + z_, lo, hi).astype(odt)
    else:
        x = RNG.integers(lo, hi + 1, shape).astype(odt)
        sdt, ddt, dz = qdt, "f32", np.zeros(shape, np.float32)
        want = ((x.astype(np.float64) - z_) * s_).astype(np.float32)
    got, impl = lib.call("aiednn_dynamic_quantize_create",
                         [C.byref(md(shape, sdt)), C.byref(md(shape, ddt)), C.byref(md((nc,), "f32")),
                          C.byref(md((nc,), "s32")), C.c_int(axis)],
                         {S: x, D: dz, 50: sc, 51: zp}, D)
    np.testing.assert_array_equal(got, want)
    if axis < 0:  # the static reorder with the same values: byte-identical
        want2, _ = lib.call("aiednn_reorder_create", [C.byref(md(shape, sdt)), C.byref(md(shape, ddt)), None,
                                                      C.c_float(sc[0]), C.c_int32(int(zp[0]))], {S: x, D: dz.copy()}, D)
        np.testing.assert_array_equal(got, want2)


@pytest.mark.parametrize("tol", [0, 1])
def test_quantized_mul_on_the_aie_only_with_the_tolerance_flag(lib, tol):
    """Quantized binary mul: exact table by default, FVEC within one code with the tolerance flag."""
    shp = (1, 8, 2048)
    x, y = RNG.integers(-128, 128, shp).astype(np.int8), RNG.integers(-128, 128, shp).astype(np.int8)
    def run(flags, filt=None):
        at, keep = attr(lib.ALG, dst_scale=0.05)
        at.src_scales[0], at.src_scales[1], at.src_zero_points[0] = 0.02, 0.03, 3
        at.flags = flags
        lib.set_filter(filt)
        try:
            return lib.call("aiednn_binary_create_attr", [C.c_int(lib.ALG["binary_mul"]), C.byref(md(shp, "s8")),
                                                          C.byref(md(shp, "s8")), C.byref(md(shp, "s8")), C.byref(at)],
                            {S: x, ARG["src_1"]: y, D: np.zeros(shp, np.int8)}, D)
        finally:
            lib.set_filter(None)
    got, impl = run(tol)
    want, _ = run(0, "ref:")
    assert impl.startswith("fvec:") == bool(tol), impl
    assert np.abs(got.astype(int) - want.astype(int)).max() <= (1 if tol else 0)


@pytest.mark.parametrize("dt", ["s8", "u8", "f32"])
@pytest.mark.parametrize("ish,osh", [((2, 3, 5, 7), (2, 3, 10, 14)), ((1, 4, 6, 6), (1, 4, 9, 4)), ((2, 2, 9), (2, 2, 20)),
                                     ((1, 2, 3, 4, 5), (1, 2, 6, 8, 10))])
def test_nearest_resampling_copy_equals_the_generic_loop(lib, monkeypatch, dt, ish, osh):
    """ref:resampling nearest, one type in and out: the byte-copy path equals the per-element loop byte for byte"""
    x = (RNG.integers(-128, 128, ish).astype(np.int8) if dt == "s8" else
         RNG.integers(0, 256, ish).astype(np.uint8) if dt == "u8" else f32(ish))
    outs = []
    for generic in ("0", "1"):
        monkeypatch.setenv("AIEDNN_REF_GENERIC", generic)
        lib.set_filter("ref:")
        try:
            got, _ = lib.call("aiednn_resampling_forward_create", [C.c_int(lib.ALG["resampling_nearest"]),
                                                                   C.byref(md(ish, dt)), C.byref(md(osh, dt))],
                              {S: x, D: np.zeros(osh, x.dtype)}, D)
        finally:
            lib.set_filter(None)
        outs.append(got)
    np.testing.assert_array_equal(outs[0], outs[1])


@pytest.mark.parametrize("alg", ["swish", "gelu_tanh"])
def test_conv_activation_to_int8_plane_rows_match_the_formula(lib, alg):
    """An NCHW conv with swish / gelu_tanh -> s8: the plane-row epilogue returns the double codes."""
    N, Cc, O, H = 1, 16, 24, 20
    x = RNG.integers(-128, 128, (N, Cc, H, H)).astype(np.int8)
    w = RNG.integers(-128, 128, (O, Cc, 3, 3)).astype(np.int8)
    sc, b = RNG.uniform(2e-5, 1e-4, O).astype(np.float32), f32((O,), -1, 1)
    ds = 0.0195
    a, keep = attr(lib.ALG, scales=sc, dst_scale=ds, post_ops=[("eltwise", alg, 1.0, 0.0)])
    cp = conv_params((1, 1), (1, 1), (1, 1), (0, 0), 1)
    got, impl = lib.call("aiednn_convolution_forward_create_attr",
                         [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                          C.byref(md((N, O, H, H), "s8")), C.byref(cp), C.byref(a)],
                         {S: x, ARG["weights"]: w, ARG["bias"]: b, D: np.zeros((N, O, H, H), np.int8)}, D)
    xp = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    acc = np.zeros((N, O, H, H), np.int64)
    for kh in range(3):
        for kw in range(3):
            acc += np.einsum("nchw,oc->nohw", xp[:, :, kh:kh + H, kw:kw + H], w[:, :, kh, kw].astype(np.int64))
    v = acc * sc.astype(np.float64)[None, :, None, None] + b.astype(np.float64)[None, :, None, None]
    f = v / (1 + np.exp(-v)) if alg == "swish" else 0.5 * v * (1 + np.tanh(0.79788456080286535588 * (v + 0.044715 * v ** 3)))
    want = np.clip(np.rint(f / ds), -128, 127).astype(np.int8)
    assert np.count_nonzero(got != want) == 0, (impl, np.argwhere(got != want)[:5])
