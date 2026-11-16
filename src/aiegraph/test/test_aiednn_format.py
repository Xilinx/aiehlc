###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Memory formats: every nhwc-operand primitive gives its dense result, transposed, bit for bit."""

from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import (ARG, DT, FMT, MULTIPLE_SRC, POST_OP_SRC, AiednnLib, attr, conv_params, f,  # noqa: E402
                                 md, pool_params)

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(11)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_fmt"))


def i8(shape):
    return RNG.integers(-128, 128, size=shape).astype(np.int8)


def f32(shape, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, size=shape).astype(np.float32)


def nhwc(x):
    return np.ascontiguousarray(x.transpose(0, 2, 3, 1)) if x.ndim == 4 else x


def nchw(x):
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def run(lib, fn, args, bufs, fmts, out_args):
    """Create with descriptors in formats fmts, execute with dense bufs; returns ({arg: dense result}, impl)."""
    cargs, keep = [], []
    for a in args:
        if isinstance(a, tuple) and a and a[0] == "md":
            m = md(a[1], a[2], fmt=fmts.get(a[3], "dense"))
            keep.append(m)
            cargs.append(C.byref(m))
        else:
            cargs.append(a)
    prim, st = lib.create(fn, *cargs)
    assert st == 0, f"{fn}: {lib.status(st)}"
    impl = lib.impl(prim)
    io = {k: (nhwc(v) if fmts.get(k) == "nhwc" else np.ascontiguousarray(v)) for k, v in bufs.items()}
    st = lib.execute(prim, io)
    lib.destroy(prim)
    assert st == 0, f"{fn} execute: {lib.status(st)}"
    return {k: (nchw(io[k]) if fmts.get(k) == "nhwc" else io[k]) for k in out_args}, impl


def both(lib, fn, args, bufs, fmts, out_args):
    dense, i0 = run(lib, fn, args, {k: v.copy() for k, v in bufs.items()}, {}, out_args)
    fm, i1 = run(lib, fn, args, {k: v.copy() for k, v in bufs.items()}, fmts, out_args)
    for k in out_args:
        np.testing.assert_array_equal(fm[k], dense[k])
    return i0, i1


S, D = ARG["src"], ARG["dst"]


def conv_case(lib, x, w, b, oshape, cp, dt_s, dt_w, dt_b, dt_d, fmts, a=None, extra=()):
    fn = "aiednn_convolution_forward_create" + ("_attr" if a is not None else "")
    args = [("md", x.shape, dt_s, S), ("md", w.shape, dt_w, ARG["weights"]), ("md", b.shape, dt_b, ARG["bias"]),
            ("md", oshape, dt_d, D), C.byref(cp)] + ([C.byref(a)] if a is not None else [])
    bufs = {S: x, ARG["weights"]: w, ARG["bias"]: b, D: np.zeros(oshape, DT[dt_d][1]), **dict(extra)}
    return both(lib, fn, args, bufs, fmts, [D])


@pytest.mark.parametrize("fmts", [{S: "nhwc", D: "nhwc"}, {S: "nhwc"}, {D: "nhwc"}], ids=["both", "src", "dst"])
@pytest.mark.parametrize("k,cin,g", [(3, 16, 1), (1, 32, 1), (7, 3, 1), (3, 8, 8)], ids=["3x3", "1x1", "stem", "dw"])
def test_conv_int8_fixed_point(lib, fmts, k, cin, g):
    N, O, H = 2, 32 if g == 1 else cin, 10
    x, w = i8((N, cin, H, H)), i8((O, cin // g, k, k))
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, post_ops=[("eltwise", "relu", 0.0, 0.0)])
    p = k // 2
    i0, i1 = conv_case(lib, x, w, b, (N, O, H, H), conv_params((1, 1), (p, p), (p, p), (0, 0), g), "s8", "s8", "f32",
                       "s8", fmts, a)
    # the AIE paths read / write nhwc themselves (im2col's pipes; depthwise items gather / scatter channels):
    # the dense implementation, no conversion
    assert i1 == i0 and i0.startswith("im2col+" if g == 1 else "dwblk:"), (i0, i1)


def test_conv_int8_residual(lib):
    N, Cc, O = 2, 16, 32
    x, w, r8 = i8((N, Cc, 9, 9)), i8((O, Cc, 3, 3)), i8((N, O, 9, 9))
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05,
                   post_ops=[("binary", "add", r8.shape, "s8", 0.03, 5, "nhwc"), ("eltwise", "relu", 0.0, 0.0)])
    fmts = {S: "nhwc", D: "nhwc", POST_OP_SRC: "nhwc"}
    dense_a, keep2 = attr(lib.ALG, scales=sc, dst_scale=0.05,
                          post_ops=[("binary", "add", r8.shape, "s8", 0.03, 5), ("eltwise", "relu", 0.0, 0.0)])
    cp = conv_params((1, 1), (1, 1), (1, 1), (0, 0), 1)
    dense, _ = run(lib, "aiednn_convolution_forward_create_attr",
                   [("md", x.shape, "s8", S), ("md", w.shape, "s8", 4), ("md", b.shape, "f32", 5),
                    ("md", (N, O, 9, 9), "s8", D), C.byref(cp), C.byref(dense_a)],
                   {S: x, 4: w, 5: b, POST_OP_SRC: r8, D: np.zeros((N, O, 9, 9), np.int8)}, {}, [D])
    fm, impl = run(lib, "aiednn_convolution_forward_create_attr",
                   [("md", x.shape, "s8", S), ("md", w.shape, "s8", 4), ("md", b.shape, "f32", 5),
                    ("md", (N, O, 9, 9), "s8", D), C.byref(cp), C.byref(a)],
                   {S: x, 4: w, 5: b, POST_OP_SRC: r8, D: np.zeros((N, O, 9, 9), np.int8)}, fmts, [D])
    np.testing.assert_array_equal(fm[D], dense[D])
    assert "fmtwrap" not in impl, impl


@pytest.mark.parametrize("fmts", [{S: "nhwc", D: "nhwc"}, {S: "nhwc"}, {D: "nhwc"}], ids=["both", "src", "dst"])
@pytest.mark.parametrize("k,s,epi", [(3, 1, "q"), (3, 2, "q"), (5, 1, "q"), (3, 1, "f32")])
def test_depthwise(lib, fmts, k, s, epi):
    """Depthwise on dwblk: with nhwc gather / scatter, on-core requant and host epilogue."""
    N, Cc, H = 2, 24, 13
    x, w = i8((N, Cc, H, H)), i8((Cc, 1, k, k))
    sc, b = f32((Cc,), 1e-4, 1e-3), f32((Cc,), -1, 1)
    p = k // 2
    Ho = (H + 2 * p - k) // s + 1
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05 if epi == "q" else 0.0,
                   post_ops=[("eltwise", "relu", 0.0, 0.0)])
    i0, i1 = conv_case(lib, x, w, b, (N, Cc, Ho, Ho), conv_params((s, s), (p, p), (p, p), (0, 0), Cc), "s8", "s8", "f32",
                       "s8" if epi == "q" else "f32", fmts, a)
    assert i1 == i0 and i0.startswith("dwblk:"), (i0, i1)


@pytest.mark.parametrize("k,cin,s", [(3, 16, 1), (7, 3, 2), (1, 32, 2)], ids=["3x3", "stem", "1x1s2"])
def test_conv_u8_zero_point(lib, k, cin, s):
    """The framework int8 conv (u8 with a zero point): zp+ writes its padded copy nhwc."""
    N, O, H = 1, 32, 12
    x = RNG.integers(0, 256, (N, cin, H, H)).astype(np.uint8)
    w = i8((O, cin, k, k))
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=3, src_zp=113)
    p = k // 2
    Ho = (H + 2 * p - k) // s + 1
    i0, i1 = conv_case(lib, x, w, b, (N, O, Ho, Ho), conv_params((s, s), (p, p), (p, p), (0, 0), 1), "u8", "s8", "f32",
                       "u8", {S: "nhwc", D: "nhwc"}, a)
    assert i1 == i0 and i0.startswith("zp+"), (i0, i1)


@pytest.mark.parametrize("fdst,fsrc1", [("nhwc", "nhwc"), ("nhwc", "dense"), ("dense", "nhwc")])
def test_conv_requant_binary_formats(lib, fdst, fsrc1):
    """Residual block conv (+qres) with the skip possibly in the other format than dst."""
    N, Cc, O, H = 1, 16, 16, 9
    x = RNG.integers(0, 256, (N, Cc, H, H)).astype(np.uint8)
    w, r = i8((O, Cc, 3, 3)), RNG.integers(0, 256, (N, O, H, H)).astype(np.uint8)
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.04, dst_zp=7, src_zp=113,
                   post_ops=[("requant_binary", "add", r.shape, "u8", 0.03, 110, 0.05, 120, "u8"),
                             ("eltwise", "relu", 0.0, 0.0)])
    fmts = {S: "nhwc", D: fdst, POST_OP_SRC: fsrc1}
    a.post_ops[0].src1.format = 1 if fsrc1 == "nhwc" else 0
    cp = conv_params((1, 1), (1, 1), (1, 1), (0, 0), 1)
    args = [("md", x.shape, "u8", S), ("md", w.shape, "s8", 4), ("md", b.shape, "f32", 5),
            ("md", (N, O, H, H), "u8", D), C.byref(cp), C.byref(a)]
    bufs = {S: x, 4: w, 5: b, POST_OP_SRC: r, D: np.zeros((N, O, H, H), np.uint8)}
    da, keep2 = attr(lib.ALG, scales=sc, dst_scale=0.04, dst_zp=7, src_zp=113,
                     post_ops=[("requant_binary", "add", r.shape, "u8", 0.03, 110, 0.05, 120, "u8"),
                               ("eltwise", "relu", 0.0, 0.0)])
    dense, i0 = run(lib, "aiednn_convolution_forward_create_attr", args[:-1] + [C.byref(da)],
                    {k: v.copy() for k, v in bufs.items()}, {}, [D])
    fm, i1 = run(lib, "aiednn_convolution_forward_create_attr", args, {k: v.copy() for k, v in bufs.items()}, fmts, [D])
    np.testing.assert_array_equal(fm[D], dense[D])
    assert i0.endswith("+qres") and "fmtwrap" not in i1, (i0, i1)


def test_conv_f32(lib):
    x, w, b = f32((1, 5, 8, 8)), f32((6, 5, 3, 3)), f32((6,))
    conv_case(lib, x, w, b, (1, 6, 8, 8), conv_params((1, 1), (1, 1), (1, 1), (0, 0), 1), "f32", "f32", "f32", "f32",
              {S: "nhwc", D: "nhwc"})


@pytest.mark.parametrize("alg", ["max", "avg_exclude_padding"])
@pytest.mark.parametrize("dt", ["s8", "u8", "f32"])
def test_pooling(lib, alg, dt):
    x = f32((2, 16, 9, 9)) if dt == "f32" else RNG.integers(-128 if dt == "s8" else 0, 128 if dt == "s8" else 256,
                                                             (2, 16, 9, 9)).astype(DT[dt][1])
    _, impl = both(lib, "aiednn_pooling_forward_create",
                   [C.c_int(lib.ALG["pooling_" + alg]), ("md", x.shape, dt, S), ("md", (2, 16, 5, 5), dt, D),
                    C.byref(pool_params((3, 3), (2, 2), (1, 1), (1, 1)))],
                   {S: x, D: np.zeros((2, 16, 5, 5), DT[dt][1])}, {S: "nhwc", D: "nhwc"}, [D])
    assert impl == ("nhwc:pooling" if dt != "f32" else impl.split("+")[0] + "+fmtwrap"), impl


@pytest.mark.parametrize("dt", ["s8", "f32"])
def test_eltwise_canon(lib, dt):
    x = i8((2, 8, 5, 6)) if dt == "s8" else f32((2, 8, 5, 6))
    i0, i1 = both(lib, "aiednn_eltwise_forward_create",
                  [C.c_int(lib.ALG["eltwise_relu"]), f(0.0), f(0.0), ("md", x.shape, dt, S), ("md", x.shape, dt, D)],
                  {S: x, D: np.zeros_like(x)}, {S: "nhwc", D: "nhwc"}, [D])
    assert i1 == i0, (i0, i1)   # canon: the same implementation as dense, no conversion


@pytest.mark.parametrize("bshape", [(2, 8, 5, 6), (1, 8, 1, 1), (2, 1, 5, 6)], ids=["same", "channel", "plane"])
def test_binary_canon(lib, bshape):
    a, b = f32((2, 8, 5, 6)), f32(bshape)
    fmts = {S: "nhwc", D: "nhwc", ARG["src_1"]: "nhwc" if bshape == a.shape else "dense"}
    both(lib, "aiednn_binary_create",
         [C.c_int(lib.ALG["binary_add"]), ("md", a.shape, "f32", S), ("md", b.shape, "f32", ARG["src_1"]), None,
          ("md", a.shape, "f32", D)], {S: a, ARG["src_1"]: b, D: np.zeros_like(a)}, fmts, [D])


def test_binary_mixed_layout_wraps(lib):
    """src nhwc, dst dense: no canon (the layouts differ), the wrap converts"""
    a, b = f32((2, 8, 5, 6)), f32((2, 8, 5, 6))
    both(lib, "aiednn_binary_create",
         [C.c_int(lib.ALG["binary_mul"]), ("md", a.shape, "f32", S), ("md", b.shape, "f32", ARG["src_1"]), None,
          ("md", a.shape, "f32", D)], {S: a, ARG["src_1"]: b, D: np.zeros_like(a)}, {S: "nhwc"}, [D])


def test_softmax_axis_follows(lib):
    x = f32((2, 6, 3, 4))
    both(lib, "aiednn_softmax_forward_create",
         [C.c_int(lib.ALG["softmax_accurate"]), C.c_int(1), ("md", x.shape, "f32", S), ("md", x.shape, "f32", D)],
         {S: x, D: np.zeros_like(x)}, {S: "nhwc", D: "nhwc"}, [D])


def test_concat_on_c(lib):
    xs = [f32((2, c, 4, 5)) for c in (3, 5)]
    MDS = type(md((1,), "f32")) * 2
    outs = {}
    for fmt in ("dense", "nhwc"):
        arr = MDS(*[md(x.shape, "f32", fmt=fmt) for x in xs])
        dmd = md((2, 8, 4, 5), "f32", fmt=fmt)
        prim, st = lib.create("aiednn_concat_create", C.c_int(2), arr, C.c_int(1), C.byref(dmd))
        assert st == 0, lib.status(st)
        bufs = {MULTIPLE_SRC + i: (nhwc(x) if fmt == "nhwc" else x) for i, x in enumerate(xs)}
        bufs[D] = np.zeros((2, 4, 5, 8) if fmt == "nhwc" else (2, 8, 4, 5), np.float32)
        assert lib.execute(prim, bufs) == 0
        lib.destroy(prim)
        outs[fmt] = nchw(bufs[D]) if fmt == "nhwc" else bufs[D]
    np.testing.assert_array_equal(outs["nhwc"], outs["dense"])


@pytest.mark.parametrize("fmts", [{D: "nhwc"}, {S: "nhwc"}, {S: "nhwc", D: "nhwc"}], ids=["to", "from", "both"])
def test_reorder_quantize(lib, fmts):
    x = f32((2, 8, 5, 6))
    both(lib, "aiednn_reorder_create", [("md", x.shape, "f32", S), ("md", x.shape, "s8", D), None, f(0.02), C.c_int32(3)],
         {S: x, D: np.zeros(x.shape, np.int8)}, fmts, [D])


def test_batch_norm_wraps(lib):
    x = f32((2, 6, 4, 5))
    mean, var, sc, sh = f32((6,)), f32((6,), 0.5, 2.0), f32((6,)), f32((6,))
    _, impl = both(lib, "aiednn_batch_normalization_forward_inference_create",
         [("md", x.shape, "f32", S), ("md", x.shape, "f32", D), f(1e-5), C.c_uint(3)],
         {S: x, ARG["mean"]: mean, ARG["variance"]: var, ARG["scale"]: sc, ARG["shift"]: sh, D: np.zeros_like(x)},
         {S: "nhwc", D: "nhwc"}, [D])
    assert impl.endswith("+fmtwrap"), impl


def test_get_size_and_set_format(lib):
    so = lib.so
    so.aiednn_memory_desc_get_size.restype = C.c_size_t
    m = md((2, 3, 4, 5), "s8")
    assert so.aiednn_memory_desc_get_size(C.byref(m)) == 120
    assert so.aiednn_memory_desc_set_format(C.byref(m), C.c_int(FMT["nhwc"])) == 0
    assert m.format == FMT["nhwc"]
    v = md((7,), "f32")
    assert so.aiednn_memory_desc_set_format(C.byref(v), C.c_int(FMT["nhwc"])) != 0   # nhwc: 4-D only
    assert so.aiednn_memory_desc_get_size(C.byref(md((5,), "f4_e2m1"))) == 3


@pytest.mark.parametrize("k,cin,hw", [(1, 160, 7), (1, 32, 14), (3, 16, 7)], ids=["1x1_7x7", "1x1_14x14", "3x3"])
def test_nhwc_conv_uses_the_dense_entries(lib, k, cin, hw):
    """An nhwc conv launches the same AIE entries as the dense one."""
    O = 96
    x, w = i8((1, cin, hw, hw)), i8((O, cin, k, k))
    sc, b = f32((O,), 1e-4, 1e-3), f32((O,), -1, 1)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05)
    p = k // 2
    cp = conv_params((1, 1), (p, p), (p, p), (0, 0), 1)
    calls = {}
    for name, fmts in (("dense", {}), ("nhwc", {S: "nhwc", D: "nhwc"})):
        lib.entry_calls()
        run(lib, "aiednn_convolution_forward_create_attr",
            [("md", x.shape, "s8", S), ("md", w.shape, "s8", 4), ("md", b.shape, "f32", 5), ("md", (1, O, hw, hw), "s8", D),
             C.byref(cp), C.byref(a)], {S: x, 4: w, 5: b, D: np.zeros((1, O, hw, hw), np.int8)}, fmts, [D])
        calls[name] = {e for e, n in lib.entry_calls().items() if n}
    assert calls["nhwc"] == calls["dense"], calls
