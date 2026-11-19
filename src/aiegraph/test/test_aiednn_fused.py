###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Fused transformer primitives (aiednn_rt_fused.h): gated MLP and RoPE against PyTorch.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_fused.py
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
RNG = np.random.default_rng(51)
S, W, D = ARG["src"], ARG["weights"], ARG["dst"]
UP, DOWN, COS, SIN = 46, 47, 48, 49


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_fused"))


def f32(shape, sd=1.0):
    return RNG.normal(0, sd, size=shape).astype(np.float32)


@pytest.mark.parametrize("act,torch_act", [("swish", F.silu), ("gelu_tanh", lambda t: F.gelu(t, approximate="tanh"))],
                         ids=["swiglu", "geglu"])
@pytest.mark.parametrize("path", ["gemm+gmlp", "ref:"])
@pytest.mark.parametrize("linear", [False, True], ids=["dkf", "nn_linear"])
def test_gated_mlp(lib, act, torch_act, path, linear):
    M, Dm, Fd = 37, 64, 96
    x, wg, wu, wd = f32((1, M, Dm)), f32((Dm, Fd), 0.1), f32((Dm, Fd), 0.1), f32((Fd, Dm), 0.1)
    m = lambda a: C.byref(md(a.shape, "f32"))
    # nn.Linear layout (AIEDNN_GMLP_WEIGHTS_TRANSPOSED): [F, D] gate / up, [D, F] down
    ag, au, ad = (np.ascontiguousarray(w.T) for w in (wg, wu, wd)) if linear else (wg, wu, wd)
    lib.set_filter(None if path.startswith("gemm+") else path)
    try:
        got, impl = lib.call("aiednn_gated_mlp_create",
                             [m(x), m(ag), m(au), m(ad), m(x), C.c_int(lib.ALG["eltwise_" + act]), f(1.0), f(0.0),
                              C.c_uint(1 if linear else 0)],
                             {S: x, W: ag, UP: au, DOWN: ad, D: np.zeros_like(x)}, D)
    finally:
        lib.set_filter(None)
    assert impl.startswith(path), impl
    t = lambda a: torch.from_numpy(a).double()
    want = ((torch_act(t(x) @ t(wg)) * (t(x) @ t(wu))) @ t(wd)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("interleaved", [0, 1], ids=["half_split", "interleaved"])
def test_rope_bit_identical_to_the_torch_chain(lib, interleaved):
    B, H, Sq, Dh = 1, 4, 12, 16
    x = f32((B, H, Sq, Dh))
    ang = f32((1, 1, Sq, Dh))
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    m = lambda a: C.byref(md(a.shape, "f32"))
    got, impl = lib.call("aiednn_rope_create", [m(x), m(cos), m(sin), m(x), C.c_int(interleaved)],
                         {S: x, COS: cos, SIN: sin, D: np.zeros_like(x)}, D)
    tx, tc, ts = (torch.from_numpy(a) for a in (x, cos, sin))
    if interleaved:
        rot = torch.stack([-tx[..., 1::2], tx[..., 0::2]], dim=-1).reshape(tx.shape)
    else:
        h = Dh // 2
        rot = torch.cat([-tx[..., h:], tx[..., :h]], dim=-1)
    want = (tx * tc + rot * ts).numpy()                     # f32, the framework's three ops
    np.testing.assert_array_equal(got, want)
    assert impl == "ref:rope"


def _reorder(lib, x, dt_out, scale, zp):
    out = np.zeros(x.shape, {"f32": np.float32, "s8": np.int8, "u8": np.uint8}[dt_out])
    dt_in = {np.float32: "f32", np.int8: "s8", np.uint8: "u8"}[x.dtype.type]
    got, _ = lib.call("aiednn_reorder_create", [C.byref(md(x.shape, dt_in)), C.byref(md(x.shape, dt_out)), None,
                                               C.c_float(scale), C.c_int32(zp)], {S: x, D: out}, D)
    return got


@pytest.mark.parametrize("interleaved", [0, 1], ids=["half_split", "interleaved"])
def test_rope_quantized_is_the_qdq_chain(lib, interleaved):
    """aiednn_rope_create_attr with s8 x, u8 cos / sin (read through) and an s8 dst: byte-identical to the
    dequantizing reorders, the f32 rope and the quantizing reorder (a PT2E graph RoPE as one primitive)"""
    from aiegraph.dnn_ctypes import attr
    shape, cshape = (1, 12, 4, 16), (1, 12, 1, 16)
    x8 = RNG.integers(-128, 128, shape).astype(np.int8)
    c8 = RNG.integers(0, 256, cshape).astype(np.uint8)
    s8 = RNG.integers(0, 256, cshape).astype(np.uint8)
    xs, xz, cs, cz, ss, sz, ds, dz = 0.031, 3, 0.0078, 128, 0.0078, 127, 0.045, -5
    a, _keep = attr(lib.ALG, dst_scale=ds, dst_zp=dz)
    a.src_scales[0], a.src_zero_points[0] = xs, xz
    a.src_scales[1], a.src_zero_points[1] = cs, cz
    a.src_scales_23[0], a.src_zero_points_23[0] = ss, sz
    got, impl = lib.call("aiednn_rope_create_attr",
                         [C.byref(md(shape, "s8")), C.byref(md(cshape, "u8")), C.byref(md(cshape, "u8")),
                          C.byref(md(shape, "s8")), C.c_int(interleaved), C.byref(a)],
                         {S: x8, COS: c8, SIN: s8, D: np.zeros(shape, np.int8)}, D)
    assert impl == "ref:rope", impl
    xf, cf, sf = _reorder(lib, x8, "f32", xs, xz), _reorder(lib, c8, "f32", cs, cz), _reorder(lib, s8, "f32", ss, sz)
    m = lambda arr: C.byref(md(arr.shape, "f32"))
    yf, _ = lib.call("aiednn_rope_create", [m(xf), m(cf), m(sf), m(xf), C.c_int(interleaved)],
                     {S: xf, COS: cf, SIN: sf, D: np.zeros_like(xf)}, D)
    want = _reorder(lib, yf, "s8", ds, dz)
    np.testing.assert_array_equal(got, want)


def test_rope_attr_refusals(lib):
    from aiegraph.dnn_ctypes import attr
    shape, cshape = (1, 4, 2, 8), (1, 4, 1, 8)
    a, _keep = attr(lib.ALG)
    _, st = lib.create("aiednn_rope_create_attr", C.byref(md(shape, "s8")), C.byref(md(cshape, "f32")),
                       C.byref(md(cshape, "f32")), C.byref(md(shape, "f32")), C.c_int(0), C.byref(a))
    assert st != 0   # an s8 x needs src_scales[0] > 0
    a, _keep = attr(lib.ALG)
    _, st = lib.create("aiednn_rope_create_attr", C.byref(md(shape, "f32")), C.byref(md(cshape, "f32")),
                       C.byref(md(cshape, "f32")), C.byref(md(shape, "s8")), C.c_int(0), C.byref(a))
    assert st != 0   # an s8 dst needs dst_scale > 0

def test_rope_f32_src_s8_dst_is_rope_then_quantize(lib):
    """aiednn_rope_create_attr with f32 x / cos / sin and an s8 dst (attr dst_scale only): byte-identical to the
    f32 rope followed by the quantizing reorder"""
    from aiegraph.dnn_ctypes import attr
    shape, cshape = (1, 16, 4, 64), (1, 16, 1, 64)
    x = f32(shape)
    ang = f32(cshape)
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    ds, dz = 0.021, 0
    a, _keep = attr(lib.ALG, dst_scale=ds, dst_zp=dz)
    m = lambda arr, dt="f32": C.byref(md(arr.shape, dt))
    got, impl = lib.call("aiednn_rope_create_attr", [m(x), m(cos), m(sin), C.byref(md(shape, "s8")), C.c_int(0), C.byref(a)],
                         {S: x, COS: cos, SIN: sin, D: np.zeros(shape, np.int8)}, D)
    assert impl == "ref:rope", impl
    yf, _ = lib.call("aiednn_rope_create", [m(x), m(cos), m(sin), m(x), C.c_int(0)],
                     {S: x, COS: cos, SIN: sin, D: np.zeros_like(x)}, D)
    np.testing.assert_array_equal(got, _reorder(lib, yf, "s8", ds, dz))
