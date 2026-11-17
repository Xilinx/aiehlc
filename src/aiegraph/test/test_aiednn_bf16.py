"""Float matmul on the GEMM ELF's bf16 mode (aie_modes_bf16.h, aiednn_rt_bf16.h): bf16 x bf16
(bf16blk:) and f32 x f32 as a bf16x3 split (bf16x3:), through ctypes on the x86 build (the kernel's
x86 reference, f32 accumulation in K order, stands in for the AIE)."""
from __future__ import annotations

import ctypes as C
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, md  # noqa: E402

RNG = np.random.default_rng(11)
BLK = "aie:bf16_f32"


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_bf16"))


def bf16(shape, scale=2.0):
    """(raw bf16 bits as uint16, the same values as float64)"""
    t = torch.from_numpy(RNG.uniform(-scale, scale, shape).astype(np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16), t.double().numpy()


def mm_call(lib, a, w, sa, sw, transposed, dt, oshape, bias=None):
    bufs = {ARG["src"]: a, ARG["weights"]: w, ARG["dst"]: np.zeros(oshape, np.float32)}
    bmd = None
    if bias is not None:
        bufs[ARG["bias"]] = bias
        bmd = C.byref(md(bias.shape, "f32"))
    return lib.call("aiednn_matmul_forward_create",
                    [C.byref(md(sa, dt)), C.byref(md(sw, dt)), bmd, C.byref(md(oshape, "f32")), C.c_int(transposed)],
                    bufs, ARG["dst"])


@pytest.mark.parametrize("sa,sw,transposed", [((37, 300), (300, 70), 0),        # partial blocks, K chained
                                              ((300, 64), (64, 129), 0),        # two N blocks, two M blocks
                                              ((2, 5, 600), (2, 33, 600), 1)])  # batched, [N][K] weights
def test_bf16_matmul_on_aie(lib, sa, sw, transposed):
    a, ad = bf16(sa)
    w, wd = bf16(sw)
    wm = np.swapaxes(wd, -1, -2) if transposed else wd
    want = np.matmul(ad, wm)
    scale = np.matmul(np.abs(ad), np.abs(wm))                  # sum |a b|: the f32 sum's error scale
    got, impl = mm_call(lib, a, w, sa, sw, transposed, "bf16", want.shape)
    assert impl == "bf16blk:" + BLK
    K = sa[-1]
    assert np.all(np.abs(got - want) <= 1e-5 * np.sqrt(K) * (scale + 1e-30)), np.max(np.abs(got - want) / scale)


@pytest.mark.parametrize("sa,sw,transposed", [((20, 1000), (1000, 50), 0), ((3, 7, 130), (3, 40, 130), 1)])
def test_f32_matmul_as_bf16x3(lib, sa, sw, transposed):
    """f32 = hi + mid + lo (bf16 each): the six significant products in one K-concatenated bf16 GEMM --
    near f32 accuracy (a plain bf16 GEMM would be ~1e-3 off)"""
    a = RNG.uniform(-2, 2, sa).astype(np.float32)
    w = RNG.uniform(-2, 2, sw).astype(np.float32)
    wm = np.swapaxes(w, -1, -2) if transposed else w
    want = np.matmul(a.astype(np.float64), wm.astype(np.float64))
    scale = np.matmul(np.abs(a.astype(np.float64)), np.abs(wm.astype(np.float64)))
    bias = RNG.uniform(-1, 1, want.shape[-1]).astype(np.float32).reshape((1,) * (len(want.shape) - 1) + (-1,))
    got, impl = mm_call(lib, a, w, sa, sw, transposed, "f32", want.shape, bias)
    assert impl == "bf16x3:" + BLK
    err = np.abs(got - (want + bias)) / (scale + 1e-30)
    assert err.max() < 2e-6, err.max()


def test_mixed_and_zero_points_stay_off(lib):
    """bf16 x f32 or a zero point is not a bf16x3 / bf16blk problem"""
    a, _ = bf16((8, 16))
    w = RNG.uniform(-1, 1, (16, 8)).astype(np.float32)
    lib.so.aiednn_set_verbose(0)
    prim, st = lib.create("aiednn_matmul_forward_create", C.byref(md((8, 16), "bf16")), C.byref(md((16, 8), "f32")),
                          None, C.byref(md((8, 8), "f32")), C.c_int(0))
    if st == 0:
        name = lib.so.aiednn_primitive_impl_info(prim).decode()
        assert not name.startswith(("bf16blk:", "bf16x3:")), name


@pytest.mark.parametrize("sa,sw,transposed", [((1, 128), (128, 512), 0),            # a recurrent step: M = 1
                                              ((2, 3, 130), (2, 40, 130), 1)])     # batched [N][K] weights
def test_f32_constant_weights_cached(lib, sa, sw, transposed):
    """set_constant_weights: the split W' is built once per (pointer, shape) and reused; every execute with
    a new src stays right (the A side is re-split each call)"""
    w = RNG.uniform(-2, 2, sw).astype(np.float32)
    oshape = tuple(sa[:-1]) + (sw[-2] if transposed else sw[-1],)
    prim, st = lib.create("aiednn_matmul_forward_create", C.byref(md(sa, "f32")), C.byref(md(sw, "f32")), None,
                          C.byref(md(oshape, "f32")), C.c_int(transposed))
    assert st == 0
    assert lib.so.aiednn_primitive_set_constant_weights(prim, 1) == 0
    assert lib.impl(prim) == "bf16x3:" + BLK
    wm = np.swapaxes(w, -1, -2) if transposed else w
    for _ in range(3):
        a = RNG.uniform(-2, 2, sa).astype(np.float32)
        got = np.zeros(oshape, np.float32)
        assert lib.execute(prim, {ARG["src"]: a, ARG["weights"]: w, ARG["dst"]: got}) == 0
        want = np.matmul(a.astype(np.float64), wm.astype(np.float64))
        scale = np.matmul(np.abs(a.astype(np.float64)), np.abs(wm.astype(np.float64)))
        assert (np.abs(got - want) / (scale + 1e-30)).max() < 2e-6
    lib.destroy(prim)
