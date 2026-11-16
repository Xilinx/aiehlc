# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""aiednn linear attention (ref: and gemm+linattn) against a float64 recurrence."""
from __future__ import annotations

import ctypes as C
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, md  # noqa: E402

GATE, BETA, STATE, STATE_OUT = 52, 53, 54, 55
PER_CHANNEL, NO_DELTA = 1, 2
RNG = np.random.default_rng(11)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_linattn"))


def recurrence(q, k, v, g, beta, h0, scale, per_ch, delta):
    """h <- Diag(e^g) h; u = beta (v - h^T k) (or v); h += k u^T; o = scale h^T q -- per (b, h), float64"""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    o = np.zeros((B, H, T, Dv))
    hs = np.zeros((B, H, Dk, Dv))
    for b in range(B):
        for hh in range(H):
            h = np.zeros((Dk, Dv)) if h0 is None else h0[b, hh].astype(np.float64)
            for t in range(T):
                if g is not None:
                    a = np.exp(g[b, hh, t].astype(np.float64))
                    h = h * (a[:, None] if per_ch else a)
                kt = k[b, hh, t].astype(np.float64)
                u = beta[b, hh, t] * (v[b, hh, t] - h.T @ kt) if delta else v[b, hh, t].astype(np.float64)
                h = h + np.outer(kt, u)
                o[b, hh, t] = scale * (h.T @ q[b, hh, t])
            hs[b, hh] = h
    return o, hs


def make(B, H, T, Dk, Dv, per_ch, delta, gated, state):
    q = RNG.normal(size=(B, H, T, Dk)).astype(np.float32)
    k = RNG.normal(size=(B, H, T, Dk))
    k = (k / np.linalg.norm(k, axis=-1, keepdims=True)).astype(np.float32)   # models L2-normalize keys
    v = RNG.normal(size=(B, H, T, Dv)).astype(np.float32)
    g = None
    if gated:
        g = (RNG.uniform(-0.3, 0.0, (B, H, T, Dk) if per_ch else (B, H, T))).astype(np.float32)
    beta = RNG.uniform(0.1, 0.9, (B, H, T)).astype(np.float32) if delta else None
    h0 = RNG.normal(size=(B, H, Dk, Dv)).astype(np.float32) * 0.3 if state else None
    return q, k, v, g, beta, h0


def run(lib, q, k, v, g, beta, h0, scale, flags, state_out, filt=None):
    B, H, T, _ = q.shape
    Dk, Dv = q.shape[-1], v.shape[-1]
    dst = np.zeros((B, H, T, Dv), np.float32)
    s1 = np.zeros((B, H, Dk, Dv), np.float32) if state_out else None
    keep = [md(q.shape, "f32"), md(k.shape, "f32"), md(v.shape, "f32"), md(g.shape, "f32") if g is not None else None,
            md(beta.shape, "f32") if beta is not None else None, md(h0.shape, "f32") if h0 is not None else None,
            md(dst.shape, "f32"), md(s1.shape, "f32") if s1 is not None else None]
    args = [C.byref(m) if m is not None else None for m in keep]
    lib.set_filter(filt)
    try:
        prim, st = lib.create("aiednn_linear_attention_create", *args, C.c_float(scale), C.c_uint(flags))
        assert st == 0, lib.status(st)
        impl = lib.impl(prim)
        bufs = {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v, ARG["dst"]: dst}
        for aid, arr in ((GATE, g), (BETA, beta), (STATE, h0), (STATE_OUT, s1)):
            if arr is not None:
                bufs[aid] = arr
        assert lib.execute(prim, bufs) == 0
        lib.destroy(prim)
    finally:
        lib.set_filter(None)
    return impl, dst, s1


CASES = {"gated_deltanet": (False, True, True), "kda": (True, True, True), "gla": (False, False, True),
         "deltanet": (False, True, False)}


@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("name", list(CASES))
def test_linear_attention(lib, name, state):
    per_ch, delta, gated = CASES[name]
    T = 37 if per_ch else 70   # not a multiple of the chunk (16 / 64)
    q, k, v, g, beta, h0 = make(1, 2, T, 32, 24, per_ch, delta, gated, state)
    flags = (PER_CHANNEL if per_ch else 0) | (0 if delta else NO_DELTA)
    want, hs = recurrence(q, k, v, g, beta, h0, 0.17, per_ch, delta)
    impl, ref_o, ref_s = run(lib, q, k, v, g, beta, h0, 0.17, flags, state, "ref:")
    assert impl == "ref:linear_attention"
    np.testing.assert_allclose(ref_o, want, rtol=1e-5, atol=1e-6)
    impl, got, got_s = run(lib, q, k, v, g, beta, h0, 0.17, flags, state)
    assert impl.startswith("gemm+linattn"), impl
    tol = 2e-4 * np.abs(want).max()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=tol)
    if state:
        assert ref_s is not None and got_s is not None
        np.testing.assert_allclose(ref_s, hs, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got_s, hs, rtol=1e-4, atol=2e-4 * np.abs(hs).max())


def test_steep_per_channel_gates_take_host_steps(lib):
    """Per-channel gates too steep for float in a chunk run as host steps, same results."""
    q, k, v, _, beta, h0 = make(1, 1, 40, 16, 8, True, True, True, True)
    g = np.full((1, 1, 40, 16), -8.0, np.float32)
    g[..., ::2] = -0.05
    want, hs = recurrence(q, k, v, g, beta, h0, 1.0, True, True)
    impl, got, got_s = run(lib, q, k, v, g, beta, h0, 1.0, PER_CHANNEL, True)
    assert impl.startswith("gemm+linattn"), impl
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=2e-4 * np.abs(want).max())
    assert got_s is not None
    np.testing.assert_allclose(got_s, hs, rtol=1e-4, atol=2e-4 * np.abs(hs).max())


def test_refusals(lib):
    q = md((1, 2, 8, 16), "f32")
    v = md((1, 2, 8, 8), "f32")
    beta = md((1, 2, 8), "f32")
    _, st = lib.create("aiednn_linear_attention_create", C.byref(q), C.byref(q), C.byref(v), None, None, None,
                       C.byref(v), None, C.c_float(1.0), C.c_uint(0))
    assert st != 0   # the delta rule needs beta
    _, st = lib.create("aiednn_linear_attention_create", C.byref(q), C.byref(q), C.byref(v), None, C.byref(beta), None,
                       C.byref(v), None, C.c_float(1.0), C.c_uint(NO_DELTA))
    assert st != 0   # no delta: no beta
    g = md((1, 2, 8), "f32")
    _, st = lib.create("aiednn_linear_attention_create", C.byref(q), C.byref(q), C.byref(v), C.byref(g), C.byref(beta),
                       None, C.byref(v), None, C.c_float(1.0), C.c_uint(PER_CHANNEL))
    assert st != 0   # a per-channel gate is [B, H, T, Dk]
