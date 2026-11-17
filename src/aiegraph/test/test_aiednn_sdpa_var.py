###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Variable-length and paged-KV attention (aiednn_sdpa_varlen_create) against a numpy oracle per sequence.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_sdpa_var.py
"""

from __future__ import annotations

import ctypes as C
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, f, md  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
SEQ_Q, SEQ_KV, PAGE_TABLE = 40, 41, 42


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_sdpa_var"))


def oracle(q, kseq, vseq, nq, scale, causal):
    """one sequence, one head: q [nq, D], k [nk, D], v [nk, Dv] (float64)"""
    nk = kseq.shape[0]
    if nk == 0:
        return np.zeros((nq, vseq.shape[1]))
    s = q @ kseq.T * scale
    if causal:
        s = np.where(np.arange(nk)[None, :] > np.arange(nq)[:, None] + (nk - nq), -np.inf, s)
    m = s.max(1, keepdims=True)
    e = np.where(np.isinf(m), 0, np.exp(s - np.where(np.isinf(m), 0, m)))
    z = e.sum(1, keepdims=True)
    return np.where(z > 0, e / np.where(z > 0, z, 1), 0) @ vseq


def run(lib, q, k, v, seq_q, seq_kv, pt, Dv, scale, causal, path):
    B, H, Sq, _ = q.shape
    dst = np.zeros((B, H, Sq, Dv), np.float32)
    m = lambda a, dt="f32": C.byref(md(a.shape, dt))
    bufs = {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v, ARG["dst"]: dst, SEQ_KV: seq_kv}
    if seq_q is not None:
        bufs[SEQ_Q] = seq_q
    if pt is not None:
        bufs[PAGE_TABLE] = pt
    lib.set_filter(None if path.startswith("gemm+") else path)   # the filter would also refuse its inner GEMM
    try:
        got, impl = lib.call("aiednn_sdpa_varlen_create",
                             [m(q), m(k), m(v), m(dst), m(seq_q, "s32") if seq_q is not None else None, m(seq_kv, "s32"),
                              m(pt, "s32") if pt is not None else None, f(scale), C.c_int(int(causal))],
                             bufs, ARG["dst"])
    finally:
        lib.set_filter(None)
    assert impl.startswith(path), impl
    return got


@pytest.mark.parametrize("path", ["gemm+sdpavar", "ref:"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("Hk", [4, 2, 1], ids=["mha", "gqa", "mqa"])
def test_varlen_prefill(lib, path, causal, Hk):
    rng = np.random.default_rng(5)
    B, H, Sq, Skv, D, Dv = 3, 4, 7, 9, 8, 6
    q = rng.normal(0, 1, (B, H, Sq, D)).astype(np.float32)
    k = rng.normal(0, 1, (B, Hk, Skv, D)).astype(np.float32)
    v = rng.normal(0, 1, (B, Hk, Skv, Dv)).astype(np.float32)
    seq_q = np.array([7, 3, 5], np.int32)
    seq_kv = np.array([9, 4, 0], np.int32)      # the last sequence has no keys: zeros
    scale = 1 / np.sqrt(D)
    got = run(lib, q, k, v, seq_q, seq_kv, None, Dv, scale, causal, path)
    for b in range(B):
        for h in range(H):
            kh, nq, nk = h // (H // Hk), seq_q[b], seq_kv[b]
            want = oracle(q[b, h, :nq].astype(np.float64), k[b, kh, :nk].astype(np.float64),
                          v[b, kh, :nk].astype(np.float64), nq, scale, causal)
            np.testing.assert_allclose(got[b, h, :nq], want, rtol=1e-4, atol=1e-5)
            assert not got[b, h, nq:].any()


@pytest.mark.parametrize("path", ["gemm+sdpavar", "ref:"])
@pytest.mark.parametrize("Sq", [1, 4], ids=["decode", "chunked_prefill"])
def test_paged_kv_cache(lib, path, Sq):
    """K / V in a shared pool of pages, each sequence reading its own pages through the page table"""
    rng = np.random.default_rng(6)
    B, H, Hk, D, Dv, page, P = 3, 4, 2, 8, 8, 4, 10
    q = rng.normal(0, 1, (B, H, Sq, D)).astype(np.float32)
    kc = rng.normal(0, 1, (P, Hk, page, D)).astype(np.float32)
    vc = rng.normal(0, 1, (P, Hk, page, Dv)).astype(np.float32)
    pt = np.array([[3, 7, 1], [0, 9, 0], [5, 2, 8]], np.int32)   # unused trailing entries may be anything valid
    seq_kv = np.array([11, 5, 12], np.int32)
    got = run(lib, q, kc, vc, None, seq_kv, pt, Dv, 0.3, True, path)
    for b in range(B):
        nk = seq_kv[b]
        rows = [(pt[b, j // page], j % page) for j in range(nk)]
        for h in range(H):
            kh = h // (H // Hk)
            ks = np.stack([kc[pg, kh, r] for pg, r in rows]).astype(np.float64)
            vs = np.stack([vc[pg, kh, r] for pg, r in rows]).astype(np.float64)
            want = oracle(q[b, h].astype(np.float64), ks, vs, Sq, 0.3, True)
            np.testing.assert_allclose(got[b, h], want, rtol=1e-4, atol=1e-5)


def test_bad_page_index_is_an_error(lib):
    q = np.zeros((1, 1, 1, 4), np.float32)
    kc = np.zeros((2, 1, 4, 4), np.float32)
    pt = np.array([[0, 5]], np.int32)                              # page 5 of 2
    m = lambda a, dt="f32": C.byref(md(a.shape, dt))
    seq_kv = np.array([6], np.int32)
    dst = np.zeros((1, 1, 1, 4), np.float32)
    prim, st = lib.create("aiednn_sdpa_varlen_create", m(q), m(kc), m(kc), m(dst), None, m(seq_kv, "s32"), m(pt, "s32"),
                          f(1.0), C.c_int(0))
    assert st == 0
    st = lib.execute(prim, {ARG["src"]: q, ARG["src_1"]: kc, ARG["src_2"]: kc, ARG["dst"]: dst, SEQ_KV: seq_kv,
                            PAGE_TABLE: pt})
    lib.destroy(prim)
    assert st != 0


def test_fixed_length_sdpa_unchanged(lib):
    """the varlen impl never takes a plain sdpa"""
    q = np.zeros((1, 2, 3, 4), np.float32)
    m = lambda a: C.byref(md(a.shape, "f32"))
    prim, st = lib.create("aiednn_sdpa_forward_create", m(q), m(q), m(q), None, m(q), f(1.0), C.c_int(0))
    assert st == 0 and "sdpavar" not in lib.impl(prim)
    lib.destroy(prim)


@pytest.mark.parametrize("path", ["gemm+sdpa", "ref:"])
def test_scale_bound_at_execute(lib, path):
    """AIEDNN_ARG_ATTR_SCALE (f32 [1]) overrides the create-time scale: a framework graph's scale is a constant
    tensor whose value the graph does not know when it partitions"""
    rng = np.random.default_rng(8)
    q, k, v = (rng.normal(0, 1, (1, 2, 5, 8)).astype(np.float32) for _ in range(3))
    m = lambda a: C.byref(md(a.shape, "f32"))
    outs = []
    for create_scale, bound in ((0.125, None), (1.0, 0.125)):
        dst = np.zeros_like(q)
        lib.set_filter(None if path.startswith("gemm+") else path)
        try:
            prim, st = lib.create("aiednn_sdpa_forward_create", m(q), m(k), m(v), None, m(dst), f(create_scale), C.c_int(0))
            assert st == 0 and lib.impl(prim).startswith(path), lib.impl(prim)
            bufs = {ARG["src"]: q, ARG["src_1"]: k, ARG["src_2"]: v, ARG["dst"]: dst}
            if bound is not None:
                bufs[45] = np.array([bound], np.float32)
            assert lib.execute(prim, bufs) == 0
            lib.destroy(prim)
        finally:
            lib.set_filter(None)
        outs.append(dst)
    np.testing.assert_array_equal(outs[1], outs[0])


@pytest.mark.parametrize("path", ["gemm+sdpavar", "ref:"])
def test_paged_int8_kv_cache(lib, path):
    """decode against a compressed (s8) paged K / V cache: equal to the f32 run on the dequantized cache"""
    rng = np.random.default_rng(9)
    B, H, Hk, D, page, P = 2, 4, 2, 8, 4, 6
    q = rng.normal(0, 1, (B, H, 1, D)).astype(np.float32)
    kq = rng.integers(-128, 128, (P, Hk, page, D)).astype(np.int8)
    vq = rng.integers(-128, 128, (P, Hk, page, D)).astype(np.int8)
    ks, kz, vs, vz = 0.02, 3, 0.03, -2
    pt = np.array([[1, 4, 0], [5, 2, 3]], np.int32)
    seq_kv = np.array([9, 12], np.int32)
    deq = lambda x, s, z: ((x.astype(np.float64) - z) * np.float32(s)).astype(np.float32)
    want = run(lib, q, deq(kq, ks, kz), deq(vq, vs, vz), None, seq_kv, pt, D, 0.35, True, path)
    dst = np.zeros((B, H, 1, D), np.float32)
    m = lambda a, dt="f32": C.byref(md(a.shape, dt))
    lib.set_filter(None if path.startswith("gemm+") else path)
    try:
        got, impl = lib.call("aiednn_sdpa_varlen_create_kvq",
                             [m(q), m(kq, "s8"), m(vq, "s8"), m(dst), None, m(seq_kv, "s32"), m(pt, "s32"), f(0.35),
                              C.c_int(1), (C.c_float * 2)(ks, vs), (C.c_int32 * 2)(kz, vz)],
                             {ARG["src"]: q, ARG["src_1"]: kq, ARG["src_2"]: vq, ARG["dst"]: dst, SEQ_KV: seq_kv,
                              PAGE_TABLE: pt}, ARG["dst"])
    finally:
        lib.set_filter(None)
    np.testing.assert_array_equal(got, want)
