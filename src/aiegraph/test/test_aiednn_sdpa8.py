# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""gemm+sdpa8: attention over s8 Q / K / V with per-tensor scales as two int8 GEMMs per head."""
from __future__ import annotations

import ctypes as C
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, attr, md  # noqa: E402

RNG = np.random.default_rng(23)
QKV_BSHD, DST_BSHD, INT8_ATTN = 2, 4, 4


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_sdpa8"))


def run(lib, q8, k8, v8, scales, causal, ddt, dscale, flags):
    B, S, H, D = q8.shape
    Hk, Skv, Dv = k8.shape[2], k8.shape[1], v8.shape[3]
    lg = lambda S_, HH, W: (B, S_, HH, W)   # BSHD flags: descriptors are the physical [B, S, H, D]
    a, keep = attr(lib.ALG, dst_scale=dscale if ddt != "f32" else 0.0)
    a.src_scales[0], a.src_scales[1], a.src_scales_23[0] = scales
    a.flags = flags
    dst = np.zeros((B, S, H, Dv), {"s8": np.int8, "f32": np.float32}[ddt])
    args = [C.byref(md(lg(S, H, D), "s8")), C.byref(md(lg(Skv, Hk, D), "s8")), C.byref(md(lg(Skv, Hk, Dv), "s8")),
            None, C.byref(md(lg(S, H, Dv), ddt)), C.c_float(1.0 / math.sqrt(D)),
            C.c_int(int(causal) | QKV_BSHD | DST_BSHD), C.byref(a)]
    got, impl = lib.call("aiednn_sdpa_forward_create_attr", args,
                         {ARG["src"]: q8, ARG["src_1"]: k8, ARG["src_2"]: v8, ARG["dst"]: dst}, ARG["dst"])
    return got, impl


def reference(q8, k8, v8, scales, causal):
    sq, sk, sv = scales
    t = lambda x, s: torch.from_numpy(x.astype(np.float64) * s).permute(0, 2, 1, 3)   # [B, H, S, D]
    q, k, v = t(q8, sq), t(k8, sk), t(v8, sv)
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=1.0 / math.sqrt(q.shape[-1]))
    return y.permute(0, 2, 1, 3).numpy()   # [B, S, H, Dv]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("H,Hk", [(4, 4), (4, 2)])
def test_int8_attention_s8_dst(lib, causal, H, Hk):
    B, S, D = 1, 40, 32
    q8 = RNG.integers(-127, 128, (B, S, H, D)).astype(np.int8)
    k8 = RNG.integers(-127, 128, (B, S, Hk, D)).astype(np.int8)
    v8 = RNG.integers(-127, 128, (B, S, Hk, D)).astype(np.int8)
    scales = (0.02, 0.025, 0.03)
    y = reference(q8, k8, v8, scales, causal)
    ds = float(np.abs(y).max()) / 127
    got, impl = run(lib, q8, k8, v8, scales, causal, "s8", ds, INT8_ATTN)
    assert impl.startswith("gemm+sdpa8"), impl
    want = np.clip(np.rint(y / ds), -128, 127)
    d = np.abs(got.astype(np.int32) - want)
    assert d.max() <= 2 and (d == 0).mean() > 0.7, (d.max(), (d == 0).mean())   # 8-bit softmax weights: ~1 code


def test_int8_attention_f32_dst(lib):
    B, S, H, D = 1, 33, 4, 16
    q8, k8, v8 = (RNG.integers(-127, 128, (B, S, H, D)).astype(np.int8) for _ in range(3))
    scales = (0.03, 0.03, 0.05)
    got, impl = run(lib, q8, k8, v8, scales, False, "f32", 0.0, INT8_ATTN)
    assert impl.startswith("gemm+sdpa8"), impl
    y = reference(q8, k8, v8, scales, False)
    assert np.abs(got - y).max() <= 0.01 * np.abs(y).max(), float(np.abs(got - y).max())


def test_without_the_flag_not_sdpa8(lib):
    B, S, H, D = 1, 16, 2, 16
    q8, k8, v8 = (RNG.integers(-127, 128, (B, S, H, D)).astype(np.int8) for _ in range(3))
    _, impl = run(lib, q8, k8, v8, (0.02, 0.02, 0.02), False, "f32", 0.0, 0)
    assert not impl.startswith("gemm+sdpa8"), impl
