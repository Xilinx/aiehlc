###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""1-D and 3-D convolution, deconvolution and pooling (aiednn_*_create_nd) against PyTorch.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_nd.py

1-D runs as 2-D (H = 1), so an int8 1-D conv takes the AIE im2col composite; 3-D runs on ref:.
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
from aiegraph.dnn_ctypes import ARG, DT, AiednnLib, conv_nd_params, md, pool_nd_params  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(5)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_nd"))


def arr(shape, dt):
    if dt == "f32":
        return RNG.uniform(-2, 2, shape).astype(np.float32)
    return RNG.integers(-128, 128, shape).astype(np.int8)


def T(x):
    return torch.from_numpy(np.asarray(x, np.float64))


def out_len(n, k, s, pl, pr, d, deconv):
    ext = (k - 1) * (d + 1) + 1
    return (n - 1) * s - pl - pr + ext if deconv else (n + pl + pr - ext) // s + 1


CASES = [  # spatial, kernel, stride, pad_l, pad_r, dil, groups
    ((17,), (3,), (2,), (1,), (0,), (1,), 2),
    ((9,), (5,), (1,), (2,), (2,), (0,), 1),
    ((5, 6, 7), (3, 2, 3), (1, 2, 1), (1, 0, 1), (1, 1, 0), (0, 0, 1), 2),
    ((4, 5, 5), (1, 3, 3), (2, 1, 1), (0, 1, 1), (0, 1, 1), (0, 0, 0), 1),
]


def torch_conv(x, w, b, s, pl, pr, d, g, deconv):
    nd = len(s)
    pad = []
    for i in reversed(range(nd)):
        pad += [pl[i], pr[i]]
    dil = tuple(v + 1 for v in d)
    if not deconv:
        fn = (F.conv1d, F.conv2d, F.conv3d)[nd - 1]
        return fn(F.pad(T(x), pad), T(w), T(b), stride=s, dilation=dil, groups=g).numpy()
    # transposed conv with asymmetric padding: full output (padding 0), then crop
    fn = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[nd - 1]
    wt = T(w)  # aiednn [O, C/g, k...] -> torch [C, O/g, k...]
    O, Cg = wt.shape[0], wt.shape[1]
    wt = wt.reshape(g, O // g, Cg, *wt.shape[2:]).transpose(1, 2).reshape(g * Cg, O // g, *wt.shape[2:])
    y = fn(T(x), wt, T(b), stride=s, dilation=dil, groups=g)
    sl = [slice(None), slice(None)] + [slice(pl[i], y.shape[2 + i] - pr[i]) for i in range(nd)]
    return y[tuple(sl)].numpy()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("deconv", [0, 1])
@pytest.mark.parametrize("dt", ["f32", "s8"])
def test_conv_nd(lib, case, deconv, dt):
    sp, k, s, pl, pr, d, g = case
    N, Cc, O = 2, 4, 6
    x, w = arr((N, Cc, *sp), dt), arr((O, Cc // g, *k), dt)
    b = RNG.integers(-100, 100, O).astype(np.int32) if dt == "s8" else arr((O,), "f32")
    osp = tuple(out_len(sp[i], k[i], s[i], pl[i], pr[i], d[i], deconv) for i in range(len(sp)))
    odt = "s32" if dt == "s8" else "f32"
    dst = np.zeros((N, O, *osp), DT[odt][1])
    fn = "aiednn_deconvolution_forward_create_nd" if deconv else "aiednn_convolution_forward_create_nd"
    got, impl = lib.call(fn, [C.byref(md(x.shape, dt)), C.byref(md(w.shape, dt)), C.byref(md(b.shape, odt)),
                              C.byref(md(dst.shape, odt)), C.byref(conv_nd_params(s, pl, pr, d, g)), None],
                         {ARG["src"]: x, ARG["weights"]: w, ARG["bias"]: b, ARG["dst"]: dst}, ARG["dst"])
    want = torch_conv(x, w, b, s, pl, pr, d, g, deconv)
    if dt == "f32":
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    else:
        np.testing.assert_array_equal(got, want.astype(np.int64))
    if len(sp) == 3:   # s8 3-D conv: gemm+ with its GEMMs on the AIE block; f32 needs a floating GEMM mode
        # 3-D: gemm+ composites with their GEMMs on the AIE (s8 on the GEMM block, f32 on bf16x3)
        assert impl.startswith("gemm+deconv+" if deconv else "gemm+conv+") and "aie:" in impl, impl
    elif dt == "s8" and not deconv and g == 1:   # narrow groups stay on ref: (RT_MIN_GROUP_N)
        assert impl.startswith("im2col")   # 1-D int8 conv: the AIE composite through the 2-D path


@pytest.mark.parametrize("alg", ["pooling_max", "pooling_avg_include_padding", "pooling_avg_exclude_padding"])
@pytest.mark.parametrize("sp,k,s,p", [((15,), (3,), (2,), (1,)), ((6, 7, 8), (2, 3, 3), (2, 2, 1), (0, 1, 1))])
def test_pool_nd(lib, alg, sp, k, s, p):
    x = arr((2, 3, *sp), "f32")
    osp = tuple((sp[i] + 2 * p[i] - k[i]) // s[i] + 1 for i in range(len(sp)))
    got, impl = lib.call("aiednn_pooling_forward_create_nd",
                         [C.c_int(lib.ALG[alg]), C.byref(md(x.shape, "f32")), C.byref(md((2, 3, *osp), "f32")),
                          C.byref(pool_nd_params(k, s, p, p))],
                         {ARG["src"]: x, ARG["dst"]: np.zeros((2, 3, *osp), np.float32)}, ARG["dst"])
    nd = len(sp)
    if alg == "pooling_max":
        want = (F.max_pool1d, F.max_pool2d, F.max_pool3d)[nd - 1](T(x), k, s, p)
    else:
        want = (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)[nd - 1](T(x), k, s, p,
                                                               count_include_pad=alg == "pooling_avg_include_padding")
    np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-6)
