###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""FVEC mode (aie_modes_fvec.h / aiednn_rt_fvec.h): f32 eltwise, binary, softmax and layer / RMS norm
on the AIE, against ref: on the same inputs within the mode's contract |d| <= 1e-5 + 1e-5 |ref|.

The library build's GEMM-ELF entries run their x86 references here (gemm_uk_ref.c: aie_modes_fvec.h's
plain C), so this checks the host side -- items, padding, broadcasts, gamma / beta, the unpacking --
and test_fvec_core_vector_code the core's vector code through the AIE-API emulation.

    python3 -m pytest -q src/aiegraph/test/test_aiednn_fvec.py
"""

from __future__ import annotations

import ctypes as C
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, AiednnLib, md  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HAS_FVEC = "AIEK_SPEC(fvec_blk_f32" in (REPO / "src" / "aietensorop" / "gemm" / "gemm_uk.cc").read_text()
pytestmark = pytest.mark.skipif(not shutil.which("g++"), reason="needs g++")
RNG = np.random.default_rng(51)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if not HAS_FVEC:
        pytest.skip("gemm_uk.cc does not carry the fvec_blk_f32 entry yet")
    return AiednnLib.build(tmp_path_factory.mktemp("aiednn_fvec"))


def m(shape, dt="f32"):
    return C.byref(md(shape, dt))


def both(lib, fn, args, bufs, out):
    res = []
    for filt in (None, "ref:"):
        b = {k: v.copy() for k, v in bufs.items()}
        lib.set_filter(filt)
        try:
            got, impl = lib.call(fn, args, b, out)
        finally:
            lib.set_filter(None)
        res.append((impl, got))
    return res[0][0], res[0][1], res[1][1]


def contract(got, want):
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    assert np.all(d <= 1e-5 + 1e-5 * np.abs(want)), float(np.max(d / (1e-5 + 1e-5 * np.abs(want))))


ELT = [("eltwise_relu", 0.0, 0.0), ("eltwise_relu", 0.1, 0.0), ("eltwise_elu", 1.0, 0.0), ("eltwise_gelu_tanh", 0, 0),
       ("eltwise_logistic", 0, 0), ("eltwise_tanh", 0, 0), ("eltwise_swish", 1.0, 0), ("eltwise_clip", -1.0, 2.0),
       ("eltwise_linear", 0.5, -1.0), ("eltwise_abs", 0, 0), ("eltwise_exp", 0, 0), ("eltwise_sqrt", 0, 0),
       ("eltwise_square", 0, 0), ("eltwise_neg", 0, 0), ("eltwise_identity", 0, 0),
       # vector programs (aie_fvec_prog): one core interpreter, the functions built on the host
       ("eltwise_gelu_erf", 0, 0), ("eltwise_selu", 0, 0), ("eltwise_soft_relu", 1.0, 0), ("eltwise_mish", 0, 0),
       ("eltwise_hardswish", 1 / 6, 0.5), ("eltwise_hardsigmoid", 1 / 6, 0.5), ("eltwise_erf", 0, 0),
       ("eltwise_sign", 0, 0), ("eltwise_softsign", 0, 0), ("eltwise_log_sigmoid", 0, 0), ("eltwise_log", 0, 0),
       ("eltwise_reciprocal", 0, 0), ("eltwise_rsqrt", 0, 0),
       # exact-valued programs: the fp32 round trick, SELP steps, integer powers
       ("eltwise_round", 0, 0), ("eltwise_floor", 0, 0), ("eltwise_ceil", 0, 0), ("eltwise_logical_not", 0, 0),
       ("eltwise_pow", 1.5, 3.0), ("eltwise_pow", 2.0, -1.0)]
POSITIVE = {"eltwise_sqrt", "eltwise_log", "eltwise_reciprocal", "eltwise_rsqrt"}


@pytest.mark.parametrize("alg,a,b", ELT)
@pytest.mark.parametrize("shape", [(3, 7, 11), (2, 16, 32, 32)])
def test_eltwise(lib, alg, a, b, shape):
    x = RNG.uniform(0.05 if alg in POSITIVE else -6, 6, shape).astype(np.float32)
    impl, got, want = both(lib, "aiednn_eltwise_forward_create", [C.c_int(lib.ALG[alg]), C.c_float(a), C.c_float(b),
                                                                  m(shape), m(shape)],
                           {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


@pytest.mark.parametrize("alg", ["binary_add", "binary_sub", "binary_mul", "binary_div", "binary_max", "binary_min",
                                 "binary_eq", "binary_ne", "binary_gt", "binary_ge", "binary_lt", "binary_le",
                                 "binary_logical_and", "binary_logical_or", "binary_logical_xor", "binary_add_square"])
@pytest.mark.parametrize("s1", [(4, 8, 33), (1, 8, 1), (1, 1, 33)])
def test_binary(lib, alg, s1):
    s0 = (4, 8, 33)
    x = RNG.uniform(-4, 4, s0).astype(np.float32)
    y = (RNG.uniform(0.5, 4, s1) * RNG.choice([-1, 1], s1)).astype(np.float32)
    if alg != "binary_div":   # equal pairs and zeros for the compares / logical ops
        x[:, :, :3] = np.broadcast_to(y, s0)[:, :, :3]
        x[:, :, 3] = 0
    impl, got, want = both(lib, "aiednn_binary_create", [C.c_int(lib.ALG[alg]), m(s0), m(s1), None, m(s0)],
                           {ARG["src"]: x, ARG["src_1"]: y, ARG["dst"]: np.zeros(s0, np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


@pytest.mark.parametrize("alg", ["softmax_accurate", "softmax_log"])
@pytest.mark.parametrize("shape", [(5, 10), (3, 4, 64), (2, 1000)])
def test_softmax(lib, alg, shape):
    x = RNG.uniform(-8, 8, shape).astype(np.float32)
    impl, got, want = both(lib, "aiednn_softmax_forward_create", [C.c_int(lib.ALG[alg]), C.c_int(len(shape) - 1),
                                                                  m(shape), m(shape)],
                           {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


@pytest.mark.parametrize("flags", [0, 1 | 2, 8, 8 | 1])   # USE_SCALE | USE_SHIFT, RMS_NORM
@pytest.mark.parametrize("shape", [(6, 64), (2, 3, 384)])
def test_layer_norm(lib, flags, shape):
    C_ = shape[-1]
    x = (RNG.uniform(-3, 3, shape) + 0.5).astype(np.float32)
    bufs = {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32)}
    if flags & 1:
        bufs[ARG["scale"]] = RNG.uniform(0.5, 1.5, C_).astype(np.float32)
    if flags & 2:
        bufs[ARG["shift"]] = RNG.uniform(-1, 1, C_).astype(np.float32)
    impl, got, want = both(lib, "aiednn_layer_normalization_forward_inference_create",
                           [m(shape), m(shape), C.c_float(1e-5), C.c_uint(flags)], bufs, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


def test_unsupported_stays_on_ref(lib):
    """a norm row too short to chunk (6 < 16): ref:"""
    x = RNG.uniform(0.5, 2, (4, 6)).astype(np.float32)
    impl, _, _ = both(lib, "aiednn_layer_normalization_forward_inference_create",
                      [m(x.shape), m(x.shape), C.c_float(1e-5), C.c_uint(0)],
                      {ARG["src"]: x, ARG["dst"]: np.zeros_like(x)}, ARG["dst"])
    assert impl == "ref:layer_normalization"


def test_fvec_core_vector_code(tmp_path):
    """aie_modes_fvec.h's vector code (exp, inv / invsqrt as the AIE-ML API computes them) within the
    contract for every op, through src/mlir/runtime/aie_api_x86emu.h"""
    exe = tmp_path / "fvec"
    subprocess.run(["g++", "-std=c++20", "-O1", "-DAIE_FVEC_EMU", f"-I{REPO / 'src' / 'mlir' / 'runtime'}",
                    f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", "-o", str(exe),
                    str(REPO / "src" / "aietensorop" / "ops" / "test" / "fvec_test.cc")], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0 and "0 mismatching" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("alg", ["reduction_sum", "reduction_mean", "reduction_max", "reduction_min", "reduction_amax"])
@pytest.mark.parametrize("src,dst", [((4, 6, 7, 7), (4, 6, 1, 1)), ((3, 50), (3, 1)), ((2, 8, 33), (2, 1, 1))])
def test_reduction_rows(lib, alg, src, dst):
    x = RNG.uniform(-4, 4, src).astype(np.float32)
    impl, got, want = both(lib, "aiednn_reduction_create", [C.c_int(lib.ALG[alg]), C.c_float(0), C.c_float(0), m(src), m(dst)],
                           {ARG["src"]: x, ARG["dst"]: np.zeros(dst, np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


@pytest.mark.parametrize("alg", ["pooling_max", "pooling_avg_include_padding", "pooling_avg_exclude_padding"])
@pytest.mark.parametrize("hw", [7, 14])
def test_global_pooling(lib, alg, hw):
    from aiegraph.dnn_ctypes import PoolParams
    x = RNG.uniform(-4, 4, (2, 40, hw, hw)).astype(np.float32)
    pp = PoolParams()
    for i in range(2):
        pp.kernel[i], pp.strides[i] = hw, 1
    impl, got, want = both(lib, "aiednn_pooling_forward_create", [C.c_int(lib.ALG[alg]), m(x.shape), m((2, 40, 1, 1)), C.byref(pp)],
                           {ARG["src"]: x, ARG["dst"]: np.zeros((2, 40, 1, 1), np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


# ROWPROG: the program over rows with two per-row constants (per-channel affine, PReLU, broadcast binary)
@pytest.mark.parametrize("flags", [0, 1 | 2, 1 | 2 | 4])   # USE_SCALE | USE_SHIFT, FUSE_NORM_RELU
@pytest.mark.parametrize("shape", [(2, 32, 14, 14), (3, 24, 7, 7), (4, 64, 1, 1), (1, 8, 56, 56), (2, 12, 1031)])
def test_batch_norm(lib, flags, shape):
    C_ = shape[1]
    x = RNG.uniform(-3, 3, shape).astype(np.float32)
    bufs = {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32),
            ARG["mean"]: RNG.uniform(-0.5, 0.5, C_).astype(np.float32),
            ARG["variance"]: RNG.uniform(0.2, 2.0, C_).astype(np.float32)}
    if flags & 1:
        bufs[ARG["scale"]] = RNG.uniform(0.5, 1.5, C_).astype(np.float32)
    if flags & 2:
        bufs[ARG["shift"]] = RNG.uniform(-1, 1, C_).astype(np.float32)
    impl, got, want = both(lib, "aiednn_batch_normalization_forward_inference_create",
                           [m(shape), m(shape), C.c_float(1e-5), C.c_uint(flags)], bufs, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


@pytest.mark.parametrize("w", [(1, 16, 1, 1), (1, 1, 1, 1), (2, 16, 9, 9), (1, 16, 9, 1)])
def test_prelu(lib, w):
    s = (2, 16, 9, 9)
    x = RNG.uniform(-4, 4, s).astype(np.float32)
    wt = RNG.uniform(-0.5, 0.5, w).astype(np.float32)
    impl, got, want = both(lib, "aiednn_prelu_forward_create", [m(s), m(w), m(s)],
                           {ARG["src"]: x, ARG["weights"]: wt, ARG["dst"]: np.zeros(s, np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


@pytest.mark.parametrize("alg", ["binary_add", "binary_mul", "binary_div", "binary_max"])
@pytest.mark.parametrize("s1", [(1, 24, 1, 1), (2, 24, 1, 1), (1, 1, 1, 1)])
def test_binary_per_channel(lib, alg, s1):
    s0 = (2, 24, 28, 28)
    x = RNG.uniform(-4, 4, s0).astype(np.float32)
    y = (RNG.uniform(0.5, 4, s1) * RNG.choice([-1, 1], s1)).astype(np.float32)
    impl, got, want = both(lib, "aiednn_binary_create", [C.c_int(lib.ALG[alg]), m(s0), m(s1), None, m(s0)],
                           {ARG["src"]: x, ARG["src_1"]: y, ARG["dst"]: np.zeros(s0, np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


def test_prefer_fastest_refusals(lib):
    """aiednn_prefer_fastest: binary with a per-element operand and layer norm go to ref: (measured faster);
    a per-channel operand (ROWPROG) and the transcendental eltwise stay on FVEC; prefer_aie keeps all"""
    s0, s1 = (2, 24, 28, 28), (1, 24, 1, 1)
    x = RNG.uniform(-4, 4, s0).astype(np.float32)
    def impl_of(shape1, kind):
        y = RNG.uniform(0.5, 2, shape1).astype(np.float32)
        if kind == "binary":
            _, impl = lib.call("aiednn_binary_create", [C.c_int(lib.ALG["binary_add"]), m(s0), m(shape1), None, m(s0)],
                               {ARG["src"]: x, ARG["src_1"]: y, ARG["dst"]: np.zeros(s0, np.float32)}, ARG["dst"])
        else:
            _, impl = lib.call("aiednn_eltwise_forward_create", [C.c_int(lib.ALG["eltwise_gelu_tanh"]), C.c_float(0),
                               C.c_float(0), m(s0), m(s0)], {ARG["src"]: x, ARG["dst"]: np.zeros(s0, np.float32)}, ARG["dst"])
        return impl
    lib.so.aiednn_set_preference(1)
    try:
        assert impl_of(s0, "binary") == "ref:binary"
        assert impl_of(s1, "binary").startswith("fvec:")
        assert impl_of(s0, "eltwise").startswith("fvec:")
    finally:
        lib.so.aiednn_set_preference(0)
    assert impl_of(s0, "binary").startswith("fvec:")


# eltwise backward: dy f'(x) as FVEC programs (aie_fvec_prog_bwd), dy in AUX
DIFF = {"diff_src": 20, "diff_dst": 23}
BWD = [("eltwise_relu", 0.0, 0.0), ("eltwise_relu", 0.1, 0.0), ("eltwise_elu", 1.0, 0.0), ("eltwise_selu", 0, 0),
       ("eltwise_logistic", 0, 0), ("eltwise_tanh", 0, 0), ("eltwise_swish", 1.0, 0), ("eltwise_soft_relu", 1.0, 0),
       ("eltwise_mish", 0, 0), ("eltwise_hardswish", 1 / 6, 0.5), ("eltwise_hardsigmoid", 1 / 6, 0.5),
       ("eltwise_clip", -1.0, 2.0), ("eltwise_linear", 0.5, -1.0), ("eltwise_abs", 0, 0), ("eltwise_exp", 0, 0),
       ("eltwise_log", 0, 0), ("eltwise_sqrt", 0, 0), ("eltwise_square", 0, 0), ("eltwise_reciprocal", 0, 0),
       ("eltwise_rsqrt", 0, 0), ("eltwise_erf", 0, 0), ("eltwise_softsign", 0, 0), ("eltwise_log_sigmoid", 0, 0),
       ("eltwise_gelu_tanh", 0, 0), ("eltwise_neg", 0, 0), ("eltwise_identity", 0, 0), ("eltwise_sin", 0, 0),
       ("eltwise_cos", 0, 0), ("eltwise_tan", 0, 0), ("eltwise_gelu_erf", 0, 0)]   # tan / gelu_erf: element passes


@pytest.mark.parametrize("alg,a,b", BWD)
def test_eltwise_backward(lib, alg, a, b):
    shape = (2, 16, 33, 7)
    x = RNG.uniform(0.05 if alg in POSITIVE else -6, 6, shape).astype(np.float32)
    x.flat[:5] = [0.0, 1e-30, -1e-30, 2.0, -1.0] if alg not in POSITIVE else x.flat[:5]   # the step points
    dy = RNG.uniform(-2, 2, shape).astype(np.float32)
    impl, got, want = both(lib, "aiednn_eltwise_backward_create", [C.c_int(lib.ALG[alg]), C.c_float(a), C.c_float(b),
                                                                   m(shape), m(shape), m(shape)],
                           {DIFF["diff_src"]: np.zeros(shape, np.float32), DIFF["diff_dst"]: dy, ARG["src"]: x},
                           DIFF["diff_src"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


def test_eltwise_backward_stays_on_ref(lib):
    """pow with a non-integer exponent: no FVEC program (no NaN for x < 0 on the core)"""
    x = RNG.uniform(0.5, 3, (4, 40)).astype(np.float32)
    impl, _, _ = both(lib, "aiednn_eltwise_backward_create", [C.c_int(lib.ALG["eltwise_pow"]), C.c_float(1.0),
                                                              C.c_float(0.5), m(x.shape), m(x.shape), m(x.shape)],
                      {DIFF["diff_src"]: np.zeros_like(x), DIFF["diff_dst"]: x.copy(), ARG["src"]: x}, DIFF["diff_src"])
    assert impl == "ref:eltwise"


# training rows: ROWSUM statistics, then ROWPROG with a per-element operand (AIE_FV_F_ELEM)
def both_bufs(lib, fn, args, bufs, out):
    """both(), but every buffer after each run (training writes several outputs)"""
    res = []
    for filt in (None, "ref:"):
        b = {k: v.copy() for k, v in bufs.items()}
        lib.set_filter(filt)
        try:
            _, impl = lib.call(fn, args, b, out)
        finally:
            lib.set_filter(None)
        res.append((impl, b))
    return res[0][0], res[0][1], res[1][1]


@pytest.mark.parametrize("alg", ["softmax_accurate", "softmax_log"])
@pytest.mark.parametrize("shape", [(6, 10), (3, 4, 64), (2, 1000), (5, 33)])
def test_softmax_backward(lib, alg, shape):
    x = RNG.uniform(-4, 4, shape)
    e = np.exp(x - x.max(-1, keepdims=True))
    y = (e / e.sum(-1, keepdims=True) if alg == "softmax_accurate" else np.log(e / e.sum(-1, keepdims=True))).astype(np.float32)
    dy = RNG.uniform(-2, 2, shape).astype(np.float32)
    impl, got, want = both(lib, "aiednn_softmax_backward_create", [C.c_int(lib.ALG[alg]), C.c_int(len(shape) - 1),
                                                                   m(shape), m(shape), m(shape)],
                           {DIFF["diff_src"]: np.zeros(shape, np.float32), DIFF["diff_dst"]: dy, ARG["dst"]: y},
                           DIFF["diff_src"])
    assert impl.startswith("fvec:"), impl
    d = np.abs(got.astype(np.float64) - want)   # dx sums a row: the contract on the row's magnitudes
    scale = np.abs(y.astype(np.float64) * dy).sum(-1, keepdims=True) + np.abs(dy) + np.abs(want)
    assert np.all(d <= 1e-5 + 1e-5 * scale), float(np.max(d / (1e-5 + 1e-5 * scale)))


@pytest.mark.parametrize("flags", [0, 1 | 2])
@pytest.mark.parametrize("shape", [(2, 16, 14, 14), (4, 8, 7, 7), (3, 24, 1, 1), (1, 6, 56, 56)])
def test_batch_norm_training(lib, flags, shape):
    C_ = shape[1]
    x = (RNG.uniform(-3, 3, shape) + RNG.uniform(-1, 1, (1, C_, 1, 1))).astype(np.float32)
    bufs = {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32), ARG["mean"]: np.zeros(C_, np.float32),
            ARG["variance"]: np.zeros(C_, np.float32)}
    if flags & 1:
        bufs[ARG["scale"]] = RNG.uniform(0.5, 1.5, C_).astype(np.float32)
    if flags & 2:
        bufs[ARG["shift"]] = RNG.uniform(-1, 1, C_).astype(np.float32)
    impl, got, want = both_bufs(lib, "aiednn_batch_normalization_forward_training_create",
                                [m(shape), m(shape), C.c_float(1e-5), C.c_uint(flags)], bufs, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    for k in (ARG["dst"], ARG["mean"], ARG["variance"]):
        contract(got[k], want[k])


@pytest.mark.parametrize("train", [0, 1])
@pytest.mark.parametrize("shape,groups", [((2, 16, 14, 14), 4), ((1, 32, 7, 7), 8), ((3, 6, 33, 1), 3), ((2, 8, 1, 1), 2)])
def test_group_norm(lib, train, shape, groups):
    C_ = shape[1]
    x = (RNG.uniform(-3, 3, shape) + 0.7).astype(np.float32)
    bufs = {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32), ARG["scale"]: RNG.uniform(0.5, 1.5, C_).astype(np.float32),
            ARG["shift"]: RNG.uniform(-1, 1, C_).astype(np.float32)}
    if train:
        bufs[ARG["mean"]] = np.zeros(shape[0] * groups, np.float32)
        bufs[ARG["variance"]] = np.zeros(shape[0] * groups, np.float32)
        fn, args = "aiednn_group_normalization_forward_training_create", [C.c_int64(groups), m(shape), m(shape),
                                                                           C.c_float(1e-5), C.c_uint(3)]
    else:
        fn, args = "aiednn_group_normalization_forward_inference_create", [C.c_int64(groups), m(shape), m(shape),
                                                                            C.c_float(1e-5), C.c_uint(3)]
    impl, got, want = both_bufs(lib, fn, args, bufs, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    for k in (ARG["dst"],) + ((ARG["mean"], ARG["variance"]) if train else ()):
        contract(got[k], want[k])


@pytest.mark.parametrize("flags", [0, 1 | 2, 8, 8 | 1])
@pytest.mark.parametrize("shape", [(6, 64), (2, 3, 384)])
def test_layer_norm_training(lib, flags, shape):
    rows = int(np.prod(shape[:-1]))
    x = (RNG.uniform(-3, 3, shape) + 0.5).astype(np.float32)
    bufs = {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32), ARG["mean"]: np.zeros(rows, np.float32),
            ARG["variance"]: np.zeros(rows, np.float32)}
    if flags & 1:
        bufs[ARG["scale"]] = RNG.uniform(0.5, 1.5, shape[-1]).astype(np.float32)
    if flags & 2:
        bufs[ARG["shift"]] = RNG.uniform(-1, 1, shape[-1]).astype(np.float32)
    impl, got, want = both_bufs(lib, "aiednn_layer_normalization_forward_training_create",
                                [m(shape), m(shape), C.c_float(1e-5), C.c_uint(flags)], bufs, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    for k in (ARG["dst"], ARG["mean"], ARG["variance"]):
        contract(got[k], want[k])


@pytest.mark.parametrize("kind,groups", [("batch", 0), ("group", 4), ("group", 2)])
@pytest.mark.parametrize("flags", [0, 1 | 2])
@pytest.mark.parametrize("shape", [(2, 8, 14, 14), (3, 4, 7, 7), (2, 8, 1, 33)])
def test_norm_backward(lib, kind, groups, flags, shape):
    N, C_ = shape[0], shape[1]
    ns = C_ if kind == "batch" else N * groups
    x = (RNG.uniform(-3, 3, shape) + 0.5).astype(np.float32)
    dy = RNG.uniform(-2, 2, shape).astype(np.float32)
    bufs = {ARG["src"]: x, DIFF["diff_dst"]: dy, DIFF["diff_src"]: np.zeros(shape, np.float32),
            ARG["mean"]: RNG.uniform(-0.3, 0.8, ns).astype(np.float32),
            ARG["variance"]: RNG.uniform(0.5, 3.0, ns).astype(np.float32)}
    if flags & 1:
        bufs[ARG["scale"]] = RNG.uniform(0.5, 1.5, C_).astype(np.float32)
        bufs[24] = np.zeros(C_, np.float32)   # DIFF_SCALE
    if flags & 2:
        bufs[25] = np.zeros(C_, np.float32)   # DIFF_SHIFT
    if kind == "batch":
        fn, args = "aiednn_batch_normalization_backward_create", [m(shape), m(shape), m(shape), C.c_float(1e-5), C.c_uint(flags)]
    else:
        fn, args = "aiednn_group_normalization_backward_create", [C.c_int64(groups), m(shape), m(shape), m(shape),
                                                                  C.c_float(1e-5), C.c_uint(flags)]
    impl, got, want = both_bufs(lib, fn, args, bufs, DIFF["diff_src"])
    assert impl.startswith("fvec:"), impl
    # dx and the parameter gradients are sums that cancel: the bound scales with the summed magnitudes
    tol = 1e-5 * (np.abs(dy).mean() * 4 + 1)
    d = np.abs(got[DIFF["diff_src"]].astype(np.float64) - want[DIFF["diff_src"]])
    assert np.all(d <= tol * (1 + np.abs(want[DIFF["diff_src"]]))), float(d.max())
    for k in (24, 25):
        if k in bufs:
            np.testing.assert_allclose(got[k], want[k], rtol=1e-5, atol=1e-5 * np.abs(dy).sum() / C_)


@pytest.mark.parametrize("alg", ["softmax_accurate", "softmax_log"])
@pytest.mark.parametrize("shape,axis", [((3, 7, 5), 1), ((6, 40, 3), 0), ((2, 33, 4, 4), 1), ((4, 5, 9), 2)])
def test_softmax_any_axis(lib, alg, shape, axis):
    """forward and backward over a non-last axis: strided rows gathered into the items"""
    x = RNG.uniform(-6, 6, shape).astype(np.float32)
    impl, got, want = both(lib, "aiednn_softmax_forward_create", [C.c_int(lib.ALG[alg]), C.c_int(axis), m(shape), m(shape)],
                           {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)
    e = np.exp(x - x.max(axis, keepdims=True))
    y = e / e.sum(axis, keepdims=True)
    y = (y if alg == "softmax_accurate" else np.log(y)).astype(np.float32)
    dy = RNG.uniform(-2, 2, shape).astype(np.float32)
    impl, got, want = both(lib, "aiednn_softmax_backward_create", [C.c_int(lib.ALG[alg]), C.c_int(axis), m(shape), m(shape), m(shape)],
                           {DIFF["diff_src"]: np.zeros(shape, np.float32), DIFF["diff_dst"]: dy, ARG["dst"]: y}, DIFF["diff_src"])
    assert impl.startswith("fvec:"), impl
    d = np.abs(got.astype(np.float64) - want)
    scale = np.abs(np.exp(y) if alg == "softmax_log" else y).max() * np.abs(dy).sum(axis, keepdims=True) + np.abs(dy) + np.abs(want)
    assert np.all(d <= 1e-5 + 1e-5 * scale), float(np.max(d / (1e-5 + 1e-5 * scale)))


@pytest.mark.parametrize("w", [(1, 8, 1, 1), (1, 1, 1, 1), (2, 8, 1, 1)])
def test_prelu_backward(lib, w):
    s = (2, 8, 9, 9)
    x = RNG.uniform(-3, 3, s).astype(np.float32)
    x.flat[:3] = 0
    wt = RNG.uniform(-0.5, 0.5, w).astype(np.float32)
    dy = RNG.uniform(-2, 2, s).astype(np.float32)
    bufs = {ARG["src"]: x, ARG["weights"]: wt, DIFF["diff_dst"]: dy, DIFF["diff_src"]: np.zeros(s, np.float32),
            21: np.zeros(w, np.float32)}   # DIFF_WEIGHTS
    impl, got, want = both_bufs(lib, "aiednn_prelu_backward_create", [m(s), m(w), m(s), m(s), m(w)], bufs, DIFF["diff_src"])
    assert impl.startswith("fvec:"), impl
    contract(got[DIFF["diff_src"]], want[DIFF["diff_src"]])
    np.testing.assert_allclose(got[21], want[21], rtol=1e-5, atol=1e-4)


# the quantized binary (aiednn_binary_create_attr): bit-identical to dequantize -> binary -> post -> quantize
def qattr(sc, zp, dsc, dzp, post=()):
    from aiegraph.dnn_ctypes import Attr
    a = Attr()
    a.src_scales[0], a.src_scales[1] = sc
    a.src_zero_points[0], a.src_zero_points[1] = zp
    a.dst_scale, a.dst_zero_point = dsc, dzp
    for i, (alg, al, be) in enumerate(post):
        a.post_ops[i].kind, a.post_ops[i].alg, a.post_ops[i].alpha, a.post_ops[i].beta = 1, alg, al, be
    a.n_post_ops = len(post)
    return a


def chain(lib, alg, x, y, xdt, ydt, ddt, sc, zp, dsc, dzp, post):
    """the unfused chain in numpy, in the primitives' precisions"""
    def deq(v, dt, s, z):
        return v.astype(np.float32) if dt == "f32" or s == 0 else ((v.astype(np.float64) - z) * np.float64(np.float32(s))).astype(np.float32)
    a, b = deq(x, xdt, sc[0], zp[0]), deq(y, ydt, sc[1], zp[1])
    r = {"binary_add": np.add, "binary_sub": np.subtract, "binary_mul": np.multiply, "binary_max": np.maximum,
         "binary_min": np.minimum}[alg](a.astype(np.float64), b.astype(np.float64)).astype(np.float32)
    for name, al, be in post:
        r = np.maximum(r, 0) if name == "relu" else np.clip(r, np.float32(al), np.float32(be))
    if ddt == "f32":
        return r
    lo, hi = (-128, 127) if ddt == "s8" else (0, 255)
    return np.clip(np.rint(r.astype(np.float64) / np.float64(np.float32(dsc))) + dzp, lo, hi).astype(np.int8 if ddt == "s8" else np.uint8)


@pytest.mark.parametrize("alg", ["binary_add", "binary_sub", "binary_max", "binary_min", "binary_mul"])
@pytest.mark.parametrize("xdt,ydt,ddt", [("s8", "s8", "s8"), ("u8", "s8", "u8"), ("s8", "f32", "s8"), ("u8", "u8", "f32")])
@pytest.mark.parametrize("post", [(), (("relu", 0.0, 0.0),), (("clip", -0.5, 1.5),)])
def test_quantized_binary(lib, alg, xdt, ydt, ddt, post):
    s = (2, 16, 9, 9)
    def gen(dt):
        return (RNG.integers(-128 if dt == "s8" else 0, 128 if dt == "s8" else 256, s).astype(np.int8 if dt == "s8" else np.uint8)
                if dt != "f32" else RNG.uniform(-3, 3, s).astype(np.float32))
    x, y = gen(xdt), gen(ydt)
    sc, zp, dsc, dzp = (0.0213, 0.0371), (3, -7), 0.0459, (5 if ddt == "s8" else 120)
    algs = {"relu": lib.ALG["eltwise_relu"], "clip": lib.ALG["eltwise_clip"]}
    attr = qattr(sc, zp, dsc, dzp, [(algs[n], a, b) for n, a, b in post])
    odt = {"s8": np.int8, "u8": np.uint8, "f32": np.float32}[ddt]
    impl, got, want = both(lib, "aiednn_binary_create_attr", [C.c_int(lib.ALG[alg]), m(s, xdt), m(s, ydt), m(s, ddt), C.byref(attr)],
                           {ARG["src"]: x, ARG["src_1"]: y, ARG["dst"]: np.zeros(s, odt)}, ARG["dst"])
    ref = chain(lib, alg, x, y, xdt, ydt, ddt, sc, zp, dsc, dzp, post)
    assert np.array_equal(want, ref), "ref: is not the unfused chain"
    assert impl.startswith("fvec:") == (alg != "binary_mul"), impl
    assert np.array_equal(got, want)


# the quantized eltwise (aiednn_eltwise_forward_create_attr): bit-identical to dequantize -> eltwise -> quantize
@pytest.mark.parametrize("alg,a,b", [("eltwise_clip", 0.0, 6.0), ("eltwise_hardswish", 1 / 6, 0.5), ("eltwise_logistic", 0, 0),
                                     ("eltwise_gelu_erf", 0, 0), ("eltwise_relu", 0, 0)])
@pytest.mark.parametrize("sdt,ddt", [("s8", "s8"), ("u8", "u8"), ("s8", "f32"), ("f32", "s8")])
@pytest.mark.parametrize("post", [(), (("clip", -1.0, 4.0),)])
def test_quantized_eltwise(lib, alg, a, b, sdt, ddt, post):
    s = (2, 16, 9, 9)
    x = (RNG.integers(-128 if sdt == "s8" else 0, 128 if sdt == "s8" else 256, s).astype(np.int8 if sdt == "s8" else np.uint8)
         if sdt != "f32" else RNG.uniform(-3, 8, s).astype(np.float32))
    x.flat[:256 if sdt != "f32" else 0] = np.arange(256).astype(np.uint8).view(np.int8 if sdt == "s8" else np.uint8) if sdt != "f32" else []
    sc, zp, dsc, dzp = 0.0391, (3 if sdt == "s8" else 120), 0.0459, (5 if ddt == "s8" else 0)
    attr = qattr((sc, 0.0), (zp, 0), dsc, dzp, [(lib.ALG["eltwise_clip"], al, be) for _, al, be in post])
    odt = {"s8": np.int8, "u8": np.uint8, "f32": np.float32}[ddt]
    impl, got, want = both(lib, "aiednn_eltwise_forward_create_attr",
                           [C.c_int(lib.ALG[alg]), C.c_float(a), C.c_float(b), m(s, sdt), m(s, ddt), C.byref(attr)],
                           {ARG["src"]: x, ARG["dst"]: np.zeros(s, odt)}, ARG["dst"])
    # numpy model of the chain: dequantize (f32), eltwise (via torch in float64 then f32), post clip, quantize
    import torch
    deq = x.astype(np.float32) if sdt == "f32" else ((x.astype(np.float64) - zp) * np.float64(np.float32(sc))).astype(np.float32)
    t = torch.from_numpy(deq.astype(np.float64))
    fn = {"eltwise_clip": lambda v: torch.clamp(v, a, b), "eltwise_relu": torch.relu, "eltwise_logistic": torch.sigmoid,
          "eltwise_gelu_erf": lambda v: torch.nn.functional.gelu(v),
          "eltwise_hardswish": lambda v: v * torch.clamp(v * a + b, 0, 1)}[alg]
    r = fn(t).numpy().astype(np.float32)
    for _, al, be in post:
        r = np.clip(r, np.float32(al), np.float32(be))
    if ddt != "f32":
        lo, hi = (-128, 127) if ddt == "s8" else (0, 255)
        r = np.clip(np.rint(r.astype(np.float64) / np.float64(np.float32(dsc))) + dzp, lo, hi).astype(odt)
    assert np.array_equal(want, r) or np.abs(want.astype(np.float64) - r).max() <= (1 if ddt != "f32" else 1e-6), "ref: vs chain"
    assert impl.startswith("lut:") == (sdt != "f32"), impl
    assert np.array_equal(got, want)


@pytest.mark.parametrize("rms", [0, 8])
@pytest.mark.parametrize("flags", [0, 1 | 2])
@pytest.mark.parametrize("shape", [(4, 7, 64), (3, 96), (2, 2048), (5, 1536)])   # 2048 / 1536: past one item, chunked
def test_layer_norm_backward_on_fvec(lib, rms, flags, shape):
    """layer / RMS norm backward over the last dim on FVEC rows (any width: chunk sums add per row)"""
    D, rows = shape[-1], int(np.prod(shape[:-1]))
    x = (RNG.uniform(-3, 3, shape) + 0.5).astype(np.float32)
    dy = RNG.uniform(-2, 2, shape).astype(np.float32)
    bufs = {ARG["src"]: x, DIFF["diff_dst"]: dy, DIFF["diff_src"]: np.zeros(shape, np.float32),
            ARG["variance"]: RNG.uniform(0.5, 3.0, rows).astype(np.float32),
            ARG["mean"]: RNG.uniform(-0.3, 0.8, rows).astype(np.float32)}   # (RMS: given, unused)
    if flags & 1:
        bufs[ARG["scale"]] = RNG.uniform(0.5, 1.5, D).astype(np.float32)
        bufs[24] = np.zeros(D, np.float32)
    if flags & 2 and not rms:
        bufs[25] = np.zeros(D, np.float32)
    fl = flags | rms if not rms else (flags & 1) | rms
    impl, got, want = both_bufs(lib, "aiednn_layer_normalization_backward_create",
                                [m(shape), m(shape), m(shape), C.c_float(1e-5), C.c_uint(fl)], bufs, DIFF["diff_src"])
    assert impl.startswith("fvec:"), impl
    tol = 1e-5 * (np.abs(dy).mean() * 4 + 1)
    d = np.abs(got[DIFF["diff_src"]].astype(np.float64) - want[DIFF["diff_src"]])
    assert np.all(d <= tol * (1 + np.abs(want[DIFF["diff_src"]]))), float(d.max())
    for k in (24, 25):
        if k in bufs:
            np.testing.assert_allclose(got[k], want[k], rtol=1e-5, atol=1e-5 * np.abs(dy).sum() / D)


@pytest.mark.parametrize("alg,post", [("binary_sub", [("square", 0, 0)]), ("binary_add", [("relu", 0, 0)]),
                                      ("binary_mul", [("tanh", 0, 0), ("linear", 2.0, -1.0)]),
                                      ("binary_add", [("swish", 1.0, 0)])])
@pytest.mark.parametrize("bshape", [(2, 16, 9, 9), (1, 16, 1, 1)])
def test_f32_binary_with_eltwise_post_ops_on_fvec(lib, alg, post, bshape):
    """an f32 binary with eltwise post-ops (SquaredDifference = sub + square): the binary's program and each post-op's,
    one FVEC program -- within the contract of ref:'s chain"""
    from aiegraph.dnn_ctypes import attr
    s = (2, 16, 9, 9)
    x, y = RNG.uniform(-2, 2, s).astype(np.float32), RNG.uniform(-2, 2, bshape).astype(np.float32)
    a, keep = attr(lib.ALG, post_ops=[("eltwise", n, al, be) for n, al, be in post])
    impl, got, want = both(lib, "aiednn_binary_create_attr", [C.c_int(lib.ALG[alg]), m(s), m(bshape), m(s), C.byref(a)],
                           {ARG["src"]: x, ARG["src_1"]: y, ARG["dst"]: np.zeros(s, np.float32)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


@pytest.mark.parametrize("rms", [0, 8])
@pytest.mark.parametrize("flags", [0, 1, 1 | 2])
@pytest.mark.parametrize("shape", [(3, 2048), (2, 5, 1000), (4, 4096)])
def test_layer_norm_wide_rows_on_fvec(lib, rms, flags, shape):
    """layer / RMS norm inference over rows wider than one FVEC item (llama's 2048 / 4096) or not a multiple of 16:
    chunked ROWSUM statistics and ROWPROG passes -- within the f32 contract of ref:"""
    D = shape[-1]
    x = (RNG.uniform(-3, 3, shape) + 0.5).astype(np.float32)
    fl = (flags & (1 if rms else 3)) | rms
    bufs = {ARG["src"]: x, ARG["dst"]: np.zeros(shape, np.float32)}
    if fl & 1:
        bufs[ARG["scale"]] = RNG.uniform(0.5, 1.5, D).astype(np.float32)
    if fl & 2:
        bufs[ARG["shift"]] = RNG.uniform(-1, 1, D).astype(np.float32)
    impl, got, want = both(lib, "aiednn_layer_normalization_forward_inference_create",
                           [m(shape), m(shape), C.c_float(1e-5), C.c_uint(fl)], bufs, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    contract(got, want)


@pytest.mark.parametrize("a,b", [(1.0, 0.5), (2.0, 1.7), (0.5, -0.3), (1.0, 2.5), (1.0, 70.5)])
def test_pow_non_integer_exponent_on_fvec(lib, a, b):
    """a x^b with a non-integer b: a e^(b log x) on the core for x > 0; x <= 0 (NaN / 0 / inf) patched as ref:"""
    x = RNG.uniform(-2, 3, (4, 300)).astype(np.float32)
    x[0, :4] = [0.0, -0.0, 1e-20, -1.0]
    impl, got, want = both(lib, "aiednn_eltwise_forward_create", [C.c_int(lib.ALG["eltwise_pow"]), C.c_float(a),
                                                                  C.c_float(b), m(x.shape), m(x.shape)],
                           {ARG["src"]: x, ARG["dst"]: np.zeros_like(x)}, ARG["dst"])
    assert impl.startswith("fvec:"), impl
    pos = x > 0
    np.testing.assert_array_equal(got[~pos], want[~pos])   # NaN / 0 / inf: ref:'s values
    contract(got[pos], want[pos])


@pytest.mark.parametrize("ddt,zp", [("s8", 0), ("u8", 100)])
@pytest.mark.parametrize("rms,D", [(0, 768), (8, 512)])
def test_layer_norm_int8_dst_on_fvec(lib, rms, D, ddt, zp):
    """layer / RMS norm -> s8 / u8 (the norm-quant fold of gpt2 / tiny_llama): with AIEDNN_ATTR_TOLERANCE_1CODE the f32
    rows run on fvec: and the unpack quantizes them as the reorder does -- every code within one of ref:'s, almost
    all equal; without the flag the problem stays on ref: (bit-exact)"""
    from aiegraph.dnn_ctypes import attr
    shape = (1, 128, D)
    x = (RNG.uniform(-3, 3, shape) + 0.5).astype(np.float32)
    fl = (1 if rms else 3) | rms
    odt = np.int8 if ddt == "s8" else np.uint8
    bufs = {ARG["src"]: x, ARG["dst"]: np.zeros(shape, odt), ARG["scale"]: RNG.uniform(0.5, 1.5, D).astype(np.float32)}
    if not rms:
        bufs[ARG["shift"]] = RNG.uniform(-1, 1, D).astype(np.float32)
    for tol in (1, 0):
        a, _keep = attr(lib.ALG, dst_scale=0.03, dst_zp=zp)
        a.flags = tol
        impl, got, want = both(lib, "aiednn_layer_normalization_forward_inference_create_attr",
                               [m(shape), m(shape, ddt), C.c_float(1e-5), C.c_uint(fl), C.byref(a)], bufs, ARG["dst"])
        d = np.abs(got.astype(np.int32) - want.astype(np.int32))
        if tol:
            assert impl.startswith("fvec:"), impl
            assert d.max() <= 1 and (d == 0).mean() > 0.999, (d.max(), (d == 0).mean())
        else:
            assert impl.startswith("ref:"), impl
            np.testing.assert_array_equal(got, want)
