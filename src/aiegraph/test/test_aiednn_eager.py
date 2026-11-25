# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eager mode: generated gemm_s8 instances (aiegraph.dnn_eager) dispatched by eager:mm (aiednn_rt_eager.h).
Workstation emulation: each instance runs its AIEK_REFERENCE; the results must equal ref: byte for byte."""
from __future__ import annotations

import ctypes as C
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph.dnn_ctypes import ARG, POST_OP_SRC, AiednnLib, attr, conv_params, md  # noqa: E402
from aiegraph.dnn_eager import REF_FILE, EagerPlan, Instance, write_sources  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RNG = np.random.default_rng(41)
S, W, B, D = ARG["src"], ARG["weights"], ARG["bias"], ARG["dst"]
INSTANCES = [Instance(64, 64, 256, 64), Instance(128, 64, 192, 64), Instance(64, 768, 128, 64)]
Q_INSTANCES = [("gemm_q", 64, 64, 256, 64, 16), ("gemm_q", 128, 64, 192, 64, 16),
               ("gemm_qr", 64, 64, 256, 64, 16), ("gemm_qr", 128, 64, 192, 64, 16)]   # gemm_q<M, N, K, TK, TM>: the requant on the core
DW_INSTANCES = [(64, 32, 12, 16), (128, 64, 28, 16)]   # dwconv_q<M, C, TP, TM>
LUT_INSTANCES = [(64, 64, 16), (128, 128, 32)]   # lut<M, N, TM>
QUANT_INSTANCES = [(64, 64, 0, 16), (64, 64, 1, 16)]   # quant<M, N, U8, TM>
LNORM_INSTANCES = [(128, 768, 2, 0), (128, 256, 8, 1)]   # lnorm<R, L, TR, RMS>
ROPE_INSTANCES = [(128, 64, 8, 0), (128, 64, 8, 1)]   # rope<R, D, TR, INTER>
BF16_INSTANCES = [(64, 64, 128, 32, 16), (128, 32, 96, 32, 8)]   # gemm_bf16<M, N, K, TK, TM>
POOL_INSTANCES = [(256, 16, 16, 0, 0), (256, 8, 16, 1, 1), (256, 16, 16, 1, 0)]   # pool<MW, KK, TR, AVG, U8>


def write_q_sources(d: Path) -> Path:
    """the gemm_q instances as the generator writes gemm_s8's: an entry with exact specs and a C reference"""
    eager = REPO / "src" / "aietensorop" / "eager"
    for f in ("gemm_q_eager_kernel.cc", "gemm_q_eager_ref.h", "gemm_qr_eager_kernel.cc", "gemm_qr_eager_ref.h",
              "eager_common.h"):
        (d / f).write_text((eager / f).read_text())
    entry = ['#include "aie_user_kernel.h"', '#include "gemm_q_eager_kernel.cc"', '#include "gemm_qr_eager_kernel.cc"', ""]
    ref = ['#include "gemm_q_eager_ref.h"', '#include "gemm_qr_eager_ref.h"', ""]
    for t, M, N, K, TK, TM in Q_INSTANCES:
        nm = f"{t}__{M}_{N}_{K}_{TK}_{TM}"
        ld = K + TK if t == "gemm_q" else K + (1 + (N + TK - 1) // TK) * TK
        spec = (f"kernel    eager_{nm}\\nop        gemm_s8s32\\nmatch     dense\\nreference eager_q_ref.c\\n"
                f"target    vek280\\nin    int8  [{M},{ld}]  src\\nparam int8  [{N},{ld}]  weight\\n"
                f"out   int8  [{M},{N}]  dst\\n")
        entry += [f"static void {nm}_launch(int8_t *A, int8_t *W, int8_t *C) {{",
                  "    aieSetDevice(0);", "    aieArray device;",
                  "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                  f"    {t}<{M}, {N}, {K}, {TK}, {TM}><<<mesh>>>(A, W, C);", "}",
                  f"AIEK_ENTRY(eager_{nm}) {{ return eager_{t}_entry(args, num_args, {M}, {N}, {K}, {TK}, {nm}_launch); }}",
                  f'AIEK_SPEC(eager_{nm}, "{spec}");', ""]
        ref.append(f"AIEK_REFERENCE(eager_{nm}) {{ return eager_{t}_ref(args, num_args, {M}, {N}, {K}, {TK}); }}")
    (d / "eager_q.cc").write_text("\n".join(entry) + "\n")
    (d / "eager_q_ref.c").write_text("\n".join(ref) + "\n")
    return d / "eager_q_ref.c"


def write_pool_sources(d: Path) -> Path:
    """the pool instances: an entry with exact specs (op pool, kind from dnn.EAGER_KIND) and a C reference"""
    eager = REPO / "src" / "aietensorop" / "eager"
    for f in ("pool_eager_kernel.cc", "pool_eager_ref.h", "dwconv_q_eager_kernel.cc", "dwconv_q_eager_ref.h",
              "lut_eager_kernel.cc", "lut_eager_ref.h", "quant_eager_kernel.cc", "quant_eager_ref.h",
              "lnorm_eager_kernel.cc", "lnorm_eager_ref.h", "rope_eager_kernel.cc", "rope_eager_ref.h",
              "gemm_bf16_eager_kernel.cc", "gemm_bf16_eager_ref.h"):
        (d / f).write_text((eager / f).read_text())
    entry = ['#include "aie_user_kernel.h"', '#include "pool_eager_kernel.cc"', '#include "dwconv_q_eager_kernel.cc"',
             '#include "lut_eager_kernel.cc"', '#include "quant_eager_kernel.cc"',
             '#include "lnorm_eager_kernel.cc"',
             '#include "rope_eager_kernel.cc"',
             '#include "gemm_bf16_eager_kernel.cc"', ""]
    ref = ['#include "pool_eager_ref.h"', '#include "dwconv_q_eager_ref.h"', '#include "lut_eager_ref.h"', '#include "quant_eager_ref.h"', '#include "lnorm_eager_ref.h"', '#include "rope_eager_ref.h"', '#include "gemm_bf16_eager_ref.h"', ""]
    for MW, KK, TR, AVG, U8 in POOL_INSTANCES:
        nm = f"pool__{MW}_{KK}_{TR}_{AVG}_{U8}"
        spec = (f"kernel    eager_{nm}\\nop        pool\\nmatch     conv2d\\nreference eager_pool_ref.c\\n"
                f"target    vek280\\nin    int8  [{MW},{KK}]  src\\nparam int8  [16,{4 * KK}]  weight\\n"
                f"out   int8  [{MW},4]  dst\\n")
        entry += [f"static void {nm}_launch(int8_t *X, int8_t *Z, int8_t *Y) {{",
                  "    aieSetDevice(0);", "    aieArray device;",
                  "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                  f"    pool<{MW}, {KK}, {TR}, {AVG}, {U8}><<<mesh>>>(X, Z, Y);", "}",
                  f"AIEK_ENTRY(eager_{nm}) {{ return eager_pool_entry(args, num_args, {MW}, {KK}, {nm}_launch); }}",
                  f'AIEK_SPEC(eager_{nm}, "{spec}");', ""]
        ref.append(f"AIEK_REFERENCE(eager_{nm}) {{ return eager_pool_ref(args, num_args, {MW}, {KK}, {AVG}, {U8}); }}")
    for M, Cc, TP, TM in DW_INSTANCES:
        nm, K = f"dwconv_q__{M}_{Cc}_{TP}_{TM}", Cc // 4 * 4 * TP
        spec = (f"kernel    eager_{nm}\\nop        gemm_s8s32\\nmatch     dense\\nreference eager_pool_ref.c\\n"
                f"target    vek280\\nin    int8  [{M},{K}]  src\\nparam int8  [{Cc},{K}]  weight\\n"
                f"out   int8  [{M},{Cc}]  dst\\n")
        entry += [f"static void {nm}_launch(int8_t *A, int8_t *W, int8_t *Y) {{",
                  "    aieSetDevice(0);", "    aieArray device;",
                  "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                  f"    dwconv_q<{M}, {Cc}, {TP}, {TM}><<<mesh>>>(A, W, Y);", "}",
                  f"AIEK_ENTRY(eager_{nm}) {{ return eager_dwconv_q_entry(args, num_args, {M}, {Cc}, {TP}, {nm}_launch); }}",
                  f'AIEK_SPEC(eager_{nm}, "{spec}");', ""]
        ref.append(f"AIEK_REFERENCE(eager_{nm}) {{ return eager_dwconv_q_ref(args, num_args, {M}, {Cc}, {TP}); }}")
    for M, N, TM in LUT_INSTANCES:
        nm = f"lut__{M}_{N}_{TM}"
        spec = (f"kernel    eager_{nm}\\nop        gemm_s8s32\\nmatch     dense\\nreference eager_pool_ref.c\\n"
                f"target    vek280\\nin    int8  [{M},{256 + N}]  src\\nparam int8  [{N},{256 + N}]  weight\\n"
                f"out   int8  [{M},{N}]  dst\\n")
        entry += [f"static void {nm}_launch(int8_t *A, int8_t *W, int8_t *Y) {{",
                  "    aieSetDevice(0);", "    aieArray device;",
                  "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                  f"    lut<{M}, {N}, {TM}><<<mesh>>>(A, W, Y);", "}",
                  f"AIEK_ENTRY(eager_{nm}) {{ return eager_lut_entry(args, num_args, {M}, {N}, {nm}_launch); }}",
                  f'AIEK_SPEC(eager_{nm}, "{spec}");', ""]
        ref.append(f"AIEK_REFERENCE(eager_{nm}) {{ return eager_lut_ref(args, num_args, {M}, {N}); }}")
    for M, N, U8, TM in QUANT_INSTANCES:
        nm = f"quant__{M}_{N}_{U8}_{TM}"
        spec = (f"kernel    eager_{nm}\\nop        gemm_s8s32\\nmatch     dense\\nreference eager_pool_ref.c\\n"
                f"target    vek280\\nin    int8  [{M},{4 * N}]  src\\nparam int8  [{N},{4 * N}]  weight\\n"
                f"out   int8  [{M},{N}]  dst\\n")
        entry += [f"static void {nm}_launch(int8_t *A, int8_t *W, int8_t *Y) {{",
                  "    aieSetDevice(0);", "    aieArray device;",
                  "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                  f"    quant<{M}, {N}, {U8}, {TM}><<<mesh>>>(A, W, Y);", "}",
                  f"AIEK_ENTRY(eager_{nm}) {{ return eager_quant_entry(args, num_args, {M}, {N}, {nm}_launch); }}",
                  f'AIEK_SPEC(eager_{nm}, "{spec}");', ""]
        ref.append(f"AIEK_REFERENCE(eager_{nm}) {{ return eager_quant_ref(args, num_args, {M}, {N}, {U8}); }}")
    for R, L, TR, RMS in LNORM_INSTANCES:
        nm = f"lnorm__{R}_{L}_{TR}_{RMS}"
        spec = (f"kernel    eager_{nm}\\nop        gemm_s8s32\\nmatch     dense\\nreference eager_pool_ref.c\\n"
                f"target    vek280\\nin    int8  [{R},{4 * L}]  src\\nparam int8  [12,{L}]  weight\\n"
                f"out   int8  [{R},{4 * L}]  dst\\n")
        entry += [f"static void {nm}_launch(int8_t *X, int8_t *P, int8_t *Y) {{",
                  "    aieSetDevice(0);", "    aieArray device;",
                  "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                  f"    lnorm<{R}, {L}, {TR}, {RMS}><<<mesh>>>(X, P, Y);", "}",
                  f"AIEK_ENTRY(eager_{nm}) {{ return eager_lnorm_entry(args, num_args, {R}, {L}, {nm}_launch); }}",
                  f'AIEK_SPEC(eager_{nm}, "{spec}");', ""]
        ref.append(f"AIEK_REFERENCE(eager_{nm}) {{ return eager_lnorm_ref(args, num_args, {R}, {L}, {RMS}); }}")
    for R, D_, TR, IN in ROPE_INSTANCES:
        nm = f"rope__{R}_{D_}_{TR}_{IN}"
        spec = (f"kernel    eager_{nm}\\nop        pool\\nmatch     conv2d\\nreference eager_pool_ref.c\\n"
                f"target    vek280\\nin    int8  [{R},{12 * D_}]  src\\nout   int8  [{R},{4 * D_}]  dst\\n")
        entry += [f"static void {nm}_launch(int8_t *X, int8_t *Y) {{",
                  "    aieSetDevice(0);", "    aieArray device;",
                  "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                  f"    rope<{R}, {D_}, {TR}, {IN}><<<mesh>>>(X, Y);", "}",
                  f"AIEK_ENTRY(eager_{nm}) {{ return eager_rope_entry(args, num_args, {R}, {D_}, {nm}_launch); }}",
                  f'AIEK_SPEC(eager_{nm}, "{spec}");', ""]
        ref.append(f"AIEK_REFERENCE(eager_{nm}) {{ return eager_rope_ref(args, num_args, {R}, {D_}, {IN}); }}")
    for M, N, K, TK, TM in BF16_INSTANCES:
        nm = f"gemm_bf16__{M}_{N}_{K}_{TK}_{TM}"
        spec = (f"kernel    eager_{nm}\\nop        gemm_s8s32\\nmatch     dense\\nreference eager_pool_ref.c\\n"
                f"target    vek280\\nin    int8  [{M},{2 * K}]  src\\nparam int8  [{N},{2 * K}]  weight\\n"
                f"out   int32 [{M},{N}]  dst\\n")
        entry += [f"static void {nm}_launch(int8_t *A, int8_t *W, int32_t *C) {{",
                  "    aieSetDevice(0);", "    aieArray device;",
                  "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                  f"    gemm_bf16<{M}, {N}, {K}, {TK}, {TM}><<<mesh>>>(A, W, C);", "}",
                  f"AIEK_ENTRY(eager_{nm}) {{ return eager_gemm_bf16_entry(args, num_args, {M}, {N}, {K}, {nm}_launch); }}",
                  f'AIEK_SPEC(eager_{nm}, "{spec}");', ""]
        ref.append(f"AIEK_REFERENCE(eager_{nm}) {{ return eager_gemm_bf16_ref(args, num_args, {M}, {N}, {K}, {TK}); }}")
    (d / "eager_pool.cc").write_text("\n".join(entry) + "\n")
    (d / "eager_pool_ref.c").write_text("\n".join(ref) + "\n")
    return d / "eager_pool_ref.c"


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    from aiegraph.dnn import impl_list_source
    from aiegraph.kernel_lib import kernel_lib
    out = tmp_path_factory.mktemp("aiednn_eager")
    write_sources(EagerPlan("test", INSTANCES, [], len(INSTANCES)), out / "src")
    ref = out / "src" / REF_FILE
    qref = write_q_sources(out / "src")
    pref = write_pool_sources(out / "src")
    picked = list(kernel_lib(out / "src", quiet=True))
    want = [f"eager_{i.name}" for i in INSTANCES] + [f"eager_{q[0]}__" + "_".join(map(str, q[1:])) for q in Q_INSTANCES]
    want += ["eager_pool__" + "_".join(map(str, q)) for q in POOL_INSTANCES]
    want += ["eager_dwconv_q__" + "_".join(map(str, q)) for q in DW_INSTANCES]
    want += ["eager_lut__" + "_".join(map(str, q)) for q in LUT_INSTANCES]
    want += ["eager_quant__" + "_".join(map(str, q)) for q in QUANT_INSTANCES]
    want += ["eager_lnorm__" + "_".join(map(str, q)) for q in LNORM_INSTANCES]
    want += ["eager_rope__" + "_".join(map(str, q)) for q in ROPE_INSTANCES]
    want += ["eager_gemm_bf16__" + "_".join(map(str, q)) for q in BF16_INSTANCES]
    assert sorted(e.name for e in picked) == sorted(want), [e.name for e in picked]
    (out / "impl_list.cc").write_text(impl_list_source(picked, x86=True))
    so = out / "libaiednn_eager.so"
    cmd = ["g++", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-std=gnu++17", f"-I{REPO / 'include'}",
           f"-I{REPO / 'src' / 'aietensorop' / 'ops'}", f"-I{REPO / 'src' / 'aiednn'}", f"-I{out / 'src'}", f"-I{out / 'src'}",
           str(REPO / "src" / "aiednn" / "aiednn_rt.cc"), str(out / "impl_list.cc"), "-x", "c++", str(ref), "-x", "c", str(qref), str(pref), "-x", "none",
           "-o", str(so), "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    return AiednnLib(so)


def i8(shape):
    return RNG.integers(-127, 128, shape).astype(np.int8)


def mm(lib, x, w, ddt, a=None, b=None, filt=None):
    M, N = x.shape[0], w.shape[0]
    out = np.zeros((M, N), {"s32": np.int32, "s8": np.int8, "u8": np.uint8, "f32": np.float32}[ddt])
    bufs = {S: x, W: w, D: out}
    if b is not None:
        bufs[B] = b
    args = [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")) if b is not None else None,
            C.byref(md((M, N), ddt)), C.c_int(1)]
    fn = "aiednn_matmul_forward_create"
    if a is not None:
        args.append(C.byref(a))
        fn += "_attr"
    lib.set_filter(filt)
    try:
        return lib.call(fn, args, bufs, D)
    finally:
        lib.set_filter(None)


def test_exact_shape_runs_the_instance(lib):
    x, w = i8((64, 256)), i8((64, 256))
    got, impl = mm(lib, x, w, "s32")
    assert impl == "eager:mm", impl
    np.testing.assert_array_equal(got, x.astype(np.int64) @ w.astype(np.int64).T)


def test_padded_shape_with_requant_epilogue(lib):
    """M 50 -> 64, K 250 -> 256: zero padding is exact; bias + per-channel scales + s8 requant on the host"""
    x, w = i8((50, 250)), i8((64, 250))
    b = RNG.standard_normal((1, 64)).astype(np.float32)
    sc = (RNG.random(64) * 1e-3 + 1e-4).astype(np.float32)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05)
    got, impl = mm(lib, x, w, "s8", a, b)
    want, wimpl = mm(lib, x, w, "s8", a, b, filt="ref:")
    assert impl.startswith("eager:mm") and wimpl.startswith("ref:"), (impl, wimpl)
    np.testing.assert_array_equal(got, want)


def test_conv_through_im2col(lib):
    """3x3 conv, Cin 20 (K 180 -> 192), output 8 x 16 = 128 pixels, 64 channels: the second instance"""
    x, w = i8((1, 20, 10, 18)), i8((64, 20, 3, 3))
    cp = conv_params()
    outs = []
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            got, impl = lib.call("aiednn_convolution_forward_create",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), None,
                                  C.byref(md((1, 64, 8, 16), "s32")), C.byref(cp)],
                                 {S: x, W: w, D: np.zeros((1, 64, 8, 16), np.int32)}, D)
        finally:
            lib.set_filter(None)
        outs.append((got, impl))
    assert "eager:mm" in outs[0][1] and outs[1][1].startswith("ref:"), (outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][0], outs[1][0])


def test_uncovered_shape_runs_tiled(lib):
    """no instance covers M 150 x N 100 x K 600: eager:mm tiles it by an instance's block (int32 K sums on the host)"""
    x, w = i8((150, 600)), i8((100, 600))
    got, impl = mm(lib, x, w, "s32")
    assert impl == "eager:mm", impl
    np.testing.assert_array_equal(got, x.astype(np.int64) @ w.astype(np.int64).T)


def test_wide_n_instance(lib):
    """N 768 (each mesh column owns 192 columns), M 40 -> 64, K 100 -> 128 on the wide instance"""
    x, w = i8((40, 100)), i8((768, 100))
    got, impl = mm(lib, x, w, "s32")
    assert "eager" in impl, impl
    np.testing.assert_array_equal(got, x.astype(np.int64) @ w.astype(np.int64).T)


@pytest.mark.parametrize("ddt,post", [("s8", [("eltwise", "relu", 0.0, 0.0)]), ("u8", [("eltwise", "clip", -1.0, 3.0)])])
def test_requant_on_the_core(lib, ddt, post):
    """eager:mmq: per-channel scales + bias + ReLU / clip -> 8-bit, the fixed-point requant inside gemm_q
    (M 50 -> 64, K 250 -> 256); byte-identical to ref:"""
    x, w = i8((50, 250)), i8((64, 250))
    b = RNG.standard_normal((1, 64)).astype(np.float32)
    sc = (RNG.random(64) * 1e-3 + 1e-4).astype(np.float32)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=3 if ddt == "u8" else 0, post_ops=post)
    got, impl = mm(lib, x, w, ddt, a, b)
    want, wimpl = mm(lib, x, w, ddt, a, b, filt="ref:")
    assert impl == "eager:mmq" and wimpl.startswith("ref:"), (impl, wimpl)
    np.testing.assert_array_equal(got, want)


def test_conv_requant_on_the_core(lib):
    """3x3 conv + bias + per-channel scales + ReLU -> s8: eager:convq (im2col + gemm_q, codes into NCHW);
    Cin 20 (K 180 -> 192), 8 x 16 = 128 pixels, 64 channels; byte-identical to ref:"""
    x, w = i8((1, 20, 10, 18)), i8((64, 20, 3, 3))
    b = RNG.standard_normal(64).astype(np.float32)
    sc = (RNG.random(64) * 1e-3 + 1e-4).astype(np.float32)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, post_ops=[("eltwise", "relu", 0.0, 0.0)])
    outs = []
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            outs.append(lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((1, 64, 8, 16), "s8")), C.byref(conv_params()), C.byref(a)],
                                 {S: x, W: w, B: b, D: np.zeros((1, 64, 8, 16), np.int8)}, D))
        finally:
            lib.set_filter(None)
    assert outs[0][1] == "eager:convq" and outs[1][1].startswith("ref:"), (outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][0], outs[1][0])


def test_residual_requant_on_the_core(lib):
    """eager:mmq on gemm_qr: scales + bias + an s8 residual (scale / zero point) + ReLU -> s8, byte-identical to ref:"""
    x, w = i8((50, 250)), i8((64, 250))
    b = RNG.standard_normal((1, 64)).astype(np.float32)
    r = i8((50, 64))
    sc = (RNG.random(64) * 1e-3 + 1e-4).astype(np.float32)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05,
                   post_ops=[("binary", "add", (50, 64), "s8", 0.04, 5), ("eltwise", "relu", 0.0, 0.0)])
    outs = []
    for filt in (None, "ref:"):
        out = np.zeros((50, 64), np.int8)
        lib.set_filter(filt)
        try:
            outs.append(lib.call("aiednn_matmul_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((50, 64), "s8")), C.c_int(1), C.byref(a)],
                                 {S: x, W: w, B: b, POST_OP_SRC: r, D: out}, D))
        finally:
            lib.set_filter(None)
    assert outs[0][1] == "eager:mmq" and outs[1][1].startswith("ref:"), (outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][0], outs[1][0])


def test_conv_residual_requant_on_the_core(lib):
    """eager:convq on gemm_qr: a ResNet block's conv + bias + skip (u8, NCHW) + ReLU -> u8 as one launch"""
    x, w = i8((1, 20, 10, 18)), i8((64, 20, 3, 3))
    b = RNG.standard_normal(64).astype(np.float32)
    r = RNG.integers(0, 256, (1, 64, 8, 16)).astype(np.uint8)
    sc = (RNG.random(64) * 1e-3 + 1e-4).astype(np.float32)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=7,
                   post_ops=[("binary", "add", (1, 64, 8, 16), "u8", 0.03, 128), ("eltwise", "relu", 0.0, 0.0)])
    outs = []
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            outs.append(lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((1, 64, 8, 16), "u8")), C.byref(conv_params()), C.byref(a)],
                                 {S: x, W: w, B: b, POST_OP_SRC: r, D: np.zeros((1, 64, 8, 16), np.uint8)}, D))
        finally:
            lib.set_filter(None)
    assert outs[0][1] == "eager:convq" and outs[1][1].startswith("ref:"), (outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][0], outs[1][0])


def pool(lib, x, k, s, pad, alg, filt=None):
    """2-D pooling of x [N, C, H, W] (8-bit) -> (out, impl)"""
    from aiegraph.dnn_ctypes import pool_params
    N, Cc, H, Wd = x.shape
    Ho, Wo = (H + 2 * pad - k) // s + 1, (Wd + 2 * pad - k) // s + 1
    dt = "u8" if x.dtype == np.uint8 else "s8"
    out = np.zeros((N, Cc, Ho, Wo), x.dtype)
    lib.set_filter(filt)
    try:
        return lib.call("aiednn_pooling_forward_create",
                        [C.c_int(lib.ALG[alg]), C.byref(md(x.shape, dt)), C.byref(md(out.shape, dt)),
                         C.byref(pool_params((k, k), (s, s), (pad, pad), (pad, pad)))], {S: x, D: out}, D)
    finally:
        lib.set_filter(None)


@pytest.mark.parametrize("dt,k,s,pad,alg", [
    ("s8", 3, 2, 1, "pooling_max"),                    # resnet's stem pool, windows in the padding
    ("u8", 2, 2, 0, "pooling_avg_exclude_padding"),
    ("u8", 2, 1, 1, "pooling_avg_exclude_padding"),    # counts 1, 2 and 4
    ("s8", 3, 1, 1, "pooling_avg_include_padding"),    # negative sums: round half to even of a negative quotient
    ("s8", 3, 1, 1, "pooling_avg_exclude_padding"),
])
def test_pool_on_the_instance(lib, dt, k, s, pad, alg):
    """eager:pool: 8-bit max / average pooling over several launches of one instance, byte-identical to ref:"""
    x = (RNG.integers(0, 256, (2, 5, 13, 11)).astype(np.uint8) if dt == "u8" else i8((2, 5, 13, 11)))
    got, impl = pool(lib, x, k, s, pad, alg)
    want, wimpl = pool(lib, x, k, s, pad, alg, filt="ref:")
    assert impl == "eager:pool" and wimpl.startswith("ref:"), (impl, wimpl)
    np.testing.assert_array_equal(got, want)


def test_pool_without_an_instance_stays_on_ref(lib):
    """a 5x5 window has no instance (KK 32): another implementation takes it, never eager:pool"""
    got, impl = pool(lib, i8((1, 3, 9, 9)), 5, 1, 0, "pooling_max")
    assert impl != "eager:pool", impl


def dwconv(lib, x, k, s, pad, dil, ddt, relu, filt):
    N, Cc, H, Wd = x.shape
    ke = (k - 1) * (dil + 1) + 1
    Ho, Wo = (H + 2 * pad - ke) // s + 1, (Wd + 2 * pad - ke) // s + 1
    w = np.random.default_rng(Cc * 7 + k).integers(-127, 128, (Cc, 1, k, k)).astype(np.int8)
    b = np.random.default_rng(Cc + k).standard_normal(Cc).astype(np.float32) * 20
    sc = (np.random.default_rng(Cc + 2 * k).random(Cc) * 2e-2 + 1e-3).astype(np.float32)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, dst_zp=(7 if ddt == "u8" else 0),
                   post_ops=[("eltwise", "relu", 0.0, 0.0)] if relu else [])
    out = np.zeros((N, Cc, Ho, Wo), np.uint8 if ddt == "u8" else np.int8)
    lib.set_filter(filt)
    try:
        return lib.call("aiednn_convolution_forward_create_attr",
                        [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                         C.byref(md(out.shape, ddt)), C.byref(conv_params((s, s), (pad, pad), (pad, pad), (dil, dil), Cc)),
                         C.byref(a)], {S: x, W: w, B: b, D: out}, D)
    finally:
        lib.set_filter(None)


@pytest.mark.parametrize("shape,k,s,pad,dil,ddt,relu", [
    ((2, 40, 9, 11), 3, 1, 1, 0, "s8", True),     # two channel slices of 32 (the second partly empty), two images
    ((1, 64, 14, 14), 3, 2, 1, 0, "u8", False),   # stride 2, a u8 dst with a zero point
    ((1, 24, 12, 10), 5, 1, 2, 0, "s8", False),   # 5x5: the TP 28 instance
    ((1, 16, 12, 12), 3, 1, 2, 1, "s8", True),    # dilation 2
])
def test_depthwise_requant_on_the_core(lib, shape, k, s, pad, dil, ddt, relu):
    """eager:dwq: depthwise conv + bias + scales [+ ReLU] -> 8-bit, several launches; byte-identical to ref:"""
    x = i8(shape)
    got, impl = dwconv(lib, x, k, s, pad, dil, ddt, relu, None)
    want, wimpl = dwconv(lib, x, k, s, pad, dil, ddt, relu, "ref:")
    assert impl == "eager:dwq" and wimpl.startswith("ref:"), (impl, wimpl)
    np.testing.assert_array_equal(got, want)


def both_lut(lib, fn, args, bufs):
    outs = []
    for filt in (None, "ref:"):
        b = {k: (v.copy() if k == D else v) for k, v in bufs.items()}
        lib.set_filter(filt)
        try:
            outs.append(lib.call(fn, args, b, D))
        finally:
            lib.set_filter(None)
    (got, impl), (want, wimpl) = outs
    assert impl == "eager:lut" and wimpl.startswith("ref:"), (impl, wimpl)
    np.testing.assert_array_equal(got, want)


SHAPE = (2, 16, 9, 9)   # runs of 81: each split across two columns, several launches


def test_lut_eltwise(lib):
    """leaky ReLU s8 -> s8: one table for the tensor"""
    x = i8(SHAPE)
    both_lut(lib, "aiednn_eltwise_forward_create",
             [C.c_int(lib.ALG["eltwise_relu"]), C.c_float(0.37), C.c_float(0.0), C.byref(md(SHAPE, "s8")),
              C.byref(md(SHAPE, "s8"))], {S: x, D: np.zeros(SHAPE, np.int8)})


def test_lut_prelu_per_channel(lib):
    """PReLU with per-channel f32 slopes, s8 -> u8: a table per channel"""
    x, w = i8(SHAPE), RNG.uniform(-0.5, 0.5, (1, 16, 1, 1)).astype(np.float32)
    both_lut(lib, "aiednn_prelu_forward_create",
             [C.byref(md(SHAPE, "s8")), C.byref(md(w.shape, "f32")), C.byref(md(SHAPE, "u8"))],
             {S: x, W: w, D: np.zeros(SHAPE, np.uint8)})


def test_lut_batch_norm(lib):
    """batch norm (scale + shift) on u8 -> s8"""
    x = RNG.integers(0, 256, SHAPE).astype(np.uint8)
    bufs = {S: x, ARG["mean"]: RNG.uniform(50, 200, 16).astype(np.float32),
            ARG["variance"]: RNG.uniform(20, 900, 16).astype(np.float32),
            ARG["scale"]: RNG.uniform(10, 40, 16).astype(np.float32),
            ARG["shift"]: RNG.uniform(-5, 5, 16).astype(np.float32), D: np.zeros(SHAPE, np.int8)}
    both_lut(lib, "aiednn_batch_normalization_forward_inference_create",
             [C.byref(md(SHAPE, "u8")), C.byref(md(SHAPE, "s8")), C.c_float(1e-5), C.c_uint(3)], bufs)


def test_lut_binary_broadcast(lib):
    """s8 * a per-channel f32 vector -> s8"""
    x, y = i8(SHAPE), RNG.uniform(-1.5, 1.5, (1, 16, 1, 1)).astype(np.float32)
    both_lut(lib, "aiednn_binary_create",
             [C.c_int(lib.ALG["binary_mul"]), C.byref(md(SHAPE, "s8")), C.byref(md(y.shape, "f32")), None,
              C.byref(md(SHAPE, "s8"))], {S: x, ARG["src_1"]: y, D: np.zeros(SHAPE, np.int8)})


def test_requant_slices_and_chunks(lib):
    """eager:mmq on a shape no instance has: N 200 = 4 channel slices of 64, M 300 = 3 row chunks of 128,
    K 150 <= 192; with an s8 residual (gemm_qr) and without; byte-identical to ref:"""
    x, w = i8((300, 150)), i8((200, 150))
    b = RNG.standard_normal((1, 200)).astype(np.float32)
    r = i8((300, 200))
    sc = (RNG.random(200) * 1e-3 + 1e-4).astype(np.float32)
    for po in ([("eltwise", "relu", 0.0, 0.0)],
               [("binary", "add", (300, 200), "s8", 0.04, 5), ("eltwise", "relu", 0.0, 0.0)]):
        a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, post_ops=po)
        outs = []
        for filt in (None, "ref:"):
            out = np.zeros((300, 200), np.int8)
            lib.set_filter(filt)
            try:
                outs.append(lib.call("aiednn_matmul_forward_create_attr",
                                     [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                      C.byref(md((300, 200), "s8")), C.c_int(1), C.byref(a)],
                                     {S: x, W: w, B: b, POST_OP_SRC: r, D: out}, D))
            finally:
                lib.set_filter(None)
        assert outs[0][1] == "eager:mmq" and outs[1][1].startswith("ref:"), (outs[0][1], outs[1][1])
        np.testing.assert_array_equal(outs[0][0], outs[1][0])


def test_conv_on_a_smaller_instance(lib):
    """3x3 conv, O 96 (two slices of 64), 2 images x 10 x 12 pixels (240 = 2 chunks of 128), K 135"""
    x, w = i8((2, 15, 12, 14)), i8((96, 15, 3, 3))
    b = RNG.standard_normal(96).astype(np.float32)
    sc = (RNG.random(96) * 1e-3 + 1e-4).astype(np.float32)
    a, keep = attr(lib.ALG, scales=sc, dst_scale=0.05, post_ops=[("eltwise", "relu", 0.0, 0.0)])
    outs = []
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            outs.append(lib.call("aiednn_convolution_forward_create_attr",
                                 [C.byref(md(x.shape, "s8")), C.byref(md(w.shape, "s8")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((2, 96, 10, 12), "s8")), C.byref(conv_params()), C.byref(a)],
                                 {S: x, W: w, B: b, D: np.zeros((2, 96, 10, 12), np.int8)}, D))
        finally:
            lib.set_filter(None)
    assert outs[0][1] == "eager:convq" and outs[1][1].startswith("ref:"), (outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][0], outs[1][0])


def test_pool_on_a_wider_window(lib):
    """s8 max 2x2 (8-byte windows) on the 3x3 instance (16-byte windows): the extra taps are padding"""
    x = i8((1, 6, 10, 10))
    got, impl = pool(lib, x, 2, 2, 0, "pooling_max")
    want, wimpl = pool(lib, x, 2, 2, 0, "pooling_max", filt="ref:")
    assert impl == "eager:pool", impl
    np.testing.assert_array_equal(got, want)




@pytest.mark.parametrize("dt,scale,zp", [("s8", 0.0371, 3), ("u8", 0.125, 113), ("s8", 3.0e-7, -5)])
def test_quantize_on_the_core(lib, dt, scale, zp):
    """eager:quant: f32 -> 8-bit reorder (input quantize), exact ties / NaN / inf / huge included, several
    launches; byte-identical to ref:"""
    n = 3 * 64 * 64 + 777
    k = RNG.integers(-300, 300, n)
    x = np.where(RNG.random(n) < 0.4, (k + 0.5) * np.float64(np.float32(scale)),
                 RNG.standard_normal(n) * 200 * scale).astype(np.float32)
    x[::97], x[1::97], x[2::97], x[3::97] = np.nan, np.inf, -np.inf, -3e38
    x = x.reshape(1, n)
    outs = []
    for filt in (None, "ref:"):
        out = np.zeros(x.shape, np.uint8 if dt == "u8" else np.int8)
        lib.set_filter(filt)
        try:
            outs.append(lib.call("aiednn_reorder_create", [C.byref(md(x.shape, "f32")), C.byref(md(x.shape, dt)), None,
                                                          C.c_float(scale), C.c_int32(zp)], {S: x, D: out}, D))
        finally:
            lib.set_filter(None)
    assert outs[0][1] == "eager:quant" and outs[1][1].startswith("ref:"), (outs[0][1], outs[1][1])
    np.testing.assert_array_equal(outs[0][0], outs[1][0])


def lnorm(lib, x, flags, gamma=None, beta=None, ddt="f32", a=None, filt=None):
    shape = x.shape
    out = np.zeros(shape, {"f32": np.float32, "s8": np.int8, "u8": np.uint8}[ddt])
    bufs = {S: x, D: out}
    if gamma is not None:
        bufs[ARG["scale"]] = gamma
    if beta is not None:
        bufs[ARG["shift"]] = beta
    args = [C.byref(md(shape, "f32")), C.byref(md(shape, ddt)), C.c_float(1e-5), C.c_uint(flags)]
    fn = "aiednn_layer_normalization_forward_inference_create"
    if a is not None:
        args.append(C.byref(a))
        fn += "_attr"
    lib.set_filter(filt)
    try:
        return lib.call(fn, args, bufs, D)
    finally:
        lib.set_filter(None)


def lcontract(got, want):
    d = np.abs(got.astype(np.float64) - want.astype(np.float64))
    assert np.all(d <= 1e-5 + 1e-5 * np.abs(want)), float(np.max(d / (1e-5 + 1e-5 * np.abs(want))))


@pytest.mark.parametrize("shape,flags,C_", [((3, 197, 768), 3, 768),   # LayerNorm with gamma / beta, 591 rows
                                            ((70, 512), 3, 512),        # a narrower row on the 768 instance
                                            ((40, 200), 1 | 8, 200)])   # RMS norm (scale only) on the 256 one
def test_lnorm_rows_on_the_core(lib, shape, flags, C_):
    """eager:lnorm, f32 -> f32: within the fvec: contract of ref:'s double arithmetic"""
    x = (RNG.standard_normal(shape) * RNG.uniform(0.1, 5, shape[:-1] + (1,)) + RNG.uniform(-10, 10, shape[:-1] + (1,))
         ).astype(np.float32)
    g = RNG.uniform(0.5, 1.5, C_).astype(np.float32) if flags & 1 else None
    b = RNG.uniform(-0.5, 0.5, C_).astype(np.float32) if flags & 2 else None
    got, impl = lnorm(lib, x, flags, g, b)
    want, wimpl = lnorm(lib, x, flags, g, b, filt="ref:")
    assert impl == "eager:lnorm" and wimpl.startswith("ref:"), (impl, wimpl)
    lcontract(got, want)


def test_lnorm_to_s8_needs_the_tolerance_flag(lib):
    """-> s8: only with AIEDNN_ATTR_TOLERANCE_1CODE (then within one code of ref:); without it ref: keeps it"""
    x = (RNG.standard_normal((64, 768)) * 3).astype(np.float32)
    g, b = RNG.uniform(0.5, 1.5, 768).astype(np.float32), RNG.uniform(-0.5, 0.5, 768).astype(np.float32)
    a0, k0 = attr(lib.ALG, dst_scale=0.02, dst_zp=3)
    _, impl = lnorm(lib, x, 3, g, b, "s8", a0)
    assert impl != "eager:lnorm", impl
    a1, k1 = attr(lib.ALG, dst_scale=0.02, dst_zp=3, flags=1)
    got, impl = lnorm(lib, x, 3, g, b, "s8", a1)
    want, _ = lnorm(lib, x, 3, g, b, "s8", a0, filt="ref:")
    assert impl == "eager:lnorm", impl
    assert np.max(np.abs(got.astype(int) - want.astype(int))) <= 1


def rope_call(lib, x, cos, sin, inter, dt="f32", filt=None):
    COS_, SIN_ = ARG["rope_cos"], ARG["rope_sin"]
    m = lambda a: C.byref(md(a.shape, "f32"))
    out = np.zeros(x.shape, {"f32": np.float32, "s8": np.int8, "u8": np.uint8}[dt])
    args = [m(x), m(cos), m(sin), C.byref(md(x.shape, dt)), C.c_int(inter)]
    fn = "aiednn_rope_create"
    if dt != "f32":
        a, keep = attr(lib.ALG, dst_scale=0.013, dst_zp=-3 if dt == "s8" else 120)
        args.append(C.byref(a))
        fn += "_attr"
    lib.set_filter(filt)
    try:
        return lib.call(fn, args, {S: x, COS_: cos, SIN_: sin, D: out}, D)
    finally:
        lib.set_filter(None)


@pytest.mark.parametrize("inter", [0, 1], ids=["half_split", "interleaved"])
@pytest.mark.parametrize("dt", ["f32", "s8", "u8"])
def test_rope_bit_identical(lib, inter, dt):
    """eager:rope: [B, H, S, 64] with cos / sin broadcast over heads, several launches; byte-identical to ref:"""
    x = (RNG.standard_normal((2, 6, 30, 64)) * 2).astype(np.float32)
    ang = RNG.uniform(-3, 3, (1, 1, 30, 64)).astype(np.float32)
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    got, impl = rope_call(lib, x, cos, sin, inter, dt)
    want, wimpl = rope_call(lib, x, cos, sin, inter, dt, filt="ref:")
    assert impl == "eager:rope" and wimpl.startswith("ref:"), (impl, wimpl)
    np.testing.assert_array_equal(got, want)


def to_bf16(x):
    """f32 -> (bf16 bits as uint16, the same values as float64), round to nearest even"""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return b, (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def fmm(lib, a, w, dt, oshape, bias=None, filt=None):
    bufs = {S: a, W: w, D: np.zeros(oshape, np.float32)}
    bmd = None
    if bias is not None:
        bufs[B], bmd = bias, C.byref(md(bias.shape, "f32"))
    lib.set_filter(filt)
    try:
        return lib.call("aiednn_matmul_forward_create",
                        [C.byref(md(a.shape, dt)), C.byref(md(w.shape, dt)), bmd, C.byref(md(oshape, "f32")),
                         C.c_int(0)], bufs, D)
    finally:
        lib.set_filter(None)


@pytest.mark.parametrize("M,K,N", [(37, 300, 70), (130, 64, 129)])
def test_bf16_matmul_on_the_core(lib, M, K, N):
    """eager:bf16: bf16 x bf16 -> f32 in tiles of the instances (K blocks summed on the host); fixed mode's
    bf16blk: contract against the exact product"""
    a, ad = to_bf16(RNG.uniform(-2, 2, (M, K)))
    w, wd = to_bf16(RNG.uniform(-2, 2, (K, N)))
    got, impl = fmm(lib, a, w, "bf16", (M, N))
    assert impl == "eager:bf16", impl
    scale = np.abs(ad) @ np.abs(wd)
    assert np.all(np.abs(got - ad @ wd) <= 1e-5 * np.sqrt(K) * (scale + 1e-30))


def test_f32_matmul_as_bf16x3_on_the_core(lib):
    """eager:bf16x3: f32 x f32 + bias as the bf16x3 split (depth 6 K): near-f32 accuracy"""
    a = RNG.uniform(-2, 2, (20, 1000)).astype(np.float32)
    w = RNG.uniform(-2, 2, (1000, 50)).astype(np.float32)
    bias = RNG.uniform(-1, 1, (1, 50)).astype(np.float32)
    got, impl = fmm(lib, a, w, "f32", (20, 50), bias)
    assert impl == "eager:bf16x3", impl
    want = a.astype(np.float64) @ w.astype(np.float64) + bias
    scale = np.abs(a.astype(np.float64)) @ np.abs(w.astype(np.float64))
    assert (np.abs(got - want) / (scale + 1e-30)).max() < 2e-6


def test_f32_conv_reaches_the_core(lib):
    """an f32 convolution: the gemm+conv composite now finds a float GEMM on the AIE (eager:bf16x3)"""
    x = RNG.uniform(-1, 1, (1, 8, 12, 12)).astype(np.float32)
    w = RNG.uniform(-1, 1, (16, 8, 3, 3)).astype(np.float32)
    b = RNG.uniform(-1, 1, 16).astype(np.float32)
    outs = []
    for filt in (None, "ref:"):
        lib.set_filter(filt)
        try:
            outs.append(lib.call("aiednn_convolution_forward_create",
                                 [C.byref(md(x.shape, "f32")), C.byref(md(w.shape, "f32")), C.byref(md(b.shape, "f32")),
                                  C.byref(md((1, 16, 10, 10), "f32")), C.byref(conv_params())],
                                 {S: x, W: w, B: b, D: np.zeros((1, 16, 10, 10), np.float32)}, D))
        finally:
            lib.set_filter(None)
    (got, impl), (want, wimpl) = outs
    assert "eager:bf16x3" in impl and wimpl.startswith("ref:"), (impl, wimpl)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

