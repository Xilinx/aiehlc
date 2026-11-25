###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.dnn_eager: eager mode's kernel instances for one model.

Eager mode gives every GEMM-family layer its own instance of a templated, shape-specialized AIE kernel
(``gemm_s8<M, N, K, TK>``, int8 x int8 -> int32), loaded right before the layer runs. This module is the
compiler half that needs no board: it walks a lowered :class:`AGraph`, maps each matmul / inner product /
convolution primitive to the (M, N, K) of the GEMM it is (a convolution through im2col), picks the template
arguments (:func:`pick`) and de-duplicates them into an instance list::

    plan = dnn_eager.plan(g)          # g = dnn_torch.references(..., "int8", ...)[0]
    plan.instances                    # [Instance(M, N, K, TK, ops=[...])], one per distinct shape
    plan.refused                      # [(op name, reason)]: those layers keep the fixed-megakernel / ref: path
    plan.write(out / "eager_instances.json")

Template constraints (``gemm_s8<M, N, K, TK, TM>``, 2026-11-17): each of the 4 mesh columns owns N / 4
output columns as one block (TN = N / 4), so N must be a multiple of 4; M is padded with zero rows to a
multiple of 4 * TM and K with zeros to a multiple of TK (both exact). The TM of :data:`TM_CHOICES`
and TK of :data:`TK_CHOICES` (whole DMA words) with the most work per round (TM x TK) whose port buffers fit
:data:`CORE_BUDGET` and whose on-core accumulator fits :data:`ACC_BUDGET` (:func:`core_bytes`).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .errors import AiegraphError

MESH_COLS = 4                      # mesh columns; each owns N / MESH_COLS output columns (TN)
TM_CHOICES = (16, 8, 4, 2, 1)      # rows of A per core round, largest first
TK_CHOICES = (256, 128, 64, 32, 16, 8, 4)   # K per core round, largest first; multiples of 4 (DMA words)
CORE_BUDGET = 29_696               # bytes of core data memory the instance's port buffers may take
ACC_BUDGET = 16_384                # bytes of the on-core int32 accumulator (TM x TN, on the stack)
Q_WORDS = 8                        # gemm_q: int32 requant words per channel {M, B lo, B hi, sh, zp, lo, hi, 0}
Q_TK_MIN = 32                      # gemm_q: smallest TK (its W chunk past K holds the 8 words)
QR_WORDS, QR_TK_MIN = 12, 48       # gemm_qr: 12 int32 words per channel (requant + residual scale / zp)
CONV_Q = True                      # convs on gemm_q (the runtime's eager:convq: im2col + gemm_q)
K_PAD_MAX = 8                      # K's zero padding stays within K / K_PAD_MAX when a TK allows it

MATMUL_FNS = ("aiednn_matmul_forward_create", "aiednn_matmul_forward_create_attr")
IP_FNS = ("aiednn_inner_product_forward_create", "aiednn_inner_product_forward_create_attr")
CONV_FNS = ("aiednn_convolution_forward_create", "aiednn_convolution_forward_create_attr")
DECONV_FNS = ("aiednn_deconvolution_forward_create", "aiednn_deconvolution_forward_create_attr")


@dataclass
class Gemm:
    """one layer as a GEMM: C[M, N] = A[M, K] . B[K, N] (int8 in, int32 out)"""
    op: int                        # index into g.ops
    name: str
    kind: str                      # matmul | inner_product | conv
    M: int
    N: int
    K: int
    tmpl: str = None               # gemm_q / gemm_qr: the epilogue runs on the core (fx_template)


@dataclass
class Instance:
    """one template instance ``<template><M, N, K, TK, TM>`` and the layers that run on it"""
    M: int                         # padded
    N: int
    K: int                         # padded
    TK: int
    TM: int = 16                   # the template's default (gemm_s8<..., TM = 16>)
    template: str = "gemm_s8"      # gemm_s8: int32 out, host epilogue | gemm_q / gemm_qr: 8-bit out, requant on the core
    params: tuple = ()             # a non-GEMM template's arguments (pool: MW, KK, TR, AVG, U8); M..TM unused
    ops: list = field(default_factory=list)      # [{"op", "name", "kind", "M", "N", "K"}]: the layers' own shape

    @property
    def name(self) -> str:
        if self.template in GEMM_TEMPLATES:
            return f"{self.template}__{self.M}_{self.N}_{self.K}_{self.TK}_{self.TM}"
        return f"{self.template}__" + "_".join(map(str, self.params))


@dataclass
class EagerPlan:
    model: str
    instances: list                # [Instance]
    refused: list                  # [(op name, reason)]
    gemm_layers: int               # GEMM-family layers seen (instances' layers + refused)
    other_layers: int = 0          # non-GEMM layers given a template instance (pooling)
    mesh_rows: int = 4             # HW_ROWS: core rows the instances split their rows over (the entry #defines it)

    def summary(self) -> str:
        lines = [f"eager plan [{self.model}, {self.mesh_rows}x{MESH_COLS} mesh]: {self.gemm_layers} GEMM layers"
                 + (f" + {self.other_layers} other" if self.other_layers else "")
                 + f" -> {len(self.instances)} instances, {len(self.refused)} refused"]
        lines += [f"  {i.name}: {len(i.ops)} layer(s) " + ", ".join(o["name"] for o in i.ops) for i in self.instances]
        lines += [f"  refused {n}: {why}" for n, why in self.refused]
        return "\n".join(lines)

    def to_json(self) -> dict:
        return dict(model=self.model, gemm_layers=self.gemm_layers, mesh=[self.mesh_rows, MESH_COLS],
                    constraints=dict(MESH_COLS=MESH_COLS, TM_CHOICES=list(TM_CHOICES), TK_CHOICES=list(TK_CHOICES),
                                     CORE_BUDGET=CORE_BUDGET, ACC_BUDGET=ACC_BUDGET),
                    instances=[dict(name=i.name, **asdict(i)) for i in self.instances],
                    refused=[dict(name=n, reason=r) for n, r in self.refused])

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1) + "\n")
        return path


def core_bytes(TM: int, TK: int, TN: int) -> tuple:
    """(port buffer bytes, accumulator bytes) of one core: double-buffered A (TM x TK) and B (TN x TK) int8
    tiles and a double-buffered TM x TN int32 C tile; the accumulator is one TM x TN int32 tile"""
    return 2 * (TM * TK + TN * TK) + 2 * TM * TN * 4, TM * TN * 4


def pick(M: int, N: int, K: int, template: str = "gemm_s8") -> tuple:
    """template arguments for a GEMM of this shape: ((Mp, N, Kp, TK, TM), None), or (None, reason) when the
    template cannot run it. gemm_q: TK >= :data:`Q_TK_MIN`, and its core also holds the TN x 8 int32 requant
    words next to the accumulator"""
    if min(M, N, K) <= 0:
        return None, f"empty GEMM ({M}, {N}, {K})"
    if N % MESH_COLS:
        return None, f"N = {N}: not a multiple of {MESH_COLS} (each mesh column owns N / {MESH_COLS} columns)"
    if template in ("gemm_q", "gemm_qr") and N % (4 * MESH_COLS):
        # an 8-bit C: a core's output row is TN bytes and the DMA moves whole 32-bit words -- TN 6 (N 24) wrote
        # garbage on the board (mobilenet_v2 features_2/3_conv_2) though workstation emulation passed; pad N
        # (exact: the runtime drops the extra columns)
        N = math.ceil(N / (4 * MESH_COLS)) * 4 * MESH_COLS
    TN = N // MESH_COLS
    # the (TM, TK) that fits with the most work per core round (TM x TK multiply-accumulates per output
    # column); ties go to the larger TM. N 64: (16, 256); N 768: (8, 32); N 3072: (2, 8)
    # padding is exact but computes zeros: keep K's padding within 1/8 of K when any (TM, TK) can (K 576:
    # TK 128 pads to 640, not 256 to 768), then the most work per round, ties to the larger TM
    def kpad(TK):
        return math.ceil(K / TK) * TK - K
    tk_min = {"gemm_q": Q_TK_MIN, "gemm_qr": QR_TK_MIN}.get(template, 0)
    # VEK280's core-memory allocator sizes every input slot from A's block: W's (TN' x TK) must not exceed A's
    # (TM x TK), so an instance's N is capped at MESH_COLS x TM and a wider layer runs in N slices (the runtime
    # tiles any GEMM over its instance). Score: work per core round TM x TK x TN'
    fit = []
    for TM in TM_CHOICES:
        tn = min(TN, TM)
        for TK in TK_CHOICES:
            if (TK >= tk_min and core_bytes(TM, TK, tn)[0] <= CORE_BUDGET
                    # on-core extras next to the accumulator: gemm_q's requant words, gemm_qr's 12 + its residual tile
                    and core_bytes(TM, TK, tn)[1] + {"gemm_q": tn * Q_WORDS * 4,
                                                     "gemm_qr": tn * QR_WORDS * 4 + TM * tn}.get(template, 0)
                    <= ACC_BUDGET):
                fit.append((TM * TK * tn, TM, TK, tn))
    fit = [f for f in fit if kpad(f[2]) * K_PAD_MAX <= K] or fit
    if fit:
        _, TM, TK, tn = max(fit)
        step = MESH_COLS * TM
        return (math.ceil(M / step) * step, MESH_COLS * tn, math.ceil(K / TK) * TK, TK, TM), None
    return None, (f"N = {N}: even TM 1 / TK 4 needs {core_bytes(1, 4, TN)[0]} B of port buffers "
                  f"(budget {CORE_BUDGET}) for TN {TN}")

def _shape(g, tid) -> tuple:
    return g.tensors[tid].shape


def _create_md(op, k) -> "int | None":
    """tensor id of the k-th memory descriptor among a create's arguments (src, weights, bias, dst)"""
    return op.create[k][1] if op.create[k][0] == "md" else None


def fx_template(g, op) -> tuple:
    """(template, why) for the layer's epilogue on the core: "gemm_q" for the fixed-point requant (an 8-bit
    dst with output scales, a per-channel bias or none, post-ops only ReLU (alpha 0) / clip), "gemm_qr" when it
    also adds one 8-bit residual of the dst's shape (a binary add post-op, alpha > 0); (None, why) otherwise"""
    attr = next((v for k, *v in op.create if k == "attr"), [None])[0] or {}
    dst, bias = _create_md(op, 3), _create_md(op, 2)
    D = g.tensors[dst]
    if D.dt not in ("s8", "u8"):
        return None, f"{D.dt} dst"
    if attr.get("scales") is None:
        return None, "no output scales"
    if bias is not None and int(np.prod(g.tensors[bias].shape)) != D.shape[-1 if "matmul" in op.fn else 1]:
        return None, "bias is not per output channel"
    residual = 0
    post_src = {k[1]: t for k, t in op.exec if isinstance(k, tuple) and k[0] == "post"}
    for j, po in enumerate(attr.get("post", ())):
        if po[0] == "binary":
            src1 = g.tensors[post_src[j]] if j in post_src else None
            if po[1] != "add" or residual or src1 is None or src1.dt not in ("s8", "u8") \
                    or tuple(src1.shape) != tuple(D.shape) or src1.base is not None or not po[3] > 0:
                return None, f"post-op binary {po[1]}: not one 8-bit residual add of the dst's shape"
            residual += 1
        elif po[0] != "eltwise" or not ((po[1] == "relu" and po[2] == 0) or po[1] == "clip"):
            return None, f"post-op {po[1]}"
    return ("gemm_qr" if residual else "gemm_q"), ""


def fx_eligible(g, op) -> tuple:
    """(True, "") when the layer's epilogue runs on the core (gemm_q / gemm_qr), else (False, why)"""
    t, why = fx_template(g, op)
    return t is not None, why


POOL_FNS = ("aiednn_pooling_forward_create", "aiednn_pooling_forward_create_attr")
POOL_TR = 16                       # pool template: rows per core round
POOL_MW_ALIGN, POOL_MW_MAX = 256, 4096   # MW: a multiple of 16 x TR; the runtime loops over chunks of MW


def as_pool(g, i: int) -> tuple:
    """(pool params (MW, KK, TR, AVG, U8), None) for an 8-bit 2-D pooling the pool template runs, (None, why)
    for a pooling it cannot, (None, None) for any other op"""
    op = g.ops[i]
    if op.fn not in POOL_FNS:
        return None, None
    mds = _cvals(op, "md")
    alg = next(iter(_cvals(op, "alg")), "")
    pp = next(iter(_cvals(op, "pool")), None)
    S, D = g.tensors[mds[0]], g.tensors[mds[1]]
    if S.dt not in ("s8", "u8") or D.dt != S.dt:
        return None, f"pooling {S.dt} -> {D.dt}: the pool template is 8-bit to the same type"
    if len(S.shape) != 4 or pp is None or len(pp["kernel"]) != 2:
        return None, "pooling: 2-D only"
    if any(pp.get("dil", (0, 0))):
        return None, "dilated pooling"
    n, c, ho, wo = D.shape
    taps = pp["kernel"][0] * pp["kernel"][1]
    kk = math.ceil(taps / 4) * 4 + 4
    mw = min(math.ceil(n * c * ho * wo / POOL_MW_ALIGN) * POOL_MW_ALIGN, POOL_MW_MAX)
    return (mw, kk, POOL_TR, int(alg != "pooling_max"), int(S.dt == "u8")), None


DW_TP_MIN = 12                     # dwconv_q: taps per channel, padded to a multiple of 4, at least 12
DW_C_ALIGN, DW_C_MAX = 16, 64      # dwconv_q: channels per launch (the runtime loops over channel slices)
DW_M_ALIGN, DW_M_MAX = 64, 1024    # dwconv_q: output pixels per launch (4 x TM; the runtime loops over chunks)
DW_TM = 16


def dw_core_bytes(M: int, C: int, TP: int, TM: int = DW_TM) -> tuple:
    """(port buffer bytes, stack bytes) of a dwconv_q core: double-buffered A {TM, 4TP}, W {C/4, 4TP} and
    Y {TM, C/4}; the stack holds TM x C/4 codes"""
    return 2 * (TM * 4 * TP + C // 4 * 4 * TP + TM * C // 4), TM * C // 4


def as_dwconv(g, i: int) -> tuple:
    """((M, C, TP), None) for a depthwise conv dwconv_q runs (groups == C == O, s8 src without zero point, s8
    weights, gemm_q's on-core epilogue), (None, why) for a depthwise conv it cannot, (None, None) otherwise"""
    op = g.ops[i]
    cp = next((v for k, *v in op.create if k == "conv"), [None])[0]
    if op.fn not in CONV_FNS or not cp or cp.get("groups", 1) == 1:
        return None, None
    src, wei, dst = (g.tensors[_create_md(op, k)] for k in (0, 1, 3))
    n, c, ho, wo = dst.shape
    if not (cp["groups"] == src.shape[1] == c):
        return None, f"grouped convolution (groups {cp['groups']}, {src.shape[1]} -> {c} channels): not depthwise"
    attr = next((v for k, *v in op.create if k == "attr"), [None])[0] or {}
    if src.dt != "s8" or wei.dt != "s8" or attr.get("src_zp"):
        return None, f"depthwise {src.dt} x {wei.dt}{' with a src zero point' if attr.get('src_zp') else ''}: dwconv_q is s8 x s8"
    tmpl, why = fx_template(g, op)
    if tmpl != "gemm_q":
        return None, f"depthwise epilogue: {why or 'residual'} (dwconv_q requantizes, no residual)"
    TP = max(DW_TP_MIN, math.ceil(wei.shape[2] * wei.shape[3] / 4) * 4)
    C = min(math.ceil(c / DW_C_ALIGN) * DW_C_ALIGN, DW_C_MAX)
    M = min(math.ceil(n * ho * wo / DW_M_ALIGN) * DW_M_ALIGN, DW_M_MAX)
    while C > DW_C_ALIGN and dw_core_bytes(M, C, TP)[0] > CORE_BUDGET:
        C -= DW_C_ALIGN
    if dw_core_bytes(M, C, TP)[0] > CORE_BUDGET:
        return None, f"depthwise {wei.shape[2]}x{wei.shape[3]}: TP {TP} does not fit core memory"
    return (M, C, TP), None


#: lut: one instance per model serves every per-channel function of one 8-bit value (the runtime builds the
#: tables with ref: on all 256 codes, so it is exact) and uses the largest M x N present
FROW_SOFTMAX, FROW_LOGSOFTMAX, FROW_SUM, FROW_MEAN, FROW_MAX, FROW_MIN, FROW_SUMDIV = range(1, 8)  # EG_ROW_*
FROW_BUDGET = 24_576               # bytes: a frow instance's double-buffered input strips + output blocks


def frow_params(op: int, n: int) -> tuple:
    """frow<R, L, TR, OP, YB> for rows of n floats (incl. SUMDIV's divisor): L = n rounded up to 16, the largest TR
    whose strips fit FROW_BUDGET"""
    L = max(16, -(-n // 16) * 16)
    YB = 16 if op >= FROW_SUM else 4 * L
    TR = next((t for t in (8, 4, 2, 1) if 2 * t * 4 * L + 2 * t * YB // 4 <= FROW_BUDGET), 1)
    return (128, L, TR, op, YB)


FELT_PARAMS, FBIN_PARAMS = (128, 256, 8), (128, 256, 4)   # feltw / fbinary <R, L, TR, F>: rows per launch, f32 per row
#: the f32 algorithms the library carries an instance of (FVEC's AIE_FVF codes; a per-model plan makes exactly its own):
#: relu, exp, logistic, tanh, gelu tanh, swish, log, gelu erf, sqrt, reciprocal, rsqrt / add, sub, mul, div, max, min
LIBRARY_FELTW = (2, 7, 8, 9, 10, 11, 13, 18, 6, 24, 25)
LIBRARY_FBINARY = (64, 65, 66, 67, 68, 69)
FAFF_PARAMS = (128, 64, 8)          # faffine <R, L, TR, F>: rows per launch, f32 per row
FAFF_AFFINE, FAFF_AFFINE_RELU, FAFF_PRELU = 96, 97, 98   # eager_fmath_ops.cc EG_F_ROW_* (= FVEC AIE_FVF_ROW_*)
#: faffine F the library carries: batch norm, batch norm + ReLU, PReLU per channel; fbinary 98: per-element PReLU
LIBRARY_FAFFINE = (FAFF_AFFINE, FAFF_AFFINE_RELU, FAFF_PRELU)
BIN8_PARAMS = (128, 256, 8)         # bin8 <R, L, TR, OP, Q>: rows per launch, elements per row
#: bin8 (OP, Q): the exact int32 add / sub / mul / max / min (comparisons ride on sub) and the quantized f32
#: add / sub / max / min / mul (mul only with AIEDNN_ATTR_TOLERANCE_1CODE)
LIBRARY_BIN8 = tuple((f, 0) for f in (64, 65, 66, 68, 69)) + tuple((f, 1) for f in (64, 65, 66, 68, 69))
BIN8_EXACT = {"binary_add": 64, "binary_sub": 65, "binary_mul": 66, "binary_max": 68, "binary_min": 69,
              "binary_eq": 65, "binary_ne": 65, "binary_gt": 65, "binary_ge": 65, "binary_lt": 65, "binary_le": 65}
#: frow (op, row floats): softmax / log softmax rows <= 1024 (and <= 256), sum / mean / max / min <= 1024, max pool
#: windows <= 16 taps, average pool windows <= 63 taps (+ the divisor)
LIBRARY_FROW = ((1, 256), (1, 1024), (2, 1024), (3, 1024), (4, 1024), (5, 1024), (6, 1024), (5, 16), (7, 64))
BF16_PARAMS = (64, 64, 128, 32, 16)  # gemm_bf16<M, N, K, TK, TM>: one instance tiles every float GEMM (board-proven)
#: float GEMM-family work: these creates with an f32 / bf16 source reach the AIE through eager:bf16 / bf16x3
FLOAT_GEMM_FNS = ("aiednn_matmul_forward_create", "aiednn_matmul_forward_create_attr",
                  "aiednn_inner_product_forward_create", "aiednn_inner_product_forward_create_attr",
                  "aiednn_convolution_forward_create", "aiednn_convolution_forward_create_attr",
                  "aiednn_deconvolution_forward_create", "aiednn_deconvolution_forward_create_attr")
FLOAT_COMPOSITE_WORDS = ("sdpa", "rnn", "lstm", "gru", "gated_mlp", "linear_attention")
LUT_PARAMS = (256, 128, 32)        # M, N, TM (N in {64, 128}; TM >= N / 4)
LUTW_PARAMS = {4: (128, 64, 4, 4), 2: (128, 128, 4, 2)}   # lutw <R, V, TR, ES> per result bytes (board-proven)
WQ_PARAMS = (128, 64, 512, 32, 16)  # gemm_wq <M, N, K, GK, TM>: GK 32 divides every group size a model uses (32 ..)
RED8_R = 128                       # red8 <R, L, TR, OP, SG>: rows per launch
RED8_OPS = {"reduction_sum": 1, "reduction_mean": 1, "reduction_max": 2, "reduction_min": 3, "reduction_amax": 6}
#: red8 (OP, SG) the library carries: sum (and mean), max, min of s8 / u8, rows of 256
LIBRARY_RED8 = tuple((op, sg) for op in (1, 2, 3) for sg in (1, 0))
STEPS_PARAMS = (128, 64, 2, 3136)  # steps <R, V, TR, S>: S holds the largest step table (511 steps)
#: eltwise algorithms steps serves (monotone per half line: fixed steps:'s steps_alg_ok)
STEPS_ELTWISE = ("eltwise_relu", "eltwise_clip", "eltwise_linear", "eltwise_abs", "eltwise_square", "eltwise_round",
                 "eltwise_floor", "eltwise_ceil", "eltwise_neg", "eltwise_hardsigmoid")
LUT_FAMILIES = ("aiednn_eltwise_forward", "aiednn_prelu_forward", "aiednn_batch_normalization_forward",
                "aiednn_binary", "aiednn_reorder")


QUANT_M, QUANT_N, QUANT_TM = 128, 64, 16   # quant: M a multiple of 4 x TM, N a multiple of 16 with N / 4 <= TM


def as_quant(g, i: int) -> tuple:
    """(quant params (M, N, U8, TM), None) for an f32 -> s8 / u8 identity reorder (the input quantize: no
    permutation, the same shape, a normal scale > 0); (None, None) for anything else"""
    op = g.ops[i]
    if not op.fn.startswith("aiednn_reorder") or any(c[0] == "perm" for c in op.create):
        return None, None
    mds = [c[1] for c in op.create if c[0] == "md"]
    scale = next((c[1] for c in op.create if c[0] == "float"), None)
    if len(mds) != 2 or any(c[0] == "attr" for c in op.create):
        return None, None
    S, D = g.tensors[mds[0]], g.tensors[mds[1]]
    if S.dt != "f32" or D.dt not in ("s8", "u8") or tuple(S.shape) != tuple(D.shape):
        return None, None
    if scale is None or not (np.isfinite(scale) and scale >= np.finfo(np.float32).tiny):
        return None, "quantize with a zero / non-normal scale"
    return (QUANT_M, QUANT_N, int(D.dt == "u8"), QUANT_TM), None


LNORM_R = 128                      # lnorm: rows per launch (the runtime loops over row chunks)
LNORM_TR = (2, 1)                  # rows per core round, largest first (core: 2*TR*4L + 3L + 2*TR*L bytes)
NORM_FUSE_RELU, NORM_RMS = 0x4, 0x8   # aiednn.h AIEDNN_FUSE_NORM_RELU / AIEDNN_RMS_NORM
ATTR_TOLERANCE_1CODE = 0x1         # aiednn.h AIEDNN_ATTR_TOLERANCE_1CODE: an 8-bit dst may be 1 code off


def lnorm_core_bytes(L: int, TR: int) -> int:
    return 2 * TR * 4 * L + 3 * L + 2 * TR * L


def as_lnorm(g, i: int) -> tuple:
    """((RMS, row width), None) for a layer / RMS norm lnorm runs (f32 src; an f32 dst, or an 8-bit dst whose
    attr allows the 1-code tolerance; no fused ReLU), (None, why) for a norm it cannot, (None, None) otherwise"""
    op = g.ops[i]
    if not op.fn.startswith("aiednn_layer_normalization_forward_inference"):
        return None, None
    mds = [c[1] for c in op.create if c[0] == "md" and c[1] is not None]
    S, D = g.tensors[mds[0]], g.tensors[mds[1]]
    flags = next((c[1] for c in op.create if c[0] == "uint"), 0)
    attr = next((v for k, *v in op.create if k == "attr"), [None])[0] or {}
    if S.dt != "f32":
        return None, f"layer norm {S.dt} src: lnorm takes f32"
    if D.dt in ("s8", "u8") and not attr.get("flags", 0) & ATTR_TOLERANCE_1CODE:
        return None, f"layer norm f32 -> {D.dt} without AIEDNN_ATTR_TOLERANCE_1CODE: stays exact on ref:"
    if D.dt not in ("f32", "s8", "u8") or flags & NORM_FUSE_RELU:
        return None, f"layer norm -> {D.dt}{' + fused ReLU' if flags & NORM_FUSE_RELU else ''}"
    return (int(bool(flags & NORM_RMS)), S.shape[-1]), None


ROPE_R = 128                       # rope: rows per launch (the runtime loops over row chunks)
ROPE_TR = (8, 4, 2, 1)             # rows per core round, largest first (core: 2*TR*12D + 2*TR*D bytes)


def as_rope(g, i: int) -> tuple:
    """((D, interleaved), None) for a RoPE rope runs (f32 x / cos / sin; an f32 dst, or an 8-bit dst through the
    _attr form), (None, why) for one it cannot, (None, None) otherwise. D is the head dim exactly: the partner
    element sits at j +- D/2, so it is never padded"""
    op = g.ops[i]
    if not op.fn.startswith("aiednn_rope_create"):
        return None, None
    mds = [g.tensors[c[1]] for c in op.create if c[0] == "md" and c[1] is not None]
    x, cos, sin, dst = mds[:4]
    inter = next((c[1] for c in op.create if c[0] == "int"), 0)
    if any(t.dt != "f32" for t in (x, cos, sin)):
        return None, "rope with quantized x / cos / sin: stays on ref:"
    if dst.dt not in ("f32", "s8", "u8"):
        return None, f"rope -> {dst.dt}"
    D = x.shape[-1]
    if D % 16:
        return None, f"rope head dim {D}: not a multiple of 16"
    return (D, int(bool(inter))), None


DT_BYTES = {"s8": 1, "u8": 1, "f16": 2, "bf16": 2, "s32": 4, "f32": 4}


def lut_candidate(g, op) -> int:
    """the result bytes (1: lut, 2 / 4: lutw) of an eltwise / PReLU / batch norm of an 8-bit source, a binary with
    exactly one broadcast operand, or an identity reorder (no permutation, same shape): what eager:lut can run;
    0 if not one. The other operands (weights, the broadcast side) may be any type"""
    if not any(op.fn.startswith(f) for f in LUT_FAMILIES):
        return 0
    ts = [g.tensors[v] for k, v in ((c[0], c[1] if len(c) > 1 else None) for c in op.create) if k == "md" and v is not None]
    dst = g.tensors[op.out]
    es = DT_BYTES.get(dst.dt, 0)
    full = [t for t in ts if tuple(t.shape) == tuple(dst.shape) and t is not dst]
    if not ts or not es or not full or full[0].dt not in ("s8", "u8") or (es == 1 and any(t.dt not in ("s8", "u8") for t in ts)):
        return 0
    return es if _lut_shape_ok(g, op, ts, dst) else 0


def steps_candidate(g, op) -> bool:
    """an f32 / s32 -> s8 / u8 per-element op eager:steps runs: a reorder, batch norm, PReLU, binary with one
    broadcast operand (add .. le, no attr) or an arithmetic eltwise (no attr)"""
    if not any(op.fn.startswith(f) for f in LUT_FAMILIES):
        return False
    ts = [g.tensors[v] for k, v in ((c[0], c[1] if len(c) > 1 else None) for c in op.create) if k == "md" and v is not None]
    dst = g.tensors[op.out]
    full = [t for t in ts if tuple(t.shape) == tuple(dst.shape) and t is not dst]
    if not full or full[0].dt not in ("f32", "s32") or dst.dt not in ("s8", "u8"):
        return False
    attr = any(c[0] == "attr" for c in op.create)
    alg = next(iter(_cvals(op, "alg")), "")
    if op.fn.startswith("aiednn_eltwise"):
        return not attr and alg in STEPS_ELTWISE
    if op.fn.startswith("aiednn_binary"):
        return not attr and (alg in BIN8_EXACT or alg == "binary_div") and _lut_shape_ok(g, op, ts, dst)
    return _lut_shape_ok(g, op, ts, dst)


def _lut_shape_ok(g, op, ts, dst) -> bool:
    if op.fn.startswith("aiednn_binary"):
        return sum(tuple(t.shape) != tuple(dst.shape) for t in ts[:2]) == 1
    if op.fn.startswith("aiednn_reorder"):
        return not any(c[0] == "perm" for c in op.create) and tuple(ts[0].shape) == tuple(dst.shape)
    return True


def as_gemm(g, i: int) -> tuple:
    """(Gemm, None) for a GEMM-family primitive, (None, reason) when eager mode cannot express it, or
    (None, None) when the op is not a GEMM at all"""
    op = g.ops[i]
    if op.fn not in MATMUL_FNS + IP_FNS + CONV_FNS + DECONV_FNS:
        return None, None
    src, wei, dst = _create_md(op, 0), _create_md(op, 1), _create_md(op, 3)
    if g.tensors[src].dt not in ("s8", "u8") or g.tensors[wei].dt != "s8":
        return None, f"{g.tensors[src].dt} x {g.tensors[wei].dt}: the eager template is int8 x int8"
    if op.fn in DECONV_FNS:
        return None, "deconvolution: no eager template yet"
    S, W, D = _shape(g, src), _shape(g, wei), _shape(g, dst)
    if op.fn in CONV_FNS:
        cp = next(iter(_cvals(op, "conv")))
        if cp.get("groups", 1) != 1:
            return None, f"grouped convolution (groups {cp['groups']}): no eager template yet"
        n, oc, oh, ow = D
        # an fx-eligible conv runs as gemm_q through eager:convq (CONV_Q=False keeps convs on gemm_s8)
        return Gemm(i, op.name, "conv", n * oh * ow, oc, int(np.prod(W[1:])), fx_template(g, op)[0] if CONV_Q else None), None
    if op.fn in MATMUL_FNS and g.tensors[wei].kind != "const" and int(np.prod(W[:-2] or (1,))) > 1:
        return None, "batched matmul of two activations: no eager template yet"
    kind = "inner_product" if op.fn in IP_FNS else "matmul"
    K = int(np.prod(S[1:])) if kind == "inner_product" else S[-1]
    return Gemm(i, op.name, kind, int(np.prod(D[:-1])), D[-1], K, fx_template(g, op)[0]), None


#: per template: (kernel source, reference header, C type of C). The kernel source is a .cc because aiehlc
#: splices only .cc includes into the kernel text; it is included only when the model launches it (aiehlc
#: refuses a template that is never launched). Both are copied flat next to the entry with COMMON_HEADER.
TEMPLATES = {"gemm_s8": ("gemm_s8_eager_kernel.cc", "gemm_s8_eager_ref.h", "int32_t"),
             "gemm_q": ("gemm_q_eager_kernel.cc", "gemm_q_eager_ref.h", "int8_t"),
             "gemm_qr": ("gemm_qr_eager_kernel.cc", "gemm_qr_eager_ref.h", "int8_t"),
             "pool": ("pool_eager_kernel.cc", "pool_eager_ref.h", "int8_t"),
             "dwconv_q": ("dwconv_q_eager_kernel.cc", "dwconv_q_eager_ref.h", "int8_t"),
             "dwconv_s8": ("dwconv_s8_eager_kernel.cc", "dwconv_s8_eager_ref.h", "int32_t"),
             "gemm_wq": ("gemm_wq_eager_kernel.cc", "gemm_wq_eager_ref.h", "int32_t"),
             "lut": ("lut_eager_kernel.cc", "lut_eager_ref.h", "int8_t"),
             "lutw": ("lutw_eager_kernel.cc", "lutw_eager_ref.h", "int8_t"),
             "steps": ("steps_eager_kernel.cc", "steps_eager_ref.h", "int8_t"),
             "red8": ("red8_eager_kernel.cc", "red8_eager_ref.h", "int8_t"),
             "quant": ("quant_eager_kernel.cc", "quant_eager_ref.h", "int8_t"),
             "lnorm": ("lnorm_eager_kernel.cc", "lnorm_eager_ref.h", "int8_t"),
             "rope": ("rope_eager_kernel.cc", "rope_eager_ref.h", "int8_t"),
             "gemm_bf16": ("gemm_bf16_eager_kernel.cc", "gemm_bf16_eager_ref.h", "int32_t"),
             "feltw": ("feltw_eager_kernel.cc", "felt_eager_ref.h", "int8_t"),
             "fbinary": ("fbinary_eager_kernel.cc", "felt_eager_ref.h", "int8_t"),
             "faffine": ("faffine_eager_kernel.cc", "felt_eager_ref.h", "int8_t"),
             "bin8": ("bin8_eager_kernel.cc", "bin8_eager_ref.h", "int8_t"),
             "frow": ("frow_eager_kernel.cc", "frow_eager_ref.h", "int8_t")}
#: headers / spliced sources every template that uses the f32 math needs next to the entry
FMATH_FILES = ("eager_fmath.h", "eager_fmath_ops.cc")
GEMM_TEMPLATES = ("gemm_s8", "gemm_q", "gemm_qr")
COMMON_HEADER = "eager_common.h"
KERNEL_HEADER = TEMPLATES["gemm_s8"][0]
ENTRY_FILE, REF_FILE = "eager_instances.cc", "eager_instances_ref.cc"   # the references are C++ (eager_fmath_ops.cc)


def _used(p: EagerPlan) -> list:
    return [t for t in TEMPLATES if any(i.template == t for i in p.instances)]


def _spec(i: Instance) -> str:
    """the GEMM block spec form (gemm_uk.cc's gemm_blk_s8s32), so kernel_lib and its lint read it too;
    gemm_q's inputs carry one TK chunk past K (A: zeros, W: the requant words) and its C is 8-bit"""
    k = i.K + _extra_chunks(i) * i.TK
    c = "int32" if i.template == "gemm_s8" else "int8 "
    return (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
            f'target    vek280\\nin    int8  [{i.M},{k}]  src\\nparam int8  [{i.N},{k}]  weight\\n'
            f'out   {c} [{i.M},{i.N}]  dst\\n"')


def _extra_chunks(i: Instance) -> int:
    """TK chunks the inputs carry past K: gemm_q 1 (requant words in W), gemm_qr 1 + ceil(N / TK) (the
    residual codes ride in A's chunks), gemm_s8 none"""
    return {"gemm_q": 1, "gemm_qr": 1 + math.ceil(i.N / i.TK)}.get(i.template, 0)


def _call_args(i: Instance) -> str:
    return f"{i.M}, {i.N}, {i.K}" + (f", {i.TK}" if i.template != "gemm_s8" else "")


def _emit(i: Instance) -> dict:
    """per template: the launch's parameters, the kernel launch, the entry and reference calls, the spec"""
    if i.template == "red8":
        R, L, TR, OP, SG = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        pool\\nmatch     conv2d\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{R},{L}]  src\\nout   int8  [{R},16]  dst\\n"')
        return dict(sig="int8_t *X, int8_t *Y", launch=f"red8<{R}, {L}, {TR}, {OP}, {SG}><<<mesh>>>(X, Y)",
                    entry=f"{R}, {L}", ref=f"{R}, {L}, {OP}, {SG}", spec=spec)
    if i.template in ("lutw", "steps"):
        R, V, TR, P = i.params
        ld, yw = (256 * P + V, V * P) if i.template == "lutw" else (P + 4 * V, V)
        spec = (f'"kernel    eager_{i.name}\\nop        pool\\nmatch     conv2d\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{R},{ld}]  src\\nout   int8  [{R},{yw}]  dst\\n"')
        return dict(sig="int8_t *X, int8_t *Y", launch=f"{i.template}<{R}, {V}, {TR}, {P}><<<mesh>>>(X, Y)",
                    entry=f"{R}, {V}, {P}", ref=f"{R}, {V}, {P}", spec=spec)
    if i.template == "lut":
        M, N, TM = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{M},{256 + N}]  src\\nparam int8  [{N},{256 + N}]  weight\\n'
                f'out   int8  [{M},{N}]  dst\\n"')
        return dict(sig="int8_t *A, int8_t *W, int8_t *Y", launch=f"lut<{M}, {N}, {TM}><<<mesh>>>(A, W, Y)",
                    entry=f"{M}, {N}", ref=f"{M}, {N}", spec=spec)
    if i.template == "frow":
        R, L, TR, OP, YB = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{R},{4 * L}]  src\\nparam int8  [4,16]  weight\\n'
                f'out   int8  [{R},{YB}]  dst\\n"')
        return dict(sig="int8_t *X, int8_t *P, int8_t *Y", launch=f"frow<{R}, {L}, {TR}, {OP}, {YB}><<<mesh>>>(X, P, Y)",
                    entry=f"{R}, {L}, {int(OP >= FROW_SUM)}", ref=f"{R}, {L}, {OP}", spec=spec)
    if i.template == "bin8":
        R, L, TR, OP, Q = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{R},{2 * L}]  src\\nparam int8  [4,32]  weight\\n'
                f'out   int8  [{R},{4 * L}]  dst\\n"')
        return dict(sig="int8_t *XY, int8_t *P, int8_t *Z", launch=f"bin8<{R}, {L}, {TR}, {OP}, {Q}><<<mesh>>>(XY, P, Z)",
                    entry=f"{R}, {L}", ref=f"{R}, {L}, {OP}, {Q}", spec=spec)
    if i.template in ("faffine", "fbinary"):   # one packed input row: [x | c0 c1 - -] / [x | y]
        R, L, TR, F = i.params
        w = 4 * L + 16 if i.template == "faffine" else 8 * L
        spec = (f'"kernel    eager_{i.name}\\nop        pool\\nmatch     conv2d\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{R},{w}]  src\\nout   int8  [{R},{4 * L}]  dst\\n"')
        return dict(sig="int8_t *X, int8_t *Y", launch=f"{i.template}<{R}, {L}, {TR}, {F}><<<mesh>>>(X, Y)",
                    entry=f"{R}, {L}", ref=f"{R}, {L}, {F}", spec=spec)
    if i.template == "feltw":
        R, L, TR, F = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{R},{4 * L}]  src\\nparam int8  [4,16]  weight\\n'
                f'out   int8  [{R},{4 * L}]  dst\\n"')
        return dict(sig="int8_t *X, int8_t *P, int8_t *Y", launch=f"feltw<{R}, {L}, {TR}, {F}><<<mesh>>>(X, P, Y)",
                    entry=f"{R}, {L}", ref=f"{R}, {L}, {F}", spec=spec)
    if i.template == "gemm_bf16":
        M, N, K, TK, TM = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{M},{2 * K}]  src\\nparam int8  [{N},{2 * K}]  weight\\n'
                f'out   int32 [{M},{N}]  dst\\n"')
        return dict(sig="int8_t *A, int8_t *W, int32_t *C", launch=f"gemm_bf16<{M}, {N}, {K}, {TK}, {TM}><<<mesh>>>(A, W, C)",
                    entry=f"{M}, {N}, {K}", ref=f"{M}, {N}, {K}, {TK}", spec=spec)
    if i.template == "rope":
        R, D, TR, IN = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        pool\\nmatch     conv2d\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{R},{12 * D}]  src\\nout   int8  [{R},{4 * D}]  dst\\n"')
        return dict(sig="int8_t *X, int8_t *Y", launch=f"rope<{R}, {D}, {TR}, {IN}><<<mesh>>>(X, Y)",
                    entry=f"{R}, {D}", ref=f"{R}, {D}, {IN}", spec=spec)
    if i.template == "lnorm":
        R, L, TR, RMS = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{R},{4 * L}]  src\\nparam int8  [12,{L}]  weight\\n'
                f'out   int8  [{R},{4 * L}]  dst\\n"')
        return dict(sig="int8_t *X, int8_t *P, int8_t *Y", launch=f"lnorm<{R}, {L}, {TR}, {RMS}><<<mesh>>>(X, P, Y)",
                    entry=f"{R}, {L}", ref=f"{R}, {L}, {RMS}", spec=spec)
    if i.template == "quant":
        M, N, U8, TM = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{M},{4 * N}]  src\\nparam int8  [{N},{4 * N}]  weight\\n'
                f'out   int8  [{M},{N}]  dst\\n"')
        return dict(sig="int8_t *A, int8_t *W, int8_t *Y", launch=f"quant<{M}, {N}, {U8}, {TM}><<<mesh>>>(A, W, Y)",
                    entry=f"{M}, {N}", ref=f"{M}, {N}, {U8}", spec=spec)
    if i.template == "gemm_wq":
        M, N, K, GK, TM = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{M},{K}]  src\\nparam int8  [{N},{K}]  weight\\n'
                f'out   int32 [{M},{N}]  dst\\n"')
        return dict(sig="int8_t *A, int8_t *W, int32_t *C", launch=f"gemm_wq<{M}, {N}, {K}, {GK}, {TM}><<<mesh>>>(A, W, C)",
                    entry=f"{M}, {N}, {K}", ref=f"{M}, {N}, {K}, {GK}", spec=spec)
    if i.template == "dwconv_s8":
        M, C, TP, TM = i.params
        w = C // 4 * 4 * TP
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{M},{w}]  src\\nparam int8  [{C},{w}]  weight\\n'
                f'out   int32 [{M},{C}]  dst\\n"')
        return dict(sig="int8_t *A, int8_t *W, int32_t *Y", launch=f"dwconv_s8<{M}, {C}, {TP}, {TM}><<<mesh>>>(A, W, Y)",
                    entry=f"{M}, {C}, {TP}", ref=f"{M}, {C}, {TP}", spec=spec)
    if i.template == "dwconv_q":
        M, C, TP, TM = i.params
        w = C // 4 * 4 * TP
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{M},{w}]  src\\nparam int8  [{C},{w}]  weight\\n'
                f'out   int8  [{M},{C}]  dst\\n"')
        return dict(sig="int8_t *A, int8_t *W, int8_t *Y", launch=f"dwconv_q<{M}, {C}, {TP}, {TM}><<<mesh>>>(A, W, Y)",
                    entry=f"{M}, {C}, {TP}", ref=f"{M}, {C}, {TP}", spec=spec)
    if i.template == "pool":
        MW, KK, TR, AVG, U8 = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        pool\\nmatch     conv2d\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{MW},{KK}]  src\\nparam int8  [16,{4 * KK}]  weight\\n'
                f'out   int8  [{MW},4]  dst\\n"')
        return dict(sig="int8_t *X, int8_t *Z, int8_t *Y", launch=f"pool<{MW}, {KK}, {TR}, {AVG}, {U8}><<<mesh>>>(X, Z, Y)",
                    entry=f"{MW}, {KK}", ref=f"{MW}, {KK}, {AVG}, {U8}", spec=spec)
    return dict(sig=f"int8_t *A, int8_t *W, {TEMPLATES[i.template][2]} *C",
                launch=f"{i.template}<{i.M}, {i.N}, {i.K}, {i.TK}, {i.TM}><<<mesh>>>(A, W, C)",
                entry=_call_args(i), ref=_call_args(i), spec=_spec(i))


def entry_source(p: EagerPlan) -> str:
    """the aiehlc entry of a model's instances (the E3 contract): the templates it launches, then per
    instance a launch, its AIEK_ENTRY / AIEK_SPEC (exact padded dims) and AIEK_REFERENCE"""
    out = ["// Generated by aiegraph/dnn_eager.py: eager mode's kernel instances. Do not edit.",
           f"// model {p.model}: {len(p.instances)} instance(s) for {sum(len(i.ops) for i in p.instances)} layer(s)",
           '#include "aie_user_kernel.h"',                                  # kernel_lib lint L004
           f"#define HW_ROWS {p.mesh_rows} /* the mesh: the kernels' row split (eager_common.h) */",
           f"#define HW_COLS {MESH_COLS}"]
    out += [f'#include "{TEMPLATES[t][0]}"' for t in _used(p)] + [""]
    for i in p.instances:
        e = _emit(i)
        out += [f"// {', '.join(o['name'] for o in i.ops)}",
                f"static void {i.name}_launch({e['sig']}) {{",
                "    aieSetDevice(0);",
                "    aieArray device;",
                # partition() passes the first kernel's routing: take the partition as it is, so only
                # __aie_launch's own-routing check can re-initialize it (a relaunch then needs none)
                "    __Runtime_partition_reuse_next();",
                "    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);",
                f"    {e['launch']};",
                "}",
                f"AIEK_ENTRY(eager_{i.name}) {{ return eager_{i.template}_entry(args, num_args, {e['entry']}, "
                f"{i.name}_launch); }}",
                f"AIEK_SPEC(eager_{i.name}, {e['spec']});",   # a static array definition
                f"AIEK_REFERENCE(eager_{i.name}) {{ return eager_{i.template}_ref(args, num_args, {e['ref']}); }}",
                ""]
    return "\n".join(out)


def ref_source(p: EagerPlan) -> str:
    """the instances' references for workstation builds (dnn_build target x86 compiles ``reference`` files
    as C): the templates' own reference headers, which are C"""
    out = ["// Generated by aiegraph/dnn_eager.py: x86 references of the eager instances. Do not edit.",
           '#include "aie_user_kernel.h"'] + [f'#include "{TEMPLATES[t][1]}"' for t in _used(p)] + [""]
    out += [f"AIEK_REFERENCE(eager_{i.name}) {{ return eager_{i.template}_ref(args, num_args, {_emit(i)['ref']}); }}"
            for i in p.instances]
    return "\n".join(out) + "\n"


def write_sources(p: EagerPlan, out) -> Path:
    """writes the entry, its x86 reference, the plan JSON and the used templates' sources into *out*;
    returns the entry"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / REF_FILE).write_text(ref_source(p))
    p.write(out / "eager_instances.json")
    entry = out / ENTRY_FILE
    entry.write_text(entry_source(p))
    # spliced .cc templates resolve against the entry dir (lint L020); their headers come along
    eager_dir = Path(__file__).resolve().parents[1] / "aietensorop" / "eager"
    fm = list(FMATH_FILES) if set(_used(p)) & {"feltw", "fbinary", "frow", "faffine", "bin8"} else []
    for f in dict.fromkeys([COMMON_HEADER] + fm + [f for t in _used(p) for f in TEMPLATES[t][:2]]):
        (out / f).write_text((eager_dir / f).read_text())
    return entry


#: f32 eltwise algorithm -> FVEC AIE_FVF code (the runtime's fvec_eltwise_fn); pow / soft_relu depend on their
#: parameters, which the plan does not check: they stay on ref:
FELT_FN = {"eltwise_relu": 2, "eltwise_elu": 12, "eltwise_selu": 20, "eltwise_gelu_tanh": 10, "eltwise_gelu_erf": 18,
           "eltwise_logistic": 8, "eltwise_tanh": 9, "eltwise_swish": 11, "eltwise_mish": 15, "eltwise_hardswish": 16,
           "eltwise_hardsigmoid": 17, "eltwise_clip": 3, "eltwise_linear": 1, "eltwise_abs": 4, "eltwise_exp": 7,
           "eltwise_log": 13, "eltwise_sqrt": 6, "eltwise_square": 5, "eltwise_reciprocal": 24, "eltwise_rsqrt": 25,
           "eltwise_erf": 19, "eltwise_sign": 23, "eltwise_softsign": 21, "eltwise_log_sigmoid": 22,
           "eltwise_neg": 1, "eltwise_identity": 1, "eltwise_round": 26, "eltwise_floor": 27, "eltwise_ceil": 28,
           "eltwise_logical_not": 29, "eltwise_sin": 31, "eltwise_cos": 32, "eltwise_tan": 33}
FBIN_FN = {"binary_add": 64, "binary_sub": 65, "binary_mul": 66, "binary_div": 67, "binary_max": 68, "binary_min": 69,
           "binary_eq": 70, "binary_ne": 71, "binary_gt": 72, "binary_ge": 73, "binary_lt": 74, "binary_le": 75,
           "binary_logical_and": 76, "binary_logical_or": 77, "binary_logical_xor": 78, "binary_add_square": 79}


def _cvals(op, kind) -> list:
    """the values of op's create entries of kind (an entry may carry no value)"""
    return [c[1] for c in op.create if c[0] == kind and len(c) > 1]


def as_fpoint(g, op) -> "tuple | None":
    """(template, F) for an f32 eltwise / binary without attr (eager:feltw / eager:fbinary), else None"""
    if op.fn not in ("aiednn_eltwise_forward_create", "aiednn_binary_create"):
        return None
    mds = [v for v in _cvals(op, "md") if v is not None]
    if not mds or any(g.tensors[t].dt != "f32" for t in mds) or g.tensors[op.out].dt != "f32":
        return None
    alg = next(iter(_cvals(op, "alg")), "")
    table, tmpl = (FELT_FN, "feltw") if op.fn.startswith("aiednn_eltwise") else (FBIN_FN, "fbinary")
    return (tmpl, table[alg]) if alg in table else None


def as_faffine(g, op) -> "tuple | None":
    """(template, F) for an f32 inference batch norm (faffine 96 / 97 with the fused ReLU) or f32 PReLU (faffine 98
    for a slope per contiguous block of >= 16, else fbinary 98 per element), else None"""
    if op.fn not in ("aiednn_batch_normalization_forward_inference_create", "aiednn_prelu_forward_create"):
        return None
    mds = [v for v in _cvals(op, "md") if v is not None]
    if not mds or any(g.tensors[t].dt != "f32" for t in mds) or g.tensors[op.out].dt != "f32":
        return None
    if op.fn.startswith("aiednn_batch"):
        flags = next(iter(_cvals(op, "uint")), 0)
        return ("faffine", FAFF_AFFINE_RELU if flags & NORM_FUSE_RELU else FAFF_AFFINE)
    w, d = list(g.tensors[mds[1]].shape), list(g.tensors[op.out].shape)
    sp = 1
    for i in range(len(d) - 1, -1, -1):          # the runtime's fvec_bcast_block: trailing dims the slope broadcasts
        if len(w) != len(d) or w[i] != 1:
            break
        sp *= d[i]
    return ("faffine", FAFF_PRELU) if sp >= 16 else ("fbinary", FAFF_PRELU)


def as_bin8(g, op) -> "tuple | None":
    """bin8 params for a binary of two 8-bit tensors: no attr (exact int32, Q 0) or the quantized form (Q 1)"""
    if op.fn not in ("aiednn_binary_create", "aiednn_binary_create_attr"):
        return None
    mds = [v for v in _cvals(op, "md") if v is not None]
    if len(mds) < 2 or any(g.tensors[t].dt not in ("s8", "u8") for t in mds[:2]):
        return None
    alg = next(iter(_cvals(op, "alg")), "")
    if op.fn == "aiednn_binary_create":   # one operand broadcast along one axis: lut serves it (the runtime's rule)
        return BIN8_PARAMS + (BIN8_EXACT[alg], 0) if alg in BIN8_EXACT and not lut_candidate(g, op) else None
    attr = next(iter(_cvals(op, "attr")), None) or {}
    qop = {"binary_add": 64, "binary_sub": 65, "binary_max": 68, "binary_min": 69}
    if alg == "binary_mul" and attr.get("flags", 0) & ATTR_TOLERANCE_1CODE:
        qop["binary_mul"] = 66
    return BIN8_PARAMS + (qop[alg], 1) if alg in qop else None


def red8_params(op_code: int, sg: int, n: int) -> tuple:
    """red8<R, L, TR, OP, SG> for segments of n: L = n rounded up to 16 (at most 1024: longer segments are cut into
    rows whose partials the host combines), TR the largest whose ping / pong strips stay within 8 KB"""
    L = min(1024, max(16, -(-n // 16) * 16))
    TR = next(t for t in (8, 4, 2, 1) if 2 * t * L <= 8192)
    return (RED8_R, L, TR, op_code, sg)


def as_wq(g, op) -> bool:
    """an s8 / u8 x s4 / u4 matmul with weight group scales (attr wei_scales): eager:wq on gemm_wq"""
    if not op.fn.startswith("aiednn_matmul_forward_create_attr"):
        return False
    attr = next(iter(_cvals(op, "attr")), None) or {}
    try:
        src, wei = g.tensors[_create_md(op, 0)].dt, g.tensors[_create_md(op, 1)].dt
    except Exception:
        return False
    return bool(attr.get("wei_scales")) and src in ("s8", "u8") and wei in ("s4", "u4")


def as_red8(g, op) -> "tuple | None":
    """red8 params for a sum / mean / max / min / amax reduction of an s8 / u8 tensor (no attr)"""
    if not op.fn.startswith("aiednn_reduction_create") or any(c[0] == "attr" for c in op.create):
        return None
    mds = [v for v in _cvals(op, "md") if v is not None]
    if not mds or g.tensors[mds[0]].dt not in ("s8", "u8"):
        return None
    alg = next(iter(_cvals(op, "alg")), "")
    if alg not in RED8_OPS:
        return None
    n = int(np.prod(g.tensors[mds[0]].shape)) // max(1, int(np.prod(g.tensors[op.out].shape)))
    return red8_params(RED8_OPS[alg], int(g.tensors[mds[0]].dt == "s8"), n)


def as_frow(g, op) -> "tuple | None":
    """frow params for an f32 softmax (last axis or any), sum / mean / max / min reduction or 2-D f32 pooling"""
    mds = [v for v in _cvals(op, "md") if v is not None]
    if not mds or any(g.tensors[t].dt != "f32" for t in mds) or g.tensors[op.out].dt != "f32":
        return None
    if any(c[0] == "attr" for c in op.create):
        return None
    alg = next(iter(_cvals(op, "alg")), "")
    src = list(g.tensors[mds[0]].shape)
    if op.fn.startswith("aiednn_softmax_forward_create") and alg in ("softmax_accurate", "softmax_log"):
        axis = next(iter(_cvals(op, "axis") + _cvals(op, "i")), len(src) - 1)
        n = src[axis] if isinstance(axis, int) and 0 <= axis < len(src) else src[-1]
        return frow_params(FROW_LOGSOFTMAX if alg == "softmax_log" else FROW_SOFTMAX, n)
    if op.fn.startswith("aiednn_reduction_create"):
        ops = {"reduction_sum": FROW_SUM, "reduction_mean": FROW_MEAN, "reduction_max": FROW_MAX,
               "reduction_min": FROW_MIN}
        if alg not in ops:
            return None
        n = int(np.prod(src)) // int(np.prod(g.tensors[op.out].shape))
        return frow_params(ops[alg], n)
    if op.fn.startswith("aiednn_pooling_forward_create") and len(src) == 4:
        pp = next(iter(_cvals(op, "pool")), None)
        kern = pp.get("kernel", (1, 1)) if isinstance(pp, dict) else (1, 1)
        taps = int(kern[0]) * int(kern[1])
        return frow_params(FROW_MAX, taps) if alg == "pooling_max" else frow_params(FROW_SUMDIV, taps + 1)
    return None


def float_gemm_candidate(g, op) -> bool:
    """a float GEMM-family op (f32 / bf16 matmul, inner product, conv, deconv) or an f32 composite built on a
    float GEMM (attention, RNN, gated MLP, linear attention): served by the gemm_bf16 instance"""
    try:
        src = g.tensors[_create_md(op, 0)].dt
    except Exception:
        return False
    if src not in ("f32", "bf16"):
        return False
    if op.fn in FLOAT_GEMM_FNS:
        return g.tensors[_create_md(op, 1)].dt in ("f32", "bf16")
    return any(w in op.fn for w in FLOAT_COMPOSITE_WORDS)


#: per non-GEMM template: (index of the row extent split over the mesh rows, index of its per-round tile, rows
#: per tile unit) -- the row extent must be a multiple of HW_ROWS x tile x unit (pool: 4 windows per X row)
ROW_SPLIT = {"pool": (0, 2, 4), "dwconv_q": (0, 3, 1), "dwconv_s8": (0, 3, 1), "lut": (0, 2, 1), "quant": (0, 3, 1),
             "gemm_bf16": (0, 4, 1), "gemm_wq": (0, 4, 1), "lnorm": (0, 2, 1), "rope": (0, 2, 1), "frow": (0, 2, 1),
             "feltw": (0, 2, 1), "fbinary": (0, 2, 1), "faffine": (0, 2, 1), "bin8": (0, 2, 1), "red8": (0, 2, 1),
             "lutw": (0, 2, 1), "steps": (0, 2, 1)}
MESH_ROWS = (4, 6, 8)              # eager_common.h partitions (HW_ROWS)


def fit_mesh_rows(p: EagerPlan, rows: int) -> EagerPlan:
    """*p* for an HW_ROWS = *rows* mesh: every instance's row extent padded to a multiple of rows x its tile
    (padding is exact: the runtime zero-fills and drops the extra rows; the runtime already tiles any layer over
    any instance), instances that become equal merged"""
    if rows not in MESH_ROWS:
        raise AiegraphError("dnn_eager", f"mesh rows {rows}", hint=f"one of {MESH_ROWS} (eager_common.h partitions)")
    by_name = {}
    for i in p.instances:
        if i.template in GEMM_TEMPLATES:
            step = rows * i.TM
            j = Instance(math.ceil(i.M / step) * step, i.N, i.K, i.TK, i.TM, i.template, i.params, list(i.ops))
        else:
            r, t, u = ROW_SPLIT[i.template]
            step = rows * i.params[t] * u
            prm = list(i.params)
            prm[r] = math.ceil(prm[r] / step) * step
            j = Instance(i.M, i.N, i.K, i.TK, i.TM, i.template, tuple(prm), list(i.ops))
        if j.name in by_name:
            by_name[j.name].ops += j.ops
        else:
            by_name[j.name] = j
    return EagerPlan(p.model, list(by_name.values()), p.refused, p.gemm_layers, p.other_layers, rows)


def plan(g, model: "str | None" = None, mesh_rows: int = 4) -> EagerPlan:
    """the instance list of AGraph *g*: every GEMM-family layer either joins the instance of its padded
    shape or is refused with a reason"""
    by_key, refused, seen, other = {}, [], 0, 0
    for i, op in enumerate(g.ops):
        fr = as_frow(g, op)
        if fr is not None:                   # f32 softmax / reduction / pooling: one instance per (op, row width)
            other += 1
            inst = by_key.setdefault(("frow",) + fr, Instance(0, 0, 0, 0, 0, "frow", fr))
            inst.ops.append(dict(op=i, name=op.name, kind="frow"))
            continue
        if as_wq(g, op):                     # grouped weight-only quantization: the one gemm_wq instance
            other += 1
            inst = by_key.setdefault(("gemm_wq",) + WQ_PARAMS, Instance(0, 0, 0, 0, WQ_PARAMS[-1], "gemm_wq", WQ_PARAMS))
            inst.ops.append(dict(op=i, name=op.name, kind="wq"))
            continue
        r8 = as_red8(g, op)
        if r8 is not None:                   # 8-bit reductions: one instance per (op, signedness, row width)
            other += 1
            inst = by_key.setdefault(("red8",) + r8, Instance(0, 0, 0, 0, 0, "red8", r8))
            inst.ops.append(dict(op=i, name=op.name, kind="red8"))
            continue
        b8 = as_bin8(g, op)
        if b8 is not None:                   # two 8-bit tensors: one instance per (op, exact / quantized)
            other += 1
            inst = by_key.setdefault(("bin8",) + b8, Instance(0, 0, 0, 0, 0, "bin8", b8))
            inst.ops.append(dict(op=i, name=op.name, kind="bin8"))
            continue
        fp = as_fpoint(g, op) or as_faffine(g, op)
        if fp is not None:                   # f32 pointwise: one instance per algorithm, compiled in
            other += 1
            params = {"feltw": FELT_PARAMS, "fbinary": FBIN_PARAMS, "faffine": FAFF_PARAMS}[fp[0]] + (fp[1],)
            inst = by_key.setdefault((fp[0],) + params, Instance(0, 0, 0, 0, 0, fp[0], params))
            inst.ops.append(dict(op=i, name=op.name, kind=fp[0]))
            continue
        if float_gemm_candidate(g, op):      # one gemm_bf16 instance tiles every float GEMM (eager:bf16 / bf16x3)
            seen += 1
            inst = by_key.setdefault(("gemm_bf16",) + BF16_PARAMS,
                                     Instance(0, 0, 0, 0, BF16_PARAMS[-1], "gemm_bf16", BF16_PARAMS))
            inst.ops.append(dict(op=i, name=op.name, kind="float_gemm"))
            continue
        pparams, why = as_pool(g, i)
        if pparams is not None:
            other += 1
            inst = by_key.setdefault(("pool",) + pparams, Instance(0, 0, 0, 0, 0, "pool", pparams))
            inst.ops.append(dict(op=i, name=op.name, kind="pool"))
            continue
        if why is not None:
            refused.append((op.name, why))
            continue
        rp, why = as_rope(g, i)
        if rp is not None:                    # one instance per (head dim, interleaved)
            D, inter = rp
            TR = next((t for t in ROPE_TR if 2 * t * 12 * D + 2 * t * D <= CORE_BUDGET), None)
            if TR is None:
                refused.append((op.name, f"rope head dim {D}: no TR fits core memory"))
                continue
            other += 1
            prm = (ROPE_R, D, TR, inter)
            inst = by_key.setdefault(("rope",) + prm, Instance(0, 0, 0, 0, 0, "rope", prm))
            inst.ops.append(dict(op=i, name=op.name, kind="rope"))
            continue
        if why is not None:
            refused.append((op.name, why))
            continue
        ln, why = as_lnorm(g, i)
        if ln is not None:                    # one instance per RMS kind, L = the widest row (rounded to 16)
            other += 1
            rms, width = ln
            old = next((k for k in by_key if k[0] == "lnorm" and k[4] == rms), None)
            inst = by_key.pop(old) if old else Instance(0, 0, 0, 0, 0, "lnorm", (LNORM_R, 0, 0, rms))
            L = max(inst.params[1], math.ceil(width / 16) * 16)
            TR = next((t for t in LNORM_TR if lnorm_core_bytes(L, t) <= CORE_BUDGET), None)
            if TR is None:
                refused.append((op.name, f"layer norm row {width}: no TR fits core memory"))
                if old:
                    by_key[old] = inst
                continue
            inst.params = (LNORM_R, L, TR, rms)
            by_key[("lnorm",) + inst.params] = inst
            inst.ops.append(dict(op=i, name=op.name, kind="lnorm"))
            continue
        if why is not None:
            refused.append((op.name, why))
            continue
        qp, why = as_quant(g, i)
        if qp is not None:                    # one instance per dst type serves every input quantize
            other += 1
            inst = by_key.setdefault(("quant",) + qp, Instance(0, 0, 0, 0, QUANT_TM, "quant", qp))
            inst.ops.append(dict(op=i, name=op.name, kind="quant"))
            continue
        if steps_candidate(g, op):
            other += 1
            inst = by_key.setdefault(("steps",) + STEPS_PARAMS, Instance(0, 0, 0, 0, 0, "steps", STEPS_PARAMS))
            inst.ops.append(dict(op=i, name=op.name, kind="steps"))
            continue
        es = lut_candidate(g, op)
        if es:
            other += 1
            t, prm = ("lut", LUT_PARAMS) if es == 1 else ("lutw", LUTW_PARAMS[es])
            inst = by_key.setdefault((t,) + prm, Instance(0, 0, 0, 0, prm[2], t, prm))
            inst.ops.append(dict(op=i, name=op.name, kind=t))
            continue
        dw, why = as_dwconv(g, i)
        if dw is not None:                    # one instance per TP (the runtime takes the smallest TP that
            other += 1                        # covers a layer), with the largest M and C of its layers
            M, C, TP = dw
            old = next((k for k in by_key if k[0] == "dwconv_q" and k[3] == TP), None)
            inst = by_key.pop(old) if old else Instance(0, 0, 0, 0, DW_TM, "dwconv_q", ())
            inst.params = (max(M, inst.params[0] if inst.params else 0), max(C, inst.params[1] if inst.params else 0),
                           TP, DW_TM)
            by_key[("dwconv_q",) + inst.params] = inst
            inst.ops.append(dict(op=i, name=op.name, kind="dwconv"))
            continue
        if why is not None and why.startswith("depthwise epilogue"):   # int32 sums on the core, host epilogue
            other += 1
            wei = g.tensors[_create_md(op, 1)]
            taps = wei.shape[2] * wei.shape[3]
            TP = next(t for t in (12, 28, 52, max(4, math.ceil(taps / 4) * 4)) if t >= taps)
            prm = (128, 64, TP, DW_TM) if TP <= 12 else (64, 32, TP, DW_TM)
            inst = by_key.setdefault(("dwconv_s8",) + prm, Instance(0, 0, 0, 0, DW_TM, "dwconv_s8", prm))
            inst.ops.append(dict(op=i, name=op.name, kind="dwconv_s8"))
            continue
        if why is not None:
            refused.append((op.name, why))
            continue
        gm, why = as_gemm(g, i)
        if gm is None and why is None:
            continue
        seen += 1
        if gm is None:
            refused.append((op.name, why))
            continue
        tmpl, args = "gemm_s8", None
        if gm.tmpl:                           # on-core epilogue when its template fits, else int32 + host epilogue
            args, _ = pick(gm.M, gm.N, gm.K, gm.tmpl)
            tmpl = gm.tmpl if args else tmpl
        if args is None:
            args, why = pick(gm.M, gm.N, gm.K)
        if args is None:
            refused.append((op.name, why))
            continue
        inst = by_key.setdefault((tmpl,) + args, Instance(*args, template=tmpl))
        inst.ops.append(dict(op=gm.op, name=gm.name, kind=gm.kind, M=gm.M, N=gm.N, K=gm.K))
    if not seen:
        raise AiegraphError("dnn_eager", f"{model or g.name}: no matmul / inner product / convolution to run eagerly")
    return fit_mesh_rows(EagerPlan(model or g.name, list(by_key.values()), refused, seen, other), mesh_rows)


#: the premade template library: a fixed instance set that serves any model with no per-model generation (the
#: runtime runs every layer as slices / chunks / tiles of whatever instances are linked, zero padding being exact,
#: and picks the one needing the fewest launches). GEMMs: (template, M, N, K) through :func:`pick`; a K past
#: every gemm_q[r] instance runs on gemm_s8 with int32 K sums on the host. Non-GEMM templates: their params.
#: Shapes are the board-proven ones (example/aiednn/eager/*_eager.cc) where one exists.
LIBRARY = (
    ("gemm_s8", 128, 64, 256),                          # any int8 GEMM (tiled, host epilogue)
    ("gemm_q", 128, 64, 256), ("gemm_q", 128, 64, 1152), ("gemm_q", 128, 64, 4608),
    ("gemm_qr", 128, 64, 1152), ("gemm_qr", 128, 64, 4608),
    ("pool", 1024, 16, 16, 0, 0), ("pool", 1024, 16, 16, 0, 1),   # max <= 3x3, s8 / u8
    ("pool", 1024, 56, 16, 1, 0), ("pool", 1024, 56, 16, 1, 1),   # average <= 7x7 (resnet's global pool)
    ("dwconv_q", 128, 64, 12, 16), ("dwconv_q", 64, 32, 28, 16),  # depthwise <= 3x3, <= 5x5
    ("dwconv_q", 64, 32, 52, 16),                                  # depthwise <= 7x7
    ("dwconv_s8", 128, 64, 12, 16), ("dwconv_s8", 64, 32, 28, 16), # depthwise to int32 (host epilogue: s32 /
    ("dwconv_s8", 64, 32, 52, 16),                                 # f32 dst, float scales, residual)
    ("lut",) + LUT_PARAMS,                                         # every 8-bit per-channel function
    ("lutw",) + LUTW_PARAMS[4], ("lutw",) + LUTW_PARAMS[2],       # ... into a 32- / 16-bit dst
    ("steps",) + STEPS_PARAMS,                                     # f32 / s32 -> 8-bit monotone ops (step tables)
) + tuple(("red8",) + red8_params(op, sg, 256) for op, sg in LIBRARY_RED8) + (   # 8-bit sum / mean / max / min
    ("quant", 128, 64, 0, 16), ("quant", 128, 64, 1, 16),         # f32 -> s8 / u8 (the input quantize)
    ("lnorm", 128, 768, 2, 0), ("lnorm", 128, 768, 2, 1),         # layer / RMS norm rows <= 768 f32
    ("rope", 128, 64, 8, 0), ("rope", 128, 64, 8, 1),             # RoPE, head dim 64, half-split / interleaved
    ("gemm_bf16",) + BF16_PARAMS,                                  # float matmul (bf16; f32 as bf16x3) + f32 composites
    ("gemm_wq",) + WQ_PARAMS,                                      # s8 x 4-bit weights with group scales
) + tuple(("feltw",) + FELT_PARAMS + (f,) for f in LIBRARY_FELTW) + tuple(("fbinary",) + FBIN_PARAMS + (f,) for f in LIBRARY_FBINARY + (FAFF_PRELU,)) \
  + tuple(("faffine",) + FAFF_PARAMS + (f,) for f in LIBRARY_FAFFINE) \
  + tuple(("bin8",) + BIN8_PARAMS + fq for fq in LIBRARY_BIN8) \
  + tuple(("frow",) + frow_params(op, n) for op, n in LIBRARY_FROW)


def library_plan(model: str = "library", mesh_rows: int = 4) -> EagerPlan:
    """the premade library (:data:`LIBRARY`) as a plan: no layers are assigned, the runtime binds each layer to
    an instance at create time"""
    insts = []
    for t, *a in LIBRARY:
        if t in GEMM_TEMPLATES:
            args, why = pick(*a, t)
            if args is None:
                raise AiegraphError("dnn_eager", f"library instance {t}{tuple(a)}: {why}")
            insts.append(Instance(*args, template=t))
        else:
            insts.append(Instance(0, 0, 0, 0, a[-1] if t != "pool" else 16, t, tuple(a)))
    n_gemm = sum(i.template in GEMM_TEMPLATES for i in insts)
    # counts of instances: no layers are assigned
    return fit_mesh_rows(EagerPlan(model, insts, [], n_gemm, len(insts) - n_gemm), mesh_rows)

