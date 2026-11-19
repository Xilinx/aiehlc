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

    def summary(self) -> str:
        lines = [f"eager plan [{self.model}]: {self.gemm_layers} GEMM layers"
                 + (f" + {self.other_layers} other" if self.other_layers else "")
                 + f" -> {len(self.instances)} instances, {len(self.refused)} refused"]
        lines += [f"  {i.name}: {len(i.ops)} layer(s) " + ", ".join(o["name"] for o in i.ops) for i in self.instances]
        lines += [f"  refused {n}: {why}" for n, why in self.refused]
        return "\n".join(lines)

    def to_json(self) -> dict:
        return dict(model=self.model, gemm_layers=self.gemm_layers,
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
    TN = N // MESH_COLS
    # the (TM, TK) that fits with the most work per core round (TM x TK multiply-accumulates per output
    # column); ties go to the larger TM. N 64: (16, 256); N 768: (8, 32); N 3072: (2, 8)
    # padding is exact but computes zeros: keep K's padding within 1/8 of K when any (TM, TK) can (K 576:
    # TK 128 pads to 640, not 256 to 768), then the most work per round, ties to the larger TM
    def kpad(TK):
        return math.ceil(K / TK) * TK - K
    # on-core extras next to the accumulator: gemm_q's requant words; gemm_qr's 12 words + the residual tile
    def extra(TM):
        return {"gemm_q": TN * Q_WORDS * 4, "gemm_qr": TN * QR_WORDS * 4 + TM * TN}.get(template, 0)
    tk_min = {"gemm_q": Q_TK_MIN, "gemm_qr": QR_TK_MIN}.get(template, 0)
    fit = [(TM * TK, TM, TK) for TM in TM_CHOICES for TK in TK_CHOICES
           if TK >= tk_min and core_bytes(TM, TK, TN)[0] <= CORE_BUDGET
           and core_bytes(TM, TK, TN)[1] + extra(TM) <= ACC_BUDGET]
    fit = [f for f in fit if kpad(f[2]) * K_PAD_MAX <= K] or fit
    if fit:
        _, TM, TK = max(fit)
        step = MESH_COLS * TM
        return (math.ceil(M / step) * step, N, math.ceil(K / TK) * TK, TK, TM), None
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
    mds = [v for k, v in op.create if k == "md"]
    alg = next((v for k, v in op.create if k == "alg"), "")
    pp = next((v for k, v in op.create if k == "pool"), None)
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
LUT_PARAMS = (256, 128, 32)        # M, N, TM (N in {64, 128}; TM >= N / 4)
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


def lut_candidate(g, op) -> bool:
    """an 8-bit -> 8-bit eltwise / PReLU / batch norm, a binary with exactly one broadcast operand, or an
    identity reorder (no permutation, same shape): what eager:lut can run"""
    if not any(op.fn.startswith(f) for f in LUT_FAMILIES):
        return False
    ts = [g.tensors[v] for k, v in ((c[0], c[1] if len(c) > 1 else None) for c in op.create) if k == "md" and v is not None]
    if not ts or any(t.dt not in ("s8", "u8") for t in ts):
        return False
    dst = g.tensors[op.out]
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
        cp = next(v for k, v in op.create if k == "conv")
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
             "lut": ("lut_eager_kernel.cc", "lut_eager_ref.h", "int8_t"),
             "quant": ("quant_eager_kernel.cc", "quant_eager_ref.h", "int8_t"),
             "lnorm": ("lnorm_eager_kernel.cc", "lnorm_eager_ref.h", "int8_t"),
             "rope": ("rope_eager_kernel.cc", "rope_eager_ref.h", "int8_t")}
GEMM_TEMPLATES = ("gemm_s8", "gemm_q", "gemm_qr")
COMMON_HEADER = "eager_common.h"
KERNEL_HEADER = TEMPLATES["gemm_s8"][0]
ENTRY_FILE, REF_FILE = "eager_instances.cc", "eager_instances_ref.c"


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
    if i.template == "lut":
        M, N, TM = i.params
        spec = (f'"kernel    eager_{i.name}\\nop        gemm_s8s32\\nmatch     dense\\nreference {REF_FILE}\\n'
                f'target    vek280\\nin    int8  [{M},{256 + N}]  src\\nparam int8  [{N},{256 + N}]  weight\\n'
                f'out   int8  [{M},{N}]  dst\\n"')
        return dict(sig="int8_t *A, int8_t *W, int8_t *Y", launch=f"lut<{M}, {N}, {TM}><<<mesh>>>(A, W, Y)",
                    entry=f"{M}, {N}", ref=f"{M}, {N}", spec=spec)
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
           '#include "aie_user_kernel.h"']                                  # kernel_lib lint L004
    out += [f'#include "{TEMPLATES[t][0]}"' for t in _used(p)] + [""]
    for i in p.instances:
        e = _emit(i)
        out += [f"// {', '.join(o['name'] for o in i.ops)}",
                f"static void {i.name}_launch({e['sig']}) {{",
                "    aieSetDevice(0);",
                "    aieArray device;",
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
    for f in [COMMON_HEADER] + [f for t in _used(p) for f in TEMPLATES[t][:2]]:
        (out / f).write_text((eager_dir / f).read_text())
    return entry


def plan(g, model: "str | None" = None) -> EagerPlan:
    """the instance list of AGraph *g*: every GEMM-family layer either joins the instance of its padded
    shape or is refused with a reason"""
    by_key, refused, seen, other = {}, [], 0, 0
    for i, op in enumerate(g.ops):
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
        if lut_candidate(g, op):
            other += 1
            inst = by_key.setdefault(("lut",) + LUT_PARAMS, Instance(0, 0, 0, 0, LUT_PARAMS[2], "lut", LUT_PARAMS))
            inst.ops.append(dict(op=i, name=op.name, kind="lut"))
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
    return EagerPlan(model or g.name, list(by_key.values()), refused, seen, other)


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
    ("lut",) + LUT_PARAMS,                                         # every 8-bit per-channel function
    ("quant", 128, 64, 0, 16), ("quant", 128, 64, 1, 16),         # f32 -> s8 / u8 (the input quantize)
    ("lnorm", 128, 768, 2, 0), ("lnorm", 128, 768, 2, 1),         # layer / RMS norm rows <= 768 f32
    ("rope", 128, 64, 8, 0), ("rope", 128, 64, 8, 1),             # RoPE, head dim 64, half-split / interleaved
)


def library_plan(model: str = "library") -> EagerPlan:
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
    return EagerPlan(model, insts, [], n_gemm, len(insts) - n_gemm)   # counts of instances: no layers are assigned

