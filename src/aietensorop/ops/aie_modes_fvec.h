/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_fvec.h -- FVEC: f32 vector programs and row operations as a mode of the GEMM ELF, opcode 0xF5.
 *
 * One item per core: IN is the core's 4 KB item gathered from its A rows (n <= 1024 floats), OUT its 4 KB C tile,
 * AUX AIE_FVEC_AUX floats copied out of the core's B rows.
 *
 *   AIE_FV_PROG   elementwise: a small vector program per 16 lanes (below), built by the host
 *                 (aie_fvec_prog). AUX: the second operand element for element (binary) in [0, 1024),
 *                 the program at AIE_FVEC_PROGAT.
 *   AIE_FV_SOFTMAX / LOGSOFTMAX / LNORM / RMSNORM   rows of rowlen; norms take eps = alpha, 1/rowlen =
 *                 beta and gamma / beta rows in AUX (flags).
 *   AIE_FV_ROWPROG  the program over rows of rowlen with R2 / R3 = that row's constants (AUX[2r], AUX[2r + 1]):
 *                 R1 = R2, or with AIE_FV_F_ELEM the AUX element and the constants at AIE_FVEC_ROWAT.
 *   AIE_FV_ROWSUM   ROWPROG, but each row writes the sum of its results (one f32 at out[row]).
 *   AIE_FV_RMAX / RSUM   row reductions (max; sum * beta: sum or mean), one f32 per row at out[row].
 *
 * Program: R0 = x, R1 = the AUX element (binary), R2, R3 scratch; the result is R0. Instruction words
 * (uint32) op | d << 8 | a << 12 | b << 16, b a register or a constant index; constants K[] f32.
 * Layout at AIE_FVEC_PROGAT: int32 nins, int32 nconst, f32 K[nconst], uint32 ins[nins].
 *
 * Code size is the constraint (16 KB program memory for the whole megakernel): heavy helpers are out of
 * line, no scalar float arithmetic (it pulls in softfloat), reductions are shuffle_down_rotate trees.
 *
 * Contract: |got - ref| <= 1e-5 + 1e-5 |ref| (ref: in double), not bit-exact. 1/x and 1/sqrt(x) are
 * aie::inv / aie::invsqrt plus one Newton step; log(x) is Newton on e^y = x.
 *
 * Descriptor in B row 127 (first 64 bytes): [0] 0xF5  [1] mesh column  [2] op  [3] flags
 *   [4..7] alpha f32  [8..11] beta f32;  core in mesh row r: 4 bytes at AIE_FVEC_ENTRY(r) = {n LE16,
 *   rowlen LE16}. The gemm kernel hands aie_fvec_apply the core's mdesc: D[0..3] = its entry, D[4..13] =
 *   bytes 2..11.
 * Alignment: IN / OUT are vector-aligned; AUX (the gemm kernel's dwb) only 16-byte: load_unaligned_v(p, 4).
 ******************************************************************************/
#ifndef AIE_MODES_FVEC_H
#define AIE_MODES_FVEC_H

#include <stdint.h>
#include <string.h>

#define AIE_MODE_OP_FVEC 0xF5
#define AIE_FVEC_N 1024                  /* floats in / out per core item */
#define AIE_FVEC_AUX 1920                /* floats of AUX per core (7680 bytes of B rows) */
#define AIE_FVEC_PROGAT 1024             /* AUX float index of a program */
#define AIE_FVEC_ROWAT 1152              /* AUX float index of the row constants under AIE_FV_F_ELEM */
#define AIE_FVEC_ENTRY(r) (16 + 4 * (r)) /* the per-core entry in the descriptor row */
#define AIE_FV_MAXK 32
#define AIE_FV_MAXI 64

enum { /* item ops */
       AIE_FV_PROG = 1,
       AIE_FV_SOFTMAX = 64, /* rows of rowlen */
       AIE_FV_LOGSOFTMAX,
       AIE_FV_LNORM, /* (x - mean) / sqrt(var + alpha) [* gamma + beta], beta = 1 / rowlen */
       AIE_FV_RMSNORM,
       AIE_FV_RMAX,    /* row reductions: one f32 per row, at out[row] */
       AIE_FV_RSUM,    /* beta = 1 (sum) or 1 / rowlen (mean) */
       AIE_FV_ROWPROG, /* the program per row of rowlen, R2 / R3 = the row's two constants AUX[2 row], AUX[2 row + 1] */
       AIE_FV_ROWSUM,  /* ROWPROG summed per row: out[row] */
};
#define AIE_FV_F_SCALE 1 /* norms: AUX holds gamma [rowlen] */
#define AIE_FV_F_ELEM 4  /* ROWPROG / ROWSUM: R1 = the AUX element, row constants at AIE_FVEC_ROWAT */
#define AIE_FV_F_SHIFT                                                                                                 \
    2 /* norms: AUX holds beta [rowlen] after gamma. The core needs both (one loop, no                                 \
       * unswitched copies): the host sends 1 / 0 for an absent one */

enum { /* program instructions: op | AIE_FVI_K when b is a constant index (K[b]) instead of a register */
       AIE_FVI_END = 0,
       AIE_FVI_MOV, /* d = a */
       AIE_FVI_MUL, /* d = a * b */
       AIE_FVI_ADD,
       AIE_FVI_SUB,
       AIE_FVI_MAX,
       AIE_FVI_MIN,
       AIE_FVI_EXP, /* d = e^a */
       AIE_FVI_INV, /* d = 1 / a */
       AIE_FVI_RSQRT,
       AIE_FVI_LOG,
       AIE_FVI_ABS,
       AIE_FVI_SELP, /* d = a > 0 ? b : 0 (the step functions of eltwise backward) */
       AIE_FVI_K = 0x80,
       AIE_FVI_MULC = AIE_FVI_MUL | AIE_FVI_K, /* d = a * K[b] */
       AIE_FVI_ADDC = AIE_FVI_ADD | AIE_FVI_K,
       AIE_FVI_MAXC = AIE_FVI_MAX | AIE_FVI_K, /* d = max(a, K[b]) */
       AIE_FVI_MINC = AIE_FVI_MIN | AIE_FVI_K,
};

enum { /* the functions aie_fvec_prog builds (host-side algorithm codes) */
       AIE_FVF_LINEAR = 1,
       AIE_FVF_RELU,
       AIE_FVF_CLIP,
       AIE_FVF_ABS,
       AIE_FVF_SQUARE,
       AIE_FVF_SQRT,
       AIE_FVF_EXP,
       AIE_FVF_LOGISTIC,
       AIE_FVF_TANH,
       AIE_FVF_GELU_TANH,
       AIE_FVF_SWISH,
       AIE_FVF_ELU,
       AIE_FVF_LOG,
       AIE_FVF_SOFTPLUS,
       AIE_FVF_MISH,
       AIE_FVF_HARDSWISH,
       AIE_FVF_HARDSIGMOID,
       AIE_FVF_GELU_ERF,
       AIE_FVF_ERF,
       AIE_FVF_SELU,
       AIE_FVF_SOFTSIGN,
       AIE_FVF_LOG_SIGMOID,
       AIE_FVF_SIGN,
       AIE_FVF_RECIPROCAL,
       AIE_FVF_RSQRT,
       AIE_FVF_ROUND,
       AIE_FVF_FLOOR,
       AIE_FVF_CEIL,
       AIE_FVF_LOGICAL_NOT,
       AIE_FVF_POW, /* pow: integer b in [-2, 4] */
       AIE_FVF_SIN,
       AIE_FVF_COS,
       AIE_FVF_TAN,
       AIE_FVF_ADD = 64,
       AIE_FVF_SUB,
       AIE_FVF_MUL,
       AIE_FVF_DIV,
       AIE_FVF_MAX,
       AIE_FVF_MIN, /* binary: R1 = AUX */
       AIE_FVF_EQ,
       AIE_FVF_NE,
       AIE_FVF_GT,
       AIE_FVF_GE,
       AIE_FVF_LT,
       AIE_FVF_LE, /* 1 / 0 */
       AIE_FVF_LAND,
       AIE_FVF_LOR,
       AIE_FVF_LXOR,
       AIE_FVF_ADD_SQUARE,
       AIE_FVF_ROW_AFFINE = 96,
       AIE_FVF_ROW_AFFINE_RELU,
       AIE_FVF_ROW_PRELU, /* ROWPROG: R2 / R3 per row */
       AIE_FVF_ROW_SMBWD,
       AIE_FVF_ROW_LSMBWD,
       AIE_FVF_ROW_SQDEV,
       AIE_FVF_PICK1, /* the training rows (AIE_FV_F_ELEM) */
       AIE_FVF_ROW_DXHAT,
       AIE_FVF_ROW_LIN2, /* norm backward: dy (x - c2) c3, c2 R1 + c3 R0 */
       AIE_FVF_ROW_PRELU_DX,
       AIE_FVF_ROW_PRELU_DW, /* PReLU backward: x > 0 ? dy : c2 dy, and dy min(x, 0) */
};

static inline void aie_fvec_entry(uint8_t *e, int n, int rowlen) {
    e[0] = (uint8_t)n, e[1] = (uint8_t)(n >> 8), e[2] = (uint8_t)rowlen, e[3] = (uint8_t)(rowlen >> 8);
}

/* ── the program builder (plain C: host, tests and both executors share it) ── */
typedef struct {
    int32_t nins, nconst;
    float k[AIE_FV_MAXK];
    uint32_t ins[AIE_FV_MAXI];
} aie_fv_prog_t;

static inline int aie_fvp_k(aie_fv_prog_t *p, float v) {
    for (int i = 0; i < p->nconst; i++)
        if (!memcmp(&p->k[i], &v, 4))
            return i;
    p->k[p->nconst] = v;
    return p->nconst++;
}
static inline void aie_fvp_i(aie_fv_prog_t *p, int op, int d, int a, int b) {
    p->ins[p->nins++] = (uint32_t)op | (uint32_t)d << 8 | (uint32_t)a << 12 | (uint32_t)b << 16;
}
static inline void aie_fvp_c(aie_fv_prog_t *p, int op, int d, int a, float k) {
    aie_fvp_i(p, op, d, a, aie_fvp_k(p, k));
}

/* d = 1 / (1 + e^(-a)) through d */
static inline void aie_fvp_sigmoid(aie_fv_prog_t *p, int d, int a) {
    aie_fvp_c(p, AIE_FVI_MULC, d, a, -1.0f);
    aie_fvp_i(p, AIE_FVI_EXP, d, d, 0);
    aie_fvp_c(p, AIE_FVI_ADDC, d, d, 1.0f);
    aie_fvp_i(p, AIE_FVI_INV, d, d, 0);
}
/* d = tanh(a) = 2 sigmoid(2a) - 1 */
static inline void aie_fvp_tanh(aie_fv_prog_t *p, int d, int a) {
    aie_fvp_c(p, AIE_FVI_MULC, d, a, 2.0f);
    aie_fvp_sigmoid(p, d, d);
    aie_fvp_c(p, AIE_FVI_MULC, d, d, 2.0f);
    aie_fvp_c(p, AIE_FVI_ADDC, d, d, -1.0f);
}
/* d = softplus(a) = max(a, 0) + log(1 + e^-|a|) (stable), using scratch t */
static inline void aie_fvp_softplus(aie_fv_prog_t *p, int d, int a, int t) {
    aie_fvp_i(p, AIE_FVI_ABS, t, a, 0);
    aie_fvp_c(p, AIE_FVI_MULC, t, t, -1.0f);
    aie_fvp_i(p, AIE_FVI_EXP, t, t, 0);
    aie_fvp_c(p, AIE_FVI_ADDC, t, t, 1.0f);
    aie_fvp_i(p, AIE_FVI_LOG, t, t, 0);
    aie_fvp_c(p, AIE_FVI_MAXC, d, a, 0.0f);
    aie_fvp_i(p, AIE_FVI_ADD, d, d, t);
}
/* d = erf(|a|) by Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7), using scratch t; a is left as |a| in t? no */
static inline void aie_fvp_erfabs(aie_fv_prog_t *p, int d, int a, int t) {
    aie_fvp_i(p, AIE_FVI_ABS, d, a, 0); /* z = |a| */
    aie_fvp_i(p, AIE_FVI_MUL, t, d, d); /* z^2 */
    aie_fvp_c(p, AIE_FVI_MULC, t, t, -1.0f);
    aie_fvp_i(p, AIE_FVI_EXP, t, t, 0); /* e^-z^2 */
    aie_fvp_c(p, AIE_FVI_MULC, d, d, 0.3275911f);
    aie_fvp_c(p, AIE_FVI_ADDC, d, d, 1.0f);
    aie_fvp_i(p, AIE_FVI_INV, d, d, 0); /* s = 1 / (1 + p z) */
    aie_fvp_i(p, AIE_FVI_MUL, t, t, d); /* e * s; then Horner in s, d keeps s */
    const int r3 = 3;
    aie_fvp_c(p, AIE_FVI_MULC, r3, d, 1.061405429f);
    aie_fvp_c(p, AIE_FVI_ADDC, r3, r3, -1.453152027f);
    aie_fvp_i(p, AIE_FVI_MUL, r3, r3, d);
    aie_fvp_c(p, AIE_FVI_ADDC, r3, r3, 1.421413741f);
    aie_fvp_i(p, AIE_FVI_MUL, r3, r3, d);
    aie_fvp_c(p, AIE_FVI_ADDC, r3, r3, -0.284496736f);
    aie_fvp_i(p, AIE_FVI_MUL, r3, r3, d);
    aie_fvp_c(p, AIE_FVI_ADDC, r3, r3, 0.254829592f); /* poly(s) / s */
    aie_fvp_i(p, AIE_FVI_MUL, r3, r3, t);             /* poly(s) e^-z^2 */
    aie_fvp_c(p, AIE_FVI_MULC, d, r3, -1.0f);
    aie_fvp_c(p, AIE_FVI_ADDC, d, d, 1.0f); /* 1 - poly e^-z^2 */
}

/* R0 = R0 - 2 pi round(R0 / 2 pi), in [-pi, pi]: 2 pi in two parts (Cody-Waite; the first has 8 bits, so k C1 is
 * exact for |k| < 2^15), R1 / R2 scratch */
static inline void aie_fvp_reduce_2pi(aie_fv_prog_t *p) {
    aie_fvp_c(p, AIE_FVI_MULC, 1, 0, 0.15915494309189535f);                                        /* x / 2 pi */
    aie_fvp_c(p, AIE_FVI_ADDC, 1, 1, 12582912.0f), aie_fvp_c(p, AIE_FVI_ADDC, 1, 1, -12582912.0f); /* k */
    aie_fvp_c(p, AIE_FVI_MULC, 2, 1, 6.28125f), aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 2);
    aie_fvp_c(p, AIE_FVI_MULC, 2, 1, 0.0019353071795864769f), aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 2);
}
/* sin(R0) for R0 in [-pi, 3 pi / 2]: folded into [-pi / 2, pi / 2] (r > pi / 2: pi - r; r < -pi / 2: -pi - r),
 * then the odd Taylor polynomial to x^11 (|error| < 6e-8 there) */
static inline void aie_fvp_sin_reduced(aie_fv_prog_t *p) {
    const float hp = 1.5707963267948966f, pi = 3.1415926535897932f;
    aie_fvp_c(p, AIE_FVI_ADDC, 1, 0, -hp), aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 1, 1, 1.0f); /* [r > pi/2] */
    aie_fvp_c(p, AIE_FVI_MULC, 2, 0, -2.0f), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, pi);
    aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2); /* r + m (pi - 2r) */
    aie_fvp_c(p, AIE_FVI_MULC, 1, 0, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 1, 1, -hp);
    aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 1, 1, 1.0f); /* [r < -pi/2] */
    aie_fvp_c(p, AIE_FVI_MULC, 2, 0, -2.0f), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, -pi);
    aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2); /* r + m (-pi - 2r) */
    aie_fvp_i(p, AIE_FVI_MUL, 1, 0, 0);                                     /* x^2 */
    aie_fvp_c(p, AIE_FVI_MULC, 2, 1, -2.5052108385441720e-8f);
    aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 2.7557319223985893e-6f), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1);
    aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, -1.9841269841269841e-4f), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1);
    aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 8.3333333333333333e-3f), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1);
    aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, -0.16666666666666667f), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1);
    aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 1.0f), aie_fvp_i(p, AIE_FVI_MUL, 0, 2, 0);
}

/* x = sin(x) for |x| <= pi / 2 by the odd polynomial, scratch s1 / s2 (any registers) */
static inline void aie_fvp_sinpoly(aie_fv_prog_t *p, int x, int s1, int s2) {
    aie_fvp_i(p, AIE_FVI_MUL, s1, x, x);
    aie_fvp_c(p, AIE_FVI_MULC, s2, s1, -2.5052108385441720e-8f);
    aie_fvp_c(p, AIE_FVI_ADDC, s2, s2, 2.7557319223985893e-6f), aie_fvp_i(p, AIE_FVI_MUL, s2, s2, s1);
    aie_fvp_c(p, AIE_FVI_ADDC, s2, s2, -1.9841269841269841e-4f), aie_fvp_i(p, AIE_FVI_MUL, s2, s2, s1);
    aie_fvp_c(p, AIE_FVI_ADDC, s2, s2, 8.3333333333333333e-3f), aie_fvp_i(p, AIE_FVI_MUL, s2, s2, s1);
    aie_fvp_c(p, AIE_FVI_ADDC, s2, s2, -0.16666666666666667f), aie_fvp_i(p, AIE_FVI_MUL, s2, s2, s1);
    aie_fvp_c(p, AIE_FVI_ADDC, s2, s2, 1.0f), aie_fvp_i(p, AIE_FVI_MUL, x, s2, x);
}

/* the program of function f (alpha a, beta b) into p; 0 if f is not one */
static inline int aie_fvec_prog(int f, float a, float b, aie_fv_prog_t *p) {
    memset(p, 0, sizeof *p);
    switch (f) {
    case AIE_FVF_LINEAR:
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, a), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, b);
        break;
    case AIE_FVF_RELU: /* max(x, 0) + a min(x, 0) */
        aie_fvp_c(p, AIE_FVI_MINC, 2, 0, 0.0f), aie_fvp_c(p, AIE_FVI_MULC, 2, 2, a);
        aie_fvp_c(p, AIE_FVI_MAXC, 0, 0, 0.0f), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2);
        break;
    case AIE_FVF_CLIP:
        aie_fvp_c(p, AIE_FVI_MAXC, 0, 0, a), aie_fvp_c(p, AIE_FVI_MINC, 0, 0, b);
        break;
    case AIE_FVF_ABS:
        aie_fvp_i(p, AIE_FVI_ABS, 0, 0, 0);
        break;
    case AIE_FVF_SQUARE:
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 0);
        break;
    case AIE_FVF_SQRT: /* x x^-1/2, 0 at 0 */
        aie_fvp_c(p, AIE_FVI_MAXC, 0, 0, 1e-38f), aie_fvp_i(p, AIE_FVI_RSQRT, 2, 0, 0),
            aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2);
        break;
    case AIE_FVF_EXP:
        aie_fvp_i(p, AIE_FVI_EXP, 0, 0, 0);
        break;
    case AIE_FVF_LOG:
        aie_fvp_i(p, AIE_FVI_LOG, 0, 0, 0);
        break;
    case AIE_FVF_LOGISTIC:
        aie_fvp_sigmoid(p, 0, 0);
        break;
    case AIE_FVF_TANH:
        aie_fvp_tanh(p, 0, 0);
        break;
    case AIE_FVF_GELU_TANH: /* 0.5 x (1 + tanh(x (sqrt(2/pi) + 0.0356774 x^2))) */
        aie_fvp_i(p, AIE_FVI_MUL, 2, 0, 0), aie_fvp_c(p, AIE_FVI_MULC, 2, 2, 0.0356774081f);
        aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 0.7978845608f), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 0);
        aie_fvp_tanh(p, 2, 2), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 1.0f);
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2), aie_fvp_c(p, AIE_FVI_MULC, 0, 0, 0.5f);
        break;
    case AIE_FVF_SWISH: /* x sigmoid(a x) */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 0, a), aie_fvp_sigmoid(p, 2, 2), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2);
        break;
    case AIE_FVF_ELU:    /* max(x, 0) + a (e^min(x, 0) - 1) */
    case AIE_FVF_SELU: { /* b (max(x, 0) + a (e^min(x, 0) - 1)) */
        aie_fvp_c(p, AIE_FVI_MINC, 2, 0, 0.0f), aie_fvp_i(p, AIE_FVI_EXP, 2, 2, 0),
            aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, -1.0f);
        aie_fvp_c(p, AIE_FVI_MULC, 2, 2, a), aie_fvp_c(p, AIE_FVI_MAXC, 0, 0, 0.0f), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2);
        if (f == AIE_FVF_SELU)
            aie_fvp_c(p, AIE_FVI_MULC, 0, 0, b);
        break;
    }
    case AIE_FVF_SOFTPLUS: /* softplus(a x) / a */
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, a), aie_fvp_softplus(p, 0, 0, 2), aie_fvp_c(p, AIE_FVI_MULC, 0, 0, 1.0f / a);
        break;
    case AIE_FVF_MISH: /* x tanh(softplus(x)) */
        aie_fvp_softplus(p, 2, 0, 3), aie_fvp_tanh(p, 2, 2), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2);
        break;
    case AIE_FVF_HARDSIGMOID:
    case AIE_FVF_HARDSWISH: /* clip(a x + b, 0, 1) [* x] */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 0, a), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, b);
        aie_fvp_c(p, AIE_FVI_MAXC, 2, 2, 0.0f), aie_fvp_c(p, AIE_FVI_MINC, 2, 2, 1.0f);
        aie_fvp_i(p, f == AIE_FVF_HARDSWISH ? AIE_FVI_MUL : AIE_FVI_MOV, 0, f == AIE_FVF_HARDSWISH ? 0 : 2, 2);
        break;
    case AIE_FVF_GELU_ERF: /* 0.5 x + 0.5 |x| erf(|x| / sqrt 2): x erf(x / sqrt 2) = |x| erf(|x| / sqrt 2) */
        aie_fvp_c(p, AIE_FVI_MULC, 1, 0, 0.70710678118f), aie_fvp_erfabs(p, 2, 1, 1);
        aie_fvp_i(p, AIE_FVI_ABS, 1, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1);
        aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2), aie_fvp_c(p, AIE_FVI_MULC, 0, 0, 0.5f);
        break;
    case AIE_FVF_ERF: /* sign(x) erf(|x|): x / max(|x|, tiny) is the sign (0 at 0) */
    case AIE_FVF_SIGN:
        aie_fvp_i(p, AIE_FVI_ABS, 1, 0, 0), aie_fvp_c(p, AIE_FVI_MAXC, 1, 1, 1e-30f),
            aie_fvp_i(p, AIE_FVI_INV, 1, 1, 0);
        aie_fvp_i(p, AIE_FVI_MUL, 1, 1, 0); /* sign */
        if (f == AIE_FVF_ERF)
            aie_fvp_erfabs(p, 2, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 1, 2);
        else
            aie_fvp_i(p, AIE_FVI_MOV, 0, 1, 0);
        break;
    case AIE_FVF_SOFTSIGN: /* x / (1 + |x|) */
        aie_fvp_i(p, AIE_FVI_ABS, 2, 0, 0), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 1.0f), aie_fvp_i(p, AIE_FVI_INV, 2, 2, 0);
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2);
        break;
    case AIE_FVF_LOG_SIGMOID: /* -softplus(-x) */
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f), aie_fvp_softplus(p, 0, 0, 2), aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f);
        break;
    case AIE_FVF_RECIPROCAL:
        aie_fvp_i(p, AIE_FVI_INV, 0, 0, 0);
        break;
    case AIE_FVF_RSQRT:
        aie_fvp_i(p, AIE_FVI_RSQRT, 0, 0, 0);
        break;
    case AIE_FVF_ROW_AFFINE: /* x * c2 + c3 (batch norm inference, per-channel scale / shift) */
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 3);
        break;
    case AIE_FVF_ROW_AFFINE_RELU:
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 3), aie_fvp_c(p, AIE_FVI_MAXC, 0, 0, 0.0f);
        break;
    case AIE_FVF_ROW_PRELU: /* max(x, 0) + w min(x, 0), w = R1: the row's c2 (ROWPROG) or the AUX element (PROG) */
        aie_fvp_c(p, AIE_FVI_MINC, 2, 0, 0.0f), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1);
        aie_fvp_c(p, AIE_FVI_MAXC, 0, 0, 0.0f), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2);
        break;
    /* round half to even by the fp32 trick (x + 1.5 2^23 - 1.5 2^23: exact for |x| < 2^22, within the
     * contract beyond); floor / ceil step it by [r > x] / [x > r] with SELP of the constant 1 */
    case AIE_FVF_ROUND:
    case AIE_FVF_FLOOR:
    case AIE_FVF_CEIL:
        aie_fvp_c(p, AIE_FVI_ADDC, 2, 0, 12582912.0f), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, -12582912.0f);
        if (f == AIE_FVF_ROUND)
            aie_fvp_i(p, AIE_FVI_MOV, 0, 2, 0);
        else {
            aie_fvp_i(p, AIE_FVI_SUB, 3, f == AIE_FVF_FLOOR ? 2 : 0, f == AIE_FVF_FLOOR ? 0 : 2);
            aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 3, 3, 1.0f);
            aie_fvp_i(p, f == AIE_FVF_FLOOR ? AIE_FVI_SUB : AIE_FVI_ADD, 0, 2, 3);
        }
        break;
    case AIE_FVF_SIN: /* range-reduced to [-pi, pi], folded, polynomial */
        aie_fvp_reduce_2pi(p), aie_fvp_sin_reduced(p);
        break;
    case AIE_FVF_COS: /* sin(pi / 2 - r), r reduced: in [-pi / 2, 3 pi / 2] */
        aie_fvp_reduce_2pi(p);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.5707963267948966f);
        aie_fvp_sin_reduced(p);
        break;
    case AIE_FVF_TAN: /* sin(r) / cos(r), r = x - k pi; cos(r) = sin(delta), delta = pi/2 - |r| kept accurate near the
                       * poles: rh = x - k C1 is exact, delta = (C1/2 - |rh|) [exact] + C2/2 + sign(rh) k C2 */
        aie_fvp_c(p, AIE_FVI_MULC, 1, 0, 0.31830988618379067f);
        aie_fvp_c(p, AIE_FVI_ADDC, 1, 1, 12582912.0f), aie_fvp_c(p, AIE_FVI_ADDC, 1, 1, -12582912.0f); /* k */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 1, 3.140625f), aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 2);               /* rh */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 1, 9.6765358979329324e-4f);                                      /* k C2 */
        aie_fvp_i(p, AIE_FVI_ABS, 3, 0, 0), aie_fvp_c(p, AIE_FVI_MULC, 3, 3, -1.0f);
        aie_fvp_c(p, AIE_FVI_ADDC, 3, 3, 1.5703125f), aie_fvp_c(p, AIE_FVI_ADDC, 3, 3, 4.8382679489664662e-4f);
        aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 1, 0, 2.0f), aie_fvp_c(p, AIE_FVI_ADDC, 1, 1, -1.0f); /* sign(rh) */
        aie_fvp_i(p, AIE_FVI_MUL, 1, 1, 2), aie_fvp_i(p, AIE_FVI_ADD, 3, 3, 1);                      /* delta */
        aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 2);                                                          /* r */
        aie_fvp_sinpoly(p, 0, 1, 2), aie_fvp_sinpoly(p, 3, 1, 2);
        aie_fvp_i(p, AIE_FVI_INV, 3, 3, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 3);
        break;
    case AIE_FVF_LOGICAL_NOT: /* 1 - [|x| > 0] */
        aie_fvp_i(p, AIE_FVI_ABS, 2, 0, 0), aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 2, 2, 1.0f);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 2, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.0f);
        break;
    case AIE_FVF_POW: { /* a x^b, b an integer in [-2, 4] (else no program: the host declines) */
        const int e = (int)b;
        if ((float)e != b || e < -2 || e > 4)
            return 0;
        if (e < 0)
            aie_fvp_i(p, AIE_FVI_INV, 0, 0, 0);
        aie_fvp_i(p, AIE_FVI_MOV, 2, 0, 0);
        if (e == 0)
            aie_fvp_c(p, AIE_FVI_MULC, 2, 2, 0.0f), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 1.0f);
        for (int i = 1; i < (e < 0 ? -e : e); i++)
            aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 0);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 2, a);
        break;
    }
    /* comparisons by the sign of the difference (a difference AIE flushes to 0 -- both operands within ~1e-38 of
     * each other -- compares equal) */
    case AIE_FVF_GT:
        aie_fvp_i(p, AIE_FVI_SUB, 2, 0, 1), aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 0, 2, 1.0f);
        break;
    case AIE_FVF_LT:
        aie_fvp_i(p, AIE_FVI_SUB, 2, 1, 0), aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 0, 2, 1.0f);
        break;
    case AIE_FVF_GE: /* 1 - [y > x] */
    case AIE_FVF_LE: /* 1 - [x > y] */
        aie_fvp_i(p, AIE_FVI_SUB, 2, f == AIE_FVF_GE ? 1 : 0, f == AIE_FVF_GE ? 0 : 1);
        aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 2, 2, 1.0f), aie_fvp_c(p, AIE_FVI_MULC, 0, 2, -1.0f);
        aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.0f);
        break;
    case AIE_FVF_NE: /* [|x - y| > 0] */
    case AIE_FVF_EQ:
        aie_fvp_i(p, AIE_FVI_SUB, 2, 0, 1), aie_fvp_i(p, AIE_FVI_ABS, 2, 2, 0);
        aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 0, 2, 1.0f);
        if (f == AIE_FVF_EQ)
            aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.0f);
        break;
    case AIE_FVF_LAND: /* nonzero = true: [x] [y], [x] + [y] capped at 1, |[x] - [y]| */
    case AIE_FVF_LOR:
    case AIE_FVF_LXOR:
        aie_fvp_i(p, AIE_FVI_ABS, 2, 1, 0), aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 2, 2, 1.0f);
        aie_fvp_i(p, AIE_FVI_ABS, 3, 0, 0), aie_fvp_c(p, AIE_FVI_SELP | AIE_FVI_K, 3, 3, 1.0f);
        if (f == AIE_FVF_LAND)
            aie_fvp_i(p, AIE_FVI_MUL, 0, 2, 3);
        else if (f == AIE_FVF_LOR)
            aie_fvp_i(p, AIE_FVI_ADD, 0, 2, 3), aie_fvp_c(p, AIE_FVI_MINC, 0, 0, 1.0f);
        else
            aie_fvp_i(p, AIE_FVI_SUB, 0, 2, 3), aie_fvp_i(p, AIE_FVI_ABS, 0, 0, 0);
        break;
    case AIE_FVF_ADD_SQUARE:
        aie_fvp_i(p, AIE_FVI_MUL, 2, 1, 1), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2);
        break;
    case AIE_FVF_ROW_SMBWD: /* softmax backward: y (dy - s), R0 = y, R1 = dy, R2 = the row's s = sum y dy */
        aie_fvp_i(p, AIE_FVI_SUB, 3, 1, 2), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 3);
        break;
    case AIE_FVF_ROW_LSMBWD: /* log softmax backward: dy - e^y s, s = sum dy */
        aie_fvp_i(p, AIE_FVI_EXP, 0, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2), aie_fvp_i(p, AIE_FVI_SUB, 0, 1, 0);
        break;
    case AIE_FVF_ROW_SQDEV: /* (x - c2)^2: a row's centred squares (ROWSUM: the variance numerator) */
        aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 2), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 0);
        break;
    case AIE_FVF_ROW_DXHAT: /* dy xhat = dy (x - mean) rstd (ROWSUM: the scale gradient's row sums) */
        aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 2), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 3), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_ROW_LIN2: /* c2 R1 + c3 R0: dx's linear part in dy and x */
        aie_fvp_i(p, AIE_FVI_MUL, 2, 1, 2), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 3), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2);
        break;
    case AIE_FVF_ROW_PRELU_DX: /* c2 dy + [x > 0] dy - c2 [x > 0] dy */
        aie_fvp_i(p, AIE_FVI_SELP, 3, 0, 1), aie_fvp_i(p, AIE_FVI_MUL, 0, 1, 2), aie_fvp_i(p, AIE_FVI_MUL, 2, 3, 2);
        aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 3), aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 2);
        break;
    case AIE_FVF_ROW_PRELU_DW: /* dy (x - [x > 0] x) = dy min(x, 0) */
        aie_fvp_i(p, AIE_FVI_SELP, 3, 0, 0), aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 3), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_PICK1: /* R1: the AUX operand itself (ROWSUM: sum dy) */
        aie_fvp_i(p, AIE_FVI_MOV, 0, 1, 0);
        break;
    case AIE_FVF_ADD:
        aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 1);
        break;
    case AIE_FVF_SUB:
        aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 1);
        break;
    case AIE_FVF_MUL:
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_DIV:
        aie_fvp_i(p, AIE_FVI_INV, 2, 1, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2);
        break;
    case AIE_FVF_MAX:
        aie_fvp_i(p, AIE_FVI_MAX, 0, 0, 1);
        break;
    case AIE_FVF_MIN:
        aie_fvp_i(p, AIE_FVI_MIN, 0, 0, 1);
        break;
    default:
        return 0;
    }
    return 1;
}

/* eltwise backward: dy f'(x) with R0 = x, R1 = dy (R1 kept until its last use); 0 if not one.
 * Steps are g + SELP(x > 0, h - g) so x = 0 takes the left branch, as ref: does. Not here: gelu_erf (erf's
 * three scratch registers do not fit beside x, dy and a partial). */
static inline int aie_fvec_prog_bwd(int f, float a, float b, aie_fv_prog_t *p) {
    memset(p, 0, sizeof *p);
    switch (f) {
    case AIE_FVF_RELU: /* dy a + (1 - a) [x > 0] dy */
        aie_fvp_i(p, AIE_FVI_SELP, 2, 0, 1), aie_fvp_c(p, AIE_FVI_MULC, 2, 2, 1.0f - a);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 1, a), aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 2);
        break;
    case AIE_FVF_ELU:  /* g = a e^min(x, 0) dy; g + [x > 0] (dy - g) */
    case AIE_FVF_SELU: /* b times that */
        aie_fvp_c(p, AIE_FVI_MINC, 2, 0, 0.0f), aie_fvp_i(p, AIE_FVI_EXP, 2, 2, 0), aie_fvp_c(p, AIE_FVI_MULC, 2, 2, a);
        aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 1), aie_fvp_i(p, AIE_FVI_SUB, 3, 1, 2), aie_fvp_i(p, AIE_FVI_SELP, 3, 0, 3);
        aie_fvp_i(p, AIE_FVI_ADD, 0, 2, 3);
        if (f == AIE_FVF_SELU)
            aie_fvp_c(p, AIE_FVI_MULC, 0, 0, b);
        break;
    /* sigmoid derivatives never as 1 - s or 1 - t^2 (cancellation): with e = e^-z and s = 1 / (1 + e),
     * s (1 - s) = e s^2 and 1 - tanh(z / 2)^2 = 4 e s^2 */
    case AIE_FVF_LOGISTIC: /* e s^2, z = x */
    case AIE_FVF_TANH:     /* 4 e s^2, z = 2x */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 0, f == AIE_FVF_TANH ? -2.0f : -1.0f), aie_fvp_i(p, AIE_FVI_EXP, 2, 2, 0);
        aie_fvp_c(p, AIE_FVI_ADDC, 3, 2, 1.0f), aie_fvp_i(p, AIE_FVI_INV, 3, 3, 0);
        aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 3), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 3);
        if (f == AIE_FVF_TANH)
            aie_fvp_c(p, AIE_FVI_MULC, 2, 2, 4.0f);
        aie_fvp_i(p, AIE_FVI_MUL, 0, 2, 1);
        break;
    case AIE_FVF_SWISH: /* s + a x s (1 - s) = s (1 + a x e s), z = a x */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 0, -a), aie_fvp_i(p, AIE_FVI_EXP, 2, 2, 0);
        aie_fvp_c(p, AIE_FVI_ADDC, 3, 2, 1.0f), aie_fvp_i(p, AIE_FVI_INV, 3, 3, 0);
        aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 3), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 0), aie_fvp_c(p, AIE_FVI_MULC, 2, 2, a);
        aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 1.0f), aie_fvp_i(p, AIE_FVI_MUL, 0, 2, 3), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_SOFTPLUS: /* sigmoid(a x) */
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, a), aie_fvp_sigmoid(p, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_SIN: /* cos(x) dy = sin(pi / 2 - r) dy; dy kept in R3 (the programs use R1 / R2) */
    case AIE_FVF_COS: /* -sin(x) dy */
        aie_fvp_i(p, AIE_FVI_MOV, 3, 1, 0), aie_fvp_reduce_2pi(p);
        if (f == AIE_FVF_SIN)
            aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.5707963267948966f);
        aie_fvp_sin_reduced(p);
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 3);
        if (f == AIE_FVF_COS)
            aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f);
        break;
    case AIE_FVF_MISH: /* t + x (1 - t^2) sigmoid(x), t = tanh(softplus(x)) */
        aie_fvp_softplus(p, 2, 0, 3), aie_fvp_tanh(p, 2, 2), aie_fvp_i(p, AIE_FVI_MUL, 3, 2, 2);
        aie_fvp_c(p, AIE_FVI_MULC, 3, 3, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 3, 3, 1.0f),
            aie_fvp_i(p, AIE_FVI_MUL, 3, 3, 0);
        aie_fvp_sigmoid(p, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 3, 3, 0), aie_fvp_i(p, AIE_FVI_ADD, 0, 2, 3);
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_HARDSWISH: /* t = a x + b: 0 (t <= 0), t + a x (0 < t < 1), 1 (t >= 1) */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 0, a), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, b);
        aie_fvp_c(p, AIE_FVI_MULC, 3, 0, a), aie_fvp_i(p, AIE_FVI_ADD, 3, 3, 2), aie_fvp_i(p, AIE_FVI_MUL, 3, 3, 1);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 2, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.0f); /* 1 - t */
        aie_fvp_i(p, AIE_FVI_SELP, 3, 0, 3), aie_fvp_i(p, AIE_FVI_SELP, 3, 2, 3);        /* the middle */
        aie_fvp_i(p, AIE_FVI_SELP, 0, 0, 1), aie_fvp_i(p, AIE_FVI_SUB, 0, 1, 0);         /* [t >= 1] dy */
        aie_fvp_i(p, AIE_FVI_ADD, 0, 0, 3);
        break;
    case AIE_FVF_HARDSIGMOID: /* a (0 < t < 1) */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 0, a), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, b), aie_fvp_c(p, AIE_FVI_MULC, 3, 1, a);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 2, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.0f);
        aie_fvp_i(p, AIE_FVI_SELP, 3, 0, 3), aie_fvp_i(p, AIE_FVI_SELP, 0, 2, 3);
        break;
    case AIE_FVF_CLIP: /* dy - [a - x > 0] dy - [x - b > 0] dy */
        aie_fvp_c(p, AIE_FVI_MULC, 2, 0, -1.0f), aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, a),
            aie_fvp_i(p, AIE_FVI_SELP, 2, 2, 1);
        aie_fvp_c(p, AIE_FVI_ADDC, 3, 0, -b), aie_fvp_i(p, AIE_FVI_SELP, 3, 3, 1);
        aie_fvp_i(p, AIE_FVI_SUB, 0, 1, 2), aie_fvp_i(p, AIE_FVI_SUB, 0, 0, 3);
        break;
    case AIE_FVF_LINEAR:
        aie_fvp_c(p, AIE_FVI_MULC, 0, 1, a);
        break;
    case AIE_FVF_ABS:
        aie_fvp_i(p, AIE_FVI_SELP, 2, 0, 1), aie_fvp_c(p, AIE_FVI_MULC, 3, 0, -1.0f),
            aie_fvp_i(p, AIE_FVI_SELP, 3, 3, 1);
        aie_fvp_i(p, AIE_FVI_SUB, 0, 2, 3);
        break;
    case AIE_FVF_EXP:
        aie_fvp_i(p, AIE_FVI_EXP, 0, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_LOG:
        aie_fvp_i(p, AIE_FVI_INV, 0, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_SQRT:
        aie_fvp_i(p, AIE_FVI_RSQRT, 0, 0, 0), aie_fvp_c(p, AIE_FVI_MULC, 0, 0, 0.5f),
            aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_SQUARE:
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, 2.0f), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_RECIPROCAL:
        aie_fvp_i(p, AIE_FVI_INV, 0, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 0), aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f);
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_RSQRT: /* -0.5 x^-3/2 */
        aie_fvp_i(p, AIE_FVI_RSQRT, 2, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 2, 2), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -0.5f), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_ERF: /* 2 / sqrt(pi) e^-x^2 */
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 0), aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f), aie_fvp_i(p, AIE_FVI_EXP, 0, 0, 0);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, 1.1283791671f), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_SOFTSIGN: /* 1 / (1 + |x|)^2 */
        aie_fvp_i(p, AIE_FVI_ABS, 0, 0, 0), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.0f), aie_fvp_i(p, AIE_FVI_INV, 0, 0, 0);
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_LOG_SIGMOID: /* sigmoid(-x) */
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, -1.0f), aie_fvp_sigmoid(p, 0, 0), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    case AIE_FVF_GELU_TANH: /* u = x (k + k c x^2), t = tanh u: 0.5 (1 + t) + 0.5 x (1 - t^2) (k + 3 k c x^2)
                             * = s (1 + 2 x e s (k + 3 k c x^2)) with e = e^-2u, s = 1 / (1 + e) */
        aie_fvp_i(p, AIE_FVI_MUL, 2, 0, 0), aie_fvp_c(p, AIE_FVI_MULC, 2, 2, 0.0356774081f);
        aie_fvp_c(p, AIE_FVI_ADDC, 2, 2, 0.7978845608f), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 0);
        aie_fvp_c(p, AIE_FVI_MULC, 2, 2, -2.0f), aie_fvp_i(p, AIE_FVI_EXP, 2, 2, 0); /* e */
        aie_fvp_c(p, AIE_FVI_ADDC, 3, 2, 1.0f), aie_fvp_i(p, AIE_FVI_INV, 3, 3, 0);  /* s */
        aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 3), aie_fvp_i(p, AIE_FVI_MUL, 2, 2, 0);      /* x e s */
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 0), aie_fvp_c(p, AIE_FVI_MULC, 0, 0, 0.1070322243f);
        aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 0.7978845608f), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 2);
        aie_fvp_c(p, AIE_FVI_MULC, 0, 0, 2.0f), aie_fvp_c(p, AIE_FVI_ADDC, 0, 0, 1.0f);
        aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 3), aie_fvp_i(p, AIE_FVI_MUL, 0, 0, 1);
        break;
    default:
        return 0;
    }
    return 1;
}

/* the program into AUX at AIE_FVEC_PROGAT */
static inline void aie_fvec_prog_store(const aie_fv_prog_t *p, float *aux) {
    float *at = aux + AIE_FVEC_PROGAT;
    memcpy(at, &p->nins, 4), memcpy(at + 1, &p->nconst, 4);
    memcpy(at + 2, p->k, (size_t)p->nconst * 4);
    memcpy(at + 2 + p->nconst, p->ins, (size_t)p->nins * 4);
}

#if defined(__AIENGINE__) || defined(AIE_FVEC_EMU)

typedef aie::vector<float, 16> aie_fv;

#if defined(__AIENGINE__)
#define AIE_FV_OUTLINE __attribute__((noinline))
#else
#define AIE_FV_OUTLINE
#endif

static inline aie_fv aie_fv_bc(float x) { return aie::broadcast<float, 16>(x); }
/* an fp32 product: on AIE-ML a multi-product bf16 sequence, so one copy, called */
AIE_FV_OUTLINE static aie_fv aie_fv_mul(aie_fv a, aie_fv b) { return aie::mul(a, b).template to_vector<float>(); }

/* e^x: n = round(x / ln 2), r = x - n ln 2 in two parts (Cody-Waite: n * LN2_HI is exact), e^r a
 * degree-6 Taylor polynomial on |r| <= ln 2 / 2 (error < 1.3e-7), 2^n from the exponent bits.
 * n comes from the fp32 round-to-nearest trick (t + 1.5 * 2^23), not aie::to_fixed: AIE-ML converts
 * through bf16, which can put n one off. */
AIE_FV_OUTLINE static aie_fv aie_fv_exp(aie_fv x) {
    x = aie::max(aie::min(x, aie_fv_bc(88.3f)), aie_fv_bc(-87.3f));
    const aie_fv tn = aie::add(aie_fv_mul(x, aie_fv_bc(1.44269504089f)), aie_fv_bc(12582912.0f));
    const aie_fv nf = aie::sub(tn, aie_fv_bc(12582912.0f));
    const aie::vector<int32, 16> n = aie::sub(aie::vector_cast<int32>(tn), aie::broadcast<int32, 16>(0x4B400000));
    const aie_fv r =
        aie::sub(aie::sub(x, aie_fv_mul(nf, aie_fv_bc(0.693359375f))), aie_fv_mul(nf, aie_fv_bc(-2.12194440e-4f)));
    static const float c[6] = {1.0f / 120, 1.0f / 24, 1.0f / 6, 0.5f, 1.0f, 1.0f};
    aie_fv p = aie_fv_bc(1.0f / 720);
    for (int i = 0; i < 6; i++)
        p = aie::add(aie_fv_mul(p, r), aie_fv_bc(c[i]));
    const aie::vector<int32, 16> e = aie::upshift(aie::add(n, aie::broadcast<int32, 16>(127)), 23);
    return aie_fv_mul(p, aie::vector_cast<float>(e));
}

/* 1 / x and 1 / sqrt(x) with one more Newton step each: aie::invsqrt alone is ~7e-4 relative */
AIE_FV_OUTLINE static aie_fv aie_fv_inv(aie_fv x) {
    const aie_fv y = aie::inv(x);
    return aie_fv_mul(y, aie::sub(aie_fv_bc(2.0f), aie_fv_mul(x, y)));
}
AIE_FV_OUTLINE static aie_fv aie_fv_rsqrt(aie_fv x) {
    const aie_fv y = aie::invsqrt(x);
    return aie_fv_mul(y, aie::sub(aie_fv_bc(1.5f), aie_fv_mul(aie_fv_mul(x, aie_fv_bc(0.5f)), aie_fv_mul(y, y))));
}
/* log(x), x > 0: y0 = exponent * ln 2 (within ln 2), then y += x e^-y - 1 (Newton on e^y = x) */
AIE_FV_OUTLINE static aie_fv aie_fv_log(aie_fv x) {
    const aie::vector<int32, 16> bits = aie::vector_cast<int32>(x);
    aie_fv y = aie_fv_mul(aie::to_float<float>(aie::sub(aie::downshift(bits, 23), aie::broadcast<int32, 16>(127)), 0),
                          aie_fv_bc(0.69314718056f));
    for (int it = 0; it < 5; it++)
        y = aie::add(y, aie::sub(aie_fv_mul(x, aie_fv_exp(aie::sub(aie_fv_bc(0.0f), y))), aie_fv_bc(1.0f)));
    return y;
}

/* every lane = the sum / max of the 16 */
static inline aie_fv aie_fv_hsum(aie_fv v) {
    for (int s = 8; s >= 1; s /= 2)
        v = aie::add(v, aie::shuffle_down_rotate(v, s));
    return v;
}
static inline aie_fv aie_fv_hmax(aie_fv v) {
    for (int s = 8; s >= 1; s /= 2)
        v = aie::max(v, aie::shuffle_down_rotate(v, s));
    return v;
}

/* the program on one 16-lane vector: R[0..3] in, the result in R[0] */
AIE_FV_OUTLINE static void aie_fv_exec(aie_fv *R, const float *kc, const uint32_t *ins, int nins) {
    for (int pc = 0; pc < nins; pc++) {
        const uint32_t w = ins[pc];
        const int op = w & 0xff, d = (w >> 8) & 3, a = (w >> 12) & 3, b = (int)(w >> 16);
        const aie_fv x = R[a], y = (w & AIE_FVI_K) ? aie_fv_bc(kc[b]) : R[b & 3];
        aie_fv v;
        switch (op & ~AIE_FVI_K) {
        case AIE_FVI_MOV:
            v = x;
            break;
        case AIE_FVI_MUL:
            v = aie_fv_mul(x, y);
            break;
        case AIE_FVI_ADD:
            v = aie::add(x, y);
            break;
        case AIE_FVI_SUB:
            v = aie::sub(x, y);
            break;
        case AIE_FVI_MAX:
            v = aie::max(x, y);
            break;
        case AIE_FVI_MIN:
            v = aie::min(x, y);
            break;
        case AIE_FVI_EXP:
            v = aie_fv_exp(x);
            break;
        case AIE_FVI_INV:
            v = aie_fv_inv(x);
            break;
        case AIE_FVI_RSQRT:
            v = aie_fv_rsqrt(x);
            break;
        case AIE_FVI_LOG:
            v = aie_fv_log(x);
            break;
        case AIE_FVI_SELP:
            v = aie::select(aie_fv_bc(0.0f), y, aie::gt(x, aie_fv_bc(0.0f)));
            break;
        default:
            v = aie::max(x, aie::sub(aie_fv_bc(0.0f), x));
            break; /* ABS */
        }
        R[d] = v;
    }
}

/* the program over n floats: elementwise (L 0: R1 = AUX element for element) or rows of L (R2 / R3 = the
 * row's AUX constants; R1 = R2, or the AUX element under AIE_FV_F_ELEM); sum: each row's results summed */
AIE_FV_OUTLINE static void aie_fv_run(const float *in, float *out, int n, const float *aux, int L, int fl, int sum) {
    const float *at = aux + AIE_FVEC_PROGAT;
    int32_t nins, nk;
    __builtin_memcpy(&nins, at, 4);
    __builtin_memcpy(&nk, at + 1, 4);
    const float *kc = at + 2; /* (not K: the GEMM's dimension macros M / N / K reach this header) */
    const uint32_t *ins = (const uint32_t *)(at + 2 + nk);
    const float *cst = (fl & AIE_FV_F_ELEM) ? aux + AIE_FVEC_ROWAT : aux;
    /* branch-free loop (a loop-invariant branch is unswitched into one copy per case): R1 by a lane mask,
     * R2 / R3 reloaded always (elementwise programs write them before reading), every result stored and
     * summed -- a ROWSUM row's stores land at index >= r L, past every sum already written */
    const auto el = aie::gt(aie_fv_bc(!L || (fl & AIE_FV_F_ELEM) ? 1.0f : 0.0f), aie_fv_bc(0.0f));
    aie_fv R[4];
    for (int r = 0, i = 0; i < n; r++) {
        const int end = L ? i + L : n;
        aie_fv acc = aie_fv_bc(0.0f);
        for (; i < end; i += 16) { /* R2 / R3 are program scratch: reload the row's constants per vector */
            R[0] = aie::load_v<16>(in + i);
            R[2] = aie_fv_bc(cst[2 * r]), R[3] = aie_fv_bc(cst[2 * r + 1]);
            R[1] = aie::select(R[2], aie::load_unaligned_v<16>(aux + i, 4), el);
            aie_fv_exec(R, kc, ins, nins);
            acc = aie::add(acc, R[0]);
            aie::store_v(out + i, R[0]);
        }
        if (sum)
            out[r] = aie_fv_hsum(acc)[0];
    }
}

/* softmax / log softmax rows of L (a multiple of 16) */
AIE_FV_OUTLINE static void aie_fv_row_softmax(int op, const float *in, float *out, int L) {
    aie_fv m = aie_fv_bc(-3.402823466e38f);
    for (int i = 0; i < L; i += 16)
        m = aie::max(m, aie::load_v<16>(in + i));
    const aie_fv mx = aie_fv_hmax(m);
    if (op == AIE_FV_RMAX) {
        *out = mx[0];
        return;
    }
    aie_fv s = aie_fv_bc(0.0f);
    for (int i = 0; i < L; i += 16) { /* e^(x - max), kept in out (log softmax overwrites it below) */
        const aie_fv e = aie_fv_exp(aie::sub(aie::load_v<16>(in + i), mx));
        s = aie::add(s, e);
        aie::store_v(out + i, e);
    }
    s = aie_fv_hsum(s);
    /* one branch-free last pass (no per-op loop copies): softmax e * (1 / sum) + 0, log x * 1 - (max + log sum) */
    const int sm = op == AIE_FV_SOFTMAX;
    const aie_fv k1 = sm ? aie_fv_inv(s) : aie_fv_bc(1.0f),
                 k2 = sm ? aie_fv_bc(0.0f) : aie::sub(aie_fv_bc(0.0f), aie::add(mx, aie_fv_log(s)));
    const float *src = sm ? out : in;
    for (int i = 0; i < L; i += 16)
        aie::store_v(out + i, aie::add(aie_fv_mul(aie::load_v<16>(src + i), k1), k2));
}

/* layer / RMS norm rows: two passes (mean, then the centred square) as the reference; rl = 1 / L */
AIE_FV_OUTLINE static void aie_fv_row_norm(int op, const float *in, float *out, int L, float eps, float rl,
                                           const float *gamma, const float *beta) {
    aie_fv s = aie_fv_bc(0.0f);
    if (op != AIE_FV_RMSNORM)
        for (int i = 0; i < L; i += 16)
            s = aie::add(s, aie::load_v<16>(in + i));
    const aie_fv mean = aie_fv_mul(aie_fv_hsum(s), aie_fv_bc(rl));
    if (op == AIE_FV_RSUM) {
        *out = mean[0];
        return;
    }
    aie_fv q = aie_fv_bc(0.0f);
    for (int i = 0; i < L; i += 16) {
        const aie_fv d = aie::sub(aie::load_v<16>(in + i), mean);
        q = aie::add(q, aie_fv_mul(d, d));
    }
    const aie_fv rs = aie_fv_rsqrt(aie::add(aie_fv_mul(aie_fv_hsum(q), aie_fv_bc(rl)), aie_fv_bc(eps)));
    for (int i = 0; i < L; i += 16) /* gamma / beta always (the host passes 1 / 0): one loop, no unswitched copies */
        aie::store_v(out + i, aie::add(aie_fv_mul(aie_fv_mul(aie::sub(aie::load_v<16>(in + i), mean), rs),
                                                  aie::load_unaligned_v<16>(gamma + i, 4)),
                                       aie::load_unaligned_v<16>(beta + i, 4)));
}

/* the core's item: D = {n LE16, rowlen LE16, op, flags, alpha f32, beta f32} (see the descriptor) */
static inline void aie_fvec_apply(const float *in, float *out, const uint8_t *D, const float *aux) {
    const int n = D[0] | (D[1] << 8), L = D[2] | (D[3] << 8), op = D[4], fl = D[5];
    float a, b;
    __builtin_memcpy(&a, D + 6, 4);
    __builtin_memcpy(&b, D + 10, 4);
    if (op == AIE_FV_PROG || op == AIE_FV_ROWPROG || op == AIE_FV_ROWSUM) {
        aie_fv_run(in, out, n, aux, op == AIE_FV_PROG ? 0 : L, fl, op == AIE_FV_ROWSUM);
        return;
    }
    if (L <= 0) /* an unused item of a partial launch: n = rowlen = 0 (a row loop would never advance) */
        return;
    const float *g = (fl & AIE_FV_F_SCALE) ? aux : 0,
                *bt = (fl & AIE_FV_F_SHIFT) ? aux + ((fl & AIE_FV_F_SCALE) ? L : 0) : 0;
    const int red = op >= AIE_FV_RMAX;
    for (int r = 0, k = 0; r + L <= n; r += L, k++)
        if (op <= AIE_FV_LOGSOFTMAX || op == AIE_FV_RMAX)
            aie_fv_row_softmax(op, in + r, out + (red ? k : r), L);
        else
            aie_fv_row_norm(op, in + r, out + (red ? k : r), L, a, b, g, bt);
}

#else /* plain C: the x86 reference of the fvec_blk_f32 entry (gemm_uk_ref.c), libm float math */

#include <math.h>

static inline float aie_fvec_ref_run1(const float *kc, const uint32_t *ins, int nins, float x, float y, float r2,
                                      float r3) {
    float R[4] = {x, y, r2, r3};
    for (int pc = 0; pc < nins; pc++) {
        const uint32_t w = ins[pc];
        const int op = w & 0xff, d = (w >> 8) & 3, a = (w >> 12) & 3, b = (int)(w >> 16);
        const float v = R[a], u = (op & AIE_FVI_K) ? kc[b] : R[b & 3];
        float r;
        switch (op & ~AIE_FVI_K) {
        case AIE_FVI_MOV:
            r = v;
            break;
        case AIE_FVI_MUL:
            r = v * u;
            break;
        case AIE_FVI_ADD:
            r = v + u;
            break;
        case AIE_FVI_SUB:
            r = v - u;
            break;
        case AIE_FVI_MAX:
            r = fmaxf(v, u);
            break;
        case AIE_FVI_MIN:
            r = fminf(v, u);
            break;
        case AIE_FVI_EXP:
            r = expf(fminf(fmaxf(v, -87.3f), 88.3f));
            break;
        case AIE_FVI_INV:
            r = 1.0f / v;
            break;
        case AIE_FVI_RSQRT:
            r = 1.0f / sqrtf(v);
            break;
        case AIE_FVI_LOG:
            r = logf(v);
            break;
        case AIE_FVI_SELP:
            r = v > 0.0f ? u : 0.0f;
            break;
        default:
            r = fabsf(v);
            break;
        }
        R[d] = r;
    }
    return R[0];
}

static inline void aie_fvec_apply(const float *in, float *out, const uint8_t *D, const float *aux) {
    const int n = D[0] | (D[1] << 8), L = D[2] | (D[3] << 8), op = D[4], fl = D[5];
    float a, b;
    memcpy(&a, D + 6, 4);
    memcpy(&b, D + 10, 4);
    if (op == AIE_FV_PROG || op == AIE_FV_ROWPROG || op == AIE_FV_ROWSUM) {
        const float *at = aux + AIE_FVEC_PROGAT, *cst = (fl & AIE_FV_F_ELEM) ? aux + AIE_FVEC_ROWAT : aux;
        const int rows = op != AIE_FV_PROG, el = !rows || (fl & AIE_FV_F_ELEM);
        int32_t nins, nk;
        memcpy(&nins, at, 4), memcpy(&nk, at + 1, 4);
        double acc = 0;
        for (int i = 0; i < n; i++) {
            const int r = rows && L ? i / L : 0;
            const float c2 = rows ? cst[2 * r] : 0.0f, c3 = rows ? cst[2 * r + 1] : 0.0f;
            const float v =
                aie_fvec_ref_run1(at + 2, (const uint32_t *)(at + 2 + nk), nins, in[i], el ? aux[i] : c2, c2, c3);
            if (op != AIE_FV_ROWSUM)
                out[i] = v;
            else if (acc += v, (i + 1) % L == 0)
                out[r] = (float)acc, acc = 0;
        }
        return;
    }
    if (L <= 0)
        return;
    const float *g = (fl & AIE_FV_F_SCALE) ? aux : 0,
                *bt = (fl & AIE_FV_F_SHIFT) ? aux + ((fl & AIE_FV_F_SCALE) ? L : 0) : 0;
    for (int r = 0, k = 0; r + L <= n; r += L, k++) {
        const float *x = in + r;
        float *y = out + r;
        if (op == AIE_FV_RMAX || op == AIE_FV_RSUM) {
            double v = op == AIE_FV_RMAX ? -INFINITY : 0.0;
            for (int i = 0; i < L; i++)
                v = op == AIE_FV_RMAX ? (x[i] > v ? x[i] : v) : v + x[i];
            out[k] = (float)(op == AIE_FV_RMAX ? v : v * b);
            continue;
        }
        if (op == AIE_FV_SOFTMAX || op == AIE_FV_LOGSOFTMAX) {
            double mx = -INFINITY, s = 0;
            for (int i = 0; i < L; i++)
                mx = x[i] > mx ? x[i] : mx;
            for (int i = 0; i < L; i++)
                s += exp((double)x[i] - mx);
            for (int i = 0; i < L; i++)
                y[i] = op == AIE_FV_SOFTMAX ? (float)(exp((double)x[i] - mx) / s) : (float)((double)x[i] - mx - log(s));
        } else {
            double mean = 0, q = 0;
            if (op == AIE_FV_LNORM)
                for (int i = 0; i < L; i++)
                    mean += x[i];
            mean *= b;
            for (int i = 0; i < L; i++)
                q += ((double)x[i] - mean) * ((double)x[i] - mean);
            const double rs = 1.0 / sqrt(q * b + a);
            for (int i = 0; i < L; i++)
                y[i] = (float)(((double)x[i] - mean) * rs * (g ? g[i] : 1.0) + (bt ? bt[i] : 0.0));
        }
    }
}

/* the D an item's geom {n, rowlen, op | flags << 8, 0} and alpha / beta give the core */
static inline void aie_fvec_desc(uint8_t *D, const int32_t *g, const float *ab) {
    aie_fvec_entry(D, g[0], g[1]);
    D[4] = (uint8_t)(g[2] & 0xff), D[5] = (uint8_t)((g[2] >> 8) & 0xff);
    memcpy(D + 6, ab, 4);
    memcpy(D + 10, ab + 1, 4);
}

#endif
#endif /* AIE_MODES_FVEC_H */
