/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_fvec.h -- fvec: f32 pointwise and row primitives on the FVEC mode of the GEMM ELF
 * (src/aietensorop/ops/aie_modes_fvec.h, opcode 0xF5).
 *
 *   eltwise   vector programs (aie_fvec_prog), any shape flattened into 1024-float items
 *   binary / prelu   src1 materialized per item, or ROWPROG rows when constant over a trailing block
 *   eltwise backward, softmax backward, batch norm training: row passes (ROWSUM, ROWPROG)
 *   batch / layer / RMS norm, softmax, reduction / pooling over rows <= 1024
 *
 * f32 in and out; |result - ref:| <= 1e-5 + 1e-5 |ref:|. Entry fvec_blk_f32: items f32 [NI,1024], aux f32
 * [NI,1920], geom s32 [NI,4] = {n, rowlen, op | flags << 8, 0}, ab f32 [1,2] = {alpha, beta}, out f32 [NI,1024].
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aie_modes_fvec.h (opcodes / limits).
 ******************************************************************************/
#ifndef AIEDNN_RT_FVEC_H
#define AIEDNN_RT_FVEC_H

#define RT_FVEC_MAX_ITEMS 64

static int is_fvec_block(const aiednn_impl_t *g) {
    return g->nargs == 5 && g->args[0].data_type == aiednn_f32 && g->args[0].ndims == 2 && g->args[0].dims[0] >= 1 &&
           g->args[0].dims[0] <= RT_FVEC_MAX_ITEMS && g->args[0].dims[1] == AIE_FVEC_N &&
           g->args[1].data_type == aiednn_f32 && g->args[1].dims[1] == AIE_FVEC_AUX &&
           g->args[2].data_type == aiednn_s32 && g->args[2].dims[1] == 4 && g->args[3].data_type == aiednn_f32 &&
           g->args[4].data_type == aiednn_f32 && g->args[4].dims[1] == AIE_FVEC_N;
}

typedef struct {
    int op, flags;
    float a, b;
    int64_t numel; /* elements of dst */
    int64_t L, Lp; /* row ops: row length, padded to 16 (else 0) */
    int s1;        /* binary: src1's md slot */
    int red;       /* row reductions (RMAX / RSUM): one output per row */
    int64_t sp;    /* ROWPROG: the elements sharing one constant (a row is L of them, L | sp) */
    int xf;        /* the item copy: 0 x, 1 -x (reduction min), 2 |x| (amax) */
    int64_t inner; /* softmax over a non-last axis: a row's elements lie inner apart (0 / 1: contiguous) */
    int qb;        /* quantized binary (attr) */
    int qd;        /* layer / RMS norm with a quantized dst (attr) */
    int train;     /* row-pass problems (fvec_train_plan): 1 softmax backward, 2 batch / group norm (block statistics),
                    * 3 layer norm training forward, 4 batch / group norm backward, 5 PReLU backward */
    aie_fv_prog_t prog; /* AIE_FV_PROG: the function as a vector program (aie_fvec_prog) */
} fvec_plan;

/* the eltwise algorithm as an FVEC function code, with its alpha / beta (0: not one) */
static int fvec_eltwise_fn(const rt_prob *p, float *a, float *b) {
    *a = p->alpha, *b = p->beta;
    switch (p->alg) {
    case aiednn_eltwise_relu:
        return AIE_FVF_RELU;
    case aiednn_eltwise_elu:
        return AIE_FVF_ELU;
    case aiednn_eltwise_selu:
        *a = 1.6732632423543772f, *b = 1.0507009873554805f;
        return AIE_FVF_SELU;
    case aiednn_eltwise_gelu_tanh:
        return AIE_FVF_GELU_TANH;
    case aiednn_eltwise_gelu_erf:
        return AIE_FVF_GELU_ERF;
    case aiednn_eltwise_logistic:
        return AIE_FVF_LOGISTIC;
    case aiednn_eltwise_tanh:
        return AIE_FVF_TANH;
    case aiednn_eltwise_swish:
        return AIE_FVF_SWISH;
    case aiednn_eltwise_soft_relu:
        return p->alpha != 0.0f ? AIE_FVF_SOFTPLUS : 0;
    case aiednn_eltwise_mish:
        return AIE_FVF_MISH;
    case aiednn_eltwise_hardswish:
        return AIE_FVF_HARDSWISH;
    case aiednn_eltwise_hardsigmoid:
        return AIE_FVF_HARDSIGMOID;
    case aiednn_eltwise_clip:
        return AIE_FVF_CLIP;
    case aiednn_eltwise_linear:
        return AIE_FVF_LINEAR;
    case aiednn_eltwise_abs:
        return AIE_FVF_ABS;
    case aiednn_eltwise_exp:
        return AIE_FVF_EXP;
    case aiednn_eltwise_log:
        return AIE_FVF_LOG;
    case aiednn_eltwise_sqrt:
        return AIE_FVF_SQRT;
    case aiednn_eltwise_square:
        return AIE_FVF_SQUARE;
    case aiednn_eltwise_reciprocal:
        return AIE_FVF_RECIPROCAL;
    case aiednn_eltwise_rsqrt:
        return AIE_FVF_RSQRT;
    case aiednn_eltwise_erf:
        return AIE_FVF_ERF;
    case aiednn_eltwise_sign:
        return AIE_FVF_SIGN;
    case aiednn_eltwise_softsign:
        return AIE_FVF_SOFTSIGN;
    case aiednn_eltwise_log_sigmoid:
        return AIE_FVF_LOG_SIGMOID;
    case aiednn_eltwise_neg:
        *a = -1.0f, *b = 0.0f;
        return AIE_FVF_LINEAR;
    case aiednn_eltwise_identity:
        *a = 1.0f, *b = 0.0f;
        return AIE_FVF_LINEAR;
    case aiednn_eltwise_round:
        return AIE_FVF_ROUND;
    case aiednn_eltwise_floor:
        return AIE_FVF_FLOOR;
    case aiednn_eltwise_ceil:
        return AIE_FVF_CEIL;
    case aiednn_eltwise_logical_not:
        return AIE_FVF_LOGICAL_NOT;
    case aiednn_eltwise_pow:
        return AIE_FVF_POW; /* integer exponents in [-2, 4] (aie_fvec_prog declines others) */
    case aiednn_eltwise_sin:
        return AIE_FVF_SIN;
    case aiednn_eltwise_cos:
        return AIE_FVF_COS;
    case aiednn_eltwise_tan:
        return AIE_FVF_TAN;
    default:
        return 0;
    }
}

/* the trailing block of dst over which operand m is constant (its trailing dims of 1) */
static int64_t fvec_bcast_block(const aiednn_memory_desc_t *m, const aiednn_memory_desc_t *d) {
    int64_t sp = 1;
    for (int i = d->ndims - 1; i >= 0 && m->ndims == d->ndims && m->dims[i] == 1; i--)
        sp *= d->dims[i];
    return sp;
}

/* ROWPROG rows over blocks of sp: the divisor of sp (<= 1024) wasting the least padding to 16 */
static void fvec_rows(fvec_plan *q, int64_t sp) {
    int64_t best = 1;
    for (int64_t l = sp < AIE_FVEC_N ? sp : AIE_FVEC_N; l >= 1; l--)
        if (sp % l == 0 && (l + 15) / 16 * 16 * best < (best + 15) / 16 * 16 * l)
            best = l;
    q->op = AIE_FV_ROWPROG, q->sp = sp, q->L = best, q->Lp = (best + 15) / 16 * 16;
}

/* the output: DIFF_SRC for eltwise backward (dy f'(x)), else DST */
static int fvec_out_arg(const rt_prob *p) { return p->prop == RT_BWD_DATA ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_DST; }

/* row R of L elements, inner apart (a softmax over a non-last axis), out of / into a dense row */
static inline void fvec_gather(float *dst, const float *src, int64_t R, int64_t L, int64_t inner) {
    if (inner <= 1) {
        memcpy(dst, src + R * L, (size_t)L * sizeof(float));
        return;
    }
    const float *b = src + (R / inner) * L * inner + R % inner;
    for (int64_t k = 0; k < L; k++)
        dst[k] = b[k * inner];
}
static inline void fvec_scatter(float *dst, const float *src, int64_t R, int64_t L, int64_t inner) {
    if (inner <= 1) {
        memcpy(dst + R * L, src, (size_t)L * sizeof(float));
        return;
    }
    float *b = dst + (R / inner) * L * inner + R % inner;
    for (int64_t k = 0; k < L; k++)
        b[k * inner] = src[k];
}
/* the softmax axis: its length and the stride of its elements */
static inline void fvec_axis(const aiednn_memory_desc_t *m, int axis, int64_t *L, int64_t *inner) {
    *L = m->dims[axis], *inner = 1;
    for (int i = axis + 1; i < m->ndims; i++)
        *inner *= m->dims[i];
}

/* quantized binary as one program: exact dequantize (code - zp) * scale for 8-bit codes, the op, relu / clip;
 * the quantize runs in the unpack. add / sub / max / min only (mul is not exact on the AIE). */
static int fvec_qbinary_plan(const rt_prob *p, fvec_plan *q, char *why, int cap) {
    const aiednn_memory_desc_t *a = &p->md[0], *b = &p->md[1], *d = &p->md[2];
    /* mul only with AIEDNN_ATTR_TOLERANCE_1CODE: the core's f32 product is not ref:'s correctly rounded one */
    const int tol1 = (p->attr_flags & AIEDNN_ATTR_TOLERANCE_1CODE) != 0;
    const int fi = p->alg == aiednn_binary_add           ? AIE_FVI_ADD
                   : p->alg == aiednn_binary_sub         ? AIE_FVI_SUB
                   : p->alg == aiednn_binary_max         ? AIE_FVI_MAX
                   : p->alg == aiednn_binary_min         ? AIE_FVI_MIN
                   : p->alg == aiednn_binary_mul && tol1 ? AIE_FVI_MUL
                                                         : 0;
    int ok = fi && md_same_shape(a, d);
    for (int k = 0; k < p->npo; k++)
        ok &= p->po[k].kind == aiednn_post_op_eltwise &&
              ((p->po[k].alg == aiednn_eltwise_relu && p->po[k].alpha == 0.0f) || p->po[k].alg == aiednn_eltwise_clip);
    if (!ok) {
        snprintf(why, cap,
                 "quantized binary: add / sub / max / min (mul with AIEDNN_ATTR_TOLERANCE_1CODE), src0 shaped as "
                 "dst, relu / clip post-ops");
        return 0;
    }
    aie_fv_prog_t *g = &q->prog;
    memset(g, 0, sizeof *g);
    for (int s = 0; s < 2; s++)
        if (p->md[s].data_type != aiednn_f32 && p->bin_sc[s] != 0.0f)
            aie_fvp_c(g, AIE_FVI_ADDC, s, s, (float)-p->bin_zp[s]), aie_fvp_c(g, AIE_FVI_MULC, s, s, p->bin_sc[s]);
    aie_fvp_i(g, fi, 0, 0, 1);
    for (int k = 0; k < p->npo; k++)
        if (p->po[k].alg == aiednn_eltwise_relu)
            aie_fvp_c(g, AIE_FVI_MAXC, 0, 0, 0.0f);
        else
            aie_fvp_c(g, AIE_FVI_MAXC, 0, 0, p->po[k].alpha), aie_fvp_c(g, AIE_FVI_MINC, 0, 0, p->po[k].beta);
    q->op = AIE_FV_PROG, q->qb = 1, q->s1 = 1, q->numel = md_numel(d);
    return 1;
}

/* the training problems served as row passes (run_fvec_train): -1 not one of them, else accepted (1) or not */
static int fvec_train_plan(const rt_prob *p, fvec_plan *q, char *why, int cap) {
    if (p->kind == aiednn_softmax && p->prop == RT_BWD_DATA) { /* y, dy -> dx over the last axis */
        const aiednn_memory_desc_t *y = rt_md(p, AIEDNN_ARG_DST), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST),
                                   *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
        fvec_axis(y, p->axis, &q->L, &q->inner);
        q->Lp = (q->L + 15) / 16 * 16, q->numel = md_numel(y);
        if (p->has_attr || y->data_type != aiednn_f32 || dy->data_type != aiednn_f32 || dx->data_type != aiednn_f32 ||
            (p->alg != aiednn_softmax_accurate && p->alg != aiednn_softmax_log) || q->Lp > AIE_FVEC_N) {
            snprintf(why, cap, "softmax backward: f32, accurate / log, rows <= 1024");
            return 0;
        }
        q->train = 1;
        return 1;
    }
    if (p->kind == aiednn_prelu && p->prop == RT_BWD) { /* weights constant over trailing blocks of >= 16 */
        const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC), *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
        const int args[] = {AIEDNN_ARG_SRC, AIEDNN_ARG_WEIGHTS, AIEDNN_ARG_DIFF_DST, AIEDNN_ARG_DIFF_SRC,
                            AIEDNN_ARG_DIFF_WEIGHTS};
        int ok = 1;
        for (int a : args)
            ok &= rt_slot(p, a) >= 0 && rt_md(p, a)->data_type == aiednn_f32;
        const int64_t sp = ok ? fvec_bcast_block(w, x) : 0;
        if (sp < 16) {
            snprintf(why, cap, "PReLU backward: f32, weights constant over a trailing block of >= 16 (per channel)");
            return 0;
        }
        fvec_rows(q, sp);
        q->numel = md_numel(x), q->train = 5;
        return 1;
    }
    if (p->kind == aiednn_layer_normalization && p->prop == RT_BWD) { /* rows of the last dim, chunked (any width) */
        const int args[] = {AIEDNN_ARG_SRC,      AIEDNN_ARG_DIFF_DST, AIEDNN_ARG_DIFF_SRC,   AIEDNN_ARG_MEAN,
                            AIEDNN_ARG_VARIANCE, AIEDNN_ARG_SCALE,    AIEDNN_ARG_DIFF_SCALE, AIEDNN_ARG_DIFF_SHIFT};
        for (int a : args)
            if (rt_slot(p, a) >= 0 && rt_md(p, a)->data_type != aiednn_f32) {
                snprintf(why, cap, "layer / RMS norm backward: f32");
                return 0;
            }
        const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC);
        fvec_rows(q, x->dims[x->ndims - 1]);
        if (q->L < 16) {
            snprintf(why, cap, "layer / RMS norm backward: row chunks of >= 16 elements");
            return 0;
        }
        q->numel = md_numel(x), q->train = 6;
        return 1;
    }
    if ((p->kind == aiednn_batch_normalization || p->kind == aiednn_group_normalization) && p->prop == RT_BWD) {
        const int args[] = {AIEDNN_ARG_SRC,      AIEDNN_ARG_DIFF_DST, AIEDNN_ARG_DIFF_SRC,   AIEDNN_ARG_MEAN,
                            AIEDNN_ARG_VARIANCE, AIEDNN_ARG_SCALE,    AIEDNN_ARG_DIFF_SCALE, AIEDNN_ARG_DIFF_SHIFT};
        for (int a : args)
            if (rt_slot(p, a) >= 0 && rt_md(p, a)->data_type != aiednn_f32) {
                snprintf(why, cap, "norm backward: f32");
                return 0;
            }
        const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC);
        fvec_rows(q, md_numel(x) / (x->dims[0] * x->dims[1]));
        q->numel = md_numel(x), q->train = 4;
        return 1;
    }
    const int bn = p->kind == aiednn_batch_normalization && p->prop == RT_FWD_TRAIN;
    const int gn = p->kind == aiednn_group_normalization && (p->prop == RT_FWD_TRAIN || p->prop == RT_FWD_INF);
    const int ln = p->kind == aiednn_layer_normalization && p->prop == RT_FWD_TRAIN;
    if (bn || gn || ln) { /* x -> y (+ mean, variance for training) */
        const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC), *y = rt_md(p, AIEDNN_ARG_DST);
        const aiednn_memory_desc_t *mm = rt_md(p, AIEDNN_ARG_MEAN), *vm = rt_md(p, AIEDNN_ARG_VARIANCE);
        const int64_t L = x->dims[x->ndims - 1];
        if (p->has_attr || x->data_type != aiednn_f32 || y->data_type != aiednn_f32 ||
            (mm && mm->data_type != aiednn_f32) || (vm && vm->data_type != aiednn_f32) ||
            (ln && (L % 16 || 2 * L > AIE_FVEC_AUX))) {
            snprintf(why, cap,
                     ln ? "layer norm training: f32, a last dim that is a multiple of 16, <= 960" : "f32 norms");
            return 0;
        }
        if (ln)
            q->L = q->Lp = L, q->train = 3;
        else /* rows: chunks of one channel's spatial block */
            fvec_rows(q, md_numel(x) / (x->dims[0] * x->dims[1])), q->train = 2;
        q->numel = md_numel(x);
        return 1;
    }
    return -1;
}

/* an f32 binary whose attr is only eltwise post-ops (SquaredDifference = sub + square, add + relu, ...) */
static int fvec_f32_binary_postops(const rt_prob *p) {
    if (p->md[0].data_type != aiednn_f32 || p->md[1].data_type != aiednn_f32 || p->md[2].data_type != aiednn_f32 ||
        p->bin_sc[0] != 0 || p->bin_sc[1] != 0 || p->n_oscales || p->npo < 1)
        return 0;
    for (int k = 0; k < p->npo; k++)
        if (p->po[k].kind != aiednn_post_op_eltwise)
            return 0;
    return 1;
}

/* dst's program, then src's: src's constants re-indexed into dst's table; 0 if it does not fit */
static int fvec_prog_append(aie_fv_prog_t *dst, const aie_fv_prog_t *src) {
    if (dst->nins + src->nins > AIE_FV_MAXI)
        return 0;
    for (int i = 0; i < src->nins; i++) {
        uint32_t w = src->ins[i];
        if (w & AIE_FVI_K) {
            const int bi = (int)((w >> 16) & 0xffff);
            int have = 0;
            for (int j = 0; j < dst->nconst; j++)
                have |= !memcmp(&dst->k[j], &src->k[bi], 4);
            if (!have && dst->nconst == AIE_FV_MAXK)
                return 0;
            w = (w & 0xffffu) | (uint32_t)aie_fvp_k(dst, src->k[bi]) << 16;
        }
        dst->ins[dst->nins++] = w;
    }
    return 1;
}

static int fvec_plan_make(const rt_prob *p, fvec_plan *q, char *why, int cap) {
    memset(q, 0, sizeof *q);
    q->s1 = -1;
    const int t = fvec_train_plan(p, q, why, cap);
    if (t >= 0)
        return t;
    if (p->kind == aiednn_binary && p->has_attr && fvec_f32_binary_postops(p)) { /* the f32 binary's program, then */
        rt_prob t = *p;                                                          /* each eltwise post-op's */
        t.has_attr = 0, t.npo = 0;
        if (!fvec_plan_make(&t, q, why, cap))
            return 0;
        for (int k = 0; k < p->npo; k++) {
            rt_prob e;
            memset(&e, 0, sizeof e);
            e.alg = p->po[k].alg, e.alpha = p->po[k].alpha, e.beta = p->po[k].beta;
            float a, b;
            aie_fv_prog_t ep;
            const int f = fvec_eltwise_fn(&e, &a, &b);
            if (!f || !aie_fvec_prog(f, a, b, &ep) || !fvec_prog_append(&q->prog, &ep)) {
                snprintf(why, cap, "binary post-op %d: not an FVEC program (or the program is full)", k);
                return 0;
            }
        }
        return 1;
    }
    if (p->kind == aiednn_binary && p->has_attr)
        return fvec_qbinary_plan(p, q, why, cap);
    const aiednn_memory_desc_t *s = &p->md[0], *d = rt_md(p, fvec_out_arg(p));
    const int bwd = p->prop == RT_BWD_DATA && p->kind == aiednn_eltwise;
    /* layer / RMS norm -> s8 / u8: f32 rows on the core, quantized in the unpack; a code may differ by one,
     * so only with AIEDNN_ATTR_TOLERANCE_1CODE */
    q->qd = p->kind == aiednn_layer_normalization && p->prop == RT_FWD_INF && p->has_attr && !p->npo && !p->n_oscales &&
            !p->ndq && d && (d->data_type == aiednn_s8 || d->data_type == aiednn_u8) && s->data_type == aiednn_f32 &&
            (p->attr_flags & AIEDNN_ATTR_TOLERANCE_1CODE);
    if (!q->qd && ((p->prop != RT_FWD_INF && !bwd) || p->has_attr || !d || s->data_type != aiednn_f32 ||
                   d->data_type != aiednn_f32)) {
        snprintf(why, cap,
                 "f32 -> f32 inference without an attr (or eltwise backward; layer norm -> s8 / u8 with "
                 "AIEDNN_ATTR_TOLERANCE_1CODE)");
        return 0;
    }
    if (bwd) { /* dy f'(x): x the item, dy element for element in AUX (R1) */
        float a, b;
        const int f = fvec_eltwise_fn(p, &a, &b);
        q->s1 = rt_slot(p, AIEDNN_ARG_DIFF_DST);
        if ((f == AIE_FVF_TAN || f == AIE_FVF_GELU_ERF) && q->s1 >= 0 && p->md[q->s1].data_type == aiednn_f32) {
            q->op = AIE_FV_PROG, q->numel = md_numel(d), q->train = 9; /* passes (run_fvec_bwd_passes) */
            return 1;
        }
        if (!f || q->s1 < 0 || p->md[q->s1].data_type != aiednn_f32 || !aie_fvec_prog_bwd(f, a, b, &q->prog)) {
            snprintf(why, cap, "eltwise backward: algorithm not in the FVEC backward programs (f32)");
            return 0;
        }
        q->op = AIE_FV_PROG, q->numel = md_numel(d);
        return 1;
    }
    q->numel = md_numel(d);
    switch (p->kind) {
    case aiednn_reduction:
    case aiednn_pooling: { /* rows of the reduced trailing axes: global pooling, sum / mean / max reductions */
        int64_t L = 1;
        if (p->kind == aiednn_pooling) {
            const aiednn_pool_params_t *pp = &p->pool;
            if (s->ndims != 4 || pp->kernel[0] != s->dims[2] || pp->kernel[1] != s->dims[3] || pp->padding_l[0] ||
                pp->padding_l[1] || pp->padding_r[0] || pp->padding_r[1] || pp->dilates[0] || pp->dilates[1]) {
                snprintf(why, cap, "global pooling only (kernel = H x W, no padding / dilation)");
                return 0;
            }
            L = s->dims[2] * s->dims[3];
            q->op = p->alg == aiednn_pooling_max ? AIE_FV_RMAX : AIE_FV_RSUM;
        } else {
            int r0 = s->ndims;
            while (r0 > 0 && d->dims[r0 - 1] == 1)
                r0--;
            for (int i = 0; i < r0; i++)
                if (d->dims[i] != s->dims[i])
                    r0 = -1;
            const int mx =
                p->alg == aiednn_reduction_max || p->alg == aiednn_reduction_min || p->alg == aiednn_reduction_amax;
            if (r0 < 0 || r0 == s->ndims || d->ndims != s->ndims ||
                (p->alg != aiednn_reduction_sum && p->alg != aiednn_reduction_mean && !mx)) {
                snprintf(why, cap, "sum / mean / max / min / amax over trailing axes");
                return 0;
            }
            for (int i = r0; i < s->ndims; i++)
                L *= s->dims[i];
            q->op = mx ? AIE_FV_RMAX : AIE_FV_RSUM;
            /* min = -max(-x), amax = max |x|: the sign taken in the item copy (fvec_fill), min's back in the unpack */
            q->xf = p->alg == aiednn_reduction_min ? 1 : p->alg == aiednn_reduction_amax ? 2 : 0;
        }
        q->L = L, q->Lp = (L + 15) / 16 * 16, q->red = 1, q->numel = md_numel(s);
        q->b = q->op == AIE_FV_RSUM && !(p->kind == aiednn_reduction && p->alg == aiednn_reduction_sum)
                   ? 1.0f / (float)L
                   : 1.0f;
        if (q->Lp > AIE_FVEC_N) {
            snprintf(why, cap, "rows of at most 1024 elements");
            return 0;
        }
        return 1;
    }
    case aiednn_eltwise: {
        float a, b;
        const int f = fvec_eltwise_fn(p, &a, &b);
        if (f == AIE_FVF_POW && !aie_fvec_prog(f, a, b, &q->prog) && b == b && fabsf(b) < INFINITY) {
            /* a x^b, any b: a e^(b log x) for x > 0 on the core, x <= 0 patched on the host (run_fvec_powpatch) */
            memset(&q->prog, 0, sizeof q->prog);
            aie_fvp_i(&q->prog, AIE_FVI_LOG, 0, 0, 0), aie_fvp_c(&q->prog, AIE_FVI_MULC, 0, 0, b);
            aie_fvp_i(&q->prog, AIE_FVI_EXP, 0, 0, 0), aie_fvp_c(&q->prog, AIE_FVI_MULC, 0, 0, a);
            q->op = AIE_FV_PROG, q->train = 11;
            return 1;
        }
        if (!f || !aie_fvec_prog(f, a, b, &q->prog)) {
            snprintf(why, cap, "algorithm not in the FVEC programs");
            return 0;
        }
        q->op = AIE_FV_PROG;
        return 1;
    }
    case aiednn_binary: {
        static const int fns[] = {AIE_FVF_ADD,  AIE_FVF_SUB, AIE_FVF_MUL,  AIE_FVF_DIV,       AIE_FVF_MAX, AIE_FVF_MIN,
                                  AIE_FVF_EQ,   AIE_FVF_NE,  AIE_FVF_GT,   AIE_FVF_GE,        AIE_FVF_LT,  AIE_FVF_LE,
                                  AIE_FVF_LAND, AIE_FVF_LOR, AIE_FVF_LXOR, AIE_FVF_ADD_SQUARE};
        static const aiednn_alg_kind_t algs[] = {
            aiednn_binary_add,         aiednn_binary_sub,        aiednn_binary_mul,         aiednn_binary_div,
            aiednn_binary_max,         aiednn_binary_min,        aiednn_binary_eq,          aiednn_binary_ne,
            aiednn_binary_gt,          aiednn_binary_ge,         aiednn_binary_lt,          aiednn_binary_le,
            aiednn_binary_logical_and, aiednn_binary_logical_or, aiednn_binary_logical_xor, aiednn_binary_add_square};
        int f = 0;
        q->s1 = rt_slot(p, AIEDNN_ARG_SRC_1);
        if (p->alg == aiednn_binary_select) { /* three tensors: element passes (run_fvec_select) */
            const aiednn_memory_desc_t *cm = rt_md(p, AIEDNN_ARG_SRC_2);
            if (q->s1 < 0 || !cm || s->data_type != aiednn_f32 || p->md[q->s1].data_type != aiednn_f32 ||
                d->data_type != aiednn_f32 || dt_is_4bit(cm->data_type)) {
                snprintf(why, cap, "select: f32 sources and dst (broadcast operands expanded on the host)");
                return 0;
            }
            q->op = AIE_FV_PROG, q->numel = md_numel(d), q->train = 10;
            return 1;
        }
        for (int i = 0; i < (int)(sizeof fns / sizeof fns[0]); i++)
            f = p->alg == algs[i] ? fns[i] : f;
        if (!f || q->s1 < 0 || p->md[q->s1].data_type != aiednn_f32 || !md_same_shape(s, d)) {
            snprintf(why, cap,
                     "add / sub / mul / div / max / min / compares / logical / add_square, f32 src0 shaped as dst");
            return 0;
        }
        aie_fvec_prog(f, 0.0f, 0.0f, &q->prog);
        q->op = AIE_FV_PROG;
        const int64_t sp = fvec_bcast_block(&p->md[q->s1], d);
        if (sp >= 16) /* src1 constant per block: rows, no per-element operand */
            fvec_rows(q, sp);
        return 1;
    }
    case aiednn_prelu: {
        q->s1 = rt_slot(p, AIEDNN_ARG_WEIGHTS);
        if (q->s1 < 0 || p->md[q->s1].data_type != aiednn_f32) {
            snprintf(why, cap, "f32 weights");
            return 0;
        }
        aie_fvec_prog(AIE_FVF_ROW_PRELU, 0.0f, 0.0f, &q->prog); /* w = R1: either form */
        q->op = AIE_FV_PROG;
        const int64_t sp = fvec_bcast_block(&p->md[q->s1], d);
        if (sp >= 16)
            fvec_rows(q, sp);
        return 1;
    }
    case aiednn_batch_normalization: { /* inference: always the given mean / variance */
        const int64_t sp = md_numel(s) / (s->dims[0] * s->dims[1]);
        aie_fvec_prog((p->flags & AIEDNN_FUSE_NORM_RELU) ? AIE_FVF_ROW_AFFINE_RELU : AIE_FVF_ROW_AFFINE, 0.0f, 0.0f,
                      &q->prog);
        fvec_rows(q, sp);
        return 1;
    }
    case aiednn_softmax:
    case aiednn_layer_normalization: {
        const int sm = p->kind == aiednn_softmax;
        if (sm) /* any axis: a non-last one gathers its strided rows into the items */
            fvec_axis(s, p->axis, &q->L, &q->inner);
        else
            q->L = s->dims[s->ndims - 1];
        q->Lp = (q->L + 15) / 16 * 16;
        if (!sm && p->prop == RT_FWD_INF && !(p->flags & ~(AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_RMS_NORM)) &&
            (q->Lp > AIE_FVEC_N || q->Lp != q->L || 2 * q->L > AIE_FVEC_AUX)) {
            /* rows the one-item norm cannot hold (llama's 2048 / 4096, or not a multiple of 16): chunked passes */
            const int64_t D = q->L;
            fvec_rows(q, D);
            if (q->L >= 16 && !q->qd) {
                q->numel = md_numel(s), q->train = 7;
                return 1;
            }
        }
        if ((sm && (p->alg != aiednn_softmax_accurate && p->alg != aiednn_softmax_log)) || q->Lp > AIE_FVEC_N ||
            (!sm && q->Lp != q->L) || (!sm && 2 * q->L > AIE_FVEC_AUX) ||
            (!sm && (p->flags & ~(AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_RMS_NORM)))) {
            snprintf(why, cap,
                     sm ? "softmax / log softmax, rows <= 1024"
                        : "layer / RMS norm over a last dim that is a multiple of 16, <= 960");
            return 0;
        }
        if (sm)
            q->op = p->alg == aiednn_softmax_log ? AIE_FV_LOGSOFTMAX : AIE_FV_SOFTMAX;
        else {
            q->op = (p->flags & AIEDNN_RMS_NORM) ? AIE_FV_RMSNORM : AIE_FV_LNORM, q->a = p->eps;
            q->b = 1.0f / (float)q->L;                  /* the core has no scalar float arithmetic */
            q->flags = AIE_FV_F_SCALE | AIE_FV_F_SHIFT; /* always both: 1 / 0 when the primitive has none */
        }
        return 1;
    }
    default:
        snprintf(why, cap, "not an FVEC kind");
        return 0;
    }
}

static int acc_fvec(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    fvec_plan q;
    if (!fvec_plan_make(p, &q, why, cap))
        return 0;
    /* aiednn_prefer_fastest only: decline where ref: is faster (per-element binary operands, norms) */
    if (p->kind == aiednn_binary && q.op == AIE_FV_PROG && !q.qb && rt_env_on("AIEDNN_NO_FVEC_ELEMBIN")) {
        snprintf(why, cap, "AIEDNN_NO_FVEC_ELEMBIN (a per-element second operand)");
        return 0;
    }
    if (rt_cost_refusals() && ((p->kind == aiednn_binary && q.op == AIE_FV_PROG && !q.qb) ||
                               (p->kind == aiednn_layer_normalization && !q.qd) || /* -> s8: ref: 4 ms (197 x 768) */
                               p->kind == aiednn_batch_normalization ||
                               /* global pooling: resnet18's 512x7x7 average 2.26 (fvec:) vs ~0.05 ms (ref:) */
                               (p->kind == aiednn_pooling && md_numel(&p->md[0]) < 65536))) {
        snprintf(why, cap, "ref: measured faster (aiednn_prefer_fastest)");
        return 0;
    }
    return 1;
}

static void *g_fv_in, *g_fv_aux, *g_fv_out;
static size_t g_fv_in_cap, g_fv_aux_cap, g_fv_out_cap;

/* n elements of an s8 / u8 / f32 tensor from element e on, as floats: one loop per type */
static inline void fvec_cvt(aiednn_data_type_t dt, const void *src, int64_t e, int64_t n, float *dst) {
    if (dt == aiednn_s8) {
        const int8_t *x = (const int8_t *)src + e;
        for (int64_t i = 0; i < n; i++)
            dst[i] = (float)x[i];
    } else if (dt == aiednn_u8) {
        const uint8_t *x = (const uint8_t *)src + e;
        for (int64_t i = 0; i < n; i++)
            dst[i] = (float)x[i];
    } else if (dt == aiednn_f32) {
        memcpy(dst, (const float *)src + e, (size_t)n * sizeof(float));
    } else {
        for (int64_t i = 0; i < n; i++)
            dst[i] = (float)rt_ld(dt, src, e + i);
    }
}

/* ROWPROG constants of block o (elements [o sp, (o + 1) sp)): c2, c3 */
static void fvec_row_consts(const rt_prob *p, const fvec_plan *q, void *const *ptr, int64_t o, float *c) {
    if (p->kind != aiednn_batch_normalization) { /* binary / prelu: the operand's value on the block */
        c[0] = ((const float *)ptr[q->s1])[bcast_off(&p->md[q->s1], rt_md(p, fvec_out_arg(p)), o * q->sp)];
        c[1] = 0.0f;
        return;
    }
    const int64_t ch = o % p->md[0].dims[1];
    const double mean = ld_opt(p, ptr, AIEDNN_ARG_MEAN, ch, 0), var = ld_opt(p, ptr, AIEDNN_ARG_VARIANCE, ch, 1);
    const double g = ld_opt(p, ptr, AIEDNN_ARG_SCALE, ch, 1), b = ld_opt(p, ptr, AIEDNN_ARG_SHIFT, ch, 0);
    const double m = g / sqrt(var + p->eps);
    c[0] = (float)m, c[1] = (float)(b - mean * m);
}

/* fill item k from element (or row) e on: returns the dst elements it covers */
static int64_t fvec_fill(const rt_prob *p, const fvec_plan *q, void *const *ptr, int64_t e, float *in, float *aux,
                         int32_t *geom) {
    const float *x = (const float *)ptr[0];
    int64_t n;
    if (q->L) { /* whole rows, each padded to Lp (softmax pads with -inf, norms need Lp == L) */
        const int64_t rows_left = (q->numel - e) / q->L,
                      rows = AIE_FVEC_N / q->Lp < rows_left ? AIE_FVEC_N / q->Lp : rows_left;
        for (int64_t r = 0; r < rows; r++) {
            fvec_gather(in + r * q->Lp, x, e / q->L + r, q->L, q->inner);
            for (int64_t i = 0; q->xf && i < q->L; i++) /* reduction min / amax */
                in[r * q->Lp + i] = q->xf == 1 ? -in[r * q->Lp + i] : fabsf(in[r * q->Lp + i]);
            for (int64_t i = q->L; i < q->Lp; i++)
                in[r * q->Lp + i] = q->op == AIE_FV_RSUM || q->op == AIE_FV_ROWPROG ? 0.0f : -INFINITY;
        }
        if (q->op == AIE_FV_ROWPROG) { /* the row's constants, then the program at AIE_FVEC_PROGAT */
            for (int64_t r = 0; r < rows; r++)
                fvec_row_consts(p, q, ptr, (e + r * q->L) / q->sp, aux + 2 * r);
            aie_fvec_prog_store(&q->prog, aux);
        }
        if (q->flags & AIE_FV_F_SCALE) { /* the norms: gamma then beta, 1 / 0 for an absent one */
            const float *g = rt_slot(p, AIEDNN_ARG_SCALE) >= 0 ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SCALE) : NULL;
            const float *bt =
                rt_slot(p, AIEDNN_ARG_SHIFT) >= 0 ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SHIFT) : NULL;
            for (int64_t i = 0; i < q->L; i++)
                aux[i] = g ? g[i] : 1.0f, aux[q->L + i] = bt ? bt[i] : 0.0f;
        }
        geom[0] = (int32_t)(rows * q->Lp), geom[1] = (int32_t)q->Lp;
        n = rows * q->L;
    } else {
        n = q->numel - e < AIE_FVEC_N ? q->numel - e : AIE_FVEC_N;
        if (q->qb) /* the quantized binary: s8 / u8 codes as floats (exact) */
            fvec_cvt(p->md[0].data_type, ptr[0], e, n, in);
        else
            memcpy(in, x + e, (size_t)n * sizeof(float));
        const int64_t np = (n + 15) / 16 * 16;
        for (int64_t i = n; i < np; i++)
            in[i] = 1.0f; /* defined lanes (sqrt / div of the padding) */
        if (q->s1 >= 0) { /* src1 element for element, broadcast to dst's shape */
            const aiednn_memory_desc_t *m1 = &p->md[q->s1], *d = rt_md(p, fvec_out_arg(p));
            if (md_same_shape(m1, d)) /* no broadcast: a typed contiguous copy */
                fvec_cvt(q->qb ? m1->data_type : aiednn_f32, ptr[q->s1], e, n, aux);
            else
                for (int64_t i = 0; i < n; i++)
                    aux[i] = q->qb ? (float)rt_ld(m1->data_type, ptr[q->s1], bcast_off(m1, d, e + i))
                                   : ((const float *)ptr[q->s1])[bcast_off(m1, d, e + i)];
            for (int64_t i = n; i < np; i++)
                aux[i] = 1.0f;
        }
        geom[0] = (int32_t)np, geom[1] = 0;
        aie_fvec_prog_store(&q->prog, aux);
    }
    geom[2] = q->op | (q->flags << 8), geom[3] = 0;
    return n;
}

/* one launch of the FVEC entry over its NI items */
static int fvec_launch(const rt_impl *self, float *in, float *aux, int32_t *geom, float *ab, float *out) {
    const int64_t NI = self->gen->args[0].dims[0];
    const int64_t shape_i[2] = {NI, AIE_FVEC_N}, shape_a[2] = {NI, AIE_FVEC_AUX}, shape_g[2] = {NI, 4},
                  shape_ab[2] = {1, 2};
    aiek_tensor t[5] = {{in, 2, AIEK_FLOAT, 32, 1, shape_i, (uint64_t)(NI * AIE_FVEC_N * 4)},
                        {aux, 2, AIEK_FLOAT, 32, 1, shape_a, (uint64_t)(NI * AIE_FVEC_AUX * 4)},
                        {geom, 2, AIEK_INT, 32, 1, shape_g, (uint64_t)(NI * 4 * sizeof(int32_t))},
                        {ab, 2, AIEK_FLOAT, 32, 1, shape_ab, 2 * sizeof(float)},
                        {out, 2, AIEK_FLOAT, 32, 1, shape_i, (uint64_t)(NI * AIE_FVEC_N * 4)}};
    ((rt_impl *)self)->launches++;
    const uint64_t te = rt_ticks();
    const int rc = self->gen->entry(t, 5);
    ((rt_impl *)self)->t_entry += rt_ticks() - te;
    return rc;
}

/* nrows rows of L (padded to Lp) through ROWPROG / ROWSUM with fl: fill(row, in[Lp], elem[Lp] or null, c[2])
 * writes a row's item data, take(row, out) reads its result (Lp floats, or the row's sum) */
template <class Fill, class Take>
static int fvec_rows_pass(const rt_impl *self, int op, int fl, const aie_fv_prog_t *prog, int64_t nrows, int64_t Lp,
                          Fill fill, Take take) {
    const int64_t NI = self->gen->args[0].dims[0], rpi = AIE_FVEC_N / Lp;
    float *in = (float *)scratch(&g_fv_in, &g_fv_in_cap, (size_t)(NI * AIE_FVEC_N) * sizeof(float));
    float *aux = (float *)scratch(&g_fv_aux, &g_fv_aux_cap, (size_t)(NI * AIE_FVEC_AUX) * sizeof(float));
    float *out = (float *)scratch(&g_fv_out, &g_fv_out_cap, (size_t)(NI * AIE_FVEC_N) * sizeof(float));
    if (!in || !aux || !out)
        return AIEK_ERR_ALLOC;
    int32_t geom[RT_FVEC_MAX_ITEMS * 4];
    int64_t r0[RT_FVEC_MAX_ITEMS], nr[RT_FVEC_MAX_ITEMS];
    float ab[2] = {0.0f, 1.0f};
    const int el = (fl & AIE_FV_F_ELEM) != 0;
    for (int64_t R = 0; R < nrows;) {
        const uint64_t tp0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++) {
            float *ik = in + k * AIE_FVEC_N, *ak = aux + k * AIE_FVEC_AUX, *cst = el ? ak + AIE_FVEC_ROWAT : ak;
            r0[k] = R, nr[k] = nrows - R < rpi ? nrows - R : rpi, R += nr[k];
            for (int64_t r = 0; r < nr[k]; r++)
                fill(r0[k] + r, ik + r * Lp, el ? ak + r * Lp : (float *)NULL, cst + 2 * r);
            aie_fvec_prog_store(prog, ak);
            geom[4 * k] = (int32_t)(nr[k] * Lp), geom[4 * k + 1] = nr[k] ? (int32_t)Lp : 0;
            geom[4 * k + 2] = op | (fl << 8), geom[4 * k + 3] = 0;
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        const int rc = fvec_launch(self, in, aux, geom, ab, out);
        if (rc != AIEK_OK)
            return rc;
        const uint64_t tc0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++)
            for (int64_t r = 0; r < nr[k]; r++)
                take(r0[k] + r, op == AIE_FV_ROWSUM ? out + k * AIE_FVEC_N + r : out + k * AIE_FVEC_N + r * Lp);
        g_phase[RT_PH_CUNPACK] += rt_ticks() - tc0;
    }
    return AIEK_OK;
}

/* one row of src (L floats) into an item row of Lp, padded with pad */
static inline void fvec_row_in(float *dst, const float *src, int64_t L, int64_t Lp, float pad) {
    memcpy(dst, src, (size_t)L * sizeof(float));
    for (int64_t i = L; i < Lp; i++)
        dst[i] = pad;
}

/* row R (strided: fvec_gather) into an item row of Lp, padded with 0 */
static inline void fvec_row_g(float *dst, const float *src, int64_t R, int64_t L, int64_t Lp, int64_t inner) {
    fvec_gather(dst, src, R, L, inner);
    for (int64_t i = L; i < Lp; i++)
        dst[i] = 0.0f;
}

/* softmax backward: s = sum y dy (log: sum dy) per row by ROWSUM, then y (dy - s) (log: dy - e^y s) */
static int run_fvec_smbwd(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    const float *y = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DST),
                *dy = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    float *dx = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    const int64_t L = q->L, Lp = q->Lp, rows = q->numel / L;
    const int lg = p->alg == aiednn_softmax_log;
    float *s = (float *)rt_alloc((size_t)rows * sizeof(float));
    if (!s)
        return AIEK_ERR_ALLOC;
    aie_fv_prog_t p1, p2;
    aie_fvec_prog(lg ? AIE_FVF_PICK1 : AIE_FVF_MUL, 0, 0, &p1);
    aie_fvec_prog(lg ? AIE_FVF_ROW_LSMBWD : AIE_FVF_ROW_SMBWD, 0, 0, &p2);
    int rc = fvec_rows_pass(
        self, AIE_FV_ROWSUM, AIE_FV_F_ELEM, &p1, rows, Lp,
        [&](int64_t R, float *in, float *e, float *c) { /* pads 0: they add 0 to both sums */
                                                        fvec_row_g(in, y, R, L, Lp, q->inner),
                                                            fvec_row_g(e, dy, R, L, Lp, q->inner), c[0] = c[1] = 0.0f;
        },
        [&](int64_t R, const float *o) { s[R] = *o; });
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWPROG, AIE_FV_F_ELEM, &p2, rows, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_g(in, y, R, L, Lp, q->inner), fvec_row_g(e, dy, R, L, Lp, q->inner), c[0] = s[R], c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { fvec_scatter(dx, o, R, L, q->inner); });
    rt_free(s);
    return rc;
}

/* batch norm training / group norm forward: block sums, centred squares (ROWSUM), then the per-channel
 * affine (ROWPROG); two-pass statistics in double as ref:. MEAN / VARIANCE written when present. */
static int run_fvec_blocknorm(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    const aiednn_memory_desc_t *xm = rt_md(p, AIEDNN_ARG_SRC);
    const float *x = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC);
    float *y = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DST);
    float *mo = rt_slot(p, AIEDNN_ARG_MEAN) >= 0 ? (float *)rt_ptr(p, ptr, AIEDNN_ARG_MEAN) : NULL;
    float *vo = rt_slot(p, AIEDNN_ARG_VARIANCE) >= 0 ? (float *)rt_ptr(p, ptr, AIEDNN_ARG_VARIANCE) : NULL;
    const int gn = p->kind == aiednn_group_normalization;
    const int64_t N = xm->dims[0], C = xm->dims[1], L = q->L, Lp = q->Lp, sp = q->sp, rows = q->numel / L;
    const int64_t G = gn ? p->groups : C, ns = gn ? N * G : C; /* statistics blocks */
    const double cnt = (double)(q->numel / ns);
    double *acc = (double *)rt_calloc((size_t)(2 * ns), sizeof(double));
    float *cs = (float *)rt_alloc((size_t)(ns + 2 * C * (gn ? N : 1)) *
                                  sizeof(float)); /* mean, then c2, c3 per affine row key */
    if (!acc || !cs) {
        rt_free(acc), rt_free(cs);
        return AIEK_ERR_ALLOC;
    }
    const auto ch = [&](int64_t R) {                               /* the statistics block of row R */
                                     const int64_t o = R * L / sp; /* n C + c */
                                     return gn ? (o / C) * G + (o % C) / (C / G) : o % C;
    };
    const auto ak = [&](int64_t R) { return gn ? R * L / sp : (R * L / sp) % C; }; /* the affine key: (n, c) or c */
    const int64_t na = gn ? N * C : C;
    aie_fv_prog_t pid = {}, psq, paff;
    aie_fvp_i(&pid, AIE_FVI_MOV, 0, 0, 0);
    aie_fvec_prog(AIE_FVF_ROW_SQDEV, 0, 0, &psq);
    aie_fvec_prog(AIE_FVF_ROW_AFFINE, 0, 0, &paff);
    int rc = fvec_rows_pass(
        self, AIE_FV_ROWSUM, 0, &pid, rows, Lp,
        [&](int64_t R, float *in, float *, float *c) { fvec_row_in(in, x + R * L, L, Lp, 0.0f), c[0] = c[1] = 0.0f; },
        [&](int64_t R, const float *o) { acc[ch(R)] += *o; });
    for (int64_t c = 0; c < ns; c++)
        cs[c] = (float)(acc[c] / cnt);
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, 0, &psq, rows, Lp,
            [&](int64_t R, float *in, float *, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, cs[ch(R)]), c[0] = cs[ch(R)], c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { acc[ns + ch(R)] += *o; });
    for (int64_t s = 0; rc == AIEK_OK && s < ns; s++) {
        if (mo)
            mo[s] = cs[s];
        if (vo)
            vo[s] = (float)(acc[ns + s] / cnt);
    }
    for (int64_t k = 0; rc == AIEK_OK && k < na; k++) { /* per affine key: channel c (and image n) */
        const int64_t c = k % C, s = gn ? (k / C) * G + c / (C / G) : c;
        const double var = acc[ns + s] / cnt, mean = cs[s];
        const double g = ld_opt(p, ptr, AIEDNN_ARG_SCALE, c, 1), b = ld_opt(p, ptr, AIEDNN_ARG_SHIFT, c, 0);
        const double m = g / sqrt(var + p->eps);
        cs[ns + k] = (float)m, cs[ns + na + k] = (float)(b - mean * m);
    }
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWPROG, 0, &paff, rows, Lp,
            [&](int64_t R, float *in, float *, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, 0.0f), c[0] = cs[ns + ak(R)], c[1] = cs[ns + na + ak(R)];
            },
            [&](int64_t R, const float *o) { memcpy(y + R * L, o, (size_t)L * sizeof(float)); });
    rt_free(acc), rt_free(cs);
    return rc;
}

/* the quantized binary's results into dst: q(f) = saturate(round_even(f / dst_scale) + dst_zp), or f */
static void fvec_qstore(const rt_prob *p, void *const *ptr, int64_t e, const float *o, int64_t n) {
    const aiednn_data_type_t dt = p->md[2].data_type;
    if (dt == aiednn_f32) {
        memcpy((float *)ptr[2] + e, o, (size_t)n * sizeof(float));
        return;
    }
    const double lo = dt == aiednn_s8 ? -128.0 : 0.0, hi = lo + 255.0, ds = p->dst_scale, zp = p->dst_zp;
    /* round_even(x): (x + 1.5 2^52) - 1.5 2^52, exact for |x| < 2^51 in the default rounding mode */
    const auto q = [&](float v) {
        const double x = (double)v / ds;
        double r = x >= 1e15 ? hi : x <= -1e15 ? lo : (x + 6755399441055744.0) - 6755399441055744.0 + zp;
        return r != r ? 0.0 : r < lo ? lo : r > hi ? hi : r;
    };
    if (dt == aiednn_s8) {
        int8_t *y = (int8_t *)ptr[2] + e;
        for (int64_t i = 0; i < n; i++)
            y[i] = (int8_t)q(o[i]);
    } else {
        uint8_t *y = (uint8_t *)ptr[2] + e;
        for (int64_t i = 0; i < n; i++)
            y[i] = (uint8_t)q(o[i]);
    }
}

/* a one-pass plan: items of elements or whole rows, one launch per NI items */
static int run_fvec_items(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *qp) {
    const fvec_plan &q = *qp;
    const int64_t NI = self->gen->args[0].dims[0];
    float *in = (float *)scratch(&g_fv_in, &g_fv_in_cap, (size_t)(NI * AIE_FVEC_N) * sizeof(float));
    float *aux = (float *)scratch(&g_fv_aux, &g_fv_aux_cap, (size_t)(NI * AIE_FVEC_AUX) * sizeof(float));
    float *out = (float *)scratch(&g_fv_out, &g_fv_out_cap, (size_t)(NI * AIE_FVEC_N) * sizeof(float));
    if (!in || !aux || !out)
        return AIEK_ERR_ALLOC;
    float *y = (float *)rt_ptr(p, ptr, fvec_out_arg(p));
    int32_t geom[RT_FVEC_MAX_ITEMS * 4];
    int64_t e0s[RT_FVEC_MAX_ITEMS], ns[RT_FVEC_MAX_ITEMS];
    float ab[2] = {q.a, q.b};
    for (int64_t e = 0; e < q.numel;) {
        const uint64_t tp0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++) {
            e0s[k] = e, ns[k] = 0, geom[4 * k] = geom[4 * k + 1] = 0, geom[4 * k + 2] = q.op | (q.flags << 8),
            geom[4 * k + 3] = 0;
            if (e < q.numel)
                e += ns[k] = fvec_fill(p, &q, ptr, e, in + k * AIE_FVEC_N, aux + k * AIE_FVEC_AUX, geom + 4 * k);
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        const int rc = fvec_launch(self, in, aux, geom, ab, out);
        if (rc != AIEK_OK)
            return rc;
        const uint64_t tc0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++) {
            if (!ns[k])
                continue;
            if (q.red) /* one value per row (min: negated back) */
                for (int64_t r = 0; r < ns[k] / q.L; r++)
                    y[e0s[k] / q.L + r] = q.xf == 1 ? -out[k * AIE_FVEC_N + r] : out[k * AIE_FVEC_N + r];
            else if (q.L && q.qd) /* norm rows quantized: the quantize reorder's arithmetic (rt_quant8) */
                for (int64_t r = 0; r < ns[k] / q.L; r++) {
                    const double ds = p->dst_scale, inv = 1.0 / ds, zp = p->dst_zp;
                    const double lo = rt_md(p, fvec_out_arg(p))->data_type == aiednn_s8 ? -128.0 : 0.0;
                    const float *src = out + k * AIE_FVEC_N + r * q.Lp;
                    uint8_t *dst = (uint8_t *)y + (e0s[k] / q.L + r) * q.L;
                    for (int64_t c = 0; c < q.L; c++)
                        dst[c] = (uint8_t)(int)rt_quant8(src[c], ds, inv, zp, lo);
                }
            else if (q.L) /* rows back without their padding */
                for (int64_t r = 0; r < ns[k] / q.L; r++)
                    fvec_scatter(y, out + k * AIE_FVEC_N + r * q.Lp, e0s[k] / q.L + r, q.L, q.inner);
            else if (q.qb) /* the quantize reorder's arithmetic into an s8 / u8 dst (f32: a copy) */
                fvec_qstore(p, ptr, e0s[k], out + k * AIE_FVEC_N, ns[k]);
            else
                memcpy(y + e0s[k], out + k * AIE_FVEC_N, (size_t)ns[k] * sizeof(float));
        }
        g_phase[RT_PH_CUNPACK] += rt_ticks() - tc0;
    }
    return AIEK_OK;
}

/* eltwise backward whose derivative needs more registers than one program has, as element passes over (src, AUX):
 *   tan       t = tan(x); dx = dy + dy t^2
 *   gelu_erf  e = erf(x / sqrt 2); g = 0.5 + 0.5 e + x e^(-x^2 / 2) / sqrt(2 pi); dx = g dy
 * each pass the eltwise item runner with its own program and buffers */
static int run_fvec_bwd_passes(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    const int sx = rt_slot(p, AIEDNN_ARG_SRC), so = rt_slot(p, AIEDNN_ARG_DIFF_SRC), sa = q->s1;
    const int64_t n = q->numel;
    float *t1 = (float *)rt_alloc((size_t)n * sizeof(float)), *t2 = (float *)rt_alloc((size_t)n * sizeof(float));
    int rc = t1 && t2 && sx >= 0 && so >= 0 ? AIEK_OK : AIEK_ERR_ALLOC;
    void *pp[RT_MAX_MD];
    const auto pass = [&](const aie_fv_prog_t *prog, void *src, void *aux, void *out) {
        for (int i = 0; i < p->nmd; i++)
            pp[i] = ptr[i];
        pp[sx] = src, pp[sa] = aux, pp[so] = out;
        fvec_plan qq = *q;
        qq.prog = *prog, qq.train = 0;
        return run_fvec_items(self, p, pp, &qq);
    };
    aie_fv_prog_t g1, g2, g3;
    memset(&g1, 0, sizeof g1), memset(&g2, 0, sizeof g2), memset(&g3, 0, sizeof g3);
    if (p->alg == aiednn_eltwise_tan) {
        aie_fvec_prog(AIE_FVF_TAN, 0, 0, &g1);
        aie_fvp_i(&g2, AIE_FVI_MUL, 2, 0, 0), aie_fvp_i(&g2, AIE_FVI_MUL, 2, 2, 1),
            aie_fvp_i(&g2, AIE_FVI_ADD, 0, 2, 1);
        if (rc == AIEK_OK)
            rc = pass(&g1, ptr[sx], ptr[sa], t1);
        if (rc == AIEK_OK)
            rc = pass(&g2, t1, ptr[sa], ptr[so]);
    } else {
        aie_fv_prog_t er;
        aie_fvp_c(&g1, AIE_FVI_MULC, 0, 0, 0.70710678118654752f), aie_fvec_prog(AIE_FVF_ERF, 0, 0, &er);
        fvec_prog_append(&g1, &er);
        aie_fvp_i(&g2, AIE_FVI_MUL, 2, 0, 0), aie_fvp_c(&g2, AIE_FVI_MULC, 2, 2, -0.5f),
            aie_fvp_i(&g2, AIE_FVI_EXP, 2, 2, 0);
        aie_fvp_c(&g2, AIE_FVI_MULC, 2, 2, 0.39894228040143268f), aie_fvp_i(&g2, AIE_FVI_MUL, 2, 2, 0);
        aie_fvp_c(&g2, AIE_FVI_MULC, 1, 1, 0.5f), aie_fvp_c(&g2, AIE_FVI_ADDC, 1, 1, 0.5f),
            aie_fvp_i(&g2, AIE_FVI_ADD, 0, 1, 2);
        aie_fvp_i(&g3, AIE_FVI_MUL, 0, 0, 1);
        if (rc == AIEK_OK)
            rc = pass(&g1, ptr[sx], ptr[sa], t1);
        if (rc == AIEK_OK)
            rc = pass(&g2, ptr[sx], t1, t2);
        if (rc == AIEK_OK)
            rc = pass(&g3, t2, ptr[sa], ptr[so]);
    }
    rt_free(t1), rt_free(t2);
    return rc;
}

/* select(cond, a, b) = cond ? a : b as element passes (broadcast operands expanded on the host first), each output
 * exactly one source or 0 before the add:
 * t = [cond != 0] a, u = [cond == 0] b (SELP), d = t + u (a -0.0 selected comes back as +0.0) */
static int run_fvec_select(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    const int sa = 0, sb = q->s1, sd = rt_slot(p, AIEDNN_ARG_DST), sc = rt_slot(p, AIEDNN_ARG_SRC_2);
    const int64_t n = q->numel;
    float *c = (float *)rt_alloc((size_t)n * sizeof(float)), *t = (float *)rt_alloc((size_t)n * sizeof(float));
    float *u = (float *)rt_alloc((size_t)n * sizeof(float));
    const aiednn_memory_desc_t *dm = &p->md[sd];
    const int ba = !md_same_shape(&p->md[sa], dm), bb = !md_same_shape(&p->md[sb], dm);
    float *ea = ba ? (float *)rt_alloc((size_t)n * sizeof(float)) : nullptr; /* broadcast operands, expanded */
    float *eb = bb ? (float *)rt_alloc((size_t)n * sizeof(float)) : nullptr;
    int rc = c && t && u && (!ba || ea) && (!bb || eb) ? AIEK_OK : AIEK_ERR_ALLOC;
    for (int64_t i = 0; rc == AIEK_OK && i < n; i++) { /* cond as 0 / 1 floats */
        c[i] = rt_ld(p->md[sc].data_type, ptr[sc], bcast_off(&p->md[sc], dm, i)) != 0 ? 1.0f : 0.0f;
        if (ba)
            ea[i] = ((const float *)ptr[sa])[bcast_off(&p->md[sa], dm, i)];
        if (bb)
            eb[i] = ((const float *)ptr[sb])[bcast_off(&p->md[sb], dm, i)];
    }
    void *pp[RT_MAX_MD];
    const auto pass = [&](const aie_fv_prog_t *prog, void *src, void *aux, void *out) {
        for (int i = 0; i < p->nmd; i++)
            pp[i] = ptr[i];
        pp[sa] = src, pp[sb] = aux, pp[sd] = out;
        fvec_plan qq = *q;
        qq.prog = *prog, qq.train = 0;
        return run_fvec_items(self, p, pp, &qq);
    };
    aie_fv_prog_t g1, g2, g3;
    memset(&g1, 0, sizeof g1), memset(&g2, 0, sizeof g2), memset(&g3, 0, sizeof g3);
    aie_fvp_i(&g1, AIE_FVI_SELP, 0, 0, 1);                                               /* c > 0 ? a : 0 */
    aie_fvp_c(&g2, AIE_FVI_MULC, 0, 0, -1.0f), aie_fvp_c(&g2, AIE_FVI_ADDC, 0, 0, 1.0f); /* 1 - c */
    aie_fvp_i(&g2, AIE_FVI_SELP, 0, 0, 1);                                               /* c == 0 ? b : 0 */
    aie_fvp_i(&g3, AIE_FVI_ADD, 0, 0, 1);
    fvec_plan qd = *q; /* the passes see dst-shaped operands */
    qd.s1 = sb;
    rt_prob pd = *p;
    pd.md[sa] = *dm, pd.md[sb] = *dm, pd.md[sb].data_type = aiednn_f32;
    const auto pass2 = [&](const aie_fv_prog_t *prog, void *src, void *aux, void *out) {
        for (int i = 0; i < p->nmd; i++)
            pp[i] = ptr[i];
        pp[sa] = src, pp[sb] = aux, pp[sd] = out;
        fvec_plan qq = qd;
        qq.prog = *prog, qq.train = 0;
        return run_fvec_items(self, &pd, pp, &qq);
    };
    (void)pass;
    if (rc == AIEK_OK)
        rc = pass2(&g1, c, ba ? ea : ptr[sa], t);
    if (rc == AIEK_OK)
        rc = pass2(&g2, c, bb ? eb : ptr[sb], u);
    if (rc == AIEK_OK)
        rc = pass2(&g3, t, u, ptr[sd]);
    rt_free(c), rt_free(t), rt_free(u), rt_free(ea), rt_free(eb);
    return rc;
}

/* pow with a non-integer exponent: the program for x > 0, then x <= 0 as ref: computes it (a pow(x, b): NaN for
 * x < 0, 0 or inf at 0) -- the core has no NaN to produce, and its log at 0 is not -inf */
static int run_fvec_powpatch(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    fvec_plan qq = *q;
    qq.train = 0;
    const int rc = run_fvec_items(self, p, ptr, &qq);
    const float *x = (const float *)ptr[0];
    float *y = (float *)rt_ptr(p, ptr, fvec_out_arg(p));
    const int64_t n = md_numel(&p->md[0]);
    for (int64_t i = 0; rc == AIEK_OK && i < n; i++)
        if (!(x[i] > 0.0f))
            y[i] = (float)eltwise_f(p->alg, (double)x[i], p->alpha, p->beta);
    return rc;
}

/* layer norm training forward: y by the LNORM rows (the inference plan), MEAN / VARIANCE per row by ROWSUM of x
 * and of (x - mean)^2 (RMS: mean 0, the second pass only) */
static int run_fvec_lntrain(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    rt_prob pi = *p;
    pi.prop = RT_FWD_INF;
    fvec_plan qi;
    char why[160];
    if (!fvec_plan_make(&pi, &qi, why, sizeof why))
        return AIEK_ERR_ARGS;
    int rc = run_fvec_items(self, &pi, ptr, &qi);
    const float *x = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC);
    float *mo = (float *)rt_ptr(p, ptr, AIEDNN_ARG_MEAN), *vo = (float *)rt_ptr(p, ptr, AIEDNN_ARG_VARIANCE);
    const int64_t L = q->L, rows = q->numel / L;
    aie_fv_prog_t pid = {}, psq;
    aie_fvp_i(&pid, AIE_FVI_MOV, 0, 0, 0);
    aie_fvec_prog(AIE_FVF_ROW_SQDEV, 0, 0, &psq);
    const float inv = 1.0f / (float)L;
    if (rc == AIEK_OK && (p->flags & AIEDNN_RMS_NORM))
        for (int64_t r = 0; r < rows; r++)
            mo[r] = 0.0f;
    else if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, 0, &pid, rows, L,
            [&](int64_t R, float *in, float *, float *c) {
                memcpy(in, x + R * L, (size_t)L * sizeof(float)), c[0] = c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { mo[R] = (float)((double)*o / (double)L); });
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, 0, &psq, rows, L,
            [&](int64_t R, float *in, float *, float *c) {
                memcpy(in, x + R * L, (size_t)L * sizeof(float)), c[0] = mo[R], c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { vo[R] = *o * inv; });
    return rc;
}

/* batch / group norm backward (ref:'s run_norm_bwd): per row sum dy and sum dy xhat (ROWSUM), the block sums
 * SG = sum scale dy, SGX = sum scale dy xhat on the host, then dx = rstd (scale dy - SG / L - xhat SGX / L) as
 * t = (rstd scale) dy + (-rstd^2 SGX / L) x (ROWPROG, LIN2) and t + (rstd^2 SGX mean / L - rstd SG / L)
 * (ROWPROG, affine 1, c). DIFF_SCALE / DIFF_SHIFT: the per-channel sums of the two ROWSUMs. */
static int run_fvec_normbwd(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    const aiednn_memory_desc_t *xm = rt_md(p, AIEDNN_ARG_SRC);
    const float *x = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC),
                *dy = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    const float *mu = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_MEAN),
                *va = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_VARIANCE);
    float *dx = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    float *dsc = rt_slot(p, AIEDNN_ARG_DIFF_SCALE) >= 0 ? (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SCALE) : NULL;
    float *dsh = rt_slot(p, AIEDNN_ARG_DIFF_SHIFT) >= 0 ? (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SHIFT) : NULL;
    const int gn = p->kind == aiednn_group_normalization;
    const int64_t N = xm->dims[0], C = xm->dims[1], L = q->L, Lp = q->Lp, sp = q->sp, rows = q->numel / L;
    const int64_t G = gn ? p->groups : C, ns = gn ? N * G : C, na = gn ? N * C : C;
    const double cnt = (double)(q->numel / ns);
    double *acc = (double *)rt_calloc((size_t)(2 * ns + 2 * C), sizeof(double)); /* SG, SGX per block; dshift, dscale */
    float *cs = (float *)rt_alloc((size_t)(2 * ns + 3 * na) * sizeof(float));    /* mean, rstd; c2, c3, c */
    float *rs = (float *)rt_alloc((size_t)(2 * rows) * sizeof(float));           /* per row: sum dy, sum dy xhat */
    int rc = acc && cs && rs ? AIEK_OK : AIEK_ERR_ALLOC;
    const auto blk = [&](int64_t R) {
        const int64_t o = R * L / sp;
        return gn ? (o / C) * G + (o % C) / (C / G) : o % C;
    };
    const auto key = [&](int64_t R) { return gn ? R * L / sp : (R * L / sp) % C; };
    for (int64_t b = 0; rc == AIEK_OK && b < ns; b++)
        cs[b] = mu[b], cs[ns + b] = (float)(1.0 / sqrt((double)va[b] + p->eps));
    aie_fv_prog_t p1, p2, p3, p4;
    aie_fvec_prog(AIE_FVF_PICK1, 0, 0, &p1), aie_fvec_prog(AIE_FVF_ROW_DXHAT, 0, 0, &p2);
    aie_fvec_prog(AIE_FVF_ROW_LIN2, 0, 0, &p3), aie_fvec_prog(AIE_FVF_ROW_AFFINE, 0, 0, &p4);
    if (rc == AIEK_OK) /* pads 0 (dy) and the mean (x): both sums add 0 */
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, AIE_FV_F_ELEM, &p1, rows, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, 0.0f), fvec_row_in(e, dy + R * L, L, Lp, 0.0f), c[0] = c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { rs[R] = *o; });
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, AIE_FV_F_ELEM, &p2, rows, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, cs[blk(R)]), fvec_row_in(e, dy + R * L, L, Lp, 0.0f);
                c[0] = cs[blk(R)], c[1] = cs[ns + blk(R)];
            },
            [&](int64_t R, const float *o) { rs[rows + R] = *o; });
    for (int64_t R = 0; rc == AIEK_OK && R < rows; R++) { /* block and channel sums */
        const int64_t c = key(R) % C;
        const double g = ld_opt(p, ptr, AIEDNN_ARG_SCALE, c, 1.0);
        acc[blk(R)] += g * rs[R], acc[ns + blk(R)] += g * rs[rows + R];
        acc[2 * ns + c] += rs[R], acc[2 * ns + C + c] += rs[rows + R];
    }
    for (int64_t k = 0; rc == AIEK_OK && k < na; k++) {
        const int64_t c = k % C, b = gn ? (k / C) * G + c / (C / G) : c;
        const double r = cs[ns + b], m = cs[b], g = ld_opt(p, ptr, AIEDNN_ARG_SCALE, c, 1.0);
        const double sgl = acc[b] / cnt, sgxl = acc[ns + b] / cnt;
        cs[2 * ns + k] = (float)(r * g), cs[2 * ns + na + k] = (float)(-r * r * sgxl);
        cs[2 * ns + 2 * na + k] = (float)(r * r * sgxl * m - r * sgl);
    }
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWPROG, AIE_FV_F_ELEM, &p3, rows, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, 0.0f), fvec_row_in(e, dy + R * L, L, Lp, 0.0f);
                c[0] = cs[2 * ns + key(R)], c[1] = cs[2 * ns + na + key(R)];
            },
            [&](int64_t R, const float *o) { memcpy(dx + R * L, o, (size_t)L * sizeof(float)); });
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWPROG, 0, &p4, rows, Lp,
            [&](int64_t R, float *in, float *, float *c) {
                fvec_row_in(in, dx + R * L, L, Lp, 0.0f), c[0] = 1.0f, c[1] = cs[2 * ns + 2 * na + key(R)];
            },
            [&](int64_t R, const float *o) { memcpy(dx + R * L, o, (size_t)L * sizeof(float)); });
    for (int64_t c = 0; rc == AIEK_OK && c < C; c++) {
        if (dsh)
            dsh[c] = (float)acc[2 * ns + c];
        if (dsc)
            dsc[c] = (float)acc[2 * ns + C + c];
    }
    rt_free(acc), rt_free(cs), rt_free(rs);
    return rc;
}

/* layer / RMS norm backward over the last dim D (run_norm_bwd's formula): with e = dy gamma, xhat = (x - mean) r,
 * dx = r (e - mean(e) - xhat mean(e xhat)) = r e - r^2 S2 / D x + (r^2 S2 / D mean - r S1 / D) [RMS: no mean, no S1].
 * Rows are chunks of L | D (ROWSUM chunk sums add up per row), so any D. S1 = sum e (PICK1), S2 = sum e xhat
 * (ROW_DXHAT); then ROW_LIN2 and ROW_AFFINE as group norm's backward. DIFF_SCALE / DIFF_SHIFT (column sums over
 * rows) on the host. */
static int run_fvec_lnbwd(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    const aiednn_memory_desc_t *xm = rt_md(p, AIEDNN_ARG_SRC);
    const float *x = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC),
                *dy = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    const int rms = (p->flags & AIEDNN_RMS_NORM) != 0;
    const float *mu = rms ? nullptr : (const float *)rt_ptr(p, ptr, AIEDNN_ARG_MEAN);
    const float *va = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_VARIANCE);
    const float *g = rt_slot(p, AIEDNN_ARG_SCALE) >= 0 ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SCALE) : nullptr;
    float *dx = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    float *dsc = rt_slot(p, AIEDNN_ARG_DIFF_SCALE) >= 0 ? (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SCALE) : NULL;
    float *dsh = rt_slot(p, AIEDNN_ARG_DIFF_SHIFT) >= 0 ? (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SHIFT) : NULL;
    const int64_t D = xm->dims[xm->ndims - 1], L = q->L, Lp = q->Lp, per = D / L, nrow = md_numel(xm) / D;
    const int64_t nch = nrow * per; /* chunks */
    float *rs = (float *)rt_alloc((size_t)(2 * nch + 4 * nrow) * sizeof(float));
    int rc = rs ? AIEK_OK : AIEK_ERR_ALLOC;
    float *mr = rs + 2 * nch, *rr = mr + nrow, *c2 = rr + nrow, *c3 = c2 + nrow; /* mean, rstd, coefficients */
    for (int64_t r = 0; rc == AIEK_OK && r < nrow; r++)
        mr[r] = rms ? 0.0f : mu[r], rr[r] = (float)(1.0 / sqrt((double)va[r] + p->eps));
    const auto fill_e = [&](int64_t R, float *e) { /* dy gamma, zero padded */
                                                   const int64_t off = R * L, c0 = (R % per) * L;
                                                   for (int64_t i = 0; i < L; i++)
                                                       e[i] = g ? dy[off + i] * g[c0 + i] : dy[off + i];
                                                   for (int64_t i = L; i < Lp; i++)
                                                       e[i] = 0.0f;
    };
    aie_fv_prog_t p1, p2, p3, p4;
    aie_fvec_prog(AIE_FVF_PICK1, 0, 0, &p1), aie_fvec_prog(AIE_FVF_ROW_DXHAT, 0, 0, &p2);
    aie_fvec_prog(AIE_FVF_ROW_LIN2, 0, 0, &p3), aie_fvec_prog(AIE_FVF_ROW_AFFINE, 0, 0, &p4);
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, AIE_FV_F_ELEM, &p1, nch, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, 0.0f), fill_e(R, e), c[0] = c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { rs[R] = *o; });
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, AIE_FV_F_ELEM, &p2, nch, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, mr[R / per]), fill_e(R, e), c[0] = mr[R / per], c[1] = rr[R / per];
            },
            [&](int64_t R, const float *o) { rs[nch + R] = *o; });
    for (int64_t r = 0; rc == AIEK_OK && r < nrow; r++) {
        double s1 = 0, s2 = 0;
        for (int64_t k = 0; k < per; k++)
            s1 += rs[r * per + k], s2 += rs[nch + r * per + k];
        const double rv = rr[r], m = mr[r], sg = rms ? 0.0 : s1 / (double)D, sgx = s2 / (double)D;
        c2[r] = (float)(-rv * rv * sgx), c3[r] = (float)(rv * rv * sgx * m - rv * sg);
    }
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWPROG, AIE_FV_F_ELEM, &p3, nch, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, 0.0f), fill_e(R, e), c[0] = rr[R / per], c[1] = c2[R / per];
            },
            [&](int64_t R, const float *o) { memcpy(dx + R * L, o, (size_t)L * sizeof(float)); });
    if (rc == AIEK_OK)
        rc = fvec_rows_pass(
            self, AIE_FV_ROWPROG, 0, &p4, nch, Lp,
            [&](int64_t R, float *in, float *, float *c) {
                fvec_row_in(in, dx + R * L, L, Lp, 0.0f), c[0] = 1.0f, c[1] = c3[R / per];
            },
            [&](int64_t R, const float *o) { memcpy(dx + R * L, o, (size_t)L * sizeof(float)); });
    if (rc == AIEK_OK && (dsc || dsh)) { /* column sums of dy xhat and dy (host) */
        double *acc = (double *)rt_calloc((size_t)(2 * D), sizeof(double));
        if (!acc)
            rc = AIEK_ERR_ALLOC;
        for (int64_t r = 0; rc == AIEK_OK && r < nrow; r++)
            for (int64_t i = 0; i < D; i++) {
                const double d = dy[r * D + i];
                acc[i] += d * (((double)x[r * D + i] - mr[r]) * (1.0 / sqrt((double)va[r] + p->eps))), acc[D + i] += d;
            }
        for (int64_t i = 0; rc == AIEK_OK && i < D; i++) {
            if (dsc)
                dsc[i] = (float)acc[i];
            if (dsh)
                dsh[i] = (float)acc[D + i];
        }
        rt_free(acc);
    }
    rt_free(rs);
    return rc;
}

/* layer / RMS norm inference over a last dim wider than one item (or not a multiple of 16): chunks of L | D,
 * ROWSUM statistics summed on the host, then ROWPROG gamma (x - mean) rstd + beta. */
static int run_fvec_lnwide(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    const aiednn_memory_desc_t *xm = &p->md[0];
    const float *x = (const float *)ptr[0];
    float *y = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DST);
    const int rms = (p->flags & AIEDNN_RMS_NORM) != 0;
    const float *g = rt_slot(p, AIEDNN_ARG_SCALE) >= 0 ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SCALE) : nullptr;
    const float *b =
        !rms && rt_slot(p, AIEDNN_ARG_SHIFT) >= 0 ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SHIFT) : nullptr;
    const int64_t D = xm->dims[xm->ndims - 1], L = q->L, Lp = q->Lp, per = D / L, nrow = md_numel(xm) / D;
    const int64_t nch = nrow * per;
    float *rs = (float *)rt_alloc((size_t)(nch + 2 * nrow) * sizeof(float));
    int rc = rs ? AIEK_OK : AIEK_ERR_ALLOC;
    float *mr = rs + nch, *rr = mr + nrow;
    const auto fill_param = [&](int64_t R, float *e, const float *v, float dflt) {
        const int64_t c0 = (R % per) * L;
        for (int64_t i = 0; i < L; i++)
            e[i] = v ? v[c0 + i] : dflt;
        for (int64_t i = L; i < Lp; i++)
            e[i] = 0.0f;
    };
    aie_fv_prog_t pid, psq, pxh, padd;
    aie_fvec_prog(AIE_FVF_LINEAR, 1.0f, 0.0f, &pid), aie_fvec_prog(AIE_FVF_ROW_SQDEV, 0, 0, &psq);
    aie_fvec_prog(AIE_FVF_ROW_DXHAT, 0, 0, &pxh), aie_fvec_prog(AIE_FVF_ADD, 0, 0, &padd);
    if (rc == AIEK_OK && !rms) /* row sums (pads 0) */
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, 0, &pid, nch, Lp,
            [&](int64_t R, float *in, float *, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, 0.0f), c[0] = c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { rs[R] = *o; });
    for (int64_t r = 0; rc == AIEK_OK && r < nrow; r++) {
        double s1 = 0;
        for (int64_t k = 0; !rms && k < per; k++)
            s1 += rs[r * per + k];
        mr[r] = (float)(s1 / (double)D);
    }
    if (rc == AIEK_OK) /* centred squares (pads the mean: they add 0) */
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, 0, &psq, nch, Lp,
            [&](int64_t R, float *in, float *, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, mr[R / per]), c[0] = mr[R / per], c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { rs[R] = *o; });
    for (int64_t r = 0; rc == AIEK_OK && r < nrow; r++) {
        double s2 = 0;
        for (int64_t k = 0; k < per; k++)
            s2 += rs[r * per + k];
        rr[r] = (float)(1.0 / sqrt(s2 / (double)D + p->eps));
    }
    if (rc == AIEK_OK) /* gamma (x - mean) rstd */
        rc = fvec_rows_pass(
            self, AIE_FV_ROWPROG, AIE_FV_F_ELEM, &pxh, nch, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, 0.0f), fill_param(R, e, g, 1.0f), c[0] = mr[R / per],
                                                                                    c[1] = rr[R / per];
            },
            [&](int64_t R, const float *o) { memcpy(y + R * L, o, (size_t)L * sizeof(float)); });
    if (rc == AIEK_OK && b) /* + beta */
        rc = fvec_rows_pass(
            self, AIE_FV_ROWPROG, AIE_FV_F_ELEM, &padd, nch, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, y + R * L, L, Lp, 0.0f), fill_param(R, e, b, 0.0f), c[0] = c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { memcpy(y + R * L, o, (size_t)L * sizeof(float)); });
    rt_free(rs);
    return rc;
}

/* PReLU backward: dx = x > 0 ? dy : w dy (ROWPROG, w the row constant), dw = sum over w's block of dy min(x, 0)
 * (ROWSUM per row, summed per weight on the host) */
static int run_fvec_prelubwd(const rt_impl *self, const rt_prob *p, void *const *ptr, const fvec_plan *q) {
    const aiednn_memory_desc_t *xm = rt_md(p, AIEDNN_ARG_SRC), *wm = rt_md(p, AIEDNN_ARG_WEIGHTS);
    const float *x = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC),
                *w = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS);
    const float *dy = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    float *dx = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), *dw = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_WEIGHTS);
    const int64_t L = q->L, Lp = q->Lp, rows = q->numel / L, nw = md_numel(wm);
    double *acc = (double *)rt_calloc((size_t)nw, sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    const auto wi = [&](int64_t R) { return bcast_off(wm, xm, R * L); };
    aie_fv_prog_t pdx, pdw;
    aie_fvec_prog(AIE_FVF_ROW_PRELU_DX, 0, 0, &pdx), aie_fvec_prog(AIE_FVF_ROW_PRELU_DW, 0, 0, &pdw);
    int rc = fvec_rows_pass(
        self, AIE_FV_ROWPROG, AIE_FV_F_ELEM, &pdx, rows, Lp,
        [&](int64_t R, float *in, float *e, float *c) {
            fvec_row_in(in, x + R * L, L, Lp, 0.0f), fvec_row_in(e, dy + R * L, L, Lp, 0.0f), c[0] = w[wi(R)],
                                                                                              c[1] = 0.0f;
        },
        [&](int64_t R, const float *o) { memcpy(dx + R * L, o, (size_t)L * sizeof(float)); });
    if (rc == AIEK_OK) /* pads 0: dy 0 adds 0 */
        rc = fvec_rows_pass(
            self, AIE_FV_ROWSUM, AIE_FV_F_ELEM, &pdw, rows, Lp,
            [&](int64_t R, float *in, float *e, float *c) {
                fvec_row_in(in, x + R * L, L, Lp, 0.0f), fvec_row_in(e, dy + R * L, L, Lp, 0.0f), c[0] = c[1] = 0.0f;
            },
            [&](int64_t R, const float *o) { acc[wi(R)] += *o; });
    for (int64_t i = 0; rc == AIEK_OK && i < nw; i++)
        dw[i] = (float)acc[i];
    rt_free(acc);
    return rc;
}

static int run_fvec(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    fvec_plan q;
    char why[160];
    if (!fvec_plan_make(p, &q, why, sizeof why))
        return AIEK_ERR_ARGS;
    switch (q.train) {
    case 1:
        return run_fvec_smbwd(self, p, ptr, &q);
    case 2:
        return run_fvec_blocknorm(self, p, ptr, &q);
    case 3:
        return run_fvec_lntrain(self, p, ptr, &q);
    case 4:
        return run_fvec_normbwd(self, p, ptr, &q);
    case 5:
        return run_fvec_prelubwd(self, p, ptr, &q);
    case 6:
        return run_fvec_lnbwd(self, p, ptr, &q);
    case 7:
        return run_fvec_lnwide(self, p, ptr, &q);
    case 9:
        return run_fvec_bwd_passes(self, p, ptr, &q);
    case 10:
        return run_fvec_select(self, p, ptr, &q);
    case 11:
        return run_fvec_powpatch(self, p, ptr, &q);
    default:
        return run_fvec_items(self, p, ptr, &q);
    }
}

#endif /* AIEDNN_RT_FVEC_H */
