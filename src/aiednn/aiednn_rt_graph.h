/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_graph.h -- the graph API (include/aiednn_graph.h).
 *
 * Partitioning, in topological order:
 *   - an anchor (convolution, matmul, inner product) absorbs the single-consumer chain after it:
 *     bias_add, up to AIEDNN_MAX_POST_OPS eltwise / binary ops, then a per-tensor quantize. With
 *     src and weights from dequantize ops of 8-bit tensors, the anchor reads the 8-bit tensors directly.
 *   - layer / batch norm absorb a following per-tensor quantize.
 *   - every other supported op is a partition of its own; a dequantize whose consumers all read
 *     through it joins the first of their partitions, otherwise it is a reorder of its own.
 * Support is never guessed: each candidate is created through aiednn.h and destroyed; on failure the
 * chain is shortened, then the 8-bit read-through dropped, then the op is unsupported.
 *
 * Uses only aiednn.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_GRAPH_H
#define AIEDNN_RT_GRAPH_H

#include "aiednn_graph.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GR_MAX_IN AIEDNN_MAX_SRCS /* concat */
#define GR_MAX_BIND (GR_MAX_IN + 8)
#define GR_MAX_DQ 4
#define GR_MAX_CHAIN 16  /* an anchor + the ops its epilogue absorbs */
#define GR_ARG_EPS 0x7E5 /* a binding the compiled partition reads itself (a norm's eps tensor), never the primitive   \
                          */

typedef struct {
    aiednn_logical_tensor_t lt;
    int producer; /* op index, -1: graph input */
    int nuse;
    int marked;
} gr_tensor;

typedef struct {
    uint64_t id;
    aiednn_graph_op_kind_t kind;
    aiednn_graph_op_attr_t attr; /* scales / zero_points point at the op's own copies */
    int nin;
    int *in;           /* tensor indices */
    int out;           /* = outs[0] */
    int nout, outs[4]; /* the training ops have several */
    int topo;
    int part;
} gr_op;

enum {
    GP_NONE,
    GP_CONV,
    GP_MATMUL,
    GP_IP,
    GP_ELT,
    GP_BIN,
    GP_POOL,
    GP_SOFTMAX,
    GP_LNORM,
    GP_BNORM,
    GP_CONCAT,
    GP_REORDER,
    GP_COPY,
    GP_SDPA,
    GP_DECONV,
    GP_RESAMPLE,
    GP_REDUCE,
    GP_GNORM,
    GP_PRELU,
    GP_QBIN,
    GP_QELT,
    GP_SLICE,
    GP_ROPE,
    GP_QCAT,
    GP_GMLP,
    /* training (md[0] src / dst, md[1] diff_dst / weights, md[2] weights / scale, md[3] the first output,
     * md[4] / md[5] the others) and dropout */
    GP_ELTB,
    GP_SMB,
    GP_POOLB,
    GP_RESB,
    GP_CONVBD,
    GP_CONVBW,
    GP_DECONVBD,
    GP_DECONVBW,
    GP_BNT,
    GP_LNT,
    GP_GNT,
    GP_BNB,
    GP_LNB,
    GP_GNB,
    GP_PRELUB,
    GP_DROP,
    GP_GENIDX,
    GP_DYNQ,
    GP_LINATT
};

typedef struct {
    int kind;
    aiednn_memory_desc_t md[6]; /* src, weights / src1 / k, bias / v, dst, mask */
    aiednn_memory_desc_t *cat;  /* concat inputs (owned) */
    int ncat;
    int has_bias, has_attr, has_perm;
    aiednn_attr_t attr;
    float *scales; /* attr.scales (owned) */
    aiednn_conv_params_t conv;
    aiednn_pool_params_t pool;
    aiednn_alg_kind_t alg;
    float alpha, beta, eps, scale, sdpa_scale;
    unsigned flags;
    int wt, axis, causal;
    int64_t groups;
    float pnorm;
    int64_t starts[AIEDNN_MAX_NDIMS]; /* slice */
    int sdiv;                         /* sdpa matched from a divide: the bound scalar is inverted per execute */
    int perm[AIEDNN_MAX_NDIMS];
    int32_t zp;
    int nb; /* bindings: aiednn argument <- logical tensor id */
    int arg[GR_MAX_BIND];
    uint64_t tid[GR_MAX_BIND];
    uint64_t dst_id;
    int const_w; /* the WEIGHTS tensor is a constant */
    /* bias prologue: the BIAS binding names an integer tensor behind a dequantize (ONNX QDQ's s32
     * bias); execute dequantizes it into the compiled partition's f32 buffer, once when constant */
    int bprep, bconst;
    aiednn_data_type_t bdt;
    int64_t bn, bnsc; /* elements, scales (1: per tensor) */
    float *bsc;       /* owned */
    int32_t *bzp;     /* owned */
    int ndq;
    int dq[GR_MAX_DQ]; /* dequantize ops read through (op indices) */
    int has_src2;      /* binary select: the condition */
    int nxo;           /* outputs after dst_id (training ops) */
    uint64_t xo[3];
    aiednn_conv_nd_params_t cnd; /* convolution / pooling backward */
    aiednn_pool_nd_params_t pnd;
    uint64_t seed; /* dropout */
    int barg;      /* the prologue's argument (0: AIEDNN_ARG_BIAS; a folded norm: AIEDNN_ARG_SCALE) */
    int eps_rt;    /* eps is a graph constant read at execute (GR_ARG_EPS): f32, or codes with esc / ezp */
    aiednn_data_type_t edt;
    float esc;
    int32_t ezp;
    aiednn_memory_desc_t lmd[3]; /* linear attention: gate, beta, state_in (ndims 0: absent) */
} gr_plan;

struct aiednn_partition {
    struct aiednn_graph *g;
    int supported;
    int nops;
    int *ops; /* op indices, topological */
    int key;  /* topo index of its last op */
    gr_plan plan;
    char info[256];
};

struct aiednn_graph {
    gr_op *ops;
    int nops, cap_ops;
    gr_tensor *t;
    int nt, cap_t;
    int *order;                   /* topological */
    int finalized, fuse, layouts; /* layouts: propagate nhwc on internal edges (default on) */
    struct aiednn_partition *parts;
    int nparts;
};

struct aiednn_compiled_partition {
    gr_plan plan;
    aiednn_primitive_t prim;
    float *bias;           /* bias prologue output */
    float sval;            /* sdpa: 1 / the bound divisor */
    const void *bias_from; /* the source it was computed from (constant: computed once) */
    char info[160];
};

/* ── small helpers ─────────────────────────────────────────────────────── */

static int64_t gr_numel(const aiednn_memory_desc_t *md) {
    int64_t n = 1;
    for (int i = 0; i < md->ndims; i++)
        n *= md->dims[i];
    return n;
}

static size_t gr_bytes(const aiednn_memory_desc_t *md) {
    const int64_t n = gr_numel(md);
    switch (md->data_type) {
    case aiednn_s8:
    case aiednn_u8:
    case aiednn_f8_e5m2:
    case aiednn_f8_e4m3:
    case aiednn_e8m0:
        return (size_t)n;
    case aiednn_bf16:
    case aiednn_f16:
        return (size_t)n * 2;
    case aiednn_s4:
    case aiednn_u4:
    case aiednn_f4_e2m1:
        return (size_t)(n + 1) / 2;
    default:
        return (size_t)n * 4;
    }
}

static int gr_is_8bit(aiednn_data_type_t dt) { return dt == aiednn_s8 || dt == aiednn_u8; }

/* md with leading 1s up to rank r */
static aiednn_memory_desc_t gr_expand(const aiednn_memory_desc_t *md, int r) {
    aiednn_memory_desc_t o = *md;
    const int k = r - md->ndims;
    if (k <= 0)
        return o;
    for (int i = r - 1; i >= 0; i--)
        o.dims[i] = i >= k ? md->dims[i - k] : 1;
    o.ndims = r;
    return o;
}

static int gr_bcast_to(const aiednn_memory_desc_t *a, const aiednn_memory_desc_t *dst) {
    if (a->ndims != dst->ndims)
        return 0;
    for (int i = 0; i < a->ndims; i++)
        if (a->dims[i] != 1 && a->dims[i] != dst->dims[i])
            return 0;
    return 1;
}

static int gr_find_tensor(const struct aiednn_graph *g, uint64_t id) {
    for (int i = 0; i < g->nt; i++)
        if (g->t[i].lt.id == id)
            return i;
    return -1;
}

static const char *gr_kind_name(aiednn_graph_op_kind_t k) {
    static const char *n[] = {"?",
                              "conv",
                              "matmul",
                              "inner_product",
                              "bias_add",
                              "add",
                              "multiply",
                              "binary",
                              "relu",
                              "clamp",
                              "gelu",
                              "sigmoid",
                              "tanh",
                              "hardswish",
                              "eltwise",
                              "max_pool",
                              "avg_pool",
                              "softmax",
                              "layer_norm",
                              "batch_norm",
                              "concat",
                              "reshape",
                              "transpose",
                              "quantize",
                              "dequantize",
                              "sdpa",
                              "wildcard",
                              "deconv",
                              "resampling",
                              "reduction",
                              "group_norm",
                              "prelu",
                              "slice",
                              "type_cast",
                              "reorder",
                              "select",
                              "squared_difference",
                              "dropout",
                              "rms_norm",
                              "eltwise_bwd",
                              "softmax_bwd",
                              "pool_bwd",
                              "resampling_bwd",
                              "bias_add_bwd",
                              "conv_bwd_data",
                              "conv_bwd_weights",
                              "deconv_bwd_data",
                              "deconv_bwd_weights",
                              "batch_norm_training",
                              "batch_norm_bwd",
                              "layer_norm_bwd",
                              "group_norm_bwd",
                              "prelu_bwd",
                              "gen_index",
                              "dynamic_quantize",
                              "dynamic_dequantize",
                              "linear_attention"};
    static_assert(sizeof n / sizeof n[0] == (size_t)aiednn_gop_last_, "a name per graph op kind");
    return (int)k > 0 && (int)k < (int)aiednn_gop_last_ ? n[k] : "?";
}

/* eltwise algorithm of an op, or 0 */
static int gr_eltwise_of(const gr_op *o, aiednn_alg_kind_t *alg, float *a, float *b) {
    *a = o->attr.alpha, *b = o->attr.beta;
    switch (o->kind) {
    case aiednn_gop_relu:
        *alg = aiednn_eltwise_relu, *b = 0;
        return 1;
    case aiednn_gop_clamp:
        *alg = aiednn_eltwise_clip;
        return 1;
    case aiednn_gop_gelu:
        *alg = o->attr.alg == aiednn_eltwise_gelu_tanh ? aiednn_eltwise_gelu_tanh : aiednn_eltwise_gelu_erf;
        *a = *b = 0;
        return 1;
    case aiednn_gop_sigmoid:
        *alg = aiednn_eltwise_logistic, *a = *b = 0;
        return 1;
    case aiednn_gop_tanh:
        *alg = aiednn_eltwise_tanh, *a = *b = 0;
        return 1;
    case aiednn_gop_hardswish:
        *alg = aiednn_eltwise_hardswish, *a = 1.f / 6.f, *b = 0.5f;
        return 1;
    case aiednn_gop_eltwise:
        *alg = o->attr.alg;
        return (int)o->attr.alg >= (int)aiednn_eltwise_relu && (int)o->attr.alg < (int)aiednn_binary_add;
    default:
        return 0;
    }
}

/* binary algorithm of an op, or 0 */
static int gr_binary_of(const gr_op *o, aiednn_alg_kind_t *alg) {
    switch (o->kind) {
    case aiednn_gop_add:
        *alg = aiednn_binary_add;
        return 1;
    case aiednn_gop_multiply:
        *alg = aiednn_binary_mul;
        return 1;
    case aiednn_gop_binary:
        *alg = o->attr.alg;
        return (int)o->attr.alg >= (int)aiednn_binary_add && (int)o->attr.alg < (int)aiednn_reduction_sum &&
               o->attr.alg != aiednn_binary_select;
    default:
        return 0;
    }
}

/* a dequantize feeding tensor t that a consumer may read through */
typedef struct {
    int op, qt; /* the dequantize op, its 8-bit input tensor */
    float s;
    int32_t zp;
    int64_t n; /* 0: per tensor */
    const float *sc;
    const int32_t *zps;
} gr_q;

/* want_axis < 0: per tensor only; else per tensor or per channel along want_axis with nch channels */
static int gr_dq(const struct aiednn_graph *g, int t, int want_axis, int64_t nch, gr_q *q) {
    const int p = g->t[t].producer;
    if (p < 0 || g->ops[p].kind != aiednn_gop_dequantize || g->ops[p].nin != 1)
        return 0;
    const gr_op *o = &g->ops[p];
    const gr_tensor *in = &g->t[o->in[0]];
    if (!gr_is_8bit(in->lt.md.data_type))
        return 0;
    /* the 8-bit tensor must exist outside its producer's partition: a graph input, or that partition's
     * output -- not an intermediate a (level-2) chain kept inside */
    const int pp = in->producer, pt = pp >= 0 ? g->ops[pp].part : -1;
    if (pt >= 0) {
        const struct aiednn_partition *P = &g->parts[pt];
        const uint64_t out = P->supported ? P->plan.dst_id : g->t[g->ops[P->ops[P->nops - 1]].out].lt.id;
        if (out != in->lt.id)
            return 0;
    }
    memset(q, 0, sizeof *q);
    q->op = p, q->qt = o->in[0], q->s = o->attr.scale, q->zp = o->attr.zero_point;
    if (o->attr.n_scales > 0) {
        int ax = o->attr.axis < 0 ? o->attr.axis + in->lt.md.ndims : o->attr.axis;
        if (want_axis < 0 || ax != want_axis || o->attr.n_scales != nch)
            return 0;
        q->n = o->attr.n_scales, q->sc = o->attr.scales, q->zps = o->attr.zero_points;
    }
    return 1;
}

/* bias tensor t: bound directly, or through the dequantize producing it (integer, per tensor or per
 * channel along its only axis) with the prologue; md[2] stays the f32 bias shape */
static int gr_bind_bias(gr_plan *p, const struct aiednn_graph *g, int t);

static void gr_bind(gr_plan *p, int arg, const struct aiednn_graph *g, int t) {
    if (p->nb < GR_MAX_BIND)
        p->arg[p->nb] = arg, p->tid[p->nb++] = g->t[t].lt.id;
}

static void gr_add_dq(gr_plan *p, int op);

static int gr_bind_prep(gr_plan *p, const struct aiednn_graph *g, int t, int arg) {
    const int pr = g->t[t].producer;
    const gr_op *o = pr >= 0 ? &g->ops[pr] : nullptr;
    if (o && o->kind == aiednn_gop_dequantize && o->nin == 1 && g->fuse && !g->t[t].marked) {
        const aiednn_memory_desc_t *qm = &g->t[o->in[0]].lt.md;
        const int64_t n = gr_numel(qm);
        const int64_t ns = o->attr.n_scales > 0 ? o->attr.n_scales : 1;
        if ((qm->data_type == aiednn_s32 || gr_is_8bit(qm->data_type)) && (ns == 1 || ns == n)) {
            p->bsc = (float *)malloc((size_t)ns * sizeof(float));
            p->bzp = (int32_t *)malloc((size_t)ns * sizeof(int32_t));
            if (!p->bsc || !p->bzp)
                return 0;
            for (int64_t c = 0; c < ns; c++) {
                p->bsc[c] = o->attr.n_scales > 0 ? o->attr.scales[c] : o->attr.scale;
                p->bzp[c] =
                    o->attr.n_scales > 0 ? (o->attr.zero_points ? o->attr.zero_points[c] : 0) : o->attr.zero_point;
            }
            p->bprep = 1, p->bdt = qm->data_type, p->bn = n, p->bnsc = ns;
            p->bconst = g->t[o->in[0]].lt.property == aiednn_lt_constant;
            p->barg = arg == AIEDNN_ARG_BIAS ? 0 : arg;
            gr_bind(p, arg, g, o->in[0]);
            gr_add_dq(p, pr);
            return 1;
        }
    }
    gr_bind(p, arg, g, t);
    return 1;
}

static int gr_bind_bias(gr_plan *p, const struct aiednn_graph *g, int t) {
    return gr_bind_prep(p, g, t, AIEDNN_ARG_BIAS);
}

static void gr_add_dq(gr_plan *p, int op) {
    for (int i = 0; i < p->ndq; i++)
        if (p->dq[i] == op)
            return;
    if (p->ndq < GR_MAX_DQ)
        p->dq[p->ndq++] = op;
}

/* append an eltwise post-op; a relu (alpha 0) or clip right after another one merges into a single clip
 * when the ranges overlap */
static int gr_push_eltwise(gr_plan *p, aiednn_alg_kind_t alg, float a, float b) {
    const int bounded = (alg == aiednn_eltwise_relu && a == 0.f) || alg == aiednn_eltwise_clip;
    const float lo = alg == aiednn_eltwise_clip ? a : 0.f, hi = alg == aiednn_eltwise_clip ? b : 3.0e38f;
    if (bounded && p->attr.n_post_ops > 0) {
        aiednn_post_op_t *q = &p->attr.post_ops[p->attr.n_post_ops - 1];
        const int qb = q->kind == aiednn_post_op_eltwise &&
                       ((q->alg == aiednn_eltwise_relu && q->alpha == 0.f) || q->alg == aiednn_eltwise_clip);
        if (qb) {
            const float ql = q->alg == aiednn_eltwise_clip ? q->alpha : 0.f,
                        qh = q->alg == aiednn_eltwise_clip ? q->beta : 3.0e38f;
            const float l = lo > ql ? lo : ql, h = hi < qh ? hi : qh;
            if (l <= h) {
                q->alg = aiednn_eltwise_clip, q->alpha = l, q->beta = h;
                p->has_attr = 1;
                return 1;
            }
        }
    }
    if (p->attr.n_post_ops >= AIEDNN_MAX_POST_OPS)
        return 0;
    aiednn_post_op_t *po = &p->attr.post_ops[p->attr.n_post_ops++];
    memset(po, 0, sizeof *po);
    po->kind = aiednn_post_op_eltwise, po->alg = alg, po->alpha = a, po->beta = b;
    p->has_attr = 1;
    return 1;
}

static void gr_plan_free(gr_plan *p) {
    free(p->scales);
    free(p->cat);
    free(p->bsc);
    free(p->bzp);
    p->scales = nullptr, p->cat = nullptr, p->bsc = nullptr, p->bzp = nullptr;
}

static int gr_plan_copy(gr_plan *dst, const gr_plan *src) {
    *dst = *src;
    dst->scales = nullptr, dst->cat = nullptr, dst->bsc = nullptr, dst->bzp = nullptr;
    if (src->bprep || src->kind == GP_QCAT) { /* (QCAT: per-input scales / zero points in bsc / bzp) */
        dst->bsc = (float *)malloc((size_t)src->bnsc * sizeof(float));
        dst->bzp = (int32_t *)malloc((size_t)src->bnsc * sizeof(int32_t));
        if (!dst->bsc || !dst->bzp)
            return 0;
        memcpy(dst->bsc, src->bsc, (size_t)src->bnsc * sizeof(float));
        memcpy(dst->bzp, src->bzp, (size_t)src->bnsc * sizeof(int32_t));
    }
    if (src->scales) {
        dst->scales = (float *)malloc((size_t)src->attr.n_scales * sizeof(float));
        if (!dst->scales)
            return 0;
        memcpy(dst->scales, src->scales, (size_t)src->attr.n_scales * sizeof(float));
        dst->attr.scales = dst->scales;
    }
    if (src->cat) {
        dst->cat = (aiednn_memory_desc_t *)malloc((size_t)src->ncat * sizeof(aiednn_memory_desc_t));
        if (!dst->cat)
            return 0;
        memcpy(dst->cat, src->cat, (size_t)src->ncat * sizeof(aiednn_memory_desc_t));
    }
    return 1;
}

/* ── plan -> primitive (public API only) ──────────────────────────────── */

/* the training kinds and dropout (gr_create's tail) */
static aiednn_status_t gr_create_train(const gr_plan *p, aiednn_primitive_t *pr) {
    const aiednn_memory_desc_t *m = p->md;
    switch (p->kind) {
    case GP_ELTB:
        return aiednn_eltwise_backward_create(pr, p->alg, p->alpha, p->beta, &m[3], &m[1], &m[0]);
    case GP_SMB:
        return aiednn_softmax_backward_create(pr, p->alg ? p->alg : aiednn_softmax_accurate, p->axis, &m[3], &m[1],
                                              &m[0]);
    case GP_POOLB:
        return aiednn_pooling_backward_create(pr, p->alg, &m[3], &m[1], &m[0], &p->pnd);
    case GP_RESB:
        return aiednn_resampling_backward_create(pr, p->alg, &m[3], &m[1]);
    case GP_CONVBD:
        return aiednn_convolution_backward_data_create(pr, &m[3], &m[1], &m[0], &p->cnd);
    case GP_DECONVBD:
        return aiednn_deconvolution_backward_data_create(pr, &m[3], &m[1], &m[0], &p->cnd);
    case GP_CONVBW:
        return aiednn_convolution_backward_weights_create(pr, &m[0], &m[3], p->has_bias ? &m[4] : nullptr, &m[1],
                                                          &p->cnd);
    case GP_DECONVBW:
        return aiednn_deconvolution_backward_weights_create(pr, &m[0], &m[3], p->has_bias ? &m[4] : nullptr, &m[1],
                                                            &p->cnd);
    case GP_BNT:
        return aiednn_batch_normalization_forward_training_create(pr, &m[0], &m[3], p->eps, p->flags);
    case GP_LNT:
        return aiednn_layer_normalization_forward_training_create(pr, &m[0], &m[3], p->eps, p->flags);
    case GP_GNT:
        return aiednn_group_normalization_forward_training_create(pr, p->groups, &m[0], &m[3], p->eps, p->flags);
    case GP_BNB:
        return aiednn_batch_normalization_backward_create(pr, &m[3], &m[1], &m[0], p->eps, p->flags);
    case GP_LNB:
        return aiednn_layer_normalization_backward_create(pr, &m[3], &m[1], &m[0], p->eps, p->flags);
    case GP_GNB:
        return aiednn_group_normalization_backward_create(pr, p->groups, &m[3], &m[1], &m[0], p->eps, p->flags);
    case GP_PRELUB:
        return aiednn_prelu_backward_create(pr, &m[3], &m[4], &m[1], &m[0], &m[2]);
    case GP_DROP:
        return aiednn_dropout_forward_create(pr, &m[0], &m[3], p->alpha, p->seed);
    case GP_GENIDX:
        return aiednn_gen_index_create(pr, &m[3], p->axis);
    case GP_LINATT:
        return aiednn_linear_attention_create(
            pr, &m[0], &m[1], &m[2], &p->lmd[0], p->lmd[1].ndims ? &p->lmd[1] : nullptr,
            p->lmd[2].ndims ? &p->lmd[2] : nullptr, &m[3], p->nxo ? &m[4] : nullptr, p->sdpa_scale, p->flags);
    case GP_DYNQ:
        return aiednn_dynamic_quantize_create(pr, &m[0], &m[3], &m[1], p->has_bias ? &m[2] : nullptr, p->axis);
    default:
        return aiednn_unimplemented;
    }
}

static aiednn_status_t gr_create(const gr_plan *p, aiednn_primitive_t *pr) {
    const aiednn_attr_t *a = p->has_attr ? &p->attr : nullptr;
    const aiednn_memory_desc_t *b = p->has_bias ? &p->md[2] : nullptr;
    *pr = nullptr;
    switch (p->kind) {
    case GP_CONV:
        return aiednn_convolution_forward_create_attr(pr, &p->md[0], &p->md[1], b, &p->md[3], &p->conv, a);
    case GP_DECONV:
        return aiednn_deconvolution_forward_create_attr(pr, &p->md[0], &p->md[1], b, &p->md[3], &p->conv, a);
    case GP_RESAMPLE:
        return a ? aiednn_resampling_forward_create_attr(pr, p->alg, &p->md[0], &p->md[3], a)
                 : aiednn_resampling_forward_create(pr, p->alg, &p->md[0], &p->md[3]);
    case GP_REDUCE:
        return a ? aiednn_reduction_create_attr(pr, p->alg, p->pnorm, p->eps, &p->md[0], &p->md[3], a)
                 : aiednn_reduction_create(pr, p->alg, p->pnorm, p->eps, &p->md[0], &p->md[3]);
    case GP_GNORM:
        return a ? aiednn_group_normalization_forward_inference_create_attr(pr, p->groups, &p->md[0], &p->md[3], p->eps,
                                                                            p->flags, a)
                 : aiednn_group_normalization_forward_inference_create(pr, p->groups, &p->md[0], &p->md[3], p->eps,
                                                                       p->flags);
    case GP_PRELU:
        return aiednn_prelu_forward_create(pr, &p->md[0], &p->md[1], &p->md[3]);
    case GP_QBIN:
        return aiednn_binary_create_attr(pr, p->alg, &p->md[0], &p->md[1], &p->md[3], &p->attr);
    case GP_QELT:
        return aiednn_eltwise_forward_create_attr(pr, p->alg, p->alpha, p->beta, &p->md[0], &p->md[3], &p->attr);
    case GP_MATMUL:
        return aiednn_matmul_forward_create_attr(pr, &p->md[0], &p->md[1], b, &p->md[3], p->wt, a);
    case GP_IP:
        return aiednn_inner_product_forward_create_attr(pr, &p->md[0], &p->md[1], b, &p->md[3], a);
    case GP_ELT:
        return aiednn_eltwise_forward_create(pr, p->alg, p->alpha, p->beta, &p->md[0], &p->md[3]);
    case GP_BIN:
        return aiednn_binary_create(pr, p->alg, &p->md[0], &p->md[1], p->has_src2 ? &p->md[2] : nullptr, &p->md[3]);
    case GP_POOL:
        return a ? aiednn_pooling_forward_create_attr(pr, p->alg, &p->md[0], &p->md[3], &p->pool, a)
                 : aiednn_pooling_forward_create(pr, p->alg, &p->md[0], &p->md[3], &p->pool);
    case GP_SOFTMAX:
        return a ? aiednn_softmax_forward_create_attr(pr, p->alg ? p->alg : aiednn_softmax_accurate, p->axis, &p->md[0],
                                                      &p->md[3], a)
                 : aiednn_softmax_forward_create(pr, p->alg ? p->alg : aiednn_softmax_accurate, p->axis, &p->md[0],
                                                 &p->md[3]);
    case GP_LNORM:
        return a ? aiednn_layer_normalization_forward_inference_create_attr(pr, &p->md[0], &p->md[3], p->eps, p->flags,
                                                                            a)
                 : aiednn_layer_normalization_forward_inference_create(pr, &p->md[0], &p->md[3], p->eps, p->flags);
    case GP_BNORM:
        return a ? aiednn_batch_normalization_forward_inference_create_attr(pr, &p->md[0], &p->md[3], p->eps, p->flags,
                                                                            a)
                 : aiednn_batch_normalization_forward_inference_create(pr, &p->md[0], &p->md[3], p->eps, p->flags);
    case GP_CONCAT:
        return aiednn_concat_create(pr, p->ncat, p->cat, p->axis, &p->md[3]);
    case GP_REORDER:
        return a ? aiednn_reorder_create_attr(pr, &p->md[0], &p->md[3], p->has_perm ? p->perm : nullptr, p->scale,
                                              p->zp, a)
                 : aiednn_reorder_create(pr, &p->md[0], &p->md[3], p->has_perm ? p->perm : nullptr, p->scale, p->zp);
    case GP_SDPA:
        return a ? aiednn_sdpa_forward_create_attr(pr, &p->md[0], &p->md[1], &p->md[2],
                                                   p->has_bias ? &p->md[4] : nullptr, &p->md[3], p->sdpa_scale,
                                                   p->causal, a)
                 : aiednn_sdpa_forward_create(pr, &p->md[0], &p->md[1], &p->md[2], p->has_bias ? &p->md[4] : nullptr,
                                              &p->md[3], p->sdpa_scale, p->causal);
    case GP_COPY:
        return p->md[0].data_type == p->md[3].data_type && gr_numel(&p->md[0]) == gr_numel(&p->md[3])
                   ? aiednn_success
                   : aiednn_invalid_arguments;
    case GP_SLICE:
        return a ? aiednn_slice_create_attr(pr, &p->md[0], &p->md[3], p->starts, nullptr, a)
                 : aiednn_slice_create(pr, &p->md[0], &p->md[3], p->starts, nullptr);
    case GP_ROPE:
        return a ? aiednn_rope_create_attr(pr, &p->md[0], &p->md[1], &p->md[2], &p->md[3], 0, a)
                 : aiednn_rope_create(pr, &p->md[0], &p->md[1], &p->md[2], &p->md[3], 0);
    case GP_GMLP:
        return aiednn_gated_mlp_create(pr, &p->md[0], &p->md[1], &p->md[2], &p->md[4], &p->md[3], p->alg, p->alpha,
                                       p->beta, p->wt ? AIEDNN_GMLP_WEIGHTS_TRANSPOSED : 0u);
    case GP_QCAT:
        return aiednn_concat_create_q(pr, p->ncat, p->cat, p->bsc, p->bzp, p->axis, &p->md[3], p->attr.dst_scale,
                                      p->attr.dst_zero_point);
    default:
        return gr_create_train(p, pr);
    }
}

/* ── anchors: conv / matmul / inner product + epilogue chain ──────────── */

/* the anchor's src / weights / bias, 8-bit through dequantize when use_int; 0 if not possible */
static int gr_anchor_inputs(const struct aiednn_graph *g, const gr_op *o, int use_int, int chan, gr_plan *p) {
    const int x = o->in[0], w = o->in[1], r = g->t[o->out].lt.md.ndims;
    const int64_t nch = g->t[o->out].lt.md.dims[chan];
    p->kind = o->kind == aiednn_gop_convolution     ? GP_CONV
              : o->kind == aiednn_gop_deconvolution ? GP_DECONV
              : o->kind == aiednn_gop_matmul        ? GP_MATMUL
                                                    : GP_IP;
    p->conv = o->attr.conv;
    p->wt = o->attr.weights_transposed;
    p->md[3] = g->t[o->out].lt.md;
    p->dst_id = g->t[o->out].lt.id;
    int xt = x, wtn = w;
    if (use_int) {
        const int wr = g->t[w].lt.md.ndims;
        const int wax = p->kind == GP_MATMUL ? (p->wt ? wr - 2 : wr - 1) : 0;
        gr_q qx, qw;
        if (!gr_dq(g, x, -1, 0, &qx) || !gr_dq(g, w, wax, nch, &qw) || g->t[qw.qt].lt.md.data_type != aiednn_s8)
            return 0;
        int32_t wzp = qw.zp;
        if (qw.n)
            for (int64_t c = 0; c < qw.n; c++)
                if ((qw.zps ? qw.zps[c] : 0) != 0)
                    return 0; /* per-channel weight zero points: no aiednn form */
        if (qw.n)
            wzp = 0;
        const int64_t ns = qw.n ? qw.n : 1;
        p->scales = (float *)malloc((size_t)ns * sizeof(float));
        if (!p->scales)
            return 0;
        for (int64_t c = 0; c < ns; c++)
            p->scales[c] = qx.s * (qw.n ? qw.sc[c] : qw.s);
        p->attr.n_scales = ns, p->attr.scales = p->scales;
        p->attr.src_zero_point = qx.zp, p->attr.wei_zero_point = wzp;
        p->has_attr = 1;
        xt = qx.qt, wtn = qw.qt;
        gr_add_dq(p, qx.op), gr_add_dq(p, qw.op);
    }
    p->md[0] = g->t[xt].lt.md, p->md[1] = g->t[wtn].lt.md;
    if (p->kind == GP_MATMUL && p->md[1].ndims < p->md[0].ndims) /* nn.Linear on tokens: weights broadcast */
        p->md[1] = gr_expand(&p->md[1], p->md[0].ndims);
    p->const_w = g->t[wtn].lt.property == aiednn_lt_constant;
    gr_bind(p, AIEDNN_ARG_SRC, g, xt), gr_bind(p, AIEDNN_ARG_WEIGHTS, g, wtn);
    if (o->nin > 2) {
        const aiednn_memory_desc_t *bm = &g->t[o->in[2]].lt.md;
        p->md[2] = p->kind == GP_MATMUL ? gr_expand(bm, r) : *bm;
        p->has_bias = 1;
        if (!gr_bind_bias(p, g, o->in[2]))
            return 0;
    }
    return 1;
}

/* one chain step after the anchor: 1 absorbed, 2 absorbed and the chain ends, 0 stop */
static int gr_chain_step(const struct aiednn_graph *g, int cur, int chan, int use_int, gr_plan *p, int u) {
    const gr_op *o = &g->ops[u];
    const aiednn_memory_desc_t *dmd = &g->t[cur].lt.md;
    aiednn_alg_kind_t alg;
    float a, b;
    if (o->kind == aiednn_gop_bias_add && o->nin == 2 && o->in[0] == cur && !p->has_bias && !p->attr.n_post_ops) {
        const aiednn_memory_desc_t *bm = &g->t[o->in[1]].lt.md;
        const int ax = o->attr.axis < 0 ? chan : o->attr.axis; /* default: the anchor's channel */
        if (ax != chan || bm->ndims != 1 || bm->dims[0] != dmd->dims[chan])
            return 0;
        p->md[2] = p->kind == GP_MATMUL ? gr_expand(bm, dmd->ndims) : *bm;
        p->has_bias = 1;
        return gr_bind_bias(p, g, o->in[1]);
    }
    if (gr_eltwise_of(o, &alg, &a, &b))
        return gr_push_eltwise(p, alg, a, b);
    int has_rb = 0; /* after a requant_* post-op only eltwise ones may follow */
    for (int k = 0; k < p->attr.n_post_ops; k++)
        has_rb |= p->attr.post_ops[k].kind == aiednn_post_op_requant_binary ||
                  p->attr.post_ops[k].kind == aiednn_post_op_requant_eltwise;
    if (!has_rb && gr_binary_of(o, &alg) && o->nin == 2 && p->attr.n_post_ops < AIEDNN_MAX_POST_OPS) {
        const int commut = alg == aiednn_binary_add || alg == aiednn_binary_mul || alg == aiednn_binary_max ||
                           alg == aiednn_binary_min;
        if (o->in[0] != cur && !(commut && o->in[1] == cur))
            return 0;
        int other = o->in[0] == cur ? o->in[1] : o->in[0];
        if (other == cur)
            return 0;
        aiednn_post_op_t *po = &p->attr.post_ops[p->attr.n_post_ops];
        memset(po, 0, sizeof *po);
        po->kind = aiednn_post_op_binary, po->alg = alg;
        gr_q q;
        if (use_int && gr_dq(g, other, -1, 0, &q)) { /* an 8-bit residual: real = (code - zp) * scale */
            gr_add_dq(p, q.op);
            other = q.qt, po->alpha = q.s, po->beta = (float)q.zp;
        }
        po->src1 = gr_expand(&g->t[other].lt.md, dmd->ndims);
        if (!gr_bcast_to(&po->src1, dmd))
            return 0;
        gr_bind(p, AIEDNN_ARG_ATTR_POST_OP_SRC(p->attr.n_post_ops), g, other);
        p->attr.n_post_ops++;
        p->has_attr = 1;
        return 1;
    }
    if (o->kind == aiednn_gop_reshape && o->nin == 1 && o->in[0] == cur) { /* an identity view (eval-mode
                                                                            * StochasticDepth): no-op */
        const aiednn_memory_desc_t *ym = &g->t[o->out].lt.md;
        return ym->ndims == dmd->ndims && ym->data_type == dmd->data_type &&
               !memcmp(ym->dims, dmd->dims, (size_t)ym->ndims * sizeof(int64_t));
    }
    if (o->kind == aiednn_gop_quantize && use_int && o->attr.n_scales == 0 &&
        gr_is_8bit(g->t[o->out].lt.md.data_type)) {
        p->attr.dst_scale = o->attr.scale, p->attr.dst_zero_point = o->attr.zero_point;
        p->has_attr = 1;
        return 2;
    }
    return 0;
}

/* the only consumer of tensor t, still unpartitioned; -1 if t is marked, shared or consumed by nobody */
static int gr_sole_user(const struct aiednn_graph *g, int t) {
    if (g->t[t].marked || g->t[t].nuse != 1)
        return -1;
    for (int k = 0; k < g->nops; k++)
        for (int i = 0; i < g->ops[k].nin; i++)
            if (g->ops[k].in[i] == t)
                return g->ops[k].part < 0 ? k : -1;
    return -1;
}

/* a dense reshape: the same bytes under other dims */
static int gr_is_view(const struct aiednn_graph *g, const gr_op *o) {
    if (o->kind != aiednn_gop_reshape || o->nin != 1)
        return 0;
    const aiednn_memory_desc_t *x = &g->t[o->in[0]].lt.md, *y = &g->t[o->out].lt.md;
    int64_t a = 1, b = 1;
    for (int i = 0; i < x->ndims; i++)
        a *= x->dims[i];
    for (int i = 0; i < y->ndims; i++)
        b *= y->dims[i];
    return a == b && x->data_type == y->data_type && x->format == aiednn_format_dense &&
           y->format == aiednn_format_dense;
}

/* past a per-tensor quantize's output `cur`: dequantize -> dense views -> quantize with the same parameters,
 * any number of times; the codes are unchanged, so the partition writes the last quantize's tensor instead
 * (exact). Appends the ops to ops[]; returns the new tensor. */
static int gr_qcopy_tail(const struct aiednn_graph *g, int cur, int *ops, int *n, int cap) {
    for (;;) {
        const int q0 = g->t[cur].producer;
        if (q0 < 0 || g->ops[q0].kind != aiednn_gop_quantize || g->ops[q0].attr.n_scales)
            return cur;
        const float s = g->ops[q0].attr.scale;
        const int zp = g->ops[q0].attr.zero_point;
        const int d = gr_sole_user(g, cur);
        if (d < 0 || g->ops[d].kind != aiednn_gop_dequantize || g->ops[d].nin != 1 || g->ops[d].attr.n_scales ||
            g->ops[d].attr.scale != s || g->ops[d].attr.zero_point != zp)
            return cur;
        int add[GR_MAX_CHAIN], na = 0, t = g->ops[d].out, u;
        add[na++] = d;
        while ((u = gr_sole_user(g, t)) >= 0 && gr_is_view(g, &g->ops[u]) && na < GR_MAX_CHAIN - 1)
            add[na++] = u, t = g->ops[u].out;
        if (u < 0 || g->ops[u].kind != aiednn_gop_quantize || g->ops[u].nin != 1 || g->ops[u].attr.n_scales ||
            g->ops[u].attr.scale != s || g->ops[u].attr.zero_point != zp ||
            g->t[g->ops[u].out].lt.md.data_type != g->t[cur].lt.md.data_type || *n + na + 1 > cap)
            return cur;
        add[na++] = u;
        for (int i = 0; i < na; i++)
            ops[(*n)++] = add[i];
        cur = g->ops[u].out;
    }
}

/* fusion level 2 only: a per-tensor quantize whose only consumer is a dequantize with the same scale /
 * zero point -- skipping both drops that rounding (not exact). Returns the dequantize, or -1. */
static int gr_qdq_pair(const struct aiednn_graph *g, int u) {
    const gr_op *q = &g->ops[u];
    if (g->fuse < 2 || q->kind != aiednn_gop_quantize || q->nin != 1 || q->attr.n_scales)
        return -1;
    const int d = gr_sole_user(g, q->out);
    if (d < 0)
        return -1;
    const gr_op *dq = &g->ops[d];
    return dq->kind == aiednn_gop_dequantize && !dq->attr.n_scales && dq->attr.scale == q->attr.scale &&
                   dq->attr.zero_point == q->attr.zero_point
               ? d
               : -1;
}

/* level 1 (exact): the anchor's value -> quantize -> dequantize (same parameters) -> add | sub | mul | max |
 * min with another tensor = one requant_binary post-op. Returns the binary op (and the dequantize in *dq),
 * -1 if the pattern is not there. */
static int gr_requant_binary(const struct aiednn_graph *g, int cur, int u, gr_plan *p, int *dq, int *skip, int *ns,
                             int cap) {
    const gr_op *q = &g->ops[u];
    if (!g->fuse || q->kind != aiednn_gop_quantize || q->nin != 1 || q->attr.n_scales ||
        !gr_is_8bit(g->t[q->out].lt.md.data_type) || p->attr.n_post_ops >= AIEDNN_MAX_POST_OPS)
        return -1;
    for (int k = 0; k < p->attr.n_post_ops; k++)
        if (p->attr.post_ops[k].kind != aiednn_post_op_eltwise)
            return -1; /* one per primitive; earlier binary post-ops would reorder with it */
    /* the add may sit behind any number of same-parameter quantize -> dequantize pairs (in the views' dims) */
    *ns = 0;
    const int qt = gr_qcopy_tail(g, q->out, skip, ns, cap);
    const int d = gr_sole_user(g, qt);
    if (d < 0 || g->ops[d].kind != aiednn_gop_dequantize || g->ops[d].attr.n_scales ||
        g->ops[d].attr.scale != q->attr.scale || g->ops[d].attr.zero_point != q->attr.zero_point)
        return -1;
    const int b = gr_sole_user(g, g->ops[d].out);
    aiednn_alg_kind_t alg;
    if (b < 0 || g->ops[b].nin != 2 || !gr_binary_of(&g->ops[b], &alg) ||
        !(alg == aiednn_binary_add || alg == aiednn_binary_sub || alg == aiednn_binary_mul ||
          alg == aiednn_binary_max || alg == aiednn_binary_min))
        return -1;
    const int t = g->ops[d].out, first = g->ops[b].in[0] == t;
    if (!first && alg == aiednn_binary_sub)
        return -1;
    int other = first ? g->ops[b].in[1] : g->ops[b].in[0];
    if (other == t)
        return -1;
    const aiednn_memory_desc_t *dmd = &g->t[cur].lt.md, *vmd = &g->t[t].lt.md;
    const int viewed = vmd->ndims != dmd->ndims || memcmp(vmd->dims, dmd->dims, (size_t)dmd->ndims * sizeof(int64_t));
    aiednn_post_op_t po;
    memset(&po, 0, sizeof po);
    po.kind = aiednn_post_op_requant_binary, po.alg = alg;
    po.q_scale = q->attr.scale, po.q_zero_point = q->attr.zero_point;
    po.q_data_type = g->t[q->out].lt.md.data_type;
    gr_q oq;
    const int odq = gr_dq(g, other, -1, 0, &oq);
    if (odq)
        other = oq.qt, po.alpha = oq.s, po.beta = (float)oq.zp;
    if (viewed) { /* the add runs in the views' dims: only a full-size operand maps element for element */
        const aiednn_memory_desc_t *om = &g->t[other].lt.md, *bm = &g->t[g->ops[b].out].lt.md;
        if (om->ndims != vmd->ndims || memcmp(om->dims, vmd->dims, (size_t)om->ndims * sizeof(int64_t)) ||
            gr_numel(bm) != gr_numel(dmd))
            return -1;
        po.src1 = *dmd, po.src1.data_type = om->data_type;
    } else {
        po.src1 = gr_expand(&g->t[other].lt.md, dmd->ndims);
        if (!gr_bcast_to(&po.src1, dmd) || g->t[g->ops[b].out].lt.md.ndims != dmd->ndims)
            return -1;
    }
    if (odq)
        gr_add_dq(p, oq.op);
    gr_bind(p, AIEDNN_ARG_ATTR_POST_OP_SRC(p->attr.n_post_ops), g, other);
    p->attr.post_ops[p->attr.n_post_ops++] = po;
    p->has_attr = 1;
    *dq = d;
    return b;
}

/* level 1 (exact): the anchor's value -> quantize -> dequantize (same parameters) -> a unary eltwise =
 * one requant_eltwise post-op. Returns the eltwise op (the dequantize in *dq), or -1. */
static int gr_requant_eltwise(const struct aiednn_graph *g, int u, gr_plan *p, int *dq) {
    const gr_op *q = &g->ops[u];
    if (!g->fuse || q->kind != aiednn_gop_quantize || q->nin != 1 || q->attr.n_scales ||
        !gr_is_8bit(g->t[q->out].lt.md.data_type) || p->attr.n_post_ops >= AIEDNN_MAX_POST_OPS)
        return -1;
    for (int k = 0; k < p->attr.n_post_ops; k++)
        if (p->attr.post_ops[k].kind != aiednn_post_op_eltwise)
            return -1; /* one requant_* per primitive */
    const int d = gr_sole_user(g, q->out);
    if (d < 0 || g->ops[d].kind != aiednn_gop_dequantize || g->ops[d].attr.n_scales ||
        g->ops[d].attr.scale != q->attr.scale || g->ops[d].attr.zero_point != q->attr.zero_point)
        return -1;
    const int e = gr_sole_user(g, g->ops[d].out);
    aiednn_alg_kind_t alg;
    float a, b;
    if (e < 0 || g->ops[e].nin != 1 || !gr_eltwise_of(&g->ops[e], &alg, &a, &b))
        return -1;
    aiednn_post_op_t *po = &p->attr.post_ops[p->attr.n_post_ops++];
    memset(po, 0, sizeof *po);
    po->kind = aiednn_post_op_requant_eltwise, po->alg = alg, po->alpha = a, po->beta = b;
    po->q_scale = q->attr.scale, po->q_zero_point = q->attr.zero_point;
    po->q_data_type = g->t[q->out].lt.md.data_type;
    p->has_attr = 1;
    *dq = d;
    return e;
}

/* plan for anchor ai with at most max_chain ops absorbed after it; members[0..*nm) = op indices */
static int gr_build_anchor(const struct aiednn_graph *g, int ai, int use_int, int max_chain, gr_plan *p, int *members,
                           int *nm) {
    const gr_op *o = &g->ops[ai];
    if (o->nin < 2 || o->nin > 3)
        return 0;
    const int r = g->t[o->out].lt.md.ndims;
    const int chan = o->kind == aiednn_gop_matmul ? r - 1 : 1;
    if (!gr_anchor_inputs(g, o, use_int, chan, p))
        return 0;
    members[0] = ai, *nm = 1;
    int cur = o->out;
    int pair_q = -1, pair_nm = 0, pair_npo = 0, pair_cur = -1; /* the last level-2 pair, until passed */
    aiednn_memory_desc_t vdims; /* the anchor's dims once a residual was read through views (ndims 0: not) */
    memset(&vdims, 0, sizeof vdims);
    aiednn_post_op_t pair_last; /* the post-op its clip may have merged into */
    memset(&pair_last, 0, sizeof pair_last);
    while (*nm - 1 < max_chain && g->fuse) {
        const gr_tensor *t = &g->t[cur];
        if (t->marked || t->nuse != 1)
            break;
        int u = -1;
        for (int k = 0; k < g->nops && u < 0; k++)
            for (int i = 0; i < g->ops[k].nin; i++)
                if (g->ops[k].in[i] == cur)
                    u = k;
        if (u < 0 || g->ops[u].part >= 0 || (!vdims.ndims && g->t[g->ops[u].out].lt.md.ndims != r))
            break;
        if (vdims.ndims) { /* past a viewed residual: a unary eltwise or the final quantize, nothing that broadcasts */
            aiednn_alg_kind_t va;
            float v0, v1;
            if (!(g->ops[u].nin == 1 &&
                  (gr_eltwise_of(&g->ops[u], &va, &v0, &v1) || g->ops[u].kind == aiednn_gop_quantize)))
                break;
        }
        {
            int rdq, skip[GR_MAX_CHAIN], ns = 0;
            const int nb = p->nb, ndq = p->ndq;
            const int rb =
                *nm + 3 <= GR_MAX_CHAIN ? gr_requant_binary(g, cur, u, p, &rdq, skip, &ns, GR_MAX_CHAIN - 3 - *nm) : -1;
            if (rb >= 0) {
                members[(*nm)++] = u;
                for (int i = 0; i < ns; i++)
                    members[(*nm)++] = skip[i];
                members[(*nm)++] = rdq, members[(*nm)++] = rb;
                if (g->t[g->ops[rb].out].lt.md.ndims != r ||
                    memcmp(g->t[g->ops[rb].out].lt.md.dims, g->t[cur].lt.md.dims, (size_t)r * sizeof(int64_t)))
                    vdims = g->t[cur].lt.md; /* from here on in the views' dims: only eltwise / quantize */
                cur = g->ops[rb].out;
                pair_q = -1;
                continue;
            }
            p->nb = nb, p->ndq = ndq;
            const int re = *nm + 3 <= GR_MAX_CHAIN ? gr_requant_eltwise(g, u, p, &rdq) : -1;
            if (re >= 0) {
                members[(*nm)++] = u, members[(*nm)++] = rdq, members[(*nm)++] = re;
                cur = g->ops[re].out;
                pair_q = -1;
                continue;
            }
        }
        int dqp = gr_qdq_pair(g, u);
        if (dqp >= 0) { /* only into a relu / clamp: the pair's clip merges with it and the anchor keeps its
                         * on-core integer requant (a clip before a residual add cannot: clip(v) + r is not
                         * clip(v + r), and that chain falls back to the host f32 epilogue, ~3x slower) */
            const int f = gr_sole_user(g, g->ops[dqp].out);
            aiednn_alg_kind_t fa;
            float x0, x1;
            if (f < 0 || g->ops[f].nin != 1 || !gr_eltwise_of(&g->ops[f], &fa, &x0, &x1) ||
                !((fa == aiednn_eltwise_relu && x0 == 0.f) || fa == aiednn_eltwise_clip))
                dqp = -1;
        }
        if (dqp >= 0 && *nm + 2 < GR_MAX_CHAIN && p->attr.n_post_ops < AIEDNN_MAX_POST_OPS) {
            /* level 2: through quantize -> dequantize, keeping its saturation as a clip post-op
             * ((lo - zp) * s .. (hi - zp) * s); only the rounding is dropped */
            const gr_op *q = &g->ops[u];
            const int u8 = g->t[q->out].lt.md.data_type == aiednn_u8;
            pair_q = u, pair_nm = *nm, pair_npo = p->attr.n_post_ops, pair_cur = cur;
            if (pair_npo > 0)
                pair_last = p->attr.post_ops[pair_npo - 1];
            gr_push_eltwise(p, aiednn_eltwise_clip, (float)((u8 ? 0 : -128) - q->attr.zero_point) * q->attr.scale,
                            (float)((u8 ? 255 : 127) - q->attr.zero_point) * q->attr.scale);
            members[(*nm)++] = u, members[(*nm)++] = dqp;
            cur = g->ops[dqp].out;
            continue;
        }
        const int nb = p->nb, ndq = p->ndq, npo = p->attr.n_post_ops;
        const int st = gr_chain_step(g, cur, chan, use_int, p, u);
        if (!st) {
            p->nb = nb, p->ndq = ndq, p->attr.n_post_ops = npo;
            break;
        }
        members[(*nm)++] = u;
        cur = g->ops[u].out;
        pair_q = -1; /* the chain went past the pair */
        if (st == 2)
            break;
    }
    if (pair_q >= 0) { /* nothing absorbed after a level-2 pair: undo it -- its 8-bit tensor would exist only
                        * inside this partition -- and end on the quantize as level 1 does */
        *nm = pair_nm, p->attr.n_post_ops = pair_npo, cur = pair_cur;
        if (pair_npo > 0)
            p->attr.post_ops[pair_npo - 1] = pair_last;
        const int nb = p->nb, ndq = p->ndq;
        if (gr_chain_step(g, cur, chan, use_int, p, pair_q) == 2) {
            members[(*nm)++] = pair_q;
            cur = g->ops[pair_q].out;
        } else {
            p->nb = nb, p->ndq = ndq;
        }
    }
    p->md[3] = g->t[cur].lt.md;
    if (vdims.ndims) /* written in the anchor's dims: the same bytes as the viewed tensor */
        vdims.data_type = p->md[3].data_type, p->md[3] = vdims;
    if (g->fuse) /* a same-parameter requant tail: the anchor's dims, the last tensor's bytes */
        cur = gr_qcopy_tail(g, cur, members, nm, GR_MAX_CHAIN);
    p->dst_id = g->t[cur].lt.id;
    gr_bind(p, AIEDNN_ARG_DST, g, cur);
    return 1;
}

/* ── everything else: one op, plus norm + quantize ────────────────────── */

static int gr_norm_axis(int axis, int r, int dflt) {
    return axis < 0 ? (axis == -1 && dflt >= 0 ? dflt : axis + r) : axis;
}

/* reduction dst as a same-rank md (reduced dims 1): from the mask, else a same-rank dst as given.
 * A keepdim-off dst has the same bytes, so only the descriptor changes. */
static int gr_reduce_dst(const aiednn_memory_desc_t *x, const aiednn_memory_desc_t *y, unsigned mask,
                         aiednn_memory_desc_t *out) {
    *out = *x;
    if (!mask) {
        if (y->ndims != x->ndims)
            return 0;
        *out = *y;
        return 1;
    }
    int k = 0;
    for (int i = 0; i < x->ndims; i++) {
        if (mask & (1u << i)) {
            out->dims[i] = 1;
            if (y->ndims == x->ndims)
                k++;
        } else if (k >= y->ndims || y->dims[k++] != x->dims[i]) {
            return 0;
        }
    }
    return k == y->ndims && gr_numel(out) == gr_numel(y);
}

/* output k of a training op: k 0 is the plan's dst (md[3]), the others md[4], md[5] and the extra list */
static void gr_bind_out(gr_plan *p, int arg, const struct aiednn_graph *g, int t, int k) {
    p->md[3 + k] = g->t[t].lt.md;
    gr_bind(p, arg, g, t);
    if (k == 0)
        p->dst_id = g->t[t].lt.id;
    else
        p->xo[p->nxo++] = g->t[t].lt.id;
}

static aiednn_conv_nd_params_t gr_conv_nd(const aiednn_conv_params_t *c) {
    aiednn_conv_nd_params_t n;
    memset(&n, 0, sizeof n);
    for (int i = 0; i < 2; i++)
        n.strides[i] = c->strides[i] ? c->strides[i] : 1, n.padding_l[i] = c->padding_l[i],
        n.padding_r[i] = c->padding_r[i], n.dilates[i] = c->dilates[i];
    n.groups = c->groups > 0 ? c->groups : 1;
    return n;
}

static aiednn_pool_nd_params_t gr_pool_nd(const aiednn_pool_params_t *c) {
    aiednn_pool_nd_params_t n;
    memset(&n, 0, sizeof n);
    for (int i = 0; i < 2; i++)
        n.kernel[i] = c->kernel[i], n.strides[i] = c->strides[i] ? c->strides[i] : 1, n.padding_l[i] = c->padding_l[i],
        n.padding_r[i] = c->padding_r[i], n.dilates[i] = c->dilates[i];
    return n;
}

/* the newer forward kinds (type_cast .. rms_norm) and the norms' training forward (3 outputs);
 * -1: not one of them */
static int gr_build_ext(const struct aiednn_graph *g, int oi, gr_plan *p) {
    const gr_op *o = &g->ops[oi];
    const int ni = o->nin, no = o->nout;
    const aiednn_memory_desc_t *x = ni ? &g->t[o->in[0]].lt.md : nullptr, *y = &g->t[o->out].lt.md;
    const int r = y->ndims;
    switch (o->kind) {
    case aiednn_gop_type_cast:
    case aiednn_gop_reorder:
        if (ni != 1 || no != 1 || x->ndims != r || memcmp(x->dims, y->dims, (size_t)r * sizeof(int64_t)))
            return 0;
        p->kind = GP_REORDER, p->md[0] = *x, p->scale = 1.f;
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
        break;
    case aiednn_gop_select: /* in: cond, then, else; the library's src0 (then) must be dst-shaped */
        if (ni != 3 || no != 1)
            return 0;
        p->kind = GP_BIN, p->alg = aiednn_binary_select, p->has_src2 = 1;
        p->md[0] = gr_expand(&g->t[o->in[1]].lt.md, r), p->md[1] = gr_expand(&g->t[o->in[2]].lt.md, r);
        p->md[2] = gr_expand(&g->t[o->in[0]].lt.md, r);
        if (gr_numel(&p->md[0]) != gr_numel(y) || !gr_bcast_to(&p->md[1], y) || !gr_bcast_to(&p->md[2], y))
            return 0;
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[1]), gr_bind(p, AIEDNN_ARG_SRC_1, g, o->in[2]);
        gr_bind(p, AIEDNN_ARG_SRC_2, g, o->in[0]);
        break;
    case aiednn_gop_squared_difference: { /* (a - b)^2 = (b - a)^2: either operand may be the full one */
        if (ni != 2 || no != 1)
            return 0;
        int s0 = o->in[0], s1 = o->in[1];
        if (gr_numel(&g->t[s0].lt.md) != gr_numel(y))
            s0 = o->in[1], s1 = o->in[0];
        p->kind = GP_QBIN, p->alg = aiednn_binary_sub;
        p->md[0] = gr_expand(&g->t[s0].lt.md, r), p->md[1] = gr_expand(&g->t[s1].lt.md, r);
        if (gr_numel(&p->md[0]) != gr_numel(y) || !gr_bcast_to(&p->md[1], y) ||
            !gr_push_eltwise(p, aiednn_eltwise_square, 0.f, 0.f))
            return 0;
        gr_bind(p, AIEDNN_ARG_SRC, g, s0), gr_bind(p, AIEDNN_ARG_SRC_1, g, s1);
        break;
    }
    case aiednn_gop_dropout:
        if (ni != 1 || o->attr.alpha < 0.f || o->attr.alpha >= 1.f)
            return 0;
        p->md[0] = *x;
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
        if (o->attr.alpha == 0.f && no == 1) { /* identity */
            p->kind = GP_COPY;
            break;
        }
        if (no != 2)
            return 0;
        p->kind = GP_DROP, p->alpha = o->attr.alpha, p->seed = o->attr.seed;
        gr_bind_out(p, AIEDNN_ARG_DST, g, o->outs[0], 0), gr_bind_out(p, AIEDNN_ARG_MASK, g, o->outs[1], 1);
        return 1;
    case aiednn_gop_gen_index: /* x gives the shape only: not bound, not read */
        if (ni != 1 || no != 1 || y->data_type != aiednn_s32 || x->ndims != r)
            return 0;
        p->kind = GP_GENIDX, p->axis = gr_norm_axis(o->attr.axis, r, -1);
        if (p->axis < 0 || p->axis >= r)
            return 0;
        break;
    case aiednn_gop_dynamic_quantize:
    case aiednn_gop_dynamic_dequantize: { /* x, scales [, zero_points]; the direction is checked on the types */
        const int q = o->kind == aiednn_gop_dynamic_quantize;
        if (ni < 2 || ni > 3 || no != 1 || (q ? !gr_is_8bit(y->data_type) : y->data_type != aiednn_f32))
            return 0;
        p->kind = GP_DYNQ, p->md[0] = *x, p->md[1] = g->t[o->in[1]].lt.md, p->has_bias = ni == 3;
        p->axis = o->attr.axis < 0 ? -1 : o->attr.axis;
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]), gr_bind(p, AIEDNN_ARG_SCALES, g, o->in[1]);
        if (ni == 3)
            p->md[2] = g->t[o->in[2]].lt.md, gr_bind(p, AIEDNN_ARG_ZERO_POINTS, g, o->in[2]);
        break;
    }
    case aiednn_gop_linear_attention: { /* q, k, v, gate [, beta] [, state_in] -> dst [, state_out] */
        const int delta = !(o->attr.flags & AIEDNN_LA_NO_DELTA), base = 4 + delta;
        if (ni < base || ni > base + 1 || no < 1 || no > 2)
            return 0;
        p->kind = GP_LINATT, p->flags = o->attr.flags, p->md[0] = *x;
        p->md[1] = g->t[o->in[1]].lt.md, p->md[2] = g->t[o->in[2]].lt.md, p->lmd[0] = g->t[o->in[3]].lt.md;
        p->sdpa_scale = o->attr.sdpa_scale > 0.f ? o->attr.sdpa_scale : 1.f / sqrtf((float)x->dims[x->ndims - 1]);
        gr_bind(p, AIEDNN_ARG_QUERY, g, o->in[0]), gr_bind(p, AIEDNN_ARG_KEY, g, o->in[1]);
        gr_bind(p, AIEDNN_ARG_VALUE, g, o->in[2]), gr_bind(p, AIEDNN_ARG_GATE, g, o->in[3]);
        if (delta)
            p->lmd[1] = g->t[o->in[4]].lt.md, gr_bind(p, AIEDNN_ARG_BETA, g, o->in[4]);
        if (ni == base + 1)
            p->lmd[2] = g->t[o->in[base]].lt.md, gr_bind(p, AIEDNN_ARG_STATE, g, o->in[base]);
        gr_bind_out(p, AIEDNN_ARG_DST, g, o->outs[0], 0);
        if (no == 2)
            gr_bind_out(p, AIEDNN_ARG_STATE_OUT, g, o->outs[1], 1);
        return 1;
    }
    case aiednn_gop_rms_norm:
        if (ni < 1 || ni > 2 || no != 1)
            return 0;
        p->kind = GP_LNORM, p->eps = o->attr.eps, p->md[0] = *x;
        p->flags = AIEDNN_RMS_NORM | (ni > 1 ? AIEDNN_USE_SCALE : 0u);
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
        if (ni > 1)
            gr_bind(p, AIEDNN_ARG_SCALE, g, o->in[1]);
        break;
    case aiednn_gop_layer_norm:
    case aiednn_gop_group_norm:
    case aiednn_gop_batch_norm_training: /* x [, gamma [, beta]] -> y, mean, variance */
        if (o->kind != aiednn_gop_batch_norm_training && no == 1)
            return -1;
        if (ni < 1 || ni > 3 || no != 3 || (o->kind == aiednn_gop_group_norm && o->attr.groups < 1))
            return 0;
        p->kind = o->kind == aiednn_gop_layer_norm ? GP_LNT : o->kind == aiednn_gop_group_norm ? GP_GNT : GP_BNT;
        p->eps = o->attr.eps, p->groups = o->attr.groups, p->md[0] = *x;
        p->flags = (o->kind == aiednn_gop_layer_norm ? o->attr.flags & AIEDNN_RMS_NORM : 0u) |
                   (ni > 1 ? AIEDNN_USE_SCALE : 0u) | (ni > 2 ? AIEDNN_USE_SHIFT : 0u);
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
        if (ni > 1)
            gr_bind(p, AIEDNN_ARG_SCALE, g, o->in[1]);
        if (ni > 2)
            gr_bind(p, AIEDNN_ARG_SHIFT, g, o->in[2]);
        gr_bind_out(p, AIEDNN_ARG_DST, g, o->outs[0], 0), gr_bind_out(p, AIEDNN_ARG_MEAN, g, o->outs[1], 1);
        gr_bind_out(p, AIEDNN_ARG_VARIANCE, g, o->outs[2], 2);
        return 1;
    default:
        return -1;
    }
    gr_bind_out(p, AIEDNN_ARG_DST, g, o->out, 0);
    return 1;
}

/* norm backward: x, diff_y, mean, variance [, gamma] -> diff_x [, diff_gamma [, diff_beta]] */
static int gr_build_norm_bwd(const struct aiednn_graph *g, const gr_op *o, gr_plan *p) {
    const int ni = o->nin, no = o->nout, has_g = ni == 5;
    const int rms = o->kind == aiednn_gop_layer_norm_backward && (o->attr.flags & AIEDNN_RMS_NORM);
    if (ni < 4 || ni > 5 || no < 1 || no > 3 || has_g != (no >= 2) || (rms && no == 3) ||
        (o->kind == aiednn_gop_group_norm_backward && o->attr.groups < 1))
        return 0;
    p->kind = o->kind == aiednn_gop_batch_norm_backward   ? GP_BNB
              : o->kind == aiednn_gop_layer_norm_backward ? GP_LNB
                                                          : GP_GNB;
    p->eps = o->attr.eps, p->groups = o->attr.groups;
    p->flags = (rms ? AIEDNN_RMS_NORM : 0u) | (has_g ? AIEDNN_USE_SCALE : 0u) | (no == 3 ? AIEDNN_USE_SHIFT : 0u);
    p->md[0] = g->t[o->in[0]].lt.md, p->md[1] = g->t[o->in[1]].lt.md;
    gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]), gr_bind(p, AIEDNN_ARG_DIFF_DST, g, o->in[1]);
    gr_bind(p, AIEDNN_ARG_MEAN, g, o->in[2]), gr_bind(p, AIEDNN_ARG_VARIANCE, g, o->in[3]);
    if (has_g)
        p->md[2] = g->t[o->in[4]].lt.md, gr_bind(p, AIEDNN_ARG_SCALE, g, o->in[4]);
    gr_bind_out(p, AIEDNN_ARG_DIFF_SRC, g, o->outs[0], 0);
    if (no >= 2)
        gr_bind_out(p, AIEDNN_ARG_DIFF_SCALE, g, o->outs[1], 1);
    if (no == 3)
        gr_bind_out(p, AIEDNN_ARG_DIFF_SHIFT, g, o->outs[2], 2);
    return 1;
}

/* the training (backward) kinds; -1: not one of them */
static int gr_build_train(const struct aiednn_graph *g, int oi, gr_plan *p) {
    const gr_op *o = &g->ops[oi];
    const int ni = o->nin, no = o->nout;
    const aiednn_memory_desc_t *y = &g->t[o->out].lt.md;
    const aiednn_memory_desc_t *i0 = ni > 0 ? &g->t[o->in[0]].lt.md : nullptr,
                               *i1 = ni > 1 ? &g->t[o->in[1]].lt.md : nullptr;
    switch (o->kind) {
    case aiednn_gop_eltwise_backward: /* src, diff_dst */
        if (ni != 2 || no != 1 || !o->attr.alg)
            return 0;
        p->kind = GP_ELTB, p->alg = o->attr.alg, p->alpha = o->attr.alpha, p->beta = o->attr.beta;
        p->md[0] = *i0, p->md[1] = *i1;
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]), gr_bind(p, AIEDNN_ARG_DIFF_DST, g, o->in[1]);
        break;
    case aiednn_gop_softmax_backward: /* dst, diff_dst */
        if (ni != 2 || no != 1)
            return 0;
        p->kind = GP_SMB, p->axis = gr_norm_axis(o->attr.axis, y->ndims, y->ndims - 1);
        p->alg = o->attr.alg == aiednn_softmax_log ? aiednn_softmax_log : aiednn_softmax_accurate;
        p->md[0] = *i0, p->md[1] = *i1;
        gr_bind(p, AIEDNN_ARG_DST, g, o->in[0]), gr_bind(p, AIEDNN_ARG_DIFF_DST, g, o->in[1]);
        break;
    case aiednn_gop_pool_backward: /* diff_dst [, src] */
        p->alg = o->attr.alg == aiednn_pooling_avg_include_padding || o->attr.alg == aiednn_pooling_avg_exclude_padding
                     ? o->attr.alg
                     : aiednn_pooling_max;
        if (ni < 1 || ni > 2 || no != 1 || (p->alg == aiednn_pooling_max && ni != 2))
            return 0;
        p->kind = GP_POOLB, p->pnd = gr_pool_nd(&o->attr.pool), p->md[1] = *i0, p->md[0] = ni > 1 ? *i1 : *y;
        gr_bind(p, AIEDNN_ARG_DIFF_DST, g, o->in[0]);
        if (ni > 1)
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[1]);
        break;
    case aiednn_gop_resampling_backward:
        if (ni != 1 || no != 1)
            return 0;
        p->kind = GP_RESB, p->md[1] = *i0;
        p->alg = o->attr.alg == aiednn_resampling_linear ? aiednn_resampling_linear : aiednn_resampling_nearest;
        gr_bind(p, AIEDNN_ARG_DIFF_DST, g, o->in[0]);
        break;
    case aiednn_gop_bias_add_backward: { /* sum of diff_dst over every dim but the channel */
        if (ni != 1 || no != 1 || y->ndims != 1)
            return 0;
        const int rx = i0->ndims, ax = o->attr.axis < 0 ? (rx >= 4 ? 1 : rx - 1) : o->attr.axis;
        if (ax >= rx || i0->dims[ax] != y->dims[0] || !gr_reduce_dst(i0, y, ((1u << rx) - 1) & ~(1u << ax), &p->md[3]))
            return 0;
        p->kind = GP_REDUCE, p->alg = aiednn_reduction_sum, p->md[0] = *i0;
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
        p->dst_id = g->t[o->out].lt.id;
        gr_bind(p, AIEDNN_ARG_DST, g, o->out);
        return 1;
    }
    case aiednn_gop_convolution_backward_data:
    case aiednn_gop_deconvolution_backward_data: /* diff_dst, weights */
        if (ni != 2 || no != 1)
            return 0;
        p->kind = o->kind == aiednn_gop_convolution_backward_data ? GP_CONVBD : GP_DECONVBD;
        p->cnd = gr_conv_nd(&o->attr.conv), p->md[0] = *i0, p->md[1] = *i1;
        gr_bind(p, AIEDNN_ARG_DIFF_DST, g, o->in[0]), gr_bind(p, AIEDNN_ARG_WEIGHTS, g, o->in[1]);
        break;
    case aiednn_gop_convolution_backward_weights:
    case aiednn_gop_deconvolution_backward_weights: /* src, diff_dst -> diff_weights [, diff_bias] */
        if (ni != 2 || no < 1 || no > 2)
            return 0;
        p->kind = o->kind == aiednn_gop_convolution_backward_weights ? GP_CONVBW : GP_DECONVBW;
        p->cnd = gr_conv_nd(&o->attr.conv), p->md[0] = *i0, p->md[1] = *i1, p->has_bias = no == 2;
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]), gr_bind(p, AIEDNN_ARG_DIFF_DST, g, o->in[1]);
        gr_bind_out(p, AIEDNN_ARG_DIFF_WEIGHTS, g, o->outs[0], 0);
        if (no == 2)
            gr_bind_out(p, AIEDNN_ARG_DIFF_BIAS, g, o->outs[1], 1);
        return 1;
    case aiednn_gop_batch_norm_backward:
    case aiednn_gop_layer_norm_backward:
    case aiednn_gop_group_norm_backward:
        return gr_build_norm_bwd(g, o, p);
    case aiednn_gop_prelu_backward: /* x, slope, diff_y -> diff_x, diff_slope */
        if (ni != 3 || no != 2)
            return 0;
        p->kind = GP_PRELUB, p->md[0] = *i0, p->md[1] = g->t[o->in[2]].lt.md;
        p->md[2] = gr_expand(i1, i0->ndims);
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]), gr_bind(p, AIEDNN_ARG_WEIGHTS, g, o->in[1]);
        gr_bind(p, AIEDNN_ARG_DIFF_DST, g, o->in[2]);
        gr_bind_out(p, AIEDNN_ARG_DIFF_SRC, g, o->outs[0], 0),
            gr_bind_out(p, AIEDNN_ARG_DIFF_WEIGHTS, g, o->outs[1], 1);
        p->md[4] = gr_expand(&p->md[4], i0->ndims);
        return 1;
    default:
        return -1;
    }
    gr_bind_out(p, AIEDNN_ARG_DIFF_SRC, g, o->out, 0);
    return 1;
}

/* the plan for a non-anchor op on its own; *why on failure */
static int gr_build_single(const struct aiednn_graph *g, int oi, gr_plan *p, const char **why) {
    const gr_op *o = &g->ops[oi];
    const aiednn_memory_desc_t *x = o->nin ? &g->t[o->in[0]].lt.md : nullptr, *y = &g->t[o->out].lt.md;
    const int r = y->ndims;
    aiednn_alg_kind_t alg;
    float a, b;
    p->md[3] = *y, p->dst_id = g->t[o->out].lt.id;
    *why = "unsupported attributes or inputs";
    int e = gr_build_ext(g, oi, p);
    if (e < 0)
        e = gr_build_train(g, oi, p);
    if (e >= 0)
        return e;
    if (o->nout != 1)
        return 0;
    if (o->kind == aiednn_gop_wildcard) {
        *why = "wildcard op";
        return 0;
    }
    if (gr_eltwise_of(o, &alg, &a, &b) && o->nin == 1) {
        p->kind = GP_ELT, p->alg = alg, p->alpha = a, p->beta = b, p->md[0] = *x;
        gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
    } else if ((gr_binary_of(o, &alg) || o->kind == aiednn_gop_bias_add) && o->nin == 2) {
        int s0 = o->in[0], s1 = o->in[1];
        aiednn_memory_desc_t m1 = g->t[s1].lt.md;
        if (o->kind == aiednn_gop_bias_add) { /* b [C] along the channel axis */
            if (m1.ndims != 1)
                return 0;
            int ax = o->attr.axis;
            if (ax < 0) { /* the dim b's length matches; a tie: channels-first at rank >= 4, else the last */
                const int m1d = r > 1 && y->dims[1] == m1.dims[0], ml = y->dims[r - 1] == m1.dims[0];
                ax = m1d && ml ? (r >= 4 ? 1 : r - 1) : m1d ? 1 : r - 1;
            }
            if (ax >= r)
                return 0;
            const int64_t c = m1.dims[0];
            m1.ndims = r;
            for (int i = 0; i < r; i++)
                m1.dims[i] = i == ax ? c : 1;
            alg = aiednn_binary_add;
        } else {
            m1 = gr_expand(&m1, r);
        }
        aiednn_memory_desc_t m0 = gr_expand(&g->t[s0].lt.md, r);
        const int commut = alg == aiednn_binary_add || alg == aiednn_binary_mul || alg == aiednn_binary_max ||
                           alg == aiednn_binary_min;
        if (!gr_bcast_to(&m0, y) || gr_numel(&m0) != gr_numel(y)) { /* src0 must be dst-shaped */
            if (!commut || gr_numel(&m1) != gr_numel(y))
                return 0;
            aiednn_memory_desc_t tm = m0;
            m0 = m1, m1 = tm;
            int ts = s0;
            s0 = s1, s1 = ts;
        }
        p->kind = GP_BIN, p->alg = alg, p->md[0] = m0, p->md[1] = m1;
        gr_bind(p, AIEDNN_ARG_SRC, g, s0), gr_bind(p, AIEDNN_ARG_SRC_1, g, s1);
    } else {
        switch (o->kind) {
        case aiednn_gop_max_pool:
        case aiednn_gop_avg_pool:
            p->kind = GP_POOL, p->pool = o->attr.pool, p->md[0] = *x;
            p->alg = o->kind == aiednn_gop_max_pool                      ? aiednn_pooling_max
                     : o->attr.alg == aiednn_pooling_avg_exclude_padding ? aiednn_pooling_avg_exclude_padding
                                                                         : aiednn_pooling_avg_include_padding;
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            break;
        case aiednn_gop_softmax:
            p->kind = GP_SOFTMAX, p->axis = gr_norm_axis(o->attr.axis, r, r - 1), p->md[0] = *x;
            p->alg = o->attr.alg == aiednn_softmax_log ? aiednn_softmax_log : aiednn_softmax_accurate;
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            break;
        case aiednn_gop_layer_norm:
        case aiednn_gop_batch_norm_inference: {
            const int bn = o->kind == aiednn_gop_batch_norm_inference, base = bn ? 3 : 1;
            if (o->nin < base || o->nin > base + 2)
                return 0;
            p->kind = bn ? GP_BNORM : GP_LNORM, p->eps = o->attr.eps, p->md[0] = *x;
            p->flags = (o->attr.flags & AIEDNN_RMS_NORM) | (o->nin > base ? AIEDNN_USE_SCALE : 0u) |
                       (o->nin > base + 1 ? AIEDNN_USE_SHIFT : 0u);
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            if (bn)
                gr_bind(p, AIEDNN_ARG_MEAN, g, o->in[1]), gr_bind(p, AIEDNN_ARG_VARIANCE, g, o->in[2]);
            if (o->nin > base)
                gr_bind(p, AIEDNN_ARG_SCALE, g, o->in[base]);
            if (o->nin > base + 1)
                gr_bind(p, AIEDNN_ARG_SHIFT, g, o->in[base + 1]);
            break;
        }
        case aiednn_gop_concat:
            if (o->nin < 1 || o->nin > AIEDNN_MAX_SRCS)
                return 0;
            p->kind = GP_CONCAT, p->axis = gr_norm_axis(o->attr.axis, r, -1), p->ncat = o->nin;
            p->cat = (aiednn_memory_desc_t *)malloc((size_t)o->nin * sizeof(aiednn_memory_desc_t));
            if (!p->cat)
                return 0;
            for (int i = 0; i < o->nin; i++) {
                p->cat[i] = g->t[o->in[i]].lt.md;
                gr_bind(p, AIEDNN_ARG_MULTIPLE_SRC + i, g, o->in[i]);
            }
            break;
        case aiednn_gop_slice:
            p->kind = GP_SLICE, p->md[0] = *x;
            memcpy(p->starts, o->attr.starts, sizeof p->starts);
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            break;
        case aiednn_gop_reshape:
            p->kind = GP_COPY, p->md[0] = *x;
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            break;
        case aiednn_gop_transpose:
        case aiednn_gop_quantize:
        case aiednn_gop_dequantize:
            if (o->kind != aiednn_gop_transpose && o->attr.n_scales > 0) {
                *why = "per-channel (de)quantize: only folded into a GEMM's weights";
                return 0;
            }
            p->kind = GP_REORDER, p->md[0] = *x, p->scale = 1.f;
            if (o->kind == aiednn_gop_transpose)
                p->has_perm = 1, memcpy(p->perm, o->attr.perm, sizeof p->perm);
            else
                p->scale = o->attr.scale, p->zp = o->attr.zero_point;
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            break;
        case aiednn_gop_resampling:
            p->kind = GP_RESAMPLE, p->md[0] = *x;
            p->alg = o->attr.alg == aiednn_resampling_linear ? aiednn_resampling_linear : aiednn_resampling_nearest;
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            break;
        case aiednn_gop_reduction:
            if (!gr_reduce_dst(x, y, o->attr.reduce_mask, &p->md[3]))
                return 0;
            p->kind = GP_REDUCE, p->md[0] = *x, p->pnorm = o->attr.pnorm, p->eps = o->attr.eps;
            p->alg = (int)o->attr.alg >= (int)aiednn_reduction_sum && (int)o->attr.alg < (int)aiednn_pooling_max
                         ? o->attr.alg
                         : aiednn_reduction_sum;
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            break;
        case aiednn_gop_group_norm:
            if (o->nin < 1 || o->nin > 3 || o->attr.groups < 1)
                return 0;
            p->kind = GP_GNORM, p->md[0] = *x, p->groups = o->attr.groups, p->eps = o->attr.eps;
            p->flags = (o->nin > 1 ? AIEDNN_USE_SCALE : 0u) | (o->nin > 2 ? AIEDNN_USE_SHIFT : 0u);
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]);
            if (o->nin > 1)
                gr_bind(p, AIEDNN_ARG_SCALE, g, o->in[1]);
            if (o->nin > 2)
                gr_bind(p, AIEDNN_ARG_SHIFT, g, o->in[2]);
            break;
        case aiednn_gop_prelu:
            if (o->nin != 2)
                return 0;
            p->kind = GP_PRELU, p->md[0] = *x, p->md[1] = gr_expand(&g->t[o->in[1]].lt.md, r);
            gr_bind(p, AIEDNN_ARG_SRC, g, o->in[0]), gr_bind(p, AIEDNN_ARG_WEIGHTS, g, o->in[1]);
            break;
        case aiednn_gop_sdpa:
            if (o->nin < 3 || o->nin > 4)
                return 0;
            p->kind = GP_SDPA, p->sdpa_scale = o->attr.sdpa_scale, p->causal = o->attr.causal;
            p->md[0] = *x, p->md[1] = g->t[o->in[1]].lt.md, p->md[2] = g->t[o->in[2]].lt.md;
            gr_bind(p, AIEDNN_ARG_QUERY, g, o->in[0]), gr_bind(p, AIEDNN_ARG_KEY, g, o->in[1]);
            gr_bind(p, AIEDNN_ARG_VALUE, g, o->in[2]);
            if (o->nin == 4) {
                p->has_bias = 1, p->md[4] = g->t[o->in[3]].lt.md;
                gr_bind(p, AIEDNN_ARG_ATTN_MASK, g, o->in[3]);
            }
            break;
        default:
            return 0;
        }
    }
    gr_bind(p, AIEDNN_ARG_DST, g, o->out);
    return 1;
}

/* the post-op chain after a non-GEMM op: the op computes into f32, then the GEMM epilogue in double.
 * Level 1 keeps what is exact against the unfused ops (relu / clip / max / min, one add / sub / mul with
 * no quantize after it, a final per-tensor quantize); level 2 takes any eltwise / binary. Returns the
 * number of ops absorbed (ops[1..]). */
static int gr_test(const gr_plan *p, char *impl, int cap);
static int gr_src_dq(const struct aiednn_graph *g, gr_plan *p, int arg, aiednn_memory_desc_t *md, int slot, int *ops,
                     int *n);
static void gr_sdpa_dq(const struct aiednn_graph *g, gr_plan *p, int *ops, int *n);

static int gr_pt_chain(const struct aiednn_graph *g, int cur, gr_plan *p, int *ops, int cap) {
    char impl[160];
    int n = 1, arith = 0, viewed = 0;
    const int r = g->t[cur].lt.md.ndims;
    while (n < cap) {
        const int u = gr_sole_user(g, cur);
        if (u < 0)
            break;
        const gr_op *o = &g->ops[u];
        gr_plan q;
        if (!gr_plan_copy(&q, p))
            break;
        aiednn_alg_kind_t alg;
        float a, b;
        int step = 0, end = 0;
        const int qonly = p->kind == GP_SLICE && g->fuse < 2; /* slice: the quantize alone is exact */
        if (qonly && o->kind != aiednn_gop_quantize) {
            gr_plan_free(&q);
            break;
        }
        if (o->nin == 1 && gr_eltwise_of(o, &alg, &a, &b)) {
            const int exact = (alg == aiednn_eltwise_relu && a == 0.f) || alg == aiednn_eltwise_clip;
            step = (exact || g->fuse >= 2) && gr_push_eltwise(&q, alg, a, b);
        } else if (!viewed && gr_is_view(g, o)) {
            step = viewed = 1; /* a dense view: the op writes it in its own dims; only eltwise / quantize after it */
        } else if (o->nin == 2 && !viewed && gr_binary_of(o, &alg) && q.attr.n_post_ops < AIEDNN_MAX_POST_OPS) {
            const int mm = alg == aiednn_binary_max || alg == aiednn_binary_min;
            const int ar = alg == aiednn_binary_add || alg == aiednn_binary_sub || alg == aiednn_binary_mul;
            const int other = o->in[0] == cur ? o->in[1] : o->in[0];
            if ((mm || (ar && !arith) || g->fuse >= 2) && (mm || ar) && other != cur &&
                (o->in[0] == cur || alg != aiednn_binary_sub)) {
                aiednn_post_op_t *po = &q.attr.post_ops[q.attr.n_post_ops];
                memset(po, 0, sizeof *po);
                po->kind = aiednn_post_op_binary, po->alg = alg;
                po->src1 = gr_expand(&g->t[other].lt.md, r);
                if (gr_bcast_to(&po->src1, &g->t[cur].lt.md)) {
                    gr_bind(&q, AIEDNN_ARG_ATTR_POST_OP_SRC(q.attr.n_post_ops), g, other);
                    q.attr.n_post_ops++, q.has_attr = 1, step = 1;
                    arith += ar;
                }
            }
        } else if (o->kind == aiednn_gop_quantize && o->nin == 1 && !o->attr.n_scales &&
                   gr_is_8bit(g->t[o->out].lt.md.data_type) && (!arith || g->fuse >= 2)) {
            q.attr.dst_scale = o->attr.scale, q.attr.dst_zero_point = o->attr.zero_point;
            q.has_attr = 1, step = 1, end = 1;
        }
        if (step) { /* the dst moves to this op's output (past a view: the op's dims, the new type) */
            const aiednn_data_type_t dt = g->t[o->out].lt.md.data_type;
            if (viewed)
                q.md[3].data_type = dt;
            else
                q.md[3] = g->t[o->out].lt.md;
            q.dst_id = g->t[o->out].lt.id;
            for (int b2 = 0; b2 < q.nb; b2++)
                if (q.arg[b2] == AIEDNN_ARG_DST)
                    q.tid[b2] = q.dst_id;
        }
        /* the library's rule: a create that accepts an attr applies it (aiednn_rt_ptail.h) */
        if (!step || !gr_test(&q, impl, sizeof impl)) {
            gr_plan_free(&q);
            break;
        }
        gr_plan_free(p);
        *p = q;
        ops[n++] = u;
        cur = o->out;
        if (end)
            break;
    }
    if (n > 1 && g->ops[ops[n - 1]].kind == aiednn_gop_quantize) { /* a same-parameter requant tail */
        const int t = gr_qcopy_tail(g, cur, ops, &n, cap);
        if (t != cur) {
            p->dst_id = g->t[t].lt.id;
            for (int b = 0; b < p->nb; b++)
                if (p->arg[b] == AIEDNN_ARG_DST)
                    p->tid[b] = p->dst_id;
        }
    }
    return n - 1;
}

/* ── partitioning ──────────────────────────────────────────────────────── */

static int gr_test(const gr_plan *p, char *impl, int cap) {
    aiednn_primitive_t pr;
    const aiednn_status_t st = gr_create(p, &pr);
    if (st != aiednn_success)
        return 0;
    if (pr) {
        snprintf(impl, (size_t)cap, "%s", aiednn_primitive_impl_info(pr));
        aiednn_primitive_destroy(pr);
    } else {
        snprintf(impl, (size_t)cap, "copy");
    }
    return 1;
}

static struct aiednn_partition *gr_new_part(struct aiednn_graph *g, int nops, const int *ops) {
    struct aiednn_partition *pt = &g->parts[g->nparts];
    memset(pt, 0, sizeof *pt);
    pt->g = g;
    pt->ops = (int *)malloc((size_t)(nops + GR_MAX_DQ) * sizeof(int));
    if (!pt->ops)
        return nullptr;
    memcpy(pt->ops, ops, (size_t)nops * sizeof(int));
    pt->nops = nops;
    for (int i = 0; i < nops; i++) {
        g->ops[ops[i]].part = g->nparts;
        if (g->ops[ops[i]].topo > pt->key)
            pt->key = g->ops[ops[i]].topo;
    }
    g->nparts++;
    return pt;
}

static void gr_describe(const struct aiednn_graph *g, struct aiednn_partition *pt, int int8, const char *tail) {
    int n = 0;
    for (int i = 0; i < pt->nops && n < (int)sizeof pt->info - 1; i++)
        n += snprintf(pt->info + n, sizeof pt->info - (size_t)n, "%s%s", i ? "+" : "",
                      gr_kind_name(g->ops[pt->ops[i]].kind));
    if (n < (int)sizeof pt->info - 1)
        snprintf(pt->info + n, sizeof pt->info - (size_t)n, "%s %s", int8 ? " (int8)" : "", tail);
}

/* the anchor at ai: the longest chain the library accepts, int8 first */
static int gr_part_anchor(struct aiednn_graph *g, int ai) {
    int members[GR_MAX_CHAIN], nm = 0;
    char impl[160];
    for (int use_int = g->fuse ? 1 : 0; use_int >= 0; use_int--) { /* the read-through is fusion too */
        gr_plan p;
        memset(&p, 0, sizeof p);
        if (!gr_build_anchor(g, ai, use_int, GR_MAX_CHAIN - 1, &p, members, &nm)) {
            gr_plan_free(&p);
            continue;
        }
        for (int len = nm - 1; len >= 0; len--) {
            gr_plan q;
            memset(&q, 0, sizeof q);
            int m2[GR_MAX_CHAIN], n2 = 0;
            if (len < nm - 1 ? !gr_build_anchor(g, ai, use_int, len, &q, m2, &n2) : !gr_plan_copy(&q, &p)) {
                gr_plan_free(&q);
                continue;
            }
            if (len == nm - 1)
                memcpy(m2, members, sizeof members), n2 = nm;
            if (gr_test(&q, impl, sizeof impl)) {
                gr_plan_free(&p);
                struct aiednn_partition *pt = gr_new_part(g, n2, m2);
                if (!pt)
                    return 0;
                pt->supported = 1, pt->plan = q;
                char tail[176];
                snprintf(tail, sizeof tail, "-> %s", impl);
                gr_describe(g, pt, use_int, tail);
                return 1;
            }
            gr_plan_free(&q);
        }
        gr_plan_free(&p);
    }
    struct aiednn_partition *pt = gr_new_part(g, 1, &ai);
    if (!pt)
        return 0;
    gr_describe(g, pt, 0, "unsupported: no implementation accepts it");
    return 1;
}

/* dequantize x2 -> add | sub | mul | max | min [-> relu | clamp] -> quantize, or with one f32 source / no
 * quantize: one aiednn_binary_create_attr (bit-identical to the unfused ops), the dequantizes read through */
static int gr_try_qbin(struct aiednn_graph *g, int oi, gr_plan *p, int *ops, int *n, char *impl, int cap) {
    const gr_op *o = &g->ops[oi];
    aiednn_alg_kind_t alg;
    if (!g->fuse || o->nin != 2 || !gr_binary_of(o, &alg) ||
        !(alg == aiednn_binary_add || alg == aiednn_binary_sub || alg == aiednn_binary_mul ||
          alg == aiednn_binary_max || alg == aiednn_binary_min))
        return 0;
    int src[2], nq = 0;
    float sc[2] = {0.f, 0.f};
    int32_t zp[2] = {0, 0};
    for (int i = 0; i < 2; i++) {
        gr_q q;
        src[i] = o->in[i];
        if (gr_dq(g, o->in[i], -1, 0, &q)) {
            src[i] = q.qt, sc[i] = q.s, zp[i] = q.zp, nq++;
            gr_add_dq(p, q.op);
        } else if (g->t[o->in[i]].lt.md.data_type != aiednn_f32) {
            return 0;
        }
    }
    const aiednn_memory_desc_t *y = &g->t[o->out].lt.md;
    const int r = y->ndims;
    aiednn_memory_desc_t m0 = gr_expand(&g->t[src[0]].lt.md, r), m1 = gr_expand(&g->t[src[1]].lt.md, r);
    if (!gr_bcast_to(&m0, y) || gr_numel(&m0) != gr_numel(y)) { /* src0 must be dst-shaped */
        if (alg == aiednn_binary_sub)
            return 0;
        aiednn_memory_desc_t tm = m0;
        m0 = m1, m1 = tm;
        const int ts = src[0];
        src[0] = src[1], src[1] = ts;
        const float tf = sc[0];
        sc[0] = sc[1], sc[1] = tf;
        const int32_t tz = zp[0];
        zp[0] = zp[1], zp[1] = tz;
    }
    if (!nq || !gr_bcast_to(&m0, y) || gr_numel(&m0) != gr_numel(y) || !gr_bcast_to(&m1, y))
        return 0;
    p->kind = GP_QBIN, p->alg = alg, p->md[0] = m0, p->md[1] = m1, p->has_attr = 1;
    p->attr.src_scales[0] = sc[0], p->attr.src_scales[1] = sc[1];
    p->attr.src_zero_points[0] = zp[0], p->attr.src_zero_points[1] = zp[1];
    ops[0] = oi, *n = 1;
    int cur = o->out, u = gr_sole_user(g, cur);
    aiednn_alg_kind_t ea;
    float a, b;
    if (u >= 0 && g->ops[u].nin == 1 && gr_eltwise_of(&g->ops[u], &ea, &a, &b) &&
        (ea == aiednn_eltwise_relu || ea == aiednn_eltwise_clip)) {
        aiednn_post_op_t *po = &p->attr.post_ops[p->attr.n_post_ops++];
        memset(po, 0, sizeof *po);
        po->kind = aiednn_post_op_eltwise, po->alg = ea, po->alpha = a, po->beta = b;
        ops[(*n)++] = u;
        cur = g->ops[u].out, u = gr_sole_user(g, cur);
    }
    if (u >= 0 && g->ops[u].kind == aiednn_gop_quantize && g->ops[u].nin == 1 && !g->ops[u].attr.n_scales &&
        gr_is_8bit(g->t[g->ops[u].out].lt.md.data_type)) {
        p->attr.dst_scale = g->ops[u].attr.scale, p->attr.dst_zero_point = g->ops[u].attr.zero_point;
        ops[(*n)++] = u;
        cur = g->ops[u].out;
    }
    p->md[3] = g->t[cur].lt.md, p->dst_id = g->t[cur].lt.id;
    gr_bind(p, AIEDNN_ARG_SRC, g, src[0]), gr_bind(p, AIEDNN_ARG_SRC_1, g, src[1]);
    gr_bind(p, AIEDNN_ARG_DST, g, cur);
    if (alg == aiednn_binary_mul && g->fuse >= 2) { /* level 2: the AIE mul, within one code of the exact table */
        p->attr.flags = AIEDNN_ATTR_TOLERANCE_1CODE;
        if (gr_test(p, impl, cap))
            return 1;
        p->attr.flags = 0;
    }
    return gr_test(p, impl, cap);
}

/* dequantize -> unary eltwise [-> up to AIEDNN_MAX_POST_OPS more] [-> quantize]: one
 * aiednn_eltwise_forward_create_attr (bit-identical to the ops) */
static int gr_try_qelt(struct aiednn_graph *g, int oi, gr_plan *p, int *ops, int *n, char *impl, int cap) {
    const gr_op *o = &g->ops[oi];
    aiednn_alg_kind_t alg;
    float a, b;
    gr_q q;
    if (!g->fuse || o->nin != 1 || !gr_eltwise_of(o, &alg, &a, &b) || !gr_dq(g, o->in[0], -1, 0, &q))
        return 0;
    gr_add_dq(p, q.op);
    p->kind = GP_QELT, p->alg = alg, p->alpha = a, p->beta = b, p->has_attr = 1;
    p->md[0] = g->t[q.qt].lt.md;
    p->attr.src_scales[0] = q.s, p->attr.src_zero_points[0] = q.zp;
    ops[0] = oi, *n = 1;
    int cur = o->out, u = gr_sole_user(g, cur);
    while (u >= 0 && p->attr.n_post_ops < AIEDNN_MAX_POST_OPS && g->ops[u].nin == 1 &&
           gr_eltwise_of(&g->ops[u], &alg, &a, &b)) {
        aiednn_post_op_t *po = &p->attr.post_ops[p->attr.n_post_ops++];
        memset(po, 0, sizeof *po);
        po->kind = aiednn_post_op_eltwise, po->alg = alg, po->alpha = a, po->beta = b;
        ops[(*n)++] = u;
        cur = g->ops[u].out, u = gr_sole_user(g, cur);
    }
    if (u >= 0 && g->ops[u].kind == aiednn_gop_quantize && g->ops[u].nin == 1 && !g->ops[u].attr.n_scales &&
        gr_is_8bit(g->t[g->ops[u].out].lt.md.data_type)) {
        p->attr.dst_scale = g->ops[u].attr.scale, p->attr.dst_zero_point = g->ops[u].attr.zero_point;
        ops[(*n)++] = u;
        cur = g->ops[u].out;
    }
    p->md[3] = g->t[cur].lt.md, p->dst_id = g->t[cur].lt.id;
    gr_bind(p, AIEDNN_ARG_SRC, g, q.qt), gr_bind(p, AIEDNN_ARG_DST, g, cur);
    return gr_test(p, impl, cap);
}

/* dequantize -> max_pool -> quantize with the same per-tensor scale / zero point / type: max_pool on the
 * 8-bit codes, exactly */
static int gr_try_qpool(struct aiednn_graph *g, int oi, gr_plan *p, int *ops, int *n, char *impl, int cap) {
    const gr_op *o = &g->ops[oi];
    gr_q q;
    if (!g->fuse || o->kind != aiednn_gop_max_pool || o->nin != 1 || !gr_dq(g, o->in[0], -1, 0, &q))
        return 0;
    const int u = gr_sole_user(g, o->out);
    if (u < 0 || g->ops[u].kind != aiednn_gop_quantize || g->ops[u].nin != 1 || g->ops[u].attr.n_scales ||
        g->ops[u].attr.scale != q.s || g->ops[u].attr.zero_point != q.zp ||
        g->t[g->ops[u].out].lt.md.data_type != g->t[q.qt].lt.md.data_type)
        return 0;
    gr_add_dq(p, q.op);
    p->kind = GP_POOL, p->alg = aiednn_pooling_max, p->pool = o->attr.pool;
    p->md[0] = g->t[q.qt].lt.md, p->md[3] = g->t[g->ops[u].out].lt.md, p->dst_id = g->t[g->ops[u].out].lt.id;
    gr_bind(p, AIEDNN_ARG_SRC, g, q.qt), gr_bind(p, AIEDNN_ARG_DST, g, g->ops[u].out);
    ops[0] = oi, ops[1] = u, *n = 2;
    return gr_test(p, impl, cap);
}

/* dequantize -> reshape | transpose | slice -> quantize with the same per-tensor parameters: the view /
 * slice on the 8-bit codes */
static int gr_try_qview(struct aiednn_graph *g, int oi, gr_plan *p, int *ops, int *n, char *impl, int cap) {
    const gr_op *o = &g->ops[oi];
    gr_q q;
    if (!g->fuse || (o->kind != aiednn_gop_reshape && o->kind != aiednn_gop_transpose && o->kind != aiednn_gop_slice) ||
        o->nin != 1 || !gr_dq(g, o->in[0], -1, 0, &q))
        return 0;
    const int u = gr_sole_user(g, o->out);
    if (u < 0 || g->ops[u].kind != aiednn_gop_quantize || g->ops[u].nin != 1 || g->ops[u].attr.n_scales ||
        g->ops[u].attr.scale != q.s || g->ops[u].attr.zero_point != q.zp ||
        g->t[g->ops[u].out].lt.md.data_type != g->t[q.qt].lt.md.data_type)
        return 0;
    gr_add_dq(p, q.op);
    p->md[0] = g->t[q.qt].lt.md, p->md[3] = g->t[g->ops[u].out].lt.md, p->dst_id = g->t[g->ops[u].out].lt.id;
    if (o->kind == aiednn_gop_reshape) {
        p->kind = GP_COPY;
    } else if (o->kind == aiednn_gop_slice) { /* the slice of the codes */
        p->kind = GP_SLICE;
        memcpy(p->starts, o->attr.starts, sizeof p->starts);
    } else { /* a same-type reorder: a pure permute of the codes */
        p->kind = GP_REORDER, p->has_perm = 1, p->scale = 1.f, p->zp = 0;
        memcpy(p->perm, o->attr.perm, sizeof p->perm);
    }
    gr_bind(p, AIEDNN_ARG_SRC, g, q.qt), gr_bind(p, AIEDNN_ARG_DST, g, g->ops[u].out);
    ops[0] = oi, ops[1] = u, *n = 2;
    return gr_test(p, impl, cap);
}

/* dequantize each input -> concat -> quantize: aiednn_concat_create_q (exact) -- densenet / yolo */
static int gr_try_qconcat(struct aiednn_graph *g, int oi, gr_plan *p, int *ops, int *n, char *impl, int cap) {
    const gr_op *o = &g->ops[oi];
    if (!g->fuse || o->kind != aiednn_gop_concat || o->nin < 2 || o->nin > AIEDNN_MAX_SRCS)
        return 0;
    const int u = gr_sole_user(g, o->out);
    if (u < 0 || g->ops[u].kind != aiednn_gop_quantize || g->ops[u].attr.n_scales ||
        !gr_is_8bit(g->t[g->ops[u].out].lt.md.data_type))
        return 0;
    p->kind = GP_QCAT, p->ncat = o->nin, p->bnsc = o->nin;
    p->axis = gr_norm_axis(o->attr.axis, g->t[o->out].lt.md.ndims, -1);
    p->cat = (aiednn_memory_desc_t *)malloc((size_t)o->nin * sizeof(aiednn_memory_desc_t));
    p->bsc = (float *)malloc((size_t)o->nin * sizeof(float));
    p->bzp = (int32_t *)malloc((size_t)o->nin * sizeof(int32_t));
    if (!p->cat || !p->bsc || !p->bzp)
        return 0;
    for (int i = 0; i < o->nin; i++) {
        gr_q q;
        if (!gr_dq(g, o->in[i], -1, 0, &q))
            return 0;
        p->cat[i] = g->t[q.qt].lt.md, p->bsc[i] = q.s, p->bzp[i] = q.zp;
        gr_add_dq(p, q.op);
        gr_bind(p, AIEDNN_ARG_MULTIPLE_SRC + i, g, q.qt);
    }
    p->attr.dst_scale = g->ops[u].attr.scale, p->attr.dst_zero_point = g->ops[u].attr.zero_point;
    p->md[3] = g->t[g->ops[u].out].lt.md, p->dst_id = g->t[g->ops[u].out].lt.id;
    gr_bind(p, AIEDNN_ARG_DST, g, g->ops[u].out);
    ops[0] = oi, ops[1] = u, *n = 2;
    return gr_test(p, impl, cap);
}

static int gr_part_single(struct aiednn_graph *g, int oi) {
    gr_plan p;
    memset(&p, 0, sizeof p);
    const char *why = "";
    char impl[160];
    int ops[2 + AIEDNN_MAX_POST_OPS + GR_MAX_CHAIN] = {oi, -1}, n = 1;
    int fused = gr_try_qbin(g, oi, &p, ops, &n, impl, sizeof impl);
    if (!fused) {
        gr_plan_free(&p);
        memset(&p, 0, sizeof p);
        n = 1;
        fused = gr_try_qelt(g, oi, &p, ops, &n, impl, sizeof impl);
    }
    if (!fused) {
        gr_plan_free(&p);
        memset(&p, 0, sizeof p);
        n = 1;
        fused = gr_try_qpool(g, oi, &p, ops, &n, impl, sizeof impl);
    }
    if (!fused) {
        gr_plan_free(&p);
        memset(&p, 0, sizeof p);
        n = 1;
        fused = gr_try_qview(g, oi, &p, ops, &n, impl, sizeof impl);
    }
    if (!fused) {
        gr_plan_free(&p);
        memset(&p, 0, sizeof p);
        n = 1;
        fused = gr_try_qconcat(g, oi, &p, ops, &n, impl, sizeof impl);
    }
    if (fused) {
        struct aiednn_partition *pt = gr_new_part(g, n, ops);
        if (!pt) {
            gr_plan_free(&p);
            return 0;
        }
        pt->supported = 1, pt->plan = p;
        char tail[176];
        snprintf(tail, sizeof tail, "-> %s", impl);
        gr_describe(g, pt, 1, tail);
        return 1;
    }
    gr_plan_free(&p);
    memset(&p, 0, sizeof p);
    n = 1;
    int ok = gr_build_single(g, oi, &p, &why);
    if (ok && g->fuse && (p.kind == GP_LNORM || p.kind == GP_SOFTMAX)) { /* an 8-bit source read through its DQ */
        gr_plan q;
        int qops[4], qn = 0;
        if (gr_plan_copy(&q, &p) && gr_src_dq(g, &q, AIEDNN_ARG_SRC, &q.md[0], 0, qops, &qn) && qn == 0 &&
            gr_test(&q, impl, sizeof impl)) {
            gr_plan_free(&p);
            p = q;
        } else {
            gr_plan_free(&q);
        }
    }
    if (ok && g->fuse &&
        (p.kind == GP_POOL || p.kind == GP_SOFTMAX || p.kind == GP_REDUCE || p.kind == GP_RESAMPLE ||
         p.kind == GP_GNORM || p.kind == GP_BNORM || p.kind == GP_LNORM || p.kind == GP_REORDER ||
         p.kind == GP_SLICE)) {
        const int k = gr_pt_chain(g, g->ops[oi].out, &p, ops, 2 + AIEDNN_MAX_POST_OPS + GR_MAX_CHAIN);
        if (k > 0)
            n = 1 + k, gr_test(&p, impl, sizeof impl);
    }
    if (ok && n == 1 && !gr_test(&p, impl, sizeof impl)) {
        ok = 0;
        why = "no implementation accepts it";
    }
    struct aiednn_partition *pt = gr_new_part(g, n, ops);
    if (!pt) {
        gr_plan_free(&p);
        return 0;
    }
    char tail[176];
    if (ok) {
        pt->supported = 1, pt->plan = p;
        snprintf(tail, sizeof tail, "-> %s", impl);
    } else {
        gr_plan_free(&p);
        snprintf(tail, sizeof tail, "unsupported: %s", why);
    }
    gr_describe(g, pt, 0, tail);
    return 1;
}

/* a dequantize nobody reads through stays a reorder; one every consumer reads through joins the
 * first of their partitions */
static int gr_part_dequant(struct aiednn_graph *g, int di) {
    const int t = g->ops[di].out;
    int first = -1, all = g->t[t].nuse > 0 && !g->t[t].marked;
    for (int k = 0; k < g->nops && all; k++) {
        const gr_op *u = &g->ops[k];
        int uses = 0;
        for (int i = 0; i < u->nin; i++)
            uses |= u->in[i] == t;
        if (!uses)
            continue;
        const struct aiednn_partition *pt = u->part >= 0 ? &g->parts[u->part] : nullptr;
        int through = 0;
        for (int j = 0; pt && pt->supported && j < pt->plan.ndq; j++)
            through |= pt->plan.dq[j] == di;
        if (!through)
            all = 0;
        else if (first < 0 || g->parts[first].key > pt->key)
            first = u->part;
    }
    if (all && first >= 0) {
        struct aiednn_partition *pt = &g->parts[first];
        int pos = pt->nops; /* keep ops topological */
        while (pos > 0 && g->ops[pt->ops[pos - 1]].topo > g->ops[di].topo)
            pos--;
        memmove(pt->ops + pos + 1, pt->ops + pos, (size_t)(pt->nops - pos) * sizeof(int));
        pt->ops[pos] = di, pt->nops++;
        g->ops[di].part = first;
        char buf[sizeof pt->info];
        snprintf(buf, sizeof buf, "dequantize+%.240s", pt->info);
        memcpy(pt->info, buf, sizeof buf);
        return 1;
    }
    return gr_part_single(g, di);
}

/* ── layout propagation (doc/design/aiednn_framework_api.md section 5) ─── */

/* partitions whose primitive takes nhwc natively or for free; a reshape / transpose / GEMM reading a 4-D
 * tensor depends on its memory order */
static int gr_nhwc_kind(int kind) {
    return kind == GP_CONV || kind == GP_DECONV || kind == GP_ELT || kind == GP_BIN || kind == GP_QBIN ||
           kind == GP_QELT || kind == GP_CONCAT || kind == GP_POOL || kind == GP_PRELU;
}

/* every descriptor of plan p bound to tensor id: the argument's md slot (-1 when none) */
static aiednn_memory_desc_t *gr_plan_md(gr_plan *p, int arg) {
    if (arg >= AIEDNN_ARG_MULTIPLE_SRC && arg < AIEDNN_ARG_MULTIPLE_SRC + p->ncat && p->cat)
        return &p->cat[arg - AIEDNN_ARG_MULTIPLE_SRC];
    if (arg >= AIEDNN_ARG_ATTR_POST_OP_SRC(0) && arg < AIEDNN_ARG_ATTR_POST_OP_SRC(AIEDNN_MAX_POST_OPS))
        return &p->attr.post_ops[arg - AIEDNN_ARG_ATTR_POST_OP_SRC(0)].src1;
    switch (arg) {
    case AIEDNN_ARG_SRC:
        return &p->md[0];
    case AIEDNN_ARG_SRC_1:
    case AIEDNN_ARG_WEIGHTS:
        return &p->md[1];
    case AIEDNN_ARG_SRC_2:
        return &p->md[2];
    case AIEDNN_ARG_DST:
        return &p->md[3];
    case AIEDNN_ARG_ATTN_MASK:
        return &p->md[4];
    default:
        return nullptr;
    }
}

/* set tensor id's format in every supported partition binding it; 0 if a descriptor is not 4-D */
static int gr_set_fmt(struct aiednn_graph *g, uint64_t id, aiednn_format_t f) {
    for (int i = 0; i < g->nparts; i++) {
        struct aiednn_partition *P = &g->parts[i];
        if (!P->supported)
            continue;
        for (int b = 0; b < P->plan.nb; b++) {
            if (P->plan.tid[b] != id)
                continue;
            aiednn_memory_desc_t *md = gr_plan_md(&P->plan, P->plan.arg[b]);
            const int t = gr_find_tensor(g, id);
            if (!md || md->ndims != 4 || t < 0 || g->t[t].lt.md.ndims != 4 || /* not through a view */
                memcmp(md->dims, g->t[t].lt.md.dims, 4 * sizeof(int64_t)))
                return 0;
            md->format = f;
        }
    }
    return 1;
}

/* nhwc on every 4-D activation edge between nhwc-friendly partitions, proposed all at once; then, until
 * nothing changes, a partition that no longer creates (or only via "fmtwrap") sends its edges back to
 * dense. The set only shrinks, so this ends. */
static void gr_propagate_layouts(struct aiednn_graph *g) {
    char impl[160];
    const char *dbg_e = getenv("AIEDNN_GRAPH_LAYOUT_DEBUG");
    const int dbg = dbg_e && dbg_e[0] && dbg_e[0] != '0';
    char *cand = (char *)calloc((size_t)(g->nt ? g->nt : 1), 1);
    if (!cand)
        return;
    for (int t = 0; t < g->nt; t++) { /* structural eligibility */
        const gr_tensor *T = &g->t[t];
        if (T->lt.md.ndims != 4 || T->marked || T->nuse == 0 || T->producer < 0 ||
            T->lt.property == aiednn_lt_constant || T->lt.md.format != aiednn_format_dense)
            continue;
        const uint64_t id = T->lt.id;
        int prod = 0, cons = 0, ok = 1;
        for (int i = 0; i < g->nparts && ok; i++) {
            const struct aiednn_partition *P = &g->parts[i];
            int binds = 0, is_dst = 0, reads = 0;
            for (int b = 0; P->supported && b < P->plan.nb; b++)
                if (P->plan.tid[b] == id)
                    binds = 1, is_dst |= P->plan.arg[b] == AIEDNN_ARG_DST;
            for (int k = 0; !P->supported && k < P->nops; k++) /* the framework runs it: it must see dense */
                for (int j = 0; j < g->ops[P->ops[k]].nin; j++)
                    reads |= g->ops[P->ops[k]].in[j] == t;
            if (reads || (!P->supported && g->ops[P->ops[P->nops - 1]].out == t))
                ok = 0;
            else if (binds) {
                ok = gr_nhwc_kind(P->plan.kind);
                prod += is_dst, cons += !is_dst;
            }
        }
        if (ok && prod == 1 && cons >= 1 && gr_set_fmt(g, id, aiednn_format_nhwc))
            cand[t] = 1;
        else
            gr_set_fmt(g, id, aiednn_format_dense);
    }
    for (int changed = 1; changed;) { /* every touched partition creates without a format wrap */
        changed = 0;
        for (int i = 0; i < g->nparts; i++) {
            const struct aiednn_partition *P = &g->parts[i];
            int touches = 0;
            for (int b = 0; P->supported && b < P->plan.nb && !touches; b++) {
                const int t = gr_find_tensor(g, P->plan.tid[b]);
                touches = t >= 0 && cand[t];
            }
            if (!touches || (gr_test(&P->plan, impl, sizeof impl) && !strstr(impl, "fmtwrap")))
                continue;
            if (dbg)
                printf("aiednn_graph_layout,dense,partition %d (%.60s) -> %s\n", i, P->info, impl);
            for (int b = 0; b < P->plan.nb; b++) {
                const int t = gr_find_tensor(g, P->plan.tid[b]);
                if (t >= 0 && cand[t]) {
                    cand[t] = 0, changed = 1;
                    gr_set_fmt(g, P->plan.tid[b], aiednn_format_dense);
                }
            }
        }
    }
    for (int t = 0; t < g->nt; t++)
        if (cand[t])
            g->t[t].lt.md.format = aiednn_format_nhwc;
    free(cand);
}

/* ── decomposed attention -> one sdpa ──────────────────────────────────── */

static int gr_is_scalar_f32(const struct aiednn_graph *g, int t) {
    return gr_numel(&g->t[t].lt.md) == 1 && g->t[t].lt.md.data_type == aiednn_f32;
}

/* matmul(Q, K^T) -> multiply | divide by a scalar -> [add mask] -> softmax(last axis) -> matmul(., V): one
 * aiednn_sdpa_forward_create. The scalar is bound at execute (AIEDNN_ARG_ATTR_SCALE; a divide binds its
 * reciprocal). f32, 4-D [B, H, S, D]. Returns 1 with the partition made, 0 otherwise. */
static void gr_sdpa_bshd(const struct aiednn_graph *g, gr_plan *p, int *ops, int *n);

/* after the sdpa (and its folded head merge): a view and a per-tensor quantize as for a norm (gr_pt_chain),
 * then any same-parameter requant tail; kept only where the create accepts it */
static void gr_sdpa_tail(const struct aiednn_graph *g, gr_plan *p, int *ops, int *n, int cap) {
    int t[2 + AIEDNN_MAX_POST_OPS + GR_MAX_CHAIN];
    const int d = g->fuse ? gr_find_tensor(g, p->dst_id) : -1;
    if (d < 0)
        return;
    const int k = gr_pt_chain(g, d, p, t, (int)(sizeof t / sizeof t[0]));
    for (int i = 1; i <= k && *n < cap; i++)
        ops[(*n)++] = t[i];
}
static int gr_sdpa_views(const struct aiednn_graph *g, gr_plan *p, int *ops, int *n);

static int gr_part_sdpa(struct aiednn_graph *g, int m1) {
    const gr_op *a = &g->ops[m1];
    if (!g->fuse || a->kind != aiednn_gop_matmul || a->nin != 2 || a->part >= 0)
        return 0;
    int ops[40], n = 0, q = a->in[0], k = a->in[1];
    ops[n++] = m1;
    const aiednn_memory_desc_t *qm = &g->t[q].lt.md;
    if (qm->ndims != 4 || qm->data_type != aiednn_f32)
        return 0;
    if (!a->attr.weights_transposed) { /* q @ k.transpose(-2, -1) */
        const int tp = g->t[k].producer;
        const gr_op *T = tp >= 0 ? &g->ops[tp] : nullptr;
        if (!T || T->kind != aiednn_gop_transpose || T->nin != 1 || gr_sole_user(g, k) != m1 || T->part >= 0 ||
            T->attr.perm[0] != 0 || T->attr.perm[1] != 1 || T->attr.perm[2] != 3 || T->attr.perm[3] != 2)
            return 0;
        k = T->in[0];
        ops[n++] = tp;
    }
    int u = gr_sole_user(g, a->out), scale_t = -1, sdiv = 0, mask = -1;
    aiednn_alg_kind_t alg;
    if (u >= 0 && gr_binary_of(&g->ops[u], &alg) && (alg == aiednn_binary_mul || alg == aiednn_binary_div) &&
        g->ops[u].nin == 2) {
        const gr_op *B = &g->ops[u];
        const int other = B->in[0] == a->out ? B->in[1] : B->in[0];
        if (!gr_is_scalar_f32(g, other) || (alg == aiednn_binary_div && B->in[0] != a->out))
            return 0;
        scale_t = other, sdiv = alg == aiednn_binary_div;
        ops[n++] = u;
        u = gr_sole_user(g, B->out);
    }
    if (u >= 0 && gr_binary_of(&g->ops[u], &alg) && alg == aiednn_binary_add && g->ops[u].nin == 2) {
        const gr_op *B = &g->ops[u];
        const int cur = g->ops[ops[n - 1]].out;
        mask = B->in[0] == cur ? B->in[1] : B->in[0];
        ops[n++] = u;
        u = gr_sole_user(g, B->out);
    }
    if (u < 0 || g->ops[u].kind != aiednn_gop_softmax || g->ops[u].nin != 1)
        return 0;
    const int sax = g->ops[u].attr.axis;
    if (!(sax == -1 || sax == 3))
        return 0;
    ops[n++] = u;
    const int m2 = gr_sole_user(g, g->ops[u].out);
    if (m2 < 0 || g->ops[m2].kind != aiednn_gop_matmul || g->ops[m2].nin != 2 || g->ops[m2].attr.weights_transposed ||
        g->ops[m2].in[0] != g->ops[u].out)
        return 0;
    ops[n++] = m2;
    gr_plan p;
    memset(&p, 0, sizeof p);
    p.kind = GP_SDPA, p.sdpa_scale = 1.f, p.causal = 0, p.sdiv = sdiv;
    p.md[0] = *qm, p.md[1] = g->t[k].lt.md, p.md[2] = g->t[g->ops[m2].in[1]].lt.md;
    p.md[3] = g->t[g->ops[m2].out].lt.md, p.dst_id = g->t[g->ops[m2].out].lt.id;
    gr_bind(&p, AIEDNN_ARG_QUERY, g, q), gr_bind(&p, AIEDNN_ARG_KEY, g, k);
    gr_bind(&p, AIEDNN_ARG_VALUE, g, g->ops[m2].in[1]);
    if (mask >= 0) {
        p.has_bias = 1, p.md[4] = gr_expand(&g->t[mask].lt.md, 4);
        gr_bind(&p, AIEDNN_ARG_ATTN_MASK, g, mask);
    }
    if (scale_t >= 0)
        gr_bind(&p, AIEDNN_ARG_ATTR_SCALE, g, scale_t);
    gr_bind(&p, AIEDNN_ARG_DST, g, g->ops[m2].out);
    char impl[160];
    if (!gr_test(&p, impl, sizeof impl))
        return 0;
    if (!gr_sdpa_views(g, &p, ops, &n))
        gr_sdpa_bshd(g, &p, ops, &n), gr_sdpa_dq(g, &p, ops, &n);
    else
        gr_sdpa_bshd(g, &p, ops, &n);
    gr_sdpa_tail(g, &p, ops, &n, 40);
    gr_test(&p, impl, sizeof impl);
    struct aiednn_partition *pt = gr_new_part(g, n, ops);
    if (!pt)
        return 0;
    pt->supported = 1, pt->plan = p;
    char tail[176];
    snprintf(tail, sizeof tail, "-> %s", impl);
    gr_describe(g, pt, 0, tail);
    return 1;
}

/* the input dims of transpose `o` that are not 1, in output order (perm == NULL: input order) */
static int gr_perm_nonunit(const struct aiednn_graph *g, const gr_op *o, const int *perm, int *seq) {
    const aiednn_memory_desc_t *x = &g->t[o->in[0]].lt.md;
    int n = 0;
    for (int i = 0; i < x->ndims; i++) {
        const int d = perm ? perm[i] : i;
        if (d < 0 || d >= x->ndims)
            return -1;
        if (x->dims[d] != 1)
            seq[n++] = d;
    }
    return n;
}

/* a transpose that only moves dims of size 1: the same bytes, a view (batch-1 seq-first attention) */
static int gr_is_unit_perm(const struct aiednn_graph *g, const gr_op *o) {
    int a[AIEDNN_MAX_NDIMS], b[AIEDNN_MAX_NDIMS];
    if (o->kind != aiednn_gop_transpose || o->nin != 1)
        return 0;
    const int n = gr_perm_nonunit(g, o, o->attr.perm, a);
    return n >= 0 && n == gr_perm_nonunit(g, o, NULL, b) && !memcmp(a, b, (size_t)n * sizeof(int));
}

/* the head split / merge around attention: a transpose that moves the bytes like (0, 2, 1, 3) of a 4-D tensor */
static int gr_is_headperm(const struct aiednn_graph *g, const gr_op *o) {
    static const int hp[4] = {0, 2, 1, 3};
    int a[AIEDNN_MAX_NDIMS], b[AIEDNN_MAX_NDIMS];
    if (o->kind != aiednn_gop_transpose || o->nin != 1 || g->t[o->in[0]].lt.md.ndims != 4)
        return 0;
    const int n = gr_perm_nonunit(g, o, o->attr.perm, a);
    return n >= 0 && n == gr_perm_nonunit(g, o, hp, b) && !memcmp(a, b, (size_t)n * sizeof(int));
}

/* the [B, S, H, D] descriptor of a [B, H, S, D] tensor's head-permuted bytes, in `like`'s data type */
static aiednn_memory_desc_t gr_bshd_md(const aiednn_memory_desc_t *bhsd, const aiednn_memory_desc_t *like) {
    aiednn_memory_desc_t m = *bhsd;
    m.dims[1] = bhsd->dims[2], m.dims[2] = bhsd->dims[1];
    m.data_type = like->data_type, m.format = aiednn_format_dense;
    return m;
}

static int gr_in_ops(const int *ops, int n, int op) {
    for (int i = 0; i < n; i++)
        if (ops[i] == op)
            return 1;
    return 0;
}

/* fold the head split (Q, K, V through transpose(0, 2, 1, 3)) and the head merge into the sdpa plan:
 * AIEDNN_SDPA_QKV_BSHD / _DST_BSHD. Only when every one of Q, K, V has it; undone if the create refuses. */
static void gr_sdpa_bshd(const struct aiednn_graph *g, gr_plan *p, int *ops, int *n) {
    char impl[160];
    gr_plan q;
    if (!gr_plan_copy(&q, p))
        return;
    int nn = *n, add[4], na = 0, all = 1;
    const int args[3] = {AIEDNN_ARG_QUERY, AIEDNN_ARG_KEY, AIEDNN_ARG_VALUE};
    for (int a = 0; a < 3 && all; a++) {
        int b = 0;
        while (b < q.nb && q.arg[b] != args[a])
            b++;
        const int t = b < q.nb ? gr_find_tensor(g, q.tid[b]) : -1;
        const int pr = t >= 0 ? g->t[t].producer : -1, u = t >= 0 ? gr_sole_user(g, t) : -1;
        if (pr < 0 || g->ops[pr].part >= 0 || !gr_is_headperm(g, &g->ops[pr]) || u < 0 || !gr_in_ops(ops, nn, u)) {
            all = 0;
            break;
        }
        q.tid[b] = g->t[g->ops[pr].in[0]].lt.id, q.md[a] = gr_bshd_md(&q.md[a], &g->t[g->ops[pr].in[0]].lt.md);
        add[na++] = pr;
    }
    if (all)
        q.causal |= AIEDNN_SDPA_QKV_BSHD;
    else
        q.nb = p->nb, memcpy(q.tid, p->tid, sizeof q.tid), memcpy(q.md, p->md, sizeof q.md), na = 0;
    const int d = gr_find_tensor(g, q.dst_id), m = d >= 0 ? gr_sole_user(g, d) : -1;
    if (m >= 0 && gr_is_headperm(g, &g->ops[m])) {
        q.md[3] = gr_bshd_md(&q.md[3], &g->t[g->ops[m].out].lt.md), q.dst_id = g->t[g->ops[m].out].lt.id;
        for (int b = 0; b < q.nb; b++)
            if (q.arg[b] == AIEDNN_ARG_DST)
                q.tid[b] = q.dst_id;
        q.causal |= AIEDNN_SDPA_DST_BSHD;
        add[na++] = m;
    }
    if (!na || !gr_test(&q, impl, sizeof impl)) {
        gr_plan_free(&q);
        return;
    }
    gr_plan_free(p);
    *p = q;
    for (int i = 0; i < na; i++)
        ops[(*n)++] = add[i];
}

/* read the plan's source `arg` (descriptor *md, slot `slot` of src_scales / src_scales_23) through the
 * per-tensor dequantize of an 8-bit tensor feeding it, directly or through one single-use reshape.
 * Adds the consumed ops to ops[]; 1 if read through. */
static int gr_src_dq(const struct aiednn_graph *g, gr_plan *p, int arg, aiednn_memory_desc_t *md, int slot, int *ops,
                     int *n) {
    int b = 0;
    while (b < p->nb && p->arg[b] != arg)
        b++;
    const int t = b < p->nb ? gr_find_tensor(g, p->tid[b]) : -1;
    if (t < 0 || !g->fuse)
        return 0;
    gr_q q;
    int view = -1;
    if (!gr_dq(g, t, -1, 0, &q)) {
        const int r = g->t[t].producer;
        if (r < 0 || g->ops[r].kind != aiednn_gop_reshape || g->ops[r].part >= 0 || g->ops[r].nin != 1 ||
            gr_sole_user(g, g->ops[r].in[0]) != r || !gr_dq(g, g->ops[r].in[0], -1, 0, &q))
            return 0;
        view = r;
    }
    const aiednn_data_type_t dt = g->t[q.qt].lt.md.data_type;
    md->data_type = dt, md->format = aiednn_format_dense; /* the view's dims, the codes' type (same bytes) */
    p->tid[b] = g->t[q.qt].lt.id;
    if (slot < 2)
        p->attr.src_scales[slot] = q.s, p->attr.src_zero_points[slot] = q.zp;
    else
        p->attr.src_scales_23[slot - 2] = q.s, p->attr.src_zero_points_23[slot - 2] = q.zp;
    p->has_attr = 1;
    gr_add_dq(p, q.op);
    if (view >= 0)
        ops[(*n)++] = view;
    return 1;
}

/* Q, K, V of an sdpa plan read through their dequantizes; kept only if the create accepts it */
static void gr_sdpa_dq(const struct aiednn_graph *g, gr_plan *p, int *ops, int *n) {
    char impl[160];
    gr_plan q;
    if (!g->fuse || !gr_plan_copy(&q, p))
        return;
    int nn = *n, any = 0;
    any |= gr_src_dq(g, &q, AIEDNN_ARG_QUERY, &q.md[0], 0, ops, &nn);
    any |= gr_src_dq(g, &q, AIEDNN_ARG_KEY, &q.md[1], 1, ops, &nn);
    any |= gr_src_dq(g, &q, AIEDNN_ARG_VALUE, &q.md[2], 2, ops, &nn);
    if (!any || !gr_test(&q, impl, sizeof impl)) {
        gr_plan_free(&q);
        return;
    }
    gr_plan_free(p);
    *p = q, *n = nn;
}

/* ── strided views: where each element of a tensor lives in the base tensor a chain of reshape / transpose /
 * slice ops started from (all graph ops are logical row-major tensors, so this composition is exact) ── */
typedef struct {
    int base; /* tensor index */
    int nd;
    int64_t dims[AIEDNN_MAX_NDIMS], str[AIEDNN_MAX_NDIMS], off;
} gr_view;

/* torch's view-compatibility rule: the strides of a reshape of a strided view, 0 if it is not a view */
static int gr_view_reshape(gr_view *v, int nd2, const int64_t *dims2) {
    int64_t ns[AIEDNN_MAX_NDIMS];
    int vd = nd2 - 1;
    int64_t chunk = v->nd ? v->str[v->nd - 1] : 1, tn = 1, vn = 1;
    for (int td = v->nd - 1; td >= 0; td--) {
        tn *= v->dims[td];
        if (td == 0 || (v->dims[td - 1] != 1 && v->str[td - 1] != tn * chunk)) {
            while (vd >= 0 && (vn < tn || dims2[vd] == 1)) {
                ns[vd] = vn * chunk;
                vn *= dims2[vd];
                vd--;
            }
            if (vn != tn)
                return 0;
            if (td > 0)
                chunk = v->str[td - 1], tn = 1, vn = 1;
        }
    }
    while (vd >= 0 && dims2[vd] == 1) /* trailing size-1 view dims */
        ns[vd--] = 1;
    if (vd != -1)
        return 0;
    v->nd = nd2;
    for (int i = 0; i < nd2; i++)
        v->dims[i] = dims2[i], v->str[i] = ns[i];
    return 1;
}

/* the view of tensor t back to its base through unpartitioned reshape / transpose / slice ops; the ops are
 * appended to vops (deduplicated) */
static void gr_view_of(const struct aiednn_graph *g, int t, gr_view *v, int *vops, int *nv, int cap, int depth) {
    const int pr = g->t[t].producer;
    const gr_op *o = pr >= 0 ? &g->ops[pr] : nullptr;
    const int viewop =
        o && o->part < 0 && o->nin == 1 && depth < 16 &&
        (o->kind == aiednn_gop_reshape || o->kind == aiednn_gop_transpose || o->kind == aiednn_gop_slice) &&
        g->t[o->in[0]].lt.md.data_type == g->t[t].lt.md.data_type;
    if (viewop) {
        gr_view u;
        int nv0 = *nv;
        gr_view_of(g, o->in[0], &u, vops, nv, cap, depth + 1);
        const aiednn_memory_desc_t *md = &g->t[t].lt.md;
        int ok = 1;
        if (o->kind == aiednn_gop_transpose) {
            gr_view w = u;
            for (int i = 0; i < md->ndims; i++)
                w.dims[i] = u.dims[o->attr.perm[i]], w.str[i] = u.str[o->attr.perm[i]];
            u = w;
        } else if (o->kind == aiednn_gop_slice) {
            for (int i = 0; i < u.nd; i++)
                u.off += o->attr.starts[i] * u.str[i], u.dims[i] = md->dims[i];
        } else {
            ok = gr_view_reshape(&u, md->ndims, md->dims);
        }
        if (ok) {
            int dup = 0;
            for (int i = 0; i < *nv; i++)
                dup |= vops[i] == pr;
            if (!dup && *nv < cap)
                vops[(*nv)++] = pr;
            *v = u;
            return;
        }
        *nv = nv0; /* not a view after all: t is the base */
    }
    const aiednn_memory_desc_t *md = &g->t[t].lt.md;
    v->base = t, v->nd = md->ndims, v->off = 0;
    int64_t st = 1;
    for (int i = md->ndims - 1; i >= 0; i--)
        v->dims[i] = md->dims[i], v->str[i] = st, st *= md->dims[i];
}

/* does view v (dims [B, H, S, D]) have strides want[] (a size-1 dim's stride does not matter) and offset off */
static int gr_view_is(const gr_view *v, const int64_t *want, int64_t off) {
    if (v->nd != 4 || v->off != off)
        return 0;
    for (int i = 0; i < 4; i++)
        if (v->dims[i] != 1 && v->str[i] != want[i])
            return 0;
    return 1;
}

/* fold the view chains in front of an sdpa plan's Q / K / V: one packed [B, S, 3, H, D] base
 * (AIEDNN_SDPA_QKV_PACKED), or each its own [B, S, H, D] (AIEDNN_SDPA_QKV_BSHD) or dense base; an 8-bit
 * dequantized base is read as codes. 1 if anything folded (kept only if the create accepts it). */
static int gr_sdpa_views(const struct aiednn_graph *g, gr_plan *p, int *ops, int *n) {
    if (!g->fuse)
        return 0;
    const int args[3] = {AIEDNN_ARG_QUERY, AIEDNN_ARG_KEY, AIEDNN_ARG_VALUE};
    gr_view v[3];
    int vops[48], nv = 0, bi[3];
    for (int a = 0; a < 3; a++) {
        bi[a] = 0;
        while (bi[a] < p->nb && p->arg[bi[a]] != args[a])
            bi[a]++;
        const int t = bi[a] < p->nb ? gr_find_tensor(g, p->tid[bi[a]]) : -1;
        if (t < 0)
            return 0;
        gr_view_of(g, t, &v[a], vops, &nv, 48, 0);
    }
    if (!nv)
        return 0;
    const int64_t B = p->md[0].dims[0], H = p->md[0].dims[1], S = p->md[0].dims[2], Dq = p->md[0].dims[3];
    const int64_t packed[4] = {S * 3 * H * Dq, Dq, 3 * H * Dq, 1};
    int mode = 0; /* 1 packed, 2 bshd, 3 dense */
    if (v[0].base == v[1].base && v[1].base == v[2].base && p->md[1].dims[1] == H && p->md[1].dims[2] == S &&
        p->md[2].dims[3] == Dq && gr_numel(&g->t[v[0].base].lt.md) == B * S * 3 * H * Dq &&
        gr_view_is(&v[0], packed, 0) && gr_view_is(&v[1], packed, H * Dq) && gr_view_is(&v[2], packed, 2 * H * Dq))
        mode = 1;
    else {
        int bshd = 1, dense = 1;
        for (int a = 0; a < 3; a++) {
            const aiednn_memory_desc_t *m = &p->md[a];
            const int64_t b[4] = {m->dims[2] * m->dims[1] * m->dims[3], m->dims[3], m->dims[1] * m->dims[3], 1};
            const int64_t d[4] = {m->dims[1] * m->dims[2] * m->dims[3], m->dims[2] * m->dims[3], m->dims[3], 1};
            bshd &= gr_view_is(&v[a], b, 0) && gr_numel(&g->t[v[a].base].lt.md) == gr_numel(m);
            dense &= gr_view_is(&v[a], d, 0) && gr_numel(&g->t[v[a].base].lt.md) == gr_numel(m);
        }
        mode = bshd ? 2 : dense ? 3 : 0;
    }
    if (!mode)
        return 0;
    for (int i = 0; i < nv; i++) { /* every folded op's output feeds only folded ops / the plan's ops */
        const int t = g->ops[vops[i]].out;
        for (int k = 0; k < g->nops; k++)
            for (int x = 0; x < g->ops[k].nin; x++)
                if (g->ops[k].in[x] == t && !gr_in_ops(vops, nv, k) && !gr_in_ops(ops, *n, k))
                    return 0;
        if (g->t[t].marked)
            return 0;
    }
    gr_plan q;
    if (!gr_plan_copy(&q, p))
        return 0;
    int nn = *n;
    for (int a = 0; a < 3; a++) {
        int base = v[a].base;
        gr_q dq;
        int rd = 0;
        if (gr_dq(g, base, -1, 0, &dq)) { /* the codes, when every reader of the dequantize is folded here */
            rd = 1;
            const int dt = g->t[base].lt.md.data_type;
            (void)dt;
            for (int k = 0; k < g->nops && rd; k++)
                for (int x = 0; x < g->ops[k].nin; x++)
                    if (g->ops[k].in[x] == base && !gr_in_ops(vops, nv, k) && !gr_in_ops(ops, *n, k))
                        rd = 0;
        }
        aiednn_memory_desc_t *m = &q.md[a];
        if (mode == 2) { /* [B, S, H, D] in memory */
            const int64_t h = m->dims[1];
            m->dims[1] = m->dims[2], m->dims[2] = h;
        }
        if (rd) {
            base = dq.qt;
            m->data_type = g->t[base].lt.md.data_type;
            if (a < 2)
                q.attr.src_scales[a] = dq.s, q.attr.src_zero_points[a] = dq.zp;
            else
                q.attr.src_scales_23[0] = dq.s, q.attr.src_zero_points_23[0] = dq.zp;
            q.has_attr = 1;
            if (!gr_in_ops(ops, nn, dq.op))
                gr_add_dq(&q, dq.op);
        } else {
            m->data_type = g->t[base].lt.md.data_type;
        }
        m->format = aiednn_format_dense;
        q.tid[bi[a]] = g->t[base].lt.id;
    }
    q.causal |= mode == 1 ? AIEDNN_SDPA_QKV_PACKED : mode == 2 ? AIEDNN_SDPA_QKV_BSHD : 0;
    for (int i = 0; i < nv && nn < 32; i++)
        if (!gr_in_ops(ops, nn, vops[i]))
            ops[nn++] = vops[i];
    char impl[160];
    if (!gr_test(&q, impl, sizeof impl)) {
        gr_plan_free(&q);
        return 0;
    }
    gr_plan_free(p);
    *p = q, *n = nn;
    return 1;
}

static int gr_make_part(struct aiednn_graph *g, gr_plan *p, int *ops, int n) {
    char impl[160];
    if (!gr_test(p, impl, sizeof impl))
        return 0;
    struct aiednn_partition *pt = gr_new_part(g, n, ops);
    if (!pt)
        return 0;
    pt->supported = 1, pt->plan = *p;
    char tail[176];
    snprintf(tail, sizeof tail, "-> %s", impl);
    gr_describe(g, pt, 0, tail);
    return 1;
}

/* a graph sdpa op with its head split / merge transposes folded in (the transposes precede it) */
static int gr_part_sdpa_op(struct aiednn_graph *g, int oi) {
    gr_plan p;
    memset(&p, 0, sizeof p);
    const char *why;
    int ops[40] = {oi}, n = 1;
    if (!g->fuse || g->ops[oi].part >= 0 || !gr_build_single(g, oi, &p, &why)) {
        gr_plan_free(&p);
        return 0;
    }
    if (!gr_sdpa_views(g, &p, ops, &n))
        gr_sdpa_bshd(g, &p, ops, &n), gr_sdpa_dq(g, &p, ops, &n);
    else
        gr_sdpa_bshd(g, &p, ops, &n); /* the dst merge transpose (Q / K / V are folded already) */
    gr_sdpa_tail(g, &p, ops, &n, 40);
    if (n == 1 || !gr_make_part(g, &p, ops, n)) { /* nothing folded: the main pass takes it as usual */
        gr_plan_free(&p);
        return 0;
    }
    return 1;
}

/* level 2: past quantize -> dequantize pairs (the rounding dropped); appends them, returns the tensor */
static int gr_skip_qdq(const struct aiednn_graph *g, int t, int *ops, int *n, int cap) {
    for (;;) {
        const int q = gr_sole_user(g, t);
        if (q < 0 || g->ops[q].kind != aiednn_gop_quantize || g->ops[q].attr.n_scales || *n + 2 > cap)
            return t;
        const int d = gr_sole_user(g, g->ops[q].out);
        if (d < 0 || g->ops[d].kind != aiednn_gop_dequantize || g->ops[d].attr.n_scales)
            return t;
        ops[(*n)++] = q, ops[(*n)++] = d;
        t = g->ops[d].out;
    }
}

/* a constant operand: itself (f32) or a per-tensor dequantize of a constant integer tensor that only this
 * use reads; *k = the tensor to bind, *dq = the dequantize op (or -1) */
static int gr_const_operand(const struct aiednn_graph *g, int t, int *k, int *dq) {
    *dq = -1, *k = t;
    if (g->t[t].lt.property == aiednn_lt_constant && g->t[t].lt.md.data_type == aiednn_f32)
        return 1;
    const int p = g->t[t].producer;
    if (p < 0 || g->ops[p].kind != aiednn_gop_dequantize || g->ops[p].part >= 0 || g->ops[p].attr.n_scales ||
        g->t[t].nuse != 1 || g->t[g->ops[p].in[0]].lt.property != aiednn_lt_constant)
        return 0;
    *k = g->ops[p].in[0], *dq = p;
    return 1;
}

/* level 2: x^2 -> mean(last dim) -> + eps -> rsqrt -> * x -> * weight [-> quantize] as one rms_norm.
 * eps is read at execute (GR_ARG_EPS). Anchored at the mean. */
static int gr_part_rmsnorm(struct aiednn_graph *g, int ri) {
    const gr_op *R = &g->ops[ri];
    if (g->fuse < 2 || R->part >= 0 || R->kind != aiednn_gop_reduction || R->attr.alg != aiednn_reduction_mean ||
        R->nin != 1)
        return 0;
    const int sq = R->in[0], e = g->t[sq].producer;
    const aiednn_memory_desc_t *xm = &g->t[sq].lt.md;
    const int r = xm->ndims;
    if (e < 0 || g->ops[e].part >= 0 || gr_sole_user(g, sq) != ri || g->ops[e].kind != aiednn_gop_eltwise ||
        !(g->ops[e].attr.alg == aiednn_eltwise_square ||
          (g->ops[e].attr.alg == aiednn_eltwise_pow && g->ops[e].attr.alpha == 1.f && g->ops[e].attr.beta == 2.f)) ||
        (R->attr.reduce_mask && R->attr.reduce_mask != 1u << (r - 1)))
        return 0;
    const int x = g->ops[e].in[0];
    int ops[48], n = 0, k, dq;
    ops[n++] = e, ops[n++] = ri;
    int t = gr_skip_qdq(g, R->out, ops, &n, 40);
    const int A = gr_sole_user(g, t); /* + eps */
    if (A < 0 || g->ops[A].kind != aiednn_gop_add || g->ops[A].nin != 2)
        return 0;
    const int eps_t = g->ops[A].in[0] == t ? g->ops[A].in[1] : g->ops[A].in[0];
    int ek, edq;
    if (!gr_const_operand(g, eps_t, &ek, &edq) || gr_numel(&g->t[eps_t].lt.md) != 1)
        return 0;
    ops[n++] = A;
    if (edq >= 0)
        ops[n++] = edq;
    t = gr_skip_qdq(g, g->ops[A].out, ops, &n, 40);
    const int S = gr_sole_user(g, t); /* rsqrt */
    if (S < 0 || g->ops[S].kind != aiednn_gop_eltwise || g->ops[S].attr.alg != aiednn_eltwise_rsqrt)
        return 0;
    ops[n++] = S;
    t = gr_skip_qdq(g, g->ops[S].out, ops, &n, 40);
    const int M1 = gr_sole_user(g, t); /* * x: x itself, or x's quantize -> dequantize (equal parameters) */
    if (M1 < 0 || g->ops[M1].kind != aiednn_gop_multiply || g->ops[M1].nin != 2)
        return 0;
    const int xo = g->ops[M1].in[0] == t ? g->ops[M1].in[1] : g->ops[M1].in[0];
    if (xo != x) { /* x's quantize -> dequantize, or a second dequantize of the codes x dequantizes */
        const int d = g->t[xo].producer, q = d >= 0 ? g->t[g->ops[d].in[0]].producer : -1;
        const int dx = g->t[x].producer;
        const int same_q = q >= 0 && g->ops[q].kind == aiednn_gop_quantize && g->ops[q].in[0] == x;
        const int same_dq = dx >= 0 && g->ops[dx].kind == aiednn_gop_dequantize &&
                            g->ops[dx].in[0] == g->ops[d].in[0] && g->ops[dx].attr.scale == g->ops[d].attr.scale &&
                            g->ops[dx].attr.zero_point == g->ops[d].attr.zero_point && !g->ops[dx].attr.n_scales;
        if (d < 0 || g->ops[d].kind != aiednn_gop_dequantize || g->ops[d].attr.n_scales || g->ops[d].part >= 0 ||
            g->t[xo].nuse != 1 || !(same_q || same_dq))
            return 0;
        ops[n++] = d;
    }
    ops[n++] = M1;
    t = gr_skip_qdq(g, g->ops[M1].out, ops, &n, 40);
    const int M2 = gr_sole_user(g, t); /* * weight [D] */
    if (M2 < 0 || g->ops[M2].kind != aiednn_gop_multiply || g->ops[M2].nin != 2)
        return 0;
    const int w = g->ops[M2].in[0] == t ? g->ops[M2].in[1] : g->ops[M2].in[0];
    if (!gr_const_operand(g, w, &k, &dq) || gr_numel(&g->t[w].lt.md) != xm->dims[r - 1])
        return 0;
    ops[n++] = M2;
    gr_plan p;
    memset(&p, 0, sizeof p);
    p.kind = GP_LNORM, p.eps = 1e-6f, p.flags = AIEDNN_RMS_NORM | AIEDNN_USE_SCALE, p.md[0] = *xm;
    gr_bind(&p, AIEDNN_ARG_SRC, g, x);
    int sn = 0, sops[4];
    if (g->t[x].nuse == 1) /* x only feeds x^2: read its 8-bit codes through the dequantize (src_scales) */
        gr_src_dq(g, &p, AIEDNN_ARG_SRC, &p.md[0], 0, sops, &sn);
    for (int i = 0; i < sn; i++)
        ops[n++] = sops[i];
    if (!gr_bind_prep(&p, g, w, AIEDNN_ARG_SCALE))
        return 0;
    if (dq >= 0)
        ops[n++] = dq;
    p.eps_rt = 1, p.edt = g->t[ek].lt.md.data_type, p.esc = 1.f;
    if (edq >= 0)
        p.esc = g->ops[edq].attr.scale, p.ezp = g->ops[edq].attr.zero_point;
    gr_bind(&p, GR_ARG_EPS, g, ek);
    int out = g->ops[M2].out;
    const int Q = gr_sole_user(g, out);
    if (Q >= 0 && g->ops[Q].kind == aiednn_gop_quantize && !g->ops[Q].attr.n_scales &&
        gr_is_8bit(g->t[g->ops[Q].out].lt.md.data_type)) {
        p.attr.dst_scale = g->ops[Q].attr.scale, p.attr.dst_zero_point = g->ops[Q].attr.zero_point, p.has_attr = 1;
        ops[n++] = Q, out = g->ops[Q].out;
    }
    p.md[3] = g->t[out].lt.md, p.dst_id = g->t[out].lt.id;
    if (p.md[3].ndims != r || gr_numel(&p.md[3]) != gr_numel(xm))
        return gr_plan_free(&p), 0;
    p.md[3].ndims = r;
    memcpy(p.md[3].dims, xm->dims, sizeof xm->dims);
    gr_bind(&p, AIEDNN_ARG_DST, g, out);
    if (!gr_make_part(g, &p, ops, n)) {
        gr_plan_free(&p);
        return 0;
    }
    return 1;
}

/* RoPE (half-split, LLaMA / HF): x * cos + cat(-x[..., h:], x[..., :h]) * sin as one aiednn_rope_create --
 * bit-identical to the mul, mul, add chain. Anchored at the add; every intermediate single-use. */
static int gr_part_rope(struct aiednn_graph *g, int ai) {
    const gr_op *A = &g->ops[ai];
    aiednn_alg_kind_t alg;
    if (!g->fuse || A->part >= 0 || A->nin != 2 || !gr_binary_of(A, &alg) || alg != aiednn_binary_add)
        return 0;
    for (int sw = 0; sw < 2; sw++) {
        const int ma = g->t[A->in[sw]].producer, mr = g->t[A->in[1 - sw]].producer;
        if (ma < 0 || mr < 0 || gr_sole_user(g, A->in[0]) != ai || gr_sole_user(g, A->in[1]) != ai)
            return 0;
        const gr_op *MA = &g->ops[ma], *MR = &g->ops[mr];
        if (!gr_binary_of(MA, &alg) || alg != aiednn_binary_mul || !gr_binary_of(MR, &alg) || alg != aiednn_binary_mul)
            return 0;
        for (int ro = 0; ro < 2; ro++) { /* which operand of MR is the rotated half */
            const int r = MR->in[ro], sin_t = MR->in[1 - ro], c = g->t[r].producer;
            if (c < 0 || g->ops[c].kind != aiednn_gop_concat || g->ops[c].nin != 2 || gr_sole_user(g, r) != mr)
                continue;
            const gr_op *C = &g->ops[c];
            const int rank = g->t[C->out].lt.md.ndims;
            if (!(C->attr.axis == -1 || C->attr.axis == rank - 1))
                continue;
            const int ng = g->t[C->in[0]].producer, s1 = g->t[C->in[1]].producer;
            float fa, fb;
            if (ng < 0 || s1 < 0 || !gr_eltwise_of(&g->ops[ng], &alg, &fa, &fb) || alg != aiednn_eltwise_neg ||
                g->ops[s1].kind != aiednn_gop_slice || gr_sole_user(g, C->in[0]) != c || gr_sole_user(g, C->in[1]) != c)
                continue;
            const int s2 = g->t[g->ops[ng].in[0]].producer;
            if (s2 < 0 || g->ops[s2].kind != aiednn_gop_slice || gr_sole_user(g, g->ops[ng].in[0]) != ng)
                continue;
            const int x = g->ops[s1].in[0];
            const int64_t D = g->t[x].lt.md.dims[rank - 1], h = D / 2;
            const gr_op *S1 = &g->ops[s1], *S2 = &g->ops[s2];
            if (S2->in[0] != x || D % 2 || g->t[S1->out].lt.md.dims[rank - 1] != h ||
                g->t[S2->out].lt.md.dims[rank - 1] != h || S1->attr.starts[rank - 1] != 0 ||
                S2->attr.starts[rank - 1] != h)
                continue;
            int lead0 = 1;
            for (int i = 0; i < rank - 1; i++)
                lead0 &= S1->attr.starts[i] == 0 && S2->attr.starts[i] == 0;
            const int cos_t = MA->in[0] == x ? MA->in[1] : MA->in[1] == x ? MA->in[0] : -1;
            if (!lead0 || cos_t < 0 || g->ops[ma].part >= 0 || g->ops[mr].part >= 0)
                continue;
            gr_plan p;
            memset(&p, 0, sizeof p);
            p.kind = GP_ROPE;
            p.md[0] = g->t[x].lt.md, p.md[1] = gr_expand(&g->t[cos_t].lt.md, rank);
            p.md[2] = gr_expand(&g->t[sin_t].lt.md, rank);
            p.md[3] = g->t[A->out].lt.md, p.dst_id = g->t[A->out].lt.id;
            gr_bind(&p, AIEDNN_ARG_SRC, g, x), gr_bind(&p, AIEDNN_ARG_ROPE_COS, g, cos_t);
            gr_bind(&p, AIEDNN_ARG_ROPE_SIN, g, sin_t), gr_bind(&p, AIEDNN_ARG_DST, g, A->out);
            int ops[7] = {s1, s2, ng, c, ma, mr, ai};
            if (gr_make_part(g, &p, ops, 7))
                return 1;
            gr_plan_free(&p);
        }
    }
    return 0;
}

/* level 2: back through a quantize -> dequantize pair (any parameters, single use) to the value quantized;
 * records both ops. Otherwise t. */
static int gr_thru_qdq(const struct aiednn_graph *g, int t, int *ops, int *n) {
    const int d = g->t[t].producer;
    if (d < 0 || g->ops[d].kind != aiednn_gop_dequantize || g->ops[d].part >= 0 || g->ops[d].attr.n_scales ||
        g->t[t].nuse != 1)
        return t;
    const int qt = g->ops[d].in[0], q = g->t[qt].producer;
    if (q < 0 || g->ops[q].kind != aiednn_gop_quantize || g->ops[q].part >= 0 || g->ops[q].attr.n_scales ||
        gr_sole_user(g, qt) != d)
        return t;
    ops[(*n)++] = d, ops[(*n)++] = q;
    return g->ops[q].in[0];
}

/* a rope source leaf: an 8-bit per-tensor dequantize read as codes (slot of src_scales), else the f32 tensor */
static void gr_rope_leaf(const struct aiednn_graph *g, gr_plan *p, int arg, int mdi, int slot, int t, int *ops,
                         int *n) {
    gr_q q;
    if (gr_dq(g, t, -1, 0, &q) && !q.n && g->t[t].nuse == 1) {
        p->md[mdi] = g->t[q.qt].lt.md;
        if (slot < 2)
            p->attr.src_scales[slot] = q.s, p->attr.src_zero_points[slot] = q.zp;
        else
            p->attr.src_scales_23[0] = q.s, p->attr.src_zero_points_23[0] = q.zp;
        p->has_attr = 1;
        gr_bind(p, arg, g, q.qt);
        ops[(*n)++] = q.op;
        return;
    }
    p->md[mdi] = g->t[t].lt.md;
    gr_bind(p, arg, g, t);
}

/* y = dequantize(quantize(x)) (level 2: the rounding dropped); the quantize op, or -1 */
static int gr_qdq_of(const struct aiednn_graph *g, int x, int y) {
    const int d = g->t[y].producer;
    if (d < 0 || g->ops[d].kind != aiednn_gop_dequantize || g->ops[d].attr.n_scales)
        return -1;
    const int q = g->t[g->ops[d].in[0]].producer;
    return q >= 0 && g->ops[q].kind == aiednn_gop_quantize && g->ops[q].in[0] == x && !g->ops[q].attr.n_scales ? q : -1;
}

/* x and y read the same values: the same tensor, dequantizes of the same codes with equal parameters, or (level
 * 2) one is the other's quantize -> dequantize */
static int gr_same_source(const struct aiednn_graph *g, int x, int y) {
    if (x == y || gr_qdq_of(g, x, y) >= 0 || gr_qdq_of(g, y, x) >= 0)
        return 1;
    const int a = g->t[x].producer, b = g->t[y].producer;
    return a >= 0 && b >= 0 && g->ops[a].kind == aiednn_gop_dequantize && g->ops[b].kind == aiednn_gop_dequantize &&
           g->ops[a].in[0] == g->ops[b].in[0] && !g->ops[a].attr.n_scales && !g->ops[b].attr.n_scales &&
           g->ops[a].attr.scale == g->ops[b].attr.scale && g->ops[a].attr.zero_point == g->ops[b].attr.zero_point;
}

/* level 2: quantized RoPE (every step through quantize -> dequantize, 8-bit x / cos / sin) as one
 * aiednn_rope_create_attr. Anchored at the add. */
static int gr_part_qrope(struct aiednn_graph *g, int ai) {
    const gr_op *A = &g->ops[ai];
    aiednn_alg_kind_t alg;
    float fa, fb;
    if (g->fuse < 2 || A->part >= 0 || A->nin != 2 || !gr_binary_of(A, &alg) || alg != aiednn_binary_add)
        return 0;
    for (int sw = 0; sw < 2; sw++) {
        int ops[48], n = 0;
        ops[n++] = ai;
        const int ma = g->t[gr_thru_qdq(g, A->in[sw], ops, &n)].producer;
        const int mr = g->t[gr_thru_qdq(g, A->in[1 - sw], ops, &n)].producer;
        if (ma < 0 || mr < 0 || !gr_binary_of(&g->ops[ma], &alg) || alg != aiednn_binary_mul ||
            !gr_binary_of(&g->ops[mr], &alg) || alg != aiednn_binary_mul || g->ops[ma].part >= 0 ||
            g->ops[mr].part >= 0)
            continue;
        const int n0 = n;
        for (int ro = 0; ro < 2; ro++) { /* which operand of the second mul is the rotated half */
            n = n0;
            const int c = g->t[gr_thru_qdq(g, g->ops[mr].in[ro], ops, &n)].producer, sin_t = g->ops[mr].in[1 - ro];
            if (c < 0 || g->ops[c].kind != aiednn_gop_concat || g->ops[c].nin != 2 || g->ops[c].part >= 0)
                continue;
            const int rank = g->t[g->ops[c].out].lt.md.ndims;
            if (!(g->ops[c].attr.axis == -1 || g->ops[c].attr.axis == rank - 1))
                continue;
            const int ng = g->t[gr_thru_qdq(g, g->ops[c].in[0], ops, &n)].producer;
            const int s1 = g->t[gr_thru_qdq(g, g->ops[c].in[1], ops, &n)].producer;
            if (ng < 0 || s1 < 0 || !gr_eltwise_of(&g->ops[ng], &alg, &fa, &fb) || alg != aiednn_eltwise_neg ||
                g->ops[s1].kind != aiednn_gop_slice || g->ops[ng].part >= 0 || g->ops[s1].part >= 0)
                continue;
            const int s2 = g->t[gr_thru_qdq(g, g->ops[ng].in[0], ops, &n)].producer;
            if (s2 < 0 || g->ops[s2].kind != aiednn_gop_slice || g->ops[s2].part >= 0)
                continue;
            const gr_op *S1 = &g->ops[s1], *S2 = &g->ops[s2];
            const int x = S1->in[0];
            const int64_t D = g->t[x].lt.md.dims[rank - 1], h = D / 2;
            int lead0 = 1;
            for (int i = 0; i < rank - 1; i++)
                lead0 &= S1->attr.starts[i] == 0 && S2->attr.starts[i] == 0;
            if (!lead0 || D % 2 || !gr_same_source(g, x, S2->in[0]) || g->t[S1->out].lt.md.dims[rank - 1] != h ||
                g->t[S2->out].lt.md.dims[rank - 1] != h || S1->attr.starts[rank - 1] != 0 ||
                S2->attr.starts[rank - 1] != h || (S2->in[0] != x && g->t[S2->in[0]].nuse != 1))
                continue;
            const int xa = gr_same_source(g, g->ops[ma].in[0], x) ? 0 : gr_same_source(g, g->ops[ma].in[1], x) ? 1 : -1;
            if (xa < 0)
                continue;
            const int cos_t = g->ops[ma].in[1 - xa];
            ops[n++] = ma, ops[n++] = mr, ops[n++] = c, ops[n++] = ng, ops[n++] = s1, ops[n++] = s2;
            gr_plan p;
            memset(&p, 0, sizeof p);
            p.kind = GP_ROPE;
            gr_rope_leaf(g, &p, AIEDNN_ARG_SRC, 0, 0, x, ops, &n);
            gr_rope_leaf(g, &p, AIEDNN_ARG_ROPE_COS, 1, 1, cos_t, ops, &n);
            gr_rope_leaf(g, &p, AIEDNN_ARG_ROPE_SIN, 2, 2, sin_t, ops, &n);
            p.md[1] = gr_expand(&p.md[1], rank), p.md[2] = gr_expand(&p.md[2], rank); /* broadcast at src's rank */
            const int others[2] = {g->ops[ma].in[xa], S2->in[0]};
            for (int i = 0; i < 2; i++) { /* x's other readers: a dequantize of the same codes, or x's Q -> DQ */
                const int xx = others[i], d = g->t[xx].producer, q = gr_qdq_of(g, x, xx);
                if (xx != x && d >= 0 && g->ops[d].kind == aiednn_gop_dequantize && g->t[xx].nuse == 1)
                    ops[n++] = d;
                if (xx != x && q >= 0 && gr_sole_user(g, g->ops[q].out) == d)
                    ops[n++] = q;
            }
            int inside = 1; /* every reader of x is in the pattern */
            for (int k = 0; k < g->nops && inside; k++)
                for (int i = 0; i < g->ops[k].nin; i++)
                    if (g->ops[k].in[i] == x) {
                        int in = 0;
                        for (int j = 0; j < n; j++)
                            in |= ops[j] == k;
                        inside &= in;
                    }
            if (!inside || g->t[x].marked) {
                gr_plan_free(&p);
                continue;
            }
            int out = A->out;
            const int Q = gr_sole_user(g, out);
            if (Q >= 0 && g->ops[Q].kind == aiednn_gop_quantize && !g->ops[Q].attr.n_scales &&
                gr_is_8bit(g->t[g->ops[Q].out].lt.md.data_type)) {
                p.attr.dst_scale = g->ops[Q].attr.scale, p.attr.dst_zero_point = g->ops[Q].attr.zero_point;
                p.has_attr = 1, ops[n++] = Q, out = g->ops[Q].out;
            }
            p.md[3] = g->t[out].lt.md, p.dst_id = g->t[out].lt.id;
            gr_bind(&p, AIEDNN_ARG_DST, g, out);
            if (gr_make_part(g, &p, ops, n))
                return 1;
            gr_plan_free(&p);
        }
    }
    return 0;
}

/* gated MLP: matmul(mul(act(matmul(x, Wg)), matmul(x, Wu)), Wd) as one aiednn_gated_mlp_create (level 2:
 * not bit-identical). All three matmuls bias-free with the same weight layout. Anchored at the down projection. */
static int gr_part_gmlp(struct aiednn_graph *g, int md) {
    const gr_op *D = &g->ops[md];
    if (g->fuse < 2 || D->kind != aiednn_gop_matmul || D->nin != 2 || D->part >= 0)
        return 0;
    const int m = g->t[D->in[0]].producer;
    aiednn_alg_kind_t alg;
    if (m < 0 || g->ops[m].nin != 2 || !gr_binary_of(&g->ops[m], &alg) || alg != aiednn_binary_mul ||
        gr_sole_user(g, D->in[0]) != md)
        return 0;
    for (int sw = 0; sw < 2; sw++) {
        const int at = g->ops[m].in[sw], ut = g->ops[m].in[1 - sw];
        const int a = g->t[at].producer, mu = g->t[ut].producer;
        float fa, fb;
        if (a < 0 || mu < 0 || g->ops[a].nin != 1 || !gr_eltwise_of(&g->ops[a], &alg, &fa, &fb) ||
            !(alg == aiednn_eltwise_swish || alg == aiednn_eltwise_gelu_erf || alg == aiednn_eltwise_gelu_tanh) ||
            gr_sole_user(g, at) != m || gr_sole_user(g, ut) != m)
            continue;
        const int mg = g->t[g->ops[a].in[0]].producer;
        if (mg < 0 || gr_sole_user(g, g->ops[a].in[0]) != a)
            continue;
        const gr_op *G = &g->ops[mg], *U = &g->ops[mu];
        if (G->kind != aiednn_gop_matmul || U->kind != aiednn_gop_matmul || G->nin != 2 || U->nin != 2 ||
            G->in[0] != U->in[0] || G->part >= 0 || U->part >= 0 ||
            G->attr.weights_transposed != U->attr.weights_transposed ||
            D->attr.weights_transposed != G->attr.weights_transposed)
            continue;
        gr_plan p;
        memset(&p, 0, sizeof p);
        p.kind = GP_GMLP, p.alg = alg, p.alpha = fa, p.beta = fb, p.wt = G->attr.weights_transposed;
        p.md[0] = g->t[G->in[0]].lt.md, p.md[1] = g->t[G->in[1]].lt.md, p.md[2] = g->t[U->in[1]].lt.md;
        p.md[4] = g->t[D->in[1]].lt.md, p.md[3] = g->t[D->out].lt.md, p.dst_id = g->t[D->out].lt.id;
        gr_bind(&p, AIEDNN_ARG_SRC, g, G->in[0]), gr_bind(&p, AIEDNN_ARG_WEIGHTS, g, G->in[1]);
        gr_bind(&p, AIEDNN_ARG_WEIGHTS_UP, g, U->in[1]), gr_bind(&p, AIEDNN_ARG_WEIGHTS_DOWN, g, D->in[1]);
        gr_bind(&p, AIEDNN_ARG_DST, g, D->out);
        p.const_w = g->t[G->in[1]].lt.property == aiednn_lt_constant;
        int ops[5] = {mg, mu, a, m, md};
        if (gr_make_part(g, &p, ops, 5))
            return 1;
        gr_plan_free(&p);
    }
    return 0;
}

/* reshape elision: a dense reshape partition whose output feeds exactly one supported partition joins it;
 * the consumer binds the reshape's input buffer. */
static void gr_elide_reshapes(struct aiednn_graph *g) {
    for (int i = 0; i < g->nparts; i++) {
        struct aiednn_partition *P = &g->parts[i];
        if (!P->supported || P->plan.kind != GP_COPY || !P->nops)
            continue;
        uint64_t in_id = 0;
        for (int b = 0; b < P->plan.nb; b++)
            if (P->plan.arg[b] == AIEDNN_ARG_SRC)
                in_id = P->plan.tid[b];
        const uint64_t out_id = P->plan.dst_id;
        const int ot = gr_find_tensor(g, out_id), it = gr_find_tensor(g, in_id);
        if (ot < 0 || it < 0 || g->t[ot].marked || g->t[ot].nuse == 0 || g->t[ot].lt.md.format != aiednn_format_dense ||
            g->t[it].lt.md.format != aiednn_format_dense)
            continue;
        int first = -1, ok = 1, nc = 0;
        for (int j = 0; j < g->nparts && ok; j++) {
            const struct aiednn_partition *Q = &g->parts[j];
            if (j == i || !Q->nops)
                continue;
            int binds = 0;
            for (int b = 0; Q->supported && b < Q->plan.nb; b++)
                binds |= Q->plan.tid[b] == out_id && Q->plan.arg[b] != AIEDNN_ARG_DST;
            for (int k = 0; !Q->supported && k < Q->nops; k++) /* the framework reads it: keep the copy */
                for (int x = 0; x < g->ops[Q->ops[k]].nin; x++)
                    ok &= g->ops[Q->ops[k]].in[x] != ot;
            nc += binds;
            if (binds && (first < 0 || g->parts[j].key < g->parts[first].key))
                first = j;
        }
        /* one consuming partition only: no other partition names a tensor without a producer */
        if (!ok || first < 0 || nc != 1)
            continue;
        for (int j = 0; j < g->nparts; j++) { /* rebind: the consumers read the reshape's input */
            struct aiednn_partition *Q = &g->parts[j];
            for (int b = 0; j != i && Q->supported && b < Q->plan.nb; b++)
                if (Q->plan.tid[b] == out_id && Q->plan.arg[b] != AIEDNN_ARG_DST)
                    Q->plan.tid[b] = in_id;
        }
        struct aiednn_partition *F = &g->parts[first];
        int *nops = (int *)realloc(F->ops, (size_t)(F->nops + P->nops + GR_MAX_DQ) * sizeof(int));
        if (!nops)
            continue;
        F->ops = nops;
        for (int k = 0; k < P->nops; k++)
            F->ops[F->nops++] = P->ops[k];
        gr_plan_free(&P->plan);
        P->nops = 0, P->supported = 0; /* removed below */
    }
    int w = 0;
    for (int i = 0; i < g->nparts; i++) {
        if (!g->parts[i].nops) {
            free(g->parts[i].ops);
            continue;
        }
        g->parts[w++] = g->parts[i];
    }
    g->nparts = w;
    for (int i = 0; i < g->nparts; i++)
        for (int j = 0; j < g->parts[i].nops; j++)
            g->ops[g->parts[i].ops[j]].part = i;
}

static int gr_partition(struct aiednn_graph *g) {
    g->parts = (struct aiednn_partition *)calloc((size_t)(g->nops ? g->nops : 1), sizeof *g->parts);
    if (!g->parts)
        return 0;
    for (int k = 0; k < g->nops; k++) { /* patterns whose inner ops precede their anchor in topological order */
        const int oi = g->order[k];
        if (g->ops[oi].part >= 0)
            continue;
        if (g->ops[oi].kind == aiednn_gop_matmul) /* attention: the K transpose / head split come first */
            gr_part_sdpa(g, oi);
        else if (g->ops[oi].kind == aiednn_gop_sdpa)
            gr_part_sdpa_op(g, oi);
    }
    for (int k = 0; k < g->nops; k++) /* level 2: hand-written RMSNorm, anchored at its mean */
        if (g->ops[g->order[k]].part < 0 && g->ops[g->order[k]].kind == aiednn_gop_reduction)
            gr_part_rmsnorm(g, g->order[k]);
    for (int k = g->nops - 1; k >= 0; k--) /* RoPE / gated MLP, anchored at their last op */
        if (g->ops[g->order[k]].part < 0 && !gr_part_rope(g, g->order[k]) && !gr_part_qrope(g, g->order[k]))
            gr_part_gmlp(g, g->order[k]);
    for (int k = 0; k < g->nops; k++) {
        const int oi = g->order[k];
        const aiednn_graph_op_kind_t kind = g->ops[oi].kind;
        if (g->ops[oi].part >= 0 || kind == aiednn_gop_dequantize)
            continue;
        const int anchor = kind == aiednn_gop_convolution || kind == aiednn_gop_matmul ||
                           kind == aiednn_gop_inner_product || kind == aiednn_gop_deconvolution;
        if (!(anchor ? gr_part_anchor(g, oi) : gr_part_single(g, oi)))
            return 0;
    }
    for (int k = 0; k < g->nops; k++)
        if (g->ops[g->order[k]].kind == aiednn_gop_dequantize && g->ops[g->order[k]].part < 0 &&
            !gr_part_dequant(g, g->order[k]))
            return 0;
    /* executable order: by each partition's last op (insertion sort; op -> partition fixed up) */
    for (int i = 1; i < g->nparts; i++)
        for (int j = i; j > 0 && g->parts[j - 1].key > g->parts[j].key; j--) {
            struct aiednn_partition tmp = g->parts[j];
            g->parts[j] = g->parts[j - 1], g->parts[j - 1] = tmp;
        }
    for (int i = 0; i < g->nparts; i++)
        for (int j = 0; j < g->parts[i].nops; j++)
            g->ops[g->parts[i].ops[j]].part = i;
    if (g->fuse)
        gr_elide_reshapes(g);
    if (g->fuse && g->layouts)
        gr_propagate_layouts(g);
    return 1;
}

/* Kahn over op -> op edges; 0 on a cycle */
static int gr_toposort(struct aiednn_graph *g) {
    int *indeg = (int *)calloc((size_t)g->nops + 1, sizeof(int));
    g->order = (int *)malloc(((size_t)g->nops + 1) * sizeof(int));
    if (!indeg || !g->order) {
        free(indeg);
        return 0;
    }
    for (int k = 0; k < g->nops; k++)
        for (int i = 0; i < g->ops[k].nin; i++)
            indeg[k] += g->t[g->ops[k].in[i]].producer >= 0;
    int n = 0;
    for (int k = 0; k < g->nops; k++)
        if (!indeg[k])
            g->order[n++] = k;
    for (int h = 0; h < n; h++) {
        const int t = g->ops[g->order[h]].out;
        for (int k = 0; k < g->nops; k++)
            for (int i = 0; i < g->ops[k].nin; i++)
                if (g->ops[k].in[i] == t && --indeg[k] == 0)
                    g->order[n++] = k;
    }
    free(indeg);
    for (int h = 0; h < n; h++)
        g->ops[g->order[h]].topo = h;
    return n == g->nops;
}

/* ── public API ────────────────────────────────────────────────────────── */

extern "C" void aiednn_graph_op_attr_init(aiednn_graph_op_attr_t *attr) {
    memset(attr, 0, sizeof *attr);
    attr->axis = -1;
    attr->scale = 1.f;
    for (int i = 0; i < AIEDNN_MAX_NDIMS; i++)
        attr->perm[i] = i;
}

extern "C" aiednn_status_t aiednn_graph_create(aiednn_graph_t *graph) {
    if (!graph)
        return aiednn_invalid_arguments;
    *graph = (aiednn_graph_t)calloc(1, sizeof(struct aiednn_graph));
    if (!*graph)
        return aiednn_out_of_memory;
    const char *e = getenv("AIEDNN_GRAPH_NO_FUSE");
    (*graph)->fuse = !(e && e[0] && e[0] != '0');
    const char *l = getenv("AIEDNN_GRAPH_DENSE"); /* A/B: no layout propagation */
    (*graph)->layouts = !(l && l[0] && l[0] != '0');
    return aiednn_success;
}

extern "C" void aiednn_graph_set_layout_propagation(aiednn_graph_t graph, int on) {
    if (graph)
        graph->layouts = on != 0;
}

extern "C" void aiednn_graph_set_fusion(aiednn_graph_t graph, int on) {
    if (graph)
        graph->fuse = on < 0 ? 0 : on;
}

/* tensor index for lt, registered on first sight; -1 if the id is known with another md */
static int gr_tensor_of(struct aiednn_graph *g, const aiednn_logical_tensor_t *lt) {
    const int i = gr_find_tensor(g, lt->id);
    if (i >= 0) {
        const aiednn_memory_desc_t *a = &g->t[i].lt.md;
        if (a->ndims != lt->md.ndims || a->data_type != lt->md.data_type ||
            memcmp(a->dims, lt->md.dims, (size_t)a->ndims * sizeof(int64_t)))
            return -1;
        if (lt->property == aiednn_lt_constant)
            g->t[i].lt.property = aiednn_lt_constant;
        return i;
    }
    if (g->nt == g->cap_t) {
        const int c = g->cap_t ? 2 * g->cap_t : 64;
        gr_tensor *nt = (gr_tensor *)realloc(g->t, (size_t)c * sizeof(gr_tensor));
        if (!nt)
            return -1;
        g->t = nt, g->cap_t = c;
    }
    gr_tensor *t = &g->t[g->nt];
    memset(t, 0, sizeof *t);
    t->lt = *lt, t->producer = -1;
    return g->nt++;
}

extern "C" aiednn_status_t aiednn_graph_add_op(aiednn_graph_t g, uint64_t op_id, aiednn_graph_op_kind_t kind,
                                               const aiednn_graph_op_attr_t *attr, int n_in,
                                               const aiednn_logical_tensor_t *in, int n_out,
                                               const aiednn_logical_tensor_t *out) {
    if (!g || g->finalized || n_out < 1 || n_out > 4 || !out || n_in < 0 || n_in > GR_MAX_IN || (n_in && !in) ||
        (int)kind < 1 || (int)kind >= (int)aiednn_gop_last_)
        return aiednn_invalid_arguments;
    if (g->nops == g->cap_ops) {
        const int c = g->cap_ops ? 2 * g->cap_ops : 64;
        gr_op *no = (gr_op *)realloc(g->ops, (size_t)c * sizeof(gr_op));
        if (!no)
            return aiednn_out_of_memory;
        g->ops = no, g->cap_ops = c;
    }
    gr_op *o = &g->ops[g->nops];
    memset(o, 0, sizeof *o);
    o->id = op_id, o->kind = kind, o->part = -1;
    if (attr)
        o->attr = *attr;
    else
        aiednn_graph_op_attr_init(&o->attr);
    o->attr.scales = nullptr, o->attr.zero_points = nullptr;
    if (attr && attr->n_scales > 0) {
        float *sc = (float *)malloc((size_t)attr->n_scales * sizeof(float));
        int32_t *zp = (int32_t *)calloc((size_t)attr->n_scales, sizeof(int32_t));
        if (!sc || !zp || !attr->scales) {
            free(sc), free(zp);
            return sc && zp ? aiednn_invalid_arguments : aiednn_out_of_memory;
        }
        memcpy(sc, attr->scales, (size_t)attr->n_scales * sizeof(float));
        if (attr->zero_points)
            memcpy(zp, attr->zero_points, (size_t)attr->n_scales * sizeof(int32_t));
        o->attr.scales = sc, o->attr.zero_points = zp;
    }
    o->in = (int *)malloc((size_t)(n_in ? n_in : 1) * sizeof(int));
    if (!o->in)
        return aiednn_out_of_memory;
    for (int i = 0; i < n_in; i++)
        if ((o->in[i] = gr_tensor_of(g, &in[i])) < 0)
            return aiednn_invalid_arguments;
    o->nin = n_in;
    for (int k = 0; k < n_out; k++) {
        const int ot = gr_tensor_of(g, &out[k]);
        if (ot < 0 || g->t[ot].producer >= 0)
            return aiednn_invalid_arguments;
        for (int j = 0; j < k; j++)
            if (o->outs[j] == ot)
                return aiednn_invalid_arguments;
        o->outs[k] = ot;
    }
    for (int i = 0; i < n_in; i++) /* re-read: gr_tensor_of may have moved g->t */
        g->t[o->in[i]].nuse++;
    for (int k = 0; k < n_out; k++)
        g->t[o->outs[k]].producer = g->nops;
    o->nout = n_out, o->out = o->outs[0];
    g->nops++;
    return aiednn_success;
}

extern "C" aiednn_status_t aiednn_graph_mark_output(aiednn_graph_t g, uint64_t tensor_id) {
    const int i = g ? gr_find_tensor(g, tensor_id) : -1;
    if (i < 0 || g->finalized)
        return aiednn_invalid_arguments;
    g->t[i].marked = 1;
    return aiednn_success;
}

extern "C" aiednn_status_t aiednn_graph_finalize(aiednn_graph_t g) {
    if (!g || g->finalized)
        return aiednn_invalid_arguments;
    if (!gr_toposort(g))
        return aiednn_invalid_arguments;
    for (int i = 0; i < g->nops; i++) /* a permute of size-1 dims is a view: reshape folds / elides it */
        if (g->fuse && gr_is_unit_perm(g, &g->ops[i]))
            g->ops[i].kind = aiednn_gop_reshape;
    g->finalized = 1;
    return gr_partition(g) ? aiednn_success : aiednn_out_of_memory;
}

extern "C" int aiednn_graph_get_partitions(aiednn_graph_t g, aiednn_partition_t *out, int cap) {
    if (!g || !g->finalized)
        return 0;
    for (int i = 0; i < g->nparts && i < cap; i++)
        out[i] = &g->parts[i];
    return g->nparts;
}

extern "C" int aiednn_partition_is_supported(aiednn_partition_t p) { return p && p->supported; }

extern "C" int aiednn_partition_get_ops(aiednn_partition_t p, uint64_t *ids, int cap) {
    for (int i = 0; p && i < p->nops && i < cap; i++)
        ids[i] = p->g->ops[p->ops[i]].id;
    return p ? p->nops : 0;
}

/* inputs: every tensor an op of the partition reads that no op of it produces (an unsupported
 * partition), or the plan's bindings except DST (a supported one: read-through tensors included) */
extern "C" int aiednn_partition_get_inputs(aiednn_partition_t p, aiednn_logical_tensor_t *lts, int cap) {
    if (!p)
        return 0;
    const struct aiednn_graph *g = p->g;
    int n = 0;
    uint64_t seen[GR_MAX_BIND];
    if (p->supported) {
        for (int b = 0; b < p->plan.nb; b++) {
            /* softmax backward reads the forward's DST: an input there */
            int is_out =
                (p->plan.arg[b] == AIEDNN_ARG_DST && p->plan.kind != GP_SMB) || p->plan.tid[b] == p->plan.dst_id;
            for (int k = 0; k < p->plan.nxo; k++)
                is_out |= p->plan.tid[b] == p->plan.xo[k];
            if (is_out)
                continue;
            int dup = 0;
            for (int j = 0; j < n && j < GR_MAX_BIND; j++)
                dup |= seen[j] == p->plan.tid[b];
            if (dup)
                continue;
            if (n < GR_MAX_BIND)
                seen[n] = p->plan.tid[b];
            if (n < cap)
                lts[n] = g->t[gr_find_tensor(g, p->plan.tid[b])].lt;
            n++;
        }
        return n;
    }
    for (int i = 0; i < p->nops; i++) {
        const gr_op *o = &g->ops[p->ops[i]];
        for (int k = 0; k < o->nin; k++) {
            const int pr = g->t[o->in[k]].producer;
            if (pr >= 0 && g->ops[pr].part == o->part)
                continue;
            if (n < cap)
                lts[n] = g->t[o->in[k]].lt;
            n++;
        }
    }
    return n;
}

extern "C" int aiednn_partition_get_outputs(aiednn_partition_t p, aiednn_logical_tensor_t *lts, int cap) {
    if (!p)
        return 0;
    const struct aiednn_graph *g = p->g;
    const gr_op *last = &g->ops[p->ops[p->nops - 1]];
    if (!p->supported) { /* every output of its (last) op */
        for (int k = 0; k < last->nout && k < cap; k++)
            lts[k] = g->t[last->outs[k]].lt;
        return last->nout;
    }
    if (cap > 0)
        lts[0] = g->t[gr_find_tensor(g, p->plan.dst_id)].lt;
    for (int k = 0; k < p->plan.nxo && k + 1 < cap; k++)
        lts[k + 1] = g->t[gr_find_tensor(g, p->plan.xo[k])].lt;
    return 1 + p->plan.nxo;
}

extern "C" const char *aiednn_partition_info(aiednn_partition_t p) { return p ? p->info : ""; }

extern "C" aiednn_status_t aiednn_partition_compile(aiednn_partition_t p, aiednn_compiled_partition_t *cp) {
    if (!p || !cp || !p->supported)
        return p && !p->supported ? aiednn_unimplemented : aiednn_invalid_arguments;
    struct aiednn_compiled_partition *c =
        (struct aiednn_compiled_partition *)calloc(1, sizeof(struct aiednn_compiled_partition));
    if (!c || !gr_plan_copy(&c->plan, &p->plan)) {
        free(c);
        return aiednn_out_of_memory;
    }
    const aiednn_status_t st = gr_create(&c->plan, &c->prim);
    if (st != aiednn_success) {
        gr_plan_free(&c->plan);
        free(c);
        return st;
    }
    if (c->prim && c->plan.const_w)
        aiednn_primitive_set_constant_weights(c->prim, 1);
    if (c->plan.bprep && !(c->bias = (float *)malloc((size_t)c->plan.bn * sizeof(float)))) {
        aiednn_compiled_partition_destroy(c);
        return aiednn_out_of_memory;
    }
    snprintf(c->info, sizeof c->info, "%s", c->prim ? aiednn_primitive_impl_info(c->prim) : "copy");
    *cp = c;
    return aiednn_success;
}

extern "C" const char *aiednn_compiled_partition_impl_info(aiednn_compiled_partition_t c) { return c ? c->info : ""; }

extern "C" size_t aiednn_compiled_partition_get_workspace_size(aiednn_compiled_partition_t c) {
    return c && c->prim ? aiednn_primitive_get_workspace_size(c->prim) : 0;
}

extern "C" int aiednn_compiled_partition_find(aiednn_compiled_partition_t c, int mode, aiednn_impl_perf_t *out,
                                              int cap) {
    if (!c)
        return 0;
    if (!c->prim) { /* a reshape: the copy is the only implementation */
        if (cap > 0) {
            memset(out, 0, sizeof *out);
            snprintf(out->impl, sizeof out->impl, "copy");
            out->time_us = -1.f;
        }
        return 1;
    }
    const int n = aiednn_primitive_find(c->prim, mode, out, cap);
    snprintf(c->info, sizeof c->info, "%s", aiednn_primitive_impl_info(c->prim));
    return n;
}

extern "C" aiednn_status_t aiednn_compiled_partition_set_impl(aiednn_compiled_partition_t c, const char *impl) {
    if (!c || !c->prim)
        return aiednn_invalid_arguments;
    const aiednn_status_t st = aiednn_primitive_set_impl(c->prim, impl);
    if (st == aiednn_success)
        snprintf(c->info, sizeof c->info, "%s", aiednn_primitive_impl_info(c->prim));
    return st;
}

static void *gr_lookup(uint64_t id, int n_in, const aiednn_tensor_t *in, int n_out, const aiednn_tensor_t *out) {
    for (int i = 0; i < n_out; i++)
        if (out[i].id == id)
            return out[i].data;
    for (int i = 0; i < n_in; i++)
        if (in[i].id == id)
            return in[i].data;
    return nullptr;
}

extern "C" aiednn_status_t aiednn_compiled_partition_execute(aiednn_compiled_partition_t c, aiednn_stream_t stream,
                                                             int n_in, const aiednn_tensor_t *in, int n_out,
                                                             const aiednn_tensor_t *out, void *workspace,
                                                             size_t workspace_size) {
    if (!c)
        return aiednn_invalid_arguments;
    aiednn_exec_arg_t args[GR_MAX_BIND];
    for (int b = 0; b < c->plan.nb; b++) {
        args[b].arg = c->plan.arg[b];
        args[b].data = gr_lookup(c->plan.tid[b], n_in, in, n_out, out);
        if (!args[b].data)
            return aiednn_invalid_arguments;
    }
    if (c->plan.bprep) /* the dequantized bias, recomputed only when the source can change */
        for (int b = 0; b < c->plan.nb; b++) {
            if (args[b].arg != (c->plan.barg ? c->plan.barg : AIEDNN_ARG_BIAS))
                continue;
            if (!c->plan.bconst || c->bias_from != args[b].data) {
                const gr_plan *p = &c->plan;
                for (int64_t i = 0; i < p->bn; i++) {
                    const int64_t k = p->bnsc == 1 ? 0 : i;
                    const int32_t q = p->bdt == aiednn_s32  ? ((const int32_t *)args[b].data)[i]
                                      : p->bdt == aiednn_s8 ? ((const int8_t *)args[b].data)[i]
                                                            : ((const uint8_t *)args[b].data)[i];
                    c->bias[i] = (float)(q - p->bzp[k]) * p->bsc[k];
                }
                c->bias_from = args[b].data;
            }
            args[b].data = c->bias;
        }
    int na = c->plan.nb;
    if (c->plan.eps_rt) { /* the norm's eps, a graph constant: (re)create when its value changes */
        int k = 0;
        while (k < na && args[k].arg != GR_ARG_EPS)
            k++;
        if (k == na)
            return aiednn_invalid_arguments;
        const void *e = args[k].data;
        const float v = c->plan.edt == aiednn_f32 ? *(const float *)e
                                                  : (float)((c->plan.edt == aiednn_s32  ? *(const int32_t *)e
                                                             : c->plan.edt == aiednn_s8 ? *(const int8_t *)e
                                                                                        : *(const uint8_t *)e) -
                                                            c->plan.ezp) *
                                                        c->plan.esc;
        if (v != c->plan.eps || !c->prim) {
            if (c->prim)
                aiednn_primitive_destroy(c->prim), c->prim = nullptr;
            c->plan.eps = v;
            const aiednn_status_t st = gr_create(&c->plan, &c->prim);
            if (st != aiednn_success)
                return st;
        }
        for (int i = k; i + 1 < na; i++)
            args[i] = args[i + 1];
        na--;
    }
    if (c->plan.sdiv) /* the graph divided the scores by this scalar: sdpa multiplies */
        for (int b = 0; b < c->plan.nb; b++)
            if (args[b].arg == AIEDNN_ARG_ATTR_SCALE) {
                c->sval = 1.f / *(const float *)args[b].data;
                args[b].data = &c->sval;
            }
    if (!c->prim) { /* reshape: a copy */
        memcpy(args[c->plan.nb - 1].data, args[0].data, gr_bytes(&c->plan.md[0]));
        return aiednn_success;
    }
    return aiednn_primitive_execute_async(c->prim, stream, na, args, workspace, workspace_size);
}

extern "C" void aiednn_compiled_partition_destroy(aiednn_compiled_partition_t c) {
    if (!c)
        return;
    if (c->prim)
        aiednn_primitive_destroy(c->prim);
    gr_plan_free(&c->plan);
    free(c->bias);
    free(c);
}

extern "C" void aiednn_graph_destroy(aiednn_graph_t g) {
    if (!g)
        return;
    for (int i = 0; i < g->nparts; i++) {
        gr_plan_free(&g->parts[i].plan);
        free(g->parts[i].ops);
    }
    for (int k = 0; k < g->nops; k++) {
        free(g->ops[k].in);
        free((void *)g->ops[k].attr.scales);
        free((void *)g->ops[k].attr.zero_points);
    }
    free(g->parts), free(g->ops), free(g->t), free(g->order);
    free(g);
}

#endif /* AIEDNN_RT_GRAPH_H */
