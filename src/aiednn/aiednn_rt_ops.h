/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn ref: implementations of the non-GEMM primitives: eltwise, binary,
 * PReLU, reduction, pooling, resampling, LRN, softmax, batch / layer / group
 * normalization (inference), concat, sum, reorder, shuffle and SDPA. All
 * compute in double and store through rt_st (integers: nearest-even, then
 * saturate). Included once by aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_OPS_H
#define AIEDNN_RT_OPS_H

#include "aiednn_rt_internal.h"

// ── eltwise ────────────────────────────────────────────────────────────────

#include "aiednn_rt_phi.h"

static double sigmoid_d(double x) { return 1.0 / (1.0 + exp(-x)); }

static double softplus_d(double x) { return x > 30 ? x : log1p(exp(x)); }

static double eltwise_f(int alg, double x, double a, double b) {
    switch (alg) {
    case aiednn_eltwise_relu:
        return x > 0 ? x : a * x;
    case aiednn_eltwise_elu:
        return x > 0 ? x : a * (exp(x) - 1.0);
    case aiednn_eltwise_gelu_erf: /* x * Phi(x), Phi by polynomials (aiednn_rt_phi.h): no libm erf */
        return x * rt_phi(x);
    case aiednn_eltwise_gelu_tanh:
        return rt_gelu_tanh(x); /* aiednn_rt_geluq.h, as swish */
    case aiednn_eltwise_logistic:
        return sigmoid_d(x);
    case aiednn_eltwise_tanh:
        return tanh(x);
    case aiednn_eltwise_swish:
        return rt_swish(x, a); /* aiednn_rt_geluq.h */
    case aiednn_eltwise_soft_relu:
        return softplus_d(a * x) / a;
    case aiednn_eltwise_mish:
        return x * tanh(softplus_d(x));
    case aiednn_eltwise_hardswish: {
        const double t = a * x + b;
        return x * (t < 0 ? 0 : t > 1 ? 1 : t);
    }
    case aiednn_eltwise_hardsigmoid: {
        const double t = a * x + b;
        return t < 0 ? 0 : t > 1 ? 1 : t;
    }
    case aiednn_eltwise_clip:
        return x < a ? a : x > b ? b : x;
    case aiednn_eltwise_linear:
        return a * x + b;
    case aiednn_eltwise_abs:
        return fabs(x);
    case aiednn_eltwise_exp:
        return exp(x);
    case aiednn_eltwise_log:
        return log(x);
    case aiednn_eltwise_sqrt:
        return sqrt(x);
    case aiednn_eltwise_square:
        return x * x;
    case aiednn_eltwise_pow:
        return a * pow(x, b);
    case aiednn_eltwise_round:
        return nearbyint(x);
    case aiednn_eltwise_neg:
        return -x;
    case aiednn_eltwise_reciprocal:
        return 1.0 / x;
    case aiednn_eltwise_rsqrt:
        return 1.0 / sqrt(x);
    case aiednn_eltwise_erf:
        return erf(x);
    case aiednn_eltwise_floor:
        return floor(x);
    case aiednn_eltwise_ceil:
        return ceil(x);
    case aiednn_eltwise_sin:
        return sin(x);
    case aiednn_eltwise_cos:
        return cos(x);
    case aiednn_eltwise_tan:
        return tan(x);
    case aiednn_eltwise_identity:
        return x;
    case aiednn_eltwise_logical_not:
        return x == 0 ? 1.0 : 0.0;
    case aiednn_eltwise_sign:
        return x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0;
    case aiednn_eltwise_softsign:
        return x / (1.0 + fabs(x));
    case aiednn_eltwise_log_sigmoid: /* -softplus(-x), stable both ways */
        return x < 0 ? x - log1p(exp(x)) : -log1p(exp(-x));
    case aiednn_eltwise_selu:
        return 1.0507009873554804934 * (x > 0 ? x : 1.6732632423543772848 * expm1(x));
    }
    return x;
}

static inline double binq_src(const rt_prob *p, int s, const void *ptr, int64_t off);

/* quantized eltwise: dequantize -> eltwise -> post-ops -> quantize, step for step in its precisions */
static int run_ref_eltwise_q(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *d = &p->md[1];
    const int64_t n = md_numel(d);
    const double lo = d->data_type == aiednn_s8 ? -128.0 : 0.0, hi = lo + 255.0;
    for (int64_t i = 0; i < n; i++) {
        double r = (float)eltwise_f(p->alg, binq_src(p, 0, ptr[0], i), p->alpha, p->beta);
        for (int k = 0; k < p->npo; k++)
            r = (float)eltwise_f(p->po[k].alg, r, p->po[k].alpha, p->po[k].beta);
        if (d->data_type != aiednn_f32) {
            r = nearbyint(r / (double)p->dst_scale) + p->dst_zp;
            r = r != r ? 0.0 : r < lo ? lo : r > hi ? hi : r;
        }
        rt_st(d->data_type, ptr[1], i, r);
    }
    return AIEK_OK;
}

/* 8-bit source: the element function evaluated once on all 256 codes, then looked up (exact) */
static int eltwise_ref_8bit_ok(const rt_prob *p) {
    const aiednn_data_type_t s = p->md[0].data_type, d = p->md[1].data_type;
    return p->kind == aiednn_eltwise && p->prop == RT_FWD_INF && (s == aiednn_s8 || s == aiednn_u8) &&
           (d == aiednn_s8 || d == aiednn_u8 || d == aiednn_f32) && md_numel(&p->md[0]) > 512;
}
static int run_ref_eltwise_table(const rt_prob *p, void *const *ptr) {
    rt_prob t = *p;
    t.md[0].ndims = t.md[1].ndims = 1, t.md[0].dims[0] = t.md[1].dims[0] = 256;
    uint8_t codes[256];
    float tab[256]; /* the largest dst element */
    for (int c = 0; c < 256; c++)
        codes[c] = (uint8_t)c;
    void *tp[2] = {codes, tab};
    const int rc = p->has_attr ? run_ref_eltwise_q(&t, tp) : AIEK_OK;
    if (!p->has_attr)
        for (int c = 0; c < 256; c++)
            rt_st(p->md[1].data_type, tab, c,
                  eltwise_f(p->alg, rt_ld(p->md[0].data_type, codes, c), p->alpha, p->beta));
    if (rc != AIEK_OK)
        return rc;
    const uint8_t *x = (const uint8_t *)ptr[0];
    const int64_t n = md_numel(&p->md[0]);
    if (p->md[1].data_type == aiednn_f32) {
        float *o = (float *)ptr[1];
        for (int64_t i = 0; i < n; i++)
            o[i] = tab[x[i]];
    } else {
        const uint8_t *tb = (const uint8_t *)tab;
        uint8_t *o = (uint8_t *)ptr[1];
        for (int64_t i = 0; i < n; i++)
            o[i] = tb[x[i]];
    }
    return AIEK_OK;
}

static int run_ref_eltwise(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    if (eltwise_ref_8bit_ok(p))
        return run_ref_eltwise_table(p, ptr);
    if (p->has_attr)
        return run_ref_eltwise_q(p, ptr);
    const int64_t n = md_numel(&p->md[0]);
    for (int64_t i = 0; i < n; i++)
        rt_st(p->md[1].data_type, ptr[1], i,
              eltwise_f(p->alg, rt_ld(p->md[0].data_type, ptr[0], i), p->alpha, p->beta));
    return AIEK_OK;
}

// ── binary / prelu ─────────────────────────────────────────────────────────

static double binary_f(int alg, double a, double b) {
    switch (alg) {
    case aiednn_binary_add:
        return a + b;
    case aiednn_binary_sub:
        return a - b;
    case aiednn_binary_mul:
        return a * b;
    case aiednn_binary_div:
        return a / b;
    case aiednn_binary_max:
        return a > b ? a : b;
    case aiednn_binary_min:
        return a < b ? a : b;
    case aiednn_binary_pow:
        return pow(a, b);
    case aiednn_binary_eq:
        return a == b;
    case aiednn_binary_ne:
        return a != b;
    case aiednn_binary_gt:
        return a > b;
    case aiednn_binary_ge:
        return a >= b;
    case aiednn_binary_lt:
        return a < b;
    case aiednn_binary_le:
        return a <= b;
    case aiednn_binary_fmod:
        return fmod(a, b);
    case aiednn_binary_logical_and:
        return a != 0 && b != 0;
    case aiednn_binary_logical_or:
        return a != 0 || b != 0;
    case aiednn_binary_logical_xor:
        return (a != 0) != (b != 0);
    case aiednn_binary_add_square:
        return a + b * b;
    }
    return 0;
}

/* a quantized binary's source element as the dequantize reorder writes it: f32((code - zp) * scale) */
static inline double binq_src(const rt_prob *p, int s, const void *ptr, int64_t off) {
    const aiednn_data_type_t dt = p->md[s].data_type;
    const double v = rt_ld(dt, ptr, off);
    return dt == aiednn_f32 || p->bin_sc[s] == 0 ? v : (double)(float)((v - p->bin_zp[s]) * (double)p->bin_sc[s]);
}

/* the quantized binary's value for dequantized sources x, y: the op in f32, eltwise post-ops in f32, the quantize */
static inline double binq_eval(const rt_prob *p, double x, double y) {
    const aiednn_memory_desc_t *d = &p->md[2];
    double r = p->alg == aiednn_binary_add   ? x + y
               : p->alg == aiednn_binary_sub ? x - y
               : p->alg == aiednn_binary_mul ? x * y
               : p->alg == aiednn_binary_max ? (x > y ? x : y)
                                             : (x < y ? x : y);
    r = (float)r; /* the binary's f32 dst */
    for (int k = 0; k < p->npo; k++)
        r = (float)eltwise_f(p->po[k].alg, r, p->po[k].alpha, p->po[k].beta);
    if (d->data_type != aiednn_f32) { /* the quantize reorder */
        const double lo = d->data_type == aiednn_s8 ? -128.0 : 0.0, hi = lo + 255.0;
        r = nearbyint(r / (double)p->dst_scale) + p->dst_zp;
        r = r != r ? 0.0 : r < lo ? lo : r > hi ? hi : r;
    }
    return r;
}

/* 8-bit a, b and dst: a 64 KB table of binq_eval over every code pair is the op exactly; tables are
 * cached by the quantization parameters across primitives */
#define RT_BINQ_LUTS 64
typedef struct {
    rt_prob key; /* the problem with shapes cleared */
    uint8_t *lut;
} rt_binq_lut;
static rt_binq_lut g_binq_lut[RT_BINQ_LUTS];
static int g_binq_nlut;

static int binq_lut_ok(const rt_prob *p) {
    for (int s = 0; s < 3; s++)
        if (p->md[s].data_type != aiednn_s8 && p->md[s].data_type != aiednn_u8)
            return 0;
    for (int k = 0; k < p->npo; k++)
        if (p->po[k].kind != aiednn_post_op_eltwise)
            return 0;
    return p->alg >= aiednn_binary_add && p->alg <= aiednn_binary_min && p->alg != aiednn_binary_div;
}

static const uint8_t *binq_lut(const rt_prob *p) {
    rt_prob k;
    memset(&k, 0, sizeof k);
    k.alg = p->alg, k.npo = p->npo, k.dst_scale = p->dst_scale, k.dst_zp = p->dst_zp;
    for (int s = 0; s < 3; s++)
        k.md[s].data_type = p->md[s].data_type;
    for (int s = 0; s < 2; s++)
        k.bin_sc[s] = p->bin_sc[s], k.bin_zp[s] = p->bin_zp[s];
    for (int i = 0; i < p->npo; i++)
        k.po[i].kind = p->po[i].kind, k.po[i].alg = p->po[i].alg, k.po[i].alpha = p->po[i].alpha,
        k.po[i].beta = p->po[i].beta;
    for (int i = 0; i < g_binq_nlut; i++)
        if (!memcmp(&g_binq_lut[i].key, &k, sizeof k))
            return g_binq_lut[i].lut;
    if (g_binq_nlut == RT_BINQ_LUTS)
        return nullptr;
    uint8_t *t = (uint8_t *)malloc(65536);
    if (!t)
        return nullptr;
    double dx[256], dy[256];
    for (int c = 0; c < 256; c++) { /* binq_src of every code */
        const uint8_t cb = (uint8_t)c;
        dx[c] = binq_src(p, 0, &cb, 0), dy[c] = binq_src(p, 1, &cb, 0);
    }
    for (int x = 0; x < 256; x++)
        for (int y = 0; y < 256; y++) {
            const double r = binq_eval(p, dx[x], dy[y]);
            t[x * 256 + y] = p->md[2].data_type == aiednn_s8 ? (uint8_t)(int8_t)r : (uint8_t)r;
        }
    g_binq_lut[g_binq_nlut++] = rt_binq_lut{k, t};
    return t;
}

/* the quantized binary (aiednn_binary_create_attr): the unfused chain, step for step, in its precisions */
static int run_ref_binary_q(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *a = &p->md[0], *b = &p->md[1], *d = &p->md[2];
    const int64_t n = md_numel(d);
    const uint8_t *lut = binq_lut_ok(p) ? binq_lut(p) : nullptr;
    if (lut) {
        const uint8_t *xa = (const uint8_t *)ptr[0], *xb = (const uint8_t *)ptr[1];
        uint8_t *o = (uint8_t *)ptr[2];
        if (same_dims(a, d) && same_dims(b, d))
            for (int64_t i = 0; i < n; i++)
                o[i] = lut[xa[i] * 256 + xb[i]];
        else
            for (int64_t i = 0; i < n; i++)
                o[i] = lut[xa[bcast_off(a, d, i)] * 256 + xb[bcast_off(b, d, i)]];
        return AIEK_OK;
    }
    for (int64_t i = 0; i < n; i++) {
        const double x = binq_src(p, 0, ptr[0], bcast_off(a, d, i)), y = binq_src(p, 1, ptr[1], bcast_off(b, d, i));
        rt_st(d->data_type, ptr[2], i, binq_eval(p, x, y));
    }
    return AIEK_OK;
}

static void *g_qres_buf;
static size_t g_qres_cap;

/* a requant_binary primitive's tail: run_ref_binary_q's arithmetic, typed for the common case, else that */
static int qres_tail(const rt_prob *t, void *const *ptr) {
    const aiednn_memory_desc_t *a = &t->md[0], *b = &t->md[1], *d = &t->md[2];
    int typed = same_dims(b, d) && b->data_type != aiednn_f32 && d->data_type != aiednn_f32 &&
                (t->alg == aiednn_binary_add || t->alg == aiednn_binary_sub || t->alg == aiednn_binary_max ||
                 t->alg == aiednn_binary_min || t->alg == aiednn_binary_mul);
    for (int k = 0; k < t->npo; k++)
        typed &= (t->po[k].alg == aiednn_eltwise_relu && t->po[k].alpha == 0) || t->po[k].alg == aiednn_eltwise_clip;
    if (!typed || t->bin_sc[0] == 0 || t->bin_sc[1] == 0)
        return run_ref_binary_q(t, ptr);
    float lx[256], ly[256]; /* the dequantize reorder's f32 for every code */
    for (int c = 0; c < 256; c++) {
        const int va = a->data_type == aiednn_s8 ? (int8_t)c : c, vb = b->data_type == aiednn_s8 ? (int8_t)c : c;
        lx[c] = (float)((va - t->bin_zp[0]) * (double)t->bin_sc[0]);
        ly[c] = (float)((vb - t->bin_zp[1]) * (double)t->bin_sc[1]);
    }
    double plo = -1e300, phi = 1e300; /* the relu / clip run as one clamp: the same function */
    for (int k = 0; k < t->npo; k++) {
        const double lo = t->po[k].alg == aiednn_eltwise_relu ? 0.0 : t->po[k].alpha;
        const double hi = t->po[k].alg == aiednn_eltwise_relu ? 1e300 : t->po[k].beta;
        const double nlo = plo < lo ? lo : plo > hi ? hi : plo, nhi = phi < lo ? lo : phi > hi ? hi : phi;
        plo = nlo, phi = nhi;
    }
    const double tlo = d->data_type == aiednn_s8 ? -128.0 : 0.0, thi = tlo + 255.0, ds = t->dst_scale, zp = t->dst_zp;
    const uint8_t *xa = (const uint8_t *)ptr[0], *xb = (const uint8_t *)ptr[1];
    const int64_t n = md_numel(d);
    for (int64_t i = 0; i < n; i++) {
        const double x = lx[xa[i]], y = ly[xb[i]];
        double r = (float)(t->alg == aiednn_binary_add   ? x + y
                           : t->alg == aiednn_binary_sub ? x - y
                           : t->alg == aiednn_binary_mul ? x * y
                           : t->alg == aiednn_binary_max ? (x > y ? x : y)
                                                         : (x < y ? x : y));
        if (t->npo) /* clips of f32 values to f32 bounds stay f32 (eltwise_f's double, rounded) */
            r = (float)(r < plo ? plo : r > phi ? phi : r);
        const double v = r / ds;
        double q = v >= 1e15 ? thi : v <= -1e15 ? tlo : (v + 6755399441055744.0) - 6755399441055744.0 + zp;
        q = q < tlo ? tlo : q > thi ? thi : q;
        if (d->data_type == aiednn_s8)
            ((int8_t *)ptr[2])[i] = (int8_t)q;
        else
            ((uint8_t *)ptr[2])[i] = (uint8_t)q;
    }
    return AIEK_OK;
}

/* requant_binary tail with 8-bit src1 and dst: a 64 KB table of qres_tail on every code pair, built once
 * per primitive. Applies when src1 is a full tensor like dst. */
static int qres_lut2_ok(const aiednn_primitive *prim) {
    const rt_prob *t = &prim->qtail;
    const aiednn_memory_desc_t *b = &t->md[1], *d = &t->md[2];
    return !(b->data_type == aiednn_f32 || d->data_type == aiednn_f32 || !same_dims(b, d));
}

/* builds prim->qlut2 once: the tail on every (code, src1 code) pair */
static int qres_lut2(aiednn_primitive *prim) {
    const rt_prob *t = &prim->qtail;
    if (!prim->qlut2) {
        uint8_t *tab = (uint8_t *)malloc(65536);
        uint8_t *xa = (uint8_t *)malloc(65536), *xb = (uint8_t *)malloc(65536);
        int rc = tab && xa && xb ? AIEK_OK : AIEK_ERR_ALLOC;
        if (rc == AIEK_OK) {
            for (int i = 0; i < 65536; i++)
                xa[i] = (uint8_t)(i >> 8), xb[i] = (uint8_t)(i & 255);
            rt_prob s = *t; /* the same tail on 65,536 elements: every (code, src1 code) pair */
            for (int k = 0; k < 3; k++)
                s.md[k].ndims = 1, s.md[k].dims[0] = 65536;
            void *pp[3] = {xa, xb, tab};
            rc = qres_tail(&s, pp);
        }
        free(xa), free(xb);
        if (rc != AIEK_OK) {
            free(tab);
            return rc;
        }
        prim->qlut2 = tab;
    }
    return AIEK_OK;
}

static int qres_tail2(aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *t = &prim->qtail;
    const aiednn_memory_desc_t *d = &t->md[2];
    if (!qres_lut2_ok(prim))
        return qres_tail(t, ptr);
    const int rl = qres_lut2(prim);
    if (rl != AIEK_OK)
        return rl;
    const uint8_t *a = (const uint8_t *)ptr[0], *bb = (const uint8_t *)ptr[1], *tab = prim->qlut2;
    uint8_t *y = (uint8_t *)ptr[2];
    const int64_t n = md_numel(d);
    for (int64_t i = 0; i < n; i++)
        y[i] = tab[(unsigned)a[i] << 8 | bb[i]];
    return AIEK_OK;
}

/* requant_eltwise tail: dst[i] = T[code[i]], T = run_ref_eltwise_q on all 256 codes, built once per primitive */
static void qelt_lut(aiednn_primitive *prim) {
    const rt_prob *t = &prim->qtail;
    if (!prim->qlut_ok) {
        rt_prob s = *t;
        int8_t codes[256];
        for (int c = 0; c < 256; c++)
            codes[c] = (int8_t)c;
        s.md[0].ndims = s.md[1].ndims = 1, s.md[0].dims[0] = s.md[1].dims[0] = 256;
        s.md[1].data_type = aiednn_f32; /* the chain's f32 value; the final quantize happens in the lookup below */
        void *pp[2] = {codes, prim->qlut};
        run_ref_eltwise_q(&s, pp);
        prim->qlut_ok = 1;
    }
}

/* the final quantize of each table value, as run_ref_eltwise_q's (8-bit dst) */
static void qelt_lut8(aiednn_primitive *prim, uint8_t *q) {
    const rt_prob *t = &prim->qtail;
    const aiednn_data_type_t dt = t->md[1].data_type;
    qelt_lut(prim);
    const double lo = dt == aiednn_s8 ? -128.0 : 0.0, hi = lo + 255.0;
    for (int c = 0; c < 256; c++) {
        double r = nearbyint((double)prim->qlut[c] / (double)t->dst_scale) + t->dst_zp;
        r = r != r ? 0.0 : r < lo ? lo : r > hi ? hi : r;
        q[c] = dt == aiednn_s8 ? (uint8_t)(int8_t)r : (uint8_t)r;
    }
}

static int qelt_tail(aiednn_primitive *prim, const uint8_t *code, void *dst) {
    const rt_prob *t = &prim->qtail;
    qelt_lut(prim);
    const aiednn_data_type_t dt = t->md[1].data_type;
    const int64_t n = md_numel(&t->md[1]);
    if (dt == aiednn_f32) {
        float *y = (float *)dst;
        for (int64_t i = 0; i < n; i++)
            y[i] = prim->qlut[code[i]];
        return AIEK_OK;
    }
    uint8_t q[256];
    qelt_lut8(prim, q);
    uint8_t *y = (uint8_t *)dst;
    for (int64_t i = 0; i < n; i++)
        y[i] = q[code[i]];
    return AIEK_OK;
}

/* operand m against dst d as element (i / dv) % mm of m: m equal to d, d's trailing dims with leading 1s, or d's
 * leading dims with trailing 1s (same rank); 0 for any other broadcast */
static int bin_bcast_simple(const aiednn_memory_desc_t *m, const aiednn_memory_desc_t *d, int64_t *dv, int64_t *mm) {
    if (m->ndims != d->ndims)
        return 0;
    const int nd_ = d->ndims;
    const int64_t nm = md_numel(m), nd = md_numel(d);
    for (int k = 0; k <= nd_; k++) { /* m = [1 .. 1, d's dims from k]: m repeats every nm outputs */
        int ok = 1;
        for (int t = 0; t < nd_ && ok; t++)
            ok = t < k ? m->dims[t] == 1 : m->dims[t] == d->dims[t];
        if (ok) {
            *dv = 1, *mm = nm;
            return 1;
        }
    }
    for (int k = 0; k <= nd_; k++) { /* m = [d's dims up to k, 1 .. 1]: each m element covers nd / nm outputs */
        int ok = 1;
        for (int t = 0; t < nd_ && ok; t++)
            ok = t < k ? m->dims[t] == d->dims[t] : m->dims[t] == 1;
        if (ok) {
            *dv = nd / nm, *mm = nm;
            return 1;
        }
    }
    return 0;
}

static int run_ref_binary(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob; // slots: SRC, SRC_1, DST, [SRC_2]
    if (p->has_attr)
        return run_ref_binary_q(p, ptr);
    const aiednn_memory_desc_t *a = &p->md[0], *b = &p->md[1], *d = &p->md[2];
    const aiednn_memory_desc_t *c = rt_md(p, AIEDNN_ARG_SRC_2);
    const void *cp = rt_ptr(p, ptr, AIEDNN_ARG_SRC_2);
    const int64_t n = md_numel(d);
    if (p->alg != aiednn_binary_select && a->data_type == aiednn_f32 && b->data_type == aiednn_f32 &&
        d->data_type == aiednn_f32 && same_dims(a, d) && same_dims(b, d)) { // no broadcast: no index math
        const float *x = (const float *)ptr[0], *y = (const float *)ptr[1];
        float *o = (float *)ptr[2];
        if (p->alg == aiednn_binary_add)
            for (int64_t i = 0; i < n; i++)
                o[i] = (float)((double)x[i] + (double)y[i]);
        else
            for (int64_t i = 0; i < n; i++)
                o[i] = (float)binary_f(p->alg, (double)x[i], (double)y[i]);
        return AIEK_OK;
    }
    int64_t adv, amd, bdv, bmd;
    if (p->alg != aiednn_binary_select && a->data_type == aiednn_f32 && b->data_type == aiednn_f32 &&
        d->data_type == aiednn_f32 && bin_bcast_simple(a, d, &adv, &amd) && bin_bcast_simple(b, d, &bdv, &bmd)) {
        /* common broadcasts: the operand element is (i / dv) % m, walked with counters */
        const float *x = (const float *)ptr[0], *y = (const float *)ptr[1];
        float *o = (float *)ptr[2];
        int64_t ia = 0, ca = 0, ib = 0, cb = 0;
        for (int64_t i = 0; i < n; i++) {
            o[i] = (float)binary_f(p->alg, (double)x[ia], (double)y[ib]);
            if (++ca == adv)
                ca = 0, ia = ia + 1 == amd ? 0 : ia + 1;
            if (++cb == bdv)
                cb = 0, ib = ib + 1 == bmd ? 0 : ib + 1;
        }
        return AIEK_OK;
    }
    for (int64_t i = 0; i < n; i++) {
        const double x = rt_ld(a->data_type, ptr[0], bcast_off(a, d, i));
        const double y = rt_ld(b->data_type, ptr[1], bcast_off(b, d, i));
        double v;
        if (p->alg == aiednn_binary_select)
            v = rt_ld(c->data_type, cp, bcast_off(c, d, i)) != 0 ? x : y;
        else
            v = binary_f(p->alg, x, y);
        rt_st(d->data_type, ptr[2], i, v);
    }
    return AIEK_OK;
}

static int run_ref_prelu(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob; // SRC, WEIGHTS, DST
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    const int64_t n = md_numel(&p->md[2]);
    for (int64_t i = 0; i < n; i++) {
        const double x = rt_ld(p->md[0].data_type, ptr[0], i);
        const double w = rt_ld(p->md[1].data_type, ptr[1], bcast_off(&p->md[1], &p->md[2], i));
        rt_st(p->md[2].data_type, ptr[2], i, x > 0 ? x : w * x);
    }
    return AIEK_OK;
}

// ── reduction ──────────────────────────────────────────────────────────────

static int run_ref_reduction(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const int64_t nd = md_numel(d), ns = md_numel(s), per = ns / nd;
    const int alg = p->alg;
    for (int64_t o = 0; o < nd; o++) {
        // iterate the reduced sub-tensor: dst coordinate fixed, reduced axes free
        int64_t dc[AIEDNN_MAX_NDIMS], rem = o;
        for (int i = d->ndims - 1; i >= 0; i--) {
            dc[i] = rem % d->dims[i];
            rem /= d->dims[i];
        }
        double acc = alg == aiednn_reduction_mul || alg == aiednn_reduction_mul_no_zeros ? 1.0
                     : alg == aiednn_reduction_max                                       ? -INFINITY
                     : alg == aiednn_reduction_min                                       ? INFINITY
                                                                                         : 0.0;
        for (int64_t r = 0; r < per; r++) {
            int64_t off = 0, rr = r;
            for (int i = s->ndims - 1, stride = 1; i >= 0; i--) {
                int64_t c = dc[i];
                if (d->dims[i] == 1 && s->dims[i] != 1) {
                    c = rr % s->dims[i];
                    rr /= s->dims[i];
                }
                off += c * stride;
                stride *= (int)s->dims[i];
            }
            const double x = rt_ld(s->data_type, ptr[0], off);
            switch (alg) {
            case aiednn_reduction_sum:
            case aiednn_reduction_mean:
                acc += x;
                break;
            case aiednn_reduction_max:
                acc = x > acc ? x : acc;
                break;
            case aiednn_reduction_min:
                acc = x < acc ? x : acc;
                break;
            case aiednn_reduction_mul:
                acc *= x;
                break;
            case aiednn_reduction_mul_no_zeros:
                acc *= x != 0 ? x : 1.0;
                break;
            case aiednn_reduction_amax:
                acc = fabs(x) > acc ? fabs(x) : acc;
                break;
            default: // lp norms
                acc += pow(fabs(x), p->pnorm);
            }
        }
        if (alg == aiednn_reduction_mean)
            acc /= (double)per;
        else if (alg == aiednn_reduction_norm_lp_power_p_sum)
            acc += p->eps;
        else if (alg == aiednn_reduction_norm_lp_sum)
            acc = pow(acc + p->eps, 1.0 / p->pnorm);
        else if (alg == aiednn_reduction_norm_lp_power_p_max)
            acc = acc > p->eps ? acc : p->eps;
        else if (alg == aiednn_reduction_norm_lp_max)
            acc = pow(acc > p->eps ? acc : p->eps, 1.0 / p->pnorm);
        rt_st(d->data_type, ptr[1], o, acc);
    }
    return AIEK_OK;
}

// ── pooling / resampling / lrn ─────────────────────────────────────────────

static int run_ref_pool3d(const rt_prob *p, void *const *ptr); /* aiednn_rt_nd.h */

static int run_ref_pooling(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    if (p->md[0].ndims == 5)
        return run_ref_pool3d(p, ptr);
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const int64_t NC = s->dims[0] * s->dims[1], H = s->dims[2], W = s->dims[3], Ho = d->dims[2], Wo = d->dims[3];
    const aiednn_pool_params_t *pp = &p->pool;
    for (int64_t nc = 0; nc < NC; nc++)
        for (int64_t oh = 0; oh < Ho; oh++)
            for (int64_t ow = 0; ow < Wo; ow++) {
                double acc = p->alg == aiednn_pooling_max ? -INFINITY : 0.0;
                int64_t cnt = 0;
                for (int64_t kh = 0; kh < pp->kernel[0]; kh++)
                    for (int64_t kw = 0; kw < pp->kernel[1]; kw++) {
                        const int64_t ih = oh * pp->strides[0] - pp->padding_l[0] + kh * (pp->dilates[0] + 1);
                        const int64_t iw = ow * pp->strides[1] - pp->padding_l[1] + kw * (pp->dilates[1] + 1);
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                            continue;
                        const double x = rt_ld(s->data_type, ptr[0], (nc * H + ih) * W + iw);
                        acc = p->alg == aiednn_pooling_max ? (x > acc ? x : acc) : acc + x;
                        cnt++;
                    }
                if (p->alg == aiednn_pooling_avg_include_padding)
                    acc /= (double)(pp->kernel[0] * pp->kernel[1]);
                else if (p->alg == aiednn_pooling_avg_exclude_padding)
                    acc = cnt ? acc / (double)cnt : 0.0;
                rt_st(d->data_type, ptr[1], (nc * Ho + oh) * Wo + ow, acc);
            }
    return AIEK_OK;
}

/* AIEDNN_REF_GENERIC=1: the generic loops only (tests compare the typed f32 paths against them byte for byte) */
static int rt_ref_generic(void) {
    const char *e = getenv("AIEDNN_REF_GENERIC");
    return e && e[0] == '1';
}

/* resampling taps of every output position: up to 2^sp (offset, weight) pairs in run_ref_resampling's
 * corner order, zero weights dropped. cnt[o] taps at tap[o * 8]. */
typedef struct {
    int64_t off;
    double w;
} rs_tap;
static int rs_taps(const rt_prob *p, const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d, rs_tap *tap,
                   uint8_t *cnt) {
    const int sp = s->ndims - 2;
    const int64_t NC = s->dims[0] * s->dims[1], nsp = md_numel(d) / NC;
    for (int64_t o = 0; o < nsp; o++) {
        int64_t oc[3] = {0, 0, 0}, rem = o, lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
        double wh[3] = {0, 0, 0};
        for (int i = sp - 1; i >= 0; i--) {
            oc[i] = rem % d->dims[2 + i];
            rem /= d->dims[2 + i];
        }
        for (int i = 0; i < sp; i++) {
            const double in = (double)s->dims[2 + i], out = (double)d->dims[2 + i];
            const double c = ((double)oc[i] + 0.5) * in / out - 0.5;
            if (p->alg == aiednn_resampling_nearest) {
                int64_t k = (int64_t)floor(((double)oc[i] + 0.5) * in / out);
                lo[i] = hi[i] = k < (int64_t)in - 1 ? k : (int64_t)in - 1;
            } else {
                const double cl = c < 0 ? 0 : c;
                lo[i] = (int64_t)floor(cl);
                hi[i] = lo[i] + 1 < (int64_t)in ? lo[i] + 1 : (int64_t)in - 1;
                wh[i] = cl - (double)lo[i];
            }
        }
        int n = 0;
        for (int corner = 0; corner < (1 << sp); corner++) {
            double w = 1;
            int64_t off = 0;
            for (int i = 0; i < sp; i++) {
                const int up = (corner >> i) & 1;
                w *= up ? wh[i] : 1.0 - wh[i];
                off = off * s->dims[2 + i] + (up ? hi[i] : lo[i]);
            }
            if (w != 0)
                tap[o * 8 + n++] = rs_tap{off, w};
        }
        cnt[o] = (uint8_t)n;
    }
    return AIEK_OK;
}

static void *g_rs_tap, *g_rs_cnt;
static size_t g_rs_tap_cap, g_rs_cnt_cap;

/* forward resampling, src and dst f32: run_ref_resampling's sums (same taps, same order) per plane */
static int resampling_f32(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const int64_t NC = s->dims[0] * s->dims[1], nsp = md_numel(d) / NC, nsrc = md_numel(s) / NC;
    rs_tap *tap = (rs_tap *)scratch(&g_rs_tap, &g_rs_tap_cap, (size_t)nsp * 8 * sizeof(rs_tap));
    uint8_t *cnt = (uint8_t *)scratch(&g_rs_cnt, &g_rs_cnt_cap, (size_t)nsp);
    if (!tap || !cnt)
        return AIEK_ERR_ALLOC;
    rs_taps(p, s, d, tap, cnt);
    const float *x = (const float *)ptr[0];
    float *y = (float *)ptr[1];
    for (int64_t nc = 0; nc < NC; nc++) {
        const float *xp = x + nc * nsrc;
        float *yp = y + nc * nsp;
        for (int64_t o = 0; o < nsp; o++) {
            double v = 0;
            const rs_tap *t = tap + o * 8;
            for (int k = 0; k < cnt[o]; k++)
                v += t[k].w * (double)xp[t[k].off];
            yp[o] = (float)v;
        }
    }
    return AIEK_OK;
}

/* nearest resampling, src and dst of one type: a byte copy through per-axis index maps; a repeated
 * output row is a memcpy */
static int resampling_nearest_copy(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const int sp = s->ndims - 2;
    const int64_t es = dt_bytes(s->data_type), NC = s->dims[0] * s->dims[1];
    int64_t *map[3] = {nullptr, nullptr, nullptr}, din[3] = {1, 1, 1}, dout[3] = {1, 1, 1};
    int64_t nmap = 0;
    for (int i = 0; i < sp; i++)
        din[3 - sp + i] = s->dims[2 + i], dout[3 - sp + i] = d->dims[2 + i], nmap += d->dims[2 + i];
    int64_t *mbuf = (int64_t *)rt_alloc((size_t)(nmap + 3) * sizeof(int64_t));
    if (!mbuf)
        return AIEK_ERR_ALLOC;
    int64_t at = 0;
    for (int i = 0; i < 3; i++) { /* run_ref_resampling's nearest index, per axis */
        map[i] = mbuf + at, at += dout[i] > 1 || din[i] > 1 ? dout[i] : 1;
        for (int64_t o = 0; o < dout[i]; o++) {
            const int64_t k = (int64_t)floor(((double)o + 0.5) * (double)din[i] / (double)dout[i]);
            map[i][o] = k < din[i] - 1 ? k : din[i] - 1;
        }
    }
    const char *src = (const char *)ptr[0];
    char *dst = (char *)ptr[1];
    const int64_t rowb = dout[2] * es;
    for (int64_t nc = 0; nc < NC; nc++)
        for (int64_t a = 0; a < dout[0]; a++)
            for (int64_t b = 0; b < dout[1]; b++) {
                char *o = dst + ((nc * dout[0] + a) * dout[1] + b) * rowb;
                if (b > 0 && map[1][b] == map[1][b - 1]) {
                    memcpy(o, o - rowb, (size_t)rowb);
                    continue;
                }
                const char *in = src + ((nc * din[0] + map[0][a]) * din[1] + map[1][b]) * din[2] * es;
                if (es == 1)
                    for (int64_t c = 0; c < dout[2]; c++)
                        o[c] = in[map[2][c]];
                else
                    for (int64_t c = 0; c < dout[2]; c++)
                        memcpy(o + c * es, in + map[2][c] * es, (size_t)es);
            }
    rt_free(mbuf);
    return AIEK_OK;
}

static int run_ref_resampling(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    if (p->alg == aiednn_resampling_nearest && s->data_type == d->data_type && !dt_is_4bit(s->data_type) &&
        !p->has_attr && !rt_ref_generic())
        return resampling_nearest_copy(p, ptr);
    if (s->data_type == aiednn_f32 && d->data_type == aiednn_f32 && !rt_ref_generic())
        return resampling_f32(p, ptr);
    const int sp = s->ndims - 2; // spatial dims (1..3)
    const int64_t NC = s->dims[0] * s->dims[1], nsp = md_numel(d) / NC, nsrc = md_numel(s) / NC;
    for (int64_t nc = 0; nc < NC; nc++)
        for (int64_t o = 0; o < nsp; o++) {
            int64_t oc[3] = {0, 0, 0}, rem = o;
            for (int i = sp - 1; i >= 0; i--) {
                oc[i] = rem % d->dims[2 + i];
                rem /= d->dims[2 + i];
            }
            int64_t lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
            double wh[3] = {0, 0, 0};
            for (int i = 0; i < sp; i++) {
                const double in = (double)s->dims[2 + i], out = (double)d->dims[2 + i];
                const double c = ((double)oc[i] + 0.5) * in / out - 0.5;
                if (p->alg == aiednn_resampling_nearest) {
                    int64_t k = (int64_t)floor(((double)oc[i] + 0.5) * in / out);
                    lo[i] = hi[i] = k < (int64_t)in - 1 ? k : (int64_t)in - 1;
                } else {
                    const double cl = c < 0 ? 0 : c;
                    lo[i] = (int64_t)floor(cl);
                    hi[i] = lo[i] + 1 < (int64_t)in ? lo[i] + 1 : (int64_t)in - 1;
                    wh[i] = cl - (double)lo[i];
                }
            }
            double v = 0;
            for (int corner = 0; corner < (1 << sp); corner++) {
                double w = 1;
                int64_t off = 0;
                for (int i = 0; i < sp; i++) {
                    const int up = (corner >> i) & 1;
                    w *= up ? wh[i] : 1.0 - wh[i];
                    off = off * s->dims[2 + i] + (up ? hi[i] : lo[i]);
                }
                if (w != 0)
                    v += w * rt_ld(s->data_type, ptr[0], nc * nsrc + off);
            }
            rt_st(d->data_type, ptr[1], nc * nsp + o, v);
        }
    return AIEK_OK;
}

static void *g_lrn_sq;
static size_t g_lrn_sq_cap;

/* LRN forward, f32: run_ref_lrn's window sums in its order; squares from an exact table */
static int lrn_f32(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *s = &p->md[0];
    const int64_t N = s->dims[0], C = s->dims[1], H = s->ndims > 2 ? s->dims[2] : 1, W = s->ndims > 3 ? s->dims[3] : 1;
    const int64_t ls = p->local_size, half = (ls - 1) / 2, n = md_numel(s);
    const int across = p->alg == aiednn_lrn_across_channels;
    const double size = across ? (double)ls : (double)(ls * ls);
    const float *x = (const float *)ptr[0];
    float *y = (float *)ptr[1];
    double *sq = (double *)scratch(&g_lrn_sq, &g_lrn_sq_cap, (size_t)n * sizeof(double));
    if (!sq)
        return AIEK_ERR_ALLOC;
    for (int64_t i = 0; i < n; i++)
        sq[i] = (double)x[i] * (double)x[i];
    const int64_t HW = H * W;
    for (int64_t nn = 0; nn < N; nn++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    double sum = 0;
                    if (across) {
                        const int64_t c0 = c - half < 0 ? 0 : c - half, c1 = c + half < C ? c + half : C - 1;
                        const double *q = sq + (nn * C) * HW + h * W + w;
                        for (int64_t cc = c0; cc <= c1; cc++)
                            sum += q[cc * HW];
                    } else {
                        const int64_t h0 = h - half < 0 ? 0 : h - half, h1 = h + half < H ? h + half : H - 1;
                        const int64_t w0 = w - half < 0 ? 0 : w - half, w1 = w + half < W ? w + half : W - 1;
                        const double *q = sq + (nn * C + c) * HW;
                        for (int64_t hh = h0; hh <= h1; hh++)
                            for (int64_t ww = w0; ww <= w1; ww++)
                                sum += q[hh * W + ww];
                    }
                    const int64_t i = ((nn * C + c) * H + h) * W + w;
                    y[i] = (float)((double)x[i] / pow(p->lrn_k + p->alpha / size * sum, p->beta));
                }
    return AIEK_OK;
}

static int run_ref_lrn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    const aiednn_memory_desc_t *s = &p->md[0];
    if (s->data_type == aiednn_f32 && p->md[1].data_type == aiednn_f32 && !rt_ref_generic())
        return lrn_f32(p, ptr);
    const int64_t N = s->dims[0], C = s->dims[1], H = s->ndims > 2 ? s->dims[2] : 1, W = s->ndims > 3 ? s->dims[3] : 1;
    const int64_t ls = p->local_size, half = (ls - 1) / 2; // window: [c - half, c + half]
    const int across = p->alg == aiednn_lrn_across_channels;
    const double size = across ? (double)ls : (double)(ls * ls);
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    double sum = 0;
                    if (across) {
                        for (int64_t cc = c - half; cc <= c + half; cc++)
                            if (cc >= 0 && cc < C) {
                                const double x = rt_ld(s->data_type, ptr[0], ((n * C + cc) * H + h) * W + w);
                                sum += x * x;
                            }
                    } else {
                        for (int64_t hh = h - half; hh <= h + half; hh++)
                            for (int64_t ww = w - half; ww <= w + half; ww++)
                                if (hh >= 0 && hh < H && ww >= 0 && ww < W) {
                                    const double x = rt_ld(s->data_type, ptr[0], ((n * C + c) * H + hh) * W + ww);
                                    sum += x * x;
                                }
                    }
                    const int64_t i = ((n * C + c) * H + h) * W + w;
                    const double x = rt_ld(s->data_type, ptr[0], i);
                    rt_st(p->md[1].data_type, ptr[1], i, x / pow(p->lrn_k + p->alpha / size * sum, p->beta));
                }
    return AIEK_OK;
}

// ── softmax / normalization ────────────────────────────────────────────────

static int run_ref_softmax(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    const aiednn_memory_desc_t *s = &p->md[0];
    int64_t outer = 1, inner = 1;
    const int64_t A = s->dims[p->axis];
    for (int i = 0; i < p->axis; i++)
        outer *= s->dims[i];
    for (int i = p->axis + 1; i < s->ndims; i++)
        inner *= s->dims[i];
    for (int64_t o = 0; o < outer; o++)
        for (int64_t in = 0; in < inner; in++) {
            const int64_t base = o * A * inner + in;
            double mx = -INFINITY, sum = 0;
            for (int64_t a = 0; a < A; a++) {
                const double x = rt_ld(s->data_type, ptr[0], base + a * inner);
                mx = x > mx ? x : mx;
            }
            for (int64_t a = 0; a < A; a++)
                sum += exp(rt_ld(s->data_type, ptr[0], base + a * inner) - mx);
            const int zero = p->alg == aiednn_softmax_accurate_inf_as_zero && mx == -INFINITY;
            for (int64_t a = 0; a < A; a++) {
                const double z = rt_ld(s->data_type, ptr[0], base + a * inner) - mx;
                rt_st(p->md[1].data_type, ptr[1], base + a * inner,
                      zero                           ? 0.0
                      : p->alg == aiednn_softmax_log ? z - log(sum)
                                                     : exp(z) / sum);
            }
        }
    return AIEK_OK;
}

static double ld_opt(const rt_prob *p, void *const *ptr, int arg, int64_t i, double dflt) {
    const aiednn_memory_desc_t *md = rt_md(p, arg);
    return md ? rt_ld(md->data_type, rt_ptr(p, ptr, arg), i) : dflt;
}

/* ref:'s f32 -> s8 / u8 quantize, sat(nearbyint((double)f / scale) + zp), via f * (1 / scale) unless g lies
 * within 1e-5 of a rounding tie; ties, huge values, inf and NaN take the divide. Exact for every float. */
static inline double rt_quant8(float f, double scale, double inv, double zp, double lo) {
    const double g = (double)f * inv, c = nearbyint(g);
    double r = fabs(g - c) < 0.49999 && fabs(g) < 16777216.0 ? c + zp : nearbyint((double)f / scale) + zp;
    return r != r ? 0.0 : r < lo ? lo : r > lo + 255.0 ? lo + 255.0 : r;
}

static double norm_out(const rt_prob *p, double v) { return (p->flags & AIEDNN_FUSE_NORM_RELU) && v < 0 ? 0 : v; }

static int run_ref_bnorm(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob; // SRC, DST, MEAN, VARIANCE, [SCALE], [SHIFT]
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    const aiednn_memory_desc_t *s = &p->md[0];
    const int64_t N = s->dims[0], C = s->dims[1], sp = md_numel(s) / (N * C);
    /* create_attr (only for f32 -> s8 / u8): the quantizing reorder that followed, folded in */
    const int q = p->has_attr && s->data_type == aiednn_f32 && dt_is_int(p->md[1].data_type);
    for (int64_t c = 0; c < C; c++) {
        const double mean = ld_opt(p, ptr, AIEDNN_ARG_MEAN, c, 0), var = ld_opt(p, ptr, AIEDNN_ARG_VARIANCE, c, 1);
        const double g = ld_opt(p, ptr, AIEDNN_ARG_SCALE, c, 1), b = ld_opt(p, ptr, AIEDNN_ARG_SHIFT, c, 0);
        const double inv = 1.0 / sqrt(var + p->eps);
        if (s->data_type == aiednn_f32 && p->md[1].data_type == aiednn_f32) { /* same ops, no per-element dispatch */
            const int relu = (p->flags & AIEDNN_FUSE_NORM_RELU) != 0;
            for (int64_t n = 0; n < N; n++) {
                const float *x = (const float *)ptr[0] + (n * C + c) * sp;
                float *y = (float *)ptr[1] + (n * C + c) * sp;
                for (int64_t i = 0; i < sp; i++) {
                    const double v = g * ((double)x[i] - mean) * inv + b;
                    y[i] = (float)(relu && v < 0 ? 0 : v);
                }
            }
            continue;
        }
        if (q) { /* folded quantize (create_attr): the f32 batch norm, then the reorder's arithmetic */
            const double ds = p->dst_scale, zp = p->dst_zp, dinv = 1.0 / ds;
            const double lo = p->md[1].data_type == aiednn_s8 ? -128.0 : 0.0;
            const int relu = (p->flags & AIEDNN_FUSE_NORM_RELU) != 0;
            for (int64_t n = 0; n < N; n++) {
                const float *x = (const float *)ptr[0] + (n * C + c) * sp;
                const int64_t o = (n * C + c) * sp;
                for (int64_t i = 0; i < sp; i++) {
                    const double v = g * ((double)x[i] - mean) * inv + b;
                    const float f = (float)(relu && v < 0 ? 0 : v);  /* what the f32 dst held */
                    const double r = rt_quant8(f, ds, dinv, zp, lo); /* reorder_contig, f32 -> s8 / u8 */
                    if (p->md[1].data_type == aiednn_s8)
                        ((int8_t *)ptr[1])[o + i] = (int8_t)r;
                    else
                        ((uint8_t *)ptr[1])[o + i] = (uint8_t)r;
                }
            }
            continue;
        }
        for (int64_t n = 0; n < N; n++)
            for (int64_t i = 0; i < sp; i++) {
                const int64_t idx = (n * C + c) * sp + i;
                rt_st(p->md[1].data_type, ptr[1], idx,
                      norm_out(p, g * (rt_ld(s->data_type, ptr[0], idx) - mean) * inv + b));
            }
    }
    return AIEK_OK;
}

// normalize groups of `len` consecutive elements; scale/shift index = (idx / per_ch) % nch
static void norm_groups(const rt_prob *p, void *const *ptr, int64_t ngroups, int64_t len, int64_t per_ch, int64_t nch) {
    const aiednn_memory_desc_t *s = &p->md[0];
    const int rms = (p->flags & AIEDNN_RMS_NORM) != 0;
    const aiednn_memory_desc_t *gmd = rt_md(p, AIEDNN_ARG_SCALE), *bmd = rt_md(p, AIEDNN_ARG_SHIFT);
    /* create_attr (layer norm f32 -> s8 / u8): the quantizing reorder that followed, folded in */
    const int q = p->has_attr && s->data_type == aiednn_f32 && dt_is_int(p->md[1].data_type);
    if (s->data_type == aiednn_f32 && (p->md[1].data_type == aiednn_f32 || q) && per_ch == 1 && nch == len &&
        (!gmd || gmd->data_type == aiednn_f32) && (!bmd || bmd->data_type == aiednn_f32)) {
        /* f32 in / out, one scale / shift per element of the group: same double ops, lookups hoisted */
        const double ds = p->dst_scale, zp = p->dst_zp, lo = p->md[1].data_type == aiednn_s8 ? -128.0 : 0.0;
        const double dinv = 1.0 / ds;
        const float *gs = gmd ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SCALE) : nullptr;
        const float *bs = bmd ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SHIFT) : nullptr;
        const int relu = (p->flags & AIEDNN_FUSE_NORM_RELU) != 0;
        for (int64_t gi = 0; gi < ngroups; gi++) {
            const float *x = (const float *)ptr[0] + gi * len;
            float *y = q ? nullptr : (float *)ptr[1] + gi * len;
            double mean = 0, var = 0;
            if (!rms) {
                for (int64_t i = 0; i < len; i++)
                    mean += (double)x[i];
                mean /= (double)len;
            }
            for (int64_t i = 0; i < len; i++) {
                const double z = (double)x[i] - mean;
                var += z * z;
            }
            const double inv = 1.0 / sqrt(var / (double)len + p->eps);
            for (int64_t i = 0; i < len; i++) {
                const double g = gs ? (double)gs[i] : 1.0, b = bs ? (double)bs[i] : 0.0;
                const double v = g * (((double)x[i] - mean) * inv) + (rms ? 0 : b);
                const float f = (float)(relu && v < 0 ? 0 : v); /* what the f32 dst held */
                if (!q) {
                    y[i] = f;
                    continue;
                }
                const double r = rt_quant8(f, ds, dinv, zp, lo); /* reorder_contig, f32 -> s8 / u8 */
                if (p->md[1].data_type == aiednn_s8)
                    ((int8_t *)ptr[1])[gi * len + i] = (int8_t)r;
                else
                    ((uint8_t *)ptr[1])[gi * len + i] = (uint8_t)r;
            }
        }
        return;
    }
    for (int64_t gi = 0; gi < ngroups; gi++) {
        const int64_t base = gi * len;
        double mean = 0, var = 0;
        if (!rms) {
            for (int64_t i = 0; i < len; i++)
                mean += rt_ld(s->data_type, ptr[0], base + i);
            mean /= (double)len;
        }
        for (int64_t i = 0; i < len; i++) {
            const double z = rt_ld(s->data_type, ptr[0], base + i) - mean;
            var += z * z;
        }
        const double inv = 1.0 / sqrt(var / (double)len + p->eps);
        for (int64_t i = 0; i < len; i++) {
            const int64_t ch = ((base + i) / per_ch) % nch;
            const double g = ld_opt(p, ptr, AIEDNN_ARG_SCALE, ch, 1), b = ld_opt(p, ptr, AIEDNN_ARG_SHIFT, ch, 0);
            const double v = (rt_ld(s->data_type, ptr[0], base + i) - mean) * inv;
            rt_st(p->md[1].data_type, ptr[1], base + i, norm_out(p, g * v + (rms ? 0 : b)));
        }
    }
}

static int run_ref_lnorm(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    const aiednn_memory_desc_t *s = &p->md[0];
    const int64_t C = s->dims[s->ndims - 1];
    norm_groups(p, ptr, md_numel(s) / C, C, 1, C);
    return AIEK_OK;
}

static int run_ref_gnorm(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    const aiednn_memory_desc_t *s = &p->md[0];
    const int64_t N = s->dims[0], C = s->dims[1], sp = md_numel(s) / (N * C);
    norm_groups(p, ptr, N * p->groups, (C / p->groups) * sp, sp, C);
    return AIEK_OK;
}

// ── data movement ──────────────────────────────────────────────────────────

static int run_ref_concat(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob; // MULTIPLE_SRC + i ..., DST (last)
    const aiednn_memory_desc_t *d = &p->md[p->nsrc];
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < p->axis; i++)
        outer *= d->dims[i];
    for (int i = p->axis + 1; i < d->ndims; i++)
        inner *= d->dims[i];
    const int64_t esz = dt_bytes(d->data_type), dchunk = d->dims[p->axis] * inner * esz;
    int64_t at = 0;
    for (int k = 0; k < p->nsrc; k++) {
        const int64_t schunk = p->md[k].dims[p->axis] * inner * esz;
        if (p->has_attr &&
            !(p->md[k].data_type == d->data_type && p->scales[k] == p->dst_scale && p->zps[k] == p->dst_zp)) {
            /* quantized concat: source k's codes requantized to dst's by a 256-entry table (exact) */
            uint8_t lut[256];
            const int sk = p->md[k].data_type == aiednn_s8;
            const double lo = d->data_type == aiednn_s8 ? -128.0 : 0.0, hi = lo + 255.0;
            for (int c = 0; c < 256; c++) {
                const int code = sk ? (int)(int8_t)c : c;
                const float v = (float)((double)(code - p->zps[k]) * (double)p->scales[k]); /* the f32 dequantize */
                double q = nearbyint((double)v / (double)p->dst_scale) + p->dst_zp;
                q = q < lo ? lo : q > hi ? hi : q;
                lut[c] = d->data_type == aiednn_s8 ? (uint8_t)(int8_t)q : (uint8_t)q;
            }
            for (int64_t o = 0; o < outer; o++) {
                const uint8_t *src = (const uint8_t *)ptr[k] + o * schunk;
                uint8_t *dst = (uint8_t *)ptr[p->nsrc] + o * dchunk + at;
                for (int64_t i = 0; i < schunk; i++)
                    dst[i] = lut[src[i]];
            }
            at += schunk;
            continue;
        }
        for (int64_t o = 0; o < outer; o++)
            memcpy((char *)ptr[p->nsrc] + o * dchunk + at, (const char *)ptr[k] + o * schunk, (size_t)schunk);
        at += schunk;
    }
    return AIEK_OK;
}

static int run_ref_sum(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *d = &p->md[p->nsrc];
    const int64_t n = md_numel(d);
    for (int64_t i = 0; i < n; i++) {
        double v = 0;
        for (int k = 0; k < p->nsrc; k++)
            v += p->scales[k] * rt_ld(p->md[k].data_type, ptr[k], i);
        rt_st(d->data_type, ptr[p->nsrc], i, v);
    }
    return AIEK_OK;
}

/* Identity-permutation reorders as contiguous loops (memcpy, quantize, dequantize), bit-identical to
 * the general loop below. 0 = not handled. */
static int reorder_contig(const rt_prob *p, const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                          void *const *ptr, int64_t n) {
    for (int k = 0; k < d->ndims; k++)
        if (p->perm[k] != k)
            return 0;
    const aiednn_data_type_t sdt = s->data_type, ddt = d->data_type;
    if (sdt == ddt) {
        memcpy(ptr[1], ptr[0], (size_t)(n * dt_bytes(sdt)));
        return 1;
    }
    const double scale = p->scale, zp = p->zero_point;
    if (sdt == aiednn_f32 && (ddt == aiednn_s8 || ddt == aiednn_u8)) {
        const float *x = (const float *)ptr[0];
        const double lo = ddt == aiednn_s8 ? -128.0 : 0.0;
        const double inv = 1.0 / scale;
        for (int64_t i = 0; i < n; i++) {
            const double v = rt_quant8(x[i], scale, inv, zp, lo);
            if (ddt == aiednn_s8)
                ((int8_t *)ptr[1])[i] = (int8_t)v;
            else
                ((uint8_t *)ptr[1])[i] = (uint8_t)v;
        }
        return 1;
    }
    if ((sdt == aiednn_s8 || sdt == aiednn_u8) && ddt == aiednn_f32) {
        float *y = (float *)ptr[1];
        for (int64_t i = 0; i < n; i++) {
            const double v = sdt == aiednn_s8 ? ((const int8_t *)ptr[0])[i] : ((const uint8_t *)ptr[0])[i];
            y[i] = (float)((v - zp) * scale);
        }
        return 1;
    }
    return 0;
}

/* same-type permutation that merges to [B][R][C] -> [B][C][R] (NHWC <-> NCHW): a batched transpose in
 * 32x32 tiles. Returns 0 if not of that form. */
static int reorder_transpose(const rt_prob *p, const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                             void *const *ptr) {
    const int n = s->ndims;
    if (s->data_type != d->data_type || n < 2 || p->perm[0] < 0)
        return 0;
    /* runs of src dims kept adjacent and in order in dst merge into one; want (b) (x) (y) in src
     * order appearing as (b) (y) (x) in dst, b possibly empty */
    int run_of[AIEDNN_MAX_NDIMS], nrun = 0;
    int64_t len[AIEDNN_MAX_NDIMS];
    int dst_runs[AIEDNN_MAX_NDIMS];
    for (int k = 0; k < n; k++) {
        if (k > 0 && p->perm[k] == p->perm[k - 1] + 1) {
            len[nrun - 1] *= s->dims[p->perm[k]];
            continue;
        }
        dst_runs[nrun] = p->perm[k]; /* first src dim of the run */
        len[nrun] = s->dims[p->perm[k]];
        run_of[nrun] = nrun;
        nrun++;
    }
    (void)run_of;
    int64_t B = 1, X, Y;
    if (nrun == 2 && dst_runs[0] > dst_runs[1])
        X = len[1], Y = len[0]; /* dst = (y, x): src = (x, y) */
    else if (nrun == 3 && dst_runs[0] == 0 && dst_runs[1] > dst_runs[2])
        B = len[0], X = len[2], Y = len[1];
    else
        return 0;
    const int64_t esz = dt_bytes(s->data_type);
    if (esz != 1 && esz != 4)
        return 0;
    for (int64_t b = 0; b < B; b++)
        for (int64_t x0 = 0; x0 < X; x0 += 32)
            for (int64_t y0 = 0; y0 < Y; y0 += 32) {
                const int64_t xn = X - x0 < 32 ? X - x0 : 32, yn = Y - y0 < 32 ? Y - y0 : 32;
                if (esz == 4) {
                    const uint32_t *src = (const uint32_t *)ptr[0] + b * X * Y;
                    uint32_t *dst = (uint32_t *)ptr[1] + b * X * Y;
                    for (int64_t y = y0; y < y0 + yn; y++)
                        for (int64_t x = x0; x < x0 + xn; x++)
                            dst[y * X + x] = src[x * Y + y];
                } else {
                    const uint8_t *src = (const uint8_t *)ptr[0] + b * X * Y;
                    uint8_t *dst = (uint8_t *)ptr[1] + b * X * Y;
                    for (int64_t y = y0; y < y0 + yn; y++)
                        for (int64_t x = x0; x < x0 + xn; x++)
                            dst[y * X + x] = src[x * Y + y];
                }
            }
    return 1;
}

/* same-type permute keeping the last axis (e.g. (0, 2, 1, 3)): one memcpy per dst row */
static int reorder_rows(const rt_prob *p, const aiednn_memory_desc_t *s, const aiednn_memory_desc_t *d,
                        void *const *ptr) {
    const int n = d->ndims;
    if (s->data_type != d->data_type || n < 2 || p->perm[n - 1] != n - 1 || dt_is_4bit(s->data_type))
        return 0;
    int64_t sstr[AIEDNN_MAX_NDIMS], st = 1;
    for (int i = n - 1; i >= 0; i--)
        sstr[i] = st, st *= s->dims[i];
    const int64_t W = d->dims[n - 1], rows = md_numel(d) / W, esz = dt_bytes(d->data_type);
    for (int64_t r = 0; r < rows; r++) {
        int64_t rem = r, off = 0;
        for (int k = n - 2; k >= 0; k--) {
            off += (rem % d->dims[k]) * sstr[p->perm[k]];
            rem /= d->dims[k];
        }
        memcpy((char *)ptr[1] + r * W * esz, (const char *)ptr[0] + off * esz, (size_t)(W * esz));
    }
    return 1;
}

static int run_ref_reorder(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    int64_t sstride[AIEDNN_MAX_NDIMS], st = 1;
    for (int i = s->ndims - 1; i >= 0; i--) {
        sstride[i] = st;
        st *= s->dims[i];
    }
    const int64_t n = md_numel(d);
    const int to_int = dt_is_int(d->data_type) && !dt_is_int(s->data_type);
    const int from_int = dt_is_int(s->data_type) && !dt_is_int(d->data_type);
    if (!dt_is_4bit(s->data_type) && !dt_is_4bit(d->data_type) &&
        (reorder_contig(p, s, d, ptr, n) || reorder_transpose(p, s, d, ptr) || reorder_rows(p, s, d, ptr)))
        return AIEK_OK;
    for (int64_t i = 0; i < n; i++) {
        int64_t rem = i, off = 0;
        for (int k = d->ndims - 1; k >= 0; k--) {
            off += (rem % d->dims[k]) * sstride[p->perm[k]];
            rem /= d->dims[k];
        }
        double v = rt_ld(s->data_type, ptr[0], off);
        if (to_int)
            v = nearbyint(v / p->scale) + p->zero_point;
        else if (from_int)
            v = (v - p->zero_point) * p->scale;
        rt_st(d->data_type, ptr[1], i, v);
    }
    return AIEK_OK;
}

static int run_ref_shuffle(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    const aiednn_memory_desc_t *s = &p->md[0];
    int64_t outer = 1, inner = 1;
    const int64_t C = s->dims[p->axis], G = p->group_size, Q = C / G, esz = dt_bytes(s->data_type);
    for (int i = 0; i < p->axis; i++)
        outer *= s->dims[i];
    for (int i = p->axis + 1; i < s->ndims; i++)
        inner *= s->dims[i];
    for (int64_t o = 0; o < outer; o++)
        for (int64_t c = 0; c < C; c++) { // dst channel j*G + i  <-  src channel i*Q + j
            const int64_t i = c % G, j = c / G;
            memcpy((char *)ptr[1] + ((o * C + c) * inner) * esz,
                   (const char *)ptr[0] + ((o * C + i * Q + j) * inner) * esz, (size_t)(inner * esz));
        }
    return AIEK_OK;
}

// ── scaled dot-product attention ───────────────────────────────────────────

static void *g_sdpa;
static size_t g_sdpa_cap;

static int run_sdpa_bwd_ref(const rt_prob *p, void *const *ptr); /* aiednn_rt_sdpa_bwd.h */
static int run_sdpa_var_ref(const rt_prob *p, void *const *ptr); /* aiednn_rt_sdpa_var.h */

static int run_ref_sdpa(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob; // Q, K, V, DST, [MASK]
    if (p->prop == RT_BWD)
        return run_sdpa_bwd_ref(p, ptr);
    if (p->flags & RT_SDPA_VAR)
        return run_sdpa_var_ref(p, ptr);
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2], *d = &p->md[3];
    const aiednn_memory_desc_t *mk = rt_md(p, AIEDNN_ARG_ATTN_MASK);
    const void *mp = rt_ptr(p, ptr, AIEDNN_ARG_ATTN_MASK);
    const int64_t B = d->dims[0], H = d->dims[1], Sq = q->dims[2], Skv = k->dims[2], D = q->dims[3], Dv = v->dims[3];
    /* scores, then one head's K [Skv][D] and V [Skv][Dv] and a Q row, each loaded once; same order */
    double *s = (double *)scratch(&g_sdpa, &g_sdpa_cap, (size_t)(Skv + Skv * D + Skv * Dv + D) * sizeof(double));
    if (!s)
        return AIEK_ERR_ALLOC;
    double *kh = s + Skv, *vh = kh + Skv * D, *qr = vh + Skv * Dv;
    aiednn_memory_desc_t sc; // the [B, H, Sq, Skv] score shape the mask broadcasts to
    memset(&sc, 0, sizeof sc);
    sc.ndims = 4;
    sc.dims[0] = B, sc.dims[1] = H, sc.dims[2] = Sq, sc.dims[3] = Skv;
    /* memory layout: [B,H,S,D], or [B,S,H,D] where p->flags says so (folded head permutes) */
    const int qs = (p->flags & AIEDNN_SDPA_QKV_BSHD) != 0, ds_ = (p->flags & AIEDNN_SDPA_DST_BSHD) != 0;
#define SDPA_AT(bshd, b, h, s, e, S, W)                                                                                \
    ((bshd) ? (((b) * (S) + (s)) * H + (h)) * (W) + (e) : ((((b) * H + (h)) * (S) + (s)) * (W) + (e)))
    /* K / V: Hk heads (GQA / MQA), query head h reads head h / (H / Hk) */
#define SDPA_KV(bshd, b, h, s, e, S, W)                                                                                \
    ((bshd) ? (((b) * (S) + (s)) * Hk + (h) / Hg) * (W) + (e) : ((((b) * Hk + (h) / Hg) * (S) + (s)) * (W) + (e)))
    const int64_t Hk = k->dims[1], Hg = H / Hk;
    for (int64_t bh = 0; bh < B * H; bh++) {
        const int64_t b = bh / H, h = bh - b * H; // create() requires equal [B, H] on Q, K, V, dst
        for (int64_t j = 0; j < Skv; j++) {
            for (int64_t e = 0; e < D; e++)
                kh[j * D + e] = rt_ld(k->data_type, ptr[1], SDPA_KV(qs, b, h, j, e, Skv, D));
            for (int64_t e = 0; e < Dv; e++)
                vh[j * Dv + e] = rt_ld(v->data_type, ptr[2], SDPA_KV(qs, b, h, j, e, Skv, Dv));
        }
        for (int64_t i = 0; i < Sq; i++) {
            for (int64_t e = 0; e < D; e++)
                qr[e] = rt_ld(q->data_type, ptr[0], SDPA_AT(qs, b, h, i, e, Sq, D));
            double mx = -INFINITY;
            for (int64_t j = 0; j < Skv; j++) {
                double acc = 0;
                const double *kj = kh + j * D;
                for (int64_t e = 0; e < D; e++)
                    acc += qr[e] * kj[e];
                acc *= p->scale;
                if (mk)
                    acc += rt_ld(mk->data_type, mp, bcast_off(mk, &sc, (bh * Sq + i) * Skv + j));
                if (p->causal && j > i + (Skv - Sq))
                    acc = -INFINITY;
                s[j] = acc;
                mx = acc > mx ? acc : mx;
            }
            double sum = 0;
            for (int64_t j = 0; j < Skv; j++) {
                s[j] = mx == -INFINITY ? 0 : exp(s[j] - mx);
                sum += s[j];
            }
            for (int64_t e = 0; e < Dv; e++) {
                double acc = 0;
                for (int64_t j = 0; j < Skv; j++)
                    acc += s[j] * vh[j * Dv + e];
                const double o = sum > 0 ? acc / sum : 0;
                if (p->has_attr && dt_is_int(d->data_type)) /* the f32 dst, then reorder's quantize */
                    rt_st(d->data_type, ptr[3], SDPA_AT(ds_, b, h, i, e, Sq, Dv),
                          nearbyint((double)(float)o / p->dst_scale) + p->dst_zp);
                else
                    rt_st(d->data_type, ptr[3], SDPA_AT(ds_, b, h, i, e, Sq, Dv), o);
            }
        }
    }
#undef SDPA_AT
#undef SDPA_KV
    return AIEK_OK;
}

#endif /* AIEDNN_RT_OPS_H */
