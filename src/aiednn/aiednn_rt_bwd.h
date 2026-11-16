/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_bwd.h -- training: backward passes, and the training forward of the normalizations.
 *
 * A training primitive has its forward's kind and rt_prob.prop != RT_FWD_INF; dispatch gives it to
 * ref: only. Gradients accumulate in double, rounded once on store. Max pooling routes to the window's
 * first maximum; norms use the biased variance; RMS norm drops the mean terms.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_ops.h and the create helpers of
 * aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_BWD_H
#define AIEDNN_RT_BWD_H

/* ── geometry: up to 3 spatial dims, front-padded to 3 (missing dims are 1) ── */
typedef struct {
    int64_t N, C, O, G, Cg, Og;
    int64_t xs[3], ys[3], k[3], s[3], pl[3], dil[3]; /* x: the conv's src side, y: its dst side */
} bwd_geo;

static void bwd_dims3(const aiednn_memory_desc_t *m, int64_t *o3) {
    const int sp = m->ndims - 2;
    for (int i = 0; i < 3; i++)
        o3[i] = i < 3 - sp ? 1 : m->dims[2 + i - (3 - sp)];
}

static void bwd_conv_geo(const rt_prob *p, const aiednn_memory_desc_t *x, const aiednn_memory_desc_t *w,
                         const aiednn_memory_desc_t *y, bwd_geo *g) {
    const int sp = x->ndims - 2;
    g->N = x->dims[0], g->C = x->dims[1], g->O = y->dims[1], g->G = p->cnd.groups;
    g->Cg = g->C / g->G, g->Og = g->O / g->G;
    bwd_dims3(x, g->xs), bwd_dims3(y, g->ys), bwd_dims3(w, g->k);
    for (int i = 0; i < 3; i++) {
        const int j = i - (3 - sp); /* index into the N-D params */
        g->s[i] = j < 0 ? 1 : p->cnd.strides[j];
        g->pl[i] = j < 0 ? 0 : p->cnd.padding_l[j];
        g->dil[i] = j < 0 ? 1 : p->cnd.dilates[j] + 1;
    }
}

/* the (de)convolution's gradients: conv relates dst position a and src position a*s - pl + k*dil;
 * deconv the other way round. which 2: data (DIFF_SRC), 3: weights (DIFF_WEIGHTS [, DIFF_BIAS]). */
static int run_bwd_conv(const rt_prob *p, void *const *ptr) {
    const int deconv = p->kind == aiednn_deconvolution, data = p->prop == RT_BWD_DATA;
    const aiednn_memory_desc_t *x = rt_md(p, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC);
    const aiednn_memory_desc_t *w = rt_md(p, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
    const aiednn_memory_desc_t *y = rt_md(p, AIEDNN_ARG_DIFF_DST), *db = rt_md(p, AIEDNN_ARG_DIFF_BIAS);
    void *px = rt_ptr(p, ptr, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC);
    void *pw = rt_ptr(p, ptr, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
    void *py = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    bwd_geo g;
    bwd_conv_geo(p, x, w, y, &g);
    const int64_t nout = data ? md_numel(x) : md_numel(w);
    double *acc = (double *)rt_calloc((size_t)nout + (size_t)g.O, sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    double *bacc = acc + nout;
    const int64_t *as = deconv ? g.xs : g.ys, *bs = deconv ? g.ys : g.xs; /* a: iterated, b: derived */
    const int64_t nk = g.k[0] * g.k[1] * g.k[2], na = as[0] * as[1] * as[2];
    const int64_t nx = g.xs[0] * g.xs[1] * g.xs[2], ny = g.ys[0] * g.ys[1] * g.ys[2];
    for (int64_t n = 0; n < g.N; n++)
        for (int64_t o = 0; o < g.O; o++) {
            const int64_t grp = o / g.Og;
            if (!data && db) /* the bias gradient sums dst */
                for (int64_t i = 0; i < ny; i++)
                    bacc[o] += rt_ld(y->data_type, py, (n * g.O + o) * ny + i);
            for (int64_t c = 0; c < g.Cg; c++) {
                const int64_t ci = grp * g.Cg + c;
                for (int64_t a = 0; a < na; a++) {
                    const int64_t a0 = a / (as[1] * as[2]), a1 = a / as[2] % as[1], a2 = a % as[2];
                    for (int64_t kk = 0; kk < nk; kk++) {
                        const int64_t k0 = kk / (g.k[1] * g.k[2]), k1 = kk / g.k[2] % g.k[1], k2 = kk % g.k[2];
                        const int64_t b0 = a0 * g.s[0] - g.pl[0] + k0 * g.dil[0];
                        const int64_t b1 = a1 * g.s[1] - g.pl[1] + k1 * g.dil[1];
                        const int64_t b2 = a2 * g.s[2] - g.pl[2] + k2 * g.dil[2];
                        if (b0 < 0 || b0 >= bs[0] || b1 < 0 || b1 >= bs[1] || b2 < 0 || b2 >= bs[2])
                            continue;
                        const int64_t b = (b0 * bs[1] + b1) * bs[2] + b2;
                        const int64_t xi = (n * g.C + ci) * nx + (deconv ? a : b);
                        const int64_t yi = (n * g.O + o) * ny + (deconv ? b : a);
                        const int64_t wi = (o * g.Cg + c) * nk + kk;
                        const double dy = rt_ld(y->data_type, py, yi);
                        if (data)
                            acc[xi] += dy * rt_ld(w->data_type, pw, wi);
                        else
                            acc[wi] += dy * rt_ld(x->data_type, px, xi);
                    }
                }
            }
        }
    void *out = data ? px : pw;
    const aiednn_data_type_t odt = data ? x->data_type : w->data_type;
    for (int64_t i = 0; i < nout; i++)
        rt_st(odt, out, i, acc[i]);
    if (!data && db)
        for (int64_t o = 0; o < g.O; o++)
            rt_st(db->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_BIAS), o, bacc[o]);
    rt_free(acc);
    return AIEK_OK;
}

/* inner product: x [N, K] (flattened), W [O, K], y [N, O] */
static int run_bwd_ip(const rt_prob *p, void *const *ptr) {
    const int data = p->prop == RT_BWD_DATA;
    const aiednn_memory_desc_t *x = rt_md(p, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC);
    const aiednn_memory_desc_t *w = rt_md(p, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
    const aiednn_memory_desc_t *y = rt_md(p, AIEDNN_ARG_DIFF_DST), *db = rt_md(p, AIEDNN_ARG_DIFF_BIAS);
    void *px = rt_ptr(p, ptr, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC);
    void *pw = rt_ptr(p, ptr, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
    void *py = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    const int64_t N = y->dims[0], O = y->dims[1], K = md_numel(x) / N;
    if (data) {
        for (int64_t n = 0; n < N; n++)
            for (int64_t k = 0; k < K; k++) {
                double a = 0;
                for (int64_t o = 0; o < O; o++)
                    a += rt_ld(y->data_type, py, n * O + o) * rt_ld(w->data_type, pw, o * K + k);
                rt_st(x->data_type, px, n * K + k, a);
            }
        return AIEK_OK;
    }
    for (int64_t o = 0; o < O; o++) {
        for (int64_t k = 0; k < K; k++) {
            double a = 0;
            for (int64_t n = 0; n < N; n++)
                a += rt_ld(y->data_type, py, n * O + o) * rt_ld(x->data_type, px, n * K + k);
            rt_st(w->data_type, pw, o * K + k, a);
        }
        if (db) {
            double a = 0;
            for (int64_t n = 0; n < N; n++)
                a += rt_ld(y->data_type, py, n * O + o);
            rt_st(db->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_BIAS), o, a);
        }
    }
    return AIEK_OK;
}

/* f'(x) of eltwise_f */
static double eltwise_df(int alg, double x, double a, double b) {
    const double pi = 3.14159265358979323846;
    switch (alg) {
    case aiednn_eltwise_relu:
        return x > 0 ? 1.0 : a;
    case aiednn_eltwise_elu:
        return x > 0 ? 1.0 : a * exp(x);
    case aiednn_eltwise_gelu_erf:
        return rt_phi(x) + x * exp(-0.5 * x * x) / sqrt(2 * pi);
    case aiednn_eltwise_gelu_tanh: {
        const double k = 0.79788456080286535588, u = k * (x + 0.044715 * x * x * x), t = tanh(u);
        return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * k * (1 + 3 * 0.044715 * x * x);
    }
    case aiednn_eltwise_logistic: {
        const double s = 1 / (1 + exp(-x));
        return s * (1 - s);
    }
    case aiednn_eltwise_tanh: {
        const double t = tanh(x);
        return 1 - t * t;
    }
    case aiednn_eltwise_swish: {
        const double s = 1 / (1 + exp(-a * x));
        return s + a * x * s * (1 - s);
    }
    case aiednn_eltwise_soft_relu:
        return 1 / (1 + exp(-a * x));
    case aiednn_eltwise_mish: {
        const double sp = x > 20 ? x : log1p(exp(x)), t = tanh(sp);
        return t + x * (1 - t * t) / (1 + exp(-x));
    }
    case aiednn_eltwise_hardswish: {
        const double t = a * x + b;
        return t <= 0 ? 0 : t >= 1 ? 1 : t + a * x;
    }
    case aiednn_eltwise_hardsigmoid: {
        const double t = a * x + b;
        return t <= 0 || t >= 1 ? 0 : a;
    }
    case aiednn_eltwise_clip:
        return x >= a && x <= b ? 1.0 : 0.0;
    case aiednn_eltwise_linear:
        return a;
    case aiednn_eltwise_abs:
        return x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0;
    case aiednn_eltwise_exp:
        return exp(x);
    case aiednn_eltwise_log:
        return 1 / x;
    case aiednn_eltwise_sqrt:
        return 0.5 / sqrt(x);
    case aiednn_eltwise_square:
        return 2 * x;
    case aiednn_eltwise_pow:
        return a * b * pow(x, b - 1);
    case aiednn_eltwise_neg:
        return -1.0;
    case aiednn_eltwise_reciprocal:
        return -1 / (x * x);
    case aiednn_eltwise_rsqrt:
        return -0.5 / (x * sqrt(x));
    case aiednn_eltwise_erf:
        return 2 / sqrt(pi) * exp(-x * x);
    case aiednn_eltwise_sin:
        return cos(x);
    case aiednn_eltwise_cos:
        return -sin(x);
    case aiednn_eltwise_tan: {
        const double t = tan(x);
        return 1 + t * t;
    }
    case aiednn_eltwise_identity:
        return 1.0;
    case aiednn_eltwise_softsign: {
        const double d = 1 + fabs(x);
        return 1 / (d * d);
    }
    case aiednn_eltwise_log_sigmoid:
        return 1 / (1 + exp(x));
    case aiednn_eltwise_selu:
        return 1.0507009873554804934 * (x > 0 ? 1.0 : 1.6732632423543772848 * exp(x));
    default: /* round, floor, ceil, sign, logical_not */
        return 0.0;
    }
}

static int run_bwd_eltwise(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST);
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const void *px = rt_ptr(p, ptr, AIEDNN_ARG_SRC), *pdy = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    void *pdx = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    const int64_t n = md_numel(x);
    for (int64_t i = 0; i < n; i++)
        rt_st(dx->data_type, pdx, i,
              rt_ld(dy->data_type, pdy, i) * eltwise_df(p->alg, rt_ld(x->data_type, px, i), p->alpha, p->beta));
    return AIEK_OK;
}

static int run_bwd_softmax(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *y = rt_md(p, AIEDNN_ARG_DST), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST);
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const void *py = rt_ptr(p, ptr, AIEDNN_ARG_DST), *pdy = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    void *pdx = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    int64_t outer = 1, inner = 1;
    const int64_t A = y->dims[p->axis];
    for (int i = 0; i < p->axis; i++)
        outer *= y->dims[i];
    for (int i = p->axis + 1; i < y->ndims; i++)
        inner *= y->dims[i];
    const int lg = p->alg == aiednn_softmax_log;
    for (int64_t o = 0; o < outer; o++)
        for (int64_t in = 0; in < inner; in++) {
            const int64_t base = o * A * inner + in;
            double sum = 0;
            for (int64_t a = 0; a < A; a++) {
                const int64_t i = base + a * inner;
                sum += lg ? rt_ld(dy->data_type, pdy, i) : rt_ld(dy->data_type, pdy, i) * rt_ld(y->data_type, py, i);
            }
            for (int64_t a = 0; a < A; a++) {
                const int64_t i = base + a * inner;
                const double yv = rt_ld(y->data_type, py, i), g = rt_ld(dy->data_type, pdy, i);
                rt_st(dx->data_type, pdx, i, lg ? g - exp(yv) * sum : yv * (g - sum));
            }
        }
    return AIEK_OK;
}

/* AIEDNN_BWD_GENERIC=1: the generic loops only (tests compare the typed f32 paths against them byte for byte) */
static int bwd_generic_only(void) {
    const char *e = getenv("AIEDNN_BWD_GENERIC");
    return (e && e[0] == '1') || rt_ref_generic();
}

/* average pooling backward, all f32: run_bwd_pool's arithmetic without the per-element type dispatch */
static int bwd_pool_avg_f32(const rt_prob *p, void *const *ptr, const int64_t *xs, const int64_t *ys, const int64_t *k,
                            const int64_t *s, const int64_t *pl, const int64_t *dl, int64_t NC) {
    const float *dy = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    float *dx = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    const int64_t nx = xs[0] * xs[1] * xs[2], ny = ys[0] * ys[1] * ys[2], nk = k[0] * k[1] * k[2];
    double *acc = (double *)rt_calloc((size_t)(NC * nx), sizeof(double));
    int64_t *idx = (int64_t *)rt_alloc((size_t)nk * sizeof(int64_t));
    if (!acc || !idx) {
        rt_free(acc), rt_free(idx);
        return AIEK_ERR_ALLOC;
    }
    const int incl = p->alg == aiednn_pooling_avg_include_padding;
    for (int64_t nc = 0; nc < NC; nc++) {
        double *ac = acc + nc * nx;
        const float *g = dy + nc * ny;
        for (int64_t a = 0; a < ny; a++) {
            const int64_t a0 = a / (ys[1] * ys[2]), a1 = a / ys[2] % ys[1], a2 = a % ys[2];
            int64_t cnt = 0;
            for (int64_t k0 = 0; k0 < k[0]; k0++) {
                const int64_t b0 = a0 * s[0] - pl[0] + k0 * dl[0];
                if (b0 < 0 || b0 >= xs[0])
                    continue;
                for (int64_t k1 = 0; k1 < k[1]; k1++) {
                    const int64_t b1 = a1 * s[1] - pl[1] + k1 * dl[1];
                    if (b1 < 0 || b1 >= xs[1])
                        continue;
                    for (int64_t k2 = 0; k2 < k[2]; k2++) {
                        const int64_t b2 = a2 * s[2] - pl[2] + k2 * dl[2];
                        if (b2 >= 0 && b2 < xs[2] && cnt < 64 * 64)
                            idx[cnt++] = (b0 * xs[1] + b1) * xs[2] + b2;
                    }
                }
            }
            const double gd = (double)g[a], div = incl ? (double)nk : (double)cnt;
            for (int64_t i = 0; i < cnt; i++)
                ac[idx[i]] += gd / div;
        }
    }
    for (int64_t i = 0; i < NC * nx; i++)
        dx[i] = (float)acc[i];
    rt_free(acc), rt_free(idx);
    return AIEK_OK;
}

static int run_bwd_pool(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST);
    const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC);
    void *pdx = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    const void *pdy = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST), *px = x ? rt_ptr(p, ptr, AIEDNN_ARG_SRC) : nullptr;
    const int sp = dx->ndims - 2;
    int64_t xs[3], ys[3], k[3], s[3], pl[3], dl[3];
    bwd_dims3(dx, xs), bwd_dims3(dy, ys);
    for (int i = 0; i < 3; i++) {
        const int j = i - (3 - sp);
        k[i] = j < 0 ? 1 : p->pnd.kernel[j], s[i] = j < 0 ? 1 : p->pnd.strides[j];
        pl[i] = j < 0 ? 0 : p->pnd.padding_l[j], dl[i] = j < 0 ? 1 : p->pnd.dilates[j] + 1;
    }
    const int64_t NC = dx->dims[0] * dx->dims[1], nx = xs[0] * xs[1] * xs[2], ny = ys[0] * ys[1] * ys[2];
    const int64_t nk = k[0] * k[1] * k[2];
    if (p->alg != aiednn_pooling_max && dx->data_type == aiednn_f32 && dy->data_type == aiednn_f32 && nk <= 64 * 64 &&
        !bwd_generic_only())
        return bwd_pool_avg_f32(p, ptr, xs, ys, k, s, pl, dl, NC);
    double *acc = (double *)rt_calloc((size_t)(NC * nx), sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    for (int64_t nc = 0; nc < NC; nc++)
        for (int64_t a = 0; a < ny; a++) {
            const int64_t a0 = a / (ys[1] * ys[2]), a1 = a / ys[2] % ys[1], a2 = a % ys[2];
            const double g = rt_ld(dy->data_type, pdy, nc * ny + a);
            int64_t idx[64 * 64], cnt = 0, best = -1;
            double bv = -INFINITY;
            for (int64_t kk = 0; kk < nk; kk++) {
                const int64_t b0 = a0 * s[0] - pl[0] + kk / (k[1] * k[2]) * dl[0];
                const int64_t b1 = a1 * s[1] - pl[1] + kk / k[2] % k[1] * dl[1];
                const int64_t b2 = a2 * s[2] - pl[2] + kk % k[2] * dl[2];
                if (b0 < 0 || b0 >= xs[0] || b1 < 0 || b1 >= xs[1] || b2 < 0 || b2 >= xs[2])
                    continue;
                const int64_t b = (b0 * xs[1] + b1) * xs[2] + b2;
                if (p->alg == aiednn_pooling_max) {
                    const double v = rt_ld(x->data_type, px, nc * nx + b);
                    if (best < 0 || v > bv)
                        bv = v, best = b;
                } else if (cnt < 64 * 64) {
                    idx[cnt++] = b;
                }
            }
            if (p->alg == aiednn_pooling_max) {
                if (best >= 0)
                    acc[nc * nx + best] += g;
                continue;
            }
            const double div = p->alg == aiednn_pooling_avg_include_padding ? (double)nk : (double)cnt;
            for (int64_t i = 0; i < cnt; i++)
                acc[nc * nx + idx[i]] += g / div;
        }
    for (int64_t i = 0; i < NC * nx; i++)
        rt_st(dx->data_type, pdx, i, acc[i]);
    rt_free(acc);
    return AIEK_OK;
}

/* ── normalizations: training forward (writes MEAN / VARIANCE) and backward ──
 * a "group" is the set one mean / variance covers: batch norm a channel (N x spatial), layer norm a
 * row (the last dim), group norm (image, group) (C/G x spatial). elem(gi, i) -> flat index and
 * channel (the SCALE / SHIFT index). */
typedef struct {
    int64_t ngroups, len, N, C, sp, G;
    int kind;
} norm_geo;

static void norm_geo_of(const rt_prob *p, const aiednn_memory_desc_t *s, norm_geo *g) {
    g->kind = p->kind;
    if (p->kind == aiednn_layer_normalization) {
        g->C = s->dims[s->ndims - 1];
        g->ngroups = md_numel(s) / g->C, g->len = g->C, g->N = g->ngroups, g->sp = 1, g->G = 1;
        return;
    }
    g->N = s->dims[0], g->C = s->dims[1], g->sp = md_numel(s) / (g->N * g->C);
    if (p->kind == aiednn_batch_normalization) {
        g->ngroups = g->C, g->len = g->N * g->sp, g->G = g->C;
    } else {
        g->G = p->groups, g->ngroups = g->N * g->G, g->len = (g->C / g->G) * g->sp;
    }
}

static inline void norm_elem(const norm_geo *g, int64_t gi, int64_t i, int64_t *flat, int64_t *ch) {
    if (g->kind == aiednn_layer_normalization) {
        *flat = gi * g->len + i, *ch = i;
    } else if (g->kind == aiednn_batch_normalization) {
        const int64_t n = i / g->sp, s = i % g->sp;
        *flat = (n * g->C + gi) * g->sp + s, *ch = gi;
    } else {
        const int64_t n = gi / g->G, grp = gi % g->G, Cg = g->C / g->G;
        *ch = grp * Cg + i / g->sp, *flat = (n * g->C + *ch) * g->sp + i % g->sp;
    }
}

static int run_norm_training(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *s = rt_md(p, AIEDNN_ARG_SRC), *d = rt_md(p, AIEDNN_ARG_DST);
    const void *px = rt_ptr(p, ptr, AIEDNN_ARG_SRC);
    void *py = rt_ptr(p, ptr, AIEDNN_ARG_DST);
    const aiednn_memory_desc_t *mm = rt_md(p, AIEDNN_ARG_MEAN), *vm = rt_md(p, AIEDNN_ARG_VARIANCE);
    const int rms = (p->flags & AIEDNN_RMS_NORM) != 0;
    norm_geo g;
    norm_geo_of(p, s, &g);
    for (int64_t gi = 0; gi < g.ngroups; gi++) {
        double mean = 0, var = 0;
        int64_t fi, ch;
        if (!rms) {
            for (int64_t i = 0; i < g.len; i++)
                norm_elem(&g, gi, i, &fi, &ch), mean += rt_ld(s->data_type, px, fi);
            mean /= (double)g.len;
        }
        for (int64_t i = 0; i < g.len; i++) {
            norm_elem(&g, gi, i, &fi, &ch);
            const double z = rt_ld(s->data_type, px, fi) - mean;
            var += z * z;
        }
        var /= (double)g.len;
        rt_st(mm->data_type, rt_ptr(p, ptr, AIEDNN_ARG_MEAN), gi, mean);
        rt_st(vm->data_type, rt_ptr(p, ptr, AIEDNN_ARG_VARIANCE), gi, var);
        const double inv = 1.0 / sqrt(var + p->eps);
        for (int64_t i = 0; i < g.len; i++) {
            norm_elem(&g, gi, i, &fi, &ch);
            const double sc = ld_opt(p, ptr, AIEDNN_ARG_SCALE, ch, 1.0),
                         sh = rms ? 0 : ld_opt(p, ptr, AIEDNN_ARG_SHIFT, ch, 0.0);
            rt_st(d->data_type, py, fi, sc * (rt_ld(s->data_type, px, fi) - mean) * inv + sh);
        }
    }
    return AIEK_OK;
}

/* layer / RMS norm backward over the last dim, all f32: run_norm_bwd's arithmetic on contiguous rows */
static int norm_bwd_ln_f32(const rt_prob *p, void *const *ptr, const norm_geo *g, int rms) {
    const float *x = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC),
                *dy = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    float *dx = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    const float *mm = rms ? nullptr : (const float *)rt_ptr(p, ptr, AIEDNN_ARG_MEAN);
    const float *vm = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_VARIANCE);
    const float *sc = rt_md(p, AIEDNN_ARG_SCALE) ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SCALE) : nullptr;
    const aiednn_memory_desc_t *dsc = rt_md(p, AIEDNN_ARG_DIFF_SCALE), *dsh = rt_md(p, AIEDNN_ARG_DIFF_SHIFT);
    const int64_t L = g->len, nch = g->C;
    double *dscale = (double *)rt_calloc((size_t)(2 * nch), sizeof(double));
    if (!dscale)
        return AIEK_ERR_ALLOC;
    double *dshift = dscale + nch;
    const double Ld = (double)L;
    for (int64_t gi = 0; gi < g->ngroups; gi++) {
        const float *xr = x + gi * L, *dr = dy + gi * L;
        float *or_ = dx + gi * L;
        const double mean = rms ? 0.0 : (double)mm[gi];
        const double inv = 1.0 / sqrt((double)vm[gi] + p->eps);
        double sg = 0, sgx = 0;
        for (int64_t i = 0; i < L; i++) {
            const double xh = ((double)xr[i] - mean) * inv, d = (double)dr[i];
            const double gg = d * (sc ? (double)sc[i] : 1.0);
            sg += gg, sgx += gg * xh;
            dscale[i] += d * xh, dshift[i] += d;
        }
        for (int64_t i = 0; i < L; i++) {
            const double xh = ((double)xr[i] - mean) * inv;
            const double gg = (double)dr[i] * (sc ? (double)sc[i] : 1.0);
            or_[i] = (float)(inv * (gg - (rms ? 0.0 : sg / Ld) - xh * sgx / Ld));
        }
    }
    for (int64_t c = 0; c < nch; c++) {
        if (dsc)
            rt_st(dsc->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SCALE), c, dscale[c]);
        if (dsh)
            rt_st(dsh->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SHIFT), c, dshift[c]);
    }
    rt_free(dscale);
    return AIEK_OK;
}

static int run_norm_bwd(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *s = rt_md(p, AIEDNN_ARG_SRC), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST);
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const aiednn_memory_desc_t *mm = rt_md(p, AIEDNN_ARG_MEAN), *vm = rt_md(p, AIEDNN_ARG_VARIANCE);
    const aiednn_memory_desc_t *dsc = rt_md(p, AIEDNN_ARG_DIFF_SCALE), *dsh = rt_md(p, AIEDNN_ARG_DIFF_SHIFT);
    const void *px = rt_ptr(p, ptr, AIEDNN_ARG_SRC), *pdy = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    void *pdx = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    const int rms = (p->flags & AIEDNN_RMS_NORM) != 0;
    norm_geo g;
    norm_geo_of(p, s, &g);
    {
        const aiednn_memory_desc_t *scm = rt_md(p, AIEDNN_ARG_SCALE);
        if (g.kind == aiednn_layer_normalization && g.C >= g.len && s->data_type == aiednn_f32 &&
            dy->data_type == aiednn_f32 && dx->data_type == aiednn_f32 && (rms || mm->data_type == aiednn_f32) &&
            vm->data_type == aiednn_f32 && (!scm || scm->data_type == aiednn_f32) && !bwd_generic_only())
            return norm_bwd_ln_f32(p, ptr, &g, rms);
    }
    const int64_t nch = g.kind == aiednn_layer_normalization ? g.C : g.C;
    double *dscale = (double *)rt_calloc((size_t)(2 * nch), sizeof(double));
    if (!dscale)
        return AIEK_ERR_ALLOC;
    double *dshift = dscale + nch;
    for (int64_t gi = 0; gi < g.ngroups; gi++) {
        const double mean = rms ? 0.0 : rt_ld(mm->data_type, rt_ptr(p, ptr, AIEDNN_ARG_MEAN), gi);
        const double inv = 1.0 / sqrt(rt_ld(vm->data_type, rt_ptr(p, ptr, AIEDNN_ARG_VARIANCE), gi) + p->eps);
        double sg = 0, sgx = 0; /* sums of g = dy * scale and of g * xhat over the group */
        int64_t fi, ch;
        for (int64_t i = 0; i < g.len; i++) {
            norm_elem(&g, gi, i, &fi, &ch);
            const double xh = (rt_ld(s->data_type, px, fi) - mean) * inv, d = rt_ld(dy->data_type, pdy, fi);
            const double gg = d * ld_opt(p, ptr, AIEDNN_ARG_SCALE, ch, 1.0);
            sg += gg, sgx += gg * xh;
            dscale[ch] += d * xh, dshift[ch] += d;
        }
        const double L = (double)g.len;
        for (int64_t i = 0; i < g.len; i++) {
            norm_elem(&g, gi, i, &fi, &ch);
            const double xh = (rt_ld(s->data_type, px, fi) - mean) * inv;
            const double gg = rt_ld(dy->data_type, pdy, fi) * ld_opt(p, ptr, AIEDNN_ARG_SCALE, ch, 1.0);
            rt_st(dx->data_type, pdx, fi, inv * (gg - (rms ? 0.0 : sg / L) - xh * sgx / L));
        }
    }
    for (int64_t c = 0; c < nch; c++) {
        if (dsc)
            rt_st(dsc->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SCALE), c, dscale[c]);
        if (dsh)
            rt_st(dsh->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SHIFT), c, dshift[c]);
    }
    rt_free(dscale);
    return AIEK_OK;
}

static int run_bwd_prelu(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC), *w = rt_md(p, AIEDNN_ARG_WEIGHTS);
    const aiednn_memory_desc_t *dy = rt_md(p, AIEDNN_ARG_DIFF_DST), *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const aiednn_memory_desc_t *dw = rt_md(p, AIEDNN_ARG_DIFF_WEIGHTS);
    const int64_t n = md_numel(x), nw = md_numel(w);
    double *acc = (double *)rt_calloc((size_t)nw, sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    for (int64_t i = 0; i < n; i++) {
        const double xv = rt_ld(x->data_type, rt_ptr(p, ptr, AIEDNN_ARG_SRC), i);
        const double g = rt_ld(dy->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST), i);
        const int64_t wi = bcast_off(w, x, i);
        const double wv = rt_ld(w->data_type, rt_ptr(p, ptr, AIEDNN_ARG_WEIGHTS), wi);
        rt_st(dx->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), i, xv > 0 ? g : wv * g);
        if (xv <= 0)
            acc[wi] += xv * g;
    }
    for (int64_t i = 0; i < nw; i++)
        rt_st(dw->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_WEIGHTS), i, acc[i]);
    rt_free(acc);
    return AIEK_OK;
}

/* lrn: y_i = x_i d_i^-b, d_i = k + a/size sum_{j in win(i)} x_j^2;
 * dx_j = dy_j d_j^-b - 2 a b / size x_j sum_{i: j in win(i)} dy_i x_i d_i^(-b-1) */
/* LRN backward, every tensor f32: run_bwd_lrn's two passes in its order, squares from a table (exact) */
static int bwd_lrn_f32(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *xm = rt_md(p, AIEDNN_ARG_SRC);
    const float *x = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC),
                *dy = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    float *dx = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
    const int64_t N = xm->dims[0], C = xm->dims[1], H = xm->ndims > 2 ? xm->dims[2] : 1,
                  W = xm->ndims > 3 ? xm->dims[3] : 1;
    const int64_t ls = p->local_size, half = (ls - 1) / 2, n = md_numel(xm), HW = H * W;
    const int across = p->alg == aiednn_lrn_across_channels;
    const double size = across ? (double)ls : (double)(ls * ls), a = p->alpha, b = p->beta;
    double *dd = (double *)rt_alloc(sizeof(double) * (size_t)(3 * n));
    if (!dd)
        return AIEK_ERR_ALLOC;
    double *t = dd + n, *sq = dd + 2 * n;
    for (int64_t i = 0; i < n; i++)
        sq[i] = (double)x[i] * (double)x[i];
    /* the window of (nn, c, h, w), visited as run_bwd_lrn's win() visits it */
    auto win = [&](const double *v, int64_t nn, int64_t c, int64_t h, int64_t w) {
        double acc = 0;
        if (across) {
            const int64_t c0 = c - half < 0 ? 0 : c - half, c1 = c + half < C ? c + half : C - 1;
            const double *q = v + nn * C * HW + h * W + w;
            for (int64_t cc = c0; cc <= c1; cc++)
                acc += q[cc * HW];
        } else {
            const int64_t h0 = h - half < 0 ? 0 : h - half, h1 = h + half < H ? h + half : H - 1;
            const int64_t w0 = w - half < 0 ? 0 : w - half, w1 = w + half < W ? w + half : W - 1;
            const double *q = v + (nn * C + c) * HW;
            for (int64_t hh = h0; hh <= h1; hh++)
                for (int64_t ww = w0; ww <= w1; ww++)
                    acc += q[hh * W + ww];
        }
        return acc;
    };
    for (int64_t nn = 0; nn < N; nn++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    const int64_t i = ((nn * C + c) * H + h) * W + w;
                    dd[i] = p->lrn_k + a / size * win(sq, nn, c, h, w);
                    t[i] = (double)dy[i] * (double)x[i] * pow(dd[i], -b - 1);
                }
    for (int64_t nn = 0; nn < N; nn++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    const int64_t j = ((nn * C + c) * H + h) * W + w;
                    const double sj = win(t, nn, c, h, w);
                    dx[j] = (float)((double)dy[j] * pow(dd[j], -b) - 2 * a * b / size * (double)x[j] * sj);
                }
    rt_free(dd);
    return AIEK_OK;
}

static int run_bwd_lrn(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *x = rt_md(p, AIEDNN_ARG_SRC), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST);
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const void *px = rt_ptr(p, ptr, AIEDNN_ARG_SRC), *pdy = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    const int64_t N = x->dims[0], C = x->dims[1], H = x->ndims > 2 ? x->dims[2] : 1, W = x->ndims > 3 ? x->dims[3] : 1;
    const int64_t ls = p->local_size, half = (ls - 1) / 2, n = md_numel(x);
    if (x->data_type == aiednn_f32 && dy->data_type == aiednn_f32 && dx->data_type == aiednn_f32 && !bwd_generic_only())
        return bwd_lrn_f32(p, ptr);
    const int across = p->alg == aiednn_lrn_across_channels;
    const double size = across ? (double)ls : (double)(ls * ls), a = p->alpha, b = p->beta;
    double *dd = (double *)rt_alloc(sizeof(double) * (size_t)(2 * n));
    if (!dd)
        return AIEK_ERR_ALLOC;
    double *t = dd + n; /* t_i = dy_i x_i d_i^(-b-1) */
    auto win = [&](int64_t nn, int64_t c, int64_t h, int64_t w, auto fn) {
        if (across) {
            for (int64_t cc = c - half; cc <= c + half; cc++)
                if (cc >= 0 && cc < C)
                    fn(((nn * C + cc) * H + h) * W + w);
        } else {
            for (int64_t hh = h - half; hh <= h + half; hh++)
                for (int64_t ww = w - half; ww <= w + half; ww++)
                    if (hh >= 0 && hh < H && ww >= 0 && ww < W)
                        fn(((nn * C + c) * H + hh) * W + ww);
        }
    };
    for (int64_t nn = 0; nn < N; nn++)
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    const int64_t i = ((nn * C + c) * H + h) * W + w;
                    double sum = 0;
                    win(nn, c, h, w, [&](int64_t j) {
                        const double v = rt_ld(x->data_type, px, j);
                        sum += v * v;
                    });
                    dd[i] = p->lrn_k + a / size * sum;
                    t[i] = rt_ld(dy->data_type, pdy, i) * rt_ld(x->data_type, px, i) * pow(dd[i], -b - 1);
                }
    for (int64_t nn = 0; nn < N; nn++) /* windows are symmetric: j in win(i) <=> i in win(j) */
        for (int64_t c = 0; c < C; c++)
            for (int64_t h = 0; h < H; h++)
                for (int64_t w = 0; w < W; w++) {
                    const int64_t j = ((nn * C + c) * H + h) * W + w;
                    double s = 0;
                    win(nn, c, h, w, [&](int64_t i) { s += t[i]; });
                    rt_st(dx->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), j,
                          rt_ld(dy->data_type, pdy, j) * pow(dd[j], -b) -
                              2 * a * b / size * rt_ld(x->data_type, px, j) * s);
                }
    rt_free(dd);
    return AIEK_OK;
}

/* resampling is linear in src: the forward's interpolation, transposed */
static int run_bwd_resampling(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST);
    const void *pdy = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    const int sp = dx->ndims - 2;
    const int64_t NC = dx->dims[0] * dx->dims[1], nsp = md_numel(dy) / NC, nsrc = md_numel(dx) / NC;
    double *acc = (double *)rt_calloc((size_t)(NC * nsrc), sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    if (dx->data_type == aiednn_f32 && dy->data_type == aiednn_f32 && !bwd_generic_only()) {
        /* typed f32: the forward's taps once for all planes (rs_taps), scattered in the same order */
        rs_tap *tap = (rs_tap *)scratch(&g_rs_tap, &g_rs_tap_cap, (size_t)nsp * 8 * sizeof(rs_tap));
        uint8_t *cnt = (uint8_t *)scratch(&g_rs_cnt, &g_rs_cnt_cap, (size_t)nsp);
        if (!tap || !cnt) {
            rt_free(acc);
            return AIEK_ERR_ALLOC;
        }
        rs_taps(p, dx, dy, tap, cnt);
        const float *g = (const float *)pdy;
        float *o = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC);
        for (int64_t nc = 0; nc < NC; nc++) {
            double *ac = acc + nc * nsrc;
            for (int64_t q = 0; q < nsp; q++) {
                const double gv = (double)g[nc * nsp + q];
                const rs_tap *t = tap + q * 8;
                for (int k = 0; k < cnt[q]; k++)
                    ac[t[k].off] += t[k].w * gv;
            }
        }
        for (int64_t i = 0; i < NC * nsrc; i++)
            o[i] = (float)acc[i];
        rt_free(acc);
        return AIEK_OK;
    }
    for (int64_t nc = 0; nc < NC; nc++)
        for (int64_t o = 0; o < nsp; o++) {
            int64_t oc[3] = {0, 0, 0}, rem = o, lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
            double wh[3] = {0, 0, 0};
            for (int i = sp - 1; i >= 0; i--) {
                oc[i] = rem % dy->dims[2 + i];
                rem /= dy->dims[2 + i];
            }
            for (int i = 0; i < sp; i++) { /* = run_ref_resampling */
                const double in = (double)dx->dims[2 + i], out = (double)dy->dims[2 + i];
                const double c = ((double)oc[i] + 0.5) * in / out - 0.5;
                if (p->alg == aiednn_resampling_nearest) {
                    const int64_t k = (int64_t)floor(((double)oc[i] + 0.5) * in / out);
                    lo[i] = hi[i] = k < (int64_t)in - 1 ? k : (int64_t)in - 1;
                } else {
                    const double cl = c < 0 ? 0 : c;
                    lo[i] = (int64_t)floor(cl);
                    hi[i] = lo[i] + 1 < (int64_t)in ? lo[i] + 1 : (int64_t)in - 1;
                    wh[i] = cl - (double)lo[i];
                }
            }
            const double g = rt_ld(dy->data_type, pdy, nc * nsp + o);
            for (int corner = 0; corner < (1 << sp); corner++) {
                double w = 1;
                int64_t off = 0;
                for (int i = 0; i < sp; i++) {
                    const int up = (corner >> i) & 1;
                    w *= up ? wh[i] : 1.0 - wh[i];
                    off = off * dx->dims[2 + i] + (up ? hi[i] : lo[i]);
                }
                if (w != 0)
                    acc[nc * nsrc + off] += w * g;
            }
        }
    for (int64_t i = 0; i < NC * nsrc; i++)
        rt_st(dx->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), i, acc[i]);
    rt_free(acc);
    return AIEK_OK;
}

/* the forward moved src channel i*Q + j to dst channel j*G + i: the gradient moves it back */
static int run_bwd_shuffle(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC), *dy = rt_md(p, AIEDNN_ARG_DIFF_DST);
    int64_t outer = 1, inner = 1;
    const int64_t C = dx->dims[p->axis], G = p->group_size, Q = C / G;
    for (int i = 0; i < p->axis; i++)
        outer *= dx->dims[i];
    for (int i = p->axis + 1; i < dx->ndims; i++)
        inner *= dx->dims[i];
    for (int64_t o = 0; o < outer; o++)
        for (int64_t c = 0; c < C; c++) {
            const int64_t i = c % G, j = c / G;
            for (int64_t e = 0; e < inner; e++)
                rt_st(dx->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), (o * C + i * Q + j) * inner + e,
                      rt_ld(dy->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST), (o * C + c) * inner + e));
        }
    return AIEK_OK;
}

static int run_bwd(const rt_prob *p, void *const *ptr) {
    switch (p->kind) {
    case aiednn_convolution:
    case aiednn_deconvolution:
        return run_bwd_conv(p, ptr);
    case aiednn_inner_product:
        return run_bwd_ip(p, ptr);
    case aiednn_eltwise:
        return run_bwd_eltwise(p, ptr);
    case aiednn_softmax:
        return run_bwd_softmax(p, ptr);
    case aiednn_pooling:
        return run_bwd_pool(p, ptr);
    case aiednn_batch_normalization:
    case aiednn_layer_normalization:
    case aiednn_group_normalization:
        return p->prop == RT_FWD_TRAIN ? run_norm_training(p, ptr) : run_norm_bwd(p, ptr);
    case aiednn_prelu:
        return run_bwd_prelu(p, ptr);
    case aiednn_lrn:
        return run_bwd_lrn(p, ptr);
    case aiednn_resampling:
        return run_bwd_resampling(p, ptr);
    case aiednn_shuffle:
        return run_bwd_shuffle(p, ptr);
    default:
        return AIEK_ERR_ARGS;
    }
}

#endif /* AIEDNN_RT_BWD_H */
