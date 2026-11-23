/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_deconvbwd.h -- conv+: deconvolution backward as convolutions (on the AIE where they are), and
 * deconv+: average pooling backward as a depthwise deconvolution; conv+: nearest resampling backward (integer
 * factors) as a depthwise convolution.
 *
 * A deconvolution relates its src position a and dst position a * s - pl + k * dil, the convolution's relation
 * with src and dst exchanged. So, with W [O, Cg, k..] (O the deconvolution's dst channels):
 *   backward data     dx = conv_forward(src = dy, weights Wc, dst = dx)
 *   backward weights  dWc = conv_backward_weights(src = dy, diff_dst = x), dW = the transpose of dWc
 * with Wc [C, Og, k..], Wc[g Cg + c][oo] = W[g Og + oo][c] per group. diff_bias sums dy per channel (host).
 * The inner primitives are the public creates, so dispatch puts them where convolutions go (gemm+conv on the
 * AIE). Floating types only, as the training creates.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, the public creates of aiednn_rt.cc / aiednn_rt_bwd_create.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_DECONVBWD_H
#define AIEDNN_RT_DECONVBWD_H

typedef struct {
    aiednn_memory_desc_t x, w, y, wc; /* deconv src side, deconv weights, deconv dst side, the conv's weights */
    int64_t G, O, C, Og, Cg, nk;
} dcb_geo;

static int dcb_geo_of(const rt_prob *p, dcb_geo *g) {
    if (p->kind != aiednn_deconvolution || (p->prop != RT_BWD_DATA && p->prop != RT_BWD_WEIGHTS))
        return 0;
    const int data = p->prop == RT_BWD_DATA;
    g->x = *rt_md(p, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC);
    g->w = *rt_md(p, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
    g->y = *rt_md(p, AIEDNN_ARG_DIFF_DST);
    g->G = p->cnd.groups < 1 ? 1 : p->cnd.groups, g->O = g->y.dims[1], g->C = g->x.dims[1];
    g->Og = g->O / g->G, g->Cg = g->C / g->G, g->nk = 1;
    for (int i = 2; i < g->w.ndims; i++)
        g->nk *= g->w.dims[i];
    if (g->w.ndims != g->x.ndims || g->w.dims[0] != g->O || g->w.dims[1] != g->Cg)
        return 0; /* grouped weights given as [O, Cg, k..] only */
    g->wc = g->w;
    g->wc.dims[0] = g->C, g->wc.dims[1] = g->Og;
    return 1;
}

/* W [O, Cg, k] <-> Wc [C, Og, k] (to_c 1: W -> Wc) */
static void dcb_swap_w(const dcb_geo *g, const void *src, void *dst, int to_c) {
    const int64_t es = dt_bytes(g->w.data_type), nk = g->nk;
    for (int64_t gi = 0; gi < g->G; gi++)
        for (int64_t oo = 0; oo < g->Og; oo++)
            for (int64_t c = 0; c < g->Cg; c++) {
                const int64_t wi = ((gi * g->Og + oo) * g->Cg + c) * nk, ci = ((gi * g->Cg + c) * g->Og + oo) * nk;
                memcpy((char *)dst + (to_c ? ci : wi) * es, (const char *)src + (to_c ? wi : ci) * es,
                       (size_t)(nk * es));
            }
}

/* the inner primitive: conv forward (data) or conv backward weights (weights), created through the public API */
static aiednn_status_t dcb_inner(const rt_prob *p, const dcb_geo *g, aiednn_primitive_t *in) {
    return p->prop == RT_BWD_DATA
               ? aiednn_convolution_forward_create_nd(in, &g->y, &g->wc, nullptr, &g->x, &p->cnd, nullptr)
               : aiednn_convolution_backward_weights_create(in, &g->y, &g->wc, nullptr, &g->x, &p->cnd);
}

static int acc_dcb(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    dcb_geo g;
    aiednn_primitive_t in = nullptr;
    if (!dcb_geo_of(p, &g) || dcb_inner(p, &g, &in) != aiednn_success) {
        snprintf(why, cap, "deconvolution backward whose convolution form creates");
        return 0;
    }
    const int ok = strncmp(aiednn_primitive_impl_info(in), "ref:", 4) != 0;
    if (!ok)
        snprintf(why, cap, "its convolution would run on %s", aiednn_primitive_impl_info(in));
    else /* impl_info: conv+deconvbwd+<the convolution's impl>+<its inner> (the "aie:" the census reads) */
        bind->inner = in->impl, bind->inner2 = in->bind.inner, bind->blk = in->bind.blk;
    aiednn_primitive_destroy(in);
    return ok;
}

static int run_dcb(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    dcb_geo g;
    aiednn_primitive_t in = nullptr;
    if (!dcb_geo_of(p, &g) || dcb_inner(p, &g, &in) != aiednn_success)
        return AIEK_ERR_ARGS;
    const int data = p->prop == RT_BWD_DATA;
    void *wc = rt_alloc((size_t)(md_numel(&g.wc) * dt_bytes(g.w.data_type)));
    int rc = wc ? AIEK_OK : AIEK_ERR_ALLOC;
    void *pw = rt_ptr(p, ptr, data ? AIEDNN_ARG_WEIGHTS : AIEDNN_ARG_DIFF_WEIGHTS);
    void *px = rt_ptr(p, ptr, data ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_SRC), *py = rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST);
    if (rc == AIEK_OK) {
        if (data)
            dcb_swap_w(&g, pw, wc, 1);
        const aiednn_exec_arg_t a_data[] = {{AIEDNN_ARG_SRC, py}, {AIEDNN_ARG_WEIGHTS, wc}, {AIEDNN_ARG_DST, px}};
        const aiednn_exec_arg_t a_wts[] = {
            {AIEDNN_ARG_SRC, py}, {AIEDNN_ARG_DIFF_WEIGHTS, wc}, {AIEDNN_ARG_DIFF_DST, px}};
        rc = aiednn_primitive_execute_args(in, 3, data ? a_data : a_wts) == aiednn_success ? AIEK_OK : AIEK_ERR_ARGS;
        if (rc == AIEK_OK && !data)
            dcb_swap_w(&g, wc, pw, 0);
    }
    const aiednn_memory_desc_t *db = rt_md(p, AIEDNN_ARG_DIFF_BIAS);
    if (rc == AIEK_OK && !data && db) { /* the bias gradient: dy summed per channel */
        const int64_t ny = md_numel(&g.y) / (g.y.dims[0] * g.O);
        for (int64_t o = 0; o < g.O; o++) {
            double s = 0;
            for (int64_t n = 0; n < g.y.dims[0]; n++)
                for (int64_t i = 0; i < ny; i++)
                    s += rt_ld(g.y.data_type, py, (n * g.O + o) * ny + i);
            rt_st(db->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_BIAS), o, s);
        }
    }
    rt_free(wc);
    aiednn_primitive_destroy(in);
    return rc;
}

// ── avg pooling backward as a depthwise deconvolution ────────────────────────
/* dx = deconv(dy, w = 1 / |window|) per channel (groups = C), the pool's strides / padding / dilation: the
 * include-padding average; exclude-padding scales dy by each window's in-bounds count first (weights 1). padding_r is
 * re-derived so the deconvolution's output is exactly dx's size (a pool may drop a partial last window). */
typedef struct {
    aiednn_memory_desc_t dx, dy, w;
    aiednn_conv_nd_params_t cp;
    double inv;
    int excl;
} apb_geo;

static int apb_geo_of(const rt_prob *p, apb_geo *g) {
    if (p->kind != aiednn_pooling || p->prop != RT_BWD_DATA)
        return 0;
    const aiednn_pool_nd_params_t *pp = &p->pnd;
    g->dx = *rt_md(p, AIEDNN_ARG_DIFF_SRC), g->dy = *rt_md(p, AIEDNN_ARG_DIFF_DST);
    const int sp = g->dx.ndims - 2;
    int pad = 0;
    for (int i = 0; i < sp; i++)
        pad |= pp->padding_l[i] || pp->padding_r[i];
    if (p->alg != aiednn_pooling_avg_include_padding && p->alg != aiednn_pooling_avg_exclude_padding)
        return 0;
    g->excl = p->alg == aiednn_pooling_avg_exclude_padding && pad; /* divisor per window: dy scaled first */
    memset(&g->cp, 0, sizeof g->cp);
    g->w = g->dx;
    g->w.dims[0] = g->dx.dims[1], g->w.dims[1] = 1, g->w.data_type = aiednn_f32;
    int64_t win = 1;
    for (int i = 0; i < sp; i++) {
        const int64_t ext = (pp->kernel[i] - 1) * (pp->dilates[i] + 1) + 1;
        g->w.dims[2 + i] = pp->kernel[i], win *= pp->kernel[i];
        g->cp.strides[i] = pp->strides[i], g->cp.padding_l[i] = pp->padding_l[i], g->cp.dilates[i] = pp->dilates[i];
        g->cp.padding_r[i] = (g->dy.dims[2 + i] - 1) * pp->strides[i] + ext - pp->padding_l[i] - g->dx.dims[2 + i];
        if (g->cp.padding_r[i] < 0)
            return 0;
    }
    g->cp.groups = g->dx.dims[1], g->inv = g->excl ? 1.0 : 1.0 / (double)win;
    return 1;
}

/* exclude padding: dy / (the window's in-bounds taps) per output position, as ref:'s divisor */
static float *apb_scaled_dy(const rt_prob *p, const apb_geo *g, const float *dy) {
    const aiednn_pool_nd_params_t *pp = &p->pnd;
    const int sp = g->dx.ndims - 2;
    const int64_t ny = md_numel(&g->dy) / (g->dy.dims[0] * g->dy.dims[1]), planes = g->dy.dims[0] * g->dy.dims[1];
    float *o = (float *)rt_alloc((size_t)md_numel(&g->dy) * sizeof(float));
    if (!o)
        return nullptr;
    for (int64_t j = 0; j < ny; j++) {
        int64_t rem = j, cnt = 1;
        for (int i = sp - 1; i >= 0; i--) {
            const int64_t oi = rem % g->dy.dims[2 + i];
            rem /= g->dy.dims[2 + i];
            int64_t c = 0;
            for (int64_t k = 0; k < pp->kernel[i]; k++) {
                const int64_t xi = oi * pp->strides[i] - pp->padding_l[i] + k * (pp->dilates[i] + 1);
                c += xi >= 0 && xi < g->dx.dims[2 + i];
            }
            cnt *= c;
        }
        const double inv = cnt > 0 ? 1.0 / (double)cnt : 0.0;
        for (int64_t pl = 0; pl < planes; pl++)
            o[pl * ny + j] = (float)((double)dy[pl * ny + j] * inv);
    }
    return o;
}

static int acc_apb(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    apb_geo g;
    aiednn_primitive_t in = nullptr;
    if (!apb_geo_of(p, &g) ||
        aiednn_deconvolution_forward_create_nd(&in, &g.dy, &g.w, nullptr, &g.dx, &g.cp, nullptr) != aiednn_success) {
        snprintf(why, cap, "average pooling backward with one divisor (include padding, or no padding)");
        return 0;
    }
    const int ok = strncmp(aiednn_primitive_impl_info(in), "ref:", 4) != 0;
    if (!ok)
        snprintf(why, cap, "its deconvolution would run on %s", aiednn_primitive_impl_info(in));
    else
        bind->inner = in->impl, bind->inner2 = in->bind.inner, bind->blk = in->bind.blk;
    aiednn_primitive_destroy(in);
    return ok;
}

static int run_apb(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    apb_geo g;
    aiednn_primitive_t in = nullptr;
    if (!apb_geo_of(p, &g) ||
        aiednn_deconvolution_forward_create_nd(&in, &g.dy, &g.w, nullptr, &g.dx, &g.cp, nullptr) != aiednn_success)
        return AIEK_ERR_ARGS;
    const int64_t nw = md_numel(&g.w);
    float *w = (float *)rt_alloc((size_t)nw * sizeof(float));
    int rc = w ? AIEK_OK : AIEK_ERR_ALLOC;
    for (int64_t i = 0; rc == AIEK_OK && i < nw; i++)
        w[i] = (float)g.inv;
    float *dys =
        rc == AIEK_OK && g.excl ? apb_scaled_dy(p, &g, (const float *)rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST)) : nullptr;
    if (rc == AIEK_OK && g.excl && !dys)
        rc = AIEK_ERR_ALLOC;
    if (rc == AIEK_OK) {
        const aiednn_exec_arg_t a[] = {{AIEDNN_ARG_SRC, g.excl ? (void *)dys : rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST)},
                                       {AIEDNN_ARG_WEIGHTS, w},
                                       {AIEDNN_ARG_DST, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC)}};
        rc = aiednn_primitive_execute_args(in, 3, a) == aiednn_success ? AIEK_OK : AIEK_ERR_ARGS;
    }
    rt_free(w), rt_free(dys);
    aiednn_primitive_destroy(in);
    return rc;
}

// ── nearest resampling backward with integer factors as a depthwise convolution ─────────────────
/* out = f in per axis: output j reads input floor((j + 0.5) / f) = j / f, so dx sums dy over f-blocks: a depthwise
 * convolution of dy with ones, kernel = stride = f */
typedef struct {
    aiednn_memory_desc_t dx, dy, w;
    aiednn_conv_nd_params_t cp;
} rsb_geo;

static int rsb_geo_of(const rt_prob *p, rsb_geo *g) {
    if (p->kind != aiednn_resampling || p->prop != RT_BWD_DATA || p->alg != aiednn_resampling_nearest)
        return 0;
    g->dx = *rt_md(p, AIEDNN_ARG_DIFF_SRC), g->dy = *rt_md(p, AIEDNN_ARG_DIFF_DST);
    memset(&g->cp, 0, sizeof g->cp);
    g->w = g->dx;
    g->w.dims[0] = g->dx.dims[1], g->w.dims[1] = 1, g->w.data_type = aiednn_f32;
    for (int i = 2; i < g->dx.ndims; i++) {
        const int64_t in = g->dx.dims[i], out = g->dy.dims[i];
        if (out % in || out < in)
            return 0;
        g->w.dims[i] = g->cp.strides[i - 2] = out / in;
    }
    g->cp.groups = g->dx.dims[1];
    return 1;
}

static int acc_rsb(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    rsb_geo g;
    aiednn_primitive_t in = nullptr;
    if (!rsb_geo_of(p, &g) ||
        aiednn_convolution_forward_create_nd(&in, &g.dy, &g.w, nullptr, &g.dx, &g.cp, nullptr) != aiednn_success) {
        snprintf(why, cap, "nearest resampling backward with integer up-scale factors");
        return 0;
    }
    const int ok = strncmp(aiednn_primitive_impl_info(in), "ref:", 4) != 0;
    if (!ok)
        snprintf(why, cap, "its convolution would run on %s", aiednn_primitive_impl_info(in));
    else
        bind->inner = in->impl, bind->inner2 = in->bind.inner, bind->blk = in->bind.blk;
    aiednn_primitive_destroy(in);
    return ok;
}

static int run_rsb(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    rsb_geo g;
    aiednn_primitive_t in = nullptr;
    if (!rsb_geo_of(p, &g) ||
        aiednn_convolution_forward_create_nd(&in, &g.dy, &g.w, nullptr, &g.dx, &g.cp, nullptr) != aiednn_success)
        return AIEK_ERR_ARGS;
    const int64_t nw = md_numel(&g.w);
    float *w = (float *)rt_alloc((size_t)nw * sizeof(float));
    int rc = w ? AIEK_OK : AIEK_ERR_ALLOC;
    for (int64_t i = 0; rc == AIEK_OK && i < nw; i++)
        w[i] = 1.0f;
    if (rc == AIEK_OK) {
        const aiednn_exec_arg_t a[] = {{AIEDNN_ARG_SRC, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST)},
                                       {AIEDNN_ARG_WEIGHTS, w},
                                       {AIEDNN_ARG_DST, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC)}};
        rc = aiednn_primitive_execute_args(in, 3, a) == aiednn_success ? AIEK_OK : AIEK_ERR_ARGS;
    }
    rt_free(w);
    aiednn_primitive_destroy(in);
    return rc;
}

// ── linear resampling backward as per-axis GEMMs ────────────────────────────
/* the linear map is separable (weights are a product over axes), so its transpose is applied an axis at a time:
 * with the axis moved last, Y [rows][in] = X [rows][out] A [out][in], A the axis' taps exactly as ref: computes
 * them (half-pixel centres, clamped at the borders). The GEMMs take the float matmul dispatch (bf16x3 on the AIE). */
static void rsl_taps(int64_t in, int64_t out, float *A) { /* A [out][in] */
    memset(A, 0, (size_t)(out * in) * sizeof(float));
    for (int64_t o = 0; o < out; o++) {
        const double c = ((double)o + 0.5) * (double)in / (double)out - 0.5, cl = c < 0 ? 0 : c;
        const int64_t lo = (int64_t)floor(cl), hi = lo + 1 < in ? lo + 1 : in - 1;
        const double wh = cl - (double)lo;
        A[o * in + lo] += (float)(1.0 - wh);
        A[o * in + hi] += (float)wh;
    }
}

static int acc_rsl(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (p->kind != aiednn_resampling || p->prop != RT_BWD_DATA || p->alg != aiednn_resampling_linear ||
        rt_md(p, AIEDNN_ARG_DIFF_DST)->data_type != aiednn_f32 ||
        rt_md(p, AIEDNN_ARG_DIFF_SRC)->data_type != aiednn_f32) {
        snprintf(why, cap, "linear resampling backward, f32");
        return 0;
    }
    const aiednn_memory_desc_t *dy = rt_md(p, AIEDNN_ARG_DIFF_DST), *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const gx_types t = {aiednn_f32, aiednn_f32, aiednn_f32};
    const int k = dy->ndims - 1;
    return gx_inner_ok(&t, md_numel(dy) / dy->dims[k], dx->dims[k], dy->dims[k], bind, why, cap);
}

static int run_rsl(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *dym = rt_md(p, AIEDNN_ARG_DIFF_DST), *dxm = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const int nd = dym->ndims;
    int64_t dims[AIEDNN_MAX_NDIMS];
    for (int i = 0; i < nd; i++)
        dims[i] = dym->dims[i];
    const int64_t cap = md_numel(dym) > md_numel(dxm) ? md_numel(dym) : md_numel(dxm);
    int64_t big = cap, sz = md_numel(dym); /* every intermediate, in the order the axes are applied */
    for (int k = nd - 1; k >= 2; k--)
        sz = sz / dym->dims[k] * dxm->dims[k], big = sz > big ? sz : big;
    float *X = (float *)rt_alloc((size_t)(3 * big) * sizeof(float)), *T = X + big, *Y = T + big;
    int rc = X ? AIEK_OK : AIEK_ERR_ALLOC;
    if (rc == AIEK_OK)
        memcpy(X, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST), (size_t)md_numel(dym) * sizeof(float));
    const gx_types gt = {aiednn_f32, aiednn_f32, aiednn_f32};
    for (int k = nd - 1; rc == AIEK_OK && k >= 2; k--) {
        const int64_t out = dims[k], in = dxm->dims[k];
        int64_t outer = 1, inner = 1;
        for (int i = 0; i < k; i++)
            outer *= dims[i];
        for (int i = k + 1; i < nd; i++)
            inner *= dims[i];
        float *A = (float *)rt_alloc((size_t)(out * in) * sizeof(float));
        if (!A) {
            rc = AIEK_ERR_ALLOC;
            break;
        }
        rsl_taps(in, out, A);
        for (int64_t o = 0; o < outer; o++) /* [outer][out][inner] -> [outer][inner][out] */
            for (int64_t m = 0; m < out; m++)
                for (int64_t i = 0; i < inner; i++)
                    T[(o * inner + i) * out + m] = X[(o * out + m) * inner + i];
        rc = gx_gemm(&gt, outer * inner, in, out, T, A, Y);
        for (int64_t o = 0; rc == AIEK_OK && o < outer; o++) /* [outer][inner][in] -> [outer][in][inner] */
            for (int64_t m = 0; m < in; m++)
                for (int64_t i = 0; i < inner; i++)
                    X[(o * in + m) * inner + i] = Y[(o * inner + i) * in + m];
        dims[k] = in;
        rt_free(A);
    }
    if (rc == AIEK_OK)
        memcpy(rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), X, (size_t)md_numel(dxm) * sizeof(float));
    rt_free(X);
    return rc;
}

#endif /* AIEDNN_RT_DECONVBWD_H */
