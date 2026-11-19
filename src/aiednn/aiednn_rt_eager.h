/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_eager.h -- eager mode: every GEMM-family layer runs on its own shape-specialized kernel instance
 * (aie:eager_<template>__<args> entries: a model's generated plan, dnn_eager.plan, or the premade library,
 * dnn_eager.LIBRARY; templates in src/aietensorop/eager/). Every impl runs any layer shape on whatever instances
 * are linked -- slices / chunks / tiles of the instance's own shape, zero padding being exact -- and prefers the
 * instance needing the fewest launches, so a per-model plan runs each layer in one launch and the library in
 * several. Each launch loads that instance's kernel ELF: the model runs eagerly, layer by layer.
 ******************************************************************************/
#ifndef AIEDNN_RT_EAGER_H
#define AIEDNN_RT_EAGER_H

static int is_eager_inst(const aiednn_impl_t *g) { return strncmp(g->name, "aie:eager_", 10) == 0 && is_gemm_block(g); }

static int64_t eager_blk_launches(const aiednn_impl_t *e, int64_t M, int64_t N, int64_t K) {
    const int64_t bm = e->args[0].dims[0], bn = e->args[1].dims[0], bk = e->args[0].dims[1];
    return ((M + bm - 1) / bm) * ((N + bn - 1) / bn) * ((K + bk - 1) / bk);
}

static int acc_eager_mm(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (p->kind != aiednn_matmul || !mm_int8_problem(p)) {
        snprintf(why, cap, "only s8 x s8 matmul");
        return 0;
    }
    mm_geo g;
    mm_geometry(p, &g);
    mm_fold_batch(&g);
    /* run_blocked tiles M, N and K by the instance's block (zero padding and the host's int32 K sums are exact),
     * so any gemm_s8 instance serves any shape: the one with the fewest launches (a generated plan has one that
     * covers the layer in one; the premade library has general ones), then the least padding */
    const aiednn_impl_t *best = nullptr;
    int64_t bl = 0, bv = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        const aiednn_impl_t *e = &aiednn_impl_list[i];
        if (!is_eager_inst(e))
            continue;
        const int64_t l = eager_blk_launches(e, g.M, g.N, g.K);
        const int64_t v = l * e->args[0].dims[0] * e->args[1].dims[0] * e->args[0].dims[1];
        if (!best || l < bl || (l == bl && v < bv))
            best = e, bl = l, bv = v;
    }
    if (best) {
        bind->blk = best;
        return 1;
    }
    snprintf(why, cap, "no gemm_s8 instance linked");
    return 0;
}

/* ── eager:mmq / eager:convq -- the requant on the core (gemm_q / gemm_qr instances,
 * aie:eager_gemm_q[r]__M_N_K_TK_TM) ──────────────────────────────────────────────────────────────────────
 * A' [Mi][Ki + XC TK] (zero past K; gemm_qr: the residual codes in the chunks after the Q chunk), W' [Ni][same]
 * (each channel's requant words in the first chunk past Ki), C [Mi][Ni] 8-bit codes: the fixed-point epilogue
 * rt_fx_init derives (the definition ref: uses too), so the result is exact. Any layer with K <= Ki runs as
 * ceil(N / Ni) channel slices x ceil(M / Mi) row chunks of the instance; a K past every instance goes to eager:mm
 * (int32 K sums on the host, the same bytes). Accepted for an 8-bit dst with output scales, a per-channel bias or
 * none, [an 8-bit residual add,] ReLU / clip. */

#define RT_EAGER_Q_WORDS 8   /* gemm_q:  {M, B lo, B hi, sh, zp, lo, hi, 0} */
#define RT_EAGER_QR_WORDS 12 /* gemm_qr: the same, then {zr, R lo, R hi, r is u8, 0} (gemm_qr_eager_ref.h) */

typedef struct {
    const aiednn_impl_t *g;
    int res; /* gemm_qr: the residual rides in A's chunks past the Q chunk */
    int64_t M, N, K, TK;
} rt_eager_q;

/* chunks past K (Q, then the residual's) and the operand row length */
static int64_t rt_eager_q_xc(const rt_eager_q *q) { return q->res ? 1 + (q->N + q->TK - 1) / q->TK : 1; }
static int64_t rt_eager_q_ld(const rt_eager_q *q) { return q->K + rt_eager_q_xc(q) * q->TK; }

/* the instance's own padded shape, read from its name (the spec dims hold K + its chunks) */
static int rt_eager_q_of(const aiednn_impl_t *g, rt_eager_q *q) {
    long m, n, k, tk, tm;
    const int res = strncmp(g->name, "aie:eager_gemm_qr__", 19) == 0;
    if ((!res && strncmp(g->name, "aie:eager_gemm_q__", 18)) || g->nargs != 3 ||
        sscanf(g->name + (res ? 19 : 18), "%ld_%ld_%ld_%ld_%ld", &m, &n, &k, &tk, &tm) != 5)
        return 0;
    q->g = g, q->res = res, q->M = m, q->N = n, q->K = k, q->TK = tk;
    return 1;
}

/* the instance of the template (res: gemm_qr) with K >= the layer's that needs the fewest launches, then the
 * least padded work */
static int rt_eager_q_find(int64_t M, int64_t N, int64_t K, int res, rt_eager_q *out) {
    int found = 0;
    int64_t bl = 0, bv = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_q q;
        if (!rt_eager_q_of(&aiednn_impl_list[i], &q) || q.res != res || q.K < K)
            continue;
        const int64_t l = ((M + q.M - 1) / q.M) * ((N + q.N - 1) / q.N), v = l * q.M * q.N * q.K;
        if (!found || l < bl || (l == bl && v < bv))
            *out = q, found = 1, bl = l, bv = v;
    }
    return found;
}

/* the epilogue rt_fx_init turns into fixed point, decided from the problem alone (rt_fx_init still checks the
 * values at execute; the runners fall back to ref: if it refuses): an 8-bit dst with output scales, a per-channel
 * bias or none, then [an 8-bit residual add of dst's shape] then ReLU / clip. *res: the residual is there. */
static int rt_eager_q_epilogue(const rt_prob *p, int64_t nch, int *res, char *why, int cap) {
    const aiednn_memory_desc_t *d = &p->md[2];
    if ((d->data_type != aiednn_s8 && d->data_type != aiednn_u8) || !p->has_attr || !p->n_oscales) {
        snprintf(why, cap, "an 8-bit dst with output scales");
        return 0;
    }
    const int bs = rt_slot(p, AIEDNN_ARG_BIAS);
    if (bs >= 0 && md_numel(&p->md[bs]) != nch) {
        snprintf(why, cap, "a per-channel bias or none");
        return 0;
    }
    int k = 0;
    *res = 0;
    if (p->npo > 0 && p->po[0].kind == aiednn_post_op_binary) {
        const aiednn_post_op_t *po = &p->po[0];
        if (po->alg != aiednn_binary_add || (po->src1.data_type != aiednn_s8 && po->src1.data_type != aiednn_u8) ||
            !(po->alpha > 0) || !same_dims(&po->src1, d) || md_is_nhwc(&po->src1)) {
            snprintf(why, cap, "a residual other than an 8-bit add of dst's shape");
            return 0;
        }
        *res = 1, k = 1;
    }
    for (; k < p->npo; k++)
        if (p->po[k].kind != aiednn_post_op_eltwise ||
            !((p->po[k].alg == aiednn_eltwise_relu && p->po[k].alpha == 0) || p->po[k].alg == aiednn_eltwise_clip)) {
            snprintf(why, cap, "post-ops other than [a residual add, then] ReLU / clip");
            return 0;
        }
    return 1;
}

static int acc_eager_mmq(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    if (p->kind != aiednn_matmul || !mm_int8_problem(p)) {
        snprintf(why, cap, "only s8 x s8 matmul");
        return 0;
    }
    int res;
    if (!rt_eager_q_epilogue(p, p->md[2].dims[p->md[2].ndims - 1], &res, why, cap))
        return 0;
    mm_geo g;
    mm_geometry(p, &g);
    mm_fold_batch(&g);
    rt_eager_q q;
    if (g.batch != 1 || !rt_eager_q_find(g.M, g.N, g.K, res, &q)) {
        snprintf(why, cap, "no %s instance with K >= %lld (or a batched matmul)", res ? "gemm_qr" : "gemm_q",
                 (long long)g.K);
        return 0;
    }
    return 1;
}

static void *g_eq_a, *g_eq_w, *g_eq_c, *g_eq_full;
static size_t g_eq_a_cap, g_eq_w_cap, g_eq_c_cap, g_eq_full_cap;

/* W' rows of channels n0 .. n0 + q->N - 1: the weights (W as [N][K]: nk_rows, else [K][N]), zero past K, each
 * channel's requant words in the first chunk past q->K; rows past N stay 0 (their codes are never read) */
static void rt_eager_q_pack_w(const rt_epi *E, const int8_t *W, int64_t N, int64_t K, int nk_rows, int64_t n0,
                              const rt_eager_q *q, int8_t *wq) {
    const int64_t ld = rt_eager_q_ld(q);
    memset(wq, 0, (size_t)(q->N * ld));
    for (int64_t i = 0; i < q->N && n0 + i < N; i++) {
        const int64_t n = n0 + i;
        int8_t *row = wq + i * ld;
        for (int64_t k = 0; k < K; k++)
            row[k] = nk_rows ? W[n * K + k] : W[k * N + n];
        const uint64_t B = (uint64_t)E->fx_b[n], R = q->res ? (uint64_t)E->fx_rm[n] : 0;
        const int32_t words[RT_EAGER_QR_WORDS] = {E->fx_m[n],
                                                  (int32_t)(uint32_t)B,
                                                  (int32_t)(uint32_t)(B >> 32),
                                                  E->fx_sh[n],
                                                  (int32_t)E->dzp,
                                                  E->fx_lo,
                                                  E->fx_hi,
                                                  q->res ? E->fx_rzp : 0,
                                                  (int32_t)(uint32_t)R,
                                                  (int32_t)(uint32_t)(R >> 32),
                                                  E->fx_rdt == aiednn_u8,
                                                  0};
        memcpy(row + q->K, words, (size_t)(q->res ? RT_EAGER_QR_WORDS : RT_EAGER_Q_WORDS) * sizeof(int32_t));
    }
}

/* one launch of the instance on the packed operands */
static int rt_eager_q_launch(const rt_impl *self, const rt_eager_q *q, int8_t *a, int8_t *w, int8_t *c) {
    const int64_t ld = rt_eager_q_ld(q);
    const int64_t a_shape[2] = {q->M, ld}, w_shape[2] = {q->N, ld}, c_shape[2] = {q->M, q->N};
    aiek_tensor t[3] = {{a, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(q->M * ld)},
                        {w, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(q->N * ld)},
                        {c, 2, AIEK_INT, 8, 1, c_shape, (uint64_t)(q->M * q->N)}};
    const uint64_t te = rt_ticks();
    const int rc = q->g->entry(t, 3);
    ((rt_impl *)self)->t_entry += rt_ticks() - te;
    ((rt_impl *)self)->launches++;
    return rc;
}

/* C [M][N] codes = requant(A [M][K] . W^T [+ R [M][N]]) on instance q: channel slices x row chunks */
static int rt_eager_q_gemm(const rt_impl *self, const rt_epi *E, const rt_eager_q *q, int64_t M, int64_t N, int64_t K,
                           const int8_t *A, const int8_t *W, int nk_rows, const int8_t *R, int8_t *C) {
    const int64_t ld = rt_eager_q_ld(q);
    int8_t *a = (int8_t *)scratch(&g_eq_a, &g_eq_a_cap, (size_t)(q->M * ld));
    int8_t *w = (int8_t *)scratch(&g_eq_w, &g_eq_w_cap, (size_t)(q->N * ld));
    int8_t *c = (int8_t *)scratch(&g_eq_c, &g_eq_c_cap, (size_t)(q->M * q->N));
    if (!a || !w || !c)
        return AIEK_ERR_ALLOC;
    for (int64_t n0 = 0; n0 < N; n0 += q->N) {
        const int64_t nn = N - n0 < q->N ? N - n0 : q->N;
        uint64_t t0 = rt_ticks();
        rt_eager_q_pack_w(E, W, N, K, nk_rows, n0, q, w);
        g_phase[RT_PH_BPACK] += rt_ticks() - t0;
        for (int64_t m0 = 0; m0 < M; m0 += q->M) {
            const int64_t mm = M - m0 < q->M ? M - m0 : q->M;
            t0 = rt_ticks();
            memset(a, 0, (size_t)(q->M * ld));
            for (int64_t i = 0; i < mm; i++) {
                memcpy(a + i * ld, A + (m0 + i) * K, (size_t)K);
                if (R) /* row's residual codes of this slice's channels */
                    memcpy(a + i * ld + q->K + q->TK, R + (m0 + i) * N + n0, (size_t)nn);
            }
            g_phase[RT_PH_APACK] += rt_ticks() - t0;
            const int rc = rt_eager_q_launch(self, q, a, w, c);
            if (rc != AIEK_OK)
                return rc;
            t0 = rt_ticks();
            for (int64_t i = 0; i < mm; i++)
                memcpy(C + (m0 + i) * N + n0, c + i * q->N, (size_t)nn);
            g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
        }
    }
    return AIEK_OK;
}

static int run_eager_mmq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    rt_epi E;
    rt_epi_init(&E, p, ptr);
    char why[8];
    int res;
    if (!E.fx || !rt_eager_q_epilogue(p, p->md[2].dims[p->md[2].ndims - 1], &res, why, sizeof why) ||
        res != (E.fx_r != nullptr))             /* acc_add (zp+) is folded into fx_b */
        return run_ref_matmul(self, prim, ptr); /* the values ruled the fixed-point form out: the same bytes on ref: */
    mm_geo g;
    mm_geometry(p, &g);
    mm_fold_batch(&g);
    rt_eager_q q;
    if (!rt_eager_q_find(g.M, g.N, g.K, res, &q))
        return AIEK_ERR_PARAM;
    return rt_eager_q_gemm(self, &E, &q, g.M, g.N, g.K, (const int8_t *)ptr[0], (const int8_t *)ptr[1], p->transposed,
                           res ? (const int8_t *)E.fx_r : nullptr, (int8_t *)ptr[2]);
}

/* ── eager:convq -- a convolution on a gemm_q / gemm_qr instance: M = N * Ho * Wo (the batch stacked),
 * K = C * KH * KW, N = O; the im2col rows are the (ci, kh, kw) order of the weights' own [O][C][KH][KW] rows ── */

static int acc_eager_convq(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    if (!conv_int8_problem(p) || p->src_zp || rt_prob_has_fmt(p)) {
        snprintf(why, cap, "an s8 x s8 dense-layout convolution without a src zero point");
        return 0;
    }
    conv_geo c;
    conv_geometry(p, 0, &c);
    if (c.G != 1) {
        snprintf(why, cap, "a grouped convolution");
        return 0;
    }
    int res;
    if (!rt_eager_q_epilogue(p, c.O, &res, why, cap))
        return 0;
    rt_eager_q q;
    if (!rt_eager_q_find(c.N * c.Ho * c.Wo, c.O, c.C * c.KH * c.KW, res, &q)) {
        snprintf(why, cap, "no %s instance with K >= %lld", res ? "gemm_qr" : "gemm_q", (long long)(c.C * c.KH * c.KW));
        return 0;
    }
    return 1;
}

static void *g_eq_r;
static size_t g_eq_r_cap;

static int run_eager_convq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    rt_epi E;
    rt_epi_init(&E, p, ptr);
    conv_geo c;
    conv_geometry(p, 0, &c);
    char why[8];
    int res;
    if (!E.fx || !rt_eager_q_epilogue(p, c.O, &res, why, sizeof why) || res != (E.fx_r != nullptr))
        return run_ref_conv(self, prim, ptr); /* the values ruled the fixed-point form out: the same bytes on ref: */
    const int64_t HoWo = c.Ho * c.Wo, M = c.N * HoWo, K = c.C * c.KH * c.KW;
    rt_eager_q q;
    if (!rt_eager_q_find(M, c.O, K, res, &q))
        return AIEK_ERR_PARAM;
    int8_t *A = (int8_t *)scratch(&g_col, &g_col_cap, (size_t)(M * K)); /* im2col of every image, [M][K] */
    int8_t *C = (int8_t *)scratch(&g_eq_full, &g_eq_full_cap, (size_t)(M * c.O));
    int8_t *R = res ? (int8_t *)scratch(&g_eq_r, &g_eq_r_cap, (size_t)(M * c.O)) : nullptr;
    if (!A || !C || (res && !R))
        return AIEK_ERR_ALLOC;
    const uint64_t t0 = rt_ticks();
    for (int64_t n = 0; n < c.N; n++) {
        im2col_pack(&c, (const int8_t *)ptr[0] + n * c.C * c.H * c.W, A + n * HoWo * K);
        if (res) /* the skip is NCHW: pixel-major [M][O] for the slices */
            for (int64_t o = 0; o < c.O; o++) {
                const int8_t *r = (const int8_t *)E.fx_r + (n * c.O + o) * HoWo;
                for (int64_t m = 0; m < HoWo; m++)
                    R[(n * HoWo + m) * c.O + o] = r[m];
            }
    }
    g_phase[RT_PH_IM2COL] += rt_ticks() - t0;
    const int rc = rt_eager_q_gemm(self, &E, &q, M, c.O, K, A, (const int8_t *)ptr[1], 1, R, C);
    if (rc != AIEK_OK)
        return rc;
    const uint64_t tu = rt_ticks();
    int8_t *dst = (int8_t *)ptr[2]; /* [M][O] codes -> [n][o][Ho*Wo] */
    for (int64_t n = 0; n < c.N; n++)
        for (int64_t o = 0; o < c.O; o++) {
            int8_t *drow = dst + (n * c.O + o) * HoWo;
            const int8_t *src = C + n * HoWo * c.O + o;
            for (int64_t m = 0; m < HoWo; m++)
                drow[m] = src[m * c.O];
        }
    g_phase[RT_PH_CUNPACK] += rt_ticks() - tu;
    return AIEK_OK;
}

/* ── eager:pool -- 8-bit max / average pooling (pool instances, aie:eager_pool__MW_KK_TR_AVG_U8) ────────────
 * One row per window: its KH x KW taps (padding taps hold the type minimum for max, 0 for average), then the
 * divisor word (k*k or the valid count) -- src/aietensorop/eager/pool_eager_ref.h. The core's max / round-half-
 * even quotient equals ref:'s rt_st(nearbyint) of the double quotient, so the codes are exact. Windows run in
 * chunks of the instance's MW, one launch each. */

typedef struct {
    const aiednn_impl_t *g;
    int64_t MW, KK;
    int avg, u8;
} rt_eager_pool;

static int rt_eager_pool_of(const aiednn_impl_t *g, rt_eager_pool *q) {
    long long mw, kk, tr;
    int avg, u8;
    if (g->kind != aiednn_pooling || g->nargs != 3 ||
        sscanf(g->name, "aie:eager_pool__%lld_%lld_%lld_%d_%d", &mw, &kk, &tr, &avg, &u8) != 5)
        return 0;
    if (g->args[0].ndims != 2 || g->args[0].dims[0] != mw || g->args[0].dims[1] != kk || kk < 8 || kk % 4)
        return 0;
    q->g = g, q->MW = mw, q->KK = kk, q->avg = avg, q->u8 = u8;
    return 1;
}

/* the instance of the layer's algorithm and type with the narrowest window >= kk (its extra taps hold padding),
 * then the most windows per launch */
static int rt_eager_pool_find(int64_t kk, int avg, int u8, rt_eager_pool *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_pool q;
        if (!rt_eager_pool_of(&aiednn_impl_list[i], &q) || q.KK < kk || q.avg != avg || q.u8 != u8)
            continue;
        if (!found || q.KK < out->KK || (q.KK == out->KK && q.MW > out->MW))
            *out = q, found = 1;
    }
    return found;
}

static int acc_eager_pool(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    if (p->prop != RT_FWD_INF || s->ndims != 4 || (s->data_type != aiednn_s8 && s->data_type != aiednn_u8) ||
        d->data_type != s->data_type) {
        snprintf(why, cap, "only 2-D inference pooling, s8 -> s8 or u8 -> u8");
        return 0;
    }
    const int64_t taps = p->pool.kernel[0] * p->pool.kernel[1], kk = (taps + 3) / 4 * 4 + 4;
    rt_eager_pool q;
    if (rt_eager_pool_find(kk, p->alg != aiednn_pooling_max, s->data_type == aiednn_u8, &q))
        return 1;
    snprintf(why, cap, "no eager pool instance with >= %lld-byte windows, %s, %s", (long long)kk,
             p->alg != aiednn_pooling_max ? "avg" : "max", s->data_type == aiednn_u8 ? "u8" : "s8");
    return 0;
}

static void *g_ep_x, *g_ep_z, *g_ep_y;
static size_t g_ep_x_cap, g_ep_z_cap, g_ep_y_cap;

/* window w (in NC x Ho x Wo order) -> its row */
static void rt_eager_pool_row(const rt_prob *p, const uint8_t *x, int64_t w, int8_t *row, int64_t kk) {
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const aiednn_pool_params_t *pp = &p->pool;
    const int64_t H = s->dims[2], W = s->dims[3], Ho = d->dims[2], Wo = d->dims[3];
    const int64_t nc = w / (Ho * Wo), oh = w / Wo % Ho, ow = w % Wo;
    const int max = p->alg == aiednn_pooling_max;
    const uint8_t pad = max && s->data_type == aiednn_s8 ? 0x80 : 0;
    memset(row, pad, (size_t)(kk - 4));
    int32_t cnt = 0, t = 0;
    for (int64_t kh = 0; kh < pp->kernel[0]; kh++)
        for (int64_t kw = 0; kw < pp->kernel[1]; kw++, t++) {
            const int64_t ih = oh * pp->strides[0] - pp->padding_l[0] + kh * (pp->dilates[0] + 1);
            const int64_t iw = ow * pp->strides[1] - pp->padding_l[1] + kw * (pp->dilates[1] + 1);
            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                continue;
            row[t] = (int8_t)x[(nc * H + ih) * W + iw];
            cnt++;
        }
    const int32_t div = max ? 0 : p->alg == aiednn_pooling_avg_include_padding ? (int32_t)t : cnt;
    memcpy(row + kk - 4, &div, 4);
}

static int run_eager_pool(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    const int64_t taps = p->pool.kernel[0] * p->pool.kernel[1], kk = (taps + 3) / 4 * 4 + 4;
    const int avg = p->alg != aiednn_pooling_max, u8 = p->md[0].data_type == aiednn_u8;
    rt_eager_pool q;
    if (!rt_eager_pool_find(kk, avg, u8, &q))
        return AIEK_ERR_PARAM;
    const int64_t total = md_numel(&p->md[1]);
    int8_t *x = (int8_t *)scratch(&g_ep_x, &g_ep_x_cap, (size_t)(q.MW * q.KK));
    int8_t *z = (int8_t *)scratch(&g_ep_z, &g_ep_z_cap, (size_t)(16 * 4 * q.KK));
    int8_t *y = (int8_t *)scratch(&g_ep_y, &g_ep_y_cap, (size_t)(q.MW * 4));
    if (!x || !z || !y)
        return AIEK_ERR_ALLOC;
    memset(z, 0, (size_t)(16 * 4 * q.KK));
    const int64_t x_shape[2] = {q.MW, q.KK}, z_shape[2] = {16, 4 * q.KK}, y_shape[2] = {q.MW, 4};
    for (int64_t base = 0; base < total; base += q.MW) {
        const int64_t n = total - base < q.MW ? total - base : q.MW;
        const uint64_t t0 = rt_ticks();
        for (int64_t w = 0; w < q.MW; w++) /* rows past the layer's windows: any value, never read back */
            if (w < n)
                rt_eager_pool_row(p, (const uint8_t *)ptr[0], base + w, x + w * q.KK, q.KK);
            else
                memset(x + w * q.KK, 0, (size_t)q.KK);
        g_phase[RT_PH_DWPACK] += rt_ticks() - t0;
        aiek_tensor t[3] = {{x, 2, AIEK_INT, 8, 1, x_shape, (uint64_t)(q.MW * q.KK)},
                            {z, 2, AIEK_INT, 8, 1, z_shape, (uint64_t)(16 * 4 * q.KK)},
                            {y, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)(q.MW * 4)}};
        const uint64_t te = rt_ticks();
        const int rc = q.g->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        const uint64_t tu = rt_ticks();
        int8_t *dst = (int8_t *)ptr[1] + base;
        for (int64_t w = 0; w < n; w++)
            dst[w] = y[w * 4];
        g_phase[RT_PH_CUNPACK] += rt_ticks() - tu;
    }
    return AIEK_OK;
}

/* ── eager:dwq -- depthwise convolution + requant on the core (dwconv_q instances,
 * aie:eager_dwconv_q__M_C_TP_TM) ──────────────────────────────────────────────────────────────────────────
 * The host gathers each output pixel's taps per channel (any kernel size / stride / padding / dilation), the core
 * forms the int32 sums and the fixed-point requant (rt_fx_init's, exact), 8-bit codes come back [pixel][channel].
 * A layer runs as channel slices of the instance's C x pixel chunks of its M. Layout:
 * src/aietensorop/eager/dwconv_q_eager_ref.h. */

typedef struct {
    const aiednn_impl_t *g;
    int64_t M, C, TP;
} rt_eager_dw;

static int rt_eager_dw_of(const aiednn_impl_t *g, rt_eager_dw *q) {
    long long m, c, tp, tm;
    if (g->kind != aiednn_matmul || g->nargs != 3 ||
        sscanf(g->name, "aie:eager_dwconv_q__%lld_%lld_%lld_%lld", &m, &c, &tp, &tm) != 4 || c % 16 || tp < 12)
        return 0;
    q->g = g, q->M = m, q->C = c, q->TP = tp;
    return 1;
}

static int64_t rt_eager_dw_tp(const conv_geo *c) {
    const int64_t t = (c->KH * c->KW + 3) / 4 * 4;
    return t < 12 ? 12 : t;
}

/* the instance with the smallest TP covering the layer's taps (then the largest M x C: fewest launches) */
static int rt_eager_dw_find(int64_t tp, rt_eager_dw *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_dw q;
        if (!rt_eager_dw_of(&aiednn_impl_list[i], &q) || q.TP < tp)
            continue;
        if (!found || q.TP < out->TP || (q.TP == out->TP && q.M * q.C > out->M * out->C))
            *out = q, found = 1;
    }
    return found;
}

static int acc_eager_dwq(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    if (!conv_int8_problem(p) || p->src_zp || rt_prob_has_fmt(p)) {
        snprintf(why, cap, "an s8 x s8 dense-layout convolution without a src zero point");
        return 0;
    }
    conv_geo c;
    conv_geometry(p, 0, &c);
    if (c.G != c.C || c.O != c.C) {
        snprintf(why, cap, "not a depthwise convolution (groups == channels, one output per channel)");
        return 0;
    }
    int res;
    if (!rt_eager_q_epilogue(p, c.O, &res, why, cap))
        return 0;
    rt_eager_dw q;
    if (res || !rt_eager_dw_find(rt_eager_dw_tp(&c), &q)) {
        snprintf(why, cap, res ? "a residual add (dwconv_q has none)" : "no dwconv_q instance with %lld taps",
                 (long long)rt_eager_dw_tp(&c));
        return 0;
    }
    return 1;
}

static void *g_ed_a, *g_ed_w, *g_ed_y;
static size_t g_ed_a_cap, g_ed_w_cap, g_ed_y_cap;

/* W rows of channels c0 .. c0 + q->C - 1: taps at the start of chunk (c - c0) / 4, requant words after them */
static void rt_eager_dw_pack_w(const rt_epi *E, const conv_geo *c, const int8_t *wt, int64_t c0, const rt_eager_dw *q,
                               int8_t *w) {
    const int64_t TK = 4 * q->TP, K = q->C / 4 * TK, taps = c->KH * c->KW;
    memset(w, 0, (size_t)(q->C * K));
    for (int64_t i = 0; i < q->C && c0 + i < c->C; i++) {
        int8_t *row = w + i * K + i / 4 * TK;
        memcpy(row, wt + (c0 + i) * taps, (size_t)taps);
        const uint64_t B = (uint64_t)E->fx_b[c0 + i];
        const int32_t words[RT_EAGER_Q_WORDS] = {E->fx_m[c0 + i],
                                                 (int32_t)(uint32_t)B,
                                                 (int32_t)(uint32_t)(B >> 32),
                                                 E->fx_sh[c0 + i],
                                                 (int32_t)E->dzp,
                                                 E->fx_lo,
                                                 E->fx_hi,
                                                 0};
        memcpy(row + q->TP, words, sizeof words);
    }
}

/* A rows of pixels p0 .. p0 + q->M - 1 for channels c0 ..: taps outside the image are 0 (no src zero point) */
static void rt_eager_dw_pack_a(const conv_geo *c, const int8_t *x, int64_t p0, int64_t c0, const rt_eager_dw *q,
                               int8_t *a) {
    const int64_t TK = 4 * q->TP, K = q->C / 4 * TK, HoWo = c->Ho * c->Wo, P = c->N * HoWo;
    memset(a, 0, (size_t)(q->M * K));
    for (int64_t m = 0; m < q->M && p0 + m < P; m++) {
        const int64_t n = (p0 + m) / HoWo, oh = (p0 + m) % HoWo / c->Wo, ow = (p0 + m) % c->Wo;
        for (int64_t kh = 0; kh < c->KH; kh++) {
            const int64_t ih = oh * c->s[0] - c->pl[0] + kh * c->d[0];
            if (ih < 0 || ih >= c->H)
                continue;
            for (int64_t kw = 0; kw < c->KW; kw++) {
                const int64_t iw = ow * c->s[1] - c->pl[1] + kw * c->d[1];
                if (iw < 0 || iw >= c->W)
                    continue;
                const int8_t *px = x + (n * c->C + c0) * c->H * c->W + ih * c->W + iw;
                int8_t *dst = a + m * K + kh * c->KW + kw;
                for (int64_t i = 0; i < q->C && c0 + i < c->C; i++)
                    dst[i / 4 * TK + i % 4 * q->TP] = px[i * c->H * c->W];
            }
        }
    }
}

static int run_eager_dwq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    rt_epi E;
    rt_epi_init(&E, p, ptr);
    conv_geo c;
    conv_geometry(p, 0, &c);
    char why[8];
    int res;
    if (!E.fx || !rt_eager_q_epilogue(p, c.O, &res, why, sizeof why) || res)
        return run_ref_conv(self, prim, ptr); /* the values ruled the fixed-point form out: the same bytes on ref: */
    rt_eager_dw q;
    if (!rt_eager_dw_find(rt_eager_dw_tp(&c), &q))
        return AIEK_ERR_PARAM;
    const int64_t K = q.C / 4 * 4 * q.TP, HoWo = c.Ho * c.Wo, P = c.N * HoWo;
    int8_t *a = (int8_t *)scratch(&g_ed_a, &g_ed_a_cap, (size_t)(q.M * K));
    int8_t *w = (int8_t *)scratch(&g_ed_w, &g_ed_w_cap, (size_t)(q.C * K));
    int8_t *y = (int8_t *)scratch(&g_ed_y, &g_ed_y_cap, (size_t)(q.M * q.C));
    if (!a || !w || !y)
        return AIEK_ERR_ALLOC;
    const int64_t a_shape[2] = {q.M, K}, w_shape[2] = {q.C, K}, y_shape[2] = {q.M, q.C};
    int8_t *dst = (int8_t *)ptr[2];
    for (int64_t c0 = 0; c0 < c.C; c0 += q.C) {
        rt_eager_dw_pack_w(&E, &c, (const int8_t *)ptr[1], c0, &q, w);
        for (int64_t p0 = 0; p0 < P; p0 += q.M) {
            const uint64_t t0 = rt_ticks();
            rt_eager_dw_pack_a(&c, (const int8_t *)ptr[0], p0, c0, &q, a);
            g_phase[RT_PH_DWPACK] += rt_ticks() - t0;
            aiek_tensor t[3] = {{a, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(q.M * K)},
                                {w, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(q.C * K)},
                                {y, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)(q.M * q.C)}};
            const uint64_t te = rt_ticks();
            const int rc = q.g->entry(t, 3);
            ((rt_impl *)self)->t_entry += rt_ticks() - te;
            ((rt_impl *)self)->launches++;
            if (rc != AIEK_OK)
                return rc;
            const uint64_t tu = rt_ticks();
            for (int64_t m = 0; m < q.M && p0 + m < P; m++) { /* [pixel][channel] -> [n][c][Ho*Wo] */
                const int64_t n = (p0 + m) / HoWo, pix = (p0 + m) % HoWo;
                for (int64_t i = 0; i < q.C && c0 + i < c.C; i++)
                    dst[(n * c.C + c0 + i) * HoWo + pix] = y[m * q.C + i];
            }
            g_phase[RT_PH_CUNPACK] += rt_ticks() - tu;
        }
    }
    return AIEK_OK;
}

/* ── eager:quant -- f32 -> s8 / u8 identity reorders (quant instances, aie:eager_quant__M_N_U8_TM) ─────────
 * The quantize reorder's arithmetic, nearbyint(f / scale) + zero point, clamped -- on the core as an exact integer
 * division of the two floats' mantissas (src/aietensorop/eager/quant_eager_ref.h), so the codes are ref:'s. The
 * tensor runs in chunks of the instance's M x N elements. */

typedef struct {
    const aiednn_impl_t *g;
    int64_t M, N;
    int u8;
} rt_eager_quant;

static int rt_eager_quant_of(const aiednn_impl_t *g, rt_eager_quant *q) {
    long long m, n, tm;
    int u8;
    if (g->nargs != 3 || sscanf(g->name, "aie:eager_quant__%lld_%lld_%d_%lld", &m, &n, &u8, &tm) != 4)
        return 0;
    q->g = g, q->M = m, q->N = n, q->u8 = u8;
    return 1;
}

static int rt_eager_quant_find(int u8, rt_eager_quant *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_quant q;
        if (rt_eager_quant_of(&aiednn_impl_list[i], &q) && q.u8 == u8 && (!found || q.M * q.N > out->M * out->N))
            *out = q, found = 1;
    }
    return found;
}

static int acc_eager_quant(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    if (p->kind != aiednn_reorder || p->has_attr || s->data_type != aiednn_f32 ||
        (d->data_type != aiednn_s8 && d->data_type != aiednn_u8) || rt_prob_has_fmt(p)) {
        snprintf(why, cap, "only an f32 -> s8 / u8 reorder without an attr");
        return 0;
    }
    for (int k = 0; k < d->ndims; k++)
        if (p->perm[k] != k) {
            snprintf(why, cap, "identity permutation only");
            return 0;
        }
    if (!(p->scale >= FLT_MIN && p->scale <= FLT_MAX)) { /* the core's division takes a normal, positive scale */
        snprintf(why, cap, "the scale must be a normal positive float");
        return 0;
    }
    rt_eager_quant q;
    if (!rt_eager_quant_find(d->data_type == aiednn_u8, &q)) {
        snprintf(why, cap, "no quant instance for a %s dst", d->data_type == aiednn_u8 ? "u8" : "s8");
        return 0;
    }
    return 1;
}

static void *g_eqt_a, *g_eqt_w, *g_eqt_y;
static size_t g_eqt_a_cap, g_eqt_w_cap, g_eqt_y_cap;

static int run_eager_quant(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    rt_eager_quant q;
    if (!rt_eager_quant_find(p->md[1].data_type == aiednn_u8, &q))
        return AIEK_ERR_PARAM;
    const int64_t n = md_numel(&p->md[1]), per = q.M * q.N;
    int8_t *a = (int8_t *)scratch(&g_eqt_a, &g_eqt_a_cap, (size_t)(per * 4));
    int8_t *w = (int8_t *)scratch(&g_eqt_w, &g_eqt_w_cap, (size_t)(q.N * 4 * q.N));
    int8_t *y = (int8_t *)scratch(&g_eqt_y, &g_eqt_y_cap, (size_t)per);
    if (!a || !w || !y)
        return AIEK_ERR_ALLOC;
    uint32_t sz[2];
    memcpy(&sz[0], &p->scale, 4);
    sz[1] = (uint32_t)p->zero_point;
    for (int64_t r = 0; r < q.N; r++) /* {scale, zp} at the start of every chunk of every row */
        for (int64_t j = 0; j < 4; j++)
            memcpy(w + r * 4 * q.N + j * q.N, sz, sizeof sz);
    const int64_t a_shape[2] = {q.M, 4 * q.N}, w_shape[2] = {q.N, 4 * q.N}, y_shape[2] = {q.M, q.N};
    for (int64_t e0 = 0; e0 < n; e0 += per) {
        const int64_t cnt = n - e0 < per ? n - e0 : per;
        uint64_t t0 = rt_ticks();
        memcpy(a, (const float *)ptr[0] + e0, (size_t)(cnt * 4));
        memset(a + cnt * 4, 0, (size_t)((per - cnt) * 4));
        g_phase[RT_PH_APACK] += rt_ticks() - t0;
        aiek_tensor t[3] = {{a, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(per * 4)},
                            {w, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(q.N * 4 * q.N)},
                            {y, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)per}};
        const uint64_t te = rt_ticks();
        const int rc = q.g->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        t0 = rt_ticks();
        memcpy((int8_t *)ptr[1] + e0, y, (size_t)cnt);
        g_phase[RT_PH_CUNPACK] += rt_ticks() - t0;
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_EAGER_H */
