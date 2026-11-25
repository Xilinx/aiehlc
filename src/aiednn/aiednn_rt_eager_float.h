/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_eager_float.h -- eager mode's float matmul on the gemm_bf16 template (aie:eager_gemm_bf16__M_N_K_TK_TM,
 * src/aietensorop/eager/gemm_bf16_eager_kernel.cc): bf16 x bf16 -> f32 on the core.
 *   eager:bf16    bf16 x bf16 matmul;
 *   eager:bf16x3  f32 x f32 matmul as fixed mode's bf16x3 split (aiednn_rt_bf16.h: bf16x3_rows, depth 6 K).
 * The problem runs as tiles of the instance's (M, N, K) block, K blocks summed in f32 on the host, every element
 * stored through rt_epi_put (ref:matmul's epilogue: bias, post-ops, any dst). Same contract as fixed mode's
 * bf16blk: / bf16x3: -- within RT_BF16_RTOL * sqrt(K) of ref:matmul's double sum relative to sum |a b|. Since a
 * float matmul on the AIE is what the f32 composites (gemm+conv, gemm+deconv, gemm+rnn, gemm+sdpa, gemm+gmlp,
 * gemm+linattn) wait for, they reach the AIE through these impls in eager builds too.
 *
 * Needs, from the including TU: aiednn_rt_bf16.h (bf16_accept, bf16x3_rows), aiednn_rt_gemm.h (mm_geometry).
 ******************************************************************************/
#ifndef AIEDNN_RT_EAGER_FLOAT_H
#define AIEDNN_RT_EAGER_FLOAT_H

typedef struct {
    const aiednn_impl_t *g;
    int64_t M, N, K;
} rt_eager_bf;

static int rt_eager_bf_of(const aiednn_impl_t *g, rt_eager_bf *q) {
    long long m, n, k, tk, tm;
    if (g->nargs != 3 || sscanf(g->name, "aie:eager_gemm_bf16__%lld_%lld_%lld_%lld_%lld", &m, &n, &k, &tk, &tm) != 5)
        return 0;
    q->g = g, q->M = m, q->N = n, q->K = k;
    return 1;
}

/* the instance needing the fewest launches for an M x N x K (K: bf16 depth) problem, then the least padding */
static int rt_eager_bf_find(int64_t M, int64_t N, int64_t K, rt_eager_bf *out) {
    int found = 0;
    int64_t bl = 0, bv = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_bf q;
        if (!rt_eager_bf_of(&aiednn_impl_list[i], &q))
            continue;
        const int64_t l = ((M + q.M - 1) / q.M) * ((N + q.N - 1) / q.N) * ((K + q.K - 1) / q.K),
                      v = l * q.M * q.N * q.K;
        if (!found || l < bl || (l == bl && v < bv))
            *out = q, found = 1, bl = l, bv = v;
    }
    return found;
}

static int eager_bf_accept(const rt_prob *p, aiednn_data_type_t dt, char *why, int cap) {
    if (!bf16_accept(p, dt, why, cap))
        return 0;
    rt_eager_bf q;
    if (!rt_eager_bf_find(1, 1, 1, &q)) {
        snprintf(why, cap, "no gemm_bf16 instance linked");
        return 0;
    }
    return 1;
}
static int acc_eager_bf16(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    return eager_bf_accept(p, aiednn_bf16, why, cap);
}
static int acc_eager_bf16x3(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    return eager_bf_accept(p, aiednn_f32, why, cap);
}

static void *g_ebf_a, *g_ebf_w, *g_ebf_c, *g_ebf_s, *g_ebf_t, *g_ebf_a3, *g_ebf_w3;
static size_t g_ebf_a_cap, g_ebf_w_cap, g_ebf_c_cap, g_ebf_s_cap, g_ebf_t_cap, g_ebf_a3_cap, g_ebf_w3_cap;

/* one M x N tile of instance q: K blocks launched and summed in f32 into s [q.M][q.N] */
static int eager_bf_tile(const rt_impl *self, const rt_eager_bf *q, const uint16_t *A16, const uint16_t *W16,
                         int64_t KK, int64_t m0, int64_t mn, int64_t n0, int64_t nn, float *s) {
    uint16_t *a = (uint16_t *)scratch(&g_ebf_a, &g_ebf_a_cap, (size_t)(q->M * q->K) * 2);
    uint16_t *w = (uint16_t *)scratch(&g_ebf_w, &g_ebf_w_cap, (size_t)(q->N * q->K) * 2);
    float *c = (float *)scratch(&g_ebf_c, &g_ebf_c_cap, (size_t)(q->M * q->N) * 4);
    if (!a || !w || !c)
        return AIEK_ERR_ALLOC;
    const int64_t a_shape[2] = {q->M, 2 * q->K}, w_shape[2] = {q->N, 2 * q->K}, c_shape[2] = {q->M, q->N};
    aiek_tensor t[3] = {{a, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(q->M * q->K * 2)},
                        {w, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(q->N * q->K * 2)},
                        {c, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)(q->M * q->N * 4)}};
    memset(s, 0, (size_t)(q->M * q->N) * 4);
    for (int64_t k0 = 0; k0 < KK; k0 += q->K) {
        const int64_t kn = KK - k0 < q->K ? KK - k0 : q->K;
        const uint64_t tp = rt_ticks();
        memset(a, 0, (size_t)(q->M * q->K) * 2);
        memset(w, 0, (size_t)(q->N * q->K) * 2);
        for (int64_t m = 0; m < mn; m++)
            memcpy(a + m * q->K, A16 + (m0 + m) * KK + k0, (size_t)kn * 2);
        for (int64_t n = 0; n < nn; n++)
            memcpy(w + n * q->K, W16 + (n0 + n) * KK + k0, (size_t)kn * 2);
        g_phase[RT_PH_APACK] += rt_ticks() - tp;
        const uint64_t te = rt_ticks();
        const int rc = q->g->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        ((rt_impl *)self)->launches++;
        if (rc != AIEK_OK)
            return rc;
        for (int64_t i = 0; i < q->M * q->N; i++)
            s[i] += c[i];
    }
    return AIEK_OK;
}

static int run_eager_bf_common(const rt_impl *self, aiednn_primitive *prim, void *const *ptr, int split) {
    const rt_prob *p = &prim->prob;
    rt_epi E;
    rt_epi_init(&E, p, ptr);
    mm_geo g;
    mm_geometry(p, &g);
    const int64_t es = split ? 4 : 2, KK = split ? 6 * g.K : g.K;
    rt_eager_bf q;
    if (!rt_eager_bf_find(g.M, g.N, KK, &q))
        return AIEK_ERR_PARAM;
    uint8_t *wtr = p->transposed ? nullptr : (uint8_t *)scratch(&g_ebf_t, &g_ebf_t_cap, (size_t)(g.N * g.K * es));
    float *s = (float *)scratch(&g_ebf_s, &g_ebf_s_cap, (size_t)(q.M * q.N) * 4);
    if ((!p->transposed && !wtr) || !s)
        return AIEK_ERR_ALLOC;
    for (int64_t b = 0; b < g.batch; b++) {
        const uint8_t *A = (const uint8_t *)ptr[0] + bcast_off(&g.bs, &g.bd, b) * g.M * g.K * es;
        const uint8_t *W = (const uint8_t *)ptr[1] + bcast_off(&g.bw, &g.bd, b) * g.N * g.K * es;
        uint64_t ts = rt_ticks();
        if (!p->transposed) { /* [K][N] -> [N][K] */
            for (int64_t k = 0; k < g.K; k++)
                for (int64_t n = 0; n < g.N; n++)
                    memcpy(wtr + (n * g.K + k) * es, W + (k * g.N + n) * es, (size_t)es);
            W = wtr;
        }
        const uint16_t *A16 = (const uint16_t *)A, *W16 = (const uint16_t *)W;
        if (split) { /* A' [M][6K], W' [N][6K] */
            uint16_t *a3 = (uint16_t *)scratch(&g_ebf_a3, &g_ebf_a3_cap, (size_t)(g.M * KK) * 2);
            uint16_t *w3 = (uint16_t *)scratch(&g_ebf_w3, &g_ebf_w3_cap, (size_t)(g.N * KK) * 2);
            if (!a3 || !w3)
                return AIEK_ERR_ALLOC;
            bf16x3_rows((const float *)A, g.M, g.K, 0, a3);
            bf16x3_rows((const float *)W, g.N, g.K, 1, w3);
            A16 = a3, W16 = w3;
        }
        g_phase[RT_PH_BPACK] += rt_ticks() - ts;
        for (int64_t m0 = 0; m0 < g.M; m0 += q.M)
            for (int64_t n0 = 0; n0 < g.N; n0 += q.N) {
                const int64_t mn = g.M - m0 < q.M ? g.M - m0 : q.M, nn = g.N - n0 < q.N ? g.N - n0 : q.N;
                const int rc = eager_bf_tile(self, &q, A16, W16, KK, m0, mn, n0, nn, s);
                if (rc != AIEK_OK)
                    return rc;
                ts = rt_ticks();
                for (int64_t m = 0; m < mn; m++)
                    for (int64_t n = 0; n < nn; n++)
                        rt_epi_put(&E, (b * g.M + m0 + m) * g.N + n0 + n, n0 + n, (double)s[m * q.N + n]);
                g_phase[RT_PH_EPI] += rt_ticks() - ts;
            }
    }
    return AIEK_OK;
}

static int run_eager_bf16(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    return run_eager_bf_common(self, prim, ptr, 0);
}
static int run_eager_bf16x3(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    return run_eager_bf_common(self, prim, ptr, 1);
}

/* ── eager:wq -- grouped weight-only quantization on the gemm_wq template (aie:eager_gemm_wq__M_N_K_GK_TM) ─────
 * An s8 / u8 src times s4 / u4 weights with per-(group, column) scales and zero points (attr.wei_scales [K / g, N],
 * wei_zero_points): the core sums each GK chunk exactly in int32 and accumulates scale x sum in f32; a problem group
 * of g = r GK is r chunks with the same scale and zero point. u8 / u4 are shifted to signed on the host (a' = a - 128,
 * w' = w - 8, the zero points with them) and the src zero point's term, -za' sum_g s_g sum_{k in g} (w' - zw'),
 * is added here in double, then the epilogue. K runs in slices of the instance's K, their f32 partials summed in
 * double. Contract: fixed-megakernel wq:'s (f32 sums of exact partials). */

typedef struct {
    const aiednn_impl_t *g;
    int64_t M, N, K, GK;
} rt_eager_wq;

static int rt_eager_wq_of(const aiednn_impl_t *g, rt_eager_wq *q) {
    long long m, n, k, gk, tm;
    if (g->nargs != 3 || sscanf(g->name, "aie:eager_gemm_wq__%lld_%lld_%lld_%lld_%lld", &m, &n, &k, &gk, &tm) != 5)
        return 0;
    q->g = g, q->M = m, q->N = n, q->K = k, q->GK = gk;
    return 1;
}

/* an instance whose GK divides the group size: the largest GK, then the most work per launch */
static int rt_eager_wq_find(int64_t gk, rt_eager_wq *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_wq q;
        if (!rt_eager_wq_of(&aiednn_impl_list[i], &q) || gk % q.GK)
            continue;
        if (!found || q.GK > out->GK || (q.GK == out->GK && q.M * q.N * q.K > out->M * out->N * out->K))
            *out = q, found = 1;
    }
    return found;
}

static int acc_eager_wq(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    const int ss = rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES);
    if (p->kind != aiednn_matmul || ss < 0 || rt_prob_has_fmt(p)) {
        snprintf(why, cap, "matmul with weight group scales only");
        return 0;
    }
    const aiednn_data_type_t sdt = p->md[0].data_type, wdt = p->md[1].data_type;
    if ((sdt != aiednn_s8 && sdt != aiednn_u8) || (wdt != aiednn_s4 && wdt != aiednn_u4) || p->md[1].ndims != 2) {
        snprintf(why, cap, "an 8-bit src x 2-D 4-bit weights");
        return 0;
    }
    mm_geo g;
    mm_geometry(p, &g);
    const int64_t G = p->md[ss].dims[0];
    rt_eager_wq q;
    if (G <= 0 || g.K % G || p->md[ss].ndims != 2 || p->md[ss].dims[1] != g.N || !rt_eager_wq_find(g.K / G, &q)) {
        snprintf(why, cap, "scales [K / g, N] with a group size some gemm_wq instance's GK divides");
        return 0;
    }
    return 1;
}

static void *g_ewq_a, *g_ewq_w, *g_ewq_c, *g_ewq_acc, *g_ewq_cor;
static size_t g_ewq_a_cap, g_ewq_w_cap, g_ewq_c_cap, g_ewq_acc_cap, g_ewq_cor_cap;

static int run_eager_wq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    rt_epi E;
    rt_epi_init(&E, p, ptr);
    mm_geo g;
    mm_geometry(p, &g);
    const int ss = rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES), zs = rt_slot(p, AIEDNN_ARG_ATTR_WEI_ZERO_POINTS);
    const int64_t M = g.M * g.batch, N = g.N, K = g.K, G = p->md[ss].dims[0], gk = K / G;
    rt_eager_wq q;
    if (!rt_eager_wq_find(gk, &q))
        return AIEK_ERR_PARAM;
    const aiednn_data_type_t sdt = p->md[0].data_type, wdt = p->md[1].data_type;
    const int ash = sdt == aiednn_u8 ? 128 : 0, wsh = wdt == aiednn_u4 ? 8 : 0;
    const double za = (double)p->src_zp - ash; /* za' */
    int8_t *a = (int8_t *)scratch(&g_ewq_a, &g_ewq_a_cap, (size_t)(q.M * q.K));
    int8_t *w = (int8_t *)scratch(&g_ewq_w, &g_ewq_w_cap, (size_t)(q.N * q.K));
    float *c = (float *)scratch(&g_ewq_c, &g_ewq_c_cap, (size_t)(q.M * q.N) * 4);
    double *acc = (double *)scratch(&g_ewq_acc, &g_ewq_acc_cap, (size_t)(q.M * q.N) * sizeof(double));
    double *cor = (double *)scratch(&g_ewq_cor, &g_ewq_cor_cap, (size_t)N * sizeof(double));
    if (!a || !w || !c || !acc || !cor)
        return AIEK_ERR_ALLOC;
    const uint8_t *A = (const uint8_t *)ptr[0];
    auto wv = [&](int64_t k, int64_t n) { /* w' */
                                          return (int)rt_ld(wdt, ptr[1], p->transposed ? n * K + k : k * N + n) - wsh;
    };
    auto scale = [&](int64_t gi, int64_t n) { return (float)rt_ld(p->md[ss].data_type, ptr[ss], gi * N + n); };
    auto zero = [&](int64_t gi, int64_t n) {
        return (zs >= 0 ? (int)rt_ld(p->md[zs].data_type, ptr[zs], gi * N + n) : 0) - wsh;
    };
    /* the src zero point's term per column: -za' sum_g s_g sum_{k in g} (w' - zw') */
    for (int64_t n = 0; n < N; n++) {
        double t = 0;
        for (int64_t gi = 0; za != 0.0 && gi < G; gi++) {
            int64_t sw = 0;
            for (int64_t k = gi * gk; k < (gi + 1) * gk; k++)
                sw += wv(k, n) - zero(gi, n);
            t += (double)scale(gi, n) * (double)sw;
        }
        cor[n] = -za * t;
    }
    const int64_t ash_w[2] = {q.M, q.K}, wsh_w[2] = {q.N, q.K}, csh[2] = {q.M, q.N};
    for (int64_t m0 = 0; m0 < M; m0 += q.M)
        for (int64_t n0 = 0; n0 < N; n0 += q.N) {
            const int64_t mn = M - m0 < q.M ? M - m0 : q.M, nn = N - n0 < q.N ? N - n0 : q.N;
            memset(acc, 0, (size_t)(q.M * q.N) * sizeof(double));
            for (int64_t k0 = 0; k0 < K; k0 += q.K) {
                const uint64_t tp = rt_ticks();
                memset(a, 0, (size_t)(q.M * q.K));
                memset(w, 0, (size_t)(q.N * q.K));
                const int64_t kn = K - k0 < q.K ? K - k0 : q.K;
                for (int64_t m = 0; m < mn; m++)
                    for (int64_t k = 0; k < kn; k++) {
                        const uint8_t v = A[(m0 + m) * K + k0 + k];
                        a[m * q.K + k] = (int8_t)((sdt == aiednn_u8 ? (int)v : (int)(int8_t)v) - ash);
                    }
                for (int64_t n = 0; n < nn; n++)
                    for (int64_t j = 0; j * q.GK < kn; j++) { /* chunk j: nibbles, scale, zero point */
                        int8_t *ch = w + n * q.K + j * q.GK;
                        const int64_t kc = k0 + j * q.GK, gi = kc / gk;
                        for (int64_t i = 0; i < q.GK / 2; i++) {
                            const int lo = wv(kc + 2 * i, n0 + n), hi = wv(kc + 2 * i + 1, n0 + n);
                            ch[i] = (int8_t)((lo & 15) | (hi << 4));
                        }
                        const float s = scale(gi, n0 + n);
                        const int32_t zw = zero(gi, n0 + n);
                        memcpy(ch + q.GK / 2, &s, 4), memcpy(ch + q.GK / 2 + 4, &zw, 4);
                    }
                g_phase[RT_PH_APACK] += rt_ticks() - tp;
                aiek_tensor t[3] = {{a, 2, AIEK_INT, 8, 1, ash_w, (uint64_t)(q.M * q.K)},
                                    {w, 2, AIEK_INT, 8, 1, wsh_w, (uint64_t)(q.N * q.K)},
                                    {c, 2, AIEK_FLOAT, 32, 1, csh, (uint64_t)(q.M * q.N) * 4}};
                const uint64_t te = rt_ticks();
                const int rc = q.g->entry(t, 3);
                ((rt_impl *)self)->t_entry += rt_ticks() - te;
                ((rt_impl *)self)->launches++;
                if (rc != AIEK_OK)
                    return rc;
                for (int64_t i = 0; i < q.M * q.N; i++)
                    acc[i] += c[i];
            }
            const uint64_t t0 = rt_ticks();
            for (int64_t m = 0; m < mn; m++)
                for (int64_t n = 0; n < nn; n++)
                    rt_epi_put(&E, (m0 + m) * N + n0 + n, n0 + n, acc[m * q.N + n] + cor[n0 + n]);
            g_phase[RT_PH_EPI] += rt_ticks() - t0;
        }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_EAGER_FLOAT_H */
