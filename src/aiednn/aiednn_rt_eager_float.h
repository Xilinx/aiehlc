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

#endif /* AIEDNN_RT_EAGER_FLOAT_H */
