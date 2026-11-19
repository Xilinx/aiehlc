/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_wq.h -- grouped weight-only quantization on the AIE (impl "wq:"): an 8-bit src times 8- or
 * 4-bit weights with per-group scales along K (attr.wei_scales [K / g, N], optional wei_zero_points),
 *
 *   y[m][n] = sum_g s_g[n] * sum_{k in g} (a[m][k] - za) * (w[k][n] - zw_g[n])
 *
 * on the GEMM ELF's WQ mode (aie_modes_wq.h, entry wq_blk_s8f32): the core sums each group's int32
 * products exactly and accumulates s_g x partial in f32. The operands go in as int8 -- u8 shifted by
 * 128 -- and everything the symmetric core does not see is corrected here, exactly, per group from row /
 * column sums:  (a' - za')(w' - zw') = a'w' - za' w' - zw' a' + za' zw'  (a' = a - shift, za' = za - shift).
 * Group size g: a multiple of 64 up to 64 * 255 (the core's k-round). Tolerance: f32 sums of exact
 * partials, ~1e-6 of sum |s_g partial| (tests). Inner product: the same problem with [OC, IC] weights
 * (ip_as_matmul). A floating src is dequant+'s, unless AIEDNN_WQ_DYNQ=1: each src row quantized to s8
 * with its own scale max|x| / 127 (not exact: the int8 rounding of the activations), then this path.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_gemm.h (mm_geometry, scratch).
 ******************************************************************************/
#ifndef AIEDNN_RT_WQ_H
#define AIEDNN_RT_WQ_H

static int is_wq_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_matmul && g->nargs == 5 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[0].dims[1] == 512 && g->args[2].data_type == aiednn_f32 && g->args[2].ndims == 2 &&
           g->args[2].dims[0] == 8 && g->args[3].data_type == aiednn_s32 && g->args[4].data_type == aiednn_f32;
}

/* AIEDNN_WQ_DYNQ=1: a floating src is quantized per row to s8 and takes wq: (not exact) */
static int wq_dynq(void) {
    const char *e = getenv("AIEDNN_WQ_DYNQ");
    return e && e[0] == '1';
}

static int wq_int_dt(aiednn_data_type_t t) {
    return t == aiednn_s8 || t == aiednn_u8 || t == aiednn_s4 || t == aiednn_u4;
}

static int acc_wq(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    const int ss = rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES);
    if (p->kind != aiednn_matmul || ss < 0) {
        snprintf(why, cap, "matmul with weight group scales only");
        return 0;
    }
    const aiednn_data_type_t sdt = p->md[0].data_type, wdt = p->md[1].data_type;
    const int dynq = sdt == aiednn_f32 && wq_dynq();
    if ((sdt != aiednn_s8 && sdt != aiednn_u8 && !dynq) || !wq_int_dt(wdt)) {
        snprintf(why, cap, "8-bit src x 8 / 4-bit weights only (a floating src: dequant, or AIEDNN_WQ_DYNQ=1)");
        return 0;
    }
    mm_geo g;
    mm_geometry(p, &g);
    if (p->md[1].ndims != 2) {
        snprintf(why, cap, "batched weights");
        return 0;
    }
    const int64_t G = p->md[ss].dims[0], gk = G > 0 ? g.K / G : 0;
    if (G <= 0 || g.K % G || gk % 64 || gk / 64 > 255 || p->md[ss].ndims != 2 || p->md[ss].dims[1] != g.N) {
        snprintf(why, cap, "group size %lld: a multiple of 64 up to 16320, scales [K/g, N]", (long long)gk);
        return 0;
    }
    return 1;
}

/* the weights as int8 [N][K] (u8 / u4 shifted so they fit: w' = w - wsh) */
#define RT_WQ_WCACHE 64
typedef struct {
    const void *key;
    int64_t n, k;
    int transposed;
    aiednn_data_type_t dt;
    int8_t *w;
} rt_wq_went;
static rt_wq_went g_wqc[RT_WQ_WCACHE];
static int g_wqc_n;
static void *g_wq_w, *g_wq_a, *g_wq_b, *g_wq_c, *g_wq_s, *g_wq_cor, *g_wq_q;
static size_t g_wq_w_cap, g_wq_a_cap, g_wq_b_cap, g_wq_c_cap, g_wq_s_cap, g_wq_cor_cap, g_wq_q_cap;

static const int8_t *wq_weights(const rt_prob *p, const void *W, int64_t N, int64_t K, int wsh) {
    const void *key = g_exec_const_w ? W : nullptr;
    for (int i = 0; key && i < g_wqc_n; i++)
        if (g_wqc[i].key == key && g_wqc[i].n == N && g_wqc[i].k == K && g_wqc[i].transposed == p->transposed &&
            g_wqc[i].dt == p->md[1].data_type)
            return g_wqc[i].w;
    int8_t *w = (int8_t *)scratch(&g_wq_w, &g_wq_w_cap, (size_t)(N * K));
    if (!w)
        return nullptr;
    const aiednn_data_type_t wdt = p->md[1].data_type;
    for (int64_t n = 0; n < N; n++)
        for (int64_t k = 0; k < K; k++)
            w[n * K + k] = (int8_t)(rt_ld(wdt, W, p->transposed ? n * K + k : k * N + n) - wsh);
    if (key && g_wqc_n < RT_WQ_WCACHE) {
        int8_t *c = (int8_t *)malloc((size_t)(N * K));
        if (c) {
            memcpy(c, w, (size_t)(N * K));
            g_wqc[g_wqc_n++] = rt_wq_went{key, N, K, p->transposed, wdt, c};
            return c;
        }
    }
    return w;
}

static int run_wq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    const aiednn_impl_t *gen = self->gen;
    rt_epi E_;
    rt_epi_init(&E_, p, ptr);
    mm_geo g;
    mm_geometry(p, &g);
    const int ss = rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES), zs = rt_slot(p, AIEDNN_ARG_ATTR_WEI_ZERO_POINTS);
    const int64_t M = g.M * g.batch, N = g.N, K = g.K, G = p->md[ss].dims[0], gk = K / G, gr = gk / 64;
    const int64_t BM = gen->args[0].dims[0], BN = gen->args[4].dims[1];
    aiednn_data_type_t sdt = p->md[0].data_type;
    const aiednn_data_type_t wdt = p->md[1].data_type;
    const uint8_t *A = (const uint8_t *)ptr[0];
    float *rs = nullptr; /* AIEDNN_WQ_DYNQ: per-row src scale */
    if (sdt == aiednn_f32) {
        int8_t *qa = (int8_t *)scratch(&g_wq_q, &g_wq_q_cap, (size_t)(M * K) + (size_t)M * sizeof(float));
        if (!qa)
            return AIEK_ERR_ALLOC;
        rs = (float *)(qa + M * K);
        const float *xf = (const float *)ptr[0];
        for (int64_t mm = 0; mm < M; mm++) {
            float mx = 0.f;
            for (int64_t k = 0; k < K; k++)
                mx = fmaxf(mx, fabsf(xf[mm * K + k]));
            rs[mm] = mx > 0.f ? mx / 127.f : 1.f;
            for (int64_t k = 0; k < K; k++)
                qa[mm * K + k] = (int8_t)lrintf(xf[mm * K + k] / rs[mm]);
        }
        A = (const uint8_t *)qa, sdt = aiednn_s8;
    }
    const int ash = sdt == aiednn_u8 ? 128 : 0, wsh = (wdt == aiednn_u8) ? 128 : 0;
    const double za = rs ? 0.0 : (double)p->src_zp - ash; /* za' */
    const uint64_t tw = rt_ticks();
    const int8_t *w8 = wq_weights(p, ptr[1], N, K, wsh);
    float *sc = (float *)scratch(&g_wq_s, &g_wq_s_cap, (size_t)(G * N) * sizeof(float));
    double *cor = (double *)scratch(&g_wq_cor, &g_wq_cor_cap, (size_t)(2 * G * N + G * M) * sizeof(double));
    int8_t *at = (int8_t *)scratch(&g_wq_a, &g_wq_a_cap, (size_t)(BM * 512));
    int8_t *bt = (int8_t *)scratch(&g_wq_b, &g_wq_b_cap, (size_t)(BN * 512));
    float *c = (float *)scratch(&g_wq_c, &g_wq_c_cap, (size_t)(BM * BN) * sizeof(float));
    if (!w8 || !sc || !cor || !at || !bt || !c)
        return AIEK_ERR_ALLOC;
    /* per (group, channel): scale s_g, zw' (zero point minus the weight shift), W'_g = sum_{k in g} w' */
    double *zw = cor, *wsum = cor + G * N, *asum = cor + 2 * G * N;
    int need_cor = za != 0.0;
    for (int64_t gi = 0; gi < G; gi++)
        for (int64_t n = 0; n < N; n++) {
            const int64_t i = gi * N + n;
            sc[i] = (float)rt_ld(p->md[ss].data_type, ptr[ss], i);
            zw[i] = (zs >= 0 ? rt_ld(p->md[zs].data_type, ptr[zs], i) : 0.0) - wsh;
            need_cor |= zw[i] != 0.0;
            double s = 0;
            for (int64_t k = gi * gk; za != 0.0 && k < (gi + 1) * gk; k++)
                s += w8[n * K + k];
            wsum[i] = s;
        }
    /* per (row, group): A'_g = sum_{k in g} a' (zero weight points only) */
    for (int64_t m = 0; need_cor && m < M; m++)
        for (int64_t gi = 0; gi < G; gi++) {
            double s = 0;
            for (int64_t k = gi * gk; k < (gi + 1) * gk; k++)
                s += (double)((int)(sdt == aiednn_u8 ? (int)A[m * K + k] : (int)(int8_t)A[m * K + k]) - ash);
            asum[m * G + gi] = s;
        }
    g_phase[RT_PH_BPACK] += rt_ticks() - tw;
    const int64_t nmb = (M + BM - 1) / BM, nnb = (N + BN - 1) / BN, nkb = (K + 511) / 512;
    float scr[8 * 512]; /* [8][BN] scale rows of one launch (BN <= 512) */
    int32_t ctl[3];
    const int64_t a_shape[2] = {BM, 512}, w_shape[2] = {BN, 512}, s_shape[2] = {8, BN}, t_shape[1] = {3},
                  c_shape[2] = {BM, BN};
    aiek_tensor t[5] = {{at, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * 512)},
                        {bt, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(BN * 512)},
                        {scr, 2, AIEK_FLOAT, 32, 1, s_shape, (uint64_t)(8 * BN * 4)},
                        {ctl, 1, AIEK_INT, 32, 1, t_shape, 12},
                        {c, 2, AIEK_FLOAT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
    for (int64_t mb = 0; mb < nmb; mb++)
        for (int64_t nb = 0; nb < nnb; nb++) {
            const int64_t m0 = mb * BM, n0 = nb * BN;
            const int64_t mn = M - m0 < BM ? M - m0 : BM, nn = N - n0 < BN ? N - n0 : BN;
            for (int64_t kb = 0; kb < nkb; kb++) {
                const uint64_t tp = rt_ticks();
                const int64_t k0 = kb * 512, kn = K - k0 < 512 ? K - k0 : 512;
                memset(at, 0, (size_t)(BM * 512));
                memset(bt, 0, (size_t)(BN * 512));
                for (int64_t m = 0; m < mn; m++) {
                    const uint8_t *src = A + (m0 + m) * K + k0;
                    for (int64_t k = 0; k < kn; k++)
                        at[m * 512 + k] = (int8_t)((int)(sdt == aiednn_u8 ? (int)src[k] : (int)(int8_t)src[k]) - ash);
                }
                for (int64_t n = 0; n < nn; n++)
                    memcpy(bt + n * 512, w8 + (n0 + n) * K + k0, (size_t)kn);
                for (int kr = 0; kr < 8; kr++) { /* round kr: the scales of the group ending there, else 0 */
                    const int64_t rg = kb * 8 + kr, gi = (rg + 1) / gr - 1;
                    const int ends = (rg + 1) % gr == 0 && gi < G;
                    for (int64_t n = 0; n < BN; n++)
                        scr[kr * BN + n] = ends && n < nn ? sc[gi * N + n0 + n] : 0.f;
                }
                ctl[0] = (kb > 0 ? 1 : 0) | (kb < nkb - 1 ? 2 : 0);
                ctl[1] = (int32_t)gr;
                ctl[2] = (int32_t)((kb * 8) % gr);
                g_phase[RT_PH_APACK] += rt_ticks() - tp;
                ((rt_impl *)self)->launches++;
                const uint64_t te = rt_ticks();
                const int rc = gen->entry(t, 5);
                ((rt_impl *)self)->t_entry += rt_ticks() - te;
                if (rc != AIEK_OK)
                    return rc;
            }
            const uint64_t t0 = rt_ticks();
            for (int64_t m = 0; m < mn; m++)
                for (int64_t n = 0; n < nn; n++) {
                    double y = c[m * BN + n];
                    for (int64_t gi = 0; need_cor && gi < G; gi++) {
                        const int64_t i = gi * N + n0 + n;
                        y += sc[i] * (-za * wsum[i] - zw[i] * asum[(m0 + m) * G + gi] + za * zw[i] * (double)gk);
                    }
                    rt_epi_put(&E_, (m0 + m) * N + n0 + n, n0 + n, rs ? y * rs[m0 + m] : y);
                }
            g_phase[RT_PH_EPI] += rt_ticks() - t0;
        }
    return AIEK_OK;
}

/* inner product: [N, IC...] x [OC, IC...] -> the matmul [N, IC] x [OC, IC]^T (ip_as_matmul) */
static int acc_wq_ip(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    rt_prob m;
    ip_as_matmul(p, &m);
    return acc_wq(self, &m, bind, why, cap);
}
static int run_wq_ip(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    aiednn_primitive q = *prim;
    ip_as_matmul(&prim->prob, &q.prob);
    return run_wq(self, &q, ptr);
}

#endif /* AIEDNN_RT_WQ_H */
