/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_bf16.h -- float matmul on the GEMM ELF's bf16 mode (aie_modes_bf16.h, opcode 0xBF):
 *   bf16blk:<blk>  bf16 x bf16 matmul;
 *   bf16x3:<blk>   f32 x f32 matmul as a bf16x3 split x = hi + mid + lo; the six significant products
 *                  run as ONE bf16 GEMM of depth 6K: A' = [Ah Ah Am Ah Al Am], W' = [Wh Wm Wh Wl Wh Wm].
 * Kernel block: A bf16 [BM][256], B^T bf16 [128][256], flags {bit 0 continue, bit 1 more K follows},
 * returns C f32 [BM][128]. Every element goes through rt_epi_put. Tolerance RT_BF16_RTOL * sqrt(K)
 * relative to sum |a b|, not bit-exact.
 *
 * Accepted: matmul, bf16 x bf16 (bf16blk:) or f32 x f32 (bf16x3:), no zero points, any dst.
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_gemm.h (mm_geometry).
 ******************************************************************************/
#ifndef AIEDNN_RT_BF16_H
#define AIEDNN_RT_BF16_H

#define RT_BF16_K 256
#define RT_BF16_RTOL 1e-5 /* bf16 operands; bf16x3: ~1e-6 (tests) */

static int is_bf16_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_matmul && g->nargs == 4 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[0].dims[1] == 2 * RT_BF16_K && g->args[1].data_type == aiednn_s8 &&
           g->args[3].data_type == aiednn_f32 && g->args[3].ndims == 2;
}

/* f32 -> bf16 bits, round to nearest even (NaN kept) */
static inline uint16_t rt_f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static inline float rt_bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
/* x = hi + mid + lo, each bf16 */
static inline void rt_bf3(float x, uint16_t *h) {
    h[0] = rt_f2bf(x);
    const float r1 = x - rt_bf2f(h[0]);
    h[1] = rt_f2bf(r1);
    h[2] = rt_f2bf(r1 - rt_bf2f(h[1]));
}

static int bf16_accept(const rt_prob *p, aiednn_data_type_t dt, char *why, int cap) {
    if (p->kind != aiednn_matmul || p->md[0].data_type != dt || p->md[1].data_type != dt) {
        snprintf(why, cap, "only %s x %s matmul", dt == aiednn_bf16 ? "bf16" : "f32",
                 dt == aiednn_bf16 ? "bf16" : "f32");
        return 0;
    }
    if (p->src_zp || p->wei_zp) {
        snprintf(why, cap, "zero points");
        return 0;
    }
    return 1;
}
static int acc_bf16blk(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    return bf16_accept(p, aiednn_bf16, why, cap);
}
static int acc_bf16x3(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    return bf16_accept(p, aiednn_f32, why, cap);
}

static void *g_b16a, *g_b16w, *g_b16t, *g_b16c;
static size_t g_b16a_cap, g_b16w_cap, g_b16t_cap, g_b16c_cap;

/* the bf16x3 operands: rows [R][K] f32 -> [R][6K] bf16, blocks of K = A' [hi hi mid hi lo mid] or
 * W' [hi mid hi lo hi mid] */
static void bf16x3_rows(const float *src, int64_t R, int64_t K, int wt, uint16_t *dst) {
    static const int ai[6] = {0, 0, 1, 0, 2, 1}, wi[6] = {0, 1, 0, 2, 0, 1};
    const int *sel = wt ? wi : ai;
    enum { CH = 256 };
    uint16_t pl[3][CH]; /* the hi / mid / lo planes of a chunk of one row */
    for (int64_t r = 0; r < R; r++)
        for (int64_t k0 = 0; k0 < K; k0 += CH) {
            const int64_t n = K - k0 < CH ? K - k0 : CH;
            const float *x = src + r * K + k0;
            for (int64_t k = 0; k < n; k++) { /* rt_bf3, with rt_f2bf's NaN test as a select */
                float v = x[k];
                for (int j = 0; j < 3; j++) {
                    uint32_t u;
                    memcpy(&u, &v, 4);
                    const uint32_t rn = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16, nan = (u >> 16) | 0x40;
                    const uint16_t h = (uint16_t)((u & 0x7fffffffu) > 0x7f800000u ? nan : rn);
                    pl[j][k] = h;
                    const uint32_t hb = (uint32_t)h << 16;
                    float hf;
                    memcpy(&hf, &hb, 4);
                    v -= hf;
                }
            }
            for (int j = 0; j < 6; j++)
                memcpy(dst + r * 6 * K + j * K + k0, pl[sel[j]], (size_t)n * 2);
        }
}

static void *g_b3a, *g_b3w;
static size_t g_b3a_cap, g_b3w_cap;

/* constant weights: the final [N][KK] bf16 operand kept per (pointer, shape), so a repeated call packs only A */
#define RT_B16_WCACHE 64
typedef struct {
    const void *key;
    int64_t n, k;
    int split, transposed;
    uint16_t *w;
} rt_b16w_ent;
static rt_b16w_ent g_b16wc[RT_B16_WCACHE];
static int g_b16wc_n;

static const uint16_t *b16_wcache_get(const void *key, int64_t n, int64_t k, int split, int tr) {
    for (int i = 0; key && i < g_b16wc_n; i++) {
        const rt_b16w_ent *e = &g_b16wc[i];
        if (e->key == key && e->n == n && e->k == k && e->split == split && e->transposed == tr)
            return e->w;
    }
    return nullptr;
}
static const uint16_t *b16_wcache_put(const void *key, int64_t n, int64_t k, int split, int tr, const uint16_t *w,
                                      int64_t kk) {
    if (!key || g_b16wc_n == RT_B16_WCACHE)
        return w;
    uint16_t *c = (uint16_t *)malloc((size_t)(n * kk) * 2);
    if (!c)
        return w;
    memcpy(c, w, (size_t)(n * kk) * 2);
    g_b16wc[g_b16wc_n++] = rt_b16w_ent{key, n, k, split, tr, c};
    return c;
}

/* ── several bf16 GEMMs per launch ─────────────────────────────────────────────
 * Between b16q_defer(1) / b16q_defer(0), a GEMM with a plain f32 dst queues its tiles; outputs are written
 * when the queue flushes. The persistent build runs a flush as one launch (aiednn_gemm_bf16_n_start), else
 * tile by tile. K chains may straddle a flush. The n-tile launch is opt-in (AIEDNN_B16N=1). */
extern "C" int aiednn_gemm_bf16_n_start(const uint16_t *a, int n, const uint16_t *w, const uint8_t *flags)
    __attribute__((weak));
extern "C" int aiednn_gemm_bf16_n_finish(void) __attribute__((weak));
extern "C" const float *aiednn_gemm_bf16_n_c(int t) __attribute__((weak));

#define RT_B16Q_MAX 32 /* gemm_uk.cc NT_MAX */
typedef struct {
    float *dst; /* the tile's [mn][nn] output (row stride ldd), NULL for a tile with more K to come */
    int64_t ldd, mn, nn;
} rt_b16q_out;
static struct {
    int defer, n;
    const rt_impl *self;
    int64_t BM, BN;
    uint16_t *a, *w; /* RT_B16Q_MAX tiles: A [BM][RT_BF16_K], W [BN][RT_BF16_K] */
    size_t a_cap, w_cap;
    uint8_t flags[RT_B16Q_MAX];
    rt_b16q_out out[RT_B16Q_MAX];
} g_b16q;

/* AIEDNN_NTILE_EMU: the n-tile launch as its tiles through the single-tile entry, tile t's C in slot t */
static float *g_b16q_emu_c;
static size_t g_b16q_emu_cap;
static int b16q_emu_start(const aiednn_impl_t *gen, int n, int64_t BM, int64_t BN) {
    const int64_t BK = RT_BF16_K;
    g_b16q_emu_c = (float *)scratch((void **)&g_b16q_emu_c, &g_b16q_emu_cap, (size_t)(n * BM * BN) * sizeof(float));
    if (!g_b16q_emu_c)
        return AIEK_ERR_ALLOC;
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, 2 * BK}, w_shape[2] = {BN, 2 * BK}, f_shape[1] = {1}, c_shape[2] = {BM, BN};
    for (int t = 0; t < n; t++) {
        aiek_tensor at[4] = {{g_b16q.a + t * BM * BK, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK * 2)},
                             {g_b16q.w + t * BN * BK, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(BN * BK * 2)},
                             {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                             {g_b16q_emu_c + t * BM * BN, 2, AIEK_FLOAT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
        flags = g_b16q.flags[t];
        const int rc = gen->entry(at, 4);
        if (rc != AIEK_OK)
            return rc;
    }
    return AIEK_OK;
}

static int b16q_flush(void) {
    const int n = g_b16q.n;
    if (!n)
        return AIEK_OK;
    rt_impl *self = (rt_impl *)g_b16q.self;
    const aiednn_impl_t *gen = self->gen;
    const int64_t BM = g_b16q.BM, BN = g_b16q.BN, BK = RT_BF16_K;
    g_b16q.n = 0;
    int rc = AIEK_OK;
    const int emu = rt_env_on("AIEDNN_NTILE_EMU"); /* aiednn_rt_ntile.h: tests, no board */
    if (((aiednn_gemm_bf16_n_start && aiednn_gemm_bf16_n_finish && aiednn_gemm_bf16_n_c) || emu) && n > 1 &&
        rt_env_on("AIEDNN_B16N")) { /* opt-in until a board run validates it */
        gen->entry(nullptr, 0);     /* the megakernel thunk: the ELF with the bf16 mode */
        self->launches++;
        const uint64_t te = rt_ticks();
        rc = emu ? b16q_emu_start(gen, n, BM, BN) : aiednn_gemm_bf16_n_start(g_b16q.a, n, g_b16q.w, g_b16q.flags);
        if (rc == AIEK_OK && !emu)
            rc = aiednn_gemm_bf16_n_finish();
        self->t_entry += rt_ticks() - te;
        for (int t = 0; rc == AIEK_OK && t < n; t++) {
            const rt_b16q_out *o = &g_b16q.out[t];
            const float *c = !o->dst ? nullptr : emu ? g_b16q_emu_c + t * BM * BN : aiednn_gemm_bf16_n_c(t);
            for (int64_t m = 0; c && m < o->mn; m++)
                memcpy(o->dst + m * o->ldd, c + m * BN, (size_t)o->nn * sizeof(float));
        }
        return rc;
    }
    float *c = (float *)scratch(&g_b16c, &g_b16c_cap, (size_t)(BM * BN) * sizeof(float));
    if (!c)
        return AIEK_ERR_ALLOC;
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, 2 * BK}, w_shape[2] = {BN, 2 * BK}, f_shape[1] = {1}, c_shape[2] = {BM, BN};
    for (int t = 0; rc == AIEK_OK && t < n; t++) {
        aiek_tensor at[4] = {{g_b16q.a + t * BM * BK, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK * 2)},
                             {g_b16q.w + t * BN * BK, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(BN * BK * 2)},
                             {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                             {c, 2, AIEK_FLOAT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
        flags = g_b16q.flags[t];
        self->launches++;
        const uint64_t te = rt_ticks();
        rc = gen->entry(at, 4);
        self->t_entry += rt_ticks() - te;
        const rt_b16q_out *o = &g_b16q.out[t];
        for (int64_t m = 0; rc == AIEK_OK && o->dst && m < o->mn; m++)
            memcpy(o->dst + m * o->ldd, c + m * BN, (size_t)o->nn * sizeof(float));
    }
    return rc;
}

/* on: queue plain bf16 GEMMs from here; off (balanced): run what is queued */
static int b16q_defer(int on) {
    if (on) {
        g_b16q.defer++;
        return AIEK_OK;
    }
    return --g_b16q.defer > 0 ? AIEK_OK : b16q_flush();
}

/* the next queue slot's A / W tiles for a GEMM of `self` (flushing first when full or of another block shape) */
static int b16q_slot(const rt_impl *self, int64_t BM, int64_t BN, uint16_t **a, uint16_t **w) {
    if (g_b16q.n && (g_b16q.n == RT_B16Q_MAX || g_b16q.self != self || g_b16q.BM != BM || g_b16q.BN != BN)) {
        const int rc = b16q_flush();
        if (rc != AIEK_OK)
            return rc;
    }
    uint16_t *qa = (uint16_t *)scratch((void **)&g_b16q.a, &g_b16q.a_cap, (size_t)(RT_B16Q_MAX * BM * RT_BF16_K) * 2);
    uint16_t *qw = (uint16_t *)scratch((void **)&g_b16q.w, &g_b16q.w_cap, (size_t)(RT_B16Q_MAX * BN * RT_BF16_K) * 2);
    if (!qa || !qw)
        return AIEK_ERR_ALLOC;
    g_b16q.self = self, g_b16q.BM = BM, g_b16q.BN = BN;
    *a = qa + g_b16q.n * BM * RT_BF16_K, *w = qw + g_b16q.n * BN * RT_BF16_K;
    return AIEK_OK;
}

static int run_bf16_common(const rt_impl *self, aiednn_primitive *prim, void *const *ptr, int split) {
    const rt_prob *p = &prim->prob;
    const aiednn_impl_t *gen = self->gen;
    rt_epi E_;
    rt_epi_init(&E_, p, ptr);
    mm_geo g;
    mm_geometry(p, &g);
    const int plain =
        !E_.fx && !E_.acc_add && !E_.nosc && !E_.bias_mode && !E_.npo && !E_.requant && E_.d->data_type == aiednn_f32;
    /* queue only for the n-tile launch */
    const int defer = g_b16q.defer > 0 && plain && rt_env_on("AIEDNN_B16N");
    if (!defer && g_b16q.n) { /* this GEMM runs now: what is queued first, in order */
        const int rq = b16q_flush();
        if (rq != AIEK_OK)
            return rq;
    }
    const int64_t es = split ? 4 : 2; /* element bytes of src / weights */
    const int64_t BM = gen->args[0].dims[0], BN = gen->args[3].dims[1], BK = RT_BF16_K, KK = split ? 6 * g.K : g.K;
    uint16_t *at = (uint16_t *)scratch(&g_b16a, &g_b16a_cap, (size_t)(BM * BK) * 2);
    uint16_t *wt = (uint16_t *)scratch(&g_b16w, &g_b16w_cap, (size_t)(BN * BK) * 2);
    uint8_t *wtr = p->transposed ? nullptr : (uint8_t *)scratch(&g_b16t, &g_b16t_cap, (size_t)(g.N * g.K * es));
    float *c = (float *)scratch(&g_b16c, &g_b16c_cap, (size_t)(BM * BN) * sizeof(float));
    if (!at || !wt || !c || (!p->transposed && !wtr))
        return AIEK_ERR_ALLOC;
    const int64_t nmb = (g.M + BM - 1) / BM, nnb = (g.N + BN - 1) / BN, nkb = (KK + BK - 1) / BK;
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, 2 * BK}, w_shape[2] = {BN, 2 * BK}, f_shape[1] = {1}, c_shape[2] = {BM, BN};
    aiek_tensor t[4] = {{at, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK * 2)},
                        {wt, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(BN * BK * 2)},
                        {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                        {c, 2, AIEK_FLOAT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
    for (int64_t b = 0; b < g.batch; b++) {
        const uint8_t *A = (const uint8_t *)ptr[0] + bcast_off(&g.bs, &g.bd, b) * g.M * g.K * es;
        const uint8_t *W = (const uint8_t *)ptr[1] + bcast_off(&g.bw, &g.bd, b) * g.N * g.K * es;
        const void *wkey = g_exec_const_w ? (const void *)W : nullptr; /* the caller's weights */
        const uint16_t *A16 = (const uint16_t *)A;
        const uint16_t *W16 = b16_wcache_get(wkey, g.N, g.K, split, p->transposed);
        if (!W16) {
            const uint64_t ts = rt_ticks();
            if (!p->transposed) { // [K][N] -> [N][K]
                for (int64_t k = 0; k < g.K; k++)
                    for (int64_t n = 0; n < g.N; n++)
                        memcpy(wtr + (n * g.K + k) * es, W + (k * g.N + n) * es, (size_t)es);
                W = wtr;
            }
            W16 = (const uint16_t *)W;
            if (split) { /* W' [N][6K] */
                uint16_t *w3 = (uint16_t *)scratch(&g_b3w, &g_b3w_cap, (size_t)(g.N * KK) * 2);
                if (!w3)
                    return AIEK_ERR_ALLOC;
                bf16x3_rows((const float *)W, g.N, g.K, 1, w3);
                W16 = w3;
            }
            W16 = b16_wcache_put(wkey, g.N, g.K, split, p->transposed, W16, KK);
            g_phase[RT_PH_BPACK] += rt_ticks() - ts;
        }
        if (split) { /* A' [M][6K] once per batch */
            const uint64_t ts = rt_ticks();
            uint16_t *a3 = (uint16_t *)scratch(&g_b3a, &g_b3a_cap, (size_t)(g.M * KK) * 2);
            if (!a3)
                return AIEK_ERR_ALLOC;
            bf16x3_rows((const float *)A, g.M, g.K, 0, a3);
            A16 = a3;
            g_phase[RT_PH_APACK] += rt_ticks() - ts;
        }
        for (int64_t mb = 0; mb < nmb; mb++)
            for (int64_t nb = 0; nb < nnb; nb++) {
                const int64_t m0 = mb * BM, n0 = nb * BN;
                const int64_t mn = g.M - m0 < BM ? g.M - m0 : BM, nn = g.N - n0 < BN ? g.N - n0 : BN;
                for (int64_t kb = 0; kb < nkb; kb++) {
                    const int64_t k0 = kb * BK, kn = KK - k0 < BK ? KK - k0 : BK;
                    uint16_t *ta = at, *tw = wt;
                    if (defer) { /* the tile goes to the queue (b16q_defer) */
                        const int rq = b16q_slot(self, BM, BN, &ta, &tw);
                        if (rq != AIEK_OK)
                            return rq;
                    }
                    const uint64_t tp = rt_ticks();
                    memset(ta, 0, (size_t)(BM * BK) * 2);
                    memset(tw, 0, (size_t)(BN * BK) * 2);
                    for (int64_t m = 0; m < mn; m++)
                        memcpy(ta + m * BK, A16 + (m0 + m) * KK + k0, (size_t)kn * 2);
                    for (int64_t n = 0; n < nn; n++)
                        memcpy(tw + n * BK, W16 + (n0 + n) * KK + k0, (size_t)kn * 2);
                    g_phase[RT_PH_APACK] += rt_ticks() - tp;
                    flags = (kb > 0 ? 1 : 0) | (kb < nkb - 1 ? 2 : 0);
                    if (defer) {
                        const int q = g_b16q.n++;
                        g_b16q.flags[q] = (uint8_t)flags;
                        g_b16q.out[q] = kb < nkb - 1
                                            ? rt_b16q_out{nullptr, 0, 0, 0}
                                            : rt_b16q_out{(float *)E_.dst + (b * g.M + m0) * g.N + n0, g.N, mn, nn};
                        continue;
                    }
                    ((rt_impl *)self)->launches++;
                    const uint64_t te = rt_ticks();
                    const int rc = gen->entry(t, 4);
                    ((rt_impl *)self)->t_entry += rt_ticks() - te;
                    if (rc != AIEK_OK)
                        return rc;
                }
                if (defer) /* written by the flush */
                    continue;
                const uint64_t t0 = rt_ticks();
                if (plain) /* rt_epi_put of an f32 sum into f32 with nothing to apply: the value itself */
                    for (int64_t m = 0; m < mn; m++)
                        memcpy((float *)E_.dst + (b * g.M + m0 + m) * g.N + n0, c + m * BN, (size_t)nn * sizeof(float));
                else
                    for (int64_t m = 0; m < mn; m++)
                        for (int64_t n = 0; n < nn; n++)
                            rt_epi_put(&E_, (b * g.M + m0 + m) * g.N + n0 + n, n0 + n, (double)c[m * BN + n]);
                g_phase[RT_PH_EPI] += rt_ticks() - t0;
            }
    }
    return AIEK_OK;
}

static int run_bf16blk(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    return run_bf16_common(self, prim, ptr, 0);
}
static int run_bf16x3(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    return run_bf16_common(self, prim, ptr, 1);
}

#endif /* AIEDNN_RT_BF16_H */
