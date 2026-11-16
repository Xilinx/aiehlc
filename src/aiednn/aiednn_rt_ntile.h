/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_ntile.h -- the im2col convolution pipes with many GEMM tiles per AIE launch.
 *
 * One n-tile launch (aiednn_gemm_ntile_*) streams a chunk of m blocks x every k block of one n block:
 * A tiles packed straight into the launch's device buffer (m outer, k inner), K blocks chained on the
 * core, B blocks through the mem tile. Same blocks and arithmetic as the per-block pipes. During chunk
 * j's launch the host packs chunk j+1 and finishes chunk j-1's output.
 *
 * Not taken (the caller's per-block pipe runs): no n-tile entry, a fused residual (B cannot be shared
 * by m blocks), more K blocks than tiles per launch.
 *
 * Needs, from the including TU: aiednn_rt_internal.h and aiednn_rt_gemm.h (included at its end).
 ******************************************************************************/
#ifndef AIEDNN_RT_NTILE_H
#define AIEDNN_RT_NTILE_H

extern "C" int aiednn_gemm_ntile_max(void) __attribute__((weak));
/* device memory for `tiles` A tiles of tile_bytes ([BM][BK]) */
extern "C" int8_t *aiednn_gemm_ntile_stage_a(int64_t tiles, int64_t tile_bytes) __attribute__((weak));
/* which 1: int8 C via the core's requant (qp: gemm_q_s8s8's, no residual), 0: int32 C; a = nm x nkb
 * A tiles (m outer, k inner), b = the n block's nkb logical B blocks [bn][bk] */
extern "C" int aiednn_gemm_ntile_start(int which, const int8_t *a, int nm, int nkb, const int8_t *b, const int32_t *qp,
                                       uint64_t key, int64_t bm, int64_t bn, int64_t bk) __attribute__((weak));
extern "C" int aiednn_gemm_ntile_finish(void) __attribute__((weak));
extern "C" const int8_t *aiednn_gemm_ntile_c(int mb) __attribute__((weak));
/* the next start's A tiles are direct-conv regions (gemm_uk.cc stamps CONV_MAGIC into its B blocks) */
extern "C" void aiednn_gemm_ntile_conv(int on) __attribute__((weak));

/* an environment switch is on: set, non-empty and not "0" */
/* a runtime switch: the environment, or the same name defined at build time (dnn_build AIEDNN_DEFINES) */
static int rt_env_on(const char *name) {
    static const char *const built[] = {
#if defined(AIEDNN_NO_TPIPE)
        "AIEDNN_NO_TPIPE",
#endif
#if defined(AIEDNN_NO_QFUSE)
        "AIEDNN_NO_QFUSE",
#endif
#if defined(AIEDNN_NO_NTILE)
        "AIEDNN_NO_NTILE",
#endif
#if defined(AIEDNN_NO_NTILE_CONV)
        "AIEDNN_NO_NTILE_CONV",
#endif
#if defined(AIEDNN_NO_ITEMSN)
        "AIEDNN_NO_ITEMSN",
#endif
#if defined(AIEDNN_NO_DWN)
        "AIEDNN_NO_DWN",
#endif
#if defined(AIEDNN_NO_BATCH_M)
        "AIEDNN_NO_BATCH_M",
#endif
#if defined(AIEDNN_NO_SDPABLK)
        "AIEDNN_NO_SDPABLK",
#endif
#if defined(AIEDNN_B16N)
        "AIEDNN_B16N",
#endif
#if defined(AIEDNN_NO_NTILE_MM)
        "AIEDNN_NO_NTILE_MM",
#endif
#if defined(AIEDNN_NTILE_MM_T)
        "AIEDNN_NTILE_MM_T",
#endif
#if defined(AIEDNN_NTILE_EMU)
        "AIEDNN_NTILE_EMU",
#endif
#if defined(AIEDNN_NO_FVEC_ELEMBIN)
        "AIEDNN_NO_FVEC_ELEMBIN",
#endif
        nullptr};
    for (int i = 0; built[i]; i++)
        if (!strcmp(built[i], name))
            return 1;
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
}

/* m blocks per launch for nkb K blocks, or 0 when n-tile launches are unavailable */
static int ntile_mblocks(int64_t nkb, int has_res) {
    if (has_res || !aiednn_gemm_ntile_max || !aiednn_gemm_ntile_stage_a || !aiednn_gemm_ntile_start ||
        !aiednn_gemm_ntile_finish || !aiednn_gemm_ntile_c || rt_env_on("AIEDNN_NO_NTILE"))
        return 0;
    const int64_t m = aiednn_gemm_ntile_max() / nkb;
    return m > 0 ? (int)m : 0;
}

/* the [BM][BK] tiles of A rows m0 .. m0+mn (every k block, kb at kb * BM * BK) straight into dst */
static void ntile_pack_a(const int8_t *A, int64_t m0, int64_t mn, int64_t K, int64_t BM, int64_t BK, int8_t *dst) {
    const int64_t nkb = (K + BK - 1) / BK;
    const uint64_t ta0 = rt_ticks();
    for (int64_t kb = 0; kb < nkb; kb++) {
        const int64_t k0 = kb * BK, kn = K - k0 < BK ? K - k0 : BK;
        int8_t *d = dst + kb * BM * BK;
        for (int64_t r = 0; r < BM; r++) {
            if (r < mn) {
                memcpy(d + r * BK, A + (m0 + r) * K + k0, (size_t)kn);
                if (kn < BK)
                    memset(d + r * BK + kn, 0, (size_t)(BK - kn));
            } else {
                memset(d + r * BK, 0, (size_t)BK);
            }
        }
    }
    g_phase[RT_PH_APACK] += rt_ticks() - ta0;
}

/* A-tile producer: m block mb's nkb tiles ([BM][BK] each, kb at kb * BM * BK) into dst; the launch
 * treats them as bytes. */
typedef void (*ntile_a_fn)(void *ctx, int64_t mb, int8_t *dst);

typedef struct {
    const int8_t *col; /* the im2col matrix [M][K] */
    int64_t M, K, BM, BK;
} ntile_im2col;

static void ntile_a_im2col(void *ctx, int64_t mb, int8_t *dst) {
    const ntile_im2col *a = (const ntile_im2col *)ctx;
    const int64_t m0 = mb * a->BM, mn = a->M - m0 < a->BM ? a->M - m0 : a->BM;
    ntile_pack_a(a->col, m0, mn, a->K, a->BM, a->BK, dst);
}

/* direct-conv A-tile producer: m block mb's nkb tiles are each mesh row's input region (conv_region_block)
 * with its geometry in the slice's last 64 bytes, where the core reads it (aie_modes_conv.h) */
typedef struct {
    const conv_geo *c;
    const int8_t *xt; /* the [H][W][Cg] images (hwc_transpose / the stem's padded input) */
    int64_t BM, BK, nkb;
} ntile_creg;

static void ntile_a_creg(void *ctx, int64_t mb, int8_t *dst) {
    const ntile_creg *a = (const ntile_creg *)ctx;
    int32_t geo[16 * 32];
    const uint64_t t0 = rt_ticks();
    for (int64_t kb = 0; kb < a->nkb; kb++) {
        int8_t *t = dst + kb * a->BM * a->BK;
        conv_region_block(a->c, a->xt, mb * a->BM, kb * a->BK, a->BM, a->BK, t, geo);
        for (int64_t r = 0; r < a->BM / 32; r++)
            memcpy(t + r * 32 * a->BK + CONV_REGION_DATA, geo + 16 * r, 64);
    }
    g_phase[RT_PH_REGION] += rt_ticks() - t0;
}

typedef struct {
    rt_pipe_work *w; /* im2col rows / output finishing (pipe_work) */
    ntile_a_fn a_fn; /* the A tiles of one m block */
    void *a_ctx;
    int8_t *stage; /* the A tiles, (mb, kb) at (mb * nkb + kb) * BM * BK */
    int64_t BM, BK, nkb;
    int64_t pack_mb0, pack_mb1; /* m blocks to pack into stage during this launch */
} ntile_work;

static void ntile_overlap(void *v) {
    ntile_work *nw = (ntile_work *)v;
    nw->w->step = 0;
    pipe_work(nw->w); /* this launch's slice: the next chunk's im2col rows, the previous chunk's output */
    for (int64_t mb = nw->pack_mb0; mb < nw->pack_mb1; mb++)
        nw->a_fn(nw->a_ctx, mb, nw->stage + mb * nw->nkb * nw->BM * nw->BK);
    nw->pack_mb0 = nw->pack_mb1;
}

/* the per-n-block {zp, lo, hi, flags, B, M, sh} of gemm_q_s8s8 (no residual), as run_q_pipe */
static void ntile_qparams(const rt_epi *E, int64_t ch0, int64_t N, int64_t BN, int64_t NP, int64_t nnb, int32_t *qps) {
    memset(qps, 0, (size_t)(nnb * NP) * sizeof(int32_t));
    for (int64_t nb = 0; nb < nnb; nb++) {
        int32_t *qp = qps + nb * NP;
        const int64_t n0 = nb * BN, nn = N - n0 < BN ? N - n0 : BN;
        qp[0] = (int32_t)E->dzp, qp[1] = E->fx_lo, qp[2] = E->fx_hi;
        int64_t *qb = (int64_t *)(qp + 4);
        for (int64_t j = 0; j < BN; j++) {
            const int64_t ch = ch0 + n0 + j;
            qb[j] = j < nn ? E->fx_b[ch] : 0;
            qp[4 + 2 * BN + j] = j < nn ? E->fx_m[ch] : (1 << 30);
            qp[4 + 3 * BN + j] = j < nn ? E->fx_sh[ch] : 31;
        }
    }
}

static inline uint64_t ntile_key(const void *bkey, int64_t nb, int64_t N, int64_t K) {
    if (!bkey)
        return 0;
    const uint64_t k = (uint64_t)(uintptr_t)bkey ^ (uint64_t)nb * 0x9E3779B97F4A7C15ull ^ ((uint64_t)N << 40) ^
                       ((uint64_t)K << 20) ^ 0x6E74696C65ull; /* "ntile": distinct from the per-block keys */
    return k ? k : 1;
}

static void *g_ntq;
static size_t g_ntq_cap;

/* which 1: int8 out via the core's requant (o8 [M][N], then the dst planes); 0: int32 o32 [M][N] and
 * the host epilogue (E, di0, ch0). creg: direct conv -- the tiles are input regions, no im2col */
static int run_ntile(const rt_impl *self, const rt_epi *E, const conv_geo *c, int mode, const int8_t *img,
                     const int8_t *wv, int64_t ch0, int8_t *col, int which, int8_t *o8, int8_t *dst, int32_t *o32,
                     int64_t di0, int creg) {
    const aiednn_impl_t *blk = which ? g_qblk : g_kcblk;
    const int64_t M = conv_m(c), N = c->Og, K = c->Cg * c->KH * c->KW;
    const int64_t BM = blk->args[0].dims[0], BK = blk->args[0].dims[1], BN = blk->args[1].dims[0];
    const int64_t NP = which ? blk->args[2].dims[0] : 0;
    const int64_t nkb = (K + BK - 1) / BK, nnb = (N + BN - 1) / BN, nmb = (M + BM - 1) / BM;
    const int64_t mpc = ntile_mblocks(nkb, 0); /* m blocks per launch */
    const void *bkey = g_exec_const_w ? (const void *)wv : nullptr;
    int8_t *bp = b_blocks(wv, N, K, blk, bkey);
    int8_t *stage = aiednn_gemm_ntile_stage_a(nmb * nkb, BM * BK);
    int32_t *qps = which ? (int32_t *)scratch(&g_ntq, &g_ntq_cap, (size_t)(nnb * NP) * sizeof(int32_t)) : nullptr;
    const int8_t *xsrc = mode == 1 ? hwc_transpose(c, img) : img; // mode 3: [H][W][Cg] already
    if (!bp || !stage || (which && !qps) || !xsrc || mpc < 1 || (creg && !aiednn_gemm_ntile_conv))
        return AIEK_ERR_ALLOC;
    if (which)
        ntile_qparams(E, ch0, N, BN, NP, nnb, qps);
    /* call the block's entry once with no arguments (nothing launched) so the usage run counts this mode */
    (void)(creg ? (which ? g_cqblk : g_ckcblk) : blk)->entry(nullptr, 0);
    rt_pipe_work w = {c, mode, xsrc, col, 0, 0, o8, dst, 0, 0, which ? nullptr : E, which ? nullptr : o32, di0, ch0, 0};
    ntile_im2col a_im2col = {col, M, K, BM, BK};
    ntile_creg a_creg = {c, xsrc, BM, BK, nkb};
    ntile_work nw = {
        &w, creg ? ntile_a_creg : ntile_a_im2col, creg ? (void *)&a_creg : (void *)&a_im2col, stage, BM, BK, nkb, 0, 0};
    const int pack = !creg; /* im2col rows only without direct conv */
    /* chunk 0's rows and tiles before the first launch */
    w.pack_hi = pack ? (mpc * BM < M ? mpc * BM : M) : 0;
    nw.pack_mb1 = mpc < nmb ? mpc : nmb;
    ntile_overlap(&nw);
    for (int64_t nb = 0; nb < nnb; nb++)
        for (int64_t c0 = 0; c0 < nmb; c0 += mpc) {
            const int64_t cn = nmb - c0 < mpc ? nmb - c0 : mpc;
            if (nb == 0) { /* during this launch: the next chunk's im2col rows and A tiles */
                w.pack_lo = (c0 + cn) * BM < M ? (c0 + cn) * BM : M;
                w.pack_hi = !pack ? w.pack_lo : (c0 + 2 * cn) * BM < M ? (c0 + 2 * cn) * BM : M;
                nw.pack_mb0 = c0 + cn, nw.pack_mb1 = c0 + 2 * cn < nmb ? c0 + 2 * cn : nmb;
            } else {
                w.pack_lo = w.pack_hi = 0, nw.pack_mb0 = nw.pack_mb1 = 0;
            }
            if (nb == nnb - 1) /* the previous chunk's output, now complete */
                w.sc_lo = c0 > 0 ? (c0 - mpc) * BM : 0, w.sc_hi = c0 > 0 ? c0 * BM : 0;
            const uint64_t te = rt_ticks();
            if (creg)
                aiednn_gemm_ntile_conv(1);
            const uint64_t key = nkb == 1 ? ntile_key(bkey, nb, N, K) ^ (creg ? 0x636F6E76ull : 0) : 0; /* "conv" */
            int rc = aiednn_gemm_ntile_start(which, stage + c0 * nkb * BM * BK, (int)cn, (int)nkb,
                                             bp + nb * nkb * BN * BK, which ? qps + nb * NP : nullptr, key, BM, BN, BK);
            if (creg)
                aiednn_gemm_ntile_conv(0);
            if (rc == AIEK_OK) {
                ntile_overlap(&nw);
                rc = aiednn_gemm_ntile_finish();
            }
            ((rt_impl *)self)->t_entry += rt_ticks() - te;
            ((rt_impl *)self)->launches++;
            if (rc != AIEK_OK)
                return rc;
            const uint64_t tc0 = rt_ticks();
            const int64_t n0 = nb * BN, nn = N - n0 < BN ? N - n0 : BN;
            for (int64_t j = 0; j < cn; j++) { /* the chunk's results into [M][N] */
                const int64_t m0 = (c0 + j) * BM, mn = M - m0 < BM ? M - m0 : BM;
                const int8_t *tile = aiednn_gemm_ntile_c((int)j);
                for (int64_t r = 0; r < mn; r++) {
                    if (which) { /* int8 C (gemm_q_post): row r is 4 * BN bytes, core column q's 32 at q * 128 */
                        const int64_t real = 32;
                        for (int64_t q = 0; q * real < nn; q++)
                            memcpy(o8 + (m0 + r) * N + n0 + q * real, tile + r * 4 * BN + q * 128,
                                   (size_t)(nn - q * real < real ? nn - q * real : real));
                    } else {
                        memcpy(o32 + (m0 + r) * N + n0, (const int32_t *)tile + r * BN, (size_t)nn * sizeof(int32_t));
                    }
                }
            }
            g_phase[RT_PH_CUNPACK] += rt_ticks() - tc0;
        }
    w.sc_lo = ((nmb - 1) / mpc) * mpc * BM, w.sc_hi = M, w.step = 0; /* the last chunk */
    w.pack_lo = w.pack_hi = 0;
    pipe_work(&w);
    return AIEK_OK;
}

/* Workstation emulation (AIEDNN_NTILE_EMU=1): the n-tile launch's tiles, in order and with the same
 * K-chain flags, through the single-block entries. */
static int8_t *g_nte_a, *g_nte_c;
static size_t g_nte_a_cap, g_nte_c_cap;
static int nte_on(void) { return rt_env_on("AIEDNN_NTILE_EMU"); }
static int8_t *nte_stage_a(int64_t tiles, int64_t tile_bytes) {
    return (int8_t *)scratch((void **)&g_nte_a, &g_nte_a_cap, (size_t)(tiles * tile_bytes));
}
/* which 0 only: tile (j, kb) = A tile a + (j * nkb + kb), B block b + kb; m block j's int32 C in slot j */
static int nte_start(const int8_t *a, int nm, int nkb, const int8_t *b, int64_t BM, int64_t BN, int64_t BK) {
    const aiednn_impl_t *kc = g_kcblk;
    int32_t *c = (int32_t *)scratch((void **)&g_nte_c, &g_nte_c_cap, (size_t)(nm * BM * BN) * sizeof(int32_t));
    if (!kc || !c)
        return AIEK_ERR_ALLOC;
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, BK}, b_shape[2] = {BN, BK}, f_shape[1] = {1}, c_shape[2] = {BM, BN};
    for (int j = 0; j < nm; j++)
        for (int kb = 0; kb < nkb; kb++) {
            aiek_tensor t[4] = {
                {(void *)(a + (j * nkb + kb) * BM * BK), 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK)},
                {(void *)(b + kb * BN * BK), 2, AIEK_INT, 8, 1, b_shape, (uint64_t)(BN * BK)},
                {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                {c + j * BM * BN, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
            flags = (kb > 0 ? 1 : 0) | (kb < nkb - 1 ? 2 : 0);
            const int rc = kc->entry(t, 4);
            if (rc != AIEK_OK)
                return rc;
        }
    return AIEK_OK;
}

/* X = P Q^T as n-tile launches: P [Pr][K] staged as A tiles, Q as nqb x nkb packed [BN][BK] blocks (qb).
 * tr 0: C[p][q] (row stride ldc); tr 1: C[q][p]. */
static int ntile_mm(const rt_impl *self, const int8_t *P, int64_t Pr, const int8_t *qb, int64_t Qr, int64_t K,
                    int64_t mpc, int emu, const void *bkey, int32_t *C, int64_t ldc, int tr) {
    const aiednn_impl_t *kc = g_kcblk;
    const int64_t BM = kc->args[0].dims[0], BK = kc->args[0].dims[1], BN = kc->args[1].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nqb = (Qr + BN - 1) / BN, npb = (Pr + BM - 1) / BM;
    int8_t *stage = emu ? nte_stage_a(npb * nkb, BM * BK) : aiednn_gemm_ntile_stage_a(npb * nkb, BM * BK);
    if (!stage)
        return AIEK_ERR_ALLOC;
    for (int64_t pb = 0; pb < npb; pb++) /* every A tile once */
        ntile_pack_a(P, pb * BM, Pr - pb * BM < BM ? Pr - pb * BM : BM, K, BM, BK, stage + pb * nkb * BM * BK);
    (void)kc->entry(nullptr, 0); /* no launch: the usage run / megakernel thunk sees this mode (run_ntile) */
    for (int64_t nb = 0; nb < nqb; nb++) {
        const int64_t q0 = nb * BN, qn = Qr - q0 < BN ? Qr - q0 : BN;
        for (int64_t c0 = 0; c0 < npb; c0 += mpc) {
            const int64_t cn = npb - c0 < mpc ? npb - c0 : mpc;
            const uint64_t key = nkb == 1 ? ntile_key(bkey, nb, Qr, K) : 0;
            const uint64_t te = rt_ticks();
            int rc = emu ? nte_start(stage + c0 * nkb * BM * BK, (int)cn, (int)nkb, qb + nb * nkb * BN * BK, BM, BN, BK)
                         : aiednn_gemm_ntile_start(0, stage + c0 * nkb * BM * BK, (int)cn, (int)nkb,
                                                   qb + nb * nkb * BN * BK, nullptr, key, BM, BN, BK);
            if (rc == AIEK_OK && !emu)
                rc = aiednn_gemm_ntile_finish();
            ((rt_impl *)self)->t_entry += rt_ticks() - te;
            ((rt_impl *)self)->launches++;
            if (rc != AIEK_OK)
                return rc;
            const uint64_t tc0 = rt_ticks();
            for (int64_t j = 0; j < cn; j++) {
                const int64_t p0 = (c0 + j) * BM, pn = Pr - p0 < BM ? Pr - p0 : BM;
                const int32_t *tile =
                    emu ? (const int32_t *)g_nte_c + j * BM * BN : (const int32_t *)aiednn_gemm_ntile_c((int)j);
                if (!tr)
                    for (int64_t r = 0; r < pn; r++)
                        memcpy(C + (p0 + r) * ldc + q0, tile + r * BN, (size_t)qn * sizeof(int32_t));
                else
                    for (int64_t r = 0; r < pn; r++)
                        for (int64_t q = 0; q < qn; q++)
                            C[(q0 + q) * ldc + p0 + r] = tile[r * BN + q];
            }
            g_phase[RT_PH_CUNPACK] += rt_ticks() - tc0;
        }
    }
    return AIEK_OK;
}

static void *g_ntt_q;
static size_t g_ntt_q_cap;

/* A matmul's K-chained blocks (blocked_gemm_kc) as n-tile launches, int32 C, host epilogue after. Either
 * C = A W^T (per n block, chunks of m blocks) or, when M is small, C^T = W A^T. Returns 1 when not taken
 * (no n-tile entry, or AIEDNN_NO_NTILE_MM; AIEDNN_NTILE_MM_T=1 allows the transposed form). */
static int blocked_gemm_ntile(const rt_impl *self, const aiednn_impl_t *blk, int64_t M, int64_t N, int64_t K,
                              const int8_t *A, const int8_t *B, int32_t *C, const void *bkey) {
    const aiednn_impl_t *kc = g_kcblk;
    const int64_t BM = kc->args[0].dims[0], BK = kc->args[0].dims[1], BN = kc->args[1].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nnb = (N + BN - 1) / BN, nmb = (M + BM - 1) / BM;
    const int emu = nte_on();
    const int64_t mpc = emu ? 32 / nkb : ntile_mblocks(nkb, 0);
    if (mpc < 1 || rt_env_on("AIEDNN_NO_NTILE_MM"))
        return 1;
    const int64_t ln = nnb * ((nmb + mpc - 1) / mpc);                               /* launches, C = A W^T */
    const int64_t lt = ((M + BN - 1) / BN) * (((N + BM - 1) / BM + mpc - 1) / mpc); /* C^T = W A^T */
    /* the transposed form re-stages the weights every call: opt-in */
    if (lt < ln && rt_env_on("AIEDNN_NTILE_MM_T")) {
        const int64_t nab = (M + BN - 1) / BN;
        int8_t *qb = (int8_t *)scratch(&g_ntt_q, &g_ntt_q_cap, (size_t)(nab * nkb * BN * BK));
        if (!qb)
            return AIEK_ERR_ALLOC;
        const uint64_t tb0 = rt_ticks();
        for (int64_t ab = 0; ab < nab; ab++) /* the activations' [BN][BK] blocks */
            for (int64_t kb = 0; kb < nkb; kb++) {
                int8_t *dst = qb + (ab * nkb + kb) * BN * BK;
                const int64_t k0 = kb * BK, kn = K - k0 < BK ? K - k0 : BK;
                for (int64_t r = 0; r < BN; r++) {
                    memset(dst + r * BK, 0, (size_t)BK);
                    if (ab * BN + r < M)
                        memcpy(dst + r * BK, A + (ab * BN + r) * K + k0, (size_t)kn);
                }
            }
        g_phase[RT_PH_APACK] += rt_ticks() - tb0;
        return ntile_mm(self, B, N, qb, M, K, mpc, emu, nullptr, C, N, 1);
    }
    int8_t *bp = b_blocks(B, N, K, blk, bkey);
    if (!bp)
        return AIEK_ERR_ALLOC;
    return ntile_mm(self, A, M, bp, N, K, mpc, emu, bkey, C, N, 0);
}

#endif /* AIEDNN_RT_NTILE_H */
