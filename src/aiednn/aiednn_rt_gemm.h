/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn GEMM family: generated aie: kernels, the blocked: matmul on an AIE
 * GEMM block, the composites that reduce convolution (im2col+), deconvolution
 * (col2im+) and inner product (ip+) to that matmul, and their ref:
 * implementations. Included once by aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_GEMM_H
#define AIEDNN_RT_GEMM_H

#include "aiednn_rt_internal.h"
#include "aie_modes_conv.h" /* direct conv: region layout + geometry (src/aietensorop/ops) */

// ── generated aie: kernels (exact AIEK_SPEC shapes) ─────────────────────────

static int spec_accepts(const aiednn_impl_t *g, const rt_prob *p, char *why, int cap) {
    static const char *role[] = {"src", "weights", "dst"};
    int64_t sym[AIEDNN_MAX_ARGS * AIEDNN_MAX_NDIMS];
    for (int i = 0; i < AIEDNN_MAX_ARGS * AIEDNN_MAX_NDIMS; i++)
        sym[i] = -1;
    if (g->kind != aiednn_matmul || p->kind != aiednn_matmul || g->nargs != 3) {
        snprintf(why, cap, "only plain matmul kernels are dispatched directly");
        return 0;
    }
    if (p->nmd != 3 || !p->transposed || p->has_attr) {
        snprintf(why, cap, "kernel takes no bias, no fused epilogue and [N,K] weights");
        return 0;
    }
    for (int a = 0; a < 3; a++) {
        const aiednn_arg_spec_t *s = &g->args[a];
        const aiednn_memory_desc_t *md = &p->md[a];
        if (s->data_type != md->data_type || s->ndims != md->ndims) {
            snprintf(why, cap, "%s is %s rank %d, spec wants %s rank %d", role[a], dt_str(md->data_type), md->ndims,
                     dt_str(s->data_type), s->ndims);
            return 0;
        }
        for (int d = 0; d < s->ndims; d++) {
            const int64_t want = s->dims[d], have = md->dims[d];
            if (want >= 0 && want != have) {
                snprintf(why, cap, "%s dim %d is %lld, spec wants %lld", role[a], d, (long long)have, (long long)want);
                return 0;
            }
            if (want <= AIEDNN_DIM_SYM(0)) {
                const int id = (int)(-2 - want);
                if (sym[id] >= 0 && sym[id] != have) {
                    snprintf(why, cap, "%s dim %d is %lld, the same spec name is %lld elsewhere", role[a], d,
                             (long long)have, (long long)sym[id]);
                    return 0;
                }
                sym[id] = have;
            }
        }
    }
    return 1;
}

static int acc_gen(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)bind;
    return spec_accepts(self->gen, p, why, cap);
}

static int run_gen(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    aiek_tensor t[3];
    for (int a = 0; a < 3; a++)
        fill_tensor(&t[a], &prim->prob.md[a], ptr[a]);
    ((rt_impl *)self)->launches++;
    const uint64_t te = rt_ticks();
    const int rc = self->gen->entry(t, 3);
    ((rt_impl *)self)->t_entry += rt_ticks() - te;
    return rc;
}

// ── blocked: any s8 x s8 -> s32 / s8 matmul on an AIE GEMM block ───────────

static int is_gemm_block(const aiednn_impl_t *g) {
    if (g->kind != aiednn_matmul || g->nargs != 3 || g->args[2].data_type != aiednn_s32 ||
        g->args[0].data_type != aiednn_s8 || g->args[1].data_type != aiednn_s8)
        return 0;
    for (int a = 0; a < 3; a++)
        if (g->args[a].ndims != 2 || g->args[a].dims[0] < 1 || g->args[a].dims[1] < 1)
            return 0;
    return g->args[0].dims[1] == g->args[1].dims[1] && g->args[2].dims[0] == g->args[0].dims[0] &&
           g->args[2].dims[1] == g->args[1].dims[0];
}

// s8 x s8 accumulates on the AIE; any dst / bias type the epilogue handles (create validated it)
static int mm_int8_problem(const rt_prob *p) {
    return p->md[0].data_type == aiednn_s8 && p->md[1].data_type == aiednn_s8;
}

static int acc_blocked(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    if (p->kind != aiednn_matmul || !mm_int8_problem(p)) {
        snprintf(why, cap, "only s8 x s8 matmul");
        return 0;
    }
    bind->blk = self->gen;
    return 1;
}

static void *g_bp, *g_ap, *g_cb;
static size_t g_bp_cap, g_ap_cap, g_cb_cap;

// C[M][N] = A[M][K] * B[N][K]^T on the block kernel; returns AIEK status.
/* packed [BN][BK] weight blocks of constant weights (aiednn_primitive_set_constant_weights),
 * kept for the process: B pointer + shape + block -> blocks; a full table packs per call */
typedef struct {
    const int8_t *b;
    int64_t n, k, bn, bk;
    int8_t *blocks;
} rt_bcache_ent;
#define RT_BCACHE_MAX 512
static rt_bcache_ent g_bcache[RT_BCACHE_MAX];
static int g_bcache_n;

static int8_t *rt_bcache_find(const void *key, int64_t N, int64_t K, const aiednn_impl_t *blk) {
    const int64_t BK = blk->args[0].dims[1], BN = blk->args[1].dims[0];
    for (int i = 0; i < g_bcache_n; i++) {
        const rt_bcache_ent *e = &g_bcache[i];
        if (e->b == key && e->n == N && e->k == K && e->bn == BN && e->bk == BK)
            return e->blocks;
    }
    return nullptr;
}

static int8_t *rt_bcache_add(const void *key, int64_t N, int64_t K, const aiednn_impl_t *blk, size_t bytes) {
    if (g_bcache_n == RT_BCACHE_MAX)
        return nullptr;
    int8_t *blocks = (int8_t *)malloc(bytes);
    if (blocks)
        g_bcache[g_bcache_n++] = {(const int8_t *)key, N, K, blk->args[1].dims[0], blk->args[0].dims[1], blocks};
    return blocks;
}

/* every zero-padded [BN][BK] block of B[N][K], block (nb, kb) at (nb * nkb + kb) * BN * BK: from
 * the constant-weight cache when *bkey* is set (packed on first use), else into scratch */
static int8_t *b_blocks(const int8_t *B, int64_t N, int64_t K, const aiednn_impl_t *blk, const void *bkey) {
    const int64_t BK = blk->args[0].dims[1], BN = blk->args[1].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nnb = (N + BN - 1) / BN;
    const size_t bytes = (size_t)(nnb * nkb * BN * BK);
    int8_t *bp = bkey ? rt_bcache_find(bkey, N, K, blk) : nullptr;
    if (bp)
        return bp;
    if (bkey)
        bp = rt_bcache_add(bkey, N, K, blk, bytes);
    if (!bp)
        bp = (int8_t *)scratch(&g_bp, &g_bp_cap, bytes);
    if (!bp)
        return nullptr;
    const uint64_t tb0 = rt_ticks();
    for (int64_t nb = 0; nb < nnb; nb++)
        for (int64_t kb = 0; kb < nkb; kb++) {
            int8_t *dst = bp + (nb * nkb + kb) * BN * BK;
            for (int64_t r = 0; r < BN; r++) {
                const int64_t n = nb * BN + r, k0 = kb * BK, kn = K - k0 < BK ? K - k0 : BK;
                memset(dst + r * BK, 0, (size_t)BK);
                if (n < N)
                    memcpy(dst + r * BK, B + n * K + k0, (size_t)kn);
            }
        }
    g_phase[RT_PH_BPACK] += rt_ticks() - tb0;
    return bp;
}

/* the chained K block (gemm_uk.cc gemm_kc_s8s32: A, B, flags s32[1], C s32): with it a K-split
 * GEMM accumulates on the core across launches -- no host int32 sum, and the last block may
 * requantize there (blocked_gemm_q) */
static const aiednn_impl_t *g_kcblk;

static int is_gemm_kc_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_matmul && g->nargs == 4 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[1].data_type == aiednn_s8 && g->args[2].data_type == aiednn_s32 && g->args[2].ndims == 1 &&
           g->args[2].dims[0] == 1 && g->args[3].data_type == aiednn_s32 && g->args[3].ndims == 2;
}

/* direct conv (aie_modes_conv.h): the chained / requantized blocks with A a raw input region per
 * mesh row and an extra geom int32 [16 * rows] argument */
static const aiednn_impl_t *g_ckcblk, *g_cqblk;
static int is_conv_kc_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_matmul && g->nargs == 5 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[1].data_type == aiednn_s8 && g->args[2].data_type == aiednn_s32 && g->args[2].ndims == 1 &&
           g->args[2].dims[0] == 1 && g->args[3].data_type == aiednn_s32 && g->args[3].ndims == 1 &&
           g->args[3].dims[0] == 16 * (g->args[0].dims[0] / 32) && g->args[4].data_type == aiednn_s32 &&
           g->args[4].ndims == 2;
}

static void *g_apk;
static size_t g_apk_cap;

/* the zero-padded [BM][BK] blocks of A rows m0 .. m0+mn, all k (block kb at kb * BM * BK) */
static int8_t *a_blocks(const int8_t *A, int64_t m0, int64_t mn, int64_t K, int64_t BM, int64_t BK) {
    const int64_t nkb = (K + BK - 1) / BK;
    int8_t *ap = (int8_t *)scratch(&g_apk, &g_apk_cap, (size_t)(nkb * BM * BK));
    if (!ap)
        return nullptr;
    const uint64_t ta0 = rt_ticks();
    for (int64_t kb = 0; kb < nkb; kb++) {
        const int64_t k0 = kb * BK, kn = K - k0 < BK ? K - k0 : BK;
        int8_t *dst = ap + kb * BM * BK;
        for (int64_t r = 0; r < BM; r++) {
            memset(dst + r * BK, 0, (size_t)BK);
            if (r < mn)
                memcpy(dst + r * BK, A + (m0 + r) * K + k0, (size_t)kn);
        }
    }
    g_phase[RT_PH_APACK] += rt_ticks() - ta0;
    return ap;
}

/* the mem tiles' B cache (gemm_uk.cc #pragma memtile_staging(weights)): the B block of the next
 * launch, named by (weights, n-block, k-block, shape) when the weights are a cached constant, else
 * 0. Consecutive launches with one key reuse the block on the array (absent on x86: no-op). */
extern "C" void aiednn_gemm_weights_key(uint64_t key) __attribute__((weak));
/* gemm_uk.cc's C launch buffer (board builds): the final launch of a block leaves C there
 * (flags bit 2) and the runtime copies only the rows / columns it keeps */
extern "C" const int32_t *aiednn_gemm_stage_c(void) __attribute__((weak));

static inline void rt_wkey(const void *bkey, int64_t nb, int64_t kb, int64_t N, int64_t K) {
    if (!aiednn_gemm_weights_key)
        return;
    uint64_t k = 0;
    if (bkey) {
        k = (uint64_t)(uintptr_t)bkey ^ (uint64_t)nb * 0x9E3779B97F4A7C15ull ^ (uint64_t)kb * 0xC2B2AE3D27D4EB4Full ^
            ((uint64_t)N << 40) ^ ((uint64_t)K << 20);
        k = k ? k : 1;
    }
    aiednn_gemm_weights_key(k);
}

static void *g_apall;
static size_t g_apall_cap;

/* every zero-padded [BM][BK] A block, (m-block, k-block) at (mb * nkb + kb) * BM * BK */
static int8_t *a_blocks_all(const int8_t *A, int64_t M, int64_t K, int64_t BM, int64_t BK) {
    const int64_t nkb = (K + BK - 1) / BK, nmb = (M + BM - 1) / BM;
    int8_t *ap = (int8_t *)scratch(&g_apall, &g_apall_cap, (size_t)(nmb * nkb * BM * BK));
    if (!ap)
        return nullptr;
    const uint64_t ta0 = rt_ticks();
    for (int64_t mb = 0; mb < nmb; mb++) {
        const int64_t m0 = mb * BM, mn = M - m0 < BM ? M - m0 : BM;
        for (int64_t kb = 0; kb < nkb; kb++) {
            const int64_t k0 = kb * BK, kn = K - k0 < BK ? K - k0 : BK;
            int8_t *dst = ap + (mb * nkb + kb) * BM * BK;
            for (int64_t r = 0; r < BM; r++) {
                memset(dst + r * BK, 0, (size_t)BK);
                if (r < mn)
                    memcpy(dst + r * BK, A + (m0 + r) * K + k0, (size_t)kn);
            }
        }
    }
    g_phase[RT_PH_APACK] += rt_ticks() - ta0;
    return ap;
}

/* GEMM through the chained block: n blocks outer, m blocks inner (consecutive launches share a B
 * block, which the mem tiles keep), the K blocks of one (m, n) block back to back on the core
 * (the core keeps the partial sum, the last launch returns the whole-K int32 block) */
static int blocked_gemm_kc(const rt_impl *self, const aiednn_impl_t *blk, int64_t M, int64_t N, int64_t K,
                           const int8_t *A, const int8_t *B, int32_t *C, const void *bkey) {
    const aiednn_impl_t *kc = g_kcblk;
    const int64_t BM = kc->args[0].dims[0], BK = kc->args[0].dims[1], BN = kc->args[1].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nnb = (N + BN - 1) / BN;
    int8_t *bp = b_blocks(B, N, K, blk, bkey);
    int32_t *cb = (int32_t *)scratch(&g_cb, &g_cb_cap, (size_t)(BM * BN) * sizeof(int32_t));
    if (!bp || !cb)
        return AIEK_ERR_ALLOC;
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, BK}, b_shape[2] = {BN, BK}, f_shape[1] = {1}, c_shape[2] = {BM, BN};
    aiek_tensor t[4] = {{nullptr, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK)},
                        {nullptr, 2, AIEK_INT, 8, 1, b_shape, (uint64_t)(BN * BK)},
                        {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                        {cb, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
    int8_t *apall = a_blocks_all(A, M, K, BM, BK);
    if (!apall)
        return AIEK_ERR_ALLOC;
    for (int64_t nb = 0; nb < nnb; nb++) {
        const int64_t n0 = nb * BN, nn = N - n0 < BN ? N - n0 : BN;
        for (int64_t m0 = 0; m0 < M; m0 += BM) {
            const int64_t mn = M - m0 < BM ? M - m0 : BM;
            const int8_t *ap = apall + (m0 / BM) * nkb * BM * BK;
            for (int64_t kb = 0; kb < nkb; kb++) {
                flags = (kb > 0 ? 1 : 0) | (kb < nkb - 1 ? 2 : 0) | (aiednn_gemm_stage_c ? 4 : 0);
                t[0].data = (void *)(ap + kb * BM * BK);
                t[1].data = bp + (nb * nkb + kb) * BN * BK;
                rt_wkey(bkey, nb, kb, N, K);
                const uint64_t te = rt_ticks();
                const int rc = kc->entry(t, 4);
                ((rt_impl *)self)->t_entry += rt_ticks() - te;
                ((rt_impl *)self)->launches++;
                if (rc != AIEK_OK)
                    return rc;
            }
            for (int64_t r = 0; r < mn; r++)
                memcpy(C + (m0 + r) * N + n0, (aiednn_gemm_stage_c ? aiednn_gemm_stage_c() : cb) + r * BN,
                       (size_t)nn * sizeof(int32_t));
        }
    }
    return AIEK_OK;
}

static int blocked_gemm_ntile(const rt_impl *self, const aiednn_impl_t *blk, int64_t M, int64_t N, int64_t K,
                              const int8_t *A, const int8_t *B, int32_t *C, const void *bkey);

/* C[M][N] = A[M][K] * B[N][K]^T in [BM][BK] x [BN][BK] blocks. *bkey*: B's identity when it
 * is a constant (its packed blocks are cached under it), NULL to pack per call */
static int blocked_gemm(const rt_impl *self, const aiednn_impl_t *blk, int64_t M, int64_t N, int64_t K, const int8_t *A,
                        const int8_t *B, int32_t *C, const void *bkey) {
    const int64_t BM = blk->args[0].dims[0], BK = blk->args[0].dims[1], BN = blk->args[1].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nnb = (N + BN - 1) / BN;
    if (g_kcblk && g_kcblk->args[0].dims[0] == BM && g_kcblk->args[0].dims[1] == BK && g_kcblk->args[1].dims[0] == BN) {
        const int rn = blocked_gemm_ntile(self, blk, M, N, K, A, B, C, bkey); /* aiednn_rt_ntile.h; 1: not taken */
        if (rn != 1)
            return rn;
        return blocked_gemm_kc(self, blk, M, N, K, A, B, C, bkey); // also K <= BK: n-outer order, B cache keys
    }
    const size_t bbytes = (size_t)(nnb * nkb * BN * BK);
    int8_t *bp = bkey ? rt_bcache_find(bkey, N, K, blk) : nullptr;
    const int fresh = !bp;
    if (!bp && bkey)
        bp = rt_bcache_add(bkey, N, K, blk, bbytes);
    if (!bp)
        bp = (int8_t *)scratch(&g_bp, &g_bp_cap, bbytes);
    int8_t *ap = (int8_t *)scratch(&g_ap, &g_ap_cap, (size_t)(BM * BK));
    int32_t *cb = (int32_t *)scratch(&g_cb, &g_cb_cap, (size_t)(BM * BN) * sizeof(int32_t));
    if (!bp || !ap || !cb)
        return AIEK_ERR_ALLOC;
    const uint64_t tb0 = rt_ticks();
    for (int64_t nb = 0; nb < nnb && fresh; nb++) // weights: every [BN][BK] block, zero padded, once
        for (int64_t kb = 0; kb < nkb; kb++) {
            int8_t *dst = bp + (nb * nkb + kb) * BN * BK;
            for (int64_t r = 0; r < BN; r++) {
                const int64_t n = nb * BN + r, k0 = kb * BK, kn = K - k0 < BK ? K - k0 : BK;
                memset(dst + r * BK, 0, (size_t)BK);
                if (n < N)
                    memcpy(dst + r * BK, B + n * K + k0, (size_t)kn);
            }
        }
    memset(C, 0, (size_t)(M * N) * sizeof(int32_t));
    g_phase[RT_PH_BPACK] += rt_ticks() - tb0;
    const int64_t a_shape[2] = {BM, BK}, b_shape[2] = {BN, BK}, c_shape[2] = {BM, BN};
    aiek_tensor t[3] = {{ap, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK)},
                        {nullptr, 2, AIEK_INT, 8, 1, b_shape, (uint64_t)(BN * BK)},
                        {cb, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
    for (int64_t m0 = 0; m0 < M; m0 += BM) {
        const int64_t mn = M - m0 < BM ? M - m0 : BM;
        for (int64_t kb = 0; kb < nkb; kb++) {
            const int64_t k0 = kb * BK, kn = K - k0 < BK ? K - k0 : BK;
            const uint64_t ta0 = rt_ticks();
            for (int64_t r = 0; r < BM; r++) {
                memset(ap + r * BK, 0, (size_t)BK);
                if (r < mn)
                    memcpy(ap + r * BK, A + (m0 + r) * K + k0, (size_t)kn);
            }
            g_phase[RT_PH_APACK] += rt_ticks() - ta0;
            for (int64_t nb = 0; nb < nnb; nb++) {
                const int64_t n0 = nb * BN, nn = N - n0 < BN ? N - n0 : BN;
                t[1].data = bp + (nb * nkb + kb) * BN * BK;
                const uint64_t te = rt_ticks();
                const int rc = blk->entry(t, 3);
                ((rt_impl *)self)->t_entry += rt_ticks() - te;
                ((rt_impl *)self)->launches++;
                if (rc != AIEK_OK)
                    return rc;
                const uint64_t tc0 = rt_ticks();
                for (int64_t r = 0; r < mn; r++) {
                    int32_t *c = C + (m0 + r) * N + n0;
                    const int32_t *s = cb + r * BN;
                    for (int64_t j = 0; j < nn; j++)
                        c[j] += s[j];
                }
                g_phase[RT_PH_ACCUM] += rt_ticks() - tc0;
            }
        }
    }
    return AIEK_OK;
}

// batch geometry of a matmul problem: dst [..., M, N]
typedef struct {
    int64_t M, N, K, batch;
    aiednn_memory_desc_t bs, bw, bd; // batch-only views (rank nd - 2, at least 1)
} mm_geo;

static void mm_geometry(const rt_prob *p, mm_geo *g) {
    const aiednn_memory_desc_t *s = &p->md[0], *w = &p->md[1], *d = &p->md[2];
    const int nd = d->ndims;
    g->M = d->dims[nd - 2];
    g->N = d->dims[nd - 1];
    g->K = s->dims[nd - 1];
    memset(&g->bs, 0, sizeof g->bs);
    memset(&g->bw, 0, sizeof g->bw);
    memset(&g->bd, 0, sizeof g->bd);
    const int nb = nd > 2 ? nd - 2 : 1;
    g->bs.ndims = g->bw.ndims = g->bd.ndims = nb;
    g->batch = 1;
    for (int i = 0; i < nb; i++) {
        g->bs.dims[i] = nd > 2 ? s->dims[i] : 1;
        g->bw.dims[i] = nd > 2 ? w->dims[i] : 1;
        g->bd.dims[i] = nd > 2 ? d->dims[i] : 1;
        g->batch *= g->bd.dims[i];
    }
}

/* weights shared by every batch (an nn.Linear on [B, S, D]) and src / dst not broadcast: the batch is just more
 * rows of one GEMM ([B * M, K] x [K, N]; src and dst are contiguous), so its blocks fill instead of B small ones */
static void mm_fold_batch(mm_geo *g) {
    if (g->batch <= 1)
        return;
    for (int i = 0; i < g->bd.ndims; i++)
        if (g->bw.dims[i] != 1 || g->bs.dims[i] != g->bd.dims[i])
            return;
    g->M *= g->batch, g->batch = 1;
    for (int i = 0; i < g->bd.ndims; i++)
        g->bs.dims[i] = g->bd.dims[i] = 1;
}

static void *g_bt, *g_c32;
static size_t g_bt_cap, g_c32_cap;

// epilogue + store: c32 [M][N] of batch b into dst (channel = last dim)
static void mm_epilogue(const rt_prob *p, const mm_geo *g, int64_t b, const int32_t *c32, void *const *ptr) {
    rt_epi E_; // epilogue resolved once per call (bit-identical to rt_epilogue)
    rt_epi_init(&E_, p, ptr);
    const uint64_t t0 = rt_ticks();
    for (int64_t m = 0; m < g->M; m++) // one row of N channels at a time
        rt_epi_row(&E_, (b * g->M + m) * g->N, 0, 1, c32 + m * g->N, 1, g->N);
    g_phase[RT_PH_EPI] += rt_ticks() - t0;
}

static int run_blocked(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    mm_geo g;
    mm_geometry(p, &g);
    mm_fold_batch(&g);
    int32_t *c32 = (int32_t *)scratch(&g_c32, &g_c32_cap, (size_t)(g.M * g.N) * sizeof(int32_t));
    int8_t *bt = p->transposed ? nullptr : (int8_t *)scratch(&g_bt, &g_bt_cap, (size_t)(g.N * g.K));
    if (!c32 || (!p->transposed && !bt))
        return AIEK_ERR_ALLOC;
    for (int64_t b = 0; b < g.batch; b++) {
        const int8_t *A = (const int8_t *)ptr[0] + bcast_off(&g.bs, &g.bd, b) * g.M * g.K;
        const int8_t *W = (const int8_t *)ptr[1] + bcast_off(&g.bw, &g.bd, b) * g.N * g.K;
        const void *bkey = g_exec_const_w ? (const void *)W : nullptr; // the caller's weights, before any transpose
        if (!p->transposed && !(bkey && rt_bcache_find(bkey, g.N, g.K, prim->bind.blk))) { // [K][N] -> [N][K]
            for (int64_t k = 0; k < g.K; k++)
                for (int64_t n = 0; n < g.N; n++)
                    bt[n * g.K + k] = W[k * g.N + n];
            W = bt;
        }
        const int rc = blocked_gemm(self, prim->bind.blk, g.M, g.N, g.K, A, W, c32, bkey);
        if (rc != AIEK_OK)
            return rc;
        mm_epilogue(p, &g, b, c32, ptr);
    }
    return AIEK_OK;
}

// ── ref: matmul / inner product ───────────────────────────────────────────

static int run_ref_matmul(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    rt_epi E_; // epilogue resolved once per call (bit-identical to rt_epilogue)
    rt_epi_init(&E_, p, ptr);
    mm_geo g;
    mm_geometry(p, &g);
    const aiednn_data_type_t sdt = p->md[0].data_type, wdt = p->md[1].data_type;
    const int ss = rt_slot(p, AIEDNN_ARG_ATTR_WEI_SCALES), zs = rt_slot(p, AIEDNN_ARG_ATTR_WEI_ZERO_POINTS);
    const int64_t gk = ss >= 0 ? g.K / p->md[ss].dims[0] : g.K; /* weight group along K */
    auto wv = [&](int64_t wi, int64_t k,
                  int64_t n) { /* a weight, dequantized per group */
                               const double c = rt_ld(wdt, ptr[1], wi);
                               if (ss < 0)
                                   return c - p->wei_zp;
                               const int64_t gi = (k / gk) * g.N + n;
                               return (c - (zs >= 0 ? rt_ld(p->md[zs].data_type, ptr[zs], gi) : 0.0)) *
                                      rt_ld(p->md[ss].data_type, ptr[ss], gi);
    };
    for (int64_t b = 0; b < g.batch; b++) {
        const int64_t so = bcast_off(&g.bs, &g.bd, b) * g.M * g.K, wo = bcast_off(&g.bw, &g.bd, b) * g.N * g.K;
        for (int64_t m = 0; m < g.M; m++)
            for (int64_t n = 0; n < g.N; n++) {
                double acc = 0;
                for (int64_t k = 0; k < g.K; k++) {
                    const int64_t wi = p->transposed ? n * g.K + k : k * g.N + n;
                    acc += (rt_ld(sdt, ptr[0], so + m * g.K + k) - p->src_zp) * wv(wo + wi, k, n);
                }
                rt_epi_put(&E_, (b * g.M + m) * g.N + n, n, acc);
            }
    }
    return AIEK_OK;
}

static int acc_any(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    (void)p;
    (void)bind;
    (void)why;
    (void)cap;
    return 1;
}

// inner product as a [N, IC] x [OC, IC]^T matmul
static void ip_as_matmul(const rt_prob *p, rt_prob *m) {
    *m = *p;
    m->kind = aiednn_matmul;
    const int64_t N = p->md[0].dims[0], K = md_numel(&p->md[0]) / N, OC = p->md[1].dims[0];
    md2(&m->md[0], N, K, p->md[0].data_type);
    md2(&m->md[1], OC, K, p->md[1].data_type);
    m->transposed = 1;
    const int bs = rt_slot(p, AIEDNN_ARG_BIAS);
    if (bs >= 0) { // [OC] -> [1, OC]
        md2(&m->md[bs], 1, OC, p->md[bs].data_type);
    }
}

static int acc_ip(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (p->md[0].data_type == aiednn_u8) { /* would tie with gemm+mm, which takes it directly */
        snprintf(why, cap, "u8 src: gemm+mm");
        return 0;
    }
    rt_prob m;
    ip_as_matmul(p, &m);
    const rt_impl *inner = nullptr;
    rt_bind ib = {nullptr, nullptr};
    if (dispatch(&m, &inner, &ib, 1) != aiednn_success || strncmp(inner->name, "ref:", 4) == 0) {
        snprintf(why, cap, "its matmul would run on %s", inner ? inner->name : "nothing");
        return 0;
    }
    bind->inner = inner;
    bind->blk = ib.blk;
    return 1;
}

static int run_ip(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    rt_prob m;
    ip_as_matmul(&prim->prob, &m);
    const rt_bind b = {nullptr, prim->bind.blk};
    const rt_impl *inner = prim->bind.inner ? prim->bind.inner : self;
    return run_sub(inner, &m, &b, ptr);
}

static int run_ref_ip(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    if (prim->prob.prop != RT_FWD_INF)
        return run_bwd(&prim->prob, ptr); /* training (aiednn_rt_bwd.h) */
    rt_prob m;
    ip_as_matmul(&prim->prob, &m);
    aiednn_primitive sub;
    memset(&sub, 0, sizeof sub);
    sub.prob = m;
    return run_ref_matmul(self, &sub, ptr);
}

// ── convolution geometry ──────────────────────────────────────────────────

typedef struct {
    int64_t N, C, H, W, O, KH, KW, Ho, Wo, G, Cg, Og, s[2], pl[2], d[2];
    /* images stacked along the GEMM's M (1: one image per pipe call). With nimg > 1 the im2col pipes
     * see M = nimg * Ho * Wo rows: row m is pixel m % (Ho*Wo) of image m / (Ho*Wo); sources and
     * destinations step by a whole image (C*H*W in, O*Ho*Wo out) -- a batch fills the M blocks */
    int64_t nimg;
    /* rows per image in that stacked M: Ho*Wo, or rounded up to 32 for direct conv (a mesh row's 32
     * pixels then never straddle two images); rows Ho*Wo .. pstride-1 of an image are padding */
    int64_t pstride;
    /* nhwc operands (aiednn_rt_layout.h; im2col's native paths, groups == 1): src read as [n][H][W][C] (no
     * hwc_transpose), dst and the fused residual written / read as [n][Ho*Wo][O] rows */
    int src_hwc, dst_hwc, res_hwc;
} conv_geo;

/* GEMM rows of a conv pipe call: every merged image's output pixels (+ per-image padding rows) */
static inline int64_t conv_m(const conv_geo *c) { return c->nimg * c->pstride; }

static void conv_geometry(const rt_prob *p, int deconv, conv_geo *g) {
    const aiednn_memory_desc_t *s = &p->md[0], *w = &p->md[1], *d = &p->md[2];
    g->N = s->dims[0];
    g->C = s->dims[1];
    g->H = s->dims[2];
    g->W = s->dims[3];
    g->O = d->dims[1];
    g->Ho = d->dims[2];
    g->Wo = d->dims[3];
    g->KH = w->dims[2];
    g->KW = w->dims[3];
    g->G = p->conv.groups;
    g->Cg = g->C / g->G;
    g->Og = g->O / g->G;
    for (int i = 0; i < 2; i++) {
        g->s[i] = p->conv.strides[i];
        g->pl[i] = p->conv.padding_l[i];
        g->d[i] = p->conv.dilates[i] + 1;
    }
    g->nimg = 1;
    g->pstride = g->Ho * g->Wo;
    g->src_hwc = md_is_nhwc(s), g->dst_hwc = md_is_nhwc(d), g->res_hwc = 0;
    for (int k = 0; k < p->npo; k++)
        if (p->po[k].kind == aiednn_post_op_binary)
            g->res_hwc = md_is_nhwc(&p->po[k].src1);
    (void)deconv;
}

// ── ref: convolution / deconvolution ──────────────────────────────────────

static void *g_crow;
static size_t g_crow_cap;

/* output columns [lo, hi) whose input column ow*s + off lies inside [0, W) */
static void conv_ow_range(int64_t off, int64_t s, int64_t W, int64_t Wo, int64_t *lo, int64_t *hi) {
    *lo = off < 0 ? (-off + s - 1) / s : 0;
    *hi = W - 1 - off < 0 ? 0 : (W - 1 - off) / s + 1;
    if (*hi > Wo)
        *hi = Wo;
}

/* Integer convolution (s8/u8 x s8): one int32 accumulator row per output row,
 * each tap a contiguous multiply-add the compiler vectorizes (NEON on the A72).
 * The int32 sum is the exact value the double loop below forms, so the result
 * is bit-identical; this is what serves depthwise and narrow grouped convs. */
template <typename T>
static int ref_conv_int(const rt_prob *p, const conv_geo *g, const T *src, const int8_t *wei, void *const *ptr) {
    rt_epi E_; // epilogue resolved once per call (bit-identical to rt_epilogue)
    rt_epi_init(&E_, p, ptr);
    int32_t *row = (int32_t *)scratch(&g_crow, &g_crow_cap, (size_t)g->Wo * sizeof(int32_t));
    if (!row)
        return AIEK_ERR_ALLOC;
    for (int64_t n = 0; n < g->N; n++)
        for (int64_t o = 0; o < g->O; o++) {
            const int64_t grp = o / g->Og;
            for (int64_t oh = 0; oh < g->Ho; oh++) {
                for (int64_t ow = 0; ow < g->Wo; ow++)
                    row[ow] = 0;
                for (int64_t c = 0; c < g->Cg; c++) {
                    const T *x = src + (n * g->C + grp * g->Cg + c) * g->H * g->W;
                    const int8_t *wk = wei + (o * g->Cg + c) * g->KH * g->KW;
                    for (int64_t kh = 0; kh < g->KH; kh++) {
                        const int64_t ih = oh * g->s[0] - g->pl[0] + kh * g->d[0];
                        if (ih < 0 || ih >= g->H)
                            continue;
                        const T *xr = x + ih * g->W;
                        for (int64_t kw = 0; kw < g->KW; kw++) {
                            const int32_t wv = wk[kh * g->KW + kw];
                            const int64_t off = kw * g->d[1] - g->pl[1], s = g->s[1];
                            int64_t lo, hi;
                            conv_ow_range(off, s, g->W, g->Wo, &lo, &hi);
                            if (wv == 0)
                                continue;
                            if (s == 1)
                                for (int64_t ow = lo; ow < hi; ow++)
                                    row[ow] += wv * (int32_t)xr[ow + off];
                            else
                                for (int64_t ow = lo; ow < hi; ow++)
                                    row[ow] += wv * (int32_t)xr[ow * s + off];
                        }
                    }
                }
                const int64_t d0 = ((n * g->O + o) * g->Ho + oh) * g->Wo;
                rt_epi_row(&E_, d0, o, 0, row, 1, g->Wo);
            }
        }
    return AIEK_OK;
}

static int run_ref_conv3d(const rt_prob *p, void *const *ptr, int deconv); /* aiednn_rt_nd.h */

static int run_ref_conv(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    if (p->md[0].ndims == 5)
        return run_ref_conv3d(p, ptr, 0);
    rt_epi E_; // epilogue resolved once per call (bit-identical to rt_epilogue)
    rt_epi_init(&E_, p, ptr);
    conv_geo g;
    conv_geometry(p, 0, &g);
    const aiednn_data_type_t sdt = p->md[0].data_type, wdt = p->md[1].data_type;
    if (wdt == aiednn_s8 && sdt == aiednn_s8 && !p->src_zp && !p->wei_zp)
        return ref_conv_int(p, &g, (const int8_t *)ptr[0], (const int8_t *)ptr[1], ptr);
    if (wdt == aiednn_s8 && sdt == aiednn_u8 && !p->src_zp && !p->wei_zp)
        return ref_conv_int(p, &g, (const uint8_t *)ptr[0], (const int8_t *)ptr[1], ptr);
    for (int64_t n = 0; n < g.N; n++)
        for (int64_t o = 0; o < g.O; o++) {
            const int64_t grp = o / g.Og;
            for (int64_t oh = 0; oh < g.Ho; oh++)
                for (int64_t ow = 0; ow < g.Wo; ow++) {
                    double acc = 0;
                    for (int64_t c = 0; c < g.Cg; c++)
                        for (int64_t kh = 0; kh < g.KH; kh++) {
                            const int64_t ih = oh * g.s[0] - g.pl[0] + kh * g.d[0];
                            if (ih < 0 || ih >= g.H)
                                continue;
                            for (int64_t kw = 0; kw < g.KW; kw++) {
                                const int64_t iw = ow * g.s[1] - g.pl[1] + kw * g.d[1];
                                if (iw < 0 || iw >= g.W)
                                    continue;
                                acc += (rt_ld(sdt, ptr[0], ((n * g.C + grp * g.Cg + c) * g.H + ih) * g.W + iw) -
                                        p->src_zp) *
                                       (rt_ld(wdt, ptr[1], ((o * g.Cg + c) * g.KH + kh) * g.KW + kw) - p->wei_zp);
                            }
                        }
                    rt_epi_put(&E_, ((n * g.O + o) * g.Ho + oh) * g.Wo + ow, o, acc);
                }
        }
    return AIEK_OK;
}

static void *g_dacc;
static size_t g_dacc_cap;

static int run_ref_deconv(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop != RT_FWD_INF)
        return run_bwd(p, ptr); /* training (aiednn_rt_bwd.h) */
    if (p->md[0].ndims == 5)
        return run_ref_conv3d(p, ptr, 1);
    rt_epi E_; // epilogue resolved once per call (bit-identical to rt_epilogue)
    rt_epi_init(&E_, p, ptr);
    conv_geo g;
    conv_geometry(p, 1, &g);
    const aiednn_data_type_t sdt = p->md[0].data_type, wdt = p->md[1].data_type;
    const int64_t nd = md_numel(&p->md[2]);
    double *acc = (double *)scratch(&g_dacc, &g_dacc_cap, (size_t)nd * sizeof(double));
    if (!acc)
        return AIEK_ERR_ALLOC;
    for (int64_t i = 0; i < nd; i++)
        acc[i] = 0;
    for (int64_t n = 0; n < g.N; n++)
        for (int64_t ci = 0; ci < g.C; ci++) {
            const int64_t grp = ci / g.Cg, c = ci % g.Cg;
            for (int64_t h = 0; h < g.H; h++)
                for (int64_t w = 0; w < g.W; w++) {
                    const double x = rt_ld(sdt, ptr[0], ((n * g.C + ci) * g.H + h) * g.W + w) - p->src_zp;
                    for (int64_t og = 0; og < g.Og; og++) {
                        const int64_t o = grp * g.Og + og;
                        for (int64_t kh = 0; kh < g.KH; kh++) {
                            const int64_t oh = h * g.s[0] - g.pl[0] + kh * g.d[0];
                            if (oh < 0 || oh >= g.Ho)
                                continue;
                            for (int64_t kw = 0; kw < g.KW; kw++) {
                                const int64_t ow = w * g.s[1] - g.pl[1] + kw * g.d[1];
                                if (ow < 0 || ow >= g.Wo)
                                    continue;
                                acc[((n * g.O + o) * g.Ho + oh) * g.Wo + ow] +=
                                    x * (rt_ld(wdt, ptr[1], ((o * g.Cg + c) * g.KH + kh) * g.KW + kw) - p->wei_zp);
                            }
                        }
                    }
                }
        }
    for (int64_t i = 0; i < nd; i++)
        rt_epi_put(&E_, i, (i / (g.Ho * g.Wo)) % g.O, acc[i]);
    return AIEK_OK;
}

// ── im2col+<matmul>: convolution as one GEMM per (image, group) ────────────

/* A grouped conv with fewer output channels per group than this goes to ref:
 * -- each group is its own GEMM launch with N = channels per group, so a
 * depthwise conv would launch once per channel on a block that is 1/128 used. */
#define RT_MIN_GROUP_N 8

static int conv_int8_problem(const rt_prob *p) { return mm_int8_problem(p); }

static int acc_gemm_composite(const rt_prob *g, rt_bind *bind, char *why, int cap) {
    const rt_impl *inner = nullptr;
    rt_bind ib = {nullptr, nullptr};
    if (dispatch(g, &inner, &ib, 1) != aiednn_success || strncmp(inner->name, "ref:", 4) == 0) {
        snprintf(why, cap, "its GEMM %lldx%lldx%lld would run on %s", (long long)g->md[0].dims[0],
                 (long long)g->md[1].dims[0], (long long)g->md[0].dims[1], inner ? inner->name : "nothing");
        return 0;
    }
    bind->inner = inner;
    bind->blk = ib.blk;
    return 1;
}

/* nhwc on im2col's native paths: one group (a group's channels are not contiguous in a pixel), and binary
 * post-op sources laid out like dst (the epilogue indexes them with dst's element index) */
static int conv_fmt_ok(const rt_prob *p, const conv_geo *c, char *why, int cap, int any_groups = 0) {
    const aiednn_memory_desc_t *d = &p->md[2];
    if (c->G != 1 && !any_groups) {
        snprintf(why, cap, "nhwc: one group only");
        return 0;
    }
    for (int k = 0; k < p->npo; k++) {
        const aiednn_memory_desc_t *r = &p->po[k].src1;
        if (p->po[k].kind == aiednn_post_op_binary &&
            (r->ndims != 4 || md_is_nhwc(r) != md_is_nhwc(d) || memcmp(r->dims, d->dims, 4 * sizeof(int64_t)))) {
            snprintf(why, cap, "nhwc: a binary post-op source must be a full tensor in dst's format");
            return 0;
        }
    }
    const int bs = rt_slot(p, AIEDNN_ARG_BIAS);
    if (bs >= 0 && p->md[bs].ndims != 1) {
        snprintf(why, cap, "nhwc: bias [O] only");
        return 0;
    }
    return 1;
}

static int acc_im2col(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (!conv_int8_problem(p)) {
        snprintf(why, cap, "only s8 x s8 convolution");
        return 0;
    }
    conv_geo c;
    conv_geometry(p, 0, &c);
    if (c.G > 1 && c.Og < RT_MIN_GROUP_N) {
        snprintf(why, cap,
                 "%lld output channels per group: one GEMM launch per channel, the AIE block "
                 "mostly padding (depthwise)",
                 (long long)c.Og);
        return 0;
    }
    if (rt_prob_has_fmt(p) && !conv_fmt_ok(p, &c, why, cap))
        return 0;
    rt_prob g;
    gemm_prob(&g, c.Ho * c.Wo, c.Og, c.Cg * c.KH * c.KW, aiednn_s8);
    return acc_gemm_composite(&g, bind, why, cap);
}

static void *g_col, *g_gout;
static size_t g_col_cap, g_gout_cap;

// ── GEMM + integer requant on the AIE (gemm_uk.cc 0xF1 mode) ──────────────
/* gemm_q block: A [BM][BK] s8, B [BN][BK] s8, params s32 [6 + 6*BN] = {zp, lo, hi, flags,
 * B int64[BN], M[BN], sh[BN], has_res, rzp, R int64[BN]} (rt_fx_q), residual s8 [BM][BN], C s8 [BM][BN] */
static const aiednn_impl_t *g_qblk;

static int is_conv_q_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_matmul && g->nargs == 6 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[1].data_type == aiednn_s8 && g->args[2].data_type == aiednn_s32 && g->args[2].ndims == 1 &&
           g->args[2].dims[0] >= 6 + 6 * g->args[1].dims[0] && g->args[3].data_type == aiednn_s8 &&
           g->args[4].data_type == aiednn_s32 && g->args[4].ndims == 1 &&
           g->args[4].dims[0] == 16 * (g->args[0].dims[0] / 32) && g->args[5].data_type == aiednn_s8;
}

static int is_gemm_q_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_matmul && g->nargs == 5 && g->args[0].data_type == aiednn_s8 &&
           g->args[1].data_type == aiednn_s8 && g->args[2].data_type == aiednn_s32 && g->args[2].ndims == 1 &&
           g->args[3].data_type == aiednn_s8 && g->args[3].ndims == 2 && g->args[4].data_type == aiednn_s8 &&
           g->args[4].ndims == 2 && g->args[0].ndims == 2 && g->args[2].dims[0] >= 6 + 6 * g->args[1].dims[0];
}

static void *g_qa, *g_qb, *g_qc, *g_qp;
static size_t g_qa_cap, g_qb_cap, g_qc_cap, g_qp_cap;

/* C8[M][N] = requant(A[M][K] * B[N][K]^T) for output channels ch0 .. ch0+N of E: the core
 * requantizes (0xF1). K <= BK: one launch per block; K > BK (with the chained block): the K
 * blocks before the last accumulate on the core (gemm_kc, flag "more"), the last continues and
 * requantizes. *bkey*: B's identity when constant (packed blocks cached) */
static void *g_qr;
static size_t g_qr_cap;

/* *res*: the fused residual of this (image, group) when E->fx_r is set, element (m, n) at
 * res[n * res_cs + m] (a conv's dst layout: channel planes of M pixels) */
static int blocked_gemm_q(const rt_impl *self, const rt_epi *E, int64_t ch0, int64_t M, int64_t N, int64_t K,
                          const int8_t *A, const int8_t *B, int8_t *C8, const void *bkey, const int8_t *res,
                          int64_t res_cs) {
    const aiednn_impl_t *q = g_qblk, *kc = g_kcblk;
    const int64_t BM = q->args[0].dims[0], BK = q->args[0].dims[1], BN = q->args[1].dims[0], NP = q->args[2].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nnb = (N + BN - 1) / BN;
    if (nkb > 1 && !kc)
        return AIEK_ERR_PARAM;
    int8_t *bp = b_blocks(B, N, K, q, bkey);
    int8_t *cb = (int8_t *)scratch(&g_qc, &g_qc_cap, (size_t)(BM * BN));
    int32_t *c32 = (int32_t *)scratch(&g_cb, &g_cb_cap, (size_t)(BM * BN) * sizeof(int32_t)); // gemm_kc's C (unused)
    int32_t *qp = (int32_t *)scratch(&g_qp, &g_qp_cap, (size_t)NP * sizeof(int32_t));
    int8_t *rb = (int8_t *)scratch(&g_qr, &g_qr_cap, (size_t)(BM * BN));
    if (!bp || !cb || !c32 || !qp || !rb)
        return AIEK_ERR_ALLOC;
    memset(rb, 0, (size_t)(BM * BN));
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, BK}, b_shape[2] = {BN, BK}, p_shape[1] = {NP}, c_shape[2] = {BM, BN},
                  f_shape[1] = {1};
    aiek_tensor t[5] = {{nullptr, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK)},
                        {nullptr, 2, AIEK_INT, 8, 1, b_shape, (uint64_t)(BN * BK)},
                        {qp, 1, AIEK_INT, 32, 1, p_shape, (uint64_t)NP * 4},
                        {rb, 2, AIEK_INT, 8, 1, c_shape, (uint64_t)(BM * BN)},
                        {cb, 2, AIEK_INT, 8, 1, c_shape, (uint64_t)(BM * BN)}};
    aiek_tensor tk[4] = {t[0],
                         t[1],
                         {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                         {c32, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
    for (int64_t nb = 0; nb < nnb; nb++) {
        const int64_t n0 = nb * BN, nn = N - n0 < BN ? N - n0 : BN;
        const uint64_t tb0 = rt_ticks();
        memset(qp, 0, (size_t)NP * sizeof(int32_t));
        qp[0] = (int32_t)E->dzp, qp[1] = E->fx_lo, qp[2] = E->fx_hi;
        int64_t *qb = (int64_t *)(qp + 4);
        for (int64_t j = 0; j < BN; j++) { // padding channels: any valid multiplier, discarded
            const int64_t ch = ch0 + n0 + j;
            qb[j] = j < nn ? E->fx_b[ch] : 0;
            qp[4 + 2 * BN + j] = j < nn ? E->fx_m[ch] : (1 << 30);
            qp[4 + 3 * BN + j] = j < nn ? E->fx_sh[ch] : 31;
        }
        if (res) { // {has_res, rzp, R int64[BN]}
            qp[4 + 4 * BN] = 1, qp[5 + 4 * BN] = E->fx_rzp;
            int64_t *rr = (int64_t *)(qp + 6 + 4 * BN);
            for (int64_t j = 0; j < BN; j++)
                rr[j] = j < nn ? E->fx_rm[ch0 + n0 + j] : 0;
        }
        g_phase[RT_PH_BPACK] += rt_ticks() - tb0;
        for (int64_t m0 = 0; m0 < M; m0 += BM) {
            const int64_t mn = M - m0 < BM ? M - m0 : BM;
            int8_t *ap = a_blocks(A, m0, mn, K, BM, BK);
            if (!ap)
                return AIEK_ERR_ALLOC;
            if (res) { // residual block [m][n] from the channel planes
                const uint64_t tr0 = rt_ticks();
                for (int64_t j = 0; j < nn; j++) {
                    const int8_t *src = res + (n0 + j) * res_cs + m0;
                    for (int64_t m = 0; m < mn; m++)
                        rb[m * BN + j] = src[m];
                }
                g_phase[RT_PH_APACK] += rt_ticks() - tr0;
            }
            for (int64_t kb = 0; kb < nkb; kb++) {
                const int last = kb == nkb - 1;
                aiek_tensor *tt = last ? t : tk;
                tt[0].data = ap + kb * BM * BK;
                tt[1].data = bp + (nb * nkb + kb) * BN * BK;
                flags = (kb > 0 ? 1 : 0) | 2;
                qp[3] = kb > 0 ? 1 : 0; // 0xF1 continues the chain
                const uint64_t te = rt_ticks();
                rt_wkey(bkey, nb, kb, N, K);
                const int rc = last ? q->entry(t, 5) : kc->entry(tk, 4);
                ((rt_impl *)self)->t_entry += rt_ticks() - te;
                ((rt_impl *)self)->launches++;
                if (rc != AIEK_OK)
                    return rc;
            }
            for (int64_t r = 0; r < mn; r++)
                memcpy(C8 + (m0 + r) * N + n0, cb + r * BN, (size_t)nn);
        }
    }
    return AIEK_OK;
}

/* col [Ho*Wo][K] of one (image, group), x = its first input channel plane: row m = (oh, ow),
 * column k = (ci, kh, kw) (OIHW order), zero outside the input. A 1x1 unpadded conv is a
 * (sub-sampled) [C][HW] -> [HW][C] transpose, done in cache tiles; otherwise one input row
 * (ci, ih) is copied into the K-strided rows of an output row oh, with the bounds checks only
 * on the border columns. (The per-byte, fully checked loop cost resnet18 132 ms on the A72.) */
static void im2col_pack(const conv_geo *c, const int8_t *x, int8_t *col) {
    const int64_t K = c->Cg * c->KH * c->KW, M = c->Ho * c->Wo, HW = c->H * c->W;
    if (c->KH == 1 && c->KW == 1 && c->pl[0] == 0 && c->pl[1] == 0) {
        for (int64_t m0 = 0; m0 < M; m0 += 64)
            for (int64_t c0 = 0; c0 < c->Cg; c0 += 64) {
                const int64_t mn = M - m0 < 64 ? M - m0 : 64, cn = c->Cg - c0 < 64 ? c->Cg - c0 : 64;
                for (int64_t m = m0; m < m0 + mn; m++) {
                    const int64_t oh = m / c->Wo, ow = m - oh * c->Wo;
                    const int8_t *in = x + c0 * HW + oh * c->s[0] * c->W + ow * c->s[1];
                    int8_t *dst = col + m * K + c0;
                    for (int64_t ci = 0; ci < cn; ci++)
                        dst[ci] = in[ci * HW];
                }
            }
        return;
    }
    const int64_t s1 = c->s[1], d1 = c->d[1], p1 = c->pl[1], KW = c->KW;
    int64_t ow_lo = (p1 + s1 - 1) / s1, ow_hi = (c->W - 1 - (KW - 1) * d1 + p1) / s1; // all taps inside
    if (c->W - 1 - (KW - 1) * d1 + p1 < 0)
        ow_hi = -1;
    if (ow_hi > c->Wo - 1)
        ow_hi = c->Wo - 1;
    for (int64_t oh = 0; oh < c->Ho; oh++) {
        int8_t *orow = col + oh * c->Wo * K;
        for (int64_t ci = 0; ci < c->Cg; ci++)
            for (int64_t kh = 0; kh < c->KH; kh++) {
                const int64_t ih = oh * c->s[0] - c->pl[0] + kh * c->d[0], kb = (ci * c->KH + kh) * KW;
                if (ih < 0 || ih >= c->H) {
                    for (int64_t ow = 0; ow < c->Wo; ow++)
                        memset(orow + ow * K + kb, 0, (size_t)KW);
                    continue;
                }
                const int8_t *in = x + (ci * c->H + ih) * c->W;
                for (int64_t ow = 0; ow < c->Wo; ow++) {
                    int8_t *dst = orow + ow * K + kb;
                    const int64_t iw0 = ow * s1 - p1;
                    if (ow >= ow_lo && ow <= ow_hi) {
                        for (int64_t kw = 0; kw < KW; kw++)
                            dst[kw] = in[iw0 + kw * d1];
                    } else {
                        for (int64_t kw = 0; kw < KW; kw++) {
                            const int64_t iw = iw0 + kw * d1;
                            dst[kw] = iw >= 0 && iw < c->W ? in[iw] : 0;
                        }
                    }
                }
            }
    }
}

/* K in (kh, kw, ci) order (NHWC-style): weights w[o][ci][kh][kw] of one group -> wr[o][kh][kw][ci],
 * cached by pointer for constant weights (aiednn_primitive_set_constant_weights) */
typedef struct {
    const int8_t *w;
    int64_t o, k;
    int8_t *wr;
} rt_wr_ent;
static rt_wr_ent g_whwc[RT_BCACHE_MAX];
static int g_whwc_n;
static void *g_wrs, *g_xhwc;
static size_t g_wrs_cap, g_xhwc_cap;

static const int8_t *w_hwc(const conv_geo *c, const int8_t *w) {
    const int64_t O = c->Og, K = c->Cg * c->KH * c->KW, KK = c->KH * c->KW;
    for (int i = 0; g_exec_const_w && i < g_whwc_n; i++)
        if (g_whwc[i].w == w && g_whwc[i].o == O && g_whwc[i].k == K)
            return g_whwc[i].wr;
    int8_t *wr = nullptr;
    if (g_exec_const_w && g_whwc_n < RT_BCACHE_MAX && (wr = (int8_t *)malloc((size_t)(O * K))))
        g_whwc[g_whwc_n++] = {w, O, K, wr};
    if (!wr)
        wr = (int8_t *)scratch(&g_wrs, &g_wrs_cap, (size_t)(O * K));
    if (!wr)
        return nullptr;
    for (int64_t o = 0; o < O; o++)
        for (int64_t ci = 0; ci < c->Cg; ci++)
            for (int64_t t = 0; t < KK; t++)
                wr[o * K + t * c->Cg + ci] = w[o * K + ci * KK + t];
    return wr;
}

/* the stem's direct conv (Cg < 4, e.g. RGB): weights [O][KH][KW][Cgp] with zero channels Cg..Cgp-1,
 * so K = (kh, kw, ci) over Cgp channels -- what the region gather needs (Cg % 4 == 0). Cached per
 * layer like w_hwc, under the padded K. */
static const int8_t *w_hwc_pad(const conv_geo *c, const int8_t *w, int64_t Cgp) {
    const int64_t O = c->Og, KK = c->KH * c->KW, K = c->Cg * KK, Kp = Cgp * KK;
    for (int i = 0; g_exec_const_w && i < g_whwc_n; i++)
        if (g_whwc[i].w == w && g_whwc[i].o == O && g_whwc[i].k == Kp)
            return g_whwc[i].wr;
    int8_t *wr = nullptr;
    if (g_exec_const_w && g_whwc_n < RT_BCACHE_MAX && (wr = (int8_t *)malloc((size_t)(O * Kp))))
        g_whwc[g_whwc_n++] = {w, O, Kp, wr};
    if (!wr)
        wr = (int8_t *)scratch(&g_wrs, &g_wrs_cap, (size_t)(O * Kp));
    if (!wr)
        return nullptr;
    memset(wr, 0, (size_t)(O * Kp));
    for (int64_t o = 0; o < O; o++)
        for (int64_t ci = 0; ci < c->Cg; ci++)
            for (int64_t t = 0; t < KK; t++)
                wr[o * Kp + t * Cgp + ci] = w[o * K + ci * KK + t];
    return wr;
}

static void *g_xpad;
static size_t g_xpad_cap;
/* [Cg][H][W] -> [H][W][Cgp], zero channels Cg..Cgp-1 */
static const int8_t *hwc_pad_transpose(const conv_geo *c, const int8_t *x, int64_t Cgp) {
    const int64_t HW = c->H * c->W;
    const uint64_t t0 = rt_ticks();
    int8_t *xt = (int8_t *)scratch(&g_xpad, &g_xpad_cap, (size_t)(HW * Cgp));
    if (!xt)
        return nullptr;
    memset(xt, 0, (size_t)(HW * Cgp));
    if (c->src_hwc) /* nhwc src (one group): pixel q's channels are contiguous */
        for (int64_t q = 0; q < HW; q++)
            memcpy(xt + q * Cgp, x + q * c->Cg, (size_t)c->Cg);
    else
        for (int64_t ci = 0; ci < c->Cg; ci++)
            for (int64_t q = 0; q < HW; q++)
                xt[q * Cgp + ci] = x[ci * HW + q];
    g_phase[RT_PH_HWC] += rt_ticks() - t0;
    return xt;
}

/* im2col with K = (kh, kw, ci): the image is first transposed to [H][W][Cg] (cache tiles), then
 * every tap of a row is one contiguous Cg-byte copy -- instead of 3-byte pieces at stride K, which
 * defeat the A72's write combining (densenet121's im2col ran at ~2 ns per byte) */
static const int8_t *hwc_transpose_(const conv_geo *c, const int8_t *x0) { /* [Cg][H][W] -> [H][W][Cg] per image */
    const int64_t HW = c->H * c->W, Cg = c->Cg;
    int8_t *xt0 = (int8_t *)scratch(&g_xhwc, &g_xhwc_cap, (size_t)(c->nimg * HW * Cg));
    if (!xt0)
        return nullptr;
    for (int64_t im = 0; im < c->nimg; im++) {
        const int8_t *x = x0 + im * c->C * HW;
        int8_t *xt = xt0 + im * HW * Cg;
        for (int64_t p0 = 0; p0 < HW; p0 += 64)
            for (int64_t c0 = 0; c0 < Cg; c0 += 64) {
                const int64_t pn = HW - p0 < 64 ? HW - p0 : 64, cn = Cg - c0 < 64 ? Cg - c0 : 64;
                for (int64_t q = p0; q < p0 + pn; q++)
                    for (int64_t ci = c0; ci < c0 + cn; ci++)
                        xt[q * Cg + ci] = x[ci * HW + q];
            }
    }
    return xt0;
}

static const int8_t *hwc_transpose(const conv_geo *c, const int8_t *x) { /* timed: RT_PH_HWC */
    if (c->src_hwc) /* nhwc src, one group: the images are [H][W][Cg] already */
        return x;
    const uint64_t t0 = rt_ticks();
    const int8_t *r = hwc_transpose_(c, x);
    g_phase[RT_PH_HWC] += rt_ticks() - t0;
    return r;
}

/* rows m_lo .. m_hi of the (kh, kw, ci) im2col from the [H][W][Cg] image */
static void im2col_rows_hwc(const conv_geo *c, const int8_t *xt0, int8_t *col, int64_t m_lo, int64_t m_hi) {
    const int64_t K = c->Cg * c->KH * c->KW, Cg = c->Cg, P = c->Ho * c->Wo;
    for (int64_t m = m_lo; m < m_hi; m++) {
        const int64_t im = m / c->pstride, mm = m - im * c->pstride, oh = mm / c->Wo, ow = mm - oh * c->Wo;
        const int8_t *xt = xt0 + im * c->H * c->W * Cg; /* hwc_transpose's image im */
        if (mm >= P) {                                  // an image's padding row (direct conv's 32-row alignment)
            memset(col + m * K, 0, (size_t)K);
            continue;
        }
        {
            int8_t *row = col + m * K;
            for (int64_t kh = 0; kh < c->KH; kh++) {
                const int64_t ih = oh * c->s[0] - c->pl[0] + kh * c->d[0];
                for (int64_t kw = 0; kw < c->KW; kw++) {
                    const int64_t iw = ow * c->s[1] - c->pl[1] + kw * c->d[1];
                    int8_t *dst = row + (kh * c->KW + kw) * Cg;
                    if (ih >= 0 && ih < c->H && iw >= 0 && iw < c->W)
                        memcpy(dst, xt + (ih * c->W + iw) * Cg, (size_t)Cg);
                    else
                        memset(dst, 0, (size_t)Cg);
                }
            }
        }
    }
}

static int im2col_pack_hwc(const conv_geo *c, const int8_t *x, int8_t *col) {
    const int8_t *xt = hwc_transpose(c, x);
    if (!xt)
        return AIEK_ERR_ALLOC;
    im2col_rows_hwc(c, xt, col, 0, conv_m(c));
    return AIEK_OK;
}

/* 8x8 byte transpose: 8 rows of 8 bytes at src (row pitch sp) -> 8 rows at dst (pitch dp), in three
 * masked swaps on 64-bit words -- 16 memory ops instead of 64 byte loads / stores */
static inline void rt_tr8x8(const int8_t *src, int64_t sp, int8_t *dst, int64_t dp) {
    uint64_t r[8];
    for (int i = 0; i < 8; i++)
        memcpy(&r[i], src + i * sp, 8);
    for (int s = 4, k = 0; s >= 1; s >>= 1, k++) { // swap s x s byte sub-blocks
        static const uint64_t m[3] = {0x00000000FFFFFFFFull, 0x0000FFFF0000FFFFull, 0x00FF00FF00FF00FFull};
        for (int i = 0; i < 8; i++)
            if (!(i & s)) {
                const uint64_t a = r[i], b = r[i + s];
                r[i] = (a & m[k]) | ((b & m[k]) << (8 * s));
                r[i + s] = ((a >> (8 * s)) & m[k]) | (b & ~m[k]);
            }
    }
    for (int i = 0; i < 8; i++)
        memcpy(dst + i * dp, &r[i], 8);
}

/* rows m_lo .. m_hi of a 1x1 unpadded conv's im2col: a sub-sampled [C][HW] -> [HW][C] transpose */
static void im2col_rows_1x1(const conv_geo *c, const int8_t *x, int8_t *col, int64_t m_lo, int64_t m_hi) {
    const int64_t K = c->Cg, HW = c->H * c->W;
    for (int64_t m0 = m_lo; m0 < m_hi; m0 += 64)
        for (int64_t c0 = 0; c0 < c->Cg; c0 += 64) {
            const int64_t mn = m_hi - m0 < 64 ? m_hi - m0 : 64, cn = c->Cg - c0 < 64 ? c->Cg - c0 : 64;
            for (int64_t m = m0; m < m0 + mn; m++) {
                const int64_t im = m / c->pstride, mm = m - im * c->pstride, oh = mm / c->Wo, ow = mm - oh * c->Wo;
                if (c->s[0] == 1 && c->s[1] == 1 && m + 8 <= m0 + mn && mm + 8 <= c->Ho * c->Wo) {
                    // stride 1: pixel mm is input pixel mm, 8 rows are 8 contiguous bytes of each plane --
                    // 8 channels x 8 pixels per rt_tr8x8, then the channels left over byte by byte
                    const int8_t *in = x + im * c->C * HW + c0 * HW + mm;
                    int64_t ci = 0;
                    for (; ci + 8 <= cn; ci += 8)
                        rt_tr8x8(in + ci * HW, HW, col + m * K + c0 + ci, K);
                    for (; ci < cn; ci++)
                        for (int64_t q = 0; q < 8; q++)
                            col[(m + q) * K + c0 + ci] = in[ci * HW + q];
                    m += 7;
                    continue;
                }
                if (mm >= c->Ho * c->Wo) { // padding row
                    memset(col + m * K + c0, 0, (size_t)cn);
                    continue;
                }
                const int8_t *in = x + im * c->C * HW + c0 * HW + oh * c->s[0] * c->W + ow * c->s[1];
                int8_t *dst = col + m * K + c0;
                for (int64_t ci = 0; ci < cn; ci++)
                    dst[ci] = in[ci * HW];
            }
        }
}

/* the persistent build's split launch (gemm_uk.cc, AIEHLC_ASYNC_LAUNCH): absent on x86 and in
 * non-persistent builds, where the same order runs synchronously */
extern "C" int aiednn_gemm_async_start(const aiek_tensor *args, int num_args, int which) __attribute__((weak));
extern "C" int aiednn_gemm_async_finish(void) __attribute__((weak));

/* one AIE launch; *overlap* (may be NULL) runs while it is in flight */
// ── direct conv: A = the raw input region per mesh row (aie_modes_conv.h) ──
/* region row pitch in pixels: the columns ow * s + kw * d can reach (left / right padding included) */
static inline int64_t conv_wp(const conv_geo *c) { return (c->Wo - 1) * c->s[1] + (c->KW - 1) * c->d[1] + 1; }

/* input rows mesh row's pixels p0 .. p0 + 31 (clipped to M) need for K columns k0 .. k0 + BK - 1 */
static int64_t conv_region_rows(const conv_geo *c, int64_t p0, int64_t k0, int64_t BK, int64_t *ih_lo) {
    const int64_t M = c->Ho * c->Wo, K = c->Cg * c->KH * c->KW;
    *ih_lo = 0;
    if (p0 >= M)
        return 0;
    const int64_t p1 = p0 + 31 < M ? p0 + 31 : M - 1, k1 = (k0 + BK < K ? k0 + BK : K) - 1;
    const int64_t kh_a = (k0 / c->Cg) / c->KW, kh_b = (k1 / c->Cg) / c->KW;
    *ih_lo = (p0 / c->Wo) * c->s[0] - c->pl[0] + kh_a * c->d[0];
    return (p1 / c->Wo) * c->s[0] - c->pl[0] + kh_b * c->d[0] - *ih_lo + 1;
}

/* every (mesh row, K block) region of the conv fits a mesh row's A slice */
static int conv_region_fits(const conv_geo *c, int64_t BM, int64_t BK) {
    const int64_t M = c->Ho * c->Wo, K = c->Cg * c->KH * c->KW, rb = conv_wp(c) * c->Cg;
    if (c->Cg % 4 || BM % 32 || 32 * BK != CONV_REGION_BYTES)
        return 0;
    for (int64_t p0 = 0; p0 < M; p0 += 32)
        for (int64_t k0 = 0; k0 < K; k0 += BK) {
            int64_t lo;
            if (conv_region_rows(c, p0, k0, BK, &lo) * rb > CONV_REGION_DATA) // the geometry takes the tail
                return 0;
        }
    return 1;
}

/* a direct-conv launch's A block (pixels m0 .. m0 + BM - 1, K columns k0 ..): mesh row r's region
 * = whole input rows of the [H][W][Cg] image, zero where the conv pads, and its geometry */
static void conv_region_block(const conv_geo *c, const int8_t *xt0, int64_t m0, int64_t k0, int64_t BM, int64_t BK,
                              int8_t *a, int32_t *geo) {
    const int64_t wp = conv_wp(c), Cg = c->Cg, pl = c->pl[1];
    const int64_t lo = pl < wp ? pl : wp, hi = pl + c->W < wp ? pl + c->W : wp; // columns with data
    for (int64_t r = 0; r < BM / 32; r++) {
        // a batch: this mesh row's 32 pixels are rows p0 .. p0+31 of image im (pstride: a multiple of 32)
        const int64_t gm = m0 + 32 * r, im = gm / c->pstride, past = im >= c->nimg;
        const int64_t p0 = past ? c->Ho * c->Wo : gm - im * c->pstride;
        const int8_t *xt = xt0 + (past ? 0 : im) * c->H * c->W * Cg;
        int64_t ih_lo;
        const int64_t n = conv_region_rows(c, p0, k0, BK, &ih_lo);
        int8_t *reg = a + r * 32 * BK;
        for (int64_t i = 0; i < n; i++) {
            int8_t *row = reg + i * wp * Cg;
            const int64_t ih = ih_lo + i;
            if (ih < 0 || ih >= c->H || hi <= lo) {
                memset(row, 0, (size_t)(wp * Cg));
                continue;
            }
            memset(row, 0, (size_t)(lo * Cg));
            memcpy(row + lo * Cg, xt + (ih * c->W + (lo - pl)) * Cg, (size_t)((hi - lo) * Cg));
            memset(row + hi * Cg, 0, (size_t)((wp - hi) * Cg));
        }
        int32_t *g = geo + 16 * r;
        memset(g, 0, 16 * sizeof(int32_t));
        g[CONV_GEOM_CG] = (int32_t)Cg, g[CONV_GEOM_KH] = (int32_t)c->KH, g[CONV_GEOM_KW] = (int32_t)c->KW;
        g[CONV_GEOM_S0] = (int32_t)c->s[0], g[CONV_GEOM_S1] = (int32_t)c->s[1];
        g[CONV_GEOM_D0] = (int32_t)c->d[0], g[CONV_GEOM_D1] = (int32_t)c->d[1];
        g[CONV_GEOM_PL0] = (int32_t)c->pl[0], g[CONV_GEOM_WP] = (int32_t)wp, g[CONV_GEOM_WO] = (int32_t)c->Wo;
        g[CONV_GEOM_M] = (int32_t)(c->Ho * c->Wo), g[CONV_GEOM_M0] = (int32_t)p0;
        g[CONV_GEOM_K0] = (int32_t)k0, g[CONV_GEOM_IHLO] = (int32_t)ih_lo;
        g[CONV_GEOM_KTOT] = (int32_t)(Cg * c->KH * c->KW);
        g[CONV_GEOM_MAGIC] = CONV_MAGIC;
    }
}

static void *g_cgeo;
static size_t g_cgeo_cap;

static int launch_overlapped(const aiednn_impl_t *g, aiek_tensor *t, int nt, int which, void (*overlap)(void *),
                             void *ctx) {
    if (aiednn_gemm_async_start && aiednn_gemm_async_finish) {
        /* the async start bypasses the entry: ping it (no arguments: launches nothing) so a library
         * build loads a megakernel holding the entry's modes first (aiednn_megakernel_need thunk) */
        g->entry(nullptr, 0);
        const int rc = aiednn_gemm_async_start(t, nt, which);
        if (rc != AIEK_OK)
            return rc;
        if (overlap)
            overlap(ctx);
        return aiednn_gemm_async_finish();
    }
    const int rc = g->entry(t, nt);
    if (rc == AIEK_OK && overlap)
        overlap(ctx);
    return rc;
}

/* rows of a [M][Og] GEMM result per epilogue tile: the channel-major walk (one output plane
 * at a time, stride Og) then stays inside ~16 KB, instead of touching a new cache line per
 * element across the whole result (the A72's 32 KB L1 thrashed: resnet18 epilogue 150 ms) */
static inline int64_t rt_epi_rows_per_tile(int64_t og, int64_t elem_bytes) {
    const int64_t r = 16384 / (og * elem_bytes);
    return r < 16 ? 16 : r;
}

/* host work done while a launch is in flight: im2col rows for the next M block, the int8 output
 * of the previous one scattered to the dst channel planes */
typedef struct {
    const conv_geo *c;
    int mode; /* 1: (kh, kw, ci) im2col from the [H][W][Cg] image, 2: 1x1 transpose */
    const int8_t *xsrc;
    int8_t *col;
    int64_t pack_lo, pack_hi;
    const int8_t *o8; /* [M][Og] */
    int8_t *dst;      /* this (image, group)'s first dst channel plane */
    int64_t sc_lo, sc_hi;
    /* int32 path: rows sc_lo .. sc_hi of o32 [M][Og] go through the host epilogue instead */
    const rt_epi *E;
    const int32_t *o32;
    int64_t di0, ch0; /* dst element of (channel ch0, pixel 0) of this (image, group) */
    int64_t step;     /* rows per call (spread over a block's launches); <= 0: everything left */
} rt_pipe_work;

static void pipe_work(void *v) {
    rt_pipe_work *w = (rt_pipe_work *)v;
    const conv_geo *c = w->c;
    if (w->pack_hi > w->pack_lo) {
        const uint64_t t0 = rt_ticks();
        const int64_t hi = w->step > 0 && w->pack_lo + w->step < w->pack_hi ? w->pack_lo + w->step : w->pack_hi;
        if (w->mode == 1 || w->mode == 3 || w->mode == 4)
            im2col_rows_hwc(c, w->xsrc, w->col, w->pack_lo, hi);
        else
            im2col_rows_1x1(c, w->xsrc, w->col, w->pack_lo, hi);
        w->pack_lo = hi;
        g_phase[RT_PH_IM2COL] += rt_ticks() - t0;
    }
    if (w->sc_hi > w->sc_lo) {
        const uint64_t t0 = rt_ticks();
        const int64_t P = c->Ho * c->Wo, Og = c->Og; // P: one image's plane (image im's rows: + im * O * P)
        const int64_t hi = w->step > 0 && w->sc_lo + w->step < w->sc_hi ? w->sc_lo + w->step : w->sc_hi;
        const int64_t mb = rt_epi_rows_per_tile(Og, w->o32 ? 4 : 1);
        for (int64_t m0 = w->sc_lo; m0 < hi;) { // [M][Og] -> channel planes, cache tiles
            const int64_t im = m0 / c->pstride, p0 = m0 - im * c->pstride, iend = im * c->pstride + P;
            if (p0 >= P) { // the image's padding rows: nothing to write
                m0 = (im + 1) * c->pstride;
                continue;
            }
            int64_t mn = hi - m0 < mb ? hi - m0 : mb;
            if (m0 + mn > iend) // a tile never crosses into the padding / the next image's planes
                mn = iend - m0;
            if (c->dst_hwc) { // nhwc dst (one group): the [M][Og] rows are dst rows
                const uint64_t tq = rt_ticks();
                if (!w->o32 && w->o8 == w->dst) { // the result was written in place (o8_of): only a fused tail
                    rt_qfuse_run(w->dst + im * c->O * P + p0 * c->O, mn * c->O);
                    g_qfuse.t += rt_ticks() - tq;
                    m0 += mn;
                    continue;
                }
                for (int64_t m = 0; m < mn; m++) {
                    const int64_t di = w->di0 + im * c->O * P + (p0 + m) * c->O;
                    if (w->o32)
                        rt_epi_row(w->E, di, w->ch0, 1, w->o32 + (m0 + m) * Og, 1, Og);
                    else
                        memcpy(w->dst + im * c->O * P + (p0 + m) * c->O, w->o8 + (m0 + m) * Og, (size_t)Og);
                }
                const uint64_t tq1 = rt_ticks();
                if (w->o32) /* the host epilogue wrote dst through E: its own base */
                    rt_qfuse_run((int8_t *)w->E->dst + w->di0 + im * c->O * P + p0 * c->O, mn * c->O);
                else
                    rt_qfuse_run(w->dst + im * c->O * P + p0 * c->O, mn * c->O);
                g_qfuse.t += rt_ticks() - tq1;
                m0 += mn;
                continue;
            }
            for (int64_t og = 0; og < Og; og++) {
                if (w->o32) { // host epilogue
                    rt_epi_row(w->E, w->di0 + im * c->O * P + og * P + p0, w->ch0 + og, 0, w->o32 + m0 * Og + og, Og,
                               mn);
                    continue;
                }
                if (og + 8 <= Og) { // 8 channels x 8 pixels at a time (rt_tr8x8), the rest byte by byte
                    int64_t m = 0;
                    for (; m + 8 <= mn; m += 8)
                        rt_tr8x8(w->o8 + (m0 + m) * Og + og, Og, w->dst + im * c->O * P + og * P + p0 + m, P);
                    for (int64_t q = 0; q < 8; q++)
                        for (int64_t r = m; r < mn; r++)
                            w->dst[im * c->O * P + (og + q) * P + p0 + r] = w->o8[(m0 + r) * Og + og + q];
                    og += 7;
                    continue;
                }
                int8_t *d8 = w->dst + im * c->O * P + og * P + p0;
                const int8_t *s8 = w->o8 + m0 * Og + og;
                for (int64_t m = 0; m < mn; m++)
                    d8[m] = s8[m * Og];
            }
            if (g_qfuse.on && g_qfuse.claim) { /* the tile's channel segments, while still in cache */
                const uint64_t tq = rt_ticks();
                for (int64_t og = 0; og < Og; og++)
                    rt_qfuse_run(w->o32 ? (int8_t *)w->E->dst + w->di0 + im * c->O * P + og * P + p0
                                        : w->dst + im * c->O * P + og * P + p0,
                                 mn);
                g_qfuse.t += rt_ticks() - tq;
            }
            m0 += mn;
        }
        w->sc_lo = hi;
        g_phase[RT_PH_EPI] += rt_ticks() - t0;
    }
}

/* the block's host work in slices, one per launch of the block; whatever is left runs after */
static inline int64_t pipe_step(int64_t rows, int64_t launches) {
    const int64_t s = (rows + launches - 1) / (launches > 0 ? launches : 1);
    return s < 8 ? 8 : s;
}

static void *g_qps;
static size_t g_qps_cap;

/* conv with the integer epilogue on the core, pipelined over M blocks: while the first launch of
 * block j runs, the host packs block j+1's im2col rows and scatters block j-1's int8 output (the
 * persistent build's split launch; elsewhere the same order, synchronously). Same launches and
 * results as im2col + blocked_gemm_q. */
static int run_ntile(const rt_impl *self, const rt_epi *E, const conv_geo *c, int mode, const int8_t *img,
                     const int8_t *wv, int64_t ch0, int8_t *col, int which, int8_t *o8, int8_t *dst, int32_t *o32,
                     int64_t di0, int creg); /* aiednn_rt_ntile.h */
extern "C" void aiednn_gemm_ntile_conv(int on) __attribute__((weak));

static int rt_env_on(const char *name); /* aiednn_rt_ntile.h */
/* a direct-conv layer may take the n-tile launch too (its tiles are input regions) */
static int ntile_conv_ok(void) { return aiednn_gemm_ntile_conv && !rt_env_on("AIEDNN_NO_NTILE_CONV"); }

/* c with each image's rows padded to a multiple of 32 when a batch goes through direct conv */
static const conv_geo *creg_geo(const conv_geo *c, conv_geo *buf) {
    if (c->nimg <= 1)
        return c;
    *buf = *c;
    buf->pstride = (c->Ho * c->Wo + 31) / 32 * 32;
    return buf;
}
static int ntile_mblocks(int64_t nkb, int has_res);
static int rt_env_on(const char *name);        /* aiednn_rt_ntile.h */
static int mmt_usable(int64_t K, int has_res); /* aiednn_rt_mmt.h */
static int run_mmt(const rt_impl *self, const rt_epi *E, int64_t M, int64_t N, int64_t K, const int8_t *col,
                   const int8_t *wv, int64_t ch0, int8_t *o8);

static int run_q_pipe(const rt_impl *self, const rt_epi *E, const conv_geo *c, int mode, const int8_t *img,
                      const int8_t *wv, int64_t ch0, int8_t *col, int8_t *o8, int8_t *dst, const int8_t *res) {
    const aiednn_impl_t *q = g_qblk, *kc = g_kcblk;
    { /* short K: MMT (aiednn_rt_mmt.h); else many tiles per launch (aiednn_rt_ntile.h) when the
       * persistent build has it -- a direct-conv layer too when the build has its n-tile variant */
        const int64_t BK_ = q->args[0].dims[1], nkb_ = (c->Cg * c->KH * c->KW + BK_ - 1) / BK_;
        const int creg_ = (mode == 1 || mode == 3) && g_cqblk && (nkb_ == 1 || g_ckcblk) &&
                          conv_region_fits(c, q->args[0].dims[0], BK_);
        const int64_t M_ = conv_m(c), N_ = c->Og, K_ = c->Cg * c->KH * c->KW;
        /* MMT is 4x a GEMM / direct-conv tile; not mode 3 (the padded stem: its input is direct conv's
         * layout, and col is sized for the unpadded K) */
        if ((mode == 1 || mode == 2 || mode == 4) && mmt_usable(K_, res != nullptr)) {
            const int8_t *xs = mode == 1 ? hwc_transpose(c, img) : img;
            if (!xs)
                return AIEK_ERR_ALLOC;
            rt_pipe_work w = {c, mode, xs, col, 0, M_, o8, dst, 0, 0, nullptr, nullptr, 0, 0, 0};
            pipe_work(&w); /* every im2col row */
            const int rc = run_mmt(self, E, M_, N_, K_, col, wv, ch0, o8);
            if (rc != AIEK_ERR_PARAM) {
                if (rc != AIEK_OK)
                    return rc;
                w.sc_lo = 0, w.sc_hi = M_; /* [M][N] -> the dst channel planes */
                pipe_work(&w);
                return AIEK_OK;
            }
        }
        if ((nkb_ == 1 || kc) && ntile_mblocks(nkb_, res != nullptr) && (!creg_ || ntile_conv_ok())) {
            conv_geo cg;
            return run_ntile(self, E, creg_ ? creg_geo(c, &cg) : c, mode, img, wv, ch0, col, 1, o8, dst, nullptr, 0,
                             creg_);
        }
    }
    conv_geo cb32 = *c; // a batch through direct conv: each image's rows padded to a multiple of 32
    if (c->nimg > 1 && (mode == 1 || mode == 3) && g_cqblk &&
        ((c->Cg * c->KH * c->KW + q->args[0].dims[1] - 1) / q->args[0].dims[1] == 1 || g_ckcblk) &&
        conv_region_fits(c, q->args[0].dims[0], q->args[0].dims[1])) {
        cb32.pstride = (c->Ho * c->Wo + 31) / 32 * 32;
        c = &cb32;
    }
    const int64_t M = conv_m(c), N = c->Og, K = c->Cg * c->KH * c->KW, P = c->Ho * c->Wo;
    const int64_t BM = q->args[0].dims[0], BK = q->args[0].dims[1], BN = q->args[1].dims[0], NP = q->args[2].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nnb = (N + BN - 1) / BN, nmb = (M + BM - 1) / BM;
    if (nkb > 1 && !kc)
        return AIEK_ERR_PARAM;
    int8_t *bp = b_blocks(wv, N, K, q, g_exec_const_w ? (const void *)wv : nullptr);
    int8_t *cb = (int8_t *)scratch(&g_qc, &g_qc_cap, (size_t)(BM * BN));
    int32_t *c32 = (int32_t *)scratch(&g_cb, &g_cb_cap, (size_t)(BM * BN) * sizeof(int32_t));
    int32_t *qps = (int32_t *)scratch(&g_qps, &g_qps_cap, (size_t)(nnb * NP) * sizeof(int32_t));
    int8_t *rb = (int8_t *)scratch(&g_qr, &g_qr_cap, (size_t)(BM * BN));
    const int8_t *xsrc = mode == 1 ? hwc_transpose(c, img) : img; // mode 3: img is [H][W][Cg] already
    if (!bp || !cb || !c32 || !qps || !rb || !xsrc)
        return AIEK_ERR_ALLOC;
    // direct conv: A blocks are input regions the cores gather from (no host im2col)
    const int creg = (mode == 1 || mode == 3) && g_cqblk && (nkb == 1 || g_ckcblk) && conv_region_fits(c, BM, BK);
    const int64_t ng = 16 * (BM / 32);
    int32_t *gall =
        creg ? (int32_t *)scratch(&g_cgeo, &g_cgeo_cap, (size_t)(nmb * nkb * ng) * sizeof(int32_t)) : nullptr;
    if (creg && !gall)
        return AIEK_ERR_ALLOC;
    memset(qps, 0, (size_t)(nnb * NP) * sizeof(int32_t));
    for (int64_t nb = 0; nb < nnb; nb++) { // {zp, lo, hi, flags, B, M, sh, has_res, rzp, R} per N block
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
        if (res) {
            qp[4 + 4 * BN] = 1,
                       qp[5 + 4 * BN] = E->fx_rdt == aiednn_u8 ? E->fx_rzp - 128 : E->fx_rzp; /* u8: shifted, see rb */
            int64_t *rr = (int64_t *)(qp + 6 + 4 * BN);
            for (int64_t j = 0; j < BN; j++)
                rr[j] = j < nn ? E->fx_rm[ch0 + n0 + j] : 0;
        }
    }
    memset(rb, 0, (size_t)(BM * BN));
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, BK}, b_shape[2] = {BN, BK}, p_shape[1] = {NP}, c_shape[2] = {BM, BN},
                  f_shape[1] = {1};
    aiek_tensor t[5] = {{nullptr, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK)},
                        {nullptr, 2, AIEK_INT, 8, 1, b_shape, (uint64_t)(BN * BK)},
                        {nullptr, 1, AIEK_INT, 32, 1, p_shape, (uint64_t)NP * 4},
                        {rb, 2, AIEK_INT, 8, 1, c_shape, (uint64_t)(BM * BN)},
                        {cb, 2, AIEK_INT, 8, 1, c_shape, (uint64_t)(BM * BN)}};
    aiek_tensor tk[4] = {t[0],
                         t[1],
                         {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                         {c32, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
    const int64_t g_shape[1] = {ng};
    aiek_tensor cgt = {nullptr, 1, AIEK_INT, 32, 1, g_shape, (uint64_t)ng * 4};
    aiek_tensor ct[6] = {t[0], t[1], t[2], t[3], cgt, t[4]}, ctk[5] = {tk[0], tk[1], tk[2], cgt, tk[3]};
    rt_pipe_work w = {c, mode, xsrc, col, 0, creg ? 0 : (BM < M ? BM : M), o8, dst, 0, 0, nullptr, nullptr, 0, 0, 0};
    int8_t *apall = (int8_t *)scratch(&g_apall, &g_apall_cap, (size_t)(nmb * nkb * BM * BK));
    if (!apall)
        return AIEK_ERR_ALLOC;
    pipe_work(&w); // block 0's rows (step 0: all); every later block is packed during the previous one
    // n blocks outer, m blocks inner: consecutive launches share a B block (mem tile cache); the
    // first n pass packs the next block's rows, the last scatters the previous block's output
    for (int64_t nb = 0; nb < nnb; nb++)
        for (int64_t mb = 0; mb < nmb; mb++) {
            const int64_t m0 = mb * BM, mn = M - m0 < BM ? M - m0 : BM;
            int8_t *ap = apall + mb * nkb * BM * BK;
            if (nb == 0 && creg) {
                const uint64_t ti = rt_ticks();
                for (int64_t kb = 0; kb < nkb; kb++)
                    conv_region_block(c, xsrc, m0, kb * BK, BM, BK, ap + kb * BM * BK, gall + (mb * nkb + kb) * ng);
                g_phase[RT_PH_REGION] += rt_ticks() - ti;
            } else if (nb == 0) {
                const int8_t *pk = a_blocks(col, m0, mn, K, BM, BK);
                if (!pk)
                    return AIEK_ERR_ALLOC;
                memcpy(ap, pk, (size_t)(nkb * BM * BK));
                w.pack_lo = m0 + BM, w.pack_hi = m0 + 2 * BM < M ? m0 + 2 * BM : M;
            }
            if (nb == nnb - 1)
                w.sc_lo = mb > 0 ? m0 - BM : 0, w.sc_hi = mb > 0 ? m0 : 0;
            w.step = pipe_step(BM, nkb);
            int first = 1;
            {
                const int64_t n0 = nb * BN, nn = N - n0 < BN ? N - n0 : BN;
                int32_t *qp = qps + nb * NP;
                t[2].data = qp;
                ct[2].data = qp; // the direct-conv tensors are copies: their qparams pointer too
                if (res) {
                    const int8_t rx = E->fx_rdt == aiednn_u8 ? (int8_t)0x80 : 0;
                    for (int64_t j = 0; j < nn; j++)
                        for (int64_t m = 0; m < mn; m++) { // residual planes as dst: image im at + im * O * P
                            const int64_t im = (m0 + m) / c->pstride, mm = m0 + m - im * c->pstride;
                            /* a u8 residual enters as s8 r - 128 (the byte ^ 0x80) with zr - 128 in qp: (r - zr)
                             * unchanged */
                            rb[m * BN + j] = (int8_t)(rx ^ (mm >= P      ? 0
                                                            : c->res_hwc ? res[im * c->O * P + mm * c->O + n0 + j]
                                                                         : res[im * c->O * P + (n0 + j) * P + mm]));
                        }
                }
                for (int64_t kb = 0; kb < nkb; kb++) {
                    const int last = kb == nkb - 1;
                    aiek_tensor *tt = creg ? (last ? ct : ctk) : (last ? t : tk);
                    tt[0].data = ap + kb * BM * BK;
                    tt[1].data = bp + (nb * nkb + kb) * BN * BK;
                    if (creg)
                        tt[last ? 4 : 3].data = gall + (mb * nkb + kb) * ng;
                    flags = (kb > 0 ? 1 : 0) | 2;
                    qp[3] = kb > 0 ? 1 : 0;
                    rt_wkey(g_exec_const_w ? (const void *)wv : nullptr, nb, kb, N, K);
                    const uint64_t te = rt_ticks();
                    const aiednn_impl_t *ge = creg ? (last ? g_cqblk : g_ckcblk) : (last ? q : kc);
                    const int rc =
                        launch_overlapped(ge, tt, (last ? 5 : 4) + creg, (last ? 1 : 0) + 2 * creg, pipe_work, &w);
                    ((rt_impl *)self)->t_entry += rt_ticks() - te;
                    ((rt_impl *)self)->launches++;
                    first = 0;
                    if (rc != AIEK_OK)
                        return rc;
                }
                for (int64_t r = 0; r < mn; r++)
                    memcpy(o8 + (m0 + r) * N + n0, cb + r * BN, (size_t)nn);
            }
            w.step = 0; // what the launches did not cover
            pipe_work(&w);
            (void)first;
        }
    w.sc_lo = (nmb - 1) * BM, w.sc_hi = M, w.step = 0;
    pipe_work(&w);
    return AIEK_OK;
}

/* conv with the host epilogue (f32 / int32 dst, a u8 residual, ...), pipelined like run_q_pipe:
 * K-chained int32 launches per M block; while block j's first launch runs, the host packs block
 * j+1's im2col rows and runs block j-1's epilogue. Needs the chained block (K > BK). */
static int run_i32_pipe(const rt_impl *self, const rt_epi *E, const conv_geo *c, int mode, const int8_t *img,
                        const int8_t *wv, int8_t *col, int32_t *o32, int64_t di0, int64_t ch0) {
    const aiednn_impl_t *kc = g_kcblk;
    { /* n tiles per launch (aiednn_rt_ntile.h); a direct-conv layer when the build has its variant */
        const int64_t BK_ = kc->args[0].dims[1];
        const int creg_ = (mode == 1 || mode == 3) && g_ckcblk && conv_region_fits(c, kc->args[0].dims[0], BK_);
        if (ntile_mblocks((c->Cg * c->KH * c->KW + BK_ - 1) / BK_, 0) && (!creg_ || ntile_conv_ok())) {
            conv_geo cg;
            return run_ntile(self, E, creg_ ? creg_geo(c, &cg) : c, mode, img, wv, ch0, col, 0, nullptr, nullptr, o32,
                             di0, creg_);
        }
    }
    conv_geo cb32 = *c; // as run_q_pipe: a batch through direct conv pads each image's rows to 32
    if (c->nimg > 1 && (mode == 1 || mode == 3) && g_ckcblk &&
        conv_region_fits(c, kc->args[0].dims[0], kc->args[0].dims[1])) {
        cb32.pstride = (c->Ho * c->Wo + 31) / 32 * 32;
        c = &cb32;
    }
    const int64_t M = conv_m(c), N = c->Og, K = c->Cg * c->KH * c->KW;
    const int64_t BM = kc->args[0].dims[0], BK = kc->args[0].dims[1], BN = kc->args[1].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nnb = (N + BN - 1) / BN, nmb = (M + BM - 1) / BM;
    int8_t *bp = b_blocks(wv, N, K, kc, g_exec_const_w ? (const void *)wv : nullptr);
    int32_t *cb = (int32_t *)scratch(&g_cb, &g_cb_cap, (size_t)(BM * BN) * sizeof(int32_t));
    const int8_t *xsrc = mode == 1 ? hwc_transpose(c, img) : img; // mode 3: img is [H][W][Cg] already
    if (!bp || !cb || !xsrc)
        return AIEK_ERR_ALLOC;
    const int creg = (mode == 1 || mode == 3) && g_ckcblk && conv_region_fits(c, BM, BK); // as run_q_pipe
    const int64_t ng = 16 * (BM / 32);
    int32_t *gall =
        creg ? (int32_t *)scratch(&g_cgeo, &g_cgeo_cap, (size_t)(nmb * nkb * ng) * sizeof(int32_t)) : nullptr;
    if (creg && !gall)
        return AIEK_ERR_ALLOC;
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, BK}, b_shape[2] = {BN, BK}, f_shape[1] = {1}, c_shape[2] = {BM, BN};
    aiek_tensor t[4] = {{nullptr, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK)},
                        {nullptr, 2, AIEK_INT, 8, 1, b_shape, (uint64_t)(BN * BK)},
                        {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                        {cb, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
    const int64_t g_shape[1] = {ng};
    aiek_tensor ct[5] = {t[0], t[1], t[2], {nullptr, 1, AIEK_INT, 32, 1, g_shape, (uint64_t)ng * 4}, t[3]};
    rt_pipe_work w = {c, mode, xsrc, col, 0, creg ? 0 : (BM < M ? BM : M), nullptr, nullptr, 0, 0, E, o32, di0, ch0, 0};
    int8_t *apall = (int8_t *)scratch(&g_apall, &g_apall_cap, (size_t)(nmb * nkb * BM * BK));
    if (!apall)
        return AIEK_ERR_ALLOC;
    pipe_work(&w);
    for (int64_t nb = 0; nb < nnb; nb++) // n outer, m inner: as run_q_pipe
        for (int64_t mb = 0; mb < nmb; mb++) {
            const int64_t m0 = mb * BM, mn = M - m0 < BM ? M - m0 : BM;
            int8_t *ap = apall + mb * nkb * BM * BK;
            if (nb == 0 && creg) {
                const uint64_t ti = rt_ticks();
                for (int64_t kb = 0; kb < nkb; kb++)
                    conv_region_block(c, xsrc, m0, kb * BK, BM, BK, ap + kb * BM * BK, gall + (mb * nkb + kb) * ng);
                g_phase[RT_PH_REGION] += rt_ticks() - ti;
            } else if (nb == 0) {
                const int8_t *pk = a_blocks(col, m0, mn, K, BM, BK);
                if (!pk)
                    return AIEK_ERR_ALLOC;
                memcpy(ap, pk, (size_t)(nkb * BM * BK));
                w.pack_lo = m0 + BM, w.pack_hi = m0 + 2 * BM < M ? m0 + 2 * BM : M;
            }
            if (nb == nnb - 1)
                w.sc_lo = mb > 0 ? m0 - BM : 0, w.sc_hi = mb > 0 ? m0 : 0;
            w.step = pipe_step(BM, nkb);
            int first = 1;
            {
                const int64_t n0 = nb * BN, nn = N - n0 < BN ? N - n0 : BN;
                for (int64_t kb = 0; kb < nkb; kb++) {
                    flags = (kb > 0 ? 1 : 0) | (kb < nkb - 1 ? 2 : 0) | (aiednn_gemm_stage_c ? 4 : 0);
                    aiek_tensor *tt = creg ? ct : t;
                    tt[0].data = ap + kb * BM * BK;
                    tt[1].data = bp + (nb * nkb + kb) * BN * BK;
                    if (creg)
                        tt[3].data = gall + (mb * nkb + kb) * ng;
                    rt_wkey(g_exec_const_w ? (const void *)wv : nullptr, nb, kb, N, K);
                    const uint64_t te = rt_ticks();
                    const int rc = launch_overlapped(creg ? g_ckcblk : kc, tt, 4 + creg, 2 * creg, pipe_work, &w);
                    ((rt_impl *)self)->t_entry += rt_ticks() - te;
                    ((rt_impl *)self)->launches++;
                    first = 0;
                    if (rc != AIEK_OK)
                        return rc;
                }
                for (int64_t r = 0; r < mn; r++)
                    memcpy(o32 + (m0 + r) * N + n0, (aiednn_gemm_stage_c ? aiednn_gemm_stage_c() : cb) + r * BN,
                           (size_t)nn * sizeof(int32_t));
            }
            w.step = 0;
            pipe_work(&w);
            (void)first;
        }
    w.sc_lo = (nmb - 1) * BM, w.sc_hi = M, w.step = 0;
    pipe_work(&w);
    return AIEK_OK;
}

static void *g_tcol, *g_tbp;
static size_t g_tcol_cap, g_tbp_cap;

/* the K-chained block's launches of the transposed tiling: C^T [Og][M] tiles, Og rows per BM and pixels per BN */
static inline int64_t tpipe_launches(int64_t M, int64_t N, int64_t K, const aiednn_impl_t *kc, int transposed) {
    const int64_t BM = kc->args[0].dims[0], BK = kc->args[0].dims[1], BN = kc->args[1].dims[0];
    const int64_t nkb = (K + BK - 1) / BK;
    return transposed ? (N + BM - 1) / BM * ((M + BN - 1) / BN) * nkb : (M + BM - 1) / BM * ((N + BN - 1) / BN) * nkb;
}

/* small-M convs (resnet18 layer4: 49 pixels x 512 channels): the GEMM with its operands swapped,
 * C^T [Og][M] = W [Og][K] x col [M][K]^T, so the weights fill BM and the pixels BN -- half the launches (and the
 * DMA bytes) of the [M][Og] tiling there. K-chained int32 on the core; the GEMMQ requant is per C column, here a
 * pixel, so the epilogue runs on the host (rt_epi, bit-identical to the core's; ~25 K elements on layer4). A row
 * of C^T is one channel's pixels: contiguous in an NCHW dst. Weight blocks [BM][BK] come from the constant-weight
 * cache like B's (keyed by the weights and the block). */
static int run_t_pipe(const rt_impl *self, const rt_epi *E, const conv_geo *c, int mode, const int8_t *img,
                      const int8_t *wv, int64_t di0, int64_t ch0) {
    const aiednn_impl_t *kc = g_kcblk;
    const int64_t M = c->Ho * c->Wo, N = c->Og, K = c->Cg * c->KH * c->KW, P = M;
    const int64_t BM = kc->args[0].dims[0], BK = kc->args[0].dims[1], BN = kc->args[1].dims[0];
    const int64_t nkb = (K + BK - 1) / BK, nmb = (N + BM - 1) / BM, nnb = (M + BN - 1) / BN;
    const uint64_t ti = rt_ticks();
    int8_t *col = (int8_t *)scratch(&g_tcol, &g_tcol_cap, (size_t)(M * K));
    const int8_t *xsrc = mode == 1 ? hwc_transpose(c, img) : img;
    if (!col || !xsrc)
        return AIEK_ERR_ALLOC;
    if (mode == 1 || mode == 4)
        im2col_rows_hwc(c, xsrc, col, 0, M);
    else
        im2col_rows_1x1(c, xsrc, col, 0, M);
    g_phase[RT_PH_IM2COL] += rt_ticks() - ti;
    /* A' = the weights as [BM][BK] blocks (b_blocks with BM rows: the same block layout), B' = the pixels */
    aiednn_impl_t ablk = *kc;
    ablk.args[1].dims[0] = BM;
    int8_t *ap = b_blocks(wv, N, K, &ablk, g_exec_const_w ? (const void *)wv : nullptr); /* cache or g_bp */
    int8_t *bp = (int8_t *)scratch(&g_tbp, &g_tbp_cap, (size_t)(nnb * nkb * BN * BK));   /* not g_bp: A' may be there */
    int32_t *cb = (int32_t *)scratch(&g_cb, &g_cb_cap, (size_t)(BM * BN) * sizeof(int32_t));
    if (!ap || !bp || !cb)
        return AIEK_ERR_ALLOC;
    const uint64_t tb0 = rt_ticks();
    for (int64_t nb = 0; nb < nnb; nb++) /* the pixels' [BN][BK] blocks, as b_blocks lays them out */
        for (int64_t kb = 0; kb < nkb; kb++) {
            int8_t *dst = bp + (nb * nkb + kb) * BN * BK;
            const int64_t k0 = kb * BK, kn = K - k0 < BK ? K - k0 : BK;
            for (int64_t r = 0; r < BN; r++) {
                const int64_t m = nb * BN + r;
                if (m < M)
                    memcpy(dst + r * BK, col + m * K + k0, (size_t)kn);
                memset(dst + r * BK + (m < M ? kn : 0), 0, (size_t)(m < M ? BK - kn : BK));
            }
        }
    g_phase[RT_PH_APACK] += rt_ticks() - tb0;
    int32_t flags = 0;
    const int64_t a_shape[2] = {BM, BK}, b_shape[2] = {BN, BK}, f_shape[1] = {1}, c_shape[2] = {BM, BN};
    aiek_tensor t[4] = {{nullptr, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(BM * BK)},
                        {nullptr, 2, AIEK_INT, 8, 1, b_shape, (uint64_t)(BN * BK)},
                        {&flags, 1, AIEK_INT, 32, 1, f_shape, 4},
                        {cb, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)(BM * BN * 4)}};
    for (int64_t mb = 0; mb < nmb; mb++) /* m (channel blocks) outer: consecutive launches share an A block */
        for (int64_t nb = 0; nb < nnb; nb++) {
            for (int64_t kb = 0; kb < nkb; kb++) {
                flags = (kb > 0 ? 1 : 0) | (kb < nkb - 1 ? 2 : 0) | (aiednn_gemm_stage_c ? 4 : 0);
                t[0].data = ap + (mb * nkb + kb) * BM * BK;
                t[1].data = bp + (nb * nkb + kb) * BN * BK;
                rt_wkey(nullptr, nb, kb, M, K); /* B' is activations: never the constant-weight path */
                const uint64_t te = rt_ticks();
                const int rc = kc->entry(t, 4);
                ((rt_impl *)self)->t_entry += rt_ticks() - te;
                ((rt_impl *)self)->launches++;
                if (rc != AIEK_OK)
                    return rc;
            }
            const uint64_t te0 = rt_ticks();
            const int32_t *res = aiednn_gemm_stage_c ? aiednn_gemm_stage_c() : cb;
            const int64_t o0 = mb * BM, on = N - o0 < BM ? N - o0 : BM, p0 = nb * BN, pn = M - p0 < BN ? M - p0 : BN;
            for (int64_t r = 0; r < on; r++) { /* row r: channel o0 + r over pixels p0 .. p0 + pn */
                const int64_t o = o0 + r;
                if (c->dst_hwc) {
                    for (int64_t j = 0; j < pn; j++)
                        rt_epi_put(E, di0 + (p0 + j) * c->O + o, ch0 + o, (double)res[r * BN + j]);
                    rt_qfuse_run((int8_t *)E->dst + di0 + p0 * c->O + o, pn, c->O);
                } else {
                    rt_epi_row(E, di0 + o * P + p0, ch0 + o, 0, res + r * BN, 1, pn);
                    rt_qfuse_run((int8_t *)E->dst + di0 + o * P + p0, pn);
                }
            }
            g_phase[RT_PH_EPI] += rt_ticks() - te0;
        }
    return AIEK_OK;
}

/* the q pipes' [M][Og] int8 result: with an nhwc dst (one group, one image per call) that IS the dst image --
 * the launches' result tiles land in place and pipe_work has nothing left to scatter */
static int8_t *o8_of(const conv_geo *c, int32_t *out, void *dst, int64_t n, int64_t M) {
    return c->dst_hwc && c->G == 1 && c->nimg == 1 ? (int8_t *)dst + n * c->O * M : (int8_t *)out;
}

static int run_im2col(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    rt_epi E_; // epilogue resolved once per call (bit-identical to rt_epilogue)
    rt_epi_init(&E_, p, ptr);
    conv_geo c;
    conv_geometry(p, 0, &c);
    rt_prob g;
    const int64_t M = c.Ho * c.Wo, K = c.Cg * c.KH * c.KW;
    gemm_prob(&g, M, c.Og, K, aiednn_s8);
    /* a batch goes through the pipes as ONE call with the images stacked along M (conv_geo.nimg): the
     * small-spatial layers' M blocks fill up and every B block / launch serves the whole batch. Only
     * when each image would take a pipe anyway (not the RGB stem's direct conv, one group). */
    /* K = (kh, kw, ci) from an [H][W][Cg] image: KxK taps over >= 16 channels, and any nhwc src (no transpose
     * to pay); else 1x1 / no padding (mode 2) or the plain im2col */
    /* mode 4: a 1x1 over an nhwc image -- rows are the image's pixels (im2col_rows_hwc), but NOT mode 1: that would
     * make a 1x1 eligible for direct conv (CONV mode, another library ELF than the 1x1's GEMMQ / MMT: mobilenet_v2
     * PT2E's 7x7 expand convs ping-ponged megakernels, +16 ms on the board) */
    const int hwc = (c.KH * c.KW > 1 && c.Cg >= 16) || c.src_hwc;
    const int pipe = !hwc                            ? (c.KH == 1 && c.KW == 1 && c.pl[0] == 0 && c.pl[1] == 0 ? 2 : 0)
                     : c.src_hwc && c.KH * c.KW == 1 ? 4
                                                     : 1;
    /* on-core requant; a u8 residual too (run_q_pipe shifts it to s8 exactly) */
    const int q_ok = pipe && E_.fx && (!E_.fx_r || E_.fx_rdt == aiednn_s8 || E_.fx_rdt == aiednn_u8) && g_qblk &&
                     (K <= g_qblk->args[0].dims[1] || g_kcblk);
    const int i_ok = pipe && g_kcblk && g_kcblk->args[0].dims[0] == prim->bind.blk->args[0].dims[0];
    const int stem = c.Cg < 4 && c.KH * c.KW > 1 && g_ckcblk;
    /* the transposed tiling where it takes fewer launches (one image per call: small-M layers are never batched) */
    /* the transposed tiling where it takes fewer tiles, and the plain one would not batch its tiles into n-tile
     * launches anyway (a fused residual, or no n-tile entry): swapped, every tile is its own launch */
    const int64_t nkb_kc = g_kcblk ? (K + g_kcblk->args[0].dims[1] - 1) / g_kcblk->args[0].dims[1] : 0;
    const int t_ok = pipe && g_kcblk && c.N == 1 && !rt_env_on("AIEDNN_NO_TPIPE") &&
                     !ntile_mblocks(nkb_kc, E_.fx_r != nullptr) &&
                     tpipe_launches(M, c.Og, K, g_kcblk, 1) < tpipe_launches(M, c.Og, K, g_kcblk, 0);
    conv_geo c4 = c;
    c4.Cg = 4;
    const int stem_fit = stem && conv_region_fits(&c4, g_ckcblk->args[0].dims[0], g_ckcblk->args[0].dims[1]);
    /* nhwc operands: native on the stem / pipe paths; any other branch runs on dense copies */
    if (rt_prob_has_fmt(p) && !(stem_fit || q_ok || i_ok || t_ok))
        return rt_fmt_rerun_dense(self, prim, ptr, run_im2col);
    if (c.N > 1 && c.G == 1 && !stem && (q_ok || i_ok) && !rt_env_on("AIEDNN_NO_BATCH_M"))
        c.nimg = c.N;
    if (stem_fit || q_ok || i_ok || t_ok) /* every dst element leaves through pipe_work / run_t_pipe: a fused tail */
        g_qfuse.claim = 1;
    const int64_t Mp = c.nimg > 1 ? (M + 31) / 32 * 32 : M; // a batched direct conv pads each image to 32 rows
    int8_t *col = (int8_t *)scratch(&g_col, &g_col_cap, (size_t)(c.nimg * Mp * K));
    int32_t *out = (int32_t *)scratch(&g_gout, &g_gout_cap, (size_t)(c.nimg * Mp * c.Og) * sizeof(int32_t));
    if (!col || !out)
        return AIEK_ERR_ALLOC;
    const int8_t *x = (const int8_t *)ptr[0];
    const rt_bind b = {nullptr, prim->bind.blk};
    for (int64_t n = 0; n < c.N; n += c.nimg)
        for (int64_t grp = 0; grp < c.G; grp++) {
            const uint64_t ti0 = rt_ticks();
            if (stem) { // the stem (RGB): direct conv over 4 padded channels
                conv_geo cp = c;
                cp.Cg = 4;
                const int qpath = E_.fx && !E_.fx_r && g_cqblk;
                if (conv_region_fits(&cp, g_ckcblk->args[0].dims[0], g_ckcblk->args[0].dims[1])) {
                    const int8_t *wg = (const int8_t *)ptr[1] + grp * c.Og * K;
                    const int8_t *w4 = w_hwc_pad(&c, wg, 4);
                    const int8_t *x4 = hwc_pad_transpose(&c, x + (n * c.C + grp * c.Cg) * c.H * c.W, 4);
                    if (!w4 || !x4)
                        return AIEK_ERR_ALLOC;
                    g_phase[RT_PH_IM2COL] += rt_ticks() - ti0;
                    const int rc =
                        qpath
                            ? run_q_pipe(self, &E_, &cp, 3, x4, w4, grp * c.Og, col, o8_of(&c, out, ptr[2], n, M),
                                         (int8_t *)ptr[2] + (n * c.O + grp * c.Og) * M, nullptr)
                            : run_i32_pipe(self, &E_, &cp, 3, x4, w4, col, out, (n * c.O + grp * c.Og) * M, grp * c.Og);
                    if (rc != AIEK_OK)
                        return rc;
                    g_fx_core += qpath;
                    continue;
                }
            }
            const int8_t *wg = (const int8_t *)ptr[1] + grp * c.Og * K, *wv = hwc ? w_hwc(&c, wg) : wg;
            if (!wv)
                return AIEK_ERR_ALLOC;
            if (t_ok && !(stem_fit && stem)) { // operands swapped: fewer launches for few pixels x many channels
                g_phase[RT_PH_IM2COL] += rt_ticks() - ti0;
                const int rc = run_t_pipe(self, &E_, &c, pipe, x + (n * c.C + grp * c.Cg) * c.H * c.W, wv,
                                          (n * c.O + grp * c.Og) * M, grp * c.Og);
                if (rc != AIEK_OK)
                    return rc;
                continue;
            }
            if (q_ok) { // on-core requant, host work overlapped
                g_phase[RT_PH_IM2COL] += rt_ticks() - ti0;
                const int8_t *res = E_.fx_r ? (const int8_t *)E_.fx_r + (n * c.O + grp * c.Og) * M : nullptr;
                const int rc =
                    run_q_pipe(self, &E_, &c, pipe, x + (n * c.C + grp * c.Cg) * c.H * c.W, wv, grp * c.Og, col,
                               o8_of(&c, out, ptr[2], n, M), (int8_t *)ptr[2] + (n * c.O + grp * c.Og) * M, res);
                if (rc != AIEK_OK)
                    return rc;
                g_fx_core++;
                continue;
            }
            if (i_ok) {
                g_phase[RT_PH_IM2COL] += rt_ticks() - ti0; // host epilogue, overlapped with the launches
                const int rc = run_i32_pipe(self, &E_, &c, pipe, x + (n * c.C + grp * c.Cg) * c.H * c.W, wv, col, out,
                                            (n * c.O + grp * c.Og) * M, grp * c.Og);
                if (rc != AIEK_OK)
                    return rc;
                continue;
            }
            if (hwc) {
                const int rc = im2col_pack_hwc(&c, x + (n * c.C + grp * c.Cg) * c.H * c.W, col);
                if (rc != AIEK_OK)
                    return rc;
            } else {
                im2col_pack(&c, x + (n * c.C + grp * c.Cg) * c.H * c.W, col);
            }
            g_phase[RT_PH_IM2COL] += rt_ticks() - ti0;
            if (E_.fx && (!E_.fx_r || E_.fx_rdt == aiednn_s8) && g_qblk && (K <= g_qblk->args[0].dims[1] || g_kcblk)) {
                // requant on the core (K-chained when K > BK, an s8 residual added there too): int8 leaves the array
                int8_t *o8 = (int8_t *)out; // reuses the int32 scratch ([M][Og] bytes)
                const int8_t *res = E_.fx_r ? (const int8_t *)E_.fx_r + (n * c.O + grp * c.Og) * M : nullptr;
                const int rc = blocked_gemm_q(self, &E_, grp * c.Og, M, c.Og, K, col, wv, o8,
                                              g_exec_const_w ? (const void *)wv : nullptr, res, M);
                if (rc != AIEK_OK)
                    return rc;
                const uint64_t te0 = rt_ticks();
                const int64_t mb8 = rt_epi_rows_per_tile(c.Og, 1);
                for (int64_t m0 = 0; m0 < M; m0 += mb8) { // [M][Og] -> dst [n][o][Ho*Wo], cache tiles
                    const int64_t mn = M - m0 < mb8 ? M - m0 : mb8;
                    for (int64_t og = 0; og < c.Og; og++) {
                        int8_t *d8 = (int8_t *)ptr[2] + (n * c.O + grp * c.Og + og) * M + m0;
                        const int8_t *s8 = o8 + m0 * c.Og + og;
                        for (int64_t m = 0; m < mn; m++)
                            d8[m] = s8[m * c.Og];
                    }
                }
                g_phase[RT_PH_EPI] += rt_ticks() - te0;
                g_fx_core++;
                continue;
            }
            void *gp[3] = {col, (void *)wv, out};
            const int rc = run_sub(prim->bind.inner, &g, &b, gp);
            if (rc != AIEK_OK)
                return rc;
            const uint64_t te0 = rt_ticks();
            const int64_t mb = rt_epi_rows_per_tile(c.Og, 4);
            for (int64_t m0 = 0; m0 < M; m0 += mb) { // [M][Og] -> dst [n][o][Ho*Wo], cache tiles
                const int64_t mn = M - m0 < mb ? M - m0 : mb;
                for (int64_t og = 0; og < c.Og; og++) {
                    const int64_t o = grp * c.Og + og;
                    rt_epi_row(&E_, (n * c.O + o) * M + m0, o, 0, out + m0 * c.Og + og, c.Og, mn);
                }
            }
            g_phase[RT_PH_EPI] += rt_ticks() - te0;
        }
    return AIEK_OK;
}

// ── col2im+<matmul>: deconvolution as one GEMM per (image, group) ──────────

static int acc_col2im(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (!conv_int8_problem(p)) {
        snprintf(why, cap, "only s8 x s8 deconvolution");
        return 0;
    }
    conv_geo c;
    conv_geometry(p, 1, &c);
    if (c.G > 1 && c.Cg < RT_MIN_GROUP_N) {
        snprintf(why, cap, "%lld input channels per group: GEMM K too small for the AIE block", (long long)c.Cg);
        return 0;
    }
    rt_prob g;
    gemm_prob(&g, c.H * c.W, c.Og * c.KH * c.KW, c.Cg, aiednn_s8);
    return acc_gemm_composite(&g, bind, why, cap);
}

static void *g_xt, *g_wr;
static size_t g_xt_cap, g_wr_cap;

static int run_col2im(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    rt_epi E_; // epilogue resolved once per call (bit-identical to rt_epilogue)
    rt_epi_init(&E_, p, ptr);
    conv_geo c;
    conv_geometry(p, 1, &c);
    const int64_t M = c.H * c.W, NN = c.Og * c.KH * c.KW, K = c.Cg, nd = md_numel(&p->md[2]);
    rt_prob g;
    gemm_prob(&g, M, NN, K, aiednn_s8);
    int8_t *xt = (int8_t *)scratch(&g_xt, &g_xt_cap, (size_t)(M * K));
    int8_t *wr = (int8_t *)scratch(&g_wr, &g_wr_cap, (size_t)(NN * K));
    int32_t *out = (int32_t *)scratch(&g_gout, &g_gout_cap, (size_t)(M * NN) * sizeof(int32_t));
    int64_t *acc = (int64_t *)scratch(&g_dacc, &g_dacc_cap, (size_t)nd * sizeof(int64_t));
    if (!xt || !wr || !out || !acc)
        return AIEK_ERR_ALLOC;
    const int8_t *x = (const int8_t *)ptr[0], *w = (const int8_t *)ptr[1];
    for (int64_t i = 0; i < nd; i++)
        acc[i] = 0;
    const rt_bind b = {nullptr, prim->bind.blk};
    for (int64_t grp = 0; grp < c.G; grp++) {
        for (int64_t og = 0; og < c.Og; og++) // [(o,kh,kw)][c] from [o][c][kh][kw]
            for (int64_t ci = 0; ci < c.Cg; ci++)
                for (int64_t k = 0; k < c.KH * c.KW; k++)
                    wr[(og * c.KH * c.KW + k) * K + ci] = w[(((grp * c.Og + og) * c.Cg + ci) * c.KH * c.KW) + k];
        for (int64_t n = 0; n < c.N; n++) {
            for (int64_t ci = 0; ci < c.Cg; ci++) // [c][h*w] -> [h*w][c]
                for (int64_t m = 0; m < M; m++)
                    xt[m * K + ci] = x[(n * c.C + grp * c.Cg + ci) * M + m];
            void *gp[3] = {xt, wr, out};
            const int rc = run_sub(prim->bind.inner, &g, &b, gp);
            if (rc != AIEK_OK)
                return rc;
            for (int64_t m = 0; m < M; m++) { // scatter-add (col2im)
                const int64_t h = m / c.W, wcol = m % c.W;
                for (int64_t og = 0; og < c.Og; og++)
                    for (int64_t kh = 0; kh < c.KH; kh++) {
                        const int64_t oh = h * c.s[0] - c.pl[0] + kh * c.d[0];
                        if (oh < 0 || oh >= c.Ho)
                            continue;
                        for (int64_t kw = 0; kw < c.KW; kw++) {
                            const int64_t ow = wcol * c.s[1] - c.pl[1] + kw * c.d[1];
                            if (ow < 0 || ow >= c.Wo)
                                continue;
                            acc[((n * c.O + grp * c.Og + og) * c.Ho + oh) * c.Wo + ow] +=
                                out[m * NN + (og * c.KH + kh) * c.KW + kw];
                        }
                    }
            }
        }
    }
    for (int64_t i = 0; i < nd; i++)
        rt_epi_put(&E_, i, (i / (c.Ho * c.Wo)) % c.O, (double)acc[i]);
    return AIEK_OK;
}

// ── dwblk:<aie dw block>: depthwise convolution, 16 output bands per launch ─
/* The AIE depthwise block (gemm_uk.cc dw_blk_s8s32: the GEMM ELF in its
 * depthwise mode): int8 items [16][4096] = zero-padded input rows of one output
 * band of one channel (pitch Wp) + KHxKW weights in the last 32 bytes,
 * geom int32 {stride, KH, KW, Wp, Rout, Wo, 0, 0} -> int32 [16][1024]. The
 * accumulators are exact, so dst = rt_epilogue(acc) equals ref:convolution. */
#define RT_DW_MAX_ITEMS 128 /* items per launch = the impl's arg 0 rows: 1 per core (16 on 4x4), 3 for dwq_blk */
#define RT_DW_BYTES 4096
#define RT_DW_WB 96 /* item tail: weights in the last 64 bytes (KH*KW <= 49), fixed-point params in the 32 before */
#define RT_DW_OUT 1024

static int is_dw_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_convolution && g->nargs == 3 && g->args[0].data_type == aiednn_s8 &&
           g->args[0].ndims == 2 && g->args[0].dims[0] >= 1 && g->args[0].dims[0] <= RT_DW_MAX_ITEMS &&
           g->args[0].dims[1] == RT_DW_BYTES && g->args[1].data_type == aiednn_s32 &&
           g->args[2].data_type == aiednn_s32 && g->args[2].ndims == 2 && g->args[2].dims[1] == RT_DW_OUT;
}

typedef struct {
    int64_t wp, rout, bands;
} dw_plan;

/* largest band of output rows whose input rows and outputs fit one item */
static int dw_band(const conv_geo *c, dw_plan *d) {
    d->wp = (c->Wo - 1) * c->s[1] + c->KW;
    for (int64_t r = c->Ho < RT_DW_OUT ? c->Ho : RT_DW_OUT; r >= 1; r--)
        if (r * c->Wo <= RT_DW_OUT && ((r - 1) * c->s[0] + c->KH) * d->wp <= RT_DW_BYTES - RT_DW_WB) {
            d->rout = r;
            d->bands = (c->Ho + r - 1) / r;
            return 1;
        }
    return 0;
}

static int acc_dwblk(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    (void)bind;
    if (!conv_int8_problem(p)) {
        snprintf(why, cap, "only s8 x s8 convolution");
        return 0;
    }
    conv_geo c;
    conv_geometry(p, 0, &c);
    dw_plan d;
    if (c.G == 1 || c.Cg != 1 || c.Og != 1 || c.KH > 7 || c.KW > 7 || c.d[0] != 1 || c.d[1] != 1 || c.s[0] != c.s[1] ||
        c.s[0] < 1 || c.s[0] > 2 || !dw_band(&c, &d)) {
        snprintf(why, cap, "not a depthwise KHxKW<=7x7, stride 1|2, undilated conv whose band fits an item");
        return 0;
    }
    /* nhwc: items gather one channel's band at stride C and scatter it back the same way (dw_item orders
     * them channel-fastest, so consecutive items share the band's cache lines) */
    return !rt_prob_has_fmt(p) || conv_fmt_ok(p, &c, why, cap, 1);
}

/* item it -> (image, channel, band): band-fastest for channel planes; channel-fastest when an operand is nhwc */
static inline void dw_item(const conv_geo *c, const dw_plan *d, int64_t it, int64_t *n, int64_t *ch, int64_t *band) {
    if (c->src_hwc || c->dst_hwc)
        *ch = it % c->C, *band = (it / c->C) % d->bands, *n = it / (c->C * d->bands);
    else
        *band = it % d->bands, *ch = (it / d->bands) % c->C, *n = it / (d->bands * c->C);
}

static void *g_dwi, *g_dwo;
static size_t g_dwi_cap, g_dwo_cap;

/* depthwise + requant with 3 items per core (gemm_uk.cc dwq_blk_s8s8): items s8 [3*cores][4096],
 * geometry s32 [8], out s8 [3*cores][1024] -- three int8 bands per launch for the same bytes */
static const aiednn_impl_t *g_dwqblk;

static int is_dwq_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_convolution && g->nargs == 3 && g->args[0].data_type == aiednn_s8 &&
           g->args[0].ndims == 2 && g->args[0].dims[0] >= 1 && g->args[0].dims[0] <= RT_DW_MAX_ITEMS &&
           g->args[0].dims[1] == RT_DW_BYTES && g->args[1].data_type == aiednn_s32 &&
           g->args[2].data_type == aiednn_s8 && g->args[2].ndims == 2 && g->args[2].dims[1] == RT_DW_OUT;
}

/* item it = (image, channel, band) of a depthwise layer into raw (zeroed by the caller): its padded
 * input rows, KHxKW weights in the tail, and (dwq) the core's requant params */
static void dw_pack_item(const conv_geo *c, const dw_plan *d, const rt_epi *E, int dwq, const int8_t *x,
                         const int8_t *w, int64_t it, int8_t *raw) {
    int64_t n, ch, band;
    dw_item(c, d, it, &n, &ch, &band);
    const int64_t oh0 = band * d->rout, rin = (d->rout - 1) * c->s[0] + c->KH;
    const int64_t q0 = c->pl[1] > 0 ? c->pl[1] : 0, q1 = c->pl[1] + c->W < d->wp ? c->pl[1] + c->W : d->wp;
    for (int64_t r = 0; r < rin; r++) {
        const int64_t ih = oh0 * c->s[0] - c->pl[0] + r;
        if (ih < 0 || ih >= c->H || q1 <= q0) /* columns q0 .. q1 hold input columns q - pl */
            continue;
        if (c->src_hwc) { /* nhwc: this channel's bytes at stride C */
            const int8_t *row = x + ((n * c->H + ih) * c->W + (q0 - c->pl[1])) * c->C + ch;
            for (int64_t q = q0; q < q1; q++)
                raw[r * d->wp + q] = row[(q - q0) * c->C];
        } else {
            memcpy(raw + r * d->wp + q0, x + ((n * c->C + ch) * c->H + ih) * c->W + (q0 - c->pl[1]), (size_t)(q1 - q0));
        }
    }
    memcpy(raw + RT_DW_BYTES - 64, w + ch * c->KH * c->KW, (size_t)(c->KH * c->KW));
    if (dwq) { /* the core requantizes (0xD8): int32 {zp, lo, hi, M}, int64 B, int32 sh */
        int32_t *qp = (int32_t *)(raw + RT_DW_BYTES - 96);
        qp[0] = (int32_t)E->dzp, qp[1] = E->fx_lo, qp[2] = E->fx_hi, qp[3] = E->fx_m[ch];
        memcpy(qp + 4, &E->fx_b[ch], 8);
        qp[6] = E->fx_sh[ch];
    }
}

/* item it's result (int8 band from the core, or int32 accumulators through the host epilogue) into dst */
static void dw_put_item(const conv_geo *c, const dw_plan *d, const rt_epi *E, int dwq, int64_t it, const void *res,
                        void *dst) {
    int64_t n, ch, band;
    dw_item(c, d, it, &n, &ch, &band);
    const int64_t oh0 = band * d->rout, rows = c->Ho - oh0 < d->rout ? c->Ho - oh0 : d->rout;
    if (c->dst_hwc) { /* nhwc: element (oh, ow) of channel ch at ((n Ho + oh) Wo + ow) O + ch */
        const int64_t d0 = (n * c->Ho + oh0) * c->Wo;
        for (int64_t i = 0; i < rows * c->Wo; i++) {
            if (dwq)
                ((int8_t *)dst)[(d0 + i) * c->O + ch] = ((const int8_t *)res)[i];
            else
                rt_epi_put(E, (d0 + i) * c->O + ch, ch, (double)((const int32_t *)res)[i]);
        }
        rt_qfuse_run((int8_t *)dst + d0 * c->O + ch, rows * c->Wo, c->O);
        return;
    }
    const int64_t d0 = ((n * c->O + ch) * c->Ho + oh0) * c->Wo;
    if (dwq)
        memcpy((int8_t *)dst + d0, res, (size_t)(rows * c->Wo));
    else
        rt_epi_row(E, d0, ch, 0, (const int32_t *)res, 1, rows * c->Wo);
    rt_qfuse_run((int8_t *)dst + d0, rows * c->Wo);
}

#define RT_NOT_TAKEN 1 /* a fast path declined (positive: never an AIEK_ status), the caller runs its own */
static int run_dwq_n(const rt_impl *self, const rt_epi *E, const conv_geo *c, const dw_plan *d, const int8_t *x,
                     const int8_t *w, void *dst); /* aiednn_rt_dwn.h: many dwq launches in one */

static int run_dwblk(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    rt_epi E_; // epilogue resolved once per call (bit-identical to rt_epilogue)
    rt_epi_init(&E_, p, ptr);
    const int dwq = E_.fx && !E_.fx_r; /* requant on the core (0xD8); a fused residual stays on the host */
    const aiednn_impl_t *gi = dwq && g_dwqblk ? g_dwqblk : self->gen; /* 3 int8 bands per core if it can */
    const int64_t NI = gi->args[0].dims[0];                           // items per launch
    conv_geo c;
    conv_geometry(p, 0, &c);
    dw_plan d;
    dw_band(&c, &d);
    const int8_t *x = (const int8_t *)ptr[0], *w = (const int8_t *)ptr[1];
    g_qfuse.claim = 1; /* every band leaves through dw_put_item */
    if (dwq && gi == g_dwqblk) {
        const int rc = run_dwq_n(self, &E_, &c, &d, x, w, ptr[2]);
        if (rc != RT_NOT_TAKEN)
            return rc;
    }
    int8_t *items = (int8_t *)scratch(&g_dwi, &g_dwi_cap, NI * RT_DW_BYTES);
    int32_t *out = (int32_t *)scratch(&g_dwo, &g_dwo_cap, NI * RT_DW_OUT * sizeof(int32_t));
    if (!items || !out)
        return AIEK_ERR_ALLOC;
    const int64_t total = c.N * c.C * d.bands;
    const int ob = gi == g_dwqblk ? 1 : (int)sizeof(int32_t); // output bytes per element
    for (int64_t base = 0; base < total; base += NI) {
        const int64_t n_items = total - base < NI ? total - base : NI;
        const uint64_t tp0 = rt_ticks();
        memset(items, 0, NI * RT_DW_BYTES);
        for (int64_t k = 0; k < n_items; k++) // item = (image, channel, band)
            dw_pack_item(&c, &d, &E_, dwq, x, w, base + k, items + k * RT_DW_BYTES);
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        const int64_t shape_i[2] = {NI, RT_DW_BYTES}, shape_g[1] = {8}, shape_o[2] = {NI, RT_DW_OUT};
        int32_t geom[8] = {(int32_t)c.s[0], (int32_t)c.KH, (int32_t)c.KW, (int32_t)d.wp,
                           (int32_t)d.rout, (int32_t)c.Wo, dwq,           0};
        aiek_tensor t[3] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * RT_DW_BYTES)},
                            {geom, 1, AIEK_INT, 32, 1, shape_g, sizeof geom},
                            {out, 2, AIEK_INT, (uint8_t)(8 * ob), 1, shape_o, (uint64_t)NI * RT_DW_OUT * ob}};
        ((rt_impl *)self)->launches++;
        const uint64_t te = rt_ticks();
        const int rc = gi->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        if (rc != AIEK_OK)
            return rc;
        const uint64_t te0 = rt_ticks();
        for (int64_t k = 0; k < n_items; k++)
            dw_put_item(&c, &d, &E_, dwq, base + k, (const int8_t *)out + k * RT_DW_OUT * ob, ptr[2]);
        g_phase[RT_PH_EPI] += rt_ticks() - te0;
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_GEMM_H */
