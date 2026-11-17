/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_mmt.h -- short-K convolutions (K <= 128, integer epilogue, no residual) on the GEMM
 * ELF's MMT mode (aie_modes_mmt.h: mlir-aie's matmul micro-kernel, opcode 0x4D).
 *
 * The GEMM block pads K to 512 and gives a core 32 rows x 32 channels per launch, so a 1x1 conv
 * with K = 16..128 (most of mobilenet / efficientnet / regnet) uses a quarter of a launch or less.
 * MMT gives a core 128 rows x 32 channels of int8 per launch for K <= 128: one launch covers
 * 128 * rows x 32 * cols outputs. A rows go in pre-tiled (aie_mmt_pack_a), each column's B with
 * the requant constants (aie_mmt_pack_b / aie_mmt_pack_q from the layer's fixed-point epilogue,
 * rt_fx_q), and the int8 result lands in o8 [M][N] as run_q_pipe's does; the caller's pipe then
 * scatters it. Bit for bit the int8 GEMM + rt_fx_q that run_q_pipe computes.
 *
 * Not taken (the caller continues with its own path): no MMT entry or the mode compiled out of
 * the core (aiednn_gemm_mmt_available), K > 128, a residual, or a channel whose staged requant
 * could leave int32 (aie_mmt_pack_q refuses).
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_gemm.h, aie_modes_mmt.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_MMT_H
#define AIEDNN_RT_MMT_H

extern "C" int aiednn_gemm_mmt_available(void) __attribute__((weak));
/* n MMT tiles per launch (the persistent build; aiednn_rt_ntile.h has the stage / finish / C entries) */
extern "C" int aiednn_gemm_mmt_n_start(const int8_t *a, int n, const int8_t *b, int kcode) __attribute__((weak));

static const aiednn_impl_t *g_mmtblk; /* gemm_uk.cc mmt_blk_s8s8 */

static int is_mmt_block(const aiednn_impl_t *g) {
    return g->nargs == 4 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 && g->args[0].dims[1] == 16384 &&
           g->args[1].data_type == aiednn_s8 && g->args[1].dims[1] == AIE_MMT_RES &&
           g->args[2].data_type == aiednn_s32 && g->args[3].data_type == aiednn_s8 && g->args[3].dims[1] == 4096 &&
           g->args[3].dims[0] == g->args[0].dims[0] * g->args[1].dims[0];
}

static int mmt_usable(int64_t K, int has_res) {
    return g_mmtblk && K <= 128 && !has_res && (!aiednn_gemm_mmt_available || aiednn_gemm_mmt_available()) &&
           !rt_env_on("AIEDNN_NO_MMT");
}

static void *g_mma, *g_mmb, *g_mmo;
static size_t g_mma_cap, g_mmb_cap, g_mmo_cap;

/* every column's constants for channels [n0, n0 + 32 * cols): 0, or -1 if any group is refused */
static int mmt_pack_cols(const rt_epi *E, const int8_t *wv, int64_t ch0, int64_t n0, int64_t N, int64_t K, int mk,
                         int64_t cols, int8_t *bbuf) {
    for (int64_t cc = 0; cc < cols; cc++) {
        int64_t qb[AIE_MMT_N];
        int32_t qm[AIE_MMT_N];
        int8_t qsh[AIE_MMT_N];
        const int64_t c0 = n0 + cc * AIE_MMT_N, nch = N - c0 < AIE_MMT_N ? (N - c0 > 0 ? N - c0 : 0) : AIE_MMT_N;
        for (int j = 0; j < AIE_MMT_N; j++) { /* padding channels: a harmless constant (not stored) */
            const int64_t ch = ch0 + c0 + j;
            qb[j] = j < nch ? E->fx_b[ch] : 0;
            qm[j] = j < nch ? E->fx_m[ch] : 0;
            qsh[j] = (int8_t)(j < nch ? E->fx_sh[ch] : 31);
        }
        int8_t *dst = bbuf + cc * AIE_MMT_RES;
        aie_mmt_pack_b(wv + (c0 < N ? c0 : 0) * K, (int)nch, K, (int)K, mk, dst);
        if (aie_mmt_pack_q(qb, qm, qsh, (int32_t)E->dzp, E->fx_lo, E->fx_hi, (int)K, (uint8_t *)dst + AIE_MMT_QOFF))
            return -1;
    }
    return 0;
}

/* core (r, cc)'s int8 rows of a launch C tile ([rows * 32][512], core (r, cc) at rows r * 32.., bytes
 * cc * 128..; its row m channel j at byte m * 32 + j) -> o8 [M][N] at (m0, n0) */
static void mmt_tile_out(const int8_t *ct, int64_t rows, int64_t cols, int64_t m00, int64_t n00, int64_t M, int64_t N,
                         int8_t *o8) {
    for (int64_t r = 0; r < rows; r++)
        for (int64_t cc = 0; cc < cols; cc++) {
            const int64_t m0 = m00 + r * AIE_MMT_ROWS, n0 = n00 + cc * AIE_MMT_N;
            const int64_t nn = N - n0 < AIE_MMT_N ? N - n0 : AIE_MMT_N;
            for (int64_t m = 0; m < AIE_MMT_ROWS && m0 + m < M && nn > 0; m++) {
                const int64_t o = m * AIE_MMT_N; /* byte o of the core's 4 KB: row o / 128, column o % 128 */
                memcpy(o8 + (m0 + m) * N + n0, ct + (r * 32 + o / 128) * 512 + cc * 128 + o % 128, (size_t)nn);
            }
        }
}

/* the persistent build: every m tile packed once into the n-tile stage, up to ntile_max tiles a launch */
static int run_mmt_n(const rt_impl *self, int64_t M, int64_t N, int64_t K, const int8_t *col, int64_t rows,
                     int64_t cols, int mk, const int8_t *bbuf, int8_t *o8) {
    const int64_t MR = AIE_MMT_ROWS * rows, NC = AIE_MMT_N * cols, tb = rows * 16384;
    const int64_t nnb = (N + NC - 1) / NC, nmb = (M + MR - 1) / MR, nmax = aiednn_gemm_ntile_max();
    int8_t *stage = aiednn_gemm_ntile_stage_a(nmb, tb);
    int8_t *tmp = (int8_t *)scratch(&g_mma, &g_mma_cap, 16384);
    if (!stage || !tmp)
        return AIEK_ERR_ALLOC;
    const uint64_t ta0 = rt_ticks();
    for (int64_t mb = 0; mb < nmb; mb++)
        for (int64_t r = 0; r < rows; r++) {
            const int64_t m0 = mb * MR + r * AIE_MMT_ROWS,
                          mn = M - m0 < AIE_MMT_ROWS ? (M - m0 > 0 ? M - m0 : 0) : AIE_MMT_ROWS;
            aie_mmt_pack_a(col + (mn ? m0 : 0) * K, mn, K, (int)K, mk, tmp);
            aie_mmt_a_to_launch(tmp, (int)r, stage + mb * tb);
        }
    g_phase[RT_PH_APACK] += rt_ticks() - ta0;
    for (int64_t nb = 0; nb < nnb; nb++)
        for (int64_t c0 = 0; c0 < nmb; c0 += nmax) {
            const int64_t cn = nmb - c0 < nmax ? nmb - c0 : nmax;
            const uint64_t te = rt_ticks();
            int rc = aiednn_gemm_mmt_n_start(stage + c0 * tb, (int)cn, bbuf + nb * cols * AIE_MMT_RES, mk / 64);
            if (rc == AIEK_OK)
                rc = aiednn_gemm_ntile_finish();
            ((rt_impl *)self)->t_entry += rt_ticks() - te;
            ((rt_impl *)self)->launches++;
            if (rc != AIEK_OK)
                return rc;
            const uint64_t tc0 = rt_ticks();
            for (int64_t j = 0; j < cn; j++)
                mmt_tile_out(aiednn_gemm_ntile_c((int)j), rows, cols, (c0 + j) * MR, nb * NC, M, N, o8);
            g_phase[RT_PH_CUNPACK] += rt_ticks() - tc0;
        }
    return AIEK_OK;
}

/* the int8 GEMM o8 [M][N] = requant(col [M][K] x wv [N][K]^T) of one (image, group); the im2col
 * rows must be in col. AIEK_ERR_PARAM when MMT does not take it (nothing was launched). */
static int run_mmt(const rt_impl *self, const rt_epi *E, int64_t M, int64_t N, int64_t K, const int8_t *col,
                   const int8_t *wv, int64_t ch0, int8_t *o8) {
    const aiednn_impl_t *g = g_mmtblk;
    const int64_t rows = g->args[0].dims[0], cols = g->args[1].dims[0];
    const int64_t MR = AIE_MMT_ROWS * rows, NC = AIE_MMT_N * cols;
    const int kcode = 2, mk = 128; /* the core has only the K-128 micro-kernel (aie_modes_mmt.h) */
    const int64_t nnb = (N + NC - 1) / NC, nmb = (M + MR - 1) / MR;
    int8_t *abuf = (int8_t *)scratch(&g_mma, &g_mma_cap, (size_t)(rows * 16384));
    int8_t *bbuf = (int8_t *)scratch(&g_mmb, &g_mmb_cap, (size_t)(nnb * cols * AIE_MMT_RES));
    int8_t *obuf = (int8_t *)scratch(&g_mmo, &g_mmo_cap, (size_t)(rows * cols * 4096));
    if (!abuf || !bbuf || !obuf)
        return AIEK_ERR_ALLOC;
    const uint64_t tb0 = rt_ticks();
    for (int64_t nb = 0; nb < nnb; nb++) /* every n block's B + constants first: a refusal launches nothing */
        if (mmt_pack_cols(E, wv, ch0, nb * NC, N, K, mk, cols, bbuf + nb * cols * AIE_MMT_RES))
            return AIEK_ERR_PARAM;
    g_phase[RT_PH_BPACK] += rt_ticks() - tb0;
    if (aiednn_gemm_mmt_n_start && ntile_mblocks(1, 0)) {
        /* the n-tile path launches through aiednn_gemm_mmt_n_start, not the entry: call the entry once with
         * no arguments (it returns AIEK_ERR_ARGS, launching nothing) so dnn_build(modes="auto")'s usage run
         * counts the MMT mode in */
        (void)g->entry(nullptr, 0);
        return run_mmt_n(self, M, N, K, col, rows, cols, mk, bbuf, o8);
    }
    int32_t geom = kcode;
    const int64_t sa[2] = {rows, 16384}, sb[2] = {cols, AIE_MMT_RES}, sg[1] = {1}, so[2] = {rows * cols, 4096};
    for (int64_t mb = 0; mb < nmb; mb++) {
        const uint64_t ta0 = rt_ticks();
        for (int64_t r = 0; r < rows; r++) {
            const int64_t m0 = mb * MR + r * AIE_MMT_ROWS,
                          mn = M - m0 < AIE_MMT_ROWS ? (M - m0 > 0 ? M - m0 : 0) : AIE_MMT_ROWS;
            aie_mmt_pack_a(col + (mn ? m0 : 0) * K, mn, K, (int)K, mk, abuf + r * 16384);
        }
        g_phase[RT_PH_APACK] += rt_ticks() - ta0;
        for (int64_t nb = 0; nb < nnb; nb++) {
            aiek_tensor t[4] = {{abuf, 2, AIEK_INT, 8, 1, sa, (uint64_t)(rows * 16384)},
                                {bbuf + nb * cols * AIE_MMT_RES, 2, AIEK_INT, 8, 1, sb, (uint64_t)(cols * AIE_MMT_RES)},
                                {&geom, 1, AIEK_INT, 32, 1, sg, 4},
                                {obuf, 2, AIEK_INT, 8, 1, so, (uint64_t)(rows * cols * 4096)}};
            const uint64_t te = rt_ticks();
            const int rc = g->entry(t, 4);
            ((rt_impl *)self)->t_entry += rt_ticks() - te;
            ((rt_impl *)self)->launches++;
            if (rc != AIEK_OK)
                return rc;
            const uint64_t tc0 = rt_ticks();
            for (int64_t r = 0; r < rows; r++) /* core (r, cc): rows m0 + m, channels n0 + cc * 32 + j */
                for (int64_t cc = 0; cc < cols; cc++) {
                    const int64_t m0 = mb * MR + r * AIE_MMT_ROWS, n0 = nb * NC + cc * AIE_MMT_N;
                    const int64_t nn = N - n0 < AIE_MMT_N ? N - n0 : AIE_MMT_N;
                    const int8_t *tile = obuf + (r * cols + cc) * 4096;
                    for (int64_t m = 0; m < AIE_MMT_ROWS && m0 + m < M && nn > 0; m++)
                        memcpy(o8 + (m0 + m) * N + n0, tile + m * AIE_MMT_N, (size_t)nn);
                }
            g_phase[RT_PH_CUNPACK] += rt_ticks() - tc0;
        }
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_MMT_H */
