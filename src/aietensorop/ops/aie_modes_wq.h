/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_wq.h -- grouped weight-only quantization as a mode of the GEMM ELF (opcode 0x57).
 *
 * y[m][n] = sum_g s_g[n] * sum_{k in group g} a[m][k] * w[k][n]   (int8 a, int8 w, f32 group scales)
 *
 * The int8 GEMM block unchanged (gemm_uk.cc: 8 k-rounds of 64 K per launch, an int32 8x8-tiled
 * accumulator per core), except at a group boundary: every `gr` rounds the core converts its int32
 * partial tile to f32, multiplies each column by that group's scale and adds it into an f32 tile kept in
 * the C buffer; the next round starts a fresh int32 sum. Group size g = 64 * gr (g = 64 .. 16320), and a
 * group may span launches (K > 512): the descriptor's phase byte says how many rounds of the current
 * group the previous launches already summed, and the int32 accumulator carries over like 0xA0.
 * The int32 partials are exact; the f32 part is within f32 rounding of ref:, not bit-exact.
 *
 * Launch layout (per mesh column's B block, rows of K bytes; a core reads 64 bytes of every row per
 * round): rows 0..31 the column's 32 channels (as the plain GEMM), row 100 bytes [64 kr, +64) the f32
 * scales of channels 0..15 for the group that ENDS at round kr, row 101 channels 16..31 (rounds where no
 * group ends carry nothing). Row 127 = descriptor: d[0] 0x57, d[1] column, d[2] bit 0 continue (add onto
 * the previous launch's f32 tile and int32 partial), bit 1 more (C is not final), d[3] gr, d[4] phase
 * (rounds of the current group already summed before this launch). C: the core's f32 tile row-major
 * [32][32] = its 4 KB C tile.
 *
 * Zero points / unsigned operands are the host's: (a - za)(w - zw) = a w - za w - zw a + za zw, the last
 * three per group from row / column sums (aiednn_rt_wq.h), so the core stays symmetric.
 ******************************************************************************/
#ifndef AIE_MODES_WQ_H
#define AIE_MODES_WQ_H

#include <stdint.h>
#include <string.h>

#define AIE_MODE_OP_WQ 0x57
#define AIE_WQ_ROW_S0                                                                                                  \
    100 /* B rows carrying the scales: 100 (channels 0..15), 101 (16..31): in the per-launch                           \
         * tail (rows 96..127), which the memtile weight cache always streams */

/* ── host: one launch's scale rows for a mesh column ───────────────────────────── */
/* rows 100, 101 of the column's B block ([rows][512] bytes): round kr's 64 bytes = the 32 scales of the
 * group ending at round kr (s32 = nullptr: none ends there) */
static inline void aie_wq_put_scales(uint8_t *blk, int kr, const float *s32) {
    if (!s32)
        return;
    memcpy(blk + (size_t)AIE_WQ_ROW_S0 * 512 + kr * 64, s32, 64);
    memcpy(blk + (size_t)(AIE_WQ_ROW_S0 + 1) * 512 + kr * 64, s32 + 16, 64);
}

/* ── core / emulation: the flush at a group's last round ───────────────────────── */
#if defined(__AIENGINE__) || defined(AIE_WQ_EMU)
/* acc: int32 8x8 tiles [(it * n_tiles + ct) * 64], row-major inside; f: f32 [32][32] row-major;
 * s_lo / s_hi: this round's 64 bytes of B rows 100 / 101 (the scales of channels 0..15 / 16..31) */
static inline void aie_wq_flush(const int32_t *acc, float *f, const float *s_lo, const float *s_hi, int m_tiles,
                                int n_tiles) {
    for (int ct = 0; ct < n_tiles; ct++) {
        const float *s8 = ct < 2 ? s_lo + ct * 8 : s_hi + (ct - 2) * 8;
        const aie::vector<float, 8> sv8 = aie::load_unaligned_v<8>(s8);
        const aie::vector<float, 16> sv = aie::concat(sv8, sv8); /* two tile rows of 8 channels */
        for (int it = 0; it < m_tiles; it++) {
            const int32_t *t = acc + (it * n_tiles + ct) * 64;
            for (int h = 0; h < 4; h++) { /* tile rows 2h, 2h + 1 */
                const aie::vector<float, 16> x = aie::to_float<float>(aie::load_v<16>(t + 16 * h), 0);
                const aie::vector<float, 16> p = aie::mul(x, sv).template to_vector<float>();
                float *r0 = f + (it * 8 + 2 * h) * 32 + ct * 8, *r1 = r0 + 32;
                const aie::vector<float, 16> cur = aie::concat(aie::load_v<8>(r0), aie::load_v<8>(r1));
                const aie::vector<float, 16> sum = aie::add(cur, p);
                aie::store_v(r0, sum.template extract<8>(0));
                aie::store_v(r1, sum.template extract<8>(1));
            }
        }
    }
}
#endif

/* reference of one core's launch: a [32][K] (row-major, int8), w [32][K] (channel x K), scales per
 * round [8][32] (only rounds where a group ends are read), gr / phase / flags as the descriptor; acc32
 * [32][32] and f [32][32] persist across launches (continue) */
static inline void aie_wq_ref_tile(const int8_t *a, int lda, const int8_t *w, int ldw, const float *scales, int gr,
                                   int phase, int flags, int32_t *acc32, float *f) {
    if (!(flags & 1)) {
        memset(acc32, 0, 32 * 32 * sizeof(int32_t));
        for (int i = 0; i < 32 * 32; i++)
            f[i] = 0.f;
    }
    for (int kr = 0; kr < 8; kr++) {
        if ((phase + kr) % gr == 0)
            memset(acc32, 0, 32 * 32 * sizeof(int32_t));
        for (int m = 0; m < 32; m++)
            for (int n = 0; n < 32; n++) {
                int32_t s = 0;
                for (int k = kr * 64; k < kr * 64 + 64; k++)
                    s += (int32_t)a[m * lda + k] * (int32_t)w[n * ldw + k];
                acc32[m * 32 + n] += s;
            }
        if ((phase + kr) % gr == gr - 1)
            for (int m = 0; m < 32; m++)
                for (int n = 0; n < 32; n++)
                    f[m * 32 + n] += (float)acc32[m * 32 + n] * scales[kr * 32 + n];
    }
}

#endif /* AIE_MODES_WQ_H */
