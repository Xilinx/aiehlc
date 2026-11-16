/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_bf16.h -- bf16 GEMM as a mode of the GEMM ELF (opcode 0xBF, AIE-ML native bf16 mmul).
 *
 * Same launch as the int8 GEMM block (gemm_uk.cc): A [GEMM_M][512 bytes], B [128][512 bytes] per mesh
 * column, 8 k-rounds of 64-byte windows, a 32 x 32 x 4-byte C tile per core. Read as bf16, a window
 * row is 32 values of K, so one launch is C[32][32] (f32) += A[32][256] * B[256][32] per core. The
 * host pre-tiles each round's 2 KB so the core loads mmul tiles straight from the window:
 *   A round kr: [mi 8][ki 4][4][8]  = A[mi*4 + r][kr*32 + ki*8 + k]   (mesh row's 32 rows)
 *   B round kr: [ki 4][ni 8][8][4]  = B[kr*32 + ki*8 + k][ni*4 + n]   (B = weights^T: [K][32 channels])
 * The 2 KB lands in the window's first 32 rows of 64 bytes (A: the mesh row's 32 rows, B: the
 * column's 32 real rows). The core runs mmul<4, 8, 4, bfloat16, bfloat16, accfloat> per (mi, ni, ki)
 * into an f32 tile kept across rounds (and launches with the continue flag), and writes C row-major f32
 * on the last round. Within f32 rounding of ref:, not bit-exact.
 *
 * Descriptor: B row 127 d[0] = 0xBF, d[1] = column, d[2] bit 0 continue (accumulate onto the previous
 * launch's tile), bit 1 more (do not write C: another K block follows).
 ******************************************************************************/
#ifndef AIE_MODES_BF16_H
#define AIE_MODES_BF16_H

#include <stdint.h>
#include <string.h>

#define AIE_MODE_OP_BF16 0xBF
#define AIE_BF16_K 256 /* K per launch (8 rounds x 32) */

/* ── host: layouts (bf16 as uint16_t bits) ─────────────────────────────────────── */

/* a mesh row's A: rows [rows][lda] of bf16, K values k0 .. k0 + 255 (zero past kk) -> its 32 launch
 * rows ([32][512] bytes). Element t of round kr's 2 KB block sits at launch row t / 32, bf16 slot
 * kr * 32 + t % 32; t = ((m / 4) * 4 + k / 8) * 32 + (m % 4) * 8 + k % 8 -- written as direct stores. */
static inline void aie_bf16_pack_a(const uint16_t *a, int64_t rows, int64_t lda, int64_t k0, int64_t kk,
                                   uint8_t *launch_rows) {
    uint16_t *L = (uint16_t *)launch_rows; /* [32][256] bf16 */
    for (int kr = 0; kr < 8; kr++)
        for (int m = 0; m < 32; m++) {
            const uint16_t *src = m < rows ? a + m * lda : 0;
            for (int ki = 0; ki < 4; ki++) {
                const int t0 = ((m / 4) * 4 + ki) * 32 + (m % 4) * 8; /* 8 consecutive t: one launch row */
                uint16_t *dst = L + (t0 / 32) * 256 + kr * 32 + t0 % 32;
                const int64_t kg0 = k0 + kr * 32 + ki * 8;
                for (int k = 0; k < 8; k++)
                    dst[k] = src && kg0 + k < kk ? src[kg0 + k] : 0;
            }
        }
}

/* a mesh column's B: 32 channels of weights [nch][ldb] (row = channel, K contiguous), K values k0 ..
 * k0 + 255 -> the column's first 32 B-block rows ([32][512] bytes); t = ((k / 8) * 8 + n / 4) * 32 +
 * (k % 8) * 4 + n % 4 */
static inline void aie_bf16_pack_b(const uint16_t *w, int nch, int64_t ldb, int64_t k0, int64_t kk, uint8_t *blk_rows) {
    uint16_t *L = (uint16_t *)blk_rows;
    for (int kr = 0; kr < 8; kr++)
        for (int n = 0; n < 32; n++) {
            const uint16_t *src = n < nch ? w + n * ldb : 0;
            for (int k = 0; k < 32; k++) {
                const int t = ((k / 8) * 8 + n / 4) * 32 + (k % 8) * 4 + n % 4;
                const int64_t kg = k0 + kr * 32 + k;
                L[(t / 32) * 256 + kr * 32 + t % 32] = src && kg < kk ? src[kg] : 0;
            }
        }
}

/* ── core / emulation: one round ──────────────────────────────────────────────── */
#if defined(__AIENGINE__) || defined(AIE_BF16_EMU)
/* acc: f32 [64 tiles][16] (tile mi * 8 + ni, row-major 4 x 4); first: start from zero */
static inline void aie_bf16_round(const bfloat16 *a, const bfloat16 *b, float *acc, int first) {
    for (int mi = 0; mi < 8; mi++)
        for (int ni = 0; ni < 8; ni++) {
            aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat> m;
            float *t = acc + (mi * 8 + ni) * 16;
            if (first) {
                m.mul(aie::load_unaligned_v<32>(a + mi * 128, 16), aie::load_unaligned_v<32>(b + ni * 32, 16));
            } else {
                aie::accum<accfloat, 16> c;
                c.from_vector(aie::load_v<16>(t));
                m = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>(c);
                m.mac(aie::load_unaligned_v<32>(a + mi * 128, 16), aie::load_unaligned_v<32>(b + ni * 32, 16));
            }
            for (int ki = 1; ki < 4; ki++)
                m.mac(aie::load_unaligned_v<32>(a + mi * 128 + ki * 32, 16),
                      aie::load_unaligned_v<32>(b + ki * 256 + ni * 32, 16));
            aie::store_v(t, m.template to_vector<float>());
        }
}

/* the f32 tile -> C row-major [32][32] */
static inline void aie_bf16_out(const float *acc, float *out) {
    for (int mi = 0; mi < 8; mi++)
        for (int ni = 0; ni < 8; ni++)
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    out[(mi * 4 + r) * 32 + ni * 4 + c] = acc[(mi * 8 + ni) * 16 + r * 4 + c];
}
#endif

/* x86 reference of one launch on one core: A [32][256] and B^T [32][256] (bf16 bits, row-major as the
 * caller holds them) -> C [32][32] f32 accumulated in f32 in K order */
static inline void aie_bf16_ref_tile(const uint16_t *a, int64_t lda, const uint16_t *w, int64_t ldb, int kk, float *c,
                                     int accumulate) {
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) {
            float s = accumulate ? c[m * 32 + n] : 0.f;
            for (int k = 0; k < kk; k++) {
                uint32_t ua = (uint32_t)a[m * lda + k] << 16, ub = (uint32_t)w[n * ldb + k] << 16;
                float fa, fb;
                memcpy(&fa, &ua, 4), memcpy(&fb, &ub, 4);
                s += fa * fb;
            }
            c[m * 32 + n] = s;
        }
}

#endif /* AIE_MODES_BF16_H */
