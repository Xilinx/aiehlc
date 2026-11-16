/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_mmt.h -- MMT (opcode 0x4D): a pre-tiled int8 GEMM mode of the GEMM ELF for short K,
 * on mlir-aie's vectorized matmul micro-kernel (mlir-aie/aie_kernels/linalg/mm_aie2.h,
 * matmul_vectorized_4x8x8_i8_i32, a submodule of this repository).
 *
 * Each core produces AIE_MMT_ROWS (128) rows x 32 channels of int8 output for K <= MMT_K = 128
 * (zero-padded; the core build has only the 128 instantiation, K = 64 is x86-reference only):
 *   - the host lays A out pre-tiled: k-round window kr (2 KB) holds rows [kr * mr, +mr) x MMT_K as
 *     [mr/4][MMT_K/8][4][8] (mr = 2048 / MMT_K; K = 64 uses rounds 0..3 only);
 *   - k-round 0's B window carries the core column's B [MMT_K/8][32/8][8][8] (rows 0..63) and the
 *     requant constants (rows 64.., aie_mmt_pack_q); the core keeps both for the launch;
 *   - per round the core zeroes an int32 [mr][32] tile (pre-tiled [mr/4][4][4][8]), runs the
 *     micro-kernel on the A window as it arrived, requantizes ((acc * M + B) >> sh) + zp, clamped,
 *     and writes int8 row m, channel c at byte m * 32 + c of its 4 KB C tile.
 * The requant is vectorized and exact: acc * M + B in 64-bit lanes (B = Bh * 2^31 + Bl), one floor
 * shift by S = min sh into int32, then the per-channel rest d = sh - S as * 2^(D - d) and a floor
 * shift by D (floor(floor(x / 2^S) / 2^d) = floor(x / 2^(S + d))). aie_mmt_pack_q refuses a layer
 * whose x / 2^S could leave int32 for this K.
 * No packing on the core; the result is the plain GEMM's + 0xF1 requant bit for bit.
 *
 * Descriptor (B row 127, first bytes only -- bytes >= 4 are never flags): [0] 0x4D [1] mesh column
 * [2] MMT_K / 64 (1 or 2).
 *
 * Kernel side: include between AIEHLC_KERNEL_PROLOGUE_BEGIN/END, after aie_modes_cfg.h;
 * the x86 build (no __AIENGINE__) gets the same arithmetic in plain C for the reference.
 ******************************************************************************/
#ifndef AIE_MODES_MMT_H
#define AIE_MODES_MMT_H

#include <stdint.h>
#include <string.h>
#if defined(__AIENGINE__) && defined(AIEK_MODE_MMT) && AIEK_MODE_MMT
/* the core build only: kc.sh puts the mlir-aie submodule's aie_kernels/ on the kernel include path
 * (found in place, so its own "../aie_kernel_utils.h" resolves) */
#include "linalg/mm_aie2.h"
#endif

#define AIE_MODE_OP_MMT 0x4D
#define AIE_MMT_ROWS 128       /* output rows per core per tile */
#define AIE_MMT_N 32           /* channels per core */
#define AIE_MMT_QOFF (64 * 64) /* requant constants' offset in the resident B copy (window row 64) */
#define AIE_MMT_QBYTES 1024    /* hdr int32[8] {zp, lo, hi, S, D}, M / Bh / Bl int32[4][16], mul int16[4][16] */
#define AIE_MMT_RES (AIE_MMT_QOFF + AIE_MMT_QBYTES) /* resident bytes copied from k-round 0's B window */
/* (the mmt_blk_s8s8 AIEK_SPEC writes its weight as [cols, 5120]: keep the two equal) */

/* ── host: layouts ─────────────────────────────────────────────────────────────── */

/* A rows [rows][lda] (K = kk <= mk, zero-padded to mk) -> one core row's A, round by round:
 * round kr at kr * 2048: rows [kr * mr, +mr) pre-tiled [mr/4][mk/8][4][8] */
static inline void aie_mmt_pack_a(const int8_t *a, int64_t rows, int64_t lda, int kk, int mk, int8_t *dst) {
    const int mr = 2048 / mk;
    memset(dst, 0, 16384);
    for (int64_t m = 0; m < rows && m < AIE_MMT_ROWS; m++) {
        const int kr = (int)(m / mr), ml = (int)(m % mr);
        int8_t *w = dst + kr * 2048;
        for (int k = 0; k < kk; k++)
            w[((ml / 4) * (mk / 8) + k / 8) * 32 + (ml % 4) * 8 + k % 8] = a[m * lda + k];
    }
}

/* mesh row r's packed A (aie_mmt_pack_a) -> the GEMM ELF's launch A buffer ([rows * 32][512]): k-round
 * kr's window is bytes [kr * 64, +64) of the row's 32 launch rows */
static inline void aie_mmt_a_to_launch(const int8_t *packed, int r, int8_t *launch) {
    for (int kr = 0; kr < 8; kr++)
        for (int i = 0; i < 32; i++)
            memcpy(launch + (int64_t)(r * 32 + i) * 512 + kr * 64, packed + kr * 2048 + i * 64, 64);
}

/* B rows [32 channels][ldb] (K = kk) -> the resident pre-tiled [mk/8][4][8][8] (k rows, n cols) */
static inline void aie_mmt_pack_b(const int8_t *b, int nch, int64_t ldb, int kk, int mk, int8_t *dst) {
    memset(dst, 0, (size_t)mk * AIE_MMT_N);
    for (int n = 0; n < nch && n < AIE_MMT_N; n++)
        for (int k = 0; k < kk; k++)
            dst[((k / 8) * (AIE_MMT_N / 8) + n / 8) * 64 + (k % 8) * 8 + n % 8] = b[n * ldb + k];
}

/* requant constants of 32 channels: q = ((acc * qm + qb) >> qsh) + zp, clamped to [lo, hi] (the
 * fixed-point epilogue, rt_fx_q). Lane l of channel group g (lanes = 2 rows x 8 channels) is
 * channel g * 8 + l % 8. Returns 0, or -1 if the staged arithmetic could leave int32 for |acc| <=
 * kk * 128 * 128 (then the layer must not use MMT). */
static inline int aie_mmt_pack_q(const int64_t *qb, const int32_t *qm, const int8_t *qsh, int32_t zp, int32_t lo,
                                 int32_t hi, int kk, uint8_t *dst) {
    int S = 63, D = 0;
    for (int c = 0; c < AIE_MMT_N; c++)
        S = qsh[c] < S ? qsh[c] : S;
    for (int c = 0; c < AIE_MMT_N; c++)
        D = qsh[c] - S > D ? qsh[c] - S : D;
    if (S < 0 || D > 14)
        return -1;
    const int64_t amax = (int64_t)kk * 128 * 128;
    for (int c = 0; c < AIE_MMT_N; c++) { /* |acc * M + B| / 2^S < 2^31 */
        const int64_t m = qm[c] < 0 ? -(int64_t)qm[c] : qm[c], b = qb[c] < 0 ? -qb[c] : qb[c];
        if (m > (((int64_t)1 << 62) - b) / (amax ? amax : 1) || ((amax * m + b) >> S) >= ((int64_t)1 << 31) - 1)
            return -1;
    }
    memset(dst, 0, AIE_MMT_QBYTES);
    int32_t *hdr = (int32_t *)dst, *m16 = hdr + 8, *bh16 = m16 + 64, *bl16 = bh16 + 64;
    int16_t *mul16 = (int16_t *)(bl16 + 64);
    hdr[0] = zp, hdr[1] = lo, hdr[2] = hi, hdr[3] = S, hdr[4] = D;
    for (int g = 0; g < 4; g++)
        for (int l = 0; l < 16; l++) {
            const int c = g * 8 + l % 8;
            const int64_t bh = qb[c] >> 31; /* floor */
            m16[g * 16 + l] = qm[c];
            bh16[g * 16 + l] = (int32_t)bh;
            bl16[g * 16 + l] = (int32_t)(qb[c] - bh * ((int64_t)1 << 31)); /* 0 .. 2^31 - 1 */
            mul16[g * 16 + l] = (int16_t)(1 << (D - (qsh[c] - S)));
        }
    return 0;
}

/* ── core / reference: one k-round ─────────────────────────────────────────────── */

/* the staged requant of one value (the vector path's arithmetic, lane by lane) */
static inline int8_t aie_mmt_q(int32_t acc, const uint8_t *q, int c) {
    const int32_t *hdr = (const int32_t *)q, *m16 = hdr + 8, *bh16 = m16 + 64, *bl16 = bh16 + 64;
    const int16_t *mul16 = (const int16_t *)(bl16 + 64);
    const int l = (c / 8) * 16 + c % 8;
    const int64_t x = (int64_t)acc * m16[l] + (int64_t)bh16[l] * ((int64_t)1 << 31) + bl16[l];
    const int32_t y = (int32_t)(x >> hdr[3]);
    int64_t v = (((int64_t)y * mul16[l]) >> hdr[4]) + hdr[0];
    v = v < hdr[1] ? hdr[1] : v > hdr[2] ? hdr[2] : v;
    return (int8_t)v;
}

/* round kr of a core: A window (pre-tiled rows [kr * mr, +mr)), resident B + q params -> int8 rows
 * [kr * mr, +mr) of out (row m channel c at m * 32 + c); c32 = an int32 [mr * 32] scratch */
#if defined(__AIENGINE__) && defined(AIEK_MODE_MMT) && AIEK_MODE_MMT
template <int MK>
static inline void aie_mmt_round_t(const int8_t *a, const int8_t *res, int kr, int32_t *c32, int8_t *out) {
    constexpr int mr = 2048 / MK;
    if (kr * mr >= AIE_MMT_ROWS)
        return;
    for (int i = 0; i < mr * AIE_MMT_N; i += 16)
        aie::store_v(c32 + i, aie::zeros<int32, 16>());
    matmul_vectorized_4x8x8_i8_i32<mr, MK, AIE_MMT_N>((const int8 *)a, (const int8 *)res, (int32 *)c32);
    const int32_t *hdr = (const int32_t *)(res + AIE_MMT_QOFF), *m16 = hdr + 8, *bh16 = m16 + 64, *bl16 = bh16 + 64;
    const int16_t *mul16 = (const int16_t *)(bl16 + 64);
    const aie::rounding_mode rnd = aie::get_rounding();
    aie::set_rounding(aie::rounding_mode::floor);
    const aie::vector<int32, 16> zp = aie::broadcast<int32, 16>(hdr[0]), lo = aie::broadcast<int32, 16>(hdr[1]),
                                 hi = aie::broadcast<int32, 16>(hdr[2]);
    const int S = hdr[3], D = hdr[4];
    for (int mi = 0; mi < mr / 4; mi++) /* C pre-tiled [mr/4][4 groups][4 rows][8 channels] */
        for (int g = 0; g < 4; g++)
            for (int h = 0; h < 2; h++) { /* rows 2h, 2h+1 of the 4x8 tile: 16 lanes */
                aie::accum<acc64, 16> x =
                    aie::mul(aie::load_v<16>(c32 + (mi * 4 + g) * 32 + h * 16), aie::load_v<16>(m16 + g * 16));
                aie::accum<acc64, 16> b;
                b.from_vector(aie::load_v<16>(bh16 + g * 16), 31);
                x = aie::add(x, b);
                b.from_vector(aie::load_v<16>(bl16 + g * 16), 0);
                x = aie::add(x, b);
                const aie::vector<int32, 16> y = x.template to_vector<int32>(S);
                aie::vector<int32, 16> v = aie::mul(y, aie::load_v<16>(mul16 + g * 16)).template to_vector<int32>(D);
                v = aie::max(aie::min(aie::add(v, zp), hi), lo);
                alignas(16) int8 p[16]; /* two rows' 8 channels: 8-byte pieces, no 8-lane int8 vector */
                aie::store_v(p, v.template pack<int8>());
                uint32_t *o = (uint32_t *)(out + (kr * mr + mi * 4 + 2 * h) * AIE_MMT_N + g * 8);
                const uint32_t *q = (const uint32_t *)p;
                o[0] = q[0], o[1] = q[1], o[AIE_MMT_N / 4] = q[2], o[AIE_MMT_N / 4 + 1] = q[3];
            }
    aie::set_rounding(rnd);
}
/* one instantiation (K tile 128): a K-64 copy would cost program memory for nothing */
static inline void aie_mmt_round(const int8_t *a, const int8_t *res, int kcode, int kr, int32_t *c32, int8_t *out) {
    (void)kcode;
    aie_mmt_round_t<128>(a, res, kr, c32, out);
}
#elif !defined(__AIENGINE__) /* x86: the reference / stand-in (no core build without MMT reaches here) */
static inline void aie_mmt_round(const int8_t *a, const int8_t *res, int kcode, int kr, int32_t *c32, int8_t *out) {
    (void)c32;
    const int mk = 64 * kcode, mr = 2048 / mk;
    const uint8_t *qp = (const uint8_t *)res + AIE_MMT_QOFF;
    if (kr * mr >= AIE_MMT_ROWS)
        return;
    for (int m = 0; m < mr; m++)
        for (int c = 0; c < AIE_MMT_N; c++) {
            int32_t acc = 0;
            for (int k = 0; k < mk; k++)
                acc += (int32_t)a[((m / 4) * (mk / 8) + k / 8) * 32 + (m % 4) * 8 + k % 8] *
                       (int32_t)res[((k / 8) * (AIE_MMT_N / 8) + c / 8) * 64 + (k % 8) * 8 + c % 8];
            out[(kr * mr + m) * AIE_MMT_N + c] = aie_mmt_q(acc, qp, c);
        }
}
#endif

#endif
