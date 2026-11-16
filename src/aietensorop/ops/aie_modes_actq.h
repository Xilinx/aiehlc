/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_actq.h -- exact quantized activation (GELU, SiLU/swish -> int8) on the core.
 *
 * Definition (src/aiednn/aiednn_rt_actq.h, rt_actq_steps): per output channel c the host
 * gives M_c (int32) and B_c (int64) so that z = acc * M_c + B_c is y * 2^S at one per-layer
 * scale; the activation is a step list sorted by Z_k, (int64 Z_k, int8 d_k = +-1), and
 *     code = clamp(c_lo + sum_{Z_k <= z} d_k, lo, hi).
 * The list may step down (GELU/SiLU dip below zero) and repeat a Z (a 2-code jump).
 *
 * Core layout (aie_actq_pack): the running sum is precomputed, so a lookup is one binary
 * search for the number of Z_k <= z plus one table read:
 *     int32 n | int16 lo | int16 hi                           (8 bytes; S is folded into M/B)
 *     int64 Z[n]                                              (8-byte aligned)
 *     int16 P[n + 1], P[i] = c_lo + sum_{k < i} d_k
 * Codes are int16 so a u8 destination (0..255) fits; the store keeps the low byte.
 *
 * Plain C: the same text is the kernel helper and the x86 reference.
 ******************************************************************************/
#ifndef AIE_MODES_ACTQ_H
#define AIE_MODES_ACTQ_H

#include <stdint.h>

#define AIE_ACTQ_MAX_STEPS 288
#define AIE_ACTQ_HDR 8

static inline int aie_actq_bytes(int n) { return AIE_ACTQ_HDR + 8 * n + 2 * (n + 1); }

/* Host: step list -> core table. lo/hi = the destination range (-128/127 for s8, 0/255 for
 * u8). Returns the size, or -1 if the list is unusable. */
static inline int aie_actq_pack(const int64_t *Z, const int8_t *d, int n, int16_t c_lo, int16_t lo, int16_t hi,
                                uint8_t *buf) {
    if (n < 0 || n > AIE_ACTQ_MAX_STEPS || lo > hi)
        return -1;
    for (int k = 1; k < n; k++)
        if (Z[k] < Z[k - 1])
            return -1;
    *(int32_t *)buf = n;
    *(int16_t *)(buf + 4) = lo;
    *(int16_t *)(buf + 6) = hi;
    int64_t *zt = (int64_t *)(buf + AIE_ACTQ_HDR);
    int16_t *pt = (int16_t *)(buf + AIE_ACTQ_HDR + 8 * n);
    int16_t run = c_lo;
    pt[0] = run;
    for (int k = 0; k < n; k++) {
        zt[k] = Z[k];
        run = (int16_t)(run + d[k]);
        pt[k + 1] = run;
    }
    return aie_actq_bytes(n);
}

/* One code: the number of Z_k <= z by binary search (upper bound), then P. */
static inline int32_t aie_actq_code(const uint8_t *tbl, int64_t z) {
    const int32_t n = *(const int32_t *)tbl;
    const int64_t *zt = (const int64_t *)(tbl + AIE_ACTQ_HDR);
    const int16_t *pt = (const int16_t *)(tbl + AIE_ACTQ_HDR + 8 * n);
    int32_t lo_i = 0, hi_i = n; /* answer in [lo_i, hi_i] */
    while (lo_i < hi_i) {
        const int32_t mid = (lo_i + hi_i) >> 1;
        if (zt[mid] <= z)
            lo_i = mid + 1;
        else
            hi_i = mid;
    }
    const int32_t code = pt[lo_i];
    const int32_t lo = *(const int16_t *)(tbl + 4), hi = *(const int16_t *)(tbl + 6);
    return code < lo ? lo : code > hi ? hi : code;
}

/* A row-major [rows][cols] int32 accumulator tile -> 8-bit codes (s8, or u8 as its bit
 * pattern); column j is output channel ch0 + j, so it uses M[ch0 + j] / B[ch0 + j]. */
static inline void aie_actq_apply(const int32_t *acc, int8_t *out, int rows, int cols, const int32_t *M,
                                  const int64_t *B, int ch0, const uint8_t *tbl) {
    for (int j = 0; j < cols; j++) {
        const int64_t m = M[ch0 + j], b = B[ch0 + j];
        for (int i = 0; i < rows; i++)
            out[i * cols + j] = (int8_t)aie_actq_code(tbl, (int64_t)acc[i * cols + j] * m + b);
    }
}

#endif
