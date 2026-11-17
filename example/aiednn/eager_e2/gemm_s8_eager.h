/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host side of gemm_s8_eager.cc: C[M][N] (int32) = A[M][K] (int8) . W[N][K]^T (int8). Row-major
 * operands, no pre-tiling (the kernel's inner product is scalar). The result is exact, so the check
 * demands every int32 equal.
 ******************************************************************************/
#ifndef GEMM_S8_EAGER_H
#define GEMM_S8_EAGER_H

#include "xil_cache.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EG_HW_ROWS 4
#define EG_HW_COLS 4

/* VEK280 / AIE-ML: cores from row 3, only some columns reach DDR; VEK385 / AIE2PS: cores from row 2 */
#if defined(AIE_GEN) && AIE_GEN == 5
#define EG_PARTITION {3, 6, 0, 6}
#else
#define EG_PARTITION {0, 8, 0, 7}
#endif

static inline uint32_t eg_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static void eg_init(int8_t *A, int8_t *W, int m, int n, int k, uint32_t seed) {
    for (long i = 0; i < (long)m * k; i++)
        A[i] = (int8_t)((int)(eg_mix32((uint32_t)i + seed) % 255u) - 127);
    for (long i = 0; i < (long)n * k; i++)
        W[i] = (int8_t)((int)(eg_mix32((uint32_t)i + seed * 7u + 99u) % 255u) - 127);
}

static void eg_reference(const int8_t *A, const int8_t *W, int32_t *C, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int32_t s = 0;
            for (int q = 0; q < k; q++)
                s += (int32_t)A[(long)i * k + q] * (int32_t)W[(long)j * k + q];
            C[(long)i * n + j] = s;
        }
}

static int eg_check(const char *tag, const int32_t *got, const int32_t *want, long count) {
    long bad = 0, first = -1;
    for (long i = 0; i < count; i++)
        if (got[i] != want[i] && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld exact", tag, count - bad, count);
    if (first >= 0)
        printf(" (first at %ld: got %d, want %d)", first, (int)got[first], (int)want[first]);
    printf("\n");
    return bad != 0;
}

/* the architectural counter, not the BSP timer (the A72 BSP's xiltimer.h does not compile: skill bspheadergen2) */
static inline uint64_t eg_now(void) {
#if defined(__aarch64__)
    uint64_t v;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return 0;
#endif
}
static inline double eg_ms(uint64_t t0, uint64_t t1) {
#if defined(__aarch64__)
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return t1 > t0 ? 1000.0 * (double)(t1 - t0) / (double)f : 0.0;
#else
    return 0.0 * (double)(t1 - t0);
#endif
}

#endif // GEMM_S8_EAGER_H
