/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared host side of the elementwise / reduction / data-movement / indexing
 * examples (tutorial/aieprogram/02_elementwise .. 05_indexing):
 * deterministic test data, compare, a timer, the board partition. The kernels use none of it -- and the
 * tensor sizes are #defines in each example: the kernel trace sees only the
 * object-like #defines of the .cc itself, not those of included headers.
 ******************************************************************************/
#ifndef SIMPLEOPS_LOAD_H
#define SIMPLEOPS_LOAD_H
#include "unistd.h"
#include "xil_cache.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint32_t ops_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Full int8 range, so a saturating add really saturates somewhere.
static void ops_fill(int8_t *p, long n, uint32_t seed) {
    for (long i = 0; i < n; i++)
        p[i] = (int8_t)(ops_mix32((uint32_t)i + seed) & 0xff);
}

// VEK385 (AIE_GEN 5): partition {3, 6, 0, 6}; VEK280 (AIE-ML): {0, 8, 0, 7}.
#if defined(AIE_GEN) && AIE_GEN == 5
#define OPS_PARTITION {3, 6, 0, 6}
#else
#define OPS_PARTITION {0, 8, 0, 7}
#endif

// The Arm generic timer: the BSP's XTime needs a per-board timestamp macro the
// A72 (VEK280) BSP does not define.
static inline uint64_t ops_now(void) {
#if defined(__aarch64__)
    uint64_t v;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return 0;
#endif
}
static double ops_ms(uint64_t t0, uint64_t t1) {
#if defined(__aarch64__)
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return t1 > t0 ? 1000.0 * (double)(t1 - t0) / (double)f : 0.0;
#else
    return 0.0 * (double)(t1 - t0);
#endif
}

static int8_t ops_sat8(int v) { return (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v); }

// Count mismatches between got and want, print the first few with (row, col).
static long ops_compare(const char *what, const int8_t *got, const int8_t *want, long rows, long cols) {
    long bad = 0;
    for (long i = 0; i < rows * cols; i++)
        if (got[i] != want[i] && ++bad <= 8)
            printf("MISMATCH %s[%ld][%ld]: got %d, expected %d\n", what, i / cols, i % cols, got[i], want[i]);
    printf("%s: %ld mismatches out of %ld\n", bad ? "FAIL" : "PASS", bad, rows * cols);
    return bad;
}
#endif
