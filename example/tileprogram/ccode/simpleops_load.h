/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared host side of the elementwise / reduction / data-movement / indexing
 * examples (tutorial/aieprogram/02_elementwise .. 05_indexing):
 * deterministic test data, compare, a timer. The kernels use none of it -- and the
 * tensor sizes are #defines in each example: the kernel trace sees only the
 * object-like #defines of the .cc itself, not those of included headers.
 ******************************************************************************/
#ifndef SIMPLEOPS_LOAD_H
#define SIMPLEOPS_LOAD_H
#include "unistd.h"
#include "xil_cache.h"
#include "xiltimer.h"
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

static double ops_ms(XTime t0, XTime t1) { return t1 > t0 ? 1000.0 * (double)(t1 - t0) / COUNTS_PER_SECOND : 0.0; }

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
