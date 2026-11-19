/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host side of the compact gemm_kacc form (tpl_two_instances.cc, tutorial/aiednn/howto/templatekernel.md):
 *   C[M][N] = epilogue(A[M][K] . W[N][K]^T),  bf16 in, fp32 accumulate, bf16 out.
 *
 * Pre-tiling. The kernel loads A and W in chunks of 16 rows x TK columns and
 * feeds aie::mmul<8, 8, 8> from them without repacking, so the host stores each
 * chunk in mmul-operand order (the DMA moves whole words and cannot do it):
 *   A chunk: [mi 2][kb TK/8][8 m][8 k]  (8x8 M x K tiles, row-major)
 *   W chunk: [ni 2][kb TK/8][8 k][8 n]  (8x8 K x N tiles, row-major)
 * A chunk stays where its rows / columns are in the [M][K] tensor; only the
 * order of its 16 x TK elements changes. The DMA delivers that region row by
 * row, so the window holds the tiles back to back.
 *
 * Test data are multiples of 1/4 (A: k/4, k in [-4, 4]; W: k/4, k in [-2, 2]):
 * every fp32 partial sum is exact, so the result does not depend on the order
 * the core accumulates in and only the final bf16 rounding (RNE) matters.
 ******************************************************************************/
#ifndef GEMM_KACC_BF16_H
#define GEMM_KACC_BF16_H

#include "unistd.h"
#include "xil_cache.h"
#include "xiltimer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AD_HW_ROWS 4
#define AD_HW_COLS 4

static inline float ad_bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint16_t ad_f2bf(float f) { // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static inline uint32_t ad_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Logical (row-major) test data.
static void ad_init(uint16_t *A, uint16_t *W, int m, int n, int k) {
    for (long i = 0; i < (long)m * k; i++)
        A[i] = ad_f2bf(((int)(ad_mix32((uint32_t)i + 1234u) % 9u) - 4) * 0.25f);
    for (long i = 0; i < (long)n * k; i++)
        W[i] = ad_f2bf(((int)(ad_mix32((uint32_t)i + 5678u) % 5u) - 2) * 0.25f);
}

// Logical [rows][k] -> pre-tiled chunks of 16 rows x tk (see the file header).
// kmajor_tile = 0: 8x8 tiles row-major in (row, k)  -- the mmul A operand
// kmajor_tile = 1: 8x8 tiles row-major in (k, row)  -- the mmul B operand
static void ad_pretile(const uint16_t *src, uint16_t *dst, int rows, int k, int tk, int kmajor_tile) {
    for (int r0 = 0; r0 < rows; r0 += 16)
        for (int k0 = 0; k0 < k; k0 += tk)
            for (int p = 0; p < 16 * tk; p++) {
                const int tile = p / 64, in = p % 64, half = tile / (tk / 8), kb = tile % (tk / 8);
                const int a = in / 8, b = in % 8; // position inside the 8x8 tile
                const int r = half * 8 + (kmajor_tile ? b : a), kk = kb * 8 + (kmajor_tile ? a : b);
                dst[(long)(r0 + p / tk) * k + k0 + p % tk] = src[(long)(r0 + r) * k + k0 + kk];
            }
}

// fp32 reference, one rounding, the same epilogue.
static void ad_reference(const uint16_t *A, const uint16_t *W, uint16_t *C, int m, int n, int k, int relu) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int q = 0; q < k; q++)
                acc += ad_bf2f(A[(long)i * k + q]) * ad_bf2f(W[(long)j * k + q]);
            if (relu && acc < 0.0f)
                acc = 0.0f;
            C[(long)i * n + j] = ad_f2bf(acc);
        }
}

// Bit-exact count and worst distance in bf16 ULPs; PASS = every output within 1 ULP.
static int ad_check(const uint16_t *got, const uint16_t *want, long count) {
    long exact = 0, at = -1;
    int worst = 0;
    for (long i = 0; i < count; i++) {
        const int d = (got[i] >> 15) == (want[i] >> 15) ? abs((int)got[i] - (int)want[i]) : 0x7fff;
        exact += d == 0;
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    printf("[gemm_kacc] check: %ld / %ld bit-exact, max %d ULP", exact, count, worst);
    if (at >= 0)
        printf(" (at %ld: got %g, want %g)", at, ad_bf2f(got[at]), ad_bf2f(want[at]));
    printf("\n%s\n", worst <= 1 ? "PASS" : "FAIL");
    return worst > 1;
}

static double ad_ms(XTime t0, XTime t1) { return t1 > t0 ? 1000.0 * (double)(t1 - t0) / COUNTS_PER_SECOND : 0.0; }

// The runtime's per-launch phase counters (C++ linkage, as in the archive). They
// count only in a `source script/aiehlc.sh --profiling` build; otherwise all 0.
void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);

// Where one launch's time went: kernel load, BD configuration, core enable,
// DMA start, waiting for the DMA / cores (PMU cycles, % of the accounted total).
static void ad_report_phases(void) {
    unsigned long long ph[4] = {0, 0, 0, 0}, wio = 0, total = 0;
    unsigned int n[4], nw;
    __Runtime_phase_cycles(ph, n);
    __Runtime_wait_io_cycles(&wio, &nw);
    const unsigned long long d[5] = {ph[0], ph[1], ph[2], ph[3], wio};
    for (int i = 0; i < 5; i++)
        total += d[i];
    if (!total) {
        printf("[gemm_kacc] PERF phases: not counted (build with `source script/aiehlc.sh --profiling`)\n");
        return;
    }
    const char *name[5] = {"kload", "bdcfg", "coreen", "startio", "wait_io"};
    printf("[gemm_kacc] PERF phases (PMU cycles):");
    for (int i = 0; i < 5; i++)
        printf(" %s %llu (%.1f%%)", name[i], d[i], 100.0 * (double)d[i] / (double)total);
    printf("\n");
}

#endif // GEMM_KACC_BF16_H
