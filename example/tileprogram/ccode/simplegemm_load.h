/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Block-load GEMM examples -- shared host side.
 *
 *   C[M, N] = A[M, K] x B[N, K]^T   (int8 in, int32 accumulate, int8 out)
 *
 * All three matrices are row-major in DDR. On the 4x4 mesh core (r, c)
 * computes the 16x16 block C[16r.., 16c..], walking K in 4 chunks of 64.
 *
 * Used by simplegemm_kernel_port.cc (DMA derived from the kernel's blocks) and
 * simplegemm_kernel_space.cc (hand-written GemmSpaces, checked against them).
 ******************************************************************************/
#include "unistd.h"
#include "xil_cache.h"
#include "xiltimer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEMM_M 64
#define GEMM_N 64
#define GEMM_K 256
#define HW_ROWS 4
#define HW_COLS 4

static inline uint32_t gemm_mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// A in [-4, 4], B in {-1, 0, 1}: a dot product stays within +-1024, and the
// int8 result is its low byte on both sides (the kernel's (int8_t) cast).
static void gemm_init(int8_t *A, int8_t *B, int8_t *C) {
    for (int i = 0; i < GEMM_M * GEMM_K; i++)
        A[i] = (int8_t)((int)(gemm_mix32((uint32_t)i + 1234u) % 9u) - 4);
    for (int i = 0; i < GEMM_N * GEMM_K; i++)
        B[i] = (int8_t)((int)(gemm_mix32((uint32_t)i + 5678u) % 3u) - 1);
    memset(C, 0, GEMM_M * GEMM_N);
}

// Bit-exact check against a CPU GEMM; prints PASS / FAIL and returns the
// number of mismatching elements.
static int gemm_verify(const int8_t *A, const int8_t *B, const int8_t *C) {
    int mismatches = 0;
    for (int m = 0; m < GEMM_M; m++)
        for (int n = 0; n < GEMM_N; n++) {
            int acc = 0;
            for (int k = 0; k < GEMM_K; k++)
                acc += A[m * GEMM_K + k] * B[n * GEMM_K + k];
            const int8_t want = (int8_t)acc;
            const int8_t got = C[m * GEMM_N + n];
            if (got != want && ++mismatches <= 16)
                printf("MISMATCH C[%d][%d] (core %d,%d): got %d, expected %d\n", m, n, m / 16, n / 16, got, want);
        }
    if (mismatches == 0)
        printf("PASS: all %d output elements match.\n", GEMM_M * GEMM_N);
    else
        printf("FAIL: %d mismatches out of %d.\n", mismatches, GEMM_M * GEMM_N);
    return mismatches;
}
