/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the requant variant on the board: gemm_q<M, N, K, TK, TM> -- 8-bit codes out of the core,
 * the per-channel fixed-point requant words carried in the weights' extra chunk. Self-checks every code against
 * eager_gemm_q_ref (src/aietensorop/eager/gemm_q_eager_ref.h), which is aiednn's rt_fx_q definition.
 ******************************************************************************/
#include "gemm_q_eager_kernel.cc"
#include <stdio.h>
#include <stdlib.h>

#define GM 128
#define GN 64
#define GK 256
#define GTK 64
#define GTM 16

static void gemm_q_run(int8_t *A, int8_t *W, int8_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_q<GM, GN, GK, GTK, GTM><<<mesh>>>(A, W, C);
}

static uint32_t gq_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

int main() {
    printf("=== aiednn eager gemm_q<%d, %d, %d, %d, %d>: C = requant(A . W^T) on the core ===\n", GM, GN, GK, GTK, GTM);
    const long ld = GK + GTK, na = (long)GM * ld, nw = (long)GN * ld, nc = (long)GM * GN;
    int8_t *A = (int8_t *)calloc(na, 1), *W = (int8_t *)calloc(nw, 1), *got = (int8_t *)calloc(nc, 1);
    int8_t *want = (int8_t *)calloc(nc, 1);
    for (long m = 0; m < GM; m++)
        for (long k = 0; k < GK; k++)
            A[m * ld + k] = (int8_t)((int)(gq_mix((uint32_t)(m * ld + k) + 11u) % 255u) - 127);
    for (long n = 0; n < GN; n++) {
        for (long k = 0; k < GK; k++)
            W[n * ld + k] = (int8_t)((int)(gq_mix((uint32_t)(n * ld + k) + 7777u) % 255u) - 127);
        /* realistic per-channel words: M in [2^30, 2^31), sh ~ 40, a 64-bit bias at the 2^sh scale */
        const uint32_t h = gq_mix((uint32_t)n + 99u);
        const int32_t M = (int32_t)(0x40000000u + (h & 0x3fffffffu));
        const int32_t sh = 38 + (int32_t)(h % 5u);
        const int64_t B = ((int64_t)((int32_t)(gq_mix(h) % 2001u) - 1000) << 30) + ((int64_t)1 << (sh - 1));
        const int32_t q[EG_Q_WORDS] = {
            M, (int32_t)(uint32_t)(uint64_t)B, (int32_t)(B >> 32), sh, (int32_t)(n % 7) - 3, n % 2 ? 0 : -128, 127, 0};
        memcpy(W + n * ld + GK, q, sizeof q);
    }
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw),
           *Cd = (int8_t *)__Runtime_Alloc(nc);
    memcpy(Ad, A, na);
    memcpy(Wd, W, nw);
    memset(Cd, 0, nc);
    Xil_DCacheFlushRange((UINTPTR)Ad, na);
    Xil_DCacheFlushRange((UINTPTR)Wd, nw);
    gemm_q_run(Ad, Wd, Cd);
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc);
    memcpy(got, Cd, nc);
    const int64_t a_shape[2] = {GM, ld}, w_shape[2] = {GN, ld}, c_shape[2] = {GM, GN};
    const aiek_tensor t[3] = {{A, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)na},
                              {W, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)nw},
                              {want, 2, AIEK_INT, 8, 1, c_shape, (uint64_t)nc}};
    eager_gemm_q_ref(t, 3, GM, GN, GK, GTK);
    long bad = 0, first = -1;
    for (long i = 0; i < nc; i++)
        if (got[i] != want[i] && bad++ == 0)
            first = i;
    printf("[gemm_q] check: %ld / %ld exact", nc - bad, nc);
    if (first >= 0)
        printf(" (first at %ld: got %d, want %d)", first, got[first], want[first]);
    printf("\n%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
