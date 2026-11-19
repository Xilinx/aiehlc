/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the residual variant on the board: gemm_qr<M, N, K, TK, TM> -- requant + an 8-bit residual
 * add on the core, 8-bit codes out. The requant words ride in W's first chunk past K, the residual codes in A's
 * chunks after it. Self-checks every code against eager_gemm_qr_ref (gemm_qr_eager_ref.h).
 ******************************************************************************/
#include "gemm_qr_eager_kernel.cc"
#include <stdio.h>
#include <stdlib.h>

#define GM 128
#define GN 64
#define GK 256
#define GTK 64
#define GTM 16

static void gemm_qr_run(int8_t *A, int8_t *W, int8_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_qr<GM, GN, GK, GTK, GTM><<<mesh>>>(A, W, C);
}

static uint32_t gr_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

int main() {
    printf("=== aiednn eager gemm_qr<%d, %d, %d, %d, %d>: C = requant(A . W^T + residual) on the core ===\n", GM, GN,
           GK, GTK, GTM);
    const long ld = GK + eager_qr_xc(GN, GTK) * GTK, na = (long)GM * ld, nw = (long)GN * ld, nc = (long)GM * GN;
    int8_t *A = (int8_t *)calloc(na, 1), *W = (int8_t *)calloc(nw, 1), *got = (int8_t *)calloc(nc, 1);
    int8_t *want = (int8_t *)calloc(nc, 1);
    for (long m = 0; m < GM; m++) {
        for (long k = 0; k < GK; k++)
            A[m * ld + k] = (int8_t)((int)(gr_mix((uint32_t)(m * ld + k) + 11u) % 255u) - 127);
        for (long n = 0; n < GN; n++) /* the residual codes, after the zero chunk */
            A[m * ld + GK + GTK + n] = (int8_t)((int)(gr_mix((uint32_t)(m * GN + n) + 4242u) % 255u) - 127);
    }
    for (long n = 0; n < GN; n++) {
        for (long k = 0; k < GK; k++)
            W[n * ld + k] = (int8_t)((int)(gr_mix((uint32_t)(n * ld + k) + 7777u) % 255u) - 127);
        const uint32_t h = gr_mix((uint32_t)n + 99u);
        const int32_t M = (int32_t)(0x40000000u + (h & 0x3fffffffu)), sh = 38 + (int32_t)(h % 5u);
        const int64_t B = ((int64_t)((int32_t)(gr_mix(h) % 2001u) - 1000) << 30) + ((int64_t)1 << (sh - 1));
        const int64_t R = (int64_t)(gr_mix(h + 1u) % 4096u) << (sh - 12); /* the residual's scale at 2^sh */
        const int32_t q[EG_QR_WORDS] = {M,
                                        (int32_t)(uint32_t)(uint64_t)B,
                                        (int32_t)(B >> 32),
                                        sh,
                                        (int32_t)(n % 7) - 3,
                                        n % 2 ? 0 : -128,
                                        127,
                                        (int32_t)(n % 5) - 2,
                                        (int32_t)(uint32_t)(uint64_t)R,
                                        (int32_t)(R >> 32),
                                        (int32_t)(n % 3 == 0),
                                        0};
        memcpy(W + n * ld + GK, q, sizeof q);
    }
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw),
           *Cd = (int8_t *)__Runtime_Alloc(nc);
    memcpy(Ad, A, na);
    memcpy(Wd, W, nw);
    memset(Cd, 0, nc);
    Xil_DCacheFlushRange((UINTPTR)Ad, na);
    Xil_DCacheFlushRange((UINTPTR)Wd, nw);
    gemm_qr_run(Ad, Wd, Cd);
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc);
    memcpy(got, Cd, nc);
    const int64_t a_shape[2] = {GM, ld}, w_shape[2] = {GN, ld}, c_shape[2] = {GM, GN};
    const aiek_tensor t[3] = {{A, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)na},
                              {W, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)nw},
                              {want, 2, AIEK_INT, 8, 1, c_shape, (uint64_t)nc}};
    eager_gemm_qr_ref(t, 3, GM, GN, GK, GTK);
    long bad = 0, first = -1;
    for (long i = 0; i < nc; i++)
        if (got[i] != want[i] && bad++ == 0)
            first = i;
    printf("[gemm_qr] check: %ld / %ld exact", nc - bad, nc);
    if (first >= 0)
        printf(" (first at %ld: got %d, want %d)", first, got[first], want[first]);
    printf("\n%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
