/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the float GEMM template on the board: two instances of gemm_bf16<M, N, K, TK, TM> in one
 * ELF. bf16 operands, f32 accumulation; self-checks every f32 BIT against eager_gemm_bf16_ref, computed here
 * on the A72 in the same order without fused multiply-add.
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "gemm_bf16_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void g64_run(int8_t *A, int8_t *W, int32_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_bf16<64, 64, 128, 32, 16><<<mesh>>>(A, W, C);
}

static void g32_run(int8_t *A, int8_t *W, int32_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_bf16<128, 32, 96, 32, 8><<<mesh>>>(A, W, C);
}

static uint32_t gb_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/* a random bf16 in about [-4, 4] (round to nearest even of an f32) */
static uint16_t gb_val(uint32_t seed) {
    const float f = (float)((((double)(gb_mix(seed) & 0xffffff)) / 16777216.0 - 0.5) * 8.0);
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static long gb_case(const char *name, long M, long N, long K, long TK, void (*run)(int8_t *, int8_t *, int32_t *)) {
    const long na = M * 2 * K, nw = N * 2 * K, nc = M * N * 4;
    uint16_t *A = (uint16_t *)calloc(M * K, 2), *W = (uint16_t *)calloc(N * K, 2);
    float *got = (float *)calloc(M * N, 4), *want = (float *)calloc(M * N, 4);
    for (long i = 0; i < M * K; i++)
        A[i] = gb_val((uint32_t)i + 11u);
    for (long i = 0; i < N * K; i++)
        W[i] = gb_val((uint32_t)i + 70001u);
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw);
    int32_t *Cd = (int32_t *)__Runtime_Alloc(nc);
    memcpy(Ad, A, na);
    memcpy(Wd, W, nw);
    memset(Cd, 0x55, nc);
    Xil_DCacheFlushRange((UINTPTR)Ad, na);
    Xil_DCacheFlushRange((UINTPTR)Wd, nw);
    Xil_DCacheFlushRange((UINTPTR)Cd, nc);
    run(Ad, Wd, Cd);
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc);
    memcpy(got, Cd, nc);
    const int64_t a_shape[2] = {M, 2 * K}, w_shape[2] = {N, 2 * K}, c_shape[2] = {M, N};
    const aiek_tensor t[3] = {{A, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)na},
                              {W, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)nw},
                              {want, 2, AIEK_INT, 32, 1, c_shape, (uint64_t)nc}};
    eager_gemm_bf16_ref(t, 3, M, N, K, TK);
    long bad = 0, first = -1;
    for (long i = 0; i < M * N; i++)
        if (memcmp(&got[i], &want[i], 4) && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld bit-identical", name, M * N - bad, M * N);
    if (first >= 0)
        printf(" (first at %ld: got %.9g, want %.9g)", first, got[first], want[first]);
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager gemm_bf16<M, N, K, TK, TM>: bf16 x bf16 -> f32 on the core, two instances ===\n");
    long bad = gb_case("gemm_bf16<64,64,128>", 64, 64, 128, 32, g64_run);
    bad += gb_case("gemm_bf16<128,32,96>", 128, 32, 96, 32, g32_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
