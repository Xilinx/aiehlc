/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, grouped weight-only quantization on the board: two instances of gemm_wq<M, N, K, GK, TM> in one
 * ELF (groups of 32 and 128). Random int8 src, 4-bit weights, scales and zero points; checked against
 * eager_gemm_wq_ref, the same f32 accumulation in the same order, the scale products rounded on their own
 * (-ffp-contract=off semantics through volatile: the A72 compiler would fuse them, the core does not).
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "gemm_wq_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void wq32_run(int8_t *A, int8_t *W, int32_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_wq<128, 64, 512, 32, 16><<<mesh>>>(A, W, C);
}

static void wq128_run(int8_t *A, int8_t *W, int32_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_wq<64, 64, 1024, 128, 16><<<mesh>>>(A, W, C);
}

static uint32_t wq_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static long wq_case(const char *name, long M, long N, long K, long GK, void (*run)(int8_t *, int8_t *, int32_t *)) {
    const long na = M * K, nw = N * K, nc = M * N;
    int8_t *A = (int8_t *)calloc(na, 1), *W = (int8_t *)calloc(nw, 1);
    float *want = (float *)calloc(nc, 4);
    for (long i = 0; i < na; i++)
        A[i] = (int8_t)(wq_mix((uint32_t)i + 1u) & 0xff);
    for (long n = 0; n < N; n++)
        for (long g = 0; g < K / GK; g++) {
            int8_t *ch = W + n * K + g * GK;
            for (long i = 0; i < GK / 2; i++)
                ch[i] = (int8_t)(wq_mix((uint32_t)(n * K + g * GK + i) + 2u) & 0xff);
            const float s = 0.005f + (float)(wq_mix((uint32_t)(n * 977 + g) + 3u) % 1000) * 4.5e-5f;
            const int32_t zw = (int32_t)(wq_mix((uint32_t)(n * 977 + g) + 4u) % 7) - 3;
            memcpy(ch + GK / 2, &s, 4), memcpy(ch + GK / 2 + 4, &zw, 4);
        }
    for (long m = 0; m < M; m++) /* the reference, products rounded on their own (volatile) */
        for (long n = 0; n < N; n++) {
            float acc = 0.0f;
            for (long g = 0; g < K / GK; g++) {
                const int8_t *ch = W + n * K + g * GK, *a = A + m * K + g * GK;
                float s;
                int32_t zw, p = 0;
                memcpy(&s, ch + GK / 2, 4), memcpy(&zw, ch + GK / 2 + 4, 4);
                for (long i = 0; i < GK / 2; i++) {
                    const int32_t lo = (int32_t)(int8_t)(ch[i] << 4) >> 4, hi = (int32_t)ch[i] >> 4;
                    p += (int32_t)a[2 * i] * (lo - zw) + (int32_t)a[2 * i + 1] * (hi - zw);
                }
                volatile float sp = s * (float)p;
                acc += sp;
            }
            want[m * N + n] = acc;
        }
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw);
    int32_t *Cd = (int32_t *)__Runtime_Alloc(nc * 4);
    memcpy(Ad, A, na);
    memcpy(Wd, W, nw);
    memset(Cd, 0x55, nc * 4);
    Xil_DCacheFlushRange((UINTPTR)Ad, na);
    Xil_DCacheFlushRange((UINTPTR)Wd, nw);
    Xil_DCacheFlushRange((UINTPTR)Cd, nc * 4);
    run(Ad, Wd, Cd);
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc * 4);
    long bad = 0, first = -1;
    for (long i = 0; i < nc; i++)
        if (memcmp(&Cd[i], &want[i], 4) && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld bit-exact", name, nc - bad, nc);
    if (first >= 0) {
        float g;
        memcpy(&g, &Cd[first], 4);
        printf(" (first at %ld: got %.9g, want %.9g)", first, g, want[first]);
    }
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager gemm_wq<M, N, K, GK, TM>: s8 x 4-bit grouped weights on the core, two instances ===\n");
    long bad = wq_case("gemm_wq<128,64,512,32>", 128, 64, 512, 32, wq32_run);
    bad += wq_case("gemm_wq<64,64,1024,128>", 64, 64, 1024, 128, wq128_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
