/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the int32 depthwise template on the board: two instances of dwconv_s8<M, C, TP, TM> in one ELF
 * (3x3 and 5x5 taps). Random codes and taps; every int32 sum checked exactly against eager_dwconv_s8_ref.
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "dwconv_s8_eager_kernel.cc"
#include <stdio.h>
#include <stdlib.h>

static void dw3_run(int8_t *A, int8_t *W, int32_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    dwconv_s8<128, 64, 12, 16><<<mesh>>>(A, W, Y);
}

static void dw5_run(int8_t *A, int8_t *W, int32_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    dwconv_s8<64, 32, 28, 16><<<mesh>>>(A, W, Y);
}

static uint32_t ds_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static long ds_case(const char *name, long M, long C, long TP, int taps, void (*run)(int8_t *, int8_t *, int32_t *)) {
    const long TK = 4 * TP, K = C / 4 * TK, na = M * K, nw = C * K, ny = M * C;
    int8_t *A = (int8_t *)calloc(na, 1), *W = (int8_t *)calloc(nw, 1);
    int32_t *want = (int32_t *)calloc(ny, 4);
    for (long m = 0; m < M; m++)
        for (long c = 0; c < C; c++)
            for (int t = 0; t < taps; t++)
                A[m * K + c / 4 * TK + c % 4 * TP + t] = (int8_t)(ds_mix((uint32_t)((m * C + c) * 64 + t) + 3u) & 0xff);
    for (long c = 0; c < C; c++)
        for (int t = 0; t < taps; t++)
            W[c * K + c / 4 * TK + t] = (int8_t)(ds_mix((uint32_t)(c * 64 + t) + 77u) & 0xff);
    const int64_t as[2] = {M, K}, ws[2] = {C, K}, ys[2] = {M, C};
    const aiek_tensor t[3] = {{A, 2, AIEK_INT, 8, 1, as, (uint64_t)na},
                              {W, 2, AIEK_INT, 8, 1, ws, (uint64_t)nw},
                              {want, 2, AIEK_INT, 32, 1, ys, (uint64_t)ny * 4}};
    eager_dwconv_s8_ref(t, 3, M, C, TP);
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw);
    int32_t *Yd = (int32_t *)__Runtime_Alloc(ny * 4);
    memcpy(Ad, A, na);
    memcpy(Wd, W, nw);
    memset(Yd, 0x55, ny * 4);
    Xil_DCacheFlushRange((UINTPTR)Ad, na);
    Xil_DCacheFlushRange((UINTPTR)Wd, nw);
    Xil_DCacheFlushRange((UINTPTR)Yd, ny * 4);
    run(Ad, Wd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ny * 4);
    long bad = 0, first = -1;
    for (long i = 0; i < ny; i++)
        if (Yd[i] != want[i] && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld exact", name, ny - bad, ny);
    if (first >= 0)
        printf(" (first at %ld: got %d, want %d)", first, (int)Yd[first], (int)want[first]);
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager dwconv_s8<M, C, TP, TM>: depthwise int32 sums on the core, two instances ===\n");
    long bad = ds_case("dwconv_s8<128,64,12> 3x3", 128, 64, 12, 9, dw3_run);
    bad += ds_case("dwconv_s8<64,32,28> 5x5", 64, 32, 28, 25, dw5_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
