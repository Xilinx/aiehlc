/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the lut template on the board: two instances of lut<M, N, TM> in one ELF (64 and 128
 * tables). Random tables and codes; self-checks every byte against eager_lut_ref (lut_eager_ref.h).
 ******************************************************************************/
#include "lut_eager_kernel.cc"
#include <stdio.h>
#include <stdlib.h>

static void lut64_run(int8_t *A, int8_t *W, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    lut<256, 64, 16><<<mesh>>>(A, W, Y);
}

static void lut128_run(int8_t *A, int8_t *W, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    lut<128, 128, 32><<<mesh>>>(A, W, Y);
}

static uint32_t lt_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static long lut_case(const char *name, long M, long N, void (*run)(int8_t *, int8_t *, int8_t *)) {
    const long ld = EG_LUT_TABLE + N, na = M * ld, nw = N * ld, ny = M * N;
    int8_t *A = (int8_t *)calloc(na, 1), *W = (int8_t *)calloc(nw, 1), *got = (int8_t *)calloc(ny, 1);
    int8_t *want = (int8_t *)calloc(ny, 1);
    for (long m = 0; m < M; m++)
        for (long n = 0; n < N; n++)
            A[m * ld + EG_LUT_TABLE + n] = (int8_t)(lt_mix((uint32_t)(m * N + n) + 3u) & 0xff);
    for (long n = 0; n < N; n++)
        for (long v = 0; v < 256; v++)
            W[n * ld + v] = (int8_t)(lt_mix((uint32_t)(n * 256 + v) + 99u) & 0xff);
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw),
           *Yd = (int8_t *)__Runtime_Alloc(ny);
    memcpy(Ad, A, na);
    memcpy(Wd, W, nw);
    memset(Yd, 0x55, ny);
    Xil_DCacheFlushRange((UINTPTR)Ad, na);
    Xil_DCacheFlushRange((UINTPTR)Wd, nw);
    Xil_DCacheFlushRange((UINTPTR)Yd, ny);
    run(Ad, Wd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ny);
    memcpy(got, Yd, ny);
    const int64_t a_shape[2] = {M, ld}, w_shape[2] = {N, ld}, y_shape[2] = {M, N};
    const aiek_tensor t[3] = {{A, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)na},
                              {W, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)nw},
                              {want, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)ny}};
    eager_lut_ref(t, 3, M, N);
    long bad = 0, first = -1;
    for (long i = 0; i < ny; i++)
        if (got[i] != want[i] && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld exact", name, ny - bad, ny);
    if (first >= 0)
        printf(" (first at %ld: got %d, want %d)", first, got[first], want[first]);
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager lut<M, N, TM>: per-column 256-entry tables on the core, two instances ===\n");
    long bad = lut_case("lut<256,64>", 256, 64, lut64_run);
    bad += lut_case("lut<128,128>", 128, 128, lut128_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
