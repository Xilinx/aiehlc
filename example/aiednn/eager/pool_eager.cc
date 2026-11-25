/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the pool template on the board: two instances of pool<MW, KK, TR, AVG, U8> in one ELF --
 * s8 max over 3x3 windows and u8 average over 2x2 windows with per-window divisors (some windows partly in the
 * padding). Self-checks every code against eager_pool_ref (pool_eager_ref.h).
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "pool_eager_kernel.cc"
#include <stdio.h>
#include <stdlib.h>

#define PMW 512
#define PTR 16
#define PKK_MAX 16 /* 9 taps -> 12 bytes + the divisor word */
#define PKK_AVG 8  /* 4 taps + the divisor word */

static void pool_max_run(int8_t *X, int8_t *Z, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    pool<PMW, PKK_MAX, PTR, 0, 0><<<mesh>>>(X, Z, Y);
}

static void pool_avg_run(int8_t *X, int8_t *Z, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    pool<PMW, PKK_AVG, PTR, 1, 1><<<mesh>>>(X, Z, Y);
}

static uint32_t pl_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/* windows: taps random; for a few windows the trailing taps are padding (max: -128, avg: 0, count smaller) */
static void pool_fill(int8_t *X, long kk, int taps, int avg, uint32_t seed) {
    for (long w = 0; w < PMW; w++) {
        int8_t *row = X + w * kk;
        memset(row, 0, kk);
        const int valid = w % 7 == 3 ? taps / 2 : w % 13 == 5 ? 1 : taps;
        for (int t = 0; t < kk - 4; t++)
            row[t] = t < valid ? (int8_t)(pl_mix((uint32_t)(w * kk + t) + seed) & 0xff) : (avg ? 0 : -128);
        const int32_t div = avg ? (w % 11 == 2 ? taps : valid) : 0; /* include- vs exclude-padding windows */
        memcpy(row + kk - 4, &div, 4);
    }
}

static long pool_case(const char *name, long kk, int taps, int avg, int u8, void (*run)(int8_t *, int8_t *, int8_t *)) {
    const long nx = PMW * kk, ny = PMW * 4;
    int8_t *X = (int8_t *)calloc(nx, 1), *got = (int8_t *)calloc(ny, 1), *want = (int8_t *)calloc(ny, 1);
    pool_fill(X, kk, taps, avg, avg ? 31u : 17u);
    int8_t *Xd = (int8_t *)__Runtime_Alloc(nx), *Yd = (int8_t *)__Runtime_Alloc(ny),
           *Zd = (int8_t *)__Runtime_Alloc(16 * kk * 4);
    memcpy(Xd, X, nx);
    memset(Yd, 0x55, ny);
    Xil_DCacheFlushRange((UINTPTR)Xd, nx);
    Xil_DCacheFlushRange((UINTPTR)Yd, ny);
    memset(Zd, 0, 16 * kk * 4);
    Xil_DCacheFlushRange((UINTPTR)Zd, 16 * kk * 4);
    run(Xd, Zd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ny);
    memcpy(got, Yd, ny);
    const int64_t x_shape[2] = {PMW, kk}, z_shape[2] = {16, 4 * kk}, y_shape[2] = {PMW, 4};
    const aiek_tensor t[3] = {{X, 2, AIEK_INT, 8, 1, x_shape, (uint64_t)nx},
                              {X, 2, AIEK_INT, 8, 1, z_shape, 0},
                              {want, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)ny}};
    eager_pool_ref(t, 3, PMW, kk, avg, u8);
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
    printf("=== aiednn eager pool<%d, KK, %d, AVG, U8>: two instances, s8 max 3x3 and u8 average 2x2 ===\n", PMW, PTR);
    long bad = pool_case("pool max s8 3x3", PKK_MAX, 9, 0, 0, pool_max_run);
    bad += pool_case("pool avg u8 2x2", PKK_AVG, 4, 1, 1, pool_avg_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
