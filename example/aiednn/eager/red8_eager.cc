/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the 8-bit reduction template on the board: red8<R, L, TR, OP, SG> for the s8 sum (OP 1), the
 * u8 max (OP 2) and the s8 sum of squares (OP 5), in one ELF. Random rows; every row's int32 checked exactly against
 * eager_red8_ref (red8_eager_ref.h).
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "red8_eager_kernel.cc"
#include <stdio.h>
#include <stdlib.h>

static void sum_run(int8_t *X, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    red8<128, 1024, 4, 1, 1><<<mesh>>>(X, Y);
}

static void max_run(int8_t *X, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    red8<128, 256, 8, 2, 0><<<mesh>>>(X, Y);
}

static void sq_run(int8_t *X, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    red8<64, 64, 4, 5, 1><<<mesh>>>(X, Y);
}

static uint32_t r8_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static long r8_case(const char *name, long R, long L, int OP, int SG, void (*run)(int8_t *, int8_t *)) {
    const long nx = R * L, ny = R * 16;
    int8_t *X = (int8_t *)calloc(nx, 1), *want = (int8_t *)calloc(ny, 1);
    for (long i = 0; i < nx; i++)
        X[i] = (int8_t)(r8_mix((uint32_t)i + 23u) & 0xff);
    const int64_t xs[2] = {R, L}, ys[2] = {R, 16};
    const aiek_tensor t[2] = {{X, 2, AIEK_INT, 8, 1, xs, (uint64_t)nx}, {want, 2, AIEK_INT, 8, 1, ys, (uint64_t)ny}};
    eager_red8_ref(t, 2, R, L, OP, SG);
    int8_t *Xd = (int8_t *)__Runtime_Alloc(nx), *Yd = (int8_t *)__Runtime_Alloc(ny);
    memcpy(Xd, X, nx);
    memset(Yd, 0x55, ny);
    Xil_DCacheFlushRange((UINTPTR)Xd, nx);
    Xil_DCacheFlushRange((UINTPTR)Yd, ny);
    run(Xd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ny);
    long bad = 0, first = -1;
    for (long r = 0; r < R; r++)
        if (memcmp(Yd + r * 16, want + r * 16, 4) && bad++ == 0)
            first = r;
    printf("[%s] check: %ld / %ld rows exact", name, R - bad, R);
    if (first >= 0) {
        int32_t g, w;
        memcpy(&g, Yd + first * 16, 4), memcpy(&w, want + first * 16, 4);
        printf(" (first at row %ld: got %d, want %d)", first, (int)g, (int)w);
    }
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager red8<R, L, TR, OP, SG>: exact 8-bit reductions on the core, three instances ===\n");
    long bad = r8_case("red8<128,1024,4,SUM,s8>", 128, 1024, 1, 1, sum_run);
    bad += r8_case("red8<128,256,8,MAX,u8>", 128, 256, 2, 0, max_run);
    bad += r8_case("red8<64,64,4,SUMSQ,s8>", 64, 64, 5, 1, sq_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
