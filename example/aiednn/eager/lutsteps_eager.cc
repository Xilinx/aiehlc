/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the per-row table templates on the board: lutw<R, V, TR, ES> (8-bit codes -> 4- / 2-byte
 * entries) and steps<R, V, TR, S> (f32 / s32 values -> 8-bit codes through step tables), in one ELF. Random
 * tables (steps: ascending thresholds, f32 and s32 rows, NaN inputs); self-checks every byte against the
 * workstation references (lutw_eager_ref.h, steps_eager_ref.h).
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "lutw_eager_kernel.cc"
#include "steps_eager_kernel.cc"
#include <stdio.h>
#include <stdlib.h>

static void lutw4_run(int8_t *X, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    lutw<128, 64, 4, 4><<<mesh>>>(X, Y);
}

static void lutw2_run(int8_t *X, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    lutw<128, 128, 4, 2><<<mesh>>>(X, Y);
}

static void steps_run(int8_t *X, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    steps<128, 64, 2, 3136><<<mesh>>>(X, Y);
}

static uint32_t ls_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static int cmp_i32(const void *a, const void *b) {
    const int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return x < y ? -1 : x > y;
}

/* runs X [R][ld] through run into Y (ny bytes), compares with want */
static long ls_check(const char *name, const int8_t *X, long nx, const int8_t *want, long ny,
                     void (*run)(int8_t *, int8_t *)) {
    int8_t *Xd = (int8_t *)__Runtime_Alloc(nx), *Yd = (int8_t *)__Runtime_Alloc(ny);
    memcpy(Xd, X, nx);
    memset(Yd, 0x55, ny);
    Xil_DCacheFlushRange((UINTPTR)Xd, nx);
    Xil_DCacheFlushRange((UINTPTR)Yd, ny);
    run(Xd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ny);
    long bad = 0, first = -1;
    for (long i = 0; i < ny; i++)
        if (Yd[i] != want[i] && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld exact", name, ny - bad, ny);
    if (first >= 0)
        printf(" (first at %ld: got %d, want %d)", first, Yd[first], want[first]);
    printf("\n");
    return bad;
}

static long lutw_case(const char *name, long R, long V, long ES, void (*run)(int8_t *, int8_t *)) {
    const long ld = 256 * ES + V, nx = R * ld, ny = R * V * ES;
    int8_t *X = (int8_t *)calloc(nx, 1), *want = (int8_t *)calloc(ny, 1);
    for (long i = 0; i < nx; i++)
        X[i] = (int8_t)(ls_mix((uint32_t)i + 17u) & 0xff);
    const int64_t xs[2] = {R, ld}, ys[2] = {R, V * ES};
    const aiek_tensor t[2] = {{X, 2, AIEK_INT, 8, 1, xs, (uint64_t)nx}, {want, 2, AIEK_INT, 8, 1, ys, (uint64_t)ny}};
    eager_lutw_ref(t, 2, R, V, ES);
    return ls_check(name, X, nx, want, ny, run);
}

static long steps_case(const char *name, long R, long V, long S, void (*run)(int8_t *, int8_t *)) {
    const long ld = S + 4 * V, nx = R * ld, ny = R * V;
    int8_t *X = (int8_t *)calloc(nx, 1), *want = (int8_t *)calloc(ny, 1);
    for (long r = 0; r < R; r++) {
        uint8_t *tb = (uint8_t *)X + r * ld;
        const int32_t ns = 1 + (int32_t)(ls_mix((uint32_t)r + 5u) % 511); /* 1 .. 511 steps */
        const int16_t nanc = (int16_t)(ls_mix((uint32_t)r + 6u) & 0xff), kind = (int16_t)(r & 1);
        memcpy(tb, &ns, 4), memcpy(tb + 4, &nanc, 2), memcpy(tb + 6, &kind, 2);
        int32_t *T = (int32_t *)(tb + EG_STEPS_HDR);
        for (int k = 0; k < ns; k++)
            T[k] = (int32_t)ls_mix((uint32_t)(r * 1024 + k) + 7u);
        qsort(T, (size_t)ns, 4, cmp_i32);
        int16_t *P = (int16_t *)(tb + EG_STEPS_HDR + 4 * ns);
        for (int k = 0; k <= ns; k++)
            P[k] = (int16_t)(ls_mix((uint32_t)(r * 1024 + k) + 8u) & 0xff);
        int32_t *val = (int32_t *)(tb + S);
        for (long j = 0; j < V; j++)
            val[j] = j % 13 == 5 ? (int32_t)0x7fc00001 : (int32_t)ls_mix((uint32_t)(r * V + j) + 9u);
    }
    const int64_t xs[2] = {R, ld}, ys[2] = {R, V};
    const aiek_tensor t[2] = {{X, 2, AIEK_INT, 8, 1, xs, (uint64_t)nx}, {want, 2, AIEK_INT, 8, 1, ys, (uint64_t)ny}};
    eager_steps_ref(t, 2, R, V, S);
    return ls_check(name, X, nx, want, ny, run);
}

int main() {
    printf("=== aiednn eager lutw<R, V, TR, ES> / steps<R, V, TR, S>: per-row tables on the core ===\n");
    long bad = lutw_case("lutw<128,64,4,4>", 128, 64, 4, lutw4_run);
    bad += lutw_case("lutw<128,128,4,2>", 128, 128, 2, lutw2_run);
    bad += steps_case("steps<128,64,2,3136>", 128, 64, 3136, steps_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
