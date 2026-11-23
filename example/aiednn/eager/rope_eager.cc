/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the rope template on the board: two instances of rope<R, D, TR, INTER> in one ELF (half-split
 * and interleaved, D 64). Self-checks every float BYTE against eager_rope_ref (computed here on the A72 without
 * fused multiply-add): IEEE single multiplies and adds on both sides.
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "rope_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void rope_half_run(int8_t *X, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    rope<128, 64, 8, 0><<<mesh>>>(X, Y);
}

static void rope_inter_run(int8_t *X, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    rope<128, 64, 8, 1><<<mesh>>>(X, Y);
}

static uint32_t rp_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static long rope_case(const char *name, int inter, void (*run)(int8_t *, int8_t *)) {
    const long R = 128, D = 64, nx = R * 12 * D, ny = R * 4 * D;
    float *X = (float *)calloc(R * 3 * D, 4), *got = (float *)calloc(R * D, 4), *want = (float *)calloc(R * D, 4);
    for (long r = 0; r < R; r++)
        for (long j = 0; j < D; j++) {
            const double ang = ((double)(rp_mix((uint32_t)(r * D + j) + 3u) & 0xffff) / 65536.0 - 0.5) * 6.0;
            X[r * 3 * D + j] =
                (float)(((double)(rp_mix((uint32_t)(r * D + j) + 8u) & 0xffffff) / 16777216.0 - 0.5) * 8.0);
            X[r * 3 * D + D + j] = (float)cos(ang);
            X[r * 3 * D + 2 * D + j] = (float)sin(ang);
        }
    int8_t *Xd = (int8_t *)__Runtime_Alloc(nx), *Yd = (int8_t *)__Runtime_Alloc(ny);
    memcpy(Xd, X, nx);
    memset(Yd, 0x55, ny);
    Xil_DCacheFlushRange((UINTPTR)Xd, nx);
    Xil_DCacheFlushRange((UINTPTR)Yd, ny);
    run(Xd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ny);
    memcpy(got, Yd, ny);
    const int64_t x_shape[2] = {R, 12 * D}, y_shape[2] = {R, 4 * D};
    const aiek_tensor t[2] = {{X, 2, AIEK_INT, 8, 1, x_shape, (uint64_t)nx},
                              {want, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)ny}};
    eager_rope_ref(t, 2, R, D, inter);
    long bad = 0, first = -1;
    for (long i = 0; i < R * D; i++)
        if (memcmp(&got[i], &want[i], 4) && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld bit-identical", name, R * D - bad, R * D);
    if (first >= 0)
        printf(" (first at %ld: got %.9g, want %.9g)", first, got[first], want[first]);
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager rope<R, D, TR, INTER>: rotary position embedding on the core, two instances ===\n");
    long bad = rope_case("rope<128,64,8,0> half-split", 0, rope_half_run);
    bad += rope_case("rope<128,64,8,1> interleaved", 1, rope_inter_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
