/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the f32 row template on the board: frow<R, L, TR, OP> for softmax over 100 of 128 (OP 1),
 * row max (OP 5) and sum / divisor (OP 7, average pooling with a per-window count) in one ELF. Checked against
 * double-precision references with the FVEC contract |got - want| <= 1e-5 + 1e-5 |want|.
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "frow_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void sm_run(int8_t *X, int8_t *P, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    frow<128, 128, 4, 1, 512><<<mesh>>>(X, P, Y);
}

static void mx_run(int8_t *X, int8_t *P, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    frow<128, 64, 8, 5, 16><<<mesh>>>(X, P, Y);
}

static void sd_run(int8_t *X, int8_t *P, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    frow<64, 16, 4, 7, 16><<<mesh>>>(X, P, Y);
}

static uint32_t fr_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static double fr_u(uint32_t h) { return (double)(h & 0xffffff) / 16777216.0; }

/* rows of L, the first Lt real; the reference per OP in double */
static long fr_case(const char *name, int OP, long R, long L, long Lt, void (*run)(int8_t *, int8_t *, int8_t *)) {
    const int red = OP >= 3;
    const long nx = R * L, ny = red ? R * 4 : R * L;
    float *x = (float *)calloc(nx, 4), *got = (float *)calloc(ny, 4);
    double *want = (double *)calloc(ny, 8);
    for (long r = 0; r < R; r++) {
        const double off = (fr_u(fr_mix((uint32_t)r + 1u)) - 0.5) * 10.0;
        for (long k = 0; k < Lt; k++)
            x[r * L + k] = (float)(off + (fr_u(fr_mix((uint32_t)(r * L + k) + 7u)) - 0.5) * 8.0);
        if (OP == 7)
            x[r * L + L - 1] = (float)(1 + r % Lt); /* the window's divisor */
    }
    const int32_t p[16] = {(int32_t)Lt, 0, 0, 0, (int32_t)Lt, 0, 0, 0, (int32_t)Lt, 0, 0, 0, (int32_t)Lt, 0, 0, 0};
    for (long r = 0; r < R; r++) {
        const float *xr = x + r * L;
        double mx = xr[0], s = 0;
        for (long k = 1; k < Lt; k++)
            mx = xr[k] > mx ? xr[k] : mx;
        if (OP == 1) {
            for (long k = 0; k < Lt; k++)
                s += exp((double)xr[k] - mx);
            for (long k = 0; k < L; k++)
                want[r * L + k] = k < Lt ? (double)(float)(exp((double)xr[k] - mx) / s) : 0.0;
        } else if (OP == 5) {
            want[r * 4] = mx;
        } else {
            for (long k = 0; k < Lt; k++)
                s += xr[k];
            want[r * 4] = (double)(float)(s / xr[L - 1]);
        }
    }
    int8_t *Xd = (int8_t *)__Runtime_Alloc(nx * 4), *Pd = (int8_t *)__Runtime_Alloc(64),
           *Yd = (int8_t *)__Runtime_Alloc(ny * 4);
    memcpy(Xd, x, nx * 4);
    memcpy(Pd, p, 64);
    memset(Yd, 0x55, ny * 4);
    Xil_DCacheFlushRange((UINTPTR)Xd, nx * 4);
    Xil_DCacheFlushRange((UINTPTR)Pd, 64);
    Xil_DCacheFlushRange((UINTPTR)Yd, ny * 4);
    run(Xd, Pd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ny * 4);
    memcpy(got, Yd, ny * 4);
    long bad = 0, first = -1, n = 0;
    double worst = 0;
    for (long i = 0; i < ny; i++) {
        if (red && i % 4)
            continue;
        n++;
        const double d = fabs((double)got[i] - want[i]), lim = 1e-5 + 1e-5 * fabs(want[i]);
        if (d / lim > worst)
            worst = d / lim;
        if (!(d <= lim) && bad++ == 0)
            first = i;
    }
    printf("[%s] check: %ld / %ld within contract, worst %.3f of the bound", name, n - bad, n, worst);
    if (first >= 0)
        printf(" (first at %ld: got %.9g, want %.9g)", first, got[first], want[first]);
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager frow<R, L, TR, OP>: f32 row operations on the core, three instances ===\n");
    long bad = fr_case("frow<128,128,4,SOFTMAX> 100 of 128", 1, 128, 128, 100, sm_run);
    bad += fr_case("frow<128,64,8,MAX>", 5, 128, 64, 64, mx_run);
    bad += fr_case("frow<64,16,4,SUMDIV> 9 taps", 7, 64, 16, 9, sd_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
