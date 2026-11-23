/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the f32 pointwise templates on the board: feltw<R, L, TR, F> for GELU (erf) and log, and
 * fbinary<R, L, TR, F> for division, in one ELF. Checked against double-precision references with the FVEC
 * contract |got - want| <= 1e-5 + 1e-5 |want| (the core computes in single precision). F as numbers (the EG_F_* codes:
 * GELU erf 18, log 13, div 67): the instance name carries the template arguments.
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "feltw_eager_kernel.cc"
#include "fbinary_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void gelu_run(int8_t *X, int8_t *P, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    feltw<128, 256, 8, 18><<<mesh>>>(X, P, Y);
}

static void log_run(int8_t *X, int8_t *P, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    feltw<64, 128, 8, 13><<<mesh>>>(X, P, Y);
}

static void div_run(int8_t *XY, int8_t *Z) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    fbinary<128, 256, 4, 67><<<mesh>>>(XY, Z);
}

static uint32_t fe_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static double fe_u(uint32_t h) { return (double)(h & 0xffffff) / 16777216.0; }

static long fe_check(const char *name, const float *got, const double *want, long n) {
    long bad = 0, first = -1;
    double worst = 0;
    for (long i = 0; i < n; i++) {
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

/* one unary instance: R x L values from gen(i), the reference fn */
static long fe_unary(const char *name, long R, long L, double lo, double hi, double (*fn)(double),
                     void (*run)(int8_t *, int8_t *, int8_t *)) {
    const long n = R * L;
    float *x = (float *)malloc(n * 4), *got = (float *)malloc(n * 4);
    double *want = (double *)malloc(n * 8);
    for (long i = 0; i < n; i++) {
        x[i] = (float)(lo + (hi - lo) * fe_u(fe_mix((uint32_t)i + 17u)));
        want[i] = (double)(float)fn((double)x[i]);
    }
    const float ab[16] = {0};
    int8_t *Xd = (int8_t *)__Runtime_Alloc(n * 4), *Pd = (int8_t *)__Runtime_Alloc(64),
           *Yd = (int8_t *)__Runtime_Alloc(n * 4);
    memcpy(Xd, x, n * 4);
    memcpy(Pd, ab, 64);
    memset(Yd, 0x55, n * 4);
    Xil_DCacheFlushRange((UINTPTR)Xd, n * 4);
    Xil_DCacheFlushRange((UINTPTR)Pd, 64);
    Xil_DCacheFlushRange((UINTPTR)Yd, n * 4);
    run(Xd, Pd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, n * 4);
    memcpy(got, Yd, n * 4);
    return fe_check(name, got, want, n);
}

static double gelu_erf_d(double x) { return 0.5 * x * (1.0 + erf(x / sqrt(2.0))); }
static double log_d(double x) { return log(x); }

static long fe_div(void) {
    const long R = 128, L = 256, n = R * L;
    float *x = (float *)malloc(n * 4), *y = (float *)malloc(n * 4), *got = (float *)malloc(n * 4);
    double *want = (double *)malloc(n * 8);
    for (long i = 0; i < n; i++) {
        x[i] = (float)((fe_u(fe_mix((uint32_t)i + 3u)) - 0.5) * 20.0);
        const double m = 0.1 + 10.0 * fe_u(fe_mix((uint32_t)i + 9u));
        y[i] = (float)((fe_mix((uint32_t)i + 5u) & 1) ? m : -m);
        want[i] = (double)(float)((double)x[i] / (double)y[i]);
    }
    float *XYd = (float *)__Runtime_Alloc(n * 8); /* each row: its L x, then its L y */
    int8_t *Zd = (int8_t *)__Runtime_Alloc(n * 4);
    for (long r = 0; r < R; r++) {
        memcpy(XYd + r * 2 * L, x + r * L, L * 4);
        memcpy(XYd + r * 2 * L + L, y + r * L, L * 4);
    }
    memset(Zd, 0x55, n * 4);
    Xil_DCacheFlushRange((UINTPTR)XYd, n * 8);
    Xil_DCacheFlushRange((UINTPTR)Zd, n * 4);
    div_run((int8_t *)XYd, Zd);
    Xil_DCacheInvalidateRange((UINTPTR)Zd, n * 4);
    memcpy(got, Zd, n * 4);
    return fe_check("fbinary<128,256,4,DIV>", got, want, n);
}

int main() {
    printf("=== aiednn eager f32 pointwise: feltw (GELU erf, log) and fbinary (div) on the core ===\n");
    long bad = fe_unary("feltw<128,256,8,GELU_ERF>", 128, 256, -6.0, 6.0, gelu_erf_d, gelu_run);
    bad += fe_unary("feltw<64,128,8,LOG>", 64, 128, 1e-3, 1e3, log_d, log_run);
    bad += fe_div();
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
