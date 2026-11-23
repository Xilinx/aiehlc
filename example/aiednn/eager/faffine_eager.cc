/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, f32 per-channel ops on the board: faffine<R, L, TR, F> for inference batch norm (affine, F 96),
 * batch norm + ReLU (97) and PReLU with a per-channel slope (98), and fbinary<R, L, TR, 98> for PReLU with a
 * per-element slope, in one ELF, each with its operands packed into one input row. Checked against double-precision
 *references with the FVEC contract |got - want| <= 1e-5 + 1e-5 |want|.
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "faffine_eager_kernel.cc"
#include "fbinary_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void bn_run(int8_t *XC, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    faffine<128, 64, 8, 96><<<mesh>>>(XC, Y);
}

static void bnr_run(int8_t *XC, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    faffine<128, 64, 8, 97><<<mesh>>>(XC, Y);
}

static void pr_run(int8_t *XC, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    faffine<64, 128, 4, 98><<<mesh>>>(XC, Y);
}

static void pre_run(int8_t *XC, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    fbinary<128, 256, 4, 98><<<mesh>>>(XC, Y);
}

static uint32_t fa_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static double fa_u(uint32_t h) { return (double)(h & 0xffffff) / 16777216.0; }

/* F 96 / 97 / 98 per row (elem 0: constants per row), or 98 per element (elem 1: the second operand [R][L]) */
static long fa_case(const char *name, int F, int elem, long R, long L, void (*run)(int8_t *, int8_t *)) {
    const long n = R * L, nc = elem ? n : R * 4;
    float *x = (float *)calloc(n, 4), *c = (float *)calloc(nc, 4), *got = (float *)calloc(n, 4);
    double *want = (double *)calloc(n, 8);
    for (long i = 0; i < n; i++)
        x[i] = (float)((fa_u(fa_mix((uint32_t)i + 3u)) - 0.5) * 12.0);
    for (long i = 0; i < nc; i++)
        c[i] = (float)((fa_u(fa_mix((uint32_t)i + 99991u)) - 0.3) * 3.0);
    for (long i = 0; i < n; i++) {
        const double c0 = elem ? c[i] : c[i / L * 4], c1 = elem ? 0.0 : c[i / L * 4 + 1], v = x[i];
        const double a = v * c0 + c1;
        want[i] = F == 96 ? a : F == 97 ? (a > 0 ? a : 0) : (v > 0 ? v : v * c0);
    }
    /* one packed input: faffine rows [x (L) | c0 c1 0 0], fbinary rows [x (L) | slope (L)] */
    const long ld = elem ? 2 * L : L + 4;
    float *XCd = (float *)__Runtime_Alloc(R * ld * 4);
    memset(XCd, 0, R * ld * 4);
    for (long r = 0; r < R; r++) {
        memcpy(XCd + r * ld, x + r * L, L * 4);
        memcpy(XCd + r * ld + L, c + (elem ? r * L : r * 4), (elem ? L : 2) * 4);
    }
    int8_t *Yd = (int8_t *)__Runtime_Alloc(n * 4);
    memset(Yd, 0x55, n * 4);
    Xil_DCacheFlushRange((UINTPTR)XCd, R * ld * 4);
    Xil_DCacheFlushRange((UINTPTR)Yd, n * 4);
    run((int8_t *)XCd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, n * 4);
    memcpy(got, Yd, n * 4);
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

int main() {
    printf("=== aiednn eager faffine<R, L, TR, F> / fbinary prelu: f32 per-channel ops on the core ===\n");
    long bad = fa_case("faffine<128,64,8,AFFINE> batch norm", 96, 0, 128, 64, bn_run);
    bad += fa_case("faffine<128,64,8,AFFINE_RELU> batch norm + relu", 97, 0, 128, 64, bnr_run);
    bad += fa_case("faffine<64,128,4,PRELU> per-channel slope", 98, 0, 64, 128, pr_run);
    bad += fa_case("fbinary<128,256,4,PRELU> per-element slope", 98, 1, 128, 256, pre_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
