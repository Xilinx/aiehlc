/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the lnorm template on the board -- the first row op: two instances of lnorm<R, L, TR, RMS> in
 * one ELF (layer norm over 768 f32, RMS norm over 200 of 256). Checked against a double-precision reference with
 * the fvec: contract |got - want| <= 1e-5 + 1e-5 |want| (the core computes in single precision).
 ******************************************************************************/
#include "lnorm_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void ln768_run(int8_t *X, int8_t *P, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    lnorm<128, 768, 2, 0><<<mesh>>>(X, P, Y);
}

static void rms256_run(int8_t *X, int8_t *P, int8_t *Y) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    lnorm<128, 256, 8, 1><<<mesh>>>(X, P, Y);
}

static uint32_t ln_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static double ln_u(uint32_t h) { return (double)(h & 0xffffff) / 16777216.0; }

static long ln_case(const char *name, long R, long L, long Lt, int rms, void (*run)(int8_t *, int8_t *, int8_t *)) {
    const long nx = R * 4 * L, np = 12 * L, qc = L / 4;
    float *x = (float *)calloc(R * L, 4), *got = (float *)calloc(R * L, 4);
    float *gam = (float *)calloc(L, 4), *bet = (float *)calloc(L, 4);
    uint8_t *P = (uint8_t *)calloc(np, 1);
    for (long r = 0; r < R; r++) { /* rows with an offset and a scale of their own, like activations */
        const double off = (ln_u(ln_mix((uint32_t)r + 5u)) - 0.5) * 20.0,
                     sc = 0.1 + 5.0 * ln_u(ln_mix((uint32_t)r + 9u));
        for (long k = 0; k < Lt; k++)
            x[r * L + k] = (float)(off + sc * (ln_u(ln_mix((uint32_t)(r * L + k) + 1u)) - 0.5));
    }
    for (long k = 0; k < L; k++) {
        gam[k] = (float)(0.5 + ln_u(ln_mix((uint32_t)k + 3u)));
        bet[k] = (float)(ln_u(ln_mix((uint32_t)k + 4u)) - 0.5);
    }
    const float eps = 1e-5f, rl = 1.0f / (float)Lt;
    const int32_t lt = (int32_t)Lt;
    for (long c = 0; c < 4; c++) { /* pre-tiled per core: its gamma, beta, params rows */
        memcpy(P + 3 * c * L, gam + c * qc, L);
        memcpy(P + (3 * c + 1) * L, bet + c * qc, L);
        memcpy(P + (3 * c + 2) * L, &eps, 4);
        memcpy(P + (3 * c + 2) * L + 4, &rl, 4);
        memcpy(P + (3 * c + 2) * L + 8, &lt, 4);
    }
    int8_t *Xd = (int8_t *)__Runtime_Alloc(nx), *Pd = (int8_t *)__Runtime_Alloc(np),
           *Yd = (int8_t *)__Runtime_Alloc(nx);
    memcpy(Xd, x, nx);
    memcpy(Pd, P, np);
    memset(Yd, 0x55, nx);
    Xil_DCacheFlushRange((UINTPTR)Xd, nx);
    Xil_DCacheFlushRange((UINTPTR)Pd, np);
    Xil_DCacheFlushRange((UINTPTR)Yd, nx);
    run(Xd, Pd, Yd);
    Xil_DCacheInvalidateRange((UINTPTR)Yd, nx);
    memcpy(got, Yd, nx);
    long bad = 0, first = -1;
    double worst = 0;
    for (long r = 0; r < R; r++) { /* ref:'s double arithmetic */
        double mean = 0, var = 0;
        if (!rms) {
            for (long k = 0; k < Lt; k++)
                mean += x[r * L + k];
            mean /= (double)Lt;
        }
        for (long k = 0; k < Lt; k++)
            var += ((double)x[r * L + k] - mean) * ((double)x[r * L + k] - mean);
        const double inv = 1.0 / sqrt(var / (double)Lt + (double)eps);
        for (long k = 0; k < L; k++) {
            const double want =
                k < Lt ? (double)(float)(gam[k] * (((double)x[r * L + k] - mean) * inv) + (rms ? 0 : bet[k])) : 0.0;
            const double d = fabs((double)got[r * L + k] - want), lim = 1e-5 + 1e-5 * fabs(want);
            if (d / lim > worst)
                worst = d / lim;
            if (!(d <= lim) && bad++ == 0)
                first = r * L + k;
        }
    }
    printf("[%s] check: %ld / %ld within contract, worst %.3f of the bound", name, R * L - bad, R * L, worst);
    if (first >= 0)
        printf(" (first at %ld: got %.9g)", first, got[first]);
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager lnorm<R, L, TR, RMS>: layer / RMS norm rows on the core, two instances ===\n");
    long bad = ln_case("lnorm<128,768,2,0> layer norm", 128, 768, 768, 0, ln768_run);
    bad += ln_case("lnorm<128,256,8,1> RMS 200 of 256", 128, 256, 200, 1, rms256_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
