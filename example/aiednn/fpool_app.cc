/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * fpool_app -- f32 windowed pooling on the AIE (impl fpool:, aiednn_rt_fpool.h, the GEMM ELF's FPOOL mode):
 * each case runs with the default dispatch and with the ref: filter; max must match ref: exactly, an
 * average to 1e-6 ("FPCHK"); times per call in "OPTIME". Deterministic inputs.
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/fpool_app.cc --target x86 --run
 *   ... --target vek280 --run --board <host> --vitis-settings <.../settings64.sh>
 ******************************************************************************/
#include "aiednn.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#if !defined(__aarch64__)
#include <time.h>
#endif

static uint32_t g_lcg = 99u;
static float g_x[64 * 112 * 112];
static float g_out[2][64 * 56 * 56];
static float g_dy[64 * 56 * 56];
static float g_dx[2][64 * 112 * 112];
static int g_fail;

static double now_ms(void) {
#if defined(__aarch64__)
    uint64_t v, f;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return 1e3 * (double)v / (double)f;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e3 * (double)ts.tv_sec + 1e-6 * (double)ts.tv_nsec;
#endif
}

/* nd 4 or 5: dims {N, C, [D,] H, W}; k / s / p per spatial dim */
static void run_case(const char *name, int nd, const int64_t *dims, const int64_t *k, const int64_t *s,
                     const int64_t *p, aiednn_alg_kind_t alg) {
    int64_t od[5] = {dims[0], dims[1], 0, 0, 0}, nin = 1, nout = 1;
    for (int i = 2; i < nd; i++)
        od[i] = (dims[i] + 2 * p[i - 2] - k[i - 2]) / s[i - 2] + 1;
    for (int i = 0; i < nd; i++)
        nin *= dims[i], nout *= od[i];
    for (int64_t i = 0; i < nin; i++)
        g_x[i] = (float)((int32_t)((g_lcg = g_lcg * 1664525u + 1013904223u) >> 8) % 20000) * 1e-3f;
    aiednn_memory_desc_t ms, md;
    aiednn_memory_desc_init(&ms, nd, dims, aiednn_f32);
    aiednn_memory_desc_init(&md, nd, od, aiednn_f32);
    aiednn_pool_nd_params_t pp;
    memset(&pp, 0, sizeof pp);
    for (int i = 0; i < nd - 2; i++)
        pp.kernel[i] = k[i], pp.strides[i] = s[i], pp.padding_l[i] = pp.padding_r[i] = p[i];
    char impl[160] = "?";
    double ms_[2][2] = {{0, 0}, {0, 0}};
    for (int pass = 0; pass < 2; pass++) {
        aiednn_set_impl_filter(pass ? "ref:" : nullptr);
        aiednn_primitive_t prim;
        if (aiednn_pooling_forward_create_nd(&prim, alg, &ms, &md, &pp) != aiednn_success) {
            printf("bad: %s create\n", name);
            g_fail = 1;
            aiednn_set_impl_filter(nullptr);
            return;
        }
        if (!pass)
            snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(prim));
        for (int rep = 0; rep < 2; rep++) {
            const double t0 = now_ms();
            aiednn_exec_arg_t args[2] = {{AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_DST, g_out[pass]}};
            if (aiednn_primitive_execute_args(prim, 2, args) != aiednn_success) {
                printf("bad: %s execute\n", name);
                g_fail = 1;
            }
            ms_[pass][rep] = now_ms() - t0;
        }
        aiednn_primitive_destroy(prim);
    }
    aiednn_set_impl_filter(nullptr);
    double dmax = 0, rmax = 0;
    for (int64_t i = 0; i < nout; i++) {
        dmax = fmax(dmax, fabs((double)g_out[0][i] - g_out[1][i]));
        rmax = fmax(rmax, fabs((double)g_out[1][i]));
    }
    const int ok =
        strncmp(impl, "fpool:", 6) == 0 && (alg == aiednn_pooling_max ? dmax == 0 : dmax <= 1e-6 * (rmax + 1));
    printf("FPCHK %s %s impl=%s maxdiff=%.3g\n", ok ? "ok " : "bad:", name, impl, dmax);
    printf("OPTIME %s aie_first_ms=%.3f aie_ms=%.3f ref_ms=%.3f\n", name, ms_[0][0], ms_[0][1], ms_[1][1]);
    g_fail |= !ok;
}

/* max pooling backward: the AIE finds each window's first maximum; dx must equal ref: bit for bit */
static void run_bwd(const char *name, int nd, const int64_t *dims, const int64_t *k, const int64_t *s,
                    const int64_t *p) {
    int64_t od[5] = {dims[0], dims[1], 0, 0, 0}, nin = 1, nout = 1;
    for (int i = 2; i < nd; i++)
        od[i] = (dims[i] + 2 * p[i - 2] - k[i - 2]) / s[i - 2] + 1;
    for (int i = 0; i < nd; i++)
        nin *= dims[i], nout *= od[i];
    for (int64_t i = 0; i < nin; i++)
        g_x[i] = (float)((int32_t)((g_lcg = g_lcg * 1664525u + 1013904223u) >> 8) % 2000) * 1e-2f;
    for (int64_t i = 0; i < nout; i++)
        g_dy[i] = (float)((int32_t)((g_lcg = g_lcg * 1664525u + 1013904223u) >> 8) % 1000) * 1e-3f;
    aiednn_memory_desc_t mx, my;
    aiednn_memory_desc_init(&mx, nd, dims, aiednn_f32);
    aiednn_memory_desc_init(&my, nd, od, aiednn_f32);
    aiednn_pool_nd_params_t pp;
    memset(&pp, 0, sizeof pp);
    for (int i = 0; i < nd - 2; i++)
        pp.kernel[i] = k[i], pp.strides[i] = s[i], pp.padding_l[i] = pp.padding_r[i] = p[i];
    char impl[160] = "?";
    double ms_[2] = {0, 0};
    for (int pass = 0; pass < 2; pass++) {
        aiednn_set_impl_filter(pass ? "ref:" : nullptr);
        aiednn_primitive_t prim;
        if (aiednn_pooling_backward_create(&prim, aiednn_pooling_max, &mx, &my, &mx, &pp) != aiednn_success) {
            printf("bad: %s create\n", name);
            g_fail = 1;
            aiednn_set_impl_filter(nullptr);
            return;
        }
        if (!pass)
            snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(prim));
        aiednn_exec_arg_t args[3] = {
            {AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_DIFF_DST, g_dy}, {AIEDNN_ARG_DIFF_SRC, g_dx[pass]}};
        for (int rep = 0; rep < 2; rep++) {
            const double t0 = now_ms();
            if (aiednn_primitive_execute_args(prim, 3, args) != aiednn_success) {
                printf("bad: %s execute\n", name);
                g_fail = 1;
            }
            ms_[pass] = now_ms() - t0;
        }
        aiednn_primitive_destroy(prim);
    }
    aiednn_set_impl_filter(nullptr);
    long bad = 0;
    for (int64_t i = 0; i < nin; i++)
        bad += memcmp(&g_dx[0][i], &g_dx[1][i], 4) != 0;
    const int ok = strncmp(impl, "fpool:", 6) == 0 && bad == 0;
    printf("FPCHK %s %s impl=%s mismatches=%ld of %lld\n", ok ? "ok " : "bad:", name, impl, bad, (long long)nin);
    printf("OPTIME %s aie_ms=%.3f ref_ms=%.3f\n", name, ms_[0], ms_[1]);
    g_fail |= !ok;
}

int main() { /* "int main()": aiehlc reads "(void)" as no main */
    const int64_t d3[5] = {1, 32, 8, 28, 28}, k3[3] = {2, 2, 2}, s3[3] = {2, 2, 2}, p3[3] = {0, 0, 0};
    run_case("pool3d_max_32x8x28x28_k2", 5, d3, k3, s3, p3, aiednn_pooling_max);
    run_case("pool3d_avg_32x8x28x28_k2", 5, d3, k3, s3, p3, aiednn_pooling_avg_include_padding);
    const int64_t d2[4] = {1, 64, 112, 112}, k2[2] = {3, 3}, s2[2] = {2, 2}, p2[2] = {1, 1};
    run_case("maxpool_64x112x112_k3s2", 4, d2, k2, s2, p2, aiednn_pooling_max);
    const int64_t d2b[4] = {1, 256, 28, 28}, k2b[2] = {2, 2}, s2b[2] = {2, 2}, p2b[2] = {0, 0};
    run_case("avgpool_256x28x28_k2s2", 4, d2b, k2b, s2b, p2b, aiednn_pooling_avg_exclude_padding);
    run_bwd("maxpool_bwd_64x112x112_k3s2", 4, d2, k2, s2, p2);
    run_bwd("pool3d_max_bwd_32x8x28x28_k2", 5, d3, k3, s3, p3);
    printf(g_fail ? "FAIL: fpool_app\n" : "PASS: fpool_app, every case on fpool: and within tolerance of ref:\n");
    return g_fail;
}
