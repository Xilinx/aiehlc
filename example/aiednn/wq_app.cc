/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * wq_app -- grouped weight-only quantization on the AIE (impl wq:, aiednn_rt_wq.h, the GEMM ELF's WQ
 * mode): 8-bit activations x 4- / 8-bit weights with per-group scales (and zero points), the LLM linear
 * layer. Each case runs with the default dispatch and with the ref: filter; the f32 results must agree to
 * 1e-5 of the output's magnitude ("WQCHK"), times per call in "OPTIME". Deterministic inputs.
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/wq_app.cc --target x86 --run
 *   ... --target vek280 --run --board <host> --vitis-settings <.../settings64.sh>
 ******************************************************************************/
#include "aiednn.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#if !defined(__aarch64__)
#include <time.h>
#endif

static uint32_t g_lcg = 4242u;
static uint32_t lcg(void) { return g_lcg = g_lcg * 1664525u + 1013904223u; }
static int g_fail;

static int8_t g_src[128 * 4096];
static uint8_t g_wei[4096 * 1024];
static float g_sc[64 * 1024];
static uint8_t g_zp[64 * 1024];
static float g_out[2][128 * 1024];

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

static aiednn_memory_desc_t D2(int64_t a, int64_t b, aiednn_data_type_t dt) {
    aiednn_memory_desc_t m;
    const int64_t d[2] = {a, b};
    aiednn_memory_desc_init(&m, 2, d, dt);
    return m;
}

static int bits(aiednn_data_type_t t) { return t == aiednn_s4 || t == aiednn_u4 ? 4 : 8; }

/* y [M, N] f32 = src [M, K] x dequant(weights [K, N]) with groups of g along K */
static void run_case(const char *name, int64_t M, int64_t K, int64_t N, int64_t g, aiednn_data_type_t sdt,
                     aiednn_data_type_t wdt, aiednn_data_type_t zdt, int32_t src_zp) {
    const int64_t G = K / g;
    for (int64_t i = 0; i < M * K; i++)
        g_src[i] = (int8_t)(lcg() >> 24);
    for (int64_t i = 0; i < K * N * bits(wdt) / 8; i++)
        g_wei[i] = (uint8_t)(lcg() >> 24);
    for (int64_t i = 0; i < G * N; i++)
        g_sc[i] = 0.002f + (float)(lcg() >> 20) * 1e-7f;
    for (int64_t i = 0; i < G * N * (zdt ? bits(zdt) : 8) / 8; i++)
        g_zp[i] = (uint8_t)(zdt == aiednn_u8 ? 120 + (lcg() >> 29) : (lcg() >> 24));
    aiednn_memory_desc_t s = D2(M, K, sdt), w = D2(K, N, wdt), d = D2(M, N, aiednn_f32);
    aiednn_attr_t at;
    memset(&at, 0, sizeof at);
    at.src_zero_point = src_zp;
    at.wei_scales = D2(G, N, aiednn_f32);
    if (zdt)
        at.wei_zero_points = D2(G, N, zdt);
    char impl[160] = "?";
    double ms[2][2] = {{0, 0}, {0, 0}};
    for (int pass = 0; pass < 2; pass++) {
        aiednn_set_impl_filter(pass ? "ref:" : nullptr);
        aiednn_primitive_t p;
        aiednn_status_t st = aiednn_matmul_forward_create_attr(&p, &s, &w, nullptr, &d, 0, &at);
        if (st != aiednn_success) {
            printf("bad: %s create: %s\n", name, aiednn_status_str(st));
            g_fail = 1;
            aiednn_set_impl_filter(nullptr);
            return;
        }
        aiednn_primitive_set_constant_weights(p, 1); /* model weights: packed once */
        if (!pass)
            snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(p));
        aiednn_exec_arg_t args[5] = {{AIEDNN_ARG_SRC, g_src},
                                     {AIEDNN_ARG_WEIGHTS, g_wei},
                                     {AIEDNN_ARG_DST, g_out[pass]},
                                     {AIEDNN_ARG_ATTR_WEI_SCALES, g_sc},
                                     {AIEDNN_ARG_ATTR_WEI_ZERO_POINTS, g_zp}};
        for (int rep = 0; rep < 2; rep++) {
            const double t0 = now_ms();
            if (aiednn_primitive_execute_args(p, zdt ? 5 : 4, args) != aiednn_success) {
                printf("bad: %s execute\n", name);
                g_fail = 1;
            }
            ms[pass][rep] = now_ms() - t0;
        }
        aiednn_primitive_destroy(p);
    }
    aiednn_set_impl_filter(nullptr);
    double dmax = 0, rmax = 0;
    for (int64_t i = 0; i < M * N; i++) {
        dmax = fmax(dmax, fabs((double)g_out[0][i] - g_out[1][i]));
        rmax = fmax(rmax, fabs((double)g_out[1][i]));
    }
    const int ok = dmax <= 1e-5 * (rmax + 1e-30) && strncmp(impl, "wq:", 3) == 0;
    printf("WQCHK %s %s impl=%s maxdiff=%.3g of max|ref| %.3g\n", ok ? "ok " : "bad:", name, impl, dmax, rmax);
    printf("OPTIME %s aie_first_ms=%.3f aie_ms=%.3f ref_ms=%.3f\n", name, ms[0][0], ms[0][1], ms[1][1]);
    g_fail |= !ok;
}

int main() { // "int main()": aiehlc reads "(void)" as no main and archives a library
    /* decode: one token through a 4096 -> 512 projection, int4 weights, group 128, zero points */
    run_case("decode_s8xs4_g128_zp", 1, 4096, 512, 128, aiednn_s8, aiednn_s4, aiednn_s8, 0);
    /* prefill: 64 tokens, u8 activations (zero point 113), u4 weights + u4 zero points */
    run_case("prefill_u8xu4_g128_zp", 64, 2048, 512, 128, aiednn_u8, aiednn_u4, aiednn_u4, 113);
    /* int8 weights, group 64, no zero points, two N blocks */
    run_case("s8xs8_g64", 128, 1024, 1024, 64, aiednn_s8, aiednn_s8, (aiednn_data_type_t)0, 0);
    printf(g_fail ? "FAIL: wq_app\n" : "PASS: wq_app, every case on wq: and within 1e-5 of ref:\n");
    return g_fail;
}
