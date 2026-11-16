/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * convprof_app -- per-shape time breakdown of an int8 conv (requant + ReLU) on the board.
 * AIEDNN_KCYC=1 adds core (0, 0)'s cycles per GEMMQ launch.
 *
 *   AIEDNN_DEFINES="AIEDNN_KCYC=1" PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/convprof_app.cc \
 *       --target vek280 --mesh 8x4 --run --board <host> --vitis-settings <.../settings64.sh>
 ******************************************************************************/
#include "aiednn.h"

#include <stdio.h>
#include <string.h>
#if !defined(__aarch64__)
#include <time.h>
#endif

#define REPS 5
static int8_t g_x[64 * 230 * 230], g_w[512 * 512 * 9], g_y[64 * 112 * 112], g_ref[64 * 112 * 112];
static int32_t g_bias[512];
static float g_scales[512];

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

static aiednn_memory_desc_t D(int nd, int64_t a, int64_t b, int64_t c, int64_t d, aiednn_data_type_t dt) {
    aiednn_memory_desc_t m;
    const int64_t dims[4] = {a, b, c, d};
    aiednn_memory_desc_init(&m, nd, dims, dt);
    return m;
}

/* one ResNet-18 conv shape: c -> o channels, h x h input, k x k kernel, stride s */
static int conv(const char *name, int64_t c, int64_t h, int64_t o, int64_t k, int64_t s) {
    const int64_t ho = (h + 2 * (k / 2) - k) / s + 1;
    aiednn_memory_desc_t sd = D(4, 1, c, h, h, aiednn_s8), wd = D(4, o, c, k, k, aiednn_s8);
    aiednn_memory_desc_t bd = D(1, o, 0, 0, 0, aiednn_s32), dd = D(4, 1, o, ho, ho, aiednn_s8);
    aiednn_conv_params_t cp;
    memset(&cp, 0, sizeof cp);
    cp.strides[0] = cp.strides[1] = s, cp.groups = 1;
    cp.padding_l[0] = cp.padding_l[1] = cp.padding_r[0] = cp.padding_r[1] = k / 2;
    aiednn_attr_t at;
    memset(&at, 0, sizeof at);
    at.n_scales = o, at.scales = g_scales, at.dst_scale = 0.05f;
    at.n_post_ops = 1, at.post_ops[0].kind = aiednn_post_op_eltwise, at.post_ops[0].alg = aiednn_eltwise_relu;
    const long nout = (long)o * ho * ho;
    { /* ref: for the byte check */
        aiednn_set_impl_filter("ref:");
        aiednn_primitive_t r;
        aiednn_exec_arg_t ra[4] = {
            {AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_WEIGHTS, g_w}, {AIEDNN_ARG_BIAS, g_bias}, {AIEDNN_ARG_DST, g_ref}};
        if (aiednn_convolution_forward_create_attr(&r, &sd, &wd, &bd, &dd, &cp, &at) != aiednn_success ||
            aiednn_primitive_execute_args(r, 4, ra) != aiednn_success) {
            printf("FAIL: %s ref:\n", name);
            return 1;
        }
        aiednn_primitive_destroy(r);
        aiednn_set_impl_filter(nullptr);
    }
    aiednn_primitive_t p;
    if (aiednn_convolution_forward_create_attr(&p, &sd, &wd, &bd, &dd, &cp, &at) != aiednn_success) {
        printf("FAIL: %s create\n", name);
        return 1;
    }
    aiednn_primitive_set_constant_weights(p, 1);
    aiednn_exec_arg_t a[4] = {
        {AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_WEIGHTS, g_w}, {AIEDNN_ARG_BIAS, g_bias}, {AIEDNN_ARG_DST, g_y}};
    double first = 0, steady = 0;
    for (int rep = 0; rep <= REPS; rep++) {
        if (rep == 1)
            aiednn_stats_reset(); /* stats: steady executes only */
        const double t0 = now_ms();
        if (aiednn_primitive_execute_args(p, 4, a) != aiednn_success) {
            printf("FAIL: %s execute\n", name);
            return 1;
        }
        const double t = now_ms() - t0;
        if (rep)
            steady += t / REPS;
        else
            first = t;
    }
    long bad = 0;
    for (long i = 0; i < nout; i++)
        bad += g_y[i] != g_ref[i];
    const double mmac = (double)ho * ho * o * c * k * k / 1e6;
    printf("CONVPROF %s impl=%s M=%lld N=%lld K=%lld t_ms=%.3f t_first_ms=%.3f gmacs=%.2f mismatches=%ld of %ld (stats "
           "over %d runs)\n",
           name, aiednn_primitive_impl_info(p), (long long)(ho * ho), (long long)o, (long long)(c * k * k), steady,
           first, mmac / steady, bad, nout, REPS);
    aiednn_print_stats();
    aiednn_primitive_destroy(p);
    return bad != 0;
}

int main() {
    for (int i = 0; i < (int)sizeof g_x; i++)
        g_x[i] = (int8_t)((i * 7 + 3) % 61 - 30);
    for (int i = 0; i < (int)sizeof g_w; i++)
        g_w[i] = (int8_t)((i * 13 + 5) % 15 - 7);
    for (int i = 0; i < 512; i++)
        g_bias[i] = (i % 17) * 100 - 800, g_scales[i] = 0.001f + 0.0001f * (float)(i % 7);
    int rc = 0;
    rc |= conv("stem_7x7_s2", 4, 224, 64, 7, 2); /* C 3 padded to 4 */
    rc |= conv("l1_3x3_64", 64, 56, 64, 3, 1);
    rc |= conv("l2_3x3_s2", 64, 56, 128, 3, 2);
    rc |= conv("l2_3x3_128", 128, 28, 128, 3, 1);
    rc |= conv("l3_3x3_256", 256, 14, 256, 3, 1);
    rc |= conv("l4_3x3_512", 512, 7, 512, 3, 1);
    rc |= conv("l4_ds_1x1_s2", 256, 14, 512, 1, 2);
    if (!rc)
        printf("PASS: convprof_app\n");
    return rc;
}
