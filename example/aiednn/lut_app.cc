/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * lut_app -- the 8-bit per-element primitives on the AIE: the table mode (lut:,
 * aiednn_rt_lut.h) for eltwise, batch norm, PReLU, binary with a per-channel
 * operand, dequantize / requantize reorders, and the eltwise mode (eltblk:,
 * aiednn_rt_eltblk.h) for binary of two 8-bit tensors, and the pool mode
 * (redblk:, aiednn_rt_red.h) for reductions; f32 -> 8-bit ops on the step-table
 * mode (steps:, aiednn_rt_steps.h). Each runs with the default dispatch,
 * then with the ref: filter, and the two outputs are compared byte for byte
 * ("OPCHK <name> impl=<impl> mismatches=0 of N"). Deterministic inputs.
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/lut_app.cc --target x86 --run
 *   ... --target vek280 --run --board <host> --vitis-settings <.../settings64.sh>
 ******************************************************************************/
#include "aiednn.h"

#include <stdio.h>
#include <string.h>
#if !defined(__aarch64__)
#include <time.h>
#endif

#define BUF (1 << 20)
static uint32_t g_lcg = 777u;
static uint8_t g_x[BUF];
static uint8_t g_out[2][4 * BUF];
static float g_w[4096];
static float g_f[BUF];
static int g_fail;
static int g_expect_ref; /* the next checks must stay on ref: (an AIE path exists but loses) */

static uint32_t lcg(void) { return g_lcg = g_lcg * 1664525u + 1013904223u; }

static aiednn_memory_desc_t D4(int64_t n, int64_t c, int64_t h, int64_t w, aiednn_data_type_t dt) {
    aiednn_memory_desc_t m;
    const int64_t d[4] = {n, c, h, w};
    aiednn_memory_desc_init(&m, 4, d, dt);
    return m;
}

static long nbytes(const aiednn_memory_desc_t *m) {
    long n = m->data_type == aiednn_f32 || m->data_type == aiednn_s32 ? 4 : 1;
    for (int i = 0; i < m->ndims; i++)
        n *= (long)m->dims[i];
    return n;
}

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

typedef aiednn_status_t (*mk_fn)(aiednn_primitive_t *, const void *);

/* default dispatch vs ref:, byte for byte; args[ndst] is DST */
static void check(const char *name, mk_fn mk, const void *ctx, int nargs, aiednn_exec_arg_t *args, int ndst,
                  const aiednn_memory_desc_t *dmd) {
    char impl[160] = "?";
    double ms[2][2] = {{0, 0}, {0, 0}}; /* [pass][first call, second call] */
    for (int pass = 0; pass < 2; pass++) {
        aiednn_set_impl_filter(pass ? "ref:" : nullptr);
        aiednn_primitive_t p;
        const aiednn_status_t st = mk(&p, ctx);
        if (st != aiednn_success) {
            printf("FAIL: %s create: %s\n", name, aiednn_status_str(st));
            g_fail = 1;
            return;
        }
        if (!pass)
            snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(p));
        memset(g_out[pass], pass ? 0xa5 : 0x5a, (size_t)nbytes(dmd));
        args[ndst].data = g_out[pass];
        for (int rep = 0; rep < 2; rep++) { /* the second call is steady state (tables cached) */
            const double t0 = now_ms();
            if (aiednn_primitive_execute_args(p, nargs, args) != aiednn_success) {
                printf("FAIL: %s execute\n", name);
                g_fail = 1;
            }
            ms[pass][rep] = now_ms() - t0;
        }
        aiednn_primitive_destroy(p);
    }
    aiednn_set_impl_filter(nullptr);
    const long n = nbytes(dmd);
    long bad = 0;
    for (long i = 0; i < n; i++)
        bad += g_out[0][i] != g_out[1][i];
    const int on_aie = strncmp(impl, "lut:", 4) == 0 || strncmp(impl, "eltblk:", 7) == 0 ||
                       strncmp(impl, "redblk:", 7) == 0 || strncmp(impl, "steps:", 6) == 0;
    printf("OPCHK %s impl=%s mismatches=%ld of %ld bytes\n", name, impl, bad, n);
    printf("OPTIME %s aie_first_ms=%.3f aie_ms=%.3f ref_ms=%.3f\n", name, ms[0][0], ms[0][1], ms[1][1]);
    g_fail |= bad != 0 || on_aie == g_expect_ref;
}

typedef struct {
    aiednn_alg_kind_t alg;
    float a, b;
    aiednn_memory_desc_t s, w, d;
    unsigned flags;
} op_ctx;

static aiednn_status_t mk_elt(aiednn_primitive_t *p, const void *c) {
    const op_ctx *o = (const op_ctx *)c;
    return aiednn_eltwise_forward_create(p, o->alg, o->a, o->b, &o->s, &o->d);
}
static aiednn_status_t mk_bn(aiednn_primitive_t *p, const void *c) {
    const op_ctx *o = (const op_ctx *)c;
    return aiednn_batch_normalization_forward_inference_create(p, &o->s, &o->d, 1e-5f, o->flags);
}
static aiednn_status_t mk_bnq(aiednn_primitive_t *p, const void *c) { /* batch norm + quantize (fold-norm-quant) */
    const op_ctx *o = (const op_ctx *)c;
    aiednn_attr_t at;
    memset(&at, 0, sizeof at);
    at.dst_scale = o->a, at.dst_zero_point = (int32_t)o->b;
    return aiednn_batch_normalization_forward_inference_create_attr(p, &o->s, &o->d, 1e-5f, o->flags, &at);
}
static aiednn_status_t mk_prelu(aiednn_primitive_t *p, const void *c) {
    const op_ctx *o = (const op_ctx *)c;
    return aiednn_prelu_forward_create(p, &o->s, &o->w, &o->d);
}
static aiednn_status_t mk_bin(aiednn_primitive_t *p, const void *c) {
    const op_ctx *o = (const op_ctx *)c;
    return aiednn_binary_create(p, o->alg, &o->s, &o->w, nullptr, &o->d);
}
static aiednn_status_t mk_red(aiednn_primitive_t *p, const void *c) {
    const op_ctx *o = (const op_ctx *)c;
    return aiednn_reduction_create(p, o->alg, o->a, o->b, &o->s, &o->d);
}
static aiednn_status_t mk_reorder(aiednn_primitive_t *p, const void *c) {
    const op_ctx *o = (const op_ctx *)c;
    return aiednn_reorder_create(p, &o->s, &o->d, nullptr, o->a, (int32_t)o->b);
}

int main() {
    printf("=== aiednn lut self-test ===\n");
    for (long i = 0; i < BUF; i++)
        g_x[i] = (uint8_t)(lcg() >> 24);
    for (int i = 0; i < 4096; i++)
        g_w[i] = (float)((int)(lcg() % 2001) - 1000) / 400.0f;
    float mean[256], var[256], scale[256], shift[256];
    for (int c = 0; c < 256; c++) {
        mean[c] = g_w[c] * 8, var[c] = 1.0f + g_w[c + 256] * g_w[c + 256] * 50, scale[c] = g_w[c + 512];
        shift[c] = g_w[c + 768] * 5;
    }
    op_ctx o;
    aiednn_exec_arg_t a2[2] = {{AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_DST, nullptr}};

    o = {aiednn_eltwise_gelu_erf, 0, 0, D4(1, 64, 56, 56, aiednn_s8), {}, D4(1, 64, 56, 56, aiednn_s8), 0};
    check("eltwise_gelu_s8_s8", mk_elt, &o, 2, a2, 1, &o.d);
    o = {aiednn_eltwise_swish, 1.0f, 0, D4(1, 32, 28, 28, aiednn_u8), {}, D4(1, 32, 28, 28, aiednn_f32), 0};
    check("eltwise_swish_u8_f32", mk_elt, &o, 2, a2, 1, &o.d);
    o = {aiednn_eltwise_logistic, 0, 0, D4(1, 96, 28, 28, aiednn_s8), {}, D4(1, 96, 28, 28, aiednn_u8), 0};
    check("eltwise_logistic_s8_u8", mk_elt, &o, 2, a2, 1, &o.d);

    o = {aiednn_eltwise_relu,
         0,
         0,
         D4(1, 128, 28, 28, aiednn_s8),
         {},
         D4(1, 128, 28, 28, aiednn_s8),
         AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_FUSE_NORM_RELU};
    aiednn_exec_arg_t bn[6] = {{AIEDNN_ARG_SRC, g_x},      {AIEDNN_ARG_DST, nullptr}, {AIEDNN_ARG_MEAN, mean},
                               {AIEDNN_ARG_VARIANCE, var}, {AIEDNN_ARG_SCALE, scale}, {AIEDNN_ARG_SHIFT, shift}};
    check("bnorm_relu_s8_s8", mk_bn, &o, 6, bn, 1, &o.d);

    o = {aiednn_binary_mul,
         0,
         0,
         D4(1, 144, 56, 56, aiednn_s8),
         D4(1, 144, 1, 1, aiednn_f32),
         D4(1, 144, 56, 56, aiednn_s8),
         0};
    aiednn_exec_arg_t bi[3] = {{AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_SRC_1, g_w}, {AIEDNN_ARG_DST, nullptr}};
    check("binary_mul_perchannel_s8_s8", mk_bin, &o, 3, bi, 2, &o.d);

    o = {aiednn_eltwise_relu,           0, 0, D4(1, 48, 40, 40, aiednn_u8), D4(1, 48, 1, 1, aiednn_f32),
         D4(1, 48, 40, 40, aiednn_f32), 0};
    aiednn_exec_arg_t pr[3] = {{AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_WEIGHTS, g_w}, {AIEDNN_ARG_DST, nullptr}};
    check("prelu_u8_f32", mk_prelu, &o, 3, pr, 2, &o.d);

    o = {aiednn_eltwise_relu, 0.0213f, -5, D4(1, 96, 56, 56, aiednn_s8), {}, D4(1, 96, 56, 56, aiednn_s32), 0};
    check("reorder_s8_s32", mk_reorder, &o, 2, a2, 1, &o.d);
    o = {aiednn_eltwise_relu, 0.5f, 100, D4(1, 64, 56, 56, aiednn_s8), {}, D4(1, 64, 56, 56, aiednn_u8), 0};
    check("reorder_s8_u8", mk_reorder, &o, 2, a2, 1, &o.d);

    static uint8_t y[BUF];
    for (long i = 0; i < BUF; i++)
        y[i] = (uint8_t)(lcg() >> 24);
    aiednn_exec_arg_t b2[3] = {{AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_SRC_1, y}, {AIEDNN_ARG_DST, nullptr}};
    o = {aiednn_binary_add,
         0,
         0,
         D4(1, 64, 56, 56, aiednn_s8),
         D4(1, 64, 56, 56, aiednn_u8),
         D4(1, 64, 56, 56, aiednn_s8),
         0};
    check("binary_add_s8_u8_s8", mk_bin, &o, 3, b2, 2, &o.d);
    o = {aiednn_binary_mul,
         0,
         0,
         D4(1, 32, 28, 28, aiednn_s8),
         D4(1, 32, 28, 28, aiednn_s8),
         D4(1, 32, 28, 28, aiednn_f32),
         0};
    check("binary_mul_s8_s8_f32", mk_bin, &o, 3, b2, 2, &o.d);
    o = {aiednn_binary_gt,
         0,
         0,
         D4(2, 16, 20, 20, aiednn_u8),
         D4(1, 16, 20, 1, aiednn_u8),
         D4(2, 16, 20, 20, aiednn_u8),
         0};
    check("binary_gt_bcast_u8", mk_bin, &o, 3, b2, 2, &o.d);

    o = {aiednn_reduction_mean, 0, 0, D4(1, 1152, 7, 7, aiednn_s8), {}, D4(1, 1152, 1, 1, aiednn_f32), 0};
    check("reduction_mean_s8_f32", mk_red, &o, 2, a2, 1, &o.d);
    o = {aiednn_reduction_max, 0, 0, D4(2, 32, 112, 112, aiednn_u8), {}, D4(2, 32, 1, 1, aiednn_u8), 0};
    check("reduction_max_u8_chunks", mk_red, &o, 2, a2, 1, &o.d);
    o = {aiednn_reduction_norm_lp_sum, 2.0f, 1e-3f, D4(3, 64, 30, 1, aiednn_s8), {}, D4(3, 64, 1, 1, aiednn_f32), 0};
    check("reduction_l2_s8_f32", mk_red, &o, 2, a2, 1, &o.d);

    for (long i = 0; i < BUF; i++)
        g_f[i] = (float)((int)(lcg() >> 16) - 32768) / 9000.0f;
    aiednn_exec_arg_t f2[2] = {{AIEDNN_ARG_SRC, g_f}, {AIEDNN_ARG_DST, nullptr}};
    static int32_t g_i[BUF];
    for (long i = 0; i < BUF; i++)
        g_i[i] = (int32_t)(lcg() >> 20) - 2048;
    aiednn_exec_arg_t i2[2] = {{AIEDNN_ARG_SRC, g_i}, {AIEDNN_ARG_DST, nullptr}};
    o = {aiednn_eltwise_relu, 1.0f, 0, D4(1, 64, 56, 56, aiednn_s32), {}, D4(1, 64, 56, 56, aiednn_s8), 0};
    check("reorder_s32_s8", mk_reorder, &o, 2, i2, 1, &o.d);
    o = {aiednn_eltwise_clip, 0.0f, 6.0f, D4(1, 32, 56, 56, aiednn_f32), {}, D4(1, 32, 56, 56, aiednn_u8), 0};
    check("eltwise_clip_f32_u8", mk_elt, &o, 2, f2, 1, &o.d);
    o = {aiednn_eltwise_relu,
         0,
         0,
         D4(1, 64, 56, 56, aiednn_f32),
         {},
         D4(1, 64, 56, 56, aiednn_s8),
         AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_FUSE_NORM_RELU};
    aiednn_exec_arg_t bf[6] = {{AIEDNN_ARG_SRC, g_f},      {AIEDNN_ARG_DST, nullptr}, {AIEDNN_ARG_MEAN, mean},
                               {AIEDNN_ARG_VARIANCE, var}, {AIEDNN_ARG_SCALE, scale}, {AIEDNN_ARG_SHIFT, shift}};
    check("bnorm_relu_f32_s8_aie", mk_bn, &o, 6, bf, 1, &o.d); /* default preference: on the AIE */
    /* f32 batch norm: ref's loop beats steps: (aiednn_rt_steps.h acc_steps), which declines it when
     * asked for the fastest implementation */
    aiednn_set_preference(aiednn_prefer_fastest);
    g_expect_ref = 1;
    check("bnorm_relu_f32_s8", mk_bn, &o, 6, bf, 1, &o.d);
    /* densenet's batch norm + ReLU + quantize, two block sizes (C <= 256: the stats arrays) */
    o = {aiednn_eltwise_relu,
         0.031f,
         0,
         D4(1, 128, 56, 56, aiednn_f32),
         {},
         D4(1, 128, 56, 56, aiednn_s8),
         AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_FUSE_NORM_RELU};
    check("bnorm_quant_f32_s8_c128_56", mk_bnq, &o, 6, bf, 1, &o.d);
    o = {aiednn_eltwise_relu,
         0.017f,
         0,
         D4(1, 256, 28, 28, aiednn_f32),
         {},
         D4(1, 256, 28, 28, aiednn_s8),
         AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_FUSE_NORM_RELU};
    check("bnorm_quant_f32_s8_c256_28", mk_bnq, &o, 6, bf, 1, &o.d);
    g_expect_ref = 0;
    aiednn_set_preference(aiednn_prefer_aie);
    o = {aiednn_binary_mul,
         0,
         0,
         D4(1, 96, 56, 56, aiednn_f32),
         D4(1, 96, 1, 1, aiednn_f32),
         D4(1, 96, 56, 56, aiednn_s8),
         0};
    aiednn_exec_arg_t bm[3] = {{AIEDNN_ARG_SRC, g_f}, {AIEDNN_ARG_SRC_1, g_w}, {AIEDNN_ARG_DST, nullptr}};
    check("binary_mul_perchannel_f32_s8", mk_bin, &o, 3, bm, 2, &o.d);

    aiednn_print_stats();
    printf(g_fail ? "FAIL: aiednn lut self-test\n"
                  : "PASS: aiednn lut self-test, every op where expected (AIE / ref:), bit-exact vs ref:\n");
    return g_fail;
}
