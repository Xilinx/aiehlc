/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * fvec_app -- FVEC mode (aie_modes_fvec.h) f32 ops vs ref: under |aie - ref| <= 1e-5 + 1e-5 |ref|:
 *
 *   FVCHK <name> impl=<impl> aie_ms=<steady> ref_ms=<steady> contract=<worst ratio> ok=<0|1>
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/fvec_app.cc --target x86 --run
 *   ... --target vek280 --run --board <host> --vitis-settings <.../settings64.sh>
 ******************************************************************************/
#include "aiednn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(__aarch64__)
#include <time.h>
#endif

#define REPS 3
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

static aiednn_memory_desc_t D(int nd, int64_t a, int64_t b = 1) {
    aiednn_memory_desc_t m;
    const int64_t dims[2] = {a, b};
    aiednn_memory_desc_init(&m, nd, dims, aiednn_f32);
    return m;
}

/* the primitive mk(), once by dispatch and once on ref:, outputs compared; out = args[o] */
template <typename Mk> static void check(const char *name, Mk mk, int nargs, aiednn_exec_arg_t *args, int o, long n) {
    float *res[2];
    double ms[2] = {0, 0};
    char impl[160] = "?";
    for (int pass = 0; pass < 2; pass++) {
        aiednn_set_impl_filter(pass ? "ref:" : nullptr);
        aiednn_primitive_t p;
        if (mk(&p) != aiednn_success) {
            printf("FVCHK_FAIL %s create\n", name);
            g_fail = 1;
            aiednn_set_impl_filter(nullptr);
            return;
        }
        if (!pass)
            snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(p));
        res[pass] = (float *)malloc((size_t)n * sizeof(float));
        args[o].data = res[pass];
        for (int r = 0; r <= REPS; r++) {
            const double t0 = now_ms();
            aiednn_primitive_execute_args(p, nargs, args);
            if (r)
                ms[pass] += (now_ms() - t0) / REPS;
        }
        aiednn_primitive_destroy(p);
    }
    aiednn_set_impl_filter(nullptr);
    double worst = 0;
    for (long i = 0; i < n; i++) {
        const double d = fabs((double)res[0][i] - res[1][i]) / (1e-5 + 1e-5 * fabs((double)res[1][i]));
        worst = d > worst || d != d ? d : worst;
    }
    const int ok = worst <= 1.0;
    g_fail |= !ok || strncmp(impl, "fvec:", 5) != 0;
    printf("FVCHK %s impl=%s aie_ms=%.3f ref_ms=%.3f contract=%.3f ok=%d\n", name, impl, ms[0], ms[1], worst, ok);
    free(res[0]), free(res[1]);
}

int main() {
    const long n = 256 * 1024;
    float *x = (float *)malloc(n * 4), *y = (float *)malloc(n * 4), *pos = (float *)malloc(n * 4);
    float *g = (float *)malloc(1024 * 4), *b = (float *)malloc(1024 * 4);
    for (long i = 0; i < n; i++) {
        x[i] = (float)((i * 2654435761u >> 8) % 12001) / 1000.0f - 6.0f;
        y[i] = 0.25f + (float)((i * 40503u >> 4) % 7001) / 1000.0f;
        pos[i] = 0.01f + (float)((i * 2246822519u >> 9) % 9001) / 1000.0f;
    }
    for (int i = 0; i < 1024; i++)
        g[i] = 0.5f + (float)(i % 37) / 37.0f, b[i] = (float)(i % 11) / 11.0f - 0.5f;
    aiednn_memory_desc_t v = D(2, 256, 1024);
    aiednn_exec_arg_t a[4] = {
        {AIEDNN_ARG_SRC, x}, {AIEDNN_ARG_DST, nullptr}, {AIEDNN_ARG_SRC_1, y}, {AIEDNN_ARG_SCALE, g}};
    struct {
        const char *name;
        aiednn_alg_kind_t alg;
        float al, be;
        int positive;
    } elt[] = {
        {"gelu_tanh", aiednn_eltwise_gelu_tanh, 0, 0, 0},
        {"gelu_erf", aiednn_eltwise_gelu_erf, 0, 0, 0},
        {"swish", aiednn_eltwise_swish, 1, 0, 0},
        {"mish", aiednn_eltwise_mish, 0, 0, 0},
        {"relu", aiednn_eltwise_relu, 0, 0, 0},
        {"tanh", aiednn_eltwise_tanh, 0, 0, 0},
        {"log", aiednn_eltwise_log, 0, 0, 1},
        {"rsqrt", aiednn_eltwise_rsqrt, 0, 0, 1},
    };
    for (auto &e : elt) {
        a[0].data = e.positive ? pos : x;
        check(
            e.name, [&](aiednn_primitive_t *p) { return aiednn_eltwise_forward_create(p, e.alg, e.al, e.be, &v, &v); },
            2, a, 1, n);
    }
    a[0].data = x;
    check(
        "add", [&](aiednn_primitive_t *p) { return aiednn_binary_create(p, aiednn_binary_add, &v, &v, nullptr, &v); },
        3, a, 1, n);
    check(
        "div", [&](aiednn_primitive_t *p) { return aiednn_binary_create(p, aiednn_binary_div, &v, &v, nullptr, &v); },
        3, a, 1, n);
    aiednn_memory_desc_t sm = D(2, 256, 1000);
    check(
        "softmax_256x1000",
        [&](aiednn_primitive_t *p) { return aiednn_softmax_forward_create(p, aiednn_softmax_accurate, 1, &sm, &sm); },
        2, a, 1, 256L * 1000);
    check(
        "logsoftmax_256x1000",
        [&](aiednn_primitive_t *p) { return aiednn_softmax_forward_create(p, aiednn_softmax_log, 1, &sm, &sm); }, 2, a,
        1, 256L * 1000);
    aiednn_memory_desc_t ln = D(2, 197, 384);
    aiednn_exec_arg_t la[4] = {
        {AIEDNN_ARG_SRC, x}, {AIEDNN_ARG_DST, nullptr}, {AIEDNN_ARG_SCALE, g}, {AIEDNN_ARG_SHIFT, b}};
    check(
        "layernorm_197x384",
        [&](aiednn_primitive_t *p) {
            return aiednn_layer_normalization_forward_inference_create(p, &ln, &ln, 1e-5f,
                                                                       AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT);
        },
        4, la, 1, 197L * 384);
    check(
        "rmsnorm_197x384",
        [&](aiednn_primitive_t *p) {
            return aiednn_layer_normalization_forward_inference_create(p, &ln, &ln, 1e-5f,
                                                                       AIEDNN_USE_SCALE | AIEDNN_RMS_NORM);
        },
        3, la, 1, 197L * 384);
    /* ROWPROG: per-channel rows (batch norm, PReLU, a [1,C,1,1] operand) on [1,64,64,64] */
    aiednn_memory_desc_t t4, c4;
    const int64_t d4[4] = {1, 64, 64, 64}, dc[4] = {1, 64, 1, 1};
    aiednn_memory_desc_init(&t4, 4, d4, aiednn_f32);
    aiednn_memory_desc_init(&c4, 4, dc, aiednn_f32);
    aiednn_exec_arg_t ba[6] = {{AIEDNN_ARG_SRC, x},      {AIEDNN_ARG_DST, nullptr}, {AIEDNN_ARG_MEAN, b},
                               {AIEDNN_ARG_VARIANCE, g}, {AIEDNN_ARG_SCALE, g},     {AIEDNN_ARG_SHIFT, b}};
    check(
        "batchnorm_relu_1x64x64x64",
        [&](aiednn_primitive_t *p) {
            return aiednn_batch_normalization_forward_inference_create(
                p, &t4, &t4, 1e-5f, AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT | AIEDNN_FUSE_NORM_RELU);
        },
        6, ba, 1, n);
    aiednn_exec_arg_t pa[3] = {{AIEDNN_ARG_SRC, x}, {AIEDNN_ARG_DST, nullptr}, {AIEDNN_ARG_WEIGHTS, b}};
    check(
        "prelu_1x64x64x64", [&](aiednn_primitive_t *p) { return aiednn_prelu_forward_create(p, &t4, &c4, &t4); }, 3, pa,
        1, n);
    aiednn_exec_arg_t ca[3] = {{AIEDNN_ARG_SRC, x}, {AIEDNN_ARG_DST, nullptr}, {AIEDNN_ARG_SRC_1, g}};
    check(
        "add_per_channel",
        [&](aiednn_primitive_t *p) { return aiednn_binary_create(p, aiednn_binary_add, &t4, &c4, nullptr, &t4); }, 3,
        ca, 1, n);
    /* eltwise backward: dy f'(x) (aie_fvec_prog_bwd), dy element for element */
    aiednn_exec_arg_t ea[3] = {{AIEDNN_ARG_SRC, x}, {AIEDNN_ARG_DIFF_SRC, nullptr}, {AIEDNN_ARG_DIFF_DST, y}};
    struct {
        const char *name;
        aiednn_alg_kind_t alg;
        float al;
    } bw[] = {
        {"relu_bwd", aiednn_eltwise_relu, 0},
        {"gelu_tanh_bwd", aiednn_eltwise_gelu_tanh, 0},
        {"tanh_bwd", aiednn_eltwise_tanh, 0},
        {"swish_bwd", aiednn_eltwise_swish, 1},
    };
    for (auto &e : bw)
        check(
            e.name,
            [&](aiednn_primitive_t *p) { return aiednn_eltwise_backward_create(p, e.alg, e.al, 0, &v, &v, &v); }, 3, ea,
            1, n);
    /* training rows (ROWSUM + ROWPROG with a per-element operand) */
    aiednn_memory_desc_t sm2 = D(2, 256, 1000);
    float *ys = (float *)malloc(256 * 1000 * 4); /* a real softmax output (rows sum to 1) */
    for (int r = 0; r < 256; r++) {
        double t = 0;
        for (int i = 0; i < 1000; i++)
            t += ys[r * 1000 + i] = expf(x[r * 1000 + i] - 6.0f);
        for (int i = 0; i < 1000; i++)
            ys[r * 1000 + i] = (float)(ys[r * 1000 + i] / t);
    }
    aiednn_exec_arg_t sb[3] = {{AIEDNN_ARG_DST, ys}, {AIEDNN_ARG_DIFF_SRC, nullptr}, {AIEDNN_ARG_DIFF_DST, y}};
    check(
        "softmax_bwd_256x1000",
        [&](aiednn_primitive_t *p) {
            return aiednn_softmax_backward_create(p, aiednn_softmax_accurate, 1, &sm2, &sm2, &sm2);
        },
        3, sb, 1, 256L * 1000);
    free(ys);
    float *mv = (float *)malloc(4096 * 4), *vv = (float *)malloc(4096 * 4);
    aiednn_memory_desc_t t5;
    const int64_t d5[4] = {4, 64, 32, 32};
    aiednn_memory_desc_init(&t5, 4, d5, aiednn_f32);
    aiednn_exec_arg_t bt[6] = {{AIEDNN_ARG_SRC, x},       {AIEDNN_ARG_DST, nullptr}, {AIEDNN_ARG_MEAN, mv},
                               {AIEDNN_ARG_VARIANCE, vv}, {AIEDNN_ARG_SCALE, g},     {AIEDNN_ARG_SHIFT, b}};
    check(
        "batchnorm_train_4x64x32x32",
        [&](aiednn_primitive_t *p) {
            return aiednn_batch_normalization_forward_training_create(p, &t5, &t5, 1e-5f,
                                                                      AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT);
        },
        6, bt, 1, n);
    check(
        "groupnorm_g8_4x64x32x32",
        [&](aiednn_primitive_t *p) {
            return aiednn_group_normalization_forward_inference_create(p, 8, &t5, &t5, 1e-5f,
                                                                       AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT);
        },
        6, bt, 1, n);
    check(
        "layernorm_train_197x384",
        [&](aiednn_primitive_t *p) {
            return aiednn_layer_normalization_forward_training_create(p, &ln, &ln, 1e-5f,
                                                                      AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT);
        },
        6, bt, 1, 197L * 384);
    free(mv), free(vv);
    { /* quantized binary over every (code, code) pair: byte-identical to ref: */
        aiednn_memory_desc_t q8;
        const int64_t dq[1] = {65536};
        aiednn_memory_desc_init(&q8, 1, dq, aiednn_s8);
        int8_t *qa = (int8_t *)malloc(65536), *qb = (int8_t *)malloc(65536),
               *qo[2] = {(int8_t *)malloc(65536), (int8_t *)malloc(65536)};
        for (int i = 0; i < 65536; i++)
            qa[i] = (int8_t)(i >> 8), qb[i] = (int8_t)(i & 255);
        const aiednn_alg_kind_t qalgs[4] = {aiednn_binary_add, aiednn_binary_sub, aiednn_binary_max, aiednn_binary_min};
        for (int ai = 0; ai < 4; ai++)
            for (int relu = 0; relu < 2; relu++) {
                aiednn_attr_t at;
                memset(&at, 0, sizeof at);
                at.src_scales[0] = 0.0213f, at.src_scales[1] = 0.0371f, at.src_zero_points[0] = 3,
                at.src_zero_points[1] = -7;
                at.dst_scale = 0.0459f, at.dst_zero_point = 5;
                if (relu)
                    at.n_post_ops = 1, at.post_ops[0].kind = aiednn_post_op_eltwise,
                    at.post_ops[0].alg = aiednn_eltwise_relu;
                char impl[160] = "?";
                double ms[2] = {0, 0};
                for (int pass = 0; pass < 2; pass++) {
                    aiednn_set_impl_filter(pass ? "ref:" : nullptr);
                    aiednn_primitive_t p;
                    if (aiednn_binary_create_attr(&p, qalgs[ai], &q8, &q8, &q8, &at) != aiednn_success) {
                        printf("FVCHK_FAIL qbinary create\n");
                        g_fail = 1;
                        break;
                    }
                    if (!pass)
                        snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(p));
                    aiednn_exec_arg_t qa3[3] = {
                        {AIEDNN_ARG_SRC, qa}, {AIEDNN_ARG_SRC_1, qb}, {AIEDNN_ARG_DST, qo[pass]}};
                    for (int r = 0; r <= REPS; r++) {
                        const double t0 = now_ms();
                        aiednn_primitive_execute_args(p, 3, qa3);
                        if (r)
                            ms[pass] += (now_ms() - t0) / REPS;
                    }
                    aiednn_primitive_destroy(p);
                }
                aiednn_set_impl_filter(nullptr);
                long bad = 0;
                for (int i = 0; i < 65536; i++)
                    bad += qo[0][i] != qo[1][i];
                const int ok = !bad && strncmp(impl, "fvec:", 5) == 0;
                g_fail |= !ok;
                printf("FVCHK qbinary_s8_all_pairs_alg%d_relu%d impl=%s aie_ms=%.3f ref_ms=%.3f mismatches=%ld ok=%d\n",
                       ai, relu, impl, ms[0], ms[1], bad, ok);
            }
        free(qa), free(qb), free(qo[0]), free(qo[1]);
    }
    printf(g_fail ? "FAIL: fvec app\n" : "PASS: fvec app\n");
    free(x), free(y), free(pos), free(g), free(b);
    return g_fail;
}
