/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ops_app -- every aiednn primitive once; int8 GEMM family checked bit for bit vs ref:,
 * every op prints an "OP" checksum line for board vs x86 comparison.
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/ops_app.cc --target x86 --run
 *   ... --target vek280 --run --board <host> --vitis-settings <.../settings64.sh>
 ******************************************************************************/
#include "aiednn.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define BUF 65536
static uint32_t g_lcg = 12345u;
static float g_f[4][BUF], g_out[BUF], g_ref[BUF];
static int8_t g_i8[3][BUF];
static int32_t g_i32[2][BUF];
static int g_fail;

static uint32_t lcg(void) { return g_lcg = g_lcg * 1664525u + 1013904223u; }
static float frand(float lo, float hi) { return lo + (hi - lo) * (float)(lcg() >> 8) / 16777216.0f; }
static void fill_f(float *x, long n, float lo, float hi) {
    for (long i = 0; i < n; i++)
        x[i] = frand(lo, hi);
}
static void fill_i8(int8_t *x, long n) {
    for (long i = 0; i < n; i++)
        x[i] = (int8_t)(lcg() >> 24);
}

static aiednn_memory_desc_t D(int nd, const int64_t *dims, aiednn_data_type_t dt) {
    aiednn_memory_desc_t m;
    aiednn_memory_desc_init(&m, nd, dims, dt);
    return m;
}

static long numel(const aiednn_memory_desc_t *m) {
    long n = 1;
    for (int i = 0; i < m->ndims; i++)
        n *= (long)m->dims[i];
    return n;
}

static void report(const char *name, aiednn_primitive_t p, const aiednn_memory_desc_t *dmd, const void *dst) {
    double s = 0, a = 0;
    const long n = numel(dmd);
    for (long i = 0; i < n; i++) {
        const double v = dmd->data_type == aiednn_f32   ? ((const float *)dst)[i]
                         : dmd->data_type == aiednn_s32 ? ((const int32_t *)dst)[i]
                         : dmd->data_type == aiednn_u8  ? ((const uint8_t *)dst)[i]
                                                        : ((const int8_t *)dst)[i];
        s += v;
        a += fabs(v);
    }
    printf("OP %s impl=%s sum=%.9e asum=%.9e\n", name, aiednn_primitive_impl_info(p), s, a);
}

// pp, not p: argument evaluation order is unspecified, create must set p before it is read.
static int run(const char *name, aiednn_status_t cst, aiednn_primitive_t *pp, int nargs, const aiednn_exec_arg_t *args,
               const aiednn_memory_desc_t *dmd, const void *dst) {
    if (cst != aiednn_success) {
        printf("FAIL: %s create: %s\n", name, aiednn_status_str(cst));
        g_fail = 1;
        return 1;
    }
    aiednn_primitive_t p = *pp;
    const aiednn_status_t st = aiednn_primitive_execute_args(p, nargs, args);
    if (st != aiednn_success) {
        printf("FAIL: %s execute: %s\n", name, aiednn_status_str(st));
        g_fail = 1;
        aiednn_primitive_destroy(p);
        return 1;
    }
    report(name, p, dmd, dst);
    aiednn_primitive_destroy(p);
    return 0;
}

// int8 GEMM family: AIE (whatever dispatch picks) vs ref:, bit for bit (s32 outputs)
typedef aiednn_status_t (*create_fn)(aiednn_primitive_t *, const void *);
static void aie_vs_ref(const char *name, create_fn mk, const void *ctx, int nargs, aiednn_exec_arg_t *args,
                       const aiednn_memory_desc_t *dmd) {
    int32_t *outs[2] = {g_i32[0], g_i32[1]};
    for (int pass = 0; pass < 2; pass++) {
        aiednn_set_impl_filter(pass ? "ref:" : nullptr);
        aiednn_primitive_t p;
        const aiednn_status_t st = mk(&p, ctx);
        for (int i = 0; i < nargs; i++)
            if (args[i].arg == AIEDNN_ARG_DST)
                args[i].data = outs[pass];
        if (run(name, st, &p, nargs, args, dmd, outs[pass]))
            break;
    }
    aiednn_set_impl_filter(nullptr);
    long bad = 0, n = numel(dmd);
    for (long i = 0; i < n; i++)
        bad += outs[0][i] != outs[1][i];
    printf("OPCHK %s mismatches=%ld of %ld\n", name, bad, n);
    g_fail |= bad != 0;
}

typedef struct {
    aiednn_memory_desc_t s, w, b, d;
    aiednn_conv_params_t cp;
    int deconv, transposed;
} gemm_ctx;

static aiednn_status_t mk_conv(aiednn_primitive_t *p, const void *c) {
    const gemm_ctx *g = (const gemm_ctx *)c;
    return g->deconv ? aiednn_deconvolution_forward_create(p, &g->s, &g->w, &g->b, &g->d, &g->cp)
                     : aiednn_convolution_forward_create(p, &g->s, &g->w, &g->b, &g->d, &g->cp);
}
static aiednn_status_t mk_mm(aiednn_primitive_t *p, const void *c) {
    const gemm_ctx *g = (const gemm_ctx *)c;
    return aiednn_matmul_forward_create(p, &g->s, &g->w, &g->b, &g->d, g->transposed);
}
static aiednn_status_t mk_ip(aiednn_primitive_t *p, const void *c) {
    const gemm_ctx *g = (const gemm_ctx *)c;
    return aiednn_inner_product_forward_create(p, &g->s, &g->w, &g->b, &g->d);
}

static void gemm_family(void) {
    fill_i8(g_i8[0], BUF);
    fill_i8(g_i8[1], BUF);
    for (int i = 0; i < 1024; i++)
        g_i32[0][i] = (int32_t)(lcg() % 2001) - 1000;
    static int32_t bias[1024];
    memcpy(bias, g_i32[0], sizeof bias);
    aiednn_exec_arg_t args[4] = {
        {AIEDNN_ARG_SRC, g_i8[0]}, {AIEDNN_ARG_WEIGHTS, g_i8[1]}, {AIEDNN_ARG_BIAS, bias}, {AIEDNN_ARG_DST, nullptr}};
    gemm_ctx g;
    memset(&g, 0, sizeof g);
    // conv 3x3, stride 2x1, pad, dilation, 2 groups, batch 2
    // 8 output channels per group: narrower groups dispatch to ref: (RT_MIN_GROUP_N)
    const int64_t cs[4] = {2, 8, 13, 11}, cw[4] = {16, 4, 3, 3}, cb[1] = {16}, cd[4] = {2, 16, 6, 9};
    g.s = D(4, cs, aiednn_s8), g.w = D(4, cw, aiednn_s8), g.b = D(1, cb, aiednn_s32), g.d = D(4, cd, aiednn_s32);
    g.cp.strides[0] = 2, g.cp.strides[1] = 1, g.cp.padding_l[0] = 1, g.cp.padding_l[1] = 0;
    g.cp.padding_r[0] = 0, g.cp.padding_r[1] = 2, g.cp.dilates[0] = 0, g.cp.dilates[1] = 1, g.cp.groups = 2;
    aie_vs_ref("conv_s8", mk_conv, &g, 4, args, &g.d);
    // deconv 3x3 stride 2, pad 1
    const int64_t ds[4] = {1, 6, 5, 5}, dw[4] = {4, 6, 3, 3}, db[1] = {4}, dd[4] = {1, 4, 9, 9};
    memset(&g, 0, sizeof g);
    g.s = D(4, ds, aiednn_s8), g.w = D(4, dw, aiednn_s8), g.b = D(1, db, aiednn_s32), g.d = D(4, dd, aiednn_s32);
    g.cp.strides[0] = g.cp.strides[1] = 2, g.cp.padding_l[0] = g.cp.padding_l[1] = 1;
    g.cp.padding_r[0] = g.cp.padding_r[1] = 1, g.cp.groups = 1, g.deconv = 1;
    aie_vs_ref("deconv_s8", mk_conv, &g, 4, args, &g.d);
    // batched matmul [2,40,150] x [1,150,70] + bias [1,1,70]
    const int64_t ms[3] = {2, 40, 150}, mw[3] = {1, 150, 70}, mb[3] = {1, 1, 70}, md[3] = {2, 40, 70};
    memset(&g, 0, sizeof g);
    g.s = D(3, ms, aiednn_s8), g.w = D(3, mw, aiednn_s8), g.b = D(3, mb, aiednn_s32), g.d = D(3, md, aiednn_s32);
    aie_vs_ref("matmul_s8", mk_mm, &g, 4, args, &g.d);
    // inner product [4, 8x4x4] -> 100
    const int64_t is[4] = {4, 8, 4, 4}, iw[4] = {100, 8, 4, 4}, ib[1] = {100}, id[2] = {4, 100};
    memset(&g, 0, sizeof g);
    g.s = D(4, is, aiednn_s8), g.w = D(4, iw, aiednn_s8), g.b = D(1, ib, aiednn_s32), g.d = D(2, id, aiednn_s32);
    aie_vs_ref("inner_product_s8", mk_ip, &g, 4, args, &g.d);
}

static void f32_ops(void) {
    for (int k = 0; k < 4; k++)
        fill_f(g_f[k], BUF, -2.0f, 2.0f);
    for (long i = 0; i < BUF; i++)
        g_f[3][i] = fabsf(g_f[3][i]) + 0.1f; // positive operand
    aiednn_primitive_t p;
    char name[64];
    const int64_t x4[4] = {2, 6, 8, 10};
    const aiednn_memory_desc_t X = D(4, x4, aiednn_f32);
    aiednn_exec_arg_t sd[2] = {{AIEDNN_ARG_SRC, g_f[0]}, {AIEDNN_ARG_DST, g_out}};
    aiednn_exec_arg_t sdp[2] = {{AIEDNN_ARG_SRC, g_f[3]}, {AIEDNN_ARG_DST, g_out}};
    for (int alg = aiednn_eltwise_relu; alg <= aiednn_eltwise_cos; alg++) {
        const int pos = alg == aiednn_eltwise_log || alg == aiednn_eltwise_sqrt || alg == aiednn_eltwise_rsqrt ||
                        alg == aiednn_eltwise_reciprocal;
        const float a = alg == aiednn_eltwise_clip                                             ? -0.5f
                        : alg == aiednn_eltwise_hardswish || alg == aiednn_eltwise_hardsigmoid ? 1.0f / 6
                                                                                               : 1.0f;
        const float b = alg == aiednn_eltwise_clip ? 0.7f : alg == aiednn_eltwise_pow ? 3.0f : 0.5f;
        snprintf(name, sizeof name, "eltwise_%d", alg);
        run(name, aiednn_eltwise_forward_create(&p, (aiednn_alg_kind_t)alg, a, b, &X, &X), &p, 2, pos ? sdp : sd, &X,
            g_out);
    }
    const int64_t bc[4] = {1, 6, 1, 10};
    const aiednn_memory_desc_t B = D(4, bc, aiednn_f32);
    aiednn_exec_arg_t bin[4] = {
        {AIEDNN_ARG_SRC, g_f[3]}, {AIEDNN_ARG_SRC_1, g_f[2]}, {AIEDNN_ARG_DST, g_out}, {AIEDNN_ARG_SRC_2, g_f[1]}};
    for (int alg = aiednn_binary_add; alg <= aiednn_binary_select; alg++) {
        snprintf(name, sizeof name, "binary_%d", alg);
        const int sel = alg == aiednn_binary_select;
        run(name, aiednn_binary_create(&p, (aiednn_alg_kind_t)alg, &X, &B, sel ? &X : nullptr, &X), &p, sel ? 4 : 3,
            bin, &X, g_out);
    }
    const int64_t r1[4] = {2, 1, 8, 1};
    const aiednn_memory_desc_t R = D(4, r1, aiednn_f32);
    for (int alg = aiednn_reduction_sum; alg <= aiednn_reduction_norm_lp_power_p_sum; alg++) {
        snprintf(name, sizeof name, "reduction_%d", alg);
        run(name, aiednn_reduction_create(&p, (aiednn_alg_kind_t)alg, 2.0f, 0.0f, &X, &R), &p, 2,
            alg == aiednn_reduction_mul ? sdp : sd, &R, g_out);
    }
    aiednn_pool_params_t pp;
    memset(&pp, 0, sizeof pp);
    pp.kernel[0] = pp.kernel[1] = 3, pp.strides[0] = pp.strides[1] = 2, pp.padding_l[0] = pp.padding_l[1] = 1;
    pp.padding_r[0] = pp.padding_r[1] = 1;
    const int64_t po[4] = {2, 6, 4, 5};
    const aiednn_memory_desc_t P = D(4, po, aiednn_f32);
    for (int alg = aiednn_pooling_max; alg <= aiednn_pooling_avg_exclude_padding; alg++) {
        snprintf(name, sizeof name, "pooling_%d", alg);
        run(name, aiednn_pooling_forward_create(&p, (aiednn_alg_kind_t)alg, &X, &P, &pp), &p, 2, sd, &P, g_out);
    }
    const int64_t up[4] = {2, 6, 13, 7};
    const aiednn_memory_desc_t U = D(4, up, aiednn_f32);
    for (int alg = aiednn_resampling_nearest; alg <= aiednn_resampling_linear; alg++) {
        snprintf(name, sizeof name, "resampling_%d", alg);
        run(name, aiednn_resampling_forward_create(&p, (aiednn_alg_kind_t)alg, &X, &U), &p, 2, sd, &U, g_out);
    }
    for (int alg = aiednn_lrn_across_channels; alg <= aiednn_lrn_within_channel; alg++) {
        snprintf(name, sizeof name, "lrn_%d", alg);
        run(name, aiednn_lrn_forward_create(&p, (aiednn_alg_kind_t)alg, 3, 1e-2f, 0.75f, 1.5f, &X, &X), &p, 2, sd, &X,
            g_out);
    }
    for (int alg = aiednn_softmax_accurate; alg <= aiednn_softmax_log; alg++) {
        snprintf(name, sizeof name, "softmax_%d", alg);
        run(name, aiednn_softmax_forward_create(&p, (aiednn_alg_kind_t)alg, 1, &X, &X), &p, 2, sd, &X, g_out);
    }
    aiednn_exec_arg_t nrm[6] = {{AIEDNN_ARG_SRC, g_f[0]},      {AIEDNN_ARG_DST, g_out},    {AIEDNN_ARG_MEAN, g_f[1]},
                                {AIEDNN_ARG_VARIANCE, g_f[3]}, {AIEDNN_ARG_SCALE, g_f[2]}, {AIEDNN_ARG_SHIFT, g_f[1]}};
    const unsigned ss = AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT;
    run("batch_normalization", aiednn_batch_normalization_forward_inference_create(&p, &X, &X, 1e-3f, ss), &p, 6, nrm,
        &X, g_out);
    run("layer_normalization", aiednn_layer_normalization_forward_inference_create(&p, &X, &X, 1e-5f, ss), &p, 6, nrm,
        &X, g_out);
    run("rms_normalization",
        aiednn_layer_normalization_forward_inference_create(&p, &X, &X, 1e-5f, AIEDNN_USE_SCALE | AIEDNN_RMS_NORM), &p,
        6, nrm, &X, g_out);
    run("group_normalization", aiednn_group_normalization_forward_inference_create(&p, 3, &X, &X, 1e-5f, ss), &p, 6,
        nrm, &X, g_out);
    run("instance_normalization", aiednn_group_normalization_forward_inference_create(&p, 6, &X, &X, 1e-5f, ss), &p, 6,
        nrm, &X, g_out);
    const int64_t pw[4] = {1, 6, 1, 1};
    const aiednn_memory_desc_t PW = D(4, pw, aiednn_f32);
    aiednn_exec_arg_t pr[3] = {{AIEDNN_ARG_SRC, g_f[0]}, {AIEDNN_ARG_WEIGHTS, g_f[3]}, {AIEDNN_ARG_DST, g_out}};
    run("prelu", aiednn_prelu_forward_create(&p, &X, &PW, &X), &p, 3, pr, &X, g_out);
    // f32 conv / deconv / matmul / inner product (ref:)
    const int64_t fw[4] = {4, 6, 3, 3}, fd[4] = {2, 4, 8, 10}, fb[1] = {4};
    const aiednn_memory_desc_t FW = D(4, fw, aiednn_f32), FD = D(4, fd, aiednn_f32), FB = D(1, fb, aiednn_f32);
    aiednn_conv_params_t cp;
    memset(&cp, 0, sizeof cp);
    cp.strides[0] = cp.strides[1] = 1, cp.padding_l[0] = cp.padding_l[1] = 1, cp.padding_r[0] = cp.padding_r[1] = 1;
    cp.groups = 1;
    aiednn_exec_arg_t cv[4] = {
        {AIEDNN_ARG_SRC, g_f[0]}, {AIEDNN_ARG_WEIGHTS, g_f[1]}, {AIEDNN_ARG_BIAS, g_f[2]}, {AIEDNN_ARG_DST, g_out}};
    run("conv_f32", aiednn_convolution_forward_create(&p, &X, &FW, &FB, &FD, &cp), &p, 4, cv, &FD, g_out);
    const int64_t dw2[4] = {4, 6, 3, 3};
    const aiednn_memory_desc_t DW2 = D(4, dw2, aiednn_f32);
    run("deconv_f32", aiednn_deconvolution_forward_create(&p, &X, &DW2, &FB, &FD, &cp), &p, 4, cv, &FD, g_out);
    const int64_t a2[2] = {12, 80}, w2[2] = {80, 30}, d2[2] = {12, 30};
    const aiednn_memory_desc_t A2 = D(2, a2, aiednn_f32), W2 = D(2, w2, aiednn_f32), D2 = D(2, d2, aiednn_f32);
    run("matmul_f32", aiednn_matmul_forward_create(&p, &A2, &W2, nullptr, &D2, 0), &p, 4, cv, &D2, g_out);
    // sdpa causal
    const int64_t q4[4] = {2, 2, 9, 16}, v4[4] = {2, 2, 9, 8};
    const aiednn_memory_desc_t Q = D(4, q4, aiednn_f32), V = D(4, v4, aiednn_f32);
    aiednn_exec_arg_t at[4] = {
        {AIEDNN_ARG_QUERY, g_f[0]}, {AIEDNN_ARG_KEY, g_f[1]}, {AIEDNN_ARG_VALUE, g_f[2]}, {AIEDNN_ARG_DST, g_out}};
    run("sdpa", aiednn_sdpa_forward_create(&p, &Q, &Q, &V, nullptr, &V, 0.25f, 1), &p, 4, at, &V, g_out);
}

static void movement_ops(void) {
    aiednn_primitive_t p;
    const int64_t a[3] = {2, 3, 5}, b[3] = {2, 4, 5}, c[3] = {2, 7, 5};
    aiednn_memory_desc_t srcs[2] = {D(3, a, aiednn_f32), D(3, b, aiednn_f32)};
    const aiednn_memory_desc_t C7 = D(3, c, aiednn_f32);
    aiednn_exec_arg_t cat[3] = {
        {AIEDNN_ARG_MULTIPLE_SRC, g_f[0]}, {AIEDNN_ARG_MULTIPLE_SRC + 1, g_f[1]}, {AIEDNN_ARG_DST, g_out}};
    run("concat", aiednn_concat_create(&p, 2, srcs, 1, &C7), &p, 3, cat, &C7, g_out);
    const float sc[2] = {0.5f, -2.0f};
    aiednn_memory_desc_t two[2] = {C7, C7};
    run("sum", aiednn_sum_create(&p, 2, sc, two, &C7), &p, 3, cat, &C7, g_out);
    const int64_t x4[4] = {2, 6, 8, 10}, xt[4] = {2, 8, 10, 6};
    const aiednn_memory_desc_t X = D(4, x4, aiednn_f32), XQ = D(4, xt, aiednn_s8), XD = D(4, x4, aiednn_f32);
    const int perm[4] = {0, 2, 3, 1};
    aiednn_exec_arg_t q[2] = {{AIEDNN_ARG_SRC, g_f[0]}, {AIEDNN_ARG_DST, g_i8[2]}};
    run("reorder_quantize_nhwc", aiednn_reorder_create(&p, &X, &XQ, perm, 0.02f, 3), &p, 2, q, &XQ, g_i8[2]);
    const aiednn_memory_desc_t XS = D(4, x4, aiednn_s8);
    aiednn_exec_arg_t dq[2] = {{AIEDNN_ARG_SRC, g_i8[0]}, {AIEDNN_ARG_DST, g_out}};
    run("reorder_dequantize", aiednn_reorder_create(&p, &XS, &XD, nullptr, 0.02f, 3), &p, 2, dq, &XD, g_out);
    aiednn_exec_arg_t sh[2] = {{AIEDNN_ARG_SRC, g_f[0]}, {AIEDNN_ARG_DST, g_out}};
    run("shuffle", aiednn_shuffle_forward_create(&p, 1, 3, &X, &X), &p, 2, sh, &X, g_out);
}

int main() {
    printf("=== aiednn ops self-test ===\n");
    gemm_family();
    f32_ops();
    movement_ops();
    aiednn_print_stats();
    printf(g_fail ? "FAIL: aiednn ops self-test\n"
                  : "PASS: aiednn ops self-test, int8 GEMM family bit-exact vs ref:\n");
    return g_fail;
}
