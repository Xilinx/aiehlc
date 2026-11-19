/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * bench_app -- the per-primitive scoreboard: one line per problem with the
 * implementation dispatch picked, its time, ref:'s time on the same problem and
 * the byte mismatches between the two. Covers every AIE mode of the GEMM
 * megakernel (plain GEMM, GEMMQ, CONV, DW, DWQ, POOL, ELTWISE, LUT, STEPS, MMT)
 * and the primitives that still run on ref: (listed so the gaps stay visible).
 * Problems alternate between the library megakernels (dnn_build modes="library"),
 * so the run also exercises the ELF switching (mk_switches per line).
 *
 *   BENCH <name> impl=<impl> aie=<0|1> t_ms=<steady> t_first_ms=<first> ref_ms=<steady>
 *         speedup=<ref/t> mismatches=<n> of <bytes> mk_switches=<n>
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/bench_app.cc --target x86 --run
 *   ... --target vek280 --modes library --run --board <host> --vitis-settings <.../settings64.sh>
 ******************************************************************************/
#include "aiednn.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#if !defined(__aarch64__)
#include <time.h>
#endif

#define BUF (1 << 21)
#define REPS 5
static uint32_t g_lcg = 4242u;
static int8_t g_x[BUF], g_y[BUF], g_w[BUF];
static float g_f[BUF];
static int32_t g_bias[4096];
static float g_scales[4096];
static uint8_t g_out[2][4 * BUF];
static int g_fail, g_cases, g_on_aie;

extern "C" long aiednn_megakernel_switches __attribute__((weak));
static long switches(void) { return &aiednn_megakernel_switches ? aiednn_megakernel_switches : 0; }

static uint32_t lcg(void) { return g_lcg = g_lcg * 1664525u + 1013904223u; }

static aiednn_memory_desc_t D(int nd, int64_t a, int64_t b, int64_t c, int64_t d, aiednn_data_type_t dt) {
    aiednn_memory_desc_t m;
    const int64_t dims[4] = {a, b, c, d};
    aiednn_memory_desc_init(&m, nd, dims, dt);
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

/* an implementation that runs on the AIE: everything but ref: and the CPU-only composites */
static int on_aie(const char *impl) {
    return strstr(impl, "aie:") != nullptr || strncmp(impl, "lut:", 4) == 0 || strncmp(impl, "eltblk:", 7) == 0 ||
           strncmp(impl, "redblk:", 7) == 0 || strncmp(impl, "steps:", 6) == 0;
}

typedef struct {
    aiednn_memory_desc_t s, w, b, d;
    aiednn_conv_params_t cp;
    aiednn_pool_params_t pp;
    aiednn_attr_t at;
    aiednn_alg_kind_t alg;
    float a, b2;
    int use_attr;
} prob;

typedef aiednn_status_t (*mk_fn)(aiednn_primitive_t *, const prob *);

/* default dispatch, then ref:, on the same problem: times (first call, mean of REPS after it) and
 * byte mismatches; args[ndst] is DST */
static void bench(const char *name, mk_fn mk, const prob *pb, int nargs, aiednn_exec_arg_t *args, int ndst) {
    char impl[160] = "?";
    double first[2] = {0, 0}, steady[2] = {0, 0};
    const long sw0 = switches();
    for (int pass = 0; pass < 2; pass++) {
        aiednn_set_impl_filter(pass ? "ref:" : nullptr);
        aiednn_primitive_t p;
        const aiednn_status_t st = mk(&p, pb);
        if (st != aiednn_success) {
            printf("FAIL: %s create (%s): %s\n", name, pass ? "ref:" : "dispatch", aiednn_status_str(st));
            g_fail = 1;
            aiednn_set_impl_filter(nullptr);
            return;
        }
        if (!pass)
            snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(p));
        memset(g_out[pass], pass ? 0xa5 : 0x5a, (size_t)nbytes(&pb->d));
        args[ndst].data = g_out[pass];
        for (int rep = 0; rep <= REPS; rep++) {
            const double t0 = now_ms();
            if (aiednn_primitive_execute_args(p, nargs, args) != aiednn_success) {
                printf("FAIL: %s execute (%s)\n", name, pass ? "ref:" : "dispatch");
                g_fail = 1;
                break;
            }
            const double t = now_ms() - t0;
            if (rep)
                steady[pass] += t / REPS;
            else
                first[pass] = t;
        }
        aiednn_primitive_destroy(p);
    }
    aiednn_set_impl_filter(nullptr);
    const long n = nbytes(&pb->d);
    long bad = 0, fd = -1;
    int maxd = 0;
    for (long i = 0; i < n; i++)
        if (g_out[0][i] != g_out[1][i]) {
            const int dd = (int)(int8_t)g_out[0][i] - (int)(int8_t)g_out[1][i];
            maxd = dd > maxd ? dd : -dd > maxd ? -dd : maxd;
            fd = fd < 0 ? i : fd;
            bad++;
        }
    if (bad) { /* the first few differing bytes, and how many of ref:'s 127 / 0 bytes differ */
        long hi = 0, hibad = 0, zero = 0, zerobad = 0, shown = 0;
        for (long i = 0; i < n; i++) {
            const int r = (int8_t)g_out[1][i], a = (int8_t)g_out[0][i];
            hi += r == 127, hibad += r == 127 && a != r, zero += r == 0, zerobad += r == 0 && a != r;
            if (a != r && shown < 8) {
                printf("BENCHDIFF %s [%ld] aie=%d ref=%d\n", name, i, a, r);
                shown++;
            }
        }
        printf("BENCHDIFF %s ref127=%ld wrong=%ld ref0=%ld wrong=%ld\n", name, hi, hibad, zero, zerobad);
    }
    if (bad) /* first differing byte and the largest byte difference (8-bit dst) */
        printf("BENCHDIFF %s first=%ld aie=%d ref=%d max_byte_diff=%d\n", name, fd, (int)(int8_t)g_out[0][fd],
               (int)(int8_t)g_out[1][fd], maxd);
    /* fvec: (f32 on the AIE) is held to |aie - ref| <= 1e-5 + 1e-5 |ref| (fvec_app.cc), not to bytes */
    double worst = -1;
    if (bad && pb->d.data_type == aiednn_f32 && strncmp(impl, "fvec:", 5) == 0) {
        const float *fa = (const float *)g_out[0], *fr = (const float *)g_out[1];
        worst = 0;
        for (long i = 0; i < n / 4; i++) {
            const double d = fabs((double)fa[i] - fr[i]) / (1e-5 + 1e-5 * fabs((double)fr[i]));
            worst = d <= worst ? worst : d; /* NaN propagates as a failure */
        }
    }
    const int aie = on_aie(impl);
    g_cases++, g_on_aie += aie;
    printf("BENCH %s impl=%s aie=%d t_ms=%.3f t_first_ms=%.3f ref_ms=%.3f speedup=%.2f mismatches=%ld of %ld "
           "mk_switches=%ld",
           name, impl, aie, steady[0], first[0], steady[1], steady[0] > 0 ? steady[1] / steady[0] : 0.0, bad, n,
           switches() - sw0);
    printf(worst >= 0 ? " contract=%.3f\n" : "\n", worst);
    g_fail |= bad != 0 && !(worst >= 0 && worst <= 1.0);
}

static aiednn_status_t mk_conv(aiednn_primitive_t *p, const prob *c) {
    return aiednn_convolution_forward_create_attr(p, &c->s, &c->w, &c->b, &c->d, &c->cp,
                                                  c->use_attr ? &c->at : nullptr);
}
static aiednn_status_t mk_mm(aiednn_primitive_t *p, const prob *c) {
    return aiednn_matmul_forward_create_attr(p, &c->s, &c->w, &c->b, &c->d, 0, c->use_attr ? &c->at : nullptr);
}
static aiednn_status_t mk_ip(aiednn_primitive_t *p, const prob *c) {
    return aiednn_inner_product_forward_create_attr(p, &c->s, &c->w, &c->b, &c->d, c->use_attr ? &c->at : nullptr);
}
static aiednn_status_t mk_pool(aiednn_primitive_t *p, const prob *c) {
    return aiednn_pooling_forward_create(p, c->alg, &c->s, &c->d, &c->pp);
}
static aiednn_status_t mk_elt(aiednn_primitive_t *p, const prob *c) {
    return aiednn_eltwise_forward_create(p, c->alg, c->a, c->b2, &c->s, &c->d);
}
static aiednn_status_t mk_bin(aiednn_primitive_t *p, const prob *c) {
    return aiednn_binary_create(p, c->alg, &c->s, &c->w, nullptr, &c->d);
}
static aiednn_status_t mk_softmax(aiednn_primitive_t *p, const prob *c) {
    return aiednn_softmax_forward_create(p, c->alg, 3, &c->s, &c->d);
}
static aiednn_status_t mk_lnorm(aiednn_primitive_t *p, const prob *c) {
    return aiednn_layer_normalization_forward_inference_create(p, &c->s, &c->d, 1e-5f, 0);
}
static aiednn_status_t mk_resample(aiednn_primitive_t *p, const prob *c) {
    return aiednn_resampling_forward_create(p, c->alg, &c->s, &c->d);
}

/* int8 conv -> s8, per-channel scales + ReLU + requantize (the quantized model layer) */
static void conv_q(const char *name, int64_t n, int64_t c, int64_t h, int64_t o, int64_t k, int64_t s, int64_t groups) {
    prob pb;
    memset(&pb, 0, sizeof pb);
    const int64_t ho = (h + 2 * (k / 2) - k) / s + 1;
    pb.s = D(4, n, c, h, h, aiednn_s8), pb.w = D(4, o, c / groups, k, k, aiednn_s8);
    pb.b = D(1, o, 0, 0, 0, aiednn_s32), pb.d = D(4, n, o, ho, ho, aiednn_s8);
    pb.cp.strides[0] = pb.cp.strides[1] = s, pb.cp.groups = groups;
    pb.cp.padding_l[0] = pb.cp.padding_l[1] = pb.cp.padding_r[0] = pb.cp.padding_r[1] = k / 2;
    pb.at.n_scales = o, pb.at.scales = g_scales, pb.at.dst_scale = 0.05f;
    pb.at.n_post_ops = 1, pb.at.post_ops[0].kind = aiednn_post_op_eltwise;
    pb.at.post_ops[0].alg = aiednn_eltwise_relu;
    pb.use_attr = 1;
    aiednn_exec_arg_t a[4] = {
        {AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_WEIGHTS, g_w}, {AIEDNN_ARG_BIAS, g_bias}, {AIEDNN_ARG_DST, nullptr}};
    bench(name, mk_conv, &pb, 4, a, 3);
}

/* int8 conv -> s8 with a fused residual add (an s8 skip tensor of the output's shape) + ReLU */
static void conv_q_res(const char *name, int64_t c, int64_t h, int64_t o, int64_t k) {
    prob pb;
    memset(&pb, 0, sizeof pb);
    pb.s = D(4, 1, c, h, h, aiednn_s8), pb.w = D(4, o, c, k, k, aiednn_s8);
    pb.b = D(1, o, 0, 0, 0, aiednn_s32), pb.d = D(4, 1, o, h, h, aiednn_s8);
    pb.cp.strides[0] = pb.cp.strides[1] = 1, pb.cp.groups = 1;
    pb.cp.padding_l[0] = pb.cp.padding_l[1] = pb.cp.padding_r[0] = pb.cp.padding_r[1] = k / 2;
    pb.at.n_scales = o, pb.at.scales = g_scales, pb.at.dst_scale = 0.05f;
    pb.at.n_post_ops = 2;
    pb.at.post_ops[0].kind = aiednn_post_op_binary, pb.at.post_ops[0].alg = aiednn_binary_add;
    pb.at.post_ops[0].alpha = 0.04f, pb.at.post_ops[0].beta = 3, pb.at.post_ops[0].src1 = pb.d;
    pb.at.post_ops[1].kind = aiednn_post_op_eltwise, pb.at.post_ops[1].alg = aiednn_eltwise_relu;
    pb.use_attr = 1;
    aiednn_exec_arg_t a[5] = {{AIEDNN_ARG_SRC, g_x},
                              {AIEDNN_ARG_WEIGHTS, g_w},
                              {AIEDNN_ARG_BIAS, g_bias},
                              {AIEDNN_ARG_ATTR_POST_OP_SRC(0), g_y},
                              {AIEDNN_ARG_DST, nullptr}};
    bench(name, mk_conv, &pb, 5, a, 4);
}

/* int8 conv -> s32 (no epilogue: the plain GEMM / K-chain entries) */
static void conv_s32(const char *name, int64_t c, int64_t h, int64_t o, int64_t k, int64_t groups) {
    prob pb;
    memset(&pb, 0, sizeof pb);
    pb.s = D(4, 1, c, h, h, aiednn_s8), pb.w = D(4, o, c / groups, k, k, aiednn_s8);
    pb.b = D(1, o, 0, 0, 0, aiednn_s32), pb.d = D(4, 1, o, h, h, aiednn_s32);
    pb.cp.strides[0] = pb.cp.strides[1] = 1, pb.cp.groups = groups;
    pb.cp.padding_l[0] = pb.cp.padding_l[1] = pb.cp.padding_r[0] = pb.cp.padding_r[1] = k / 2;
    aiednn_exec_arg_t a[4] = {
        {AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_WEIGHTS, g_w}, {AIEDNN_ARG_BIAS, g_bias}, {AIEDNN_ARG_DST, nullptr}};
    bench(name, mk_conv, &pb, 4, a, 3);
}

static void gemm_ops(void) {
    conv_q("conv3x3_s8_c64_h56", 1, 64, 56, 64, 3, 1, 1);
    conv_q("conv3x3s2_s8_c128_h28", 1, 128, 28, 256, 3, 2, 1);
    conv_q_res("conv3x3_s8_residual_c64_h56", 64, 56, 64, 3);
    conv_q("conv1x1_s8_c64_h56_o256", 1, 64, 56, 256, 1, 1, 1); /* K 64: short-K (MMT) candidate */
    conv_q("conv1x1_s8_c256_h14_o1024", 1, 256, 14, 1024, 1, 1, 1);
    conv_q("conv1x1_s8_c512_h7_o2048", 1, 512, 7, 2048, 1, 1, 1); /* M 49: weight-traffic bound */
    conv_q("conv7x7s2_s8_stem", 1, 3, 224, 64, 7, 2, 1);
    conv_q("conv3x3_s8_grouped4", 1, 128, 28, 128, 3, 1, 4);
    conv_q("dwconv3x3_s8_c144_h56", 1, 144, 56, 144, 3, 1, 144);
    conv_q("dwconv3x3s2_s8_c96_h112", 1, 96, 112, 96, 3, 2, 96);
    conv_q("dwconv5x5_s8_c240_h28", 1, 240, 28, 240, 5, 1, 240);
    conv_s32("conv3x3_s32_c32_h28", 32, 28, 64, 3, 1);
    conv_s32("dwconv3x3_s32_c64_h28", 64, 28, 64, 3, 64);

    prob pb;
    memset(&pb, 0, sizeof pb);
    aiednn_exec_arg_t a[4] = {
        {AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_WEIGHTS, g_w}, {AIEDNN_ARG_BIAS, g_bias}, {AIEDNN_ARG_DST, nullptr}};
    pb.s = D(3, 4, 128, 256, 0, aiednn_s8), pb.w = D(3, 1, 256, 512, 0, aiednn_s8);
    pb.b = D(3, 1, 1, 512, 0, aiednn_s32), pb.d = D(3, 4, 128, 512, 0, aiednn_s32);
    bench("matmul_s8_b4_128x256x512", mk_mm, &pb, 4, a, 3);
    pb.s = D(2, 8, 512, 0, 0, aiednn_s8), pb.w = D(2, 512, 1000, 0, 0, aiednn_s8);
    pb.b = D(2, 1, 1000, 0, 0, aiednn_s32), pb.d = D(2, 8, 1000, 0, 0, aiednn_s32);
    bench("matmul_s8_skinny_8x512x1000", mk_mm, &pb, 4, a, 3);
    pb.s = D(2, 1, 2048, 0, 0, aiednn_s8), pb.w = D(2, 1000, 2048, 0, 0, aiednn_s8);
    pb.b = D(1, 1000, 0, 0, 0, aiednn_s32), pb.d = D(2, 1, 1000, 0, 0, aiednn_s32);
    bench("inner_product_s8_2048x1000", mk_ip, &pb, 4, a, 3);
}

static void elementwise_ops(void) {
    prob pb;
    aiednn_exec_arg_t a2[2] = {{AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_DST, nullptr}};
    aiednn_exec_arg_t b3[3] = {{AIEDNN_ARG_SRC, g_x}, {AIEDNN_ARG_SRC_1, g_y}, {AIEDNN_ARG_DST, nullptr}};
    aiednn_exec_arg_t f2[2] = {{AIEDNN_ARG_SRC, g_f}, {AIEDNN_ARG_DST, nullptr}};

    memset(&pb, 0, sizeof pb);
    pb.alg = aiednn_pooling_max, pb.s = D(4, 1, 64, 112, 112, aiednn_s8), pb.d = D(4, 1, 64, 56, 56, aiednn_s8);
    pb.pp.kernel[0] = pb.pp.kernel[1] = 3, pb.pp.strides[0] = pb.pp.strides[1] = 2;
    pb.pp.padding_l[0] = pb.pp.padding_l[1] = pb.pp.padding_r[0] = pb.pp.padding_r[1] = 1;
    bench("maxpool3x3s2_s8_c64_h112", mk_pool, &pb, 2, a2, 1);
    pb.alg = aiednn_pooling_avg_exclude_padding, pb.s = D(4, 1, 512, 7, 7, aiednn_s8);
    pb.d = D(4, 1, 512, 1, 1, aiednn_s8);
    memset(&pb.pp, 0, sizeof pb.pp);
    pb.pp.kernel[0] = pb.pp.kernel[1] = 7, pb.pp.strides[0] = pb.pp.strides[1] = 1;
    bench("avgpool_global_s8_c512", mk_pool, &pb, 2, a2, 1);

    memset(&pb, 0, sizeof pb);
    pb.alg = aiednn_eltwise_gelu_erf, pb.s = D(4, 1, 64, 56, 56, aiednn_s8), pb.d = D(4, 1, 64, 56, 56, aiednn_s8);
    bench("eltwise_gelu_s8", mk_elt, &pb, 2, a2, 1);
    conv_q("conv3x3_s8_c64_h56_again", 1, 64, 56, 64, 3, 1, 1); /* back to the conv megakernel */
    pb.alg = aiednn_eltwise_logistic, pb.s = D(4, 1, 96, 28, 28, aiednn_s8), pb.d = D(4, 1, 96, 28, 28, aiednn_u8);
    bench("eltwise_sigmoid_s8_u8", mk_elt, &pb, 2, a2, 1);
    pb.alg = aiednn_eltwise_clip, pb.a = 0.0f, pb.b2 = 6.0f;
    pb.s = D(4, 1, 32, 56, 56, aiednn_f32), pb.d = D(4, 1, 32, 56, 56, aiednn_u8);
    bench("eltwise_clip_f32_u8", mk_elt, &pb, 2, f2, 1);

    memset(&pb, 0, sizeof pb);
    pb.alg = aiednn_binary_add, pb.s = D(4, 1, 64, 56, 56, aiednn_s8), pb.w = D(4, 1, 64, 56, 56, aiednn_s8);
    pb.d = D(4, 1, 64, 56, 56, aiednn_s8);
    bench("binary_add_s8", mk_bin, &pb, 3, b3, 2);
    pb.alg = aiednn_binary_mul, pb.d = D(4, 1, 64, 56, 56, aiednn_f32);
    bench("binary_mul_s8_f32", mk_bin, &pb, 3, b3, 2);

    /* still on ref: -- listed so the scoreboard shows the gaps */
    memset(&pb, 0, sizeof pb);
    pb.alg = aiednn_softmax_accurate, pb.s = D(4, 1, 12, 128, 128, aiednn_f32), pb.d = pb.s;
    bench("softmax_f32_12x128x128", mk_softmax, &pb, 2, f2, 1);
    pb.s = D(3, 1, 197, 768, 0, aiednn_f32), pb.d = pb.s;
    bench("layernorm_f32_197x768", mk_lnorm, &pb, 2, f2, 1);
    pb.alg = aiednn_resampling_nearest, pb.s = D(4, 1, 64, 28, 28, aiednn_f32), pb.d = D(4, 1, 64, 56, 56, aiednn_f32);
    bench("resample_nearest_f32_x2", mk_resample, &pb, 2, f2, 1);
}

int main() {
    printf("=== aiednn per-primitive scoreboard ===\n");
    for (long i = 0; i < BUF; i++) {
        g_x[i] = (int8_t)(lcg() >> 24), g_y[i] = (int8_t)(lcg() >> 24);
        g_w[i] = (int8_t)((int)(lcg() % 15) - 7);
        g_f[i] = (float)((int)(lcg() >> 16) - 32768) / 9000.0f;
    }
    for (int i = 0; i < 4096; i++)
        g_bias[i] = (int32_t)(lcg() % 4001) - 2000, g_scales[i] = 0.002f + 0.0001f * (float)(i % 17);
    gemm_ops();
    elementwise_ops();
    aiednn_print_stats();
    printf("SCOREBOARD %d problems, %d on the AIE, %ld megakernel switches\n", g_cases, g_on_aie, switches());
    printf(g_fail ? "FAIL: aiednn scoreboard\n"
                  : "PASS: aiednn scoreboard, every problem bit-exact vs ref: (fvec: within its f32 contract)\n");
    return g_fail;
}
