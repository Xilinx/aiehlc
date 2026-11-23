/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * parity_bench -- times the primitives added for parity (doc/design/aiednn.md §24) at the sizes
 * networks use, with the implementation dispatch picks: which of them are slow enough on the host
 * to deserve an AIE implementation. One line per problem:
 *
 *   PBENCH <name> impl=<impl> t_ms=<mean of REPS after the first> t_first_ms=<first> ref_ms=<ref: steady,
 *          when dispatch picked another implementation> speedup=<ref / t> mflop=<n>
 *
 * A case that cannot create / execute prints PBENCH_FAIL (not "FAIL:", which ends a board run).
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/parity_bench.cc --target x86 --run
 *   ... --target vek280 --run --board <host> --vitis-settings <.../settings64.sh>
 * Data is random but untouched by checks (correctness is test_aiednn_*.py's job).
 ******************************************************************************/
#include "aiednn.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#if !defined(__aarch64__)
#include <time.h>
#endif

#define REPS 3
#define NSLOT 10
#define SLOT_BYTES (4L << 20) /* 10 slots + a copy: 44 MB of the board's 128 MB default heap */
static unsigned char *g_pool, *g_save;
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

static aiednn_memory_desc_t D(aiednn_data_type_t dt, int nd, int64_t a, int64_t b = 0, int64_t c = 0, int64_t d = 0,
                              int64_t e = 0) {
    aiednn_memory_desc_t m;
    const int64_t dims[5] = {a, b, c, d, e};
    aiednn_memory_desc_init(&m, nd, dims, dt);
    return m;
}

/* random contents for every slot: f32 in [-1, 1) where a slot is read as f32, small bytes otherwise
 * (a byte pattern of 0x3c.. is a moderate float, bf16, f16 and e8m0 scale alike) */
static void fill_pool(void) {
    uint32_t s = 4242u;
    float *f = (float *)g_pool;
    for (long i = 0; i < NSLOT * SLOT_BYTES / 4; i++) {
        s = s * 1664525u + 1013904223u;
        f[i] = (float)((int)(s >> 9) - (1 << 22)) / (float)(1 << 22);
    }
}

/* one primitive's timing: create once, execute REPS + 1 times with args[i] in pool slot i */
template <typename Mk>
static int time_one(Mk mk, int nargs, const int *tags, int constw, double *first, double *steady, char *impl) {
    aiednn_primitive_t p;
    if (mk(&p) != aiednn_success)
        return 0;
    if (constw) /* the weights a model holds constant: packed once, like an app */
        aiednn_primitive_set_constant_weights(p, 1);
    snprintf(impl, 160, "%s", aiednn_primitive_impl_info(p));
    aiednn_exec_arg_t a[NSLOT];
    for (int i = 0; i < nargs; i++)
        a[i].arg = tags[i], a[i].data = g_pool + i * SLOT_BYTES;
    *first = *steady = 0;
    for (int r = 0; r <= REPS; r++) {
        const double t0 = now_ms();
        if (aiednn_primitive_execute_args(p, nargs, a) != aiednn_success) {
            aiednn_primitive_destroy(p);
            return 0;
        }
        const double t = now_ms() - t0;
        if (r)
            *steady += t / REPS;
        else
            *first = t;
    }
    aiednn_primitive_destroy(p);
    return 1;
}

/* the dispatched implementation, and ref: on the same problem when dispatch picked something else */
template <typename Mk>
static void run(const char *name, double mflop, int nargs, const int *tags, Mk mk, int constw = 0) {
    char impl[160], rimpl[160];
    fill_pool(); /* fresh inputs: an earlier case's output (inf / NaN from random bf16 / e8m0 bytes) is not this one's
                    input */
    double first, steady, rfirst = 0, rsteady = 0;
    if (!time_one(mk, nargs, tags, constw, &first, &steady, impl)) {
        printf("PBENCH_FAIL %s (create or execute failed)\n", name);
        g_fail = 1;
        return;
    }
    char acc[64] = "-";
    if (strncmp(impl, "ref:", 4) != 0) {
        /* the output slot (dst, or the gradient a backward pass writes) after the AIE run, then ref:'s */
        int os = 0;
        for (int i = 0; i < nargs; i++)
            if (tags[i] == AIEDNN_ARG_DST || tags[i] == AIEDNN_ARG_DIFF_SRC || tags[i] == AIEDNN_ARG_DIFF_WEIGHTS) {
                os = i;
                break;
            }
        memcpy(g_save, g_pool + os * SLOT_BYTES, SLOT_BYTES);
        aiednn_set_impl_filter("ref:");
        time_one(mk, nargs, tags, constw, &rfirst, &rsteady, rimpl);
        aiednn_set_impl_filter(nullptr);
        if (!memcmp(g_save, g_pool + os * SLOT_BYTES, SLOT_BYTES))
            snprintf(acc, sizeof acc, "exact");
        else { /* f32 outputs: ||aie - ref|| / ||ref|| over the slot (the untouched tail is equal) */
            const float *a = (const float *)g_save, *r = (const float *)(g_pool + os * SLOT_BYTES);
            double dd = 0, rr = 0;
            long nf = 0; /* finite on one side only: a real difference (random bf16 / e8m0 bytes give inf on both) */
            for (long i = 0; i < SLOT_BYTES / 4; i++)
                if (isfinite(a[i]) && isfinite(r[i]))
                    dd += ((double)a[i] - r[i]) * ((double)a[i] - r[i]), rr += (double)r[i] * r[i];
                else
                    nf += isfinite(a[i]) != isfinite(r[i]);
            snprintf(acc, sizeof acc, nf ? "rel=%.2e,nonfinite_diff=%ld" : "rel=%.2e",
                     rr > 0 ? sqrt(dd / rr) : sqrt(dd), nf);
        }
    }
    printf("PBENCH %s impl=%s t_ms=%.3f t_first_ms=%.3f ref_ms=%.3f speedup=%.2f acc=%s mflop=%.1f\n", name, impl,
           steady, first, rsteady, rsteady > 0 && steady > 0 ? rsteady / steady : 0.0, acc, mflop);
    fflush(stdout);
}

enum { S_ = AIEDNN_ARG_SRC, W_ = AIEDNN_ARG_WEIGHTS, B_ = AIEDNN_ARG_BIAS, D_ = AIEDNN_ARG_DST };

static void spatial(void) {
    { /* 1-D conv (audio / sequence models), int8: rewritten as 2-D, so it can reach the AIE */
        aiednn_memory_desc_t s = D(aiednn_s8, 3, 1, 64, 2048), w = D(aiednn_s8, 3, 64, 64, 3),
                             d = D(aiednn_s32, 3, 1, 64, 2048);
        aiednn_conv_nd_params_t cp = {};
        cp.groups = 1;
        cp.strides[0] = 1, cp.padding_l[0] = cp.padding_r[0] = 1;
        const int t[] = {S_, W_, D_};
        run("conv1d_s8_c64_l2048_k3", 2.0 * 64 * 64 * 3 * 2048 / 1e6, 3, t, [&](aiednn_primitive_t *p) {
            return aiednn_convolution_forward_create_nd(p, &s, &w, nullptr, &d, &cp, nullptr);
        });
    }
    for (int q = 0; q < 2; q++) { /* 3-D conv (video / medical): ref: */
        const aiednn_data_type_t dt = q ? aiednn_s8 : aiednn_f32;
        aiednn_memory_desc_t s = D(dt, 5, 1, 16, 8, 28, 28), w = D(dt, 5, 32, 16, 3, 3, 3),
                             d = D(q ? aiednn_s32 : aiednn_f32, 5, 1, 32, 8, 28, 28);
        aiednn_conv_nd_params_t cp = {};
        cp.groups = 1;
        for (int i = 0; i < 3; i++)
            cp.strides[i] = 1, cp.padding_l[i] = cp.padding_r[i] = 1;
        const int t[] = {S_, W_, D_};
        run(q ? "conv3d_s8_16to32_8x28x28_k3" : "conv3d_f32_16to32_8x28x28_k3", 2.0 * 32 * 16 * 27 * 8 * 28 * 28 / 1e6,
            3, t, [&](aiednn_primitive_t *p) {
                return aiednn_convolution_forward_create_nd(p, &s, &w, nullptr, &d, &cp, nullptr);
            });
    }
    {
        aiednn_memory_desc_t s = D(aiednn_f32, 5, 1, 32, 8, 28, 28), d = D(aiednn_f32, 5, 1, 32, 4, 14, 14);
        aiednn_pool_nd_params_t pp = {};
        for (int i = 0; i < 3; i++)
            pp.kernel[i] = pp.strides[i] = 2;
        const int t[] = {S_, D_};
        run("pool3d_max_f32_32x8x28x28_k2", 32.0 * 8 * 28 * 28 / 1e6, 2, t, [&](aiednn_primitive_t *p) {
            return aiednn_pooling_forward_create_nd(p, aiednn_pooling_max, &s, &d, &pp);
        });
    }
}

static void recurrent(void) {
    const int64_t T = 32, N = 1, H = 128;
    const aiednn_alg_kind_t cells[2] = {aiednn_rnn_lstm, aiednn_rnn_lbr_gru};
    for (int c = 0; c < 2; c++) {
        const int64_t G = c ? 3 : 4, Gb = c ? 4 : 4;
        aiednn_memory_desc_t x = D(aiednn_f32, 3, T, N, H), wl = D(aiednn_f32, 5, 1, 1, H, G, H),
                             wi = D(aiednn_f32, 5, 1, 1, H, G, H), b = D(aiednn_f32, 4, 1, 1, Gb, H),
                             y = D(aiednn_f32, 3, T, N, H);
        const int t[] = {S_, W_, AIEDNN_ARG_WEIGHTS_ITER, B_, D_};
        run(c ? "gru_f32_T32_N1_H128" : "lstm_f32_T32_N1_H128", 2.0 * T * N * G * H * 2 * H / 1e6, 5, t,
            [&](aiednn_primitive_t *p) {
                return aiednn_rnn_forward_create(p, cells[c], aiednn_rnn_left2right, 0.f, &x, nullptr, nullptr, &wl,
                                                 &wi, &b, &y, nullptr, nullptr);
            });
    }
    { /* training: BPTT of the LSTM above (recomputes the forward) */
        aiednn_memory_desc_t x = D(aiednn_f32, 3, T, N, H), wl = D(aiednn_f32, 5, 1, 1, H, 4, H),
                             b = D(aiednn_f32, 4, 1, 1, 4, H), y = D(aiednn_f32, 3, T, N, H);
        aiednn_rnn_descs_t f = {&x, nullptr, nullptr, &wl, &wl, &b, &y, nullptr, nullptr};
        const int t[] = {S_,
                         W_,
                         AIEDNN_ARG_WEIGHTS_ITER,
                         B_,
                         AIEDNN_ARG_DIFF_SRC,
                         AIEDNN_ARG_DIFF_WEIGHTS,
                         AIEDNN_ARG_DIFF_WEIGHTS_ITER,
                         AIEDNN_ARG_DIFF_BIAS,
                         AIEDNN_ARG_DIFF_DST};
        run("lstm_bwd_f32_T32_N1_H128", 3 * 2.0 * T * N * 4 * H * 2 * H / 1e6, 9, t, [&](aiednn_primitive_t *p) {
            return aiednn_rnn_backward_create(p, aiednn_rnn_lstm, aiednn_rnn_left2right, 0.f, &f, &f);
        });
    }
}

/* the LLM decode GEMM: M = 1 or 16 tokens against a K x N weight in a narrow / grouped format */
static void weight_formats(void) {
    const int64_t K = 1024, N = 1024, g = 128; /* 2048 x 2048 takes 5-9 s per ref: call on the A72 */
    struct {
        const char *name;
        int64_t M;
        aiednn_data_type_t sdt, wdt, scl; /* scl: group-scale type, or 0 for none */
    } cs[] = {
        {"matmul_f32_M16_K1024_N1024", 16, aiednn_f32, aiednn_f32, (aiednn_data_type_t)0},
        {"matmul_bf16_M16_K1024_N1024", 16, aiednn_bf16, aiednn_bf16, (aiednn_data_type_t)0},
        {"matmul_f16_M16_K1024_N1024", 16, aiednn_f16, aiednn_f16, (aiednn_data_type_t)0},
        {"matmul_s8_M16_K1024_N1024", 16, aiednn_s8, aiednn_s8, (aiednn_data_type_t)0},
        {"wq_int4_g128_M1_K1024_N1024", 1, aiednn_f32, aiednn_s4, aiednn_f32},
        {"wq_int4_g128_M16_K1024_N1024", 16, aiednn_f32, aiednn_s4, aiednn_f32},
        {"wq_s8src_int4_g128_M16", 16, aiednn_s8, aiednn_s4, aiednn_f32},
        {"mxfp4_M16_K1024_N1024", 16, aiednn_f32, aiednn_f4_e2m1, aiednn_e8m0},
        {"mxfp8_e4m3_M16_K1024_N1024", 16, aiednn_f32, aiednn_f8_e4m3, aiednn_e8m0},
    };
    for (auto &c : cs) {
        const int64_t gg = c.wdt == aiednn_f4_e2m1 || c.wdt == aiednn_f8_e4m3 ? 32 : g;
        aiednn_memory_desc_t s = D(c.sdt, 2, c.M, K), w = D(c.wdt, 2, K, N),
                             d = D(c.sdt == aiednn_s8 && !c.scl ? aiednn_s32 : aiednn_f32, 2, c.M, N);
        aiednn_attr_t at;
        memset(&at, 0, sizeof at);
        if (c.scl)
            at.wei_scales = D(c.scl, 2, K / gg, N);
        const int t[] = {S_, W_, D_, AIEDNN_ARG_ATTR_WEI_SCALES};
        run(
            c.name, 2.0 * c.M * K * N / 1e6, c.scl ? 4 : 3, t,
            [&](aiednn_primitive_t *p) {
                return aiednn_matmul_forward_create_attr(p, &s, &w, nullptr, &d, 0, c.scl ? &at : nullptr);
            },
            1);
    }
}

static void training(void) {
    { /* a ResNet stage's 3x3 conv: both backward passes */
        aiednn_memory_desc_t x = D(aiednn_f32, 4, 8, 64, 28, 28), w = D(aiednn_f32, 4, 64, 64, 3, 3),
                             y = D(aiednn_f32, 4, 8, 64, 28, 28);
        aiednn_conv_nd_params_t cp = {};
        cp.groups = 1;
        for (int i = 0; i < 2; i++)
            cp.strides[i] = 1, cp.padding_l[i] = cp.padding_r[i] = 1;
        const double mf = 2.0 * 8 * 64 * 64 * 9 * 28 * 28 / 1e6;
        const int td[] = {AIEDNN_ARG_DIFF_SRC, W_, AIEDNN_ARG_DIFF_DST};
        run("conv_bwd_data_f32_8x64x28x28_k3", mf, 3, td,
            [&](aiednn_primitive_t *p) { return aiednn_convolution_backward_data_create(p, &x, &w, &y, &cp); });
        const int tw[] = {S_, AIEDNN_ARG_DIFF_WEIGHTS, AIEDNN_ARG_DIFF_DST};
        run("conv_bwd_weights_f32_8x64x28x28_k3", mf, 3, tw, [&](aiednn_primitive_t *p) {
            return aiednn_convolution_backward_weights_create(p, &x, &w, nullptr, &y, &cp);
        });
    }
    {
        aiednn_memory_desc_t x = D(aiednn_f32, 4, 8, 64, 28, 28);
        const int tf[] = {S_, D_, AIEDNN_ARG_MEAN, AIEDNN_ARG_VARIANCE};
        run("bnorm_train_fwd_f32_8x64x28x28", 0, 4, tf, [&](aiednn_primitive_t *p) {
            return aiednn_batch_normalization_forward_training_create(p, &x, &x, 1e-5f, 0);
        });
        const int tb[] = {AIEDNN_ARG_DIFF_SRC, AIEDNN_ARG_DIFF_DST, S_, AIEDNN_ARG_MEAN, AIEDNN_ARG_VARIANCE};
        run("bnorm_bwd_f32_8x64x28x28", 0, 5, tb,
            [&](aiednn_primitive_t *p) { return aiednn_batch_normalization_backward_create(p, &x, &x, &x, 1e-5f, 0); });
        const int te[] = {AIEDNN_ARG_DIFF_SRC, AIEDNN_ARG_DIFF_DST, S_};
        run("relu_bwd_f32_8x64x28x28", 0, 3, te, [&](aiednn_primitive_t *p) {
            return aiednn_eltwise_backward_create(p, aiednn_eltwise_relu, 0.f, 0.f, &x, &x, &x);
        });
    }
    {
        aiednn_memory_desc_t x = D(aiednn_f32, 2, 256, 1000);
        const int t[] = {AIEDNN_ARG_DIFF_SRC, AIEDNN_ARG_DIFF_DST, D_};
        run("softmax_bwd_f32_256x1000", 0, 3, t, [&](aiednn_primitive_t *p) {
            return aiednn_softmax_backward_create(p, aiednn_softmax_accurate, 1, &x, &x, &x);
        });
    }
    { /* layer norm backward (typed host path) */
        aiednn_memory_desc_t x = D(aiednn_f32, 2, 512, 768);
        const int t[] = {AIEDNN_ARG_DIFF_SRC, AIEDNN_ARG_DIFF_DST, S_, AIEDNN_ARG_MEAN, AIEDNN_ARG_VARIANCE};
        run("layernorm_bwd_f32_512x768", 0, 5, t,
            [&](aiednn_primitive_t *p) { return aiednn_layer_normalization_backward_create(p, &x, &x, &x, 1e-5f, 0); });
    }
    { /* average pooling backward 3x3 s2 p1 (typed host path) */
        aiednn_memory_desc_t x = D(aiednn_f32, 4, 1, 64, 112, 112), y = D(aiednn_f32, 4, 1, 64, 56, 56);
        aiednn_pool_nd_params_t pp = {};
        for (int i = 0; i < 2; i++)
            pp.kernel[i] = 3, pp.strides[i] = 2, pp.padding_l[i] = pp.padding_r[i] = 1;
        const int t[] = {AIEDNN_ARG_DIFF_SRC, AIEDNN_ARG_DIFF_DST, S_};
        run("avgpool_bwd_f32_64x112x112_k3s2", 0, 3, t, [&](aiednn_primitive_t *p) {
            return aiednn_pooling_backward_create(p, aiednn_pooling_avg_exclude_padding, &x, &y, &x, &pp);
        });
    }
}

static void composites(void) {
    { /* f32 2-D conv (gemm+conv over the float GEMM) */
        aiednn_memory_desc_t x = D(aiednn_f32, 4, 1, 64, 56, 56), w = D(aiednn_f32, 4, 64, 64, 3, 3),
                             y = D(aiednn_f32, 4, 1, 64, 56, 56);
        aiednn_conv_params_t cp = {};
        cp.strides[0] = cp.strides[1] = 1, cp.padding_l[0] = cp.padding_l[1] = cp.padding_r[0] = cp.padding_r[1] = 1;
        cp.groups = 1;
        const int t[] = {S_, W_, D_};
        run(
            "conv2d_f32_64x56x56_k3", 2.0 * 64 * 64 * 9 * 56 * 56 / 1e6, 3, t,
            [&](aiednn_primitive_t *p) { return aiednn_convolution_forward_create(p, &x, &w, nullptr, &y, &cp); }, 1);
        aiednn_memory_desc_t dy = D(aiednn_f32, 4, 1, 32, 28, 28),
                             dw = D(aiednn_f32, 4, 16, 32, 4, 4), /* [O, C/G, KH, KW] */
            dd = D(aiednn_f32, 4, 1, 16, 56, 56);
        aiednn_conv_params_t dp = {};
        dp.strides[0] = dp.strides[1] = 2, dp.padding_l[0] = dp.padding_l[1] = dp.padding_r[0] = dp.padding_r[1] = 1;
        dp.groups = 1;
        run(
            "deconv2d_f32_32to16_28to56_k4s2", 2.0 * 32 * 16 * 16 * 28 * 28 / 1e6, 3, t,
            [&](aiednn_primitive_t *p) { return aiednn_deconvolution_forward_create(p, &dy, &dw, nullptr, &dd, &dp); },
            1);
    }
    { /* inner product backward (a training step's fc) */
        aiednn_memory_desc_t x = D(aiednn_f32, 2, 64, 1024), w = D(aiednn_f32, 2, 1000, 1024),
                             y = D(aiednn_f32, 2, 64, 1000);
        const int td[] = {AIEDNN_ARG_DIFF_SRC, W_, AIEDNN_ARG_DIFF_DST};
        run("ip_bwd_data_f32_64x1024x1000", 2.0 * 64 * 1024 * 1000 / 1e6, 3, td,
            [&](aiednn_primitive_t *p) { return aiednn_inner_product_backward_data_create(p, &x, &w, &y); });
        const int tw[] = {S_, AIEDNN_ARG_DIFF_WEIGHTS, AIEDNN_ARG_DIFF_DST};
        run("ip_bwd_weights_f32_64x1024x1000", 2.0 * 64 * 1024 * 1000 / 1e6, 3, tw, [&](aiednn_primitive_t *p) {
            return aiednn_inner_product_backward_weights_create(p, &x, &w, nullptr, &y);
        });
    }
    { /* f32 attention: ViT-S/16 one layer (B 1, H 6, S 197, D 64) */
        aiednn_memory_desc_t q = D(aiednn_f32, 4, 1, 6, 197, 64);
        const int t[] = {S_, AIEDNN_ARG_SRC_1, AIEDNN_ARG_SRC_2, D_};
        run("sdpa_f32_h6_s197_d64", 4.0 * 6 * 197 * 197 * 64 / 1e6, 4, t,
            [&](aiednn_primitive_t *p) { return aiednn_sdpa_forward_create(p, &q, &q, &q, nullptr, &q, 0.125f, 0); });
    }
}

/* f32 pointwise / row primitives (the FVEC mode) */
static void pointwise(void) {
    aiednn_memory_desc_t x = D(aiednn_f32, 2, 1024, 1024);
    const int t[] = {S_, D_};
    run("gelu_tanh_f32_1M", 0, 2, t, [&](aiednn_primitive_t *p) {
        return aiednn_eltwise_forward_create(p, aiednn_eltwise_gelu_tanh, 0.f, 0.f, &x, &x);
    });
    run("swish_f32_1M", 0, 2, t, [&](aiednn_primitive_t *p) {
        return aiednn_eltwise_forward_create(p, aiednn_eltwise_swish, 1.f, 0.f, &x, &x);
    });
    aiednn_memory_desc_t sm = D(aiednn_f32, 2, 1000, 1000);
    run("softmax_f32_1000x1000", 0, 2, t,
        [&](aiednn_primitive_t *p) { return aiednn_softmax_forward_create(p, aiednn_softmax_accurate, 1, &sm, &sm); });
    aiednn_memory_desc_t ln = D(aiednn_f32, 2, 197, 384), g = D(aiednn_f32, 1, 384);
    (void)g;
    const int tl[] = {S_, D_, AIEDNN_ARG_SCALE, AIEDNN_ARG_SHIFT};
    run("layernorm_f32_197x384", 0, 4, tl, [&](aiednn_primitive_t *p) {
        return aiednn_layer_normalization_forward_inference_create(p, &ln, &ln, 1e-5f,
                                                                   AIEDNN_USE_SCALE | AIEDNN_USE_SHIFT);
    });
    const int tb[] = {S_, AIEDNN_ARG_SRC_1, D_};
    run("add_f32_1M", 0, 3, tb,
        [&](aiednn_primitive_t *p) { return aiednn_binary_create(p, aiednn_binary_add, &x, &x, nullptr, &x); });
}

static void misc(void) {
    {
        aiednn_memory_desc_t x = D(aiednn_f32, 1, 1L << 20);
        const int t[] = {S_, D_, AIEDNN_ARG_MASK};
        run("dropout_f32_1M", 0, 3, t,
            [&](aiednn_primitive_t *p) { return aiednn_dropout_forward_create(p, &x, &x, 0.1f, 7); });
    }
    {
        aiednn_memory_desc_t s = D(aiednn_f32, 4, 1, 16, 64, 64), gr = D(aiednn_f32, 4, 1, 64, 64, 2),
                             d = D(aiednn_f32, 4, 1, 16, 64, 64);
        const int t[] = {S_, AIEDNN_ARG_GRID, D_};
        run("grid_sample_bilinear_f32_16x64x64", 0, 3, t,
            [&](aiednn_primitive_t *p) { return aiednn_grid_sample_forward_create(p, &s, &gr, &d, 0); });
    }
}

int main() {
    g_pool = (unsigned char *)malloc(NSLOT * SLOT_BYTES);
    g_save = (unsigned char *)malloc(SLOT_BYTES);
    if (!g_pool || !g_save) {
        printf("PBENCH_FAIL pool\n");
        return 1;
    }
    fill_pool();
    spatial();
    recurrent();
    weight_formats();
    training();
    composites();
    pointwise();
    misc();
    /* the board runner stops at the first "PASS:" / "FAIL:": only the last line may carry one */
    printf("PBENCH done fail=%d\n%s: parity bench\n", g_fail, g_fail ? "FAIL" : "PASS");
    free(g_pool), free(g_save);
    return g_fail;
}
