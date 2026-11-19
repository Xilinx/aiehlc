/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * resnet18_app -- int8 ResNet-18, float image -> logits, through aiednn.
 *
 * Every convolution (stem 7x7, 16 3x3, 3 fused 1x1 downsamples) and the fc are
 * aiednn primitives: the app describes the tensors and aiednn's dispatch picks
 * an implementation per layer from what the builder linked in (on VEK280: the
 * AIE int8 GEMM block behind im2col / blocked matmul; ref: otherwise). The
 * integer epilogues between them -- input quantize, requantize, residual
 * combine, maxpool, avgpool -- are the app's, written exactly as
 * resnetengine/gen_engine.py's numpy model, which is checked against TVM's
 * own dump of every layer when the data is generated.
 *
 * Self-checks: every layer's uint8 output against that model (r18_gold.bin),
 * and the 1000 float logits bit-for-bit against TVM's (r18_logits.bin).
 *
 *   python3 example/aiednn/resnet18/gen_resnet18.py      # -> worklocal/aiednn/resnet18_data
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/resnet18/resnet18_app.cc \
 *       --headers worklocal/aiednn/resnet18_data --target x86 --run
 *   ... --target vek280 --run --board galerina13 --vitis-settings <.../settings64.sh>
 *
 * doc/design/aiednn.md.
 ******************************************************************************/
#include "aiednn.h"
#include "resnet18_net.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define R18_INCBIN(sym, path)                                                                                          \
    __asm__(".section .rodata\n.balign 64\n.global " #sym "\n" #sym ":\n.incbin \"" path "\"\n.previous\n")
R18_INCBIN(r18_weights, R18_WEIGHTS_PATH);
R18_INCBIN(r18_dense, R18_DENSE_PATH);
R18_INCBIN(r18_image, R18_IMAGE_PATH);
R18_INCBIN(r18_logits, R18_LOGITS_PATH);
R18_INCBIN(r18_gold, R18_GOLD_PATH);
extern "C" const int8_t r18_weights[], r18_dense[];
extern "C" const uint8_t r18_gold[];
extern "C" const float r18_image[], r18_logits[];

// ── timing: the architectural counter, no BSP needed ──────────────────────
#if defined(__aarch64__)
static uint64_t now_ticks(void) {
    uint64_t v;
    __asm__ volatile("isb; mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static double ticks_ms(uint64_t t) {
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return 1e3 * (double)t / (double)f;
}
#else
#include <time.h>
static uint64_t now_ticks(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static double ticks_ms(uint64_t t) { return (double)t / 1e6; }
#endif

// ── buffers (largest: the stem's 64x112x112) ──────────────────────────────
static int8_t g_xpad[3 * 230 * 230 > 64 * 58 * 58 ? 3 * 230 * 230 : 64 * 58 * 58];
static int8_t g_x8[64 * 56 * 56];     // a layer's unpadded int8 input (downsample source)
static int32_t g_acc[64 * 112 * 112]; // conv accumulators
static int32_t g_accd[128 * 28 * 28]; // downsample accumulators
static uint8_t g_v[64 * 112 * 112];   // requantized output
static uint8_t g_act[64 * 56 * 56];   // activation between layers (uint8)
static uint8_t g_res[64 * 56 * 56];   // residual carried to the next layer
static uint8_t g_nres[64 * 56 * 56];
static int g_fail;

static int64_t qmul(int64_t x, int l, long long M, int s) {
    return (x * ((int64_t)1 << l) * (int64_t)M + ((int64_t)1 << (s - 1))) >> s;
}

// v = clip(((acc + bz) * mult + 2^(sh-1)) >> sh + c0, 0, 255), sh = shift + 31
static void requant(const int32_t *acc, int C, int HW, const int32_t *bz, const int32_t *mult, const int32_t *shift,
                    int c0, uint8_t *v) {
    for (int c = 0; c < C; c++) {
        const int sh = shift[c] + 31;
        for (int i = 0; i < HW; i++) {
            int64_t t = (((int64_t)acc[c * HW + i] + bz[c]) * (int64_t)mult[c] + ((int64_t)1 << (sh - 1))) >> sh;
            t += c0;
            v[c * HW + i] = (uint8_t)(t < 0 ? 0 : t > 255 ? 255 : t);
        }
    }
}

// one conv through aiednn: src [1,C,H,W] s8 (already padded), weights OIHW, dst s32
static int conv(const char *what, const int8_t *x, int C, int H, int W, const int8_t *w, int O, int KH, int S,
                int32_t *y) {
    const int Ho = (H - KH) / S + 1, Wo = (W - KH) / S + 1;
    const int64_t ds[4] = {1, C, H, W}, dw[4] = {O, C, KH, KH}, dd[4] = {1, O, Ho, Wo};
    aiednn_memory_desc_t sm, wm, dm;
    aiednn_memory_desc_init(&sm, 4, ds, aiednn_s8);
    aiednn_memory_desc_init(&wm, 4, dw, aiednn_s8);
    aiednn_memory_desc_init(&dm, 4, dd, aiednn_s32);
    aiednn_primitive_t p;
    aiednn_status_t st = aiednn_convolution_primitive_create(&p, &sm, &wm, &dm, S);
    if (st == aiednn_success) {
        printf("  %-22s %3dx%3dx%3d k%d s%d -> %s\n", what, C, H, W, KH, S, aiednn_primitive_impl_info(p));
        st = aiednn_primitive_execute(p, x, w, y);
        aiednn_primitive_destroy(p);
    }
    if (st != aiednn_success)
        printf("FAIL: %s: %s\n", what, aiednn_status_str(st));
    return st != aiednn_success;
}

static void check_gold(const char *what, const uint8_t *v, long off, long n) {
    long bad = 0;
    for (long i = 0; i < n; i++)
        bad += v[i] != r18_gold[off + i];
    if (bad) {
        printf("  %s: %ld of %ld outputs differ from the model\n", what, bad, n);
        g_fail = 1;
    }
}

// x [C][H][W] uint8 -> g_xpad [C][H+2][W+2] int8 (x - 128), border -128; also g_x8 (unpadded)
static void pad_input(const uint8_t *x, int C, int H, int W) {
    const int Hp = H + 2, Wp = W + 2;
    memset(g_xpad, -128, (size_t)C * Hp * Wp);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                const int8_t v = (int8_t)(x[(c * H + h) * W + w] - 128);
                g_xpad[(c * Hp + h + 1) * Wp + w + 1] = v;
                g_x8[(c * H + h) * W + w] = v;
            }
}

// TVM call 00 (float -> int8) into the stem's zero-point-padded [3][230][230]
static void quantize_image(void) {
    memset(g_xpad, R18_Q_PAD, 3 * 230 * 230);
    for (int c = 0; c < 3; c++)
        for (int h = 0; h < 224; h++)
            for (int w = 0; w < 224; w++) {
                float v = roundf(r18_image[(c * 224 + h) * 224 + w] * R18_Q_SCALE) + R18_Q_ZP;
                v = v > 255.0f ? 255.0f : v < 0.0f ? 0.0f : v;
                g_xpad[(c * 230 + h + 3) * 230 + w + 3] = (int8_t)((int)v - 128);
            }
}

static void stem(void) {
    quantize_image();
    if (conv("stem conv 7x7", g_xpad, R18_STEM_CIN, 230, 230, r18_weights + R18_STEM_W_OFF, R18_STEM_COUT, 7, 2,
             g_acc)) {
        g_fail = 1;
        return;
    }
    requant(g_acc, R18_STEM_COUT, 112 * 112, r18_stem_bz, r18_stem_mult, r18_stem_shift, R18_STEM_C0, g_v);
    for (int c = 0; c < R18_STEM_COUT; c++) // maxpool 3x3 s2 pad 1 (pad value 0: post-ReLU)
        for (int oh = 0; oh < 56; oh++)
            for (int ow = 0; ow < 56; ow++) {
                uint8_t m = 0;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) {
                        const int h = oh * 2 - 1 + i, w = ow * 2 - 1 + j;
                        if (h >= 0 && h < 112 && w >= 0 && w < 112 && g_v[(c * 112 + h) * 112 + w] > m)
                            m = g_v[(c * 112 + h) * 112 + w];
                    }
                g_act[(c * 56 + oh) * 56 + ow] = m;
            }
    check_gold("stem + maxpool", g_act, 0, 64L * 56 * 56);
}

static void layer(int i) {
    const r18_layer_t *L = &r18_layers[i];
    const int HW = L->Ho * L->Wo;
    char what[48];
    pad_input(g_act, L->Cin, L->Hin, L->Hin);
    snprintf(what, sizeof what, "layer %2d (call %2d)", i, L->call);
    if (conv(what, g_xpad, L->Cin, L->Hin + 2, L->Hin + 2, r18_weights + L->w_off, L->Cout, 3, L->S, g_acc)) {
        g_fail = 1;
        return;
    }
    requant(g_acc, L->Cout, HW, L->bz, L->mult, L->shift, L->c0, g_v);
    int have_nres = 0;
    if (L->dw_off >= 0) {
        snprintf(what, sizeof what, "  downsample 1x1");
        if (conv(what, g_x8, L->Cin, L->Hin, L->Hin, r18_weights + L->dw_off, L->Cout, 1, 2, g_accd)) {
            g_fail = 1;
            return;
        }
        requant(g_accd, L->Cout, HW, L->dbz, L->dmult, L->dshift, L->dc0, g_nres);
        have_nres = 1;
    }
    if (L->grab) { // the residual of the next layer is this layer's input
        memcpy(g_nres, g_act, (size_t)L->Cin * L->Hin * L->Hin);
        have_nres = 1;
    }
    if (L->comb)
        for (int c = 0; c < L->Cout; c++)
            for (int k = 0; k < HW; k++) {
                const int idx = c * HW + k;
                const int64_t t = qmul((int64_t)g_v[idx] - L->szp, L->sl, L->sM, L->ss) +
                                  qmul((int64_t)g_res[idx] - L->rzr, L->rl, L->rM, L->rs);
                g_v[idx] = (uint8_t)(t < 0 ? 0 : t > 255 ? 255 : t);
            }
    memcpy(g_act, g_v, (size_t)L->Cout * HW);
    if (have_nres)
        memcpy(g_res, g_nres, (size_t)L->Cout * HW);
    snprintf(what, sizeof what, "layer %d (call %d)", i, L->call);
    check_gold(what, g_act, L->gold_off, (long)L->Cout * HW);
}

// TVM calls 38..40: Q(v << l) per element, 7x7 average, clamp, - 128, fc 512 -> 1000, requant, to float
static int tail(float *logits) {
    const r18_layer_t *L = &r18_layers[R18_NLAYERS - 1];
    static int8_t feat[R18_FEAT];
    static int32_t acc[R18_NCLASS];
    const int HW = L->Ho * L->Wo;
    for (int c = 0; c < R18_FEAT; c++) {
        int32_t sum = 0;
        for (int k = 0; k < HW; k++)
            sum += r18_final_lut[g_act[c * HW + k]];
        int32_t v = sum / HW;
        v = v < 255 ? v : 255;
        feat[c] = (int8_t)((v > 0 ? v : 0) - 128);
    }
    const int64_t ds[2] = {1, R18_FEAT}, dw[2] = {R18_NCLASS, R18_FEAT}, dd[2] = {1, R18_NCLASS};
    aiednn_memory_desc_t sm, wm, dm;
    aiednn_memory_desc_init(&sm, 2, ds, aiednn_s8);
    aiednn_memory_desc_init(&wm, 2, dw, aiednn_s8);
    aiednn_memory_desc_init(&dm, 2, dd, aiednn_s32);
    aiednn_primitive_t p;
    aiednn_status_t st = aiednn_matmul_primitive_create(&p, &sm, &wm, &dm);
    if (st == aiednn_success) {
        printf("  %-22s 1x%d x %dx%d -> %s\n", "fc", R18_FEAT, R18_NCLASS, R18_FEAT, aiednn_primitive_impl_info(p));
        st = aiednn_primitive_execute(p, feat, r18_dense, acc);
        aiednn_primitive_destroy(p);
    }
    if (st != aiednn_success) {
        printf("FAIL: fc: %s\n", aiednn_status_str(st));
        return 1;
    }
    for (int o = 0; o < R18_NCLASS; o++) {
        const int64_t sh = (int64_t)r18_dense_shift[o] + 31;
        int32_t v =
            (int32_t)((((int64_t)acc[o] + r18_dense_bz[o]) * (int64_t)r18_dense_mult[o] + ((int64_t)1 << (sh - 1))) >>
                      sh) +
            R18_DENSE_C0;
        v = v < 255 ? v : 255;
        logits[o] = (float)((v > 0 ? v : 0) - R18_DENSE_ZP) * R18_DENSE_SCALE;
    }
    return 0;
}

int main() {
    printf("=== aiednn ResNet-18 int8, float image -> logits ===\n");
    static float logits[R18_NCLASS];
    const uint64_t t0 = now_ticks();
    stem();
    for (int i = 0; i < R18_NLAYERS && !g_fail; i++)
        layer(i);
    if (!g_fail)
        g_fail |= tail(logits);
    const uint64_t t1 = now_ticks();
    aiednn_print_stats();
    if (g_fail) {
        printf("FAIL: aiednn ResNet-18: a layer differs from the model or a primitive failed\n");
        return 1;
    }
    int top[5] = {-1, -1, -1, -1, -1};
    for (int k = 0; k < 5; k++)
        for (int o = 0; o < R18_NCLASS; o++) {
            int used = 0;
            for (int j = 0; j < k; j++)
                used |= top[j] == o;
            if (!used && (top[k] < 0 || logits[o] > logits[top[k]]))
                top[k] = o;
        }
    printf("top-5: ");
    for (int k = 0; k < 5; k++)
        printf("%d (%.4f)%s", top[k], (double)logits[top[k]], k < 4 ? ", " : "\n");
    const int same = memcmp(logits, r18_logits, sizeof logits) == 0;
    printf("inference %.1f ms (float image -> logits)\n", ticks_ms(t1 - t0));
    if (!same) {
        printf("FAIL: aiednn ResNet-18 logits differ from TVM's\n");
        return 1;
    }
    printf("PASS: aiednn ResNet-18, all %d logits bit-identical to TVM, top-1 class %d\n", R18_NCLASS, top[0]);
    return 0;
}
