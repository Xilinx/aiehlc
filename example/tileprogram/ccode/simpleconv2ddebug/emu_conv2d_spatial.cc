/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// emu_conv2d_spatial.cc -- run the GENERATED conv2d_spatial kernel of
// simpleconv2d_new_load.cc on x86 for all 16 cores of the 4x4 mesh and compare with
// the CPU reference (scalar_conv2d). The windows serve blocks exactly as the
// host shim BDs stream them (base 51520 B per mesh row, iteration step 12880 B
// per height round, D2 step 224 B per width round, 19 rows x 244 B at pitch
// 920) and the output blocks are scattered like the gather DMA, so a PASS means
// kernel + derived tiling agree before any board run. BLKE counts the
// aie::load_buffer/store_buffer block-check failures (aie_block_io.h).
// Vector kernels run on the AIE API x86 emulation (aie_api_x86emu.h); the
// windows are 32-byte but not 64-byte aligned, like DMA buffers on the core,
// so a vector access the hardware cannot do is reported.
//
//   source script/aiehlc.sh --aie-version 5 --runtime-source-file \
//       ./example/tileprogram/ccode/simpleconv2d_new_load.cc       # -> aout/worklocal/
//   bash example/tileprogram/ccode/simpleconv2ddebug/emu_conv2d_spatial.sh aout/worklocal
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "aie_api_x86emu.h"
static int g_blke = 0, g_core_row = 0, g_core_col = 0;
static void klog(const char *tag, int v) {
    if (!strcmp(tag, "BLKE")) {
        g_blke++;
        if (g_blke < 6)
            printf("klog %s %d\n", tag, v);
    }
}
static unsigned get_coreid() { return ((unsigned)(3 + g_core_col) << 16) | (unsigned)(3 + g_core_row); }
// A DMA buffer: 32-byte aligned, deliberately not 64-byte aligned.
struct WinBuf {
    void *raw = nullptr;
    int8_t *p = nullptr;
    explicit WinBuf(size_t n) {
        if (posix_memalign(&raw, 64, n + 32))
            abort();
        p = (int8_t *)raw + 32;
    }
    ~WinBuf() { free(raw); }
    int8_t *data() { return p; }
};
struct input_window_int8 {
    WinBuf buf;
    int (*fill)(int8_t *, int);
    int round;
};
struct output_window_int8 {
    WinBuf buf;
    void (*drain)(const int8_t *, int);
    int round;
};
static void *acquire_input_window(input_window_int8 *w) {
    w->fill(w->buf.data(), w->round++);
    return w->buf.data();
}
static void release_input_window(input_window_int8 *) {}
static void *acquire_output_window(output_window_int8 *w) { return w->buf.data(); }
static void release_output_window(output_window_int8 *w) { w->drain(w->buf.data(), w->round++); }
#include "simpleconv2d_new_load.h"
static int8_t *g_in, *g_flt, *g_out;
static int fill_a(int8_t *dst, int r) {
    const int i = r / 4, j = r % 4;
    const int8_t *src = g_in + 51520 * g_core_row + 12880 * i + 224 * j;
    for (int row = 0; row < 19; row++)
        memcpy(dst + row * 244, src + row * 920, 244);
    return 0;
}
static int fill_b(int8_t *dst, int) {
    memcpy(dst, g_flt + 3136 * g_core_col, 3136);
    return 0;
}
static void drain_c(const int8_t *src, int r) {
    const int i = r / 4, j = r % 4;
    for (int oh = 0; oh < 7; oh++)
        for (int ow = 0; ow < 28; ow++)
            memcpy(g_out + ((g_core_row * 28 + i * 7 + oh) * 112 + j * 28 + ow) * 64 + g_core_col * 16,
                   src + (oh * 28 + ow) * 16, 16);
}
#define KERNEL_COMPILE
// The generated kernel never sees the example header; keep its macros out.
static const int H_K = K, H_N = N;
#undef M
#undef N
#undef K
#include "conv2d_spatial.cc"
int main() {
    g_in = new int8_t[INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN]();
    g_flt = new int8_t[H_K * H_N]();
    g_out = new int8_t[OUTPUT_H * OUTPUT_W * NUM_FILTERS]();
    for (int h = 0; h < INPUT_H; h++)
        for (int w = 0; w < INPUT_W; w++)
            for (int c = 0; c < INPUT_C; c++)
                g_in[((h + PAD) * INPUT_W_PAD + (w + PAD)) * INPUT_C_ALIGN + c] =
                    (int8_t)((int)(mix32((uint32_t)((h * INPUT_W + w) * INPUT_C + c) + INPUT_SEED) % 9u) - 4);
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int kh = 0; kh < KERNEL_H; kh++)
            for (int kw = 0; kw < KERNEL_W; kw++)
                for (int c = 0; c < INPUT_C; c++)
                    g_flt[f * H_K + (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c] =
                        (int8_t)((int)(mix32((uint32_t)(((f * KERNEL_H + kh) * KERNEL_W + kw) * INPUT_C + c) +
                                             FILTER_SEED) %
                                       3u) -
                                 1);
    for (g_core_row = 0; g_core_row < 4; g_core_row++)
        for (g_core_col = 0; g_core_col < 4; g_core_col++) {
            input_window_int8 a{WinBuf(4636), fill_a, 0}, b{WinBuf(3136), fill_b, 0};
            output_window_int8 c{WinBuf(3136), drain_c, 0};
            conv2d_spatial(&a, &b, &c);
        }
    std::vector<int8_t> ref(OUTPUT_H * OUTPUT_W * NUM_FILTERS);
    scalar_conv2d(g_in, g_flt, ref.data());
    int bad = 0;
    for (size_t i = 0; i < ref.size(); i++)
        bad += ref[i] != g_out[i];
    printf("BLKE=%d mismatches=%d/%zu -> %s\n", g_blke, bad, ref.size(), (bad || g_blke) ? "FAIL" : "PASS");
    return bad || g_blke;
}
