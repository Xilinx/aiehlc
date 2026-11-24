/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// emu_conv2d_tensor.cc -- run the GENERATED kernel of the tensor-form
// simpleconv2dkernel.cc on x86 for every core of the 4x4 mesh, end to end:
// the host's own staging (aiehlc_stage: zero border + aligned channels, taken
// from `aiehlc-convtile trace --emit`), windows served block by block from the
// staged buffers exactly as the kernel's derived descriptors (aiehlc_desc_<port>,
// the numbers the port DMAs are built from) say, outputs scattered back, and
// the example's own golden check (simpleconv2dkernel.h verify) on the REAL
// [224,224,3] tensors. Tiling-independent: whatever aie::autotune picked.
//
//   bash example/tileprogram/ccode/simpleconv2ddebug/emu_conv2d_tensor.sh \
//       example/tileprogram/ccode/simpleconv2dkernel.cc aout/worklocal
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" void *__Runtime_Alloc(size_t n) { return calloc(n, 1); }
#include "simpleconv2dkernel.h"
#include "tensors.inc" // the example's aie::tensor aliases (+ the host prelude)
static int g_blke = 0, g_row = 0, g_col = 0;
static void klog(const char *tag, int) { g_blke += !strcmp(tag, "BLKE"); }
static unsigned get_coreid() { return ((unsigned)(3 + g_col) << 16) | (unsigned)(3 + g_row); }
struct Win {
    int8_t *base;     // the staged tensor the DMA reads / writes
    int dims[4];      // its extents
    const void *desc; // aiehlc_blk_desc of the port
    int round;
    std::vector<int8_t> buf;
};
using input_window_int8 = Win;
using output_window_int8 = Win;
static void serve(Win *w, bool write);
static void *acquire_input_window(Win *w) {
    serve(w, false);
    return w->buf.data();
}
static void release_input_window(Win *w) { w->round++; }
static void *acquire_output_window(Win *w) {
    serve(w, false); // sizes the block (contents are overwritten)
    return w->buf.data();
}
static void release_output_window(Win *w) {
    serve(w, true);
    w->round++;
}
#define KERNEL_COMPILE
#include "conv2d.cc"
// Block `round` of this core: offset = base_p0*p0 + base_p1*p1 + step_i*i + step_j*j.
static void serve(Win *w, bool write) {
    const aiehlc_blk_desc &d = *(const aiehlc_blk_desc *)w->desc;
    const int per = d.rounds_i * d.rounds_j, r = w->round % per, i = r / d.rounds_j, j = r % d.rounds_j;
    int off[4] = {0, 0, 0, 0}, sh[4] = {1, 1, 1, 1}, dm[4] = {1, 1, 1, 1};
    for (int k = 0; k < d.rank; k++) {
        off[k] = d.base_p0[k] * g_row + d.base_p1[k] * g_col + d.step_i[k] * i + d.step_j[k] * j;
        sh[k] = d.shape[k];
        dm[k] = w->dims[k];
    }
    w->buf.resize((size_t)sh[0] * sh[1] * sh[2] * sh[3]);
    long b = 0;
    for (int a0 = 0; a0 < sh[0]; a0++)
        for (int a1 = 0; a1 < sh[1]; a1++)
            for (int a2 = 0; a2 < sh[2]; a2++)
                for (int a3 = 0; a3 < sh[3]; a3++, b++) {
                    long t = (((long)(off[0] + a0) * dm[1] + off[1] + a1) * dm[2] + off[2] + a2) * dm[3] + off[3] + a3;
                    if (write)
                        w->base[t] = w->buf[b];
                    else
                        w->buf[b] = w->base[t];
                }
}
int main() {
    auto in = aie::alloc<Ifm>();
    auto wt = aie::alloc<Wgt>();
    auto out = aie::alloc<Ofm>();
    init_input(in);
    init_filter(wt);
#include "staging.inc" // int8_t *aiehlc_0_x_t = aiehlc_stage(in, ...); ...
    Win x{aiehlc_0_x_t, {STAGED_X}, &aiehlc_desc_x_t, 0, {}}, w{aiehlc_0_w_t, {STAGED_W}, &aiehlc_desc_w_t, 0, {}},
        y{aiehlc_0_y_t, {STAGED_Y}, &aiehlc_desc_y_t, 0, {}};
    for (g_row = 0; g_row < 4; g_row++)
        for (g_col = 0; g_col < 4; g_col++) {
            x.round = w.round = y.round = 0;
            conv2d(&x, &w, &y);
        }
    printf("BLKE=%d\n", g_blke);
    return verify(in, wt, out) || g_blke;
}
