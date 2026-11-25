/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "simpleops_load.h"
// STATUS (2026-11-15): passes the kernel analysis (`build/aiehlc-convtile trace` this file);
// the lowering does not support this op shape yet, so aiehlc.sh stops with
// "the core window is ... but the kernel's blocks (#routing.kernel_block) are ...".
#define OPS_R 256  // x rows
#define OPS_C 1024 // x columns (bytes)
#define SL_R0 32   // y = x[SL_R0 : SL_R0 + 2*64 : 2, SL_C0 : SL_C0 + 2*256 : 2], int8[64, 256]
#define SL_C0 64
// Indexing with compile-time indices: crop + stride-2 subsample, on the 4x4 mesh.
// Crop: a port's blocks are tiled from the tensor's origin, so the HOST passes a
// sub-view -- x + SL_R0 * OPS_C + SL_C0 -- and the kernel declares that view:
// 128 rows, still OPS_C bytes per row (the row pitch in DDR).
// Stride 2: a load is a dense block, so mesh row pm receives the footprint
// (32 x 256) of a 16 x 128 output block, broadcast to its 4 cores, and core
// (pm, pn) makes the 16 x 32 quarter pn of it, picking every other element.
// (A gather with RUNTIME indices cannot be written: see the tutorial.)
// Walkthrough: tutorial/aieprogram/05_indexing/indexing.md
__global__ void slice2(aie::port<input_window_int8 *, aie::shape<128, OPS_C>> xv, // the cropped view
                       aie::port<output_window_int8 *, aie::shape<64, 256>> y) {
    const int yr = 16, yc = 128;                // y block per mesh row and round
    const int qc = yc / aie::num_programs(1);   // 32 y columns per core
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int j = 0; j < 256 / yc; j++) {        // 2 column halves
        const int8_t *a = aie::load_buffer(xv, {2 * yr * pm, 2 * yc * j}, {2 * yr, 2 * yc});
        int8_t *o = aie::store_buffer(y, {yr * pm, yc * j + qc * pn}, {yr, qc});
        for (int r = 0; r < yr; r++)
            for (int c = 0; c < qc; c++)
                o[r * qc + c] = a[(2 * r) * (2 * yc) + 2 * (qc * pn + c)];
        aie::release(xv);
        aie::commit(y);
    }
}
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(OPS_PARTITION, 4, 4);
    const long n = (long)OPS_R * OPS_C;
    int8_t *x = (int8_t *)__Runtime_Alloc(n), *y = (int8_t *)__Runtime_Alloc(64 * 256), *ref = (int8_t *)malloc(64 * 256);
    ops_fill(x, n, 55);
    memset(y, 0, 64 * 256);
    uint64_t t0 = ops_now(), t1;
    slice2<<<mesh>>>(x + SL_R0 * OPS_C + SL_C0, y); // the view starts at the crop origin
    t1 = ops_now();
    printf("[BENCH] aie_wall_ms=%.3f\n", ops_ms(t0, t1));
    for (int r = 0; r < 64; r++)
        for (int c = 0; c < 256; c++)
            ref[r * 256 + c] = x[(long)(SL_R0 + 2 * r) * OPS_C + SL_C0 + 2 * c];
    return ops_compare("y", y, ref, 64, 256) ? 1 : 0;
}
