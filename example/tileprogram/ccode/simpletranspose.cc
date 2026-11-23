/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "simpleops_load.h"
// STATUS (2026-11-15): passes the kernel analysis (`build/aiehlc-convtile trace` this file);
// the lowering does not support this op shape yet, so aiehlc.sh stops with
// "the core window is ... but the kernel's blocks (#routing.kernel_block) are ...".
#define OPS_T 256 // y = x^T, int8[256, 256]
// Data movement: transpose on the 4x4 mesh by redundant broadcast. Mesh row pm
// owns x rows 64 pm..; each 16-row strip is broadcast to the row's 4 cores, and
// core (pm, pn) transposes its 64-column quarter of it into a 64 x 16 block of y
// (rows 64 pn.., columns r0..): the DMA scatters it with the 256-byte row pitch.
// Walkthrough: tutorial/aieprogram/04_data_movement/data_movement.md
__global__ void transpose(aie::port<input_window_int8 *, aie::shape<OPS_T, OPS_T>> x,
                          aie::port<output_window_int8 *, aie::shape<OPS_T, OPS_T>> y) {
    const int tr = 16;                             // strip: 16 rows of x = 16 columns of y
    const int rows = OPS_T / aie::num_programs(0); // 64 x rows per mesh row
    const int qc = OPS_T / aie::num_programs(1);   // 64 x columns per core
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / tr; i++) {
        const int r0 = pm * rows + i * tr;
        const int8_t *a = aie::load_buffer(x, {r0, 0}, {tr, OPS_T});     // a[r][c] = x[r0 + r][c]
        int8_t *o = aie::store_buffer(y, {pn * qc, r0}, {qc, tr});       // o[c][r] = y[pn*qc + c][r0 + r]
        for (int r = 0; r < tr; r++)
            for (int c = 0; c < qc; c++)
                o[c * tr + r] = a[r * OPS_T + pn * qc + c];
        aie::release(x);
        aie::commit(y);
    }
}
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 4);
    const long n = (long)OPS_T * OPS_T;
    int8_t *x = (int8_t *)__Runtime_Alloc(n), *y = (int8_t *)__Runtime_Alloc(n), *ref = (int8_t *)malloc(n);
    ops_fill(x, n, 44);
    memset(y, 0, n);
    XTime t0, t1;
    XTime_GetTime(&t0);
    transpose<<<mesh>>>(x, y);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", ops_ms(t0, t1));
    for (int r = 0; r < OPS_T; r++)
        for (int c = 0; c < OPS_T; c++)
            ref[(long)c * OPS_T + r] = x[(long)r * OPS_T + c];
    return ops_compare("y", y, ref, OPS_T, OPS_T) ? 1 : 0;
}
