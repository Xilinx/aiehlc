/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "simpleops_load.h"
// STATUS (2026-11-15): passes the kernel analysis (`build/aiehlc-convtile trace` this file);
// the lowering does not support this op shape yet, so aiehlc.sh stops with
// "the core window is ... but the kernel's blocks (#routing.kernel_block) are ...".
#define OPS_R 256  // rows
#define OPS_C 1024 // columns (bytes)
// Reduction: s[r] = sum over c of x[r][c], x int8[256, 1024], on the 4x4 mesh by
// redundant broadcast, in two levels:
//   in the array: mesh row pm owns rows 64 pm..; each 16 x 256 block of them is
//     broadcast to the row's 4 cores, and core (pm, pn) sums column chunk pn
//     (columns 256 pn ..) of every row: an int32 PARTIAL sum per row and core;
//   on the host: s[r] = the 4 partial sums of row r.
// y int8[256, 16]: row r's partial from mesh column pn is the int32 at y[r][4 pn..].
// Walkthrough: tutorial/aieprogram/03_reduction/reduction.md
__global__ void rowsum(aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_C>> x,
                       aie::port<output_window_int8 *, aie::shape<OPS_R, 16>> y) {
    const int tr = 16, tc = 256;                    // input block: 16 rows x 256 columns
    const int rows = OPS_R / aie::num_programs(0); // 64 rows per mesh row
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / tr; i++) {           // 4 row strips
        int8_t *o = aie::store_buffer(y, {pm * rows + i * tr, pn * 4}, {tr, 4});
        int acc[tr];
        for (int r = 0; r < tr; r++)
            acc[r] = 0;
        for (int j = 0; j < OPS_C / tc; j++) {      // 4 column chunks, all received
            const int8_t *a = aie::load_buffer(x, {pm * rows + i * tr, j * tc}, {tr, tc});
            if (j == pn)                            // ... this core sums its own
                for (int r = 0; r < tr; r++)
                    for (int c = 0; c < tc; c++)
                        acc[r] += a[r * tc + c];
            aie::release(x);
        }
        for (int r = 0; r < tr; r++) // int32 little-endian
            for (int b = 0; b < 4; b++)
                o[r * 4 + b] = (int8_t)(acc[r] >> (8 * b));
        aie::commit(y);
    }
}
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 4);
    const long n = (long)OPS_R * OPS_C;
    int8_t *x = (int8_t *)__Runtime_Alloc(n), *y = (int8_t *)__Runtime_Alloc(OPS_R * 16);
    int8_t *ref = (int8_t *)malloc(OPS_R * 16);
    ops_fill(x, n, 33);
    memset(y, 0, OPS_R * 16);
    XTime t0, t1;
    XTime_GetTime(&t0);
    rowsum<<<mesh>>>(x, y);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", ops_ms(t0, t1));
    long bad_sum = 0;
    for (int r = 0; r < OPS_R; r++) {
        int total = 0, got = 0;
        for (int q = 0; q < 4; q++) {
            int part = 0;
            for (int c = 0; c < OPS_C / 4; c++)
                part += x[(long)r * OPS_C + q * (OPS_C / 4) + c];
            for (int b = 0; b < 4; b++)
                ref[r * 16 + q * 4 + b] = (int8_t)(part >> (8 * b));
            total += part;
            int32_t v; // the level-2 combine: add the 4 partial sums of the row
            memcpy(&v, &y[r * 16 + q * 4], 4);
            got += v;
        }
        bad_sum += got != total;
    }
    printf("row sums after the host combine: %ld wrong out of %d\n", bad_sum, OPS_R);
    return ops_compare("y", y, ref, OPS_R, 16) || bad_sum ? 1 : 0;
}
