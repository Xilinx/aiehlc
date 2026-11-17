/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GEMM optimization, step 1: the low-level schedule (tutorial/aieprogram/06_gemm_optimization/1_low_level_schedule.md).
 *   C[2048, 256] = A[2048, 256] x B[256, 256]^T, int8, on the 4x4 mesh.
 * Step 0 (simplegemm_opt0_baseline.cc) plus three schedule options -- same
 * compute, same result, different buffers and DMA:
 *   B {.resident = true}  one 16 KB buffer, sent once, held all launch
 *   A {.depth = 2}        ping-pong strips: the next arrives during compute
 *   C {.depth = 2}        ping-pong output: the last drains during compute
 * Step 2 (simplegemm_dataflow.cc) adds the vector unit.
 ******************************************************************************/
#include "simplegemm_dataflow.h"

#define GD_M 2048
#define GD_N 256
#define GD_K 256
#define GD_TS 16
#ifndef GD_ITERS
#define GD_ITERS 3
#endif

__global__ void gemm_s1(aie::port<input_window_int8 *, aie::IFM, aie::shape<GD_M, GD_K>> a,
                        aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<GD_N, GD_K>> b,
                        aie::port<output_window_int8 *, aie::OFM, aie::shape<GD_M, GD_N>> c) {
    const int cm = GD_M / 4, cn = GD_N / 4; // per core: 512 rows of C, 64 columns
    const int pm = aie::program_id(0);      // mesh row: which rows of C
    const int pn = aie::program_id(1);      // mesh col: which 64 columns of C
    const int8_t *w = aie::load_buffer(b, {pn * cn, 0}, {cn, GD_K}, {.resident = true}); // this column's 64 rows of B
    for (int s = 0; s < cm / GD_TS; s++) {                             // 32 strips of 16 rows
        const int r0 = pm * cm + s * GD_TS;
        const int8_t *x = aie::load_buffer(a, {r0, 0}, {GD_TS, GD_K}, {.depth = 2});
        int8_t *y = aie::store_buffer(c, {r0, pn * cn}, {GD_TS, cn}, {.depth = 2});
        for (int m = 0; m < GD_TS; m++)
            for (int n = 0; n < cn; n++) {
                int acc = 0;
                for (int k = 0; k < GD_K; k++)
                    acc += x[m * GD_K + k] * w[n * GD_K + k];
                y[m * cn + n] = (int8_t)acc; // low byte, like gemm_df_verify
            }
        aie::release(a);
        aie::commit(c);
    }
    aie::release(b);
}

static void gemm_s1_run(int8_t *A, int8_t *B, int8_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, GD_HW_ROWS, GD_HW_COLS);
    gemm_s1<<<mesh>>>(A, B, C);
}

int main() {
    int8_t *A = (int8_t *)__Runtime_Alloc(GD_M * GD_K);
    int8_t *B = (int8_t *)__Runtime_Alloc(GD_N * GD_K);
    int8_t *C = (int8_t *)__Runtime_Alloc(GD_M * GD_N);
    gemm_df_init(A, B, GD_M, GD_N, GD_K);
    memset(C, 0, GD_M * GD_N);
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm_s1_run(A, B, C); // first call: partition init + kernel load
    XTime_GetTime(&t1);
    const double first_ms = gd_ms(t0, t1);
    double sum = 0;
    for (int it = 0; it < GD_ITERS; it++) {
        XTime_GetTime(&t0);
        gemm_s1_run(A, B, C);
        XTime_GetTime(&t1);
        sum += gd_ms(t0, t1);
    }
    const double mean = sum / GD_ITERS;
    printf("PERF SUMMARY gemm_opt1 first=%.3f mean=%.3f gops=%.2f (ms)\n", first_ms, mean,
           2.0 * GD_M * GD_N * GD_K / (mean * 1e-3) / 1e9);
    const long bad = gemm_df_verify(A, B, C, GD_M, GD_N, GD_K);
    free(A);
    free(B);
    free(C);
    return bad ? 1 : 0;
}
