/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kernel-only GEMM with dataflow options: the DMA is derived from the kernel,
 * and three optional load / store options steer it.
 *
 *   C[GD_M, 256] = A[GD_M, 256] x B[256, 256]^T, int8, on the 4x4 mesh,
 *   GD_M = GD_P * 256 (GD_P panels sharing B, all in ONE launch)
 *
 * Same model as simplegemm_kernel_port.cc -- ports declare only their tensors,
 * the tiling lives in the kernel -- plus:
 *   {.resident = true}  B: this core's 64 x 256 block is loaded once and held for
 *                       the whole launch (one buffer); the DMA sends it once
 *                       instead of with every A strip.
 *   {.depth = 2}        A: two core buffers, so the DMA fills the next strip
 *                       while the core computes on this one.
 *   {.tile = {8, 8}}    A: the strip arrives panel-major (8-column panels, each
 *                       with all 16 rows), so every 8x8 aie::mmul A tile is one
 *                       contiguous 64-byte load; the shim DMA does the re-layout.
 * .resident and .depth change only the DMA and the buffers: drop them and the
 * result is the same (B re-sent with every strip, the compiler's buffer depth).
 * .tile changes the bytes the kernel sees, so it goes with the A tile indexing
 * below. `build/aiehlc-convtile trace <this file>` checks every option against
 * the kernel's blocks.
 *
 * B is passed packed in mmul order (simplegemm_dataflow.h, gemm_df_pack_b).
 *
 * main(): one timed first call, GD_ITERS timed steady-state calls, a
 * `PERF SUMMARY` line, then a bit-exact CPU check of the last call's C.
 ******************************************************************************/
#include "simplegemm_dataflow.h"

// GD_P 256x256x256 panels per launch; GD_TS A / C strip rows per round. (Kernel
// macros are plain #defines: aiehlc copies those into the kernel, not #ifndef blocks.)
#define GD_P 8
#define GD_M (GD_P * 256)
#define GD_N 256
#define GD_K 256
#define GD_TS 16
#ifndef GD_ITERS
#define GD_ITERS 5
#endif

__global__ void gemm_df(aie::port<input_window_int8 *, aie::IFM, aie::shape<GD_M, GD_K>> a,
                        aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<GD_N, GD_K>> b,
                        aie::port<output_window_int8 *, aie::OFM, aie::shape<GD_M, GD_N>> c) {
    using MM = aie::mmul<8, 8, 8, int8_t, int8_t>;
    const int cm = GD_M / 4, cn = GD_N / 4;  // per core: cm rows of C, cn columns
    const int kt = GD_K / 8, st = GD_TS / 8; // 8x8 tiles along K / along a strip
    const int pm = aie::program_id(0);       // mesh row: which rows of C
    const int pn = aie::program_id(1);       // mesh col: which 64 columns of C
    const aie::accum<acc32, 64> zero = aie::zeros<acc32, 64>();
    // This column's 64 rows of B, in 8x8 (k x n) tiles: tile (ni, ki) at (ni * kt + ki) * 64.
    const int8_t *w = aie::load_buffer(b, {pn * cn, 0}, {cn, GD_K}, {.resident = true});
    for (int s = 0; s < cm / GD_TS; s++) {
        const int r0 = pm * cm + s * GD_TS;
        // Panel-major strip: 8x8 tile (mi, ki) at (ki * st + mi) * 64.
        const int8_t *x = aie::load_buffer(a, {r0, 0}, {GD_TS, GD_K}, {.depth = 2, .tile = {8, 8}});
        int8_t *y = aie::store_buffer(c, {r0, pn * cn}, {GD_TS, cn});
        for (int mi = 0; mi < st; mi++)
            for (int ni = 0; ni < cn / 8; ni += 2) { // two 8x8 C tiles side by side
                MM c0(zero), c1(zero);
                for (int ki = 0; ki < kt; ki++)
                    chess_prepare_for_pipelining {
                        const aie::vector<int8_t, 64> va = aie::load_unaligned_v<64>(x + (ki * st + mi) * 64, 32);
                        c0.mac(va, aie::load_unaligned_v<64>(w + (ni * kt + ki) * 64, 32));
                        c1.mac(va, aie::load_unaligned_v<64>(w + ((ni + 1) * kt + ki) * 64, 32));
                    }
                // Row r of the two tiles = 16 contiguous bytes of C (low byte of each sum).
                const auto z = aie::interleave_zip(c0.to_vector<int8_t>(0), c1.to_vector<int8_t>(0), 8);
                for (int r = 0; r < 4; r++) {
                    aie::store_v(y + (mi * 8 + r) * cn + ni * 8, z.first.extract<16>(r));
                    aie::store_v(y + (mi * 8 + 4 + r) * cn + ni * 8, z.second.extract<16>(r));
                }
            }
        aie::release(a);
        aie::commit(c);
    }
    aie::release(b);
}

// One launch: the mesh must be declared in the function that launches.
static void gemm_df_run(int8_t *A, int8_t *Bp, int8_t *C) { // the generated launch takes void *
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, GD_HW_ROWS, GD_HW_COLS);
    gemm_df<<<mesh>>>(A, Bp, C);
}

int main() {
    printf("=== Kernel-only GEMM with dataflow options on AIE %dx%d ===\n", GD_HW_ROWS, GD_HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] x B^T[%dx%d], int8 (%d panels per launch)\n", GD_M, GD_N, GD_M, GD_K, GD_K, GD_N,
           GD_P);
    int8_t *A = (int8_t *)__Runtime_Alloc(GD_M * GD_K);
    int8_t *B = (int8_t *)__Runtime_Alloc(GD_N * GD_K);
    int8_t *Bp = (int8_t *)__Runtime_Alloc(GD_N * GD_K);
    int8_t *C = (int8_t *)__Runtime_Alloc(GD_M * GD_N);
    gemm_df_init(A, B, GD_M, GD_N, GD_K);
    memset(C, 0, GD_M * GD_N);

    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm_df_pack_b(B, Bp, GD_N, GD_K);
    XTime_GetTime(&t1);
    const double pack_ms = gd_ms(t0, t1);

    // First call: partition init, kernel load and the first configuration.
    XTime_GetTime(&t0);
    gemm_df_run(A, Bp, C);
    XTime_GetTime(&t1);
    const double first_ms = gd_ms(t0, t1);

    unsigned long long ph0[4] = {0, 0, 0, 0}, wio0 = 0;
    unsigned int n0[4], nw0;
    __Runtime_phase_cycles(ph0, n0);
    __Runtime_wait_io_cycles(&wio0, &nw0);
    double sum = 0, lo = 1e30, hi = 0;
    for (int it = 0; it < GD_ITERS; it++) {
        XTime_GetTime(&t0);
        gemm_df_run(A, Bp, C);
        XTime_GetTime(&t1);
        const double ms = gd_ms(t0, t1);
        sum += ms;
        lo = ms < lo ? ms : lo;
        hi = ms > hi ? ms : hi;
        printf("[gemm_df] PERF iter %d/%d: %.3f ms\n", it + 1, GD_ITERS, ms);
    }
    const double mean = GD_ITERS > 0 ? sum / GD_ITERS : first_ms;
    const double gops = 2.0 * GD_M * GD_N * GD_K / (mean * 1e-3) / 1e9;
    printf("[gemm_df] PERF first call %.3f ms (partition init + kernel load), host B pack %.3f ms (once)\n", first_ms,
           pack_ms);
    printf("[gemm_df] PERF steady state: mean %.3f ms, min %.3f, max %.3f over %d calls -> %.2f GOPS\n", mean, lo, hi,
           GD_ITERS, gops);
    gemm_df_report_phases(ph0, wio0, GD_ITERS > 0 ? GD_ITERS : 1);
    printf("PERF SUMMARY gemm_df first=%.3f mean=%.3f min=%.3f max=%.3f gops=%.2f (ms)\n", first_ms, mean, lo, hi,
           gops);

    const long bad = gemm_df_verify(A, B, C, GD_M, GD_N, GD_K);
    free(A);
    free(B);
    free(Bp);
    free(C);
    return bad ? 1 : 0;
}
