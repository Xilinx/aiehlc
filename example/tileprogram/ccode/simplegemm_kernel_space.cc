/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hand-written GemmSpaces + block-load GEMM.
 *
 *   C[64, 64] = A[64, 256] x B[64, 256]^T, int8, on the 4x4 mesh
 *
 * The same kernel body as simplegemm_kernel_port.cc, but the DMA is written by
 * hand: one GemmSpace per port, in the explicit form of simplematmul2.cc.
 * aiehlc traces the body's aie::load_buffer / store_buffer blocks and requires
 * every GemmSpace to give exactly that DMA (convautotile verifySpaceKernels);
 * a conflict stops the build naming the field, e.g.
 *   a (GemmSpace RowBA) .d2.tile_size = 32, the kernel's blocks give 64
 * pp_depth / l1_budget / pad and the GlobalPolicy are not compared.
 *
 * Note A's GemmSpace: d1 and d2 both say tile_round = 4, yet d1 is the mesh
 * split (M over the 4 mesh rows) and d2 the on-core loop (4 K chunks). The
 * GemmSpace alone does not say which is which -- the kernel's blocks do, and
 * the check compares against them instead of guessing.
 *
 * Status (palmyra 213, 2026-11-10): runs, 18.5 ms, PASS (all 4096 outputs
 * bit-exact). It hung until three compiler fixes; see
 * doc/design/trace_to_routing.md, next step 3.
 ******************************************************************************/
#include "simplegemm_load.h"
// A[64, 256]: 16 rows per mesh row (step 16), 4 K chunks of 64 per core;
// the same rows for every core of a mesh row.
constexpr aie::GemmSpace RowBA = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = 64, .tile_round = 4, .tile_size = 16, .stride = 16},   // M: mesh rows
    .d2 = {.fullsize = 256, .tile_round = 4, .tile_size = 64, .stride = 64}}; // K: 4 chunks per core
// B[64, 256]: 16 rows per mesh column, 4 K chunks; the same rows for every
// core of a mesh column.
constexpr aie::GemmSpace ColBB = {
    .policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = 64, .tile_round = 4, .tile_size = 16, .stride = 16},   // N: mesh cols
    .d2 = {.fullsize = 256, .tile_round = 4, .tile_size = 64, .stride = 64}}; // K: 4 chunks per core
// C[64, 64]: one 16x16 block per core, gathered (rows over mesh rows, columns
// over mesh columns).
constexpr aie::GemmSpace LtoR_Merge = {.policy = {.map = {.layout = aie::Layout::Row,
                                                          .merge_order = aie::Flow::LeftToRight,
                                                          .mesh_tiling_group1_dim = 1,  // d1 (M) across mesh rows
                                                          .mesh_tiling_group2_dim = 2}, // d2 (N) across mesh cols
                                                  .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                                  .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                       .d1 = {.fullsize = 64, .tile_round = 4, .tile_size = 16, .stride = 16},
                                       .d2 = {.fullsize = 64, .tile_round = 4, .tile_size = 16}};

constexpr aie::GlobalPolicy gemm_policy = {.fullconnect_auto = 1};
__global__(gemm_policy) void gemm(aie::port<input_window_int8 *, RowBA> a, aie::port<input_window_int8 *, ColBB> b,
                                  aie::port<output_window_int8 *, LtoR_Merge> c) {
    const int tm = 16, tn = 16, tk = 64; // C block per core, K chunk per round
    const int pm = aie::program_id(0);   // mesh row: which 16 rows of C
    const int pn = aie::program_id(1);   // mesh col: which 16 columns of C
    int8_t *y = aie::store_buffer(c, {pm * tm, pn * tn}, {tm, tn});
    int acc[tm * tn];
    for (int i = 0; i < tm * tn; i++)
        acc[i] = 0;
    for (int j = 0; j < 256 / tk; j++) { // K chunks
        const int8_t *x = aie::load_buffer(a, {pm * tm, j * tk}, {tm, tk});
        const int8_t *w = aie::load_buffer(b, {pn * tn, j * tk}, {tn, tk});
        for (int m = 0; m < tm; m++)
            for (int n = 0; n < tn; n++) {
                int s = 0;
                for (int k = 0; k < tk; k++)
                    s += x[m * tk + k] * w[n * tk + k];
                acc[m * tn + n] += s;
            }
        aie::release(a);
        aie::release(b);
    }
    for (int i = 0; i < tm * tn; i++)
        y[i] = (int8_t)acc[i];
    aie::commit(c);
}

int main() {
    printf("=== Hand-written GemmSpace GEMM (block loads) on AIE %dx%d ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] x B^T[%dx%d], int8\n", GEMM_M, GEMM_N, GEMM_M, GEMM_K, GEMM_K, GEMM_N);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    int8_t *A = (int8_t *)__Runtime_Alloc(GEMM_M * GEMM_K);
    int8_t *B = (int8_t *)__Runtime_Alloc(GEMM_N * GEMM_K);
    int8_t *C = (int8_t *)__Runtime_Alloc(GEMM_M * GEMM_N);
    gemm_init(A, B, C);
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm<<<mesh>>>(A, B, C);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0);
    const int mismatches = gemm_verify(A, B, C);
    free(A);
    free(B);
    free(C);
    return mismatches ? 1 : 0;
}
