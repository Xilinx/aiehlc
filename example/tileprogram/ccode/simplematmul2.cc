/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Matrix Multiplication (Parameterized Kernel API)
 */
#include "simplematmul.h"
// #pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)
#pragma aie_debug_level(0 | AIE_KERNEL_CONFIG_TRACE | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)
// Declarative per-tile core trace: mesh/partition-relative (col,row) of the
// compute tile to trace. Repeatable; supports ranges e.g. #pragma aie_trace(1:2, 3).
// #pragma aie_trace((0, 3), (PARAMETER, "win_a"))
// #pragma CONTROL_PLAN_GROUP_REG_WRITE  // enable use control pla nto gorup send config op
// #pragma control_plan_op_control_packet // enable control plan #1. reserve stream switch routing to cooperate with ir
// #2. call control plan init in platform init
#pragma aie_trace((0, 3), (STREAM, "s2mm", 1))
// Composition-based spatial spaces: a generic SpatialPolicy composed with a
// PER-PORT 2D iteration space. Each port describes its OWN matrix via d1/d2:
//   win_a A=[M,K] -> d1 = M-tile,  d2 = K-chunk
//   win_b B=[N,K] -> d1 = N-tile,  d2 = K-chunk
//   win_c C=[M,N] -> d1 = M-tile,  d2 = N-tile
// 3-part orthogonal SpatialPolicy: map (array mapping) / mat (materialization)
// / sched (resource & pipeline). Per-port act/wgt is resolved by input ordinal:
//   win_a (1st input)  -> map.act,  win_b (2nd input) -> map.wgt,
//   win_c (output)     -> Gather + map.merge_order.
constexpr aie::GemmSpace RowBA = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = M, .tile_size = 16, .stride = 16},  // A: M-tile (Partition: stride == size)
    .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}}; // A: K chunk (4 k-rounds for K=256)
constexpr aie::GemmSpace ColBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                  .d1 = {.fullsize = N, .tile_size = 16, .stride = 16},  // B: N-tile
                                  .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}}; // B: K chunk
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = M, .tile_size = 16, .stride = 16},  // C: M-tile
    .d2 = {.fullsize = N, .tile_size = 16, .stride = 16}}; // C: N-tile
#define DEBUG_OUTPUT_ORDER 1
#define DEBUG_NOCOMPUTE 1
//  Per-kernel GLOBAL policy, bound explicitly at the declaration site via
//  __global__(matmul_policy). The <kernel>_policy naming convention still works
//  as a fallback when no explicit binding is given.
//  fullconnect_auto = 0 -> disable the M×N cartesian DMA repeat: A and B are
//  each sent once following the tiling distribution (no host round loop).
constexpr aie::GlobalPolicy matmul_policy = {.fullconnect_auto = 1};
__global__(matmul_policy) void matmul(aie::port<input_window_int8 *, RowBA> win_a,
                                      aie::port<input_window_int8 *, ColBB> win_b,
                                      aie::port<output_window_int8 *, LtoR_Merge> win_c) {

    // Compiler-resolved tiling parameters
    // const int eff_k = aie::get_effective_k();            // K chunk size per k-round
    // const int k_rounds = aie::get_k_rounds();            // number of K-accumulation rounds
    const int k_rounds = aie::get_arg_total_rounds_in_dim(1, win_a);
    const int eff_k = aie::get_arg_per_round_size_in_dim(1, win_a);
    const int eff_k_b = aie::get_arg_per_round_size_in_dim(1, win_b);
    assert(eff_k == eff_k_b);

    const int num_a_rounds = aie::get_num_rounds(win_a); // DMA rounds per k-round for A
    const int num_b_rounds = aie::get_num_rounds(win_b); // DMA rounds per k-round for B
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);

    // Spatial sub-tile iteration counts (per-port: A->M rounds, B->N rounds)
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);

    // const int tile_rows = aie::get_tile_rows();
    // const int tile_cols = aie::get_tile_cols();

    const int tile_rows = aie::get_arg_per_round_size_in_dim(0, win_a);
    const int tile_cols = aie::get_arg_per_round_size_in_dim(0, win_b);

    // Derived per-round sizes (using effective_k, not full k_dim)
    const int rows_per_round = buf_sz_a / eff_k;
    const int cols_per_round = buf_sz_b / eff_k;

    // N-partition width across all nr sub-tiles
    // const int data_cols = n_rounds * tile_cols;

#if DEBUG_OUTPUT_ORDER
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    int8_t tag = (int8_t)((row & 0x7) | ((col & 0x7) << 3));
    klog("DEBUJ", 3);
    // IR-sourced per-round tiling (read from routing.partitiontensor TilingAttr)
    klog("PRA0", (int32_t)aie::get_arg_per_round_size_in_dim(0, win_a));
    klog("PRA1", (int32_t)aie::get_arg_per_round_size_in_dim(1, win_a));
    klog("PRB0", (int32_t)aie::get_arg_per_round_size_in_dim(0, win_b));
    klog("PRB1", (int32_t)aie::get_arg_per_round_size_in_dim(1, win_b));
    klog("PRC0", (int32_t)aie::get_arg_per_round_size_in_dim(0, win_c));
    klog("PRC1", (int32_t)aie::get_arg_per_round_size_in_dim(1, win_c));
    // IR-sourced total on-core rounds over the dim (sibling of per_round_size_in_dim)
    klog("TRA0", (int32_t)aie::get_arg_total_rounds_in_dim(0, win_a)); // expect 16
    klog("TRA1", (int32_t)aie::get_arg_total_rounds_in_dim(1, win_a)); // expect 4
#endif

    // Local buffers — accum/local_out hold one M-sub-tile strip (tile_rows × data_cols)
    int8_t all_A[tile_rows * eff_k];
    int16_t accum[tile_rows * tile_cols];
    int8_t local_out[tile_rows * tile_cols];

    // ===== M sub-tile loop: each mr gets fresh A data across all k_rounds =====
    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
#ifndef DEBUG_NOCOMPUTE
        klog("MR  ", (int32_t)mr);
        // Zero accumulators for this M sub-tile
        for (int i = 0; i < tile_rows * tile_cols; i++)
            accum[i] = 0;
#endif
        // ===== K-round loop: accumulate partial products =====
        for (int kr = 0; kr < k_rounds; kr++) {
            // klog("KRA ", (int32_t)kr);
            //  --- Phase 1: Receive and cache A chunk for this (mr, kr) ---
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
#ifndef DEBUG_NOCOMPUTE
                for (int i = 0; i < buf_sz_a; i++) {
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                }
#endif
#if DEBUG_OUTPUT_ORDER
                for (int l = 0; l < (buf_sz_a < 8 ? buf_sz_a : 8); l++) {
                    klog("A   ", (int32_t)A_ptr[l]);
                }
#endif
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
#ifndef DEBUG_NOCOMPUTE
                for (int i = 0; i < tile_rows; i++) {
                    for (int j = 0; j < cols_per_round; j++) {
                        int16_t sum = 0;
                        for (int k = 0; k < eff_k; k++) {
                            sum += (int16_t)all_A[i * eff_k + k] * (int16_t)B_ptr[j * eff_k + k];
                        }
                        accum[i * tile_cols + rb * cols_per_round + j] += sum;
                    }
                }
#endif

#if DEBUG_OUTPUT_ORDER
                // if (kr == 0 && mr == 0 && rb == 0) {
                klog("B0  ", (int32_t)B_ptr[0]);
                //}
#endif

                release_input_window(win_b);
            }
        } // end k_rounds

        // ===== Saturate accumulators to int8 for this M sub-tile =====
#ifndef DEBUG_NOCOMPUTE
        for (int i = 0; i < tile_rows * tile_cols; i++) {
            int16_t val = accum[i];
            if (val > 127)
                val = 127;
            else if (val < -128)
                val = -128;
            local_out[i] = (int8_t)val;
        }
#endif
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
#ifndef DEBUG_NOCOMPUTE
            const int rows_per_c_round = buf_sz_c / tile_cols;
            for (int i = 0; i < rows_per_c_round; i++) {
                for (int j = 0; j < tile_cols; j++) {
                    out[i * tile_cols + j] = local_out[rc * buf_sz_c + i * tile_cols + j];
                }
            }
#endif
#if DEBUG_OUTPUT_ORDER
            klog("C0 ", (int32_t)out[0]);
#endif

            release_output_window(win_c);
        }
    } // end mr
}

__global__ void mul2(aie::port<input_window_int8 *, RowBA> win_a, aie::port<input_window_int8 *, ColBB> win_b,
                     aie::port<output_window_int8 *, LtoR_Merge> win_c) {

    // Compiler-resolved tiling parameters
    const int tile_rows = aie::get_data_row();
    const int tile_cols = aie::get_data_col();
    const int eff_k = aie::get_effective_k();
    const int k_rounds = aie::get_k_rounds();
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);

    // Spatial sub-tile iteration counts (per-port: A->M rounds, B->N rounds)
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);

    // Derived per-round sizes (using effective_k, not full k_dim)
    const int rows_per_round = buf_sz_a / eff_k;
    const int cols_per_round = buf_sz_b / eff_k;

    // N-partition width across all nr sub-tiles
    const int data_cols = n_rounds * tile_cols;

#if DEBUG_OUTPUT_ORDER
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    int8_t tag = (int8_t)((row & 0x7) | ((col & 0x7) << 3));
    klog("DEBUG", 3);
#endif

    // Local buffers — accum/local_out hold one M-sub-tile strip (tile_rows × data_cols)
    int8_t all_A[tile_rows * eff_k];
    int16_t accum[tile_rows * data_cols];
    int8_t local_out[tile_rows * data_cols];

    // ===== M sub-tile loop: each mr gets fresh A data across all k_rounds =====
    for (int mr = 0; mr < m_rounds; mr++) {

        // Zero accumulators for this M sub-tile
        for (int i = 0; i < tile_rows * data_cols; i++)
            accum[i] = 0;

        // ===== K-round loop: accumulate partial products =====
        for (int kr = 0; kr < k_rounds; kr++) {

            // --- Phase 1: Receive and cache A chunk for this (mr, kr) ---
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                release_input_window(win_a);
            }

#if DEBUG_OUTPUT_ORDER
            if (kr == 0 && mr == 0) {
                for (int i = 0; i < 16; i++) {
                    klog("A0  ", (int32_t)all_A[i]);
                }
            }
#endif

            // --- N sub-tile loop ---
            for (int nr = 0; nr < n_rounds; nr++) {

                // --- Phase 2: Stream B chunk, accumulate partial products ---
                for (int rb = 0; rb < num_b_rounds; rb++) {
                    int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);

#if DEBUG_OUTPUT_ORDER
                    if (kr == 0 && mr == 0 && nr == 0 && rb == 0) {
                        for (int i = 0; i < 16; i++) {
                            klog("B0  ", (int32_t)B_ptr[i]);
                        }
                    }
#endif

                    for (int ra = 0; ra < num_a_rounds; ra++) {
                        for (int i = 0; i < rows_per_round; i++) {
                            for (int j = 0; j < cols_per_round; j++) {
                                int16_t sum = 0;
                                for (int k = 0; k < eff_k; k++)
                                    sum += (int16_t)all_A[(ra * rows_per_round + i) * eff_k + k] *
                                           (int16_t)B_ptr[j * eff_k + k];
                                accum[(ra * rows_per_round + i) * data_cols + nr * tile_cols + rb * cols_per_round +
                                      j] += sum;
                            }
                        }
                    }

                    release_input_window(win_b);
                }
            } // end nr
        } // end k_rounds

        // ===== Saturate accumulators to int8 for this M sub-tile =====
        for (int i = 0; i < tile_rows * data_cols; i++) {
            int16_t val = accum[i];
            if (val > 127)
                val = 127;
            else if (val < -128)
                val = -128;
            local_out[i] = (int8_t)val;
        }

        // ===== Phase 3: Output this M sub-tile, iterating N sub-tiles =====
        for (int nr = 0; nr < n_rounds; nr++) {
            for (int rc = 0; rc < num_c_rounds; rc++) {
                int8_t *out = (int8_t *)acquire_output_window(win_c);
                const int rows_per_c_round = buf_sz_c / tile_cols;
                for (int i = 0; i < rows_per_c_round; i++) {
                    for (int j = 0; j < tile_cols; j++) {
                        out[i * tile_cols + j] =
                            local_out[(rc * rows_per_c_round + i) * data_cols + nr * tile_cols + j];
                    }
                }
#if DEBUG_OUTPUT_ORDER
                if (rc == 0) {
                    for (int l = 0; l < 8; l++) {
                        klog("C0 ", (int32_t)out[l]);
                    }
                }
#endif
                release_output_window(win_c);
            }
        }
    } // end mr
}

// HOST
int main() {
    printf("=== Matrix Multiply with Data Caching on AIE %dx%d Mesh ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8\n", M, N, M, K, K, N);
    // --- Device + mesh ---
    aieSetDevice(0);
    aieArray device;
    // Carve a partition from the AIE array.
    // Fields: {startCol, endCol, startRow, endRow}
    // This partition uses columns [0,3] and rows [0,6], which covers
    // Gen2 NoC shim columns {0,1} — enough for a 4x4 GEMM's ~12 DataIOs.
    aieMesh mesh = device.partition({0, 3, 0, 6}, HW_ROWS, HW_COLS);
    // aieDim mesh(HW_ROWS, HW_COLS);
    //  --- Allocate DMA-capable host memory (cache stays enabled) ---
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t) * 4);
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
    // --- Initialize input matrices ---
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < M * N; i++)
        C[i] = 0;

    // time start
    XTime t_start, t_end;
    XTime_GetTime(&t_start);
    // --- Launch kernel on tile mesh ---
    matmul<<<mesh>>>(A, B, C, M, N, K);
    //
    XTime_GetTime(&t_end);
    double elapsed_ms = 1.0 * (t_end - t_start) / COUNTS_PER_SECOND * 1000.0;
    printf("aie matmul time: %.3f ms\n", elapsed_ms);
    // stlkernel<<mesh>>>(A, B, C);
    //  device.synchronizecpu(C, M * N * sizeof(int8_t) * 4);
#ifndef DEBUG_NOCOMPUTE
    int result = verify_matmul(A, B, C);
#else
    int result = 0;
    printf("test end=----------------------------------%.3f ms\n", elapsed_ms);
#endif
    /// mul2<<<mesh>>>(C, B, A, M, N, K);
    // int result2 = verify_matmul(C, B, A);
    //   --- Wait for all partitions and teardown ---
    // --- Verify output ---
    // --- Cleanup ---
    device.free(A);
    device.free(B);
    device.free(C);
    return result;
}
