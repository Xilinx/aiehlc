/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Matrix Multiplication (Parameterized Kernel API)
 */
#include "simplematmul.h"
#pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)
constexpr aie::SpatialPolicy RowBC = {
    .pattern = aie::Pattern::Broadcast, .distribution = aie::Layout::Row, .pp_depth = 2, .max_buffer_bytes = 4096};
constexpr aie::SpatialPolicy ColBC = {
    .pattern = aie::Pattern::Broadcast, .distribution = aie::Layout::Col, .pp_depth = 2, .max_buffer_bytes = 4096};
constexpr aie::SpatialPolicy LtoR_Merge = {.pattern = aie::Pattern::Gather,
                                           .distribution = aie::Layout::Row,
                                           .merge_order = aie::Flow::LeftToRight,
                                           .pp_depth = 2,
                                           .max_buffer_bytes = 4096};
#define DEBUG_OUTPUT_ORDER 0
__global__ void matmul(aie::port<input_window_int8 *, RowBC> win_a, aie::port<input_window_int8 *, ColBC> win_b,
                       aie::port<output_window_int8 *, LtoR_Merge> win_c) {

    // Compiler-resolved tiling parameters
    const int tile_rows = aie::get_tile_rows();
    const int tile_cols = aie::get_tile_cols();
    const int k_dim = aie::get_k_dim();
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);

    // Derived per-round sizes
    const int rows_per_round = buf_sz_a / k_dim;
    const int cols_per_round = buf_sz_b / k_dim;

#if DEBUG_OUTPUT_ORDER
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    int8_t tag = (int8_t)((row & 0x7) | ((col & 0x7) << 3));
    klog("DEBUG", 3);
#endif

    // Local buffers
    int8_t all_A[tile_rows * k_dim];
    int8_t local_out[tile_rows * tile_cols];

    // ===== Phase 1: Receive and cache all A chunks =====
    for (int ra = 0; ra < num_a_rounds; ra++) {
        int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
        for (int i = 0; i < buf_sz_a; i++)
            all_A[ra * buf_sz_a + i] = A_ptr[i];
        release_input_window(win_a);
    }

#if DEBUG_OUTPUT_ORDER
    for (int i = 0; i < 16; i++) {
        klog("A0  ", (int32_t)all_A[i]);
    }
#endif

    // ===== Phase 2: Stream B, compute all sub-blocks =====
    for (int rb = 0; rb < num_b_rounds; rb++) {
        int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);

#if DEBUG_OUTPUT_ORDER
        if (rb == 0) {
            for (int i = 0; i < 16; i++) {
                klog("B0  ", (int32_t)B_ptr[i]);
            }
        }
#endif

        for (int ra = 0; ra < num_a_rounds; ra++) {
            for (int i = 0; i < rows_per_round; i++) {
                for (int j = 0; j < cols_per_round; j++) {
                    int16_t sum = 0;
                    for (int k = 0; k < k_dim; k++)
                        sum += (int16_t)all_A[(ra * rows_per_round + i) * k_dim + k] * (int16_t)B_ptr[j * k_dim + k];
                    if (sum > 127)
                        sum = 127;
                    else if (sum < -128)
                        sum = -128;
#if DEBUG_OUTPUT_ORDER
                    if (j == 0)
                        sum = tag | (((ra * num_b_rounds + rb) & 0x3) << 6);
#endif
                    local_out[(ra * rows_per_round + i) * tile_cols + rb * cols_per_round + j] = (int8_t)sum;
                }
            }
        }

        release_input_window(win_b);
    }

    // ===== Phase 3: Output =====
    for (int rc = 0; rc < num_c_rounds; rc++) {
        int8_t *out = (int8_t *)acquire_output_window(win_c);
        for (int i = 0; i < buf_sz_c; i++)
            out[i] = local_out[rc * buf_sz_c + i];
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

__global__ void mul2(aie::port<input_window_int8 *, RowBC> win_a, aie::port<input_window_int8 *, ColBC> win_b,
                     aie::port<output_window_int8 *, LtoR_Merge> win_c) {

    // Compiler-resolved tiling parameters
    const int tile_rows = aie::get_tile_rows();
    const int tile_cols = aie::get_tile_cols();
    const int k_dim = aie::get_k_dim();
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);

    // Derived per-round sizes
    const int rows_per_round = buf_sz_a / k_dim;
    const int cols_per_round = buf_sz_b / k_dim;

#if DEBUG_OUTPUT_ORDER
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    int8_t tag = (int8_t)((row & 0x7) | ((col & 0x7) << 3));
    klog("DEBUG", 3);
#endif

    // Local buffers
    int8_t all_A[tile_rows * k_dim];
    int8_t local_out[tile_rows * tile_cols];

    // ===== Phase 1: Receive and cache all A chunks =====
    for (int ra = 0; ra < num_a_rounds; ra++) {
        int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
        for (int i = 0; i < buf_sz_a; i++)
            all_A[ra * buf_sz_a + i] = A_ptr[i];
        release_input_window(win_a);
    }

#if DEBUG_OUTPUT_ORDER
    for (int i = 0; i < 16; i++) {
        klog("A0  ", (int32_t)all_A[i]);
    }
#endif

    // ===== Phase 2: Stream B, compute all sub-blocks =====
    for (int rb = 0; rb < num_b_rounds; rb++) {
        int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);

#if DEBUG_OUTPUT_ORDER
        if (rb == 0) {
            for (int i = 0; i < 16; i++) {
                klog("B0  ", (int32_t)B_ptr[i]);
            }
        }
#endif

        for (int ra = 0; ra < num_a_rounds; ra++) {
            for (int i = 0; i < rows_per_round; i++) {
                for (int j = 0; j < cols_per_round; j++) {
                    int16_t sum = 0;
                    for (int k = 0; k < k_dim; k++)
                        sum += (int16_t)all_A[(ra * rows_per_round + i) * k_dim + k] * (int16_t)B_ptr[j * k_dim + k];
                    if (sum > 127)
                        sum = 127;
                    else if (sum < -128)
                        sum = -128;
#if DEBUG_OUTPUT_ORDER
                    if (j == 0)
                        sum = tag | (((ra * num_b_rounds + rb) & 0x3) << 6);
#endif
                    local_out[(ra * rows_per_round + i) * tile_cols + rb * cols_per_round + j] = (int8_t)sum;
                }
            }
        }

        release_input_window(win_b);
    }

    // ===== Phase 3: Output =====
    for (int rc = 0; rc < num_c_rounds; rc++) {
        int8_t *out = (int8_t *)acquire_output_window(win_c);
        for (int i = 0; i < buf_sz_c; i++)
            out[i] = local_out[rc * buf_sz_c + i];
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

// HOST
int main() {
    printf("=== Matrix Multiply with Data Caching on AIE %dx%d Mesh ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8\n", M, N, M, K, K, N);
    // --- Device + mesh ---
    aieSetDevice(0);
    aieArray device;
    // Carve a partition from the AIE array (startCol=0, numCols=HW_COLS).
    // This initializes _dev BEFORE device.alloc() below, so allocations get a
    // valid device instead of NULL. Uses the aieMesh launch path.
    aieMesh mesh = device.partition(HW_ROWS, HW_COLS); // startCol=0
    // aieDim mesh(HW_ROWS, HW_COLS);
    //  --- Allocate host memory ---
    int8_t *A = (int8_t *)malloc(M * K * sizeof(int8_t));
    int8_t *B = (int8_t *)malloc(K * N * sizeof(int8_t));
    int8_t *C = (int8_t *)malloc(M * N * sizeof(int8_t));
    // --- Initialize input matrices ---
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < M * N; i++)
        C[i] = 0;
    // --- Launch kernel on tile mesh ---
    matmul<<<mesh>>>(A, B, C, M, N, K);
    int result = verify_matmul(A, B, C);
    mul2<<<mesh>>>(C, B, A, M, N, K);
    int result2 = verify_matmul(C, B, A);
    //  --- Wait for all partitions and teardown ---
    // device.synchronize();
    // --- Verify output ---
    // --- Cleanup ---
    free(A);
    free(B);
    free(C);
    return result;
}
