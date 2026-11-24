/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// User-provided compute kernel (extracted from __global__ function)
// User macro definitions from source file
#define DEBUG_OUTPUT_ORDER 1

void matmul(input_window_int8 *win_a, input_window_int8 *win_b, output_window_int8 *win_c) {

    // Compiler-resolved tiling parameters
    const int tile_rows = 16;
    const int tile_cols = 16;
    const int eff_k = 64;       // K chunk size per k-round
    const int k_rounds = 4;     // number of K-accumulation rounds
    const int num_a_rounds = 1; // DMA rounds per k-round for A
    const int num_b_rounds = 1; // DMA rounds per k-round for B
    const int num_c_rounds = 1;
    const int buf_sz_a = 1024;
    const int buf_sz_b = 1024;
    const int buf_sz_c = 256;

    // Spatial sub-tile iteration counts (per-port: A->M rounds, B->N rounds)
    const int m_rounds = 4;
    const int n_rounds = 4;

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
    klog("DEBUG", 3);
#endif

    // Local buffers — accum/local_out hold one M-sub-tile strip (tile_rows × data_cols)
    int8_t all_A[tile_rows * eff_k];
    int16_t accum[tile_rows * tile_cols];
    int8_t local_out[tile_rows * tile_cols];

    // ===== M sub-tile loop: each mr gets fresh A data across all k_rounds =====
    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {

        klog("MR  ", (int32_t)mr);
        // Zero accumulators for this M sub-tile
        for (int i = 0; i < tile_rows * tile_cols; i++)
            accum[i] = 0;

        // ===== K-round loop: accumulate partial products =====
        for (int kr = 0; kr < k_rounds; kr++) {
            klog("KRA ", (int32_t)kr);
            // --- Phase 1: Receive and cache A chunk for this (mr, kr) ---
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++) {
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                }
#if DEBUG_OUTPUT_ORDER
                for (int l = 0; l < (buf_sz_a < 8 ? buf_sz_a : 8); l++) {
                    klog("A   ", (int32_t)A_ptr[l]);
                }
#endif
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
                for (int i = 0; i < tile_rows; i++) {
                    for (int j = 0; j < cols_per_round; j++) {
                        int16_t sum = 0;
                        for (int k = 0; k < eff_k; k++) {
                            sum += (int16_t)all_A[i * eff_k + k] * (int16_t)B_ptr[j * eff_k + k];
                        }
                        accum[i * tile_cols + rb * cols_per_round + j] += sum;
                    }
                }

#if DEBUG_OUTPUT_ORDER
                // if (kr == 0 && mr == 0 && rb == 0) {
                klog("B0  ", (int32_t)B_ptr[0]);
                //}
#endif

                release_input_window(win_b);
            }
        } // end k_rounds

        // ===== Saturate accumulators to int8 for this M sub-tile =====
        for (int i = 0; i < tile_rows * tile_cols; i++) {
            int16_t val = accum[i];
            if (val > 127)
                val = 127;
            else if (val < -128)
                val = -128;
            local_out[i] = (int8_t)val;
        }
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            const int rows_per_c_round = buf_sz_c / tile_cols;
            for (int i = 0; i < rows_per_c_round; i++) {
                for (int j = 0; j < tile_cols; j++) {
                    out[i * tile_cols + j] = local_out[rc * buf_sz_c + i * tile_cols + j];
                }
            }
#if DEBUG_OUTPUT_ORDER
            klog("C0 ", (int32_t)out[0]);
#endif

            release_output_window(win_c);
        }
    } // end mr
}
