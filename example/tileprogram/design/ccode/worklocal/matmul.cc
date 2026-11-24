/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// User-provided compute kernel (extracted from __global__ function)
// C_tile[4x4] = A_tile[4x16] * B_tile^T[4x16], int8
// B transposed: C[i][j] = dot(A_row[i,:], B_row[j,:])
//
// Matches pipeline: BUF_SZ=8 (32 bytes), 2 symmetric iterations
// Output DMA BD len=2 (8 bytes per cycle), ping-pong = 16 bytes total
// Iter 0: cache A0,B0 (2x16 each), write C[0:1,0:1] + zeros -> 8 bytes
// Iter 1: compute C[2:3,0:1] and C[2:3,2:3] -> 8 bytes
#define K_DIM 16

void matmul(input_window_int8 *window_in_0, input_window_int8 *window_in_1, output_window_int8 *window_out_0) {

    // Local cache for round-0 data
    int8_t cache_A[2 * K_DIM]; // 2 x 16 = 32 bytes
    int8_t cache_B[2 * K_DIM]; // 2 x 16 = 32 bytes

    // ===== Iter 0: read A[0:1,:], B[0:1,:], cache, write partial output =====
    int8_t *A0 = (int8_t *)acquire_input_window(window_in_0);
    int8_t *B0 = (int8_t *)acquire_input_window(window_in_1);

    // Cache for use in iter 1
    for (int i = 0; i < 2 * K_DIM; i++) {
        cache_A[i] = A0[i];
        cache_B[i] = B0[i];
    }

    // Write 8 bytes: C[0:1, 0:1] = A0 * B0^T, columns 2-3 = 0
    {
        int8_t *out = acquire_output_window(window_out_0);
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int16_t sum = 0;
                for (int k = 0; k < K_DIM; k++) {
                    sum += (int16_t)A0[i * K_DIM + k] * (int16_t)B0[j * K_DIM + k];
                }
                if (sum > 127)
                    sum = 127;
                else if (sum < -128)
                    sum = -128;
                out[i * 4 + j] = (int8_t)sum;
            }
            for (int j = 2; j < 4; j++) {
                out[i * 4 + j] = 0;
            }
        }
        release_output_window(window_out_0);
    }

    release_input_window(window_in_0);
    release_input_window(window_in_1);

    // ===== Iter 1: read A[2:3,:], B[2:3,:], compute remaining output =====
    int8_t *A1 = (int8_t *)acquire_input_window(window_in_0);
    int8_t *B1 = (int8_t *)acquire_input_window(window_in_1);

    // Write 8 bytes: C[2:3, 0:1] + C[2:3, 2:3]
    {
        int8_t *out = acquire_output_window(window_out_0);
        // C[2:3, 0:1] = A1 * cached_B0^T
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int16_t sum = 0;
                for (int k = 0; k < K_DIM; k++) {
                    sum += (int16_t)A1[i * K_DIM + k] * (int16_t)cache_B[j * K_DIM + k];
                }
                if (sum > 127)
                    sum = 127;
                else if (sum < -128)
                    sum = -128;
                out[i * 4 + j] = (int8_t)sum;
            }
        }
        // C[2:3, 2:3] = A1 * B1^T
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int16_t sum = 0;
                for (int k = 0; k < K_DIM; k++) {
                    sum += (int16_t)A1[i * K_DIM + k] * (int16_t)B1[j * K_DIM + k];
                }
                if (sum > 127)
                    sum = 127;
                else if (sum < -128)
                    sum = -128;
                out[i * 4 + j + 2] = (int8_t)sum;
            }
        }
        release_output_window(window_out_0);
    }

    release_input_window(window_in_0);
    release_input_window(window_in_1);
}
