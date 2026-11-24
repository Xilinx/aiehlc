/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Conv2d via Im2col on AIE — Header
 *
 * Conv2d is converted to GEMM using the im2col transformation:
 *   Input[H,W,C] → im2col_matrix[OH*OW, KH*KW*C]  (done by DMA multi-dim BD)
 *   Filter[KH,KW,C,F] → filter_matrix[KH*KW*C, F]  (simple reshape)
 *   Output[OH,OW,F] = im2col_matrix @ filter_matrix  (standard matmul on core)
 *
 * The AIE DMA hardware performs the im2col rearrangement using multi-dimensional
 * buffer descriptors — no explicit im2col buffer is materialized on the host.
 * The kernel running on core tiles is a standard matmul, identical to simplematmul.
 *
 ******************************************************************************/
#include "unistd.h"
#include "xil_cache.h"
#include "xiltimer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ═══════════════════════════════════════════════════════════════════════════
// Conv2d Parameters
//
// Input:  [H=8, W=8, C=1]  — 8x8 single-channel image
// Filter: [KH=3, KW=3, C=1, F=1] — one 3x3 filter
// Stride: 1, Padding: 0
// Output: [OH=6, OW=6, F=1] — 6x6 output
// ═══════════════════════════════════════════════════════════════════════════

// Conv2d spatial parameters
#define INPUT_H 224
#define INPUT_W 224
#define INPUT_C 3       // real (semantic) input channels
#define INPUT_C_ALIGN 4 // channel-layout stride: cin padded with a zero channel
#define KERNEL_H 7
#define KERNEL_W 7
#define NUM_FILTERS 64
#define STRIDE 2
#define PAD 3

// Derived output dimensions
#define OUTPUT_H ((INPUT_H + 2 * PAD - KERNEL_H) / STRIDE + 1) // 112
#define OUTPUT_W ((INPUT_W + 2 * PAD - KERNEL_W) / STRIDE + 1) // 112

// Spatially pre-padded host-buffer dimensions. The CPU reference functions
// (host_im2col / scalar_conv2d) index a DDR buffer that is zero-padded by PAD on
// every spatial border, so window position (oh*S+kh, ow*S+kw) indexes the padded
// buffer directly (real pixel sits at (h+PAD, w+PAD)). This implements true
// padded conv and removes the previous out-of-bounds reads (ih/iw reached
// INPUT_H/W+2*PAD-KERNEL into a buffer that was only [INPUT_H, INPUT_W, ...]).
#define INPUT_H_PAD (INPUT_H + 2 * PAD) // padded input rows
#define INPUT_W_PAD (INPUT_W + 2 * PAD) // padded input cols (row pitch in pixels)

// Im2col → GEMM dimensions (K uses the ALIGNED channel count, INPUT_C_ALIGN)
//   A (im2col matrix): [M, K] = [OH*OW, KH*KW*C_align] = [12544, 196]
//   B (filter matrix): [K, N] = [KH*KW*C_align, F]      = [196, 64]
//   C (output):        [M, N] = [OH*OW, F]              = [12544, 64]
#define M (OUTPUT_H * OUTPUT_W)                 // 12544
#define K (KERNEL_H * KERNEL_W * INPUT_C_ALIGN) // 196 (channel-aligned)
#define N NUM_FILTERS                           // 64

// HW mesh dimensions
#define HW_ROWS 4
#define HW_COLS 4
// Mesh extents the Mapping splits output axes across (aie::split(axis, MESH_ROW)).
#define MESH_ROW HW_ROWS
#define MESH_COL HW_COLS

// ═══════════════════════════════════════════════════════════════════════════
// Host-side im2col (for verification only — on HW this is done by DMA)
//
// Extracts sliding-window patches from input[H][W][C] into a matrix
// of shape [OH*OW, KH*KW*C], stored row-major.
// ═══════════════════════════════════════════════════════════════════════════
static void host_im2col(const int8_t *input, int8_t *im2col_out) {
    // Input is spatially pre-padded [INPUT_H_PAD, INPUT_W_PAD, INPUT_C_ALIGN];
    // each patch is KH*KW*INPUT_C_ALIGN (= K) elements. The spatial-border and
    // channel padding are zero in the input, so they are copied through as zero —
    // keeping im2col_out the GEMM A matrix [M, K]. Window position (oh*S+kh,
    // ow*S+kw) indexes the padded buffer directly (no OOB).
    int idx = 0;
    for (int oh = 0; oh < OUTPUT_H; oh++) {
        for (int ow = 0; ow < OUTPUT_W; ow++) {
            // Extract one patch: KH x KW x INPUT_C_ALIGN elements
            for (int kh = 0; kh < KERNEL_H; kh++) {
                for (int kw = 0; kw < KERNEL_W; kw++) {
                    for (int c = 0; c < INPUT_C_ALIGN; c++) {
                        int ih = oh * STRIDE + kh;
                        int iw = ow * STRIDE + kw;
                        im2col_out[idx++] = input[(ih * INPUT_W_PAD + iw) * INPUT_C_ALIGN + c];
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Naive conv2d (for reference verification)
//
// Computes output[oh][ow][f] = sum over (kh,kw,c) of
//   input[oh*S+kh][ow*S+kw][c] * filter[kh][kw][c][f]
// ═══════════════════════════════════════════════════════════════════════════
static void scalar_conv2d(const int8_t *input, const int8_t *filter, int8_t *output) {
    // Input is spatially pre-padded [INPUT_H_PAD, INPUT_W_PAD, INPUT_C_ALIGN];
    // filter is B^T [N, K] with K = KH*KW*INPUT_C_ALIGN (same buffer/indexing the
    // AIE kernel consumes). The spatial-border and channel padding are zero in
    // both input and filter, so iterating over the padded extents yields the
    // correct padded-conv result without out-of-bounds reads.
    for (int oh = 0; oh < OUTPUT_H; oh++) {
        for (int ow = 0; ow < OUTPUT_W; ow++) {
            for (int f = 0; f < NUM_FILTERS; f++) {
                int16_t acc = 0;
                for (int kh = 0; kh < KERNEL_H; kh++) {
                    for (int kw = 0; kw < KERNEL_W; kw++) {
                        for (int c = 0; c < INPUT_C_ALIGN; c++) {
                            int ih = oh * STRIDE + kh;
                            int iw = ow * STRIDE + kw;
                            int kk = (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c;
                            int8_t iv = input[(ih * INPUT_W_PAD + iw) * INPUT_C_ALIGN + c];
                            int8_t fv = filter[f * K + kk];
                            acc += (int16_t)iv * (int16_t)fv;
                        }
                    }
                }
                if (acc > 127)
                    acc = 127;
                else if (acc < -128)
                    acc = -128;
                // HWC [OH, OW, F]: row H outer, col W middle, channel F inner.
                output[(oh * OUTPUT_W + ow) * NUM_FILTERS + f] = (int8_t)acc;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify AIE conv2d output against CPU reference
//
// Computes naive conv2d on CPU, then compares flat output arrays.
// Both the AIE output and the CPU reference are in HWC layout [OH, OW, F]
// (channel F fastest-varying), so a flat element-wise compare is valid.
// ═══════════════════════════════════════════════════════════════════════════
static int verify_conv2d(const int8_t *input, const int8_t *filter, const int8_t *output_aie) {
    const int total = OUTPUT_H * OUTPUT_W * NUM_FILTERS;
    int8_t ref[OUTPUT_H * OUTPUT_W * NUM_FILTERS];
    int mismatches = 0;
    /*
    for (int i = total - 160; i < total; i++) {
        printf("DEBUG: output_aie[%d] = %d\n", i, output_aie[i]);
    }

    return 0;
    */
    printf("Computing CPU reference conv2d...\n");
    XTime t_start, t_end;
    XTime_GetTime(&t_start);
    scalar_conv2d(input, filter, ref);
    XTime_GetTime(&t_end);
    double elapsed_ms = 1.0 * (t_end - t_start) / COUNTS_PER_SECOND * 1000.0;
    printf("CPU conv2d time: %.3f ms\n", elapsed_ms);

    for (int i = 0; i < total; i++) {
        if (output_aie[i] != ref[i]) {
            int col = output_aie[i] / 10;
            int row = output_aie[i] % 10;
            // if (col >7 || col < 0 || row < 3 || row > 6) {
            //     break;
            // }
            //  HWC [OH, OW, F] decode: oh outer, ow middle, f inner.
            int oh = i / (OUTPUT_W * NUM_FILTERS);
            int ow = (i / NUM_FILTERS) % OUTPUT_W;
            int f = i % NUM_FILTERS;
            printf("MISMATCH output[f=%d,oh=%d,ow=%d] (flat %d): got %d, expected %d\n", f, oh, ow, i, output_aie[i],
                   ref[i]);
            mismatches++;
            if (mismatches > 128)
                break;
        }
    }

    // Print input image (real pixels live at (h+PAD, w+PAD) in the padded buffer)
    printf("\nInput [%dx%dx%d]:\n", INPUT_H, INPUT_W, INPUT_C);
    for (int h = 0; h < INPUT_H; h++) {
        printf("  [");
        for (int w = 0; w < INPUT_W; w++) {
            printf("%4d", input[((h + PAD) * INPUT_W_PAD + (w + PAD)) * INPUT_C_ALIGN]);
            if (w < INPUT_W - 1)
                printf(",");
        }
        printf("]\n");
    }

    // Print filter
    printf("\nFilter [%dx%dx%dx%d]:\n", KERNEL_H, KERNEL_W, INPUT_C, NUM_FILTERS);
    for (int kh = 0; kh < KERNEL_H; kh++) {
        printf("  [");
        for (int kw = 0; kw < KERNEL_W; kw++) {
            // filter is B^T [N, K]; print f=0, c=0 weight at K-index (kh,kw,0)
            printf("%4d", filter[(kh * KERNEL_W + kw) * INPUT_C_ALIGN]);
            if (kw < KERNEL_W - 1)
                printf(",");
        }
        printf("]\n");
    }

    // Print output (HWC: plane f=0 = output_aie[(oh*OW + ow)*NUM_FILTERS + 0])
    for (int f = 0; f < NUM_FILTERS; f++) {
        printf("\nOutput AIE [%dx%dx%d] (f=%d plane):\n", NUM_FILTERS, OUTPUT_H, OUTPUT_W, f);
        for (int oh = 0; oh < OUTPUT_H; oh++) {
            printf("  [");
            if ((oh % 4) != 0) {
                printf("...]\n");
                continue;
            }
            for (int ow = 0; ow < OUTPUT_W; ow++) {
                int var = output_aie[(oh * OUTPUT_W + ow) * NUM_FILTERS + f];
                int col = var / 10;
                int row = var % 10;
                // if (col >7 || col < 0 || row < 3 || row > 6) {
                //     printf("ERROR: output value out of expected range: %d\n", var);
                // }
                // if (oh == 0 && ow == 0) {
                //     printf("DEBUG: rest is all reproduce for debug...\n");
                //     printf("DEBUG: output value at (f=%d,oh=%d,ow=%d) = %d\n", f, oh, ow, var);
                // }
                printf("%4d", output_aie[(oh * OUTPUT_W + ow) * NUM_FILTERS + f]);
                if (ow < OUTPUT_W - 1)
                    printf(",");
            }
            printf("]\n");
        }
        /*
        // Print reference (HWC: plane f=0)
        printf("\nOutput REF [%dx%dx%d] (f=%d plane):\n", NUM_FILTERS, OUTPUT_H, OUTPUT_W, f);
        for (int oh = 0; oh < OUTPUT_H; oh++) {
            printf("  [");
            for (int ow = 0; ow < OUTPUT_W; ow++) {
                printf("%4d", ref[(oh * OUTPUT_W + ow) * NUM_FILTERS + f]);
                if (ow < OUTPUT_W - 1)
                    printf(",");
            }
            printf("]\n");
        }
            */
    }
    if (mismatches == 0)
        printf("PASS: all %d output elements match.\n", total);
    else
        printf("FAIL: %d mismatches out of %d.\n", mismatches, total);

    return mismatches;
}

// ═══════════════════════════════════════════════════════════════════════════
// Verify that host_im2col + matmul == naive conv2d
// (Sanity check that the im2col math is correct, independent of AIE)
// ═══════════════════════════════════════════════════════════════════════════
static int verify_im2col_equivalence(const int8_t *input, const int8_t *filter) {
    // Step 1: im2col
    int8_t im2col_mat[M * K]; // [36, 9]
    host_im2col(input, im2col_mat);

    // Step 2: matmul  C[M,N] = im2col[M,K] * B^T[N,K]
    // Following the simplematmul convention: B is stored in B^T[N][K] layout,
    // i.e. filter[j * K + k]. For the [KH,KW,C,F] filter with F=N:
    //   B^T[j][k] = filter_hwcf[k * N + j] when treating filter as [K, N].
    // For N=1, both layouts are identical: filter[k] in either case.
    int8_t gemm_out[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int16_t sum = 0;
            for (int k = 0; k < K; k++) {
                // B^T[j, k] indexing — same as kernel: B_ptr[j * k_dim + k]
                sum += (int16_t)im2col_mat[i * K + k] * (int16_t)filter[j * K + k];
            }
            if (sum > 127)
                sum = 127;
            else if (sum < -128)
                sum = -128;
            gemm_out[i * N + j] = (int8_t)sum;
        }
    }

    // Step 3: naive conv2d
    int8_t conv_out[OUTPUT_H * OUTPUT_W * NUM_FILTERS];
    scalar_conv2d(input, filter, conv_out);

    // Compare
    int mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        if (gemm_out[i] != conv_out[i]) {
            // printf("IM2COL SANITY FAIL at %d: im2col+matmul=%d, naive=%d\n", i, gemm_out[i], conv_out[i]);
            mismatches++;
        }
    }
    if (mismatches == 0)
        printf("IM2COL SANITY PASS: im2col+matmul == naive conv2d for all %d elements.\n", M * N);

    return mismatches;
}
// ═══════════════════════════════════════════════════════════════════════════
// Deterministic value generator (shared with conv2d_debug.py)
// ═══════════════════════════════════════════════════════════════════════════
#ifndef INPUT_SEED
#define INPUT_SEED 1234u
#endif
#ifndef FILTER_SEED
#define FILTER_SEED 5678u
#endif
static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
// ═══════════════════════════════════════════════════════════════════════════
// Test data: the real [INPUT_H, INPUT_W, INPUT_C] pixels in [-4, 4] and the
// [NUM_FILTERS, KH, KW, INPUT_C] weights in {-1, 0, 1}, written into the
// pre-padded / channel-aligned buffers (border and pad channel stay 0). The hash
// key is the UNPADDED index, so conv2d_debug.py reproduces the same values.
// ═══════════════════════════════════════════════════════════════════════════
static inline void init_conv2d_test_data(int8_t *input, int8_t *filter) {
    memset(input, 0, INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN);
    memset(filter, 0, K * NUM_FILTERS);
    for (int h = 0; h < INPUT_H; h++)
        for (int w = 0; w < INPUT_W; w++)
            for (int c = 0; c < INPUT_C; c++)
                input[((h + PAD) * INPUT_W_PAD + w + PAD) * INPUT_C_ALIGN + c] =
                    (int8_t)((int)(mix32((uint32_t)((h * INPUT_W + w) * INPUT_C + c) + INPUT_SEED) % 9u) - 4);
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int t = 0; t < KERNEL_H * KERNEL_W; t++)
            for (int c = 0; c < INPUT_C; c++)
                filter[f * K + t * INPUT_C_ALIGN + c] =
                    (int8_t)((int)(mix32((uint32_t)((f * KERNEL_H * KERNEL_W + t) * INPUT_C + c) + FILTER_SEED) % 3u) -
                             1);
}
