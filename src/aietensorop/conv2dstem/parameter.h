/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
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