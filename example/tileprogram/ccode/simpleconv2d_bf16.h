/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host side of simpleconv2dkernel_port_bf16.cc: the same 7x7 stride-2 conv as
 * simpleconv2d_new_load.h (230x230x4 pre-padded input, 64 filters, 112x112x64
 * output), in bfloat16. On the host a bfloat16 is its 16 bits (uint16_t).
 *
 * Test data are multiples of 1/4 (pixels k/4, k in [-4, 4]; weights k/4, k in
 * [-2, 2]), so every product and partial sum is exact in fp32: the fp32 sum
 * does not depend on the order the core accumulates in, and only the final
 * float -> bfloat16 rounding can differ (round-to-nearest-even here).
 ******************************************************************************/
#ifndef SIMPLECONV2D_BF16_H
#define SIMPLECONV2D_BF16_H

#include "simpleconv2d_new_load.h" // geometry (INPUT_*, KERNEL_*, NUM_FILTERS, OUTPUT_*), mix32

static inline float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Round to nearest even (NaN not produced by this test).
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Pre-padded, channel-aligned input [230][230][4] and weights [64][7][7][4];
// the border and the pad channel stay 0.
static void init_conv2d_bf16(uint16_t *input, uint16_t *filter) {
    memset(input, 0, INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN * 2);
    memset(filter, 0, K * NUM_FILTERS * 2);
    for (int h = 0; h < INPUT_H; h++)
        for (int w = 0; w < INPUT_W; w++)
            for (int c = 0; c < INPUT_C; c++) {
                const int k = (int)(mix32((uint32_t)((h * INPUT_W + w) * INPUT_C + c) + INPUT_SEED) % 9u) - 4;
                input[((h + PAD) * INPUT_W_PAD + w + PAD) * INPUT_C_ALIGN + c] = f32_to_bf16(k * 0.25f);
            }
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int t = 0; t < KERNEL_H * KERNEL_W; t++)
            for (int c = 0; c < INPUT_C; c++) {
                const int k =
                    (int)(mix32((uint32_t)((f * KERNEL_H * KERNEL_W + t) * INPUT_C + c) + FILTER_SEED) % 5u) - 2;
                filter[f * K + t * INPUT_C_ALIGN + c] = f32_to_bf16(k * 0.25f);
            }
}

// fp32 accumulate in the kernel's (kh, kw, c) order, one rounding to bfloat16.
static void ref_conv2d_bf16(const uint16_t *input, const uint16_t *filter, uint16_t *out) {
    for (int oh = 0; oh < OUTPUT_H; oh++)
        for (int ow = 0; ow < OUTPUT_W; ow++)
            for (int oc = 0; oc < NUM_FILTERS; oc++) {
                float acc = 0.0f;
                for (int kh = 0; kh < KERNEL_H; kh++)
                    for (int kw = 0; kw < KERNEL_W; kw++)
                        for (int c = 0; c < INPUT_C_ALIGN; c++)
                            acc +=
                                bf16_to_f32(
                                    input[((oh * STRIDE + kh) * INPUT_W_PAD + ow * STRIDE + kw) * INPUT_C_ALIGN + c]) *
                                bf16_to_f32(filter[oc * K + (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c]);
                out[(oh * OUTPUT_W + ow) * NUM_FILTERS + oc] = f32_to_bf16(acc);
            }
}

// Bit-exact count and the largest distance in bfloat16 ULPs (same-sign values:
// the bit patterns are ordered). PASS = every output within 1 ULP.
static int check_conv2d_bf16(const uint16_t *got, const uint16_t *want, int n) {
    int exact = 0, worst = 0, at = -1;
    for (int i = 0; i < n; i++) {
        const int d = (got[i] >> 15) == (want[i] >> 15) ? abs((int)got[i] - (int)want[i]) : 0x7fff;
        exact += d == 0;
        if (d > worst) {
            worst = d;
            at = i;
        }
    }
    printf("bf16 check: %d / %d bit-exact, max %d ULP", exact, n, worst);
    if (at >= 0)
        printf(" (at %d: got %g, want %g)", at, bf16_to_f32(got[at]), bf16_to_f32(want[at]));
    printf("\n%s\n", worst <= 1 ? "PASS" : "FAIL");
    return worst > 1;
}

#endif // SIMPLECONV2D_BF16_H
