/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the gemm_bf16 template's operand layout and its workstation reference.
 *   C[M][N] (f32) = A[M][K] . W[N][K]^T with bf16 A, W: every product is exact in f32 (8-bit mantissas), the
 *   sum is f32 -- per TK chunk s = sum of its products in k order, then acc += s, chunk after chunk -- the same
 *   float operations, in the same order, as the core (gemm_bf16_eager_kernel.cc), so the two are bit-identical.
 *   A [M][2 K], W [N][2 K]: bf16 as little-endian bytes;  C [M][N]: f32 bits in 32-bit words.
 * An f32 x f32 matmul runs as a bf16x3 split of depth 6 K (the runtime builds it: aiednn_rt_bf16.h's scheme).
 ******************************************************************************/
#ifndef GEMM_BF16_EAGER_REF_H
#define GEMM_BF16_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

static inline float eager_bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline int eager_gemm_bf16_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K,
                                      int64_t TK) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const uint16_t *A = AIEK_PTR(&args[0], const uint16_t), *W = AIEK_PTR(&args[1], const uint16_t);
    float *C = AIEK_PTR(&args[2], float);
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            volatile float acc = 0.0f; /* volatile: no fused multiply-add, the core's order */
            for (int64_t k0 = 0; k0 < K; k0 += TK) {
                volatile float s = 0.0f;
                for (int64_t k = k0; k < k0 + TK; k++) {
                    volatile float p = eager_bf2f(A[m * K + k]) * eager_bf2f(W[n * K + k]);
                    s = s + p;
                }
                acc = acc + s;
            }
            C[m * N + n] = acc;
        }
    return AIEK_OK;
}

#endif /* GEMM_BF16_EAGER_REF_H */
