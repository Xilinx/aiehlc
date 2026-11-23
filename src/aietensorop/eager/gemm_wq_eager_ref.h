/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the gemm_wq template's operand layout and its workstation reference -- grouped weight-only
 * quantization, an int8 src times 4-bit weights with a scale and a zero point per (group of GK along K, column):
 *   C[m][n] = sum_g s_g[n] * sum_{k in g} a[m][k] (w[k][n] - zw_g[n])     (f32, summed over the groups in order)
 *   A [M][K]: int8
 *   W [N][K]: row n's chunk g (bytes g GK .. g GK + GK - 1) holds group g: GK / 2 bytes of nibbles (k = g GK + 2 i in
 *             the low nibble of byte i, k + 1 in the high one; signed 4-bit), then the f32 scale s_g[n], then the
 *             int32 zero point zw_g[n]; the rest of the chunk is not read
 *   C [M][N]: f32 (as 4-byte words)
 * Each group's sum is an exact int32; the f32 accumulation order is the groups' (the core's and this one's agree).
 ******************************************************************************/
#ifndef GEMM_WQ_EAGER_REF_H
#define GEMM_WQ_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

static inline int eager_gemm_wq_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K,
                                    int64_t GK) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int8_t *A = AIEK_PTR(&args[0], const int8_t), *W = AIEK_PTR(&args[1], const int8_t);
    float *C = AIEK_PTR(&args[2], float);
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t g = 0; g < K / GK; g++) {
                const int8_t *ch = W + n * K + g * GK, *a = A + m * K + g * GK;
                float s;
                int32_t zw, p = 0;
                memcpy(&s, ch + GK / 2, 4), memcpy(&zw, ch + GK / 2 + 4, 4);
                for (int64_t i = 0; i < GK / 2; i++) {
                    const int32_t lo = (int32_t)(int8_t)(ch[i] << 4) >> 4, hi = (int32_t)ch[i] >> 4;
                    p += (int32_t)a[2 * i] * (lo - zw) + (int32_t)a[2 * i + 1] * (hi - zw);
                }
                acc += s * (float)p;
            }
            C[m * N + n] = acc;
        }
    return AIEK_OK;
}

#endif /* GEMM_WQ_EAGER_REF_H */
