/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the gemm_q template's requant words and its workstation reference.
 * C[M][N] (8-bit codes) = requant(A[M][K] . W[N][K]^T), the fixed-point requant aiednn defines (rt_fx_q):
 *   q = clamp(((acc * M[n] + B[n]) >> sh[n]) + zp[n], lo[n], hi[n])
 * A' [M][K + TK] ends in a zero chunk; W' [N][K + TK]: row n is channel n's K weights, then a chunk whose first
 *EG_Q_WORDS int32 are {M, B low word, B high word, sh, zp, lo, hi, 0}.
 ******************************************************************************/
#ifndef GEMM_Q_EAGER_REF_H
#define GEMM_Q_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

#define EG_Q_WORDS 8

static inline int32_t eager_q_code(int32_t acc, const int32_t *p) {
    const int64_t B = (int64_t)(((uint64_t)(uint32_t)p[2] << 32) | (uint32_t)p[1]);
    const int64_t r = (((int64_t)acc * (int64_t)p[0] + B) >> p[3]) + p[4];
    return (int32_t)(r < p[5] ? p[5] : r > p[6] ? p[6] : r);
}

static inline int eager_gemm_q_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K, int64_t TK) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int8_t *A = AIEK_PTR(&args[0], const int8_t), *W = AIEK_PTR(&args[1], const int8_t);
    uint8_t *C = AIEK_PTR(&args[2], uint8_t); /* s8 / u8 codes: the same byte */
    const int64_t ld = K + TK;                /* A' ends in a zero chunk, W' in the Q chunk */
    for (int64_t n = 0; n < N; n++) {
        int32_t q[EG_Q_WORDS];
        memcpy(q, W + n * ld + K, sizeof q);
        for (int64_t m = 0; m < M; m++) {
            int32_t s = 0;
            for (int64_t k = 0; k < K; k++)
                s += (int32_t)A[m * ld + k] * (int32_t)W[n * ld + k];
            C[m * N + n] = (uint8_t)eager_q_code(s, q);
        }
    }
    return AIEK_OK;
}

#endif /* GEMM_Q_EAGER_REF_H */
