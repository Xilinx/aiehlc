/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the gemm_qr template's operand layout and its workstation reference. gemm_q plus an 8-bit
 * residual added inside the fixed-point requant (aiednn's integer residual, rt_fx_q with fx_r):
 *   q = clamp(((acc * M[n] + B[n] + (r[m][n] - zr) * R[n]) >> sh[n]) + zp, lo, hi)
 * Both operands carry XC = 1 + ceil(N / TK) chunks past K (the K walks stay equal):
 *   A' [M][K + XC*TK]: chunk 0 past K is zero; chunks 1.. hold row m's residual codes r[m][0..N)
 *   W' [N][K + XC*TK]: chunk 0 past K leads with channel n's EG_QR_WORDS int32
 *                      {M, B lo, B hi, sh, zp, lo, hi, zr, R lo, R hi, r is u8, 0}; the rest is zero
 ******************************************************************************/
#ifndef GEMM_QR_EAGER_REF_H
#define GEMM_QR_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

#define EG_QR_WORDS 12

/* the chunks past K: Q, then the residual */
static inline int64_t eager_qr_xc(int64_t N, int64_t TK) { return 1 + (N + TK - 1) / TK; }

static inline int32_t eager_qr_code(int32_t acc, int32_t rcode, const int32_t *p) {
    const int64_t B = (int64_t)(((uint64_t)(uint32_t)p[2] << 32) | (uint32_t)p[1]);
    const int64_t R = (int64_t)(((uint64_t)(uint32_t)p[9] << 32) | (uint32_t)p[8]);
    const int64_t r = (p[10] ? (int64_t)(uint8_t)rcode : (int64_t)(int8_t)rcode) - p[7];
    const int64_t v = (((int64_t)acc * (int64_t)p[0] + B + r * R) >> p[3]) + p[4];
    return (int32_t)(v < p[5] ? p[5] : v > p[6] ? p[6] : v);
}

static inline int eager_gemm_qr_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K,
                                    int64_t TK) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int8_t *A = AIEK_PTR(&args[0], const int8_t), *W = AIEK_PTR(&args[1], const int8_t);
    uint8_t *C = AIEK_PTR(&args[2], uint8_t); /* s8 / u8 codes: the same byte */
    const int64_t ld = K + eager_qr_xc(N, TK) * TK;
    for (int64_t n = 0; n < N; n++) {
        int32_t q[EG_QR_WORDS];
        memcpy(q, W + n * ld + K, sizeof q);
        for (int64_t m = 0; m < M; m++) {
            int32_t s = 0;
            for (int64_t k = 0; k < K; k++)
                s += (int32_t)A[m * ld + k] * (int32_t)W[n * ld + k];
            C[m * N + n] = (uint8_t)eager_qr_code(s, A[m * ld + K + TK + n], q);
        }
    }
    return AIEK_OK;
}

#endif /* GEMM_QR_EAGER_REF_H */
