/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the dwconv_q template's operand layout and its workstation reference. Depthwise int8
 * convolution with the fixed-point requant on the core (gemm_q's words, eager_q_code), 8-bit codes out:
 *   A [M][K], K = C/4 chunks of TK = 4 TP: pixel m's chunk g holds the TP taps of channels 4g .. 4g + 3
 *     (channel 4g + i at i TP; the taps past KH x KW and taps in the padding are 0)
 *   W [C][K]: row c's chunk c/4 holds its TP taps, then its EG_Q_WORDS requant words (TP >= 12, so they fit);
 *     the row's other chunks are never read
 *   Y [M][C]: the codes
 ******************************************************************************/
#ifndef DWCONV_Q_EAGER_REF_H
#define DWCONV_Q_EAGER_REF_H

#include "gemm_q_eager_ref.h" /* EG_Q_WORDS, eager_q_code */

static inline int eager_dwconv_q_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t C, int64_t TP) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int8_t *A = AIEK_PTR(&args[0], const int8_t), *W = AIEK_PTR(&args[1], const int8_t);
    uint8_t *Y = AIEK_PTR(&args[2], uint8_t);
    const int64_t TK = 4 * TP, K = C / 4 * TK;
    for (int64_t c = 0; c < C; c++) {
        const int8_t *w = W + c * K + c / 4 * TK;
        int32_t q[EG_Q_WORDS];
        memcpy(q, w + TP, sizeof q);
        for (int64_t m = 0; m < M; m++) {
            const int8_t *a = A + m * K + c / 4 * TK + c % 4 * TP;
            int32_t s = 0;
            for (int64_t t = 0; t < TP; t++)
                s += (int32_t)a[t] * (int32_t)w[t];
            Y[m * C + c] = (uint8_t)eager_q_code(s, q);
        }
    }
    return AIEK_OK;
}

#endif /* DWCONV_Q_EAGER_REF_H */
