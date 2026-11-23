/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the dwconv_s8 template's operand layout and its workstation reference. Depthwise int8
 * convolution to exact int32 sums (the host applies any epilogue: scales, bias, post-ops, an s32 / f32 / 8-bit dst):
 *   A [M][K], K = C/4 chunks of TK = 4 TP: pixel m's chunk g holds the TP taps of channels 4g .. 4g + 3
 *     (channel 4g + i at i TP; the taps past KH x KW and taps in the padding are 0)
 *   W [C][K]: row c's chunk c/4 holds its TP taps; the row's other chunks are never read
 *   Y [M][C]: the int32 sums
 ******************************************************************************/
#ifndef DWCONV_S8_EAGER_REF_H
#define DWCONV_S8_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>

static inline int eager_dwconv_s8_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t C, int64_t TP) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int8_t *A = AIEK_PTR(&args[0], const int8_t), *W = AIEK_PTR(&args[1], const int8_t);
    int32_t *Y = AIEK_PTR(&args[2], int32_t);
    const int64_t TK = 4 * TP, K = C / 4 * TK;
    for (int64_t c = 0; c < C; c++) {
        const int8_t *w = W + c * K + c / 4 * TK;
        for (int64_t m = 0; m < M; m++) {
            const int8_t *a = A + m * K + c / 4 * TK + c % 4 * TP;
            int32_t s = 0;
            for (int64_t t = 0; t < TP; t++)
                s += (int32_t)a[t] * (int32_t)w[t];
            Y[m * C + c] = s;
        }
    }
    return AIEK_OK;
}

#endif /* DWCONV_S8_EAGER_REF_H */
