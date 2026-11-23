/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the lut template's operand layout and its workstation reference. Any function of an 8-bit
 * value, one 256-entry table per output column (so per channel, when the host gives each column one channel):
 *   A [M][256 + N]: bytes 256 .. 256 + N - 1 of row m are the column's input codes (the first 256 are not read)
 *   W [N][256 + N]: row n's first 256 bytes are column n's table, indexed by the code as a byte (0 .. 255)
 *   Y [M][N]:       Y[m][n] = W[n][(uint8) A[m][256 + n]]
 ******************************************************************************/
#ifndef LUT_EAGER_REF_H
#define LUT_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>

#define EG_LUT_TABLE 256

static inline int eager_lut_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t N) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const uint8_t *A = AIEK_PTR(&args[0], const uint8_t), *W = AIEK_PTR(&args[1], const uint8_t);
    uint8_t *Y = AIEK_PTR(&args[2], uint8_t);
    const int64_t ld = EG_LUT_TABLE + N;
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++)
            Y[m * N + n] = W[n * ld + A[m * ld + EG_LUT_TABLE + n]];
    return AIEK_OK;
}

#endif /* LUT_EAGER_REF_H */
