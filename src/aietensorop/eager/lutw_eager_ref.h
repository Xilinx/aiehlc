/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the lutw template's operand layout and its workstation reference -- table lookup with ES-byte
 * results (s32 / f32: ES 4, f16 / bf16: ES 2). Every row carries its own 256-entry table, then its codes:
 *   X [R][256 ES + V]: bytes 0 .. 256 ES - 1 the row's table (entry v at v ES .. v ES + ES - 1), then V codes
 *   Y [R][V ES]:       Y[r][j] = entry (uint8) X[r][256 ES + j] of the row's table
 ******************************************************************************/
#ifndef LUTW_EAGER_REF_H
#define LUTW_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>

static inline int eager_lutw_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t V, int64_t ES) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    const uint8_t *X = AIEK_PTR(&args[0], const uint8_t);
    uint8_t *Y = AIEK_PTR(&args[1], uint8_t);
    const int64_t tb = 256 * ES, ld = tb + V;
    for (int64_t r = 0; r < R; r++)
        for (int64_t j = 0; j < V; j++)
            for (int64_t b = 0; b < ES; b++)
                Y[(r * V + j) * ES + b] = X[r * ld + X[r * ld + tb + j] * ES + b];
    return AIEK_OK;
}

#endif /* LUTW_EAGER_REF_H */
