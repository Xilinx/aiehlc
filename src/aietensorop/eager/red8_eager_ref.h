/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the red8 template's operand layout and its workstation reference -- the exact int32 reduction
 * of each row of 8-bit values (SG 1 signed, 0 unsigned; the host pads a short row with the op's neutral value).
 *   X [R][L]:  8-bit rows
 *   Y [R][16]: the row's int32 in the first 4 bytes (each mesh column writes its own 4)
 * OP: 1 sum, 2 max, 3 min, 4 sum |x|, 5 sum x x, 6 max |x|
 ******************************************************************************/
#ifndef RED8_EAGER_REF_H
#define RED8_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

static inline int eager_red8_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t L, int OP, int SG) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    const uint8_t *X = AIEK_PTR(&args[0], const uint8_t);
    uint8_t *Y = AIEK_PTR(&args[1], uint8_t);
    for (int64_t r = 0; r < R; r++) {
        int32_t acc = 0;
        for (int64_t k = 0; k < L; k++) {
            const int32_t v = SG ? (int32_t)(int8_t)X[r * L + k] : (int32_t)X[r * L + k], a = v < 0 ? -v : v;
            if (k == 0 && (OP == 2 || OP == 3))
                acc = v;
            else
                acc = OP == 1   ? acc + v
                      : OP == 2 ? (v > acc ? v : acc)
                      : OP == 3 ? (v < acc ? v : acc)
                      : OP == 4 ? acc + a
                      : OP == 5 ? acc + v * v
                                : (a > acc ? a : acc);
        }
        memset(Y + r * 16, 0, 16);
        memcpy(Y + r * 16, &acc, 4);
    }
    return AIEK_OK;
}

#endif /* RED8_EAGER_REF_H */
