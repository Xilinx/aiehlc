/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the steps template's operand layout and its workstation reference -- a 32-bit value (f32 or
 * s32) -> 8-bit code through a step table, out = P[#{k : T[k] <= key}] (the key: the s32 itself, or the float's
 * bits made order-preserving, negatives xor 0x7fffffff; a NaN takes the table's NaN code). Every row carries its
 * own table, then its values:
 *   X [R][S + 4 V]: bytes 0 .. S - 1 the row's table
 *                   int32 ns | int16 nan code | int16 kind (0 f32, 1 s32) | int32 T[ns] ascending | int16 P[ns + 1]
 *                   then V 4-byte values
 *   Y [R][V]:       the codes
 ******************************************************************************/
#ifndef STEPS_EAGER_REF_H
#define STEPS_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

#define EG_STEPS_HDR 8

static inline int eager_steps_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t V, int64_t S) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    const uint8_t *X = AIEK_PTR(&args[0], const uint8_t);
    uint8_t *Y = AIEK_PTR(&args[1], uint8_t);
    const int64_t ld = S + 4 * V;
    for (int64_t r = 0; r < R; r++) {
        const uint8_t *tb = X + r * ld;
        int32_t ns;
        int16_t nanc, kind;
        memcpy(&ns, tb, 4), memcpy(&nanc, tb + 4, 2), memcpy(&kind, tb + 6, 2);
        for (int64_t j = 0; j < V; j++) {
            int32_t v;
            memcpy(&v, tb + S + 4 * j, 4);
            if (kind == 0 && (v & 0x7fffffff) > 0x7f800000) {
                Y[r * V + j] = (uint8_t)nanc;
                continue;
            }
            const int32_t key = kind == 0 && v < 0 ? v ^ 0x7fffffff : v;
            int lo = 0, hi = ns;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                int32_t t;
                memcpy(&t, tb + EG_STEPS_HDR + 4 * mid, 4);
                if (t <= key)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            int16_t code;
            memcpy(&code, tb + EG_STEPS_HDR + 4 * ns + 2 * lo, 2);
            Y[r * V + j] = (uint8_t)code;
        }
    }
    return AIEK_OK;
}

#endif /* STEPS_EAGER_REF_H */
