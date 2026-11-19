/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the quant template's operand layout and its workstation reference. f32 -> 8-bit, the
 * quantize reorder's arithmetic (aiednn_rt_internal.h rt_quant8):
 *   q = clamp(round_half_even((double) f / scale) + zp, lo, lo + 255), NaN -> 0
 *   A [M][4 N]: row m's N floats (little-endian bytes)
 *   W [N][4 N]: the first 8 bytes of every N-byte chunk hold {scale (f32 bits), zp (int32)}
 *   Y [M][N]:   the codes
 * eager_quant_code is the arithmetic the core runs: integers only, exact (see there).
 ******************************************************************************/
#ifndef QUANT_EAGER_REF_H
#define QUANT_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

/* the code of f: exact rational round-half-even of f / scale (the two are floats: q = mf / ms * 2^D), which equals
 * ref:'s nearbyint of the double quotient -- a double quotient of two 24-bit mantissas can not land on a half-integer
 * the exact one is not. Integers only: AIE-ML's core has no double. */
static inline int32_t eager_quant_code(uint32_t fu, uint32_t su, int32_t zp, int32_t lo) {
    const int32_t hi = lo + 255;
    const int32_t ef = (int32_t)((fu >> 23) & 255u), es = (int32_t)((su >> 23) & 255u);
    const int neg = (fu >> 31) != 0;
    if (ef == 255)
        return (fu & 0x7fffffu) ? 0 : neg ? lo : hi; /* NaN -> 0, inf saturates */
    const int64_t mf = (int64_t)(fu & 0x7fffffu) | (ef ? (int64_t)1 << 23 : 0);
    const int64_t ms = (int64_t)(su & 0x7fffffu) | (es ? (int64_t)1 << 23 : 0);
    const int32_t D = (ef ? ef : 1) - (es ? es : 1);
    int64_t k = 0;
    if (mf != 0 && D >= -38) {
        if (D >= 12) /* |q| >= 2048: saturates */
            return neg ? lo : hi;
        int64_t num = D > 0 ? mf << D : mf;
        const int64_t den = D < 0 ? ms << -D : ms;
        for (int b = 12; b >= 0; b--)
            if ((num >> b) >= den) {
                num -= den << b;
                k |= (int64_t)1 << b;
            }
        if (2 * num > den || (2 * num == den && (k & 1)))
            k += 1;
    }
    const int64_t q = (neg ? -k : k) + zp;
    return (int32_t)(q < lo ? lo : q > hi ? hi : q);
}

static inline int eager_quant_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int u8) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const uint8_t *A = AIEK_PTR(&args[0], const uint8_t), *W = AIEK_PTR(&args[1], const uint8_t);
    uint8_t *Y = AIEK_PTR(&args[2], uint8_t);
    uint32_t sz[2];
    memcpy(sz, W, sizeof sz);
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            uint32_t fu;
            memcpy(&fu, A + m * 4 * N + 4 * n, 4);
            Y[m * N + n] = (uint8_t)eager_quant_code(fu, sz[0], (int32_t)sz[1], u8 ? 0 : -128);
        }
    return AIEK_OK;
}

#endif /* QUANT_EAGER_REF_H */
