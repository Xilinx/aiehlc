/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the pool template's operand layout and its workstation reference. 8-bit max / average
 * pooling over packed windows:
 *   X [MW][KK]: row w is window w's taps (KK - 4 bytes; taps outside the input hold the type's minimum for max,
 *               0 for average), then one int32: the divisor (average: the window's tap count, or k*k when the
 *               padding is included; max: unused)
 *   Z [16][4 KK]: the WEIGHTS port's stand-in (the template lowers on the GEMM skeleton), never read
 *   Y [MW][4]:  byte 0 of row w is window w's code (rows are whole words for the DMA)
 * max = the largest tap; average = sum / divisor rounded half to even (ref:'s nearbyint of the exact quotient),
 * clamped to the type -- exact against ref:pooling.
 ******************************************************************************/
#ifndef POOL_EAGER_REF_H
#define POOL_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

/* sum / div rounded half to even (div > 0); 0 for div 0 (ref: with every tap in the padding) */
static inline int32_t eager_div_rne(int32_t sum, int32_t div) {
    if (div <= 0)
        return 0;
    int32_t q = sum / div, r = sum % div; /* truncated: make it floor */
    if (r < 0)
        q -= 1, r += div;
    return 2 * r > div || (2 * r == div && (q & 1)) ? q + 1 : q;
}

static inline int32_t eager_pool_row(const int8_t *row, int64_t kk, int avg, int u8) {
    const int64_t taps = kk - 4;
    int32_t div;
    memcpy(&div, row + taps, 4);
    int32_t v = avg ? 0 : (u8 ? 0 : -128);
    for (int64_t t = 0; t < taps; t++) {
        const int32_t x = u8 ? (int32_t)(uint8_t)row[t] : (int32_t)row[t];
        v = avg ? v + x : (x > v ? x : v);
    }
    if (avg)
        v = eager_div_rne(v, div);
    const int32_t lo = u8 ? 0 : -128, hi = u8 ? 255 : 127;
    return v < lo ? lo : v > hi ? hi : v;
}

static inline int eager_pool_ref(const aiek_tensor *args, int num_args, int64_t MW, int64_t KK, int avg, int u8) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int8_t *X = AIEK_PTR(&args[0], const int8_t);
    uint8_t *Y = AIEK_PTR(&args[2], uint8_t);
    for (int64_t w = 0; w < MW; w++) {
        Y[w * 4] = (uint8_t)eager_pool_row(X + w * KK, KK, avg, u8);
        Y[w * 4 + 1] = Y[w * 4 + 2] = Y[w * 4 + 3] = 0;
    }
    return AIEK_OK;
}

#endif /* POOL_EAGER_REF_H */
