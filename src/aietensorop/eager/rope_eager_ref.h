/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the rope template's operand layout and its workstation reference. Rotary position
 * embedding of f32 rows of D (even), f32 out, in the float operations ref:rope uses (no fused multiply-add):
 *   y[j] = x[j] * cos[j] + s * x[p(j)] * sin[j]
 *   half-split (INTER 0): p(j) = j + D/2, s = -1 for j < D/2; p(j) = j - D/2, s = +1 after
 *   interleaved (INTER 1): p(j) = j ^ 1, s = -1 for even j, +1 for odd j
 *   X [R][12 D]: row r = its x, cos, sin (f32 bytes, the host gathers the broadcast cos / sin rows)
 *   Y [R][4 D]:  the rotated row (f32 bytes)
 * Float multiplies and adds are IEEE on the core and the host alike, so the result is ref:'s bit for bit.
 ******************************************************************************/
#ifndef ROPE_EAGER_REF_H
#define ROPE_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

static inline float eager_rope_elem(const float *x, const float *c, const float *s, int64_t D, int64_t j, int inter) {
    const int64_t h = D / 2;
    const int64_t pj = inter ? (j ^ 1) : (j < h ? j + h : j - h);
    const float sign = inter ? ((j & 1) ? 1.0f : -1.0f) : (j < h ? -1.0f : 1.0f);
    volatile float a = x[j] * c[j], b = (sign * x[pj]) * s[j]; /* volatile: no fused multiply-add */
    return a + b;
}

static inline int eager_rope_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t D, int inter) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    const uint8_t *X = AIEK_PTR(&args[0], const uint8_t);
    uint8_t *Y = AIEK_PTR(&args[1], uint8_t);
    for (int64_t r = 0; r < R; r++) {
        const float *x = (const float *)(X + r * 12 * D), *c = x + D, *s = x + 2 * D;
        float *y = (float *)(Y + r * 4 * D);
        for (int64_t j = 0; j < D; j++)
            y[j] = eager_rope_elem(x, c, s, D, j, inter);
    }
    return AIEK_OK;
}

#endif /* ROPE_EAGER_REF_H */
