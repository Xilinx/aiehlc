/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the bin8 template's operand layout and workstation reference.
 *   XY [R][2 L]: 8-bit codes, a in bytes [0, L) of a row, b in [L, 2 L) (the host packs the two sources side by side)
 *   P  [4][32]:  every row (one per mesh column: a resident input takes whole rows) the words
 *                {sa, -za, sb, -zb, lo, hi} (f32) and word 6 the signedness (int32: bit 0 a signed, bit 1 b signed)
 *   Z  [R][4 L]: Q 0: the exact int32 a OP b; Q 1: f32 clamp(((a - za) sa) OP ((b - zb) sb), lo, hi)
 * OP: eager_fmath_ops.cc's EG_F_ADD / SUB / MUL / MAX / MIN. Q 0 is exact (ref:'s small integers); Q 1 is FVEC's
 * quantized binary program in the same float order (the host quantizes the f32 result as ref: does).
 ******************************************************************************/
#ifndef BIN8_EAGER_REF_H
#define BIN8_EAGER_REF_H

#include "aie_user_kernel.h"
#ifndef EAGER_FMATH_OPS /* an aiehlc build has the spliced text already; a workstation build includes it */
#include "eager_fmath_ops.cc"
#endif
#include <string.h>

static inline int eager_bin8_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t L, int OP, int Q) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int8_t *XY = AIEK_PTR(&args[0], const int8_t);
    const float *P = AIEK_PTR(&args[1], const float);
    int32_t sg;
    memcpy(&sg, P + 6, 4);
    int8_t *Z = AIEK_PTR(&args[2], int8_t);
    EAGER_FMATH_OPS;
    for (int64_t r = 0; r < R; r++)
        for (int64_t c = 0; c < L; c++) {
            const int8_t ac = XY[r * 2 * L + c], bc = XY[r * 2 * L + L + c];
            const int32_t a = (sg & 1) ? (int32_t)ac : (int32_t)(uint8_t)ac,
                          b = (sg & 2) ? (int32_t)bc : (int32_t)(uint8_t)bc;
            if (Q) {
                const float v = eager_fop(OP, ((float)a + P[1]) * P[0], ((float)b + P[3]) * P[2], 0.0f, 0.0f);
                const float w = eg_fmin(eg_fmax(v, P[4]), P[5]);
                memcpy(Z + (r * L + c) * 4, &w, 4);
            } else {
                const int32_t w = OP == EG_F_ADD   ? a + b
                                  : OP == EG_F_SUB ? a - b
                                  : OP == EG_F_MUL ? a * b
                                  : OP == EG_F_MAX ? (a > b ? a : b)
                                                   : (a < b ? a : b);
                memcpy(Z + (r * L + c) * 4, &w, 4);
            }
        }
    return AIEK_OK;
}

#endif /* BIN8_EAGER_REF_H */
