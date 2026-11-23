/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the lnorm template's operand layout and its workstation reference. Layer / RMS norm of f32
 * rows, f32 out, in single precision on the core (AIE-ML has no double): within the 1e-5 contract of the
 * fixed-megakernel fvec: rows against ref:'s double arithmetic, not bit-exact.
 *   X [R][4 L]: row r's f32 (bytes), the first Lt of them real (Lt <= L, the rest unread)
 *   P [12][L]:  pre-tiled per core (.resident takes whole rows): rows 3 q, 3 q + 1 = gamma, beta (f32) of elements
 *               q L/4 .. (q L/4 + L/4 - 1); row 3 q + 2 starts {eps (f32), 1 / Lt (f32, unused: the core divides by
 *               Lt), Lt (int32)}
 *   Y [R][4 L]: the normalized rows (f32 bytes); elements past Lt are 0
 * y = (x - mean) * rsqrt(var + eps) * gamma + beta (RMS: mean 0, no beta), two passes of compensated (Kahan) sums,
 * rsqrt by Newton's method.
 ******************************************************************************/
#ifndef LNORM_EAGER_REF_H
#define LNORM_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>
#include <string.h>

/* 1 / sqrt(v), v > 0: the bit-trick seed and four Newton steps (float, as the core) */
static inline float eager_rsqrtf(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    u = 0x5f3759dfu - (u >> 1);
    float y;
    memcpy(&y, &u, 4);
    for (int k = 0; k < 4; k++)
        y = y * (1.5f - 0.5f * v * y * y);
    return y;
}

static inline void eager_lnorm_row(const float *x, float *y, int64_t L, int64_t Lt, const float *gam, const float *bet,
                                   float eps, float rl, int rms) {
    (void)rl;
    float mean = 0.0f, var = 0.0f, cm = 0.0f, cv = 0.0f; /* Kahan sums: a row's offset would swamp a plain one */
    if (!rms) {
        for (int64_t i = 0; i < Lt; i++) {
            const float yk = x[i] - cm, tk = mean + yk;
            cm = (tk - mean) - yk;
            mean = tk;
        }
        mean /= (float)Lt; /* 1 / L is not exact in f32: divide */
    }
    for (int64_t i = 0; i < Lt; i++) {
        const float d = x[i] - mean, yk = d * d - cv, tk = var + yk;
        cv = (tk - var) - yk;
        var = tk;
    }
    const float rs = eager_rsqrtf(var / (float)Lt + eps);
    for (int64_t i = 0; i < L; i++)
        y[i] = i < Lt ? (x[i] - mean) * rs * gam[i] + (rms ? 0.0f : bet[i]) : 0.0f;
}

static inline int eager_lnorm_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t L, int rms) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const uint8_t *X = AIEK_PTR(&args[0], const uint8_t), *P = AIEK_PTR(&args[1], const uint8_t);
    uint8_t *Y = AIEK_PTR(&args[2], uint8_t);
    float eps, rl;
    int32_t Lt;
    memcpy(&eps, P + 2 * L, 4);
    memcpy(&rl, P + 2 * L + 4, 4);
    memcpy(&Lt, P + 2 * L + 8, 4);
    Lt = Lt < 0 ? 0 : Lt > L ? (int32_t)L : Lt; /* as the core */
    float gam[4096], bet[4096];                 /* gather the per-core quarters back into rows (L <= 4096) */
    if (L > 4096)
        return AIEK_ERR_ARGS;
    for (int64_t q = 0; q < 4; q++) {
        memcpy(gam + q * (L / 4), P + 3 * q * L, (size_t)L);
        memcpy(bet + q * (L / 4), P + (3 * q + 1) * L, (size_t)L);
    }
    for (int64_t r = 0; r < R; r++)
        eager_lnorm_row((const float *)(X + r * 4 * L), (float *)(Y + r * 4 * L), L, Lt, gam, bet, eps, rl, rms);
    return AIEK_OK;
}

#endif /* LNORM_EAGER_REF_H */
