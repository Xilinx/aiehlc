/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the f32 pointwise templates' operand layouts and workstation references.
 *   feltw<R, L, TR, F>:   Y[r][c] = F(X[r][c]; alpha, beta) -- X, Y [R][4 L] (f32 bytes); P [4][16]: every row
 *                         {alpha, beta} (f32), one row per mesh column (a resident input takes whole rows)
 *   fbinary<R, L, TR, F>: Z[r][c] = F(X[r][c], Y[r][c]) -- XY [R][8 L] (f32 bytes: each row x then y; the host
 *                         materializes a broadcast operand), Z [R][4 L]
 *   faffine<R, L, TR, F>: Y[r][c] = F(X[r][c]; c0, c1) -- XC [R][4 L + 16]: each row its L f32 values, then
 *                         {c0, c1, -, -} (f32); Y [R][4 L]: affine (F 96), affine + ReLU (97), PReLU with slope c0 (98)
 * F: eager_fmath.h's algorithm (the fixed-megakernel FVEC mode's AIE_FVF code), compiled into the instance.
 ******************************************************************************/
#ifndef FELT_EAGER_REF_H
#define FELT_EAGER_REF_H

#include "aie_user_kernel.h"
#ifndef EAGER_FMATH_OPS /* an aiehlc build has the spliced text already; a workstation build includes it */
#include "eager_fmath_ops.cc"
#endif
#include <string.h>

static inline int eager_feltw_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t L, int F) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const float *X = AIEK_PTR(&args[0], const float), *P = AIEK_PTR(&args[1], const float);
    float *Y = AIEK_PTR(&args[2], float);
    EAGER_FMATH_OPS;
    for (int64_t i = 0; i < R * L; i++)
        Y[i] = eager_fop(F, X[i], 0.0f, P[0], P[1]);
    return AIEK_OK;
}

static inline int eager_fbinary_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t L, int F) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    const float *XY = AIEK_PTR(&args[0], const float);
    float *Z = AIEK_PTR(&args[1], float);
    EAGER_FMATH_OPS;
    for (int64_t i = 0; i < R * L; i++)
        Z[i] = eager_fop(F, XY[i / L * 2 * L + i % L], XY[i / L * 2 * L + L + i % L], 0.0f, 0.0f);
    return AIEK_OK;
}

static inline int eager_faffine_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t L, int F) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    const float *XC = AIEK_PTR(&args[0], const float);
    float *Y = AIEK_PTR(&args[1], float);
    EAGER_FMATH_OPS;
    for (int64_t i = 0; i < R * L; i++) {
        const float *row = XC + i / L * (L + 4), c0 = row[L], c1 = row[L + 1];
        Y[i] = eager_fop(F, row[i % L], c0, c0, c1);
    }
    return AIEK_OK;
}

#endif /* FELT_EAGER_REF_H */
