/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the frow template's operations, operand layout and workstation reference.
 *   X [R][4 L]: f32 rows; the first Lt of each row are real (the rest padding, never read except SUMDIV's divisor)
 *   P [4][16]:  every row's first int32 = Lt (one row per mesh column: a resident input takes whole rows)
 *   Y:          softmax / log softmax [R][4 L] f32 rows (0 past Lt); reductions [R][16]: the row's value in the
 *               first 4 bytes (each mesh column writes its own 4)
 * Operations (single precision, Kahan sums; within the FVEC contract of ref:'s double arithmetic):
 *   SOFTMAX e^(x - max) / sum, LOGSOFTMAX x - max - log(sum), SUM, MEAN sum / Lt, MAX, MIN,
 *   SUMDIV sum / x[L - 1] (pooling with a per-window divisor in the row's last float)
 ******************************************************************************/
#ifndef FROW_EAGER_REF_H
#define FROW_EAGER_REF_H

#include "aie_user_kernel.h"
#ifndef EAGER_FMATH_OPS /* an aiehlc build has the spliced text already; a workstation build includes it */
#include "eager_fmath_ops.cc"
#endif

/* the operation codes EG_ROW_* are in eager_fmath_ops.cc (spliced text: the kernel trace sees them) */

static inline int eager_frow_ref(const aiek_tensor *args, int num_args, int64_t R, int64_t L, int OP) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const float *X = AIEK_PTR(&args[0], const float);
    const int32_t lt = AIEK_PTR(&args[1], const int32_t)[0];
    float *Y = AIEK_PTR(&args[2], float);
    const int64_t Lt = lt < 1 ? 1 : lt > L ? L : lt;
    EAGER_FMATH_OPS;
    for (int64_t r = 0; r < R; r++) {
        const float *xr = X + r * L;
        if (OP == EG_ROW_SOFTMAX || OP == EG_ROW_LOGSOFTMAX) {
            float mx = xr[0];
            for (int64_t k = 1; k < Lt; k++)
                mx = eg_fmax(mx, xr[k]);
            float s = 0.0f, cs = 0.0f;
            for (int64_t k = 0; k < Lt; k++) {
                const float yk = eg_exp(xr[k] - mx) - cs, tk = s + yk;
                cs = (tk - s) - yk;
                s = tk;
            }
            const float is = eg_inv(s), ls = mx + eg_log(s);
            for (int64_t k = 0; k < L; k++)
                Y[r * L + k] = k >= Lt ? 0.0f : OP == EG_ROW_SOFTMAX ? eg_exp(xr[k] - mx) * is : xr[k] - ls;
        } else if (OP == EG_ROW_MAX || OP == EG_ROW_MIN) {
            float v = xr[0];
            for (int64_t k = 1; k < Lt; k++)
                v = OP == EG_ROW_MAX ? eg_fmax(v, xr[k]) : eg_fmin(v, xr[k]);
            Y[r * 4] = v;
        } else {
            float s = 0.0f, cs = 0.0f;
            for (int64_t k = 0; k < Lt; k++) {
                const float yk = xr[k] - cs, tk = s + yk;
                cs = (tk - s) - yk;
                s = tk;
            }
            Y[r * 4] = OP == EG_ROW_MEAN ? s / (float)Lt : OP == EG_ROW_SUMDIV ? s / xr[L - 1] : s;
        }
    }
    return AIEK_OK;
}

#endif /* FROW_EAGER_REF_H */
