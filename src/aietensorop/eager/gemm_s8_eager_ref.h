/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the workstation reference of the gemm_s8 template (gemm_s8_eager_kernel.cc), shared by
 * every generated instance's AIEK_REFERENCE. C[M][N] (int32) = A[M][K] . W[N][K]^T (int8), exact.
 ******************************************************************************/
#ifndef GEMM_S8_EAGER_REF_H
#define GEMM_S8_EAGER_REF_H

#include "aie_user_kernel.h"
#include <stdint.h>

static inline int eager_gemm_s8_ref(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int8_t *A = AIEK_PTR(&args[0], const int8_t), *W = AIEK_PTR(&args[1], const int8_t);
    int32_t *C = AIEK_PTR(&args[2], int32_t);
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            int32_t s = 0;
            for (int64_t k = 0; k < K; k++)
                s += (int32_t)A[m * K + k] * (int32_t)W[n * K + k];
            C[m * N + n] = s;
        }
    return AIEK_OK;
}

#endif /* GEMM_S8_EAGER_REF_H */
