/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * matmul_s8_ref -- x86 / CPU reference for matmul_uk.cc:
 * dst[m][n] = sat8(sum_k src[m][k] * weight[n][k]), int32 accumulate.
 * Shape-generic: M, K from src, N from weight.
 ******************************************************************************/
#include "aie_user_kernel.h"

AIEK_REFERENCE(matmul_s8) {
    if (num_args != 3 || args[0].ndim != 2 || args[1].ndim != 2 || args[2].ndim != 2)
        return AIEK_ERR_ARGS;
    const int64_t m = args[0].shape[0], k = args[0].shape[1], n = args[1].shape[0];
    if (args[1].shape[1] != k || args[2].shape[0] != m || args[2].shape[1] != n)
        return AIEK_ERR_ARGS;
    const int8_t *a = AIEK_PTR(&args[0], const int8_t);
    const int8_t *b = AIEK_PTR(&args[1], const int8_t);
    int8_t *c = AIEK_PTR(&args[2], int8_t);
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int64_t q = 0; q < k; q++)
                acc += (int32_t)a[i * k + q] * (int32_t)b[j * k + q];
            c[i * n + j] = (int8_t)(acc > 127 ? 127 : acc < -128 ? -128 : acc);
        }
    return AIEK_OK;
}
