/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
for (int i = 2; i < nargs - 1; i++) { // epilogue: bias / temb (per channel) or residual (same shape)
    const uc_dlt *e = (const uc_dlt *)argv[i];
    const float *ev = (const float *)e->data;
    const long ne = uc_numel(e);
    if (ne == N && e->ndim == 5) {
        for (int n = 0; n < N; n++) { // [1,N/eb,1,1,eb] -> flat n
            float *o = yout + (size_t)(n / yb) * M * yb + n % yb;
            for (int m = 0; m < M; m++)
                o[(size_t)m * yb] += ev[n];
        }
    } else if (ne == (long)M * N && e->ndim == 5 && (int)e->shape[4] == yb) {
        for (long q = 0; q < (long)M * N; q++)
            yout[q] += ev[q];
    } else {
        return -5;
    }
    }
