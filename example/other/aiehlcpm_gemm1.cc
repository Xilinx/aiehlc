/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <aiehlcpm_gemm1.h>
__global__ void gemm(aie::port<input_window_int8 *, aie::IFM, aie::shape<64, 256>> a,
                     aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 256>> b,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<64, 64>> c) {
    const int tm = 16, tn = 16,
              tk = 64; // this core get tm row of a tn row of b , the each load get 64 of 256 in another direction
    const int pm = aie::program_id(0); // row index
    const int pn = aie::program_id(1); // col index
    int acc[tm * tn];
    for (int i = 0; i < tm * tn; i++)
        acc[i] = 0;
    for (int j = 0; j < 256 / tk; j++) {
        const int8_t *x = aie::load_buffer(a, {pm * tm, j * tk}, {tm, tk});
        const int8_t *w = aie::load_buffer(b, {pn * tn, j * tk}, {tn, tk});
        for (int m = 0; m < tm; m++)
            for (int n = 0; n < tn; n++) {
                int s = 0;
                for (int k = 0; k < tk; k++)
                    s += x[m * tk + k] * w[n * tk + k];
                acc[m * tn + n] += s;
            }
        aie::release(a);
        aie::release(b);
    }
    int8_t *y = aie::store_buffer(c, {pm * tm, pn * tn}, {tm, tn});
    for (int i = 0; i < tm * tn; i++)
        y[i] = acc[i];
    aie::commit(c);
}

int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    int8_t *A = (int8_t *)__Runtime_Alloc(64 * 256);
    int8_t *B = (int8_t *)__Runtime_Alloc(64 * 256);
    int8_t *C = (int8_t *)__Runtime_Alloc(64 * 256);
    gemm_init(A, B, C);
    gemm<<<mesh>>>(A, B, C);
    const int mismatches = gemm_verify(A, B, C);
    free(A);
    free(B);
    free(C);
    return 0;
}