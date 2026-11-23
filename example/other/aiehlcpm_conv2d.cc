/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "aiehlcpm_conv2d.h"
#include <ctime>
__global__ void stem_conv2d(aie::port<input_window_int8 *, aie::IFM, aie::shape<230, 230, 4>> a,
                            aie::port<input_window_int8 *, aie::IFM, aie::shape<64, 7, 7, 4>> b,
                            aie::port<output_window_int8 *, aie::OFM, aie::shape<112, 112, 64>> c) {
    const int KH = 7, KW = 7, S = 2, C = 4;   // 7x7 stride-2 conv, 4 (aligned) channels
    const int oh_t = 7, ow_t = 28, oc_t = 16; // output block computed per round
    const int ih_t = (oh_t - 1) * S + KH;
    const int iw_t = (ow_t - 1) * S + KW;
    const int oh_core = 112 / aie::num_programs(0); // 28 output rows per mesh row
    const int pid_h = aie::program_id(0);           // mesh row: which OH band
    const int pid_c = aie::program_id(1);           // mesh col: which OC group
    const int8_t *w = aie::load_buffer(b, {pid_c * oc_t, 0, 0, 0} /*OFFSET*/, {oc_t, KH, KW, C} /*SIZE*/);

    for (int i = 0; i < oh_core / oh_t; i++) {
        for (int j = 0; j < 112 / ow_t; j++) {
            const int oh0 = pid_h * oh_core + i * oh_t;
            const int ow0 = j * ow_t;
            const int8_t *x = aie::load_buffer(a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C});
            int8_t *y = aie::store_buffer(c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t});
            for (int oh = 0; oh < oh_t; oh++)
                for (int ow = 0; ow < ow_t; ow++)
                    for (int oc = 0; oc < oc_t; oc++) {
                        int acc = 0;
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                                for (int cc = 0; cc < C; cc++) {
                                    acc += x[((oh * S + kh) * iw_t + ow * S + kw) * C + cc] *
                                           w[((oc * KH + kh) * KW + kw) * C + cc];
                                }
                        y[(oh * ow_t + ow) * oc_t + oc] = (int8_t)(acc > 127 ? 127 : acc < -128 ? -128 : acc);
                    }
            aie::release(a);
            aie::commit(c);
        }
    }
    aie::release(b);
    return;
}

int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    const int in_n = INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN;
    const int w_n = K * NUM_FILTERS, out_n = M * N;
    int8_t *input = (int8_t *)__Runtime_Alloc(in_n);
    int8_t *filter = (int8_t *)__Runtime_Alloc(w_n);
    int8_t *output = (int8_t *)__Runtime_Alloc(out_n);
    int8_t *ref = (int8_t *)__Runtime_Alloc(out_n);
    memset(output, 0, out_n);
    init_conv2d_test_data(input, filter);
    XTime t0, t1;
    XTime_GetTime(&t0);
    stem_conv2d<<<mesh>>>(input, filter, output, M, N, K);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND);
    scalar_conv2d(input, filter, ref);
    int bad = 0;
    for (int i = 0; i < out_n; i++)
        bad += output[i] != ref[i];

    printf("%s: %d mismatches out of %d\n", bad ? "FAIL" : "PASS", bad, out_n);
    return 0;
}