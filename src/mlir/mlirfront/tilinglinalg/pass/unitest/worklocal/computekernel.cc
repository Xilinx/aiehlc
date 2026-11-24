/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// Auto-generated compute kernel: computekernel
// 2 input(s) + 1 output(s)
void computekernel(input_window_int8 *window_in_0, input_window_int8 *window_in_1, output_window_int8 *window_out_0) {
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    /*  for (int k = 0; k < 2; k++) {
            klog("CENk", k);
            int8_t *in0 = (int8_t *)acquire_input_window(window_in_0);
            int8_t *in1 = (int8_t *)acquire_input_window(window_in_1);
            int8_t *out0 = (int8_t *)acquire_output_window(window_out_0);

            // Debug: dump in0 buffer contents
            klog("IN0", BUF_SZ_IN_0 * 4);
            for (int di = 0; di < BUF_SZ_IN_0 * 4; di++) {
                klog("IV", (int)in0[di]);
            }

            // GEMM kernel: out0[i] = in0[i] * in1[i]
            for (int i = 0; i < BUF_SZ_OUT_0; i++) {
                v4int8 data0 = *((v4int8 *)&in0[i * 4]);
                v4int8 data1 = *((v4int8 *)&in1[i * 4]);
                *((v4int8 *)&out0[i * 4]) = data0;
            }
            klog("CLOP", BUF_SZ_OUT_0);

            // Debug: dump out0 buffer contents
            klog("OUT0", BUF_SZ_OUT_0 * 4);
            for (int di = 0; di < BUF_SZ_OUT_0 * 4; di++) {
                klog("OV", (int)out0[di]);
            }

            release_input_window(window_in_0);
            release_input_window(window_in_1);
            release_output_window(window_out_0);
            klog("CEXT", 1);
        }*/
}
