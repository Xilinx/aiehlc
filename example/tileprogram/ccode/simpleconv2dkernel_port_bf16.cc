/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "simpleconv2d_bf16.h"
// bfloat16 twin of simpleconv2dkernel_port_less.cc: bf16 in / out, fp32 accumulation, scalar MACs.
// bf16 blocks are twice the bytes, so the output tile is 7x14 (not 7x28) to fit the core buffers.
__global__ void conv2d_spatial(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<230, 230, 4>> win_a,
                               aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<64, 7, 7, 4>> win_b,
                               aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<112, 112, 64>> win_c) {
    const int KH = 7, KW = 7, S = 2, C = 4;         // 7x7 stride-2 conv, 4 (aligned) channels
    const int oh_t = 7, ow_t = 14, oc_t = 16;       // output block computed per round
    const int ih_t = (oh_t - 1) * S + KH;           // 19: input footprint of one output block
    const int iw_t = (ow_t - 1) * S + KW;           // 33
    const int oh_core = 112 / aie::num_programs(0); // 28 output rows per mesh row
    const int pid_h = aie::program_id(0);           // mesh row: which OH band
    const int pid_c = aie::program_id(1);           // mesh col: which OC group
    const bfloat16 *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C}, {.resident = true});
    for (int i = 0; i < oh_core / oh_t; i++) {          // 4 row blocks
        for (int j = 0; j < 112 / ow_t; j++) {          // 8 column blocks
            const int oh0 = pid_h * oh_core + i * oh_t; // output block origin
            const int ow0 = j * ow_t;
            const bfloat16 *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C});
            bfloat16 *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t});
            for (int oh = 0; oh < oh_t; oh++)
                for (int ow = 0; ow < ow_t; ow++)
                    for (int oc = 0; oc < oc_t; oc++) {
                        float acc = 0.0f;
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                                for (int c = 0; c < C; c++)
                                    acc += (float)x[((oh * S + kh) * iw_t + ow * S + kw) * C + c] *
                                           (float)w[((oc * KH + kh) * KW + kw) * C + c];
                        y[(oh * ow_t + ow) * oc_t + oc] = (bfloat16)acc;
                    }
            aie::release(win_a);
            aie::commit(win_c);
        }
    }
    aie::release(win_b);
}
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    const int in_n = INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN, w_n = K * NUM_FILTERS, out_n = M * N;
    uint16_t *input = (uint16_t *)__Runtime_Alloc(in_n * 2), *filter = (uint16_t *)__Runtime_Alloc(w_n * 2);
    uint16_t *output = (uint16_t *)__Runtime_Alloc(out_n * 2), *ref = (uint16_t *)__Runtime_Alloc(out_n * 2);
    memset(output, 0, out_n * 2);
    init_conv2d_bf16(input, filter); // simpleconv2d_bf16.h
    XTime t0, t1;
    XTime_GetTime(&t0);
    conv2d_spatial<<<mesh>>>(input, filter, output, M, N, K);
    XTime_GetTime(&t1);
    // The launch invalidates `output` only BEFORE the DMA; a line the A78 refetched while the
    // core wrote it stays stale (~1 run in 6: a few hundred outputs still read 0). Drop it.
    Xil_DCacheInvalidateRange((UINTPTR)output, out_n * 2);
    printf("[BENCH] aie_wall_ms=%.3f\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0);
    ref_conv2d_bf16(input, filter, ref);
    check_conv2d_bf16(output, ref, out_n);
    return 0;
}
