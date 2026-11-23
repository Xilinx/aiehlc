/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "simpleconv2d_new_load.h"
// simpleconv2dkernel_port.cc (aie::mmul<8,8,8> compute) + the per-port schedule
// options of simpleconv2dkernel_port_opts.cc + a golden-checking host.
// Walkthrough: tutorial/aieprogram/06_gemm_optimization/4_conv_vector_scheduling.md
//
// output_tile_major: each core's 16 output blocks land back to back in DDR
// ([mesh row][mesh col][round][7][28][16]) through ONE shim BD per core. The
// dense HWC layout needs the host to re-program the shim S2MM BDs between
// height slabs, and this kernel is fast enough to send the next block while
// they are rewritten: whole blocks then land at another round's address.
#pragma output_tile_major
__global__ void conv2d_spatial(aie::port<input_window_int8 *, aie::IFM, aie::shape<230, 230, 4>> win_a,
                               aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 7, 7, 4>> win_b,
                               aie::port<output_window_int8 *, aie::OFM, aie::shape<112, 112, 64>> win_c) {
    const int KH = 7, KW = 7, S = 2, C = 4;         // 7x7 stride-2 conv, 4 (aligned) channels
    const int oh_t = 7, ow_t = 28, oc_t = 16;       // output block computed per round
    const int ih_t = (oh_t - 1) * S + KH;           // 19: input footprint of one output block
    const int iw_t = (ow_t - 1) * S + KW;           // 61
    const int oh_core = 112 / aie::num_programs(0); // 28 output rows per mesh row
    const int TAPS = KH * KW * C;                   // 196 bytes per filter
    const int ROW = iw_t * C;                       // 244 bytes per slab row
    const int pid_h = aie::program_id(0);           // mesh row: which OH band
    const int pid_c = aie::program_id(1);           // mesh col: which OC group
    using MM = aie::mmul<8, 8, 8, int8_t, int8_t>;
    aie::set_saturation(aie::saturation_mode::saturate); // int8 results clamp like the scalar reference

    // Weights: one block per core, sent once, ONE buffer -- packed into mmul B order right away.
    const int8_t *w = aie::load_buffer(win_b, {pid_c * oc_t, 0, 0, 0}, {oc_t, KH, KW, C}, {.resident = true});
    alignas(64) int8_t Bp[7 * 4 * 2 * 64]; // [kh][kb][nb][k 8][n 8]
    for (int kh = 0; kh < KH; kh++)
        for (int kb = 0; kb < 4; kb++)
            for (int nb = 0; nb < 2; nb++)
                for (int k = 0; k < 8; k++)
                    for (int n = 0; n < 8; n++) {
                        const int byte = kb < 3 ? kb * 8 + k : 20 + k; // tap byte within kernel row kh
                        const bool live = kb < 3 || k >= 4;            // bytes 20..23 already in kb 2
                        Bp[(((kh * 4 + kb) * 2 + nb) * 8 + k) * 8 + n] =
                            live ? w[(nb * 8 + n) * TAPS + kh * KW * C + byte] : (int8_t)0;
                    }
    aie::release(win_b);
    const aie::accum<acc32, 64> zero = aie::zeros<acc32, 64>();

    for (int i = 0; i < oh_core / oh_t; i++) {          // 4 row blocks
        for (int j = 0; j < 112 / ow_t; j++) {          // 4 column blocks
            const int oh0 = pid_h * oh_core + i * oh_t; // output block origin
            const int ow0 = j * ow_t;
            // Ping-pong: the DMA fills the next slab / drains the last block while the core computes.
            const int8_t *x = aie::load_buffer(win_a, {oh0 * S, ow0 * S, 0}, {ih_t, iw_t, C}, {.depth = 2});
            int8_t *y = aie::store_buffer(win_c, {oh0, ow0, pid_c * oc_t}, {oh_t, ow_t, oc_t}, {.depth = 2});
            for (int oh = 0; oh < oh_t; oh++) {
                for (int pass = 0; pass < 2; pass++) {
                    const int owA = pass * 16, owB = pass * 16 + (pass ? 4 : 8); // 0,8 then 16,20
                    MM cA0(zero), cA1(zero), cB0(zero), cB1(zero);
                    for (int kh = 0; kh < KH; kh++)
                        chess_prepare_for_pipelining chess_loop_range(7, 7) {
                            const int8_t *r = x + (oh * S + kh) * ROW;
                            const int8_t *bk = Bp + kh * 4 * 2 * 64;
                            for (int kb = 0; kb < 4; kb++)
                                chess_unroll_loop() {
                                    const int off = kb < 3 ? kb * 8 : 20;
                                    const aie::vector<int8_t, 64> aA = aie::load_unaligned_v<64>(r + owA * 8 + off, 4);
                                    const aie::vector<int8_t, 64> aB = aie::load_unaligned_v<64>(r + owB * 8 + off, 4);
                                    const aie::vector<int8_t, 64> b0 = aie::load_v<64>(bk + (kb * 2 + 0) * 64);
                                    const aie::vector<int8_t, 64> b1 = aie::load_v<64>(bk + (kb * 2 + 1) * 64);
                                    cA0.mac(aA, b0);
                                    cA1.mac(aA, b1);
                                    cB0.mac(aB, b0);
                                    cB1.mac(aB, b1);
                                }
                        }
                    // 8 columns x 16 channels = 128 contiguous output bytes:
                    // interleave the two N-blocks 8 channels at a time.
                    const int owc[2] = {owA, owB};
                    const MM *acc[2][2] = {{&cA0, &cA1}, {&cB0, &cB1}};
                    for (int m = 0; m < 2; m++) {
                        const auto z =
                            aie::interleave_zip(acc[m][0]->to_vector<int8_t>(0), acc[m][1]->to_vector<int8_t>(0), 8);
                        int8_t *o = y + (oh * ow_t + owc[m]) * oc_t; // 32-byte aligned (window base is)
                        aie::store_v(o, z.first.extract<32>(0));
                        aie::store_v(o + 32, z.first.extract<32>(1));
                        aie::store_v(o + 64, z.second.extract<32>(0));
                        aie::store_v(o + 96, z.second.extract<32>(1));
                    }
                }
            }
            aie::release(win_a);
            aie::commit(win_c);
        }
    }
}
// HOST: same data as simpleconv2dkernel_port_less.cc, then a golden check vs scalar_conv2d().
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    const int in_n = INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN, w_n = K * NUM_FILTERS, out_n = M * N;
    int8_t *input = (int8_t *)__Runtime_Alloc(in_n), *filter = (int8_t *)__Runtime_Alloc(w_n);
    int8_t *tiles = (int8_t *)__Runtime_Alloc(out_n); // what the AIE writes (tile-major)
    int8_t *output = (int8_t *)__Runtime_Alloc(out_n), *ref = (int8_t *)__Runtime_Alloc(out_n);
    memset(tiles, 0, out_n);
    init_conv2d_test_data(input, filter); // simpleconv2d_new_load.h
    XTime t0, t1;
    XTime_GetTime(&t0);
    conv2d_spatial<<<mesh>>>(input, filter, tiles, M, N, K);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0);
    // De-tile: core (r, c), round (i, j) holds output rows r*28 + i*7.., columns j*28.., channels c*16..
    const int8_t *src = tiles;
    for (int r = 0; r < HW_ROWS; r++)
        for (int c = 0; c < HW_COLS; c++)
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    for (int y = 0; y < 7; y++)
                        for (int x = 0; x < 28; x++, src += 16)
                            memcpy(&output[((r * 28 + i * 7 + y) * OUTPUT_W + j * 28 + x) * NUM_FILTERS + c * 16], src, 16);
    scalar_conv2d(input, filter, ref);
    int bad = 0;
    for (int i = 0; i < out_n; i++)
        bad += output[i] != ref[i];
    printf("%s: %d mismatches out of %d\n", bad ? "FAIL" : "PASS", bad, out_n);
    return 0;
}
