/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// aiednn/conv2d/dw3x3_bf16.cc -- depthwise 3x3 conv (groups = C, stride 1, pad 1; the YOLO DWConv), bf16 in / out,
// fp32 accumulation, bias + optional activation fused (ACT: 0 none, 1 ReLU, 2 SiLU).
// Per channel, 16 output pixels of a row per step: 9 aie::mac of a shifted 16-pixel input vector by the
// broadcast tap (accfloat).
// Ports: x [C][HP][WX] (padded), w = ONNX weight [C][1*3*3], b fp32 [C] (as bf16 [C][2]), y [C][H][W].
// The input is zero-padded by the host: [C][HP][WX] = [C][H + 2][1 + W + 1, right-padded to whole 16-lane row steps]
// (aiehlc's conv convention: the IFM is the DDR buffer as the kernel reads it), so every block lies inside it.
// The fp32 bias travels as bf16 pairs [C][2] (aiehlc gives all ports of a kernel one window element type).
template <int C, int H, int W, int HP, int WX, int TC, int TOH, int ACT>
__global__ void dw3x3_bf16(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<C, HP, WX>> x_port,
                           aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<C, 9>> w_port,
                           aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<C, 2>> b_port,
                           aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<C, H, W>> y_port) {
    constexpr int OH = H, OW = W;
    static_assert(C % TC == 0 && OH % TOH == 0, "block tiling");
    static_assert(HP == H + 2 && WX == (W + 15) / 16 * 16 + 2, "padded input: 1 row / col halo, whole 16-lane rows");
    constexpr int NJ = (OW + 15) / 16, IHT = TOH + 2, IWT = NJ * 16 + 2; // 16-lane row steps; padded footprint
    // Each core reads its own input block, and aiehlc routes an input that depends on one program id only
    // (broadcast along the other mesh axis): the TC-channel blocks are dealt over mesh rows, the TOH-row bands
    // walked in time -- launch on an R x 1 mesh.
    const int pr = aie::program_id(0), nr = aie::num_programs(0);
    for (int cb = pr; cb < C / TC; cb += nr)
        for (int rb = 0; rb < OH / TOH; rb++) {
            const int c0 = cb * TC, oh0 = rb * TOH;
            const bfloat16 *w = aie::load_buffer(w_port, {c0, 0}, {TC, 9});
            const float *bias = (const float *)aie::load_buffer(b_port, {c0, 0}, {TC, 2});
            const bfloat16 *x = aie::load_buffer(x_port, {c0, oh0, 0}, {TC, IHT, IWT});
            bfloat16 *y = aie::store_buffer(y_port, {c0, oh0, 0}, {TC, TOH, OW});
            for (int c = 0; c < TC; c++) {
                aie::vector<bfloat16, 16> tap[9];
                for (int t = 0; t < 9; t++)
                    tap[t] = aie::broadcast<bfloat16, 16>(w[c * 9 + t]);
                const aie::vector<float, 16> bv = aie::broadcast<float, 16>(bias[c]);
                for (int r = 0; r < TOH; r++)
                    for (int j = 0; j < NJ; j++) {
                        aie::accum<accfloat, 16> a = aie::zeros<accfloat, 16>();
                        for (int kh = 0; kh < 3; kh++)
                            for (int kw = 0; kw < 3; kw++)
                                a = aie::mac(a, aie::load_unaligned_v<16>(x + (c * IHT + r + kh) * IWT + j * 16 + kw),
                                             tap[kh * 3 + kw]);
                        aie::vector<float, 16> v = aie::add(a.template to_vector<float>(), bv);
                        if constexpr (ACT == 1)
                            v = aie::max(v, aie::zeros<float, 16>());
                        if constexpr (ACT == 2) {
                            const aie::vector<float, 16> e = aie::exp2<float>(aie::mul(v, -1.4426950408889634f).template to_vector<float>());
                            v = aie::mul(v, aie::inv(aie::add(e, aie::broadcast<float, 16>(1.0f)))).template to_vector<float>();
                        }
                        aie::accum<accfloat, 16> o;
                        o.from_vector(v);
                        const aie::vector<bfloat16, 16> ob = o.template to_vector<bfloat16>();
                        bfloat16 *yr = y + (c * TOH + r) * OW + j * 16;
                        if (j * 16 + 16 <= OW)
                            aie::store_unaligned_v(yr, ob);
                        else
                            for (int i = 0; i < OW - j * 16; i++)
                                yr[i] = ob[i];
                    }
            }
            aie::commit(y_port);
            aie::release(x_port);
            aie::release(b_port);
            aie::release(w_port);
        }
}
