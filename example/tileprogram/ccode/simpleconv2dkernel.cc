/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "simpleconv2dkernel.h"

using Ifm = aie::tensor<int8_t, 224, 224, 3>;  // the REAL input: no manual padding
using Wgt = aie::tensor<int8_t, 64, 7, 7, 3>;  // OHWI
using Ofm = aie::tensor<int8_t, 112, 112, 64>; // HWC

template <int OH_T = 7, int OW_T = 28, int OC_T = 16>
__global__ aie::autotune<OH_T, {7, 14, 28}, OW_T, {14, 28, 56}, OC_T, {16, 32}> void
conv2d(aie::in<Ifm> x_t, aie::in<Wgt> w_t, aie::out<Ofm> y_t) {
    constexpr int KH = 7, KW = 7, S = 2, P = 3, C = 3;
    constexpr int IH_T = (OH_T - 1) * S + KH, IW_T = (OW_T - 1) * S + KW;
    constexpr int OH_CORE = Ofm::dim(0) / aie::num_programs(0);

    const int pr = aie::program_id(0), pc = aie::program_id(1);
    auto w =
        aie::load_buffer(w_t, {pc * OC_T, 0, 0, 0}, {OC_T, KH, KW, C}, {.reuse = OH_CORE / OH_T * Ofm::dim(1) / OW_T});

    for (int i = 0; i < OH_CORE / OH_T; i++)
        for (int j = 0; j < Ofm::dim(1) / OW_T; j++) {
            const int oh0 = pr * OH_CORE + i * OH_T, ow0 = j * OW_T;
            // Negative / past-the-end rows and cols are the conv padding: zero-filled.
            auto x =
                aie::load_buffer(x_t, {oh0 * S - P, ow0 * S - P, 0}, {IH_T, IW_T, C}, {.pad = aie::pad_value::zero});
            auto y = aie::store_buffer(y_t, {oh0, ow0, pc * OC_T}, {OH_T, OW_T, OC_T});
            for (int oh = 0; oh < OH_T; oh++)
                for (int ow = 0; ow < OW_T; ow++)
                    for (int oc = 0; oc < OC_T; oc++) {
                        int32_t acc = 0;
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                                for (int c = 0; c < C; c++)
                                    acc += x(oh * S + kh, ow * S + kw, c) * w(oc, kh, kw, c);
                        y(oh, ow, oc) = aie::saturate<int8_t>(acc);
                    }
        } // x released, y committed here
}

int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 4);
    auto in = aie::alloc<Ifm>(), wt = aie::alloc<Wgt>();
    auto out = aie::alloc<Ofm>();
    init_input(in);
    init_filter(wt);
    conv2d<<<mesh>>>(in, wt, out); // tuned block sizes chosen at build time
    return verify(in, wt, out);
}