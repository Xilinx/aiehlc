/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_dwq.h -- the core side of the depthwise + requant mode (0xD8) in vector lanes.
 *
 * One item: the zero-padded input rows of one output band of one channel (row pitch wp), its KHxKW
 * weights at byte 4096 - 64, and {zp, lo, hi, M} int32, B int64, sh int32 at 4096 - 96 (the host's
 * dw_pack_item). Output: the int8 band [rout][wo]. 32 output columns per step: per tap one 32-lane
 * int8 x int8 MAC into acc32 (stride 2: the even lanes of 64 loaded bytes), then the host's exact
 * requant ((acc * M + B) >> sh) + zp clamped to [lo, hi], in acc64 lanes: acc * M, plus B as
 * Bh 2^31 + Bl, one floor shift by sh. Bit-identical to dwq_blk_s8s8_ref.
 *
 * A step writes all 32 lanes, up to 31 bytes past the band: rows go in order, so a row's overhang
 * is rewritten by the next row, and the last row's lands in the unused rest of the 4 KB C tile.
 ******************************************************************************/
#ifndef AIE_MODES_DWQ_H
#define AIE_MODES_DWQ_H

#if defined(__AIENGINE__) && defined(AIEK_MODE_DWQ) && AIEK_MODE_DWQ
static inline void aie_dwq_item(const int8_t *raw, int s, int kh_n, int kw_n, int wp, int rout, int wo, int8_t *ob) {
    const int8_t *wq = raw + 4096 - 64;
    const int32_t *qp = (const int32_t *)(raw + 4096 - 96);
    const int64_t B = *(const int64_t *)(qp + 4);
    const int sh = qp[6];
    const aie::vector<int32, 16> m = aie::broadcast<int32, 16>(qp[3]);
    const aie::vector<int32, 16> bh = aie::broadcast<int32, 16>((int32)(B >> 31));
    const aie::vector<int32, 16> bl = aie::broadcast<int32, 16>((int32)(B & 0x7fffffff));
    const aie::vector<int32, 16> zp = aie::broadcast<int32, 16>(qp[0]);
    const aie::vector<int32, 16> lo = aie::broadcast<int32, 16>(qp[1] - qp[0]),
                                 hi = aie::broadcast<int32, 16>(qp[2] - qp[0]);
    const aie::rounding_mode rnd = aie::get_rounding();
    const aie::saturation_mode sat = aie::get_saturation();
    aie::set_rounding(aie::rounding_mode::floor);
    aie::set_saturation(aie::saturation_mode::saturate);
    for (int oh = 0; oh < rout; oh++)
        for (int ow = 0; ow < wo; ow += 32) {
            const int8_t *in = raw + oh * s * wp + ow * s;
            aie::accum<acc32, 32> acc = aie::zeros<acc32, 32>();
            for (int kh = 0; kh < kh_n; kh++)
                for (int kw = 0; kw < kw_n; kw++) {
                    const int8 *p = (const int8 *)(in + kh * wp + kw);
                    const aie::vector<int8, 32> x =
                        s == 1 ? aie::load_unaligned_v<32>(p) : aie::filter_even(aie::load_unaligned_v<64>(p));
                    acc = aie::mac(acc, x, aie::broadcast<int8, 32>((int8)wq[kh * kw_n + kw]));
                }
            const aie::vector<int32, 32> a = acc.template to_vector<int32>(0);
            aie::vector<int8, 32> out;
            for (int h = 0; h < 2; h++) {
                aie::accum<acc64, 16> x = aie::mul(a.template extract<16>(h), m);
                aie::accum<acc64, 16> b;
                b.from_vector(bh, 31);
                x = aie::add(x, b);
                b.from_vector(bl, 0);
                x = aie::add(x, b);
                aie::vector<int32, 16> v = x.template to_vector<int32>(sh);
                v = aie::add(aie::max(aie::min(v, hi), lo), zp);
                out.insert(h, v.template pack<int8>());
            }
            aie::store_unaligned_v((int8 *)(ob + oh * wo + ow), out);
        }
    aie::set_rounding(rnd);
    aie::set_saturation(sat);
}
#endif

#endif
