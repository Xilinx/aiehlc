/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_poolv.h -- the pool mode's vector path (aie_modes_pool.h AIE_MODE_POOL on the core).
 *
 * MAX and SUM windows (max / avg pooling, the sum reductions) with stride 1 or 2: 16 outputs of an
 * output row at a time, int32 lanes. Each tap is one unaligned 16-byte load (stride 1) or a 32-byte
 * load and its even lanes (stride 2), widened int8 -> int16 -> int32. uint8 input is taken as int8
 * after an xor with 0x80 (x - 128), and the result corrected exactly: + 128 for a max, + 128 * KH * KW
 * for a sum. Integer max / sum, so the result is AIE_MODE_POOL's bit for bit; anything else (min, the
 * |x| / x^2 variants, other strides, a row tail shorter than 16) returns 0 or falls back per output.
 *
 * Core build (aie_api) and the x86 AIE-API emulation (src/mlir/runtime/aie_api_x86emu.h) both compile
 * it: test/poolv_test.cc checks it against AIE_MODE_POOL. Include after aie_modes_pool.h.
 ******************************************************************************/
#ifndef AIE_MODES_POOLV_H
#define AIE_MODES_POOLV_H

#if defined(__AIENGINE__) || defined(AIE_POOLV_EMU)

/* 16 taps' bytes at p (stride s in {1, 2}) as int32 lanes; ux: 0x80 for uint8 (x - 128), else 0 */
static inline aie::vector<int32, 16> aie_poolv_ld(const int8 *p, int s, aie::vector<int8, 16> ux) {
    aie::vector<int8, 16> v;
    if (s == 1)
        v = aie::load_unaligned_v<16>(p);
    else
        v = aie::filter_even(aie::load_unaligned_v<32>(p), 1);
    return aie::unpack(aie::unpack(aie::bit_xor(v, ux)));
}

/* AIE_MODE_POOL's output for one item, or 0 if this path does not take the descriptor */
static inline int aie_pool_vec(const int8_t *in, int32_t *out, const uint8_t *d) {
    const int op = d[0], sgn = d[1], s = d[2], kh = d[3], kw = d[4];
    const int wp = d[5] | (d[6] << 8), rout = d[7] | (d[8] << 8), wo = d[9] | (d[10] << 8);
    if ((op != AIE_MODE_POOL_MAX && op != AIE_MODE_POOL_SUM) || (s != 1 && s != 2) || wo < 16 ||
        rout * wo > AIE_MODE_OUT)
        return 0;
    const aie::vector<int8, 16> ux = aie::broadcast<int8, 16>((int8)(sgn ? 0 : -128));
    const int32_t fix = sgn ? 0 : op == AIE_MODE_POOL_MAX ? 128 : 128 * kh * kw;
    const aie::vector<int32, 16> init = aie::broadcast<int32, 16>(op == AIE_MODE_POOL_MAX ? -2147483647 - 1 : 0);
    const aie::vector<int32, 16> fixv = aie::broadcast<int32, 16>(fix);
    alignas(32) int32_t lane[16]; /* AIE-ML: stack alignment <= 32 */
    for (int i = rout * wo; i < AIE_MODE_OUT; i++)
        out[i] = 0;
    for (int oh = 0; oh < rout; oh++) {
        int ow0 = 0;
        for (; ow0 + 16 <= wo; ow0 += 16) {
            const int8 *row = (const int8 *)in + oh * s * wp + ow0 * s;
            aie::vector<int32, 16> acc = init;
            for (int y = 0; y < kh; y++)
                for (int x = 0; x < kw; x++) {
                    const aie::vector<int32, 16> e = aie_poolv_ld(row + y * wp + x, s, ux);
                    acc = op == AIE_MODE_POOL_MAX ? aie::max(acc, e) : aie::add(acc, e);
                }
            aie::store_v(lane, aie::add(acc, fixv));
            for (int i = 0; i < 16; i++)
                out[oh * wo + ow0 + i] = lane[i];
        }
        for (; ow0 < wo; ow0++) { /* the row's tail: AIE_MODE_POOL's loop for one output */
            const int base = oh * s * wp + ow0 * s;
            int32_t v = op == AIE_MODE_POOL_MAX ? -2147483647 - 1 : 0;
            for (int y = 0; y < kh; y++)
                for (int x = 0; x < kw; x++) {
                    const int32_t e = AIE_MODE_LD(in, base + y * wp + x, sgn);
                    v = op == AIE_MODE_POOL_MAX ? (e > v ? e : v) : v + e;
                }
            out[oh * wo + ow0] = v;
        }
    }
    return 1;
}

#endif
#endif /* AIE_MODES_POOLV_H */
