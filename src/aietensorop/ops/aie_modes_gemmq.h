/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_gemmq.h -- the GEMM + integer requant mode (0xF1) in vector lanes.
 *
 * The host's exact epilogue per output (rt_fx_q): q = ((acc * M + B + (r - rzp) * R) >> sh) + zp,
 * clamped to [lo, hi], per channel M / B / sh (/ R with a fused residual r). The GEMMQ row: {zp, lo,
 * hi} int32 at +4, B int64[32] at +16, M int32[32] at +272, sh int8[32] at +400; the residual row:
 * R int64[32] at +0, rzp int32 at +256. The scalar loop cost an int64 multiply-add-shift per output
 * (~200k cycles a launch, most of it). Here an 8x8 mmul result is four 16-lane groups (rows 2h,
 * 2h+1 x channels ct*8 .. +7): the sum in acc64 lanes (B and R*t as hi * 2^31 + lo), one floor shift
 * by S = min sh, then * 2^(D - e) (int16) and a floor shift by D (e = sh - S, D = max e):
 * floor(floor(x / 2^S) / 2^e) = floor(x / 2^sh) -- the MMT requant's scheme (aie_modes_mmt.h), on purpose
 * also in not touching the saturation mode: switching it around the block (to make overflowing lanes
 * saturate) gave 0 for ~75 % of the outputs that clamp to hi on the board. No lane overflows: M <
 * 2^31 and sh >= 31 (scale < 1) keep acc * M / 2^S inside int32, and the host refuses the integer
 * epilogue for a shift spread above 14 (rt_fx_init), so 2^(D - e) fits int16.
 ******************************************************************************/
#ifndef AIE_MODES_GEMMQ_H
#define AIE_MODES_GEMMQ_H

#if defined(__AIENGINE__) && defined(AIEK_MODE_GEMMQ) && AIEK_MODE_GEMMQ
/* The core's whole int32 tile (rows x 32 channels, row pitch 32 int32 = 128 bytes) -> int8 in place:
 * row r's channels at bytes [r * 128, +32). One pass after the K rounds, one call site: requantizing
 * each 8x8 block where the mmul produced it inlined 16 copies (17 KB of program memory). Channel group
 * ct outer, row pairs inner: group ct's int8 bytes land on int32 channels 0..7 of the same rows, which
 * group 0 has already read. Per group ~180 bytes of temporaries (the GEMMQ + CONV core runs within
 * ~2 KB of its 32 KB stack). rq / rt: the residual row and the core's residual tile (32 x 32 int8), or
 * null. */
static inline void aie_gq_tile(int32_t *c, int rows, const uint8_t *qrow, const uint8_t *rq, const int8_t *rt) {
    const int32_t *qp = (const int32_t *)(qrow + 4);
    const int64_t *qb = (const int64_t *)(qrow + 16);
    const int32_t *qm = (const int32_t *)(qrow + 272);
    const int8_t *qsh = (const int8_t *)(qrow + 400);
    int S = 127, hi = 0;
    for (int ch = 0; ch < 32; ch++) {
        S = qsh[ch] < S ? qsh[ch] : S;
        hi = qsh[ch] > hi ? qsh[ch] : hi;
    }
    const int D = hi - S;
    const int32 zp = qp[0];
    const aie::vector<int32, 16> vzp = aie::broadcast<int32, 16>(zp);
    const aie::vector<int32, 16> vlo = aie::broadcast<int32, 16>(qp[1] - zp),
                                 vhi = aie::broadcast<int32, 16>(qp[2] - zp);
    const aie::vector<int32, 16> rzp = aie::broadcast<int32, 16>(rq ? *(const int32_t *)(rq + 256) : 0);
    const aie::rounding_mode rnd = aie::get_rounding();
    aie::set_rounding(aie::rounding_mode::floor);
    alignas(32) int32 m8[8], bh8[8], bl8[8], rh8[8], rl8[8];
    alignas(16) int16 mul8[8];
    int8_t *ob = (int8_t *)c;
    for (int ct = 0; ct < 4; ct++) {
        for (int j = 0; j < 8; j++) {
            const int ch = ct * 8 + j;
            m8[j] = qm[ch];
            bh8[j] = (int32)(qb[ch] >> 31), bl8[j] = (int32)(qb[ch] & 0x7fffffff);
            mul8[j] = (int16)(1 << (D - (qsh[ch] - S)));
            const int64_t R = rq ? ((const int64_t *)rq)[ch] : 0;
            rh8[j] = (int32)(R >> 31), rl8[j] = (int32)(R & 0x7fffffff);
        }
        const aie::vector<int32, 8> m_ = aie::load_v<8>(m8), bh_ = aie::load_v<8>(bh8), bl_ = aie::load_v<8>(bl8);
        const aie::vector<int32, 8> rh_ = aie::load_v<8>(rh8), rl_ = aie::load_v<8>(rl8);
        const aie::vector<int16, 8> mu_ = aie::load_v<8>(mul8);
        const aie::vector<int32, 16> m = aie::concat(m_, m_), bh = aie::concat(bh_, bh_), bl = aie::concat(bl_, bl_);
        const aie::vector<int32, 16> rh = aie::concat(rh_, rh_), rl = aie::concat(rl_, rl_);
        const aie::vector<int16, 16> mul = aie::concat(mu_, mu_);
        for (int r = 0; r < rows; r += 2) { /* rows r, r + 1 of channels ct*8 .. +7 */
            const aie::vector<int32, 8> a0 = aie::load_v<8>(c + r * 32 + ct * 8),
                                        a1 = aie::load_v<8>(c + (r + 1) * 32 + ct * 8);
            aie::accum<acc64, 16> x = aie::mul(aie::concat(a0, a1), m);
            aie::accum<acc64, 16> b;
            b.from_vector(bh, 31);
            x = aie::add(x, b);
            b.from_vector(bl, 0);
            x = aie::add(x, b);
            if (rt) { /* + (r - rzp) * R: |r - rzp| <= 383 and R < 2^53, so (r - rzp) * Rh fits int32 */
                alignas(16) int8 r8[16];
                uint32_t *rw = (uint32_t *)r8;
                const uint32_t *r0 = (const uint32_t *)(rt + r * 32 + ct * 8),
                               *r1 = (const uint32_t *)(rt + (r + 1) * 32 + ct * 8);
                rw[0] = r0[0], rw[1] = r0[1], rw[2] = r1[0], rw[3] = r1[1];
                const aie::vector<int32, 16> t = aie::sub(aie::unpack(aie::unpack(aie::load_v<16>(r8))), rzp);
                b.from_vector(aie::mul(t, rh).template to_vector<int32>(0), 31);
                x = aie::add(x, b);
                x = aie::add(x, aie::mul(t, rl));
            }
            const aie::vector<int32, 16> y = x.template to_vector<int32>(S);
            aie::vector<int32, 16> q = aie::mul(y, mul).template to_vector<int32>(D);
            q = aie::add(aie::max(aie::min(q, vhi), vlo), vzp);
            alignas(16) int8 p[16]; /* rows r, r + 1: 8 bytes each (no 8-lane int8 vector) */
            aie::store_v(p, q.template pack<int8>());
            const uint32_t *w = (const uint32_t *)p;
            uint32_t *o0 = (uint32_t *)(ob + r * 128 + ct * 8), *o1 = (uint32_t *)(ob + (r + 1) * 128 + ct * 8);
            o0[0] = w[0], o0[1] = w[1], o1[0] = w[2], o1[1] = w[3];
        }
    }
    aie::set_rounding(rnd);
}
#endif

#endif
