/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_fpool.h -- f32 windowed pooling (2-D and 3-D, max and sum) as a mode of the GEMM ELF
 * (opcode 0xB2), the floating counterpart of the 8-bit pool mode (aie_modes_pool.h).
 *
 * Item (one per core, the 4 KB gathered from its A rows, read as 1024 floats): KD input slices of
 * RIN = (ROUT - 1) * SH + KH rows of pitch WP floats, already padded (-FLT_MAX for max, 0 for sum);
 * slice q starts at float q * RIN * WP. out[oh * WO + ow] = op over q < KD, y < KH, x < KW of
 * in[q * RIN * WP + (oh * SH + y) * WP + ow * SW + x]; the rest of the 1024 outputs are 0. Max is exact;
 * a sum is the f32 sum in tap order (the host divides for an average).
 *
 * Op 2, ARGMAX (max pooling backward): out[o] = the int32 tap index t = (q * KH + y) * KW + x of the
 * window's FIRST strict maximum in (q, y, x) order -- ref:'s rule -- written as int32 bits; the host routes
 * dy through it (aiednn_rt_fpool.h), in double as ref: does, so the gradient stays bit-identical.
 *
 * Descriptor (D = the bytes after opcode and column): D[0] op 0 max / 1 sum / 2 argmax, D[1] KD, D[2] KH, D[3] KW,
 * D[4] SH, D[5] SW (1 or 2), D[6..7] WP, D[8..9] ROUT, D[10..11] WO (LE16).
 * Needs KD * RIN * WP <= 1024 and ROUT * WO <= 1024.
 *
 * The core path is aie_api vector code only (AIE-ML has no scalar FPU: a scalar float op pulls softfloat
 * into program memory); aie_fpool_item_ref is the plain C reference (x86 stand-in, gemm_uk_ref.c).
 ******************************************************************************/
#ifndef AIE_MODES_FPOOL_H
#define AIE_MODES_FPOOL_H

#include <stdint.h>
#include <string.h>

#define AIE_MODE_OP_FPOOL 0xB2
#define AIE_FPOOL_IN 1024  /* floats per item */
#define AIE_FPOOL_OUT 1024 /* outputs per item */
#define AIE_FPOOL_NEG (-3.4028234663852886e38f)

/* geom int32 {op, kd, kh, kw, sh | sw << 8, wo, wp, rout} -> the 12 descriptor bytes */
#define AIE_FPOOL_DESC(D, G)                                                                                           \
    do {                                                                                                               \
        (D)[0] = (uint8_t)(G)[0], (D)[1] = (uint8_t)(G)[1], (D)[2] = (uint8_t)(G)[2], (D)[3] = (uint8_t)(G)[3];        \
        (D)[4] = (uint8_t)((G)[4] & 0xff), (D)[5] = (uint8_t)((G)[4] >> 8);                                            \
        (D)[6] = (uint8_t)(G)[6], (D)[7] = (uint8_t)((G)[6] >> 8);                                                     \
        (D)[8] = (uint8_t)(G)[7], (D)[9] = (uint8_t)((G)[7] >> 8);                                                     \
        (D)[10] = (uint8_t)(G)[5], (D)[11] = (uint8_t)((G)[5] >> 8);                                                   \
    } while (0)

#define AIE_FPOOL_LE16(D, i) ((int)(D)[i] | ((int)(D)[(i) + 1] << 8))

/* plain C: the reference of one item (x86 stand-in and tests) */
static inline void aie_fpool_item_ref(const float *in, float *out, const uint8_t *d) {
    const int op = d[0], kd = d[1], kh = d[2], kw = d[3], sh = d[4], sw = d[5];
    const int wp = AIE_FPOOL_LE16(d, 6), rout = AIE_FPOOL_LE16(d, 8), wo = AIE_FPOOL_LE16(d, 10);
    const int pp = ((rout - 1) * sh + kh) * wp;
    for (int i = 0; i < AIE_FPOOL_OUT; i++)
        out[i] = 0.f;
    if (op == 2) {
        int32_t *oi = (int32_t *)out;
        for (int oh = 0; oh < rout; oh++)
            for (int ow = 0; ow < wo; ow++) {
                float bv = 0.f;
                int32_t bt = -1;
                for (int t = 0; t < kd * kh * kw; t++) {
                    const int q = t / (kh * kw), y = t / kw % kh, x = t % kw;
                    const float e = in[q * pp + (oh * sh + y) * wp + ow * sw + x];
                    if (bt < 0 || e > bv)
                        bv = e, bt = t;
                }
                oi[oh * wo + ow] = bt;
            }
        return;
    }
    for (int oh = 0; oh < rout; oh++)
        for (int ow = 0; ow < wo; ow++) {
            float v = op == 0 ? AIE_FPOOL_NEG : 0.f;
            for (int q = 0; q < kd; q++)
                for (int y = 0; y < kh; y++)
                    for (int x = 0; x < kw; x++) {
                        const float e = in[q * pp + (oh * sh + y) * wp + ow * sw + x];
                        v = op == 0 ? (e > v ? e : v) : v + e;
                    }
            out[oh * wo + ow] = v;
        }
}

#if defined(__AIENGINE__) || defined(AIE_FPOOL_EMU)
/* 16 outputs of a row at a time; taps of 16 lanes (stride 1: one unaligned load, stride 2: the even
 * lanes of 32). A row's tail computes a full vector and keeps its first lanes. */
static inline void aie_fpool_item_vec(const float *in, float *out, const uint8_t *d) {
    const int op = d[0], kd = d[1], kh = d[2], kw = d[3], sh = d[4], sw = d[5];
    const int wp = AIE_FPOOL_LE16(d, 6), rout = AIE_FPOOL_LE16(d, 8), wo = AIE_FPOOL_LE16(d, 10);
    const int pp = ((rout - 1) * sh + kh) * wp;
    const aie::vector<float, 16> init = aie::broadcast<float, 16>(op == 0 ? AIE_FPOOL_NEG : 0.f);
    alignas(32) float lane[16];
    for (int i = 0; i < AIE_FPOOL_OUT; i += 16)
        aie::store_v(out + i, aie::zeros<float, 16>());
    if (op == 2) { /* first strict maximum: a later tap wins only when greater */
        alignas(32) int32_t ilane[16];
        int32_t *oi = (int32_t *)out;
        for (int oh = 0; oh < rout; oh++)
            for (int ow0 = 0; ow0 < wo; ow0 += 16) {
                aie::vector<float, 16> best;
                aie::vector<int32_t, 16> bidx = aie::zeros<int32_t, 16>();
                int t = 0;
                for (int q = 0; q < kd; q++)
                    for (int y = 0; y < kh; y++) {
                        const float *row = in + q * pp + (oh * sh + y) * wp + ow0 * sw;
                        for (int x = 0; x < kw; x++, t++) {
                            const aie::vector<float, 16> e =
                                sw == 1 ? aie::load_unaligned_v<16>(row + x)
                                        : aie::filter_even(aie::load_unaligned_v<32>(row + x), 1);
                            if (t == 0) {
                                best = e;
                                continue;
                            }
                            const auto m = aie::gt(e, best);
                            best = aie::select(best, e, m);
                            bidx = aie::select(bidx, aie::broadcast<int32_t, 16>(t), m);
                        }
                    }
                const int n = wo - ow0 < 16 ? wo - ow0 : 16;
                aie::store_v(ilane, bidx);
                memcpy(oi + oh * wo + ow0, ilane, (size_t)n * sizeof(int32_t));
            }
        return;
    }
    for (int oh = 0; oh < rout; oh++)
        for (int ow0 = 0; ow0 < wo; ow0 += 16) {
            aie::vector<float, 16> acc = init;
            for (int q = 0; q < kd; q++)
                for (int y = 0; y < kh; y++) {
                    const float *row = in + q * pp + (oh * sh + y) * wp + ow0 * sw;
                    for (int x = 0; x < kw; x++) {
                        const aie::vector<float, 16> e = sw == 1
                                                             ? aie::load_unaligned_v<16>(row + x)
                                                             : aie::filter_even(aie::load_unaligned_v<32>(row + x), 1);
                        acc = op == 0 ? aie::max(acc, e) : aie::add(acc, e);
                    }
                }
            const int n = wo - ow0 < 16 ? wo - ow0 : 16;
            if (n == 16) {
                aie::store_unaligned_v(out + oh * wo + ow0, acc);
            } else {
                aie::store_v(lane, acc);
                memcpy(out + oh * wo + ow0, lane, (size_t)n * sizeof(float)); /* bytes, no float op */
            }
        }
}
#define aie_fpool_item aie_fpool_item_vec
#else
#define aie_fpool_item aie_fpool_item_ref
#endif

#endif /* AIE_MODES_FPOOL_H */
