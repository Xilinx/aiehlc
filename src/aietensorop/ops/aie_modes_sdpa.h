/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_sdpa.h -- scaled dot-product attention as a mode of the GEMM ELF (opcode 0x5D).
 *
 * One item per core, as the pool / eltwise modes (aie_modes_pool.h): IN is the core's 4 KB item,
 * OUT its 1024-entry int32 C tile. An item is one (batch, head, query block) of int8 tensors:
 *   Q [Sq][D] at 0, K^T [D][Skp] at Sq*D (keys padded to Skp = Skv rounded up to 16, zero),
 *   V [Skv][Dv] at Sq*D + D*Skp  (Sq*D + D*Skp + Skv*Dv <= 4092),
 *   and int32 LE at AIE_SDPA_COFF: the causal offset c of this item (query row i sees keys j <= i + c;
 *   the item's first row is global query q0, so c = q0 + Skv_total - Sq_total)
 * and OUT[i*Dv + e] (Sq*Dv <= 1024) = sum_j w_ij V[j][e] / sum_j w_ij in Q16, where
 *   s_ij = sum_e Q[i][e] K[j][e]                       exact int32
 *   w_ij = 2^(-t_ij / 65536) in Q14,  t_ij = ((max_j s_ij - s_ij) * M) >> SH  (Q14: an int16 lane)
 * M / 2^SH = the score scale (sdpa's scale times the Q and K quantization steps) * log2(e) * 65536,
 * so w_ij = exp(-(max - s) * scale) up to the table: 2^-x from 65 entries (x in 1/64 steps) with
 * linear interpolation, relative error < 2e-5. Causal: key j > i + c gets no weight; a row
 * without a key is 0. The host dequantizes OUT (/ 65536 * V's step) and stores like ref:sdpa.
 * Integer arithmetic only, the same on the core and on x86, so the reference is this text.
 *
 * Descriptor (D = the bytes after opcode and column): D[0..1] Sq, D[2..3] Skv (LE16), D[4] D,
 * D[5] Dv, D[6] causal, D[7] SH, D[8..11] M (int32 LE). Host geometry geom[8] =
 * {Sq, Skv, D, Dv, causal, M, SH, 0} (AIE_MODE_SDPA_DESC packs it).
 * Limits (the host checks): Skv <= AIE_SDPA_MAX_KV (the 32-bit sums), D, Dv <= 255.
 *
 * Kernel side: include between AIEHLC_KERNEL_PROLOGUE_BEGIN/END, after aie_modes_pool.h.
 ******************************************************************************/
#ifndef AIE_MODES_SDPA_H
#define AIE_MODES_SDPA_H

#include <stdint.h>

#define AIE_MODE_OP_SDPA 0x5D
#define AIE_SDPA_MAX_KV 256 /* sum_j w * v < 2^31: 256 * 32768 * 128 */
#define AIE_SDPA_COFF 4092  /* the item's causal offset (int32 LE) */

#define AIE_MODE_SDPA_DESC(D, g)                                                                                       \
    do {                                                                                                               \
        (D)[0] = (uint8_t)((g)[0] & 0xff), (D)[1] = (uint8_t)((g)[0] >> 8);                                            \
        (D)[2] = (uint8_t)((g)[1] & 0xff), (D)[3] = (uint8_t)((g)[1] >> 8);                                            \
        (D)[4] = (uint8_t)(g)[2], (D)[5] = (uint8_t)(g)[3], (D)[6] = (uint8_t)(g)[4], (D)[7] = (uint8_t)(g)[6];        \
        for (int b_ = 0; b_ < 4; b_++)                                                                                 \
            (D)[8 + b_] = (uint8_t)((uint32_t)(g)[5] >> (8 * b_));                                                     \
    } while (0)

/* 2^(-k/64) in Q15, k = 0..64 */
static const int32_t aie_sdpa_exp2[65] = {32768, 32415, 32066, 31720, 31379, 31041, 30706, 30376, 30048, 29725, 29405,
                                          29088, 28774, 28464, 28158, 27855, 27554, 27258, 26964, 26674, 26386, 26102,
                                          25821, 25543, 25268, 24995, 24726, 24460, 24196, 23936, 23678, 23423, 23170,
                                          22921, 22674, 22430, 22188, 21949, 21713, 21479, 21247, 21019, 20792, 20568,
                                          20347, 20127, 19911, 19696, 19484, 19274, 19066, 18861, 18658, 18457, 18258,
                                          18061, 17867, 17674, 17484, 17296, 17109, 16925, 16743, 16562, 16384};

/* 2^(-t / 65536) in Q14 for t >= 0 (<= 16384: an int16 lane) */
static inline int32_t aie_sdpa_w(int64_t t) {
    const int64_t ip = t >> 16;
    if (ip >= 15)
        return 0;
    const int32_t fr = (int32_t)(t & 0xFFFF), k = fr >> 10, r = fr & 1023;
    const int32_t a = aie_sdpa_exp2[k], b = aie_sdpa_exp2[k + 1];
    return (a - (((a - b) * r) >> 10)) >> (ip + 1);
}

/* one item: IN as above, OUT[AIE_MODE_OUT] (the unused entries zero), S scratch of Skp int32. The scalar
 * reference (x86 stand-ins; the core runs aie_sdpa_item_vec, the same integers). */
static void aie_sdpa_item_ref(const int8_t *in, int32_t *out, const uint8_t *d, int32_t *s) {
    /* (no single-letter upper-case names: gemm.h's M / N / K are macros in the kernel's TU) */
    const int Sq = (int)d[0] | ((int)d[1] << 8), Skv = (int)d[2] | ((int)d[3] << 8), dq = d[4], dv = d[5];
    const int causal = d[6], sh = d[7];
    const int32_t mul =
        (int32_t)((uint32_t)d[8] | ((uint32_t)d[9] << 8) | ((uint32_t)d[10] << 16) | ((uint32_t)d[11] << 24));
    const int skp = (Skv + 15) & ~15;
    const int8_t *q = in, *kt = in + Sq * dq, *v = kt + dq * skp;
    const uint8_t *cb = (const uint8_t *)in + AIE_SDPA_COFF;
    const int32_t coff =
        (int32_t)((uint32_t)cb[0] | ((uint32_t)cb[1] << 8) | ((uint32_t)cb[2] << 16) | ((uint32_t)cb[3] << 24));
    for (int o = 0; o < 1024; o++)
        out[o] = 0;
    for (int i = 0; i < Sq; i++) {
        const int jn = causal ? (i + coff + 1 < Skv ? i + coff + 1 : Skv) : Skv; /* keys 0 .. jn-1 */
        if (jn <= 0)
            continue;
        int32_t mx = 0;
        for (int j = 0; j < jn; j++) {
            int32_t acc = 0;
            for (int e = 0; e < dq; e++)
                acc += (int32_t)q[i * dq + e] * (int32_t)kt[e * skp + j];
            s[j] = acc;
            mx = j == 0 || acc > mx ? acc : mx;
        }
        int32_t z = 0;
        for (int j = 0; j < jn; j++) {
            s[j] = aie_sdpa_w(((int64_t)(mx - s[j]) * mul) >> sh);
            z += s[j];
        }
        const int64_t inv = ((int64_t)1 << 47) / z; /* z >= 16384: the max's own weight */
        for (int e = 0; e < dv; e++) {
            int32_t o = 0;
            for (int j = 0; j < jn; j++)
                o += s[j] * (int32_t)v[j * dv + e];
            out[i * dv + e] = (int32_t)(((int64_t)o * inv + ((int64_t)1 << 30)) >> 31);
        }
    }
}

#if defined(__AIENGINE__) || defined(AIE_SDPA_EMU)
/* the core's version: 16 keys per vector for the scores (K^T rows), 16 value lanes for the weighted V */
static void aie_sdpa_item_vec(const int8_t *in, int32_t *out, const uint8_t *d, int32_t *s) {
    const int Sq = (int)d[0] | ((int)d[1] << 8), Skv = (int)d[2] | ((int)d[3] << 8), dq = d[4], dv = d[5];
    const int causal = d[6], sh = d[7];
    const int32_t mul =
        (int32_t)((uint32_t)d[8] | ((uint32_t)d[9] << 8) | ((uint32_t)d[10] << 16) | ((uint32_t)d[11] << 24));
    const int skp = (Skv + 15) & ~15;
    const int8 *q = (const int8 *)in, *kt = q + Sq * dq, *v = kt + dq * skp;
    const uint8_t *cb = (const uint8_t *)in + AIE_SDPA_COFF;
    const int32_t coff =
        (int32_t)((uint32_t)cb[0] | ((uint32_t)cb[1] << 8) | ((uint32_t)cb[2] << 16) | ((uint32_t)cb[3] << 24));
    alignas(32) int32_t lane[16];
    for (int o = 0; o < 1024; o++)
        out[o] = 0;
    for (int i = 0; i < Sq; i++) {
        const int jn = causal ? (i + coff + 1 < Skv ? i + coff + 1 : Skv) : Skv;
        if (jn <= 0)
            continue;
        for (int j0 = 0; j0 < jn; j0 += 16) { /* s[j0 .. +16) = sum_e q[i][e] * K^T[e][j0 ..] */
            aie::accum<acc32, 16> acc = aie::mul<acc32>(aie::load_unaligned_v<16>(kt + j0), q[i * dq]);
            for (int e = 1; e < dq; e++)
                acc = aie::mac(acc, aie::load_unaligned_v<16>(kt + e * skp + j0), q[i * dq + e]);
            aie::store_v(lane, acc.template to_vector<int32>(0));
            for (int l = 0; l < 16; l++)
                s[j0 + l] = lane[l];
        }
        int32_t mx = s[0];
        for (int j = 1; j < jn; j++)
            mx = s[j] > mx ? s[j] : mx;
        int32_t z = 0;
        for (int j = 0; j < jn; j++) {
            s[j] = aie_sdpa_w(((int64_t)(mx - s[j]) * mul) >> sh);
            z += s[j];
        }
        const int64_t inv = ((int64_t)1 << 47) / z;
        int e0 = 0;
        for (; e0 + 16 <= dv; e0 += 16) { /* 16 output lanes: sum_j w_j * V[j][e0 ..] */
            aie::accum<acc32, 16> acc = aie::mul<acc32>(aie::unpack(aie::load_unaligned_v<16>(v + e0)), (int16)s[0]);
            for (int j = 1; j < jn; j++)
                acc = aie::mac(acc, aie::unpack(aie::load_unaligned_v<16>(v + j * dv + e0)), (int16)s[j]);
            aie::store_v(lane, acc.template to_vector<int32>(0));
            for (int l = 0; l < 16; l++)
                out[i * dv + e0 + l] = (int32_t)(((int64_t)lane[l] * inv + ((int64_t)1 << 30)) >> 31);
        }
        for (int e = e0; e < dv; e++) { /* the tail lanes */
            int32_t o = 0;
            for (int j = 0; j < jn; j++)
                o += s[j] * (int32_t)v[j * dv + e];
            out[i * dv + e] = (int32_t)(((int64_t)o * inv + ((int64_t)1 << 30)) >> 31);
        }
    }
}
#endif

#if defined(__AIENGINE__)
#define aie_sdpa_item aie_sdpa_item_vec
#else
#define aie_sdpa_item aie_sdpa_item_ref
#endif

#endif /* AIE_MODES_SDPA_H */
