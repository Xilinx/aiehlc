/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_lut.h -- table lookup (LUT8) as a mode of the GEMM ELF, opcode 0x7A.
 *
 * Any elementwise function of one 8-bit value, per channel: out = T[ch][byte(in)].
 * The host builds T by running the reference on all 256 inputs of every channel,
 * so the core reproduces the reference bit for bit whatever the function (eltwise,
 * PReLU, batch norm, x op per-channel-constant, requantize / dequantize reorders).
 *
 * One item per core, as the depthwise / pool modes: IN is the core's 4 KB item
 * gathered from its A rows, OUT its 4 KB C tile. Element size ES of the result:
 * 1 (s8 / u8, n <= 4096) or 4 (s32 / f32 bit patterns, n <= 1024).
 *
 * Tables: AIE_LUT_TBYTES bytes per core, 256 * ES bytes each, slot k at k * 256 * ES
 * (30 one-byte or 7 four-byte tables). The entry places core (r, c)'s tables in rows
 * 15r .. 15r+14 of column c's B block (B row 127 is the descriptor row; 15 rows per
 * core for up to 8 mesh rows), and the core copies them out of its B windows.
 *
 * Element i of the item uses slot t: t starts at 0 with run0 elements left in the
 * first run; each later run is `inner` elements and moves to the next slot, wrapping
 * after slot nt - 1 to slot 0. A channel-major (NCHW) tensor has runs of H*W; the
 * host puts the item's channels in slots 0, 1, ... and, when an item spans more than
 * a full turn of channels, all C of them (nt = C) so the wrap is the next turn.
 *
 * Descriptor in B row 127 (first 64 bytes, i.e. in the k-round-0 window):
 *   [0] 0x7A  [1] mesh column  [2] ES  [3..7] reserved
 *   core in mesh row r: 7 bytes at 8 + 7r = {n LE16, run0 LE16, inner LE16, nt}
 * The gemm kernel hands the core's 7 bytes and ES to aie_lut_apply as D[0..7].
 *
 * STEPS (opcode 0x75) -- a 32-bit value -> 8-bit code, per channel, through a step
 * table: out = P[#{k : T[k] <= key}]. The key is the value itself (s32) or the
 * float's bit pattern made order-preserving (f32: negatives xor 0x7fffffff, so
 * -0 < +0 and int compares order floats; NaN -> the table's nan code). The host
 * finds every step of the reference by bisection over the key space, so any
 * function monotone between known breakpoints is reproduced bit for bit. Same
 * items, entries and table slots as LUT8, with 4-byte inputs (n <= 1024), 1-byte
 * results, slot k at k * stride:
 *     int32 n | int16 nan code | int16 0 | int32 T[n] (ascending) | int16 P[n + 1]
 * Descriptor [3] = key kind (0 f32, 1 s32), [4..5] = slot stride (LE16); the gemm
 * kernel passes them as D[8], D[9..10].
 *
 * Plain C: the same text is the kernel helper (included between
 * AIEHLC_KERNEL_PROLOGUE_BEGIN/END) and the x86 reference.
 ******************************************************************************/
#ifndef AIE_MODES_LUT_H
#define AIE_MODES_LUT_H

#include <stdint.h>

#define AIE_MODE_OP_LUT 0x7A
#define AIE_MODE_OP_STEPS 0x75
#define AIE_STEPS_F32 0
#define AIE_STEPS_S32 1
#define AIE_STEPS_HDR 8
#define AIE_STEPS_BYTES(n) (AIE_STEPS_HDR + 4 * (n) + 2 * ((n) + 1))
#define AIE_LUT_ROWS 15                     /* B rows (512 bytes) of tables per core */
#define AIE_LUT_TBYTES (AIE_LUT_ROWS * 512) /* 7680 */
#define AIE_LUT_ITEM 4096                   /* bytes in and out per core */
#define AIE_LUT_ENTRY(r) (8 + 7 * (r))      /* descriptor offset of mesh row r's entry */

/* host: one core's {n, run0, inner, nt} -> the 7 entry bytes at E */
static inline void aie_lut_entry(uint8_t *e, int n, int run0, int inner, int nt) {
    e[0] = (uint8_t)(n & 0xff), e[1] = (uint8_t)(n >> 8);
    e[2] = (uint8_t)(run0 & 0xff), e[3] = (uint8_t)(run0 >> 8);
    e[4] = (uint8_t)(inner & 0xff), e[5] = (uint8_t)(inner >> 8);
    e[6] = (uint8_t)nt;
}

/* host / x86: geom row {n, run0, inner, nt | es << 8 | stride/8 << 16 | kind << 26 | steps << 27}
 * -> the D bytes the gemm kernel hands to aie_lut_apply / aie_steps_apply */
static inline void aie_lut_desc(uint8_t *d, const int32_t *g) {
    const uint32_t g3 = (uint32_t)g[3], stride = ((g3 >> 16) & 0x3ff) * 8;
    aie_lut_entry(d, g[0], g[1], g[2], (int)(g3 & 0xff));
    d[7] = (uint8_t)(g3 >> 8), d[8] = (uint8_t)((g3 >> 26) & 1);
    d[9] = (uint8_t)(stride & 0xff), d[10] = (uint8_t)(stride >> 8);
}

/* D = the core's 7 entry bytes, then ES; TAB = its AIE_LUT_TBYTES of tables. Writes
 * all AIE_LUT_ITEM bytes of OUT (zero past n). */
static inline void aie_lut_apply(const uint8_t *in, uint8_t *out, const uint8_t *d, const uint8_t *tab) {
    const int n = d[0] | (d[1] << 8), nt = d[6], es = d[7];
    const int inner = (d[4] | (d[5] << 8)) > 0 ? (d[4] | (d[5] << 8)) : 1; /* a zero run never advances */
    int left = d[2] | (d[3] << 8), t = 0;
    const int cap = es == 4 ? AIE_LUT_ITEM / 4 : AIE_LUT_ITEM;
    const int cnt = n < cap ? n : cap;
    int i = 0;
    while (i < cnt) {
        int run = left < cnt - i ? left : cnt - i;
        if (es == 4) {
            const uint32_t *tw = (const uint32_t *)tab + t * 256;
            uint32_t *ow = (uint32_t *)out;
            for (int k = 0; k < run; k++)
                ow[i + k] = tw[in[i + k]];
        } else {
            const uint8_t *tb = tab + t * 256;
            for (int k = 0; k < run; k++)
                out[i + k] = tb[in[i + k]];
        }
        i += run;
        left = inner;
        t = t + 1 < nt ? t + 1 : 0;
    }
    for (int b = i * es; b < AIE_LUT_ITEM; b++)
        out[b] = 0;
}

/* f32 bit pattern -> order-preserving int32 key */
static inline int32_t aie_steps_key(int32_t w) { return w < 0 ? w ^ 0x7fffffff : w; }

/* D = entry bytes, ES, key kind, stride; IN = n 4-byte values; writes AIE_LUT_ITEM bytes of OUT */
static inline void aie_steps_apply(const uint8_t *in, uint8_t *out, const uint8_t *d, const uint8_t *tab) {
    const int n = d[0] | (d[1] << 8), nt = d[6], kind = d[8], stride = d[9] | (d[10] << 8);
    const int inner = (d[4] | (d[5] << 8)) > 0 ? (d[4] | (d[5] << 8)) : 1;
    const int cnt = n < AIE_LUT_ITEM / 4 ? n : AIE_LUT_ITEM / 4;
    const int32_t *w = (const int32_t *)in;
    int left = d[2] | (d[3] << 8), t = 0, i = 0;
    while (i < cnt) {
        const int run = left < cnt - i ? left : cnt - i;
        const uint8_t *tb = tab + t * stride;
        const int32_t ns = *(const int32_t *)tb;
        const int16_t nanc = *(const int16_t *)(tb + 4);
        const int32_t *T = (const int32_t *)(tb + AIE_STEPS_HDR);
        const int16_t *P = (const int16_t *)(tb + AIE_STEPS_HDR + 4 * ns);
        for (int k = i; k < i + run; k++) {
            const int32_t v = w[k];
            if (kind == AIE_STEPS_F32 && (v & 0x7fffffff) > 0x7f800000) {
                out[k] = (uint8_t)nanc;
                continue;
            }
            const int32_t key = kind == AIE_STEPS_F32 ? aie_steps_key(v) : v;
            int lo = 0, hi = ns; /* first T > key */
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (T[mid] <= key)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            out[k] = (uint8_t)P[lo];
        }
        i += run;
        left = inner;
        t = t + 1 < nt ? t + 1 : 0;
    }
    for (int b = i; b < AIE_LUT_ITEM; b++)
        out[b] = 0;
}

#endif
