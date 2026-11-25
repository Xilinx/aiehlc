/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aie_modes_pool.h -- pooling and elementwise as modes of the GEMM ELF.
 *
 * Core side of the megakernel modes dispatched from gemm_uk.cc by the opcode
 * in B row 127 (0xB0 pool, 0xE1 eltwise; 0xD7 is depthwise). Same item format
 * as the depthwise overlay: IN is the core's 4 KB item gathered from its A rows,
 * OUT is its int32 C tile (AIE_MODE_OUT entries). D points at the descriptor
 * byte after the opcode and column bytes.
 *
 * The core computes only exact integers -- a window's max or sum, a op b in
 * int32 -- and zeroes the unused outputs; every divide, rounding, saturation
 * and requantization is the host's, so the host matches its reference bit
 * for bit whatever the rounding rule.
 *
 * Statement macros, not functions: helpers outside the kernel body are not
 * carried into the kernel by aiehlc. Plain C, so the same text is the x86
 * reference (aiednn_rt_pool.h). aiehlc does not carry .h includes into the
 * kernel either, so the kernel source must include this header between
 * "// AIEHLC_KERNEL_PROLOGUE_BEGIN" and "// AIEHLC_KERNEL_PROLOGUE_END".
 *
 * Pool descriptor (LE16 = little-endian 16-bit):
 *   D[0] op 0 = max, 1 = sum, 2 = min, 3 = sum |x|, 4 = sum x*x, 5 = max |x|
 *   D[1] 1 = int8, 0 = uint8
 *   D[2] stride  D[3] KH  D[4] KW  D[5..6] Wp  D[7..8] Rout  D[9..10] Wo
 *   input rows of pitch Wp, already padded; out[oh*Wo+ow] = op over the
 *   KH x KW window at oh*stride*Wp + ow*stride.
 *   Needs Rout*Wo <= AIE_MODE_OUT and ((Rout-1)*stride + KH) * Wp <= AIE_MODE_IN.
 * Eltwise descriptor:
 *   D[0] op 0 add, 1 sub, 2 mul, 3 max, 4 min, 5 relu (b unused)
 *   D[1] bit 0: a is int8 (else uint8), bit 1: b is int8   D[2..3] n (LE16) <= AIE_MODE_OUT
 *   a = IN[0..n), b = IN[AIE_MODE_B_OFF..+n); out[i] = a[i] op b[i].
 ******************************************************************************/
#ifndef AIE_MODES_POOL_H
#define AIE_MODES_POOL_H

#define AIE_MODE_OP_POOL 0xB0
#define AIE_MODE_OP_ELTWISE 0xE1
#define AIE_MODE_IN 4096
#define AIE_MODE_OUT 1024
#define AIE_MODE_B_OFF 2048

#define AIE_MODE_POOL_MAX 0
#define AIE_MODE_POOL_SUM 1
#define AIE_MODE_POOL_MIN 2
#define AIE_MODE_POOL_SUMABS 3
#define AIE_MODE_POOL_SUMSQ 4
#define AIE_MODE_POOL_MAXABS 5

#define AIE_MODE_ELT_ADD 0
#define AIE_MODE_ELT_SUB 1
#define AIE_MODE_ELT_MUL 2
#define AIE_MODE_ELT_MAX 3
#define AIE_MODE_ELT_MIN 4
#define AIE_MODE_ELT_RELU 5

#define AIE_MODE_LE16(D, i) ((int)(D)[i] | ((int)(D)[(i) + 1] << 8))

/* Host side: pool geometry int32 G[8] = {op, signed, stride, KH, KW, Wp, Rout,
 * Wo} -> the 11 descriptor bytes at D; eltwise G = {op, signed bits, n} -> 4 bytes. */
#define AIE_MODE_POOL_DESC(D, G)                                                                                       \
    do {                                                                                                               \
        for (int amd_i = 0; amd_i < 5; amd_i++)                                                                        \
            (D)[amd_i] = (uint8_t)(G)[amd_i];                                                                          \
        for (int amd_i = 0; amd_i < 3; amd_i++) {                                                                      \
            (D)[5 + 2 * amd_i] = (uint8_t)((G)[5 + amd_i] & 0xff);                                                     \
            (D)[6 + 2 * amd_i] = (uint8_t)(((G)[5 + amd_i] >> 8) & 0xff);                                              \
        }                                                                                                              \
    } while (0)
#define AIE_MODE_ELTWISE_DESC(D, G)                                                                                    \
    do {                                                                                                               \
        (D)[0] = (uint8_t)(G)[0];                                                                                      \
        (D)[1] = (uint8_t)(G)[1];                                                                                      \
        (D)[2] = (uint8_t)((G)[2] & 0xff);                                                                             \
        (D)[3] = (uint8_t)(((G)[2] >> 8) & 0xff);                                                                      \
    } while (0)
#define AIE_MODE_LD(P, i, sgn) ((sgn) ? (int32_t)((const int8_t *)(P))[i] : (int32_t)((const uint8_t *)(P))[i])

#define AIE_MODE_POOL(IN, OUT, D)                                                                                      \
    do {                                                                                                               \
        const uint8_t *amp_d = (const uint8_t *)(D);                                                                   \
        const int amp_op = amp_d[0], amp_sgn = amp_d[1], amp_s = amp_d[2], amp_kh = amp_d[3], amp_kw = amp_d[4];       \
        const int amp_wp = AIE_MODE_LE16(amp_d, 5), amp_rout = AIE_MODE_LE16(amp_d, 7);                                \
        const int amp_wo = AIE_MODE_LE16(amp_d, 9);                                                                    \
        const int amp_n = amp_rout * amp_wo;                                                                           \
        for (int amp_i = 0; amp_i < AIE_MODE_OUT; amp_i++) {                                                           \
            int32_t amp_v = 0;                                                                                         \
            if (amp_i < amp_n) {                                                                                       \
                const int amp_oh = amp_i / amp_wo, amp_ow = amp_i - amp_oh * amp_wo;                                   \
                const int amp_base = amp_oh * amp_s * amp_wp + amp_ow * amp_s;                                         \
                amp_v = amp_op == AIE_MODE_POOL_MAX ? -2147483647 - 1 : amp_op == AIE_MODE_POOL_MIN ? 2147483647 : 0;  \
                for (int amp_y = 0; amp_y < amp_kh; amp_y++)                                                           \
                    for (int amp_x = 0; amp_x < amp_kw; amp_x++) {                                                     \
                        const int32_t amp_e = AIE_MODE_LD(IN, amp_base + amp_y * amp_wp + amp_x, amp_sgn);             \
                        const int32_t amp_a = amp_e < 0 ? -amp_e : amp_e;                                              \
                        switch (amp_op) {                                                                              \
                        case AIE_MODE_POOL_MAX:                                                                        \
                            amp_v = amp_e > amp_v ? amp_e : amp_v;                                                     \
                            break;                                                                                     \
                        case AIE_MODE_POOL_MIN:                                                                        \
                            amp_v = amp_e < amp_v ? amp_e : amp_v;                                                     \
                            break;                                                                                     \
                        case AIE_MODE_POOL_SUMABS:                                                                     \
                            amp_v += amp_a;                                                                            \
                            break;                                                                                     \
                        case AIE_MODE_POOL_SUMSQ:                                                                      \
                            amp_v += amp_e * amp_e;                                                                    \
                            break;                                                                                     \
                        case AIE_MODE_POOL_MAXABS:                                                                     \
                            amp_v = amp_a > amp_v ? amp_a : amp_v;                                                     \
                            break;                                                                                     \
                        default:                                                                                       \
                            amp_v += amp_e;                                                                            \
                            break;                                                                                     \
                        }                                                                                              \
                    }                                                                                                  \
            }                                                                                                          \
            (OUT)[amp_i] = amp_v;                                                                                      \
        }                                                                                                              \
    } while (0)

#define AIE_MODE_ELTWISE(IN, OUT, D)                                                                                   \
    do {                                                                                                               \
        const uint8_t *ame_d = (const uint8_t *)(D);                                                                   \
        const int ame_op = ame_d[0], ame_sgn = ame_d[1], ame_n = AIE_MODE_LE16(ame_d, 2);                              \
        for (int ame_i = 0; ame_i < AIE_MODE_OUT; ame_i++) {                                                           \
            int32_t ame_v = 0;                                                                                         \
            if (ame_i < ame_n) {                                                                                       \
                const int32_t ame_a = AIE_MODE_LD(IN, ame_i, ame_sgn & 1);                                             \
                const int32_t ame_b = AIE_MODE_LD(IN, AIE_MODE_B_OFF + ame_i, (ame_sgn >> 1) & 1);                     \
                switch (ame_op) {                                                                                      \
                case AIE_MODE_ELT_ADD:                                                                                 \
                    ame_v = ame_a + ame_b;                                                                             \
                    break;                                                                                             \
                case AIE_MODE_ELT_SUB:                                                                                 \
                    ame_v = ame_a - ame_b;                                                                             \
                    break;                                                                                             \
                case AIE_MODE_ELT_MUL:                                                                                 \
                    ame_v = ame_a * ame_b;                                                                             \
                    break;                                                                                             \
                case AIE_MODE_ELT_MAX:                                                                                 \
                    ame_v = ame_a > ame_b ? ame_a : ame_b;                                                             \
                    break;                                                                                             \
                case AIE_MODE_ELT_MIN:                                                                                 \
                    ame_v = ame_a < ame_b ? ame_a : ame_b;                                                             \
                    break;                                                                                             \
                default:                                                                                               \
                    ame_v = ame_a > 0 ? ame_a : 0;                                                                     \
                    break;                                                                                             \
                }                                                                                                      \
            }                                                                                                          \
            (OUT)[ame_i] = ame_v;                                                                                      \
        }                                                                                                              \
    } while (0)

#endif
