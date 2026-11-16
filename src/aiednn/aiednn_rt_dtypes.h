/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_dtypes.h -- the narrow floating-point types: bf16, f16, f8 (e5m2, e4m3 "fn").
 *
 * Bit-level conversions to / from f32, round to nearest even, matching PyTorch's (c10):
 *   bf16: the top 16 bits of an f32 (NaN -> 0x7fc0);
 *   f16:  IEEE binary16, overflow -> inf;
 *   f8 e5m2: the top byte of an f16 (overflow -> inf);
 *   f8 e4m3 (fn: no inf, one NaN 0x7f, max 448): overflow -> NaN.
 *   e8m0: a scale 2^(b - 127), b = 0xff NaN; from f32 the nearest power of two (ties to even
 *         exponent), clamped to [2^-127, 2^127]; NaN / inf -> 0xff.
 *   f4 e2m1: 4 bits s.ee.m (0, 0.5, 1, 1.5, 2, 3, 4, 6), round to nearest even, saturate at 6,
 *         NaN -> 0; two per byte like s4 / u4.
 * Reference implementations read through rt_ld and write through rt_st; values are computed in double
 * and rounded once on store (via f32).
 ******************************************************************************/
#ifndef AIEDNN_RT_DTYPES_H
#define AIEDNN_RT_DTYPES_H

#include <stdint.h>
#include <string.h>

static inline uint32_t rt_f2u(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static inline float rt_u2f(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint16_t rt_f32_to_bf16(float f) {
    const uint32_t u = rt_f2u(f);
    if (f != f)
        return 0x7fc0;
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static inline float rt_bf16_to_f32(uint16_t h) { return rt_u2f((uint32_t)h << 16); }

static inline uint16_t rt_f32_to_f16(float f) {
    const float scale_to_inf = rt_u2f(0x77800000u), scale_to_zero = rt_u2f(0x08800000u); /* 2^112, 2^-110 */
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    const uint32_t w = rt_f2u(f), shl1_w = w + w, sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xff000000u;
    if (bias < 0x71000000u)
        bias = 0x71000000u;
    base = rt_u2f((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = rt_f2u(base);
    const uint32_t nonsign = ((bits >> 13) & 0x00007c00u) + (bits & 0x00000fffu);
    return (uint16_t)((sign >> 16) | (shl1_w > 0xff000000u ? 0x7e00u : nonsign));
}
static inline float rt_f16_to_f32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16, sign = w & 0x80000000u, two_w = w + w;
    const float normalized = rt_u2f((two_w >> 4) + (0xe0u << 23)) * rt_u2f(0x07800000u); /* 2^-112 */
    const float denormalized = rt_u2f((two_w >> 17) | (126u << 23)) - 0.5f;
    return rt_u2f(sign | (two_w < (1u << 27) ? rt_f2u(denormalized) : rt_f2u(normalized)));
}

static inline uint8_t rt_f32_to_e5m2(float f) {
    uint32_t u = rt_f2u(f);
    const uint32_t sign = u & 0x80000000u;
    u ^= sign;
    uint8_t r;
    if (u >= (143u << 23)) {
        r = u > (255u << 23) ? 0x7f : 0x7c;
    } else if (u < (113u << 23)) {
        const uint32_t dm = 134u << 23;
        r = (uint8_t)(rt_f2u(rt_u2f(u) + rt_u2f(dm)) - dm);
    } else {
        const uint32_t odd = (u >> 21) & 1u;
        u += ((uint32_t)(15 - 127) << 23) + 0xfffffu;
        u += odd;
        r = (uint8_t)(u >> 21);
    }
    return (uint8_t)(r | (sign >> 24));
}
static inline float rt_e5m2_to_f32(uint8_t b) { return rt_f16_to_f32((uint16_t)((uint16_t)b << 8)); }

static inline uint8_t rt_f32_to_e4m3(float f) {
    uint32_t u = rt_f2u(f);
    const uint32_t sign = u & 0x80000000u;
    u ^= sign;
    uint8_t r;
    if (u >= (1087u << 20)) {
        r = 0x7f;
    } else if (u < (121u << 23)) {
        const uint32_t dm = 141u << 23;
        r = (uint8_t)(rt_f2u(rt_u2f(u) + rt_u2f(dm)) - dm);
    } else {
        const uint32_t odd = (u >> 20) & 1u;
        u += ((uint32_t)(7 - 127) << 23) + 0x7ffffu;
        u += odd;
        r = (uint8_t)(u >> 20);
    }
    return (uint8_t)(r | (sign >> 24));
}
static inline float rt_e4m3_to_f32(uint8_t b) {
    const uint32_t w = (uint32_t)b << 24, sign = w & 0x80000000u, nonsign = w & 0x7fffffffu;
    uint32_t sh = nonsign ? (uint32_t)__builtin_clz(nonsign) : 32u;
    sh = sh > 4 ? sh - 4 : 0;
    const uint32_t inf_nan = (uint32_t)((int32_t)(nonsign + 0x01000000u) >> 8) & 0x7f800000u;
    const uint32_t zero = (uint32_t)((int32_t)(nonsign - 1u) >> 31);
    return rt_u2f(sign | ((((nonsign << sh >> 4) + ((0x78u - sh) << 23)) | inf_nan) & ~zero));
}

static inline float rt_e8m0_to_f32(uint8_t b) {
    return b == 0xff ? rt_u2f(0x7fc00000u) : b == 0 ? rt_u2f(0x00400000u) : rt_u2f((uint32_t)b << 23);
}
static inline uint8_t rt_f32_to_e8m0(float f) {
    const uint32_t u = rt_f2u(f);
    if ((u & 0x7f800000u) == 0x7f800000u)
        return 0xff;
    if ((u & 0x80000000u) || !(u & 0x7fffffffu)) /* no sign, no zero: the smallest scale */
        return 0;
    uint32_t e = (u >> 23) & 0xff, m = u & 0x7fffffu;
    if (e == 0) {                     /* subnormal: 2^-127 (e8m0 0x00) or 2^-126 (0x01) */
        return m > 0x600000u ? 1 : 0; /* midpoint 1.5 * 2^-127; a tie takes the even code 0x00 */
    }
    if (m > 0x400000u || (m == 0x400000u && (e & 1)))
        e++;
    return (uint8_t)(e > 0xfe ? 0xfe : e);
}

static const float rt_e2m1_mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
static inline float rt_e2m1_to_f32(unsigned code) {
    return (code & 8) ? -rt_e2m1_mag[code & 7] : rt_e2m1_mag[code & 7];
}
static inline unsigned rt_f32_to_e2m1(float f) {
    if (f != f)
        return 0;
    const unsigned sign = f < 0 || (f == 0 && (rt_f2u(f) >> 31)) ? 8u : 0u;
    const float a = f < 0 ? -f : f;
    unsigned c = 7;
    for (unsigned i = 0; i < 7; i++) { /* the nearest of the 8 magnitudes, ties to the even code */
        const float mid = 0.5f * (rt_e2m1_mag[i] + rt_e2m1_mag[i + 1]);
        if (a < mid || (a == mid && !(i & 1))) {
            c = i;
            break;
        }
    }
    return sign | c;
}

#endif /* AIEDNN_RT_DTYPES_H */
