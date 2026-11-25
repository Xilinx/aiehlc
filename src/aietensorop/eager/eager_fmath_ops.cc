/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * eager_fmath_ops.cc -- the scalar f32 math of eager_fmath.h as one macro of local lambdas, EAGER_FMATH_OPS: a
 * kernel expands it at the top of its body (only a kernel's own text and the #defines of a spliced .cc reach the
 * core compile and the kernel-analysis trace, so the math travels as text), the workstation references expand
 * the same macro -- one copy of the math. A .cc so that aiehlc splices it into the kernel source. No include guard:
 * aiehlc collects only the #defines outside #if blocks, and an identical redefinition is legal.
 * Defines: eg_fbits, eg_ffrom, eg_fabs, eg_fmax, eg_fmin, eg_exp, eg_inv, eg_rsqrt, eg_log, eg_sigmoid, eg_tanh,
 * eg_softplus, eg_erfabs, eg_round, eg_sinpoly, eg_reduce_2pi, eg_sin_reduced, eager_fop(F, x, y, a, b).
 ******************************************************************************/
#include <stdint.h>

/* the algorithm codes = aie_modes_fvec.h's AIE_FVF_* (the runtime checks they agree); #defines, so they travel
 * with the spliced text into the kernel trace and the core compile */
#define EG_F_LINEAR 1
#define EG_F_RELU 2
#define EG_F_CLIP 3
#define EG_F_ABS 4
#define EG_F_SQUARE 5
#define EG_F_SQRT 6
#define EG_F_EXP 7
#define EG_F_LOGISTIC 8
#define EG_F_TANH 9
#define EG_F_GELU_TANH 10
#define EG_F_SWISH 11
#define EG_F_ELU 12
#define EG_F_LOG 13
#define EG_F_SOFTPLUS 14
#define EG_F_MISH 15
#define EG_F_HARDSWISH 16
#define EG_F_HARDSIGMOID 17
#define EG_F_GELU_ERF 18
#define EG_F_ERF 19
#define EG_F_SELU 20
#define EG_F_SOFTSIGN 21
#define EG_F_LOG_SIGMOID 22
#define EG_F_SIGN 23
#define EG_F_RECIPROCAL 24
#define EG_F_RSQRT 25
#define EG_F_ROUND 26
#define EG_F_FLOOR 27
#define EG_F_CEIL 28
#define EG_F_LOGICAL_NOT 29
#define EG_F_POW 30
#define EG_F_SIN 31
#define EG_F_COS 32
#define EG_F_TAN 33
#define EG_F_ADD 64
#define EG_F_SUB 65
#define EG_F_MUL 66
#define EG_F_DIV 67
#define EG_F_MAX 68
#define EG_F_MIN 69
#define EG_F_EQ 70
#define EG_F_NE 71
#define EG_F_GT 72
#define EG_F_GE 73
#define EG_F_LT 74
#define EG_F_LE 75
#define EG_F_LAND 76
#define EG_F_LOR 77
#define EG_F_LXOR 78
#define EG_F_ADD_SQUARE 79
/* per-row constants (faffine: a = the row's c0, b = c1; fbinary prelu: y the slope) */
#define EG_F_ROW_AFFINE 96
#define EG_F_ROW_AFFINE_RELU 97
#define EG_F_ROW_PRELU 98
/* frow's row operations (frow_eager_ref.h) */
#define EG_ROW_SOFTMAX 1
#define EG_ROW_LOGSOFTMAX 2
#define EG_ROW_SUM 3 /* reductions from here: one f32 per row */
#define EG_ROW_MEAN 4
#define EG_ROW_MAX 5
#define EG_ROW_MIN 6
#define EG_ROW_SUMDIV 7

// clang-format off
#define EAGER_FMATH_OPS \
    auto eg_fbits = [&](float f) -> uint32_t { \
        union { \
            float f; \
            uint32_t u; \
        } c; \
        c.f = f; \
        return c.u; \
    }; \
    auto eg_ffrom = [&](uint32_t u) -> float { \
        union { \
            float f; \
            uint32_t u; \
        } c; \
        c.u = u; \
        return c.f; \
    }; \
    auto eg_fabs = [&](float x) -> float { return eg_ffrom(eg_fbits(x) & 0x7fffffffu); }; \
    auto eg_fmax = [&](float x, float y) -> float { return x > y ? x : y; }; \
    auto eg_fmin = [&](float x, float y) -> float { return x < y ? x : y; }; \
    auto eg_exp = [&](float x) -> float { \
        x = eg_fmax(eg_fmin(x, 88.3f), -87.3f); \
        const float tn = x * 1.44269504089f + 12582912.0f, nf = tn - 12582912.0f; \
        const int32_t n = (int32_t)eg_fbits(tn) - 0x4B400000; \
        const float r = (x - nf * 0.693359375f) - nf * -2.12194440e-4f; \
        float p = 1.0f / 720; \
        p = p * r + 1.0f / 120, p = p * r + 1.0f / 24, p = p * r + 1.0f / 6; \
        p = p * r + 0.5f, p = p * r + 1.0f, p = p * r + 1.0f; \
        return p * eg_ffrom((uint32_t)(n + 127) << 23); \
    }; \
    auto eg_inv = [&](float x) -> float { /* 1 / x: bit-trick seed, three Newton steps */ \
        float y = eg_ffrom(0x7EF311C7u - eg_fbits(x)); \
        y = y * (2.0f - x * y), y = y * (2.0f - x * y), y = y * (2.0f - x * y); \
        return y; \
    }; \
    auto eg_rsqrt = [&](float x) -> float { \
        float y = eg_ffrom(0x5f3759dfu - (eg_fbits(x) >> 1)); \
        y = y * (1.5f - 0.5f * x * y * y), y = y * (1.5f - 0.5f * x * y * y), y = y * (1.5f - 0.5f * x * y * y); \
        return y; \
    }; \
    auto eg_log = [&](float x) -> float { /* x > 0: y0 = exponent ln 2, then y += x e^-y - 1 */ \
        float y = (float)((int32_t)(eg_fbits(x) >> 23) - 127) * 0.69314718056f; \
        for (int it = 0; it < 5; it++) \
            y = y + (x * eg_exp(-y) - 1.0f); \
        return y; \
    }; \
    auto eg_sigmoid = [&](float x) -> float { return eg_inv(eg_exp(-x) + 1.0f); }; \
    auto eg_tanh = [&](float x) -> float { return eg_sigmoid(2.0f * x) * 2.0f - 1.0f; }; \
    auto eg_softplus = [&](float x) -> float { return eg_fmax(x, 0.0f) + eg_log(eg_exp(-eg_fabs(x)) + 1.0f); }; \
    auto eg_erfabs = [&](float x) -> float { /* erf(|x|), A&S 7.1.26 */ \
        const float z = eg_fabs(x), e = eg_exp(-(z * z)), s = eg_inv(z * 0.3275911f + 1.0f); \
        float p = s * 1.061405429f - 1.453152027f; \
        p = p * s + 1.421413741f, p = p * s - 0.284496736f, p = p * s + 0.254829592f; \
        return 1.0f - p * s * e; \
    }; \
    auto eg_round = [&](float x) -> float { return (x + 12582912.0f) - 12582912.0f; }; /* half to even, |x| < 2^22 */ \
    auto eg_sinpoly = [&](float x) -> float { /* |x| <= pi / 2 */ \
        const float x2 = x * x; \
        float p = x2 * -2.5052108385441720e-8f + 2.7557319223985893e-6f; \
        p = p * x2 - 1.9841269841269841e-4f, p = p * x2 + 8.3333333333333333e-3f, p = p * x2 - 0.16666666666666667f; \
        return (p * x2 + 1.0f) * x; \
    }; \
    auto eg_reduce_2pi = [&](float x) -> float { \
        const float k = eg_round(x * 0.15915494309189535f); \
        return (x - k * 6.28125f) - k * 0.0019353071795864769f; \
    }; \
    auto eg_sin_reduced = [&](float r) -> float { /* r in [-pi, 3 pi / 2] */ \
        const float hp = 1.5707963267948966f, pi = 3.1415926535897932f; \
        if (r > hp) \
            r = pi - r; \
        else if (r < -hp) \
            r = -pi - r; \
        return eg_sinpoly(r); \
    }; \
    /* algorithm F of x (and y), alpha a, beta b */ \
    auto eager_fop = [&](int F, float x, float y, float a, float b) -> float { \
        switch (F) { \
        case EG_F_LINEAR: \
            return x * a + b; \
        case EG_F_RELU: \
            return eg_fmax(x, 0.0f) + eg_fmin(x, 0.0f) * a; \
        case EG_F_CLIP: \
            return eg_fmin(eg_fmax(x, a), b); \
        case EG_F_ABS: \
            return eg_fabs(x); \
        case EG_F_SQUARE: \
            return x * x; \
        case EG_F_SQRT: { \
            const float m = eg_fmax(x, 1e-38f); \
            return m * eg_rsqrt(m); \
        } \
        case EG_F_EXP: \
            return eg_exp(x); \
        case EG_F_LOG: \
            return eg_log(x); \
        case EG_F_LOGISTIC: \
            return eg_sigmoid(x); \
        case EG_F_TANH: \
            return eg_tanh(x); \
        case EG_F_GELU_TANH: \
            return (eg_tanh((x * x * 0.0356774081f + 0.7978845608f) * x) + 1.0f) * x * 0.5f; \
        case EG_F_SWISH: \
            return x * eg_sigmoid(a * x); \
        case EG_F_ELU: \
        case EG_F_SELU: { \
            const float v = eg_fmax(x, 0.0f) + (eg_exp(eg_fmin(x, 0.0f)) - 1.0f) * a; \
            return F == EG_F_SELU ? v * b : v; \
        } \
        case EG_F_SOFTPLUS: \
            return eg_softplus(x * a) * (1.0f / a); \
        case EG_F_MISH: \
            return x * eg_tanh(eg_softplus(x)); \
        case EG_F_HARDSIGMOID: \
        case EG_F_HARDSWISH: { \
            const float c = eg_fmin(eg_fmax(a * x + b, 0.0f), 1.0f); \
            return F == EG_F_HARDSWISH ? x * c : c; \
        } \
        case EG_F_GELU_ERF: /* 0.5 x + 0.5 |x| erf(|x| / sqrt 2) */ \
            return (x + eg_erfabs(x * 0.70710678118f) * eg_fabs(x)) * 0.5f; \
        case EG_F_ERF: \
        case EG_F_SIGN: { \
            const float s = eg_inv(eg_fmax(eg_fabs(x), 1e-30f)) * x; \
            return F == EG_F_ERF ? s * eg_erfabs(x) : s; \
        } \
        case EG_F_SOFTSIGN: \
            return x * eg_inv(eg_fabs(x) + 1.0f); \
        case EG_F_LOG_SIGMOID: \
            return -eg_softplus(-x); \
        case EG_F_RECIPROCAL: \
            return eg_inv(x); \
        case EG_F_RSQRT: \
            return eg_rsqrt(x); \
        case EG_F_ROUND: \
            return eg_round(x); \
        case EG_F_FLOOR: { \
            const float r = eg_round(x); \
            return r > x ? r - 1.0f : r; \
        } \
        case EG_F_CEIL: { \
            const float r = eg_round(x); \
            return x > r ? r + 1.0f : r; \
        } \
        case EG_F_LOGICAL_NOT: \
            return eg_fabs(x) > 0.0f ? 0.0f : 1.0f; \
        case EG_F_POW: { /* a x^b, b an integer in [-2, 4] (the host declines others) */ \
            const int e = (int)b; \
            const float v = e < 0 ? eg_inv(x) : x; \
            float r = e == 0 ? 1.0f : v; \
            for (int i = 1; i < (e < 0 ? -e : e); i++) \
                r = r * v; \
            return r * a; \
        } \
        case EG_F_SIN: \
            return eg_sin_reduced(eg_reduce_2pi(x)); \
        case EG_F_COS: \
            return eg_sin_reduced(1.5707963267948966f - eg_reduce_2pi(x)); \
        case EG_F_TAN: { /* sin(r) / cos(r), r = x - k pi, cos(r) = sin(pi / 2 - |r|) kept accurate near the poles */ \
            const float k = eg_round(x * 0.31830988618379067f), kc2 = k * 9.6765358979329324e-4f; \
            const float rh = x - k * 3.140625f, sg = rh > 0.0f ? 1.0f : -1.0f; \
            const float delta = (1.5703125f - eg_fabs(rh)) + 4.8382679489664662e-4f + sg * kc2; \
            return eg_sinpoly(rh - kc2) * eg_inv(eg_sinpoly(delta)); \
        } \
        case EG_F_ADD: \
            return x + y; \
        case EG_F_SUB: \
            return x - y; \
        case EG_F_MUL: \
            return x * y; \
        case EG_F_DIV: \
            return x * eg_inv(y); \
        case EG_F_MAX: \
            return eg_fmax(x, y); \
        case EG_F_MIN: \
            return eg_fmin(x, y); \
        case EG_F_EQ: \
            return eg_fabs(x - y) > 0.0f ? 0.0f : 1.0f; \
        case EG_F_NE: \
            return eg_fabs(x - y) > 0.0f ? 1.0f : 0.0f; \
        case EG_F_GT: \
            return x - y > 0.0f ? 1.0f : 0.0f; \
        case EG_F_LT: \
            return y - x > 0.0f ? 1.0f : 0.0f; \
        case EG_F_GE: \
            return y - x > 0.0f ? 0.0f : 1.0f; \
        case EG_F_LE: \
            return x - y > 0.0f ? 0.0f : 1.0f; \
        case EG_F_LAND: \
            return (eg_fabs(x) > 0.0f && eg_fabs(y) > 0.0f) ? 1.0f : 0.0f; \
        case EG_F_LOR: \
            return (eg_fabs(x) > 0.0f || eg_fabs(y) > 0.0f) ? 1.0f : 0.0f; \
        case EG_F_LXOR: \
            return ((eg_fabs(x) > 0.0f) != (eg_fabs(y) > 0.0f)) ? 1.0f : 0.0f; \
        case EG_F_ADD_SQUARE: \
            return x + y * y; \
        case EG_F_ROW_AFFINE: \
            return x * a + b; \
        case EG_F_ROW_AFFINE_RELU: \
            return eg_fmax(x * a + b, 0.0f); \
        case EG_F_ROW_PRELU: \
            return eg_fmax(x, 0.0f) + eg_fmin(x, 0.0f) * y; \
        } \
        return x; \
    }; \
    (void)0
// clang-format on
