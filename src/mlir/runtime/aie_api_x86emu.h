/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// aie_api_x86emu.h -- a scalar x86 emulation of the AIE API subset the block-
// load kernels use, so a VECTOR kernel body can be traced by the kernel
// analysis (convautotile/kernel_trace.cpp, compiled with g++ under ASan) and
// run by the x86 emulators (example/tileprogram/ccode/simpleconv2ddebug).
//
// Not the AIE API: no intrinsics, every element is a plain C++ access, so ASan
// sees exactly the bytes a load / store touches. Semantics follow the AIE API
// documentation for the covered calls:
//   aie::vector<T, N>      get / set / extract<M> / insert<M>
//   aie::accum<acc32|acc64, N> (global tags, as in the AIE API)  from_vector / to_vector (SRS) / extract / insert
//   aie::mmul<M, K, N, int8_t|int16_t, int8_t|int16_t>  A M x K, B K x N, C M x N,
//                          all row-major; mul / mac / to_accum / to_vector
//   aie::load_v / load_unaligned_v / store_v / store_unaligned_v, aie::zeros, aie::concat,
//   aie::shuffle_down_fill, aie::interleave_zip, aie::bit_and, aie::bit_xor, aie::add, aie::max / min,
//   aie::filter_even, aie::unpack (int8 -> int16 -> int32)
//   bfloat16, accum<accfloat, N>, aie::mmul<M, K, N, bfloat16, bfloat16, accfloat>
//   aie::mul / aie::mac (vector x scalar into acc32 / acc64)
//   aie::set_rounding / set_saturation (the SRS modes of to_vector)
//   Floating point (AIE2p bf16 path): aie::accum<accfloat, N> (fp32 lanes),
//   aie::mmul<M, K, N, bfloat16, bfloat16, accfloat> (fp32 accumulation of exact
//   bf16 products), aie::mul / mac on bfloat16 / float vectors into accfloat,
//   aie::max / sub / mul(vector, scalar), aie::reduce_max / reduce_add,
//   aie::exp2<T> (fp32 exp2, result converted to T), aie::inv (fp32 1/x) -- enough for an fp32-score
//   online-softmax (flash) attention kernel.
//   f32: sub, mul (-> accfloat), inv, invsqrt, to_fixed / to_float, vector_cast, up/downshift,
//   reduce_add / reduce_max, shuffle_down_rotate, mask / gt / lt / select (aie_modes_fvec.h)
//   bfloat16 (global scalar type: fp32 upper half, round-to-nearest-even from float)
// Anything else is a compile error in the harness -- extend this file.
//
// Alignment: load_v / store_v need the pointer aligned to the vector size
// (capped at 64 bytes); load_unaligned_v to `aligned_elems` elements. A
// violation is reported through AIEHLC_EMU_REPORT (abort by default; the kernel
// analysis turns it into a diagnostic). DMA windows are only 32-byte aligned on
// the core, and the analysis allocates blocks that way.
#ifndef AIE_API_X86EMU_H
#define AIE_API_X86EMU_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <source_location>
#include <type_traits>
#include <utility>

#ifndef AIEHLC_EMU_REPORT
#define AIEHLC_EMU_REPORT(msg) (std::fprintf(stderr, "aie_api_x86emu: %s\n", msg), std::abort())
#endif

#define chess_prepare_for_pipelining
#define chess_loop_range(...)
#define chess_unroll_loop(...)
#define chess_flatten_loop

// Accumulator tags are global in the AIE API (acc32, not aie::acc32).
struct acc32 {};
struct acc64 {};
struct accfloat {};

// bfloat16 (global, as chess's native type): the upper 16 bits of an IEEE fp32;
// float -> bfloat16 rounds to nearest even (NaN stays NaN), bfloat16 -> float
// is exact. Arithmetic goes through float, as a scalar bf16 expression does.
struct bfloat16 {
    uint16_t bits = 0;
    bfloat16() = default;
    bfloat16(float f) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu))
            bits = (uint16_t)((u >> 16) | 0x40); // quiet NaN
        else
            bits = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    operator float() const {
        const uint32_t u = (uint32_t)bits << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
};
static_assert(sizeof(bfloat16) == 2, "bfloat16 is a 16-bit type");

namespace aie {

enum class saturation_mode { none = 0, truncate = 0, saturate = 1, symmetric = 2 };
enum class rounding_mode {
    floor = 0,
    ceil,
    positive_inf,
    negative_inf,
    symmetric_inf,
    symmetric_zero,
    conv_even,
    conv_odd
};

namespace emu {
inline saturation_mode &sat() {
    static saturation_mode m = saturation_mode::none;
    return m;
}
inline rounding_mode &rnd() {
    static rounding_mode m = rounding_mode::floor;
    return m;
}
inline void checkAlign(const void *p, unsigned bytes, const char *what, const std::source_location &at) {
    if (bytes && (uintptr_t)p % bytes) {
        char msg[256];
        std::snprintf(msg, sizeof msg, "%s at %s:%u needs a %u-byte aligned address%s", what, at.file_name(),
                      (unsigned)at.line(), bytes,
                      bytes > 32 ? " (DMA windows are only 32-byte aligned: use aie::load_unaligned_v or "
                                   "two 32-byte accesses)"
                                 : "");
        AIEHLC_EMU_REPORT(msg);
    }
}
// Shift right by s with the current rounding mode (the SRS of to_vector).
inline int64_t srs(int64_t x, int s) {
    if (s <= 0)
        return s == 0 ? x : x << -s;
    const int64_t fl = x >> s, rem = x - (fl << s), half = (int64_t)1 << (s - 1);
    switch (rnd()) {
    case rounding_mode::floor:
        return fl;
    case rounding_mode::ceil:
        return fl + (rem != 0);
    case rounding_mode::positive_inf:
        return fl + (rem >= half);
    case rounding_mode::negative_inf:
        return fl + (rem > half);
    case rounding_mode::symmetric_inf:
        return x >= 0 ? fl + (rem >= half) : fl + (rem > half);
    case rounding_mode::symmetric_zero:
        return x >= 0 ? fl + (rem > half) : fl + (rem >= half);
    case rounding_mode::conv_even:
        return fl + (rem > half || (rem == half && (fl & 1)));
    case rounding_mode::conv_odd:
        return fl + (rem > half || (rem == half && !(fl & 1)));
    }
    return fl;
}
template <class T> T narrow(int64_t v) {
    constexpr int64_t lo = (int64_t)std::numeric_limits<T>::min(), hi = (int64_t)std::numeric_limits<T>::max();
    if (sat() == saturation_mode::saturate)
        return (T)(v < lo ? lo : v > hi ? hi : v);
    if (sat() == saturation_mode::symmetric)
        return (T)(v < -hi ? -hi : v > hi ? hi : v);
    return (T)v; // wrap
}
} // namespace emu

inline void set_saturation(saturation_mode m) { emu::sat() = m; }
inline void set_rounding(rounding_mode m) { emu::rnd() = m; }
inline saturation_mode get_saturation() { return emu::sat(); }
inline rounding_mode get_rounding() { return emu::rnd(); }

using ::bfloat16; // one type: the global chess-native bfloat16 above (aie::bfloat16 names it too)

template <class T, unsigned N> struct vector {
    T v[N];
    using value_type = T;
    static constexpr unsigned size() { return N; }
    T get(unsigned i) const { return v[i]; }
    vector &set(T x, unsigned i) {
        v[i] = x;
        return *this;
    }
    T operator[](unsigned i) const { return v[i]; }
    template <unsigned M> vector<T, M> extract(unsigned idx) const {
        vector<T, M> r;
        std::memcpy(r.v, v + idx * M, M * sizeof(T));
        return r;
    }
    template <unsigned M> vector &insert(unsigned idx, const vector<T, M> &s) {
        std::memcpy(v + idx * M, s.v, M * sizeof(T));
        return *this;
    }
};

template <class Tag, unsigned N> struct accum {
    static constexpr bool is_float = std::is_same<Tag, accfloat>::value;
    using storage = std::conditional_t<is_float, float, int64_t>;
    storage v[N];
    static constexpr int bits = std::is_same<Tag, acc32>::value ? 32 : 64;
    static storage wrap(storage x) {
        if constexpr (is_float)
            return x;
        else
            return bits == 32 ? (int64_t)(int32_t)(uint32_t)x : x;
    }
    // accfloat: fp32 lanes, shift ignored; to_vector<bfloat16> rounds to nearest even.
    template <class T> accum &from_vector(const vector<T, N> &x, int shift = 0) {
        for (unsigned i = 0; i < N; i++) {
            if constexpr (is_float)
                v[i] = (float)x.v[i];
            else
                v[i] = wrap((int64_t)x.v[i] << shift);
        }
        return *this;
    }
    template <class T> vector<T, N> to_vector(int shift = 0) const {
        vector<T, N> r;
        for (unsigned i = 0; i < N; i++) {
            if constexpr (is_float)
                r.v[i] = T(v[i]);
            else
                r.v[i] = emu::narrow<T>(emu::srs(v[i], shift));
        }
        return r;
    }
    template <unsigned M> accum<Tag, M> extract(unsigned idx) const {
        accum<Tag, M> r;
        for (unsigned i = 0; i < M; i++)
            r.v[i] = v[idx * M + i];
        return r;
    }
    template <unsigned M> accum &insert(unsigned idx, const accum<Tag, M> &s) {
        for (unsigned i = 0; i < M; i++)
            v[idx * M + i] = s.v[i];
        return *this;
    }
};

namespace emu {
template <class T> struct is_acc : std::false_type {};
template <> struct is_acc<acc32> : std::true_type {};
template <> struct is_acc<acc64> : std::true_type {};
template <> struct is_acc<accfloat> : std::true_type {};
template <class T> struct is_fp : std::bool_constant<std::is_floating_point<T>::value || std::is_same<T, bfloat16>::value> {};
} // namespace emu

template <class T, unsigned N> auto zeros() {
    if constexpr (emu::is_acc<T>::value) {
        accum<T, N> a;
        for (unsigned i = 0; i < N; i++)
            a.v[i] = 0;
        return a;
    } else {
        vector<T, N> r;
        for (unsigned i = 0; i < N; i++)
            r.v[i] = T(0.0f);
        return r;
    }
}

template <class T, unsigned N> vector<T, N> broadcast(T x) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = x;
    return r;
}

template <unsigned N, class T>
vector<std::remove_const_t<T>, N> load_v(T *p, const std::source_location at = std::source_location::current()) {
    emu::checkAlign(p, N * sizeof(T) < 64 ? N * sizeof(T) : 64, "aie::load_v", at);
    vector<std::remove_const_t<T>, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = p[i];
    return r;
}
template <unsigned N, class T>
vector<std::remove_const_t<T>, N> load_unaligned_v(T *p, unsigned aligned_elems = 1,
                                                   const std::source_location at = std::source_location::current()) {
    emu::checkAlign(p, aligned_elems * sizeof(T), "aie::load_unaligned_v", at);
    vector<std::remove_const_t<T>, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = p[i];
    return r;
}
template <class T, unsigned N>
T *store_v(T *p, const vector<T, N> &x, const std::source_location at = std::source_location::current()) {
    emu::checkAlign(p, N * sizeof(T) < 64 ? N * sizeof(T) : 64, "aie::store_v", at);
    for (unsigned i = 0; i < N; i++)
        p[i] = x.v[i];
    return p + N;
}

template <class T, unsigned N>
T *store_unaligned_v(T *p, const vector<T, N> &x, unsigned aligned_elems = 1,
                     const std::source_location at = std::source_location::current()) {
    emu::checkAlign(p, aligned_elems * sizeof(T), "aie::store_unaligned_v", at);
    for (unsigned i = 0; i < N; i++)
        p[i] = x.v[i];
    return p + N;
}

template <class T, unsigned N> vector<T, 2 * N> concat(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, 2 * N> r;
    std::memcpy(r.v, a.v, N * sizeof(T));
    std::memcpy(r.v + N, b.v, N * sizeof(T));
    return r;
}

// Elements n.. of v, then the first n of fill.
template <class T, unsigned N>
vector<T, N> shuffle_down_fill(const vector<T, N> &v, const vector<T, N> &fill, unsigned n) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = i + n < N ? v.v[i + n] : fill.v[i + n - N];
    return r;
}

// Chunks of `chunk` elements, alternating a, b: the first N go to .first.
template <class T, unsigned N>
std::pair<vector<T, N>, vector<T, N>> interleave_zip(const vector<T, N> &a, const vector<T, N> &b, unsigned chunk) {
    T tmp[2 * N];
    unsigned o = 0;
    for (unsigned i = 0; i < N; i += chunk)
        for (const vector<T, N> *s : {&a, &b})
            for (unsigned k = 0; k < chunk; k++)
                tmp[o++] = s->v[i + k];
    std::pair<vector<T, N>, vector<T, N>> r;
    std::memcpy(r.first.v, tmp, N * sizeof(T));
    std::memcpy(r.second.v, tmp + N, N * sizeof(T));
    return r;
}

template <class T, unsigned N> vector<T, N> bit_and(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = a.v[i] & b.v[i];
    return r;
}
template <class T, unsigned N> vector<T, N> add(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = (T)(a.v[i] + b.v[i]);
    return r;
}

// Floating-point element-wise helpers (computed in fp32, result converted to T).
// Integers subtract exactly (int32 past 2^24 must not round through fp32).
template <class T, unsigned N> vector<T, N> sub(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++) {
        if constexpr (std::is_integral<T>::value)
            r.v[i] = (T)(a.v[i] - b.v[i]);
        else
            r.v[i] = T((float)a.v[i] - (float)b.v[i]);
    }
    return r;
}
template <class T, unsigned N> vector<T, N> sub(const vector<T, N> &a, T b) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = T((float)a.v[i] - (float)b);
    return r;
}
template <class T, unsigned N> vector<T, N> max(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = (float)a.v[i] >= (float)b.v[i] ? a.v[i] : b.v[i];
    return r;
}
template <class T, unsigned N> T reduce_max(const vector<T, N> &a) {
    T m = a.v[0];
    for (unsigned i = 1; i < N; i++)
        if ((float)a.v[i] > (float)m)
            m = a.v[i];
    return m;
}
template <class T, unsigned N> T reduce_add(const vector<T, N> &a) {
    if constexpr (std::is_integral<T>::value) { // exact: no fp32 rounding past 2^24
        T s = 0;
        for (unsigned i = 0; i < N; i++)
            s += a.v[i];
        return s;
    } else {
        float s = 0.0f;
        for (unsigned i = 0; i < N; i++)
            s += (float)a.v[i];
        return T(s);
    }
}
// 2^x per lane in fp32 (the AIE2p exp2 unit takes bf16/fp32 in, gives fp32 out).
template <class TR = float, class T, unsigned N> vector<TR, N> exp2(const vector<T, N> &a) {
    vector<TR, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = TR(__builtin_exp2f((float)a.v[i]));
    return r;
}

// 1/x per lane in fp32 (aie::inv; the SiLU / sigmoid epilogues of the aiednn conv templates).
template <class T, unsigned N> vector<T, N> inv(const vector<T, N> &a) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = T(1.0f / (float)a.v[i]);
    return r;
}

// Element-wise multiply into an accumulator (the AIE API's default precision:
// 64-bit when either operand is 32-bit, else 32-bit), and multiply-accumulate.
namespace emu {
template <class A, class B>
using mul_tag = std::conditional_t<(is_fp<A>::value || is_fp<B>::value), accfloat,
                                   std::conditional_t<(sizeof(A) > 2 || sizeof(B) > 2), acc64, acc32>>;
template <class Tag, class A, class B> typename accum<Tag, 1>::storage prod(A a, B b) {
    if constexpr (std::is_same<Tag, accfloat>::value)
        return (float)a * (float)b;
    else
        return (int64_t)a * (int64_t)b;
}
} // namespace emu
template <class A, class B, unsigned N> accum<emu::mul_tag<A, B>, N> mul(const vector<A, N> &a, const vector<B, N> &b) {
    using Tag = emu::mul_tag<A, B>;
    accum<Tag, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = accum<Tag, N>::wrap(emu::prod<Tag>(a.v[i], b.v[i]));
    return r;
}
// vector x scalar (floating point only: the scalar broadcast form used by softmax rescaling).
template <class A, class B, unsigned N, class = std::enable_if_t<emu::is_fp<B>::value>>
accum<accfloat, N> mul(const vector<A, N> &a, B b) {
    accum<accfloat, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = (float)a.v[i] * (float)b;
    return r;
}
template <class Tag, class A, class B, unsigned N>
accum<Tag, N> mac(const accum<Tag, N> &acc, const vector<A, N> &a, const vector<B, N> &b) {
    accum<Tag, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = accum<Tag, N>::wrap(acc.v[i] + emu::prod<Tag>(a.v[i], b.v[i]));
    return r;
}

// aie::tile::current().cycles(): no cycle counter on x86 -- 0.
struct tile {
    static tile current() { return tile(); }
    uint64_t cycles() const { return 0; }
};

template <class T, unsigned N> vector<T, N> bit_xor(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = (T)(a.v[i] ^ b.v[i]);
    return r;
}
template <class T, unsigned N> vector<T, N> min(const vector<T, N> &a, const vector<T, N> &b) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = a.v[i] < b.v[i] ? a.v[i] : b.v[i];
    return r;
}
// Lane masks: gt / lt compare, select(a, b, m) = m ? b : a lane by lane (the AIE API's order).
template <unsigned N> struct mask {
    bool m[N];
    bool test(unsigned i) const { return m[i]; }
};
template <class T, unsigned N> mask<N> gt(const vector<T, N> &a, const vector<T, N> &b) {
    mask<N> r;
    for (unsigned i = 0; i < N; i++)
        r.m[i] = a.v[i] > b.v[i];
    return r;
}
template <class T, unsigned N> mask<N> lt(const vector<T, N> &a, const vector<T, N> &b) {
    mask<N> r;
    for (unsigned i = 0; i < N; i++)
        r.m[i] = a.v[i] < b.v[i];
    return r;
}
template <class T, unsigned N> vector<T, N> select(const vector<T, N> &a, const vector<T, N> &b, const mask<N> &m) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = m.m[i] ? b.v[i] : a.v[i];
    return r;
}
// Even elements (step 1): lanes 0, 2, 4, ... of a 2N-lane vector.
template <class T, unsigned N> vector<T, N / 2> filter_even(const vector<T, N> &v, unsigned step = 1) {
    vector<T, N / 2> r;
    for (unsigned i = 0, o = 0; i < N && o < N / 2; i += 2 * step)
        for (unsigned k = 0; k < step && o < N / 2; k++)
            r.v[o++] = v.v[i + k];
    return r;
}
// Widen to the next type: int8 -> int16 -> int32 (sign-extending), as aie::unpack.
namespace emu {
template <class T> struct wider;
template <> struct wider<int8_t> {
    using type = int16_t;
};
template <> struct wider<int16_t> {
    using type = int32_t;
};
} // namespace emu
template <class T, unsigned N> vector<typename emu::wider<T>::type, N> unpack(const vector<T, N> &v) {
    vector<typename emu::wider<T>::type, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = v.v[i];
    return r;
}

// Lane-wise vector x integer scalar into a 32-bit accumulator (as aie::mul / aie::mac with a scalar
// operand); a floating-point scalar takes the accfloat overload above.
template <class Tag = acc32, class T, unsigned N, class S, class = std::enable_if_t<std::is_integral<S>::value>>
accum<Tag, N> mul(const vector<T, N> &v, S x) {
    accum<Tag, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = accum<Tag, N>::wrap((int64_t)v.v[i] * (int64_t)x);
    return r;
}
template <class Tag, unsigned N, class T, class S, class = std::enable_if_t<std::is_integral<S>::value>>
accum<Tag, N> mac(const accum<Tag, N> &a, const vector<T, N> &v, S x) {
    accum<Tag, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = accum<Tag, N>::wrap(a.v[i] + (int64_t)v.v[i] * (int64_t)x);
    return r;
}

// C (M x N) += A (M x K) * B (K x N), row-major, 32-bit accumulation for
// 8/16-bit operands (acc32 wraps like the hardware accumulator); accfloat:
// bf16 x bf16 products are exact in fp32 and are summed in fp32.
template <unsigned M, unsigned K, unsigned N, class TA, class TB, class Tag = acc32> struct mmul {
    using accum_type = accum<Tag, M * N>;
    accum_type c;
    mmul() { c = zeros<Tag, M * N>(); }
    explicit mmul(const accum_type &a) : c(a) {}
    void mac(const vector<TA, M * K> &a, const vector<TB, K * N> &b) {
        for (unsigned m = 0; m < M; m++)
            for (unsigned n = 0; n < N; n++) {
                typename accum_type::storage s = c.v[m * N + n];
                for (unsigned k = 0; k < K; k++)
                    s += emu::prod<Tag>(a.v[m * K + k], b.v[k * N + n]);
                c.v[m * N + n] = accum_type::wrap(s);
            }
    }
    void mul(const vector<TA, M * K> &a, const vector<TB, K * N> &b) {
        c = zeros<Tag, M * N>();
        mac(a, b);
    }
    const accum_type &to_accum() const { return c; }
    template <class T> vector<T, M * N> to_vector(int shift = 0) const { return c.template to_vector<T>(shift); }
};

// ── f32 vector arithmetic (the FVEC mode, aie_modes_fvec.h) ──────────────────
// fp32 products exactly (the core's AIE2_FP32_EMULATION_ACCURACY_FAST products are close, not equal);
// inv / invsqrt follow the AIE-ML aie_api algorithms (detail/aie2/elementary.hpp) step for step, so
// their approximation error shows up here as it does on the core.
template <unsigned N> accum<accfloat, N> mul(const vector<float, N> &a, const vector<float, N> &b) {
    accum<accfloat, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = a.v[i] * b.v[i];
    return r;
}
// a bit reinterpretation between equally wide lanes (float <-> int32)
template <class To, class From, unsigned N> vector<To, N> vector_cast(const vector<From, N> &v) {
    static_assert(sizeof(To) == sizeof(From), "vector_cast between equally wide lanes only");
    vector<To, N> r;
    std::memcpy(r.v, v.v, sizeof r.v);
    return r;
}
template <class T, unsigned N> vector<T, N> upshift(const vector<T, N> &v, unsigned s) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = (T)((uint32_t)v.v[i] << s);
    return r;
}
template <class T, unsigned N> vector<T, N> downshift(const vector<T, N> &v, unsigned s) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = (T)(v.v[i] >> s); /* arithmetic */
    return r;
}
// v * 2^shift to an integer under the current rounding mode (floor / conv_even / ...), saturated
template <class TR, unsigned N> vector<TR, N> to_fixed(const vector<float, N> &v, int shift = 0) {
    vector<TR, N> r;
    for (unsigned i = 0; i < N; i++) {
        const double x = (double)v.v[i] * (double)((int64_t)1 << shift);
        double q = emu::rnd() == rounding_mode::floor  ? __builtin_floor(x)
                   : emu::rnd() == rounding_mode::ceil ? __builtin_ceil(x)
                                                       : __builtin_nearbyint(x);
        constexpr double lo = (double)std::numeric_limits<TR>::min(), hi = (double)std::numeric_limits<TR>::max();
        r.v[i] = (TR)(q < lo ? lo : q > hi ? hi : q);
    }
    return r;
}
template <class TR, class T, unsigned N> vector<TR, N> to_float(const vector<T, N> &v, int shift = 0) {
    vector<TR, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = (TR)((double)v.v[i] / (double)((int64_t)1 << shift));
    return r;
}
// lane i takes lane (i + n) mod N
template <class T, unsigned N> vector<T, N> shuffle_down_rotate(const vector<T, N> &v, unsigned n) {
    vector<T, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = v.v[(i + n) % N];
    return r;
}
namespace emu {
inline float bits_f(int32_t i) {
    float f;
    std::memcpy(&f, &i, 4);
    return f;
}
inline int32_t f_bits(float f) {
    int32_t i;
    std::memcpy(&i, &f, 4);
    return i;
}
} // namespace emu
template <unsigned N> vector<float, N> inv(const vector<float, N> &v) {
    vector<float, N> r;
    for (unsigned i = 0; i < N; i++) {
        const float x = v.v[i];
        float y = emu::bits_f(0x7eb53567 - emu::f_bits(x));
        y = -(((-x * y) + 1.436142f) * y * 1.9395974f); /* -y1 */
        r.v[i] = -(y * (y * x + 2.0f));                 /* y1 (2 - y1 x) */
    }
    return r;
}
template <unsigned N> vector<float, N> invsqrt(const vector<float, N> &v) {
    vector<float, N> r;
    for (unsigned i = 0; i < N; i++) {
        const float x = v.v[i], x2 = x * 0.703952253f;
        const float y = emu::bits_f(0x5F1FFFF9 - (emu::f_bits(x) >> 1));
        r.v[i] = y * (0.703952253f * 2.38924456f - x2 * (y * y));
    }
    return r;
}

} // namespace aie

#endif // AIE_API_X86EMU_H
