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
//   aie::mul / aie::mac (vector x vector into acc32 / acc64), aie::tile::current()
//   aie::load_v / load_unaligned_v / store_v / store_unaligned_v, aie::zeros, aie::concat,
//   aie::shuffle_down_fill, aie::interleave_zip, aie::bit_and, aie::add
//   aie::set_rounding / set_saturation (the SRS modes of to_vector)
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
    int64_t v[N];
    static constexpr int bits = std::is_same<Tag, acc32>::value ? 32 : 64;
    static int64_t wrap(int64_t x) { return bits == 32 ? (int64_t)(int32_t)(uint32_t)x : x; }
    template <class T> accum &from_vector(const vector<T, N> &x, int shift = 0) {
        for (unsigned i = 0; i < N; i++)
            v[i] = wrap((int64_t)x.v[i] << shift);
        return *this;
    }
    template <class T> vector<T, N> to_vector(int shift = 0) const {
        vector<T, N> r;
        for (unsigned i = 0; i < N; i++)
            r.v[i] = emu::narrow<T>(emu::srs(v[i], shift));
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
            r.v[i] = T(0);
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

// Element-wise multiply into an accumulator (the AIE API's default precision:
// 64-bit when either operand is 32-bit, else 32-bit), and multiply-accumulate.
namespace emu {
template <class A, class B> using mul_tag = std::conditional_t<(sizeof(A) > 2 || sizeof(B) > 2), acc64, acc32>;
}
template <class A, class B, unsigned N> accum<emu::mul_tag<A, B>, N> mul(const vector<A, N> &a, const vector<B, N> &b) {
    accum<emu::mul_tag<A, B>, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = accum<emu::mul_tag<A, B>, N>::wrap((int64_t)a.v[i] * (int64_t)b.v[i]);
    return r;
}
template <class Tag, class A, class B, unsigned N>
accum<Tag, N> mac(const accum<Tag, N> &acc, const vector<A, N> &a, const vector<B, N> &b) {
    accum<Tag, N> r;
    for (unsigned i = 0; i < N; i++)
        r.v[i] = accum<Tag, N>::wrap(acc.v[i] + (int64_t)a.v[i] * (int64_t)b.v[i]);
    return r;
}

// aie::tile::current().cycles(): no cycle counter on x86 -- 0.
struct tile {
    static tile current() { return tile(); }
    uint64_t cycles() const { return 0; }
};

// C (M x N) += A (M x K) * B (K x N), row-major, 32-bit accumulation for
// 8/16-bit operands (acc32 wraps like the hardware accumulator).
template <unsigned M, unsigned K, unsigned N, class TA, class TB, class Tag = acc32> struct mmul {
    using accum_type = accum<Tag, M * N>;
    accum_type c;
    mmul() { c = zeros<Tag, M * N>(); }
    explicit mmul(const accum_type &a) : c(a) {}
    void mac(const vector<TA, M * K> &a, const vector<TB, K * N> &b) {
        for (unsigned m = 0; m < M; m++)
            for (unsigned n = 0; n < N; n++) {
                int64_t s = c.v[m * N + n];
                for (unsigned k = 0; k < K; k++)
                    s += (int64_t)a.v[m * K + k] * (int64_t)b.v[k * N + n];
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

} // namespace aie

#endif // AIE_API_X86EMU_H
