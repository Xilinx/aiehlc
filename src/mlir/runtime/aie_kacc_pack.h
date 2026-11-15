/* aie_kacc_pack.h -- host-side RUNTIME operand packing / output un-tiling for the gemm_kacc kernel
 * (example/aiegraph/unet_bk_sdm_tiny/unet_aie.py: kacc_geom / kacc_kernel_text).
 *
 * GAP gap-aiehlc-bmm-lowering (2026-11-15, Maxwell, PBK): until now the kacc operand layout existed only as a
 * build-time numpy pre-layout (kacc_gemm.py layout()) -- fine for weights, impossible for attention bmm where
 * BOTH operands are activations. These routines do the same layout at run time on the APU, from natural f32
 * tensors, with zero padding of K / rows to the kernel geometry (aiegraph.gemmpad semantics), bf16 RNE.
 *
 *   akp_pack():   src (r, k) = src[r*rs + k*ks] (f32, r < rreal, k < kreal, else 0)  ->  dst [R][kp] bf16 in
 *                 the kacc chunk layout: 16-row block, chunk j (kch bf16) at j*kch of each row, in-chunk flat
 *                 f = hi*8*kch + kb*64 + (W side: lo*8 + kk | X side: kk*8 + lo).
 *                 W side = the mmul's left operand (out channels, i.e. B^T rows: rs = 1, ks = N for B [K][N]);
 *                 X side = pixels / rows of A (rs = K, ks = 1 for A [M][K]).
 *   akp_unpad():  one launch's fp32 c buffer (P4bx [s][mi][ni][8x8] tiles, transposed when tr) -> out[m][n]
 *                 (row stride ldo) as bf16 RNE (one SRS), dropping the padded rows / columns.
 * Header-only C/C++; tested on x86 by src/mlir/runtime/test/test_aie_kacc_pack.py against kacc_gemm.layout().
 */
#ifndef AIE_KACC_PACK_H
#define AIE_KACC_PACK_H
#include <stdint.h>
#include <string.h>

#if defined(__GNUC__) && !defined(__clang__)
#define AKP_HOT __attribute__((optimize("O3")))
#else
#define AKP_HOT
#endif

static inline uint16_t akp_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u)
        return (uint16_t)((u >> 16) | 0x40); /* NaN stays NaN */
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static inline int akp_log2(int x) {
    int l = 0;
    while ((1 << l) < x)
        l++;
    return l;
}

#if defined(__aarch64__) && !defined(AKP_NO_NEON)
#include <arm_neon.h>
#define AKP_NEON 1
/* 8 f32 -> 8 bf16 (RNE, NaN kept NaN) */
static inline uint16x8_t akp_cvt8(const float *p) {
    const uint32x4_t lo = vreinterpretq_u32_f32(vld1q_f32(p)), hi = vreinterpretq_u32_f32(vld1q_f32(p + 4));
    const uint32x4_t one = vdupq_n_u32(1), rnd = vdupq_n_u32(0x7fff), inf = vdupq_n_u32(0x7f800000);
    const uint32x4_t abs7 = vdupq_n_u32(0x7fffffff), q = vdupq_n_u32(0x400000);
    uint32x4_t rl = vaddq_u32(vaddq_u32(lo, rnd), vandq_u32(vshrq_n_u32(lo, 16), one));
    uint32x4_t rh = vaddq_u32(vaddq_u32(hi, rnd), vandq_u32(vshrq_n_u32(hi, 16), one));
    rl = vbslq_u32(vcgtq_u32(vandq_u32(lo, abs7), inf), vorrq_u32(lo, q), rl);
    rh = vbslq_u32(vcgtq_u32(vandq_u32(hi, abs7), inf), vorrq_u32(hi, q), rh);
    return vcombine_u16(vshrn_n_u32(rl, 16), vshrn_n_u32(rh, 16));
}
/* in place 8x8 u16 transpose: r[i][j] -> r[j][i] */
static inline void akp_tr8(uint16x8_t r[8]) {
    uint16x8_t t[8];
    for (int i = 0; i < 8; i += 2) {
        t[i] = vtrn1q_u16(r[i], r[i + 1]);
        t[i + 1] = vtrn2q_u16(r[i], r[i + 1]);
    }
    uint32x4_t a[8];
    for (int i = 0; i < 8; i += 4)
        for (int j = 0; j < 2; j++) {
            a[i + j] = vtrn1q_u32(vreinterpretq_u32_u16(t[i + j]), vreinterpretq_u32_u16(t[i + j + 2]));
            a[i + j + 2] = vtrn2q_u32(vreinterpretq_u32_u16(t[i + j]), vreinterpretq_u32_u16(t[i + j + 2]));
        }
    for (int j = 0; j < 4; j++) {
        r[j] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(a[j]), vreinterpretq_u64_u32(a[j + 4])));
        r[j + 4] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(a[j]), vreinterpretq_u64_u32(a[j + 4])));
    }
}
#endif

/* R % 16 == 0, kp % kch == 0, kch a power of two >= 8 */
static AKP_HOT void akp_pack(uint16_t *dst, const float *src, int R, int rreal, int kp, int kreal, int kch,
                             long rs, long ks, int wside) {
    int off[64];
    const int lk = akp_log2(kch);
    for (int lo = 0; lo < 8; lo++)
        for (int kk = 0; kk < 8; kk++) {
            const int e = wside ? lo * 8 + kk : kk * 8 + lo;
            off[lo * 8 + kk] = (e >> lk) * kp + (e & (kch - 1)); /* f0 % kch == 0 when kch < 64 */
        }
    for (int blk = 0; blk < R / 16; blk++)
        for (int hi = 0; hi < 2; hi++) {
            const int r0 = blk * 16 + hi * 8;
            for (int k0 = 0; k0 < kp; k0 += 8) {
                const int j = k0 >> lk, kb = (k0 & (kch - 1)) >> 3;
                const int f0 = hi * 8 * kch + kb * 64;
                uint16_t *d = dst + (long)(blk * 16 + (f0 >> lk)) * kp + j * kch + (f0 & (kch - 1));
                if (r0 + 8 <= rreal && k0 + 8 <= kreal) {
                    const float *s = src + r0 * rs + k0 * ks;
#ifdef AKP_NEON
                    /* X side (ks == 1): vector i = row lo=i over kk; W side (rs == 1): vector i = k row kk=i over
                     * lo. Either way out e-block j*8.. = transposed vector j, stored at the offset of e = j*8. */
                    if ((wside ? rs : ks) == 1) {
                        const long so = wside ? ks : rs;
                        uint16x8_t v[8];
                        for (int i = 0; i < 8; i++)
                            v[i] = akp_cvt8(s + i * so);
                        akp_tr8(v);
                        for (int j = 0; j < 8; j++)
                            vst1q_u16(d + ((j * 8) >> lk) * kp + ((j * 8) & (kch - 1)), v[j]);
                        continue;
                    }
#endif
                    for (int lo = 0; lo < 8; lo++)
                        for (int kk = 0; kk < 8; kk++)
                            d[off[lo * 8 + kk]] = akp_bf16(s[lo * rs + kk * ks]);
                } else {
                    for (int lo = 0; lo < 8; lo++)
                        for (int kk = 0; kk < 8; kk++) {
                            const int r = r0 + lo, k = k0 + kk;
                            d[off[lo * 8 + kk]] = (r < rreal && k < kreal) ? akp_bf16(src[r * rs + k * ks]) : 0;
                        }
                }
            }
        }
}

/* device index of local Y[m][n] in one launch's c buffer (Ml = px of the instance, Nl = ch) -- = kacc_yidx */
static inline long akp_yidx(int m, int n, int Ml, int Nl, int tr) {
    const int r = n & 15, c = m & 15;
    if (!tr) {
        const int q = Ml / 4, cc = m % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
        return (long)((n & ~15) + wi / q) * Ml + (m - cc) + wi % q;
    } else {
        const int q = Nl / 4, cc = n % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
        return (long)((m & ~15) + wi / q) * Nl + (n - cc) + wi % q;
    }
}

/* one 8x8 output block at (m0, n0) */
static inline void akp_unpad_blk(uint16_t *out, long ldo, const float *dev, int Ml, int Nl, int tr, int mreal,
                                 int nreal, int m0, int n0, int q) {
    if (q % 64 == 0 && m0 + 8 <= mreal && n0 + 8 <= nreal) {
        const float *s = dev + akp_yidx(m0, n0, Ml, Nl, tr); /* element (m0+a, n0+b) at s[b*8 + a] */
#ifdef AKP_NEON
        uint16x8_t v[8];
        for (int b = 0; b < 8; b++)
            v[b] = akp_cvt8(s + b * 8);
        akp_tr8(v);
        for (int a = 0; a < 8; a++)
            vst1q_u16(out + (long)(m0 + a) * ldo + n0, v[a]);
#else
        for (int a = 0; a < 8; a++) {
            uint16_t *o = out + (long)(m0 + a) * ldo + n0;
            for (int b = 0; b < 8; b++)
                o[b] = akp_bf16(s[b * 8 + a]);
        }
#endif
    } else {
        for (int a = 0; a < 8 && m0 + a < mreal; a++)
            for (int b = 0; b < 8 && n0 + b < nreal; b++)
                out[(long)(m0 + a) * ldo + n0 + b] = akp_bf16(dev[akp_yidx(m0 + a, n0 + b, Ml, Nl, tr)]);
    }
}

/* walk the blocks in DEVICE order (one {16 rows, q} tile of the [ch][px] / [px][ch] buffer after the other, wi
 * ascending inside it): the reads stream; the output band (q rows / cols x all) stays in L2 */
static AKP_HOT void akp_unpad(uint16_t *out, long ldo, const float *dev, int Ml, int Nl, int tr, int mreal,
                              int nreal) {
    const int q = tr ? Nl / 4 : Ml / 4;
    const int qlen = tr ? nreal : mreal, glen = tr ? mreal : nreal; /* q dim: m (n if tr); 16-groups: n (m) */
    for (int qb = 0; qb < qlen; qb += q)
        for (int g = 0; g < glen; g += 16)
            for (int b16 = 0; b16 < q && qb + b16 < qlen; b16 += 16)
                for (int rb = 0; rb < 16; rb += 8)      /* r = n & 15 */
                    for (int cb = 0; cb < 16; cb += 8) { /* c = m & 15 */
                        const int m0 = tr ? g + cb : qb + b16 + cb, n0 = tr ? qb + b16 + rb : g + rb;
                        if (m0 < mreal && n0 < nreal)
                            akp_unpad_blk(out, ldo, dev, Ml, Nl, tr, mreal, nreal, m0, n0, q);
                    }
}
#endif
