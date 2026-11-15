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
 *                 (row stride ldo) as bf16 (one SRS: RNE on x86, saturating round-half-up on aarch64 NEON --
 *                 differs only on exact ties), dropping the padded rows / columns.
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
#if defined(__GNUC__)
#define AKP_INL static inline __attribute__((always_inline)) /* the host is built -Os: helpers would stay calls */
#else
#define AKP_INL static inline
#endif

AKP_INL uint16_t akp_bf16(float f) {
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
/* 8 f32 -> 8 bf16, RNE: (u + 0x7fff + lsb) >> 16 with a SATURATING add, so the all-ones NaN sentinel stays a bf16
 * NaN (0xffff) and +-inf stay inf (5 ops / 4 lanes; the earlier compare+select NaN path cost 9) */
AKP_INL uint16x8_t akp_cvt8(const float *p) {
    const uint32x4_t lo = vreinterpretq_u32_f32(vld1q_f32(p)), hi = vreinterpretq_u32_f32(vld1q_f32(p + 4));
    const uint32x4_t one = vdupq_n_u32(1), rnd = vdupq_n_u32(0x7fff);
    const uint32x4_t tl = vqaddq_u32(lo, vaddq_u32(vandq_u32(vshrq_n_u32(lo, 16), one), rnd));
    const uint32x4_t th = vqaddq_u32(hi, vaddq_u32(vandq_u32(vshrq_n_u32(hi, 16), one), rnd));
    return vcombine_u16(vshrn_n_u32(tl, 16), vshrn_n_u32(th, 16));
}
/* output SRS: 8 f32 -> 8 bf16 by saturating rounding narrow (2 ops / 8 lanes): round-half-up on the magnitude,
 * i.e. differs from RNE only on exact ties; NaN sentinel 0xffffffff saturates to bf16 NaN 0xffff, inf stays inf */
AKP_INL uint16x8_t akp_cvt8o(const float *p) {
    return vqrshrn_high_n_u32(vqrshrn_n_u32(vreinterpretq_u32_f32(vld1q_f32(p)), 16),
                              vreinterpretq_u32_f32(vld1q_f32(p + 4)), 16);
}
/* in place 8x8 u16 transpose: r[i][j] -> r[j][i] */
AKP_INL void akp_tr8(uint16x8_t r[8]) {
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
/* zeroed = 1: dst's pad was zeroed once (persistent buffer) -> tiles with no real data are not rewritten */
static AKP_HOT void akp_pack(uint16_t *dst, const float *src, int R, int rreal, int kp, int kreal, int kch,
                             long rs, long ks, int wside, int zeroed) {
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
            if (zeroed && r0 >= rreal)
                continue;
            for (int k0 = 0; k0 < (zeroed ? kreal : kp); k0 += 8) {
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
                    if ((wside ? ks : rs) == 1) { /* W side from row-major rows / X side from column-major: no tr */
                        const long so = wside ? rs : ks;
                        for (int i = 0; i < 8; i++)
                            vst1q_u16(d + ((i * 8) >> lk) * kp + ((i * 8) & (kch - 1)), akp_cvt8(s + i * so));
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
AKP_INL long akp_yidx(int m, int n, int Ml, int Nl, int tr) {
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
AKP_INL void akp_unpad_blk(uint16_t *out, long ldo, const float *dev, int Ml, int Nl, int tr, int mreal,
                                 int nreal, int m0, int n0, int q, int otr) {
    if (q % 64 == 0 && m0 + 8 <= mreal && n0 + 8 <= nreal) {
        const float *s = dev + akp_yidx(m0, n0, Ml, Nl, tr); /* element (m0+a, n0+b) at s[b*8 + a] */
        if (otr) { /* out[n][m]: 8 contiguous m per n row, no transpose */
            for (int b = 0; b < 8; b++) {
                uint16_t *o = out + (long)(n0 + b) * ldo + m0;
#ifdef AKP_NEON
                vst1q_u16(o, akp_cvt8o(s + b * 8));
#else
                for (int a = 0; a < 8; a++)
                    o[a] = akp_bf16(s[b * 8 + a]);
#endif
            }
            return;
        }
#ifdef AKP_NEON
        uint16x8_t v[8];
        for (int b = 0; b < 8; b++)
            v[b] = akp_cvt8o(s + b * 8);
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
                out[otr ? (long)(n0 + b) * ldo + m0 + a : (long)(m0 + a) * ldo + n0 + b] =
                    akp_bf16(dev[akp_yidx(m0 + a, n0 + b, Ml, Nl, tr)]);
    }
}

#if defined(__aarch64__)
static inline int akp_zva_ok(void) { /* DC ZVA allowed and 64-byte blocks */
    uint64_t v;
    __asm__ volatile("mrs %0, dczid_el0" : "=r"(v));
    return !(v & 16) && (v & 15) == 4;
}
#endif

/* unpad tuning: bit0 DC ZVA the output lines, bit1 prefetch the next device tile, bits 8..: n-chunk (0 = all) */
#ifndef AKP_UNPAD_CFG
#define AKP_UNPAD_CFG 1 /* PBK board sweep (c004 swap): ZVA 11.42 ms, none 11.97, prefetch 12.09, both 11.54 */
#endif
static int akp_unpad_cfg = AKP_UNPAD_CFG;

/* one {16, q} device tile (qb: start on the q dim = m (n if tr), g: 16-group on the other dim) */
static AKP_HOT void akp_unpad_tile(uint16_t *out, long ldo, const float *dev, int Ml, int Nl, int tr, int mreal,
                                   int nreal, int qb, int g, int otr) {
    const int q = tr ? Nl / 4 : Ml / 4;
    const int qlen = tr ? nreal : mreal, glen = tr ? mreal : nreal; /* q dim: m (n if tr); 16-groups: n (m) */
#if defined(__aarch64__)
    /* non-tr: an output line (32 bf16 of one row m) is fully written by the groups g, g+16 -> DC ZVA it first */
    /* otr: this tile writes out rows g..g+15, columns qb..qb+q-1 completely (q bf16 = q/32 whole lines each) */
    if ((akp_unpad_cfg & 1) && otr && q % 32 == 0 && qb + q <= qlen && g + 16 <= glen &&
        ((uintptr_t)(out + qb) & 63) == 0 && (ldo * 2) % 64 == 0 && akp_zva_ok())
        for (int r = g; r < g + 16; r++)
            for (int c = qb; c < qb + q; c += 32)
                __asm__ volatile("dc zva, %0" : : "r"(out + (long)r * ldo + c) : "memory");
    if ((akp_unpad_cfg & 1) && !tr && !otr && q % 64 == 0 && ((uintptr_t)out & 63) == 0 && (ldo * 2) % 64 == 0 &&
        nreal % 32 == 0 && g % 32 == 0 && akp_zva_ok())
        for (int m = qb; m < qb + q && m < mreal; m++)
            __asm__ volatile("dc zva, %0" : : "r"(out + (long)m * ldo + g) : "memory");
    if ((akp_unpad_cfg & 2) && q % 64 == 0) { /* next tile in walk order (g outer, qb inner): 16 rows x q f32 */
        const long pitch = tr ? Nl : Ml;
        const int nq = qb + q < qlen ? qb + q : 0, ng = qb + q < qlen ? g : g + 16;
        if (ng < glen) {
            const float *nx = dev + (long)ng * pitch + nq;
            for (int r = 0; r < 16; r++)
                for (int c = 0; c < q; c += 16)
                    __builtin_prefetch(nx + r * pitch + c);
        }
    }
#endif
#ifdef AKP_NEON
    if (q % 64 == 0 && qb + q <= qlen && g + 16 <= glen) {
        /* full tile: base dev + g*pitch + qb, 64-element blocks in wi order (no divides) */
        const long pitch = tr ? Nl : Ml;
        const int lq = akp_log2(q);
        const float *T = dev + (long)g * pitch + qb;
        for (int j = 0; j < q / 16 * 4; j++) {
            const int wi0 = j * 64, b16 = (j >> 2) * 16, rb = ((j >> 1) & 1) * 8, cb = (j & 1) * 8;
            const int m0 = tr ? g + cb : qb + b16 + cb, n0 = tr ? qb + b16 + rb : g + rb;
            const float *sp = T + (long)(wi0 >> lq) * pitch + (wi0 & (q - 1));
            if (otr) {
                for (int b = 0; b < 8; b++)
                    vst1q_u16(out + (long)(n0 + b) * ldo + m0, akp_cvt8o(sp + b * 8));
                continue;
            }
            uint16x8_t v[8];
            for (int b = 0; b < 8; b++)
                v[b] = akp_cvt8o(sp + b * 8);
            akp_tr8(v);
            for (int a = 0; a < 8; a++)
                vst1q_u16(out + (long)(m0 + a) * ldo + n0, v[a]);
        }
        return;
    }
#endif
    for (int b16 = 0; b16 < q && qb + b16 < qlen; b16 += 16)
        for (int rb = 0; rb < 16; rb += 8)      /* r = n & 15 */
            for (int cb = 0; cb < 16; cb += 8) { /* c = m & 15 */
                const int m0 = tr ? g + cb : qb + b16 + cb, n0 = tr ? qb + b16 + rb : g + rb;
                if (m0 < mreal && n0 < nreal)
                    akp_unpad_blk(out, ldo, dev, Ml, Nl, tr, mreal, nreal, m0, n0, q, otr);
            }
}

/* walk the tiles in DEVICE order: 16-row group g outer, q-band inner -> each group's rows (16 x pitch f32) are
 * read as one contiguous stream (PBK board: band-outer order read 512 B at 2 KB stride, 14.8 ms of loads for
 * c004 vs 4.7 ms sequential) */
/* otr = 0: out[m][n] (m = px index, n = ch index of the instance); otr = 1: out[n][m] (the kernel computed Y^T) */
static AKP_HOT void akp_unpad(uint16_t *out, long ldo, const float *dev, int Ml, int Nl, int tr, int mreal,
                              int nreal, int otr) {
    const int q = tr ? Nl / 4 : Ml / 4;
    const int qlen = tr ? nreal : mreal, glen = tr ? mreal : nreal;
    for (int g = 0; g < glen; g += 16) {
#ifdef AKP_NEON
        if (q % 64 == 0 && qlen % q == 0 && g + 16 <= glen) {
            /* whole 16-row group, device rows read linearly (row r of the group = blocks r*q/64 .. of every band) */
            const long pitch = tr ? Nl : Ml;
            const int bpr = q / 64; /* 64-element blocks per device row of a tile */
            if ((akp_unpad_cfg & 1) && otr && ((uintptr_t)out & 63) == 0 && (ldo * 2) % 64 == 0 && qlen % 32 == 0 &&
                akp_zva_ok())
                for (int r = g; r < g + 16; r++)
                    for (int c = 0; c < qlen; c += 32)
                        __asm__ volatile("dc zva, %0" : : "r"(out + (long)r * ldo + c) : "memory");
            for (int r = 0; r < 16; r++) {
                const float *row = dev + (long)(g + r) * pitch;
                for (int qb = 0; qb < qlen; qb += q)
                    for (int jj = 0; jj < bpr; jj++) {
                        const int j = r * bpr + jj, b16 = (j >> 2) * 16, rb = ((j >> 1) & 1) * 8, cb = (j & 1) * 8;
                        const int m0 = tr ? g + cb : qb + b16 + cb, n0 = tr ? qb + b16 + rb : g + rb;
                        const float *sp = row + qb + jj * 64;
                        if (otr) {
                            for (int b = 0; b < 8; b++)
                                vst1q_u16(out + (long)(n0 + b) * ldo + m0, akp_cvt8o(sp + b * 8));
                            continue;
                        }
                        uint16x8_t v[8];
                        for (int b = 0; b < 8; b++)
                            v[b] = akp_cvt8o(sp + b * 8);
                        akp_tr8(v);
                        for (int a = 0; a < 8; a++)
                            vst1q_u16(out + (long)(m0 + a) * ldo + n0, v[a]);
                    }
            }
            continue;
        }
#endif
        for (int qb = 0; qb < qlen; qb += q)
            akp_unpad_tile(out, ldo, dev, Ml, Nl, tr, mreal, nreal, qb, g, otr);
    }
}

/* incremental un-tiling (one tile per call, for callers that interleave it with other work) */
typedef struct {
    uint16_t *out;
    long ldo;
    const float *dev;
    int Ml, Nl, tr, mreal, nreal, qb, g, otr;
} akp_unpad_job;

static inline void akp_unpad_job_init(akp_unpad_job *j, uint16_t *out, long ldo, const float *dev, int Ml, int Nl,
                                      int tr, int mreal, int nreal, int otr) {
    j->out = out, j->ldo = ldo, j->dev = dev, j->Ml = Ml, j->Nl = Nl, j->tr = tr, j->mreal = mreal;
    j->nreal = nreal, j->qb = 0, j->g = 0, j->otr = otr;
}

/* one tile; returns 0 when the job is complete */
static inline int akp_unpad_step(akp_unpad_job *j) {
    const int q = j->tr ? j->Nl / 4 : j->Ml / 4;
    const int qlen = j->tr ? j->nreal : j->mreal, glen = j->tr ? j->mreal : j->nreal;
    if (j->g >= glen)
        return 0;
    akp_unpad_tile(j->out, j->ldo, j->dev, j->Ml, j->Nl, j->tr, j->mreal, j->nreal, j->qb, j->g, j->otr);
    if ((j->qb += q) >= qlen) { /* same order as akp_unpad: g outer, qb inner */
        j->qb = 0;
        j->g += 16;
    }
    return j->g < glen;
}
#endif
