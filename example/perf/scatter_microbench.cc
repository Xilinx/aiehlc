/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* Host scatter microbench (costmodel SCMB), bare-metal A78 (palmyra213), no AIE work.
 * Driven by script/perf/run_microbench.sh --section scatter; script/perf/scmb_fit.py reads the log (writes nothing
 * to script/perf/params/).
 * Function under test: uc_scatter16 (src/aiednn/conv2d/kacc_conv_host.cc:103, NEON), body copied verbatim below,
 * with uc_tr4 from src/aiednn/layout/neon_tr4_f32.cc; g_uc_var = 214 = the UNet build value
 * (costmodel_m2/unet_{off,aot-kco}/aiek_src/aiek_unetconv.cc: #define UC_VAR 214).
 * Call = the linear site form: uc_scatter16(Y, P=M, N, tr=0, 0, M, yout, M, yb=0, no epilogue).
 * Grid (24 points): Y total 2621440 B (M64 x N10240 fp32), N in {320..10240}, M = 2621440/(4N),
 *   align: yout at a 64-B aligned address (64) or +16 B (80); Y is always 64-B aligned,
 *   cache: cold = Y and yout cleaned+invalidated to PoC (DC CIVAC by range) before every timed call,
 *          warm = one untimed call right before every timed call.
 * 5 timed calls per point (CNTPCT), median printed as raw cntpct:
 *   "SCMBRAW N=<n> M=<m> align=64|80 cache=cold|warm bytes=2621440 med_cntpct=<t> cntfrq=<f> reps=5 bad=<nonfinite>
 *    chk=OK|BAD"
 */
#include "xaiengine.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>

#ifndef __AIESIM__
#include "xil_cache.h"
#include "xil_printf.h"
#endif

void routing(XAie_DevInst *) {}
__global__ void loop_kernel(int *in, int *out) { (void)in; (void)out; }

static inline uint64_t cntpct(void) { uint64_t v; __asm__ volatile("isb; mrs %0, cntpct_el0" : "=r"(v)); return v; }
static inline uint64_t cntfrq(void) { uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v)); return v; }

#define UC_VAR 214
static int g_uc_var = UC_VAR;
static int uc_zva64(void) {
    static int z = -1;
    if (z < 0) {
        uint64_t d;
        __asm__ volatile("mrs %0, dczid_el0" : "=r"(d));
        z = ((d & 0x10u) == 0 && (d & 0xfu) == 4u) ? 1 : 0;
    }
    return z;
}
/* GHL: in-place 4x4 fp32 transpose (rows r0..r3 -> columns) */
static inline void uc_tr4(float32x4_t &r0, float32x4_t &r1, float32x4_t &r2, float32x4_t &r3) {
    const float32x4x2_t t01 = vtrnq_f32(r0, r1), t23 = vtrnq_f32(r2, r3);
    /* 64-bit lane trn1/trn2 = combine(low, low) / combine(high, high), one instruction each */
    r0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0])));
    r1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1])));
    r2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0])));
    r3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1])));
}
/* ---- uc_scatter16: verbatim copy of src/aiednn/conv2d/kacc_conv_host.cc:97-265 ---- */
/* GHL: scatter of one launch's tiled fp32 block Y (P4bx: [N][P] normal, [P][N] transposed walk) into the output,
 * 16 n x 16 px per tile, m-tiles outer / n-tiles inner: a tile reads 1 KiB of Y in contiguous chunks at a fixed
 * stride and every output row gets whole 64-B pieces. For one n the 8 px of an 8-aligned group are 8 contiguous
 * fp32 (c & 7 adds to wi, q % 16 == 0); 4 n x 4 px per 4x4 transpose. Output offset: yb == 0 -> [M][N] row-major
 * (linear), else NCHWc with channel block yb (conv, yb % 4 == 0). Epilogue in argument order: et 0 bias [N],
 * et 1 same layout as the output. Values and add order = the scalar loops. Returns the nonfinite count. */
static int uc_scatter16(const float *Ys, int P, int N, int tr, int mbase, int mvalid, float *yout, int M, int yb,
                        const float *const *ep, const int *et, int ne_, int mt0 = 0, int mt1 = 1 << 30) {
    const float32x4_t lim = vdupq_n_f32(1e30f);
    uint32x4_t badv = vdupq_n_u32(0);
    const float *src[16][2];
    const int nouter = g_uc_var & 1, pf = (g_uc_var & 2) && !tr;
    const int zva = (g_uc_var & 8) && yb == 0 && ((uintptr_t)yout & 63) == 0 && uc_zva64();
    const int nmt = (mvalid + 15) / 16, nnt = N / 16;
    /* GHL: src offsets tabulated once per call (2 divisions each): normal walk -> per m-tile (+ nt*P at use),
     * transposed -> per n-tile (+ mt*N at use) */
    static int32_t g_uc_srct[32768];
    static int g_srct_key[4] = {-1, -1, -1, -1}; /* GHL: (tr, P, N, ntt) of the table: m-tile range calls reuse it */
    const int ntt = tr ? nnt : nmt, tab = ntt * 32 <= 32768;
    const int fresh = !(g_srct_key[0] == tr && g_srct_key[1] == P && g_srct_key[2] == N && g_srct_key[3] == ntt);
    if (tab && fresh) {
        g_srct_key[0] = tr;
        g_srct_key[1] = P;
        g_srct_key[2] = N;
        g_srct_key[3] = ntt;
    }
    if (tab && fresh)
        for (int ti = 0; ti < ntt; ti++)
            for (int r = 0; r < 16; r++)
                for (int h = 0; h < 2; h++) {
                    int32_t o;
                    if (!tr) {
                        const int ml = ti * 16 + 8 * h, c = ml & 15, q = P / 4, cc = ml % q;
                        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                        o = (wi / q) * P + (ml - cc) + wi % q;
                    } else {
                        const int n = ti * 16 + r, q = N / 4, cc = n % q;
                        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + h) * 64 + (r & 7) * 8;
                        o = (wi / q) * N + (n - cc) + wi % q;
                    }
                    g_uc_srct[ti * 32 + r * 2 + h] = o;
                }
    if (mt1 > nmt)
        mt1 = nmt;
    for (int it = nouter ? 0 : mt0 * nnt; it < (nouter ? nmt * nnt : mt1 * nnt); it++) {
        const int mt = 16 * (nouter ? it % nmt : it / nnt), nt = 16 * (nouter ? it / nmt : it % nnt);
        if (mt < 16 * mt0 || mt >= 16 * mt1) /* GHL: m-tile range [mt0, mt1) (wait-hook chunks) */
            continue;
        if (tab) {
            const float *base = Ys + (tr ? (size_t)mt * N : (size_t)nt * P);
            const int32_t *tt = g_uc_srct + (tr ? nt : mt) * 2; /* (tile/16)*32 */
            for (int r = 0; r < 16; r++) {
                src[r][0] = base + tt[2 * r];
                src[r][1] = base + tt[2 * r + 1];
            }
        } else {
            for (int r = 0; r < 16; r++)
                for (int h = 0; h < 2; h++) {
                    const int n = nt + r, ml = mt + 8 * h, c = ml & 15;
                    if (!tr) {
                        const int q = P / 4, cc = ml % q;
                        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                        src[r][h] = Ys + (size_t)((n & ~15) + wi / q) * P + (ml - cc) + wi % q;
                    } else {
                        const int q = N / 4, cc = n % q;
                        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                        src[r][h] = Ys + (size_t)((ml & ~15) + wi / q) * N + (n - cc) + wi % q;
                    }
                }
        }
        {
            if (pf && nt + 16 < N) /* normal walk: the next n-tile is the same tile 16 rows of P further */
                for (int r = 0; r < 16; r++)
                    for (int h = 0; h < 2; h++)
                        __builtin_prefetch(src[r][h] + (size_t)16 * P);
#if defined(__aarch64__)
            if (zva) /* 16 n x 4 B = one whole 64-B line per row (row-major output, N % 16 == 0, 64-B aligned) */
                for (int q = 0; q < 16 && mt + q < mvalid; q++)
                    __asm__ volatile("dc zva, %0" ::"r"(yout + (size_t)(mbase + mt + q) * N + nt) : "memory");
#endif
            /* output offset of (m, n0) = ob[n0] + m * ds (row-major: n0 + m*N; NCHWc: block(n0) + m*yb + n0%yb) */
            const size_t ds = yb ? (size_t)yb : (size_t)N;
            size_t ob[4];
            for (int i = 0; i < 4; i++) {
                const int n0 = nt + 4 * i;
                ob[i] = yb ? (size_t)(n0 / yb) * M * yb + n0 % yb : (size_t)n0;
            }
            for (int mq = 0; mq < 16 && mt + mq < mvalid; mq += 4) {
                const int h = mq >> 3, off = mq & 7, m0 = mbase + mt + mq;
                const int nv = mvalid - (mt + mq) < 4 ? mvalid - (mt + mq) : 4; /* valid rows of the 4 */
                for (int nq = 0; nq < 16; nq += 4) {
                    float32x4_t v0 = vld1q_f32(src[nq][h] + off), v1 = vld1q_f32(src[nq + 1][h] + off),
                                v2 = vld1q_f32(src[nq + 2][h] + off), v3 = vld1q_f32(src[nq + 3][h] + off);
                    if (yb == 1) { /* P4a keep-layout: planar NCHW output (GN input): px m0..+3 of one channel are
                                    * 4 contiguous floats -> no transpose; residual (et 2+rb) in NCHWc block rb */
                        const float32x4_t vv[4] = {v0, v1, v2, v3};
                        for (int k = 0; k < 4; k++) {
                            const int n = nt + nq + k;
                            float *o = yout + (size_t)n * M + m0;
                            float32x4_t a = vv[k];
                            if (nv == 4)
                                badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(a, lim)), 31));
                            for (int e = 0; e < ne_; e++) {
                                if (et[e] == 0) {
                                    a = vaddq_f32(a, vdupq_n_f32(ep[e][n]));
                                } else {
                                    const int rb = et[e] >= 2 ? et[e] - 2 : 1;
                                    const float *r = ep[e] + (size_t)(n / rb) * M * rb + (size_t)m0 * rb + n % rb;
                                    float rv[4] = {r[0], 0.0f, 0.0f, 0.0f};
                                    for (int t = 1; t < 4; t++)
                                        rv[t] = (mt + mq + t < mvalid) ? r[(size_t)t * rb] : 0.0f;
                                    a = vaddq_f32(a, vld1q_f32(rv));
                                }
                            }
                            if (nv == 4)
                                vst1q_f32(o, a);
                            else {
                                float tmp[4];
                                vst1q_f32(tmp, a);
                                for (int t = 0; t < nv; t++)
                                    o[t] = tmp[t];
                            }
                        }
                        continue;
                    }
                    uc_tr4(v0, v1, v2, v3); /* vt = n nt+nq..+3 of px m0+t */
                    const size_t o0 = ob[nq >> 2] + (size_t)m0 * ds;
                    if (nv == 4) {
                        badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(v0, lim)), 31));
                        badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(v1, lim)), 31));
                        badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(v2, lim)), 31));
                        badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(v3, lim)), 31));
                        for (int e = 0; e < ne_; e++) {
                            if (et[e]) {
                                const float *ev = ep[e] + o0;
                                v0 = vaddq_f32(v0, vld1q_f32(ev));
                                v1 = vaddq_f32(v1, vld1q_f32(ev + ds));
                                v2 = vaddq_f32(v2, vld1q_f32(ev + 2 * ds));
                                v3 = vaddq_f32(v3, vld1q_f32(ev + 3 * ds));
                            } else {
                                const float32x4_t bv = vld1q_f32(ep[e] + nt + nq);
                                v0 = vaddq_f32(v0, bv);
                                v1 = vaddq_f32(v1, bv);
                                v2 = vaddq_f32(v2, bv);
                                v3 = vaddq_f32(v3, bv);
                            }
                        }
                        float *o = yout + o0;
                        vst1q_f32(o, v0);
                        vst1q_f32(o + ds, v1);
                        vst1q_f32(o + 2 * ds, v2);
                        vst1q_f32(o + 3 * ds, v3);
                    } else {
                        const float32x4_t vv[4] = {v0, v1, v2, v3};
                        for (int t = 0; t < nv; t++) {
                            const size_t o = o0 + (size_t)t * ds;
                            badv = vaddq_u32(badv, vshrq_n_u32(vmvnq_u32(vcaleq_f32(vv[t], lim)), 31));
                            float32x4_t a = vv[t];
                            for (int e = 0; e < ne_; e++)
                                a = vaddq_f32(a, vld1q_f32(ep[e] + (et[e] ? o : (size_t)(nt + nq))));
                            vst1q_f32(yout + o, a);
                        }
                    }
                }
            }
        }
    }
    return (int)vaddvq_u32(badv);
}
/* ---- end of copy ---- */

#define YBYTES 2621440u
#define YWORDS (YBYTES / 4u)
static float g_Y[YWORDS] __attribute__((aligned(64)));
static float g_out[YWORDS + 64] __attribute__((aligned(64)));

static void civac(const void *p, size_t n) {
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)63, e = (uintptr_t)p + n;
    for (; a < e; a += 64) __asm__ volatile("dc civac, %0" ::"r"(a) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
}
static int cmpu64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
/* reference check: Y layout [N][P] tiled (P4bx normal walk) -> out [M][N]; spot check one element per 16x16 tile
 * corner against a scalar recompute of the same source offset formula */
static int check(const float *out, int M, int N) {
    const int P = M, q = P / 4;
    for (int n = 0; n < N; n += 7)
        for (int m = 0; m < M; m += 5) {
            const int r = n & 15, ml = m & ~7, c = ml & 15, cc = ml % q;
            const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
            const float v = g_Y[(size_t)((n & ~15) + wi / q) * P + (ml - cc) + wi % q + (m & 7)];
            if (out[(size_t)m * N + n] != v) return 0;
        }
    return 1;
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    {
        unsigned long midr;
        __asm__ volatile("mrs %0, midr_el1" : "=r"(midr));
        printf("BOARD machine=aarch64 midr=0x%08lx\n", midr);
    }
#ifndef __AIESIM__
    Xil_DCacheEnable();
    Xil_ICacheEnable();
#endif
    printf("SCMB start scatter_microbench v1 uc_var=%d zva64=%d\n", g_uc_var, uc_zva64());
    fflush(stdout);
    for (uint32_t i = 0; i < YWORDS; i++) g_Y[i] = (float)((i * 2654435761u) >> 12) * (1.0f / 1048576.0f);
    static const int Ns[6] = {320, 640, 1280, 2560, 5120, 10240};
    const uint64_t f = cntfrq();
    int fails = 0;
    for (int ni = 0; ni < 6; ni++) {
        const int N = Ns[ni], M = (int)(YBYTES / 4u / (unsigned)N);
        for (int al = 0; al < 2; al++) {
            float *yout = g_out + (al ? 4 : 0); /* 64-B aligned, or +16 B */
            for (int warm = 0; warm < 2; warm++) {
                uint64_t t[5];
                int bad = 0;
                for (int r = 0; r < 5; r++) {
                    if (warm)
                        (void)uc_scatter16(g_Y, M, N, 0, 0, M, yout, M, 0, nullptr, nullptr, 0);
                    else {
                        civac(g_Y, YBYTES);
                        civac(yout, YBYTES);
                    }
                    const uint64_t c0 = cntpct();
                    bad += uc_scatter16(g_Y, M, N, 0, 0, M, yout, M, 0, nullptr, nullptr, 0);
                    t[r] = cntpct() - c0;
                }
                qsort(t, 5, sizeof t[0], cmpu64);
                const int ok = bad == 0 && check(yout, M, N);
                fails += !ok;
                printf("SCMBRAW N=%d M=%d align=%d cache=%s bytes=%u med_cntpct=%llu cntfrq=%llu reps=5 bad=%d chk=%s\n",
                       N, M, al ? 80 : 64, warm ? "warm" : "cold", YBYTES, (unsigned long long)t[2],
                       (unsigned long long)f, bad, ok ? "OK" : "BAD");
                fflush(stdout);
            }
        }
    }
    printf("SCMB END rc=%d %s\n", fails, fails ? "FAIL" : "PASS");
    fflush(stdout);
    return fails ? 1 : 0;
}
