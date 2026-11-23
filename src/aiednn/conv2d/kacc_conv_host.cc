#else
#define UC_NEON 0
#endif
static unsigned g_ul_wzc = 0; /* GHL: linear sites on the zero-copy W path */
#ifndef UC_OPT
#define UC_OPT 1
#endif
/* GHL host<->DMA sync options: bit0 sparse Y sentinel (one NaN word per 4 KiB instead of a full 0xff memset +
 * flush of Y), bit1 whole D-cache clean+invalidate to PoC before the launch instead of DC CIVAC per 64-B line
 * over W, X and Y (the zero-copy W can be 26 MB = 400k lines), bit2 the same after the launch instead of the
 * Y range invalidate. Set/way cost is bounded by the cache sizes; both make every line the launch touches
 * come from / go to DRAM, as the ranges did. */
static int g_uc_opt = UC_OPT;
#ifndef UC_VAR
#define UC_VAR 214
#endif
#ifndef UC_WHOOK /* GHL: linear S>1 sites scatter block s-1 from the runtime's wait_io poll during launch s.
 * OFF: palmyra213 run of 8dd51b6 (=1) moved aie_host 177 -> 159 ms but aie_wall 1022 -> 1038 and graph_run
 * 5609 -> 5610 (linear9 site 29.1 -> 29.0 ms): the CPU has no idle time inside a launch (bdcfg host glue
 * ~3.6 ms/launch, wait_io short), so the hook only relabels host time as launch time. */
#define UC_WHOOK 0
#endif
/* GHL loop variants (A78 microbench s7/hl/mb, default 214 = 2|4|16|64|128): bit0 scatter n-tiles outer, bit1 m-outer +
 * prefetch of the next n-tile, bit2 im2col in destination-row order, bit3 DC ZVA of fully written lines (no gain),
 * bit4 linear X tile by 8x8 transposes, bit5 im2col chunk-outer (slower), bit6 im2col rows R/R+8 from one 32-B load,
 * bit7 pad columns zeroed in the planar pass (not a separate sweep over every plane) */
static int g_uc_var = UC_VAR;
/* GHL: DC ZVA usable with a 64-B block (DCZID_EL0.BS == 4, DZP == 0): a full output line is allocated zeroed
 * instead of read from DRAM before the scatter overwrites all of it */
static int uc_zva64(void) {
#if defined(__aarch64__)
    static int z = -1;
    if (z < 0) {
        uint64_t d;
        __asm__ volatile("mrs %0, dczid_el0" : "=r"(d));
        z = ((d & 0x10u) == 0 && (d & 0xfu) == 4u) ? 1 : 0;
    }
    return z;
#else
    return 0;
#endif
}
static void uc_dcache_clean_all(void) {
#if defined(__aarch64__)
    uint64_t clidr;
    __asm__ volatile("dsb sy\n\tmrs %0, clidr_el1" : "=r"(clidr)::"memory");
    const unsigned loc = (unsigned)(clidr >> 24) & 7u;
    for (unsigned lvl = 0; lvl < loc; lvl++) {
        if (((clidr >> (3 * lvl)) & 7u) < 2u) /* no data / unified cache at this level */
            continue;
        uint64_t ccs;
        __asm__ volatile("msr csselr_el1, %1\n\tisb\n\tmrs %0, ccsidr_el1" : "=r"(ccs) : "r"((uint64_t)(lvl << 1)) : "memory");
        const unsigned ls = (unsigned)(ccs & 7u) + 4u, ways = (unsigned)((ccs >> 3) & 0x3ffu) + 1u,
                       sets = (unsigned)((ccs >> 13) & 0x7fffu) + 1u;
        const unsigned wsh = ways > 1 ? (unsigned)__builtin_clz(ways - 1u) : 0u;
        for (unsigned w = 0; w < ways; w++)
            for (unsigned s = 0; s < sets; s++)
                __asm__ volatile("dc cisw, %0" ::"r"(((uint64_t)w << wsh) | ((uint64_t)s << ls) | (uint64_t)(lvl << 1))
                                 : "memory");
        __asm__ volatile("dsb sy" ::: "memory");
    }
    __asm__ volatile("msr csselr_el1, xzr\n\tdsb sy\n\tisb" ::: "memory");
#endif
}
/* sentinel: unwritten -> NaN -> counted by the scatter. bit0: one word per 4 KiB (flushed by line unless the
 * whole-cache clean follows), else the full memset (flushed with Y below) */
static void uc_ysentinel(int32_t *ky, size_t bytes) {
    if (g_uc_opt & 1) {
        for (size_t o = 0; o < bytes / 4; o += 1024) {
            ky[o] = -1;
#if defined(__aarch64__)
            if (!(g_uc_opt & 2)) /* clean+invalidate to PoC, one DSB for all (Xil_DCacheFlushLine: 2 DSB + IRQ mask each) */
                __asm__ volatile("dc civac, %0" ::"r"(ky + o) : "memory");
#else
            if (!(g_uc_opt & 2))
                Xil_DCacheFlushLine((INTPTR)(ky + o));
#endif
        }
#if defined(__aarch64__)
        __asm__ volatile("dsb sy" ::: "memory");
#endif
    } else {
        memset(ky, 0xff, bytes);
    }
}
static void uc_sync_pre(const void *w, size_t wb, const void *x, size_t xb, const void *y, size_t yb) {
    if (g_uc_opt & 2) {
        uc_dcache_clean_all();
        return;
    }
    Xil_DCacheFlushRange((UINTPTR)w, wb);
    Xil_DCacheFlushRange((UINTPTR)x, xb);
    if (!(g_uc_opt & 1))
        Xil_DCacheFlushRange((UINTPTR)y, yb);
}
#if UC_NEON
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
#endif
static void uc_sync_post(const void *y, size_t yb) {
    if (g_uc_opt & 4)
        uc_dcache_clean_all();
    else
        Xil_DCacheInvalidateRange((UINTPTR)y, yb);
}
/* GPOOL: pooled host-glue buffers, grown once and reused by every layer (no per-call alloc) */
static uint16_t *g_uc_pb = nullptr; /* padded bf16 planes [C][H+2ph][W+2pw] + slack */
static int *g_uc_sk = nullptr, *g_uc_dk = nullptr; /* per-k source / destination offsets */
static size_t g_uc_pcap = 0, g_uc_kcap = 0;
static unsigned g_uc_fast = 0, g_uc_wzc = 0; /* layers on the fast im2col / zero-copy W path */
static int uc_pool_need(size_t pel, size_t kel) {
    if (pel > g_uc_pcap) {
        free(g_uc_pb);
        g_uc_pb = (uint16_t *)__Runtime_Alloc(pel * 2);
        g_uc_pcap = g_uc_pb ? pel : 0;
    }
    if (kel > g_uc_kcap) {
        free(g_uc_sk);
        free(g_uc_dk);
        g_uc_sk = (int *)__Runtime_Alloc(kel * sizeof(int));
        g_uc_dk = (int *)__Runtime_Alloc(kel * sizeof(int));
        g_uc_kcap = (g_uc_sk && g_uc_dk) ? kel : 0;
    }
    return (g_uc_pb && g_uc_sk && g_uc_dk) ? 0 : -1;
}
static int uc_kalloc(void) {
    if (g_ky)
        return 0;
    g_kw = (int32_t *)__Runtime_Alloc((size_t)UC_KW_ELEMS * 2);
    g_kx = (int32_t *)__Runtime_Alloc((size_t)UC_KX_ELEMS * 2);
    g_ky = (int32_t *)__Runtime_Alloc((size_t)UC_KY_ELEMS * 4);
    return (g_kw && g_kx && g_ky) ? 0 : -1;
}
/* one launch of the layer's gemm_kacc instance (ports W [N][K/2], X [M][K/2], Y f32 [N][M]) */
static int uc_kacc_launch(int N, int K, int M, int32_t *kw, int32_t *kx, int32_t *ky) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({@PART_C0@, 6, 0, 6}, 4, 4); // aiegraph lint L015: the mesh VarDecl in the launching function
@DISPATCH@
    return -6;
}

static int uc_conv(void *const *argv, int nargs, const uint16_t *wbf, int sh, int sw, int ph, int pw) {
    XTime ut0; /* [UCL] per-layer split: host pre-tiling (W re-tile + im2col) | AIE window | host post */
    XTime_GetTime(&ut0);
    ahs_begin(&g_ahs, AHS_KIND_CONV);
    if (nargs < 4)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 5 || w->ndim != 6 || y->ndim != 5 || x->shape[0] != 1 || y->shape[0] != 1)
        return -2;
    const int cb = (int)x->shape[4], C = (int)x->shape[1] * cb, H = (int)x->shape[2], W = (int)x->shape[3];
    const int ib = (int)w->shape[4], ob = (int)w->shape[5], CI = (int)w->shape[1] * ib;
    const int O = (int)w->shape[0] * ob, KH = (int)w->shape[2], KW = (int)w->shape[3];
    const int yb = (int)y->shape[4], OH = (int)y->shape[2], OW = (int)y->shape[3];
    if (CI != C || (int)y->shape[1] * yb != O || OH != (H + 2 * ph - KH) / sh + 1 || OW != (W + 2 * pw - KW) / sw + 1)
        return -3;
    const int M = OH * OW, N = O, K = C * KH * KW, KK = KH * KW, kc = K / 64;
    if ((M | N | K) & 63 || (long)N * K > UC_KW_ELEMS || (long)M * K > UC_KX_ELEMS || (long)M * N > UC_KY_ELEMS)
        return -3;
    const float *xin = (const float *)x->data;
    float *yout = (float *)y->data;
    if (uc_kalloc() != 0)
        return -1;
#if UC_CTRLPKT
    if (!g_uc_ht) { // once, before the first launch
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    /* natural layout (unet_aie.kacc_geom): W [N][K], X [M][K], each 16-row block pre-tiled per K chunk of
     * kch bf16 as bf16_gemm_kacc.cc: W chunk = [mi 2][kb kch/8][8x8 ch x k], X chunk = [ni 2][kb][8x8 k x px] */
    int tkw = 4096 / M < 32 ? 4096 / M : 32; /* = kacc_geom tkw */
    const int tr = 2 * (16 * tkw * 4 + (M / 4) * tkw * 4 + 16 * (M / 4) * 4) > 29696; /* = kacc_geom tr: Y^T walk */
    if (tr)
        for (tkw = 32; (N / 4) * tkw > 1024 || (K / 2) % tkw; tkw /= 2)
            ;
    const int kch = 2 * tkw;
    const int Kp = (N == K / 2) ? K + kch : K; /* = kacc_geom kp: zero chunk when the W port would be square */
    if (K % kch || (long)N * Kp > UC_KW_ELEMS || (long)M * Kp > UC_KX_ELEMS)
        return -3;
    uint16_t *Wd = (uint16_t *)g_kw, *Xd = (uint16_t *)g_kx;
    /* P4w: wbf already holds W in the kacc chunk layout for K (unet_aie.prelayout_weights, build time: each
     * 16-row block = 16 rows of K, chunk j at j*kch, f = mi*8*kch + kb*64 + mm*8 + kk within a chunk). With
     * Kp > K each row only gains the zero chunk at its end. */
    int32_t *const kw_pool = g_kw; /* GPOOL: the pooled W port, restored after the launch */
    int wz = 0;
    if (Kp == K && ((uintptr_t)wbf & 63) == 0) {
        /* GPOOL: wbf is already the exact port image (P4w prelayout) -> the launch reads it in place (DMA
         * takes the raw address, as for the aligned_alloc pool; flushed below like the pool) */
        g_kw = (int32_t *)(uintptr_t)wbf;
        g_uc_wzc++;
        wz = 1;
    } else if (Kp == K) {
        memcpy(Wd, wbf, (size_t)N * K * 2);
    } else {
        for (int n = 0; n < N; n++) {
            memcpy(Wd + (size_t)n * Kp, wbf + (size_t)n * K, (size_t)K * 2);
            memset(Wd + (size_t)n * Kp + K, 0, (size_t)(Kp - K) * 2);
        }
    }
    /* GPOOL: im2col via pooled padded bf16 planes P[C][H+2ph][W+2pw] (one RNE convert per input element)
     * and per-k offset tables; 8 output pixels of one row at a fixed k are 8 contiguous bf16 in the X chunk
     * layout ([ni 2][kb][8x8 k x px]) -> one 16-byte copy (stride 2: vld2 even lanes). */
    if (kch > 64 || KK > 64)
        return -3;
    const int fast = (OW % 8) == 0 && sh == sw && (sh == 1 || sh == 2) && (kch % 8) == 0 && (cb % 4) == 0 &&
                     (W % 4) == 0;
    if (fast) {
        const int Hp = H + 2 * ph, Wp = W + 2 * pw;
        const size_t HWp = (size_t)Hp * Wp, pn = (size_t)C * HWp + 64;
        if (uc_pool_need(pn, (size_t)Kp) != 0)
            return -1;
        uint16_t *P = g_uc_pb;
        /* zero only the halo (pad rows/cols) and the slack the stride-2 vld2 may read past the last plane */
        for (int c = 0; c < C; c++) {
            uint16_t *pl = P + (size_t)c * HWp;
            if (ph) {
                memset(pl, 0, (size_t)ph * Wp * 2);
                memset(pl + (size_t)(Hp - ph) * Wp, 0, (size_t)ph * Wp * 2);
            }
            if (pw && !(g_uc_var & 128))
                for (int h = ph; h < Hp - ph; h++) {
                    uint16_t *rw = pl + (size_t)h * Wp;
                    for (int i = 0; i < pw; i++)
                        rw[i] = rw[Wp - 1 - i] = 0;
                }
        }
        memset(P + (size_t)C * HWp, 0, 64 * 2);
        /* NCHWc fp32 -> planar bf16: 4 pixels x 4 channels per 4x4 transpose */
        const size_t HWc = (size_t)H * W * cb;
        for (int cbk = 0; cbk < C / cb; cbk++)
            for (int h = 0; h < H; h++) {
                const float *src = xin + (size_t)cbk * HWc + (size_t)h * W * cb;
                uint16_t *dst0 = P + (size_t)cbk * cb * HWp + (size_t)(h + ph) * Wp + pw;
                for (int w0 = 0; w0 < W; w0 += 4)
                    for (int c0 = 0; c0 < cb; c0 += 4) {
                        const float *s0 = src + (size_t)w0 * cb + c0;
                        uint16_t *d = dst0 + (size_t)c0 * HWp + w0;
#if UC_NEON
                        const float32x4_t r0 = vld1q_f32(s0), r1 = vld1q_f32(s0 + cb), r2 = vld1q_f32(s0 + 2 * cb),
                                          r3 = vld1q_f32(s0 + 3 * cb);
                        const float32x4x2_t t01 = vtrnq_f32(r0, r1), t23 = vtrnq_f32(r2, r3);
                        vst1_u16(d, uc_f2bf4(vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]))));
                        vst1_u16(d + HWp, uc_f2bf4(vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]))));
                        vst1_u16(d + 2 * HWp,
                                 uc_f2bf4(vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]))));
                        vst1_u16(d + 3 * HWp,
                                 uc_f2bf4(vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]))));
#else
                        for (int i = 0; i < 4; i++)
                            for (int j = 0; j < 4; j++)
                                d[(size_t)i * HWp + j] = uc_f2bf(s0[(size_t)j * cb + i]);
#endif
                    }
                if (pw && (g_uc_var & 128)) /* GHL: pad columns of this row while its lines are in L1 */
                    for (int c = 0; c < cb; c++) {
                        uint16_t *rw = dst0 + (size_t)c * HWp - pw;
                        for (int i = 0; i < pw; i++)
                            rw[i] = rw[Wp - 1 - i] = 0;
                    }
            }
        /* per-k tables: source offset in P (c, ky, kx) and destination offset in the 16-row block (ni = 0) */
        int *sk = g_uc_sk, *dk = g_uc_dk;
        {
            int k = 0;
            for (int c = 0; c < C; c++)
                for (int ky = 0; ky < KH; ky++)
                    for (int kx = 0; kx < KW; kx++, k++)
                        sk[k] = (int)((size_t)c * HWp + (size_t)ky * Wp + kx);
            for (k = 0; k < Kp; k++) {
                const int j = k / kch, kr = k - j * kch, f = (kr >> 3) * 64 + (kr & 7) * 8;
                dk[k] = j * kch + (f / kch) * Kp + f % kch;
            }
        }
        /* GHL: channel groups outer -- the CG padded planes (<= 32 KiB) stay in L1 while every 8-px block
         * takes their k range (k-inner over all C streamed the whole P from DRAM once per block) */
        const int CG = (int)(32768 / (HWp * 2)) > 0 ? (int)(32768 / (HWp * 2)) : 1;
#if UC_NEON
        if (g_uc_var & 4) {
            /* GHL: destination-row order -- row R of a 16-px block, chunk j, unit u (8 bf16 = one k, 8 px):
             * f = R*kch + 8u = ni*8*kch + (kb*64 + kk*8), k = j*kch + kb*8 + kk, px = mb0 + ni*8 (= the dk table
             * inverted); every row of the X port is written front to back */
            int kofs[8];
            const int zx = (g_uc_var & 8) && ((uintptr_t)Xd & 63) == 0 && uc_zva64();
            if (g_uc_var & 32) {
                /* GHL: per 16-px block, chunk j0 outer / row R inner: one chunk's kch k (<= 8 channels) stay in L1
                 * while its 16 row pieces (kch bf16 each, whole lines over 1-2 chunks) are written */
                int ko[16][8];
                const uint16_t *rw[16];
                for (int R = 0; R < 16; R++)
                    for (int u = 0; u < kch / 8; u++) {
                        const int rem = (R & 7) * kch + u * 8;
                        ko[R][u] = (rem >> 6) * 8 + ((rem & 63) >> 3);
                    }
                for (int mb0 = 0; mb0 < M; mb0 += 16) {
                    for (int ni = 0; ni < 2; ni++) {
                        const int mb = mb0 + ni * 8, oh = mb / OW, ow0 = mb % OW;
                        rw[ni] = P + (size_t)(oh * sh) * Wp + (size_t)ow0 * sw;
                    }
                    uint16_t *blk = Xd + (size_t)(mb0 >> 4) * 16 * Kp;
                    for (int j0 = 0; j0 < Kp; j0 += kch)
                        for (int R = 0; R < 16; R++) {
                            const uint16_t *row0 = rw[R >> 3];
                            uint16_t *d = blk + (size_t)R * Kp + j0;
                            for (int u = 0; u < kch / 8; u++) {
                                const int k = j0 + ko[R][u];
                                uint16x8_t v;
                                if (k >= K)
                                    v = vdupq_n_u16(0);
                                else if (sw == 1)
                                    v = vld1q_u16(row0 + sk[k]);
                                else
                                    v = vld2q_u16(row0 + sk[k]).val[0];
                                vst1q_u16(d + u * 8, v);
                            }
                        }
                }
            } else if ((g_uc_var & 64) && sw == 1 && OW % 16 == 0) {
                /* GHL: rows R and R + 8 take the same k at px mb0..+7 and mb0+8..+15 of one output row: one 32-B
                 * load (16 contiguous px of the padded plane) feeds both */
                for (int mb0 = 0; mb0 < M; mb0 += 16) {
                    const int oh = mb0 / OW, ow0 = mb0 % OW;
                    const uint16_t *row0 = P + (size_t)(oh * sh) * Wp + (size_t)ow0;
                    uint16_t *blk = Xd + (size_t)(mb0 >> 4) * 16 * Kp;
                    for (int R = 0; R < 8; R++) {
                        for (int u = 0; u < kch / 8; u++) {
                            const int rem = R * kch + u * 8;
                            kofs[u] = (rem >> 6) * 8 + ((rem & 63) >> 3);
                        }
                        uint16_t *d0 = blk + (size_t)R * Kp, *d1 = d0 + (size_t)8 * Kp;
                        for (int j0 = 0; j0 < Kp; j0 += kch)
                            for (int u = 0; u < kch / 8; u++) {
                                const int k = j0 + kofs[u];
                                if (k >= K) {
                                    vst1q_u16(d0 + j0 + u * 8, vdupq_n_u16(0));
                                    vst1q_u16(d1 + j0 + u * 8, vdupq_n_u16(0));
                                } else {
                                    const uint16_t *s = row0 + sk[k];
                                    vst1q_u16(d0 + j0 + u * 8, vld1q_u16(s));
                                    vst1q_u16(d1 + j0 + u * 8, vld1q_u16(s + 8));
                                }
                            }
                    }
                }
            } else
            for (int mb0 = 0; mb0 < M; mb0 += 16)
                for (int R = 0; R < 16; R++) {
                    const int ni = R >> 3, mb = mb0 + ni * 8, oh = mb / OW, ow0 = mb % OW;
                    const uint16_t *row0 = P + (size_t)(oh * sh) * Wp + (size_t)ow0 * sw;
                    uint16_t *dst = Xd + (size_t)(mb0 >> 4) * 16 * Kp + (size_t)R * Kp;
                    for (int u = 0; u < kch / 8; u++) {
                        const int rem = (R & 7) * kch + u * 8;
                        kofs[u] = (rem >> 6) * 8 + ((rem & 63) >> 3);
                    }
                    for (int j0 = 0; j0 < Kp; j0 += kch)
                        for (int u = 0; u < kch / 8; u++) {
#if defined(__aarch64__)
                            if (zx && ((j0 + u * 8) & 31) == 0) /* row written front to back: allocate the line zeroed */
                                __asm__ volatile("dc zva, %0" ::"r"(dst + j0 + u * 8) : "memory");
#endif
                            const int k = j0 + kofs[u];
                            uint16x8_t v;
                            if (k >= K)
                                v = vdupq_n_u16(0);
                            else if (sw == 1)
                                v = vld1q_u16(row0 + sk[k]);
                            else
                                v = vld2q_u16(row0 + sk[k]).val[0];
                            vst1q_u16(dst + j0 + u * 8, v);
                        }
                }
        } else
#endif
        for (int c0 = 0; c0 < C; c0 += CG) {
            const int k0 = c0 * KK, k1 = (c0 + CG < C ? c0 + CG : C) * KK;
            for (int mb = 0; mb < M; mb += 8) {
                const int oh = mb / OW, ow0 = mb % OW;
                uint16_t *blk = Xd + (size_t)(mb >> 4) * 16 * Kp + (size_t)((mb >> 3) & 1) * 8 * Kp;
                const uint16_t *row0 = P + (size_t)(oh * sh) * Wp + (size_t)ow0 * sw;
#if UC_NEON
                if (sw == 1)
                    for (int k = k0; k < k1; k++)
                        vst1q_u16(blk + dk[k], vld1q_u16(row0 + sk[k]));
                else
                    for (int k = k0; k < k1; k++)
                        vst1q_u16(blk + dk[k], vld2q_u16(row0 + sk[k]).val[0]);
#else
                for (int k = k0; k < k1; k++)
                    for (int i = 0; i < 8; i++)
                        blk[dk[k] + i] = row0[sk[k] + i * sw];
#endif
            }
        }
        for (int mb = 0; mb < M && Kp > K && !(UC_NEON && (g_uc_var & 4)); mb += 8) {
            uint16_t *blk = Xd + (size_t)(mb >> 4) * 16 * Kp + (size_t)((mb >> 3) & 1) * 8 * Kp;
            for (int k = K; k < Kp; k++)
                memset(blk + dk[k], 0, 16);
        }
        g_uc_fast++;
    } else {
        /* im2col of the activation, fp32 -> bf16 RNE; the in-chunk offset of (ni, nn, kr) from a table */
        static int g_uc_tab[16][64];
        static int g_uc_ty[64], g_uc_tx[64];
        for (int q = 0; q < 16; q++)
            for (int kr = 0; kr < kch; kr++) {
                const int f = (q >> 3) * 8 * kch + (kr >> 3) * 64 + (kr & 7) * 8 + (q & 7);
                g_uc_tab[q][kr] = (f / kch) * Kp + f % kch;
            }
        for (int t = 0; t < KK; t++) {
            g_uc_ty[t] = t / KW;
            g_uc_tx[t] = t % KW;
        }
        const size_t HWc = (size_t)H * W * cb;
        for (int m = 0; m < M; m++) {
            const int oh = m / OW, ow = m % OW, ih0 = oh * sh - ph, iw0 = ow * sw - pw;
            const int *tab = g_uc_tab[m & 15];
            uint16_t *dst = Xd + (size_t)(m >> 4) * 16 * Kp;
            int kr = 0;
            uint16_t *dj = dst;
            for (int c = 0; c < C; c++) {
                const float *xc = xin + (size_t)(c / cb) * HWc + c % cb;
                for (int t = 0; t < KK; t++) {
                    const int ih = ih0 + g_uc_ty[t], iw = iw0 + g_uc_tx[t];
                    const float v = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? xc[((size_t)ih * W + iw) * cb] : 0.0f;
                    dj[tab[kr]] = uc_f2bf(v);
                    if (++kr == kch) {
                        kr = 0;
                        dj += kch;
                    }
                }
            }
            for (int k = K; k < Kp; k++) {
                dj[tab[kr]] = 0;
                if (++kr == kch) {
                    kr = 0;
                    dj += kch;
                }
            }
        }
    }
    const size_t wbytes = (size_t)N * Kp * 2, xbytes = (size_t)M * Kp * 2;
    (void)kc;
    uc_ysentinel(g_ky, (size_t)M * N * 4); // sentinel: unwritten -> NaN -> counted
    XTime lt0, lt1;
    XTime_GetTime(&lt0);
    const unsigned long long hw0 = uc_hs_wait();
    ahs_add_span(&g_ahs, AHS_PACK, ut0, lt0);
    const unsigned long long sp0 = uc_pmc();
#if UC_WARM
    unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
    __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
    uc_sync_pre(g_kw, wbytes, g_kx, xbytes, g_ky, (size_t)M * N * 4);
    const unsigned long long rt0 = uc_rt_cyc(), sp1 = uc_pmc();
    printf("[UCS] l=%d M=%d N=%d K=%d tkw=%d\n", g_aie_launches, M, N, K, kch / 2); /* P4bp: names a hanging launch */
    fflush(stdout);
    const int lrc = uc_kacc_launch(N, K, M, g_kw, g_kx, g_ky); // ONE launch: the whole layer, K summed in the cores
    g_kw = kw_pool;
    const unsigned long long sp2 = uc_pmc(), rt1 = uc_rt_cyc();
    uc_sync_post(g_ky, (size_t)M * N * 4);
#if UC_WARM
    __Runtime_pm_warm_stats(&seg1, &cold1);
    if (cold1 != cold0) {
        g_uc_kloads++;
        g_uc_cold_tiles += cold1 - cold0;
    }
#endif
    const unsigned long long sp3 = uc_pmc();
    const unsigned long long hw1 = uc_hs_wait();
    XTime_GetTime(&lt1);
    ahs_add_window(&g_ahs, (unsigned long long)(lt1 - lt0), hw1 - hw0);
    if (lrc != 0)
        return lrc;
    g_sp_pre += sp1 - sp0;
    g_sp_call += sp2 - sp1;
    g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;
    g_sp_post += sp3 - sp2;
    g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
    g_aie_launches++;
    g_uc_macs += (unsigned long long)M * N * K;
    /* GPOOL: epilogue (bias / temb per channel, or residual in the output layout) fused into the scatter,
     * applied in argument order as before; 8 px of one channel are 8 contiguous floats of the c window. */
    const float *ep[4];
    int et[4], ne_ = 0; /* et: 0 = per channel, 1 = same shape */
    for (int i = 2; i < nargs - 1; i++) {
        const uc_dlt *e = (const uc_dlt *)argv[i];
        const long ne = uc_numel(e);
        if (ne_ == 4)
            return -5;
        if (ne == N && e->ndim == 5)
            et[ne_] = 0;
        else if (ne == (long)M * N && e->ndim == 5 && (int)e->shape[4] == yb && yb != 1)
            et[ne_] = 1;
        else if (ne == (long)M * N && e->ndim == 5 && yb == 1 && UC_NEON) /* P4a planar out, NCHWc residual */
            et[ne_] = 2 + (int)e->shape[4];
        else
            return -5;
        ep[ne_++] = (const float *)e->data;
    }
    int N_ = N; /* GHL: rows left for the scalar scatter */
    const float *Y = (const float *)g_ky; // [N][M] fp32 (NCHW)
    int bad = 0;
#if UC_NEON
    if (yb % 4 == 0 || yb == 1) { /* P4a: yb == 1 = planar NCHW (keep_layout_rt.py label) */
        bad = uc_scatter16(Y, M, N, tr, 0, M, yout, M, yb, ep, et, ne_); /* GHL: 16x16 tiles */
        N_ = 0; /* scalar loop below skipped */
    }
#endif
    for (int n = 0; n < N_; n++) {
        const int r = n & 15;
        float *on = yout + (size_t)(n / yb) * M * yb + n % yb;
        for (int m = 0; m < M; m += 8) {
            /* P4bx: each core's c block holds 16x16 tiles (ch r, px c) in [s][mi][ni][8x8] order. Normal: block
             * (n/16, m/(M/4)) of Y [N][M], s = px sub-tile. Transposed: block (m/16, n/(N/4)) of Y^T [M][N],
             * s = ch sub-tile. */
            const float *src;
            if (!tr) {
                const int c = m & 15, q = M / 4, cc = m % q;
                const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                src = Y + (size_t)((n & ~15) + wi / q) * M + (m - cc) + wi % q;
            } else {
                const int c = m & 15, q = N / 4, cc = n % q;
                const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
                src = Y + (size_t)((m & ~15) + wi / q) * N + (n - cc) + wi % q;
            }
            float v[8]; /* (c & 7) = wi & 7 and q % 8 == 0: the 8 px are contiguous in both walks */
            memcpy(v, src, 32);
            for (int i = 0; i < 8; i++)
                if (!(v[i] == v[i]) || v[i] > 1e30f || v[i] < -1e30f)
                    bad++;
            float *o = on + (size_t)m * yb;
            for (int e = 0; e < ne_; e++) {
                if (et[e] == 0) {
                    const float bv = ep[e][n];
                    for (int i = 0; i < 8; i++)
                        v[i] += bv;
                } else {
                    const float *ev = ep[e] + (o - yout);
                    for (int i = 0; i < 8; i++)
                        v[i] += ev[(size_t)i * yb];
                }
            }
            for (int i = 0; i < 8; i++)
                o[(size_t)i * yb] = v[i];
        }
    }
    if (bad) {
        printf("[aiednn_aie_conv] %d nonfinite AIE outputs\n", bad);
        return -4;
    }
    {
        XTime ut3;
        XTime_GetTime(&ut3);
        ahs_add_span(&g_ahs, AHS_SCATTER, lt1, ut3);
        const double us = 1e6 / (double)COUNTS_PER_SECOND;
        /* GPOOL: [UCL] is recorded here and printed by aiednn_aie_split_stats() after graph_run (a printf per
         * layer cost ~3.5 ms of A78 time inside the site) */
        const int li = g_aie_launches - 1;
        if (li >= 0 && li < UC_UCL_MAX) {
            uc_ucl *u = &g_uc_ucl[li];
            u->M = M;
            u->N = N;
            u->K = K;
            u->fast = fast | (wz << 1); /* bit0 fast im2col, bit1 zero-copy W */
            u->pre = (float)((double)(lt0 - ut0) * us);
            u->aie = (float)((double)(lt1 - lt0) * us);
            u->post = (float)((double)(ut3 - lt1) * us);
            if (li + 1 > g_uc_nucl)
                g_uc_nucl = li + 1;
        }
    }
    return 0;
}

