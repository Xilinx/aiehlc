
/* PDN (--aie conv,linear): one TVM nn_contrib_dense_pack call site on the AIE. Y[M][N] = X[M][K] . W[N][K]^T
 * (+ bias [1,N] / residual [M,N] epilogue) as S launches of ONE gemm_kacc instance (N, K, px=Mp/S), split
 * along px (kacc_split, as kacc_gemm.py / PDK c009). W is pre-laid at build time in the kacc chunk layout
 * (unet_aie.kacc_dense_wlayout); X is tiled here fp32 -> bf16 RNE, rows M..Mp-1 zero. */
#if UC_NEON && UC_WHOOK
/* GHL: scatter of launch s-1's Y block stepped from the runtime's wait_io poll while launch s computes (one
 * 16-row m-tile per poll iteration); the block was invalidated after launch s-1 returned and launch s's
 * DMA neither reads nor writes it. Whatever the poll did not reach is finished after the launch. */
typedef int (*Runtime_wait_hook_fn)(void *arg);
void __Runtime_set_wait_hook(Runtime_wait_hook_fn fn, void *arg); /* C++ linkage, aie_runtime.h */
typedef struct {
    const float *Ys;
    int P, N, tr, mbase, mvalid, M, nmt, t, bad, ne;
    float *yout;
    const float *const *ep;
    const int *et;
} uc_whook;
static int uc_whook_step(void *a) {
    uc_whook *h = (uc_whook *)a;
    if (h->t < h->nmt) {
        h->bad += uc_scatter16(h->Ys, h->P, h->N, h->tr, h->mbase, h->mvalid, h->yout, h->M, 0, h->ep, h->et, h->ne,
                               h->t, h->t + 1);
        h->t++;
    }
    return h->t >= h->nmt;
}
static unsigned long long g_uc_whook_tiles = 0, g_uc_whook_all = 0; /* m-tiles done in the poll / overlappable */
#endif
static int uc_linear(void *const *argv, int nargs, const uint16_t *wbf, int S) {
    XTime ut0;
    XTime_GetTime(&ut0);
    ahs_begin(&g_ahs, AHS_KIND_LINEAR);
    int wzc_ = 0;
    if (nargs < 3)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 2 || w->ndim != 3 || y->ndim != 2)
        return -2;
    const int M = (int)x->shape[0], K = (int)x->shape[1], N = (int)y->shape[1];
    if ((int)y->shape[0] != M || (int)w->shape[1] != K || (int)(w->shape[0] * w->shape[2]) != N)
        return -3;
    const int Mp = (M + 63) & ~63;
    if (S < 1 || Mp % (64 * S) || (N | K) & 63)
        return -3;
    const int P = Mp / S; /* px per launch */
    int tkw = 4096 / P < 32 ? 4096 / P : 32; /* = kacc_geom(N, K, P) */
    const int tr = 2 * (16 * tkw * 4 + (P / 4) * tkw * 4 + 16 * (P / 4) * 4) > 29696;
    if (tr)
        for (tkw = 32; (N / 4) * tkw > 1024 || (K / 2) % tkw; tkw /= 2)
            ;
    const int kch = 2 * tkw;
    const int Kp = (N == K / 2) ? K + kch : K;
    if (kch > 64 || K % kch || (long)N * Kp > UC_KW_ELEMS || (long)Mp * Kp > UC_KX_ELEMS ||
        (long)Mp * N > UC_KY_ELEMS)
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
    uint16_t *Wd = (uint16_t *)g_kw, *Xd = (uint16_t *)g_kx;
    int32_t *const kw_pool = g_kw; /* GHL: the pooled W port, restored after the launches */
    if (Kp == K && ((uintptr_t)wbf & 63) == 0) {
        /* GHL (= GPOOL zero-copy W): wbf is the exact port image (kacc_dense_wlayout, build time) -> read in
         * place by the launches (flushed below like the pool) */
        g_kw = (int32_t *)(uintptr_t)wbf;
        g_ul_wzc++;
        wzc_ = 1;
    } else if (Kp == K) {
        memcpy(Wd, wbf, (size_t)N * K * 2);
    } else {
        for (int n = 0; n < N; n++) {
            memcpy(Wd + (size_t)n * Kp, wbf + (size_t)n * K, (size_t)K * 2);
            memset(Wd + (size_t)n * Kp + K, 0, (size_t)(Kp - K) * 2);
        }
    }
    /* X chunk layout = uc_conv im2col table: row m -> block m/16, in-chunk f = (q>>3)*8*kch + (kr>>3)*64 +
     * (kr&7)*8 + (q&7), q = m&15, chunk j at j*kch of each of the 16 rows (row stride Kp) */
    static int g_ul_tab[16][64];
    for (int q = 0; q < 16; q++)
        for (int kr = 0; kr < kch; kr++) {
            const int f = (q >> 3) * 8 * kch + (kr >> 3) * 64 + (kr & 7) * 8 + (q & 7);
            g_ul_tab[q][kr] = (f / kch) * Kp + f % kch;
        }
#if UC_NEON
    /* GHL: 4 rows x 4 k per 4x4 transpose: for a fixed k the 4 rows q0..q0+3 (q0 % 4 == 0) are 4 contiguous
     * bf16 of the chunk ((q & 7) is the fastest index of f) -> one 8-byte store, RNE as uc_f2bf */
    for (int m0 = 0; m0 < Mp; m0 += 4) {
        const int q0 = m0 & 15;
        const int *tab = g_ul_tab[q0];
        uint16_t *dst = Xd + (size_t)(m0 >> 4) * 16 * Kp;
        if ((g_uc_var & 16) && (m0 & 7) == 0 && m0 + 8 <= M) {
            /* GHL: 8 rows x 8 k: rows -> bf16 (RNE), 8x8 u16 transpose, one 16-byte store per k (8 px) */
            const float *xr = xin + (size_t)m0 * K;
            for (int k0 = 0; k0 < K; k0 += 8) {
                uint16x8_t r[8];
                for (int i = 0; i < 8; i++)
                    r[i] = vcombine_u16(uc_f2bf4(vld1q_f32(xr + (size_t)i * K + k0)),
                                        uc_f2bf4(vld1q_f32(xr + (size_t)i * K + k0 + 4)));
                const uint16x8x2_t t0 = vtrnq_u16(r[0], r[1]), t1 = vtrnq_u16(r[2], r[3]), t2 = vtrnq_u16(r[4], r[5]),
                                   t3 = vtrnq_u16(r[6], r[7]);
                const uint32x4x2_t u0 = vtrnq_u32(vreinterpretq_u32_u16(t0.val[0]), vreinterpretq_u32_u16(t1.val[0])),
                                   u1 = vtrnq_u32(vreinterpretq_u32_u16(t0.val[1]), vreinterpretq_u32_u16(t1.val[1])),
                                   u2 = vtrnq_u32(vreinterpretq_u32_u16(t2.val[0]), vreinterpretq_u32_u16(t3.val[0])),
                                   u3 = vtrnq_u32(vreinterpretq_u32_u16(t2.val[1]), vreinterpretq_u32_u16(t3.val[1]));
#define UC_LO(a, b) vcombine_u16(vget_low_u16(vreinterpretq_u16_u32(a)), vget_low_u16(vreinterpretq_u16_u32(b)))
#define UC_HI(a, b) vcombine_u16(vget_high_u16(vreinterpretq_u16_u32(a)), vget_high_u16(vreinterpretq_u16_u32(b)))
                const uint16x8_t c[8] = {UC_LO(u0.val[0], u2.val[0]), UC_LO(u1.val[0], u3.val[0]),
                                         UC_LO(u0.val[1], u2.val[1]), UC_LO(u1.val[1], u3.val[1]),
                                         UC_HI(u0.val[0], u2.val[0]), UC_HI(u1.val[0], u3.val[0]),
                                         UC_HI(u0.val[1], u2.val[1]), UC_HI(u1.val[1], u3.val[1])};
#undef UC_LO
#undef UC_HI
                const int j = k0 / kch, kr = k0 - j * kch;
                uint16_t *dj = dst + (size_t)j * kch;
                for (int kk = 0; kk < 8; kk++)
                    vst1q_u16(dj + tab[kr + kk], c[kk]);
            }
            for (int k = K; k < Kp; k++) {
                const int j = k / kch, kr = k - j * kch;
                vst1q_u16(dst + (size_t)j * kch + tab[kr], vdupq_n_u16(0));
            }
            m0 += 4; /* + the loop's 4: next 8-row group */
            continue;
        }
        if (m0 + 4 <= M) {
            const float *x0 = xin + (size_t)m0 * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
            for (int k0 = 0; k0 < K; k0 += 4) {
                float32x4_t r0 = vld1q_f32(x0 + k0), r1 = vld1q_f32(x1 + k0), r2 = vld1q_f32(x2 + k0),
                            r3 = vld1q_f32(x3 + k0);
                uc_tr4(r0, r1, r2, r3);
                const int j = k0 / kch, kr = k0 - j * kch;
                uint16_t *dj = dst + (size_t)j * kch;
                vst1_u16(dj + tab[kr], uc_f2bf4(r0));
                vst1_u16(dj + tab[kr + 1], uc_f2bf4(r1));
                vst1_u16(dj + tab[kr + 2], uc_f2bf4(r2));
                vst1_u16(dj + tab[kr + 3], uc_f2bf4(r3));
            }
        } else {
            for (int k0 = 0; k0 < K; k0 += 4) {
                const int j = k0 / kch, kr = k0 - j * kch;
                uint16_t *dj = dst + (size_t)j * kch;
                for (int jj = 0; jj < 4; jj++)
                    for (int i = 0; i < 4; i++)
                        dj[tab[kr + jj] + i] = m0 + i < M ? uc_f2bf(xin[(size_t)(m0 + i) * K + k0 + jj]) : 0;
            }
        }
        for (int k = K; k < Kp; k++) {
            const int j = k / kch, kr = k - j * kch;
            vst1_u16(dst + (size_t)j * kch + tab[kr], vdup_n_u16(0));
        }
    }
    if (0)
#endif
    for (int m = 0; m < Mp; m++) {
        const int *tab = g_ul_tab[m & 15];
        uint16_t *dst = Xd + (size_t)(m >> 4) * 16 * Kp;
        const float *xr = xin + (size_t)m * K;
        for (int j = 0; j * kch < Kp; j++) {
            uint16_t *dj = dst + (size_t)j * kch;
            const int k0 = j * kch;
            if (m < M && k0 < K)
                for (int kr = 0; kr < kch; kr++)
                    dj[tab[kr]] = uc_f2bf(xr[k0 + kr]);
            else
                for (int kr = 0; kr < kch; kr++)
                    dj[tab[kr]] = 0;
        }
    }
    const size_t wbytes = (size_t)N * Kp * 2, xbytes = (size_t)Mp * Kp * 2, ybytes = (size_t)Mp * N * 4;
    /* GHL: epilogue (bias [1,N] / residual [M,N], argument order kept) fused into the scatter */
    const float *ep[4];
    int et[4], ne_ = 0; /* et: 0 = bias [N], 1 = residual [M][N] */
    for (int i = 2; i < nargs - 1; i++) {
        const uc_dlt *e = (const uc_dlt *)argv[i];
        const long ne = uc_numel(e);
        if (ne_ == 4)
            return -5;
        if (ne == N)
            et[ne_] = 0;
        else if (ne == (long)M * N)
            et[ne_] = 1;
        else
            return -5;
        ep[ne_++] = (const float *)e->data;
    }
    const float *Y = (const float *)g_ky; /* launch s: [N][P] tiled block at s*N*P (P4bx [s][mi][ni][8x8]) */
    int bad = 0;
#if UC_NEON && UC_WHOOK
    uc_whook wh;
    memset(&wh, 0, sizeof(wh));
    wh.t = wh.nmt = 0;
#endif
    uc_ysentinel(g_ky, ybytes); // sentinel: unwritten -> NaN -> counted
    XTime la0, la1; /* GHL: [UCL] split of the site, aie = all its launch windows */
    XTime_GetTime(&la0);
    ahs_add_span(&g_ahs, AHS_PACK, ut0, la0);
    unsigned long long hs_win = 0ULL; /* GAP-RT1: sum of the launch windows; rest of [la0, la1] = loop glue -> dma */
    for (int s = 0; s < S; s++) {
        XTime lt0, lt1;
        XTime_GetTime(&lt0);
        const unsigned long long hw0 = uc_hs_wait();
        const unsigned long long sp0 = uc_pmc();
#if UC_WARM
        unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
        __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
        if (s == 0) /* GHL: host->DMA sync in the first launch's window, as uc_conv */
            uc_sync_pre(g_kw, wbytes, g_kx, xbytes, g_ky, ybytes);
        const unsigned long long rt0 = uc_rt_cyc(), sp1 = uc_pmc();
        printf("[UCS] l=%d lin M=%d N=%d K=%d px=%d s=%d/%d tkw=%d\n", g_aie_launches, M, N, K, P, s, S, tkw);
        fflush(stdout);
#if UC_NEON && UC_WHOOK
        if (s > 0 && (g_uc_var & 1) == 0 && (s - 1) * P < M) { /* block s-1 scattered while launch s computes */
            wh.Ys = Y + (size_t)(s - 1) * N * P;
            wh.P = P;
            wh.N = N;
            wh.tr = tr;
            wh.mbase = (s - 1) * P;
            wh.mvalid = M - (s - 1) * P < P ? M - (s - 1) * P : P;
            wh.M = M;
            wh.yout = yout;
            wh.ep = ep;
            wh.et = et;
            wh.ne = ne_;
            wh.t = 0;
            wh.nmt = (wh.mvalid + 15) / 16;
            __Runtime_set_wait_hook(uc_whook_step, &wh);
        }
#endif
        const int lrc = uc_kacc_launch(N, K, P, g_kw, g_kx + (size_t)s * P * Kp / 2, g_ky + (size_t)s * N * P);
        const unsigned long long sp2 = uc_pmc(), rt1 = uc_rt_cyc();
#if UC_NEON && UC_WHOOK
        __Runtime_set_wait_hook(0, 0);
        if (g_uc_opt & 4) {
            if (s == S - 1)
                uc_sync_post(g_ky, ybytes);
        } else { /* per-block invalidate (blocks tile Y): block s is read (next launch's poll / below) only after */
            Xil_DCacheInvalidateRange((UINTPTR)(g_ky + (size_t)s * N * P), (size_t)N * P * 4);
        }
#else
        if (s == S - 1)
            uc_sync_post(g_ky, ybytes);
#endif
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
        hs_win += (unsigned long long)(lt1 - lt0);
        if (lrc != 0) {
            g_kw = kw_pool;
            return lrc;
        }
#if UC_NEON && UC_WHOOK
        if (s > 0 && wh.nmt > 0) { /* the rest of block s-1 the poll did not reach (host time) */
            g_uc_whook_tiles += (unsigned long long)wh.t;
            g_uc_whook_all += (unsigned long long)wh.nmt;
            bad += wh.bad;
            if (wh.t < wh.nmt)
                bad += uc_scatter16(wh.Ys, P, N, tr, wh.mbase, wh.mvalid, yout, M, 0, ep, et, ne_, wh.t, wh.nmt);
            wh.t = wh.nmt = 0;
            wh.bad = 0;
        }
#endif
        g_sp_call += sp2 - sp1;
        g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;
        g_sp_post += sp3 - sp2;
        g_sp_pre += sp1 - sp0;
        /* M2 N segment: the same in-launch host glue per site (PMU cycles; ahs_cal pairs them with XTime) */
        ahs_add(&g_ahs, AHS_GLUE, (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL);
        ahs_add(&g_ahs, AHS_FLUSH, sp1 - sp0);
        ahs_add(&g_ahs, AHS_INVAL, sp3 - sp2);
        ahs_launch(&g_ahs);
        ahs_cal(&g_ahs, sp0, (unsigned long long)lt0);
        ahs_cal(&g_ahs, sp3, (unsigned long long)lt1);
        g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
        g_aie_launches++;
    }
    g_kw = kw_pool;
    XTime_GetTime(&la1);
    if ((unsigned long long)(la1 - la0) > hs_win)
        ahs_add(&g_ahs, AHS_DMA, (unsigned long long)(la1 - la0) - hs_win);
    g_uc_macs += (unsigned long long)M * N * K; /* real rows only (M, not Mp) */
#if UC_NEON
#if UC_WHOOK
    for (int s = (g_uc_var & 1) == 0 ? S - 1 : 0; s < S && s * P < M; s++) /* blocks < S-1: done above */
#else
    for (int s = 0; s < S && s * P < M; s++) /* GHL: 16x16 tiles per launch block */
#endif
        bad += uc_scatter16(Y + (size_t)s * N * P, P, N, tr, s * P, (M - s * P < P ? M - s * P : P), yout, M, 0, ep,
                            et, ne_);
    if (0)
#endif
    for (int m = 0; m < M; m++) {
        const int s = m / P, ml = m % P, c = ml & 15;
        const float *Ys = Y + (size_t)s * N * P;
        float *yo = yout + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const int r = n & 15;
            size_t i;
            if (!tr) {
                const int q = P / 4, cc = ml % q;
                const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
                i = (size_t)((n & ~15) + wi / q) * P + (ml - cc) + wi % q;
            } else {
                const int q = N / 4, cc = n % q;
                const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
                i = (size_t)((ml & ~15) + wi / q) * N + (n - cc) + wi % q;
            }
            float v = Ys[i];
            if (!(v == v) || v > 1e30f || v < -1e30f)
                bad++;
            for (int e = 0; e < ne_; e++)
                v += ep[e][et[e] ? (size_t)m * N + n : (size_t)n];
            yo[n] = v;
        }
    }
    if (bad) {
        printf("[aiednn_aie_linear] %d nonfinite AIE outputs\n", bad);
        return -4;
    }
    {
        XTime ut3;
        XTime_GetTime(&ut3);
        ahs_add_span(&g_ahs, AHS_SCATTER, la1, ut3);
        const double us = 1e6 / (double)COUNTS_PER_SECOND;
        const int li = g_aie_launches - 1; /* GHL: [UCL] of the site's last launch; fast = 4 (linear) | 2 zero-copy W */
        if (li >= 0 && li < UC_UCL_MAX) {
            uc_ucl *u = &g_uc_ucl[li];
            u->M = M;
            u->N = N;
            u->K = K;
            u->fast = 4 | (wzc_ << 1);
            u->pre = (float)((double)(la0 - ut0) * us);
            u->aie = (float)((double)(la1 - la0) * us);
            u->post = (float)((double)(ut3 - la1) * us);
            if (li + 1 > g_uc_nucl)
                g_uc_nucl = li + 1;
        }
    }
    return 0;
}

extern "C" int aiednn_aie_linear(int cid, void *args, int nargs, const uint16_t *wbf, int S) {
    g_uc_calls++;
    const int rc = uc_linear((void *const *)args, nargs, wbf, S);
    if (rc != 0) {
        printf("[aiednn_aie_linear] layer %d rc=%d\n", cid, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

