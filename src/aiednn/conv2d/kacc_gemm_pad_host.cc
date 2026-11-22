/* aiednn/conv2d/kacc_gemm_pad_host.cc -- any-shape conv2d_NCHWc / dense on the gemm_kacc instances (aiegraph.cnn).
 *
 * The layer GEMM is Y[n][m] = sum_k W[n][k] X[m][k] (n = output channel, m = output pixel, k = c*KH*KW + kh*KW + kw).
 * One gemm_kacc instance (Ng, Kg, PX) per distinct padded shape: Ng = ceil64(N), Kg = ceil64(K), PX = the pixel
 * width of one launch (aiegraph.cnn.instance; a small K runs on a larger-K instance, cnn.merge_small_k). The pixels are walked in ceil(M / PX) launches; every padded row /
 * column is zero. Host side per call: W natural [N][K] bf16 (pre-laid by aiegraph at build time) -> the kacc
 * chunk layout; per launch im2col of PX pixels (fp32 -> bf16 RNE) -> launch -> scatter of the valid [N][px]
 * block into the NCHWc output with the fused adds (per-channel [N] or the output's own shape, argument order)
 * and an optional ReLU (act = 1). Needs kacc_conv_host.cc before it (uc_kalloc, uc_kacc_launch, uc_sync_*).
 *
 * Entries (C linkage, the "args" ABI of an aiegraph one-call group; px / kg = the instance's PX / Kg):
 *   aiednn_conv(cid, args, nargs, wbf, sh, sw, ph, pw, act, px, kg)   x [1,C/cb,H,W,cb], w [O/ob,C/ib,KH,KW,ib,ob]
 *   aiednn_dense(cid, args, nargs, wbf, act, px, kg)                  x [M][K], w [N/nb][K][nb], y [M][N]
 */
#if AIEDNN_SELFCHECK
#include <math.h>
#endif
/* kacc_geom of aiegraph.cnn: K chunk, transposed walk, padded K of instance (N, K, M) */
static int ap_geom(int N, int K, int M, int *kch_o, int *tr_o, int *kp_o) {
    int tkw = 4096 / M < 32 ? 4096 / M : 32;
    const int tr = 2 * (16 * tkw * 4 + (M / 4) * tkw * 4 + 16 * (M / 4) * 4) > 29696;
    if (tr)
        for (tkw = 32; (N / 4) * tkw > 1024 || (K / 2) % tkw; tkw /= 2)
            ;
    if (tkw < 4 || (K / 2) % tkw)
        return -3;
    const int kch = 2 * tkw;
    *kch_o = kch;
    *tr_o = tr;
    *kp_o = (N == K / 2) ? K + kch : K;
    return 0;
}
/* W port: row r (channel) of 16-row block, chunk j at j*kch, in-chunk f = mi*8*kch + kb*64 + mm*8 + kk */
static inline size_t ap_woff(int r, int k, int kch, int Kp) {
    const int kr = k % kch, q = r & 15;
    const int f = (q >> 3) * 8 * kch + (kr >> 3) * 64 + (q & 7) * 8 + (kr & 7);
    return (size_t)(r >> 4) * 16 * Kp + (size_t)(k - kr) + (size_t)(f / kch) * Kp + (size_t)(f % kch);
}
/* X port: row r (pixel), in-chunk f = ni*8*kch + kb*64 + kk*8 + nn (8x8 k x px) */
static inline size_t ap_xoff(int r, int k, int kch, int Kp) {
    const int kr = k % kch, q = r & 15;
    const int f = (q >> 3) * 8 * kch + (kr >> 3) * 64 + (kr & 7) * 8 + (q & 7);
    return (size_t)(r >> 4) * 16 * Kp + (size_t)(k - kr) + (size_t)(f / kch) * Kp + (size_t)(f % kch);
}
/* the 8 contiguous fp32 of channel n, launch pixels m8..m8+7 (m8 % 8 == 0) in the c window layout */
static inline const float *ap_y8(const float *Y, int n, int m8, int Ng, int P, int tr) {
    const int r = n & 15, c = m8 & 15;
    if (!tr) {
        const int q = P / 4, cc = m8 % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
        return Y + (size_t)((n & ~15) + wi / q) * P + (m8 - cc) + wi % q;
    }
    const int q = Ng / 4, cc = n % q;
    const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8;
    return Y + (size_t)((m8 & ~15) + wi / q) * Ng + (n - cc) + wi % q;
}

/* x element (c, ih, iw) = xin[(c / cb) * H * W * cb + (ih * W + iw) * cb + c % cb]; output (n, p) at
 * yout[(n / yb) * M * yb + p * yb + n % yb] (M = OH * OW). */
static int ap_gemm(const float *xin, int C, int H, int W, int cb, int KH, int KW, int sh, int sw, int ph, int pw,
                   int OH, int OW, const uint16_t *wbf, int N, void *const *argv, int nargs, float *yout, int yb,
                   int act, int PX, int Kg, int hs_kind) {
    XTime ut0;
    XTime_GetTime(&ut0);
    ahs_begin(&g_ahs, hs_kind); /* per-site host split + site perf (aie_host_split.h, as kacc_conv_host.cc) */
    const int M = OH * OW, K = C * KH * KW, KK = KH * KW;
    const int Ng = (N + 63) & ~63;
    int kch = 0, tr = 0, Kp = 0;
    if (PX <= 0 || PX % 64 || Kg < K || Kg % 64 || ap_geom(Ng, Kg, PX, &kch, &tr, &Kp) != 0)
        return -3;
    if ((long)Ng * Kp > UC_KW_ELEMS || (long)PX * Kp > UC_KX_ELEMS || (long)Ng * PX > UC_KY_ELEMS)
        return -3;
    /* fused adds in argument order: et 0 = per channel [N], 1 = the output's own index */
    const float *ep[4];
    int et[4], ne_ = 0;
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
    if (uc_kalloc() != 0)
        return -1;
#if UC_CTRLPKT
    if (!g_uc_ht) { // once, before the first launch
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    uint16_t *Wd = (uint16_t *)g_kw, *Xd = (uint16_t *)g_kx;
    memset(Wd, 0, (size_t)Ng * Kp * 2);
    for (int n = 0; n < N; n++) {
        const uint16_t *wr = wbf + (size_t)n * K;
        for (int k = 0; k < K; k++)
            Wd[ap_woff(n, k, kch, Kp)] = wr[k];
    }
    const size_t HWc = (size_t)H * W * cb;
    const size_t wbytes = (size_t)Ng * Kp * 2, xbytes = (size_t)PX * Kp * 2, ybytes = (size_t)Ng * PX * 4;
    const float *Y = (const float *)g_ky;
    int bad = 0;
    for (int m0 = 0; m0 < M; m0 += PX) {
        const int mv = M - m0 < PX ? M - m0 : PX;
        if (m0 > 0)
            XTime_GetTime(&ut0);
        memset(Xd, 0, xbytes);
        for (int r = 0; r < mv; r++) {
            const int m = m0 + r, oh = m / OW, ow = m % OW, ih0 = oh * sh - ph, iw0 = ow * sw - pw;
            for (int c = 0; c < C; c++) {
                const float *xc = xin + (size_t)(c / cb) * HWc + c % cb;
                for (int t = 0; t < KK; t++) {
                    const int ih = ih0 + t / KW, iw = iw0 + t % KW;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        Xd[ap_xoff(r, c * KK + t, kch, Kp)] = uc_f2bf(xc[((size_t)ih * W + iw) * cb]);
                }
            }
        }
        uc_ysentinel(g_ky, ybytes);
        XTime lt0, lt1;
        XTime_GetTime(&lt0);
        const unsigned long long hw0 = uc_hs_wait();
        ahs_add_span(&g_ahs, AHS_PACK, ut0, lt0);
#if UC_WARM
        unsigned seg0 = 0, cold0 = 0, seg1 = 0, cold1 = 0;
        __Runtime_pm_warm_stats(&seg0, &cold0);
#endif
        const unsigned long long sp0 = uc_pmc();
        uc_sync_pre(g_kw, wbytes, g_kx, xbytes, g_ky, ybytes);
        const unsigned long long rt0 = uc_rt_cyc(), sp1 = uc_pmc();
        const int lrc = uc_kacc_launch(Ng, Kg, PX, g_kw, g_kx, g_ky);
        const unsigned long long sp2 = uc_pmc(), rt1 = uc_rt_cyc();
        uc_sync_post(g_ky, ybytes);
        const unsigned long long sp3 = uc_pmc();
#if UC_WARM
        __Runtime_pm_warm_stats(&seg1, &cold1);
        if (cold1 != cold0) {
            g_uc_kloads++;
            g_uc_cold_tiles += cold1 - cold0;
        }
#endif
        const unsigned long long hw1 = uc_hs_wait();
        XTime_GetTime(&lt1);
        ahs_add_window(&g_ahs, (unsigned long long)(lt1 - lt0), hw1 - hw0); /* + site-perf snapshot (SITE_PERF) */
        if (lrc != 0) {
            printf("[aiednn] launch N=%d K=%d PX=%d rc=%d\n", Ng, Kg, PX, lrc);
            return lrc;
        }
        g_sp_pre += sp1 - sp0;
        g_sp_call += sp2 - sp1;
        g_sp_glue += (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL;
        g_sp_post += sp3 - sp2;
        ahs_add(&g_ahs, AHS_GLUE, (sp2 - sp1) > (rt1 - rt0) ? (sp2 - sp1) - (rt1 - rt0) : 0ULL);
        ahs_add(&g_ahs, AHS_FLUSH, sp1 - sp0);
        ahs_add(&g_ahs, AHS_INVAL, sp3 - sp2);
        ahs_launch(&g_ahs);
        ahs_cal(&g_ahs, sp0, (unsigned long long)lt0);
        ahs_cal(&g_ahs, sp3, (unsigned long long)lt1);
        g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
        g_aie_launches++;
        g_uc_macs += (unsigned long long)mv * N * K;
#if AIEDNN_SELFCHECK
        { /* debug build only: the launch's raw GEMM vs an A78 GEMM of the same bf16 operands */
            double dot = 0, ng = 0, nr = 0, mx = 0;
            for (int n = 0; n < N; n++)
                for (int r = 0; r < mv; r++) {
                    const int m = m0 + r, oh = m / OW, ow = m % OW, ih0 = oh * sh - ph, iw0 = ow * sw - pw;
                    float acc = 0.0f;
                    for (int c = 0; c < C; c++) {
                        const float *xc = xin + (size_t)(c / cb) * HWc + c % cb;
                        for (int t = 0; t < KK; t++) {
                            const int ih = ih0 + t / KW, iw = iw0 + t % KW;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                acc += uc_bf2f(uc_f2bf(xc[((size_t)ih * W + iw) * cb])) *
                                       uc_bf2f(wbf[(size_t)n * K + c * KK + t]);
                        }
                    }
                    const double g = ap_y8(Y, n, r & ~7, Ng, PX, tr)[r & 7], d = g - acc;
                    dot += g * acc;
                    ng += g * g;
                    nr += (double)acc * acc;
                    if (d > mx || -d > mx)
                        mx = d > 0 ? d : -d;
                }
            printf("[aiednn-check] N=%d K=%d M=%d m0=%d inst=%dx%dx%d cos=%.6f maxabs=%g\n", N, K, M, m0, Ng, Kg, PX,
                   (ng > 0 && nr > 0) ? dot / sqrt(ng * nr) : 0.0, mx);
        }
#endif
        for (int n = 0; n < N; n++) {
            float *on = yout + (size_t)(n / yb) * M * yb + n % yb;
            for (int r8 = 0; r8 < mv; r8 += 8) {
                const float *src = ap_y8(Y, n, r8, Ng, PX, tr);
                for (int i = 0; i < 8 && r8 + i < mv; i++) {
                    float v = src[i];
                    if (!(v == v) || v > 1e30f || v < -1e30f)
                        bad++;
                    const size_t o = (size_t)(m0 + r8 + i) * yb;
                    for (int e = 0; e < ne_; e++)
                        v += et[e] == 0 ? ep[e][n] : ep[e][(on - yout) + o];
                    if (act == 1 && v < 0.0f)
                        v = 0.0f;
                    on[o] = v;
                }
            }
        }
        {
            XTime ut3;
            XTime_GetTime(&ut3);
            ahs_add_span(&g_ahs, AHS_SCATTER, lt1, ut3);
        }
    }
    if (bad) {
        printf("[aiednn] %d nonfinite AIE outputs (N=%d K=%d M=%d)\n", bad, N, K, M);
        return -4;
    }
#if AIEDNN_SELFCHECK
    /* debug build only: per-channel sums of the final output (compare with the x86 reference's layer outputs) */
    printf("[aiednn-stat] N=%d M=%d K=%d", N, M, K);
    for (int n = 0; n < N; n++) {
        double s = 0;
        for (int m = 0; m < M; m++)
            s += yout[(size_t)(n / yb) * M * yb + (size_t)m * yb + n % yb];
        printf(" %.6g", s);
    }
    printf("\n");
#endif
    return 0;
}

static int ap_done(const char *who, int cid, int rc) {
    g_uc_calls++;
    if (rc != 0) {
        printf("[%s] layer %d rc=%d\n", who, cid, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

extern "C" int aiednn_conv(int cid, void *args, int nargs, const uint16_t *wbf, int sh, int sw, int ph, int pw,
                           int act, int px, int kg) {
    void *const *argv = (void *const *)args;
    if (nargs < 3)
        return ap_done("aiednn_conv", cid, -2);
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 5 || w->ndim != 6 || y->ndim != 5 || x->shape[0] != 1 || y->shape[0] != 1)
        return ap_done("aiednn_conv", cid, -2);
    const int cb = (int)x->shape[4], C = (int)x->shape[1] * cb, H = (int)x->shape[2], W = (int)x->shape[3];
    const int CI = (int)w->shape[1] * (int)w->shape[4], N = (int)w->shape[0] * (int)w->shape[5];
    const int KH = (int)w->shape[2], KW = (int)w->shape[3];
    const int yb = (int)y->shape[4], OH = (int)y->shape[2], OW = (int)y->shape[3];
    if (CI != C || (int)y->shape[1] * yb != N || OH != (H + 2 * ph - KH) / sh + 1 || OW != (W + 2 * pw - KW) / sw + 1)
        return ap_done("aiednn_conv", cid, -3);
    return ap_done("aiednn_conv", cid,
                   ap_gemm((const float *)x->data, C, H, W, cb, KH, KW, sh, sw, ph, pw, OH, OW, wbf, N, argv, nargs,
                           (float *)y->data, yb, act, px, kg, AHS_KIND_CONV));
}

extern "C" int aiednn_dense(int cid, void *args, int nargs, const uint16_t *wbf, int act, int px, int kg) {
    void *const *argv = (void *const *)args;
    if (nargs < 3)
        return ap_done("aiednn_dense", cid, -2);
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 2 || w->ndim != 3 || y->ndim != 2)
        return ap_done("aiednn_dense", cid, -2);
    const int M = (int)x->shape[0], K = (int)x->shape[1], N = (int)w->shape[0] * (int)w->shape[2];
    if ((int)w->shape[1] != K || (int)y->shape[0] != M || (int)y->shape[1] != N)
        return ap_done("aiednn_dense", cid, -3);
    /* x [M][K] = NCHWc (C = K, H = 1, W = M, cb = K); y [M][N] = NCHWc (OH = 1, OW = M, yb = N) */
    return ap_done("aiednn_dense", cid,
                   ap_gemm((const float *)x->data, K, 1, M, K, 1, 1, 1, 1, 0, 0, 1, M, wbf, N, argv, nargs,
                           (float *)y->data, N, act, px, kg, AHS_KIND_LINEAR));
}
