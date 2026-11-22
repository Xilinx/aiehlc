/* aiednn/conv2d/kacc_chunk_conv.cc -- any NCHWc conv2d (stride / pad / KxK / depthwise) as bf16 GEMMs on ONE
 * gemm_kacc instance per (Np, Kk, MC): the output pixels are cut into chunks of MC (one launch each), the
 * output channels padded to Np, K = C*KH*KW padded to Kp (zero rows / columns). fp32 accumulation in the
 * core; the fused epilogue of the TVM call runs here on the A78 in argument order:
 *   z = acc (+ per-channel adds) ; act 1: z = z * sigmoid(z) (SiLU, as TVM's sigmoid + multiply) ;
 *   (+ same-shape adds, the residual after the activation).
 * Depthwise (dw = 1, weight [C/cb][1][KH][KW][1][cb]): the weight is expanded block-diagonally to a dense
 * [N][C*KH*KW] (zeros add exactly 0 to the fp32 accumulators).
 * split = 3 (fp32-accurate): K is laid out as 3 segments of ks, W' = [Wh | Wh | Wl], X' = [Xh | Xl | Xh] with
 * h = bf16(v), l = bf16(v - h): one fp32-accumulated GEMM sums Wh.Xh + Wh.Xl + Wl.Xh (the Wl.Xl term, ~2^-16
 * relative, is dropped). split = 1: one segment, W and X rounded to bf16.
 * Ports (gemm_kacc, non-transposed walk, see aiegraph.cnn_chunk.kacc_geom): W [Np][Kp] bf16 and X [MC][Kp] bf16,
 * each 16-row block pre-tiled per K chunk of kch bf16; Y [Np][MC] fp32 in [s][mi][ni][8x8] core tiles.
 * Needs (before this text): uc_f2bf, uc_dlt, uc_numel, uc_kalloc + g_kw / g_kx / g_ky (UC_K*_ELEMS) and
 * uc_launch(N, K, M, first) and uc_hs_mark(phase) / AHS_SCATTER (per-site host split) -- the board glue. */
typedef struct {
    const float *p;
    int full; /* 0: per output channel, 1: the output's shape */
} uc_ep;

static inline float uc_bfv(float v) { /* bf16 RNE value */
    const uint16_t h = uc_f2bf(v);
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
/* segment s of a split value: 0 = hi, 1 = lo (v - hi) */
static inline uint16_t uc_seg(float v, int lo) { return uc_f2bf(lo ? v - uc_bfv(v) : v); }

static int uc_chunk_conv(void *const *argv, int nargs, int sh, int sw, int ph, int pw, int act, int dw, int Np,
                         int Kk, int Kp, int MC, int kch, int split, int ks) {
    if (nargs < 3)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[nargs - 1];
    if (x->ndim != 5 || w->ndim != 6 || y->ndim != 5 || x->shape[0] != 1 || y->shape[0] != 1)
        return -2;
    const int cb = (int)x->shape[4], C = (int)x->shape[1] * cb, H = (int)x->shape[2], W = (int)x->shape[3];
    const int ib = (int)w->shape[4], ob = (int)w->shape[5], KH = (int)w->shape[2], KW = (int)w->shape[3];
    const int O = (int)w->shape[0] * ob, yb = (int)y->shape[4], OH = (int)y->shape[2], OW = (int)y->shape[3];
    const int CI = dw ? C : (int)w->shape[1] * ib;
    if (dw && (w->shape[1] != 1 || ib != 1 || O != C))
        return -3;
    if (CI != C || (int)y->shape[1] * yb != O || OH != (H + 2 * ph - KH) / sh + 1 || OW != (W + 2 * pw - KW) / sw + 1)
        return -3;
    const int M = OH * OW, N = O, KK = KH * KW, K = C * KK;
    if (N > Np || K > ks || ks * split != Kk || (split != 1 && split != 3) || Kk > Kp || Np % 64 || Kk % 64 ||
        MC % 64 || Kp % kch || kch % 8)
        return -3;
    if ((long)Np * Kp > UC_KW_ELEMS || (long)MC * Kp > UC_KX_ELEMS || (long)Np * MC > UC_KY_ELEMS)
        return -3;
    uc_ep ep[4];
    int ne = 0;
    for (int i = 2; i < nargs - 1; i++) {
        const uc_dlt *e = (const uc_dlt *)argv[i];
        const long n = uc_numel(e);
        if (ne == 4 || e->ndim != 5)
            return -5;
        if (n == N)
            ep[ne].full = 0;
        else if (n == (long)M * N && (int)e->shape[4] == yb)
            ep[ne].full = 1;
        else
            return -5;
        ep[ne++].p = (const float *)e->data;
    }
    if (uc_kalloc() != 0)
        return -1;
    const float *xin = (const float *)x->data, *wt = (const float *)w->data;
    float *yout = (float *)y->data;
    uint16_t *Wd = (uint16_t *)g_kw, *Xd = (uint16_t *)g_kx;
    /* W port: row n, column k (k = c*KK + t); dense, or block-diagonal for depthwise */
    for (int n = 0; n < Np; n++) {
        const int r = n & 15;
        uint16_t *blk = Wd + (size_t)(n & ~15) * Kp;
        for (int k = 0; k < Kp; k++) {
            float v = 0.0f;
            const int sg = k < Kk ? k / ks : 0, kk = k < Kk ? k % ks : K;
            if (n < N && kk < K) {
                const int c = kk / KK, t = kk % KK;
                if (dw)
                    v = c == n ? wt[((size_t)(n / ob) * KK + t) * ob + n % ob] : 0.0f;
                else
                    v = wt[((((size_t)(n / ob) * (C / ib) + c / ib) * KK + t) * ib + c % ib) * ob + n % ob];
            }
            const int j = k / kch, kr = k % kch;
            const int f = (r >> 3) * 8 * kch + (kr >> 3) * 64 + (r & 7) * 8 + (kr & 7);
            blk[(size_t)(f / kch) * Kp + j * kch + f % kch] = uc_seg(v, sg == 2);
        }
    }
    const int q4 = MC / 4; /* pixels per mesh column (core c block {16, MC/4}) */
    int bad = 0;
    for (int m0 = 0; m0 < M; m0 += MC) {
        /* X port: row q (pixel m0 + q), column k; im2col of NCHWc fp32 -> bf16 RNE, zero outside */
        for (int q = 0; q < MC; q++) {
            const int p = m0 + q, r = q & 15, oh = p / OW, ow = p % OW;
            uint16_t *blk = Xd + (size_t)(q & ~15) * Kp;
            for (int k = 0; k < Kp; k++) {
                float v = 0.0f;
                const int sg = k < Kk ? k / ks : 0, kk = k < Kk ? k % ks : K;
                if (p < M && kk < K) {
                    const int c = kk / KK, t = kk % KK, ih = oh * sh - ph + t / KW, iw = ow * sw - pw + t % KW;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        v = xin[(((size_t)(c / cb) * H + ih) * W + iw) * cb + c % cb];
                }
                const int j = k / kch, kr = k % kch;
                const int f = (r >> 3) * 8 * kch + (kr >> 3) * 64 + (kr & 7) * 8 + (r & 7);
                blk[(size_t)(f / kch) * Kp + j * kch + f % kch] = uc_seg(v, sg == 1);
            }
        }
        const int mv = M - m0 < MC ? M - m0 : MC;
        const int lrc = uc_launch(Np, Kk, MC, m0 == 0, (unsigned long long)mv * N * (dw ? KK : K));
        if (lrc != 0)
            return lrc;
        const float *Y = (const float *)g_ky;
#if UC_SELFCHECK
        { /* board self-check: sampled outputs of this launch vs an A78 dot product of the same bf16 ports */
            double worst = 0.0;
            const int so = N > 8 ? N / 8 : 1, sq = mv > 8 ? mv / 8 : 1;
            for (int o = 0; o < N; o += so)
                for (int q = 0; q < mv; q += sq) {
                    double ref = 0.0, mag = 0.0;
                    for (int k = 0; k < Kp; k++) {
                        const int j = k / kch, kr = k % kch, r = o & 15, rq = q & 15;
                        const int fw = (r >> 3) * 8 * kch + (kr >> 3) * 64 + (r & 7) * 8 + (kr & 7);
                        const int fx = (rq >> 3) * 8 * kch + (kr >> 3) * 64 + (kr & 7) * 8 + (rq & 7);
                        const uint16_t wb = Wd[(size_t)(o & ~15) * Kp + (size_t)(fw / kch) * Kp + j * kch + fw % kch];
                        const uint16_t xb = Xd[(size_t)(q & ~15) * Kp + (size_t)(fx / kch) * Kp + j * kch + fx % kch];
                        uint32_t uw = (uint32_t)wb << 16, ux = (uint32_t)xb << 16;
                        float fwv, fxv;
                        memcpy(&fwv, &uw, 4);
                        memcpy(&fxv, &ux, 4);
                        ref += (double)fwv * fxv;
                        mag += fabs((double)fwv * fxv);
                    }
                    const int m = o & 15, pn = q / q4, qq = q % q4, s = qq >> 4, n = qq & 15;
                    const int f = s * 256 + ((m >> 3) * 2 + (n >> 3)) * 64 + (m & 7) * 8 + (n & 7);
                    const double got = Y[(size_t)(o & ~15) * MC + (size_t)(f / q4) * MC + pn * q4 + f % q4];
                    const double e = fabs(got - ref) / (mag + 1e-20);
                    if (e > worst)
                        worst = e;
                }
            if (worst > g_chk_worst)
                g_chk_worst = worst;
            if (m0 == 0 || worst > 1e-3)
                printf("[CHK] N=%d K=%d inst=%dx%dx%d m0=%d worst_rel=%.3g%s\n", N, K, Np, Kk, MC, m0, worst,
                       worst > 1e-3 ? " BAD" : "");
        }
#endif
        for (int o = 0; o < N; o++) {
            const int m = o & 15;
            const float *yr = Y + (size_t)(o & ~15) * MC;
            float *on = yout + (size_t)(o / yb) * M * yb + o % yb;
            float bsum = 0.0f;
            for (int e = 0; e < ne; e++)
                if (!ep[e].full)
                    bsum += ep[e].p[o];
            for (int q = 0; q < mv; q++) {
                const int pn = q / q4, qq = q % q4, s = qq >> 4, n = qq & 15;
                const int f = s * 256 + ((m >> 3) * 2 + (n >> 3)) * 64 + (m & 7) * 8 + (n & 7);
                float v = yr[(size_t)(f / q4) * MC + pn * q4 + f % q4];
                if (!(v == v) || v > 1e30f || v < -1e30f)
                    bad++;
                v += bsum;
                if (act == 1)
                    v = v * (1.0f / (1.0f + expf(0.0f - v)));
                const size_t oi = (size_t)(m0 + q) * yb;
                for (int e = 0; e < ne; e++)
                    if (ep[e].full)
                        v = ep[e].p[(on - yout) + oi] + v;
                on[oi] = v;
            }
        }
        uc_hs_mark(AHS_SCATTER); /* this chunk's scatter + epilogue (after the launch window) */
    }
    if (bad) {
        printf("[aie_conv] %d nonfinite AIE outputs\n", bad);
        return -4;
    }
    return 0;
}
