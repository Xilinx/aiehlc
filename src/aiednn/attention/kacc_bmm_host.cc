
/* GATT (--aie conv,linear,attn; GAP-aiegraph-attn-site): one TVM nn_batch_matmul call site (an attention QK^T or
 * PV bmm of an aiegraph.partition.match_attention group) on the AIE. TVM batch_matmul is transpose_b:
 * Y[h][M][N] = X[h][M][K] . W[h][N][K]^T, BOTH operands activations -> packed at run time from their natural f32
 * layout by aie_kacc_pack.h (akp_pack, bf16 RNE, zero pad to the kernel geometry; PBK, bmk fd4cb96), H heads x S
 * launches of ONE gemm_kacc instance (unet_aie.bmm_geom = kacc_gemm.main_bmm rules), output un-tiled by akp_unpad
 * (one bf16 SRS) and widened to the f32 TVM tensor. PKC recipe kc_bf16out_pad64: for QK^T the keys (W rows) are
 * centred over the REAL keys per head (softmax is shift-invariant per query row; pad rows stay 0). Softmax stays
 * on the CPU (TVM). */
#include "aie_kacc_pack.h"
#ifndef UC_HS_DEFINED /* GAP-RT1: standalone (x86 harness) -> local split table, no runtime wait counter */
#define UC_HS_DEFINED 1
#include "aie_host_split.h"
static ahs_table g_ahs;
static inline unsigned long long uc_hs_wait(void) { return 0ULL; }
#endif
static float *g_ua_kc = nullptr;    /* one head's centred keys [N][K] (QK^T) */
static uint16_t *g_ua_yb = nullptr; /* one head's bf16 output [M][N] */
static size_t g_ua_kcap = 0, g_ua_ycap = 0;
static unsigned g_ua_sites = 0;
static int uc_attn(void *const *argv, int nargs, int qk, int swap, int CHP, int PXP, int Kp, int kp, int kch,
                   int splitpx, int S, int chs, int pxs, int tr) {
    ahs_begin(&g_ahs, AHS_KIND_ATTN);
    if (nargs != 3)
        return -2;
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[2];
    if (x->ndim != 3 || w->ndim != 3 || y->ndim != 3)
        return -2;
    const int H = (int)x->shape[0], M = (int)x->shape[1], K = (int)x->shape[2], N = (int)w->shape[1];
    if ((int)w->shape[0] != H || (int)w->shape[2] != K || (int)y->shape[0] != H || (int)y->shape[1] != M ||
        (int)y->shape[2] != N)
        return -3;
    const int CHR = swap ? M : N, PXR = swap ? N : M;
    if (CHR > CHP || PXR > PXP || K > Kp || Kp > kp || S < 1 || (CHP | PXP) & 63 ||
        (splitpx ? (PXP != S * pxs || CHP != chs) : (CHP != S * chs || PXP != pxs)))
        return -3;
    const size_t wel = (size_t)CHP * kp, xel = (size_t)PXP * kp, yel = (size_t)CHP * PXP;
    if (wel > (size_t)UC_KW_ELEMS || xel > (size_t)UC_KX_ELEMS || yel > (size_t)UC_KY_ELEMS)
        return -3;
    const float *X = (const float *)x->data, *Wt = (const float *)w->data;
    float *Y = (float *)y->data;
    if (uc_kalloc() != 0)
        return -1;
    if ((size_t)M * N > g_ua_ycap) {
        free(g_ua_yb);
        g_ua_ycap = (size_t)M * N;
        g_ua_yb = (uint16_t *)aligned_alloc(64, (g_ua_ycap * 2 + 63) & ~(size_t)63);
    }
    if (qk && (size_t)N * K > g_ua_kcap) {
        free(g_ua_kc);
        g_ua_kcap = (size_t)N * K;
        g_ua_kc = (float *)aligned_alloc(64, (g_ua_kcap * 4 + 63) & ~(size_t)63);
    }
    if (!g_ua_yb || (qk && !g_ua_kc))
        return -1;
#if UC_CTRLPKT
    if (!g_uc_ht) { // once, before the first launch
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    int32_t *const kw = g_kw, *const kx = g_kx, *const ky = g_ky;
    const size_t wbytes = wel * 2, xbytes = xel * 2, ybytes = yel * 4;
    for (int h = 0; h < H; h++) {
        XTime ht0, ht1, ht2, ht3; /* GAP-RT1: head split pack [ht0, ht1) | launches | scatter [ht2, ht3) */
        XTime_GetTime(&ht0);
        const float *xh = X + (size_t)h * M * K, *wh = Wt + (size_t)h * N * K;
        if (qk) { /* PKC kc_bf16out_pad64: keys centred over the N real keys (fp64 mean per column) */
            for (int k = 0; k < K; k++) {
                double mu = 0.0;
                for (int n = 0; n < N; n++)
                    mu += wh[(size_t)n * K + k];
                const float m = (float)(mu / N);
                for (int n = 0; n < N; n++)
                    g_ua_kc[(size_t)n * K + k] = wh[(size_t)n * K + k] - m;
            }
            wh = g_ua_kc;
        }
        /* W side = instance channels (rows of W, or of X when swapped), X side = pixels; both [rows][K], k stride 1 */
        akp_pack((uint16_t *)kw, swap ? xh : wh, CHP, CHR, kp, K, kch, K, 1, 1, 0);
        akp_pack((uint16_t *)kx, swap ? wh : xh, PXP, PXR, kp, K, kch, K, 1, 0, 0);
        XTime_GetTime(&ht1);
        ahs_add_span(&g_ahs, AHS_PACK, ht0, ht1);
        for (int s = 0; s < S; s++) {
            XTime lt0, lt1;
            XTime_GetTime(&lt0);
            const unsigned long long hw0 = uc_hs_wait();
            if (s == 0)
                uc_sync_pre(kw, wbytes, kx, xbytes, ky, ybytes);
            int32_t *Ws = kw + (splitpx ? 0 : (size_t)s * chs * kp / 2);
            int32_t *Xs = kx + (splitpx ? (size_t)s * pxs * kp / 2 : 0);
            int32_t *Ys = ky + (size_t)s * chs * pxs;
            const int lrc = uc_kacc_launch(chs, Kp, pxs, Ws, Xs, Ys);
            if (s == S - 1)
                uc_sync_post(ky, ybytes);
            const unsigned long long hw1 = uc_hs_wait();
            XTime_GetTime(&lt1);
            ahs_add_window(&g_ahs, (unsigned long long)(lt1 - lt0), hw1 - hw0);
            g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
            g_aie_launches++;
            if (lrc != 0)
                return lrc;
        }
        XTime_GetTime(&ht2);
        for (int s = 0; s < S; s++) { /* launch s: Y block -> yb [M][N] (bf16) */
            const int p0 = splitpx ? s * pxs : 0, c0 = splitpx ? 0 : s * chs;
            const int pr = PXR - p0 < pxs ? PXR - p0 : pxs, cr = CHR - c0 < chs ? CHR - c0 : chs;
            if (pr <= 0 || cr <= 0)
                continue;
            const size_t off = swap ? (size_t)c0 * N + p0 : (size_t)p0 * N + c0;
            akp_unpad(g_ua_yb + off, N, (const float *)(ky + (size_t)s * chs * pxs), pxs, chs, tr, pr, cr, swap);
        }
        float *yh = Y + (size_t)h * M * N;
        const uint16_t *yb = g_ua_yb;
        for (size_t i = 0; i < (size_t)M * N; i++) {
            const uint32_t u = (uint32_t)yb[i] << 16;
            memcpy(yh + i, &u, 4);
        }
        XTime_GetTime(&ht3);
        ahs_add_span(&g_ahs, AHS_SCATTER, ht2, ht3);
    }
    g_uc_macs += (unsigned long long)H * M * N * K;
    g_ua_sites++;
    return 0;
}

extern "C" int aiednn_aie_attn(int cid, void *args, int nargs, int qk, int swap, int CHP, int PXP, int Kp, int kp,
                             int kch, int splitpx, int S, int chs, int pxs, int tr) {
    g_uc_calls++;
    const int rc = uc_attn((void *const *)args, nargs, qk, swap, CHP, PXP, Kp, kp, kch, splitpx, S, chs, pxs, tr);
    if (rc != 0) {
        printf("[aiednn_aie_attn] layer %d rc=%d\n", cid, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

