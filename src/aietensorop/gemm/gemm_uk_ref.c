/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * gemm_blk_s8s32_ref -- x86 / CPU reference for gemm_uk.cc:
 * dst[m][n] = sum_k src[m][k] * weight[n][k], int32, no saturation.
 ******************************************************************************/
#include "aie_user_kernel.h"
#include <stdlib.h>
#include <string.h>
#include "aie_modes_pool.h" /* src/aietensorop/ops */
#include "aie_modes_lut.h"
#include "aie_modes_mmt.h"
#include "aie_modes_conv.h"
#include "aie_modes_sdpa.h"
#include "aie_modes_bf16.h"
#include "aie_modes_wq.h"
#include "aie_modes_fpool.h"
#include "aie_modes_fvec.h"
#ifndef HW_ROWS /* the mesh (gemm.h on the board; -DHW_ROWS/-DHW_COLS from the x86 builds' mesh) */
#define HW_ROWS 4
#endif
#ifndef HW_COLS
#define HW_COLS 4
#endif

AIEK_REFERENCE(gemm_blk_s8s32) {
    if (num_args != 3 || args[0].ndim != 2 || args[1].ndim != 2 || args[2].ndim != 2)
        return AIEK_ERR_ARGS;
    const int64_t m = args[0].shape[0], k = args[0].shape[1], n = args[1].shape[0];
    if (args[1].shape[1] != k || args[2].shape[0] != m || args[2].shape[1] != n)
        return AIEK_ERR_ARGS;
    const int8_t *a = AIEK_PTR(&args[0], const int8_t);
    const int8_t *b = AIEK_PTR(&args[1], const int8_t);
    int32_t *c = AIEK_PTR(&args[2], int32_t);
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int64_t q = 0; q < k; q++)
                acc += (int32_t)a[i * k + q] * (int32_t)b[j * k + q];
            c[i * n + j] = acc;
        }
    return AIEK_OK;
}

/* dw_blk_s8s32_ref -- the depthwise overlay of gemm_uk.cc: item i (4096 bytes:
 * zero-padded input rows, pitch Wp, weights in the last 32 bytes) -> int32
 * out[i][0 .. Rout*Wo), row-major, the rest of the 1024 zero.
 * geom = {stride, KH, KW, Wp, Rout, Wo, 0, 0}. */
AIEK_REFERENCE(dw_blk_s8s32) {
    if (num_args != 3 || args[0].ndim != 2 || args[2].ndim != 2)
        return AIEK_ERR_ARGS;
    const int64_t items = args[0].shape[0], bytes = args[0].shape[1], outs = args[2].shape[1];
    const int8_t *x = AIEK_PTR(&args[0], const int8_t);
    const int32_t *g = AIEK_PTR(&args[1], const int32_t);
    int32_t *y = AIEK_PTR(&args[2], int32_t);
    const int s = g[0], kh_n = g[1], kw_n = g[2], wp = g[3], rout = g[4], wo = g[5], q8 = g[6];
    if (rout * wo > outs || kh_n * kw_n > 64 || ((rout - 1) * s + kh_n) * wp > bytes - (q8 ? 96 : 64))
        return AIEK_ERR_PARAM;
    for (int64_t i = 0; i < items; i++) {
        const int8_t *raw = x + i * bytes, *wt = raw + bytes - 64;
        for (int64_t o = 0; o < outs; o++) {
            int32_t a = 0;
            if (o < rout * wo) {
                const int oh = (int)(o / wo), ow = (int)(o % wo);
                const int8_t *in = raw + oh * s * wp + ow * s;
                for (int kh = 0; kh < kh_n; kh++)
                    for (int kw = 0; kw < kw_n; kw++)
                        a += (int32_t)wt[kh * kw_n + kw] * (int32_t)in[kh * wp + kw];
            }
            if (q8) { /* geom[6]: requant like the core's 0xD8 mode, int8 item i at bytes [i*outs, +outs) */
                const int32_t *qp = (const int32_t *)(raw + bytes - 96);
                int64_t B;
                memcpy(&B, qp + 4, 8);
                int64_t q = (((int64_t)a * qp[3] + B) >> qp[6]) + qp[0];
                ((int8_t *)y)[i * outs + o] = (int8_t)(o < rout * wo ? (q < qp[1] ? qp[1] : q > qp[2] ? qp[2] : q) : 0);
                continue;
            }
            y[i * outs + o] = a;
        }
    }
    return AIEK_OK;
}

/* dwq_blk_s8s8_ref -- gemm_uk.cc's 3-items-per-core depthwise + requant (dwq_blk_run): every item
 * as dw_blk_s8s32 with geom[6] set, int8 [items][outs] */
AIEK_REFERENCE(dwq_blk_s8s8) {
    if (num_args != 3 || args[0].ndim != 2 || args[2].ndim != 2)
        return AIEK_ERR_ARGS;
    const int64_t items = args[0].shape[0], bytes = args[0].shape[1], outs = args[2].shape[1];
    const int8_t *x = AIEK_PTR(&args[0], const int8_t);
    const int32_t *g = AIEK_PTR(&args[1], const int32_t);
    int8_t *y = AIEK_PTR(&args[2], int8_t);
    const int s = g[0], kh_n = g[1], kw_n = g[2], wp = g[3], rout = g[4], wo = g[5];
    if (rout * wo > outs || kh_n * kw_n > 64 || ((rout - 1) * s + kh_n) * wp > bytes - 96)
        return AIEK_ERR_PARAM;
    for (int64_t i = 0; i < items; i++) {
        const int8_t *raw = x + i * bytes, *wt = raw + bytes - 64;
        const int32_t *qp = (const int32_t *)(raw + bytes - 96);
        int64_t B;
        memcpy(&B, qp + 4, 8);
        for (int64_t o = 0; o < outs; o++) {
            if (o >= rout * wo) {
                y[i * outs + o] = 0;
                continue;
            }
            const int oh = (int)(o / wo), ow = (int)(o % wo);
            const int8_t *in = raw + oh * s * wp + ow * s;
            int32_t a = 0;
            for (int kh = 0; kh < kh_n; kh++)
                for (int kw = 0; kw < kw_n; kw++)
                    a += (int32_t)wt[kh * kw_n + kw] * (int32_t)in[kh * wp + kw];
            int64_t q = (((int64_t)a * qp[3] + B) >> qp[6]) + qp[0];
            y[i * outs + o] = (int8_t)(q < qp[1] ? qp[1] : q > qp[2] ? qp[2] : q);
        }
    }
    return AIEK_OK;
}

/* bf16_blk_f32_ref -- the bf16 mode of gemm_uk.cc: per core (r, c) the f32 tile accumulated in K order
 * (aie_bf16_ref_tile), kept across launches for the continue flag; C written unless more K follows */
AIEK_REFERENCE(bf16_blk_f32) {
    static float tile[HW_ROWS * 32 * HW_COLS * 32];
    if (num_args != 4 || args[0].ndim != 2 || args[3].ndim != 2 || args[0].shape[1] != 512 ||
        args[0].shape[0] != 32 * HW_ROWS || args[3].shape[1] != 32 * HW_COLS)
        return AIEK_ERR_ARGS;
    const uint16_t *a = AIEK_PTR(&args[0], const uint16_t), *w = AIEK_PTR(&args[1], const uint16_t);
    const int32_t flags = *AIEK_PTR(&args[2], const int32_t);
    float *c = AIEK_PTR(&args[3], float);
    for (int r = 0; r < HW_ROWS; r++)
        for (int j = 0; j < HW_COLS; j++) {
            float *t = tile + (r * HW_COLS + j) * 1024;
            aie_bf16_ref_tile(a + r * 32 * 256, 256, w + j * 32 * 256, 256, 256, t, flags & 1);
            if (!(flags & 2))
                for (int m = 0; m < 32; m++)
                    for (int n = 0; n < 32; n++)
                        c[(r * 32 + m) * 32 * HW_COLS + j * 32 + n] = t[m * 32 + n];
        }
    return AIEK_OK;
}

/* fpool_blk_f32_ref -- the FPOOL overlay of gemm_uk.cc: aie_fpool_item_ref per item */
AIEK_REFERENCE(fpool_blk_f32) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int32_t *g = AIEK_PTR(&args[1], const int32_t);
    uint8_t d[12];
    AIE_FPOOL_DESC(d, g);
    for (int64_t i = 0; i < args[0].shape[0]; i++)
        aie_fpool_item_ref((const float *)(AIEK_PTR(&args[0], const int8_t) + i * 4096),
                           AIEK_PTR(&args[2], float) + i * 1024, d);
    return AIEK_OK;
}

/* wq_blk_s8f32_ref -- the WQ mode of gemm_uk.cc: per core (r, c) aie_wq_ref_tile on its 32 rows / 32
 * channels; the int32 partial and the f32 tile persist across launches for the continue flag */
AIEK_REFERENCE(wq_blk_s8f32) {
    static int32_t part[HW_ROWS * HW_COLS * 1024];
    static float tile[HW_ROWS * HW_COLS * 1024];
    static float s32[8 * 32];
    if (num_args != 5 || args[0].ndim != 2 || args[4].ndim != 2 || args[0].shape[1] != 512 ||
        args[0].shape[0] != 32 * HW_ROWS || args[4].shape[1] != 32 * HW_COLS)
        return AIEK_ERR_ARGS;
    const int8_t *a = AIEK_PTR(&args[0], const int8_t), *w = AIEK_PTR(&args[1], const int8_t);
    const float *sc = AIEK_PTR(&args[2], const float);
    const int32_t *ctl = AIEK_PTR(&args[3], const int32_t);
    float *c = AIEK_PTR(&args[4], float);
    const int gr = ctl[1] < 1 ? 1 : ctl[1], ph = ctl[2] % gr, flags = ctl[0];
    for (int r = 0; r < HW_ROWS; r++)
        for (int j = 0; j < HW_COLS; j++) {
            float *t = tile + (r * HW_COLS + j) * 1024;
            for (int kr = 0; kr < 8; kr++)
                for (int n = 0; n < 32; n++)
                    s32[kr * 32 + n] = sc[kr * 32 * HW_COLS + j * 32 + n];
            aie_wq_ref_tile(a + r * 32 * 512, 512, w + j * 32 * 512, 512, s32, gr, ph, flags,
                            part + (r * HW_COLS + j) * 1024, t);
            if (!(flags & 2))
                for (int m = 0; m < 32; m++)
                    for (int n = 0; n < 32; n++)
                        c[(r * 32 + m) * 32 * HW_COLS + j * 32 + n] = t[m * 32 + n];
        }
    return AIEK_OK;
}

/* fvec_blk_f32_ref -- the FVEC overlay of gemm_uk.cc: aie_modes_fvec.h's plain C reference per item */
AIEK_REFERENCE(fvec_blk_f32) {
    if (num_args != 5 || args[0].ndim != 2 || args[0].shape[1] != AIE_FVEC_N || args[1].shape[1] != AIE_FVEC_AUX)
        return AIEK_ERR_ARGS;
    const float *ab = AIEK_PTR(&args[3], const float);
    for (int64_t k = 0; k < args[0].shape[0]; k++) {
        const int32_t *g = AIEK_PTR(&args[2], const int32_t) + 4 * k;
        uint8_t d[16];
        if (!g[0])
            continue;
        aie_fvec_desc(d, g, ab);
        aie_fvec_apply(AIEK_PTR(&args[0], const float) + k * AIE_FVEC_N, AIEK_PTR(&args[4], float) + k * AIE_FVEC_N, d,
                       AIEK_PTR(&args[1], const float) + k * AIE_FVEC_AUX);
    }
    return AIEK_OK;
}

/* sdpa_blk_s8s32_ref -- the SDPA overlay of gemm_uk.cc: the same aie_sdpa_item the core runs */
AIEK_REFERENCE(sdpa_blk_s8s32) {
    if (num_args != 3 || args[0].ndim != 2 || args[2].ndim != 2 || args[0].shape[1] != 4096 || args[2].shape[1] != 1024)
        return AIEK_ERR_ARGS;
    const int32_t *g = AIEK_PTR(&args[1], const int32_t);
    uint8_t d[16];
    int32_t s[AIE_SDPA_MAX_KV];
    AIE_MODE_SDPA_DESC(d, g);
    if (g[1] > AIE_SDPA_MAX_KV)
        return AIEK_ERR_PARAM;
    for (int64_t i = 0; i < args[0].shape[0]; i++)
        aie_sdpa_item(AIEK_PTR(&args[0], const int8_t) + i * 4096, AIEK_PTR(&args[2], int32_t) + i * 1024, d, s);
    return AIEK_OK;
}

/* pool_blk_s8s32_ref -- the pool overlay of gemm_uk.cc: the same AIE_MODE_POOL the core runs */
AIEK_REFERENCE(pool_blk_s8s32) {
    if (num_args != 3 || args[0].ndim != 2 || args[2].ndim != 2)
        return AIEK_ERR_ARGS;
    const int64_t items = args[0].shape[0], bytes = args[0].shape[1], outs = args[2].shape[1];
    if (bytes != AIE_MODE_IN || outs != AIE_MODE_OUT)
        return AIEK_ERR_ARGS;
    uint8_t d[16] = {0};
    const int32_t *g = AIEK_PTR(&args[1], const int32_t);
    AIE_MODE_POOL_DESC(d, g);
    for (int64_t i = 0; i < items; i++) {
        const int8_t *in = AIEK_PTR(&args[0], const int8_t) + i * bytes;
        int32_t *out = AIEK_PTR(&args[2], int32_t) + i * outs;
        AIE_MODE_POOL(in, out, d);
    }
    return AIEK_OK;
}

/* mmt_blk_s8s8_ref -- the MMT overlay of gemm_uk.cc: the same per-round arithmetic as the core
 * (aie_modes_mmt.h), core (r, c) = mesh row r's A x column c's B */
AIEK_REFERENCE(mmt_blk_s8s8) {
    if (num_args != 4 || args[0].ndim != 2 || args[0].shape[1] != 16384 || args[1].shape[1] != AIE_MMT_RES)
        return AIEK_ERR_ARGS;
    const int64_t rows = args[0].shape[0], cols = args[1].shape[0];
    const int kcode = *AIEK_PTR(&args[2], const int32_t);
    if (kcode < 1 || kcode > 2 || args[3].shape[0] != rows * cols)
        return AIEK_ERR_ARGS;
    for (int64_t r = 0; r < rows; r++)
        for (int64_t c = 0; c < cols; c++) {
            int8_t *out = AIEK_PTR(&args[3], int8_t) + (r * cols + c) * 4096;
            memset(out, 0, 4096);
            for (int kr = 0; kr < 8; kr++)
                aie_mmt_round(AIEK_PTR(&args[0], const int8_t) + r * 16384 + kr * 2048,
                              AIEK_PTR(&args[1], const int8_t) + c * AIE_MMT_RES, kcode, kr, NULL, out);
        }
    return AIEK_OK;
}

/* elt_blk_s8s32_ref -- the eltwise overlay of gemm_uk.cc: the same AIE_MODE_ELTWISE the core runs */
AIEK_REFERENCE(elt_blk_s8s32) {
    if (num_args != 3 || args[0].ndim != 2 || args[0].shape[1] != AIE_MODE_IN || args[2].shape[1] != AIE_MODE_OUT)
        return AIEK_ERR_ARGS;
    uint8_t d[4];
    AIE_MODE_ELTWISE_DESC(d, AIEK_PTR(&args[1], const int32_t));
    for (int64_t i = 0; i < args[0].shape[0]; i++)
        AIE_MODE_ELTWISE(AIEK_PTR(&args[0], const int8_t) + i * AIE_MODE_IN,
                         AIEK_PTR(&args[2], int32_t) + i * AIE_MODE_OUT, d);
    return AIEK_OK;
}

/* lut_blk_s8_ref -- the LUT overlay of gemm_uk.cc: the same aie_lut_apply the core runs */
AIEK_REFERENCE(lut_blk_s8) {
    if (num_args != 4 || args[0].ndim != 2 || args[0].shape[1] != AIE_LUT_ITEM || args[1].shape[1] != AIE_LUT_TBYTES)
        return AIEK_ERR_ARGS;
    for (int64_t k = 0; k < args[0].shape[0]; k++) {
        const int32_t *g = AIEK_PTR(&args[2], const int32_t) + 4 * k;
        uint8_t d[11];
        aie_lut_desc(d, g);
        const uint8_t *in = AIEK_PTR(&args[0], const uint8_t) + k * AIE_LUT_ITEM;
        uint8_t *out = AIEK_PTR(&args[3], uint8_t) + k * AIE_LUT_ITEM;
        const uint8_t *tab = AIEK_PTR(&args[1], const uint8_t) + k * AIE_LUT_TBYTES;
        if (((uint32_t)g[3] >> 27) & 1)
            aie_steps_apply(in, out, d, tab);
        else
            aie_lut_apply(in, out, d, tab);
    }
    return AIEK_OK;
}

/* the core's accumulators between the launches of a K chain (gemm_kc_s8s32 / gemm_q_s8s8 with
 * the continue flag): what a persistent gemm_uk core keeps in its acc array */
static int32_t *g_kc_acc;
static size_t g_kc_cap;

static int32_t *kc_acc(int64_t m, int64_t n, int cont) {
    const size_t need = (size_t)(m * n);
    if (need > g_kc_cap) {
        int32_t *grown = (int32_t *)realloc(g_kc_acc, need * sizeof(int32_t));
        if (!grown)
            return NULL;
        g_kc_acc = grown;
        g_kc_cap = need;
    }
    if (!cont)
        memset(g_kc_acc, 0, need * sizeof(int32_t));
    return g_kc_acc;
}

/* gemm_kc_s8s32_ref -- one K block of a chained GEMM: acc (+)= A * B^T; flags bit 0 continue,
 * bit 1 more blocks follow (C untouched), else C = acc */
AIEK_REFERENCE(gemm_kc_s8s32) {
    if (num_args != 4 || args[0].ndim != 2 || args[1].ndim != 2 || args[3].ndim != 2)
        return AIEK_ERR_ARGS;
    const int64_t m = args[0].shape[0], k = args[0].shape[1], n = args[1].shape[0];
    const int8_t *a = AIEK_PTR(&args[0], const int8_t), *b = AIEK_PTR(&args[1], const int8_t);
    const int32_t flags = *AIEK_PTR(&args[2], const int32_t);
    int32_t *acc = kc_acc(m, n, flags & 1);
    if (!acc)
        return AIEK_ERR_ALLOC;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            int32_t s = 0;
            for (int64_t q = 0; q < k; q++)
                s += (int32_t)a[i * k + q] * (int32_t)b[j * k + q];
            acc[i * n + j] += s;
        }
    if (!(flags & 2))
        memcpy(AIEK_PTR(&args[3], int32_t), acc, (size_t)(m * n) * sizeof(int32_t));
    return AIEK_OK;
}

/* gemm_q_s8s8_ref -- GEMM + the integer requant of gemm_uk.cc's 0xF1 mode (same formula as
 * aiednn_rt_internal.h rt_fx_q): dst[m][n] = clamp((acc * M[n] + B[n]) >> sh[n] + zp), params
 * int32 {zp, lo, hi, flags, B int64[n], M[n], sh[n], has_res, rzp, R int64[n]}; flags bit 0:
 * continue a K chain; has_res: + (res[m][n] - rzp) * R[n] (residual int8 [m][n], args[3]) */
AIEK_REFERENCE(gemm_q_s8s8) {
    if (num_args != 5 || args[0].ndim != 2 || args[1].ndim != 2 || args[3].ndim != 2 || args[4].ndim != 2)
        return AIEK_ERR_ARGS;
    const int64_t m = args[0].shape[0], k = args[0].shape[1], n = args[1].shape[0];
    const int8_t *a = AIEK_PTR(&args[0], const int8_t), *b = AIEK_PTR(&args[1], const int8_t);
    const int32_t *qp = AIEK_PTR(&args[2], const int32_t);
    const int8_t *res = AIEK_PTR(&args[3], const int8_t);
    int8_t *c = AIEK_PTR(&args[4], int8_t);
    const int32_t *rp = qp + 4 + 4 * n; /* has_res, rzp, R int64[n] */
    int32_t *ka = kc_acc(m, n, qp[3] & 1);
    if (!ka)
        return AIEK_ERR_ALLOC;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            int32_t acc = ka[i * n + j];
            for (int64_t q = 0; q < k; q++)
                acc += (int32_t)a[i * k + q] * (int32_t)b[j * k + q];
            int64_t B;
            memcpy(&B, qp + 4 + 2 * j, 8);
            if (rp[0] & 1) {
                int64_t R;
                memcpy(&R, rp + 2 + 2 * j, 8);
                B += ((int64_t)res[i * n + j] - rp[1]) * R;
            }
            int64_t v = (((int64_t)acc * qp[4 + 2 * n + j] + B) >> qp[4 + 3 * n + j]) + qp[0];
            v = v < qp[1] ? qp[1] : v > qp[2] ? qp[2] : v;
            c[i * n + j] = (int8_t)v;
        }
    return AIEK_OK;
}

/* direct conv (aie_modes_conv.h): A is a raw input region per 32-row mesh row, geom int32 [16 *
 * rows] its geometry. The im2col block is gathered with the core's own tables (conv_tables), then
 * the launch is the matching chained / requantized GEMM. */
static int8_t *g_conv_a;
static size_t g_conv_cap;

static const int8_t *conv_unregion(const aiek_tensor *a_t, const int32_t *geom) {
    const int64_t m = a_t->shape[0], k = a_t->shape[1];
    const int8_t *a = AIEK_PTR(a_t, const int8_t);
    if ((size_t)(m * k) > g_conv_cap) {
        int8_t *grown = (int8_t *)realloc(g_conv_a, (size_t)(m * k));
        if (!grown)
            return NULL;
        g_conv_a = grown;
        g_conv_cap = (size_t)(m * k);
    }
    int32_t base[32], toff[512 / 4];
    if (k != 512 || m % 32)
        return NULL;
    for (int64_t r = 0; r < m / 32; r++) {
        const int8_t *reg = a + r * 32 * k;
        conv_tables(geom + 16 * r, base, toff, (int)(k / 4), 32);
        for (int p = 0; p < 32; p++)
            for (int u = 0; u < k / 4; u++) {
                uint32_t v = 0;
                if (base[p] != CONV_NOPX && toff[u] >= 0)
                    memcpy(&v, reg + base[p] + toff[u], 4);
                memcpy(g_conv_a + (r * 32 + p) * k + 4 * u, &v, 4);
            }
    }
    return g_conv_a;
}

AIEK_REFERENCE(conv_kc_s8s32) { /* A, B, flags, geom, C */
    if (num_args != 5 || args[0].ndim != 2)
        return AIEK_ERR_ARGS;
    const int8_t *ga = conv_unregion(&args[0], AIEK_PTR(&args[3], const int32_t));
    if (!ga)
        return AIEK_ERR_ALLOC;
    aiek_tensor t[4] = {args[0], args[1], args[2], args[4]};
    t[0].data = (void *)ga;
    return gemm_kc_s8s32_ref(t, 4);
}

AIEK_REFERENCE(conv_q_s8s8) { /* A, B, qparams, residual, geom, C */
    if (num_args != 6 || args[0].ndim != 2)
        return AIEK_ERR_ARGS;
    const int8_t *ga = conv_unregion(&args[0], AIEK_PTR(&args[4], const int32_t));
    if (!ga)
        return AIEK_ERR_ALLOC;
    aiek_tensor t[5] = {args[0], args[1], args[2], args[3], args[5]};
    t[0].data = (void *)ga;
    return gemm_q_s8s8_ref(t, 5);
}

/* x86 stand-ins for gemm_uk.cc's n-tile launch (aiednn_gemm_ntile_*): the same tiles, K chain and
 * requant as the core, result tiles in the board's layout. */
static int8_t *g_ntx_a, *g_ntx_c;
static int64_t g_ntx_a_bytes, g_ntx_c_bytes, g_ntx_bm, g_ntx_bn, g_ntx_nkb;
static int g_ntx_which;

int aiednn_gemm_ntile_max(void) { return 32; }

int8_t *aiednn_gemm_ntile_stage_a(int64_t tiles, int64_t tile_bytes) {
    if (tiles * tile_bytes > g_ntx_a_bytes) {
        int8_t *grown = (int8_t *)realloc(g_ntx_a, (size_t)(tiles * tile_bytes));
        if (!grown)
            return NULL;
        g_ntx_a = grown, g_ntx_a_bytes = tiles * tile_bytes;
    }
    return g_ntx_a;
}

static int g_ntx_conv; /* aiednn_gemm_ntile_conv: the next start's A tiles are direct-conv regions */
static int8_t *g_ntx_unreg;
static int64_t g_ntx_unreg_bytes;

void aiednn_gemm_ntile_conv(int on) { g_ntx_conv = on; }

int aiednn_gemm_ntile_start(int which, const int8_t *a, int nm, int nkb, const int8_t *b, const int32_t *qp,
                            uint64_t key, int64_t bm, int64_t bn, int64_t bk) {
    (void)key;
    if (g_ntx_conv) { /* the core gathers each tile from its regions: the same rows, decoded here */
        const int64_t tb = bm * bk, nt = (int64_t)nm * nkb;
        if (nt * tb > g_ntx_unreg_bytes) {
            int8_t *grown = (int8_t *)realloc(g_ntx_unreg, (size_t)(nt * tb));
            if (!grown)
                return AIEK_ERR_ALLOC;
            g_ntx_unreg = grown, g_ntx_unreg_bytes = nt * tb;
        }
        for (int64_t t = 0; t < nt; t++) {
            int32_t geom[16 * 32];
            const int64_t shape[2] = {bm, bk};
            if (bm / 32 > 32)
                return AIEK_ERR_ARGS;
            for (int64_t r = 0; r < bm / 32; r++)
                memcpy(geom + 16 * r, a + t * tb + r * 32 * bk + CONV_REGION_DATA, 64);
            aiek_tensor at = {(void *)(a + t * tb), 2, AIEK_INT, 8, 1, shape, (uint64_t)tb};
            const int8_t *u = conv_unregion(&at, geom);
            if (!u)
                return AIEK_ERR_ALLOC;
            memcpy(g_ntx_unreg + t * tb, u, (size_t)tb);
        }
        a = g_ntx_unreg;
    }
    const int64_t tile = 4 * bm * bn; /* the board's C tile: int32 [bm][bn], or int8 rows of 4 * bn bytes */
    if (nm * nkb > 32 || (which && !qp))
        return AIEK_ERR_ARGS;
    if (nm * nkb * tile > g_ntx_c_bytes) {
        int8_t *grown = (int8_t *)realloc(g_ntx_c, (size_t)(nm * nkb * tile));
        if (!grown)
            return AIEK_ERR_ALLOC;
        g_ntx_c = grown, g_ntx_c_bytes = nm * nkb * tile;
    }
    g_ntx_bm = bm, g_ntx_bn = bn, g_ntx_nkb = nkb, g_ntx_which = which;
    for (int mb = 0; mb < nm; mb++) {
        int8_t *out = g_ntx_c + (int64_t)(mb * nkb + nkb - 1) * tile;
        for (int64_t i = 0; i < bm; i++)
            for (int64_t j = 0; j < bn; j++) {
                int32_t acc = 0;
                for (int kb = 0; kb < nkb; kb++) {
                    const int8_t *ar = a + ((int64_t)(mb * nkb + kb) * bm + i) * bk,
                                 *br = b + ((int64_t)kb * bn + j) * bk;
                    for (int64_t q = 0; q < bk; q++)
                        acc += (int32_t)ar[q] * (int32_t)br[q];
                }
                if (!which) {
                    ((int32_t *)out)[i * bn + j] = acc;
                    continue;
                }
                int64_t B;
                memcpy(&B, qp + 4 + 2 * j, 8);
                int64_t v = (((int64_t)acc * qp[4 + 2 * bn + j] + B) >> qp[4 + 3 * bn + j]) + qp[0];
                v = v < qp[1] ? qp[1] : v > qp[2] ? qp[2] : v;
                out[i * 4 * bn + (j / 32) * 128 + j % 32] = (int8_t)v; /* core column j / 32 */
            }
    }
    return AIEK_OK;
}

int aiednn_gemm_ntile_finish(void) { return AIEK_OK; }

const int8_t *aiednn_gemm_ntile_c(int mb) {
    return g_ntx_c + (int64_t)(mb * g_ntx_nkb + g_ntx_nkb - 1) * 4 * g_ntx_bm * g_ntx_bn;
}

/* x86 stand-in for aiednn_gemm_mmt_n_start: n MMT launch tiles (the launch A layout, aie_mmt_a_to_launch)
 * against one set of columns, results in the board's C tile layout (aiednn_gemm_ntile_c) */
int aiednn_gemm_mmt_n_start(const int8_t *a, int n, const int8_t *b, int kcode) {
    const int64_t rows = HW_ROWS, cols = HW_COLS, tile = 4 * (32 * HW_ROWS) * 128; /* the mesh's block */
    if (n < 1 || n > 32 || kcode < 1 || kcode > 2)
        return AIEK_ERR_ARGS;
    if (n * tile > g_ntx_c_bytes) {
        int8_t *grown = (int8_t *)realloc(g_ntx_c, (size_t)(n * tile));
        if (!grown)
            return AIEK_ERR_ALLOC;
        g_ntx_c = grown, g_ntx_c_bytes = n * tile;
    }
    g_ntx_bm = 32 * HW_ROWS, g_ntx_bn = 128, g_ntx_nkb = 1;
    static int8_t packed[16384], out[4096];
    for (int t = 0; t < n; t++)
        for (int64_t r = 0; r < rows; r++) {
            const int8_t *la = a + (int64_t)t * (32 * HW_ROWS) * 512;
            for (int kr = 0; kr < 8; kr++) /* the launch layout back to the packed rounds */
                for (int i = 0; i < 32; i++)
                    memcpy(packed + kr * 2048 + i * 64, la + (r * 32 + i) * 512 + kr * 64, 64);
            for (int64_t c = 0; c < cols; c++) {
                memset(out, 0, sizeof out);
                for (int kr = 0; kr < 8; kr++)
                    aie_mmt_round(packed + kr * 2048, b + c * AIE_MMT_RES, kcode, kr, NULL, out);
                int8_t *ct = g_ntx_c + (int64_t)t * tile;
                for (int row = 0; row < 32; row++)
                    memcpy(ct + (r * 32 + row) * 512 + c * 128, out + row * 128, 128);
            }
        }
    return AIEK_OK;
}

/* x86 stand-in for aiednn_gemm_dwq_n_start / _finish: n tiles of dwq_blk_s8s8 items (3 * cores)
 * through dwq_blk_s8s8_ref, outputs in item order */
static int8_t *g_dwnx_out;
static int64_t g_dwnx_items, g_dwnx_cap;

int aiednn_gemm_dwq_n_start(const int8_t *items, int n, const int32_t *geom) {
    const int64_t ni = (int64_t)n * 3 * HW_ROWS * HW_COLS, shape_i[2] = {ni, 4096}, shape_g[1] = {8},
                  shape_o[2] = {ni, 1024};
    if (!items || n < 1 || n > 32)
        return AIEK_ERR_ARGS;
    if (ni * 1024 > g_dwnx_cap) {
        int8_t *grown = (int8_t *)realloc(g_dwnx_out, (size_t)(ni * 1024));
        if (!grown)
            return AIEK_ERR_ALLOC;
        g_dwnx_out = grown, g_dwnx_cap = ni * 1024;
    }
    aiek_tensor t[3] = {{(void *)items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(ni * 4096)},
                        {(void *)geom, 1, AIEK_INT, 32, 1, shape_g, 32},
                        {g_dwnx_out, 2, AIEK_INT, 8, 1, shape_o, (uint64_t)(ni * 1024)}};
    g_dwnx_items = ni;
    return dwq_blk_s8s8_ref(t, 3);
}

int aiednn_gemm_dwq_n_finish(int8_t *out) {
    if (!g_dwnx_items || !out)
        return AIEK_ERR_ARGS;
    memcpy(out, g_dwnx_out, (size_t)(g_dwnx_items * 1024));
    g_dwnx_items = 0;
    return AIEK_OK;
}

/* x86 stand-in for aiednn_gemm_items_n_start / _finish: n tiles of HW_ROWS * HW_COLS one-per-core items
 * under one descriptor, through the same mode code the core runs (pool / eltwise / sdpa; 0xD7 as dw_blk_s8s32) */
static int32_t *g_itnx_out;
static int64_t g_itnx_items, g_itnx_cap;

int aiednn_gemm_items_n_start(const int8_t *items, int n, int opcode, const uint8_t *desc, int desc_n) {
    const int64_t ni = (int64_t)n * HW_ROWS * HW_COLS;
    if (!items || n < 1 || n > 32 || desc_n < 0 || desc_n > 30)
        return AIEK_ERR_ARGS;
    if (ni * 1024 > g_itnx_cap) {
        int32_t *grown = (int32_t *)realloc(g_itnx_out, (size_t)(ni * 1024) * sizeof(int32_t));
        if (!grown)
            return AIEK_ERR_ALLOC;
        g_itnx_out = grown, g_itnx_cap = ni * 1024;
    }
    uint8_t d[32] = {0};
    memcpy(d, desc, (size_t)desc_n);
    for (int64_t i = 0; i < ni; i++) {
        const int8_t *in = items + i * 4096;
        int32_t *o = g_itnx_out + i * 1024;
        if (opcode == AIE_MODE_OP_POOL) {
            AIE_MODE_POOL(in, o, d);
        } else if (opcode == AIE_MODE_OP_ELTWISE) {
            AIE_MODE_ELTWISE(in, o, d);
        } else if (opcode == AIE_MODE_OP_FPOOL) {
            aie_fpool_item_ref((const float *)in, (float *)o, d);
        } else if (opcode == AIE_MODE_OP_SDPA) {
            int32_t sc[AIE_SDPA_MAX_KV];
            aie_sdpa_item(in, o, d, sc);
        } else if (opcode == 0xD7) { /* {s, KH, KW, Wp LE16, Rout LE16, Wo LE16} */
            const int32_t g[8] = {d[0], d[1], d[2], AIE_MODE_LE16(d, 3), AIE_MODE_LE16(d, 5), AIE_MODE_LE16(d, 7),
                                  0,    0};
            const int64_t shape_i[2] = {1, 4096}, shape_g[1] = {8}, shape_o[2] = {1, 1024};
            aiek_tensor t[3] = {{(void *)in, 2, AIEK_INT, 8, 1, shape_i, 4096},
                                {(void *)g, 1, AIEK_INT, 32, 1, shape_g, 32},
                                {o, 2, AIEK_INT, 32, 1, shape_o, 4096}};
            const int rc = dw_blk_s8s32_ref(t, 3);
            if (rc != AIEK_OK)
                return rc;
        } else {
            return AIEK_ERR_PARAM;
        }
    }
    g_itnx_items = ni;
    return AIEK_OK;
}

int aiednn_gemm_items_n_finish(int32_t *out) {
    if (!g_itnx_items || !out)
        return AIEK_ERR_ARGS;
    memcpy(out, g_itnx_out, (size_t)(g_itnx_items * 1024) * sizeof(int32_t));
    g_itnx_items = 0;
    return AIEK_OK;
}
