/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_sdpa8.h -- gemm+sdpa8: attention over s8 Q / K / V (per-tensor scales, attr src_scales) as two
 * int8 GEMMs per head on the matmul dispatch. Scores Q K^T are exact int32; the softmax runs on the host in
 * float; P is quantized per row to 8 bits (u8 codes shifted to s8, the shift corrected exactly with V's
 * column sums) for P V. Within a few codes of ref:, so only with AIEDNN_ATTR_TOLERANCE_INT8_ATTN.
 ******************************************************************************/
#ifndef AIEDNN_RT_SDPA8_H
#define AIEDNN_RT_SDPA8_H

static const gx_types g_s8_t = {aiednn_s8, aiednn_s8, aiednn_s32};

static int acc_gx_sdpa8(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2], *d = &p->md[3];
    const int s8 = q->data_type == aiednn_s8 && k->data_type == aiednn_s8 && v->data_type == aiednn_s8;
    if (!s8 || p->ndq < 3 || !(p->dq_sc[0] > 0 && p->dq_sc[1] > 0 && p->dq_sc[2] > 0) || p->dq_zp[0] || p->dq_zp[1] ||
        p->dq_zp[2] || !(p->attr_flags & AIEDNN_ATTR_TOLERANCE_INT8_ATTN) || (p->flags & RT_SDPA_VAR) ||
        !(d->data_type == aiednn_f32 || ((d->data_type == aiednn_s8 || d->data_type == aiednn_u8) && p->has_attr))) {
        snprintf(why, cap,
                 "s8 Q / K / V with attr scales (zero point 0), AIEDNN_ATTR_TOLERANCE_INT8_ATTN, f32 or "
                 "quantized dst");
        return 0;
    }
    const int64_t Sq = q->dims[2], Skv = k->dims[2], D = q->dims[3], Dv = v->dims[3];
    rt_bind b2 = {nullptr, nullptr};
    return gx_inner_ok(&g_s8_t, Sq, Skv, D, bind, why, cap, 1) && gx_inner_ok(&g_s8_t, Sq, Dv, Skv, &b2, why, cap);
}

typedef struct {
    int8_t *Q, *K, *V, *P;
    int32_t *S, *O;
    float *row, *pstep;
    int32_t *vsum;
} sdpa8_bufs;

/* one head (b, h): scores, softmax, P quantized, P V; the result rows into dst */
static int sdpa8_head(const rt_prob *p, void *const *ptr, int64_t b, int64_t h, sdpa8_bufs *B) {
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2], *d = &p->md[3];
    const int64_t H = d->dims[1], Sq = q->dims[2], Skv = k->dims[2], D = q->dims[3], Dv = v->dims[3], Hk = k->dims[1];
    const int qs = (p->flags & AIEDNN_SDPA_QKV_BSHD) != 0, ds = (p->flags & AIEDNN_SDPA_DST_BSHD) != 0;
    const int8_t *Qp = (const int8_t *)gx_head_rows(aiednn_s8, ptr[0], qs, H, b, h, Sq, D, aiednn_s8, (char *)B->Q);
    const int8_t *Kp =
        (const int8_t *)gx_head_rows(aiednn_s8, ptr[1], qs, Hk, b, h / (H / Hk), Skv, D, aiednn_s8, (char *)B->K);
    const int8_t *Vp =
        (const int8_t *)gx_head_rows(aiednn_s8, ptr[2], qs, Hk, b, h / (H / Hk), Skv, Dv, aiednn_s8, (char *)B->V);
    int rc = gx_gemm(&g_s8_t, Sq, Skv, D, Qp, Kp, B->S, 1); /* exact int32 scores */
    if (rc)
        return rc;
    const aiednn_memory_desc_t *mk = rt_md(p, AIEDNN_ARG_ATTN_MASK);
    const void *mp = rt_ptr(p, ptr, AIEDNN_ARG_ATTN_MASK);
    aiednn_memory_desc_t sc = *d;
    sc.dims[3] = Skv;
    const float c = (float)((double)p->scale * p->dq_sc[0] * p->dq_sc[1]);
    for (int64_t i = 0; i < Sq; i++) {
        const int64_t jn = p->causal ? (i + (Skv - Sq) + 1 < Skv ? i + (Skv - Sq) + 1 : Skv) : Skv;
        float *r = B->row;
        for (int64_t j = 0; j < Skv; j++)
            r[j] = j >= jn ? -INFINITY : (float)B->S[i * Skv + j] * c;
        if (mk)
            for (int64_t j = 0; j < jn; j++)
                r[j] += (float)rt_ld(mk->data_type, mp, bcast_off(mk, &sc, ((b * H + h) * Sq + i) * Skv + j));
        float *pr = r + Skv; /* P row (row holds 2 Skv floats) */
        gx_softmax_row_f32(r, Skv, pr);
        float mx = 0;
        for (int64_t j = 0; j < Skv; j++)
            mx = pr[j] > mx ? pr[j] : mx;
        const float st = mx > 0 ? mx / 255.0f : 1.0f, inv = 1.0f / st;
        B->pstep[i] = st;
        for (int64_t j = 0; j < Skv; j++) { /* u8 code - 128: s8 for the AIE GEMM */
            const int u = (int)nearbyintf(pr[j] * inv);
            B->P[i * Skv + j] = (int8_t)((u > 255 ? 255 : u) - 128);
        }
    }
    for (int64_t e = 0; e < Dv; e++) { /* V's column sums: (u - 128) V + 128 sum(V) = u V */
        int32_t s = 0;
        for (int64_t j = 0; j < Skv; j++)
            s += Vp[j * Dv + e];
        B->vsum[e] = s;
    }
    rc = gx_gemm(&g_s8_t, Sq, Dv, Skv, B->P, Vp, B->O);
    if (rc)
        return rc;
    const int q8 = d->data_type == aiednn_s8 || d->data_type == aiednn_u8;
    const double sv = p->dq_sc[2], ds8 = p->dst_scale, inv8 = 1.0 / ds8, zp = p->dst_zp;
    const double lo = d->data_type == aiednn_s8 ? -128.0 : 0.0;
    for (int64_t i = 0; i < Sq; i++) {
        const int64_t at = ds ? ((b * Sq + i) * H + h) * Dv : ((b * H + h) * Sq + i) * Dv;
        const double rs = (double)B->pstep[i] * sv;
        for (int64_t e = 0; e < Dv; e++) {
            const float f = (float)(((double)B->O[i * Dv + e] + 128.0 * B->vsum[e]) * rs);
            if (q8)
                ((uint8_t *)ptr[3])[at + e] = (uint8_t)(int)rt_quant8(f, ds8, inv8, zp, lo);
            else
                ((float *)ptr[3])[at + e] = f;
        }
    }
    return AIEK_OK;
}

static int run_gx_sdpa8(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2], *d = &p->md[3];
    const int64_t B = d->dims[0], H = d->dims[1], Sq = q->dims[2], Skv = k->dims[2], D = q->dims[3], Dv = v->dims[3];
    sdpa8_bufs bf;
    bf.Q = (int8_t *)rt_alloc((size_t)(Sq * D)), bf.K = (int8_t *)rt_alloc((size_t)(Skv * D));
    bf.V = (int8_t *)rt_alloc((size_t)(Skv * Dv)), bf.P = (int8_t *)rt_alloc((size_t)(Sq * Skv));
    bf.S = (int32_t *)rt_alloc((size_t)(Sq * Skv) * 4), bf.O = (int32_t *)rt_alloc((size_t)(Sq * Dv) * 4);
    bf.row = (float *)rt_alloc((size_t)(2 * Skv) * 4), bf.pstep = (float *)rt_alloc((size_t)Sq * 4);
    bf.vsum = (int32_t *)rt_alloc((size_t)Dv * 4);
    int rc = bf.Q && bf.K && bf.V && bf.P && bf.S && bf.O && bf.row && bf.pstep && bf.vsum ? AIEK_OK : AIEK_ERR_ALLOC;
    for (int64_t bh = 0; rc == AIEK_OK && bh < B * H; bh++)
        rc = sdpa8_head(p, ptr, bh / H, bh % H, &bf);
    rt_free(bf.Q), rt_free(bf.K), rt_free(bf.V), rt_free(bf.P), rt_free(bf.S), rt_free(bf.O), rt_free(bf.row);
    rt_free(bf.pstep), rt_free(bf.vsum);
    return rc;
}

#endif /* AIEDNN_RT_SDPA8_H */
