/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_sdpa_var.h -- attention over variable-length sequences, optionally reading K / V from a paged
 * cache (LLM serving: prefill of a padded batch, decode against a block KV cache).
 *
 * Per sequence b: nq = seq_q[b] valid query rows (all Sq without seq_q), nk = seq_kv[b] valid keys. Query
 * head h reads K / V head h / (H / Hk) (GQA / MQA). Paged: K [P, Hk, page, D], V [P, Hk, page, Dv] and
 * page_table [B, maxp] s32 -- key j of sequence b is row j % page of page page_table[b, j / page]. The
 * scores are Q K^T * scale (+ causal: key j visible to query i when j <= i + nk - nq, bottom-right aligned
 * per sequence), softmax in double over the nk valid keys, then P V; query rows >= nq are written 0.
 *
 * ref: runs the two products in double (sdpab_mm_ref); gemm+sdpavar puts them on the AIE GEMM per (b, h)
 * with the sequence's own lengths.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_gemmx.h, aiednn_rt_sdpa_bwd.h (sdpab_mm_*).
 ******************************************************************************/
#ifndef AIEDNN_RT_SDPA_VAR_H
#define AIEDNN_RT_SDPA_VAR_H

typedef struct {
    int64_t B, H, Hk, Sq, Skv, D, Dv, page, maxp, npages;
    int paged;
} sdpv_geo;

static void sdpv_geo_of(const rt_prob *p, sdpv_geo *g) {
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2];
    const aiednn_memory_desc_t *pt = rt_md(p, AIEDNN_ARG_PAGE_TABLE);
    g->B = q->dims[0], g->H = q->dims[1], g->Sq = q->dims[2], g->D = q->dims[3], g->Dv = v->dims[3];
    g->Hk = k->dims[1], g->paged = pt != nullptr;
    g->page = g->paged ? k->dims[2] : 0, g->maxp = g->paged ? pt->dims[1] : 0, g->npages = g->paged ? k->dims[0] : 0;
    g->Skv = g->paged ? g->maxp * g->page : k->dims[2];
}

/* element offset of row j (width W) of K or V for (b, kv head kh); -1: a page index out of range */
static inline int64_t sdpv_kv_off(const sdpv_geo *g, const int32_t *pt, int64_t b, int64_t kh, int64_t j, int64_t W) {
    if (!g->paged)
        return ((b * g->Hk + kh) * g->Skv + j) * W;
    const int64_t pg = pt[b * g->maxp + j / g->page];
    return pg < 0 || pg >= g->npages ? -1 : ((pg * g->Hk + kh) * g->page + j % g->page) * W;
}

static int sdpv_run(const rt_prob *p, void *const *ptr, sdpab_mm_fn mm) {
    sdpv_geo g;
    sdpv_geo_of(p, &g);
    const aiednn_memory_desc_t *q = &p->md[0], *k = &p->md[1], *v = &p->md[2], *d = &p->md[3];
    const int32_t *sq = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_SEQ_LENS_Q);
    const int32_t *sk = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_SEQ_LENS_KV);
    const int32_t *pt = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_PAGE_TABLE);
    const int64_t Sq = g.Sq, Skv = g.Skv, D = g.D, Dv = g.Dv, Hg = g.H / g.Hk;
    float *Q = (float *)rt_alloc((size_t)(Sq * D) * sizeof(float)),
          *K = (float *)rt_alloc((size_t)(Skv * D) * sizeof(float));
    float *V = (float *)rt_alloc((size_t)(Skv * Dv) * sizeof(float)),
          *S = (float *)rt_alloc((size_t)(Sq * Skv) * sizeof(float));
    float *O = (float *)rt_alloc((size_t)(Sq * Dv) * sizeof(float));
    double *row = (double *)rt_alloc((size_t)(Skv > 0 ? Skv : 1) * sizeof(double));
    int rc = Q && K && V && S && O && row ? AIEK_OK : AIEK_ERR_ALLOC;
    for (int64_t b = 0; rc == AIEK_OK && b < g.B; b++) {
        const int64_t nq = sq ? sq[b] : Sq, nk = sk[b];
        if (nq < 0 || nq > Sq || nk < 0 || nk > Skv) {
            rc = AIEK_ERR_ARGS;
            break;
        }
        for (int64_t h = 0; rc == AIEK_OK && h < g.H; h++) {
            const int64_t kh = h / Hg, qo = (b * g.H + h) * Sq;
            for (int64_t i = 0; i < nq * D; i++)
                Q[i] = (float)rt_ld(q->data_type, ptr[0], qo * D + i);
            for (int64_t j = 0; j < nk && rc == AIEK_OK; j++) {
                const int64_t ok = sdpv_kv_off(&g, pt, b, kh, j, D), ov = sdpv_kv_off(&g, pt, b, kh, j, Dv);
                if (ok < 0 || ov < 0) {
                    rc = AIEK_ERR_ARGS;
                    break;
                }
                /* a quantized (compressed) K / V cache: dequantized in f32 as it is gathered */
                const int kq = dt_is_int(k->data_type), vq = dt_is_int(v->data_type);
                for (int64_t e = 0; e < D; e++) {
                    const double x = rt_ld(k->data_type, ptr[1], ok + e);
                    K[j * D + e] = kq ? (float)((x - p->zps[0]) * (double)p->scales[0]) : (float)x;
                }
                for (int64_t e = 0; e < Dv; e++) {
                    const double x = rt_ld(v->data_type, ptr[2], ov + e);
                    V[j * Dv + e] = vq ? (float)((x - p->zps[1]) * (double)p->scales[1]) : (float)x;
                }
            }
            if (rc == AIEK_OK && nq && nk)
                rc = mm(nq, nk, D, Q, K, S, 1);                       /* scores: Q K^T */
            for (int64_t i = 0; rc == AIEK_OK && nk && i < nq; i++) { /* the softmax of row i, normalized, in place */
                double mx = -INFINITY, sum = 0;
                for (int64_t j = 0; j < nk; j++) {
                    double a = (double)S[i * nk + j] * (double)p->scale;
                    if (p->causal && j > i + (nk - nq))
                        a = -INFINITY;
                    row[j] = a, mx = a > mx ? a : mx;
                }
                for (int64_t j = 0; j < nk; j++)
                    sum += row[j] = mx == -INFINITY ? 0 : exp(row[j] - mx);
                for (int64_t j = 0; j < nk; j++)
                    S[i * nk + j] = (float)(sum > 0 ? row[j] / sum : 0.0);
            }
            if (rc == AIEK_OK && nq && nk)
                rc = mm(nq, Dv, nk, S, V, O, 0); /* P V */
            for (int64_t i = 0; rc == AIEK_OK && i < Sq; i++)
                for (int64_t e = 0; e < Dv; e++)
                    rt_st(d->data_type, ptr[3], (qo + i) * Dv + e, i < nq && nk ? (double)O[i * Dv + e] : 0.0);
        }
    }
    rt_free(Q), rt_free(K), rt_free(V), rt_free(S), rt_free(O), rt_free(row);
    return rc;
}

static int run_sdpa_var_ref(const rt_prob *p, void *const *ptr) { return sdpv_run(p, ptr, sdpab_mm_ref); }

static int acc_gx_sdpavar(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (p->kind != aiednn_sdpa || !(p->flags & RT_SDPA_VAR) || p->prop != RT_FWD_INF) {
        snprintf(why, cap, "variable-length / paged sdpa only");
        return 0;
    }
    sdpv_geo g;
    sdpv_geo_of(p, &g);
    const gx_types t = {aiednn_f32, aiednn_f32, aiednn_f32};
    return gx_inner_ok(&t, g.Sq, g.Skv, g.D, bind, why, cap, 1);
}

static int run_gx_sdpavar(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    return sdpv_run(&prim->prob, ptr, sdpab_mm_gx);
}

static aiednn_status_t sdpa_varlen_create_(aiednn_primitive_t *pr, const aiednn_memory_desc_t *q,
                                           const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                           const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *seq_q,
                                           const aiednn_memory_desc_t *seq_kv, const aiednn_memory_desc_t *page_table,
                                           float scale, int causal, const float *kv_scales, const int32_t *kv_zps);

extern "C" aiednn_status_t aiednn_sdpa_varlen_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *q,
                                                     const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                                     const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *seq_q,
                                                     const aiednn_memory_desc_t *seq_kv,
                                                     const aiednn_memory_desc_t *page_table, float scale, int causal) {
    return sdpa_varlen_create_(pr, q, k, v, dst, seq_q, seq_kv, page_table, scale, causal, nullptr, nullptr);
}

/* the same over an 8-bit (compressed) K / V cache: K = (k - kv_zps[0]) * kv_scales[0], V likewise with [1] */
extern "C" aiednn_status_t aiednn_sdpa_varlen_create_kvq(
    aiednn_primitive_t *pr, const aiednn_memory_desc_t *q, const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
    const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *seq_q, const aiednn_memory_desc_t *seq_kv,
    const aiednn_memory_desc_t *page_table, float scale, int causal, const float *kv_scales, const int32_t *kv_zps) {
    return sdpa_varlen_create_(pr, q, k, v, dst, seq_q, seq_kv, page_table, scale, causal, kv_scales, kv_zps);
}

static aiednn_status_t sdpa_varlen_create_(aiednn_primitive_t *pr, const aiednn_memory_desc_t *q,
                                           const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
                                           const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *seq_q,
                                           const aiednn_memory_desc_t *seq_kv, const aiednn_memory_desc_t *page_table,
                                           float scale, int causal, const float *kv_scales, const int32_t *kv_zps) {
    const char *kd = "sdpa";
    REQUIRE(pr && md_ok(q) && md_ok(k) && md_ok(v) && md_ok(dst) && md_ok(seq_kv) && (!seq_q || md_ok(seq_q)) &&
                (!page_table || md_ok(page_table)),
            kd, "null or malformed descriptor");
    REQUIRE(q->ndims == 4 && k->ndims == 4 && v->ndims == 4 && dst->ndims == 4, kd, "Q, K, V, dst rank 4");
    const int kvq = kv_scales != nullptr;
    REQUIRE(dt_is_float(q->data_type) && dt_is_float(dst->data_type) &&
                (kvq ? (k->data_type == aiednn_s8 || k->data_type == aiednn_u8) &&
                           (v->data_type == aiednn_s8 || v->data_type == aiednn_u8) && kv_scales[0] > 0 &&
                           kv_scales[1] > 0
                     : dt_is_float(k->data_type) && dt_is_float(v->data_type)),
            kd, "floating Q / dst; K / V floating, or s8 / u8 with kv_scales > 0 (_kvq)");
    const int64_t B = q->dims[0], H = q->dims[1], Sq = q->dims[2], D = q->dims[3], Hk = k->dims[1];
    REQUIRE(dst->dims[0] == B && dst->dims[1] == H && dst->dims[2] == Sq && dst->dims[3] == v->dims[3], kd,
            "dst [B, H, Sq, Dv]");
    REQUIRE(k->dims[3] == D && v->dims[0] == k->dims[0] && v->dims[1] == Hk && v->dims[2] == k->dims[2] && Hk >= 1 &&
                H % Hk == 0,
            kd, "K [.., Hk, .., D], V [.., Hk, .., Dv] alike, Hk dividing H");
    REQUIRE(seq_kv->ndims == 1 && seq_kv->dims[0] == B && seq_kv->data_type == aiednn_s32, kd, "seq_kv s32 [B]");
    REQUIRE(!seq_q || (seq_q->ndims == 1 && seq_q->dims[0] == B && seq_q->data_type == aiednn_s32), kd,
            "seq_q s32 [B]");
    if (page_table)
        REQUIRE(page_table->ndims == 2 && page_table->dims[0] == B && page_table->data_type == aiednn_s32, kd,
                "page_table s32 [B, max pages]; K / V then [pages, Hk, page, D]");
    else
        REQUIRE(k->dims[0] == B, kd, "dense K / V [B, Hk, Skv, D] (or a page table)");
    rt_prob p;
    prob_init(&p, aiednn_sdpa, 0);
    prob_add(&p, AIEDNN_ARG_QUERY, q);
    prob_add(&p, AIEDNN_ARG_KEY, k);
    prob_add(&p, AIEDNN_ARG_VALUE, v);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    prob_add(&p, AIEDNN_ARG_SEQ_LENS_KV, seq_kv);
    if (seq_q)
        prob_add(&p, AIEDNN_ARG_SEQ_LENS_Q, seq_q);
    if (page_table)
        prob_add(&p, AIEDNN_ARG_PAGE_TABLE, page_table);
    p.scale = scale;
    p.causal = causal != 0;
    p.flags = RT_SDPA_VAR;
    if (kvq)
        p.scales[0] = kv_scales[0], p.scales[1] = kv_scales[1], p.zps[0] = kv_zps ? kv_zps[0] : 0,
        p.zps[1] = kv_zps ? kv_zps[1] : 0;
    return create(pr, &p);
}

#endif /* AIEDNN_RT_SDPA_VAR_H */
