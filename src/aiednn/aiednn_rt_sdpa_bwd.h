/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_sdpa_bwd.h -- scaled dot-product attention backward (training), f32, [B, H, S, D].
 *
 * Per (batch, head) the forward's softmax is recomputed (no stored statistics), then
 *   dV  = P^T dO           dP = dO V^T           dS = P * (dP - rowsum(dP * P))
 *   dQ  = scale dS K       dK = scale dS^T Q
 * with the mask / causal rule of the forward (bottom-right aligned causal). GQA / MQA: K and V carry Hk heads,
 * query head h uses head h / (H / Hk), and dK / dV sum over the H / Hk query heads of a group.
 *
 * One routine, two GEMMs behind a callback: ref:sdpa (double-accumulated loops) and gemm+sdpabwd (the float
 * GEMM through dispatch, bf16x3 on the AIE). Softmax and dS run in double on the host.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_ops.h, aiednn_rt.cc's create helpers.
 ******************************************************************************/
#ifndef AIEDNN_RT_SDPA_BWD_H
#define AIEDNN_RT_SDPA_BWD_H

/* C[M][N] = A[M][K] B[K][N] (bt: B as [N][K]), f32 */
typedef int (*sdpab_mm_fn)(int64_t M, int64_t N, int64_t K, const float *A, const float *B, float *C, int bt);

static int sdpab_mm_ref(int64_t M, int64_t N, int64_t K, const float *A, const float *B, float *C, int bt) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            double s = 0;
            for (int64_t k = 0; k < K; k++)
                s += (double)A[i * K + k] * (double)(bt ? B[j * K + k] : B[k * N + j]);
            C[i * N + j] = (float)s;
        }
    return AIEK_OK;
}

/* one (b, h): S, P, dV += P^T dO, dP, dS, dQ, dK += scale dS^T Q; buffers from the caller */
typedef struct {
    float *Q, *K, *V, *dO, *P, *Pt, *dP, *dS, *dSt, *tmpKD, *tmpKV, *dQ;
    double *row;
} sdpab_buf;

#define SDPAB_AT(bshd, HH, b, h, s, e, S, W)                                                                           \
    ((bshd) ? ((((b) * (S) + (s)) * (HH) + (h)) * (W) + (e)) : (((((b) * (HH) + (h)) * (S) + (s)) * (W) + (e))))

static int sdpab_head(const rt_prob *p, void *const *ptr, sdpab_mm_fn mm, const sdpab_buf *u, int64_t b, int64_t h,
                      double *dK, double *dV) {
    const aiednn_memory_desc_t *q = rt_md(p, AIEDNN_ARG_QUERY), *k = rt_md(p, AIEDNN_ARG_KEY),
                               *v = rt_md(p, AIEDNN_ARG_VALUE);
    const aiednn_memory_desc_t *mk = rt_md(p, AIEDNN_ARG_ATTN_MASK);
    const int64_t H = q->dims[1], Hk = k->dims[1], Sq = q->dims[2], Skv = k->dims[2], D = q->dims[3], Dv = v->dims[3];
    const int64_t hk = h / (H / Hk);
    const int bs = (p->flags & AIEDNN_SDPA_QKV_BSHD) != 0;
    const double sc = p->scale;
    for (int64_t i = 0; i < Sq; i++) {
        for (int64_t e = 0; e < D; e++)
            u->Q[i * D + e] =
                (float)rt_ld(q->data_type, rt_ptr(p, ptr, AIEDNN_ARG_QUERY), SDPAB_AT(bs, H, b, h, i, e, Sq, D));
        for (int64_t e = 0; e < Dv; e++)
            u->dO[i * Dv + e] = (float)rt_ld(rt_md(p, AIEDNN_ARG_DIFF_DST)->data_type,
                                             rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST), SDPAB_AT(bs, H, b, h, i, e, Sq, Dv));
    }
    for (int64_t j = 0; j < Skv; j++) {
        for (int64_t e = 0; e < D; e++)
            u->K[j * D + e] =
                (float)rt_ld(k->data_type, rt_ptr(p, ptr, AIEDNN_ARG_KEY), SDPAB_AT(bs, Hk, b, hk, j, e, Skv, D));
        for (int64_t e = 0; e < Dv; e++)
            u->V[j * Dv + e] =
                (float)rt_ld(v->data_type, rt_ptr(p, ptr, AIEDNN_ARG_VALUE), SDPAB_AT(bs, Hk, b, hk, j, e, Skv, Dv));
    }
    int rc = mm(Sq, Skv, D, u->Q, u->K, u->P, 1); /* S = Q K^T */
    aiednn_memory_desc_t scm;                     /* the [B, H, Sq, Skv] score shape the mask broadcasts to */
    memset(&scm, 0, sizeof scm);
    scm.ndims = 4, scm.dims[0] = q->dims[0], scm.dims[1] = H, scm.dims[2] = Sq, scm.dims[3] = Skv;
    for (int64_t i = 0; rc == AIEK_OK && i < Sq; i++) { /* P = softmax(scale S + mask), the forward's rule */
        double mx = -INFINITY, sum = 0;
        for (int64_t j = 0; j < Skv; j++) {
            double a = u->P[i * Skv + j] * sc;
            if (mk)
                a += rt_ld(mk->data_type, rt_ptr(p, ptr, AIEDNN_ARG_ATTN_MASK),
                           bcast_off(mk, &scm, ((b * H + h) * Sq + i) * Skv + j));
            if (p->causal && j > i + (Skv - Sq))
                a = -INFINITY;
            u->row[j] = a, mx = a > mx ? a : mx;
        }
        for (int64_t j = 0; j < Skv; j++)
            sum += u->row[j] = mx == -INFINITY ? 0 : exp(u->row[j] - mx);
        for (int64_t j = 0; j < Skv; j++)
            u->P[i * Skv + j] = (float)(sum > 0 ? u->row[j] / sum : 0.0), u->Pt[j * Sq + i] = u->P[i * Skv + j];
    }
    if (rc == AIEK_OK) /* dV (this head's part) = P^T dO */
        rc = mm(Skv, Dv, Sq, u->Pt, u->dO, u->tmpKV, 0);
    for (int64_t i = 0; rc == AIEK_OK && i < Skv * Dv; i++)
        dV[i] += u->tmpKV[i];
    if (rc == AIEK_OK) /* dP = dO V^T */
        rc = mm(Sq, Skv, Dv, u->dO, u->V, u->dP, 1);
    for (int64_t i = 0; rc == AIEK_OK && i < Sq; i++) { /* dS = P (dP - sum_j dP P), scaled into dQ / dK below */
        double t = 0;
        for (int64_t j = 0; j < Skv; j++)
            t += (double)u->dP[i * Skv + j] * u->P[i * Skv + j];
        for (int64_t j = 0; j < Skv; j++) {
            const float ds = (float)(sc * u->P[i * Skv + j] * ((double)u->dP[i * Skv + j] - t));
            u->dS[i * Skv + j] = ds, u->dSt[j * Sq + i] = ds;
        }
    }
    if (rc == AIEK_OK) /* dQ = scale dS K */
        rc = mm(Sq, D, Skv, u->dS, u->K, u->dQ, 0);
    for (int64_t i = 0; rc == AIEK_OK && i < Sq; i++)
        for (int64_t e = 0; e < D; e++)
            rt_st(rt_md(p, AIEDNN_ARG_DIFF_QUERY)->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_QUERY),
                  SDPAB_AT(bs, H, b, h, i, e, Sq, D), u->dQ[i * D + e]);
    if (rc == AIEK_OK) /* dK (this head's part) = scale dS^T Q */
        rc = mm(Skv, D, Sq, u->dSt, u->Q, u->tmpKD, 0);
    for (int64_t i = 0; rc == AIEK_OK && i < Skv * D; i++)
        dK[i] += u->tmpKD[i];
    return rc;
}

static int run_sdpa_bwd(const rt_prob *p, void *const *ptr, sdpab_mm_fn mm) {
    const aiednn_memory_desc_t *q = rt_md(p, AIEDNN_ARG_QUERY), *k = rt_md(p, AIEDNN_ARG_KEY),
                               *v = rt_md(p, AIEDNN_ARG_VALUE);
    const int64_t B = q->dims[0], H = q->dims[1], Hk = k->dims[1], Sq = q->dims[2], Skv = k->dims[2], D = q->dims[3],
                  Dv = v->dims[3], G = H / Hk;
    const int bs = (p->flags & AIEDNN_SDPA_QKV_BSHD) != 0;
    const size_t nf =
        (size_t)(Sq * D + Skv * D + Skv * Dv + Sq * Dv + 3 * Sq * Skv + 2 * Sq * Skv + Skv * D + Skv * Dv + Sq * D);
    float *f = (float *)rt_alloc(nf * sizeof(float));
    double *acc = (double *)rt_alloc((size_t)(Skv * D + Skv * Dv + Skv) * sizeof(double));
    if (!f || !acc) {
        rt_free(f), rt_free(acc);
        return AIEK_ERR_ALLOC;
    }
    sdpab_buf u;
    u.Q = f, u.K = u.Q + Sq * D, u.V = u.K + Skv * D, u.dO = u.V + Skv * Dv, u.P = u.dO + Sq * Dv;
    u.Pt = u.P + Sq * Skv, u.dP = u.Pt + Sq * Skv, u.dS = u.dP + Sq * Skv, u.dSt = u.dS + Sq * Skv;
    u.tmpKD = u.dSt + Sq * Skv, u.tmpKV = u.tmpKD + Skv * D, u.dQ = u.tmpKV + Skv * Dv;
    double *dK = acc, *dV = dK + Skv * D;
    u.row = dV + Skv * Dv;
    int rc = AIEK_OK;
    for (int64_t b = 0; rc == AIEK_OK && b < B; b++)
        for (int64_t hk = 0; rc == AIEK_OK && hk < Hk; hk++) { /* a KV head and the G query heads that read it */
            memset(dK, 0, (size_t)(Skv * (D + Dv)) * sizeof(double));
            for (int64_t g = 0; rc == AIEK_OK && g < G; g++)
                rc = sdpab_head(p, ptr, mm, &u, b, hk * G + g, dK, dV);
            for (int64_t j = 0; rc == AIEK_OK && j < Skv; j++) {
                for (int64_t e = 0; e < D; e++)
                    rt_st(rt_md(p, AIEDNN_ARG_DIFF_KEY)->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_KEY),
                          SDPAB_AT(bs, Hk, b, hk, j, e, Skv, D), dK[j * D + e]);
                for (int64_t e = 0; e < Dv; e++)
                    rt_st(rt_md(p, AIEDNN_ARG_DIFF_VALUE)->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_VALUE),
                          SDPAB_AT(bs, Hk, b, hk, j, e, Skv, Dv), dV[j * Dv + e]);
            }
        }
    rt_free(f), rt_free(acc);
    return rc;
}
#undef SDPAB_AT

static int run_sdpa_bwd_ref(const rt_prob *p, void *const *ptr) { return run_sdpa_bwd(p, ptr, sdpab_mm_ref); }

/* gemm+sdpabwd: the same routine, its GEMMs on the float matmul dispatch (bf16x3 on the AIE) */
static int sdpab_mm_gx(int64_t M, int64_t N, int64_t K, const float *A, const float *B, float *C, int bt) {
    const gx_types t = {aiednn_f32, aiednn_f32, aiednn_f32};
    return gx_gemm(&t, M, N, K, A, B, C, bt);
}
static int acc_gx_sdpabwd(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (p->kind != aiednn_sdpa || p->prop != RT_BWD) {
        snprintf(why, cap, "sdpa backward only");
        return 0;
    }
    const gx_types t = {aiednn_f32, aiednn_f32, aiednn_f32};
    const aiednn_memory_desc_t *q = rt_md(p, AIEDNN_ARG_QUERY), *k = rt_md(p, AIEDNN_ARG_KEY);
    return gx_inner_ok(&t, q->dims[2], k->dims[2], q->dims[3], bind, why, cap, 1);
}
static int run_gx_sdpabwd(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    return run_sdpa_bwd(&prim->prob, ptr, sdpab_mm_gx);
}

extern "C" aiednn_status_t aiednn_sdpa_backward_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *q,
                                                       const aiednn_memory_desc_t *kk, const aiednn_memory_desc_t *v,
                                                       const aiednn_memory_desc_t *mask,
                                                       const aiednn_memory_desc_t *diff_dst,
                                                       const aiednn_memory_desc_t *diff_q,
                                                       const aiednn_memory_desc_t *diff_k,
                                                       const aiednn_memory_desc_t *diff_v, float scale, int causal) {
    const char *k = "sdpa";
    REQUIRE(pr && md_ok(q) && md_ok(kk) && md_ok(v) && md_ok(diff_dst) && md_ok(diff_q) && md_ok(diff_k) &&
                md_ok(diff_v) && (!mask || md_ok(mask)),
            k, "null or malformed descriptor");
    REQUIRE(!(causal & AIEDNN_SDPA_DST_BSHD) && md_same_shape(q, diff_q) && md_same_shape(kk, diff_k) &&
                md_same_shape(v, diff_v),
            k, "diff_q / diff_k / diff_v shaped as q / k / v (no folded dst layout)");
    const aiednn_memory_desc_t *m[7] = {q, kk, v, diff_dst, diff_q, diff_k, diff_v};
    for (int i = 0; i < 7; i++)
        REQUIRE(m[i]->ndims == 4 && m[i]->data_type == aiednn_f32, k, "f32, rank 4");
    aiednn_memory_desc_t qc = *q, kc = *kk, vc = *v, dc = *diff_dst; /* the [B, H, S, D] view */
    const auto swap12 = [](aiednn_memory_desc_t *x) {
        const int64_t t = x->dims[1];
        x->dims[1] = x->dims[2], x->dims[2] = t;
    };
    if (causal & AIEDNN_SDPA_QKV_BSHD)
        swap12(&qc), swap12(&kc), swap12(&vc), swap12(&dc);
    REQUIRE(qc.dims[0] == kc.dims[0] && kc.dims[0] == vc.dims[0] && kc.dims[1] == vc.dims[1] && kc.dims[1] >= 1 &&
                qc.dims[1] % kc.dims[1] == 0 && kc.dims[2] == vc.dims[2] && kc.dims[3] == qc.dims[3] &&
                dc.dims[0] == qc.dims[0] && dc.dims[1] == qc.dims[1] && dc.dims[2] == qc.dims[2] &&
                dc.dims[3] == vc.dims[3],
            k, "Q [B,H,Sq,D], K [B,Hk,Skv,D], V [B,Hk,Skv,Dv], diff_dst [B,H,Sq,Dv], Hk dividing H");
    rt_prob p;
    prob_init(&p, aiednn_sdpa, 0);
    p.prop = RT_BWD;
    prob_add(&p, AIEDNN_ARG_QUERY, &qc);
    prob_add(&p, AIEDNN_ARG_KEY, &kc);
    prob_add(&p, AIEDNN_ARG_VALUE, &vc);
    prob_add(&p, AIEDNN_ARG_DIFF_DST, &dc);
    prob_add(&p, AIEDNN_ARG_DIFF_QUERY, &qc);
    prob_add(&p, AIEDNN_ARG_DIFF_KEY, &kc);
    prob_add(&p, AIEDNN_ARG_DIFF_VALUE, &vc);
    if (mask) {
        aiednn_memory_desc_t scm = qc;
        scm.dims[3] = kc.dims[2];
        REQUIRE(md_bcast_ok(mask, &scm), k, "mask must broadcast to [B, H, Sq, Skv]");
        prob_add(&p, AIEDNN_ARG_ATTN_MASK, mask);
    }
    p.scale = scale;
    p.causal = (causal & 1) != 0;
    p.flags = (unsigned)(causal & AIEDNN_SDPA_QKV_BSHD);
    return create(pr, &p);
}

#endif /* AIEDNN_RT_SDPA_BWD_H */
