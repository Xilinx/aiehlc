/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_linattn.h -- linear attention with a gated delta rule: Gated DeltaNet (a scalar gate per head),
 * KDA (a gate per key channel), and gated linear attention / RetNet (AIEDNN_LA_NO_DELTA). Per (b, h), with a
 * state h [Dk][Dv] (state_in or 0) and a_t = exp(gate_t) (1 without a gate):
 *   h <- Diag(a_t) h;  u_t = beta_t (v_t - h^T k_t)  (no delta: u_t = v_t);  h <- h + k_t u_t^T;
 *   o_t = scale h^T q_t
 *
 * ref: runs it token by token in double. gemm+linattn runs the chunked form: with G_i the cumulative log gate
 * inside a chunk of n tokens and h0 the state at its start,
 *   (I + A) U = R,  A_ij = beta_i (k_i . k_j)_{G_i - G_j} (j < i),  R_i = beta_i (v_i - (k_i)_{G_i}^T h0)
 *   O = scale ((Q)_G h0 + (Q K^T)_{G_i - G_j, j <= i} U),   h1 = Diag(e^{G_n}) h0 + ((K)_{G_n - G_j})^T U
 * where (x)_D weights channel d by e^D: four stacked f32 GEMMs per chunk on the matmul dispatch, (I + A)^-1
 * and the gate factors on the host. Per-channel gates split around the chunk's middle (16-token chunks);
 * chunks whose gates exceed float range take host steps.
 ******************************************************************************/
#ifndef AIEDNN_RT_LINATTN_H
#define AIEDNN_RT_LINATTN_H

typedef struct {
    int64_t B, H, T, Dk, Dv;
    int per_ch, delta;
    double scale;
} la_geo;

static void la_geo_of(const rt_prob *p, la_geo *L) {
    const aiednn_memory_desc_t *q = rt_md(p, AIEDNN_ARG_SRC), *v = rt_md(p, AIEDNN_ARG_SRC_2);
    L->B = q->dims[0], L->H = q->dims[1], L->T = q->dims[2], L->Dk = q->dims[3], L->Dv = v->dims[3];
    L->per_ch = (p->flags & AIEDNN_LA_GATE_PER_CHANNEL) != 0;
    L->delta = !(p->flags & AIEDNN_LA_NO_DELTA);
    L->scale = (double)p->scale;
}

/* the log gate of token t (row = (b H + h) T + t) for key channel d; 0 without a gate */
static inline double la_gate(const rt_prob *p, void *const *ptr, const la_geo *L, int64_t row, int64_t d) {
    const aiednn_memory_desc_t *g = rt_md(p, AIEDNN_ARG_GATE);
    if (!g)
        return 0.0;
    return rt_ld(g->data_type, rt_ptr(p, ptr, AIEDNN_ARG_GATE), L->per_ch ? row * L->Dk + d : row);
}

/* tokens [t0, t1) of head bh on the host in double: the reference recurrence (ref:, and the composite's
 * chunks whose gates float cannot carry); h [Dk][Dv] */
static void la_steps(const rt_prob *p, void *const *ptr, const la_geo *L, int64_t bh, int64_t t0, int64_t t1, double *h,
                     double *u) {
    const aiednn_memory_desc_t *q = rt_md(p, AIEDNN_ARG_SRC), *k = rt_md(p, AIEDNN_ARG_SRC_1),
                               *v = rt_md(p, AIEDNN_ARG_SRC_2), *d = rt_md(p, AIEDNN_ARG_DST),
                               *be = rt_md(p, AIEDNN_ARG_BETA);
    const void *qp = rt_ptr(p, ptr, AIEDNN_ARG_SRC), *kp = rt_ptr(p, ptr, AIEDNN_ARG_SRC_1),
               *vp = rt_ptr(p, ptr, AIEDNN_ARG_SRC_2), *bp = be ? rt_ptr(p, ptr, AIEDNN_ARG_BETA) : nullptr;
    void *dp = rt_ptr(p, ptr, AIEDNN_ARG_DST);
    const int64_t Dk = L->Dk, Dv = L->Dv;
    for (int64_t t = t0; t < t1; t++) {
        const int64_t row = bh * L->T + t;
        for (int64_t c = 0; c < Dk; c++) {
            const double a = exp(la_gate(p, ptr, L, row, c));
            for (int64_t e = 0; e < Dv; e++)
                h[c * Dv + e] *= a;
        }
        const double beta = L->delta && bp ? rt_ld(be->data_type, bp, row) : 1.0;
        for (int64_t e = 0; e < Dv; e++) {
            double s = 0;
            if (L->delta)
                for (int64_t c = 0; c < Dk; c++)
                    s += h[c * Dv + e] * rt_ld(k->data_type, kp, row * Dk + c);
            u[e] = beta * (rt_ld(v->data_type, vp, row * Dv + e) - s);
            if (!L->delta)
                u[e] = rt_ld(v->data_type, vp, row * Dv + e);
        }
        for (int64_t c = 0; c < Dk; c++) {
            const double kc = rt_ld(k->data_type, kp, row * Dk + c);
            for (int64_t e = 0; e < Dv; e++)
                h[c * Dv + e] += kc * u[e];
        }
        for (int64_t e = 0; e < Dv; e++) {
            double s = 0;
            for (int64_t c = 0; c < Dk; c++)
                s += h[c * Dv + e] * rt_ld(q->data_type, qp, row * Dk + c);
            rt_st(d->data_type, dp, row * Dv + e, L->scale * s);
        }
    }
}

static int run_ref_linattn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    la_geo L;
    la_geo_of(p, &L);
    const aiednn_memory_desc_t *s0 = rt_md(p, AIEDNN_ARG_STATE), *s1 = rt_md(p, AIEDNN_ARG_STATE_OUT);
    double *h = (double *)rt_alloc((size_t)(L.Dk * L.Dv) * sizeof(double)),
           *u = (double *)rt_alloc((size_t)L.Dv * sizeof(double));
    if (!h || !u) {
        rt_free(h), rt_free(u);
        return AIEK_ERR_ALLOC;
    }
    const int64_t hs = L.Dk * L.Dv;
    for (int64_t bh = 0; bh < L.B * L.H; bh++) {
        for (int64_t i = 0; i < hs; i++)
            h[i] = s0 ? rt_ld(s0->data_type, rt_ptr(p, ptr, AIEDNN_ARG_STATE), bh * hs + i) : 0.0;
        la_steps(p, ptr, &L, bh, 0, L.T, h, u);
        for (int64_t i = 0; s1 && i < hs; i++)
            rt_st(s1->data_type, rt_ptr(p, ptr, AIEDNN_ARG_STATE_OUT), bh * hs + i, h[i]);
    }
    rt_free(h), rt_free(u);
    return AIEK_OK;
}

// ── gemm+linattn: the chunked form ──────────────────────────────────────────────

#define LA_CHUNK 64
#define LA_CHUNK_PC 16
#define LA_MAX_SPAN 80.0 /* a per-channel chunk's gate span (log) float carries split around its middle */

static const gx_types g_la_t = {aiednn_f32, aiednn_f32, aiednn_f32};

static int acc_gx_linattn(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    for (int s = 0; s < p->nmd; s++)
        if (p->md[s].data_type != aiednn_f32) {
            snprintf(why, cap, "f32 operands only (ref: takes the other floating types)");
            return 0;
        }
    la_geo L;
    la_geo_of(p, &L);
    const int64_t C = L.per_ch ? LA_CHUNK_PC : LA_CHUNK;
    rt_bind b2 = {nullptr, nullptr}, b3 = {nullptr, nullptr}, b4 = {nullptr, nullptr};
    /* the stacked GEMMs of la_chunk: [K; Q] h0, [Kd; Qd] Kn^T, (I + A)^-1 R, [QK; Ke^T] U */
    return gx_inner_ok(&g_la_t, 2 * C, L.Dv, L.Dk, bind, why, cap) &&
           gx_inner_ok(&g_la_t, 2 * C, C, L.Dk, &b2, why, cap, 1) && gx_inner_ok(&g_la_t, C, L.Dv, C, &b3, why, cap) &&
           gx_inner_ok(&g_la_t, C + L.Dk, L.Dv, C, &b4, why, cap);
}

/* per-chunk buffers; GEMM operand pairs stacked so one GEMM serves both */
typedef struct {
    float *A1; /* [2n][Dk]: K e^{G} over Q e^{G} (against h0) */
    float *A2; /* [2n][Dk]: Kd over Qd (against Kn^T) */
    float *Kn; /* [n][Dk] */
    float *C1; /* [2n][Dv]: KH over QH */
    float *C2; /* [2n][n]: KK over QK */
    float *A3; /* [n + Dk][n]: masked QK over Ke^T */
    float *C3; /* [n + Dk][Dv]: O over Ke^T U */
    float *R, *U, *Tm, *h0;
    double *G, *Gd, *hd, *ud; /* [C][Dk] cumulative log gates, [C] column scratch, host-step state */
} la_bufs;

/* the gate-weighted copies of a chunk's rows (see the header): returns 0 when the per-channel gates span too
 * much for float */
static int la_chunk_rows(const la_geo *L, const float *q, const float *k, int64_t n, la_bufs *B) {
    const int64_t Dk = L->Dk;
    for (int64_t c = 0; L->per_ch && c < Dk; c++)
        if (!(B->G[c] - B->G[(n - 1) * Dk + c] <= LA_MAX_SPAN))
            return 0;
    float *Kg = B->A1, *Qg = B->A1 + n * Dk, *Kd = B->A2, *Qd = B->A2 + n * Dk, *Ke = B->A3 + n * n;
    for (int64_t i = 0; i < n; i++)
        for (int64_t c = 0; c < Dk; c++) {
            const double g = B->G[i * Dk + c], gn = B->G[(n - 1) * Dk + c], mid = B->G[(n / 2) * Dk + c];
            const float kv = k[i * Dk + c], qv = q[i * Dk + c];
            Kg[i * Dk + c] = (float)(kv * exp(g)), Qg[i * Dk + c] = (float)(qv * exp(g));
            Ke[c * n + i] = (float)(kv * exp(gn - g));
            if (L->per_ch) {
                Kd[i * Dk + c] = (float)(kv * exp(g - mid)), Qd[i * Dk + c] = (float)(qv * exp(g - mid));
                B->Kn[i * Dk + c] = (float)(kv * exp(mid - g));
            } else
                Kd[i * Dk + c] = kv, Qd[i * Dk + c] = qv, B->Kn[i * Dk + c] = kv;
        }
    return 1;
}

/* (I + A)^-1, A strictly lower (A_ij = beta_i KK_ij [scalar gate: x e^{G_i - G_j}]), into Tm, in double */
static void la_tri_inverse(const la_geo *L, const float *beta, int64_t n, la_bufs *B) {
    const int64_t Dk = L->Dk;
    const float *KK = B->C2;
    double *col = B->Gd; /* n doubles */
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) {
            double s = i == j ? 1.0 : 0.0;
            for (int64_t m = j; m < i; m++) {
                double a = (double)beta[i] * KK[i * n + m];
                if (!L->per_ch)
                    a *= exp(B->G[i * Dk] - B->G[m * Dk]);
                s -= a * col[m];
            }
            col[i] = i < j ? 0.0 : s;
        }
        for (int64_t i = 0; i < n; i++)
            B->Tm[i * n + j] = (float)col[i];
    }
}

/* one chunk [t0, t0 + n) of head bh: O rows into dst, h0 advanced */
static int la_chunk(const rt_prob *p, void *const *ptr, const la_geo *L, int64_t bh, int64_t t0, int64_t n,
                    la_bufs *B) {
    const int64_t Dk = L->Dk, Dv = L->Dv, row0 = bh * L->T + t0;
    const float *q = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC) + row0 * Dk;
    const float *k = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC_1) + row0 * Dk;
    const float *v = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC_2) + row0 * Dv;
    const float *beta = rt_md(p, AIEDNN_ARG_BETA) ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_BETA) + row0 : nullptr;
    float *o = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DST) + row0 * Dv;
    for (int64_t c = 0; c < Dk; c++) /* cumulative log gates (a scalar gate: every channel alike) */
        for (int64_t i = 0; i < n; i++)
            B->G[i * Dk + c] = (i ? B->G[(i - 1) * Dk + c] : 0.0) + la_gate(p, ptr, L, row0 + i, c);
    if (!la_chunk_rows(L, q, k, n, B)) { /* host steps, in double, from h0 */
        for (int64_t i = 0; i < Dk * Dv; i++)
            B->hd[i] = B->h0[i];
        la_steps(p, ptr, L, bh, t0, t0 + n, B->hd, B->ud);
        for (int64_t i = 0; i < Dk * Dv; i++)
            B->h0[i] = (float)B->hd[i];
        return AIEK_OK;
    }
    const gx_types *t = &g_la_t;
    int rc = gx_gemm(t, 2 * n, Dv, Dk, B->A1, B->h0, B->C1);         /* KH, QH */
    rc = rc ? rc : gx_gemm(t, 2 * n, n, Dk, B->A2, B->Kn, B->C2, 1); /* KK, QK */
    if (rc)
        return rc;
    const float *KH = B->C1, *QH = B->C1 + n * Dv, *QK = B->C2 + n * n, *U = v;
    if (L->delta) { /* U = (I + A)^-1 R */
        for (int64_t i = 0; i < n; i++)
            for (int64_t e = 0; e < Dv; e++)
                B->R[i * Dv + e] = (float)((double)beta[i] * ((double)v[i * Dv + e] - KH[i * Dv + e]));
        la_tri_inverse(L, beta, n, B);
        rc = gx_gemm(t, n, Dv, n, B->Tm, B->R, B->U);
        if (rc)
            return rc;
        U = B->U;
    }
    for (int64_t i = 0; i < n; i++) /* (Q K^T) masked to j <= i, gate-weighted, over Ke^T (la_chunk_rows) */
        for (int64_t j = 0; j < n; j++)
            B->A3[i * n + j] = j > i       ? 0.0f
                               : L->per_ch ? QK[i * n + j]
                                           : (float)(QK[i * n + j] * exp(B->G[i * Dk] - B->G[j * Dk]));
    rc = gx_gemm(t, n + Dk, Dv, n, B->A3, U, B->C3); /* O, Ke^T U */
    if (rc)
        return rc;
    for (int64_t i = 0; i < n * Dv; i++)
        o[i] = (float)(L->scale * ((double)QH[i] + B->C3[i]));
    const float *KU = B->C3 + n * Dv;
    for (int64_t c = 0; c < Dk; c++) {
        const double a = exp(B->G[(n - 1) * Dk + c]);
        for (int64_t e = 0; e < Dv; e++)
            B->h0[c * Dv + e] = (float)(a * B->h0[c * Dv + e] + KU[c * Dv + e]);
    }
    return AIEK_OK;
}

static int run_gx_linattn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    la_geo L;
    la_geo_of(p, &L);
    const int64_t C = L.per_ch ? LA_CHUNK_PC : LA_CHUNK, Dk = L.Dk, Dv = L.Dv;
    const size_t f = sizeof(float);
    la_bufs B;
    B.A1 = (float *)rt_alloc((size_t)(2 * C * Dk) * f), B.A2 = (float *)rt_alloc((size_t)(2 * C * Dk) * f);
    B.Kn = (float *)rt_alloc((size_t)(C * Dk) * f), B.C1 = (float *)rt_alloc((size_t)(2 * C * Dv) * f);
    B.C2 = (float *)rt_alloc((size_t)(2 * C * C) * f), B.A3 = (float *)rt_alloc((size_t)((C + Dk) * C) * f);
    B.C3 = (float *)rt_alloc((size_t)((C + Dk) * Dv) * f), B.R = (float *)rt_alloc((size_t)(C * Dv) * f);
    B.U = (float *)rt_alloc((size_t)(C * Dv) * f), B.Tm = (float *)rt_alloc((size_t)(C * C) * f);
    B.h0 = (float *)rt_alloc((size_t)(Dk * Dv) * f);
    B.G = (double *)rt_alloc((size_t)(C * Dk) * sizeof(double)), B.Gd = (double *)rt_alloc((size_t)C * sizeof(double));
    B.hd = (double *)rt_alloc((size_t)(Dk * Dv) * sizeof(double)),
    B.ud = (double *)rt_alloc((size_t)Dv * sizeof(double));
    void *all[] = {B.A1, B.A2, B.Kn, B.C1, B.C2, B.A3, B.C3, B.R, B.U, B.Tm, B.h0, B.G, B.Gd, B.hd, B.ud};
    int rc = AIEK_OK;
    for (void *b : all)
        rc = b ? rc : AIEK_ERR_ALLOC;
    const float *s0 = rt_md(p, AIEDNN_ARG_STATE) ? (const float *)rt_ptr(p, ptr, AIEDNN_ARG_STATE) : nullptr;
    float *s1 = rt_md(p, AIEDNN_ARG_STATE_OUT) ? (float *)rt_ptr(p, ptr, AIEDNN_ARG_STATE_OUT) : nullptr;
    for (int64_t bh = 0; rc == AIEK_OK && bh < L.B * L.H; bh++) {
        for (int64_t i = 0; i < Dk * Dv; i++)
            B.h0[i] = s0 ? s0[bh * Dk * Dv + i] : 0.0f;
        for (int64_t t0 = 0; rc == AIEK_OK && t0 < L.T; t0 += C)
            rc = la_chunk(p, ptr, &L, bh, t0, L.T - t0 < C ? L.T - t0 : C, &B);
        if (s1)
            memcpy(s1 + bh * Dk * Dv, B.h0, (size_t)(Dk * Dv) * f);
    }
    for (void *b : all)
        rt_free(b);
    return rc;
}

extern "C" aiednn_status_t aiednn_linear_attention_create(
    aiednn_primitive_t *pr, const aiednn_memory_desc_t *q, const aiednn_memory_desc_t *k, const aiednn_memory_desc_t *v,
    const aiednn_memory_desc_t *gate, const aiednn_memory_desc_t *beta, const aiednn_memory_desc_t *state_in,
    const aiednn_memory_desc_t *dst, const aiednn_memory_desc_t *state_out, float scale, unsigned flags) {
    const char *kd = "linear_attention";
    REQUIRE(pr && md_ok(q) && md_ok(k) && md_ok(v) && md_ok(dst) && q->ndims == 4 && md_same_shape(q, k) &&
                v->ndims == 4 && md_same_shape(v, dst) && v->dims[0] == q->dims[0] && v->dims[1] == q->dims[1] &&
                v->dims[2] == q->dims[2],
            kd, "q, k [B, H, T, Dk]; v, dst [B, H, T, Dv]");
    REQUIRE(!(flags & ~(AIEDNN_LA_GATE_PER_CHANNEL | AIEDNN_LA_NO_DELTA)), kd, "unknown flag");
    const int64_t B = q->dims[0], H = q->dims[1], T = q->dims[2], Dk = q->dims[3], Dv = v->dims[3];
    REQUIRE(!gate ||
                (md_ok(gate) && ((flags & AIEDNN_LA_GATE_PER_CHANNEL) ? md_same_shape(gate, q)
                                                                      : gate->ndims == 3 && gate->dims[0] == B &&
                                                                            gate->dims[1] == H && gate->dims[2] == T)),
            kd, "gate [B, H, T] (per channel: [B, H, T, Dk]), log values");
    REQUIRE((flags & AIEDNN_LA_NO_DELTA) ? !beta
                                         : beta && md_ok(beta) && beta->ndims == 3 && beta->dims[0] == B &&
                                               beta->dims[1] == H && beta->dims[2] == T,
            kd, "beta [B, H, T] with the delta rule; none with AIEDNN_LA_NO_DELTA");
    const aiednn_memory_desc_t *st[2] = {state_in, state_out};
    for (const aiednn_memory_desc_t *s : st)
        REQUIRE(!s || (md_ok(s) && s->ndims == 4 && s->dims[0] == B && s->dims[1] == H && s->dims[2] == Dk &&
                       s->dims[3] == Dv),
                kd, "states [B, H, Dk, Dv]");
    const aiednn_memory_desc_t *all[8] = {q, k, v, dst, gate, beta, state_in, state_out};
    for (const aiednn_memory_desc_t *m : all)
        REQUIRE(!m || dt_is_float(m->data_type), kd, "floating types");
    rt_prob p;
    prob_init(&p, aiednn_linear_attention, 0);
    prob_add(&p, AIEDNN_ARG_SRC, q);
    prob_add(&p, AIEDNN_ARG_SRC_1, k);
    prob_add(&p, AIEDNN_ARG_SRC_2, v);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    if (gate)
        prob_add(&p, AIEDNN_ARG_GATE, gate);
    if (beta)
        prob_add(&p, AIEDNN_ARG_BETA, beta);
    if (state_in)
        prob_add(&p, AIEDNN_ARG_STATE, state_in);
    if (state_out)
        prob_add(&p, AIEDNN_ARG_STATE_OUT, state_out);
    p.flags = flags;
    p.scale = scale;
    return create(pr, &p);
}

#endif /* AIEDNN_RT_LINATTN_H */
