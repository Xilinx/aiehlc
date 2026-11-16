/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_rnn_bwd.h -- RNN backward (back-propagation through time), ref:, in double.
 *
 * The forward is recomputed keeping every step's state, then each (layer, direction) is walked against
 * its processing order, accumulating dW / dU / db and propagating to dx and the previous state.
 *
 * With rnnb_ctx.mm set (gemm+rnnbwd, aiednn_rt_gemmx.h) the non-recurrent products (W x, dW, dx, dU) run
 * as GEMMs over all T x N steps; the loops keep the gates, dB and the recurrence nh = U dA.
 ******************************************************************************/
#ifndef AIEDNN_RT_RNN_BWD_H
#define AIEDNN_RT_RNN_BWD_H

typedef struct {
    const rt_prob *p;
    void *const *ptr;
    int cell, G, Gb, dir;
    int64_t T, N, H, L, D, SLC, DLC;
    /* C[M][N] = A[M][K] B[K][N] (bt: B given as [N][K]) in f32, or null for the double loops */
    int (*mm)(int64_t M, int64_t N, int64_t K, const float *A, const float *B, float *C, int bt);
} rnnb_ctx;

static double rnnb_ld(const rnnb_ctx *c, int arg, int64_t i, double dflt) {
    const int s = rt_slot(c->p, arg);
    return s < 0 ? dflt : rt_ld(c->p->md[s].data_type, c->ptr[s], i);
}

/* forward of (l, d) over its input seq [T][N][I] keeping per step: hs [T][N][H] (output h), cs (LSTM c),
 * act [T][N][G][H] (gate activations), aux [T][N][H] (GRU: U_o h + b for lbr, r * h for gru) */
static void rnnb_fwd(const rnnb_ctx *c, int64_t l, int64_t d, int rev, int64_t I, const double *in, double *hs,
                     double *cs, double *act, double *aux, const float *xw /* W x of every step, or null */) {
    const int64_t N = c->N, H = c->H, G = c->G, ld = l * c->D + d;
    double *h = (double *)rt_alloc(sizeof(double) * (size_t)(2 * N * H + G * H));
    double *cc = h + N * H, *a = cc + N * H;
    for (int64_t i = 0; i < N * H; i++) {
        h[i] = rnnb_ld(c, AIEDNN_ARG_SRC_ITER, ld * N * H + i, 0.0);
        cc[i] = rnnb_ld(c, AIEDNN_ARG_SRC_ITER_C, ld * N * H + i, 0.0);
    }
    auto W = [&](int64_t k, int64_t g, int64_t j) {
        return rnnb_ld(c, AIEDNN_ARG_WEIGHTS, ((ld * I + k) * G + g) * H + j, 0);
    };
    auto U = [&](int64_t k, int64_t g, int64_t j) {
        return rnnb_ld(c, AIEDNN_ARG_WEIGHTS_ITER, ((ld * H + k) * G + g) * H + j, 0);
    };
    auto B = [&](int64_t g, int64_t j) { return rnnb_ld(c, AIEDNN_ARG_BIAS, (ld * c->Gb + g) * H + j, 0.0); };
    const int gru = c->cell == aiednn_rnn_gru || c->cell == aiednn_rnn_lbr_gru;
    for (int64_t s = 0; s < c->T; s++) {
        const int64_t t = rev ? c->T - 1 - s : s;
        for (int64_t n = 0; n < N; n++) {
            const double *x = in + (t * N + n) * I, *hp = h + n * H;
            for (int64_t g = 0; g < G; g++)
                for (int64_t j = 0; j < H; j++) {
                    double v = B(g, j);
                    if (xw)
                        v += xw[(t * N + n) * G * H + g * H + j];
                    else
                        for (int64_t k = 0; k < I; k++)
                            v += x[k] * W(k, g, j);
                    if (!(gru && g == 2))
                        for (int64_t k = 0; k < H; k++)
                            v += hp[k] * U(k, g, j);
                    a[g * H + j] = v;
                }
            double *A = act + (t * N + n) * G * H, *X = aux + (t * N + n) * H;
            for (int64_t j = 0; j < H; j++) {
                if (c->cell == aiednn_rnn_lstm) {
                    A[j] = rnn_sig(a[j]), A[H + j] = rnn_sig(a[H + j]), A[2 * H + j] = tanh(a[2 * H + j]);
                    A[3 * H + j] = rnn_sig(a[3 * H + j]);
                } else if (gru) {
                    A[j] = rnn_sig(a[j]), A[H + j] = rnn_sig(a[H + j]);
                } else {
                    const double v = a[j];
                    A[j] = c->cell == aiednn_rnn_vanilla_tanh       ? tanh(v)
                           : c->cell == aiednn_rnn_vanilla_logistic ? rnn_sig(v)
                           : v > 0                                  ? v
                                                                    : (double)c->p->alpha * v;
                    X[j] = v; /* the pre-activation: ReLU's derivative needs its sign */
                }
            }
            if (gru)
                for (int64_t j = 0; j < H; j++) {
                    double o = a[2 * H + j];
                    if (c->cell == aiednn_rnn_lbr_gru) {
                        double n_ = B(3, j);
                        for (int64_t k = 0; k < H; k++)
                            n_ += hp[k] * U(k, 2, j);
                        X[j] = n_;
                        o += A[H + j] * n_;
                    } else {
                        for (int64_t k = 0; k < H; k++)
                            o += A[H + k] * hp[k] * U(k, 2, j);
                    }
                    A[2 * H + j] = tanh(o);
                }
            for (int64_t j = 0; j < H; j++) {
                double v;
                if (c->cell == aiednn_rnn_lstm) {
                    cc[n * H + j] = A[H + j] * cc[n * H + j] + A[j] * A[2 * H + j];
                    v = A[3 * H + j] * tanh(cc[n * H + j]);
                    cs[(t * N + n) * H + j] = cc[n * H + j];
                } else if (gru) {
                    v = A[j] * hp[j] + (1 - A[j]) * A[2 * H + j];
                } else {
                    v = A[j];
                }
                hs[(t * N + n) * H + j] = v;
            }
            for (int64_t j = 0; j < H; j++)
                h[n * H + j] = hs[(t * N + n) * H + j];
        }
    }
    rt_free(h);
}

/* f32 copies of a layer input X [T N][I] and of (l, d)'s W [I][G H] and U [H][G H], and the forward's W x */
typedef struct {
    float *X, *W, *U, *XW;
} rnnb_f32;

static int rnnb_f32_make(const rnnb_ctx *c, int64_t ld, int64_t I, const double *in, rnnb_f32 *f) {
    const int64_t TN = c->T * c->N, GH = c->G * c->H;
    f->X = (float *)rt_alloc(sizeof(float) * (size_t)(TN * I + I * GH + c->H * GH + TN * GH));
    if (!f->X)
        return AIEK_ERR_ALLOC;
    f->W = f->X + TN * I, f->U = f->W + I * GH, f->XW = f->U + c->H * GH;
    for (int64_t i = 0; i < TN * I; i++)
        f->X[i] = (float)in[i];
    for (int64_t i = 0; i < I * GH; i++)
        f->W[i] = (float)rnnb_ld(c, AIEDNN_ARG_WEIGHTS, ld * I * GH + i, 0.0);
    for (int64_t i = 0; i < c->H * GH; i++)
        f->U[i] = (float)rnnb_ld(c, AIEDNN_ARG_WEIGHTS_ITER, ld * c->H * GH + i, 0.0);
    return c->mm(TN, GH, I, f->X, f->W, f->XW, 0);
}

/* dW += X^T dA, dx += dA W^T, dU += Hp^T dA' (dA' without GRU's candidate columns) as GEMMs */
static int rnnb_bwd_gemms(const rnnb_ctx *c, int64_t I, const rnnb_f32 *f, const float *dA, const float *Hp, double *dW,
                          double *dU, double *dx) {
    const int64_t TN = c->T * c->N, GH = c->G * c->H, H = c->H;
    const int gru = c->cell == aiednn_rnn_gru || c->cell == aiednn_rnn_lbr_gru;
    float *t =
        (float *)rt_alloc(sizeof(float) * (size_t)(TN * (I > H ? I : H) + TN * GH + (I > H ? I : H) * GH + TN * I));
    if (!t)
        return AIEK_ERR_ALLOC;
    float *dA2 = t + TN * (I > H ? I : H), *C = dA2 + TN * GH, *C2 = C + (I > H ? I : H) * GH;
    for (int64_t r = 0; r < TN; r++) /* X^T */
        for (int64_t k = 0; k < I; k++)
            t[k * TN + r] = f->X[r * I + k];
    int rc = c->mm(I, GH, TN, t, dA, C, 0);
    for (int64_t i = 0; rc == AIEK_OK && i < I * GH; i++)
        dW[i] += C[i];
    if (rc == AIEK_OK) /* dx = dA W^T: W [I][G H] is B transposed */
        rc = c->mm(TN, I, GH, dA, f->W, C2, 1);
    for (int64_t i = 0; rc == AIEK_OK && i < TN * I; i++)
        dx[i] += C2[i];
    for (int64_t r = 0; rc == AIEK_OK && r < TN; r++) { /* Hp^T, and dA without GRU's candidate gate */
        for (int64_t k = 0; k < H; k++)
            t[k * TN + r] = Hp[r * H + k];
        for (int64_t j = 0; j < GH; j++)
            dA2[r * GH + j] = gru && j >= 2 * H && j < 3 * H ? 0.0f : dA[r * GH + j];
    }
    if (rc == AIEK_OK)
        rc = c->mm(H, GH, TN, t, dA2, C, 0);
    for (int64_t i = 0; rc == AIEK_OK && i < H * GH; i++)
        dU[i] += C[i];
    rt_free(t);
    return rc;
}

/* the backward walk of (l, d) against its processing order: dx (+=), the (l, d) parameter gradients and the
 * final state gradients written; dh_out the gradient of the layer's output */
static int rnnb_bwd_ld(const rnnb_ctx *c, int64_t l, int64_t d, int64_t I, const double *in, const double *st,
                       const double *dh_out, double *dx, const rnnb_f32 *f) {
    const rt_prob *p = c->p;
    const int64_t T = c->T, N = c->N, H = c->H, G = c->G, D = c->D, DLC = c->DLC, per = T * N * H, GH = G * H;
    const int gru = c->cell == aiednn_rnn_gru || c->cell == aiednn_rnn_lbr_gru, lstm = c->cell == aiednn_rnn_lstm;
    const int concat = c->dir == aiednn_rnn_bidirectional_concat, rev = c->dir == aiednn_rnn_right2left || d == 1;
    const int64_t ld = l * D + d;
    const double *hs = st, *cs = hs + per, *aux = cs + per, *act = aux + per;
    auto W = [&](int64_t k, int64_t g, int64_t j) {
        return rnnb_ld(c, AIEDNN_ARG_WEIGHTS, ((ld * I + k) * G + g) * H + j, 0);
    };
    auto U = [&](int64_t k, int64_t g, int64_t j) {
        return f ? (double)f->U[(k * G + g) * H + j]
                 : rnnb_ld(c, AIEDNN_ARG_WEIGHTS_ITER, ((ld * H + k) * G + g) * H + j, 0);
    };
    double *dW = (double *)rt_calloc((size_t)(I * GH + H * GH + c->Gb * H + 4 * N * H + GH + 3 * H), sizeof(double));
    float *dA = f ? (float *)rt_alloc(sizeof(float) * (size_t)(T * N * (GH + H))) : NULL;
    if (!dW || (f && !dA)) {
        rt_free(dW), rt_free(dA);
        return AIEK_ERR_ALLOC;
    }
    float *Hp = dA ? dA + T * N * GH : NULL;
    double *dU = dW + I * GH, *dB = dU + H * GH, *dh = dB + c->Gb * H, *dc = dh + N * H, *dhn = dc + N * H;
    double *dcn = dhn + N * H, *da = dcn + N * H, *drh = da + GH, *hp = drh + H, *cp = hp + H;
    for (int64_t i = 0; i < N * H; i++) {
        dh[i] = rnnb_ld(c, AIEDNN_ARG_DIFF_DST_ITER, ld * N * H + i, 0.0);
        dc[i] = rnnb_ld(c, AIEDNN_ARG_DIFF_DST_ITER_C, ld * N * H + i, 0.0);
    }
    for (int64_t s = T - 1; s >= 0; s--) {
        const int64_t t = rev ? T - 1 - s : s, tp = rev ? t + 1 : t - 1; /* tp: the step before */
        const int first = s == 0;
        for (int64_t n = 0; n < N; n++) {
            const double *x = in + (t * N + n) * I, *A = act + (t * N + n) * GH, *X = aux + (t * N + n) * H;
            for (int64_t j = 0; j < H; j++) {
                hp[j] = first ? rnnb_ld(c, AIEDNN_ARG_SRC_ITER, (ld * N + n) * H + j, 0.0) : hs[(tp * N + n) * H + j];
                cp[j] = first ? rnnb_ld(c, AIEDNN_ARG_SRC_ITER_C, (ld * N + n) * H + j, 0.0) : cs[(tp * N + n) * H + j];
            }
            double *gh = dh + n * H, *gc = dc + n * H, *nh = dhn + n * H, *nc = dcn + n * H;
            for (int64_t j = 0; j < H; j++) /* this step's output gradient from above */
                gh[j] += dh_out[(t * N + n) * DLC + (concat ? d * H : 0) + j];
            for (int64_t j = 0; j < H; j++)
                nh[j] = 0, nc[j] = 0;
            if (lstm) {
                for (int64_t j = 0; j < H; j++) {
                    const double ig = A[j], fg = A[H + j], gg = A[2 * H + j], og = A[3 * H + j];
                    const double tc = tanh(cs[(t * N + n) * H + j]);
                    const double dct = gc[j] + gh[j] * og * (1 - tc * tc);
                    da[j] = dct * gg * ig * (1 - ig);
                    da[H + j] = dct * cp[j] * fg * (1 - fg);
                    da[2 * H + j] = dct * ig * (1 - gg * gg);
                    da[3 * H + j] = gh[j] * tc * og * (1 - og);
                    nc[j] = dct * fg;
                }
            } else if (gru) {
                for (int64_t j = 0; j < H; j++) {
                    const double u = A[j], o = A[2 * H + j];
                    da[j] = gh[j] * (hp[j] - o) * u * (1 - u);
                    da[2 * H + j] = gh[j] * (1 - u) * (1 - o * o);
                    nh[j] += gh[j] * u;
                }
                if (c->cell == aiednn_rnn_lbr_gru) {
                    for (int64_t j = 0; j < H; j++) { /* o's pre = W x + b + r * n, n = U_o h + b' */
                        const double r = A[H + j];
                        da[H + j] = da[2 * H + j] * X[j] * r * (1 - r);
                        drh[j] = da[2 * H + j] * r; /* d n */
                        dB[3 * H + j] += drh[j];
                    }
                    for (int64_t k = 0; k < H; k++)
                        for (int64_t j = 0; j < H; j++) {
                            nh[k] += U(k, 2, j) * drh[j];
                            dU[(k * G + 2) * H + j] += hp[k] * drh[j];
                        }
                } else {
                    for (int64_t k = 0; k < H; k++) { /* o's pre = W x + U_o (r * h) + b */
                        double g = 0;
                        for (int64_t j = 0; j < H; j++)
                            g += U(k, 2, j) * da[2 * H + j];
                        const double r = A[H + k];
                        drh[k] = g; /* d (r_k h_k) */
                        nh[k] += g * r;
                        da[H + k] = g * hp[k] * r * (1 - r);
                        for (int64_t j = 0; j < H; j++)
                            dU[(k * G + 2) * H + j] += r * hp[k] * da[2 * H + j];
                    }
                }
            } else {
                for (int64_t j = 0; j < H; j++) {
                    const double hv = A[j];
                    const double dact = c->cell == aiednn_rnn_vanilla_tanh       ? 1 - hv * hv
                                        : c->cell == aiednn_rnn_vanilla_logistic ? hv * (1 - hv)
                                        : X[j] > 0                               ? 1.0
                                                                                 : (double)p->alpha;
                    da[j] = gh[j] * dact;
                }
            }
            if (f) /* the GEMMs after the loop take dW, dx and dU's generic part */
                for (int64_t j = 0; j < H; j++)
                    Hp[(t * N + n) * H + j] = (float)hp[j];
            for (int64_t g = 0; g < G; g++)
                for (int64_t j = 0; j < H; j++) {
                    const double v = da[g * H + j];
                    dB[g * H + j] += v;
                    if (f)
                        dA[(t * N + n) * GH + g * H + j] = (float)v;
                    else
                        for (int64_t k = 0; k < I; k++) {
                            dW[(k * G + g) * H + j] += x[k] * v;
                            dx[(t * N + n) * I + k] += W(k, g, j) * v;
                        }
                    if (gru && g == 2)
                        continue;
                    for (int64_t k = 0; k < H; k++) {
                        if (!f)
                            dU[(k * G + g) * H + j] += hp[k] * v;
                        nh[k] += U(k, g, j) * v;
                    }
                }
        }
        for (int64_t i = 0; i < N * H; i++)
            dh[i] = dhn[i], dc[i] = dcn[i];
    }
    int rc = f ? rnnb_bwd_gemms(c, I, f, dA, Hp, dW, dU, dx) : AIEK_OK;
    auto set = [&](int arg, int64_t i, double v) {
        const int sl = rt_slot(p, arg);
        if (sl >= 0)
            rt_st(p->md[sl].data_type, c->ptr[sl], i, v);
    };
    for (int64_t i = 0; rc == AIEK_OK && i < N * H; i++) {
        set(AIEDNN_ARG_DIFF_SRC_ITER, ld * N * H + i, dh[i]);
        set(AIEDNN_ARG_DIFF_SRC_ITER_C, ld * N * H + i, dc[i]);
    }
    for (int64_t i = 0; rc == AIEK_OK && i < I * GH; i++)
        set(AIEDNN_ARG_DIFF_WEIGHTS, ld * I * GH + i, dW[i]);
    for (int64_t i = 0; rc == AIEK_OK && i < H * GH; i++)
        set(AIEDNN_ARG_DIFF_WEIGHTS_ITER, ld * H * GH + i, dU[i]);
    for (int64_t i = 0; rc == AIEK_OK && i < c->Gb * H; i++)
        set(AIEDNN_ARG_DIFF_BIAS, ld * c->Gb * H + i, dB[i]);
    rt_free(dW), rt_free(dA);
    return rc;
}

static int run_ref_rnn_bwd(const rnnb_ctx *c) {
    const rt_prob *p = c->p;
    const int64_t T = c->T, N = c->N, H = c->H, G = c->G, L = c->L, D = c->D, SLC = c->SLC, DLC = c->DLC;
    const int bisum = c->dir == aiednn_rnn_bidirectional_sum, concat = c->dir == aiednn_rnn_bidirectional_concat;
    const int64_t W_ = SLC > DLC ? SLC : DLC, per = T * N * H;
    /* layer inputs (L + 1 sequences) and every (l, d)'s step states */
    double *seq = (double *)rt_calloc((size_t)((L + 1) * T * N * W_), sizeof(double));
    double *st = (double *)rt_calloc((size_t)(L * D * per * (G + 3)), sizeof(double));
    double *dseq = (double *)rt_calloc((size_t)(2 * T * N * W_), sizeof(double));
    rnnb_f32 *fs = c->mm ? (rnnb_f32 *)rt_calloc((size_t)(L * D), sizeof(rnnb_f32)) : NULL;
    int rc = seq && st && dseq && (!c->mm || fs) ? AIEK_OK : AIEK_ERR_ALLOC;
    auto HS = [&](int64_t l, int64_t d) { return st + (l * D + d) * per * (G + 3); };
    for (int64_t i = 0; rc == AIEK_OK && i < T * N * SLC; i++)
        seq[i] = rnnb_ld(c, AIEDNN_ARG_SRC, i, 0.0);
    for (int64_t l = 0; rc == AIEK_OK && l < L; l++) {
        const int64_t I = l == 0 ? SLC : DLC;
        double *in = seq + l * T * N * W_, *out = seq + (l + 1) * T * N * W_;
        for (int64_t d = 0; rc == AIEK_OK && d < D; d++) {
            double *hs = HS(l, d), *cs = hs + per, *aux = cs + per, *act = aux + per;
            if (fs)
                rc = rnnb_f32_make(c, l * D + d, I, in, &fs[l * D + d]);
            if (rc != AIEK_OK)
                break;
            rnnb_fwd(c, l, d, c->dir == aiednn_rnn_right2left || d == 1, I, in, hs, cs, act, aux,
                     fs ? fs[l * D + d].XW : NULL);
            for (int64_t t = 0; t < T; t++)
                for (int64_t n = 0; n < N; n++)
                    for (int64_t j = 0; j < H; j++) {
                        double *o = out + (t * N + n) * DLC + (concat ? d * H : 0) + j;
                        *o = (bisum && d == 1) ? *o + hs[(t * N + n) * H + j] : hs[(t * N + n) * H + j];
                    }
        }
    }
    double *dh_out = dseq, *dx = dseq + T * N * W_; /* d(layer output), d(layer input) */
    for (int64_t i = 0; rc == AIEK_OK && i < T * N * DLC; i++)
        dh_out[i] = rnnb_ld(c, AIEDNN_ARG_DIFF_DST, i, 0.0);
    for (int64_t l = L - 1; rc == AIEK_OK && l >= 0; l--) {
        const int64_t I = l == 0 ? SLC : DLC;
        for (int64_t i = 0; i < T * N * I; i++)
            dx[i] = 0;
        for (int64_t d = 0; rc == AIEK_OK && d < D; d++)
            rc = rnnb_bwd_ld(c, l, d, I, seq + l * T * N * W_, HS(l, d), dh_out, dx, fs ? &fs[l * D + d] : NULL);
        if (l > 0) /* the layer below's output gradient is this layer's input gradient */
            for (int64_t i = 0; i < T * N * DLC; i++)
                dh_out[i] = dx[i];
    }
    for (int64_t i = 0; rc == AIEK_OK && i < T * N * SLC; i++) {
        const int sl = rt_slot(p, AIEDNN_ARG_DIFF_SRC);
        if (sl >= 0)
            rt_st(p->md[sl].data_type, c->ptr[sl], i, dx[i]);
    }
    for (int64_t i = 0; fs && i < L * D; i++)
        rt_free(fs[i].X);
    rt_free(seq), rt_free(st), rt_free(dseq), rt_free(fs);
    return rc;
}

static int run_ref_rnn_backward(const rt_prob *p, void *const *ptr) {
    const aiednn_memory_desc_t *sl = rt_md(p, AIEDNN_ARG_SRC), *wl = rt_md(p, AIEDNN_ARG_WEIGHTS);
    rnnb_ctx c;
    c.p = p, c.ptr = ptr, c.cell = p->alg, c.G = rnn_gates(p->alg), c.Gb = c.G + (p->alg == aiednn_rnn_lbr_gru);
    c.dir = (int)p->flags, c.T = sl->dims[0], c.N = sl->dims[1], c.SLC = sl->dims[2];
    c.L = wl->dims[0], c.D = wl->dims[1], c.H = wl->dims[4], c.DLC = rt_md(p, AIEDNN_ARG_DIFF_DST)->dims[2];
    c.mm = NULL;
    return run_ref_rnn_bwd(&c);
}

extern "C" aiednn_status_t aiednn_rnn_backward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t cell,
                                                      aiednn_rnn_direction_t dir, float alpha,
                                                      const aiednn_rnn_descs_t *f, const aiednn_rnn_descs_t *g) {
    const char *k = "rnn";
    REQUIRE(pr && f && g && g->src_layer && g->weights_layer && g->weights_iter && g->dst_layer, k,
            "forward and gradient descriptors (diff src_layer / weights_layer / weights_iter / dst_layer required)");
    /* validate the forward's shapes with the forward create, on a scratch primitive */
    aiednn_primitive_t chk = nullptr;
    const aiednn_status_t st =
        aiednn_rnn_forward_create(&chk, cell, dir, alpha, f->src_layer, f->src_iter, f->src_iter_c, f->weights_layer,
                                  f->weights_iter, f->bias, f->dst_layer, nullptr, nullptr);
    if (st != aiednn_success)
        return st;
    aiednn_primitive_destroy(chk);
    const aiednn_memory_desc_t *pairs[9][2] = {
        {f->src_layer, g->src_layer},         {f->src_iter, g->src_iter},         {f->src_iter_c, g->src_iter_c},
        {f->weights_layer, g->weights_layer}, {f->weights_iter, g->weights_iter}, {f->bias, g->bias},
        {f->dst_layer, g->dst_layer},         {f->dst_iter, g->dst_iter},         {f->dst_iter_c, g->dst_iter_c}};
    for (int i = 0; i < 9; i++)
        REQUIRE(!pairs[i][1] || (md_ok(pairs[i][1]) && (!pairs[i][0] || md_same_shape(pairs[i][0], pairs[i][1]))), k,
                "each gradient has its forward tensor's shape");
    rt_prob p;
    prob_init(&p, aiednn_rnn, cell);
    p.prop = RT_BWD;
    const int fa[9] = {AIEDNN_ARG_SRC,
                       AIEDNN_ARG_SRC_ITER,
                       AIEDNN_ARG_SRC_ITER_C,
                       AIEDNN_ARG_WEIGHTS,
                       AIEDNN_ARG_WEIGHTS_ITER,
                       AIEDNN_ARG_BIAS,
                       0,
                       0,
                       0};
    const int ga[9] = {AIEDNN_ARG_DIFF_SRC,     AIEDNN_ARG_DIFF_SRC_ITER,     AIEDNN_ARG_DIFF_SRC_ITER_C,
                       AIEDNN_ARG_DIFF_WEIGHTS, AIEDNN_ARG_DIFF_WEIGHTS_ITER, AIEDNN_ARG_DIFF_BIAS,
                       AIEDNN_ARG_DIFF_DST,     AIEDNN_ARG_DIFF_DST_ITER,     AIEDNN_ARG_DIFF_DST_ITER_C};
    for (int i = 0; i < 6; i++)
        if (pairs[i][0])
            prob_add(&p, fa[i], pairs[i][0]);
    for (int i = 0; i < 9; i++)
        if (pairs[i][1])
            prob_add(&p, ga[i], pairs[i][1]);
    p.flags = (unsigned)dir;
    p.alpha = alpha;
    return create(pr, &p);
}

#endif /* AIEDNN_RT_RNN_BWD_H */
