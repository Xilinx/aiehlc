/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_rnn.h -- recurrent networks, forward: vanilla RNN, LSTM, GRU, linear-before-reset GRU;
 * multi-layer, left-to-right / right-to-left / bidirectional (concat or sum). ref: only, in double.
 * Weights [L, D, in, G, DHC]; gates LSTM i f c~ o, GRU u r o.
 *
 * Needs, from the including TU: aiednn_rt_internal.h and the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_RNN_H
#define AIEDNN_RT_RNN_H

static inline int rnn_is_augru(int cell) { return cell == aiednn_rnn_augru || cell == aiednn_rnn_lbr_augru; }
static inline int rnn_is_lbr(int cell) { return cell == aiednn_rnn_lbr_gru || cell == aiednn_rnn_lbr_augru; }
static inline int rnn_is_gru(int cell) { return cell == aiednn_rnn_gru || rnn_is_lbr(cell) || rnn_is_augru(cell); }

static int rnn_gates(int cell) { return cell == aiednn_rnn_lstm ? 4 : rnn_is_gru(cell) ? 3 : 1; }

/* AUGRU: the update gate scaled by 1 - a[t, n] */
static inline double rnn_att(const rt_prob *p, void *const *ptr, int64_t t, int64_t n, int64_t N) {
    const int sa = rt_slot(p, AIEDNN_ARG_AUGRU_ATTENTION);
    return sa >= 0 ? 1.0 - rt_ld(p->md[sa].data_type, ptr[sa], t * N + n) : 1.0;
}

static double rnn_sig(double x) { return 1.0 / (1.0 + exp(-x)); }

/* one (layer, direction) over the sequence: in [T][N][I] (double), writes h_t into out at column
 * offset col of [T][N][oc] (accumulating for a bidirectional sum), final h / c into hT / cT */
typedef struct {
    const rt_prob *p;
    void *const *ptr;
    int cell, G, Gb;
    int64_t T, N, I, H, L, D, oc;
} rnn_ctx;

static void rnn_layer_dir(const rnn_ctx *c, int64_t l, int64_t d, int rev, const double *in, double *out, int64_t col,
                          int acc, double *h, double *cs, double *gates) {
    const rt_prob *p = c->p;
    void *const *ptr = c->ptr;
    const int si = rt_slot(p, AIEDNN_ARG_SRC_ITER), sc = rt_slot(p, AIEDNN_ARG_SRC_ITER_C);
    const int sw = rt_slot(p, AIEDNN_ARG_WEIGHTS), su = rt_slot(p, AIEDNN_ARG_WEIGHTS_ITER);
    const int sb = rt_slot(p, AIEDNN_ARG_BIAS);
    const int64_t N = c->N, I = c->I, H = c->H, G = c->G, ld = l * c->D + d;
    for (int64_t n = 0; n < N; n++)
        for (int64_t j = 0; j < H; j++) {
            h[n * H + j] = si >= 0 ? rt_ld(p->md[si].data_type, ptr[si], (ld * N + n) * H + j) : 0.0;
            cs[n * H + j] = sc >= 0 ? rt_ld(p->md[sc].data_type, ptr[sc], (ld * N + n) * H + j) : 0.0;
        }
    const aiednn_data_type_t wt = p->md[sw].data_type, ut = p->md[su].data_type;
    auto W = [&](int64_t k, int64_t g, int64_t j) { return rt_ld(wt, ptr[sw], ((ld * I + k) * G + g) * H + j); };
    auto U = [&](int64_t k, int64_t g, int64_t j) { return rt_ld(ut, ptr[su], ((ld * H + k) * G + g) * H + j); };
    auto B = [&](int64_t g, int64_t j) {
        return sb >= 0 ? rt_ld(p->md[sb].data_type, ptr[sb], (ld * c->Gb + g) * H + j) : 0.0;
    };
    for (int64_t s = 0; s < c->T; s++) {
        const int64_t t = rev ? c->T - 1 - s : s;
        for (int64_t n = 0; n < N; n++) {
            const double *x = in + (t * N + n) * I, *hp = h + n * H;
            for (int64_t g = 0; g < G; g++) /* W x + b, and U h for every gate but the GRUs' o */
                for (int64_t j = 0; j < H; j++) {
                    double a = B(g, j);
                    for (int64_t k = 0; k < I; k++)
                        a += x[k] * W(k, g, j);
                    gates[g * H + j] = a;
                }
            const int gru = rnn_is_gru(c->cell);
            for (int64_t g = 0; g < G; g++) {
                if (gru && g == 2)
                    continue;
                for (int64_t j = 0; j < H; j++) {
                    double a = 0;
                    for (int64_t k = 0; k < H; k++)
                        a += hp[k] * U(k, g, j);
                    gates[g * H + j] += a;
                }
            }
            for (int64_t j = 0; j < H; j++) {
                double v;
                if (c->cell == aiednn_rnn_lstm) {
                    const double ig = rnn_sig(gates[j]), fg = rnn_sig(gates[H + j]), cg = tanh(gates[2 * H + j]);
                    const double og = rnn_sig(gates[3 * H + j]);
                    cs[n * H + j] = fg * cs[n * H + j] + ig * cg;
                    v = og * tanh(cs[n * H + j]);
                } else if (gru) {
                    const double u = rnn_sig(gates[j]) * rnn_att(p, ptr, t, n, N), r = rnn_sig(gates[H + j]);
                    double o = gates[2 * H + j];
                    if (rnn_is_lbr(c->cell)) {
                        double uh = B(3, j);
                        for (int64_t k = 0; k < H; k++)
                            uh += hp[k] * U(k, 2, j);
                        o += r * uh;
                    } else {
                        for (int64_t k = 0; k < H; k++) /* U_o (r * h): r_k scales h_k */
                            o += rnn_sig(gates[H + k]) * hp[k] * U(k, 2, j);
                    }
                    v = u * hp[j] + (1.0 - u) * tanh(o);
                } else {
                    const double a = gates[j];
                    v = c->cell == aiednn_rnn_vanilla_tanh       ? tanh(a)
                        : c->cell == aiednn_rnn_vanilla_logistic ? rnn_sig(a)
                        : a > 0                                  ? a
                                                                 : (double)p->alpha * a;
                }
                gates[G * H + j] = v; /* the new h, applied after every j used the old one */
            }
            for (int64_t j = 0; j < H; j++) {
                h[n * H + j] = gates[G * H + j];
                double *o = out + (t * N + n) * c->oc + col + j;
                *o = acc ? *o + h[n * H + j] : h[n * H + j];
            }
        }
    }
}

static int run_ref_rnn_backward(const rt_prob *p, void *const *ptr); /* aiednn_rt_rnn_bwd.h */

static int run_ref_rnn(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    if (p->prop == RT_BWD)
        return run_ref_rnn_backward(p, ptr);
    const aiednn_memory_desc_t *sl = rt_md(p, AIEDNN_ARG_SRC), *dl = rt_md(p, AIEDNN_ARG_DST);
    const aiednn_memory_desc_t *wl = rt_md(p, AIEDNN_ARG_WEIGHTS);
    rnn_ctx c;
    c.p = p, c.ptr = ptr, c.cell = p->alg, c.G = rnn_gates(p->alg), c.Gb = c.G + rnn_is_lbr(p->alg);
    c.T = sl->dims[0], c.N = sl->dims[1], c.L = wl->dims[0], c.D = wl->dims[1], c.H = wl->dims[4];
    const int dir = (int)p->flags;
    const int64_t SLC = sl->dims[2], DLC = dl->dims[2], T = c.T, N = c.N, H = c.H;
    const int64_t W = SLC > DLC ? SLC : DLC;
    double *a = (double *)rt_alloc(sizeof(double) * (size_t)(2 * T * N * W + 2 * N * H + (c.G + 1) * H));
    if (!a)
        return AIEK_ERR_ALLOC;
    double *in = a, *out = a + T * N * W, *h = out + T * N * W, *cs = h + N * H, *gates = cs + N * H;
    for (int64_t i = 0; i < T * N * SLC; i++)
        in[i] = rt_ld(sl->data_type, ptr[0], i);
    const int sdi = rt_slot(p, AIEDNN_ARG_DST_ITER), sdc = rt_slot(p, AIEDNN_ARG_DST_ITER_C);
    for (int64_t l = 0; l < c.L; l++) {
        c.I = l == 0 ? SLC : DLC;
        c.oc = DLC;
        for (int64_t d = 0; d < c.D; d++) {
            const int rev = dir == aiednn_rnn_right2left || d == 1;
            const int64_t col = dir == aiednn_rnn_bidirectional_concat ? d * H : 0;
            rnn_layer_dir(&c, l, d, rev, in, out, col, dir == aiednn_rnn_bidirectional_sum && d == 1, h, cs, gates);
            for (int64_t i = 0; i < N * H; i++) {
                if (sdi >= 0)
                    rt_st(p->md[sdi].data_type, ptr[sdi], (l * c.D + d) * N * H + i, h[i]);
                if (sdc >= 0)
                    rt_st(p->md[sdc].data_type, ptr[sdc], (l * c.D + d) * N * H + i, cs[i]);
            }
        }
        double *tmp = in; /* this layer's output is the next one's input */
        in = out, out = tmp;
    }
    for (int64_t i = 0; i < T * N * DLC; i++)
        rt_st(dl->data_type, ptr[rt_slot(p, AIEDNN_ARG_DST)], i, in[i]);
    rt_free(a);
    return AIEK_OK;
}

static aiednn_status_t rnn_create_(aiednn_primitive_t *pr, aiednn_alg_kind_t cell, aiednn_rnn_direction_t dir,
                                   float alpha, const aiednn_memory_desc_t *src_layer,
                                   const aiednn_memory_desc_t *src_iter, const aiednn_memory_desc_t *src_iter_c,
                                   const aiednn_memory_desc_t *weights_layer, const aiednn_memory_desc_t *weights_iter,
                                   const aiednn_memory_desc_t *bias, const aiednn_memory_desc_t *dst_layer,
                                   const aiednn_memory_desc_t *dst_iter, const aiednn_memory_desc_t *dst_iter_c,
                                   const aiednn_memory_desc_t *attention) {
    const char *k = "rnn";
    REQUIRE(pr, k, "null primitive");
    REQUIRE(dir >= aiednn_rnn_left2right && dir <= aiednn_rnn_bidirectional_sum, k, "not an rnn direction");
    REQUIRE(md_ok(src_layer) && md_ok(weights_layer) && md_ok(weights_iter) && md_ok(dst_layer), k,
            "src_layer, weights_layer, weights_iter, dst_layer are required");
    REQUIRE(src_layer->ndims == 3 && dst_layer->ndims == 3 && weights_layer->ndims == 5 && weights_iter->ndims == 5, k,
            "src_layer / dst_layer [T, N, C], weights [L, D, C, G, DHC]");
    const int64_t T = src_layer->dims[0], N = src_layer->dims[1], SLC = src_layer->dims[2];
    const int64_t L = weights_layer->dims[0], D = weights_layer->dims[1], G = rnn_gates(cell),
                  H = weights_layer->dims[4];
    const int bi = dir == aiednn_rnn_bidirectional_concat || dir == aiednn_rnn_bidirectional_sum;
    const int64_t DLC = dir == aiednn_rnn_bidirectional_concat ? 2 * H : H;
    REQUIRE(D == (bi ? 2 : 1), k, "weights' D: 1 unidirectional, 2 bidirectional");
    REQUIRE(weights_layer->dims[2] == SLC && weights_layer->dims[3] == G, k,
            "weights_layer must be [L, D, SLC, G, DHC]");
    REQUIRE(weights_iter->dims[0] == L && weights_iter->dims[1] == D && weights_iter->dims[2] == H &&
                weights_iter->dims[3] == G && weights_iter->dims[4] == H,
            k, "weights_iter must be [L, D, DHC, G, DHC]");
    REQUIRE(dst_layer->dims[0] == T && dst_layer->dims[1] == N && dst_layer->dims[2] == DLC, k,
            "dst_layer must be [T, N, DHC] (2 DHC for bidirectional concat)");
    REQUIRE(L == 1 || SLC == DLC, k, "more than one layer needs SLC == DLC");
    const int64_t Gb = G + rnn_is_lbr(cell);
    REQUIRE(!bias || (md_ok(bias) && bias->ndims == 4 && bias->dims[0] == L && bias->dims[1] == D &&
                      bias->dims[2] == Gb && bias->dims[3] == H),
            k, "bias must be [L, D, G, DHC] (lbr_gru: G + 1)");
    const aiednn_memory_desc_t *its[4] = {src_iter, src_iter_c, dst_iter, dst_iter_c};
    for (int i = 0; i < 4; i++)
        REQUIRE(!its[i] || (md_ok(its[i]) && its[i]->ndims == 4 && its[i]->dims[0] == L && its[i]->dims[1] == D &&
                            its[i]->dims[2] == N && its[i]->dims[3] == H),
                k, "src_iter / dst_iter (_c) must be [L, D, N, DHC]");
    REQUIRE(cell == aiednn_rnn_lstm || (!src_iter_c && !dst_iter_c), k, "iter_c is LSTM only");
    REQUIRE(!attention ||
                (md_ok(attention) && (attention->ndims == 2 || (attention->ndims == 3 && attention->dims[2] == 1)) &&
                 attention->dims[0] == T && attention->dims[1] == N && dt_is_float(attention->data_type)),
            k, "augru attention must be [T, N] (or [T, N, 1]), floating");
    const aiednn_memory_desc_t *all[9] = {src_layer, src_iter,  src_iter_c, weights_layer, weights_iter,
                                          bias,      dst_layer, dst_iter,   dst_iter_c};
    for (int i = 0; i < 9; i++)
        REQUIRE(!all[i] || dt_is_float(all[i]->data_type), k, "floating types (f32 / bf16 / f16)");
    rt_prob p;
    prob_init(&p, aiednn_rnn, cell);
    prob_add(&p, AIEDNN_ARG_SRC, src_layer);
    prob_add(&p, AIEDNN_ARG_WEIGHTS, weights_layer);
    prob_add(&p, AIEDNN_ARG_DST, dst_layer);
    prob_add(&p, AIEDNN_ARG_WEIGHTS_ITER, weights_iter);
    if (bias)
        prob_add(&p, AIEDNN_ARG_BIAS, bias);
    if (src_iter)
        prob_add(&p, AIEDNN_ARG_SRC_ITER, src_iter);
    if (src_iter_c)
        prob_add(&p, AIEDNN_ARG_SRC_ITER_C, src_iter_c);
    if (dst_iter)
        prob_add(&p, AIEDNN_ARG_DST_ITER, dst_iter);
    if (dst_iter_c)
        prob_add(&p, AIEDNN_ARG_DST_ITER_C, dst_iter_c);
    if (attention)
        prob_add(&p, AIEDNN_ARG_AUGRU_ATTENTION, attention);
    p.flags = (unsigned)dir;
    p.alpha = alpha;
    return create(pr, &p);
}

extern "C" aiednn_status_t
aiednn_rnn_forward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t cell, aiednn_rnn_direction_t dir, float alpha,
                          const aiednn_memory_desc_t *src_layer, const aiednn_memory_desc_t *src_iter,
                          const aiednn_memory_desc_t *src_iter_c, const aiednn_memory_desc_t *weights_layer,
                          const aiednn_memory_desc_t *weights_iter, const aiednn_memory_desc_t *bias,
                          const aiednn_memory_desc_t *dst_layer, const aiednn_memory_desc_t *dst_iter,
                          const aiednn_memory_desc_t *dst_iter_c) {
    REQUIRE(cell >= aiednn_rnn_vanilla_relu && cell <= aiednn_rnn_lbr_gru, "rnn",
            "not an rnn cell (augru: aiednn_augru_forward_create)");
    return rnn_create_(pr, cell, dir, alpha, src_layer, src_iter, src_iter_c, weights_layer, weights_iter, bias,
                       dst_layer, dst_iter, dst_iter_c, nullptr);
}

extern "C" aiednn_status_t
aiednn_augru_forward_create(aiednn_primitive_t *pr, aiednn_alg_kind_t cell, aiednn_rnn_direction_t dir,
                            const aiednn_memory_desc_t *src_layer, const aiednn_memory_desc_t *src_iter,
                            const aiednn_memory_desc_t *attention, const aiednn_memory_desc_t *weights_layer,
                            const aiednn_memory_desc_t *weights_iter, const aiednn_memory_desc_t *bias,
                            const aiednn_memory_desc_t *dst_layer, const aiednn_memory_desc_t *dst_iter) {
    REQUIRE(rnn_is_augru(cell) && attention, "rnn", "augru / lbr_augru cell with an attention descriptor");
    return rnn_create_(pr, cell, dir, 0.0f, src_layer, src_iter, nullptr, weights_layer, weights_iter, bias, dst_layer,
                       dst_iter, nullptr, attention);
}

#endif /* AIEDNN_RT_RNN_H */
