/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_misc.h -- dropout, the spatial transformer (affine grid, grid sample) and CTC loss:
 * ref: implementations and their create functions (aiednn.h).
 *
 * Needs, from the including TU: aiednn_rt_internal.h and the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_MISC_H
#define AIEDNN_RT_MISC_H

/* ── dropout ── */
static inline uint64_t rt_mix64(uint64_t z) { /* splitmix64's finalizer: a counter-based hash */
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static int run_ref_dropout(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const double keep = 1.0 - (double)p->alpha;
    uint8_t *mask = (uint8_t *)rt_ptr(p, ptr, AIEDNN_ARG_MASK);
    const int bwd = p->prop == RT_BWD_DATA;
    const aiednn_memory_desc_t *x = rt_md(p, bwd ? AIEDNN_ARG_DIFF_DST : AIEDNN_ARG_SRC);
    const aiednn_memory_desc_t *y = rt_md(p, bwd ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_DST);
    const void *px = rt_ptr(p, ptr, bwd ? AIEDNN_ARG_DIFF_DST : AIEDNN_ARG_SRC);
    void *py = rt_ptr(p, ptr, bwd ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_DST);
    const uint64_t seed = ((uint64_t)p->local_size << 32) ^ (uint64_t)(uint32_t)p->group_size;
    const int64_t n = md_numel(x);
    if (x->data_type == aiednn_f32 && y->data_type == aiednn_f32) { /* typed f32 loop: the same values */
        const float *xf = (const float *)px;
        float *yf = (float *)py;
        const uint64_t base = seed * 0x2545f4914f6cdd1dULL;
        for (int64_t i = 0; i < n; i++) {
            if (!bwd)
                mask[i] = (double)(rt_mix64(base + (uint64_t)i) >> 11) * (1.0 / 9007199254740992.0) < keep;
            yf[i] = mask[i] ? (float)((double)xf[i] / keep) : 0.0f;
        }
        return AIEK_OK;
    }
    for (int64_t i = 0; i < n; i++) {
        if (!bwd) /* keep with probability 1 - p: the top 53 bits of the hash as a uniform in [0, 1) */
            mask[i] =
                (double)(rt_mix64(seed * 0x2545f4914f6cdd1dULL + (uint64_t)i) >> 11) * (1.0 / 9007199254740992.0) <
                keep;
        rt_st(y->data_type, py, i, mask[i] ? rt_ld(x->data_type, px, i) / keep : 0.0);
    }
    return AIEK_OK;
}

static aiednn_status_t dropout_create(aiednn_primitive_t *pr, int bwd, const aiednn_memory_desc_t *x,
                                      const aiednn_memory_desc_t *y, float prob, uint64_t seed) {
    const char *k = "dropout";
    REQUIRE(pr && md_ok(x) && md_ok(y) && md_same_shape(x, y), k, "src and dst: same shape");
    REQUIRE(prob >= 0.0f && prob < 1.0f, k, "probability in [0, 1)");
    REQUIRE(dt_is_float(x->data_type) && dt_is_float(y->data_type), k, "floating types");
    rt_prob p;
    prob_init(&p, aiednn_dropout, 0);
    p.prop = bwd ? RT_BWD_DATA : RT_FWD_TRAIN;
    prob_add(&p, bwd ? AIEDNN_ARG_DIFF_DST : AIEDNN_ARG_SRC, x);
    prob_add(&p, bwd ? AIEDNN_ARG_DIFF_SRC : AIEDNN_ARG_DST, y);
    aiednn_memory_desc_t m = *x;
    m.data_type = aiednn_u8;
    prob_add(&p, AIEDNN_ARG_MASK, &m);
    p.alpha = prob;
    p.local_size = (int64_t)(seed >> 32), p.group_size = (int64_t)(uint32_t)seed; /* seed rides here */
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_dropout_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                         const aiednn_memory_desc_t *dst, float prob, uint64_t seed) {
    return dropout_create(p, 0, src, dst, prob, seed);
}
extern "C" aiednn_status_t aiednn_dropout_backward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *diff_src,
                                                          const aiednn_memory_desc_t *diff_dst, float prob) {
    return dropout_create(p, 1, diff_dst, diff_src, prob, 0);
}

/* ── spatial transformer ── */
static double grid_unnorm(double g, int64_t size, int align) {
    return align ? (g + 1) / 2 * (double)(size - 1) : ((g + 1) * (double)size - 1) / 2;
}

/* forward grid sample, all f32: run_ref_grid_sample's arithmetic (double, same order) without the
 * per-element type dispatch; the four taps of an output position are found once for every channel */
static int grid_sample_f32(const rt_prob *p, void *const *ptr, int nearest, int align) {
    const aiednn_memory_desc_t *s = rt_md(p, AIEDNN_ARG_SRC), *d = rt_md(p, AIEDNN_ARG_DST);
    const float *src = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_SRC),
                *grid = (const float *)rt_ptr(p, ptr, AIEDNN_ARG_GRID);
    float *dst = (float *)rt_ptr(p, ptr, AIEDNN_ARG_DST);
    const int64_t N = s->dims[0], C = s->dims[1], H = s->dims[2], W = s->dims[3], Ho = d->dims[2], Wo = d->dims[3];
    for (int64_t n = 0; n < N; n++)
        for (int64_t h = 0; h < Ho; h++)
            for (int64_t w = 0; w < Wo; w++) {
                const int64_t gi = ((n * Ho + h) * Wo + w) * 2;
                const double x = grid_unnorm((double)grid[gi], W, align),
                             y = grid_unnorm((double)grid[gi + 1], H, align);
                const double x0 = floor(x), y0 = floor(y), wx = x - x0, wy = y - y0;
                const int64_t ix = (int64_t)x0, iy = (int64_t)y0;
                const int64_t ny = (int64_t)nearbyint(y), nx = (int64_t)nearbyint(x);
                const int64_t ty[4] = {iy, iy, iy + 1, iy + 1}, tx[4] = {ix, ix + 1, ix, ix + 1};
                int64_t off[4];
                for (int q = 0; q < 4; q++) /* -1: outside, reads 0 */
                    off[q] = ty[q] < 0 || ty[q] >= H || tx[q] < 0 || tx[q] >= W ? -1 : ty[q] * W + tx[q];
                const int64_t noff = ny < 0 || ny >= H || nx < 0 || nx >= W ? -1 : ny * W + nx;
                for (int64_t c = 0; c < C; c++) {
                    const float *pl = src + (n * C + c) * H * W;
                    double v;
                    if (nearest) {
                        v = noff < 0 ? 0.0 : (double)pl[noff];
                    } else {
                        const double a = off[0] < 0 ? 0.0 : pl[off[0]], b = off[1] < 0 ? 0.0 : pl[off[1]];
                        const double cc = off[2] < 0 ? 0.0 : pl[off[2]], dd = off[3] < 0 ? 0.0 : pl[off[3]];
                        v = a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + cc * (1 - wx) * wy + dd * wx * wy;
                    }
                    dst[((n * C + c) * Ho + h) * Wo + w] = (float)v;
                }
            }
    return AIEK_OK;
}

static int run_ref_grid_sample(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const int nearest = (p->flags & AIEDNN_GRID_SAMPLE_NEAREST) != 0,
              align = (p->flags & AIEDNN_GRID_ALIGN_CORNERS) != 0;
    if (p->alg == 1) { /* affine grid: theta [N, 2, 3] -> grid [N, Ho, Wo, 2] */
        const aiednn_memory_desc_t *t = rt_md(p, AIEDNN_ARG_SRC), *g = rt_md(p, AIEDNN_ARG_DST);
        const int64_t N = g->dims[0], Ho = g->dims[1], Wo = g->dims[2];
        for (int64_t n = 0; n < N; n++)
            for (int64_t h = 0; h < Ho; h++)
                for (int64_t w = 0; w < Wo; w++) {
                    const double x = align ? (Wo > 1 ? -1 + 2.0 * (double)w / (double)(Wo - 1) : 0.0)
                                           : (2.0 * (double)w + 1) / (double)Wo - 1;
                    const double y = align ? (Ho > 1 ? -1 + 2.0 * (double)h / (double)(Ho - 1) : 0.0)
                                           : (2.0 * (double)h + 1) / (double)Ho - 1;
                    for (int r = 0; r < 2; r++) {
                        const double v = rt_ld(t->data_type, rt_ptr(p, ptr, AIEDNN_ARG_SRC), n * 6 + r * 3) * x +
                                         rt_ld(t->data_type, rt_ptr(p, ptr, AIEDNN_ARG_SRC), n * 6 + r * 3 + 1) * y +
                                         rt_ld(t->data_type, rt_ptr(p, ptr, AIEDNN_ARG_SRC), n * 6 + r * 3 + 2);
                        rt_st(g->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DST), ((n * Ho + h) * Wo + w) * 2 + r, v);
                    }
                }
        return AIEK_OK;
    }
    const int bwd = p->prop != RT_FWD_INF;
    const aiednn_memory_desc_t *s = rt_md(p, AIEDNN_ARG_SRC), *gm = rt_md(p, AIEDNN_ARG_GRID);
    const aiednn_memory_desc_t *d = rt_md(p, bwd ? AIEDNN_ARG_DIFF_DST : AIEDNN_ARG_DST);
    if (!bwd && s->data_type == aiednn_f32 && gm->data_type == aiednn_f32 && d->data_type == aiednn_f32)
        return grid_sample_f32(p, ptr, nearest, align); /* typed loop: the same values */
    const void *ps = rt_ptr(p, ptr, AIEDNN_ARG_SRC), *pg = rt_ptr(p, ptr, AIEDNN_ARG_GRID);
    const int64_t N = s->dims[0], C = s->dims[1], H = s->dims[2], W = s->dims[3], Ho = d->dims[2], Wo = d->dims[3];
    double *dsrc = nullptr;
    if (bwd && !(dsrc = (double *)rt_calloc((size_t)md_numel(s), sizeof(double))))
        return AIEK_ERR_ALLOC;
    auto at = [&](int64_t n, int64_t c, int64_t y, int64_t x) {
        return y < 0 || y >= H || x < 0 || x >= W ? 0.0 : rt_ld(s->data_type, ps, ((n * C + c) * H + y) * W + x);
    };
    for (int64_t n = 0; n < N; n++)
        for (int64_t h = 0; h < Ho; h++)
            for (int64_t w = 0; w < Wo; w++) {
                const int64_t gi = ((n * Ho + h) * Wo + w) * 2;
                const double x = grid_unnorm(rt_ld(gm->data_type, pg, gi), W, align);
                const double y = grid_unnorm(rt_ld(gm->data_type, pg, gi + 1), H, align);
                const double x0 = floor(x), y0 = floor(y), wx = x - x0, wy = y - y0;
                const int64_t ix = (int64_t)x0, iy = (int64_t)y0;
                double gx = 0, gy = 0;
                for (int64_t c = 0; c < C; c++) {
                    const int64_t di = ((n * C + c) * Ho + h) * Wo + w;
                    if (!bwd) {
                        double v;
                        if (nearest) {
                            v = at(n, c, (int64_t)nearbyint(y), (int64_t)nearbyint(x));
                        } else {
                            v = at(n, c, iy, ix) * (1 - wx) * (1 - wy) + at(n, c, iy, ix + 1) * wx * (1 - wy) +
                                at(n, c, iy + 1, ix) * (1 - wx) * wy + at(n, c, iy + 1, ix + 1) * wx * wy;
                        }
                        rt_st(d->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DST), di, v);
                        continue;
                    }
                    const double g = rt_ld(d->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_DST), di);
                    const int64_t yy[4] = {iy, iy, iy + 1, iy + 1}, xx[4] = {ix, ix + 1, ix, ix + 1};
                    const double ww[4] = {(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy};
                    for (int q = 0; q < 4; q++)
                        if (yy[q] >= 0 && yy[q] < H && xx[q] >= 0 && xx[q] < W)
                            dsrc[((n * C + c) * H + yy[q]) * W + xx[q]] += g * ww[q];
                    gx += g * ((at(n, c, iy, ix + 1) - at(n, c, iy, ix)) * (1 - wy) +
                               (at(n, c, iy + 1, ix + 1) - at(n, c, iy + 1, ix)) * wy);
                    gy += g * ((at(n, c, iy + 1, ix) - at(n, c, iy, ix)) * (1 - wx) +
                               (at(n, c, iy + 1, ix + 1) - at(n, c, iy, ix + 1)) * wx);
                }
                if (bwd) { /* d unnormalized / d normalized */
                    const double sx = align ? (double)(W - 1) / 2 : (double)W / 2,
                                 sy = align ? (double)(H - 1) / 2 : (double)H / 2;
                    const aiednn_memory_desc_t *dg = rt_md(p, AIEDNN_ARG_DIFF_WEIGHTS);
                    rt_st(dg->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_WEIGHTS), gi, gx * sx);
                    rt_st(dg->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_WEIGHTS), gi + 1, gy * sy);
                }
            }
    if (bwd) {
        const aiednn_memory_desc_t *ds = rt_md(p, AIEDNN_ARG_DIFF_SRC);
        for (int64_t i = 0; i < md_numel(s); i++)
            rt_st(ds->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), i, dsrc[i]);
        rt_free(dsrc);
    }
    return AIEK_OK;
}

extern "C" aiednn_status_t aiednn_affine_grid_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *t,
                                                     const aiednn_memory_desc_t *g, unsigned flags) {
    const char *k = "grid_sample";
    REQUIRE(pr && md_ok(t) && md_ok(g) && t->ndims == 3 && t->dims[1] == 2 && t->dims[2] == 3 && g->ndims == 4 &&
                g->dims[0] == t->dims[0] && g->dims[3] == 2,
            k, "theta [N, 2, 3], grid [N, Ho, Wo, 2]");
    REQUIRE(dt_is_float(t->data_type) && dt_is_float(g->data_type), k, "floating types");
    rt_prob p;
    prob_init(&p, aiednn_grid_sample, 1);
    prob_add(&p, AIEDNN_ARG_SRC, t);
    prob_add(&p, AIEDNN_ARG_DST, g);
    p.flags = flags;
    return create(pr, &p);
}

static aiednn_status_t grid_sample_create(aiednn_primitive_t *pr, int bwd, const aiednn_memory_desc_t *s,
                                          const aiednn_memory_desc_t *g, const aiednn_memory_desc_t *d,
                                          const aiednn_memory_desc_t *ds, const aiednn_memory_desc_t *dg,
                                          unsigned flags) {
    const char *k = "grid_sample";
    REQUIRE(pr && md_ok(s) && md_ok(g) && md_ok(d) && s->ndims == 4 && g->ndims == 4 && d->ndims == 4 &&
                g->dims[0] == s->dims[0] && g->dims[3] == 2 && d->dims[0] == s->dims[0] && d->dims[1] == s->dims[1] &&
                d->dims[2] == g->dims[1] && d->dims[3] == g->dims[2],
            k, "src [N, C, H, W], grid [N, Ho, Wo, 2], dst [N, C, Ho, Wo]");
    REQUIRE(!(flags & ~(AIEDNN_GRID_SAMPLE_NEAREST | AIEDNN_GRID_ALIGN_CORNERS)), k, "unknown flag");
    REQUIRE(!bwd || (md_ok(ds) && md_same_shape(ds, s) && md_ok(dg) && md_same_shape(dg, g) &&
                     !(flags & AIEDNN_GRID_SAMPLE_NEAREST)),
            k, "backward: diff_src like src, diff_grid like grid, bilinear");
    const aiednn_memory_desc_t *all[5] = {s, g, d, ds, dg};
    for (int i = 0; i < 5; i++)
        REQUIRE(!all[i] || dt_is_float(all[i]->data_type), k, "floating types");
    rt_prob p;
    prob_init(&p, aiednn_grid_sample, 2);
    p.prop = bwd ? RT_BWD : RT_FWD_INF;
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_GRID, g);
    prob_add(&p, bwd ? AIEDNN_ARG_DIFF_DST : AIEDNN_ARG_DST, d);
    if (bwd) {
        prob_add(&p, AIEDNN_ARG_DIFF_SRC, ds);
        prob_add(&p, AIEDNN_ARG_DIFF_WEIGHTS, dg);
    }
    p.flags = flags;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_grid_sample_forward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *s,
                                                             const aiednn_memory_desc_t *g,
                                                             const aiednn_memory_desc_t *d, unsigned flags) {
    return grid_sample_create(p, 0, s, g, d, nullptr, nullptr, flags);
}
extern "C" aiednn_status_t aiednn_grid_sample_backward_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *ds,
                                                              const aiednn_memory_desc_t *dg,
                                                              const aiednn_memory_desc_t *dd,
                                                              const aiednn_memory_desc_t *s,
                                                              const aiednn_memory_desc_t *g, unsigned flags) {
    return grid_sample_create(p, 1, s, g, dd, ds, dg, flags);
}

/* ── CTC loss: alpha / beta in log space over the label sequence with blanks between ── */
static double lse(double a, double b) {
    if (a == -INFINITY)
        return b;
    if (b == -INFINITY)
        return a;
    return a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b));
}

static int run_ref_ctc(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *lp = rt_md(p, AIEDNN_ARG_SRC), *loss = rt_md(p, AIEDNN_ARG_DST);
    const aiednn_memory_desc_t *dx = rt_md(p, AIEDNN_ARG_DIFF_SRC);
    const void *plp = rt_ptr(p, ptr, AIEDNN_ARG_SRC);
    const int32_t *lab = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_LABELS);
    const int32_t *ll = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_LABEL_LENGTHS);
    const int32_t *il = (const int32_t *)rt_ptr(p, ptr, AIEDNN_ARG_INPUT_LENGTHS);
    const int64_t T = lp->dims[0], N = lp->dims[1], Cn = lp->dims[2], Lmax = p->group_size;
    const int64_t S = 2 * Lmax + 1;
    double *al = (double *)rt_alloc(sizeof(double) * (size_t)(2 * T * S + Cn));
    if (!al)
        return AIEK_ERR_ALLOC;
    double *be = al + T * S, *lc = be + T * S;
    auto L = [&](int64_t t, int64_t n, int64_t c) { return rt_ld(lp->data_type, plp, (t * N + n) * Cn + c); };
    for (int64_t n = 0; n < N; n++) {
        const int64_t Tn = il[n], Ln = ll[n], Sn = 2 * Ln + 1;
        if (Tn < 0 || Tn > T || Ln < 0 || Ln > Lmax) {
            rt_free(al);
            return AIEK_ERR_PARAM;
        }
        auto lbl = [&](int64_t s) { return s % 2 ? (int64_t)lab[n * Lmax + s / 2] : (int64_t)0; };
        for (int64_t i = 0; i < T * S; i++)
            al[i] = be[i] = -INFINITY;
        for (int64_t t = 0; t < Tn; t++)
            for (int64_t s = 0; s < Sn; s++) {
                double a;
                if (t == 0) {
                    a = s < 2 ? 0.0 : -INFINITY;
                } else {
                    a = al[(t - 1) * S + s];
                    if (s >= 1)
                        a = lse(a, al[(t - 1) * S + s - 1]);
                    if (s >= 2 && lbl(s) != 0 && lbl(s) != lbl(s - 2))
                        a = lse(a, al[(t - 1) * S + s - 2]);
                }
                al[t * S + s] = a == -INFINITY ? a : a + L(t, n, lbl(s));
            }
        for (int64_t t = Tn - 1; t >= 0; t--)
            for (int64_t s = Sn - 1; s >= 0; s--) {
                double b;
                if (t == Tn - 1) {
                    b = s >= Sn - 2 ? 0.0 : -INFINITY;
                } else {
                    b = be[(t + 1) * S + s];
                    if (s + 1 < Sn)
                        b = lse(b, be[(t + 1) * S + s + 1]);
                    if (s + 2 < Sn && lbl(s) != 0 && lbl(s) != lbl(s + 2))
                        b = lse(b, be[(t + 1) * S + s + 2]);
                }
                be[t * S + s] = b == -INFINITY ? b : b + L(t, n, lbl(s));
            }
        const double ll_ =
            Tn > 0 ? lse(al[(Tn - 1) * S + Sn - 1], Sn > 1 ? al[(Tn - 1) * S + Sn - 2] : -INFINITY) : -INFINITY;
        const double nll = -ll_;
        rt_st(loss->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DST), n, nll);
        if (!dx)
            continue;
        for (int64_t t = 0; t < T; t++) { /* d nll / d logits under log-softmax: e^lp - posterior */
            for (int64_t c = 0; c < Cn; c++)
                lc[c] = -INFINITY;
            if (t < Tn)
                for (int64_t s = 0; s < Sn; s++)
                    lc[lbl(s)] = lse(lc[lbl(s)], al[t * S + s] + be[t * S + s]);
            for (int64_t c = 0; c < Cn; c++) {
                const double l = L(t, n, c);
                const double g = t < Tn ? exp(l) - exp(lc[c] + nll - l) : 0.0;
                rt_st(dx->data_type, rt_ptr(p, ptr, AIEDNN_ARG_DIFF_SRC), (t * N + n) * Cn + c, g);
            }
        }
    }
    rt_free(al);
    return AIEK_OK;
}

extern "C" aiednn_status_t aiednn_ctc_loss_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *lp,
                                                  int64_t max_label_length, const aiednn_memory_desc_t *loss,
                                                  const aiednn_memory_desc_t *dx) {
    const char *k = "ctc_loss";
    REQUIRE(pr && md_ok(lp) && md_ok(loss) && lp->ndims == 3 && loss->ndims == 1 && loss->dims[0] == lp->dims[1], k,
            "log_probs [T, N, C], loss [N]");
    REQUIRE(max_label_length >= 0 && (!dx || (md_ok(dx) && md_same_shape(dx, lp))), k,
            "max_label_length >= 0, diff_src like log_probs");
    REQUIRE(dt_is_float(lp->data_type) && dt_is_float(loss->data_type) && (!dx || dt_is_float(dx->data_type)), k,
            "floating types");
    rt_prob p;
    prob_init(&p, aiednn_ctc_loss, 0);
    p.prop = dx ? RT_BWD : RT_FWD_TRAIN;
    prob_add(&p, AIEDNN_ARG_SRC, lp);
    prob_add(&p, AIEDNN_ARG_DST, loss);
    aiednn_memory_desc_t lb, nv;
    memset(&lb, 0, sizeof lb);
    lb.ndims = 2, lb.dims[0] = lp->dims[1], lb.dims[1] = max_label_length > 0 ? max_label_length : 1;
    lb.data_type = aiednn_s32;
    memset(&nv, 0, sizeof nv);
    nv.ndims = 1, nv.dims[0] = lp->dims[1], nv.data_type = aiednn_s32;
    prob_add(&p, AIEDNN_ARG_LABELS, &lb);
    prob_add(&p, AIEDNN_ARG_LABEL_LENGTHS, &nv);
    prob_add(&p, AIEDNN_ARG_INPUT_LENGTHS, &nv);
    if (dx)
        prob_add(&p, AIEDNN_ARG_DIFF_SRC, dx);
    p.group_size = max_label_length;
    return create(pr, &p);
}

#endif /* AIEDNN_RT_MISC_H */
