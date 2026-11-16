/* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Activation -> int8 by table (GELU, swish). The epilogue of a GEMM with one fused activation
 * and an s8 / u8 dst computes, per output element,
 *   q(y) = sat(nearbyint(f(y) / ds + zp)),   y = acc * oscale[c] + bias[c]
 * with f = y * Phi(y) (gelu_erf), its tanh form (gelu_tanh) or y * sigmoid(a * y) (swish). q is a step
 * function of y with at most 256 steps, so one table per (f, a, ds, zp, dst type) replaces it: a grid over y
 * whose cells each hold the code before and after their one change point t (found by bisection on
 * rt_actq_direct). y within `band` of t, cells holding the minimum xm, and cells with two steps take the
 * formula, so the table returns exactly what the formula returns (rt_actq_check). Left of ya q = sat(zp);
 * from yb on q is the dst maximum -- both verified when the table is built. */
#ifndef AIEDNN_RT_GELUQ_H
#define AIEDNN_RT_GELUQ_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "aiednn_rt_phi.h"

enum { RT_ACTQ_GELU = 1, RT_ACTQ_SWISH = 2, RT_ACTQ_GELU_TANH = 3 };

/* swish exactly as eltwise_f evaluates it (aiednn_rt_ops.h calls this): x * sigmoid(a * x) */
static inline double rt_swish(double x, double a) { return x * (1.0 / (1.0 + exp(-(a * x)))); }

/* the tanh GELU exactly as eltwise_f evaluates it (GPT-2, BERT's "gelu_new") */
static inline double rt_gelu_tanh(double x) {
    return 0.5 * x * (1.0 + tanh(0.79788456080286535588 /* sqrt(2/pi) */ * (x + 0.044715 * x * x * x)));
}

static inline double rt_actq_f(int kind, double a, double v) {
    return kind == RT_ACTQ_GELU ? v * rt_phi(v) : kind == RT_ACTQ_GELU_TANH ? rt_gelu_tanh(v) : rt_swish(v, a);
}

/* the reference: rt_epi_row's double operations, in its order */
static inline int rt_actq_direct(int kind, double a, double v, double ds, double zp, double lo, double hi) {
    v = rt_actq_f(kind, a, v);
    v = v / ds + zp;
    v = v != v ? 0.0 : nearbyint(v);
    return (int)(v < lo ? lo : v > hi ? hi : v);
}

#define RT_ACTQ_MAXCELLS 16384
#define RT_ACTQ_NTAB 256

typedef struct {
    double t;       /* change point: y >= t takes c1 */
    double w;       /* |y - t| <= w takes the formula: the band, or +inf for a slow cell */
    int16_t c0, c1; /* code below / from t */
} rt_actq_cell;

typedef struct {
    int kind;
    double a, ds, zp, lo, hi; /* key: activation, alpha, dst scale / zero point, dst range */
    double ya, xm, yb;        /* left end, minimum, right end */
    double inv_h;             /* 1 / cell width */
    int ncell;
    double band;              /* |y - t| <= band: the formula (double jitter at a step) */
    int16_t lo_code, hi_code; /* y < ya, y >= yb */
    rt_actq_cell *cell;       /* [0] and [ncell + 1]: constant sentinels (lo_code / hi_code) */
} rt_actq;

/* smallest y in (a, b] with direct(y) == cb, given direct(a) != cb == direct(b) */
static double rt_actq_bisect(const rt_actq *g, double a, double b, int cb, int *ok) {
    for (int it = 0; it < 256; it++) {
        const double m = a + (b - a) * 0.5;
        if (!(m > a && m < b))
            return b;
        if (rt_actq_direct(g->kind, g->a, m, g->ds, g->zp, g->lo, g->hi) == cb)
            b = m;
        else
            a = m;
    }
    *ok = 0;
    return b;
}

static void rt_actq_fill_cell(const rt_actq *g, rt_actq_cell *c, double lo, double hi, double win) {
    /* the cell answers for [lo, hi), widened by pad: the float index may round across a boundary */
    const double pad = (hi - lo) * 1e-6;
    lo -= pad;
    hi += pad;
    const int c0 = rt_actq_direct(g->kind, g->a, lo, g->ds, g->zp, g->lo, g->hi);
    const int c1 = rt_actq_direct(g->kind, g->a, hi, g->ds, g->zp, g->lo, g->hi);
    c->c0 = (int16_t)c0;
    c->c1 = (int16_t)c1;
    c->t = hi;
    c->w = g->band;
    int slow = 0;
    if (lo - win <= g->xm && g->xm <= hi + win) {
        /* contains the minimum: constant iff both ends and the minimum agree, clear of a rounding boundary */
        const int cm = rt_actq_direct(g->kind, g->a, g->xm, g->ds, g->zp, g->lo, g->hi);
        const double vm = rt_actq_f(g->kind, g->a, g->xm) / g->ds + g->zp;
        slow = !(c0 == c1 && c0 == cm && fabs(vm - floor(vm) - 0.5) > 1e-6);
    } else if (c0 != c1) {
        int ok = c1 - c0 == 1 || c0 - c1 == 1; /* else two steps in one cell */
        if (ok)
            c->t = rt_actq_bisect(g, lo, hi, c1, &ok);
        slow = !ok;
    }
    if (slow) {
        c->t = 0.5 * (lo + hi);
        c->w = INFINITY;
    }
}

/* the ends: ya where |f| / ds < 1/4 (and |f| only shrinks further left), yb where q saturates */
static int rt_actq_ends(rt_actq *g, double w) {
    const int zc = rt_actq_direct(g->kind, g->a, -1e300, g->ds, g->zp, g->lo, g->hi);
    double ya = g->xm - w;
    while (fabs(rt_actq_f(g->kind, g->a, ya)) / g->ds >= 0.25) {
        ya -= w;
        if (ya < g->xm - 1e4 * w)
            return 0;
    }
    if (rt_actq_direct(g->kind, g->a, ya, g->ds, g->zp, g->lo, g->hi) != zc)
        return 0;
    double yb = fmax(1.0, w);
    while (rt_actq_direct(g->kind, g->a, yb, g->ds, g->zp, g->lo, g->hi) != (int)g->hi) {
        yb *= 2;
        if (yb > 1e6)
            return 0;
    }
    g->ya = ya;
    g->yb = yb;
    g->lo_code = (int16_t)zc;
    g->hi_code = (int16_t)g->hi;
    return 1;
}

static int rt_actq_build(rt_actq *g) {
    /* minimum of f and its width scale: GELU at -0.7518 (tanh form -0.7525); swish(x) = x sigmoid(a x)
     * at -1.2785 / a */
    const double w = g->kind == RT_ACTQ_SWISH ? 1.0 / g->a : 1.0;
    g->xm = g->kind == RT_ACTQ_GELU ? -0.7517916 : g->kind == RT_ACTQ_GELU_TANH ? -0.7524614 : -1.2784645 * w;
    if (!rt_actq_ends(g, w))
        return 0;
    /* steps are >= ds / max |f'| apart (max |f'| = 1.13 for either GELU, 1.1 for swish): cells ds / 2 */
    double h = g->ds * 0.5, n = ceil((g->yb - g->ya) / h);
    if (n > RT_ACTQ_MAXCELLS) { /* coarser: more cells hold two steps and go slow, still exact */
        n = RT_ACTQ_MAXCELLS;
        h = (g->yb - g->ya) / n;
    }
    g->ncell = (int)n;
    g->yb = g->ya + g->ncell * h; /* only moves right: q stays saturated */
    g->inv_h = 1.0 / h;
    g->band = fmax(-g->ya, g->yb) * 1e-12;
    g->cell = (rt_actq_cell *)malloc((size_t)(g->ncell + 2) * sizeof(rt_actq_cell));
    if (!g->cell)
        return 0;
    const rt_actq_cell left = {INFINITY, 0.0, g->lo_code, g->lo_code}, right = {INFINITY, 0.0, g->hi_code, g->hi_code};
    g->cell[0] = left;
    g->cell[g->ncell + 1] = right;
    const double win = 1e-5 * w; /* xm's precision, with margin */
    for (int i = 0; i < g->ncell; i++)
        rt_actq_fill_cell(g, &g->cell[1 + i], g->ya + i * h, g->ya + (i + 1) * h, win);
    return 1;
}

/* the table for (kind, a, ds, zp, lo, hi), built on first use and kept (one per activation layer) */
static const rt_actq *rt_actq_get(int kind, double a, double ds, double zp, double lo, double hi) {
    static rt_actq tab[RT_ACTQ_NTAB];
    static int ntab;
    for (int i = 0; i < ntab; i++)
        if (tab[i].kind == kind && tab[i].a == a && tab[i].ds == ds && tab[i].zp == zp && tab[i].lo == lo &&
            tab[i].hi == hi)
            return tab[i].cell ? &tab[i] : NULL;
    if (ntab == RT_ACTQ_NTAB || !(ds > 0) || !(fabs(zp) < 1e6) || (kind == RT_ACTQ_SWISH && !(a > 0)))
        return NULL;
    rt_actq *g = &tab[ntab++]; /* a failed build stays in the cache with cell NULL: not retried */
    memset(g, 0, sizeof *g);
    g->kind = kind;
    g->a = a;
    g->ds = ds;
    g->zp = zp;
    g->lo = lo;
    g->hi = hi;
    if (!rt_actq_build(g)) {
        free(g->cell);
        g->cell = NULL;
        return NULL;
    }
    return g;
}

/* one conversion, one clamp, one guard compare: ends are sentinel cells reached by clamping the index,
 * a slow cell has w = +inf. Clamps are compares, not fmin / fmax. */
static inline int rt_actq_code(const rt_actq *g, double y) {
    double d = (y - g->ya) * g->inv_h;
    d = d < -1.0 ? -1.0 : d > (double)g->ncell ? (double)g->ncell : d;
    if (d != d) /* NaN: the formula, as it would have */
        return rt_actq_direct(g->kind, g->a, y, g->ds, g->zp, g->lo, g->hi);
    const rt_actq_cell *c = &g->cell[1 + (int)d];
    if (fabs(y - c->t) <= c->w)
        return rt_actq_direct(g->kind, g->a, y, g->ds, g->zp, g->lo, g->hi);
    return y >= c->t ? c->c1 : c->c0;
}

/* table vs formula on n points uniform in [ya - 1, yb + 1], every double within 64 ulps of each
 * change point, and steps of band / 8 around it; returns the number of disagreements (0), -1 if
 * no table could be built */
static int64_t rt_actq_check(int kind, double a, double ds, double zp, double lo, double hi, int64_t n, uint64_t seed) {
    const rt_actq *g = rt_actq_get(kind, a, ds, zp, lo, hi);
    if (!g)
        return -1;
    int64_t bad = 0;
#define RT_ACTQ_CMP(y) (bad += rt_actq_code(g, (y)) != rt_actq_direct(kind, a, (y), ds, zp, lo, hi))
    for (int64_t k = 0; k < n; k++) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const double u = (double)(seed >> 11) * (1.0 / 9007199254740992.0);
        RT_ACTQ_CMP(g->ya - 1.0 + u * (g->yb - g->ya + 2.0));
    }
    for (int i = 1; i <= g->ncell; i++) {
        const double t = g->cell[i].t;
        double l = t, h = t;
        for (int s = 0; s < 64; s++) {
            RT_ACTQ_CMP(l);
            RT_ACTQ_CMP(h);
            l = nextafter(l, -INFINITY);
            h = nextafter(h, INFINITY);
        }
        for (int s = -40; s <= 40; s++)
            RT_ACTQ_CMP(t + s * (g->band / 8));
    }
#undef RT_ACTQ_CMP
    return bad;
}

#endif /* AIEDNN_RT_GELUQ_H */
