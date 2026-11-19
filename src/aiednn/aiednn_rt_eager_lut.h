/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_eager_lut.h -- eager:lut: any per-channel function of one 8-bit value with an 8-bit result on the
 * lut template's instances (aie:eager_lut__M_N_TM, src/aietensorop/eager/lut_eager_kernel.cc). The tables are
 * fixed-megakernel lut:'s (lut_tables: ref: run on all 256 codes of every channel), so the result is ref:'s bit
 * for bit. Each of the instance's N columns takes one run of up to M elements of one channel, with that
 * channel's table; a launch carries N runs.
 *
 * Needs, from the including TU: aiednn_rt_lut.h (lut_plan_shape, lut_tables).
 ******************************************************************************/
#ifndef AIEDNN_RT_EAGER_LUT_H
#define AIEDNN_RT_EAGER_LUT_H

#define RT_EAGER_LUT_TABLE 256 /* the template's table chunks (lut_eager_ref.h: EG_LUT_TABLE) */

typedef struct {
    const aiednn_impl_t *g;
    int64_t M, N;
} rt_eager_lut;

static int rt_eager_lut_of(const aiednn_impl_t *g, rt_eager_lut *q) {
    long long m, n, tm;
    if (g->nargs != 3 || sscanf(g->name, "aie:eager_lut__%lld_%lld_%lld", &m, &n, &tm) != 3 || (n != 64 && n != 128))
        return 0;
    q->g = g, q->M = m, q->N = n;
    return 1;
}

/* the instance with the most columns x rows: fewest launches */
static int rt_eager_lut_find(rt_eager_lut *out) {
    int found = 0;
    for (int i = 0; i < aiednn_impl_count; i++) {
        rt_eager_lut q;
        if (rt_eager_lut_of(&aiednn_impl_list[i], &q) && (!found || q.M * q.N > out->M * out->N))
            *out = q, found = 1;
    }
    return found;
}

static int acc_eager_lut(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    lut_plan q;
    if (!lut_plan_shape(p, &q, why, cap))
        return 0;
    const aiednn_data_type_t mt = p->md[q.main_slot].data_type, dt = p->md[q.dst_slot].data_type;
    if ((mt != aiednn_s8 && mt != aiednn_u8) || (dt != aiednn_s8 && dt != aiednn_u8)) {
        snprintf(why, cap, "an 8-bit full-size source and an 8-bit dst");
        return 0;
    }
    rt_eager_lut e;
    if (!rt_eager_lut_find(&e)) {
        snprintf(why, cap, "no lut instance");
        return 0;
    }
    return 1;
}

static void *g_el_tab, *g_el_a, *g_el_w, *g_el_y;
static size_t g_el_tab_cap, g_el_a_cap, g_el_w_cap, g_el_y_cap;

static int rt_eager_lut_launch(const rt_impl *self, const rt_eager_lut *e, int8_t *a, int8_t *w, int8_t *y) {
    const int64_t ld = RT_EAGER_LUT_TABLE + e->N;
    const int64_t a_shape[2] = {e->M, ld}, w_shape[2] = {e->N, ld}, y_shape[2] = {e->M, e->N};
    aiek_tensor t[3] = {{a, 2, AIEK_INT, 8, 1, a_shape, (uint64_t)(e->M * ld)},
                        {w, 2, AIEK_INT, 8, 1, w_shape, (uint64_t)(e->N * ld)},
                        {y, 2, AIEK_INT, 8, 1, y_shape, (uint64_t)(e->M * e->N)}};
    const uint64_t te = rt_ticks();
    const int rc = e->g->entry(t, 3);
    ((rt_impl *)self)->t_entry += rt_ticks() - te;
    ((rt_impl *)self)->launches++;
    return rc;
}

static int run_eager_lut(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    lut_plan q;
    char why[160];
    rt_eager_lut e;
    if (!lut_plan_shape(p, &q, why, sizeof why) || !rt_eager_lut_find(&e))
        return AIEK_ERR_ARGS;
    const int64_t ld = RT_EAGER_LUT_TABLE + e.N;
    uint8_t *tab = (uint8_t *)scratch(&g_el_tab, &g_el_tab_cap, (size_t)(q.C * 256));
    int8_t *a = (int8_t *)scratch(&g_el_a, &g_el_a_cap, (size_t)(e.M * ld));
    int8_t *w = (int8_t *)scratch(&g_el_w, &g_el_w_cap, (size_t)(e.N * ld));
    int8_t *y = (int8_t *)scratch(&g_el_y, &g_el_y_cap, (size_t)(e.M * e.N));
    if (!tab || !a || !w || !y)
        return AIEK_ERR_ALLOC;
    const uint64_t tt0 = rt_ticks();
    int rc = lut_tables(p, &q, ptr, tab);
    g_phase[RT_PH_BPACK] += rt_ticks() - tt0;
    if (rc != AIEK_OK)
        return rc;
    const uint8_t *x = (const uint8_t *)ptr[q.main_slot];
    uint8_t *dst = (uint8_t *)ptr[q.dst_slot];
    int64_t e0s[128], ns[128];
    memset(a, 0, (size_t)(e.M * ld));
    memset(w, 0, (size_t)(e.N * ld));
    for (int64_t el = 0; el < q.numel;) {
        const uint64_t tp0 = rt_ticks();
        int64_t col = 0;
        for (; col < e.N && el < q.numel; col++) { /* column col: one run of one channel */
            const int64_t run = q.inner - el % q.inner, n = run < e.M ? run : e.M;
            const int64_t ch = el / q.inner % q.C;
            memcpy(w + col * ld, tab + ch * 256, 256);
            for (int64_t m = 0; m < n; m++)
                a[m * ld + RT_EAGER_LUT_TABLE + col] = (int8_t)x[el + m];
            e0s[col] = el, ns[col] = n;
            el += n;
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        if ((rc = rt_eager_lut_launch(self, &e, a, w, y)) != AIEK_OK)
            return rc;
        const uint64_t tu = rt_ticks();
        for (int64_t k = 0; k < col; k++)
            for (int64_t m = 0; m < ns[k]; m++)
                dst[e0s[k] + m] = (uint8_t)y[m * e.N + k];
        g_phase[RT_PH_CUNPACK] += rt_ticks() - tu;
    }
    return AIEK_OK;
}

#endif /* AIEDNN_RT_EAGER_LUT_H */
