/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_eltblk.h -- eltblk:<aie eltwise block>: binary on two 8-bit tensors
 * on the GEMM ELF's eltwise mode (aie_modes_pool.h, opcode 0xE1).
 *
 * The core returns the exact int32 a op b for add / sub / mul / max / min, and
 * a - b for the comparisons (eq .. le: its sign decides). ref:binary computes
 * the same small integers in double, so storing them through rt_st gives ref's
 * result bit for bit, for any destination type -- except the sign of a zero
 * product (0 * -3 is -0.0 in double, visible in an f32 dst), which the finish
 * takes from the packed operands. div and pow are not exact from
 * one integer and stay on ref:. Either operand may broadcast; the host packs
 * the broadcast values per element. A problem lut: takes (one operand
 * broadcast along at most one axis) is left to it.
 *
 * Kernel block: items s8 [NI][4096] = a at bytes [0, 1024), b at [2048, 3072),
 * geom s32 [3] = {op, signed bits, n}, out s32 [NI][1024].
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aie_modes_pool.h,
 * aiednn_rt_lut.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_ELTBLK_H
#define AIEDNN_RT_ELTBLK_H

#define RT_EB_MAX_ITEMS 64

static int is_elt_block(const aiednn_impl_t *g) {
    return g->kind == aiednn_binary && g->nargs == 3 && g->args[0].data_type == aiednn_s8 && g->args[0].ndims == 2 &&
           g->args[0].dims[0] >= 1 && g->args[0].dims[0] <= RT_EB_MAX_ITEMS && g->args[0].dims[1] == AIE_MODE_IN &&
           g->args[1].data_type == aiednn_s32 && g->args[1].ndims == 1 && g->args[1].dims[0] == 3 &&
           g->args[2].data_type == aiednn_s32 && g->args[2].dims[1] == AIE_MODE_OUT;
}

/* binary alg -> the core's op, or -1 */
static int eltblk_op(int alg) {
    switch (alg) {
    case aiednn_binary_add:
        return AIE_MODE_ELT_ADD;
    case aiednn_binary_mul:
        return AIE_MODE_ELT_MUL;
    case aiednn_binary_max:
        return AIE_MODE_ELT_MAX;
    case aiednn_binary_min:
        return AIE_MODE_ELT_MIN;
    case aiednn_binary_sub:
    case aiednn_binary_eq:
    case aiednn_binary_ne:
    case aiednn_binary_gt:
    case aiednn_binary_ge:
    case aiednn_binary_lt:
    case aiednn_binary_le:
        return AIE_MODE_ELT_SUB;
    }
    return -1;
}

static int acc_eltblk(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    (void)bind;
    if (p->kind != aiednn_binary || eltblk_op(p->alg) < 0 || p->has_attr) {
        snprintf(why, cap, "binary add / sub / mul / max / min / compare only, no attr (the quantized binary: fvec:)");
        return 0;
    }
    const aiednn_memory_desc_t *a = rt_md(p, AIEDNN_ARG_SRC), *b = rt_md(p, AIEDNN_ARG_SRC_1);
    if ((a->data_type != aiednn_s8 && a->data_type != aiednn_u8) ||
        (b->data_type != aiednn_s8 && b->data_type != aiednn_u8)) {
        snprintf(why, cap, "both sources s8 or u8");
        return 0;
    }
    lut_plan q;
    char lw[8];
    if (lut_plan_make(p, &q, lw, sizeof lw)) {
        snprintf(why, cap, "one operand broadcast along at most one axis: lut: serves it");
        return 0;
    }
    return 1;
}

/* the core's exact integer r -> ref:binary's value */
static double eltblk_value(int alg, int32_t r) {
    switch (alg) {
    case aiednn_binary_eq:
        return r == 0;
    case aiednn_binary_ne:
        return r != 0;
    case aiednn_binary_gt:
        return r > 0;
    case aiednn_binary_ge:
        return r >= 0;
    case aiednn_binary_lt:
        return r < 0;
    case aiednn_binary_le:
        return r <= 0;
    }
    return (double)r;
}

static void *g_ebi, *g_ebo;
static size_t g_ebi_cap, g_ebo_cap;

typedef struct {
    const rt_prob *p;
    const uint8_t *x, *y;
    void *dst;
    int64_t n;
    int a_full, b_full, sgn;
} elt_itn_ctx;
/* item k = elements [k * AIE_MODE_OUT, +AIE_MODE_OUT): a at 0, b at AIE_MODE_B_OFF */
static void elt_itn_pack(void *c, int64_t k, uint8_t *it) {
    const elt_itn_ctx *e = (const elt_itn_ctx *)c;
    const aiednn_memory_desc_t *a = &e->p->md[0], *b = &e->p->md[1], *d = &e->p->md[2];
    memset(it, 0, AIE_MODE_IN);
    const int64_t o0 = k * AIE_MODE_OUT, m = e->n - o0 < AIE_MODE_OUT ? e->n - o0 : AIE_MODE_OUT;
    if (e->a_full)
        memcpy(it, e->x + o0, (size_t)m);
    else
        for (int64_t i = 0; i < m; i++)
            it[i] = e->x[bcast_off(a, d, o0 + i)];
    if (e->b_full)
        memcpy(it + AIE_MODE_B_OFF, e->y + o0, (size_t)m);
    else
        for (int64_t i = 0; i < m; i++)
            it[AIE_MODE_B_OFF + i] = e->y[bcast_off(b, d, o0 + i)];
}
static void elt_itn_put(void *c, int64_t k, const int32_t *res) {
    const elt_itn_ctx *e = (const elt_itn_ctx *)c;
    const aiednn_memory_desc_t *a = &e->p->md[0], *b = &e->p->md[1], *d = &e->p->md[2];
    const int64_t o0 = k * AIE_MODE_OUT, m = e->n - o0 < AIE_MODE_OUT ? e->n - o0 : AIE_MODE_OUT;
    const int zsign = e->p->alg == aiednn_binary_mul && d->data_type == aiednn_f32;
    for (int64_t i = 0; i < m; i++) {
        double v = eltblk_value(e->p->alg, res[i]);
        if (zsign && res[i] == 0) { /* ref's double product, with its signed zero */
            const int64_t o = o0 + i;
            const uint8_t xa = e->x[e->a_full ? o : bcast_off(a, d, o)], yb = e->y[e->b_full ? o : bcast_off(b, d, o)];
            v = (double)AIE_MODE_LD(&xa, 0, e->sgn & 1) * (double)AIE_MODE_LD(&yb, 0, e->sgn >> 1);
        }
        rt_st(d->data_type, e->dst, o0 + i, v);
    }
}

static int run_eltblk(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *a = &p->md[0], *b = &p->md[1], *d = &p->md[2];
    const int64_t NI = self->gen->args[0].dims[0], n = md_numel(d);
    { /* many launches' worth of items in one (aiednn_rt_itemsn.h) */
        const int32_t g[3] = {eltblk_op(p->alg), (a->data_type == aiednn_s8) | (b->data_type == aiednn_s8) << 1,
                              AIE_MODE_OUT};
        uint8_t dsc[4];
        AIE_MODE_ELTWISE_DESC(dsc, g);
        elt_itn_ctx k = {p, (const uint8_t *)ptr[0], (const uint8_t *)ptr[1], ptr[2],
                         n, same_dims(a, d),         same_dims(b, d),         g[1]};
        const int rc = rt_items_n(self, (n + AIE_MODE_OUT - 1) / AIE_MODE_OUT, AIE_MODE_OP_ELTWISE, dsc, 4,
                                  elt_itn_pack, elt_itn_put, &k);
        if (rc != RT_NOT_TAKEN)
            return rc;
    }
    uint8_t *items = (uint8_t *)scratch(&g_ebi, &g_ebi_cap, (size_t)(NI * AIE_MODE_IN));
    int32_t *out = (int32_t *)scratch(&g_ebo, &g_ebo_cap, (size_t)(NI * AIE_MODE_OUT) * sizeof(int32_t));
    if (!items || !out)
        return AIEK_ERR_ALLOC;
    const uint8_t *x = (const uint8_t *)ptr[0], *y = (const uint8_t *)ptr[1];
    const int a_full = same_dims(a, d), b_full = same_dims(b, d);
    int32_t geom[3] = {eltblk_op(p->alg), (a->data_type == aiednn_s8) | (b->data_type == aiednn_s8) << 1, AIE_MODE_OUT};
    const int64_t shape_i[2] = {NI, AIE_MODE_IN}, shape_g[1] = {3}, shape_o[2] = {NI, AIE_MODE_OUT};
    for (int64_t e = 0; e < n; e += NI * AIE_MODE_OUT) {
        const uint64_t tp0 = rt_ticks();
        for (int64_t k = 0; k < NI; k++) {
            uint8_t *it = items + k * AIE_MODE_IN;
            for (int64_t i = 0; i < AIE_MODE_OUT; i++) {
                const int64_t o = e + k * AIE_MODE_OUT + i;
                it[i] = o < n ? x[a_full ? o : bcast_off(a, d, o)] : 0;
                it[AIE_MODE_B_OFF + i] = o < n ? y[b_full ? o : bcast_off(b, d, o)] : 0;
            }
        }
        g_phase[RT_PH_DWPACK] += rt_ticks() - tp0;
        aiek_tensor t[3] = {{items, 2, AIEK_INT, 8, 1, shape_i, (uint64_t)(NI * AIE_MODE_IN)},
                            {geom, 1, AIEK_INT, 32, 1, shape_g, sizeof geom},
                            {out, 2, AIEK_INT, 32, 1, shape_o, (uint64_t)NI * AIE_MODE_OUT * sizeof(int32_t)}};
        ((rt_impl *)self)->launches++;
        const uint64_t te = rt_ticks();
        const int rc = self->gen->entry(t, 3);
        ((rt_impl *)self)->t_entry += rt_ticks() - te;
        if (rc != AIEK_OK)
            return rc;
        const uint64_t te0 = rt_ticks();
        const int64_t m = n - e < NI * AIE_MODE_OUT ? n - e : NI * AIE_MODE_OUT;
        const int zsign = p->alg == aiednn_binary_mul && d->data_type == aiednn_f32;
        for (int64_t i = 0; i < m; i++) {
            double v = eltblk_value(p->alg, out[i]);
            if (zsign && out[i] == 0) { /* ref's double product, with its signed zero */
                const uint8_t *it = items + (i / AIE_MODE_OUT) * AIE_MODE_IN + i % AIE_MODE_OUT;
                v = (double)AIE_MODE_LD(it, 0, geom[1] & 1) * (double)AIE_MODE_LD(it, AIE_MODE_B_OFF, geom[1] >> 1);
            }
            rt_st(d->data_type, ptr[2], e + i, v);
        }
        g_phase[RT_PH_EPI] += rt_ticks() - te0;
    }
    return AIEK_OK;
}

/* x86 stand-in for the AIE eltwise block (the macro the core runs). */
static int rt_elt_block_x86(const aiek_tensor *t, int n) {
    if (n != 3)
        return AIEK_ERR_ARGS;
    uint8_t dsc[4];
    AIE_MODE_ELTWISE_DESC(dsc, (const int32_t *)t[1].data);
    for (int64_t k = 0; k < t[0].shape[0]; k++)
        AIE_MODE_ELTWISE((const int8_t *)t[0].data + k * AIE_MODE_IN, (int32_t *)t[2].data + k * AIE_MODE_OUT, dsc);
    return AIEK_OK;
}

#endif /* AIEDNN_RT_ELTBLK_H */
