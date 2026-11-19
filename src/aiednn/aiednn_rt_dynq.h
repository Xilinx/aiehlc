/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_dynq.h -- gen_index and dynamic quantize / dequantize (scales / zero points as run-time tensors).
 *
 * gen_index writes the s32 position along one axis. dynamic quantize is the reorder's arithmetic with the
 * scale / zero point read at execute: ref: per element (per tensor or per channel along axis); dynq+ (per
 * tensor, impl "dynq") creates -- once per value pair, through the quiet dispatch memo -- the static reorder with those
 * values and runs whatever implements it, so the AIE's steps: / lut: modes serve it where they serve a reorder.
 *
 * Needs, from the including TU: aiednn_rt_internal.h and the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_DYNQ_H
#define AIEDNN_RT_DYNQ_H

static int run_ref_gen_index(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *d = &p->md[0];
    int64_t inner = 1;
    for (int i = p->axis + 1; i < d->ndims; i++)
        inner *= d->dims[i];
    const int64_t n = md_numel(d), len = d->dims[p->axis];
    for (int64_t i = 0; i < n; i++)
        ((int32_t *)ptr[0])[i] = (int32_t)((i / inner) % len);
    return AIEK_OK;
}

extern "C" aiednn_status_t aiednn_gen_index_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *dst, int axis) {
    const char *k = "gen_index";
    REQUIRE(pr && md_ok(dst) && dst->data_type == aiednn_s32 && axis >= 0 && axis < dst->ndims, k,
            "s32 dst, axis in range");
    rt_prob p;
    prob_init(&p, aiednn_gen_index, 0);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    p.axis = axis;
    return create(pr, &p);
}

/* source / dst slots 0 / 1, scales 2, zero points 3 (optional); axis -1: per tensor */
static int dynq_to_int(const rt_prob *p) { return dt_is_int(p->md[1].data_type); }

static int run_ref_dynq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1], *sm = &p->md[2];
    const int zs = rt_slot(p, AIEDNN_ARG_ZERO_POINTS);
    const aiednn_memory_desc_t *zm = zs >= 0 ? &p->md[zs] : nullptr;
    int64_t inner = 1, len = 1;
    if (p->axis >= 0) {
        len = s->dims[p->axis];
        for (int i = p->axis + 1; i < s->ndims; i++)
            inner *= s->dims[i];
    }
    const double lo = d->data_type == aiednn_s8 ? -128.0 : 0.0, hi = lo + 255.0;
    const int toi = dynq_to_int(p);
    const int64_t n = md_numel(s);
    for (int64_t i = 0; i < n; i++) {
        const int64_t c = p->axis >= 0 ? (i / inner) % len : 0;
        const double sc = (float)rt_ld(sm->data_type, ptr[2], c); /* a float scale, as the reorder's */
        const double zp = zm ? (double)(int32_t)rt_ld(zm->data_type, ptr[zs], c) : 0.0;
        double v = rt_ld(s->data_type, ptr[0], i);
        if (toi) {
            v = nearbyint(v / sc) + zp;
            v = v != v ? 0.0 : v < lo ? lo : v > hi ? hi : v;
        } else {
            v = (v - zp) * sc;
        }
        rt_st(d->data_type, ptr[1], i, v);
    }
    return AIEK_OK;
}

/* the per-tensor problem as the static reorder with scale / zero point (the values only change the memo key) */
static void dynq_reorder_prob(const rt_prob *p, float sc, int32_t zp, rt_prob *r) {
    prob_init(r, aiednn_reorder, 0);
    prob_add(r, AIEDNN_ARG_SRC, &p->md[0]);
    prob_add(r, AIEDNN_ARG_DST, &p->md[1]);
    for (int i = 0; i < p->md[0].ndims; i++)
        r->perm[i] = i;
    r->scale = sc, r->zero_point = zp;
}

static int acc_dynq_reorder(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self;
    if (p->axis >= 0) {
        snprintf(why, cap, "per tensor only (per channel: ref:)");
        return 0;
    }
    rt_prob r;
    dynq_reorder_prob(p, 0.0625f, 0, &r);
    const rt_impl *in = nullptr;
    rt_bind b = {nullptr, nullptr};
    if (dispatch(&r, &in, &b, 1) != aiednn_success || strncmp(in->name, "ref:", 4) == 0) {
        snprintf(why, cap, "its reorder would run on %s", in ? in->name : "nothing");
        return 0;
    }
    bind->inner = in, bind->blk = b.blk;
    return 1;
}

static int run_dynq_reorder(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const int zs = rt_slot(p, AIEDNN_ARG_ZERO_POINTS);
    const float sc = (float)rt_ld(p->md[2].data_type, ptr[2], 0);
    const int32_t zp = zs >= 0 ? (int32_t)rt_ld(p->md[zs].data_type, ptr[zs], 0) : 0;
    if (!(sc != 0) || sc != sc)
        return AIEK_ERR_ARGS;
    rt_prob r;
    dynq_reorder_prob(p, sc, zp, &r);
    const rt_impl *in = nullptr;
    rt_bind b = {nullptr, nullptr};
    if (dispatch(&r, &in, &b, 1) != aiednn_success)
        return AIEK_ERR_ARGS;
    void *rp[2] = {ptr[0], ptr[1]};
    return run_sub(in, &r, &b, rp);
}

extern "C" aiednn_status_t aiednn_dynamic_quantize_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *src,
                                                          const aiednn_memory_desc_t *dst,
                                                          const aiednn_memory_desc_t *scales,
                                                          const aiednn_memory_desc_t *zero_points, int axis) {
    const char *k = "dynamic_quantize";
    REQUIRE(pr && md_ok(src) && md_ok(dst) && md_ok(scales) && (!zero_points || md_ok(zero_points)) &&
                md_same_shape(src, dst),
            k, "src and dst: same shape");
    const int q = dt_is_float(src->data_type) && (dst->data_type == aiednn_s8 || dst->data_type == aiednn_u8);
    const int dq = (src->data_type == aiednn_s8 || src->data_type == aiednn_u8 || src->data_type == aiednn_s32) &&
                   dt_is_float(dst->data_type);
    REQUIRE(q || dq, k, "floating -> s8 / u8 (quantize) or s8 / u8 / s32 -> floating (dequantize)");
    REQUIRE(axis >= -1 && axis < src->ndims, k, "axis: -1 (per tensor) or a dim of src");
    const int64_t nc = axis < 0 ? 1 : src->dims[axis];
    REQUIRE(md_numel(scales) == nc && dt_is_float(scales->data_type), k, "floating scales: 1 or dims[axis] values");
    REQUIRE(!zero_points || (md_numel(zero_points) == nc && dt_is_int(zero_points->data_type)), k,
            "integer zero points of the scales' count");
    rt_prob p;
    prob_init(&p, aiednn_dynamic_quantize, 0);
    prob_add(&p, AIEDNN_ARG_SRC, src);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    prob_add(&p, AIEDNN_ARG_SCALES, scales);
    if (zero_points)
        prob_add(&p, AIEDNN_ARG_ZERO_POINTS, zero_points);
    p.axis = axis;
    return create(pr, &p);
}

#endif /* AIEDNN_RT_DYNQ_H */
