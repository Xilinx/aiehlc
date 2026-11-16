/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_slice.h -- slice: a strided sub-tensor copy. dst[i0, i1, ...] = src[off0 + i0 * st0, off1 + i1 * st1, ...]
 * (dst dims are the sizes). ref: only; contiguous rows are memcpy'd. A slice never converts types.
 *
 * Needs, from the including TU: aiednn_rt_internal.h and the create helpers of aiednn_rt.cc.
 ******************************************************************************/
#ifndef AIEDNN_RT_SLICE_H
#define AIEDNN_RT_SLICE_H

static int run_ref_slice(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const int nd = d->ndims;
    const int64_t esz = dt_bytes(d->data_type);
    int64_t sstr[AIEDNN_MAX_NDIMS], st = 1;
    for (int i = nd - 1; i >= 0; i--)
        sstr[i] = st, st *= s->dims[i];
    const int64_t W = d->dims[nd - 1], rows = md_numel(d) / W;
    const int contig = p->slice_step[nd - 1] == 1;
    /* dst quantization alone: the quantizing reorder's arithmetic, bit-identical to slice + quantize */
    const int q = p->has_attr && s->data_type == aiednn_f32 && (d->data_type == aiednn_s8 || d->data_type == aiednn_u8);
    const double qs = p->dst_scale, qinv = 1.0 / qs, qz = p->dst_zp, qlo = d->data_type == aiednn_s8 ? -128.0 : 0.0;
    for (int64_t r = 0; q && r < rows; r++) {
        int64_t rem = r, off = p->slice_off[nd - 1] * sstr[nd - 1];
        for (int i = nd - 2; i >= 0; i--) {
            off += (p->slice_off[i] + (rem % d->dims[i]) * p->slice_step[i]) * sstr[i];
            rem /= d->dims[i];
        }
        const float *src = (const float *)ptr[0] + off;
        uint8_t *dst = (uint8_t *)ptr[1] + r * W;
        const int64_t st1 = p->slice_step[nd - 1];
        for (int64_t j = 0; j < W; j++)
            dst[j] = (uint8_t)(int)rt_quant8(src[j * st1], qs, qinv, qz, qlo);
    }
    if (q)
        return AIEK_OK;
    for (int64_t r = 0; r < rows; r++) {
        int64_t rem = r, off = p->slice_off[nd - 1] * sstr[nd - 1];
        for (int i = nd - 2; i >= 0; i--) {
            off += (p->slice_off[i] + (rem % d->dims[i]) * p->slice_step[i]) * sstr[i];
            rem /= d->dims[i];
        }
        const char *src = (const char *)ptr[0] + off * esz;
        char *dst = (char *)ptr[1] + r * W * esz;
        if (contig)
            memcpy(dst, src, (size_t)(W * esz));
        else
            for (int64_t j = 0; j < W; j++)
                memcpy(dst + j * esz, src + j * p->slice_step[nd - 1] * esz, (size_t)esz);
    }
    return AIEK_OK;
}

static aiednn_status_t slice_create_(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                     const aiednn_memory_desc_t *d, const int64_t *offsets, const int64_t *steps,
                                     const aiednn_attr_t *attr) {
    const char *k = "slice";
    REQUIRE(pr && md_ok(s) && md_ok(d) && s->ndims == d->ndims && !dt_is_4bit(s->data_type), k,
            "src and dst: same rank (not 4-bit)");
    REQUIRE(attr ? s->data_type == aiednn_f32 &&
                       (d->data_type == aiednn_f32 || d->data_type == aiednn_s8 || d->data_type == aiednn_u8)
                 : s->data_type == d->data_type,
            k, "same type; with an attr: f32 src, f32 / s8 / u8 dst");
    rt_prob p;
    prob_init(&p, aiednn_slice, 0);
    for (int i = 0; i < d->ndims; i++) {
        const int64_t o = offsets ? offsets[i] : 0, t = steps ? steps[i] : 1;
        REQUIRE(o >= 0 && t >= 1 && o + (d->dims[i] - 1) * t < s->dims[i], k,
                "offset >= 0, step >= 1, the last element inside src");
        p.slice_off[i] = o, p.slice_step[i] = t;
    }
    prob_add(&p, AIEDNN_ARG_SRC, s);
    prob_add(&p, AIEDNN_ARG_DST, d);
    if (attr) { /* dst quantization alone: run_ref_slice; post-ops: aiednn_rt_ptail.h */
        const char *why = attr->n_scales > 1 ? "per-tensor output scale only" : attach_attr(&p, attr, 0);
        if (!why && d->data_type == aiednn_f32 && !p.npo && !p.n_oscales)
            why = "an f32 dst with no post-ops: nothing for the attr to do (use aiednn_slice_create)";
        if (why) {
            free((void *)p.oscales);
            return bad(k, why);
        }
    }
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_slice_create(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                               const aiednn_memory_desc_t *d, const int64_t *offsets,
                                               const int64_t *steps) {
    return slice_create_(pr, s, d, offsets, steps, nullptr);
}

extern "C" aiednn_status_t aiednn_slice_create_attr(aiednn_primitive_t *pr, const aiednn_memory_desc_t *s,
                                                    const aiednn_memory_desc_t *d, const int64_t *offsets,
                                                    const int64_t *steps, const aiednn_attr_t *attr) {
    return slice_create_(pr, s, d, offsets, steps, attr);
}

#endif /* AIEDNN_RT_SLICE_H */
