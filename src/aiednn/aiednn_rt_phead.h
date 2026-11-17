/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_phead.h -- dequantized sources: an s8 / u8 input read through its scale and zero point.
 *
 * A quantized framework graph feeds float ops through a dequantize: DQ -> layer_norm, DQ -> softmax,
 * DQ x3 -> sdpa (PT2E, ONNX QDQ). With attr.src_scales[i] > 0 on such a create (attr.src_scales_23 for
 * sources 2, 3) the op takes the 8-bit tensor directly: create dispatches the f32 problem, execute
 * dequantizes the source into scratch -- f32((code - zp) * scale), the dequantize reorder's arithmetic, so
 * the result is bit-identical to the two primitives -- and runs it. One primitive, one pass less over memory
 * for the graph (aiednn_graph.h reads through the DQ instead of keeping it as a partition).
 *
 * Needs, from the including TU: aiednn_rt_internal.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_PHEAD_H
#define AIEDNN_RT_PHEAD_H

/* the attr's scale / zero point of source i */
static float attr_src_sc(const aiednn_attr_t *a, int i) { return i < 2 ? a->src_scales[i] : a->src_scales_23[i - 2]; }
static int32_t attr_src_zp(const aiednn_attr_t *a, int i) {
    return i < 2 ? a->src_zero_points[i] : a->src_zero_points_23[i - 2];
}

/* nothing but source dequantization in the attr (no post-ops, scales or dst quantization) */
static int attr_only_src_dq(const aiednn_attr_t *a) {
    return a && !a->n_post_ops && !a->n_scales && (a->dst_scale == 0 || a->dst_scale == 1) && !a->dst_zero_point;
}

/* mark the first nsrc sources: an s8 / u8 one needs a scale > 0, a floating one none; NULL or the reason */
static const char *attach_src_dq(rt_prob *p, const aiednn_attr_t *a, int nsrc) {
    for (int i = 0; i < nsrc && i < 4; i++) {
        const aiednn_data_type_t dt = p->md[i].data_type;
        const float sc = a ? attr_src_sc(a, i) : 0.0f;
        if (dt == aiednn_s8 || dt == aiednn_u8) {
            if (!(sc > 0))
                return "an s8 / u8 source needs attr src_scales[i] > 0 (dequantized as read)";
            p->ndq = i + 1, p->dq_sc[i] = sc, p->dq_zp[i] = attr_src_zp(a, i);
        } else if (sc != 0) {
            return "attr src_scales: only for an s8 / u8 source";
        }
    }
    return nullptr;
}

/* create: the problem dispatch sees (dequantized sources as f32); the primitive keeps the 8-bit side */
static void phead_split(const rt_prob *p0, rt_prob *f) {
    *f = *p0;
    for (int i = 0; i < p0->ndq; i++)
        if (p0->dq_sc[i] > 0)
            f->md[i].data_type = aiednn_f32;
    f->ndq = 0, f->dq_packed = 0;
    if (p0->dq_packed) /* the unpacked copies are [B, S, H, D] */
        f->flags = (f->flags & ~(unsigned)AIEDNN_SDPA_QKV_PACKED) | AIEDNN_SDPA_QKV_BSHD;
}
static void phead_keep(aiednn_primitive *prim, const rt_prob *p0) {
    prim->ndq = p0->ndq, prim->dq_packed = p0->dq_packed;
    for (int i = 0; i < (p0->dq_packed ? 3 : p0->ndq); i++) /* packed: all three slices, dequantized or not */
        prim->dq_md[i] = p0->md[i], prim->dq_sc[i] = i < p0->ndq ? p0->dq_sc[i] : 0.0f,
        prim->dq_zp[i] = i < p0->ndq ? p0->dq_zp[i] : 0;
}

static void *g_phead_buf[4];
static size_t g_phead_cap[4];

/* n elements from x (8-bit codes or raw dt) into dst: dequantized to f32 when sc > 0, else copied */
static void phead_rows(aiednn_data_type_t dt, const void *x, int64_t n, double sc, double zp, void *dst) {
    if (!(sc > 0)) {
        memcpy(dst, x, (size_t)(n * dt_bytes(dt)));
    } else if (dt == aiednn_s8) {
        float *f = (float *)dst;
        for (int64_t e = 0; e < n; e++)
            f[e] = (float)(((double)((const int8_t *)x)[e] - zp) * sc);
    } else {
        float *f = (float *)dst;
        for (int64_t e = 0; e < n; e++)
            f[e] = (float)(((double)((const uint8_t *)x)[e] - zp) * sc);
    }
}

/* execute: a dequantized and / or packed source -> its own buffer (f32 when dequantized) */
static int phead_in(const aiednn_primitive *prim, void **ptr) {
    const uint64_t t0 = rt_ticks();
    const int nsrc = prim->dq_packed ? 3 : prim->ndq;
    void *base[4] = {ptr[0], ptr[1], ptr[2], nullptr};
    for (int i = 0; i < nsrc; i++) {
        const double sc = i < prim->ndq ? prim->dq_sc[i] : 0.0, zp = i < prim->ndq ? prim->dq_zp[i] : 0.0;
        if (!(sc > 0) && !prim->dq_packed)
            continue;
        const aiednn_memory_desc_t *m = &prim->dq_md[i];
        const int64_t n = md_numel(m);
        const size_t es = sc > 0 ? sizeof(float) : (size_t)dt_bytes(m->data_type);
        void *f = scratch(&g_phead_buf[i], &g_phead_cap[i], (size_t)n * es);
        if (!f)
            return AIEK_ERR_ALLOC;
        if (prim->dq_packed) { /* [B, S, 3, H, D] -> [B, S, H, D]: rows of H * D at stride 3 * H * D */
            const int64_t B = m->dims[0], H = m->dims[1], S = m->dims[2], HD = H * m->dims[3];
            const int64_t xb = dt_bytes(m->data_type);
            for (int64_t r = 0; r < B * S; r++)
                phead_rows(m->data_type, (const char *)base[i] + ((r * 3 + i) * HD) * xb, HD, sc, zp,
                           (char *)f + r * HD * es);
        } else {
            phead_rows(m->data_type, base[i], n, sc, zp, f);
        }
        ptr[i] = f;
    }
    g_phase[RT_PH_FMT] += rt_ticks() - t0;
    return AIEK_OK;
}

#endif /* AIEDNN_RT_PHEAD_H */
