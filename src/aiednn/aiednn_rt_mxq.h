/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_mxq.h -- block-scale (OCP microscaling, MX) quantize / dequantize along the last axis.
 *
 *   quantize   f32 / bf16 / f16 src -> f8_e4m3 / f8_e5m2 / f4_e2m1 elements + one e8m0 scale per block
 *   dequantize those -> f32 / bf16 / f16
 *
 * Per block of `block` elements (32 for MXFP8 / MXFP4): shared exponent s = floor(log2(max |x|)) - emax with
 * emax = 8 (e4m3), 15 (e5m2), 2 (e2m1), clamped to e8m0's [-127, 127]; elements x / 2^s rounded to nearest even
 * and saturated to the type's largest normal (448, 57344, 6); a zero block takes 2^-127. A NaN in a block makes
 * its scale NaN (0xff). Dequantize: element x 2^s (NaN scale -> NaN). Bit-exact against the numpy model in
 * test_aiednn_mx.py.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_dtypes.h, aiednn_rt.cc's create helpers.
 ******************************************************************************/
#ifndef AIEDNN_RT_MXQ_H
#define AIEDNN_RT_MXQ_H

static int mxq_emax(aiednn_data_type_t dt, double *vmax) {
    *vmax = dt == aiednn_f8_e4m3 ? 448.0 : dt == aiednn_f8_e5m2 ? 57344.0 : 6.0;
    return dt == aiednn_f8_e4m3 ? 8 : dt == aiednn_f8_e5m2 ? 15 : 2;
}

static int run_ref_blockq(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const int quant = p->flags == 0; /* 0: quantize (SRC -> DST + SCALE), 1: dequantize (SRC + SCALE -> DST) */
    const aiednn_memory_desc_t *s = rt_md(p, AIEDNN_ARG_SRC), *d = rt_md(p, AIEDNN_ARG_DST),
                               *sc = rt_md(p, AIEDNN_ARG_SCALE);
    const void *x = rt_ptr(p, ptr, AIEDNN_ARG_SRC);
    void *y = rt_ptr(p, ptr, AIEDNN_ARG_DST), *scp = rt_ptr(p, ptr, AIEDNN_ARG_SCALE);
    const aiednn_data_type_t et = quant ? d->data_type : s->data_type; /* the MX element type */
    double vmax;
    const int emax = mxq_emax(et, &vmax);
    const int64_t n = md_numel(s), bs = p->group_size, nb = n / bs;
    for (int64_t b = 0; b < nb; b++) {
        if (!quant) {
            const float sv = rt_e8m0_to_f32(((const uint8_t *)scp)[b]);
            for (int64_t i = b * bs; i < (b + 1) * bs; i++)
                rt_st(d->data_type, y, i, (double)(float)(rt_ld(s->data_type, x, i) * (double)sv));
            continue;
        }
        double amax = 0;
        int nan = 0;
        for (int64_t i = b * bs; i < (b + 1) * bs; i++) {
            const double v = rt_ld(s->data_type, x, i);
            nan |= v != v;
            amax = fabs(v) > amax ? fabs(v) : amax;
        }
        int se = -127;
        if (amax > 0 && !nan && amax < INFINITY) {
            int e;
            frexp(amax, &e); /* amax = f 2^e, f in [0.5, 1): floor(log2 amax) = e - 1 */
            se = e - 1 - emax;
            se = se < -127 ? -127 : se > 127 ? 127 : se;
        }
        ((uint8_t *)scp)[b] = nan ? 0xff : (uint8_t)(se + 127);
        for (int64_t i = b * bs; i < (b + 1) * bs; i++) {
            double v = ldexp(rt_ld(s->data_type, x, i), -se); /* x / 2^s: exact unless it underflows */
            if (v == v)
                v = v > vmax ? vmax : v < -vmax ? -vmax : v;
            rt_st(d->data_type, y, i, v);
        }
    }
    return AIEK_OK;
}

static aiednn_status_t blockq_create(aiednn_primitive_t *pr, int dequant, const aiednn_memory_desc_t *src,
                                     const aiednn_memory_desc_t *scales, const aiednn_memory_desc_t *dst,
                                     int64_t block) {
    const char *k = "block_quantize";
    REQUIRE(pr && md_ok(src) && md_ok(scales) && md_ok(dst) && md_same_shape(src, dst), k, "src and dst: same shape");
    const aiednn_data_type_t ft = dequant ? dst->data_type : src->data_type,
                             et = dequant ? src->data_type : dst->data_type;
    REQUIRE(ft == aiednn_f32 || ft == aiednn_bf16 || ft == aiednn_f16, k, "the wide side: f32 / bf16 / f16");
    REQUIRE(et == aiednn_f8_e4m3 || et == aiednn_f8_e5m2 || et == aiednn_f4_e2m1, k,
            "elements: f8_e4m3 / f8_e5m2 / f4_e2m1");
    REQUIRE(scales->data_type == aiednn_e8m0, k, "scales: e8m0");
    const int nd = src->ndims;
    REQUIRE(block >= 1 && src->dims[nd - 1] % block == 0 && scales->ndims == nd &&
                scales->dims[nd - 1] == src->dims[nd - 1] / block,
            k, "scales [..., K / block] for src [..., K] (block dividing K)");
    for (int i = 0; i < nd - 1; i++)
        REQUIRE(scales->dims[i] == src->dims[i], k, "scales share src's leading dims");
    rt_prob p;
    prob_init(&p, aiednn_block_quantize, 0);
    prob_add(&p, AIEDNN_ARG_SRC, src);
    prob_add(&p, AIEDNN_ARG_SCALE, scales);
    prob_add(&p, AIEDNN_ARG_DST, dst);
    p.group_size = block;
    p.flags = dequant ? 1u : 0u;
    return create(pr, &p);
}

extern "C" aiednn_status_t aiednn_block_quantize_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                        const aiednn_memory_desc_t *dst,
                                                        const aiednn_memory_desc_t *scales, int64_t block) {
    return blockq_create(p, 0, src, scales, dst, block);
}
extern "C" aiednn_status_t aiednn_block_dequantize_create(aiednn_primitive_t *p, const aiednn_memory_desc_t *src,
                                                          const aiednn_memory_desc_t *scales,
                                                          const aiednn_memory_desc_t *dst, int64_t block) {
    return blockq_create(p, 1, src, scales, dst, block);
}

#endif /* AIEDNN_RT_MXQ_H */
