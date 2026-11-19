/* aiednn_rt_layout.h -- memory formats (aiednn_format_t).
 *
 * An nhwc descriptor keeps its logical [N, C, H, W] dims; its bytes are those of a dense [N, H, W, C].
 * create() resolves a problem with nhwc operands one of three ways:
 *   canon  -- kinds whose result does not depend on the axis order (eltwise, binary, ...): every nhwc
 *             descriptor becomes the dense descriptor of its bytes, axis parameters follow; zero cost.
 *   native -- the chosen implementation reads / writes nhwc itself (rt_impl.caps & RT_CAP_NHWC and its
 *             acceptor said so for this problem): prim->prob keeps the formats.
 *   wrap   -- anything else: the implementation runs on dense copies; execute converts the nhwc inputs in
 *             and the nhwc outputs back (prim->fmt_mask). Correct everywhere, never faster than dense.
 * Included at the end of aiednn_rt.cc (needs rt_tr8x8 from aiednn_rt_gemm.h). */

static const int RT_NHWC_PHYS[4] = {0, 2, 3, 1}; /* physical axis j holds logical axis RT_NHWC_PHYS[j] */
static const int RT_NHWC_POS[4] = {0, 3, 1, 2};  /* logical axis i sits at physical axis RT_NHWC_POS[i] */

/* dense and nhwc bytes coincide: one channel, or one pixel */
static inline int md_fmt_free(const aiednn_memory_desc_t *m) {
    return m->ndims != 4 || m->dims[1] == 1 || m->dims[2] * m->dims[3] == 1;
}

/* the dense descriptor of an nhwc descriptor's bytes; force: a format-free dense one too (its bytes read
 * the same either way) */
static void md_to_phys(aiednn_memory_desc_t *m, int force = 0) {
    if (m->ndims != 4 || (!force && m->format != aiednn_format_nhwc))
        return;
    const int64_t d[4] = {m->dims[0], m->dims[1], m->dims[2], m->dims[3]};
    for (int j = 0; j < 4; j++)
        m->dims[j] = d[RT_NHWC_PHYS[j]];
    m->format = aiednn_format_dense;
}

/* results independent of the axis order once the axis parameter follows (dropout is not: its mask is drawn
 * in element order, so it would differ from the dense run) */
static int rt_fmt_equivariant(aiednn_primitive_kind_t k) {
    return k == aiednn_eltwise || k == aiednn_binary || k == aiednn_prelu || k == aiednn_sum || k == aiednn_concat ||
           k == aiednn_softmax || k == aiednn_shuffle || k == aiednn_reduction || k == aiednn_reorder ||
           k == aiednn_slice;
}

/* reorder: each operand independently -- dst physical axis j is logical axis Ld(j) of dst, which is logical
 * axis perm[Ld(j)] of src, found at src physical axis Ps(perm[Ld(j)]) */
static void rt_fmt_canon_reorder(rt_prob *p) {
    const int sn = md_is_nhwc(&p->md[0]), dn = md_is_nhwc(&p->md[1]), nd = p->md[1].ndims;
    int perm[AIEDNN_MAX_NDIMS];
    for (int j = 0; j < nd; j++) {
        const int l = p->perm[dn ? RT_NHWC_PHYS[j] : j];
        perm[j] = sn ? RT_NHWC_POS[l] : l;
    }
    memcpy(p->perm, perm, sizeof perm);
    md_to_phys(&p->md[0]);
    md_to_phys(&p->md[1]);
}

/* canon: 1 when p was rewritten (or had no formats), 0 when it does not apply (wrap / native then) */
static int rt_fmt_canon(rt_prob *p) {
    if (!rt_prob_has_fmt(p))
        return 1;
    if (!rt_fmt_equivariant(p->kind) || p->prop != RT_FWD_INF)
        return 0;
    if (p->kind == aiednn_reorder) {
        rt_fmt_canon_reorder(p);
        return 1;
    }
    if (p->n_oscales > 1) /* per-channel output scales index the channel axis */
        return 0;
    for (int s = 0; s < p->nmd; s++) /* every operand 4-D, each nhwc or format-free */
        if (p->md[s].ndims != 4 || (!md_is_nhwc(&p->md[s]) && !md_fmt_free(&p->md[s])))
            return 0;
    for (int s = 0; s < p->nmd; s++)
        md_to_phys(&p->md[s], 1);
    for (int k = 0; k < p->npo; k++) /* the post-op's own copy of its source descriptor */
        if (p->po[k].src1.ndims == 4)
            md_to_phys(&p->po[k].src1, 1);
    if (p->kind == aiednn_softmax || p->kind == aiednn_concat || p->kind == aiednn_shuffle)
        p->axis = RT_NHWC_POS[p->axis];
    if (p->kind == aiednn_slice) { /* physical axis j takes logical axis RT_NHWC_PHYS[j]'s offset / step */
        int64_t o[4], t[4];
        for (int j = 0; j < 4; j++)
            o[j] = p->slice_off[RT_NHWC_PHYS[j]], t[j] = p->slice_step[RT_NHWC_PHYS[j]];
        for (int j = 0; j < 4; j++)
            p->slice_off[j] = o[j], p->slice_step[j] = t[j];
    }
    return 1;
}

/* the dense problem the implementations see (logical dims kept, formats dropped) */
static void rt_fmt_strip(rt_prob *p) {
    for (int s = 0; s < p->nmd; s++)
        p->md[s].format = aiednn_format_dense;
    for (int k = 0; k < p->npo; k++)
        p->po[k].src1.format = aiednn_format_dense;
}

/* arguments an execute writes */
static int rt_arg_written(int arg) {
    return arg == AIEDNN_ARG_DST || arg == AIEDNN_ARG_DST_ITER || arg == AIEDNN_ARG_DST_ITER_C ||
           (arg >= AIEDNN_ARG_DIFF_SRC && arg <= AIEDNN_ARG_DIFF_SHIFT) ||
           (arg >= AIEDNN_ARG_DIFF_SRC_ITER && arg <= AIEDNN_ARG_DIFF_WEIGHTS_ITER) || arg == AIEDNN_ARG_DIFF_KEY ||
           arg == AIEDNN_ARG_DIFF_VALUE;
}

/* [N][C][HW] <-> [N][HW][C], esz-byte elements; 8x8 byte blocks for 8-bit data */
static void rt_nchw_nhwc(const void *src, void *dst, int64_t N, int64_t C, int64_t HW, int64_t esz, int to_nhwc) {
    for (int64_t n = 0; n < N; n++) {
        const char *s = (const char *)src + n * C * HW * esz;
        char *d = (char *)dst + n * C * HW * esz;
        /* to nhwc: rows C, cols HW -> rows HW, cols C; back: the other way round */
        const int64_t R = to_nhwc ? C : HW, Q = to_nhwc ? HW : C;
        for (int64_t r0 = 0; r0 < R; r0 += 64)
            for (int64_t q0 = 0; q0 < Q; q0 += 64) {
                const int64_t r1 = r0 + 64 < R ? r0 + 64 : R, q1 = q0 + 64 < Q ? q0 + 64 : Q;
                if (esz == 1) {
                    int64_t r = r0;
                    for (; r + 8 <= r1; r += 8) {
                        int64_t q = q0;
                        for (; q + 8 <= q1; q += 8)
                            rt_tr8x8((const int8_t *)s + r * Q + q, Q, (int8_t *)d + q * R + r, R);
                        for (; q < q1; q++)
                            for (int64_t i = 0; i < 8; i++)
                                d[q * R + r + i] = s[(r + i) * Q + q];
                    }
                    for (; r < r1; r++)
                        for (int64_t q = q0; q < q1; q++)
                            d[q * R + r] = s[r * Q + q];
                    continue;
                }
                for (int64_t r = r0; r < r1; r++)
                    for (int64_t q = q0; q < q1; q++)
                        memcpy(d + (q * R + r) * esz, s + (r * Q + q) * esz, (size_t)esz);
            }
    }
}

static void rt_fmt_convert(const aiednn_memory_desc_t *m, const void *src, void *dst, int to_nhwc) {
    rt_nchw_nhwc(src, dst, m->dims[0], m->dims[1], m->dims[2] * m->dims[3], dt_bytes(m->data_type), to_nhwc);
}

/* wrap: the slots to convert (formats of the problem as created; prim->prob is the stripped one) */
static void rt_fmt_wrap_plan(aiednn_primitive *prim, const rt_prob *orig) {
    prim->fmt_mask = 0;
    for (int s = 0; s < orig->nmd; s++)
        if (md_is_nhwc(&orig->md[s]) && !md_fmt_free(&orig->md[s]))
            prim->fmt_mask |= 1u << s;
}

/* before the run: nhwc inputs -> dense scratch; outputs get dense scratch. ptr[] is rewritten, user[] keeps
 * the caller's buffers for rt_fmt_out. */
static int rt_fmt_in(aiednn_primitive *prim, void **ptr, void **user) {
    const rt_prob *p = &prim->prob;
    const uint64_t t0 = rt_ticks();
    for (int s = 0; s < p->nmd; s++) {
        user[s] = ptr[s];
        if (!(prim->fmt_mask >> s & 1))
            continue;
        const aiednn_memory_desc_t *m = &p->md[s];
        void *buf = scratch(&prim->fmt_buf[s], &prim->fmt_cap[s], (size_t)dt_size(m->data_type, md_numel(m)));
        if (!buf)
            return AIEK_ERR_ALLOC;
        if (!rt_arg_written(p->arg[s]))
            rt_fmt_convert(m, ptr[s], buf, 0);
        ptr[s] = buf;
    }
    g_phase[RT_PH_FMT] += rt_ticks() - t0;
    return AIEK_OK;
}

static void rt_fmt_out(aiednn_primitive *prim, void *const *ptr, void *const *user) {
    const rt_prob *p = &prim->prob;
    const uint64_t t0 = rt_ticks();
    for (int s = 0; s < p->nmd; s++)
        if ((prim->fmt_mask >> s & 1) && rt_arg_written(p->arg[s]))
            rt_fmt_convert(&p->md[s], ptr[s], user[s], 1);
    g_phase[RT_PH_FMT] += rt_ticks() - t0;
}

/* an nhwc-capable impl on a branch without an nhwc path: run itself on dense copies of the nhwc operands */
static int rt_fmt_rerun_dense(const rt_impl *self, aiednn_primitive *prim, void *const *ptr0,
                              int (*run)(const rt_impl *, aiednn_primitive *, void *const *)) {
    const rt_prob keep = prim->prob;
    void *ptr[RT_MAX_MD], *user[RT_MAX_MD];
    memcpy(ptr, ptr0, sizeof(void *) * (size_t)keep.nmd);
    rt_fmt_wrap_plan(prim, &keep);
    rt_fmt_strip(&prim->prob);
    int rc = rt_fmt_in(prim, ptr, user);
    if (rc == AIEK_OK)
        rc = run(self, prim, ptr);
    if (rc == AIEK_OK)
        rt_fmt_out(prim, ptr, user);
    prim->prob = keep;
    prim->fmt_mask = 0;
    return rc;
}

// ── nhwc:pooling -- 2-D pooling on nhwc src / dst, every channel of a pixel in one contiguous pass ─────────

static int acc_nhwc_pool(const rt_impl *self, const rt_prob *p, rt_bind *bind, char *why, int cap) {
    (void)self, (void)bind;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    if (!md_is_nhwc(s) || !md_is_nhwc(d) || p->prop != RT_FWD_INF || p->has_attr || p->npo) {
        snprintf(why, cap, "nhwc src and dst, inference, no attr");
        return 0;
    }
    /* 8-bit sources: every dense path (poolblk:, ref:) is exact, so is this. A float window sum differs in the
     * last ulp between fpool:'s float and ref:'s double: floats keep the dense AIE path (wrap). */
    if (s->data_type != aiednn_s8 && s->data_type != aiednn_u8) {
        snprintf(why, cap, "s8 / u8 source (f32: fpool: through the wrap)");
        return 0;
    }
    return 1;
}

/* the window's taps in ref:pooling's order (kh, kw), each a contiguous run of C values; max of 8-bit codes as
 * integers, everything else in double as ref: does -- bit-identical to it */
static int run_nhwc_pool(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob;
    const aiednn_memory_desc_t *s = &p->md[0], *d = &p->md[1];
    const aiednn_pool_params_t *pp = &p->pool;
    const int64_t N = s->dims[0], C = s->dims[1], H = s->dims[2], W = s->dims[3], Ho = d->dims[2], Wo = d->dims[3];
    const int mx = p->alg == aiednn_pooling_max, sdt = s->data_type;
    const int i8max = mx && (sdt == aiednn_s8 || sdt == aiednn_u8) && sdt == d->data_type;
    static double *acc;
    static int32_t *iacc;
    static size_t acc_cap, iacc_cap;
    double *a = (double *)scratch((void **)&acc, &acc_cap, (size_t)C * sizeof(double));
    int32_t *ia = (int32_t *)scratch((void **)&iacc, &iacc_cap, (size_t)C * sizeof(int32_t));
    if (!a || !ia)
        return AIEK_ERR_ALLOC;
    for (int64_t n = 0; n < N; n++)
        for (int64_t oh = 0; oh < Ho; oh++)
            for (int64_t ow = 0; ow < Wo; ow++) {
                int64_t cnt = 0;
                const int64_t o = ((n * Ho + oh) * Wo + ow) * C;
                for (int64_t c = 0; c < C; c++)
                    a[c] = mx ? -INFINITY : 0.0, ia[c] = sdt == aiednn_u8 ? 0 : -128;
                for (int64_t kh = 0; kh < pp->kernel[0]; kh++)
                    for (int64_t kw = 0; kw < pp->kernel[1]; kw++) {
                        const int64_t ih = oh * pp->strides[0] - pp->padding_l[0] + kh * (pp->dilates[0] + 1);
                        const int64_t iw = ow * pp->strides[1] - pp->padding_l[1] + kw * (pp->dilates[1] + 1);
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                            continue;
                        cnt++;
                        const int64_t i = ((n * H + ih) * W + iw) * C;
                        if (i8max && sdt == aiednn_s8) {
                            const int8_t *x = (const int8_t *)ptr[0] + i;
                            for (int64_t c = 0; c < C; c++)
                                ia[c] = x[c] > ia[c] ? x[c] : ia[c];
                        } else if (i8max) {
                            const uint8_t *x = (const uint8_t *)ptr[0] + i;
                            for (int64_t c = 0; c < C; c++)
                                ia[c] = x[c] > ia[c] ? x[c] : ia[c];
                        } else {
                            for (int64_t c = 0; c < C; c++) {
                                const double x = rt_ld(s->data_type, ptr[0], i + c);
                                a[c] = mx ? (x > a[c] ? x : a[c]) : a[c] + x;
                            }
                        }
                    }
                if (i8max && cnt) { /* a window with no valid tap: ref: stores -inf, as the generic branch */
                    if (sdt == aiednn_s8)
                        for (int64_t c = 0; c < C; c++)
                            ((int8_t *)ptr[1])[o + c] = (int8_t)ia[c];
                    else
                        for (int64_t c = 0; c < C; c++)
                            ((uint8_t *)ptr[1])[o + c] = (uint8_t)ia[c];
                    continue;
                }
                for (int64_t c = 0; c < C; c++) {
                    double v = a[c];
                    if (p->alg == aiednn_pooling_avg_include_padding)
                        v /= (double)(pp->kernel[0] * pp->kernel[1]);
                    else if (p->alg == aiednn_pooling_avg_exclude_padding)
                        v = cnt ? v / (double)cnt : 0.0;
                    rt_st(d->data_type, ptr[1], o + c, v);
                }
            }
    return AIEK_OK;
}

// ── API ───────────────────────────────────────────────────────────────────

extern "C" aiednn_status_t aiednn_memory_desc_set_format(aiednn_memory_desc_t *md, aiednn_format_t format) {
    if (!md || (format != aiednn_format_dense && format != aiednn_format_nhwc) ||
        (format == aiednn_format_nhwc && md->ndims != 4))
        return aiednn_invalid_arguments;
    md->format = format;
    return aiednn_success;
}

extern "C" size_t aiednn_memory_desc_get_size(const aiednn_memory_desc_t *md) {
    if (!md || md->ndims < 1 || md->ndims > AIEDNN_MAX_NDIMS)
        return 0;
    return (size_t)dt_size(md->data_type, md_numel(md));
}
