/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_custom: a user operator with no aiednn primitive, run on the host between the primitives around it
 * (a framework's opaque custom op lowered by the model compiler). The library only carries the call: the
 * user's function sees the sources' / dst's descriptors and buffers, in the order they were created.
 */

static int run_ref_custom(const rt_impl *self, aiednn_primitive *prim, void *const *ptr) {
    (void)self;
    const rt_prob *p = &prim->prob; /* md[0..nsrc-1] = MULTIPLE_SRC + i, md[nsrc] = DST */
    const void *src[AIEDNN_MAX_SRCS];
    for (int i = 0; i < p->nsrc; i++)
        src[i] = ptr[i];
    return p->custom_fn(p->nsrc, p->md, src, &p->md[p->nsrc], ptr[p->nsrc], p->custom_ctx) == 0 ? 0 : -1;
}

extern "C" aiednn_status_t aiednn_custom_create(aiednn_primitive_t *pr, int nsrc, const aiednn_memory_desc_t *srcs,
                                                const aiednn_memory_desc_t *d, aiednn_custom_fn_t fn, void *ctx) {
    const char *k = "custom";
    REQUIRE(pr && fn && md_ok(d) && nsrc >= 1 && nsrc <= AIEDNN_MAX_SRCS && srcs, k,
            "a function, 1..AIEDNN_MAX_SRCS sources and a dst");
    rt_prob p;
    prob_init(&p, aiednn_custom, 0);
    for (int i = 0; i < nsrc; i++) {
        REQUIRE(md_ok(&srcs[i]), k, "every source needs a valid descriptor");
        prob_add(&p, AIEDNN_ARG_MULTIPLE_SRC + i, &srcs[i]);
    }
    prob_add(&p, AIEDNN_ARG_DST, d);
    p.nsrc = nsrc;
    p.custom_fn = fn;
    p.custom_ctx = ctx;
    return create(pr, &p);
}
