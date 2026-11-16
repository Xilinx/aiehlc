/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_dwn.h -- depthwise + requant with many launches' worth of items in one AIE launch.
 *
 * With the persistent build's n-tile launch (aiednn_gemm_dwq_n_start / _finish) one launch carries up to
 * aiednn_gemm_ntile_max() dwq tiles: tile t's A is its first `cores` items, its B block the other
 * 2 * cores and the descriptor. Bit for bit run_dwblk's result. Host packing / copy-out overlap the launch.
 *
 * Not taken: no n-tile entry, AIEDNN_NO_NTILE / AIEDNN_NO_DWN set, or a layer that fits one launch.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_gemm.h and aiednn_rt_ntile.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_DWN_H
#define AIEDNN_RT_DWN_H

/* n tiles of dwq_blk_s8s8 items (3 * cores each) in one launch; finish waits and writes the int8
 * bands [n * 3 * cores][RT_DW_OUT] in item order */
extern "C" int aiednn_gemm_dwq_n_start(const int8_t *items, int n, const int32_t *geom) __attribute__((weak));
extern "C" int aiednn_gemm_dwq_n_finish(int8_t *out) __attribute__((weak));

static void *g_dwn_items[2], *g_dwn_out;
static size_t g_dwn_items_cap[2], g_dwn_out_cap;

/* items [first, first + count) of the layer into buf (count <= tiles * NI; the rest zero) */
static void dwn_pack(const conv_geo *c, const dw_plan *d, const rt_epi *E, const int8_t *x, const int8_t *w,
                     int64_t first, int64_t count, int64_t span, int8_t *buf) {
    const uint64_t t0 = rt_ticks();
    memset(buf, 0, (size_t)(span * RT_DW_BYTES));
    for (int64_t k = 0; k < count; k++)
        dw_pack_item(c, d, E, 1, x, w, first + k, buf + k * RT_DW_BYTES);
    g_phase[RT_PH_DWPACK] += rt_ticks() - t0;
}

static int run_dwq_n(const rt_impl *self, const rt_epi *E, const conv_geo *c, const dw_plan *d, const int8_t *x,
                     const int8_t *w, void *dst) {
    if (!aiednn_gemm_dwq_n_start || !aiednn_gemm_dwq_n_finish || !aiednn_gemm_ntile_max || !g_dwqblk ||
        rt_env_on("AIEDNN_NO_NTILE") || rt_env_on("AIEDNN_NO_DWN"))
        return RT_NOT_TAKEN;
    const int64_t NI = g_dwqblk->args[0].dims[0], total = c->N * c->C * d->bands;
    const int64_t tiles_all = (total + NI - 1) / NI, tmax = aiednn_gemm_ntile_max();
    if (tiles_all < 2 || tmax < 2)
        return RT_NOT_TAKEN;
    const int64_t T = tiles_all < tmax ? tiles_all : tmax, chunk = T * NI; // items per launch
    const int64_t nchunks = (total + chunk - 1) / chunk;
    int8_t *buf[2];
    for (int i = 0; i < 2; i++)
        buf[i] = (int8_t *)scratch(&g_dwn_items[i], &g_dwn_items_cap[i], chunk * RT_DW_BYTES);
    int8_t *out = (int8_t *)scratch(&g_dwn_out, &g_dwn_out_cap, chunk * RT_DW_OUT);
    if (!buf[0] || !buf[1] || !out)
        return AIEK_ERR_ALLOC;
    g_dwqblk->entry(nullptr, 0); /* counts the entry (modes="auto") and loads a megakernel with DWQ (library) */
    const int32_t geom[8] = {
        (int32_t)c->s[0], (int32_t)c->KH, (int32_t)c->KW, (int32_t)d->wp, (int32_t)d->rout, (int32_t)c->Wo, 1, 0};
    auto count_of = [&](int64_t j) { return total - j * chunk < chunk ? total - j * chunk : chunk; };
    auto tiles_of = [&](int64_t j) { return (int)((count_of(j) + NI - 1) / NI); };
    dwn_pack(c, d, E, x, w, 0, count_of(0), tiles_of(0) * NI, buf[0]);
    uint64_t te = rt_ticks();
    int rc = aiednn_gemm_dwq_n_start(buf[0], tiles_of(0), geom);
    for (int64_t j = 0; rc == AIEK_OK && j < nchunks; j++) {
        ((rt_impl *)self)->launches++;
        if (j + 1 < nchunks) // next chunk's items while this one runs
            dwn_pack(c, d, E, x, w, (j + 1) * chunk, count_of(j + 1), tiles_of(j + 1) * NI, buf[(j + 1) & 1]);
        rc = aiednn_gemm_dwq_n_finish(out);
        if (rc == AIEK_OK && j + 1 < nchunks)
            rc = aiednn_gemm_dwq_n_start(buf[(j + 1) & 1], tiles_of(j + 1), geom);
        const uint64_t t0 = rt_ticks();
        ((rt_impl *)self)->t_entry += t0 - te;    // the launch, overlapped with the next chunk's packing
        for (int64_t k = 0; k < count_of(j); k++) // this chunk's bands while the next one runs
            dw_put_item(c, d, E, 1, j * chunk + k, out + k * RT_DW_OUT, dst);
        te = rt_ticks();
        g_phase[RT_PH_CUNPACK] += te - t0;
    }
    return rc;
}

#endif /* AIEDNN_RT_DWN_H */
