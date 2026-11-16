/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn_rt_itemsn.h -- many launches' worth of one-item-per-core work in one AIE launch.
 *
 * rt_items_n(): the caller packs item i (pack) and consumes its result (put), double-buffered against
 * the n-tile launch. Returns RT_NOT_TAKEN when there is no n-tile entry, AIEDNN_NO_NTILE /
 * AIEDNN_NO_ITEMSN is set, or the work fits one launch.
 *
 * Needs, from the including TU: aiednn_rt_internal.h, aiednn_rt_gemm.h, aiednn_rt_ntile.h.
 ******************************************************************************/
#ifndef AIEDNN_RT_ITEMSN_H
#define AIEDNN_RT_ITEMSN_H

/* n tiles of `cores` items under one descriptor (opcode, desc[0..desc_n)); finish waits and writes
 * int32 [n * cores][1024] in item order */
extern "C" int aiednn_gemm_items_n_start(const int8_t *items, int n, int opcode, const uint8_t *desc, int desc_n)
    __attribute__((weak));
extern "C" int aiednn_gemm_items_n_finish(int32_t *out) __attribute__((weak));

#define RT_ITEM_BYTES 4096
#define RT_ITEM_OUT 1024

typedef void (*rt_item_pack_fn)(void *ctx, int64_t item, uint8_t *raw);      /* raw: 4 KB, not cleared */
typedef void (*rt_item_put_fn)(void *ctx, int64_t item, const int32_t *res); /* res: int32 [1024] */

static void *g_itn_items[2], *g_itn_out;
static size_t g_itn_items_cap[2], g_itn_out_cap;

static int rt_items_n(const rt_impl *self, int64_t total, int opcode, const uint8_t *desc, int desc_n,
                      rt_item_pack_fn pack, rt_item_put_fn put, void *ctx) {
    if (!aiednn_gemm_items_n_start || !aiednn_gemm_items_n_finish || !aiednn_gemm_ntile_max ||
        rt_env_on("AIEDNN_NO_NTILE") || rt_env_on("AIEDNN_NO_ITEMSN"))
        return RT_NOT_TAKEN;
    const int64_t NI = self->gen->args[0].dims[0], tiles_all = (total + NI - 1) / NI, tmax = aiednn_gemm_ntile_max();
    if (tiles_all < 2 || tmax < 2)
        return RT_NOT_TAKEN;
    const int64_t T = tiles_all < tmax ? tiles_all : tmax, chunk = T * NI; // items per launch
    const int64_t nchunks = (total + chunk - 1) / chunk;
    uint8_t *buf[2];
    for (int i = 0; i < 2; i++)
        buf[i] = (uint8_t *)scratch(&g_itn_items[i], &g_itn_items_cap[i], chunk * RT_ITEM_BYTES);
    int32_t *out = (int32_t *)scratch(&g_itn_out, &g_itn_out_cap, chunk * RT_ITEM_OUT * sizeof(int32_t));
    if (!buf[0] || !buf[1] || !out)
        return AIEK_ERR_ALLOC;
    self->gen->entry(nullptr, 0); /* counts the entry (modes="auto"), loads a megakernel with its mode (library) */
    auto count_of = [&](int64_t j) { return total - j * chunk < chunk ? total - j * chunk : chunk; };
    auto tiles_of = [&](int64_t j) { return (int)((count_of(j) + NI - 1) / NI); };
    auto fill = [&](int64_t j, uint8_t *b) {
        const uint64_t t0 = rt_ticks();
        for (int64_t k = 0; k < count_of(j); k++)
            pack(ctx, j * chunk + k, b + k * RT_ITEM_BYTES);
        memset(b + count_of(j) * RT_ITEM_BYTES, 0, (size_t)((tiles_of(j) * NI - count_of(j)) * RT_ITEM_BYTES));
        g_phase[RT_PH_DWPACK] += rt_ticks() - t0;
    };
    fill(0, buf[0]);
    uint64_t te = rt_ticks();
    int rc = aiednn_gemm_items_n_start((const int8_t *)buf[0], tiles_of(0), opcode, desc, desc_n);
    for (int64_t j = 0; rc == AIEK_OK && j < nchunks; j++) {
        ((rt_impl *)self)->launches++;
        if (j + 1 < nchunks) // next chunk's items while this one runs
            fill(j + 1, buf[(j + 1) & 1]);
        rc = aiednn_gemm_items_n_finish(out);
        if (rc == AIEK_OK && j + 1 < nchunks)
            rc = aiednn_gemm_items_n_start((const int8_t *)buf[(j + 1) & 1], tiles_of(j + 1), opcode, desc, desc_n);
        const uint64_t t0 = rt_ticks();
        ((rt_impl *)self)->t_entry += t0 - te;    // the launch, overlapped with the next chunk's packing
        for (int64_t k = 0; k < count_of(j); k++) // this chunk's results while the next one runs
            put(ctx, j * chunk + k, out + k * RT_ITEM_OUT);
        te = rt_ticks();
        g_phase[RT_PH_CUNPACK] += te - t0;
    }
    return rc;
}

#endif /* AIEDNN_RT_ITEMSN_H */
