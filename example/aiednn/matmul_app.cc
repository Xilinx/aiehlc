/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * matmul_app -- the smallest aiednn application: one int8 matmul.
 * Dispatch picks the implementation; the result is checked against ref:.
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/matmul_app.cc \
 *       --target vek280 --run --board <host> --vitis-settings <.../Vitis/settings64.sh>
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/matmul_app.cc --target x86 --run
 ******************************************************************************/
#include "aiednn.h"

#include <stdio.h>
#include <string.h>

#define MM_M 256
#define MM_K 256
#define MM_N 256
#define MM_RUNS 2

static int8_t g_src[MM_M * MM_K];
static int8_t g_wei[MM_N * MM_K];
static int8_t g_dst[MM_RUNS][MM_M * MM_N];
static int8_t g_ref[MM_M * MM_N];

static int check(aiednn_status_t st, const char *what) {
    if (st == aiednn_success)
        return 0;
    printf("FAIL: %s: %s\n", what, aiednn_status_str(st));
    return 1;
}

int main() {
    printf("=== aiednn matmul %dx%dx%d s8 ===\n", MM_M, MM_N, MM_K);
    aiednn_set_verbose(2);

    const int64_t src_dims[2] = {MM_M, MM_K}, wei_dims[2] = {MM_N, MM_K}, dst_dims[2] = {MM_M, MM_N};
    aiednn_memory_desc_t src_md, wei_md, dst_md;
    if (check(aiednn_memory_desc_init(&src_md, 2, src_dims, aiednn_s8), "src md") ||
        check(aiednn_memory_desc_init(&wei_md, 2, wei_dims, aiednn_s8), "weights md") ||
        check(aiednn_memory_desc_init(&dst_md, 2, dst_dims, aiednn_s8), "dst md"))
        return 1;

    for (int i = 0; i < MM_M * MM_K; i++)
        g_src[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < MM_N * MM_K; i++)
        g_wei[i] = (int8_t)((i % 5) - 2);

    aiednn_primitive_t mm;
    if (check(aiednn_matmul_primitive_create(&mm, &src_md, &wei_md, &dst_md), "create matmul"))
        return 1;
    char impl[96]; // impl_info lives as long as the primitive; keep a copy
    snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(mm));
    printf("aiednn picked %s\n", impl);
    for (int r = 0; r < MM_RUNS; r++) {
        memset(g_dst[r], 0x55, sizeof g_dst[r]);
        if (check(aiednn_primitive_execute(mm, g_src, g_wei, g_dst[r]), "execute"))
            return 1;
    }
    aiednn_primitive_destroy(mm);

    aiednn_set_verbose(1);
    aiednn_set_impl_filter("ref:");
    aiednn_primitive_t ref;
    if (check(aiednn_matmul_primitive_create(&ref, &src_md, &wei_md, &dst_md), "create ref matmul") ||
        check(aiednn_primitive_execute(ref, g_src, g_wei, g_ref), "execute ref"))
        return 1;
    aiednn_primitive_destroy(ref);
    aiednn_set_impl_filter(nullptr);

    int bad = 0;
    for (int r = 0; r < MM_RUNS; r++)
        for (int i = 0; i < MM_M * MM_N; i++) {
            if (g_dst[r][i] == g_ref[i])
                continue;
            if (bad < 8)
                printf("MISMATCH run %d C[%d][%d] = %d, ref %d\n", r + 1, i / MM_N, i % MM_N, (int)g_dst[r][i],
                       (int)g_ref[i]);
            bad++;
        }
    if (bad) {
        printf("FAIL: aiednn matmul on %s: %d of %d outputs differ from ref:\n", impl, bad, MM_RUNS * MM_M * MM_N);
        return 1;
    }
    printf("PASS: aiednn matmul %dx%dx%d on %s, %d runs bit-exact vs ref:matmul\n", MM_M, MM_N, MM_K, impl, MM_RUNS);
    return 0;
}
