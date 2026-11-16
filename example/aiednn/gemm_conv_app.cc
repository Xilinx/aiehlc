/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * gemm_conv_app -- aiednn's int32 paths on the board, against ref:.
 *
 * matmul 300x200x700 (blocked:), conv 3x3 (im2col+), conv 1x1 s2; each vs ref:.
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/gemm_conv_app.cc --target x86 --run
 ******************************************************************************/
#include "aiednn.h"

#include <stdio.h>
#include <string.h>

static int8_t g_a[300 * 700], g_b[200 * 700], g_x[64 * 30 * 30], g_w[96 * 64 * 9], g_w1[128 * 64];
static int32_t g_out[96 * 28 * 28], g_ref[96 * 28 * 28];

static int run_case(const char *name, int conv, const int64_t *ds, const int64_t *dw, const int64_t *dd, int nd,
                    int stride, const void *src, const void *wei, long nout) {
    aiednn_memory_desc_t s, w, d;
    aiednn_memory_desc_init(&s, nd, ds, aiednn_s8);
    aiednn_memory_desc_init(&w, nd, dw, aiednn_s8);
    aiednn_memory_desc_init(&d, nd, dd, aiednn_s32);
    char impl[96] = "";
    for (int pass = 0; pass < 2; pass++) {
        aiednn_set_impl_filter(pass ? "ref:" : nullptr);
        aiednn_primitive_t p;
        aiednn_status_t st = conv ? aiednn_convolution_primitive_create(&p, &s, &w, &d, stride)
                                  : aiednn_matmul_primitive_create(&p, &s, &w, &d);
        if (st == aiednn_success)
            st = aiednn_primitive_execute(p, src, wei, pass ? (void *)g_ref : (void *)g_out);
        if (st != aiednn_success) {
            printf("FAIL: %s pass %d: %s\n", name, pass, aiednn_status_str(st));
            return 1;
        }
        if (!pass)
            snprintf(impl, sizeof impl, "%s", aiednn_primitive_impl_info(p));
        aiednn_primitive_destroy(p);
    }
    aiednn_set_impl_filter(nullptr);
    long bad = 0;
    for (long i = 0; i < nout; i++)
        if (g_out[i] != g_ref[i] && bad++ < 4)
            printf("MISMATCH %s [%ld] = %d, ref %d\n", name, i, (int)g_out[i], (int)g_ref[i]);
    printf("%s %s on %s: %ld of %ld differ from ref:\n", bad ? "BAD " : "ok  ", name, impl, bad, nout);
    if (strncmp(impl, "ref:", 4) == 0) {
        printf("%s ran on ref:, not on the AIE\n", name);
        return 1;
    }
    return bad != 0;
}

int main() {
    printf("=== aiednn gemm/conv self-test (int32 paths) ===\n");
    aiednn_set_verbose(1);
    for (long i = 0; i < 300 * 700; i++)
        g_a[i] = (int8_t)((i * 7 + 3) % 255 - 127);
    for (long i = 0; i < 200 * 700; i++)
        g_b[i] = (int8_t)((i * 13 + 5) % 251 - 125);
    for (long i = 0; i < 64 * 30 * 30; i++)
        g_x[i] = (int8_t)((i * 11 + 1) % 256 - 128);
    for (long i = 0; i < 96 * 64 * 9; i++)
        g_w[i] = (int8_t)((i * 5 + 2) % 255 - 127);
    for (long i = 0; i < 128 * 64; i++)
        g_w1[i] = (int8_t)((i * 3 + 9) % 253 - 126);

    int fail = 0;
    const int64_t ms[2] = {300, 700}, mw[2] = {200, 700}, md[2] = {300, 200};
    fail |= run_case("matmul 300x200x700", 0, ms, mw, md, 2, 1, g_a, g_b, 300L * 200);
    const int64_t cs[4] = {1, 64, 30, 30}, cw[4] = {96, 64, 3, 3}, cd[4] = {1, 96, 28, 28};
    fail |= run_case("conv 3x3 s1 64->96 28px", 1, cs, cw, cd, 4, 1, g_x, g_w, 96L * 28 * 28);
    const int64_t c1s[4] = {1, 64, 28, 28}, c1w[4] = {128, 64, 1, 1}, c1d[4] = {1, 128, 14, 14};
    fail |= run_case("conv 1x1 s2 64->128 14px", 1, c1s, c1w, c1d, 4, 2, g_x, g_w1, 128L * 14 * 14);
    aiednn_print_stats();
    printf(fail ? "FAIL: aiednn int32 self-test\n" : "PASS: aiednn int32 self-test, all cases bit-exact vs ref:\n");
    return fail;
}
