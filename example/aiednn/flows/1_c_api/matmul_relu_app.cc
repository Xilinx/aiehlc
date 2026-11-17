/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Flow 1 -- the C primitive API (include/aiednn.h): two primitives, an int8 matmul and a relu.
 *
 * The app only describes tensors and asks for primitives. Dispatch picks each implementation from
 * what the build linked: the AIE GEMM for the 256x256x256 s8 matmul, the AIE table mode (lut:) for
 * the s8 relu, ref: (the CPU) for anything no AIE kernel accepts. The results are checked against
 * ref: through the same API.
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/flows/1_c_api/matmul_relu_app.cc --target x86 --run
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/flows/1_c_api/matmul_relu_app.cc \
 *       --target vek280 --run --board <host> --vitis-settings <.../Vitis/settings64.sh>
 *
 * Prints "PASS: ..." / "FAIL: ...".
 ******************************************************************************/
#include "aiednn.h"

#include <stdio.h>
#include <string.h>

#define M 256
#define K 256
#define N 256

static int8_t g_x[M * K], g_w[N * K], g_y[M * N], g_z[M * N], g_yref[M * N], g_zref[M * N];

static int check(aiednn_status_t st, const char *what) {
    if (st == aiednn_success)
        return 0;
    printf("FAIL: %s: %s\n", what, aiednn_status_str(st));
    return 1;
}

/* y = x @ w^T (s8), z = relu(y); impl names into *mm_impl / *relu_impl */
static int run(int8_t *y, int8_t *z, char *mm_impl, char *relu_impl, size_t cap) {
    const int64_t xd[2] = {M, K}, wd[2] = {N, K}, yd[2] = {M, N};
    aiednn_memory_desc_t xm, wm, ym;
    if (check(aiednn_memory_desc_init(&xm, 2, xd, aiednn_s8), "x md") ||
        check(aiednn_memory_desc_init(&wm, 2, wd, aiednn_s8), "w md") ||
        check(aiednn_memory_desc_init(&ym, 2, yd, aiednn_s8), "y md"))
        return 1;
    aiednn_primitive_t mm, relu;
    if (check(aiednn_matmul_primitive_create(&mm, &xm, &wm, &ym), "create matmul") ||
        check(aiednn_eltwise_forward_create(&relu, aiednn_eltwise_relu, 0.f, 0.f, &ym, &ym), "create relu"))
        return 1;
    snprintf(mm_impl, cap, "%s", aiednn_primitive_impl_info(mm));
    snprintf(relu_impl, cap, "%s", aiednn_primitive_impl_info(relu));
    const aiednn_exec_arg_t ra[2] = {{AIEDNN_ARG_SRC, y}, {AIEDNN_ARG_DST, z}};
    if (check(aiednn_primitive_execute(mm, g_x, g_w, y), "execute matmul") ||
        check(aiednn_primitive_execute_args(relu, 2, ra), "execute relu"))
        return 1;
    aiednn_primitive_destroy(mm);
    aiednn_primitive_destroy(relu);
    return 0;
}

int main() {
    printf("=== flow 1: aiednn C API, s8 matmul %dx%dx%d + relu ===\n", M, N, K);
    for (int i = 0; i < M * K; i++)
        g_x[i] = (int8_t)((i * 7) % 13 - 6);
    for (int i = 0; i < N * K; i++)
        g_w[i] = (int8_t)((i * 5) % 11 - 5);
    char mm[96], rl[96], mm_ref[96], rl_ref[96];
    if (run(g_y, g_z, mm, rl, sizeof mm))
        return 1;
    printf("matmul -> %s\nrelu   -> %s\n", mm, rl);
    aiednn_set_impl_filter("ref:"); /* the same calls on the CPU reference */
    if (run(g_yref, g_zref, mm_ref, rl_ref, sizeof mm_ref))
        return 1;
    aiednn_set_impl_filter(nullptr);
    if (memcmp(g_y, g_yref, sizeof g_y) || memcmp(g_z, g_zref, sizeof g_z)) {
        printf("FAIL: flow 1 differs from ref:\n");
        return 1;
    }
    printf("PASS: flow 1 C API, matmul on %s, relu on %s, bit-exact vs ref:\n", mm, rl);
    return 0;
}
