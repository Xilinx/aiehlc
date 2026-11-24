/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* aiednn_stream_test -- counts heap allocations (interposed malloc / calloc / realloc) during an execute
 * with a caller workspace (the contract: none). */
#include "aiednn.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void *__libc_malloc(size_t);
extern "C" void *__libc_calloc(size_t, size_t);
extern "C" void *__libc_realloc(void *, size_t);
static volatile int g_counting;
static volatile long g_allocs;
extern "C" void *malloc(size_t n) {
    if (g_counting)
        g_allocs++;
    return __libc_malloc(n);
}
extern "C" void *calloc(size_t n, size_t s) {
    if (g_counting)
        g_allocs++;
    return __libc_calloc(n, s);
}
extern "C" void *realloc(void *p, size_t n) {
    if (g_counting)
        g_allocs++;
    return __libc_realloc(p, n);
}

static int g_fail;
#define CHECK(c, ...)                                                                                                  \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                                                                \
            printf(__VA_ARGS__);                                                                                       \
            printf("\n");                                                                                              \
            g_fail = 1;                                                                                                \
        }                                                                                                              \
    } while (0)

static aiednn_memory_desc_t D(aiednn_data_type_t dt, int nd, int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1,
                              int64_t e = 1) {
    aiednn_memory_desc_t m;
    const int64_t dims[5] = {a, b, c, d, e};
    aiednn_memory_desc_init(&m, nd, dims, dt);
    return m;
}
static size_t bytes(const aiednn_memory_desc_t *m) {
    size_t n = m->data_type == aiednn_f32 || m->data_type == aiednn_s32 ? 4 : 1;
    for (int i = 0; i < m->ndims; i++)
        n *= (size_t)m->dims[i];
    return n;
}
static void *filled(const aiednn_memory_desc_t *m, unsigned seed) {
    const size_t n = bytes(m);
    unsigned char *p = (unsigned char *)aligned_alloc(64, (n + 63) / 64 * 64);
    for (size_t i = 0; i < n; i++)
        p[i] = (unsigned char)((i * 2654435761u + seed) >> 13);
    if (m->data_type == aiednn_f32)
        for (size_t i = 0; i < n / 4; i++)
            ((float *)p)[i] = (float)((int)((i * 37 + seed) % 201) - 100) / 64.0f;
    return p;
}

/* execute_async with a caller workspace vs execute_args; output at index o of size ob */
static void ws_case(const char *name, aiednn_primitive_t p, int nargs, aiednn_exec_arg_t *a, int o, size_t ob) {
    const size_t ws = aiednn_primitive_get_workspace_size(p);
    void *work = ws ? aligned_alloc(64, (ws + 63) / 64 * 64) : nullptr;
    void *ref = malloc(ob), *keep = a[o].data;
    CHECK(aiednn_primitive_execute_args(p, nargs, a) == aiednn_success, "%s execute_args", name);
    memcpy(ref, keep, ob);
    memset(keep, 0x5a, ob);
    aiednn_stream_t s;
    CHECK(aiednn_stream_create(&s, 0) == aiednn_success, "stream");
    CHECK(aiednn_primitive_execute_async(p, s, nargs, a, work, ws) == aiednn_success, "%s warm-up", name);
    memset(keep, 0x5a, ob);
    g_allocs = 0, g_counting = 1;
    const aiednn_status_t st = aiednn_primitive_execute_async(p, s, nargs, a, work, ws);
    g_counting = 0;
    CHECK(st == aiednn_success && aiednn_stream_wait(s) == aiednn_success, "%s execute_async", name);
    CHECK(g_allocs == 0, "%s: %ld heap allocations in an execute with a caller workspace", name, g_allocs);
    CHECK(!memcmp(ref, keep, ob), "%s: workspace result differs from execute_args", name);
    if (ws > 64) { /* too small a workspace is an error, not a heap fallback */
        CHECK(aiednn_primitive_execute_async(p, s, nargs, a, work, 64) == aiednn_out_of_memory, "%s small ws", name);
        CHECK(aiednn_stream_wait(s) == aiednn_out_of_memory, "%s stream error", name);
    }
    printf("WS %s impl=%s workspace=%zu allocs=%ld\n", name, aiednn_primitive_impl_info(p), ws, g_allocs);
    aiednn_stream_destroy(s);
    free(ref), free(work);
}

int main() {
    { /* s8 conv 2-D: im2col + the AIE GEMM block */
        aiednn_memory_desc_t x = D(aiednn_s8, 4, 1, 16, 14, 14), w = D(aiednn_s8, 4, 32, 16, 3, 3),
                             y = D(aiednn_s32, 4, 1, 32, 12, 12);
        aiednn_conv_params_t cp = {};
        cp.strides[0] = cp.strides[1] = 1, cp.groups = 1;
        aiednn_primitive_t p;
        CHECK(aiednn_convolution_forward_create(&p, &x, &w, nullptr, &y, &cp) == aiednn_success, "conv create");
        aiednn_exec_arg_t a[3] = {
            {AIEDNN_ARG_SRC, filled(&x, 1)}, {AIEDNN_ARG_WEIGHTS, filled(&w, 2)}, {AIEDNN_ARG_DST, filled(&y, 3)}};
        ws_case("conv_s8", p, 3, a, 2, bytes(&y));
        /* find: every acceptor, AIE first; MEASURE times them on library buffers; set_impl pins one */
        aiednn_impl_perf_t perf[8];
        const int nh = aiednn_primitive_find(p, AIEDNN_FIND_HEURISTIC, perf, 8);
        CHECK(nh >= 2 && perf[0].on_aie && !strcmp(perf[nh - 1].impl, "ref:convolution"), "heuristic order (%d)", nh);
        const unsigned char canary = ((unsigned char *)a[2].data)[0];
        const int nm = aiednn_primitive_find(p, AIEDNN_FIND_MEASURE, perf, 8);
        CHECK(nm == nh && ((unsigned char *)a[2].data)[0] == canary, "measure leaves the caller's buffers");
        for (int i = 0; i < nm; i++) {
            CHECK(perf[i].status == aiednn_success && perf[i].time_us > 0, "measured %s", perf[i].impl);
            CHECK(i == 0 || perf[i].time_us >= perf[i - 1].time_us, "fastest first");
            printf("FIND %s on_aie=%d time_us=%.1f workspace=%zu\n", perf[i].impl, perf[i].on_aie, perf[i].time_us,
                   perf[i].workspace_size);
        }
        CHECK(aiednn_primitive_set_impl(p, "ref:convolution") == aiednn_success &&
                  !strcmp(aiednn_primitive_impl_info(p), "ref:convolution"),
              "set_impl");
        CHECK(aiednn_primitive_set_impl(p, "no-such-impl") == aiednn_unimplemented, "set_impl unknown");
        ws_case("conv_s8_ref", p, 3, a, 2, bytes(&y));
        aiednn_primitive_destroy(p);
    }
    { /* s8 3-D conv: the gemm+conv composite */
        aiednn_memory_desc_t x = D(aiednn_s8, 5, 1, 8, 6, 10, 10), w = D(aiednn_s8, 5, 16, 8, 3, 3, 3),
                             y = D(aiednn_s32, 5, 1, 16, 4, 8, 8);
        aiednn_conv_nd_params_t cp = {};
        for (int i = 0; i < 3; i++)
            cp.strides[i] = 1;
        cp.groups = 1;
        aiednn_primitive_t p;
        CHECK(aiednn_convolution_forward_create_nd(&p, &x, &w, nullptr, &y, &cp, nullptr) == aiednn_success,
              "conv3d create");
        aiednn_exec_arg_t a[3] = {
            {AIEDNN_ARG_SRC, filled(&x, 4)}, {AIEDNN_ARG_WEIGHTS, filled(&w, 5)}, {AIEDNN_ARG_DST, filled(&y, 6)}};
        ws_case("conv3d_s8", p, 3, a, 2, bytes(&y));
        g_allocs = 0, g_counting = 1; /* control: without a workspace the composite allocates (the counter sees it) */
        aiednn_primitive_execute_async(p, nullptr, 3, a, nullptr, 0);
        g_counting = 0;
        CHECK(g_allocs > 0, "the malloc counter saw no allocation without a workspace");
        printf("CONTROL conv3d_s8 without a workspace: %ld heap allocations\n", g_allocs);
        aiednn_primitive_destroy(p);
    }
    { /* f32 LSTM: gemm+rnn (the float GEMM per step) */
        const int64_t T = 4, N = 2, H = 16;
        aiednn_memory_desc_t xs = D(aiednn_f32, 3, T, N, H), wl = D(aiednn_f32, 5, 1, 1, H, 4, H),
                             b = D(aiednn_f32, 4, 1, 1, 4, H);
        aiednn_primitive_t p;
        CHECK(aiednn_rnn_forward_create(&p, aiednn_rnn_lstm, aiednn_rnn_left2right, 0.f, &xs, nullptr, nullptr, &wl,
                                        &wl, &b, &xs, nullptr, nullptr) == aiednn_success,
              "rnn create");
        aiednn_exec_arg_t a[5] = {{AIEDNN_ARG_SRC, filled(&xs, 7)},
                                  {AIEDNN_ARG_WEIGHTS, filled(&wl, 8)},
                                  {AIEDNN_ARG_WEIGHTS_ITER, filled(&wl, 9)},
                                  {AIEDNN_ARG_BIAS, filled(&b, 10)},
                                  {AIEDNN_ARG_DST, filled(&xs, 11)}};
        ws_case("lstm_f32", p, 5, a, 4, bytes(&xs));
        aiednn_primitive_destroy(p);
    }
    { /* f32 GELU (FVEC where the library has it) and LRN (ref: only) */
        aiednn_memory_desc_t x = D(aiednn_f32, 4, 1, 8, 16, 16);
        aiednn_primitive_t p;
        CHECK(aiednn_eltwise_forward_create(&p, aiednn_eltwise_gelu_tanh, 0.f, 0.f, &x, &x) == aiednn_success, "gelu");
        aiednn_exec_arg_t a[2] = {{AIEDNN_ARG_SRC, filled(&x, 12)}, {AIEDNN_ARG_DST, filled(&x, 13)}};
        ws_case("gelu_f32", p, 2, a, 1, bytes(&x));
        aiednn_primitive_destroy(p);
        CHECK(aiednn_lrn_forward_create(&p, aiednn_lrn_across_channels, 3, 1e-2f, 0.75f, 1.5f, &x, &x) ==
                  aiednn_success,
              "lrn");
        ws_case("lrn_f32", p, 2, a, 1, bytes(&x));
        /* stream: a failing execute (missing argument) is reported by wait */
        aiednn_stream_t s;
        aiednn_stream_create(&s, 0);
        CHECK(aiednn_primitive_execute_async(p, s, 1, a, nullptr, 0) == aiednn_invalid_arguments, "missing arg");
        CHECK(aiednn_stream_wait(s) == aiednn_invalid_arguments && aiednn_stream_wait(s) == aiednn_success,
              "wait error");
        aiednn_stream_destroy(s);
        aiednn_primitive_destroy(p);
    }
    printf(g_fail ? "STREAM TEST FAIL\n" : "STREAM TEST PASS\n");
    return g_fail;
}
