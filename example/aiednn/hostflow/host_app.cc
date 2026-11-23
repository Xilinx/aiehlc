/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * host_app -- a host program around a three-op model.
 *
 *   y = x @ W^T          library (graph partition)
 *   u = 0.5 * y          this file
 *   z = relu(u)          library
 *
 * The host fills x, walks the partitions, and reads z. Weights stay put.
 * dnn_build picks kernels by scanning this file for create calls, so the
 * matmul the partition compiles to is named here: aiednn_matmul_forward_create(
 *
 *   PYTHONPATH=src python -m aiegraph dnn-build example/aiednn/hostflow/host_app.cc --target x86 --run
 *   ... --target vek280 --run --board <host> --vitis-settings <.../settings64.sh>
 ******************************************************************************/
#include "aiednn_graph.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define M 2
#define K 8
#define N 4
#define IMAGES 2
#define SCALE 0.5f

enum { ID_X = 1, ID_W, ID_Y, ID_U, ID_Z };

static float g_x_img[IMAGES][M * K] __attribute__((aligned(64)));
static float g_w[N * K] __attribute__((aligned(64)));
static float g_y[M * N] __attribute__((aligned(64)));
static float g_u[M * N] __attribute__((aligned(64)));
static float g_z[M * N] __attribute__((aligned(64)));
static float *g_x;

static int fail(const char *what, aiednn_status_t st) {
    printf("FAIL: %s%s%s\n", what, st ? ": " : "", st ? aiednn_status_str(st) : "");
    return 1;
}

static void *data_of(uint64_t id) {
    switch (id) {
    case ID_X:
        return g_x;
    case ID_W:
        return g_w;
    case ID_Y:
        return g_y;
    case ID_U:
        return g_u;
    case ID_Z:
        return g_z;
    default:
        return nullptr;
    }
}

static aiednn_logical_tensor_t tensor(uint64_t id, int ndims, const int64_t *dims, int constant) {
    aiednn_logical_tensor_t t;
    memset(&t, 0, sizeof t);
    t.id = id;
    t.property = constant ? aiednn_lt_constant : aiednn_lt_variable;
    aiednn_memory_desc_init(&t.md, ndims, dims, aiednn_f32);
    return t;
}

static aiednn_status_t add_op(aiednn_graph_t g, uint64_t op, aiednn_graph_op_kind_t kind, aiednn_graph_op_attr_t *attr,
                              const aiednn_logical_tensor_t *in, int n_in, aiednn_logical_tensor_t out) {
    return aiednn_graph_add_op(g, op, kind, attr, n_in, in, 1, &out);
}

struct step {
    int host;
    aiednn_partition_t part;
    aiednn_compiled_partition_t cp;
};

static int bind(aiednn_partition_t part, aiednn_tensor_t *in, int *n_in, aiednn_tensor_t *out, int *n_out) {
    aiednn_logical_tensor_t lt[4];
    *n_in = aiednn_partition_get_inputs(part, lt, 4);
    for (int i = 0; i < *n_in; i++) {
        in[i].id = lt[i].id;
        in[i].data = data_of(lt[i].id);
        if (!in[i].data)
            return fail("partition input has no buffer", aiednn_invalid_arguments);
    }
    *n_out = aiednn_partition_get_outputs(part, lt, 4);
    for (int i = 0; i < *n_out; i++) {
        out[i].id = lt[i].id;
        out[i].data = data_of(lt[i].id);
        if (!out[i].data)
            return fail("partition output has no buffer", aiednn_invalid_arguments);
    }
    return 0;
}

static int run_host(aiednn_partition_t part) {
    aiednn_tensor_t in[4], out[4];
    int n_in, n_out;
    if (bind(part, in, &n_in, out, &n_out) || n_in != 1 || n_out != 1)
        return fail("host scale wants one input and one output", aiednn_invalid_arguments);
    const int n = M * N;
    const float *a = (const float *)in[0].data;
    float *b = (float *)out[0].data;
    for (int i = 0; i < n; i++)
        b[i] = SCALE * a[i];
    return 0;
}

static int run_library(const struct step *s) {
    aiednn_tensor_t in[4], out[4];
    int n_in, n_out;
    if (bind(s->part, in, &n_in, out, &n_out))
        return 1;
    aiednn_status_t st = aiednn_compiled_partition_execute(s->cp, nullptr, n_in, in, n_out, out, nullptr, 0);
    return st ? fail(aiednn_partition_info(s->part), st) : 0;
}

static int build(aiednn_graph_t *g, struct step *steps, int *nsteps) {
    const int64_t xd[2] = {M, K}, wd[2] = {N, K}, yd[2] = {M, N};
    aiednn_logical_tensor_t x = tensor(ID_X, 2, xd, 0);
    aiednn_logical_tensor_t w = tensor(ID_W, 2, wd, 1);
    aiednn_logical_tensor_t y = tensor(ID_Y, 2, yd, 0);
    aiednn_logical_tensor_t u = tensor(ID_U, 2, yd, 0);
    aiednn_logical_tensor_t z = tensor(ID_Z, 2, yd, 0);
    aiednn_graph_op_attr_t attr;
    aiednn_status_t st;
    if ((st = aiednn_graph_create(g)))
        return fail("graph_create", st);
    aiednn_graph_op_attr_init(&attr);
    attr.weights_transposed = 1;
    aiednn_logical_tensor_t xw[2] = {x, w};
    if ((st = add_op(*g, 1, aiednn_gop_matmul, &attr, xw, 2, y)))
        return fail("add matmul", st);
    aiednn_graph_op_attr_init(&attr);
    if ((st = add_op(*g, 2, aiednn_gop_wildcard, &attr, &y, 1, u)))
        return fail("add scale", st);
    if ((st = add_op(*g, 3, aiednn_gop_relu, &attr, &u, 1, z)))
        return fail("add relu", st);
    if ((st = aiednn_graph_finalize(*g)))
        return fail("finalize", st);

    aiednn_partition_t parts[8];
    *nsteps = aiednn_graph_get_partitions(*g, parts, 8);
    for (int i = 0; i < *nsteps; i++) {
        steps[i].part = parts[i];
        steps[i].cp = nullptr;
        steps[i].host = !aiednn_partition_is_supported(parts[i]);
        if (steps[i].host) {
            printf("part %d host     %s\n", i, aiednn_partition_info(parts[i]));
            continue;
        }
        if ((st = aiednn_partition_compile(parts[i], &steps[i].cp)))
            return fail(aiednn_partition_info(parts[i]), st);
        printf("part %d library  %s\n", i, aiednn_compiled_partition_impl_info(steps[i].cp));
    }
    return 0;
}

static float expect(int m, int n) {
    float acc = 0.f;
    for (int k = 0; k < K; k++)
        acc += g_x[m * K + k] * g_w[n * K + k];
    acc *= SCALE;
    return acc > 0.f ? acc : 0.f;
}

static int check_image(int image) {
    float worst = 0.f;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float d = fabsf(g_z[m * N + n] - expect(m, n));
            if (d > worst)
                worst = d;
        }
    printf("image %d max_diff %.3g\n", image, worst);
    return worst > 1e-4f ? fail("output differs from the host reference", aiednn_success) : 0;
}

int main() { // "int main()": aiehlc reads "(void)" as no main and archives a library
    printf("=== hostflow: matmul, host scale %.2f, relu ===\n", SCALE);
    for (int i = 0; i < N * K; i++)
        g_w[i] = 0.25f * (float)((i % 5) - 2);
    for (int image = 0; image < IMAGES; image++)
        for (int i = 0; i < M * K; i++)
            g_x_img[image][i] = (float)(((i + 3 * image) % 7) - 3);

    aiednn_graph_t g = nullptr;
    struct step steps[8];
    int nsteps = 0;
    if (build(&g, steps, &nsteps))
        return 1;
    for (int image = 0; image < IMAGES; image++) {
        g_x = g_x_img[image];
        for (int i = 0; i < nsteps; i++)
            if (steps[i].host ? run_host(steps[i].part) : run_library(&steps[i]))
                return 1;
        if (check_image(image))
            return 1;
    }
    for (int i = 0; i < nsteps; i++)
        if (steps[i].cp)
            aiednn_compiled_partition_destroy(steps[i].cp);
    aiednn_graph_destroy(g);
    printf("PASS: hostflow, %d images\n", IMAGES);
    return 0;
}
