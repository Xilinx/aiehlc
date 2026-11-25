/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Eager mode, task E2: ONE template kernel (gemm_s8, ../eager/gemm_s8_eager.cc) with TWO instances of
 * different shapes in one host program, launched one after the other on the same partition and then back
 * to the first. Each instance has its own kernel ELF and its own routing (routing_<instance>); the launch
 * selects both. Every result is checked bit-exact against the host reference.
 *
 *   source script/aiehlc.sh --aie-version 2 --runtime-source-file example/aiednn/eager_e2/two_instances_s8.cc
 ******************************************************************************/
#include "gemm_s8_eager.h"

#define M1 128
#define K1 256
#define M2 64
#define K2 128
#define GN 64
#define GTK 64

template <int M, int N, int K, int TK = 64>
__global__ void gemm_s8(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, K>> a,
                        aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, K>> w,
                        aie::port<output_window_int32 *, aie::OFM, aie::shape<M, N>> c) {
    const int TM = 16, TN = 16;
    const int ntile = M / (4 * TM);
    const int nkc = K / TK;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        int32_t acc[TM * TN];
        for (int i = 0; i < TM * TN; i++)
            acc[i] = 0;
        for (int j = 0; j < nkc; j++) {
            const int8_t *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
            const int8_t *v = aie::load_buffer(w, {pn * TN, j * TK}, {TN, TK});
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++) {
                    int32_t s = 0;
                    for (int k = 0; k < TK; k++)
                        s += (int32_t)x[m * TK + k] * (int32_t)v[n * TK + k];
                    acc[m * TN + n] += s;
                }
            aie::release(a);
            aie::release(w);
        }
        int32_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int i = 0; i < TM * TN; i++)
            y[i] = acc[i];
        aie::commit(c);
    }
}

// The meshes must be declared in the function that launches.
static void run1(int8_t *A, int8_t *W, int32_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_s8<M1, GN, K1, GTK><<<mesh>>>(A, W, C);
}

static void run2(int8_t *A, int8_t *W, int32_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_s8<M2, GN, K2, GTK><<<mesh>>>(A, W, C);
}

static int one(int which, int M, int K, uint32_t seed) {
    const long na = (long)M * K, nw = (long)GN * K, nc = (long)M * GN;
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw);
    int32_t *Cd = (int32_t *)__Runtime_Alloc(nc * 4), *ref = (int32_t *)malloc(nc * 4);
    eg_init(Ad, Wd, M, GN, K, seed);
    memset(Cd, 0, nc * 4);
    if (which == 1)
        run1(Ad, Wd, Cd);
    else
        run2(Ad, Wd, Cd);
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc * 4);
    eg_reference(Ad, Wd, ref, M, GN, K);
    printf("[e2] instance %d: gemm_s8<%d, %d, %d, %d>\n", which, M, GN, K, GTK);
    return eg_check(which == 1 ? "gemm_s8 #1" : "gemm_s8 #2", Cd, ref, nc);
}

int main() {
    printf("=== eager E2: two gemm_s8 instances in one program ===\n");
    int bad = one(1, M1, K1, 1234u);
    bad |= one(2, M2, K2, 99u);
    bad |= one(1, M1, K1, 7u); /* back to the first instance: its kernel ELF and routing again */
    printf(bad ? "FAIL: e2 two instances\n" : "PASS: e2 two gemm_s8 instances bit-exact, incl. a switch back\n");
    return bad;
}
