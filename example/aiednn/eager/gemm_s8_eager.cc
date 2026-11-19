/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, the first template: C[M][N] (int32) = A[M][K] . W[N][K]^T (int8), one launch per
 * layer on the 4x4 mesh. Each layer's shape is one instance of the template; the launch loads that
 * instance's kernel ELF, so a model runs eagerly, layer by layer.
 *
 *   tiling      each core owns a 16 x 16 output tile at a time
 *   scheduling  core (pm, pn) walks M / (4 * 16) tiles of its mesh row inside the launch; mesh column
 *               pn owns output columns 16 pn .. 16 pn + 15 (N = 64 in this template)
 *   K           streamed in chunks of TK, accumulated in int32 on the core (exact)
 *   output      the int32 accumulators: the caller applies the epilogue
 * The inner product is scalar -- the same code on the core and in the x86 trace. Vector (mmul)
 * instances come later as their own template.
 ******************************************************************************/
#include "gemm_s8_eager.h"

// ---- this program's instance ----
#define GM 128
#define GN 64
#define GK 256
#define GTK 64
// ---- end of instance ----

// M a multiple of 4 * 16, N = 4 * 16, K a multiple of TK
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

static void gemm_s8_run(int8_t *A, int8_t *W, int32_t *C) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    gemm_s8<GM, GN, GK, GTK><<<mesh>>>(A, W, C);
}

int main() {
    printf("=== aiednn eager gemm_s8<%d, %d, %d, %d>: C = A . W^T, int8 -> int32, one launch ===\n", GM, GN, GK, GTK);
    const long na = (long)GM * GK, nw = (long)GN * GK, nc = (long)GM * GN;
    int8_t *Ad = (int8_t *)__Runtime_Alloc(na), *Wd = (int8_t *)__Runtime_Alloc(nw);
    int32_t *Cd = (int32_t *)__Runtime_Alloc(nc * 4), *ref = (int32_t *)malloc(nc * 4);
    eg_init(Ad, Wd, GM, GN, GK, 1234u);
    memset(Cd, 0, nc * 4);
    const uint64_t t0 = eg_now();
    gemm_s8_run(Ad, Wd, Cd);
    const uint64_t t1 = eg_now();
    Xil_DCacheInvalidateRange((UINTPTR)Cd, nc * 4);
    printf("[gemm_s8] launch %.3f ms\n", eg_ms(t0, t1));
    eg_reference(Ad, Wd, ref, GM, GN, GK);
    const int bad = eg_check("gemm_s8", Cd, ref, nc);
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad;
}
