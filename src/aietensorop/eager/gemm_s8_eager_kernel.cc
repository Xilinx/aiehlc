/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the int8 GEMM template. C[M][N] (int32) = A[M][K] . W[N][K]^T (int8), one launch per
 * layer on the 4x4 mesh. dnn_build(mode="eager") writes one instance per distinct layer shape (the launch
 * gemm_s8<M, N, K, TK> selects it) with an AIEK_ENTRY calling eager_gemm_s8_entry; each launch loads that
 * instance's kernel ELF, so a model runs eagerly, layer by layer.
 *
 *   tiling      each core owns a TM x N / 4 output tile at a time (all of its mesh column N / 4 columns)
 *   scheduling  core (pm, pn) walks M / (4 TM) tiles of its mesh row
 *   K           streamed in chunks of TK, accumulated in int32 on the core (exact)
 *   output      the int32 accumulators; the runtime applies the epilogue on the host
 * The inner product is scalar: the same code on the core and in the x86 trace.
 ******************************************************************************/
#include "eager_common.h"
#include "gemm_s8_eager_ref.h" /* eager_gemm_s8_ref: the instances' AIEK_REFERENCE in the entry */

// M a multiple of 4 * TM, N of 4 (mesh column pn owns columns pn * N / 4 .. as one block: every port's block
// walk stays two deep, which the kernel analysis can route), K of TK, TK of 4 (whole DMA words)
template <int M, int N, int K, int TK = 64, int TM = 16>
__global__ void gemm_s8(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, K>> a,
                        aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, K>> w,
                        aie::port<output_window_int32 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4;
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

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_gemm_s8_launch_fn)(int8_t *A, int8_t *W, int32_t *C);

/* args: A [M][K] s8, W [N][K] s8, C [M][N] s32 (the instance's own padded shape) */
static int eager_gemm_s8_entry(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K,
                               eager_gemm_s8_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const size_t nc = (size_t)(M * N) * sizeof(int32_t);
    int8_t *a = (int8_t *)eager_stage_in(0, &args[0], (size_t)(M * K));
    int8_t *w = (int8_t *)eager_stage_in(1, &args[1], (size_t)(N * K));
    int32_t *c = (int32_t *)eager_slot(2, nc);
    if (!a || !w || !c)
        return AIEK_ERR_ALLOC;
    launch(a, w, c);
    eager_fetch_out(2, &args[2], nc);
    return AIEK_OK;
}
