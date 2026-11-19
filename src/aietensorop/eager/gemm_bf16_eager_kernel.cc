/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the float GEMM template. C[M][N] (f32) = A[M][K] . W[N][K]^T with bf16 A, W and f32
 * accumulation (gemm_bf16_eager_ref.h: the layout and the exact float order). gemm_s8's tiling and schedule:
 * each core owns a TM x N / 4 output tile at a time, K streams in TK-element chunks (2 TK bytes). An f32 matmul
 * runs as a bf16x3 split (depth 6 K) built by the runtime, which is what makes every f32 GEMM-family composite
 * (f32 conv, deconv, RNN, attention, gated MLP, linear attention) an AIE path in eager mode.
 * The inner product is scalar float: the same code on the core and in the x86 trace.
 ******************************************************************************/
#include "eager_common.h"
#include "gemm_bf16_eager_ref.h" /* eager_gemm_bf16_ref: the instances' AIEK_REFERENCE */

// M a multiple of 4 * TM, N of 4, K of TK, TK of 2 (whole DMA words: 2 TK bytes); K in bf16 elements; TM >= N / 4
// (W's block {N/4, 2 TK} must not exceed A's {TM, 2 TK}: the core-memory allocator sizes input slots from A)
template <int M, int N, int K, int TK = 32, int TM = 16>
__global__ void gemm_bf16(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, 2 * K>> a,
                          aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, 2 * K>> w,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4, TKB = 2 * TK;
    const int ntile = M / (4 * TM);
    const int nkc = K / TK;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        float acc[TM * TN];
        for (int i = 0; i < TM * TN; i++)
            acc[i] = 0.0f;
        for (int j = 0; j < nkc; j++) {
            const int8_t *x = aie::load_buffer(a, {m0, j * TKB}, {TM, TKB});
            const int8_t *v = aie::load_buffer(w, {pn * TN, j * TKB}, {TN, TKB});
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++) {
                    const uint16_t *xr = (const uint16_t *)(x + m * TKB), *vr = (const uint16_t *)(v + n * TKB);
                    float s = 0.0f;
                    for (int k = 0; k < TK; k++) { /* eager_bf2f, written out: only kernel text reaches the core */
                        const uint32_t ua = (uint32_t)xr[k] << 16, ub = (uint32_t)vr[k] << 16;
                        const float p = *(const float *)&ua * *(const float *)&ub;
                        s = s + p;
                    }
                    acc[m * TN + n] = acc[m * TN + n] + s;
                }
            aie::release(a);
            aie::release(w);
        }
        int32_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int i = 0; i < TM * TN; i++)
            y[i] = *(const int32_t *)&acc[i];
        aie::commit(c);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_gemm_bf16_launch_fn)(int8_t *A, int8_t *W, int32_t *C);

/* args: A [M][2 K] (bf16 bytes), W [N][2 K], C [M][N] f32 bits (the instance's own padded shape) */
static int eager_gemm_bf16_entry(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K,
                                 eager_gemm_bf16_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const size_t nc = (size_t)(M * N) * sizeof(int32_t);
    int8_t *a = (int8_t *)eager_stage_in(0, &args[0], (size_t)(M * 2 * K));
    int8_t *w = (int8_t *)eager_stage_in(1, &args[1], (size_t)(N * 2 * K));
    int32_t *c = (int32_t *)eager_slot(2, nc);
    if (!a || !w || !c)
        return AIEK_ERR_ALLOC;
    launch(a, w, c);
    eager_fetch_out(2, &args[2], nc);
    return AIEK_OK;
}
