/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the int8 GEMM with the requant on the core. A different kernel from gemm_s8: the weights
 * carry each channel's requant parameters and the output is 8-bit codes, so the layer's int8 result leaves the
 * array directly. C[M][N] = requant(A[M][K] . W[N][K]^T) with aiednn's fixed-point definition
 * (gemm_q_eager_ref.h).
 *
 * The parameters ride in the weights tensor, W' [N][K + TK]: row n is channel n's K weights, then one more
 * chunk whose first 32 bytes are its 8 int32 Q words; A' [M][K + TK] ends in a zero chunk, so both K walks stay
 * equal (the GEMM lowering takes one K loop for both ports). A separate Q port would need more shim streams than
 * VEK280's partition has (4 DDR columns x 2 MM2S, all taken by A and W). Tiling as gemm_s8.
 ******************************************************************************/
#include "eager_common.h"
#include "gemm_q_eager_ref.h" /* the Q words, eager_gemm_q_ref (the instances' AIEK_REFERENCE) */

// M a multiple of 4 * TM, N of 4, K of TK, TK of 4 and >= 32 (the Q chunk holds 8 int32 per channel), TN * 8
// int32 of Q fit the core's stack
template <int M, int N, int K, int TK = 64, int TM = 16>
__global__ void gemm_q(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, K + TK>> a,
                       aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, K + TK>> w,
                       aie::port<output_window_int8 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4;
    const int ntile = M / (4 * TM);
    const int nkc = K / TK;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        int32_t acc[TM * TN];
        int32_t qw[TN * 8];
        for (int i = 0; i < TM * TN; i++)
            acc[i] = 0;
        for (int j = 0; j <= nkc; j++) {
            const int8_t *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
            const int8_t *v = aie::load_buffer(w, {pn * TN, j * TK}, {TN, TK});
            if (j < nkc) {
                for (int m = 0; m < TM; m++)
                    for (int n = 0; n < TN; n++) {
                        int32_t s = 0;
                        for (int k = 0; k < TK; k++)
                            s += (int32_t)x[m * TK + k] * (int32_t)v[n * TK + k];
                        acc[m * TN + n] += s;
                    }
            } else { /* the Q chunk: each channel's 8 words lead its row */
                for (int n = 0; n < TN; n++)
                    for (int e = 0; e < 8; e++)
                        qw[n * 8 + e] = ((const int32_t *)(v + n * TK))[e];
            }
            aie::release(a);
            aie::release(w);
        }
        int8_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int m = 0; m < TM; m++)
            for (int n = 0; n < TN; n++) { /* eager_q_code, written out: only the kernel's own text reaches the core */
                const int32_t *p = qw + n * 8;
                const int64_t B = (int64_t)(((uint64_t)(uint32_t)p[2] << 32) | (uint32_t)p[1]);
                const int64_t r = (((int64_t)acc[m * TN + n] * (int64_t)p[0] + B) >> p[3]) + p[4];
                y[m * TN + n] = (int8_t)(r < p[5] ? p[5] : r > p[6] ? p[6] : r);
            }
        aie::commit(c);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_gemm_q_launch_fn)(int8_t *A, int8_t *Wq, int8_t *C);

/* args: A' [M][K + TK] s8 (zero last chunk), W' [N][K + TK] s8 (weights + Q chunk), C [M][N] 8-bit */
static int eager_gemm_q_entry(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K, int64_t TK,
                              eager_gemm_q_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    int8_t *a = (int8_t *)eager_stage_in(0, &args[0], (size_t)(M * (K + TK)));
    int8_t *w = (int8_t *)eager_stage_in(1, &args[1], (size_t)(N * (K + TK)));
    int8_t *c = (int8_t *)eager_slot(2, (size_t)(M * N));
    if (!a || !w || !c)
        return AIEK_ERR_ALLOC;
    launch(a, w, c);
    eager_fetch_out(2, &args[2], (size_t)(M * N));
    return AIEK_OK;
}
