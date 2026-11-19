/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: the int8 GEMM with the requant AND an 8-bit residual add on the core (a ResNet block's
 * conv -> + skip -> ReLU as one launch). gemm_q's layout plus the residual codes: both operands carry
 * XC = 1 + ceil(N / TK) chunks past K -- W's first holds the 12 requant words, A's chunks 1.. hold each output
 * pixel's residual codes for every channel (gemm_qr_eager_ref.h). The residual rides in A because VEK280's
 * partition has no shim stream left for a fourth port.
 ******************************************************************************/
#include "eager_common.h"
#include "gemm_qr_eager_ref.h" /* the operand layout, eager_gemm_qr_ref (the instances' AIEK_REFERENCE) */

// M a multiple of 4 * TM, N of 4, K of TK, TK of 4 and >= 48 (12 int32 per channel)
template <int M, int N, int K, int TK = 64, int TM = 16>
__global__ void gemm_qr(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, K + (1 + (N + TK - 1) / TK) * TK>> a,
                        aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, K + (1 + (N + TK - 1) / TK) * TK>> w,
                        aie::port<output_window_int8 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4;
    const int ntile = M / (4 * TM);
    const int nkc = K / TK;
    const int nch = nkc + 1 + (N + TK - 1) / TK; /* K chunks, the Q chunk, the residual chunks */
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        int32_t acc[TM * TN];
        int32_t qw[TN * 12];
        int8_t rr[TM * TN];
        for (int i = 0; i < TM * TN; i++)
            acc[i] = 0;
        for (int j = 0; j < nch; j++) {
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
            } else if (j == nkc) { /* the Q chunk: each channel's 12 words lead its row */
                for (int n = 0; n < TN; n++)
                    for (int e = 0; e < 12; e++)
                        qw[n * 12 + e] = ((const int32_t *)(v + n * TK))[e];
            } else { /* residual chunk: global columns (j - nkc - 1) * TK .. ; keep this core's */
                const int g0 = (j - nkc - 1) * TK;
                for (int n = 0; n < TN; n++) {
                    const int gn = pn * TN + n;
                    if (gn >= g0 && gn < g0 + TK)
                        for (int m = 0; m < TM; m++)
                            rr[m * TN + n] = x[m * TK + gn - g0];
                }
            }
            aie::release(a);
            aie::release(w);
        }
        int8_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int m = 0; m < TM; m++)
            for (int n = 0; n < TN; n++) { /* eager_qr_code, written out: only the kernel's own text reaches the core */
                const int32_t *p = qw + n * 12;
                const int64_t B = (int64_t)(((uint64_t)(uint32_t)p[2] << 32) | (uint32_t)p[1]);
                const int64_t R = (int64_t)(((uint64_t)(uint32_t)p[9] << 32) | (uint32_t)p[8]);
                const int8_t rc = rr[m * TN + n];
                const int64_t r = (p[10] ? (int64_t)(uint8_t)rc : (int64_t)rc) - p[7];
                const int64_t s = (((int64_t)acc[m * TN + n] * (int64_t)p[0] + B + r * R) >> p[3]) + p[4];
                y[m * TN + n] = (int8_t)(s < p[5] ? p[5] : s > p[6] ? p[6] : s);
            }
        aie::commit(c);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_gemm_qr_launch_fn)(int8_t *A, int8_t *W, int8_t *C);

/* args: A' / W' [.][K + XC*TK] (gemm_qr_eager_ref.h), C [M][N] 8-bit (the instance's own padded shape) */
static int eager_gemm_qr_entry(const aiek_tensor *args, int num_args, int64_t M, int64_t N, int64_t K, int64_t TK,
                               eager_gemm_qr_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int64_t ld = K + eager_qr_xc(N, TK) * TK;
    int8_t *a = (int8_t *)eager_stage_in(0, &args[0], (size_t)(M * ld));
    int8_t *w = (int8_t *)eager_stage_in(1, &args[1], (size_t)(N * ld));
    int8_t *c = (int8_t *)eager_slot(2, (size_t)(M * N));
    if (!a || !w || !c)
        return AIEK_ERR_ALLOC;
    launch(a, w, c);
    eager_fetch_out(2, &args[2], (size_t)(M * N));
    return AIEK_OK;
}
