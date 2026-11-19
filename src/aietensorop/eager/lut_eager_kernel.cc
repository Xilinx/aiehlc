/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: table lookup, Y[m][n] = table_n[A[m][n]] for 8-bit codes (lut_eager_ref.h: the layout).
 * The host builds the tables by running ref: on all 256 codes, so eltwise / PReLU / batch norm / broadcast
 * binary / 8-bit reorders are exact by construction.
 *
 * Routing: the GEMM skeleton (kernel-only ports lower through the GEMM path). The K walk is 4 chunks of tables
 * (W rows; A's are not read), then N / 64 chunks of codes (A; W's are not read). Mesh column pn owns columns
 * pn N/4 .. -- its tables arrive in its W rows; it reads its codes from the data chunk that holds them.
 ******************************************************************************/
#include "eager_common.h"
#include "lut_eager_ref.h" /* the layout, eager_lut_ref (the instances' AIEK_REFERENCE) */

// M a multiple of 4 * TM, N 64 or 128 (TN = N / 4 tables of 256 B on the core's stack); TM * 64 >= TN * 64, i.e.
// TM >= N / 4: the core-memory allocator sizes every input port's ping / pong from A's block
template <int M, int N, int TM = 16>
__global__ void lut(aie::port<input_window_int8 *, aie::IFM, aie::shape<M, 256 + N>> a,
                    aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<N, 256 + N>> w,
                    aie::port<output_window_int8 *, aie::OFM, aie::shape<M, N>> c) {
    const int TN = N / 4, TK = 64, nkc = (256 + N) / TK;
    const int ntile = M / (4 * TM);
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    const int dj = 4 + pn * TN / TK, off = pn * TN % TK; /* the data chunk holding this core's columns */
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * TM) + t * TM;
        uint8_t tab[TN * 256];
        int8_t code[TM * TN];
        for (int j = 0; j < nkc; j++) {
            const int8_t *x = aie::load_buffer(a, {m0, j * TK}, {TM, TK});
            const int8_t *v = aie::load_buffer(w, {pn * TN, j * TK}, {TN, TK});
            if (j < 4) {
                for (int n = 0; n < TN; n++)
                    for (int k = 0; k < TK; k++)
                        tab[n * 256 + j * TK + k] = (uint8_t)v[n * TK + k];
            } else if (j == dj) {
                for (int m = 0; m < TM; m++)
                    for (int n = 0; n < TN; n++)
                        code[m * TN + n] = (int8_t)tab[n * 256 + (uint8_t)x[m * TK + off + n]];
            }
            aie::release(a);
            aie::release(w);
        }
        int8_t *y = aie::store_buffer(c, {m0, pn * TN}, {TM, TN});
        for (int i = 0; i < TM * TN; i++)
            y[i] = code[i];
        aie::commit(c);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_lut_launch_fn)(int8_t *A, int8_t *W, int8_t *Y);

/* args: A [M][256 + N] codes, W [N][256 + N] tables, Y [M][N] */
static int eager_lut_entry(const aiek_tensor *args, int num_args, int64_t M, int64_t N, eager_lut_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const int64_t ld = EG_LUT_TABLE + N;
    int8_t *a = (int8_t *)eager_stage_in(0, &args[0], (size_t)(M * ld));
    int8_t *w = (int8_t *)eager_stage_in(1, &args[1], (size_t)(N * ld));
    int8_t *y = (int8_t *)eager_slot(2, (size_t)(M * N));
    if (!a || !w || !y)
        return AIEK_ERR_ALLOC;
    launch(a, w, y);
    eager_fetch_out(2, &args[2], (size_t)(M * N));
    return AIEK_OK;
}
