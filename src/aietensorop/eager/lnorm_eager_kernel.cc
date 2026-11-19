/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: layer / RMS norm rows (lnorm_eager_ref.h: the layout and the arithmetic). The first row op:
 * it uses the non-GEMM gather lowering (example/tileprogram/ccode/simplerow_shift.cc's pattern) -- mesh row pm
 * owns rows pm R / 4 ..; each TR-row strip of whole rows is broadcast to the row's 4 cores, which all form the row
 * statistics and each write their quarter of the columns, an {TR, L} byte block at {pm R/4 + i TR, pn L}. gamma /
 * beta / the parameters are one resident input (taken once; pre-tiled by the host so each core's part is whole rows,
 * as .resident requires), so only X is a looped input.
 ******************************************************************************/
#include "eager_common.h"
#include "lnorm_eager_ref.h" /* the layout, eager_lnorm_ref (the instances' AIEK_REFERENCE) */

// R a multiple of 4 * TR, L (f32 per row) of 16; TR rows of 4 L bytes double-buffered fit the core
template <int R, int L, int TR, int RMS>
__global__ void lnorm(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 4 * L>> x,
                      aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<12, L>> p,
                      aie::port<output_window_int8 *, aie::OFM, aie::shape<R, 4 * L>> y) {
    const int rows = R / 4, q = L / 4; /* rows per mesh row; f32 columns per core */
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    const int8_t *pp = aie::load_buffer(p, {3 * pn, 0}, {3, L}, {.resident = true}); /* whole rows: host pre-tiled */
    const float *gam = (const float *)pp, *bet = (const float *)(pp + L);
    const float eps = ((const float *)(pp + 2 * L))[0];
    const int32_t lt = ((const int32_t *)(pp + 2 * L))[2];
    const int Lt = lt < 0 ? 0 : lt > L ? L : lt; /* never past the row (the trace runs on arbitrary bytes) */
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *a = aie::load_buffer(x, {pm * rows + i * TR, 0}, {TR, 4 * L});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * L}, {TR, L});
        for (int r = 0; r < TR; r++) { /* eager_lnorm_row, written out: only the kernel's own text reaches the core */
            const float *xr = (const float *)(a + r * 4 * L);
            float mean = 0.0f, var = 0.0f, cm = 0.0f, cv = 0.0f; /* Kahan sums */
            if (!RMS) {
                for (int k = 0; k < Lt; k++) {
                    const float yk = xr[k] - cm, tk = mean + yk;
                    cm = (tk - mean) - yk;
                    mean = tk;
                }
                mean /= (float)Lt; /* 1 / L is not exact in f32: divide */
            }
            for (int k = 0; k < Lt; k++) {
                const float d = xr[k] - mean, yk = d * d - cv, tk = var + yk;
                cv = (tk - var) - yk;
                var = tk;
            }
            const float v = var / (float)Lt + eps;
            uint32_t u = *(const uint32_t *)&v;
            u = 0x5f3759dfu - (u >> 1);
            float rs = *(const float *)&u;
            for (int k = 0; k < 4; k++)
                rs = rs * (1.5f - 0.5f * v * rs * rs);
            float *yr = (float *)(o + r * L);
            for (int c = 0; c < q; c++) {
                const int k = pn * q + c;
                yr[c] = k < Lt ? (xr[k] - mean) * rs * gam[c] + (RMS ? 0.0f : bet[c]) : 0.0f;
            }
        }
        aie::release(x);
        aie::commit(y);
    }
    aie::release(p);
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_lnorm_launch_fn)(int8_t *X, int8_t *P, int8_t *Y);

/* args: X [R][4 L] f32 bytes, P [12][L] (per core: gamma, beta, params quarters), Y [R][4 L] f32 bytes */
static int eager_lnorm_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t L,
                             eager_lnorm_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    int8_t *x = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * 4 * L));
    int8_t *p = (int8_t *)eager_stage_in(1, &args[1], (size_t)(12 * L));
    int8_t *y = (int8_t *)eager_slot(2, (size_t)(R * 4 * L));
    if (!x || !p || !y)
        return AIEK_ERR_ALLOC;
    launch(x, p, y);
    eager_fetch_out(2, &args[2], (size_t)(R * 4 * L));
    return AIEK_OK;
}
