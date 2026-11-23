/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: f32 unary pointwise -- Y = F(X; alpha, beta) for one algorithm F compiled into the instance
 * (felt_eager_ref.h: the layout; eager_fmath_ops.cc: the math, expanded in the kernel body).
 * A row op on the non-GEMM gather lowering: mesh row pm owns rows pm R / HW_ROWS .., each TR-row strip goes to the
 *row's 4 cores (inputs are broadcast along a mesh row) and core pn writes its quarter of the columns.
 ******************************************************************************/
#include "eager_common.h"
#include "eager_fmath_ops.cc" // EAGER_FMATH_OPS: the math as text (spliced; the kernel expands it)
#include "felt_eager_ref.h"   /* the layout, eager_feltw_ref (the instances' AIEK_REFERENCE) */

// R a multiple of HW_ROWS * TR, L (f32 per row) of 16
template <int R, int L, int TR, int F>
__global__ void feltw(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 4 * L>> x,
                      aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<4, 16>> p,
                      aie::port<output_window_int8 *, aie::OFM, aie::shape<R, 4 * L>> y) {
    const int rows = R / HW_ROWS, q = L / 4; /* rows per mesh row; f32 columns per core */
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    EAGER_FMATH_OPS;
    const float *ab = (const float *)aie::load_buffer(p, {pn, 0}, {1, 16}, {.resident = true});
    const float a = ab[0], b = ab[1];
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *s = aie::load_buffer(x, {pm * rows + i * TR, 0}, {TR, 4 * L});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * L}, {TR, L});
        for (int r = 0; r < TR; r++) {
            const float *xr = (const float *)(s + r * 4 * L) + pn * q;
            float *yr = (float *)(o + r * L);
            for (int c = 0; c < q; c++)
                yr[c] = eager_fop(F, xr[c], 0.0f, a, b);
        }
        aie::release(x);
        aie::commit(y);
    }
    aie::release(p);
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_feltw_launch_fn)(int8_t *X, int8_t *P, int8_t *Y);

/* args: X [R][L] f32, P [4][4] f32 ({alpha, beta} per row), Y [R][L] f32 */
static int eager_feltw_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t L,
                             eager_feltw_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    int8_t *x = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * 4 * L));
    int8_t *p = (int8_t *)eager_stage_in(1, &args[1], 64);
    int8_t *y = (int8_t *)eager_slot(2, (size_t)(R * 4 * L));
    if (!x || !p || !y)
        return AIEK_ERR_ALLOC;
    launch(x, p, y);
    eager_fetch_out(2, &args[2], (size_t)(R * 4 * L));
    return AIEK_OK;
}
