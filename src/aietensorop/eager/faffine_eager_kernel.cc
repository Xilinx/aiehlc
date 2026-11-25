/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: f32 per-row affine -- Y = F(X; c0, c1) with two constants per row, F compiled into the
 * instance: affine x c0 + c1 (inference batch norm), affine + ReLU (batch norm with the fused ReLU), PReLU with a
 * per-row slope c0 (felt_eager_ref.h: the layout; eager_fmath_ops.cc: the math). The host lays each channel's
 * contiguous block out as rows, so a row's elements share one channel's constants.
 * Each row carries its constants after its values ([x | c0 c1 - -]): ONE streamed input, so an 8-row mesh fits the
 * partition's 20 shim streams. Whole-row strips broadcast along the mesh row; core pn writes its quarter.
 ******************************************************************************/
#include "eager_common.h"
#include "eager_fmath_ops.cc" // EAGER_FMATH_OPS: the math as text (spliced; the kernel expands it)
#include "felt_eager_ref.h"   /* the layout, eager_faffine_ref (the instances' AIEK_REFERENCE) */

// R a multiple of HW_ROWS * TR, L (f32 per row) of 16
template <int R, int L, int TR, int F>
__global__ void faffine(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 4 * L + 16>> xc,
                        aie::port<output_window_int8 *, aie::OFM, aie::shape<R, 4 * L>> y) {
    const int rows = R / HW_ROWS, q = L / 4;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    EAGER_FMATH_OPS;
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *s = aie::load_buffer(xc, {pm * rows + i * TR, 0}, {TR, 4 * L + 16});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * L}, {TR, L});
        for (int r = 0; r < TR; r++) {
            const float *xr = (const float *)(s + r * (4 * L + 16)) + pn * q,
                        *kr = (const float *)(s + r * (4 * L + 16)) + L;
            const float c0 = kr[0], c1 = kr[1];
            float *yr = (float *)(o + r * L);
            for (int j = 0; j < q; j++)
                yr[j] = eager_fop(F, xr[j], c0, c0, c1);
        }
        aie::release(xc);
        aie::commit(y);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_faffine_launch_fn)(int8_t *XC, int8_t *Y);

/* args: XC [R][L + 4] f32 (each row its L values, then {c0, c1, -, -}), Y [R][L] f32 */
static int eager_faffine_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t L,
                               eager_faffine_launch_fn launch) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    int8_t *xc = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * (4 * L + 16)));
    int8_t *y = (int8_t *)eager_slot(1, (size_t)(R * 4 * L));
    if (!xc || !y)
        return AIEK_ERR_ALLOC;
    launch(xc, y);
    eager_fetch_out(1, &args[1], (size_t)(R * 4 * L));
    return AIEK_OK;
}
