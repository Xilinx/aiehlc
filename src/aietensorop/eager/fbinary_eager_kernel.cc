/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: f32 binary -- Z = F(X, Y) for one operation F compiled into the instance (add / sub / mul /
 * div / max / min / comparisons / logical; felt_eager_ref.h: the layout; eager_fmath_ops.cc: the math). The two
 * operands travel side by side in ONE input row ([x | y]): a partition gives an instance 20 shim streams, and two
 * streamed inputs per mesh row do not fit at 8 rows (16 + 8 outputs). Whole-row strips broadcast along the mesh row;
 * core pn writes its quarter of the columns.
 ******************************************************************************/
#include "eager_common.h"
#include "eager_fmath_ops.cc" // EAGER_FMATH_OPS: the math as text (spliced; the kernel expands it)
#include "felt_eager_ref.h"   /* the layout, eager_fbinary_ref (the instances' AIEK_REFERENCE) */

// R a multiple of HW_ROWS * TR, L (f32 per row) of 16
template <int R, int L, int TR, int F>
__global__ void fbinary(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 8 * L>> xy,
                        aie::port<output_window_int8 *, aie::OFM, aie::shape<R, 4 * L>> z) {
    const int rows = R / HW_ROWS, q = L / 4;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    EAGER_FMATH_OPS;
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *s = aie::load_buffer(xy, {pm * rows + i * TR, 0}, {TR, 8 * L});
        int8_t *o = aie::store_buffer(z, {pm * rows + i * TR, pn * L}, {TR, L});
        for (int r = 0; r < TR; r++) {
            const float *xr = (const float *)(s + r * 8 * L) + pn * q,
                        *yr = (const float *)(s + r * 8 * L) + L + pn * q;
            float *zr = (float *)(o + r * L);
            for (int c = 0; c < q; c++)
                zr[c] = eager_fop(F, xr[c], yr[c], 0.0f, 0.0f);
        }
        aie::release(xy);
        aie::commit(z);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_fbinary_launch_fn)(int8_t *XY, int8_t *Z);

/* args: XY [R][2 L] f32 (each row x then y), Z [R][L] f32 */
static int eager_fbinary_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t L,
                               eager_fbinary_launch_fn launch) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    int8_t *xy = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * 8 * L));
    int8_t *z = (int8_t *)eager_slot(1, (size_t)(R * 4 * L));
    if (!xy || !z)
        return AIEK_ERR_ALLOC;
    launch(xy, z);
    eager_fetch_out(1, &args[1], (size_t)(R * 4 * L));
    return AIEK_OK;
}
