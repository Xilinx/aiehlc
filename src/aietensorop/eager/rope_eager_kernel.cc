/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: rotary position embedding of f32 rows (rope_eager_ref.h: the layout and the arithmetic). A row
 * op on the non-GEMM gather lowering, with ONE looped input: the host packs every row as [x | cos | sin]; mesh row
 * pm owns rows pm R/4 .., each TR-row strip of whole rows goes to the row's 4 cores, and core pn writes columns
 * pn D/4 .. of its rows (the rotated partner may lie in another core's quarter: every core holds the whole row).
 ******************************************************************************/
#include "eager_common.h"
#include "rope_eager_ref.h" /* the layout, eager_rope_ref (the instances' AIEK_REFERENCE) */

// R a multiple of 4 * TR, D of 16 (each core's D/4 floats are whole DMA words)
template <int R, int D, int TR, int INTER>
__global__ void rope(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 12 * D>> x,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<R, 4 * D>> y) {
    const int rows = R / 4, q = D / 4, h = D / 2;
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *a = aie::load_buffer(x, {pm * rows + i * TR, 0}, {TR, 12 * D});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * D}, {TR, D});
        for (int r = 0; r < TR; r++) { /* eager_rope_elem, written out: only the kernel's own text reaches the core */
            const float *xr = (const float *)(a + r * 12 * D), *cr = xr + D, *sr = xr + 2 * D;
            float *yr = (float *)(o + r * D);
            for (int c = 0; c < q; c++) {
                const int j = pn * q + c;
                const int pj = INTER ? (j ^ 1) : (j < h ? j + h : j - h);
                const float sign = INTER ? ((j & 1) ? 1.0f : -1.0f) : (j < h ? -1.0f : 1.0f);
                const float m1 = xr[j] * cr[j], m2 = (sign * xr[pj]) * sr[j];
                yr[c] = m1 + m2;
            }
        }
        aie::release(x);
        aie::commit(y);
    }
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_rope_launch_fn)(int8_t *X, int8_t *Y);

/* args: X [R][12 D] ([x | cos | sin] f32 bytes per row), Y [R][4 D] f32 bytes */
static int eager_rope_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t D, eager_rope_launch_fn launch) {
    if (num_args != 2)
        return AIEK_ERR_ARGS;
    int8_t *x = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * 12 * D));
    int8_t *y = (int8_t *)eager_slot(1, (size_t)(R * 4 * D));
    if (!x || !y)
        return AIEK_ERR_ALLOC;
    launch(x, y);
    eager_fetch_out(1, &args[1], (size_t)(R * 4 * D));
    return AIEK_OK;
}
