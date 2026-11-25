/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode: f32 row operations -- one operation OP compiled into the instance (frow_eager_ref.h: the
 * layout and the operations; eager_fmath_ops.cc: the math). Softmax / log softmax write their rows; the reductions
 * (sum, mean, max, min, sum / divisor) write one f32 per row. The host gathers rows of any axis / any reduced
 * sub-tensor / any pooling window, so this one template serves softmax, reductions and f32 pooling.
 * A row op on the non-GEMM gather lowering: mesh row pm owns rows pm R / HW_ROWS .., each TR-row strip of whole rows
 * goes to the row's 4 cores; core pn writes its quarter of a softmax row, or (reductions) the row's value into its
 * own 4 bytes of the 16-byte output row (the host reads column 0).
 ******************************************************************************/
#include "eager_common.h"
#include "eager_fmath_ops.cc" // EAGER_FMATH_OPS: the math as text (spliced; the kernel expands it)
#include "frow_eager_ref.h"   /* the layout, eager_frow_ref (the instances' AIEK_REFERENCE) */

// R a multiple of HW_ROWS * TR, L (f32 per row) of 16; YB the output row's bytes: 16 for a reduction (OP >=
// EG_ROW_SUM), else 4 L (a template argument: port shapes take arithmetic, not a conditional)
template <int R, int L, int TR, int OP, int YB>
__global__ void frow(aie::port<input_window_int8 *, aie::IFM, aie::shape<R, 4 * L>> x,
                     aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<4, 16>> p,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<R, YB>> y) {
    const int rows = R / HW_ROWS, q = L / 4, ob = YB / 4; /* output bytes per row and core */
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    EAGER_FMATH_OPS;
    const int32_t lt = ((const int32_t *)aie::load_buffer(p, {pn, 0}, {1, 16}, {.resident = true}))[0];
    const int Lt = lt < 1 ? 1 : lt > L ? L : lt; /* the row's valid length (the trace runs on arbitrary bytes) */
    for (int i = 0; i < rows / TR; i++) {
        const int8_t *a = aie::load_buffer(x, {pm * rows + i * TR, 0}, {TR, 4 * L});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * TR, pn * ob}, {TR, ob});
        for (int r = 0; r < TR; r++) { /* eager_frow_row, written out: only the kernel's own text reaches the core */
            const float *xr = (const float *)(a + r * 4 * L);
            float *yr = (float *)(o + r * ob);
            if (OP == EG_ROW_SOFTMAX || OP == EG_ROW_LOGSOFTMAX) {
                float mx = xr[0];
                for (int k = 1; k < Lt; k++)
                    mx = eg_fmax(mx, xr[k]);
                float s = 0.0f, cs = 0.0f; /* Kahan */
                for (int k = 0; k < Lt; k++) {
                    const float yk = eg_exp(xr[k] - mx) - cs, tk = s + yk;
                    cs = (tk - s) - yk;
                    s = tk;
                }
                const float is = eg_inv(s), ls = mx + eg_log(s);
                for (int c = 0; c < q; c++) {
                    const int k = pn * q + c;
                    yr[c] = k >= Lt ? 0.0f : OP == EG_ROW_SOFTMAX ? eg_exp(xr[k] - mx) * is : xr[k] - ls;
                }
            } else if (OP == EG_ROW_MAX || OP == EG_ROW_MIN) {
                float v = xr[0];
                for (int k = 1; k < Lt; k++)
                    v = OP == EG_ROW_MAX ? eg_fmax(v, xr[k]) : eg_fmin(v, xr[k]);
                yr[0] = v;
            } else { /* SUM / MEAN / SUMDIV */
                float s = 0.0f, cs = 0.0f;
                for (int k = 0; k < Lt; k++) {
                    const float yk = xr[k] - cs, tk = s + yk;
                    cs = (tk - s) - yk;
                    s = tk;
                }
                yr[0] = OP == EG_ROW_MEAN ? s / (float)Lt : OP == EG_ROW_SUMDIV ? s / xr[L - 1] : s;
            }
        }
        aie::release(x);
        aie::commit(y);
    }
    aie::release(p);
}

/* ── host: one instance's entry ────────────────────────────────────────────── */

typedef void (*eager_frow_launch_fn)(int8_t *X, int8_t *P, int8_t *Y);

/* args: X [R][L] f32, P [4][4] int32 (the valid row length in every row's first word), Y [R][L] (softmax) or [R][4] */
static int eager_frow_entry(const aiek_tensor *args, int num_args, int64_t R, int64_t L, int red,
                            eager_frow_launch_fn launch) {
    if (num_args != 3)
        return AIEK_ERR_ARGS;
    const size_t ny = (size_t)(R * (red ? 16 : 4 * L));
    int8_t *x = (int8_t *)eager_stage_in(0, &args[0], (size_t)(R * 4 * L));
    int8_t *p = (int8_t *)eager_stage_in(1, &args[1], 64);
    int8_t *y = (int8_t *)eager_slot(2, ny);
    if (!x || !p || !y)
        return AIEK_ERR_ALLOC;
    launch(x, p, y);
    eager_fetch_out(2, &args[2], ny);
    return AIEK_OK;
}
