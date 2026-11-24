/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// @E@_ref.c -- x86 model of the carrier step for aiegraph's group-level check (offload check="group"):
// conv2d_NCHWc on TVM's layouts with the AIE numerics (activation and weight rounded to bf16, fp32
// accumulation), then the fused adds. aiegraph compares it with TVM's own C on x86: cos >= 0.999, no nonfinite.
//   args: x [1,Cc,H,W,cb], w [Kc,Cc,KH,KW,cb,kb], then 5-D adds ([1,Kc,1,1,kb] broadcast or the output
//   shape), out [1,Kc,OH,OW,kb]; stride H/OH, pad (KH-1)/2 ("same" convs of the UNet). Anything else -> PARAM.
#include <math.h>
#include <string.h>
#include "aie_user_kernel.h"

static float uc_bf16(float f) { /* round to nearest even, keep NaN/Inf */
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) != 0x7f800000u) u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    memcpy(&f, &u, 4);
    return f;
}

AIEK_REFERENCE(@E@) {
    if (num_args < 3) return AIEK_ERR_ARGS;
    const aiek_tensor *X = &args[0], *Wt = &args[1], *O = &args[num_args - 1];
    for (int i = 0; i < num_args; i++)
        if (args[i].code != AIEK_FLOAT || args[i].bits != 32) return AIEK_ERR_ARGS;
    if (X->ndim != 5 || Wt->ndim != 6 || O->ndim != 5) return AIEK_ERR_PARAM;
    const int64_t Cc = X->shape[1], H = X->shape[2], W = X->shape[3], cb = X->shape[4];
    const int64_t Kc = Wt->shape[0], KH = Wt->shape[2], KW = Wt->shape[3], kb = Wt->shape[5];
    const int64_t OH = O->shape[2], OW = O->shape[3];
    if (Wt->shape[1] != Cc || Wt->shape[4] != cb || O->shape[1] != Kc || O->shape[4] != kb) return AIEK_ERR_PARAM;
    if (OH <= 0 || OW <= 0 || H % OH || W % OW || H / OH != W / OW || !(KH & 1) || !(KW & 1)) return AIEK_ERR_PARAM;
    const int64_t S = H / OH, PH = (KH - 1) / 2, PW = (KW - 1) / 2;
    for (int i = 2; i < num_args - 1; i++) {
        const aiek_tensor *A = &args[i];
        int bc = A->ndim == 5 && A->shape[1] == Kc && A->shape[2] == 1 && A->shape[3] == 1 && A->shape[4] == kb;
        int full = A->ndim == 5 && aiek_numel(A) == aiek_numel(O);
        if (!bc && !full) return AIEK_ERR_PARAM;
    }
    const float *x = (const float *)X->data, *w = (const float *)Wt->data;
    float *out = (float *)O->data;
    for (int64_t kc = 0; kc < Kc; kc++)
        for (int64_t oh = 0; oh < OH; oh++)
            for (int64_t ow = 0; ow < OW; ow++)
                for (int64_t k = 0; k < kb; k++) {
                    float acc = 0.0f;
                    for (int64_t cc = 0; cc < Cc; cc++)
                        for (int64_t r = 0; r < KH; r++) {
                            int64_t ih = oh * S + r - PH;
                            if (ih < 0 || ih >= H) continue;
                            for (int64_t q = 0; q < KW; q++) {
                                int64_t iw = ow * S + q - PW;
                                if (iw < 0 || iw >= W) continue;
                                const float *xp = x + ((cc * H + ih) * W + iw) * cb;
                                const float *wp = w + ((((kc * Cc + cc) * KH + r) * KW + q) * cb) * kb + k;
                                for (int64_t c = 0; c < cb; c++) acc += uc_bf16(xp[c]) * uc_bf16(wp[c * kb]);
                            }
                        }
                    int64_t oi = ((kc * OH + oh) * OW + ow) * kb + k;
                    for (int i = 2; i < num_args - 1; i++) {
                        const aiek_tensor *A = &args[i];
                        const float *a = (const float *)A->data;
                        acc += aiek_numel(A) == aiek_numel(O) ? a[oi] : a[kc * kb + k];
                    }
                    out[oi] = acc;
                }
    return AIEK_OK;
}
