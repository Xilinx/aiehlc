/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * conv3x3_requant -- a REFERENCE-ONLY user kernel: step 1 of writing a new one.
 *
 * No AIE code yet. This file holds the spec and the x86 reference, which is
 * everything `python -m frontend.tvmrelay.aiekernel check` needs to answer the
 * questions worth answering BEFORE any AIE work starts:
 *
 *   detect     which layers of the deployed ResNet-18 this op covers -- one
 *              spec with symbolic dims (CIC, COC, HI, ...) matches every plain
 *              3x3 int8 -> int8 conv, at any width and either stride;
 *   calibrate  does my model of the op agree with TVM bit for bit, per call,
 *              including the requant rounding and the -128 output shift;
 *   emulate    does the whole network still produce the same logits with my
 *              model standing in for all of those layers at once.
 *
 * Step 2 is an aiehlc source with an AIE kernel and AIEK_ENTRY that matches
 * THIS function (the conv2dstem_uk.cc pattern), with the spec moved there and
 * a `reference conv3x3_requant_ref.c` line pointing back here. Its geometry
 * must then be concrete: aiehlc bakes one shape per build.
 *
 * The op, as TVM's fused NCHWc kernel computes it (layers/04_*.c epilogue):
 *   acc = sum x[icc,oh*s+kh,ow*s+kw,ici] * w[occ,icc,kh,kw,ici,ocb]      int32
 *   v   = (int32)(((int64)(acc + bias - zp_a - zp_b) * mult + 2^(sh+30)) >> (sh+31))
 *   out = (int8)(clamp(v, 0, 255) - 128)
 * Layouts: x NCHW4c (already padded by the preceding nn.pad layer), w
 * OIHW4i4o, out NCHW4c.
 ******************************************************************************/
#include "aie_user_kernel.h"

AIEK_REFERENCE(conv3x3_requant) {
    (void)num_args;
    const aiek_tensor *x = &args[0], *w = &args[1], *o = &args[8];
    const int cic = (int)x->shape[1], hi = (int)x->shape[2], wi = (int)x->shape[3];
    const int coc = (int)o->shape[1], ho = (int)o->shape[2], wo = (int)o->shape[3];
    const int8_t *xd = AIEK_PTR(x, const int8_t), *wd = AIEK_PTR(w, const int8_t);
    const int32_t *zp_a = AIEK_PTR(&args[2], const int32_t);
    const int32_t *bias = AIEK_PTR(&args[3], const int32_t);
    const int32_t *zp_b = AIEK_PTR(&args[4], const int32_t);
    const int32_t *mult = AIEK_PTR(&args[5], const int32_t);
    const int32_t *shift = AIEK_PTR(&args[7], const int32_t);
    int8_t *od = AIEK_PTR(o, int8_t);

    // The stride is not in the signature; recover it from the extents, the way
    // aie_offload._layer_geometry does (padding is a separate layer upstream).
    int s = 0;
    for (int c = 1; c <= 4 && !s; c++)
        if ((hi - 3) / c + 1 == ho && (wi - 3) / c + 1 == wo)
            s = c;
    if (!s)
        return AIEK_ERR_ARGS;

    for (int occ = 0; occ < coc; occ++)
        for (int oh = 0; oh < ho; oh++)
            for (int ow = 0; ow < wo; ow++)
                for (int ocb = 0; ocb < 4; ocb++) {
                    int32_t acc = 0;
                    for (int icc = 0; icc < cic; icc++)
                        for (int kh = 0; kh < 3; kh++)
                            for (int kw = 0; kw < 3; kw++) {
                                const int8_t *xp = &xd[((icc * hi + oh * s + kh) * wi + ow * s + kw) * 4];
                                const int8_t *wp = &wd[(((occ * cic + icc) * 3 + kh) * 3 + kw) * 16 + ocb];
                                for (int ici = 0; ici < 4; ici++)
                                    acc += (int32_t)xp[ici] * (int32_t)wp[ici * 4];
                            }
                    const int oc = occ * 4 + ocb, sh = shift[oc];
                    const int32_t t = acc + bias[oc] - zp_a[oc] - zp_b[oc];
                    int32_t v = (int32_t)(((int64_t)t * (int64_t)mult[oc] + ((int64_t)1 << (sh + 30))) >> (sh + 31));
                    v = v < 255 ? v : 255;
                    v = v > 0 ? v : 0;
                    od[((occ * ho + oh) * wo + ow) * 4 + ocb] = (int8_t)(v - 128);
                }
    return AIEK_OK;
}

// clang-format off
AIEK_SPEC(conv3x3_requant,
    "kernel    conv3x3_requant\n"
    "op        conv2d_3x3_requant_relu_s8\n"
    "match     contrib_conv2d_NCHWc\n"
    "in    int8  [1,CIC,HI,WI,4]    ifm_pad\n"
    "param int8  [COC,CIC,3,3,4,4]  weight\n"
    "param int32 [1,COC,1,1,4]      zero_point_a\n"
    "param int32 [1,COC,1,1,4]      bias\n"
    "param int32 [1,COC,1,1,4]      zero_point_b\n"
    "param int32 [COC,4]            multiplier\n"
    "param int32 [COC,4]            unused\n"
    "param int32 [COC,4]            shift   require=range(-30,32)\n"
    "out   int8  [1,COC,HO,WO,4]    ofm\n");
// clang-format on
