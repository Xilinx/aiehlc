/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * x86 reference for conv2dstem_uk (conv2dstem_uk.cc). Plain C11, host gcc.
 *
 * It models what the AIE path computes, not what TVM computes -- that is the
 * point of calibrating it against TVM: the two agree only if the model of the
 * AIE path is right. So it does what the entry + core do, in their order:
 *
 *   refuse a live second zero-point / `unused`   (the entry's fold)
 *   int32 acc over 7x7x3 in TVM's packed layouts (OIHW3i4o x NCHW3c)
 *   acc += bias; acc -= zero_point                (the core epilogue)
 *   (acc*mult + 2^(s+30)) >> (s+31), clamp [0,255] -> NCHW4c uint8
 *
 * Index arithmetic matches stage_weights_nchwc()/collect_ofm_nchwc() in
 * conv2dstem.cc and the strides TVM's generated layer 01 uses (588/84/12/4).
 ******************************************************************************/
#include "aie_user_kernel.h"

#define IHP 230 /* padded input, border = input zero-point */
#define IC 3
#define K 7
#define S 2
#define OH 112
#define OW 112
#define OCB 4  /* TVM's oc_bn */
#define OCC 16 /* 64 / OCB */

AIEK_REFERENCE(conv2dstem_uk) {
    (void)num_args;
    const int8_t *ifm = AIEK_PTR(&args[0], const int8_t);
    const int8_t *w = AIEK_PTR(&args[1], const int8_t);
    const int32_t *zp_a = AIEK_PTR(&args[2], const int32_t);
    const int32_t *bias = AIEK_PTR(&args[3], const int32_t);
    const int32_t *zp_b = AIEK_PTR(&args[4], const int32_t);
    const int32_t *mult = AIEK_PTR(&args[5], const int32_t);
    const int32_t *unused = AIEK_PTR(&args[6], const int32_t);
    const int32_t *shift = AIEK_PTR(&args[7], const int32_t);
    uint8_t *ofm = AIEK_PTR(&args[8], uint8_t);

    for (int oc = 0; oc < OCC * OCB; oc++)
        if (zp_b[oc] != 0 || unused[oc] != 0)
            return AIEK_ERR_PARAM;

    for (int occ = 0; occ < OCC; occ++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++)
                for (int ocb = 0; ocb < OCB; ocb++) {
                    const int oc = occ * OCB + ocb;
                    int32_t acc = 0;
                    for (int kh = 0; kh < K; kh++)
                        for (int kw = 0; kw < K; kw++)
                            for (int ic = 0; ic < IC; ic++) {
                                const int8_t x = ifm[((oh * S + kh) * IHP + (ow * S + kw)) * IC + ic];
                                const int8_t f = w[occ * (K * K * IC * OCB) + kh * (K * IC * OCB) + kw * (IC * OCB) +
                                                   ic * OCB + ocb];
                                acc += (int32_t)x * (int32_t)f;
                            }
                    acc += bias[oc];
                    acc -= zp_a[oc];
                    const int sh = shift[oc];
                    int64_t v = ((int64_t)acc * (int64_t)mult[oc] + ((int64_t)1 << (sh + 30))) >> (sh + 31);
                    v = v < 0 ? 0 : v > 255 ? 255 : v;
                    ofm[((occ * OH + oh) * OW + ow) * OCB + ocb] = (uint8_t)v;
                }
    return AIEK_OK;
}
