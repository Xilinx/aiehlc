/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * conv2dstem_uk -- the ResNet-18 stem as a USER KERNEL (include/aie_user_kernel.h).
 *
 * This is the worked example of the user-kernel interface and the subject of
 * doc/tutorial_userkernel.md. It adds NOTHING to the AIE side: the kernel, the
 * GemmSpace descriptors, the control-plane pragmas and conv2d_stem_nchwc() all
 * come from conv2dstem.cc, spliced in by aiehlc exactly as test_conv2d.cc does.
 * What this file adds is the contract the generic Python side needs:
 *
 *   AIEK_ENTRY(conv2dstem_uk)  TVM's nine arguments in -> conv2d_stem_nchwc()
 *   AIEK_SPEC(conv2dstem_uk)   which layers that fits, and the value checks
 *                              (second zero-point and `unused` all-zero) that
 *                              make the one-field epilogue exact.
 *
 * Compare with frontend/tvmrelay/aie_stem_lib.py, whose _ENTRY_TMPL hard-codes
 * the same fold in Python. Here the fold is the user's C, the packed-call
 * adapter is generated, and `python -m frontend.tvmrelay.aiekernel check`
 * proves the whole thing against TVM on x86 before any board is involved.
 *
 * Kept in this directory on purpose: aiehlc.sh copies the entry dir's *.h
 * FLAT and resolves spliced .cc includes against the entry dir, so
 * conv2dstem.h and conv2dstem.cc must sit next to it.
 ******************************************************************************/
#include "conv2dstem.cc"

// After the kernel (it is inside conv2dstem.cc above). This header declares
// nothing that reaches the launch, so the position is not load-bearing here --
// but "user headers after the kernel" is the rule that keeps every other
// header honest (skill: aiesourcetextrewrite).
#include "aie_user_kernel.h"

// The last qparams packed, so a change in values with the SAME weight pointer
// still re-packs (conv2d_stem_nchwc() caches by pointer identity only).
static conv2dstem_qparam g_uk_qp_last[CONV2DSTEM_NUM_FILTERS];

AIEK_ENTRY(conv2dstem_uk) {
    (void)num_args;                                       // the generated adapter already checked count, dtypes, shapes
    const int8_t *ifm = AIEK_PTR(&args[0], const int8_t); // [1,1,230,230,3] zp-padded
    const int8_t *wts = AIEK_PTR(&args[1], const int8_t); // [16,1,7,7,3,4] OIHW3i4o
    const int32_t *zp_a = AIEK_PTR(&args[2], const int32_t); // subtracted
    const int32_t *bias = AIEK_PTR(&args[3], const int32_t); // added
    const int32_t *zp_b = AIEK_PTR(&args[4], const int32_t); // subtracted -- must be 0
    const int32_t *mult = AIEK_PTR(&args[5], const int32_t); // Q31 multiplier
    const int32_t *unused = AIEK_PTR(&args[6], const int32_t);
    const int32_t *shift = AIEK_PTR(&args[7], const int32_t);
    uint8_t *ofm = AIEK_PTR(&args[8], uint8_t); // [1,16,112,112,4] NCHW4c

    // TVM computes (acc + bias - zp_a - zp_b); the core has ONE zero-point
    // field. Exact only while zp_b is zero -- the spec's require=zero proved it
    // for the deployed weights, and this re-check refuses rather than computing
    // wrong pixels if a different model is ever wired in. The [1,16,1,1,4] and
    // [16,4] params are both flat in plain `oc` order (oc = chunk*4 + block).
    conv2dstem_qparam qp[CONV2DSTEM_NUM_FILTERS];
    for (int oc = 0; oc < CONV2DSTEM_NUM_FILTERS; oc++) {
        if (zp_b[oc] != 0 || unused[oc] != 0) {
            printf("[conv2dstem_uk] channel %d: zp_b=%d unused=%d, want 0\n", oc, (int)zp_b[oc], (int)unused[oc]);
            return AIEK_ERR_PARAM;
        }
        qp[oc].bias = bias[oc];
        qp[oc].zero_point = zp_a[oc];
        qp[oc].multiplier = mult[oc];
        qp[oc].shift = shift[oc];
    }
    if (memcmp(qp, g_uk_qp_last, sizeof qp) != 0) {
        conv2d_stem_invalidate_weights();
        memcpy(g_uk_qp_last, qp, sizeof qp);
    }
    return conv2d_stem_nchwc(ifm, wts, qp, ofm) == CONV2DSTEM_OK ? AIEK_OK : AIEK_ERR_KERNEL;
}

// clang-format off
AIEK_SPEC(conv2dstem_uk,
    "kernel    conv2dstem\n"
    "op        conv2d_requant_relu_u8\n"
    "match     contrib_conv2d_NCHWc\n"
    "reference conv2dstem_uk_ref.c\n"
    "in    int8  [1,1,230,230,3]   ifm_pad\n"
    "param int8  [16,1,7,7,3,4]    weight\n"
    "param int32 [1,16,1,1,4]      zero_point_a\n"
    "param int32 [1,16,1,1,4]      bias\n"
    "param int32 [1,16,1,1,4]      zero_point_b  require=zero\n"
    "param int32 [16,4]            multiplier\n"
    "param int32 [16,4]            unused        require=zero\n"
    "param int32 [16,4]            shift         require=range(-30,32)\n"
    "out   uint8 [1,16,112,112,4]  ofm\n");
// clang-format on
