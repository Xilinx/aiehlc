/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* aiednn elementwise/silu_scale_dense.cc -- second driver step of a "dense+silu" group: the TVM dense
 * call with its input (argument 0) replaced by x*s from elementwise/silu_scale_x.cc.
 *
 * params: FN = the dense's TVM symbol, NARGS = its packed argument count.
 */
//@@ step
args[0].v_handle = &g_silu_x_t;
@FN@(args, codes, @NARGS@, 0, 0, 0);
