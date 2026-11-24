/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
int aiednn_aie_conv(int cid, void *args, int nargs, const uint16_t *wbf, int sh, int sw, int ph, int pw);   /* aiegraph.attnsite.patch_driver: --aie conv */
int aiednn_aie_linear(int cid, void *args, int nargs, const uint16_t *wbf, int S);   /* --aie conv,linear */
