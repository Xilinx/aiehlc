/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
int aiednn_aie_attn_kl(int cid, int role, int cls, const float *a, int lda, float *b, int ldb,
                     float s);   /* FA group steps (aiegraph offload_group, fa_attn.fa_step_emit) */
