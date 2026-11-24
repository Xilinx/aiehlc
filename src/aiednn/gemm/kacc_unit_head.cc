/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// @E@.cc -- aiegraph K-in-core conv/gemm translation unit head. Kernels: gemm_kacc of @SRC@,
// one instance per layer shape (@NSHAPES@), one launch per layer, K accumulated in the core.
#define AIEDNN_KACC_KERNEL 1
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

