/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// x86 stub for emu_conv2d_spatial.cc: the example header times its CPU reference.
#pragma once
#include <ctime>
typedef unsigned long long XTime;
#define COUNTS_PER_SECOND 1000000000ULL
static inline void XTime_GetTime(XTime *t) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *t = (XTime)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
