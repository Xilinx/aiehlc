/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
{
    double aie_ms_p = 0.0;
    aiednn_aie_launch_stats(&aie_ms_p, 0, 0);
    aiednn_prof_print(1000.0 * (double)(t1 - t0) / (double)COUNTS_PER_SECOND, aie_ms_p);
    fflush(stdout);
    }
