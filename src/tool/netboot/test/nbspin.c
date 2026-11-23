/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/* netboot test app: prints, busy-waits SPIN_S seconds, prints, returns.
 * Runs on the app core like any BSP app (its _exit gets patched to park), so
 * it exercises the supervisor's busy / max-run-time / force paths with a known
 * run time. Build: src/tool/netboot/test/build.sh */
#include "xil_printf.h"
#include "xiltimer.h"

#ifndef SPIN_S
#define SPIN_S 20
#endif

int main(void) {
    XTime t0, t;
    XTime_GetTime(&t0);
    xil_printf("[nbspin] start: busy for %d s\r\n", SPIN_S);
    for (int s = 1; s <= SPIN_S; s++) {
        do
            XTime_GetTime(&t);
        while (t - t0 < (XTime)s * COUNTS_PER_SECOND);
        if (s % 5 == 0)
            xil_printf("[nbspin] %d s\r\n", s);
    }
    xil_printf("[nbspin] done after %d s\r\n", SPIN_S);
    return 0;
}
