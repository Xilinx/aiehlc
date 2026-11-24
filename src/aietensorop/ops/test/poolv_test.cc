/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// x86 check of aie_modes_poolv.h (the pool mode's vector path) against AIE_MODE_POOL, through the AIE-API
// emulation: g++ -std=c++20 -DAIE_POOLV_EMU -I src/mlir/runtime -I src/aietensorop/ops poolv_test.cc
#include "aie_api_x86emu.h"
using int8 = int8_t;
using int32 = int32_t;
#include <cstdio>
#include <random>
#include "aie_modes_pool.h"
#include "aie_modes_poolv.h"

int main() {
    std::mt19937 rng(7);
    alignas(64) int8_t in[AIE_MODE_IN + 64];
    alignas(64) int32_t a[AIE_MODE_OUT], b[AIE_MODE_OUT];
    int taken = 0, bad = 0;
    for (int t = 0; t < 4000; t++) {
        const int op = rng() % 2 ? AIE_MODE_POOL_MAX : AIE_MODE_POOL_SUM, sgn = rng() % 2, s = 1 + rng() % 2;
        const int kh = 1 + rng() % 7, kw = 1 + rng() % 7, wo = 1 + rng() % 70;
        const int wp = (wo - 1) * s + kw + rng() % 3;
        int rout = 1 + rng() % 16;
        while (rout > 1 && (((rout - 1) * s + kh) * wp > AIE_MODE_IN || rout * wo > AIE_MODE_OUT))
            rout--;
        if (((rout - 1) * s + kh) * wp > AIE_MODE_IN || rout * wo > AIE_MODE_OUT)
            continue;
        for (int i = 0; i < AIE_MODE_IN + 64; i++)
            in[i] = (int8_t)rng();
        const int32_t g[8] = {op, sgn, s, kh, kw, wp, rout, wo};
        uint8_t d[11];
        AIE_MODE_POOL_DESC(d, g);
        AIE_MODE_POOL(in, a, d);
        if (!aie_pool_vec(in, b, d))
            continue;
        taken++;
        for (int i = 0; i < AIE_MODE_OUT; i++)
            if (a[i] != b[i]) {
                if (bad++ < 5)
                    printf("MISMATCH op %d sgn %d s %d k %dx%d wp %d rout %d wo %d at %d: %d vs %d\n", op, sgn, s, kh,
                           kw, wp, rout, wo, i, a[i], b[i]);
                break;
            }
    }
    printf("poolv: %d cases on the vector path, %d mismatching\n", taken, bad);
    return bad || taken < 1000;
}
