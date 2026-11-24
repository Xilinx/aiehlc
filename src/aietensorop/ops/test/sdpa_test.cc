/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// x86 check of aie_modes_sdpa.h: the core's vector item (aie_sdpa_item_vec) == the scalar reference
// (aie_sdpa_item_ref) on random items, through the AIE-API emulation.
#include "aie_api_x86emu.h"
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
#include <cstdio>
#include <random>
#include "aie_modes_sdpa.h"

int main() {
    std::mt19937 rng(5);
    alignas(64) int8_t in[4096];
    alignas(64) int32_t a[1024], b[1024], s1[256], s2[256];
    int done = 0, bad = 0;
    for (int t = 0; t < 3000; t++) {
        const int dq = 1 + rng() % 40, dv = 1 + rng() % 40, skv = 1 + rng() % 200, causal = rng() % 2;
        const int skp = (skv + 15) & ~15;
        const int kv = dq * skp + skv * dv;
        if (kv + dq > AIE_SDPA_COFF)
            continue;
        int sq = 1 + rng() % 64;
        while (sq > 1 && (sq * dq + kv > AIE_SDPA_COFF || sq * dv > 1024))
            sq--;
        if (sq * dq + kv > AIE_SDPA_COFF || sq * dv > 1024)
            continue;
        for (int i = 0; i < 4096; i++)
            in[i] = (int8_t)(rng() % 255 - 127);
        for (int e = 0; e < dq; e++) /* K^T's padding keys are zero */
            for (int j = skv; j < skp; j++)
                in[sq * dq + e * skp + j] = 0;
        const int32_t coff = (int32_t)(rng() % (skv + sq)) - sq;
        for (int k = 0; k < 4; k++)
            in[AIE_SDPA_COFF + k] = (int8_t)((uint32_t)coff >> (8 * k));
        const int32_t g[8] = {
            sq, skv, dq, dv, causal, (int32_t)(1 << 30) + (int32_t)(rng() % 1000000), (int32_t)(30 + rng() % 12), 0};
        uint8_t d[16];
        AIE_MODE_SDPA_DESC(d, g);
        aie_sdpa_item_ref(in, a, d, s1);
        aie_sdpa_item_vec(in, b, d, s2);
        done++;
        for (int i = 0; i < 1024; i++)
            if (a[i] != b[i]) {
                if (bad++ < 5)
                    printf("MISMATCH sq %d skv %d d %d dv %d causal %d at %d: %d vs %d\n", sq, skv, dq, dv, causal, i,
                           a[i], b[i]);
                break;
            }
    }
    printf("sdpa: %d items, %d mismatching\n", done, bad);
    return bad || done < 500;
}
