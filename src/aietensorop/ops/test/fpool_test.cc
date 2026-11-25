/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
// x86 check of aie_modes_fpool.h: the core's vector item (16 outputs per vector, stride 1 / 2, KD slices)
// through the AIE-API emulation == the plain C reference, max bit for bit, sum within f32 reassociation
#include "aie_api_x86emu.h"
#include <cmath>
#include <cstdio>
#include <random>
#define AIE_FPOOL_EMU 1
#include "aie_modes_fpool.h"

int main() {
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u(-5.f, 5.f);
    int bad = 0, cases = 0;
    for (int op = 0; op < 3; op++)
        for (int sw = 1; sw <= 2; sw++)
            for (int kd = 1; kd <= 3; kd++)
                for (int k = 1; k <= 3; k++)
                    for (int wo : {5, 16, 21, 33}) {
                        const int sh = sw, wp = (wo - 1) * sw + k;
                        int rout = 1024 / wo;
                        while (rout > 1 && kd * ((rout - 1) * sh + k) * wp > 1024 - 32)
                            rout--;
                        if (kd * ((rout - 1) * sh + k) * wp > 1024 - 32)
                            continue;
                        alignas(64) float in[1024 + 64], o1[1024], o2[1024];
                        for (auto &x : in)
                            x = op == 2 ? (float)(int)(u(rng)) : u(rng); /* integers: ties to test "first" */
                        uint8_t d[12];
                        const int32_t g[8] = {op, kd, k, k, sh | (sw << 8), wo, wp, rout};
                        AIE_FPOOL_DESC(d, g);
                        aie_fpool_item_vec(in, o1, d);
                        aie_fpool_item_ref(in, o2, d);
                        cases++;
                        for (int i = 0; i < 1024; i++) {
                            const float tol = op == 1 ? 1e-5f * (float)(kd * k * k) * 5.f : 0.f;
                            const bool diff =
                                op == 2 ? ((int32_t *)o1)[i] != ((int32_t *)o2)[i] : std::fabs(o1[i] - o2[i]) > tol;
                            if (diff) {
                                if (bad++ < 5)
                                    printf("MISMATCH op %d sw %d kd %d k %d wo %d at %d: %g vs %g\n", op, sw, kd, k, wo,
                                           i, o1[i], o2[i]);
                            }
                        }
                    }
    printf("fpool: %d cases, %d mismatching\n", cases, bad);
    return bad != 0;
}
