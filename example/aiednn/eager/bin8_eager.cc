/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * aiednn eager mode, binary of two 8-bit tensors on the board: bin8<R, L, TR, OP, Q> for the exact int32 s8 + u8
 * (OP add, Q 0) and s8 x s8 (mul), and the quantized s8 + s8 with a ReLU in f32 (Q 1), in one ELF. Q 0 is checked
 * bit-exact; Q 1 bit-exact against the same float arithmetic done here, the two products rounded on their own
 * (volatile: the A72 compiler would contract one into a fused multiply-add, which the core does not do -- the sum
 * then differs by an ulp of the terms where they cancel).
 ******************************************************************************/
#define HW_ROWS 4 /* the mesh (eager_common.h); the kernel text needs it */
#define HW_COLS 4
#include "bin8_eager_kernel.cc"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void add_run(int8_t *XY, int8_t *P, int8_t *Z) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    bin8<128, 256, 8, 64, 0><<<mesh>>>(XY, P, Z);
}

static void mul_run(int8_t *XY, int8_t *P, int8_t *Z) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    bin8<64, 128, 4, 66, 0><<<mesh>>>(XY, P, Z);
}

static void qadd_run(int8_t *XY, int8_t *P, int8_t *Z) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(EG_PARTITION, EG_HW_ROWS, EG_HW_COLS);
    bin8<128, 256, 8, 64, 1><<<mesh>>>(XY, P, Z);
}

static uint32_t b8_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/* OP add (64) / mul (66), Q 0 or 1; sg the signedness bits; Q 1: scales / zero points, ReLU */
static long b8_case(const char *name, int OP, int Q, int sg, long R, long L,
                    void (*run)(int8_t *, int8_t *, int8_t *)) {
    const long n = R * L;
    int8_t *xy = (int8_t *)calloc(2 * n, 1);
    uint32_t *want = (uint32_t *)calloc(n, 4), *got = (uint32_t *)calloc(n, 4);
    for (long i = 0; i < 2 * n; i++)
        xy[i] = (int8_t)(b8_mix((uint32_t)i + 5u) & 0xff);
    float pw[32] = {0};
    const float sa = Q ? 0.0213f : 1.0f, sb = Q ? 0.0371f : 1.0f, nza = Q ? -3.0f : 0.0f, nzb = Q ? 7.0f : 0.0f;
    for (int c = 0; c < 4; c++) {
        float *w = pw + 8 * c;
        w[0] = sa, w[1] = nza, w[2] = sb, w[3] = nzb, w[4] = Q ? 0.0f : -3.4e38f, w[5] = 3.4e38f;
        memcpy(w + 6, &sg, 4);
    }
    for (long r = 0; r < R; r++)
        for (long c = 0; c < L; c++) {
            const int8_t ac = xy[r * 2 * L + c], bc = xy[r * 2 * L + L + c];
            const int a = (sg & 1) ? ac : (uint8_t)ac, b = (sg & 2) ? bc : (uint8_t)bc;
            if (Q) {
                volatile float ta = ((float)a + nza) * sa, tb = ((float)b + nzb) * sb;
                const float v = ta + tb, w = v > 0.0f ? v : 0.0f;
                memcpy(&want[r * L + c], &w, 4);
            } else {
                const int32_t w = OP == 64 ? a + b : a * b;
                memcpy(&want[r * L + c], &w, 4);
            }
        }
    int8_t *XYd = (int8_t *)__Runtime_Alloc(2 * n), *Pd = (int8_t *)__Runtime_Alloc(128),
           *Zd = (int8_t *)__Runtime_Alloc(n * 4);
    memcpy(XYd, xy, 2 * n);
    memcpy(Pd, pw, 128);
    memset(Zd, 0x55, n * 4);
    Xil_DCacheFlushRange((UINTPTR)XYd, 2 * n);
    Xil_DCacheFlushRange((UINTPTR)Pd, 128);
    Xil_DCacheFlushRange((UINTPTR)Zd, n * 4);
    run(XYd, Pd, Zd);
    Xil_DCacheInvalidateRange((UINTPTR)Zd, n * 4);
    memcpy(got, Zd, n * 4);
    long bad = 0, first = -1;
    for (long i = 0; i < n; i++)
        if (got[i] != want[i] && bad++ == 0)
            first = i;
    printf("[%s] check: %ld / %ld bit-exact", name, n - bad, n);
    if (first >= 0)
        printf(" (first at %ld: got 0x%08x, want 0x%08x)", first, (unsigned)got[first], (unsigned)want[first]);
    printf("\n");
    return bad;
}

int main() {
    printf("=== aiednn eager bin8<R, L, TR, OP, Q>: binary of two 8-bit tensors on the core ===\n");
    long bad = b8_case("bin8<128,256,8,ADD,0> s8 + u8 -> int32", 64, 0, 1, 128, 256, add_run);
    bad += b8_case("bin8<64,128,4,MUL,0> s8 x s8 -> int32", 66, 0, 3, 64, 128, mul_run);
    bad += b8_case("bin8<128,256,8,ADD,1> quantized s8 + s8, relu -> f32", 64, 1, 3, 128, 256, qadd_run);
    printf("%s\n", bad ? "FAIL:" : "PASS:");
    return bad != 0;
}
