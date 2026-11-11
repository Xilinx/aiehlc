/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Kernel-only GEMM: the DMA is derived from the kernel.
 *
 *   C[64, 64] = A[64, 256] x B[64, 256]^T, int8, on the 4x4 mesh
 *
 * The ports declare only their tensors (aie::shape). The tiling lives in the
 * kernel: core (pm, pn) stores the 16x16 block C[16pm.., 16pn..] and loads A's
 * rows 16pm.. and B's rows 16pn.. in 4 K-chunks of 64. aiehlc traces the body
 * on x86 for every core, fits each port's blocks, and routes them -- with no
 * operator recognition -- into one GemmSpace per port (see aout/newfile.cpp):
 *   a: M over mesh rows, K chunks in the inner loop, broadcast along a mesh row
 *   b: N over mesh cols, K chunks in the inner loop, broadcast along a mesh col
 *   c: M over mesh rows, N over mesh cols, gathered
 * `build/aiehlc-convtile trace <this file>` prints that plan in seconds.
 *
 * The same kernel with hand-written GemmSpaces: simplegemm_kernel_space.cc.
 *
 * Status (palmyra 213, 2026-11-10): builds and passes the trace check, but
 * HANGS on the board (wait_io TIMEOUT on the output shim, cols 3 ch 0/1). The
 * pipeline gives each core's C block the tiling d0 = 16 rows in 4 rounds of 4
 * (#routing.tiling in ir/simplerouting/0_initial.mlir), while the kernel
 * commits one 16x16 block: tile_round on a mesh-split GEMM dim is applied
 * twice. Same with fullconnect_auto = 0 or 1. Open item 3 of
 * doc/design/trace_to_routing.md; not specific to either example.
 ******************************************************************************/
#include "simplegemm_load.h"
__global__ void gemm(aie::port<input_window_int8 *, aie::IFM, aie::shape<64, 256>> a,
                     aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 256>> b,
                     aie::port<output_window_int8 *, aie::OFM, aie::shape<64, 64>> c) {
    const int tm = 16, tn = 16, tk = 64; // C block per core, K chunk per round
    const int pm = aie::program_id(0);   // mesh row: which 16 rows of C
    const int pn = aie::program_id(1);   // mesh col: which 16 columns of C
    int8_t *y = aie::store_buffer(c, {pm * tm, pn * tn}, {tm, tn});
    int acc[tm * tn];
    for (int i = 0; i < tm * tn; i++)
        acc[i] = 0;
    for (int j = 0; j < 256 / tk; j++) { // K chunks
        const int8_t *x = aie::load_buffer(a, {pm * tm, j * tk}, {tm, tk});
        const int8_t *w = aie::load_buffer(b, {pn * tn, j * tk}, {tn, tk});
        for (int m = 0; m < tm; m++)
            for (int n = 0; n < tn; n++) {
                int s = 0;
                for (int k = 0; k < tk; k++)
                    s += x[m * tk + k] * w[n * tk + k];
                acc[m * tn + n] += s;
            }
        aie::release(a);
        aie::release(b);
    }
    for (int i = 0; i < tm * tn; i++)
        y[i] = (int8_t)acc[i];
    aie::commit(c);
}

int main() {
    printf("=== Kernel-only GEMM (DMA from the kernel's blocks) on AIE %dx%d ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] x B^T[%dx%d], int8\n", GEMM_M, GEMM_N, GEMM_M, GEMM_K, GEMM_K, GEMM_N);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    int8_t *A = (int8_t *)__Runtime_Alloc(GEMM_M * GEMM_K);
    int8_t *B = (int8_t *)__Runtime_Alloc(GEMM_N * GEMM_K);
    int8_t *C = (int8_t *)__Runtime_Alloc(GEMM_M * GEMM_N);
    gemm_init(A, B, C);
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm<<<mesh>>>(A, B, C);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0);
    const int mismatches = gemm_verify(A, B, C);
    free(A);
    free(B);
    free(C);
    return mismatches ? 1 : 0;
}
