#include "simpleops_load.h"
// STATUS (2026-11-15): passes the kernel analysis (`build/aiehlc-convtile trace` this file);
// the lowering does not support this op shape yet, so aiehlc.sh stops with
// "the core window is ... but the kernel's blocks (#routing.kernel_block) are ...".
#define OPS_R 256  // rows
#define OPS_C 1024 // columns (bytes)
// Elementwise: z = saturate_int8(x + y), x / y / z int8[256, 1024], on a 4x1 mesh.
// Core pm owns rows 64 pm .. 64 pm + 63 and walks them in 16 x 256 blocks.
// The 4x4 version (redundant broadcast) is simpleeltwise_add.cc; this one is for
// comparison: one core per mesh row. Tutorial: 02_elementwise/elementwise.md
__global__ void eltwise_add(aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_C>> x,
                            aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_C>> y,
                            aie::port<output_window_int8 *, aie::shape<OPS_R, OPS_C>> z) {
    const int tr = 16, tc = 256;                       // block: 16 rows x 256 columns (4 KB)
    const int rows = OPS_R / aie::num_programs(0);    // 64 rows per core
    const int pm = aie::program_id(0);
    for (int i = 0; i < rows / tr; i++)                // 4 row strips
        for (int j = 0; j < OPS_C / tc; j++) {         // 4 column chunks
            const int8_t *a = aie::load_buffer(x, {pm * rows + i * tr, j * tc}, {tr, tc});
            const int8_t *b = aie::load_buffer(y, {pm * rows + i * tr, j * tc}, {tr, tc});
            int8_t *o = aie::store_buffer(z, {pm * rows + i * tr, j * tc}, {tr, tc});
            for (int e = 0; e < tr * tc; e++) {
                const int v = a[e] + b[e];
                o[e] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
            }
            aie::release(x);
            aie::release(y);
            aie::commit(z);
        }
}
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 1); // 4 mesh rows, 1 mesh column
    const long n = (long)OPS_R * OPS_C;
    int8_t *x = (int8_t *)__Runtime_Alloc(n), *y = (int8_t *)__Runtime_Alloc(n);
    int8_t *z = (int8_t *)__Runtime_Alloc(n), *ref = (int8_t *)malloc(n);
    ops_fill(x, n, 11);
    ops_fill(y, n, 22);
    memset(z, 0, n);
    XTime t0, t1;
    XTime_GetTime(&t0);
    eltwise_add<<<mesh>>>(x, y, z);
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f\n", ops_ms(t0, t1));
    for (long e = 0; e < n; e++)
        ref[e] = ops_sat8(x[e] + y[e]);
    return ops_compare("z", z, ref, OPS_R, OPS_C) ? 1 : 0;
}
