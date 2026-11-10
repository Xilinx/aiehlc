#include "simpleconv2d_new_load.h"
// Kernel-only GEMM fixture for kernel_trace_cases.py (trace -> routing, no conv).
// GEMM C[64,64] = A[64,256] x B^T[64,256] on the 4x4 mesh: M over rows, N over cols, K in 4 chunks.
__global__ void gemm_k(aie::port<input_window_int8 *, aie::IFM, aie::shape<64, 256>> a,
                       aie::port<input_window_int8 *, aie::WEIGHTS, aie::shape<64, 256>> b,
                       aie::port<output_window_int8 *, aie::OFM, aie::shape<64, 64>> c) {
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    int8_t *y = aie::store_buffer(c, {pm * 16, pn * 16}, {16, 16});
    int acc[16 * 16] = {0};
    for (int j = 0; j < 4; j++) {
        const int8_t *x = aie::load_buffer(a, {pm * 16, j * 64}, {16, 64});
        const int8_t *w = aie::load_buffer(b, {pn * 16, j * 64}, {16, 64});
        for (int m = 0; m < 16; m++)
            for (int n = 0; n < 16; n++)
                for (int k = 0; k < 64; k++)
                    acc[m * 16 + n] += x[m * 64 + k] * w[n * 64 + k];
        aie::release(a);
        aie::release(b);
    }
    for (int i = 0; i < 256; i++)
        y[i] = (int8_t)acc[i];
    aie::commit(c);
}
int main() {
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, 4, 4);
    gemm_k<<<mesh>>>(0, 0, 0);
    return 0;
}
