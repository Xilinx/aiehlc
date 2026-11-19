#include "simpleops_load.h"
// Row op: y[r][c] = x[r][c] - max over c' of x[r][c'], x and y f32[128, 256] carried
// as bytes. Every core needs WHOLE rows (the max) but writes only a quarter of the
// columns -- the shape of layer norm / softmax / RoPE rows:
//   mesh row pm owns rows 32 pm ..; each 8-row strip of them (whole rows) is
//     broadcast to the row's 4 cores;
//   core (pm, pn) writes columns 64 pn .. 64 pn + 63 of those 8 rows, an
//     {8, 256 B} block at {32 pm + 8 i, 256 pn} (bytes).
// The output blocks of one core stack straight down (step = block height), so the
// gather sends them block by block into the core's own 32 x 256 B panel.
#define OPS_R 128          // rows
#define OPS_L 256          // f32 per row
#define OPS_LB (OPS_L * 4) // bytes per row
__global__ void rowshift(aie::port<input_window_int8 *, aie::shape<OPS_R, OPS_LB>> x,
                         aie::port<output_window_int8 *, aie::shape<OPS_R, OPS_LB>> y) {
    const int tr = 8;                               // rows per strip
    const int rows = OPS_R / aie::num_programs(0);  // 32 rows per mesh row
    const int cols = OPS_LB / aie::num_programs(1); // 256 B = 64 f32 per core
    const int pm = aie::program_id(0), pn = aie::program_id(1);
    for (int i = 0; i < rows / tr; i++) { // 4 strips
        const int8_t *a = aie::load_buffer(x, {pm * rows + i * tr, 0}, {tr, OPS_LB});
        int8_t *o = aie::store_buffer(y, {pm * rows + i * tr, pn * cols}, {tr, cols});
        for (int r = 0; r < tr; r++) {
            const float *xr = (const float *)(a + r * OPS_LB);
            float m = xr[0];
            for (int c = 1; c < OPS_L; c++)
                if (xr[c] > m)
                    m = xr[c];
            const float *xs = xr + pn * (cols / 4);
            float *yr = (float *)(o + r * cols);
            for (int c = 0; c < cols / 4; c++)
                yr[c] = xs[c] - m;
        }
        aie::release(x);
        aie::commit(y);
    }
}
int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition(OPS_PARTITION, 4, 4);
    const long n = (long)OPS_R * OPS_L;
    float *x = (float *)__Runtime_Alloc(n * 4), *y = (float *)__Runtime_Alloc(n * 4);
    float *ref = (float *)malloc(n * 4);
    for (long i = 0; i < n; i++) // exact in f32: multiples of 1/16 in [-62.5, 62.5]
        x[i] = (float)((int)(ops_mix32((uint32_t)i + 7) % 2001) - 1000) / 16.0f;
    memset(y, 0, n * 4);
    uint64_t t0 = ops_now(), t1;
    rowshift<<<mesh>>>((int8_t *)x, (int8_t *)y);
    t1 = ops_now();
    printf("[BENCH] aie_wall_ms=%.3f\n", ops_ms(t0, t1));
    for (int r = 0; r < OPS_R; r++) {
        float m = x[(long)r * OPS_L];
        for (int c = 1; c < OPS_L; c++)
            if (x[(long)r * OPS_L + c] > m)
                m = x[(long)r * OPS_L + c];
        for (int c = 0; c < OPS_L; c++)
            ref[(long)r * OPS_L + c] = x[(long)r * OPS_L + c] - m;
    }
    return ops_compare("y", (const int8_t *)y, (const int8_t *)ref, OPS_R, OPS_LB) ? 1 : 0;
}
