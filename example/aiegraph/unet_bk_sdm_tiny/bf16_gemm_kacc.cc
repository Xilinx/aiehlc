/******************************************************************************
 * P4bk: bf16 GEMM with K accumulated in the core (fp32), ONE launch per layer.
 * Contract: $U/p4/K_CONTRACT.md (16x16 fp32 acc per core, kc = 64, 4x4 mesh,
 * one write-back per output tile). Layer: UNet c013 conv 3x3 1280<-2560 on 8x8,
 * im2col X[64 px][K = 23040], W[1280 ch][K]. The kernel computes the transposed
 * GEMM  Y[ch][px] = W[ch, :] . X[px, :]   (= NCHW output, no transpose on the host):
 *   "M" = 1280 channels (mesh rows, 320 per row, walked on the core in 20 tiles of 16)
 *   "N" = 64 pixels     (mesh cols, 16 per col)
 *   "K" = 23040         (360 chunks of 64, accumulated in fp32 in the core)
 * so the tile walk is aiehlc's on-core M-round GEMM path (tile_m < tile_rows,
 * input rounds = m_rounds x k_rounds, output rounds = m_rounds).
 *
 * Ports (int32 words, 2 bf16 per word, low half first), laid out by kacc_bench.sh:
 *   a  IFM      [1280][11520]  W rows (row broadcast)
 *   b  WEIGHTS  [64][11520]    X rows (col broadcast; re-sent in every m-round)
 *               (needs aiehlc with the P4bk M-round fix: B's window was divided by m_rounds too)
 *   c  OFM      [1280][64]     fp32 Y, one 16x16 block per tile, written once after its last K chunk
 * Every 16-row x 64-bf16 chunk is pre-tiled by the host, so the core does no repacking:
 *   a chunk = [mi 2][kb 8][8x8 M x K row-major],  b chunk = [ni 2][kb 8][8x8 K x N row-major]
 * Per tile: 4 aie::mmul<8,8,8,bf16,bf16,accfloat> accumulators over 360 chunks x 8 kb.
 * aie::mmul only under __chess__; the x86 kernel-analysis trace gets the same
 * math in scalar (aie_api_x86emu.h has no bfloat16/accfloat, cfe529d).
 ******************************************************************************/
#define UNET_KACC_KERNEL 1
#include "kacc_data.h"

__global__ void gemm_kacc(aie::port<input_window_int32 *, aie::IFM, aie::shape<1280, 11520>> a,
                          aie::port<input_window_int32 *, aie::WEIGHTS, aie::shape<64, 11520>> b,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<1280, 64>> c) {
    const int tm = 16, tn = 16, tkw = 32; // tile; K chunk = 32 words = 64 bf16
    const int nkc = 360;                  // K chunks per tile (23040 / 64)
    const int ntile = 20;                 // tiles per core (1280 / 4 mesh rows / 16)
    const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * tm) + t * tm;
#if defined(__chess__)
        using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
        MM c00(aie::zeros<accfloat, 64>()), c01(aie::zeros<accfloat, 64>());
        MM c10(aie::zeros<accfloat, 64>()), c11(aie::zeros<accfloat, 64>());
#else
        float acc[tm * tn];
        for (int i = 0; i < tm * tn; i++)
            acc[i] = 0.0f;
#endif
        for (int j = 0; j < nkc; j++) {
            const int32_t *x = (const int32_t *)aie::load_buffer(a, {m0, j * tkw}, {tm, tkw});
            const int32_t *w = (const int32_t *)aie::load_buffer(b, {pn * tn, j * tkw}, {tn, tkw});
#if defined(__chess__)
            const bfloat16 *a0 = (const bfloat16 *)x, *a1 = a0 + 8 * 64;
            const bfloat16 *b0 = (const bfloat16 *)w, *b1 = b0 + 8 * 64;
            for (int kb = 0; kb < 8; kb++) {
                auto va0 = aie::load_v<64>(a0 + kb * 64);
                auto va1 = aie::load_v<64>(a1 + kb * 64);
                auto vb0 = aie::load_v<64>(b0 + kb * 64);
                auto vb1 = aie::load_v<64>(b1 + kb * 64);
                c00.mac(va0, vb0);
                c01.mac(va0, vb1);
                c10.mac(va1, vb0);
                c11.mac(va1, vb1);
            }
#else // scalar, same pre-tiled layout
            const uint16_t *xs = (const uint16_t *)x, *ws = (const uint16_t *)w;
            for (int m = 0; m < tm; m++)
                for (int n = 0; n < tn; n++) {
                    float s = 0.0f;
                    for (int k = 0; k < 64; k++) {
                        union {
                            uint32_t u;
                            float f;
                        } xa, wb;
                        xa.u = (uint32_t)xs[((m >> 3) * 8 + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7)] << 16;
                        wb.u = (uint32_t)ws[((n >> 3) * 8 + (k >> 3)) * 64 + (k & 7) * 8 + (n & 7)] << 16;
                        s += xa.f * wb.f;
                    }
                    acc[m * tn + n] += s;
                }
#endif
            aie::release(a);
            aie::release(b);
        }
        // tile done: its one fp32 write-back
        int32_t *y = (int32_t *)aie::store_buffer(c, {m0, pn * tn}, {tm, tn});
#if defined(__chess__)
        alignas(64) float ob[64];
#define KACC_STORE_BLK(BLK, MI, NI)                                                                                   \
    aie::store_v(ob, BLK.to_vector<float>());                                                                          \
    for (int rr = 0; rr < 8; rr++)                                                                                     \
        for (int cc = 0; cc < 8; cc++) {                                                                               \
            union {                                                                                                    \
                uint32_t u;                                                                                            \
                float f;                                                                                               \
            } o;                                                                                                       \
            o.f = ob[rr * 8 + cc];                                                                                     \
            y[((MI) * 8 + rr) * tn + (NI) * 8 + cc] = (int32_t)o.u;                                                    \
        }
        KACC_STORE_BLK(c00, 0, 0)
        KACC_STORE_BLK(c01, 0, 1)
        KACC_STORE_BLK(c10, 1, 0)
        KACC_STORE_BLK(c11, 1, 1)
#undef KACC_STORE_BLK
#else
        for (int i = 0; i < tm * tn; i++) {
            union {
                uint32_t u;
                float f;
            } o;
            o.f = acc[i];
            y[i] = (int32_t)o.u;
        }
#endif
        aie::commit(c);
    }
}

int main() {
    printf("=== P4bk K-in-core bf16 GEMM c013 M=%d K=%d N=%d on AIE %dx%d, one launch ===\n", KACC_M, KACC_K, KACC_N,
           HW_ROWS, HW_COLS);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    /* KACC_M = 64 pixels, KACC_N = 1280 channels (the conv's GEMM); the kernel runs it transposed */
    const size_t wbytes = (size_t)KACC_N * KACC_K * 2, xbytes = (size_t)KACC_M * KACC_K * 2;
    const size_t ybytes = (size_t)KACC_N * KACC_M * 4;
    int32_t *Wd = (int32_t *)__Runtime_Alloc(wbytes);
    int32_t *Xd = (int32_t *)__Runtime_Alloc(xbytes);
    int32_t *Yd = (int32_t *)__Runtime_Alloc(ybytes);
    if (!Wd || !Xd || !Yd) {
        printf("KCMP %s data=%s y_sha256=%s cos=0.000000 FAIL (alloc)\n", KACC_TAG, KACC_DATA, KACC_Y_SHA);
        return 1;
    }
    memcpy(Wd, kacc_w_bin, wbytes);
    memcpy(Xd, kacc_x_bin, xbytes);
    memset(Yd, 0xff, ybytes); /* sentinel: an unwritten element shows up as nonfinite */
    Xil_DCacheFlushRange((UINTPTR)Wd, wbytes);
    Xil_DCacheFlushRange((UINTPTR)Xd, xbytes);
    Xil_DCacheFlushRange((UINTPTR)Yd, ybytes);
    int launches = 0;
    XTime t0, t1;
    XTime_GetTime(&t0);
    gemm_kacc<<<mesh>>>(Wd, Xd, Yd);
    XTime_GetTime(&t1);
    launches++;
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ybytes);
    printf("[BENCH] aie_wall_ms=%.3f launches=%d\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0, launches);
    const float *bias = (const float *)kacc_bias_bin;
    const float *ref = (const float *)kacc_y_bin;
    double dot = 0, na = 0, nr = 0, maxabs = 0;
    int nonfinite = 0;
    for (int n = 0; n < KACC_N; n++)
        for (int m = 0; m < KACC_M; m++) {
            float f;
            memcpy(&f, &Yd[n * KACC_M + m], 4);
            const double o = (double)f + bias[n], r = ref[n * KACC_M + m]; /* NCHW */
            if (!(o == o) || o > 1e30 || o < -1e30) {
                nonfinite++;
                continue;
            }
            dot += o * r;
            na += o * o;
            nr += r * r;
            if (fabs(o - r) > maxabs)
                maxabs = fabs(o - r);
        }
    const double cs = nonfinite ? 0.0 : dot / (sqrt(na) * sqrt(nr) + 1e-30);
    const int pass = nonfinite == 0 && cs >= 0.999;
    printf("[P4BK] maxabs_vs_ref=%.6g nonfinite=%d\n", maxabs, nonfinite);
    printf("KCMP %s data=%s y_sha256=%s cos=%.6f %s\n", KACC_TAG, KACC_DATA, KACC_Y_SHA, cs, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
