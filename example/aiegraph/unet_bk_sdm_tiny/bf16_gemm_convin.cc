/******************************************************************************
 * P2k: bf16 GEMM (fp32 accumulate) = UNet part_01 conv_in via im2col, on the
 * AIE 4x4 mesh, checked on the board against the x86 ORT part_01 output.
 * Contract (ports, dtypes, layout): bf16_gemm_convin.h.
 ******************************************************************************/
#include "bf16_gemm_convin.h"
#include "convin_data.h"
__global__ void gemm_bf16(aie::port<input_window_int32 *, aie::IFM, aie::shape<64, 64>> a,
                          aie::port<input_window_int32 *, aie::WEIGHTS, aie::shape<64, 64>> b,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<64, 64>> c) {
    // ports are int32 words: a/b word k holds bf16 elements 2k (low half) and 2k+1 (high half)
    const int tm = 16, tn = 16, tkw = 32; // C block per core, K chunk per round (32 words = 64 bf16)
    const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    int32_t *y = (int32_t *)aie::store_buffer(c, {pm * tm, pn * tn}, {tm, tn});
    // P4a: bf16 vector MAC (aie::mmul 8x8x8 bf16 x bf16 -> accfloat), fp32 result written once at the end.
    // A tiles = x rows (M x K, row-major 8x8); B tiles = w rows transposed to K x N (8x8) while packing.
#if defined(__chess__)
    using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
#endif
    alignas(64) uint16_t apk[2 * 8 * 64]; // [m block][k block][8x8 M x K]
    alignas(64) uint16_t bpk[2 * 8 * 64]; // [n block][k block][8x8 K x N]
    alignas(64) float ob[64];
    for (int j = 0; j < 64 / tkw; j++) { // chunk 1 = K padding (zeros): loaded, not computed
        const int32_t *x = (const int32_t *)aie::load_buffer(a, {pm * tm, j * tkw}, {tm, tkw});
        const int32_t *w = (const int32_t *)aie::load_buffer(b, {pn * tn, j * tkw}, {tn, tkw});
        if (j == 0) {
            const uint16_t *xs = (const uint16_t *)x, *ws = (const uint16_t *)w; // row r: 64 bf16 (2 per word)
            for (int m = 0; m < tm; m++)
                for (int k = 0; k < 2 * tkw; k++)
                    apk[((m >> 3) * 8 + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7)] = xs[m * 2 * tkw + k];
            for (int n = 0; n < tn; n++)
                for (int k = 0; k < 2 * tkw; k++)
                    bpk[((n >> 3) * 8 + (k >> 3)) * 64 + (k & 7) * 8 + (n & 7)] = ws[n * 2 * tkw + k];
#if defined(__chess__)
            for (int mi = 0; mi < 2; mi++)
                for (int ni = 0; ni < 2; ni++) {
                    const bfloat16 *at = (const bfloat16 *)(apk + mi * 8 * 64);
                    const bfloat16 *bt = (const bfloat16 *)(bpk + ni * 8 * 64);
                    MM blk;
                    blk.mul(aie::load_v<64>(at), aie::load_v<64>(bt));
                    for (int kb = 1; kb < 8; kb++)
                        blk.mac(aie::load_v<64>(at + kb * 64), aie::load_v<64>(bt + kb * 64));
                    aie::store_v(ob, blk.to_vector<float>());
                    for (int r = 0; r < 8; r++)
                        for (int cc = 0; cc < 8; cc++) {
                            union {
                                uint32_t u;
                                float f;
                            } o;
                            o.f = ob[r * 8 + cc];
                            y[(mi * 8 + r) * tn + ni * 8 + cc] = (int32_t)o.u;
                        }
                }
#else // x86 kernel-analysis trace (aie_api_x86emu.h has no bfloat16/accfloat): same result, scalar
            for (int m = 0; m < tm; m++)
                for (int n = 0; n < tn; n++) {
                    float s = 0.0f;
                    for (int k = 0; k < 2 * tkw; k++) {
                        union {
                            uint32_t u;
                            float f;
                        } xa, wb;
                        xa.u = (uint32_t)xs[m * 2 * tkw + k] << 16;
                        wb.u = (uint32_t)ws[n * 2 * tkw + k] << 16;
                        s += xa.f * wb.f;
                    }
                    union {
                        uint32_t u;
                        float f;
                    } o;
                    o.f = s;
                    y[m * tn + n] = (int32_t)o.u;
                }
#endif
        }
        aie::release(a);
        aie::release(b);
    }
    aie::commit(c);
}

int main() {
    printf("=== P2k bf16 GEMM conv_in M=%d K=%d(%d real) N=%d on AIE %dx%d ===\n", CI_M, CI_K, CI_KREAL, CI_N, HW_ROWS,
           HW_COLS);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);
    const uint32_t in_crc = ci_crc32(convin_sample, sizeof(convin_sample));
    uint16_t *Afull = (uint16_t *)malloc(CI_M * CI_K * 2);
    uint16_t *Bfull = (uint16_t *)malloc(CI_N * CI_K * 2);
    float *out = (float *)malloc(CI_M * CI_N * 4);
    ci_im2col(convin_sample, Afull);
    ci_weights(convin_w, convin_b, Bfull);
    int32_t *A = (int32_t *)__Runtime_Alloc(CI_BM * CI_KPORT * 2);
    int32_t *B = (int32_t *)__Runtime_Alloc(CI_BN * CI_KPORT * 2);
    memset(A, 0, CI_BM * CI_KPORT * 2);
    memset(B, 0, CI_BN * CI_KPORT * 2);
    int32_t *C = (int32_t *)__Runtime_Alloc(CI_BM * CI_BN * 4);
    int launches = 0;
    XTime t0, t1;
    XTime_GetTime(&t0);
    for (int mb = 0; mb < CI_M / CI_BM; mb++)
        for (int nb = 0; nb < CI_N / CI_BN; nb++) {
            for (int r = 0; r < CI_BM; r++)
                memcpy((uint16_t *)A + r * CI_KPORT, Afull + (mb * CI_BM + r) * CI_K, CI_K * 2);
            for (int r = 0; r < CI_BN; r++)
                memcpy((uint16_t *)B + r * CI_KPORT, Bfull + (nb * CI_BN + r) * CI_K, CI_K * 2);
            memset(C, 0xff, CI_BM * CI_BN * 4); /* sentinel: an unwritten element shows up as nonfinite */
            Xil_DCacheFlushRange((UINTPTR)A, CI_BM * CI_KPORT * 2);
            Xil_DCacheFlushRange((UINTPTR)B, CI_BN * CI_KPORT * 2);
            Xil_DCacheFlushRange((UINTPTR)C, CI_BM * CI_BN * 4);
            gemm_bf16<<<mesh>>>(A, B, C);
            Xil_DCacheInvalidateRange((UINTPTR)C, CI_BM * CI_BN * 4);
            launches++;
            for (int m = 0; m < CI_BM; m++)
                for (int n = 0; n < CI_BN; n++) {
                    float f;
                    memcpy(&f, &C[m * CI_BN + n], 4);
                    out[(nb * CI_BN + n) * CI_M + mb * CI_BM + m] = f; /* NCHW */
                }
        }
    XTime_GetTime(&t1);
    printf("[BENCH] aie_wall_ms=%.3f launches=%d\n", 1.0 * (t1 - t0) / COUNTS_PER_SECOND * 1000.0, launches);
    double dot = 0, na = 0, nr = 0, maxabs = 0, maxabs_cpu = 0;
    int nonfinite = 0;
    for (int i = 0; i < CI_M * CI_N; i++) {
        double o = out[i], r = convin_ref[i];
        if (!(o == o) || o > 1e30 || o < -1e30)
            nonfinite++;
        dot += o * r;
        na += o * o;
        nr += r * r;
        if (fabs(o - r) > maxabs)
            maxabs = fabs(o - r);
    }
    /* same bf16 GEMM on the APU, to tell AIE errors from bf16 rounding */
    for (int m = 0; m < CI_M; m += 7)
        for (int n = 0; n < CI_N; n++) {
            float s = 0;
            for (int k = 0; k < CI_K; k++)
                s += ci_bf2f(Afull[m * CI_K + k]) * ci_bf2f(Bfull[n * CI_K + k]);
            double d = fabs((double)s - out[n * CI_M + m]);
            if (d > maxabs_cpu)
                maxabs_cpu = d;
        }
    const double cs = dot / (sqrt(na) * sqrt(nr) + 1e-30);
    char crc_s[16];
    snprintf(crc_s, sizeof(crc_s), "%08x", (unsigned)in_crc);
    const int pass = nonfinite == 0 && cs >= 0.999 && strcmp(crc_s, CONVIN_SAMPLE_CRC32) == 0;
    printf("[P2K] out[0..3]=%.6f %.6f %.6f %.6f ref=%.6f %.6f %.6f %.6f maxabs_vs_ort=%.6g maxabs_vs_apu_bf16=%.6g "
           "nonfinite=%d\n",
           out[0], out[1], out[2], out[3], convin_ref[0], convin_ref[1], convin_ref[2], convin_ref[3], maxabs,
           maxabs_cpu, nonfinite);
    printf("[P2K] aie_launches=%d embedded_crc32=%s\n", launches, CONVIN_SAMPLE_CRC32);
    printf("KCMP conv_in-bf16 data=%s in_crc32=%s cos=%.6f %s\n", CONVIN_DATA, crc_s, cs, pass ? "PASS" : "FAIL");
    free(Afull);
    free(Bfull);
    free(out);
    return pass ? 0 : 1;
}
