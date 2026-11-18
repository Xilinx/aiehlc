__global__ void @NAME@(aie::port<input_window_int32 *, aie::IFM, aie::shape<@AROWS@, @KW@>> a_@SID@,
                          aie::port<input_window_int32 *, aie::WEIGHTS, aie::shape<@BROWS@, @KW@>> b_@SID@,
                          aie::port<output_window_int32 *, aie::OFM, aie::shape<@AROWS@, @BROWS@>> c_@SID@) {
    // ports named per instance: aiehlc's block descriptors (aiehlc_desc_<port>) are file-scope statics
    // P4bg: gemm_kacc (bf16_gemm_kacc.cc, Maxwell P4bk) for ch=@CH@ K=@K@ px=@PX@ -- see unet_aie.kacc_geom
    const int tm = 16, tn = 16, tkw = @TKW@, kch = @KCH@; // tile; K chunk = tkw words = kch bf16
    const int nkc = @NKC@;   // K chunks per tile
    const int ntile = @NT@;  // a-port 16-row tiles per core (channels; pixels when transposed)
    const int ns = @S@;      // 16-row sub-tiles of the core's b block (pixels; channels when transposed)
    // transposed=@TR@: the mmul's left operand is always W (pre-tiled [mi][kb][8x8 ch x k]), right is X
    const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    for (int t = 0; t < ntile; t++) {
        const int m0 = pm * (ntile * tm) + t * tm;
        // P4bx: the c window {16, ns*16} fp32 is the accumulator: ns 16x16 tiles in [s][mi][ni][8x8] order
        int32_t *y = (int32_t *)aie::store_buffer(c_@SID@, {m0, pn * (ns * tn)}, {tm, ns * tn});
        float *const kacc = (float *)y;
        for (int i = 0; i < ns * 256; i++)
            y[i] = 0;
        for (int j = 0; j < nkc; j++) {
            const int32_t *x = (const int32_t *)aie::load_buffer(a_@SID@, {m0, j * tkw}, {tm, tkw});
            const int32_t *w = (const int32_t *)aie::load_buffer(b_@SID@, {pn * (ns * tn), j * tkw}, {ns * tn, tkw});
            for (int s = 0; s < ns; s++) {
                float *ac = kacc + s * 256;
#if defined(__chess__)
                using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
                aie::accum<accfloat, 64> q00, q01, q10, q11;
                q00.from_vector(aie::load_v<64>(ac));
                q01.from_vector(aie::load_v<64>(ac + 64));
                q10.from_vector(aie::load_v<64>(ac + 128));
                q11.from_vector(aie::load_v<64>(ac + 192));
                MM c00(q00), c01(q01), c10(q10), c11(q11);
                const bfloat16 *a0 = (const bfloat16 *)(@LHS@), *a1 = a0 + 8 * kch;
                const bfloat16 *b0 = (const bfloat16 *)(@RHS@), *b1 = b0 + 8 * kch;
                for (int kb = 0; kb < kch / 8; kb++) {
                    auto va0 = aie::load_v<64>(a0 + kb * 64);
                    auto va1 = aie::load_v<64>(a1 + kb * 64);
                    auto vb0 = aie::load_v<64>(b0 + kb * 64);
                    auto vb1 = aie::load_v<64>(b1 + kb * 64);
                    c00.mac(va0, vb0);
                    c01.mac(va0, vb1);
                    c10.mac(va1, vb0);
                    c11.mac(va1, vb1);
                }
                aie::store_v(ac, c00.to_vector<float>());
                aie::store_v(ac + 64, c01.to_vector<float>());
                aie::store_v(ac + 128, c10.to_vector<float>());
                aie::store_v(ac + 192, c11.to_vector<float>());
#else // scalar, same pre-tiled layout (chunk = [mi 2][kb kch/8][8x8])
                const uint16_t *xs = (const uint16_t *)(@LHS@), *ws = (const uint16_t *)(@RHS@);
                for (int m = 0; m < tm; m++)
                    for (int n = 0; n < tn; n++) {
                        float sum = 0.0f;
                        for (int k = 0; k < kch; k++) {
                            union {
                                uint32_t u;
                                float f;
                            } xa, wb;
                            xa.u = (uint32_t)xs[((m >> 3) * (kch / 8) + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7)] << 16;
                            wb.u = (uint32_t)ws[((n >> 3) * (kch / 8) + (k >> 3)) * 64 + (k & 7) * 8 + (n & 7)] << 16;
                            sum += xa.f * wb.f;
                        }
                        ac[((m >> 3) * 2 + (n >> 3)) * 64 + (m & 7) * 8 + (n & 7)] += sum;
                    }
#endif
            }
            aie::release(a_@SID@);
            aie::release(b_@SID@);
        }
        aie::commit(c_@SID@); // one fp32 write-back of the core's 16 x px/4 block after the last chunk
    }
}
