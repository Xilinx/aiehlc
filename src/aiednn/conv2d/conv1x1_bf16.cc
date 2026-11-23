// aiednn/conv2d/conv1x1_bf16.cc -- 1x1 conv (pointwise; stride 1, or 2 for the ResNet downsample shortcut) as a
// GEMM over pixels: y[co][p] = sum_ci w[co][ci] x[ci][p] + b[co], bf16 in / out, fp32 accumulation on
// aie::mmul<4,8,4,bf16,bf16,accfloat>, bias + optional activation fused (ACT: 0 none, 1 ReLU, 2 SiLU).
// The core gathers each B tile (8 channels x 4 pixels) from its input block, for stride 1 and 2 alike (the old
// stride-1 4-pixel vector load is an aie::vector<bfloat16, 4>, which the AIE API does not have).
// Ports: x [CI][H][W], w = ONNX weight [CO][CI], b fp32 [CO] (as bf16 [CO][2]), y [CO][OH][OW].
template <int CI, int H, int W, int CO, int OH, int OW, int S, int TCO, int TOH, int TCI, int ACT>
__global__ void conv1x1_bf16(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<CI, H, W>> x_port,
                             aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<CO, CI>> w_port,
                             aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<CO, 2>> b_port,
                             aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<CO, OH, OW>> y_port) {
    // TCO / TOH / TCI: a block is TCO out channels x TOH out rows; the K chunk is TCI input channels
    static_assert(OH == (H - 1) / S + 1 && OW == (W - 1) / S + 1, "1x1 conv, no padding");
    static_assert(CO % TCO == 0 && TCO % 4 == 0 && OH % TOH == 0 && CI % TCI == 0 && TCI % 8 == 0, "block tiling");
    constexpr int NP = TOH * OW, NPG = (NP + 3) / 4, KB = TCI / 8;
    constexpr int IHT = (TOH - 1) * S + 1, IWT = (OW - 1) * S + 1; // input footprint of a block
    using MM = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>;
    alignas(64) static float acc[TCO / 4][NPG][16];
    alignas(64) static bfloat16 wp[TCO / 4][KB][32];
    alignas(64) static bfloat16 bt[KB][32];
    const int pr = aie::program_id(0), nr = aie::num_programs(0);
    const int pc = aie::program_id(1), nc = aie::num_programs(1);
    for (int cb = pc; cb < CO / TCO; cb += nc)
        for (int rb = pr; rb < OH / TOH; rb += nr) {
            const int co0 = cb * TCO, oh0 = rb * TOH;
            for (int g = 0; g < TCO / 4; g++)
                for (int q = 0; q < NPG; q++)
                    aie::store_v(acc[g][q], aie::zeros<float, 16>());
            for (int c0 = 0; c0 < CI; c0 += TCI) {
                const bfloat16 *w = aie::load_buffer(w_port, {co0, c0}, {TCO, TCI});
                for (int g = 0; g < TCO / 4; g++)
                    for (int kb = 0; kb < KB; kb++) {
                        aie::vector<bfloat16, 32> a;
                        for (int r = 0; r < 4; r++)
                            a.insert(r, aie::load_unaligned_v<8>(w + (g * 4 + r) * TCI + kb * 8));
                        aie::store_v(wp[g][kb], a);
                    }
                aie::release(w_port);
                const bfloat16 *x = aie::load_buffer(x_port, {c0, oh0 * S, 0}, {TCI, IHT, IWT});
                for (int q = 0; q < NPG; q++) {
                    for (int kb = 0; kb < KB; kb++) {
                        aie::vector<bfloat16, 32> b = aie::zeros<bfloat16, 32>();
                        for (int kk = 0; kk < 8; kk++) {
                            const bfloat16 *xr = x + (kb * 8 + kk) * IHT * IWT;
                            for (int j = 0; j < 4; j++) {
                                const int p = q * 4 + j;
                                if (p < NP)
                                    b.set(xr[(p / OW) * S * IWT + (p % OW) * S], kk * 4 + j);
                            }
                        }
                        aie::store_v(bt[kb], b);
                    }
                    for (int g = 0; g < TCO / 4; g++) {
                        aie::accum<accfloat, 16> c;
                        c.from_vector(aie::load_v<16>(acc[g][q]));
                        MM m(c);
                        for (int kb = 0; kb < KB; kb++)
                            m.mac(aie::load_v<32>(wp[g][kb]), aie::load_v<32>(bt[kb]));
                        aie::store_v(acc[g][q], m.template to_vector<float>());
                    }
                }
                aie::release(x_port);
            }
            const float *bias = (const float *)aie::load_buffer(b_port, {co0, 0}, {TCO, 2});
            bfloat16 *y = aie::store_buffer(y_port, {co0, oh0, 0}, {TCO, TOH, OW});
            for (int g = 0; g < TCO / 4; g++)
                for (int q = 0; q < NPG; q++) {
                    aie::vector<float, 16> bv;
                    for (int i = 0; i < 16; i++)
                        bv.set(bias[g * 4 + i / 4], i);
                    aie::vector<float, 16> v = aie::add(aie::load_v<16>(acc[g][q]), bv);
                    if constexpr (ACT == 1)
                        v = aie::max(v, aie::zeros<float, 16>());
                    if constexpr (ACT == 2) {
                        const aie::vector<float, 16> e = aie::exp2<float>(aie::mul(v, -1.4426950408889634f).template to_vector<float>());
                        v = aie::mul(v, aie::inv(aie::add(e, aie::broadcast<float, 16>(1.0f)))).template to_vector<float>();
                    }
                    aie::accum<accfloat, 16> o;
                    o.from_vector(v);
                    const aie::vector<bfloat16, 16> ob = o.template to_vector<bfloat16>();
                    for (int i = 0; i < 16; i++)
                        if (q * 4 + i % 4 < NP)
                            y[(g * 4 + i / 4) * NP + q * 4 + i % 4] = ob[i];
                }
            aie::release(b_port);
            aie::commit(y_port);
        }
}
