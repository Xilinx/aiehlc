// aiednn/conv2d/conv_s2_bf16.cc -- KxK stride-2 conv (the 3x3 s2 downsampling convs of ResNet / YOLO),
// bf16 in / out, fp32 accumulation over TCI-channel K chunks, bias + optional activation fused
// (ACT: 0 none, 1 ReLU, 2 SiLU). Implicit GEMM on aie::mmul<4,8,4,bf16,bf16,accfloat>.
// Ports: x [CI][HP][WP] (padded), w = ONNX weight [CO][CI*KH*KW], b fp32 [CO] (as bf16 [CO][2]), y [CO][OH][OW].
// The input is the zero-padded tensor [CI][HP][WP] = [CI][H + 2P][W + 2P] (aiehlc's conv convention: the IFM is
// the DDR buffer as the kernel reads it, already padded), so every block lies inside it. The fp32 bias travels
// as bf16 pairs [CO][2] (aiehlc gives all ports of a kernel one window element type and has no float window).
template <int CI, int HP, int WP, int CO, int OH, int OW, int KH, int KW, int S, int K, int TCO, int TOH, int TCI, int ACT>
__global__ void conv_s2_bf16(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<CI, HP, WP>> x_port,
                             aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<CO, K>> w_port,
                             aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<CO, 2>> b_port,
                             aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<CO, OH, OW>> y_port) {
    // TCO / TOH / TCI: a block is TCO out channels x TOH out rows; the K chunk is TCI input channels
    static_assert(K == CI * KH * KW, "w is the ONNX weight [CO][CI][KH][KW] seen as [CO][K]");
    static_assert(S == 2 && OH == (HP - KH) / S + 1 && OW == (WP - KW) / S + 1, "stride-2 conv");
    static_assert(CO % TCO == 0 && TCO % 4 == 0 && OH % TOH == 0 && CI % TCI == 0, "block must tile the layer");
    constexpr int NP = TOH * OW, NPG = (NP + 3) / 4;      // output pixels of a block, in mmul groups of 4
    constexpr int KC = TCI * KH * KW, KB = (KC + 7) / 8;  // K of one chunk, in 8-deep mmul steps (tail = 0)
    constexpr int IHT = (TOH - 1) * S + KH, IWT = (OW - 1) * S + KW; // input footprint of a block
    using MM = aie::mmul<4, 8, 4, bfloat16, bfloat16, accfloat>;  // A = W 4co x 8k, B = im2col 8k x 4px
    alignas(64) static float acc[TCO / 4][NPG][16];   // fp32 accumulators of the block, [co group][px group][4x4]
    alignas(64) static bfloat16 wp[TCO / 4][KB][32];  // W chunk in A-tile order
    alignas(64) static bfloat16 bt[KB][32];           // im2col of one px group in B-tile order
    const int pr = aie::program_id(0), nr = aie::num_programs(0);
    const int pc = aie::program_id(1), nc = aie::num_programs(1);
    for (int cb = pc; cb < CO / TCO; cb += nc)          // mesh col: output-channel blocks
        for (int rb = pr; rb < OH / TOH; rb += nr) {    // mesh row: output-row bands
            const int co0 = cb * TCO, oh0 = rb * TOH;
            for (int g = 0; g < TCO / 4; g++)
                for (int q = 0; q < NPG; q++)
                    aie::store_v(acc[g][q], aie::zeros<float, 16>());
            for (int c0 = 0; c0 < CI; c0 += TCI) {
                const bfloat16 *w = aie::load_buffer(w_port, {co0, c0 * KH * KW}, {TCO, KC});
                for (int g = 0; g < TCO / 4; g++)
                    for (int kb = 0; kb < KB; kb++) {
                        aie::vector<bfloat16, 32> a = aie::zeros<bfloat16, 32>();
                        for (int r = 0; r < 4; r++)
                            for (int kk = 0; kk < 8 && kb * 8 + kk < KC; kk++)
                                a.set(w[(g * 4 + r) * KC + kb * 8 + kk], r * 8 + kk);
                        aie::store_v(wp[g][kb], a);
                    }
                aie::release(w_port);
                const bfloat16 *x = aie::load_buffer(x_port, {c0, oh0 * S, 0}, {TCI, IHT, IWT});
                for (int q = 0; q < NPG; q++) {
                    for (int kb = 0; kb < KB; kb++) {
                        aie::vector<bfloat16, 32> b = aie::zeros<bfloat16, 32>();
                        for (int kk = 0; kk < 8; kk++) {
                            const int k = kb * 8 + kk, ci = k / (KH * KW), kh = k / KW % KH, kw = k % KW;
                            for (int j = 0; j < 4 && k < KC; j++) {
                                const int p = q * 4 + j, oh = p / OW, ow = p % OW;
                                if (p < NP)
                                    b.set(x[(ci * IHT + oh * S + kh) * IWT + ow * S + kw], kk * 4 + j);
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
                    if constexpr (ACT == 1) // fused ReLU
                        v = aie::max(v, aie::zeros<float, 16>());
                    if constexpr (ACT == 2) { // fused SiLU: v * sigmoid(v) = v / (1 + 2^(-v * log2 e))
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
