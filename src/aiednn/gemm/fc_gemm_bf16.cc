// aiednn/gemm/fc_gemm_bf16.cc -- fully connected layer at batch 1 (ONNX Gemm transB=1, e.g. the ResNet
// classifier 512 -> 1000): y[n] = sum_k x[k] w[n][k] + b[n], bf16 x / w, fp32 accumulation and fp32 out (logits).
// The core keeps x resident and streams NB-row blocks of w; per row, 16-lane aie::mac (accfloat) over K and
// one reduce_add. Ports: x [K], w = ONNX B [N][K], b fp32 [N], y fp32 [N].
__global__ void @NAME@(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<@K@>> x_@NAME@,
                       aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<@N@, @K@>> w_@NAME@,
                       aie::port<input_window_float *, aie::WEIGHTS, aie::shape<@N@>> b_@NAME@,
                       aie::port<output_window_float *, aie::OFM, aie::shape<@N@>> y_@NAME@) {
    constexpr int K = @K@, N = @N@, NB = @NB@; // NB output rows per weight block
    static_assert(K % 16 == 0 && N % NB == 0, "K in 16-lane steps, N in NB-row blocks");
    const int pid = aie::program_id(0) * aie::num_programs(1) + aie::program_id(1);
    const int np = aie::num_programs(0) * aie::num_programs(1);
    const bfloat16 *x = aie::load_buffer(x_@NAME@, {0}, {K}, {.resident = true});
    for (int blk = pid; blk < N / NB; blk += np) {
        const bfloat16 *w = aie::load_buffer(w_@NAME@, {blk * NB, 0}, {NB, K});
        const float *bias = aie::load_buffer(b_@NAME@, {blk * NB}, {NB});
        float *y = aie::store_buffer(y_@NAME@, {blk * NB}, {NB});
        for (int r = 0; r < NB; r++) {
            aie::accum<accfloat, 16> a = aie::zeros<accfloat, 16>();
            for (int k = 0; k < K; k += 16)
                a = aie::mac(a, aie::load_unaligned_v<16>(x + k, 8), aie::load_unaligned_v<16>(w + r * K + k, 8));
            y[r] = aie::reduce_add(a.template to_vector<float>()) + bias[r];
        }
        aie::commit(y_@NAME@);
        aie::release(b_@NAME@);
        aie::release(w_@NAME@);
    }
    aie::release(x_@NAME@);
}
