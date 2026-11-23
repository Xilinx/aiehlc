// aiednn/gemm/fc_gemm_bf16.cc -- fully connected layer at batch 1 (ONNX Gemm transB=1, e.g. the ResNet
// classifier 512 -> 1000): y[n] = sum_k x[k] w[n][k] + b[n], bf16 x / w, fp32 accumulation and fp32 out (logits).
// The core keeps x resident and streams NB-row blocks of w; per row, 16-lane aie::mac (accfloat) over K and
// one reduce_add. Ports: x [1][K], w = ONNX B [N][K], b fp32 [N], y fp32 [N]. aiehlc gives all ports of a kernel one window
// element type and has no float window, so the fp32 tensors travel as bf16: b as pairs [N][2], y as rows of
// 8 fp32 [NY = N / 8][16] (an output window [N][2] is mis-sized by aiehlc).
template <int K, int N, int NB, int NY>
__global__ void fc_gemm_bf16(aie::port<input_window_bfloat16 *, aie::IFM, aie::shape<1, K>> x_port,
                             aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<N, K>> w_port,
                             aie::port<input_window_bfloat16 *, aie::WEIGHTS, aie::shape<N, 2>> b_port,
                             aie::port<output_window_bfloat16 *, aie::OFM, aie::shape<NY, 16>> y_port) {
    // NB output rows per weight block
    static_assert(K % 16 == 0 && N % NB == 0, "K in 16-lane steps, N in NB-row blocks");
    static_assert(NY * 8 == N && NB % 8 == 0, "y is [N / 8][8 fp32] = bf16 [NY][16]; a block is whole rows of it");
    // y is 1-D (N) and aiehlc splits a tensor dim by one mesh axis only: the NB-row blocks are dealt over mesh
    // rows, so launch on an R x 1 mesh (aiehlc builds it on 1 x 1 today: on R > 1 rows it splits the broadcast x).
    const int pid = aie::program_id(0);
    const int np = aie::num_programs(0);
    const bfloat16 *x = aie::load_buffer(x_port, {0, 0}, {1, K}, {.resident = true});
    for (int blk = pid; blk < N / NB; blk += np) {
        const bfloat16 *w = aie::load_buffer(w_port, {blk * NB, 0}, {NB, K});
        const float *bias = (const float *)aie::load_buffer(b_port, {blk * NB, 0}, {NB, 2});
        float *y = (float *)aie::store_buffer(y_port, {blk * NB / 8, 0}, {NB / 8, 16});
        for (int r = 0; r < NB; r++) {
            aie::accum<accfloat, 16> a = aie::zeros<accfloat, 16>();
            for (int k = 0; k < K; k += 16)
                a = aie::mac(a, aie::load_unaligned_v<16>(x + k, 8), aie::load_unaligned_v<16>(w + r * K + k, 8));
            y[r] = aie::reduce_add(a.template to_vector<float>()) + bias[r];
        }
        aie::commit(y_port);
        aie::release(b_port);
        aie::release(w_port);
    }
    aie::release(x_port);
}
