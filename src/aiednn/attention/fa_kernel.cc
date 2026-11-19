// fa_kernel: GFA flash attention (fused QK^T -> online softmax -> PV), one C++ template global kernel, instantiated by
// aiehlc per launch fa_kernel<NS, HEADS, NQ, MK, DH, SPLIT, E, NKV, QKROW, VROW, OROW> (fa_attn.fa_launch):
//   NS              the class's helper namespace (fa_attn.fa_prologue: setq / reset / qkb / softb / pvb / fin; aiehlc
//                   instantiates on the source text, so a namespace name is a valid typename argument)
//   HEADS/NQ/MK/DH  attention class H heads, N queries, M keys, head dim D
//   SPLIT           D > 80: every 32xD block is split in two 80-wide halves (D = 160, N = 64), else 0
//   E               elements of one (half) block = 32 * dq; NKV key blocks of 32
//   QKROW/VROW/OROW port row lengths (q blocks / per-head kv stream / o blocks)
// Was fa_kernel.cc + fa_body_full.cc / fa_body_split.cc filled by text substitution (da30c4d); the two bodies
// are the if constexpr (SPLIT) branches below, every instance is unchanged.
template <typename NS, int HEADS, int NQ, int MK, int DH, int SPLIT, int E, int NKV, int QKROW, int VROW, int OROW>
__global__ void fa_kernel(aie::port<input_window_int16 *, aie::IFM, aie::shape<4, QKROW>> qk_port,
                          aie::port<input_window_int16 *, aie::WEIGHTS, aie::shape<4, VROW>> v_port,
                          aie::port<output_window_int16 *, aie::OFM, aie::shape<4, OROW>> o_port) {
    if constexpr (SPLIT) {// GFA class H=HEADS N=NQ M=MK D=DH (fa_attn.py) v5, gemm_kacc routing topology: q = row broadcast (row pm holds
    // the 4 heads' blocks), kv = column broadcast (column pn = one head's K/V stream), o = gather; core = head pn, query block pm>>1, output half pm&1
    } else {// GFA class H=HEADS N=NQ M=MK D=DH (fa_attn.py) v5, gemm_kacc routing topology: q = row broadcast (row pm holds
    // the 4 heads' blocks), kv = column broadcast (column pn = one head's K/V stream), o = gather; core = head pn, query block pm
    }const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    if constexpr (SPLIT) {for (int j = 0; j < 8; j++) {
        const int16_t *qb = (const int16_t *)aie::load_buffer(qk_port, {pm, j * E}, {1, E});
        if ((j >> 1) == pn)
            NS::setq(qb, j & 1);
        aie::release(qk_port);
    }} else {for (int j = 0; j < 4; j++) {
        const int16_t *qb = (const int16_t *)aie::load_buffer(qk_port, {pm, j * E}, {1, E});
        if (j == pn)
            NS::setq(qb, 0);
        aie::release(qk_port);
    }}
    NS::reset();
    for (int b = 0; b < NKV; b++) {
        if constexpr (SPLIT) {const int16_t *k0 = (const int16_t *)aie::load_buffer(v_port, {pn, (4 * b) * E}, {1, E});
        NS::qkb(k0, b, 0);
        aie::release(v_port);
        const int16_t *k1 = (const int16_t *)aie::load_buffer(v_port, {pn, (4 * b + 1) * E}, {1, E});
        NS::qkb(k1, b, 1);
        aie::release(v_port);
        NS::softb(b);
        const int16_t *v0 = (const int16_t *)aie::load_buffer(v_port, {pn, (4 * b + 2) * E}, {1, E});
        if ((pm & 1) == 0)
            NS::pvb(v0);
        aie::release(v_port);
        const int16_t *v1 = (const int16_t *)aie::load_buffer(v_port, {pn, (4 * b + 3) * E}, {1, E});
        if ((pm & 1) == 1)
            NS::pvb(v1);
        aie::release(v_port);} else {const int16_t *kb = (const int16_t *)aie::load_buffer(v_port, {pn, (2 * b) * E}, {1, E});
        NS::qkb(kb, b, 0);
        aie::release(v_port);
        NS::softb(b);
        const int16_t *vb = (const int16_t *)aie::load_buffer(v_port, {pn, (2 * b + 1) * E}, {1, E});
        NS::pvb(vb);
        aie::release(v_port);}
    }
    for (int j = 0; j < 4; j++) {
        if (j == pn) {
            int16_t *ob = (int16_t *)aie::store_buffer(o_port, {pm, j * E}, {1, E});
            NS::fin(ob);
            aie::commit(o_port);
        }
    }
}
