__global__ void @K@(aie::port<input_window_int16 *, aie::IFM, aie::shape<4, @QKROW@>> qk_@K@,
                    aie::port<input_window_int16 *, aie::WEIGHTS, aie::shape<4, @VROW@>> v_@K@,
                    aie::port<output_window_int16 *, aie::OFM, aie::shape<4, @OROW@>> o_@K@) {
    // GFA class H=@H@ N=@N@ M=@M@ D=@D@ (fa_attn.py) v5, gemm_kacc routing topology: q = row broadcast (row pm holds
    // the 4 heads' blocks), kv = column broadcast (column pn = one head's K/V stream), o = gather; core = @PNDOC@
    const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    for (int j = 0; j < @NQB@; j++) {
        const int16_t *qb = (const int16_t *)aie::load_buffer(qk_@K@, {pm, j * @E@}, {1, @E@});
        if (@QSEL@)
            @NS@::setq(qb, @QHALF@);
        aie::release(qk_@K@);
    }
    @NS@::reset();
    for (int b = 0; b < @NKV@; b++) {
@BODY@
    }
    for (int j = 0; j < 4; j++) {
        if (j == pn) {
            int16_t *ob = (int16_t *)aie::store_buffer(o_@K@, {pm, j * @E@}, {1, @E@});
            @NS@::fin(ob);
            aie::commit(o_@K@);
        }
    }
}
