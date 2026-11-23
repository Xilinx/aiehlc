/* GHL: in-place 4x4 fp32 transpose (rows r0..r3 -> columns) */
static inline void uc_tr4(float32x4_t &r0, float32x4_t &r1, float32x4_t &r2, float32x4_t &r3) {
    const float32x4x2_t t01 = vtrnq_f32(r0, r1), t23 = vtrnq_f32(r2, r3);
    /* 64-bit lane trn1/trn2 = combine(low, low) / combine(high, high), one instruction each */
    r0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0])));
    r1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1])));
    r2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[0]), vreinterpretq_f64_f32(t23.val[0])));
    r3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01.val[1]), vreinterpretq_f64_f32(t23.val[1])));
}
