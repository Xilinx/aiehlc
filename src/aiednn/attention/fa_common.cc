
// AIEHLC_KERNEL_PROLOGUE_BEGIN
// GFA: flash-attention helpers (Maxwell M4 fa15 body); every symbol is static so a kernel's core program keeps
// only the class namespace it calls (gemm_kacc kernels keep none).
#if defined(__chess__)
namespace fac {
constexpr float MAG = 12582912.0f;
constexpr unsigned L = 32, TM = 4, TK = 8, TN = 8;
using VB = aie::vector<bfloat16, L>;
using VF = aie::vector<float, L>;
using VI = aie::vector<int32, L>;
using AF = aie::accum<accfloat, L>;
using MM = aie::mmul<TM, TK, TN, bfloat16, bfloat16>;
#define R __restrict
#define AI __attribute__((always_inline)) static inline
AI AF af(VF v) { AF t; t.from_vector(v); return t; }
AI aie::accum<accfloat, 16> af16(aie::vector<float, 16> v) { aie::accum<accfloat, 16> t; t.from_vector(v); return t; }
AI VI rnd_i(VF x) { return aie::sub(aie::add(af(x), af(aie::broadcast<float, L>(MAG))).template to_vector<float>().template cast_to<int32>(),
                                    aie::broadcast<int32, L>(0x4B400000)); }
AI VI pw2(VI x, VI kq) {
  const VI ne = aie::add(aie::bit_and(x, aie::broadcast<int32, L>(0x7f800000)), kq);
  return aie::select(aie::add(x, kq), aie::zeros<int32, L>(), aie::le(ne, aie::zeros<int32, L>()));
}
AI aie::vector<int32, 8> red_max(VI x) {
  const auto a = aie::max(x.template extract<16>(0), x.template extract<16>(1));
  return aie::max(a, aie::shuffle_down(a, 8)).template extract<8>(0);
}
AI aie::vector<int32, 8> red_min(VI x) {
  const auto a = aie::min(x.template extract<16>(0), x.template extract<16>(1));
  return aie::min(a, aie::shuffle_down(a, 8)).template extract<8>(0);
}
AI aie::vector<float, 8> red_sum(VF x) {
  const auto a = aie::add(af16(x.template extract<16>(0)), af16(x.template extract<16>(1))).template to_vector<float>();
  return aie::add(af16(a), af16(aie::shuffle_down(a, 8))).template to_vector<float>().template extract<8>(0);
}
} // namespace fac
#endif
