
#define UC_KW_ELEMS @MAXW@ /* largest ch*K (bf16) */
#define UC_KX_ELEMS @MAXX@ /* largest px*K (bf16) */
#define UC_KY_ELEMS @MAXY@ /* largest ch*px (fp32) */
static int32_t *g_kw = nullptr, *g_kx = nullptr, *g_ky = nullptr;
#if defined(__aarch64__) && !defined(__chess__)
#include <arm_neon.h>
#define UC_NEON 1
/* 4 x fp32 -> bf16, round-to-nearest-even: bit-identical to uc_f2bf */
static inline uint16x4_t uc_f2bf4(float32x4_t v) {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1u));
    u = vaddq_u32(u, vaddq_u32(lsb, vdupq_n_u32(0x7fffu)));
    return vshrn_n_u32(u, 16);
}
