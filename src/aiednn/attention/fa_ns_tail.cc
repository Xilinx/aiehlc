/* copy one 32-row Q block (half h of a split block) into Qt */
__attribute__((noinline)) static void setq(const void *qv, int h) {
  const bfloat16 *q = (const bfloat16 *)qv;
  bfloat16 *dst = Qt + h * DQ * BR;
  for (unsigned i = 0; i < DQ * BR / L; ++i) chess_prepare_for_pipelining aie::store_v(dst + i * L, aie::load_v<L>(q + i * L));
}
/* reset the online-softmax state (statics survive between launches) and build the acc-init tables */
__attribute__((noinline)) static void reset() {
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned i = 0; i < DO * BR / L; ++i) chess_prepare_for_pipelining aie::store_v(Ot + i * L, aie::zeros<float, L>());
  aie::store_v(Mx, aie::zeros<float, L>()); aie::store_v(LS, aie::zeros<float, L>());
  for (unsigned jq = 0; jq < BC / TM; ++jq) for (unsigned ln = 0; ln < L; ++ln) {
    QOFF[jq * L + ln] = OFF;
    QMSK[jq * L + ln] = (jq * TM + ln / TN < KV_LAST) ? OFF : MASKV;
  }
}
/* key block b: S = Q K^T (pass 0 from the OFF/mask table, pass 1 of a split block accumulates onto S) */
__attribute__((noinline)) static void qkb(const void *kv, int b, int pass) {
  qk((const bfloat16 *)kv, Qt + pass * DQ * BR, S, b == (int)NKV - 1 ? QMSK : QOFF, pass);
}
__attribute__((noinline)) static void softb(int b) { soft(S, Pt, Ot, Mx, LS, b == 0); }
__attribute__((noinline)) static void pvb(const void *vv) { pv((const bfloat16 *)vv, Pt, Ot); }
__attribute__((noinline)) static void fin(void *ov) { fin1(Ot, LS, IL, (bfloat16 *)ov); }
} // namespace @NS@
#else
namespace @NS@ {
static inline void setq(const void *, int) {}
static inline void reset() {}
static inline void qkb(const void *, int, int) {}
static inline void softb(int) {}
static inline void pvb(const void *) {}
static inline void fin(void *) {}
} // namespace @NS@
#endif
