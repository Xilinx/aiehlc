
#if defined(__chess__)
namespace @NS@ {
using namespace fac;
constexpr float MASKV = -1024.0f;
constexpr unsigned KV_LAST = @KVL@, NKV = @NKV@;
constexpr float TH = 32.0f, OFF = 64.0f, EXB = 895.001f;
constexpr unsigned BR = 32, BC = 32, DQ = @DQ@, DO = @DQ@, NQH = @NQH@;
constexpr unsigned NDB = DO / TM, NJB = BC / TK, NRB = BR / TN, NJQ = BC / TM, NDK = DQ / TK;
alignas(64) static bfloat16 Qt[NQH * DQ * BR];
alignas(64) static float S[BR * BC];
alignas(64) static bfloat16 Pt[BC * BR];
alignas(64) static float Ot[DO * BR];
alignas(64) static float Mx[BR], LS[BR], IL[BR];
alignas(64) static float QOFF[(BC / TM) * L], QMSK[(BC / TM) * L];
__attribute__((noinline)) static void qk(const bfloat16 *R kt, const bfloat16 *R qt, float *R st, const float *R ini, int acc) {
  constexpr unsigned GJ = 2;
  for (unsigned g = 0; g < NJQ / GJ; ++g) {
    MM mm[GJ][NRB];
    for (unsigned i = 0; i < GJ; ++i) chess_unroll_loop() {
      const AF a0 = af(aie::load_v<L>(ini + (g * GJ + i) * L));
      for (unsigned k = 0; k < NRB; ++k) chess_unroll_loop()
        mm[i][k] = acc ? MM(af(aie::load_v<L>(st + ((g * GJ + i) * NRB + k) * L))) : MM(a0);
    }
    for (unsigned db = 0; db < NDK; ++db) chess_prepare_for_pipelining chess_loop_range(NDK, NDK) {
      aie::vector<bfloat16, TM * TK> a[GJ]; aie::vector<bfloat16, TK * TN> bb[NRB];
      for (unsigned i = 0; i < GJ; ++i) chess_unroll_loop() a[i] = aie::load_v<TM * TK>(kt + ((g * GJ + i) * NDK + db) * TM * TK);
      for (unsigned k = 0; k < NRB; ++k) chess_unroll_loop() bb[k] = aie::load_v<TK * TN>(qt + (db * NRB + k) * TK * TN);
      for (unsigned i = 0; i < GJ; ++i) chess_unroll_loop() for (unsigned k = 0; k < NRB; ++k) chess_unroll_loop() mm[i][k].mac(a[i], bb[k]);
    }
    for (unsigned i = 0; i < GJ; ++i) chess_unroll_loop() for (unsigned k = 0; k < NRB; ++k) chess_unroll_loop()
      aie::store_v(st + ((g * GJ + i) * NRB + k) * L, mm[i][k].template to_vector<float>());
  }
}
__attribute__((noinline)) static void pv(const bfloat16 *R vt, const bfloat16 *R pt, float *R ot) {
  constexpr unsigned GD = 2, GR = 4;
  for (unsigned c = 0; c < (NDB / GD) * (NRB / GR); ++c) {
    const unsigned db = (c / (NRB / GR)) * GD, rb = (c % (NRB / GR)) * GR;
    MM mm[GD][GR];
    for (unsigned i = 0; i < GD; ++i) chess_unroll_loop() for (unsigned k = 0; k < GR; ++k) chess_unroll_loop()
      mm[i][k] = MM(af(aie::load_v<L>(ot + ((db + i) * NRB + rb + k) * (TM * TN))));
    for (unsigned jb = 0; jb < NJB; ++jb) chess_prepare_for_pipelining chess_loop_range(NJB, NJB) {
      aie::vector<bfloat16, TM * TK> a[GD]; aie::vector<bfloat16, TK * TN> bb[GR];
      for (unsigned i = 0; i < GD; ++i) chess_unroll_loop() a[i] = aie::load_v<TM * TK>(vt + ((db + i) * NJB + jb) * TM * TK);
      for (unsigned k = 0; k < GR; ++k) chess_unroll_loop() bb[k] = aie::load_v<TK * TN>(pt + (jb * NRB + rb + k) * TK * TN);
      for (unsigned i = 0; i < GD; ++i) chess_unroll_loop() for (unsigned k = 0; k < GR; ++k) chess_unroll_loop() mm[i][k].mac(a[i], bb[k]);
    }
    for (unsigned i = 0; i < GD; ++i) chess_unroll_loop() for (unsigned k = 0; k < GR; ++k) chess_unroll_loop()
      aie::store_v(ot + ((db + i) * NRB + rb + k) * (TM * TN), mm[i][k].template to_vector<float>());
  }
}
