"""GFA (sprint unet_m3-s1 P2): fused flash-attention sites for build_unet.py --aie conv,linear,attn --attn-fa.

One aiegraph.partition.match_attention group (QK^T bmm -> softmax -> PV bmm) becomes ONE online-softmax attention
on the AIE: the QK site packs Q (x log2 e, the 1/sqrt(d) scale is already in TVM's reshape_transpose_multiply) and
the centred K (PKC: keys centred over the real keys per head, softmax is shift-invariant per query row), the TVM
softmax call is removed from graph_driver.c, the PV site packs V, launches the class kernel and writes O (f32).

Kernel body = Maxwell's board-proven M4 flash attention (tasks/2026-11-12_m4-fa-aiehlc fa15 / fa_cross_d40.cc:
QK^T mmul<4,8,8> with OFF / key-mask in the acc init, int32 max + exact-min fallback, exp2_hw, lazy power-of-2
rescale, PV mmul, bf16 hi/lo 1/l finalize), Br = Bc = 32, 4x4 mesh:
  D = 40 / 80   : mesh row pm = one unit (head h, 128 query rows), core pn = query block pn of the unit.
                  qk port row = [4 Q blocks | NKV K blocks], v port row = [NKV V blocks], o port row = [4 O blocks].
  D = 160 (N=64): a 32x160 block does not fit core DM with 3 ping-pong ports, so every block is split in two
                  80-wide halves (5 KB): unit = head (64 rows), core pn = query block pn>>1, output half pn&1.
                  qk row = [Q0h0 Q0h1 Q1h0 Q1h1 | K_b h0, K_b h1 ...], v row = [V_b h0, V_b h1 ...],
                  o row = [O0h0 O0h1 O1h0 O1h1]; S = Q_h0 K_h0^T + Q_h1 K_h1^T (second pass starts from S).
K/V are replicated per mesh row by the host (the KO aiehlc has no row broadcast for a size-1 port dim, 0ee6397).
Block layouts (Maxwell fa_load.h): Q/O qidx(d,r) = ((d/8)*4 + r/8)*64 + (d%8)*8 + r%8,
K kidx(j,d) = (((j/4)*(D/8) + d/8)*4 + j%4)*8 + d%8, V vidx(d,j) = (((d/4)*4 + j/8)*4 + d%4)*8 + j%8."""

import math

E_BR = 32


def fa_classes(att: list) -> list:
    """attn groups -> FA classes (H, N, M, D); every group gets c['fa'] = class index on both of its sites"""
    groups = {}
    for c in att:
        groups.setdefault(c["group"], {})[c["role"]] = c
    classes = []
    for g, d in sorted(groups.items()):
        assert set(d) == {"qk", "pv"}, (g, set(d))
        qk, pv = d["qk"], d["pv"]
        H = int(qk["xshape"][0])
        N, D = qk["xshape"][1], qk["xshape"][2]       # Q [H][N][D]
        M = qk["wshape"][1]                            # K [H][M][D]
        assert pv["wshape"] == [H, D, M] and pv["yshape"] == [H, N, D], (g, pv["wshape"], pv["yshape"])
        key = (H, N, M, D)
        if key not in classes:
            classes.append(key)
        for c in (qk, pv):
            c["fa"] = classes.index(key)
    out = []
    for i, (H, N, M, D) in enumerate(classes):
        split = D > 80
        dq = D // 2 if split else D
        assert dq % 8 == 0 and dq <= 80, D
        nkv = -(-M // 32)
        kvl = M - 32 * (nkv - 1)
        E = 32 * dq
        if split:
            assert N == 64, N
            units, upr = H, 1
            qkrow, vrow, orow = 4 * E + 2 * nkv * E, 2 * nkv * E, 4 * E
        else:
            assert N % 128 == 0, N
            units, upr = H * N // 128, N // 128
            qkrow, vrow, orow = 4 * E + nkv * E, nkv * E, 4 * E
        assert units % 4 == 0, units
        out.append(dict(idx=i, H=H, N=N, M=M, D=D, split=int(split), dq=dq, nkv=nkv, kvl=kvl, E=E, units=units,
                        upr=upr, launches=units // 4, qkrow=qkrow, vrow=vrow, orow=orow,
                        name=f"fa_h{H}_n{N}_m{M}_d{D}"))
    return out


FA_COMMON = r"""
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
"""

FA_NS = r"""
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
/* online softmax of one 32-key block (Maxwell blk() without its final pv) */
__attribute__((noinline)) static void soft(const float *R st, bfloat16 *R pt, float *R ot, float *R m, float *R l, bool first) {
  const VF mo = aie::load_v<L>(m);
  const VI moi = first ? aie::broadcast<int32, L>((int32)0xff800000) : mo.template cast_to<int32>();
  VI mr[NRB];
  for (unsigned rb = 0; rb < NRB; ++rb) chess_unroll_loop() mr[rb] = moi.template extract<TN>(rb).template grow_replicate<L>();
  const int32 *R si = (const int32 *)st;
  for (unsigned jq = 0; jq < NJQ; ++jq) chess_prepare_for_pipelining chess_loop_range(4, )
    for (unsigned rb = 0; rb < NRB; ++rb) chess_unroll_loop() mr[rb] = aie::max(mr[rb], aie::load_v<L>(si + (jq * NRB + rb) * L));
  VI mxk = aie::concat(red_max(mr[0]), red_max(mr[1]), red_max(mr[2]), red_max(mr[3]));
  if (aie::reduce_min(mxk) < 0) { // some row has no positive score: exact min pass (MXF=1)
    VI mn[NRB];
    for (unsigned rb = 0; rb < NRB; ++rb) chess_unroll_loop() mn[rb] = moi.template extract<TN>(rb).template grow_replicate<L>();
    for (unsigned jq = 0; jq < NJQ; ++jq) chess_prepare_for_pipelining chess_loop_range(4, )
      for (unsigned rb = 0; rb < NRB; ++rb) chess_unroll_loop() mn[rb] = aie::min(mn[rb], aie::load_v<L>(si + (jq * NRB + rb) * L));
    const VI mni = aie::concat(red_min(mn[0]), red_min(mn[1]), red_min(mn[2]), red_min(mn[3]));
    mxk = aie::select(mxk, mni, aie::lt(mxk, aie::zeros<int32, L>()));
  }
  VF mx = mxk.template cast_to<float>();
  const bool rs = first || aie::reduce_max(aie::sub(mx, mo)) > TH;
  VF lo;
  VI kq = aie::zeros<int32, L>();
  if (rs) {
    const VI rm = rnd_i(mx);
    mx = aie::to_float(rm);
    if (first) lo = aie::zeros<float, L>();
    else {
      kq = aie::upshift(aie::sub(rnd_i(mo), rm), 23);
      lo = pw2(aie::load_v<L>(l).template cast_to<int32>(), kq).template cast_to<float>();
    }
  } else { mx = mo; lo = aie::load_v<L>(l); }
  const VB one = aie::broadcast<bfloat16, L>(1.0f);
  const VF bzv = aie::sub(af(aie::broadcast<float, L>(EXB)), af(mx)).template to_vector<float>();
  aie::vector<float, 8> lt[NRB];
  for (unsigned rp = 0; rp < NRB / 2; ++rp) chess_loop_range(NRB / 2, NRB / 2) {
    const AF bz0 = af(bzv.extract<TN>(2 * rp).template grow_replicate<L>());
    const AF bz1 = af(bzv.extract<TN>(2 * rp + 1).template grow_replicate<L>());
    AF ls0 = aie::zeros<accfloat, L>(), ls1 = aie::zeros<accfloat, L>();
    const float *R s = st + 2 * rp * L;
    bfloat16 *R p = pt + 2 * rp * TK * TN;
    for (unsigned jq = 0; jq < NJQ; ++jq) chess_prepare_for_pipelining chess_loop_range(4, ) {
      v32accfloat u0 = aie::add(af(aie::load_v<L>(s + jq * NRB * L)), bz0);
      v32accfloat u1 = aie::add(af(aie::load_v<L>(s + jq * NRB * L + L)), bz1);
      VB p0 = concat(me_primitive::exp2_hw(extract_v16accfloat(u0, 0)), me_primitive::exp2_hw(extract_v16accfloat(u0, 1)));
      VB p1 = concat(me_primitive::exp2_hw(extract_v16accfloat(u1, 0)), me_primitive::exp2_hw(extract_v16accfloat(u1, 1)));
      bfloat16 *R pq = p + (jq / 2) * NRB * TK * TN + (jq % 2) * TM * TN;
      aie::store_v(pq, p0);
      aie::store_v(pq + TK * TN, p1);
      ls0 = aie::mac(ls0, p0, one); ls1 = aie::mac(ls1, p1, one);
    }
    lt[2 * rp] = red_sum(ls0.template to_vector<float>());
    lt[2 * rp + 1] = red_sum(ls1.template to_vector<float>());
  }
  const VF ln = aie::add(af(lo), af(aie::concat(lt[0], lt[1], lt[2], lt[3]))).template to_vector<float>();
  if (rs && !first) {
    VI kt[NRB];
    for (unsigned rb = 0; rb < NRB; ++rb) chess_unroll_loop() kt[rb] = kq.template extract<TN>(rb).template grow_replicate<L>();
    int32 *R oi = (int32 *)ot;
    for (unsigned db = 0; db < NDB; ++db) chess_prepare_for_pipelining chess_loop_range(4, )
      for (unsigned rb = 0; rb < NRB; ++rb) chess_unroll_loop()
        aie::store_v(oi + (db * NRB + rb) * L, pw2(aie::load_v<L>(oi + (db * NRB + rb) * L), kt[rb]));
  }
  aie::store_v(m, mx); aie::store_v(l, ln);
}
__attribute__((noinline)) static void fin1(const float *R ot, const float *R l, float *R il, bfloat16 *R out) {
  aie::store_v(il, aie::inv(aie::load_v<L>(l)));
  const VF iv = aie::load_v<L>(il);
  const VB one = aie::broadcast<bfloat16, L>(1.0f);
  const AF ai = af(iv);
  const VB ihv = ai.template to_vector<bfloat16>();
  const VB ilv = aie::msc(ai, ihv, one).template to_vector<bfloat16>();
  VB ih[NRB], il2[NRB];
  for (unsigned rb = 0; rb < NRB; ++rb) chess_unroll_loop() {
    ih[rb] = ihv.template extract<TN>(rb).template grow_replicate<L>();
    il2[rb] = ilv.template extract<TN>(rb).template grow_replicate<L>();
  }
  for (unsigned db = 0; db < NDB; ++db) chess_prepare_for_pipelining chess_loop_range(4, )
    for (unsigned rb = 0; rb < NRB; ++rb) chess_unroll_loop() {
      const AF ao = af(aie::load_v<L>(ot + (db * NRB + rb) * L));
      const VB oh = ao.template to_vector<bfloat16>();
      AF y = aie::mul(oh, ih[rb]);
      y = aie::mac(y, oh, il2[rb]);
      y = aie::mac(y, aie::msc(ao, oh, one).template to_vector<bfloat16>(), ih[rb]);
      aie::store_v(out + (((db / 2) * NRB + rb) * 2 + db % 2) * L, y.template to_vector<bfloat16>());
    }
}
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
"""

FA_KERNEL = r"""__global__ void @K@(aie::port<input_window_int16 *, aie::IFM, aie::shape<4, @QKROW@>> qk_@K@,
                    aie::port<input_window_int16 *, aie::WEIGHTS, aie::shape<4, @VROW@>> v_@K@,
                    aie::port<output_window_int16 *, aie::OFM, aie::shape<4, @OROW@>> o_@K@) {
    // GFA class H=@H@ N=@N@ M=@M@ D=@D@ (fa_attn.py): mesh row pm = one unit, core pn = @PNDOC@
    const int pm = aie::program_id(0);
    const int pn = aie::program_id(1);
    for (int j = 0; j < 4; j++) {
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
"""

BODY_FULL = r"""        const int16_t *kb = (const int16_t *)aie::load_buffer(qk_@K@, {pm, @E4@ + b * @E@}, {1, @E@});
        const int16_t *vb = (const int16_t *)aie::load_buffer(v_@K@, {pm, b * @E@}, {1, @E@});
        @NS@::qkb(kb, b, 0);
        @NS@::softb(b);
        @NS@::pvb(vb);
        aie::release(qk_@K@);
        aie::release(v_@K@);"""

BODY_SPLIT = r"""        const int16_t *k0 = (const int16_t *)aie::load_buffer(qk_@K@, {pm, @E4@ + (2 * b) * @E@}, {1, @E@});
        @NS@::qkb(k0, b, 0);
        aie::release(qk_@K@);
        const int16_t *k1 = (const int16_t *)aie::load_buffer(qk_@K@, {pm, @E4@ + (2 * b + 1) * @E@}, {1, @E@});
        @NS@::qkb(k1, b, 1);
        aie::release(qk_@K@);
        @NS@::softb(b);
        const int16_t *v0 = (const int16_t *)aie::load_buffer(v_@K@, {pm, (2 * b) * @E@}, {1, @E@});
        if ((pn & 1) == 0)
            @NS@::pvb(v0);
        aie::release(v_@K@);
        const int16_t *v1 = (const int16_t *)aie::load_buffer(v_@K@, {pm, (2 * b + 1) * @E@}, {1, @E@});
        if ((pn & 1) == 1)
            @NS@::pvb(v1);
        aie::release(v_@K@);"""


def _sub(t: str, kv: dict) -> str:
    for k, v in kv.items():
        t = t.replace("@%s@" % k, str(v))
    assert "@" not in t.replace("aie::", ""), t[t.index("@") - 40:t.index("@") + 40]
    return t


def fa_prologue(classes: list) -> str:
    t = FA_COMMON
    for c in classes:
        t += _sub(FA_NS, dict(NS="ns_" + c["name"], KVL=c["kvl"], NKV=c["nkv"], DQ=c["dq"], NQH=2 if c["split"] else 1))
    return t + "// AIEHLC_KERNEL_PROLOGUE_END\n"


def fa_kernel_text(c: dict) -> str:
    kv = dict(K=c["name"], NS="ns_" + c["name"], E=c["E"], E4=4 * c["E"], NKV=c["nkv"], QKROW=c["qkrow"],
              VROW=c["vrow"], OROW=c["orow"], H=c["H"], N=c["N"], M=c["M"], D=c["D"])
    body = _sub(BODY_SPLIT if c["split"] else BODY_FULL, kv)
    if c["split"]:
        kv.update(PNDOC="query block pn>>1, output half pn&1", QSEL="(j >> 1) == (pn >> 1)", QHALF="j & 1")
    else:
        kv.update(PNDOC="query block pn", QSEL="j == pn", QHALF="0")
    kv["BODY"] = body
    return _sub(FA_KERNEL, kv)


FA_HOST = r"""
/* GFA (--attn-fa): fused flash-attention sites. unet_aie_attn(cid, args, nargs, role, cls): role 0 = the QK^T
 * bmm site (packs Q x log2e and the centred K into the class qk buffer, no launch), role 1 = the PV bmm site
 * (packs V, launches the class kernel units/4 times, writes O f32). The TVM softmax between them is gone. */
typedef struct {
    int H, N, M, D, split, dq, nkv, E, units, upr, launches, qkrow, vrow, orow;
} fa_cls;
static const fa_cls g_fa_cls[] = {
@CLASSES@
};
#define FA_NCLS ((int)(sizeof(g_fa_cls) / sizeof(g_fa_cls[0])))
static uint16_t *g_fa_qk[FA_NCLS], *g_fa_v[FA_NCLS], *g_fa_o[FA_NCLS];
static float *g_fa_mu = nullptr;
static unsigned g_fa_sites = 0;
static inline uint16_t fa_f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static inline float fa_bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static int fa_alloc(int k) {
    if (g_fa_qk[k])
        return 0;
    const fa_cls &c = g_fa_cls[k];
    g_fa_qk[k] = (uint16_t *)__Runtime_Alloc((size_t)c.units * c.qkrow * 2);
    g_fa_v[k] = (uint16_t *)__Runtime_Alloc((size_t)c.units * c.vrow * 2);
    g_fa_o[k] = (uint16_t *)__Runtime_Alloc((size_t)c.units * c.orow * 2);
    if (!g_fa_mu)
        g_fa_mu = (float *)aligned_alloc(64, 256 * 4);
    return (g_fa_qk[k] && g_fa_v[k] && g_fa_o[k] && g_fa_mu) ? 0 : -1;
}
static int uc_fa_launch(int k, uint16_t *qk, uint16_t *v, uint16_t *o) {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({@PART_C0@, 6, 0, 6}, 4, 4); // aiegraph lint L015: the mesh VarDecl in the launching function
    int16_t *q16 = (int16_t *)qk, *v16 = (int16_t *)v, *o16 = (int16_t *)o;
@DISPATCH@
    return -6;
}
/* block layouts (Maxwell fa_load.h); dq = block width (80 for the halves of a split D=160 block) */
static inline unsigned fa_qidx(unsigned d, unsigned r) { return (((d >> 3) * 4 + (r >> 3)) * 8 + (d & 7)) * 8 + (r & 7); }
static inline unsigned fa_kidx(unsigned j, unsigned d, unsigned dq) { return (((j >> 2) * (dq >> 3) + (d >> 3)) * 4 + (j & 3)) * 8 + (d & 7); }
static inline unsigned fa_vidx(unsigned d, unsigned j) { return (((d >> 2) * 4 + (j >> 3)) * 4 + (d & 3)) * 8 + (j & 7); }
static int uc_fa(void *const *argv, int nargs, int role, int k) {
    if (nargs != 3 || k < 0 || k >= FA_NCLS)
        return -2;
    const fa_cls &c = g_fa_cls[k];
    const uc_dlt *x = (const uc_dlt *)argv[0], *w = (const uc_dlt *)argv[1], *y = (const uc_dlt *)argv[2];
    if (x->ndim != 3 || w->ndim != 3 || y->ndim != 3)
        return -2;
    const int H = c.H, N = c.N, M = c.M, D = c.D, dq = c.dq, E = c.E, nkv = c.nkv;
    if (fa_alloc(k) != 0)
        return -1;
    uint16_t *QK = g_fa_qk[k], *V = g_fa_v[k], *O = g_fa_o[k];
    const int nh = c.split ? 2 : 1, npu = c.split ? 1 : c.upr; /* d halves per block, units per head */
    if (role == 0) { /* x = Q [H][N][D], w = K [H][M][D] */
        if ((int)x->shape[0] != H || (int)x->shape[1] != N || (int)x->shape[2] != D || (int)w->shape[0] != H ||
            (int)w->shape[1] != M || (int)w->shape[2] != D)
            return -3;
        const float *Q = (const float *)x->data, *K = (const float *)w->data;
        const float lg2e = 1.4426950408889634f;
        for (int h = 0; h < H; h++) {
            const float *kh = K + (size_t)h * M * D, *qh = Q + (size_t)h * N * D;
            /* PKC: keys centred over the M real keys (per column mean) */
            float *mu = g_fa_mu;
            for (int d = 0; d < D; d++)
                mu[d] = 0.0f;
            for (int j = 0; j < M; j++) {
                const float *kr = kh + (size_t)j * D;
                for (int d = 0; d < D; d++)
                    mu[d] += kr[d];
            }
            const float im = 1.0f / (float)M;
            for (int d = 0; d < D; d++)
                mu[d] *= im;
            const int u0 = h * npu;
            uint16_t *kd = QK + (size_t)u0 * c.qkrow + 4 * E;
            /* K block (b, half hh): kidx order = j4, d8, j&3, d&7 -> sequential writes */
            uint16_t *kp = kd;
            for (int b = 0; b < nkv; b++)
                for (int hh = 0; hh < nh; hh++)
                    for (int j4 = 0; j4 < 8; j4++)
                        for (int d8 = 0; d8 < dq / 8; d8++)
                            for (int jj = 0; jj < 4; jj++) {
                                const int j = b * 32 + j4 * 4 + jj, c0 = hh * dq + d8 * 8;
                                if (j < M) {
                                    const float *kr = kh + (size_t)j * D + c0, *mr = mu + c0;
                                    for (int dd = 0; dd < 8; dd++)
                                        kp[dd] = fa_f2bf(kr[dd] - mr[dd]);
                                } else
                                    for (int dd = 0; dd < 8; dd++)
                                        kp[dd] = 0;
                                kp += 8;
                            }
            for (int u = 1; u < npu; u++) /* every unit (mesh row) of this head carries its K */
                memcpy(QK + (size_t)(u0 + u) * c.qkrow + 4 * E, kd, (size_t)nkv * nh * E * 2);
            /* Q blocks: non-split unit u = rows u*128.., block qb; split: sub-block (qb, hh). qidx order = d8, r8, d&7, r&7 */
            for (int u = 0; u < npu; u++) {
                uint16_t *qp = QK + (size_t)(u0 + u) * c.qkrow;
                for (int qb = 0; qb < (c.split ? 2 : 4); qb++)
                    for (int hh = 0; hh < nh; hh++) {
                        const float *q0 = qh + (size_t)(u * 128 + qb * 32) * D + hh * dq;
                        for (int d8 = 0; d8 < dq / 8; d8++)
                            for (int r8 = 0; r8 < 4; r8++)
                                for (int dd = 0; dd < 8; dd++) {
                                    const float *qc = q0 + (size_t)(r8 * 8) * D + d8 * 8 + dd;
                                    for (int rr = 0; rr < 8; rr++)
                                        qp[rr] = fa_f2bf(qc[(size_t)rr * D] * lg2e);
                                    qp += 8;
                                }
                    }
            }
        }
        return 0;
    }
    /* role 1: x = P (unused), w = V^T [H][D][M], y = O [H][N][D] */
    if ((int)w->shape[0] != H || (int)w->shape[1] != D || (int)w->shape[2] != M || (int)y->shape[0] != H ||
        (int)y->shape[1] != N || (int)y->shape[2] != D)
        return -3;
    const float *VT = (const float *)w->data;
    float *Y = (float *)y->data;
    for (int h = 0; h < H; h++) {
        const float *vh = VT + (size_t)h * D * M;
        const int u0 = h * npu;
        uint16_t *vd = V + (size_t)u0 * c.vrow, *vp = vd;
        /* V block (b, half hh): vidx order = d4, j8, d&3, j&7 -> sequential writes */
        for (int b = 0; b < nkv; b++)
            for (int hh = 0; hh < nh; hh++)
                for (int d4 = 0; d4 < dq / 4; d4++)
                    for (int j8 = 0; j8 < 4; j8++)
                        for (int dd = 0; dd < 4; dd++) {
                            const int j0 = b * 32 + j8 * 8;
                            const float *vr = vh + (size_t)(hh * dq + d4 * 4 + dd) * M + j0;
                            if (j0 + 8 <= M)
                                for (int jj = 0; jj < 8; jj++)
                                    vp[jj] = fa_f2bf(vr[jj]);
                            else
                                for (int jj = 0; jj < 8; jj++)
                                    vp[jj] = j0 + jj < M ? fa_f2bf(vr[jj]) : 0;
                            vp += 8;
                        }
        for (int u = 1; u < npu; u++)
            memcpy(V + (size_t)(u0 + u) * c.vrow, vd, (size_t)c.vrow * 2);
    }
#if UC_CTRLPKT
    if (!g_uc_ht) {
        __Runtime_ctrl_high_throughput_enable(1);
        g_uc_ht = 1;
    }
#endif
    const size_t qkb = (size_t)c.units * c.qkrow * 2, vb = (size_t)c.units * c.vrow * 2, ob = (size_t)c.units * c.orow * 2;
    for (int l = 0; l < c.launches; l++) {
        XTime lt0, lt1;
        XTime_GetTime(&lt0);
        if (l == 0)
            uc_sync_pre(QK, qkb, V, vb, O, ob);
        const int lrc = uc_fa_launch(k, QK + (size_t)l * 4 * c.qkrow, V + (size_t)l * 4 * c.vrow, O + (size_t)l * 4 * c.orow);
        if (l == c.launches - 1)
            uc_sync_post(O, ob);
        XTime_GetTime(&lt1);
        g_uc_aie_ticks += (unsigned long long)(lt1 - lt0);
        g_aie_launches++;
        if (lrc != 0)
            return lrc;
    }
    /* O: same sub-block order as Q (sequential reads) */
    for (int h = 0; h < H; h++) {
        float *yh = Y + (size_t)h * N * D;
        for (int u = 0; u < npu; u++) {
            const uint16_t *op = O + (size_t)(h * npu + u) * c.orow;
            for (int qb = 0; qb < (c.split ? 2 : 4); qb++)
                for (int hh = 0; hh < nh; hh++) {
                    float *y0 = yh + (size_t)(u * 128 + qb * 32) * D + hh * dq;
                    for (int d8 = 0; d8 < dq / 8; d8++)
                        for (int r8 = 0; r8 < 4; r8++)
                            for (int dd = 0; dd < 8; dd++) {
                                float *yc = y0 + (size_t)(r8 * 8) * D + d8 * 8 + dd;
                                for (int rr = 0; rr < 8; rr++)
                                    yc[(size_t)rr * D] = fa_bf2f(op[rr]);
                                op += 8;
                            }
                }
        }
    }
    g_uc_macs += 2ULL * H * N * M * D;
    g_fa_sites++;
    return 0;
}

extern "C" int unet_aie_attn(int cid, void *args, int nargs, int role, int cls) {
    g_uc_calls++;
    const int rc = uc_fa((void *const *)args, nargs, role, cls);
    if (rc != 0) {
        printf("[unet_aie_attn] fa layer %d role %d cls %d rc=%d\n", cid, role, cls, rc);
        if (!g_uc_rc)
            g_uc_rc = rc;
    }
    return rc;
}

"""


def fa_host(classes: list) -> str:
    rows = ",\n".join("    {%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d}"
                      % (c["H"], c["N"], c["M"], c["D"], c["split"], c["dq"], c["nkv"], c["E"], c["units"], c["upr"],
                         c["launches"], c["qkrow"], c["vrow"], c["orow"]) for c in classes)
    disp = "\n".join("    if (k == %d) {\n        %s<<<mesh>>>(q16, v16, o16);\n        return 0;\n    }"
                     % (c["idx"], c["name"]) for c in classes)
    return FA_HOST.replace("@CLASSES@", rows).replace("@DISPATCH@", disp)


DECL_FA = "int unet_aie_attn(int cid, void *args, int nargs, int role, int cls);   /* --attn-fa (GFA) */\n"
