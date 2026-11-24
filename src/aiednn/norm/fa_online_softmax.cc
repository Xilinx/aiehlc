/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
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
