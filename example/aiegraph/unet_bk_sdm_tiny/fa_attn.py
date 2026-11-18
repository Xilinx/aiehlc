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
v5 (one global routing() per aiehlc multi-kernel program: FA must use gemm_kacc's topology): q = row broadcast,
row pm = query block pm (split: block pm>>1) of the 4 heads; kv = column broadcast, column pn = head pn's stream;
o = gather o[pm][pn]. v3: q port row = the unit's 4 Q (sub-)blocks only; K and V share the second port as one stream per HEAD
([K_b V_b] per key block b; split: [K_b h0, K_b h1, V_b h0, V_b h1]). A launch = chunk c x 4 heads (mesh row = head),
so the kv rows of a head group are packed once and shared by every chunk's launch (no K/V replication; the KO
aiehlc has no row broadcast for a size-1 port dim, 0ee6397).
Block layouts (Maxwell fa_load.h): Q/O qidx(d,r) = ((d/8)*4 + r/8)*64 + (d%8)*8 + r%8,
K kidx(j,d) = (((j/4)*(D/8) + d/8)*4 + j%4)*8 + d%8, V vidx(d,j) = (((d/4)*4 + j/8)*4 + d%4)*8 + j%8."""

import math
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402
_SRC = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../src"))
if _SRC not in _sys.path:  # C text lives in src/aiednn/<cat>/ (kernels) and src/aiegraph/runtime/
    _sys.path.insert(0, _SRC)
from aiednn import runtime_source as _rt, template as _tpl  # noqa: E402

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
        assert H % 4 == 0, H
        if split:
            assert N == 64, N
            units, upr = H, 1
            qkrow, vrow, orow = 8 * E, 4 * nkv * E, 4 * E
        else:
            assert N % 128 == 0, N
            units, upr = H * N // 128, N // 128
            qkrow, vrow, orow = 4 * E, 2 * nkv * E, 4 * E
        assert units % 4 == 0, units
        out.append(dict(idx=i, H=H, N=N, M=M, D=D, split=int(split), dq=dq, nkv=nkv, kvl=kvl, E=E, units=units,
                        upr=upr, launches=units // 4, qkrow=qkrow, vrow=vrow, orow=orow,
                        name=f"fa_h{H}_n{N}_m{M}_d{D}"))
    return out


FA_COMMON = _tpl("attention/fa_common.cc")

FA_NS = (_tpl("attention/fa_ns_head.cc") + _tpl("norm/fa_online_softmax.cc") + _tpl("attention/fa_ns_tail.cc"))

FA_KERNEL = _tpl("attention/fa_kernel.cc")

BODY_FULL = _tpl("attention/fa_body_full.cc")

BODY_SPLIT = _tpl("attention/fa_body_split.cc")


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
        kv.update(PNDOC="head pn, query block pm>>1, output half pm&1", QSEL="(j >> 1) == pn", QHALF="j & 1", NQB=8)
    else:
        kv.update(PNDOC="head pn, query block pm", QSEL="j == pn", QHALF="0", NQB=4)
    kv["BODY"] = body
    return _sub(FA_KERNEL, kv)


FA_HOST = _tpl("attention/fa_host.cc")


def fa_host(classes: list) -> str:
    rows = ",\n".join("    {%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d}"
                      % (c["H"], c["N"], c["M"], c["D"], c["split"], c["dq"], c["nkv"], c["E"], c["units"], c["upr"],
                         c["launches"], c["qkrow"], c["vrow"], c["orow"]) for c in classes)
    disp = "\n".join("    if (k == %d) {\n        %s<<<mesh>>>(q16, v16, o16);\n        return 0;\n    }"
                     % (c["idx"], c["name"]) for c in classes)
    return FA_HOST.replace("@CLASSES@", rows).replace("@DISPATCH@", disp)


DECL_FA = _tpl("attention/fa_decl.h")


# ── Group-offload steps (design 2026-11-16-aiegraph-group-offload §5.3, sprint unet-m3 P1abi) ─────────────────────
# Thin AIEK_STEP wrappers over unet_aie_attn_kl, so aiegraph's offload_group can place them instead of
# keep_layout_rt.py's text rewrite. One wrapper per keep-layout role (aie_user_kernel.h has the step <-> role
# table): a step's aiek_buf carries the [rows][H*D] linear output's pointer and row stride (ld = H*D), which is
# exactly the (a, lda) / (b, ldb) the kl roles take today, so the generated call is the same kl call.
# Not appended to the kernel source yet (that is S3: the driver must stay byte-equivalent until then).
FA_STEPS = _tpl("attention/fa_steps.cc")

#: step -> keep-layout role of unet_aie_attn_kl (keep_layout_rt.py roles 2/3/5/6)
FA_STEP_ROLES = {"fa_step_pack_q": 2, "fa_step_pack_k": 3, "fa_step_pack_v": 5, "fa_step_run_o": 6}


def fa_steps() -> str:
    """The AIEK_STEP wrapper text (C++), for the group-offload kernel source."""
    return FA_STEPS


# ── S3: FA groups as aiegraph group steps ─────────────────────────────────────────────────────────────────────────
#: the keep-layout entry the steps call (was inserted by keep_layout_rt.fa_pass)
DECL_KL = ("int unet_aie_attn_kl(int cid, int role, int cls, const float *a, int lda, float *b, int ldb,\n"
           "                     float s);   /* FA group steps (aiegraph offload_group, fa_attn.fa_step_emit) */\n")


def fa_group_steps(ag, g, cid_qk: int, cid_pv: int, cls: int) -> list:
    """The four ag.step()s of one attention group (design §5.2): pack Q / K / V at the reads, run + write O"""
    if g.scale is None or g.scale_k is None:
        raise SystemExit(f"{g.name}: no Q/K scale from the matcher (match with onnx=)")
    return [ag.step("fa_step_pack_q", reads=g.q, consts=dict(cid=cid_qk, cls=cls, s=g.scale)),
            ag.step("fa_step_pack_k", reads=g.k, consts=dict(cid=cid_qk, cls=cls, s=g.scale_k)),
            ag.step("fa_step_pack_v", reads=g.v, consts=dict(cid=cid_pv, cls=cls)),
            ag.step("fa_step_run_o", writes=g.o, consts=dict(cid=cid_pv, cls=cls))]


def fa_step_emit(g):
    """offload_group(emit=): the driver code of a step = the unet_aie_attn_kl call its AIEK_STEP wrapper makes
    (FA_STEP_ROLES, aie_user_kernel.h step <-> role table), with the boundary buffer's pointer and row stride"""
    bufs = {"fa_step_pack_q": g.q, "fa_step_pack_k": g.k, "fa_step_pack_v": g.v, "fa_step_run_o": g.o}

    def emit(s):
        role, b, c = FA_STEP_ROLES[s["name"]], bufs[s["name"]], s["consts"]
        t = (s["writes"] if role == 6 else s["reads"])
        assert t == [b.tensor], (g.name, s["name"], t, b.tensor)
        if role == 6:
            call = "unet_aie_attn_kl(%d, 6, %d, 0, 0, (float *)tensors[%d].data, %d, 1.0f);" % (
                c["cid"], c["cls"], b.tensor, b.ld)
        else:
            call = "unet_aie_attn_kl(%d, %d, %d, (const float *)tensors[%d].data, %d, 0, 0, %.9ef);" % (
                c["cid"], role, c["cls"], b.tensor, b.ld, float(c.get("s", 1.0)))
        return {"c": ["UP_AIE_BEGIN(@K@);", call + "   /* %s %s */" % (g.name, s["name"]), "UP_AIE_END(@K@);"],
                "tag": "unet_aie_attn %d" % c["cid"]}
    return emit
