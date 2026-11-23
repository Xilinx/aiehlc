###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.flashattn -- one matched attention group (QK^T bmm -> softmax -> PV bmm) = ONE flash-attention
(online softmax) on the AIE, placed as an aiegraph group offload.

    from aiegraph import flashattn as fa
    classes = fa.fa_classes(att)                     # att: attention.partition_attention rows; sets c["fa"]
    kw = fa.fa_kernel_parts(classes)                 # kincore.write_source(..., **kw) extra parts
    steps = fa.fa_group_steps(ag, g, cid_qk, cid_pv, cls)
    emod.offload_group(g, source=kcc, steps=steps, emit=fa.fa_step_emit(g))
    emod.driver_template(**fa.fa_driver_glue(att))

Generic over the model: a class is keyed on the attention shape (H heads, N query rows, M key rows, D head dim)
only. The C text (kernel, host entries, step wrappers, declarations) is the aiednn attention/ templates; this
module only picks the class constants and fills them in.

Kernel = the board-proven M4 flash attention (QK^T mmul<4,8,8> with key mask in the acc init, int32 max + exact
min fallback, exp2_hw, lazy power-of-2 rescale, PV mmul, bf16 hi/lo 1/l finalize), Br = Bc = 32, 4x4 mesh:
  D <= 80 : mesh row = one unit (head h, 128 query rows), core pn = query block pn of the unit.
  D > 80  : (N = 64, D/2 <= 80) every block is split into two D/2-wide halves: unit = head (64 rows),
            core pn = query block pn>>1, output half pn&1.
The QK site packs Q (x log2 e) and the centred K, the softmax call is removed, the PV site packs V, launches the
class kernel and writes O (f32). Block layouts (fa_load.h): Q/O qidx(d,r) = ((d/8)*4 + r/8)*64 + (d%8)*8 + r%8,
K kidx(j,d) = (((j/4)*(D/8) + d/8)*4 + j%4)*8 + d%8, V vidx(d,j) = (((d/4)*4 + j/8)*4 + d%4)*8 + j%8.
"""
from __future__ import annotations

from aiednn import template as _tpl

from .errors import AiegraphError

_STAGE = "flashattn"
E_BR = 32                       # query / key block rows (Br = Bc)
DQ_MAX = 80                     # widest head-dim block a core holds with 3 ping-pong ports
__all__ = ["fa_classes", "fa_prologue", "fa_host", "fa_kernel_parts", "fa_group_steps", "fa_step_emit",
           "fa_driver_glue", "FA_STEPS", "FA_STEP_ROLES", "DECL_FA", "DECL_KL"]

FA_COMMON = _tpl("attention/fa_common.cc")
FA_NS = _tpl("attention/fa_ns_head.cc") + _tpl("norm/fa_online_softmax.cc") + _tpl("attention/fa_ns_tail.cc")
#: one C++ template __global__ fa_kernel<NS, HEADS, NQ, MK, DH, SPLIT, E, NKV, QKROW, VROW, OROW> (aiehlc
#: instantiates it per launch in the same source file); emitted once, every class launches its instance
FA_KERNEL = _tpl("attention/fa_kernel.cc")
FA_TPARAMS = ("NS", "HEADS", "NQ", "MK", "DH", "SPLIT", "E", "NKV", "QKROW", "VROW", "OROW")
FA_HOST = _tpl("attention/fa_host.cc")
#: kernel-side declarations the driver needs: the role entry (fa_decl.h) and the keep-layout entry the steps call
DECL_FA = _tpl("attention/fa_decl.h")
DECL_KL = _tpl("attention/fa_kl_decl.h")
#: AIEK_STEP wrappers over aiednn_aie_attn_kl (design 2026-11-16-aiegraph-group-offload §5.3)
FA_STEPS = _tpl("attention/fa_steps.cc")
#: step -> keep-layout role of aiednn_aie_attn_kl (aie_user_kernel.h has the step <-> role table)
FA_STEP_ROLES = {"fa_step_pack_q": 2, "fa_step_pack_k": 3, "fa_step_pack_v": 5, "fa_step_run_o": 6}
_BUF = {"fa_step_pack_q": "q", "fa_step_pack_k": "k", "fa_step_pack_v": "v", "fa_step_run_o": "o"}


def fa_classes(att: list) -> list:
    """Attention bmm rows (``group``, ``role`` qk|pv, ``xshape``, ``wshape``, ``yshape``) -> the flash-attention
    classes, one per distinct (H, N, M, D) in first-seen group order. Sets ``c["fa"]`` = class index on both
    rows of every group. A group without exactly one qk and one pv row, a PV shape that does not continue the
    QK one, or a shape the kernel has no walk for is an AiegraphError."""
    groups = {}
    for c in att:
        groups.setdefault(c["group"], {})[c["role"]] = c
    keys, member = [], []
    for g, d in sorted(groups.items()):
        if set(d) != {"qk", "pv"}:
            raise AiegraphError(_STAGE, f"attention group {g}: roles {sorted(d)}, need exactly qk and pv")
        qk, pv = d["qk"], d["pv"]
        H = int(qk["xshape"][0])
        N, D = qk["xshape"][1], qk["xshape"][2]       # Q [H][N][D]
        M = qk["wshape"][1]                            # K [H][M][D]
        if list(pv["wshape"]) != [H, D, M] or list(pv["yshape"]) != [H, N, D]:
            raise AiegraphError(_STAGE, f"attention group {g}: PV w{pv['wshape']} y{pv['yshape']} does not "
                                f"continue QK (H={H} N={N} M={M} D={D})")
        key = (H, N, M, D)
        if key not in keys:
            keys.append(key)
        member.append((key, qk, pv))
    out = [_fa_class(i, *k) for i, k in enumerate(keys)]
    for key, qk, pv in member:
        qk["fa"] = pv["fa"] = keys.index(key)
    return out


def _fa_class(i: int, H: int, N: int, M: int, D: int) -> dict:
    split = D > DQ_MAX
    dq = D // 2 if split else D
    why = []
    if dq % 8 or dq > DQ_MAX:
        why.append(f"head dim block {dq} (D={D}) must be a multiple of 8 and <= {DQ_MAX}")
    if H % 4:
        why.append(f"H={H} heads must be a multiple of 4 (mesh rows)")
    if split and N != 64:
        why.append(f"split head dim (D={D} > {DQ_MAX}) needs N == 64 query rows, got {N}")
    if not split and N % 128:
        why.append(f"N={N} query rows must be a multiple of 128")
    if why:
        raise AiegraphError(_STAGE, f"attention class (H={H}, N={N}, M={M}, D={D}): " + "; ".join(why))
    nkv = -(-M // E_BR)
    kvl = M - E_BR * (nkv - 1)
    E = E_BR * dq
    if split:
        units, upr = H, 1
        qkrow, vrow, orow = 8 * E, 4 * nkv * E, 4 * E
    else:
        units, upr = H * N // 128, N // 128
        qkrow, vrow, orow = 4 * E, 2 * nkv * E, 4 * E
    if units % 4:
        raise AiegraphError(_STAGE, f"attention class (H={H}, N={N}, M={M}, D={D}): {units} units, not a "
                            "multiple of 4 (one launch = 4 units)")
    return dict(idx=i, H=H, N=N, M=M, D=D, split=int(split), dq=dq, nkv=nkv, kvl=kvl, E=E, units=units,
                upr=upr, launches=units // 4, qkrow=qkrow, vrow=vrow, orow=orow, name=f"fa_h{H}_n{N}_m{M}_d{D}")


def _sub(t: str, kv: dict) -> str:
    for k, v in kv.items():
        t = t.replace("@%s@" % k, str(v))
    if "@" in t.replace("aie::", ""):
        i = t.index("@")
        raise AiegraphError(_STAGE, f"unfilled template parameter near {t[max(0, i - 40):i + 40]!r}")
    return t


def _need_classes(classes, who: str):
    if not classes:
        raise AiegraphError(_STAGE, f"{who}: no flash-attention class")
    for c in classes:
        miss = [k for k in ("idx", "name", "H", "N", "M", "D", "split", "dq", "nkv", "kvl", "E", "units", "upr",
                            "launches", "qkrow", "vrow", "orow") if k not in c]
        if miss:
            raise AiegraphError(_STAGE, f"{who}: class {c.get('name', c)} lacks {miss}", hint="use fa_classes()")


def fa_prologue(classes: list) -> str:
    """The kernel prologue: the common FA helpers + one online-softmax namespace per class."""
    _need_classes(classes, "fa_prologue")
    t = FA_COMMON
    for c in classes:
        t += _sub(FA_NS, dict(NS="ns_" + c["name"], KVL=c["kvl"], NKV=c["nkv"], DQ=c["dq"], NQH=2 if c["split"] else 1))
    return t + "// AIEHLC_KERNEL_PROLOGUE_END\n"


def _fa_launch(c: dict) -> str:
    """The launch callee ``fa_kernel<...>`` of class *c* (FA_TPARAMS order; the TU holds FA_KERNEL once)."""
    vals = ("ns_" + c["name"], c["H"], c["N"], c["M"], c["D"], c["split"], c["E"], c["nkv"], c["qkrow"], c["vrow"],
            c["orow"])
    return "fa_kernel<" + ", ".join(str(v) for v in vals) + ">"


def fa_host(classes: list) -> str:
    """The host entries: the class table and one launch per class (aiednn attention/fa_host.cc)."""
    _need_classes(classes, "fa_host")
    rows = ",\n".join("    {%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d}"
                      % (c["H"], c["N"], c["M"], c["D"], c["split"], c["dq"], c["nkv"], c["E"], c["units"], c["upr"],
                         c["launches"], c["qkrow"], c["vrow"], c["orow"]) for c in classes)
    disp = "\n".join("    if (k == %d) {\n        %s<<<mesh>>>(q16, v16, o16);\n        return 0;\n    }"
                     % (c["idx"], _fa_launch(c)) for c in classes)
    return FA_HOST.replace("@CLASSES@", rows).replace("@DISPATCH@", disp)


def fa_kernel_parts(classes: list) -> dict:
    """The extra parts of ``kincore.kernel_source`` / ``write_source`` for the classes: the two exports, the
    prologue, the fa_kernel template and the host entries."""
    _need_classes(classes, "fa_kernel_parts")
    return dict(exports=("aiednn_aie_attn", "aiednn_aie_attn_kl"), prologue=fa_prologue(classes) + "\n",
                kernels="\n\n" + FA_KERNEL, host_entries=fa_host(classes).lstrip("\n"))


def fa_group_steps(ag, g, cid_qk: int, cid_pv: int, cls: int) -> list:
    """The four ``ag.step()`` of one attention group (design §5.2): pack Q / K / V at the reads, run + write O.
    *g* = an ``attention.AttentionGroup`` matched with onnx= (it carries the Q / K scales)."""
    if g.scale is None or g.scale_k is None:
        raise AiegraphError(_STAGE, f"{g.name}: no Q/K scale from the matcher", hint="match with onnx=")
    return [ag.step("fa_step_pack_q", reads=g.q, consts=dict(cid=cid_qk, cls=cls, s=g.scale)),
            ag.step("fa_step_pack_k", reads=g.k, consts=dict(cid=cid_qk, cls=cls, s=g.scale_k)),
            ag.step("fa_step_pack_v", reads=g.v, consts=dict(cid=cid_pv, cls=cls)),
            ag.step("fa_step_run_o", writes=g.o, consts=dict(cid=cid_pv, cls=cls))]


def fa_step_emit(g):
    """``offload_group(emit=)``: the driver code of a step = the aiednn_aie_attn_kl call its AIEK_STEP wrapper
    makes (FA_STEP_ROLES), with the boundary buffer's pointer and row stride. A step that is not an FA step,
    or that reads / writes another tensor than the group's boundary buffer, is an AiegraphError."""
    bufs = {name: getattr(g, b) for name, b in _BUF.items()}

    def emit(s):
        if s["name"] not in FA_STEP_ROLES:
            raise AiegraphError(_STAGE, f"{g.name}: {s['name']} is not a flash-attention step")
        role, b, c = FA_STEP_ROLES[s["name"]], bufs[s["name"]], s["consts"]
        t = (s["writes"] if role == 6 else s["reads"])
        if t != [b.tensor]:
            raise AiegraphError(_STAGE, f"{g.name}: {s['name']} touches {t}, the group buffer is [{b.tensor}]")
        if role == 6:
            call = "aiednn_aie_attn_kl(%d, 6, %d, 0, 0, (float *)tensors[%d].data, %d, 1.0f);" % (
                c["cid"], c["cls"], b.tensor, b.ld)
        else:
            call = "aiednn_aie_attn_kl(%d, %d, %d, (const float *)tensors[%d].data, %d, 0, 0, %.9ef);" % (
                c["cid"], role, c["cls"], b.tensor, b.ld, float(c.get("s", 1.0)))
        return {"c": ["UP_AIE_BEGIN(@K@);", call + "   /* %s %s */" % (g.name, s["name"]), "UP_AIE_END(@K@);"],
                "tag": "aiednn_aie_attn %d" % c["cid"]}
    return emit


def fa_driver_glue(att: list = (), kprefix: str = None) -> dict:
    """``emod.driver_template(**fa_driver_glue(att))``: boardmain.driver_glue with the FA declarations and the
    attention PROF branch when *att* (attention rows) is not empty. Every row must be an FA group step
    (``fa`` class set by fa_classes, ``fa_steps`` set once offload_group placed it)."""
    from .boardmain import driver_glue
    bad = [c.get("onnx", c.get("cid")) for c in att if c.get("fa") is None or not c.get("fa_steps")]
    if bad:
        raise AiegraphError(_STAGE, f"fa_driver_glue: {len(bad)} attention rows are not FA group steps: {bad[:3]}",
                            hint="fa_classes + offload_group(fa_group_steps) first")
    return driver_glue(DECL_FA + DECL_KL if att else "", attn=bool(att), kprefix=kprefix)
