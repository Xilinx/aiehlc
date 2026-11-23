###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.cnn_chunk -- generic CNN conv offload, chunked (the padded-instance flow is aiegraph.cnn): every
matched conv call (``emod.match("conv", groups=(1, "dw"))``) as ONE
aiegraph group step on the AIE gemm_kacc kernel (bf16 in, fp32 accumulation), any stride / pad / KxK /
depthwise, with the TVM call's fused epilogue (bias, SiLU, residual) on the A78.

    plans = cnn.plan_convs(groups)                    # per conv: Np / Kk / Kp / chunk MC / launches / act
    kcc   = cnn.write_kernel(out / "aiek_src", proto.spec(), plans, kprefix="cnn", entry=ENTRY, warm=True)
    cnn.offload_convs(emod, groups, plans, kcc, kprefix="cnn")

Kernel (``aiednn/gemm/gemm_kacc_kernel.cc``, one instance per distinct (Np, Kk, MC)): Y[Np][MC] =
W[Np][Kp] . X[MC][Kp]^T on the 4x4 mesh, K accumulated in the core. Host glue
(``aiednn/conv2d/kacc_chunk_{board,conv,entry}.cc``): the output pixels are cut into chunks of MC (one
launch each), the channels padded to Np = ceil64(N), K = C*KH*KW padded to Kk = ceil64(K) (depthwise
weights expanded block-diagonally), im2col of the NCHWc input, scatter + epilogue into the NCHWc output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import AiegraphError

__all__ = ["MC", "ConvPlan", "kacc_geom", "kacc_name", "kernel_text", "plan_conv", "plan_convs",
           "write_kernel", "pick_carrier", "offload_convs"]

_STAGE = "cnn"
MC = 256            # output pixels per launch (4 mesh columns x 64; board-proven non-transposed walk)
# palmyra213 self-check (UC_SELFCHECK, 2026-11-18): gemm_kacc instances with a K walk of <= 64 words
# (64x64x256, 128x64x256, 128x128x256, 256x128x512) and 64x2304x256 run but return wrong sums (rel err
# 0.2-0.5) while their x86 model is exact; the kernel K is >= KK_MIN and a K > KK_BIG gets >= NP_MIN_BIGK rows
KK_MIN = 192
KK_BIG = 1152
NP_MIN_BIGK = 128
L1_PORTS = 29696    # aiehlc's port-buffer region (trace_routing kPortBufferBytes); windows are ping-pong
# fused TVM conv functions this glue reproduces: conv [+ per-channel add] [+ SiLU] [+ same-shape add]
_SYM = re.compile(r"(?:tvmgen_default_)?fused_nn_contrib_(depthwise_)?conv2d_NCHWc(_add)?(_sigmoid_multiply)?"
                  r"(_add)?(?:_\d+)?")


def _c64(v: int) -> int:
    return -(-int(v) // 64) * 64


def kacc_geom(ch: int, k: int, px: int) -> dict:
    """The non-transposed gemm_kacc walk of one (ch, k, px) instance: core (pm, pn) owns ch/4 channels
    (ntile 16-row tiles) x px/4 pixels (S 16-px sub-tiles); K chunk kch = 2*tkw bf16, tkw = min(32, 4096/px);
    K gains one zero chunk when the W port would be square (ch == k/2: aiehlc mis-sizes that window).
    ch == px is refused: aiehlc's link cannot place that instance's ping-pong windows (gemm_kacc_256x*x256,
    "cannot find free area ... in memory DMb"); :func:`plan_conv` takes another chunk."""
    if ch % 64 or px % 64 or k % 64:
        raise AiegraphError(_STAGE, f"gemm_kacc ({ch}, {k}, {px}): every dim must be a multiple of 64")
    if ch == px:
        raise AiegraphError(_STAGE, f"gemm_kacc ({ch}, {k}, {px}): ch == px does not link (DM placement)")
    tkw = min(32, 4096 // px)
    l1 = 2 * (16 * tkw * 4 + (px // 4) * tkw * 4 + 16 * (px // 4) * 4)
    if l1 > L1_PORTS or tkw < 4 or (k // 2) % tkw:
        raise AiegraphError(_STAGE, f"gemm_kacc ({ch}, {k}, {px}): port buffers {l1} B > {L1_PORTS} B or "
                            f"K chunk {2 * tkw} does not tile k", hint=f"a chunk MC <= 512 pixels")
    kp = k + 2 * tkw if ch == k // 2 else k
    return dict(ntile=ch // 64, S=px // 64, tkw=tkw, kch=2 * tkw, nkc=(kp // 2) // tkw, kp=kp, l1=l1)


def kacc_name(ch: int, k: int, px: int) -> str:
    """the launch callee of instance (ch, k, px): gemm_kacc<template args>"""
    import aiednn
    return aiednn.gemm_kacc_launch(ch, k, px, kacc_geom(ch, k, px))


def kernel_text() -> str:
    """The gemm_kacc template (``aiednn/gemm/gemm_kacc_kernel.cc``); aiehlc instantiates it per launch."""
    import aiednn
    return aiednn.template("gemm/gemm_kacc_kernel.cc")


@dataclass
class ConvPlan:
    cid: int
    call: int
    onnx: str
    symbol: str
    M: int
    N: int
    K: int          # dense K = C*KH*KW (depthwise expanded)
    dw: int
    act: int        # 1 = SiLU (sigmoid + multiply) fused after the per-channel adds
    sh: int
    sw: int
    ph: int
    pw: int
    Np: int
    Kk: int         # the kernel's K = split * ks
    Kp: int
    kch: int
    mc: int
    launches: int
    macs: int       # the conv's own MACs (ONNX)
    split: int = 1  # 3: fp32-accurate (W' = [Wh|Wh|Wl], X' = [Xh|Xl|Xh]), 1: plain bf16
    ks: int = 0     # K segment (>= K, multiple of 64)

    @property
    def shape(self) -> tuple:
        return (self.Np, self.Kk, self.mc)


def plan_conv(info: dict, mc: int = MC, split: int = 1) -> ConvPlan:
    """One matched conv group's ``g.info`` -> its launch plan (raises on a fused epilogue it cannot run).
    *split* 3: the GEMM runs on bf16 hi/lo parts of the fp32 weight and activation (3x K, ~fp32 accuracy;
    weights.bin must hold fp32, ``AIEGRAPH_WDTYPE=fp32``); 1: bf16 weight and activation."""
    if split not in (1, 3):
        raise AiegraphError(_STAGE, f"split={split}", hint="1 (bf16) or 3 (bf16 hi/lo, ~fp32)")
    sym = info["symbol"]
    m = _SYM.fullmatch(sym)
    if not m:
        raise AiegraphError(_STAGE, f"conv {info['onnx']}: fused TVM function {sym} is not conv [+add] "
                            f"[+sigmoid*multiply] [+add]")
    dw = int(info.get("groups", 1) > 1)
    if bool(m.group(1)) != bool(dw):
        raise AiegraphError(_STAGE, f"conv {info['onnx']}: groups={info.get('groups')} but TVM function {sym}")
    nadd = bool(m.group(2)) + bool(m.group(4))
    if info["nargs"] != 3 + nadd:
        raise AiegraphError(_STAGE, f"conv {info['onnx']}: {sym} has {info['nargs']} args, expected {3 + nadd}")
    N, M = int(info["N"]), int(info["M"])
    K = int(info["K"]) * (int(info["groups"]) if dw else 1)
    ks = max(_c64(K), -(-KK_MIN // split // 64) * 64)
    Kk = split * ks
    Np = max(_c64(N), NP_MIN_BIGK if Kk > KK_BIG else 64)
    if Np == mc:    # ch == px: aiehlc cannot place the instance's ping-pong windows in data memory
        mc = 2 * mc if 2 * mc <= 512 else mc // 2
    g = kacc_geom(Np, Kk, mc)
    return ConvPlan(cid=int(info["cid"]), call=int(info["call"]), onnx=info["onnx"], symbol=sym, M=M, N=N, K=K,
                    dw=dw, act=int(bool(m.group(3))), sh=int(info["sh"]), sw=int(info["sw"]), ph=int(info["ph"]),
                    pw=int(info["pw"]), Np=Np, Kk=Kk, Kp=g["kp"], kch=g["kch"], mc=mc, launches=-(-M // mc),
                    macs=int(info.get("macs", 0)), split=split, ks=ks)


def plan_convs(groups, mc: int = MC, split: int = 1) -> list:
    return [plan_conv(g.info, mc, split) for g in groups]


def _spec_exports(spec: str, kprefix: str) -> str:
    names = [f"{kprefix}_aie_{n}" for n in ("conv", "stats", "launch_stats", "host_split_site",
                                              "host_split_line", "split_stats")]
    ex = "".join(f'"export    {n}\\n"\n        ' for n in names)
    if '"out ' not in spec:
        raise AiegraphError(_STAGE, "carrier spec has no \"out\" line")
    return spec.replace('"out ', ex + '"out ', 1)


def write_kernel(kdst, spec: str, plans: list, *, kprefix: str, entry: str, warm: bool = True,
                 ctrlpkt: str = "off", selfcheck: bool = False) -> Path:
    """``<kdst>/<entry>.cc``: one gemm_kacc instance per distinct (Np, Kk, MC) of *plans* + the host glue
    (``<kprefix>_aie_conv`` and the stats ABI the generated board main reads) + the carrier entry
    (*spec* = ``emod.offload_prototype(layer).spec()``); ``<entry>_ref.c`` = the carrier's x86 model."""
    import aiednn
    if ctrlpkt not in ("off", "aot"):
        raise AiegraphError(_STAGE, f"ctrlpkt={ctrlpkt!r}", hint="'off' or 'aot'")
    kdst = Path(kdst)
    kdst.mkdir(parents=True, exist_ok=True)
    shapes = sorted({p.shape for p in plans})
    geo = {s: kacc_geom(*s) for s in shapes}
    head = aiednn.template("gemm/kacc_unit_head.cc").replace("@E@", entry).replace(
        "@SRC@", str(aiednn.path("gemm/gemm_kacc_kernel.cc"))).replace("@NSHAPES@", str(len(shapes)))
    if ctrlpkt == "aot":
        head = aiednn.template("gemm/ctrlpkt_aot_pragmas.h") + head
    kernels = kernel_text()
    disp = "\n".join(f"    if (N == {n} && K == {k} && M == {px}) {{\n        {kacc_name(n, k, px)}<<<mesh>>>(kw, kx, ky);"
                     f"\n        return 0;\n    }}" for n, k, px in shapes)
    board = aiednn.render("conv2d/kacc_chunk_board.cc", K=kprefix, WARM=int(warm), CTRLPKT=int(ctrlpkt == "aot"),
                          PART_C0=2 if ctrlpkt == "aot" else 3,
                          MAXW=max(n * geo[(n, k, px)]["kp"] for n, k, px in shapes),
                          MAXX=max(px * geo[(n, k, px)]["kp"] for n, k, px in shapes),
                          MAXY=max(n * px for n, k, px in shapes), DISPATCH=disp, SELFCHECK=int(selfcheck))
    tail = aiednn.render("conv2d/kacc_chunk_entry.cc", K=kprefix, E=entry, SPEC=_spec_exports(spec, kprefix))
    kcc = kdst / f"{entry}.cc"
    kcc.write_text(head + kernels + "\n" + board + aiednn.template("conv2d/kacc_chunk_conv.cc") + tail)
    (kdst / f"{entry}_ref.c").write_text(aiednn.template("conv2d/conv_bf16_carrier_ref.c").replace("@E@", entry))
    return kcc


def pick_carrier(plans: list, layers_dir) -> tuple:
    """The layer whose TVM function carries the AIE archive: a plain conv [+ adds] (what the carrier's x86
    model ``conv_bf16_carrier_ref.c`` computes: odd square kernel, pad (K-1)/2) whose every call is offloaded.
    Returns (layer index, plan)."""
    for p in plans:
        if p.act or p.dw or p.ph != p.pw or p.sh != p.sw:
            continue
        d = [q.parent for q in Path(layers_dir).glob("*/*.c") if f" {p.symbol}(void* args" in q.read_text()]
        if len(d) == 1:
            return int(d[0].name.split("_")[0]), p
    raise AiegraphError(_STAGE, "no carrier layer: no offloaded conv without activation / depthwise")


def offload_convs(emod, groups, plans, kcc, *, kprefix: str, log=print) -> int:
    """One aiegraph group step ``<kprefix>_aie_conv`` per conv group (the "args" ABI: the TVM call's own
    packed arguments, geometry + plan as consts). Returns the planned launches."""
    import aiegraph as ag
    name = f"{kprefix}_aie_conv"
    planned = 0
    for g, p in zip(groups, plans):
        consts = dict(cid=p.cid, sh=p.sh, sw=p.sw, ph=p.ph, pw=p.pw, act=p.act, dw=p.dw, np=p.Np, kk=p.Kk, kp=p.Kp,
                      mc=p.mc, kch=p.kch, split=p.split, ks=p.ks)
        st = ag.step(name, reads=g.x, writes=g.y, consts=consts,
                     call=("cid", "args", "nargs", "sh", "sw", "ph", "pw", "act", "dw", "np", "kk", "kp", "mc", "kch",
                           "split", "ks"),
                     wrap=("UP_AIE_BEGIN", "UP_AIE_END"), tag=f"aie {name} {p.cid}")
        emod.offload_group(g, str(kcc), steps=[st], launch=dict(aie=p.launches))
        planned += p.launches
    log(f"aiegraph offload_group: {len(plans)} conv groups -> {name} ({sum(p.dw for p in plans)} depthwise, "
        f"{sum(p.act for p in plans)} with SiLU), {len(set(p.shape for p in plans))} gemm_kacc instances, "
        f"{planned} launches planned")
    return planned
