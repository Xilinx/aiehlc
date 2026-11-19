###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.cnn -- every Conv / weighted Gemm of a CNN on the AIE as bf16 gemm_kacc launches.

    from aiegraph import cnn
    res = cnn.offload_all(emod, onnx_model, out_dir, warm=True, ctrlpkt="aot-kco", log=print)

Generic over the model (no per-model code):

  match     ``autopart.inventory`` -> one partition row per Conv (group 1) and weighted Gemm / MatMul;
            ``emod.match("conv")`` / ``emod.match("linear")`` pair each row with its TVM call.
  shapes    each layer GEMM (N = out channels, K = C*KH*KW, M = pixels) runs on ONE gemm_kacc instance
            (Ng, Kg, PX) = (ceil64 N, ceil64 K, launch pixel width): :func:`instance`. M is walked in
            ceil(M / PX) launches; padded rows / columns are zeros (aiednn/conv2d/kacc_gemm_pad_host.cc).
  kernel    one aiek source: the gemm_kacc instances (aiednn/gemm/gemm_kacc_kernel.cc), the kacc host glue
            (aiednn/conv2d) and the padded host entries ``aiednn_conv`` / ``aiednn_dense``.
  groups    one aiegraph one-call group per layer with an "args" ABI step; its weight goes into weights.bin
            as natural [N][K] bf16 (read in place, never expanded); fused bias / residual adds and ReLU (from
            the TVM function name) are applied in the scatter.
  archive   built by ``emod.offload`` on a carrier layer (a TVM conv function whose every call site is offloaded
            and whose x86 reference model applies: no fused activation), group-level x86 check binding.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from aiednn import gemm_kacc_launch, path as _tplpath, template as _tpl

from .errors import AiegraphError

ENTRY = "aiek_cnn"
STEP_CONV, STEP_DENSE = "aiednn_conv", "aiednn_dense"
PX_SMALL, PX_WIDE = 64, 256          # launch pixel widths (the gemm_kacc walks proven on the board)
K_SMALL = 128                        # instances with K <= this are merged into a larger-K one (see merge_small_k)
__all__ = ["ENTRY", "instance", "merge_small_k", "kacc_geom", "kernel_source", "plan_layers", "offload_all"]


def _c64(v: int) -> int:
    return -(-v // 64) * 64


def instance(N: int, K: int, M: int) -> tuple:
    """(Ng, Kg, PX) of the gemm_kacc instance a layer GEMM N x K x M runs on. Square ports (Ng == PX: the a and
    b blocks walk the same rows) are avoided -- aiehlc cannot place their buffers (gemm_kacc_256x1152x256: no
    4096 B area in DMb at 0x78000) -- by taking the other launch width."""
    ng, px = _c64(N), PX_SMALL if M <= PX_SMALL else PX_WIDE
    if ng == px:
        px = PX_WIDE if px == PX_SMALL else PX_SMALL
    return ng, _c64(K), px


def merge_small_k(insts: list) -> dict:
    """{instance: instance used}: an instance with K <= K_SMALL runs on the smallest larger-K instance of the same
    (Ng, PX) when the plan has one (K zero-padded). Small-K walks ran on the board but computed wrong:
    gemm_kacc_128x64x256 / 128x128x256 raw GEMM cos 0.56-0.84 against the A78 GEMM of the same bf16 operands
    (AIEDNN_SELFCHECK debug build), while every instance with K >= 192 had cos 1.000000."""
    out = {}
    for ng, kg, px in insts:
        big = sorted(k for n, k, p in insts if (n, p) == (ng, px) and k > kg)
        out[(ng, kg, px)] = (ng, big[0], px) if kg <= K_SMALL and big else (ng, kg, px)
    return out


def kacc_geom(ch: int, k: int, px: int) -> dict:
    """Walk of one gemm_kacc launch on the 4x4 mesh (same rule as the host's ap_geom): core (pm, pn) owns
    ch/4 channels x px/4 pixels; K chunk kch = 2*tkw bf16 with tkw = min(32, 4096/px); when the ping-pong
    port buffers do not fit 29696 B the launch runs transposed (tr: X is the walked a port)."""
    if ch % 64 or px % 64 or k % 64:
        raise AiegraphError("cnn", f"gemm_kacc instance {ch}x{k}x{px}: every dim must be a multiple of 64")
    tkw = min(32, 4096 // px)
    tr = 2 * (16 * tkw * 4 + (px // 4) * tkw * 4 + 16 * (px // 4) * 4) > 29696
    if tr:
        tkw = 32
        while (ch // 4) * tkw > 1024 or (k // 2) % tkw:
            tkw //= 2
    if tkw < 4 or (k // 2) % tkw:
        raise AiegraphError("cnn", f"gemm_kacc instance {ch}x{k}x{px}: no K chunk (tkw={tkw})")
    kp = k + 2 * tkw if ch == k // 2 else k
    rows, cols = (px, ch) if tr else (ch, px)
    l1 = 2 * (16 * tkw * 4 + (cols // 4) * tkw * 4 + 16 * (cols // 4) * 4)
    if l1 > 29696:
        raise AiegraphError("cnn", f"gemm_kacc instance {ch}x{k}x{px}: port buffers {l1} B > 29696")
    return dict(ntile=rows // 64, S=cols // 64, tkw=tkw, kch=2 * tkw, nkc=(kp // 2) // tkw, kp=kp, tr=int(tr),
                arows=rows, brows=cols, l1=l1)


def _kname(ch, k, px):
    """the launch callee of instance (ch, k, px): gemm_kacc<template args> (aiednn/gemm/gemm_kacc_kernel.cc)"""
    return gemm_kacc_launch(ch, k, px, kacc_geom(ch, k, px))


# the kacc host glue's C-linkage entries (stats / split / legacy conv entry) keep the template's own prefix
_EXPORT_RE = re.compile(r'^extern "C" (?:int|void) (\w+_aie_(?:conv|stats|launch_stats|host_split_site|host_split_line|'
                        r'split_stats))\(', re.M)
_ENTRY_RE = re.compile(r'^extern "C" int \w+_aie_conv\(int cid', re.M)
_DEAD = ("static float g_uc_acc[UC_MAX_ACC] __attribute__((aligned(64))); // .bss, never malloc\n",
         "static int32_t *g_uc_a = nullptr; // port a: bf16 [64][128]\n",
         "static int32_t *g_uc_b = nullptr; // port b: bf16 [UC_BR][128]\n",
         "static int32_t *g_uc_c = nullptr; // port c: f32  [64][UC_CW]\n")


def kernel_source(spec: str, shapes: list, *, warm: bool = True, ctrlpkt: str = "off", entry: str = ENTRY) -> str:
    """The aiek source: ``ctrlpkt`` pragmas, the gemm_kacc instances of *shapes* ``[(Ng, Kg, PX)]``, the kacc host
    glue (launch dispatch per instance) and the padded entries; *spec* = the carrier prototype's AIEK_SPEC."""
    if ctrlpkt not in ("off", "aot", "aot-kco"):
        raise AiegraphError("cnn", f"ctrlpkt={ctrlpkt!r}", hint="off, aot or aot-kco")
    shapes = sorted(set(shapes))
    head = (_tpl("gemm/kacc_unit_head.cc").replace("@E@", entry).replace("@SRC@", str(_tplpath("gemm/gemm_kacc_kernel.cc")))
            .replace("@NSHAPES@", str(len(shapes))))
    if ctrlpkt != "off":
        head = _tpl("gemm/ctrlpkt_aot_pragmas.h") + (_tpl("gemm/ctrlpkt_kco_pragma.h") if ctrlpkt == "aot-kco"
                                                     else "") + head
    for s in shapes:
        kacc_geom(*s)   # every instance must have a walk
    ktext = _tpl("gemm/gemm_kacc_kernel.cc")   # the template; one instance per launch below, same TU
    from .boardmain import kt_select
    host = kt_select(_tpl("conv2d/tile64_conv_host.cc") + _tpl("elementwise/epilogue_bias_residual.cc")
                     + _tpl("conv2d/tile64_conv_entry.cc"), False)
    disp = "\n".join("    if (N == %d && K == %d && M == %d) {\n        %s<<<mesh>>>(%s, ky);\n        return 0;\n    }"
                     % (ch, k, px, _kname(ch, k, px), "kx, kw" if kacc_geom(ch, k, px)["tr"] else "kw, kx")
                     for ch, k, px in shapes)
    kacc = (_tpl("conv2d/kacc_conv_host_head.cc") + _tpl("layout/neon_tr4_f32.cc") + _tpl("conv2d/kacc_conv_host.cc"))
    kacc = (kacc.replace("@DISPATCH@", disp)
            .replace("@MAXW@", str(max(ch * kacc_geom(ch, k, px)["kp"] for ch, k, px in shapes)))
            .replace("@MAXX@", str(max(px * kacc_geom(ch, k, px)["kp"] for ch, k, px in shapes)))
            .replace("@MAXY@", str(max(ch * px for ch, k, px in shapes))))
    i0, i1 = host.index("static int uc_conv("), _ENTRY_RE.search(host).start()
    host = host[:i0] + kacc + host[i1:]
    j = _ENTRY_RE.search(host).start()
    host = host[:j] + _tpl("conv2d/kacc_gemm_pad_host.cc").lstrip("\n") + "\n" + host[j:]
    a0, a1 = host.index("static int uc_alloc(void) {"), host.index("static long uc_numel(")
    host = host[:a0] + host[a1:]
    for dead in _DEAD:
        if host.count(dead) != 1:
            raise AiegraphError("cnn", f"host template changed: {dead.strip()}")
        host = host.replace(dead, "")
    exports = sorted(set(_EXPORT_RE.findall(host))) + [STEP_CONV, STEP_DENSE]
    if len(exports) != 8:
        raise AiegraphError("cnn", f"host glue exports {exports}: expected the 6 kacc stats / split entries")
    ex = "".join(f'"export    {e}\\n"\n        ' for e in exports)
    if spec.count('"out ') < 1:
        raise AiegraphError("cnn", "carrier AIEK_SPEC has no out line")
    spec = spec.replace('"out ', ex + '"out ', 1)
    on = ctrlpkt != "off"
    text = head + ktext + "\n" + (host.replace("@MAXACC@", "1").replace("@WARM@", "1" if warm else "0")
                                  .replace("@CTRLPKT@", "1" if on else "0").replace("@PART_C0@", "2" if on else "3")
                                  .replace("@E@", entry).replace("@SPEC@", spec))
    left = sorted(set(re.findall(r"@[A-Z][A-Z0-9_]*@", text)))
    if left:
        raise AiegraphError("cnn", f"kernel source has unfilled parameters {left}")
    return text


def _rows(model) -> tuple:
    """(partition dict, ONNX rows skipped) -- every group-1 Conv and weighted Gemm / MatMul."""
    from .autopart import inventory
    rows, skip = [], []
    for i, r in enumerate(inventory(model)):
        if r["cls"] == "attn" or (r["op"] == "Conv" and r["why"]):
            skip.append(r)
            continue
        rows.append(dict(idx=i, node=r["node"], weight=r["weight"], cls="conv" if r["op"] == "Conv" else "linear",
                         mkn=[[r["m"], r["k"], r["n"]]], macs=r["macs"]))
    return {"layers": rows}, skip


def plan_layers(emod, model) -> dict:
    """Match every offloadable layer to its TVM call and pick its instance. Returns dict(conv=[...], linear=[...],
    shapes, total_macs, skipped) -- each layer = ``g.info`` + group, instance, launches, act."""
    from .autopart import inventory
    part, skip = _rows(model)
    conv = emod.match("conv", partition=part, model=model, groups=1)
    lin = emod.match("linear", partition=part, model=model) if any(L["cls"] == "linear" for L in part["layers"]) \
        else []
    out = {"conv": [], "linear": []}
    for kind, gs in (("conv", conv), ("linear", lin)):
        for g in gs:
            c = dict(g.info, group=g)
            c["inst"] = instance(c["N"], c["K"], c["M"])
            c["act"] = 1 if re.search(r"(^|_)relu(_|$|\d)", c["func"]) else 0
            if re.search(r"(sigmoid|tanh|clip|silu|gelu|leaky)", c["func"]):
                raise AiegraphError("cnn", f"{c['onnx']}: TVM function {c['func']} fuses an activation the "
                                    "padded host does not apply", hint="only bias / residual adds and ReLU")
            out[kind].append(c)
    merged = merge_small_k(sorted({c["inst"] for c in out["conv"] + out["linear"]}))
    for c in out["conv"] + out["linear"]:
        c["inst"] = merged[c["inst"]]
        c["launches"] = -(-c["M"] // c["inst"][2])
    small = sorted({i for i, j in merged.items() if i == j and i[1] <= K_SMALL})
    if small:
        raise AiegraphError("cnn", f"gemm_kacc instances {small} have K <= {K_SMALL} and no larger-K instance of the "
                            "same (N, PX) to run on", hint="small-K walks compute wrong on the board")
    out["shapes"] = sorted({c["inst"] for c in out["conv"] + out["linear"]})
    out["total_macs"] = sum(r["macs"] for r in inventory(model))
    out["n_conv"] = sum(1 for n in model.graph.node if n.op_type == "Conv")
    out["n_gemm"] = sum(1 for r in inventory(model) if r["cls"] != "attn" and r["op"] != "Conv")
    out["skipped"] = skip
    return out


def _carrier(plan: dict, cg, layers_dir: Path):
    """A TVM conv function all of whose call sites are offloaded, without a fused activation (its x86 reference
    model = conv + adds) and with "same" padding / stride H/OH (what the carrier reference assumes)."""
    sel = {c["node"] for c in plan["conv"]}
    from .callsite import all_conv_calls
    calls = all_conv_calls(cg)
    for c in plan["conv"]:
        if c["act"] or c["ph"] != (c["wshape"][2] - 1) // 2 or c["pw"] != (c["wshape"][3] - 1) // 2:
            continue
        if c["xshape"][2] % c["yshape"][2] or c["xshape"][2] // c["yshape"][2] != c["sh"]:
            continue
        if not all(x["node"] in sel for x in calls if x["symbol"] == c["symbol"]):
            continue
        d = [p.parent for p in layers_dir.glob("*/*.c") if f" {c['symbol']}(void* args" in p.read_text()]
        if len(d) == 1:
            return int(d[0].name.split("_")[0]), c
    raise AiegraphError("cnn", "no carrier layer: every offloaded conv function has a fused activation, "
                        "CPU call sites or a padding the reference model does not cover")


def _natural(kind: str):
    """bf16 TVM weight -> natural [N][K] (k = c*KH*KW + kh*KW + kw), a permutation."""
    if kind == "conv":
        return lambda w: w.transpose(0, 5, 1, 4, 2, 3).reshape(w.shape[0] * w.shape[5], -1)
    return lambda w: w.transpose(0, 2, 1).reshape(w.shape[0] * w.shape[2], w.shape[1])


def offload_all(emod, model, out, *, warm: bool = True, ctrlpkt: str = "off", verbose: bool = False,
                log=print) -> dict:
    """Plan, write the aiek source (``<out>/aiek_src/aiek_cnn.cc``), place every layer as an aiegraph group step,
    register the driver glue and build the archive on the carrier (binding group-level x86 check). Returns
    dict(defs=<board-main header defines>, conv=(n, of), gemm=(n, of), mac_share=<% of the model's GEMM MACs>,
    launches=<planned>, shapes, layers)."""
    import aiegraph as ag
    from . import boardmain as bm
    out = Path(out)
    plan = plan_layers(emod, model)
    layers = plan["conv"] + plan["linear"]
    if not layers:
        raise AiegraphError("cnn", "no Conv / weighted Gemm to offload")
    work = Path(emod.workdir)
    lidx, car = _carrier(plan, emod.call_graph(), work / "layers")
    log(f"[cnn] {len(plan['conv'])} conv + {len(plan['linear'])} gemm layers, {len(plan['shapes'])} gemm_kacc "
        f"instances {['%dx%dx%d' % s for s in plan['shapes']]}; carrier layer {lidx} ({car['func']})")
    p = emod.offload_prototype(lidx, entry=ENTRY)
    kdst = out / "aiek_src"
    kdst.mkdir(parents=True, exist_ok=True)
    kcc = kdst / f"{ENTRY}.cc"
    kcc.write_text(kernel_source(p.spec(), plan["shapes"], warm=warm, ctrlpkt=ctrlpkt))
    (kdst / f"{ENTRY}_ref.c").write_text(_tpl("conv2d/conv_bf16_carrier_ref.c").replace("@E@", ENTRY))
    for c in layers:
        conv = c["kind"] == "conv"
        name = STEP_CONV if conv else STEP_DENSE
        consts = dict(cid=c["cid"], sh=c["sh"], sw=c["sw"], ph=c["ph"], pw=c["pw"], act=c["act"], px=c["inst"][2],
                      kg=c["inst"][1])
        if not conv:
            consts = dict(cid=c["cid"], act=c["act"], px=c["inst"][2], kg=c["inst"][1])
        g = c["group"]
        st = ag.step(name, reads=g.x, writes=g.y, weights=g.w.as_bf16(layout=_natural(c["kind"])), consts=consts,
                     call=("cid", "args", "nargs", "wbf") + tuple(k for k in consts if k != "cid"),
                     wrap=("UP_AIE_BEGIN", "UP_AIE_END"), tag=f"aie {name} {c['cid']}")
        emod.offload_group(g, kcc, steps=[st], launch=dict(aie=c["launches"]))
        log(f"  aie {c['kind']} {c['cid']:3d} {c['onnx']} -> {c['func']} M={c['M']} N={c['N']} K={c['K']} "
            f"s={c['sh']} p={c['ph']} act={c['act']} inst={'%dx%dx%d' % c['inst']} launches={c['launches']}")
    emod.driver_template(**bm.driver_glue("", attn=False))
    rep = emod.offload(f"layer{lidx}", str(kcc), check="group", verbose=verbose)
    log(f"[cnn] offload: {rep}")
    lp = json.loads((work / "layers" / "partition.json").read_text())
    if not any(L.get("target") == "aie" and L.get("archive") for L in lp["layers"]):
        raise AiegraphError("cnn", "carrier layer has no AIE archive in layers/partition.json")
    planned = sum(c["launches"] for c in layers)
    macs = sum(c["macs"] for c in layers)
    defs = dict(LAUNCH_WARM=1, LAUNCH_OPTS=bm.LAUNCH_OPTS_WARM) if warm else {}
    defs.update(CTRLPKT="aot" if ctrlpkt != "off" else "off", AIE_PLANNED_LAUNCHES=planned,
                AIE_PLANNED_CALLS=len(layers), TOTAL_GEMM_MACS=(int(plan["total_macs"]), "ULL"))
    return dict(defs=defs, conv=(len(plan["conv"]), plan["n_conv"]), gemm=(len(plan["linear"]), plan["n_gemm"]),
                mac_share=100.0 * macs / max(1, plan["total_macs"]), launches=planned, shapes=plan["shapes"],
                layers=layers, kernel=kcc)


def check_driver(drv: str, res: dict) -> tuple:
    """Call sites of the padded entries in the linked graph_driver.c == the planned conv / gemm layers."""
    got = (drv.count(f"    {STEP_CONV}("), drv.count(f"    {STEP_DENSE}("))
    want = (res["conv"][0], res["gemm"][0])
    if got != want:
        raise AiegraphError("cnn", f"linked driver has {got} {STEP_CONV}/{STEP_DENSE} call sites, expected {want}")
    return got
