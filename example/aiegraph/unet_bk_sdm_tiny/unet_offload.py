"""The UNet's AIE offload declarations for build_unet.py (unet-m4 P4: moved out of build_unet.py).

offload(a, emod, out) maps the partition's conv / linear / attention layers to aiegraph groups (match /
offload_group / merge_groups / fold matches), places the flash-attention groups, registers the driver glue
(aiegraph.boardmain.driver_glue) and builds the AIE archive on the carrier layer. It returns the header
defines of the board main and the expected (conv, linear, attn) call-site counts that check_driver()
verifies in the linked graph_driver.c. No C text here: kernels / host glue come from src/aiednn, the board
main and the PROF glue from aiegraph.boardmain.
"""
import json
import re
import sys
from pathlib import Path


def log(msg):
    print(f"[build_unet] {msg}", flush=True)


def keep_layout_step(ua, onnx_path, part: dict, layers: list, out: Path):
    """GAP-B: partition rows with ``keep_layout`` -- the output stays in the AIE layout, Transposes / Reshapes
    to its AIE consumers are labels only (aiegraph.partition.plan_keep_layout). Tags ``layers``, writes
    <out>/keep_layout.json, prints KEEP_LAYOUT. None when no row has the field (build unchanged)."""
    if not any(L.get("keep_layout") for L in part["layers"]):
        return None
    keep = ua.keep_layout_plan(onnx_path, part, layers)
    out.mkdir(parents=True, exist_ok=True)
    (out / "keep_layout.json").write_text(json.dumps(keep, indent=1))
    log(f"KEEP_LAYOUT edges={keep['kept']} label_ops={keep['label_ops']} "
        f"consumers={sum(len(e['consumers']) for e in keep['edges'])} ({out / 'keep_layout.json'})")
    return keep


def fa_offload_groups(emod, att: list, kcc: Path, out: Path) -> int:
    """S3 (design 2026-11-16 §5.2): every attention group = one aiegraph group offload with the four FA steps
    (fa_attn.FA_STEPS, checked against their AIEK_STEP declarations); aiegraph places them, removes the other
    calls (QK bmm, softmax, O merge) after its liveness / external-reader / early-write checks, and records the
    group in layers/partition.json, from which arm_build emits the driver."""
    import aiegraph as ag
    import fa_attn
    from aiegraph.group import reset_groups
    stepsrc = out / "aiek_src" / "fa_steps.cc"
    stepsrc.write_text(fa_attn.fa_steps())
    specs = ag.load_steps(stepsrc)
    reset_groups(emod.workdir)
    by = {}
    for c in att:
        by.setdefault(c["ag_group"].name, {})[c["role"]] = c
    ncalls = 0
    for name, d in by.items():
        qk, pv = d["qk"], d["pv"]
        g = qk["ag_group"]
        if qk["fa"] != pv["fa"] or qk["fa"] != g.cls:
            raise SystemExit(f"{name}: FA class {qk['fa']}/{pv['fa']} != aiegraph class {g.cls}")
        steps = fa_attn.fa_group_steps(ag, g, qk["cid"], pv["cid"], qk["fa"])
        ag.check_steps(specs, steps)
        plan = emod.offload_group(g, source=str(kcc), steps=steps, emit=fa_attn.fa_step_emit(g))
        want = {"fa_step_pack_q": g.q_call, "fa_step_pack_k": g.k_call, "fa_step_pack_v": g.v_call,
                "fa_step_run_o": g.pv}
        if plan.placement != want or sorted(plan.removed) != sorted([g.qk, g.softmax, g.merge]):
            raise SystemExit(f"{name}: placement {plan.placement} removed {plan.removed}, expected {want} "
                             f"removed {[g.qk, g.softmax, g.merge]}")
        for c in (qk, pv):
            c["fa_steps"] = True
        ncalls += len(g.calls)
    log(f"FA_GROUPS groups={len(by)} calls={ncalls} steps={4 * len(by)} removed={3 * len(by)} "
        f"(aiegraph offload_group, layers/partition.json)")
    return len(by)


def offload(a, emod, out: Path):
    """--aie conv[,linear[,attn]]: map partition layers to aiegraph groups, build the AIE archive on a
    carrier layer. Returns (board-main header defines, expected (conv, linear, attn) call sites)."""
    import aiegraph._env as env
    from aiegraph import boardmain as bm
    env.frontend("")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import unet_aie as ua
    import aiegraph as ag
    import frontend.tvmrelay.arm_build as ab
    import onnx
    from aiegraph.callsite import all_conv_calls
    part = json.loads(a.partition.resolve().read_text())
    work = Path(emod.workdir)
    # S1 (unet-m3): aiegraph matches every partition conv / linear row to its TVM call -- one group
    # of one call each (g.x / g.y / g.w / g.info); the steps are placed by emod.offload_group below
    onnx_model = onnx.load(str(a.onnx.resolve()))
    cgroups = emod.match("conv", partition=part, model=onnx_model)
    folded = [g.info["onnx"] for g in cgroups if g.info["folded"]]
    assert all("proj_in" in x for x in folded), f"weight mismatch outside proj_in: {folded}"
    sel = [dict(g.info, group=g) for g in cgroups]
    for c in sel:
        launches = ua.ceil64(c["M"]) * ua.ceil64(c["N"]) * ua.ceil64(c["K"])
        L = next(L for L in part["layers"] if L["idx"] == c["cid"])
        assert launches == L["launches"], (c["onnx"], launches, L["launches"])
        c["launches"] = launches
    convs = all_conv_calls(emod.call_graph())
    log(f"aiegraph match('conv'): {len(sel)} partition conv layers -> one-call groups "
        f"({len(convs)} TVM conv calls, {len(folded)} with a TVM-folded weight: proj_in)")
    nlin_part = sum(1 for L in part["layers"] if L.get("cls") == "linear")
    lin = []
    if "linear" in a.aie.split(","):
        # PDN: the partition's linear layers (TVM nn_contrib_dense_pack) -> aiednn_aie_linear, gemm_kacc
        lin = [dict(g.info, group=g) for g in emod.match("linear", partition=part, model=onnx_model)]
        for c in lin:
            c["S"], c["pxs"] = ua.kacc_split(c["N"], c["K"], -(-c["M"] // 64) * 64)
            c["launches"] = c["S"]
        log(f"aiegraph match('linear'): {len(lin)} partition linear layers -> one-call groups "
            f"(weights equal by value), {sum(c['S'] for c in lin)} gemm_kacc launches")
        if len(lin) != nlin_part:
            raise SystemExit(f"mapped {len(lin)} linear layers, partition has {nlin_part}")
    elif nlin_part:
        raise SystemExit(f"partition has {nlin_part} linear layers: build with --aie conv,linear")
    del onnx_model
    if lin and not a.kincore:
        raise SystemExit("--aie conv,linear needs --kincore (gemm_kacc)")
    att = []
    if "attn" in a.aie.split(","):
        # GATT: the partition's attention bmm layers (TVM nn_batch_matmul) -> aiednn_aie_attn, gemm_kacc + runtime pack
        natt_part = sum(1 for L in part["layers"] if L.get("cls") == "attn")
        if not natt_part:
            raise SystemExit("--aie conv,linear,attn: partition has no attn layers (autopart --mode attn)")
        if not a.kincore:
            raise SystemExit("--aie conv,linear,attn needs --kincore (gemm_kacc)")
        att = ua.map_attn(part, work, onnx=a.onnx.resolve())   # S3: aiegraph attention matcher
        if len(att) != natt_part:
            raise SystemExit(f"mapped {len(att)} attn layers, partition has {natt_part}")
    fa = []
    if a.attn_fa:
        # GFA (P2): each attention group = ONE flash-attention (online softmax) on the AIE; TVM softmax removed
        if not att:
            raise SystemExit("--attn-fa needs --aie conv,linear,attn")
        import fa_attn
        fa = fa_attn.fa_classes(att)
        for c in att:
            c["launches"] = fa[c["fa"]]["launches"] if c["role"] == "pv" else 0
        for k in fa:
            log(f"GFA class {k['idx']} {k['name']} split={k['split']} nkv={k['nkv']} kvl={k['kvl']} units={k['units']} "
                f"launches/site={k['launches']}")
    keep = keep_layout_step(ua, a.onnx.resolve(), part, sel + lin + att, out)
    if a.kincore:
        # P4bg: one launch per layer from kplan.py (K in the core) -- the partition.json N (64x64x64
        # tile launches) does not apply on this path
        import kplan
        if a.kincore_layers != "all":
            # P4bp: probe subset -- unselected layers keep their TVM C call site (no aiegraph group for them)
            want = sorted({int(x) for x in a.kincore_layers.split(",") if x.strip()})
            bad = [i for i in want if not 0 <= i < len(sel)]
            if not want or bad:
                raise SystemExit(f"--kincore-layers {a.kincore_layers}: indices must be in 0..{len(sel) - 1}")
            sel = [sel[i] for i in want]
            log(f"KINCORE subset --kincore-layers {','.join(map(str, want))} -> cids {[c['cid'] for c in sel]}")
        for c in sel:
            kp = kplan.plan(c["M"], c["K"], c["N"])
            if not kp["legal"] or kp["launches"] != 1:
                raise SystemExit(f"kplan: layer {c['cid']} M={c['M']} K={c['K']} N={c['N']} "
                                 f"legal={kp['legal']} launches={kp['launches']} why={kp['why']}")
            c["launches"] = kp["launches"]
        convsel = sel
        sel = sel + lin
        planned = sum(c["launches"] for c in sel + att)
        shapes = sorted({(c["N"], c["K"], c["M"]) for c in convsel} | {(c["N"], c["K"], c["pxs"]) for c in lin} |
                        (set() if fa else set(ua.attn_shapes(att))))
        log(f"KINCORE layers={len(sel) + len(att)} launches_planned={planned} kernel=bf16_gemm_kacc; "
            f"shapes={len(shapes)} (old 64x64x64 launches {part['N']})")
    else:
        convsel = sel
        planned = sum(c["launches"] for c in sel)
        if planned != int(part["N"]):
            raise SystemExit(f"planned launches {planned} != partition.json N {part['N']}")
    if att:
        log(f"aie_launches_planned={planned} (partition.json N={part['N']}, {len(sel) + len(att)} layers: "
            f"{len(sel) - len(lin)} conv + {len(lin)} linear + {len(att)} attn, "
            f"{sum(c['macs'] for c in sel + att)} MACs of {part['total_gemm_macs']} GEMM MACs)")
    else:
        log(f"aie_launches_planned={planned} (partition.json N={part['N']}, {len(sel)} layers: "
            f"{len(sel) - len(lin)} conv + {len(lin)} linear, "
            f"{sum(c['macs'] for c in sel)} MACs of {part['total_gemm_macs']} GEMM MACs)")
    for c in sel:
        if c.get("kind") == "linear":
            log(f"  aie linear {c['cid']:3d} {c['onnx']} -> node {c['node']} {c['func']} "
                f"M={c['M']} N={c['N']} K={c['K']} px_split={c['S']}x{c['pxs']} launches={c['launches']}")
        else:
            log(f"  aie conv {c['cid']:2d} {c['onnx']} -> node {c['node']} {c['func']} "
                f"M={c['M']} N={c['N']} K={c['K']} s={c['sh']} p={c['ph']} launches={c['launches']}")
    for c in att:
        g = c["geo"]
        log(f"  aie attn {c['cid']:3d} {c['role']} g{c['group']} {c['onnx']} -> node {c['node']} {c['func']} "
            f"H={g['H']} M={c['M']} K={c['K']} N={c['N']} swap={g['swap']} ch={g['chs']} k={g['Kp']} px={g['pxs']} "
            f"S={g['S']} launches={c['launches']}")
    ua.check_data_arg0(work / "layers", convsel)
    lidx, car = ua.pick_carrier(convsel, convs, work / "layers")
    log(f"carrier layer {lidx} ({car['func']}): aiegraph offload builds the gemm_bf16 archive on it")
    p = emod.offload_prototype(lidx, entry=ua.ENTRY)
    if a.kincore:
        kcc = ua.write_kernel_kacc(out / "aiek_src", p.spec(), shapes, warm=(a.launch_opt == "warm"),
                                   ctrlpkt=a.ctrlpkt,
                                   lin=[(c["N"], c["K"], -(-c["M"] // 64) * 64, c["pxs"]) for c in lin],
                                   **({"att": att} if att else {}), **({"fa": fa} if fa else {}))
    else:
        kcc = ua.write_kernel(out / "aiek_src", p.spec(), max(c["M"] * c["N"] for c in sel),
                              warm=(a.launch_opt == "warm"), ctrlpkt=a.ctrlpkt, ktimer=a.ktimer)
    if a.ctrlpkt == "aot":
        log(f"ctrlpkt=aot pragmas={ua.CTRLPKT_NPRAGMAS}")
    log(f"kernel source {kcc}")
    if fa:
        fa_offload_groups(emod, att, kcc, out)

    # S1: conv / linear call sites through aiegraph -- one-call groups, the "args" step ABI, bf16 weights
    # pre-laid out for gemm_kacc in weights.bin (aiegraph drops their graph_bf16_expand)
    if not a.kincore:
        raise SystemExit("aiegraph conv/linear groups need --kincore (kacc weight layout)")
    # P3 (unet-m4): --keep-layout = fold declarations -- match("conv+layout") (the conv scatter writes the
    # NCHW / NHWC its reader wants), match("linear+layout") (residual into the linear scatter, the next conv
    # reads [M][C] as NHWC), match("dense+silu") (SiLU scale on the dense input, weight pre-packed); every
    # fold is checked on graph.json + opsem and offloaded as an aiegraph group (no driver text pass)
    from aiegraph.fold import step_io
    by_call = {c["group"].calls[0]: c for c in sel}
    groups = [c["group"] for c in sel]
    if a.keep_layout:
        convg = [c["group"] for c in sel if c["kind"] != "linear"]
        f1 = emod.match("conv+layout", base=convg, log=log)
        f2 = emod.match("linear+layout", base=[c["group"] for c in lin], convs=convg, log=log)
        groups += f1 + f2
        log(f"aiegraph match: conv+layout={len(f1)} linear+layout={len(f2)} (folds, removed calls "
            f"{sum(len(g.info['fold'].removed) for g in f1 + f2)})")
        if keep is not None:   # partition keep_layout edges: each producer must be folded
            folded = {k for g in f1 + f2 for k in g.calls}
            miss = [e["node"] for e in keep["edges"] if next(c for c in sel + lin + att
                                                              if c.get("onnx") == e["node"])["call"] not in folded]
            if miss:
                raise SystemExit(f"planned keep_layout edges not realized by a fold: {miss}")
    cg = emod.call_graph()
    nw = ng = 0
    for g in emod.merge_groups(groups):
        steps = []
        for k in g.calls:
            if k not in by_call:
                continue        # a folded layout call: removed
            c = by_call[k]
            rd, wr, args = step_io(cg, g, k) if g.info.get("fold") else (c["group"].x, c["group"].y, None)
            if c["kind"] == "linear":
                steps.append(ag.step("aiednn_aie_linear", reads=rd, writes=wr, args=args, at=k if args else None,
                                     weights=c["group"].w.as_bf16(layout=lambda w, c=c: ua.kacc_dense_wlayout(
                                         w, c["N"], c["K"], c["pxs"])),
                                     consts=dict(cid=c["cid"], S=c["S"]), call=("cid", "args", "nargs", "wbf", "S"),
                                     wrap=("UP_AIE_BEGIN", "UP_AIE_END"), tag=f"aie aiednn_aie_lin {c['cid']}"))
            else:
                steps.append(ag.step("aiednn_aie_conv", reads=rd, writes=wr, args=args, at=k if args else None,
                                     weights=c["group"].w.as_bf16(layout=lambda w, c=c: ua.kacc_wlayout(
                                         w, c["N"], c["K"], c["M"])),
                                     consts=dict(cid=c["cid"], sh=c["sh"], sw=c["sw"], ph=c["ph"], pw=c["pw"]),
                                     call=("cid", "args", "nargs", "wbf", "sh", "sw", "ph", "pw"),
                                     wrap=("UP_AIE_BEGIN", "UP_AIE_END"), tag=f"aie aiednn_aie_conv {c['cid']}"))
        plan = emod.offload_group(g, kcc, steps=steps,
                                  launch=dict(aie=sum(by_call[k]["launches"] for k in g.calls if k in by_call)))
        nw += sum(s["weights"]["nbytes"] for s in plan.steps)
        ng += 1
    log(f"aiegraph offload_group: {len(sel)} call steps in {ng} groups ({len(sel) - len(lin)} aiednn_aie_conv + "
        f"{len(lin)} aiednn_aie_linear), WPRE bf16 kacc-layout weights {nw} B (layers/partition.json groups)")
    if a.keep_layout:
        nf = 0
        for g in emod.match("dense+silu"):
            i = g.info
            pack = lambda w, i=i: w.reshape(i["N"] // i["block"], i["block"], i["K"]).transpose(0, 2, 1)
            steps = [ag.step("silu_scale_x", reads=g.inputs, weights=g.w.as_bf16(layout=pack), at=g.calls[0]),
                     ag.step("silu_scale_dense", writes=g.outputs, at=i["dense"])]
            emit = {"silu_scale_x": {"template": "aiednn/elementwise/silu_scale_x.cc", "tag": "cpu silu_scale_x",
                                     "params": dict(WPACK=i["packed"], NK=i["N"] * i["K"], X=i["x"], KLEN=i["K"],
                                                    S="0" if i["scale"] is None else f"&tensors[{i['scale']}]")},
                    "silu_scale_dense": {"template": "aiednn/elementwise/silu_scale_dense.cc",
                                         "tag": "cpu " + i["symbol"].replace("tvmgen_default_fused_", ""),
                                         "params": dict(FN=i["symbol"], NARGS=i["nargs"])}}
            emod.offload_group(g, steps=steps, emit=emit)
            nf += 1
        log(f"aiegraph match('dense+silu'): {nf} groups (weight pre-packed in weights.bin, dense reads x*s)")

    # S6: the C glue (FA kernel declarations, PROF macros, expand timing) is the driver template that
    # aiegraph places at checked anchors of every generated driver -- registered before keep_layout so
    # it runs first; the build_unet.py side only declares groups + this template
    if att and not all(c.get("fa") is not None and c.get("fa_steps") for c in att):
        raise SystemExit("attention layers must be aiegraph FA group steps (--attn-fa)")
    import fa_attn
    emod.driver_template(**bm.driver_glue(fa_attn.DECL_FA + fa_attn.DECL_KL if att else "", attn=bool(att)))
    n = len(sel) + len(att)
    log(f"graph_driver.c: {n} partition call sites -> aiednn_aie_conv ({n - len(lin) - len(att)}) + "
        f"aiednn_aie_linear ({len(lin)})" + (f" + aiednn_aie_attn ({len(att)})" if att else "") +
        " (aiegraph group steps + driver template)")

    # gates 3-4 = aiegraph's group-level x86 check (TVM's C vs the step's x86 model, cos >= 0.999,
    # nonfinite == 0), binding: a failing carrier is rejected and the build stops below
    rep = emod.offload(f"layer{lidx}", str(kcc), check="group", verbose=a.verbose)
    log(f"offload: {rep}")
    lp = json.loads((work / "layers" / "partition.json").read_text())
    if not any(L.get("target") == "aie" and L.get("archive") for L in lp["layers"]):
        raise SystemExit("offload FAILED: carrier layer has no AIE archive in layers/partition.json")
    defs = dict(LAUNCH_WARM=1, LAUNCH_OPTS=bm.LAUNCH_OPTS_WARM) if a.launch_opt == "warm" else {}
    defs.update(CTRLPKT=a.ctrlpkt, AIE_PLANNED_LAUNCHES=planned, AIE_PLANNED_CALLS=len(sel) + len(att),
                TOTAL_GEMM_MACS=(int(part["total_gemm_macs"]), "ULL"))
    if keep is not None:
        defs["KEEP_LAYOUT_EDGES"] = keep["kept"]
    log(f"ktimer={'on' if a.ktimer else 'off'}")
    return defs, (len(sel) - len(lin), len(lin), len(att))


def check_driver(drv: str, n_patched) -> tuple:
    """The linked graph_driver.c's (aiednn_aie_conv, aiednn_aie_linear, aiednn_aie_attn) call sites == *n_patched*
    (an FA group's QK / PV site = aiednn_aie_attn_kl role 3 (K pack) / role 6 (launch + O))."""
    nkl = len(re.findall(r"    aiednn_aie_attn_kl\(\d+, [36], ", drv))
    got = (drv.count("    aiednn_aie_conv("), drv.count("    aiednn_aie_linear("), drv.count("    aiednn_aie_attn(") + nkl)
    if got != tuple(n_patched):
        raise SystemExit(f"build FAILED: linked driver has {got} aiednn_aie_conv/aiednn_aie_linear/aiednn_aie_attn "
                         f"call sites, expected {n_patched}")
    log(f"linked driver: {got[0]} call sites -> aiednn_aie_conv + {got[1]} -> aiednn_aie_linear (AIE), "
        f"weights of those read as bf16 in place" + (f"; {got[2]} -> aiednn_aie_attn" if got[2] else ""))
    return got
