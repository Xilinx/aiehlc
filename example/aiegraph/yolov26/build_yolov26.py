#!/usr/bin/env python3
"""yolo26n (end2end ONNX, trained) -> aiegraph -> A78 main.elf, every Conv (incl. depthwise) on the AIE.

  $PY build_yolov26.py --onnx yolo26n_e2e_trained_op17.onnx --out <dir> [--input images.bin]

This file only declares the model: where its post-processing tail starts (TopK: cut before it, the
TopK / GatherElements selection runs on the host from the pre-TopK head), its input, and which calls go
to the AIE (every matched conv group, aiegraph.cnn). aiegraph generates the kernel source (src/aiednn),
the driver steps, the board main and the .aiehlc_io runtime-input section.
"""
import argparse
import os
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src")]   # this checkout's aiegraph (gate: plain $PY)
MODEL, KPREFIX, ENTRY = "yolo26n", "cnn", "aiek_cnnconv"
TAIL_OP = "TopK"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--input", type=Path, help="fp32 [1,3,640,640] build input (default: a fixed synthetic image)")
    ap.add_argument("--mc", type=int, default=256, help="output pixels per AIE launch")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    log = lambda msg: print(f"[build_yolov26] {msg}", flush=True)
    import onnx
    import onnxruntime as ort
    import aiegraph as ag
    from aiegraph import boardmain as bm, cnn
    from aiegraph.autopart import inventory
    from aiegraph.cut import cut_before

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(ag.__file__).resolve().parents[2]
    os.environ.setdefault("AIEHLC_ROOT", str(root))
    os.environ["AIEGRAPH_WDTYPE"] = "bf16"                 # bf16 weights, fp32 accumulation
    os.environ.setdefault("HOSTCOMPILE_HOST_OPT", "-O2")
    bm.profiling_env()
    opts = bm.launch_env("warm")
    log(f"aiegraph from {ag.__file__}; onnx {a.onnx}; launch warm ({opts})")

    # 1. the offloaded model: everything before the TopK tail (output = the pre-TopK head)
    cut, tensor = cut_before(onnx.load(str(a.onnx)), TAIL_OP, out=out / f"{MODEL}_cut.onnx")
    log(f"cut before {TAIL_OP}: output {tensor} {[d.dim_value for d in cut.graph.output[0].type.tensor_type.shape.dim]}"
        f", {len(cut.graph.node)} nodes")
    iname = cut.graph.input[0].name
    shape = [d.dim_value for d in cut.graph.input[0].type.tensor_type.shape.dim]
    if a.input:
        x = np.fromfile(a.input, "<f4").reshape(shape)
    else:   # letterbox gray with a smooth pattern, in [0, 1]
        yy, xx = np.meshgrid(np.linspace(0, 1, shape[2]), np.linspace(0, 1, shape[3]), indexing="ij")
        x = np.stack([0.45 + 0.4 * np.sin(6.0 * xx + c) * yy for c in range(shape[1])])[None].astype("<f4")
    xin = out / "images.bin"
    x.astype("<f4").tofile(xin)
    crc = "%08x" % zlib.crc32(x.astype("<f4").tobytes())
    ref = ort.InferenceSession(str(out / f"{MODEL}_cut.onnx"), providers=["CPUExecutionProvider"]).run(None, {iname: x})[0]
    log(f"input {iname} {shape} crc32={crc}; ORT fp32 head {list(ref.shape)} (board CMP reference)")

    # 2. convert (TVM, per-layer C) and match every Conv, depthwise included, to its TVM call
    shutil.rmtree(out / "ag_work", ignore_errors=True)
    emod = ag.load(str(out / f"{MODEL}_cut.onnx")).convert(target="a78", quantize="fp32", workdir=out / "ag_work",
                                                           verbose=a.verbose)
    bm.cross_toolchain(root, log=log)
    inv = inventory(cut)
    rows = [dict(idx=i, node=r["node"], weight=r["weight"], cls="conv" if r["op"] == "Conv" else "linear",
                 mkn=[[r["m"], r["k"], r["n"]]]) for i, r in enumerate(inv) if r["cls"] != "attn"]
    groups = emod.match("conv", partition={"layers": rows}, model=cut, groups=(1, "dw"))
    nconv = sum(1 for n in cut.graph.node if n.op_type == "Conv")
    ndw = sum(1 for r in inv if r["op"] == "Conv" and r["why"])
    plans = cnn.plan_convs(groups, mc=a.mc)
    total = sum(r["macs"] for r in inv)
    aie = sum(r["macs"] for r in inv if r["node"] in {p.onnx for p in plans})
    for p in plans:
        log(f"  aie conv {p.cid:3d} {p.onnx} -> call {p.call} {p.symbol.replace('tvmgen_default_fused_', '')} "
            f"M={p.M} N={p.N} K={p.K} s={p.sh} p={p.ph} dw={p.dw} act={p.act} -> {p.Np}x{p.Kk}x{p.mc} "
            f"launches={p.launches}")
    log(f"matmul (host): {sum(1 for r in inv if r['op'] != 'Conv')} nodes, "
        f"{sum(r['macs'] for r in inv if r['op'] != 'Conv')} MACs")
    print(f"OFFLOAD {MODEL} conv={len(plans)}/{nconv} dw={sum(p.dw for p in plans)}/{ndw} "
          f"AIE_MAC_SHARE={100.0 * aie / total:.1f}%", flush=True)

    # 3. kernel source on a carrier layer, one group step per conv, the PROF driver glue, AIE archive
    lidx, car = cnn.pick_carrier(plans, Path(emod.workdir) / "layers")
    proto = emod.offload_prototype(lidx, entry=ENTRY)
    kcc = cnn.write_kernel(out / "aiek_src", proto.spec(), plans, kprefix=KPREFIX, entry=ENTRY, warm=True)
    log(f"carrier layer {lidx} ({car.symbol}); kernel source {kcc}")
    planned = cnn.offload_convs(emod, groups, plans, kcc, kprefix=KPREFIX, log=log)
    emod.driver_template(**bm.driver_glue(kprefix=KPREFIX))
    log(f"offload: {emod.offload(f'layer{lidx}', str(kcc), check='group', verbose=a.verbose)}")

    # 4. board main + inputs header + reference, link
    hdr = out / f"{MODEL}_inputs.h"
    bm.inputs_header(hdr, MODEL, [(iname, iname, x.reshape(-1))], dict(AIE_MODE="conv", WDTYPE="bf16"), origin=xin)
    bm.append_defines(hdr, MODEL, dict(LAUNCH_WARM=1, LAUNCH_OPTS=bm.LAUNCH_OPTS_WARM, CTRLPKT="off",
                                       AIE_PLANNED_LAUNCHES=planned, AIE_PLANNED_CALLS=len(plans),
                                       TOTAL_GEMM_MACS=(total, "ULL")))
    refh = bm.ref_header(out / "p1_ref.h", MODEL, ref, crc, data="build-input")
    main_c = bm.main_c(MODEL, [iname], aie=True, runtime_io=True, kprefix=KPREFIX)
    rep = emod.build(target="a78", verbose=a.verbose, main_c=main_c, extra_sources=[bm.cmp_source(), refh, hdr],
                     extra_cflags=f"{bm.CMP_CFLAGS} {bm.HEAP_LDFLAG}", runtime_io=[(iname, xin)], dump_out=True,
                     launch="warm")
    if not rep.ok:
        log(f"build FAILED: {rep.reason}")
        return 1
    for ln in rep.io_info.splitlines():
        log(ln)
    drv = (Path(rep.elf).parent / "graph_driver.c").read_text()
    sites = drv.count(f"    {KPREFIX}_aie_conv(")
    if sites != len(plans):
        log(f"build FAILED: linked driver has {sites} {KPREFIX}_aie_conv call sites, expected {len(plans)}")
        return 1
    log(f"linked driver: {sites} call sites -> {KPREFIX}_aie_conv (AIE), {planned} launches planned")
    elf, wb = out / "main.elf", Path(rep.elf).parent / "weights.bin"
    shutil.copy2(rep.elf, elf)
    if wb.is_file():
        shutil.copy2(wb, out / "weights.bin")
    log(f"built {rep.elf} -> {elf} ({elf.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (print(e.code, flush=True) or 1)
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    print(f"GATE_RC={rc}", flush=True)
    sys.exit(rc)
