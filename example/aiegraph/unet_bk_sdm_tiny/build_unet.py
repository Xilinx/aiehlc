#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""bk-sdm-tiny UNet (unet_sim.onnx, 3 inputs) -> aiegraph -> A78 main.elf, conv / linear / attention on the AIE.

  $PY build_unet.py --onnx unet_sim.onnx --real-io <real_io> --wdtype bf16 --partition partition_p4a.json \
      --aie conv,linear,attn --attn-fa --keep-layout --kincore --launch-opt warm --ctrlpkt aot-kco \
      --arm-layer-cflags-file a78_layer_cflags.txt --runtime-io --dump-out --out <dir>

This file only declares: the model's inputs, the offload groups (unet_offload.py) and the options.
aiegraph generates the board main.c, the inputs header, the PROF / FA driver glue (aiegraph.boardmain)
and the .aiehlc_io runtime-input section (emod.build(runtime_io=...)). The ONNX timestep input is int64;
it is rewritten to int32 + Cast so the board receives exactly the 4 bytes of real_io/timestep_i32.bin.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src"), str(HERE)]   # this checkout's aiegraph (gate: plain $PY)
SITE_PERF_DEF = "-DAIEHLC_SITE_PERF=1"   # M3.1 site-perf (aie_runtime.c / aie_host_split.h / site_prof.c)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--real-io", required=True, type=Path, help="sample.bin timestep_i32.bin ehs.bin p1_ref.h")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--aie", choices=("none", "conv", "conv,linear", "conv,linear,attn"), default="none",
                    help="partition layer classes put on the AIE (linear / attn need --kincore)")
    ap.add_argument("--attn-fa", action="store_true", help="each attention group = one flash-attention group")
    ap.add_argument("--keep-layout", action="store_true", help="aiegraph layout folds (needs --attn-fa)")
    ap.add_argument("--wdtype", choices=("fp32", "bf16"), default="fp32", help="weights.bin dtype")
    ap.add_argument("--partition", type=Path, default=None, help="partition.json (layers on the AIE)")
    ap.add_argument("--launch-opt", choices=("cold", "warm"), default="cold", help="warm: kernel loaded once")
    ap.add_argument("--ctrlpkt", choices=("off", "aot", "aot-kco"), default="off", help="control-packet plan")
    ap.add_argument("--kincore", action="store_true", help="one launch per layer, K in the core (gemm_kacc)")
    ap.add_argument("--kincore-layers", default="all", help="--kincore probe subset: partition conv indices")
    ap.add_argument("--arm-layer-cflags-file", type=Path, default=None, help="extra CFLAGS of the TVM layer objects")
    ap.add_argument("--runtime-io", action="store_true", help="inputs in the .aiehlc_io ELF section")
    ap.add_argument("--dump-out", action="store_true", help="--runtime-io: dump the output (AIO_*) on the board")
    ap.add_argument("--site-perf", action="store_true",
                    help="M3.1 diagnostic: -DAIEHLC_SITE_PERF=1, one PROF aie_site_perf core-counter line per "
                         "launches>0 part (counter setup adds MMIO per launch: not for L)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    log = lambda msg: print(f"[build_unet] {msg}", flush=True)
    a.ctrlpkt_kco = a.ctrlpkt == "aot-kco"   # aot-kco = the aot plan + KERNELCONFIGOFFLOAD (aiegraph.kincore)
    if a.ctrlpkt_kco:
        a.ctrlpkt = "aot"
    import aiegraph as ag
    from aiegraph import boardmain as bm
    errs = [m for bad, m in (
        (a.aie != "none" and (a.partition is None or a.wdtype != "bf16"), "--aie needs --partition and --wdtype bf16"),
        (a.keep_layout and not a.attn_fa, "--keep-layout needs --attn-fa"),
        (a.dump_out and not a.runtime_io, "--dump-out needs --runtime-io"),
        (a.kincore_layers != "all" and not a.kincore, "--kincore-layers needs --kincore"),
        (a.kincore and a.aie == "none", "--kincore needs --aie conv"),
        (a.ctrlpkt == "aot" and a.aie == "none", "--ctrlpkt aot needs --aie conv"),
        (a.launch_opt == "warm" and a.aie == "none", "--launch-opt warm needs --aie conv"),
        (a.site_perf and a.aie == "none", "--site-perf needs --aie conv")) if bad]
    try:
        layer_cflags = bm.read_cflags_file(a.arm_layer_cflags_file) if a.arm_layer_cflags_file else ""
    except ValueError as e:
        errs.append(f"--arm-layer-cflags-file {e}")
    if errs:
        log("; ".join(errs))
        return 2
    if layer_cflags:
        log(f"arm_layer_cflags ({a.arm_layer_cflags_file}): {layer_cflags}")
    if a.aie != "none":
        bm.profiling_env()
        opts = bm.launch_env(a.launch_opt)
        if a.site_perf:  # runtime + kernel host archives (EXTRA_DEFS); graph_driver.c gets it via extra_cflags
            os.environ["EXTRA_DEFS"] = (os.environ.get("EXTRA_DEFS", "") + " " + SITE_PERF_DEF).strip()
        log(f"launch_opt={a.launch_opt} ({opts}) defs={os.environ['EXTRA_DEFS']} "
            f"aiehlc.sh={os.environ.get('AIEHLC_SH_ARGS', '')}")
    out, real_io = a.out.resolve(), a.real_io.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(ag.__file__).resolve().parents[2]
    os.environ.setdefault("AIEHLC_ROOT", str(root))
    os.environ["AIEGRAPH_PARAMS_INPLACE"] = "1"         # 1.29 GB weights read in place from the linked blob
    os.environ["AIEGRAPH_WDTYPE"] = a.wdtype
    os.environ.setdefault("HOSTCOMPILE_HOST_OPT", "-O2")  # aiek host.cc (im2col / scatter glue inside graph_run)
    log(f"aiegraph from {ag.__file__} (repo {root}); HOSTCOMPILE_HOST_OPT={os.environ['HOSTCOMPILE_HOST_OPT']}")

    # the model's inputs, in graph order: (graph name, C name, real_io file)
    ins = [("sample", "sample", "sample.bin", "<f4"), ("timestep", "timestep", "timestep_i32.bin", "<i4"),
           ("encoder_hidden_states", "ehs", "ehs.bin", "<f4")]
    arrs = [(g, c, np.fromfile(real_io / f, dtype=dt)) for g, c, f, dt in ins]
    assert [x.size for _, _, x in arrs] == [4 * 32 * 32, 1, 77 * 768], [x.size for _, _, x in arrs]
    hdr = out / "unet_inputs.h"
    crc = bm.inputs_header(hdr, "unet", arrs, dict(AIE_MODE=a.aie, WDTYPE=a.wdtype), origin=real_io)
    want = (real_io / "unet_in.crc32").read_text().strip()
    if crc != want:
        log(f"inputs crc32 {crc} != {real_io / 'unet_in.crc32'} {want}")
        return 1
    log(f"inputs: {', '.join(f'{g} {x.size} {x.dtype}' for g, _, x in arrs)}; crc32={crc} == unet_in.crc32")

    onnx_i32 = bm.input_int32(a.onnx.resolve(), out / "unet_sim_t32.onnx", "timestep", log=log)
    m = ag.load(str(onnx_i32))
    log(f"loaded {m.name}: inputs {m.input_shapes} -> {m.output_shape}, {sum(m.ops.values())} nodes")
    shutil.rmtree(out / "ag_work", ignore_errors=True)   # a fresh conversion every build
    emod = m.convert(target="a78", quantize="fp32", image=str(real_io / "sample.bin"),
                     workdir=out / "ag_work", verbose=a.verbose)
    log(f"converted -> {emod.workdir}")
    bm.cross_toolchain(root, log=log)
    n_patched = None
    if a.aie != "none":
        import unet_offload
        defs, n_patched = unet_offload.offload(a, emod, out)
        bm.append_defines(hdr, "unet", defs)
    main_c = bm.main_c("unet", [Path(f).stem for _, _, f, _ in ins], aie=a.aie != "none",
                       runtime_io=a.runtime_io, note=(1, "timestep"))
    rio = [(Path(f).stem, real_io / f) for _, _, f, _ in ins] if a.runtime_io else None
    rep = emod.build(target="a78", verbose=a.verbose, main_c=main_c,
                     extra_sources=[bm.cmp_source(), real_io / "p1_ref.h", hdr],
                     extra_cflags=f"{bm.CMP_CFLAGS} {bm.HEAP_LDFLAG}" + (f" {SITE_PERF_DEF}" if a.site_perf else ""),
                     layer_cflags=layer_cflags,
                     runtime_io=rio, dump_out=a.dump_out, launch=a.launch_opt)
    if not rep.ok:
        log(f"build FAILED: {rep.reason}")
        return 1
    for ln in rep.io_info.splitlines():
        log(ln)
    if n_patched is not None:
        unet_offload.check_driver((Path(rep.elf).parent / "graph_driver.c").read_text(), n_patched)
    elf, wb = out / "main.elf", Path(rep.elf).parent / "weights.bin"
    shutil.copy2(rep.elf, elf)
    if wb.is_file():   # the blob netload streams to the board, beside the ELF
        shutil.copy2(wb, out / "weights.bin")
    log(f"built {rep.elf} -> {elf} ({elf.stat().st_size:,} B; weights.bin "
        f"{wb.stat().st_size if wb.is_file() else -1:,} B)")
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
