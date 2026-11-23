#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""ResNet18 (ONNX, 1x3x224x224 -> 1000 logits) -> aiegraph -> A78 main.elf, every Conv and the Gemm on the AIE.

  $PY build_resnet18.py --onnx resnet18_fp32_op17.onnx --out <dir> [--input in.bin] [--ctrlpkt aot-kco]

This file only declares the model's input, its reference and the options. aiegraph does the rest:
  aiegraph.cnn        every Conv / weighted Gemm -> one aiegraph group on a bf16 gemm_kacc instance
                      (templates in src/aiednn/{gemm,conv2d}), archive on a carrier layer;
  aiegraph.boardmain  board main.c, inputs header, reference header, driver glue;
  emod.build          main.elf with the input in the .aiehlc_io section (patch it with aiehlc_elfio.py).
The build's own input is --input (raw fp32 NCHW) or a seeded N(0,1) tensor; its onnxruntime output is the
reference the board compares against when it runs that input. The build log prints
'OFFLOAD resnet18 conv=n/N gemm=m/M AIE_MAC_SHARE=x%'.
"""
import argparse
import os
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src")]   # this checkout's aiegraph / aiednn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--input", type=Path, default=None, help="raw fp32 1x3x224x224 (default: seeded N(0,1))")
    ap.add_argument("--ctrlpkt", choices=("off", "aot", "aot-kco"), default="aot-kco", help="control-packet plan")
    ap.add_argument("--launch-opt", choices=("cold", "warm"), default="warm", help="warm: kernels loaded once")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    log = lambda msg: print(f"[build_resnet18] {msg}", flush=True)
    import onnx
    import onnxruntime as ort
    import aiegraph as ag
    from aiegraph import boardmain as bm, cnn
    bm.profiling_env()
    opts = bm.launch_env(a.launch_opt)
    log(f"launch_opt={a.launch_opt} ({opts}) ctrlpkt={a.ctrlpkt} defs={os.environ.get('EXTRA_DEFS', '')}")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(ag.__file__).resolve().parents[2]
    os.environ.setdefault("AIEHLC_ROOT", str(root))
    os.environ["AIEGRAPH_PARAMS_INPLACE"] = "1"
    os.environ["AIEGRAPH_WDTYPE"] = "bf16"
    os.environ.setdefault("HOSTCOMPILE_HOST_OPT", "-O2")

    # the model's input and its x86 reference (onnxruntime, fp32)
    shape = (1, 3, 224, 224)
    x = (np.fromfile(a.input, dtype="<f4") if a.input else
         np.random.default_rng(0).standard_normal(shape).astype("<f4")).reshape(shape)
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].astype("<f4").reshape(-1)
    inb = out / "input.bin"
    x.tofile(inb)
    hdr = out / "resnet18_inputs.h"
    crc = bm.inputs_header(hdr, "resnet18", [("data", "input", x.reshape(-1))],
                           dict(AIE_MODE="conv,gemm", WDTYPE="bf16"), origin=inb)
    assert crc == "%08x" % zlib.crc32(x.tobytes())
    src = f"input {a.input.name}" if a.input else "seeded N(0,1) input"
    bm.ref_header(out / "p1_ref.h", "resnet18", ref, crc, f"ort/{src}")
    log(f"input {src} crc32={crc}; ORT reference top-1={int(ref.argmax())} ({ref.size} logits)")

    m = ag.load(str(a.onnx.resolve()))
    log(f"loaded {m.name}: {m.input_shapes} -> {m.output_shape}, {sum(m.ops.values())} nodes")
    shutil.rmtree(out / "ag_work", ignore_errors=True)
    emod = m.convert(target="a78", quantize="fp32", image=str(inb), workdir=out / "ag_work", verbose=a.verbose)
    log(f"converted -> {emod.workdir}")
    bm.cross_toolchain(root, log=log)
    res = cnn.offload_all(emod, onnx.load(str(m.onnx_path)), out, warm=a.launch_opt == "warm",
                          ctrlpkt=a.ctrlpkt, verbose=a.verbose, log=log)
    log(f"OFFLOAD resnet18 conv={res['conv'][0]}/{res['conv'][1]} gemm={res['gemm'][0]}/{res['gemm'][1]} "
        f"AIE_MAC_SHARE={res['mac_share']:.1f}%")
    log(f"aie_launches_planned={res['launches']} instances={len(res['shapes'])} kernel={res['kernel']}")
    bm.append_defines(hdr, "resnet18", res["defs"])

    main_c = bm.main_c("resnet18", ["input"], aie=True, runtime_io=True)
    rep = emod.build(target="a78", verbose=a.verbose, main_c=main_c,
                     extra_sources=[bm.cmp_source(), out / "p1_ref.h", hdr],
                     extra_cflags=f"{bm.CMP_CFLAGS} {bm.HEAP_LDFLAG}",
                     runtime_io=[("input", inb)], dump_out=True, launch=a.launch_opt)
    if not rep.ok:
        log(f"build FAILED: {rep.reason}")
        return 1
    for ln in rep.io_info.splitlines():
        log(ln)
    got = cnn.check_driver((Path(rep.elf).parent / "graph_driver.c").read_text(), res)
    log(f"linked driver: {got[0]} aiednn_conv + {got[1]} aiednn_dense call sites")
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
