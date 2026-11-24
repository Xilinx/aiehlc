#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Flow 2 -- PyTorch to the AI Engine: torch.compile in the process, torch.export to a board program.

torch.compile(model, backend="aiednn"): Dynamo hands the fx graph to aiegraph.dnn_torch_backend; FxGraph maps
each node to one op of the partition graph API (include/aiednn_graph.h), the library partitions it, and the
supported partitions run through aiednn while the rest stays torch code (workstation emulation here: the AIE
kernels' C references stand in for the array).

torch.export.export(model, (x,)): the ExportedProgram goes through the offline model compiler (flow 4):
aiegraph.dnn_export.to_fx turns its ATen graph back into module / function calls, dnn_torch lowers it (int8
calibrated on x), the aiednn MLIR pipeline fuses it, and dnn_build compiles the C app with aiehlc into a board
ELF carrying the AIE megakernel. The exported program is saved (.pt2) and can be compiled later with --pt2.

    python example/aiednn/flows/2_torch_compile/run.py [--model resnet18]              # torch.compile
    python example/aiednn/flows/2_torch_compile/run.py --export [--model resnet18]     # torch.export -> x86 app
    python example/aiednn/flows/2_torch_compile/run.py --export --emit model [--model tiny]  # session API + smoke host
    python example/aiednn/flows/2_torch_compile/run.py --export --model gpt2_small --target vek385 \\
        --board portobello13 --vitis-settings <.../Vitis/settings64.sh>               # -> board ELF, run there
    python example/aiednn/flows/2_torch_compile/run.py --pt2 model.pt2 --model resnet18 # a saved program
    python example/aiednn/flows/2_torch_compile/run.py --export --eager plan --eager-mesh 8x4 --model gpt2_small \\
        --target vek280 --board <host> --vitis-settings <.../Vitis/settings64.sh>    # eager template kernels, 8x4
    python example/aiednn/flows/2_torch_compile/run.py --export --eager plan --eager-mesh 8x4 --synth-weights \\
        --model gpt2_small --target vek280 --board <host> ...           # weights generated on the board: small ELF
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                   # noqa: E402

from model import model_and_input, model_arg   # noqa: E402
from aiegraph import dnn_torch_backend as be   # noqa: E402
from aiegraph.backends import aiednn as flags  # noqa: E402  (registers backend="aiednn")

OUT = REPO / "worklocal" / "aiednn" / "flows"


def run_compile(m, x, model: str) -> int:
    flags.graph = True                         # partition with the graph API (the default when present)
    flags.benchmark = False                    # True: time every implementation once, keep the fastest
    torch._dynamo.reset()
    with torch.no_grad():
        got = torch.compile(m, backend="aiednn")(x)
        want = m(x)
    fg = be.aiednn_backend.last.aiednn_graph   # the FxGraph Dynamo's graph became
    print(fg.report())
    rel = float((got - want).abs().max() / want.abs().max())
    top, wtop = int(got.flatten().argmax()), int(want.flatten().argmax())
    ok = rel < 1e-3 and top == wtop
    print(f"{'PASS' if ok else 'FAIL'}: flow 2 torch.compile [{model}], max rel diff vs eager {rel:.2e}, "
          f"top-1 {top} (eager {wtop})")
    return 0 if ok else 1


def run_export(m, x, a) -> int:
    from aiegraph import dnn_export
    out = OUT / (f"flow2_export_{a.model}_{a.mode}" + (f"_eager_{a.eager}" + (f"_{a.eager_mesh}" if a.eager_mesh != "4x4" else "")
                                                     if a.eager else "")
                 + (f"_emit_{a.emit}" if a.emit != "app" else "") + ("_synth" if a.synth_weights else ""))
    out.mkdir(parents=True, exist_ok=True)
    if a.pt2:                                  # a program exported earlier (torch.export.save)
        ep = torch.export.load(a.pt2)
    else:
        # prepare_for_export: nn.MultiheadAttention as fusable modules (same math); the rest is the model as is
        ep = torch.export.export(dnn_export.prepare_for_export(m), (x,))
        torch.export.save(ep, out / f"{a.model}.pt2")
        print(f"exported: {out / f'{a.model}.pt2'}")
    rep, g, _ = dnn_export.compile_exported(ep, x, mode=a.mode, name=a.model, target=a.target, out=out,
                                            vitis_settings=a.vitis_settings, top1_range=getattr(m, "top1_range", None),
                                            eager=a.eager, emit=a.emit, eager_mesh=a.eager_mesh,
                                            synth_weights=a.synth_weights, custom_sources=a.custom_c)
    print(g.summary())
    if a.emit == "model":
        print(f"session: {out / f'{a.model}_model.h'}  (create/run/destroy; replace smoke host with your main)")
    if rep.ok and (a.target == "x86" or a.board):   # the program is built; running it is a separate step
        dnn_export.run(rep, board=a.board)
    log = rep.run_log.read_text(errors="replace") if rep.run_log and rep.run_log.exists() else ""
    print(f"app {rep.app}\nelf {rep.elf}" + ("" if rep.ok else f"\nbuild refused: {rep.reason}"))
    print("\n".join(ln for ln in log.splitlines() if ln.startswith(("MODEL ", "PASS", "FAIL", "MEGAKERNEL"))))
    ran = a.target == "x86" or bool(a.board)
    ok = rep.ok and (rep.run_ok if ran else True)
    print(f"{'PASS' if ok else 'FAIL'}: flow 2 torch.export [{a.model}] ({a.mode}) emit={a.emit} -> {a.target} program"
          f"{'' if ran else ' (built, not run)'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    model_arg(ap)
    ap.add_argument("--export", action="store_true", help="torch.export -> the model compiler -> a C app / board ELF")
    ap.add_argument("--pt2", default=None, help="compile a saved ExportedProgram (implies --export)")
    ap.add_argument("--mode", default="int8", choices=["f32", "int8"])
    ap.add_argument("--emit", default="app", choices=["app", "model"],
                    help="--export: 'app' = full benchmark main (default); 'model' = session API "
                         "(create/run/destroy) + smoke host — write your own main against <name>_model.h")
    ap.add_argument("--eager", nargs="?", const="library", default=None, choices=["library", "plan"],
                    help="--export: eager mode -- every layer on a template kernel instance loaded right before it "
                         "runs; 'library' (default) the premade template library, 'plan' instances generated for "
                         "this model's shapes")
    ap.add_argument("--eager-mesh", default="4x4", choices=["4x4", "6x4", "8x4"],
                    help="--eager: the instances' mesh (rows x 4); 8x4 is the fastest on VEK280")
    ap.add_argument("--synth-weights", action="store_true",
                    help="--export: the app generates its large int8 weights on the device (a board ELF downloads in "
                         "seconds); checked against the graph run on the workstation with the same weights")
    ap.add_argument("--custom-c", action="append", default=[], metavar="FILE",
                    help="--export: a C file implementing the program's opaque custom ops, run on the host "
                         "(aiednn_custom_<ns>_<op>_<overload>; repeatable)")
    ap.add_argument("--target", default="x86", choices=["x86", "vek280", "vek385"])
    ap.add_argument("--board", default=None, help="run the built ELF on this board (a non-x86 --target)")
    ap.add_argument("--vitis-settings", default=None)
    a = ap.parse_args()
    m, x = model_and_input(a.model)
    return run_export(m, x, a) if (a.export or a.pt2) else run_compile(m, x, a.model)


if __name__ == "__main__":
    sys.exit(main())
