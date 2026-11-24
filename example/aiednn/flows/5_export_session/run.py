#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Flow 5 -- torch.export -> session library; you write the host ``main``.

``compile_exported(..., emit="model")`` / ``dnn_torch.emit_model`` writes
``<name>_model.h`` (create / run / destroy). This flow's ``host.cc`` is the
hand-written main: it includes the generated session and drives one inference.

    python example/aiednn/flows/5_export_session/run.py [--model tiny]
    python example/aiednn/flows/5_export_session/run.py --target vek280 --board galerina13 \\
        --vitis-settings <.../Vitis/settings64.sh>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from model import model_and_input, model_arg  # noqa: E402
from aiegraph import dnn_export  # noqa: E402
from aiegraph import dnn_torch as dt  # noqa: E402
from aiegraph.dnn import dnn_build  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = REPO / "worklocal" / "aiednn" / "flows"


def _write_host(out: Path, name: str, cname: str, paths: dict, in_bin: Path, gold_bin: Path, tol: float) -> Path:
    """Fill host.cc placeholders and absolute-include the generated session (aiehlc splice)."""
    text = (HERE / "host.cc").read_text()
    text = (text.replace("@@NAME@@", name)
                .replace("@@CNAME_UP@@", cname.upper())
                .replace("@@CNAME@@", cname)
                .replace("@@TOL@@", repr(tol)))
    text = text.replace(f'#include "{name}_model.h"', f'#include "{paths["h"].resolve()}"')
    text = text.replace(f'#include "{name}_model.cc"', f'#include "{paths["cc"].resolve()}"')
    text = text.replace(f'INCBIN(host_in_data, "{name}_host_in.bin");',
                        f'INCBIN(host_in_data, "{in_bin.resolve()}");')
    text = text.replace(f'INCBIN(host_golden_data, "{name}_golden.bin");',
                        f'INCBIN(host_golden_data, "{gold_bin.resolve()}");')
    # create-name comments for dnn_build kernel pick (creates live in the model .cc)
    host = out / "host.cc"
    host.write_text(text)
    return host


def main() -> int:
    ap = argparse.ArgumentParser()
    model_arg(ap)
    ap.add_argument("--mode", default="int8", choices=["f32", "int8"])
    ap.add_argument("--no-mlir", action="store_true",
                    help="Python epilogue fusion (default when aiednn-opt is not built)")
    ap.add_argument("--target", default="x86", choices=["x86", "vek280", "vek385"])
    ap.add_argument("--board", default=None, help="run the built ELF on this board (a non-x86 --target)")
    ap.add_argument("--vitis-settings", default=None)
    a = ap.parse_args()

    out = OUT / f"flow5_{a.model}_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    m, x = model_and_input(a.model)
    ep = torch.export.export(dnn_export.prepare_for_export(m), (x,))
    torch.export.save(ep, out / f"{a.model}.pt2")
    mlir = not a.no_mlir
    try:
        g, ref, yf = dt.references(dnn_export.to_fx(ep, [x]), x, a.mode, a.model, mlir=mlir)
    except Exception as e:
        if mlir and "aiednn-opt" in str(e):
            print(f"[flow 5] aiednn-opt missing; falling back to --no-mlir ({e})")
            mlir = False
            g, ref, yf = dt.references(dnn_export.to_fx(ep, [x]), x, a.mode, a.model, mlir=False)
        else:
            raise
    paths = dt.emit_model(g, out, prefer="fastest")
    cname = paths["cname"]
    xin = g.tensors[g.root(g.input)].data
    in_bin = out / f"{a.model}_host_in.bin"
    in_bin.write_bytes(xin.astype(dt.DT_NP[g.tensors[g.root(g.input)].dt]).tobytes())
    # the reference output the host checks against (rel err < dt.TOL[mode], same top-1)
    gold_bin = out / f"{a.model}_golden.bin"
    gold_bin.write_bytes(ref.astype("float32").tobytes())
    host = _write_host(out, a.model, cname, paths, in_bin, gold_bin, dt.TOL[a.mode])
    print(f"exported: {out / f'{a.model}.pt2'}")
    print(f"session:  {paths['h']}")
    print(f"host:     {host}")
    print(g.summary())

    run = a.target == "x86" or bool(a.board)
    rep = dnn_build(host, target=a.target, out=out / a.target, run=run, board=a.board,
                    vitis_settings=a.vitis_settings)
    log = rep.run_log.read_text(errors="replace") if run and rep.run_log and rep.run_log.exists() else ""
    print(f"app {host}\nelf {rep.elf}" + ("" if rep.ok else f"\nbuild refused: {rep.reason}"))
    print("\n".join(ln for ln in log.splitlines()
                    if ln.startswith(("PASS", "FAIL", "top-1", "===", "input "))))
    ok = rep.ok and (rep.run_ok if run else True)
    print(f"{'PASS' if ok else 'FAIL'}: flow 5 export session [{a.model}] ({a.mode}"
          f"{', mlir' if mlir else ''}) -> {a.target} host-owned main"
          f"{'' if run else ' (built, not run)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
