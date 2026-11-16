#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Flow 3 -- ONNX via OnnxGraph and via the ORT plugin EP, checked against onnxruntime's CPU EP.

    python example/aiednn/flows/3_onnx/run.py [--model resnet18 | --onnx file.onnx] [--no-ep]
"""
import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

from model import model_and_input, model_arg          # noqa: E402
from aiegraph.dnn_graph import build_graph_lib        # noqa: E402
from aiegraph.dnn_onnx_graph import OnnxGraph         # noqa: E402


def main() -> int:
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    model_arg(ap)
    ap.add_argument("--onnx", default=None, help="an existing .onnx file (e.g. an int8 QDQ model) instead of --model")
    ap.add_argument("--no-ep", action="store_true", help="only the OnnxGraph importer")
    a = ap.parse_args()
    if a.onnx:
        path = Path(a.onnx)
        x = model_and_input("resnet18")[1] if a.model == "tiny" else model_and_input(a.model)[1]
    else:
        m, x = model_and_input(a.model)
        path = Path(tempfile.mkdtemp()) / f"{a.model}.onnx"
        torch.onnx.export(m, (x,), str(path), input_names=["x"], output_names=["y"], opset_version=17)
    cpu = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    xin, yout = cpu.get_inputs()[0].name, cpu.get_outputs()[0].name
    want = cpu.run(None, {xin: x.numpy()})[0]
    tol = 0.05 if a.onnx else 1e-4                       # int8 QDQ: ORT float-vs-integer spread
    ok = True

    og = OnnxGraph.load(build_graph_lib(REPO / "worklocal" / "aiednn" / "flows_graphlib"), path)   # a)
    print(og.report())
    y = og.run({xin: x.numpy()})[yout].reshape(want.shape)
    rel = float(np.abs(y - want).max() / np.abs(want).max())
    ok &= rel < tol and int(y.argmax()) == int(want.argmax())
    print(f"a) OnnxGraph: max rel diff vs onnxruntime CPU {rel:.2e}, top-1 {int(y.argmax())} (ORT {int(want.argmax())})")

    if not a.no_ep:                                                                              # b)
        from aiegraph import dnn_ort_ep
        sess = dnn_ort_ep.session(str(path))
        y = sess.run(None, {xin: x.numpy()})[0]
        rel = float(np.abs(y - want).max() / np.abs(want).max())
        ok &= rel < tol and int(y.argmax()) == int(want.argmax())
        print(f"b) ORT plugin EP: providers {sess.get_providers()}, nodes per EP {dnn_ort_ep.assigned(sess) or 'n/a'}, "
              f"max rel diff {rel:.2e}")
    print(f"{'PASS' if ok else 'FAIL'}: flow 3 ONNX [{path.name}]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
