#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Probe (x86 only): the generic aiegraph path -- convert + match -- on non-UNet models.

    python3 src/aiegraph/test/probe_generic_models.py [--model resnet18|yolo26n ...] [--work DIR]

Per pinned ONNX model:
  1. inventory    node count, ONNX Conv (depthwise = group == Cin) and constant-weight Gemm/MatMul;
  2. cut          a model with a TopK tail is cut before it (``aiegraph.cut.cut_before``) --
                  the offloaded part ends at the pre-TopK head, the tail stays on the host;
  3. convert      ``aiegraph.convert(..., quantize="fp32")`` (TVM, per-layer C, no make);
  4. match        a partition with every Conv / weighted Gemm row (``autopart.inventory``),
                  ``emod.match("conv", groups=(1, "dw"))`` + ``emod.match("linear")``: each row
                  must come back as a one-call group on a distinct TVM call.

Prints one ``PROBE <id> onnx=<path> nodes=<n> conv_matched=<k>/<n> ...`` line per model;
rc 0 only if every Conv / Gemm matched.
"""
import argparse
import collections
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ZOO = Path(os.environ.get("MODELZOO", "/scratch/staff/huaj/modelzoo"))
# pinned in the unet-m4 sprint (phase PR brief); NOT modelzoo/yolo26n.onnx (no TopK)
MODELS = {
    "resnet18": ZOO / "artifacts/resnet18/onnx/resnet18_fp32_op17.onnx",
    "yolo26n": ZOO / "artifacts/yolo26/yolo26n.onnx",
}
TAIL_OP = "TopK"


def _attrs(n):
    from onnx import helper
    return {a.name: helper.get_attribute_value(a) for a in n.attribute}


def probe(mid: str, path: Path, work: Path) -> bool:
    import onnx
    import aiegraph as ag
    from aiegraph.autopart import inventory
    from aiegraph.cut import cut_before

    full = onnx.load(str(path))
    nodes = len(full.graph.node)
    ops = collections.Counter(n.op_type for n in full.graph.node)
    work.mkdir(parents=True, exist_ok=True)
    cut, extra = None, ""
    m = full
    if ops.get(TAIL_OP):
        m, t = cut_before(full, TAIL_OP, out=work / f"{mid}_cut.onnx")
        cut = TAIL_OP
        extra = f" cut_tensor={t} cut_nodes={len(m.graph.node)}"
        src = work / f"{mid}_cut.onnx"
    else:
        src = path
    init = {t.name for t in m.graph.initializer}
    oconv = [n for n in m.graph.node if n.op_type == "Conv"]
    shapes = {}
    m_inf = onnx.shape_inference.infer_shapes(m)
    for v in list(m_inf.graph.value_info) + list(m_inf.graph.input):
        shapes[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    odw = [n for n in oconv if _attrs(n).get("group", 1) > 1
           and _attrs(n)["group"] == shapes.get(n.input[0], [0, -1])[1]]
    ogemm = [n for n in m.graph.node if n.op_type in ("Gemm", "MatMul") and n.input[1] in init]
    full_conv = ops.get("Conv", 0)
    full_gemm = sum(1 for n in full.graph.node if n.op_type in ("Gemm", "MatMul")
                    and n.input[1] in {t.name for t in full.graph.initializer})

    emod = ag.load(str(src)).convert(target="a78", quantize="fp32", workdir=work / f"ag_{mid}")
    rows = []
    for i, r in enumerate(inventory(m)):
        if r["cls"] == "attn":
            continue                    # activation x activation: not a weighted call
        rows.append(dict(idx=i, node=r["node"], weight=r["weight"],
                         cls="conv" if r["op"] == "Conv" else "linear",
                         mkn=[[r["m"], r["k"], r["n"]]]))
    part = {"layers": rows}
    conv = emod.match("conv", partition=part, model=m, groups=(1, "dw"))
    lin = emod.match("linear", partition=part, model=m) if ogemm else []
    calls = [g.info["call"] for g in conv + lin]
    distinct = len(set(calls)) == len(calls)
    dw = sum(1 for g in conv if g.info["groups"] > 1)
    folded = sum(1 for g in conv if g.info.get("folded"))
    line = f"PROBE {mid} onnx={path} nodes={nodes} conv_matched={len(conv)}/{full_conv}"
    if full_gemm:
        line += f" gemm_matched={len(lin)}/{full_gemm}"
    if odw:
        line += f" dw={dw}/{len(odw)}"
    if cut:
        line += f" cut_before={cut}"
    print(f"probe {mid}: workdir={emod.workdir} tvm_calls={len(emod.call_graph().calls)} "
          f"distinct_calls={distinct} folded={folded}{extra}")
    print(line, flush=True)
    return (distinct and len(conv) == full_conv == len(oconv) and len(lin) == full_gemm == len(ogemm)
            and dw == len(odw))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", choices=sorted(MODELS))
    ap.add_argument("--work", type=Path,
                    default=Path(os.environ.get("TMP") or tempfile.gettempdir()) / "aiegraph_probe")
    a = ap.parse_args()
    ok = True
    for mid in a.model or list(MODELS):
        ok &= probe(mid, MODELS[mid], a.work.resolve() / mid)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
