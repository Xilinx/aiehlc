#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Probe: does every conv of a real model map onto aiegraph ops?

Walks the ONNX graph (x86 only, no TVM, no board). For each ``Conv`` the fused
epilogue is the chain TVM's FuseOps folds into a conv: the Conv bias, then
single-consumer ``Add`` / ``Relu`` nodes. The op list and the shapes of the
non-bias add operands go through ``aiegraph.partition.match_conv``; a conv is
*mapped* when the result is not None and every op is an aiegraph dialect op.

    python3 probe_unet_patterns.py --onnx unet_sim.onnx [--manifest layers/manifest.json]

prints ``PT unet convs=<n> mapped=<m> `` plus one ``PT family`` line per
pattern (and every unmapped conv). ``--manifest`` also maps the fused TVM
conv layers of a build (their name tokens, no shapes).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from aiegraph.partition import AIEGRAPH_OP_KINDS, match_conv


def _shapes(model) -> dict:
    """value name -> static shape (list of ints) from ONNX shape inference."""
    import onnx
    from onnx import shape_inference

    skel = onnx.ModelProto()
    skel.CopyFrom(model)
    for t in skel.graph.initializer:          # dims are enough; drop the 1.3 GB
        t.ClearField("raw_data")
        for f in ("float_data", "int32_data", "int64_data", "double_data"):
            t.ClearField(f)
    out = {}
    for t in model.graph.initializer:
        out[t.name] = list(t.dims)
    try:
        inf = shape_inference.infer_shapes(skel)
    except Exception as e:                      # noqa: BLE001
        print(f"PT warn shape inference failed: {e}", file=sys.stderr)
        inf = skel
    g = inf.graph
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            out[vi.name] = [d.dim_value for d in dims]
    return out


def conv_patterns(model) -> list:
    """One dict per Conv: name, ops, add_shapes, out_shape, pattern."""
    g = model.graph
    shapes = _shapes(model)
    consumers = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n)
    graph_outs = {o.name for o in g.output}
    rows = []
    for n in g.node:
        if n.op_type != "Conv":
            continue
        ops = ["conv2d"] + (["bias_add"] if len(n.input) > 2 and n.input[2] else [])
        out_shape = shapes.get(n.output[0])
        add_shapes, cur = [], n.output[0]
        while cur not in graph_outs and len(consumers[cur]) == 1:
            c = consumers[cur][0]
            if c.op_type == "Add" and len(c.input) == 2:
                other = c.input[1] if c.input[0] == cur else c.input[0]
                ops.append("add")
                add_shapes.append(shapes.get(other))
            elif c.op_type == "Relu":
                ops.append("relu")
            else:
                break
            cur = c.output[0]
        p = match_conv(ops, add_shapes, out_shape)
        ok = p is not None and all(k in AIEGRAPH_OP_KINDS for k in p.aiegraph)
        rows.append(dict(name=n.name, ops=ops, add_shapes=add_shapes,
                         out_shape=out_shape, pattern=p, mapped=ok))
    return rows


def manifest_patterns(path: Path) -> list:
    doc = json.loads(Path(path).read_text())
    rows = []
    for L in doc["layers"]:
        ops = L.get("ops", [])
        if "conv2d" not in ops:
            continue
        p = match_conv(ops)
        rows.append(dict(name=L.get("dir", L.get("symbol")), ops=ops, pattern=p,
                         mapped=p is not None and all(k in AIEGRAPH_OP_KINDS for k in p.aiegraph),
                         calls=L.get("calls")))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, help="layers/manifest.json of a TVM build (optional)")
    ap.add_argument("--tag", default="unet")
    a = ap.parse_args(argv)
    import onnx

    model = onnx.load(str(a.onnx))
    rows = conv_patterns(model)
    fam = collections.Counter()
    for r in rows:
        p = r["pattern"]
        fam[(p.family, " + ".join(p.aiegraph)) if p else ("UNMAPPED", "-")] += 1
    for (f, ops), k in sorted(fam.items(), key=lambda x: -x[1]):
        print(f"PT family {f:<28} -> {ops:<32} n={k}")
    for r in rows:
        if not r["mapped"]:
            print(f"PT unmapped {r['name']} ops={r['ops']} add_shapes={r['add_shapes']} "
                  f"out={r['out_shape']}")
    n, m = len(rows), sum(r["mapped"] for r in rows)
    epi = sum(1 for r in rows if r["mapped"] and r["pattern"].adds and not r["pattern"].relu)
    print(f"PT {a.tag} convs={n} mapped={m} epilogue_add={epi} onnx={a.onnx.name}")
    if a.manifest:
        mr = manifest_patterns(a.manifest)
        print(f"PT manifest conv_layers={len(mr)} mapped={sum(r['mapped'] for r in mr)} "
              f"({a.manifest})")
    return 0 if n and m == n else 1


if __name__ == "__main__":
    sys.exit(main())
