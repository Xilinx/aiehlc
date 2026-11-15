###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Lift the **whole** Relay graph into aiegraph IR, then partition AIE vs CPU.

This is the ``--aiegraph`` path, and it does the two things in that order:

1. **Lift every layer** it can onto an ``aiegraph`` op, building one
   ``aiegraph.func`` whose SSA def-use edges are the model's real dataflow.
2. **Partition**: ops in ``aie_ops`` (default the conv2d family, which is all
   ``run_aie_pipeline`` has a kernel body for) are offloaded to the aiehlc
   backend; every other layer **reuses the TVM-generated C** from stage 5,
   untouched.

Contrast with ``aie_offload.py``, which is index-selected and only ever looks at
the layers you name. This module always walks the whole graph, so the output is
a complete accounting: every layer gets a verdict and a reason.

Why the IR is built over graph *nodes*, not manifest layers
-----------------------------------------------------------
``layers/manifest.json`` has one entry per **distinct generated symbol** (22 for
ResNet-18), but the graph invokes 24 -- two kernels run twice, because the two
64ch/56x56 blocks are shape-identical and TVM emits one function for each pair.
Dataflow lives on the nodes, not the symbols, so the IR is built by walking
``graph["nodes"]`` in execution order. Layer folders are then keyed back by
symbol, and a folder reused by two nodes gets **one** set of AIE artifacts --
the invocations are the same kernel at the same shape, so the artifacts would
be byte-identical.

(Note this is exactly what ``aie_offload.select_layers`` gets wrong: it zips
manifest index *i* against ``call_nodes[i]`` positionally, which desynchronizes
at the first repeated symbol and hands every later layer another layer's
geometry. ``_nodes_by_symbol`` here maps by name instead.)

Op mapping
----------
The dialect has four ops; the fused TVM layer kinds map on like this::

    conv2d_add_relu      -> conv_bn_relu
    conv2d_add           -> conv_bn
    conv2d_add_add_relu  -> conv_bn + residual_add_relu   (two ops, one layer)
    global_avg_pool2d               \\_ folded into one avgpool_fc
      + dense_add                   /
    max_pool2d           -> (none)  CPU: no aiegraph op
    batch_flatten        -> (none)  CPU: dead kernel, the graph elides it

A layer is offloaded only when **every** aiegraph op it expands to is in
``aie_ops``. That is why ``conv2d_add_add_relu`` stays on the CPU even though
its conv half is eligible: the generated C is one fused function computing
conv+bias+residual+relu, so there is no seam to split it at. Claiming the layer
for AIE would mean the residual never runs.

Gaps are expected, not errors
-----------------------------
A layer with no aiegraph op (``max_pool2d``) breaks the SSA chain: its consumer
has no producer to reference, so ``build_aiegraph_module`` materializes a fresh
block argument (``main_index=-1``, a network input). The IR still verifies, and
the discontinuity is recorded in ``partition.json`` rather than papered over --
which is the point, since each offloaded layer is launched standalone anyway
(the APU owns the buffers between launches).

**The block args do not alias.** Two consumers of the same CPU-produced tensor
each get their *own* block argument, because ``-1`` is the only way to say "not
from an aiegraph op" across the pybind boundary and it allocates a fresh one
every time. ResNet-18 shows this at the pool: ``%1`` reads ``%arg1`` and the
first residual's skip reads ``%arg2``, though both are really node 4's output.
So the IR is faithful about *which ops run where* and about every edge between
two aiegraph ops, but it under-shares at the CPU boundary. Nothing downstream
depends on that -- launches are standalone and the APU owns the buffers -- and
fixing it would mean teaching ``build_aiegraph_module`` to reference an existing
block argument by index.
"""

from __future__ import annotations

import json
from pathlib import Path

from frontend.tvmrelay.aie_offload import aie_backend

#: Repo root: .../src/frontend/tvmrelay/aiegraph_partition.py -> up three.
#: Needed to reach script/hostcompile.sh when archiving a layer.
_REPO = Path(__file__).resolve().parents[3]

__all__ = ["AIEGRAPH_OP_KINDS", "DEFAULT_AIE_OPS", "build_nodes",
           "build_ir_ops", "partition_layers", "run_aiegraph"]

#: Every op the aiegraph dialect defines (``td/aiegraphop.td``).
AIEGRAPH_OP_KINDS = ("conv_bn_relu", "conv_bn", "residual_add_relu",
                     "avgpool_fc")

#: Ops actually offloaded to the aiehlc kernel backend. Only the conv2d family
#: has a kernel body in ``frontend.tvmrelay.kernels`` and a ``run_aie_pipeline``
#: path; the rest reuse TVM's CPU C.
DEFAULT_AIE_OPS = ("conv_bn_relu", "conv_bn")

#: Placeholder quant params, matching ``aie_offload.build_aiegraph_ir``. The
#: Q7 BN fold pattern from ``frontend.tvm.model``; real per-op scales would come
#: from a QNN-quantized graph, which the no-LLVM build cannot produce.
_QUANT = {"in_scale": 1.0, "in_zp": 0, "out_scale": 1.0, "out_zp": 0,
          "bn_scale": 64, "bn_bias": 0}


# ═══════════════════════════════════════════════════════════════════════════
#  Graph walking
# ═══════════════════════════════════════════════════════════════════════════

def _is_call(graph: dict, idx: int) -> bool:
    """True for a real kernel invocation (a ``tvm_op`` node that is not a nop)."""
    node = graph["nodes"][idx]
    return (node.get("op") == "tvm_op"
            and node["attrs"].get("func_name") != "__nop")


def _resolve_nop(graph: dict, idx: int) -> int:
    """Follow ``__nop`` passthroughs back to the node that really produces data.

    ResNet-18 has one: the ``[1,512,1,1] -> [1,512]`` reshape between the global
    pool and the classifier, which is layout-free so TVM elides it. Without this
    the classifier looks like it consumes a non-kernel and the avgpool/dense
    fold never fires.
    """
    seen = set()
    while idx not in seen:
        node = graph["nodes"][idx]
        if node.get("op") != "tvm_op" or node["attrs"].get("func_name") != "__nop":
            return idx
        inputs = node.get("inputs") or []
        if not inputs:
            return idx
        seen.add(idx)
        idx = inputs[0][0]
    return idx


def _data_inputs(graph: dict, idx: int) -> list:
    """Producer node indices feeding *idx*, weights/bias constants dropped.

    Constants arrive as ``op == "null"`` nodes; what is left is the tensor
    dataflow. For a residual conv that is ``[conv_input, skip_source]``, in
    that order -- the first data input is the feature map the convolution reads,
    the last is the skip path it adds.
    """
    out = []
    for src, _slot, _ver in graph["nodes"][idx].get("inputs") or []:
        resolved = _resolve_nop(graph, src)
        if _is_call(graph, resolved):
            out.append(resolved)
    return out


def _shape(graph: dict, node_idx: int, slot: int = 0) -> list:
    """Output shape of *node_idx*, via the graph's entry-id table."""
    return graph["attrs"]["shape"][1][graph["node_row_ptr"][node_idx] + slot]


def _input_shapes(graph: dict, idx: int) -> list:
    """Every input shape of *idx*, constants included (weights live here)."""
    row_ptr = graph["node_row_ptr"]
    shapes = graph["attrs"]["shape"][1]
    return [shapes[row_ptr[src] + slot]
            for src, slot, _ver in graph["nodes"][idx].get("inputs") or []]


def _conv_geometry(graph: dict, idx: int) -> dict:
    """``{H,W,Cin,Cout,K,stride,...}`` for a conv node, from TVM's own shapes.

    Nothing here re-derives convolution arithmetic: the shapes are the result of
    TVM's type inference, so ``stride`` is read back out of the input/output
    ratio and ``K`` off the weight tensor.

    The int8 default graph is **NCHWc** (5-D feature maps, 6-D weights) after
    ``AlterOpLayout``; that case is handed to ``aie_offload._layer_geometry``,
    which folds ``C = C_outer * c`` (skill: nchwclayoutfold). Without it every
    conv -- the stem included -- came back ``non-4D shapes; not a conv2d``.
    """
    if any(len(s) == 5 for s in _input_shapes(graph, idx) + [_shape(graph, idx)]):
        from frontend.tvmrelay.aie_offload import _layer_geometry

        return _layer_geometry(graph, idx)
    producers = _data_inputs(graph, idx)
    inputs = _input_shapes(graph, idx)
    if not producers and not inputs:
        return {}
    feat = _shape(graph, producers[0]) if producers else inputs[0]
    out = _shape(graph, idx)
    if len(feat) != 4 or len(out) != 4:
        return {}
    # The conv weight is the 4-D input shaped [Cout, Cin, K, K]. Matching on
    # that rather than "any 4-D input that isn't the feature map" keeps a
    # broadcast bias like [1, Cout, 1, 1] -- which is also 4-D and also not the
    # feature map -- from being read as the weight and yielding K=1.
    weights = [s for s in inputs
               if len(s) == 4 and s[0] == out[1] and s[1] == feat[1]
               and s[2] == s[3]]
    if len(weights) != 1:
        return {}
    k = weights[0][2]
    stride = max(1, feat[2] // out[2]) if out[2] else 1
    return {"H": feat[2], "W": feat[3], "Cin": feat[1], "Cout": out[1],
            "K": k, "stride": stride,
            "in_elems": feat[1] * feat[2] * feat[3],
            "out_elems": out[1] * out[2] * out[3]}


# ═══════════════════════════════════════════════════════════════════════════
#  Layer  ->  aiegraph op mapping
# ═══════════════════════════════════════════════════════════════════════════

def _nodes_by_symbol(graph: dict) -> dict:
    """``{func_name: [node_idx, ...]}`` in execution order.

    Keying by symbol rather than by position is what keeps a repeated kernel
    from shifting every later layer's geometry by one.
    """
    out = {}
    for idx, node in enumerate(graph["nodes"]):
        if _is_call(graph, idx):
            out.setdefault(node["attrs"]["func_name"], []).append(idx)
    return out


def _aiegraph_ops_for(kind: str, ops: list) -> list:
    """The aiegraph op kinds a fused TVM layer expands to (possibly empty).

    Driven off the fused op list rather than the folder name so an unseen
    fusion degrades to "no aiegraph op" instead of being silently mis-mapped.
    The pattern family lives in AIEHLC ``aiegraph.partition.match_conv``: a
    float conv(+bias)(+add) with no ReLU (UNet temb / skip / attention
    residual) is ``conv_bn`` with an fp32 epilogue (see ``_conv_pattern``);
    ``partition_layers`` keeps such a layer on the CPU unless the backend runs
    the epilogue, so the add is never dropped.
    """
    p = _conv_pattern(ops)
    return list(p.aiegraph) if p else []


def _conv_pattern(ops: list):
    from aiegraph.partition import match_conv

    return match_conv(ops)


def _float_bias(ops: list) -> bool:
    """True when the conv bias came in as ``nn.bias_add``.

    ``ops`` is either real Relay op names (``bias_add``) or the underscore-split
    tokens of the fused function name (``bias``, ``add``); accept both.
    """
    return "bias_add" in ops or "bias" in ops


def _residual_joins(ops: list) -> int:
    """How many element-wise ``add``s in a fused conv are residual joins.

    Exactly one bias is assumed. It arrives as ``bias_add`` (one op name, not
    counted as ``add``), as tokens ``bias`` + ``add``, or as a plain ``add``
    (BN folded, the ResNet-18 path); in the last two one ``add`` is the bias.
    """
    adds = ops.count("add")
    if "bias_add" in ops:
        return adds
    return max(adds - 1, 0)


def build_nodes(manifest: dict, graph: dict) -> list:
    """Walk the graph and describe each node as aiegraph ops + provenance.

    Returns one record per executed kernel node, in execution order, carrying
    the layer folder it came from, the aiegraph ops it expands to, and its
    geometry. The avgpool/dense pair is folded here: the pool node is marked
    ``folded`` and contributes no op of its own.
    """
    by_symbol = _nodes_by_symbol(graph)
    layer_of = {}
    for entry in manifest["layers"]:
        for node_idx in by_symbol.get(entry["symbol"], []):
            layer_of[node_idx] = entry

    records = []
    pooled = {}                       # gap node -> its data input, for the fold
    for idx in range(len(graph["nodes"])):
        if not _is_call(graph, idx):
            continue
        entry = layer_of.get(idx, {})
        ops = entry.get("ops", [])
        rec = {"node": idx, "index": entry.get("index"),
               "dir": entry.get("dir", ""), "kind": entry.get("kind", ""),
               "symbol": entry.get("symbol", ""), "ops": ops,
               "inputs": _data_inputs(graph, idx), "aiegraph": []}

        if "global_avg_pool2d" in ops:
            # Hold it: the classifier that consumes it folds both into one op.
            pooled[idx] = rec
            rec.update(folded=True, reason="folded into the avgpool_fc of the "
                                           "classifier that consumes it")
            records.append(rec)
            continue

        if "dense" in ops:
            _fold_classifier(graph, idx, rec, pooled)
            records.append(rec)
            continue

        kinds = _aiegraph_ops_for(rec["kind"], ops)
        if not kinds:
            rec["reason"] = (f"no aiegraph op for {rec['kind'] or 'this layer'} "
                             f"(the dialect defines {', '.join(AIEGRAPH_OP_KINDS)})")
        else:
            geom = _conv_geometry(graph, idx)
            if not geom:
                rec["reason"] = "non-4D shapes; not a conv2d"
            else:
                rec.update(geom)
                rec["aiegraph"] = kinds
                p = _conv_pattern(ops)
                if p is not None and p.adds and "residual_add_relu" not in kinds:
                    rec["epilogue"] = list(p.adds)
        records.append(rec)

    _mark_dead_layers(manifest, by_symbol, records)
    return records


def _fold_classifier(graph: dict, idx: int, rec: dict, pooled: dict) -> None:
    """Fold ``global_avg_pool2d -> dense(+bias)`` into a single ``avgpool_fc``.

    The dialect has one op for the whole classifier tail, so the pool is not
    representable on its own. When the dense's producer is not a pool the dense
    is left unmapped rather than guessed at.
    """
    producers = rec["inputs"]
    pool = pooled.get(producers[0]) if producers else None
    if pool is None:
        rec["reason"] = "dense with no global_avg_pool2d producer to fold with"
        return
    feat = _shape(graph, pool["inputs"][0]) if pool["inputs"] else []
    out = _shape(graph, idx)
    if len(feat) != 4 or len(out) != 2:
        rec["reason"] = "unexpected classifier shapes"
        return
    rec.update(aiegraph=["avgpool_fc"], spatial_h=feat[2], spatial_w=feat[3],
               channels=feat[1], num_classes=out[1],
               # The fold consumes the pool's input, not the pool's output.
               inputs=list(pool["inputs"]), folded_with=pool["node"])


def _mark_dead_layers(manifest: dict, by_symbol: dict, records: list) -> None:
    """Append records for emitted kernels the graph never calls.

    ``batch_flatten`` is the case: a layout-free reshape TVM emits and then
    elides into a ``__nop``. Reporting it keeps the partition's layer count
    equal to the manifest's, so a missing layer means a bug rather than a
    known-dead kernel.
    """
    for entry in manifest["layers"]:
        if entry["symbol"] in by_symbol:
            continue
        records.append({"node": None, "index": entry["index"],
                        "dir": entry.get("dir", ""), "kind": entry.get("kind", ""),
                        "symbol": entry["symbol"], "ops": entry.get("ops", []),
                        "inputs": [], "aiegraph": [],
                        "reason": "dead kernel -- the graph never calls it"})


# ═══════════════════════════════════════════════════════════════════════════
#  IR construction
# ═══════════════════════════════════════════════════════════════════════════

def build_ir_ops(records: list) -> tuple:
    """``(op_dicts, owners)`` ready for ``build_aiegraph_module``.

    *owners* is parallel to *op_dicts*: entry *i* is the record that produced
    aiegraph op *i*, so a launch descriptor can be mapped back to its layer.

    Dataflow is passed by explicit index. A producer that mapped to no aiegraph
    op (``max_pool2d``) simply is not in ``produced_by``, so its consumer gets
    ``-1`` and the backend materializes a network input -- the intended
    behaviour for a graph with CPU layers in the middle of it.
    """
    op_dicts, owners = [], []
    produced_by = {}                   # graph node -> index of its last op

    for rec in records:
        for kind in rec["aiegraph"]:
            main = produced_by.get(rec["inputs"][0], -1) if rec["inputs"] else -1
            if kind in ("conv_bn", "conv_bn_relu"):
                d = {"op": kind, "main_index": main, "H": rec["H"], "W": rec["W"],
                     "Cin": rec["Cin"], "Cout": rec["Cout"], "K": rec["K"],
                     "stride": rec["stride"], "weights": f"w{rec['node']}",
                     **_QUANT}
            elif kind == "residual_add_relu":
                # Consumes this record's own conv result plus the skip path,
                # which is the *last* data input (the first is the conv's).
                skip = rec["inputs"][-1] if len(rec["inputs"]) > 1 else -1
                d = {"op": kind, "main_index": len(op_dicts) - 1,
                     "skip_index": produced_by.get(skip, -1),
                     "length": rec["out_elems"]}
            else:                                            # avgpool_fc
                d = {"op": kind, "main_index": main,
                     "spatial_h": rec["spatial_h"], "spatial_w": rec["spatial_w"],
                     "channels": rec["channels"],
                     "num_classes": rec["num_classes"],
                     "weights": f"w{rec['node']}"}
            op_dicts.append(d)
            owners.append(rec)
        if rec["aiegraph"] and rec["node"] is not None:
            produced_by[rec["node"]] = len(op_dicts) - 1
    return op_dicts, owners


# ═══════════════════════════════════════════════════════════════════════════
#  Partition
# ═══════════════════════════════════════════════════════════════════════════

def partition_layers(records: list, aie_ops=DEFAULT_AIE_OPS,
                     layers=None, epilogue=False) -> list:
    """Decide AIE vs CPU per layer. Returns one verdict dict per record.

    A layer goes to AIE only when **every** aiegraph op it expands to is in
    *aie_ops*. Partial eligibility is not offloadable: the TVM C for a residual
    block is a single fused function, so taking its conv to AIE would drop the
    residual add entirely.

    *layers* is the ``--aie-layers`` selection: a list of ``layers/`` folder
    indices, or ``None`` for "every eligible layer". A layer outside the
    selection stays on the APU even when it is otherwise eligible -- this is the
    same selection ``--aie-offload`` uses, so both flags target the same conv
    (layer 6, the 7x7/s2 stem, under the int8-legalized graph).
    """
    sel = None if layers is None else set(layers)
    verdicts = []
    for rec in records:
        kinds = rec["aiegraph"]
        if sel is not None and rec.get("index") not in sel:
            want = ",".join(str(i) for i in sorted(sel))
            verdicts.append({**rec, "target": "cpu",
                             "verdict_reason": f"not in the --aie-layers "
                                               f"selection ({want})"})
            continue
        if not kinds:
            reason = rec.get("reason", "no aiegraph op")
            target = "cpu"
        elif rec.get("epilogue") and not epilogue:
            target = "cpu"
            reason = (f"aiegraph {' + '.join(kinds)} + fp32 epilogue add "
                      f"({', '.join(rec['epilogue'])}), but this backend's conv "
                      f"kernel takes no epilogue input; the fused C cannot be "
                      f"split, so the whole layer stays on the APU")
        elif all(k in aie_ops for k in kinds):
            target, reason = "aie", f"aiegraph {' + '.join(kinds)}"
        else:
            unsupported = sorted({k for k in kinds if k not in aie_ops})
            target = "cpu"
            reason = (f"aiegraph {' + '.join(kinds)}, but "
                      f"{', '.join(unsupported)} has no AIE kernel; the fused C "
                      f"cannot be split, so the whole layer stays on the APU")
        verdicts.append({**rec, "target": target, "verdict_reason": reason})
    return verdicts


# ═══════════════════════════════════════════════════════════════════════════
#  Driver
# ═══════════════════════════════════════════════════════════════════════════

def run_aiegraph(out_dir, aie_ops=DEFAULT_AIE_OPS, mesh=(2, 2),
                 emit_aie: bool = True, layers=None,
                 verbose: bool = True) -> dict:
    """Lift the whole graph to aiegraph, partition it, and offload the AIE part.

    Writes ``layers/aiegraph.mlir`` (the verified whole-graph IR) and
    ``layers/partition.json`` (the per-layer verdict), plus
    ``layers/<NN_name>/aie/`` for each offloaded layer. The TVM C of every layer
    is left exactly as stage 5 wrote it -- CPU layers reuse it as-is, and it is
    kept for AIE layers too so the APU ELF still links.

    *layers* narrows the offload to those ``layers/`` indices (``--aie-layers``;
    ``None`` = every eligible layer). The whole graph is still lifted and
    verified either way -- the selection only decides which verified layers get
    an ``aie/`` build.
    """
    out_dir = Path(out_dir).resolve()
    layers_dir = out_dir / "layers"
    manifest_path = layers_dir / "manifest.json"
    graph_path = next(out_dir.glob("*_graph.json"), None)
    if not manifest_path.is_file():
        return {"ok": False, "reason": "layers/manifest.json not found "
                                       "(stage 5 did not run)"}
    if graph_path is None:
        return {"ok": False, "reason": "no *_graph.json in the output dir"}

    try:
        manifest = json.loads(manifest_path.read_text())
        graph = json.loads(graph_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        # Match the missing-file contract above rather than raising: a truncated
        # stage-4/5 artifact is a stage-4/5 problem, not a crash here.
        return {"ok": False, "reason": f"unreadable {type(exc).__name__}: {exc}"}
    records = build_nodes(manifest, graph)
    verdicts = partition_layers(records, aie_ops, layers)

    op_dicts, owners = build_ir_ops(records)
    if not op_dicts:
        return {"ok": False, "reason": "no layer mapped onto an aiegraph op",
                "layers": verdicts}

    backend = aie_backend()
    ir = backend.build_aiegraph_module(op_dicts, "resnet18")
    ir_path = layers_dir / "aiegraph.mlir"
    ir_path.write_text(ir)
    if verbose:
        print(f"  [aiegraph] {len(op_dicts)} ops over {len(records)} layers, "
              f"verified -> {ir_path.name}")

    built = _offload(backend, ir, owners, verdicts, layers_dir, out_dir,
                     aie_ops, mesh, emit_aie, verbose)

    n_aie = sum(1 for v in verdicts if v["target"] == "aie")
    result = {"ok": all(b["ok"] for b in built) if built else True,
              "ir": str(ir_path), "op_count": len(op_dicts),
              "aie_count": n_aie, "cpu_count": len(verdicts) - n_aie,
              "aie_ops": list(aie_ops), "layers": verdicts, "built": built,
              "selection": None if layers is None else list(layers)}
    _write_partition(layers_dir, result, verbose)
    return result


def _offload(backend, ir: str, owners: list, verdicts: list, layers_dir: Path,
             out_dir: Path, aie_ops, mesh, emit_aie: bool,
             verbose: bool) -> list:
    """Run ``run_aie_pipeline`` for each AIE-bound layer. Returns build records.

    Deduplicated by ``(folder, op)``, which is the key that distinguishes the
    two things that both look like "the same layer again":

    * the graph invoking one kernel twice (same folder, same op) -- identical
      artifacts, so it is built once and the second node is recorded against
      the entry that owns the folder;
    * a layer that expands to **several** aiegraph ops (same folder, different
      op) -- a residual block is ``conv_bn`` *and* ``residual_add_relu``, two
      separate kernels. Deduplicating those away would silently drop the
      residual add while still reporting the layer as offloaded, so each op
      gets its own ``aie/<op>/`` subdirectory rather than overwriting the
      other's ``host.cc``.
    """
    if not emit_aie:
        return []
    from frontend.tvmrelay import kernels

    aie_dirs = {v["node"]: v for v in verdicts if v["target"] == "aie"}
    launches = backend.lower_aiegraph(ir)
    # The launches are matched to owners positionally, so a backend that ever
    # reorders or merges ops would misattribute every later layer. Fail loudly
    # instead: silent misattribution is the exact bug this module exists to
    # avoid (see the module docstring on aie_offload.select_layers).
    if len(launches) != len(owners):
        raise RuntimeError(
            f"lower_aiegraph returned {len(launches)} launches for "
            f"{len(owners)} ops; cannot map launches back to layers")
    built, done = [], {}
    multi_op = {rec["node"] for rec in owners if len(rec["aiegraph"]) > 1}

    for launch, rec in zip(launches, owners):
        verdict = aie_dirs.get(rec["node"])
        if verdict is None or launch["op"] not in aie_ops:
            continue
        stem = verdict["dir"] or f"{verdict['index']:02d}"
        key = (stem, launch["op"])
        if key in done:
            done[key].setdefault("extra_nodes", []).append(rec["node"])
            continue

        # Only sub-divide when the layer really is several kernels, so the
        # common single-op layer keeps the flat aie/ layout.
        aie_dir = layers_dir / stem / "aie"
        if rec["node"] in multi_op:
            aie_dir = aie_dir / launch["op"]
        entry = _build_one(backend, kernels, launch, rec, verdict, aie_dir,
                           out_dir, mesh, verbose)
        built.append(entry)
        done[key] = entry
    return built


def _build_one(backend, kernels, launch: dict, rec: dict, verdict: dict,
               aie_dir: Path, out_dir: Path, mesh, verbose: bool) -> dict:
    """Run ``run_aie_pipeline`` for one launch into *aie_dir*. Returns its record.

    A conv with exactly the ResNet stem geometry is built from
    ``conv2dstem.cc`` by aiehlc instead -- the same logic ``test_conv2d.cc``
    verifies bit-exact against this layer -- not from the generic placeholder
    body (``aie_stem_lib``).
    """
    from frontend.tvmrelay import aie_stem_lib

    if launch["op"].startswith("conv_") and aie_stem_lib.matches_stem(rec):
        return _build_stem(aie_stem_lib, launch, rec, verdict, aie_dir,
                           out_dir, verbose)
    aie_dir.mkdir(parents=True, exist_ok=True)
    specs = [(list(shape), int(bits), bool(is_in))
             for (shape, bits, is_in) in launch["tensor_specs"]]
    body = kernels.kernel_body_for(launch["op"], launch["func_name"])
    ok = backend.run_aie_pipeline(mesh[0], mesh[1], specs, str(aie_dir), body,
                               launch["func_name"])
    produced = sorted(p.name for p in aie_dir.iterdir()) if ok else []

    if verbose:
        print(f"  [aiegraph] layer {verdict['index']:02d} {verdict['kind']}"
              f" -> {launch['op']} -> {aie_dir.relative_to(out_dir)}/ "
              f"({len(produced)} files)")

    entry = {"index": verdict["index"], "node": rec["node"],
             "dir": str(aie_dir.relative_to(out_dir)), "ok": bool(ok),
             "op": launch["op"], "func_name": launch["func_name"],
             "files": produced}

    # The pipeline stops at source. Wrap it in an op entry and archive it, so
    # the layer folder holds something a host program can actually link
    # against rather than artifacts nothing consumes.
    if ok:
        entry.update(_archive_one(aie_dir, verdict, launch, specs, mesh,
                                  out_dir, verbose))
    return entry


def _build_stem(aie_stem_lib, launch: dict, rec: dict, verdict: dict,
                aie_dir: Path, out_dir: Path, verbose: bool) -> dict:
    """aiehlc-build ``conv2dstem.cc`` into *aie_dir*. Returns its record."""
    # The TVM kernel symbol rides along so host.cc gets a packed-call entry
    # graph_driver.c can call in its place (aie_stem_lib._ENTRY_TMPL).
    res = aie_stem_lib.build_stem_layer(aie_dir, index=verdict["index"],
                                        func_name=rec["symbol"], verbose=False)
    if verbose:
        print(f"  [aiegraph] layer {verdict['index']:02d} {verdict['kind']}"
              f" -> {launch['op']} (stem) -> aiehlc "
              f"{aie_stem_lib.STEM_SOURCE.name} -> "
              f"{aie_dir.relative_to(out_dir)}/ ({len(res['files'])} files)")
    entry = {"index": verdict["index"], "node": rec["node"],
             "dir": str(aie_dir.relative_to(out_dir)), "ok": res["ok"],
             "op": launch["op"], "func_name": launch["func_name"],
             "files": res["files"], "backend": "aiehlc",
             "source": str(aie_stem_lib.STEM_SOURCE.relative_to(_REPO)),
             "entry": res.get("entry"), "replaces": rec["symbol"]}
    if res["archive"]:
        entry["archive"] = str(Path(res["archive"]).relative_to(out_dir))
        if verbose:
            print(f"  [aiegraph]   -> {Path(res['archive']).name} "
                  f"({Path(res['archive']).stat().st_size / 1024:,.0f} KB)")
    else:
        entry.update(archive=None,
                     archive_reason=res.get("reason", "aiehlc built no archive"))
        if verbose:
            print(f"  [aiegraph]   no archive: {entry['archive_reason']}")
    return entry


def _archive_one(aie_dir: Path, verdict: dict, launch: dict, specs,
                 mesh, out_dir: Path, verbose: bool) -> dict:
    """Emit the op entry for one built layer and archive it. Never raises."""
    from frontend.tvmrelay import aie_layer_lib

    stem = aie_dir.parent.name if aie_dir.name == "aie" else aie_dir.name
    try:
        src = aie_layer_lib.emit_op_entry(aie_dir, stem, launch["func_name"],
                                          specs, mesh)
        built = aie_layer_lib.build_archive(aie_dir, stem, _REPO,
                                            verbose=verbose)
    except Exception as exc:                       # noqa: BLE001
        return {"archive": None, "archive_reason": f"{type(exc).__name__}: {exc}"}

    if not built.get("ok"):
        if verbose:
            print(f"  [aiegraph]   no archive: {built['reason']}")
        return {"archive": None, "archive_reason": built["reason"],
                "op_entry": src.name}

    # Resolve both sides: aie_dir may arrive relative to the cwd while out_dir
    # is absolute, and relative_to() raises rather than coping with the mix.
    archive = Path(built["archive"]).resolve()
    try:
        shown = archive.relative_to(Path(out_dir).resolve())
    except ValueError:                  # archive redirected outside out_dir
        shown = archive
    return {"archive": str(shown), "op_entry": src.name}


def _write_partition(layers_dir: Path, result: dict, verbose: bool) -> Path:
    """Write ``partition.json``: the per-layer AIE/CPU verdict and why.

    Only the fields worth reading back are kept -- the raw graph edges stay in
    the IR, which is the authoritative record of the dataflow.
    """
    by_index = {}
    for b in result["built"]:
        by_index[b["index"]] = b
    layers = []
    for v in result["layers"]:
        built = by_index.get(v["index"]) if v["target"] == "aie" else None
        layers.append({
            "index": v["index"], "dir": v["dir"], "kind": v["kind"],
            "symbol": v["symbol"], "graph_node": v["node"],
            "aiegraph": v["aiegraph"], "target": v["target"],
            "reason": v["verdict_reason"],
            "aie_dir": built["dir"] if built else None,
            # How aie/ was produced: "aiehlc" from a hand-written source
            # (the stem -> conv2dstem.cc), else the generic aiegraph pipeline.
            "backend": (built.get("backend", "run_aie_pipeline")
                        if built else None),
            "source": built.get("source") if built else None,
            # graph_driver.c calls `entry` instead of `replaces` when built
            # with -DGRAPH_AIE_OFFLOAD (arm_build.aie_entries).
            "entry": built.get("entry") if built else None,
            "replaces": built.get("replaces") if built else None,
            # The linkable product of that folder, or why there isn't one.
            # Null with a reason is the normal state on a box without the
            # Vitis cross toolchain; null without one means it was never tried.
            "archive": built.get("archive") if built else None,
            "archive_reason": built.get("archive_reason") if built else None,
        })
    # Counts are over graph *invocations*, so two nodes sharing a symbol count
    # twice -- that is the honest number for "how much of the network runs on
    # AIE". The folder counts are also reported, since those are what is on
    # disk: a reused kernel gets one folder, not two.
    doc = {"aie_ops": result["aie_ops"], "op_count": result["op_count"],
           "aie_count": result["aie_count"], "cpu_count": result["cpu_count"],
           "aie_dirs": len({l["dir"] for l in layers if l["target"] == "aie"}),
           "cpu_dirs": len({l["dir"] for l in layers if l["target"] == "cpu"}),
           "counts_are": "graph invocations; a kernel called twice counts twice",
           "ir": Path(result["ir"]).name, "layers": layers}
    path = layers_dir / "partition.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    if verbose:
        print(f"  [aiegraph] partition: {result['aie_count']} AIE / "
              f"{result['cpu_count']} CPU invocation(s), reusing TVM C "
              f"({doc['aie_dirs']} AIE folder(s)) -> {path.name}")
    return path
