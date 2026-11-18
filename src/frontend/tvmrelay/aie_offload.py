###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Map ``--aie-offload --aie-layers`` onto the convs the AIE BYOC path offloads.

``--aie-offload`` runs through TVM BYOC (``byoc/``): the selected layers are
partitioned out of the Relay graph BEFORE ``relay.build``, and TVM's graph
executor calls the generated wrapper -- which calls the aiehlc-built AIE
library -- in place of its own kernel. See ``deploy_flow.run`` and
``doc/design/byoc_aie_plan.md``.

Layer index -> conv identity
----------------------------
``--aie-layers`` numbers layers the way stage 5 does (``layers/NN_<name>/``,
from the CPU build), but BYOC partitions Relay, where those layers do not exist
yet. Position cannot bridge the two: the graph executor runs each residual
shortcut conv AFTER its main branch (fused with the residual add) while
Relay's post-order meets it first. So a selected layer is identified by its
conv's **weights** (``byoc.aie_annotate.weight_fingerprint``, a hash of the
sorted values -- layout- and dtype-independent): ``resolve_conv_targets``
fingerprints the layer's weight param in the CPU build, and
``check_targets`` demands exactly one Relay conv with that fingerprint AND the
same geometry before anything is offloaded.

**No tile-budget check here -- offload is blind by design.** Tiling and the
per-tile memory budget are aiehlc's job (skill: aieoffloadblind).
"""

from __future__ import annotations

import atexit
import json
import sys
from pathlib import Path

__all__ = ["select_layers", "parse_selection", "resolve_conv_targets",
           "check_targets", "aie_backend"]

#: Only the conv2d family has a kernel body + runtime path. Everything else in
#: the graph stays on the APU regardless of what is selected.
AIE_OP_KINDS = ("conv_bn_relu", "conv_bn")


# ═══════════════════════════════════════════════════════════════════════════
#  Selection
# ═══════════════════════════════════════════════════════════════════════════

def parse_selection(spec) -> list:
    """``"0"`` / ``"0,2,4"`` / ``"all"`` / ``None`` -> list of indices or None.

    ``None`` means "every layer" and is what ``all`` resolves to; the caller
    still filters by op kind afterwards.
    """
    if spec is None or str(spec).strip().lower() in ("all", "*"):
        return None
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if not part.lstrip("-").isdigit():
            raise ValueError(f"bad layer selector {part!r} (want ints or 'all')")
        out.append(int(part))
    return out


def _nchw(shape: list) -> tuple:
    """Normalize a feature-map shape to ``(C, H, W)``, or ``None`` if we cannot.

    Two layouts arrive here, because ``_layer_geometry`` reads the graph AFTER
    ``relay.build`` has run ``AlterOpLayout``:

        [N, C, H, W]          plain NCHW
        [N, C/c, H, W, c]     NCHWc -- C split so the inner block vectorizes

    The 5-D form is a *layout* artifact of TVM's x86 schedule, not a real fifth
    spatial axis, so folding ``C = C_outer * c`` recovers the geometry aiehlc
    wants. aiehlc's spatial spaces carry d1..d4 (``aiehlc.cc`` defines no
    ``tdD5``), and the Gen5 DMA tops out at ``NumAddrDim`` 3 for the shim and
    core tiles / 4 for the MemTile, so a genuine 5-D descriptor could not be
    programmed anyway.

    The fold is valid for SIZING unconditionally -- ``Cin = C_outer * c`` is the
    true channel count either way, and that is all the callers here use it for
    (``tensor_specs`` dims, never addressing).

    It is NOT a statement about memory order. Only ``C_outer == 1`` makes the
    NCHWc buffer byte-identical to the HWC that aietensorop reads; with
    ``C_outer > 1`` the data interleaves as ``(c_outer, h, w, c_inner)`` and a
    repack is required at the data-movement boundary or the kernel computes
    against transposed channels. ``needs_repack`` in ``_layer_geometry`` carries
    that fact forward so the emit path can act on it instead of rediscovering
    it. Both cases verified empirically.
    """
    if len(shape) == 4:
        return shape[1], shape[2], shape[3]
    if len(shape) == 5:
        c_outer, h, w, c_inner = shape[1], shape[2], shape[3], shape[4]
        return c_outer * c_inner, h, w
    return None


def _layer_geometry(graph: dict, node_idx: int) -> dict:
    """Recover ``{H,W,Cin,Cout,K,stride}`` for a conv node from the graph JSON.

    The shapes are authoritative -- they come from TVM's own type inference --
    so nothing here has to re-derive convolution arithmetic. ``K`` and
    ``stride`` come from the weight shape and the input/output ratio.

    Accepts both NCHW and NCHWc; see ``_nchw`` for why the 5-D case folds and
    when it must not.
    """
    attrs = graph["attrs"]
    shapes = attrs["shape"][1]
    row_ptr = graph["node_row_ptr"]
    node = graph["nodes"][node_idx]

    in_eids = [row_ptr[src] + slot for src, slot, _ in node["inputs"]]
    raw_in = shapes[in_eids[0]]                  # input  [N, Cin, H, W](c)
    raw_out = shapes[row_ptr[node_idx]]          # output [N, Cout, OH, OW](c)
    feat = _nchw(raw_in)
    out = _nchw(raw_out)
    if feat is None or out is None:
        return {}
    cin, h, w = feat
    cout, oh, ow = out

    # Kernel size, from the weight tensor.
    #
    # Do NOT assume the weights are input[1]. In a fused residual block the
    # second input is another feature map and the weights land at input[2]
    # (e.g. [1,16,56,56,4], [1,16,56,56,4], [16,16,3,3,4,4]) -- indexing [1]
    # blindly yields K=0 -- a kernel that does not exist. Identify it by RANK instead: weights are 4-D [Cout,Cin,KH,KW]
    # or 6-D NCHWc-blocked [Co/c,Ci/c,KH,KW,ci,co]; feature maps are 4-D/5-D
    # with a leading batch of 1, so require a non-1 leading dim to disambiguate
    # the 4-D case. KH sits at index 2 in both weight forms.
    k = 0
    for eid in in_eids[1:]:
        s = shapes[eid]
        if len(s) == 6 or (len(s) == 4 and s[0] != 1):
            k = s[2]
            break
    stride = max(1, h // oh) if oh else 1

    # Does the buffer need a channel repack before/after the AIE kernel?
    # NCHWc with C_outer == 1 is already byte-identical to HWC; anything bigger
    # interleaves and must be transposed at the data-movement boundary. Carried
    # here so the emit path acts on it instead of rediscovering it.
    def _blocked(s):
        return len(s) == 5 and s[1] != 1

    return {"H": h, "W": w, "Cin": cin, "Cout": cout,
            "K": k, "stride": stride,
            "in_elems": cin * h * w,
            "out_elems": cout * oh * ow,
            "in_layout": "NCHWc" if len(raw_in) == 5 else "NCHW",
            "out_layout": "NCHWc" if len(raw_out) == 5 else "NCHW",
            "needs_repack": _blocked(raw_in) or _blocked(raw_out)}


def _geometry_reject_reason(graph: dict, node_idx: int) -> str:
    """Say WHY the geometry was unusable, with the actual shapes."""
    shapes = graph["attrs"]["shape"][1]
    row_ptr = graph["node_row_ptr"]
    node = graph["nodes"][node_idx]
    in_eids = [row_ptr[src] + slot for src, slot, _ in node["inputs"]]
    feat = shapes[in_eids[0]]
    out = shapes[row_ptr[node_idx]]
    return (f"unsupported shapes: input {feat}, output {out} "
            f"(want rank 4 NCHW or rank 5 NCHWc)")


def select_layers(manifest: dict, graph: dict, indices=None,
                  op_kinds=AIE_OP_KINDS) -> list:
    """Pick the layers to offload. Returns a list of selection dicts.

    *manifest* is ``layers/manifest.json`` from stage 5 (so the per-layer
    folder names line up) and *graph* the stage-4 graph JSON (so geometry comes
    from TVM's shapes). A requested index whose op kind has no AIE path is
    returned with ``eligible=False`` and a reason rather than silently dropped
    -- asking for a layer and getting nothing back with no explanation is the
    failure mode worth avoiding.
    """
    # Graph nodes that are real kernel calls, in execution order.
    call_nodes = [i for i, n in enumerate(graph["nodes"])
                  if n.get("op") == "tvm_op"
                  and n["attrs"].get("func_name") != "__nop"]

    # Map manifest entry -> graph node by SYMBOL, not by position.
    #
    # `call_nodes[idx]` looks right and is not: the manifest has one entry per
    # emitted C function while the graph has one node per CALL, and a function
    # invoked twice appears once in the manifest and twice here. On the reference CNN
    # that is 28 vs 30, and the two desynchronize from index 6 onward -- every
    # later layer then gets another layer's geometry, silently. (The same bug
    # is called out in aiegraph_partition.py's docstring.)
    #
    # Symbols repeat, so consume them in order: the n-th manifest entry naming
    # a symbol takes the n-th graph node calling it.
    by_symbol = {}
    for ni in call_nodes:
        by_symbol.setdefault(graph["nodes"][ni]["attrs"]["func_name"], []).append(ni)
    taken = {}

    def _node_for(entry):
        sym = entry["symbol"]
        nodes = by_symbol.get(sym, [])
        seq = taken.get(sym, 0)
        if seq >= len(nodes):
            return None
        taken[sym] = seq + 1
        return nodes[seq]

    picked = []
    for entry in manifest["layers"]:
        idx = entry["index"]
        # Claim this entry's graph node BEFORE the index filter. _node_for
        # consumes repeated symbols in order, so skipping entries early would
        # hand the next selected layer an earlier layer's node -- making the
        # result depend on which --aie-layers were asked for.
        node_idx = _node_for(entry)
        if indices is not None and idx not in indices:
            continue
        kind = entry.get("kind", "")
        ops = entry.get("ops", [])
        # Map the split-layer name onto an aiegraph op kind.
        if "conv2d" in ops:
            aie_op = "conv_bn_relu" if "relu" in ops else "conv_bn"
        else:
            aie_op = None

        sel = {"index": idx, "dir": entry.get("dir", ""), "kind": kind,
               "symbol": entry["symbol"], "ops": ops, "aie_op": aie_op}

        if aie_op is None or aie_op not in op_kinds:
            sel.update(eligible=False,
                       reason=f"op kind {kind!r} is not a conv -- only convs "
                              f"offload through the AIE BYOC path")
            picked.append(sel)
            continue
        if node_idx is None:
            sel.update(eligible=False,
                       reason=f"no graph node calls {entry['symbol']!r}")
            picked.append(sel)
            continue

        geom = _layer_geometry(graph, node_idx)
        if not geom:
            sel.update(eligible=False, reason=_geometry_reject_reason(graph, node_idx))
            picked.append(sel)
            continue
        sel["node"] = node_idx
        sel.update(eligible=True, **geom)
        picked.append(sel)
    return picked


# ═══════════════════════════════════════════════════════════════════════════
#  Offload
# ═══════════════════════════════════════════════════════════════════════════

#: ``src/mlir/mlirfront/frontend`` -- parent of the ``aiebackend`` package.
_MLIR_FRONTEND_DIR = (Path(__file__).resolve().parents[2]
                      / "mlir" / "mlirfront" / "frontend")

_BACKEND = None


def aie_backend():
    """The AIE backend (``_aiebackend``), running in a child process.

    Returns an ``aiebackend.BackendProcess`` whose methods mirror the pybind
    module (``build_aiegraph_module`` / ``lower_aiegraph`` /
    ``run_aie_pipeline``). It is never imported in-process here: this package
    always has TVM loaded, and TVM's LLVM and the backend's static LLVM abort on
    duplicate command-line options (skill: aiebackendtvmllvm). Started once and
    reused; closed at interpreter exit.
    """
    global _BACKEND
    if _BACKEND is None:
        if str(_MLIR_FRONTEND_DIR) not in sys.path:
            sys.path.insert(0, str(_MLIR_FRONTEND_DIR))
        import aiebackend

        _BACKEND = aiebackend.spawn()
        atexit.register(_BACKEND.close)
    return _BACKEND


# ═══════════════════════════════════════════════════════════════════════════
#  Layer index -> conv identity
# ═══════════════════════════════════════════════════════════════════════════

_GEOM_KEYS = ("Cin", "Cout", "K", "stride")


def _weight_param(graph: dict, node_idx: int):
    """Name of the conv node's weight input (rank 6 NCHWc-blocked, or rank 4 OIHW)."""
    shapes = graph["attrs"]["shape"][1]
    row_ptr = graph["node_row_ptr"]
    for src, slot, _ in graph["nodes"][node_idx]["inputs"][1:]:
        s = shapes[row_ptr[src] + slot]
        if len(s) == 6 or (len(s) == 4 and s[0] != 1):
            return graph["nodes"][src]["name"]
    return None


def resolve_conv_targets(out_dir, indices=None, op_kinds=AIE_OP_KINDS) -> dict:
    """Stage-5 layer indices -> conv weight fingerprints, from the CPU build in *out_dir*.

    Returns ``{"ok", "reason"?, "selections", "targets": {layer: {fingerprint,
    geometry}}}``. Layers that are not convs come back in ``selections`` with
    ``eligible=False`` and a reason.
    """
    from tvm import relay

    from frontend.tvmrelay.byoc.aie_annotate import weight_fingerprint

    out_dir = Path(out_dir).resolve()
    manifest_path = out_dir / "layers" / "manifest.json"
    graph_path = next(out_dir.glob("*_graph.json"), None)
    params_path = next(out_dir.glob("*_params.bin"), None)
    if not manifest_path.is_file() or graph_path is None or params_path is None:
        return {"ok": False, "reason": "needs stage 4's *_graph.json + *_params.bin "
                                       "and stage 5's layers/manifest.json"}
    manifest = json.loads(manifest_path.read_text())
    graph = json.loads(graph_path.read_text())
    params = relay.load_param_dict(params_path.read_bytes())

    selections = select_layers(manifest, graph, indices, op_kinds)
    targets = {}
    for sel in selections:
        if not sel.get("eligible"):
            continue
        name = _weight_param(graph, sel["node"])
        if name is None or name not in params:
            sel.update(eligible=False, reason="cannot find the conv's weight param")
            continue
        targets[sel["index"]] = {
            "fingerprint": weight_fingerprint(params[name].numpy()),
            **{k: sel[k] for k in _GEOM_KEYS}}
    return {"ok": True, "selections": selections, "targets": targets}


def check_targets(targets: dict, relay_convs: list) -> list:
    """Every target must be exactly one Relay conv with the same geometry.

    Returns a list of error strings (empty = OK).
    """
    bad = []
    for layer, t in targets.items():
        hits = [c for c in relay_convs if c["fingerprint"] == t["fingerprint"]]
        if len(hits) != 1:
            bad.append(f"layer {layer}: {len(hits)} Relay convs carry its weights, want 1")
        elif any(hits[0][k] != t[k] for k in _GEOM_KEYS):
            bad.append(f"layer {layer}: graph {[t[k] for k in _GEOM_KEYS]} vs Relay "
                       f"{[hits[0][k] for k in _GEOM_KEYS]} (Cin,Cout,K,stride)")
    return bad
