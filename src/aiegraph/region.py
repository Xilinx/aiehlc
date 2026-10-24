###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Model-level region names (``"layer1.0.conv1"``) -> the TVM call that runs them.

A *region* is one weighted node of the int8 ONNX model the flow compiled
(``<workdir>/*.int8.qdq.onnx``): a ``Conv`` or a ``Gemm``. Its names:

    torch export   "/layer1/layer1.0/conv1/Conv"   -> "layer1.0.conv1"
    ONNX zoo       "resnetv15_stage1_conv0_fwd"    -> "resnetv15_stage1_conv0"
                                                      + alias "layer1.0.conv1"

The zoo ResNet is gluon ``resnet*_v1``; its basic-block convs also answer to
torchvision's module paths, so one script runs on either source.

The region -> call map is by **weight fingerprint** (values sorted, cast int64,
hashed -- ``byoc.aie_annotate.weight_fingerprint``): layout- and dtype-
independent, so ONNX ``int8[64,3,7,7]`` OIHW and TVM's ``int8[16,1,7,7,3,4]``
OIHW3i4o hash identically. Call order is NOT used: TVM schedules downsample
convs differently from the ONNX node order.

A TVM *layer* is a distinct kernel and may serve several calls
(``layer1.0.conv1`` and ``layer1.1.conv1`` share one), and the board driver
swaps a kernel by symbol -- so offloading a region offloads its siblings.
``Region.siblings`` names them.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AiegraphError

__all__ = ["Region", "Regions", "build_regions"]

_ONNX_DTYPE = {2: "uint8", 3: "int8", 5: "int16", 6: "int32", 1: "float32"}
_GLUON = re.compile(r"(?P<net>resnetv1\d*)_(?:stage(?P<stage>\d+)_)?(?P<op>conv|dense)(?P<k>\d+)$")


def _fingerprint(arr) -> str:
    import numpy as np
    a = np.sort(np.asarray(arr).astype(np.int64).reshape(-1))
    return hashlib.sha1(a.tobytes()).hexdigest()


@dataclass
class Region:
    name: str                       # canonical (from the ONNX node name)
    node: str                       # ONNX node name
    op: str                         # conv2d | dense
    weight_shape: tuple
    weight_dtype: str
    quantization: str               # per_channel | per_tensor
    input_layout: str               # NCHW | NC
    weight_layout: str              # OIHW | OI | IO
    aliases: list = field(default_factory=list)
    calls: list = field(default_factory=list)   # TVM call indices
    layer: int = None               # TVM layer (manifest index)
    siblings: list = field(default_factory=list)  # other regions on the same layer

    @property
    def names(self) -> list:
        return [self.name] + self.aliases

    @property
    def label(self) -> str:
        """``layer1.0.conv1 (resnetv15_stage1_conv0)`` -- the alias users type first."""
        return f"{self.aliases[0]} ({self.name})" if self.aliases else self.name

    @property
    def where(self) -> str:
        calls = ",".join(f"[{c:02d}]" for c in self.calls)
        return f"layer{self.layer} call {calls}" if self.layer is not None else "unmapped"


class Regions(list):
    """All regions of a converted model; prints as a table."""

    def find(self, ref) -> list:
        """Exact name/alias -> one region; ``"layer1"`` -> every region under it;
        ``"layer1.*.conv1"`` -> glob; ``"call[08]"`` -> the region that call runs."""
        if isinstance(ref, Region):
            return [ref]
        if isinstance(ref, (list, tuple, set)):
            out = []
            for r in ref:
                out += [x for x in self.find(r) if x not in out]
            return out
        text = str(ref).strip()
        hit = [r for r in self if text in r.names]
        if not hit:
            hit = [r for r in self if any(n.startswith(text + ".") for n in r.names)]
        if not hit and any(ch in text for ch in "*?["):
            hit = [r for r in self if any(fnmatch.fnmatchcase(n, text) for n in r.names)]
        c = re.fullmatch(r"call\[?(\d+)\]?", text)
        if not hit and c:
            hit = [r for r in self if int(c.group(1)) in r.calls]
        if not hit:
            raise AiegraphError("region", f"no region matches {ref!r}",
                                hint="print(module.regions) lists them; refs: 'conv1', "
                                     "'layer1.0.conv1', 'layer1' (all under it), "
                                     "'layer2.*.conv1', 'call[08]'")
        return hit

    def __repr__(self) -> str:
        w = max([len(r.name) for r in self] + [6])
        lines = [f"{len(self)} regions (weighted ops of the int8 model)",
                 f"  {'region':<{w}}  {'op':<6} {'weight':<22} {'quant':<11} tvm"]
        for r in self:
            shape = f"{r.weight_dtype}[{','.join(map(str, r.weight_shape))}]"
            alias = f"  (= {', '.join(r.aliases)})" if r.aliases else ""
            lines.append(f"  {r.name:<{w}}  {r.op:<6} {shape:<22} {r.quantization:<11} "
                         f"{r.where}{alias}")
        return "\n".join(lines)


def _canonical(node: str) -> str:
    """``/layer1/layer1.0/conv1/Conv`` -> ``layer1.0.conv1``; ``x_fwd`` -> ``x``."""
    if node.startswith("/"):
        # Each scope repeats its parent's attribute: /layer1/layer1.0/...,
        # /downsample/downsample.0/ -- so a part may replace the last one.
        path = []
        for part in [p for p in node.split("/") if p][:-1]:
            if path and part.startswith(path[-1] + "."):
                path[-1] = part
            else:
                path.append(part)
        return ".".join(path) or node
    return node[:-4] if node.endswith("_fwd") else node


def _gluon_aliases(regions: list) -> None:
    """Torchvision module paths for a gluon ResNet v1 built of basic blocks."""
    stages = {}
    for r in regions:
        m = _GLUON.fullmatch(r.name)
        if not m:
            continue
        if m["op"] == "dense":
            r.aliases.append("fc")
        elif m["stage"] is None:
            r.aliases.append("conv1")
        else:
            stages.setdefault(int(m["stage"]), []).append((int(m["k"]), r))
    for s, convs in stages.items():
        if any(len(r.weight_shape) != 4 or r.weight_shape[2] not in (1, 3) for _, r in convs):
            continue                                   # not a basic-block stage
        block, slot = 0, 0
        for _, r in sorted(convs, key=lambda kr: kr[0]):
            if r.weight_shape[2] == 1:
                r.aliases.append(f"layer{s}.{block}.downsample.0")
                continue
            if slot == 2:
                block, slot = block + 1, 0
            slot += 1
            r.aliases.append(f"layer{s}.{block}.conv{slot}")


def _onnx_regions(path: Path) -> list:
    import onnx
    from onnx import numpy_helper

    proto = onnx.load(str(path))
    init = {i.name: i for i in proto.graph.initializer}
    prod = {o: n for n in proto.graph.node for o in n.output}
    out = []
    for n in proto.graph.node:
        if n.op_type not in ("Conv", "Gemm") or len(n.input) < 2:
            continue
        dq = prod.get(n.input[1])
        w = init.get(dq.input[0]) if dq is not None and dq.op_type == "DequantizeLinear" else None
        if w is None:
            continue
        scale = init.get(dq.input[1])
        per_ch = scale is not None and len(scale.dims) == 1 and scale.dims[0] > 1
        if n.op_type == "Conv":
            op, in_l, w_l = "conv2d", "NCHW", "OIHW"
        else:
            trans_b = any(a.name == "transB" and a.i for a in n.attribute)
            op, in_l, w_l = "dense", "NC", "OI" if trans_b else "IO"
        r = Region(name=_canonical(n.name), node=n.name, op=op, weight_shape=tuple(w.dims),
                   weight_dtype=_ONNX_DTYPE.get(w.data_type, str(w.data_type)),
                   quantization="per_channel" if per_ch else "per_tensor",
                   input_layout=in_l, weight_layout=w_l)
        r._fp = _fingerprint(numpy_helper.to_array(w))
        out.append(r)
    return out


def build_regions(workdir, network) -> Regions:
    """Regions of the deploy tree *workdir*, mapped onto *network*'s calls."""
    workdir = Path(workdir)
    qdq = next(workdir.glob("*.int8.qdq.onnx"), None)
    if qdq is None:
        raise AiegraphError("region", f"no *.int8.qdq.onnx in {workdir}",
                            hint="regions need the int8 flow (ag.convert(..., quantize='int8'))")
    regions = _onnx_regions(qdq)
    _gluon_aliases(regions)
    by_fp = {}
    for call in network.calls:
        for a in call.args:
            if a.kind == "param" and a.dtype in ("int8", "uint8") and len(a.shape) >= 2:
                fp = _fingerprint(network.param_array(a.param, a.dtype, a.shape))
                by_fp.setdefault(fp, []).append(call)
    for r in regions:
        calls = by_fp.get(r._fp, [])
        r.calls = sorted({c.call for c in calls})
        layers = {c.layer for c in calls if c.layer is not None}
        r.layer = min(layers) if len(layers) == 1 else None
    by_layer = {}
    for r in regions:
        if r.layer is not None:
            by_layer.setdefault(r.layer, []).append(r)
    for r in regions:
        r.siblings = [o.label for o in by_layer.get(r.layer, []) if o is not r]
    return Regions(regions)
