###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The deployed network as a list of kernel CALLS with typed arguments. TVM-free.

Everything comes from the files ``deploy_flow.py`` already wrote:

* ``<stem>_graph.json``  -- nodes, shapes, dtypes (TVM's own type inference);
* ``<stem>_params.bin``  -- the constants, read with ``arm_build.param_offsets``
  (no TVM deserializer);
* ``layers/manifest.json`` -- the stage-5 folder index of each kernel SYMBOL.

Three numbering spaces, kept apart on purpose (CLAUDE.md: "three numbering
spaces are all called layer 01"):

``call``   execution-order index over real kernel calls -- ``[NN]`` in a
           ``TRACE=1`` dump, ``l<NN>_`` in ``layeriohex/``;
``layer``  the manifest/folder index ``NN_`` -- one per distinct SYMBOL, which
           is also what ``--aie-layers`` and this package's ``--layers`` take;
``node``   the graph-JSON node id.

A layer maps to one or more calls (ResNet-18 calls layer 04's kernel twice),
and the generated driver swaps by SYMBOL, so selecting a layer offloads every
call of it -- each call has its own weights, and each is checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["TensorArg", "LayerCall", "Network", "load_network"]

_NP_DTYPE = {"int8": "i1", "int16": "<i2", "int32": "<i4", "int64": "<i8",
             "uint8": "u1", "uint16": "<u2", "uint32": "<u4", "uint64": "<u8",
             "float16": "<f2", "float32": "<f4", "float64": "<f8"}


@dataclass
class TensorArg:
    slot: int                   # position in the packed call
    kind: str                   # in | param | out
    dtype: str
    shape: tuple
    eid: int                    # graph entry id
    param: str = None           # graph parameter name, for kind == param
    producer: int = None        # producing call index, for kind == in

    @property
    def nbytes(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n * _bytes(self.dtype)

    def __str__(self) -> str:
        shape = ",".join(str(d) for d in self.shape)
        extra = f" {self.param}" if self.param else ""
        return f"{self.kind} {self.dtype}[{shape}]{extra}"


@dataclass
class LayerCall:
    call: int
    node: int
    symbol: str
    layer: int                  # manifest index, None if the manifest lacks it
    dir: str
    args: list = field(default_factory=list)

    @property
    def label(self) -> str:
        layer = f"{self.layer:02d}" if self.layer is not None else "??"
        return f"layer {layer} call [{self.call:02d}]"


def _bytes(dtype: str) -> int:
    return max(1, int("".join(ch for ch in dtype if ch.isdigit())) // 8)


class Network:
    """One deployed graph, read from ``out_dir``."""

    def __init__(self, out_dir, stem: str = None):
        self.out_dir = Path(out_dir).resolve()
        graph_path = (self.out_dir / f"{stem}_graph.json" if stem
                      else next(self.out_dir.glob("*_graph.json"), None))
        if graph_path is None or not graph_path.is_file():
            raise FileNotFoundError(f"no *_graph.json in {self.out_dir} -- run "
                                    f"deploy_flow.py first")
        self.stem = graph_path.name[:-len("_graph.json")]
        self.graph_path = graph_path
        self.params_path = self.out_dir / f"{self.stem}_params.bin"
        self.graph = json.loads(graph_path.read_text())
        mpath = self.out_dir / "layers" / "manifest.json"
        self.manifest = json.loads(mpath.read_text()) if mpath.is_file() else {"layers": []}
        self.offsets = self._offsets()
        self.calls = self._calls()
        self._cache = {}

    # ── construction ──────────────────────────────────────────────────────

    def _offsets(self) -> dict:
        if not self.params_path.is_file():
            return {}
        from frontend.tvmrelay.arm_build import param_offsets
        return param_offsets(self.params_path)

    def _calls(self) -> list:
        g = self.graph
        nodes, rp = g["nodes"], g["node_row_ptr"]
        shapes, dtypes = g["attrs"]["shape"][1], g["attrs"]["dltype"][1]
        by_symbol = {e["symbol"]: e for e in self.manifest.get("layers", [])}
        producer, calls = {}, []
        for idx, node in enumerate(nodes):
            if node.get("op") != "tvm_op":
                continue
            fn = node["attrs"]["func_name"]
            if fn == "__nop":
                # Layout-free passthrough: its output IS its input's buffer.
                src, slot, _ = node["inputs"][0]
                producer[rp[idx]] = producer.get(rp[src] + slot)
                continue
            entry = by_symbol.get(fn, {})
            call = LayerCall(call=len(calls), node=idx, symbol=fn,
                             layer=entry.get("index"), dir=entry.get("dir", ""))
            for src, slot, _ in node["inputs"]:
                eid = rp[src] + slot
                name = nodes[src]["name"]
                is_param = nodes[src].get("op") == "null" and name in self.offsets
                call.args.append(TensorArg(
                    slot=len(call.args), kind="param" if is_param else "in",
                    dtype=dtypes[eid], shape=tuple(shapes[eid]), eid=eid,
                    param=name if is_param else None,
                    producer=None if is_param else producer.get(eid)))
            for k in range(int(node["attrs"].get("num_outputs", 1))):
                eid = rp[idx] + k
                producer[eid] = call.call
                call.args.append(TensorArg(slot=len(call.args), kind="out",
                                           dtype=dtypes[eid], shape=tuple(shapes[eid]),
                                           eid=eid))
            calls.append(call)
        return calls

    # ── queries ───────────────────────────────────────────────────────────

    def layers(self) -> dict:
        """``{layer index: [LayerCall, ...]}`` for every manifest layer that is called."""
        out = {}
        for c in self.calls:
            if c.layer is not None:
                out.setdefault(c.layer, []).append(c)
        return out

    def param_array(self, name: str, dtype: str, shape):
        """A graph constant as a numpy array (read-only view into params.bin)."""
        import numpy as np

        key = name
        if key not in self._cache:
            off, nbytes = self.offsets[name]
            if "_blob" not in self._cache:
                self._cache["_blob"] = self.params_path.read_bytes()
            arr = np.frombuffer(self._cache["_blob"], dtype=_NP_DTYPE[dtype],
                                count=nbytes // _bytes(dtype), offset=off)
            self._cache[key] = arr.reshape(tuple(int(d) for d in shape))
        return self._cache[key]

    def param_bytes(self, name: str) -> bytes:
        off, nbytes = self.offsets[name]
        if "_blob" not in self._cache:
            self._cache["_blob"] = self.params_path.read_bytes()
        return self._cache["_blob"][off:off + nbytes]

    def layer_source(self, layer: int):
        """The stage-5 ``layers/NN_op/NN_op.c`` holding *layer*'s TVM kernel."""
        for e in self.manifest.get("layers", []):
            if e["index"] == layer:
                p = self.out_dir / "layers" / e["path"]
                return p if p.is_file() else None
        return None


def load_network(out_dir, stem: str = None) -> Network:
    return Network(out_dir, stem)
