###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""DETECT: which calls of the deployed network a kernel spec can replace, and why not.

A call matches when, in this order,

1. every ``match`` substring occurs in its TVM symbol;
2. it has exactly as many arguments as the spec;
3. each argument agrees on kind (``in`` activation / ``param`` constant /
   ``out``), dtype, rank and every dim -- symbolic dims bind on first use;
4. every ``require=`` holds on the constant's REAL values from ``params.bin``.

Step 4 is the part that cannot be done by looking at shapes, and it is the one
that bit the stem: TVM subtracts two zero-point terms where the AIE epilogue
has one field, so the kernel is exact only while the second is all-zero
(``require=zero``). It is checked per CALL, because two calls of one layer
carry different weights.

A LAYER (manifest index = one TVM symbol) is offloadable only if all of its
calls match: the driver swaps the call by symbol, so a layer cannot go to the
AIE for one call and stay on the APU for the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CallVerdict", "LayerVerdict", "match_call", "detect", "detect_layers"]


@dataclass
class CallVerdict:
    call: object                         # graph.LayerCall
    ok: bool
    reasons: list = field(default_factory=list)
    bindings: dict = field(default_factory=dict)

    @property
    def reason(self) -> str:
        return self.reasons[0] if self.reasons else "matches"


@dataclass
class LayerVerdict:
    layer: int
    symbol: str
    dir: str
    calls: list                          # [CallVerdict]

    @property
    def ok(self) -> bool:
        return bool(self.calls) and all(v.ok for v in self.calls)

    @property
    def reason(self) -> str:
        bad = [v for v in self.calls if not v.ok]
        if not bad:
            return f"matches ({len(self.calls)} call(s))"
        v = bad[0]
        return f"call [{v.call.call:02d}]: {v.reason}"


def _check_dims(arg, targ, binds: dict, reasons: list) -> None:
    where = f"arg {arg.index} ({arg.name or arg.kind})"
    if len(arg.dims) != len(targ.shape):
        reasons.append(f"{where}: rank {len(targ.shape)} {list(targ.shape)} vs "
                       f"spec {arg.shape_text()}")
        return
    for k, (want, got) in enumerate(zip(arg.dims, targ.shape)):
        got = int(got)
        if want == "*":
            continue
        if isinstance(want, int):
            if want != got:
                reasons.append(f"{where}: dim {k} is {got}, spec says {want} "
                               f"({list(targ.shape)} vs {arg.shape_text()})")
                return
            continue
        if want in binds and binds[want] != got:
            reasons.append(f"{where}: dim {k} is {got}, but {want} = "
                           f"{binds[want]} from an earlier argument")
            return
        binds.setdefault(want, got)


def _check_requires(arg, targ, net, reasons: list) -> None:
    import numpy as np

    vals = net.param_array(targ.param, targ.dtype, targ.shape).astype(np.int64)
    where = f"arg {arg.index} ({arg.name or targ.param})"
    for req in arg.requires:
        kind = req[0]
        if kind == "zero" and vals.any():
            nz = int(np.count_nonzero(vals))
            reasons.append(f"{where}: require=zero, but {nz}/{vals.size} values "
                           f"are non-zero (e.g. {int(vals.flat[np.flatnonzero(vals)[0]])})")
        elif kind == "nonzero" and not vals.all():
            reasons.append(f"{where}: require=nonzero, but "
                           f"{int(vals.size - np.count_nonzero(vals))} values are 0")
        elif kind == "uniform" and vals.size and (vals != vals.flat[0]).any():
            reasons.append(f"{where}: require=uniform, but it ranges "
                           f"[{int(vals.min())}, {int(vals.max())}]")
        elif kind == "nonneg" and (vals < 0).any():
            reasons.append(f"{where}: require=nonneg, but min is {int(vals.min())}")
        elif kind == "range" and vals.size and (vals.min() < req[1] or vals.max() > req[2]):
            reasons.append(f"{where}: require=range({req[1]},{req[2]}), but values "
                           f"span [{int(vals.min())}, {int(vals.max())}]")


def match_call(spec, call, net, check_values: bool = True) -> CallVerdict:
    """Does *spec* fit *call*? All reasons, not just the first."""
    reasons, binds = [], {}
    missing = [s for s in spec.match if s not in call.symbol]
    if missing:
        reasons.append(f"symbol lacks {', '.join(repr(s) for s in missing)}")
    if len(call.args) != len(spec.args):
        n_in = sum(1 for a in call.args if a.kind != "out")
        reasons.append(f"{len(call.args)} args ({n_in} in + "
                       f"{len(call.args) - n_in} out), spec has {len(spec.args)} "
                       f"({len(spec.inputs)} + {len(spec.outputs)})")
        return CallVerdict(call, False, reasons)
    for arg, targ in zip(spec.args, call.args):
        where = f"arg {arg.index} ({arg.name or arg.kind})"
        if arg.kind != targ.kind:
            what = {"param": "a constant", "in": "an activation", "out": "an output"}
            reasons.append(f"{where}: is {what[targ.kind]}, spec says {arg.kind}")
            continue
        if arg.dtype != targ.dtype:
            reasons.append(f"{where}: dtype {targ.dtype}, spec says {arg.dtype}")
            continue
        _check_dims(arg, targ, binds, reasons)
    if check_values and not reasons:
        for arg, targ in zip(spec.args, call.args):
            if arg.requires:
                _check_requires(arg, targ, net, reasons)
    return CallVerdict(call, not reasons, reasons, binds)


def detect(spec, net, check_values: bool = True) -> list:
    """A :class:`CallVerdict` for every kernel call in the network."""
    return [match_call(spec, c, net, check_values) for c in net.calls]


def detect_layers(spec, net, layers=None, check_values: bool = True) -> list:
    """One :class:`LayerVerdict` per manifest layer (all of them, or *layers*)."""
    out = []
    for layer, calls in sorted(net.layers().items()):
        if layers is not None and layer not in layers:
            continue
        out.append(LayerVerdict(layer=layer, symbol=calls[0].symbol, dir=calls[0].dir,
                                calls=[match_call(spec, c, net, check_values)
                                       for c in calls]))
    return out
