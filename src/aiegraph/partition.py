###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``module.partition()``: turn ``module.assign(...)`` calls into a placement.

The partition is the WHOLE placement, not a delta: every layer that is on the
AIE from an earlier run goes back to the A78 first, then the assigned regions
that pass their contract go through ``aiekernel.flow.offload`` (gates lint ..
link). A region that fails anywhere runs on ``fallback`` (``"cpu"``), or, with
``fallback=None``, the call raises.

Granularity is the TVM kernel (``Region.layer``): the board driver swaps by
symbol, so regions sharing a kernel move together -- two of them assigned to
different backends is a conflict, and an unassigned one is reported as dragged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import _env
from .contract import check_region
from .errors import AiegraphError
from .plan import PLAN_MODES, plan_conv_layers
from .report import PartitionPlan

__all__ = ["run_partition", "reset_aie_rows", "plan_partition_launches",
           "ConvPattern", "match_conv", "conv_epilogue_ref", "AIEGRAPH_OP_KINDS"]

#: Every op the aiegraph dialect defines (``td/aiegraphop.td``).
AIEGRAPH_OP_KINDS = ("conv_bn_relu", "conv_bn", "residual_add_relu", "avgpool_fc")

FALLBACKS = ("cpu", None)
OBJECTIVES = ("latency", None)


# ═══════════════════════════════════════════════════════════════════════════
#  Conv pattern family: conv (+bias) (+relu) | conv (+bias) (+add ...) no relu
# ═══════════════════════════════════════════════════════════════════════════
#
# A fused conv layer is described by its op list -- real Relay op names
# (``conv2d``, ``bias_add``, ``add``, ``relu``) or the underscore tokens of the
# fused TVM function name (``contrib_conv2d_NCHWc_add_add`` -> ``conv2d``,
# ``NCHWc``, ``add``, ``add``). ``match_conv`` maps it onto aiegraph ops:
#
#   conv (+bias)                         -> conv_bn
#   conv (+bias) + relu                  -> conv_bn_relu
#   conv (+bias) + add + relu            -> conv_bn + residual_add_relu (ResNet tail)
#   conv (+bias) + add [+ add] (no relu) -> conv_bn with an fp32 epilogue:
#       UNet timestep add (per channel), skip / shortcut / attention residual
#       (same shape as the output). The adds run on the fp32 accumulator BEFORE
#       any bf16/int8 cast -- the rule of the board host path (unet_aie.py
#       uc_conv epilogue, cos 1.000000); casting first and adding after is a
#       different number (see ``conv_epilogue_ref``).
#
# Anything else (a multiply / sigmoid / divide fused in, a relu after two joins,
# an add of a shape that is neither per-channel nor the output's) is None:
# unmapped, never mis-mapped with the extra op dropped.

#: QNN requant tokens: in these graphs a ReLU is folded into the requant clip.
_QNN_TOKENS = ("fixed", "fixed_point_multiply", "requantize", "qnn", "clip", "cli")
#: Float element-wise ops that are NOT part of the conv(+bias)(+add)(+relu) family.
_FOREIGN_TOKENS = ("multiply", "divide", "subtract", "sigmoid", "tanh", "exp", "erf",
                   "power", "sqrt", "rsqrt", "mean", "variance", "silu", "gelu",
                   "clip", "maximum", "minimum", "negative")
#: Bound on fused epilogue adds (UNet has at most one join per conv).
MAX_EPILOGUE_ADDS = 2


@dataclass(frozen=True)
class ConvPattern:
    """How one fused conv layer maps onto aiegraph ops.

    ``aiegraph``: the dialect ops, in order. ``adds``: the epilogue joins, each
    ``"channel"`` (per-output-channel, e.g. UNet temb), ``"tensor"`` (same shape
    as the output: skip / residual) or ``"join"`` (shape not given; resolved at
    run time by element count, as the board host path does). ``epilogue_fp32``:
    the bias and adds are applied to the fp32 accumulator before the output cast.
    """
    aiegraph: tuple
    bias: bool
    relu: bool
    adds: tuple = ()
    qnn: bool = False
    epilogue_fp32: bool = True

    @property
    def family(self) -> str:
        parts = ["conv"] + (["bias"] if self.bias else []) + [f"add:{a}" for a in self.adds]
        return "+".join(parts + (["relu"] if self.relu else []))


def _add_kind(shape, out_shape) -> str | None:
    """``"channel"`` / ``"tensor"`` for an epilogue operand, None if neither."""
    if shape is None or out_shape is None:
        return "join"
    s, o = [int(x) for x in shape], [int(x) for x in out_shape]
    if s == o:
        return "tensor"
    n = 1
    for x in s:
        n *= x
    if len(o) == 4:                                  # NCHW
        cout = o[1]
        if n == cout and (len(s) <= 1 or (len(s) == 4 and s[1] == cout)
                          or (len(s) == 3 and s[0] == cout)):
            return "channel"
    elif len(o) == 5:                                # NCHWc
        cout = o[1] * o[4]
        if n == cout and len(s) == 5 and s[1] == o[1] and s[4] == o[4]:
            return "channel"
    return None


def match_conv(ops, add_shapes=None, out_shape=None) -> ConvPattern | None:
    """Map a fused conv layer's op list onto aiegraph ops (None: not this family).

    *add_shapes* (optional) are the shapes of the non-bias ``add`` operands in
    order and *out_shape* the conv output shape; with them each join is typed
    ``channel`` / ``tensor`` and an add of any other shape is rejected.
    """
    ops = [str(o) for o in ops]
    if "conv2d" not in ops:
        return None
    relu = "relu" in ops
    qnn = any(t in ops for t in _QNN_TOKENS)
    explicit_bias = "bias_add" in ops or "bias" in ops
    adds = ops.count("add")
    if "bias_add" in ops:
        joins = adds
    else:
        # tokens "bias" + "add", or a plain add that is the bias (BN folded /
        # bias_add legalized to add under NCHWc): exactly one bias is assumed.
        joins = max(adds - 1, 0)
    bias = explicit_bias or adds > 0
    if qnn:
        # ResNet-18 QNN: a ReLU cannot be told apart from int8 saturation, so a
        # single join is the block tail (unchanged from the 1c7316a mapping).
        if joins == 0:
            return ConvPattern(("conv_bn_relu" if relu else "conv_bn",), bias, relu, qnn=True)
        if joins == 1:
            return ConvPattern(("conv_bn", "residual_add_relu"), bias, True, ("tensor",), qnn=True)
        return None
    if any(t in ops for t in _FOREIGN_TOKENS):
        return None
    if add_shapes is not None and len(add_shapes) != joins:
        return None
    kinds = tuple(_add_kind(add_shapes[i] if add_shapes is not None else None, out_shape)
                  for i in range(joins))
    if any(k is None for k in kinds):
        return None
    if joins == 0:
        return ConvPattern(("conv_bn_relu" if relu else "conv_bn",), bias, relu)
    if relu:
        # add + relu: the residual_add_relu op, only for one same-shape join.
        if joins == 1 and kinds[0] in ("tensor", "join"):
            return ConvPattern(("conv_bn", "residual_add_relu"), bias, True, kinds)
        return None
    if joins > MAX_EPILOGUE_ADDS:
        return None
    # No relu: the joins are fp32 epilogue adds of conv_bn (UNet temb / skip /
    # attention residual) -- never a residual_add_relu (that would add a ReLU).
    return ConvPattern(("conv_bn",), bias, False, kinds)


def conv_epilogue_ref(acc, bias=None, adds=(), out_dtype="fp32"):
    """Reference epilogue: fp32 ``acc + bias + adds`` (in order), then ONE cast.

    *acc* is the conv accumulator ``[N, C, ...]`` (fp32), *bias* ``[C]``, each
    add either per-channel (``C`` elements) or the output's shape. ``out_dtype``
    ``"bf16"`` rounds the final fp32 sum to bf16 (RNE), returned as float32.
    """
    import numpy as np
    y = np.asarray(acc, dtype=np.float32).copy()
    c = y.shape[1]
    bshape = (1, c) + (1,) * (y.ndim - 2)
    if bias is not None:
        y += np.asarray(bias, dtype=np.float32).reshape(bshape)
    for a in adds:
        a = np.asarray(a, dtype=np.float32)
        y += a.reshape(bshape) if a.size == c else a.reshape(y.shape)
    if out_dtype == "bf16":
        y = _bf16_rne(y)
    elif out_dtype != "fp32":
        raise AiegraphError("partition", f"out_dtype={out_dtype!r}", hint="'fp32' or 'bf16'")
    return y


def _bf16_rne(x):
    import numpy as np
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def reset_aie_rows(workdir) -> int:
    """Put every AIE row of ``layers/partition.json`` back on the CPU."""
    path = workdir / "layers" / "partition.json"
    if not path.is_file():
        return 0
    doc = json.loads(path.read_text())
    n = 0
    for row in doc.get("layers", []):
        if row.get("target") == "aie":
            row.update(target="cpu", entry=None, replaces=None,
                       reason="not assigned to the AIE (aiegraph partition)")
            n += 1
    rows = doc.get("layers", [])
    doc["aie_count"], doc["cpu_count"] = 0, len(rows)
    doc["aie_dirs"], doc["cpu_dirs"] = 0, len({r["dir"] for r in rows})
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return n


def _failures(emod, requested, by_layer) -> dict:
    """Region name -> why it cannot go where it was assigned (before any gate)."""
    fail = {}
    for r, backend in requested:
        if backend == "cpu":
            continue
        bad = check_region(emod._kernels[backend[4:]], r)
        if bad:
            fail[r.name] = "contract: " + "; ".join(bad)
    for layer, group in by_layer.items():
        if len({b for _, b in group}) > 1:
            for r, b in group:
                if b != "cpu":
                    others = ", ".join(f"{o.label}->{ob}" for o, ob in group if o is not r)
                    fail.setdefault(r.name, f"conflict: tvm layer{layer} also runs {others}; "
                                            f"one kernel, one backend")
    return fail


def _placed(emod) -> dict:
    """TVM layer -> ``"cpu"`` / ``"aie:<registered name>"`` from partition.json."""
    names = {f"aie:{k.spec.name}": f"aie:{k.name}" for k in emod._kernels.values()}
    return {l.index: names.get(l.placement, l.placement) for l in emod.layers}


def _gate_text(rep, layer) -> str:
    row = next((r for r in (rep.rows if rep else []) if r[1] == layer), None)
    return row[3] if row and not row[2] else "not swapped (see the offload report)"


def run_partition(emod, fallback="cpu", optimize_for="latency", verbose=False) -> PartitionPlan:
    if fallback not in FALLBACKS:
        raise AiegraphError("partition", f"fallback={fallback!r}",
                            hint="'cpu' (failing regions stay on the A78) or None (raise)")
    if optimize_for not in OBJECTIVES:
        raise AiegraphError("partition", f"optimize_for={optimize_for!r}",
                            hint="'latency' or None; there is no cost model for anything else")
    if not emod._assign:
        raise AiegraphError("partition", "nothing assigned",
                            hint="module.assign(region=..., backend='aie:<kernel>')")
    regions = {r.name: r for r in emod.regions}
    requested = [(regions[n], b) for n, b in emod._assign.items()]
    by_layer = {}
    for r, b in requested:
        if r.layer is not None:
            by_layer.setdefault(r.layer, []).append((r, b))
    fail = _failures(emod, requested, by_layer)
    if fail and fallback is None:
        name, why = next(iter(fail.items()))
        raise AiegraphError("partition", f"{regions[name].label}: {why}", hint="fallback='cpu' keeps it on the A78")

    groups = {}
    for r, b in requested:
        if b != "cpu" and r.name not in fail:
            groups.setdefault(b[4:], set()).add(r.layer)
    reset_aie_rows(emod.workdir)
    rep = None
    if groups:
        spec = [(sorted(layers), str(emod._kernels[k].source)) for k, layers in groups.items()]
        rep = emod.offload(spec, verbose=verbose)
    else:
        _env.frontend("arm_build").build_arm_elf(
            emod.workdir, c_name=f"{emod._stem()}.c", run_make=False, image=emod.image,
            verbose=False)

    placed = _placed(emod)
    rows = []
    for r, b in requested:
        where = placed.get(r.layer, "cpu")
        why = "" if where == b else fail.get(r.name) or _gate_text(rep, r.layer)
        rows.append((r, b, where, why))
    assigned = {r.name for r, _ in requested}
    dragged = [(r, placed[r.layer], ", ".join(o.label for o, _ in by_layer[r.layer]))
               for r in regions.values()
               if r.name not in assigned and r.layer in by_layer
               and placed.get(r.layer, "cpu") != "cpu"]
    plan = PartitionPlan(rows, fallback, optimize_for, rep, dragged)
    if fallback is None and not plan.ok:
        raise AiegraphError("partition", plan.report().splitlines()[0],
                            hint="print the plan: module.partition(fallback='cpu').report()")
    return plan


def plan_partition_launches(model, partition, mode: str = "kincore") -> dict:
    """Launch plan of the AIE conv layers of a partition (``{"layers": [{"node", "mkn"?, ...}]}``
    or a path to that JSON) on ``model`` (ONNX path / ModelProto).

    ``mode="kincore"``: one launch per layer, K accumulated in the core (plan.kincore_geom);
    ``mode="tile"``: one launch per 64x64x64 tile. The GEMM geometry comes from the ONNX graph;
    a partition row that carries ``mkn`` must agree with it (else AiegraphError).
    Returns ``dict(layers=[...], launches_planned=N, tile_launches=T, fallbacks=[node...])``."""
    from pathlib import Path
    if mode not in PLAN_MODES:
        raise AiegraphError("plan", f"mode={mode!r}", hint=f"one of {PLAN_MODES}")
    doc = json.loads(Path(partition).read_text()) if isinstance(partition, (str, Path)) else partition
    rows = doc.get("layers", [])
    plans = plan_conv_layers(model, [r["node"] for r in rows], mode)
    for r, p in zip(rows, plans):
        mkn = r.get("mkn")
        if mkn and p.get("m") is not None and [list(x) for x in mkn] != [[p["m"], p["k"], p["n"]]]:
            raise AiegraphError("plan", f"{r['node']}: partition mkn {mkn} != graph "
                                        f"{[p['m'], p['k'], p['n']]}",
                                hint="the partition was made for another model / export")
    ok = [p for p in plans if p["launches"] is not None]
    return dict(layers=plans, mode=mode,
                launches_planned=sum(p["launches"] for p in ok),
                tile_launches=sum(p["tile_launches"] for p in ok),
                fallbacks=[p["node"] for p in plans if p["mode"] != mode])
