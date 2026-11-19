###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``module.register_kernel(...)``: a named kernel and the regions it accepts.

Two descriptions of one kernel, checked against each other:

* the **contract** the user writes (``op``, ``constraints``) is model-level --
  it describes the ONNX region the kernel replaces (``Conv`` is NCHW/OIHW by
  definition, whatever layout TVM later picks);
* the **AIEK_SPEC** in the source is call-level -- exact TVM tensors
  (``int8[1,1,230,230,3]`` NCHWc ...); ``check_offload`` matches it with the
  aiekernel gates (detect / calibrate / emulate).

``register_kernel`` refuses a contract its own spec contradicts (op family,
weight dtype); ``check_region`` reports where a region breaks the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import _env
from .errors import AiegraphError

__all__ = ["KernelContract", "make_contract", "check_region", "CONSTRAINTS"]

#: constraint key -> allowed values (and the Region attribute it is checked on)
CONSTRAINTS = {
    "input_layout": (("NCHW", "NC"), "input_layout"),
    "weight_layout": (("OIHW", "OI", "IO"), "weight_layout"),
    "dtype": (("int8", "uint8"), "weight_dtype"),
    "quantization": (("per_channel", "per_tensor"), "quantization"),
}
OPS = ("conv2d", "dense")
KERNEL_TARGETS = ("aie", "a78")     # a78: the board ELF that launches the AIE kernel


@dataclass
class KernelContract:
    name: str
    source: Path
    op: str
    target: str
    constraints: dict
    spec: object                    # aiekernel KernelSpec
    lint: object                    # aiekernel LintReport
    notes: list = field(default_factory=list)

    @property
    def backend(self) -> str:
        return f"aie:{self.name}"

    def __repr__(self) -> str:
        cons = ", ".join(f"{k}={v}" for k, v in self.constraints.items()) or "none"
        lines = [f"kernel {self.name}  ({self.backend})  op {self.op}  target {self.target}",
                 f"  source: {self.source}",
                 f"  constraints: {cons}",
                 f"  AIEK_SPEC {self.spec.name}: op {self.spec.op or '?'}, "
                 f"match {', '.join(self.spec.match) or '-'}, {len(self.spec.args)} args"]
        lines += [f"  note: {n}" for n in self.notes]
        lines.append(f"  next: module.check_offload(region=..., kernel={self.name!r})")
        return "\n".join(lines)


def op_family(spec) -> str:
    """``conv2d`` / ``dense`` from the spec's ``op`` and ``match`` lines, or ''."""
    text = " ".join([spec.op or ""] + list(spec.match))
    return next((op for op in OPS if op in text), "")


def logical_weight(spec, op: str) -> "tuple | None":
    """The spec's weight as model-level dims -- ``(O, I, KH, KW)`` for a conv,
    ``(O, I)`` for a dense -- each an int, or None where the spec leaves it open.

    TVM packs weights (``OIHW{i}i{o}o`` = ``[O/o, I/i, KH, KW, i, o]``, dense
    ``[O/n, I, n]``); a region's weight is ONNX OIHW / OI. Returns None when
    the spec has no weight of a shape we can read."""
    want = {"conv2d": (4, 6), "dense": (2, 3)}.get(op, ())
    w = next((a for a in getattr(spec, "args", []) if a.kind == "param" and len(a.dims) in want), None)
    if w is None:
        return None
    d = [x if isinstance(x, int) else None for x in w.dims]
    mul = lambda a, b: a * b if a is not None and b is not None else None   # noqa: E731
    if op == "conv2d":
        return tuple(d) if len(d) == 4 else (mul(d[0], d[5]), mul(d[1], d[4]), d[2], d[3])
    return tuple(d) if len(d) == 2 else (mul(d[0], d[2]), d[1])


def weight_dtype(spec, op: str):
    want = {"conv2d": (4, 6), "dense": (2, 3)}.get(op, ())
    w = next((a for a in getattr(spec, "args", []) if a.kind == "param" and len(a.dims) in want), None)
    return w.dtype if w is not None else None


def _pick_spec(name: str, specs: list, source: Path):
    named = [s for s in specs if s.name == name]
    if named:
        return named[0], []
    if len(specs) == 1:
        return specs[0], [f"AIEK_SPEC is named {specs[0].name!r}; registered as {name!r}"]
    raise AiegraphError("register", f"{source.name} has {len(specs)} AIEK_SPECs, none named "
                        f"{name!r}", hint="name= one of " + ", ".join(s.name for s in specs))


def _spec_conflicts(op: str, constraints: dict, spec) -> list:
    """What the source's own AIEK_SPEC says against the contract."""
    out = []
    family = " ".join([spec.op or ""] + list(spec.match))
    if op not in family:
        out.append(f"op={op!r} but the AIEK_SPEC says op {spec.op!r}, match "
                   f"{', '.join(spec.match) or '-'}")
    weights = [a for a in spec.args if a.kind == "param" and len(a.dims) >= 2]
    want = constraints.get("dtype")
    if want and weights and weights[0].dtype != want:
        out.append(f"dtype={want!r} but the AIEK_SPEC weight {weights[0].name or 'param'} "
                   f"is {weights[0].dtype}")
    return out


def make_contract(name, source, op, target="aie", constraints=None) -> KernelContract:
    if op not in OPS:
        raise AiegraphError("register", f"op={op!r}", hint=f"one of {', '.join(OPS)}")
    if target not in KERNEL_TARGETS:
        raise AiegraphError("register", f"target={target!r}",
                            hint="'aie' (where it runs) or 'a78' (the ELF that launches it)")
    constraints = dict(constraints or {})
    for key, val in constraints.items():
        if key not in CONSTRAINTS:
            raise AiegraphError("register", f"unknown constraint {key!r}",
                                hint=f"one of {', '.join(CONSTRAINTS)}")
        if val not in CONSTRAINTS[key][0]:
            raise AiegraphError("register", f"{key}={val!r}",
                                hint=f"one of {', '.join(CONSTRAINTS[key][0])}")
    src = Path(source)
    if not src.is_file():
        raise AiegraphError("register", f"{source} does not exist",
                            hint="paths are relative to the current directory")
    rep = _env.aiekernel_pure("lint").lint_source(src)
    if not rep.ok:
        raise AiegraphError("register", f"{src.name}: lint: {rep.errors[0]}",
                            hint=f"ag.lint('{source}') lists every finding")
    if not rep.specs:
        raise AiegraphError("register", f"{src.name} has no AIEK_SPEC",
                            hint="doc/tutorial_userkernel.md §3 (AIEK_SPEC)")
    spec, notes = _pick_spec(name, rep.specs, src)
    bad = _spec_conflicts(op, constraints, spec)
    if bad:
        raise AiegraphError("register", f"contract contradicts {src.name}: {bad[0]}",
                            hint="fix the contract or the AIEK_SPEC; they must describe one kernel")
    if not rep.has_aie_kernel:
        notes.append("reference-only source: checks run, offload cannot build it")
    return KernelContract(name, src.resolve(), op, target, constraints, spec, rep, notes)


def check_region(contract: KernelContract, region) -> list:
    """Every way *region* breaks *contract*; empty means it fits the contract."""
    out = []
    if region.op != contract.op:
        out.append(f"op is {region.op}, contract says {contract.op}")
    for key, want in contract.constraints.items():
        have = getattr(region, CONSTRAINTS[key][1])
        if have != want:
            out.append(f"{key} is {have}, contract says {want}")
    if region.layer is None:
        out.append("no TVM call carries this region's weights")
    return out
