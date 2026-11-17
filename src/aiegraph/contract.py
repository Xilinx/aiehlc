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

__all__ = ["KernelContract", "make_contract", "check_region", "CONSTRAINTS",
           "Buf", "Step", "buf", "step", "BoundStep", "check_step", "check_steps", "load_steps"]

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
    _env.frontend("")
    from frontend.tvmrelay.aiekernel.lint import lint_source
    rep = lint_source(src)
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



# ═══════════════════════════════════════════════════════════════════════════
#  Group-offload steps: ag.step(...) checked against the source's AIEK_STEP
#  (design docs/plans/2026-11-16-aiegraph-group-offload-design.md §5)
# ═══════════════════════════════════════════════════════════════════════════

_DTYPE_ALIASES = {"fp32": "float32", "f32": "float32", "float": "float32", "fp16": "float16",
                  "i8": "int8", "u8": "uint8", "i32": "int32"}


@dataclass(frozen=True)
class Buf:
    """A boundary tensor as a step sees it (C: ``aiek_buf``).

    ``shape`` is the logical extent; ``ld`` the elements between consecutive
    rows (index ndim-2). ``ld == shape[-1]`` is dense; a head's column block of
    a ``[N][H*D]`` linear output is ``shape=(N, D), ld=H*D`` -- strided."""

    shape: tuple
    dtype: str = "float32"
    ld: int = None
    name: str = ""
    tensor: object = None          # graph tensor id (tensors[i]) once groups exist

    def __post_init__(self):
        shape = tuple(int(d) for d in self.shape)
        if not shape or any(d <= 0 for d in shape):
            raise AiegraphError("step", f"buffer {self.name or '?'}: shape {self.shape}",
                                hint="positive extents, at least one dim")
        object.__setattr__(self, "shape", shape)
        dt = _DTYPE_ALIASES.get(self.dtype, self.dtype)
        object.__setattr__(self, "dtype", dt)
        ld = shape[-1] if self.ld is None else int(self.ld)
        if ld < shape[-1]:
            raise AiegraphError("step", f"buffer {self.name or '?'}: ld={ld} < last extent "
                                f"{shape[-1]} (rows would overlap)")
        object.__setattr__(self, "ld", ld)

    @property
    def dense(self) -> bool:
        return len(self.shape) < 2 or self.ld == self.shape[-1]

    def __str__(self) -> str:
        return (f"{self.name or 'buf'} {self.dtype}{list(self.shape)}"
                + ("" if self.dense else f" ld={self.ld}"))


def buf(shape, dtype="float32", ld=None, name="", tensor=None) -> Buf:
    """``ag.buf((1024, 64), ld=320)`` -- a (possibly strided) step buffer."""
    return Buf(tuple(shape), dtype, ld, name, tensor)


def _as_buf(x, who: str) -> Buf:
    if isinstance(x, Buf):
        return x
    if all(hasattr(x, a) for a in ("shape", "dtype")):     # a group boundary tensor / ndarray
        return Buf(tuple(x.shape), str(x.dtype), getattr(x, "ld", None),
                   getattr(x, "name", "") or "", getattr(x, "tensor", None))
    raise AiegraphError("step", f"{who}: {x!r} is not a buffer",
                        hint="ag.buf(shape, dtype, ld=) or a group's boundary tensor")


def _as_list(x) -> tuple:
    if x is None:
        return ()
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


@dataclass(frozen=True)
class Step:
    """``ag.step(name, reads=, writes=, consts=, weights=)`` -- what the user
    says one step of a group does. Checked against the source's AIEK_STEP."""

    name: str
    reads: tuple = ()
    writes: tuple = ()
    consts: dict = field(default_factory=dict)
    weights: object = None

    def __repr__(self) -> str:
        c = ", ".join(f"{k}={v!r}" for k, v in self.consts.items())
        return (f"step {self.name}: reads [{', '.join(map(str, self.reads))}] "
                f"writes [{', '.join(map(str, self.writes))}] consts {{{c}}}")


def step(name: str, reads=None, writes=None, consts=None, weights=None) -> Step:
    if not isinstance(name, str) or not name.isidentifier():
        raise AiegraphError("step", f"step name {name!r}", hint="the C entry name of the step")
    r = tuple(_as_buf(x, f"{name} reads") for x in _as_list(reads))
    w = tuple(_as_buf(x, f"{name} writes") for x in _as_list(writes))
    c = dict(consts or {})
    for k, v in c.items():
        if not isinstance(k, str) or not k.isidentifier():
            raise AiegraphError("step", f"{name}: const name {k!r}")
        if isinstance(v, bool) or not isinstance(v, (int, float)) and not hasattr(v, "__index__") \
                and not hasattr(v, "__float__"):
            raise AiegraphError("step", f"{name}: const {k}={v!r} is not a number")
    return Step(name, r, w, c, weights)


@dataclass
class BoundStep:
    """A Step that matched its AIEK_STEP: constants in the C order."""

    step: Step
    spec: object                    # aiekernel StepSpec
    const_values: tuple             # (int | float, ...) in consts() order

    @property
    def name(self) -> str:
        return self.step.name


def _check_bufs(sname: str, what: str, have: tuple, want: list) -> list:
    out = []
    if len(have) != len(want):
        return [f"{sname}: {what} {len(have)} buffer(s) but AIEK_STEP({sname}) declares "
                f"{what}({', '.join(map(str, want)) or 0}) -- {len(want)}"]
    for b, s in zip(have, want):
        tag = f"{sname}: {what}[{s.index}]" + (f" ({s.name})" if s.name else "")
        if s.dtype and b.dtype != s.dtype:
            out.append(f"{tag} is {b.dtype}, AIEK_STEP wants {s.dtype}")
        if s.dense and not b.dense:
            out.append(f"{tag} is strided ({b}: ld={b.ld} != last extent {b.shape[-1]}) but "
                       f"AIEK_STEP marks it dense -- the step would ignore ld and read the wrong rows")
    return out


def check_step(spec, st: Step) -> list:
    """Every way *st* disagrees with *spec* (a StepSpec); empty = consistent."""
    if spec.entry != st.name:
        return [f"step {st.name} checked against AIEK_STEP({spec.entry})"]
    out = _check_bufs(st.name, "reads", st.reads, spec.reads)
    out += _check_bufs(st.name, "writes", st.writes, spec.writes)
    have, want = list(st.consts), list(spec.const_names)
    if set(have) != set(want):
        miss = [n for n in want if n not in have]
        extra = [n for n in have if n not in want]
        out.append(f"{st.name}: consts {have} but AIEK_STEP declares consts({', '.join(want)})"
                   + (f"; missing {miss}" if miss else "") + (f"; unknown {extra}" if extra else ""))
    else:
        for c in spec.consts:
            v = st.consts[c.name]
            if c.type == "int" and not (isinstance(v, int) or hasattr(v, "__index__")):
                if not (isinstance(v, float) and v.is_integer()):
                    out.append(f"{st.name}: const {c.name}={v!r} but AIEK_STEP declares it :int")
    return out


def _const_values(spec, st: Step) -> tuple:
    vals = []
    for c in spec.consts:
        v = st.consts[c.name]
        vals.append(int(v) if c.type == "int" else float(v))
    return tuple(vals)


def load_steps(source) -> dict:
    """``{entry: StepSpec}`` for every AIEK_STEP in *source* (path, spliced
    like aiehlc reads it)."""
    src = Path(source)
    if not src.is_file():
        raise AiegraphError("step", f"{source} does not exist")
    _env.frontend("")
    from frontend.tvmrelay.aiekernel.spec import SpecError, load_source, parse_steps
    try:
        specs = parse_steps(load_source(src))
    except SpecError as exc:
        raise AiegraphError("step", f"{src.name}: {exc}",
                            hint="include/aie_user_kernel.h documents the AIEK_STEP grammar") from None
    return {s.entry: s for s in specs}


def check_steps(source, steps) -> list:
    """Check every ``ag.step`` in *steps* against the AIEK_STEPs in *source*
    (a path or a ``{entry: StepSpec}`` from :func:`load_steps`). Returns the
    :class:`BoundStep` list; raises :class:`AiegraphError` listing every
    mismatch -- arity, const names / types, dtype, a strided buffer handed to
    a dense step."""
    specs = source if isinstance(source, dict) else load_steps(source)
    bound, bad = [], []
    for st in _as_list(steps):
        if not isinstance(st, Step):
            raise AiegraphError("step", f"{st!r} is not an ag.step(...)")
        spec = specs.get(st.name)
        if spec is None:
            bad.append(f"{st.name}: no AIEK_STEP({st.name}) in the source "
                       f"(has: {', '.join(sorted(specs)) or 'none'})")
            continue
        problems = check_step(spec, st)
        if problems:
            bad += problems
        else:
            bound.append(BoundStep(st, spec, _const_values(spec, st)))
    if bad:
        raise AiegraphError("step", "; ".join(bad),
                            hint="ag.step(...) and the source's AIEK_STEP must describe one call")
    return bound
