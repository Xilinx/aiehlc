###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``ag.kernel_lib(root)``: every ``AIEK_SPEC`` under a directory, as a dispatch list."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import _env
from .contract import KernelContract, logical_weight, make_contract, op_family, weight_dtype
from .errors import AiegraphError

__all__ = ["KernelLib", "LibEntry", "Dispatch", "kernel_lib"]

_EXTS = (".cc", ".cpp", ".cxx", ".c")
_DIMS = {"conv2d": ("O", "I", "KH", "KW"), "dense": ("O", "I")}


@dataclass
class LibEntry:
    contract: KernelContract
    origin: str = "library"         # "library" (found under root) | "added" (lib.add)
    verified: bool = True           # False: added with needs_numerics_test, not yet bit-exact
    evidence: str = ""              # what verified it

    @property
    def name(self) -> str:
        return self.contract.name

    @property
    def weight(self) -> tuple:
        return logical_weight(self.contract.spec, self.contract.op)

    @property
    def specificity(self) -> int:
        return sum(d is not None for d in (self.weight or ()))

    @property
    def runnable(self) -> bool:
        return self.contract.lint.has_aie_kernel

    def accepts(self, region) -> str:
        """'' if *region* fits this kernel's spec, else the reason it does not."""
        k = self.contract
        if region.op != k.op:
            return f"op {region.op}, kernel is {k.op}"
        want = weight_dtype(k.spec, k.op)
        if want and region.weight_dtype != want:
            return f"weight {region.weight_dtype}, spec says {want}"
        if self.weight is None:
            return f"AIEK_SPEC {k.spec.name} has no {k.op} weight argument"
        have = _region_weight(region)
        for dim, w, h in zip(_DIMS[k.op], self.weight, have):
            if w is not None and h is not None and w != h:
                return f"{dim} {h}, spec says {w}"
        return ""

    def geometry(self) -> str:
        if self.weight is None:
            return "?"
        return " ".join(f"{d}={'*' if v is None else v}" for d, v in zip(_DIMS[self.contract.op],
                                                                         self.weight))


def _region_weight(region) -> tuple:
    """Region weight as ``(O, I, KH, KW)`` / ``(O, I)`` -- ONNX OIHW, OI or IO."""
    s = tuple(region.weight_shape)
    if region.op == "dense" and region.weight_layout == "IO" and len(s) == 2:
        return (s[1], s[0])
    return s


@dataclass
class Dispatch:
    """One ``match`` call: the winner (or None) and every entry's verdict."""

    regions: list
    winner: LibEntry = None
    reason: str = ""                                   # why there is no winner
    rows: list = field(default_factory=list)           # (entry, verdict text)

    @property
    def kernel(self):
        return self.winner.contract if self.winner else None

    def __repr__(self) -> str:
        names = ", ".join(r.name for r in self.regions)
        head = (f"match {names} -> {self.winner.name}" if self.winner
                else f"match {names} -> none ({self.reason})")
        lines = [head] + [f"  {e.name:<20} {e.geometry():<24} {v}" for e, v in self.rows]
        return "\n".join(lines)


class KernelLib:
    """Kernels by name, each a :class:`KernelContract` from its own ``AIEK_SPEC``."""

    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else None
        self.entries: dict = {}             # name -> LibEntry
        self.skipped: list = []             # (path, reason)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries.values())

    def __contains__(self, name) -> bool:
        return name in self.entries

    def __getitem__(self, name) -> LibEntry:
        return self.entries[name]

    def _put(self, contract: KernelContract, origin: str, verified: bool) -> LibEntry:
        old = self.entries.get(contract.name)
        if old is not None and old.contract.source != contract.source:
            raise AiegraphError("kernel_lib", f"two kernels named {contract.name!r}: "
                                f"{old.contract.source} and {contract.source}",
                                hint="rename one AIEK_SPEC (its `kernel` line)")
        e = LibEntry(contract, origin, verified)
        self.entries[contract.name] = e
        return e

    def add(self, name: str, source, needs_numerics_test: bool = True, op: str = None,
            constraints: dict = None) -> LibEntry:
        """Add a ported kernel (``offload_AI_proto``'s ``.source``)."""
        spec = _first_spec(Path(source))
        op = op or op_family(spec)
        if not op:
            raise AiegraphError("kernel_lib", f"{Path(source).name}: cannot tell the op family "
                                f"from AIEK_SPEC op {spec.op!r}", hint="pass op='conv2d' or 'dense'")
        k = make_contract(name, source, op, "a78", constraints)
        e = self._put(k, "added", verified=not needs_numerics_test)
        state = "awaiting numerics (calibrate)" if not e.verified else "dispatchable"
        print(f"[aiegraph] kernel_lib: added {name} ({k.source.name}), {state}")
        return e

    def record_numerics(self, name: str, bit_exact: bool, evidence: str = "") -> None:
        """A calibrate result for *name*: BIT-EXACT makes an added kernel dispatchable."""
        e = self.entries.get(name)
        if e is None or e.verified or not bit_exact:
            return
        e.verified, e.evidence = True, evidence or "calibrate BIT-EXACT"
        print(f"[aiegraph] kernel_lib: {name} passed numerics ({e.evidence}); now dispatchable")

    def _verdict(self, e: LibEntry, regions: list, reference_only: bool) -> str:
        if not e.verified:
            return "awaiting numerics (lib.add needs_numerics_test)"
        if not e.runnable and not reference_only:
            return "reference-only source (no AIE kernel body)"
        for r in regions:
            why = e.accepts(r)
            if why:
                return f"{r.name}: {why}"
        return ""

    def match(self, regions, reference_only: bool = False) -> Dispatch:
        """The most specific kernel that accepts every region, or None."""
        regions = list(regions)
        d = Dispatch(regions)
        if not regions:
            d.reason = "no regions"
            return d
        shapes = {(r.op, _region_weight(r)[2] if r.op == "conv2d" else None) for r in regions}
        if len(shapes) > 1:
            d.reason = ("regions disagree on (op, KH): "
                        + ", ".join(f"{o} {k}x{k}" if k else o for o, k in sorted(shapes, key=str))
                        + " -- one kernel cannot serve them")
            d.rows = [(e, "not tried") for e in self]
            return d
        ok = []
        for e in self:
            why = self._verdict(e, regions, reference_only)
            d.rows.append((e, f"no -- {why}" if why else f"accepts (specificity {e.specificity})"))
            if not why:
                ok.append(e)
        if not ok:
            d.reason = "no kernel accepts them"
            return d
        best = max(e.specificity for e in ok)
        top = [e for e in ok if e.specificity == best]
        if len(top) > 1:
            d.reason = f"ambiguous: {', '.join(e.name for e in top)} are equally specific"
            return d
        d.winner = top[0]
        return d

    def __repr__(self) -> str:
        where = f" under {self.root}" if self.root else ""
        lines = [f"kernel library: {len(self)} kernel(s){where}"]
        for e in self:
            k = e.contract
            flags = [] if e.runnable else ["reference-only"]
            if not e.verified:
                flags.append("awaiting numerics")
            lines.append(f"  {e.name:<20} {k.op:<7} {e.geometry():<24} "
                         f"{k.source.name}{'  [' + ', '.join(flags) + ']' if flags else ''}")
        for path, why in self.skipped:
            lines.append(f"  skipped {path}: {why}")
        return "\n".join(lines)


def _first_spec(path: Path):
    spec_mod = _env.aiekernel_pure("spec")
    specs = spec_mod.parse_specs(spec_mod.load_source(path))
    if not specs:
        raise AiegraphError("kernel_lib", f"{path.name} has no AIEK_SPEC")
    return specs[0]


def _candidates(root: Path) -> list:
    return sorted(p for p in root.rglob("*") if p.suffix in _EXTS and p.is_file()
                  and "AIEK_SPEC(" in p.read_text(errors="replace"))


def kernel_lib(root, quiet: bool = False) -> KernelLib:
    """Index every ``AIEK_SPEC`` under *root* (``src/aietensorop``)."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise AiegraphError("kernel_lib", f"{root} is not a directory")
    spec_mod = _env.aiekernel_pure("spec")
    lib = KernelLib(root)
    for path in _candidates(root):
        rel = path.relative_to(root)
        try:
            specs = spec_mod.parse_specs(spec_mod.load_source(path))
        except spec_mod.SpecError as exc:
            lib.skipped.append((rel, f"SpecError: {exc}"))
            continue
        for spec in specs:
            if Path(spec.spec_file).resolve() != path:
                continue                    # indexed from the file that holds it
            op = op_family(spec)
            if not op:
                lib.skipped.append((rel, f"AIEK_SPEC {spec.name}: op {spec.op!r} is neither "
                                         "conv2d nor dense"))
                continue
            try:
                k = make_contract(spec.name, path, op, "a78", _spec_constraints(spec, op))
            except AiegraphError as exc:
                lib.skipped.append((rel, str(exc).splitlines()[0]))
                continue
            lib._put(k, "library", verified=True)     # a duplicate name raises
    if not quiet:
        for path, why in lib.skipped:
            print(f"[aiegraph] kernel_lib: skipped {path}: {why}")
    return lib


def _spec_constraints(spec, op: str) -> dict:
    """Only what the spec states: its weight dtype."""
    dt = weight_dtype(spec, op)
    return {"dtype": dt} if dt in ("int8", "uint8") else {}
