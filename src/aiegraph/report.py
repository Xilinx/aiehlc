###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Report objects: they print as tables, answer ``.ok``, and can raise.

They wrap the ``frontend.tvmrelay.aiekernel`` report dict (``flow._report``) so
the user never reads ``report.json`` or ``partition.json`` by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .errors import AiegraphError

__all__ = ["CheckReport", "DetectReport", "BuildReport", "RegionCheck", "PartitionPlan"]


def _gates_line(k: dict, layer: str) -> str:
    calib = k["layers"].get(layer, {}).get("calibration", [])
    exact = [c for c in calib if "BIT-EXACT" in c]
    parts = ["lint ok", f"detect {len(k['layers'].get(layer, {}).get('calls', []))} call(s)"]
    if calib:
        parts.append(f"calibrate {'BIT-EXACT' if len(exact) == len(calib) else 'MISMATCH'}"
                     f" ({len(calib)} call(s))")
    return " | ".join(parts)


@dataclass
class CheckReport:
    """Result of ``offload_check`` (gates lint..emulate) or ``offload`` (..build, link)."""

    kind: str                       # "offload_check" | "offload"
    raw: dict                       # aiekernel flow report
    requested: dict                 # {source Path: [layers] | None}
    rows: list = field(default_factory=list)   # (source, layer, ok, text)

    def __post_init__(self):
        by_src = {Path(k["source"]).resolve(): k for k in self.raw["kernels"]}
        for src, wanted in self.requested.items():
            k = by_src.get(Path(src).resolve())
            if k is None:
                self.rows.append((src, None, False, "not processed"))
                continue
            if not k["lint_ok"]:
                errs = [f for f in k["lint"] if f.startswith("ERROR")]
                self.rows.append((src, None, False, "lint: " + (errs[0] if errs else "failed")))
                continue
            layers = wanted if wanted is not None else sorted(
                set(k["selected"]) | {int(x) for x in k["rejected"]})
            if wanted is None and not layers:
                self.rows.append((src, None, False, "detect: the spec fits no layer "
                                                    "(try emod.detect(src))"))
            for layer in layers:
                if layer in k["selected"]:
                    self.rows.append((src, layer, True, _gates_line(k, str(layer))))
                else:
                    r = k["rejected"].get(str(layer), {"gate": "?", "reason": "not selected"})
                    self.rows.append((src, layer, False, f"stopped at {r['gate']}: {r['reason']}"))

    @property
    def ok(self) -> bool:
        return bool(self.rows) and all(r[2] for r in self.rows)

    @property
    def ready_layers(self) -> list:
        return sorted({r[1] for r in self.rows if r[2] and r[1] is not None})

    @property
    def emulation(self) -> str:
        e = self.raw.get("emulation")
        return e["summary"].removeprefix("emulate: ") if e else "not run"

    def raise_for_status(self) -> "CheckReport":
        if not self.ok:
            bad = next(r for r in self.rows if not r[2])
            layer = f"layer{bad[1]}" if bad[1] is not None else Path(bad[0]).name
            raise AiegraphError(self.kind, f"{layer}: {bad[3]}",
                                hint="emod.detect(<kernel>) shows what fits; "
                                     "doc/tutorial_userkernel.md §8 lists the fixes")
        return self

    def __repr__(self) -> str:
        lines = [f"{self.kind}  {'READY' if self.ok else 'NOT READY'}"]
        for src, layer, ok, text in self.rows:
            name = f"layer{layer}" if layer is not None else "-"
            lines.append(f"  {name:<8} {'ok ' if ok else 'NO '} {Path(src).name}: {text}")
        lines.append(f"  emulate: {self.emulation}")
        for k in self.raw["kernels"]:
            b = k.get("build")
            if b:
                lines.append(f"  build {k['name']}: " + (
                    f"{Path(b['archive']).name}" if b.get("ok") else f"FAILED -- {b.get('reason')}"))
        arm = self.raw.get("arm")
        if arm:
            lines.append("  main.elf: " + (arm.get("elf") or (
                f"not linked ({arm['reason']})" if arm.get("reason")
                else "driver regenerated with the swap; link it with emod.build()")))
        if self.ok and self.kind == "offload_check":
            lines.append("  next: emod.offload(" + ", ".join(
                f"('layer{l}', '{Path(s).name}')" for s, l, ok, _ in self.rows if l is not None)
                + ")")
        elif self.ok:
            lines.append("  next: emod.build()   (links main.elf; never runs it)")
        lines.append(f"  details: {self.raw.get('report')}")
        return "\n".join(lines)


@dataclass
class DetectReport:
    source: Path
    verdicts: list                  # aiekernel LayerVerdict

    @property
    def matches(self) -> list:
        return [v.layer for v in self.verdicts if v.ok]

    def __repr__(self) -> str:
        lines = [f"detect {self.source.name}: fits {len(self.matches)} layer(s): "
                 + (", ".join(f"layer{l}" for l in self.matches) or "none")]
        for v in self.verdicts:
            if v.ok or any(s in v.symbol for s in ("conv", "dense")):
                calls = len(v.calls)
                lines.append(f"  layer{v.layer:<3} {'MATCH' if v.ok else '-----'} "
                             f"({calls} call{'s' if calls > 1 else ''}) {v.reason}")
        return "\n".join(lines)


@dataclass
class BuildReport:
    ok: bool
    elf: str = None
    reason: str = ""
    local: dict = None
    output: str = None              # build(output=...): where the artifacts were copied
    files: list = field(default_factory=list)

    def __repr__(self) -> str:
        if self.ok:
            s = f"build: {self.elf}"
        else:
            s = f"build: not linked -- {self.reason}"
        if self.local:
            top = self.local.get("top1")
            s += (f"\n  x86 main_local.elf: {self.local.get('elf')}"
                  + (f"  top1 class={top[0]} logit={top[1]:.6f}" if top and top[0] is not None else ""))
        if self.output:
            s += f"\n  output: {self.output}/  ({', '.join(Path(f).name for f in self.files)})"
        return s


# ── region API (module.check_offload / module.partition) ──────────────────

@dataclass
class RegionCheck:
    """``module.check_offload(region=, kernel=)``: contract per region, then the
    aiekernel gates (lint, detect, calibrate, emulate) per TVM layer."""

    kernel: str
    regions: list                   # [Region]
    contract: dict                  # region name -> [violations]
    gates: CheckReport = None       # None when no region passed its contract

    def _gate(self, layer) -> tuple:
        if self.gates is None:
            return False, "not run (contract)"
        row = next((r for r in self.gates.rows if r[1] == layer), None)
        return (row[2], row[3]) if row else (False, "not checked")

    @property
    def ok(self) -> bool:
        return bool(self.regions) and all(
            not self.contract[r.name] and self._gate(r.layer)[0] for r in self.regions)

    def raise_for_status(self) -> "RegionCheck":
        if not self.ok:
            raise AiegraphError("check_offload", self.summary().splitlines()[0],
                                hint="print(check.summary()) has the reason per region")
        return self

    def summary(self) -> str:
        names = ", ".join(r.label for r in self.regions)
        lines = [f"check_offload {self.kernel} -> {names}   {'READY' if self.ok else 'NOT READY'}"]
        for r in self.regions:
            bad = self.contract[r.name]
            lines.append(f"  {r.label}  (onnx {r.node}; tvm {r.where})")
            lines.append(f"    contract: {'ok' if not bad else 'NO -- ' + '; '.join(bad)}")
            if not bad:
                ok, text = self._gate(r.layer)
                lines.append(f"    gates:    {'ok -- ' if ok else 'NO -- '}{text}")
            if r.siblings:
                lines.append(f"    note: tvm layer{r.layer} also runs {', '.join(r.siblings)} "
                             f"-- the driver swaps the kernel, so they move together")
        if self.gates is not None:
            lines.append(f"  emulate: {self.gates.emulation}")
        if self.ok:
            lines.append(f"  next: module.assign(region=..., backend='aie:{self.kernel}')")
        return "\n".join(lines)

    __repr__ = summary


@dataclass
class PartitionPlan:
    """``module.partition()``: where every assigned region ended up, and why."""

    rows: list                      # (region, requested, placed, reason)
    fallback: str
    optimize_for: str
    offload: CheckReport = None
    dragged: list = field(default_factory=list)   # (region, backend, because-of)

    @property
    def ok(self) -> bool:
        return all(req == placed for _, req, placed, _ in self.rows)

    @property
    def on_aie(self) -> list:
        return [r.label for r, _, placed, _ in self.rows if placed != "cpu"] + \
               [r.label for r, _, _ in self.dragged]

    def report(self) -> str:
        lines = [f"partition  fallback={self.fallback}  optimize_for={self.optimize_for}  "
                 f"{'as assigned' if self.ok else 'FELL BACK'}"]
        w = max([len(r[0].label) for r in self.rows + self.dragged] + [6])
        lines.append(f"  {'region':<{w}}  {'assigned':<16} {'placed':<16} tvm")
        for r, req, placed, why in self.rows:
            lines.append(f"  {r.label:<{w}}  {req:<16} {placed:<16} {r.where}")
            if why:
                lines.append(f"  {'':<{w}}    {why}")
        for r, backend, cause in self.dragged:
            lines.append(f"  {r.label:<{w}}  {'(unassigned)':<16} {backend:<16} {r.where}"
                         f"  -- shares its TVM kernel with {cause}")
        lines.append(f"  on the AIE: {', '.join(self.on_aie) or 'nothing'}; every other "
                     f"layer runs on the A78")
        if self.optimize_for:
            lines.append(f"  optimize_for={self.optimize_for}: no cost model yet -- placement "
                         f"is the assignments that passed every gate")
        lines.append("  next: module.build(target='a78', output=...)")
        return "\n".join(lines)

    __repr__ = report
