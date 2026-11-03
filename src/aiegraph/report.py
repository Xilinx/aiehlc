###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Report objects: they print as tables, answer ``.ok``, and can raise.

They wrap the ``frontend.tvmrelay.aiekernel`` report dict (``flow._report``) so
the user never reads ``report.json`` or ``partition.json`` by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AiegraphError

__all__ = ["CheckReport", "DetectReport", "BuildReport", "RunReport", "ProfileReport",
           "RegionCheck", "PartitionPlan", "AIProtoResult", "PerfReport", "KernelOptResult"]


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


@dataclass
class RunReport:
    """``emod.run(board, host)``: what ``main.elf`` printed on the board."""

    ok: bool
    board: str
    host: str
    elf: str
    rc: int = None
    log: str = None
    reason: str = ""
    top1: tuple = None              # (class, logit)
    top5: list = field(default_factory=list)   # [(class, label, logit)]
    inference_ms: float = None
    aie_enter: int = 0              # "[aie-offload] ... ENTER" lines (AIE layers entered)
    aie_exit: int = 0
    seconds: float = 0.0            # wall clock of the whole board session
    command: list = field(default_factory=list)
    profile: list = field(default_factory=list)  # [(call, ms, "cpu"|"aie", op)] -- PROFILE=1 ELF

    def raise_for_status(self) -> "RunReport":
        if not self.ok:
            raise AiegraphError("run", self.reason, hint=f"console log: {self.log}")
        return self

    def __repr__(self) -> str:
        head = f"run on {self.board} @ {self.host}: {'OK' if self.ok else 'FAILED -- ' + self.reason}"
        lines = [head, f"  elf: {self.elf}"]
        if self.top1:
            lines.append(f"  top1 class={self.top1[0]} logit={self.top1[1]:.6f}")
        for i, (c, label, logit) in enumerate(self.top5, 1):
            lines.append(f"    {i}. class={c:<4} {label:<28} logit={logit:.4f}")
        if self.inference_ms is not None:
            lines.append(f"  inference {self.inference_ms:.3f} ms on the board")
        if self.aie_enter or self.aie_exit:
            lines.append(f"  AIE layers: ENTER x{self.aie_enter}, EXIT x{self.aie_exit}")
        if self.profile:
            lines.append(f"  per-call times: {len(self.profile)} calls -- emod.profile(ag.ALL)")
        lines.append(f"  log: {self.log}   ({self.seconds:.0f} s, rc={self.rc})")
        return "\n".join(lines)


@dataclass
class ProfileReport:
    """``emod.profile(ag.ALL)``: where the board spent ``graph_run()``, per call
    (``by="call"``) or per layer (``by="layer"``). Times are the mean per
    ``graph_run()``, measured on the board with the clock main.c times
    "inference" with, so the rows add up to it (minus the driver's glue)."""

    rows: list                      # [{"call"|"calls", "layer", "op", "where", "ms", "pct"}]
    by: str = "call"
    board: str = None
    inference_ms: float = None
    total_ms: float = 0.0           # sum over ALL calls (the % base), not just the shown rows
    log: str = None

    def ms(self, where: str) -> float:
        """Total ms of the shown rows on ``"cpu"`` or ``"aie"``."""
        return sum(r["ms"] for r in self.rows if r["where"] == where)

    def __repr__(self) -> str:
        head = "call" if self.by == "call" else "calls"
        w = max([len(r["op"]) for r in self.rows] + [8])
        lines = [f"profile ({self.by}) on {self.board or '?'}: mean ms per graph_run()",
                 f"  {head:<10} {'layer':<8} {'op':<{w}}  on    {'ms':>10}  {'%':>6}"]
        for r in self.rows:
            key = (f"[{r['call']:02d}]" if self.by == "call"
                   else ",".join(f"{c:02d}" for c in r["calls"]))
            key = key if len(key) <= 10 else key[:9] + "+"
            lines.append(f"  {key:<10} {r['layer']:<8} {r['op']:<{w}}  {r['where']:<4} "
                         f"{r['ms']:>10.3f}  {r['pct']:>5.1f}%")
        shown = sum(r["ms"] for r in self.rows)
        lines.append(f"  shown {shown:.3f} ms of {self.total_ms:.3f} ms (all calls); "
                     f"cpu {self.ms('cpu'):.3f} ms, aie {self.ms('aie'):.3f} ms")
        if self.inference_ms is not None:
            lines.append(f"  inference {self.inference_ms:.3f} ms (graph_run incl. driver glue)")
        if self.log:
            lines.append(f"  from {self.log}")
        return "\n".join(lines)


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


@dataclass
class AIProtoResult:
    """Result of ``offload_AI_proto``: the ported kernel (path-like) and its last check."""

    source: Path                    # the kernel to use: the port, or the original if it was READY
    layers: list
    report: CheckReport             # our own offload_check of ``source``
    rounds: int                     # claude rounds run (0: nothing to port)
    session: str = None             # claude session id (``claude --resume <id>`` to go on by hand)
    log: Path = None
    original: Path = None

    @property
    def ok(self) -> bool:
        return self.report.ok

    def __fspath__(self) -> str:
        return str(self.source)

    def __str__(self) -> str:
        return str(self.source)

    def raise_for_status(self) -> "AIProtoResult":
        self.report.raise_for_status()
        return self

    def __repr__(self) -> str:
        tag = ",".join(f"layer{i}" for i in self.layers)
        lines = [f"offload_AI_proto  {'READY' if self.ok else 'NOT READY'}  {tag}  "
                 f"after {self.rounds} round(s)",
                 f"  kernel: {self.source}"]
        if self.original and self.original != self.source:
            lines.append(f"  from:   {self.original}")
        lines += ["  " + l for l in repr(self.report).splitlines()]
        if self.log:
            lines.append(f"  log:    {self.log}" + (f"  (claude --resume {self.session})"
                                                   if self.session else ""))
        lines.append(f"  next: emod.offload(('{tag.split(',')[0]}', new_kernel))" if self.ok else
                     "  next: read the log, edit the kernel by hand, or run again with rounds=")
        return "\n".join(lines)


# ── AI performance analysis / optimization (aiperf.py) ─────────────────────

def _ms(v) -> str:
    return "?" if v is None else f"{v:.3f} ms"


@dataclass
class PerfReport:
    """``emod.AI_perf_analysis("layerN")``: facts + cost model + rule findings,
    and Claude's reading (``.ai``: summary / bottlenecks / suggestions /
    predicted_ms_after, in skill kernelperfcheck's format)."""

    layer: str
    facts: dict
    model: dict
    rules: list
    ai: dict = None
    text: str = ""                  # Claude's full answer
    dest: Path = None               # report.json / report.md live here
    log: Path = None
    session: str = None

    @property
    def measured_ms(self):
        return self.facts["run"]["measured_ms"]

    @property
    def suggestions(self) -> list:
        """Claude's suggestions, else the rules' fixes (same keys as far as they go)."""
        if self.ai and self.ai.get("suggestions"):
            return self.ai["suggestions"]
        return [{"id": r["id"], "change": r["fix"], "where": r["evidence"],
                 "expected_saving_ms": r["est_ms"]} for r in self.rules]

    @property
    def predicted_ms_after(self):
        return (self.ai or {}).get("predicted_ms_after")

    def to_dict(self) -> dict:
        return {"layer": self.layer, "facts": self.facts, "model": self.model, "rules": self.rules,
                "ai": self.ai, "session": self.session, "log": str(self.log) if self.log else None}

    @classmethod
    def load(cls, path) -> "PerfReport":
        """A saved report back (``report.json``, or its ``aiperf/<layer>/`` dir) --
        e.g. in a new Python session, to pass to ``AI_kernel_accelerate``."""
        p = Path(path)
        p = p / "report.json" if p.is_dir() else p
        if not p.is_file():
            raise AiegraphError("perf", f"no saved report at {p}", hint='emod.AI_perf_analysis("layer1")')
        d = json.loads(p.read_text())
        text = ""
        md = p.parent / "report.md"
        if md.is_file() and "\n\n## claude\n\n" in md.read_text():
            text = md.read_text().split("\n\n## claude\n\n", 1)[1]
        return cls(d["layer"], d["facts"], d["model"], d["rules"], d.get("ai"), text, p.parent,
                   Path(d["log"]) if d.get("log") else None, d.get("session"))

    def save(self) -> None:
        if self.dest:
            (self.dest / "report.json").write_text(json.dumps(self.to_dict(), indent=1, default=str) + "\n")
            (self.dest / "report.md").write_text(repr(self) + ("\n\n## claude\n\n" + self.text if self.text else "") + "\n")

    def __repr__(self) -> str:
        f, m = self.facts, self.model
        run = f["run"]
        lines = [f"AI_perf_analysis {self.layer} ({f['op']})  kernel {Path(f['source']).name}",
                 f"  measured {_ms(run['measured_ms'])} per call"
                 + ("" if run["profiled"] else "  (no profiled run: emod.run(profile=True))")
                 + (f", EXIT rc={run['exit_rc']}" if run["exit_rc"] is not None else ""),
                 f"  model: kernel load {_ms(m['elf_load_ms'])} ({m['cores']} cores x "
                 f"{m['elf_words_per_core']} words), config {_ms(m['config_ms'])} "
                 f"(~{m['config_words']} words), teardown each call: {'yes' if m['teardown'] else 'no'}, "
                 f"computing: {'yes' if m['computing'] else 'NO (backbone)'}",
                 f"  explained {_ms(m['known_ms'])}; unexplained {_ms(m['unexplained_ms'])}"]
        if self.ai:
            if self.ai.get("summary"):
                lines.append(f"  claude: {self.ai['summary']}")
            for b in self.ai.get("bottlenecks", []):
                lines.append(f"    - {b.get('what')}  [{_ms(b.get('est_ms'))}]  {b.get('evidence', '')}")
        else:
            for r in self.rules:
                lines.append(f"    {r['id']} {r['what']}  [{_ms(r['est_ms'])}]  {r['evidence']}")
        lines.append("  suggestions:")
        for s in self.suggestions:
            lines.append(f"    {s.get('id', '-')}: {s.get('change')}  -> saves {_ms(s.get('expected_saving_ms'))}"
                         + (f"  (risk {s['risk']}, effort {s['effort']})" if s.get("risk") else ""))
        if self.predicted_ms_after is not None:
            lines.append(f"  predicted after: {_ms(self.predicted_ms_after)} (claude)")
        if self.dest:
            lines.append(f"  report: {self.dest / 'report.md'}")
        lines.append(f"  next: emod.AI_kernel_accelerate(\"{self.layer}\", report)")
        return "\n".join(lines)


@dataclass
class KernelOptResult:
    """``emod.AI_kernel_accelerate("layerN", report)``: the optimized copy (path-like),
    our gates' verdict, and the improvement -- predicted from the re-read facts
    of the NEW build, measured when ``board=`` was given."""

    source: Path
    original: Path
    layer: str
    rounds: int
    ok: bool                        # offload_check READY and aiehlc build OK
    reason: str
    changes: str                    # Claude's summary of what it changed
    report: PerfReport
    after: dict = None              # facts of the new build
    model_after: dict = None
    measured_after: float = None
    session: str = None
    log: Path = None

    def __fspath__(self) -> str:
        return str(self.source)

    @property
    def saved_ms(self):
        """Model-predicted saving: the priced components, before minus after."""
        if not self.model_after:
            return None
        return round(self.report.model["known_ms"] - self.model_after["known_ms"], 3)

    @property
    def predicted_ms(self):
        before = self.report.measured_ms
        if before is None or self.saved_ms is None:
            return None
        return round(before - self.saved_ms, 3)

    def raise_for_status(self) -> "KernelOptResult":
        if not self.ok:
            raise AiegraphError("perf", self.reason.splitlines()[0] if self.reason else "not ok",
                                hint=f"log: {self.log}")
        return self

    def __repr__(self) -> str:
        lines = [f"AI_kernel_accelerate {self.layer}  {'OK' if self.ok else 'FAILED'} after {self.rounds} round(s)",
                 f"  kernel: {self.source}", f"  from:   {self.original}"]
        if not self.ok:
            lines.append("  " + (self.reason.splitlines()[0] if self.reason else ""))
        if self.changes:
            lines += ["  changes (claude):"] + ["    " + l for l in self.changes.strip().splitlines()[:20]]
        if self.model_after:
            b, a = self.report.model, self.model_after
            lines.append(f"  {'':<16}{'before':>14}{'after':>14}")
            for k, name in (("elf_load_ms", "kernel load"), ("config_ms", "config"), ("known_ms", "priced total")):
                lines.append(f"  {name:<16}{_ms(b[k]):>14}{_ms(a[k]):>14}")
            lines.append(f"  {'teardown/call':<16}{str(b['teardown']):>14}{str(a['teardown']):>14}")
            lines.append(f"  {'computing':<16}{str(b['computing']):>14}{str(a['computing']):>14}")
            lines.append(f"  measured before {_ms(self.report.measured_ms)} -> predicted {_ms(self.predicted_ms)}"
                         f" (model saves {_ms(self.saved_ms)}; teardown/compute not priced)")
        if self.measured_after is not None:
            lines.append(f"  measured after  {_ms(self.measured_after)} on the board")
        if self.log:
            lines.append(f"  log: {self.log}" + (f"  (claude --resume {self.session})" if self.session else ""))
        if self.ok and self.measured_after is None:
            lines.append(f"  next: emod.offload((\"{self.layer}\", result)); emod.run(profile=True); "
                         f"emod.profile(\"{self.layer}\")")
        return "\n".join(lines)
