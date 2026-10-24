###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""CHECK and OFFLOAD: the gates, in order, and what passing them writes.

Gates (each layer must pass every one, in this order; a failure names its gate):

====  =========  =============================================================
1     lint       the source is safe to hand to aiehlc (``lint.py``)
2     detect     the layer's signature AND param values fit the spec
                 (``match.py``); all calls of the layer
3     calibrate  ``<entry>_ref`` == TVM's kernel on adversarial inputs, every
                 call (``calibrate.py``)
4     emulate    whole network on x86 with the reference swapped in through
                 the generated adapter: logits byte-identical (``emulate.py``)
5     build      aiehlc + hostcompile + isolate (``build.py``) -- offload only;
                 needs Vitis, never a board
6     link       one aiehlc library per ELF (the runtime's ``routing()``
                 singleton) and no symbol collisions among called archives
====  =========  =============================================================

``check`` runs 1-4 and writes nothing outside ``<out>/aiek/``. ``offload`` runs
all six, then records the decision in ``layers/partition.json`` (``target:
aie``, ``backend: aiehlc-user``, ``entry``/``replaces``/``archive``) and
regenerates ``arm_build/`` so ``graph_driver.c`` calls ``aie_<symbol>`` under
``-DGRAPH_AIE_OFFLOAD`` -- the same mechanism the stem uses
(``arm_build.aie_entries``).

A failed or impossible build (no Vitis) rejects the layer at gate 5 and leaves
``partition.json`` as it was. ``--no-build`` is the one way to record a
decision without an archive: the row gets ``archive_reason`` and the driver
does NOT swap it -- it only ever swaps what can link.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .graph import load_network
from .hostbuild import REPO, HostBuildError
from .lint import lint_source
from .match import detect_layers

__all__ = ["KernelPlan", "parse_kernel_arg", "plan", "check", "offload"]


@dataclass
class KernelPlan:
    source: Path
    lint: object                            # LintReport
    spec: object = None                     # KernelSpec (first AIEK_SPEC)
    requested: list = None                  # explicit layers, None = auto
    verdicts: dict = field(default_factory=dict)    # layer -> LayerVerdict
    calib: dict = field(default_factory=dict)       # layer -> [CalibResult]
    rejected: dict = field(default_factory=dict)    # layer -> (gate, reason)
    selected: list = field(default_factory=list)    # layers past every gate so far
    build: dict = None

    @property
    def name(self) -> str:
        return self.spec.name if self.spec else self.source.stem

    def reject(self, layer, gate, reason):
        self.rejected[layer] = (gate, reason)
        if layer in self.selected:
            self.selected.remove(layer)


def parse_kernel_arg(arg) -> tuple:
    """``"k.cc"`` / ``"k.cc@1,4"`` / ``(path, layers)`` -> ``(Path, layers|None)``."""
    if isinstance(arg, (tuple, list)):
        return Path(arg[0]), (None if arg[1] in (None, "auto") else [int(x) for x in arg[1]])
    text = str(arg)
    if "@" in text:
        path, sel = text.rsplit("@", 1)
        layers = None if sel.strip() in ("", "auto", "all") else \
            [int(x) for x in sel.split(",") if x.strip()]
        return Path(path), layers
    return Path(text), None


def plan(net, kernels, layers=None, verbose: bool = True) -> list:
    """Gates 1-2 for every kernel. *layers* applies to kernels given without ``@``."""
    plans, claimed = [], {}
    for arg in kernels:
        path, sel = parse_kernel_arg(arg)
        rep = lint_source(path)
        kp = KernelPlan(source=rep.source, lint=rep, requested=sel if sel is not None else layers)
        plans.append(kp)
        if verbose:
            print(f"[aiek] {path.name}: lint -- {len(rep.errors)} error(s), "
                  f"{sum(f.level == 'warning' for f in rep.findings)} warning(s)")
            for f in rep.findings:
                print(f"        {f}")
        if not rep.ok or not rep.specs:
            continue
        kp.spec = rep.specs[0]
        if len(rep.specs) > 1 and verbose:
            print(f"        note: {len(rep.specs)} AIEK_SPECs; using {kp.spec.entry}")
        for lv in detect_layers(kp.spec, net):
            kp.verdicts[lv.layer] = lv
        wanted = kp.requested if kp.requested is not None else \
            [l for l, v in kp.verdicts.items() if v.ok]
        for layer in wanted:
            lv = kp.verdicts.get(layer)
            if lv is None:
                kp.reject(layer, "detect", "no such layer (not in the manifest, or never called)")
            elif not lv.ok:
                kp.reject(layer, "detect", lv.reason)
            elif layer in claimed:
                kp.reject(layer, "detect", f"already claimed by {claimed[layer]}")
            else:
                kp.selected.append(layer)
                claimed[layer] = kp.name
        if verbose:
            _print_detect(kp)
    return plans


def _print_detect(kp: KernelPlan) -> None:
    ok = [v for v in kp.verdicts.values() if v.ok]
    print(f"[aiek] {kp.name}: detect -- {len(ok)} of {len(kp.verdicts)} layers fit "
          f"the spec: {', '.join(f'{v.layer:02d}' for v in ok) or 'none'}")
    for layer in kp.selected:
        print(f"        layer {layer:02d}: selected -- {kp.verdicts[layer].reason}")
    for layer, (gate, why) in sorted(kp.rejected.items()):
        print(f"        layer {layer:02d}: rejected at {gate} -- {why}")


def _calibrate(net, plans, verbose: bool) -> None:
    from .calibrate import calibrate_layer

    for kp in plans:
        for layer in list(kp.selected):
            calls = [v.call for v in kp.verdicts[layer].calls]
            res = calibrate_layer(net, kp.spec, calls, verbose=verbose)
            kp.calib[layer] = res
            bad = [r for r in res if not r.ok]
            if bad:
                kp.reject(layer, "calibrate", bad[0].summary())


def _emulate(net, plans, verbose: bool):
    """Gate 4, with bisection: if the combined run fails, each swap runs alone
    so the layer that moves the logits is the one rejected."""
    from .emulate import emulate

    swaps = [(kp, layer) for kp in plans for layer in kp.selected]
    if not swaps:
        return None

    def as_arg(items):
        return [(kp.spec, [v.call for v in kp.verdicts[l].calls]) for kp, l in items]

    res = emulate(net, as_arg(swaps), verbose=verbose)
    if res.ok or len(swaps) == 1:
        if not res.ok:
            kp, layer = swaps[0]
            kp.reject(layer, "emulate", res.summary())
        return res
    if verbose:
        print("  [emul] combined run failed -- emulating each layer alone to find it")
    for kp, layer in swaps:
        one = emulate(net, as_arg([(kp, layer)]), verbose=verbose, tag=f"emul_L{layer:02d}")
        if not one.ok:
            kp.reject(layer, "emulate", one.summary())
    rest = [(kp, l) for kp, l in swaps if l in kp.selected]
    return emulate(net, as_arg(rest), verbose=verbose) if rest else res


def check(out_dir, kernels, layers=None, calibrate: bool = True, emulate: bool = True,
          verbose: bool = True) -> dict:
    """Gates 1-4. Writes only under ``<out_dir>/aiek/``. Returns the report dict."""
    net = load_network(out_dir)
    plans = plan(net, kernels, layers, verbose)
    try:
        if calibrate:
            _calibrate(net, plans, verbose)
        emul = _emulate(net, plans, verbose) if emulate else None
    except HostBuildError as exc:
        for kp in plans:
            for layer in list(kp.selected):
                kp.reject(layer, "host-build", str(exc))
        emul = None
    report = _report(net, plans, emul, gates=("lint", "detect")
                     + (("calibrate",) if calibrate else ()) + (("emulate",) if emulate else ()))
    if verbose:
        _print_verdicts(report)
    return report


def _report(net, plans, emul, gates) -> dict:
    out = {"out_dir": str(net.out_dir), "gates": list(gates), "kernels": [],
           "emulation": None if emul is None else {
               "ok": emul.ok, "summary": emul.summary(), "swaps": emul.swaps,
               "baseline_top1": emul.baseline_top1, "emulated_top1": emul.emulated_top1}}
    for kp in plans:
        k = {"name": kp.name, "source": str(kp.source),
             "entry": kp.spec.entry if kp.spec else None,
             "aie_kernel": kp.lint.kernels, "lint": [str(f) for f in kp.lint.findings],
             "lint_ok": kp.lint.ok, "selected": list(kp.selected),
             "rejected": {str(l): {"gate": g, "reason": r} for l, (g, r) in kp.rejected.items()},
             "layers": {}}
        for layer in sorted(set(kp.selected) | set(kp.calib)):
            lv = kp.verdicts[layer]
            k["layers"][str(layer)] = {
                "symbol": lv.symbol, "dir": lv.dir,
                "calls": [v.call.call for v in lv.calls],
                "calibration": [r.summary() for r in kp.calib.get(layer, [])]}
        if kp.build is not None:
            k["build"] = {key: (str(v) if isinstance(v, Path) else v)
                          for key, v in kp.build.items() if key != "isolate"}
        out["kernels"].append(k)
    path = net.out_dir / "aiek" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    out["report"] = str(path)
    return out


def _print_verdicts(report: dict) -> None:
    gates = ", ".join(report["gates"])
    for k in report["kernels"]:
        sel = k["selected"]
        if sel:
            print(f"VERDICT {k['name']}: READY -- layer(s) "
                  f"{', '.join(f'{l:02d}' for l in sel)} passed {gates}"
                  + ("" if k["aie_kernel"] else " (reference-only: write the AIE "
                     "kernel before offload)"))
        else:
            print(f"VERDICT {k['name']}: NOT READY -- no layer passed {gates}")
        for layer, r in sorted(k["rejected"].items(), key=lambda kv: int(kv[0])):
            print(f"        layer {int(layer):02d} stopped at {r['gate']}: {r['reason']}")
    print(f"        report: {report['report']}")


# ═══════════════════════════════════════════════════════════════════════════
#  offload: gates 5-6 and the writes
# ═══════════════════════════════════════════════════════════════════════════

def _check_one_aie_library(plans, verbose: bool) -> None:
    """Gate 6a. The runtime calls one global ``routing()`` and caches partition
    init per meshId, so the ELF can hold ONE aiehlc library. The first kernel
    (command-line order) with selected layers keeps them."""
    owner = None
    for kp in plans:
        if not kp.selected:
            continue
        if not kp.lint.has_aie_kernel:
            for layer in list(kp.selected):
                kp.reject(layer, "build", "reference-only source: no AIE kernel to build")
            continue
        if owner is None:
            owner = kp
            continue
        for layer in list(kp.selected):
            kp.reject(layer, "link", f"{owner.name} already owns this ELF's aiehlc "
                                     f"library; the runtime has ONE routing() per ELF. "
                                     f"Put both kernels in one source, or offload them "
                                     f"in separate builds")


def _build(net, plans, aie_version: int, verbose: bool) -> None:
    """Gate 5 for the (single) kernel that survived gate 6a. A failure rejects
    every selected layer of that kernel, so ``partition.json`` keeps whatever
    those layers had before -- a failed build changes nothing."""
    from .build import aiehlc_compile, install, toolchain

    ok, detail = toolchain()
    for kp in plans:
        if not kp.selected:
            continue
        if not ok:
            kp.build = {"ok": False, "archive": None, "reason": f"no toolchain: {detail}"}
        else:
            comp = aiehlc_compile(kp.spec, net.out_dir, aie_version, verbose)
            if not comp["ok"]:
                kp.build = comp
            else:
                first = sorted(kp.selected)[0]
                aie_dir = net.out_dir / "layers" / kp.verdicts[first].dir / "aie"
                layer_calls = [[v.call for v in kp.verdicts[l].calls] for l in sorted(kp.selected)]
                kp.build = {**install(kp.spec, comp["worklocal"], aie_dir, layer_calls, verbose),
                            "aie_dir": aie_dir, "aiehlc_cached": comp.get("cached")}
        if not kp.build.get("ok"):
            log = kp.build.get("log")
            for layer in list(kp.selected):
                kp.reject(layer, "build", kp.build.get("reason", "build failed")
                          + (f" (log: {log})" if log else ""))


def _partition_doc(net) -> dict:
    path = net.out_dir / "layers" / "partition.json"
    if path.is_file():
        return json.loads(path.read_text())
    rows = [{"index": c.layer, "dir": c.dir, "kind": c.dir[3:], "symbol": c.symbol,
             "graph_node": c.node, "aiegraph": [], "target": "cpu",
             "reason": "not offloaded", "aie_dir": None, "backend": None,
             "source": None, "entry": None, "replaces": None, "archive": None,
             "archive_reason": None} for c in net.calls]
    return {"aie_ops": [], "counts_are": "graph invocations; a kernel called twice "
                                         "counts twice", "layers": rows}


def _rel(net, p) -> str:
    """Relative to the deploy tree (artifacts) or else the repo (sources), as
    ``aiegraph_partition`` writes them."""
    for base in (net.out_dir, REPO):
        try:
            return str(Path(p).resolve().relative_to(base))
        except ValueError:
            continue
    return str(p)


def _write_partition(net, plans, verbose: bool) -> Path:
    doc = _partition_doc(net)
    claimed = {kp.verdicts[l].symbol: (kp, l) for kp in plans for l in kp.selected}
    # Only a user library that actually exists displaces another one; with no
    # archive the user kernel swaps nothing and the old offload must stand.
    users = [kp for kp in plans if kp.selected and (kp.build or {}).get("ok")]
    for row in doc["layers"]:
        hit = claimed.get(row.get("symbol"))
        if hit is None:
            # Another aiehlc library still offloaded (the stem via --aie-offload)
            # would be a second routing() in the ELF -- see gate 6a.
            if users and row.get("target") == "aie":
                row.update(target="cpu", entry=None, replaces=None,
                           reason=f"superseded by user kernel {users[0].name}: one "
                                  f"aiehlc library per ELF (runtime routing())")
                if verbose:
                    print(f"[aiek] layer {row['index']:02d}: was on the AIE via "
                          f"{row.get('backend')}; back on the APU ({row['reason']})")
            continue
        kp, layer = hit
        b = kp.build or {}
        row.update(target="aie", backend="aiehlc-user",
                   reason=f"user kernel {kp.name} (calibrated, emulated)",
                   source=_rel(net, kp.source), user_kernel=kp.name,
                   aie_dir=_rel(net, b["aie_dir"]) if b.get("aie_dir") else None,
                   entry=(b.get("entries") or {}).get(row["symbol"]),
                   replaces=row["symbol"] if b.get("ok") else None,
                   archive=_rel(net, b["archive"]) if b.get("ok") else None,
                   archive_reason=None if b.get("ok") else b.get("reason", "not built"))
    targets = [r for r in doc["layers"] if r.get("target") == "aie"]
    doc["aie_count"], doc["cpu_count"] = len(targets), len(doc["layers"]) - len(targets)
    doc["aie_dirs"] = len({r["dir"] for r in targets})
    doc["cpu_dirs"] = len({r["dir"] for r in doc["layers"] if r.get("target") != "aie"})
    path = net.out_dir / "layers" / "partition.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def _link_gate(net, plans, verbose: bool) -> list:
    """Gate 6b: collisions among the archives the regenerated driver will call."""
    from frontend.tvmrelay.arm_build import aie_entries

    from .build import ROUTING_SYMBOL, link_collisions, toolchain

    if not toolchain()[0]:
        return []
    doc = json.loads((net.out_dir / "layers" / "partition.json").read_text())
    called = set(aie_entries(net.out_dir / "layers"))
    archives = sorted({net.out_dir / r["archive"] for r in doc["layers"]
                       if r.get("replaces") in called and r.get("archive")})
    hits = link_collisions(archives)
    if verbose:
        print(f"[aiek] link gate: {len(archives)} AIE archive(s) called by the driver, "
              f"{len(hits)} collision(s)")
        for sym, owners in hits[:8]:
            why = " <- one routing() per ELF" if sym == ROUTING_SYMBOL else ""
            print(f"        {sym}: {', '.join(owners)}{why}")
    return hits


def offload(out_dir, kernels, layers=None, build: bool = True, make: bool = False,
            regenerate: bool = True, force: bool = False, aie_version: int = 5,
            image=None, verbose: bool = True) -> dict:
    """Gates 1-6, then ``partition.json`` + ``arm_build/``. ``force`` skips the
    calibrate/emulate VERDICTS (they still run and are reported)."""
    net = load_network(out_dir)
    plans = plan(net, kernels, layers, verbose)
    saved = {id(kp): (list(kp.selected), dict(kp.rejected)) for kp in plans}
    _calibrate(net, plans, verbose)
    emul = _emulate(net, plans, verbose)
    if force:
        for kp in plans:
            kp.selected, kp.rejected = saved[id(kp)]
    _check_one_aie_library(plans, verbose)
    if build:
        _build(net, plans, aie_version, verbose)
    else:
        for kp in plans:
            kp.build = {"ok": False, "archive": None, "reason": "not built (--no-build)"}
    part = _write_partition(net, plans, verbose)
    hits = _link_gate(net, plans, verbose) if build else []
    report = _report(net, plans, emul, gates=("lint", "detect", "calibrate", "emulate",
                                              "build", "link"))
    report.update(partition=str(part), link_collisions=hits)
    if regenerate and (net.out_dir / "arm_build").is_dir():
        from frontend.tvmrelay.arm_build import build_arm_elf

        arm = build_arm_elf(net.out_dir, c_name=f"{net.stem}.c", run_make=make,
                            image=image, verbose=False)
        report["arm"] = {k: arm.get(k) for k in ("ok", "elf", "reason", "stderr")}
        if verbose:
            state = (f"main.elf linked: {arm['elf']}" if arm.get("elf") else
                     f"sources regenerated ({arm.get('reason') or 'make not run'})")
            print(f"[aiek] arm_build: {state}")
    if verbose:
        _print_verdicts(report)
        _print_offload(net, plans)
    return report


def _print_offload(net, plans) -> None:
    for kp in plans:
        if not kp.selected:
            continue
        b = kp.build or {}
        if b.get("ok"):
            print(f"OFFLOAD {kp.name}: {Path(b['archive']).name} -> graph_driver.c calls "
                  f"{len(b['entries'])} adapter(s) under -DGRAPH_AIE_OFFLOAD")
        else:
            print(f"OFFLOAD {kp.name}: recorded in partition.json (--no-build), NOT "
                  f"swapped -- the driver only swaps a layer whose archive exists")
