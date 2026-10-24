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

from . import _env
from .contract import check_region
from .errors import AiegraphError
from .report import PartitionPlan

__all__ = ["run_partition", "reset_aie_rows"]

FALLBACKS = ("cpu", None)
OBJECTIVES = ("latency", None)


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
