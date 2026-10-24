###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""``ag.EmbeddedModel``: an int8 network as per-layer C for the A78, plus offload.

It is a handle on one deploy tree (the directory ``deploy_flow.py`` writes):
``embedded_convert`` produces one, ``ag.open`` adopts an existing one. Every
method delegates -- conversion to ``deploy_flow.run``, layers to
``aiekernel.graph``, checks and offload to ``aiekernel.flow``, builds to
``arm_build`` -- and wraps the result in a report that prints itself.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import _env
from .contract import KernelContract, check_region, make_contract
from .errors import AiegraphError
from .model import Model, copy_into, load
from .plan import group_by_source, parse_offload
from .region import Regions, build_regions
from .report import BuildReport, CheckReport, DetectReport, PartitionPlan, RegionCheck

__all__ = ["EmbeddedModel", "Layer", "embedded_convert", "convert", "open_workdir", "TARGETS"]

TARGETS = ("a78", "cpu", "x86", "aie")
BUILD_TARGETS = ("a78", "cpu", "aie", "x86")    # x86: main_local.elf; the rest: main.elf
_STATE = "aiegraph.json"
_STATE_VERSION = 1


@dataclass
class Layer:
    index: int
    symbol: str
    dir: str
    calls: list            # call indices
    label: str
    signature: str
    placement: str

    @property
    def name(self) -> str:
        return f"layer{self.index}"

    def __repr__(self) -> str:
        return (f"Layer {self.name}: {self.label}  [{self.placement}]  calls "
                f"{', '.join(f'[{c:02d}]' for c in self.calls)}\n  {self.signature}\n"
                f"  TVM C: layers/{self.dir}/")


def _label(call) -> str:
    """``conv7x7/2 ->uint8``, ``conv3x3/1 +res ->int8``, ``max pool2d`` ..."""
    acts = [a for a in call.args if a.kind == "in"]
    outs = [a for a in call.args if a.kind == "out"]
    wts = [a for a in call.args if a.kind == "param" and len(a.shape) in (4, 6)]
    if "conv2d" in call.symbol and wts and acts and outs:
        k = int(wts[0].shape[2])
        src = max(acts, key=lambda a: a.shape[2] if len(a.shape) > 2 else 0)
        hi, ho = int(src.shape[2]), int(outs[0].shape[2])
        s = (hi - k) // (ho - 1) if ho > 1 else 1
        res = " +res" if len(acts) > 1 else ""
        return f"conv{k}x{k}/{s}{res} ->{outs[0].dtype}"
    kind = re.sub(r"_[0-9a-f]{16}_*\d*$", "", call.dir[3:] if call.dir else call.symbol)
    return kind.replace("_", " ")[:28]


def _sig(call) -> str:
    acts = [a for a in call.args if a.kind == "in"]
    outs = [a for a in call.args if a.kind == "out"]
    fmt = lambda a: f"{a.dtype}[{','.join(str(d) for d in a.shape)}]"   # noqa: E731
    return (" ".join(fmt(a) for a in acts) or "-") + " -> " + " ".join(fmt(a) for a in outs)


class EmbeddedModel:
    """An int8 model compiled to per-layer C for the A78 (and maybe the AIE)."""

    def __init__(self, workdir, model: Model = None, target: str = "a78",
                 image=None, state: dict = None):
        self.workdir = Path(workdir).resolve()
        self.model = model
        self.target = target
        self.image = image
        self.state = state or {}
        self.last_build = None
        self._kernels = {}          # register_kernel: name -> KernelContract
        self._assign = {}           # assign: region name -> "cpu" | "aie:<kernel>"
        self._regions = None

    # ── structure ────────────────────────────────────────────────────────

    def _network(self):
        _env.frontend("")
        from frontend.tvmrelay.aiekernel.graph import load_network
        return load_network(self.workdir)

    def _placement(self) -> dict:
        part = self.workdir / "layers" / "partition.json"
        if not part.is_file():
            return {}
        out = {}
        for row in json.loads(part.read_text()).get("layers", []):
            if row.get("target") == "aie" and row.get("replaces"):
                out[row["symbol"]] = f"aie:{row.get('user_kernel') or row.get('backend')}"
        return out

    @property
    def layers(self) -> list:
        net, place = self._network(), self._placement()
        out = []
        for idx, calls in sorted(net.layers().items()):
            c = calls[0]
            out.append(Layer(idx, c.symbol, c.dir, [x.call for x in calls], _label(c),
                             _sig(c), place.get(c.symbol, "cpu")))
        return out

    def resolve(self, ref) -> list:
        """Any layer reference -> sorted layer indices (see the design doc §4.4)."""
        layers = self.layers
        known = {l.index for l in layers}
        if ref in (None, "auto", "*"):
            return []
        if isinstance(ref, Layer):
            return [ref.index]
        if isinstance(ref, (list, tuple, range, set)):
            return sorted({i for r in ref for i in self.resolve(r)})
        if isinstance(ref, int):
            out = [ref]
        else:
            text = str(ref).strip().lower()
            m = re.fullmatch(r"(?:layer)?(\d+)(?:\s*-\s*(?:layer)?(\d+))?", text)
            c = re.fullmatch(r"call\[?(\d+)\]?", text)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
                out = [i for i in range(lo, hi + 1) if i in known] if m.group(2) else [lo]
            elif c:
                out = [l.index for l in layers if int(c.group(1)) in l.calls]
            else:
                out = [l.index for l in layers
                       if fnmatch.fnmatch(l.label.lower(), text) or fnmatch.fnmatch(l.dir.lower(), text)]
        bad = [i for i in out if i not in known]
        if not out or bad:
            raise AiegraphError("layer", f"no layer matches {ref!r}",
                                hint="print(emod) lists them; refs: 'layer1', 1, 'layer4-9', "
                                     "'conv3x3*', 'call[08]'")
        return sorted(out)

    def layer(self, ref) -> Layer:
        (idx,) = self.resolve(ref)[:1] or [None]
        return next(l for l in self.layers if l.index == idx)

    # ── checks and offload ──────────────────────────────────────────────

    def detect(self, source) -> DetectReport:
        _env.frontend("")
        from frontend.tvmrelay import aiekernel
        src = Path(source).resolve()
        if not src.is_file():
            raise AiegraphError("check", f"{source} does not exist")
        try:
            verdicts = aiekernel.detect(self.workdir, src)
        except ValueError as exc:
            raise AiegraphError("check", str(exc).splitlines()[0],
                                hint=f"run ag.lint('{source}') for the details")
        return DetectReport(src, verdicts)

    def _kernel_args(self, spec, source):
        pairs = parse_offload(spec if source is None else (spec, source))
        groups = group_by_source(pairs, self.resolve)
        args = [str(src) if layers is None else f"{src}@{','.join(map(str, layers))}"
                for src, layers in groups.items()]
        return groups, args

    def offload_check(self, spec, source=None, verbose: bool = False) -> CheckReport:
        """lint + detect + calibrate + whole-network emulate, all on x86.
        ``offload_check("layer1", "k.cc")`` or any spec ``offload`` takes."""
        _env.frontend("")
        from frontend.tvmrelay.aiekernel.flow import check
        groups, args = self._kernel_args(spec, source)
        print(f"[aiegraph] offload_check {', '.join(Path(a).name for a in args)}: "
              f"lint, detect, calibrate vs TVM, whole-network emulation (x86)...")
        raw = check(self.workdir, args, verbose=verbose)
        return CheckReport("offload_check", raw, groups)

    def offload(self, spec, source=None, build: bool = True, make: bool = False,
                force: bool = False, verbose: bool = False) -> CheckReport:
        """``offload_check`` + aiehlc build + link gate + driver swap."""
        _env.frontend("")
        from frontend.tvmrelay.aiekernel.flow import offload
        groups, args = self._kernel_args(spec, source)
        print(f"[aiegraph] offload {', '.join(Path(a).name for a in args)}: checks, then "
              f"aiehlc build (~2 min the first time, cached after)...")
        raw = offload(self.workdir, args, build=build, make=make, force=force,
                      image=self.image, verbose=verbose)
        return CheckReport("offload", raw, groups)

    # ── regions: register / check / assign / partition ───────────────────

    @property
    def regions(self) -> Regions:
        """Weighted ops of the model by name (``conv1``, ``layer1.0.conv1``, ``fc``)."""
        if self._regions is None:
            self._regions = build_regions(self.workdir, self._network())
        return self._regions

    def register_kernel(self, name: str, source, op: str, target: str = "aie",
                        constraints: dict = None) -> KernelContract:
        """Name a kernel and the model regions it accepts; refused if the source's
        own ``AIEK_SPEC`` contradicts the contract."""
        k = make_contract(name, source, op, target, constraints)
        self._kernels[name] = k
        print(f"[aiegraph] registered {k.backend} ({k.source.name}, AIEK_SPEC {k.spec.name})")
        return k

    def _contract(self, kernel) -> KernelContract:
        if isinstance(kernel, KernelContract):
            return kernel
        name = str(kernel).removeprefix("aie:")
        if name not in self._kernels:
            raise AiegraphError("check_offload", f"no kernel registered as {name!r}",
                                hint="module.register_kernel(name=..., source=..., op=...)"
                                     + (f"; registered: {', '.join(self._kernels)}"
                                        if self._kernels else ""))
        return self._kernels[name]

    def check_offload(self, region, kernel, verbose: bool = False) -> RegionCheck:
        """Contract per region, then lint/detect/calibrate/emulate on x86 for the
        TVM layers behind the regions that fit it. Writes only under ``aiek/``."""
        k = self._contract(kernel)
        regs = self.regions.find(region)
        bad = {r.name: check_region(k, r) for r in regs}
        layers = sorted({r.layer for r in regs if not bad[r.name]})
        gates = self.offload_check(layers, str(k.source), verbose=verbose) if layers else None
        return RegionCheck(k.name, regs, bad, gates)

    def assign(self, region, backend: str) -> None:
        """Record where *region* should run: ``"cpu"`` or ``"aie:<kernel>"``.
        Nothing is built until ``partition()``."""
        if backend != "cpu":
            if not str(backend).startswith("aie:"):
                raise AiegraphError("assign", f"backend={backend!r}",
                                    hint="'cpu' or 'aie:<registered kernel name>'")
            self._contract(backend)
        for r in self.regions.find(region):
            self._assign[r.name] = backend
            share = f"; shares tvm layer{r.layer} with {', '.join(r.siblings)}" if r.siblings else ""
            print(f"[aiegraph] assign {r.label} ({r.where}) -> {backend}{share}")

    def partition(self, fallback: str = "cpu", optimize_for: str = "latency",
                  verbose: bool = False) -> PartitionPlan:
        """Apply the assignments: everything back on the A78, then offload the
        assigned regions that pass every gate; the rest run on *fallback*."""
        from .partition import run_partition
        print("[aiegraph] partition: contracts, then offload of the assigned layers "
              "(checks + aiehlc build, ~2 min the first time)...")
        return run_partition(self, fallback, optimize_for, verbose)

    # ── build / run ──────────────────────────────────────────────────────

    def build(self, target: str = "a78", output=None, *, local: bool = False,
              verbose: bool = False) -> BuildReport:
        """Link ``main.elf`` (A78; needs the Vitis cross toolchain), or with
        ``target="x86"`` ``main_local.elf`` for this host. ``local=True`` builds
        both. ``output=`` copies the ELF(s), ``partition.json`` and ``network.md``
        into that directory. Never runs anything on a board."""
        if target not in BUILD_TARGETS:
            raise AiegraphError("build", f"target={target!r}", hint=f"one of {', '.join(BUILD_TARGETS)}")
        ab = _env.frontend("arm_build")
        x86 = target == "x86"
        res = ab.build_arm_elf(self.workdir, c_name=f"{self._stem()}.c", run_make=not x86,
                               image=self.image, local=local or x86, local_run=False,
                               verbose=verbose)
        if x86:
            loc = res.get("local") or {}
            rep = BuildReport(bool(loc.get("ok")), loc.get("elf"), loc.get("reason", ""), loc)
        else:
            rep = BuildReport(bool(res.get("elf")), res.get("elf"), res.get("reason", ""),
                              res.get("local"))
        if output and rep.ok:
            rep.output, rep.files = self._export(output, rep)
        self.last_build = rep
        return rep

    def _export(self, output, rep: BuildReport) -> tuple:
        dest = Path(output).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        files = [rep.elf, (rep.local or {}).get("elf") if rep.local else None,
                 self.workdir / "layers" / "partition.json", self.network_md]
        out = []
        for f in files:
            if f and Path(f).is_file() and str(dest / Path(f).name) not in out:
                shutil.copy2(f, dest / Path(f).name)
                out.append(str(dest / Path(f).name))
        return str(dest), out

    def run_local(self, verbose: bool = False) -> BuildReport:
        """Build and run ``main_local.elf`` on this x86 host (CPU kernels everywhere)."""
        ab = _env.frontend("arm_build")
        res = ab.build_arm_elf(self.workdir, c_name=f"{self._stem()}.c", run_make=False,
                               image=self.image, local=True, local_run=True, verbose=verbose)
        local = res.get("local") or {}
        return BuildReport(bool(local.get("ok")), local.get("elf"),
                           local.get("reason", ""), local)

    def _stem(self) -> str:
        g = next(self.workdir.glob("*_graph.json"), None)
        return g.name[:-len("_graph.json")] if g else "resnet18"

    @property
    def network_md(self) -> Path:
        return self.workdir / "network.md"

    # ── printing ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        layers = self.layers
        name = self.model.name if self.model else self.workdir.name
        acc = self.state.get("accuracy") or {}
        lines = [f"EmbeddedModel {name} -> {self.target}   int8   workdir {self.workdir}"]
        if acc:
            f0, i0 = acc["fp32"][0], acc["int8"][0]
            lines.append(f"  int8 vs fp32 top-5: {'identical order' if acc.get('same_order') else 'top-1 kept' if acc.get('same_top1') else 'TOP-1 CHANGED'}"
                         f" ({i0[1]} {i0[2]:.2f} vs {f0[2]:.2f})")
        w = max(len(l.label) for l in layers) if layers else 10
        lines.append(f"  {'layer':<8} {'calls':<6} {'op':<{w}}  {'in -> out':<58} on")
        for l in layers:
            sig = l.signature if len(l.signature) <= 58 else l.signature[:55] + "..."
            lines.append(f"  {l.name:<8} {len(l.calls):<6} {l.label:<{w}}  {sig:<58} {l.placement}")
        n_calls = sum(len(l.calls) for l in layers)
        aie = [l.name for l in layers if l.placement != "cpu"]
        lines.append(f"  {len(layers)} layers, {n_calls} calls" +
                     (f"; on AIE: {', '.join(aie)}" if aie else ""))
        lines.append("  next: " + ("module.build(target='a78', output=...)" if aie else
                                   "print(module.regions)  /  module.register_kernel(name=..., "
                                   "source=..., op=...)"))
        return "\n".join(lines)


def _clean(workdir: Path) -> None:
    """Remove a previous conversion; refuse to touch a directory that is not ours."""
    if not workdir.exists() or not any(workdir.iterdir()):
        return
    if not (workdir / _STATE).is_file():
        raise AiegraphError("convert", f"{workdir} exists and was not written by aiegraph",
                            hint="pass another workdir=, or adopt it with ag.open(workdir)")
    shutil.rmtree(workdir)


def embedded_convert(model, target: str = "a78", quantize: str = "int8", calib=None,
                     image=None, workdir=None, force: bool = False,
                     verbose: bool = False, example_inputs=None) -> EmbeddedModel:
    """fp32 model -> int8 -> one C file per layer + A78 driver (no ``make``).

    Cached: the same model + options into the same workdir returns at once.
    """
    if target not in TARGETS:
        raise AiegraphError("convert", f"unknown target {target!r}",
                            hint=f"one of {', '.join(TARGETS)}")
    if quantize not in ("int8", "fp32"):
        raise AiegraphError("convert", f"quantize={quantize!r}", hint="'int8' or 'fp32'")
    m = load(model, example_inputs=example_inputs)
    workdir = Path(workdir or Path.cwd() / "worklocal" / "aiegraph" / m.name).resolve()
    state = {"version": _STATE_VERSION, "model": m.name, "model_sha": m.sha,
             "quantize": quantize, "calib": [str(c) for c in calib] if calib else None,
             "image": str(image) if image else None}
    stamp = workdir / _STATE
    done = stamp.is_file() and next(workdir.glob("*_graph.json"), None) is not None \
        and (workdir / "layers" / "manifest.json").is_file() \
        and (workdir / "arm_build" / "graph_driver.c").is_file()
    if done and not force:
        old = json.loads(stamp.read_text())
        if {k: old.get(k) for k in state} == state:
            emod = EmbeddedModel(workdir, m, target, image, old)
            print(f"[aiegraph] reusing {workdir} (same model and options)")
            set_current(emod)
            return emod
    _clean(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({**state, "complete": False}, indent=2))

    df = _env.frontend("deploy_flow")
    copy_into(m, workdir / df.MODEL_NAME)          # where fetch_model looks
    print(f"[aiegraph] converting {m.name} -> {quantize} per-layer C for {target} in {workdir}"
          f"\n           (a few minutes; verbose=True shows every stage)")
    res = df.run(workdir, skip_quantize=(quantize == "fp32"), arm=False, image=image,
                 calib=calib, verbose=verbose)
    if not res.get("ok") or not res.get("layer_count"):
        raise AiegraphError("convert", f"deploy flow failed (stage result {res.get('stage') or res})",
                            hint="re-run with verbose=True to see the failing stage")
    ab = _env.frontend("arm_build")
    arm = ab.build_arm_elf(workdir, c_name=Path(res["c_path"]).name, run_make=False,
                           image=image, verbose=verbose)
    full = {**state, "complete": True, "accuracy": res.get("accuracy"),
            "cpu_top": res.get("cpu_top"), "quantizer": res.get("quantizer"),
            "layer_count": res.get("layer_count"), "arm_sources": bool(arm.get("makefile"))}
    stamp.write_text(json.dumps(full, indent=2, default=str))
    emod = EmbeddedModel(workdir, m, target, image, full)
    set_current(emod)
    return emod


convert = embedded_convert          # ag.convert(model, target="a78")


def open_workdir(workdir, target: str = "a78") -> EmbeddedModel:
    """Adopt an existing deploy tree (e.g. ``worklocal/tvmrelay_deploy``)."""
    workdir = Path(workdir).resolve()
    if next(workdir.glob("*_graph.json"), None) is None or \
            not (workdir / "layers" / "manifest.json").is_file():
        raise AiegraphError("convert", f"{workdir} is not a deploy tree (no *_graph.json "
                            "or layers/manifest.json)", hint="ag.embedded_convert(model) makes one")
    stamp = workdir / _STATE
    state = json.loads(stamp.read_text()) if stamp.is_file() else {}
    emod = EmbeddedModel(workdir, None, target, state.get("image"), state)
    set_current(emod)
    return emod


_CURRENT = None


def set_current(emod) -> None:
    global _CURRENT
    _CURRENT = emod


def current() -> EmbeddedModel:
    if _CURRENT is None:
        raise AiegraphError("convert", "no current model",
                            hint="call ag.embedded_convert(model) (or ag.open(workdir)) first")
    return _CURRENT
