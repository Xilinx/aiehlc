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
from .report import (AIProtoResult, BuildReport, CheckReport, DetectReport, PartitionPlan,
                     KernelOptResult, PerfReport, ProfileReport, RegionCheck, RunReport)

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
    source: str = ""       # the code this layer runs: its TVM C, or the user kernel source

    @property
    def name(self) -> str:
        return f"layer{self.index}"

    def __repr__(self) -> str:
        return (f"Layer {self.name}: {self.label}  [{self.placement}]  calls "
                f"{', '.join(f'[{c:02d}]' for c in self.calls)}\n  {self.signature}\n"
                f"  TVM C: layers/{self.dir}/" +
                (f"\n  runs:  {self.source}" if self.placement != "cpu" else ""))


class _Text(str):
    """A string that prints as itself at the prompt (no quotes)."""

    def __repr__(self) -> str:
        return str(self)


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
        self.last_run = None
        self._kernels = {}          # register_kernel: name -> KernelContract
        self._assign = {}           # assign: region name -> "cpu" | "aie:<kernel>"
        self._regions = None

    # ── structure ────────────────────────────────────────────────────────

    def _network(self):
        _env.frontend("")
        from frontend.tvmrelay.aiekernel.graph import load_network
        return load_network(self.workdir)

    def _placement(self) -> dict:
        """{symbol: (placement, source)} for the layers partition.json puts on the AIE."""
        part = self.workdir / "layers" / "partition.json"
        if not part.is_file():
            return {}
        out = {}
        for row in json.loads(part.read_text()).get("layers", []):
            if row.get("target") == "aie" and row.get("replaces"):
                src = row.get("source") or row.get("aie_dir") or ""
                out[row["symbol"]] = (f"aie:{row.get('user_kernel') or row.get('backend')}", src)
        return out

    @property
    def layers(self) -> list:
        net, place = self._network(), self._placement()
        out = []
        for idx, calls in sorted(net.layers().items()):
            c = calls[0]
            where, src = place.get(c.symbol, ("cpu", ""))
            if where == "cpu":
                src = f"layers/{c.dir}/{c.dir}.c" if c.dir else "resnet18.c"
            out.append(Layer(idx, c.symbol, c.dir, [x.call for x in calls], _label(c),
                             _sig(c), where, src))
        return out

    def recorded_path(self, p) -> Path:
        """A path as partition.json records it, made absolute. aiekernel writes
        it relative to the workdir when the file is inside it (TVM C, an
        ``aiopt/`` copy) and to the repo otherwise (``aiekernel.flow._rel``) --
        so the root depends on WHERE the file is, not on CPU vs AIE."""
        p = Path(p)
        if p.is_absolute():
            return p
        for base in (self.workdir, _env.repo_root()):
            if (base / p).exists():
                return (base / p).resolve()
        return (self.workdir / p).resolve()

    def source_path(self, layer) -> Path:
        """Absolute path of the code *layer* runs: its TVM C file (CPU) or its
        user kernel source (AIE)."""
        l = layer if isinstance(layer, Layer) else self.layer(layer)
        return self.recorded_path(l.source)

    def sources(self, full: bool = True) -> str:
        """Which code every layer runs: the TVM C file of a CPU layer, the user
        kernel source of an AIE one. Absolute paths; ``full=False`` prints them
        as partition.json records them (workdir- / repo-relative). A table."""
        rows = [(l.name, l.placement, str(self.source_path(l)) if full else l.source)
                for l in self.layers]
        w = max(len(p) for _, p, _ in rows)
        text = "\n".join([f"  {'layer':<8} {'on':<{w}}  source"] +
                         [f"  {n:<8} {p:<{w}}  {s}" for n, p, s in rows])
        return _Text(f"layer sources ({self.workdir}):\n" + text)

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

    def offload_prototype(self, layer=None, *, region=None, entry: str = None):
        """A kernel-file backbone for a layer: its arguments in TVM call order
        (kind, dtype, shape, role, value facts), a proven ``AIEK_SPEC``, and
        ``.write(dir)`` -> ``<entry>.cc`` (aiehlc kernel frame + host run
        function + entry, the conv2dstem_uk.cc architecture; compute and staging
        are TODOs) and ``<entry>_ref.c``. ``layer`` uses the TVM layer numbering
        of ``offload_check``; ``region=`` an ONNX region name. No argument: the
        general kernel-file format."""
        from .prototype import prototype_for
        return prototype_for(self, layer, region, entry)

    def ai_offload_prototype(self, layer=None, *, region=None, entry: str = None, rounds: int = 3,
                             build: bool = True, model: str = None, claude: str = "claude",
                             timeout_s: float = 2400, out_dir=None, quiet: bool = False):
        """``offload_prototype(layer)`` writes the backbone, then ``claude -p`` fills its
        TODOs until ``prototype_check`` passes (no TODO left; the kernel AND host code,
        emulated on x86, byte-equal to TVM's output; DMA trace; ``offload_check``; then
        an aiehlc build of ``test_<entry>.cc``). At most *rounds* Claude turns. Written
        to ``<workdir>/aiproto/layerNN_prototype/``. Returns the kernel, path-like:
        ``emod.offload("layer4", res)``."""
        from .aiprototype import ai_offload_prototype
        return ai_offload_prototype(self, layer, region=region, entry=entry, rounds=rounds, build=build,
                                    model=model, claude=claude, timeout_s=timeout_s, out_dir=out_dir,
                                    quiet=quiet)

    def prototype_check(self, layer, src_dir, *, entry: str = None, build: bool = False):
        """Is the prototype in *src_dir* a logically correct kernel for *layer*? Gates:
        todo, kemu (x86 emulation of kernel + host vs TVM), trace, check, [build]."""
        from .aiprototype import prototype_check
        return prototype_check(self, layer, src_dir, entry=entry, build=build)

    def offload_reset(self, layers=None, verbose: bool = False) -> list:
        """Put offloaded layers back on the CPU: all of them, or *layers* (any
        ``resolve`` reference). ``partition.json`` rows lose their AIE entry and
        the driver is regenerated without the swap; the conversion and the built
        libraries stay (an unreferenced archive links nothing). Returns the
        layer names reset."""
        part = self.workdir / "layers" / "partition.json"
        if not part.is_file():
            return []
        want = None if layers is None else set(self.resolve(layers))
        doc = json.loads(part.read_text())
        reset = []
        for row in doc.get("layers", []):
            if row.get("target") != "aie" or (want is not None and row.get("index") not in want):
                continue
            row.update(target="cpu", entry=None, replaces=None, archive=None, aie_dir=None,
                       backend=None, user_kernel=None, source=None,
                       reason="offload reset (aiegraph): back on the APU")
            if f"layer{row.get('index')}" not in reset:
                reset.append(f"layer{row.get('index')}")
        if not reset:
            return []
        targets = [r for r in doc["layers"] if r.get("target") == "aie"]
        doc["aie_count"], doc["cpu_count"] = len(targets), len(doc["layers"]) - len(targets)
        doc["aie_dirs"] = len({r["dir"] for r in targets})
        doc["cpu_dirs"] = len({r["dir"] for r in doc["layers"] if r.get("target") != "aie"})
        part.write_text(json.dumps(doc, indent=2) + "\n")
        if (self.workdir / "arm_build").is_dir():
            ab = _env.frontend("arm_build")
            ab.build_arm_elf(self.workdir, c_name=f"{self._stem()}.c", run_make=False,
                             image=self.image, verbose=verbose)
        self.last_build = None
        print(f"[aiegraph] offload reset: {', '.join(reset)} back on the CPU (driver regenerated)")
        return reset

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
        """``offload_check`` + aiehlc build + link gate + driver swap.

        ``offload("layer1", False)`` undoes it: the layer(s) go back on the CPU
        (= ``offload_reset("layer1")``); returns the layer names reset."""
        if source is False:
            reset = self.offload_reset(spec, verbose=verbose)
            if not reset:
                print(f"[aiegraph] offload {spec!r} False: nothing to do, not on the AIE")
            return reset
        _env.frontend("")
        from frontend.tvmrelay.aiekernel.flow import offload
        groups, args = self._kernel_args(spec, source)
        print(f"[aiegraph] offload {', '.join(Path(a).name for a in args)}: checks, then "
              f"aiehlc build (~2 min the first time, cached after)...")
        raw = offload(self.workdir, args, build=build, make=make, force=force,
                      image=self.image, verbose=verbose)
        return CheckReport("offload", raw, groups)

    def offload_AI_proto(self, spec, source, *, rounds: int = 3, model: str = None,
                         claude: str = "claude", timeout_s: float = 1800, out_dir=None,
                         quiet: bool = False, verbose: bool = False) -> AIProtoResult:
        """When ``offload_check(spec, source)`` is NOT READY, have ``claude -p`` port
        *source* to the layer(s) in a copy of its directory, until our own
        ``offload_check`` says READY (at most *rounds* Claude turns). Returns the new
        kernel, path-like: ``emod.offload(("layer4", new_kernel))``."""
        from .aiproto import offload_ai_proto
        return offload_ai_proto(self, spec, source, rounds=rounds, model=model, claude=claude,
                                timeout_s=timeout_s, out_dir=out_dir, quiet=quiet,
                                verbose=verbose)

    def AI_perf_analysis(self, layer, *, ai: bool = True, model: str = None,
                         claude: str = "claude", timeout_s: float = 900, out_dir=None,
                         quiet: bool = False) -> PerfReport:
        """Why an offloaded layer is slow, and what to change (skill kernelperfcheck).

        Deterministic facts (measured ms from the last ``run(profile=True)``,
        pragmas, generated host.cc, kernel ELF / kernel.lst, aiehlc.log), a cost
        model and the skill's rules, then ``claude -p`` (read-only) reads it all
        and returns bottlenecks + ranked suggestions with predicted savings.
        ``ai=False``: facts, model and rules only. Saved to
        ``<workdir>/aiperf/<layer>/report.{md,json}``."""
        from .aiperf import perf_analysis
        return perf_analysis(self, layer, ai=ai, model=model, claude=claude,
                             timeout_s=timeout_s, out_dir=out_dir, quiet=quiet)

    AI_perf_analaysis = AI_perf_analysis      # the spelling first asked for

    def AI_kernel_accelerate(self, layer, report: PerfReport = None, *, rounds: int = 2,
                             board: str = None, host: str = None, model: str = None,
                             claude: str = "claude", timeout_s: float = 2400, out_dir=None,
                             quiet: bool = False, verbose: bool = False) -> KernelOptResult:
        """Let ``claude -p`` apply *report*'s suggestions to a copy of the layer's
        kernel. *report*: a ``PerfReport``, a path to a saved one, or ``None`` = the
        saved ``aiperf/<layer>/report.json`` (else a fresh ``AI_perf_analysis``).
        The copy lives in ``<workdir>/aiopt/``; every round is gated by our own
        ``offload_check`` + an aiehlc build of the copy. The result compares the
        facts and cost model of the old and the NEW build (predicted ms). With
        ``board=`` it also offloads the copy and measures it there
        (``run(profile=True)``) -- that changes this model's placement."""
        from .aiperf import kernel_accelerate
        return kernel_accelerate(self, layer, report, rounds=rounds, board=board, host=host,
                                 model=model, claude=claude, timeout_s=timeout_s, out_dir=out_dir,
                                 quiet=quiet, verbose=verbose)

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
              profile: bool = False, verbose: bool = False) -> BuildReport:
        """Link ``main.elf`` (A78; needs the Vitis cross toolchain), or with
        ``target="x86"`` ``main_local.elf`` for this host. ``local=True`` builds
        both. ``profile=True`` links ``main.elf`` with ``make PROFILE=1`` (a
        per-call time table after "inference", read by ``profile()``).
        ``output=`` copies the ELF(s), ``partition.json`` and ``network.md``
        into that directory. Never runs anything on a board."""
        if target not in BUILD_TARGETS:
            raise AiegraphError("build", f"target={target!r}", hint=f"one of {', '.join(BUILD_TARGETS)}")
        ab = _env.frontend("arm_build")
        x86 = target == "x86"
        res = ab.build_arm_elf(self.workdir, c_name=f"{self._stem()}.c", run_make=not x86,
                               image=self.image, local=local or x86, local_run=False,
                               verbose=verbose, make_vars={"PROFILE": 1} if profile else None)
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

    def run(self, board: str = "palmyra", host: str = None, *, elf=None, build: bool = True,
            profile: bool = False, netload: bool = None, reboot: bool = None,
            timeout: float = 1800, log=None, args=(), verbose: bool = True) -> RunReport:
        """Run ``main.elf`` on a lab board and parse what it printed.

        ``emod.run()`` -- palmyra, host from ``$PALIP`` / ``script/test/envlocal.sh``;
        ``emod.run(ag.VEK385, "hostname")`` -- a VEK385. ``main.elf`` is (re)built
        first when missing or older than the driver / placement / AIE archives
        (``build=False`` refuses instead). ``netload`` (palmyra default True) and
        ``reboot`` (VEK385 default True) override the board defaults; ``args`` go
        to the board script verbatim. Console log: ``arm_build/run_<board>.log``.
        ``profile=True`` makes ``main.elf`` a ``make PROFILE=1`` build (two timer
        reads per call), so ``emod.profile(ag.ALL)`` can show each layer's time
        afterwards; ``profile=False`` (default) a plain one. Switching rebuilds.
        """
        from .board import run_on_board
        path = Path(elf) if elf else self._board_elf(build, verbose, profile)
        rep = run_on_board(path, board, host, netload=netload, reboot=reboot, timeout=timeout,
                           log=log, args=args, verbose=verbose)
        self.last_run = rep
        return rep

    def _board_elf(self, build: bool, verbose: bool, profile: bool = False) -> Path:
        arm = self.workdir / "arm_build"
        elf = arm / "main.elf"
        inputs = [arm / "graph_driver.c", self.workdir / "layers" / "partition.json",
                  *self.workdir.glob("layers/*/aie/build/*.a")]
        newest = max((f.stat().st_mtime for f in inputs if f.is_file()), default=0)
        ab = _env.frontend("arm_build")
        profiled = ab.built_make_vars(arm).get("PROFILE") == "1"
        if not elf.is_file():
            why = "missing"
        elif elf.stat().st_mtime < newest:
            why = "older than its inputs"
        elif profile != profiled:
            why = "built with" + ("out" if profile else "") + " PROFILE=1"
        else:
            return elf
        if not build:
            raise AiegraphError("run", f"{elf} is {why}", hint="emod.build(target='a78'), or run(build=True)")
        print(f"[aiegraph] run: main.elf {why}; building it first")
        rep = self.build(target="a78", profile=profile, verbose=verbose)
        if not rep.ok:
            raise AiegraphError("run", f"main.elf did not link: {rep.reason}",
                                hint="emod.build(target='a78', verbose=True)")
        return Path(rep.elf)

    def profile(self, which="all", *, by: str = "call", top: int = None,
                run: RunReport = None) -> ProfileReport:
        """Each layer's time on the board, from the last ``run(profile=True)``.
        Without one (no run, or a run without profiling) it prints that
        profiling is not enabled and returns ``None``.

        ``which``: ``ag.ALL`` (every call), or layers (``"layer1"``, ``[1, 4]``,
        any ``resolve`` reference). ``by="call"`` one row per call, ``"layer"``
        one row per layer (its calls summed). ``top=N`` keeps the N slowest
        rows. Reads ``run`` / ``last_run``, else the newest
        ``arm_build/run_*.log`` (so it works after ``ag.open``)."""
        from .board import parse_console
        if by not in ("call", "layer"):
            raise AiegraphError("profile", f"by={by!r}", hint="'call' or 'layer'")
        rep = run or self.last_run
        board = rep.board if rep else None
        if rep is None:
            logs = sorted((self.workdir / "arm_build").glob("run_*.log"),
                          key=lambda f: f.stat().st_mtime)
            if not logs:
                print("[aiegraph] profile: profiling not enabled -- no board run yet; "
                      "emod.run(profile=True), then emod.profile(ag.ALL)")
                return None
            got = parse_console(logs[-1].read_text(errors="replace"))
            rep = RunReport(ok=True, board=logs[-1].stem[4:], host=None, elf=None,
                            log=str(logs[-1]), inference_ms=got["inference_ms"],
                            profile=got["profile"])
            board = rep.board
        if not rep.profile:
            print(f"[aiegraph] profile: profiling not enabled -- the last run ({rep.log}) used a "
                  "main.elf without PROFILE=1; emod.run(profile=True), then emod.profile(ag.ALL)")
            return None
        return self._profile_rows(rep, which, by, top, board)

    def _profile_rows(self, rep: RunReport, which, by: str, top, board) -> ProfileReport:
        of_call = {c: l for l in self.layers for c in l.calls}
        want = None if which in (None, "all", "ALL") else set(self.resolve(which))
        total = sum(ms for _, ms, _, _ in rep.profile) or 1.0
        rows = []
        for call, ms, where, op in rep.profile:
            l = of_call.get(call)
            if want is not None and (l is None or l.index not in want):
                continue
            rows.append({"call": call, "layer": l.name if l else "?",
                         "op": l.label if l else op[:28], "where": where, "ms": ms})
        if by == "layer":
            agg = {}
            for r in rows:
                a = agg.setdefault(r["layer"], {"calls": [], "layer": r["layer"], "op": r["op"],
                                                "where": r["where"], "ms": 0.0})
                a["calls"].append(r["call"])
                a["ms"] += r["ms"]
                if r["where"] != a["where"]:
                    a["where"] = "mix"
            rows = list(agg.values())
        for r in rows:
            r["pct"] = 100.0 * r["ms"] / total
        if top:
            rows = sorted(rows, key=lambda r: -r["ms"])[:top]
        return ProfileReport(rows, by, board, rep.inference_ms, total, rep.log)

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
            src = f"  <- {l.source}" if l.placement != "cpu" else ""
            lines.append(f"  {l.name:<8} {len(l.calls):<6} {l.label:<{w}}  {sig:<58} {l.placement}{src}")
        n_calls = sum(len(l.calls) for l in layers)
        aie = [l.name for l in layers if l.placement != "cpu"]
        lines.append(f"  {len(layers)} layers, {n_calls} calls" +
                     (f"; on AIE: {', '.join(aie)}" if aie else ""))
        lines.append("  every layer's code: print(module.sources())")
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
                     verbose: bool = False, example_inputs=None,
                     keep_offloads: bool = False) -> EmbeddedModel:
    """fp32 model -> int8 -> one C file per layer + A78 driver (no ``make``).

    Starts FRESH: every layer on the CPU. The conversion itself is cached -- the
    same model + options into the same workdir reuses it at once -- but layers
    offloaded by an earlier session are put back on the CPU unless
    ``keep_offloads=True``. ``force=True`` redoes the conversion from scratch.
    ``ag.open(workdir)`` adopts a tree as it is, offloads included.
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
            if not keep_offloads:
                emod.offload_reset(verbose=verbose)
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
