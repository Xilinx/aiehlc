###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Routing IR API for agents and tuners (doc/design/routing_ir_reentry.md).

The routing IR is the frontend's whole tiling decision for one kernel: mesh,
per-tensor ``pp_depth`` / ``halo``, per-partition ``#routing.tiling`` (one dim
per tensor dim, nested levels base/total/slice/step/rounds). aiehlc dumps it as
``ir/dfschedule/0_initial.mlir`` and reads an edited copy back through
``AIEHLC_ROUTING_IR_IN``; a round trip is byte-identical
(``script/test/routing_ir_roundtrip.sh``).

    from aiebackend.routing_ir import RoutingIR, build
    ir = RoutingIR.load("ir/dfschedule/0_initial.mlir")
    ir.tensors()                       # [{arg, direction, shape, attrs}]
    ir.set_pp_depth(2, 1)              # output C single-buffered
    ir.set_level(0, dim=1, depth=1, slice=128, step=128, rounds=2)
    ir.lint()                          # static findings, milliseconds
    r = ir.check()                     # whole pipeline in a scratch dir
    r.ok, r.errors, r.buffers          # dense signals, no board needed
    ir.save("edited.mlir"); build("k.cc", "edited.mlir")   # full aiehlc build

Edits go through the C++ dialect (``_aiebackend.routing_ir_edit``), so the
result always parses and verifies; ``history`` records every edit for replay.
What the IR alone cannot tell: the kernel body. A block-load kernel's own
``aie::load_buffer`` blocks are checked against the DMA only at build time
(kernel trace), and frontend-substituted ``aie::get_*`` values do not follow
IR edits -- change those in the source.
"""
import dataclasses
import difflib
import os
import re
import subprocess
import tempfile
from pathlib import Path


def _backend():
    from . import load
    return load()


@dataclasses.dataclass
class CheckResult:
    ok: bool                 # pipeline produced host/kernel/routing code
    verified: bool           # the IR parsed and verified
    diagnostics: list        # MLIR diagnostics (parse / verify)
    errors: list             # pipeline log lines that report an error
    buffers: dict            # kernel window buffers: name -> bytes
    core_free_bytes: int     # core data memory left after the buffers (-1 unknown)
    lint: list               # RoutingIR.lint() of the checked IR
    flowcheck: dict          # script/test/flowcheck.py on the artifacts: {ok, errors, flows}
    artifacts: str           # generated host.cc / kernel.cc / routing.cc
    log: str                 # the whole pipeline log

    def summary(self):
        state = "OK" if self.ok and self.flowcheck.get("ok", True) else "FAIL"
        return (f"{state}: {len(self.errors)} error(s), {len(self.lint)} lint finding(s), "
                f"{len(self.flowcheck.get('errors', []))} flowcheck error(s), buffers {self.buffers}, "
                f"core free {self.core_free_bytes} B")


class RoutingIR:
    """One routing IR module, edited in memory through the C++ dialect."""

    def __init__(self, text, history=None):
        self.text = text
        self.history = list(history or [])
        self._view = None

    @classmethod
    def load(cls, path):
        return cls(Path(path).read_text())

    def save(self, path):
        Path(path).write_text(self.text)
        return path

    # -- view ------------------------------------------------------------
    @property
    def view(self):
        if self._view is None:
            self._view = _backend().routing_ir_parse(self.text)
        return self._view

    def tensors(self):
        return self.view["tensors"]

    def partitions(self):
        return self.view["partitions"]

    def mesh(self):
        return self.view.get("mesh", {})

    def tensor(self, arg):
        return next(t for t in self.tensors() if t["arg"] == arg)

    def partition(self, tensor_arg):
        return next(p for p in self.partitions() if p["tensor_arg"] == tensor_arg)

    # -- edits -------------------------------------------------------------
    def edit(self, *edits):
        """Apply raw edit dicts (see routing_ir_api.cpp) and return self."""
        self.text = _backend().routing_ir_edit(self.text, list(edits))
        self.history.extend(edits)
        self._view = None
        return self

    def set_pp_depth(self, arg, depth):
        return self.edit({"target": "tensor", "arg": arg, "attr": "pp_depth", "value": f"{int(depth)} : i64"})

    def set_level(self, tensor_arg, dim, depth=0, **fields):
        """Change one tiling level of a tensor's partition (base/total/slice/step/rounds)."""
        return self.edit({"target": "level", "tensor_arg": tensor_arg, "dim": dim, "depth": depth, "set": fields})

    def set_attr(self, target, attr, value, **select):
        """Replace any attribute by MLIR attribute text, e.g.
        set_attr("module", "routing.fullconnect_auto", "1 : i64"),
        set_attr("tensor", "halo", "#routing.halo<...>", arg=0)."""
        return self.edit(dict(target=target, attr=attr, value=value, **select))

    def diff(self, other):
        other_text = other.text if isinstance(other, RoutingIR) else str(other)
        return "".join(difflib.unified_diff(other_text.splitlines(True), self.text.splitlines(True),
                                            "before", "after", n=0))

    # -- signals -----------------------------------------------------------
    def lint(self):
        """Static findings: tiling arithmetic, mesh agreement, knobs with no effect.
        Each rule holds on every IR the pipeline is known to build correctly."""
        out = []
        mesh = self.mesh()
        for t in self.tensors():
            pp = t["attrs"].get("pp_depth")
            if isinstance(pp, int) and pp < 1:
                out.append(f"tensor {t['arg']}: pp_depth {pp} < 1")
            if isinstance(pp, int) and pp != 2 and t["direction"] == "input":
                out.append(f"tensor {t['arg']}: note: pp_depth on an input is not read by code generation "
                           "(only output Pull flows use it); this edit has no effect")
        for p in self.partitions():
            who = f"partition {p['index']} (tensor {p['tensor_arg']})"
            part = p["partition"]
            axis_size = {"row": mesh.get("rows"), "col": mesh.get("cols")}.get(part["hwAxisOwner"])
            if axis_size and part["splitnum"] not in (1, axis_size):
                out.append(f"{who}: splitnum {part['splitnum']} != mesh {part['hwAxisOwner']}s {axis_size}")
            tensor = next((t for t in self.tensors() if t["arg"] == p["tensor_arg"]), None)
            if tensor and "halo" not in tensor["attrs"] and isinstance(tensor["shape"], list):
                sd = part["splitdim"]
                if sd < len(tensor["shape"]) and part["splitnum"] > 0 and tensor["shape"][sd] % part["splitnum"]:
                    out.append(f"{who}: tensor dim {sd} ({tensor['shape'][sd]}) is not divisible by splitnum "
                               f"{part['splitnum']}")
            for d, dim in enumerate(p["tiling"]):
                parent = None
                for k, lv in enumerate(dim["levels"]):
                    at = f"{who} dim {d} level {k}"
                    if lv["slice"] <= 0 or lv["step"] <= 0 or lv["rounds"] < 1:
                        out.append(f"{at}: slice/step must be > 0 and rounds >= 1 ({lv})")
                        continue
                    if lv["total"] != lv["slice"] * lv["rounds"]:
                        out.append(f"{at}: total {lv['total']} != slice {lv['slice']} x rounds {lv['rounds']}")
                    if lv["step"] > lv["slice"]:
                        out.append(f"{at}: step {lv['step']} > slice {lv['slice']} leaves gaps")
                    if parent and lv["base"] != parent["slice"]:
                        out.append(f"{at}: base {lv['base']} != the outer level's slice {parent['slice']}")
                    parent = lv
        return out

    def check(self, aie_gen="Gen5", work_dir=None):
        """Verify, then run the whole pipeline on this IR in a scratch dir (no kernel
        body, no board). Seconds; returns a CheckResult."""
        wd = work_dir or tempfile.mkdtemp(prefix="routing_ir_check.")
        os.makedirs(wd, exist_ok=True)
        r = _backend().routing_ir_check(self.text, aie_gen, os.path.abspath(wd))
        log = r.get("log", "")
        errors = [l for l in log.splitlines() if re.search(r"\bERROR\b|error:|FAILED|Budget exceeded", l)]
        buffers, free = {}, -1
        kernel = Path(r.get("artifacts", wd)) / "kernel.cc"
        if kernel.is_file():
            for m in re.finditer(r"^v4int8 (buf_\w+)\[(\d+)\];", kernel.read_text(), re.M):
                buffers[m.group(1)] = int(m.group(2)) * 4
        m = re.search(r"Free space: (\d+) bytes", log)
        if m:
            free = int(m.group(1))
        flow = {}
        if r["ok"]:
            flow = _flowcheck().check(r["artifacts"])
        return CheckResult(bool(r["ok"]), bool(r["verified"]), list(r["diagnostics"]), errors, buffers, free,
                           self.lint() if r["verified"] else [], flow, r.get("artifacts", ""), log)


def _flowcheck():
    """script/test/flowcheck.py (stdlib only), loaded by path from the checkout."""
    import importlib.util
    path = Path(__file__).resolve().parents[5] / "script" / "test" / "flowcheck.py"
    spec = importlib.util.spec_from_file_location("aiehlc_flowcheck", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(src, ir_path, aie_version=5, work_dir=None, repo=None, timeout=1800):
    """Full aiehlc build of `src` with the routing IR read from `ir_path`
    (AIEHLC_ROUTING_IR_IN). Returns {rc, reentry, elf, log}."""
    repo = Path(repo or Path(__file__).resolve().parents[5])
    wd = Path(work_dir or tempfile.mkdtemp(prefix="routing_ir_build."))
    wd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, AIEHLC_ROUTING_IR_IN=os.path.abspath(ir_path))
    cmd = (f"source '{repo}/script/aiehlc.sh' --aie-version {aie_version} "
           f"--runtime-source-file '{os.path.abspath(src)}'")
    p = subprocess.run(["bash", "-c", cmd], cwd=wd, env=env, capture_output=True, text=True, timeout=timeout)
    log = p.stdout + p.stderr
    (wd / "build.log").write_text(log)
    line = next((l for l in log.splitlines() if "AIEHLC_ROUTING_IR_IN" in l), "")
    elf = wd / "aout" / "main.elf"
    return {"rc": p.returncode, "reentry": line, "elf": str(elf) if elf.is_file() else None,
            "log": str(wd / "build.log"), "work_dir": str(wd)}
