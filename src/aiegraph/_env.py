###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Where things are, and making them importable: repo, TVM 0.16, the frontends.

``aiegraph`` is installed **editable from a checkout** (``pip install -e .``):
the flow it drives needs the repo itself -- ``script/aiehlc.sh``,
``include/``, ``thirdparty/`` and the example kernels -- not just Python files.

TVM 0.16 has no wheel (PyPI starts at 0.25), so it lives as a source build in
``thirdparty/tvm-0.16``. :func:`attach_tvm` puts that build's ``python/`` on
``sys.path`` (and on ``PYTHONPATH`` for child processes) when ``import tvm``
does not already give 0.16 -- TVM locates ``libtvm.so`` relative to its package
directory, so nothing has to be copied. Only when no build exists does
anything get compiled, and then only through :func:`setup` or the existing
first-use provisioning in ``frontend.tvmrelay`` (10-25 minutes, announced).
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AiegraphError

__all__ = ["repo_root", "attach_tvm", "tvm_status", "frontend", "doctor", "setup",
           "Doctor", "aiehlc_binary"]

_PROBE = ("import tvm, sys\n"
          "from tvm import relay\n"
          "sys.stdout.write(str(tvm.__version__))\n")


def repo_root() -> Path:
    """The checkout this package was installed from (``AIEHLC_ROOT`` overrides)."""
    env = os.environ.get("AIEHLC_ROOT")
    root = Path(env).resolve() if env else Path(__file__).resolve().parents[2]
    if not (root / "script" / "aiehlc.sh").is_file():
        raise AiegraphError(
            "install", f"no aiehlc checkout at {root}",
            hint="install aiegraph EDITABLE from the repo: `pip install -e <aiehlc repo>` "
                 "(or set AIEHLC_ROOT to the checkout)")
    src = str(root / "src")
    if src not in sys.path:               # `frontend.*` without PYTHONPATH=src
        sys.path.append(src)
    return root


def _tvm_build_python() -> Path:
    return repo_root() / "thirdparty" / "tvm-0.16" / "python"


def tvm_status() -> tuple:
    """``(ok, detail)`` for TVM 0.16 + Relay, probed in a CHILD process so a
    broken native library cannot take this interpreter down."""
    proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.startswith("0.16"):
        return True, f"tvm {proc.stdout.strip()}"
    tail = (proc.stderr.strip().splitlines() or ["?"])[-1]
    if proc.returncode == 0:
        return False, f"tvm {proc.stdout.strip()} is installed, need 0.16"
    return False, tail[-160:]


_ATTACHED = None


def attach_tvm() -> tuple:
    """Make TVM 0.16 importable without building anything, if a build exists.
    A success is cached for the process (each probe imports TVM in a child)."""
    global _ATTACHED
    if _ATTACHED:
        return _ATTACHED
    ok, detail = tvm_status()
    if ok:
        _ATTACHED = (ok, detail)
        return ok, detail
    py = _tvm_build_python()
    if not (py.parent / "build" / "libtvm.so").is_file():
        return False, f"{detail}; no TVM 0.16 build in {py.parent}"
    if str(py) not in sys.path:
        sys.path.insert(0, str(py))
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if str(py) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(py), *parts])
    ok, detail = tvm_status()
    if ok:
        _ATTACHED = (ok, f"{detail} (attached {py.parent.name}/build)")
        return _ATTACHED
    return ok, detail


def frontend(module: str, auto_install: bool = True):
    """Import ``frontend.tvmrelay.<module>`` after attaching TVM.

    Importing anything under ``frontend.tvmrelay`` runs its package
    ``__init__``, which PROVISIONS TVM 0.16 if it is missing. With
    ``auto_install=False`` (``doctor``) that is switched off for the import.
    """
    repo_root()
    attach_tvm()
    old = os.environ.get("AIEHLC_TVM_AUTO_INSTALL")
    if not auto_install:
        os.environ["AIEHLC_TVM_AUTO_INSTALL"] = "0"
    try:
        return importlib.import_module(f"frontend.tvmrelay.{module}" if module else
                                       "frontend.tvmrelay")
    finally:
        if not auto_install:
            if old is None:
                os.environ.pop("AIEHLC_TVM_AUTO_INSTALL", None)
            else:
                os.environ["AIEHLC_TVM_AUTO_INSTALL"] = old


def aiehlc_binary():
    """``build/aiehlc`` if built from source, else ``release/prebuild/aiehlc``
    -- the same order ``script/aiehlc.sh`` uses."""
    root = repo_root()
    for p in (root / "build" / "aiehlc", root / "release" / "prebuild" / "aiehlc"):
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  doctor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    needed_for: str = ""


@dataclass
class Doctor:
    checks: list = field(default_factory=list)

    def ok_for(self, stage: str) -> bool:
        """``convert`` / ``check`` / ``offload`` -- every check that stage needs passes."""
        return all(c.ok for c in self.checks if stage in c.needed_for.split(","))

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.needed_for)

    def __repr__(self) -> str:
        width = max(len(c.name) for c in self.checks)
        lines = ["aiegraph doctor"]
        for c in self.checks:
            mark = "ok  " if c.ok else ("--  " if not c.needed_for else "FAIL")
            lines.append(f"  {mark} {c.name:<{width}}  {c.detail}")
            if not c.ok and c.fix:
                lines.append(f"       {'':<{width}}  fix: {c.fix}")
        lines.append("  ready for: " + (", ".join(
            s for s in ("convert", "check", "offload") if self.ok_for(s)) or "nothing yet"))
        if not self.ok_for("convert"):
            lines.append("  next: `aiegraph setup` (wires TVM 0.16; builds it only if absent)")
        return "\n".join(lines)


def _module(name: str):
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "installed")
    except Exception as exc:                     # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:120]


def doctor() -> Doctor:
    """What works on this box, what does not, and the exact fix. Changes nothing."""
    d = Doctor()
    try:
        root = repo_root()
        d.checks.append(Check("aiehlc checkout", True, str(root), needed_for="convert,check,offload"))
    except AiegraphError as exc:
        d.checks.append(Check("aiehlc checkout", False, exc.reason, exc.hint,
                              "convert,check,offload"))
        return d
    pip = f"{sys.executable} -m pip install -e {root}"
    for mod, need in (("numpy", "convert"), ("onnx", "convert"), ("onnxruntime", "convert"),
                      ("PIL", "convert"), ("torch", "convert")):
        ok, ver = _module(mod)
        d.checks.append(Check(mod, ok, ver, pip, need))
    ok, detail = attach_tvm()
    d.checks.append(Check("TVM 0.16 (Relay)", ok, detail, "aiegraph setup", "convert"))
    ok, ver = _module("torchvision")
    d.checks.append(Check("torchvision (optional)", ok, ver,
                          f"{sys.executable} -m pip install -e '{root}[torch]'"))
    gcc = shutil.which(os.environ.get("AIEK_HOST_CC", "gcc"))
    d.checks.append(Check("host gcc", bool(gcc), gcc or "not on PATH",
                          "install gcc (x86 calibration/emulation compile with it)", "check"))
    binary = aiehlc_binary()
    d.checks.append(Check("aiehlc binary", bool(binary), str(binary or "none"),
                          "build it (README Build) or restore release/prebuild/aiehlc", "offload"))
    chess = shutil.which("xchesscc")
    vitis = os.environ.get("XILINX_VITIS")
    d.checks.append(Check("Vitis (xchesscc)", bool(vitis and chess),
                          f"XILINX_VITIS={vitis or '-'}, xchesscc={chess or '-'}",
                          "install Vitis, then `source <Vitis>/settings64.sh` "
                          "(not pip-installable: licensed)", "offload"))
    try:
        ab = frontend("arm_build", auto_install=False)
        ok, detail = ab.toolchain_status(root)
    except Exception as exc:                     # noqa: BLE001
        ok, detail = False, f"{type(exc).__name__}: {exc}"[:120]
    d.checks.append(Check("aarch64 cross toolchain", ok, detail,
                          "`source script/setup.sh --path-set-only` after Vitis", "offload"))
    return d


def setup(verbose: bool = True) -> Doctor:
    """Make ``convert`` work: attach TVM 0.16, else build it (10-25 min), then doctor."""
    ok, detail = attach_tvm()
    if not ok:
        root = repo_root()
        if (root / "thirdparty" / "tvm-0.16" / "build" / "libtvm.so").is_file():
            # The build exists but does not import: its Python deps are missing.
            setup016 = frontend("setup_tvm016", auto_install=False)
            subprocess.run([sys.executable, "-m", "pip", "install", *setup016.CORE_DEPS],
                           check=False)
        else:
            if verbose:
                print("[aiegraph] no TVM 0.16 build -- building it now (10-25 minutes)")
            setup016 = frontend("setup_tvm016", auto_install=False)
            setup016.provision(assume_yes=True)
        ok, detail = attach_tvm()
        if verbose:
            print(f"[aiegraph] TVM: {'ok' if ok else 'FAILED'} -- {detail}")
    d = doctor()
    if verbose:
        print(d)
    return d
