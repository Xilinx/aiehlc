###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""x86 compile/link/run helpers shared by ``calibrate`` and ``emulate``.

Everything here runs on the build host with the host gcc -- no Vitis, no board.
TVM's generated layer C compiles on x86 with exactly the flags ``make local``
uses (``-std=gnu11 -D__AIESIM__``; skill tvmlocalhostelf); the TVM runtime it
needs is the same four-symbol shim and header stubs ``arm_build`` writes for
the board, written here into a private directory so this works on a tree whose
``arm_build/`` was never generated (``--no-arm``).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

__all__ = ["HostBuildError", "host_dir", "compile_c", "link", "run", "REPO"]

#: Repo root: .../src/frontend/tvmrelay/aiekernel/hostbuild.py -> up four.
REPO = Path(__file__).resolve().parents[4]

CC = os.environ.get("AIEK_HOST_CC", "gcc")
CXX = os.environ.get("AIEK_HOST_CXX", "g++")
CFLAGS = ["-O2", "-std=gnu11", "-D__AIESIM__", "-Wall", "-Wno-unused-variable",
          "-Wno-unused-function", "-Wno-unused-but-set-variable"]
CXXFLAGS = ["-O2", "-std=c++17", "-D__AIESIM__"]


class HostBuildError(RuntimeError):
    """A host compile/link failed. ``str()`` ends with the compiler's last lines."""


def host_dir(out_dir) -> Path:
    """``<out_dir>/aiek/_host``: TVM header stubs + runtime shim, written once."""
    from frontend.tvmrelay import arm_build

    d = Path(out_dir) / "aiek" / "_host"
    if not (d / "tvm_runtime_shim.c").is_file():
        d.mkdir(parents=True, exist_ok=True)
        (d / "tvm_graph_types.h").write_text(arm_build._TYPES_H)
        arm_build._write_tvm_header_stubs(d)
        (d / "tvm_runtime_shim.c").write_text(arm_build._SHIM_C)
    return d


def _tail(proc) -> str:
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    errs = [ln for ln in lines if "error" in ln.lower()] or lines
    return "\n".join(errs[-8:])


def compile_c(src: Path, obj: Path, includes=(), defines=(), force: bool = False) -> Path:
    """Compile one C or C++ file (by suffix). Skipped when *obj* is newer."""
    src, obj = Path(src), Path(obj)
    if not force and obj.is_file() and obj.stat().st_mtime >= src.stat().st_mtime:
        return obj
    obj.parent.mkdir(parents=True, exist_ok=True)
    cxx = src.suffix in (".cc", ".cpp", ".cxx")
    cmd = [CXX if cxx else CC, *(CXXFLAGS if cxx else CFLAGS),
           *[f"-I{i}" for i in includes], *[f"-D{d}" for d in defines],
           "-c", str(src), "-o", str(obj)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HostBuildError(f"compile {src.name} failed:\n{_tail(proc)}")
    return obj


def link(objs, exe: Path, cxx: bool = False) -> Path:
    cmd = [CXX if cxx else CC, "-o", str(exe), *[str(o) for o in objs], "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HostBuildError(f"link {Path(exe).name} failed:\n{_tail(proc)}")
    return Path(exe)


def run(cmd, cwd=None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)
