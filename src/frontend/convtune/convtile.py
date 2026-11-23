###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Thin wrapper over the ``aiehlc-convtile`` CLI (the C++ conv2d auto-tiling model).

The C++ model (``src/llvm/convautotile``) is the single source of truth for
legality and for the derived d1/d2/d3 + halo; Python never re-implements it.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_SRC = REPO / "src" / "llvm" / "convautotile"


class ConvTileError(RuntimeError):
    """The model rejected a configuration (or the CLI failed)."""


def _newest_source_mtime():
    return max(p.stat().st_mtime for p in _SRC.glob("*.[ch]*"))


def find_cli():
    """Locate aiehlc-convtile: $AIEHLC_CONVTILE, build/, else compile a cached copy."""
    env = os.environ.get("AIEHLC_CONVTILE")
    if env:
        return env
    built = REPO / "build" / "aiehlc-convtile"
    if built.is_file() and built.stat().st_mtime >= _newest_source_mtime():
        return str(built)
    # The model is plain C++17 with no dependencies: build a private copy rather
    # than require a full LLVM/MLIR rebuild of the repo.
    cache = Path(tempfile.gettempdir()) / f"aiehlc-convtile-{os.getuid()}"
    exe = cache / "aiehlc-convtile"
    if exe.is_file() and exe.stat().st_mtime >= _newest_source_mtime():
        return str(exe)
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise ConvTileError("aiehlc-convtile not built and no C++ compiler found; run `make aiehlc-convtile` in build/")
    cache.mkdir(parents=True, exist_ok=True)
    srcs = [str(_SRC / f) for f in ("convtile_cli.cpp", "conv_autotile.cpp", "conv_desugar.cpp")]
    subprocess.run([cxx, "-std=c++17", "-O2", "-o", str(exe)] + srcs, check=True)
    return str(exe)


def _run(args, overrides=None, model_args=()):
    cmd = [find_cli()] + list(args) + list(model_args)
    if overrides:
        cmd += ["--overrides", json.dumps(overrides)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 1:
        raise ConvTileError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")
    return proc.returncode, proc.stdout, proc.stderr


def space(src, pp_depth=None, model_args=(), l1_budget=None):
    """All legal candidates for ``src`` (ranked by the analytical model)."""
    ov = {k: v for k, v in (("pp_depth", pp_depth), ("l1_budget", l1_budget)) if v} or None
    rc, out, err = _run(["space", str(src)], ov, model_args)
    if rc != 0:
        raise ConvTileError(err.strip() or "no legal candidates")
    return json.loads(out)


def derive(src, knobs=None, model_args=()):
    """Derived d1/d2/d3, halo, features for ``src`` with tuner ``knobs`` applied.

    Returns the per-kernel ``derived`` dict; raises ConvTileError when illegal.
    """
    rc, out, err = _run(["derive", str(src)], knobs, model_args)
    report = json.loads(out) if out.strip().startswith("{") else None
    if rc != 0 or not report or not report.get("ok"):
        msg = (report or {}).get("error") or err.strip() or "rejected"
        raise ConvTileError(msg)
    return report["kernels"][0]["derived"]


def desugar(src, knobs=None, model_args=()):
    """The rewritten source aiehlc would compile (explicit GemmSpace form)."""
    rc, out, err = _run(["desugar", str(src)], knobs, model_args)
    if rc != 0:
        raise ConvTileError(err.strip() or out.strip())
    return out
