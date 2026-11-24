#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Moved to src/costmodel/tools/l_static.py -- backward-compatible shim for gates/logs that call this old path."""
import runpy
import sys
from pathlib import Path

_T = Path(__file__).resolve().parents[2] / "src" / "costmodel" / "tools" / "l_static.py"
if __name__ == "__main__":
    sys.argv[0] = str(_T)
    runpy.run_path(str(_T), run_name="__main__")
else:  # loaded as a module (importlib / bare import): execute the moved file in this module's globals
    __file__ = str(_T)
    exec(compile(_T.read_text(), str(_T), "exec"))
