#!/usr/bin/env python3
"""Moved to src/aiegraph/costmodel/tools/report_scatter.py -- backward-compatible shim for gates/logs that call this old path."""
import runpy
import sys
from pathlib import Path

_T = Path(__file__).resolve().parents[2] / "src" / "aiegraph" / "costmodel" / "tools" / "report_scatter.py"
if __name__ == "__main__":
    sys.argv[0] = str(_T)
    runpy.run_path(str(_T), run_name="__main__")
else:  # loaded as a module (importlib / bare import): execute the moved file in this module's globals
    __file__ = str(_T)
    exec(compile(_T.read_text(), str(_T), "exec"))
