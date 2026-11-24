"""Cost-model fitting / measurement / report tools (moved here from script/perf/).

Each tool is a stand-alone script (``python3 -I <tool>.py ...``, the way the board gates run them) and the tools
import each other by bare module name (``import n_fit``, ``import net_sites``) with this directory on sys.path.
Keep that single import style: load a tool with :func:`load` (bare name, one module instance per process) rather
than as ``aiegraph.costmodel.tools.<name>``, which would create a second, independent copy of its globals.

The old ``script/perf/<tool>.py`` paths are thin shims that run / load the file here.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent


def names():
    """sorted tool names (file stems) in this directory"""
    return sorted(p.stem for p in TOOLS_DIR.glob("*.py") if not p.stem.startswith("_"))


def path(name: str) -> Path:
    p = TOOLS_DIR / (name + ".py")
    if not p.is_file():
        raise FileNotFoundError("no cost-model tool %r in %s (known: %s)" % (name, TOOLS_DIR, ", ".join(names())))
    return p


def load(name: str):
    """import tool ``name`` by bare module name (as the tools import each other)"""
    path(name)
    d = str(TOOLS_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(name)
