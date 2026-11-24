###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""L segment (launch overhead) cost model, M1.1 / M1.2.1.

L = kload + core_reset + coreen + rt_bdcfg + rt_startio, per launch, ms.
host_* glue is N (M2); rt_wait_io is K.  Measured side comes from the board log lines
``LAUNCH split`` / ``LAUNCH split_detail calls`` / ``LAUNCH split_words`` (see launch_split.py).

Features (per launch, P0 choice): bdcfg calls, startio calls, bdcfg_w (words written by
BD config), cp_w (control-packet words), plus one constant per control-path config
(off = MMIO, off-cold = MMIO cold start, aot = control packets, aot-kco = KCO).  kload count is
1 per launch on every point, so its fixed cost is carried by the constant term.
M1.2.1 adds kload_w (split_words: words written by kload; a reload of program memory, e.g. UNet
switching kernels on the same tiles, writes ~20x the warm words) and rst = core_reset calls per
launch (split_detail).  split_words' rst_n is not used: it is 33 on every off point.

predict_l(calls, words, cfg) -> ms per launch, with parameters from
src/costmodel/params/l_palmyra213.json (fitted by src/costmodel/tools/l_fit.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .launch_split import lines, parse

CFGS = ("off", "off-cold", "aot", "aot-kco")
FEATURES = ("const", "bdcfg_calls", "startio_calls", "bdcfg_w", "cp_w", "kload_w", "rst")
PARAMS = Path(__file__).resolve().parent / "params" / "l_palmyra213.json"

_WORDS = re.compile(r"LAUNCH split_words (.*)$")
_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+(?:\.[0-9]+)?)")


def _words(path: str) -> Dict[str, float]:
    last: Dict[str, float] = {}
    for line in lines(path):
        m = _WORDS.search(line)
        if m:
            last = {k: float(v) for k, v in _KV.findall(m.group(1))}
    return last


def point(path: str) -> Dict[str, Any]:
    """One board log -> per-launch features + L_meas (ms/launch)."""
    p = parse(path)
    sp, det = p["split"], p["split_detail"]
    calls, rt = det["calls"], det["rt"]
    # launches= can be missing when the board line is cut; one kload per launch on every run.
    n = float(det["other"].get("launches") or calls["kload"])
    w = _words(path)
    l_meas = (sp["kload"] + sp.get("core_reset", 0.0) + sp["coreen"]
              + rt["bdcfg"] + rt["startio"])
    return {
        "calls": {"bdcfg": calls["bdcfg"] / n, "startio": calls["startio"] / n,
                  "kload": calls["kload"] / n, "core_reset": calls.get("core_reset", 0) / n},
        "words": {"bdcfg_w": w.get("bdcfg_w", 0.0), "cp_w": w.get("cp_w", 0.0),
                  "kload_w": w.get("kload_w", 0.0)},
        "l_meas": l_meas,
        "launches": int(n),
    }


def feature_row(calls: Dict[str, float], words: Dict[str, float]) -> List[float]:
    return [1.0, float(calls.get("bdcfg", 0.0)), float(calls.get("startio", 0.0)),
            float(words.get("bdcfg_w", 0.0)), float(words.get("cp_w", 0.0)),
            float(words.get("kload_w", 0.0)), float(calls.get("core_reset", 0.0))]


_CACHE: Dict[str, Any] = {}


def load_params(path: Optional[str] = None) -> Dict[str, Any]:
    key = str(path or PARAMS)
    if key not in _CACHE:
        with open(key) as f:
            _CACHE[key] = json.load(f)
    return _CACHE[key]


def predict_l(calls: Dict[str, float], words: Dict[str, float], cfg: str,
              params: Optional[Dict[str, Any]] = None) -> float:
    """Predicted L in ms per launch.  calls/words are per launch."""
    if cfg not in CFGS:
        raise ValueError(f"cfg must be one of {CFGS}, got {cfg!r}")
    prm = params if params is not None else load_params()
    coef = prm["cfg"][cfg]["coef"]
    row = feature_row(calls, words)
    return sum(coef.get(k, 0.0) * x for k, x in zip(FEATURES, row))  # pre-M1.2.1 params lack kload_w/rst
