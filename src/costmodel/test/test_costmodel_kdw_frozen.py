###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""M3.1.2.2 PD: frozen kdw params json -> k_dmafeat --dwpredict (no fit), predict-path audit."""
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
J = SRC / "costmodel" / "params" / "kdw_palmyra213.m3112c.json"


def _kf(name):
    spec = importlib.util.spec_from_file_location(name, SRC / "costmodel" / "tools" / "k_dmafeat.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kdw_json_shape():
    """per (cfg, cls): feats (<= 2 of DW_FEATS), w >= 0, b, fit_runs = [a]"""
    kf = _kf("k_dmafeat_kdw1")
    doc = json.load(open(J))
    assert doc["fit_runs"] == ["a"] and set(doc["params"]) == {"off", "aot-kco"}
    for by in doc["params"].values():
        assert set(by) == set(kf.DW_CLS)
        for q in by.values():
            assert q["fit_runs"] == ["a"] and len(q["feats"]) == len(q["w"]) <= kf.DW_MAXF
            assert set(q["feats"]) <= set(kf.DW_FEATS) and all(w >= 0 for w in q["w"])
    pp = kf.dw_load_params(str(J))
    assert pp["off"]["conv"]["feats"] == ("ddr_t_ms", "bd_k")


def test_dwpredict_rows_and_audit():
    kf = _kf("k_dmafeat_kdw2")
    pp = kf.dw_load_params(str(J))
    s = dict(part=7, site="conv121", kind="conv", launches=1, ddr_t=68.5381, maxch_t=15521.7, dom_bytes=2.94912e7,
             shim_ch=3.0, rw_ratio=0.98, bd_n=384.0)
    row = kf.dwp_rows("off", [s], pp["off"])[0]
    p = pp["off"]["conv"]
    assert float(row[-1]) == float("%.6g" % max(0.0, p["w"][0] * 68.5381 + p["w"][1] * 0.384 + p["b"]))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n = kf.dwpredict_audit()
    assert n == 0 and buf.getvalue().splitlines()[-1] == "KDWP AUDIT meas_at_predict=0"
