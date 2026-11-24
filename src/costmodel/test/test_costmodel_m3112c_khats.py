###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""M3.1.1.2c PK: k_diag.py --khat-static (all-static K_hat: core + dw - ovl, fitted on run a only)."""
import csv
import importlib.util
import math
from pathlib import Path

PERF = Path(__file__).resolve().parents[1] / "tools"


def _kd():
    spec = importlib.util.spec_from_file_location("k_diag_m3112c_khats", PERF / "k_diag.py")
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    return kd


def _w(path, cols, rows):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def test_khat_static_same_on_all_runs(tmp_path, monkeypatch):
    """K_hat_ms is bitwise identical on runs a/b/c (static), K_loo only on run a, measured fields never reach it"""
    kd = _kd()
    sites = [(10 + i, "linear%d" % i, "linear", 1 + i % 3) for i in range(8)]
    _w(tmp_path / "core.tsv", kd.KCORE_COLS,
       [("s", p, n, c, L, "%.6f" % (1.0 + 0.3 * i), "iss") for i, (p, n, c, L) in enumerate(sites)])
    _w(tmp_path / "dw.tsv", ("cfg", "part", "site", "cls", "launches", "ddr_t_ms", "dw_hat_ms"),
       [("s", p, n, c, L, 0.1, 0.2 + 0.05 * i) for i, (p, n, c, L) in enumerate(sites)])
    _w(tmp_path / "np.tsv", ("cfg", "run", "part", "site", "launches", "site_ms", "N_meas", "N_hat", "ovl_meas",
                             "ovl_hat"),
       [("s", r, p, n, L, 9.0, 1.0, 1.0, 0.1, 0.1 * (i % 3)) for r in "abc" for i, (p, n, c, L) in enumerate(sites)])

    # run-a join stub: K_meas differs per site (measured), nothing else of the log is used
    class KL:
        @staticmethod
        def norm_run(x):
            return x if len(x) == 1 else "a"

        @staticmethod
        def parse_logs(specs):
            return [("s", "/x/unet_s.a.log")]

        @staticmethod
        def k_meas(r, src):
            return r["Km"]

    rows = [dict(cfg="s", run="a", part=p, site=n, launches=L, site_ms_log=10.0 * L, Km=(1.2 + 0.4 * i) * L)
            for i, (p, n, c, L) in enumerate(sites)]
    monkeypatch.setattr(kd, "khats_fit_rows", lambda a: (rows, True, KL))

    class A:
        core_tsv, dwhat, npred = str(tmp_path / "core.tsv"), str(tmp_path / "dw.tsv"), str(tmp_path / "np.tsv")
        khat_tsv, out_params, fit = str(tmp_path / "khat_static.tsv"), str(tmp_path / "k.json"), ["s=x"]

    assert kd.khat_static(A) == 0
    with open(A.khat_tsv) as f:
        t = list(csv.DictReader(f, delimiter="\t"))
    assert list(t[0].keys()) == list(kd.KHATS_COLS) and len(t) == 3 * len(sites)
    by = {}
    for r in t:
        by.setdefault((r["cfg"], r["part"], r["site"]), {})[r["run"]] = r
    for k, v in by.items():
        assert set(v) == {"a", "b", "c"}
        assert v["a"]["K_hat_ms"] == v["b"]["K_hat_ms"] == v["c"]["K_hat_ms"]   # K_hat the same on all three runs
        assert math.isfinite(float(v["a"]["K_loo_ms"])) and v["b"]["K_loo_ms"] == v["c"]["K_loo_ms"] == "nan"
        assert {v["a"][c] for c in ("core_src", "dw_src", "ovl_src")} == {"static"}
    assert kd.khats_audit()
