###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Self-check for frontend.convtune (no board, no aiehlc.sh run).

    PYTHONPATH=src python src/frontend/convtune/test_convtune.py
"""

import sys
import tempfile
from pathlib import Path

from frontend.convtune import convtile

SRC = convtile.REPO / "example" / "tileprogram" / "ccode" / "simpleconv2d_new_load.cc"


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"ok   {msg}")


def main():
    sp = convtile.space(SRC)
    cands = sp["candidates"]
    check(sp["mesh_rows"] == 4 and sp["mesh_cols"] == 4, "MESH_ROW/MESH_COL resolve to the 4x4 launch mesh")
    check(len(cands) > 10, f"legal space is non-trivial ({len(cands)} tilings)")
    check(all(c["features"]["l1_bytes"] <= 49152 for c in cands), "every candidate fits the core data budget")

    d = convtile.derive(SRC, {"time_oh": 7, "time_ow": 28})
    ifm = d["ports"]["IFM"]
    check((ifm["d1"]["tile_size"], ifm["d1"]["stride"], ifm["d1"]["slice_tiling"]["tile_size"],
           ifm["d1"]["slice_tiling"]["stride"]) == (61, 56, 19, 14),
          "IFM d1 halo == hand-written RowBC_spatial (61/56, 19/14)")
    check((ifm["d2"]["tile_size"], ifm["d2"]["stride"], ifm["d2"]["tile_round"]) == (61, 56, 4),
          "IFM d2 == hand-written width halo (61/56 x4)")
    check(d["features"]["l1_bytes"] == 21816, "L1 == pipeline 'Buffer memory check' (21816)")

    try:
        convtile.derive(SRC, {"time_oh": 5})
        check(False, "tile(OH,5) must be rejected")
    except convtile.ConvTileError as exc:
        check("divide" in str(exc), f"tile(OH,5) rejected: {exc}")

    from frontend.convtune import autotvm_bridge as br

    task, legal = br.create_task(SRC)
    check(len(task.config_space) == len(legal), "AutoTVM ConfigSpace == legal set (multi_filter)")
    with tempfile.TemporaryDirectory() as tmp:
        s = br.tune(SRC, n_trial=24, mode="analytic", tuner="xgb", workdir=tmp, seed=0)
        best_model = min(c["features"]["est_ms"] for c in cands)
        check(s["failed"] == 0, "no failed trials in analytic mode")
        check(s["best"]["cost_ms"] <= best_model * 1.01,
              f"XGBTuner best {s['best']['cost_ms']:.3f} ms within 1% of the space optimum {best_model:.3f} ms")
        check(Path(tmp, "best_knobs.json").is_file() and Path(tmp, "tune.log").is_file(),
              "best_knobs.json + AutoTVM log written")
    print("convtune: all checks passed")


if __name__ == "__main__":
    main()
