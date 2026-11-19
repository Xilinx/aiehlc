"""Tests for src/aiegraph/costmodel (AIEHLC local cost layer, AIE side)."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph.costmodel.launch_split import parse  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "costmodel"


def test_split_unet_c16():
    r = parse(DATA / "c16.log")
    s = r["split"]
    assert r["ms_per_launch"] == pytest.approx(2.938, abs=1e-12)
    assert r["pmu_hz"] == 1316653928
    assert (s["kload"], s["bdcfg"], s["startio"], s["coreen"], s["wait_io"], s["core_reset"]) == \
        (0.040, 1.294, 0.147, 0.011, 1.392, 0.000)
    d = r["split_detail"]
    assert d["calls"] == {"kload": 258, "bdcfg": 11853, "startio": 1584, "coreen": 258, "wait_io": 34212,
                          "core_reset": 0}
    assert d["rt"] == {"bdcfg": 0.0960, "startio": 0.0006, "wait_io": 1.3683}
    assert d["host"] == {"glue_bdcfg": 1.1981, "flush_startio": 0.1468, "inval_wait_io": 0.0234}
    assert d["other"]["aie_wall_ms"] == 758.053 and d["other"]["pmc_delta"] == 1719306034
    # wait_io in split = rt_wait_io + host_inval_wait_io (rounding of the 4-decimal fields)
    assert s["wait_io"] == pytest.approx(d["rt"]["wait_io"] + d["host"]["inval_wait_io"], abs=2e-3)
    p14 = next(p for p in r["prof"] if p["part"] == 14)
    assert p14["op"] == "aie_host_conv0" and p14["site_ms"] == 12.787 and p14["aie_ms"] == 9.402
    hs = r["prof_host_split"][0]
    assert hs == {"part": 14, "kind": "conv", "pack_ms": 2.033, "scatter_ms": 1.351, "dma_ms": 2.392,
                  "wait_ms": 7.010}
    assert r["perf"] == []


def test_split_yolo_p7():
    r = parse(DATA / "p7_yboard0.log.excerpt")
    assert r["ms_per_launch"] == pytest.approx(1.202, abs=1e-12)
    assert r["split"]["core_reset"] == 0.361
    d = r["split_detail"]
    assert d["rt"]["wait_io"] == 0.0564 and d["host"]["glue_bdcfg"] == 0.3300
    assert d["calls"]["core_reset"] == 58409
    assert "rt_wait_io" not in d["other"] and "host_glue_bdcfg" not in d["other"]
    p1 = r["prof"][1]
    assert (p1["part"], p1["op"], p1["ms"], p1["site_ms"], p1["aie_ms"]) == \
        (1, "aie_host_conv0", 423.624, 834.035, 410.411)


def test_split_perf_counters_board_c2():
    r = parse(DATA / "board_c2.log")
    assert r["split"] is None and r["ms_per_launch"] is None
    perf = r["perf"]
    assert len(perf) >= 2 and [p["tag"] for p in perf[:2]] == [0, 1]
    p0 = perf[0]
    assert p0["tiles"] == 16 and p0["rc_bad"] == 0 and len(p0["io"]) == 16
    assert p0["active_max"] == 619514
    # computed aggregate equals the board's own [PERFSUM] line
    for p in perf:
        ps = p["perfsum"]
        assert ps is not None
        assert (p["tiles"], p["active_min"], p["active_max"], p["active_mean"]) == \
            (ps["tiles"], ps["min"], ps["max"], ps["mean"])
    t36 = next(t for t in p0["per_tile"] if (t["col"], t["row"]) == (3, 6))
    assert (t36["active"], t36["lock_stall"], t36["stream_stall"]) == (619514, 587999, 0)
    assert t36["compute"] == 619514 - 587999
    assert p0["compute_max"] == max(t["active"] - t["lock_stall"] - t["stream_stall"] for t in p0["per_tile"])
    io03 = p0["io"][0]
    assert io03 == {"col": 0, "row": 3, "pre_io_active": 97975, "pre_io_lock_stall": 97206,
                    "io_active": 215594, "io_lock_stall": 185275}
    t03 = p0["per_tile"][0]
    assert io03["pre_io_active"] + io03["io_active"] == t03["active"]
