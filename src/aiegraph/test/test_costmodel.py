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


# ---- DES (segment K): src/aiegraph/costmodel/des.py ----
from aiegraph.costmodel import des  # noqa: E402

_HAND_PARAMS = {"aie_clk_mhz": 1000.0, "lock_acq_cyc": 0.0, "dma_b_per_cyc": 8.0, "dma_first_byte_cyc": 0.0,
                "bd_switch_cyc": 0.0, "ddr_rd_gbps": 8.0, "ddr_wr_gbps": 8.0}


def _inst(k=64, kch=32, ntile=1, S=1):
    return {"ch": 64, "k": k, "px": 64, "ntile": ntile, "S": S, "tkw": kch // 2, "kch": kch, "nkc": k // kch, "tr": 0}


def test_des_hand_one_launch():
    # 1 round, 2 K-chunks; a = b = 16*64*2 = 2048 B per core, mesh-unique DDR 4*(2048+2048) = 16384 B per round
    # ddr 8 B/cyc -> 1024 cyc per chunk (tile stream 4096/2/8 = 256 < 1024); compute 1000 cyc -> 500 per chunk
    # chunk0: arrive 1024, done 1524; chunk1: arrive 2048, done 2548; c drain 16*16*4/8 = 128 -> 2676 cyc
    op = {"instance": _inst(), "host": {"bd_us": 5.0}}
    d = des.predict_launch(op, dict(_HAND_PARAMS), {"core_cyc": 1000})
    assert d["wait_ms"] == pytest.approx(2676 / 1e6, rel=1e-9)
    assert d["compute_ms"] == pytest.approx(1000 / 1e6, rel=1e-9)
    assert d["aie_ms"] == pytest.approx(2676 / 1e6, rel=1e-9)
    assert d["dma_ms"] == pytest.approx(2048 / 1e6, rel=1e-9)
    assert d["lock_ms"] == 0.0 and d["bound"] == "ddr" and d["rounds"] == 1
    # compute-bound variant: 8000 cyc -> 4000 per chunk; chunk0 done 5024, chunk1 arrives 2048 -> done 9024
    d2 = des.predict_launch(op, dict(_HAND_PARAMS), {"core_cyc": 8000})
    assert d2["wait_ms"] == pytest.approx((9024 + 128) / 1e6, rel=1e-9) and d2["bound"] == "compute"
    # calibration keys: fixed per-round latency adds 1:1, a b-prefetch removes DDR bytes from the paced part
    d3 = des.predict_launch(op, dict(_HAND_PARAMS, round_fixed_cyc=100.0), {"core_cyc": 1000})
    assert d3["wait_ms"] == pytest.approx(2776 / 1e6, rel=1e-9)
    d4 = des.predict_launch(op, dict(_HAND_PARAMS, b_prefetch_b=8192.0), {"core_cyc": 1000})
    # rest 8192 B -> 512 per chunk: chunk0 arrive 512 done 1012, chunk1 arrive 1024 start 1024 done 1524 (+128)
    assert d4["wait_ms"] == pytest.approx(1652 / 1e6, rel=1e-9)


def test_des_compute_monotonic_in_k():
    prev = None
    for k in (320, 640, 1280, 2880, 5760):
        inst = _inst(k=k, kch=32, ntile=4, S=4)
        op = {"instance": inst, "host": {"bd_us": 5.0}}
        cyc = {"core_cyc": 300 * inst["nkc"] * inst["ntile"]}
        p = dict(_HAND_PARAMS, chunk_ovh_cyc=183.0, round_ovh_cyc=6800.0)
        d = des.predict_launch(op, p, cyc)
        assert d["compute_ms"] == pytest.approx(des.compute_cycles(op, p, cyc) / 1e6)
        if prev is not None:
            assert d["compute_ms"] > prev["compute_ms"] and d["wait_ms"] > prev["wait_ms"]
        prev = d


def test_des_unet_c16_fixture():
    # host BD-issue schedule from the shared fixture: 258 launches, 11853 BD configs,
    # (rt_bdcfg 0.0960 + rt_startio 0.0006 + host_glue_bdcfg 1.1981) ms per launch; rt_wait_io is NOT an input
    r = parse(DATA / "c16.log")
    h = des.host_from_split(r["split_detail"], rounds=16)
    assert h["bd_per_launch"] == pytest.approx(11853 / 258)
    assert h["issue_ms"] == pytest.approx(0.0960 + 0.0006 + 1.1981)
    assert h["bd_us"] == pytest.approx((0.0960 + 0.0006 + 1.1981) * 1e3 / (11853 / 258))
    op = {"instance": {"ch": 320, "k": 320, "px": 1024, "ntile": 16, "S": 5, "tkw": 8, "kch": 16, "nkc": 20,
                       "tr": 1}, "host": h}
    p = dict(_HAND_PARAMS, ddr_rd_gbps=9.47, ddr_wr_gbps=8.341, aie_clk_mhz=999.98, lock_acq_cyc=17.0,
             dma_first_byte_cyc=21.0, bd_switch_cyc=1.0)
    d = des.predict_launch(op, p, {"core_cyc": 657788})
    assert set(d) >= {"compute_ms", "dma_ms", "lock_ms", "stall_ms", "wait_ms", "bound"}
    assert 0.0 < d["compute_ms"] < d["aie_ms"] and d["wait_ms"] > 0.0 and d["bound"] in ("compute", "ddr")
    # 16 rounds x 20 chunks x 2 acquires + 16 commits
    assert d["lock_ms"] == pytest.approx(16 * 41 * 17.0 / (999.98 * 1e3))


def test_des_import():
    # public entry + roofline + the K-stage fit script import offline (no board, no ISS)
    import importlib.util

    from aiegraph.costmodel import roofline as rl
    from aiegraph.costmodel.des import predict_launch

    assert callable(predict_launch) and callable(rl.roofline)
    fit = SRC.parent / "script" / "perf" / "costmodel_fit.py"
    spec = importlib.util.spec_from_file_location("costmodel_fit", fit)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main) and mod.des is des
    op = {"instance": _inst(), "host": {"bd_us": 5.0}}
    r = rl.roofline(op, dict(_HAND_PARAMS), {"core_cyc": 1000})
    assert isinstance(r, dict)


def test_des_ddr_model_select():
    # cal key ddr_model="bytes" overrides the per-burst DDR model even when the params carry the burst microbenchmark
    p = dict(_HAND_PARAMS, ddr_burst_cyc0=8.0, ddr_burst_b_per_cyc=8.0)
    assert des.burst_model(p) and not des.burst_model(dict(p, ddr_model="bytes"))
    op = {"instance": _inst(), "host": {"bd_us": 5.0}}
    d = des.predict_launch(op, dict(p, ddr_model="bytes"), {"core_cyc": 1000})
    assert d["wait_ms"] == pytest.approx(des.predict_launch(op, dict(_HAND_PARAMS), {"core_cyc": 1000})["wait_ms"])
