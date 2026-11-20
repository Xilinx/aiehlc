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


# ---- P0b: L / N partition of the LAUNCH split (no overlap between rt_* and host_*) ----
SPLIT_LOGS = ["c16.log", "p7_yboard0.log.excerpt", "u_lin_1024x320x320.log",
              "l_offreload_u_lin_1024x320x320.log", "l_offcold_u_lin_1024x320x320.log"]


@pytest.mark.parametrize("name", SPLIT_LOGS)
def test_split_partition(name):
    """Board logs (palmyra213 / yboard0): each split column is exactly rt_* + host_* (main_launch_line.c:31-32).

    L (M1) = kload (incl. core_reset) + coreen + rt_bdcfg + rt_startio; N (M2) = host_glue_bdcfg +
    host_flush_startio + host_inval_wait_io; K = rt_wait_io. Split fields are 3-decimal, rt/host are
    4-decimal, so the identity holds to <= 0.002 ms.

    sp_call (printed as gemm_call) goes into NO split column (main_launch_line.c:29-34: c_kl/c_bd/c_si/
    c_ce/c_wi use ph[*], wio, sp_pre, sp_glue, sp_post only) and must never be added to L or N.
    It IS a superset of ph[*]: sp_call = gemm_bf16 call = sp_glue + the ph[0..3] cycles accrued in
    the call + the part of wait_io accrued inside the call (tile64_conv_host.cc:273-275), so
      kload + coreen + bdcfg + rt_startio <= gemm_call <= that + rt_wait_io.
    c16: gemm_call=2.6463 vs bdcfg+startio+coreen+kload=1.492 (lower bound 1.3456, upper 2.7139);
    u_lin and p7 sit on the upper bound (all of wait_io happens inside the call).
    """
    r = parse(DATA / name)
    s, d = r["split"], r["split_detail"]
    rt, host = d["rt"], d["host"]
    tol = 2e-3
    assert s["bdcfg"] == pytest.approx(rt["bdcfg"] + host["glue_bdcfg"], abs=tol)
    assert s["startio"] == pytest.approx(rt["startio"] + host["flush_startio"], abs=tol)
    assert s["wait_io"] == pytest.approx(rt["wait_io"] + host["inval_wait_io"], abs=tol)
    # sp_call is not one of the summed columns: the five columns alone give sum_vs_aie_wall
    five = s["kload"] + s["bdcfg"] + s["startio"] + s["coreen"] + s["wait_io"]
    assert five == pytest.approx(s["ms_per_launch"] * s["sum_vs_aie_wall"], abs=6e-3)
    call = d["other"]["gemm_call"]
    assert five + call > s["ms_per_launch"] * s["sum_vs_aie_wall"] + 1.0  # adding it would double count
    # sp_call is a superset of ph[*] (L phases + glue), bounded above by adding all of rt_wait_io
    lo = s["kload"] + s["coreen"] + s["bdcfg"] + rt["startio"]
    assert lo - tol <= call <= lo + rt["wait_io"] + tol
    assert call > s["kload"] + s["coreen"] + s["bdcfg"] + s["startio"]  # c16: 2.6463 > 1.492


def test_split_partition_c16_sp_call_numbers():
    d = parse(DATA / "c16.log")["split_detail"]
    s = parse(DATA / "c16.log")["split"]
    assert d["other"]["gemm_call"] == 2.6463
    assert s["bdcfg"] + s["startio"] + s["coreen"] + s["kload"] == pytest.approx(1.492, abs=1e-9)


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


# ---- L segment (launch overhead), M1.1 P4 ----
from aiegraph.costmodel import launch_l  # noqa: E402


def test_l_hand_one_launch():
    # const + a*bdcfg + b*startio + c*bdcfg_w + d*cp_w, hand-computed
    prm = {"cfg": {"aot": {"coef": {"const": 0.05, "bdcfg_calls": 0.002, "startio_calls": 0.001,
                                    "bdcfg_w": 0.0001, "cp_w": 0.00002}}}}
    ms = launch_l.predict_l({"bdcfg": 100, "startio": 50, "kload": 1}, {"bdcfg_w": 1000, "cp_w": 2000}, "aot", prm)
    assert ms == pytest.approx(0.05 + 0.2 + 0.05 + 0.1 + 0.04)
    with pytest.raises(ValueError):
        launch_l.predict_l({}, {}, "mmio", prm)


@pytest.mark.parametrize("log,cfg,l_hand", [
    # kload + core_reset + coreen + rt_bdcfg + rt_startio from the LAUNCH split / split_detail lines
    ("l_aotkco_u_lin_1024x320x320.log", "aot-kco", 0.024 + 0.0 + 0.013 + 0.1084 + 0.0),
    ("l_off_y_conv_1600x64x64.log", "off", 0.387 + 0.0 + 0.023 + 0.1870 + 0.0104),
])
def test_l_board_log(log, cfg, l_hand):
    p = launch_l.point(str(DATA / log))
    assert p["l_meas"] == pytest.approx(l_hand, abs=1e-9)
    assert p["calls"]["kload"] == 1.0 and p["launches"] == 8
    pred = launch_l.predict_l(p["calls"], p["words"], cfg)  # committed params/l_palmyra213.json
    assert abs(pred - p["l_meas"]) / p["l_meas"] <= 0.20


def test_l_params_committed():
    prm = launch_l.load_params()
    assert prm["board"] == "palmyra213" and set(prm["cfg"]) == set(launch_l.CFGS)
    for c in launch_l.CFGS:
        assert prm["cfg"][c]["r2"] >= 0.9 and prm["cfg"][c]["maxrel"] <= 0.20
        assert set(prm["cfg"][c]["coef"]) == set(launch_l.FEATURES)


# ---- L segment M1.2.1: kload_w / rst features, off-cold cfg, interleaved console lines ----
def test_l_kload_w_rst_hand():
    prm = {"cfg": {"off": {"coef": {"const": 0.3, "bdcfg_calls": 0.002, "startio_calls": 0.001, "bdcfg_w": 0.0,
                                    "cp_w": 0.0, "kload_w": 0.0002, "rst": 0.01}}}}
    ms = launch_l.predict_l({"bdcfg": 100, "startio": 50, "kload": 1, "core_reset": 33},
                            {"bdcfg_w": 576, "cp_w": 0, "kload_w": 5000}, "off", prm)
    assert ms == pytest.approx(0.3 + 0.2 + 0.05 + 1.0 + 0.33)
    assert "off-cold" in launch_l.CFGS and launch_l.FEATURES[-2:] == ("kload_w", "rst")
    # pre-M1.2.1 params (no kload_w/rst keys) still predict
    old = {"cfg": {"off": {"coef": {"const": 0.3, "bdcfg_calls": 0.0, "startio_calls": 0.0, "bdcfg_w": 0.0, "cp_w": 0.0}}}}
    assert launch_l.predict_l({}, {"kload_w": 5000}, "off", old) == pytest.approx(0.3)


def test_l_interleaved_host_line(tmp_path):
    # a host '[hh:mm:ss DEBUG] ...' message cut the board's LAUNCH lines (M1-2 off-reload/u_lin_1024x320x960.log:560)
    log = tmp_path / "x.log"
    log.write_text(
        "LAUNCH split klo[20:27:39 DEBUG] main: program_done detected, draining trailing output...\n"
        "ad=1.872 bdcfg=2.534 startio=0.280 coreen=0.023 wait_io=0.733 core_reset=0.000 ms_per_launch=5.441\n"
        "LAUNCH split_detail calls kload=16 bdcfg=7296 startio=3648 coreen=16 wait_io=2880 core_reset=528 "
        "rt_bdcfg=0.9875 rt_startio=0.0237 rt_wait_io=0.6525 launches=16\n"
        "LAUNCH split_words mmio_w=10784.0 cp_w=0.0 bd_n=456.0 rst_n=33.0 kload_w=97[16:59:44 DEBUG] main: x\n"
        "28.0 bdcfg_w=576.0 startio_w=228.0 coreen_w=16.0\n")
    p = launch_l.point(str(log))
    assert p["l_meas"] == pytest.approx(1.872 + 0.0 + 0.023 + 0.9875 + 0.0237)
    assert p["words"]["kload_w"] == 9728.0 and p["words"]["bdcfg_w"] == 576.0
    assert p["calls"]["core_reset"] == 33.0 and p["calls"]["bdcfg"] == 456.0
