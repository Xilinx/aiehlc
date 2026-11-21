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


# ---- L segment M1.2.2 P4: L prediction from static (build-directory) features only ----
PERF = SRC.parent / "script" / "perf"
KBUILD = DATA / "build_off_u_lin_1024x320x320"  # trimmed k_off/u_lin_1024x320x320 build, see its README
KLOG = DATA / "l_off_u_lin_1024x320x320.log"    # board log of the same build (M1-L/kernels/off)


def _script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, PERF / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_static_feat_kernel_u_lin_1024x320x320():
    from aiegraph.costmodel import launch_l
    ls = _script("l_static")
    calls, words = ls.static_calls_words(KBUILD)
    got = ls.flat(calls, words)
    # host.cc: 480 bd_config + 240 startio per launch; 16 core tiles x (8 DMA reset + 3 core ctrl + 17 DM words), PM warm
    assert got == {"bdcfg_calls": 480.0, "startio_calls": 240.0, "bdcfg_w": 576.0, "cp_w": 0.0,
                   "kload_w": 448.0, "rst": 0.0}
    p = launch_l.point(str(KLOG))
    meas = ls.flat(p["calls"], p["words"])
    from aiegraph.costmodel.static_feat import rel_of
    assert max(rel_of(got[k], meas[k], k.endswith("_w")) for k in got) == 0.0


def test_l_static_ignores_log_counts(tmp_path):
    import re
    from aiegraph.costmodel import launch_l
    lf = _script("l_fit")
    prm = launch_l.load_params()
    txt = KLOG.read_text()

    def x10(m):
        return re.sub(r"=(\d+(?:\.\d+)?)", lambda v: "=%g" % (float(v.group(1)) * 10), m.group(0))
    hacked = re.sub(r"LAUNCH split_words .*", x10, txt)
    hacked = re.sub(r"(LAUNCH split_detail calls )(\w+=\d+ ?)+", x10, hacked)
    assert hacked != txt and "kload_w=4480 " in hacked and "bdcfg=38400 " in hacked
    bad = tmp_path / "x10.log"
    bad.write_text(hacked)
    # log-feature pred does move with the counts ...
    p0, p1 = launch_l.point(str(KLOG)), launch_l.point(str(bad))
    assert launch_l.predict_l(p1["calls"], p1["words"], "off", prm) != pytest.approx(
        launch_l.predict_l(p0["calls"], p0["words"], "off", prm))
    # ... the static pred reads the build only: same value, equal to the log-feature pred of the real log
    s = lf.pred_static(KBUILD, "off", prm)
    assert s == lf.pred_static(KBUILD, "off", prm)
    assert s == pytest.approx(launch_l.predict_l(p0["calls"], p0["words"], "off", prm), abs=1e-12)


# N (host glue) segment, costmodel M2/N-P2: frozen params vs the M2c b board run (palmyra213, off + aot-kco).
# Excerpts = PROF / [UCL] / LAUNCH (warm + split*) / CMP / BOARD lines of
# shuhua/clones/cost/tasks/2026-11-20_M2c/unet_{off,aot-kco}.b.log ('LAUNCH warm' carries launches= for NSUM).
# The site features come from the UNet build dirs (~10 MB of host.cc / graph_driver.c / aiek_unetconv.cc read
# out of 31 GB), not committed: N_BUILD_ROOT overrides the default M2c build root.
NPARAMS = PERF / "params" / "n_unet_palmyra213.json"
NBUILD = Path(__import__("os").environ.get("N_BUILD_ROOT", "/scratch/staff/huaj/tmp/costmodel_m2c"))


def test_n_fit_frozen_unet(capsys):
    if not (NBUILD / "unet_off").is_dir() or not (NBUILD / "unet_aot-kco").is_dir():
        pytest.skip("M2c UNet build root %s missing (set N_BUILD_ROOT)" % NBUILD)
    nf = _script("n_fit")
    rc = nf.main(["--build-root", str(NBUILD), "--params", str(NPARAMS), "--check",
                  "off=%s" % (DATA / "n_unet_off.b.excerpt"), "aot-kco=%s" % (DATA / "n_unet_aot-kco.b.excerpt")])
    out = capsys.readouterr().out
    print(out)
    assert rc == 0
    assert "NGATE ALL src=board cfgs=2" in out
    import re
    m = re.search(r"^NGATE ALL src=board cfgs=2 sites=(\d+) r2_min=\S+ maxrel=([\d.]+) (PASS|FAIL)$", out, re.M)
    assert m, out
    assert int(m.group(1)) == 276            # 138 launch sites per cfg
    assert float(m.group(2)) <= 0.2
    assert m.group(3) == "PASS"
    for cfg in ("off", "aot-kco"):
        assert re.search(r"^NSRC cfg=%s .*board=palmyra213 launches=258 run_rc=0$" % cfg, out, re.M)
        assert re.search(r"^NSUM cfg=%s .* PASS$" % cfg, out, re.M)
        assert re.search(r"^NGATE cfg=%s sites=138 .* PASS$" % cfg, out, re.M)
    assert "NSITE_BAD" not in out


def test_n_feat_pack_slow_m77():
    from aiegraph.costmodel.n_feat import pack_slow_elems
    K = 768
    # M=77: Mp=(77+63)&~63=128, NEON path takes 8*floor(77/8)=72 rows, scalar slow path 128-72=56 rows x K
    assert ((77 + 63) & ~63) == 128 and 8 * (77 // 8) == 72
    assert pack_slow_elems(77, K) == 56 * K
    assert pack_slow_elems(77, 1) == 56
    # M=64: Mp=64, all rows in full 8-row groups -> no slow path
    assert pack_slow_elems(64, K) == 0


def test_k_diag_selftest():
    """k_diag.py (M3.1 K anchor): split, bound, class, bucket and the K-floor fit on synthetic known-answer data"""
    import re
    import subprocess
    k_diag = SRC.parent / "script" / "perf" / "k_diag.py"
    r = subprocess.run([sys.executable, "-I", str(k_diag), "--selftest"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    m = re.search(r"^KDIAG SELFTEST PASS \((\d+)/(\d+)\)$", r.stdout, re.M)
    assert m and m.group(1) == m.group(2) and int(m.group(1)) >= 32, r.stdout
    for name in ("ovl_parse", "ovl_model_sum", "ovl_model_max", "ovl_missing_rc2", "spreadx", "spreadx_diff",
                 "dma7_recover", "kcap_violation"):
        assert re.search(r"^KDIAG SELFTEST %s\s+ok" % name, r.stdout, re.M), name


def test_k_diag_spreadx(capsys):
    """k_diag.py SPREADX (M3.1.2.2 PR): own per-site spread vs a spread_abc.tsv; exact -> PASS,
    0.002 ms off / missing / extra site -> FAIL"""
    import importlib.util
    path = SRC.parent / "script" / "perf" / "k_diag.py"
    spec = importlib.util.spec_from_file_location("k_diag_spreadx", path)
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    runs = [[dict(site="fa%d" % i, part=i, kind="fa", launches=1, K=0.5 + d) for i in (3, 5)] for d in (0, .01, .02)]
    mine = kd.spread_sites("c", runs)
    assert set(mine) == {("c", 3), ("c", 5)} and abs(mine[("c", 3)] - 0.02) < 1e-12
    assert kd.spreadx_line(mine, dict(mine), want=2)
    assert not kd.spreadx_line(mine, {k: v + 0.002 for k, v in mine.items()}, want=2)
    assert not kd.spreadx_line(mine, {("c", 3): mine[("c", 3)]}, want=2)
    assert not kd.spreadx_line(mine, {**mine, ("c", 9): 0.0}, want=2)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("SPREADX sites=2 maxdiff_ms=0.000000 PASS") and " FAIL " in out[1]


def test_k_noise_spread(tmp_path):
    """k_noise.py --spread-out (M3.1.2.2 PS): per-site a/b/c spread table, KSPREAD before GATED ALL (still last),
    n_runs!=3 -> FAIL rc=1; plus the built-in selftest"""
    import re
    import subprocess
    kn = SRC.parent / "script" / "perf" / "k_noise.py"
    r = subprocess.run([sys.executable, "-I", str(kn), "--selftest"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and re.search(r"^KNOISE SELFTEST PASS \((\d+)/\1\)$", r.stdout, re.M), r.stdout + r.stderr
    cmp = "CMP unet board=palmyra213 a=1 aie_launches=9 b=2 run_rc=0 c=3 PASS\n"
    logs = []
    for i, (w1, w2) in enumerate(((2.0, 0.30), (2.2, 0.31), (1.9, 0.29))):
        f = tmp_path / ("r%d.log" % i)
        f.write_text("PROF aie_host_split part=5 kind=conv q=1 wait_ms=%.3f q=1 inval_ms=0.100 launches=2\n"
                     "PROF aie_host_split part=7 kind=fa q=1 wait_ms=%.3f q=1 inval_ms=0.000 launches=1\n"
                     "PROF aie_host_split part=8 kind=fa q=1 wait_ms=1.000 q=1 inval_ms=0.000 launches=0\n" % (w1, w2)
                     + cmp)
        logs.append(str(f))
    sp = tmp_path / "sp.tsv"
    r = subprocess.run([sys.executable, "-I", str(kn), "--spread-out", str(sp), "c=" + ",".join(logs)],
                       capture_output=True, text=True, timeout=120)
    out = r.stdout.strip().splitlines()
    assert r.returncode == 0 and out[-2] == "KSPREAD ALL cfgs=1 sites=2 runs=3 PASS", r.stdout + r.stderr
    assert out[-1].startswith("KNOISE GATED ALL")
    rows = [l.split() for l in sp.read_text().splitlines()]
    assert rows[0][0] == "#" and len(rows) == 3 and all(len(x) == 9 for x in rows[1:])
    assert rows[1][:5] == ["c", "5", "conv", "2", "3"] and abs(float(rows[1][8]) - 0.3) < 1e-9
    assert abs(float(rows[2][8]) - 0.02) < 1e-9
    r = subprocess.run([sys.executable, "-I", str(kn), "--spread-out", str(sp), "c=" + ",".join(logs[:2])],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "KSPREAD ALL cfgs=1 sites=2 runs=2 FAIL" in r.stdout
    assert r.stdout.strip().splitlines()[-1].startswith("KNOISE GATED ALL")


def test_k_diag_noscale():
    """k_diag.py (M3.1.1 counter window): perf_n < launches is short (never scaled, out of KSANITY);
    optional pre_io_act_cyc/snap_gap_cyc tail; KSANITY denominator is K_meas + snap_gap"""
    import importlib.util
    path = SRC.parent / "script" / "perf" / "k_diag.py"
    spec = importlib.util.spec_from_file_location("k_diag_noscale", path)
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    txt = ("PROF aie_site_perf part=1 kind=conv launches=4 perf_n=1 io_act_cyc=2000000 lock_stall_cyc=10 "
           "stream_stall_cyc=20\n"
           "PROF aie_site_perf part=2 kind=fa launches=3 perf_n=3 io_act_cyc=3000000 lock_stall_cyc=0 "
           "stream_stall_cyc=0 pre_io_act_cyc=77 snap_gap_cyc=500000\n")
    pf = kd.parse_perf(txt)
    assert pf[1]["io_act"] == 2000000 and pf[1]["lock"] == 10 and pf[1]["stream"] == 20  # not x4
    assert pf[1]["short"] and pf[1]["gap"] == 0
    assert not pf[2]["short"] and pf[2]["gap"] == 500000 and pf[2]["pre_io_act"] == 77
    assert "scale" not in path.read_text().split("def parse_perf", 1)[1].split("def ", 1)[0].replace("scaled", "")
    sites = []
    for prt, K in ((1, 0.5), (2, 2.6)):
        s = dict(site="x%d" % prt, part=prt, launches=pf[prt]["launches"], cls="linear", ddr_b=0.0, bucket=2,
                 perf=pf[prt])
        s.update(kd.split_site(K, 0.0, pf[prt]))
        sites.append(s)
    iok = kd.diag_lines("t", "b", sites)
    # short part (io 2 ms over K 0.5 ms) excluded; part 2: 3 ms / (2.6 + 0.5 gap) ms
    assert abs(iok - 3.0 / 3.1) < 1e-12 and iok <= kd.IOK_MAX
    assert kd.kperf_line("t", "b", pf) == (2, 1)


def test_k_diag_sanity_only():
    """k_diag.py --sanity-only (M3.1.2 PK): runs a/b/c on old-format logs (no ovl fields), KSANITY per log,
    one KDRIFT (max over runs) per cfg, KSANITY FAIL -> rc=1, no CMP line -> rc=2; no fit"""
    import contextlib
    import importlib.util
    import io
    path = SRC.parent / "script" / "perf" / "k_diag.py"
    spec = importlib.util.spec_from_file_location("k_diag_sanity", path)
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    old = ("PROF aie_site_perf part=1 kind=linear launches=2 perf_n=2 io_act_cyc=2000000 lock_stall_cyc=0 "
           "stream_stall_cyc=0\n")
    kms = {"a": 1.0, "b": 2.0, "c": 4.0}   # core = 2 ms -> io/K = 2.0 FAIL / 1.0 / 0.5

    def load(cfg, log, build_root, need_perf=True):
        pf = kd.parse_perf(old)
        s = dict(cfg=cfg, site="linear1", part=1, kind="linear", launches=2, cls="linear", bd_ok=True, run_b=32,
                 ddr_b=1e6, bd_n=1, num_dims=1, perf=pf[1], bucket=0)
        s.update(kd.split_site(kms[log], 0.0, pf[1]))
        cmp_ = "" if cfg == "nocmp" else "CMP unet board=palmyra213 aie_launches=258 run_rc=0\n"
        return [s], cmp_ + old
    kd.load = load
    kd.ref_sites = lambda a, cfg, log: [dict(K=2.0)]   # drifts 0.5 / 0 / 1.0

    def run(argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = kd.main(argv)
        return rc, buf.getvalue().splitlines()
    rc, out = run(["--sanity-only", "--build-root", "/x", "off=a,b,c", "--ref", "off=r"])
    assert rc == 1
    assert [x for x in out if x.startswith("KSANITY")] == [
        "KSANITY cfg=off run=a io_act_over_K_max=2.000 FAIL", "KSANITY cfg=off run=b io_act_over_K_max=1.000 PASS",
        "KSANITY cfg=off run=c io_act_over_K_max=0.500 PASS"]
    assert [x for x in out if x.startswith("KDRIFT")] == ["KDRIFT cfg=off runs=3 wait_sum_rel_vs_ref_max=1.000"]
    assert sum(x.startswith("KDSRC cfg=off log=") and "board=palmyra213 launches=258 run_rc=0 sites=1 perf_sites=1"
               in x for x in out) == 3
    assert not any(x.startswith(("KFIT", "KMODEL", "KDIAG ERROR")) for x in out)
    rc, out = run(["--sanity-only", "--build-root", "/x", "off=b,c"])
    assert rc == 0 and not any(x.startswith("KDRIFT") for x in out)
    rc, out = run(["--sanity-only", "--build-root", "/x", "nocmp=b"])
    assert rc == 2 and "board=? launches=0 run_rc=-1" in out[0]


def test_k_diag_ovl_model():
    """k_diag.py --model ovl (M3.1.2): compute/ovl tail parsed; sum / max overlap forms; old lines refused"""
    import importlib.util
    path = SRC.parent / "script" / "perf" / "k_diag.py"
    spec = importlib.util.spec_from_file_location("k_diag_ovl", path)
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    txt = ("PROF aie_site_perf part=1 kind=conv launches=8 perf_n=8 io_act_cyc=5466554 lock_stall_cyc=0 "
           "stream_stall_cyc=0 pre_io_act_cyc=4136485 snap_gap_cyc=14171 compute_cyc=1176402 ovl_cyc=2842704 "
           "dact_cyc=5400000\n"
           "PROF aie_site_perf part=2 kind=fa launches=3 perf_n=3 io_act_cyc=3000000 lock_stall_cyc=0 "
           "stream_stall_cyc=0 pre_io_act_cyc=77 snap_gap_cyc=500000\n"
           "PROF aie_site_perf part=3 kind=fa launches=3 perf_n=1 io_act_cyc=1 lock_stall_cyc=0 stream_stall_cyc=0\n")
    pf = kd.parse_perf(txt)
    assert pf[1]["has_ovl"] and pf[1]["compute_cyc"] == 1176402 and pf[1]["ovl_cyc"] == 2842704
    assert pf[1]["dact_cyc"] == 5400000 and pf[1]["gap"] == 14171
    assert not pf[2]["has_ovl"] and kd.ovl_missing(pf) == 1   # part 3 is short: not counted
    d = kd.split_site(5.49, 0.0, pf[1])
    assert abs(d["oc"] - 1.176402) < 1e-12 and abs(d["ovl"] - 2.842704) < 1e-12
    s = dict(d, ddr_b=4e6, bucket=1, launches=8)
    p = dict(inv={1: 1e-6}, t0=0.01)  # dma = 4 ms, t0 * L = 0.08 ms
    assert abs(kd.k_hat(dict(p, form="sum"), s) - (1.176402 + 4.0 - 2.842704 + 0.08)) < 1e-12
    assert abs(kd.k_hat(dict(p, form="max"), s) - (4.0 - 2.842704 + 0.08)) < 1e-12
    for form in ("sum", "max"):
        A = kd._synth(21, 30, (1 / 2.0e6, 1 / 6.0e6, 1 / 9.0e6), 0.02, form=form)
        q = kd.fit(A, form)
        assert q["form"] == form and max(kd.rels(q, A)) < 1e-9


def test_k_diag_tol_3runs(capsys):
    """k_diag.py (M3.1.2.1): per-site tol = max(0.15 K_i, 2 spread_abc, 0.01) over runs a/b/c, kge0.1 scope -> KSMALL,
    KRESID terms sum to K_hat - K_meas"""
    import importlib.util
    path = SRC.parent / "script" / "perf" / "k_diag.py"
    spec = importlib.util.spec_from_file_location("k_diag_tol", path)
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    rows = [dict(part=1, K=[1.0, 1.02, 0.98], Kh=[1.12] * 3),     # within 0.15 K (spread .04)
            dict(part=2, K=[1.0, 1.0, 1.0], Kh=[1.2] * 3),        # excess 0.05
            dict(part=3, K=[1.0, 1.3, 1.0], Kh=[1.5] * 3),        # noisy: tol = 2 * 0.3
            dict(part=4, K=[0.05, 0.2, 0.2], Kh=[0.5] * 3)]       # K < 0.1 on run a
    t = kd.ktol_eval(rows, "all")
    assert (t["sites"], t["npass"], t["fail"], t["worst_part"]) == (4, 2, 2, 4)
    t = kd.ktol_eval(rows, "kge0.1")
    assert (t["sites"], t["npass"], t["fail"], t["worst_part"], t["small"], t["small_within"]) == (3, 2, 1, 2, 1, 0)
    assert abs(t["worst_excess"] - 0.05) < 1e-12 and abs(t["small_maxrel"] - 9.0) < 1e-12
    # residual split: sum form, dma 1 ms (model) vs dact 1.5 (measured), 0.4 ms unmodelled in-window, tail 0.25 vs t0*L 0.1
    s = dict(oc=3.0, ovl=2.0, compute=2.5, ddr_b=1e6, bucket=0, launches=10, dact=1.5, core=2.9, K=3.15)
    terms, r = kd.resid_terms(dict(inv={0: 1e-6}, t0=0.01, form="sum"), s)
    assert abs(r + 1.05) < 1e-12 and abs(sum(terms.values()) - r) < 1e-12
    assert abs(terms["dma"] + 0.5) < 1e-12 and abs(terms["launches"] + 0.4) < 1e-12 and abs(terms["t0"] + 0.15) < 1e-12
    capsys.readouterr()
    assert kd.selftest()
    out = capsys.readouterr().out
    for name in ("3run_tol", "small_scope", "kresid_sum"):
        assert "KDIAG SELFTEST %-28s ok" % name in out


def _k_diag_mod(name):
    import importlib.util
    path = SRC.parent / "script" / "perf" / "k_diag.py"
    spec = importlib.util.spec_from_file_location(name, path)
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    return kd


def _ksem_perf(io_act, lock, stream, cc, oc):
    return dict(kind="conv", launches=1, perf_n=1, short=False, has_ovl=True, io_act=io_act, lock=lock,
                stream=stream, gap=0, compute_cyc=cc, ovl_cyc=oc, dact_cyc=0)


def test_ksem_compute_same_window():
    """KSEM (M3.1.1 RS): compute = logged compute_cyc (same A->B window), dwait = lock + stream; the board row
    conv0 (io_act - stalls < 0) no longer gives a negative compute"""
    kd = _k_diag_mod("k_diag_ksem1")
    pf = _ksem_perf(6369000, 7267000, 0, 1900000, 2798000)
    d = kd.split_site(6.755, 0.0, pf)
    assert abs(d["compute"] - 1.9) < 1e-12 and abs(d["dwait"] - 7.267) < 1e-12
    assert d["compute"] >= 0 and abs(d["oc"] + d["dwait"] - (d["core"] + d["ovl"])) < 1e-9
    old = kd.split_site(6.755, 0.0, dict(pf, has_ovl=False))  # no ovl tail: legacy io_act - stalls kept
    assert abs(old["compute"] - (6.369 - 7.267)) < 1e-12 and "oc" not in old


def test_ksem_line_pass_and_fail(capsys):
    """KSEM line: PASS on window-consistent rows; a negative compute and a broken identity -> FAIL;
    sites with core <= 0.05 ms are not dact-checked"""
    kd = _k_diag_mod("k_diag_ksem2")
    good = dict(perf=_ksem_perf(1, 1, 1, 1, 1), **kd.split_site(9.0, 0.0, _ksem_perf(6000000, 3000000, 1000000,
                                                                                   2500000, 500000)))
    tiny = dict(perf=_ksem_perf(1, 1, 1, 1, 1), **kd.split_site(0.1, 0.0, _ksem_perf(40000, 0, 0, 1000, 0)))
    r = kd.ksem_line("t", [[good, tiny], [dict(good)], [dict(good)]])
    assert r["ok"] and r["rows"] == 4
    bad = dict(good, compute=-0.1, oc=-0.1)
    r = kd.ksem_line("t", [[good], [bad]])
    assert not r["ok"] and r["neg"] == 1 and r["bad"] == 1
    out = capsys.readouterr().out.splitlines()
    assert out == ["KSEM cfg=t runs=3 site_runs=4 neg_compute=0 dact_bad=0 dact_tol=0.05 PASS",
                   "KSEM cfg=t runs=2 site_runs=2 neg_compute=1 dact_bad=1 dact_tol=0.05 FAIL"]


def _kdmaf_mod(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, SRC.parent / "script" / "perf" / "k_dmafeat.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kdmaf_perch_selftest():
    """k_dmafeat.py --selftest (M3.1.2.3 RA): >= 12 cases incl. dominant-channel pick, constant drop, nan reject"""
    import re
    import subprocess
    p = SRC.parent / "script" / "perf" / "k_dmafeat.py"
    r = subprocess.run([sys.executable, "-I", str(p), "--selftest"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    m = re.search(r"^KDMAF SELFTEST PASS \((\d+)/(\d+)\)$", r.stdout, re.M)
    assert m and m.group(1) == m.group(2) and int(m.group(1)) >= 12, r.stdout
    for name in ("dom_channel", "const_drop", "nan_reject", "maxch_t_bw"):
        assert re.search(r"^KDMAF SELFTEST %s ok$" % name, r.stdout, re.M), name


def test_kdmaf_perch_dom_and_rank():
    """dominant = MM2S channel with most bytes; constant features dropped; varying nan feature rejects the TGT"""
    kf = _kdmaf_mod("k_dmafeat_perch1")
    ch = {(0, "MM2S", 0): dict(bytes=10.0, bd_n=1.0, run_b=32), (0, "MM2S", 1): dict(bytes=50.0, bd_n=5.0, run_b=64),
          (0, "S2MM", 0): dict(bytes=500.0, bd_n=1.0, run_b=128)}
    f = kf.ch_feats(ch)
    assert (f["dom_bytes"], f["dom_bd_n"], f["dom_run_b"]) == (50.0, 5.0, 64)
    assert abs(f["maxch_t"] - 500 / 10800.0) < 1e-12
    ss = [dict(k=3, v=x, dwait=-x) for x in (1, 2, 3, 4)]
    best, rho, cr, nc, nn = kf.rank_feats(ss, ("k", "v"))
    assert best == "v" and abs(rho + 1.0) < 1e-12 and nc == 1 and nn == 0 and kf.tgt_pass(rho, nn)
    best, rho, cr, nc, nn = kf.rank_feats([dict(v=x, dwait=1.0) for x in (1, 2, 3)], ("v",))
    assert nn == 1 and not kf.tgt_pass(rho, nn)


def test_kdmaf_perch_no_resid_needed(tmp_path):
    """--perch does not require --resid (argparse error rc=2 only without --build-root / runs); plain mode still does"""
    import subprocess
    p = SRC.parent / "script" / "perf" / "k_dmafeat.py"
    r = subprocess.run([sys.executable, "-I", str(p), "--perch"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "--perch needs" in r.stderr and "--resid" not in r.stderr.split("error:")[-1]
    r = subprocess.run([sys.executable, "-I", str(p), "--build-root", str(tmp_path), "x=y"], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 2 and "--resid" in r.stderr


def test_kdmaf_perch_unet_logs():
    """M3.1.2.3 RA on the UNet a-logs (skipped when the build dir / logs are absent): conv TGT |rho| >= 0.5 nan=0"""
    import re
    import subprocess
    b = Path("/scratch/staff/huaj/tmp/costmodel_m3121")
    d = Path("/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks/2026-11-20_M3121")
    logs = [d / "unet_off.a.log", d / "unet_aot-kco.a.log"]
    if not (b.is_dir() and all(x.is_file() for x in logs)):
        pytest.skip("UNet build dir / logs not present")
    p = SRC.parent / "script" / "perf" / "k_dmafeat.py"
    r = subprocess.run([sys.executable, "-I", str(p), "--perch", "--build-root", str(b), "off=%s" % logs[0],
                        "aot-kco=%s" % logs[1]], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr
    assert r.stdout.splitlines()[-1] == "KDMAF ALL sites=276 feats=dmach PASS"
    for cfg in ("off", "aot-kco"):
        assert re.search(r"^KDMAF TGT cfg=%s cls=conv n=48 const=\d+ best=\S+ spearman=-?(0\.[5-9]\d*|1\.0*) nan=0 PASS$"
                         % cfg, r.stdout, re.M), cfg


# --- M3.1.2.3 RF: k_core_check --max-rel / src= / fa_src= (FA core from ISS) ---
def _kcore_mod(name, monkeypatch, fa_src="iss", fa_pred=900.0, lin_pred=1000.0):
    """k_core_check with a fake k_diag (one log, two sites: linear0 + fa1, compute_cyc 1000 each) and a fake kcyc"""
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location(name, SRC.parent / "script" / "perf" / "k_core_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    kd = types.ModuleType("k_diag")
    import re
    kd._CMP = re.compile(r"^CMP unet board=(\S+) .*aie_launches=(\d+) .*run_rc=(-?\d+)", re.M)
    perf = dict(compute_cyc=1000, perf_n=1, has_ovl=True)
    kd.load = lambda cfg, lg, br: ([dict(kind="linear", site="linear0", cls="linear", launches=1, perf=perf),
                                    dict(kind="fa", site="fa1", cls="fa16", launches=1, perf=perf)],
                                   "CMP unet board=palmyra213 x aie_launches=258 y run_rc=0\n")
    monkeypatch.setitem(sys.modules, "k_diag", kd)
    monkeypatch.setattr(m, "SITES_PER_CFG", 2)
    monkeypatch.setattr(m, "N_LOGS", 1)
    monkeypatch.setattr(m, "kcyc", lambda b, i: (0, {("linear", 0): dict(core_cyc=lin_pred, src="iss"),
                                                     ("fa", 1): dict(core_cyc=fa_pred, src=fa_src)}))
    return m


def _kcore_run(m, capsys, *extra):
    rc = m.main(["--build-root", "/nonexistent", *extra, "c=x.log"])
    return rc, capsys.readouterr().out


def test_kcore_maxrel_fail_and_pass(monkeypatch, capsys):
    """--max-rel: maxrel 0.25 > 0.20 -> FAIL rc=1; <= 0.30 -> PASS; no --max-rel -> no accuracy limit"""
    m = _kcore_mod("k_core_check_rf1", monkeypatch, fa_pred=750.0)
    rc, out = _kcore_run(m, capsys, "--max-rel", "0.20")
    last = out.splitlines()[-1]
    assert rc == 1 and last.startswith("KCORE ALL sites=2 cover=2 finite=1 maxrel=0.2500 ") and last.endswith(
        " fa_src=iss FAIL"), out
    rc, out = _kcore_run(m, capsys, "--max-rel", "0.30")
    assert rc == 0 and out.splitlines()[-1].endswith(" fa_src=iss PASS"), out
    rc, out = _kcore_run(m, capsys)
    assert rc == 0 and out.splitlines()[-1].endswith(" PASS"), out


def test_kcore_maxrel_src_and_fa_src(monkeypatch, capsys):
    """per-class src=<iss|formula|mixed>; fa_src=formula on the ALL line -> FAIL even within --max-rel"""
    import re
    m = _kcore_mod("k_core_check_rf2", monkeypatch, fa_src="formula_noloo", fa_pred=1000.0)
    rc, out = _kcore_run(m, capsys, "--max-rel", "0.20")
    assert re.search(r"^KCORE cfg=c cls=fa16 sites=1 .* src=formula$", out, re.M), out
    assert re.search(r"^KCORE cfg=c cls=linear sites=1 .* src=iss$", out, re.M), out
    assert re.search(r"^KCORE cfg=c cls=ALL sites=2 .* src=mixed$", out, re.M), out
    assert rc == 1 and re.search(r"^KCORE ALL sites=2 cover=2 finite=1 maxrel=0\.0000 med_rel=0\.0000 worst_cls=\S+ "
                                 r"fa_src=formula FAIL$", out.splitlines()[-1]), out
    assert "no accuracy threshold" not in m.__doc__.lower()


def test_kcore_maxrel_kcyc_fa_src(tmp_path):
    """kcyc_unet: fa_src from the data (iss only when every FA site has an md5-matching ISS value), variants"""
    import hashlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("kcyc_unet_rf", SRC.parent / "script" / "perf" / "kcyc_unet.py")
    k = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(k)
    assert (k.fa_src_of(18, 18), k.fa_src_of(0, 18), k.fa_src_of(6, 18)) == ("iss", "formula_noloo", "mixed")
    K = "fa_kernel__x"
    for f, t in (("kernel_%s.cc" % K, b"kco"), ("%s.cc" % K, b"fa"), ("aieml_%s.bcf" % K, b"bcf")):
        (tmp_path / f).write_bytes(t)
    h = lambda b: hashlib.md5(b).hexdigest()  # noqa: E731
    off = dict(core_cyc=100, src_md5=dict(kernel=h(b"off"), fa=h(b"fa"), bcf=h(b"bcf")))
    assert k.fa_iss_pick(str(tmp_path), K, off) is None
    kco = dict(core_cyc=120, src_md5=dict(kernel=h(b"kco"), fa=h(b"fa"), bcf=h(b"bcf")))
    assert k.fa_iss_pick(str(tmp_path), K, dict(off, variants=[kco]))["core_cyc"] == 120
    assert k.fa_iss_pick(str(tmp_path / "missing"), K, dict(off, variants=[kco])) is None


def _m3_lpred_mod(monkeypatch, l_hat, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("m3_site_lp", SRC.parent / "script" / "perf" / "m3_site.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "SITES_PER_CFG", {"fa": 1})
    rows = [dict(kind="fa", cid=3, part=7, launches=2, K=0.5, L=0.2, N=0.3)]
    monkeypatch.setattr(m, "site_rows", lambda b, lg: (rows, ""))
    monkeypatch.setattr(m, "probe", lambda c, lg, b: print("M3SITE stub"))
    monkeypatch.setattr(m, "l_sites", lambda b, c, p, k: {("fa", 3): l_hat if k == "site" else 0.1})
    prm = tmp_path / "l.json"
    prm.write_text('{"cfg": {"off": {"coef": {"rst": 0.020983}}}}')
    return m, prm


def test_m3_lpred_pass_and_tsv(monkeypatch, capsys, tmp_path):
    """Tshare = |L_hat - L_meas| / (K+L+N) = 0.05 <= 0.10 -> PASS rc=0; rst per cfg; TSV mkdir -p + 1 row"""
    m, prm = _m3_lpred_mod(monkeypatch, 0.25, tmp_path)
    tsv = tmp_path / "new" / "lp.tsv"
    rc = m.main(["--lpred", "--params", str(prm), "--tsv", str(tsv), "--build-root", "/x", "off=a.log"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "LCHK cfg=off logs=1 sites=1 rst=0.020983 kfirst=site L_maxrel=0.2500 Tshare_max=0.0500 PASS" in out
    assert "LFA cfg=off small_sites=1 maxrel_avg=0.5000 maxrel_site=0.2500" in out
    assert out.splitlines()[-1] == "LCHK ALL logs=1 sites=1 kfirst=site L_maxrel=0.2500 Tshare_max=0.0500 PASS"
    lines = tsv.read_text().splitlines()
    assert lines[0].split("\t") == ["cfg", "run", "part", "site", "kind", "launches", "site_ms", "L_meas", "L_hat"]
    assert lines[1].split("\t") == ["off", "a.log", "7", "fa3", "fa", "2", "1.0000", "0.2000", "0.2500"]


def test_m3_lpred_tshare_fail(monkeypatch, capsys, tmp_path):
    """Tshare 0.15 > 0.10 -> FAIL rc=1 (probe verdict never enters rc)"""
    m, prm = _m3_lpred_mod(monkeypatch, 0.35, tmp_path)
    rc = m.main(["--lpred", "--params", str(prm), "--tsv", str(tmp_path / "lp.tsv"), "--build-root", "/x", "off=a.log"])
    out = capsys.readouterr().out
    assert rc == 1 and out.splitlines()[-1].endswith("Tshare_max=0.1500 FAIL"), out


# --- M3.1.2.3 PA: k_dmafeat --perch --ddrt (ddr_t = sum_ch bytes / BW(run_b), ms/launch) ---
def test_kdmaf_ddrt_sum_and_reject():
    """ddr_t sums every channel's bytes / BW(run_b) in ms; run_b outside the BW table raises (ch_feats -> nan)"""
    kf = _kdmaf_mod("k_dmafeat_ddrt1")
    ch = {(0, "MM2S", 0): dict(bytes=1.9e6, bd_n=1.0, run_b=16), (0, "S2MM", 0): dict(bytes=10.8e6, bd_n=1.0, run_b=4096),
          (1, "MM2S", 0): dict(bytes=0.0, bd_n=0.0, run_b=0)}
    assert abs(kf.ddr_t(ch) - 2.0) < 1e-12
    with pytest.raises(ValueError):
        kf.bw_run(0)
    v = kf.ch_feats({(0, "MM2S", 0): dict(bytes=5.0, bd_n=1.0, run_b=0)})["ddr_t"]
    assert v != v  # nan
    assert abs(kf.ddr_t({(0, "MM2S", 0): dict(bytes=1.32383e8, bd_n=1.0, run_b=16)}) - 1.32383e8 / 1.9e6) < 1e-9


def test_kdmaf_ddrt_dwait_ms_from_ksite():
    """dwait_ms = (KSITE lock + stream) / launches via k_diag.split_site (AIE 1 GHz: 1e6 cyc = 1 ms)"""
    kf = _kdmaf_mod("k_dmafeat_ddrt2")
    d = kf.load_dwait_ms("PROF aie_site_perf part=7 kind=conv launches=4 perf_n=4 io_act_cyc=1 "
                         "lock_stall_cyc=6000000 stream_stall_cyc=2000000\n")
    assert abs(d[7] - 2.0) < 1e-12


def test_kdmaf_ddrt_tsv(tmp_path):
    """write_tsv: header = TSV_COLS, one row per site, parent dir created"""
    kf = _kdmaf_mod("k_dmafeat_ddrt3")
    s = dict(cfg="off", part=414, site="conv121", kind="conv", launches=1, ddr_b=1.32383e8, dom_bytes=2.94912e7,
             maxch_t=15521.7, ddr_t=68.5381, rw_ratio=0.98, bd_n=384)
    p = tmp_path / "a" / "b" / "x.tsv"
    kf.write_tsv(p, [s, s])
    rows = p.read_text().splitlines()
    assert rows[0].split("\t") == list(kf.TSV_COLS) and len(rows) == 3
    assert rows[1].split("\t")[:4] == ["off", "414", "conv121", "conv"] and rows[1].split("\t")[8] == "68.5381"


def test_kdmaf_ddrt_unet_logs(tmp_path):
    """M3.1.2.3 PA on the UNet a-logs (skipped when absent): conv TGT feat=ddr_t PASS, conv121 DDRT, 276 TSV rows"""
    import re
    import subprocess
    b = Path("/scratch/staff/huaj/tmp/costmodel_m3121")
    d = Path("/scratch/staff/huaj/aiengineers/shuhua/clones/cost/tasks/2026-11-20_M3121")
    logs = [d / "unet_off.a.log", d / "unet_aot-kco.a.log"]
    if not (b.is_dir() and all(x.is_file() for x in logs)):
        pytest.skip("UNet build dir / logs not present")
    p = SRC.parent / "script" / "perf" / "k_dmafeat.py"
    t = tmp_path / "kln" / "dmach.tsv"
    r = subprocess.run([sys.executable, "-I", str(p), "--perch", "--ddrt", "--tsv", str(t), "--build-root", str(b),
                        "off=%s" % logs[0], "aot-kco=%s" % logs[1]], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr
    assert r.stdout.splitlines()[-1] == "KDMAF ALL sites=276 feats=dmach+ddrt PASS"
    for cfg in ("off", "aot-kco"):
        assert re.search(r"^KDMAF TGT cfg=%s cls=conv n=48 feat=ddr_t spearman=-?[0-9.]+ nan=0 PASS$" % cfg,
                         r.stdout, re.M), cfg
    assert re.search(r"^KDMAF DDRT cfg=off site=conv121 part=414 ddr_t_ms=\d+\.\d+ dwait_ms=6\d\.\d+ ratio=\d+\.\d+$",
                     r.stdout, re.M)
    assert len(t.read_text().splitlines()) == 277


def test_khat_tsv_probe(tmp_path, capsys):
    """k_diag.py --khat-tsv (M3.1.1.2b PKP): KORACLE run-a class params -> per-launch K_hat rows (k_hat / launches),
    sites without counters -> none / nan, launches=0 sites dropped; TSV header + 8 cols, KHAT line names the file;
    the selftest carries the khat_tsv case"""
    kd = _k_diag_mod("k_diag_khat")
    w, t0, kf = dict(ch=0.08, run=0.004), 0.03, 0.2
    A = kd._synth_dma(61, 12, w, t0, kf)
    po = kd.oracle_fit_dma(A)
    p = po["linear"][0]
    nc = dict(A[0], site="dx", part=99)
    del nc["oc"], nc["ovl"]
    rows = kd.khat_rows("c", po, kd.dma_cls, A + [nc, dict(A[1], site="dz", part=98, launches=0)])
    assert len(rows) == 13 and [r["part"] for r in rows] == sorted(r["part"] for r in rows)
    by = {s["part"]: s for s in A}
    for r in rows[:-1]:
        assert r["compute_src"] == r["ovl_src"] == "meas"
        assert abs(r["K_hat_ms"] * r["launches"] - kd.k_hat(p, by[r["part"]])) < 1e-9
    assert rows[-1]["compute_src"] == "none" and rows[-1]["K_hat_ms"] != rows[-1]["K_hat_ms"]
    tsv = tmp_path / "khat_probe.tsv"
    capsys.readouterr()
    assert kd.write_khat(tsv, {"c": rows[:-1]}, "koracle_dma7") == 12
    out = capsys.readouterr().out.splitlines()
    assert out == ["KHAT cfg=c src=koracle_dma7 fit=a sites=12 compute=meas ovl=meas tsv=%s" % tsv]
    lines = tsv.read_text().splitlines()
    assert lines[0] == "# cfg part site cls launches K_hat_ms compute_src ovl_src" and len(lines) == 13
    assert all(len(x.split()) == 8 for x in lines[1:])
    assert kd.selftest()
    assert "KDIAG SELFTEST %-28s ok" % "khat_tsv" in capsys.readouterr().out


# ---- M3.1.1.2 PC: kln_site per-launch K + L + N synthesis ----
def test_kln_norm_run_join():
    """lpred run 'unet_off.a.log' and npred run 'a' unify to the same (cfg, run, part) key"""
    ks = _script("kln_site")
    assert ks.norm_run("unet_off.a.log") == ks.norm_run("a") == "a"
    assert ks.norm_run("/x/unet_aot-kco.c.log") == "c"
    with pytest.raises(ValueError):
        ks.norm_run("unet_off.log")
    lp = ks.tsv_map([dict(cfg="off", run="unet_off.b.log", part="14", site="conv0", site_ms="1")], ("site_ms",))
    np_ = ks.tsv_map([dict(cfg="off", run="b", part="14", site="conv0", site_ms="1")], ("site_ms",))
    assert set(lp) == set(np_) == {("off", "b", 14)}


def test_kln_selftest_audit(capsys):
    """selftest PASS; the predict path reads no measured field"""
    ks = _script("kln_site")
    assert ks.selftest()
    assert ks.audit()
    assert capsys.readouterr().out.splitlines()[-1] == "KLN AUDIT meas_at_predict=0"
    pv = ks.pred_view(dict(L_hat=1.0, N_hat=1.0, launches=2, L_meas=5.0, N_meas=5.0, site_ms=50.0))
    assert set(pv) == {"L_hat", "N_hat", "launches"} and ks.predict_launch(2.0, pv) == 2.0


def test_kln_insample_khat_missing(tmp_path):
    """--insample with a missing or empty khat file -> rc=2"""
    ks = _script("kln_site")
    e = tmp_path / "k.tsv"
    e.write_text("")
    for k in (tmp_path / "none.tsv", e):
        assert ks.main(["--insample", "--khat", str(k), "--lpred", "x", "--npred", "y", "--fit", "off=z"]) == 2


def test_kln_insample_tsv_only(tmp_path, capsys):
    """--insample without logs: run-a rows of the LPRED/NPRED join, k_diag KHAT (per launch, no run col), rc=0"""
    ks = _script("kln_site")
    lp, npd, kh = tmp_path / "l.tsv", tmp_path / "n.tsv", tmp_path / "k.tsv"
    lp.write_text("cfg\trun\tpart\tsite\tkind\tlaunches\tsite_ms\tL_meas\tL_hat\n"
                  "off\tunet_off.a.log\t1\ts1\tconv\t1\t10\t1\t1\n"
                  "off\tunet_off.a.log\t2\ts2\tfa\t2\t4\t1\t1\n"
                  "off\tunet_off.b.log\t1\ts1\tconv\t1\t99\t1\t1\n")
    npd.write_text("cfg\trun\tpart\tsite\tlaunches\tsite_ms\tN_meas\tN_hat\n"
                   "off\ta\t1\ts1\t1\t10\t2\t2\noff\ta\t2\ts2\t2\t4\t1\t1\noff\tb\t1\ts1\t1\t99\t2\t2\n")
    kh.write_text("# cfg part site cls launches K_hat_ms compute_src ovl_src\n"
                  "off 1 s1 conv 1 7.000000 meas meas\noff 2 s2 fa16 2 1.000000 meas meas\n")
    assert ks.main(["--insample", "--khat", str(kh), "--lpred", str(lp), "--npred", str(npd)]) == 0
    ln = capsys.readouterr().out.strip().splitlines()
    assert len(ln) == 1 and ln[0].startswith("KLNPROBE cfg=off src=koracle_dma7 run=a site_src=lpred sites=2 nan=0 ")
    assert " r2=1.0000 maxrel=0.0000 " in ln[0] and ln[0].endswith(" probe")


# ---- M3.1.1.2 PL1: m3_site --lpred --lrefit (aot-kco L + offset / kswitch, fit on run a only) ----
def _m3_lrefit_mod(monkeypatch, tmp_path, l_meas):
    """2 sites per log; site_ms 1.0; base L_hat 0.1; feats: (launches, kswitch) = (2, 1) and (1, 0)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("m3_site_lr", SRC.parent / "script" / "perf" / "m3_site.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "SITES_PER_CFG", {"fa": 1, "linear": 1})
    monkeypatch.setattr(m, "LREFIT_GRID", [round(0.01 * i, 2) for i in range(21)])

    def rows(b, lg):
        lm = l_meas[m.run_label(lg)] if "aot-kco" in b else (0.12, 0.12)
        return [dict(kind="fa", cid=3, part=7, launches=2, K=0.5, L=lm[0], N=1.0 - 0.5 - lm[0]),
                dict(kind="linear", cid=4, part=8, launches=1, K=0.5, L=lm[1], N=1.0 - 0.5 - lm[1])], ""
    monkeypatch.setattr(m, "site_rows", rows)
    monkeypatch.setattr(m, "probe", lambda c, lg, b: print("M3SITE stub"))
    monkeypatch.setattr(m, "l_sites", lambda b, c, p, k: {("fa", 3): 0.1, ("linear", 4): 0.1})
    monkeypatch.setattr(m, "site_launch_feats", lambda b: {("fa", 3): dict(launches=2, kswitch=1, kfirst=1, cp_w=5.0),
                                                           ("linear", 4): dict(launches=1, kswitch=0, kfirst=0,
                                                                               cp_w=1.0)})
    prm = tmp_path / "l.json"
    prm.write_text('{"cfg": {"off": {"coef": {"rst": 0.020983}}, "aot-kco": {"coef": {"const": 0.03}}}}')
    return m, prm


def _lrefit_argv(prm, tmp_path, x="0.08"):
    return ["--lpred", "--params", str(prm), "--lrefit", "aot-kco", "--refit-feats", "offset,kswitch", "--fit-run", "a",
            "--tshare-max", "aot-kco=" + x, "--out-params", str(tmp_path / "o.json"), "--tsv", str(tmp_path / "t.tsv"),
            "--build-root", "/x", "off=unet_off.a.log", "aot-kco=unet_aot-kco.a.log,unet_aot-kco.b.log"]


def test_m3_lrefit_fit_a_only_and_params(monkeypatch, capsys, tmp_path):
    """run a: L = 0.1 + 0.02 x launches + 0.05 x kswitch exactly -> offset 0.02 kswitch 0.05 (b never fitted);
    off copied unchanged, fit_runs [a]; LBIAS before LREFIT; rc=0; --params on the out json prints the same LREFIT"""
    m, prm = _m3_lrefit_mod(monkeypatch, tmp_path, {"a": (0.19, 0.12), "b": (0.20, 0.12)})
    rc = m.main(_lrefit_argv(prm, tmp_path))
    out = capsys.readouterr().out
    assert rc == 0, out
    lr = "LREFIT cfg=aot-kco fit=a sites=2 feats=offset,kswitch offset_ms=0.0200 kswitch_ms=0.0500 loo_Tshare_max="
    assert lr in out and out.index("LBIAS cfg=aot-kco run=a group=kswitch=1") < out.index(lr)
    assert "LCHKRUN cfg=aot-kco run=a fit=1 sites=2 Tshare_max=0.0000 Lsum_rel=0.0000 PASS" in out
    assert "LCHKRUN cfg=aot-kco run=b fit=0 sites=2 Tshare_max=0.0100 Lsum_rel=-0.0313 PASS" in out
    import json
    o = json.load(open(tmp_path / "o.json"))
    assert o["cfg"]["off"] == json.load(open(prm))["cfg"]["off"]
    assert o["cfg"]["aot-kco"]["fit_runs"] == ["a"] and o["cfg"]["aot-kco"]["coef"]["kswitch"] == 0.05
    assert len((tmp_path / "t.tsv").read_text().splitlines()) == 1 + 2 + 4
    rc2 = m.main(["--lpred", "--params", str(tmp_path / "o.json"), "--tsv", str(tmp_path / "t2.tsv"), "--build-root",
                  "/x", "aot-kco=unet_aot-kco.a.log,unet_aot-kco.b.log"])
    out2 = capsys.readouterr().out
    assert lr in out2 and "LBIAS" not in out2 and "LCHKRUN cfg=aot-kco run=b fit=0" in out2


def test_m3_lrefit_rc_strict_and_off(monkeypatch, capsys, tmp_path):
    """held-out Tshare_max exactly 0.0800 is not < 0.08 -> FAIL rc=1; LCHK aot-kco never enters rc, LCHK off does"""
    m, prm = _m3_lrefit_mod(monkeypatch, tmp_path, {"a": (0.19, 0.12), "b": (0.11, 0.20)})
    rc = m.main(_lrefit_argv(prm, tmp_path))
    out = capsys.readouterr().out
    assert rc == 1 and "LCHKRUN cfg=aot-kco run=b fit=0 sites=2 Tshare_max=0.0800" in out and \
        [x for x in out.splitlines() if x.startswith("LCHKRUN cfg=aot-kco run=b")][0].endswith(" FAIL"), out
    rc = m.main(_lrefit_argv(prm, tmp_path, "0.09"))
    out = capsys.readouterr().out
    rb = [x for x in out.splitlines() if x.startswith("LCHKRUN cfg=aot-kco run=b")]
    assert rc == 0 and rb[0].startswith("LCHKRUN cfg=aot-kco run=b fit=0 sites=2 Tshare_max=0.0800 Lsum_rel=") \
        and rb[0].endswith(" PASS"), out
    m2, prm2 = _m3_lrefit_mod(monkeypatch, tmp_path, {"a": (0.19, 0.12), "b": (0.19, 0.12)})
    monkeypatch.setattr(m2, "l_sites", lambda b, c, p, k: {("fa", 3): 0.1, ("linear", 4): 0.1 if "aot" in b else 0.3})
    rc = m2.main(_lrefit_argv(prm2, tmp_path))
    out = capsys.readouterr().out
    assert rc == 1 and "LCHK cfg=off logs=1 sites=2" in out and "Tshare_max=0.1800 FAIL" in out, out
