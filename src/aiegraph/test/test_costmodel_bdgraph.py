"""aiegraph.costmodel.bdgraph -- the bdgraph input form of estimate() (structural counts from the build JSON).

Given / when / then in the test names.  Synthetic build JSONs in tmp_path; one optional check on a real build.
"""
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph.costmodel import bdgraph, estimate  # noqa: E402
from test_costmodel_estimate import _op, cycles_file  # noqa: E402,F401  (shared fixture + kacc op)


def _bd(i, n, **kw):
    b = {"bd_id": i, "len": n, "enable_packet": False, "next_bd": -1,
         "acquire_lock": [{"id": -1, "val": 0}], "release_lock": [{"id": -1, "val": 0}]}
    b.update(kw)
    return b


def _write(d, starts, bds, host_lines, shim_bds=None):
    dfs = {"module_attrs": [{"k_rounds": 0}], "tiles": [
        {"col": 0, "row": 0, "type": "shim", "dma_channels": [
            {"channel": 0, "direction": "MM2S", "bd_chain": shim_bds or [_bd(0, 4096)], "start_io": starts}]},
        {"col": 0, "row": 3, "type": "core", "dma_channels": [
            {"channel": 0, "direction": "S2MM", "bd_chain": bds, "start_io": starts}]}]}
    sv = {"tiles": [{"loc": [0, 0], "dma_channels": [{"direction": "MM2S", "channel": 0, "host_lines": host_lines}]}]}
    (d / "dfscheduleprovenancemap.json").write_text(json.dumps(dfs))
    (d / "schedule_view.json").write_text(json.dumps(sv))
    return d


# 2 BD configs once per launch, then a loop with 3 BD configs + 1 wait per round
HOST = [{"line": 10, "kind": "bd_config", "bd_id": 0}, {"line": 11, "kind": "bd_config", "bd_id": 1},
        {"line": 20, "kind": "loop"}, {"line": 21, "kind": "bd_config", "bd_id": 0},
        {"line": 22, "kind": "bd_config", "bd_id": 1}, {"line": 23, "kind": "bd_config", "bd_id": 2},
        {"line": 24, "kind": "startio"}, {"line": 25, "kind": "wait"}, {"line": 30, "kind": "loopend"}]
LOOPED = [{"repeat_count": 1, "inside_scf_for": True, "loop_range": "0..5"}]


def test_given_a_numeric_loop_range_then_rounds_and_bd_configs_per_launch_follow_it(tmp_path):
    f = bdgraph.features(_write(tmp_path, LOOPED, [_bd(0, 256)], HOST))
    assert f["rounds"] == 5 and not f["rounds_unknown"]
    assert f["bd_config"] == {"launch": 2, "round": 3}
    assert f["bd_configs_per_launch"] == 2 + 3 * 5
    assert f["waits"] == {"launch": 0, "round": 1}


def test_given_an_ssa_loop_range_then_rounds_are_unknown_unless_trip_count_is_passed(tmp_path):
    starts = [{"repeat_count": 1, "inside_scf_for": True, "loop_range": "v7..v6 step v5"}]
    d = _write(tmp_path, starts, [_bd(0, 256)], HOST)
    f = bdgraph.features(d)
    assert f["rounds"] is None and f["rounds_unknown"] and f["bd_configs_per_launch"] is None
    g = bdgraph.features({"worklocal": str(d), "trip_count": 4})
    assert g["rounds"] == 4 and g["bd_configs_per_launch"] == 2 + 3 * 4


def test_given_no_loop_then_one_round_and_all_bd_configs_are_per_launch(tmp_path):
    host = [h for h in HOST if h["kind"] not in ("loop", "loopend")]
    f = bdgraph.features(_write(tmp_path, [{"repeat_count": 1}], [_bd(0, 256)], host))
    assert f["rounds"] == 1 and f["bd_config"] == {"launch": 5, "round": 0} and f["bd_configs_per_launch"] == 5


def test_given_multidim_bds_then_run_b_applies_the_runtime_eff_dims_trim():
    # UNet conv shim S2MM: [4, 20480] / [1280, 1] -> trailing wrap 1 dropped -> plain 5120 B BD
    assert bdgraph.run_bytes(_bd(0, 5120, dim_strides=[4, 20480], dim_wraps=[1280, 1])) == 5120
    # strided: d0 = 16 words contiguous -> 64 B runs
    assert bdgraph.run_bytes(_bd(0, 4096, dim_strides=[4, 256, 64], dim_wraps=[16, 16, 4])) == 64
    # d0 not unit-stride -> one word per run
    assert bdgraph.run_bytes(_bd(0, 4096, dim_strides=[8, 256], dim_wraps=[16, 32])) == 4
    assert bdgraph.run_bytes(_bd(0, 4096)) == 4096


def test_given_runtime_and_locked_bds_then_they_are_counted_not_priced(tmp_path, cycles_file):
    bds = [_bd(0, 256, acquire_lock=[{"id": 1, "val": -1}], release_lock=[{"id": 2, "val": 1}]),
           _bd(1, 256, acquire_lock=[{"id": 1, "val": -1}]),           # one id alone -> no lock
           {"bd_id": "runtime", "len": 0}]
    d = _write(tmp_path, LOOPED, bds, HOST)
    f = bdgraph.features(d)
    assert f["runtime_bds_skipped"] == 1 and f["locked_bds"] == 1
    r = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "jit_mmio"},
                   "host": {"bd_us": 2.0}}], cycles=cycles_file)
    assert "locks(AM020)" in r["unpriced"] and "runtime_bds(1)" in r["unpriced"]


def test_given_bd_write_path_and_bd_us_then_issue_ms_is_bd_configs_times_bd_us(tmp_path, cycles_file):
    d = _write(tmp_path, LOOPED, [_bd(0, 256)], HOST)
    r = estimate({"ops": [{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "aot_window"},
                           "host": {"bd_us": 2.0}}], "launches": 3}, cycles=cycles_file)
    assert r["issue_ms"] == pytest.approx(17 * 2.0 / 1e3 * 3)
    assert r["total_ms"] == pytest.approx(r["issue_ms"])
    assert r["wait"] is None and "wait(bdgraph K model)" in r["unpriced"] and "dma_time(bytes counted)" in r["unpriced"]


def test_given_no_bd_write_path_then_issue_is_unpriced_and_total_is_zero(tmp_path, cycles_file):
    d = _write(tmp_path, LOOPED, [_bd(0, 256)], HOST)
    r = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d)}, "host": {"bd_us": 2.0}}], cycles=cycles_file)
    assert r["issue_ms"] is None and r["total_ms"] == 0.0 and "bd_write_path" in r["unpriced"]


def test_given_an_unknown_bd_write_path_then_it_is_rejected(tmp_path):
    d = _write(tmp_path, LOOPED, [_bd(0, 256)], HOST)
    with pytest.raises(ValueError, match="bd_write_path"):
        bdgraph.features({"worklocal": str(d), "bd_write_path": "dma_magic"})


def test_given_a_kacc_op_and_a_bdgraph_op_in_one_group_then_terms_add_up(tmp_path, cycles_file):
    d = _write(tmp_path, LOOPED, [_bd(0, 256)], HOST)
    k = estimate([_op()], cycles=cycles_file)
    r = estimate([_op(), {"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "jit_mmio"},
                          "host": {"bd_us": 2.0}}], cycles=cycles_file)
    assert r["wait"] == pytest.approx(k["wait"]) and r["bound"] == k["bound"]
    assert r["total_ms"] == pytest.approx(k["total_ms"] + 17 * 2.0 / 1e3)


REAL = Path("/scratch/staff/huaj/tmp/maxwell_2k_mesh/aout/worklocal")


@pytest.mark.skipif(not (REAL / "schedule_view.json").exists(), reason="real build dir not present")
def test_given_the_maxwell_2k_mesh_build_then_counts_match_its_host_cc():
    # host.cc (per host function): 96 bd_config calls before the loop, 24 inside `for v384 in 0..4`
    f = bdgraph.features(REAL)
    assert f["rounds"] == 4 and f["bd_config"] == {"launch": 96, "round": 24}
    assert f["bd_configs_per_launch"] == 192


# ---- DDR bytes per launch (one start runs the BD chain R times, every run moves the full len; peer:1120010346-95530d)

def test_given_a_chained_shim_bd_and_a_looped_startio_then_ddr_bytes_are_chain_times_r_times_trip(tmp_path):
    chain = [_bd(0, 4096, next_bd=1), _bd(1, 1024, next_bd=-1), _bd(2, 999)]   # bd 2 is not on the chain
    host = [{"line": 10, "kind": "createio", "bd_id": 0}, {"line": 20, "kind": "loop"},
            {"line": 21, "kind": "startio", "code": "ioevent v9 = __Runtime_startio(v1, v8, 0, 3);"},
            {"line": 30, "kind": "loopend"}]
    f = bdgraph.features(_write(tmp_path, LOOPED, [_bd(0, 64)], host, shim_bds=chain))
    ch = f["ddr_bytes"]["channels"][0]
    assert ch["run_bytes"] == 5120 and ch["runs"] == 3 * 5
    assert f["ddr_bytes"]["mm2s"] == 5120 * 15 and f["ddr_bytes"]["s2mm"] == 0


def test_given_a_startio_without_a_literal_r_then_start_io_repeat_count_is_used(tmp_path):
    host = [{"line": 10, "kind": "createio", "bd_id": 0}, {"line": 11, "kind": "startio", "code": "x = f(v);"}]
    f = bdgraph.features(_write(tmp_path, [{"repeat_count": 2}], [_bd(0, 64)], host))
    assert f["ddr_bytes"]["mm2s"] == 2 * 4096


def test_given_a_cyclic_chain_then_each_bd_is_counted_once_per_run():
    assert bdgraph.chain_bytes([_bd(0, 100, next_bd=1), _bd(1, 50, next_bd=0)], 0) == 150
    assert bdgraph.chain_bytes([_bd(0, 100)], 7) is None


def test_given_a_looped_startio_and_unknown_rounds_then_ddr_bytes_are_none(tmp_path):
    ssa = [{"repeat_count": 1, "inside_scf_for": True, "loop_range": "%c0..%n"}]
    host = [{"line": 10, "kind": "createio", "bd_id": 0}, {"line": 20, "kind": "loop"},
            {"line": 21, "kind": "startio", "code": "__Runtime_startio(v1, v8, 0, 1);"}, {"line": 30, "kind": "loopend"}]
    f = bdgraph.features(_write(tmp_path, ssa, [_bd(0, 64)], host))
    assert f["ddr_bytes"]["mm2s"] is None
    assert bdgraph.features({"worklocal": str(tmp_path), "trip_count": 4})["ddr_bytes"]["mm2s"] == 4 * 4096


S12 = Path("/scratch/staff/huaj/tmp/s12dma_run/m2/aout/worklocal")


@pytest.mark.skipif(not (S12 / "schedule_view.json").exists(), reason="real build dir not present")
def test_given_the_s12_m2_build_then_ddr_bytes_equal_its_2x2048_i32_in_and_out():
    d = bdgraph.features(S12)["ddr_bytes"]
    assert d["mm2s"] == 2 * 2048 * 4 and d["s2mm"] == 2 * 2048 * 4


@pytest.mark.skipif(not (REAL / "schedule_view.json").exists(), reason="real build dir not present")
def test_given_the_maxwell_2k_mesh_build_then_shim_mm2s0_moves_4x4x4096_per_launch():
    # host.cc l.632/636: startio(..., 2, 4) inside the 0..4 loop, BD len 4096 -> 64 KiB per launch (Amit's count)
    d = bdgraph.features(REAL)["ddr_bytes"]
    mm2s0 = [c for c in d["channels"] if c["tile"] == (0, 0) and c["dir"] == "MM2S" and c["ch"] == 0][0]
    assert mm2s0["bytes"] == 4 * 4 * 4096
    assert d["mm2s"] == 8 * 65536 and d["s2mm"] == 4 * 16 * 256 * 4


# ---- DDR floor: bytes -> report-only lower bound from the board's stream + DDR column points

PRM = {"aie_clk_mhz": 1000.0, "dma_b_per_cyc": 8.0,
       "ddr_rd_cols_points": [[1, 4.0], [2, 8.0]], "ddr_wr_cols_points": [[1, 2.0], [4, 6.0]]}


def _ch(col, d, b):
    return {"tile": (col, 0), "dir": d, "ch": 0, "bytes": b}


def test_given_one_column_then_the_slower_of_stream_and_ddr_bounds_the_launch():
    # 8000 B in: stream 1000 cyc, rd 4 GB/s = 4 B/cyc -> 2000 cyc; 4000 B out: wr 2 B/cyc -> 2000 cyc
    fl = bdgraph.ddr_floor({"mm2s": 8000, "s2mm": 4000, "channels": [_ch(0, "MM2S", 8000), _ch(0, "S2MM", 4000)]}, PRM)
    assert fl["terms"] == {"stream": 1000.0, "rd": 2000.0, "wr": 2000.0}
    assert fl["cyc"] == 2000.0 and fl["ms"] == pytest.approx(0.002)


def test_given_more_active_columns_then_the_column_point_is_used_and_clamped():
    chs = [_ch(c, "MM2S", 8000) for c in range(3)]   # 3 cols -> first point >= 3 is absent -> last (8 GB/s)
    fl = bdgraph.ddr_floor({"mm2s": 24000, "s2mm": 0, "channels": chs}, PRM)
    assert fl["terms"]["rd"] == 3000.0 and "wr" not in fl["terms"] and fl["bound"] == "rd"


def test_given_unknown_bytes_or_missing_points_then_no_floor():
    assert bdgraph.ddr_floor({"mm2s": None, "s2mm": 0, "channels": []}, PRM) is None
    assert bdgraph.ddr_floor({"mm2s": 1, "s2mm": 0, "channels": [_ch(0, "MM2S", 1)]}, {"aie_clk_mhz": 1.0}) is None


def test_given_a_bdgraph_op_then_ddr_is_the_floor_and_stays_out_of_total(tmp_path, cycles_file):
    host = [{"line": 10, "kind": "createio", "bd_id": 0}, {"line": 11, "kind": "startio", "code": "__Runtime_startio(a, b, 0, 1);"}]
    d = _write(tmp_path, [{"repeat_count": 1}], [_bd(0, 64)], host)
    r = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "jit_mmio"}, "host": {"bd_us": 1.0}}],
                 params=PRM, cycles=cycles_file)
    assert r["ddr"] == pytest.approx(4096 / 4.0 / 1e6)
    assert "dma_time(bytes counted)" not in r["unpriced"] and r["total_ms"] == r["issue_ms"]


@pytest.mark.skipif(not (REAL / "schedule_view.json").exists(), reason="real build dir not present")
def test_given_the_maxwell_2k_mesh_build_on_palmyra213_then_the_floor_is_read_bound():
    from aiegraph.costmodel.estimate import board_params
    fl = bdgraph.ddr_floor(bdgraph.features(REAL)["ddr_bytes"], board_params("palmyra213"))
    # 512 KiB read over 4 shim cols at the 4-col point 9.301 GB/s = 56.4 us; stream 8192 cyc, write 7.8k cyc
    assert fl["bound"] == "rd" and fl["ms"] == pytest.approx(524288 / 9.301e6, rel=1e-3)


def test_given_no_bd_us_on_palmyra213_then_the_board_fit_prices_issue(tmp_path, cycles_file):
    d = _write(tmp_path, LOOPED, [_bd(0, 256)], HOST)
    r = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "aot_window"}}], cycles=cycles_file)
    op = r["ops"][0]
    assert op["bd_us"] == bdgraph.BD_US_FIT["palmyra213"] == 5.67 and op["bd_us_source"] == "board_fit"
    assert r["issue_ms"] == pytest.approx(17 * 5.67 / 1e3) and "host_bd_us" not in r["unpriced"]


def test_given_an_op_bd_us_then_it_wins_over_the_board_fit(tmp_path, cycles_file):
    d = _write(tmp_path, LOOPED, [_bd(0, 256)], HOST)
    r = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "aot_window"},
                   "host": {"bd_us": 2.0}}], cycles=cycles_file)
    assert r["ops"][0]["bd_us_source"] == "op" and r["issue_ms"] == pytest.approx(17 * 2.0 / 1e3)


def test_given_a_board_without_a_fit_and_no_bd_us_then_issue_stays_unpriced(tmp_path, cycles_file):
    d = _write(tmp_path, LOOPED, [_bd(0, 256)], HOST)
    prm = estimate_mod_board_params()
    r = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "aot_window"}}],
                 board="other_board", params=prm, cycles=cycles_file)
    assert r["issue_ms"] is None and "host_bd_us" in r["unpriced"]


def estimate_mod_board_params():
    import sys as _s
    return {k: v for k, v in _s.modules["aiegraph.costmodel.estimate"].board_params("palmyra213").items()
            if k not in ("bdgraph_bd_us", "host_bd_us")}


# --- run-length (run_b) factor on the DDR floor (P4f) ---

def test_given_mixed_read_runs_then_the_factor_is_byte_weighted_and_writes_are_ignored():
    chs = [dict(_ch(0, "MM2S", 3000), run_b=32), dict(_ch(1, "MM2S", 1000), run_b=128), dict(_ch(2, "S2MM", 9999), run_b=32)]
    r = bdgraph.ddr_runb({"ms": 2.0}, {"mm2s": 4000, "s2mm": 9999, "channels": chs}, {32: 1.4})
    assert r["factor"] == pytest.approx((3000 * 1.4 + 1000) / 4000) and r["ms"] == pytest.approx(2.0 * r["factor"])


def test_given_no_floor_or_no_factors_then_no_run_b_estimate():
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=32)]}
    assert bdgraph.ddr_runb(None, ddr, {32: 1.4}) is None and bdgraph.ddr_runb({"ms": 1.0}, ddr, None) is None


def test_given_a_32b_run_shim_read_on_palmyra213_then_ddr_runb_uses_the_board_fit_and_stays_out_of_total(tmp_path, cycles_file):
    host = [{"line": 10, "kind": "createio", "bd_id": 0}, {"line": 11, "kind": "startio", "code": "__Runtime_startio(a, b, 0, 1);"}]
    shim = [_bd(0, 4096, dim_strides=[4, 64], dim_wraps=[8, 128])]   # 8 words = 32 B contiguous, then jump
    d = _write(tmp_path, [{"repeat_count": 1}], [_bd(0, 64)], host, shim_bds=shim)
    r = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "jit_mmio"}}], cycles=cycles_file)
    op = r["ops"][0]
    assert op["bdgraph"]["ddr_bytes"]["channels"][0]["run_b"] == 32
    rb = op["ddr_runb"]
    assert op["bdgraph"]["ddr_bytes"]["channels"][0]["row_stride_b"] == 64
    assert rb["factor"] == pytest.approx(bdgraph.RUN_B_FACTOR_FIT["palmyra213"][32]["default"])   # 64 B stride: not a fast one
    assert rb["ms"] == pytest.approx(op["ddr"] * rb["factor"]) and r["total_ms"] == r["issue_ms"]
    assert rb["launch_ms"] == pytest.approx(op["issue_ms"] + rb["ms"])


def test_given_a_dma_bound_32b_launch_then_the_run_penalty_sets_launch_ms():
    # palmyra213, measured at 32 B runs: u_conv_1024x2880x320 10.174 ms, u_lin_1024x5760x320 17.972 ms (both floor > issue)
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=32)]}
    for floor, issue, meas in ((3.81, 2.73, 10.174), (7.61, 2.733, 17.972)):
        r = bdgraph.ddr_runb({"ms": floor}, ddr, bdgraph.RUN_B_FACTOR_FIT_P4I["palmyra213"], issue_ms=issue)
        assert r["regime"] == "dma" and r["launch_ms"] == pytest.approx(meas, rel=0.03)


def test_given_an_issue_bound_32b_launch_then_run_size_is_hidden_under_issue():
    # palmyra213 issue-bound 32 B kernels (floor < issue): u_lin_1024x{1280x320, 320x2560, 320x320, 320x960}
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=32)]}
    for floor, issue, meas in ((1.69, 2.73, 5.259), (1.59, 6.00, 8.310), (0.42, 2.73, 3.295), (0.59, 2.60, 3.413)):
        r = bdgraph.ddr_runb({"ms": floor}, ddr, bdgraph.RUN_B_FACTOR_FIT_P4I["palmyra213"], issue_ms=issue)
        assert r["regime"] == "issue" and r["launch_ms"] == pytest.approx(meas, rel=0.04)


def test_given_a_32b_launch_between_the_old_and_new_switch_then_it_is_dma_bound():
    # P4i: palmyra213 u_lin_1024x{1600,1920,2560}x320 (r = floor/issue .77/.93/1.24) measured 6.879/7.719/9.711 ms;
    # P4h (switch at r 1) put r .77 and .93 in the issue regime and missed u_lin_1024x1920x320 by -18%
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=32)]}
    for floor, meas in ((2.114, 6.879), (2.537, 7.719), (3.382, 9.711)):
        r = bdgraph.ddr_runb({"ms": floor}, ddr, bdgraph.RUN_B_FACTOR_FIT_P4I["palmyra213"], issue_ms=2.733)
        assert r["regime"] == "dma" and r["launch_ms"] == pytest.approx(meas, rel=0.03)


def test_given_factors_without_r_switch_then_the_regime_switches_at_floor_equal_issue():
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=32)]}
    fac = {32: {"issue": 1.4, "dma": 2.0}}
    assert bdgraph.ddr_runb({"ms": 0.9}, ddr, fac, issue_ms=1.0)["regime"] == "issue"
    assert bdgraph.ddr_runb({"ms": 1.1}, ddr, fac, issue_ms=1.0)["regime"] == "dma"
    assert bdgraph.ddr_runb({"ms": 0.9}, ddr, {**fac, "r_switch": 0.7}, issue_ms=1.0)["regime"] == "dma"


def test_given_a_run_b_without_a_fit_then_its_factor_is_1_in_both_regimes():
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=64)]}
    for issue in (0.5, 5.0):
        r = bdgraph.ddr_runb({"ms": 1.0}, ddr, bdgraph.RUN_B_FACTOR_FIT["palmyra213"], issue_ms=issue)
        assert r["factor"] == 1.0 and r["launch_ms"] == pytest.approx(issue + 1.0)


def test_given_no_issue_then_ddr_runb_has_no_launch_ms():
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=32)]}
    assert "launch_ms" not in bdgraph.ddr_runb({"ms": 1.0}, ddr, {32: 2.0})


UCONV = Path("/scratch/staff/huaj/tmp/costmodel_p0/u_conv_1024x2880x320/aout/worklocal")


@pytest.mark.skipif(not (UCONV / "schedule_view.json").exists(), reason="real build dir not present")
def test_given_the_u_conv_1024_build_then_every_shim_read_is_a_32b_run():
    from aiegraph.costmodel.estimate import board_params
    ddr = bdgraph.features(UCONV)["ddr_bytes"]
    rd = [c for c in ddr["channels"] if c["dir"] == "MM2S"]
    assert rd and {c["run_b"] for c in rd} == {32}
    rb = bdgraph.ddr_runb(bdgraph.ddr_floor(ddr, board_params("palmyra213")), ddr, bdgraph.RUN_B_FACTOR_FIT_P4I["palmyra213"],
                          issue_ms=480 * 5.67e-3)
    assert rb["regime"] == "dma" and rb["launch_ms"] == pytest.approx(10.174, rel=0.03)   # measured 32 B launch
    assert rb["launch_ms"] == pytest.approx(10.174, rel=0.05)   # measured 10.174 ms/launch (in-sample: f was fit here)


# --- P4j: issue grows with bytes; one run_b factor, no regime switch ---

def test_given_the_issue_fit_then_issue_is_configs_plus_a_share_of_the_rd_floor_plus_a_constant():
    fit = {"bd_ms": 5e-3, "rd_frac": 0.2, "c_ms": 0.1}
    prm = {"aie_clk_mhz": 1000.0}
    floor = {"ms": 3.0, "terms": {"rd": 2.0e6, "wr": 3.0e6}}          # rd 2 ms at 1 GHz
    assert bdgraph.issue_fit_ms(fit, 100, floor, prm) == pytest.approx(0.5 + 0.4 + 0.1)
    assert bdgraph.issue_fit_ms(fit, 100, None, prm) is None
    assert bdgraph.issue_fit_ms(fit, 100, {"ms": 1.0, "terms": {"wr": 1.0}}, prm) is None


def test_given_palmyra213_p4j_then_one_32b_factor_for_both_regimes():
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=32)]}
    for floor, issue in ((0.4, 2.7), (7.6, 3.9)):
        r = bdgraph.ddr_runb({"ms": floor}, ddr, bdgraph.RUN_B_FACTOR_FIT["palmyra213"], issue_ms=issue)
        assert r["factor"] == 1.87 and r["launch_ms"] == pytest.approx(issue + 1.87 * floor)


def test_given_a_32b_read_and_no_op_bd_us_then_estimate_prices_issue_with_the_p4j_fit(tmp_path, cycles_file):
    host = [{"line": 10, "kind": "createio", "bd_id": 0}, {"line": 11, "kind": "startio", "code": "__Runtime_startio(a, b, 0, 1);"}]
    shim = [_bd(0, 4096, dim_strides=[4, 64], dim_wraps=[8, 128])]
    d = _write(tmp_path, [{"repeat_count": 1}], [_bd(0, 64)], host, shim_bds=shim)
    r = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "jit_mmio"}}], cycles=cycles_file)
    op = r["ops"][0]
    fit = bdgraph.ISSUE_FIT["palmyra213"]
    rd_ms = op["ddr_floor"]["terms"]["rd"] / (estimate_mod_board_params()["aie_clk_mhz"] * 1e3)
    assert op["bd_us_source"] == "board_fit_p4j"
    assert op["issue_ms"] == pytest.approx(fit["bd_ms"] * op["bdgraph"]["bd_configs_per_launch"] + fit["rd_frac"] * rd_ms + fit["c_ms"])
    r2 = estimate([{"kernel": "x", "bdgraph": {"worklocal": str(d), "bd_write_path": "jit_mmio"}, "host": {"bd_us": 2.0}}],
                  cycles=cycles_file)
    assert r2["ops"][0]["bd_us_source"] == "op"     # an explicit bd_us keeps the plain cfg x bd_us form


RUNB = Path("/scratch/staff/huaj/tmp")


@pytest.mark.parametrize("build,meas", [("costmodel_runb2/u_lin_1024x5760x320_tkw8", 17.972),
                                        ("costmodel_runb4/u_lin_2048x960x320_tkw8", 9.617),
                                        ("costmodel_p0/u_conv_1024x2880x320", 10.174)])
def test_given_palmyra213_32b_builds_then_p4j_launch_ms_is_within_3pct(build, meas):
    wl = RUNB / build / "aout/worklocal"
    if not (wl / "schedule_view.json").exists():
        pytest.skip("real build dir not present")
    op = estimate([{"kernel": build, "bdgraph": {"worklocal": str(wl), "bd_write_path": "jit_mmio"}}])["ops"][0]
    assert op["ddr_runb"]["launch_ms"] == pytest.approx(meas, rel=0.03)    # P4i: -11.5% on u_lin_2048x960x320


# --- P4k: at 32 B runs the factor follows the DDR row stride (empirical, palmyra213) ---

def test_given_a_strided_bd_then_row_stride_is_its_second_dim_stride_and_a_plain_bd_has_none():
    assert bdgraph.row_stride({"len": 40960, "dim_strides": [4, 2560, 32], "dim_wraps": [8, 16, 80]}) == 2560
    assert bdgraph.row_stride({"len": 4096, "dim_strides": [4], "dim_wraps": [1024]}) is None
    assert bdgraph.row_stride({"len": 4096}) is None and bdgraph.row_stride(None) is None


@pytest.mark.parametrize("stride,f", [(640, 1.33), (1312, 1.33), (2560, 1.33),
                                      (1920, 1.87), (3200, 1.87), (5120, 1.87), (None, 1.87), (2576, 1.87)])
def test_given_palmyra213_p4k_then_a_32b_read_takes_the_fast_factor_only_at_a_fast_stride(stride, f):
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=32, row_stride_b=stride)]}
    r = bdgraph.ddr_runb({"ms": 1.0}, ddr, bdgraph.RUN_B_FACTOR_FIT["palmyra213"], issue_ms=2.0)
    assert r["factor"] == f and r["launch_ms"] == pytest.approx(2.0 + f)


def test_given_a_by_stride_fit_then_the_factor_is_byte_weighted_over_channels_with_different_strides():
    fac = {32: {"by_stride": {640: 1.0}, "default": 2.0}}
    chs = [dict(_ch(0, "MM2S", 3000), run_b=32, row_stride_b=640), dict(_ch(1, "MM2S", 1000), run_b=32, row_stride_b=1920)]
    r = bdgraph.ddr_runb({"ms": 1.0}, {"mm2s": 4000, "s2mm": 0, "channels": chs}, fac)
    assert r["factor"] == pytest.approx((3000 * 1.0 + 1000 * 2.0) / 4000)


def test_given_a_16b_read_then_p4k_keeps_the_single_16b_factor():
    ddr = {"mm2s": 1, "s2mm": 0, "channels": [dict(_ch(0, "MM2S", 1), run_b=16, row_stride_b=2576)]}
    assert bdgraph.ddr_runb({"ms": 1.0}, ddr, bdgraph.RUN_B_FACTOR_FIT["palmyra213"])["factor"] == 2.53


@pytest.mark.parametrize("build,meas", [("costmodel_runb2/u_lin_1024x1280x320_tkw8", 5.259),     # P4j +15.9%
                                        ("costmodel_runb7/u_lin_512x640x320_tkw8", 1.758),       # P4j +18.5%
                                        ("costmodel_runb5/u_lin_2048x1280x320_tkw8", 9.941),     # P4j +16.3%
                                        ("costmodel_runb7/u_lin_512x960x320_tkw8", 2.377)])      # slow stride
def test_given_palmyra213_fast_and_slow_stride_builds_then_p4k_launch_ms_is_within_7pct(build, meas):
    hit = sorted(RUNB.glob("costmodel_runb*/" + build.split("/", 1)[1] + "/aout/worklocal/schedule_view.json"))
    if not hit:
        pytest.skip("real build dir not present")
    wl = hit[0].parent
    op = estimate([{"kernel": build, "bdgraph": {"worklocal": str(wl), "bd_write_path": "jit_mmio"}}])["ops"][0]
    assert op["ddr_runb"]["launch_ms"] == pytest.approx(meas, rel=0.07)
