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
