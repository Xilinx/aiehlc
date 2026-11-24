"""costmodel as one package: tools/ + params/ moved from script/perf, old paths still work, CLI, examples."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
PKG = SRC / "costmodel"
sys.path.insert(0, str(SRC))

from costmodel import cli, tools  # noqa: E402
from costmodel.estimate import PARAMS_DIR, TOOLS_DIR  # noqa: E402

PY = sys.executable
ROW = {"kernel": "u_lin_1024x320x320", "recipe": "kacc_gemm_bf16", "M": 1024, "K": 320, "N": 320,
       "instance": {"ch": 320, "k": 320, "px": 1024, "ntile": 16, "S": 5, "tkw": 8, "kch": 16, "nkc": 20, "tr": 1},
       "core_cyc": 694124}


def _run(args, **kw):
    return subprocess.run([PY, "-I"] + [str(a) for a in args], capture_output=True, text=True, cwd=str(ROOT), **kw)


def test_given_the_package_then_tools_and_params_live_inside_it():
    assert TOOLS_DIR == PKG / "tools" and PARAMS_DIR == PKG / "params"
    assert len(tools.names()) == 29 and "n_fit" in tools.names() and "build_fresh" not in tools.names()
    assert (PARAMS_DIR / "palmyra213.json").is_file()


def test_given_an_old_script_perf_path_then_every_moved_tool_has_a_shim():
    for n in tools.names():
        shim = ROOT / "script" / "perf" / (n + ".py")
        assert shim.is_file() and ("costmodel\" / \"tools\" / \"%s.py\"" % n) in shim.read_text()


def test_given_the_old_params_path_then_it_resolves_to_the_package_params():
    old = ROOT / "script" / "perf" / "params"
    assert old.is_symlink() and old.resolve() == PARAMS_DIR.resolve()


def test_given_the_frozen_md5_files_then_every_pinned_file_matches():
    assert cli.check_frozen() == []
    for mf in PARAMS_DIR.glob("*.md5"):
        assert "script/perf" not in mf.read_text()


def test_given_an_old_path_shim_when_run_then_the_moved_tool_runs():
    r = _run([ROOT / "script" / "perf" / "tsv_cmp.py", "--selftest"])
    assert r.returncode == 0 and "TSVCMP SELFTEST PASS" in r.stdout


def test_given_an_old_path_shim_when_imported_then_it_is_the_moved_module():
    spec = importlib.util.spec_from_file_location("kd_shim", ROOT / "script" / "perf" / "k_diag.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert Path(m.__file__) == TOOLS_DIR / "k_diag.py" and m.HERE == TOOLS_DIR and callable(m.khats_predict)


def test_given_the_cli_when_estimating_one_kernel_then_k_is_priced_and_l_n_unpriced(tmp_path, capsys):
    cyc = tmp_path / "cycles.json"
    cyc.write_text(json.dumps({"kernels": [ROW]}))
    assert cli.main(["estimate", "--kernel", "u_lin_1024x320x320", "--cycles", str(cyc)]) == 0
    out = capsys.readouterr().out
    assert "ESTIMATE board=palmyra213 launches=1 total_ms=0.4716 wait=0.4716" in out
    assert "unpriced (counted as 0 in total_ms): glue, host_bd_us(default 5.0), l" in out


def test_given_python_dash_m_when_listing_then_the_cli_answers():
    r = subprocess.run([PY, "-m", "costmodel", "tools"], capture_output=True, text=True, cwd=str(SRC))
    assert r.returncode == 0 and r.stdout.split() == tools.names()


def test_given_example_a_when_run_then_m5_prediction_is_within_20pct_of_the_committed_log():
    r = _run([PKG / "example" / "single_kernel" / "estimate_kernel.py"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "EXAMPLE_A kernel=u_lin_1024x320x320 meas_ms=0.5166 pred_m5_ms=0.5250 rel=+0.016 PASS" in r.stdout


_NET_BUILD = "/scratch/staff/huaj/tmp/costmodel_m40/resnet18_off"


@pytest.mark.skipif(not os.path.isdir(_NET_BUILD), reason="archived resnet18 build dir not present")
def test_given_example_b_when_run_then_resnet18_matches_golden_and_board():
    r = _run([PKG / "example" / "whole_model" / "estimate_net.py", "--rows", "0"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "EXAMPLE_B net=resnet18 pts=42 r2=1.000 maxrel=0.040 PASS" in r.stdout


@pytest.mark.skipif(not os.path.isdir(_NET_BUILD), reason="archived build dirs not present")
def test_given_example_c_when_run_without_refit_then_the_board_best_is_kept():
    r = _run([PKG / "example" / "offline_tuning_proposal" / "tune_loop.py", "--no-refit"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TUNE nets=3 candidates=6 board_runs=3 saved=3 best_kept=3/3" in r.stdout
