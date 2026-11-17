###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""flowcheck against two board-verified builds of simplegemm_kernel_port.cc.

flowcheck_data/good     -- the normal build (board: PASS, 4096/4096)
flowcheck_data/repaired -- AIEHLC_DMAPHOP_IR_IN with B's col-0 / col-1 shim columns
                           swapped (board: wrong data; RoutingHWVerify: 0 errors).
host.cc is reduced to its "Create IO" comment lines, all flowcheck reads.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import flowcheck  # noqa: E402

DATA = HERE / "flowcheck_data"


def test_good_build_is_consistent():
    r = flowcheck.check(DATA / "good")
    assert r["ok"], r["errors"]
    assert len(r["flows"]) == 12  # 4 B cols + 4 A rows push, 4 C row gathers pull


def test_repaired_build_names_the_board_failure():
    r = flowcheck.check(DATA / "repaired")
    assert not r["ok"]
    e = "\n".join(r["errors"])
    # B col 0 (now from shim col 1) misses its column and lands on row 4's A port ...
    assert "push_0 from shim(1,0).mm2s0: does not reach [('core', 0, 3, 1)" in e
    assert "push_0 from shim(1,0).mm2s0: also reaches [('core', 0, 4, 0)" in e
    # ... B col 1 also feeds column 0, the A rows slide down one mesh row ...
    assert "push_1 from shim(0,0).mm2s0: also reaches [('core', 0, 3, 1)" in e
    assert "push_6 from shim(2,0).mm2s1: also reaches [('core', 0, 6, 0)" in e
    # ... and A row 3's shim port is never opened.
    assert "push_7: shim (3,0) MM2S ch1 -> SOUTH 7 is not enabled in routing.cc" in e


def test_cli_exit_code(capsys):
    sys.argv = ["flowcheck", str(DATA / "repaired")]
    assert flowcheck.main() == 1
    assert "flowcheck: FAIL" in capsys.readouterr().out
    sys.argv = ["flowcheck", str(DATA / "good")]
    assert flowcheck.main() == 0
