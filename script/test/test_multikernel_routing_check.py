###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""GAP-MKROUTE: the multi-kernel routing-topology guard on two real aiehlc host.cc plan excerpts (P2 probes:
fap6b = gemm_kacc + FA with row-broadcast K/V -> hung on palmyra213; fap7 = FA in gemm_kacc topology -> PASS)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import multikernel_routing_check as mk  # noqa: E402


def test_mismatch_fails():
    ok, topo, groups = mk.check((HERE / "fixtures_mkroute" / "mismatch_host_plans.txt").read_text())
    assert not ok and len(topo) == 4 and len(groups) == 2
    assert mk.main([str(HERE / "fixtures_mkroute" / "mismatch_host_plans.txt")]) == 1


def test_match_passes():
    ok, topo, groups = mk.check((HERE / "fixtures_mkroute" / "match_host_plans.txt").read_text())
    assert ok and len(topo) == 4 and len(groups) == 1
    assert mk.main([str(HERE / "fixtures_mkroute" / "match_host_plans.txt")]) == 0


def test_single_kernel_ok():
    t = (HERE / "fixtures_mkroute" / "match_host_plans.txt").read_text().splitlines()[:4]
    assert mk.check("\n".join(t))[0]
