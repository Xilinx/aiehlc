###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""costmodel.estimate -- the public cost-layer API (cost plan P4a).

Given / when / then in the test names:  an op group in Steve's schema -> per-term ms + the list of unpriced terms.
"""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph import aiperf  # noqa: E402
from costmodel import des, estimate  # noqa: E402
from costmodel.estimate import board_params  # noqa: E402

# u_lin_1024x320x320 from aie2ps_kernel_cycles.json (K1b, Maxwell); inline so the test needs no external file
INST = {"ch": 320, "k": 320, "px": 1024, "ntile": 16, "S": 5, "tkw": 8, "kch": 16, "nkc": 20, "tr": 1}
CORE = 694124
TABLE = [{"kernel": "u_lin_1024x320x320", "recipe": "kacc_gemm_bf16", "M": 1024, "K": 320, "N": 320,
          "instance": INST, "core_cyc": CORE}]


@pytest.fixture
def cycles_file(tmp_path):
    import json
    p = tmp_path / "cycles.json"
    p.write_text(json.dumps({"kernels": TABLE}))
    return str(p)


def _op(**kw):
    op = {"kernel": "u_lin_1024x320x320", "recipe": "kacc_gemm_bf16", "M": 1024, "K": 320, "N": 320,
          "bytes_in": 1024 * 320 * 2, "bytes_out": 1024 * 320 * 2, "resident": False, "host": {"bd_us": 4.0}}
    op.update(kw)
    return op


def test_given_a_kacc_op_when_estimated_then_wait_equals_the_des_prediction(cycles_file):
    prm = board_params("palmyra213")
    e = estimate([_op()], params=prm, cycles=cycles_file)
    ref = des.predict_launch({"instance": INST, "host": {"bd_us": 4.0}}, prm, {"core_cyc": CORE})
    assert e["wait"] == pytest.approx(ref["wait_ms"]) and e["compute"] == pytest.approx(ref["compute_ms"])
    assert e["ddr"] == pytest.approx(ref["ddr_ms"]) and e["bound"] == ref["bound"]
    assert e["total_ms"] == pytest.approx(ref["wait_ms"])          # no L features -> wait only
    for k in ("kload", "cfg", "start", "wait", "glue", "compute", "ddr", "total_ms", "bound", "launches"):
        assert k in e
    assert "l" in e["unpriced"] and "glue" in e["unpriced"]


def test_given_only_recipe_and_mkn_when_estimated_then_the_instance_is_looked_up(cycles_file):
    a = estimate([_op()], cycles=cycles_file)
    b = estimate([_op(kernel=None)], cycles=cycles_file)
    assert a["wait"] == pytest.approx(b["wait"]) and a["wait"] > 0


def test_given_launches_when_estimated_then_terms_scale(cycles_file):
    one = estimate({"ops": [_op()]}, cycles=cycles_file)
    three = estimate({"ops": [_op(), _op()], "launches": 3}, cycles=cycles_file)
    assert three["wait"] == pytest.approx(6 * one["wait"]) and three["launches"] == 3


def test_given_l_features_when_estimated_then_l_ms_is_added(cycles_file):
    calls = {"bdcfg": 400, "startio": 30, "core_reset": 0}
    words = {"kload_w": 100}
    e = estimate([_op(l={"cfg": "aot", "calls": calls, "words": words})], cycles=cycles_file)
    assert e["l_ms"] is not None and "l" not in e["unpriced"]
    assert e["total_ms"] == pytest.approx(e["wait"] + e["l_ms"])


def test_given_a_resident_op_then_resident_is_flagged_unpriced(cycles_file):
    assert "resident" in estimate([_op(resident=True)], cycles=cycles_file)["unpriced"]


def test_given_a_bdgraph_op_without_build_json_then_it_says_which_file_is_missing(cycles_file, tmp_path):
    with pytest.raises(FileNotFoundError, match="dfscheduleprovenancemap"):
        estimate([_op(bdgraph={"worklocal": str(tmp_path)})], cycles=cycles_file)


def test_given_an_unknown_recipe_or_kernel_then_it_fails_loudly(cycles_file):
    with pytest.raises(NotImplementedError, match="no K model"):
        estimate([_op(recipe="softmax_bf16")], cycles=cycles_file)
    with pytest.raises(KeyError, match="no kacc instance"):
        estimate([_op(kernel="nope", M=1, K=2, N=3)], cycles=cycles_file)


def test_given_an_unknown_board_then_it_fails_loudly(cycles_file):
    with pytest.raises(FileNotFoundError, match="no params for board"):
        estimate([_op()], board="nosuchboard", cycles=cycles_file)


def _facts(**extra):
    f = {"src": {"mesh": [4, 4], "backbone": False},
         "elf": {"text": 400, "data": 40, "bss": 4000},
         "host": {"ctrl_elf_load": False, "ctrl_calls": False, "calls": {}, "routing_calls": 0, "teardown": False},
         "aiehlc": {"skip_bss": False}, "run": {"measured_ms": 1.0}}
    f.update(extra)
    return f


def test_given_facts_without_a_group_then_aiperf_cost_model_is_unchanged():
    m = aiperf.cost_model(_facts())
    assert m["estimate"] is None and m["elf_words_per_core"] == 1110


def test_given_facts_with_a_group_then_aiperf_cost_model_forwards_to_estimate(monkeypatch, cycles_file):
    em = sys.modules["costmodel.estimate"]  # the package attr `estimate` is the function
    monkeypatch.setattr(em, "CYCLES", cycles_file)
    m = aiperf.cost_model(_facts(group=[_op()]))
    assert m["estimate"]["wait"] == pytest.approx(estimate([_op()], cycles=cycles_file)["wait"])
