###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.kaccbench: the gemm_kacc names a stand-alone bench uses (unet-decl P4d: they were the example's
unet_aie.py, which delegated to aiegraph.cnn / aiegraph.kincore; d5e9f15 has it).

    PYTHONPATH=src python3 -m pytest -q src/aiegraph/test/test_kaccbench.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from aiegraph import cnn, kaccbench as kb, kincore  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402


@pytest.mark.parametrize("ch,k,px", [(320, 320, 1024), (960, 320, 512), (64, 64, 64), (1280, 2560, 64)])
def test_geom_and_name_are_the_generic_ones(ch, k, px):
    assert kb.kacc_geom(ch, k, px) == cnn.kacc_geom(ch, k, px)
    assert kb.kacc_name(ch, k, px) == kincore.kacc_name(ch, k, px)
    assert kb.kacc_name(ch, k, px).startswith("gemm_kacc<")


@pytest.mark.parametrize("ch,k,px", [(100, 64, 64), (64, 64, 100), (64, 96, 64)])
def test_kacc_geom_rejects_no_walk(ch, k, px):
    """a shape without a walk: NoWalk, catchable both as AssertionError (bench split search) and AiegraphError"""
    with pytest.raises(AssertionError):
        kb.kacc_geom(ch, k, px)
    with pytest.raises(AiegraphError):
        kb.kacc_geom(ch, k, px)
    with pytest.raises(kb.NoWalk):
        kb.kacc_name(ch, k, px)


def test_ctrlpkt_mode_rejects_other_knobs():
    assert kb.ctrlpkt_mode({}) == "aot"
    assert kb.ctrlpkt_mode({kb.KCO_ENV: "0"}) == "aot"
    assert kb.ctrlpkt_mode({kb.KCO_ENV: "1"}) == "aot-kco"
    assert kb.ctrlpkt_mode({"UNRELATED_CTRLPKT_KCO": "1"}) == "aot"   # counter-example: only the aiegraph knob


def test_module_constants():
    assert kb.KACC_KERNEL == kincore.KACC_KERNEL
    assert kb.CTRLPKT_PRAGMAS == kincore.ctrlpkt_head(kb.CTRLPKT_MODE)
    assert kb.CTRLPKT_NPRAGMAS == kincore.ctrlpkt_npragmas(kb.CTRLPKT_MODE) > 0
    assert kb.CTRLPKT_KCO == (kb.CTRLPKT_MODE == "aot-kco")
