###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiegraph.kaccbench bench-only walk knobs (cost-model probes; was example test_kacc_kp_pad.py on unet_aie.kacc_geom).

KACC_KP_PAD legality (opt-in): kp is padded only to a legal value. runb12 (palmyra213): 1024x1600x320 / 1024x1920x320
with kp 2048 failed to build -- tr=1, the a port is [px][kp/2] and kp/2 == px is the square-port GemmSpace mis-size.
KACC_TR_TKW_CAP caps the transposed b window; AIEGRAPH_CTRLPKT=off drops the control-packet pragmas.
"""
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aiegraph import cnn, kaccbench as kb, kincore  # noqa: E402
from aiegraph.errors import AiegraphError  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (kb.KP_PAD_ENV, kb.TKW_CAP_ENV, kb.CTRLPKT_ENV, kb.KCO_ENV):
        monkeypatch.delenv(v, raising=False)


@pytest.mark.parametrize("ch,k,px,pad,kp", [
    (320, 960, 1024, 0, 960),       # default off
    (320, 960, 1024, 1024, 1024),   # RESULT11: tr=1, kp/2 512 != px 1024 -> padded (board KCMP PASS)
    (320, 960, 512, 1024, 1024),    # tr=0
    (320, 1600, 1024, 1664, 1664),  # runb12 kp 1664 (board KCMP PASS)
    (320, 1600, 1024, 2048, 1600),  # tr=1, kp/2 == px -> illegal, not applied
    (320, 1920, 1024, 2048, 1920),  # same
    (320, 1920, 512, 2048, 2048),   # tr=0, legal (runb12b KCMP PASS)
    (320, 960, 1024, 968, 960),     # not a multiple of 2*tkw
    (320, 960, 1024, 640, 960),     # pad below kp is ignored
    (320, 2048, 1024, 0, 2064),     # default rule: tr=1, k/2 == px -> +2*tkw (tkw 8; peer:1120164720-61cc6e)
    (320, 2048, 512, 0, 2048),      # tr=0, k/2 != ch -> no pad
])
def test_kp_pad(monkeypatch, ch, k, px, pad, kp):
    if pad:
        monkeypatch.setenv(kb.KP_PAD_ENV, str(pad))
    g = kb.kacc_geom(ch, k, px)
    assert g["kp"] == kp
    assert g["nkc"] * g["tkw"] * 2 == g["kp"]


def test_tr_tkw_floor_raises_nowalk_not_zerodiv():
    """ch/4 > KACC_TR_TKW_CAP halved tkw to 0 -> ZeroDivisionError (512px GEGLU 5120x640 at px 1024)."""
    with pytest.raises(AssertionError):          # NoWalk is an AssertionError too
        kb.kacc_geom(5120, 640, 1024)
    with pytest.raises(kb.NoWalk):
        kb.kacc_geom(5120, 640, 1024)


def test_tr_tkw_cap_env_widens_the_window(monkeypatch):
    assert kb.kacc_geom(320, 960, 1024)["tkw"] == 8
    monkeypatch.setenv(kb.TKW_CAP_ENV, "2048")
    assert kb.kacc_geom(320, 960, 1024)["tkw"] == 16


def test_kacc_split_falls_back_to_px_split():
    s, px = kincore.kacc_split(5120, 640, 1024)
    assert (s, px) == (2, 512)
    assert kb.kacc_geom(5120, 640, px)["tr"] == 0


def test_default_bench_walk_equals_the_network_walk_except_square_tr():
    """without knobs the bench walk is cnn.kacc_geom, except the extra pad of a transposed px == k/2 launch"""
    n = 0
    for ch, k, px in itertools.product((64, 128, 320, 640, 1280, 2560), (64, 320, 640, 1280, 2048, 2560),
                                       (64, 256, 512, 1024, 4096)):
        try:
            want = cnn.kacc_geom(ch, k, px)
        except AiegraphError:
            with pytest.raises(kb.NoWalk):
                kb.kacc_geom(ch, k, px)
            continue
        got = kb.kacc_geom(ch, k, px)
        if want["tr"] and px == k // 2 and ch != k // 2:
            assert got["kp"] == k + 2 * got["tkw"] and want["kp"] == k
            assert {x: got[x] for x in got if x not in ("kp", "nkc")} == \
                {x: want[x] for x in want if x not in ("kp", "nkc")}
        else:
            assert got == want, (ch, k, px)
            assert kb.kacc_name(ch, k, px) == kincore.kacc_name(ch, k, px)
            n += 1
    assert n > 50


def test_ctrlpkt_mode_off_kco_and_default():
    assert kb.ctrlpkt_mode({}) == "aot"
    assert kb.ctrlpkt_mode({kb.KCO_ENV: "1"}) == "aot-kco"
    assert kb.ctrlpkt_mode({kb.CTRLPKT_ENV: "off", kb.KCO_ENV: "1"}) == "off"
    assert kincore.ctrlpkt_head("off") == "" and kincore.ctrlpkt_npragmas("off") == 0
