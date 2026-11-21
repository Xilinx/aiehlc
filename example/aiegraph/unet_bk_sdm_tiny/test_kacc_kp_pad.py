"""KACC_KP_PAD legality (costmodel stride probe, opt-in): unet_aie.kacc_geom pads kp only to a legal value.
runb12 (palmyra213): 1024x1600x320 / 1024x1920x320 with kp 2048 failed to build -- tr=1, the a port is
[px][kp/2] and kp/2 == px is the square-port GemmSpace mis-size. The guard now skips that pad.
Run: python3 -m pytest -q example/aiegraph/unet_bk_sdm_tiny/test_kacc_kp_pad.py
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import unet_aie as u  # noqa: E402


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
])
def test_kp_pad(monkeypatch, ch, k, px, pad, kp):
    if pad:
        monkeypatch.setenv("KACC_KP_PAD", str(pad))
    else:
        monkeypatch.delenv("KACC_KP_PAD", raising=False)
    g = u.kacc_geom(ch, k, px)
    assert g["kp"] == kp
    assert g["nkc"] * g["tkw"] * 2 == g["kp"]


def test_tr_tkw_floor_raises_assert_not_zerodiv(monkeypatch):
    """ch/4 > KACC_TR_TKW_CAP halved tkw to 0 -> ZeroDivisionError (512px GEGLU 5120x640 at px 1024)."""
    monkeypatch.delenv("KACC_TR_TKW_CAP", raising=False)
    with pytest.raises(AssertionError):
        u.kacc_geom(5120, 640, 1024)


def test_kacc_split_falls_back_to_px_split(monkeypatch):
    monkeypatch.delenv("KACC_TR_TKW_CAP", raising=False)
    s, px = u.kacc_split(5120, 640, 1024)
    assert (s, px) == (2, 512)
    assert u.kacc_geom(5120, 640, px)["tr"] == 0
