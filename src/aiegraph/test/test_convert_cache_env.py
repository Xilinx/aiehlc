###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""convert cache (P3 of -fast): which dir is used, and AIEHLC_FAST does not change the key.

x86 only, no TVM, no conversion.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph import convert_cache as cc  # noqa: E402


def _clear(mp):
    for k in ("AIEGRAPH_CONVERT_CACHE", "AIEHLC_FAST", "XDG_CACHE_HOME"):
        mp.delenv(k, raising=False)


def test_off_by_default(monkeypatch):
    _clear(monkeypatch)
    assert cc.cache_dir() is None
    monkeypatch.setenv("AIEHLC_FAST", "0")
    assert cc.cache_dir() is None


def test_fast_uses_default_dir(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("AIEHLC_FAST", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cc.cache_dir() == (tmp_path / "aiehlc" / "convert").resolve()
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    assert cc.cache_dir() == (tmp_path / "h" / ".cache" / "aiehlc" / "convert").resolve()


def test_explicit_env_wins(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("AIEHLC_FAST", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("AIEGRAPH_CONVERT_CACHE", str(tmp_path / "mine"))
    assert cc.cache_dir() == (tmp_path / "mine").resolve()


def test_fast_flag_not_in_key(monkeypatch, tmp_path):
    _clear(monkeypatch)
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"not really onnx")
    model = SimpleNamespace(onnx_path=onnx)
    repo = SRC.parent
    k0, _ = cc.key(model, None, None, None, repo)
    monkeypatch.setenv("AIEHLC_FAST", "1")
    k1, parts = cc.key(model, None, None, None, repo)
    assert k0 == k1 and "AIEHLC_FAST" not in parts["env"]
    monkeypatch.setenv("AIEGRAPH_WDTYPE", "int8")     # a real knob still changes the key
    k2, _ = cc.key(model, None, None, None, repo)
    assert k2 != k1
