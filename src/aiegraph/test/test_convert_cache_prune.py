###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""convert cache size cap: LRU prune after store, restore marks used, cap knob not in key.

x86 only, no TVM, no conversion.
"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph import convert_cache as cc  # noqa: E402

MB = 1 << 20


def _env(mp, tmp_path, cap_gb):
    for k in ("AIEHLC_FAST", "XDG_CACHE_HOME"):
        mp.delenv(k, raising=False)
    mp.setenv("AIEGRAPH_CONVERT_CACHE", str(tmp_path / "cache"))
    if cap_gb is None:
        mp.delenv("AIEGRAPH_CONVERT_CACHE_MAX_GB", raising=False)
    else:
        mp.setenv("AIEGRAPH_CONVERT_CACHE_MAX_GB", str(cap_gb))


def _wd(tmp_path, name, mb):
    wd = tmp_path / "wd" / name
    wd.mkdir(parents=True)
    (wd / "blob.bin").write_bytes(b"\1" * (mb * MB))
    (wd / "Makefile").write_text(f"W={wd}\n")
    return wd


def _age(key, tmp_path, secs):
    used = tmp_path / "cache" / key / "used"
    t = time.time() - secs
    os.utime(used, (t, t))


def test_default_cap_and_off(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, None)
    assert cc.max_bytes() == 20 << 30
    monkeypatch.setenv("AIEGRAPH_CONVERT_CACHE_MAX_GB", "0")
    assert cc.max_bytes() is None
    monkeypatch.setenv("AIEGRAPH_CONVERT_CACHE_MAX_GB", "bogus")
    assert cc.max_bytes() == 20 << 30


def test_lru_prune_keeps_recent_and_new(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, 5 / 1024)                     # 5 MB cap
    for i, age in ((0, 300), (1, 200)):
        cc.store(f"k{i}", {}, _wd(tmp_path, f"w{i}", 2), None)
        _age(f"k{i}", tmp_path, age)
    meta = json.loads((tmp_path / "cache" / "k0" / "meta.json").read_text())
    assert meta["bytes"] >= 2 * MB
    # k0 is the oldest, but a restore makes it the most recent
    assert cc.restore("k0", tmp_path / "r0", None) is not None
    cc.store("k2", {}, _wd(tmp_path, "w2", 2), None)          # 6 MB > 5 MB -> evict k1
    left = sorted(p.name for p in (tmp_path / "cache").iterdir() if not p.name.startswith("."))
    assert left == ["k0", "k2"]


def test_new_entry_bigger_than_cap_is_kept(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, 1 / 1024)                     # 1 MB cap
    cc.store("old", {}, _wd(tmp_path, "a", 1), None)
    cc.store("big", {}, _wd(tmp_path, "b", 3), None)
    left = sorted(p.name for p in (tmp_path / "cache").iterdir() if not p.name.startswith("."))
    assert left == ["big"]
    assert cc.restore("big", tmp_path / "r", None) is not None


def test_no_cap_keeps_everything(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, 0)
    for i in range(3):
        cc.store(f"k{i}", {}, _wd(tmp_path, f"w{i}", 2), None)
    assert len([p for p in (tmp_path / "cache").iterdir() if not p.name.startswith(".")]) == 3


def test_stale_tmp_removed(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, 0)
    cd = tmp_path / "cache"
    old = cd / ".dead.1.tmp"
    new = cd / ".live.2.tmp"
    old.mkdir(parents=True)
    new.mkdir()
    t = time.time() - 2 * 86400
    os.utime(old, (t, t))
    cc.prune()
    assert not old.exists() and new.exists()


def test_cap_knob_not_in_key(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, None)
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    model = SimpleNamespace(onnx_path=onnx)
    k0, _ = cc.key(model, None, None, None, SRC.parent)
    monkeypatch.setenv("AIEGRAPH_CONVERT_CACHE_MAX_GB", "3")
    k1, parts = cc.key(model, None, None, None, SRC.parent)
    assert k0 == k1 and "AIEGRAPH_CONVERT_CACHE_MAX_GB" not in parts["env"]


def test_restore_after_prune_is_miss(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, 0)
    cc.store("k", {}, _wd(tmp_path, "w", 1), None)
    import shutil
    shutil.rmtree(tmp_path / "cache" / "k" / "tree")          # half-deleted entry
    assert cc.restore("k", tmp_path / "r", None) is None
    assert not (tmp_path / "r").exists()
