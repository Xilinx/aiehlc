###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""F10 convert cache hardlink restore: big files linked + read-only, arm_build/ and
small files copied, relocation never writes through a link, LINK=0 copies all.

x86 only, no TVM, no conversion.
"""
import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from aiegraph import convert_cache as cc  # noqa: E402

BIG = cc._LINK_MIN + 4096


def _seed(mp, tmp_path, link="1"):
    for k in ("AIEHLC_FAST", "XDG_CACHE_HOME", "AIEGRAPH_CONVERT_CACHE_MAX_GB"):
        mp.delenv(k, raising=False)
    mp.setenv("AIEGRAPH_CONVERT_CACHE", str(tmp_path / "cache"))
    mp.setenv(cc.LINK_ENV, link)
    wd = tmp_path / "old_wd"
    (wd / "arm_build").mkdir(parents=True)
    (wd / "m.onnx").write_bytes(b"\1\0" * (BIG // 2))
    (wd / "arm_build" / "weights.bin").write_bytes(b"\2" * BIG)
    (wd / "small.txt").write_text(f"path {wd}\n")
    big_txt = f"see {wd}\n".encode() * (BIG // 20)     # >= _LINK_MIN, < _TEXT_MAX, relocatable
    (wd / "big.mk").write_bytes(big_txt)
    cc.store("k1", {}, wd, None)
    return cc.cache_dir() / "k1" / "tree", big_txt


def test_link_restore(monkeypatch, tmp_path):
    tree, big_txt = _seed(monkeypatch, tmp_path)
    new = tmp_path / "new_wd"
    meta = cc.restore("k1", new, None)
    assert meta["linked"] == 2          # m.onnx + big.mk; relocation then unlinks big.mk
    on, cn = os.stat(new / "m.onnx"), os.stat(tree / "m.onnx")
    assert on.st_ino == cn.st_ino and not (cn.st_mode & 0o222)
    # arm_build/weights.bin is copied: in-place rewrite leaves the cache alone
    assert os.stat(new / "arm_build" / "weights.bin").st_ino != os.stat(tree / "arm_build" / "weights.bin").st_ino
    with open(new / "arm_build" / "weights.bin", "r+b") as f:
        f.write(b"X")
    assert (tree / "arm_build" / "weights.bin").read_bytes()[:1] == b"\2"
    # big text file was linked, then relocated by rename: cache keeps the old path
    assert (new / "big.mk").read_bytes() == big_txt.replace(str(tmp_path / "old_wd").encode(), str(new).encode())
    assert (tree / "big.mk").read_bytes() == big_txt
    assert os.stat(new / "big.mk").st_ino != os.stat(tree / "big.mk").st_ino
    assert (new / "small.txt").read_text() == f"path {new}\n"
    # a writer that missed the audit fails loudly instead of corrupting the cache
    if os.geteuid() != 0:
        with pytest.raises(PermissionError):
            open(new / "m.onnx", "wb")
    assert (tree / "m.onnx").stat().st_size == BIG


def test_link_off_copies_all(monkeypatch, tmp_path):
    tree, _ = _seed(monkeypatch, tmp_path, link="0")
    meta = cc.restore("k1", tmp_path / "new_wd", None)
    assert meta["linked"] == 0
    assert os.stat(tmp_path / "new_wd" / "m.onnx").st_ino != os.stat(tree / "m.onnx").st_ino


def test_link_knob_not_in_key():
    assert cc.LINK_ENV in cc._ENV_SKIP
    assert "AIEHLC_WL_CACHE_LINK" in cc._ENV_SKIP
    assert "AIEHLC_WL_KEEP_KPAR" in cc._ENV_SKIP
