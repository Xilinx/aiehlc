###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""F4 (-fast): weights.bin reuse in generate_driver.

blob_layout must give exactly what write_aligned_blob writes, and the sidecar
check must only say "reuse" while the inputs AND the file are unchanged.

Run:  python -m pytest -q src/frontend/tvmrelay/test/test_arm_build_wbin_reuse.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from frontend.tvmrelay import arm_build as ab  # noqa: E402
from frontend.tvmrelay.test.test_arm_build_blob_order import _blob, _params  # noqa: E402

ORDER = ["p3", "p0", "p5", "p1", "p4"]


def _setup(tmp_path):
    src = tmp_path / "x_params.bin"
    src.write_bytes(_blob(_params()))
    offs = ab.param_offsets(src)
    lay, end = ab.blob_layout(offs, bf16_names={"p1"}, order=ORDER)
    return src, offs, lay, end


def test_layout_matches_write(tmp_path):
    src, offs, lay, end = _setup(tmp_path)
    dst = tmp_path / "weights.bin"
    assert ab.write_aligned_blob(src, offs, dst, bf16_names={"p1"}, order=ORDER) == lay
    assert dst.stat().st_size == end


def test_reuse_only_when_unchanged(tmp_path):
    src, offs, lay, end = _setup(tmp_path)
    layers = tmp_path / "layers"                    # absent: no group steps
    wbin = tmp_path / "weights.bin"
    key = ab._wbin_key(src, lay, {"p1"}, layers)
    assert not ab._wbin_reusable(wbin, key, end)    # nothing written yet
    ab.write_aligned_blob(src, offs, wbin, bf16_names={"p1"}, order=ORDER)
    assert not ab._wbin_reusable(wbin, key, end)    # no sidecar
    ab._wbin_record(wbin, key)
    assert ab._wbin_reusable(wbin, key, end)
    # same bytes, new inode/mtime (e.g. a cache restore): content hash still matches
    data = wbin.read_bytes()
    wbin.unlink()
    wbin.write_bytes(data)
    assert ab._wbin_reusable(wbin, key, end)
    # same size, one byte changed by another writer -> rewrite
    with open(wbin, "r+b") as f:
        f.seek(lay["p0"][0])
        f.write(bytes([data[lay["p0"][0]] ^ 1]))
    os.utime(wbin, ns=(1, 1))
    assert not ab._wbin_reusable(wbin, key, end)
    # different params content -> different key
    p2 = _params()
    p2[0] = (p2[0][0], p2[0][1] + 1.0)
    src.write_bytes(_blob(p2))
    assert ab._wbin_key(src, lay, {"p1"}, layers) != key
    # different bf16 set -> different key
    src.write_bytes(_blob(_params()))
    assert ab._wbin_key(src, lay, {"p1"}, layers) == key
    assert ab._wbin_key(src, lay, set(), layers) != key


def test_switch(monkeypatch):
    monkeypatch.delenv("AIEHLC_WBIN_REUSE", raising=False)
    monkeypatch.delenv("AIEHLC_FAST", raising=False)
    assert not ab._wbin_reuse_on()                  # default build unchanged
    monkeypatch.setenv("AIEHLC_FAST", "1")
    assert ab._wbin_reuse_on()
    monkeypatch.setenv("AIEHLC_WBIN_REUSE", "0")
    assert not ab._wbin_reuse_on()
