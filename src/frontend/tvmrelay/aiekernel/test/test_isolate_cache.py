###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""F13: content cache for isolate_archive (AIEHLC_ISO_CACHE / AIEHLC_FAST). Toolchain-free: the
real isolation step is replaced by a fake that rewrites the archive and counts calls."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from frontend.tvmrelay.aiekernel import build as B  # noqa: E402


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def iso(archive, keep, verbose):
        calls.append(archive)
        data = archive.read_bytes()
        B._iso_backup(archive)
        archive.write_bytes(b"ISO:" + data)
        return {"ok": "bad" not in keep, "kept": sorted(keep), "runtime_hooks": [],
                "localized": 7, "missing": ["bad"] if "bad" in keep else []}
    monkeypatch.setattr(B, "_isolate_archive", iso)
    for k in ("AIEHLC_ISO_CACHE", "AIEHLC_FAST", "AIEHLC_ISO_CACHE_KEEP"):
        monkeypatch.delenv(k, raising=False)
    return calls


def _lib(root: Path, name: str, data: bytes) -> Path:
    (root / name / "build").mkdir(parents=True)
    a = root / name / "build" / "liba.a"
    a.write_bytes(data)
    return a


def test_hit_reuses_and_keeps_backup(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("AIEHLC_ISO_CACHE", str(tmp_path / "c"))
    a1 = _lib(tmp_path, "x", b"A" * 100)
    r1 = B.isolate_archive(a1, ["e"], verbose=False)
    a2 = _lib(tmp_path, "y", b"A" * 100)
    r2 = B.isolate_archive(a2, ["e"], verbose=False)
    assert len(fake) == 1 and r1 == r2
    assert a2.read_bytes() == b"ISO:" + b"A" * 100
    assert (tmp_path / "y" / "isolate_backup_liba.a.bak").read_bytes() == b"A" * 100
    assert os.stat(a2).st_mode & 0o222 == 0          # shared entry is read-only


def test_key_covers_bytes_and_keep(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("AIEHLC_ISO_CACHE", str(tmp_path / "c"))
    B.isolate_archive(_lib(tmp_path, "a", b"A" * 100), ["e"], verbose=False)
    B.isolate_archive(_lib(tmp_path, "b", b"B" * 100), ["e"], verbose=False)       # same size, new bytes
    B.isolate_archive(_lib(tmp_path, "c", b"A" * 100), ["e", "f"], verbose=False)  # new keep set
    assert len(fake) == 3


def test_off_by_default_and_zero_disables(tmp_path, fake, monkeypatch):
    B.isolate_archive(_lib(tmp_path, "a", b"A"), ["e"], verbose=False)
    monkeypatch.setenv("AIEHLC_FAST", "1")
    monkeypatch.setenv("AIEHLC_ISO_CACHE", "0")
    B.isolate_archive(_lib(tmp_path, "b", b"A"), ["e"], verbose=False)
    assert len(fake) == 2 and B._iso_cache_dir() is None


def test_fast_uses_xdg_and_failures_not_stored(tmp_path, fake, monkeypatch):
    monkeypatch.setenv("AIEHLC_FAST", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert B._iso_cache_dir() == (tmp_path / "xdg" / "aiehlc" / "iso").resolve()
    B.isolate_archive(_lib(tmp_path, "a", b"A"), ["bad"], verbose=False)
    B.isolate_archive(_lib(tmp_path, "b", b"A"), ["bad"], verbose=False)
    assert len(fake) == 2


def test_prune_keeps_newest(tmp_path, fake, monkeypatch):
    c = tmp_path / "c"
    monkeypatch.setenv("AIEHLC_ISO_CACHE", str(c))
    monkeypatch.setenv("AIEHLC_ISO_CACHE_KEEP", "2")
    for i in range(4):
        B.isolate_archive(_lib(tmp_path, f"l{i}", bytes([i]) * 10), ["e"], verbose=False)
        ent = max(c.iterdir(), key=lambda p: p.stat().st_mtime_ns)
        os.utime(ent, (1000 + i, 1000 + i))
    assert len([p for p in c.iterdir() if not p.name.startswith(".")]) == 2
