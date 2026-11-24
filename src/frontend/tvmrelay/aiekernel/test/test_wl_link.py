###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""F15: wl-cache restore and install hardlink the large read-only parts of a worklocal
(build/*.a, build/kernel_*.o, build/kpar/**) and copy everything written in place."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from frontend.tvmrelay.aiekernel import build as B  # noqa: E402

FILES = {
    "host.cc": b"int host;\n",
    "kernel_k1.cc": b"k1\n",
    "aieml_k1.prx": b"prx\n",
    "build/libaiek_x.a": b"!<arch>\nAAAA",
    "build/host.o": b"HOSTO",
    "build/kernel_k1.o": b"K1O",
    "build/kernel": b"ELF",
    "build/kernel.stamp": b"st\n",
    "build/aie_runtime.o": b"RT",
    "build/kpar/k1/build/kernel": b"KPELF",
    "build/kpar/k1/build/aiehlc_ctrlpkt": b"TOOL",
}
LINKED = {"build/libaiek_x.a", "build/kernel_k1.o", "build/kpar/k1/build/kernel"}


def _tree(root: Path) -> Path:
    for rel, data in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (root / "build" / "kpar" / "k1" / "host.cc").symlink_to(root / "host.cc")
    return root


@pytest.mark.parametrize("rel,want", [
    ("build/libaiek_x.a", True), ("build/kernel_k1.o", True), ("build/kpar/k/build/kernel", True),
    ("build/kpar/k/build/aiehlc_ctrlpkt", False), ("build/host.o", False), ("build/kernel", False),
    ("build/aie_runtime.o", False), ("host.cc", False), ("kernel_k1.o", False),
    ("build/sub/kernel_k1.o", False), ("build/kernel.stamp", False)])
def test_linkable(rel, want):
    assert B._wl_linkable(rel) is want


@pytest.fixture(autouse=True)
def _keep_kpar_default(request, monkeypatch):
    """F15 tests below run with kpar kept (AIEHLC_WL_KEEP_KPAR=1); F16 tests opt out."""
    if "f16" not in request.node.name:
        monkeypatch.setenv("AIEHLC_WL_KEEP_KPAR", "1")


def test_copytree_links_and_copies(tmp_path, monkeypatch):
    monkeypatch.delenv("AIEHLC_WL_CACHE_LINK", raising=False)
    src = _tree(tmp_path / "src")
    n = B._wl_copytree(src, tmp_path / "dst")
    dst = tmp_path / "dst"
    for rel, data in FILES.items():
        s, d = src / rel, dst / rel
        assert d.read_bytes() == data
        assert (os.stat(s).st_ino == os.stat(d).st_ino) is (rel in LINKED), rel
        if rel in LINKED:
            assert os.stat(d).st_mode & 0o222 == 0, rel         # read-only: in-place write fails
        else:
            assert os.stat(d).st_mode & 0o200, rel
    assert (dst / "build/kpar/k1/host.cc").is_symlink()
    assert n == {"linked": len(LINKED), "copied": len(FILES) - len(LINKED), "skipped": 0, "dropped": 0}
    with pytest.raises(PermissionError):
        (dst / "build/kernel_k1.o").write_bytes(b"x")
    # the writers hostcompile.sh uses are safe: mv -f / unlink replace the name, not the inode
    os.replace(tmp_path / "dst/build/kernel", dst / "build/kernel_k1.o")
    assert (src / "build/kernel_k1.o").read_bytes() == b"K1O"


def test_copytree_link_off(tmp_path, monkeypatch):
    monkeypatch.setenv("AIEHLC_WL_CACHE_LINK", "0")
    src = _tree(tmp_path / "src")
    n = B._wl_copytree(src, tmp_path / "dst")
    assert n["linked"] == 0
    for rel in FILES:
        assert os.stat(src / rel).st_ino != os.stat(tmp_path / "dst" / rel).st_ino


def test_copytree_link_error_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("AIEHLC_WL_CACHE_LINK", raising=False)
    src = _tree(tmp_path / "src")

    def nolink(*a, **k):
        raise OSError(18, "Invalid cross-device link")
    monkeypatch.setattr(B.os, "link", nolink)
    n = B._wl_copytree(src, tmp_path / "dst")
    assert n["linked"] == 0 and n["copied"] == len(FILES)
    assert (tmp_path / "dst/build/kernel_k1.o").read_bytes() == b"K1O"


def test_cache_put_get_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AIEHLC_WL_CACHE", str(tmp_path / "c"))
    monkeypatch.delenv("AIEHLC_WL_CACHE_LINK", raising=False)
    dest = _tree(tmp_path / "a" / "worklocal")
    log = tmp_path / "a" / "aiehlc.log"
    log.write_text("log\n")
    B._wl_cache_put("s1", dest, log)
    entry = tmp_path / "c" / "s1" / "worklocal"
    dest2, log2 = tmp_path / "b" / "worklocal", tmp_path / "b" / "aiehlc.log"
    dest2.parent.mkdir()
    assert B._wl_cache_get("s1", dest2, log2)
    for rel, data in FILES.items():
        assert (dest2 / rel).read_bytes() == data
        assert (os.stat(entry / rel).st_ino == os.stat(dest2 / rel).st_ino) is (rel in LINKED)
    # a later restore over an existing dest unlinks the old tree; the entry survives
    assert B._wl_cache_get("s1", dest2, log2)
    assert (entry / "build/kernel_k1.o").read_bytes() == b"K1O"


def test_install_skips_archive_and_host_o(tmp_path, monkeypatch):
    """install(): build/*.a and build/host.o are not copied (deleted before hostcompile anyway);
    host.cc is a private copy because the adapters are appended to it in place."""
    from frontend.tvmrelay import aie_layer_lib
    monkeypatch.delenv("AIEHLC_WL_CACHE_LINK", raising=False)
    wl = _tree(tmp_path / "wl")
    seen = {}

    def fake_build(aie_dir, stem, repo, verbose=False):
        b = Path(aie_dir) / "build"
        seen["names"] = sorted(p.name for p in b.iterdir())
        return {"ok": False, "reason": "stop here"}
    monkeypatch.setattr(aie_layer_lib, "build_archive", fake_build)

    class Spec:
        name = "x"
    monkeypatch.setattr(B, "adapter_source", lambda *a, **k: "// adapter\n")
    r = B.install(Spec(), wl, tmp_path / "aie", [[object()]], verbose=False)
    assert r["ok"] is False and r["reason"] == "stop here"
    assert "host.o" not in seen["names"] and "libaiek_x.a" not in seen["names"]
    assert "kernel_k1.o" in seen["names"]
    assert (wl / "host.cc").read_bytes() == FILES["host.cc"]      # source untouched
    assert b"adapter" in (tmp_path / "aie" / "host.cc").read_bytes()
    assert os.stat(wl / "build/kernel_k1.o").st_ino == os.stat(tmp_path / "aie/build/kernel_k1.o").st_ino


# ---- F16: build/kpar/ (hostcompile.sh scratch, dangling links) is dropped by default ----
KPAR = {r for r in FILES if r.startswith("build/kpar/")}


def test_f16_copytree_drops_kpar(tmp_path, monkeypatch):
    monkeypatch.delenv("AIEHLC_WL_CACHE_LINK", raising=False)
    monkeypatch.delenv("AIEHLC_WL_KEEP_KPAR", raising=False)
    src = _tree(tmp_path / "src")
    n = B._wl_copytree(src, tmp_path / "dst")
    dst = tmp_path / "dst"
    assert not (dst / "build/kpar").exists()
    for rel, data in FILES.items():
        if rel not in KPAR:
            assert (dst / rel).read_bytes() == data, rel
    kept_linked = LINKED - KPAR
    assert n == {"linked": len(kept_linked), "dropped": 1, "skipped": 0,
                 "copied": len(FILES) - len(KPAR) - len(kept_linked)}
    assert (src / "build/kpar/k1/build/kernel").read_bytes() == b"KPELF"     # source untouched


def test_f16_only_top_level_kpar(tmp_path, monkeypatch):
    """a 'kpar' anywhere else (or a symlink named build/kpar) is not scratch"""
    monkeypatch.delenv("AIEHLC_WL_KEEP_KPAR", raising=False)
    src = _tree(tmp_path / "src")
    (src / "kpar").mkdir()
    (src / "kpar" / "x").write_bytes(b"x")
    (src / "build/sub/kpar").mkdir(parents=True)
    (src / "build/sub/kpar/y").write_bytes(b"y")
    B._wl_copytree(src, tmp_path / "dst")
    assert (tmp_path / "dst/kpar/x").read_bytes() == b"x"
    assert (tmp_path / "dst/build/sub/kpar/y").read_bytes() == b"y"
    s2 = tmp_path / "s2"
    (s2 / "build").mkdir(parents=True)
    (s2 / "elsewhere").mkdir()
    (s2 / "build/kpar").symlink_to(s2 / "elsewhere")
    B._wl_copytree(s2, tmp_path / "d2")
    assert (tmp_path / "d2/build/kpar").is_symlink()


def test_f16_cache_roundtrip_and_install(tmp_path, monkeypatch):
    from frontend.tvmrelay import aie_layer_lib
    monkeypatch.setenv("AIEHLC_WL_CACHE", str(tmp_path / "c"))
    monkeypatch.delenv("AIEHLC_WL_KEEP_KPAR", raising=False)
    dest = _tree(tmp_path / "a" / "worklocal")
    log = tmp_path / "a" / "aiehlc.log"
    log.write_text("log\n")
    B._wl_cache_put("s1", dest, log)
    assert not (tmp_path / "c/s1/worklocal/build/kpar").exists()
    assert (dest / "build/kpar/k1/build/kernel").is_file()           # the live worklocal keeps it
    dest2 = tmp_path / "b" / "worklocal"
    dest2.parent.mkdir()
    assert B._wl_cache_get("s1", dest2, tmp_path / "b" / "aiehlc.log")
    assert not (dest2 / "build/kpar").exists()
    seen = {}

    def fake_build(aie_dir, stem, repo, verbose=False):
        seen["kpar"] = (Path(aie_dir) / "build/kpar").exists()
        return {"ok": False, "reason": "stop"}
    monkeypatch.setattr(aie_layer_lib, "build_archive", fake_build)
    monkeypatch.setattr(B, "adapter_source", lambda *a, **k: "// adapter\n")

    class Spec:
        name = "x"
    B.install(Spec(), dest, tmp_path / "aie", [[object()]], verbose=False)
    assert seen["kpar"] is False


def test_f16_keep_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AIEHLC_WL_KEEP_KPAR", "1")
    src = _tree(tmp_path / "src")
    n = B._wl_copytree(src, tmp_path / "dst")
    assert n["dropped"] == 0 and (tmp_path / "dst/build/kpar/k1/build/kernel").is_file()
