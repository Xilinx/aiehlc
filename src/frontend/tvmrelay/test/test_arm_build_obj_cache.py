"""F8: -fast content cache for graph_driver.o (arm_build._obj_cache_*), with the host gcc."""
import shutil
import subprocess

import pytest

from frontend.tvmrelay import arm_build as ab

pytestmark = pytest.mark.skipif(not shutil.which("gcc") or not shutil.which("make"),
                                reason="needs host gcc + make")

MK = "CC := gcc\nCFLAGS := -O2 $(if $(PROFILE),-DGRAPH_PROFILE)\n%.o: %.c\n\t$(CC) $(CFLAGS) -c $< -o $@\n"


def _build(d, hdr="#define K 3\n"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "Makefile").write_text(MK)
    (d / "k.h").write_text(hdr)
    (d / "graph_driver.c").write_text('#include "k.h"\nint f(int x){return x*K;}\n')
    return d


def _make(d, args=()):
    subprocess.run(["make", "-C", str(d), *args, "graph_driver.o"], check=True, capture_output=True)


def test_off_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(ab._OBJ_ENV, raising=False)
    monkeypatch.delenv("AIEHLC_FAST", raising=False)
    assert ab._obj_cache_dir() is None
    assert ab._obj_cache_get(_build(tmp_path / "a"), []) == {}


def test_fast_default_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(ab._OBJ_ENV, raising=False)
    monkeypatch.setenv("AIEHLC_FAST", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert ab._obj_cache_dir() == (tmp_path / "aiehlc" / "armobj").resolve()


def test_miss_put_hit_across_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv(ab._OBJ_ENV, str(tmp_path / "c"))
    a = _build(tmp_path / "a")
    miss = ab._obj_cache_get(a, [])
    assert list(miss) == ["graph_driver.o"]
    _make(a)
    ab._obj_cache_put(a, miss)
    b = _build(tmp_path / "b")                     # another build dir, same content -> hit
    assert ab._obj_cache_get(b, []) == {}
    assert (b / "graph_driver.o").read_bytes() == (a / "graph_driver.o").read_bytes()
    r = subprocess.run(["make", "-q", "-C", str(b), "graph_driver.o"])
    assert r.returncode == 0                       # make sees the restored object as up to date


def test_key_covers_headers_and_make_vars(tmp_path):
    a = _build(tmp_path / "a")
    k = ab._obj_key(a, "graph_driver.o", [])
    assert k and k == ab._obj_key(_build(tmp_path / "b"), "graph_driver.o", [])
    assert k != ab._obj_key(a, "graph_driver.o", ["PROFILE=1"])
    assert k != ab._obj_key(_build(tmp_path / "c", "#define K 4\n"), "graph_driver.o", [])


def test_prune_keeps_newest(tmp_path, monkeypatch):
    c = tmp_path / "c"
    for i, n in enumerate(("old", "new")):
        (c / n).mkdir(parents=True)
        (c / n / "graph_driver.o").write_bytes(b"x" * (1 << 20))
        import os
        os.utime(c / n, (1000 + i, 1000 + i))
    monkeypatch.setenv(ab._OBJ_MAX_ENV, "1")
    assert ab._obj_cache_prune(c) == ["old"]
    assert (c / "new").is_dir()
