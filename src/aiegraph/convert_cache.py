"""Content-addressed cache of ``embedded_convert`` output across build dirs (P3 of -fast).

Opt-in: ``AIEGRAPH_CONVERT_CACHE=<dir>``. Unset = nothing is read or written and
the conversion behaves exactly as before.

The in-workdir reuse in ``embedded_convert`` only helps when the same workdir is
converted twice; the build scripts wipe ``ag_work`` first, so every build redoes the
ONNX -> TVM -> per-layer C step (~20 s ResNet18, ~6-7.5 min UNet). This cache keys
the converted tree by CONTENT, so a fresh build dir with the same inputs restores it.

Key (sha256 over):
  * the model file bytes (full sha, not the 16-char ``Model.sha``)
  * quantize, calib file bytes, image file bytes (content, not the path)
  * every file under src/frontend/tvmrelay and src/aiegraph (frontend source)
  * TVM: python dir git HEAD + libtvm.so size/mtime
  * the repo root path (generated Makefiles point into it)
  * env vars AIEGRAPH_* / AIE_* / AIEK_* / AIEHLC_* / CC / CROSS / HOST_CC / EXTRA_DEFS,
    except the cache knobs themselves and AIEHLC_ROOT (covered by the repo root)

Entry = the workdir right after conversion (before any offload), plus meta.json with
the original workdir and image paths. On restore, text files that mention the old
workdir / image path get the new ones; binary files are copied unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

ENV = "AIEGRAPH_CONVERT_CACHE"
_KEY_VERSION = 1
_ENV_PREFIXES = ("AIEGRAPH_", "AIE_", "AIEK_", "AIEHLC_")
_ENV_EXACT = ("CC", "CROSS", "HOST_CC", "EXTRA_DEFS")
_ENV_SKIP = {ENV, "AIEHLC_KOBJ_CACHE", "AIEHLC_ROOT"}
_TEXT_MAX = 64 << 20            # do not try to relocate text files bigger than this


def cache_dir():
    d = os.environ.get(ENV, "").strip()
    return Path(d).resolve() if d else None


def _file_sha(p, h=None):
    h = h or hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h


def _tree_sha(root: Path, h):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        h.update(str(p.relative_to(root)).encode() + b"\0")
        _file_sha(p, h)


def key(model, quantize, calib, image, repo_root: Path) -> tuple:
    """(hex key, dict of the parts) -- the dict goes into meta.json for debugging."""
    parts = {"v": _KEY_VERSION, "model": _file_sha(model.onnx_path).hexdigest(),
             "quantize": quantize,
             "calib": [_file_sha(c).hexdigest() for c in calib] if calib else None,
             "image": _file_sha(image).hexdigest() if image else None,
             "repo": str(repo_root)}
    h = hashlib.sha256()
    for sub in ("src/frontend/tvmrelay", "src/aiegraph"):
        _tree_sha(repo_root / sub, h)
    parts["src"] = h.hexdigest()
    tvm = repo_root / "thirdparty" / "tvm-0.16"
    try:
        import subprocess
        rev = subprocess.run(["git", "-C", str(tvm), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
    except Exception:                                  # noqa: BLE001
        rev = ""
    so = tvm / "build" / "libtvm.so"
    st = so.stat() if so.is_file() else None
    parts["tvm"] = f"{rev}:{st.st_size}:{int(st.st_mtime)}" if st else rev
    parts["env"] = {k: v for k, v in sorted(os.environ.items())
                    if k not in _ENV_SKIP and not k.endswith("_CACHE")
                    and (k.startswith(_ENV_PREFIXES) or k in _ENV_EXACT)}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32], parts


def _relocate(root: Path, subs: list) -> int:
    subs = [(a.encode(), b.encode()) for a, b in subs if a and a != b]
    if not subs:
        return 0
    n = 0
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink() or p.stat().st_size > _TEXT_MAX:
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192] or not any(a in data for a, _ in subs):
            continue
        for a, b in subs:
            data = data.replace(a, b)
        p.write_bytes(data)
        n += 1
    return n


def restore(k: str, workdir: Path, image) -> dict:
    """Copy entry ``k`` into the (empty or missing) workdir; None on miss."""
    cd = cache_dir()
    ent = cd / k if cd else None
    if not ent or not (ent / "meta.json").is_file():
        return None
    meta = json.loads((ent / "meta.json").read_text())
    t0 = time.time()
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(ent / "tree", workdir, symlinks=True)
    now = time.time()
    for p in workdir.rglob("*"):                       # fresh mtimes, like the KOBJ cache
        if p.is_file() and not p.is_symlink():
            os.utime(p, (now, now))
    subs = [(meta["workdir"], str(workdir))]
    if meta.get("image") and image:
        subs.append((meta["image"], str(image)))
    n = _relocate(workdir, subs)
    meta["restore_s"] = round(time.time() - t0, 1)
    meta["relocated_files"] = n
    return meta


def store(k: str, parts: dict, workdir: Path, image) -> float:
    """Copy the converted workdir into the cache; returns the seconds spent (0 if skipped)."""
    cd = cache_dir()
    if not cd:
        return 0.0
    ent = cd / k
    if (ent / "meta.json").is_file():
        return 0.0
    t0 = time.time()
    cd.mkdir(parents=True, exist_ok=True)
    tmp = cd / f".{k}.{os.getpid()}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(workdir, tmp / "tree", symlinks=True)
    (tmp / "meta.json").write_text(json.dumps(
        {"key": k, "parts": parts, "workdir": str(workdir),
         "image": str(image) if image else None, "stored": time.time()}, indent=2))
    try:
        os.rename(tmp, ent)                            # atomic; loser of a race drops its copy
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return time.time() - t0
