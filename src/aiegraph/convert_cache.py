"""Content-addressed cache of ``embedded_convert`` output across build dirs (P3 of -fast).

Opt-in: ``AIEGRAPH_CONVERT_CACHE=<dir>``, or ``AIEHLC_FAST=1`` (aiehlc.sh -fast /
build_unet.py --fast), which uses ``${XDG_CACHE_HOME:-~/.cache}/aiehlc/convert`` when
AIEGRAPH_CONVERT_CACHE is unset (same rule as the F2 kernel .o cache in hostcompile.sh).
An explicit AIEGRAPH_CONVERT_CACHE wins. Neither set = nothing is read or written and
the conversion behaves exactly as before.

Size cap: ``AIEGRAPH_CONVERT_CACHE_MAX_GB`` (default 20; 0 = no cap). After every store
the least-recently-used entries are deleted until the cache fits (a UNet entry is
~4.3 GB, ResNet18 ~0.17 GB). A restore marks the entry used. The entry just stored is
never evicted, even when it alone is bigger than the cap. Leftover ``.*.tmp`` dirs of
crashed stores older than a day are removed by the same prune.

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
    except the cache knobs themselves (incl. AIEHLC_FAST) and AIEHLC_ROOT (covered by the repo root)

Entry = the workdir right after conversion (before any offload), plus meta.json with
the original workdir and image paths. On restore, text files that mention the old
workdir / image path get the new ones; binary files are copied unchanged.

F10 hardlink restore (``AIEGRAPH_CONVERT_CACHE_LINK``, default 1; 0 = copy everything):
files of at least ``_LINK_MIN`` bytes are hardlinked instead of copied when the cache and
the workdir share a filesystem (UNet: 3 x 1.29 GB .onnx/.folded.onnx/_params.bin). Both
names then share one inode, so the linked files are made read-only (0444) in the cache:
a later writer that opens the restored path for writing gets PermissionError instead of
silently corrupting the cache. Writers that replace by rename (boardmain int32 rewrite:
tmp + replace) are fine. Paths the build mutates in place are always copied
(``_NO_LINK_DIRS``: ``arm_build/`` holds weights.bin, rewritten with "wb" by
write_aligned_blob and "r+b" by the F4 layout pass). Cross-device (EXDEV, e.g. the
default ~/.cache on an overlay root) falls back to copy per file.
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
MAX_ENV = "AIEGRAPH_CONVERT_CACHE_MAX_GB"
_MAX_GB_DEFAULT = 20.0
# EXTRA_DEFS / AIEHLC_SH_ARGS are read only at offload (aiekernel/build.py stamp,
# boardmain); convert (deploy_flow + arm_build run_make=False) never reads them, and
# the cached tree is pre-offload, so launch/profiling options must not miss the cache.
# AIEHLC_WL_CACHE (F6 worklocal cache) and AIEHLC_ARM_JOBS (F7 make -j) only change
# where/how fast offload and the ARM make run, never the converted tree; same for
# AIEHLC_ARM_OBJ_CACHE[_MAX_MB] (F8 graph_driver.o cache, used only when make runs).
_ENV_SKIP = {ENV, MAX_ENV, "AIEHLC_KOBJ_CACHE", "AIEHLC_ROOT", "AIEHLC_FAST",
             "EXTRA_DEFS", "AIEHLC_SH_ARGS", "AIEHLC_WL_CACHE", "AIEHLC_ARM_JOBS",
             "AIEHLC_ARM_OBJ_CACHE", "AIEHLC_ARM_OBJ_CACHE_MAX_MB",
             "AIEHLC_WBIN_REUSE", "AIEGRAPH_CONVERT_CACHE_LINK", "AIEHLC_WL_CACHE_LINK"}
LINK_ENV = "AIEGRAPH_CONVERT_CACHE_LINK"
_LINK_MIN = 16 << 20            # hardlink files at least this big (F10)
_NO_LINK_DIRS = ("arm_build",)  # first path parts the build rewrites in place
_TMP_STALE_S = 86400
_TEXT_MAX = 64 << 20            # do not try to relocate text files bigger than this


def cache_dir():
    d = os.environ.get(ENV, "").strip()
    if not d and os.environ.get("AIEHLC_FAST", "0").strip() == "1":
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
        d = str(Path(xdg) / "aiehlc" / "convert")
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
        tmp = p.with_name(p.name + ".reloc.tmp")       # rename, never write in place: p may
        tmp.write_bytes(data)                          # be a hardlink into the cache (F10)
        shutil.copymode(p, tmp) if p.stat().st_nlink == 1 else None
        os.replace(tmp, p)
        n += 1
    return n


def _link_enabled() -> bool:
    return os.environ.get(LINK_ENV, "1").strip() != "0"


def _copy_tree(src: Path, dst: Path, link: bool) -> dict:
    """copytree(symlinks=True) that hardlinks big immutable files; returns counts."""
    st = {"linked": 0, "linked_bytes": 0, "copied": 0}

    def cp(s, d):
        rel = Path(s).relative_to(src)
        if link and rel.parts[0] not in _NO_LINK_DIRS:
            try:
                sz = os.stat(s).st_size
                if sz >= _LINK_MIN:
                    os.chmod(s, 0o444)             # shared inode: writers must fail loudly
                    os.link(s, d)
                    st["linked"] += 1
                    st["linked_bytes"] += sz
                    return d
            except OSError:                            # EXDEV / EPERM / ... -> copy
                pass
        st["copied"] += 1
        return shutil.copy2(s, d)

    shutil.copytree(src, dst, symlinks=True, copy_function=cp)
    return st


def restore(k: str, workdir: Path, image) -> dict:
    """Copy entry ``k`` into the (empty or missing) workdir; None on miss."""
    cd = cache_dir()
    ent = cd / k if cd else None
    if not ent or not (ent / "meta.json").is_file():
        return None
    meta = json.loads((ent / "meta.json").read_text())
    t0 = time.time()
    _touch(ent)
    if workdir.exists():
        shutil.rmtree(workdir)
    try:
        cst = _copy_tree(ent / "tree", workdir, _link_enabled())
    except (OSError, shutil.Error):                    # entry pruned under us -> miss
        shutil.rmtree(workdir, ignore_errors=True)
        return None
    now = time.time()
    for p in workdir.rglob("*"):                       # fresh mtimes, like the KOBJ cache
        if p.is_file() and not p.is_symlink():         # (linked: also bumps the cache inode; harmless)
            os.utime(p, (now, now))
    subs = [(meta["workdir"], str(workdir))]
    if meta.get("image") and image:
        subs.append((meta["image"], str(image)))
    n = _relocate(workdir, subs)
    meta["restore_s"] = round(time.time() - t0, 1)
    meta["relocated_files"] = n
    meta.update(cst)
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
         "image": str(image) if image else None, "stored": time.time(),
         "bytes": _du(tmp / "tree")}, indent=2))
    _touch(tmp)
    try:
        os.rename(tmp, ent)                            # atomic; loser of a race drops its copy
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        prune(keep=k)
    except Exception as e:                             # noqa: BLE001 -- never fail a build on prune
        print(f"aiegraph convert cache: prune failed: {e}")
    return time.time() - t0


def max_bytes():
    """Cap in bytes from AIEGRAPH_CONVERT_CACHE_MAX_GB (default 20); None = no cap."""
    v = os.environ.get(MAX_ENV, "").strip()
    try:
        gb = float(v) if v else _MAX_GB_DEFAULT
    except ValueError:
        gb = _MAX_GB_DEFAULT
    return int(gb * (1 << 30)) if gb > 0 else None


def _du(root: Path) -> int:
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            n += p.stat().st_size
    return n


def _touch(ent: Path):
    """Mark an entry used (LRU clock = mtime of <entry>/used)."""
    try:
        (ent / "used").touch()
    except OSError:
        pass


def _entry_info(ent: Path):
    """(last-used time, bytes) of a finished entry; bytes from meta.json when present."""
    used = ent / "used"
    t = (used if used.is_file() else ent / "meta.json").stat().st_mtime
    try:
        b = int(json.loads((ent / "meta.json").read_text()).get("bytes", -1))
    except (OSError, ValueError):
        b = -1
    return t, (b if b >= 0 else _du(ent / "tree"))


def prune(keep: str = None, cap: int = None) -> list:
    """Delete least-recently-used entries until the cache is <= cap; returns deleted keys."""
    cd = cache_dir()
    if not cd or not cd.is_dir():
        return []
    now = time.time()
    for t in cd.glob(".*.tmp"):                        # crashed stores
        try:
            if now - t.stat().st_mtime > _TMP_STALE_S:
                shutil.rmtree(t, ignore_errors=True)
        except OSError:
            pass
    cap = max_bytes() if cap is None else cap
    if not cap:
        return []
    ents = []
    for ent in cd.iterdir():
        if ent.name.startswith(".") or not (ent / "meta.json").is_file():
            continue
        try:
            t, b = _entry_info(ent)
        except OSError:                                # deleted by a concurrent prune
            continue
        ents.append((t, b, ent))
    total = sum(b for _, b, _ in ents)
    gone = []
    for t, b, ent in sorted(ents, key=lambda e: e[0]):
        if total <= cap:
            break
        if ent.name == keep:
            continue
        # rename first so a concurrent restore never sees a half-deleted entry
        dead = cd / f".{ent.name}.{os.getpid()}.del.tmp"
        try:
            os.rename(ent, dead)
        except OSError:
            continue
        shutil.rmtree(dead, ignore_errors=True)
        total -= b
        gone.append(ent.name)
    if gone:
        print(f"aiegraph convert cache: pruned {len(gone)} LRU entr{'y' if len(gone) == 1 else 'ies'}"
              f", now {total / (1 << 30):.1f} GB (cap {cap / (1 << 30):.1f} GB)")
    return gone
