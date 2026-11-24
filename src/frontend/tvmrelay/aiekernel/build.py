###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""BUILD: aiehlc -> ``layers/NN_op/aie/build/libaiek_<name>.a`` with the adapters in.

Compile only. Nothing here talks to a board.

1. **aiehlc** -- ``source script/aiehlc.sh --runtime-source-file <kernel.cc>``
   in a scratch cwd (it writes ``aout/`` relative to the CWD and ``rm -rf``s it
   first), exactly as ``aie_stem_lib.build_stem_layer`` does. Cached under
   ``<out>/aiek/<name>/worklocal`` by a hash of the spliced source + every
   header next to it, so re-running ``offload`` with new layers costs no
   aiehlc run.
2. **install** -- copy that tree to the FIRST selected layer's ``aie/``, append
   one ``aie_<symbol>`` adapter per selected layer to ``host.cc``, and
   re-archive with ``hostcompile.sh`` (``aie_layer_lib.build_archive``).
   ``build/host.o`` is deleted first: ``hostcompile.sh`` skips any object that
   already exists, so the adapters would otherwise silently not be compiled.
   ``build/kernel.o`` is KEPT: install changes only host.cc, and
   ``compile_one_kernel`` reuses the object while ``build/kernel.stamp``
   (kernel.cc, bcf, prx, kc.sh) matches -- no second xchesscc run.
3. **isolate** -- merge the archive's own objects (``host.o``, ``routing.o``,
   ``kernel*.o``) with ``ld -r`` and make every global LOCAL except the
   adapters, the entry, the spec's ``export`` lines, and the RUNTIME HOOKS --
   symbols the shared runtime members reference by name (``routing``,
   ``g_runtime_debug_level``). The user's helpers, ``host_canonicalized``,
   ``_binary_kernel_<func>_*`` and any public API of a wrapped library stop
   being able to collide with anything else in the ELF.

   Per-library routing: a library built by the current aiehlc passes its own
   ``routing`` to ``__Runtime_init_mesh_partition_with`` (generated
   ``aieArray::partition``), and the runtime re-initializes a partition another
   library's routing set up. Its ``routing`` is therefore made LOCAL, and
   several libraries share one ELF. A library built the old way (it calls
   ``__Runtime_init_mesh_partition``, so the runtime uses the global, weak
   ``routing``) keeps ``routing`` strong and global -- weakening it would let a
   second library link cleanly and run with the first one's stream-switch
   configuration; kept strong, two such libraries collide and
   :func:`link_collisions` reports it before ``make``.
4. **verify** -- ``nm``: every adapter is a defined ``T``; and across the
   archives the driver will actually call, no strong symbol is defined twice.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .adapter import ADAPTER_MARK, adapter_source, entry_symbol
from .hostbuild import REPO

__all__ = ["toolchain", "aiehlc_compile", "install", "isolate_archive",
           "link_collisions", "archive_name"]

CROSS = os.environ.get("CROSS", "aarch64-none-elf-")
_RUNTIME_PREFIXES = ("aie_runtime", "aie_ctrlpkt")


def archive_name(spec) -> str:
    return f"aiek_{spec.name}"


def toolchain() -> tuple:
    """``(ok, detail)``: can this box run aiehlc + the cross binutils?"""
    from frontend.tvmrelay.arm_build import toolchain_status

    ok, detail = toolchain_status(REPO)
    if ok and shutil.which(f"{CROSS}nm") is None:
        return False, f"{CROSS}nm not on PATH"
    return ok, detail


def _stamp(spec) -> str:
    """Hash of everything aiehlc reads: the spliced text, the headers beside
    it, and the user-kernel header. A changed header must rebuild."""
    from .spec import load_source

    h = hashlib.sha1(load_source(spec.source).text.encode())
    for p in sorted(spec.source.parent.glob("*.h")):
        if p.stat().st_size < (1 << 20):          # skip multi-MB data dumps
            h.update(p.name.encode() + p.read_bytes())
    h.update((REPO / "include" / "aie_user_kernel.h").read_bytes())
    h.update(("aiehlc.sh args: " + os.environ.get("AIEHLC_SH_ARGS", "") +
              " defs: " + os.environ.get("EXTRA_DEFS", "")).encode())
    # The generator itself: a rebuilt aiehlc may emit different host/kernel code
    # (e.g. per-library routing), so a cached tree from an older one is stale.
    exe = REPO / "build" / "aiehlc"
    if exe.is_file():
        st = exe.stat()
        h.update(f"aiehlc {st.st_size} {st.st_mtime_ns}".encode())
    return h.hexdigest()


# -fast: share the aiehlc output tree (worklocal) across build dirs, keyed by _stamp (everything
# aiehlc reads, incl. AIEHLC_SH_ARGS/EXTRA_DEFS and the aiehlc binary). UNet: the aiehlc stage is
# ~395 s of a 698 s warm --fast build even with every kernel .o cached; an entry is ~1.3 GB.
# AIEHLC_WL_CACHE=<dir> or AIEHLC_FAST=1 (default ${XDG_CACHE_HOME:-~/.cache}/aiehlc/wl);
# LRU-pruned to AIEHLC_WL_CACHE_MAX_MB (default 8192) after each store.
_WL_ENV, _WL_MAX_ENV = "AIEHLC_WL_CACHE", "AIEHLC_WL_CACHE_MAX_MB"


def _wl_cache_dir():
    d = os.environ.get(_WL_ENV, "").strip()
    if not d and os.environ.get("AIEHLC_FAST", "0").strip() == "1":
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
        d = str(Path(xdg) / "aiehlc" / "wl")
    return Path(d).resolve() if d else None


# F15: the three ~430 MB parts of a UNet worklocal -- build/*.a, build/kernel_*.o (44 x 10 MB) and
# build/kpar/** (aiehlc's per-kernel scratch, 44 x 10 MB ELFs) -- are hardlinked, not copied, from the
# wl cache into worklocal and from worklocal into the layer's aie/ (install). Audit of every writer
# (hostcompile.sh, install): kernel_<k>.o is only replaced by `mv -f`, kpar/ is `rm -rf`ed before a
# rebuild, install unlinks build/*.a and build/host.o before hostcompile.sh, and gcc/as/ld/ar unlink
# or rename their outputs. Files written in place (host.cc, *.stamp, build/kernel, aiehlc_ctrlpkt via
# cp) are NOT in the set and are still copied. Linked files are made 0444 so a future in-place writer
# fails loudly instead of corrupting the cache. EXDEV/EPERM -> copy. AIEHLC_WL_CACHE_LINK=0 = copy all.
_WL_LINK_ENV = "AIEHLC_WL_CACHE_LINK"


def _wl_linkable(rel: str) -> bool:
    """*rel*: path relative to a worklocal root, '/'-separated."""
    if not rel.startswith("build/"):
        return False
    tail = rel[len("build/"):]
    if tail.startswith("kpar/"):
        name = tail.rsplit("/", 1)[-1]
        return name != "aiehlc_ctrlpkt"              # cp -p'd into kpar/<k>/build/ (in place)
    return "/" not in tail and (tail.endswith(".a") or
                                (tail.startswith("kernel_") and tail.endswith(".o")))


def _wl_link_on() -> bool:
    return os.environ.get(_WL_LINK_ENV, "1").strip() != "0"


# F16: build/kpar/ is hostcompile.sh's per-kernel scratch for the parallel kernel build. Nothing reads
# it after hostcompile.sh returns: the next hostcompile.sh run (install) either reuses every
# kernel_<k>.o (-nt) and never looks at kpar/, or `rm -rf`s it before a rebuild. Its symlinks are
# absolute and point into the aiehlc run's own worklocal, so in the cache and in aie/ they all dangle
# (UNet: 12,056 of 12,056 links, 220 files, 458 MB). The wl cache get/put and install drop it.
# AIEHLC_WL_KEEP_KPAR=1 keeps it (F15 behaviour: files hardlinked, links copied).
_WL_KEEP_KPAR_ENV = "AIEHLC_WL_KEEP_KPAR"
_WL_SCRATCH = "build/kpar"


def _wl_keep_kpar() -> bool:
    return os.environ.get(_WL_KEEP_KPAR_ENV, "0").strip() == "1"


def _wl_copytree(src: Path, dst: Path, skip=None) -> dict:
    """copytree(symlinks=True) that hardlinks :func:`_wl_linkable` files (0444) and drops the
    ``build/kpar`` scratch dir (F16, unless AIEHLC_WL_KEEP_KPAR=1).
    *skip(rel)* -> True drops a file. Returns ``{"linked", "copied", "skipped", "dropped"}`` counts
    (*dropped*: scratch directories left out)."""
    src = Path(src)
    n = {"linked": 0, "copied": 0, "skipped": 0, "dropped": 0}
    link = _wl_link_on()
    keep_kpar = _wl_keep_kpar()

    def _copy(s, d):
        rel = Path(s).relative_to(src).as_posix()
        if link and _wl_linkable(rel) and not os.path.islink(s):
            try:
                if os.stat(s).st_mode & 0o222:
                    os.chmod(s, 0o444)
                os.link(s, d)
                n["linked"] += 1
                return d
            except OSError:                       # EXDEV / EPERM / exists -> copy
                pass
        n["copied"] += 1
        return shutil.copy2(s, d)

    def _ignore(d, names):
        base = Path(d).relative_to(src)
        out = []
        if not keep_kpar:
            drop = [x for x in names if (base / x).as_posix() == _WL_SCRATCH
                    and (Path(d) / x).is_dir() and not (Path(d) / x).is_symlink()]
            n["dropped"] += len(drop)
            out += drop
        if skip is not None:
            files = [x for x in names if x not in out and not (Path(d) / x).is_dir()
                     and skip((base / x).as_posix())]
            n["skipped"] += len(files)
            out += files
        return out

    shutil.copytree(src, dst, symlinks=True, copy_function=_copy, ignore=_ignore)
    return n


def _wl_cache_get(stamp: str, dest: Path, log: Path) -> bool:
    c = _wl_cache_dir()
    e = c / stamp if c else None
    if e is None or not (e / "DONE").is_file() or not (e / "worklocal" / "host.cc").is_file():
        return False
    tmp = dest.with_name(dest.name + f".get{os.getpid()}")
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        _wl_copytree(e / "worklocal", tmp)
        if (e / "aiehlc.log").is_file():
            shutil.copy2(e / "aiehlc.log", log)
    except (OSError, shutil.Error):                 # entry pruned under us -> miss
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    (e / "DONE").touch()                            # LRU hint
    return True


def _wl_cache_put(stamp: str, dest: Path, log: Path) -> None:
    c = _wl_cache_dir()
    if c is None:
        return
    e = c / stamp
    if (e / "DONE").is_file():
        return
    tmp = c / f".tmp-{stamp}-{os.getpid()}"
    try:
        c.mkdir(parents=True, exist_ok=True)
        _wl_copytree(dest, tmp / "worklocal")
        if log.is_file():
            shutil.copy2(log, tmp / "aiehlc.log")
        (tmp / "DONE").write_text(stamp + "\n")
        tmp.rename(e)
    except OSError as ex:                           # never fail a build on the cache
        print(f"  [build] aiehlc wl cache: store failed: {ex}")
        shutil.rmtree(tmp, ignore_errors=True)
        return
    try:
        _wl_cache_prune(c, keep=stamp)
    except Exception as ex:                         # noqa: BLE001
        print(f"  [build] aiehlc wl cache: prune failed: {ex}")


def _du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def _wl_cache_prune(c: Path, keep: str = None) -> list:
    cap = int(os.environ.get(_WL_MAX_ENV, "8192")) << 20
    now, gone = time.time(), []
    for t in c.glob(".tmp-*"):                      # crashed stores
        if now - t.stat().st_mtime > 86400:
            shutil.rmtree(t, ignore_errors=True)
    ents = sorted(((e / "DONE").stat().st_mtime, e) for e in c.iterdir()
                  if not e.name.startswith(".") and (e / "DONE").is_file())
    sizes = {e: _du(e) for _, e in ents}
    total = sum(sizes.values())
    for _, e in ents:                               # oldest first
        if total <= cap:
            break
        if e.name == keep:
            continue
        shutil.rmtree(e, ignore_errors=True)
        total -= sizes[e]
        gone.append(e.name)
    if gone:
        print(f"  [build] aiehlc wl cache: pruned {len(gone)} entr{'y' if len(gone) == 1 else 'ies'}, "
              f"now {total >> 20} MB")
    return gone


def aiehlc_compile(spec, out_dir, aie_version: int = 5, verbose: bool = True) -> dict:
    """Run aiehlc on the kernel source. Returns ``{"ok", "worklocal"|"reason", "log", "cached"}``."""
    work = Path(out_dir) / "aiek" / spec.name
    work.mkdir(parents=True, exist_ok=True)
    dest, log, stamp_file = work / "worklocal", work / "aiehlc.log", work / "aiehlc.stamp"
    stamp = _stamp(spec)
    if (dest / "host.cc").is_file() and stamp_file.is_file() \
            and stamp_file.read_text().strip() == stamp:
        return {"ok": True, "worklocal": dest, "log": str(log), "cached": True}
    t_get = time.time()
    if _wl_cache_get(stamp, dest, log):
        stamp_file.write_text(stamp + "\n")
        print(f"  [build] aiehlc wl cache hit {stamp} -> {dest} ({time.time() - t_get:.1f}s)")
        return {"ok": True, "worklocal": dest, "log": str(log), "cached": "wl"}

    # Scratch next to the cache, not in /tmp: aiehlc's aout/ is large and /tmp
    # is often a small shared volume (a full one fails the build mid-way).
    scratch = Path(tempfile.mkdtemp(prefix="aiehlc_run_", dir=work))
    # AIEHLC_SH_ARGS: extra aiehlc.sh options (e.g. --skip-bss), part of the stamp.
    cmd = (f'source "{REPO}/script/aiehlc.sh" --aie-version {aie_version} '
           f'--runtime-source-file "{spec.source}" {os.environ.get("AIEHLC_SH_ARGS", "")}').rstrip()
    start = time.time()
    try:
        with log.open("w") as fh:
            rc = subprocess.run(["bash", "-c", cmd], cwd=str(scratch),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        src = scratch / "aout" / "worklocal"
        # one kernel -> kernel.cc; several __global__ kernels -> kernel_<name>.cc each (hostcompile.sh multi-kernel)
        has_kernel = (src / "kernel.cc").is_file() or any(src.glob("kernel_*.cc"))
        if rc != 0 or not (src / "host.cc").is_file() or not has_kernel:
            return {"ok": False, "log": str(log),
                    "reason": f"aiehlc.sh rc={rc}, no host.cc/kernel.cc; see {log}"}
        if dest.exists():
            shutil.rmtree(dest)
        host_text = (src / "host.cc").read_text(errors="replace")
        # Data headers aiehlc's flat *.h glob copied along but nothing includes
        # (conv2dstem's 6 MB layeriohex dumps) are not artifacts of the build.
        shutil.copytree(src, dest, symlinks=True, ignore=lambda d, names: [
            n for n in names if n.endswith(".h") and Path(d) == src
            and f'"{n}"' not in host_text and (Path(d) / n).stat().st_size > (1 << 20)])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    stamp_file.write_text(stamp + "\n")
    _wl_cache_put(stamp, dest, log)
    if verbose:
        print(f"  [build] aiehlc {spec.source.name} -> {dest} ({time.time() - start:.0f}s)")
    return {"ok": True, "worklocal": dest, "log": str(log), "cached": False}


def _nm(paths, *flags) -> list:
    """``(name, type)`` rows from ``nm -P`` over *paths*."""
    proc = subprocess.run([f"{CROSS}nm", "-P", *flags, *[str(p) for p in paths]],
                          capture_output=True, text=True)
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.endswith(":") and len(parts[1]) == 1:
            rows.append((parts[0], parts[1]))
    return rows


# F13 -fast: content cache for isolate_archive. The UNet carrier archive is ~430 MB; ar x + ld -r +
# objcopy + ar rcs take 8-13 s per build although hostcompile.sh rebuilds it byte-identically (F11).
# AIEHLC_ISO_CACHE=<dir>, or under AIEHLC_FAST=1 ${XDG_CACHE_HOME:-~/.cache}/aiehlc/iso;
# AIEHLC_ISO_CACHE=0 disables. Key = sha1(version, keep, binutils identity, archive bytes); entries are
# 0444 and hardlinked into the build (copy on EXDEV); the newest AIEHLC_ISO_CACHE_KEEP (default 3) stay.
_ISO_VERSION = 1            # bump when isolate_archive's output changes
_ISO_ENV, _ISO_KEEP_ENV = "AIEHLC_ISO_CACHE", "AIEHLC_ISO_CACHE_KEEP"


def _iso_cache_dir():
    d = os.environ.get(_ISO_ENV, "").strip()
    if d == "0":
        return None
    if not d and os.environ.get("AIEHLC_FAST", "0").strip() == "1":
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip() or str(Path.home() / ".cache")
        d = str(Path(xdg) / "aiehlc" / "iso")
    return Path(d).resolve() if d else None


def _iso_key(archive: Path, keep) -> str:
    h = hashlib.sha1(f"v{_ISO_VERSION}\0{CROSS}\0{chr(1).join(sorted(set(keep)))}\0".encode())
    for tool in ("ar", "ld", "objcopy", "nm"):     # a toolchain change must miss
        exe = shutil.which(f"{CROSS}{tool}")
        if exe:
            st = Path(exe).resolve().stat()
            h.update(f"{Path(exe).resolve()}:{st.st_size}:{st.st_mtime_ns}\0".encode())
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso_backup(archive: Path) -> Path:
    # The original stays next to the layer for debugging; rename, not copy (430 MB for UNet).
    backup = archive.parent.parent / f"isolate_backup_{archive.name}.bak"   # not *.a: Makefile wildcard
    os.replace(archive, backup)
    return backup


def _iso_place(ent: Path, archive: Path) -> None:
    tmp = archive.with_name(archive.name + f".iso{os.getpid()}")
    tmp.unlink(missing_ok=True)
    try:
        os.link(ent, tmp)
    except OSError:
        shutil.copyfile(ent, tmp)
        os.chmod(tmp, 0o644)
    tmp.replace(archive)


def _iso_cache_get(c: Path, key: str, archive: Path):
    import json
    e = c / key
    try:
        res = json.loads((e / "result.json").read_text())
        if not (e / "lib.a").is_file():
            return None
        _iso_backup(archive)
        _iso_place(e / "lib.a", archive)
    except (OSError, ValueError):
        return None
    try:
        os.utime(e)                                  # LRU hint
    except OSError:
        pass
    return res


def _iso_cache_put(c: Path, key: str, archive: Path, res: dict) -> None:
    import json
    e = c / key
    tmp = c / f".tmp-{key}-{os.getpid()}"
    try:
        if e.is_dir():
            return
        tmp.mkdir(parents=True)
        try:
            os.link(archive, tmp / "lib.a")
        except OSError:
            shutil.copyfile(archive, tmp / "lib.a")
        os.chmod(tmp / "lib.a", 0o444)               # an in-place writer must fail loudly
        (tmp / "result.json").write_text(json.dumps(res))
        tmp.rename(e)
    except OSError as ex:                            # never fail a build on the cache
        print(f"  [build] isolate cache: store failed: {ex}")
        shutil.rmtree(tmp, ignore_errors=True)
        return
    try:
        n = max(1, int(os.environ.get(_ISO_KEEP_ENV, "3")))
        ents = sorted((x.stat().st_mtime, x) for x in c.iterdir()
                      if not x.name.startswith(".") and (x / "result.json").is_file())
        for _, x in ents[:-n]:
            if x.name != key:
                shutil.rmtree(x, ignore_errors=True)
        for t in c.glob(".tmp-*"):
            if time.time() - t.stat().st_mtime > 86400:
                shutil.rmtree(t, ignore_errors=True)
    except (OSError, ValueError) as ex:
        print(f"  [build] isolate cache: prune failed: {ex}")


def isolate_archive(archive: Path, keep, verbose: bool = True) -> dict:
    """Localize every own global of *archive* except *keep* (see module doc)."""
    archive = Path(archive).resolve()
    keep = list(keep)
    c = _iso_cache_dir()
    key = _iso_key(archive, keep) if c else None
    if key:
        res = _iso_cache_get(c, key, archive)
        if res is not None:
            if verbose:
                print(f"  [build] isolate {archive.name}: cached ({len(res['kept']) - len(res['missing'])} "
                      f"entry symbol(s) + {len(res['runtime_hooks'])} runtime hook(s), "
                      f"{res['localized']} made local)")
            return res
    res = _isolate_archive(archive, keep, verbose)
    if key and res["ok"]:
        _iso_cache_put(c, key, archive, res)
    return res


def _isolate_archive(archive: Path, keep, verbose: bool) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="aiek_iso_", dir=archive.parent))
    try:
        subprocess.run([f"{CROSS}ar", "x", str(archive)], cwd=tmp, check=True)
        members = subprocess.run([f"{CROSS}ar", "t", str(archive)], capture_output=True,
                                 text=True, check=True).stdout.split()
        own = [m for m in members if not m.startswith(_RUNTIME_PREFIXES)]
        runtime = [m for m in members if m.startswith(_RUNTIME_PREFIXES)]
        merged = tmp / f"{archive.stem}_own.o"
        subprocess.run([f"{CROSS}ld", "-r", "-o", str(merged), *own], cwd=tmp, check=True)

        strong = {n for n, t in _nm([merged], "-g", "--defined-only") if t in "TDBRAGSC"}
        # Only STRONG references from the runtime make a hook; its weak routing()
        # reference is satisfied by the partition's own routing (see below).
        needed = {n for n, t in _nm([tmp / m for m in runtime], "-u") if t == "U"}
        keep = set(keep)
        # Runtime hooks stay STRONG and global: the runtime calls them by name.
        # routing() is one only for a library built before per-library routing
        # (its host.cc still calls the old __Runtime_init_mesh_partition, so the
        # runtime falls back to the global routing()); a current library passes
        # its routing explicitly, so routing() is made local and several
        # libraries coexist (doc/design/aiegraph_python_api.md, multi-library).
        if not _multi_library_merged(merged) and ROUTING_SYMBOL in strong:
            needed.add(ROUTING_SYMBOL)
        hooks = sorted((strong & needed) - keep)
        localize = sorted(strong - keep - set(hooks))
        (tmp / "loc.txt").write_text("\n".join(localize) + "\n")
        subprocess.run([f"{CROSS}objcopy", f"--localize-symbols={tmp / 'loc.txt'}",
                        str(merged)], check=True)

        missing = sorted(k for k in keep if k not in {n for n, t in _nm([merged], "-g", "--defined-only")})
        _iso_backup(archive)
        subprocess.run([f"{CROSS}ar", "rcs", str(archive), str(merged),
                        *[str(tmp / m) for m in runtime]], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if verbose:
        print(f"  [build] isolate {archive.name}: {len(keep) - len(missing)} entry symbol(s) "
              f"+ {len(hooks)} runtime hook(s) ({', '.join(hooks)}) stay global, "
              f"{len(localize)} made local")
    return {"ok": not missing, "kept": sorted(keep), "runtime_hooks": hooks,
            "localized": len(localize), "missing": missing}


#: The runtime's link-time singleton: ``__Runtime_routing_init`` calls ONE
#: global ``routing(XAie_DevInst*)``. Two aiehlc libraries whose host objects
#: are both pulled into an ELF each define it -- unless they were built with
#: per-library routing (:func:`multi_library`), where it is local to each.
ROUTING_SYMBOL = "_Z7routingP12XAie_DevInst"
#: What a per-library-routing host.cc calls (aiehlc's generated aieArray).
MULTI_LIBRARY_API = "__Runtime_init_mesh_partition_with"


def _multi_library_merged(obj: Path) -> bool:
    return any(MULTI_LIBRARY_API in n for n, _ in _nm([obj], "-u"))


def multi_library(archive) -> bool:
    """True when *archive*'s own objects pass their routing to the runtime
    (``__Runtime_init_mesh_partition_with``): such libraries can share an ELF."""
    a = Path(archive)
    if not a.is_file():
        return False
    members = subprocess.run([f"{CROSS}ar", "t", str(a)], capture_output=True, text=True).stdout.split()
    own = [m for m in members if not m.startswith(_RUNTIME_PREFIXES)]
    if not own:
        return False
    tmp = Path(tempfile.mkdtemp(prefix="aiek_ml_"))
    try:
        subprocess.run([f"{CROSS}ar", "x", str(a.resolve()), *own], cwd=tmp, check=True)
        return any(_multi_library_merged(tmp / m) for m in own)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def link_collisions(archives) -> list:
    """Strong symbols defined by the own (non-runtime) members of more than one
    archive. Each entry: ``(symbol, [archive names])``.

    Pass only archives whose adapters the driver will CALL: ``ld`` pulls an
    archive member only to resolve an undefined symbol, so an archive nothing
    references contributes nothing and cannot collide. A hit on
    :data:`ROUTING_SYMBOL` is not a naming accident but the runtime's one-
    ``routing()``-per-ELF design -- see ``flow._check_one_aie_library``."""
    owners = {}
    for a in archives:
        members = subprocess.run([f"{CROSS}ar", "t", str(a)], capture_output=True,
                                 text=True).stdout.split()
        own = [m for m in members if not m.startswith(_RUNTIME_PREFIXES)]
        if not own:
            continue
        tmp = Path(tempfile.mkdtemp(prefix="aiek_lc_"))
        try:
            subprocess.run([f"{CROSS}ar", "x", str(Path(a).resolve()), *own], cwd=tmp, check=True)
            for name, t in _nm([tmp / m for m in own], "-g", "--defined-only"):
                if t in "TDBRAGSC":
                    owners.setdefault(name, set()).add(Path(a).name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return sorted((s, sorted(o)) for s, o in owners.items() if len(o) > 1)


def install(spec, worklocal: Path, aie_dir: Path, layer_calls, verbose: bool = True) -> dict:
    """Copy *worklocal* to *aie_dir*, add the adapters, archive and isolate.

    *layer_calls* is ``[[LayerCall, ...], ...]`` -- one list per selected layer.
    Returns ``{"ok", "archive", "entries": {symbol: adapter}, "reason"?}``.
    """
    from frontend.tvmrelay import aie_layer_lib

    aie_dir = Path(aie_dir)
    if aie_dir.exists():
        shutil.rmtree(aie_dir)
    # build/*.a and build/host.o are deleted below before hostcompile.sh anyway: don't copy them
    # (430 MB for UNet). Large read-only parts are hardlinked (F15, _wl_copytree).
    _wl_copytree(worklocal, aie_dir, skip=lambda rel: rel == "build/host.o" or (
        rel.startswith("build/") and rel.endswith(".a") and rel.count("/") == 1))
    host = aie_dir / "host.cc"
    text = host.read_text()
    cut = text.find(ADAPTER_MARK)
    if cut != -1:
        text = text[:cut].rstrip() + "\n"
    blocks = [adapter_source(spec, calls[0], target="aie", header=(i == 0))
              for i, calls in enumerate(layer_calls)]
    host.write_text(text.rstrip() + "\n\n" + "\n".join(blocks))

    stem = archive_name(spec)
    build = aie_dir / "build"
    for stale in [build / "host.o", *build.glob("*.a")]:
        if stale.exists():
            stale.unlink()
    built = aie_layer_lib.build_archive(aie_dir, stem, REPO, verbose=False)
    if not built.get("ok"):
        return {"ok": False, "archive": None, "reason": built.get("reason"),
                "log": built.get("log")}
    entries = {calls[0].symbol: entry_symbol(calls[0].symbol) for calls in layer_calls}
    iso = isolate_archive(Path(built["archive"]),
                          [*entries.values(), spec.entry, *spec.exports], verbose)
    if not iso["ok"]:
        return {"ok": False, "archive": built["archive"], "isolate": iso,
                "reason": f"after isolation {', '.join(iso['missing'])} not defined -- "
                          f"the adapter/entry did not compile into host.o"}
    return {"ok": True, "archive": built["archive"], "entries": entries, "isolate": iso}
