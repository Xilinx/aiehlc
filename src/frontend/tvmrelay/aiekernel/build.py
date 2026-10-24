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
3. **isolate** -- merge the archive's own objects (``host.o``, ``routing.o``,
   ``kernel*.o``) with ``ld -r`` and make every global LOCAL except the
   adapters, the entry, the spec's ``export`` lines, and the RUNTIME HOOKS --
   symbols the shared runtime members reference by name (``routing``,
   ``g_runtime_debug_level``). The user's helpers, ``host_canonicalized``,
   ``_binary_kernel_<func>_*`` and any public API of a wrapped library stop
   being able to collide with anything else in the ELF.

   The hooks are deliberately left strong. ``__Runtime_routing_init`` calls
   ONE global ``routing(dev)``, and partition init is cached per ``meshId``:
   the runtime is one-aiehlc-library-per-ELF by design. Weakening ``routing``
   would let a second library link cleanly and then run with the first one's
   stream-switch configuration -- the worst possible failure. Kept strong, two
   libraries collide, and :func:`link_collisions` reports it before ``make``.
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
    return h.hexdigest()


def aiehlc_compile(spec, out_dir, aie_version: int = 5, verbose: bool = True) -> dict:
    """Run aiehlc on the kernel source. Returns ``{"ok", "worklocal"|"reason", "log", "cached"}``."""
    work = Path(out_dir) / "aiek" / spec.name
    work.mkdir(parents=True, exist_ok=True)
    dest, log, stamp_file = work / "worklocal", work / "aiehlc.log", work / "aiehlc.stamp"
    stamp = _stamp(spec)
    if (dest / "host.cc").is_file() and stamp_file.is_file() \
            and stamp_file.read_text().strip() == stamp:
        return {"ok": True, "worklocal": dest, "log": str(log), "cached": True}

    # Scratch next to the cache, not in /tmp: aiehlc's aout/ is large and /tmp
    # is often a small shared volume (a full one fails the build mid-way).
    scratch = Path(tempfile.mkdtemp(prefix="aiehlc_run_", dir=work))
    cmd = (f'source "{REPO}/script/aiehlc.sh" --aie-version {aie_version} '
           f'--runtime-source-file "{spec.source}"')
    start = time.time()
    try:
        with log.open("w") as fh:
            rc = subprocess.run(["bash", "-c", cmd], cwd=str(scratch),
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        src = scratch / "aout" / "worklocal"
        if rc != 0 or not (src / "host.cc").is_file() or not (src / "kernel.cc").is_file():
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


def isolate_archive(archive: Path, keep, verbose: bool = True) -> dict:
    """Localize every own global of *archive* except *keep* (see module doc)."""
    archive = Path(archive).resolve()
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
        needed = {n for n, _ in _nm([tmp / m for m in runtime], "-u")}
        keep = set(keep)
        # Runtime hooks stay STRONG and global: the runtime calls them by name
        # (routing(dev) from __Runtime_routing_init). Weakening them would let
        # a second library link "fine" and run with the first one's routing.
        hooks = sorted((strong & needed) - keep)
        localize = sorted(strong - keep - set(hooks))
        (tmp / "loc.txt").write_text("\n".join(localize) + "\n")
        subprocess.run([f"{CROSS}objcopy", f"--localize-symbols={tmp / 'loc.txt'}",
                        str(merged)], check=True)

        missing = sorted(k for k in keep if k not in {n for n, t in _nm([merged], "-g", "--defined-only")})
        backup = archive.parent.parent / f"isolate_backup_{archive.name}.bak"
        shutil.copy2(archive, backup)       # not *.a: the Makefile wildcard must not see it
        archive.unlink()
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
#: are both pulled into an ELF each define it.
ROUTING_SYMBOL = "_Z7routingP12XAie_DevInst"


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
    shutil.copytree(worklocal, aie_dir, symlinks=True)
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
