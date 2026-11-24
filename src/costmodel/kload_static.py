#!/usr/bin/env python3
"""Static per-launch kernel-load words (kload_w) and core resets (rst) from a build directory (M1.2.2 P2).

Nothing here reads a board counter: the prediction is a pure function of the build products.  The board logs
are read only for the comparison columns (measured kload_w = ``LAUNCH split_words kload_w``, measured rst =
``LAUNCH split_detail calls core_reset / launches``).

kload_w = MMIO words the runtime charges to PH_KLOAD per launch (aie_runtime.c g_wc_ph[PH_KLOAD]: Write32 /
MaskWrite32 = 1, BlockWrite32 / BlockSet32 = n, PMLOAD_WIDE PM writes = n via RT_WC_MMIO; ZFILL_WIDE stores are
not counted).  Per kernel-group load call (__Runtime_load_kernel_group_nt, aie_runtime.c:7590-7633):

  * MMIO path, per AIE core tile (row >= 3):
      - rt_relaunch_dma_reset (every load but the very first): XAie_DmaChannelResetAll reset + unreset over the
        AIE2PS tile DMA's 2 S2MM + 2 MM2S channels (xaie2psgbl_reginit.c Aie2PSTileDmaMod.NumChannels=2)
        = 8 MaskWrite32;
      - XAie_CoreDisable + XAie_CoreReset + XAie_CoreUnreset = 3 MaskWrite32;
      - the ELF (__Runtime_load_elf_mem_skip_bss when -DAIEHLC_SKIP_BSS_DEFAULT!=0, else XAie_LoadElfMem), per
        PT_LOAD segment:
          bss (p_filesz==0) with skip-bss mode 1 ................................. 0
          PM segment (p_paddr < ProgMemSize) and the tile already holds this ELF and
            -DAIEHLC_PM_WARM_SKIP=1 (rt_pm_warm_resident) ........................ 0
          bss with -DAIEHLC_ZFILL_WIDE=1 (own DM, 128-bit stores, not counted) ..... 0
          PM segment: PMLOAD_WIDE or aie-rt _XAie_LoadProgMemSection ............... ceil(p_memsz/4)
          DM segment: aie-rt _XAie_DataMemoryBlockWrite per 64 KiB DM bank chunk ..... [lead MaskWrite] +
                                                                                    floor(rest/4) + [tail MaskWrite]
        The "already holds this ELF" state is replayed launch by launch: kernel builds (P0 host) relaunch one
        ELF, so every timed launch is warm; the UNet build replays the graph_run launch order (kswitch.py).
  * control-packet path (__Runtime_load_kernel_group_*_ctrl, aot / aot-kco): the ELF body goes by DMA
    (rt_ctrl_load_elf_aot), PH_KLOAD only sees the fixed MMIO kick of rt_ctrl_elf_dma_packed
    (shim entry + return arm + shim BD push + queue) = CTRL_KLOAD_W per load, independent of the ELF.

rst per launch: costmodel.rst_static.rst_per_launch (M1.2.2 P0).

CLI:
  kload_static.py --build-root R --kernel-logs DIR [--unet CFG=LOG ...] [--alt-logs DIR]
      kernel points: DIR/<cfg>/<kernel>.log for cfg in off, aot, aot-kco (fixed set, sorted), build R/k_<cfg>/<kernel>
      UNet points  : --unet CFG=LOG, build R/unet_<CFG>
      alt points   : --alt-logs DIR -> DIR/*.log, cfg=off-alt, build R/k_off-alt/<pair>
  Prints per point
      KSTATIC cfg=<cfg> {k=<kernel>|target=unet} kload_w=<pred>/<meas> rst=<pred>/<meas> kload_w_rel=<rel> PASS|FAIL
  and  KSTATIC ALL [src=alt] n=<n> maxrel=<max rel> PASS|FAIL   (rel = max of the kload_w and rst relative errors;
  0/0 -> 0; measured <= 16 words -> absolute tolerance 16 words; PASS <= 0.050, UNet off kload_w <= 0.025).
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from costmodel import kswitch  # noqa: E402
from costmodel.launch_split import lines  # noqa: E402
from costmodel.rst_static import AIE_TILE_ROW_START, rst_per_launch  # noqa: E402

PROG_MEM_SIZE = 16 * 1024  # xaie2psgbl_reginit.c AIE2PS core ProgMemSize
DATA_MEM_SIZE = 64 * 1024  # AIE2PS tile DataMemSize (per bank window, _XAie_LoadDataMemSection AddrMask)
DMA_NCH = 2  # Aie2PSTileDmaMod.NumChannels (per direction)
DMA_RESET_W = 2 * 2 * DMA_NCH  # reset + unreset over S2MM+MM2S
CORE_CTRL_W = 3  # CoreDisable, CoreReset, CoreUnreset
CTRL_KLOAD_W = 11  # rt_ctrl_elf_dma_packed fixed MMIO kick per control-packet load
REL_TOL = 0.050
UNET_OFF_TOL = 0.025
ABS_W = 16.0
KCFGS = ("off", "aot", "aot-kco")

_LOAD = re.compile(r"__Runtime_load_kernel_group_(\d+)t(_ctrl)?\(([^;]*)\);")
_TILELOC = re.compile(r"XAie_LocType\s+(v\d+)\s*=\s*XAie_TileLoc\((\d+),\s*(\d+)\)")
_DEF = re.compile(r"-DAIEHLC_(PROFILING|ZFILL_WIDE|PMLOAD_WIDE|PM_WARM_SKIP|SKIP_BSS_DEFAULT)=(\d+)")
_HOSTS = ("aout/worklocal/build/host_fixed.cc", "ag_work/aiek/*/worklocal/build/host_fixed.cc")


# ------------------------------------------------------------------ build products
def build_defs(b: Path) -> dict:
    """runtime -D switches of the build (kernel: <b>/build.log; UNet: <b>.build.log '[build_unet] ... defs=')"""
    d = {}
    for p in (b / "build.log", b.parent / (b.name + ".build.log")):
        if p.is_file():
            for k, v in _DEF.findall(p.read_text(errors="replace")):
                d[k] = max(d.get(k, 0), int(v))
    return d


def load_sites(b: Path):
    """[(ctrl, n_core)] for every kernel-group load call in the generated host code"""
    out = []
    for pat in _HOSTS:
        for h in sorted(b.glob(pat)):
            text = h.read_text(errors="replace")
            for m in _LOAD.finditer(text):
                w, ctrl, args = int(m.group(1)), bool(m.group(2)), [a.strip() for a in m.group(3).split(",")]
                first = 2 if ctrl else 1
                tiles, n = args[first:first + w], int(args[first + w])
                fn = text.rfind("\nvoid ", 0, m.start())
                locs = {v: (int(c), int(r)) for v, c, r in _TILELOC.findall(text, max(fn, 0), m.start())}
                rows = [locs[t][1] for t in tiles[:n]]
                out.append((ctrl, sum(1 for r in rows if r >= AIE_TILE_ROW_START)))
    if not out:
        raise FileNotFoundError(f"{b}: no __Runtime_load_kernel_group_* call")
    return out


def elf_from_obj(obj: Path, sym: str) -> bytes:
    """bytes of _binary_<x>_start.._end inside an `ld -r -b binary` aarch64 object (the blob the runtime loads)"""
    data = obj.read_bytes()
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sh = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    syms = {}
    for s in sh:
        if s[1] == 2:  # SHT_SYMTAB
            strtab = sh[s[6]]
            for off in range(s[4], s[4] + s[5], 24):
                name, _info, _oth, shndx, val, _sz = struct.unpack_from("<IBBHQQ", data, off)
                nm = data[strtab[4] + name:data.index(b"\0", strtab[4] + name)].decode()
                syms[nm] = (shndx, val)
    base = sym[:-len("_start")]
    shndx, start = syms[base + "_start"]
    _, end = syms[base + "_end"]
    sec = sh[shndx]
    return data[sec[4] + start:sec[4] + end]


def elf_segments(elf: bytes):
    """PT_LOAD (paddr, filesz, memsz) of an AIE ELF32"""
    if elf[:4] != b"\x7fELF":
        raise ValueError("not an ELF")
    phoff, = struct.unpack_from("<I", elf, 0x1C)
    phentsize, phnum = struct.unpack_from("<HH", elf, 0x2A)
    out = []
    for i in range(phnum):
        p_type, _off, _va, pa, fsz, msz, _fl, _al = struct.unpack_from("<8I", elf, phoff + i * phentsize)
        if p_type == 1:
            out.append((pa, fsz, msz))
    return out


def _dm_words(paddr: int, size: int) -> int:
    """_XAie_LoadDataMemSection -> _XAie_DataMemoryBlockWrite word count"""
    w = 0
    while size > 0:
        off = paddr & (DATA_MEM_SIZE - 1)
        n = min(size, DATA_MEM_SIZE - off)
        lead = (4 - (off & 3)) & 3
        lead = min(lead, n)
        rest = n - lead
        w += (1 if lead else 0) + rest // 4 + (1 if rest % 4 else 0)
        size -= n
        paddr += n
    return w


def tile_words(segs, defs: dict) -> tuple:
    """(pm_words, other_words) one tile load writes for this ELF under the build's switches"""
    skip = defs.get("SKIP_BSS_DEFAULT", 0)
    if skip == 2:
        raise NotImplementedError("skip-bss mode 2 (port-buffer bss) is not modelled")
    pm = other = 0
    for pa, fsz, msz in segs:
        is_pm = pa < PROG_MEM_SIZE
        if fsz == 0 and skip == 1:
            continue
        if is_pm and fsz:
            pm += (msz + 3) // 4
            continue
        if fsz == 0 and skip and defs.get("ZFILL_WIDE"):
            continue
        other += _dm_words(pa, msz)
    return pm, other


# ------------------------------------------------------------------ per-build prediction
_ELFSYM = re.compile(r"__Runtime_set_kernel_elf\((_binary_\w+_start)\)")


def _kload_alt(b: Path, sites, defs: dict, warm_skip: bool) -> dict:
    """p0_one_kernel.sh --alt build: kernels A and B on the same tiles, timed launches A,B,A,B... -> every timed
    launch loads the ELF the tiles do not hold (PM never warm-resident), mean over the two ELFs."""
    objdir = b / "aout/worklocal/build"
    blobs = list(dict.fromkeys(_ELFSYM.findall((objdir / "host_fixed.cc").read_text(errors="replace"))))
    if len(blobs) != 2:
        raise ValueError(f"{b}: alt build wants 2 kernel ELFs, found {len(blobs)}")
    per = []
    for (_, n_core), blob in zip(sites, blobs):
        pm, other = tile_words(elf_segments(elf_from_obj(objdir / (blob[len("_binary_"):-len("_start")] + ".o"),
                                                         blob)), defs)
        per.append((n_core * (DMA_RESET_W + CORE_CTRL_W + other + pm), n_core, pm, other))
    kw = sum(p[0] for p in per) / len(per)
    return {"kload_w": float(kw), "why": "P0 alt A,B: " + " | ".join(
        f"n_core={n} pm={pm} other={o}" for _, n, pm, o in per) + f" warm_skip={int(warm_skip)} (alternating: cold)"}


def kload_static(build_dir) -> dict:
    b = Path(build_dir)
    defs = build_defs(b)
    sites = load_sites(b)
    if all(c for c, _ in sites):
        return {"kload_w": float(CTRL_KLOAD_W), "why": f"{len(sites)} ctrl load sites"}
    if any(c for c, _ in sites):
        raise ValueError(f"{b}: mixed ctrl/mmio load sites")
    warm_skip = bool(defs.get("PM_WARM_SKIP"))
    if (b / "kacc_gemm.cc").is_file():  # P0 kernel host: one ELF, relaunched (all timed launches steady)
        if len(sites) == 2 and (b / "p0_alt_host.cc").is_file():
            return _kload_alt(b, sites, defs, warm_skip)
        if len(sites) != 1:
            raise ValueError(f"{b}: {len(sites)} load sites in a P0 kernel build")
        n_core = sites[0][1]
        pm, other = tile_words(elf_segments((b / "aout/worklocal/build/kernel").read_bytes()), defs)
        per_tile = DMA_RESET_W + CORE_CTRL_W + other + (0 if warm_skip else pm)
        return {"kload_w": float(n_core * per_tile), "why": f"P0 n_core={n_core} pm={pm} other={other} "
                f"warm_skip={int(warm_skip)}"}
    # UNet aiegraph build: replay the launch order on the 16 tiles
    req_warm = warm_skip
    launches, _ = kswitch.unet_launches(str(b), require_warm=req_warm)
    objdir = b / "ag_work/aiek/aiek_unetconv/worklocal/build"
    cache = {}
    resident = {}
    tot = 0
    cold_tiles = 0
    for i, L in enumerate(launches):
        blob = L["blob"]
        if blob not in cache:
            obj = objdir / (blob[len("_binary_"):-len("_start")] + ".o")
            cache[blob] = tile_words(elf_segments(elf_from_obj(obj, blob)), defs)
        pm, other = cache[blob]
        for t in L["tiles"]:
            if t[1] < AIE_TILE_ROW_START:
                continue
            cold = not (warm_skip and resident.get(t) == blob)
            resident[t] = blob
            cold_tiles += cold
            tot += (DMA_RESET_W if i else 0) + CORE_CTRL_W + other + (pm if cold else 0)
    n = len(launches)
    return {"kload_w": tot / n, "why": f"unet launches={n} elfs={len(cache)} pm_tile_loads={cold_tiles} "
            f"warm_skip={int(warm_skip)}"}


# ------------------------------------------------------------------ measured side (comparison only)
_SW = re.compile(r"LAUNCH split_words .*?kload_w=([0-9.]+)")
_DET = re.compile(r"LAUNCH split_detail calls .*?core_reset=(\d+)")
_LAUNCHES = re.compile(r"\blaunches=(\d+)")
_CMP = re.compile(r"^CMP \S+ .*?aie_launches=(\d+)")


def measured(log: str) -> dict:
    kw = rst = launches = None
    for ln in lines(log):
        m = _SW.search(ln)
        if m:
            kw = float(m.group(1))
        m = _DET.search(ln)
        if m:
            rst = int(m.group(1))
            ml = _LAUNCHES.search(ln[m.end():])
            if ml:
                launches = int(ml.group(1))
        m = _CMP.search(ln)
        if m and launches is None:
            launches = int(m.group(1))
    if kw is None or rst is None:
        raise ValueError(f"{log}: no split_words kload_w / split_detail core_reset")
    if not launches:
        m = re.search(r"LAUNCH warm .*?launches=(\d+)", Path(log).read_text(errors="replace"))
        launches = int(m.group(1)) if m else None
    if not launches:
        raise ValueError(f"{log}: launch count not found")
    return {"kload_w": kw, "rst": rst / launches}


def rel(pred: float, meas: float) -> float:
    if pred == 0 and meas == 0:
        return 0.0
    if meas <= ABS_W:
        return 0.0 if abs(pred - meas) <= ABS_W else abs(pred - meas) / max(meas, 1e-9)
    return abs(pred - meas) / meas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-root", required=True)
    ap.add_argument("--kernel-logs", help="DIR/<cfg>/<kernel>.log for cfg in off, aot, aot-kco")
    ap.add_argument("--unet", nargs="*", default=[], metavar="CFG=LOG")
    ap.add_argument("--alt-logs", help="DIR/*.log, cfg=off-alt, build <root>/k_off-alt/<pair>")
    a = ap.parse_args(argv)
    R = Path(a.build_root)
    pts = []  # (cfg, label, build, log)
    if a.kernel_logs:
        for cfg in KCFGS:
            d = Path(a.kernel_logs) / cfg
            for log in sorted(d.glob("*.log")):
                pts.append((cfg, "k=" + log.stem, R / f"k_{cfg}" / log.stem, log))
    for spec in a.unet:
        cfg, _, log = spec.partition("=")
        if not log:
            ap.error(f"--unet {spec}: want CFG=LOG")
        pts.append((cfg, "target=unet", R / f"unet_{cfg}", Path(log)))
    if a.alt_logs:
        for log in sorted(Path(a.alt_logs).glob("*.log")):
            pts.append(("off-alt", "k=" + log.stem, R / "k_off-alt" / log.stem, log))
    if not pts:
        ap.error("no points")
    worst = 0.0
    ok_all = True
    for cfg, label, b, log in pts:
        try:
            kp = kload_static(b)
            rp = rst_per_launch(b)
            ms = measured(str(log))
        except Exception as e:  # noqa: BLE001
            print(f"KSTATIC cfg={cfg} {label} ERROR {type(e).__name__}: {e}")
            ok_all = False
            worst = max(worst, 1.0)
            continue
        kr = rel(kp["kload_w"], ms["kload_w"])
        rr = rel(rp, ms["rst"])
        r = max(kr, rr)
        tol = UNET_OFF_TOL if (cfg == "off" and label == "target=unet") else REL_TOL
        ok = kr <= tol and r <= REL_TOL
        ok_all &= ok
        worst = max(worst, r)
        print(f"KSTATIC cfg={cfg} {label} kload_w={kp['kload_w']:.1f}/{ms['kload_w']:.1f} rst={rp:.1f}/{ms['rst']:.1f} "
              f"kload_w_rel={kr:.3f} {'PASS' if ok else 'FAIL'}")
        print(f"  why: {kp['why']}; build={b}")
    ok_all &= worst <= REL_TOL
    src = "src=alt " if all(c == "off-alt" for c, *_ in pts) else ""
    print(f"KSTATIC ALL {src}n={len(pts)} maxrel={worst:.3f} {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
