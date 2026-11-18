#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiehlc_elfio -- swap the inputs of a built board ELF / read its output back.

An ELF built with ``arm_build.build_arm_elf(runtime_io=...)`` (aiegraph
``build_unet.py --runtime-io``) carries its inputs in ONE section
``.aiehlc_io`` (layout: ``frontend/tvmrelay/arm_build.py`` "Runtime IO
section"). This tool needs no rebuild and no toolchain:

  aiehlc_elfio.py info    <elf>
  aiehlc_elfio.py patch   <elf> -o <out.elf> --in <name>=<file.bin> ... [--dump-out|--no-dump-out]
      -> PATCHED crc32=<8hex> outside_io_diff=<n> bytes=<n>
         crc32 = new header crc (crc32 of all inputs in order); outside_io_diff =
         bytes of <out.elf> that differ from <elf> OUTSIDE the section (must be 0);
         bytes = input bytes written by --in.
  aiehlc_elfio.py readout <board.log> -o <out.npy> --ref <ref.npy|ref.bin(f32)>
      -> READOUT elems=<n> crc_ok=<0|1> cos=<f> maxabs=<f> thr=0.999 cos_OK=<0|1>
         parses the AIO_BEGIN / AIO_D / AIO_END dump the ELF prints with dump_out set.
         The threshold is the fixed constant COS_THR (no override).
Printed field orders are pinned by test_elfio.py (gates match them).
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

SECTION = b".aiehlc_io"
MAGIC = b"AIO1"
HDR = struct.Struct("<4sIIIIIII")
ENT = struct.Struct("<40sIIIIQ")
FLAG_DUMP_OUT = 1
COS_THR = 0.999   # fixed on purpose: no CLI override


class ElfioError(Exception):
    pass


def find_section(f):
    """-> (file_offset, size, sh_addr) of .aiehlc_io in an ELF64 LE file; checks it is loaded."""
    f.seek(0)
    eh = f.read(64)
    if eh[:4] != b"\x7fELF" or eh[4] != 2 or eh[5] != 1:
        raise ElfioError("not an ELF64 little-endian file")
    e_phoff, e_shoff = struct.unpack_from("<QQ", eh, 32)
    e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHHHH", eh, 54)
    f.seek(e_shoff)
    shs = [struct.unpack("<IIQQQQIIQQ", f.read(e_shentsize)[:64]) for _ in range(e_shnum)]
    st = shs[e_shstrndx]
    f.seek(st[4])
    strtab = f.read(st[5])
    hit = [s for s in shs if strtab[s[0]:strtab.index(b"\0", s[0])] == SECTION]
    if len(hit) != 1:
        raise ElfioError(f"{len(hit)} {SECTION.decode()} sections (need exactly 1; built without --runtime-io?)")
    name, typ, flags, addr, off, size = hit[0][:6]
    if typ != 1 or not flags & 2:   # SHT_PROGBITS, SHF_ALLOC
        raise ElfioError(f"{SECTION.decode()} is not an allocated PROGBITS section (type={typ} flags={flags:#x})")
    f.seek(e_phoff)
    phs = [struct.unpack("<IIQQQQQQ", f.read(e_phentsize)[:56]) for _ in range(e_phnum)]
    if not any(p[0] == 1 and p[2] <= off and off + size <= p[2] + p[5] for p in phs):
        raise ElfioError(f"{SECTION.decode()} is not inside a PT_LOAD segment's file image")
    return off, size, addr


def parse(img: bytes) -> dict:
    magic, ver, n, flags, crc, hdr_bytes, total, _ = HDR.unpack_from(img)
    if magic != MAGIC or ver != 1:
        raise ElfioError(f"bad .aiehlc_io header magic={magic!r} version={ver}")
    if total != len(img):
        raise ElfioError(f"header total_bytes={total} != section size {len(img)}")
    ents = []
    for k in range(n):
        nm, off, nb, ecrc, _, _ = ENT.unpack_from(img, HDR.size + ENT.size * k)
        if off < hdr_bytes or off + nb > total:
            raise ElfioError(f"input {k} out of the section: off={off} bytes={nb}")
        ents.append(dict(name=nm.rstrip(b"\0").decode(), off=off, bytes=nb, crc=ecrc))
    return dict(n=n, flags=flags, crc=crc, hdr_bytes=hdr_bytes, total=total, ents=ents)


def patch_image(img: bytes, repl: dict, dump_out=None):
    """-> (new image, new crc, input bytes written). ``repl``: {name: bytes}."""
    h = parse(img)
    out = bytearray(img)
    names = [e["name"] for e in h["ents"]]
    for nm in repl:
        if nm not in names:
            raise ElfioError(f"--in {nm}: no such input (section has {', '.join(names)})")
    written, crc = 0, 0
    for k, e in enumerate(h["ents"]):
        if e["name"] in repl:
            d = repl[e["name"]]
            if len(d) != e["bytes"]:
                raise ElfioError(f"--in {e['name']}: {len(d)} bytes, the section slot holds {e['bytes']}")
            out[e["off"]:e["off"] + e["bytes"]] = d
            written += len(d)
        d = bytes(out[e["off"]:e["off"] + e["bytes"]])
        ecrc = zlib.crc32(d)
        crc = zlib.crc32(d, crc)
        p = HDR.size + ENT.size * k
        nm, off, nb, _, r0, r1 = ENT.unpack_from(out, p)
        ENT.pack_into(out, p, nm, off, nb, ecrc, r0, r1)
    flags = h["flags"]
    if dump_out is not None:
        flags = (flags | FLAG_DUMP_OUT) if dump_out else (flags & ~FLAG_DUMP_OUT)
    struct.pack_into("<II", out, 12, flags, crc)
    return bytes(out), crc, written


def count_diff(a_path, b_path, skip_off, skip_len, chunk=1 << 24) -> int:
    """bytes that differ between two files outside [skip_off, skip_off+skip_len) (+ any size difference)"""
    import numpy as np
    diff = 0
    with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
        pos = 0
        while True:
            a, b = fa.read(chunk), fb.read(chunk)
            if not a and not b:
                break
            n = min(len(a), len(b))
            diff += abs(len(a) - len(b))
            if n:
                d = np.frombuffer(a[:n], np.uint8) != np.frombuffer(b[:n], np.uint8)
                lo, hi = max(skip_off - pos, 0), min(skip_off + skip_len - pos, n)
                if lo < hi:
                    d[lo:hi] = False
                diff += int(d.sum())
            pos += max(len(a), len(b))
    return diff


def cmd_info(a):
    with open(a.elf, "rb") as f:
        off, size, addr = find_section(f)
        f.seek(off)
        h = parse(f.read(size))
    print(f"AIO section off={off:#x} addr={addr:#x} bytes={size} n_in={h['n']} crc32={h['crc']:08x} "
          f"dump_out={h['flags'] & FLAG_DUMP_OUT}")
    for e in h["ents"]:
        print(f"AIO input {e['name']} off={e['off']} bytes={e['bytes']} crc32={e['crc']:08x}")
    return 0


def cmd_patch(a):
    import shutil
    repl = {}
    for spec in a.inputs or ():
        if "=" not in spec:
            raise ElfioError(f"--in {spec}: want <name>=<file>")
        nm, fn = spec.split("=", 1)
        repl[nm] = Path(fn).read_bytes()
    src, dst = Path(a.elf), Path(a.out)
    if src.resolve() == dst.resolve():
        raise ElfioError("-o must differ from the input ELF (the original is the A/B base)")
    with open(src, "rb") as f:
        off, size, _ = find_section(f)
        f.seek(off)
        img = f.read(size)
    new, crc, written = patch_image(img, repl, a.dump_out)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    with open(tmp, "r+b") as f:
        f.seek(off)
        f.write(new)
    tmp.replace(dst)
    diff = count_diff(src, dst, off, size)
    print(f"PATCHED crc32={crc:08x} outside_io_diff={diff} bytes={written}", flush=True)
    return 0 if diff == 0 else 1


def parse_dump(text: str):
    """-> (data bytes, crc_ok, elems) from the AIO_BEGIN/AIO_D/AIO_END lines of a board log (last dump wins)."""
    import re
    blocks, cur = [], None
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.startswith("AIO_BEGIN "):
            cur = {"d": {}, "end": None}
        elif ln.startswith("AIO_D ") and cur is not None:
            m = re.match(r"AIO_D ([0-9a-f]+) ([0-9a-f]*)$", ln)
            if m:
                cur["d"][int(m.group(1), 16)] = bytes.fromhex(m.group(2))
        elif ln.startswith("AIO_END ") and cur is not None:
            cur["end"] = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
            blocks.append(cur)
            cur = None
    if not blocks:
        raise ElfioError("no complete AIO_BEGIN..AIO_END dump in the log (ELF patched with --dump-out?)")
    b = blocks[-1]
    nbytes, elems = int(b["end"]["bytes"]), int(b["end"]["elems"])
    buf = bytearray(nbytes)
    got = 0
    for o, d in b["d"].items():
        if o + len(d) <= nbytes:
            buf[o:o + len(d)] = d
            got += len(d)
    crc_ok = int(got == nbytes and "%08x" % zlib.crc32(bytes(buf)) == b["end"].get("crc32"))
    return bytes(buf), crc_ok, elems


def load_ref(p: Path):
    import numpy as np
    return np.load(p).astype(np.float64).ravel() if p.suffix == ".npy" else np.fromfile(p, "<f4").astype(np.float64)


def compare(got, ref):
    import numpy as np
    g = np.asarray(got, np.float64).ravel()
    r = np.asarray(ref, np.float64).ravel()
    if g.size != r.size or not np.all(np.isfinite(g)):
        return 0.0, float("inf")
    ng, nr = np.linalg.norm(g), np.linalg.norm(r)
    cos = float(g @ r / (ng * nr)) if ng > 0 and nr > 0 else 0.0
    return cos, float(np.max(np.abs(g - r)))


def readout_line(elems: int, crc_ok: int, cos: float, maxabs: float) -> str:
    ok = int(bool(crc_ok) and cos >= COS_THR)
    return f"READOUT elems={elems} crc_ok={crc_ok} cos={cos:.6f} maxabs={maxabs:g} thr={COS_THR} cos_OK={ok}"


def cmd_readout(a):
    import numpy as np
    data, crc_ok, elems = parse_dump(Path(a.log).read_text(errors="replace"))
    got = np.frombuffer(data, "<f4")
    if a.out:
        np.save(a.out, got.astype(np.float32))
    cos, mx = compare(got, load_ref(Path(a.ref)))
    line = readout_line(elems if elems == got.size else got.size, crc_ok, cos, mx)
    print(line, flush=True)
    return 0 if line.endswith("cos_OK=1") else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("info")
    p.add_argument("elf")
    p = sp.add_parser("patch")
    p.add_argument("elf")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--in", dest="inputs", action="append", metavar="NAME=FILE")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dump-out", dest="dump_out", action="store_const", const=True, default=None)
    g.add_argument("--no-dump-out", dest="dump_out", action="store_const", const=False)
    p = sp.add_parser("readout")
    p.add_argument("log")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--ref", required=True)
    a = ap.parse_args(argv)
    try:
        return {"info": cmd_info, "patch": cmd_patch, "readout": cmd_readout}[a.cmd](a)
    except (ElfioError, OSError) as e:
        print(f"aiehlc_elfio: ERROR {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
