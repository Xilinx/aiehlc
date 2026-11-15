"""Unit tests for aiehlc_elfio.py + arm_build's .aiehlc_io section (GIN, AIEHLC GAP-aiegraph-arm_build-a20e67).

Pins the printed formats the sprint gates match:
  PATCHED crc32=<8hex> outside_io_diff=<n> bytes=<n>
  READOUT elems=<n> crc_ok=<0|1> cos=<f> maxabs=<f> thr=0.999 cos_OK=<0|1>
  CMP unet skipped runtime-input aie_launches=<n> run_rc=<rc>
Run: python3 -m pytest -q src/tool/elfio/test_elfio.py
"""
import re
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "example" / "aiegraph" / "unet_bk_sdm_tiny"))

import aiehlc_elfio as eio  # noqa: E402
import frontend.tvmrelay.arm_build as ab  # noqa: E402

PATCHED_RE = re.compile(r"^PATCHED crc32=([0-9a-f]{8}) outside_io_diff=(\d+) bytes=(\d+)$")
READOUT_RE = re.compile(r"^READOUT elems=(\d+) crc_ok=([01]) cos=(-?[0-9.]+) maxabs=(\S+) thr=0\.999 cos_OK=([01])$")
CMP_SKIP_RE = re.compile(r'printf\("CMP unet skipped runtime-input aie_launches=%d run_rc=%d\\n", (\w+), (\w+)\);')


def _inputs(seed=0):
    r = np.random.default_rng(seed)
    return [("sample", r.standard_normal(4 * 32 * 32).astype("<f4").tobytes()),
            ("timestep_i32", np.array([999 - seed], "<i4").tobytes()),
            ("ehs", r.standard_normal(77 * 768).astype("<f4").tobytes())]


def _fake_elf(path: Path, sect: bytes, pre=b"\x11" * 300, post=b"\x22" * 500, in_load=True):
    """minimal ELF64 LE: [ehdr][phdr][pre][.aiehlc_io @64-aligned][post][shstrtab][shdrs]"""
    body = bytearray(64 + 56)
    body += pre
    while len(body) % 64:
        body += b"\0"
    s_off = len(body)
    body += sect
    body += post
    names = b"\0.aiehlc_io\0.shstrtab\0"
    str_off = len(body)
    body += names
    while len(body) % 8:
        body += b"\0"
    sh_off = len(body)
    shdr = struct.Struct("<IIQQQQIIQQ")
    body += shdr.pack(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    body += shdr.pack(1, 1, 3, 0x100000 + s_off, s_off, len(sect), 0, 0, 64, 0)
    body += shdr.pack(12, 3, 0, 0, str_off, len(names), 0, 0, 1, 0)
    eh = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    eh += struct.pack("<HHIQQQIHHHHHH", 2, 183, 1, 0x100000, 64, sh_off, 0, 64, 56, 1, 64, 3, 2)
    body[:64] = eh
    lo, hi = (0, s_off + len(sect)) if in_load else (0, s_off)
    body[64:120] = struct.pack("<IIQQQQQQ", 1, 7, lo, 0x100000, 0x100000, hi - lo, hi - lo, 64)
    path.write_bytes(bytes(body))
    return s_off


def test_pack_layout_and_crc():
    ins = _inputs()
    img = ab.aio_pack(ins, dump_out=False)
    h = eio.parse(img)
    assert img[:4] == b"AIO1" and h["n"] == 3 and h["flags"] == 0
    assert h["crc"] == zlib.crc32(b"".join(d for _, d in ins))
    assert [e["name"] for e in h["ents"]] == ["sample", "timestep_i32", "ehs"]
    for e, (_, d) in zip(h["ents"], ins):
        assert e["off"] % 64 == 0 and img[e["off"]:e["off"] + e["bytes"]] == d and e["crc"] == zlib.crc32(d)
    assert len(img) % 64 == 0 and h["total"] == len(img)
    assert ab.AIO_HDR.size == eio.HDR.size and ab.AIO_ENT.size == eio.ENT.size


def test_patch_image_swaps_only_inputs():
    a, b = _inputs(0), _inputs(12)
    img = ab.aio_pack(a)
    new, crc, n = eio.patch_image(img, dict(b), dump_out=True)
    assert crc == zlib.crc32(b"".join(d for _, d in b)) and n == sum(len(d) for _, d in b)
    assert new == ab.aio_pack(b, dump_out=True)
    back, crc2, _ = eio.patch_image(new, dict(a), dump_out=False)
    assert back == img and crc2 == zlib.crc32(b"".join(d for _, d in a))


def test_patch_rejects_bad_inputs():
    img = ab.aio_pack(_inputs())
    with pytest.raises(eio.ElfioError, match="no such input"):
        eio.patch_image(img, {"nope": b"x"})
    with pytest.raises(eio.ElfioError, match="slot holds"):
        eio.patch_image(img, {"timestep_i32": b"12345678"})
    with pytest.raises(eio.ElfioError, match="magic"):
        eio.parse(b"XXXX" + img[4:])


def test_patch_cli_line_format(tmp_path, capsys):
    a, b = _inputs(0), _inputs(12)
    elf = tmp_path / "main.elf"
    off = _fake_elf(elf, ab.aio_pack(a))
    args = []
    for nm, d in b:
        (tmp_path / f"{nm}.bin").write_bytes(d)
        args += ["--in", f"{nm}={tmp_path / (nm + '.bin')}"]
    assert eio.main(["patch", str(elf), "-o", str(tmp_path / "b.elf"), *args, "--dump-out"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    m = PATCHED_RE.match(out[0])
    assert m, out[0]
    assert m.group(1) == "%08x" % zlib.crc32(b"".join(d for _, d in b))
    assert m.group(2) == "0" and int(m.group(3)) == sum(len(d) for _, d in b)
    old, new = elf.read_bytes(), (tmp_path / "b.elf").read_bytes()
    size = len(ab.aio_pack(a))
    assert old[:off] == new[:off] and old[off + size:] == new[off + size:]
    assert eio.parse(new[off:off + size])["flags"] & eio.FLAG_DUMP_OUT
    # the original is never patched in place
    assert eio.main(["patch", str(elf), "-o", str(elf), *args]) == 2


def test_patch_requires_loaded_section(tmp_path):
    elf = tmp_path / "x.elf"
    _fake_elf(elf, ab.aio_pack(_inputs()), in_load=False)
    with open(elf, "rb") as f, pytest.raises(eio.ElfioError, match="PT_LOAD"):
        eio.find_section(f)


def test_count_diff_counts_outside_only(tmp_path):
    x, y = tmp_path / "x", tmp_path / "y"
    x.write_bytes(bytes(1000))
    y.write_bytes(bytes(100) + b"\1" * 50 + bytes(849) + b"\1")
    assert eio.count_diff(x, y, 100, 50) == 1
    assert eio.count_diff(x, y, 0, 0) == 51


def _dump_log(arr: np.ndarray, corrupt=False) -> str:
    d = arr.astype("<f4").tobytes()
    lines = ["noise", f"AIO_BEGIN tag=unet bytes={len(d)} elems={arr.size} dtype=f32"]
    for o in range(0, len(d), 64):
        lines.append("AIO_D %06x %s" % (o, d[o:o + 64].hex()))
    crc = zlib.crc32(d) ^ (1 if corrupt else 0)
    lines.append(f"AIO_END tag=unet bytes={len(d)} elems={arr.size} crc32={crc:08x} dump_ms=1.000")
    return "\n".join(lines) + "\n"


def test_readout_line_format_and_threshold(tmp_path, capsys):
    ref = np.random.default_rng(1).standard_normal(4096).astype(np.float32)
    np.save(tmp_path / "ref.npy", ref.reshape(1, 4, 32, 32))
    (tmp_path / "board.log").write_text(_dump_log(ref))
    assert eio.main(["readout", str(tmp_path / "board.log"), "-o", str(tmp_path / "o.npy"),
                     "--ref", str(tmp_path / "ref.npy")]) == 0
    ln = capsys.readouterr().out.strip()
    m = READOUT_RE.match(ln)
    assert m and m.group(1) == "4096" and m.group(2) == "1" and m.group(5) == "1", ln
    assert np.array_equal(np.load(tmp_path / "o.npy"), ref)
    # fixed threshold: cos=0.9985 must fail
    ln = eio.readout_line(4096, 1, 0.9985, 0.5)
    assert ln == "READOUT elems=4096 crc_ok=1 cos=0.998500 maxabs=0.5 thr=0.999 cos_OK=0"
    assert eio.readout_line(4096, 1, 0.999, 0.5).endswith("cos_OK=1")
    assert eio.readout_line(4096, 0, 1.0, 0.0).endswith("crc_ok=0 cos=1.000000 maxabs=0 thr=0.999 cos_OK=0")
    assert eio.COS_THR == 0.999


def test_readout_crc_mismatch_and_raw_ref(tmp_path, capsys):
    ref = np.linspace(-1, 1, 4096, dtype=np.float32)
    ref.tofile(tmp_path / "ref.bin")
    (tmp_path / "board.log").write_text(_dump_log(ref, corrupt=True))
    assert eio.main(["readout", str(tmp_path / "board.log"), "--ref", str(tmp_path / "ref.bin")]) == 1
    m = READOUT_RE.match(capsys.readouterr().out.strip())
    assert m and m.group(2) == "0" and m.group(5) == "0"


def test_readout_without_dump_errors(tmp_path):
    (tmp_path / "board.log").write_text("CMP unet skipped runtime-input aie_launches=37 run_rc=0\n")
    np.save(tmp_path / "ref.npy", np.zeros(4, np.float32))
    assert eio.main(["readout", str(tmp_path / "board.log"), "--ref", str(tmp_path / "ref.npy")]) == 2


def test_runtime_io_main_cmp_skip_format():
    import build_unet as bu
    import unet_aie as ua
    conv = bu.runtime_io_main(ua.main_c(bu.MAIN_C, ktimer=False))
    m = CMP_SKIP_RE.search(conv)
    assert m and m.groups() == ("launches", "run_rc")
    assert '#include "aiehlc_io.h"' in conv and "aiehlc_io_crc()" in conv and "unet_rtio_own" in conv
    assert 'expect=%s %s (timestep=%d) runtime_io=1' in conv and "aiehlc_io_dump(" in conv
    assert "unet_in_ptr[k]" not in conv          # inputs come from the section only
    plain = bu.runtime_io_main(bu.MAIN_C)
    assert CMP_SKIP_RE.search(plain).groups() == ("0", "0")


@pytest.mark.skipif(not shutil.which("gcc"), reason="no host gcc")
def test_host_elf_ab_runs_patched_input(tmp_path):
    """x86 end to end: write_runtime_io C + a tiny main, build once, patch B, both runs see their own bytes
    (the section is read through a laundered pointer, never constant-folded)."""
    a, b = _inputs(0), _inputs(12)
    ab.write_runtime_io(tmp_path, a, dump_out=False, verbose=False)
    (tmp_path / "t.c").write_text(r'''#include <stdio.h>
#include <string.h>
#include "aiehlc_io.h"
int main(void) {
    if (aiehlc_io_check()) return 3;
    int ts; memcpy(&ts, aiehlc_io_ptr(1), 4);
    float o[4]; memcpy(o, aiehlc_io_ptr(0), sizeof o);
    printf("RUN crc=%08x hdr=%08x ts=%d dump=%d\n", (unsigned)aiehlc_io_crc(), (unsigned)aiehlc_io_hdr_crc(), ts,
           aiehlc_io_dump_out());
    if (aiehlc_io_dump_out()) aiehlc_io_dump("t", o, sizeof o, 4);
    return 0;
}
''')
    exe = tmp_path / "t.elf"
    subprocess.run(["gcc", "-O2", "-std=gnu11", "-D__AIESIM__", f"-I{tmp_path}", f"-I{REPO / 'include'}",
                    str(tmp_path / "t.c"), str(tmp_path / "aiehlc_io.c"), "-o", str(exe)], check=True)
    for nm, d in b:
        (tmp_path / f"{nm}.bin").write_bytes(d)
    eb = tmp_path / "tb.elf"
    assert eio.main(["patch", str(exe), "-o", str(eb), "--dump-out",
                     *sum((["--in", f"{nm}={tmp_path / (nm + '.bin')}"] for nm, _ in b), [])]) == 0
    eb.chmod(0o755)
    ra = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout
    rb = subprocess.run([str(eb)], capture_output=True, text=True, check=True).stdout
    ca, cb = (zlib.crc32(b"".join(d for _, d in x)) for x in (a, b))
    assert f"RUN crc={ca:08x} hdr={ca:08x} ts=999 dump=0" in ra
    assert f"RUN crc={cb:08x} hdr={cb:08x} ts=987 dump=1" in rb
    data, crc_ok, elems = eio.parse_dump(rb)
    assert crc_ok == 1 and elems == 4 and data == b[0][1][:16]
