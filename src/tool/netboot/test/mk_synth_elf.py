#!/usr/bin/env python3
"""Build a large synthetic baremetal A78 ELF to exercise netboot streamload.

    python3 src/tool/netboot/test/mk_synth_elf.py --mib 300 --out /tmp/synth300.elf

.data holds <mib> MiB of seeded PRNG bytes (one PT_LOAD with everything else).
At start the program computes crc32 over that array and compares it with the
value compiled in, then prints

    SYNTH bytes=<n> crc=<x> PASS               (match)
    SYNTH bytes=<n> crc=<x> FAIL expect=<y>    (mismatch)

That proves the A78 reads back at the destination address exactly what was
generated, i.e. an ELF larger than NB_STAGING_SIZE (98566144 B) was placed
correctly by the streaming LOAD_RUN path. Link recipe as script/hostcompile.sh
(BSP lscript, --specs=nosys.specs, --defsym end=__bss_end__).
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ARCH = os.path.join(REPO, "thirdparty", "arch", "cortexa78_0")
BSP = os.path.join(ARCH, "bsp")
DEFAULT_VITIS = "/proj/custbuilds/SWIP/2026.1_0616_1700/installs/lin64/2026.1/Vitis"

MAIN_C = r"""
#include <stdint.h>
#include "xil_printf.h"

extern const uint8_t synth_data[];
extern const uint8_t synth_data_end[];
#define SYNTH_EXPECT 0x%(crc)08xU
#define SYNTH_BYTES %(nbytes)dULL

static uint32_t tbl[256];

static void hex8(char *o, uint32_t v)
{
    for (int i = 7; i >= 0; i--, v >>= 4)
        o[i] = "0123456789abcdef"[v & 15];
    o[8] = 0;
}

int main(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++)
            c = (c & 1) ? 0xEDB88320U ^ (c >> 1) : c >> 1;
        tbl[i] = c;
    }
    uint64_t n = (uint64_t)(synth_data_end - synth_data);
    uint32_t c = 0xFFFFFFFFU;
    for (uint64_t i = 0; i < n; i++)
        c = tbl[(c ^ synth_data[i]) & 0xFF] ^ (c >> 8);
    c ^= 0xFFFFFFFFU;
    char got[9], exp[9], dec[24];
    int d = 23;
    dec[d] = 0;
    uint64_t t = n;
    do { dec[--d] = (char)('0' + t %% 10); t /= 10; } while (t);
    hex8(got, c);
    hex8(exp, SYNTH_EXPECT);
    if (c == SYNTH_EXPECT && n == SYNTH_BYTES)
        xil_printf("SYNTH bytes=%%s crc=%%s PASS\r\n", dec + d, got);
    else
        xil_printf("SYNTH bytes=%%s crc=%%s FAIL expect=%%s\r\n", dec + d, got, exp);
    return 0;
}
"""

DATA_S = """
    .section .data.synth,"aw"
    .balign 64
    .global synth_data
synth_data:
    .incbin "%s"
    .global synth_data_end
synth_data_end:
"""


def find_cc(cross):
    cc = shutil.which(cross + "gcc")
    if cc:
        return cc
    vitis = os.environ.get("VITIS", DEFAULT_VITIS)
    cc = os.path.join(vitis, "gnu", "aarch64", "lin", "aarch64-none", "bin", cross + "gcc")
    if os.path.exists(cc):
        return cc
    sys.exit(f"[synth] {cross}gcc not on PATH and not under {vitis} (set VITIS)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mib", type=int, required=True, help="size of the PRNG .data in MiB")
    ap.add_argument("--out", required=True, help="output ELF")
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0x5EED1113)
    ap.add_argument("-v", "--verbose", action="store_true", help="print size/crc summary on stderr")
    ap.add_argument("--keep", help="keep the build dir here instead of a temp dir")
    a = ap.parse_args()

    nbytes = a.mib << 20
    work = a.keep or tempfile.mkdtemp(prefix="synth_elf_")
    os.makedirs(work, exist_ok=True)
    try:
        binp = os.path.join(work, "synth.bin")
        rng, crc = random.Random(a.seed), 0
        with open(binp, "wb") as f:  # 1 MiB at a time (randbytes caps at 2**31 bits)
            for _ in range(a.mib):
                chunk = rng.randbytes(1 << 20)
                crc = zlib.crc32(chunk, crc)
                f.write(chunk)
        crc &= 0xFFFFFFFF
        with open(os.path.join(work, "main.c"), "w") as f:
            f.write(MAIN_C % {"crc": crc, "nbytes": nbytes})
        with open(os.path.join(work, "synth_data.S"), "w") as f:
            f.write(DATA_S % binp)
        cc = find_cc(os.environ.get("CROSS", "aarch64-none-elf-"))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        cmd = [cc, "-mcpu=cortex-a78", "-O2", "-Wall", "-I" + os.path.join(BSP, "include"),
               os.path.join(work, "main.c"), os.path.join(work, "synth_data.S"),
               "--specs=nosys.specs", "-Wl,--defsym,end=__bss_end__",
               "-Wl,--defsym,_HEAP_SIZE=0x100000",
               "-Wl,-T," + os.path.join(ARCH, "lscript.ld"), "-L" + os.path.join(BSP, "lib"),
               "-Wl,--start-group,-lxil,-lgcc,-lc,-lxiltimer,-lxilstandalone,-lxilpm_ng,--end-group",
               "-o", a.out]
        r = subprocess.run(cmd, cwd=work)
        if r.returncode:
            sys.exit(f"[synth] link failed (rc {r.returncode}): {' '.join(cmd)}")
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    if a.verbose:
        print(f"[synth] {a.out}: data {nbytes} B seed 0x{a.seed:x} crc32 {crc:08x} elf {os.path.getsize(a.out)} B",
              file=sys.stderr)


if __name__ == "__main__":
    main()
