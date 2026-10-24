#!/usr/bin/env bash
# Build one resnetengine app (e0, ...) into a scratch worklocal and link a host ELF
# with hostcompile.sh. Usage: build.sh <app dir under resnetengine> [out dir]
set -u
usage() {
    cat <<'EOF'
Usage: [EXTRA_DEFS="..."] [KDEFS="..."] build.sh <app> [out dir]

  <app>      app dir under src/aietensorop/resnetengine (the engine: eng)
  [out dir]  default /tmp/claude/re_<app>; the host ELF lands in
             <out dir>/aout/worklocal/build/host (the dir is wiped first)

Run from the repo root, after
  python3 src/aietensorop/resnetengine/gen_engine.py --out worklocal/resnetengine

EXTRA_DEFS  -D flags for the host/runtime compile. For eng:
  -DENG_MODES_DEFAULT=<mask>  which runtime modes main() runs (default 1):
        bit value  mode
         0    1    persistent: one kernel ELF loaded once, image-major
         1    2    eager: kernel reloaded before every layer, image-major
         2    4    eager layer-major, 1 image per layer
         3    8    eager layer-major, 4 images per layer
         4   16    eager layer-major, 8 images per layer
         5   32    eager layer-major, 16 images per layer
         6   64    eager layer-major, 32 images per layer
        add values to combine: 3 = persistent + eager, 127 = all.
        Layer-major modes (4..64) can hang intermittently (skill shimctrlpushloss).
  -DENG_NRUNS=<n>             timed runs per layer-major mode (default 3)
  debug: -DENG_TIMELINE (one unit's channel timeline) -DENG_CHECK_EVERY_BATCH
         -DRT_SB_NO_KEEP_ARMED (rewrite the shim-row BD chain every commit)
KDEFS       space-separated macros #define'd at the top of kernel.cc
PROFILING=1 build the runtime with AIEHLC_PROFILING (PMU phase counters)

Examples:
  bash src/aietensorop/resnetengine/build.sh eng /tmp/claude/re_eng                  # persistent
  EXTRA_DEFS="-DENG_MODES_DEFAULT=2" bash src/aietensorop/resnetengine/build.sh eng  # eager
  EXTRA_DEFS="-DENG_MODES_DEFAULT=3" bash src/aietensorop/resnetengine/build.sh eng  # both
Then: python3 script/test/apppaltest.py -y --netload <out dir>/aout/worklocal/build/host
EOF
}
case "${1:-}" in -h | --help | "") usage; exit $([ -n "${1:-}" ] && echo 0 || echo 1) ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
APP="$1"; OUT="${2:-/tmp/claude/re_$APP}"
[ -d "$HERE/$APP" ] || { echo "build.sh: no app dir $HERE/$APP (see --help)"; exit 1; }
SRC="$HERE/$APP"
rm -rf "$OUT"; mkdir -p "$OUT/aout/worklocal"
W="$OUT/aout/worklocal"
cp "$SRC"/*.cc "$SRC"/*.h "$SRC"/aieml.bcf "$W"/ 2>/dev/null
cp "$HERE"/*.h "$W"/
# Generated inputs (e.g. layeriohex dumps) the app includes: paths relative to the repo.
if [ -f "$SRC/extra_inputs.txt" ]; then
    while read -r f; do [ -n "$f" ] && cp "$REPO/$f" "$W"/ || exit 1; done < "$SRC/extra_inputs.txt"
fi
cp "$HERE/fabric/routing.cc" "$HERE/fabric/aieml.prx" "$W"/
# Experiment knobs: KDEFS="A B" prepends #define A / #define B to kernel.cc.
for d in ${KDEFS:-}; do sed -i "1i #define $d" "$W/kernel.cc"; done
# Kernel variants: each line "<name> [DEFINE ...]" of <app>/kernel_variants becomes
# kernel_<name>.cc (those defines + kernel.cc); hostcompile.sh's multi-kernel mode
# then embeds one ELF per variant as _binary_kernel_<name>_start.
if [ -f "$SRC/kernel_variants" ]; then
    while read -r name defs; do
        [ -z "$name" ] && continue
        { for d in $defs; do echo "#define $d"; done; cat "$W/kernel.cc"; } > "$W/kernel_$name.cc"
    done < "$SRC/kernel_variants"
    rm -f "$W/kernel.cc"
fi
source "$REPO/script/setup.sh" >/dev/null 2>&1
# hostcompile links thirdparty/alib/lib/libxaienginea78.a, which aiehlc.sh
# (re)stages at the start of every run and wipes per run -- stage it if absent.
if [ ! -f "$REPO/thirdparty/alib/lib/libxaienginea78.a" ]; then
    (cd "$REPO" && source script/aiehlc.sh --aie-version 5 --aielib-only) > "$OUT/aielib.log" 2>&1 \
        || { echo "aielib staging failed (log $OUT/aielib.log)"; exit 1; }
fi
cd "$OUT"
WORKLOCAL_DIR="$W" AIE_VERSION=5 PLATFORM=baremetal AIEHLC_PROFILING="${PROFILING:-0}" \
    bash "$REPO/script/hostcompile.sh" > "$OUT/build.log" 2>&1
rc=$?
if [ -f "$W/build/host" ]; then echo "built $W/build/host"; else echo "BUILD FAILED rc=$rc (log $OUT/build.log)"; grep -n 'error' "$OUT/build.log" | head; exit 1; fi
# AIE2PS program memory is 16 KB and the chess linker does not enforce it (its
# PM model is 1 MB); an overflowing ELF only fails later, at load. Check here.
for o in "$W"/build/kernel*.o; do
    [ -f "$o" ] || continue
    llvm-objcopy -O binary "$o" "$OUT/$(basename "$o" .o).elf" 2>/dev/null || { echo "  (cannot extract $o)"; continue; }
    pm=$(readelf -lW "$OUT/$(basename "$o" .o).elf" | awk '$1=="LOAD" && ($7 ~ /E/ || $8 ~ /E/) {e=strtonum($3)+strtonum($5); if (e>m) m=e} END {print m+0}')
    echo "  $(basename "$o" .o): $pm B of program memory (16384 max)"
    [ "$pm" -le 16384 ] || { echo "BUILD FAILED: $(basename "$o" .o) overflows program memory"; exit 1; }
done
