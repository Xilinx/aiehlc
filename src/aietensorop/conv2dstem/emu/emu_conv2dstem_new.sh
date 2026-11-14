#!/usr/bin/env bash
# emu_conv2dstem_new.sh <aout/worklocal of a conv2dstem_uk_new.cc / conv2dstem_new.cc build>:
# x86-emulate the generated kernel with the library's real host code (see the .cc).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEM="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$STEM/../../.." && pwd)"
WL="$(cd "${1:?usage: $0 <aout/worklocal>}" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
# The host part of conv2dstem_new.cc: staging buffers .. collect_ofm_hwc, and the
# scalar reference -- copied verbatim so the emulation tests the library's code.
awk '/^#define IFM_PAD_ELEMS/{on=1} /HOST — library entry/{on=0} on' "$STEM/conv2dstem_new.cc" | sed '$d' > "$OUT/host_part.inc"
awk '/^static void scalar_conv2d_ref/{on=1} on{print} on && /^}/{exit}' "$STEM/conv2dstem_new.cc" >> "$OUT/host_part.inc"
cp "$HERE/emu_conv2dstem_new.cc" "$WL/conv2d_spatial.cc" "$OUT/"
g++ -std=c++20 -O2 -w -I"$OUT" -I"$REPO/example/tileprogram/ccode/simpleconv2ddebug/emustubs" -I"$STEM" \
    -I"$REPO/src/mlir/runtime" "$OUT/emu_conv2dstem_new.cc" -o "$OUT/emu"
"$OUT/emu"
