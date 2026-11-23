#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# emu_conv2d_tensor.sh <simpleconv2dkernel.cc> <aout/worklocal>: x86-emulate the
# generated tensor-form kernel end to end (see emu_conv2d_tensor.cc).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
SRC="$(cd "$(dirname "${1:?usage: $0 <src.cc> <aout/worklocal>}")" && pwd)/$(basename "$1")"
WL="$(cd "${2:?usage: $0 <src.cc> <aout/worklocal>}" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
"$REPO/build/aiehlc-convtile" trace "$SRC" --emit > "$OUT/emit.cc"
grep -E 'using [A-Za-z_0-9]+ = aie::tensor<' "$OUT/emit.cc" > "$OUT/tensors.inc"
grep '= aiehlc_stage(' "$OUT/emit.cc" | grep -v '^namespace aie' | sed 's/[A-Za-z_0-9]*<<<.*//' > "$OUT/staging.inc"
# Staged extents of a port: the aiehlc_stage arguments, else the tensor's own.
dims() { grep -o "aiehlc_0_$1 = aiehlc_stage([A-Za-z_0-9]*, [0-9]*, [0-9]*, [0-9]*, [0-9]*" "$OUT/staging.inc" |
    sed 's/.*stage([A-Za-z_0-9]*, //' || echo "$2::dim(0), $2::dim(1), $2::dim(2), 1"; }
cp "$HERE/emu_conv2d_tensor.cc" "$WL/conv2d.cc" "$OUT/"
g++ -std=c++20 -O1 -g -w -fsanitize=address -I"$OUT" -I"$HERE/emustubs" -I"$(dirname "$SRC")" -I"$REPO/src/mlir/runtime" \
    -DSTAGED_X="$(dims x_t Ifm)" -DSTAGED_W="$(dims w_t Wgt)" -DSTAGED_Y="$(dims y_t Ofm)" \
    "$OUT/emu_conv2d_tensor.cc" -o "$OUT/emu"
ASAN_OPTIONS=detect_leaks=0 "$OUT/emu"
