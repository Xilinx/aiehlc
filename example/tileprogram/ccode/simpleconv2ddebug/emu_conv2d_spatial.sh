#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# emu_conv2d_spatial.sh <aout/worklocal>: x86-emulate the generated
# simpleconv2d_new_load.cc / simpleconv2dkernel_port.cc kernel (scalar or vector) against
# the CPU reference (see emu_conv2d_spatial.cc).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
WL="$(cd "${1:?usage: $0 <aout/worklocal>}" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
cp "$HERE/emu_conv2d_spatial.cc" "$WL/conv2d_spatial.cc" "$OUT/"
g++ -std=c++20 -O2 -w -I"$OUT" -I"$HERE/emustubs" -I"$HERE/.." -I"$REPO/src/mlir/runtime" \
    "$OUT/emu_conv2d_spatial.cc" -o "$OUT/emu"
"$OUT/emu"
