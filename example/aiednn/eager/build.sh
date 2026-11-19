#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build one eager-mode template board app (two instances of a template, a self-check, PASS: / FAIL:) for VEK280:
#   example/aiednn/eager/build.sh pool_eager.cc        -> worklocal/eager/pool_eager/aout/main.elf
# aiehlc resolves a spliced .cc include against the entry's own directory (lint L020), so the app and the template
# sources of src/aietensorop/eager/ are staged together under worklocal/ (never copied into this directory).
# Run the ELF on the board with vek280_mlir_aie/run_vek280.py (stage it under /proj first: the board host sees /proj).
set -euo pipefail
app=${1:?usage: build.sh <app>_eager.cc}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../.." && pwd)
name=$(basename "$app" .cc)
work="$repo/worklocal/eager/$name"
mkdir -p "$work/src"
cp "$here/$(basename "$app")" "$repo"/src/aietensorop/eager/*.cc "$repo"/src/aietensorop/eager/*.h "$work/src/"
[ -f "$here/gemm_s8_eager.h" ] && cp "$here/gemm_s8_eager.h" "$work/src/"
cd "$work"
rm -rf aout
bash -c "source '$repo/script/aiehlc.sh' --aie-version 2 --runtime-source-file '$work/src/$(basename "$app")'" > build.log 2>&1
test -f aout/main.elf && echo "built $work/aout/main.elf" || { echo "build failed: $work/build.log"; exit 1; }
