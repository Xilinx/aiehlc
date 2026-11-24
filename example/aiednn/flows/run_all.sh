#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Run the four aiednn entry flows on workstation emulation; each prints "PASS: flow N ..." (README.md).
#   bash example/aiednn/flows/run_all.sh            (from the repo root; PYTHON= picks the interpreter)
set -u
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PYTHON:-python3}
cd "$REPO"
out=$(PYTHONPATH=src "$PY" -m aiegraph dnn-build example/aiednn/flows/1_c_api/matmul_relu_app.cc --target x86 --run \
    2>&1 | grep -E "^ *(PASS|FAIL)" | sed 's/^ *//')
for f in 2_torch_compile 3_onnx 4_model_compiler; do
    out+=$'\n'$("$PY" "example/aiednn/flows/$f/run.py" 2>&1 | grep -E "^(PASS|FAIL)")
done
echo "$out"
[ "$(grep -c '^PASS: flow' <<< "$out")" = 4 ] && ! grep -q FAIL <<< "$out"
