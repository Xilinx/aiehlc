#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# run_orange.sh -- text -> 256x256 image with bk-sdm-tiny: the UNet runs on the board (APU + AIE, the ELF
# built by build_unet.py --runtime-io --dump-out), the text encoder and the VAE decoder run on the host.
#
#   usage: run_orange.sh <main.elf> <work dir> [seed] [steps]
#
# Every UNet call (steps x 2 for CFG) goes through aie_imagegen.palmyra_stage_runner: it writes the call's
# inputs into a copy of the ELF's .aiehlc_io section (src/tool/elfio/aiehlc_elfio.py patch), netloads that
# ELF to the board with script/test/apppaltest.py, and reads the output back from the board's AIO dump.
# One STEP line per call goes to <work dir>/steps_s<seed>.log (board crc, graph_run, cos vs onnxruntime).
#
# Environment (defaults are the lab paths):
#   PY        python with numpy, onnxruntime, Pillow       (/scratch/staff/huaj/aiehlc/.venv/bin/python3)
#   ZOO_SRC   bk-sdm-tiny ONNX model zoo                     (/scratch/staff/huaj/aiehlc/model_zoo/bk-sdm-tiny)
#   IMAGEGEN  dir holding the aie_imagegen package (palmyra_stage_runner)
#             (/scratch/staff/huaj/aiehlc/aiehlc-sdxl-wt/palmyra-runner/script)
#   BOARD     palmyra board ip                               (10.23.224.213)
#   PROMPT / NEG / GUIDANCE
#   REF       optional host-generated PNG to compare against (PSNR / cos)
set -eu
[ $# -ge 2 ] || { echo "usage: $0 <main.elf> <work dir> [seed] [steps]"; exit 2; }
ELF=$(readlink -f "$1"); WORK=$(mkdir -p "$2" && cd "$2" && pwd); SEED=${3:-2}; STEPS=${4:-25}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
PY=${PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}
ZOO_SRC=${ZOO_SRC:-/scratch/staff/huaj/aiehlc/model_zoo/bk-sdm-tiny}
IMAGEGEN=${IMAGEGEN:-/scratch/staff/huaj/aiehlc/aiehlc-sdxl-wt/palmyra-runner/script}
BOARD=${BOARD:-10.23.224.213}
PROMPT=${PROMPT:-"a photo of an orange, citrus fruit, high quality"}
NEG=${NEG:-"blurry, deformed, low quality"}
GUIDANCE=${GUIDANCE:-7.5}
[ -f "$ELF" ] || { echo "no ELF: $ELF"; exit 2; }
[ -f "$IMAGEGEN/aie_imagegen/palmyra_stage_runner.py" ] || { echo "no aie_imagegen in $IMAGEGEN"; exit 2; }

# 1. a private model zoo: the ONNX files of bk-sdm-tiny + aie/aie_manifest.json that sends the UNet stage
#    to the board runner (text encoder and VAE stay on the host)
Z=$WORK/zoo/bk-sdm-tiny; mkdir -p "$Z/aie"
for f in "$ZOO_SRC"/*; do [ "$(basename "$f")" = aie ] || ln -sfn "$f" "$Z/"; done
CALLS=$WORK/calls_s$SEED; STEPLOG=$WORK/steps_s$SEED.log; rm -rf "$CALLS" "$STEPLOG"; mkdir -p "$CALLS"
"$PY" - "$Z/aie/aie_manifest.json" <<EOF
import json, sys
cmd = ["$PY", "-m", "aie_imagegen.palmyra_stage_runner", "{in}", "{out}",
       "--elf", "$ELF", "--elfio", "$REPO/src/tool/elfio/aiehlc_elfio.py", "--launcher", "$REPO",
       "--host", "$BOARD", "--calls-dir", "$CALLS", "--steps-log", "$STEPLOG",
       "--map", "timestep=timestep_i32:int32", "--map", "encoder_hidden_states=ehs",
       "--out-shape", "1,4,32,32",
       "--ref-onnx", "$ZOO_SRC/unet.onnx"]
json.dump({"stages": {"unet": {"cmd": cmd, "timeout": 600}}}, open(sys.argv[1], "w"), indent=1)
EOF

# 2. generate: --backend aie, no host fallback (a failed board call fails the image)
OUT=$WORK/orange_seed${SEED}_s${STEPS}.png
cd "$WORK"
echo "[run_orange] seed=$SEED steps=$STEPS -> $OUT  (UNet calls: $((STEPS * 2)), log $STEPLOG)"
PYTHONPATH=$IMAGEGEN "$PY" -m aie_imagegen -p "$PROMPT" --negative-prompt "$NEG" -o "$OUT" \
    --backend aie --zoo "$WORK/zoo" --model bk-sdm-tiny --size 256 --steps "$STEPS" --seed "$SEED" \
    --guidance "$GUIDANCE" --decoder vae > "$WORK/report_s$SEED.json"

# 3. summary: every call must have run on the board with the right input and matched onnxruntime
N=$(grep -c '^STEP ' "$STEPLOG" || true); OK=$(grep -c '^STEP .* cos_OK=1' "$STEPLOG" || true)
GR=$(grep -oE 'graph_run_ms=[0-9.]+' "$STEPLOG" | cut -d= -f2 | awk '{s+=$1} END {if (NR) printf "%.1f", s/NR}')
echo "[run_orange] UNet calls on the board: $N, cos_OK: $OK, mean graph_run_ms: ${GR:-na}"
echo "[run_orange] report: $WORK/report_s$SEED.json   image: $OUT"
if [ -n "${REF:-}" ]; then
    "$PY" -c "import numpy as np,sys;from PIL import Image;a=np.asarray(Image.open(sys.argv[1]).convert('RGB'),np.float64);b=np.asarray(Image.open(sys.argv[2]).convert('RGB'),np.float64);m=((a-b)**2).mean();print('[run_orange] vs %s: PSNR %.2f dB, cos %.5f'%(sys.argv[2],99.0 if m==0 else 10*np.log10(255**2/m),a.ravel()@b.ravel()/np.linalg.norm(a)/np.linalg.norm(b)))" "$OUT" "$REF"
fi
[ "$N" -eq $((STEPS * 2)) ] && [ "$OK" -eq "$N" ]
