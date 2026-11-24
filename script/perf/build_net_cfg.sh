#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# build_net_cfg.sh -- the ResNet18 / yolo26n board builds for the cost-model M4 three-net K+L+N check
# (structure after example/aiegraph/unet_bk_sdm_tiny/build_unet_cfg.sh).
#
#   usage: build_net_cfg.sh <resnet18|yolo26n> <cfg> <outroot> [--site-perf] [--dry-run|--check-fresh]
#          cfg: resnet18 -> off|aot-kco ; yolo26n -> off|aot          (options in any order)
#
# Usage errors (argument count, unknown net, illegal (net,cfg), unknown/duplicate option, --dry-run with
# --check-fresh) print usage and exit 2 before any git / python / mkdir.
#
# Build inputs: inputs = `git log -1 --format=%h -- INPUT_PATHS`, dirty = `git status --porcelain` lines
# under INPUT_PATHS = src/mlir/runtime src/aiegraph src/aiednn example/aiegraph/<dir>, minus src/aiegraph/costmodel (the pre-move home of src/costmodel) and
# src/aiegraph/test (since the unet-decl refactor the kernel / host assembly lives in src/aiegraph)
# (dir = resnet18 | yolov26; script/perf is not a build input).
#
# --dry-run: writes nothing, prints one line
#   BUILDCMD net=<n> cfg=<c> site_perf=<0|1> inputs=<h> dirty=<d> paths=<p1,p2,..> cmd=<full command>
# real build: rc=3 when dirty != 0; refuses an existing <outroot>/<net>_<cfg> unless FORCE=1; writes
#   <outroot>/<net>_<cfg>.src  "BUILDSRC net= cfg= inputs= head= start= site_perf="
#   <outroot>/<net>_<cfg>.build.log (driver output via tee) + a last line
#   NETBUILD net=<n> cfg=<c> planned=<N> secs=<s> rc=<rc>
#   exit rc (the driver's own, PIPESTATUS[0]); rc=0 with planned=0 -> exit 5.
# --check-fresh: writes nothing; prints
#   BUILD FRESH net=<n> cfg=<c> inputs=<i> head_inputs=<h> dirty=<d> elf_newer=<0|1> site_perf=<0|1> PASS|FAIL
#   PASS iff inputs==head_inputs, dirty=0, main.elf mtime >= start and (with --site-perf) site_perf=1; FAIL -> rc 1.
#
# Environment: PY (default /scratch/staff/huaj/aiehlc/.venv/bin/python3), FORCE=1.

usage() {
    echo "usage: $0 <resnet18|yolo26n> <cfg> <outroot> [--site-perf] [--dry-run|--check-fresh]" >&2
    echo "       cfg: resnet18 -> off|aot-kco ; yolo26n -> off|aot" >&2
    exit 2
}
# ---- usage check: bash only, before any git / python / mkdir ----
[ $# -ge 3 ] && [ $# -le 5 ] || usage
net=$1; cfg=$2; outroot=$3
case "$net:$cfg" in
resnet18:off|resnet18:aot-kco) dir=resnet18 ;;
yolo26n:off|yolo26n:aot) dir=yolov26 ;;
*) usage ;;
esac
case "$outroot" in ''|-*) usage ;; esac
dry=0; check=0; site_perf=0
shift 3
for o in "$@"; do
    case "$o" in
    --dry-run) [ "$dry" = 0 ] || usage; dry=1 ;;
    --check-fresh) [ "$check" = 0 ] || usage; check=1 ;;
    --site-perf) [ "$site_perf" = 0 ] || usage; site_perf=1 ;;
    *) usage ;;
    esac
done
[ "$dry" = 1 ] && [ "$check" = 1 ] && usage

set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
top=$(git -C "$here" rev-parse --show-toplevel)
PY=${PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}
case "$net" in
resnet18) ONNX=/scratch/staff/huaj/modelzoo/artifacts/resnet18/onnx/resnet18_fp32_op17.onnx ;;
yolo26n) ONNX=/scratch/staff/huaj/modelzoo/artifacts/yolo26/yolo26n_e2e_trained_op17.onnx ;;
esac
out=$outroot/${net}_$cfg
src=$outroot/${net}_$cfg.src
blog=$outroot/${net}_$cfg.build.log

INPUT_PATHS=(src/mlir/runtime src/aiegraph src/aiednn "example/aiegraph/$dir" :!src/aiegraph/costmodel :!src/aiegraph/test)
head_inputs=$(git -C "$top" log -1 --format=%h -- "${INPUT_PATHS[@]}")
[ -n "$head_inputs" ] || head_inputs=none
dirty=$(git --no-optional-locks -C "$top" status --porcelain -- "${INPUT_PATHS[@]}" | wc -l)
paths=$(IFS=,; echo "${INPUT_PATHS[*]}")

if [ "$check" = 1 ]; then
    rin=none; rstart=; rsp=0
    if [ -s "$src" ]; then
        line=$(grep -m1 '^BUILDSRC ' "$src" || true)
        for kv in $line; do
            case "$kv" in
            inputs=*) rin=${kv#inputs=} ;;
            start=*) rstart=${kv#start=} ;;
            site_perf=*) rsp=${kv#site_perf=} ;;
            esac
        done
    fi
    elf_newer=0
    if [ -f "$out/main.elf" ] && [[ "$rstart" =~ ^[0-9]+$ ]]; then
        [ "$(stat -c %Y "$out/main.elf")" -ge "$rstart" ] && elf_newer=1
    fi
    verdict=FAIL
    [ "$rin" != none ] && [ "$rin" = "$head_inputs" ] && [ "$dirty" = 0 ] && [ "$elf_newer" = 1 ] \
        && { [ "$site_perf" = 0 ] || [ "$rsp" = 1 ]; } && verdict=PASS
    echo "BUILD FRESH net=$net cfg=$cfg inputs=$rin head_inputs=$head_inputs dirty=$dirty elf_newer=$elf_newer site_perf=$rsp $verdict"
    [ "$verdict" = PASS ] && exit 0
    exit 1
fi

drv=$top/example/aiegraph/$dir/build_$dir.py
args=(--onnx "$ONNX" --out "$out" --ctrlpkt "$cfg")
[ "$net" = resnet18 ] && args+=(--launch-opt warm)
[ "$site_perf" = 1 ] && args+=(--site-perf)
[ -s "$ONNX" ] || { echo "missing $ONNX" >&2; exit 1; }
[ -f "$drv" ] || { echo "missing $drv" >&2; exit 1; }

if [ "$dry" = 1 ]; then
    echo "BUILDCMD net=$net cfg=$cfg site_perf=$site_perf inputs=$head_inputs dirty=$dirty paths=$paths cmd=$PY $drv ${args[*]}"
    exit 0
fi

if [ "$dirty" != 0 ]; then
    echo "build inputs dirty ($dirty porcelain lines under ${INPUT_PATHS[*]}); commit first" >&2
    git -C "$top" status --porcelain -- "${INPUT_PATHS[@]}" >&2
    exit 3
fi
if [ -e "$out" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "$out exists; set FORCE=1 to rebuild over it" >&2
    exit 1
fi
mkdir -p "$outroot"
start=$(date +%s)
echo "BUILDSRC net=$net cfg=$cfg inputs=$head_inputs head=$(git -C "$top" rev-parse --short HEAD) start=$start site_perf=$site_perf" > "$src"
echo "CMD $PY $drv ${args[*]}" > "$blog"
set +e +o pipefail
"$PY" "$drv" "${args[@]}" 2>&1 | tee -a "$blog"
rc=${PIPESTATUS[0]}
set -e
secs=$(( $(date +%s) - start ))
if [ "$net" = resnet18 ]; then
    planned=$(grep -oE 'aie_launches_planned=[0-9]+' "$blog" | tail -1 | cut -d= -f2)
else
    planned=$(grep -oE '[0-9]+ launches planned' "$blog" | tail -1 | cut -d' ' -f1)
fi
[ -n "$planned" ] || planned=0
echo "NETBUILD net=$net cfg=$cfg planned=$planned secs=$secs rc=$rc" | tee -a "$blog"
[ "$rc" = 0 ] && [ "$planned" = 0 ] && exit 5
exit "$rc"
