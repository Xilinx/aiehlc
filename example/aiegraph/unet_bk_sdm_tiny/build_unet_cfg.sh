#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# build_unet_cfg.sh -- the exact UNet (bk-sdm-tiny 256) build used by the cost-model M2 N-fit
# (try5, 2026-11-20: p4a partition, 258 launches, kincore + flash-attn, per-site AHS_GLUE).
#
#   usage: build_unet_cfg.sh <off|aot-kco> <outroot> [--site-perf] [--dry-run|--check-fresh]   (options in any order)
#
# Real build: writes <outroot>/unet_<cfg>/ (main.elf, aiek_src/...), <outroot>/unet_<cfg>.build.log
# and <outroot>/unet_<cfg>.rc. It refuses to overwrite an existing <outroot>/unet_<cfg> unless FORCE=1.
# --dry-run: checks only, writes nothing. It checks that partition_p4a.json and a78_layer_cflags.txt
# exist, that the argument list holds --kincore and --attn-fa, reads N from the partition file and
# prints one BUILDCMD line. --site-perf (M3.1, diagnostic build only -- the counter setup adds ~128 MMIO
# writes per launch, so do not measure L with it): passes --site-perf to build_unet.py
# (-DAIEHLC_SITE_PERF=1: one PROF aie_site_perf counter line per launches>0 part). build.log always labels the partition "partition.json"
# (unet_offload.py:172), so N=216200 (p4a; partition.json has N=86600) is what identifies p4a.
#
# Build inputs (M3.1.2 PF): inputs = `git log -1 --format=%h -- $INPUT_PATHS` and dirty = number of
# `git status --porcelain` lines under the same paths (script/perf is not a build input). Both are appended
# to every BUILDCMD line. A real build refuses (rc=3) when dirty != 0, and before building writes
# <outroot>/unet_<cfg>.src: "BUILDSRC inputs=<h> start=<epoch> site_perf=<0|1> kincore=<0|1>".
# --check-fresh: writes nothing; prints
#   BUILD FRESH cfg=<c> inputs=<.src> head_inputs=<now> dirty=<n> elf_newer=<0|1> site_perf=<n> kincore=<n> PASS|FAIL
# PASS iff inputs==head_inputs, dirty=0, main.elf mtime >= start, site_perf=1 and kincore=1; FAIL -> rc=1.
#
# Environment: PY  python for build_unet.py (/scratch/staff/huaj/aiehlc/.venv/bin/python3)
set -euo pipefail

usage() { echo "usage: $0 <off|aot-kco> <outroot> [--site-perf] [--dry-run|--check-fresh]" >&2; exit 2; }
[ $# -ge 2 ] && [ $# -le 4 ] || usage
cfg=$1
outroot=$2
dry=0
site_perf=0
check=0
shift 2
for o in "$@"; do
    case "$o" in
    --dry-run) [ "$dry" = 0 ] || usage; dry=1 ;;
    --check-fresh) [ "$check" = 0 ] || usage; check=1 ;;
    --site-perf) [ "$site_perf" = 0 ] || usage; site_perf=1 ;;
    *) usage ;;
    esac
done
case "$cfg" in off|aot-kco) ;; *) usage ;; esac
[ "$dry" = 1 ] && [ "$check" = 1 ] && usage

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
PY=${PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}
PART=partition_p4a.json
CFLAGS=a78_layer_cflags.txt
out=$outroot/unet_$cfg
src=$outroot/unet_$cfg.src

# build inputs, relative to the repository top (script/perf, src/aiegraph/costmodel -- the pre-move home of src/costmodel, kept so its old commits stay excluded -- and src/aiegraph/test deliberately excluded)
INPUT_PATHS=(src/mlir/runtime src/aiegraph src/aiednn example/aiegraph/unet_bk_sdm_tiny :!src/aiegraph/costmodel :!src/aiegraph/test)
top=$(git -C "$here" rev-parse --show-toplevel)
head_inputs=$(git -C "$top" log -1 --format=%h -- "${INPUT_PATHS[@]}")
[ -n "$head_inputs" ] || head_inputs=none
dirty=$(git -C "$top" status --porcelain -- "${INPUT_PATHS[@]}" | wc -l)

if [ "$check" = 1 ]; then
    rin=none; rstart=; rsp=0; rkc=0
    if [ -s "$src" ]; then
        line=$(grep -m1 '^BUILDSRC ' "$src" || true)
        for kv in $line; do
            case "$kv" in
            inputs=*) rin=${kv#inputs=} ;;
            start=*) rstart=${kv#start=} ;;
            site_perf=*) rsp=${kv#site_perf=} ;;
            kincore=*) rkc=${kv#kincore=} ;;
            esac
        done
    fi
    elf_newer=0
    if [ -f "$out/main.elf" ] && [[ "$rstart" =~ ^[0-9]+$ ]]; then
        [ "$(stat -c %Y "$out/main.elf")" -ge "$rstart" ] && elf_newer=1
    fi
    verdict=FAIL
    [ "$rin" != none ] && [ "$rin" = "$head_inputs" ] && [ "$dirty" = 0 ] && [ "$elf_newer" = 1 ] \
        && [ "$rsp" = 1 ] && [ "$rkc" = 1 ] && verdict=PASS
    echo "BUILD FRESH cfg=$cfg inputs=$rin head_inputs=$head_inputs dirty=$dirty elf_newer=$elf_newer site_perf=$rsp kincore=$rkc $verdict"
    [ "$verdict" = PASS ] && exit 0
    exit 1
fi

# argument list verbatim from try5 (/proc/1392294/cmdline, read 00:49)
args=(--onnx /scratch/staff/huaj/modelzoo/artifacts/bk-sdm-tiny/unet_sim.onnx
      --real-io /scratch/staff/huaj/tmp/unet_tutorial/real_io
      --wdtype bf16 --aie conv,linear,attn --attn-fa --keep-layout --kincore
      --runtime-io --dump-out --arm-layer-cflags-file "$CFLAGS"
      --partition "$PART" --launch-opt warm --ctrlpkt "$cfg" --out "$out")
[ "$site_perf" = 1 ] && args+=(--site-perf)

has() { local a; for a in "${args[@]}"; do [ "$a" = "$1" ] && return 0; done; return 1; }

[ -s "$PART" ] || { echo "missing $here/$PART" >&2; exit 1; }
[ -s "$CFLAGS" ] || { echo "missing $here/$CFLAGS" >&2; exit 1; }
kincore=0; has --kincore && kincore=1
attn_fa=0; has --attn-fa && attn_fa=1
N=$(python3 -I -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["N"]))' "$PART")
args_ok=0
[ "$kincore" = 1 ] && [ "$attn_fa" = 1 ] && [ -n "$N" ] && args_ok=1
[ "$args_ok" = 1 ] || { echo "argument check failed: kincore=$kincore attn_fa=$attn_fa N=$N" >&2; exit 1; }

if [ "$dry" = 1 ]; then
    echo "CMD $PY build_unet.py ${args[*]}"
    echo "BUILDCMD cfg=$cfg kincore=$kincore attn_fa=$attn_fa partition=$PART N=$N args_ok=$args_ok site_perf=$site_perf out=$out inputs=$head_inputs dirty=$dirty"
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
echo "BUILDSRC inputs=$head_inputs start=$(date +%s) site_perf=$site_perf kincore=$kincore" > "$src"
set +e
"$PY" build_unet.py "${args[@]}" > "$outroot/unet_$cfg.build.log" 2>&1
rc=$?
set -e
echo "$rc" > "$outroot/unet_$cfg.rc"
echo "BUILDCMD cfg=$cfg kincore=$kincore attn_fa=$attn_fa partition=$PART N=$N args_ok=$args_ok site_perf=$site_perf out=$out inputs=$head_inputs dirty=$dirty rc=$rc"
exit "$rc"
