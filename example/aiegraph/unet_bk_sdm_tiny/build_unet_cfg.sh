#!/bin/bash
# build_unet_cfg.sh -- the exact UNet (bk-sdm-tiny 256) build used by the cost-model M2 N-fit
# (try5, 2026-11-20: p4a partition, 258 launches, kincore + flash-attn, per-site AHS_GLUE).
#
#   usage: build_unet_cfg.sh <off|aot-kco> <outroot> [--site-perf] [--dry-run]   (options in any order)
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
# Environment: PY  python for build_unet.py (/scratch/staff/huaj/aiehlc/.venv/bin/python3)
set -euo pipefail

usage() { echo "usage: $0 <off|aot-kco> <outroot> [--site-perf] [--dry-run]" >&2; exit 2; }
[ $# -ge 2 ] && [ $# -le 4 ] || usage
cfg=$1
outroot=$2
dry=0
site_perf=0
shift 2
for o in "$@"; do
    case "$o" in
    --dry-run) [ "$dry" = 0 ] || usage; dry=1 ;;
    --site-perf) [ "$site_perf" = 0 ] || usage; site_perf=1 ;;
    *) usage ;;
    esac
done
case "$cfg" in off|aot-kco) ;; *) usage ;; esac

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"
PY=${PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}
PART=partition_p4a.json
CFLAGS=a78_layer_cflags.txt
out=$outroot/unet_$cfg

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
    echo "BUILDCMD cfg=$cfg kincore=$kincore attn_fa=$attn_fa partition=$PART N=$N args_ok=$args_ok site_perf=$site_perf out=$out"
    exit 0
fi

if [ -e "$out" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "$out exists; set FORCE=1 to rebuild over it" >&2
    exit 1
fi
mkdir -p "$outroot"
set +e
"$PY" build_unet.py "${args[@]}" > "$outroot/unet_$cfg.build.log" 2>&1
rc=$?
set -e
echo "$rc" > "$outroot/unet_$cfg.rc"
echo "BUILDCMD cfg=$cfg kincore=$kincore attn_fa=$attn_fa partition=$PART N=$N args_ok=$args_ok site_perf=$site_perf out=$out rc=$rc"
exit "$rc"
