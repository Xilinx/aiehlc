#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Cost-model segment K microbench, one command (costmodel K1a).
#   run_microbench.sh --board <ip> [--name palmyra213|vek385] [--section aie|ddr|launch|all] [--rebuild] [--out <dir>]
# section aie: builds example/perf/aie_microbench.cc (aiehlc.sh, bare-metal A78 host, AIE2PS), runs it on the board with
# script/test/apppaltest.py --netload (bare-metal xsdb/netboot, the same path as LAUNCH split -> path="baremetal"),
# prints the console (CR stripped; the BOARD marker gets " board=<name>"), fits the raw "MB ..." lines and writes
# src/costmodel/params/<name>.json: aie_clk_mhz pmu_hz ddr_rd_gbps ddr_wr_gbps dma_b_per_cyc dma_first_byte_cyc
# bd_switch_cyc lock_acq_cyc dm_bank_conflict_cyc ss_hop_cyc board path log, then
#   PARAMS <name>.json written (<n> params) section=aie
# section ddr (costmodel K2d): builds example/perf/ddr_microbench.cc, same board path, and MERGES into the existing
# src/costmodel/params/<name>.json (every section=aie key kept): ddr_rd_gbps/ddr_wr_gbps = aggregate over ddr_rd_cols
# columns from the MEDIAN rep (MB ddr bytes= is the TOTAL over reps and cols), ddr_*_cols_points for cols 1..4,
# ddr_*_reps, ddr_burst_points [[bytes, aie cyc per burst]] (32..256 B at 4 KiB stride), ddr_burst_cyc0, then
#   PARAMS <name>.json written (<len(merged json)> params) section=ddr
# section all = aie then ddr.
# section scatter (costmodel SCMB): builds example/perf/scatter_microbench.cc (host uc_scatter16, 24-point grid
# N x yout align x cache), prints the console and the SCMB lines of src/costmodel/tools/scmb_fit.py; writes nothing to params/.
# section launch: stub for the later L sprint (MMIO / ctrlpkt / PM-load); not implemented here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${KACC_PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}"
BOARD="" NAME="" SECTION=aie REBUILD=0 OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --section) SECTION="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --rebuild) REBUILD=1; shift ;;
    *) echo "usage: $0 --board <ip> [--name palmyra213|vek385] [--section aie|ddr|scatter|launch|all] [--rebuild]" >&2; exit 2 ;;
    esac
done
[ -n "$BOARD" ] || { echo "run_microbench: --board <ip> required" >&2; exit 2; }
case "$BOARD" in 10.23.224.213) NAME="${NAME:-palmyra213}"; BKIND=palmyra ;; *) NAME="${NAME:-$BOARD}"; BKIND="${BOARD_KIND:-palmyra}" ;; esac
case "$SECTION" in
aie|ddr|all|scatter) ;;
launch) echo "run_microbench: --section launch is a stub (L sprint: MMIO / ctrlpkt / PM-load), nothing measured" >&2; exit 3 ;;
*) echo "run_microbench: bad --section $SECTION" >&2; exit 2 ;;
esac
[ "$SECTION" = all ] && echo "[mb] section=all: launch part is a stub (L sprint); running aie then ddr"
OUTBASE="${OUT:-/scratch/staff/huaj/tmp/costmodel_mb/$NAME}"
run_section() {
local SEC="$1" SRC OUT
case "$SEC" in
aie) SRC="$ROOT/example/perf/aie_microbench.cc"; OUT="$OUTBASE" ;;
ddr) SRC="$ROOT/example/perf/ddr_microbench.cc"; OUT="$OUTBASE/ddr" ;;
scatter) SRC="$ROOT/example/perf/scatter_microbench.cc"; OUT="$OUTBASE/scatter" ;;
esac
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
# stamp: source + last commit touching the build inputs (unrelated commits on the branch do not force a rebuild)
STAMP="$(md5sum "$SRC" | cut -c1-12) $(git -C "$ROOT" log -1 --format=%h -- script/aiehlc.sh src/mlir/runtime thirdparty/alib 2>/dev/null)"
echo "[mb] board=$NAME ($BOARD) section=$SEC src=$SRC out=$OUT"
if [ "$REBUILD" = 1 ] || [ ! -f "$OUT/main.elf" ] || [ "$(cat "$OUT/stamp" 2>/dev/null)" != "$STAMP" ]; then
    T0=$(date +%s); rm -rf "$OUT/aout" "$OUT/main.elf"
    (
        cd "$OUT"; export AIEHLC_ROOT="$ROOT"; set +eu
        # shellcheck disable=SC1091
        source "$ROOT/script/aiehlc.sh" --aie-version 5 --runtime-source-file "$SRC"
        rc=$?; set -u
        [ $rc -eq 0 ] && [ -f "$OUT/aout/main.elf" ] || { echo "[mb] aiehlc.sh failed rc=$rc"; exit 1; }
    ) > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; echo "[mb] BUILD FAILED (log $OUT/build.log)"; return 1; }
    cp -f "$OUT/aout/main.elf" "$OUT/main.elf"; echo "$STAMP" > "$OUT/stamp"
    echo "[mb] BUILD_OK $OUT/main.elf in $(( $(date +%s) - T0 )) s md5=$(md5sum "$OUT/main.elf" | cut -c1-12)"
else
    echo "[mb] reuse $OUT/main.elf (stamp $STAMP)"
fi
local RUN="$OUT/run_$(date +%Y%m%d_%H%M%S).log" RRC
set +e
(VITIS_ENV_QUIET=1 source "${VITIS_ENV:-/scratch/staff/huaj/aiengineers/skills/vitis-2026.2/vitis_env.sh}" >/dev/null 2>&1 || true
 cd "$ROOT/script/test" && timeout 1500 "$PY" ./apppaltest.py -board "$BKIND" -host "$BOARD" -y --netload \
    --netload-wait 1200 "$OUT/main.elf") 2>&1 | tr -d '\r' | sed -E "s/^(BOARD machine=\S+ midr=0x[0-9a-f]+)\$/\1 board=$NAME/" > "$RUN"
RRC=${PIPESTATUS[0]}
set -e
cat "$RUN"
echo "[mb] apppaltest rc=$RRC log=$RUN"
[ "$RRC" -eq 0 ] || { echo "[mb] apppaltest rc=$RRC"; return "$RRC"; }
if [ "$SEC" = scatter ]; then
    "$PY" -I "$ROOT/src/costmodel/tools/scmb_fit.py" "$RUN"   # SCMB: prints only, nothing written to params/
else
    "$PY" -I "$ROOT/src/costmodel/tools/mb_fit.py" "$RUN" "$ROOT/src/costmodel/params/$NAME.json" "$NAME" "$SEC"
fi
}
case "$SECTION" in
all) run_section aie && run_section ddr ;;
*) run_section "$SECTION" ;;
esac
