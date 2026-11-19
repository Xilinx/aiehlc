#!/usr/bin/env bash
# Cost-model P0 / K1a: one gemm_kacc kernel from kernels.tsv on the board, with per-launch AIE core perf counters.
#   p0_one_kernel.sh --board <ip> [--kernel <name>] [--kernels <kernels.tsv>] [--name palmyra213] [--out <dir>]
#                    [--reps 8] [--aie-clk-mhz 1000] [--rebuild]
# Builds (cached by a stamp of shape + host + commit) the gemm_kacc instance of the kernel's GEMM shape (M,K,N) through
# example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py with the P0 host script/perf/p0_kacc_host.cc and an aiehlc.sh
# --profiling host (core perf counters ACTIVE/VEC/STREAM_STALL/LOCK_STALL armed at every launch), runs it with
# script/test/apppaltest.py (bare-metal, netload) and prints the console (CR stripped) plus one line:
#   P0 board=<name> machine=aarch64 kernel=<k> launches=<n> rt_wait_io_ms=<split_detail rt_wait_io, %.3f>
#      core_src=perfctr active_cyc=<a> compute_cyc=<c> pred_ms=<c/aie_clk> meas_ms=<= rt_wait_io_ms> rel=<|p-m|/m>
# active_cyc / compute_cyc: per timed launch, the slowest core tile (max active_cyc over the [PERF] tiles of that tag),
# compute = active - lock_stall - stream_stall of that tile; the P0 line carries the mean over the launches.
# Inputs are synthetic (seeded N(0,1) A[M,K], B[K,N]; y = A@B fp32): the shape is the kernel's, the data is not.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${KACC_PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}"
BOARD="" KERNEL="u_lin_1024x320x320" NAME="" OUT="" REPS=8 CLK=1000 REBUILD=0
KTSV=/scratch/staff/huaj/aiengineers/arthur/tasks/2026-11-19_costmodel/kernels.tsv
while [ $# -gt 0 ]; do
    case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --kernel) KERNEL="$2"; shift 2 ;;
    --kernels) KTSV="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --aie-clk-mhz) CLK="$2"; shift 2 ;;
    --rebuild) REBUILD=1; shift ;;
    *) echo "usage: $0 --board <ip> [--kernel <name>] [--kernels <tsv>] [--name <board>] [--out <dir>] [--reps N] [--rebuild]" >&2; exit 2 ;;
    esac
done
[ -n "$BOARD" ] || { echo "p0_one_kernel: --board <ip> required" >&2; exit 2; }
case "$BOARD" in 10.23.224.213) NAME="${NAME:-palmyra213}"; BKIND=palmyra ;; *) NAME="${NAME:-$BOARD}"; BKIND="${BOARD_KIND:-palmyra}" ;; esac
read -r M K N RECIPE < <(awk -F'\t' -v k="$KERNEL" 'NR>1 && $1==k {print $3, $4, $5, $2}' "$KTSV")
[ -n "${M:-}" ] || { echo "p0_one_kernel: kernel $KERNEL not in $KTSV" >&2; exit 2; }
if [ "$RECIPE" = kacc_chunk_conv_bf16 ]; then
    # yolo chunked conv: one launch = the gemm_kacc instance of src/aiegraph/cnn_chunk.plan_conv (split 1):
    # ks = max(c64(K), 192), Np = max(c64(N), 128 if ks > 1152 else 64), px chunk MC = 256 (x2 / /2 when Np == MC);
    # the layer is ceil(M / mc) launches of (M=mc, K=ks, N=Np). Board-time per launch is what the cost model fits.
    c64() { echo $(( ($1 + 63) / 64 * 64 )); }
    KS=$(c64 "$K"); [ "$KS" -ge 192 ] || KS=192
    NP=$(c64 "$N"); if [ "$KS" -gt 1152 ]; then [ "$NP" -ge 128 ] || NP=128; fi
    MC=256; if [ "$NP" -eq "$MC" ]; then MC=512; fi
    echo "[p0] $RECIPE layer M=$M K=$K N=$N -> per-launch gemm_kacc instance M=$MC K=$KS N=$NP (launches_per_layer=$(( (M + MC - 1) / MC )), cnn_chunk.plan_conv)"
    M=$MC K=$KS N=$NP
fi
OUT="${OUT:-/scratch/staff/huaj/tmp/costmodel_p0/$KERNEL}"
mkdir -p "$OUT/golden"
OUT="$(cd "$OUT" && pwd)"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
STAMP="$M $K $N reps=$REPS $(cat "$HERE/p0_kacc_host.cc" "$ROOT/example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py" | md5sum | cut -c1-12) $COMMIT"
echo "[p0] kernel=$KERNEL recipe=$RECIPE M=$M K=$K N=$N board=$NAME ($BOARD) out=$OUT commit=$COMMIT"

if [ "$REBUILD" = 1 ] || [ ! -f "$OUT/main.elf" ] || [ "$(cat "$OUT/stamp" 2>/dev/null)" != "$STAMP" ]; then
    T0=$(date +%s)
    # synthetic MatMul golden in the op_golden layout kacc_gemm.py reads (a [1,M,K], b [K,N], y [1,M,N], meta.json)
    "$PY" -I - "$OUT/golden/c000_matmul_p0" "$M" "$K" "$N" <<'EOF'
import sys, json, hashlib, os, numpy as np
d, M, K, N = sys.argv[1], *map(int, sys.argv[2:5])
os.makedirs(d, exist_ok=True)
r = np.random.default_rng(0)
a = r.standard_normal((1, M, K)).astype(np.float32)
b = (r.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
y = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
np.save(f"{d}/a.npy", a); np.save(f"{d}/b.npy", b); np.save(f"{d}/y.npy", y)
json.dump({"op": "MatMul", "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(), "synthetic": "p0_one_kernel seed 0"},
          open(f"{d}/meta.json", "w"))
EOF
    rm -rf "$OUT/aout" "$OUT/main.elf"
    KACC_HOST_FILE="$HERE/p0_kacc_host.cc" "$PY" -I "$ROOT/example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py" \
        "$OUT/golden/c000_matmul_p0" "$OUT"
    sed -i "1i #define KACC_REPS $REPS" "$OUT/kacc_gemm.cc"
    (
        cd "$OUT"
        export AIEHLC_ROOT="$ROOT"
        # same launch options as kacc_bench.sh --gemm (build_unet.py --launch-opt warm), plus --profiling
        export EXTRA_DEFS="${EXTRA_DEFS:-} -DAIEHLC_ZFILL_WIDE=1 -DAIEHLC_PMLOAD_WIDE=1 -DAIEHLC_PM_WARM_SKIP=1 -DAIEHLC_SKIP_BSS_DEFAULT=1"
        set +eu
        # shellcheck disable=SC1091
        source "$ROOT/script/aiehlc.sh" --aie-version 5 --skip-bss --profiling --runtime-source-file "$OUT/kacc_gemm.cc"
        rc=$?
        set -u
        [ $rc -eq 0 ] && [ -f "$OUT/aout/main.elf" ] || { echo "[p0] aiehlc.sh failed rc=$rc"; exit 1; }
    ) > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; echo "[p0] BUILD FAILED (log $OUT/build.log)"; exit 1; }
    grep -q -- "-DAIEHLC_PROFILING=1" "$OUT/build.log" || echo "[p0] note: -DAIEHLC_PROFILING=1 not echoed in build.log"
    cp -f "$OUT/aout/main.elf" "$OUT/main.elf"
    echo "$STAMP" > "$OUT/stamp"
    echo "[p0] BUILD_OK $OUT/main.elf in $(( $(date +%s) - T0 )) s"
else
    echo "[p0] reuse $OUT/main.elf (stamp $STAMP)"
fi

LOCK=/scratch/staff/huaj/tmp/board_locks/${BOARD}_3121.lock
if [ -e "$LOCK" ] && ! flock -n "$LOCK" true; then echo "[p0] board lock $LOCK held: apppaltest waits for it"; fi
RUN="$OUT/run.log"
set +e
(VITIS_ENV_QUIET=1 source "${VITIS_ENV:-/scratch/staff/huaj/aiengineers/skills/vitis-2026.2/vitis_env.sh}" >/dev/null 2>&1 || true
 cd "$ROOT/script/test" && timeout 1500 "$PY" ./apppaltest.py -board "$BKIND" -host "$BOARD" -y --netload \
    --netload-wait 1200 "$OUT/main.elf") 2>&1 | tr -d '\r' | sed -E "s/^(BOARD machine=\S+ midr=0x[0-9a-f]+)\$/\1 board=$NAME/" > "$RUN"
RRC=${PIPESTATUS[0]}
set -e
cat "$RUN"
echo "[p0] apppaltest rc=$RRC log=$RUN"

awk -v name="$NAME" -v kern="$KERNEL" -v clk="$CLK" '
/^BOARD machine=/ { split($2, a, "="); mach = a[2] }
/^\[PERF\] tag=[0-9]+ tile=/ {
    t = ""; ac = ls = ss = 0
    for (i = 2; i <= NF; i++) { split($i, f, "=")
        if (f[1] == "tag") t = f[2]; if (f[1] == "active_cyc") ac = f[2] + 0
        if (f[1] == "lock_stall") ls = f[2] + 0; if (f[1] == "stream_stall") ss = f[2] + 0 }
    if (!(t in A) || ac > A[t]) { A[t] = ac; C[t] = ac - ls - ss }
}
/^\[PERFSUM\] tag=/ { nps++ }
/^LAUNCH split_detail calls/ { for (i = 1; i <= NF; i++) { split($i, f, "=")
        if (f[1] == "rt_wait_io") w = f[2]; if (f[1] == "launches") L = f[2] } }
/^KCMP / { kcmp = $0 }
END {
    n = 0; sa = sc = 0
    for (t in A) { n++; sa += A[t]; sc += C[t] }
    if (w == "" || n == 0 || mach == "") { printf "P0 FAIL board=%s kernel=%s perf_tags=%d perfsum=%d rt_wait_io=%s machine=%s\n", name, kern, n, nps, w, mach; exit 1 }
    meas = sprintf("%.3f", w); act = sa / n; cmp = sc / n; pred = cmp / (clk * 1000.0)
    if (L == "") L = n
    printf "[p0] perf_tags=%d perfsum=%d aie_clk_mhz=%s (assumed unless --aie-clk-mhz) %s\n", n, nps, clk, kcmp
    printf "P0 board=%s machine=%s kernel=%s launches=%d rt_wait_io_ms=%s core_src=perfctr active_cyc=%d compute_cyc=%d pred_ms=%.3f meas_ms=%s rel=%.3f\n",
        name, mach, kern, L, meas, act, cmp, pred, meas, (meas > 0 ? (pred > meas ? pred - meas : meas - pred) / meas : 0)
}' "$RUN"
[ "$RRC" -eq 0 ] || { echo "[p0] apppaltest rc=$RRC"; exit "$RRC"; }
