#!/usr/bin/env bash
# Cost-model P0 / K1a: one gemm_kacc kernel from kernels.tsv on the board, with per-launch AIE core perf counters.
#   p0_one_kernel.sh --board <ip> [--kernel <name>] [--kernels <kernels.tsv>] [--name palmyra213] [--out <dir>]
#                    [--reps 8] [--aie-clk-mhz 1000] [--rebuild] [--ctrlpkt off|aot|aot-kco] [--build-only]
#                    [--launch warm|reload|cold] [--alt <k2>]
#   p0_one_kernel.sh --parse-log <run.log> --kernel <k> --ctrlpkt <m> --launch <l>   (offline replay of a board run.log)
# --alt <k2> (M1.2.2 P0b, core_reset probe): a multi-kernel build of two gemm_kacc instances, <kernel> (A) and <k2> (B),
# both on the same 4x4 mesh (the same 16 tiles), launched A,B,A,B... so every launch loads the other kernel's ELF
# (UNet LAUNCH_OPTS_WARM core_reset_unreset_per_launch, boardmain.py:33; use with --ctrlpkt off --launch warm). The
# host is the alt host below (p0_alt_host.cc in the build dir): p0_kacc_host.cc's lines, with split_detail core_reset =
# the runtime counter g_kl_rst_n (__Runtime_kload_split_cycles) instead of the literal 0, one KCMP line per kernel.
# Both kernels must have the same port count (GAP-aiehlc-multi-kernel-__ai-2d6356: __aie_launch is generated for the
# largest port count); two plain gemm_kacc kernels (3 ports) are fine. The P0 line carries alt=<k2>.
# --launch (M1.2.1): warm (default) = WARM_DEFS (ZFILL_WIDE PMLOAD_WIDE PM_WARM_SKIP SKIP_BSS_DEFAULT); reload = warm
# minus PM_WARM_SKIP (program memory reloaded every launch, PMLOAD_WIDE kept); cold = no WARM_DEFS (runtime defaults).
# --ctrlpkt (M1.1): off = MMIO (UNET_CTRLPKT=off, no control-packet pragmas), aot (default, the 5 aot pragmas),
# aot-kco (UNET_CTRLPKT_KCO=1, + KERNELCONFIGOFFLOAD; as build_unet.py --ctrlpkt aot-kco). --build-only: stop after
# BUILD_OK (no --board needed). The stamp carries the ctrlpkt mode + md5 of unet_aie.py / gemm/ctrlpkt_*.h.
# Builds (cached by a stamp of shape + host + commit) the gemm_kacc instance of the kernel's GEMM shape (M,K,N) through
# example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py with the P0 host script/perf/p0_kacc_host.cc and an aiehlc.sh
# --profiling host (core perf counters ACTIVE/VEC/STREAM_STALL/LOCK_STALL armed at every launch), runs it with
# script/test/apppaltest.py (bare-metal, netload) and prints the console (CR stripped) plus one line:
#   P0 board=<name> machine=aarch64 kernel=<k> launches=<n> rt_wait_io_ms=<split_detail rt_wait_io, %.3f>
#      core_src=perfctr active_cyc=<a> compute_cyc=<c> pred_ms=<c/aie_clk> meas_ms=<= rt_wait_io_ms> rel=<|p-m|/m>
# active_cyc / compute_cyc: per timed launch, the slowest core tile (max active_cyc over the [PERF] tiles of that tag),
# compute = active - lock_stall - stream_stall of that tile; the P0 line carries the mean over the launches.
# Then one line PERFCHK cfg=<ctrlpkt> kernel=<k> active_cyc=<P0 mean> io_active_cyc=<[PERFIO] io_active, slowest core,
# mean> wait_cyc=<rt_wait_io_ms x clk_mhz x 1e3> io_le_wait=<0|1> (counter 0 sanity after the ECC-scrub fix a56aa0f3).
# Inputs are synthetic (seeded N(0,1) A[M,K], B[K,N]; y = A@B fp32): the shape is the kernel's, the data is not.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${KACC_PY:-/scratch/staff/huaj/aiehlc/.venv/bin/python3}"
PARSE_LOG="" BOARD="" KERNEL="u_lin_1024x320x320" NAME="" OUT="" REPS=8 CLK=1000 REBUILD=0 CTRLPKT=aot BUILD_ONLY=0 LAUNCH=warm ALT=""
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
    --ctrlpkt) CTRLPKT="${2:-}"; shift 2 ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --launch) LAUNCH="${2:-}"; shift 2 ;;
    --alt) ALT="${2:-}"; shift 2 ;;
    --parse-log) PARSE_LOG="${2:-}"; shift 2 ;;
    *) echo "usage: $0 --board <ip> [--kernel <name>] [--kernels <tsv>] [--name <board>] [--out <dir>] [--reps N] [--rebuild] [--ctrlpkt off|aot|aot-kco] [--build-only] [--launch warm|reload|cold] [--alt <k2>] [--parse-log <run.log>]" >&2; exit 2 ;;
    esac
done
case "$CTRLPKT" in
off) export UNET_CTRLPKT=off; unset UNET_CTRLPKT_KCO ;;
aot) unset UNET_CTRLPKT UNET_CTRLPKT_KCO ;;
aot-kco) export UNET_CTRLPKT_KCO=1; unset UNET_CTRLPKT ;;
*) echo "p0_one_kernel: --ctrlpkt off|aot|aot-kco (got '$CTRLPKT')" >&2; exit 2 ;;
esac
case "$LAUNCH" in
warm) LAUNCH_DEFS="-DAIEHLC_ZFILL_WIDE=1 -DAIEHLC_PMLOAD_WIDE=1 -DAIEHLC_PM_WARM_SKIP=1 -DAIEHLC_SKIP_BSS_DEFAULT=1" ;;
reload) LAUNCH_DEFS="-DAIEHLC_ZFILL_WIDE=1 -DAIEHLC_PMLOAD_WIDE=1 -DAIEHLC_SKIP_BSS_DEFAULT=1" ;;
cold) LAUNCH_DEFS="" ;;
*) echo "p0_one_kernel: --launch warm|reload|cold (got '$LAUNCH')" >&2; exit 2 ;;
esac
# [PERF]/[PERFIO]/LAUNCH lines of $RUN -> P0, PERFCHK, KWIN, KOVL (also used by --parse-log)
parse_run() {
    awk -v name="$NAME" -v kern="$KERNEL" -v alt="$ALT" -v cp="$CTRLPKT" -v lm="$LAUNCH" -v clk="$CLK" '
/^BOARD machine=/ { split($2, a, "="); mach = a[2] }
/^\[PERF\] tag=[0-9]+ tile=/ {
    t = ""; ac = ls = ss = 0
    for (i = 2; i <= NF; i++) { split($i, f, "=")
        if (f[1] == "tag") t = f[2]; if (f[1] == "active_cyc") ac = f[2] + 0
        if (f[1] == "lock_stall") ls = f[2] + 0; if (f[1] == "stream_stall") ss = f[2] + 0 }
    if (!(t in A) || ac > A[t]) { A[t] = ac; C[t] = ac - ls - ss }
}
/^\[PERFIO\] tag=[0-9]+ tile=/ {
    t = ""; io = 0; pio = 0; gp = 0; cc = -1; oc = -1
    for (i = 2; i <= NF; i++) { split($i, f, "="); if (f[1] == "tag") t = f[2]; if (f[1] == "io_active") io = f[2] + 0
        if (f[1] == "pre_io_active") pio = f[2] + 0; if (f[1] == "snap_gap_cyc") gp = f[2] + 0
        if (f[1] == "compute_cyc") cc = f[2] + 0; if (f[1] == "ovl_cyc") oc = f[2] + 0 }
    if (!(t in IO) || io > IO[t]) { IO[t] = io; PIO[t] = pio }
    if (!(t in GP) || gp > GP[t]) GP[t] = gp
    # KOVL: compute_cyc / ovl_cyc per launch = the slowest tile (max); -1 when the field is missing (pre-KOVL logs)
    if (!(t in CC) || cc > CC[t]) CC[t] = cc
    if (!(t in OC) || oc > OC[t]) OC[t] = oc
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
    printf "P0 board=%s machine=%s kernel=%s ctrlpkt=%s launch=%s%s launches=%d rt_wait_io_ms=%s core_src=perfctr active_cyc=%d compute_cyc=%d pred_ms=%.3f meas_ms=%s rel=%.3f\n",
        name, mach, kern, cp, lm, (alt != "" ? " alt=" alt : ""), L, meas, act, cmp, pred, meas, (meas > 0 ? (pred > meas ? pred - meas : meas - pred) / meas : 0)
    # PERFCHK (M3.1.1 PC): io_active = core counter 0 summed over the wait windows only (rt_core_perf_wait_snap
    # after rt_sb_ack_wait, next to __wio_t0, read after the window; [PERFIO] io_active after PA), per launch the
    # slowest core tile (max io_active), mean over launches; snap_gap = host snapshot -> __wio_t0 gap per launch
    # (counters already running), mean over launches; wait_cyc = rt_wait_io_ms x 1e6 x clk/1000.
    # io_le_wait=1 when io <= wait + gap.  KWIN: ratio = io/(wait+gap); ok=1 iff perfio_n==launches, ratio<=1.02 and
    # (io>0 or wait<1e4 cyc): admin rule "io_act=0 is allowed when wait<1e4 cyc" (M3.1.1 PB) -- a launch whose wait
    # window is that short can legitimately see no core activity (PD.verify.2.log:29, y_conv off wait=4600, io=0).
    ni = 0; si = 0; spi = 0; sg = 0
    for (t in IO) { ni++; si += IO[t]; spi += PIO[t]; sg += GP[t] }
    io_m = (ni > 0 ? si / ni : -1); pio_m = (ni > 0 ? spi / ni : -1); gp_m = (ni > 0 ? sg / ni : 0); wc = w * clk * 1000.0
    printf "PERFCHK cfg=%s kernel=%s active_cyc=%d io_active_cyc=%d wait_cyc=%d io_le_wait=%d\n",
        cp, kern, act, io_m, wc, (ni > 0 && io_m <= wc + gp_m ? 1 : 0)
    den = wc + gp_m; ratio = (den > 0 ? io_m / den : -1)
    printf "KWIN cfg=%s kernel=%s launches=%d perfio_n=%d io_active_cyc=%d pre_io_active_cyc=%d wait_cyc=%d snap_gap_cyc=%d ratio=%.3f ok=%d\n",
        cp, kern, L, ni, io_m, pio_m, wc, gp_m, ratio, (ni == L + 0 && ratio >= 0 && ratio <= 1.02 && (io_m > 0 || wc < 10000) ? 1 : 0)
    # KOVL (M3.1.1 PB): compute_cyc / ovl_cyc = [PERFIO] fields, slowest tile per launch, mean over launches (-1 if any
    # launch lacks them); io_unhidden_cyc = wait_cyc - ovl_cyc (>= 0), the wait not hidden behind compute; -1 w/o ovl.
    nc = 0; scc = 0; soc = 0; cmiss = 0; omiss = 0
    for (t in CC) { nc++; if (CC[t] < 0) cmiss = 1; else scc += CC[t]; if (OC[t] < 0) omiss = 1; else soc += OC[t] }
    cc_m = (nc > 0 && !cmiss ? scc / nc : -1); oc_m = (nc > 0 && !omiss ? soc / nc : -1)
    iu = (oc_m >= 0 ? (wc > oc_m ? wc - oc_m : 0) : -1)
    printf "KOVL cfg=%s kernel=%s compute_cyc=%d ovl_cyc=%d io_unhidden_cyc=%d\n", cp, kern, cc_m, oc_m, iu
}' "$RUN"
}
if [ -n "$PARSE_LOG" ]; then
    # --parse-log <run.log> (M3.1.1 PB): offline replay of a board run.log -- no kmat lookup, no build, no board;
    # machine and board name come from the log's BOARD line.
    [ -f "$PARSE_LOG" ] || { echo "p0_one_kernel: --parse-log $PARSE_LOG: no such file" >&2; exit 2; }
    RUN="$PARSE_LOG"
    NAME="$(sed -n 's/^BOARD machine=.* board=\(\S*\).*/\1/p' "$RUN" | head -1)"; NAME="${NAME:-${BOARD:-unknown}}"
    echo "[p0] --parse-log $RUN kernel=$KERNEL ctrlpkt=$CTRLPKT launch=$LAUNCH board=$NAME"
    parse_run
    exit $?
fi
[ -n "$BOARD" ] || [ "$BUILD_ONLY" = 1 ] || { echo "p0_one_kernel: --board <ip> required (or --build-only)" >&2; exit 2; }
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
if [ -n "$ALT" ]; then
    [ "$RECIPE" != kacc_chunk_conv_bf16 ] || { echo "p0_one_kernel: --alt needs a plain gemm kernel (got $RECIPE)" >&2; exit 2; }
    read -r M2 K2 N2 RECIPE2 < <(awk -F'\t' -v k="$ALT" 'NR>1 && $1==k {print $3, $4, $5, $2}' "$KTSV")
    [ -n "${M2:-}" ] || { echo "p0_one_kernel: --alt kernel $ALT not in $KTSV" >&2; exit 2; }
    [ "$RECIPE2" != kacc_chunk_conv_bf16 ] || { echo "p0_one_kernel: --alt needs a plain gemm kernel (got $RECIPE2)" >&2; exit 2; }
    [ "$ALT" != "$KERNEL" ] || { echo "p0_one_kernel: --alt must differ from --kernel" >&2; exit 2; }
fi
OUT="${OUT:-/scratch/staff/huaj/tmp/costmodel_p0/$KERNEL${ALT:++$ALT}}"
mkdir -p "$OUT/golden"
OUT="$(cd "$OUT" && pwd)"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
STAMP="$M $K $N reps=$REPS ctrlpkt=$CTRLPKT launch=$LAUNCH $(cat "$HERE/p0_kacc_host.cc" "$ROOT/example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py" \
    "$ROOT/example/aiegraph/unet_bk_sdm_tiny/unet_aie.py" "$ROOT"/src/aiednn/gemm/ctrlpkt_aot_pragmas.h \
    "$ROOT"/src/aiednn/gemm/ctrlpkt_kco_pragma.h | md5sum | cut -c1-12) $COMMIT"
[ -z "$ALT" ] || STAMP="$STAMP alt=$ALT $M2 $K2 $N2 $(md5sum < "${BASH_SOURCE[0]}" | cut -c1-12)"
echo "[p0] kernel=$KERNEL recipe=$RECIPE M=$M K=$K N=$N ctrlpkt=$CTRLPKT launch=$LAUNCH board=$NAME ($BOARD) out=$OUT commit=$COMMIT"
[ -z "$ALT" ] || echo "[p0] alt=$ALT recipe=$RECIPE2 M=$M2 K=$K2 N=$N2: A,B,A,B... on the same 16 tiles (core_reset_unreset_per_launch)"

# --alt host (KACC_HOST_FILE for A; @NAME@/@ARGS@ = A filled by kacc_gemm.py, @NAME2@/@ARGS2@ + /*@KALT_DATA@*/ = B)
alt_host() {
    cat <<'EOF'
/* p0_alt_host.cc -- generated by script/perf/p0_one_kernel.sh --alt: two gemm_kacc instances on the same 4x4 mesh,
 * launched A,B,A,B... (every launch loads the other kernel: core reset/unreset per launch). p0_kacc_host.cc lines;
 * split_detail core_reset = g_kl_rst_n delta over the timed pass (__Runtime_kload_split_cycles). */
#ifndef KACC_REPS
#define KACC_REPS 8
#endif
/*@KALT_DATA@*/
void __Runtime_ctrl_high_throughput_enable(int on); /* C++ linkage, as aie_runtime.h:936 */
void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
void __Runtime_core_perf_report(int tag);
void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc,
                                  unsigned int *rst_n);
void __Runtime_launch_words(unsigned long long *mmio_w, unsigned long long *cp_w, unsigned long long *ph_w,
                            unsigned int *bd_n, unsigned int *rst_n, unsigned long long *miss);
int g_runtime_debug_level = 0x100; /* AIE_DEBUG_FLAG_CORE_PERF_COUNTER (aie_runtime.h:69) */

static inline unsigned long long p0_pmc(void) {
    unsigned long long v;
#if defined(__aarch64__)
    __asm__ volatile("isb; mrs %0, pmccntr_el0" : "=r"(v));
#else
    v = 0ULL;
#endif
    return v;
}
static inline unsigned long long p0_rt_cyc(void) {
    unsigned long long ph[4] = {0, 0, 0, 0}, w = 0;
    unsigned int pn[4] = {0, 0, 0, 0}, wn = 0;
    __Runtime_phase_cycles(ph, pn);
    __Runtime_wait_io_cycles(&w, &wn);
    return ph[0] + ph[1] + ph[2] + ph[3] + w;
}
/* p0_kacc_host.cc kacc_yidx with the instance geometry as arguments */
static size_t p0_yidx(int m, int n, int split_px, int pxs, int chs, int tr) {
    const int s = split_px ? m / pxs : n / chs;
    const int ml = split_px ? m % pxs : m, nl = split_px ? n : n % chs;
    const int Ml = pxs, Nl = chs, r = nl & 15, c = ml & 15;
    size_t base = (size_t)s * Ml * Nl, i;
    if (!tr) {
        const int q = Ml / 4, cc = ml % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
        i = (size_t)((nl & ~15) + wi / q) * Ml + (ml - cc) + wi % q;
    } else {
        const int q = Nl / 4, cc = nl % q;
        const int wi = (cc >> 4) * 256 + ((r >> 3) * 2 + (c >> 3)) * 64 + (r & 7) * 8 + (c & 7);
        i = (size_t)((ml & ~15) + wi / q) * Nl + (nl - cc) + wi % q;
    }
    return base + i;
}
static int p0_kcmp(const int32_t *Yd, const uint8_t *ybin, int M, int N, int split_px, int pxs, int chs, int tr,
                   const char *tag, const char *data, const char *ysha) {
    const float *ref = (const float *)ybin; /* [M][N] */
    double dot = 0, na = 0, nr = 0, maxabs = 0;
    int nonfinite = 0;
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float f;
            memcpy(&f, &Yd[p0_yidx(m, n, split_px, pxs, chs, tr)], 4);
            const double o = (double)f, r = ref[(size_t)m * N + n];
            if (!(o == o) || o > 1e30 || o < -1e30) {
                nonfinite++;
                continue;
            }
            dot += o * r;
            na += o * o;
            nr += r * r;
            if (fabs(o - r) > maxabs)
                maxabs = fabs(o - r);
        }
    const double cs = nonfinite ? 0.0 : dot / (sqrt(na) * sqrt(nr) + 1e-30);
    const int pass = nonfinite == 0 && cs >= 0.9995;
    printf("[PDK] %s maxabs_vs_ref=%.6g nonfinite=%d M=%d N=%d\n", tag, maxabs, nonfinite, M, N);
    printf("KCMP %s data=%s y_sha256=%s cos=%.6f %s\n", tag, data, ysha, cs, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
#if defined(__aarch64__)
    {
        unsigned long midr;
        __asm__ volatile("mrs %0, midr_el1" : "=r"(midr));
        printf("BOARD machine=aarch64 midr=0x%08lx\n", midr);
    }
#else
    printf("BOARD machine=other midr=0x00000000\n");
#endif
    printf("=== P0 alt: A @NAME@ (M=%d K=%d N=%d, %d launch(es)) and B @NAME2@ (M=%d K=%d N=%d, %d launch(es)) "
           "alternating A,B per pass x %d timed passes, same 4x4 mesh, core perf counters on ===\n",
           KACC_M, KACC_K, KACC_N, KACC_S, KALT_M, KALT_K, KALT_N, KALT_S, KACC_REPS);
    __Runtime_ctrl_high_throughput_enable(1);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({2, 6, 0, 6}, 4, 4);
    const size_t wbytes = (size_t)KACC_N * KACC_KP * 2, xbytes = (size_t)KACC_MP * KACC_KP * 2;
    const size_t ybytes = (size_t)KACC_N * KACC_MP * 4;
    const size_t wbytes2 = (size_t)KALT_N * KALT_KP * 2, xbytes2 = (size_t)KALT_MP * KALT_KP * 2;
    const size_t ybytes2 = (size_t)KALT_N * KALT_MP * 4;
    int32_t *Wd = (int32_t *)__Runtime_Alloc(wbytes);
    int32_t *Xd = (int32_t *)__Runtime_Alloc(xbytes);
    int32_t *Yd = (int32_t *)__Runtime_Alloc(ybytes);
    int32_t *Wd2 = (int32_t *)__Runtime_Alloc(wbytes2);
    int32_t *Xd2 = (int32_t *)__Runtime_Alloc(xbytes2);
    int32_t *Yd2 = (int32_t *)__Runtime_Alloc(ybytes2);
    if (!Wd || !Xd || !Yd || !Wd2 || !Xd2 || !Yd2) {
        printf("KCMP %s data=%s y_sha256=%s cos=0.000000 FAIL (alloc)\n", KACC_TAG, KACC_DATA, KACC_Y_SHA);
        return 1;
    }
    memcpy(Wd, kacc_w_bin, wbytes);
    memcpy(Xd, kacc_x_bin, xbytes);
    memcpy(Wd2, kalt_w_bin, wbytes2);
    memcpy(Xd2, kalt_x_bin, xbytes2);
    Xil_DCacheFlushRange((UINTPTR)Wd, wbytes);
    Xil_DCacheFlushRange((UINTPTR)Xd, xbytes);
    Xil_DCacheFlushRange((UINTPTR)Wd2, wbytes2);
    Xil_DCacheFlushRange((UINTPTR)Xd2, xbytes2);
    memset(Yd, 0xff, ybytes);
    memset(Yd2, 0xff, ybytes2);
    Xil_DCacheFlushRange((UINTPTR)Yd, ybytes);
    Xil_DCacheFlushRange((UINTPTR)Yd2, ybytes2);
    /* warm-up pass A..., B...: both kernels loaded once + first-launch config, not counted */
    for (int s = 0; s < KACC_S; s++) {
        int32_t *Ws = Wd + (KACC_SPLIT_PX ? 0 : (size_t)s * KACC_CHS * KACC_KP / 2);
        int32_t *Xs = Xd + (KACC_SPLIT_PX ? (size_t)s * KACC_PXS * KACC_KP / 2 : 0);
        int32_t *Ys = Yd + (size_t)s * KACC_CHS * KACC_PXS;
        @NAME@<<<mesh>>>(@ARGS@);
    }
    for (int s = 0; s < KALT_S; s++) {
        int32_t *Ws = Wd2 + (KALT_SPLIT_PX ? 0 : (size_t)s * KALT_CHS * KALT_KP / 2);
        int32_t *Xs = Xd2 + (KALT_SPLIT_PX ? (size_t)s * KALT_PXS * KALT_KP / 2 : 0);
        int32_t *Ys = Yd2 + (size_t)s * KALT_CHS * KALT_PXS;
        @NAME2@<<<mesh>>>(@ARGS2@);
    }
    Xil_DCacheInvalidateRange((UINTPTR)Yd, ybytes);
    Xil_DCacheInvalidateRange((UINTPTR)Yd2, ybytes2);
    printf("[P0] warmup done launches=%d\n", KACC_S + KALT_S);

    unsigned long long ph0[4] = {0, 0, 0, 0}, ph1[4] = {0, 0, 0, 0}, wio0 = 0, wio1 = 0;
    unsigned int phn0[4] = {0, 0, 0, 0}, phn1[4] = {0, 0, 0, 0}, wion0 = 0, wion1 = 0;
    unsigned long long kel0 = 0, kel1 = 0, krst0 = 0, krst1 = 0;
    unsigned int keln0 = 0, keln1 = 0, krstn0 = 0, krstn1 = 0;
    unsigned long long sp_pre = 0, sp_call = 0, sp_glue = 0, sp_post = 0, ticks = 0;
    __Runtime_phase_cycles(ph0, phn0);
    __Runtime_wait_io_cycles(&wio0, &wion0);
    __Runtime_kload_split_cycles(&kel0, &keln0, &krst0, &krstn0);
    unsigned long long wm0 = 0, wc0 = 0, wp0[4] = {0, 0, 0, 0}, wx0 = 0;
    unsigned int wbd0 = 0, wrs0 = 0;
    __Runtime_launch_words(&wm0, &wc0, wp0, &wbd0, &wrs0, &wx0);
    XTime g0, g1;
    XTime_GetTime(&g0);
    const unsigned long long pmc0 = p0_pmc();
    int launches = 0;
    const int smax = KACC_S > KALT_S ? KACC_S : KALT_S;
/* one timed launch (p0_kacc_host.cc regions): flush W/X/Y | launch | invalidate Y; perf report outside.
 * The launch itself stays outside the macros (its template argument list has commas). */
#define P0_PRE(WS, WB, XS, XB, YS, YB)                                                                            \
    XTime lt0, lt1;                                                                                               \
    XTime_GetTime(&lt0);                                                                                          \
    const unsigned long long a0 = p0_pmc();                                                                       \
    Xil_DCacheFlushRange((UINTPTR)(WS), (WB));                                                                    \
    Xil_DCacheFlushRange((UINTPTR)(XS), (XB));                                                                    \
    Xil_DCacheFlushRange((UINTPTR)(YS), (YB));                                                                    \
    const unsigned long long r0 = p0_rt_cyc(), a1 = p0_pmc();
#define P0_POST(YS, YB)                                                                                           \
    const unsigned long long a2 = p0_pmc(), r1 = p0_rt_cyc();                                                     \
    Xil_DCacheInvalidateRange((UINTPTR)(YS), (YB));                                                               \
    const unsigned long long a3 = p0_pmc();                                                                       \
    XTime_GetTime(&lt1);                                                                                          \
    sp_pre += a1 - a0;                                                                                            \
    sp_call += a2 - a1;                                                                                           \
    sp_glue += (a2 - a1) > (r1 - r0) ? (a2 - a1) - (r1 - r0) : 0ULL;                                              \
    sp_post += a3 - a2;                                                                                           \
    ticks += (unsigned long long)(lt1 - lt0);                                                                     \
    __Runtime_core_perf_report(launches);                                                                         \
    launches++;
    for (int rep = 0; rep < KACC_REPS; rep++) {
        memset(Yd, 0xff, ybytes);
        memset(Yd2, 0xff, ybytes2);
        for (int s = 0; s < smax; s++) {
            if (s < KACC_S) {
                int32_t *Ws = Wd + (KACC_SPLIT_PX ? 0 : (size_t)s * KACC_CHS * KACC_KP / 2);
                int32_t *Xs = Xd + (KACC_SPLIT_PX ? (size_t)s * KACC_PXS * KACC_KP / 2 : 0);
                int32_t *Ys = Yd + (size_t)s * KACC_CHS * KACC_PXS;
                P0_PRE(Ws, (size_t)KACC_CHS * KACC_KP * 2, Xs, (size_t)KACC_PXS * KACC_KP * 2, Ys,
                       (size_t)KACC_CHS * KACC_PXS * 4)
                @NAME@<<<mesh>>>(@ARGS@);
                P0_POST(Ys, (size_t)KACC_CHS * KACC_PXS * 4)
            }
            if (s < KALT_S) {
                int32_t *Ws = Wd2 + (KALT_SPLIT_PX ? 0 : (size_t)s * KALT_CHS * KALT_KP / 2);
                int32_t *Xs = Xd2 + (KALT_SPLIT_PX ? (size_t)s * KALT_PXS * KALT_KP / 2 : 0);
                int32_t *Ys = Yd2 + (size_t)s * KALT_CHS * KALT_PXS;
                P0_PRE(Ws, (size_t)KALT_CHS * KALT_KP * 2, Xs, (size_t)KALT_PXS * KALT_KP * 2, Ys,
                       (size_t)KALT_CHS * KALT_PXS * 4)
                @NAME2@<<<mesh>>>(@ARGS2@);
                P0_POST(Ys, (size_t)KALT_CHS * KALT_PXS * 4)
            }
        }
    }
    const unsigned long long pmc1 = p0_pmc();
    XTime_GetTime(&g1);
    __Runtime_phase_cycles(ph1, phn1);
    __Runtime_wait_io_cycles(&wio1, &wion1);
    __Runtime_kload_split_cycles(&kel1, &keln1, &krst1, &krstn1);
    unsigned long long wm1 = 0, wc1 = 0, wp1[4] = {0, 0, 0, 0}, wx1 = 0;
    unsigned int wbd1 = 0, wrs1 = 0;
    __Runtime_launch_words(&wm1, &wc1, wp1, &wbd1, &wrs1, &wx1);
    unsigned long long ph[4];
    unsigned int phn[4];
    for (int i = 0; i < 4; i++) {
        ph[i] = ph1[i] - ph0[i];
        phn[i] = phn1[i] - phn0[i];
    }
    const unsigned long long wio = wio1 - wio0, krst = krst1 - krst0;
    const unsigned int wion = wion1 - wion0, krstn = krstn1 - krstn0;
    const double g_s = (double)(g1 - g0) / (double)COUNTS_PER_SECOND;
    const unsigned long long pmu_hz = g_s > 0.0 ? (unsigned long long)((double)(pmc1 - pmc0) / g_s + 0.5) : 0ULL;
    const double cms = pmu_hz ? 1000.0 / (double)pmu_hz : 0.0;
    const double nl = launches ? (double)launches : 1.0;
    const double aie_ms = 1000.0 * (double)ticks / (double)COUNTS_PER_SECOND;
    const unsigned long long c_kl = ph[0], c_bd = ph[1] + sp_glue, c_si = ph[3] + sp_pre, c_ce = ph[2],
                             c_wi = wio + sp_post;
    const double rt_ms = (double)(ph[0] + ph[1] + ph[2] + ph[3] + wio) * cms;
    const double sum_ms = (double)(c_kl + c_bd + c_si + c_ce + c_wi) * cms;
    printf("[BENCH] aie_wall_ms=%.3f launches=%d gmac=%.4f\n", aie_ms, launches,
           (KACC_GMAC + KALT_GMAC) * KACC_REPS);
    printf("LAUNCH split kload=%.3f bdcfg=%.3f startio=%.3f coreen=%.3f wait_io=%.3f core_reset=%.3f "
           "ms_per_launch=%.3f pmu_hz=%llu sum_vs_aie_wall=%.3f\n",
           c_kl * cms / nl, c_bd * cms / nl, c_si * cms / nl, c_ce * cms / nl, c_wi * cms / nl, krst * cms / nl,
           aie_ms / nl, pmu_hz, aie_ms > 0.0 ? sum_ms / aie_ms : 0.0);
    printf("LAUNCH split_detail calls kload=%u bdcfg=%u startio=%u coreen=%u wait_io=%u core_reset=%u "
           "rt_bdcfg=%.4f rt_startio=%.4f rt_wait_io=%.4f "
           "host_glue_bdcfg=%.4f host_flush_startio=%.4f host_inval_wait_io=%.4f gemm_call=%.4f "
           "rt_phase_sum_ms=%.3f phase_sum_ms=%.3f aie_wall_ms=%.3f unattributed_ms_per_launch=%.4f "
           "pmc_delta=%llu xtime_s=%.3f launches=%d\n",
           phn[0], phn[1], phn[3], phn[2], wion, krstn, ph[1] * cms / nl, ph[3] * cms / nl, wio * cms / nl,
           sp_glue * cms / nl, sp_pre * cms / nl, sp_post * cms / nl, sp_call * cms / nl, rt_ms, sum_ms, aie_ms,
           (aie_ms - sum_ms) / nl, pmc1 - pmc0, g_s, launches);
#if defined(AIEHLC_PROFILING) && AIEHLC_PROFILING
    printf("LAUNCH split_words mmio_w=%.1f cp_w=%.1f bd_n=%.1f rst_n=%.1f kload_w=%.1f bdcfg_w=%.1f startio_w=%.1f coreen_w=%.1f\n",
           (double)(wm1 - wm0) / nl, (double)(wc1 - wc0) / nl, (double)(wbd1 - wbd0) / nl, (double)(wrs1 - wrs0) / nl,
           (double)(wp1[0] - wp0[0]) / nl, (double)(wp1[1] - wp0[1]) / nl, (double)(wp1[3] - wp0[3]) / nl,
           (double)(wp1[2] - wp0[2]) / nl);
    printf("[P0] split_words_miss=%llu kload_elf_n=%u\n", wx1 - wx0, keln1 - keln0);
#endif
    const int pa = p0_kcmp(Yd, kacc_y_bin, KACC_M, KACC_N, KACC_SPLIT_PX, KACC_PXS, KACC_CHS, KACC_TR, KACC_TAG,
                           KACC_DATA, KACC_Y_SHA);
    const int pb = p0_kcmp(Yd2, kalt_y_bin, KALT_M, KALT_N, KALT_SPLIT_PX, KALT_PXS, KALT_CHS, KALT_TR,
                           KALT_TAG "-alt", KALT_DATA, KALT_Y_SHA);
    fflush(stdout);
    return pa && pb ? 0 : 1;
}
EOF
}

if [ "$REBUILD" = 1 ] || [ ! -f "$OUT/main.elf" ] || [ "$(cat "$OUT/stamp" 2>/dev/null)" != "$STAMP" ]; then
    T0=$(date +%s)
    # synthetic MatMul golden in the op_golden layout kacc_gemm.py reads (a [1,M,K], b [K,N], y [1,M,N], meta.json)
    golden() { "$PY" -I - "$@" <<'EOF'
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
    }
    golden "$OUT/golden/c000_matmul_p0" "$M" "$K" "$N"
    rm -rf "$OUT/aout" "$OUT/main.elf"
    if [ -z "$ALT" ]; then
        KACC_HOST_FILE="$HERE/p0_kacc_host.cc" "$PY" -I "$ROOT/example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py" \
            "$OUT/golden/c000_matmul_p0" "$OUT"
    else
        # B: data + instance name from its own kacc_gemm.py run in $OUT/alt; A: kacc_gemm.py with the alt host
        mkdir -p "$OUT/alt/golden"
        golden "$OUT/alt/golden/c000_matmul_p0" "$M2" "$K2" "$N2"
        KACC_HOST_FILE="$HERE/p0_kacc_host.cc" "$PY" -I "$ROOT/example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py" \
            "$OUT/alt/golden/c000_matmul_p0" "$OUT/alt" | tee "$OUT/alt/kacc_gemm.log"
        alt_host > "$OUT/p0_alt_host.cc"
        KACC_HOST_FILE="$OUT/p0_alt_host.cc" "$PY" -I "$ROOT/example/aiegraph/unet_bk_sdm_tiny/kacc_gemm.py" \
            "$OUT/golden/c000_matmul_p0" "$OUT"
        # B's data header inlined with KACC_/kacc_ renamed KALT_/kalt_; B's launch callee + argument order
        "$PY" -I - "$OUT" <<'EOF'
import re, sys
from pathlib import Path
o = Path(sys.argv[1])
name = re.findall(r"^\[kacc_bench\] .* kernel=(.+) launches_per_pass=", (o / "alt/kacc_gemm.log").read_text(), re.M)[-1]
hdr = (o / "alt/kacc_data.h").read_text().replace("KACC_", "KALT_").replace("kacc_", "kalt_")
hdr = hdr.replace("/alt/kalt_", "/alt/kacc_")  # .incbin paths: B's files keep their names
tr = int(re.search(r"#define KALT_TR (\d+)", hdr).group(1))
src = (o / "kacc_gemm.cc").read_text()
assert src.count("/*@KALT_DATA@*/") == 1 and "@NAME2@" in src, "alt host placeholders missing"
src = src.replace("/*@KALT_DATA@*/", hdr).replace("@NAME2@", name).replace("@ARGS2@", "Xs, Ws, Ys" if tr else "Ws, Xs, Ys")
(o / "kacc_gemm.cc").write_text(src)
print(f"[p0] alt kernel B={name} tr={tr} inlined into {o / 'kacc_gemm.cc'}")
EOF
    fi
    sed -i "1i #define KACC_REPS $REPS" "$OUT/kacc_gemm.cc"
    (
        cd "$OUT"
        export AIEHLC_ROOT="$ROOT"
        # --launch warm = kacc_bench.sh --gemm (build_unet.py --launch-opt warm); reload / cold see header; plus --profiling
        export EXTRA_DEFS="${EXTRA_DEFS:-} $LAUNCH_DEFS"
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
[ "$BUILD_ONLY" = 1 ] && { echo "[p0] --build-only ctrlpkt=$CTRLPKT launch=$LAUNCH: $OUT/main.elf"; exit 0; }

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

parse_run
[ "$RRC" -eq 0 ] || { echo "[p0] apppaltest rc=$RRC"; exit "$RRC"; }
