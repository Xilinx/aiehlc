#!/usr/bin/env bash
# GAP-125ece regression test (x86, no board): KERNELCONFIGOFFLOAD in a multi-kernel program.
#
#   script/test/kco_multikernel/run.sh [out dir]      (AIEHLC_BIN=<aiehlc> to test a side build)
#
# Builds kco2.cc (two gemm_kacc kernels with different core output windows, 5120 B and 8192 B) through
# script/aiehlc.sh --aie-version 5 and checks the KERNELCONFIGOFFLOAD code each kernel's core runs:
#   1. every core MM2S BD the kernel self-programs (aie_kc_encode_bd on buf_out_p*ng_0) has the length of
#      that kernel's own output buffer -- the bug armed every later kernel with kernel #0's length (the
#      ResourceMgr singleton's core_offload_plan was never reset between kernels; first entry per tile won),
#      so on palmyra213 the 960x320x512 cores sent 5120 B windows into 8192 B shim S2MM BDs -> wait_io hang;
#   2. each kernel attaches a plan of its own size (not the running total of all kernels compiled so far).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/kco_multikernel.XXXXXX")}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
rm -rf "$OUT/aout"
cp -f "$HERE/kco2.cc" "$OUT/kco2.cc"
echo "[kco_multikernel] root=$ROOT out=$OUT AIEHLC_BIN=${AIEHLC_BIN:-default}"
(
    cd "$OUT" || exit 1
    set +u
    # shellcheck disable=SC1091
    source "$ROOT/script/aiehlc.sh" --aie-version 5 --runtime-source-file "$OUT/kco2.cc"
) > "$OUT/build.log" 2>&1
W="$OUT/aout/worklocal"
fail=0
nk=0
for k in gemm_kacc_320x320x1024 gemm_kacc_960x320x512; do
    f="$W/kernel_$k.cc"
    if [ ! -f "$f" ]; then
        echo "[kco_multikernel] FAIL $k: no $f (see $OUT/build.log)"
        fail=1
        continue
    fi
    nk=$((nk + 1))
    n=$(grep -oE 'v4int32 buf_out_ping_0\[[0-9]+\]' "$f" | grep -oE '[0-9]+' | tail -1)
    want=$(( ${n:-0} * 16 ))
    lens=$(grep -oE 'aie_kc_encode_bd\(_kc, [0-9]+, \(uintptr_t\)buf_out_p[io]ng_0, [0-9]+' "$f" | awk -F', ' '{print $NF}' | sort | uniq -c | tr -s ' ' | tr '\n' ';')
    bad=$(grep -oE 'aie_kc_encode_bd\(_kc, [0-9]+, \(uintptr_t\)buf_out_p[io]ng_0, [0-9]+' "$f" | awk -F', ' -v w="$want" '$NF != w' | wc -l)
    arms=$(grep -cE 'aie_kc_encode_bd\(_kc, [0-9]+, \(uintptr_t\)buf_out_p[io]ng_0' "$f")
    if [ "$want" -le 0 ] || [ "$arms" -eq 0 ] || [ "$bad" -ne 0 ]; then
        echo "[kco_multikernel] FAIL $k: out buffer ${want} B, core MM2S BD lengths {${lens}} (${bad}/${arms} wrong)"
        fail=1
    else
        echo "[kco_multikernel] ok   $k: out buffer ${want} B, core MM2S BD lengths {${lens}}"
    fi
done
plans=$(grep -oE 'attached core_offload_plan with [0-9]+' "$OUT/build.log" | awk '{print $NF}' | tr '\n' ' ')
nplan=$(echo $plans | wc -w)
nuniq=$(printf '%s\n' $plans | sort -u | grep -c .)
if [ "$nplan" -ne 2 ] || [ "$nuniq" -ne 1 ]; then
    echo "[kco_multikernel] FAIL core_offload_plan sizes per kernel: {${plans}} (want 2 equal sizes)"
    fail=1
else
    echo "[kco_multikernel] ok   core_offload_plan sizes per kernel: {${plans}}"
fi
if [ "$fail" -eq 0 ] && [ "$nk" -eq 2 ]; then
    echo "KCO_MULTIKERNEL PASS kernels=$nk plans={${plans}} log=$OUT/build.log"
    exit 0
fi
echo "KCO_MULTIKERNEL FAIL log=$OUT/build.log"
exit 1
