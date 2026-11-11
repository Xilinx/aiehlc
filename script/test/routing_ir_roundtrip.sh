#!/usr/bin/env bash
# Routing IR re-entry round trip (doc/design/routing_ir_reentry.md).
#
#   script/test/routing_ir_roundtrip.sh <src.cc> [aiehlc.sh args...]
#
# 1. builds <src.cc> normally (baseline),
# 2. builds it again with AIEHLC_ROUTING_IR_IN = the baseline's
#    ir/dfschedule/0_initial.mlir (the routing IR fed back unchanged),
# 3. requires every generated file and every dumped IR stage to be byte-identical.
# Exit 0 = the routing IR carries the whole tiling decision for this source.
# Default aiehlc.sh args: --aie-version 5. Work dirs under ${TMPDIR:-/tmp}.
set -uo pipefail
SRC="$(realpath "${1:?usage: $0 <src.cc> [aiehlc.sh args...]}")"
shift
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--aie-version 5)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/routing_ir_rt.XXXXXX")"
build_in() {
    mkdir -p "$1"
    (cd "$1" && bash -c "source '$REPO/script/aiehlc.sh' ${ARGS[*]} --runtime-source-file '$SRC'" > build.log 2>&1)
}
echo "[roundtrip] $SRC (work dir $WORK)"
build_in "$WORK/base" || { echo "[roundtrip] FAIL: baseline build failed ($WORK/base/build.log)"; exit 1; }
IR="$WORK/base/ir/dfschedule/0_initial.mlir"
[ -f "$IR" ] || { echo "[roundtrip] FAIL: no $IR"; exit 1; }
cp "$IR" "$WORK/routing_in.mlir"
AIEHLC_ROUTING_IR_IN="$WORK/routing_in.mlir" build_in "$WORK/reentry" ||
    { echo "[roundtrip] FAIL: re-entry build failed ($WORK/reentry/build.log)"; exit 1; }
grep -h "AIEHLC_ROUTING_IR_IN" "$WORK/reentry/build.log" | head -3
grep -q "identical to the generated IR" "$WORK/reentry/build.log" ||
    { echo "[roundtrip] FAIL: the IR read back differs from the generated IR"; exit 1; }
fail=0
for f in kernel.cc routing.cc host.cc aieml.bcf aieml.prx; do
    if cmp -s "$WORK/base/aout/worklocal/$f" "$WORK/reentry/aout/worklocal/$f"; then
        echo "  identical  $f"
    else
        echo "  DIFFERS    $f"; fail=1
    fi
done
n=0
while IFS= read -r ir; do
    n=$((n + 1))
    cmp -s "$WORK/base/$ir" "$WORK/reentry/$ir" || { echo "  DIFFERS    $ir"; fail=1; }
done < <(cd "$WORK/base" && find ir -name '*.mlir' | sort)
echo "  $n IR stage dumps compared"
if [ $fail -eq 0 ]; then
    echo "[roundtrip] PASS"
    rm -rf "$WORK"
else
    echo "[roundtrip] FAIL (kept $WORK)"
fi
exit $fail
