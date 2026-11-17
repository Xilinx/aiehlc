#!/usr/bin/env bash
# Routing IR re-entry round trip (doc/design/routing_ir_reentry.md).
#
#   script/test/routing_ir_roundtrip.sh [--stage routing|dmaphop] <src.cc> [aiehlc.sh args...]
#
# 1. builds <src.cc> normally (baseline),
# 2. builds it again feeding the baseline's IR of that stage back unchanged:
#    routing (default): AIEHLC_ROUTING_IR_IN = ir/dfschedule/0_initial.mlir
#    dmaphop:           AIEHLC_DMAPHOP_IR_IN = ir/dfschedule/3_DmapToDmaphopPass.mlir
# 3. requires every generated file and every dumped IR stage to be byte-identical.
# Exit 0 = that IR carries everything the rest of the pipeline needs for this source.
# Default aiehlc.sh args: --aie-version 5. Work dirs under ${TMPDIR:-/tmp}.
set -uo pipefail
STAGE=routing
if [ "${1:-}" = "--stage" ]; then
    STAGE="${2:?--stage routing|dmaphop}"
    shift 2
fi
case "$STAGE" in
routing) ENV=AIEHLC_ROUTING_IR_IN IRFILE=0_initial.mlir ;;
dmaphop) ENV=AIEHLC_DMAPHOP_IR_IN IRFILE=3_DmapToDmaphopPass.mlir ;;
*) echo "unknown --stage $STAGE (routing|dmaphop)"; exit 2 ;;
esac
SRC="$(realpath "${1:?usage: $0 [--stage routing|dmaphop] <src.cc> [aiehlc.sh args...]}")"
shift
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--aie-version 5)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/routing_ir_rt.XXXXXX")"
build_in() {
    mkdir -p "$1"
    (cd "$1" && bash -c "source '$REPO/script/aiehlc.sh' ${ARGS[*]} --runtime-source-file '$SRC'" > build.log 2>&1)
}
echo "[roundtrip] $SRC, $STAGE IR (work dir $WORK)"
build_in "$WORK/base" || { echo "[roundtrip] FAIL: baseline build failed ($WORK/base/build.log)"; exit 1; }
IR="$WORK/base/ir/dfschedule/$IRFILE"
[ -f "$IR" ] || { echo "[roundtrip] FAIL: no $IR"; exit 1; }
cp "$IR" "$WORK/ir_in.mlir"
export "$ENV=$WORK/ir_in.mlir"
build_in "$WORK/reentry" || { echo "[roundtrip] FAIL: re-entry build failed ($WORK/reentry/build.log)"; exit 1; }
unset "$ENV"
grep -h "$ENV" "$WORK/reentry/build.log" | head -3
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
