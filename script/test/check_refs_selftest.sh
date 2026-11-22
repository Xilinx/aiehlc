#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Self-test for script/check_refs.sh: 1 good md must pass, 6 bad mds (one per
# rule + zero refs) must be rejected. Prints "SELFTEST PASS good=1 bad_rejected=6".
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHK="$HERE/../check_refs.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
RT=src/mlir/runtime/aie_runtime.c
ARGS=(--min 3 --require "$RT")

GOOD='# good
Copyright header in `script/aiehlc.sh`. <!-- ref script/aiehlc.sh:1-5 SPDX-License-Identifier -->
S2MM channel enable at `'"$RT"':1339`. <!-- ref '"$RT"':1339 XAie_DmaChannelEnable -->
<!-- sym XAie_DmaChannelEnable -->'
printf '%s\n' "$GOOD" > "$T/good.md"

# bad 1: backticked path that does not exist
printf '%s\n' "$GOOD" 'See `src/no_such_dir/nope.c`.' > "$T/bad_path.md"
# bad 2: symbol that does not exist
printf '%s\n' "$GOOD" '<!-- sym XAie_NoSuchSymbol_tut07 -->' > "$T/bad_sym.md"
# bad 3: token not on the cited lines
printf '%s\n' "$GOOD" '<!-- ref script/aiehlc.sh:1-5 XAie_DmaChannelEnable -->' > "$T/bad_token.md"
# bad 4: backticked path:N with no matching ref comment
printf '%s\n' "$GOOD" 'Also `script/aiehlc.sh:600`.' > "$T/bad_uncited.md"
# bad 5: zero refs (checked with default --min)
printf '%s\n' '# nothing' 'Only `script/aiehlc.sh`.' > "$T/bad_zero.md"
# bad 6: required file never cited
printf '%s\n' "$GOOD" > "$T/bad_require.md"

good=0; bad=0; rc=0
if out="$(bash "$CHK" "${ARGS[@]}" "$T/good.md")" && grep -q '^ALL REFS FOUND (base [0-9a-f]\{40\}, 3 refs)$' <<<"$out"; then
    good=1
else
    echo "FAIL good.md:"; echo "$out"; rc=1
fi
reject() { # name, args...
    local n="$1"; shift
    local out; out="$(bash "$CHK" "$@" 2>&1)"; local r=$?
    if [ $r -ne 0 ] && ! grep -q 'ALL REFS FOUND' <<<"$out" && grep -q '^MISSING ' <<<"$out" && grep -q '^REFS FAILED' <<<"$out"; then
        bad=$((bad + 1))
    else
        echo "FAIL $n was not rejected (rc=$r):"; echo "$out"; rc=1
    fi
}
reject bad_path    "${ARGS[@]}" "$T/bad_path.md"
reject bad_sym     "${ARGS[@]}" "$T/bad_sym.md"
reject bad_token   "${ARGS[@]}" "$T/bad_token.md"
reject bad_uncited "${ARGS[@]}" "$T/bad_uncited.md"
reject bad_zero    "$T/bad_zero.md"
reject bad_require --min 3 --require script/hostcompile.sh "$T/bad_require.md"

if [ $rc -eq 0 ] && [ $good -eq 1 ] && [ $bad -eq 6 ]; then
    echo "SELFTEST PASS good=$good bad_rejected=$bad"; exit 0
fi
echo "SELFTEST FAIL good=$good bad_rejected=$bad"; exit 1
