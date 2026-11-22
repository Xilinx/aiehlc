#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# check_refs.sh — verify code references in a tutorial markdown file against a
# FIXED base commit (never HEAD / the working tree).
#
# Usage: script/check_refs.sh [--base REV] [--min N] [--require PATH]... FILE.md
#
# Rules (each violation prints a "MISSING ..." line):
#   1. every backticked repo path (`dir/file`, `dir/file:N`, `dir/file:N-M`)
#      must exist in the base commit;
#   2. <!-- ref PATH:L[-M] TOKEN --> passes only if TOKEN (fixed string)
#      occurs on lines L..M of PATH in the base commit;
#   3. <!-- sym TOKEN --> passes only if `git grep -F` finds TOKEN in the base
#      commit under src/ include/ script/ example/;
#   4. a backticked `path:N[-M]` needs a ref comment on the same PATH whose
#      line range covers N..M;
#   5. fewer than N ref+sym comments (default 1) is a failure;
#   6. each --require PATH must be cited by at least one ref comment.
# Success: prints "ALL REFS FOUND (base <sha>, N refs)" and exits 0.
# Failure: prints MISSING lines, then "REFS FAILED (N refs)", exits 1.
###############################################################################
set -u

BASE_REV=43fcb8f
MIN=1
REQUIRES=()
FILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --base)    BASE_REV="$2"; shift 2 ;;
        --min)     MIN="$2"; shift 2 ;;
        --require) REQUIRES+=("$2"); shift 2 ;;
        -h|--help) sed -n '7,24p' "$0"; exit 0 ;;
        -*)        echo "MISSING unknown option $1"; echo "REFS FAILED (0 refs)"; exit 1 ;;
        *)         FILE="$1"; shift ;;
    esac
done

fail=0
nrefs=0
miss() { echo "MISSING $*"; fail=1; }
finish() {
    if [ "$fail" -eq 0 ] && [ "$nrefs" -ge 1 ]; then
        echo "ALL REFS FOUND (base $BASE, $nrefs refs)"; exit 0
    fi
    echo "REFS FAILED ($nrefs refs)"; exit 1
}

if ! [[ "$MIN" =~ ^[0-9]+$ ]]; then miss "bad --min '$MIN'"; BASE=?; finish; fi
[ "$MIN" -lt 1 ] && MIN=1   # 0 refs is never a pass

# Run git in the repo that contains this script.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
g() { git -C "$REPO" "$@"; }
BASE="$(g rev-parse --verify -q "${BASE_REV}^{commit}")" || { BASE=?; miss "base commit $BASE_REV"; finish; }

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then miss "input file '${FILE}'"; finish; fi

exists() { g cat-file -e "$BASE:${1%/}" 2>/dev/null; }

# ---- ref / sym comments --------------------------------------------------
declare -a REF_PATH=() REF_L=() REF_M=()
while IFS= read -r c; do
    body="${c#<!--}"; body="${body%-->}"
    read -r kind spec rest <<<"$body"
    case "$kind" in
    ref)
        nrefs=$((nrefs + 1))
        tok="$(sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' <<<"$rest")"
        if ! [[ "$spec" =~ ^([^:]+):([0-9]+)(-([0-9]+))?$ ]]; then
            miss "ref malformed: $c"; continue
        fi
        p="${BASH_REMATCH[1]}"; l="${BASH_REMATCH[2]}"; m="${BASH_REMATCH[4]:-$l}"
        REF_PATH+=("$p"); REF_L+=("$l"); REF_M+=("$m")
        if [ -z "$tok" ]; then miss "ref $spec: empty token"; continue; fi
        if [ "$m" -lt "$l" ]; then miss "ref $spec: bad range"; continue; fi
        if ! exists "$p"; then miss "ref path $p (not in $BASE)"; continue; fi
        if ! g show "$BASE:$p" | sed -n "${l},${m}p" | grep -qF -- "$tok"; then
            miss "ref $spec: token '$tok' not on lines $l-$m"
        fi
        ;;
    sym)
        nrefs=$((nrefs + 1))
        tok="$(sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' <<<"$spec $rest")"
        if [ -z "$tok" ]; then miss "sym: empty token"; continue; fi
        if ! g grep -q -F -e "$tok" "$BASE" -- src include script example 2>/dev/null; then
            miss "sym '$tok' (not under src/ include/ script/ example/ in $BASE)"
        fi
        ;;
    esac
done < <(grep -oE '<!--[[:space:]]*(ref|sym)[[:space:]][^>]*-->' "$FILE")

# ---- backticked paths ----------------------------------------------------
# A repo path: no leading '/', at least one '/', path chars only, optional :N[-M].
while IFS= read -r bt; do
    t="${bt#\`}"; t="${t%\`}"
    [[ "$t" =~ ^([A-Za-z0-9_.][A-Za-z0-9_.+-]*(/[A-Za-z0-9_.+-]+)+/?)(:([0-9]+)(-([0-9]+))?)?$ ]] || continue
    p="${BASH_REMATCH[1]}"; n="${BASH_REMATCH[4]}"; nm="${BASH_REMATCH[6]:-${BASH_REMATCH[4]}}"
    if ! exists "$p"; then miss "path $p (not in $BASE)"; continue; fi
    if [ -n "$n" ]; then
        ok=0
        for i in "${!REF_PATH[@]}"; do
            if [ "${REF_PATH[$i]}" = "$p" ] && [ "${REF_L[$i]}" -le "$n" ] && [ "$nm" -le "${REF_M[$i]}" ]; then ok=1; break; fi
        done
        [ "$ok" -eq 1 ] || miss "uncited line ref \`$t\` (no <!-- ref $p:$n... TOKEN --> covering it)"
    fi
done < <(grep -oE '`[^`[:space:]]+`' "$FILE")

# ---- counts / required files ---------------------------------------------
[ "$nrefs" -ge "$MIN" ] || miss "refs: $nrefs < --min $MIN"
for r in "${REQUIRES[@]+"${REQUIRES[@]}"}"; do
    found=0
    for p in "${REF_PATH[@]+"${REF_PATH[@]}"}"; do [ "$p" = "$r" ] && { found=1; break; }; done
    [ "$found" -eq 1 ] || miss "required file $r is not cited by any ref"
done

finish
