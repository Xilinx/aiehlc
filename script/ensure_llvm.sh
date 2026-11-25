#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Find an LLVM 24+ build, or build the pinned one with script/build_llvm.sh.
#   bash script/ensure_llvm.sh [--out FILE] [--check-only] [--jobs N]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_MIN=24
OUT=""
CHECK_ONLY="${AIEHLC_NO_LLVM_BUILD:-0}"
JOBS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --jobs) JOBS=(--jobs "$2"); shift 2 ;;
        *) echo "usage: $0 [--out FILE] [--check-only] [--jobs N]" >&2; exit 2 ;;
    esac
done

llvm_major() { "$1/bin/llvm-config" --version 2>/dev/null | cut -d. -f1; }
usable() {
    local d="${1%/}"
    [ -n "$d" ] && [ -x "$d/bin/llvm-config" ] && [ -x "$d/bin/mlir-tblgen" ] &&
        [ "$(llvm_major "$d")" -ge "$LLVM_MIN" ] 2>/dev/null
}

user="${USER:-$(id -un)}"
cands=("${LLVM_INSTALL_DIR:-}" "${AIEHLC_LLVM_BUILD:-}")
[ -n "${AIEHLC_LLVM_SRC:-}" ] && cands+=("${AIEHLC_LLVM_SRC}/build")
cands+=("/scratch/staff/${user}/llvm-project/build" "${ROOT}/thirdparty/llvm-project/build")

found=""
for c in "${cands[@]}"; do
    [ -n "$c" ] || continue
    if usable "$c"; then
        found="${c%/}"
        break
    fi
    if [ -x "${c%/}/bin/llvm-config" ]; then
        echo "[ensure_llvm] skipping ${c}: LLVM $("${c%/}/bin/llvm-config" --version), aiehlc needs ${LLVM_MIN}+"
    fi
done

if [ -z "$found" ]; then
    if [ "$CHECK_ONLY" = "1" ]; then
        echo "[ensure_llvm] no LLVM ${LLVM_MIN}+ build found; build one with: bash script/build_llvm.sh" >&2
        exit 1
    fi
    echo "[ensure_llvm] no LLVM ${LLVM_MIN}+ build found: building the pinned LLVM (script/build_llvm.sh)."
    echo "[ensure_llvm] this clones llvm-project and takes a while (about an hour on 32 cores)."
    bash "${ROOT}/script/build_llvm.sh" "${JOBS[@]}"
    src="${AIEHLC_LLVM_SRC:-}"
    if [ -z "$src" ]; then
        if [ -d "/scratch/staff/${user}" ]; then src="/scratch/staff/${user}/llvm-project"; else src="${ROOT}/thirdparty/llvm-project"; fi
    fi
    found="${src}/build"
    usable "$found" || { echo "[ensure_llvm] build finished but ${found} is not a usable LLVM ${LLVM_MIN}+" >&2; exit 1; }
fi

echo "[ensure_llvm] LLVM ${found} ($("${found}/bin/llvm-config" --version))"
[ -n "$OUT" ] && printf '%s\n' "$found" > "$OUT"
exit 0
