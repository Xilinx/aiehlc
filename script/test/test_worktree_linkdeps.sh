#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Self-test for `script/worktree.sh link-deps` (GAP-aiehlc-worktree-linkdeps).
# Usage: bash script/test/test_worktree_linkdeps.sh [<ref-worktree>]
#        (default ref: the repo's main worktree, first entry of `git worktree list`)
# Creates a detached temp worktree, runs link-deps twice, checks that
# thirdparty/alib, thirdparty/tvm-0.16, build/aiehlc and the A78 BSP
# workspace (thirdparty/arch/cortexa78_0/workspace) resolve, removes the
# temp worktree and prints: WT_LINKDEPS links=<n> idempotent=<0|1> PASS|FAIL
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_SH="${SCRIPT_DIR}/../worktree.sh"
REPO="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REF="${1:-$(git -C "${REPO}" worktree list --porcelain | awk '/^worktree /{print substr($0,10); exit}')}"
echo "ref=${REF}"
TMP="$(mktemp -d /tmp/wt_linkdeps_test.XXXXXX)"
WT="${TMP}/wt"
cleanup() { git -C "${REPO}" worktree remove --force "${WT}" >/dev/null 2>&1; rm -rf "${TMP}"; git -C "${REPO}" worktree prune; }
trap cleanup EXIT

git -C "${REPO}" worktree add -q --detach "${WT}" HEAD || { echo "WT_LINKDEPS links=0 idempotent=0 FAIL (worktree add)"; exit 1; }

echo "== run 1"; bash "${WT_SH}" link-deps "${WT}" --from "${REF}"; rc1=$?
snap1="$(for d in thirdparty/alib thirdparty/tvm-0.16 build/aiehlc thirdparty/arch/cortexa78_0/workspace; do echo "$d $(readlink "${WT}/$d")"; done)"
echo "== run 2"; bash "${WT_SH}" link-deps "${WT}" --from "${REF}"; rc2=$?
snap2="$(for d in thirdparty/alib thirdparty/tvm-0.16 build/aiehlc thirdparty/arch/cortexa78_0/workspace; do echo "$d $(readlink "${WT}/$d")"; done)"

links=0
for d in thirdparty/alib thirdparty/tvm-0.16 thirdparty/arch/cortexa78_0/workspace; do
    if [ -L "${WT}/$d" ] && ls -d "${WT}/$d/" >/dev/null 2>&1; then links=$((links+1)); else echo "MISS $d"; fi
done
if [ -L "${WT}/build/aiehlc" ] && [ -x "${WT}/build/aiehlc" ]; then links=$((links+1)); else echo "MISS build/aiehlc"; fi
echo "${snap2}"

idem=0
[ "${rc1}" -eq 0 ] && [ "${rc2}" -eq 0 ] && [ "${snap1}" = "${snap2}" ] && idem=1
echo "rc1=${rc1} rc2=${rc2}"
if [ "${links}" -eq 4 ] && [ "${idem}" -eq 1 ]; then
    echo "WT_LINKDEPS links=${links} idempotent=${idem} PASS"; exit 0
fi
echo "WT_LINKDEPS links=${links} idempotent=${idem} FAIL"; exit 1
