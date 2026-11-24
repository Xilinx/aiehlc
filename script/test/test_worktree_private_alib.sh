#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Self-test for `script/worktree.sh link-deps --private-alib`
# (GAP-aiehlc-build-61a37e / d69027, GAP-aiehlc-script-aiehlc.sh-f2e008: the shared
# thirdparty/alib was rebuilt in place by concurrent aiehlc.sh runs).
# Usage: bash script/test/test_worktree_private_alib.sh [<ref-worktree>]
#        (default ref: the repo's main worktree, first entry of `git worktree list`)
# Creates a detached temp worktree and runs link-deps:
#   run 0: plain (alib becomes a symlink to the shared alib),
#   run 1: --private-alib (the symlink is replaced by a private cp -a copy),
#   run 2: --private-alib again (must keep the private dir: same inode, marker kept).
# Checks the private alib is a real dir with the aie-rt headers and the same file
# count as the shared one, the other deps are still links, and the shared alib's
# include/ mtime and file count did not change. Removes the temp worktree and prints:
#   WT_PRIVATE_ALIB real_dir=<0|1> headers=<0|1> idempotent=<0|1> PASS|FAIL
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_SH="${SCRIPT_DIR}/../worktree.sh"
REPO="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
REF="${1:-$(git -C "${REPO}" worktree list --porcelain | awk '/^worktree /{print substr($0,10); exit}')}"
TMP="$(mktemp -d /tmp/wt_private_alib_test.XXXXXX)"
WT="${TMP}/wt"
cleanup() { git -C "${REPO}" worktree remove --force "${WT}" >/dev/null 2>&1; rm -rf "${TMP}"; git -C "${REPO}" worktree prune; }
trap cleanup EXIT

SHARED="$(readlink -f "${REF}/thirdparty/alib")"
[ -d "${SHARED}/include" ] || { echo "WT_PRIVATE_ALIB real_dir=0 headers=0 idempotent=0 FAIL (no shared alib in ${REF})"; exit 1; }
echo "ref=${REF} shared_alib=${SHARED}"
SH_MT0="$(stat -c %Y "${SHARED}/include")"; SH_N0="$(find "${SHARED}" | wc -l)"

git -C "${REPO}" worktree add -q --detach "${WT}" HEAD || { echo "WT_PRIVATE_ALIB real_dir=0 headers=0 idempotent=0 FAIL (worktree add)"; exit 1; }
A="${WT}/thirdparty/alib"

echo "== run 0 (plain link-deps)"; bash "${WT_SH}" link-deps "${WT}" --from "${REF}"; rc0=$?
was_link=0; [ -L "${A}" ] && was_link=1
echo "== run 1 (--private-alib)"; bash "${WT_SH}" link-deps "${WT}" --from "${REF}" --private-alib; rc1=$?
ino1="$(stat -c %i "${A}" 2>/dev/null)"; touch "${A}/.private_alib_marker"
echo "== run 2 (--private-alib again)"; bash "${WT_SH}" link-deps "${WT}" --from "${REF}" --private-alib; rc2=$?
ino2="$(stat -c %i "${A}" 2>/dev/null)"

real_dir=0; [ -d "${A}" ] && [ ! -L "${A}" ] && [ "$(readlink -f "${A}")" = "$(cd "${A}" && pwd -P)" ] && real_dir=1
headers=0; n_priv="$(find "${A}" ! -name .private_alib_marker | wc -l)"
[ -f "${A}/include/xaiengine/xaiegbl.h" ] && [ -f "${A}/aie-rt/driver/src/global/xaiegbl.h" ] && [ "${n_priv}" -eq "${SH_N0}" ] && headers=1
others=0
for d in thirdparty/tvm-0.16 build/aiehlc thirdparty/arch/cortexa78_0/workspace; do
    if [ -L "${WT}/$d" ] && [ -e "${WT}/$d" ]; then others=$((others+1)); else echo "MISS $d"; fi
done
idem=0
[ "${rc0}" -eq 0 ] && [ "${rc1}" -eq 0 ] && [ "${rc2}" -eq 0 ] && [ -n "${ino1}" ] && [ "${ino1}" = "${ino2}" ] \
    && [ -f "${A}/.private_alib_marker" ] && idem=1
SH_MT1="$(stat -c %Y "${SHARED}/include")"; SH_N1="$(find "${SHARED}" | wc -l)"
shared_ok=0; [ "${SH_MT0}" = "${SH_MT1}" ] && [ "${SH_N0}" = "${SH_N1}" ] && shared_ok=1
echo "rc0=${rc0} rc1=${rc1} rc2=${rc2} was_link=${was_link} ino1=${ino1} ino2=${ino2} files_priv=${n_priv} files_shared=${SH_N0} other_links=${others} shared_untouched=${shared_ok}"
if [ "${real_dir}" -eq 1 ] && [ "${headers}" -eq 1 ] && [ "${idem}" -eq 1 ] && [ "${was_link}" -eq 1 ] \
   && [ "${others}" -eq 3 ] && [ "${shared_ok}" -eq 1 ]; then
    echo "WT_PRIVATE_ALIB real_dir=${real_dir} headers=${headers} idempotent=${idem} PASS"; exit 0
fi
echo "WT_PRIVATE_ALIB real_dir=${real_dir} headers=${headers} idempotent=${idem} FAIL"; exit 1
