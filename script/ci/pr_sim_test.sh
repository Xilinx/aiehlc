#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

set -euo pipefail

mode="${1:-}"
log="ci-sim.log"
public="ci-public.log"
: >"$log"
rm -f "$public"

redact() {
    local ws="${GITHUB_WORKSPACE:-}"
    if [ -n "${ws}" ]; then
        sed -E \
            -e "s|${ws}|.|g" \
            -e 's|/scratch/staff/[^[:space:]]+|<user>|g' \
            -e 's|/home/[^[:space:]]+|<user>|g' \
            -e 's|/Users/[^[:space:]]+|<user>|g' \
            -e 's|/proj/[^[:space:]]+|<tools>|g' \
            -e 's#(https?://)[^/@[:space:]]+@#\1***@#g'
    else
        sed -E \
            -e 's|/scratch/staff/[^[:space:]]+|<user>|g' \
            -e 's|/home/[^[:space:]]+|<user>|g' \
            -e 's|/Users/[^[:space:]]+|<user>|g' \
            -e 's|/proj/[^[:space:]]+|<tools>|g' \
            -e 's#(https?://)[^/@[:space:]]+@#\1***@#g'
    fi
}

finish_fail() {
    trap - EXIT
    {
        echo "FAIL: ${label:-pr_sim_test.sh ${mode}}"
        excerpt="$(grep -E 'Mismatch|Failure|failed|error:|Error|ERROR|fatal:|403|forbidden|CMake Error|unbound|undefined|CRITICAL|axi_mm|Sim result|PASS:|passed' "$log" | tail -n 30 || true)"
        if [ -n "${excerpt}" ]; then
            printf '%s\n' "${excerpt}" | redact
        else
            tail -n 20 "$log" | redact
        fi
    } | tee "$public"
    exit 1
}

case "$mode" in
    tutorial)
        label="single-kernel tutorial/example.cpp"
        sim_dir="aout"
        src="tutorial/example.cpp"
        tiles="4:4"
        markers=(
            "Sucess: CPU result matches AIE."
            "Kernel test passed!"
            "Simulation Finished, Sim result: 0"
        )
        ;;
    passthrough)
        label="tilinglinalg passthrough_sim.cc"
        sim_dir="aout/worklocal"
        src="example/tileprogram/ccode/passthrough_sim.cc"
        tiles="0:3"
        markers=(
            "[passthrough] PASS:"
            "Kernel test passed!"
            "Simulation Finished, Sim result: 0"
        )
        ;;
    *)
        echo "usage: pr_sim_test.sh tutorial|passthrough"
        exit 2
        ;;
esac

shift
on_exit() {
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        finish_fail
    fi
}
trap on_exit EXIT

root="$(pwd)"
# The tree builds against LLVM 24 (script/llvm_env.sh, script/build_llvm.sh): its .td files use
# TableGen options an older mlir-tblgen rejects ("Value 'useStrictPropertiesInAssemblyFormat' unknown!").
# Take LLVM_INSTALL_DIR if set, else llvm_env.sh's build, else the shared fallback -- the first one that is
# LLVM >= 24.
llvm_min=24
llvm_major() { "$1/bin/llvm-config" --version 2>/dev/null | cut -d. -f1; }
ci_user="${USER:-$(id -un)}"
llvm_dir=""
for cand in "${LLVM_INSTALL_DIR:-}" "${AIEHLC_LLVM_BUILD:-}" \
            "${AIEHLC_LLVM_SRC:-/scratch/staff/${ci_user}/llvm-project}/build" \
            "/scratch/staff/huaj/llvm-project/build"; do
    [ -n "${cand}" ] && [[ "${cand}" != /Users/* ]] && [ -x "${cand}/bin/llvm-config" ] || continue
    if [ "$(llvm_major "${cand}")" -ge "${llvm_min}" ] 2>/dev/null; then
        llvm_dir="${cand}"
        break
    fi
    echo "skipping LLVM ${cand}: $("${cand}/bin/llvm-config" --version), this tree needs ${llvm_min}+"
done
if [ -z "${llvm_dir}" ]; then
    echo "No LLVM ${llvm_min}+ build found: set LLVM_INSTALL_DIR (script/build_llvm.sh builds one)."
    exit 1
fi
echo "LLVM: ${llvm_dir} ($("${llvm_dir}/bin/llvm-config" --version))"

source script/setup.sh >>"$log" 2>&1
export LLVM_INSTALL_DIR="${llvm_dir}"
cd "${root}"

cmake -S . -B build -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}" -DCMAKE_BUILD_TYPE=Release -Wno-dev >>"$log" 2>&1
cmake --build build -j"$(nproc)" --target aiehlc >>"$log" 2>&1

source script/aiehlc.sh \
    --platform sim --aie-version 5 --sim-tiles "${tiles}" \
    --runtime-source-file "${src}" >>"$log" 2>&1
cd "${root}"

export AIEHLC_DBG_HOLD_SEC=0
set +e
bash script/runsim.sh "${sim_dir}" >>"$log" 2>&1
sim_rc=$?
set -e

missing=0
for marker in "${markers[@]}"; do
    if ! grep -q -F "${marker}" "$log"; then
        missing=1
    fi
done

if [ "${missing}" -ne 0 ] || [ "${sim_rc}" -ne 0 ]; then
    finish_fail
fi

{
    echo "PASS: ${label}"
    printf '%s\n' "${markers[@]}"
} | tee "$public"
exit 0
