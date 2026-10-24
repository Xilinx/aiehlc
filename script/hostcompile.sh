#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Compile generated host.cc with aie_runtime and Xilinx AIE libs.
# Uses the same env and link recipe as script/aiehlc.sh (baremetal).
# Usage: run from worklocal/ or: ./hostcompile.sh [rebuild|-rebuild] [--aie-version 2] [--platform baremetal]
#   rebuild/-rebuild  - recompile all *.cc files even if .o files already exist (default: skip if .o exists)
###############################################################################

set -e

# Parse arguments: accept 'rebuild' or '-rebuild' anywhere to force recompile of *.cc files
REBUILD=1
PASSTHROUGH_ARGS=()
#for arg in "$@"; do
#    if [[ "$arg" == "rebuild" || "$arg" == "-rebuild" ]]; then
#        REBUILD=1
#    else
#        PASSTHROUGH_ARGS+=("$arg")
#    fi
#done
#set -- "${PASSTHROUGH_ARGS[@]}"

# Repo root: this script always lives in script/ -> one level up.
AIEHLC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIEHLC_DIR="${AIEHLC_ROOT}"
# WORKLOCAL_DIR comes from the caller. Generated host.cc/kernel.cc and all build
# artifacts live here. When not explicitly set, prefer the current dir if it holds
# the generated sources, otherwise fall back to the aiehlc default aout/worklocal.
if [ -z "${WORKLOCAL_DIR:-}" ]; then
    if [ -f "$(pwd)/host.cc" ] || [ -f "$(pwd)/kernel.cc" ]; then
        WORKLOCAL_DIR="$(pwd)"
    elif [ -f "$(pwd)/aout/worklocal/host.cc" ] || [ -f "$(pwd)/aout/worklocal/kernel.cc" ]; then
        WORKLOCAL_DIR="$(pwd)/aout/worklocal"
        echo "[hostcompile] WORKLOCAL_DIR not set; using detected ${WORKLOCAL_DIR}"
    else
        WORKLOCAL_DIR="$(pwd)"
    fi
fi
BUILD_DIR="${WORKLOCAL_DIR}/build"
_HOSTCOMPILE_ORIG_PWD="$(pwd)"

OPT_FLAGS="-Os"
[ "${DEBUG_SYMS:-0}" -eq 1 ] && OPT_FLAGS="${OPT_FLAGS} -g"

KERNEL_FUNC_NAME="computekernel"   # default fallback; overridden by auto-detection below

compile_one_kernel() {
    local kernel_func_name="${1:-compute_kernel}"

    if [ ! -f "${WORKLOCAL_DIR}/kernel.cc" ]; then
        echo "Error: kernel.cc not found in ${WORKLOCAL_DIR}"
        exit 1
    fi

    if [ -f "${AIEHLC_ROOT}/script/setup.sh" ]; then
        local saved_pwd
        saved_pwd="$(pwd)"
        source "${AIEHLC_ROOT}/script/setup.sh" --path-set-only
        cd "${saved_pwd}"
    fi
    if [ -z "$XILINX_VITIS" ]; then
        echo "Error: XILINX_VITIS environment variable not set"
        exit 1
    fi

    rm -f "${BUILD_DIR}"/chesswork/kernel* "${BUILD_DIR}"/kernel.o \
          "${BUILD_DIR}"/kernel 2>/dev/null || true

    cd "${WORKLOCAL_DIR}"

    if [[ "${USE_LLVM_AIE:-false}" == "true" ]] && [[ -n "${LLVM_AIE_PATH:-}" ]] \
       && [[ -x "${LLVM_AIE_PATH}/bin/clang++" ]]; then
        echo "[hostcompile] Using llvm-aie (${LLVM_AIE_PATH})"
        source "${AIEHLC_ROOT}/script/kc.sh" \
            --kernel-cc ./kernel.cc \
            --func-name "${kernel_func_name}" \
            --aie-version "${AIE_VERSION:-5}" \
            --platform "${PLATFORM:-baremetal}" \
            --debug-output \
            --output-dir "${BUILD_DIR}" \
            --use-llvm-aie \
            --ld-script "${WORKLOCAL_DIR}/main.ld.script"
    else
        local prx_file
        if [ -f "${WORKLOCAL_DIR}/aieml.prx" ]; then
            prx_file="${WORKLOCAL_DIR}/aieml.prx"
        else
            prx_file="aie2ps.prx"
        fi
        source "${AIEHLC_ROOT}/script/kc.sh" \
            --kernel-cc ./kernel.cc \
            --func-name "${kernel_func_name}" \
            --aie-version "${AIE_VERSION:-5}" \
            --platform "${PLATFORM:-baremetal}" \
            --debug-output \
            --output-dir "${BUILD_DIR}" \
            --prx "${prx_file}"
    fi
}

if [ "${KERNEL_ONLY:-0}" = "1" ]; then
    compile_one_kernel "${1:-compute_kernel}"
    _kret=$?
    cd "${_HOSTCOMPILE_ORIG_PWD}" 2>/dev/null || true
    return $_kret 2>/dev/null || exit $_kret
fi

# Detect multi-kernel mode: check for kernel_<name>.cc files
MULTI_KERNEL_FILES=()
for f in "${WORKLOCAL_DIR}"/kernel_*.cc; do
    [ -f "$f" ] && MULTI_KERNEL_FILES+=("$f")
done

KERNEL_OBJ_LIST=""
if [ ${#MULTI_KERNEL_FILES[@]} -gt 0 ]; then
    echo "[Multi-kernel] Detected ${#MULTI_KERNEL_FILES[@]} kernel(s)"
    pushd ${WORKLOCAL_DIR}
    for kf in "${MULTI_KERNEL_FILES[@]}"; do
        kfname="$(basename "$kf")"
        # Extract kernel name: kernel_<name>.cc -> <name>
        kname="${kfname#kernel_}"
        kname="${kname%.cc}"
        echo "[Multi-kernel] Compiling kernel: ${kname} (${kfname})"

        # Use per-kernel PRX if available, fall back to aieml.prx
        KPRX=""
        if [ -f "${WORKLOCAL_DIR}/aieml_${kname}.prx" ]; then
            KPRX="${WORKLOCAL_DIR}/aieml_${kname}.prx"
        elif [ -f "${WORKLOCAL_DIR}/aieml.prx" ]; then
            KPRX="${WORKLOCAL_DIR}/aieml.prx"
        fi

        # compile_one_kernel builds kernel.cc — for multi-kernel, temporarily symlink
        ln -sf "${kfname}" "${WORKLOCAL_DIR}/kernel.cc"
        # (no per-kernel PRX: aieml.prx is used as is -- linking it onto itself fails)
        if [ -n "$KPRX" ] && [ "$KPRX" != "${WORKLOCAL_DIR}/aieml.prx" ]; then
            ln -sf "$(basename "$KPRX")" "${WORKLOCAL_DIR}/aieml.prx"
        fi

        compile_one_kernel "${kname}"
        if [ $? -ne 0 ]; then
            echo "Error: kernel build failed for kernel ${kname}"
            exit 1
        fi
        # Rename kernel.o to kernel_<name>.o to avoid overwriting
        mv -f "${BUILD_DIR}/kernel.o" "${BUILD_DIR}/kernel_${kname}.o"
        echo "✓ Compiled kernel: ${BUILD_DIR}/kernel_${kname}.o (func: ${kname})"
        KERNEL_OBJ_LIST="${KERNEL_OBJ_LIST} ${BUILD_DIR}/kernel_${kname}.o"
    done
    # Remove temporary symlink
    rm -f "${WORKLOCAL_DIR}/kernel.cc"
    popd
elif [ ! -f "${WORKLOCAL_DIR}/kernel.cc" ]; then
    # Host-only mode: no kernel_<name>.cc and no kernel.cc. This is a legitimate
    # build (e.g. the TVM frontend with AIE offload disabled — every op is
    # force-offloaded to the CPU backend, so there is nothing for xchesscc to
    # compile), not a missing-file error. Link host.cc with no embedded kernel.
    echo "[Host-only] No kernel sources found; linking host.cc without an AIE kernel"
    KERNEL_OBJ_LIST=""
else
    # Single-kernel mode (backward compat)
    # Auto-detect kernel function name from kernel.cc's "// kernel_decl <name>" comment
    if [ -f "${WORKLOCAL_DIR}/kernel.cc" ]; then
        DETECTED_NAME=$(grep -oP '// kernel_decl \K\S+' "${WORKLOCAL_DIR}/kernel.cc" | head -1)
        if [ -n "$DETECTED_NAME" ]; then
            KERNEL_FUNC_NAME="$DETECTED_NAME"
            echo "[Single-kernel] Detected kernel function name: ${KERNEL_FUNC_NAME}"
        fi
    fi
    pushd ${WORKLOCAL_DIR}
    compile_one_kernel "${KERNEL_FUNC_NAME}"
    if [ $? -ne 0 ]; then
        echo "Error: kernel build failed"
        exit 1
    fi
    echo "✓ Compiled kernel: ${BUILD_DIR}/kernel.o (func: ${KERNEL_FUNC_NAME})"
    KERNEL_OBJ_LIST="${BUILD_DIR}/kernel.o"
    popd
fi

# Defaults (match aiehlc.sh)
aie_version="${AIE_VERSION:-5}"
platform="${PLATFORM:-baremetal}"

if [ ! -f "${WORKLOCAL_DIR}/host.cc" ]; then
    echo "Error: host.cc not found in ${WORKLOCAL_DIR}"
    echo "Run the unitest (e.g. ./test dfschedule) to generate host.cc first."
    exit 1
fi

# Check for optional routing.cc (generated by routingtoroutinghw)
HAS_ROUTING=0
if [ -f "${WORKLOCAL_DIR}/routing.cc" ]; then
    HAS_ROUTING=1
    echo "Found routing.cc – will compile and link into host app."
fi

# Source env (XILINX_VITIS, etc.) same as aiehlc.sh
# Export PATH_SET_ONLY so setup.sh's first test does not see unset (avoids "integer expression expected")
if [ -f "${AIEHLC_ROOT}/script/setup.sh" ]; then
    export PATH_SET_ONLY="${PATH_SET_ONLY:-1}"
    source "${AIEHLC_ROOT}/script/setup.sh" --path-set-only
fi
# Restore AIEHLC_DIR after setup.sh (setup.sh overwrites it relative to script/)
AIEHLC_DIR="${AIEHLC_ROOT}"

if [ -z "${XILINX_VITIS}" ]; then
    echo "Warning: XILINX_VITIS not set. Link may fail (undefined XAie_*)."
    echo "Source script/setup.sh from repo root or set XILINX_VITIS."
fi

# Toolchain: same as aiehlc.sh (baremetal -> aarch64-none-elf-g++/gcc, linux -> aarch64-linux-gnu-).
# Override with CROSS_COMPILE (e.g. CROSS_COMPILE=aarch64-none-elf- ./hostcompile.sh) when another toolchain is first in PATH.
if [[ -n "${CROSS_COMPILE}" ]]; then
    TOOL_PREFIX="${CROSS_COMPILE}"
    [[ "${TOOL_PREFIX}" != *"-" ]] && TOOL_PREFIX="${TOOL_PREFIX}-"
elif [[ "$platform" == "linux" ]]; then
    TOOL_PREFIX="aarch64-linux-gnu-"
elif [[ "$platform" == "baremetal" ]]; then
    TOOL_PREFIX="aarch64-none-elf-"
elif [[ "$platform" == "sim" ]]; then
    TOOL_PREFIX=""
else
    echo "Unsupported platform: $platform (use baremetal, linux, or sim)"
    exit 1
fi

if [[ "$platform" == "sim" ]]; then
    SIM_DIR="${AIEHLC_ROOT}/script/sim"
    RUNSIM="${AIEHLC_ROOT}/script/runsim.sh"
    if [ ! -f "${RUNSIM}" ]; then
        echo "Error: runsim.sh not found at ${RUNSIM}"
        exit 1
    fi
    KERNEL_NAMES_LIST=""
    for kobj in ${KERNEL_OBJ_LIST}; do
        bname="$(basename "$kobj" .o)"
        fname="${bname#kernel_}"
        [ "$fname" == "kernel" ] && fname="${KERNEL_FUNC_NAME}"
        KERNEL_NAMES_LIST="${KERNEL_NAMES_LIST:+$KERNEL_NAMES_LIST }${fname}"
    done
    echo "[sim] Tilinglinalg — building simulator artifacts (kernel objs: ${KERNEL_OBJ_LIST})"
    SIM_HOST_FIXED="${WORKLOCAL_DIR}/host_sim_fixed.cc"
    sed 's/int main()/int main(int, char**)/' "${WORKLOCAL_DIR}/host.cc" > "${SIM_HOST_FIXED}"
    SIM_TILES_ARG="${SIM_TILES:-}"
    # shellcheck disable=SC1090
    source "${SIM_DIR}/write_sim_config.sh"
    write_sim_config "${WORKLOCAL_DIR}" "${SIM_HOST_FIXED}" "${KERNEL_OBJ_LIST}" \
                     "${KERNEL_NAMES_LIST}" "${aie_version}" "${SIM_TILES_ARG}"

    bash "${RUNSIM}" \
        --host-src     "${SIM_HOST_FIXED}" \
        --kernel-objs  "${KERNEL_OBJ_LIST}" \
        --kernel-names "${KERNEL_NAMES_LIST}" \
        --aie-gen      "${aie_version}" \
        --no-launch \
        $([ -n "$SIM_TILES_ARG" ] && echo "--stub-tiles $SIM_TILES_ARG" || echo "--stub-all")
    exit $?
fi

XILINX_VITIS_AIETOOLS="${XILINX_VITIS}/aietools"
LOCAL_ALIB_INCLUDE="${AIEHLC_ROOT}/thirdparty/alib/include"
if [ -d "${LOCAL_ALIB_INCLUDE}/xaiengine" ] && [ "$(ls -A ${LOCAL_ALIB_INCLUDE}/xaiengine 2>/dev/null)" ]; then
    AIETOOLS_INCLUDE_BASE="${LOCAL_ALIB_INCLUDE}"
else
    AIETOOLS_INCLUDE_BASE="${XILINX_VITIS_AIETOOLS}/include/drivers/aiengine"
fi

ARCH_DIR="${AIEHLC_ROOT}/thirdparty/arch"
# Support both nested workspace layout and flat bsp layout
if [ -d "${ARCH_DIR}/psv_cortexa72_0/workspace/platform_baremetal/psv_cortexa72_0/standalone_psv_cortexa72_0/bsp" ]; then
    ARCH_72_DIR="${ARCH_DIR}/psv_cortexa72_0/workspace/platform_baremetal/psv_cortexa72_0/standalone_psv_cortexa72_0/bsp"
else
    ARCH_72_DIR="${ARCH_DIR}/psv_cortexa72_0/bsp"
fi
if [ -d "${ARCH_DIR}/cortexa78_0/workspace/platform_baremetal/cortexa78_0/standalone_cortexa78_0/bsp" ]; then
    ARCH_78_DIR="${ARCH_DIR}/cortexa78_0/workspace/platform_baremetal/cortexa78_0/standalone_cortexa78_0/bsp"
else
    ARCH_78_DIR="${ARCH_DIR}/cortexa78_0/bsp"
fi
AIE_DRIVER_PARENT_DIR="${AIEHLC_ROOT}/thirdparty/alib"
ALIB_LIB_DIR="${AIEHLC_ROOT}/thirdparty/alib/lib"
USE_LOCAL_AIERT_BSP=0
if [ -d "${AIE_DRIVER_PARENT_DIR}/aie-rt/driver/" ]; then
    USE_LOCAL_AIERT_BSP=1
fi
USE_LOCAL_AIERT_BSP="${USE_LOCAL_AIERT_BSP:-0}"

# If default aie_version=2 but only libxaienginea78.a exists, use aie version 5
if [[ "$aie_version" == "2" ]] && [ -d "${ALIB_LIB_DIR}" ]; then
    if [ ! -f "${ALIB_LIB_DIR}/libxaienginea72.a" ] && [ -f "${ALIB_LIB_DIR}/libxaienginea78.a" ]; then
        aie_version="5"
        echo "Using AIE version 5 (libxaienginea78.a found; libxaienginea72.a not in ${ALIB_LIB_DIR})"
    fi
fi

if [ "$USE_LOCAL_AIERT_BSP" -eq 0 ]; then
    BAREMETAL_AIENGINE_INCLUDE="${XILINX_VITIS_AIETOOLS}/include/drivers/aiengine"
else
    BAREMETAL_AIENGINE_INCLUDE="${AIE_DRIVER_PARENT_DIR}/include"
fi

# Fallback xaiengine path when headers live under huaj-aie-rt/driver/include (for XAie_MemInst etc.)
XAIE_INCLUDE="${AIEHLC_ROOT}/thirdparty/alib/huaj-aie-rt/driver/include"

if [[ "$aie_version" == "1" || "$aie_version" == "2" ]]; then
    ARCH_APU_ALIB="${ARCH_72_DIR}/lib"
    ARCH_APU_AINC="${ARCH_72_DIR}/include"
    SECONDARY_ARCH_APU_AINC="${ARCH_DIR}/psv_cortexa72_0/bsp/include"
    AIELIB_APU_NAME="libxaienginea72.a"
    ARCH_APU_LD="${ARCH_DIR}/psv_cortexa72_0/lscript.ld"
    compiler_cpu_flag="-mcpu=cortex-a72"
    LINK_EXTRA="-lxil,-lxilstandalone,-lxiltimer"
elif [[ "$aie_version" == "5" ]]; then
    ARCH_APU_ALIB="${ARCH_78_DIR}/lib"
    ARCH_APU_AINC="${ARCH_78_DIR}/include"
    SECONDARY_ARCH_APU_AINC="${ARCH_DIR}/cortexa78_0/bsp/include"
    AIELIB_APU_NAME="libxaienginea78.a"
    ARCH_APU_LD="${ARCH_DIR}/cortexa78_0/lscript.ld"
    compiler_cpu_flag="-mcpu=cortex-a78"
    LINK_EXTRA="-lxiltimer,-lxilstandalone,-lxilpm"
else
    echo "Unsupported AIE version: $aie_version"
    exit 1
fi

# AIENGINE_LIB_DIR: Vitis BSP build output for aienginev2 (same as aiehlc.sh)
AIENGINE_LIB_DIR="${ARCH_APU_ALIB}/../libsrc/build_configs/gen_bsp/libsrc/aienginev2/src"
if [ "$USE_LOCAL_AIERT_BSP" -eq 1 ]; then
    BAREMETAL_AIENGINE_LIB="${AIELIB_APU_NAME:3:-2}"
elif [ -f "${ALIB_LIB_DIR}/${AIELIB_APU_NAME}" ]; then
    # aienginev2 not built, but the pre-built AIE engine lib exists in alib/lib
    BAREMETAL_AIENGINE_LIB="${AIELIB_APU_NAME:3:-2}"
else
    BAREMETAL_AIENGINE_LIB="aienginev2"
fi

# Include options: aiehlc host + aie_runtime.h; add xaiengine path when present (xaiengine.h, xaiengine/*.h)
# IMPORTANT: the BSP aie-rt driver headers (ARCH_APU_AINC) must precede the Vitis
# aietools aie-codegen headers (AIETOOLS_INCLUDE_BASE). The aie-codegen umbrella is
# a reduced subset that omits XAIE_IO_BACKEND_BAREMETAL and the high-level routing
# API (XAie_RoutingInstance, XAie_MoveDataExternal2Aie, ...) that aie_runtime.c needs.
# When USE_LOCAL_AIERT_BSP=1, the host links against the freshly-built local aie-rt
# lib (thirdparty/alib/lib/libxaienginea78.a), so the matching local headers
# (AIE_DRIVER_PARENT_DIR/include, a full aie-rt header set incl. routing + baremetal
# backend + newer APIs like XAie_PartitionInitialize_v2) must win over the older,
# BSP-snapshot headers. Otherwise fall back to BSP-first ordering.
if [ "$USE_LOCAL_AIERT_BSP" -eq 1 ]; then
    INCLUDE_OPTS="-I${AIEHLC_ROOT}/src/mlir/runtime -I${AIEHLC_ROOT}/include -I${AIE_DRIVER_PARENT_DIR}/include -I${ARCH_APU_AINC} -I${SECONDARY_ARCH_APU_AINC} -I${AIETOOLS_INCLUDE_BASE}"
else
    INCLUDE_OPTS="-I${AIEHLC_ROOT}/src/mlir/runtime -I${AIEHLC_ROOT}/include -I${ARCH_APU_AINC} -I${SECONDARY_ARCH_APU_AINC} -I${AIETOOLS_INCLUDE_BASE} -I${AIE_DRIVER_PARENT_DIR}/include"
fi
if [ -d "${XAIE_INCLUDE}" ]; then
    INCLUDE_OPTS="${INCLUDE_OPTS} -I${XAIE_INCLUDE}"
fi
# User headers are copied to worklocal/ by aiehlc alongside host.cc
INCLUDE_OPTS="${INCLUDE_OPTS} -I${WORKLOCAL_DIR}"
DEFS="-DAIE_GEN=${aie_version} ${EXTRA_DEFS:-}"
if [ "${AIEHLC_PROFILING:-0}" != "0" ]; then
    DEFS="${DEFS} -DAIEHLC_PROFILING=1"
    echo "[hostcompile] profiling runtime layer ENABLED (-DAIEHLC_PROFILING=1)"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Kernel .o file(s) are already built by compile_one_kernel with canonical symbols
# (_binary_kernel_<func_name>_{start,end,size})
# KERNEL_OBJ_LIST is set above (single file or multiple files for multi-kernel)

echo "Kernel object(s):"
for kobj in ${KERNEL_OBJ_LIST}; do
    echo "  $(basename ${kobj}):"
    nm "${kobj}" | grep _binary_ || true
done
echo ""

# Library vs executable mode.
#
# An app source with no main() (e.g. simplematmul2_aiegraph.cc, whose host entry
# is `int test_matmul(int)`) cannot be linked into an ELF — there is nothing to
# start. Such a source is meant to be embedded in a larger program, so archive
# the objects into a static library instead and let the caller supply main().
#
# Detect the entry point from the GENERATED host.cc: `int main()` is a
# user-written main passed through verbatim, `void main()` is the MLIR-only
# emission that the fixups below promote to `int main()`. Neither => library.
if grep -q 'int main()' "${WORKLOCAL_DIR}/host.cc"; then
    HOST_ENTRY_KIND="user-main"
elif grep -q 'void main()' "${WORKLOCAL_DIR}/host.cc"; then
    HOST_ENTRY_KIND="mlir-main"
else
    HOST_ENTRY_KIND="library"
fi

# Library name follows the app source: simplematmul2_aiegraph.cc -> libsimplematmul2_aiegraph.a
HOST_LIB_NAME="libhost.a"
if [ -f "${WORKLOCAL_DIR}/app_source.txt" ]; then
    _app_base="$(basename "$(cat "${WORKLOCAL_DIR}/app_source.txt")")"
    _app_base="${_app_base%.*}"
    [ -n "${_app_base}" ] && HOST_LIB_NAME="lib${_app_base}.a"
fi

echo "============================================"
echo "Host compilation (host.cc + aie_runtime)"
echo "============================================"
echo "Source: ${WORKLOCAL_DIR}/host.cc"
if [ "${HOST_ENTRY_KIND}" = "library" ]; then
    echo "Output: ${BUILD_DIR}/${HOST_LIB_NAME} (static library: host.cc has no main())"
else
    echo "Output: ${BUILD_DIR}/host"
fi
echo "Platform: $platform  AIE version: $aie_version"
echo ""

# Fix generated host.cc for C++: forward decl, int main(), return 0 in main, __global__
HOST_FIXED="${BUILD_DIR}/host_fixed.cc"
if [ "${HOST_ENTRY_KIND}" = "user-main" ]; then
    # User source already provides int main() — only add __global__ define
    sed -e '/#include "aie_runtime.h"/a\
#define __global__
' \
        "${WORKLOCAL_DIR}/host.cc" > "${HOST_FIXED}"
elif [ "${HOST_ENTRY_KIND}" = "mlir-main" ]; then
    # MLIR-only output: apply all fixups (void main → int main, return 0, forward decl)
    sed -e '/#include "aie_runtime.h"/a\
void host_canonicalized();\
#define __global__
' \
        -e 's/void main()/int main()/' \
        -e '0,/^  return;$/s/^  return;$/  return 0;/' \
        "${WORKLOCAL_DIR}/host.cc" > "${HOST_FIXED}"
else
    # Library: no main to promote. The `return 0` rewrite above is deliberately
    # NOT applied -- it targets the first `return;` in the file, which without a
    # main belongs to `void host_canonicalized()` and would be rewritten into
    # "return-statement with a value, in function returning 'void'".
    sed -e '/#include "aie_runtime.h"/a\
void host_canonicalized();\
#define __global__
' \
        "${WORKLOCAL_DIR}/host.cc" > "${HOST_FIXED}"
fi
HOST_SRC="${HOST_FIXED}"

# Compile host
if [ "${REBUILD}" -eq 1 ] || [ ! -f host.o ]; then
    set -x
    echo "Compiling host..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++20 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${HOST_SRC}" -o host.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile host.cc"
        exit 1
    fi
    set +x
else
    echo "Skipping host.cc (host.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

# Compile aie_runtime.c
if [ "${REBUILD}" -eq 1 ] || [ ! -f aie_runtime.o ]; then
    echo "Compiling aie_runtime.c..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${AIEHLC_ROOT}/src/mlir/runtime/aie_runtime.c" -o aie_runtime.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile aie_runtime.c"
        exit 1
    fi
else
    echo "Skipping aie_runtime.c (aie_runtime.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

# Compile aie_runtime_debug.c
if [ "${REBUILD}" -eq 1 ] || [ ! -f aie_runtime_debug.o ]; then
    echo "Compiling aie_runtime_debug.c..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${AIEHLC_ROOT}/src/mlir/runtime/aie_runtime_debug.c" -o aie_runtime_debug.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile aie_runtime_debug.c"
        exit 1
    fi
else
    echo "Skipping aie_runtime_debug.c (aie_runtime_debug.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

# Compile aie_runtime_stream_debug.c
if [ "${REBUILD}" -eq 1 ] || [ ! -f aie_runtime_stream_debug.o ]; then
    echo "Compiling aie_runtime_stream_debug.c..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${AIEHLC_ROOT}/src/mlir/runtime/aie_runtime_stream_debug.c" -o aie_runtime_stream_debug.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile aie_runtime_stream_debug.c"
        exit 1
    fi
else
    echo "Skipping aie_runtime_stream_debug.c (aie_runtime_stream_debug.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

# Compile aie_runtime_common.c
if [ "${REBUILD}" -eq 1 ] || [ ! -f aie_runtime_common.o ]; then
    echo "Compiling aie_runtime_common.c..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${AIEHLC_ROOT}/src/mlir/runtime/aie_runtime_common.c" -o aie_runtime_common.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile aie_runtime_common.c"
        exit 1
    fi
else
    echo "Skipping aie_runtime_common.c (aie_runtime_common.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

# Compile aie_runtime_control_plan.c (pure row-control planner)
if [ "${REBUILD}" -eq 1 ] || [ ! -f aie_runtime_control_plan.o ]; then
    echo "Compiling aie_runtime_control_plan.c..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${AIEHLC_ROOT}/src/mlir/runtime/aie_runtime_control_plan.c" -o aie_runtime_control_plan.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile aie_runtime_control_plan.c"
        exit 1
    fi
else
    echo "Skipping aie_runtime_control_plan.c (aie_runtime_control_plan.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

# Compile aie_runtime_resource.c (control-plane resource reservation table)
if [ "${REBUILD}" -eq 1 ] || [ ! -f aie_runtime_resource.o ]; then
    echo "Compiling aie_runtime_resource.c..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${AIEHLC_ROOT}/src/mlir/runtime/aie_runtime_resource.c" -o aie_runtime_resource.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile aie_runtime_resource.c"
        exit 1
    fi
else
    echo "Skipping aie_runtime_resource.c (aie_runtime_resource.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

if [ "${REBUILD}" -eq 1 ] || [ ! -f aie_ctrlpkt_encode.o ]; then
    echo "Compiling aie_ctrlpkt_encode.c..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${AIEHLC_ROOT}/src/mlir/runtime/aie_ctrlpkt_encode.c" -o aie_ctrlpkt_encode.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile aie_ctrlpkt_encode.c"
        exit 1
    fi
else
    echo "Skipping aie_ctrlpkt_encode.c (aie_ctrlpkt_encode.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

if [ "${REBUILD}" -eq 1 ] || [ ! -f aie_runtime_xaie_seq.o ]; then
    echo "Compiling aie_runtime_xaie_seq.c..."
    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${AIEHLC_ROOT}/src/mlir/runtime/aie_runtime_xaie_seq.c" -o aie_runtime_xaie_seq.o
    if [ $? -ne 0 ]; then
        echo "Error: failed to compile aie_runtime_xaie_seq.c"
        exit 1
    fi
else
    echo "Skipping aie_runtime_xaie_seq.c (aie_runtime_xaie_seq.o exists; pass 'rebuild' or '-rebuild' to force)"
fi

# Compile routing.cc (if present)
ROUTING_OBJ=""
if [ "${HAS_ROUTING}" -eq 1 ]; then
    if [ "${REBUILD}" -eq 1 ] || [ ! -f routing.o ]; then
        echo "Compiling routing.cc..."
        ${TOOL_PREFIX}g++ ${OPT_FLAGS} -std=c++17 ${DEFS} ${INCLUDE_OPTS} ${compiler_cpu_flag} -c "${WORKLOCAL_DIR}/routing.cc" -o routing.o
        if [ $? -ne 0 ]; then
            echo "Error: failed to compile routing.cc"
            exit 1
        fi
    else
        echo "Skipping routing.cc (routing.o exists; pass 'rebuild' or '-rebuild' to force)"
    fi
    ROUTING_OBJ="routing.o"
fi
# Must list every runtime object the ELF link below uses: aie_runtime.o calls
# rt_seq_* (aie_runtime_xaie_seq.o) and the ctrl-pkt encoder, so an archive
# without them links only until something reaches __Runtime_startio -- then
# `undefined reference to rt_seq_start_queue` in the consumer's link.
HOST_OBJS="host.o aie_runtime.o aie_runtime_debug.o aie_runtime_stream_debug.o aie_runtime_common.o aie_runtime_control_plan.o aie_runtime_resource.o aie_ctrlpkt_encode.o aie_runtime_xaie_seq.o ${ROUTING_OBJ} ${KERNEL_OBJ_LIST}"

if [ "${HOST_ENTRY_KIND}" = "library" ]; then
    # No main() => archive rather than link. The kernel ELF objects are included
    # so the embedded-kernel symbols (_binary_kernel_*_start/_end/_size) travel
    # with the library; the final executable still needs the BSP/XAie libs that
    # the link branch below passes, since an archive records no dependencies.
    echo "Archiving host static library (no main() in host.cc)..."
    set -x
    rm -f "${HOST_LIB_NAME}"
    ${TOOL_PREFIX}ar rcs "${HOST_LIB_NAME}" ${HOST_OBJS}
    _ar_rc=$?
    set +x
    if [ ${_ar_rc} -ne 0 ]; then
        echo "Error: failed to archive ${HOST_LIB_NAME}"
        exit 1
    fi
    ${TOOL_PREFIX}ranlib "${HOST_LIB_NAME}" 2>/dev/null || true

    echo ""
    echo "============================================"
    echo "Host static library built: ${BUILD_DIR}/${HOST_LIB_NAME}"
    echo "============================================"
    ls -l "${HOST_LIB_NAME}"
    echo ""
    echo "host.cc defines no main(), so no ELF was linked. Provide your own main()"
    echo "and link against this archive, e.g.:"
    echo ""
    # Same -L set and --start-group as the executable link below. All four are
    # needed: -l${BAREMETAL_AIENGINE_LIB} lives in ALIB_LIB_DIR, the BSP libs in
    # ARCH_APU_ALIB. Dropping either gives "cannot find -l...".
    echo "    ${TOOL_PREFIX}g++ ${OPT_FLAGS} -o app main.o \\"
    echo "        ${BUILD_DIR}/${HOST_LIB_NAME} \\"
    echo "        --specs=nosys.specs -Wl,--defsym,end=__bss_end__ \\"
    echo "        -Wl,-T -Wl,${ARCH_APU_LD} \\"
    echo "        -L\"${ALIB_LIB_DIR}\" \\"
    echo "        -L\"${AIENGINE_LIB_DIR}\" \\"
    echo "        -L\"${ARCH_APU_ALIB}\" \\"
    echo "        -L\"${AIE_DRIVER_PARENT_DIR}/lib\" \\"
    echo "        -Wl,--start-group,-lm,-l${BAREMETAL_AIENGINE_LIB},-lxil,-lgcc,-lc,-lstdc++,${LINK_EXTRA},--end-group"
    echo ""
    echo "NOTE: ${ALIB_LIB_DIR} is wiped and rebuilt on every aiehlc.sh run."
    echo "      Copy lib${BAREMETAL_AIENGINE_LIB}.a elsewhere if you link later."
    echo ""
    echo "Entry points available in the archive:"
    ${TOOL_PREFIX}nm --defined-only host.o 2>/dev/null | grep ' T ' | head -20 || true

    # Publish next to the ELF the executable path would have produced, so
    # callers have one predictable location to look in.
    AOUT_DIR="$(dirname "${WORKLOCAL_DIR}")"
    cp -f "${HOST_LIB_NAME}" "${AOUT_DIR}/${HOST_LIB_NAME}"
    echo "Published: ${AOUT_DIR}/${HOST_LIB_NAME}"

    cd "${_HOSTCOMPILE_ORIG_PWD}"
    exit 0
fi

set -x
# Link (same libs as aiehlc.sh baremetal host link: -L and -l for XAie_* and BSP)
# --specs=nosys.specs provides stubs for _exit, _close, _fstat, etc. (baremetal/newlib)
# -Wl,--defsym,end=__bss_end__ defines 'end' for newlib _sbrk (lscript.ld defines __bss_end__)
echo "Linking host (with embedded kernel binary)..."
${TOOL_PREFIX}g++ ${OPT_FLAGS} -o host host.o aie_runtime.o aie_runtime_debug.o aie_runtime_stream_debug.o aie_runtime_common.o aie_runtime_control_plan.o aie_runtime_resource.o aie_ctrlpkt_encode.o aie_runtime_xaie_seq.o ${ROUTING_OBJ} ${KERNEL_OBJ_LIST} \
    --specs=nosys.specs \
    -Wl,--defsym,end=__bss_end__ \
    -Wl,-T -Wl,${ARCH_APU_LD} \
    -L"${ALIB_LIB_DIR}" \
    -L"${AIENGINE_LIB_DIR}" \
    -L"${ARCH_APU_ALIB}" \
    -L"${AIE_DRIVER_PARENT_DIR}/lib" \
    -Wl,--start-group,-lm,-l${BAREMETAL_AIENGINE_LIB},-lxil,-lgcc,-lc,-lstdc++,${LINK_EXTRA},--end-group
if [ $? -ne 0 ]; then
    echo "Error: link failed (check XILINX_VITIS and BSP libs)"
    exit 1
fi

echo ""
echo "============================================"
echo "Host built successfully: ${BUILD_DIR}/host"
echo "============================================"
ls -l host

AOUT_DIR="$(dirname "${WORKLOCAL_DIR}")"

# Publish a stripped main.elf, keeping the full one as main.debug.elf.
#
# ~70% of the linked ELF is DWARF that none of our own code produces: host.cc,
# the kernels and routing.cc all compile at -Os with no -g (see OPT_FLAGS), but
# the prebuilt BSP, libgloss and aie-rt libraries we link against were built
# with -g and drag their debug info in. On a 2.8 MB A2 build that is ~2.0 MB of
# DWARF riding along on every `dow -force` over JTAG.
#
# --strip-debug (not --strip-all) keeps .symtab, so xsdb/aiedbg can still
# resolve function names for breakpoints and backtraces. Only non-loadable
# sections go: the LOAD segment is byte-identical before and after, so this
# cannot change what executes on the board.
#
# DEBUG_SYMS=1 means "I am debugging" -- it already adds -g above, so it also
# suppresses the strip and publishes the unstripped ELF directly.
if [ "${DEBUG_SYMS:-0}" -eq 1 ]; then
    cp -f "${BUILD_DIR}/host" "${AOUT_DIR}/main.elf"
    # This ELF already has symbols, so a main.debug.elf left over from an
    # earlier stripped build would be a stale sibling of a different binary --
    # exactly the file someone would then load symbols from.
    rm -f "${AOUT_DIR}/main.debug.elf"
    echo "Published: ${AOUT_DIR}/main.elf (unstripped, DEBUG_SYMS=1)"
else
    cp -f "${BUILD_DIR}/host" "${AOUT_DIR}/main.debug.elf"
    # Fall back to publishing unstripped rather than failing the build: a
    # missing objcopy is a smaller problem than no ELF at all.
    if ! "${TOOL_PREFIX}objcopy" --strip-debug \
            "${BUILD_DIR}/host" "${AOUT_DIR}/main.elf" 2>/dev/null; then
        echo "Warning: ${TOOL_PREFIX}objcopy unavailable; publishing unstripped"
        cp -f "${BUILD_DIR}/host" "${AOUT_DIR}/main.elf"
    fi
    echo "Published: ${AOUT_DIR}/main.elf (stripped; full symbols in main.debug.elf)"
    ls -l "${AOUT_DIR}/main.debug.elf"
fi
ls -l "${AOUT_DIR}/main.elf"

cd "${_HOSTCOMPILE_ORIG_PWD}"
