#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################


# Compile and generate the aie-rt driver header files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEHLC_DIR="${SCRIPT_DIR}/../"
ARCH_DIR=$AIEHLC_DIR/thirdparty/arch/
AIE_DRIVER_PARENT_DIR=$AIEHLC_DIR/thirdparty/alib/
export PATH=$PATH:$AIEHLC_DIR:$SCRIPT_DIR:$SCRIPT_DIR/test

usage() {
    echo "Usage: $0 [--help] [--path-set-only] [--enable-llvmaie] [--bsp-use-git-repo=<repo>] [--vitis-settings PATH]"
    echo "  --help                     Show this help message"
    echo "  --path-set-only            Set up the path only"
    echo "  --enable-llvmaie           Skip llvm-aie installation"
    echo "  --bsp-use-git-repo=<repo>  Skip BSP generation and build aie-rt from git repo"
    echo "  --vitis-settings PATH      Set  VITIS_SETTINGS_PATH"
}

aierepo_download_check() {
    # No `set -e` here: this file is sourced, and errexit would leak into (and
    # later kill) the caller's interactive shell. Failures return non-zero.
    local AIE_REPO="$1"
    local AIE_BRANCH="${AIE_RT_BRANCH:-xlnx_rel_v2026.2}"
    if [ -z "$AIE_REPO" ]; then
         echo "AIE_REPO is $AIE_REPO set to empty exit"
         return 1
    fi

    if [ ! -d "${AIE_DRIVER_PARENT_DIR}" ] ; then
        mkdir -p ${AIE_DRIVER_PARENT_DIR}
    fi

    if [ ! -d "${AIE_DRIVER_PARENT_DIR}/lib" ] ; then
        mkdir -p ${AIE_DRIVER_PARENT_DIR}/lib
    fi

    if [ ! -d "${AIE_DRIVER_PARENT_DIR}/include" ] ; then
        mkdir -p ${AIE_DRIVER_PARENT_DIR}/include
    fi

    if [ ! -d "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src" ] ; then
        if ! git clone --branch "$AIE_BRANCH" "$AIE_REPO" "$AIE_DRIVER_PARENT_DIR/aie-rt"; then
            echo "ERROR: failed to clone aie-rt (branch $AIE_BRANCH) from $AIE_REPO."
            echo "       Check GitHub credentials for that repo, or set AIE_RT_REPO to a reachable URL or local clone."
            return 1
        fi
    else
        echo "aie-rt already present at $AIE_DRIVER_PARENT_DIR/aie-rt; skipping clone"
    fi
    ##temporary disable PLM support privledge register, as compile have issue
    find $AIE_DRIVER_PARENT_DIR/aie-rt/driver/src -name "Makefile" -exec sed -i 's/-DXAIE_PROD//g' {} \;
    make -C "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src" -f Makefile.Linux include || return 1
    make -C "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src" -f Makefile.Linux include \
        INCLUDEDIR=../../../include INTERNALDIR=../../../internal || return 1
}

USE_LLVMAIE=0
SKIP_BSP=0
LOCAL_AIE_RT_REPO=0
PATH_SET_ONLY=0
# Public upstream aie-rt. Default branch xlnx_rel_v2026.2: it has the AIE2PS
# driver and XAIE_PART_INIT_OPT_CTRL_TLASTERROR_DISABLE (used by the
# high-throughput control path in src/mlir/runtime/aie_runtime.c, required);
# main-aie and xlnx_rel_v2026.1 lack it and are not supported.
# Override with AIE_RT_REPO=<url|local clone> and AIE_RT_BRANCH=<branch>.
AIE_RT_DEFAULT_REPO="${AIE_RT_REPO:-https://github.com/Xilinx/aie-rt.git}"
#VITIS_SETTINGS_PATH="/proj/xbuilds/2025.2_0414_1/installs/lin64/HEAD/Vitis/settings64.sh"
#VITIS_SETTINGS_PATH="/proj/xbuilds/HEAD_qualified_latest/installs/lin64/HEAD/Vitis/settings64.sh"
VITIS_SETTINGS_PATH="/proj/xbuilds/HEAD_qualified_latest/installs/lin64/HEAD/Vitis/settings64.sh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            usage
            return
            ;;
        --enable-llvmaie)
            USE_LLVMAIE=1
            shift
            ;;
        --path-set-only)
            SKIP_BSP=1
            PATH_SET_ONLY=1
            shift
            ;;
        --bsp-use-git-repo=*)
            SKIP_BSP=1
            LOCAL_AIE_RT_REPO="${1#*=}"
            shift
            ;;
        --vitis-settings|--path-to-vitis-settings)
            VITIS_SETTINGS_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            return 1
            ;;
    esac
done

# Run precommit setup only when not in path-set-only mode
# (must be after argument parsing so --path-set-only is processed)
if [ "${PATH_SET_ONLY:-0}" -eq 0 ]; then
    if [ -f "$SCRIPT_DIR/precommitsetup.sh" ]; then
        bash "$SCRIPT_DIR/precommitsetup.sh"
    fi
fi

pushd .

if [ "$USE_LLVMAIE" -eq 1 ]; then
    if [ ! -e "${AIEHLC_DIR}/thirdparty/llvm-aie" ]; then
        cd "$AIEHLC_DIR/thirdparty"
        pip download llvm_aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
        unzip llvm_aie*whl
    else
        echo "llvm-aie directory already exists. Skipping download and extraction."
    fi
else
    echo "Skipping llvm-aie installation due to --skip-llvm option."
fi
# Respect a caller-set LLVM_AIE_PATH (e.g. a versioned install like llvm-aie-22.*).
if [[ -z "${LLVM_AIE_PATH:-}" ]]; then
    export LLVM_AIE_PATH="${AIEHLC_DIR}/thirdparty/llvm-aie"
fi
popd

#set up the vitis path
if [ -n "${XILINX_VITIS:-}" ]; then
    echo "XILINX_VITIS is set to $XILINX_VITIS"
elif which aiecompiler > /dev/null 2>&1; then
    XILINX_VITIS=$(dirname $(which aiecompiler))/../..
    export XILINX_VITIS
    echo "XILINX_VITIS is set to $XILINX_VITIS based on aiecompiler location."
else
    echo "XILINX_VITIS is not set or aiecompiler not found."
    echo "Trying to source ${VITIS_SETTINGS_PATH}"
    case $- in
        *u*) _aiehlc_nounset=1; set +u ;;
        *) _aiehlc_nounset=0 ;;
    esac
    if ! source "$VITIS_SETTINGS_PATH" || ! command -v vitis >/dev/null 2>&1; then
        echo "ERROR: sourcing $VITIS_SETTINGS_PATH did not set up the Vitis environment."
        echo "       Pass --vitis-settings <Vitis>/settings64.sh, or (in docker) mount /proj with"
        echo "       rslave propagation (script/docker.sh does) so autofs paths resolve."
    fi
    if [ -n "${XILINX_VITIS:-}" ]; then
        export XILINX_VITIS
    fi
    if [ "${_aiehlc_nounset}" -eq 1 ]; then
        set -u
    fi
    unset _aiehlc_nounset
fi

echo "XILINX_VITIS: ${XILINX_VITIS:-}"

if [ -n "${XILINX_VITIS:-}" ]; then
    bash "$SCRIPT_DIR/sim/gen_aiesimulator.sh" "$AIEHLC_DIR/aiehlc_aiesimulator" || \
        echo "WARNING: failed to generate aiehlc_aiesimulator launcher."
fi

#set up the petalinux path
if [ -n "${PETALINUX:-}" ]; then
    echo "PETALINUX is set to $PETALINUX"
elif [ -d "/proj/petalinux/2025.1/petalinux-v2025.1_daily_latest/tool/petalinux-v2025.1-final" ]; then
    PETALINUX="/proj/petalinux/2025.1/petalinux-v2025.1_daily_latest/tool/petalinux-v2025.1-final"
    export PETALINUX
    echo "PETALINUX is set to $PETALINUX based on directory existence."
else
    echo "PETALINUX is not set or directory does not exist."
fi
echo "PETALINUX: ${PETALINUX:-}"

export LIB_PATH="${XILINX_VITIS}/gnu/aarch64/lin/aarch64-none/x86_64-oesdk-linux/usr/lib"
export LIB_BASE_PATH="${XILINX_VITIS}/gnu/aarch64/lin/aarch64-none/x86_64-oesdk-linux/lib"
export LD_SO="${LIB_BASE_PATH}/ld-linux-x86-64.so.2"
# Auto-detect GCC version for Clang builtin headers (stddef.h, stdarg.h, etc.)
_gcc_include_base="${XILINX_VITIS}/gnu/aarch64/lin/aarch64-none/x86_64-oesdk-linux/usr/lib/aarch64-xilinx-elf/gcc/aarch64-xilinx-elf"
_gcc_version=$(ls -1 "$_gcc_include_base" 2>/dev/null | head -1)
if [ -z "$_gcc_version" ]; then
    echo "WARNING: Could not detect GCC version under ${_gcc_include_base}"
    export CLANG_INCLUDE_PATH="${_gcc_include_base}/13.3.0/include/"
else
    export CLANG_INCLUDE_PATH="${_gcc_include_base}/${_gcc_version}/include/"
fi
unset _gcc_include_base _gcc_version
# LLVM/MLIR/clang build (script/llvm/build.sh). Respect a caller-set value;
# otherwise use <main checkout>/thirdparty/llvm-project/build (script/llvm/build.sh
# default, shared by all git worktrees), then this tree's own
# thirdparty/llvm-project/build, then an older sibling ../llvm-project/build.
if [ -z "${LLVM_INSTALL_DIR:-}" ]; then
    _aiehlc_main="$(cd "$AIEHLC_DIR" && pwd)"
    if _gcd="$(git -C "$AIEHLC_DIR" rev-parse --git-common-dir 2>/dev/null)"; then
        case "$_gcd" in /*) ;; *) _gcd="$AIEHLC_DIR/$_gcd" ;; esac
        _gcd="$(cd "$_gcd" && pwd)"
        [ "$(basename "$_gcd")" = ".git" ] && _aiehlc_main="$(dirname "$_gcd")"
    fi
    for _llvm_cand in "$_aiehlc_main/thirdparty/llvm-project/build" \
                      "$(cd "$AIEHLC_DIR" && pwd)/thirdparty/llvm-project/build" \
                      "$(cd "$AIEHLC_DIR/.." && pwd)/llvm-project/build" /usr/local; do
        if [ -x "$_llvm_cand/bin/llvm-config" ] && [ -d "$_llvm_cand/lib/cmake/mlir" ]; then
            export LLVM_INSTALL_DIR="$_llvm_cand"
            break
        fi
    done
    unset _llvm_cand _aiehlc_main _gcd
fi
if [ -z "${LLVM_INSTALL_DIR:-}" ]; then
    echo "WARNING: LLVM_INSTALL_DIR not set and no LLVM+MLIR build found; run: bash script/llvm/build.sh"
else
    echo "LLVM_INSTALL_DIR: $LLVM_INSTALL_DIR"
fi

if [ "$SKIP_BSP" -eq 0 ]; then
    if [ -d "${AIE_DRIVER_PARENT_DIR}/include/" ]; then
        echo "Cleaning thirdparty/alib/include/ directory..."
        rm -rf "${AIE_DRIVER_PARENT_DIR}/include/"
    fi
    if [ -d "${AIE_DRIVER_PARENT_DIR}/lib" ] && [ "$(ls -A ${AIE_DRIVER_PARENT_DIR}/lib 2>/dev/null)" ]; then
        echo "Cleaning thirdparty/alib/lib directory..."
        rm -rf "${AIE_DRIVER_PARENT_DIR}/lib"/
    fi

    if [ ! -d "${AIEHLC_DIR}/thirdparty/arch/cortexa78_0/workspace" ] || \
        [ ! -d "${AIEHLC_DIR}/thirdparty/arch/psv_cortexa72_0/workspace" ] || \
        [ ! -d "${AIEHLC_DIR}/thirdparty/arch/psv_cortexr5_0/workspace" ]; then
        echo "Generating BSPs..."
        pushd "$SCRIPT_DIR"
        
        # Auto-detect XSA platform paths (works across Vitis versions)
        VEK280_XSA=$(ls -d "$XILINX_VITIS/base_platforms/xilinx_vek280_base_"*/hw/hw.xsa 2>/dev/null | head -1)
        VEK385_XSA=$(ls -d "$XILINX_VITIS/base_platforms/"*vek385*/hw/hw.xsa 2>/dev/null | head -1)

        if [ -z "$VEK280_XSA" ]; then
            echo "Error: Could not find VEK280 XSA under $XILINX_VITIS/base_platforms/"
            echo "Expected pattern: xilinx_vek280_base_*/hw/hw.xsa"
            popd
            return 1
        fi
        echo "VEK280 XSA: $VEK280_XSA"

        vitis -s gen_bsp.py --xsa $VEK280_XSA --processor psv_cortexr5_0
        vitis -s gen_bsp.py --xsa $VEK280_XSA --processor psv_cortexa72_0

        if [ -n "$VEK385_XSA" ]; then
            echo "VEK385 XSA: $VEK385_XSA"
            vitis -s gen_bsp.py --xsa $VEK385_XSA --processor cortexa78_0
        else
            echo "Warning: VEK385 XSA not found, skipping cortexa78_0 BSP generation"
        fi

        popd
    else
        echo "BSPs already exist. Skipping generation."
    fi

    echo "Installing aie-rt headers from ${AIE_RT_DEFAULT_REPO}"
    aierepo_download_check "${AIE_RT_DEFAULT_REPO}"
else
    if [ "${PATH_SET_ONLY:-0}" -eq 0 ]; then
        echo "Skipping BSP generation due to --skip-bsp option."
        ARCH_72_DIR=$ARCH_DIR/psv_cortexa72_0/workspace/platform_baremetal/psv_cortexa72_0/standalone_psv_cortexa72_0/
        ARCH_53_DIR=$ARCH_DIR/psv_cortexr5_0/workspace/platform_baremetal/psv_cortexr5_0/standalone_psv_cortexr5_0/
        ARCH_78_DIR=$ARCH_DIR/cortexa78_0/workspace/platform_baremetal/cortexa78_0/standalone_cortexa78_0/
        if [ ! -d "${ARCH_72_DIR}" ] ; then
            mkdir -p ${ARCH_72_DIR}
        fi

        if [ ! -d "${ARCH_53_DIR}" ] ; then
            mkdir -p ${ARCH_53_DIR}
        fi

        if [ ! -d "${ARCH_78_DIR}" ] ; then
            mkdir -p ${ARCH_78_DIR}
        fi

        cp -rf "$ARCH_DIR/psv_cortexa72_0/bsp/" "${ARCH_72_DIR}/"
        cp -r "$ARCH_DIR/psv_cortexr5_0/bsp/" "${ARCH_53_DIR}/"
        cp -r "$ARCH_DIR/cortexa78_0/bsp/" "${ARCH_78_DIR}/"

        #clone aie-rt
        echo $LOCAL_AIE_RT_REPO
        aierepo_download_check $LOCAL_AIE_RT_REPO
    fi
    
fi

