#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################


# Compile and generate the aie-rt driver header files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEHLC_DIR="${SCRIPT_DIR}/../"
ARCH_DIR=$AIEHLC_DIR/thirdparty/arch/
AIE_DRIVER_PARENT_DIR=$AIEHLC_DIR/thirdparty/alib/
export PATH=$PATH:$AIEHLC_DIR:$SCRIPT_DIR

usage() {
    echo "Usage: $0 [--help] [--path-set-only] [--enable-llvmaie] [--bsp-use-git-repo=<repo>] [--vitis-settings PATH]"
    echo "  --help                     Show this help message"
    echo "  --path-set-only            Set up the path only"
    echo "  --enable-llvmaie           Skip llvm-aie installation"
    echo "  --bsp-use-git-repo=<repo>  Skip BSP generation and build aie-rt from git repo"
    echo "  --vitis-settings PATH      Set  VITIS_SETTINGS_PATH"
}

aierepo_download_check() {
    set -e

    local AIE_REPO="$1"
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

    bash -c "pushd  $AIE_DRIVER_PARENT_DIR;git clone $AIE_REPO; popd"
    bash -c "cd $AIE_DRIVER_PARENT_DIR/aie-rt/driver/src; make -f Makefile.Linux; make clean"
    ##temporary disable PLM support privledge register, as compile have issue
    find $AIE_DRIVER_PARENT_DIR/aie-rt/driver/src -name "Makefile" -exec sed -i 's/-DXAIE_PROD//g' {} \;
}

USE_LLVMAIE=0
SKIP_BSP=0
LOCAL_AIE_RT_REPO=0
PATH_SET_ONLY=0
#VITIS_SETTINGS_PATH="/proj/xbuilds/2025.2_0414_1/installs/lin64/HEAD/Vitis/settings64.sh"
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
        --path-to-vitis-settings)
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
export LLVM_AIE_PATH="${AIEHLC_DIR}/thirdparty/llvm-aie"
popd

#set up the vitis path
if [ -n "$XILINX_VITIS" ]; then
    echo "XILINX_VITIS is set to $XILINX_VITIS"
elif which aiecompiler > /dev/null 2>&1; then
    XILINX_VITIS=$(dirname $(which aiecompiler))/../..
    export XILINX_VITIS
    echo "XILINX_VITIS is set to $XILINX_VITIS based on aiecompiler location."
else
    echo "XILINX_VITIS is not set or aiecompiler not found."
    echo "Trying to source ${VITIS_SETTINGS_PATH}"
    source $VITIS_SETTINGS_PATH
    export XILINX_VITIS
fi

echo "XILINX_VITIS: ${XILINX_VITIS}"

#set up the petalinux path
if [ -n "$PETALINUX" ]; then
    echo "PETALINUX is set to $PETALINUX"
elif [ -d "/proj/petalinux/2025.1/petalinux-v2025.1_daily_latest/tool/petalinux-v2025.1-final" ]; then
    PETALINUX="/proj/petalinux/2025.1/petalinux-v2025.1_daily_latest/tool/petalinux-v2025.1-final"
    export PETALINUX
    echo "PETALINUX is set to $PETALINUX based on directory existence."
else
    echo "PETALINUX is not set or directory does not exist."
fi
echo "PETALINUX: ${PETALINUX}"

export LIB_PATH="${XILINX_VITIS}/gnu/aarch64/lin/aarch64-none/x86_64-oesdk-linux/usr/lib"
export LIB_BASE_PATH="${XILINX_VITIS}/gnu/aarch64/lin/aarch64-none/x86_64-oesdk-linux/lib"
export LD_SO="${LIB_BASE_PATH}/ld-linux-x86-64.so.2"
export CLANG_INCLUDE_PATH="${XILINX_VITIS}/gnu/aarch64/lin/aarch64-none/x86_64-oesdk-linux/usr/lib/aarch64-xilinx-elf/gcc/aarch64-xilinx-elf/13.3.0/include/"

if [ "$SKIP_BSP" -eq 0 ]; then
    if [ ! -d "${AIEHLC_DIR}/thirdparty/arch/cortexa78_0/workspace" ] || \
        [ ! -d "${AIEHLC_DIR}/thirdparty/arch/psv_cortexa72_0/workspace" ] || \
        [ ! -d "${AIEHLC_DIR}/thirdparty/arch/psv_cortexr5_0/workspace" ]; then
        echo "Generating BSPs..."
        pushd "$SCRIPT_DIR"
        
        VEK280_XSA=$XILINX_VITIS/base_platforms/xilinx_vek280_base_202520_1/hw/hw.xsa
        VEK385_XSA=/proj/xbuilds/2025.2_daily_latest/internal_platforms/vek385_base_202520_1/hw/hw.xsa

        vitis -s gen_bsp.py --xsa $VEK280_XSA --processor psv_cortexr5_0       
        vitis -s gen_bsp.py --xsa $VEK280_XSA --processor psv_cortexa72_0
        vitis -s gen_bsp.py --xsa $VEK385_XSA --processor cortexa78_0

        popd
    else
        echo "BSPs already exist. Skipping generation."
    fi
else
    if [ "$PATH_SET_ONLY" -eq 0 ]; then
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
