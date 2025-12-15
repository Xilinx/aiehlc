#!/usr/bin/env bash  
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

dbg_echo() {
    if [ "$DEBUG_OUTPUT" = 1 ]; then
        echo "$@"
    fi
}

run_cmd() {
    local command="$1"
    if [ "$DEBUG_OUTPUT" = 0 ]; then
        $command > /dev/null 2>&1
    else
        $command
    fi
}

usage() {
    echo "Usage: $0 --runtime-source-file <path> --aie-version <version> [--kernel-count <count>] [--kernel <source> [<directory>]]"
    exit 1
}

redefine_symbols() {
    local obj_file="$1"
    local func_name="$2"
    local objcopy_tool="$3"

    nm "$obj_file" | while read -r line; do
        symbol=$(echo "$line" | awk '{print $3}')
        
        if echo "$symbol" | grep -q "_binary__.*_aout_build_${func_name}_.*_end$"; then
            dbg_echo "Renaming symbol: $symbol to _binary_kernel_${func_name}_end"
            "$objcopy_tool" --redefine-sym "$symbol"=_binary_kernel_"${func_name}"_end "$obj_file"
        elif echo "$symbol" | grep -q "_binary__.*_aout_build_${func_name}_.*_start$"; then
            dbg_echo "Renaming symbol: $symbol to _binary_kernel_${func_name}_start"
            "$objcopy_tool" --redefine-sym "$symbol"=_binary_kernel_"${func_name}"_start "$obj_file"
        elif echo "$symbol" | grep -q "_binary__.*_aout_build_${func_name}_.*_size$"; then
            dbg_echo "Renaming symbol: $symbol to _binary_kernel_${func_name}_size"
            "$objcopy_tool" --redefine-sym "$symbol"=_binary_kernel_"${func_name}"_size "$obj_file"
        fi
    done
}

build_hw_lib() {
    set -e

    local header_dir="$1"
    local compiler="$2"
    local assembler="$3"
    local archiver="$4"
    local compiler_flags="$5"
    local extra_flags="$6"
    local output_lib="$7"
    local lib_dir="$8"
    local ld_dir="$9"
    local lplatform="${10}"

        if [ -d "${AIE_DRIVER_PARENT_DIR}/aie-rt/driver/" ] ; then
            dbg_echo "header_dir is $header_dir"

            dbg_echo "[INFO] Cleaning include and lib folders..."
            rm -rf "$AIE_DRIVER_PARENT_DIR/include/"*.h
            rm -rf "$AIE_DRIVER_PARENT_DIR/lib/"*.a
            rm -rf "$AIE_DRIVER_PARENT_DIR/include/xaiengine"

            dbg_echo "[INFO] Copying headers..."
            dbg_echo "$lplatform is lplatform"

            if [[ "$lplatform" == "linux" ]]; then
                bash -c "cd $AIE_DRIVER_PARENT_DIR/aie-rt/driver/src; make -C ./ CC=aarch64-linux-gnu-gcc CFLAGS='-Wall -g -Wextra --std=c11 -D__AIELINUX__ -fpermissive'  -f Makefile.Linux"
                ##keep the order, first copy bsp header, then copy the aiert header, then even bsp header have old aiegnine header
                ##we still get the latest aiert header
                cp -rf "$header_dir"/*.h "$AIE_DRIVER_PARENT_DIR/include/"
                cp -rf "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/internal/"* "$AIE_DRIVER_PARENT_DIR/include/"
                
            elif [[ "$platform" == "baremetal" ]]; then
                bash -c "cd $AIE_DRIVER_PARENT_DIR/aie-rt/driver/src; make -f Makefile.Linux"
                ##keep the order, first copy bsp header, then copy the aiert header, then even bsp header have old aiegnine header
                ##we still get the latest aiert header
                cp -rf "$header_dir"/*.h "$AIE_DRIVER_PARENT_DIR/include/"
                cp -rf "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/internal/"* "$AIE_DRIVER_PARENT_DIR/include/"


                dbg_echo "[INFO] Cleaning previous build..."
                make clean -C "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src/"

                dbg_echo "[INFO] Copying extra libs and ld script..."
                cp "$lib_dir"/*.a "$AIE_DRIVER_PARENT_DIR/lib/"
                cp "$ld_dir" "$AIE_DRIVER_PARENT_DIR/lib/"  

                dbg_echo "[INFO] Building library with $compiler"
                make -C "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src" -s libs \
                    SHELL=/bin/sh \
                    COMPILER="$compiler" \
                    ASSEMBLER="$assembler" \
                    ARCHIVER="$archiver" \
                    COMPILER_FLAGS="$compiler_flags -DAIE_GEN=$AIE_GEN" \
                    EXTRA_COMPILER_FLAGS="$extra_flags"        

                dbg_echo "[INFO] Moving libxil.a to $output_lib"
                bash -c "mv $AIE_DRIVER_PARENT_DIR/lib/libxil.a $AIE_DRIVER_PARENT_DIR/lib/$output_lib"
            fi
        fi
    }


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AIEHLC_DIR="${SCRIPT_DIR}/../"
runtime_source_file=""
aie_version="2"
use_llvm_aie="false"
DEBUG_OUTPUT=0
platform="baremetal"
USE_LOCAL_AIERT_BSP=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -help)
            usage
            ;;
        --runtime-source-file)
            runtime_source_file="$2"
            shift 2
            ;;
        --aie-version)
            aie_version="$2"
            shift 2
            ;;
        --platform)
            platform="$2"
            shift 2
            ;;
        --kernel)
            kernel_source="$2"
            shift 2
            ;;
        --use-llvm-aie)
            use_llvm_aie="true"
            shift
            ;;
        --debug-output)
            DEBUG_OUTPUT=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

#set up env
run_cmd "source $SCRIPT_DIR/setup.sh --path-set-only"

if [[ "$platform" == "linux" ]]; then
    TOOL_PREFIX="aarch64-linux-gnu-"
elif [[ "$platform" == "baremetal" ]]; then
    TOOL_PREFIX="aarch64-none-elf-"
fi

KERNEL_DIR=$(pwd)/aout/
rm -rf $KERNEL_DIR
XILINX_VITIS_AIETOOLS=$XILINX_VITIS/aietools
CARDANO_AIE_ARCH_MODEL_DIR="$XILINX_VITIS_AIETOOLS/data/versal_prod/lib"

INCLUDE_PATH="-I$XILINX_VITIS_AIETOOLS/include \
-I$XILINX_VITIS_AIETOOLS//include/aie_api \
-I$XILINX_VITIS_AIETOOLS//include/drivers/aiengine"

LLVM_AIE_INCLUDE_PATH="-I$XILINX_VITIS_AIETOOLS/include \
-I$XILINX_VITIS_AIETOOLS//include/aie_api \
-I$XILINX_VITIS_AIETOOLS//include/drivers/aiengine \
"
LLVM_AIE_LIB_PATH="-Wl,-L$XILINX_VITIS_AIETOOLS/data/aie_ml/lib/Release"

compiler_flags_llvm_aie_aie="-include ${LLVM_AIE_PATH}/../llvm-aie-extra.h -Wno-unknown-attributes -Wno-macro-redefined -O2 -std=c++20 --target=aie2-none-unknown-elf -D__AIECC__ -D__AIENGINE__ -D__AIE_ARCH__=10 -D__AIEARCH=10 -D_LIBCPP_HAS_NO_THREADS -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR $LLVM_AIE_INCLUDE_PATH $INCLUDE_PATH"
compiler_flags_llvm_aie_aieml="-include ${LLVM_AIE_PATH}/../llvm-aie-extra.h -Wno-unknown-attributes -Wno-macro-redefined -O2 -std=c++20 --target=aie2-none-unknown-elf -D__AIECC__ -D__AIENGINE__ -D__AIE_ARCH__=20 -D__AIEARCH=20 -D_LIBCPP_HAS_NO_THREADS -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST $LLVM_AIE_INCLUDE_PATH $INCLUDE_PATH"
compiler_flags_llvm_aie_aie2ps="-include ${LLVM_AIE_PATH}/../llvm-aie-extra.h -Wno-unknown-attributes -Wno-macro-redefined -O2 -std=c++20 --target=aie2-none-unknown-elf -D__AIECC__ -D__AIENGINE__ -D__AIE_ARCH__=22 -D__AIEARCH=22 -D_LIBCPP_HAS_NO_THREADS -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST $LLVM_AIE_INCLUDE_PATH $INCLUDE_PATH"

if [ $DEBUG_OUTPUT = 1 ]; then
    compiler_flags_llvm_aie_aie+="-v"
    compiler_flags_llvm_aie_aieml+="-v"
    compiler_flags_llvm_aie_aie2ps+="-v"
fi

arch_model_dir_aie="${XILINX_VITIS}/aietools/data/aie/lib"
arch_model_dir_aieml="${XILINX_VITIS}/aietools/data/aie_ml/lib"
arch_model_dir_aie2ps="${XILINX_VITIS}/aietools/data/aie2ps/lib"


silent_flag="-s"
if [ $DEBUG_OUTPUT = 1 ]; then
    silent_flag="+s"
fi

compiler_flags_aie="$silent_flag +f -p me -P $arch_model_dir_aie +P 4 +Wllvm,-O2,-fno-jump-tables,-fno-discard-value-names,-mllvm,-chess-collapse-struct-types-during-linking=0,-Xclang,-chess-only-info-critical-passes -D__AIENGINE__ -D__AIE_ARCH__=10 -D__AIEARCH=10 -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR $INCLUDE_PATH -I"$KERNEL_DIR/TheHouseOfCommons/" "
compiler_flags_aieml="-aiearch aie-ml $silent_flag +f -p me -P $arch_model_dir_aieml +P 4 +Wllvm,-O2,-fno-jump-tables,-fno-discard-value-names,-mllvm,-chess-collapse-struct-types-during-linking=0,-Xclang,-chess-only-info-critical-passes -D__AIENGINE__ -D__AIE_ARCH__=20 -D__AIEARCH=20 -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST $INCLUDE_PATH -I"$KERNEL_DIR/TheHouseOfCommons/" "
compiler_flags_aie2ps="-aiearch aie2ps $silent_flag +f -p me -P $arch_model_dir_aie2ps +P 4 +Wllvm,-O2,-fno-jump-tables,-fno-discard-value-names,-mllvm,-chess-collapse-struct-types-during-linking=0,-Xclang,-chess-only-info-critical-passes -D__AIENGINE__ -D__AIE_ARCH__=22 -D__AIEARCH=22 -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST $INCLUDE_PATH -I"$KERNEL_DIR/TheHouseOfCommons/" "

extra_chess_flag=""
if [[ "$aie_version" == "1" ]]; then
    compiler_flags_chess="$compiler_flags_aie"
    compiler_flags_llvm_aie="$compiler_flags_llvm_aie_aie"
    arch_model_dir="$arch_model_dir_aie"
elif [[ "$aie_version" == "2" ]]; then
    compiler_flags_chess="$compiler_flags_aieml"
    compiler_flags_llvm_aie="$compiler_flags_llvm_aie_aieml"
    arch_model_dir="$arch_model_dir_aieml"
    extra_chess_flag+=" -aiearch aie-ml"
elif [[ "$aie_version" == "5" ]]; then
    compiler_flags_chess="$compiler_flags_aie2ps"
    compiler_flags_llvm_aie="$compiler_flags_llvm_aie_aie2ps"
    arch_model_dir="$arch_model_dir_aie2ps"
    extra_chess_flag+=" -aiearch aie2ps"
else
    echo "Unsupported AIE version: $aie_version"
    exit 1
fi

chess_elf_compiler="xchessmk $extra_chess_flag $silent_flag -s -C Release_LLVM -P $arch_model_dir +P 4 -DDEPLOYMENT_ELF=1 -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST"
linker="${TOOL_PREFIX}ld -EL -r -b binary"

objcopy_tool="${TOOL_PREFIX}objcopy"

# Arch paths
ARCH_DIR=$AIEHLC_DIR/thirdparty/arch/
ARCH_72_DIR=$ARCH_DIR/psv_cortexa72_0/workspace/platform_baremetal/psv_cortexa72_0/standalone_psv_cortexa72_0/bsp/
ARCH_53_DIR=$ARCH_DIR/psv_cortexr5_0/workspace/platform_baremetal/psv_cortexr5_0/standalone_psv_cortexr5_0/bsp/
ARCH_78_DIR=$ARCH_DIR/cortexa78_0/workspace/platform_baremetal/cortexa78_0/standalone_cortexa78_0/bsp/

# A72 
ARCH_APU_A72_INC=${ARCH_72_DIR}/include/
SECONDARY_ARCH_APU_A72_INC=${ARCH_DIR}/psv_cortexa72_0/bsp/include/
ARCH_APU_A72_LIB=${ARCH_72_DIR}/lib/
AIELIB_APU_A72_NAME=libxaienginea72.a
ARCH_APU_A72_LD=$ARCH_DIR/psv_cortexa72_0/lscript.ld

# A78
ARCH_APU_A78_INC=${ARCH_78_DIR}/include/
SECONDARY_ARCH_APU_A78_INC=${ARCH_DIR}/cortexa78_0/bsp/include/
ARCH_APU_A78_LIB=${ARCH_78_DIR}/lib/
AIELIB_APU_A78_NAME=libxaienginea78.a
ARCH_APU_A78_LD=$ARCH_DIR/cortexa78_0/lscript.ld

AIE_GEN=$aie_version
AIE_DRIVER_PARENT_DIR=$AIEHLC_DIR/thirdparty/alib/
AIE_DRIVER_LIB_DIR=$XILINX_VITIS/aietools/include/drivers/aiengine/xaiengine
AIE_DRIVER_INCLUDE_DIR=$XILINX_VITIS/aietools/lib/lnx64.o/

if [[ $aie_version == "1" || $aie_version == "2" ]]; then
    ARCH_APU_ALIB=$ARCH_APU_A72_LIB
    ARCH_APU_AINC=$ARCH_APU_A72_INC
    SECONDARY_ARCH_APU_AINC=$SECONDARY_ARCH_APU_A72_INC
    AIELIB_APU_NAME=$AIELIB_APU_A72_NAME
    ARCH_APU_LD=$ARCH_APU_A72_LD
    hw_lib_flag="-DARMA72_EL3"
    compiler_cpu_flag="-mcpu=cortex-a72"
    EXTRA_LIBS="-lxiltimer,-lxilstandalone,"
elif [[ $aie_version == "5" ]]; then
    ARCH_APU_ALIB=$ARCH_APU_A78_LIB
    ARCH_APU_AINC=$ARCH_APU_A78_INC
    SECONDARY_ARCH_APU_AINC=$SECONDARY_ARCH_APU_A78_INC
    AIELIB_APU_NAME=$AIELIB_APU_A78_NAME
    ARCH_APU_LD=$ARCH_APU_A78_LD
    hw_lib_flag="-DARMA78_EL3"
    compiler_cpu_flag="-mcpu=cortex-a78"
    EXTRA_LIBS="-lxiltimer,-lxilstandalone,"
else
    echo "Unsupported AIE version: $aie_version"
    exit 1
fi
AIENGINE_LIB_DIR=$ARCH_APU_ALIB/../libsrc/build_configs/gen_bsp/libsrc/aienginev2/src

AIEHLC="${AIEHLC_DIR}/build/aiehlc"
if [[ ! -f "${AIEHLC_DIR}/build/aiehlc" ]]; then
    AIEHLC="${AIEHLC_DIR}/release/prebuild/aiehlc"
fi

# compile aie-rt, if the local aie-rt exist
if [ -d "${AIE_DRIVER_PARENT_DIR}/aie-rt/driver/" ] ; then
    USE_LOCAL_AIERT_BSP=1
    build_hw_lib \
        "${ARCH_APU_AINC}" \
        ${TOOL_PREFIX}gcc \
        ${TOOL_PREFIX}as \
        ${TOOL_PREFIX}ar \
        "-O2 -c" \
        "-g -Wall -Wextra -Dversal ${hw_lib_flag} -fno-tree-loop-distribute-patterns" \
        "$AIELIB_APU_NAME" \
        "$ARCH_APU_ALIB" \
        "$ARCH_APU_LD" \
                "$platform"

    if [[ $aie_version == "1" || $aie_version == "2" ]]; then
        EXTRA_LIBS=""
    elif [[ $aie_version == "5" ]]; then
        EXTRA_LIBS="-lxiltimer,-lxilstandalone,"
    else
        echo "aie_version $aie_version is unknown"
    fi
fi

#Set the aie-rt include path FOR aiehlc 
if [ "$USE_LOCAL_AIERT_BSP" -eq 0 ]; then
    BAREMETAL_AIENGINE_INCLUDE=$XILINX_VITIS_AIETOOLS/include/drivers/aiengine/
else
    BAREMETAL_AIENGINE_INCLUDE=${AIEHLC_DIR}/thirdparty/alib/include/
fi

dbg_echo $BAREMETAL_AIENGINE_INCLUDE
echo -e "\nStarting Merged File Compilation..."
echo -e "    ${runtime_source_file}\n"
#Convert the host&kernel merged source code
if [[ "$use_llvm_aie" == "true" ]]; then
    "$LD_SO" --library-path "${LIB_PATH}:${LIB_BASE_PATH}" "${AIEHLC}" --use-llvm-aie --extra-arg="-DAIE_GEN=${aie_version}" \
        --extra-arg="-I${ARCH_APU_AINC}" --extra-arg="-I${SECONDARY_ARCH_APU_AINC}" --extra-arg="-I$BAREMETAL_AIENGINE_INCLUDE" \
        --extra-arg="-I$XILINX_VITIS_AIETOOLS/include" --extra-arg="-I${CLANG_INCLUDE_PATH}" --extra-arg="-I${AIEHLC_DIR}/include/llvm" \
        --extra-arg="-include"aie_compat.h"" ${runtime_source_file} --
else
    "$LD_SO" --library-path "${LIB_PATH}:${LIB_BASE_PATH}" "${AIEHLC}" --extra-arg="-DAIE_GEN=${aie_version}" \
        --extra-arg="-I${ARCH_APU_AINC}" --extra-arg="-I${SECONDARY_ARCH_APU_AINC}" --extra-arg="-I$BAREMETAL_AIENGINE_INCLUDE" \
        --extra-arg="-I$XILINX_VITIS_AIETOOLS/include" --extra-arg="-I${CLANG_INCLUDE_PATH}" --extra-arg="-I${AIEHLC_DIR}/include/llvm" \
        --extra-arg="-include"aie_compat.h"" ${runtime_source_file} --
fi

HOST_BUILD_DIR=$(pwd)/aout/
mkdir -p $HOST_BUILD_DIR

# compile kernels
kernel_list_file="$(pwd)/aout/kernel_list"
final_obj_file="$(pwd)/aout/build/final_kernel.o"
temp_obj_files=()
kernel_names=()

rm -f $final_obj_file

echo -e "\nStarting Kernel Compilation..."
while IFS= read -r kernel_source_file; do
    kernel_names+=("$kernel_source_file")
    func_name=$(basename "$kernel_source_file" .cc)

    KERNEL_SRC=$KERNEL_DIR/kernelcfg/${kernel_source_file}
    KERNEL_BUILD_DIR=$(pwd)/aout/build/${kernel_source_file}/obj/
    obj_file="${KERNEL_BUILD_DIR}/kernel.o"
    temp_obj_files+=("$obj_file")

    mkdir -p "$KERNEL_BUILD_DIR"

    if [[ "$use_llvm_aie" != "true" ]]; then
        echo "Compiling Kernel (using chess): $kernel_source_file"

        xchesscc $compiler_flags_chess -o "${KERNEL_BUILD_DIR}/kernel_orig.ll" "$KERNEL_SRC/wrapper.cc"
        
        run_cmd "$XILINX_VITIS_AIETOOLS/lnx64.o/tools/clang/bin/opt -S -load-pass-plugin="$XILINX_VITIS_AIETOOLS/lib/lnx64.o/libLLVMXLOpt.so" -passes=xlopt "${KERNEL_BUILD_DIR}/kernel_orig.ll" -o "${KERNEL_BUILD_DIR}/kernel.ll""
        run_cmd "$XILINX_VITIS_AIETOOLS/lnx64.o/tools/clang/bin/opt -S -load-pass-plugin="$XILINX_VITIS_AIETOOLS/lib/lnx64.o/libLLVMXLOpt.so" -passes=xlopt "${KERNEL_BUILD_DIR}/kernel.ll" -o "${KERNEL_BUILD_DIR}/kernel.ll""

        dbg_echo $chess_elf_compiler +o "$KERNEL_BUILD_DIR" "$KERNEL_SRC/aieml.prx"
        $chess_elf_compiler +o "$KERNEL_BUILD_DIR" "$KERNEL_SRC/aieml.prx"
    else
        echo "Compiling Kernel (using llvm-aie): $kernel_source_file"

        dbg_echo ${LLVM_AIE_PATH}/bin/clang++ $compiler_flags_llvm_aie "$KERNEL_SRC/wrapper.cc" -Wl,-T $KERNEL_SRC/main.ld.script -o $KERNEL_BUILD_DIR/kernel
        ${LLVM_AIE_PATH}/bin/clang++ $compiler_flags_llvm_aie "$KERNEL_SRC/wrapper.cc" -Wl,-T $KERNEL_SRC/main.ld.script -o $KERNEL_BUILD_DIR/kernel
    fi

    dbg_echo $linker -o "$obj_file" "$KERNEL_BUILD_DIR/kernel"
    $linker -o "$obj_file" "$KERNEL_BUILD_DIR/kernel"
    
    redefine_symbols "$obj_file" "$func_name" "$objcopy_tool"
done < "$kernel_list_file"

#Compile host
host_file=$(pwd)/aout/host.cc
dbg_echo "INCLUDE_DIR: ${AIE_DRIVER_PARENT_DIR}include/"
dbg_echo "LIB_DIR: ${AIE_DRIVER_PARENT_DIR}lib/"
dbg_echo "ARCH_APU_AINC: $ARCH_APU_AINC"
dbg_echo "SECONDARY_ARCH_APU_AINC: $SECONDARY_ARCH_APU_AINC"
dbg_echo "ARCH_APU_ALIB: $ARCH_APU_ALIB"
dbg_echo "ARCH_APU_LD: ${ARCH_APU_LD}"
dbg_echo -e "OBJ FILES: ${temp_obj_files[@]}\n"
dbg_echo "AIENGINE_LIB_DIR: ${AIENGINE_LIB_DIR}"

# when use local aiert_bsp then set the local aiengine lib name
if [ "$USE_LOCAL_AIERT_BSP" -eq 0 ]; then
    BAREMETAL_AIENGINE_LIB=aienginev2
else
    BAREMETAL_AIENGINE_LIB=${AIELIB_APU_NAME:3:-2}
fi
dbg_echo "BAREMETAL_AIENGINE_LIB: ${BAREMETAL_AIENGINE_LIB}"

echo -e "\n"
echo "Targeting $platform platform..."
echo "Compiling host..."
echo "    $host_file"
echo "Linking kernels..."
echo "    ${temp_obj_files[@]}"
# opt_flag="-O3"
opt_flag="-Os"
if [[ "$platform" == "baremetal" ]]; then
    dbg_echo ${TOOL_PREFIX}g++ $opt_flag -D__HOST_CODE_ONLY__ -L$XILINX_VITIS/aietools/lib/lnx64.o/ -L$AIENGINE_LIB_DIR -DAIE_GEN=${aie_version} ${compiler_cpu_flag} -Wl,-T -Wl,${ARCH_APU_LD} -I$ARCH_APU_AINC -I$SECONDARY_ARCH_APU_AINC -I$AIE_DRIVER_PARENT_DIR/include/ -L$ARCH_APU_ALIB -L$AIE_DRIVER_PARENT_DIR/lib/ -o $HOST_BUILD_DIR/main.elf $host_file ${temp_obj_files[@]} -Wl,--start-group,-lm,-l${BAREMETAL_AIENGINE_LIB},-lxil,-lgcc,-lc,-lstdc++,${EXTRA_LIBS}--end-group
    ${TOOL_PREFIX}g++ $opt_flag -D__HOST_CODE_ONLY__ -L$XILINX_VITIS/aietools/lib/lnx64.o/ -L$AIENGINE_LIB_DIR -DAIE_GEN=${aie_version} ${compiler_cpu_flag} -Wl,-T -Wl,${ARCH_APU_LD} -I$ARCH_APU_AINC -I$SECONDARY_ARCH_APU_AINC -I$AIE_DRIVER_PARENT_DIR/include/ -L$ARCH_APU_ALIB -L$AIE_DRIVER_PARENT_DIR/lib/ -o $HOST_BUILD_DIR/main.elf $host_file ${temp_obj_files[@]} -Wl,--start-group,-lm,-l${BAREMETAL_AIENGINE_LIB},-lxil,-lgcc,-lc,-lstdc++,${EXTRA_LIBS}--end-group
elif [[ "$platform" == "linux" ]]; then
    dbg_echo ${TOOL_PREFIX}g++ $opt_flag -D__AIELINUX__ -D__HOST_CODE_ONLY__ -DAIE_GEN=${aie_version} ${compiler_cpu_flag} \
        -I${AIE_DRIVER_DIR}/include -I$ARCH_APU_AINC -I$SECONDARY_ARCH_APU_AINC -I$AIE_DRIVER_PARENT_DIR/include/ \
        -L${AIE_DRIVER_DIR}/src -L$ARCH_APU_ALIB -L$AIE_DRIVER_PARENT_DIR/lib/ -L$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src/ \
        -o $HOST_BUILD_DIR/main.elf $host_file ${temp_obj_files[@]} \
        -Wl,--start-group,-lxaiengine,-lxil,--end-group
    ${TOOL_PREFIX}g++ $opt_flag -D__AIELINUX__ -D__HOST_CODE_ONLY__ -DAIE_GEN=${aie_version} ${compiler_cpu_flag} \
        -I${AIE_DRIVER_DIR}/include -I$ARCH_APU_AINC -I$SECONDARY_ARCH_APU_AINC -I$AIE_DRIVER_PARENT_DIR/include/ \
        -L${AIE_DRIVER_DIR}/src -L$ARCH_APU_ALIB -L$AIE_DRIVER_PARENT_DIR/lib/ -L$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src/  \
        -o $HOST_BUILD_DIR/main.elf $host_file ${temp_obj_files[@]} \
        -Wl,--start-group,-lxaiengine,-lxil,--end-group
fi

echo "Stripping extra ELF symbols..."
${TOOL_PREFIX}strip $HOST_BUILD_DIR/main.elf

echo "Build complete."
echo "    $HOST_BUILD_DIR/main.elf"