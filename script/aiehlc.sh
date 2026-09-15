#!/usr/bin/env bash  
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

dbg_echo() {
    if [ "$DEBUG_OUTPUT" = 1 ]; then
        echo "$@"
    fi
}

run_cmd() {
    local command="$1"
    if [ "$DEBUG_OUTPUT" = 0 ]; then
        $command > /dev/null
    else
        $command
    fi
}

usage() {
    echo "Usage: $0 --runtime-source-file <path> --aie-version <version> [--kernel-count <count>] [--kernel <source> [<directory>]] [--aielib-only] [--prettydebug] [-gdb] [--profiling] [--skip-bss]"
    echo "  --platform    baremetal (default) | sim | linux | baremetal_debug"
    echo "                'baremetal_debug' builds a baremetal ELF and writes sim_config.sh so the debug"
    echo "                UI can run either a hardware test or a simulator run from one build."
    echo "  --profiling   build the host with the PMU profiling layer (-DAIEHLC_PROFILING=1)"
    echo "  --skip-bss    skip .bss zero-init on kernel load (-DAIEHLC_SKIP_BSS_DEFAULT=1)"
    return 1
}

# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sim/write_sim_config.sh"

prepare_raw_xaie_debug_view() {
    local host_src="$1"
    local worklocal="$2"
    local app_src="$3"
    local aie_version="$4"
    local platform="$5"
    local host_elf="${6:-}"
    local artifacts_dir
    artifacts_dir="$(dirname "$host_src")"

    mkdir -p "$worklocal"
    cp -f "$host_src" "$worklocal/host.cc" 2>/dev/null || true
    echo "$app_src" > "$worklocal/app_source.txt" 2>/dev/null || true

    echo "Generating static provenance (raw-XAie flow)..."
    if ! python3 "${AIEHLC_DIR}/src/tool/debug/xaiehost2provenance.py" \
            "$worklocal/host.cc" --out-dir "$worklocal" \
            --aie-version "$aie_version" --platform "$platform" \
            --artifacts-dir "$artifacts_dir"; then
        echo "    warning: provenance generation failed (non-fatal)."
        return 0
    fi
    if [ ! -f "$worklocal/dfscheduleprovenancemap.json" ]; then
        return 0
    fi

    echo "Generating readable schedule view..."
    if ! python3 "${AIEHLC_DIR}/src/tool/debug/schedule_view.py" "$worklocal"; then
        echo "    warning: schedule_view.py failed (non-fatal); skipping view."
        return 0
    fi
    echo "    $worklocal/host_schedule.html"

    if [ "$PRETTY_DEBUG" -ne 1 ]; then
        return 0
    fi

    echo "Launching live schedule-debug server (--prettydebug)..."
    local server_args=(--app "$worklocal" --aie-version "$aie_version" --open)
    if [ "$platform" = "sim" ]; then
        server_args+=(--sim-only)
    elif [ -n "$host_elf" ] && [ -f "$host_elf" ]; then
        server_args+=(--elf "$host_elf")
    fi
    python3 "${AIEHLC_DIR}/src/tool/debug/schedule_debug_server.py" \
        "${server_args[@]}"
}

check_bsp_environment() {
    # Verify the one-time BSP setup (source ./script/setup.sh) has been run.
    # $1 = target aie_version, used to pick the version-specific BSP dir.
    local ver="$1"
    local proc
    case "$ver" in
        1|2) proc="psv_cortexa72_0" ;;
        5)   proc="cortexa78_0" ;;
        *)   proc="" ;;   # unknown version handled later by existing check
    esac
    [ -z "$proc" ] && return 0

    local bsp_ws="${AIEHLC_DIR}/thirdparty/arch/${proc}/workspace"
    if [ ! -d "$bsp_ws" ]; then
        echo ""
        echo "==========================================================="
        echo "Error: AIE BSP not found -- environment is not set up."
        echo "  Missing: ${bsp_ws}"
        echo ""
        echo "Please run the one-time environment setup first:"
        echo "    source ./script/setup.sh"
        echo "==========================================================="
        return 1
    fi
    return 0
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
                
            elif [[ "$lplatform" == "baremetal" ]]; then
                bash -c "cd $AIE_DRIVER_PARENT_DIR/aie-rt/driver/src; CC=\${CC:-gcc} CXX=\${CXX:-g++} make -f Makefile.Linux"
                ##keep the order, first copy bsp header, then copy the aiert header, then even bsp header have old aiegnine header
                ##we still get the latest aiert header
                cp -rf "$header_dir"/*.h "$AIE_DRIVER_PARENT_DIR/include/"
                cp -rf "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/internal/"* "$AIE_DRIVER_PARENT_DIR/include/"

                dbg_echo "[INFO] Cleaning previous build..."
                make clean -C "$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src/"

                dbg_echo "[INFO] Copying extra libs and ld script..."
                cp "$lib_dir"/*.a "$AIE_DRIVER_PARENT_DIR/lib/"
                cp "$ld_dir" "$AIE_DRIVER_PARENT_DIR/lib/"

                if [ -f "$AIE_DRIVER_PARENT_DIR/lib/libxil.a" ]; then
                    aie_bsp_objs=$("$archiver" t "$AIE_DRIVER_PARENT_DIR/lib/libxil.a" 2>/dev/null | grep -E '^xaie.*\.obj$' || true)
                    if [ -n "$aie_bsp_objs" ]; then
                        dbg_echo "[INFO] Stripping BSP aienginev2 objects from libxil.a (local aie-rt is authoritative)"
                        "$archiver" d "$AIE_DRIVER_PARENT_DIR/lib/libxil.a" $aie_bsp_objs
                    fi
                fi

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


main() {
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AIEHLC_DIR="${SCRIPT_DIR}/../"

# aiedbg pip install PATH (written by ensure_aiedbg.py on first bootstrap)
if [[ -f "${AIEHLC_DIR}.aiehlc/aiedbg_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${AIEHLC_DIR}.aiehlc/aiedbg_env.sh"
fi
runtime_source_file=""
aie_version="2"
use_llvm_aie="false"
DEBUG_OUTPUT=0
DEBUG_SYMS=0
platform="baremetal"
COMPILE_AIELIB_ONLY=0
USE_LOCAL_AIERT_BSP=0
PRETTY_DEBUG=0
SIM_TILES=""
GDB_MODE=0
# Reset every run: this script is sourced, so a flag from a previous invocation
# in the same shell would otherwise persist.
PROFILING=0
SKIP_BSS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -help)
            usage
            return 0
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
        --sim-tiles)
            SIM_TILES="$2"
            shift 2
            ;;
        --debug-output)
            DEBUG_OUTPUT=1
            shift
            ;;
        --debug-syms)
            DEBUG_SYMS=1
            shift
            ;;
        --aielib-only)
            COMPILE_AIELIB_ONLY=1
            shift
            ;;
        --prettydebug)
            PRETTY_DEBUG=1
            shift
            ;;
        -gdb|--gdb)
            GDB_MODE=1
            shift
            ;;
        --profiling)
            PROFILING=1
            shift
            ;;
        --skip-bss)
            SKIP_BSS=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            return 1
            ;;
    esac
done

# Parameter validation
if [ "$COMPILE_AIELIB_ONLY" -eq 0 ] && [ -z "$runtime_source_file" ]; then
    echo "Error: --runtime-source-file is required."
    usage
    return 0
fi

# Convert runtime_source_file to absolute path (resolve relative to AIEHLC_DIR)
if [ -n "$runtime_source_file" ] && [[ "$runtime_source_file" != /* ]]; then
    runtime_source_file="$(cd "${AIEHLC_DIR}" && cd "$(dirname "$runtime_source_file")" && pwd)/$(basename "$runtime_source_file")"
fi

#set up env
run_cmd "source $SCRIPT_DIR/setup.sh --path-set-only"

if [[ "$platform" == "sim" ]] && [ -n "$runtime_source_file" ]; then
    if ! grep -q '__global__' "$runtime_source_file" 2>/dev/null; then
        echo "[sim] No __global__ kernel found — writing sim config (no auto-run)..."
        run_cmd "source $SCRIPT_DIR/setup.sh --path-set-only"
        rm -f  "${SCRIPT_DIR}/sim/build/aiehlc_ps.so" \
               "${SCRIPT_DIR}/sim/build/aiehlc_ps_wrapper.o" \
               "${SCRIPT_DIR}/sim/build/kernel_elf_init."{cc,o} \
               "${SCRIPT_DIR}/sim/build/"*.o
        rm -rf "${SCRIPT_DIR}/sim/build/kernel_"*
        write_sim_config "$(pwd)/aout" "$runtime_source_file" "" "" "$aie_version" "$SIM_TILES"
        return 0
    fi
    echo "[sim] __global__ kernel detected — running aiehlc extraction first..."
fi

# Verify one-time BSP setup was done (source ./script/setup.sh).
check_bsp_environment "$aie_version" || return 1

if [[ "$platform" == "linux" ]]; then
    TOOL_PREFIX="aarch64-linux-gnu-"
elif [[ "$platform" == "baremetal" || "$platform" == "baremetal_debug" ]]; then
    TOOL_PREFIX="aarch64-none-elf-"
elif [[ "$platform" == "sim" ]]; then
    TOOL_PREFIX="aarch64-none-elf-"
fi

KERNEL_DIR=$(pwd)/aout/
rm -rf $KERNEL_DIR
# Also clean build/aout/ if it exists (stale from previous runs with different CWD)
[ -d "$(pwd)/build/aout" ] && rm -rf "$(pwd)/build/aout"
export XILINX_VITIS_AIETOOLS=$XILINX_VITIS/aietools
CARDANO_AIE_ARCH_MODEL_DIR="$XILINX_VITIS_AIETOOLS/data/versal_prod/lib"

# Check if local thirdparty/alib/include has xaiengine headers
LOCAL_ALIB_INCLUDE="${AIEHLC_DIR}/thirdparty/alib/include"
echo "Local include directory: ${LOCAL_ALIB_INCLUDE}"
if [ -d "${LOCAL_ALIB_INCLUDE}/xaiengine" ] && [ "$(ls -A ${LOCAL_ALIB_INCLUDE}/xaiengine 2>/dev/null)" ]; then
    echo "Using local aie-rt headers from ${LOCAL_ALIB_INCLUDE}"
    AIETOOLS_INCLUDE_BASE="${LOCAL_ALIB_INCLUDE}"
else
    echo "Using Vitis aie-rt headers from $XILINX_VITIS_AIETOOLS//include/drivers/aiengine"
    AIETOOLS_INCLUDE_BASE="$XILINX_VITIS_AIETOOLS//include/drivers/aiengine"
fi

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
    AIE_ARCH_MACRO=10
elif [[ $aie_version == "5" ]]; then
    ARCH_APU_ALIB=$ARCH_APU_A78_LIB
    ARCH_APU_AINC=$ARCH_APU_A78_INC
    SECONDARY_ARCH_APU_AINC=$SECONDARY_ARCH_APU_A78_INC
    AIELIB_APU_NAME=$AIELIB_APU_A78_NAME
    ARCH_APU_LD=$ARCH_APU_A78_LD
    hw_lib_flag="-DARMA78_EL3 -DSDT"
    compiler_cpu_flag="-mcpu=cortex-a78"
    EXTRA_LIBS="-lxiltimer,-lxilstandalone,-lxilpm_ng,"
    AIE_ARCH_MACRO=20
else
    echo "Unsupported AIE version: $aie_version"
    return 1
fi
AIENGINE_LIB_DIR=$ARCH_APU_ALIB/../libsrc/build_configs/gen_bsp/libsrc/aienginev2/src

AIEHLC="${AIEHLC_DIR}/build/aiehlc"
if [[ ! -f "${AIEHLC_DIR}/build/aiehlc" ]]; then
    AIEHLC="${AIEHLC_DIR}/release/prebuild/aiehlc"
fi

# compile aie-rt, if the local aie-rt exist
if [ -d "${AIE_DRIVER_PARENT_DIR}/aie-rt/driver/" ] && [[ "$platform" != "sim" ]]; then
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
        EXTRA_LIBS="-lxiltimer,-lxilstandalone,-lxilpm_ng,"
    else
        echo "aie_version $aie_version is unknown"
    fi
fi

if [ "$COMPILE_AIELIB_ONLY" -eq 1 ]; then
    echo "aielib compilation complete (--aielib-only)."
    return 0
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
# Add the source file's directory to include path so relative includes resolve
SOURCE_DIR="$(cd "$(dirname "${runtime_source_file}")" && pwd)"
AIEHLC_ERRLOG=$(mktemp /tmp/aiehlc_err.XXXXXX)
# Simulator builds need the __AIESIM__ define passed through to clang.
AIEHLC_SIM_DEFINE=""
if [[ "$platform" == "sim" ]]; then
    AIEHLC_SIM_DEFINE='--extra-arg=-D__AIESIM__'
fi

# Build the aiehlc argument list once, then run it either directly through the
# custom loader or under gdb (-gdb). This keeps the two invocation paths in sync.
AIEHLC_ARGS=()
if [[ "$use_llvm_aie" == "true" ]]; then
    AIEHLC_ARGS+=(--use-llvm-aie)
fi
AIEHLC_ARGS+=(
    --extra-arg="-DAIE_GEN=${aie_version}"
    --extra-arg="-D__AIE_ARCH__=${AIE_ARCH_MACRO}"
    --extra-arg="-I${AIETOOLS_INCLUDE_BASE}" --extra-arg="-I$BAREMETAL_AIENGINE_INCLUDE"
    --extra-arg="-I${ARCH_APU_AINC}" --extra-arg="-I${SECONDARY_ARCH_APU_AINC}"
    --extra-arg="-I$XILINX_VITIS_AIETOOLS/include" --extra-arg="-I${CLANG_INCLUDE_PATH}" --extra-arg="-I${AIEHLC_DIR}/include/llvm"
    --extra-arg="-I${SOURCE_DIR}" --extra-arg="-I${AIEHLC_DIR}/src/mlir/runtime"
    --extra-arg="-I${AIEHLC_DIR}/include"
)
if [ -n "$AIEHLC_SIM_DEFINE" ]; then
    AIEHLC_ARGS+=("$AIEHLC_SIM_DEFINE")
fi

if [ "$GDB_MODE" -eq 1 ]; then
    # Debug aiehlc under gdb. aiehlc is passed directly as the *exec file* so gdb
    # loads its symbols ('i files' shows aiehlc) AND breakpoints actually trigger.
    #
    # NOTE: do NOT use 'set exec-wrapper $LD_SO --library-path ...' here. Launching
    # aiehlc through the explicit SDK loader breaks gdb's breakpoint handling
    # ("During startup program exited normally" - no breakpoint ever hits, even
    # main). On this host aiehlc runs fine under the system loader, so gdb runs it
    # directly. If a future host truly needs the SDK glibc, run the direct-loader
    # path (non -gdb) instead, or patchelf the binary's PT_INTERP/RPATH.
    echo "Launching aiehlc under gdb (-gdb)..."
    echo "  Tip: 'run' to start, 'bt' on crash, 'break <fn>' before run. aiehlc args are pre-set."
    dbg_echo "+ gdb --args ${AIEHLC} ${AIEHLC_ARGS[*]} ${runtime_source_file} --"
    gdb --args "${AIEHLC}" \
        "${AIEHLC_ARGS[@]}" \
        ${runtime_source_file} --
    AIEHLC_RC=$?
else
    dbg_echo "+ $LD_SO --library-path ${LIB_PATH}:${LIB_BASE_PATH} ${AIEHLC} ${AIEHLC_ARGS[*]} ${runtime_source_file} --"
    "$LD_SO" --library-path "${LIB_PATH}:${LIB_BASE_PATH}" "${AIEHLC}" \
        "${AIEHLC_ARGS[@]}" \
        ${runtime_source_file} -- 2> >(tee "$AIEHLC_ERRLOG" >&2)
    AIEHLC_RC=$?
fi
if [ $AIEHLC_RC -ne 0 ]; then
    echo ""
    echo "==========================================================="
    echo "Error: aiehlc code generation failed (exit code $AIEHLC_RC)."
    echo "==========================================================="
    if [ -s "$AIEHLC_ERRLOG" ]; then
        echo ""
        echo "--- Error details ---"
        cat "$AIEHLC_ERRLOG"
        echo "--- End of error details ---"
    fi
    rm -f "$AIEHLC_ERRLOG"
    return $AIEHLC_RC
fi
rm -f "$AIEHLC_ERRLOG"

# -gdb is for debugging aiehlc / inspecting the generated IR only. Once the gdb
# session ends, skip the rest of the flow (tiling delegation, kernel + host
# compilation, prettydebug). The IR is already emitted under aout/.
if [ "$GDB_MODE" -eq 1 ]; then
    echo "gdb session finished (-gdb): skipping kernel/host build. Generated IR is under aout/."
    return 0
fi

HOST_BUILD_DIR=$(pwd)/aout/
mkdir -p $HOST_BUILD_DIR


_host_cc="$(pwd)/aout/host.cc"
if [ -f "$_host_cc" ]; then
    for _kdir in "$(pwd)/aout/kernelcfg"/*/; do
        _kname="$(basename "$_kdir")"
        _dm="${_kdir}dm_offsets.h"
        if [ -f "$_dm" ] && ! grep -q "#include.*dm_offsets.h" "$_host_cc"; then
            sed -i "1s|^|#include \"kernelcfg/${_kname}/dm_offsets.h\"\n|" "$_host_cc"
            echo "[aiehlc] Injected kernelcfg/${_kname}/dm_offsets.h into host.cc"
        fi
    done
fi

# Check if tiling pipeline generated output in worklocal/
if [ -f "${HOST_BUILD_DIR}/worklocal/host.cc" ]; then
    echo -e "\n[TilingLinalg] Detected multi-tile pipeline output in aout/worklocal/"
    echo "Delegating to hostcompile.sh for tiling mode compilation..."

    # Build directly in aout/worklocal/ — no copy to the unitest sandbox.
    WORKLOCAL_DIR="${HOST_BUILD_DIR}/worklocal"

    # Copy user headers directly from SOURCE_DIR to ensure freshness
    # (guards against stale definitions, e.g. M/K/N).
    for f in "${SOURCE_DIR}"/*.h; do
        [ -f "$f" ] && cp -f "$f" "${WORKLOCAL_DIR}/$(basename "$f")"
    done

    echo "${runtime_source_file}" > "${WORKLOCAL_DIR}/app_source.txt"

    # Convert the routing resource map JSON into a C header compiled into the host.
    # Non-fatal: header generation must never break the build (set -e is active).
    if [ -f "${WORKLOCAL_DIR}/routingresourcemap.json" ]; then
        echo "Generating routing resource map header..."
        if python3 "${AIEHLC_DIR}/src/tool/debug/resource_json_to_header.py" \
                "${WORKLOCAL_DIR}/routingresourcemap.json" \
                --out "${WORKLOCAL_DIR}/aie_resource_map.h"; then
            echo "    ${WORKLOCAL_DIR}/aie_resource_map.h"
        else
            echo "    warning: resource_json_to_header.py failed (non-fatal); skipping header."
        fi
    fi

    # Run hostcompile.sh with WORKLOCAL_DIR pointing at aout/worklocal.
    hc_extra_defs="${EXTRA_DEFS:-}"
    if [ "${SKIP_BSS}" -eq 1 ]; then
        hc_extra_defs="${hc_extra_defs} -DAIEHLC_SKIP_BSS_DEFAULT=1"
    fi
    WORKLOCAL_DIR="${WORKLOCAL_DIR}" AIE_VERSION="${aie_version}" PLATFORM="${platform}" \
        SIM_TILES="${SIM_TILES:-}" DEBUG_SYMS="${DEBUG_SYMS}" \
        AIEHLC_PROFILING="${PROFILING}" EXTRA_DEFS="${hc_extra_defs}" \
        bash "${AIEHLC_DIR}/script/hostcompile.sh"
    TILING_RC=$?
    if [ $TILING_RC -ne 0 ]; then
        echo "Error: tiling mode compilation failed."
        return $TILING_RC
    fi

    # Copy the built ELF to the standard output location
    TILING_BUILD_DIR="${WORKLOCAL_DIR}/build"
    if [ -f "${TILING_BUILD_DIR}/host" ]; then
        cp -f "${TILING_BUILD_DIR}/host" "${HOST_BUILD_DIR}/main.elf"
        echo "Build complete (tiling mode)."
        echo "    ${HOST_BUILD_DIR}/main.elf"
    fi

    # Generate the readable schedule view (schedule_view.json + host_schedule.html).
    # Non-fatal: view generation must never break the build (set -e is active).
    if [ -f "${WORKLOCAL_DIR}/dfscheduleprovenancemap.json" ] && [ -f "${WORKLOCAL_DIR}/host.cc" ]; then
        echo "Generating readable schedule view..."
        if python3 "${AIEHLC_DIR}/src/tool/debug/schedule_view.py" "${WORKLOCAL_DIR}"; then
            echo "    ${WORKLOCAL_DIR}/host_schedule.html"
        else
            echo "    warning: schedule_view.py failed (non-fatal); skipping view."
        fi
    fi

    # --prettydebug: launch the live schedule-debug server (stdlib http.server) and
    # open the browser so the user can deploy/test + inspect tile DMA/core/event status
    # live. Runs in the foreground for the interactive debug session; Ctrl-C to stop.
    if [ "$PRETTY_DEBUG" -eq 1 ]; then
        if [ -f "${WORKLOCAL_DIR}/host_schedule.html" ]; then
            if ! command -v aiedbg &>/dev/null; then
                echo "aiedbg not found — bootstrapping (clone + pip install)..."
                python3 "${AIEHLC_DIR}src/tool/debug/ensure_aiedbg.py" \
                    --repo-root "${AIEHLC_DIR}" || \
                    echo "    warning: aiedbg bootstrap failed; live HW reads disabled."
                if [[ -f "${AIEHLC_DIR}.aiehlc/aiedbg_env.sh" ]]; then
                    # shellcheck disable=SC1091
                    source "${AIEHLC_DIR}.aiehlc/aiedbg_env.sh"
                fi
            fi
            echo "Launching live schedule-debug server (--prettydebug)..."
            PD_ELF_ARGS=()
            if [ -f "${HOST_BUILD_DIR}/main.elf" ]; then
                PD_ELF_ARGS=(--elf "${HOST_BUILD_DIR}/main.elf")
            fi
            python3 "${AIEHLC_DIR}/src/tool/debug/schedule_debug_server.py" \
                "${WORKLOCAL_DIR}" \
                "${PD_ELF_ARGS[@]}" \
                --aie-version "${aie_version}" \
                --open
        else
            echo "    warning: --prettydebug requested but host_schedule.html missing; skipping server."
        fi
    fi
    return 0
fi

# compile kernels
kernel_list_file="$(pwd)/aout/kernel_list"
final_obj_file="$(pwd)/aout/build/final_kernel.o"
temp_obj_files=()
kernel_names=()

rm -f $final_obj_file

_kc_platform="$platform"
[[ "$platform" == "baremetal_debug" ]] && _kc_platform="baremetal"

echo -e "\nStarting Kernel Compilation..."
while IFS= read -r kernel_source_file; do
    kernel_names+=("$kernel_source_file")
    func_name=$(basename "$kernel_source_file" .cc)

    KERNEL_SRC=$KERNEL_DIR/kernelcfg/${kernel_source_file}
    KERNEL_BUILD_DIR=$(pwd)/aout/build/${kernel_source_file}/obj/
    obj_file="${KERNEL_BUILD_DIR}/kernel.o"
    temp_obj_files+=("$obj_file")

    compile_args=(
        --kernel-cc "$KERNEL_SRC/wrapper.cc"
        --output-dir "$KERNEL_BUILD_DIR"
        --func-name "$func_name"
        --aie-version "$aie_version"
        --platform "$_kc_platform"
        --include-base "$AIETOOLS_INCLUDE_BASE"
    )

    if [[ "$use_llvm_aie" == "true" && "$_kc_platform" != "sim" ]]; then
        compile_args+=(--use-llvm-aie --ld-script "$KERNEL_SRC/main.ld.script")
    elif [[ "$_kc_platform" == "sim" ]]; then
        compile_args+=(--prx "$KERNEL_SRC/aieml.prx" --commons-dir "$KERNEL_BUILD_DIR")
    else
        compile_args+=(--prx "$KERNEL_SRC/aieml.prx" --commons-dir "$KERNEL_DIR/TheHouseOfCommons/")
    fi

    if [ "$DEBUG_OUTPUT" = 1 ]; then
        compile_args+=(--debug-output)
    fi

    source "$SCRIPT_DIR/kc.sh" "${compile_args[@]}"
    KC_RC=$?
    if [ $KC_RC -ne 0 ]; then
        echo ""
        echo "==========================================================="
        echo "Error: kernel compilation failed for $func_name (exit code $KC_RC)."
        echo "==========================================================="
        return $KC_RC
    fi
    if [ ! -f "$obj_file" ]; then
        echo ""
        echo "==========================================================="
        echo "Error: kernel object not produced: $obj_file"
        echo "==========================================================="
        return 1
    fi
done < "$kernel_list_file"

sim_obj_files=()
if [[ "$platform" == "baremetal_debug" ]]; then
    echo -e "\n[baremetal_debug] Compiling kernels for simulator..."
    while IFS= read -r kernel_source_file; do
        func_name=$(basename "$kernel_source_file" .cc)
        KERNEL_SRC=$KERNEL_DIR/kernelcfg/${kernel_source_file}
        KERNEL_BUILD_DIR_SIM=$(pwd)/aout/build/${kernel_source_file}/obj_sim/
        obj_file_sim="${KERNEL_BUILD_DIR_SIM}/kernel.o"
        sim_obj_files+=("$obj_file_sim")

        compile_args_sim=(
            --kernel-cc "$KERNEL_SRC/wrapper.cc"
            --output-dir "$KERNEL_BUILD_DIR_SIM"
            --func-name "$func_name"
            --aie-version "$aie_version"
            --platform "sim"
            --include-base "$AIETOOLS_INCLUDE_BASE"
            --prx "$KERNEL_SRC/aieml.prx"
            --commons-dir "$KERNEL_BUILD_DIR_SIM"
        )
        [ "$DEBUG_OUTPUT" = 1 ] && compile_args_sim+=(--debug-output)

        source "$SCRIPT_DIR/kc.sh" "${compile_args_sim[@]}"
        KC_RC=$?
        if [ $KC_RC -ne 0 ]; then
            echo ""
            echo "==========================================================="
            echo "Error: sim kernel compilation failed for $func_name (exit code $KC_RC)."
            echo "==========================================================="
            return $KC_RC
        fi
    done < "$kernel_list_file"
fi

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

# Full src/mlir/runtime set to link into the single-kernel (raw-XAie) host.
# Previously only aie_runtime_common.c was linked; the tiling flow
# (script/hostcompile.sh) links all four. Kept in ONE array so the baremetal
# and linux link branches below cannot drift. The raw aout/host.cc references
# no __Runtime_* symbols, so -Wl,--gc-sections (added on the links) dead-strips
# the unused runtime, including __Runtime_routing_init's unconditional call to
# routing() — a symbol defined only in the tiling flow's routing.cc. As a
# belt-and-suspenders for any future single-kernel host.cc that DOES call the
# routing path, emit a weak no-op routing() stub (real routing.cc, never
# present here, would still win); no-op is correct because the raw flow does
# its routing inline in test_routing().
RUNTIME_SRCS=(
    "${AIEHLC_DIR}/src/mlir/runtime/aie_runtime.c"
    "${AIEHLC_DIR}/src/mlir/runtime/aie_runtime_debug.c"
    "${AIEHLC_DIR}/src/mlir/runtime/aie_runtime_stream_debug.c"
    "${AIEHLC_DIR}/src/mlir/runtime/aie_runtime_common.c"
    "${AIEHLC_DIR}/src/mlir/runtime/aie_runtime_control_plan.c"
    "${AIEHLC_DIR}/src/mlir/runtime/aie_runtime_resource.c"
)
RT_ROUTING_STUB="$(pwd)/aout/rt_routing_stub.c"
cat > "${RT_ROUTING_STUB}" <<'EOF'
#include "xaiengine.h"
/* Weak no-op routing() for the single-kernel (raw-XAie) flow, which does its
 * routing inline in test_routing(). Resolves the unconditional call in
 * aie_runtime.c's __Runtime_routing_init if that path is ever retained. A real
 * routing.cc (tiling flow only) overrides this. */
__attribute__((weak)) void routing(XAie_DevInst *dev) { (void)dev; }
EOF
RUNTIME_SRCS+=("${RT_ROUTING_STUB}")
# Dead-strip unused runtime code; keep C++ std consistent with hostcompile.sh.
RUNTIME_LINK_FLAGS="-std=c++17 -ffunction-sections -fdata-sections"

echo -e "\n"
echo "Targeting $platform platform..."
echo "Compiling host..."
echo "    $host_file"
echo "Linking kernels..."
echo "    ${temp_obj_files[@]}"
# opt_flags="-O2"
opt_flags="-Os"
[ "${DEBUG_SYMS}" -eq 1 ] && opt_flags="${opt_flags} -g"
if [[ "$platform" == "sim" ]]; then
    echo -e "\n[sim] Writing sim config (no auto-run)..."
    rm -f  "${SCRIPT_DIR}/sim/build/aiehlc_ps.so" \
           "${SCRIPT_DIR}/sim/build/kernel_elf_init."{cc,o}
    write_sim_config "$HOST_BUILD_DIR" "$host_file" "${temp_obj_files[*]}" "${kernel_names[*]}" "$aie_version" "$SIM_TILES"
    prepare_raw_xaie_debug_view \
        "$host_file" "${HOST_BUILD_DIR}/worklocal" "$runtime_source_file" \
        "$aie_version" "$platform"
    return 0
elif [[ "$platform" == "baremetal" || "$platform" == "baremetal_debug" ]]; then
    dbg_echo ${TOOL_PREFIX}g++ $opt_flags ${RUNTIME_LINK_FLAGS} -L$XILINX_VITIS/aietools/lib/lnx64.o/ -L$AIENGINE_LIB_DIR -DAIE_GEN=${aie_version} ${compiler_cpu_flag} -Wl,-T -Wl,${ARCH_APU_LD} -I${AIETOOLS_INCLUDE_BASE} -I$AIE_DRIVER_PARENT_DIR/include/ -I$ARCH_APU_AINC -I$SECONDARY_ARCH_APU_AINC -I${SOURCE_DIR} -I${AIEHLC_DIR}/src/mlir/runtime -I${AIEHLC_DIR}/include -L$ARCH_APU_ALIB -L$AIE_DRIVER_PARENT_DIR/lib/ -o $HOST_BUILD_DIR/main.elf $host_file "${RUNTIME_SRCS[@]}" ${temp_obj_files[@]} -Wl,--gc-sections -Wl,--start-group,-lm,-l${BAREMETAL_AIENGINE_LIB},-lxil,-lgcc,-lc,-lstdc++,${EXTRA_LIBS}--end-group
    ${TOOL_PREFIX}g++ $opt_flags ${RUNTIME_LINK_FLAGS} -L$XILINX_VITIS/aietools/lib/lnx64.o/ -L$AIENGINE_LIB_DIR -DAIE_GEN=${aie_version} ${compiler_cpu_flag} -Wl,-T -Wl,${ARCH_APU_LD} -I${AIETOOLS_INCLUDE_BASE} -I$AIE_DRIVER_PARENT_DIR/include/ -I$ARCH_APU_AINC -I$SECONDARY_ARCH_APU_AINC -I${SOURCE_DIR} -I${AIEHLC_DIR}/src/mlir/runtime -I${AIEHLC_DIR}/include -L$ARCH_APU_ALIB -L$AIE_DRIVER_PARENT_DIR/lib/ -o $HOST_BUILD_DIR/main.elf $host_file "${RUNTIME_SRCS[@]}" ${temp_obj_files[@]} -Wl,--gc-sections -Wl,--start-group,-lm,-l${BAREMETAL_AIENGINE_LIB},-lxil,-lgcc,-lc,-lstdc++,${EXTRA_LIBS}--end-group
    GPP_RC=$?
elif [[ "$platform" == "linux" ]]; then
    dbg_echo ${TOOL_PREFIX}g++ $opt_flags ${RUNTIME_LINK_FLAGS} -D__AIELINUX__ -DAIE_GEN=${aie_version} ${compiler_cpu_flag} \
        -I${AIETOOLS_INCLUDE_BASE} -I$AIE_DRIVER_PARENT_DIR/include/ -I${AIE_DRIVER_DIR}/include -I$ARCH_APU_AINC -I$SECONDARY_ARCH_APU_AINC -I${SOURCE_DIR} -I${AIEHLC_DIR}/src/mlir/runtime \
        -L${AIE_DRIVER_DIR}/src -L$ARCH_APU_ALIB -L$AIE_DRIVER_PARENT_DIR/lib/ -L$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src/ \
        -o $HOST_BUILD_DIR/main.elf $host_file "${RUNTIME_SRCS[@]}" ${temp_obj_files[@]} \
        -Wl,--gc-sections -Wl,--start-group,-lxaiengine,-lxil,--end-group
    ${TOOL_PREFIX}g++ $opt_flags ${RUNTIME_LINK_FLAGS} -D__AIELINUX__ -DAIE_GEN=${aie_version} ${compiler_cpu_flag} \
        -I${AIETOOLS_INCLUDE_BASE} -I$AIE_DRIVER_PARENT_DIR/include/ -I${AIE_DRIVER_DIR}/include -I$ARCH_APU_AINC -I$SECONDARY_ARCH_APU_AINC -I${SOURCE_DIR} -I${AIEHLC_DIR}/src/mlir/runtime -I${AIEHLC_DIR}/include \
        -L${AIE_DRIVER_DIR}/src -L$ARCH_APU_ALIB -L$AIE_DRIVER_PARENT_DIR/lib/ -L$AIE_DRIVER_PARENT_DIR/aie-rt/driver/src/  \
        -o $HOST_BUILD_DIR/main.elf $host_file "${RUNTIME_SRCS[@]}" ${temp_obj_files[@]} \
        -Wl,--gc-sections -Wl,--start-group,-lxaiengine,-lxil,--end-group
    GPP_RC=$?
fi
if [ ${GPP_RC:-1} -ne 0 ]; then
    echo ""
    echo "==========================================================="
    echo "Error: host g++ compilation failed (exit code ${GPP_RC})."
    echo "==========================================================="
    return ${GPP_RC}
fi
if [ "${DEBUG_SYMS}" -eq 0 ]; then
    echo "Stripping extra ELF symbols..."
    ${TOOL_PREFIX}strip $HOST_BUILD_DIR/main.elf
fi

echo "Build complete."
echo "    $HOST_BUILD_DIR/main.elf"

if [[ "$platform" == "baremetal_debug" ]]; then
    echo -e "\n[baremetal_debug] Writing sim config for simulator run..."
    rm -f  "${SCRIPT_DIR}/sim/build/aiehlc_ps.so" \
           "${SCRIPT_DIR}/sim/build/kernel_elf_init."{cc,o}
    write_sim_config "$HOST_BUILD_DIR" "$host_file" "${sim_obj_files[*]}" "${kernel_names[*]}" "$aie_version" "$SIM_TILES"
fi

if [ -f "$host_file" ]; then
    prepare_raw_xaie_debug_view \
        "$host_file" "${HOST_BUILD_DIR}/worklocal" "$runtime_source_file" \
        "$aie_version" "baremetal" "${HOST_BUILD_DIR}/main.elf"
fi
}

main "$@"