#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# kernelcompile.sh - Compile a single AIE kernel source to a relocatable object
#
# Supports both chess (xchesscc + xchessmk) and llvm-aie compiler flows.
# Produces a kernel.o with canonical binary symbols for host linking.
#
# Usage:
#   kernelcompile.sh --kernel-cc <wrapper.cc> --prx <file.prx>
#                    --output-dir <dir> --func-name <name>
#                    --aie-version <1|2|5> [--platform <baremetal|linux>]
#                    [--use-llvm-aie] [--ld-script <file.ld>]
#                    [--debug-output] [--include-base <path>]
#                    [--commons-dir <path>]
#
# Examples:
#
#   # Chess flow (xchesscc + xchessmk) — default
#   kernelcompile.sh --kernel-cc ./my_kernel/compute_kernel.cc \
#                    --prx ./my_kernel/compute_kernel.prx \
#                    --output-dir ./build/kernel_out \
#                    --func-name matmul \
#                    --aie-version 2 \
#                    --platform baremetal \
#                    --debug-output
#
#   # LLVM-AIE flow (clang++ from llvm-aie)
#   kernelcompile.sh --kernel-cc ./my_kernel/compute_kernel.cc \
#                    --ld-script ./my_kernel/kernel.ld \
#                    --output-dir ./build/kernel_out \
#                    --func-name matmul \
#                    --aie-version 2 \
#                    --platform baremetal \
#                    --use-llvm-aie \
#                    --debug-output
###############################################################################

dbg_echo() {
    if [ "$DEBUG_OUTPUT" = 1 ]; then
        echo "$@"
    fi
}

run_cmd() {
    local command="$1"
    local rc=0
    if [ "$DEBUG_OUTPUT" = 0 ]; then
        $command > /dev/null 2>&1 || rc=$?
    else
        $command || rc=$?
    fi
    return $rc
}

redefine_symbols() {
    local obj_file="$1"
    local func_name="$2"
    local objcopy_tool="$3"

    # ld -r -b binary creates symbols like:
    #   _binary_<path_with_slashes_as_underscores>_{start,end,size}
    # The path varies depending on where ld is invoked from, so we match
    # any _binary_..._end / _start / _size and rename to canonical names.
    nm "$obj_file" | while read -r line; do
        symbol=$(echo "$line" | awk '{print $3}')
        [ -z "$symbol" ] && continue

        if echo "$symbol" | grep -q "^_binary_.*_end$"; then
            dbg_echo "Renaming symbol: $symbol to _binary_kernel_${func_name}_end"
            "$objcopy_tool" --redefine-sym "$symbol"=_binary_kernel_"${func_name}"_end "$obj_file"
        elif echo "$symbol" | grep -q "^_binary_.*_start$"; then
            dbg_echo "Renaming symbol: $symbol to _binary_kernel_${func_name}_start"
            "$objcopy_tool" --redefine-sym "$symbol"=_binary_kernel_"${func_name}"_start "$obj_file"
        elif echo "$symbol" | grep -q "^_binary_.*_size$"; then
            dbg_echo "Renaming symbol: $symbol to _binary_kernel_${func_name}_size"
            "$objcopy_tool" --redefine-sym "$symbol"=_binary_kernel_"${func_name}"_size "$obj_file"
        fi
    done
}

# Emit DWARF line-table artifacts next to the kernel ELF:
#   <out>/kernel.decodedline.txt  - readelf --debug-dump=decodedline (human readable)
#   <out>/kernel.linemap.json     - parsed addr->file:line map for aiediag
# Prefers system readelf, falls back to ${TOOL_PREFIX}readelf. Non-fatal.
emit_linemap_artifacts() {
    local out_dir="$1"
    local elf="${out_dir}/kernel"
    local decoded="${out_dir}/kernel.decodedline.txt"
    local linemap="${out_dir}/kernel.linemap.json"

    if [ ! -f "$elf" ]; then
        echo "Warning: kernel ELF not found ($elf); skipping line-map artifacts"
        return 0
    fi

    # Pick a readelf: prefer system, then cross-toolchain readelf.
    local readelf_tool=""
    if command -v readelf >/dev/null 2>&1; then
        readelf_tool="readelf"
    elif command -v "${TOOL_PREFIX}readelf" >/dev/null 2>&1; then
        readelf_tool="${TOOL_PREFIX}readelf"
    else
        echo "Warning: no readelf found; skipping line-map artifacts"
        return 0
    fi

    "$readelf_tool" --debug-dump=decodedline "$elf" > "$decoded" 2>/dev/null || true
    if [ ! -s "$decoded" ]; then
        echo "Warning: empty .debug_line table from $elf (no DWARF?); kernel.linemap.json not generated"
        return 0
    fi
    dbg_echo "Wrote $decoded"

    # Parse decodedline -> JSON line map.
    local parser="${AIEHLC_ROOT_DIR}/script/parse_linemap.py"
    if [ -f "$parser" ]; then
        python3 "$parser" "$decoded" --elf "$elf" -o "$linemap" >/dev/null 2>&1 \
            && dbg_echo "Wrote $linemap" \
            || echo "Warning: parse_linemap.py failed; $linemap not generated"
    else
        echo "Warning: parse_linemap.py not found at $parser; $linemap not generated"
    fi
}

# Drop the DWARF sections nothing downstream reads, keeping the line table.
#
# The chess backend software-pipelines a VLIW schedule, so every local variable
# gets a location list entry per scheduling slot it moves through: on the matmul
# kernel that is 12.9 MB of .debug_loc describing 6.7 KB of .text — 98.4% of the
# ELF. The kernel ELF is embedded verbatim into the host ELF (`ld -r -b binary`),
# so all of it is downloaded over JTAG by `dow -force` on every run.
#
# Kept: .debug_line (parse_linemap.py / aiediag pc need it — that is the whole
# reason -g is on), plus .debug_frame/.debug_ranges, which are small.
# Dropped: .debug_loc and the .debug_info graph that references it. .debug_info
# has to go with it because llvm-objcopy rebuilds SHT_STRTAB sections, and in a
# chess ELF .debug_str IS an SHT_STRTAB — it comes back empty, leaving every
# DW_FORM_strp in .debug_info dangling. Better to remove the whole group than to
# ship an ELF whose .debug_info makes readelf print offset-too-big warnings.
#
# The full-DWARF original is kept beside it as `kernel_debug`; point a DWARF
# reader at that when you need variable locations or type info.
#
# Must be llvm-objcopy: the chess ELF's e_machine is 0x108, which GNU binutils
# rejects outright ("Unable to recognise the format of the input file").
# In place, under the same filename, because `ld -r -b binary` derives the
# symbol names the host links against (_binary_..._start/_end) from the path.
strip_kernel_debug_loc() {
    local out_dir="$1"
    local elf="${out_dir}/kernel"
    local backup="${out_dir}/kernel_debug"

    [ -f "$elf" ] || { echo "Warning: no kernel ELF at $elf; skipping DWARF strip"; return 0; }

    local objcopy=""
    for cand in llvm-objcopy llvm-objcopy-19 llvm-objcopy-18 llvm-objcopy-14; do
        if command -v "$cand" >/dev/null 2>&1; then objcopy="$cand"; break; fi
    done
    if [ -z "$objcopy" ]; then
        echo "Warning: no llvm-objcopy found; keeping full DWARF in $elf"
        return 0
    fi

    cp -f "$elf" "$backup" || return 0
    if "$objcopy" \
            --remove-section=.debug_loc \
            --remove-section=.debug_info \
            --remove-section=.debug_abbrev \
            --remove-section=.debug_str \
            --remove-section=.debug_pubnames \
            --remove-section=.debug_pubtypes \
            "$backup" "$elf" 2>/dev/null; then
        dbg_echo "Stripped DWARF from kernel ELF ($(stat -c%s "$backup") -> $(stat -c%s "$elf") bytes); full copy at $backup"
    else
        # Leave the original in place rather than half a kernel.
        cp -f "$backup" "$elf"
        echo "Warning: $objcopy failed on kernel ELF; keeping full DWARF"
    fi
}

usage() {
    echo "Usage: $0 --kernel-cc <file.cc> --output-dir <dir> --func-name <name>"
    echo "           --aie-version <1|2|5> [--prx <file.prx>] [--ld-script <file.ld>]"
    echo "           [--platform <baremetal|linux>] [--use-llvm-aie] [--debug-output]"
    echo "           [--include-base <path>] [--commons-dir <path>] [--keep-debug-loc]"
    return 1
}

# --- Parse arguments ---

kernel_cc=""
prx_file=""
ld_script=""
output_dir=""
func_name=""
aie_version="2"
platform="baremetal"
use_llvm_aie="false"
DEBUG_OUTPUT=0
include_base=""
commons_dir=""
KEEP_DEBUG_LOC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -help|--help) usage ;;
        --kernel-cc)    kernel_cc="$2";    shift 2 ;;
        --prx)          prx_file="$2";     shift 2 ;;
        --ld-script)    ld_script="$2";    shift 2 ;;
        --output-dir)   output_dir="$2";   shift 2 ;;
        --func-name)    func_name="$2";    shift 2 ;;
        --aie-version)  aie_version="$2";  shift 2 ;;
        --platform)     platform="$2";     shift 2 ;;
        --use-llvm-aie) use_llvm_aie="true"; shift ;;
        --debug-output) DEBUG_OUTPUT=1;    shift ;;
        --include-base) include_base="$2"; shift 2 ;;
        --commons-dir)  commons_dir="$2";  shift 2 ;;
        --keep-debug-loc) KEEP_DEBUG_LOC=1; shift ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# --- Validate required parameters ---

if [ -z "$kernel_cc" ]; then echo "Error: --kernel-cc is required"; usage; fi
if [ -z "$output_dir" ]; then echo "Error: --output-dir is required"; usage; fi
if [ -z "$func_name" ]; then echo "Error: --func-name is required"; usage; fi

if [[ "$use_llvm_aie" != "true" ]] && [ -z "$prx_file" ]; then
    echo "Error: --prx is required when using chess (not --use-llvm-aie)"
    usage
fi

if [[ "$use_llvm_aie" == "true" ]] && [ -z "$ld_script" ]; then
    echo "Error: --ld-script is required when using --use-llvm-aie"
    usage
fi

# --- Validate environment ---

if [ -z "$XILINX_VITIS" ]; then
    echo "Error: XILINX_VITIS environment variable not set"
    return 1
fi

XILINX_VITIS_AIETOOLS="$XILINX_VITIS/aietools"

# Ensure Vitis aietools shared libraries (e.g. libLLVM.so.20.1) are discoverable.
# Vitis 2026.1+ ships a dynamically-linked `opt` that requires this.
export LD_LIBRARY_PATH="$XILINX_VITIS_AIETOOLS/lib/lnx64.o${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Set up include paths ---

if [ -z "$include_base" ]; then
    include_base="$XILINX_VITIS_AIETOOLS/include/drivers/aiengine"
fi

# Locate aiehlc project root (kc.sh lives at aiehlc/script/kc.sh)
AIEHLC_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INCLUDE_PATH="-I$XILINX_VITIS_AIETOOLS/include \
-I$XILINX_VITIS_AIETOOLS/include/aie_api \
-I$include_base \
-I${AIEHLC_ROOT_DIR}/src/mlir/runtime \
-I${AIEHLC_ROOT_DIR}/include"

if [ -n "$kernel_cc" ]; then
    _kernel_src_dir="$(cd "$(dirname "$kernel_cc")" 2>/dev/null && pwd)"
    if [ -n "$_kernel_src_dir" ]; then
        INCLUDE_PATH="$INCLUDE_PATH -I$_kernel_src_dir"
    fi
fi

if [ -n "$commons_dir" ]; then
    INCLUDE_PATH="$INCLUDE_PATH -I$commons_dir"
fi

# --- Set up tool prefix ---

if [[ "$platform" == "linux" ]]; then
    TOOL_PREFIX="aarch64-linux-gnu-"
elif [[ "$platform" == "baremetal" ]]; then
    TOOL_PREFIX="aarch64-none-elf-"
elif [[ "$platform" == "sim" ]]; then
    TOOL_PREFIX=""
    linker="/usr/bin/ld -m elf_x86_64 -EL -r -b binary"
    objcopy_tool="objcopy"
fi

if [[ "$platform" != "sim" ]]; then
    linker="${TOOL_PREFIX}ld -EL -r -b binary"
    objcopy_tool="${TOOL_PREFIX}objcopy"
fi

# --- Set up compiler flags per AIE version ---

silent_flag="-s"
if [ $DEBUG_OUTPUT = 1 ]; then
    silent_flag="+s"
fi

arch_model_dir_aie="${XILINX_VITIS}/aietools/data/aie/lib"
arch_model_dir_aieml="${XILINX_VITIS}/aietools/data/aie_ml/lib"
arch_model_dir_aie2ps="${XILINX_VITIS}/aietools/data/aie2ps/lib"

# Force-include kernel_log.h so klog() is available in all kernel builds
KERNEL_LOG_INCLUDE="+Wllvm,-include,${AIEHLC_ROOT_DIR}/src/mlir/runtime/kernel_log.h"

# xchesscc compiler flags per AIE version
# DWARF debug info is always-on (-g) so the kernel ELF carries a .debug_line
# table (DICompileUnit/DILocation referencing kernel.cc). This enables PC->source
# mapping for HW debug (see parse_linemap.py / aiediag pc). -g is routed through
# +Wllvm, alongside -O2 because it is a clang frontend/codegen flag. NOTE: with -O2
# the line table is approximate (inlining/optimization) — coarse PC->line only.
#
# -g (not -gline-tables-only) is deliberate. The variable metadata it adds is
# what the backend turns into the multi-MB .debug_loc — swapping in
# -gline-tables-only drops DILocalVariable to zero and would suppress .debug_loc
# at the source. It is kept because that metadata is worth having *somewhere*:
# strip_kernel_debug_loc removes the bulk from the deployed ELF after the link
# and parks the full-DWARF copy at <out>/kernel_debug, so a DWARF reader still
# has variable locations and types to work with. Use --keep-debug-loc to opt out.
compiler_flags_aie="$silent_flag +f -p me -P $arch_model_dir_aie +P 4 +Wllvm,-O2,-g,-fno-jump-tables,-fno-discard-value-names,-mllvm,-chess-collapse-struct-types-during-linking=0,-Xclang,-chess-only-info-critical-passes -D__AIENGINE__ -D__AIE_ARCH__=10 -D__AIEARCH=10 -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR $KERNEL_LOG_INCLUDE $INCLUDE_PATH"
compiler_flags_aieml="-aiearch aie-ml $silent_flag +f -p me -P $arch_model_dir_aieml +P 4 +Wllvm,-O2,-g,-fno-jump-tables,-fno-discard-value-names,-mllvm,-chess-collapse-struct-types-during-linking=0,-Xclang,-chess-only-info-critical-passes -D__AIENGINE__ -D__AIE_ARCH__=20 -D__AIEARCH=20 -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST $KERNEL_LOG_INCLUDE $INCLUDE_PATH"
compiler_flags_aie2ps="-aiearch aie2ps $silent_flag +f -p me -P $arch_model_dir_aie2ps +P 4 +Wllvm,-O2,-g,-fno-jump-tables,-fno-discard-value-names,-mllvm,-chess-collapse-struct-types-during-linking=0,-Xclang,-chess-only-info-critical-passes -D__AIENGINE__ -D__AIE_ARCH__=22 -D__AIEARCH=22 -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST $KERNEL_LOG_INCLUDE $INCLUDE_PATH"

# llvm-aie compiler flags per AIE version
LLVM_AIE_INCLUDE_PATH="-I$XILINX_VITIS_AIETOOLS/include \
-I$XILINX_VITIS_AIETOOLS/include/aie_api \
-I$include_base"

KERNEL_LOG_INCLUDE_LLVM="-include ${AIEHLC_ROOT_DIR}/src/mlir/runtime/kernel_log.h"

compiler_flags_llvm_aie_aie="-include ${LLVM_AIE_PATH}/../llvm-aie-extra.h $KERNEL_LOG_INCLUDE_LLVM -Wno-unknown-attributes -Wno-macro-redefined -O2 -std=c++20 --target=aie2-none-unknown-elf -D__AIECC__ -D__AIENGINE__ -D__AIE_ARCH__=10 -D__AIEARCH=10 -D_LIBCPP_HAS_NO_THREADS -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR $LLVM_AIE_INCLUDE_PATH $INCLUDE_PATH"
compiler_flags_llvm_aie_aieml="-include ${LLVM_AIE_PATH}/../llvm-aie-extra.h $KERNEL_LOG_INCLUDE_LLVM -Wno-unknown-attributes -Wno-macro-redefined -O2 -std=c++20 --target=aie2-none-unknown-elf -D__AIECC__ -D__AIENGINE__ -D__AIE_ARCH__=20 -D__AIEARCH=20 -D_LIBCPP_HAS_NO_THREADS -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST $LLVM_AIE_INCLUDE_PATH $INCLUDE_PATH"
compiler_flags_llvm_aie_aie2ps="-include ${LLVM_AIE_PATH}/../llvm-aie-extra.h $KERNEL_LOG_INCLUDE_LLVM -Wno-unknown-attributes -Wno-macro-redefined -O2 -std=c++20 --target=aie2-none-unknown-elf -D__AIECC__ -D__AIENGINE__ -D__AIE_ARCH__=22 -D__AIEARCH=22 -D_LIBCPP_HAS_NO_THREADS -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST $LLVM_AIE_INCLUDE_PATH $INCLUDE_PATH"

# Select flags based on AIE version
extra_chess_flag=""
if [[ "$aie_version" == "1" ]]; then
    if [ ! -d "$arch_model_dir_aie" ]; then
        echo "Error: AIE gen1 chess library not found at '$arch_model_dir_aie'."
        echo "       AIE gen1 is not supported by this Vitis installation."
        echo "       Use --aie-version 2 (AIE-ML) or --aie-version 5 (AIE2PS)."
        exit 1
    fi
    compiler_flags_chess="$compiler_flags_aie"
    compiler_flags_llvm_aie_sel="$compiler_flags_llvm_aie_aie"
    arch_model_dir="$arch_model_dir_aie"
elif [[ "$aie_version" == "2" ]]; then
    compiler_flags_chess="$compiler_flags_aieml"
    compiler_flags_llvm_aie_sel="$compiler_flags_llvm_aie_aieml"
    arch_model_dir="$arch_model_dir_aieml"
    extra_chess_flag+=" -aiearch aie-ml"
elif [[ "$aie_version" == "5" ]]; then
    compiler_flags_chess="$compiler_flags_aie2ps"
    compiler_flags_llvm_aie_sel="$compiler_flags_llvm_aie_aie2ps"
    arch_model_dir="$arch_model_dir_aie2ps"
    extra_chess_flag+=" -aiearch aie2ps"
else
    echo "Unsupported AIE version: $aie_version"
    return 1
fi

if [ $DEBUG_OUTPUT = 1 ]; then
    compiler_flags_llvm_aie_sel+=" -v"
fi

chess_elf_compiler="xchessmk $extra_chess_flag $silent_flag -s -C Release_LLVM -P $arch_model_dir +P 4 -DDEPLOYMENT_ELF=1 -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR -DAIE2_FP32_EMULATION_ACCURACY_FAST"

if [[ "$platform" == "sim" ]]; then
    compiler_flags_chess+=" -DAIEHLC_KERNEL_SIM"
    compiler_flags_llvm_aie_sel+=" -DAIEHLC_KERNEL_SIM"
    chess_elf_compiler+=" -DAIEHLC_KERNEL_SIM"
fi

# Kernel-side logging + KERNELCONFIGOFFLOAD register tracing.
#
# Driven by the user's source:
#
#     #pragma aie_debug_level(AIE_KERNEL_CONFIG_TRACE)
#
# aiehlc parses that and writes aout/kernel_build_flags.sh (the kernel is built
# by a separate process, so the pragma cannot reach xchesscc any other way).
# Sourcing it here sets AIEHLC_KERNEL_LOG. The env var can also be set directly
# to force tracing without editing the source.
#
# Two defines, one switch:
#   KERNEL_LOG_ENABLED         -> compiles klog() in at all (kernel_log.h). Without
#                                 it klog is an empty inline and ALL kernel logging
#                                 — the compute kernel's own calls included —
#                                 costs nothing.
#   KERNELCONFIGOFFLOAD_TRACE  -> the generated kernel.cc additionally klogs every
#                                 (register, value) the offload writes: BD words,
#                                 lock inits, channel-start.
#                                 Tags "OFF "/"VAL " per write, plus "STS2"/"STMM"
#                                 (channel) and "STOF"/"STVL" (start reg + value).
#
# Off by default: klog costs cycles and trace-buffer space in a normal run.
if [ -f "${AIEHLC_ROOT_DIR}/aout/kernel_build_flags.sh" ]; then
    . "${AIEHLC_ROOT_DIR}/aout/kernel_build_flags.sh"
fi
if [ "${AIEHLC_KERNEL_LOG:-0}" != "0" ]; then
    for _d in -DKERNEL_LOG_ENABLED -DKERNELCONFIGOFFLOAD_TRACE; do
        compiler_flags_chess+=" $_d"
        compiler_flags_llvm_aie_sel+=" $_d"
        chess_elf_compiler+=" $_d"
    done
    unset _d
    echo "[kc.sh] kernel logging enabled (klog + KERNELCONFIGOFFLOAD register trace)"
fi

# --- Create output directory ---

mkdir -p "$output_dir"

# --- Compile kernel ---

obj_file="${output_dir}/kernel.o"

if [[ "$use_llvm_aie" != "true" ]]; then
    echo "Compiling kernel (using chess): $(basename "$kernel_cc")"

    # Step 1: xchesscc -> LLVM IR
    dbg_echo xchesscc $compiler_flags_chess -o "${output_dir}/kernel_orig.ll" "$kernel_cc"
    set -x
    xchesscc $compiler_flags_chess -o "${output_dir}/kernel_orig.ll" "$kernel_cc"
    _xchesscc_rc=$?
    set +x
    if [ $_xchesscc_rc -ne 0 ]; then
        echo "Error: xchesscc compilation failed for $kernel_cc"
        return 1
    fi

    # Step 2: LLVM opt passes (xlopt x2).
    _xlopt_lib=$(ls -d "$XILINX_VITIS_AIETOOLS"/[0-9]*.*/lnx64.o/lib 2>/dev/null | head -1)
    export LD_LIBRARY_PATH="${_xlopt_lib:+${_xlopt_lib}:}$XILINX_VITIS_AIETOOLS/lib/lnx64.o:${LD_LIBRARY_PATH}"
    run_cmd "$XILINX_VITIS_AIETOOLS/lnx64.o/tools/clang/bin/opt -S -load-pass-plugin=$XILINX_VITIS_AIETOOLS/lib/lnx64.o/libLLVMXLOpt.so -passes=xlopt ${output_dir}/kernel_orig.ll -o ${output_dir}/kernel.ll"
    if [ $? -ne 0 ]; then
        echo "Error: LLVM opt pass 1 (xlopt) failed"
        return 1
    fi
    run_cmd "$XILINX_VITIS_AIETOOLS/lnx64.o/tools/clang/bin/opt -S -load-pass-plugin=$XILINX_VITIS_AIETOOLS/lib/lnx64.o/libLLVMXLOpt.so -passes=xlopt ${output_dir}/kernel.ll -o ${output_dir}/kernel.ll"
    if [ $? -ne 0 ]; then
        echo "Error: LLVM opt pass 2 (xlopt) failed"
        return 1
    fi

    # Step 3: xchessmk -> kernel ELF
    dbg_echo $chess_elf_compiler +o "$output_dir" "$prx_file"
    $chess_elf_compiler +o "$output_dir" "$prx_file"
    if [ $? -ne 0 ]; then
        echo "Error: xchessmk ELF linking failed"
        return 1
    fi

    # Step 3b: Shed the DWARF that only bloats the JTAG download. Before the
    # line-map step and before embedding, so everything downstream sees the
    # small ELF; the decodedline output is byte-identical either way.
    if [ "$KEEP_DEBUG_LOC" = "0" ]; then
        strip_kernel_debug_loc "$output_dir"
    fi

    # Step 4: Emit DWARF line-table artifacts for PC->source debug.
    # .debug_line parsing is architecture-agnostic, so a host/aarch64 readelf
    # works on the AIE ELF. Non-fatal: a missing/empty table only warns.
    emit_linemap_artifacts "$output_dir"
else
    echo "Compiling kernel (using llvm-aie): $(basename "$kernel_cc")"

    if [ -z "$LLVM_AIE_PATH" ]; then
        echo "Error: LLVM_AIE_PATH environment variable not set (required for --use-llvm-aie)"
        return 1
    fi

    dbg_echo ${LLVM_AIE_PATH}/bin/clang++ $compiler_flags_llvm_aie_sel "$kernel_cc" -Wl,-T "$ld_script" -o "$output_dir/kernel"
    ${LLVM_AIE_PATH}/bin/clang++ $compiler_flags_llvm_aie_sel "$kernel_cc" -Wl,-T "$ld_script" -o "$output_dir/kernel"
    if [ $? -ne 0 ]; then
        echo "Error: llvm-aie clang++ compilation failed for $kernel_cc"
        return 1
    fi

    # Same treatment as the chess path: this ELF is embedded and JTAG-downloaded
    # too, and a no-op when llvm-aie emitted no .debug_loc to begin with.
    if [ "$KEEP_DEBUG_LOC" = "0" ]; then
        strip_kernel_debug_loc "$output_dir"
    fi
fi

# --- Create relocatable object from kernel ELF binary ---

dbg_echo $linker -o "$obj_file" "$output_dir/kernel"
$linker -o "$obj_file" "$output_dir/kernel"
if [ $? -ne 0 ]; then
    echo "Error: ld binary embedding failed (kernel ELF -> relocatable object)"
    return 1
fi

# --- Rename ELF binary symbols to canonical names ---

redefine_symbols "$obj_file" "$func_name" "$objcopy_tool"
if [ $? -ne 0 ]; then
    echo "Error: objcopy symbol renaming failed"
    return 1
fi

echo "Kernel compiled: $obj_file"
