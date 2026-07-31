#!/usr/bin/env bash
###############################################################################
# run_tilinglinalg.sh — Build and compile the tilinglinalg multi-tile GEMM
# pipeline end-to-end (no hardware deployment).
#
# Steps:
#   1. Build the unitest binary (cmake --build)
#   2. Run ./test dfschedule  → generates worklocal/host.cc + kernel.cc
#   3. Compile host + kernel  → worklocal/build/host (ARM ELF, ready for board)
#
# Usage:
#   source run_tilinglinalg.sh [options]
#   bash   run_tilinglinalg.sh [options]
#
# Options:
#   --host-src <path>        Use an existing host.cc instead of running codegen
#                            (implies --skip-cmake and --skip-codegen)
#   --gen <Gen1|Gen2|Gen5>   AIE generation passed to ./test (default: Gen2)
#   --output-pp-depth <N>    Output tensor ping-pong depth (default: 2)
#   --skip-cmake             Skip cmake --build (use existing test binary)
#   --skip-codegen           Skip ./test dfschedule (use existing host.cc/kernel.cc)
#   --skip-compile           Skip hostcompile.sh (stop after codegen)
#   -j <N>                   Parallel jobs for cmake --build (default: 4)
###############################################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITEST_DIR="${SCRIPT_DIR}/src/mlir/mlirfront/tilinglinalg/pass/unitest"
BUILD_DIR="${UNITEST_DIR}/build"
WORKLOCAL_DIR="${UNITEST_DIR}/worklocal"

# Defaults
GEN="Gen2"
OUTPUT_PP_DEPTH=""
HOST_SRC=""
SKIP_CMAKE=0
SKIP_CODEGEN=0
SKIP_COMPILE=0
JOBS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-src)         HOST_SRC="$2";         shift 2 ;;
        --gen)              GEN="$2";              shift 2 ;;
        --output-pp-depth)  OUTPUT_PP_DEPTH="$2";  shift 2 ;;
        --skip-cmake)       SKIP_CMAKE=1;           shift ;;
        --skip-codegen)     SKIP_CODEGEN=1;         shift ;;
        --skip-compile)     SKIP_COMPILE=1;         shift ;;
        -j)                 JOBS="$2";              shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --host-src: copy the provided file into worklocal/ and skip codegen steps
if [ -n "$HOST_SRC" ]; then
    HOST_SRC_ABS="$(cd "$(dirname "$HOST_SRC")" && pwd)/$(basename "$HOST_SRC")"
    if [ ! -f "$HOST_SRC_ABS" ]; then
        echo "ERROR: --host-src file not found: $HOST_SRC"
        exit 1
    fi
    SKIP_CMAKE=1
    SKIP_CODEGEN=1
    cp "$HOST_SRC_ABS" "${WORKLOCAL_DIR}/host.cc"
    echo "Using provided host.cc: $HOST_SRC_ABS"
fi

# Build extra args for ./test
TEST_EXTRA_ARGS="--gen ${GEN}"
if [ -n "$OUTPUT_PP_DEPTH" ]; then
    TEST_EXTRA_ARGS="${TEST_EXTRA_ARGS} --output-pp-depth ${OUTPUT_PP_DEPTH}"
fi

echo "============================================"
echo " run_tilinglinalg.sh"
echo "  AIE generation : ${GEN}"
[ -n "$HOST_SRC" ] && echo "  Host src       : ${HOST_SRC_ABS}"
echo "  Unitest dir    : ${UNITEST_DIR}"
echo "  Worklocal dir  : ${WORKLOCAL_DIR}"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Build unitest binary
# ---------------------------------------------------------------------------
if [ "$SKIP_CMAKE" -eq 0 ]; then
    echo "=== [1/3] Building unitest (cmake --build -j${JOBS}) ==="
    if [ ! -d "${BUILD_DIR}" ]; then
        echo "ERROR: Build directory not found: ${BUILD_DIR}"
        echo "Run 'cmake ..' inside ${BUILD_DIR} first."
        exit 1
    fi
    cmake --build "${BUILD_DIR}" -- -j"${JOBS}"
    echo "✓ unitest build complete"
else
    echo "=== [1/3] Skipping cmake build (--skip-cmake) ==="
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: Generate host.cc + kernel.cc via ./test dfschedule
# ---------------------------------------------------------------------------
if [ "$SKIP_CODEGEN" -eq 0 ]; then
    echo "=== [2/3] Generating host.cc + kernel.cc (./test dfschedule) ==="
    if [ ! -f "${BUILD_DIR}/test" ]; then
        echo "ERROR: test binary not found at ${BUILD_DIR}/test"
        echo "Build the unitest first (remove --skip-cmake)."
        exit 1
    fi
    pushd "${BUILD_DIR}" > /dev/null
    ./test dfschedule ${TEST_EXTRA_ARGS}
    popd > /dev/null
    echo ""
    echo "Generated files:"
    for f in host.cc kernel.cc routing.cc; do
        [ -f "${WORKLOCAL_DIR}/${f}" ] && echo "  ${WORKLOCAL_DIR}/${f}" || true
    done
    echo "✓ Code generation complete"
else
    echo "=== [2/3] Skipping code generation (--skip-codegen) ==="
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Compile host + kernel → worklocal/build/host
# ---------------------------------------------------------------------------
if [ "$SKIP_COMPILE" -eq 0 ]; then
    echo "=== [3/3] Compiling host + kernel (hostcompile.sh) ==="
    if [ ! -f "${WORKLOCAL_DIR}/host.cc" ]; then
        echo "ERROR: host.cc not found at ${WORKLOCAL_DIR}/host.cc"
        echo "Run code generation first (remove --skip-codegen)."
        exit 1
    fi
    pushd "${WORKLOCAL_DIR}" > /dev/null
    source ./hostcompile.sh
    popd > /dev/null
    echo ""
    if [ -f "${WORKLOCAL_DIR}/build/host" ]; then
        echo "✓ Host ELF ready: ${WORKLOCAL_DIR}/build/host"
        ls -lh "${WORKLOCAL_DIR}/build/host"
    fi
else
    echo "=== [3/3] Skipping compilation (--skip-compile) ==="
fi

echo ""
echo "============================================"
echo " Done."
echo "============================================"
