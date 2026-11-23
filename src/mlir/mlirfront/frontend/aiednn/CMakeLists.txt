###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# aiednn dialect + aiednn-opt, built standalone against the local LLVM:
#   source script/llvm_env.sh
#   cmake -G Ninja -S src/mlir/mlirfront/frontend/aiednn -B build/aiednn && ninja -C build/aiednn
cmake_minimum_required(VERSION 3.20)
project(aiednn-dialect LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{LLVM_INSTALL_DIR} AND NOT MLIR_DIR)
  set(MLIR_DIR "$ENV{LLVM_INSTALL_DIR}/lib/cmake/mlir")
endif()
find_package(MLIR REQUIRED CONFIG)
message(STATUS "aiednn: MLIR from ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)
set(LLVM_TARGET_DEFINITIONS td/AiednnOps.td)
mlir_tablegen(include/AiednnOps.h.inc -gen-op-decls)
mlir_tablegen(include/AiednnOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/AiednnOpsDialect.h.inc -gen-dialect-decls -dialect=aiednn)
mlir_tablegen(include/AiednnOpsDialect.cpp.inc -gen-dialect-defs -dialect=aiednn)
mlir_tablegen(include/AiednnOpsAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=aiednn)
mlir_tablegen(include/AiednnOpsAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=aiednn)
add_public_tablegen_target(AiednnIncGen)

add_library(AiednnDialect STATIC lib/AiednnDialect.cpp lib/ExportPlan.cpp lib/FuseEpilogue.cpp lib/PlanMemory.cpp)
add_dependencies(AiednnDialect AiednnIncGen)
target_link_libraries(AiednnDialect PUBLIC MLIRIR MLIRFuncDialect MLIRPass MLIRSupport MLIRSideEffectInterfaces)

add_executable(aiednn-opt tool/aiednn-opt.cpp)
target_link_libraries(aiednn-opt PRIVATE AiednnDialect MLIROptLib MLIRFuncDialect MLIRTransforms)
