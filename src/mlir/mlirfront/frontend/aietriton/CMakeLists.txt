###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_aietriton_core
    aietriton_pybind.cpp
    aie_pass/kernel_body_emitter.cpp
)

target_link_libraries(_aietriton_core PRIVATE
    mlirtestlib
    clangTooling
    clangBasic
    clangASTMatchers
    MLIREmitCDialect
    MLIRTargetCpp
    MLIRAMDGPUDialect
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUTransforms
    MLIRIR
    MLIRControlFlowToLLVM
    MLIRBytecodeWriter
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRMathToLLVM
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRTransformDialect
    MLIRTransformUtils
    MLIRLinalgTransformOps
    MLIRTransformDialectTransforms
    LLVMPasses
    dl
)

target_include_directories(_aietriton_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/routing/inc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/routing/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/routinghw/inc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/routinghw/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/routingimplement/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passroutingtodmap/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passblueprinttoschedule/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passblueprinttoschedulekernel/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passdmaphoptodfscheblueprint/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passdmaphopprovenancemap/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passdfscheduleprovenancemap/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passroutingprovenancemap/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passdmaphoptoroutinghw/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passdmaptodmaphop/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passschedulecanonicalize/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passdfscheduletoapi/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/passdfscheduletokernelapi/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/pass/kernelconfig/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/dataflowmap/dmap/inc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/dataflowmap/dmap/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/dataflowmap/dmaphop/inc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/dataflowmap/dmaphop/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/dataflowmap/dfschedule/inc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/dataflowmap/dfschedule/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/dataflowmap/dfscheblueprint/inc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../tilinglinalg/dataflowmap/dfscheblueprint/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_options(_aietriton_core PRIVATE ${LLVM_LDFLAGS})

# Install the .so next to the Python package files
install(TARGETS _aietriton_core DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
