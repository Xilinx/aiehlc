###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_aiebackend
    aiebackend_pybind.cpp
    kernel_body_emitter.cpp
    routing_ir_api.cpp
)

target_link_libraries(_aiebackend PRIVATE
    mlirtestlib
    clangTooling
    clangBasic
    clangASTMatchers
    MLIREmitCDialect
    MLIRSCFToEmitC
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

target_include_directories(_aiebackend PRIVATE
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../aiegraph/inc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../aiegraph/
    ${CMAKE_CURRENT_SOURCE_DIR}/../aiegraph/lower/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
)

target_link_options(_aiebackend PRIVATE ${LLVM_LDFLAGS})

# Install the .so next to the aiebackend Python package (aiebackend.find_module_dir
# also searches the build tree, so this step is optional)
install(TARGETS _aiebackend DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
