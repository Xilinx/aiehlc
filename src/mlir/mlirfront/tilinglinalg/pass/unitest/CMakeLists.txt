###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

cmake_minimum_required(VERSION 3.5)
project(tablegen)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(LLVM_INSTALL_DIR /usr/local/)
#set(CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/clang" ${CMAKE_PREFIX_PATH})
set(LLVM_ROOT /usr/local/)
if(DEFINED LLVM_INSTALL_DIR AND NOT "${LLVM_INSTALL_DIR}" STREQUAL "")
  set(LLVM_ROOT "${LLVM_INSTALL_DIR}" CACHE PATH "Root of LLVM/Clang/MLIR install" FORCE)
endif()
#set(LLVM_ROOT "/path/to/thirdparty/llvm-project/build" CACHE PATH "Root of an LLVM/Clang/MLIR install")

if(LLVM_ROOT)
  list(APPEND CMAKE_PREFIX_PATH
       "${LLVM_ROOT}"
       "${LLVM_ROOT}/lib/cmake"
       "${LLVM_ROOT}/lib/cmake/llvm"
       "${LLVM_ROOT}/lib/cmake/clang"
       "${LLVM_ROOT}/lib/cmake/mlir")
endif()

set(LLVM_ROOT_BIN_ARG "")
list(APPEND LLVM_ROOT_BIN_ARG --llvm-bin "${LLVM_ROOT}/bin/")

find_package(Clang REQUIRED CONFIG)
find_package(MLIR  REQUIRED CONFIG)

message(STATUS "Found Clang ${Clang_PACKAGE_VERSION}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

message(STATUS "Clang_INCLUDE_DIRS is: ${Clang_INCLUDE_DIRS}")
message(STATUS "CLANG_INCLUDE_DIRS is: ${CLANG_INCLUDE_DIRS}")

include_directories(${Clang_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

include_directories(${LLVM_INSTALL_DIR}/include/)

message(STATUS "Using ${CLANG_INCLUDE_DIRS}")

set(CMAKE_BUILD_TYPE Debug)

include_directories(../../../../)
include_directories(../../routing/inc/)
include_directories(../../routinghw/inc/)
include_directories(../../routing/)
include_directories(../../routinghw/)
include_directories(../)
include_directories(../routingimplement/include/)


#file(GLOB PARSER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../Parser/*.cpp)
message("Parser src Directory: ${PARSER_FILES}")

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp 
                ${CMAKE_CURRENT_SOURCE_DIR}/../../routinghw/routinghwmanager.cpp 
                ${CMAKE_CURRENT_SOURCE_DIR}/../../routing/routingmanager.cpp 
                ${CMAKE_CURRENT_SOURCE_DIR}/../routinghwlower.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../routinglower.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../routingunrolling.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../routingdeadargclean.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../routingconstantfold.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../routingimplement/routing/routingpath.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../routingimplement/hw/hwresource.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../routingimplement/hw/ResourceManager.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../routingimplement/routing/routingtopology.cpp)

add_executable(test ${SOURCE_FILES})

set(MLIR_INC_ARGS "")
foreach(dir IN LISTS MLIR_INCLUDE_DIRS)
  list(APPEND MLIR_INC_ARGS --mlir-include "${dir}")
endforeach()

add_custom_target(
	routingdilaect ALL
  COMMAND ${CMAKE_COMMAND} -E echo "Running custom script..."
	COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../../routing/gen.sh ${MLIR_INC_ARGS} ${LLVM_ROOT_BIN_ARG}
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/../../routinghw/gen.sh ${MLIR_INC_ARGS} ${LLVM_ROOT_BIN_ARG}
  COMMENT "Executing custom script before building AIE Dialect from tablegen"
)

add_dependencies(test routingdilaect)

#set(LLVM_PATH /scratch/staff/huaj/mlirdocker/)
#set(LLVM_LIBRARY_DIR ${LLVM_PATH}/llvm-project/build/lib/)
#if(NOT MLIR_DIR)
#	set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
#endif()
#message(STATUS "${LLVM_LIBRARY_DIR}/cmake/mlir")
 
# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "mlir include is ${MLIR_INCLUDE_DIRS} lib is ${MLIR_LIB_DIRS} llvm flag is ${LLVM_LDFLAGS}")

target_link_libraries(test PUBLIC
    clangTooling
    clangBasic
    clangASTMatchers
	${ANTLR4_LIBRARY_PATHS_LIST}
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
    # MLIR
    # MLIRStandardOps
	MLIRPass
	MLIRTransforms
	MLIRLLVMDialect
	MLIRSupport
	MLIRTargetLLVMIRExport
	MLIRMathToLLVM
			#MLIRROCDLToLLVMIRTranslation
			#MLIRGPUDialect
			#MLIRSCFToControlFlow
			#MLIRIndexToLLVM
			#MLIRGPUToROCDLTransforms
			#LLVM
      #the tiling logic need these
      MLIRFuncDialect
      MLIRFuncTransforms
      MLIRLinalgDialect
      MLIRLinalgTransforms
      MLIRMemRefDialect
      MLIRTransformDialect
      MLIRTransformUtils
      MLIRLinalgTransformOps
      MLIRTransformDialectTransforms
      #tiling logic end
    LLVMPasses
    #LLVMNVPTXCodeGen
    #LLVMNVPTXAsmPrinter
    #LLVMAMDGPUCodeGen
    #LLVMAMDGPUAsmParser
    dl
)
target_link_options(test PRIVATE ${LLVM_LDFLAGS})
