###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

cmake_minimum_required(VERSION 3.5)
project(mlirtest)
#set(CMAKE_C_COMPILER "/scratch/staff/huaj/mlirdocker/llvm-project/build/bin/clang")
#set(CMAKE_CXX_COMPILER "/scratch/staff/huaj/mlirdocker/llvm-project/build/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#cmake_policy(SET CMP0075 NEW)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(LLVM_INSTALL_DIR /usr/local/)
set(LLVM_ROOT /usr/local/)
if(DEFINED LLVM_INSTALL_DIR AND NOT "${LLVM_INSTALL_DIR}" STREQUAL "")
  set(LLVM_ROOT "${LLVM_INSTALL_DIR}" CACHE PATH "Root of LLVM/Clang/MLIR install" FORCE)
endif()
#set(LLVM_ROOT "/path/to/thirdparty/llvm-project/build" CACHE PATH "Root of an LLVM/Clang/MLIR install")

set(LLVM_ROOT_BIN_ARG "")
list(APPEND LLVM_ROOT_BIN_ARG --llvm-bin "${LLVM_ROOT}/bin/")

if(LLVM_ROOT)
  list(APPEND CMAKE_PREFIX_PATH
       "${LLVM_ROOT}"
       "${LLVM_ROOT}/lib/cmake"
       "${LLVM_ROOT}/lib/cmake/llvm"
       "${LLVM_ROOT}/lib/cmake/clang"
       "${LLVM_ROOT}/lib/cmake/mlir")
endif()

# ---- Find packages (auto-resolve via CMAKE_PREFIX_PATH) ----------------------
find_package(LLVM  REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(MLIR  REQUIRED CONFIG)
find_package(Boost REQUIRED CONFIG)

message(STATUS "Found Clang ${Clang_PACKAGE_VERSION}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

message(STATUS "Clang_INCLUDE_DIRS is: ${Clang_INCLUDE_DIRS}")
message(STATUS "CLANG_INCLUDE_DIRS is: ${CLANG_INCLUDE_DIRS}")

include_directories(${Clang_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

include_directories(${Boost_INCLUDE_DIRS})

include_directories(${LLVM_INSTALL_DIR}/include/)

message(STATUS "Using ${CLANG_INCLUDE_DIRS}")

set(MLIR_INC_ARGS "")
foreach(dir IN LISTS MLIR_INCLUDE_DIRS)
  list(APPEND MLIR_INC_ARGS --mlir-include "${dir}")
endforeach()

set(CMAKE_BUILD_TYPE Debug)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Parser
#	                  ./mlirpass/)
include_directories(./mlirpass/)
include_directories(./mlirlinkrelocate/)
include_directories(./tilinglinalg/)

#file(GLOB PARSER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../Parser/*.cpp)
message("Parser src Directory: ${PARSER_FILES}")



#fiind antlr include
#find_path(ANTLR4_INCLUDE_DIR antlr4-runtime/antlr4-runtime.h PATH_SUFFIXES include)
#if(ANTLR4_INCLUDE_DIR)
#    message("ANTLR4 Include Directory: ${ANTLR4_INCLUDE_DIR}")
#    include_directories(${ANTLR4_INCLUDE_DIR}/antlr4-runtime/)
#else()
#    message(FATAL_ERROR "ANTLR4 runtime header not found.")
#endif()
#find antlr lib
#execute_process(COMMAND dpkg-query -L libantlr4-runtime-dev
#                OUTPUT_VARIABLE ANTLR4_PACKAGE_FILES
#                RESULT_VARIABLE DPkg_RESULT)

#string(STRIP ${ANTLR4_PACKAGE_FILES} ANTLR4_PACKAGE_FILES)

#string(REPLACE "\n" ";" ANTLR4_PACKAGE_FILES_LIST ${ANTLR4_PACKAGE_FILES})

#foreach(FILE_PATH IN LISTS ANTLR4_PACKAGE_FILES_LIST)
#    if(FILE_PATH MATCHES "\\.so$")
#        list(APPEND ANTLR4_LIBRARY_PATHS_LIST ${FILE_PATH})
#    endif()
#endforeach()

#message("ANTLR4 Library Paths: ${ANTLR4_LIBRARY_PATHS_LIST}")

set(SOURCE_FILES ./AieFrontEndTest.cc)

set(SOURCE_LIB_FILES ./AieFrontEnd.cc)
list(APPEND SOURCE_LIB_FILES ./AieDialect.cc)
list(APPEND SOURCE_LIB_FILES ./AieLinkDialect.cpp)
list(APPEND SOURCE_LIB_FILES ./mlirpass/aiehybrid.cc)
list(APPEND SOURCE_LIB_FILES ./tilinglinalg/tilinglinalg.cpp)
list(APPEND SOURCE_LIB_FILES ./tilinglinalg/tilingpass.cpp)
list(APPEND SOURCE_LIB_FILES ./tilinglinalg/routing/routingmanager.cpp)


add_executable(mlirtest ${SOURCE_FILES})
add_library(mlirtestlib ${SOURCE_LIB_FILES} )
#add_library(mlirtestlib ${SOURCE_LIB_FILES} ${PARSER_FILES})


add_custom_target(
	aiedialectteablegen ALL
  COMMAND ${CMAKE_COMMAND} -E echo "Running custom script for AIE dialect..."
	COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/gen.sh ${MLIR_INC_ARGS} ${LLVM_ROOT_BIN_ARG}
  COMMENT "Executing custom script before building AIE Dialect from tablegen"
)

add_custom_target(
	aielinkerdialectgen ALL
  COMMAND ${CMAKE_COMMAND} -E echo "Running custom script for linker dialect..."
	COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/mlirlinkrelocate/gen.sh ${MLIR_INC_ARGS} ${LLVM_ROOT_BIN_ARG}
  COMMENT "Executing custom script before building AIE Linker Dialect from tablegen"
)

add_custom_target(
	routingtablegen ALL
  COMMAND ${CMAKE_COMMAND} -E echo "Running custom script for routing..."
	COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tilinglinalg/routing/gen.sh ${MLIR_INC_ARGS} ${LLVM_ROOT_BIN_ARG}
  COMMENT "Executing custom script before building Routing Dialect from tablegen"
)

#for the routine dialect tablegen
include_directories(./tilinglinalg/routing/inc)
#for the routine dialect cpp
include_directories(./tilinglinalg/routing/)
#for the routinehw dialect tablegen
include_directories(./tilinglinalg/routinghw/inc)
#for the routinehw dialect cpp
include_directories(./tilinglinalg/routinghw/)
#for the routing lower pass
include_directories(./tilinglinalg/pass/)

add_dependencies(mlirtestlib aiedialectteablegen)
add_dependencies(mlirtestlib aielinkerdialectgen)
add_dependencies(mlirtestlib routingtablegen)

#add_library(mlirtestlib ${SOURCE_LIB_FILES} ${PARSER_FILES})
#add_executable(mlirtest ./AieDialect.cc)

#find_package(Curses REQUIRED)
#include_directories(${CURSES_INCLUDE_DIR})

#message(STATUS "curse include is ${CURSES_INCLUDE_DIR} lib is ${CURSES_LIBRARIES}")

#set(LLVM_PATH /scratch/staff/huaj/mlirdocker/)
#set(LLVM_LIBRARY_DIR ${LLVM_PATH}/acompiler/download/llvm/lib/)
#set(LLVM_LIBRARY_DIR ${LLVM_PATH}/llvm-project/build/lib/)
#if(NOT MLIR_DIR)
#	set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
#endif()
#message(STATUS "${LLVM_LIBRARY_DIR}/cmake/mlir")
 
# MLIR
#find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "mlir include is ${MLIR_INCLUDE_DIRS} lib is ${MLIR_LIB_DIRS} llvm flag is ${LLVM_LDFLAGS}")
#need -DLLVM_ENABLE_RTTI=ON to enable RTTI, as some thirdparty lib like antlr4 use RTTI
#and LLVM default disable RTTI , this will cause the typeinfo error without do the said
#enable
#target_compile_options(mlirtest PRIVATE -fno-rtti)

target_link_libraries(mlirtest PUBLIC
    clangTooling
    clangBasic
    clangASTMatchers
	mlirtestlib
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
target_link_options(mlirtest PRIVATE ${LLVM_LDFLAGS})
