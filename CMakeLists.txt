###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

cmake_minimum_required(VERSION 3.5)
project(aiehlc)



# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Default to Release, but let the user override on the command line
# (e.g. -DCMAKE_BUILD_TYPE=Debug / RelWithDebInfo) to get a debuggable aiehlc.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Size reduction flags ----
# -ffunction-sections/-fdata-sections: put each function/data in its own ELF section
# -Wl,--gc-sections: linker discards unreferenced sections (removes unused LLVM code)
# -Wl,--icf=all: Identical Code Folding via gold linker (merges duplicate functions)
# -fvisibility=hidden/-fvisibility-inlines-hidden: hide symbols to enable more ICF
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections -fvisibility=hidden -fvisibility-inlines-hidden")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -ffunction-sections -fdata-sections -fvisibility=hidden")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--icf=all -fuse-ld=gold")

message(STATUS "aFound LLVM ${LLVM_INSTALL_DIR}/bin/")
#set the llvm path find path into the path you installed
if(NOT DEFINED LLVM_INSTALL_DIR OR "${LLVM_INSTALL_DIR}" STREQUAL "")
  # Assign a default value if LLVM_INSTALL_DIR is not defined or is empty
  set(LLVM_INSTALL_DIR "/usr/local/")
endif()
message(STATUS "use set LLVM_ROOT to change LLVM into your local llvm if the compile failed!!!")
#set(LLVM_INSTALL_DIR "/Users/username/src/project/thirdparty/llvm-project/build/" CACHE PATH "Root of an LLVM/Clang/MLIR install")
set(LLVM_INSTALL_DIR "/scratch/staff/huaj/llvm-project/build/" CACHE PATH "Root of an LLVM/Clang/MLIR install")
message(STATUS "Found LLVM ${LLVM_INSTALL_DIR}/bin/")

set(CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/bin/" ${CMAKE_PREFIX_PATH})
find_program(LLVM_CONFIG ${LLVM_INSTALL_DIR}/bin/llvm-config REQUIRED)
#find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_CONFIG}")

#message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
#message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

execute_process(
  COMMAND ${LLVM_CONFIG} --includedir
  OUTPUT_VARIABLE LLVM_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${LLVM_CONFIG} --cxxflags
  OUTPUT_VARIABLE LLVM_DEFINITIONS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)


include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${LLVM_INSTALL_DIR}/include/)

message(STATUS "LLVM_INCLUDE_DIRS is: ${LLVM_INCLUDE_DIRS}")

#LLVM_DEFINITIONS will cause a conflict with boost::btree, and complain undefined reference to `boost::throw_exception(
# the reason why that happen is because in LLVM_DEFINITIONS there is a -fno-exceptions ,that will cause the said issue
#add_definitions(${LLVM_DEFINITIONS})

#set(CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/clang" ${CMAKE_PREFIX_PATH})
#find_package(Clang REQUIRED CONFIG)
set(CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/clang" ${CMAKE_PREFIX_PATH})
find_package(Clang REQUIRED CONFIG)

message(STATUS "Found Clang ${Clang_PACKAGE_VERSION}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")

message(STATUS "Clang_INCLUDE_DIRS is: ${Clang_INCLUDE_DIRS}")
message(STATUS "CLANG_INCLUDE_DIRS is: ${CLANG_INCLUDE_DIRS}")

include_directories(${Clang_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories("./src/mlir/mlirfront/")
include_directories("./src/mlir/mlirfront/tilinglinalg/pass/")
# routingmanager.h (routing dialect attrs/ops: aiehlc.cc records the kernel
# tiling and halo slab size on routing IR ops) and its generated .inc files.
include_directories("./src/mlir/mlirfront/tilinglinalg/routing/")
include_directories("./src/mlir/mlirfront/tilinglinalg/routing/inc/")


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
#####
find_package(Boost REQUIRED CONFIG)
#find_package(Boost 1.89.0 REQUIRED)
# MLIR
set(CMAKE_PREFIX_PATH "${LLVM_INSTALL_DIR}/lib/cmake/mlir" ${CMAKE_PREFIX_PATH})
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

include_directories(${MLIR_INCLUDE_DIRS})

set(SOURCE_FILES ./src/llvm/aiehlc.cc
    ./src/llvm/convautotile/conv_autotile.cpp
    ./src/llvm/convautotile/conv_desugar.cpp
    ./src/llvm/convautotile/kernel_trace.cpp
    ./src/llvm/convautotile/kernel_front.cpp
    ./src/llvm/convautotile/trace_routing.cpp)
set(mlirfrontend src/mlir/mlirfront/)

add_executable(aiehlc ${SOURCE_FILES})
target_include_directories(aiehlc PRIVATE ./src/llvm)
target_compile_definitions(aiehlc PRIVATE AIEHLC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Conv2d auto-tiling model CLI (backend of script/convtune, the AutoTVM tuner).
add_executable(aiehlc-convtile ./src/llvm/convautotile/convtile_cli.cpp
    ./src/llvm/convautotile/conv_autotile.cpp ./src/llvm/convautotile/conv_desugar.cpp
    ./src/llvm/convautotile/kernel_trace.cpp
    ./src/llvm/convautotile/kernel_front.cpp ./src/llvm/convautotile/trace_routing.cpp)

target_compile_definitions(aiehlc-convtile PRIVATE AIEHLC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Strip only for non-debug builds. In Debug / RelWithDebInfo we keep the symbol
# table + DWARF so gdb (e.g. `source script/aiehlc.sh -gdb ...`) can resolve
# functions like buildOutputTileDescriptor and set real breakpoints.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_custom_command(TARGET aiehlc POST_BUILD
        COMMAND ${CMAKE_STRIP} $<TARGET_FILE:aiehlc>
        COMMENT "Stripping binary..."
    )
endif()

llvm_map_components_to_libnames(llvm_libs support core irreader)
# this will cause "LLVM Option 'help-list' registered more than once“ issue
#target_link_libraries(aiehlc PRIVATE ${llvm_libs})
add_subdirectory(${mlirfrontend})

message(STATUS "Found sub dir ${sub_lib_dir}")
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/${sub_lib_dir})
target_link_libraries(aiehlc PRIVATE
    clangTooling
    clangBasic
    clangASTMatchers
	mlirtestlib
	${ANTLR4_LIBRARY_PATHS_LIST}
    MLIREmitCDialect  
    MLIRTargetCpp 
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
			MLIRSCFToEmitC
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
		#LLVM # use this replace ${llvm_libs} to fix the register more than once issue
)
