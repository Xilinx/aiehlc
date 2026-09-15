# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT

REPO_ROOT      ?= $(abspath ../..)
SIM_SRC        := $(REPO_ROOT)/src/sim
RUNTIME_SRC    := $(REPO_ROOT)/src/mlir/runtime
REPO_INCLUDE   := $(REPO_ROOT)/include
SCRIPT_DIR     := $(REPO_ROOT)/script

AIE_RT_SRC_DIR  ?= $(REPO_ROOT)/thirdparty/alib/aie-rt/driver/src
AIE_RT_INC_DIR  ?= $(REPO_ROOT)/thirdparty/alib/aie-rt/driver/internal

BUILD_DIR    := build

AIE_RT_OBJS_DIR ?= $(BUILD_DIR)/aie_rt_objs
AIE_RT_C_SRCS   := $(filter-out $(AIE_RT_SRC_DIR)/liburing/% $(AIE_RT_SRC_DIR)/uc_driver/%,\
                     $(wildcard $(AIE_RT_SRC_DIR)/*/*.c) \
                     $(wildcard $(AIE_RT_SRC_DIR)/*/*/*.c))
AIE_RT_OBJS     := $(patsubst $(AIE_RT_SRC_DIR)/%.c,$(AIE_RT_OBJS_DIR)/%.o,$(AIE_RT_C_SRCS))
AIE_RT_SRC_INCS := $(shell find $(AIE_RT_SRC_DIR) -maxdepth 2 -mindepth 1 -type d \
                     | grep -v 'liburing\|swig' \
                     | sed 's/^/-I/')
AIE_RT_LIB      := $(BUILD_DIR)/libxaiengine.a

ifeq ($(strip $(XILINX_VITIS)),)
  $(error XILINX_VITIS is not set. Source script/setup.sh first.)
endif
VITIS_TOOLS    := $(XILINX_VITIS)/aietools
VITIS_CLANG    := $(VITIS_TOOLS)/lnx64.o/tools/clang/bin/clang++
VITIS_LIB      := $(VITIS_TOOLS)/lib/lnx64.o
SYSTEMC_INC    := $(VITIS_TOOLS)/data/osci_systemc/include
SYSTEMC_LIB    := $(VITIS_TOOLS)/data/osci_systemc/lib/lnx64

GCC_BASE := $(firstword $(wildcard $(VITIS_TOOLS)/tps/lnx64/gcc) \
                         $(wildcard $(VITIS_TOOLS)/tps/lnx64.o/gcc))
GCC_VER  := $(notdir $(firstword $(wildcard $(GCC_BASE)/lib/gcc/x86_64-pc-linux-gnu/*)))
GCC_INSTALL := $(GCC_BASE)/lib/gcc/x86_64-pc-linux-gnu/$(GCC_VER)
GCC_INC     := $(GCC_BASE)/include/c++/$(GCC_VER)
GCC_INC_ARCH := $(GCC_INC)/x86_64-pc-linux-gnu

AIE_ARCH       ?= 20

ifeq ($(AIE_ARCH),10)
  AIE_GEN := 1
else ifeq ($(AIE_ARCH),22)
  AIE_GEN := 5
else
  AIE_GEN := 2
endif

_STUB_ONLY_GOALS := stub_all_tiles build_stub
ifeq ($(filter-out $(_STUB_ONLY_GOALS),$(MAKECMDGOALS)),)
  _STUB_ONLY := 1
endif
ifneq ($(_STUB_ONLY),1)
ifeq ($(strip $(AIEHLC_HOST_SRC)),)
  $(error AIEHLC_HOST_SRC is not set. Pass the absolute path to host.cc.)
endif
endif

WORKLOCAL_DIR  := $(if $(AIEHLC_HOST_SRC),$(dir $(abspath $(AIEHLC_HOST_SRC))),)

ROUTING_SRC ?= $(if $(WORKLOCAL_DIR),$(wildcard $(WORKLOCAL_DIR)/routing.cc),)
ROUTING_OBJ :=
ifneq ($(strip $(ROUTING_SRC)),)
  ROUTING_OBJ := $(BUILD_DIR)/routing.o
endif

KERNEL_OBJS  ?=
KERNEL_NAMES ?=
ALL_KERNEL_OBJS := $(KERNEL_OBJS)

CXXFLAGS = \
    --gcc-install-dir=$(GCC_INSTALL) \
    -stdlib=libstdc++ \
    -fPIC -std=c++20 -O1 \
    -D__AIESIM__ \
    -DAIE_GEN=$(AIE_GEN) \
    -D__PS_INIT_AIE__ \
    -DXAIE_DEBUG \
    -D__AIE_ARCH__=$(AIE_ARCH) \
    -D__LOCK_FENCE_MODE__=2 \
    -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR \
    -DAIE2_FP32_EMULATION_ACCURACY_FAST \
    -DSC_INCLUDE_DYNAMIC_PROCESSES \
    -Dmain=aiehlc_ps_main \
    -DAIEHLC_HOST_SRC=\"$(abspath $(AIEHLC_HOST_SRC))\" \
    -Wno-enum-constexpr-conversion \
    -Wno-unknown-attributes \
    -Wno-macro-redefined \
    -Wno-deprecated-declarations \
    -Wno-format-security \
    -I$(SIM_SRC) \
    -I$(RUNTIME_SRC) \
    -I$(REPO_INCLUDE) \
    -I$(WORKLOCAL_DIR) \
    -I$(AIE_RT_INC_DIR) \
    -I$(AIE_RT_INC_DIR)/xaiengine \
    -I$(VITIS_TOOLS)/include \
    -I$(VITIS_TOOLS)/include/drivers/aiengine \
    -I$(VITIS_TOOLS)/include/aie_api \
    -I$(SYSTEMC_INC) \
    -I$(VITIS_TOOLS)/tps/boost_1_72_0 \
    -I$(VITIS_TOOLS)/include/xtlm/include \
    -I$(VITIS_TOOLS)/include/common_cpp/common_cpp_v1_0/include \
    -I$(GCC_INC) \
    -I$(GCC_INC_ARCH)

LDFLAGS = \
    --gcc-install-dir=$(GCC_INSTALL) \
    -stdlib=libstdc++ \
    -Wl,-rpath,$(GCC_BASE)/lib \
    -shared \
    -pthread \
    -Wl,--as-needed \
    -L$(VITIS_LIB) \
    -L$(SYSTEMC_LIB) \
    -lsystemc \
    -lxtlm \
    -lstdc++fs \
    -Wl,-rpath,$(VITIS_LIB):$(SYSTEMC_LIB)

export LD_LIBRARY_PATH := $(VITIS_LIB):$(VITIS_TOOLS)/tps/lnx64.o/gcc/lib64:$(LD_LIBRARY_PATH)

TOOL_PREFIX := aarch64-none-elf-

WRAPPER_OBJ  := $(BUILD_DIR)/aiehlc_ps_wrapper.o
IOUTILS_OBJ  := $(BUILD_DIR)/ioutils.o
DBGSRV_OBJ   := $(BUILD_DIR)/aiehlc_dbg_server.o
RUNTIME_OBJS := \
    $(BUILD_DIR)/aie_runtime.o \
    $(BUILD_DIR)/aie_runtime_debug.o \
    $(BUILD_DIR)/aie_runtime_stream_debug.o \
    $(BUILD_DIR)/aie_runtime_common.o \
    $(BUILD_DIR)/aie_runtime_control_plan.o

_FIRST_KERNEL := $(firstword $(KERNEL_NAMES))
KERNEL_ELF_INIT_OBJ :=
ifneq ($(strip $(_FIRST_KERNEL)),)
  KERNEL_ELF_INIT_OBJ := $(BUILD_DIR)/kernel_elf_init.o
endif

OUR_OBJS := $(WRAPPER_OBJ) $(IOUTILS_OBJ) $(DBGSRV_OBJ) $(RUNTIME_OBJS) $(ROUTING_OBJ) $(KERNEL_ELF_INIT_OBJ)

.DEFAULT_GOAL := all
.PHONY: all clean info stub_all_tiles build_stub

all: $(BUILD_DIR)/aiehlc_ps.so

$(BUILD_DIR):
	mkdir -p $@

$(AIE_RT_OBJS_DIR)/%.o: $(AIE_RT_SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	gcc -O1 -std=c11 \
	    -D_POSIX_C_SOURCE=200809 -D_DEFAULT_SOURCE \
	    -DXAIE_FEATURE_PRIVILEGED_ENABLE -DXAIE_AIG_SOURCE \
	    -D__AIESIM__ \
	    -fPIC \
	    -I$(AIE_RT_SRC_DIR) \
	    -I$(AIE_RT_INC_DIR) \
	    $(AIE_RT_SRC_INCS) \
	    -I$(VITIS_TOOLS)/include/drivers/aiengine/aie_codegen_inc \
	    -w \
	    -c $< -o $@

$(AIE_RT_LIB): $(AIE_RT_OBJS) | $(BUILD_DIR)
	@echo "[aie-rt] Archiving libxaiengine.a ($(words $(AIE_RT_OBJS)) objects)..."
	ar rcs $@ $^
	@echo "[aie-rt] Done: $@"


$(WRAPPER_OBJ): $(SIM_SRC)/aiehlc_ps_wrapper.cpp $(SIM_SRC)/ioutils.h $(abspath $(AIEHLC_HOST_SRC)) | $(BUILD_DIR)
	@echo "[compile] aiehlc_ps_wrapper.cpp  (host: $(notdir $(AIEHLC_HOST_SRC)))"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@

$(IOUTILS_OBJ): $(SIM_SRC)/ioutils.cpp $(SIM_SRC)/ioutils.h | $(BUILD_DIR)
	@echo "[compile] ioutils.cpp"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@

$(DBGSRV_OBJ): $(SIM_SRC)/aiehlc_dbg_server.cpp $(SIM_SRC)/aiehlc_dbg_server.h $(SIM_SRC)/aiehlc_dbg_protocol.h | $(BUILD_DIR)
	@echo "[compile] aiehlc_dbg_server.cpp"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/aie_runtime.o: $(RUNTIME_SRC)/aie_runtime.c | $(BUILD_DIR)
	@echo "[compile] aie_runtime.c"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/aie_runtime_debug.o: $(RUNTIME_SRC)/aie_runtime_debug.c | $(BUILD_DIR)
	@echo "[compile] aie_runtime_debug.c"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/aie_runtime_stream_debug.o: $(RUNTIME_SRC)/aie_runtime_stream_debug.c | $(BUILD_DIR)
	@echo "[compile] aie_runtime_stream_debug.c"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/aie_runtime_common.o: $(RUNTIME_SRC)/aie_runtime_common.c | $(BUILD_DIR)
	@echo "[compile] aie_runtime_common.c"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/aie_runtime_control_plan.o: $(RUNTIME_SRC)/aie_runtime_control_plan.c | $(BUILD_DIR)
	@echo "[compile] aie_runtime_control_plan.c"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@

ifneq ($(strip $(ROUTING_SRC)),)
$(ROUTING_OBJ): $(ROUTING_SRC) | $(BUILD_DIR)
	@echo "[compile] routing.cc  (tilinglinalg stream-switch config)"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@
endif

ifneq ($(strip $(KERNEL_ELF_INIT_OBJ)),)
.PHONY: FORCE
FORCE:

$(BUILD_DIR)/kernel_elf_init.cc: FORCE | $(BUILD_DIR)
	@printf '#include "aie_runtime.h"\n\
extern "C" {\n\
    extern unsigned char _binary_kernel_$(if $(KERNEL_FUNCNAME),$(KERNEL_FUNCNAME)_,)start[];\n\
    extern unsigned char _binary_kernel_$(if $(KERNEL_FUNCNAME),$(KERNEL_FUNCNAME)_,)end[];\n\
}\n\
__attribute__((constructor))\n\
static void _kernel_elf_register(void) {\n\
    if (_binary_kernel_$(if $(KERNEL_FUNCNAME),$(KERNEL_FUNCNAME)_,)start &&\n\
            _binary_kernel_$(if $(KERNEL_FUNCNAME),$(KERNEL_FUNCNAME)_,)start !=\n\
            _binary_kernel_$(if $(KERNEL_FUNCNAME),$(KERNEL_FUNCNAME)_,)end)\n\
        __Runtime_set_kernel_elf(\n\
            _binary_kernel_$(if $(KERNEL_FUNCNAME),$(KERNEL_FUNCNAME)_,)start);\n\
}\n' > $@.tmp
	@if ! cmp -s $@.tmp $@ 2>/dev/null; then \
	    mv $@.tmp $@; \
	    echo "[gen] kernel_elf_init.cc  (sym=$(if $(KERNEL_FUNCNAME),_binary_kernel_$(KERNEL_FUNCNAME)_start,_binary_kernel_start))"; \
	else rm -f $@.tmp; fi

$(KERNEL_ELF_INIT_OBJ): $(BUILD_DIR)/kernel_elf_init.cc | $(BUILD_DIR)
	@echo "[compile] kernel_elf_init.cc"
	$(VITIS_CLANG) $(CXXFLAGS) -c $< -o $@
endif

$(BUILD_DIR)/aiehlc_ps.so: $(OUR_OBJS) $(AIE_RT_LIB)
	@echo "[link] aiehlc_ps.so"
	$(VITIS_CLANG) $(LDFLAGS) -o $@ $(OUR_OBJS) $(ALL_KERNEL_OBJS) \
	    -Wl,--whole-archive $(AIE_RT_LIB) -Wl,--no-whole-archive
	@echo "Done: $@"
	@ls -lh $@

STUB_SRC         := $(SIM_SRC)/stub_kernel.cc
STUB_BUILD_DIR   := $(BUILD_DIR)/stub_kernel_build
STUB_PRX         := $(STUB_BUILD_DIR)/kernel.prx
STUB_ELF         := $(STUB_BUILD_DIR)/stub_kernel
WORK_DIR         ?= $(abspath .)/Work_gen2

ifeq ($(AIE_ARCH),10)
  STUB_ROW_START := 1
  STUB_ROW_END   := 8
  STUB_COL_END   := 49
else
  STUB_ROW_START := 3
  STUB_ROW_END   := 10
  STUB_COL_END   := 37
endif

$(STUB_PRX): | $(BUILD_DIR)
	mkdir -p $(STUB_BUILD_DIR)
	@STUB_BCF=$(SIM_SRC)/stub_kernel.bcf; \
	BCF_LINE=""; \
	if [ -f "$$STUB_BCF" ]; then \
	    cp "$$STUB_BCF" $(STUB_BUILD_DIR)/stub_kernel.bcf; \
	    BCF_LINE="    <option id=\"bridge.cfg\" value=\"stub_kernel.bcf\" path=\"./\"/>"; \
	fi; \
	{ echo '<project name="kernel" processor="me">'; \
	  echo '    <issinit/>'; \
	  echo '    <file type="lbc" name="kernel.ll" path="./"/>'; \
	  [ -n "$$BCF_LINE" ] && echo "$$BCF_LINE"; \
	  echo "    <option id=\"cpp.define\" value=\"__AIENGINE__ __AIEARCH__=$(AIE_ARCH)\" inherit=\"1\"/>"; \
	  echo '    <option id="llvm.xargs" value="-fno-jump-tables -fno-discard-value-names -mllvm -chess-collapse-struct-types-during-linking=0" inherit="1"/>'; \
	  echo '    <option id="project.name" value="kernel"/>'; \
	  echo '    <option id="project.type" value="exe"/>'; \
	  echo '</project>'; \
	} > $@

ifeq ($(AIE_ARCH),10)
  CHESS_ARCH_FLAG := 
  CHESS_ARCH_DATA := $(VITIS_TOOLS)/data/versal_prod/lib
else ifeq ($(AIE_ARCH),22)
  CHESS_ARCH_FLAG := -aiearch aie2ps
  CHESS_ARCH_DATA := $(VITIS_TOOLS)/data/aie2ps/lib
else
  CHESS_ARCH_FLAG := -aiearch aie-ml
  CHESS_ARCH_DATA := $(VITIS_TOOLS)/data/aie_ml/lib
endif

CHESS_CC   := xchesscc
CHESS_MK   := xchessmk
CHESS_OPT  := $(VITIS_TOOLS)/lnx64.o/tools/clang/bin/opt
CHESS_XLOPT := -load-pass-plugin=$(VITIS_TOOLS)/lib/lnx64.o/libLLVMXLOpt.so -passes=xlopt

$(STUB_ELF): $(STUB_SRC) $(STUB_PRX)
	@echo "[stub] Compiling stub kernel (aie-gen=$(AIE_GEN), __AIEARCH__=$(AIE_ARCH))..."
	$(CHESS_CC) $(CHESS_ARCH_FLAG) -s +f -p me -P $(CHESS_ARCH_DATA) +P 4 \
	    +Wllvm,-O2,-fno-jump-tables,-fno-discard-value-names \
	    -D__AIENGINE__ -D__AIE_ARCH__=$(AIE_ARCH) -D__AIEARCH=$(AIE_ARCH) \
	    -D__LOCK_FENCE_MODE__=0 -DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR \
	    -I$(VITIS_TOOLS)/include \
	    -o $(STUB_BUILD_DIR)/kernel_orig.ll $(STUB_SRC)
	$(CHESS_OPT) -S $(CHESS_XLOPT) $(STUB_BUILD_DIR)/kernel_orig.ll \
	    -o $(STUB_BUILD_DIR)/kernel.ll
	$(CHESS_OPT) -S $(CHESS_XLOPT) $(STUB_BUILD_DIR)/kernel.ll \
	    -o $(STUB_BUILD_DIR)/kernel.ll
	cd $(abspath $(STUB_BUILD_DIR)) && \
	$(CHESS_MK) $(CHESS_ARCH_FLAG) -s -s -C Release_LLVM -P $(CHESS_ARCH_DATA) +P 4 \
	    -DDEPLOYMENT_ELF=1 -D__LOCK_FENCE_MODE__=0 \
	    +o . kernel.prx
	cp $(STUB_BUILD_DIR)/kernel $(STUB_ELF)
	for ext in '#' '##' sdr srv; do \
	    [ -f "$(STUB_BUILD_DIR)/kernel.$$ext" ] && \
	        cp "$(STUB_BUILD_DIR)/kernel.$$ext" "$(STUB_ELF).$$ext" || true; \
	done
	@echo "[stub] Stub ELF: $(STUB_ELF)"

build_stub: $(STUB_ELF)
	@echo "[stub] Stub ELF ready: $(STUB_ELF)"

stub_all_tiles: $(STUB_ELF)
	@echo "[stub] Populating all core tiles in $(WORK_DIR)/aie/ (col_row naming) ..."
	@TILE_ROW_START=$(STUB_ROW_START); \
	for abs_row in $$(seq $(STUB_ROW_START) $(STUB_ROW_END)); do \
	    row=$$((abs_row - TILE_ROW_START + 1)); \
	    for col in $$(seq 0 $(STUB_COL_END)); do \
	        tile_name="$${col}_$${row}"; \
	        dir="$(WORK_DIR)/aie/$${tile_name}/Release"; \
	        mkdir -p "$$dir"; \
	        ln -sf "$(abspath $(STUB_ELF))" "$$dir/$${tile_name}"; \
	        for ext in '#' '##' sdr srv; do \
	            src="$(STUB_BUILD_DIR)/stub_kernel.$$ext"; \
	            [ -f "$$src" ] && ln -sf "$(abspath $$src)" "$$dir/$${tile_name}.$$ext" || true; \
	        done; \
	    done; \
	done
	@echo "[stub] Done — all core tiles activated for runtime kernel loading"

clean:
	rm -rf $(BUILD_DIR)

info:
	@echo "AIEHLC_HOST_SRC : $(AIEHLC_HOST_SRC)"
	@echo "AIE_ARCH        : $(AIE_ARCH)  (__AIE_ARCH__=$(AIE_ARCH))"
	@echo "AIE_GEN         : $(AIE_GEN)   (-DAIE_GEN=$(AIE_GEN))"
	@echo "AIE_RT_SRC_DIR  : $(AIE_RT_SRC_DIR)"
	@echo "AIE_RT_OBJS     : $(words $(AIE_RT_OBJS)) files"
	@echo "VITIS_CLANG     : $(VITIS_CLANG)"
