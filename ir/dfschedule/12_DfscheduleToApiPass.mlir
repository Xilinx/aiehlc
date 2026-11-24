module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_aot = 0 : i64, routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  emitc.verbatim "#include \22aie_runtime.h\22"
  emitc.verbatim "#include \22aie_runtime_debug.h\22"
  emitc.verbatim "#include \22aie_timer.h\22"
  emitc.verbatim "XTime g_xtimer_start;"
  emitc.verbatim "XTime g_xtimer_end;"
  emitc.verbatim "static inline void timerStart() {"
  emitc.verbatim "    XTime_GetTime(&g_xtimer_start);"
  emitc.verbatim "}"
  emitc.verbatim "static inline void timerEnd() {"
  emitc.verbatim "    XTime_GetTime(&g_xtimer_end);"
  emitc.verbatim "    double elapsed_ms = 1.0 * (g_xtimer_end - g_xtimer_start) / COUNTS_PER_SECOND * 1000.0;"
  emitc.verbatim "    printf(\22aie kernel time: %.3f ms\\n\22, elapsed_ms);"
  emitc.verbatim "}"
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) attributes {routing.kernel_tiling = #routing.kernel_tiling<out_rounds = 16>} {
    emitc.call_opaque "host_canonicalized"() : () -> ()
    return
  }
  emitc.func @host_canonicalized(%arg0: !emitc.ptr<!emitc.opaque<"XAie_DevInst">>, %arg1: !emitc.ptr<!emitc.opaque<"void">>, %arg2: !emitc.ptr<!emitc.opaque<"void">>, %arg3: !emitc.ptr<!emitc.opaque<"void">>) {
    emitc.verbatim "XAie_DevInst* dev = v1;"
    %0 = "emitc.constant"() <{value = #emitc.opaque<"XAIE_MEM_CACHEABLE">}> : () -> i32
    %1 = "emitc.constant"() <{value = 15 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 14 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = 13 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = 12 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %6 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %7 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %8 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %9 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %10 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %11 = "emitc.constant"() <{value = 1 : index}> : () -> index
    %12 = "emitc.constant"() <{value = 4 : index}> : () -> index
    %13 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %14 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %15 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %16 = "emitc.constant"() <{value = 48 : i32}> : () -> i32
    %17 = "emitc.constant"() <{value = 50176 : i32}> : () -> i32
    %18 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %19 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %20 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %21 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %22 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %23 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %24 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %25 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %26 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %25) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %27 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %28 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %29 = emitc.call_opaque "XAie_TileLoc"(%27, %28) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %30 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %31 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %32 = emitc.call_opaque "XAie_TileLoc"(%30, %31) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %33 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %34 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,3) */"
    %35 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %36 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %37 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %38 = emitc.call_opaque "XAie_TileLoc"(%36, %37) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %39 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %40 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,4) */"
    %41 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %42 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %43 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %44 = emitc.call_opaque "XAie_TileLoc"(%42, %43) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %45 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %46 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,5) */"
    %47 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %48 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %49 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %50 = emitc.call_opaque "XAie_TileLoc"(%48, %49) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %51 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %52 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,6) */"
    %53 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,0) */"
    %54 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %55 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %56 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %55) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %57 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %58 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %59 = emitc.call_opaque "XAie_TileLoc"(%57, %58) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %60 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %61 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %62 = emitc.call_opaque "XAie_TileLoc"(%60, %61) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %63 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %64 = emitc.call_opaque "__runtime_buffer_offset"(%56, %63) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %65 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %66 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,3) */"
    %67 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %68 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %69 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %70 = emitc.call_opaque "XAie_TileLoc"(%68, %69) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %71 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %72 = emitc.call_opaque "__runtime_buffer_offset"(%56, %71) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %73 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %74 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,4) */"
    %75 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %76 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %77 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %78 = emitc.call_opaque "XAie_TileLoc"(%76, %77) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %79 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %80 = emitc.call_opaque "__runtime_buffer_offset"(%56, %79) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %81 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %82 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,5) */"
    %83 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %84 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %85 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %86 = emitc.call_opaque "XAie_TileLoc"(%84, %85) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %87 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %88 = emitc.call_opaque "__runtime_buffer_offset"(%56, %87) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %89 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %90 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,6) */"
    %91 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,0) */"
    %92 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %93 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %94 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %93) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %95 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %96 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %97 = emitc.call_opaque "XAie_TileLoc"(%95, %96) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %98 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %99 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %100 = emitc.call_opaque "XAie_TileLoc"(%98, %99) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %101 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %102 = emitc.call_opaque "__runtime_buffer_offset"(%94, %101) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %103 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %104 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,3) */"
    %105 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %106 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %107 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %108 = emitc.call_opaque "XAie_TileLoc"(%106, %107) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %109 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %110 = emitc.call_opaque "__runtime_buffer_offset"(%94, %109) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %111 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %112 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,4) */"
    %113 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %114 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %115 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %116 = emitc.call_opaque "XAie_TileLoc"(%114, %115) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %117 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %118 = emitc.call_opaque "__runtime_buffer_offset"(%94, %117) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %119 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %120 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,5) */"
    %121 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %122 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %123 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %124 = emitc.call_opaque "XAie_TileLoc"(%122, %123) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %125 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %126 = emitc.call_opaque "__runtime_buffer_offset"(%94, %125) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %127 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %128 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,6) */"
    %129 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,0) */"
    %130 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %131 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %132 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %131) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %133 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %134 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %135 = emitc.call_opaque "XAie_TileLoc"(%133, %134) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %136 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %137 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %138 = emitc.call_opaque "XAie_TileLoc"(%136, %137) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %139 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %140 = emitc.call_opaque "__runtime_buffer_offset"(%132, %139) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %141 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %142 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,3) */"
    %143 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %144 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %145 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %146 = emitc.call_opaque "XAie_TileLoc"(%144, %145) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %147 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %148 = emitc.call_opaque "__runtime_buffer_offset"(%132, %147) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %149 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %150 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,4) */"
    %151 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %152 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %153 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %154 = emitc.call_opaque "XAie_TileLoc"(%152, %153) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %155 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %156 = emitc.call_opaque "__runtime_buffer_offset"(%132, %155) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %157 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %158 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,5) */"
    %159 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %160 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %161 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %162 = emitc.call_opaque "XAie_TileLoc"(%160, %161) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %163 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %164 = emitc.call_opaque "__runtime_buffer_offset"(%132, %163) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %165 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %166 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,6) */"
    %167 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,0) */"
    %168 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %169 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %170 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %169) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %171 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %172 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,3) */"
    %173 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %174 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %175 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,3) */"
    %176 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %177 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %178 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,3) */"
    %179 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %180 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %181 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,3) */"
    %182 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,0) */"
    %183 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %184 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %185 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %184) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %186 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %187 = emitc.call_opaque "__runtime_buffer_offset"(%185, %186) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %188 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %189 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,3) */"
    %190 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %191 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %192 = emitc.call_opaque "__runtime_buffer_offset"(%185, %191) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %193 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %194 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,3) */"
    %195 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %196 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %197 = emitc.call_opaque "__runtime_buffer_offset"(%185, %196) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %198 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %199 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,3) */"
    %200 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %201 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %202 = emitc.call_opaque "__runtime_buffer_offset"(%185, %201) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %203 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %204 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,3) */"
    %205 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,0) */"
    %206 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %207 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %208 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %207) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %209 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %210 = emitc.call_opaque "__runtime_buffer_offset"(%208, %209) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %211 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %212 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,4) */"
    %213 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %214 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %215 = emitc.call_opaque "__runtime_buffer_offset"(%208, %214) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %216 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %217 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,4) */"
    %218 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %219 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %220 = emitc.call_opaque "__runtime_buffer_offset"(%208, %219) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %221 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %222 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,4) */"
    %223 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %224 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %225 = emitc.call_opaque "__runtime_buffer_offset"(%208, %224) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %226 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %227 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,4) */"
    %228 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,0) */"
    %229 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %230 = "emitc.constant"() <{value = 200704 : i64}> : () -> i64
    %231 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %230) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %232 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %233 = emitc.call_opaque "__runtime_buffer_offset"(%231, %232) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %234 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %235 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,4) */"
    %236 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %237 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %238 = emitc.call_opaque "__runtime_buffer_offset"(%231, %237) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %239 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %240 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,4) */"
    %241 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %242 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %243 = emitc.call_opaque "__runtime_buffer_offset"(%231, %242) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %244 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %245 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,4) */"
    %246 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %247 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %248 = emitc.call_opaque "__runtime_buffer_offset"(%231, %247) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %249 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %250 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,4) */"
    %251 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,0) */"
    %252 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %253 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %254 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %253) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %255 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %256 = emitc.call_opaque "__runtime_buffer_offset"(%254, %255) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %257 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %258 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,5) */"
    %259 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %260 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %261 = emitc.call_opaque "__runtime_buffer_offset"(%254, %260) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %262 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %263 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,5) */"
    %264 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %265 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %266 = emitc.call_opaque "__runtime_buffer_offset"(%254, %265) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %267 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %268 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,5) */"
    %269 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %270 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %271 = emitc.call_opaque "__runtime_buffer_offset"(%254, %270) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %272 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %273 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,5) */"
    %274 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,0) */"
    %275 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %276 = "emitc.constant"() <{value = 401408 : i64}> : () -> i64
    %277 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %276) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %278 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %279 = emitc.call_opaque "__runtime_buffer_offset"(%277, %278) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %280 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %281 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,5) */"
    %282 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %283 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %284 = emitc.call_opaque "__runtime_buffer_offset"(%277, %283) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %285 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %286 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,5) */"
    %287 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %288 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %289 = emitc.call_opaque "__runtime_buffer_offset"(%277, %288) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %290 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %291 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,5) */"
    %292 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %293 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %294 = emitc.call_opaque "__runtime_buffer_offset"(%277, %293) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %295 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %296 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,5) */"
    %297 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,0) */"
    %298 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %299 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %300 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %299) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %301 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %302 = emitc.call_opaque "__runtime_buffer_offset"(%300, %301) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %303 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %304 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,6) */"
    %305 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %306 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %307 = emitc.call_opaque "__runtime_buffer_offset"(%300, %306) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %308 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %309 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,6) */"
    %310 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %311 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %312 = emitc.call_opaque "__runtime_buffer_offset"(%300, %311) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %313 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %314 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,6) */"
    %315 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %316 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %317 = emitc.call_opaque "__runtime_buffer_offset"(%300, %316) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %318 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %319 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,6) */"
    %320 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 3 for tile (3,0) */"
    %321 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %322 = "emitc.constant"() <{value = 602112 : i64}> : () -> i64
    %323 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %322) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %324 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %325 = emitc.call_opaque "__runtime_buffer_offset"(%323, %324) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %326 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %327 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,6) */"
    %328 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %329 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %330 = emitc.call_opaque "__runtime_buffer_offset"(%323, %329) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %331 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %332 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,6) */"
    %333 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %334 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %335 = emitc.call_opaque "__runtime_buffer_offset"(%323, %334) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %336 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %337 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,6) */"
    %338 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %339 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %340 = emitc.call_opaque "__runtime_buffer_offset"(%323, %339) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %341 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %342 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,6) */"
    %343 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 3 for tile (2,0) */"
    %344 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    emitc.verbatim "/* Load Kernel Group: 16 tile(s) */"
    %345 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %346 = emitc.call_opaque "__Runtime_load_kernel_group_16t"(%arg0, %32, %38, %44, %50, %62, %70, %78, %86, %100, %108, %116, %124, %138, %146, %154, %162, %345) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, i32) -> !emitc.opaque<"kernel_group">
    emitc.verbatim "/* Lock inits \E2\80\94 must precede kernel launch to avoid contention */"
    emitc.verbatim "/* Lock init: tile(0,3) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(0,4) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(0,5) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(0,6) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(1,3) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(1,4) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(1,5) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(1,6) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(2,3) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(2,4) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(2,5) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(2,6) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(3,3) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(3,4) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(3,5) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(3,6) lock=2 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* Lock init: tile(0,3) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(1,3) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(2,3) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(3,3) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(0,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(1,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(2,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(3,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(0,4) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(1,4) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(2,4) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(3,4) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(0,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(1,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(2,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(3,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(0,5) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(1,5) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(2,5) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(3,5) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(0,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(1,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(2,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(3,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(0,6) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(1,6) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(2,6) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(3,6) lock=0 init_value=2  */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* Lock init: tile(0,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(0, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(1,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(1, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(2,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(2, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Lock init: tile(3,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(dev, XAie_TileLoc(3, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* Launch Kernel Group */"
    %347 = emitc.call_opaque "__Runtime_launch_kernel_group"(%arg0, %346) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"kernel_group">) -> !emitc.opaque<"event">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %348 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %349 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %350 = emitc.call_opaque "__runtime_buffer_arg"(%26) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %351 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %352 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %353 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %354 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %355 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %356 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %350, %22, %348, %349, %355, %22, %351, %352, %353, %354, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %357 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %358 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %359 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(0,0), direction=MM2S */"
    %360 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %356, %357, %358, %359) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %361 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %362 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %363 = emitc.call_opaque "__runtime_buffer_arg"(%34) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %364 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %365 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %366 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %367 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %368 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %369 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %32, %363, %24, %361, %362, %368, %22, %364, %365, %366, %367, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %370 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %371 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %372 = emitc.call_opaque "__runtime_buffer_arg"(%33) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %373 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %374 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %375 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %376 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %377 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %378 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %32, %372, %22, %370, %371, %377, %22, %373, %374, %375, %376, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %379 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %380 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %381 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,3), direction=S2MM */"
    %382 = emitc.call_opaque "__Runtime_dma_createio_4"(%32, %378, %379, %380, %381) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %383 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %384 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %385 = emitc.call_opaque "__runtime_buffer_arg"(%40) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %386 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %387 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %388 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %389 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %390 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %391 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %38, %385, %24, %383, %384, %390, %22, %386, %387, %388, %389, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %392 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %393 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %394 = emitc.call_opaque "__runtime_buffer_arg"(%39) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %395 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %396 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %397 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %398 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %399 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %400 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %38, %394, %22, %392, %393, %399, %22, %395, %396, %397, %398, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %401 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %402 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %403 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,4), direction=S2MM */"
    %404 = emitc.call_opaque "__Runtime_dma_createio_4"(%38, %400, %401, %402, %403) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %405 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %406 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %407 = emitc.call_opaque "__runtime_buffer_arg"(%46) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %408 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %409 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %410 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %411 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %412 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %413 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %44, %407, %24, %405, %406, %412, %22, %408, %409, %410, %411, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %414 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %415 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %416 = emitc.call_opaque "__runtime_buffer_arg"(%45) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %417 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %418 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %419 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %420 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %421 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %422 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %44, %416, %22, %414, %415, %421, %22, %417, %418, %419, %420, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %423 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %424 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %425 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,5), direction=S2MM */"
    %426 = emitc.call_opaque "__Runtime_dma_createio_4"(%44, %422, %423, %424, %425) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %427 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %428 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %429 = emitc.call_opaque "__runtime_buffer_arg"(%52) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %430 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %431 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %432 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %433 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %434 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %435 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %50, %429, %24, %427, %428, %434, %22, %430, %431, %432, %433, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %436 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %437 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %438 = emitc.call_opaque "__runtime_buffer_arg"(%51) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %439 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %440 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %441 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %442 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %443 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %444 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %50, %438, %22, %436, %437, %443, %22, %439, %440, %441, %442, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %445 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %446 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %447 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,6), direction=S2MM */"
    %448 = emitc.call_opaque "__Runtime_dma_createio_4"(%50, %444, %445, %446, %447) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %449 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %450 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %451 = emitc.call_opaque "__runtime_buffer_arg"(%56) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %452 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %453 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %454 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %455 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %456 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %457 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %59, %451, %22, %449, %450, %456, %22, %452, %453, %454, %455, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %458 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %459 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %460 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(1,0), direction=MM2S */"
    %461 = emitc.call_opaque "__Runtime_dma_createio_4"(%59, %457, %458, %459, %460) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %462 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %463 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %464 = emitc.call_opaque "__runtime_buffer_arg"(%66) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %465 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %466 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %467 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %468 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %469 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %470 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %62, %464, %24, %462, %463, %469, %22, %465, %466, %467, %468, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %471 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %472 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %473 = emitc.call_opaque "__runtime_buffer_arg"(%65) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %474 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %475 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %476 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %477 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %478 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %479 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %62, %473, %22, %471, %472, %478, %22, %474, %475, %476, %477, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %480 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %481 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %482 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,3), direction=S2MM */"
    %483 = emitc.call_opaque "__Runtime_dma_createio_4"(%62, %479, %480, %481, %482) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %484 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %485 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %486 = emitc.call_opaque "__runtime_buffer_arg"(%74) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %487 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %488 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %489 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %490 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %491 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %492 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %70, %486, %24, %484, %485, %491, %22, %487, %488, %489, %490, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %493 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %494 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %495 = emitc.call_opaque "__runtime_buffer_arg"(%73) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %496 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %497 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %498 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %499 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %500 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %501 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %70, %495, %22, %493, %494, %500, %22, %496, %497, %498, %499, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %502 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %503 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %504 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,4), direction=S2MM */"
    %505 = emitc.call_opaque "__Runtime_dma_createio_4"(%70, %501, %502, %503, %504) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %506 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %507 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %508 = emitc.call_opaque "__runtime_buffer_arg"(%82) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %509 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %510 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %511 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %512 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %513 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %514 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %78, %508, %24, %506, %507, %513, %22, %509, %510, %511, %512, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %515 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %516 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %517 = emitc.call_opaque "__runtime_buffer_arg"(%81) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %518 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %519 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %520 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %521 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %522 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %523 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %78, %517, %22, %515, %516, %522, %22, %518, %519, %520, %521, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %524 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %525 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %526 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,5), direction=S2MM */"
    %527 = emitc.call_opaque "__Runtime_dma_createio_4"(%78, %523, %524, %525, %526) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %528 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %529 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %530 = emitc.call_opaque "__runtime_buffer_arg"(%90) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %531 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %532 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %533 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %534 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %535 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %536 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %86, %530, %24, %528, %529, %535, %22, %531, %532, %533, %534, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %537 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %538 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %539 = emitc.call_opaque "__runtime_buffer_arg"(%89) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %540 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %541 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %542 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %543 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %544 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %545 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %86, %539, %22, %537, %538, %544, %22, %540, %541, %542, %543, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %546 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %547 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %548 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,6), direction=S2MM */"
    %549 = emitc.call_opaque "__Runtime_dma_createio_4"(%86, %545, %546, %547, %548) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %550 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %551 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %552 = emitc.call_opaque "__runtime_buffer_arg"(%94) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %553 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %554 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %555 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %556 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %557 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %558 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %97, %552, %22, %550, %551, %557, %22, %553, %554, %555, %556, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %559 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %560 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %561 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */"
    %562 = emitc.call_opaque "__Runtime_dma_createio_4"(%97, %558, %559, %560, %561) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %563 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %564 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %565 = emitc.call_opaque "__runtime_buffer_arg"(%104) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %566 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %567 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %568 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %569 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %570 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %571 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %100, %565, %24, %563, %564, %570, %22, %566, %567, %568, %569, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %572 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %573 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %574 = emitc.call_opaque "__runtime_buffer_arg"(%103) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %575 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %576 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %577 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %578 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %579 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %580 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %100, %574, %22, %572, %573, %579, %22, %575, %576, %577, %578, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %581 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %582 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %583 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,3), direction=S2MM */"
    %584 = emitc.call_opaque "__Runtime_dma_createio_4"(%100, %580, %581, %582, %583) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %585 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %586 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %587 = emitc.call_opaque "__runtime_buffer_arg"(%112) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %588 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %589 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %590 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %591 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %592 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %593 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %108, %587, %24, %585, %586, %592, %22, %588, %589, %590, %591, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %594 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %595 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %596 = emitc.call_opaque "__runtime_buffer_arg"(%111) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %597 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %598 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %599 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %600 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %601 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %602 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %108, %596, %22, %594, %595, %601, %22, %597, %598, %599, %600, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %603 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %604 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %605 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,4), direction=S2MM */"
    %606 = emitc.call_opaque "__Runtime_dma_createio_4"(%108, %602, %603, %604, %605) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %607 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %608 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %609 = emitc.call_opaque "__runtime_buffer_arg"(%120) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %610 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %611 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %612 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %613 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %614 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %615 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %116, %609, %24, %607, %608, %614, %22, %610, %611, %612, %613, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %616 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %617 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %618 = emitc.call_opaque "__runtime_buffer_arg"(%119) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %619 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %620 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %621 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %622 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %623 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %624 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %116, %618, %22, %616, %617, %623, %22, %619, %620, %621, %622, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %625 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %626 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %627 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,5), direction=S2MM */"
    %628 = emitc.call_opaque "__Runtime_dma_createio_4"(%116, %624, %625, %626, %627) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %629 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %630 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %631 = emitc.call_opaque "__runtime_buffer_arg"(%128) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %632 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %633 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %634 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %635 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %636 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %637 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %124, %631, %24, %629, %630, %636, %22, %632, %633, %634, %635, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %638 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %639 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %640 = emitc.call_opaque "__runtime_buffer_arg"(%127) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %641 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %642 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %643 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %644 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %645 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %646 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %124, %640, %22, %638, %639, %645, %22, %641, %642, %643, %644, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %647 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %648 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %649 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,6), direction=S2MM */"
    %650 = emitc.call_opaque "__Runtime_dma_createio_4"(%124, %646, %647, %648, %649) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %651 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %652 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %653 = emitc.call_opaque "__runtime_buffer_arg"(%132) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %654 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %655 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %656 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %657 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %658 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %659 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %135, %653, %22, %651, %652, %658, %22, %654, %655, %656, %657, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %660 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %661 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %662 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */"
    %663 = emitc.call_opaque "__Runtime_dma_createio_4"(%135, %659, %660, %661, %662) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %664 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %665 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %666 = emitc.call_opaque "__runtime_buffer_arg"(%142) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %667 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %668 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %669 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %670 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %671 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %672 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %138, %666, %24, %664, %665, %671, %22, %667, %668, %669, %670, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %673 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %674 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %675 = emitc.call_opaque "__runtime_buffer_arg"(%141) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %676 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %677 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %678 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %679 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %680 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %681 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %138, %675, %22, %673, %674, %680, %22, %676, %677, %678, %679, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %682 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %683 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %684 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,3), direction=S2MM */"
    %685 = emitc.call_opaque "__Runtime_dma_createio_4"(%138, %681, %682, %683, %684) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %686 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %687 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %688 = emitc.call_opaque "__runtime_buffer_arg"(%150) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %689 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %690 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %691 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %692 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %693 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %694 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %146, %688, %24, %686, %687, %693, %22, %689, %690, %691, %692, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %695 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %696 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %697 = emitc.call_opaque "__runtime_buffer_arg"(%149) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %698 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %699 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %700 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %701 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %702 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %703 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %146, %697, %22, %695, %696, %702, %22, %698, %699, %700, %701, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %704 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %705 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %706 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,4), direction=S2MM */"
    %707 = emitc.call_opaque "__Runtime_dma_createio_4"(%146, %703, %704, %705, %706) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %708 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %709 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %710 = emitc.call_opaque "__runtime_buffer_arg"(%158) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %711 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %712 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %713 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %714 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %715 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %716 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %154, %710, %24, %708, %709, %715, %22, %711, %712, %713, %714, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %717 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %718 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %719 = emitc.call_opaque "__runtime_buffer_arg"(%157) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %720 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %721 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %722 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %723 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %724 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %725 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %154, %719, %22, %717, %718, %724, %22, %720, %721, %722, %723, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %726 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %727 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %728 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,5), direction=S2MM */"
    %729 = emitc.call_opaque "__Runtime_dma_createio_4"(%154, %725, %726, %727, %728) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %730 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %731 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %732 = emitc.call_opaque "__runtime_buffer_arg"(%166) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %733 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %734 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %735 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %736 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %737 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %738 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %162, %732, %24, %730, %731, %737, %22, %733, %734, %735, %736, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %739 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %740 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %741 = emitc.call_opaque "__runtime_buffer_arg"(%165) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %742 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %743 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %744 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %745 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %746 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %747 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %162, %741, %22, %739, %740, %746, %22, %742, %743, %744, %745, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %748 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %749 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %750 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,6), direction=S2MM */"
    %751 = emitc.call_opaque "__Runtime_dma_createio_4"(%162, %747, %748, %749, %750) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %752 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %753 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %754 = emitc.call_opaque "__runtime_buffer_arg"(%170) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %755 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %756 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %757 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %758 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %759 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %760 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %761 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %762 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %763 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %764 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %765 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %766 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %767 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %768 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %769 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %29, %754, %24, %752, %753, %768, %22, %755, %756, %757, %758, %23, %759, %760, %761, %762, %763, %764, %765, %766, %767) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %770 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %771 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %772 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(0,0), direction=MM2S */"
    %773 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %769, %770, %771, %772) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %774 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %775 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %776 = emitc.call_opaque "__runtime_buffer_arg"(%172) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %777 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %778 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %779 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %780 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %781 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %782 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %32, %776, %20, %774, %775, %781, %22, %777, %778, %779, %780, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %783 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %784 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %785 = emitc.call_opaque "__runtime_buffer_arg"(%171) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %786 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %787 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %788 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %789 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %790 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %791 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %32, %785, %21, %783, %784, %790, %22, %786, %787, %788, %789, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %792 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %793 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %794 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,3), direction=S2MM */"
    %795 = emitc.call_opaque "__Runtime_dma_createio_4"(%32, %791, %792, %793, %794) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %796 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %797 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %798 = emitc.call_opaque "__runtime_buffer_arg"(%175) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %799 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %800 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %801 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %802 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %803 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %804 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %62, %798, %20, %796, %797, %803, %22, %799, %800, %801, %802, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %805 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %806 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %807 = emitc.call_opaque "__runtime_buffer_arg"(%174) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %808 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %809 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %810 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %811 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %812 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %813 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %62, %807, %21, %805, %806, %812, %22, %808, %809, %810, %811, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %814 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %815 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %816 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,3), direction=S2MM */"
    %817 = emitc.call_opaque "__Runtime_dma_createio_4"(%62, %813, %814, %815, %816) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %818 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %819 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %820 = emitc.call_opaque "__runtime_buffer_arg"(%178) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %821 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %822 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %823 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %824 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %825 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %826 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %100, %820, %20, %818, %819, %825, %22, %821, %822, %823, %824, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %827 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %828 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %829 = emitc.call_opaque "__runtime_buffer_arg"(%177) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %830 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %831 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %832 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %833 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %834 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %835 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %100, %829, %21, %827, %828, %834, %22, %830, %831, %832, %833, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %836 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %837 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %838 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,3), direction=S2MM */"
    %839 = emitc.call_opaque "__Runtime_dma_createio_4"(%100, %835, %836, %837, %838) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %840 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %841 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %842 = emitc.call_opaque "__runtime_buffer_arg"(%181) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %843 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %844 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %845 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %846 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %847 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %848 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %138, %842, %20, %840, %841, %847, %22, %843, %844, %845, %846, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %849 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %850 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %851 = emitc.call_opaque "__runtime_buffer_arg"(%180) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %852 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %853 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %854 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %855 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %856 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %857 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %138, %851, %21, %849, %850, %856, %22, %852, %853, %854, %855, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %858 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %859 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %860 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,3), direction=S2MM */"
    %861 = emitc.call_opaque "__Runtime_dma_createio_4"(%138, %857, %858, %859, %860) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=1, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */"
    %862 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %863 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %864 = emitc.call_opaque "__runtime_buffer_arg"(%189) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %865 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %866 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %867 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %868 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %869 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %870 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %32, %864, %19, %862, %863, %869, %24, %865, %866, %867, %868, %21) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=1, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */"
    %871 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %872 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %873 = emitc.call_opaque "__runtime_buffer_arg"(%188) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %874 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %875 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %876 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %877 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %878 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %879 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %32, %873, %18, %871, %872, %878, %24, %874, %875, %876, %877, %21) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %880 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %881 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %882 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,3), direction=MM2S */"
    %883 = emitc.call_opaque "__Runtime_dma_createio_4"(%32, %879, %880, %881, %882) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=2, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %884 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %885 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %886 = emitc.call_opaque "__runtime_buffer_arg"(%194) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %887 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %888 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %889 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %890 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %891 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %892 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %62, %886, %19, %884, %885, %891, %21, %887, %888, %889, %890, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=2, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %893 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %894 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %895 = emitc.call_opaque "__runtime_buffer_arg"(%193) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %896 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %897 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %898 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %899 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %900 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %901 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %62, %895, %18, %893, %894, %900, %21, %896, %897, %898, %899, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %902 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %903 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %904 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,3), direction=MM2S */"
    %905 = emitc.call_opaque "__Runtime_dma_createio_4"(%62, %901, %902, %903, %904) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=3, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %906 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %907 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %908 = emitc.call_opaque "__runtime_buffer_arg"(%199) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %909 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %910 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %911 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %912 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %913 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %914 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %100, %908, %19, %906, %907, %913, %20, %909, %910, %911, %912, %18) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=3, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %915 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %916 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %917 = emitc.call_opaque "__runtime_buffer_arg"(%198) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %918 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %919 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %920 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %921 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %922 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %923 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %100, %917, %18, %915, %916, %922, %20, %918, %919, %920, %921, %18) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %924 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %925 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %926 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,3), direction=MM2S */"
    %927 = emitc.call_opaque "__Runtime_dma_createio_4"(%100, %923, %924, %925, %926) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=4, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %928 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %929 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %930 = emitc.call_opaque "__runtime_buffer_arg"(%204) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %931 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %932 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %933 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %934 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %935 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %936 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %138, %930, %19, %928, %929, %935, %18, %931, %932, %933, %934, %19) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=4, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %937 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %938 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %939 = emitc.call_opaque "__runtime_buffer_arg"(%203) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %940 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %941 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %942 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %943 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %944 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %945 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %138, %939, %18, %937, %938, %944, %18, %940, %941, %942, %943, %19) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %946 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %947 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %948 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,3), direction=MM2S */"
    %949 = emitc.call_opaque "__Runtime_dma_createio_4"(%138, %945, %946, %947, %948) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %950 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %951 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %952 = emitc.call_opaque "__runtime_buffer_arg"(%208) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %953 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %954 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %955 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %956 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %957 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %958 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %959 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %960 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %961 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %962 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %963 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %964 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %965 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %966 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %967 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %59, %952, %24, %950, %951, %966, %22, %953, %954, %955, %956, %23, %957, %958, %959, %960, %961, %962, %963, %964, %965) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %968 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %969 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %970 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(1,0), direction=MM2S */"
    %971 = emitc.call_opaque "__Runtime_dma_createio_4"(%59, %967, %968, %969, %970) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %972 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %973 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %974 = emitc.call_opaque "__runtime_buffer_arg"(%212) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %975 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %976 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %977 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %978 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %979 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %980 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %38, %974, %20, %972, %973, %979, %22, %975, %976, %977, %978, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %981 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %982 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %983 = emitc.call_opaque "__runtime_buffer_arg"(%211) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %984 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %985 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %986 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %987 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %988 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %989 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %38, %983, %21, %981, %982, %988, %22, %984, %985, %986, %987, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %990 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %991 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %992 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,4), direction=S2MM */"
    %993 = emitc.call_opaque "__Runtime_dma_createio_4"(%38, %989, %990, %991, %992) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %994 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %995 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %996 = emitc.call_opaque "__runtime_buffer_arg"(%217) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %997 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %998 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %999 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1000 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1001 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1002 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %70, %996, %20, %994, %995, %1001, %22, %997, %998, %999, %1000, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1003 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1004 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1005 = emitc.call_opaque "__runtime_buffer_arg"(%216) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1006 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1007 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1008 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1009 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1010 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1011 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %70, %1005, %21, %1003, %1004, %1010, %22, %1006, %1007, %1008, %1009, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1012 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1013 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1014 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,4), direction=S2MM */"
    %1015 = emitc.call_opaque "__Runtime_dma_createio_4"(%70, %1011, %1012, %1013, %1014) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1016 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1017 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1018 = emitc.call_opaque "__runtime_buffer_arg"(%222) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1019 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1020 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1021 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1022 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1023 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1024 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %108, %1018, %20, %1016, %1017, %1023, %22, %1019, %1020, %1021, %1022, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1025 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1026 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1027 = emitc.call_opaque "__runtime_buffer_arg"(%221) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1028 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1029 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1030 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1031 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1032 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1033 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %108, %1027, %21, %1025, %1026, %1032, %22, %1028, %1029, %1030, %1031, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1034 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1035 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1036 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,4), direction=S2MM */"
    %1037 = emitc.call_opaque "__Runtime_dma_createio_4"(%108, %1033, %1034, %1035, %1036) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1038 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1039 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1040 = emitc.call_opaque "__runtime_buffer_arg"(%227) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1041 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1042 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1043 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1044 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1045 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1046 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %146, %1040, %20, %1038, %1039, %1045, %22, %1041, %1042, %1043, %1044, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1047 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1048 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1049 = emitc.call_opaque "__runtime_buffer_arg"(%226) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1050 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1051 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1052 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1053 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1054 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1055 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %146, %1049, %21, %1047, %1048, %1054, %22, %1050, %1051, %1052, %1053, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1056 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1057 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1058 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,4), direction=S2MM */"
    %1059 = emitc.call_opaque "__Runtime_dma_createio_4"(%146, %1055, %1056, %1057, %1058) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=5, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */"
    %1060 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1061 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1062 = emitc.call_opaque "__runtime_buffer_arg"(%235) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1063 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1064 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1065 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1066 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1067 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1068 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %38, %1062, %19, %1060, %1061, %1067, %19, %1063, %1064, %1065, %1066, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=5, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */"
    %1069 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1070 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1071 = emitc.call_opaque "__runtime_buffer_arg"(%234) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1072 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1073 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1074 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1075 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1076 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1077 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %38, %1071, %18, %1069, %1070, %1076, %19, %1072, %1073, %1074, %1075, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1078 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1079 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1080 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,4), direction=MM2S */"
    %1081 = emitc.call_opaque "__Runtime_dma_createio_4"(%38, %1077, %1078, %1079, %1080) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=6, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1082 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1083 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1084 = emitc.call_opaque "__runtime_buffer_arg"(%240) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1085 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1086 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1087 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1088 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1089 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1090 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %70, %1084, %19, %1082, %1083, %1089, %9, %1085, %1086, %1087, %1088, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=6, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1091 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1092 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1093 = emitc.call_opaque "__runtime_buffer_arg"(%239) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1094 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1095 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1096 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1097 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1098 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1099 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %70, %1093, %18, %1091, %1092, %1098, %9, %1094, %1095, %1096, %1097, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1100 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1101 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1102 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,4), direction=MM2S */"
    %1103 = emitc.call_opaque "__Runtime_dma_createio_4"(%70, %1099, %1100, %1101, %1102) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=7, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1104 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1105 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1106 = emitc.call_opaque "__runtime_buffer_arg"(%245) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1107 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1108 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1109 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1110 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1111 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1112 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %108, %1106, %19, %1104, %1105, %1111, %10, %1107, %1108, %1109, %1110, %7) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=7, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1113 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1114 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1115 = emitc.call_opaque "__runtime_buffer_arg"(%244) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1116 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1117 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1118 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1119 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1120 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1121 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %108, %1115, %18, %1113, %1114, %1120, %10, %1116, %1117, %1118, %1119, %7) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1122 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1123 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1124 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,4), direction=MM2S */"
    %1125 = emitc.call_opaque "__Runtime_dma_createio_4"(%108, %1121, %1122, %1123, %1124) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=8, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1126 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1127 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1128 = emitc.call_opaque "__runtime_buffer_arg"(%250) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1129 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1130 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1131 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1132 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1133 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1134 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %146, %1128, %19, %1126, %1127, %1133, %8, %1129, %1130, %1131, %1132, %6) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=8, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1135 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1136 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1137 = emitc.call_opaque "__runtime_buffer_arg"(%249) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1138 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1139 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1140 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1141 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1142 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1143 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %146, %1137, %18, %1135, %1136, %1142, %8, %1138, %1139, %1140, %1141, %6) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1144 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1145 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1146 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,4), direction=MM2S */"
    %1147 = emitc.call_opaque "__Runtime_dma_createio_4"(%146, %1143, %1144, %1145, %1146) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %1148 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %1149 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1150 = emitc.call_opaque "__runtime_buffer_arg"(%254) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1151 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1152 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1153 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1154 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1155 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1156 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1157 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %1158 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %1159 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %1160 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %1161 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1162 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %1163 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1164 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1165 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %1150, %24, %1148, %1149, %1164, %22, %1151, %1152, %1153, %1154, %23, %1155, %1156, %1157, %1158, %1159, %1160, %1161, %1162, %1163) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1166 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1167 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1168 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */"
    %1169 = emitc.call_opaque "__Runtime_dma_createio_4"(%97, %1165, %1166, %1167, %1168) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1170 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1171 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1172 = emitc.call_opaque "__runtime_buffer_arg"(%258) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1173 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1174 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1175 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1176 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1177 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1178 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %44, %1172, %20, %1170, %1171, %1177, %22, %1173, %1174, %1175, %1176, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1179 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1180 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1181 = emitc.call_opaque "__runtime_buffer_arg"(%257) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1182 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1183 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1184 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1185 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1186 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1187 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %44, %1181, %21, %1179, %1180, %1186, %22, %1182, %1183, %1184, %1185, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1188 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1189 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1190 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,5), direction=S2MM */"
    %1191 = emitc.call_opaque "__Runtime_dma_createio_4"(%44, %1187, %1188, %1189, %1190) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1192 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1193 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1194 = emitc.call_opaque "__runtime_buffer_arg"(%263) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1195 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1196 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1197 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1198 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1199 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1200 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %78, %1194, %20, %1192, %1193, %1199, %22, %1195, %1196, %1197, %1198, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1201 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1202 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1203 = emitc.call_opaque "__runtime_buffer_arg"(%262) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1204 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1205 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1206 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1207 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1208 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1209 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %78, %1203, %21, %1201, %1202, %1208, %22, %1204, %1205, %1206, %1207, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1210 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1211 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1212 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,5), direction=S2MM */"
    %1213 = emitc.call_opaque "__Runtime_dma_createio_4"(%78, %1209, %1210, %1211, %1212) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1214 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1215 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1216 = emitc.call_opaque "__runtime_buffer_arg"(%268) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1217 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1218 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1219 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1220 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1221 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1222 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %116, %1216, %20, %1214, %1215, %1221, %22, %1217, %1218, %1219, %1220, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1223 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1224 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1225 = emitc.call_opaque "__runtime_buffer_arg"(%267) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1226 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1227 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1228 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1229 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1230 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1231 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %116, %1225, %21, %1223, %1224, %1230, %22, %1226, %1227, %1228, %1229, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1232 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1233 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1234 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,5), direction=S2MM */"
    %1235 = emitc.call_opaque "__Runtime_dma_createio_4"(%116, %1231, %1232, %1233, %1234) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1236 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1237 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1238 = emitc.call_opaque "__runtime_buffer_arg"(%273) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1239 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1240 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1241 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1242 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1243 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1244 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %154, %1238, %20, %1236, %1237, %1243, %22, %1239, %1240, %1241, %1242, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1245 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1246 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1247 = emitc.call_opaque "__runtime_buffer_arg"(%272) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1248 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1249 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1250 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1251 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1252 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1253 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %154, %1247, %21, %1245, %1246, %1252, %22, %1248, %1249, %1250, %1251, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1254 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1255 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1256 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,5), direction=S2MM */"
    %1257 = emitc.call_opaque "__Runtime_dma_createio_4"(%154, %1253, %1254, %1255, %1256) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=9, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %1258 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1259 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1260 = emitc.call_opaque "__runtime_buffer_arg"(%281) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1261 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1262 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1263 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1264 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1265 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1266 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %44, %1260, %19, %1258, %1259, %1265, %7, %1261, %1262, %1263, %1264, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=9, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %1267 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1268 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1269 = emitc.call_opaque "__runtime_buffer_arg"(%280) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1270 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1271 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1272 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1273 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1274 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1275 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %44, %1269, %18, %1267, %1268, %1274, %7, %1270, %1271, %1272, %1273, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1276 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1277 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1278 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,5), direction=MM2S */"
    %1279 = emitc.call_opaque "__Runtime_dma_createio_4"(%44, %1275, %1276, %1277, %1278) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=10, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %1280 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1281 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1282 = emitc.call_opaque "__runtime_buffer_arg"(%286) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1283 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1284 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1285 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1286 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1287 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1288 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %78, %1282, %19, %1280, %1281, %1287, %6, %1283, %1284, %1285, %1286, %18) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=10, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %1289 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1290 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1291 = emitc.call_opaque "__runtime_buffer_arg"(%285) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1292 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1293 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1294 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1295 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1296 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1297 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %78, %1291, %18, %1289, %1290, %1296, %6, %1292, %1293, %1294, %1295, %18) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1298 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1299 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1300 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,5), direction=MM2S */"
    %1301 = emitc.call_opaque "__Runtime_dma_createio_4"(%78, %1297, %1298, %1299, %1300) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=11, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %1302 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1303 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1304 = emitc.call_opaque "__runtime_buffer_arg"(%291) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1305 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1306 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1307 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1308 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1309 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1310 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %116, %1304, %19, %1302, %1303, %1309, %5, %1305, %1306, %1307, %1308, %19) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=11, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %1311 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1312 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1313 = emitc.call_opaque "__runtime_buffer_arg"(%290) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1314 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1315 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1316 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1317 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1318 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1319 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %116, %1313, %18, %1311, %1312, %1318, %5, %1314, %1315, %1316, %1317, %19) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1320 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1321 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1322 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,5), direction=MM2S */"
    %1323 = emitc.call_opaque "__Runtime_dma_createio_4"(%116, %1319, %1320, %1321, %1322) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=12, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %1324 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1325 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1326 = emitc.call_opaque "__runtime_buffer_arg"(%296) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1327 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1328 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1329 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1330 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1331 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1332 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %154, %1326, %19, %1324, %1325, %1331, %4, %1327, %1328, %1329, %1330, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=12, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %1333 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1334 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1335 = emitc.call_opaque "__runtime_buffer_arg"(%295) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1336 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1337 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1338 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1339 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1340 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1341 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %154, %1335, %18, %1333, %1334, %1340, %4, %1336, %1337, %1338, %1339, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1342 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1343 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1344 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,5), direction=MM2S */"
    %1345 = emitc.call_opaque "__Runtime_dma_createio_4"(%154, %1341, %1342, %1343, %1344) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %1346 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %1347 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1348 = emitc.call_opaque "__runtime_buffer_arg"(%300) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1349 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1350 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1351 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1352 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1353 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1354 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1355 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %1356 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %1357 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %1358 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %1359 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1360 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %1361 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1362 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1363 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1348, %5, %1346, %1347, %1362, %22, %1349, %1350, %1351, %1352, %23, %1353, %1354, %1355, %1356, %1357, %1358, %1359, %1360, %1361) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1364 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1365 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %1366 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=11, tile=(3,0), direction=MM2S */"
    %1367 = emitc.call_opaque "__Runtime_dma_createio_4"(%135, %1363, %1364, %1365, %1366) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1368 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1369 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1370 = emitc.call_opaque "__runtime_buffer_arg"(%304) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1371 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1372 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1373 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1374 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1375 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1376 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %50, %1370, %20, %1368, %1369, %1375, %22, %1371, %1372, %1373, %1374, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1377 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1378 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1379 = emitc.call_opaque "__runtime_buffer_arg"(%303) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1380 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1381 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1382 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1383 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1384 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1385 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %50, %1379, %21, %1377, %1378, %1384, %22, %1380, %1381, %1382, %1383, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1386 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1387 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1388 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,6), direction=S2MM */"
    %1389 = emitc.call_opaque "__Runtime_dma_createio_4"(%50, %1385, %1386, %1387, %1388) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1390 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1391 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1392 = emitc.call_opaque "__runtime_buffer_arg"(%309) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1393 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1394 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1395 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1396 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1397 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1398 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %86, %1392, %20, %1390, %1391, %1397, %22, %1393, %1394, %1395, %1396, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1399 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1400 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1401 = emitc.call_opaque "__runtime_buffer_arg"(%308) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1402 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1403 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1404 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1405 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1406 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1407 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %86, %1401, %21, %1399, %1400, %1406, %22, %1402, %1403, %1404, %1405, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1408 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1409 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1410 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,6), direction=S2MM */"
    %1411 = emitc.call_opaque "__Runtime_dma_createio_4"(%86, %1407, %1408, %1409, %1410) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1412 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1413 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1414 = emitc.call_opaque "__runtime_buffer_arg"(%314) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1415 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1416 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1417 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1418 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1419 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1420 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %124, %1414, %20, %1412, %1413, %1419, %22, %1415, %1416, %1417, %1418, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1421 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1422 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1423 = emitc.call_opaque "__runtime_buffer_arg"(%313) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1424 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1425 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1426 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1427 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1428 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1429 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %124, %1423, %21, %1421, %1422, %1428, %22, %1424, %1425, %1426, %1427, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1430 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1431 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1432 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,6), direction=S2MM */"
    %1433 = emitc.call_opaque "__Runtime_dma_createio_4"(%124, %1429, %1430, %1431, %1432) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1434 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1435 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1436 = emitc.call_opaque "__runtime_buffer_arg"(%319) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1437 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1438 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1439 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1440 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1441 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1442 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %162, %1436, %20, %1434, %1435, %1441, %22, %1437, %1438, %1439, %1440, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1443 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1444 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1445 = emitc.call_opaque "__runtime_buffer_arg"(%318) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1446 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1447 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1448 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1449 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1450 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1451 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %162, %1445, %21, %1443, %1444, %1450, %22, %1446, %1447, %1448, %1449, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1452 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1453 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1454 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,6), direction=S2MM */"
    %1455 = emitc.call_opaque "__Runtime_dma_createio_4"(%162, %1451, %1452, %1453, %1454) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=13, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1456 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1457 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1458 = emitc.call_opaque "__runtime_buffer_arg"(%327) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1459 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1460 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1461 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1462 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1463 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1464 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %50, %1458, %19, %1456, %1457, %1463, %3, %1459, %1460, %1461, %1462, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=13, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1465 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1466 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1467 = emitc.call_opaque "__runtime_buffer_arg"(%326) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1468 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1469 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1470 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1471 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1472 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1473 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %50, %1467, %18, %1465, %1466, %1472, %3, %1468, %1469, %1470, %1471, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1474 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1475 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1476 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,6), direction=MM2S */"
    %1477 = emitc.call_opaque "__Runtime_dma_createio_4"(%50, %1473, %1474, %1475, %1476) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=14, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1478 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1479 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1480 = emitc.call_opaque "__runtime_buffer_arg"(%332) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1481 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1482 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1483 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1484 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1485 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1486 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %86, %1480, %19, %1478, %1479, %1485, %2, %1481, %1482, %1483, %1484, %7) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=14, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1487 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1488 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1489 = emitc.call_opaque "__runtime_buffer_arg"(%331) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1490 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1491 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1492 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1493 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1494 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1495 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %86, %1489, %18, %1487, %1488, %1494, %2, %1490, %1491, %1492, %1493, %7) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1496 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1497 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1498 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,6), direction=MM2S */"
    %1499 = emitc.call_opaque "__Runtime_dma_createio_4"(%86, %1495, %1496, %1497, %1498) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=15, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1500 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1501 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1502 = emitc.call_opaque "__runtime_buffer_arg"(%337) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1503 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1504 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1505 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1506 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1507 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1508 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %124, %1502, %19, %1500, %1501, %1507, %1, %1503, %1504, %1505, %1506, %6) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=15, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1509 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1510 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1511 = emitc.call_opaque "__runtime_buffer_arg"(%336) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1512 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1513 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1514 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1515 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1516 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1517 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %124, %1511, %18, %1509, %1510, %1516, %1, %1512, %1513, %1514, %1515, %6) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1518 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1519 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1520 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,6), direction=MM2S */"
    %1521 = emitc.call_opaque "__Runtime_dma_createio_4"(%124, %1517, %1518, %1519, %1520) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=16, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %1522 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1523 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1524 = emitc.call_opaque "__runtime_buffer_arg"(%342) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1525 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1526 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1527 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1528 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1529 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1530 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %162, %1524, %19, %1522, %1523, %1529, %14, %1525, %1526, %1527, %1528, %5) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=16, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %1531 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1532 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1533 = emitc.call_opaque "__runtime_buffer_arg"(%341) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1534 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1535 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1536 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1537 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1538 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1539 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %162, %1533, %18, %1531, %1532, %1538, %14, %1534, %1535, %1536, %1537, %5) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1540 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1541 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1542 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,6), direction=MM2S */"
    %1543 = emitc.call_opaque "__Runtime_dma_createio_4"(%162, %1539, %1540, %1541, %1542) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %1544 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1545 = emitc.call_opaque "__Runtime_startio"(%arg0, %382, %35, %1544) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1546 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1547 = emitc.call_opaque "__Runtime_startio"(%arg0, %404, %41, %1546) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1548 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1549 = emitc.call_opaque "__Runtime_startio"(%arg0, %426, %47, %1548) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1550 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1551 = emitc.call_opaque "__Runtime_startio"(%arg0, %448, %53, %1550) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1552 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1553 = emitc.call_opaque "__Runtime_startio"(%arg0, %483, %67, %1552) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1554 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1555 = emitc.call_opaque "__Runtime_startio"(%arg0, %505, %75, %1554) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1556 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1557 = emitc.call_opaque "__Runtime_startio"(%arg0, %527, %83, %1556) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1558 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1559 = emitc.call_opaque "__Runtime_startio"(%arg0, %549, %91, %1558) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1560 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1561 = emitc.call_opaque "__Runtime_startio"(%arg0, %584, %105, %1560) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1562 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1563 = emitc.call_opaque "__Runtime_startio"(%arg0, %606, %113, %1562) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1564 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1565 = emitc.call_opaque "__Runtime_startio"(%arg0, %628, %121, %1564) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1566 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1567 = emitc.call_opaque "__Runtime_startio"(%arg0, %650, %129, %1566) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1568 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1569 = emitc.call_opaque "__Runtime_startio"(%arg0, %685, %143, %1568) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1570 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1571 = emitc.call_opaque "__Runtime_startio"(%arg0, %707, %151, %1570) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1572 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1573 = emitc.call_opaque "__Runtime_startio"(%arg0, %729, %159, %1572) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1574 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1575 = emitc.call_opaque "__Runtime_startio"(%arg0, %751, %167, %1574) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1576 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1577 = emitc.call_opaque "__Runtime_startio"(%arg0, %795, %173, %1576) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1578 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1579 = emitc.call_opaque "__Runtime_startio"(%arg0, %817, %176, %1578) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1580 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1581 = emitc.call_opaque "__Runtime_startio"(%arg0, %839, %179, %1580) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1582 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1583 = emitc.call_opaque "__Runtime_startio"(%arg0, %861, %182, %1582) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1584 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1585 = emitc.call_opaque "__Runtime_startio"(%arg0, %993, %213, %1584) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1586 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1587 = emitc.call_opaque "__Runtime_startio"(%arg0, %1015, %218, %1586) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1588 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1589 = emitc.call_opaque "__Runtime_startio"(%arg0, %1037, %223, %1588) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1590 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1591 = emitc.call_opaque "__Runtime_startio"(%arg0, %1059, %228, %1590) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1592 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1593 = emitc.call_opaque "__Runtime_startio"(%arg0, %1191, %259, %1592) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1594 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1595 = emitc.call_opaque "__Runtime_startio"(%arg0, %1213, %264, %1594) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1596 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1597 = emitc.call_opaque "__Runtime_startio"(%arg0, %1235, %269, %1596) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1598 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1599 = emitc.call_opaque "__Runtime_startio"(%arg0, %1257, %274, %1598) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1600 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1601 = emitc.call_opaque "__Runtime_startio"(%arg0, %1389, %305, %1600) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1602 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1603 = emitc.call_opaque "__Runtime_startio"(%arg0, %1411, %310, %1602) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1604 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1605 = emitc.call_opaque "__Runtime_startio"(%arg0, %1433, %315, %1604) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1606 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1607 = emitc.call_opaque "__Runtime_startio"(%arg0, %1455, %320, %1606) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1608 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1609 = emitc.call_opaque "__Runtime_startio"(%arg0, %883, %190, %1608) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1610 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1611 = emitc.call_opaque "__Runtime_startio"(%arg0, %905, %195, %1610) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1612 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1613 = emitc.call_opaque "__Runtime_startio"(%arg0, %927, %200, %1612) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1614 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1615 = emitc.call_opaque "__Runtime_startio"(%arg0, %949, %205, %1614) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1616 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1617 = emitc.call_opaque "__Runtime_startio"(%arg0, %1081, %236, %1616) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1618 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1619 = emitc.call_opaque "__Runtime_startio"(%arg0, %1103, %241, %1618) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1620 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1621 = emitc.call_opaque "__Runtime_startio"(%arg0, %1125, %246, %1620) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1622 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1623 = emitc.call_opaque "__Runtime_startio"(%arg0, %1147, %251, %1622) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1624 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1625 = emitc.call_opaque "__Runtime_startio"(%arg0, %1279, %282, %1624) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1626 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1627 = emitc.call_opaque "__Runtime_startio"(%arg0, %1301, %287, %1626) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1628 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1629 = emitc.call_opaque "__Runtime_startio"(%arg0, %1323, %292, %1628) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1630 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1631 = emitc.call_opaque "__Runtime_startio"(%arg0, %1345, %297, %1630) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1632 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1633 = emitc.call_opaque "__Runtime_startio"(%arg0, %1477, %328, %1632) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1634 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1635 = emitc.call_opaque "__Runtime_startio"(%arg0, %1499, %333, %1634) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1636 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1637 = emitc.call_opaque "__Runtime_startio"(%arg0, %1521, %338, %1636) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1638 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1639 = emitc.call_opaque "__Runtime_startio"(%arg0, %1543, %343, %1638) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1640 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1641 = emitc.call_opaque "__Runtime_startio"(%arg0, %360, %54, %1640) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1642 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1643 = emitc.call_opaque "__Runtime_startio"(%arg0, %461, %92, %1642) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1644 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1645 = emitc.call_opaque "__Runtime_startio"(%arg0, %562, %130, %1644) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1646 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1647 = emitc.call_opaque "__Runtime_startio"(%arg0, %663, %168, %1646) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1648 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1649 = emitc.call_opaque "__Runtime_startio"(%arg0, %773, %183, %1648) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1650 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1651 = emitc.call_opaque "__Runtime_startio"(%arg0, %971, %229, %1650) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1652 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1653 = emitc.call_opaque "__Runtime_startio"(%arg0, %1169, %275, %1652) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1654 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1655 = emitc.call_opaque "__Runtime_startio"(%arg0, %1367, %321, %1654) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    emitc.for %arg4 = %13 to %12 step %11 {
      %1656 = emitc.cast %arg4 : index to i32
      %1657 = emitc.mul %1656, %17 : (i32, i32) -> i32
      %1658 = emitc.add %1657, %16 : (i32, i32) -> i32
      %1659 = emitc.add %1657, %15 : (i32, i32) -> i32
      %1660 = emitc.add %1657, %14 : (i32, i32) -> i32
      emitc.verbatim "/* Allocated BD ID 4 for tile (3,0) */"
      %1661 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1662 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1663 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1664 = emitc.call_opaque "__runtime_buffer_arg"(%185) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1665 = emitc.cast %1658 : i32 to i64
      %1666 = emitc.call_opaque "__runtime_buffer_offset"(%1664, %1665) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1667 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1668 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1669 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1670 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1671 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1672 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1673 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1674 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1675 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1676 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1677 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1678 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1679 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1680 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1681 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1666, %19, %1662, %1663, %1680, %18, %1667, %1668, %1669, %1670, %23, %1671, %1672, %1673, %1674, %1675, %1676, %1677, %1678, %1679) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=3, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1682 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1683 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1684 = emitc.call_opaque "__runtime_buffer_arg"(%185) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1685 = emitc.cast %1659 : i32 to i64
      %1686 = emitc.call_opaque "__runtime_buffer_offset"(%1684, %1685) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1687 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1688 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1689 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1690 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1691 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1692 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1693 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1694 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1695 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1696 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1697 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1698 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1699 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1700 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1701 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1686, %18, %1682, %1683, %1700, %20, %1687, %1688, %1689, %1690, %23, %1691, %1692, %1693, %1694, %1695, %1696, %1697, %1698, %1699) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=2, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1702 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1703 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1704 = emitc.call_opaque "__runtime_buffer_arg"(%185) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1705 = emitc.cast %1660 : i32 to i64
      %1706 = emitc.call_opaque "__runtime_buffer_offset"(%1704, %1705) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1707 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1708 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1709 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1710 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1711 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1712 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1713 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1714 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1715 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1716 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1717 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1718 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1719 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1720 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1721 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1706, %20, %1702, %1703, %1720, %21, %1707, %1708, %1709, %1710, %23, %1711, %1712, %1713, %1714, %1715, %1716, %1717, %1718, %1719) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=2, len=3136, enable_packet=false, packet_id=1, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1722 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1723 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1724 = emitc.call_opaque "__runtime_buffer_arg"(%185) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1725 = emitc.cast %1657 : i32 to i64
      %1726 = emitc.call_opaque "__runtime_buffer_offset"(%1724, %1725) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1727 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1728 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1729 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1730 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1731 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1732 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1733 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1734 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1735 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1736 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1737 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1738 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1739 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1740 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1741 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1726, %21, %1722, %1723, %1740, %24, %1727, %1728, %1729, %1730, %23, %1731, %1732, %1733, %1734, %1735, %1736, %1737, %1738, %1739) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      %1742 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1743 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
      %1744 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
      emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,0), direction=S2MM */"
      emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */"
      emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %135, %1742, %1744) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
      %1745 = emitc.call_opaque "__Runtime_dma_createio_4"(%135, %1741, %1742, %1743, %1744) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
      %1746 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
      %1747 = emitc.call_opaque "__Runtime_startio"(%arg0, %1745, %1661, %1746) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
      %1748 = emitc.cast %arg4 : index to i32
      %1749 = emitc.mul %1748, %17 : (i32, i32) -> i32
      %1750 = emitc.add %1749, %16 : (i32, i32) -> i32
      %1751 = emitc.add %1749, %15 : (i32, i32) -> i32
      %1752 = emitc.add %1749, %14 : (i32, i32) -> i32
      emitc.verbatim "/* Allocated BD ID 5 for tile (3,0) */"
      %1753 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
      emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1754 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1755 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1756 = emitc.call_opaque "__runtime_buffer_arg"(%231) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1757 = emitc.cast %1750 : i32 to i64
      %1758 = emitc.call_opaque "__runtime_buffer_offset"(%1756, %1757) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1759 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1760 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1761 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1762 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1763 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1764 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1765 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1766 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1767 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1768 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1769 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1770 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1771 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1772 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1773 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1758, %6, %1754, %1755, %1772, %8, %1759, %1760, %1761, %1762, %23, %1763, %1764, %1765, %1766, %1767, %1768, %1769, %1770, %1771) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1774 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1775 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1776 = emitc.call_opaque "__runtime_buffer_arg"(%231) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1777 = emitc.cast %1751 : i32 to i64
      %1778 = emitc.call_opaque "__runtime_buffer_offset"(%1776, %1777) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1779 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1780 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1781 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1782 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1783 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1784 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1785 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1786 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1787 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1788 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1789 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1790 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1791 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1792 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1793 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1778, %7, %1774, %1775, %1792, %10, %1779, %1780, %1781, %1782, %23, %1783, %1784, %1785, %1786, %1787, %1788, %1789, %1790, %1791) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1794 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1795 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1796 = emitc.call_opaque "__runtime_buffer_arg"(%231) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1797 = emitc.cast %1752 : i32 to i64
      %1798 = emitc.call_opaque "__runtime_buffer_offset"(%1796, %1797) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1799 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1800 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1801 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1802 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1803 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1804 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1805 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1806 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1807 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1808 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1809 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1810 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1811 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1812 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1813 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1798, %8, %1794, %1795, %1812, %9, %1799, %1800, %1801, %1802, %23, %1803, %1804, %1805, %1806, %1807, %1808, %1809, %1810, %1811) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=7, len=3136, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1814 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1815 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1816 = emitc.call_opaque "__runtime_buffer_arg"(%231) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1817 = emitc.cast %1749 : i32 to i64
      %1818 = emitc.call_opaque "__runtime_buffer_offset"(%1816, %1817) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1819 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1820 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1821 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1822 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1823 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1824 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1825 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1826 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1827 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1828 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1829 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1830 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1831 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1832 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1833 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %135, %1818, %10, %1814, %1815, %1832, %19, %1819, %1820, %1821, %1822, %23, %1823, %1824, %1825, %1826, %1827, %1828, %1829, %1830, %1831) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      %1834 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
      %1835 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1836 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
      emitc.verbatim "/* Create IO: channel_id=1, bd_id=7, tile=(3,0), direction=S2MM */"
      emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */"
      emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %135, %1834, %1836) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
      %1837 = emitc.call_opaque "__Runtime_dma_createio_4"(%135, %1833, %1834, %1835, %1836) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
      %1838 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
      %1839 = emitc.call_opaque "__Runtime_startio"(%arg0, %1837, %1753, %1838) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
      %1840 = emitc.cast %arg4 : index to i32
      %1841 = emitc.mul %1840, %17 : (i32, i32) -> i32
      %1842 = emitc.add %1841, %16 : (i32, i32) -> i32
      %1843 = emitc.add %1841, %15 : (i32, i32) -> i32
      %1844 = emitc.add %1841, %14 : (i32, i32) -> i32
      emitc.verbatim "/* Allocated BD ID 4 for tile (2,0) */"
      %1845 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      emitc.verbatim "/* DMA BD Config: bd_id=6, len=3136, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1846 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1847 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1848 = emitc.call_opaque "__runtime_buffer_arg"(%277) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1849 = emitc.cast %1842 : i32 to i64
      %1850 = emitc.call_opaque "__runtime_buffer_offset"(%1848, %1849) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1851 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1852 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1853 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1854 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1855 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1856 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1857 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1858 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1859 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1860 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1861 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1862 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1863 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1864 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1865 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %1850, %9, %1846, %1847, %1864, %4, %1851, %1852, %1853, %1854, %23, %1855, %1856, %1857, %1858, %1859, %1860, %1861, %1862, %1863) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1866 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1867 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1868 = emitc.call_opaque "__runtime_buffer_arg"(%277) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1869 = emitc.cast %1843 : i32 to i64
      %1870 = emitc.call_opaque "__runtime_buffer_offset"(%1868, %1869) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1871 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1872 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1873 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1874 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1875 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1876 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1877 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1878 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1879 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1880 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1881 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1882 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1883 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1884 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1885 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %1870, %19, %1866, %1867, %1884, %5, %1871, %1872, %1873, %1874, %23, %1875, %1876, %1877, %1878, %1879, %1880, %1881, %1882, %1883) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1886 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1887 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1888 = emitc.call_opaque "__runtime_buffer_arg"(%277) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1889 = emitc.cast %1844 : i32 to i64
      %1890 = emitc.call_opaque "__runtime_buffer_offset"(%1888, %1889) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1891 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1892 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1893 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1894 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1895 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1896 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1897 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1898 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1899 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1900 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1901 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1902 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1903 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1904 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1905 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %1890, %18, %1886, %1887, %1904, %6, %1891, %1892, %1893, %1894, %23, %1895, %1896, %1897, %1898, %1899, %1900, %1901, %1902, %1903) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1906 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1907 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1908 = emitc.call_opaque "__runtime_buffer_arg"(%277) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1909 = emitc.cast %1841 : i32 to i64
      %1910 = emitc.call_opaque "__runtime_buffer_offset"(%1908, %1909) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1911 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1912 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1913 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1914 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1915 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1916 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1917 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1918 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1919 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1920 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1921 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1922 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1923 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1924 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1925 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %1910, %20, %1906, %1907, %1924, %7, %1911, %1912, %1913, %1914, %23, %1915, %1916, %1917, %1918, %1919, %1920, %1921, %1922, %1923) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      %1926 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1927 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1928 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
      emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(2,0), direction=S2MM */"
      emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=0 dir=S2MM */"
      emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %97, %1926, %1928) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
      %1929 = emitc.call_opaque "__Runtime_dma_createio_4"(%97, %1925, %1926, %1927, %1928) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
      %1930 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
      %1931 = emitc.call_opaque "__Runtime_startio"(%arg0, %1929, %1845, %1930) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
      %1932 = emitc.cast %arg4 : index to i32
      %1933 = emitc.mul %1932, %17 : (i32, i32) -> i32
      %1934 = emitc.add %1933, %16 : (i32, i32) -> i32
      %1935 = emitc.add %1933, %15 : (i32, i32) -> i32
      %1936 = emitc.add %1933, %14 : (i32, i32) -> i32
      emitc.verbatim "/* Allocated BD ID 5 for tile (2,0) */"
      %1937 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
      emitc.verbatim "/* DMA BD Config: bd_id=11, len=3136, enable_packet=false, packet_id=16, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1938 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1939 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1940 = emitc.call_opaque "__runtime_buffer_arg"(%323) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1941 = emitc.cast %1934 : i32 to i64
      %1942 = emitc.call_opaque "__runtime_buffer_offset"(%1940, %1941) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1943 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1944 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1945 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1946 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1947 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1948 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1949 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1950 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1951 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1952 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1953 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1954 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1955 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1956 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1957 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %1942, %5, %1938, %1939, %1956, %14, %1943, %1944, %1945, %1946, %23, %1947, %1948, %1949, %1950, %1951, %1952, %1953, %1954, %1955) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1958 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1959 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1960 = emitc.call_opaque "__runtime_buffer_arg"(%323) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1961 = emitc.cast %1935 : i32 to i64
      %1962 = emitc.call_opaque "__runtime_buffer_offset"(%1960, %1961) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1963 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1964 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1965 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1966 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1967 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1968 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1969 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1970 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1971 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1972 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1973 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1974 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1975 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1976 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1977 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %1962, %6, %1958, %1959, %1976, %1, %1963, %1964, %1965, %1966, %23, %1967, %1968, %1969, %1970, %1971, %1972, %1973, %1974, %1975) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1978 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1979 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1980 = emitc.call_opaque "__runtime_buffer_arg"(%323) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %1981 = emitc.cast %1936 : i32 to i64
      %1982 = emitc.call_opaque "__runtime_buffer_offset"(%1980, %1981) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %1983 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1984 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1985 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %1986 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1987 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %1988 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1989 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1990 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %1991 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %1992 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %1993 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %1994 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %1995 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %1996 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %1997 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %1982, %7, %1978, %1979, %1996, %2, %1983, %1984, %1985, %1986, %23, %1987, %1988, %1989, %1990, %1991, %1992, %1993, %1994, %1995) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %1998 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
      %1999 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %2000 = emitc.call_opaque "__runtime_buffer_arg"(%323) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %2001 = emitc.cast %1933 : i32 to i64
      %2002 = emitc.call_opaque "__runtime_buffer_offset"(%2000, %2001) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %2003 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %2004 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %2005 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
      %2006 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %2007 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
      %2008 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %2009 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %2010 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
      %2011 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
      %2012 = "emitc.constant"() <{value = 7168 : i32}> : () -> i32
      %2013 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
      %2014 = "emitc.constant"() <{value = 1792 : i32}> : () -> i32
      %2015 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
      %2016 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
      %2017 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %97, %2002, %8, %1998, %1999, %2016, %3, %2003, %2004, %2005, %2006, %23, %2007, %2008, %2009, %2010, %2011, %2012, %2013, %2014, %2015) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
      %2018 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
      %2019 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
      %2020 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
      emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(2,0), direction=S2MM */"
      emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=1 dir=S2MM */"
      emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %97, %2018, %2020) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
      %2021 = emitc.call_opaque "__Runtime_dma_createio_4"(%97, %2017, %2018, %2019, %2020) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
      %2022 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
      %2023 = emitc.call_opaque "__Runtime_startio"(%arg0, %2021, %1937, %2022) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
      emitc.verbatim "/* Wait for 1 event(s) */"
      emitc.call_opaque "__Runtime_wait"(%arg0, %1747) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
      emitc.verbatim "/* Wait for 1 event(s) */"
      emitc.call_opaque "__Runtime_wait"(%arg0, %1839) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
      emitc.verbatim "/* Wait for 1 event(s) */"
      emitc.call_opaque "__Runtime_wait"(%arg0, %1931) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
      emitc.verbatim "/* Wait for 1 event(s) */"
      emitc.call_opaque "__Runtime_wait"(%arg0, %2023) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    }
    emitc.verbatim "/* Wait for 1 event(s) */"
    emitc.call_opaque "__Runtime_wait"(%arg0, %347) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"event">) -> ()
    emitc.verbatim "/* AieRt debug snapshot (gated by AIE_DEBUG_LOG flag) */"
    emitc.verbatim "if (AIEHLC_LOG_ENABLED()) {"
    emitc.verbatim "  uint8_t _dbg_io_cols[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 1, 2, 3, 0, 1, 2, 3, 1, 0, 1, 2, 3, 0, 1, 2, 3, 2, 0, 1, 2, 3, 0, 1, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3, 3, 3, 2, 2};"
    emitc.verbatim "  uint8_t _dbg_io_rows[] = {0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0};"
    emitc.verbatim "  uint8_t _dbg_io_chs[] = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1};"
    emitc.verbatim "  uint8_t _dbg_io_bds[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 4, 4, 4, 4, 1, 2, 2, 2, 2, 4, 4, 4, 4, 1, 2, 2, 2, 2, 4, 4, 4, 4, 11, 2, 2, 2, 2, 4, 4, 4, 4, 2, 7, 3, 8};"
    emitc.verbatim "  int _dbg_io_dirs[] = {DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM};"
    emitc.verbatim "  uint8_t _dbg_t_cols[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};"
    emitc.verbatim "  uint8_t _dbg_t_rows[] = {3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6};"
    emitc.verbatim "  AieRt_DebugSnapshotFromCoords(dev,\0A      _dbg_io_cols, _dbg_io_rows, _dbg_io_chs, _dbg_io_bds, _dbg_io_dirs, 60,\0A      _dbg_t_cols, _dbg_t_rows, 16);"
    emitc.verbatim "}"
    emitc.verbatim "/* Kernel klog only (AIE_KERNEL_CONFIG_TRACE without AIE_DEBUG_LOG) */"
    emitc.verbatim "if (AIEHLC_KLOG_ENABLED() && !AIEHLC_LOG_ENABLED()) {"
    emitc.verbatim "  uint8_t _klog_cols[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};"
    emitc.verbatim "  uint8_t _klog_rows[] = {3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6};"
    emitc.verbatim "  __Runtime_read_kernel_logs(dev, _klog_cols, _klog_rows, 16);"
    emitc.verbatim "}"
    emitc.return
  }
  emitc.func @dskernel_receiver(%arg0: index) attributes {specifiers = ["__global__"]} {
    emitc.verbatim "// the real kernel will be emitted separately\0A"
    emitc.return
  }
}
