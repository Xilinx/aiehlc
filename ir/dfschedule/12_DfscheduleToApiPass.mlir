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
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) {
    emitc.call_opaque "host_canonicalized"() : () -> ()
    return
  }
  emitc.func @host_canonicalized(%arg0: !emitc.ptr<!emitc.opaque<"XAie_DevInst">>, %arg1: !emitc.ptr<!emitc.opaque<"void">>, %arg2: !emitc.ptr<!emitc.opaque<"void">>, %arg3: !emitc.ptr<!emitc.opaque<"void">>) {
    emitc.verbatim "XAie_DevInst* dev = v1;"
    %0 = "emitc.constant"() <{value = #emitc.opaque<"XAIE_MEM_CACHEABLE">}> : () -> i32
    %1 = "emitc.constant"() <{value = 13 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 14 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = 15 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %6 = "emitc.constant"() <{value = 12 : i32}> : () -> i32
    %7 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %8 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %9 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %10 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %11 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %12 = "emitc.constant"() <{value = 12544 : i32}> : () -> i32
    %13 = "emitc.constant"() <{value = 25088 : i32}> : () -> i32
    %14 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %15 = "emitc.constant"() <{value = 37632 : i32}> : () -> i32
    %16 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %17 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %18 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %19 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %20 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %21 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %22 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %23 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %22) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %24 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %25 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %26 = emitc.call_opaque "XAie_TileLoc"(%24, %25) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %27 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %28 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %29 = emitc.call_opaque "XAie_TileLoc"(%27, %28) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %30 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %31 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,3) */"
    %32 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %33 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %34 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %35 = emitc.call_opaque "XAie_TileLoc"(%33, %34) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %36 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %37 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,4) */"
    %38 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %39 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %40 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %41 = emitc.call_opaque "XAie_TileLoc"(%39, %40) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %42 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %43 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,5) */"
    %44 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %45 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %46 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %47 = emitc.call_opaque "XAie_TileLoc"(%45, %46) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %48 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %49 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,6) */"
    %50 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,0) */"
    %51 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %52 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %53 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %52) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %54 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %55 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %56 = emitc.call_opaque "XAie_TileLoc"(%54, %55) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %57 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %58 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %59 = emitc.call_opaque "XAie_TileLoc"(%57, %58) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %60 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %61 = emitc.call_opaque "__runtime_buffer_offset"(%53, %60) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %62 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %63 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,3) */"
    %64 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %65 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %66 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %67 = emitc.call_opaque "XAie_TileLoc"(%65, %66) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %68 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %69 = emitc.call_opaque "__runtime_buffer_offset"(%53, %68) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %70 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %71 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,4) */"
    %72 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %73 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %74 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %75 = emitc.call_opaque "XAie_TileLoc"(%73, %74) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %76 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %77 = emitc.call_opaque "__runtime_buffer_offset"(%53, %76) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %78 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %79 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,5) */"
    %80 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %81 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %82 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %83 = emitc.call_opaque "XAie_TileLoc"(%81, %82) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %84 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %85 = emitc.call_opaque "__runtime_buffer_offset"(%53, %84) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %86 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %87 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,6) */"
    %88 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,0) */"
    %89 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %90 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %91 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %90) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %92 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %93 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %94 = emitc.call_opaque "XAie_TileLoc"(%92, %93) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %95 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %96 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %97 = emitc.call_opaque "XAie_TileLoc"(%95, %96) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %98 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %99 = emitc.call_opaque "__runtime_buffer_offset"(%91, %98) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %100 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %101 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,3) */"
    %102 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %103 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %104 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %105 = emitc.call_opaque "XAie_TileLoc"(%103, %104) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %106 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %107 = emitc.call_opaque "__runtime_buffer_offset"(%91, %106) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %108 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %109 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,4) */"
    %110 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %111 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %112 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %113 = emitc.call_opaque "XAie_TileLoc"(%111, %112) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %114 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %115 = emitc.call_opaque "__runtime_buffer_offset"(%91, %114) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %116 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %117 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,5) */"
    %118 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %119 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %120 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %121 = emitc.call_opaque "XAie_TileLoc"(%119, %120) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %122 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %123 = emitc.call_opaque "__runtime_buffer_offset"(%91, %122) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %124 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %125 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,6) */"
    %126 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,0) */"
    %127 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %128 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %129 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %128) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %130 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %131 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %132 = emitc.call_opaque "XAie_TileLoc"(%130, %131) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %133 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %134 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %135 = emitc.call_opaque "XAie_TileLoc"(%133, %134) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %136 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %137 = emitc.call_opaque "__runtime_buffer_offset"(%129, %136) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %138 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %139 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,3) */"
    %140 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %141 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %142 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %143 = emitc.call_opaque "XAie_TileLoc"(%141, %142) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %144 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %145 = emitc.call_opaque "__runtime_buffer_offset"(%129, %144) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %146 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %147 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,4) */"
    %148 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %149 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %150 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %151 = emitc.call_opaque "XAie_TileLoc"(%149, %150) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %152 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %153 = emitc.call_opaque "__runtime_buffer_offset"(%129, %152) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %154 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %155 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,5) */"
    %156 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %157 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %158 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %159 = emitc.call_opaque "XAie_TileLoc"(%157, %158) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %160 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %161 = emitc.call_opaque "__runtime_buffer_offset"(%129, %160) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %162 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %163 = "emitc.constant"() <{value = #emitc.opaque<"(void*)35904">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,6) */"
    %164 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,0) */"
    %165 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %166 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %167 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %166) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %168 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %169 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,3) */"
    %170 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %171 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %172 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,3) */"
    %173 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %174 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %175 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,3) */"
    %176 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %177 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %178 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,3) */"
    %179 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,0) */"
    %180 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %181 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %182 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %181) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %183 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %184 = emitc.call_opaque "__runtime_buffer_offset"(%182, %183) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %185 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %186 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,3) */"
    %187 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %188 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %189 = emitc.call_opaque "__runtime_buffer_offset"(%182, %188) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %190 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %191 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,3) */"
    %192 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %193 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %194 = emitc.call_opaque "__runtime_buffer_offset"(%182, %193) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %195 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %196 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,3) */"
    %197 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %198 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %199 = emitc.call_opaque "__runtime_buffer_offset"(%182, %198) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %200 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %201 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,3) */"
    %202 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,0) */"
    %203 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %204 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %205 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %204) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %206 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %207 = emitc.call_opaque "__runtime_buffer_offset"(%205, %206) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %208 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %209 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,4) */"
    %210 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %211 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %212 = emitc.call_opaque "__runtime_buffer_offset"(%205, %211) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %213 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %214 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,4) */"
    %215 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %216 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %217 = emitc.call_opaque "__runtime_buffer_offset"(%205, %216) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %218 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %219 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,4) */"
    %220 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %221 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %222 = emitc.call_opaque "__runtime_buffer_offset"(%205, %221) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %223 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %224 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,4) */"
    %225 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,0) */"
    %226 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %227 = "emitc.constant"() <{value = 200704 : i64}> : () -> i64
    %228 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %227) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %229 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %230 = emitc.call_opaque "__runtime_buffer_offset"(%228, %229) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %231 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %232 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,4) */"
    %233 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %234 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %235 = emitc.call_opaque "__runtime_buffer_offset"(%228, %234) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %236 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %237 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,4) */"
    %238 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %239 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %240 = emitc.call_opaque "__runtime_buffer_offset"(%228, %239) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %241 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %242 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,4) */"
    %243 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %244 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %245 = emitc.call_opaque "__runtime_buffer_offset"(%228, %244) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %246 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %247 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,4) */"
    %248 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,0) */"
    %249 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %250 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %251 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %250) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %252 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %253 = emitc.call_opaque "__runtime_buffer_offset"(%251, %252) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %254 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %255 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,5) */"
    %256 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %257 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %258 = emitc.call_opaque "__runtime_buffer_offset"(%251, %257) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %259 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %260 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,5) */"
    %261 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %262 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %263 = emitc.call_opaque "__runtime_buffer_offset"(%251, %262) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %264 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %265 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,5) */"
    %266 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %267 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %268 = emitc.call_opaque "__runtime_buffer_offset"(%251, %267) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %269 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %270 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,5) */"
    %271 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,0) */"
    %272 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %273 = "emitc.constant"() <{value = 401408 : i64}> : () -> i64
    %274 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %273) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %275 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %276 = emitc.call_opaque "__runtime_buffer_offset"(%274, %275) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %277 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %278 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,5) */"
    %279 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %280 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %281 = emitc.call_opaque "__runtime_buffer_offset"(%274, %280) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %282 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %283 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,5) */"
    %284 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %285 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %286 = emitc.call_opaque "__runtime_buffer_offset"(%274, %285) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %287 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %288 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,5) */"
    %289 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %290 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %291 = emitc.call_opaque "__runtime_buffer_offset"(%274, %290) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %292 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %293 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,5) */"
    %294 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,0) */"
    %295 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %296 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %297 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %296) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %298 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %299 = emitc.call_opaque "__runtime_buffer_offset"(%297, %298) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %300 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %301 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,6) */"
    %302 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %303 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %304 = emitc.call_opaque "__runtime_buffer_offset"(%297, %303) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %305 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %306 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,6) */"
    %307 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %308 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %309 = emitc.call_opaque "__runtime_buffer_offset"(%297, %308) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %310 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %311 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,6) */"
    %312 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %313 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %314 = emitc.call_opaque "__runtime_buffer_offset"(%297, %313) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %315 = "emitc.constant"() <{value = #emitc.opaque<"(void*)39040">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %316 = "emitc.constant"() <{value = #emitc.opaque<"(void*)43680">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,6) */"
    %317 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 3 for tile (3,0) */"
    %318 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %319 = "emitc.constant"() <{value = 602112 : i64}> : () -> i64
    %320 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %319) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %321 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %322 = emitc.call_opaque "__runtime_buffer_offset"(%320, %321) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %323 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %324 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,6) */"
    %325 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %326 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %327 = emitc.call_opaque "__runtime_buffer_offset"(%320, %326) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %328 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %329 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,6) */"
    %330 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %331 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %332 = emitc.call_opaque "__runtime_buffer_offset"(%320, %331) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %333 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %334 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,6) */"
    %335 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %336 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %337 = emitc.call_opaque "__runtime_buffer_offset"(%320, %336) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %338 = "emitc.constant"() <{value = #emitc.opaque<"(void*)48320">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %339 = "emitc.constant"() <{value = #emitc.opaque<"(void*)51456">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,6) */"
    %340 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 3 for tile (2,0) */"
    %341 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    emitc.verbatim "/* Load Kernel Group: 16 tile(s) */"
    %342 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %343 = emitc.call_opaque "__Runtime_load_kernel_group_16t"(%arg0, %29, %35, %41, %47, %59, %67, %75, %83, %97, %105, %113, %121, %135, %143, %151, %159, %342) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, i32) -> !emitc.opaque<"kernel_group">
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
    %344 = emitc.call_opaque "__Runtime_launch_kernel_group"(%arg0, %343) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"kernel_group">) -> !emitc.opaque<"event">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %345 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %346 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %347 = emitc.call_opaque "__runtime_buffer_arg"(%23) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %348 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %349 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %350 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %351 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %352 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %353 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %26, %347, %19, %345, %346, %352, %19, %348, %349, %350, %351, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %354 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %355 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %356 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(0,0), direction=MM2S */"
    %357 = emitc.call_opaque "__Runtime_dma_createio_4"(%26, %353, %354, %355, %356) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %358 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %359 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %360 = emitc.call_opaque "__runtime_buffer_arg"(%31) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %361 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %362 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %363 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %364 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %365 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %366 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %360, %21, %358, %359, %365, %19, %361, %362, %363, %364, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %367 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %368 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %369 = emitc.call_opaque "__runtime_buffer_arg"(%30) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %370 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %371 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %372 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %373 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %374 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %375 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %369, %19, %367, %368, %374, %19, %370, %371, %372, %373, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %376 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %377 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %378 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,3), direction=S2MM */"
    %379 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %375, %376, %377, %378) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %380 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %381 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %382 = emitc.call_opaque "__runtime_buffer_arg"(%37) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %383 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %384 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %385 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %386 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %387 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %388 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %382, %21, %380, %381, %387, %19, %383, %384, %385, %386, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %389 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %390 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %391 = emitc.call_opaque "__runtime_buffer_arg"(%36) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %392 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %393 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %394 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %395 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %396 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %397 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %391, %19, %389, %390, %396, %19, %392, %393, %394, %395, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %398 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %399 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %400 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,4), direction=S2MM */"
    %401 = emitc.call_opaque "__Runtime_dma_createio_4"(%35, %397, %398, %399, %400) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %402 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %403 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %404 = emitc.call_opaque "__runtime_buffer_arg"(%43) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %405 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %406 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %407 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %408 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %409 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %410 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %404, %21, %402, %403, %409, %19, %405, %406, %407, %408, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %411 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %412 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %413 = emitc.call_opaque "__runtime_buffer_arg"(%42) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %414 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %415 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %416 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %417 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %418 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %419 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %413, %19, %411, %412, %418, %19, %414, %415, %416, %417, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %420 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %421 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %422 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,5), direction=S2MM */"
    %423 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %419, %420, %421, %422) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %424 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %425 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %426 = emitc.call_opaque "__runtime_buffer_arg"(%49) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %427 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %428 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %429 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %430 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %431 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %432 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %47, %426, %21, %424, %425, %431, %19, %427, %428, %429, %430, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %433 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %434 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %435 = emitc.call_opaque "__runtime_buffer_arg"(%48) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %436 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %437 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %438 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %439 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %440 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %441 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %47, %435, %19, %433, %434, %440, %19, %436, %437, %438, %439, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %442 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %443 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %444 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,6), direction=S2MM */"
    %445 = emitc.call_opaque "__Runtime_dma_createio_4"(%47, %441, %442, %443, %444) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %446 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %447 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %448 = emitc.call_opaque "__runtime_buffer_arg"(%53) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %449 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %450 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %451 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %452 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %453 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %454 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %56, %448, %19, %446, %447, %453, %19, %449, %450, %451, %452, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %455 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %456 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %457 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(1,0), direction=MM2S */"
    %458 = emitc.call_opaque "__Runtime_dma_createio_4"(%56, %454, %455, %456, %457) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %459 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %460 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %461 = emitc.call_opaque "__runtime_buffer_arg"(%63) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %462 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %463 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %464 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %465 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %466 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %467 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %59, %461, %21, %459, %460, %466, %19, %462, %463, %464, %465, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %468 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %469 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %470 = emitc.call_opaque "__runtime_buffer_arg"(%62) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %471 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %472 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %473 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %474 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %475 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %476 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %59, %470, %19, %468, %469, %475, %19, %471, %472, %473, %474, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %477 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %478 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %479 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,3), direction=S2MM */"
    %480 = emitc.call_opaque "__Runtime_dma_createio_4"(%59, %476, %477, %478, %479) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %481 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %482 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %483 = emitc.call_opaque "__runtime_buffer_arg"(%71) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %484 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %485 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %486 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %487 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %488 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %489 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %67, %483, %21, %481, %482, %488, %19, %484, %485, %486, %487, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %490 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %491 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %492 = emitc.call_opaque "__runtime_buffer_arg"(%70) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %493 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %494 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %495 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %496 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %497 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %498 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %67, %492, %19, %490, %491, %497, %19, %493, %494, %495, %496, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %499 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %500 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %501 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,4), direction=S2MM */"
    %502 = emitc.call_opaque "__Runtime_dma_createio_4"(%67, %498, %499, %500, %501) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %503 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %504 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %505 = emitc.call_opaque "__runtime_buffer_arg"(%79) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %506 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %507 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %508 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %509 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %510 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %511 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %75, %505, %21, %503, %504, %510, %19, %506, %507, %508, %509, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %512 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %513 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %514 = emitc.call_opaque "__runtime_buffer_arg"(%78) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %515 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %516 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %517 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %518 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %519 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %520 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %75, %514, %19, %512, %513, %519, %19, %515, %516, %517, %518, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %521 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %522 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %523 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,5), direction=S2MM */"
    %524 = emitc.call_opaque "__Runtime_dma_createio_4"(%75, %520, %521, %522, %523) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %525 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %526 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %527 = emitc.call_opaque "__runtime_buffer_arg"(%87) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %528 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %529 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %530 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %531 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %532 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %533 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %83, %527, %21, %525, %526, %532, %19, %528, %529, %530, %531, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %534 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %535 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %536 = emitc.call_opaque "__runtime_buffer_arg"(%86) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %537 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %538 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %539 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %540 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %541 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %542 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %83, %536, %19, %534, %535, %541, %19, %537, %538, %539, %540, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %543 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %544 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %545 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,6), direction=S2MM */"
    %546 = emitc.call_opaque "__Runtime_dma_createio_4"(%83, %542, %543, %544, %545) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %547 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %548 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %549 = emitc.call_opaque "__runtime_buffer_arg"(%91) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %550 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %551 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %552 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %553 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %554 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %555 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %549, %19, %547, %548, %554, %19, %550, %551, %552, %553, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %556 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %557 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %558 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */"
    %559 = emitc.call_opaque "__Runtime_dma_createio_4"(%94, %555, %556, %557, %558) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %560 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %561 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %562 = emitc.call_opaque "__runtime_buffer_arg"(%101) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %563 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %564 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %565 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %566 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %567 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %568 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %97, %562, %21, %560, %561, %567, %19, %563, %564, %565, %566, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %569 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %570 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %571 = emitc.call_opaque "__runtime_buffer_arg"(%100) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %572 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %573 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %574 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %575 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %576 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %577 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %97, %571, %19, %569, %570, %576, %19, %572, %573, %574, %575, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %578 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %579 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %580 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,3), direction=S2MM */"
    %581 = emitc.call_opaque "__Runtime_dma_createio_4"(%97, %577, %578, %579, %580) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %582 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %583 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %584 = emitc.call_opaque "__runtime_buffer_arg"(%109) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %585 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %586 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %587 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %588 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %589 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %590 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %105, %584, %21, %582, %583, %589, %19, %585, %586, %587, %588, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %591 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %592 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %593 = emitc.call_opaque "__runtime_buffer_arg"(%108) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %594 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %595 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %596 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %597 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %598 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %599 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %105, %593, %19, %591, %592, %598, %19, %594, %595, %596, %597, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %600 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %601 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %602 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,4), direction=S2MM */"
    %603 = emitc.call_opaque "__Runtime_dma_createio_4"(%105, %599, %600, %601, %602) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %604 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %605 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %606 = emitc.call_opaque "__runtime_buffer_arg"(%117) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %607 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %608 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %609 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %610 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %611 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %612 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %113, %606, %21, %604, %605, %611, %19, %607, %608, %609, %610, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %613 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %614 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %615 = emitc.call_opaque "__runtime_buffer_arg"(%116) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %616 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %617 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %618 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %619 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %620 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %621 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %113, %615, %19, %613, %614, %620, %19, %616, %617, %618, %619, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %622 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %623 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %624 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,5), direction=S2MM */"
    %625 = emitc.call_opaque "__Runtime_dma_createio_4"(%113, %621, %622, %623, %624) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %626 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %627 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %628 = emitc.call_opaque "__runtime_buffer_arg"(%125) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %629 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %630 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %631 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %632 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %633 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %634 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %121, %628, %21, %626, %627, %633, %19, %629, %630, %631, %632, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %635 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %636 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %637 = emitc.call_opaque "__runtime_buffer_arg"(%124) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %638 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %639 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %640 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %641 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %642 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %643 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %121, %637, %19, %635, %636, %642, %19, %638, %639, %640, %641, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %644 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %645 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %646 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,6), direction=S2MM */"
    %647 = emitc.call_opaque "__Runtime_dma_createio_4"(%121, %643, %644, %645, %646) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %648 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %649 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %650 = emitc.call_opaque "__runtime_buffer_arg"(%129) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %651 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %652 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %653 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %654 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %655 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %656 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %650, %19, %648, %649, %655, %19, %651, %652, %653, %654, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %657 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %658 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %659 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */"
    %660 = emitc.call_opaque "__Runtime_dma_createio_4"(%132, %656, %657, %658, %659) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %661 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %662 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %663 = emitc.call_opaque "__runtime_buffer_arg"(%139) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %664 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %665 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %666 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %667 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %668 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %669 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %135, %663, %21, %661, %662, %668, %19, %664, %665, %666, %667, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %670 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %671 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %672 = emitc.call_opaque "__runtime_buffer_arg"(%138) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %673 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %674 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %675 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %676 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %677 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %678 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %135, %672, %19, %670, %671, %677, %19, %673, %674, %675, %676, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %679 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %680 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %681 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,3), direction=S2MM */"
    %682 = emitc.call_opaque "__Runtime_dma_createio_4"(%135, %678, %679, %680, %681) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %683 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %684 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %685 = emitc.call_opaque "__runtime_buffer_arg"(%147) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %686 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %687 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %688 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %689 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %690 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %691 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %143, %685, %21, %683, %684, %690, %19, %686, %687, %688, %689, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %692 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %693 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %694 = emitc.call_opaque "__runtime_buffer_arg"(%146) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %695 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %696 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %697 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %698 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %699 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %700 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %143, %694, %19, %692, %693, %699, %19, %695, %696, %697, %698, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %701 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %702 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %703 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,4), direction=S2MM */"
    %704 = emitc.call_opaque "__Runtime_dma_createio_4"(%143, %700, %701, %702, %703) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %705 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %706 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %707 = emitc.call_opaque "__runtime_buffer_arg"(%155) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %708 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %709 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %710 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %711 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %712 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %713 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %151, %707, %21, %705, %706, %712, %19, %708, %709, %710, %711, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %714 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %715 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %716 = emitc.call_opaque "__runtime_buffer_arg"(%154) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %717 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %718 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %719 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %720 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %721 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %722 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %151, %716, %19, %714, %715, %721, %19, %717, %718, %719, %720, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %723 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %724 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %725 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,5), direction=S2MM */"
    %726 = emitc.call_opaque "__Runtime_dma_createio_4"(%151, %722, %723, %724, %725) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %727 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %728 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %729 = emitc.call_opaque "__runtime_buffer_arg"(%163) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %730 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %731 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %732 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %733 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %734 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %735 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %159, %729, %21, %727, %728, %734, %19, %730, %731, %732, %733, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %736 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %737 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %738 = emitc.call_opaque "__runtime_buffer_arg"(%162) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %739 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %740 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %741 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %742 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %743 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %744 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %159, %738, %19, %736, %737, %743, %19, %739, %740, %741, %742, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %745 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %746 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %747 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,6), direction=S2MM */"
    %748 = emitc.call_opaque "__Runtime_dma_createio_4"(%159, %744, %745, %746, %747) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %749 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %750 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %751 = emitc.call_opaque "__runtime_buffer_arg"(%167) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %752 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %753 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %754 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %755 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %756 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %757 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %758 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %759 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %760 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %761 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %762 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %763 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %764 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %765 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %766 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %26, %751, %21, %749, %750, %765, %19, %752, %753, %754, %755, %20, %756, %757, %758, %759, %760, %761, %762, %763, %764) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %767 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %768 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %769 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(0,0), direction=MM2S */"
    %770 = emitc.call_opaque "__Runtime_dma_createio_4"(%26, %766, %767, %768, %769) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %771 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %772 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %773 = emitc.call_opaque "__runtime_buffer_arg"(%169) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %774 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %775 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %776 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %777 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %778 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %779 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %773, %17, %771, %772, %778, %19, %774, %775, %776, %777, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %780 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %781 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %782 = emitc.call_opaque "__runtime_buffer_arg"(%168) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %783 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %784 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %785 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %786 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %787 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %788 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %782, %18, %780, %781, %787, %19, %783, %784, %785, %786, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %789 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %790 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %791 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,3), direction=S2MM */"
    %792 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %788, %789, %790, %791) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %793 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %794 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %795 = emitc.call_opaque "__runtime_buffer_arg"(%172) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %796 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %797 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %798 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %799 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %800 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %801 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %59, %795, %17, %793, %794, %800, %19, %796, %797, %798, %799, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %802 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %803 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %804 = emitc.call_opaque "__runtime_buffer_arg"(%171) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %805 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %806 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %807 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %808 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %809 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %810 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %59, %804, %18, %802, %803, %809, %19, %805, %806, %807, %808, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %811 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %812 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %813 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,3), direction=S2MM */"
    %814 = emitc.call_opaque "__Runtime_dma_createio_4"(%59, %810, %811, %812, %813) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %815 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %816 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %817 = emitc.call_opaque "__runtime_buffer_arg"(%175) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %818 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %819 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %820 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %821 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %822 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %823 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %97, %817, %17, %815, %816, %822, %19, %818, %819, %820, %821, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %824 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %825 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %826 = emitc.call_opaque "__runtime_buffer_arg"(%174) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %827 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %828 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %829 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %830 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %831 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %832 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %97, %826, %18, %824, %825, %831, %19, %827, %828, %829, %830, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %833 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %834 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %835 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,3), direction=S2MM */"
    %836 = emitc.call_opaque "__Runtime_dma_createio_4"(%97, %832, %833, %834, %835) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %837 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %838 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %839 = emitc.call_opaque "__runtime_buffer_arg"(%178) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %840 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %841 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %842 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %843 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %844 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %845 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %135, %839, %17, %837, %838, %844, %19, %840, %841, %842, %843, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %846 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %847 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %848 = emitc.call_opaque "__runtime_buffer_arg"(%177) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %849 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %850 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %851 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %852 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %853 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %854 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %135, %848, %18, %846, %847, %853, %19, %849, %850, %851, %852, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %855 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %856 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %857 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,3), direction=S2MM */"
    %858 = emitc.call_opaque "__Runtime_dma_createio_4"(%135, %854, %855, %856, %857) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %859 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %860 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %861 = emitc.call_opaque "__runtime_buffer_arg"(%182) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %862 = emitc.cast %15 : i32 to i64
    %863 = emitc.call_opaque "__runtime_buffer_offset"(%861, %862) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %864 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %865 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %866 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %867 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %868 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %869 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %863, %16, %859, %860, %868, %14, %864, %865, %866, %867, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=3, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %870 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %871 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %872 = emitc.call_opaque "__runtime_buffer_arg"(%182) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %873 = emitc.cast %13 : i32 to i64
    %874 = emitc.call_opaque "__runtime_buffer_offset"(%872, %873) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %875 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %876 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %877 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %878 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %879 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %880 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %874, %14, %870, %871, %879, %17, %875, %876, %877, %878, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=2, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %881 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %882 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %883 = emitc.call_opaque "__runtime_buffer_arg"(%182) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %884 = emitc.cast %12 : i32 to i64
    %885 = emitc.call_opaque "__runtime_buffer_offset"(%883, %884) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %886 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %887 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %888 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %889 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %890 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %891 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %885, %17, %881, %882, %890, %18, %886, %887, %888, %889, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=3136, enable_packet=false, packet_id=1, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %892 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %893 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %894 = emitc.call_opaque "__runtime_buffer_arg"(%182) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %895 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %896 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %897 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %898 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %899 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %900 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %894, %18, %892, %893, %899, %21, %895, %896, %897, %898, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %901 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %902 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %903 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %132, %901, %903) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %904 = emitc.call_opaque "__Runtime_dma_createio_4"(%132, %900, %901, %902, %903) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=1, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */"
    %905 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %906 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %907 = emitc.call_opaque "__runtime_buffer_arg"(%186) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %908 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %909 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %910 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %911 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %912 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %913 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %907, %16, %905, %906, %912, %21, %908, %909, %910, %911, %18) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=1, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */"
    %914 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %915 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %916 = emitc.call_opaque "__runtime_buffer_arg"(%185) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %917 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %918 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %919 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %920 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %921 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %922 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %916, %14, %914, %915, %921, %21, %917, %918, %919, %920, %18) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %923 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %924 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %925 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,3), direction=MM2S */"
    %926 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %922, %923, %924, %925) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=2, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %927 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %928 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %929 = emitc.call_opaque "__runtime_buffer_arg"(%191) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %930 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %931 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %932 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %933 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %934 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %935 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %59, %929, %16, %927, %928, %934, %18, %930, %931, %932, %933, %17) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=2, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %936 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %937 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %938 = emitc.call_opaque "__runtime_buffer_arg"(%190) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %939 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %940 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %941 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %942 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %943 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %944 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %59, %938, %14, %936, %937, %943, %18, %939, %940, %941, %942, %17) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %945 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %946 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %947 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,3), direction=MM2S */"
    %948 = emitc.call_opaque "__Runtime_dma_createio_4"(%59, %944, %945, %946, %947) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=3, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %949 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %950 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %951 = emitc.call_opaque "__runtime_buffer_arg"(%196) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %952 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %953 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %954 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %955 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %956 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %957 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %97, %951, %16, %949, %950, %956, %17, %952, %953, %954, %955, %14) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=3, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %958 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %959 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %960 = emitc.call_opaque "__runtime_buffer_arg"(%195) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %961 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %962 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %963 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %964 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %965 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %966 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %97, %960, %14, %958, %959, %965, %17, %961, %962, %963, %964, %14) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %967 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %968 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %969 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,3), direction=MM2S */"
    %970 = emitc.call_opaque "__Runtime_dma_createio_4"(%97, %966, %967, %968, %969) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=4, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %971 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %972 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %973 = emitc.call_opaque "__runtime_buffer_arg"(%201) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %974 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %975 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %976 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %977 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %978 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %979 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %135, %973, %16, %971, %972, %978, %14, %974, %975, %976, %977, %16) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=4, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %980 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %981 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %982 = emitc.call_opaque "__runtime_buffer_arg"(%200) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %983 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %984 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %985 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %986 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %987 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %988 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %135, %982, %14, %980, %981, %987, %14, %983, %984, %985, %986, %16) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %989 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %990 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %991 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,3), direction=MM2S */"
    %992 = emitc.call_opaque "__Runtime_dma_createio_4"(%135, %988, %989, %990, %991) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %993 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %994 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %995 = emitc.call_opaque "__runtime_buffer_arg"(%205) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %996 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %997 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %998 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %999 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1000 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1001 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1002 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %1003 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %1004 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %1005 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %1006 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1007 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %1008 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1009 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1010 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %56, %995, %21, %993, %994, %1009, %19, %996, %997, %998, %999, %20, %1000, %1001, %1002, %1003, %1004, %1005, %1006, %1007, %1008) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1011 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1012 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1013 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(1,0), direction=MM2S */"
    %1014 = emitc.call_opaque "__Runtime_dma_createio_4"(%56, %1010, %1011, %1012, %1013) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1015 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1016 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1017 = emitc.call_opaque "__runtime_buffer_arg"(%209) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1018 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1019 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1020 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1021 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1022 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1023 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %1017, %17, %1015, %1016, %1022, %19, %1018, %1019, %1020, %1021, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1024 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1025 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1026 = emitc.call_opaque "__runtime_buffer_arg"(%208) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1027 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1028 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1029 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1030 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1031 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1032 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %1026, %18, %1024, %1025, %1031, %19, %1027, %1028, %1029, %1030, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1033 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1034 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1035 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,4), direction=S2MM */"
    %1036 = emitc.call_opaque "__Runtime_dma_createio_4"(%35, %1032, %1033, %1034, %1035) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1037 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1038 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1039 = emitc.call_opaque "__runtime_buffer_arg"(%214) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1040 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1041 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1042 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1043 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1044 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1045 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %67, %1039, %17, %1037, %1038, %1044, %19, %1040, %1041, %1042, %1043, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1046 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1047 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1048 = emitc.call_opaque "__runtime_buffer_arg"(%213) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1049 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1050 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1051 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1052 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1053 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1054 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %67, %1048, %18, %1046, %1047, %1053, %19, %1049, %1050, %1051, %1052, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1055 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1056 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1057 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,4), direction=S2MM */"
    %1058 = emitc.call_opaque "__Runtime_dma_createio_4"(%67, %1054, %1055, %1056, %1057) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1059 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1060 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1061 = emitc.call_opaque "__runtime_buffer_arg"(%219) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1062 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1063 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1064 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1065 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1066 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1067 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %105, %1061, %17, %1059, %1060, %1066, %19, %1062, %1063, %1064, %1065, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1068 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1069 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1070 = emitc.call_opaque "__runtime_buffer_arg"(%218) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1071 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1072 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1073 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1074 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1075 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1076 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %105, %1070, %18, %1068, %1069, %1075, %19, %1071, %1072, %1073, %1074, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1077 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1078 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1079 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,4), direction=S2MM */"
    %1080 = emitc.call_opaque "__Runtime_dma_createio_4"(%105, %1076, %1077, %1078, %1079) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1081 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1082 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1083 = emitc.call_opaque "__runtime_buffer_arg"(%224) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1084 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1085 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1086 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1087 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1088 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1089 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %143, %1083, %17, %1081, %1082, %1088, %19, %1084, %1085, %1086, %1087, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1090 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1091 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1092 = emitc.call_opaque "__runtime_buffer_arg"(%223) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1093 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1094 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1095 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1096 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1097 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1098 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %143, %1092, %18, %1090, %1091, %1097, %19, %1093, %1094, %1095, %1096, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1099 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1100 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1101 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,4), direction=S2MM */"
    %1102 = emitc.call_opaque "__Runtime_dma_createio_4"(%143, %1098, %1099, %1100, %1101) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1103 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1104 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1105 = emitc.call_opaque "__runtime_buffer_arg"(%228) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1106 = emitc.cast %15 : i32 to i64
    %1107 = emitc.call_opaque "__runtime_buffer_offset"(%1105, %1106) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1108 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1109 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1110 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1111 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1112 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1113 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %1107, %11, %1103, %1104, %1112, %10, %1108, %1109, %1110, %1111, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1114 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1115 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1116 = emitc.call_opaque "__runtime_buffer_arg"(%228) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1117 = emitc.cast %13 : i32 to i64
    %1118 = emitc.call_opaque "__runtime_buffer_offset"(%1116, %1117) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1119 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1120 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1121 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1122 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1123 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1124 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %1118, %9, %1114, %1115, %1123, %8, %1119, %1120, %1121, %1122, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1125 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1126 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1127 = emitc.call_opaque "__runtime_buffer_arg"(%228) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1128 = emitc.cast %12 : i32 to i64
    %1129 = emitc.call_opaque "__runtime_buffer_offset"(%1127, %1128) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1130 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1131 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1132 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1133 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1134 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1135 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %1129, %10, %1125, %1126, %1134, %7, %1130, %1131, %1132, %1133, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=7, len=3136, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1136 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1137 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1138 = emitc.call_opaque "__runtime_buffer_arg"(%228) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1139 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1140 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1141 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1142 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1143 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1144 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %132, %1138, %8, %1136, %1137, %1143, %16, %1139, %1140, %1141, %1142, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1145 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1146 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1147 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=7, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %132, %1145, %1147) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %1148 = emitc.call_opaque "__Runtime_dma_createio_4"(%132, %1144, %1145, %1146, %1147) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=5, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */"
    %1149 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1150 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1151 = emitc.call_opaque "__runtime_buffer_arg"(%232) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1152 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1153 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1154 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1155 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1156 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1157 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %1151, %16, %1149, %1150, %1156, %16, %1152, %1153, %1154, %1155, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=5, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */"
    %1158 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1159 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1160 = emitc.call_opaque "__runtime_buffer_arg"(%231) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1161 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1162 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1163 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1164 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1165 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1166 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %1160, %14, %1158, %1159, %1165, %16, %1161, %1162, %1163, %1164, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1167 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1168 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1169 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,4), direction=MM2S */"
    %1170 = emitc.call_opaque "__Runtime_dma_createio_4"(%35, %1166, %1167, %1168, %1169) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=6, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1171 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1172 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1173 = emitc.call_opaque "__runtime_buffer_arg"(%237) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1174 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1175 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1176 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1177 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1178 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1179 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %67, %1173, %16, %1171, %1172, %1178, %7, %1174, %1175, %1176, %1177, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=6, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1180 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1181 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1182 = emitc.call_opaque "__runtime_buffer_arg"(%236) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1183 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1184 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1185 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1186 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1187 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1188 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %67, %1182, %14, %1180, %1181, %1187, %7, %1183, %1184, %1185, %1186, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1189 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1190 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1191 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,4), direction=MM2S */"
    %1192 = emitc.call_opaque "__Runtime_dma_createio_4"(%67, %1188, %1189, %1190, %1191) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=7, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1193 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1194 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1195 = emitc.call_opaque "__runtime_buffer_arg"(%242) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1196 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1197 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1198 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1199 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1200 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1201 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %105, %1195, %16, %1193, %1194, %1200, %8, %1196, %1197, %1198, %1199, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=7, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1202 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1203 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1204 = emitc.call_opaque "__runtime_buffer_arg"(%241) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1205 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1206 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1207 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1208 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1209 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1210 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %105, %1204, %14, %1202, %1203, %1209, %8, %1205, %1206, %1207, %1208, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1211 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1212 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1213 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,4), direction=MM2S */"
    %1214 = emitc.call_opaque "__Runtime_dma_createio_4"(%105, %1210, %1211, %1212, %1213) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=8, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1215 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1216 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1217 = emitc.call_opaque "__runtime_buffer_arg"(%247) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1218 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1219 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1220 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1221 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1222 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1223 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %143, %1217, %16, %1215, %1216, %1222, %10, %1218, %1219, %1220, %1221, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=8, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1224 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1225 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1226 = emitc.call_opaque "__runtime_buffer_arg"(%246) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1227 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1228 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1229 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1230 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1231 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1232 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %143, %1226, %14, %1224, %1225, %1231, %10, %1227, %1228, %1229, %1230, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1233 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1234 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1235 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,4), direction=MM2S */"
    %1236 = emitc.call_opaque "__Runtime_dma_createio_4"(%143, %1232, %1233, %1234, %1235) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %1237 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %1238 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1239 = emitc.call_opaque "__runtime_buffer_arg"(%251) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1240 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1241 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1242 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1243 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1244 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1245 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1246 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %1247 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %1248 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %1249 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %1250 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1251 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %1252 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1253 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1254 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %94, %1239, %21, %1237, %1238, %1253, %19, %1240, %1241, %1242, %1243, %20, %1244, %1245, %1246, %1247, %1248, %1249, %1250, %1251, %1252) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1255 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1256 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1257 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */"
    %1258 = emitc.call_opaque "__Runtime_dma_createio_4"(%94, %1254, %1255, %1256, %1257) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1259 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1260 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1261 = emitc.call_opaque "__runtime_buffer_arg"(%255) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1262 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1263 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1264 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1265 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1266 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1267 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %1261, %17, %1259, %1260, %1266, %19, %1262, %1263, %1264, %1265, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1268 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1269 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1270 = emitc.call_opaque "__runtime_buffer_arg"(%254) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1271 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1272 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1273 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1274 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1275 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1276 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %1270, %18, %1268, %1269, %1275, %19, %1271, %1272, %1273, %1274, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1277 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1278 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1279 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,5), direction=S2MM */"
    %1280 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %1276, %1277, %1278, %1279) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1281 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1282 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1283 = emitc.call_opaque "__runtime_buffer_arg"(%260) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1284 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1285 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1286 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1287 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1288 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1289 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %75, %1283, %17, %1281, %1282, %1288, %19, %1284, %1285, %1286, %1287, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1290 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1291 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1292 = emitc.call_opaque "__runtime_buffer_arg"(%259) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1293 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1294 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1295 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1296 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1297 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1298 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %75, %1292, %18, %1290, %1291, %1297, %19, %1293, %1294, %1295, %1296, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1299 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1300 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1301 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,5), direction=S2MM */"
    %1302 = emitc.call_opaque "__Runtime_dma_createio_4"(%75, %1298, %1299, %1300, %1301) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1303 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1304 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1305 = emitc.call_opaque "__runtime_buffer_arg"(%265) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1306 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1307 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1308 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1309 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1310 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1311 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %113, %1305, %17, %1303, %1304, %1310, %19, %1306, %1307, %1308, %1309, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1312 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1313 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1314 = emitc.call_opaque "__runtime_buffer_arg"(%264) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1315 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1316 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1317 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1318 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1319 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1320 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %113, %1314, %18, %1312, %1313, %1319, %19, %1315, %1316, %1317, %1318, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1321 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1322 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1323 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,5), direction=S2MM */"
    %1324 = emitc.call_opaque "__Runtime_dma_createio_4"(%113, %1320, %1321, %1322, %1323) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1325 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1326 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1327 = emitc.call_opaque "__runtime_buffer_arg"(%270) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1328 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1329 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1330 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1331 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1332 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1333 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %151, %1327, %17, %1325, %1326, %1332, %19, %1328, %1329, %1330, %1331, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1334 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1335 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1336 = emitc.call_opaque "__runtime_buffer_arg"(%269) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1337 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1338 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1339 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1340 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1341 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1342 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %151, %1336, %18, %1334, %1335, %1341, %19, %1337, %1338, %1339, %1340, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1343 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1344 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1345 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,5), direction=S2MM */"
    %1346 = emitc.call_opaque "__Runtime_dma_createio_4"(%151, %1342, %1343, %1344, %1345) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=6, len=3136, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1347 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1348 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1349 = emitc.call_opaque "__runtime_buffer_arg"(%274) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1350 = emitc.cast %15 : i32 to i64
    %1351 = emitc.call_opaque "__runtime_buffer_offset"(%1349, %1350) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1352 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1353 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1354 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1355 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1356 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1357 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %1351, %7, %1347, %1348, %1356, %6, %1352, %1353, %1354, %1355, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1358 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1359 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1360 = emitc.call_opaque "__runtime_buffer_arg"(%274) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1361 = emitc.cast %13 : i32 to i64
    %1362 = emitc.call_opaque "__runtime_buffer_offset"(%1360, %1361) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1363 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1364 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1365 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1366 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1367 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1368 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %1362, %16, %1358, %1359, %1367, %5, %1363, %1364, %1365, %1366, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1369 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1370 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1371 = emitc.call_opaque "__runtime_buffer_arg"(%274) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1372 = emitc.cast %12 : i32 to i64
    %1373 = emitc.call_opaque "__runtime_buffer_offset"(%1371, %1372) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1374 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1375 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1376 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1377 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1378 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1379 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %1373, %14, %1369, %1370, %1378, %11, %1374, %1375, %1376, %1377, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1380 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1381 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1382 = emitc.call_opaque "__runtime_buffer_arg"(%274) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1383 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1384 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1385 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1386 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1387 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1388 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %1382, %17, %1380, %1381, %1387, %9, %1383, %1384, %1385, %1386, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1389 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1390 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1391 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(2,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %94, %1389, %1391) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %1392 = emitc.call_opaque "__Runtime_dma_createio_4"(%94, %1388, %1389, %1390, %1391) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=9, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %1393 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1394 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1395 = emitc.call_opaque "__runtime_buffer_arg"(%278) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1396 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1397 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1398 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1399 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1400 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1401 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %1395, %16, %1393, %1394, %1400, %9, %1396, %1397, %1398, %1399, %17) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=9, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %1402 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1403 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1404 = emitc.call_opaque "__runtime_buffer_arg"(%277) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1405 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1406 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1407 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1408 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1409 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1410 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %1404, %14, %1402, %1403, %1409, %9, %1405, %1406, %1407, %1408, %17) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1411 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1412 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1413 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,5), direction=MM2S */"
    %1414 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %1410, %1411, %1412, %1413) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=10, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %1415 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1416 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1417 = emitc.call_opaque "__runtime_buffer_arg"(%283) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1418 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1419 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1420 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1421 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1422 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1423 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %75, %1417, %16, %1415, %1416, %1422, %11, %1418, %1419, %1420, %1421, %14) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=10, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %1424 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1425 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1426 = emitc.call_opaque "__runtime_buffer_arg"(%282) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1427 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1428 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1429 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1430 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1431 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1432 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %75, %1426, %14, %1424, %1425, %1431, %11, %1427, %1428, %1429, %1430, %14) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1433 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1434 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1435 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,5), direction=MM2S */"
    %1436 = emitc.call_opaque "__Runtime_dma_createio_4"(%75, %1432, %1433, %1434, %1435) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=11, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %1437 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1438 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1439 = emitc.call_opaque "__runtime_buffer_arg"(%288) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1440 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1441 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1442 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1443 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1444 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1445 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %113, %1439, %16, %1437, %1438, %1444, %5, %1440, %1441, %1442, %1443, %16) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=11, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %1446 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1447 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1448 = emitc.call_opaque "__runtime_buffer_arg"(%287) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1449 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1450 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1451 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1452 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1453 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1454 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %113, %1448, %14, %1446, %1447, %1453, %5, %1449, %1450, %1451, %1452, %16) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1455 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1456 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1457 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,5), direction=MM2S */"
    %1458 = emitc.call_opaque "__Runtime_dma_createio_4"(%113, %1454, %1455, %1456, %1457) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=12, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %1459 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1460 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1461 = emitc.call_opaque "__runtime_buffer_arg"(%293) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1462 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1463 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1464 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1465 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1466 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1467 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %151, %1461, %16, %1459, %1460, %1466, %6, %1462, %1463, %1464, %1465, %7) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=12, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %1468 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1469 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1470 = emitc.call_opaque "__runtime_buffer_arg"(%292) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1471 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1472 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1473 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1474 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1475 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1476 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %151, %1470, %14, %1468, %1469, %1475, %6, %1471, %1472, %1473, %1474, %7) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1477 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1478 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1479 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,5), direction=MM2S */"
    %1480 = emitc.call_opaque "__Runtime_dma_createio_4"(%151, %1476, %1477, %1478, %1479) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %1481 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %1482 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1483 = emitc.call_opaque "__runtime_buffer_arg"(%297) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1484 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1485 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1486 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1487 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1488 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1489 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1490 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %1491 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %1492 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %1493 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %1494 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1495 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %1496 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1497 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1498 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %132, %1483, %5, %1481, %1482, %1497, %19, %1484, %1485, %1486, %1487, %20, %1488, %1489, %1490, %1491, %1492, %1493, %1494, %1495, %1496) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1499 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1500 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %1501 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=11, tile=(3,0), direction=MM2S */"
    %1502 = emitc.call_opaque "__Runtime_dma_createio_4"(%132, %1498, %1499, %1500, %1501) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1503 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1504 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1505 = emitc.call_opaque "__runtime_buffer_arg"(%301) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1506 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1507 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1508 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1509 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1510 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1511 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %47, %1505, %17, %1503, %1504, %1510, %19, %1506, %1507, %1508, %1509, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1512 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1513 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1514 = emitc.call_opaque "__runtime_buffer_arg"(%300) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1515 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1516 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1517 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1518 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1519 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1520 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %47, %1514, %18, %1512, %1513, %1519, %19, %1515, %1516, %1517, %1518, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1521 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1522 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1523 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,6), direction=S2MM */"
    %1524 = emitc.call_opaque "__Runtime_dma_createio_4"(%47, %1520, %1521, %1522, %1523) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1525 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1526 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1527 = emitc.call_opaque "__runtime_buffer_arg"(%306) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1528 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1529 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1530 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1531 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1532 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1533 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %83, %1527, %17, %1525, %1526, %1532, %19, %1528, %1529, %1530, %1531, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1534 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1535 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1536 = emitc.call_opaque "__runtime_buffer_arg"(%305) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1537 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1538 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1539 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1540 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1541 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1542 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %83, %1536, %18, %1534, %1535, %1541, %19, %1537, %1538, %1539, %1540, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1543 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1544 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1545 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,6), direction=S2MM */"
    %1546 = emitc.call_opaque "__Runtime_dma_createio_4"(%83, %1542, %1543, %1544, %1545) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1547 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1548 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1549 = emitc.call_opaque "__runtime_buffer_arg"(%311) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1550 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1551 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1552 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1553 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1554 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1555 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %121, %1549, %17, %1547, %1548, %1554, %19, %1550, %1551, %1552, %1553, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1556 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1557 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1558 = emitc.call_opaque "__runtime_buffer_arg"(%310) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1559 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1560 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1561 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1562 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1563 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1564 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %121, %1558, %18, %1556, %1557, %1563, %19, %1559, %1560, %1561, %1562, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1565 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1566 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1567 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,6), direction=S2MM */"
    %1568 = emitc.call_opaque "__Runtime_dma_createio_4"(%121, %1564, %1565, %1566, %1567) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1569 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1570 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1571 = emitc.call_opaque "__runtime_buffer_arg"(%316) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1572 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1573 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1574 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1575 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1576 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1577 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %159, %1571, %17, %1569, %1570, %1576, %19, %1572, %1573, %1574, %1575, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1578 = "emitc.constant"() <{value = 4636 : i32}> : () -> i32
    %1579 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1580 = emitc.call_opaque "__runtime_buffer_arg"(%315) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1581 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1582 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1583 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1584 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1585 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1586 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %159, %1580, %18, %1578, %1579, %1585, %19, %1581, %1582, %1583, %1584, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1587 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1588 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1589 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,6), direction=S2MM */"
    %1590 = emitc.call_opaque "__Runtime_dma_createio_4"(%159, %1586, %1587, %1588, %1589) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=3136, enable_packet=false, packet_id=16, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1591 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1592 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1593 = emitc.call_opaque "__runtime_buffer_arg"(%320) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1594 = emitc.cast %15 : i32 to i64
    %1595 = emitc.call_opaque "__runtime_buffer_offset"(%1593, %1594) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1596 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1597 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1598 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1599 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1600 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1601 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %1595, %5, %1591, %1592, %1600, %4, %1596, %1597, %1598, %1599, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1602 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1603 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1604 = emitc.call_opaque "__runtime_buffer_arg"(%320) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1605 = emitc.cast %13 : i32 to i64
    %1606 = emitc.call_opaque "__runtime_buffer_offset"(%1604, %1605) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1607 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1608 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1609 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1610 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1611 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1612 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %1606, %11, %1602, %1603, %1611, %3, %1607, %1608, %1609, %1610, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1613 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1614 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1615 = emitc.call_opaque "__runtime_buffer_arg"(%320) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1616 = emitc.cast %12 : i32 to i64
    %1617 = emitc.call_opaque "__runtime_buffer_offset"(%1615, %1616) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1618 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1619 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1620 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1621 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1622 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1623 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %1617, %9, %1613, %1614, %1622, %2, %1618, %1619, %1620, %1621, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1624 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1625 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1626 = emitc.call_opaque "__runtime_buffer_arg"(%320) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1627 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1628 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1629 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1630 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1631 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1632 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %94, %1626, %10, %1624, %1625, %1631, %1, %1627, %1628, %1629, %1630, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1633 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1634 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1635 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(2,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %94, %1633, %1635) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %1636 = emitc.call_opaque "__Runtime_dma_createio_4"(%94, %1632, %1633, %1634, %1635) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=13, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1637 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1638 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1639 = emitc.call_opaque "__runtime_buffer_arg"(%324) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1640 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1641 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1642 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1643 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1644 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1645 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %47, %1639, %16, %1637, %1638, %1644, %1, %1640, %1641, %1642, %1643, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=13, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1646 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1647 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1648 = emitc.call_opaque "__runtime_buffer_arg"(%323) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1649 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1650 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1651 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1652 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1653 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1654 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %47, %1648, %14, %1646, %1647, %1653, %1, %1649, %1650, %1651, %1652, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1655 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1656 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1657 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,6), direction=MM2S */"
    %1658 = emitc.call_opaque "__Runtime_dma_createio_4"(%47, %1654, %1655, %1656, %1657) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=14, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1659 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1660 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1661 = emitc.call_opaque "__runtime_buffer_arg"(%329) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1662 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1663 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1664 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1665 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1666 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1667 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %83, %1661, %16, %1659, %1660, %1666, %2, %1662, %1663, %1664, %1665, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=14, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1668 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1669 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1670 = emitc.call_opaque "__runtime_buffer_arg"(%328) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1671 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1672 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1673 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1674 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1675 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1676 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %83, %1670, %14, %1668, %1669, %1675, %2, %1671, %1672, %1673, %1674, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1677 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1678 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1679 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,6), direction=MM2S */"
    %1680 = emitc.call_opaque "__Runtime_dma_createio_4"(%83, %1676, %1677, %1678, %1679) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=15, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1681 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1682 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1683 = emitc.call_opaque "__runtime_buffer_arg"(%334) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1684 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1685 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1686 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1687 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1688 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1689 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %121, %1683, %16, %1681, %1682, %1688, %3, %1684, %1685, %1686, %1687, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=15, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1690 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1691 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1692 = emitc.call_opaque "__runtime_buffer_arg"(%333) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1693 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1694 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1695 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1696 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1697 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1698 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %121, %1692, %14, %1690, %1691, %1697, %3, %1693, %1694, %1695, %1696, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1699 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1700 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1701 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,6), direction=MM2S */"
    %1702 = emitc.call_opaque "__Runtime_dma_createio_4"(%121, %1698, %1699, %1700, %1701) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=16, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %1703 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1704 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1705 = emitc.call_opaque "__runtime_buffer_arg"(%339) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1706 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1707 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1708 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1709 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1710 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1711 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %159, %1705, %16, %1703, %1704, %1710, %4, %1706, %1707, %1708, %1709, %5) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=16, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %1712 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %1713 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1714 = emitc.call_opaque "__runtime_buffer_arg"(%338) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1715 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1716 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1717 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1718 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1719 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1720 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %159, %1714, %14, %1712, %1713, %1719, %4, %1715, %1716, %1717, %1718, %5) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1721 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1722 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1723 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,6), direction=MM2S */"
    %1724 = emitc.call_opaque "__Runtime_dma_createio_4"(%159, %1720, %1721, %1722, %1723) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %1725 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1726 = emitc.call_opaque "__Runtime_startio"(%arg0, %904, %203, %1725) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1727 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1728 = emitc.call_opaque "__Runtime_startio"(%arg0, %1148, %249, %1727) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1729 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1730 = emitc.call_opaque "__Runtime_startio"(%arg0, %1392, %295, %1729) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1731 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1732 = emitc.call_opaque "__Runtime_startio"(%arg0, %1636, %341, %1731) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1733 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1734 = emitc.call_opaque "__Runtime_startio"(%arg0, %379, %32, %1733) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1735 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1736 = emitc.call_opaque "__Runtime_startio"(%arg0, %401, %38, %1735) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1737 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1738 = emitc.call_opaque "__Runtime_startio"(%arg0, %423, %44, %1737) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1739 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1740 = emitc.call_opaque "__Runtime_startio"(%arg0, %445, %50, %1739) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1741 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1742 = emitc.call_opaque "__Runtime_startio"(%arg0, %480, %64, %1741) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1743 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1744 = emitc.call_opaque "__Runtime_startio"(%arg0, %502, %72, %1743) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1745 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1746 = emitc.call_opaque "__Runtime_startio"(%arg0, %524, %80, %1745) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1747 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1748 = emitc.call_opaque "__Runtime_startio"(%arg0, %546, %88, %1747) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1749 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1750 = emitc.call_opaque "__Runtime_startio"(%arg0, %581, %102, %1749) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1751 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1752 = emitc.call_opaque "__Runtime_startio"(%arg0, %603, %110, %1751) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1753 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1754 = emitc.call_opaque "__Runtime_startio"(%arg0, %625, %118, %1753) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1755 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1756 = emitc.call_opaque "__Runtime_startio"(%arg0, %647, %126, %1755) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1757 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1758 = emitc.call_opaque "__Runtime_startio"(%arg0, %682, %140, %1757) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1759 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1760 = emitc.call_opaque "__Runtime_startio"(%arg0, %704, %148, %1759) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1761 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1762 = emitc.call_opaque "__Runtime_startio"(%arg0, %726, %156, %1761) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1763 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1764 = emitc.call_opaque "__Runtime_startio"(%arg0, %748, %164, %1763) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1765 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1766 = emitc.call_opaque "__Runtime_startio"(%arg0, %792, %170, %1765) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1767 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1768 = emitc.call_opaque "__Runtime_startio"(%arg0, %814, %173, %1767) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1769 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1770 = emitc.call_opaque "__Runtime_startio"(%arg0, %836, %176, %1769) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1771 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1772 = emitc.call_opaque "__Runtime_startio"(%arg0, %858, %179, %1771) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1773 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1774 = emitc.call_opaque "__Runtime_startio"(%arg0, %1036, %210, %1773) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1775 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1776 = emitc.call_opaque "__Runtime_startio"(%arg0, %1058, %215, %1775) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1777 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1778 = emitc.call_opaque "__Runtime_startio"(%arg0, %1080, %220, %1777) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1779 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1780 = emitc.call_opaque "__Runtime_startio"(%arg0, %1102, %225, %1779) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1781 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1782 = emitc.call_opaque "__Runtime_startio"(%arg0, %1280, %256, %1781) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1783 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1784 = emitc.call_opaque "__Runtime_startio"(%arg0, %1302, %261, %1783) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1785 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1786 = emitc.call_opaque "__Runtime_startio"(%arg0, %1324, %266, %1785) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1787 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1788 = emitc.call_opaque "__Runtime_startio"(%arg0, %1346, %271, %1787) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1789 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1790 = emitc.call_opaque "__Runtime_startio"(%arg0, %1524, %302, %1789) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1791 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1792 = emitc.call_opaque "__Runtime_startio"(%arg0, %1546, %307, %1791) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1793 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1794 = emitc.call_opaque "__Runtime_startio"(%arg0, %1568, %312, %1793) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1795 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1796 = emitc.call_opaque "__Runtime_startio"(%arg0, %1590, %317, %1795) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1797 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1798 = emitc.call_opaque "__Runtime_startio"(%arg0, %926, %187, %1797) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1799 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1800 = emitc.call_opaque "__Runtime_startio"(%arg0, %948, %192, %1799) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1801 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1802 = emitc.call_opaque "__Runtime_startio"(%arg0, %970, %197, %1801) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1803 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1804 = emitc.call_opaque "__Runtime_startio"(%arg0, %992, %202, %1803) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1805 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1806 = emitc.call_opaque "__Runtime_startio"(%arg0, %1170, %233, %1805) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1807 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1808 = emitc.call_opaque "__Runtime_startio"(%arg0, %1192, %238, %1807) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1809 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1810 = emitc.call_opaque "__Runtime_startio"(%arg0, %1214, %243, %1809) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1811 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1812 = emitc.call_opaque "__Runtime_startio"(%arg0, %1236, %248, %1811) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1813 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1814 = emitc.call_opaque "__Runtime_startio"(%arg0, %1414, %279, %1813) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1815 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1816 = emitc.call_opaque "__Runtime_startio"(%arg0, %1436, %284, %1815) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1817 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1818 = emitc.call_opaque "__Runtime_startio"(%arg0, %1458, %289, %1817) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1819 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1820 = emitc.call_opaque "__Runtime_startio"(%arg0, %1480, %294, %1819) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1821 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1822 = emitc.call_opaque "__Runtime_startio"(%arg0, %1658, %325, %1821) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1823 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1824 = emitc.call_opaque "__Runtime_startio"(%arg0, %1680, %330, %1823) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1825 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1826 = emitc.call_opaque "__Runtime_startio"(%arg0, %1702, %335, %1825) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1827 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1828 = emitc.call_opaque "__Runtime_startio"(%arg0, %1724, %340, %1827) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1829 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1830 = emitc.call_opaque "__Runtime_startio"(%arg0, %357, %51, %1829) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1831 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1832 = emitc.call_opaque "__Runtime_startio"(%arg0, %458, %89, %1831) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1833 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1834 = emitc.call_opaque "__Runtime_startio"(%arg0, %559, %127, %1833) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1835 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1836 = emitc.call_opaque "__Runtime_startio"(%arg0, %660, %165, %1835) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1837 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1838 = emitc.call_opaque "__Runtime_startio"(%arg0, %770, %180, %1837) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1839 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1840 = emitc.call_opaque "__Runtime_startio"(%arg0, %1014, %226, %1839) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1841 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1842 = emitc.call_opaque "__Runtime_startio"(%arg0, %1258, %272, %1841) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1843 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1844 = emitc.call_opaque "__Runtime_startio"(%arg0, %1502, %318, %1843) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* Wait for 5 event(s) */"
    emitc.call_opaque "__Runtime_wait"(%arg0, %344) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"event">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %1726) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %1728) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %1730) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %1732) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.verbatim "/* AieRt debug snapshot (gated by AIE_DEBUG_LOG flag) */"
    emitc.verbatim "if (AIEHLC_LOG_ENABLED()) {"
    emitc.verbatim "  uint8_t _dbg_io_cols[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 1, 2, 3, 3, 0, 1, 2, 3, 1, 0, 1, 2, 3, 3, 0, 1, 2, 3, 2, 0, 1, 2, 3, 2, 0, 1, 2, 3, 3, 0, 1, 2, 3, 2, 0, 1, 2, 3};"
    emitc.verbatim "  uint8_t _dbg_io_rows[] = {0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 3, 3, 3, 0, 3, 3, 3, 3, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6};"
    emitc.verbatim "  uint8_t _dbg_io_chs[] = {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};"
    emitc.verbatim "  uint8_t _dbg_io_bds[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 1, 2, 2, 2, 2, 7, 4, 4, 4, 4, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 11, 2, 2, 2, 2, 8, 4, 4, 4, 4};"
    emitc.verbatim "  int _dbg_io_dirs[] = {DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S};"
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
