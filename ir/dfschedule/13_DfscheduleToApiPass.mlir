module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], dfschedule.control_kernel_ops = 1 : i64, dfschedule.grouped_lock_inits = [], routing.control_packet_aot = 1 : i64, routing.control_packet_mode = 1 : i64, routing.control_packet_nresp = 16 : i64, routing.control_packet_stream_id = 16 : i64, routing.control_plan_dedicated_shim = 1 : i64, routing.control_plan_exclusive = 1 : i64, routing.control_plan_group_reg_write = 1 : i64, routing.control_plan_mm2s_ch = 0 : i64, routing.control_plan_op_control_packet = 1 : i64, routing.control_plan_s2mm_ch = 0 : i64, routing.control_plan_shim_bd_ctrl = 1 : i64, routing.control_plan_shim_col = 0 : i64, routing.ctrlpkt_plan = [0, 1, 0, 4, 3, 1, 4, 4, 1, 4, 5, 1, 4, 6, 1, 4], routing.ctrlpkt_writes = [], routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 1 : i64, routing.output_tile_major = 1 : i64} {
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
    %1 = "emitc.constant"() <{value = 17 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 18 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = 20 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 12 : i32}> : () -> i32
    %6 = "emitc.constant"() <{value = 13 : i32}> : () -> i32
    %7 = "emitc.constant"() <{value = 14 : i32}> : () -> i32
    %8 = "emitc.constant"() <{value = 15 : i32}> : () -> i32
    %9 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %10 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %11 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %12 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %13 = "emitc.constant"() <{value = 50176 : i32}> : () -> i32
    %14 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %15 = "emitc.constant"() <{value = 100352 : i32}> : () -> i32
    %16 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %17 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %18 = "emitc.constant"() <{value = 150528 : i32}> : () -> i32
    %19 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %20 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %21 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %22 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %23 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %24 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %25 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %26 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %25) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %27 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %28 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %29 = emitc.call_opaque "XAie_TileLoc"(%27, %28) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %30 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %31 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %32 = emitc.call_opaque "XAie_TileLoc"(%30, %31) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %33 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %34 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %35 = emitc.call_opaque "XAie_TileLoc"(%33, %34) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %36 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %37 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %38 = emitc.call_opaque "XAie_TileLoc"(%36, %37) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %39 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %40 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %41 = emitc.call_opaque "XAie_TileLoc"(%39, %40) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,0) */"
    %42 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %43 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %44 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %43) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %45 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %46 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %47 = emitc.call_opaque "XAie_TileLoc"(%45, %46) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %48 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %49 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %50 = emitc.call_opaque "XAie_TileLoc"(%48, %49) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %51 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %52 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %53 = emitc.call_opaque "XAie_TileLoc"(%51, %52) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %54 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %55 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %56 = emitc.call_opaque "XAie_TileLoc"(%54, %55) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %57 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %58 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %59 = emitc.call_opaque "XAie_TileLoc"(%57, %58) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,0) */"
    %60 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %61 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %62 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %61) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %63 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %64 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %65 = emitc.call_opaque "XAie_TileLoc"(%63, %64) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %66 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %67 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %68 = emitc.call_opaque "XAie_TileLoc"(%66, %67) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %69 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %70 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %71 = emitc.call_opaque "XAie_TileLoc"(%69, %70) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %72 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %73 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %74 = emitc.call_opaque "XAie_TileLoc"(%72, %73) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %75 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %76 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %77 = emitc.call_opaque "XAie_TileLoc"(%75, %76) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,0) */"
    %78 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %79 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %80 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %79) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %81 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %82 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %83 = emitc.call_opaque "XAie_TileLoc"(%81, %82) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %84 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %85 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %86 = emitc.call_opaque "XAie_TileLoc"(%84, %85) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %87 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %88 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %89 = emitc.call_opaque "XAie_TileLoc"(%87, %88) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %90 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %91 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %92 = emitc.call_opaque "XAie_TileLoc"(%90, %91) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %93 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %94 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %95 = emitc.call_opaque "XAie_TileLoc"(%93, %94) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (4,0) */"
    %96 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %97 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %98 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %97) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,0) */"
    %99 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %100 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %101 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %100) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (4,0) */"
    %102 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %103 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %104 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %103) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,0) */"
    %105 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %106 = "emitc.constant"() <{value = 200704 : i64}> : () -> i64
    %107 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %106) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (4,0) */"
    %108 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %109 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %110 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %109) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,0) */"
    %111 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %112 = "emitc.constant"() <{value = 401408 : i64}> : () -> i64
    %113 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %112) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,0) */"
    %114 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %115 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %116 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %115) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 3 for tile (4,0) */"
    %117 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %118 = "emitc.constant"() <{value = 602112 : i64}> : () -> i64
    %119 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %118) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 3 for tile (3,0) */"
    %120 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    emitc.verbatim "/* control-packet group-write fabric setup */"
    emitc.verbatim "static __Runtime_CtrlRowFabric __ctrl_fabric_0;"
    emitc.verbatim "static __Runtime_CtrlRowChain __ctrl_fabric_0_rows[] = {{3, 1, 4}, {4, 1, 4}, {5, 1, 4}, {6, 1, 4}};"
    emitc.verbatim "__Runtime_ctrl_plan_init(&__ctrl_fabric_0, dev, 0, 0, 1, __ctrl_fabric_0_rows, 4);"
    emitc.verbatim "__Runtime_ctrl_plan_set_exclusive(&__ctrl_fabric_0, 1);"
    %121 = "emitc.constant"() <{value = #emitc.opaque<"&__ctrl_fabric_0">}> : () -> !emitc.ptr<!emitc.opaque<"__Runtime_CtrlRowFabric">>
    emitc.verbatim "/* Load Kernel Group: 16 tile(s) */"
    %122 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %123 = "emitc.constant"() <{value = #emitc.opaque<"&__ctrl_fabric_0">}> : () -> !emitc.ptr<!emitc.opaque<"__Runtime_CtrlRowFabric">>
    %124 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %125 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %126 = emitc.call_opaque "__Runtime_load_kernel_group_16t_ctrl"(%arg0, %123, %32, %35, %38, %41, %50, %53, %56, %59, %68, %71, %74, %77, %86, %89, %92, %95, %122, %124, %125) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.ptr<!emitc.opaque<"__Runtime_CtrlRowFabric">>, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, i32, i32, i32) -> !emitc.opaque<"kernel_group">
    emitc.verbatim "/* Launch Kernel Group */"
    %127 = "emitc.constant"() <{value = #emitc.opaque<"&__ctrl_fabric_0">}> : () -> !emitc.ptr<!emitc.opaque<"__Runtime_CtrlRowFabric">>
    %128 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %129 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %130 = emitc.call_opaque "__Runtime_launch_kernel_group_ctrl"(%arg0, %127, %126, %128, %129) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.ptr<!emitc.opaque<"__Runtime_CtrlRowFabric">>, !emitc.opaque<"kernel_group">, i32, i32) -> !emitc.opaque<"event">
    emitc.verbatim "__Runtime_ctrl_shim_bd_begin(&__ctrl_fabric_0);"
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %131 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %132 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %133 = emitc.call_opaque "__runtime_buffer_arg"(%26) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %134 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %135 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %136 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %137 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %138 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %139 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %133, %23, %131, %132, %138, %23, %134, %135, %136, %137, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %140 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %141 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %142 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(1,0), direction=MM2S */"
    %143 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %139, %140, %141, %142) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %144 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %145 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %146 = emitc.call_opaque "__runtime_buffer_arg"(%44) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %147 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %148 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %149 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %150 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %151 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %152 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %47, %146, %23, %144, %145, %151, %23, %147, %148, %149, %150, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %153 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %154 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %155 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */"
    %156 = emitc.call_opaque "__Runtime_dma_createio_4"(%47, %152, %153, %154, %155) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %157 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %158 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %159 = emitc.call_opaque "__runtime_buffer_arg"(%62) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %160 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %161 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %162 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %163 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %164 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %165 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %65, %159, %23, %157, %158, %164, %23, %160, %161, %162, %163, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %166 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %167 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %168 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */"
    %169 = emitc.call_opaque "__Runtime_dma_createio_4"(%65, %165, %166, %167, %168) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %170 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %171 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %172 = emitc.call_opaque "__runtime_buffer_arg"(%80) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %173 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %174 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %175 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %176 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %177 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %178 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %83, %172, %23, %170, %171, %177, %23, %173, %174, %175, %176, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %179 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %180 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %181 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(4,0), direction=MM2S */"
    %182 = emitc.call_opaque "__Runtime_dma_createio_4"(%83, %178, %179, %180, %181) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %183 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %184 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %185 = emitc.call_opaque "__runtime_buffer_arg"(%98) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %186 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %187 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %188 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %189 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %190 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %191 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %192 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %193 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %194 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %195 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %196 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %197 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %198 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %199 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %200 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %29, %185, %22, %183, %184, %199, %23, %186, %187, %188, %189, %24, %190, %191, %192, %193, %194, %195, %196, %197, %198) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %201 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %202 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %203 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(1,0), direction=MM2S */"
    %204 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %200, %201, %202, %203) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %205 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %206 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %207 = emitc.call_opaque "__runtime_buffer_arg"(%101) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %208 = emitc.cast %18 : i32 to i64
    %209 = emitc.call_opaque "__runtime_buffer_offset"(%207, %208) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %210 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %211 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %212 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %213 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %214 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %215 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %216 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %217 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %218 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %219 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %220 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %221 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %222 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %223 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %224 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %209, %19, %205, %206, %223, %17, %210, %211, %212, %213, %24, %214, %215, %216, %217, %218, %219, %220, %221, %222) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %225 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %226 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %227 = emitc.call_opaque "__runtime_buffer_arg"(%101) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %228 = emitc.cast %15 : i32 to i64
    %229 = emitc.call_opaque "__runtime_buffer_offset"(%227, %228) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %230 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %231 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %232 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %233 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %234 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %235 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %236 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %237 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %238 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %239 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %240 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %241 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %242 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %243 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %244 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %229, %16, %225, %226, %243, %14, %230, %231, %232, %233, %24, %234, %235, %236, %237, %238, %239, %240, %241, %242) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %245 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %246 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %247 = emitc.call_opaque "__runtime_buffer_arg"(%101) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %248 = emitc.cast %13 : i32 to i64
    %249 = emitc.call_opaque "__runtime_buffer_offset"(%247, %248) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %250 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %251 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %252 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %253 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %254 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %255 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %256 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %257 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %258 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %259 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %260 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %261 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %262 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %263 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %264 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %249, %20, %245, %246, %263, %19, %250, %251, %252, %253, %24, %254, %255, %256, %257, %258, %259, %260, %261, %262) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=3136, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %265 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %266 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %267 = emitc.call_opaque "__runtime_buffer_arg"(%101) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %268 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %269 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %270 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %271 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %272 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %273 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %274 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %275 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %276 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %277 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %278 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %279 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %280 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %281 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %282 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %267, %21, %265, %266, %281, %16, %268, %269, %270, %271, %24, %272, %273, %274, %275, %276, %277, %278, %279, %280) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %283 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %284 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %285 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(4,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(4,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %83, %283, %285) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %286 = emitc.call_opaque "__Runtime_dma_createio_4"(%83, %282, %283, %284, %285) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %287 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %288 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %289 = emitc.call_opaque "__runtime_buffer_arg"(%104) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %290 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %291 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %292 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %293 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %294 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %295 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %296 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %297 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %298 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %299 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %300 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %301 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %302 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %303 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %304 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %47, %289, %22, %287, %288, %303, %23, %290, %291, %292, %293, %24, %294, %295, %296, %297, %298, %299, %300, %301, %302) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %305 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %306 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %307 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */"
    %308 = emitc.call_opaque "__Runtime_dma_createio_4"(%47, %304, %305, %306, %307) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %309 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %310 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %311 = emitc.call_opaque "__runtime_buffer_arg"(%107) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %312 = emitc.cast %18 : i32 to i64
    %313 = emitc.call_opaque "__runtime_buffer_offset"(%311, %312) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %314 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %315 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %316 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %317 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %318 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %319 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %320 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %321 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %322 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %323 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %324 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %325 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %326 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %327 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %328 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %313, %12, %309, %310, %327, %11, %314, %315, %316, %317, %24, %318, %319, %320, %321, %322, %323, %324, %325, %326) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %329 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %330 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %331 = emitc.call_opaque "__runtime_buffer_arg"(%107) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %332 = emitc.cast %15 : i32 to i64
    %333 = emitc.call_opaque "__runtime_buffer_offset"(%331, %332) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %334 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %335 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %336 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %337 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %338 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %339 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %340 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %341 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %342 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %343 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %344 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %345 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %346 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %347 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %348 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %333, %10, %329, %330, %347, %12, %334, %335, %336, %337, %24, %338, %339, %340, %341, %342, %343, %344, %345, %346) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %349 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %350 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %351 = emitc.call_opaque "__runtime_buffer_arg"(%107) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %352 = emitc.cast %13 : i32 to i64
    %353 = emitc.call_opaque "__runtime_buffer_offset"(%351, %352) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %354 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %355 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %356 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %357 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %358 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %359 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %360 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %361 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %362 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %363 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %364 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %365 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %366 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %367 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %368 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %353, %9, %349, %350, %367, %10, %354, %355, %356, %357, %24, %358, %359, %360, %361, %362, %363, %364, %365, %366) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=7, len=3136, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %369 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %370 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %371 = emitc.call_opaque "__runtime_buffer_arg"(%107) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %372 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %373 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %374 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %375 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %376 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %377 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %378 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %379 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %380 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %381 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %382 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %383 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %384 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %385 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %386 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %371, %17, %369, %370, %385, %9, %372, %373, %374, %375, %24, %376, %377, %378, %379, %380, %381, %382, %383, %384) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %387 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %388 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %389 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=7, tile=(4,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(4,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %83, %387, %389) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %390 = emitc.call_opaque "__Runtime_dma_createio_4"(%83, %386, %387, %388, %389) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %391 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %392 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %393 = emitc.call_opaque "__runtime_buffer_arg"(%110) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %394 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %395 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %396 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %397 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %398 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %399 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %400 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %401 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %402 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %403 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %404 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %405 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %406 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %407 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %408 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %393, %22, %391, %392, %407, %23, %394, %395, %396, %397, %24, %398, %399, %400, %401, %402, %403, %404, %405, %406) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %409 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %410 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %411 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(3,0), direction=MM2S */"
    %412 = emitc.call_opaque "__Runtime_dma_createio_4"(%65, %408, %409, %410, %411) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=6, len=3136, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %413 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %414 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %415 = emitc.call_opaque "__runtime_buffer_arg"(%113) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %416 = emitc.cast %18 : i32 to i64
    %417 = emitc.call_opaque "__runtime_buffer_offset"(%415, %416) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %418 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %419 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %420 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %421 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %422 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %423 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %424 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %425 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %426 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %427 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %428 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %429 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %430 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %431 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %432 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %417, %14, %413, %414, %431, %8, %418, %419, %420, %421, %24, %422, %423, %424, %425, %426, %427, %428, %429, %430) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %433 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %434 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %435 = emitc.call_opaque "__runtime_buffer_arg"(%113) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %436 = emitc.cast %15 : i32 to i64
    %437 = emitc.call_opaque "__runtime_buffer_offset"(%435, %436) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %438 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %439 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %440 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %441 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %442 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %443 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %444 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %445 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %446 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %447 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %448 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %449 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %450 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %451 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %452 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %437, %19, %433, %434, %451, %7, %438, %439, %440, %441, %24, %442, %443, %444, %445, %446, %447, %448, %449, %450) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %453 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %454 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %455 = emitc.call_opaque "__runtime_buffer_arg"(%113) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %456 = emitc.cast %13 : i32 to i64
    %457 = emitc.call_opaque "__runtime_buffer_offset"(%455, %456) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %458 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %459 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %460 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %461 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %462 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %463 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %464 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %465 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %466 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %467 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %468 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %469 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %470 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %471 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %472 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %457, %16, %453, %454, %471, %6, %458, %459, %460, %461, %24, %462, %463, %464, %465, %466, %467, %468, %469, %470) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %473 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %474 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %475 = emitc.call_opaque "__runtime_buffer_arg"(%113) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %476 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %477 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %478 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %479 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %480 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %481 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %482 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %483 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %484 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %485 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %486 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %487 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %488 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %489 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %490 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %475, %20, %473, %474, %489, %5, %476, %477, %478, %479, %24, %480, %481, %482, %483, %484, %485, %486, %487, %488) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %491 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %492 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %493 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %65, %491, %493) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %494 = emitc.call_opaque "__Runtime_dma_createio_4"(%65, %490, %491, %492, %493) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %495 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %496 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %497 = emitc.call_opaque "__runtime_buffer_arg"(%116) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %498 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %499 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %500 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %501 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %502 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %503 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %504 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %505 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %506 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %507 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %508 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %509 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %510 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %511 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %512 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %83, %497, %11, %495, %496, %511, %23, %498, %499, %500, %501, %24, %502, %503, %504, %505, %506, %507, %508, %509, %510) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %513 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %514 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %515 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=11, tile=(4,0), direction=MM2S */"
    %516 = emitc.call_opaque "__Runtime_dma_createio_4"(%83, %512, %513, %514, %515) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=3136, enable_packet=false, packet_id=20, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %517 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %518 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %519 = emitc.call_opaque "__runtime_buffer_arg"(%119) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %520 = emitc.cast %18 : i32 to i64
    %521 = emitc.call_opaque "__runtime_buffer_offset"(%519, %520) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %522 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %523 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %524 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %525 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %526 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %527 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %528 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %529 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %530 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %531 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %532 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %533 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %534 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %535 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %536 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %521, %11, %517, %518, %535, %4, %522, %523, %524, %525, %24, %526, %527, %528, %529, %530, %531, %532, %533, %534) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=19, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %537 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %538 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %539 = emitc.call_opaque "__runtime_buffer_arg"(%119) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %540 = emitc.cast %15 : i32 to i64
    %541 = emitc.call_opaque "__runtime_buffer_offset"(%539, %540) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %542 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %543 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %544 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %545 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %546 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %547 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %548 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %549 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %550 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %551 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %552 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %553 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %554 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %555 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %556 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %541, %12, %537, %538, %555, %3, %542, %543, %544, %545, %24, %546, %547, %548, %549, %550, %551, %552, %553, %554) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=18, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %557 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %558 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %559 = emitc.call_opaque "__runtime_buffer_arg"(%119) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %560 = emitc.cast %13 : i32 to i64
    %561 = emitc.call_opaque "__runtime_buffer_offset"(%559, %560) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %562 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %563 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %564 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %565 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %566 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %567 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %568 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %569 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %570 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %571 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %572 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %573 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %574 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %575 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %576 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %561, %10, %557, %558, %575, %2, %562, %563, %564, %565, %24, %566, %567, %568, %569, %570, %571, %572, %573, %574) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=17, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %577 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %578 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %579 = emitc.call_opaque "__runtime_buffer_arg"(%119) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %580 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %581 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %582 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %583 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %584 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %585 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %586 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %587 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %588 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %589 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %590 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %591 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %592 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %593 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %594 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %65, %579, %9, %577, %578, %593, %1, %580, %581, %582, %583, %24, %584, %585, %586, %587, %588, %589, %590, %591, %592) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %595 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %596 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %597 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %65, %595, %597) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %598 = emitc.call_opaque "__Runtime_dma_createio_4"(%65, %594, %595, %596, %597) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %599 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %600 = emitc.call_opaque "__Runtime_startio"(%arg0, %286, %102, %599) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %601 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %602 = emitc.call_opaque "__Runtime_startio"(%arg0, %390, %108, %601) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %603 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %604 = emitc.call_opaque "__Runtime_startio"(%arg0, %494, %114, %603) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %605 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %606 = emitc.call_opaque "__Runtime_startio"(%arg0, %598, %120, %605) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %607 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %608 = emitc.call_opaque "__Runtime_startio"(%arg0, %143, %42, %607) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %609 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %610 = emitc.call_opaque "__Runtime_startio"(%arg0, %156, %60, %609) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %611 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %612 = emitc.call_opaque "__Runtime_startio"(%arg0, %169, %78, %611) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %613 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %614 = emitc.call_opaque "__Runtime_startio"(%arg0, %182, %96, %613) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %615 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %616 = emitc.call_opaque "__Runtime_startio"(%arg0, %204, %99, %615) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %617 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %618 = emitc.call_opaque "__Runtime_startio"(%arg0, %308, %105, %617) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %619 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %620 = emitc.call_opaque "__Runtime_startio"(%arg0, %412, %111, %619) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %621 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %622 = emitc.call_opaque "__Runtime_startio"(%arg0, %516, %117, %621) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* Wait for 5 event(s) */"
    emitc.verbatim "__Runtime_ctrl_shim_bd_commit();"
    emitc.call_opaque "__Runtime_wait"(%arg0, %130) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"event">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %600) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %602) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %604) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %606) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.verbatim "/* AieRt debug snapshot (gated by AIE_DEBUG_LOG flag) */"
    emitc.verbatim "if (AIEHLC_LOG_ENABLED()) {"
    emitc.verbatim "  uint8_t _dbg_io_cols[] = {1, 2, 3, 4, 1, 4, 2, 4, 3, 3, 4, 3};"
    emitc.verbatim "  uint8_t _dbg_io_rows[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};"
    emitc.verbatim "  uint8_t _dbg_io_chs[] = {0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1};"
    emitc.verbatim "  uint8_t _dbg_io_bds[] = {0, 0, 0, 0, 1, 2, 1, 7, 1, 3, 11, 8};"
    emitc.verbatim "  int _dbg_io_dirs[] = {DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_MM2S, DMA_S2MM};"
    emitc.verbatim "  uint8_t _dbg_t_cols[] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};"
    emitc.verbatim "  uint8_t _dbg_t_rows[] = {3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6};"
    emitc.verbatim "  AieRt_DebugSnapshotFromCoords(dev,\0A      _dbg_io_cols, _dbg_io_rows, _dbg_io_chs, _dbg_io_bds, _dbg_io_dirs, 12,\0A      _dbg_t_cols, _dbg_t_rows, 16);"
    emitc.verbatim "}"
    emitc.verbatim "/* Kernel klog only (AIE_KERNEL_CONFIG_TRACE without AIE_DEBUG_LOG) */"
    emitc.verbatim "if (AIEHLC_KLOG_ENABLED() && !AIEHLC_LOG_ENABLED()) {"
    emitc.verbatim "  uint8_t _klog_cols[] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};"
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
