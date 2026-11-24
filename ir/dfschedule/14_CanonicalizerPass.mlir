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
    %0 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    %1 = "emitc.constant"() <{value = 49 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = 12880 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = 224 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 920 : i32}> : () -> i32
    %6 = "emitc.constant"() <{value = 61 : i32}> : () -> i32
    %7 = "emitc.constant"() <{value = 18544 : i32}> : () -> i32
    %8 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    %9 = "emitc.constant"() <{value = 3136 : i32}> : () -> i32
    %10 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %11 = "emitc.constant"() <{value = #emitc.opaque<"&__ctrl_fabric_0">}> : () -> !emitc.ptr<!emitc.opaque<"__Runtime_CtrlRowFabric">>
    %12 = "emitc.constant"() <{value = 602112 : i64}> : () -> i64
    %13 = "emitc.constant"() <{value = 154560 : i64}> : () -> i64
    %14 = "emitc.constant"() <{value = 401408 : i64}> : () -> i64
    %15 = "emitc.constant"() <{value = 103040 : i64}> : () -> i64
    %16 = "emitc.constant"() <{value = 200704 : i64}> : () -> i64
    %17 = "emitc.constant"() <{value = 51520 : i64}> : () -> i64
    %18 = "emitc.constant"() <{value = 9408 : i64}> : () -> i64
    %19 = "emitc.constant"() <{value = 6272 : i64}> : () -> i64
    %20 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %21 = "emitc.constant"() <{value = 3136 : i64}> : () -> i64
    %22 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %23 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %24 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %25 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %26 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %27 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %28 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %29 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %30 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %31 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %32 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %33 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %34 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %35 = "emitc.constant"() <{value = 150528 : i32}> : () -> i32
    %36 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %37 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %38 = "emitc.constant"() <{value = 100352 : i32}> : () -> i32
    %39 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %40 = "emitc.constant"() <{value = 50176 : i32}> : () -> i32
    %41 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %42 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %43 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %44 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %45 = "emitc.constant"() <{value = 15 : i32}> : () -> i32
    %46 = "emitc.constant"() <{value = 14 : i32}> : () -> i32
    %47 = "emitc.constant"() <{value = 13 : i32}> : () -> i32
    %48 = "emitc.constant"() <{value = 12 : i32}> : () -> i32
    %49 = "emitc.constant"() <{value = 20 : i32}> : () -> i32
    %50 = "emitc.constant"() <{value = 19 : i32}> : () -> i32
    %51 = "emitc.constant"() <{value = 18 : i32}> : () -> i32
    %52 = "emitc.constant"() <{value = 17 : i32}> : () -> i32
    %53 = "emitc.constant"() <{value = #emitc.opaque<"XAIE_MEM_CACHEABLE">}> : () -> i32
    emitc.verbatim "XAie_DevInst* dev = v1;"
    %54 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %28) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %55 = emitc.call_opaque "XAie_TileLoc"(%27, %26) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %56 = emitc.call_opaque "XAie_TileLoc"(%27, %25) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %57 = emitc.call_opaque "XAie_TileLoc"(%27, %24) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %58 = emitc.call_opaque "XAie_TileLoc"(%27, %23) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %59 = emitc.call_opaque "XAie_TileLoc"(%27, %22) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,0) */"
    %60 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %21) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %61 = emitc.call_opaque "XAie_TileLoc"(%20, %26) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %62 = emitc.call_opaque "XAie_TileLoc"(%20, %25) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %63 = emitc.call_opaque "XAie_TileLoc"(%20, %24) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %64 = emitc.call_opaque "XAie_TileLoc"(%20, %23) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %65 = emitc.call_opaque "XAie_TileLoc"(%20, %22) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,0) */"
    %66 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %19) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %67 = emitc.call_opaque "XAie_TileLoc"(%25, %26) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %68 = emitc.call_opaque "XAie_TileLoc"(%25, %25) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %69 = emitc.call_opaque "XAie_TileLoc"(%25, %24) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %70 = emitc.call_opaque "XAie_TileLoc"(%25, %23) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %71 = emitc.call_opaque "XAie_TileLoc"(%25, %22) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,0) */"
    %72 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %18) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %73 = emitc.call_opaque "XAie_TileLoc"(%24, %26) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %74 = emitc.call_opaque "XAie_TileLoc"(%24, %25) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %75 = emitc.call_opaque "XAie_TileLoc"(%24, %24) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %76 = emitc.call_opaque "XAie_TileLoc"(%24, %23) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %77 = emitc.call_opaque "XAie_TileLoc"(%24, %22) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (4,0) */"
    %78 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %28) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,0) */"
    %79 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %28) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (4,0) */"
    %80 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %17) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,0) */"
    %81 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %16) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (4,0) */"
    %82 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %15) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,0) */"
    %83 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %14) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,0) */"
    %84 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %13) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 3 for tile (4,0) */"
    %85 = emitc.call_opaque "__runtime_buffer_offset"(%arg3, %12) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 3 for tile (3,0) */"
    emitc.verbatim "/* control-packet group-write fabric setup */"
    emitc.verbatim "static __Runtime_CtrlRowFabric __ctrl_fabric_0;"
    emitc.verbatim "static __Runtime_CtrlRowChain __ctrl_fabric_0_rows[] = {{3, 1, 4}, {4, 1, 4}, {5, 1, 4}, {6, 1, 4}};"
    emitc.verbatim "__Runtime_ctrl_plan_init(&__ctrl_fabric_0, dev, 0, 0, 1, __ctrl_fabric_0_rows, 4);"
    emitc.verbatim "__Runtime_ctrl_plan_set_exclusive(&__ctrl_fabric_0, 1);"
    emitc.verbatim "/* Load Kernel Group: 16 tile(s) */"
    %86 = emitc.call_opaque "__Runtime_load_kernel_group_16t_ctrl"(%arg0, %11, %56, %57, %58, %59, %62, %63, %64, %65, %68, %69, %70, %71, %74, %75, %76, %77, %10, %32, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.ptr<!emitc.opaque<"__Runtime_CtrlRowFabric">>, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, i32, i32, i32) -> !emitc.opaque<"kernel_group">
    emitc.verbatim "/* Launch Kernel Group */"
    %87 = emitc.call_opaque "__Runtime_launch_kernel_group_ctrl"(%arg0, %11, %86, %32, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.ptr<!emitc.opaque<"__Runtime_CtrlRowFabric">>, !emitc.opaque<"kernel_group">, i32, i32) -> !emitc.opaque<"event">
    emitc.verbatim "__Runtime_ctrl_shim_bd_begin(&__ctrl_fabric_0);"
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %88 = emitc.call_opaque "__runtime_buffer_arg"(%54) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %89 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %55, %88, %30, %9, %29, %30, %30, %30, %30, %30, %30, %29) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(1,0), direction=MM2S */"
    %90 = emitc.call_opaque "__Runtime_dma_createio_4"(%55, %89, %30, %30, %8) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %91 = emitc.call_opaque "__runtime_buffer_arg"(%60) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %92 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %61, %91, %30, %9, %29, %30, %30, %30, %30, %30, %30, %29) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */"
    %93 = emitc.call_opaque "__Runtime_dma_createio_4"(%61, %92, %30, %30, %8) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %94 = emitc.call_opaque "__runtime_buffer_arg"(%66) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %95 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %67, %94, %30, %9, %29, %30, %30, %30, %30, %30, %30, %29) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */"
    %96 = emitc.call_opaque "__Runtime_dma_createio_4"(%67, %95, %30, %30, %8) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %97 = emitc.call_opaque "__runtime_buffer_arg"(%72) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %98 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %73, %97, %30, %9, %29, %30, %30, %30, %30, %30, %30, %29) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(4,0), direction=MM2S */"
    %99 = emitc.call_opaque "__Runtime_dma_createio_4"(%73, %98, %30, %30, %8) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %100 = emitc.call_opaque "__runtime_buffer_arg"(%78) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %101 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %55, %100, %31, %7, %29, %30, %30, %30, %30, %30, %30, %29, %33, %37, %6, %5, %50, %4, %37, %3, %37) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(1,0), direction=MM2S */"
    %102 = emitc.call_opaque "__Runtime_dma_createio_4"(%55, %101, %31, %31, %8) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %103 = emitc.call_opaque "__runtime_buffer_arg"(%79) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %104 = emitc.cast %35 : i32 to i64
    %105 = emitc.call_opaque "__runtime_buffer_offset"(%103, %104) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %106 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %105, %34, %9, %29, %30, %36, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %107 = emitc.call_opaque "__runtime_buffer_arg"(%79) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %108 = emitc.cast %38 : i32 to i64
    %109 = emitc.call_opaque "__runtime_buffer_offset"(%107, %108) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %110 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %109, %37, %9, %29, %30, %39, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %111 = emitc.call_opaque "__runtime_buffer_arg"(%79) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %112 = emitc.cast %40 : i32 to i64
    %113 = emitc.call_opaque "__runtime_buffer_offset"(%111, %112) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %114 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %113, %33, %9, %29, %30, %34, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=3136, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %115 = emitc.call_opaque "__runtime_buffer_arg"(%79) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %116 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %115, %32, %9, %29, %30, %37, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(4,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(4,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %73, %30, %0) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %117 = emitc.call_opaque "__Runtime_dma_createio_4"(%73, %116, %30, %32, %0) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %118 = emitc.call_opaque "__runtime_buffer_arg"(%80) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %119 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %61, %118, %31, %7, %29, %30, %30, %30, %30, %30, %30, %29, %33, %37, %6, %5, %50, %4, %37, %3, %37) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */"
    %120 = emitc.call_opaque "__Runtime_dma_createio_4"(%61, %119, %31, %31, %8) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %121 = emitc.call_opaque "__runtime_buffer_arg"(%81) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %122 = emitc.cast %35 : i32 to i64
    %123 = emitc.call_opaque "__runtime_buffer_offset"(%121, %122) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %124 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %123, %41, %9, %29, %30, %42, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %125 = emitc.call_opaque "__runtime_buffer_arg"(%81) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %126 = emitc.cast %38 : i32 to i64
    %127 = emitc.call_opaque "__runtime_buffer_offset"(%125, %126) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %128 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %127, %43, %9, %29, %30, %41, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %129 = emitc.call_opaque "__runtime_buffer_arg"(%81) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %130 = emitc.cast %40 : i32 to i64
    %131 = emitc.call_opaque "__runtime_buffer_offset"(%129, %130) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %132 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %131, %44, %9, %29, %30, %43, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=7, len=3136, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %133 = emitc.call_opaque "__runtime_buffer_arg"(%81) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %134 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %133, %36, %9, %29, %30, %44, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=7, tile=(4,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(4,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %73, %31, %0) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %135 = emitc.call_opaque "__Runtime_dma_createio_4"(%73, %134, %31, %36, %0) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %136 = emitc.call_opaque "__runtime_buffer_arg"(%82) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %137 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %136, %31, %7, %29, %30, %30, %30, %30, %30, %30, %29, %33, %37, %6, %5, %50, %4, %37, %3, %37) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(3,0), direction=MM2S */"
    %138 = emitc.call_opaque "__Runtime_dma_createio_4"(%67, %137, %31, %31, %8) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=6, len=3136, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %139 = emitc.call_opaque "__runtime_buffer_arg"(%83) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %140 = emitc.cast %35 : i32 to i64
    %141 = emitc.call_opaque "__runtime_buffer_offset"(%139, %140) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %142 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %141, %39, %9, %29, %30, %45, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %143 = emitc.call_opaque "__runtime_buffer_arg"(%83) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %144 = emitc.cast %38 : i32 to i64
    %145 = emitc.call_opaque "__runtime_buffer_offset"(%143, %144) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %146 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %145, %34, %9, %29, %30, %46, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %147 = emitc.call_opaque "__runtime_buffer_arg"(%83) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %148 = emitc.cast %40 : i32 to i64
    %149 = emitc.call_opaque "__runtime_buffer_offset"(%147, %148) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %150 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %149, %37, %9, %29, %30, %47, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %151 = emitc.call_opaque "__runtime_buffer_arg"(%83) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %152 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %151, %33, %9, %29, %30, %48, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %67, %30, %0) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %153 = emitc.call_opaque "__Runtime_dma_createio_4"(%67, %152, %30, %33, %0) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %154 = emitc.call_opaque "__runtime_buffer_arg"(%84) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %155 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %73, %154, %42, %7, %29, %30, %30, %30, %30, %30, %30, %29, %33, %37, %6, %5, %50, %4, %37, %3, %37) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=11, tile=(4,0), direction=MM2S */"
    %156 = emitc.call_opaque "__Runtime_dma_createio_4"(%73, %155, %31, %42, %8) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=3136, enable_packet=false, packet_id=20, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %157 = emitc.call_opaque "__runtime_buffer_arg"(%85) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %158 = emitc.cast %35 : i32 to i64
    %159 = emitc.call_opaque "__runtime_buffer_offset"(%157, %158) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %160 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %159, %42, %9, %29, %30, %49, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=19, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %161 = emitc.call_opaque "__runtime_buffer_arg"(%85) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %162 = emitc.cast %38 : i32 to i64
    %163 = emitc.call_opaque "__runtime_buffer_offset"(%161, %162) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %164 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %163, %41, %9, %29, %30, %50, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=18, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %165 = emitc.call_opaque "__runtime_buffer_arg"(%85) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %166 = emitc.cast %40 : i32 to i64
    %167 = emitc.call_opaque "__runtime_buffer_offset"(%165, %166) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %168 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %167, %43, %9, %29, %30, %51, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=17, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %169 = emitc.call_opaque "__runtime_buffer_arg"(%85) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %170 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %67, %169, %44, %9, %29, %30, %52, %29, %30, %29, %30, %29, %32, %37, %10, %2, %1, %30, %30, %9, %10) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %67, %31, %0) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %171 = emitc.call_opaque "__Runtime_dma_createio_4"(%67, %170, %31, %44, %0) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %172 = emitc.call_opaque "__Runtime_startio"(%arg0, %117, %31, %2) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %173 = emitc.call_opaque "__Runtime_startio"(%arg0, %135, %32, %2) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %174 = emitc.call_opaque "__Runtime_startio"(%arg0, %153, %32, %2) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %175 = emitc.call_opaque "__Runtime_startio"(%arg0, %171, %33, %2) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %176 = emitc.call_opaque "__Runtime_startio"(%arg0, %90, %30, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %177 = emitc.call_opaque "__Runtime_startio"(%arg0, %93, %30, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %178 = emitc.call_opaque "__Runtime_startio"(%arg0, %96, %30, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %179 = emitc.call_opaque "__Runtime_startio"(%arg0, %99, %30, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %180 = emitc.call_opaque "__Runtime_startio"(%arg0, %102, %31, %37) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %181 = emitc.call_opaque "__Runtime_startio"(%arg0, %120, %31, %37) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %182 = emitc.call_opaque "__Runtime_startio"(%arg0, %138, %31, %37) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %183 = emitc.call_opaque "__Runtime_startio"(%arg0, %156, %33, %37) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* Wait for 5 event(s) */"
    emitc.verbatim "__Runtime_ctrl_shim_bd_commit();"
    emitc.call_opaque "__Runtime_wait"(%arg0, %87) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"event">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %172) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %173) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %174) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %175) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
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
