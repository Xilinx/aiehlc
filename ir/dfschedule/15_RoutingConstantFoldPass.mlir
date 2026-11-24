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
    %0 = "emitc.constant"() <{value = 150528 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 100352 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 50176 : i32}> : () -> i32
    emitc.verbatim "XAie_DevInst* dev = v1;"
    %3 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %4 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %5 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %6 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %7 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %8 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,0) */"
    %9 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 3136]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %10 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %11 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %12 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %13 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %14 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,0) */"
    %15 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 6272]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %16 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %17 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %18 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %19 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %20 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,0) */"
    %21 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 9408]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %22 = emitc.call_opaque "XAie_TileLoc"() {args = [4 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %23 = emitc.call_opaque "XAie_TileLoc"() {args = [4 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %24 = emitc.call_opaque "XAie_TileLoc"() {args = [4 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %25 = emitc.call_opaque "XAie_TileLoc"() {args = [4 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %26 = emitc.call_opaque "XAie_TileLoc"() {args = [4 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (4,0) */"
    %27 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,0) */"
    %28 = emitc.call_opaque "__runtime_buffer_offset"(%arg3) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (4,0) */"
    %29 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 51520]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,0) */"
    %30 = emitc.call_opaque "__runtime_buffer_offset"(%arg3) {args = [0 : index, 200704]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (4,0) */"
    %31 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 103040]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,0) */"
    %32 = emitc.call_opaque "__runtime_buffer_offset"(%arg3) {args = [0 : index, 401408]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,0) */"
    %33 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 154560]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 3 for tile (4,0) */"
    %34 = emitc.call_opaque "__runtime_buffer_offset"(%arg3) {args = [0 : index, 602112]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 3 for tile (3,0) */"
    emitc.verbatim "/* control-packet group-write fabric setup */"
    emitc.verbatim "static __Runtime_CtrlRowFabric __ctrl_fabric_0;"
    emitc.verbatim "static __Runtime_CtrlRowChain __ctrl_fabric_0_rows[] = {{3, 1, 4}, {4, 1, 4}, {5, 1, 4}, {6, 1, 4}};"
    emitc.verbatim "__Runtime_ctrl_plan_init(&__ctrl_fabric_0, dev, 0, 0, 1, __ctrl_fabric_0_rows, 4);"
    emitc.verbatim "__Runtime_ctrl_plan_set_exclusive(&__ctrl_fabric_0, 1);"
    emitc.verbatim "/* Load Kernel Group: 16 tile(s) */"
    %35 = emitc.call_opaque "__Runtime_load_kernel_group_16t_ctrl"(%arg0, %5, %6, %7, %8, %11, %12, %13, %14, %17, %18, %19, %20, %23, %24, %25, %26) {args = [0 : index, #emitc.opaque<"&__ctrl_fabric_0">, 1 : index, 2 : index, 3 : index, 4 : index, 5 : index, 6 : index, 7 : index, 8 : index, 9 : index, 10 : index, 11 : index, 12 : index, 13 : index, 14 : index, 15 : index, 16 : index, 16 : i32, 2 : i32, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">) -> !emitc.opaque<"kernel_group">
    emitc.verbatim "/* Launch Kernel Group */"
    %36 = emitc.call_opaque "__Runtime_launch_kernel_group_ctrl"(%arg0, %35) {args = [0 : index, #emitc.opaque<"&__ctrl_fabric_0">, 1 : index, 2 : i32, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"kernel_group">) -> !emitc.opaque<"event">
    emitc.verbatim "__Runtime_ctrl_shim_bd_begin(&__ctrl_fabric_0);"
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %37 = emitc.call_opaque "__runtime_buffer_arg"(%3) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %38 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %4, %37) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(1,0), direction=MM2S */"
    %39 = emitc.call_opaque "__Runtime_dma_createio_4"(%4, %38) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %40 = emitc.call_opaque "__runtime_buffer_arg"(%9) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %41 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %10, %40) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */"
    %42 = emitc.call_opaque "__Runtime_dma_createio_4"(%10, %41) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %43 = emitc.call_opaque "__runtime_buffer_arg"(%15) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %44 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %16, %43) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */"
    %45 = emitc.call_opaque "__Runtime_dma_createio_4"(%16, %44) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %46 = emitc.call_opaque "__runtime_buffer_arg"(%21) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %47 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %22, %46) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(4,0), direction=MM2S */"
    %48 = emitc.call_opaque "__Runtime_dma_createio_4"(%22, %47) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %49 = emitc.call_opaque "__runtime_buffer_arg"(%27) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %50 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %4, %49) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 18544 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 61 : i32, 920 : i32, 19 : i32, 224 : i32, 4 : i32, 12880 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(1,0), direction=MM2S */"
    %51 = emitc.call_opaque "__Runtime_dma_createio_4"(%4, %50) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %52 = emitc.call_opaque "__runtime_buffer_arg"(%28) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %53 = emitc.cast %0 : i32 to i64
    %54 = emitc.call_opaque "__runtime_buffer_offset"(%52, %53) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %55 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %54) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, -1 : i32, 0 : i32, 7 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %56 = emitc.call_opaque "__runtime_buffer_arg"(%28) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %57 = emitc.cast %1 : i32 to i64
    %58 = emitc.call_opaque "__runtime_buffer_offset"(%56, %57) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %59 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %58) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, -1 : i32, 0 : i32, 6 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %60 = emitc.call_opaque "__runtime_buffer_arg"(%28) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %61 = emitc.cast %2 : i32 to i64
    %62 = emitc.call_opaque "__runtime_buffer_offset"(%60, %61) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %63 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %62) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 3136 : i32, -1 : i32, 0 : i32, 5 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=3136, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %64 = emitc.call_opaque "__runtime_buffer_arg"(%28) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %65 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %64) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 3136 : i32, -1 : i32, 0 : i32, 4 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(4,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(4,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %22) {args = [0 : index, 1 : index, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">) -> ()
    %66 = emitc.call_opaque "__Runtime_dma_createio_4"(%22, %65) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %67 = emitc.call_opaque "__runtime_buffer_arg"(%29) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %68 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %10, %67) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 18544 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 61 : i32, 920 : i32, 19 : i32, 224 : i32, 4 : i32, 12880 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */"
    %69 = emitc.call_opaque "__Runtime_dma_createio_4"(%10, %68) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %70 = emitc.call_opaque "__runtime_buffer_arg"(%30) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %71 = emitc.cast %0 : i32 to i64
    %72 = emitc.call_opaque "__runtime_buffer_offset"(%70, %71) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %73 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %72) {args = [0 : index, 1 : index, 2 : index, 10 : i32, 3136 : i32, -1 : i32, 0 : i32, 11 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %74 = emitc.call_opaque "__runtime_buffer_arg"(%30) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %75 = emitc.cast %1 : i32 to i64
    %76 = emitc.call_opaque "__runtime_buffer_offset"(%74, %75) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %77 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %76) {args = [0 : index, 1 : index, 2 : index, 9 : i32, 3136 : i32, -1 : i32, 0 : i32, 10 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %78 = emitc.call_opaque "__runtime_buffer_arg"(%30) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %79 = emitc.cast %2 : i32 to i64
    %80 = emitc.call_opaque "__runtime_buffer_offset"(%78, %79) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %81 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %80) {args = [0 : index, 1 : index, 2 : index, 8 : i32, 3136 : i32, -1 : i32, 0 : i32, 9 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=7, len=3136, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %82 = emitc.call_opaque "__runtime_buffer_arg"(%30) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %83 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %82) {args = [0 : index, 1 : index, 2 : index, 7 : i32, 3136 : i32, -1 : i32, 0 : i32, 8 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=7, tile=(4,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(4,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %22) {args = [0 : index, 1 : index, 1 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">) -> ()
    %84 = emitc.call_opaque "__Runtime_dma_createio_4"(%22, %83) {args = [0 : index, 1 : index, 1 : i32, 7 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %85 = emitc.call_opaque "__runtime_buffer_arg"(%31) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %86 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %85) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 18544 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 61 : i32, 920 : i32, 19 : i32, 224 : i32, 4 : i32, 12880 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(3,0), direction=MM2S */"
    %87 = emitc.call_opaque "__Runtime_dma_createio_4"(%16, %86) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=6, len=3136, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %88 = emitc.call_opaque "__runtime_buffer_arg"(%32) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %89 = emitc.cast %0 : i32 to i64
    %90 = emitc.call_opaque "__runtime_buffer_offset"(%88, %89) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %91 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %90) {args = [0 : index, 1 : index, 2 : index, 6 : i32, 3136 : i32, -1 : i32, 0 : i32, 15 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %92 = emitc.call_opaque "__runtime_buffer_arg"(%32) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %93 = emitc.cast %1 : i32 to i64
    %94 = emitc.call_opaque "__runtime_buffer_offset"(%92, %93) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %95 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %94) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, -1 : i32, 0 : i32, 14 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %96 = emitc.call_opaque "__runtime_buffer_arg"(%32) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %97 = emitc.cast %2 : i32 to i64
    %98 = emitc.call_opaque "__runtime_buffer_offset"(%96, %97) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %99 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %98) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, -1 : i32, 0 : i32, 13 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %100 = emitc.call_opaque "__runtime_buffer_arg"(%32) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %101 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %100) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 3136 : i32, -1 : i32, 0 : i32, 12 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %16) {args = [0 : index, 1 : index, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">) -> ()
    %102 = emitc.call_opaque "__Runtime_dma_createio_4"(%16, %101) {args = [0 : index, 1 : index, 0 : i32, 3 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %103 = emitc.call_opaque "__runtime_buffer_arg"(%33) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %104 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %22, %103) {args = [0 : index, 1 : index, 2 : index, 11 : i32, 18544 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 61 : i32, 920 : i32, 19 : i32, 224 : i32, 4 : i32, 12880 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=11, tile=(4,0), direction=MM2S */"
    %105 = emitc.call_opaque "__Runtime_dma_createio_4"(%22, %104) {args = [0 : index, 1 : index, 1 : i32, 11 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=3136, enable_packet=false, packet_id=20, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %106 = emitc.call_opaque "__runtime_buffer_arg"(%34) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %107 = emitc.cast %0 : i32 to i64
    %108 = emitc.call_opaque "__runtime_buffer_offset"(%106, %107) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %109 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %108) {args = [0 : index, 1 : index, 2 : index, 11 : i32, 3136 : i32, -1 : i32, 0 : i32, 20 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=19, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %110 = emitc.call_opaque "__runtime_buffer_arg"(%34) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %111 = emitc.cast %1 : i32 to i64
    %112 = emitc.call_opaque "__runtime_buffer_offset"(%110, %111) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %113 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %112) {args = [0 : index, 1 : index, 2 : index, 10 : i32, 3136 : i32, -1 : i32, 0 : i32, 19 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=18, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %114 = emitc.call_opaque "__runtime_buffer_arg"(%34) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %115 = emitc.cast %2 : i32 to i64
    %116 = emitc.call_opaque "__runtime_buffer_offset"(%114, %115) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %117 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %116) {args = [0 : index, 1 : index, 2 : index, 9 : i32, 3136 : i32, -1 : i32, 0 : i32, 18 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=17, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=2, d0_stride=4, d0_wrap=16, d1_stride=64, d1_wrap=49, iter_step_size=3136, iter_wrap=16 */"
    %118 = emitc.call_opaque "__runtime_buffer_arg"(%34) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %119 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %16, %118) {args = [0 : index, 1 : index, 2 : index, 8 : i32, 3136 : i32, -1 : i32, 0 : i32, 17 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 64 : i32, 49 : i32, 0 : i32, 0 : i32, 3136 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %16) {args = [0 : index, 1 : index, 1 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">) -> ()
    %120 = emitc.call_opaque "__Runtime_dma_createio_4"(%16, %119) {args = [0 : index, 1 : index, 1 : i32, 8 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %121 = emitc.call_opaque "__Runtime_startio"(%arg0, %66) {args = [0 : index, 1 : index, 1 : i32, 64 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %122 = emitc.call_opaque "__Runtime_startio"(%arg0, %84) {args = [0 : index, 1 : index, 2 : i32, 64 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %123 = emitc.call_opaque "__Runtime_startio"(%arg0, %102) {args = [0 : index, 1 : index, 2 : i32, 64 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %124 = emitc.call_opaque "__Runtime_startio"(%arg0, %120) {args = [0 : index, 1 : index, 3 : i32, 64 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %125 = emitc.call_opaque "__Runtime_startio"(%arg0, %39) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %126 = emitc.call_opaque "__Runtime_startio"(%arg0, %42) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %127 = emitc.call_opaque "__Runtime_startio"(%arg0, %45) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %128 = emitc.call_opaque "__Runtime_startio"(%arg0, %48) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %129 = emitc.call_opaque "__Runtime_startio"(%arg0, %51) {args = [0 : index, 1 : index, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %130 = emitc.call_opaque "__Runtime_startio"(%arg0, %69) {args = [0 : index, 1 : index, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %131 = emitc.call_opaque "__Runtime_startio"(%arg0, %87) {args = [0 : index, 1 : index, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %132 = emitc.call_opaque "__Runtime_startio"(%arg0, %105) {args = [0 : index, 1 : index, 3 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* Wait for 5 event(s) */"
    emitc.verbatim "__Runtime_ctrl_shim_bd_commit();"
    emitc.call_opaque "__Runtime_wait"(%arg0, %36) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"event">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %121) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %122) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %123) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%arg0, %124) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
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
