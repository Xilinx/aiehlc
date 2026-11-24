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
    %0 = "emitc.constant"() <{value = 50176 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 48 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = 0 : index}> : () -> index
    %5 = "emitc.constant"() <{value = 4 : index}> : () -> index
    %6 = "emitc.constant"() <{value = 1 : index}> : () -> index
    emitc.verbatim "XAie_DevInst* dev = v1;"
    %7 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %8 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %9 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,3) */"
    %10 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,4) */"
    %11 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,5) */"
    %12 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,6) */"
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,0) */"
    %13 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 3136]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %14 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %15 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %16 = emitc.call_opaque "__runtime_buffer_offset"(%13) {args = [0 : index, 3136]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,3) */"
    %17 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %18 = emitc.call_opaque "__runtime_buffer_offset"(%13) {args = [0 : index, 3136]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,4) */"
    %19 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %20 = emitc.call_opaque "__runtime_buffer_offset"(%13) {args = [0 : index, 3136]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,5) */"
    %21 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %22 = emitc.call_opaque "__runtime_buffer_offset"(%13) {args = [0 : index, 3136]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,6) */"
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,0) */"
    %23 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 6272]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %24 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %25 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %26 = emitc.call_opaque "__runtime_buffer_offset"(%23) {args = [0 : index, 6272]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,3) */"
    %27 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %28 = emitc.call_opaque "__runtime_buffer_offset"(%23) {args = [0 : index, 6272]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,4) */"
    %29 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %30 = emitc.call_opaque "__runtime_buffer_offset"(%23) {args = [0 : index, 6272]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,5) */"
    %31 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %32 = emitc.call_opaque "__runtime_buffer_offset"(%23) {args = [0 : index, 6272]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,6) */"
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,0) */"
    %33 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 9408]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %34 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %35 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %36 = emitc.call_opaque "__runtime_buffer_offset"(%33) {args = [0 : index, 9408]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,3) */"
    %37 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %38 = emitc.call_opaque "__runtime_buffer_offset"(%33) {args = [0 : index, 9408]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,4) */"
    %39 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %40 = emitc.call_opaque "__runtime_buffer_offset"(%33) {args = [0 : index, 9408]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,5) */"
    %41 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %42 = emitc.call_opaque "__runtime_buffer_offset"(%33) {args = [0 : index, 9408]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,6) */"
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,0) */"
    %43 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,3) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,3) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,3) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,3) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,0) */"
    %44 = emitc.call_opaque "__runtime_buffer_offset"(%arg3) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %45 = emitc.call_opaque "__runtime_buffer_offset"(%44) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,3) */"
    %46 = emitc.call_opaque "__runtime_buffer_offset"(%44) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,3) */"
    %47 = emitc.call_opaque "__runtime_buffer_offset"(%44) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,3) */"
    %48 = emitc.call_opaque "__runtime_buffer_offset"(%44) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,3) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,0) */"
    %49 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 51520]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %50 = emitc.call_opaque "__runtime_buffer_offset"(%49) {args = [0 : index, 51520]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,4) */"
    %51 = emitc.call_opaque "__runtime_buffer_offset"(%49) {args = [0 : index, 51520]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,4) */"
    %52 = emitc.call_opaque "__runtime_buffer_offset"(%49) {args = [0 : index, 51520]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,4) */"
    %53 = emitc.call_opaque "__runtime_buffer_offset"(%49) {args = [0 : index, 51520]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,4) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,0) */"
    %54 = emitc.call_opaque "__runtime_buffer_offset"(%arg3) {args = [0 : index, 200704]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %55 = emitc.call_opaque "__runtime_buffer_offset"(%54) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,4) */"
    %56 = emitc.call_opaque "__runtime_buffer_offset"(%54) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,4) */"
    %57 = emitc.call_opaque "__runtime_buffer_offset"(%54) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,4) */"
    %58 = emitc.call_opaque "__runtime_buffer_offset"(%54) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,4) */"
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,0) */"
    %59 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 103040]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %60 = emitc.call_opaque "__runtime_buffer_offset"(%59) {args = [0 : index, 103040]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,5) */"
    %61 = emitc.call_opaque "__runtime_buffer_offset"(%59) {args = [0 : index, 103040]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,5) */"
    %62 = emitc.call_opaque "__runtime_buffer_offset"(%59) {args = [0 : index, 103040]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,5) */"
    %63 = emitc.call_opaque "__runtime_buffer_offset"(%59) {args = [0 : index, 103040]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,5) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,0) */"
    %64 = emitc.call_opaque "__runtime_buffer_offset"(%arg3) {args = [0 : index, 401408]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %65 = emitc.call_opaque "__runtime_buffer_offset"(%64) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,5) */"
    %66 = emitc.call_opaque "__runtime_buffer_offset"(%64) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,5) */"
    %67 = emitc.call_opaque "__runtime_buffer_offset"(%64) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,5) */"
    %68 = emitc.call_opaque "__runtime_buffer_offset"(%64) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,5) */"
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,0) */"
    %69 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 154560]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %70 = emitc.call_opaque "__runtime_buffer_offset"(%69) {args = [0 : index, 154560]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,6) */"
    %71 = emitc.call_opaque "__runtime_buffer_offset"(%69) {args = [0 : index, 154560]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,6) */"
    %72 = emitc.call_opaque "__runtime_buffer_offset"(%69) {args = [0 : index, 154560]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,6) */"
    %73 = emitc.call_opaque "__runtime_buffer_offset"(%69) {args = [0 : index, 154560]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,6) */"
    emitc.verbatim "/* Allocated BD ID 3 for tile (3,0) */"
    %74 = emitc.call_opaque "__runtime_buffer_offset"(%arg3) {args = [0 : index, 602112]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %75 = emitc.call_opaque "__runtime_buffer_offset"(%74) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,6) */"
    %76 = emitc.call_opaque "__runtime_buffer_offset"(%74) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,6) */"
    %77 = emitc.call_opaque "__runtime_buffer_offset"(%74) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,6) */"
    %78 = emitc.call_opaque "__runtime_buffer_offset"(%74) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,6) */"
    emitc.verbatim "/* Allocated BD ID 3 for tile (2,0) */"
    emitc.verbatim "/* Load Kernel Group: 16 tile(s) */"
    %79 = emitc.call_opaque "__Runtime_load_kernel_group_16t"(%arg0, %9, %10, %11, %12, %15, %17, %19, %21, %25, %27, %29, %31, %35, %37, %39, %41) {args = [0 : index, 1 : index, 2 : index, 3 : index, 4 : index, 5 : index, 6 : index, 7 : index, 8 : index, 9 : index, 10 : index, 11 : index, 12 : index, 13 : index, 14 : index, 15 : index, 16 : index, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">) -> !emitc.opaque<"kernel_group">
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
    %80 = emitc.call_opaque "__Runtime_launch_kernel_group"(%arg0, %79) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"kernel_group">) -> !emitc.opaque<"event">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %81 = emitc.call_opaque "__runtime_buffer_arg"(%7) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %82 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %8, %81) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(0,0), direction=MM2S */"
    %83 = emitc.call_opaque "__Runtime_dma_createio_4"(%8, %82) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %84 = emitc.call_opaque "__Runtime_startio"(%arg0, %83) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %85 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %86 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %9, %85) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %87 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %88 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %9, %87) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,3), direction=S2MM */"
    %89 = emitc.call_opaque "__Runtime_dma_createio_4"(%9, %88) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %90 = emitc.call_opaque "__Runtime_startio"(%arg0, %89) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %91 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %92 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %10, %91) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %93 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %94 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %10, %93) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,4), direction=S2MM */"
    %95 = emitc.call_opaque "__Runtime_dma_createio_4"(%10, %94) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %96 = emitc.call_opaque "__Runtime_startio"(%arg0, %95) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %97 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %98 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %11, %97) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %99 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %100 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %11, %99) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,5), direction=S2MM */"
    %101 = emitc.call_opaque "__Runtime_dma_createio_4"(%11, %100) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %102 = emitc.call_opaque "__Runtime_startio"(%arg0, %101) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %103 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %104 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %12, %103) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %105 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %106 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %12, %105) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,6), direction=S2MM */"
    %107 = emitc.call_opaque "__Runtime_dma_createio_4"(%12, %106) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %108 = emitc.call_opaque "__Runtime_startio"(%arg0, %107) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %109 = emitc.call_opaque "__runtime_buffer_arg"(%13) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %110 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %14, %109) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(1,0), direction=MM2S */"
    %111 = emitc.call_opaque "__Runtime_dma_createio_4"(%14, %110) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %112 = emitc.call_opaque "__Runtime_startio"(%arg0, %111) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %113 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %114 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %15, %113) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %115 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %116 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %15, %115) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,3), direction=S2MM */"
    %117 = emitc.call_opaque "__Runtime_dma_createio_4"(%15, %116) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %118 = emitc.call_opaque "__Runtime_startio"(%arg0, %117) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %119 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %120 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %17, %119) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %121 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %122 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %17, %121) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,4), direction=S2MM */"
    %123 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %122) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %124 = emitc.call_opaque "__Runtime_startio"(%arg0, %123) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %125 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %126 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %19, %125) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %127 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %128 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %19, %127) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,5), direction=S2MM */"
    %129 = emitc.call_opaque "__Runtime_dma_createio_4"(%19, %128) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %130 = emitc.call_opaque "__Runtime_startio"(%arg0, %129) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %131 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %132 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %21, %131) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %133 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %134 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %21, %133) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,6), direction=S2MM */"
    %135 = emitc.call_opaque "__Runtime_dma_createio_4"(%21, %134) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %136 = emitc.call_opaque "__Runtime_startio"(%arg0, %135) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %137 = emitc.call_opaque "__runtime_buffer_arg"(%23) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %138 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %24, %137) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */"
    %139 = emitc.call_opaque "__Runtime_dma_createio_4"(%24, %138) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %140 = emitc.call_opaque "__Runtime_startio"(%arg0, %139) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %141 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %142 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %25, %141) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %143 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %144 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %25, %143) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,3), direction=S2MM */"
    %145 = emitc.call_opaque "__Runtime_dma_createio_4"(%25, %144) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %146 = emitc.call_opaque "__Runtime_startio"(%arg0, %145) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %147 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %148 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %27, %147) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %149 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %150 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %27, %149) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,4), direction=S2MM */"
    %151 = emitc.call_opaque "__Runtime_dma_createio_4"(%27, %150) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %152 = emitc.call_opaque "__Runtime_startio"(%arg0, %151) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %153 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %154 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %153) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %155 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %156 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %155) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,5), direction=S2MM */"
    %157 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %156) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %158 = emitc.call_opaque "__Runtime_startio"(%arg0, %157) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %159 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %160 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %31, %159) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %161 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %162 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %31, %161) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,6), direction=S2MM */"
    %163 = emitc.call_opaque "__Runtime_dma_createio_4"(%31, %162) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %164 = emitc.call_opaque "__Runtime_startio"(%arg0, %163) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %165 = emitc.call_opaque "__runtime_buffer_arg"(%33) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %166 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %34, %165) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */"
    %167 = emitc.call_opaque "__Runtime_dma_createio_4"(%34, %166) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %168 = emitc.call_opaque "__Runtime_startio"(%arg0, %167) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %169 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %170 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %169) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %171 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %172 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %171) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,3), direction=S2MM */"
    %173 = emitc.call_opaque "__Runtime_dma_createio_4"(%35, %172) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %174 = emitc.call_opaque "__Runtime_startio"(%arg0, %173) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %175 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %176 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %37, %175) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %177 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %178 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %37, %177) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,4), direction=S2MM */"
    %179 = emitc.call_opaque "__Runtime_dma_createio_4"(%37, %178) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %180 = emitc.call_opaque "__Runtime_startio"(%arg0, %179) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %181 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %182 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %39, %181) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %183 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %184 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %39, %183) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,5), direction=S2MM */"
    %185 = emitc.call_opaque "__Runtime_dma_createio_4"(%39, %184) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %186 = emitc.call_opaque "__Runtime_startio"(%arg0, %185) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=3136, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %187 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)35904">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %188 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %187) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 3136 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=0, len=3136, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %189 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %190 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %189) {args = [0 : index, 1 : index, 2 : index, 0 : i32, 3136 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,6), direction=S2MM */"
    %191 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %190) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %192 = emitc.call_opaque "__Runtime_startio"(%arg0, %191) {args = [0 : index, 1 : index, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %193 = emitc.call_opaque "__runtime_buffer_arg"(%43) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %194 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %8, %193) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 18544 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 61 : i32, 920 : i32, 19 : i32, 224 : i32, 4 : i32, 12880 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(0,0), direction=MM2S */"
    %195 = emitc.call_opaque "__Runtime_dma_createio_4"(%8, %194) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %196 = emitc.call_opaque "__Runtime_startio"(%arg0, %195) {args = [0 : index, 1 : index, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %197 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %198 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %9, %197) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %199 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %200 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %9, %199) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,3), direction=S2MM */"
    %201 = emitc.call_opaque "__Runtime_dma_createio_4"(%9, %200) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %202 = emitc.call_opaque "__Runtime_startio"(%arg0, %201) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %203 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %204 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %15, %203) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %205 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %206 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %15, %205) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,3), direction=S2MM */"
    %207 = emitc.call_opaque "__Runtime_dma_createio_4"(%15, %206) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %208 = emitc.call_opaque "__Runtime_startio"(%arg0, %207) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %209 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %210 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %25, %209) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %211 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %212 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %25, %211) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,3), direction=S2MM */"
    %213 = emitc.call_opaque "__Runtime_dma_createio_4"(%25, %212) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %214 = emitc.call_opaque "__Runtime_startio"(%arg0, %213) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %215 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %216 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %215) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %217 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %218 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %217) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,3), direction=S2MM */"
    %219 = emitc.call_opaque "__Runtime_dma_createio_4"(%35, %218) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %220 = emitc.call_opaque "__Runtime_startio"(%arg0, %219) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=1, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */"
    %221 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %222 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %9, %221) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 1 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 2 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=1, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */"
    %223 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %224 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %9, %223) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 1 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 2 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,3), direction=MM2S */"
    %225 = emitc.call_opaque "__Runtime_dma_createio_4"(%9, %224) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %226 = emitc.call_opaque "__Runtime_startio"(%arg0, %225) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=2, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %227 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %228 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %15, %227) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 2 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 3 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=2, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %229 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %230 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %15, %229) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 2 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 3 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,3), direction=MM2S */"
    %231 = emitc.call_opaque "__Runtime_dma_createio_4"(%15, %230) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %232 = emitc.call_opaque "__Runtime_startio"(%arg0, %231) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=3, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %233 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %234 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %25, %233) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 3 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=3, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %235 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %236 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %25, %235) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 3 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,3), direction=MM2S */"
    %237 = emitc.call_opaque "__Runtime_dma_createio_4"(%25, %236) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %238 = emitc.call_opaque "__Runtime_startio"(%arg0, %237) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=4, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %239 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %240 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %239) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 4 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 5 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=4, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %241 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %242 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %35, %241) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 4 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 5 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,3), direction=MM2S */"
    %243 = emitc.call_opaque "__Runtime_dma_createio_4"(%35, %242) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %244 = emitc.call_opaque "__Runtime_startio"(%arg0, %243) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %245 = emitc.call_opaque "__runtime_buffer_arg"(%49) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %246 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %14, %245) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 18544 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 61 : i32, 920 : i32, 19 : i32, 224 : i32, 4 : i32, 12880 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(1,0), direction=MM2S */"
    %247 = emitc.call_opaque "__Runtime_dma_createio_4"(%14, %246) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %248 = emitc.call_opaque "__Runtime_startio"(%arg0, %247) {args = [0 : index, 1 : index, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %249 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %250 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %10, %249) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %251 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %252 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %10, %251) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,4), direction=S2MM */"
    %253 = emitc.call_opaque "__Runtime_dma_createio_4"(%10, %252) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %254 = emitc.call_opaque "__Runtime_startio"(%arg0, %253) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %255 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %256 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %17, %255) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %257 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %258 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %17, %257) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,4), direction=S2MM */"
    %259 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %258) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %260 = emitc.call_opaque "__Runtime_startio"(%arg0, %259) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %261 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %262 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %27, %261) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %263 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %264 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %27, %263) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,4), direction=S2MM */"
    %265 = emitc.call_opaque "__Runtime_dma_createio_4"(%27, %264) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %266 = emitc.call_opaque "__Runtime_startio"(%arg0, %265) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %267 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %268 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %37, %267) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %269 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %270 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %37, %269) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,4), direction=S2MM */"
    %271 = emitc.call_opaque "__Runtime_dma_createio_4"(%37, %270) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %272 = emitc.call_opaque "__Runtime_startio"(%arg0, %271) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=5, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */"
    %273 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %274 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %10, %273) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 5 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 7 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=5, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */"
    %275 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %276 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %10, %275) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 5 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 7 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,4), direction=MM2S */"
    %277 = emitc.call_opaque "__Runtime_dma_createio_4"(%10, %276) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %278 = emitc.call_opaque "__Runtime_startio"(%arg0, %277) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=6, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %279 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %280 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %17, %279) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 6 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 8 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=6, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %281 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %282 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %17, %281) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 6 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 8 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,4), direction=MM2S */"
    %283 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %282) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %284 = emitc.call_opaque "__Runtime_startio"(%arg0, %283) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=7, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %285 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %286 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %27, %285) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 7 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 9 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=7, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %287 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %288 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %27, %287) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 7 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 9 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,4), direction=MM2S */"
    %289 = emitc.call_opaque "__Runtime_dma_createio_4"(%27, %288) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %290 = emitc.call_opaque "__Runtime_startio"(%arg0, %289) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=8, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %291 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %292 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %37, %291) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 8 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 10 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=8, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %293 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %294 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %37, %293) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 8 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 10 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,4), direction=MM2S */"
    %295 = emitc.call_opaque "__Runtime_dma_createio_4"(%37, %294) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %296 = emitc.call_opaque "__Runtime_startio"(%arg0, %295) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=1, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %297 = emitc.call_opaque "__runtime_buffer_arg"(%59) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %298 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %297) {args = [0 : index, 1 : index, 2 : index, 1 : i32, 18544 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 61 : i32, 920 : i32, 19 : i32, 224 : i32, 4 : i32, 12880 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */"
    %299 = emitc.call_opaque "__Runtime_dma_createio_4"(%24, %298) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %300 = emitc.call_opaque "__Runtime_startio"(%arg0, %299) {args = [0 : index, 1 : index, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %301 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %302 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %11, %301) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %303 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %304 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %11, %303) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,5), direction=S2MM */"
    %305 = emitc.call_opaque "__Runtime_dma_createio_4"(%11, %304) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %306 = emitc.call_opaque "__Runtime_startio"(%arg0, %305) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %307 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %308 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %19, %307) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %309 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %310 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %19, %309) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,5), direction=S2MM */"
    %311 = emitc.call_opaque "__Runtime_dma_createio_4"(%19, %310) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %312 = emitc.call_opaque "__Runtime_startio"(%arg0, %311) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %313 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %314 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %313) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %315 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %316 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %315) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,5), direction=S2MM */"
    %317 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %316) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %318 = emitc.call_opaque "__Runtime_startio"(%arg0, %317) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %319 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %320 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %39, %319) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %321 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %322 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %39, %321) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,5), direction=S2MM */"
    %323 = emitc.call_opaque "__Runtime_dma_createio_4"(%39, %322) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %324 = emitc.call_opaque "__Runtime_startio"(%arg0, %323) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=9, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %325 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %326 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %11, %325) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 9 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 3 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=9, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %327 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %328 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %11, %327) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 9 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 3 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,5), direction=MM2S */"
    %329 = emitc.call_opaque "__Runtime_dma_createio_4"(%11, %328) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %330 = emitc.call_opaque "__Runtime_startio"(%arg0, %329) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=10, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %331 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %332 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %19, %331) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 10 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=10, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %333 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %334 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %19, %333) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 10 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,5), direction=MM2S */"
    %335 = emitc.call_opaque "__Runtime_dma_createio_4"(%19, %334) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %336 = emitc.call_opaque "__Runtime_startio"(%arg0, %335) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=11, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %337 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %338 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %337) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 11 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 5 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=11, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %339 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %340 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %29, %339) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 11 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 5 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,5), direction=MM2S */"
    %341 = emitc.call_opaque "__Runtime_dma_createio_4"(%29, %340) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %342 = emitc.call_opaque "__Runtime_startio"(%arg0, %341) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=12, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %343 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %344 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %39, %343) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 12 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 6 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=12, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %345 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %346 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %39, %345) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 12 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 6 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,5), direction=MM2S */"
    %347 = emitc.call_opaque "__Runtime_dma_createio_4"(%39, %346) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %348 = emitc.call_opaque "__Runtime_startio"(%arg0, %347) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=11, len=18544, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=61, d1_stride=920, d1_wrap=19, d2_stride=224, d2_wrap=4, iter_step_size=12880, iter_wrap=4 */"
    %349 = emitc.call_opaque "__runtime_buffer_arg"(%69) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %350 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %349) {args = [0 : index, 1 : index, 2 : index, 11 : i32, 18544 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 61 : i32, 920 : i32, 19 : i32, 224 : i32, 4 : i32, 12880 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=11, tile=(3,0), direction=MM2S */"
    %351 = emitc.call_opaque "__Runtime_dma_createio_4"(%34, %350) {args = [0 : index, 1 : index, 1 : i32, 11 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %352 = emitc.call_opaque "__Runtime_startio"(%arg0, %351) {args = [0 : index, 1 : index, 3 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %353 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %354 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %12, %353) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %355 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %356 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %12, %355) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,6), direction=S2MM */"
    %357 = emitc.call_opaque "__Runtime_dma_createio_4"(%12, %356) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %358 = emitc.call_opaque "__Runtime_startio"(%arg0, %357) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %359 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %360 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %21, %359) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %361 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %362 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %21, %361) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,6), direction=S2MM */"
    %363 = emitc.call_opaque "__Runtime_dma_createio_4"(%21, %362) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %364 = emitc.call_opaque "__Runtime_startio"(%arg0, %363) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %365 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %366 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %31, %365) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %367 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %368 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %31, %367) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,6), direction=S2MM */"
    %369 = emitc.call_opaque "__Runtime_dma_createio_4"(%31, %368) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %370 = emitc.call_opaque "__Runtime_startio"(%arg0, %369) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=3, len=4636, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %371 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)43680">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %372 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %371) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 4636 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=2, len=4636, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %373 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)39040">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %374 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %373) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 4636 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,6), direction=S2MM */"
    %375 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %374) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %376 = emitc.call_opaque "__Runtime_startio"(%arg0, %375) {args = [0 : index, 1 : index, 1 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=13, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %377 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %378 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %12, %377) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 13 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 8 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=13, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %379 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %380 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %12, %379) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 13 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 8 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,6), direction=MM2S */"
    %381 = emitc.call_opaque "__Runtime_dma_createio_4"(%12, %380) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %382 = emitc.call_opaque "__Runtime_startio"(%arg0, %381) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=14, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %383 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %384 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %21, %383) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 14 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 9 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=14, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %385 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %386 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %21, %385) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 14 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 9 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,6), direction=MM2S */"
    %387 = emitc.call_opaque "__Runtime_dma_createio_4"(%21, %386) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %388 = emitc.call_opaque "__Runtime_startio"(%arg0, %387) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=15, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %389 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %390 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %31, %389) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 15 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 10 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=15, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %391 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %392 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %31, %391) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 15 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 10 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,6), direction=MM2S */"
    %393 = emitc.call_opaque "__Runtime_dma_createio_4"(%31, %392) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %394 = emitc.call_opaque "__Runtime_startio"(%arg0, %393) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=true, packet_id=16, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %395 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)51456">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %396 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %395) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, 4 : i32, 1 : i32, 16 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 11 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=true, packet_id=16, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %397 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)48320">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %398 = emitc.call_opaque "__Runtime_dma_bd_config"(%arg0, %41, %397) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, 5 : i32, 1 : i32, 16 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 11 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,6), direction=MM2S */"
    %399 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %398) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %400 = emitc.call_opaque "__Runtime_startio"(%arg0, %399) {args = [0 : index, 1 : index, 2 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.for %arg4 = %4 to %5 step %6 {
      %401 = emitc.cast %arg4 : index to i32
      %402 = emitc.mul %401, %0 : (i32, i32) -> i32
      %403 = emitc.add %402, %1 : (i32, i32) -> i32
      %404 = emitc.add %402, %2 : (i32, i32) -> i32
      %405 = emitc.add %402, %3 : (i32, i32) -> i32
      emitc.verbatim "/* Allocated BD ID 4 for tile (3,0) */"
      emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %406 = emitc.call_opaque "__runtime_buffer_arg"(%44) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %407 = emitc.cast %403 : i32 to i64
      %408 = emitc.call_opaque "__runtime_buffer_offset"(%406, %407) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %409 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %408) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, -1 : i32, 0 : i32, 4 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=3, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %410 = emitc.call_opaque "__runtime_buffer_arg"(%44) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %411 = emitc.cast %404 : i32 to i64
      %412 = emitc.call_opaque "__runtime_buffer_offset"(%410, %411) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %413 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %412) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, -1 : i32, 0 : i32, 3 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=2, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %414 = emitc.call_opaque "__runtime_buffer_arg"(%44) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %415 = emitc.cast %405 : i32 to i64
      %416 = emitc.call_opaque "__runtime_buffer_offset"(%414, %415) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %417 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %416) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 3136 : i32, -1 : i32, 0 : i32, 2 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=2, len=3136, enable_packet=false, packet_id=1, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %418 = emitc.call_opaque "__runtime_buffer_arg"(%44) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %419 = emitc.cast %402 : i32 to i64
      %420 = emitc.call_opaque "__runtime_buffer_offset"(%418, %419) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %421 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %420) {args = [0 : index, 1 : index, 2 : index, 2 : i32, 3136 : i32, -1 : i32, 0 : i32, 1 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,0), direction=S2MM */"
      emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */"
      emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %34) {args = [0 : index, 1 : index, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">) -> ()
      %422 = emitc.call_opaque "__Runtime_dma_createio_4"(%34, %421) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
      %423 = emitc.call_opaque "__Runtime_startio"(%arg0, %422) {args = [0 : index, 1 : index, 4 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
      %424 = emitc.cast %arg4 : index to i32
      %425 = emitc.mul %424, %0 : (i32, i32) -> i32
      %426 = emitc.add %425, %1 : (i32, i32) -> i32
      %427 = emitc.add %425, %2 : (i32, i32) -> i32
      %428 = emitc.add %425, %3 : (i32, i32) -> i32
      emitc.verbatim "/* Allocated BD ID 5 for tile (3,0) */"
      emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %429 = emitc.call_opaque "__runtime_buffer_arg"(%54) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %430 = emitc.cast %426 : i32 to i64
      %431 = emitc.call_opaque "__runtime_buffer_offset"(%429, %430) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %432 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %431) {args = [0 : index, 1 : index, 2 : index, 10 : i32, 3136 : i32, -1 : i32, 0 : i32, 8 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %433 = emitc.call_opaque "__runtime_buffer_arg"(%54) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %434 = emitc.cast %427 : i32 to i64
      %435 = emitc.call_opaque "__runtime_buffer_offset"(%433, %434) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %436 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %435) {args = [0 : index, 1 : index, 2 : index, 9 : i32, 3136 : i32, -1 : i32, 0 : i32, 7 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %437 = emitc.call_opaque "__runtime_buffer_arg"(%54) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %438 = emitc.cast %428 : i32 to i64
      %439 = emitc.call_opaque "__runtime_buffer_offset"(%437, %438) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %440 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %439) {args = [0 : index, 1 : index, 2 : index, 8 : i32, 3136 : i32, -1 : i32, 0 : i32, 6 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=7, len=3136, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %441 = emitc.call_opaque "__runtime_buffer_arg"(%54) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %442 = emitc.cast %425 : i32 to i64
      %443 = emitc.call_opaque "__runtime_buffer_offset"(%441, %442) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %444 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %34, %443) {args = [0 : index, 1 : index, 2 : index, 7 : i32, 3136 : i32, -1 : i32, 0 : i32, 5 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* Create IO: channel_id=1, bd_id=7, tile=(3,0), direction=S2MM */"
      emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */"
      emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %34) {args = [0 : index, 1 : index, 1 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">) -> ()
      %445 = emitc.call_opaque "__Runtime_dma_createio_4"(%34, %444) {args = [0 : index, 1 : index, 1 : i32, 7 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
      %446 = emitc.call_opaque "__Runtime_startio"(%arg0, %445) {args = [0 : index, 1 : index, 5 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
      %447 = emitc.cast %arg4 : index to i32
      %448 = emitc.mul %447, %0 : (i32, i32) -> i32
      %449 = emitc.add %448, %1 : (i32, i32) -> i32
      %450 = emitc.add %448, %2 : (i32, i32) -> i32
      %451 = emitc.add %448, %3 : (i32, i32) -> i32
      emitc.verbatim "/* Allocated BD ID 4 for tile (2,0) */"
      emitc.verbatim "/* DMA BD Config: bd_id=6, len=3136, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %452 = emitc.call_opaque "__runtime_buffer_arg"(%64) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %453 = emitc.cast %449 : i32 to i64
      %454 = emitc.call_opaque "__runtime_buffer_offset"(%452, %453) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %455 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %454) {args = [0 : index, 1 : index, 2 : index, 6 : i32, 3136 : i32, -1 : i32, 0 : i32, 12 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=5, len=3136, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %456 = emitc.call_opaque "__runtime_buffer_arg"(%64) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %457 = emitc.cast %450 : i32 to i64
      %458 = emitc.call_opaque "__runtime_buffer_offset"(%456, %457) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %459 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %458) {args = [0 : index, 1 : index, 2 : index, 5 : i32, 3136 : i32, -1 : i32, 0 : i32, 11 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=4, len=3136, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %460 = emitc.call_opaque "__runtime_buffer_arg"(%64) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %461 = emitc.cast %451 : i32 to i64
      %462 = emitc.call_opaque "__runtime_buffer_offset"(%460, %461) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %463 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %462) {args = [0 : index, 1 : index, 2 : index, 4 : i32, 3136 : i32, -1 : i32, 0 : i32, 10 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=3, len=3136, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %464 = emitc.call_opaque "__runtime_buffer_arg"(%64) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %465 = emitc.cast %448 : i32 to i64
      %466 = emitc.call_opaque "__runtime_buffer_offset"(%464, %465) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %467 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %466) {args = [0 : index, 1 : index, 2 : index, 3 : i32, 3136 : i32, -1 : i32, 0 : i32, 9 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(2,0), direction=S2MM */"
      emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=0 dir=S2MM */"
      emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %24) {args = [0 : index, 1 : index, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">) -> ()
      %468 = emitc.call_opaque "__Runtime_dma_createio_4"(%24, %467) {args = [0 : index, 1 : index, 0 : i32, 3 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
      %469 = emitc.call_opaque "__Runtime_startio"(%arg0, %468) {args = [0 : index, 1 : index, 4 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
      %470 = emitc.cast %arg4 : index to i32
      %471 = emitc.mul %470, %0 : (i32, i32) -> i32
      %472 = emitc.add %471, %1 : (i32, i32) -> i32
      %473 = emitc.add %471, %2 : (i32, i32) -> i32
      %474 = emitc.add %471, %3 : (i32, i32) -> i32
      emitc.verbatim "/* Allocated BD ID 5 for tile (2,0) */"
      emitc.verbatim "/* DMA BD Config: bd_id=11, len=3136, enable_packet=false, packet_id=16, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %475 = emitc.call_opaque "__runtime_buffer_arg"(%74) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %476 = emitc.cast %472 : i32 to i64
      %477 = emitc.call_opaque "__runtime_buffer_offset"(%475, %476) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %478 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %477) {args = [0 : index, 1 : index, 2 : index, 11 : i32, 3136 : i32, -1 : i32, 0 : i32, 16 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=10, len=3136, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %479 = emitc.call_opaque "__runtime_buffer_arg"(%74) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %480 = emitc.cast %473 : i32 to i64
      %481 = emitc.call_opaque "__runtime_buffer_offset"(%479, %480) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %482 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %481) {args = [0 : index, 1 : index, 2 : index, 10 : i32, 3136 : i32, -1 : i32, 0 : i32, 15 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=9, len=3136, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %483 = emitc.call_opaque "__runtime_buffer_arg"(%74) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %484 = emitc.cast %474 : i32 to i64
      %485 = emitc.call_opaque "__runtime_buffer_offset"(%483, %484) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %486 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %485) {args = [0 : index, 1 : index, 2 : index, 9 : i32, 3136 : i32, -1 : i32, 0 : i32, 14 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* DMA BD Config: bd_id=8, len=3136, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1, num_dims=3, d0_stride=4, d0_wrap=4, d1_stride=64, d1_wrap=28, d2_stride=7168, d2_wrap=7, iter_step_size=1792, iter_wrap=4 */"
      %487 = emitc.call_opaque "__runtime_buffer_arg"(%74) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
      %488 = emitc.cast %471 : i32 to i64
      %489 = emitc.call_opaque "__runtime_buffer_offset"(%487, %488) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
      %490 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%arg0, %24, %489) {args = [0 : index, 1 : index, 2 : index, 8 : i32, 3136 : i32, -1 : i32, 0 : i32, 13 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 3 : i32, 4 : i32, 4 : i32, 64 : i32, 28 : i32, 7168 : i32, 7 : i32, 1792 : i32, 4 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
      emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(2,0), direction=S2MM */"
      emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=1 dir=S2MM */"
      emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%arg0, %24) {args = [0 : index, 1 : index, 1 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">) -> ()
      %491 = emitc.call_opaque "__Runtime_dma_createio_4"(%24, %490) {args = [0 : index, 1 : index, 1 : i32, 8 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
      %492 = emitc.call_opaque "__Runtime_startio"(%arg0, %491) {args = [0 : index, 1 : index, 5 : i32, 16 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
      emitc.verbatim "/* Wait for 1 event(s) */"
      emitc.call_opaque "__Runtime_wait"(%arg0, %423) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
      emitc.verbatim "/* Wait for 1 event(s) */"
      emitc.call_opaque "__Runtime_wait"(%arg0, %446) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
      emitc.verbatim "/* Wait for 1 event(s) */"
      emitc.call_opaque "__Runtime_wait"(%arg0, %469) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
      emitc.verbatim "/* Wait for 1 event(s) */"
      emitc.call_opaque "__Runtime_wait"(%arg0, %492) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"ioevent">) -> ()
    }
    emitc.verbatim "/* Wait for 1 event(s) */"
    emitc.call_opaque "__Runtime_wait"(%arg0, %80) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"event">) -> ()
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
