module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 1 : i64, routing.control_plan_dedicated_shim = 1 : i64, routing.control_plan_exclusive = 1 : i64, routing.control_plan_group_reg_write = 1 : i64, routing.control_plan_mm2s_ch = 0 : i64, routing.control_plan_op_control_packet = 1 : i64, routing.control_plan_s2mm_ch = 0 : i64, routing.control_plan_shim_bd_ctrl = 1 : i64, routing.control_plan_shim_col = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 1 : i64, routing.output_tile_major = 1 : i64} {
  dfschedule.module @kernel_driver_dskernel_receiver {dfschedule.core_offload_plan = [{bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 1 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 1 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 1 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 1 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 2 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 2 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 2 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 2 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 3 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 3 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 3 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 3 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 4 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 4 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 4 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 3136 : i32, channel = 1 : i32, col = 4 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 1 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 2 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 3 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 4 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 1 : i32, enable_packet = true, is_output = true, ooo_bd_id = 2 : i32, packet_id = 4 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 2 : i32, enable_packet = true, is_output = true, ooo_bd_id = 3 : i32, packet_id = 5 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 3 : i32, enable_packet = true, is_output = true, ooo_bd_id = 4 : i32, packet_id = 6 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 4 : i32, enable_packet = true, is_output = true, ooo_bd_id = 5 : i32, packet_id = 7 : i32, pp_depth = 2 : i32, row = 3 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 1 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 2 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 3 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 4 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 1 : i32, enable_packet = true, is_output = true, ooo_bd_id = 7 : i32, packet_id = 8 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 2 : i32, enable_packet = true, is_output = true, ooo_bd_id = 8 : i32, packet_id = 9 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 3 : i32, enable_packet = true, is_output = true, ooo_bd_id = 9 : i32, packet_id = 10 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 4 : i32, enable_packet = true, is_output = true, ooo_bd_id = 10 : i32, packet_id = 11 : i32, pp_depth = 2 : i32, row = 4 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 1 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 2 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 3 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 4 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 1 : i32, enable_packet = true, is_output = true, ooo_bd_id = 3 : i32, packet_id = 12 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 2 : i32, enable_packet = true, is_output = true, ooo_bd_id = 4 : i32, packet_id = 13 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 3 : i32, enable_packet = true, is_output = true, ooo_bd_id = 5 : i32, packet_id = 14 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 4 : i32, enable_packet = true, is_output = true, ooo_bd_id = 6 : i32, packet_id = 15 : i32, pp_depth = 2 : i32, row = 5 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 1 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 2 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 3 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 4636 : i32, channel = 0 : i32, col = 4 : i32, enable_packet = false, is_output = false, ooo_bd_id = -1 : i32, packet_id = 0 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 1 : i32, enable_packet = true, is_output = true, ooo_bd_id = 8 : i32, packet_id = 17 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 2 : i32, enable_packet = true, is_output = true, ooo_bd_id = 9 : i32, packet_id = 18 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 3 : i32, enable_packet = true, is_output = true, ooo_bd_id = 10 : i32, packet_id = 19 : i32, pp_depth = 2 : i32, row = 6 : i32}, {bd_len_bytes = 3136 : i32, channel = 0 : i32, col = 4 : i32, enable_packet = true, is_output = true, ooo_bd_id = 11 : i32, packet_id = 20 : i32, pp_depth = 2 : i32, row = 6 : i32}], dfschedule.kernel_config_offload = 1 : i64} {
    dfschedule.kernel_config_def @config {buffer_size = 1159 : i32, element_type = i8, kernel_file = "computekernel.cc", kernel_name = "computekernel", vector_width = 4 : i32}
    dfschedule.lock_def @LOCK_window_in_0_ACQ {id = 48 : i32, init_value = 2 : i32}
    dfschedule.lock_def @LOCK_window_in_0_REL {id = 49 : i32}
    dfschedule.buffer_def @buf_in_ping_0 : memref<1159xvector<4xi8>, "LOCAL"> {address = 497792 : i64}
    dfschedule.buffer_def @buf_in_pong_0 : memref<1159xvector<4xi8>, "LOCAL"> {address = 502432 : i64}
    dfschedule.window_def @window_in_0 {acquire_lock = @LOCK_window_in_0_ACQ, acquire_lock_hw_id = 0 : i32, async = true, buffer_size = 1159 : i32, direction = "in", dma_channel = 0 : i32, num_rounds = 4 : i32, ping_bd_id = 0 : i32, ping_buffer = @buf_in_ping_0, pong_bd_id = 1 : i32, pong_buffer = @buf_in_pong_0, release_lock = @LOCK_window_in_0_REL, release_lock_hw_id = 1 : i32, single_buffer = false}
    dfschedule.lock_def @LOCK_window_in_1_ACQ {id = 50 : i32, init_value = 2 : i32}
    dfschedule.lock_def @LOCK_window_in_1_REL {id = 51 : i32}
    dfschedule.buffer_def @buf_in_ping_1 : memref<784xvector<4xi8>, "LOCAL"> {address = 491520 : i64}
    dfschedule.buffer_def @buf_in_pong_1 : memref<784xvector<4xi8>, "LOCAL"> {address = 494656 : i64}
    dfschedule.window_def @window_in_1 {acquire_lock = @LOCK_window_in_1_ACQ, acquire_lock_hw_id = 2 : i32, async = true, buffer_size = 784 : i32, direction = "in", dma_channel = 1 : i32, num_rounds = 1 : i32, ping_bd_id = 2 : i32, ping_buffer = @buf_in_ping_1, pong_bd_id = 3 : i32, pong_buffer = @buf_in_pong_1, release_lock = @LOCK_window_in_1_REL, release_lock_hw_id = 3 : i32, single_buffer = false}
    dfschedule.lock_def @LOCK_window_out_0_ACQ {id = 52 : i32, init_value = 0 : i32}
    dfschedule.lock_def @LOCK_window_out_0_REL {id = 53 : i32}
    dfschedule.buffer_def @buf_out_ping_0 : memref<196xvector<4xi8>, "LOCAL"> {address = 507072 : i64}
    dfschedule.buffer_def @buf_out_pong_0 : memref<196xvector<4xi8>, "LOCAL"> {address = 510208 : i64}
    dfschedule.window_def @window_out_0 {acquire_lock = @LOCK_window_out_0_ACQ, acquire_lock_hw_id = 4 : i32, async = true, buffer_size = 196 : i32, direction = "out", dma_channel = 0 : i32, num_rounds = 16 : i32, ping_bd_id = 4 : i32, ping_buffer = @buf_out_ping_0, pong_bd_id = 5 : i32, pong_buffer = @buf_out_pong_0, release_lock = @LOCK_window_out_0_REL, release_lock_hw_id = 5 : i32, single_buffer = false}
    dfschedule.kernel_decl @computekernel {inputs = [@window_in_0, @window_in_1], iteration_style = "internal", outputs = [@window_out_0]}
    %0 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
    %alloc = memref.alloc() : memref<16x7x7x4xi8>
    %1 = dfschedule.memref_mapping %alloc : (memref<16x7x7x4xi8>) -> memref<16x7x7x4xi8>
    %2 = dfschedule.bind_core_buffer(%1, %0) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %3 = dfschedule.bind_core_buffer(%1, %0) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_0 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %4 = dfschedule.config.dma_bd(%3, %0, %c1_i32, %c0_i32, %c0_i32_0, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c0_i32_1 = arith.constant 0 : i32
    %c0_i32_2 = arith.constant 0 : i32
    %c0_i32_3 = arith.constant 0 : i32
    %c-1_i32_4 = arith.constant -1 : i32
    %5 = dfschedule.config.dma_bd(%2, %0, %c0_i32_1, %c0_i32_2, %c0_i32_3, %c-1_i32_4, %4) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %6 = dfschedule.config.create_io(%5, %0) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %7 = dfschedule.schedule.getbdid(%0) : (!dfschedule.tile) -> i32
    %8 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
    %9 = dfschedule.schedule.getbdid(%8) : (!dfschedule.tile) -> i32
    %10 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
    %11 = dfschedule.schedule.getbdid(%10) : (!dfschedule.tile) -> i32
    %12 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
    %13 = dfschedule.schedule.getbdid(%12) : (!dfschedule.tile) -> i32
    %14 = dfschedule.schedule.start_io(%6, %7) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %15 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
    %16 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
    %17 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
    %18 = dfschedule.schedule.getbdid(%17) : (!dfschedule.tile) -> i32
    %19 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
    %20 = dfschedule.schedule.getbdid(%19) : (!dfschedule.tile) -> i32
    %21 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
    %22 = dfschedule.schedule.getbdid(%21) : (!dfschedule.tile) -> i32
    %23 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
    %24 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
    %25 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
    %26 = dfschedule.schedule.getbdid(%25) : (!dfschedule.tile) -> i32
    %27 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
    %28 = dfschedule.schedule.getbdid(%27) : (!dfschedule.tile) -> i32
    %29 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
    %30 = dfschedule.schedule.getbdid(%29) : (!dfschedule.tile) -> i32
    %31 = dfschedule.declaretile {col = 4 : i32, row = 3 : i32} : !dfschedule.tile
    %32 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
    %33 = dfschedule.declaretile {col = 4 : i32, row = 4 : i32} : !dfschedule.tile
    %34 = dfschedule.schedule.getbdid(%33) : (!dfschedule.tile) -> i32
    %35 = dfschedule.declaretile {col = 4 : i32, row = 5 : i32} : !dfschedule.tile
    %36 = dfschedule.schedule.getbdid(%35) : (!dfschedule.tile) -> i32
    %37 = dfschedule.declaretile {col = 4 : i32, row = 6 : i32} : !dfschedule.tile
    %38 = dfschedule.schedule.getbdid(%37) : (!dfschedule.tile) -> i32
    %39 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
    %alloc_5 = memref.alloc() : memref<61x230x4xi8>
    %40 = dfschedule.memref_mapping %alloc_5 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8>) -> memref<61x230x4xi8>
    %41 = dfschedule.bind_core_buffer(%40, %39) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %42 = dfschedule.bind_core_buffer(%40, %39) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %c3_i32 = arith.constant 3 : i32
    %c0_i32_6 = arith.constant 0 : i32
    %c0_i32_7 = arith.constant 0 : i32
    %c-1_i32_8 = arith.constant -1 : i32
    %43 = dfschedule.config.dma_bd(%42, %39, %c3_i32, %c0_i32_6, %c0_i32_7, %c-1_i32_8) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c2_i32 = arith.constant 2 : i32
    %c0_i32_9 = arith.constant 0 : i32
    %c0_i32_10 = arith.constant 0 : i32
    %c-1_i32_11 = arith.constant -1 : i32
    %44 = dfschedule.config.dma_bd(%41, %39, %c2_i32, %c0_i32_9, %c0_i32_10, %c-1_i32_11, %43) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %45 = dfschedule.config.create_io(%44, %39) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %46 = dfschedule.schedule.getbdid(%39) : (!dfschedule.tile) -> i32
    %47 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
    %48 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
    %49 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
    %50 = dfschedule.schedule.getbdid(%49) : (!dfschedule.tile) -> i32
    %51 = dfschedule.declaretile {col = 4 : i32, row = 3 : i32} : !dfschedule.tile
    %52 = dfschedule.schedule.getbdid(%51) : (!dfschedule.tile) -> i32
    %53 = dfschedule.schedule.start_io(%45, %46) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %54 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
    %alloc_12 = memref.alloc() : memref<7x28x16xi8>
    %55 = dfschedule.memref_mapping %alloc_12 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %56 = dfschedule.bind_core_buffer(%55, %54) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %57 = dfschedule.bind_core_buffer(%55, %54) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32 = arith.constant 5 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32_14 = arith.constant 2 : i32
    %58 = dfschedule.config.dma_bd(%57, %54, %c5_i32, %c0_i32_13, %c4_i32, %c2_i32_14) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_15 = arith.constant 4 : i32
    %c0_i32_16 = arith.constant 0 : i32
    %c4_i32_17 = arith.constant 4 : i32
    %c2_i32_18 = arith.constant 2 : i32
    %59 = dfschedule.config.dma_bd(%56, %54, %c4_i32_15, %c0_i32_16, %c4_i32_17, %c2_i32_18, %58) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %60 = dfschedule.config.create_io(%59, %54) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %61 = dfschedule.schedule.getbdid(%54) : (!dfschedule.tile) -> i32
    %62 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
    %alloc_19 = memref.alloc() : memref<7x28x16xi8>
    %63 = dfschedule.memref_mapping %alloc_19 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %64 = dfschedule.bind_core_buffer(%63, %62) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %65 = dfschedule.bind_core_buffer(%63, %62) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_20 = arith.constant 5 : i32
    %c0_i32_21 = arith.constant 0 : i32
    %c5_i32_22 = arith.constant 5 : i32
    %c3_i32_23 = arith.constant 3 : i32
    %66 = dfschedule.config.dma_bd(%65, %62, %c5_i32_20, %c0_i32_21, %c5_i32_22, %c3_i32_23) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_24 = arith.constant 4 : i32
    %c0_i32_25 = arith.constant 0 : i32
    %c5_i32_26 = arith.constant 5 : i32
    %c3_i32_27 = arith.constant 3 : i32
    %67 = dfschedule.config.dma_bd(%64, %62, %c4_i32_24, %c0_i32_25, %c5_i32_26, %c3_i32_27, %66) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %68 = dfschedule.config.create_io(%67, %62) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %69 = dfschedule.schedule.getbdid(%62) : (!dfschedule.tile) -> i32
    %70 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
    %alloc_28 = memref.alloc() : memref<7x28x16xi8>
    %71 = dfschedule.memref_mapping %alloc_28 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %72 = dfschedule.bind_core_buffer(%71, %70) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %73 = dfschedule.bind_core_buffer(%71, %70) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_29 = arith.constant 5 : i32
    %c0_i32_30 = arith.constant 0 : i32
    %c6_i32 = arith.constant 6 : i32
    %c4_i32_31 = arith.constant 4 : i32
    %74 = dfschedule.config.dma_bd(%73, %70, %c5_i32_29, %c0_i32_30, %c6_i32, %c4_i32_31) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_32 = arith.constant 4 : i32
    %c0_i32_33 = arith.constant 0 : i32
    %c6_i32_34 = arith.constant 6 : i32
    %c4_i32_35 = arith.constant 4 : i32
    %75 = dfschedule.config.dma_bd(%72, %70, %c4_i32_32, %c0_i32_33, %c6_i32_34, %c4_i32_35, %74) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %76 = dfschedule.config.create_io(%75, %70) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %77 = dfschedule.schedule.getbdid(%70) : (!dfschedule.tile) -> i32
    %78 = dfschedule.declaretile {col = 4 : i32, row = 3 : i32} : !dfschedule.tile
    %alloc_36 = memref.alloc() : memref<7x28x16xi8>
    %79 = dfschedule.memref_mapping %alloc_36 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %80 = dfschedule.bind_core_buffer(%79, %78) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %81 = dfschedule.bind_core_buffer(%79, %78) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_37 = arith.constant 5 : i32
    %c0_i32_38 = arith.constant 0 : i32
    %c7_i32 = arith.constant 7 : i32
    %c5_i32_39 = arith.constant 5 : i32
    %82 = dfschedule.config.dma_bd(%81, %78, %c5_i32_37, %c0_i32_38, %c7_i32, %c5_i32_39) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_40 = arith.constant 4 : i32
    %c0_i32_41 = arith.constant 0 : i32
    %c7_i32_42 = arith.constant 7 : i32
    %c5_i32_43 = arith.constant 5 : i32
    %83 = dfschedule.config.dma_bd(%80, %78, %c4_i32_40, %c0_i32_41, %c7_i32_42, %c5_i32_43, %82) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %84 = dfschedule.config.create_io(%83, %78) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %85 = dfschedule.schedule.getbdid(%78) : (!dfschedule.tile) -> i32
    %86 = dfschedule.schedule.start_io(%60, %61) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %87 = dfschedule.schedule.start_io(%68, %69) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %88 = dfschedule.schedule.start_io(%76, %77) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %89 = dfschedule.schedule.start_io(%84, %85) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %90 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
    %91 = dfschedule.schedule.getbdid(%90) : (!dfschedule.tile) -> i32
    %92 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
    %93 = dfschedule.schedule.getbdid(%92) : (!dfschedule.tile) -> i32
    %94 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
    %95 = dfschedule.schedule.getbdid(%94) : (!dfschedule.tile) -> i32
    %96 = dfschedule.declaretile {col = 4 : i32, row = 4 : i32} : !dfschedule.tile
    %97 = dfschedule.schedule.getbdid(%96) : (!dfschedule.tile) -> i32
    %98 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
    %alloc_44 = memref.alloc() : memref<7x28x16xi8>
    %99 = dfschedule.memref_mapping %alloc_44 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %100 = dfschedule.bind_core_buffer(%99, %98) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %101 = dfschedule.bind_core_buffer(%99, %98) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_45 = arith.constant 5 : i32
    %c0_i32_46 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c7_i32_47 = arith.constant 7 : i32
    %102 = dfschedule.config.dma_bd(%101, %98, %c5_i32_45, %c0_i32_46, %c8_i32, %c7_i32_47) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_48 = arith.constant 4 : i32
    %c0_i32_49 = arith.constant 0 : i32
    %c8_i32_50 = arith.constant 8 : i32
    %c7_i32_51 = arith.constant 7 : i32
    %103 = dfschedule.config.dma_bd(%100, %98, %c4_i32_48, %c0_i32_49, %c8_i32_50, %c7_i32_51, %102) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %104 = dfschedule.config.create_io(%103, %98) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %105 = dfschedule.schedule.getbdid(%98) : (!dfschedule.tile) -> i32
    %106 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
    %alloc_52 = memref.alloc() : memref<7x28x16xi8>
    %107 = dfschedule.memref_mapping %alloc_52 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %108 = dfschedule.bind_core_buffer(%107, %106) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %109 = dfschedule.bind_core_buffer(%107, %106) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_53 = arith.constant 5 : i32
    %c0_i32_54 = arith.constant 0 : i32
    %c9_i32 = arith.constant 9 : i32
    %c8_i32_55 = arith.constant 8 : i32
    %110 = dfschedule.config.dma_bd(%109, %106, %c5_i32_53, %c0_i32_54, %c9_i32, %c8_i32_55) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_56 = arith.constant 4 : i32
    %c0_i32_57 = arith.constant 0 : i32
    %c9_i32_58 = arith.constant 9 : i32
    %c8_i32_59 = arith.constant 8 : i32
    %111 = dfschedule.config.dma_bd(%108, %106, %c4_i32_56, %c0_i32_57, %c9_i32_58, %c8_i32_59, %110) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %112 = dfschedule.config.create_io(%111, %106) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %113 = dfschedule.schedule.getbdid(%106) : (!dfschedule.tile) -> i32
    %114 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
    %alloc_60 = memref.alloc() : memref<7x28x16xi8>
    %115 = dfschedule.memref_mapping %alloc_60 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %116 = dfschedule.bind_core_buffer(%115, %114) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %117 = dfschedule.bind_core_buffer(%115, %114) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_61 = arith.constant 5 : i32
    %c0_i32_62 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c9_i32_63 = arith.constant 9 : i32
    %118 = dfschedule.config.dma_bd(%117, %114, %c5_i32_61, %c0_i32_62, %c10_i32, %c9_i32_63) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_64 = arith.constant 4 : i32
    %c0_i32_65 = arith.constant 0 : i32
    %c10_i32_66 = arith.constant 10 : i32
    %c9_i32_67 = arith.constant 9 : i32
    %119 = dfschedule.config.dma_bd(%116, %114, %c4_i32_64, %c0_i32_65, %c10_i32_66, %c9_i32_67, %118) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %120 = dfschedule.config.create_io(%119, %114) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %121 = dfschedule.schedule.getbdid(%114) : (!dfschedule.tile) -> i32
    %122 = dfschedule.declaretile {col = 4 : i32, row = 4 : i32} : !dfschedule.tile
    %alloc_68 = memref.alloc() : memref<7x28x16xi8>
    %123 = dfschedule.memref_mapping %alloc_68 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %124 = dfschedule.bind_core_buffer(%123, %122) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %125 = dfschedule.bind_core_buffer(%123, %122) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_69 = arith.constant 5 : i32
    %c0_i32_70 = arith.constant 0 : i32
    %c11_i32 = arith.constant 11 : i32
    %c10_i32_71 = arith.constant 10 : i32
    %126 = dfschedule.config.dma_bd(%125, %122, %c5_i32_69, %c0_i32_70, %c11_i32, %c10_i32_71) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_72 = arith.constant 4 : i32
    %c0_i32_73 = arith.constant 0 : i32
    %c11_i32_74 = arith.constant 11 : i32
    %c10_i32_75 = arith.constant 10 : i32
    %127 = dfschedule.config.dma_bd(%124, %122, %c4_i32_72, %c0_i32_73, %c11_i32_74, %c10_i32_75, %126) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %128 = dfschedule.config.create_io(%127, %122) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %129 = dfschedule.schedule.getbdid(%122) : (!dfschedule.tile) -> i32
    %130 = dfschedule.schedule.start_io(%104, %105) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %131 = dfschedule.schedule.start_io(%112, %113) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %132 = dfschedule.schedule.start_io(%120, %121) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %133 = dfschedule.schedule.start_io(%128, %129) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %134 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
    %135 = dfschedule.schedule.getbdid(%134) : (!dfschedule.tile) -> i32
    %136 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
    %137 = dfschedule.schedule.getbdid(%136) : (!dfschedule.tile) -> i32
    %138 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
    %139 = dfschedule.schedule.getbdid(%138) : (!dfschedule.tile) -> i32
    %140 = dfschedule.declaretile {col = 4 : i32, row = 5 : i32} : !dfschedule.tile
    %141 = dfschedule.schedule.getbdid(%140) : (!dfschedule.tile) -> i32
    %142 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
    %alloc_76 = memref.alloc() : memref<7x28x16xi8>
    %143 = dfschedule.memref_mapping %alloc_76 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %144 = dfschedule.bind_core_buffer(%143, %142) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %145 = dfschedule.bind_core_buffer(%143, %142) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_77 = arith.constant 5 : i32
    %c0_i32_78 = arith.constant 0 : i32
    %c12_i32 = arith.constant 12 : i32
    %c3_i32_79 = arith.constant 3 : i32
    %146 = dfschedule.config.dma_bd(%145, %142, %c5_i32_77, %c0_i32_78, %c12_i32, %c3_i32_79) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_80 = arith.constant 4 : i32
    %c0_i32_81 = arith.constant 0 : i32
    %c12_i32_82 = arith.constant 12 : i32
    %c3_i32_83 = arith.constant 3 : i32
    %147 = dfschedule.config.dma_bd(%144, %142, %c4_i32_80, %c0_i32_81, %c12_i32_82, %c3_i32_83, %146) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %148 = dfschedule.config.create_io(%147, %142) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %149 = dfschedule.schedule.getbdid(%142) : (!dfschedule.tile) -> i32
    %150 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
    %alloc_84 = memref.alloc() : memref<7x28x16xi8>
    %151 = dfschedule.memref_mapping %alloc_84 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %152 = dfschedule.bind_core_buffer(%151, %150) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %153 = dfschedule.bind_core_buffer(%151, %150) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_85 = arith.constant 5 : i32
    %c0_i32_86 = arith.constant 0 : i32
    %c13_i32 = arith.constant 13 : i32
    %c4_i32_87 = arith.constant 4 : i32
    %154 = dfschedule.config.dma_bd(%153, %150, %c5_i32_85, %c0_i32_86, %c13_i32, %c4_i32_87) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_88 = arith.constant 4 : i32
    %c0_i32_89 = arith.constant 0 : i32
    %c13_i32_90 = arith.constant 13 : i32
    %c4_i32_91 = arith.constant 4 : i32
    %155 = dfschedule.config.dma_bd(%152, %150, %c4_i32_88, %c0_i32_89, %c13_i32_90, %c4_i32_91, %154) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %156 = dfschedule.config.create_io(%155, %150) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %157 = dfschedule.schedule.getbdid(%150) : (!dfschedule.tile) -> i32
    %158 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
    %alloc_92 = memref.alloc() : memref<7x28x16xi8>
    %159 = dfschedule.memref_mapping %alloc_92 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %160 = dfschedule.bind_core_buffer(%159, %158) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %161 = dfschedule.bind_core_buffer(%159, %158) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_93 = arith.constant 5 : i32
    %c0_i32_94 = arith.constant 0 : i32
    %c14_i32 = arith.constant 14 : i32
    %c5_i32_95 = arith.constant 5 : i32
    %162 = dfschedule.config.dma_bd(%161, %158, %c5_i32_93, %c0_i32_94, %c14_i32, %c5_i32_95) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_96 = arith.constant 4 : i32
    %c0_i32_97 = arith.constant 0 : i32
    %c14_i32_98 = arith.constant 14 : i32
    %c5_i32_99 = arith.constant 5 : i32
    %163 = dfschedule.config.dma_bd(%160, %158, %c4_i32_96, %c0_i32_97, %c14_i32_98, %c5_i32_99, %162) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %164 = dfschedule.config.create_io(%163, %158) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %165 = dfschedule.schedule.getbdid(%158) : (!dfschedule.tile) -> i32
    %166 = dfschedule.declaretile {col = 4 : i32, row = 5 : i32} : !dfschedule.tile
    %alloc_100 = memref.alloc() : memref<7x28x16xi8>
    %167 = dfschedule.memref_mapping %alloc_100 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %168 = dfschedule.bind_core_buffer(%167, %166) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %169 = dfschedule.bind_core_buffer(%167, %166) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_101 = arith.constant 5 : i32
    %c0_i32_102 = arith.constant 0 : i32
    %c15_i32 = arith.constant 15 : i32
    %c6_i32_103 = arith.constant 6 : i32
    %170 = dfschedule.config.dma_bd(%169, %166, %c5_i32_101, %c0_i32_102, %c15_i32, %c6_i32_103) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_104 = arith.constant 4 : i32
    %c0_i32_105 = arith.constant 0 : i32
    %c15_i32_106 = arith.constant 15 : i32
    %c6_i32_107 = arith.constant 6 : i32
    %171 = dfschedule.config.dma_bd(%168, %166, %c4_i32_104, %c0_i32_105, %c15_i32_106, %c6_i32_107, %170) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %172 = dfschedule.config.create_io(%171, %166) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %173 = dfschedule.schedule.getbdid(%166) : (!dfschedule.tile) -> i32
    %174 = dfschedule.schedule.start_io(%148, %149) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %175 = dfschedule.schedule.start_io(%156, %157) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %176 = dfschedule.schedule.start_io(%164, %165) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %177 = dfschedule.schedule.start_io(%172, %173) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %178 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
    %179 = dfschedule.schedule.getbdid(%178) : (!dfschedule.tile) -> i32
    %180 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
    %181 = dfschedule.schedule.getbdid(%180) : (!dfschedule.tile) -> i32
    %182 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
    %183 = dfschedule.schedule.getbdid(%182) : (!dfschedule.tile) -> i32
    %184 = dfschedule.declaretile {col = 4 : i32, row = 6 : i32} : !dfschedule.tile
    %185 = dfschedule.schedule.getbdid(%184) : (!dfschedule.tile) -> i32
    %186 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
    %alloc_108 = memref.alloc() : memref<7x28x16xi8>
    %187 = dfschedule.memref_mapping %alloc_108 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %188 = dfschedule.bind_core_buffer(%187, %186) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %189 = dfschedule.bind_core_buffer(%187, %186) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_109 = arith.constant 5 : i32
    %c0_i32_110 = arith.constant 0 : i32
    %c17_i32 = arith.constant 17 : i32
    %c8_i32_111 = arith.constant 8 : i32
    %190 = dfschedule.config.dma_bd(%189, %186, %c5_i32_109, %c0_i32_110, %c17_i32, %c8_i32_111) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_112 = arith.constant 4 : i32
    %c0_i32_113 = arith.constant 0 : i32
    %c17_i32_114 = arith.constant 17 : i32
    %c8_i32_115 = arith.constant 8 : i32
    %191 = dfschedule.config.dma_bd(%188, %186, %c4_i32_112, %c0_i32_113, %c17_i32_114, %c8_i32_115, %190) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %192 = dfschedule.config.create_io(%191, %186) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %193 = dfschedule.schedule.getbdid(%186) : (!dfschedule.tile) -> i32
    %194 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
    %alloc_116 = memref.alloc() : memref<7x28x16xi8>
    %195 = dfschedule.memref_mapping %alloc_116 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %196 = dfschedule.bind_core_buffer(%195, %194) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %197 = dfschedule.bind_core_buffer(%195, %194) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_117 = arith.constant 5 : i32
    %c0_i32_118 = arith.constant 0 : i32
    %c18_i32 = arith.constant 18 : i32
    %c9_i32_119 = arith.constant 9 : i32
    %198 = dfschedule.config.dma_bd(%197, %194, %c5_i32_117, %c0_i32_118, %c18_i32, %c9_i32_119) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_120 = arith.constant 4 : i32
    %c0_i32_121 = arith.constant 0 : i32
    %c18_i32_122 = arith.constant 18 : i32
    %c9_i32_123 = arith.constant 9 : i32
    %199 = dfschedule.config.dma_bd(%196, %194, %c4_i32_120, %c0_i32_121, %c18_i32_122, %c9_i32_123, %198) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %200 = dfschedule.config.create_io(%199, %194) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %201 = dfschedule.schedule.getbdid(%194) : (!dfschedule.tile) -> i32
    %202 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
    %alloc_124 = memref.alloc() : memref<7x28x16xi8>
    %203 = dfschedule.memref_mapping %alloc_124 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %204 = dfschedule.bind_core_buffer(%203, %202) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %205 = dfschedule.bind_core_buffer(%203, %202) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_125 = arith.constant 5 : i32
    %c0_i32_126 = arith.constant 0 : i32
    %c19_i32 = arith.constant 19 : i32
    %c10_i32_127 = arith.constant 10 : i32
    %206 = dfschedule.config.dma_bd(%205, %202, %c5_i32_125, %c0_i32_126, %c19_i32, %c10_i32_127) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_128 = arith.constant 4 : i32
    %c0_i32_129 = arith.constant 0 : i32
    %c19_i32_130 = arith.constant 19 : i32
    %c10_i32_131 = arith.constant 10 : i32
    %207 = dfschedule.config.dma_bd(%204, %202, %c4_i32_128, %c0_i32_129, %c19_i32_130, %c10_i32_131, %206) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %208 = dfschedule.config.create_io(%207, %202) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %209 = dfschedule.schedule.getbdid(%202) : (!dfschedule.tile) -> i32
    %210 = dfschedule.declaretile {col = 4 : i32, row = 6 : i32} : !dfschedule.tile
    %alloc_132 = memref.alloc() : memref<7x28x16xi8>
    %211 = dfschedule.memref_mapping %alloc_132 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8>) -> memref<7x28x16xi8>
    %212 = dfschedule.bind_core_buffer(%211, %210) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %213 = dfschedule.bind_core_buffer(%211, %210) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %c5_i32_133 = arith.constant 5 : i32
    %c0_i32_134 = arith.constant 0 : i32
    %c20_i32 = arith.constant 20 : i32
    %c11_i32_135 = arith.constant 11 : i32
    %214 = dfschedule.config.dma_bd(%213, %210, %c5_i32_133, %c0_i32_134, %c20_i32, %c11_i32_135) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %c4_i32_136 = arith.constant 4 : i32
    %c0_i32_137 = arith.constant 0 : i32
    %c20_i32_138 = arith.constant 20 : i32
    %c11_i32_139 = arith.constant 11 : i32
    %215 = dfschedule.config.dma_bd(%212, %210, %c4_i32_136, %c0_i32_137, %c20_i32_138, %c11_i32_139, %214) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %216 = dfschedule.config.create_io(%215, %210) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %217 = dfschedule.schedule.getbdid(%210) : (!dfschedule.tile) -> i32
    %218 = dfschedule.schedule.start_io(%192, %193) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %219 = dfschedule.schedule.start_io(%200, %201) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %220 = dfschedule.schedule.start_io(%208, %209) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %221 = dfschedule.schedule.start_io(%216, %217) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    dfschedule.main @main {
      %222 = dfschedule.alloc_sync_buffer {size = 8 : i32} : !dfschedule.sync_buffer
      %c0_i32_140 = arith.constant 0 : i32
      dfschedule.sync_buffer_write(%222, %c0_i32_140) {index = 0 : i32}
      %c1_i32_141 = arith.constant 1 : i32
      dfschedule.log(%c1_i32_141)
      %223 = dfschedule.window_init(@window_in_0) : !dfschedule.input_window<i8>
      %224 = dfschedule.window_init(@window_in_1) : !dfschedule.input_window<i8>
      %225 = dfschedule.window_init(@window_out_0) : !dfschedule.output_window<i8>
      dfschedule.kernel_invoke @computekernel(%223, %224, %225) : (!dfschedule.input_window<i8>, !dfschedule.input_window<i8>, !dfschedule.output_window<i8>)
      dfschedule.done
      dfschedule.kernel_return
    }
  }
}
