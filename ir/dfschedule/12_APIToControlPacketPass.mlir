module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], dfschedule.control_kernel_ops = 1 : i64, dfschedule.grouped_lock_inits = [], routing.control_packet_aot = 1 : i64, routing.control_packet_mode = 1 : i64, routing.control_packet_nresp = 16 : i64, routing.control_packet_stream_id = 16 : i64, routing.control_plan_dedicated_shim = 1 : i64, routing.control_plan_exclusive = 1 : i64, routing.control_plan_group_reg_write = 1 : i64, routing.control_plan_mm2s_ch = 0 : i64, routing.control_plan_op_control_packet = 1 : i64, routing.control_plan_s2mm_ch = 0 : i64, routing.control_plan_shim_bd_ctrl = 1 : i64, routing.control_plan_shim_col = 0 : i64, routing.ctrlpkt_plan = [0, 1, 0, 4, 3, 1, 4, 4, 1, 4, 5, 1, 4, 6, 1, 4], routing.ctrlpkt_writes = [], routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 1 : i64, routing.output_tile_major = 1 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) attributes {routing.kernel_tiling = #routing.kernel_tiling<out_rounds = 16>} {
    dfschedule.launchhost @host_canonicalized
    return
  }
  dfschedule.host @host_canonicalized {
    %c17_i32 = arith.constant 17 : i32
    %c18_i32 = arith.constant 18 : i32
    %c19_i32 = arith.constant 19 : i32
    %c20_i32 = arith.constant 20 : i32
    %c12_i32 = arith.constant 12 : i32
    %c13_i32 = arith.constant 13 : i32
    %c14_i32 = arith.constant 14 : i32
    %c15_i32 = arith.constant 15 : i32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c11_i32 = arith.constant 11 : i32
    %c10_i32 = arith.constant 10 : i32
    %c50176_i32 = arith.constant 50176 : i32
    %c6_i32 = arith.constant 6 : i32
    %c100352_i32 = arith.constant 100352 : i32
    %c4_i32 = arith.constant 4 : i32
    %c7_i32 = arith.constant 7 : i32
    %c150528_i32 = arith.constant 150528 : i32
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %subview = memref.subview %arg1[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
    %0 = dfschedule.declaretile {col = 1 : i32, row = 0 : i32} : !dfschedule.tile
    %1 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
    %2 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
    %3 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
    %4 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
    %5 = dfschedule.schedule.getbdid(%0) : (!dfschedule.tile) -> i32
    %subview_0 = memref.subview %arg1[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>
    %6 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
    %7 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
    %8 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
    %9 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
    %10 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
    %11 = dfschedule.schedule.getbdid(%6) : (!dfschedule.tile) -> i32
    %subview_1 = memref.subview %arg1[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %12 = dfschedule.declaretile {col = 3 : i32, row = 0 : i32} : !dfschedule.tile
    %13 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
    %14 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
    %15 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
    %16 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
    %17 = dfschedule.schedule.getbdid(%12) : (!dfschedule.tile) -> i32
    %subview_2 = memref.subview %arg1[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>
    %18 = dfschedule.declaretile {col = 4 : i32, row = 0 : i32} : !dfschedule.tile
    %19 = dfschedule.declaretile {col = 4 : i32, row = 3 : i32} : !dfschedule.tile
    %20 = dfschedule.declaretile {col = 4 : i32, row = 4 : i32} : !dfschedule.tile
    %21 = dfschedule.declaretile {col = 4 : i32, row = 5 : i32} : !dfschedule.tile
    %22 = dfschedule.declaretile {col = 4 : i32, row = 6 : i32} : !dfschedule.tile
    %23 = dfschedule.schedule.getbdid(%18) : (!dfschedule.tile) -> i32
    %subview_3 = memref.subview %arg0[0, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1]>>
    %24 = dfschedule.schedule.getbdid(%0) : (!dfschedule.tile) -> i32
    %subview_4 = memref.subview %arg2[0, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1]>>
    %25 = dfschedule.schedule.getbdid(%18) : (!dfschedule.tile) -> i32
    %subview_5 = memref.subview %arg0[56, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>>
    %26 = dfschedule.schedule.getbdid(%6) : (!dfschedule.tile) -> i32
    %subview_6 = memref.subview %arg2[28, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %27 = dfschedule.schedule.getbdid(%18) : (!dfschedule.tile) -> i32
    %subview_7 = memref.subview %arg0[112, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %28 = dfschedule.schedule.getbdid(%12) : (!dfschedule.tile) -> i32
    %subview_8 = memref.subview %arg2[56, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %29 = dfschedule.schedule.getbdid(%12) : (!dfschedule.tile) -> i32
    %subview_9 = memref.subview %arg0[168, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>>
    %30 = dfschedule.schedule.getbdid(%18) : (!dfschedule.tile) -> i32
    %subview_10 = memref.subview %arg2[84, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %31 = dfschedule.schedule.getbdid(%12) : (!dfschedule.tile) -> i32
    %32 = dfschedule.ctrl_plan_init {ctrl_id = 1 : i32, resp_s2mm_ch = 0 : i32, rows = [{col_hi = 4 : i32, col_lo = 1 : i32, row = 3 : i32}, {col_hi = 4 : i32, col_lo = 1 : i32, row = 4 : i32}, {col_hi = 4 : i32, col_lo = 1 : i32, row = 5 : i32}, {col_hi = 4 : i32, col_lo = 1 : i32, row = 6 : i32}], shim_col = 0 : i32} : !dfschedule.ctrl_fabric
    %33 = dfschedule.config.load_kernel_group(%1, %2, %3, %4, %7, %8, %9, %10, %13, %14, %15, %16, %19, %20, %21, %22) {
      callee = [@dskernel_receiver]
    } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
    %34 = dfschedule.schedule.launch_kernel_group(%33) : (!dfschedule.kernelgroup) -> !dfschedule.event
    %35 = dfschedule.config.dma_bd(%subview, %0, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %36 = dfschedule.config.create_io(%35, %0) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %37 = dfschedule.config.dma_bd(%subview_0, %6, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %38 = dfschedule.config.create_io(%37, %6) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %39 = dfschedule.config.dma_bd(%subview_1, %12, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %40 = dfschedule.config.create_io(%39, %12) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %41 = dfschedule.config.dma_bd(%subview_2, %18, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %42 = dfschedule.config.create_io(%41, %18) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %43 = dfschedule.config.dma_bd(%subview_3, %0, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 18544 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 1 : i32,
      dim_strides = [4, 920, 224],
      dim_wraps = [61, 19, 4],
      iter_step_size = 12880 : i32,
      iter_wrap = 4 : i32
    } : (memref<61x230x4xi8, strided<[920, 4, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %44 = dfschedule.config.create_io(%43, %0) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %45 = dfschedule.config.dma_bd(%subview_4, %18, %c5_i32, %c150528_i32, %c7_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %46 = dfschedule.config.dma_bd(%subview_4, %18, %c4_i32, %c100352_i32, %c6_i32, %c-1_i32, %45) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %47 = dfschedule.config.dma_bd(%subview_4, %18, %c3_i32, %c50176_i32, %c5_i32, %c-1_i32, %46) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %48 = dfschedule.config.dma_bd(%subview_4, %18, %c2_i32, %c0_i32, %c4_i32, %c-1_i32, %47) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %49 = dfschedule.config.create_io(%48, %18) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = true
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %50 = dfschedule.config.dma_bd(%subview_5, %6, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 18544 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 1 : i32,
      dim_strides = [4, 920, 224],
      dim_wraps = [61, 19, 4],
      iter_step_size = 12880 : i32,
      iter_wrap = 4 : i32
    } : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %51 = dfschedule.config.create_io(%50, %6) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %52 = dfschedule.config.dma_bd(%subview_6, %18, %c10_i32, %c150528_i32, %c11_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %53 = dfschedule.config.dma_bd(%subview_6, %18, %c9_i32, %c100352_i32, %c10_i32, %c-1_i32, %52) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %54 = dfschedule.config.dma_bd(%subview_6, %18, %c8_i32, %c50176_i32, %c9_i32, %c-1_i32, %53) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %55 = dfschedule.config.dma_bd(%subview_6, %18, %c7_i32, %c0_i32, %c8_i32, %c-1_i32, %54) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %56 = dfschedule.config.create_io(%55, %18) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = true
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %57 = dfschedule.config.dma_bd(%subview_7, %12, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 18544 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 1 : i32,
      dim_strides = [4, 920, 224],
      dim_wraps = [61, 19, 4],
      iter_step_size = 12880 : i32,
      iter_wrap = 4 : i32
    } : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %58 = dfschedule.config.create_io(%57, %12) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %59 = dfschedule.config.dma_bd(%subview_8, %12, %c6_i32, %c150528_i32, %c15_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %60 = dfschedule.config.dma_bd(%subview_8, %12, %c5_i32, %c100352_i32, %c14_i32, %c-1_i32, %59) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %61 = dfschedule.config.dma_bd(%subview_8, %12, %c4_i32, %c50176_i32, %c13_i32, %c-1_i32, %60) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %62 = dfschedule.config.dma_bd(%subview_8, %12, %c3_i32, %c0_i32, %c12_i32, %c-1_i32, %61) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %63 = dfschedule.config.create_io(%62, %12) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = true
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %64 = dfschedule.config.dma_bd(%subview_9, %18, %c11_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 18544 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 1 : i32,
      dim_strides = [4, 920, 224],
      dim_wraps = [61, 19, 4],
      iter_step_size = 12880 : i32,
      iter_wrap = 4 : i32
    } : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %65 = dfschedule.config.create_io(%64, %18) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %66 = dfschedule.config.dma_bd(%subview_10, %12, %c11_i32, %c150528_i32, %c20_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %67 = dfschedule.config.dma_bd(%subview_10, %12, %c10_i32, %c100352_i32, %c19_i32, %c-1_i32, %66) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %68 = dfschedule.config.dma_bd(%subview_10, %12, %c9_i32, %c50176_i32, %c18_i32, %c-1_i32, %67) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %69 = dfschedule.config.dma_bd(%subview_10, %12, %c8_i32, %c0_i32, %c17_i32, %c-1_i32, %68) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      dim_strides = [4, 64],
      dim_wraps = [16, 49],
      iter_step_size = 3136 : i32,
      iter_wrap = 16 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %70 = dfschedule.config.create_io(%69, %12) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = true
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %71 = dfschedule.schedule.start_io(%49, %25) {flow_index = 5 : i32, repeat_count = 64 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %72 = dfschedule.schedule.start_io(%56, %27) {flow_index = 7 : i32, repeat_count = 64 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %73 = dfschedule.schedule.start_io(%63, %29) {flow_index = 9 : i32, repeat_count = 64 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %74 = dfschedule.schedule.start_io(%70, %31) {flow_index = 11 : i32, repeat_count = 64 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %75 = dfschedule.schedule.start_io(%36, %5) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %76 = dfschedule.schedule.start_io(%38, %11) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %77 = dfschedule.schedule.start_io(%40, %17) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %78 = dfschedule.schedule.start_io(%42, %23) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %79 = dfschedule.schedule.start_io(%44, %24) {flow_index = 4 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %80 = dfschedule.schedule.start_io(%51, %26) {flow_index = 6 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %81 = dfschedule.schedule.start_io(%58, %28) {flow_index = 8 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %82 = dfschedule.schedule.start_io(%65, %30) {flow_index = 10 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    dfschedule.schedule.wait(%34, %71, %72, %73, %74) : (!dfschedule.event, !dfschedule.event, !dfschedule.event, !dfschedule.event, !dfschedule.event)
  }
  dfschedule.dskernel_receiver @dskernel_receiver {
  }
}
