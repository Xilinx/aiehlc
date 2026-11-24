module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) attributes {routing.kernel_tiling = #routing.kernel_tiling<out_rounds = 16>} {
    dfschedule.launchhost @host_canonicalized
    return
  }
  dfschedule.host @host_canonicalized {
    %c15_i32 = arith.constant 15 : i32
    %c14_i32 = arith.constant 14 : i32
    %c13_i32 = arith.constant 13 : i32
    %c12_i32 = arith.constant 12 : i32
    %c11_i32 = arith.constant 11 : i32
    %c10_i32 = arith.constant 10 : i32
    %c9_i32 = arith.constant 9 : i32
    %c8_i32 = arith.constant 8 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c50176_i32 = arith.constant 50176 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %subview = memref.subview %arg1[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
    %0 = dfschedule.declaretile {col = 0 : i32, row = 0 : i32} : !dfschedule.tile
    %1 = dfschedule.config.dma_bd(%subview, %0, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %2 = dfschedule.config.create_io(%1, %0) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %3 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
    %4 = dfschedule.memref_mapping %subview : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
    %5 = dfschedule.bind_core_buffer(%4, %3) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %6 = dfschedule.bind_core_buffer(%4, %3) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %7 = dfschedule.config.dma_bd(%6, %3, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %8 = dfschedule.config.dma_bd(%5, %3, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %7) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %9 = dfschedule.config.create_io(%8, %3) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %10 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32
    %11 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
    %12 = dfschedule.memref_mapping %subview : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
    %13 = dfschedule.bind_core_buffer(%12, %11) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %14 = dfschedule.bind_core_buffer(%12, %11) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %15 = dfschedule.config.dma_bd(%14, %11, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %16 = dfschedule.config.dma_bd(%13, %11, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %15) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %17 = dfschedule.config.create_io(%16, %11) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %18 = dfschedule.schedule.getbdid(%11) : (!dfschedule.tile) -> i32
    %19 = dfschedule.declaretile {col = 0 : i32, row = 5 : i32} : !dfschedule.tile
    %20 = dfschedule.memref_mapping %subview : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
    %21 = dfschedule.bind_core_buffer(%20, %19) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %22 = dfschedule.bind_core_buffer(%20, %19) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %23 = dfschedule.config.dma_bd(%22, %19, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %24 = dfschedule.config.dma_bd(%21, %19, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %23) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %25 = dfschedule.config.create_io(%24, %19) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %26 = dfschedule.schedule.getbdid(%19) : (!dfschedule.tile) -> i32
    %27 = dfschedule.declaretile {col = 0 : i32, row = 6 : i32} : !dfschedule.tile
    %28 = dfschedule.memref_mapping %subview : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
    %29 = dfschedule.bind_core_buffer(%28, %27) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %30 = dfschedule.bind_core_buffer(%28, %27) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %31 = dfschedule.config.dma_bd(%30, %27, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %32 = dfschedule.config.dma_bd(%29, %27, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %31) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %33 = dfschedule.config.create_io(%32, %27) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %34 = dfschedule.schedule.getbdid(%27) : (!dfschedule.tile) -> i32
    %35 = dfschedule.schedule.getbdid(%0) : (!dfschedule.tile) -> i32
    %subview_0 = memref.subview %arg1[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>
    %36 = dfschedule.declaretile {col = 1 : i32, row = 0 : i32} : !dfschedule.tile
    %37 = dfschedule.config.dma_bd(%subview_0, %36, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %38 = dfschedule.config.create_io(%37, %36) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %39 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
    %subview_1 = memref.subview %subview_0[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %40 = dfschedule.memref_mapping %subview_1 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
    %41 = dfschedule.bind_core_buffer(%40, %39) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %42 = dfschedule.bind_core_buffer(%40, %39) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %43 = dfschedule.config.dma_bd(%42, %39, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %44 = dfschedule.config.dma_bd(%41, %39, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %43) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %45 = dfschedule.config.create_io(%44, %39) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %46 = dfschedule.schedule.getbdid(%39) : (!dfschedule.tile) -> i32
    %47 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
    %subview_2 = memref.subview %subview_0[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %48 = dfschedule.memref_mapping %subview_2 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
    %49 = dfschedule.bind_core_buffer(%48, %47) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %50 = dfschedule.bind_core_buffer(%48, %47) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %51 = dfschedule.config.dma_bd(%50, %47, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %52 = dfschedule.config.dma_bd(%49, %47, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %51) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %53 = dfschedule.config.create_io(%52, %47) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %54 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
    %55 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
    %subview_3 = memref.subview %subview_0[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %56 = dfschedule.memref_mapping %subview_3 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
    %57 = dfschedule.bind_core_buffer(%56, %55) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %58 = dfschedule.bind_core_buffer(%56, %55) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %59 = dfschedule.config.dma_bd(%58, %55, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %60 = dfschedule.config.dma_bd(%57, %55, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %59) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %61 = dfschedule.config.create_io(%60, %55) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %62 = dfschedule.schedule.getbdid(%55) : (!dfschedule.tile) -> i32
    %63 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
    %subview_4 = memref.subview %subview_0[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %64 = dfschedule.memref_mapping %subview_4 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
    %65 = dfschedule.bind_core_buffer(%64, %63) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %66 = dfschedule.bind_core_buffer(%64, %63) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %67 = dfschedule.config.dma_bd(%66, %63, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %68 = dfschedule.config.dma_bd(%65, %63, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %67) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %69 = dfschedule.config.create_io(%68, %63) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %70 = dfschedule.schedule.getbdid(%63) : (!dfschedule.tile) -> i32
    %71 = dfschedule.schedule.getbdid(%36) : (!dfschedule.tile) -> i32
    %subview_5 = memref.subview %arg1[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %72 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
    %73 = dfschedule.config.dma_bd(%subview_5, %72, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %74 = dfschedule.config.create_io(%73, %72) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %75 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
    %subview_6 = memref.subview %subview_5[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
    %76 = dfschedule.memref_mapping %subview_6 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
    %77 = dfschedule.bind_core_buffer(%76, %75) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %78 = dfschedule.bind_core_buffer(%76, %75) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %79 = dfschedule.config.dma_bd(%78, %75, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %80 = dfschedule.config.dma_bd(%77, %75, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %79) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %81 = dfschedule.config.create_io(%80, %75) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %82 = dfschedule.schedule.getbdid(%75) : (!dfschedule.tile) -> i32
    %83 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
    %subview_7 = memref.subview %subview_5[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
    %84 = dfschedule.memref_mapping %subview_7 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
    %85 = dfschedule.bind_core_buffer(%84, %83) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %86 = dfschedule.bind_core_buffer(%84, %83) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %87 = dfschedule.config.dma_bd(%86, %83, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %88 = dfschedule.config.dma_bd(%85, %83, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %87) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %89 = dfschedule.config.create_io(%88, %83) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %90 = dfschedule.schedule.getbdid(%83) : (!dfschedule.tile) -> i32
    %91 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
    %subview_8 = memref.subview %subview_5[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
    %92 = dfschedule.memref_mapping %subview_8 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
    %93 = dfschedule.bind_core_buffer(%92, %91) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %94 = dfschedule.bind_core_buffer(%92, %91) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %95 = dfschedule.config.dma_bd(%94, %91, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %96 = dfschedule.config.dma_bd(%93, %91, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %95) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %97 = dfschedule.config.create_io(%96, %91) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %98 = dfschedule.schedule.getbdid(%91) : (!dfschedule.tile) -> i32
    %99 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
    %subview_9 = memref.subview %subview_5[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
    %100 = dfschedule.memref_mapping %subview_9 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
    %101 = dfschedule.bind_core_buffer(%100, %99) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %102 = dfschedule.bind_core_buffer(%100, %99) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %103 = dfschedule.config.dma_bd(%102, %99, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %104 = dfschedule.config.dma_bd(%101, %99, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %103) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %105 = dfschedule.config.create_io(%104, %99) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %106 = dfschedule.schedule.getbdid(%99) : (!dfschedule.tile) -> i32
    %107 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    %subview_10 = memref.subview %arg1[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>
    %108 = dfschedule.declaretile {col = 3 : i32, row = 0 : i32} : !dfschedule.tile
    %109 = dfschedule.config.dma_bd(%subview_10, %108, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %110 = dfschedule.config.create_io(%109, %108) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %111 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
    %subview_11 = memref.subview %subview_10[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
    %112 = dfschedule.memref_mapping %subview_11 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
    %113 = dfschedule.bind_core_buffer(%112, %111) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %114 = dfschedule.bind_core_buffer(%112, %111) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %115 = dfschedule.config.dma_bd(%114, %111, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %116 = dfschedule.config.dma_bd(%113, %111, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %115) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %117 = dfschedule.config.create_io(%116, %111) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %118 = dfschedule.schedule.getbdid(%111) : (!dfschedule.tile) -> i32
    %119 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
    %subview_12 = memref.subview %subview_10[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
    %120 = dfschedule.memref_mapping %subview_12 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
    %121 = dfschedule.bind_core_buffer(%120, %119) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %122 = dfschedule.bind_core_buffer(%120, %119) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %123 = dfschedule.config.dma_bd(%122, %119, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %124 = dfschedule.config.dma_bd(%121, %119, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %123) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %125 = dfschedule.config.create_io(%124, %119) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %126 = dfschedule.schedule.getbdid(%119) : (!dfschedule.tile) -> i32
    %127 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
    %subview_13 = memref.subview %subview_10[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
    %128 = dfschedule.memref_mapping %subview_13 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
    %129 = dfschedule.bind_core_buffer(%128, %127) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %130 = dfschedule.bind_core_buffer(%128, %127) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %131 = dfschedule.config.dma_bd(%130, %127, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %132 = dfschedule.config.dma_bd(%129, %127, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %131) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %133 = dfschedule.config.create_io(%132, %127) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %134 = dfschedule.schedule.getbdid(%127) : (!dfschedule.tile) -> i32
    %135 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
    %subview_14 = memref.subview %subview_10[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
    %136 = dfschedule.memref_mapping %subview_14 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
    %137 = dfschedule.bind_core_buffer(%136, %135) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %138 = dfschedule.bind_core_buffer(%136, %135) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %139 = dfschedule.config.dma_bd(%138, %135, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %140 = dfschedule.config.dma_bd(%137, %135, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %139) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %141 = dfschedule.config.create_io(%140, %135) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %142 = dfschedule.schedule.getbdid(%135) : (!dfschedule.tile) -> i32
    %143 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
    %subview_15 = memref.subview %arg0[0, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1]>>
    %144 = dfschedule.config.dma_bd(%subview_15, %0, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
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
    %145 = dfschedule.config.create_io(%144, %0) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %146 = dfschedule.memref_mapping %subview_15 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
    %147 = dfschedule.bind_core_buffer(%146, %3) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %148 = dfschedule.bind_core_buffer(%146, %3) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %149 = dfschedule.config.dma_bd(%148, %3, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %150 = dfschedule.config.dma_bd(%147, %3, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %149) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %151 = dfschedule.config.create_io(%150, %3) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %152 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32
    %153 = dfschedule.memref_mapping %subview_15 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
    %154 = dfschedule.bind_core_buffer(%153, %39) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %155 = dfschedule.bind_core_buffer(%153, %39) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %156 = dfschedule.config.dma_bd(%155, %39, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %157 = dfschedule.config.dma_bd(%154, %39, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %156) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %158 = dfschedule.config.create_io(%157, %39) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %159 = dfschedule.schedule.getbdid(%39) : (!dfschedule.tile) -> i32
    %160 = dfschedule.memref_mapping %subview_15 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
    %161 = dfschedule.bind_core_buffer(%160, %75) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %162 = dfschedule.bind_core_buffer(%160, %75) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %163 = dfschedule.config.dma_bd(%162, %75, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %164 = dfschedule.config.dma_bd(%161, %75, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %163) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %165 = dfschedule.config.create_io(%164, %75) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %166 = dfschedule.schedule.getbdid(%75) : (!dfschedule.tile) -> i32
    %167 = dfschedule.memref_mapping %subview_15 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
    %168 = dfschedule.bind_core_buffer(%167, %111) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %169 = dfschedule.bind_core_buffer(%167, %111) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %170 = dfschedule.config.dma_bd(%169, %111, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %171 = dfschedule.config.dma_bd(%168, %111, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %170) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %172 = dfschedule.config.create_io(%171, %111) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %173 = dfschedule.schedule.getbdid(%111) : (!dfschedule.tile) -> i32
    %174 = dfschedule.schedule.getbdid(%0) : (!dfschedule.tile) -> i32
    %subview_16 = memref.subview %arg2[0, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1]>>
    %subview_17 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %175 = dfschedule.memref_mapping %subview_17 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %176 = dfschedule.bind_core_buffer(%175, %3) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %177 = dfschedule.bind_core_buffer(%175, %3) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %178 = dfschedule.config.dma_bd(%177, %3, %c5_i32, %c0_i32, %c1_i32, %c2_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %179 = dfschedule.config.dma_bd(%176, %3, %c4_i32, %c0_i32, %c1_i32, %c2_i32, %178) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %180 = dfschedule.config.create_io(%179, %3) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %181 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32
    %subview_18 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %182 = dfschedule.memref_mapping %subview_18 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %183 = dfschedule.bind_core_buffer(%182, %39) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %184 = dfschedule.bind_core_buffer(%182, %39) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %185 = dfschedule.config.dma_bd(%184, %39, %c5_i32, %c0_i32, %c2_i32, %c3_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %186 = dfschedule.config.dma_bd(%183, %39, %c4_i32, %c0_i32, %c2_i32, %c3_i32, %185) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %187 = dfschedule.config.create_io(%186, %39) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %188 = dfschedule.schedule.getbdid(%39) : (!dfschedule.tile) -> i32
    %subview_19 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %189 = dfschedule.memref_mapping %subview_19 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %190 = dfschedule.bind_core_buffer(%189, %75) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %191 = dfschedule.bind_core_buffer(%189, %75) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %192 = dfschedule.config.dma_bd(%191, %75, %c5_i32, %c0_i32, %c3_i32, %c4_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %193 = dfschedule.config.dma_bd(%190, %75, %c4_i32, %c0_i32, %c3_i32, %c4_i32, %192) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %194 = dfschedule.config.create_io(%193, %75) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %195 = dfschedule.schedule.getbdid(%75) : (!dfschedule.tile) -> i32
    %subview_20 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %196 = dfschedule.memref_mapping %subview_20 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %197 = dfschedule.bind_core_buffer(%196, %111) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %198 = dfschedule.bind_core_buffer(%196, %111) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %199 = dfschedule.config.dma_bd(%198, %111, %c5_i32, %c0_i32, %c4_i32, %c5_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %200 = dfschedule.config.dma_bd(%197, %111, %c4_i32, %c0_i32, %c4_i32, %c5_i32, %199) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %201 = dfschedule.config.create_io(%200, %111) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %202 = dfschedule.schedule.getbdid(%111) : (!dfschedule.tile) -> i32
    %203 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %442 = arith.index_cast %arg3 : index to i32
      %443 = arith.muli %442, %c50176_i32 : i32
      %444 = arith.addi %443, %c48_i32 : i32
      %445 = dfschedule.config.dma_bd(%subview_16, %108, %c5_i32, %444, %c4_i32, %c-1_i32) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %446 = arith.addi %443, %c32_i32 : i32
      %447 = dfschedule.config.dma_bd(%subview_16, %108, %c4_i32, %446, %c3_i32, %c-1_i32, %445) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %448 = arith.addi %443, %c16_i32 : i32
      %449 = dfschedule.config.dma_bd(%subview_16, %108, %c3_i32, %448, %c2_i32, %c-1_i32, %447) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %450 = dfschedule.config.dma_bd(%subview_16, %108, %c2_i32, %443, %c1_i32, %c-1_i32, %449) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %451 = dfschedule.config.create_io(%450, %108) {
        channel = 0,
        direction = "S2MM",
        io_operation = "RECV",
        enable_out_of_order = true
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %452 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
      %453 = dfschedule.schedule.start_io(%451, %452) {flow_index = 5 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
      dfschedule.schedule.wait(%453) : (!dfschedule.event)
    }
    %subview_21 = memref.subview %arg0[56, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>>
    %204 = dfschedule.config.dma_bd(%subview_21, %36, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
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
    %205 = dfschedule.config.create_io(%204, %36) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_22 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %206 = dfschedule.memref_mapping %subview_22 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %207 = dfschedule.bind_core_buffer(%206, %11) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %208 = dfschedule.bind_core_buffer(%206, %11) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %209 = dfschedule.config.dma_bd(%208, %11, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %210 = dfschedule.config.dma_bd(%207, %11, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %209) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %211 = dfschedule.config.create_io(%210, %11) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %212 = dfschedule.schedule.getbdid(%11) : (!dfschedule.tile) -> i32
    %subview_23 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %213 = dfschedule.memref_mapping %subview_23 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %214 = dfschedule.bind_core_buffer(%213, %47) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %215 = dfschedule.bind_core_buffer(%213, %47) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %216 = dfschedule.config.dma_bd(%215, %47, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %217 = dfschedule.config.dma_bd(%214, %47, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %216) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %218 = dfschedule.config.create_io(%217, %47) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %219 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
    %subview_24 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %220 = dfschedule.memref_mapping %subview_24 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %221 = dfschedule.bind_core_buffer(%220, %83) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %222 = dfschedule.bind_core_buffer(%220, %83) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %223 = dfschedule.config.dma_bd(%222, %83, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %224 = dfschedule.config.dma_bd(%221, %83, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %223) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %225 = dfschedule.config.create_io(%224, %83) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %226 = dfschedule.schedule.getbdid(%83) : (!dfschedule.tile) -> i32
    %subview_25 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %227 = dfschedule.memref_mapping %subview_25 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %228 = dfschedule.bind_core_buffer(%227, %119) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %229 = dfschedule.bind_core_buffer(%227, %119) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %230 = dfschedule.config.dma_bd(%229, %119, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %231 = dfschedule.config.dma_bd(%228, %119, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %230) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %232 = dfschedule.config.create_io(%231, %119) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %233 = dfschedule.schedule.getbdid(%119) : (!dfschedule.tile) -> i32
    %234 = dfschedule.schedule.getbdid(%36) : (!dfschedule.tile) -> i32
    %subview_26 = memref.subview %arg2[28, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %subview_27 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %235 = dfschedule.memref_mapping %subview_27 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %236 = dfschedule.bind_core_buffer(%235, %11) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %237 = dfschedule.bind_core_buffer(%235, %11) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %238 = dfschedule.config.dma_bd(%237, %11, %c5_i32, %c0_i32, %c5_i32, %c7_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %239 = dfschedule.config.dma_bd(%236, %11, %c4_i32, %c0_i32, %c5_i32, %c7_i32, %238) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %240 = dfschedule.config.create_io(%239, %11) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %241 = dfschedule.schedule.getbdid(%11) : (!dfschedule.tile) -> i32
    %subview_28 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %242 = dfschedule.memref_mapping %subview_28 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %243 = dfschedule.bind_core_buffer(%242, %47) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %244 = dfschedule.bind_core_buffer(%242, %47) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %245 = dfschedule.config.dma_bd(%244, %47, %c5_i32, %c0_i32, %c6_i32, %c8_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %246 = dfschedule.config.dma_bd(%243, %47, %c4_i32, %c0_i32, %c6_i32, %c8_i32, %245) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %247 = dfschedule.config.create_io(%246, %47) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %248 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
    %subview_29 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %249 = dfschedule.memref_mapping %subview_29 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %250 = dfschedule.bind_core_buffer(%249, %83) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %251 = dfschedule.bind_core_buffer(%249, %83) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %252 = dfschedule.config.dma_bd(%251, %83, %c5_i32, %c0_i32, %c7_i32, %c9_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %253 = dfschedule.config.dma_bd(%250, %83, %c4_i32, %c0_i32, %c7_i32, %c9_i32, %252) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %254 = dfschedule.config.create_io(%253, %83) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %255 = dfschedule.schedule.getbdid(%83) : (!dfschedule.tile) -> i32
    %subview_30 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %256 = dfschedule.memref_mapping %subview_30 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %257 = dfschedule.bind_core_buffer(%256, %119) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %258 = dfschedule.bind_core_buffer(%256, %119) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %259 = dfschedule.config.dma_bd(%258, %119, %c5_i32, %c0_i32, %c8_i32, %c10_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %260 = dfschedule.config.dma_bd(%257, %119, %c4_i32, %c0_i32, %c8_i32, %c10_i32, %259) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %261 = dfschedule.config.create_io(%260, %119) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %262 = dfschedule.schedule.getbdid(%119) : (!dfschedule.tile) -> i32
    %263 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %442 = arith.index_cast %arg3 : index to i32
      %443 = arith.muli %442, %c50176_i32 : i32
      %444 = arith.addi %443, %c48_i32 : i32
      %445 = dfschedule.config.dma_bd(%subview_26, %108, %c10_i32, %444, %c8_i32, %c-1_i32) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %446 = arith.addi %443, %c32_i32 : i32
      %447 = dfschedule.config.dma_bd(%subview_26, %108, %c9_i32, %446, %c7_i32, %c-1_i32, %445) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %448 = arith.addi %443, %c16_i32 : i32
      %449 = dfschedule.config.dma_bd(%subview_26, %108, %c8_i32, %448, %c6_i32, %c-1_i32, %447) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %450 = dfschedule.config.dma_bd(%subview_26, %108, %c7_i32, %443, %c5_i32, %c-1_i32, %449) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %451 = dfschedule.config.create_io(%450, %108) {
        channel = 1,
        direction = "S2MM",
        io_operation = "RECV",
        enable_out_of_order = true
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %452 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
      %453 = dfschedule.schedule.start_io(%451, %452) {flow_index = 7 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
      dfschedule.schedule.wait(%453) : (!dfschedule.event)
    }
    %subview_31 = memref.subview %arg0[112, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %264 = dfschedule.config.dma_bd(%subview_31, %72, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
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
    %265 = dfschedule.config.create_io(%264, %72) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_32 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %266 = dfschedule.memref_mapping %subview_32 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %267 = dfschedule.bind_core_buffer(%266, %19) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %268 = dfschedule.bind_core_buffer(%266, %19) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %269 = dfschedule.config.dma_bd(%268, %19, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %270 = dfschedule.config.dma_bd(%267, %19, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %269) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %271 = dfschedule.config.create_io(%270, %19) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %272 = dfschedule.schedule.getbdid(%19) : (!dfschedule.tile) -> i32
    %subview_33 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %273 = dfschedule.memref_mapping %subview_33 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %274 = dfschedule.bind_core_buffer(%273, %55) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %275 = dfschedule.bind_core_buffer(%273, %55) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %276 = dfschedule.config.dma_bd(%275, %55, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %277 = dfschedule.config.dma_bd(%274, %55, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %276) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %278 = dfschedule.config.create_io(%277, %55) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %279 = dfschedule.schedule.getbdid(%55) : (!dfschedule.tile) -> i32
    %subview_34 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %280 = dfschedule.memref_mapping %subview_34 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %281 = dfschedule.bind_core_buffer(%280, %91) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %282 = dfschedule.bind_core_buffer(%280, %91) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %283 = dfschedule.config.dma_bd(%282, %91, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %284 = dfschedule.config.dma_bd(%281, %91, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %283) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %285 = dfschedule.config.create_io(%284, %91) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %286 = dfschedule.schedule.getbdid(%91) : (!dfschedule.tile) -> i32
    %subview_35 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %287 = dfschedule.memref_mapping %subview_35 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %288 = dfschedule.bind_core_buffer(%287, %127) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %289 = dfschedule.bind_core_buffer(%287, %127) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %290 = dfschedule.config.dma_bd(%289, %127, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %291 = dfschedule.config.dma_bd(%288, %127, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %290) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %292 = dfschedule.config.create_io(%291, %127) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %293 = dfschedule.schedule.getbdid(%127) : (!dfschedule.tile) -> i32
    %294 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    %subview_36 = memref.subview %arg2[56, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %subview_37 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %295 = dfschedule.memref_mapping %subview_37 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %296 = dfschedule.bind_core_buffer(%295, %19) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %297 = dfschedule.bind_core_buffer(%295, %19) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %298 = dfschedule.config.dma_bd(%297, %19, %c5_i32, %c0_i32, %c9_i32, %c3_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %299 = dfschedule.config.dma_bd(%296, %19, %c4_i32, %c0_i32, %c9_i32, %c3_i32, %298) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %300 = dfschedule.config.create_io(%299, %19) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %301 = dfschedule.schedule.getbdid(%19) : (!dfschedule.tile) -> i32
    %subview_38 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %302 = dfschedule.memref_mapping %subview_38 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %303 = dfschedule.bind_core_buffer(%302, %55) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %304 = dfschedule.bind_core_buffer(%302, %55) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %305 = dfschedule.config.dma_bd(%304, %55, %c5_i32, %c0_i32, %c10_i32, %c4_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %306 = dfschedule.config.dma_bd(%303, %55, %c4_i32, %c0_i32, %c10_i32, %c4_i32, %305) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %307 = dfschedule.config.create_io(%306, %55) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %308 = dfschedule.schedule.getbdid(%55) : (!dfschedule.tile) -> i32
    %subview_39 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %309 = dfschedule.memref_mapping %subview_39 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %310 = dfschedule.bind_core_buffer(%309, %91) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %311 = dfschedule.bind_core_buffer(%309, %91) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %312 = dfschedule.config.dma_bd(%311, %91, %c5_i32, %c0_i32, %c11_i32, %c5_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %313 = dfschedule.config.dma_bd(%310, %91, %c4_i32, %c0_i32, %c11_i32, %c5_i32, %312) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %314 = dfschedule.config.create_io(%313, %91) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %315 = dfschedule.schedule.getbdid(%91) : (!dfschedule.tile) -> i32
    %subview_40 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %316 = dfschedule.memref_mapping %subview_40 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %317 = dfschedule.bind_core_buffer(%316, %127) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %318 = dfschedule.bind_core_buffer(%316, %127) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %319 = dfschedule.config.dma_bd(%318, %127, %c5_i32, %c0_i32, %c12_i32, %c6_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %320 = dfschedule.config.dma_bd(%317, %127, %c4_i32, %c0_i32, %c12_i32, %c6_i32, %319) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %321 = dfschedule.config.create_io(%320, %127) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %322 = dfschedule.schedule.getbdid(%127) : (!dfschedule.tile) -> i32
    %323 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %442 = arith.index_cast %arg3 : index to i32
      %443 = arith.muli %442, %c50176_i32 : i32
      %444 = arith.addi %443, %c48_i32 : i32
      %445 = dfschedule.config.dma_bd(%subview_36, %72, %c6_i32, %444, %c12_i32, %c-1_i32) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %446 = arith.addi %443, %c32_i32 : i32
      %447 = dfschedule.config.dma_bd(%subview_36, %72, %c5_i32, %446, %c11_i32, %c-1_i32, %445) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %448 = arith.addi %443, %c16_i32 : i32
      %449 = dfschedule.config.dma_bd(%subview_36, %72, %c4_i32, %448, %c10_i32, %c-1_i32, %447) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %450 = dfschedule.config.dma_bd(%subview_36, %72, %c3_i32, %443, %c9_i32, %c-1_i32, %449) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %451 = dfschedule.config.create_io(%450, %72) {
        channel = 0,
        direction = "S2MM",
        io_operation = "RECV",
        enable_out_of_order = true
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %452 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
      %453 = dfschedule.schedule.start_io(%451, %452) {flow_index = 9 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
      dfschedule.schedule.wait(%453) : (!dfschedule.event)
    }
    %subview_41 = memref.subview %arg0[168, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>>
    %324 = dfschedule.config.dma_bd(%subview_41, %108, %c11_i32, %c0_i32, %c0_i32, %c-1_i32) {
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
    %325 = dfschedule.config.create_io(%324, %108) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_42 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %326 = dfschedule.memref_mapping %subview_42 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %327 = dfschedule.bind_core_buffer(%326, %27) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %328 = dfschedule.bind_core_buffer(%326, %27) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %329 = dfschedule.config.dma_bd(%328, %27, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %330 = dfschedule.config.dma_bd(%327, %27, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %329) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %331 = dfschedule.config.create_io(%330, %27) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %332 = dfschedule.schedule.getbdid(%27) : (!dfschedule.tile) -> i32
    %subview_43 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %333 = dfschedule.memref_mapping %subview_43 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %334 = dfschedule.bind_core_buffer(%333, %63) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %335 = dfschedule.bind_core_buffer(%333, %63) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %336 = dfschedule.config.dma_bd(%335, %63, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %337 = dfschedule.config.dma_bd(%334, %63, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %336) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %338 = dfschedule.config.create_io(%337, %63) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %339 = dfschedule.schedule.getbdid(%63) : (!dfschedule.tile) -> i32
    %subview_44 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %340 = dfschedule.memref_mapping %subview_44 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %341 = dfschedule.bind_core_buffer(%340, %99) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %342 = dfschedule.bind_core_buffer(%340, %99) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %343 = dfschedule.config.dma_bd(%342, %99, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %344 = dfschedule.config.dma_bd(%341, %99, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %343) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %345 = dfschedule.config.create_io(%344, %99) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %346 = dfschedule.schedule.getbdid(%99) : (!dfschedule.tile) -> i32
    %subview_45 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %347 = dfschedule.memref_mapping %subview_45 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %348 = dfschedule.bind_core_buffer(%347, %135) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %349 = dfschedule.bind_core_buffer(%347, %135) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %350 = dfschedule.config.dma_bd(%349, %135, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %351 = dfschedule.config.dma_bd(%348, %135, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %350) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %352 = dfschedule.config.create_io(%351, %135) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %353 = dfschedule.schedule.getbdid(%135) : (!dfschedule.tile) -> i32
    %354 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
    %subview_46 = memref.subview %arg2[84, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %subview_47 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %355 = dfschedule.memref_mapping %subview_47 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %356 = dfschedule.bind_core_buffer(%355, %27) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %357 = dfschedule.bind_core_buffer(%355, %27) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %358 = dfschedule.config.dma_bd(%357, %27, %c5_i32, %c0_i32, %c13_i32, %c8_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %359 = dfschedule.config.dma_bd(%356, %27, %c4_i32, %c0_i32, %c13_i32, %c8_i32, %358) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %360 = dfschedule.config.create_io(%359, %27) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %361 = dfschedule.schedule.getbdid(%27) : (!dfschedule.tile) -> i32
    %subview_48 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %362 = dfschedule.memref_mapping %subview_48 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %363 = dfschedule.bind_core_buffer(%362, %63) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %364 = dfschedule.bind_core_buffer(%362, %63) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %365 = dfschedule.config.dma_bd(%364, %63, %c5_i32, %c0_i32, %c14_i32, %c9_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %366 = dfschedule.config.dma_bd(%363, %63, %c4_i32, %c0_i32, %c14_i32, %c9_i32, %365) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %367 = dfschedule.config.create_io(%366, %63) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %368 = dfschedule.schedule.getbdid(%63) : (!dfschedule.tile) -> i32
    %subview_49 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %369 = dfschedule.memref_mapping %subview_49 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %370 = dfschedule.bind_core_buffer(%369, %99) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %371 = dfschedule.bind_core_buffer(%369, %99) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %372 = dfschedule.config.dma_bd(%371, %99, %c5_i32, %c0_i32, %c15_i32, %c10_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %373 = dfschedule.config.dma_bd(%370, %99, %c4_i32, %c0_i32, %c15_i32, %c10_i32, %372) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %374 = dfschedule.config.create_io(%373, %99) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %375 = dfschedule.schedule.getbdid(%99) : (!dfschedule.tile) -> i32
    %subview_50 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %376 = dfschedule.memref_mapping %subview_50 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %377 = dfschedule.bind_core_buffer(%376, %135) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %378 = dfschedule.bind_core_buffer(%376, %135) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %379 = dfschedule.config.dma_bd(%378, %135, %c5_i32, %c0_i32, %c16_i32, %c11_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %380 = dfschedule.config.dma_bd(%377, %135, %c4_i32, %c0_i32, %c16_i32, %c11_i32, %379) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %381 = dfschedule.config.create_io(%380, %135) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %382 = dfschedule.schedule.getbdid(%135) : (!dfschedule.tile) -> i32
    %383 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %442 = arith.index_cast %arg3 : index to i32
      %443 = arith.muli %442, %c50176_i32 : i32
      %444 = arith.addi %443, %c48_i32 : i32
      %445 = dfschedule.config.dma_bd(%subview_46, %72, %c11_i32, %444, %c16_i32, %c-1_i32) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %446 = arith.addi %443, %c32_i32 : i32
      %447 = dfschedule.config.dma_bd(%subview_46, %72, %c10_i32, %446, %c15_i32, %c-1_i32, %445) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %448 = arith.addi %443, %c16_i32 : i32
      %449 = dfschedule.config.dma_bd(%subview_46, %72, %c9_i32, %448, %c14_i32, %c-1_i32, %447) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %450 = dfschedule.config.dma_bd(%subview_46, %72, %c8_i32, %443, %c13_i32, %c-1_i32, %449) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %451 = dfschedule.config.create_io(%450, %72) {
        channel = 1,
        direction = "S2MM",
        io_operation = "RECV",
        enable_out_of_order = true
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %452 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
      %453 = dfschedule.schedule.start_io(%451, %452) {flow_index = 11 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
      dfschedule.schedule.wait(%453) : (!dfschedule.event)
    }
    %384 = dfschedule.config.load_kernel_group(%3, %11, %19, %27, %39, %47, %55, %63, %75, %83, %91, %99, %111, %119, %127, %135) {
      callee = [@dskernel_receiver]
    } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
    %385 = dfschedule.schedule.launch_kernel_group(%384) : (!dfschedule.kernelgroup) -> !dfschedule.event
    %386 = dfschedule.schedule.start_io(%9, %10) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %387 = dfschedule.schedule.start_io(%17, %18) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %388 = dfschedule.schedule.start_io(%25, %26) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %389 = dfschedule.schedule.start_io(%33, %34) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %390 = dfschedule.schedule.start_io(%45, %46) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %391 = dfschedule.schedule.start_io(%53, %54) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %392 = dfschedule.schedule.start_io(%61, %62) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %393 = dfschedule.schedule.start_io(%69, %70) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %394 = dfschedule.schedule.start_io(%81, %82) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %395 = dfschedule.schedule.start_io(%89, %90) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %396 = dfschedule.schedule.start_io(%97, %98) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %397 = dfschedule.schedule.start_io(%105, %106) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %398 = dfschedule.schedule.start_io(%117, %118) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %399 = dfschedule.schedule.start_io(%125, %126) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %400 = dfschedule.schedule.start_io(%133, %134) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %401 = dfschedule.schedule.start_io(%141, %142) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %402 = dfschedule.schedule.start_io(%151, %152) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %403 = dfschedule.schedule.start_io(%158, %159) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %404 = dfschedule.schedule.start_io(%165, %166) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %405 = dfschedule.schedule.start_io(%172, %173) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %406 = dfschedule.schedule.start_io(%180, %181) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %407 = dfschedule.schedule.start_io(%187, %188) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %408 = dfschedule.schedule.start_io(%194, %195) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %409 = dfschedule.schedule.start_io(%201, %202) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %410 = dfschedule.schedule.start_io(%211, %212) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %411 = dfschedule.schedule.start_io(%218, %219) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %412 = dfschedule.schedule.start_io(%225, %226) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %413 = dfschedule.schedule.start_io(%232, %233) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %414 = dfschedule.schedule.start_io(%240, %241) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %415 = dfschedule.schedule.start_io(%247, %248) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %416 = dfschedule.schedule.start_io(%254, %255) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %417 = dfschedule.schedule.start_io(%261, %262) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %418 = dfschedule.schedule.start_io(%271, %272) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %419 = dfschedule.schedule.start_io(%278, %279) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %420 = dfschedule.schedule.start_io(%285, %286) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %421 = dfschedule.schedule.start_io(%292, %293) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %422 = dfschedule.schedule.start_io(%300, %301) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %423 = dfschedule.schedule.start_io(%307, %308) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %424 = dfschedule.schedule.start_io(%314, %315) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %425 = dfschedule.schedule.start_io(%321, %322) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %426 = dfschedule.schedule.start_io(%331, %332) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %427 = dfschedule.schedule.start_io(%338, %339) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %428 = dfschedule.schedule.start_io(%345, %346) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %429 = dfschedule.schedule.start_io(%352, %353) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %430 = dfschedule.schedule.start_io(%360, %361) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %431 = dfschedule.schedule.start_io(%367, %368) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %432 = dfschedule.schedule.start_io(%374, %375) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %433 = dfschedule.schedule.start_io(%381, %382) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %434 = dfschedule.schedule.start_io(%2, %35) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %435 = dfschedule.schedule.start_io(%38, %71) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %436 = dfschedule.schedule.start_io(%74, %107) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %437 = dfschedule.schedule.start_io(%110, %143) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %438 = dfschedule.schedule.start_io(%145, %174) {flow_index = 4 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %439 = dfschedule.schedule.start_io(%205, %234) {flow_index = 6 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %440 = dfschedule.schedule.start_io(%265, %294) {flow_index = 8 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %441 = dfschedule.schedule.start_io(%325, %354) {flow_index = 10 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    dfschedule.schedule.wait(%385) : (!dfschedule.event)
  }
  dfschedule.dskernel_receiver @dskernel_receiver {
  }
}
