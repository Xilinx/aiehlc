module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) {
    dfschedule.launchhost @host_canonicalized
    return
  }
  dfschedule.host @host_canonicalized {
    %c13_i32 = arith.constant 13 : i32
    %c14_i32 = arith.constant 14 : i32
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c11_i32 = arith.constant 11 : i32
    %c12_i32 = arith.constant 12 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c9_i32 = arith.constant 9 : i32
    %c8_i32 = arith.constant 8 : i32
    %c10_i32 = arith.constant 10 : i32
    %c12544_i32 = arith.constant 12544 : i32
    %c25088_i32 = arith.constant 25088 : i32
    %c4_i32 = arith.constant 4 : i32
    %c37632_i32 = arith.constant 37632 : i32
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
    %175 = dfschedule.config.dma_bd(%subview_16, %108, %c5_i32, %c37632_i32, %c4_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %176 = dfschedule.config.dma_bd(%subview_16, %108, %c4_i32, %c25088_i32, %c3_i32, %c-1_i32, %175) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %177 = dfschedule.config.dma_bd(%subview_16, %108, %c3_i32, %c12544_i32, %c2_i32, %c-1_i32, %176) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %178 = dfschedule.config.dma_bd(%subview_16, %108, %c2_i32, %c0_i32, %c1_i32, %c-1_i32, %177) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %179 = dfschedule.config.create_io(%178, %108) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = true
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_17 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %180 = dfschedule.memref_mapping %subview_17 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %181 = dfschedule.bind_core_buffer(%180, %3) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %182 = dfschedule.bind_core_buffer(%180, %3) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %183 = dfschedule.config.dma_bd(%182, %3, %c5_i32, %c0_i32, %c1_i32, %c2_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %184 = dfschedule.config.dma_bd(%181, %3, %c4_i32, %c0_i32, %c1_i32, %c2_i32, %183) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %185 = dfschedule.config.create_io(%184, %3) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %186 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32
    %subview_18 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %187 = dfschedule.memref_mapping %subview_18 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %188 = dfschedule.bind_core_buffer(%187, %39) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %189 = dfschedule.bind_core_buffer(%187, %39) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %190 = dfschedule.config.dma_bd(%189, %39, %c5_i32, %c0_i32, %c2_i32, %c3_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %191 = dfschedule.config.dma_bd(%188, %39, %c4_i32, %c0_i32, %c2_i32, %c3_i32, %190) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %192 = dfschedule.config.create_io(%191, %39) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %193 = dfschedule.schedule.getbdid(%39) : (!dfschedule.tile) -> i32
    %subview_19 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %194 = dfschedule.memref_mapping %subview_19 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %195 = dfschedule.bind_core_buffer(%194, %75) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %196 = dfschedule.bind_core_buffer(%194, %75) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %197 = dfschedule.config.dma_bd(%196, %75, %c5_i32, %c0_i32, %c3_i32, %c4_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %198 = dfschedule.config.dma_bd(%195, %75, %c4_i32, %c0_i32, %c3_i32, %c4_i32, %197) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %199 = dfschedule.config.create_io(%198, %75) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %200 = dfschedule.schedule.getbdid(%75) : (!dfschedule.tile) -> i32
    %subview_20 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %201 = dfschedule.memref_mapping %subview_20 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %202 = dfschedule.bind_core_buffer(%201, %111) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %203 = dfschedule.bind_core_buffer(%201, %111) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %204 = dfschedule.config.dma_bd(%203, %111, %c5_i32, %c0_i32, %c4_i32, %c5_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %205 = dfschedule.config.dma_bd(%202, %111, %c4_i32, %c0_i32, %c4_i32, %c5_i32, %204) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %206 = dfschedule.config.create_io(%205, %111) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %207 = dfschedule.schedule.getbdid(%111) : (!dfschedule.tile) -> i32
    %208 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
    %209 = dfschedule.schedule.start_io(%179, %208) {flow_index = 5 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %subview_21 = memref.subview %arg0[56, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>>
    %210 = dfschedule.config.dma_bd(%subview_21, %36, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
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
    %211 = dfschedule.config.create_io(%210, %36) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_22 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %212 = dfschedule.memref_mapping %subview_22 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %213 = dfschedule.bind_core_buffer(%212, %11) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %214 = dfschedule.bind_core_buffer(%212, %11) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %215 = dfschedule.config.dma_bd(%214, %11, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %216 = dfschedule.config.dma_bd(%213, %11, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %215) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %217 = dfschedule.config.create_io(%216, %11) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %218 = dfschedule.schedule.getbdid(%11) : (!dfschedule.tile) -> i32
    %subview_23 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %219 = dfschedule.memref_mapping %subview_23 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %220 = dfschedule.bind_core_buffer(%219, %47) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %221 = dfschedule.bind_core_buffer(%219, %47) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %222 = dfschedule.config.dma_bd(%221, %47, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %223 = dfschedule.config.dma_bd(%220, %47, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %222) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %224 = dfschedule.config.create_io(%223, %47) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %225 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
    %subview_24 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %226 = dfschedule.memref_mapping %subview_24 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %227 = dfschedule.bind_core_buffer(%226, %83) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %228 = dfschedule.bind_core_buffer(%226, %83) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %229 = dfschedule.config.dma_bd(%228, %83, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %230 = dfschedule.config.dma_bd(%227, %83, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %229) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %231 = dfschedule.config.create_io(%230, %83) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %232 = dfschedule.schedule.getbdid(%83) : (!dfschedule.tile) -> i32
    %subview_25 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %233 = dfschedule.memref_mapping %subview_25 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %234 = dfschedule.bind_core_buffer(%233, %119) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %235 = dfschedule.bind_core_buffer(%233, %119) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %236 = dfschedule.config.dma_bd(%235, %119, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %237 = dfschedule.config.dma_bd(%234, %119, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %236) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %238 = dfschedule.config.create_io(%237, %119) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %239 = dfschedule.schedule.getbdid(%119) : (!dfschedule.tile) -> i32
    %240 = dfschedule.schedule.getbdid(%36) : (!dfschedule.tile) -> i32
    %subview_26 = memref.subview %arg2[28, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %241 = dfschedule.config.dma_bd(%subview_26, %108, %c10_i32, %c37632_i32, %c8_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %242 = dfschedule.config.dma_bd(%subview_26, %108, %c9_i32, %c25088_i32, %c7_i32, %c-1_i32, %241) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %243 = dfschedule.config.dma_bd(%subview_26, %108, %c8_i32, %c12544_i32, %c6_i32, %c-1_i32, %242) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %244 = dfschedule.config.dma_bd(%subview_26, %108, %c7_i32, %c0_i32, %c5_i32, %c-1_i32, %243) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %245 = dfschedule.config.create_io(%244, %108) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = true
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_27 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %246 = dfschedule.memref_mapping %subview_27 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %247 = dfschedule.bind_core_buffer(%246, %11) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %248 = dfschedule.bind_core_buffer(%246, %11) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %249 = dfschedule.config.dma_bd(%248, %11, %c5_i32, %c0_i32, %c5_i32, %c7_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %250 = dfschedule.config.dma_bd(%247, %11, %c4_i32, %c0_i32, %c5_i32, %c7_i32, %249) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %251 = dfschedule.config.create_io(%250, %11) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %252 = dfschedule.schedule.getbdid(%11) : (!dfschedule.tile) -> i32
    %subview_28 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %253 = dfschedule.memref_mapping %subview_28 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %254 = dfschedule.bind_core_buffer(%253, %47) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %255 = dfschedule.bind_core_buffer(%253, %47) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %256 = dfschedule.config.dma_bd(%255, %47, %c5_i32, %c0_i32, %c6_i32, %c8_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %257 = dfschedule.config.dma_bd(%254, %47, %c4_i32, %c0_i32, %c6_i32, %c8_i32, %256) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %258 = dfschedule.config.create_io(%257, %47) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %259 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
    %subview_29 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %260 = dfschedule.memref_mapping %subview_29 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %261 = dfschedule.bind_core_buffer(%260, %83) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %262 = dfschedule.bind_core_buffer(%260, %83) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %263 = dfschedule.config.dma_bd(%262, %83, %c5_i32, %c0_i32, %c7_i32, %c9_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %264 = dfschedule.config.dma_bd(%261, %83, %c4_i32, %c0_i32, %c7_i32, %c9_i32, %263) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %265 = dfschedule.config.create_io(%264, %83) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %266 = dfschedule.schedule.getbdid(%83) : (!dfschedule.tile) -> i32
    %subview_30 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %267 = dfschedule.memref_mapping %subview_30 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %268 = dfschedule.bind_core_buffer(%267, %119) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %269 = dfschedule.bind_core_buffer(%267, %119) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %270 = dfschedule.config.dma_bd(%269, %119, %c5_i32, %c0_i32, %c8_i32, %c10_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %271 = dfschedule.config.dma_bd(%268, %119, %c4_i32, %c0_i32, %c8_i32, %c10_i32, %270) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %272 = dfschedule.config.create_io(%271, %119) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %273 = dfschedule.schedule.getbdid(%119) : (!dfschedule.tile) -> i32
    %274 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
    %275 = dfschedule.schedule.start_io(%245, %274) {flow_index = 7 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %subview_31 = memref.subview %arg0[112, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %276 = dfschedule.config.dma_bd(%subview_31, %72, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
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
    %277 = dfschedule.config.create_io(%276, %72) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_32 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %278 = dfschedule.memref_mapping %subview_32 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %279 = dfschedule.bind_core_buffer(%278, %19) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %280 = dfschedule.bind_core_buffer(%278, %19) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %281 = dfschedule.config.dma_bd(%280, %19, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %282 = dfschedule.config.dma_bd(%279, %19, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %281) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %283 = dfschedule.config.create_io(%282, %19) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %284 = dfschedule.schedule.getbdid(%19) : (!dfschedule.tile) -> i32
    %subview_33 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %285 = dfschedule.memref_mapping %subview_33 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %286 = dfschedule.bind_core_buffer(%285, %55) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %287 = dfschedule.bind_core_buffer(%285, %55) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %288 = dfschedule.config.dma_bd(%287, %55, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %289 = dfschedule.config.dma_bd(%286, %55, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %288) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %290 = dfschedule.config.create_io(%289, %55) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %291 = dfschedule.schedule.getbdid(%55) : (!dfschedule.tile) -> i32
    %subview_34 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %292 = dfschedule.memref_mapping %subview_34 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %293 = dfschedule.bind_core_buffer(%292, %91) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %294 = dfschedule.bind_core_buffer(%292, %91) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %295 = dfschedule.config.dma_bd(%294, %91, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %296 = dfschedule.config.dma_bd(%293, %91, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %295) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %297 = dfschedule.config.create_io(%296, %91) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %298 = dfschedule.schedule.getbdid(%91) : (!dfschedule.tile) -> i32
    %subview_35 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %299 = dfschedule.memref_mapping %subview_35 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %300 = dfschedule.bind_core_buffer(%299, %127) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %301 = dfschedule.bind_core_buffer(%299, %127) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %302 = dfschedule.config.dma_bd(%301, %127, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %303 = dfschedule.config.dma_bd(%300, %127, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %302) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %304 = dfschedule.config.create_io(%303, %127) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %305 = dfschedule.schedule.getbdid(%127) : (!dfschedule.tile) -> i32
    %306 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    %subview_36 = memref.subview %arg2[56, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %307 = dfschedule.config.dma_bd(%subview_36, %72, %c6_i32, %c37632_i32, %c12_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %308 = dfschedule.config.dma_bd(%subview_36, %72, %c5_i32, %c25088_i32, %c11_i32, %c-1_i32, %307) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %309 = dfschedule.config.dma_bd(%subview_36, %72, %c4_i32, %c12544_i32, %c10_i32, %c-1_i32, %308) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %310 = dfschedule.config.dma_bd(%subview_36, %72, %c3_i32, %c0_i32, %c9_i32, %c-1_i32, %309) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %311 = dfschedule.config.create_io(%310, %72) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = true
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_37 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %312 = dfschedule.memref_mapping %subview_37 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %313 = dfschedule.bind_core_buffer(%312, %19) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %314 = dfschedule.bind_core_buffer(%312, %19) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %315 = dfschedule.config.dma_bd(%314, %19, %c5_i32, %c0_i32, %c9_i32, %c3_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %316 = dfschedule.config.dma_bd(%313, %19, %c4_i32, %c0_i32, %c9_i32, %c3_i32, %315) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %317 = dfschedule.config.create_io(%316, %19) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %318 = dfschedule.schedule.getbdid(%19) : (!dfschedule.tile) -> i32
    %subview_38 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %319 = dfschedule.memref_mapping %subview_38 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %320 = dfschedule.bind_core_buffer(%319, %55) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %321 = dfschedule.bind_core_buffer(%319, %55) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %322 = dfschedule.config.dma_bd(%321, %55, %c5_i32, %c0_i32, %c10_i32, %c4_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %323 = dfschedule.config.dma_bd(%320, %55, %c4_i32, %c0_i32, %c10_i32, %c4_i32, %322) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %324 = dfschedule.config.create_io(%323, %55) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %325 = dfschedule.schedule.getbdid(%55) : (!dfschedule.tile) -> i32
    %subview_39 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %326 = dfschedule.memref_mapping %subview_39 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %327 = dfschedule.bind_core_buffer(%326, %91) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %328 = dfschedule.bind_core_buffer(%326, %91) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %329 = dfschedule.config.dma_bd(%328, %91, %c5_i32, %c0_i32, %c11_i32, %c5_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %330 = dfschedule.config.dma_bd(%327, %91, %c4_i32, %c0_i32, %c11_i32, %c5_i32, %329) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %331 = dfschedule.config.create_io(%330, %91) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %332 = dfschedule.schedule.getbdid(%91) : (!dfschedule.tile) -> i32
    %subview_40 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %333 = dfschedule.memref_mapping %subview_40 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %334 = dfschedule.bind_core_buffer(%333, %127) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %335 = dfschedule.bind_core_buffer(%333, %127) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %336 = dfschedule.config.dma_bd(%335, %127, %c5_i32, %c0_i32, %c12_i32, %c6_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %337 = dfschedule.config.dma_bd(%334, %127, %c4_i32, %c0_i32, %c12_i32, %c6_i32, %336) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %338 = dfschedule.config.create_io(%337, %127) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %339 = dfschedule.schedule.getbdid(%127) : (!dfschedule.tile) -> i32
    %340 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    %341 = dfschedule.schedule.start_io(%311, %340) {flow_index = 9 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %subview_41 = memref.subview %arg0[168, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>>
    %342 = dfschedule.config.dma_bd(%subview_41, %108, %c11_i32, %c0_i32, %c0_i32, %c-1_i32) {
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
    %343 = dfschedule.config.create_io(%342, %108) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_42 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %344 = dfschedule.memref_mapping %subview_42 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %345 = dfschedule.bind_core_buffer(%344, %27) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %346 = dfschedule.bind_core_buffer(%344, %27) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %347 = dfschedule.config.dma_bd(%346, %27, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %348 = dfschedule.config.dma_bd(%345, %27, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %347) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %349 = dfschedule.config.create_io(%348, %27) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %350 = dfschedule.schedule.getbdid(%27) : (!dfschedule.tile) -> i32
    %subview_43 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %351 = dfschedule.memref_mapping %subview_43 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %352 = dfschedule.bind_core_buffer(%351, %63) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %353 = dfschedule.bind_core_buffer(%351, %63) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %354 = dfschedule.config.dma_bd(%353, %63, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %355 = dfschedule.config.dma_bd(%352, %63, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %354) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %356 = dfschedule.config.create_io(%355, %63) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %357 = dfschedule.schedule.getbdid(%63) : (!dfschedule.tile) -> i32
    %subview_44 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %358 = dfschedule.memref_mapping %subview_44 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %359 = dfschedule.bind_core_buffer(%358, %99) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %360 = dfschedule.bind_core_buffer(%358, %99) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %361 = dfschedule.config.dma_bd(%360, %99, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %362 = dfschedule.config.dma_bd(%359, %99, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %361) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %363 = dfschedule.config.create_io(%362, %99) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %364 = dfschedule.schedule.getbdid(%99) : (!dfschedule.tile) -> i32
    %subview_45 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %365 = dfschedule.memref_mapping %subview_45 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %366 = dfschedule.bind_core_buffer(%365, %135) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %367 = dfschedule.bind_core_buffer(%365, %135) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %368 = dfschedule.config.dma_bd(%367, %135, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %369 = dfschedule.config.dma_bd(%366, %135, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %368) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %370 = dfschedule.config.create_io(%369, %135) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %371 = dfschedule.schedule.getbdid(%135) : (!dfschedule.tile) -> i32
    %372 = dfschedule.schedule.getbdid(%108) : (!dfschedule.tile) -> i32
    %subview_46 = memref.subview %arg2[84, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %373 = dfschedule.config.dma_bd(%subview_46, %72, %c11_i32, %c37632_i32, %c16_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %374 = dfschedule.config.dma_bd(%subview_46, %72, %c10_i32, %c25088_i32, %c15_i32, %c-1_i32, %373) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %375 = dfschedule.config.dma_bd(%subview_46, %72, %c9_i32, %c12544_i32, %c14_i32, %c-1_i32, %374) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %376 = dfschedule.config.dma_bd(%subview_46, %72, %c8_i32, %c0_i32, %c13_i32, %c-1_i32, %375) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = -1 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = -1 : i32,
      release_lock_val = 0 : i32,
      data_id = 2 : i32,
      iter_step_size = 0 : i32,
      iter_wrap = 4 : i32
    } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %377 = dfschedule.config.create_io(%376, %72) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = true
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %subview_47 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %378 = dfschedule.memref_mapping %subview_47 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %379 = dfschedule.bind_core_buffer(%378, %27) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %380 = dfschedule.bind_core_buffer(%378, %27) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %381 = dfschedule.config.dma_bd(%380, %27, %c5_i32, %c0_i32, %c13_i32, %c8_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %382 = dfschedule.config.dma_bd(%379, %27, %c4_i32, %c0_i32, %c13_i32, %c8_i32, %381) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %383 = dfschedule.config.create_io(%382, %27) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %384 = dfschedule.schedule.getbdid(%27) : (!dfschedule.tile) -> i32
    %subview_48 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %385 = dfschedule.memref_mapping %subview_48 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %386 = dfschedule.bind_core_buffer(%385, %63) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %387 = dfschedule.bind_core_buffer(%385, %63) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %388 = dfschedule.config.dma_bd(%387, %63, %c5_i32, %c0_i32, %c14_i32, %c9_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %389 = dfschedule.config.dma_bd(%386, %63, %c4_i32, %c0_i32, %c14_i32, %c9_i32, %388) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %390 = dfschedule.config.create_io(%389, %63) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %391 = dfschedule.schedule.getbdid(%63) : (!dfschedule.tile) -> i32
    %subview_49 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %392 = dfschedule.memref_mapping %subview_49 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %393 = dfschedule.bind_core_buffer(%392, %99) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %394 = dfschedule.bind_core_buffer(%392, %99) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %395 = dfschedule.config.dma_bd(%394, %99, %c5_i32, %c0_i32, %c15_i32, %c10_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %396 = dfschedule.config.dma_bd(%393, %99, %c4_i32, %c0_i32, %c15_i32, %c10_i32, %395) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %397 = dfschedule.config.create_io(%396, %99) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %398 = dfschedule.schedule.getbdid(%99) : (!dfschedule.tile) -> i32
    %subview_50 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %399 = dfschedule.memref_mapping %subview_50 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %400 = dfschedule.bind_core_buffer(%399, %135) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %401 = dfschedule.bind_core_buffer(%399, %135) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %402 = dfschedule.config.dma_bd(%401, %135, %c5_i32, %c0_i32, %c16_i32, %c11_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %403 = dfschedule.config.dma_bd(%400, %135, %c4_i32, %c0_i32, %c16_i32, %c11_i32, %402) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %404 = dfschedule.config.create_io(%403, %135) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %405 = dfschedule.schedule.getbdid(%135) : (!dfschedule.tile) -> i32
    %406 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    %407 = dfschedule.schedule.start_io(%377, %406) {flow_index = 11 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %408 = dfschedule.config.load_kernel_group(%3, %11, %19, %27, %39, %47, %55, %63, %75, %83, %91, %99, %111, %119, %127, %135) {
      callee = [@dskernel_receiver]
    } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
    %409 = dfschedule.schedule.launch_kernel_group(%408) : (!dfschedule.kernelgroup) -> !dfschedule.event
    %410 = dfschedule.schedule.start_io(%9, %10) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %411 = dfschedule.schedule.start_io(%17, %18) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %412 = dfschedule.schedule.start_io(%25, %26) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %413 = dfschedule.schedule.start_io(%33, %34) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %414 = dfschedule.schedule.start_io(%45, %46) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %415 = dfschedule.schedule.start_io(%53, %54) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %416 = dfschedule.schedule.start_io(%61, %62) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %417 = dfschedule.schedule.start_io(%69, %70) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %418 = dfschedule.schedule.start_io(%81, %82) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %419 = dfschedule.schedule.start_io(%89, %90) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %420 = dfschedule.schedule.start_io(%97, %98) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %421 = dfschedule.schedule.start_io(%105, %106) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %422 = dfschedule.schedule.start_io(%117, %118) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %423 = dfschedule.schedule.start_io(%125, %126) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %424 = dfschedule.schedule.start_io(%133, %134) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %425 = dfschedule.schedule.start_io(%141, %142) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %426 = dfschedule.schedule.start_io(%151, %152) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %427 = dfschedule.schedule.start_io(%158, %159) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %428 = dfschedule.schedule.start_io(%165, %166) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %429 = dfschedule.schedule.start_io(%172, %173) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %430 = dfschedule.schedule.start_io(%185, %186) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %431 = dfschedule.schedule.start_io(%192, %193) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %432 = dfschedule.schedule.start_io(%199, %200) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %433 = dfschedule.schedule.start_io(%206, %207) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %434 = dfschedule.schedule.start_io(%217, %218) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %435 = dfschedule.schedule.start_io(%224, %225) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %436 = dfschedule.schedule.start_io(%231, %232) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %437 = dfschedule.schedule.start_io(%238, %239) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %438 = dfschedule.schedule.start_io(%251, %252) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %439 = dfschedule.schedule.start_io(%258, %259) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %440 = dfschedule.schedule.start_io(%265, %266) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %441 = dfschedule.schedule.start_io(%272, %273) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %442 = dfschedule.schedule.start_io(%283, %284) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %443 = dfschedule.schedule.start_io(%290, %291) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %444 = dfschedule.schedule.start_io(%297, %298) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %445 = dfschedule.schedule.start_io(%304, %305) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %446 = dfschedule.schedule.start_io(%317, %318) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %447 = dfschedule.schedule.start_io(%324, %325) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %448 = dfschedule.schedule.start_io(%331, %332) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %449 = dfschedule.schedule.start_io(%338, %339) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %450 = dfschedule.schedule.start_io(%349, %350) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %451 = dfschedule.schedule.start_io(%356, %357) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %452 = dfschedule.schedule.start_io(%363, %364) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %453 = dfschedule.schedule.start_io(%370, %371) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %454 = dfschedule.schedule.start_io(%383, %384) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %455 = dfschedule.schedule.start_io(%390, %391) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %456 = dfschedule.schedule.start_io(%397, %398) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %457 = dfschedule.schedule.start_io(%404, %405) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %458 = dfschedule.schedule.start_io(%2, %35) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %459 = dfschedule.schedule.start_io(%38, %71) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %460 = dfschedule.schedule.start_io(%74, %107) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %461 = dfschedule.schedule.start_io(%110, %143) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %462 = dfschedule.schedule.start_io(%145, %174) {flow_index = 4 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %463 = dfschedule.schedule.start_io(%211, %240) {flow_index = 6 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %464 = dfschedule.schedule.start_io(%277, %306) {flow_index = 8 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %465 = dfschedule.schedule.start_io(%343, %372) {flow_index = 10 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    dfschedule.schedule.wait(%409, %209, %275, %341, %407) : (!dfschedule.event, !dfschedule.event, !dfschedule.event, !dfschedule.event, !dfschedule.event)
  }
  dfschedule.dskernel_receiver @dskernel_receiver {
  }
}
