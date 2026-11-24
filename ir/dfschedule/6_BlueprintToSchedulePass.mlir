module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.execute_region {
      %0 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %subview = memref.subview %arg1[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
        %4 = dfschedule.declaretile {col = 0 : i32, row = 0 : i32} : !dfschedule.tile
        %c0_i32_0 = arith.constant 0 : i32
        %c0_i32_1 = arith.constant 0 : i32
        %c0_i32_2 = arith.constant 0 : i32
        %c-1_i32 = arith.constant -1 : i32
        %5 = dfschedule.config.dma_bd(%subview, %4, %c0_i32_0, %c0_i32_1, %c0_i32_2, %c-1_i32) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 4294967295 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = 0 : i32,
          release_lock_id = 0 : i32,
          release_lock_val = 0 : i32,
          data_id = 0 : i32
        } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %6 = dfschedule.config.create_io(%5, %4) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_3 = memref.subview %subview[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
        %8 = dfschedule.memref_mapping %subview_3 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_4 = arith.constant 1 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %c-1_i32_7 = arith.constant -1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_4, %c0_i32_5, %c0_i32_6, %c-1_i32_7) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_8 = arith.constant 0 : i32
        %c0_i32_9 = arith.constant 0 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %c-1_i32_11 = arith.constant -1 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_8, %c0_i32_9, %c0_i32_10, %c-1_i32_11, %11) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_12 = memref.subview %subview[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
        %16 = dfschedule.memref_mapping %subview_12 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
        %17 = dfschedule.bind_core_buffer(%16, %15) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %18 = dfschedule.bind_core_buffer(%16, %15) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_13 = arith.constant 1 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %c-1_i32_16 = arith.constant -1 : i32
        %19 = dfschedule.config.dma_bd(%18, %15, %c1_i32_13, %c0_i32_14, %c0_i32_15, %c-1_i32_16) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_17 = arith.constant 0 : i32
        %c0_i32_18 = arith.constant 0 : i32
        %c0_i32_19 = arith.constant 0 : i32
        %c-1_i32_20 = arith.constant -1 : i32
        %20 = dfschedule.config.dma_bd(%17, %15, %c0_i32_17, %c0_i32_18, %c0_i32_19, %c-1_i32_20, %19) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %21 = dfschedule.config.create_io(%20, %15) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %22 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
        %23 = dfschedule.declaretile {col = 0 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_21 = memref.subview %subview[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
        %24 = dfschedule.memref_mapping %subview_21 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
        %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_22 = arith.constant 1 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %c0_i32_24 = arith.constant 0 : i32
        %c-1_i32_25 = arith.constant -1 : i32
        %27 = dfschedule.config.dma_bd(%26, %23, %c1_i32_22, %c0_i32_23, %c0_i32_24, %c-1_i32_25) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_26 = arith.constant 0 : i32
        %c0_i32_27 = arith.constant 0 : i32
        %c0_i32_28 = arith.constant 0 : i32
        %c-1_i32_29 = arith.constant -1 : i32
        %28 = dfschedule.config.dma_bd(%25, %23, %c0_i32_26, %c0_i32_27, %c0_i32_28, %c-1_i32_29, %27) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %29 = dfschedule.config.create_io(%28, %23) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %30 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
        %31 = dfschedule.declaretile {col = 0 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_30 = memref.subview %subview[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
        %32 = dfschedule.memref_mapping %subview_30 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
        %33 = dfschedule.bind_core_buffer(%32, %31) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %34 = dfschedule.bind_core_buffer(%32, %31) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_31 = arith.constant 1 : i32
        %c0_i32_32 = arith.constant 0 : i32
        %c0_i32_33 = arith.constant 0 : i32
        %c-1_i32_34 = arith.constant -1 : i32
        %35 = dfschedule.config.dma_bd(%34, %31, %c1_i32_31, %c0_i32_32, %c0_i32_33, %c-1_i32_34) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_35 = arith.constant 0 : i32
        %c0_i32_36 = arith.constant 0 : i32
        %c0_i32_37 = arith.constant 0 : i32
        %c-1_i32_38 = arith.constant -1 : i32
        %36 = dfschedule.config.dma_bd(%33, %31, %c0_i32_35, %c0_i32_36, %c0_i32_37, %c-1_i32_38, %35) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %37 = dfschedule.config.create_io(%36, %31) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %38 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
        %39 = dfschedule.schedule.getbdid(%4) : (!dfschedule.tile) -> i32
        %40 = dfschedule.config.load_kernel_group(%7, %15, %23, %31) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %41 = dfschedule.schedule.launch_kernel_group(%40) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %42 = dfschedule.schedule.start_io(%13, %14) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %43 = dfschedule.schedule.start_io(%21, %22) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %44 = dfschedule.schedule.start_io(%29, %30) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %45 = dfschedule.schedule.start_io(%37, %38) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %46 = dfschedule.schedule.start_io(%6, %39) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%41) : (!dfschedule.event)
        dfschedule.free_device_mem %subview : memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
        "routing.yield"() : () -> ()
      }
      %1 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %subview = memref.subview %arg1[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>
        %4 = dfschedule.declaretile {col = 1 : i32, row = 0 : i32} : !dfschedule.tile
        %c0_i32_0 = arith.constant 0 : i32
        %c0_i32_1 = arith.constant 0 : i32
        %c0_i32_2 = arith.constant 0 : i32
        %c-1_i32 = arith.constant -1 : i32
        %5 = dfschedule.config.dma_bd(%subview, %4, %c0_i32_0, %c0_i32_1, %c0_i32_2, %c-1_i32) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 4294967295 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = 0 : i32,
          release_lock_id = 0 : i32,
          release_lock_val = 0 : i32,
          data_id = 0 : i32
        } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %6 = dfschedule.config.create_io(%5, %4) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_3 = memref.subview %subview[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
        %8 = dfschedule.memref_mapping %subview_3 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_4 = arith.constant 1 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %c-1_i32_7 = arith.constant -1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_4, %c0_i32_5, %c0_i32_6, %c-1_i32_7) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_8 = arith.constant 0 : i32
        %c0_i32_9 = arith.constant 0 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %c-1_i32_11 = arith.constant -1 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_8, %c0_i32_9, %c0_i32_10, %c-1_i32_11, %11) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_12 = memref.subview %subview[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
        %16 = dfschedule.memref_mapping %subview_12 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
        %17 = dfschedule.bind_core_buffer(%16, %15) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %18 = dfschedule.bind_core_buffer(%16, %15) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_13 = arith.constant 1 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %c-1_i32_16 = arith.constant -1 : i32
        %19 = dfschedule.config.dma_bd(%18, %15, %c1_i32_13, %c0_i32_14, %c0_i32_15, %c-1_i32_16) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_17 = arith.constant 0 : i32
        %c0_i32_18 = arith.constant 0 : i32
        %c0_i32_19 = arith.constant 0 : i32
        %c-1_i32_20 = arith.constant -1 : i32
        %20 = dfschedule.config.dma_bd(%17, %15, %c0_i32_17, %c0_i32_18, %c0_i32_19, %c-1_i32_20, %19) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %21 = dfschedule.config.create_io(%20, %15) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %22 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
        %23 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_21 = memref.subview %subview[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
        %24 = dfschedule.memref_mapping %subview_21 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
        %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_22 = arith.constant 1 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %c0_i32_24 = arith.constant 0 : i32
        %c-1_i32_25 = arith.constant -1 : i32
        %27 = dfschedule.config.dma_bd(%26, %23, %c1_i32_22, %c0_i32_23, %c0_i32_24, %c-1_i32_25) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_26 = arith.constant 0 : i32
        %c0_i32_27 = arith.constant 0 : i32
        %c0_i32_28 = arith.constant 0 : i32
        %c-1_i32_29 = arith.constant -1 : i32
        %28 = dfschedule.config.dma_bd(%25, %23, %c0_i32_26, %c0_i32_27, %c0_i32_28, %c-1_i32_29, %27) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %29 = dfschedule.config.create_io(%28, %23) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %30 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
        %31 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_30 = memref.subview %subview[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
        %32 = dfschedule.memref_mapping %subview_30 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
        %33 = dfschedule.bind_core_buffer(%32, %31) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %34 = dfschedule.bind_core_buffer(%32, %31) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_31 = arith.constant 1 : i32
        %c0_i32_32 = arith.constant 0 : i32
        %c0_i32_33 = arith.constant 0 : i32
        %c-1_i32_34 = arith.constant -1 : i32
        %35 = dfschedule.config.dma_bd(%34, %31, %c1_i32_31, %c0_i32_32, %c0_i32_33, %c-1_i32_34) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_35 = arith.constant 0 : i32
        %c0_i32_36 = arith.constant 0 : i32
        %c0_i32_37 = arith.constant 0 : i32
        %c-1_i32_38 = arith.constant -1 : i32
        %36 = dfschedule.config.dma_bd(%33, %31, %c0_i32_35, %c0_i32_36, %c0_i32_37, %c-1_i32_38, %35) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %37 = dfschedule.config.create_io(%36, %31) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %38 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
        %39 = dfschedule.schedule.getbdid(%4) : (!dfschedule.tile) -> i32
        %40 = dfschedule.config.load_kernel_group(%7, %15, %23, %31) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %41 = dfschedule.schedule.launch_kernel_group(%40) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %42 = dfschedule.schedule.start_io(%13, %14) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %43 = dfschedule.schedule.start_io(%21, %22) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %44 = dfschedule.schedule.start_io(%29, %30) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %45 = dfschedule.schedule.start_io(%37, %38) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %46 = dfschedule.schedule.start_io(%6, %39) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%41) : (!dfschedule.event)
        dfschedule.free_device_mem %subview : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>
        "routing.yield"() : () -> ()
      }
      %2 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %subview = memref.subview %arg1[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
        %4 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
        %c0_i32_0 = arith.constant 0 : i32
        %c0_i32_1 = arith.constant 0 : i32
        %c0_i32_2 = arith.constant 0 : i32
        %c-1_i32 = arith.constant -1 : i32
        %5 = dfschedule.config.dma_bd(%subview, %4, %c0_i32_0, %c0_i32_1, %c0_i32_2, %c-1_i32) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 4294967295 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = 0 : i32,
          release_lock_id = 0 : i32,
          release_lock_val = 0 : i32,
          data_id = 0 : i32
        } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %6 = dfschedule.config.create_io(%5, %4) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_3 = memref.subview %subview[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
        %8 = dfschedule.memref_mapping %subview_3 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_4 = arith.constant 1 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %c-1_i32_7 = arith.constant -1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_4, %c0_i32_5, %c0_i32_6, %c-1_i32_7) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_8 = arith.constant 0 : i32
        %c0_i32_9 = arith.constant 0 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %c-1_i32_11 = arith.constant -1 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_8, %c0_i32_9, %c0_i32_10, %c-1_i32_11, %11) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_12 = memref.subview %subview[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
        %16 = dfschedule.memref_mapping %subview_12 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
        %17 = dfschedule.bind_core_buffer(%16, %15) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %18 = dfschedule.bind_core_buffer(%16, %15) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_13 = arith.constant 1 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %c-1_i32_16 = arith.constant -1 : i32
        %19 = dfschedule.config.dma_bd(%18, %15, %c1_i32_13, %c0_i32_14, %c0_i32_15, %c-1_i32_16) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_17 = arith.constant 0 : i32
        %c0_i32_18 = arith.constant 0 : i32
        %c0_i32_19 = arith.constant 0 : i32
        %c-1_i32_20 = arith.constant -1 : i32
        %20 = dfschedule.config.dma_bd(%17, %15, %c0_i32_17, %c0_i32_18, %c0_i32_19, %c-1_i32_20, %19) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %21 = dfschedule.config.create_io(%20, %15) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %22 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
        %23 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_21 = memref.subview %subview[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
        %24 = dfschedule.memref_mapping %subview_21 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
        %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_22 = arith.constant 1 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %c0_i32_24 = arith.constant 0 : i32
        %c-1_i32_25 = arith.constant -1 : i32
        %27 = dfschedule.config.dma_bd(%26, %23, %c1_i32_22, %c0_i32_23, %c0_i32_24, %c-1_i32_25) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_26 = arith.constant 0 : i32
        %c0_i32_27 = arith.constant 0 : i32
        %c0_i32_28 = arith.constant 0 : i32
        %c-1_i32_29 = arith.constant -1 : i32
        %28 = dfschedule.config.dma_bd(%25, %23, %c0_i32_26, %c0_i32_27, %c0_i32_28, %c-1_i32_29, %27) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %29 = dfschedule.config.create_io(%28, %23) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %30 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
        %31 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_30 = memref.subview %subview[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
        %32 = dfschedule.memref_mapping %subview_30 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
        %33 = dfschedule.bind_core_buffer(%32, %31) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %34 = dfschedule.bind_core_buffer(%32, %31) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_31 = arith.constant 1 : i32
        %c0_i32_32 = arith.constant 0 : i32
        %c0_i32_33 = arith.constant 0 : i32
        %c-1_i32_34 = arith.constant -1 : i32
        %35 = dfschedule.config.dma_bd(%34, %31, %c1_i32_31, %c0_i32_32, %c0_i32_33, %c-1_i32_34) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_35 = arith.constant 0 : i32
        %c0_i32_36 = arith.constant 0 : i32
        %c0_i32_37 = arith.constant 0 : i32
        %c-1_i32_38 = arith.constant -1 : i32
        %36 = dfschedule.config.dma_bd(%33, %31, %c0_i32_35, %c0_i32_36, %c0_i32_37, %c-1_i32_38, %35) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %37 = dfschedule.config.create_io(%36, %31) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %38 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
        %39 = dfschedule.schedule.getbdid(%4) : (!dfschedule.tile) -> i32
        %40 = dfschedule.config.load_kernel_group(%7, %15, %23, %31) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %41 = dfschedule.schedule.launch_kernel_group(%40) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %42 = dfschedule.schedule.start_io(%13, %14) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %43 = dfschedule.schedule.start_io(%21, %22) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %44 = dfschedule.schedule.start_io(%29, %30) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %45 = dfschedule.schedule.start_io(%37, %38) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %46 = dfschedule.schedule.start_io(%6, %39) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%41) : (!dfschedule.event)
        dfschedule.free_device_mem %subview : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
        "routing.yield"() : () -> ()
      }
      %3 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %subview = memref.subview %arg1[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>
        %4 = dfschedule.declaretile {col = 3 : i32, row = 0 : i32} : !dfschedule.tile
        %c0_i32_0 = arith.constant 0 : i32
        %c0_i32_1 = arith.constant 0 : i32
        %c0_i32_2 = arith.constant 0 : i32
        %c-1_i32 = arith.constant -1 : i32
        %5 = dfschedule.config.dma_bd(%subview, %4, %c0_i32_0, %c0_i32_1, %c0_i32_2, %c-1_i32) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 4294967295 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = 0 : i32,
          release_lock_id = 0 : i32,
          release_lock_val = 0 : i32,
          data_id = 0 : i32
        } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %6 = dfschedule.config.create_io(%5, %4) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_3 = memref.subview %subview[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
        %8 = dfschedule.memref_mapping %subview_3 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_4 = arith.constant 1 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %c-1_i32_7 = arith.constant -1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_4, %c0_i32_5, %c0_i32_6, %c-1_i32_7) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_8 = arith.constant 0 : i32
        %c0_i32_9 = arith.constant 0 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %c-1_i32_11 = arith.constant -1 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_8, %c0_i32_9, %c0_i32_10, %c-1_i32_11, %11) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_12 = memref.subview %subview[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
        %16 = dfschedule.memref_mapping %subview_12 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
        %17 = dfschedule.bind_core_buffer(%16, %15) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %18 = dfschedule.bind_core_buffer(%16, %15) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_13 = arith.constant 1 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %c-1_i32_16 = arith.constant -1 : i32
        %19 = dfschedule.config.dma_bd(%18, %15, %c1_i32_13, %c0_i32_14, %c0_i32_15, %c-1_i32_16) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_17 = arith.constant 0 : i32
        %c0_i32_18 = arith.constant 0 : i32
        %c0_i32_19 = arith.constant 0 : i32
        %c-1_i32_20 = arith.constant -1 : i32
        %20 = dfschedule.config.dma_bd(%17, %15, %c0_i32_17, %c0_i32_18, %c0_i32_19, %c-1_i32_20, %19) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %21 = dfschedule.config.create_io(%20, %15) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %22 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
        %23 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_21 = memref.subview %subview[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
        %24 = dfschedule.memref_mapping %subview_21 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
        %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_22 = arith.constant 1 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %c0_i32_24 = arith.constant 0 : i32
        %c-1_i32_25 = arith.constant -1 : i32
        %27 = dfschedule.config.dma_bd(%26, %23, %c1_i32_22, %c0_i32_23, %c0_i32_24, %c-1_i32_25) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_26 = arith.constant 0 : i32
        %c0_i32_27 = arith.constant 0 : i32
        %c0_i32_28 = arith.constant 0 : i32
        %c-1_i32_29 = arith.constant -1 : i32
        %28 = dfschedule.config.dma_bd(%25, %23, %c0_i32_26, %c0_i32_27, %c0_i32_28, %c-1_i32_29, %27) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %29 = dfschedule.config.create_io(%28, %23) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %30 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
        %31 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_30 = memref.subview %subview[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
        %32 = dfschedule.memref_mapping %subview_30 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
        %33 = dfschedule.bind_core_buffer(%32, %31) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %34 = dfschedule.bind_core_buffer(%32, %31) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
        %c1_i32_31 = arith.constant 1 : i32
        %c0_i32_32 = arith.constant 0 : i32
        %c0_i32_33 = arith.constant 0 : i32
        %c-1_i32_34 = arith.constant -1 : i32
        %35 = dfschedule.config.dma_bd(%34, %31, %c1_i32_31, %c0_i32_32, %c0_i32_33, %c-1_i32_34) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 0 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c0_i32_35 = arith.constant 0 : i32
        %c0_i32_36 = arith.constant 0 : i32
        %c0_i32_37 = arith.constant 0 : i32
        %c-1_i32_38 = arith.constant -1 : i32
        %36 = dfschedule.config.dma_bd(%33, %31, %c0_i32_35, %c0_i32_36, %c0_i32_37, %c-1_i32_38, %35) {
          len = 3136 : i32,
          enable_packet = false,
          next_bd = 1 : i32,
          acquire_lock_id = 2 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 3 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %37 = dfschedule.config.create_io(%36, %31) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %38 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
        %39 = dfschedule.schedule.getbdid(%4) : (!dfschedule.tile) -> i32
        %40 = dfschedule.config.load_kernel_group(%7, %15, %23, %31) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %41 = dfschedule.schedule.launch_kernel_group(%40) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %42 = dfschedule.schedule.start_io(%13, %14) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %43 = dfschedule.schedule.start_io(%21, %22) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %44 = dfschedule.schedule.start_io(%29, %30) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %45 = dfschedule.schedule.start_io(%37, %38) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %46 = dfschedule.schedule.start_io(%6, %39) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%41) : (!dfschedule.event)
        dfschedule.free_device_mem %subview : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %0 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %subview = memref.subview %arg0[0, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1]>>
        %4 = dfschedule.declaretile {col = 0 : i32, row = 0 : i32} : !dfschedule.tile
        %c1_i32_0 = arith.constant 1 : i32
        %c0_i32_1 = arith.constant 0 : i32
        %c0_i32_2 = arith.constant 0 : i32
        %c-1_i32 = arith.constant -1 : i32
        %5 = dfschedule.config.dma_bd(%subview, %4, %c1_i32_0, %c0_i32_1, %c0_i32_2, %c-1_i32) {
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
        %6 = dfschedule.config.create_io(%5, %4) {
          channel = 1,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_3 = memref.subview %subview[0, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1]>> to memref<61x230x4xi8, strided<[920, 4, 1]>>
        %8 = dfschedule.memref_mapping %subview_3 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_4 = arith.constant 3 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %c-1_i32_7 = arith.constant -1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c3_i32_4, %c0_i32_5, %c0_i32_6, %c-1_i32_7) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_8 = arith.constant 2 : i32
        %c0_i32_9 = arith.constant 0 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %c-1_i32_11 = arith.constant -1 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c2_i32_8, %c0_i32_9, %c0_i32_10, %c-1_i32_11, %11) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_12 = memref.subview %subview[0, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1]>> to memref<61x230x4xi8, strided<[920, 4, 1]>>
        %16 = dfschedule.memref_mapping %subview_12 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
        %17 = dfschedule.bind_core_buffer(%16, %15) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %18 = dfschedule.bind_core_buffer(%16, %15) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_13 = arith.constant 3 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %c-1_i32_16 = arith.constant -1 : i32
        %19 = dfschedule.config.dma_bd(%18, %15, %c3_i32_13, %c0_i32_14, %c0_i32_15, %c-1_i32_16) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_17 = arith.constant 2 : i32
        %c0_i32_18 = arith.constant 0 : i32
        %c0_i32_19 = arith.constant 0 : i32
        %c-1_i32_20 = arith.constant -1 : i32
        %20 = dfschedule.config.dma_bd(%17, %15, %c2_i32_17, %c0_i32_18, %c0_i32_19, %c-1_i32_20, %19) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %21 = dfschedule.config.create_io(%20, %15) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %22 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
        %23 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_21 = memref.subview %subview[0, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1]>> to memref<61x230x4xi8, strided<[920, 4, 1]>>
        %24 = dfschedule.memref_mapping %subview_21 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
        %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_22 = arith.constant 3 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %c0_i32_24 = arith.constant 0 : i32
        %c-1_i32_25 = arith.constant -1 : i32
        %27 = dfschedule.config.dma_bd(%26, %23, %c3_i32_22, %c0_i32_23, %c0_i32_24, %c-1_i32_25) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_26 = arith.constant 2 : i32
        %c0_i32_27 = arith.constant 0 : i32
        %c0_i32_28 = arith.constant 0 : i32
        %c-1_i32_29 = arith.constant -1 : i32
        %28 = dfschedule.config.dma_bd(%25, %23, %c2_i32_26, %c0_i32_27, %c0_i32_28, %c-1_i32_29, %27) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %29 = dfschedule.config.create_io(%28, %23) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %30 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
        %31 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_30 = memref.subview %subview[0, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1]>> to memref<61x230x4xi8, strided<[920, 4, 1]>>
        %32 = dfschedule.memref_mapping %subview_30 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
        %33 = dfschedule.bind_core_buffer(%32, %31) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %34 = dfschedule.bind_core_buffer(%32, %31) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_31 = arith.constant 3 : i32
        %c0_i32_32 = arith.constant 0 : i32
        %c0_i32_33 = arith.constant 0 : i32
        %c-1_i32_34 = arith.constant -1 : i32
        %35 = dfschedule.config.dma_bd(%34, %31, %c3_i32_31, %c0_i32_32, %c0_i32_33, %c-1_i32_34) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_35 = arith.constant 2 : i32
        %c0_i32_36 = arith.constant 0 : i32
        %c0_i32_37 = arith.constant 0 : i32
        %c-1_i32_38 = arith.constant -1 : i32
        %36 = dfschedule.config.dma_bd(%33, %31, %c2_i32_35, %c0_i32_36, %c0_i32_37, %c-1_i32_38, %35) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %37 = dfschedule.config.create_io(%36, %31) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %38 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
        %39 = dfschedule.schedule.getbdid(%4) : (!dfschedule.tile) -> i32
        %40 = dfschedule.config.load_kernel_group(%7, %15, %23, %31) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %41 = dfschedule.schedule.launch_kernel_group(%40) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %42 = dfschedule.schedule.start_io(%13, %14) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %43 = dfschedule.schedule.start_io(%21, %22) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %44 = dfschedule.schedule.start_io(%29, %30) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %45 = dfschedule.schedule.start_io(%37, %38) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %46 = dfschedule.schedule.start_io(%6, %39) {flow_index = 4 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%41) : (!dfschedule.event)
        dfschedule.free_device_mem %subview : memref<61x230x4xi8, strided<[920, 4, 1]>>
        %subview_39 = memref.subview %arg2[0, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1]>>
        %47 = dfschedule.declaretile {col = 3 : i32, row = 0 : i32} : !dfschedule.tile
        %c1_i32_40 = arith.constant 1 : i32
        %c5_i32 = arith.constant 5 : i32
        %c37632_i32 = arith.constant 37632 : i32
        %c4_i32 = arith.constant 4 : i32
        %c-1_i32_41 = arith.constant -1 : i32
        %48 = dfschedule.config.dma_bd(%subview_39, %47, %c5_i32, %c37632_i32, %c4_i32, %c-1_i32_41) {
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
        %c4_i32_42 = arith.constant 4 : i32
        %c25088_i32 = arith.constant 25088 : i32
        %c3_i32_43 = arith.constant 3 : i32
        %c-1_i32_44 = arith.constant -1 : i32
        %49 = dfschedule.config.dma_bd(%subview_39, %47, %c4_i32_42, %c25088_i32, %c3_i32_43, %c-1_i32_44, %48) {
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
        %c3_i32_45 = arith.constant 3 : i32
        %c12544_i32 = arith.constant 12544 : i32
        %c2_i32_46 = arith.constant 2 : i32
        %c-1_i32_47 = arith.constant -1 : i32
        %50 = dfschedule.config.dma_bd(%subview_39, %47, %c3_i32_45, %c12544_i32, %c2_i32_46, %c-1_i32_47, %49) {
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
        %c2_i32_48 = arith.constant 2 : i32
        %c0_i32_49 = arith.constant 0 : i32
        %c1_i32_50 = arith.constant 1 : i32
        %c-1_i32_51 = arith.constant -1 : i32
        %51 = dfschedule.config.dma_bd(%subview_39, %47, %c2_i32_48, %c0_i32_49, %c1_i32_50, %c-1_i32_51, %50) {
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
        %52 = dfschedule.config.create_io(%51, %47) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = true
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %53 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_52 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
        %54 = dfschedule.memref_mapping %subview_52 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
        %55 = dfschedule.bind_core_buffer(%54, %53) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %56 = dfschedule.bind_core_buffer(%54, %53) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_53 = arith.constant 5 : i32
        %c0_i32_54 = arith.constant 0 : i32
        %c1_i32_55 = arith.constant 1 : i32
        %c2_i32_56 = arith.constant 2 : i32
        %57 = dfschedule.config.dma_bd(%56, %53, %c5_i32_53, %c0_i32_54, %c1_i32_55, %c2_i32_56) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_57 = arith.constant 4 : i32
        %c0_i32_58 = arith.constant 0 : i32
        %c1_i32_59 = arith.constant 1 : i32
        %c2_i32_60 = arith.constant 2 : i32
        %58 = dfschedule.config.dma_bd(%55, %53, %c4_i32_57, %c0_i32_58, %c1_i32_59, %c2_i32_60, %57) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %59 = dfschedule.config.create_io(%58, %53) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %60 = dfschedule.schedule.getbdid(%53) : (!dfschedule.tile) -> i32
        %61 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_61 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
        %62 = dfschedule.memref_mapping %subview_61 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
        %63 = dfschedule.bind_core_buffer(%62, %61) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %64 = dfschedule.bind_core_buffer(%62, %61) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_62 = arith.constant 5 : i32
        %c0_i32_63 = arith.constant 0 : i32
        %c2_i32_64 = arith.constant 2 : i32
        %c3_i32_65 = arith.constant 3 : i32
        %65 = dfschedule.config.dma_bd(%64, %61, %c5_i32_62, %c0_i32_63, %c2_i32_64, %c3_i32_65) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_66 = arith.constant 4 : i32
        %c0_i32_67 = arith.constant 0 : i32
        %c2_i32_68 = arith.constant 2 : i32
        %c3_i32_69 = arith.constant 3 : i32
        %66 = dfschedule.config.dma_bd(%63, %61, %c4_i32_66, %c0_i32_67, %c2_i32_68, %c3_i32_69, %65) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %67 = dfschedule.config.create_io(%66, %61) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %68 = dfschedule.schedule.getbdid(%61) : (!dfschedule.tile) -> i32
        %69 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_70 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
        %70 = dfschedule.memref_mapping %subview_70 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
        %71 = dfschedule.bind_core_buffer(%70, %69) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %72 = dfschedule.bind_core_buffer(%70, %69) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_71 = arith.constant 5 : i32
        %c0_i32_72 = arith.constant 0 : i32
        %c3_i32_73 = arith.constant 3 : i32
        %c4_i32_74 = arith.constant 4 : i32
        %73 = dfschedule.config.dma_bd(%72, %69, %c5_i32_71, %c0_i32_72, %c3_i32_73, %c4_i32_74) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_75 = arith.constant 4 : i32
        %c0_i32_76 = arith.constant 0 : i32
        %c3_i32_77 = arith.constant 3 : i32
        %c4_i32_78 = arith.constant 4 : i32
        %74 = dfschedule.config.dma_bd(%71, %69, %c4_i32_75, %c0_i32_76, %c3_i32_77, %c4_i32_78, %73) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %75 = dfschedule.config.create_io(%74, %69) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %76 = dfschedule.schedule.getbdid(%69) : (!dfschedule.tile) -> i32
        %77 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_79 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
        %78 = dfschedule.memref_mapping %subview_79 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
        %79 = dfschedule.bind_core_buffer(%78, %77) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %80 = dfschedule.bind_core_buffer(%78, %77) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_80 = arith.constant 5 : i32
        %c0_i32_81 = arith.constant 0 : i32
        %c4_i32_82 = arith.constant 4 : i32
        %c5_i32_83 = arith.constant 5 : i32
        %81 = dfschedule.config.dma_bd(%80, %77, %c5_i32_80, %c0_i32_81, %c4_i32_82, %c5_i32_83) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_84 = arith.constant 4 : i32
        %c0_i32_85 = arith.constant 0 : i32
        %c4_i32_86 = arith.constant 4 : i32
        %c5_i32_87 = arith.constant 5 : i32
        %82 = dfschedule.config.dma_bd(%79, %77, %c4_i32_84, %c0_i32_85, %c4_i32_86, %c5_i32_87, %81) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %83 = dfschedule.config.create_io(%82, %77) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %84 = dfschedule.schedule.getbdid(%77) : (!dfschedule.tile) -> i32
        %85 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
        %86 = dfschedule.config.load_kernel_group(%53, %61, %69, %77) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %87 = dfschedule.schedule.launch_kernel_group(%86) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %88 = dfschedule.schedule.start_io(%59, %60) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %89 = dfschedule.schedule.start_io(%67, %68) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %90 = dfschedule.schedule.start_io(%75, %76) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %91 = dfschedule.schedule.start_io(%83, %84) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %92 = dfschedule.schedule.start_io(%52, %85) {flow_index = 5 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%87, %92) : (!dfschedule.event, !dfschedule.event)
        dfschedule.free_device_mem %subview_39 : memref<28x112x16xi8, strided<[7168, 64, 1]>>
        "routing.yield"() : () -> ()
      }
      %1 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %subview = memref.subview %arg0[56, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>>
        %4 = dfschedule.declaretile {col = 1 : i32, row = 0 : i32} : !dfschedule.tile
        %c1_i32_0 = arith.constant 1 : i32
        %c0_i32_1 = arith.constant 0 : i32
        %c0_i32_2 = arith.constant 0 : i32
        %c-1_i32 = arith.constant -1 : i32
        %5 = dfschedule.config.dma_bd(%subview, %4, %c1_i32_0, %c0_i32_1, %c0_i32_2, %c-1_i32) {
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
        %6 = dfschedule.config.create_io(%5, %4) {
          channel = 1,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_3 = memref.subview %subview[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
        %8 = dfschedule.memref_mapping %subview_3 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_4 = arith.constant 3 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %c-1_i32_7 = arith.constant -1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c3_i32_4, %c0_i32_5, %c0_i32_6, %c-1_i32_7) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_8 = arith.constant 2 : i32
        %c0_i32_9 = arith.constant 0 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %c-1_i32_11 = arith.constant -1 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c2_i32_8, %c0_i32_9, %c0_i32_10, %c-1_i32_11, %11) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_12 = memref.subview %subview[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
        %16 = dfschedule.memref_mapping %subview_12 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
        %17 = dfschedule.bind_core_buffer(%16, %15) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %18 = dfschedule.bind_core_buffer(%16, %15) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_13 = arith.constant 3 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %c-1_i32_16 = arith.constant -1 : i32
        %19 = dfschedule.config.dma_bd(%18, %15, %c3_i32_13, %c0_i32_14, %c0_i32_15, %c-1_i32_16) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_17 = arith.constant 2 : i32
        %c0_i32_18 = arith.constant 0 : i32
        %c0_i32_19 = arith.constant 0 : i32
        %c-1_i32_20 = arith.constant -1 : i32
        %20 = dfschedule.config.dma_bd(%17, %15, %c2_i32_17, %c0_i32_18, %c0_i32_19, %c-1_i32_20, %19) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %21 = dfschedule.config.create_io(%20, %15) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %22 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
        %23 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_21 = memref.subview %subview[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
        %24 = dfschedule.memref_mapping %subview_21 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
        %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_22 = arith.constant 3 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %c0_i32_24 = arith.constant 0 : i32
        %c-1_i32_25 = arith.constant -1 : i32
        %27 = dfschedule.config.dma_bd(%26, %23, %c3_i32_22, %c0_i32_23, %c0_i32_24, %c-1_i32_25) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_26 = arith.constant 2 : i32
        %c0_i32_27 = arith.constant 0 : i32
        %c0_i32_28 = arith.constant 0 : i32
        %c-1_i32_29 = arith.constant -1 : i32
        %28 = dfschedule.config.dma_bd(%25, %23, %c2_i32_26, %c0_i32_27, %c0_i32_28, %c-1_i32_29, %27) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %29 = dfschedule.config.create_io(%28, %23) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %30 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
        %31 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_30 = memref.subview %subview[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
        %32 = dfschedule.memref_mapping %subview_30 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
        %33 = dfschedule.bind_core_buffer(%32, %31) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %34 = dfschedule.bind_core_buffer(%32, %31) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_31 = arith.constant 3 : i32
        %c0_i32_32 = arith.constant 0 : i32
        %c0_i32_33 = arith.constant 0 : i32
        %c-1_i32_34 = arith.constant -1 : i32
        %35 = dfschedule.config.dma_bd(%34, %31, %c3_i32_31, %c0_i32_32, %c0_i32_33, %c-1_i32_34) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_35 = arith.constant 2 : i32
        %c0_i32_36 = arith.constant 0 : i32
        %c0_i32_37 = arith.constant 0 : i32
        %c-1_i32_38 = arith.constant -1 : i32
        %36 = dfschedule.config.dma_bd(%33, %31, %c2_i32_35, %c0_i32_36, %c0_i32_37, %c-1_i32_38, %35) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %37 = dfschedule.config.create_io(%36, %31) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %38 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
        %39 = dfschedule.schedule.getbdid(%4) : (!dfschedule.tile) -> i32
        %40 = dfschedule.config.load_kernel_group(%7, %15, %23, %31) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %41 = dfschedule.schedule.launch_kernel_group(%40) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %42 = dfschedule.schedule.start_io(%13, %14) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %43 = dfschedule.schedule.start_io(%21, %22) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %44 = dfschedule.schedule.start_io(%29, %30) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %45 = dfschedule.schedule.start_io(%37, %38) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %46 = dfschedule.schedule.start_io(%6, %39) {flow_index = 6 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%41) : (!dfschedule.event)
        dfschedule.free_device_mem %subview : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>>
        %subview_39 = memref.subview %arg2[28, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>
        %47 = dfschedule.declaretile {col = 3 : i32, row = 0 : i32} : !dfschedule.tile
        %c6_i32 = arith.constant 6 : i32
        %c10_i32 = arith.constant 10 : i32
        %c37632_i32 = arith.constant 37632 : i32
        %c8_i32 = arith.constant 8 : i32
        %c-1_i32_40 = arith.constant -1 : i32
        %48 = dfschedule.config.dma_bd(%subview_39, %47, %c10_i32, %c37632_i32, %c8_i32, %c-1_i32_40) {
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
        %c9_i32 = arith.constant 9 : i32
        %c25088_i32 = arith.constant 25088 : i32
        %c7_i32 = arith.constant 7 : i32
        %c-1_i32_41 = arith.constant -1 : i32
        %49 = dfschedule.config.dma_bd(%subview_39, %47, %c9_i32, %c25088_i32, %c7_i32, %c-1_i32_41, %48) {
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
        %c8_i32_42 = arith.constant 8 : i32
        %c12544_i32 = arith.constant 12544 : i32
        %c6_i32_43 = arith.constant 6 : i32
        %c-1_i32_44 = arith.constant -1 : i32
        %50 = dfschedule.config.dma_bd(%subview_39, %47, %c8_i32_42, %c12544_i32, %c6_i32_43, %c-1_i32_44, %49) {
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
        %c7_i32_45 = arith.constant 7 : i32
        %c0_i32_46 = arith.constant 0 : i32
        %c5_i32 = arith.constant 5 : i32
        %c-1_i32_47 = arith.constant -1 : i32
        %51 = dfschedule.config.dma_bd(%subview_39, %47, %c7_i32_45, %c0_i32_46, %c5_i32, %c-1_i32_47, %50) {
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
        %52 = dfschedule.config.create_io(%51, %47) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = true
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %53 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_48 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
        %54 = dfschedule.memref_mapping %subview_48 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
        %55 = dfschedule.bind_core_buffer(%54, %53) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %56 = dfschedule.bind_core_buffer(%54, %53) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_49 = arith.constant 5 : i32
        %c0_i32_50 = arith.constant 0 : i32
        %c5_i32_51 = arith.constant 5 : i32
        %c7_i32_52 = arith.constant 7 : i32
        %57 = dfschedule.config.dma_bd(%56, %53, %c5_i32_49, %c0_i32_50, %c5_i32_51, %c7_i32_52) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32 = arith.constant 4 : i32
        %c0_i32_53 = arith.constant 0 : i32
        %c5_i32_54 = arith.constant 5 : i32
        %c7_i32_55 = arith.constant 7 : i32
        %58 = dfschedule.config.dma_bd(%55, %53, %c4_i32, %c0_i32_53, %c5_i32_54, %c7_i32_55, %57) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %59 = dfschedule.config.create_io(%58, %53) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %60 = dfschedule.schedule.getbdid(%53) : (!dfschedule.tile) -> i32
        %61 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_56 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
        %62 = dfschedule.memref_mapping %subview_56 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
        %63 = dfschedule.bind_core_buffer(%62, %61) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %64 = dfschedule.bind_core_buffer(%62, %61) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_57 = arith.constant 5 : i32
        %c0_i32_58 = arith.constant 0 : i32
        %c6_i32_59 = arith.constant 6 : i32
        %c8_i32_60 = arith.constant 8 : i32
        %65 = dfschedule.config.dma_bd(%64, %61, %c5_i32_57, %c0_i32_58, %c6_i32_59, %c8_i32_60) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_61 = arith.constant 4 : i32
        %c0_i32_62 = arith.constant 0 : i32
        %c6_i32_63 = arith.constant 6 : i32
        %c8_i32_64 = arith.constant 8 : i32
        %66 = dfschedule.config.dma_bd(%63, %61, %c4_i32_61, %c0_i32_62, %c6_i32_63, %c8_i32_64, %65) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %67 = dfschedule.config.create_io(%66, %61) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %68 = dfschedule.schedule.getbdid(%61) : (!dfschedule.tile) -> i32
        %69 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_65 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
        %70 = dfschedule.memref_mapping %subview_65 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
        %71 = dfschedule.bind_core_buffer(%70, %69) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %72 = dfschedule.bind_core_buffer(%70, %69) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_66 = arith.constant 5 : i32
        %c0_i32_67 = arith.constant 0 : i32
        %c7_i32_68 = arith.constant 7 : i32
        %c9_i32_69 = arith.constant 9 : i32
        %73 = dfschedule.config.dma_bd(%72, %69, %c5_i32_66, %c0_i32_67, %c7_i32_68, %c9_i32_69) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_70 = arith.constant 4 : i32
        %c0_i32_71 = arith.constant 0 : i32
        %c7_i32_72 = arith.constant 7 : i32
        %c9_i32_73 = arith.constant 9 : i32
        %74 = dfschedule.config.dma_bd(%71, %69, %c4_i32_70, %c0_i32_71, %c7_i32_72, %c9_i32_73, %73) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %75 = dfschedule.config.create_io(%74, %69) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %76 = dfschedule.schedule.getbdid(%69) : (!dfschedule.tile) -> i32
        %77 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_74 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
        %78 = dfschedule.memref_mapping %subview_74 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
        %79 = dfschedule.bind_core_buffer(%78, %77) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %80 = dfschedule.bind_core_buffer(%78, %77) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_75 = arith.constant 5 : i32
        %c0_i32_76 = arith.constant 0 : i32
        %c8_i32_77 = arith.constant 8 : i32
        %c10_i32_78 = arith.constant 10 : i32
        %81 = dfschedule.config.dma_bd(%80, %77, %c5_i32_75, %c0_i32_76, %c8_i32_77, %c10_i32_78) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_79 = arith.constant 4 : i32
        %c0_i32_80 = arith.constant 0 : i32
        %c8_i32_81 = arith.constant 8 : i32
        %c10_i32_82 = arith.constant 10 : i32
        %82 = dfschedule.config.dma_bd(%79, %77, %c4_i32_79, %c0_i32_80, %c8_i32_81, %c10_i32_82, %81) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %83 = dfschedule.config.create_io(%82, %77) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %84 = dfschedule.schedule.getbdid(%77) : (!dfschedule.tile) -> i32
        %85 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
        %86 = dfschedule.config.load_kernel_group(%53, %61, %69, %77) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %87 = dfschedule.schedule.launch_kernel_group(%86) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %88 = dfschedule.schedule.start_io(%59, %60) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %89 = dfschedule.schedule.start_io(%67, %68) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %90 = dfschedule.schedule.start_io(%75, %76) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %91 = dfschedule.schedule.start_io(%83, %84) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %92 = dfschedule.schedule.start_io(%52, %85) {flow_index = 7 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%87, %92) : (!dfschedule.event, !dfschedule.event)
        dfschedule.free_device_mem %subview_39 : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>
        "routing.yield"() : () -> ()
      }
      %2 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %subview = memref.subview %arg0[112, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
        %4 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
        %c1_i32_0 = arith.constant 1 : i32
        %c0_i32_1 = arith.constant 0 : i32
        %c0_i32_2 = arith.constant 0 : i32
        %c-1_i32 = arith.constant -1 : i32
        %5 = dfschedule.config.dma_bd(%subview, %4, %c1_i32_0, %c0_i32_1, %c0_i32_2, %c-1_i32) {
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
        %6 = dfschedule.config.create_io(%5, %4) {
          channel = 1,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 0 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_3 = memref.subview %subview[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
        %8 = dfschedule.memref_mapping %subview_3 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_4 = arith.constant 3 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %c0_i32_6 = arith.constant 0 : i32
        %c-1_i32_7 = arith.constant -1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c3_i32_4, %c0_i32_5, %c0_i32_6, %c-1_i32_7) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_8 = arith.constant 2 : i32
        %c0_i32_9 = arith.constant 0 : i32
        %c0_i32_10 = arith.constant 0 : i32
        %c-1_i32_11 = arith.constant -1 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c2_i32_8, %c0_i32_9, %c0_i32_10, %c-1_i32_11, %11) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_12 = memref.subview %subview[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
        %16 = dfschedule.memref_mapping %subview_12 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
        %17 = dfschedule.bind_core_buffer(%16, %15) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %18 = dfschedule.bind_core_buffer(%16, %15) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_13 = arith.constant 3 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %c-1_i32_16 = arith.constant -1 : i32
        %19 = dfschedule.config.dma_bd(%18, %15, %c3_i32_13, %c0_i32_14, %c0_i32_15, %c-1_i32_16) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_17 = arith.constant 2 : i32
        %c0_i32_18 = arith.constant 0 : i32
        %c0_i32_19 = arith.constant 0 : i32
        %c-1_i32_20 = arith.constant -1 : i32
        %20 = dfschedule.config.dma_bd(%17, %15, %c2_i32_17, %c0_i32_18, %c0_i32_19, %c-1_i32_20, %19) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %21 = dfschedule.config.create_io(%20, %15) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %22 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
        %23 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_21 = memref.subview %subview[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
        %24 = dfschedule.memref_mapping %subview_21 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
        %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_22 = arith.constant 3 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %c0_i32_24 = arith.constant 0 : i32
        %c-1_i32_25 = arith.constant -1 : i32
        %27 = dfschedule.config.dma_bd(%26, %23, %c3_i32_22, %c0_i32_23, %c0_i32_24, %c-1_i32_25) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_26 = arith.constant 2 : i32
        %c0_i32_27 = arith.constant 0 : i32
        %c0_i32_28 = arith.constant 0 : i32
        %c-1_i32_29 = arith.constant -1 : i32
        %28 = dfschedule.config.dma_bd(%25, %23, %c2_i32_26, %c0_i32_27, %c0_i32_28, %c-1_i32_29, %27) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %29 = dfschedule.config.create_io(%28, %23) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %30 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
        %31 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_30 = memref.subview %subview[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
        %32 = dfschedule.memref_mapping %subview_30 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
        %33 = dfschedule.bind_core_buffer(%32, %31) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %34 = dfschedule.bind_core_buffer(%32, %31) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_31 = arith.constant 3 : i32
        %c0_i32_32 = arith.constant 0 : i32
        %c0_i32_33 = arith.constant 0 : i32
        %c-1_i32_34 = arith.constant -1 : i32
        %35 = dfschedule.config.dma_bd(%34, %31, %c3_i32_31, %c0_i32_32, %c0_i32_33, %c-1_i32_34) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_35 = arith.constant 2 : i32
        %c0_i32_36 = arith.constant 0 : i32
        %c0_i32_37 = arith.constant 0 : i32
        %c-1_i32_38 = arith.constant -1 : i32
        %36 = dfschedule.config.dma_bd(%33, %31, %c2_i32_35, %c0_i32_36, %c0_i32_37, %c-1_i32_38, %35) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %37 = dfschedule.config.create_io(%36, %31) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %38 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
        %39 = dfschedule.schedule.getbdid(%4) : (!dfschedule.tile) -> i32
        %40 = dfschedule.config.load_kernel_group(%7, %15, %23, %31) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %41 = dfschedule.schedule.launch_kernel_group(%40) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %42 = dfschedule.schedule.start_io(%13, %14) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %43 = dfschedule.schedule.start_io(%21, %22) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %44 = dfschedule.schedule.start_io(%29, %30) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %45 = dfschedule.schedule.start_io(%37, %38) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %46 = dfschedule.schedule.start_io(%6, %39) {flow_index = 8 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%41) : (!dfschedule.event)
        dfschedule.free_device_mem %subview : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
        %subview_39 = memref.subview %arg2[56, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>
        %47 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
        %c2_i32_40 = arith.constant 2 : i32
        %c6_i32 = arith.constant 6 : i32
        %c37632_i32 = arith.constant 37632 : i32
        %c12_i32 = arith.constant 12 : i32
        %c-1_i32_41 = arith.constant -1 : i32
        %48 = dfschedule.config.dma_bd(%subview_39, %47, %c6_i32, %c37632_i32, %c12_i32, %c-1_i32_41) {
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
        %c5_i32 = arith.constant 5 : i32
        %c25088_i32 = arith.constant 25088 : i32
        %c11_i32 = arith.constant 11 : i32
        %c-1_i32_42 = arith.constant -1 : i32
        %49 = dfschedule.config.dma_bd(%subview_39, %47, %c5_i32, %c25088_i32, %c11_i32, %c-1_i32_42, %48) {
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
        %c4_i32 = arith.constant 4 : i32
        %c12544_i32 = arith.constant 12544 : i32
        %c10_i32 = arith.constant 10 : i32
        %c-1_i32_43 = arith.constant -1 : i32
        %50 = dfschedule.config.dma_bd(%subview_39, %47, %c4_i32, %c12544_i32, %c10_i32, %c-1_i32_43, %49) {
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
        %c3_i32_44 = arith.constant 3 : i32
        %c0_i32_45 = arith.constant 0 : i32
        %c9_i32 = arith.constant 9 : i32
        %c-1_i32_46 = arith.constant -1 : i32
        %51 = dfschedule.config.dma_bd(%subview_39, %47, %c3_i32_44, %c0_i32_45, %c9_i32, %c-1_i32_46, %50) {
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
        %52 = dfschedule.config.create_io(%51, %47) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = true
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %53 = dfschedule.declaretile {col = 0 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_47 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
        %54 = dfschedule.memref_mapping %subview_47 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
        %55 = dfschedule.bind_core_buffer(%54, %53) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %56 = dfschedule.bind_core_buffer(%54, %53) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_48 = arith.constant 5 : i32
        %c0_i32_49 = arith.constant 0 : i32
        %c9_i32_50 = arith.constant 9 : i32
        %c3_i32_51 = arith.constant 3 : i32
        %57 = dfschedule.config.dma_bd(%56, %53, %c5_i32_48, %c0_i32_49, %c9_i32_50, %c3_i32_51) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_52 = arith.constant 4 : i32
        %c0_i32_53 = arith.constant 0 : i32
        %c9_i32_54 = arith.constant 9 : i32
        %c3_i32_55 = arith.constant 3 : i32
        %58 = dfschedule.config.dma_bd(%55, %53, %c4_i32_52, %c0_i32_53, %c9_i32_54, %c3_i32_55, %57) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %59 = dfschedule.config.create_io(%58, %53) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %60 = dfschedule.schedule.getbdid(%53) : (!dfschedule.tile) -> i32
        %61 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_56 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
        %62 = dfschedule.memref_mapping %subview_56 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
        %63 = dfschedule.bind_core_buffer(%62, %61) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %64 = dfschedule.bind_core_buffer(%62, %61) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_57 = arith.constant 5 : i32
        %c0_i32_58 = arith.constant 0 : i32
        %c10_i32_59 = arith.constant 10 : i32
        %c4_i32_60 = arith.constant 4 : i32
        %65 = dfschedule.config.dma_bd(%64, %61, %c5_i32_57, %c0_i32_58, %c10_i32_59, %c4_i32_60) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_61 = arith.constant 4 : i32
        %c0_i32_62 = arith.constant 0 : i32
        %c10_i32_63 = arith.constant 10 : i32
        %c4_i32_64 = arith.constant 4 : i32
        %66 = dfschedule.config.dma_bd(%63, %61, %c4_i32_61, %c0_i32_62, %c10_i32_63, %c4_i32_64, %65) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %67 = dfschedule.config.create_io(%66, %61) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %68 = dfschedule.schedule.getbdid(%61) : (!dfschedule.tile) -> i32
        %69 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_65 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
        %70 = dfschedule.memref_mapping %subview_65 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
        %71 = dfschedule.bind_core_buffer(%70, %69) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %72 = dfschedule.bind_core_buffer(%70, %69) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_66 = arith.constant 5 : i32
        %c0_i32_67 = arith.constant 0 : i32
        %c11_i32_68 = arith.constant 11 : i32
        %c5_i32_69 = arith.constant 5 : i32
        %73 = dfschedule.config.dma_bd(%72, %69, %c5_i32_66, %c0_i32_67, %c11_i32_68, %c5_i32_69) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_70 = arith.constant 4 : i32
        %c0_i32_71 = arith.constant 0 : i32
        %c11_i32_72 = arith.constant 11 : i32
        %c5_i32_73 = arith.constant 5 : i32
        %74 = dfschedule.config.dma_bd(%71, %69, %c4_i32_70, %c0_i32_71, %c11_i32_72, %c5_i32_73, %73) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %75 = dfschedule.config.create_io(%74, %69) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %76 = dfschedule.schedule.getbdid(%69) : (!dfschedule.tile) -> i32
        %77 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
        %subview_74 = memref.subview %subview_39[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
        %78 = dfschedule.memref_mapping %subview_74 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
        %79 = dfschedule.bind_core_buffer(%78, %77) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %80 = dfschedule.bind_core_buffer(%78, %77) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_75 = arith.constant 5 : i32
        %c0_i32_76 = arith.constant 0 : i32
        %c12_i32_77 = arith.constant 12 : i32
        %c6_i32_78 = arith.constant 6 : i32
        %81 = dfschedule.config.dma_bd(%80, %77, %c5_i32_75, %c0_i32_76, %c12_i32_77, %c6_i32_78) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_79 = arith.constant 4 : i32
        %c0_i32_80 = arith.constant 0 : i32
        %c12_i32_81 = arith.constant 12 : i32
        %c6_i32_82 = arith.constant 6 : i32
        %82 = dfschedule.config.dma_bd(%79, %77, %c4_i32_79, %c0_i32_80, %c12_i32_81, %c6_i32_82, %81) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %83 = dfschedule.config.create_io(%82, %77) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %84 = dfschedule.schedule.getbdid(%77) : (!dfschedule.tile) -> i32
        %85 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
        %86 = dfschedule.config.load_kernel_group(%53, %61, %69, %77) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %87 = dfschedule.schedule.launch_kernel_group(%86) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %88 = dfschedule.schedule.start_io(%59, %60) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %89 = dfschedule.schedule.start_io(%67, %68) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %90 = dfschedule.schedule.start_io(%75, %76) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %91 = dfschedule.schedule.start_io(%83, %84) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %92 = dfschedule.schedule.start_io(%52, %85) {flow_index = 9 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%87, %92) : (!dfschedule.event, !dfschedule.event)
        dfschedule.free_device_mem %subview_39 : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>
        "routing.yield"() : () -> ()
      }
      %3 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %subview = memref.subview %arg0[168, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>>
        %4 = dfschedule.declaretile {col = 3 : i32, row = 0 : i32} : !dfschedule.tile
        %c11_i32 = arith.constant 11 : i32
        %c0_i32_0 = arith.constant 0 : i32
        %c0_i32_1 = arith.constant 0 : i32
        %c-1_i32 = arith.constant -1 : i32
        %5 = dfschedule.config.dma_bd(%subview, %4, %c11_i32, %c0_i32_0, %c0_i32_1, %c-1_i32) {
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
        %6 = dfschedule.config.create_io(%5, %4) {
          channel = 1,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 0 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_2 = memref.subview %subview[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
        %8 = dfschedule.memref_mapping %subview_2 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_3 = arith.constant 3 : i32
        %c0_i32_4 = arith.constant 0 : i32
        %c0_i32_5 = arith.constant 0 : i32
        %c-1_i32_6 = arith.constant -1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c3_i32_3, %c0_i32_4, %c0_i32_5, %c-1_i32_6) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_7 = arith.constant 2 : i32
        %c0_i32_8 = arith.constant 0 : i32
        %c0_i32_9 = arith.constant 0 : i32
        %c-1_i32_10 = arith.constant -1 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c2_i32_7, %c0_i32_8, %c0_i32_9, %c-1_i32_10, %11) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_11 = memref.subview %subview[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
        %16 = dfschedule.memref_mapping %subview_11 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
        %17 = dfschedule.bind_core_buffer(%16, %15) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %18 = dfschedule.bind_core_buffer(%16, %15) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_12 = arith.constant 3 : i32
        %c0_i32_13 = arith.constant 0 : i32
        %c0_i32_14 = arith.constant 0 : i32
        %c-1_i32_15 = arith.constant -1 : i32
        %19 = dfschedule.config.dma_bd(%18, %15, %c3_i32_12, %c0_i32_13, %c0_i32_14, %c-1_i32_15) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_16 = arith.constant 2 : i32
        %c0_i32_17 = arith.constant 0 : i32
        %c0_i32_18 = arith.constant 0 : i32
        %c-1_i32_19 = arith.constant -1 : i32
        %20 = dfschedule.config.dma_bd(%17, %15, %c2_i32_16, %c0_i32_17, %c0_i32_18, %c-1_i32_19, %19) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %21 = dfschedule.config.create_io(%20, %15) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %22 = dfschedule.schedule.getbdid(%15) : (!dfschedule.tile) -> i32
        %23 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_20 = memref.subview %subview[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
        %24 = dfschedule.memref_mapping %subview_20 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
        %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_21 = arith.constant 3 : i32
        %c0_i32_22 = arith.constant 0 : i32
        %c0_i32_23 = arith.constant 0 : i32
        %c-1_i32_24 = arith.constant -1 : i32
        %27 = dfschedule.config.dma_bd(%26, %23, %c3_i32_21, %c0_i32_22, %c0_i32_23, %c-1_i32_24) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_25 = arith.constant 2 : i32
        %c0_i32_26 = arith.constant 0 : i32
        %c0_i32_27 = arith.constant 0 : i32
        %c-1_i32_28 = arith.constant -1 : i32
        %28 = dfschedule.config.dma_bd(%25, %23, %c2_i32_25, %c0_i32_26, %c0_i32_27, %c-1_i32_28, %27) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %29 = dfschedule.config.create_io(%28, %23) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %30 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
        %31 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_29 = memref.subview %subview[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
        %32 = dfschedule.memref_mapping %subview_29 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
        %33 = dfschedule.bind_core_buffer(%32, %31) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %34 = dfschedule.bind_core_buffer(%32, %31) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
        %c3_i32_30 = arith.constant 3 : i32
        %c0_i32_31 = arith.constant 0 : i32
        %c0_i32_32 = arith.constant 0 : i32
        %c-1_i32_33 = arith.constant -1 : i32
        %35 = dfschedule.config.dma_bd(%34, %31, %c3_i32_30, %c0_i32_31, %c0_i32_32, %c-1_i32_33) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 2 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c2_i32_34 = arith.constant 2 : i32
        %c0_i32_35 = arith.constant 0 : i32
        %c0_i32_36 = arith.constant 0 : i32
        %c-1_i32_37 = arith.constant -1 : i32
        %36 = dfschedule.config.dma_bd(%33, %31, %c2_i32_34, %c0_i32_35, %c0_i32_36, %c-1_i32_37, %35) {
          len = 4636 : i32,
          enable_packet = false,
          next_bd = 3 : i32,
          acquire_lock_id = 0 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 1 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %37 = dfschedule.config.create_io(%36, %31) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %38 = dfschedule.schedule.getbdid(%31) : (!dfschedule.tile) -> i32
        %39 = dfschedule.schedule.getbdid(%4) : (!dfschedule.tile) -> i32
        %40 = dfschedule.config.load_kernel_group(%7, %15, %23, %31) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %41 = dfschedule.schedule.launch_kernel_group(%40) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %42 = dfschedule.schedule.start_io(%13, %14) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %43 = dfschedule.schedule.start_io(%21, %22) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %44 = dfschedule.schedule.start_io(%29, %30) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %45 = dfschedule.schedule.start_io(%37, %38) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %46 = dfschedule.schedule.start_io(%6, %39) {flow_index = 10 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%41) : (!dfschedule.event)
        dfschedule.free_device_mem %subview : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>>
        %subview_38 = memref.subview %arg2[84, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>
        %47 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
        %c7_i32 = arith.constant 7 : i32
        %c11_i32_39 = arith.constant 11 : i32
        %c37632_i32 = arith.constant 37632 : i32
        %c16_i32 = arith.constant 16 : i32
        %c-1_i32_40 = arith.constant -1 : i32
        %48 = dfschedule.config.dma_bd(%subview_38, %47, %c11_i32_39, %c37632_i32, %c16_i32, %c-1_i32_40) {
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
        %c10_i32 = arith.constant 10 : i32
        %c25088_i32 = arith.constant 25088 : i32
        %c15_i32 = arith.constant 15 : i32
        %c-1_i32_41 = arith.constant -1 : i32
        %49 = dfschedule.config.dma_bd(%subview_38, %47, %c10_i32, %c25088_i32, %c15_i32, %c-1_i32_41, %48) {
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
        %c9_i32 = arith.constant 9 : i32
        %c12544_i32 = arith.constant 12544 : i32
        %c14_i32 = arith.constant 14 : i32
        %c-1_i32_42 = arith.constant -1 : i32
        %50 = dfschedule.config.dma_bd(%subview_38, %47, %c9_i32, %c12544_i32, %c14_i32, %c-1_i32_42, %49) {
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
        %c8_i32 = arith.constant 8 : i32
        %c0_i32_43 = arith.constant 0 : i32
        %c13_i32 = arith.constant 13 : i32
        %c-1_i32_44 = arith.constant -1 : i32
        %51 = dfschedule.config.dma_bd(%subview_38, %47, %c8_i32, %c0_i32_43, %c13_i32, %c-1_i32_44, %50) {
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
        %52 = dfschedule.config.create_io(%51, %47) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV",
          enable_out_of_order = true
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %53 = dfschedule.declaretile {col = 0 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_45 = memref.subview %subview_38[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
        %54 = dfschedule.memref_mapping %subview_45 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
        %55 = dfschedule.bind_core_buffer(%54, %53) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %56 = dfschedule.bind_core_buffer(%54, %53) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32 = arith.constant 5 : i32
        %c0_i32_46 = arith.constant 0 : i32
        %c13_i32_47 = arith.constant 13 : i32
        %c8_i32_48 = arith.constant 8 : i32
        %57 = dfschedule.config.dma_bd(%56, %53, %c5_i32, %c0_i32_46, %c13_i32_47, %c8_i32_48) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32 = arith.constant 4 : i32
        %c0_i32_49 = arith.constant 0 : i32
        %c13_i32_50 = arith.constant 13 : i32
        %c8_i32_51 = arith.constant 8 : i32
        %58 = dfschedule.config.dma_bd(%55, %53, %c4_i32, %c0_i32_49, %c13_i32_50, %c8_i32_51, %57) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %59 = dfschedule.config.create_io(%58, %53) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %60 = dfschedule.schedule.getbdid(%53) : (!dfschedule.tile) -> i32
        %61 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_52 = memref.subview %subview_38[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
        %62 = dfschedule.memref_mapping %subview_52 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
        %63 = dfschedule.bind_core_buffer(%62, %61) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %64 = dfschedule.bind_core_buffer(%62, %61) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_53 = arith.constant 5 : i32
        %c0_i32_54 = arith.constant 0 : i32
        %c14_i32_55 = arith.constant 14 : i32
        %c9_i32_56 = arith.constant 9 : i32
        %65 = dfschedule.config.dma_bd(%64, %61, %c5_i32_53, %c0_i32_54, %c14_i32_55, %c9_i32_56) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_57 = arith.constant 4 : i32
        %c0_i32_58 = arith.constant 0 : i32
        %c14_i32_59 = arith.constant 14 : i32
        %c9_i32_60 = arith.constant 9 : i32
        %66 = dfschedule.config.dma_bd(%63, %61, %c4_i32_57, %c0_i32_58, %c14_i32_59, %c9_i32_60, %65) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %67 = dfschedule.config.create_io(%66, %61) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %68 = dfschedule.schedule.getbdid(%61) : (!dfschedule.tile) -> i32
        %69 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_61 = memref.subview %subview_38[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
        %70 = dfschedule.memref_mapping %subview_61 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
        %71 = dfschedule.bind_core_buffer(%70, %69) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %72 = dfschedule.bind_core_buffer(%70, %69) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_62 = arith.constant 5 : i32
        %c0_i32_63 = arith.constant 0 : i32
        %c15_i32_64 = arith.constant 15 : i32
        %c10_i32_65 = arith.constant 10 : i32
        %73 = dfschedule.config.dma_bd(%72, %69, %c5_i32_62, %c0_i32_63, %c15_i32_64, %c10_i32_65) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_66 = arith.constant 4 : i32
        %c0_i32_67 = arith.constant 0 : i32
        %c15_i32_68 = arith.constant 15 : i32
        %c10_i32_69 = arith.constant 10 : i32
        %74 = dfschedule.config.dma_bd(%71, %69, %c4_i32_66, %c0_i32_67, %c15_i32_68, %c10_i32_69, %73) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %75 = dfschedule.config.create_io(%74, %69) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %76 = dfschedule.schedule.getbdid(%69) : (!dfschedule.tile) -> i32
        %77 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
        %subview_70 = memref.subview %subview_38[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
        %78 = dfschedule.memref_mapping %subview_70 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
        %79 = dfschedule.bind_core_buffer(%78, %77) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %80 = dfschedule.bind_core_buffer(%78, %77) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
        %c5_i32_71 = arith.constant 5 : i32
        %c0_i32_72 = arith.constant 0 : i32
        %c16_i32_73 = arith.constant 16 : i32
        %c11_i32_74 = arith.constant 11 : i32
        %81 = dfschedule.config.dma_bd(%80, %77, %c5_i32_71, %c0_i32_72, %c16_i32_73, %c11_i32_74) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 4 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
        %c4_i32_75 = arith.constant 4 : i32
        %c0_i32_76 = arith.constant 0 : i32
        %c16_i32_77 = arith.constant 16 : i32
        %c11_i32_78 = arith.constant 11 : i32
        %82 = dfschedule.config.dma_bd(%79, %77, %c4_i32_75, %c0_i32_76, %c16_i32_77, %c11_i32_78, %81) {
          len = 3136 : i32,
          enable_packet = true,
          next_bd = 5 : i32,
          acquire_lock_id = 5 : i32,
          acquire_lock_val = -1 : i32,
          release_lock_id = 4 : i32,
          release_lock_val = 1 : i32,
          data_id = -1 : i32
        } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %83 = dfschedule.config.create_io(%82, %77) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND",
          enable_out_of_order = false
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %84 = dfschedule.schedule.getbdid(%77) : (!dfschedule.tile) -> i32
        %85 = dfschedule.schedule.getbdid(%47) : (!dfschedule.tile) -> i32
        %86 = dfschedule.config.load_kernel_group(%53, %61, %69, %77) {
          callee = [@dskernel_receiver]
        } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %87 = dfschedule.schedule.launch_kernel_group(%86) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %88 = dfschedule.schedule.start_io(%59, %60) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %89 = dfschedule.schedule.start_io(%67, %68) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %90 = dfschedule.schedule.start_io(%75, %76) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %91 = dfschedule.schedule.start_io(%83, %84) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %92 = dfschedule.schedule.start_io(%52, %85) {flow_index = 11 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%87, %92) : (!dfschedule.event, !dfschedule.event)
        dfschedule.free_device_mem %subview_38 : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
  dfschedule.dskernel_receiver @dskernel_receiver {
  }
}
