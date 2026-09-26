// *****************************************************************************
// * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// *****************************************************************************
// Module-level constant: read-only 16x16 matrix of i8 values.
// Replaces: arith.constant dense<"0x..."> : tensor<16x16xi8>
memref.global "private" constant @my_constant : memref<16x16xi8> = dense<"0x0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF00">

module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  func.func @main() {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32

    // Step 1: get handle to the module-level constant.
    %const_ptr = memref.get_global @my_constant : memref<16x16xi8>

    // Step 2: allocate writable DDR root buffer (device-accessible).
    // Replaces: arith.constant dense<...> : tensor<16x16xi8>
    //   + routing.partitiontensor (splitnum=2, splitdim=0)
    %root = memref.alloc() : memref<16x16xi8>

    // Step 3: copy constant data into the DDR buffer.
    memref.copy %const_ptr, %root : memref<16x16xi8> to memref<16x16xi8>

    // -----------------------------------------------------------------------
    // Partition 0  (rows 0..7)
    // Replaces: routing.RoutingCreate<Memo="row"> scf_idx=%c0_i32 { ^bb0: ... }
    // -----------------------------------------------------------------------
    scf.execute_region {

      // Level 1 subview: partition slice (rows 0..7, all 16 cols).
      // Replaces: tensor.extract_slice %0[0, 0] [8, 16] [1, 1] {tag="partitionslice0"}
      %part0 = memref.subview %root[0, 0][8, 16][1, 1]
               : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1], offset: 0>>

      // Level 2 subview: intermediate (mirrors original two-level extract_slice nesting).
      // Replaces: the %extracted_slice level before per-tile slicing.
      %extracted_slice_p0 = memref.subview %part0[0, 0][8, 16][1, 1]
                            : memref<8x16xi8, strided<[16, 1], offset: 0>>
                           to memref<8x16xi8, strided<[16, 1], offset: 0>>

      // Level 3 subviews: per-tile slices.
      // Replaces: tensor.extract_slice %extracted_slice[0, 0] [4, 16] [1, 1] {tag="producer0"}
      %slice0_tile0 = memref.subview %extracted_slice_p0[0, 0][4, 16][1, 1]
                      : memref<8x16xi8, strided<[16, 1], offset: 0>>
                     to memref<4x16xi8, strided<[16, 1], offset: 0>>

      // Replaces: tensor.extract_slice %extracted_slice[4, 0] [4, 16] [1, 1] {tag="producer1"}
      %slice0_tile1 = memref.subview %extracted_slice_p0[4, 0][4, 16][1, 1]
                      : memref<8x16xi8, strided<[16, 1], offset: 0>>
                     to memref<4x16xi8, strided<[16, 1], offset: 64>>

      // --- Shim tile (2, 0): alloc DDR, buffer_view, dma_bd, create_io ---
      // Pattern: alloc_device_mem(partN) -> buffer_view -> dma_bd(%7) -> create_io
      %5 = dfschedule.alloc_device_mem(%part0)
           : (memref<8x16xi8, strided<[16, 1], offset: 0>>) -> memref<128xi8, 1 : i32>
      %6 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
      %c0_i32_2 = arith.constant 0 : i32
      %7 = dfschedule.buffer_view %5 {len = 128 : i64, offset = 0 : i64}
           : memref<128xi8, 1 : i32> -> memref<128xi8, 1 : i32>
      %noff1 = arith.constant 0 : i32
      %npkt1 = arith.constant 0 : i32
      %nooo1 = arith.constant -1 : i32
      %8 = dfschedule.config.dma_bd(%7, %6, %c0_i32_2, %noff1, %npkt1, %nooo1) {
        len = 128 : i32,
        enable_packet = true,
        next_bd = 4294967295 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = 0 : i32,
        release_lock_val = 0 : i32,
        data_id = 0 : i32
      } : (memref<128xi8, 1 : i32>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %9 = dfschedule.config.create_io(%8, %6) {
        channel = 0,
        direction = "S2MM",
        io_operation = "RECV"
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

      // --- Core tile (0, 3) ---
      // Pattern: memref_mapping -> bind_core_buffer(ping+pong) -> dma_bd(pong) -> dma_bd(ping,linked) -> create_io
      %10 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
      %mapped0_tile0 = dfschedule.memref_mapping %slice0_tile0
                       : (memref<4x16xi8, strided<[16, 1], offset: 0>>) -> memref<4x16xi8>
      %13 = dfschedule.bind_core_buffer(%mapped0_tile0, %10) {offset = 0 : i64}
            : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
      %14 = dfschedule.bind_core_buffer(%mapped0_tile0, %10) {offset = 64 : i64}
            : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
      %c1_i32_3 = arith.constant 1 : i32
      %noff2 = arith.constant 0 : i32
      %npkt2 = arith.constant 0 : i32
      %nooo2 = arith.constant -1 : i32
      %15 = dfschedule.config.dma_bd(%14, %10, %c1_i32_3, %noff2, %npkt2, %nooo2) {
        len = 64 : i32,
        enable_packet = true,
        next_bd = 0 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = -1 : i32,
        release_lock_id = 1 : i32,
        release_lock_val = 1 : i32,
        data_id = -1 : i32
      } : (memref<4x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %c0_i32_4 = arith.constant 0 : i32
      %noff3 = arith.constant 0 : i32
      %npkt3 = arith.constant 0 : i32
      %nooo3 = arith.constant -1 : i32
      %16 = dfschedule.config.dma_bd(%13, %10, %c0_i32_4, %noff3, %npkt3, %nooo3, %15) {
        len = 64 : i32,
        enable_packet = true,
        next_bd = 1 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = -1 : i32,
        release_lock_id = 1 : i32,
        release_lock_val = 1 : i32,
        data_id = -1 : i32
      } : (memref<4x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %17 = dfschedule.config.create_io(%16, %10) {
        channel = 0,
        direction = "MM2S",
        io_operation = "SEND"
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %18 = dfschedule.schedule.getbdid(%10) : (!dfschedule.tile) -> i32
      %19 = dfschedule.schedule.start_io(%17, %18) {flow_index = 0 : i32}
            : (!dfschedule.io_handle, i32) -> !dfschedule.event

      // --- Core tile (1, 3) ---
      // Pattern: memref_mapping -> bind_core_buffer(ping+pong) -> dma_bd(pong) -> dma_bd(ping,linked) -> create_io
      %20 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
      %mapped0_tile1 = dfschedule.memref_mapping %slice0_tile1
                       : (memref<4x16xi8, strided<[16, 1], offset: 64>>) -> memref<4x16xi8>
      %23 = dfschedule.bind_core_buffer(%mapped0_tile1, %20) {offset = 0 : i64}
            : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
      %24 = dfschedule.bind_core_buffer(%mapped0_tile1, %20) {offset = 64 : i64}
            : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
      %c1_i32_5 = arith.constant 1 : i32
      %noff4 = arith.constant 0 : i32
      %npkt4 = arith.constant 1 : i32
      %nooo4 = arith.constant -1 : i32
      %25 = dfschedule.config.dma_bd(%24, %20, %c1_i32_5, %noff4, %npkt4, %nooo4) {
        len = 64 : i32,
        enable_packet = true,
        next_bd = 0 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = -1 : i32,
        release_lock_id = 1 : i32,
        release_lock_val = 1 : i32,
        data_id = -1 : i32
      } : (memref<4x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %c0_i32_6 = arith.constant 0 : i32
      %noff5 = arith.constant 0 : i32
      %npkt5 = arith.constant 1 : i32
      %nooo5 = arith.constant -1 : i32
      %26 = dfschedule.config.dma_bd(%23, %20, %c0_i32_6, %noff5, %npkt5, %nooo5, %25) {
        len = 64 : i32,
        enable_packet = true,
        next_bd = 1 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = -1 : i32,
        release_lock_id = 1 : i32,
        release_lock_val = 1 : i32,
        data_id = -1 : i32
      } : (memref<4x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %27 = dfschedule.config.create_io(%26, %20) {
        channel = 0,
        direction = "MM2S",
        io_operation = "SEND"
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %28 = dfschedule.schedule.getbdid(%20) : (!dfschedule.tile) -> i32
      %29 = dfschedule.schedule.start_io(%27, %28) {flow_index = 0 : i32}
            : (!dfschedule.io_handle, i32) -> !dfschedule.event

      %32 = dfschedule.config.load_kernel_group(%10, %20) {
        callee = [@dskernel_receiver],
        distributed_compute_kernel_args = [@compute0, @compute0]
      } : (!dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
      %33 = dfschedule.schedule.launch_kernel_group(%32) : (!dfschedule.kernelgroup) -> !dfschedule.event
      %34 = dfschedule.schedule.getbdid(%6) : (!dfschedule.tile) -> i32
      %35 = dfschedule.schedule.start_io(%9, %34) {flow_index = 0 : i32}
            : (!dfschedule.io_handle, i32) -> !dfschedule.event
      dfschedule.schedule.wait(%33, %35) : (!dfschedule.event, !dfschedule.event)
      dfschedule.free_device_mem %5 : memref<128xi8, 1 : i32>
    }

    // -----------------------------------------------------------------------
    // Partition 1  (rows 8..15)
    // Replaces: routing.RoutingCreate<Memo="row"> scf_idx=%c1_i32 { ^bb0: ... }
    // -----------------------------------------------------------------------
    scf.execute_region {

      // Level 1 subview: partition slice (rows 8..15, all 16 cols).
      // Replaces: tensor.extract_slice %0[8, 0] [8, 16] [1, 1] {tag="partitionslice1"}
      %part1 = memref.subview %root[8, 0][8, 16][1, 1]
               : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1], offset: 128>>

      // Level 2 subview: intermediate.
      %extracted_slice_p1 = memref.subview %part1[0, 0][8, 16][1, 1]
                            : memref<8x16xi8, strided<[16, 1], offset: 128>>
                           to memref<8x16xi8, strided<[16, 1], offset: 128>>

      // Level 3 subviews: per-tile slices.
      // Replaces: tensor.extract_slice %extracted_slice[0, 0] [4, 16] [1, 1] {tag="producer0"}
      %slice1_tile0 = memref.subview %extracted_slice_p1[0, 0][4, 16][1, 1]
                      : memref<8x16xi8, strided<[16, 1], offset: 128>>
                     to memref<4x16xi8, strided<[16, 1], offset: 128>>

      // Replaces: tensor.extract_slice %extracted_slice[4, 0] [4, 16] [1, 1] {tag="producer1"}
      %slice1_tile1 = memref.subview %extracted_slice_p1[4, 0][4, 16][1, 1]
                      : memref<8x16xi8, strided<[16, 1], offset: 128>>
                     to memref<4x16xi8, strided<[16, 1], offset: 192>>

      // --- Shim tile (2, 0): alloc DDR, buffer_view, dma_bd, create_io ---
      %5 = dfschedule.alloc_device_mem(%part1)
           : (memref<8x16xi8, strided<[16, 1], offset: 128>>) -> memref<128xi8, 1 : i32>
      %6 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
      %c0_i32_2 = arith.constant 0 : i32
      %7 = dfschedule.buffer_view %5 {len = 128 : i64, offset = 0 : i64}
           : memref<128xi8, 1 : i32> -> memref<128xi8, 1 : i32>
      %noff6 = arith.constant 0 : i32
      %npkt6 = arith.constant 0 : i32
      %nooo6 = arith.constant -1 : i32
      %8 = dfschedule.config.dma_bd(%7, %6, %c0_i32_2, %noff6, %npkt6, %nooo6) {
        len = 128 : i32,
        enable_packet = true,
        next_bd = 4294967295 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = 0 : i32,
        release_lock_val = 0 : i32,
        data_id = 0 : i32
      } : (memref<128xi8, 1 : i32>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %9 = dfschedule.config.create_io(%8, %6) {
        channel = 1,
        direction = "S2MM",
        io_operation = "RECV"
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

      // --- Core tile (0, 4) ---
      // Pattern: memref_mapping -> bind_core_buffer(ping+pong) -> dma_bd(pong) -> dma_bd(ping,linked) -> create_io
      %10 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
      %mapped1_tile0 = dfschedule.memref_mapping %slice1_tile0
                       : (memref<4x16xi8, strided<[16, 1], offset: 128>>) -> memref<4x16xi8>
      %13 = dfschedule.bind_core_buffer(%mapped1_tile0, %10) {offset = 0 : i64}
            : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
      %14 = dfschedule.bind_core_buffer(%mapped1_tile0, %10) {offset = 64 : i64}
            : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
      %c1_i32_3 = arith.constant 1 : i32
      %noff7 = arith.constant 0 : i32
      %npkt7 = arith.constant 0 : i32
      %nooo7 = arith.constant -1 : i32
      %15 = dfschedule.config.dma_bd(%14, %10, %c1_i32_3, %noff7, %npkt7, %nooo7) {
        len = 64 : i32,
        enable_packet = true,
        next_bd = 0 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = -1 : i32,
        release_lock_id = 1 : i32,
        release_lock_val = 1 : i32,
        data_id = -1 : i32
      } : (memref<4x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %c0_i32_4 = arith.constant 0 : i32
      %noff8 = arith.constant 0 : i32
      %npkt8 = arith.constant 0 : i32
      %nooo8 = arith.constant -1 : i32
      %16 = dfschedule.config.dma_bd(%13, %10, %c0_i32_4, %noff8, %npkt8, %nooo8, %15) {
        len = 64 : i32,
        enable_packet = true,
        next_bd = 1 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = -1 : i32,
        release_lock_id = 1 : i32,
        release_lock_val = 1 : i32,
        data_id = -1 : i32
      } : (memref<4x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %17 = dfschedule.config.create_io(%16, %10) {
        channel = 0,
        direction = "MM2S",
        io_operation = "SEND"
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %18 = dfschedule.schedule.getbdid(%10) : (!dfschedule.tile) -> i32
      %19 = dfschedule.schedule.start_io(%17, %18) {flow_index = 1 : i32}
            : (!dfschedule.io_handle, i32) -> !dfschedule.event

      // --- Core tile (1, 4) ---
      // Pattern: memref_mapping -> bind_core_buffer(ping+pong) -> dma_bd(pong) -> dma_bd(ping,linked) -> create_io
      %20 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
      %mapped1_tile1 = dfschedule.memref_mapping %slice1_tile1
                       : (memref<4x16xi8, strided<[16, 1], offset: 192>>) -> memref<4x16xi8>
      %23 = dfschedule.bind_core_buffer(%mapped1_tile1, %20) {offset = 0 : i64}
            : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
      %24 = dfschedule.bind_core_buffer(%mapped1_tile1, %20) {offset = 64 : i64}
            : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
      %c1_i32_5 = arith.constant 1 : i32
      %noff9 = arith.constant 0 : i32
      %npkt9 = arith.constant 1 : i32
      %nooo9 = arith.constant -1 : i32
      %25 = dfschedule.config.dma_bd(%24, %20, %c1_i32_5, %noff9, %npkt9, %nooo9) {
        len = 64 : i32,
        enable_packet = true,
        next_bd = 0 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = -1 : i32,
        release_lock_id = 1 : i32,
        release_lock_val = 1 : i32,
        data_id = -1 : i32
      } : (memref<4x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %c0_i32_6 = arith.constant 0 : i32
      %noff10 = arith.constant 0 : i32
      %npkt10 = arith.constant 1 : i32
      %nooo10 = arith.constant -1 : i32
      %26 = dfschedule.config.dma_bd(%23, %20, %c0_i32_6, %noff10, %npkt10, %nooo10, %25) {
        len = 64 : i32,
        enable_packet = true,
        next_bd = 1 : i32,
        acquire_lock_id = 0 : i32,
        acquire_lock_val = -1 : i32,
        release_lock_id = 1 : i32,
        release_lock_val = 1 : i32,
        data_id = -1 : i32
      } : (memref<4x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %27 = dfschedule.config.create_io(%26, %20) {
        channel = 0,
        direction = "MM2S",
        io_operation = "SEND"
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %28 = dfschedule.schedule.getbdid(%20) : (!dfschedule.tile) -> i32
      %29 = dfschedule.schedule.start_io(%27, %28) {flow_index = 1 : i32}
            : (!dfschedule.io_handle, i32) -> !dfschedule.event

      %32 = dfschedule.config.load_kernel_group(%10, %20) {
        callee = [@dskernel_receiver],
        distributed_compute_kernel_args = [@compute0, @compute0]
      } : (!dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
      %33 = dfschedule.schedule.launch_kernel_group(%32) : (!dfschedule.kernelgroup) -> !dfschedule.event
      %34 = dfschedule.schedule.getbdid(%6) : (!dfschedule.tile) -> i32
      %35 = dfschedule.schedule.start_io(%9, %34) {flow_index = 1 : i32}
            : (!dfschedule.io_handle, i32) -> !dfschedule.event
      dfschedule.schedule.wait(%33, %35) : (!dfschedule.event, !dfschedule.event)
      dfschedule.free_device_mem %5 : memref<128xi8, 1 : i32>
    }

    memref.dealloc %root : memref<16x16xi8>
    return
  }
  dfschedule.dskernel_receiver @dskernel_receiver {
  }
}
