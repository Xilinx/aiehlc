<\!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

> **OUTDATED (kept for design history):** The `dfschedule.declare_kernel_config`
> ops and the `load_kernel_group` `distributed_args` operand discussed throughout
> this document have since been **removed** from the pipeline. They were pure
> metadata (logged but never lowered to runtime code); the per-tile
> DMA/lock/buffer parameters are materialized directly by the `config.dma_bd`,
> lock-init, and `start_io` ops. `ScheduleCanonicalizePass` now emits
> `load_kernel_group` with `distributed_args` left null and no
> `kernelconfig_merged*` ops. Treat the `declare_kernel_config` / `distributed_args`
> snippets below as historical context only.

S A MLIR EXPERT please advise what is the best solution to cannolize my IR "module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {

  memref.global "private" constant @my_constant : memref<16x16xi8> = dense<"0x0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF00">

  func.func @main() {

    %c1_i32 = arith.constant 1 : i32

    %c0_i32 = arith.constant 0 : i32

    %0 = memref.get_global @my_constant : memref<16x16xi8>

    %alloc = memref.alloc() : memref<16x16xi8>

    memref.copy %0, %alloc : memref<16x16xi8> to memref<16x16xi8>

    scf.execute_region {

      %1 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{

      ^bb0(%arg0: i32):

        %subview = memref.subview %alloc[0, 0] [8, 16] [1, 1] : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1]>>

        %3 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile

        %c0_i32_0 = arith.constant 0 : i32

        %4 = dfschedule.buffer_view %subview {len = 128 : i64, offset = 0 : i64} : memref<8x16xi8, strided<[16, 1]>> -> memref<8x16xi8, strided<[16, 1]>>

        %5 = dfschedule.config.dma_bd(%4, %3, %c0_i32_0) {

          offset = 0,

          len = 128,

          enable_packet = true,

          packet_id = 0,

          next_bd = 4294967295,

          acquire_lock_id = 0,

          acquire_lock_val = 0,

          release_lock_id = 0,

          release_lock_val = 0,

          data_id = 0

        } : (memref<8x16xi8, strided<[16, 1]>>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %6 = dfschedule.config.create_io(%5, %3) {

          channel = 0,

          direction = "S2MM",

          io_operation = "RECV"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %7 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile

        %subview_1 = memref.subview %subview[0, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1]>> to memref<4x16xi8, strided<[16, 1]>>

        %8 = dfschedule.memref_mapping %subview_1 : (memref<4x16xi8, strided<[16, 1]>>) -> memref<4x16xi8>

        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %c1_i32_2 = arith.constant 1 : i32

        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_2) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 0,

          next_bd = 0,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %c0_i32_3 = arith.constant 0 : i32

        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_3, %11) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 0,

          next_bd = 1,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle

        %13 = dfschedule.config.create_io(%12, %7) {

          channel = 0,

          direction = "MM2S",

          io_operation = "SEND"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32

        %15 = dfschedule.schedule.start_io(%13, %14) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        %16 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile

        %subview_4 = memref.subview %subview[4, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1]>> to memref<4x16xi8, strided<[16, 1], offset: 64>>

        %17 = dfschedule.memref_mapping %subview_4 : (memref<4x16xi8, strided<[16, 1], offset: 64>>) -> memref<4x16xi8>

        %18 = dfschedule.bind_core_buffer(%17, %16) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %19 = dfschedule.bind_core_buffer(%17, %16) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %c1_i32_5 = arith.constant 1 : i32

        %20 = dfschedule.config.dma_bd(%19, %16, %c1_i32_5) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 1,

          next_bd = 0,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %c0_i32_6 = arith.constant 0 : i32

        %21 = dfschedule.config.dma_bd(%18, %16, %c0_i32_6, %20) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 1,

          next_bd = 1,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle

        %22 = dfschedule.config.create_io(%21, %16) {

          channel = 0,

          direction = "MM2S",

          io_operation = "SEND"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %23 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32

        %24 = dfschedule.schedule.start_io(%22, %23) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 0 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}

        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}

        %27 = dfschedule.config.load_kernel_group(%7, %16) {

          callee = [@dskernel_receiver],

          distributed_compute_kernel_args = [@compute0, @compute0],

          distributed_args = [@kernelconfig0, @kernelconfig1]

        } : (!dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup

        %28 = dfschedule.schedule.launch_kernel_group(%27) : (!dfschedule.kernelgroup) -> !dfschedule.event

        %29 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32

        %30 = dfschedule.schedule.start_io(%6, %29) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        dfschedule.schedule.wait(%28, %30) : (!dfschedule.event, !dfschedule.event)

        dfschedule.free_device_mem %subview : memref<8x16xi8, strided<[16, 1]>>

      }

      %2 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{

      ^bb0(%arg0: i32):

        %subview = memref.subview %alloc[8, 0] [8, 16] [1, 1] : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1], offset: 128>>

        %3 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile

        %c0_i32_0 = arith.constant 0 : i32

        %4 = dfschedule.buffer_view %subview {len = 128 : i64, offset = 0 : i64} : memref<8x16xi8, strided<[16, 1], offset: 128>> -> memref<8x16xi8, strided<[16, 1], offset: 128>>

        %5 = dfschedule.config.dma_bd(%4, %3, %c0_i32_0) {

          offset = 0,

          len = 128,

          enable_packet = true,

          packet_id = 0,

          next_bd = 4294967295,

          acquire_lock_id = 0,

          acquire_lock_val = 0,

          release_lock_id = 0,

          release_lock_val = 0,

          data_id = 0

        } : (memref<8x16xi8, strided<[16, 1], offset: 128>>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %6 = dfschedule.config.create_io(%5, %3) {

          channel = 1,

          direction = "S2MM",

          io_operation = "RECV"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %7 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile

        %subview_1 = memref.subview %subview[0, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1], offset: 128>> to memref<4x16xi8, strided<[16, 1], offset: 128>>

        %8 = dfschedule.memref_mapping %subview_1 : (memref<4x16xi8, strided<[16, 1], offset: 128>>) -> memref<4x16xi8>

        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %c1_i32_2 = arith.constant 1 : i32

        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_2) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 0,

          next_bd = 0,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %c0_i32_3 = arith.constant 0 : i32

        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_3, %11) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 0,

          next_bd = 1,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle

        %13 = dfschedule.config.create_io(%12, %7) {

          channel = 0,

          direction = "MM2S",

          io_operation = "SEND"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32

        %15 = dfschedule.schedule.start_io(%13, %14) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        %16 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile

        %subview_4 = memref.subview %subview[4, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1], offset: 128>> to memref<4x16xi8, strided<[16, 1], offset: 192>>

        %17 = dfschedule.memref_mapping %subview_4 : (memref<4x16xi8, strided<[16, 1], offset: 192>>) -> memref<4x16xi8>

        %18 = dfschedule.bind_core_buffer(%17, %16) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %19 = dfschedule.bind_core_buffer(%17, %16) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %c1_i32_5 = arith.constant 1 : i32

        %20 = dfschedule.config.dma_bd(%19, %16, %c1_i32_5) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 1,

          next_bd = 0,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %c0_i32_6 = arith.constant 0 : i32

        %21 = dfschedule.config.dma_bd(%18, %16, %c0_i32_6, %20) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 1,

          next_bd = 1,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle

        %22 = dfschedule.config.create_io(%21, %16) {

          channel = 0,

          direction = "MM2S",

          io_operation = "SEND"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %23 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32

        %24 = dfschedule.schedule.start_io(%22, %23) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 0 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}

        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}

        %27 = dfschedule.config.load_kernel_group(%7, %16) {

          callee = [@dskernel_receiver],

          distributed_compute_kernel_args = [@compute0, @compute0],

          distributed_args = [@kernelconfig0, @kernelconfig1]

        } : (!dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup

        %28 = dfschedule.schedule.launch_kernel_group(%27) : (!dfschedule.kernelgroup) -> !dfschedule.event

        %29 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32

        %30 = dfschedule.schedule.start_io(%6, %29) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        dfschedule.schedule.wait(%28, %30) : (!dfschedule.event, !dfschedule.event)

        dfschedule.free_device_mem %subview : memref<8x16xi8, strided<[16, 1], offset: 128>>

      }

      scf.yield

    } {routing_memo = "row"}

    memref.dealloc %alloc : memref<16x16xi8>

    return

  }

  dfschedule.dskernel_receiver @dskernel_receiver {

  }

}

" , what I need is to remove the routingcreate and merge the shim receiving logic to merge the multiple shim recieve into one

Gemini said
To canonicalize this MLIR and merge multiple shim receiving operations into a single logical block, you need to lift the repetitive DMA configuration logic out of the routing.RoutingCreate regions.

In MLIR, the standard approach for this kind of "de-nesting" and "merging" involves a combination of Region Hoisting and Op Folding. Here is the expert advice on the best solution:

1. The Strategy: Hoisting and Merging
The goal is to move the dfschedule.config.dma_bd and dfschedule.config.create_io ops that target the same dfschedule.tile (the Shim tile at col = 2, row = 0) into a single initialization or configuration sequence.

Step A: Hoist Invariant Tile Declarations
Currently, %3 = dfschedule.declaretile {col = 2, row = 0} is declared inside every region.

Action: Move the Shim tile declaration to the top of the func.func. This allows all subsequent DMA ops to reference a single SSA value for that physical tile.

Step B: Merge Shim IO Configurations
Your Shim tile (Tile 2,0) is receiving data on multiple channels (Channel 0 in the first block, Channel 1 in the second).

Action: Instead of calling create_io inside separate routing regions, create a single sequence where you define the BDs (Buffer Descriptors) for both channels.

Optimization: If the memory being transferred is contiguous in the source %alloc, you can potentially merge the two 128-byte transfers into one larger transfer if the hardware/dialect allows, or simply chain the BDs using the next_bd attribute.

2. Recommended Canonical Form
To eliminate the routing.RoutingCreate and merge the logic, your IR should look like this:

MLIR
func.func @main() {
  %shim_tile = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
  %0 = memref.get_global @my_constant : memref<16x16xi8>
  %alloc = memref.alloc() : memref<16x16xi8>
  memref.copy %0, %alloc : memref<16x16xi8> to memref<16x16xi8>

  // 1. Unified Shim Configuration
  // Map the whole buffer or specific views once
  %full_view = dfschedule.buffer_view %alloc {len = 256 : i64, offset = 0 : i64} : memref<16x16xi8> -> memref<16x16xi8>
  
  // Configure BD for Channel 0
  %bd0 = dfschedule.config.dma_bd(%full_view, %shim_tile, %c0) { len = 128, ... }
  %io0 = dfschedule.config.create_io(%bd0, %shim_tile) { channel = 0, direction = "S2MM" }

  // Configure BD for Channel 1
  %bd1 = dfschedule.config.dma_bd(%full_view, %shim_tile, %c1) { len = 128, offset = 128, ... }
  %io1 = dfschedule.config.create_io(%bd1, %shim_tile) { channel = 1, direction = "S2MM" }

  // 2. Compute/Core Configurations
  // (Place your kernel group loading and core BD configs here)

  // 3. Unified Execution
  %ev0 = dfschedule.schedule.start_io(%io0, %id0)
  %ev1 = dfschedule.schedule.start_io(%io1, %id1)
  %ev_kernel = dfschedule.schedule.launch_kernel_group(...)
  
  dfschedule.schedule.wait(%ev0, %ev1, %ev_kernel)
}
3. Implementation Steps
To automate this, I recommend writing a MLIR Transformation Pass rather than just a simple rewrite pattern:

Analyze Dependencies: Identify all dfschedule.config.create_io ops that share the same !dfschedule.tile.

Topological Hoisting: Move declaretile ops to the entry block of the function.

Buffer View Normalization: If multiple subviews are used for the same alloc, try to compose them or reference the base memref with offsets in the dma_bd op to simplify the SSA graph.

Inline Region logic: Use the mlir::PatternRewriter::inlineRegionBefore method to move the ops out of routing.RoutingCreate and then delete the routing op.

Why this is the "Best" solution
You stopped this response
You said
AS A MLIR EXPERT please advise what is the best solution to cannolize my IR "module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {

  memref.global "private" constant @my_constant : memref<16x16xi8> = dense<"0x0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF00">

  func.func @main() {

    %c1_i32 = arith.constant 1 : i32

    %c0_i32 = arith.constant 0 : i32

    %0 = memref.get_global @my_constant : memref<16x16xi8>

    %alloc = memref.alloc() : memref<16x16xi8>

    memref.copy %0, %alloc : memref<16x16xi8> to memref<16x16xi8>

    scf.execute_region {

      %1 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{

      ^bb0(%arg0: i32):

        %subview = memref.subview %alloc[0, 0] [8, 16] [1, 1] : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1]>>

        %3 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile

        %c0_i32_0 = arith.constant 0 : i32

        %4 = dfschedule.buffer_view %subview {len = 128 : i64, offset = 0 : i64} : memref<8x16xi8, strided<[16, 1]>> -> memref<8x16xi8, strided<[16, 1]>>

        %5 = dfschedule.config.dma_bd(%4, %3, %c0_i32_0) {

          offset = 0,

          len = 128,

          enable_packet = true,

          packet_id = 0,

          next_bd = 4294967295,

          acquire_lock_id = 0,

          acquire_lock_val = 0,

          release_lock_id = 0,

          release_lock_val = 0,

          data_id = 0

        } : (memref<8x16xi8, strided<[16, 1]>>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %6 = dfschedule.config.create_io(%5, %3) {

          channel = 0,

          direction = "S2MM",

          io_operation = "RECV"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %7 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile

        %subview_1 = memref.subview %subview[0, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1]>> to memref<4x16xi8, strided<[16, 1]>>

        %8 = dfschedule.memref_mapping %subview_1 : (memref<4x16xi8, strided<[16, 1]>>) -> memref<4x16xi8>

        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %c1_i32_2 = arith.constant 1 : i32

        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_2) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 0,

          next_bd = 0,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %c0_i32_3 = arith.constant 0 : i32

        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_3, %11) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 0,

          next_bd = 1,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle

        %13 = dfschedule.config.create_io(%12, %7) {

          channel = 0,

          direction = "MM2S",

          io_operation = "SEND"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32

        %15 = dfschedule.schedule.start_io(%13, %14) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        %16 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile

        %subview_4 = memref.subview %subview[4, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1]>> to memref<4x16xi8, strided<[16, 1], offset: 64>>

        %17 = dfschedule.memref_mapping %subview_4 : (memref<4x16xi8, strided<[16, 1], offset: 64>>) -> memref<4x16xi8>

        %18 = dfschedule.bind_core_buffer(%17, %16) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %19 = dfschedule.bind_core_buffer(%17, %16) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %c1_i32_5 = arith.constant 1 : i32

        %20 = dfschedule.config.dma_bd(%19, %16, %c1_i32_5) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 1,

          next_bd = 0,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %c0_i32_6 = arith.constant 0 : i32

        %21 = dfschedule.config.dma_bd(%18, %16, %c0_i32_6, %20) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 1,

          next_bd = 1,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle

        %22 = dfschedule.config.create_io(%21, %16) {

          channel = 0,

          direction = "MM2S",

          io_operation = "SEND"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %23 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32

        %24 = dfschedule.schedule.start_io(%22, %23) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 0 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}

        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}

        %27 = dfschedule.config.load_kernel_group(%7, %16) {

          callee = [@dskernel_receiver],

          distributed_compute_kernel_args = [@compute0, @compute0],

          distributed_args = [@kernelconfig0, @kernelconfig1]

        } : (!dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup

        %28 = dfschedule.schedule.launch_kernel_group(%27) : (!dfschedule.kernelgroup) -> !dfschedule.event

        %29 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32

        %30 = dfschedule.schedule.start_io(%6, %29) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        dfschedule.schedule.wait(%28, %30) : (!dfschedule.event, !dfschedule.event)

        dfschedule.free_device_mem %subview : memref<8x16xi8, strided<[16, 1]>>

      }

      %2 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{

      ^bb0(%arg0: i32):

        %subview = memref.subview %alloc[8, 0] [8, 16] [1, 1] : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1], offset: 128>>

        %3 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile

        %c0_i32_0 = arith.constant 0 : i32

        %4 = dfschedule.buffer_view %subview {len = 128 : i64, offset = 0 : i64} : memref<8x16xi8, strided<[16, 1], offset: 128>> -> memref<8x16xi8, strided<[16, 1], offset: 128>>

        %5 = dfschedule.config.dma_bd(%4, %3, %c0_i32_0) {

          offset = 0,

          len = 128,

          enable_packet = true,

          packet_id = 0,

          next_bd = 4294967295,

          acquire_lock_id = 0,

          acquire_lock_val = 0,

          release_lock_id = 0,

          release_lock_val = 0,

          data_id = 0

        } : (memref<8x16xi8, strided<[16, 1], offset: 128>>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %6 = dfschedule.config.create_io(%5, %3) {

          channel = 1,

          direction = "S2MM",

          io_operation = "RECV"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %7 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile

        %subview_1 = memref.subview %subview[0, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1], offset: 128>> to memref<4x16xi8, strided<[16, 1], offset: 128>>

        %8 = dfschedule.memref_mapping %subview_1 : (memref<4x16xi8, strided<[16, 1], offset: 128>>) -> memref<4x16xi8>

        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %c1_i32_2 = arith.constant 1 : i32

        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_2) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 0,

          next_bd = 0,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %c0_i32_3 = arith.constant 0 : i32

        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_3, %11) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 0,

          next_bd = 1,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle

        %13 = dfschedule.config.create_io(%12, %7) {

          channel = 0,

          direction = "MM2S",

          io_operation = "SEND"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32

        %15 = dfschedule.schedule.start_io(%13, %14) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        %16 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile

        %subview_4 = memref.subview %subview[4, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1], offset: 128>> to memref<4x16xi8, strided<[16, 1], offset: 192>>

        %17 = dfschedule.memref_mapping %subview_4 : (memref<4x16xi8, strided<[16, 1], offset: 192>>) -> memref<4x16xi8>

        %18 = dfschedule.bind_core_buffer(%17, %16) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %19 = dfschedule.bind_core_buffer(%17, %16) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>

        %c1_i32_5 = arith.constant 1 : i32

        %20 = dfschedule.config.dma_bd(%19, %16, %c1_i32_5) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 1,

          next_bd = 0,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle

        %c0_i32_6 = arith.constant 0 : i32

        %21 = dfschedule.config.dma_bd(%18, %16, %c0_i32_6, %20) {

          offset = 0,

          len = 64,

          enable_packet = true,

          packet_id = 1,

          next_bd = 1,

          acquire_lock_id = 0,

          acquire_lock_val = -1,

          release_lock_id = 1,

          release_lock_val = 1,

          data_id = -1

        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle

        %22 = dfschedule.config.create_io(%21, %16) {

          channel = 0,

          direction = "MM2S",

          io_operation = "SEND"

        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle

        %23 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32

        %24 = dfschedule.schedule.start_io(%22, %23) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 0 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}

        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}

        %27 = dfschedule.config.load_kernel_group(%7, %16) {

          callee = [@dskernel_receiver],

          distributed_compute_kernel_args = [@compute0, @compute0],

          distributed_args = [@kernelconfig0, @kernelconfig1]

        } : (!dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup

        %28 = dfschedule.schedule.launch_kernel_group(%27) : (!dfschedule.kernelgroup) -> !dfschedule.event

        %29 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32

        %30 = dfschedule.schedule.start_io(%6, %29) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event

        dfschedule.schedule.wait(%28, %30) : (!dfschedule.event, !dfschedule.event)

        dfschedule.free_device_mem %subview : memref<8x16xi8, strided<[16, 1], offset: 128>>

      }

      scf.yield

    } {routing_memo = "row"}

    memref.dealloc %alloc : memref<16x16xi8>

    return

  }

  dfschedule.dskernel_receiver @dskernel_receiver {

  }

}

" , what I need is to remove the routingcreate and merge the shim receiving logic to merge the multiple shim recieve into one

Gemini said
To canonicalize this IR effectively and merge the shim receiving logic, you should focus on Region Hoisting and SSA Value Normalization.

Currently, your IR is "fragmented" because each routing.RoutingCreate op essentially creates a local scope that hides redundant operations from global optimization passes. By flattening these regions and merging the redundant Shim tile configurations, you can significantly reduce the IR's complexity and the overhead of the final generated code.

The Canonical Solution: Global Hoisting and Unified IO
The best solution is to lift the static resource declarations (like tiles and constant descriptors) to the function entry block and consolidate the DMA configurations.

1. Hoist Resource Declarations
Move %3 = dfschedule.declaretile {col = 2, row = 0} out of the sub-regions. Since both routing blocks reference the same physical Shim tile, they should reference the same SSA value. This allows the compiler to see that the configuration is happening on a single hardware resource.

2. Consolidate Shim BDs and Channels
Instead of separate create_io calls spread across regions, you can define your Shim's behavior in a single block. Your current IR uses Channel 0 and Channel 1 on the same tile. You can merge these into a sequence of configurations that act on the base %alloc memref using different offsets.

Recommended "Canonical" IR Structure
Here is how the IR should look after removing the routing.RoutingCreate and merging the shim logic:

MLIR
func.func @main() {
  // --- 1. Global Resource Declarations ---
  %shim_tile = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
  %tile_r3 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
  %tile_r4 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
  // ... other tiles ...

  %0 = memref.get_global @my_constant : memref<16x16xi8>
  %alloc = memref.alloc() : memref<16x16xi8>
  memref.copy %0, %alloc : memref<16x16xi8> to memref<16x16xi8>

  // --- 2. Unified Shim Configuration ---
  // Create a single view or use offsets on the base memref
  %view = dfschedule.buffer_view %alloc {len = 256 : i64, offset = 0 : i64} : memref<16x16xi8> -> memref<16x16xi8>

  // Receive for Group 1 (Channel 0)
  %bd0 = dfschedule.config.dma_bd(%view, %shim_tile, %c0) { len = 128, offset = 0, ... }
  %io0 = dfschedule.config.create_io(%bd0, %shim_tile) { channel = 0, direction = "S2MM" }

  // Receive for Group 2 (Channel 1)
  %bd1 = dfschedule.config.dma_bd(%view, %shim_tile, %c1) { len = 128, offset = 128, ... }
  %io1 = dfschedule.config.create_io(%bd1, %shim_tile) { channel = 1, direction = "S2MM" }

  // --- 3. Merged Execution Logic ---
  // Instead of separate regions, we now have a linear schedule or a single scf.execute_region
  %ev0 = dfschedule.schedule.start_io(%io0, %shim_id)
  %ev1 = dfschedule.schedule.start_io(%io1, %shim_id)
  
  // Launch your kernel groups...
  %ev_k = dfschedule.schedule.launch_kernel_group(...)

  dfschedule.schedule.wait(%ev0, %ev1, %ev_k)
  
  memref.dealloc %alloc : memref<16x16xi8>
  return
}
Best Implementation Path
To achieve this programmatically, I recommend writing a Cleanup Pass using the following logic:

Tile Unification: Use a SymbolTable or a simple Map to track {col, row} pairs. Replace all duplicate declaretile ops with the first instance found in the function.

Region Inlining: Use mlir::PatternRewriter::inlineRegionBefore. This will "pull" the ops from inside routing.RoutingCreate into the main func.func body.

Dead Code Elimination (DCE): Once the ops are moved out, the routing.RoutingCreate ops will have no useful results and can be deleted.

Buffer Normalization: Identify memref.subview ops that point to the same base buffer. You can often merge these into a single larger view if the DMA engine supports strided access or simply adjust the offset attribute in the dma_bd ops.

Why this works
By merging the shim receives into one logical block, you allow the backend to potentially optimize the DMA descriptors (e.g., using a single multi-BD chain rather than two separate triggers). This significantly reduces the shim-to-core latency.

Would you like me to draft a C++ snippet for the MLIR rewrite pattern that performs this hoisting and merging?

You said
mlir::PatternRewriter::inlineRegionBefore: this method "pulls" the inner operations out of the routing.RoutingCreate body into func.func. What does that mean?

Gemini said
In the MLIR world, you can think of routing.RoutingCreate as a **"nested box"**, while func.func is the big room you work in.

Calling inlineRegionBefore is essentially an **"unbox and take out the contents"** operation.

1. Core meaning: Scope Flattening
In MLIR, Operations are hierarchically nested. Your original IR structure looks like this:

func.func (the big room)

arith.constant

routing.RoutingCreate (the box)

Region (the space inside the box)

Block

dfschedule.config.dma_bd (tool A in the box)

dfschedule.config.create_io (tool B in the box)

"Pulling in" (Inlining) means:
You take "tool A" and "tool B" out of the box and put them directly on the floor of the "big room". After the operation, the "parent" of these tools is no longer routing.RoutingCreate but func.func.

2. Side-by-side code comparison
Before inlining (nested state)
MLIR
func.func @main() {
  %c0 = arith.constant 0 : i32
  // At this point dma_bd lives inside RoutingCreate's isolated region
  %result = routing.RoutingCreate (scf_idx = %c0) {
    %bd = dfschedule.config.dma_bd ... 
    routing.yield %bd
  }
  return
}
After inlining (flat state)
MLIR
func.func @main() {
  %c0 = arith.constant 0 : i32
  // RoutingCreate has been "unboxed"; dma_bd now belongs directly to func.func
  %bd = dfschedule.config.dma_bd ... 
  // The box is thrown away; the remaining logic is wired directly into the body
  return
}
3. What happens at the C++ level?
When you write rewriter.inlineRegionBefore(sourceRegion, targetBlock, insertionPoint):

Move Blocks: it moves all Blocks in sourceRegion directly to the given position in targetBlock.

Change the parent: getParentOp() on these Ops used to return routing.RoutingCreate; now it returns func.func.

Remove the encapsulation: this addresses your earlier requirement of "removing routing.RoutingCreate". Once the inner operations are pulled out, the routing op becomes a meaningless empty shell, and you can safely delete it with rewriter.eraseOp(op).

4. Why do this?
As an expert, the biggest benefit I can point out is that it breaks optimization barriers:

Global visibility: while ops are inside the box, MLIR's regular optimizations (e.g. CSE to eliminate duplicate operations, dead-code elimination) have a hard time looking across Region boundaries.

Resource merging: just as you wanted to merge multiple Shim receive paths earlier, only once they are pulled to the same level (the same Block) can the compiler notice: "Hey, these two DMAs configure the same Tile, I can merge them!"

Would you like to see a concrete PatternRewrite class that shows how to write this C++ code?

You said
rewriter.inlineRegionBefore: give an example

Gemini said
Sure, let's go straight to the code. Suppose we want to write a Canonicalization pattern dedicated to taking apart the routing.RoutingCreate "box".

In MLIR, inlineRegionBefore is usually accompanied by result replacement (connecting the values yielded inside the box to the SSA variables outside the box).

C++ example code
This is a typical OpRewritePattern implementation:

C++
#include "mlir/IR/PatternMatch.h"
#include "your/dialect/RoutingOps.h" // assuming your Dialect is defined here

struct FlattenRoutingPattern : public mlir::OpRewritePattern<routing::RoutingCreateOp> {
    using OpRewritePattern<routing::RoutingCreateOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(routing::RoutingCreateOp op,
                                       mlir::PatternRewriter &rewriter) const override {
        // 1. Get the "space" (Region) inside the box
        mlir::Region &sourceRegion = op.getRegion();

        // 2. Find the "room location" (Block) where the box sits
        mlir::Block *currentBlock = op->getBlock();

        // 3. "Unbox": move all Blocks of the box in front of the box
        // This method "pulls" everything in sourceRegion up to the level of currentBlock
        rewriter.inlineRegionBefore(sourceRegion, op->getBlock());

        // 4. Rewiring (Handling Yield)
        // There is usually a routing.yield inside the box that passes values out.
        // We need to redirect uses of op's results to the yield's operands instead.
        mlir::Block &inlinedBlock = currentBlock->getPredecessor()->getNextNode(); // simplified
        auto yieldOp = mlir::cast<routing::RoutingYieldOp>(inlinedBlock.getTerminator());
        
        // Replace the results of the whole RoutingCreateOp with the yielded values
        rewriter.replaceOp(op, yieldOp.getOperands());

        // 5. Final cleanup: delete the now-useless yield op (it is now directly exposed in func)
        rewriter.eraseOp(yieldOp);

        return mlir::success();
    }
};
IR changes before and after this code runs
Before
Your IR looks like "nesting dolls":

MLIR
func.func @main() {
  %0 = routing.RoutingCreate ... {
    ^bb0:
      %1 = dfschedule.config.dma_bd ... // at this point the "parent" of this line is RoutingCreate
      routing.yield %1
  }
  // other code...
}
After running inlineRegionBefore
The structure is flattened; dma_bd has been "promoted":

MLIR
func.func @main() {
  // The box is opened; bb0 becomes part of func.func
  %1 = dfschedule.config.dma_bd ... // now the "parent" of this line is func.func directly
  
  // The original %0 is replaced directly by %1
  // routing.yield has been deleted
  // other code...
}
Key points
Block movement: inlineRegionBefore does not copy Ops one by one; it moves whole Block pointers directly. This is very efficient.

Control-flow conversion: if the box contains multiple Blocks, once pulled out they merge directly into the outer control flow.

SSA continuity: after pulling them out, you must make sure the subsequent operations that depended on RoutingCreate's results now correctly get the values computed inside the box. That is what rewriter.replaceOp is for.

Expert tip
Once you have pulled these operations up to the func.func level, you can write another Merge Pattern. That Pattern scans all dfschedule.config.create_io in the same Block and, if it finds that their tile attributes are identical, uses the rewriter to merge them into one.

Next, do you want me to write the C++ logic for "merging the Shim receive logic of two identical Tiles"?

You said
How do I use FlattenRoutingPattern?

Gemini said
To use FlattenRoutingPattern you cannot "call" it directly. In MLIR, a Pattern must be put into a **"set" (RewritePatternSet)** and then driven by a **Pass**.

You can deploy it in the following three steps:

Step 1: create a Pass to host the pattern
In MLIR, a Pass is the smallest unit that executes a transformation. You need to create an OperationPass (usually on func::FuncOp or ModuleOp).

C++
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Define your Pass
struct FlattenRoutingPass : public mlir::PassWrapper<FlattenRoutingPass, OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenRoutingPass)

  void runOnOperation() override {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    
    // Add the pattern we defined earlier to the set
    patterns.add<FlattenRoutingPattern>(&getContext());

    // Apply these patterns with the Greedy driver
    // It keeps scanning the IR until no RoutingCreate is left to Flatten
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  llvm::StringRef getArgument() const final { return "flatten-routing"; }
  llvm::StringRef getDescription() const final { return "Remove routing.RoutingCreate and hoist ops."; }
};
Step 2: register the Pass in the pipeline
Once you have the Pass, you need to tell the compiler when to run it. Usually you register it in your main function or in a PassPipeline:

C++
void registerMyPasses() {
    mlir::PassRegistration<FlattenRoutingPass>();
}

// Or add it directly to a PassManager
mlir::PassManager pm(&context);
pm.addPass(std::make_unique<FlattenRoutingPass>());
Step 3: invoke it from the command line
If you have registered the Pass with the mlir-opt tool, you can run it directly on your IR file:

Bash
# Run the pass you defined
my-mlir-opt --flatten-routing input.mlir -o output.mlir
Expert advice: how to implement the "merge" logic?
After FlattenRoutingPattern has "pulled" all operations into func.func, you will find multiple dfschedule.config.create_io in the IR targeting the same Tile.

To achieve what you originally wanted, **"merging multiple Shim receive paths"**, you should write another pattern: MergeShimIOOpsPattern.

Its logic should be:

Match dfschedule.config.create_io.

Check whether its tile is a Shim Tile (e.g. 2,0).

Look in the same Block for another create_io already configured for the same Tile.

If one exists, link the two BD chains together (via the next_bd attribute) and delete the redundant create_io.

Summary
Flatten (unbox): use inlineRegionBefore to take the logic out of RoutingCreate.

DCE (cleanup): delete the empty RoutingCreate.

Merge: at the same level, use another Pattern to identify and merge configurations of the same resource.

Do you want me to write pseudocode for the Merge Pattern that "identifies and merges identical Tile configurations"?

You said
GREEDY PATTERN MATCH breaks down on a million lines of code -- what problems will it have, and how do you fix them?

Gemini said
When processing million-line MLIR code, applyPatternsAndFoldGreedily really does turn from "the compiler's helper" into "a compile-time nightmare". As an expert I have to warn you: on very large IR the greedy driver faces three main challenges -- time complexity, memory churn, and convergence stability.

1. Core problems at a million lines
🌀 Time-complexity blow-up (Non-linear Complexity)
The Greedy Driver works by maintaining a Worklist. Whenever an Op is modified, the driver puts the affected neighboring Ops back onto the list.

Repeated scanning: at a million lines, if a Pattern is badly written (e.g. it triggers local modifications frequently), a single Op may be visited thousands of times.

Cost of convergence: greedy matching must run to a Fixed Point before it stops. For very large IR, the number of steps to reach that stable state can grow exponentially.

💾 Memory and cache pressure
Worklist bloat: a million lines means the worklist may suddenly be stuffed with hundreds of thousands of Op pointers, consuming a lot of memory.

Cache misses: the greedy driver jumps back and forth between different Blocks and even different Regions, giving a very low CPU cache hit rate.

🔄 Oscillation and infinite loops (Infinite Loops)
In small IR, the logic bug of Pattern A turning into Pattern B and Pattern B turning back into Pattern A is easy to spot. In million-line IR, such a "cycle" can hide in complex logic branches and make the compiler simply hang.

2. Expert-level solutions
For large IR we usually abandon or restrict the "greedy mode" and switch to more deterministic methods.

🚀 Option A: use an explicit walk instead of the greedy driver
This is the first choice for very large code. Rather than letting the driver "bump around", control the traversal path yourself.

Logic: use op->walk<WalkOrder::PreOrder>([](OpType op) { ... }).

Advantage: each Op is visited exactly once, so the complexity stays strictly O(N). Your earlier FlattenRoutingPattern only needs a single hoist, so greedy mode is completely unnecessary.

🛠️ Option B: fine-tune GreedyRewriteConfig
If you must use greedy mode, you can put "shackles" on it through configuration options:

C++
GreedyRewriteConfig config;
config.useTopDownTraversal = true; // top-down traversal, usually converges faster
config.maxIterations = 10;          // hard cap on iterations, prevents infinite loops
config.strictMode = GreedyRewriteStrictness::ExistingOps; // only process existing Ops, not newly created ones
📦 Option C: parallelize (Pass Manager)
Don't try to greedily process everything in a single ModulePass.

Strategy: push the transformation down to the FunctionPass level.

Rationale: MLIR's PassManager runs function-level Passes in parallel on multiple threads. That limits the IR each thread handles to the ten-thousand-line range and makes full use of a multi-core CPU.

🔍 Option D: targeted jumps based on the SymbolTable
Since you want to merge the Shim logic for the same Tile, don't make the driver traverse every Op.

Optimization: at the start of the Pass, build an index Map<Tile, SmallVector<Op>>.

Operation: operate directly on this index; the time complexity of the merge logic drops from O(N
2
 ) to close to O(N).

3. Performance comparison
Aspect	Greedy mode (Default)	Hand-written Walk / targeted optimization
Time at a million lines	Possibly minutes, or even a timeout	Seconds
Memory use	Fluctuates wildly with the Worklist	Stable and linear
Convergence	Depends on Pattern quality	100% deterministic
Typical use	Complex algebraic simplification (Arith)	Structural transformations (Hoisting/Merge)
Recommendation
For your current RoutingCreate flattening and Shim merging tasks, Option A (Walk) or Option D (indexed merge) is strongly recommended. That keeps your compiler running fast even on large neural-network topologies.

Would you like me to show how to turn the earlier Pattern into an efficient walk form?

You said
In this case (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {

signalPassFailure();

} how should this be changed?

Gemini said
For million-line code, the best way to replace applyPatternsAndFoldGreedily is to go **"from broad sweep to targeted point"**.

Rather than having the compiler look for a needle in a haystack, do a **"surgical" targeted traversal**. Here are two improvements, in order of preference:

Option 1: an O(N) targeted traversal with walk (most recommended)
For structural transformations such as "hoisting operations" or "deleting a specific box", pattern matching is not needed at all. With walk, each Op is guaranteed to be visited exactly once, and the complexity is strictly linear.

Modified code example:
C++
void runOnOperation() override {
    auto func = getOperation();
    mlir::IRRewriter rewriter(&getContext());

    // 1. Find all RoutingCreateOps directly; PreOrder ensures unboxing from the outside in
    func.walk<mlir::WalkOrder::PreOrder>([&](routing::RoutingCreateOp op) {
        // Run your earlier Flatten logic
        mlir::Region &sourceRegion = op.getRegion();
        
        // Get the inner yield info (assuming you defined RoutingYieldOp)
        auto yieldOp = mlir::cast<routing::RoutingYieldOp>(sourceRegion.front().getTerminator());
        
        // Record the yielded values
        mlir::ValueRange yieldedValues = yieldOp.getOperands();

        // [Key change]: inline manually, no longer relying on the Greedy Driver
        rewriter.inlineBlockBefore(&sourceRegion.front(), op);

        // Replace the original RoutingCreate's results
        rewriter.replaceOp(op, yieldedValues);
        
        // Delete the yield placeholder that has been pulled out
        rewriter.eraseOp(yieldOp);
    });
}
Why does this solve the problem?

Zero repeated scanning: it does not recheck the ten thousand Ops around an Op just because that Op changed.

Determinism: the execution flow is clear, and non-convergence cannot happen.

Option 2: if you must use Patterns, limit how "greedy" they are
If you have some very complex folding logic that must use Patterns, be sure to configure GreedyRewriteConfig.

Modified code example:
C++
void runOnOperation() override {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FlattenRoutingPattern>(&getContext());

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true; // top-down traversal, fewer re-enqueues
    config.maxIterations = 2;          // hard cap on iterations; if 2 are not enough, the logic is wrong
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps; // only process existing ops, not newly created ones

    // Pass config explicitly
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), config);
}
Final advice for your "merge Shim receive" goal
When you process million-line IR and need to merge configurations for the same Tile, the most efficient approach is **"Indexing"**:

Step 1 (Flatten): use "Option 1 (Walk)" above to take apart all RoutingCreates.

Step 2 (Collect): in the same walk, store the configurations belonging to the same Tile in a DenseMap<TileAttr, SmallVector<CreateIOOp>>.

Step 3 (Merge): loop-merge only over the SmallVectors in this Map.

This way the time to process million-line IR drops from **"minutes" straight to "milliseconds"**.

💡 Pitfall guide
In large-scale IR transformations, never modify the SymbolTable (e.g. renaming global variables) while applyPatternsAndFoldGreedily is running; it invalidates the driver's state over and over.

Apart from taking the box apart, does your current code logic also need to do complex algebraic computation on the inner data (like 1+1=2)? If not, drop Greedy mode without hesitation.

You said
What are the downsides? Isn't it a bit HARDCODED?

Gemini said
Your instinct is very sharp. It is indeed more "hardcoded" -- or, in compiler terms, it regresses from "Declarative" to "Procedural".

In the eyes of an MLIR expert, this approach trades flexibility for maximum performance. Here are the notable downsides of this "scalpel" approach, and why it feels "hard" to you:

1. Why it is a bit "hard" (Disadvantages)
🌀 No "chain reaction" (Missing Recursive Folding)
The most powerful thing about the Greedy Driver is that it handles the "domino" effect.

Example: if taking a box apart turns %1 + %2 inside it into 1 + 1, greedy mode automatically turns 1 + 1 into 2 for you.

Walk mode: it only takes boxes apart. Afterwards, 1 + 1 inside is still 1 + 1 unless you write another walk to handle additions. You have to predict and plan every follow-up action by hand.

🧩 Very poor composability (Poor Composability)
Pattern matching: you can easily combine three people's optimization logic with patterns.add<A, B, C>().

Walk mode: if you want to do other optimizations while taking boxes apart, you have to cram all the logic into that huge lambda. The code quickly becomes a tangled mess of "spaghetti" that is hard to maintain.

🛡️ Fragility and safety (Fragility)
Pattern matching: managed by PatternRewriter, which takes care of a lot of low-level pointer updates and legality checks for you.

Walk mode: you are manipulating memory Blocks directly. If you accidentally delete an Op that is still referenced, or forget to handle the Terminator, the compiler simply crashes (Segfault) instead of giving you a friendly error message.

2. In-depth comparison: elegance vs. efficiency
Feature	Greedy mode (Patterns)	Manual traversal (Walk)
Mental model	"When you see A, turn it into B"	"Walk from the first line to the last, take apart every box encountered"
Automatic folding	Strong (handles related changes automatically)	None (must be triggered manually)
At a million lines	Prone to hanging or very slow	Very fast (linear time)
Code size	Concise, highly abstract	Verbose, lots of low-level operations
Hardcoded feel	Low (rule engine)	High (procedural script)
3. The expert's middle way: Dialect Conversion
If Greedy feels too slow and Walk too hard, MLIR has one more "ultimate weapon": Dialect Conversion (the legalization framework).

It is designed for large-scale conversion. You set up a ConversionTarget and tell it which Ops are "illegal" (e.g. RoutingCreate). The framework sweeps the IR once and converts only the illegal Ops.

Pros: faster than greedy mode because its target is explicit; more elegant than Walk because it is still written with Patterns.

When to use: when you need to change the IR structure at scale but don't want to write that very low-level inlineBlock logic.

Summary of recommendations
If you are dealing with million-line IR now:

Early development: use Greedy first to verify the logic is correct, even if it takes a minute to run.

Production optimization: if the speed is unacceptable, switch to Walk for operations that are structurally simple but very numerous (like your RoutingCreate hoisting); for complex computational optimizations, keep Greedy but limit its scope.

For your Shim-merging requirement, do you plan to do it in a single C++ Pass, or as a pipeline of several steps (Passes)?

You said
What does mlir::WalkOrder::PreOrder mean?

Gemini said
Simply put, mlir::WalkOrder::PreOrder determines the "visit order" when you traverse a nested tree structure like MLIR.

In MLIR, the IR is nested layer by layer: Module -> Function -> Block -> Operation -> Region -> Block -> Operation.

1. Core definition: parent first, then children
PreOrder (pre-order traversal): visit the Parent Op first, then enter its Regions/Blocks to visit the Child Ops inside.

PostOrder (post-order traversal - the default): visit all the inner child operations first, and only then return to the parent op itself.

2. Why must you use PreOrder when "unboxing"?
In your earlier scenario (deleting routing.RoutingCreate), PreOrder is the only correct choice.

Scenario logic: you want to take the contents out of RoutingCreate and then delete the "shell".

Flow with PreOrder:

The walker reaches RoutingCreate.

Your function fires: you see this is a box, so you "promote" its contents to the outer level.

At this point the inner contents have already been handled.

If you use PostOrder (the default):

The walker first dives deep into the box.

It goes and optimizes every small operation inside the box (e.g. the arith.constant inside).

Only after all of that does it come back out to handle RoutingCreate.

Risk: if there are nested boxes inside the box, PostOrder makes you start unboxing from the deepest level, which is logically very inefficient and error-prone.

3. Code behavior comparison
Suppose you have IR like this:

MLIR
%0 = "outer_op"() ({
  %1 = "inner_op"() : () -> i32
}) : () -> i32
Traversal order	Visit sequence	Where your logic fires
PreOrder	1. outer_op -> 2. inner_op	You see the big box first and just take it apart -- easy.
PostOrder	1. inner_op -> 2. outer_op	You stare at the parts first and only see the big box at the end.
4. Key API differences
In C++, you can specify the order explicitly:

C++
// The default is PostOrder: children are handled first
op->walk([](MyOp op) { ... }); 

// Explicit PreOrder: parents are handled first
op->walk<mlir::WalkOrder::PreOrder>([](MyOp op) { ... }); 
5. Potential pitfalls
With PreOrder, be especially careful: do not arbitrarily delete nodes other than the current one while walking.
Because PreOrder still has to visit the child nodes, if you delete the whole Region while handling the parent and the walker still intends to go inside, you get a null-pointer crash.

Expert tip: if you delete the current Op or its Region in a PreOrder step, you must return WalkResult::skip() to tell the walker: "I've already taken this apart, don't go inside."

Now that you understand traversal order, do you want me to write a safe unboxing routine combining WalkOrder::PreOrder and WalkResult::skip()?

You said
When should I use POSTORDER and when PREORDER?

Gemini said
In compiler development, choosing the traversal order is like the difference between **"demolishing a house"** and **"assembling parts"**. Choose the wrong order and the code is not only inefficient but also prone to crashing.

As an MLIR expert, here is a set of survival rules I put together for you:

1. When to use PREORDER (Top-Down)?
Core logic: the parent operation decides the fate of the child operations.

If what you are doing is a **"structural"** modification, or what you care about is the **"context"**, choose PreOrder.

Hoisting / Flattening: e.g. your earlier task of taking things out of routing.RoutingCreate. You must see the "box" first before you can decide to pull its contents out.

Scope Initialization: if you are handling nested scopes and the children depend on symbols or attributes defined by the parent.

Deleting a whole region (Deletion): if you want to delete a whole Region based on some attribute of the parent, seeing the parent first lets you skip the useless traversal of all its children.

⚠️ Note: after deleting the current node in PreOrder, you must return WalkResult::skip(); otherwise the walker tries to enter a region that no longer exists and you get a Segfault.

2. When to use POSTORDER (Bottom-Up)?
Core logic: the parent operation's result depends on its child operations.

This is MLIR's default order and the first choice for most local optimizations (Canonicalization).

Arithmetic Folding:

Example: add(mul(1, 2), 3).

You must first handle the inner mul(1, 2) -> 2; only then can the parent add become add(2, 3) -> 5.

Dead-code elimination (DCE): only when you know that none of an operation's "Users" (i.e. child or subsequent nodes) need it any more can you safely delete it.

Type Inference: a parent operation's result type is usually determined by its operands (the values provided by the children).

3. Expert comparison table
Feature	PREORDER (pre-order)	POSTORDER (post-order)
Traversal path	Root → leaf	Leaf → root
Intuitive analogy	Demolition: take down the outer walls first, then deal with the interior.	Assembly line: build the parts first, then assemble.
Main uses	Structural transformation, inlining, hoisting, pushing down permissions.	Computational optimization, constant folding, DCE, type inference.
Effect on IR modification	Good for deleting whole Regions or moving Ops.	Good for simplifying a complex Op tree into a constant.
Performance advantage	In large IR, skip() lets you avoid huge numbers of nodes.	Guarantees that when the current Op is handled, its inputs are already in simplest form.
4. Summary: which one to choose?
A simple criterion: "Do I need to see the big picture first?"

If your logic is "because this box is of Routing type, I will take its logic ..." → PREORDER.

If your logic is "if both addends are constants, I sum them ..." → POSTORDER.

💡 Advanced technique: parallelism and skipping
When processing a million lines of code, if you use PreOrder together with an index, you can avoid walking some Blocks entirely. For your Shim Tile logic, for example, you can first search for all Shim Tile declarations (PreOrder) and then "jump" directly to their associated operations, which is much faster than aimlessly walking the whole IR.

Now that you have the ordering logic down, would you like me to write a template for a safe unboxing (Flatten) function with WalkResult::skip()? It can keep you from crashing unexpectedly while walking a million lines of code.

You said
THIS IS MY CUrrent canonicalize logic , please let me know the problem "/******************************************************************************

* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

* SPDX-License-Identifier: Apache-2.0

******************************************************************************/



#include "passschedulecanonicalize.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/Builders.h"

#include "mlir/IR/BuiltinDialect.h"

#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"

#include "llvm/ADT/DenseSet.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/raw_ostream.h"

#include "dfschedulemanager.h"

#include <map>

#include <set>

#include <vector>

#include <iostream>



using namespace mlir;

using namespace dfschedule;



namespace {



// Key for tile identification: (col, row)

using TileKey = std::pair<int64_t, int64_t>;



// Structure to hold collected schedule information per tile

struct TileScheduleInfo {

    TileKey key;

    SmallVector<Value> tileValues;           // All tile values for this (col, row)

    SmallVector<Operation*> packetOps;       // Packet ops targeting this tile

    SmallVector<SymbolRefAttr> packetSymbols; // Packet symbols for this tile (OLD)

    SmallVector<DictionaryAttr> configDicts;  // Config dictionaries for this tile (NEW)

    SmallVector<SymbolRefAttr> computeKernelArgs; // Compute args for this tile

    bool isShimTile = false;

    bool isCoreTile = false;

};



// Structure to hold shim tile DMA info

struct ShimDmaInfo {

    TileKey key;

    Value tileValue;

    SmallVector<Operation*> dmaBdOps;

    SmallVector<Operation*> createIoOps;

    SmallVector<Value> bdHandles;

    SmallVector<Value> ioHandles;

};



// Structure to hold tensor slice parameters

struct SliceParams {

    size_t index;  // Unique index for this slice

    RankedTensorType sourceType;

    SmallVector<int64_t, 4> offsets;

    SmallVector<int64_t, 4> sizes;

    SmallVector<int64_t, 4> strides;

    RankedTensorType resultType;

    size_t partitionIndex;  // Index of the partitiontensor this slice ultimately comes from

    int64_t parentSliceIndex;  // Index of parent slice if nested, -1 if directly from partition

    bool isFromPartition;  // true if immediate source is partitiontensor, false if from another slice

};



// Structure to hold partition tensor parameters

struct PartitionParams {

    size_t index;  // Unique index to distinguish partitions with same params but different flows

    RankedTensorType tensorType;

    int64_t splitnum;

    int64_t splitdim;

    std::string hw_axis_owner;

    std::string replicate_on;

    std::string single_tile_owner;

};



// Structure to hold operation with parent information

struct OpWithParent {

    Operation* op;

    Operation* parent;  // Parent operation (can be nullptr if top-level)

    bool isInDSKernelReceiver;  // true if parent is dfschedule.dskernel_receiver

    

    OpWithParent(Operation* operation, Operation* parentOp = nullptr)

        : op(operation), parent(parentOp) {

        // Check if parent is dskernel_receiver

        isInDSKernelReceiver = false;

        if (parentOp && parentOp->getName().getStringRef() == "dfschedule.dskernel_receiver") {

            isInDSKernelReceiver = true;

        }

    }

};



// Collected module-level schedule info

struct ModuleScheduleInfo {

    // Map from (col, row) to tile info

    std::map<TileKey, TileScheduleInfo> coreTiles;

    std::map<TileKey, ShimDmaInfo> shimTiles;

    

    // All collected operations with parent information

    SmallVector<OpWithParent> declareTensorOps;

    SmallVector<OpWithParent> declareTileOps;

    SmallVector<OpWithParent> configDmaBdOps;

    SmallVector<OpWithParent> configCreateIoOps;

    SmallVector<OpWithParent> packetOps;

    SmallVector<OpWithParent> loadKernelGroupOps;

    SmallVector<OpWithParent> launchKernelGroupOps;

    SmallVector<OpWithParent> getBdIdOps;

    SmallVector<OpWithParent> startIoOps;

    SmallVector<OpWithParent> scheduleWaitOps;

    SmallVector<OpWithParent> dskernelReceiverOps;

    

    // Operations to move from func.func main (with parent information)

    SmallVector<OpWithParent> tensorEmptyOps;

    SmallVector<OpWithParent> extractSliceOps;

    SmallVector<OpWithParent> partitionTensorOps;

    SmallVector<OpWithParent> executeRegionOps;

    SmallVector<OpWithParent> routingCreateOps;

    SmallVector<OpWithParent> declareDataOps;        // dfscheblueprint.declare_data

    SmallVector<OpWithParent> topLevelConstantOps;   // arith.constant at top level of main

    

    // Unique source tensor types (deduplicated)

    SmallVector<RankedTensorType> sourceTensorTypes;

    

    // Map from tensor type string key to its init tensor Value

    std::map<std::string, Value> initTensorMap;

    

    // Unique partition params (deduplicated)

    SmallVector<PartitionParams> uniquePartitionParams;

    

    // Unique extract_slice params (deduplicated)

    SmallVector<SliceParams> uniqueSliceParams;

    

    // Events to wait for

    SmallVector<Value> allEvents;

    

    // Kernel info

    StringRef kernelName = "dskernel_receiver";

    RankedTensorType kernelTensorType;

    int64_t bufferLen = 0;

    uint32_t basePacketId = 0;

    

    // Track how many packet streams each tile needs

    std::map<TileKey, int> tilePacketCount;

};



// Helper: Extract (col, row) from DeclareTileOp

static TileKey getTileKey(dfschedule::DeclareTileOp op) {

    return {op.getCol(), op.getRow()};

}



// Helper: Check if a tile is a shim tile (row == 0)

static bool isShimTile(TileKey key) {

    return key.second == 0;

}



// Helper: Get the parent operation (walks up the region hierarchy)

static Operation* getParentOp(Operation *op) {

    if (!op) return nullptr;

    Region *region = op->getParentRegion();

    if (!region) return nullptr;

    return region->getParentOp();

}



// Collect all dfschedule operations from the module

static void collectScheduleOps(ModuleOp moduleOp, ModuleScheduleInfo &info) {

    moduleOp.walk([&](Operation *op) {

        // Get parent operation

        Operation *parentOp = getParentOp(op);



        // DeclareTensorOp removed from dialect (tensor path deprecated)

        if (auto declareTile = dyn_cast<dfschedule::DeclareTileOp>(op)) {

            info.declareTileOps.push_back(OpWithParent(op, parentOp));

            TileKey key = getTileKey(declareTile);

            

            if (isShimTile(key)) {

                if (info.shimTiles.find(key) == info.shimTiles.end()) {

                    info.shimTiles[key] = ShimDmaInfo{key, declareTile.getTile(), {}, {}, {}, {}};

                }

            } else {

                if (info.coreTiles.find(key) == info.coreTiles.end()) {

                    info.coreTiles[key] = TileScheduleInfo{key, {}, {}, {}, {}, {}, false, true};

                }

                info.coreTiles[key].tileValues.push_back(declareTile.getTile());

            }

        } else if (auto configDmaBd = dyn_cast<dfschedule::ConfigDmaBdOp>(op)) {

            info.configDmaBdOps.push_back(OpWithParent(op, parentOp));

        } else if (auto createIo = dyn_cast<dfschedule::ConfigCreateIoOp>(op)) {

            info.configCreateIoOps.push_back(OpWithParent(op, parentOp));

        } else if (auto packet = dyn_cast<dfschedule::PacketOp>(op)) {

            info.packetOps.push_back(OpWithParent(op, parentOp));

        } else if (auto loadKernel = dyn_cast<dfschedule::LoadKernelGroupOp>(op)) {

            info.loadKernelGroupOps.push_back(OpWithParent(op, parentOp));

        } else if (auto launchKernel = dyn_cast<dfschedule::LaunchKernelGroupOp>(op)) {

            info.launchKernelGroupOps.push_back(OpWithParent(op, parentOp));

            info.allEvents.push_back(launchKernel.getEvent());

        } else if (auto getBdId = dyn_cast<dfschedule::GetBdIdOp>(op)) {

            info.getBdIdOps.push_back(OpWithParent(op, parentOp));

        } else if (auto startIo = dyn_cast<dfschedule::StartIoOp>(op)) {

            info.startIoOps.push_back(OpWithParent(op, parentOp));

            info.allEvents.push_back(startIo.getEvent());

        } else if (auto wait = dyn_cast<dfschedule::ScheduleWaitOp>(op)) {

            info.scheduleWaitOps.push_back(OpWithParent(op, parentOp));

        } else if (auto receiver = dyn_cast<dfschedule::DSKernelReceiverOp>(op)) {

            info.dskernelReceiverOps.push_back(OpWithParent(op, parentOp));

        } else if (auto emptyOp = dyn_cast<tensor::EmptyOp>(op)) {

            // Collect tensor.empty ops

            info.tensorEmptyOps.push_back(OpWithParent(op, parentOp));

            auto tensorType = cast<RankedTensorType>(emptyOp.getType());

            // Track unique tensor types

            bool found = false;

            for (auto &existingType : info.sourceTensorTypes) {

                if (existingType == tensorType) {

                    found = true;

                    break;

                }

            }

            if (!found) {

                info.sourceTensorTypes.push_back(tensorType);

            }

        } else if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(op)) {

            // Collect tensor.extract_slice ops - NO deduplication!

            // Each slice is unique even if it has same offsets/sizes

            info.extractSliceOps.push_back(OpWithParent(op, parentOp));

            

            // Create slice params

            SliceParams params;

            params.index = info.uniqueSliceParams.size();  // Unique index

            params.sourceType = cast<RankedTensorType>(extractSlice.getSource().getType());

            for (auto o : extractSlice.getStaticOffsets()) params.offsets.push_back(o);

            for (auto s : extractSlice.getStaticSizes()) params.sizes.push_back(s);

            for (auto s : extractSlice.getStaticStrides()) params.strides.push_back(s);

            params.resultType = cast<RankedTensorType>(extractSlice.getType());

            params.partitionIndex = 0;

            params.parentSliceIndex = -1;  // Default: no parent slice

            params.isFromPartition = false;

            

            // Check immediate source

            Value source = extractSlice.getSource();

            if (Operation *defOp = source.getDefiningOp()) {

                if (defOp->getName().getStringRef() == "routing.partitiontensor") {

                    // Direct source is partitiontensor

                    params.isFromPartition = true;

                    for (size_t i = 0; i < info.partitionTensorOps.size(); ++i) {

                        if (info.partitionTensorOps[i].op == defOp) {

                            params.partitionIndex = i;

                            break;

                        }

                    }

                } else if (auto parentSlice = dyn_cast<tensor::ExtractSliceOp>(defOp)) {

                    // Direct source is another slice - find its index

                    params.isFromPartition = false;

                    for (size_t i = 0; i < info.extractSliceOps.size(); ++i) {

                        if (info.extractSliceOps[i].op == defOp) {

                            params.parentSliceIndex = static_cast<int64_t>(i);

                            break;

                        }

                    }

                    

                    // Walk up to find the ultimate partitiontensor

                    Value walkSource = parentSlice.getSource();

                    while (Operation *walkDefOp = walkSource.getDefiningOp()) {

                        if (walkDefOp->getName().getStringRef() == "routing.partitiontensor") {

                            for (size_t i = 0; i < info.partitionTensorOps.size(); ++i) {

                                if (info.partitionTensorOps[i].op == walkDefOp) {

                                    params.partitionIndex = i;

                                    break;

                                }

                            }

                            break;

                        } else if (auto grandParentSlice = dyn_cast<tensor::ExtractSliceOp>(walkDefOp)) {

                            walkSource = grandParentSlice.getSource();

                        } else {

                            break;

                        }

                    }

                }

            }

            

            // Always add - no deduplication for slices

            info.uniqueSliceParams.push_back(params);

        } else if (auto execRegion = dyn_cast<scf::ExecuteRegionOp>(op)) {

            // Collect scf.execute_region ops

            info.executeRegionOps.push_back(OpWithParent(op, parentOp));

        }



        // Check by operation name for routing dialect ops

        if (op->getName().getStringRef() == "routing.partitiontensor") {

            info.partitionTensorOps.push_back(OpWithParent(op, parentOp));



            // Extract partition parameters - NO deduplication here!

            // Each execute_region has its own partitiontensor representing different data flows

            // (e.g., producer flow vs consumer flow may have same partition params but different purposes)

            PartitionParams params;

            params.index = info.uniquePartitionParams.size();  // Unique index for each partition

            if (op->getNumResults() > 0) {

                params.tensorType = cast<RankedTensorType>(op->getResult(0).getType());

            }

            if (auto attr = op->getAttrOfType<IntegerAttr>("splitnum")) {

                params.splitnum = attr.getInt();

            }

            if (auto attr = op->getAttrOfType<IntegerAttr>("splitdim")) {

                params.splitdim = attr.getInt();

            }

            if (auto attr = op->getAttrOfType<StringAttr>("hw_axis_owner")) {

                params.hw_axis_owner = attr.getValue().str();

            }

            if (auto attr = op->getAttrOfType<StringAttr>("replicate_on")) {

                params.replicate_on = attr.getValue().str();

            }

            if (auto attr = op->getAttrOfType<StringAttr>("single_tile_owner")) {

                params.single_tile_owner = attr.getValue().str();

            }



            // Populate initTensorMap from the partitiontensor's source operand.

            // In new IR (after DeclareDataOpConversion), there is no declare_data wrapper —

            // the arith.constant tensor feeds partitiontensor directly.

            if (op->getNumOperands() > 0 && op->getNumResults() > 0) {

                auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());

                std::string typeKey;

                llvm::raw_string_ostream os(typeKey);

                os << tensorType;

                if (info.initTensorMap.find(typeKey) == info.initTensorMap.end()) {

                    info.initTensorMap[typeKey] = op->getOperand(0);

                }

                // Also track the tensor type in sourceTensorTypes

                bool found = false;

                for (auto &existingType : info.sourceTensorTypes) {

                    if (existingType == tensorType) {

                        found = true;

                        break;

                    }

                }

                if (!found) {

                    info.sourceTensorTypes.push_back(tensorType);

                }

            }



            // Always add - each partitiontensor represents a unique data flow

            info.uniquePartitionParams.push_back(params);

        } else if (op->getName().getStringRef().starts_with("routing.RoutingCreate")) {

            info.routingCreateOps.push_back(OpWithParent(op, parentOp));

        }

        

        // Collect dfscheblueprint.declare_data

        if (op->getName().getStringRef() == "dfscheblueprint.declare_data") {

            info.declareDataOps.push_back(OpWithParent(op, parentOp));

            // Extract tensor type and init tensor for source tensor creation

            if (op->getNumResults() > 0) {

                if (auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {

                    // Create a key for this tensor type

                    std::string typeKey;

                    llvm::raw_string_ostream os(typeKey);

                    os << tensorType;

                    

                    bool found = false;

                    for (auto &existingType : info.sourceTensorTypes) {

                        if (existingType == tensorType) {

                            found = true;

                            break;

                        }

                    }

                    if (!found) {

                        info.sourceTensorTypes.push_back(tensorType);

                    }

                    

                    // Capture the init tensor from the operand (new DeclareDataOp takes init_tensor as input)

                    if (op->getNumOperands() > 0 && info.initTensorMap.find(typeKey) == info.initTensorMap.end()) {

                        info.initTensorMap[typeKey] = op->getOperand(0);

                    }

                }

            }

        }

        

        // Collect top-level arith.constant in main function

        if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {

            // Only collect if direct child of func.func (not nested in regions)

            if (auto funcOp = dyn_cast<func::FuncOp>(op->getParentOp())) {

                if (funcOp.getName() == "main") {

                    info.topLevelConstantOps.push_back(OpWithParent(op, parentOp));

                }

            }

        }

    });

}



// Debug: Print parent information for all collected operations

static void printParentInfo(const ModuleScheduleInfo &info) {

    llvm::errs() << "\n=== Operation Parent Information ===\n";

    

    auto printOpList = [](const SmallVector<OpWithParent> &ops, StringRef name) {

        if (ops.empty()) return;

        llvm::errs() << "\n" << name << " (" << ops.size() << " operations):\n";

        for (size_t i = 0; i < ops.size(); ++i) {

            llvm::errs() << "  [" << i << "] " << ops[i].op->getName() << "\n";

            if (ops[i].parent) {

                llvm::errs() << "      Parent: " << ops[i].parent->getName();

                if (ops[i].isInDSKernelReceiver) {

                    llvm::errs() << " (IS dfschedule.dskernel_receiver)";

                } else {

                    llvm::errs() << " (NOT dfschedule.dskernel_receiver)";

                }

                llvm::errs() << "\n";

            } else {

                llvm::errs() << "      Parent: <none> (top-level)\n";

            }

        }

    };

    

    printOpList(info.declareTensorOps, "DeclareTensor");

    printOpList(info.declareTileOps, "DeclareTile");

    printOpList(info.configDmaBdOps, "ConfigDmaBd");

    printOpList(info.configCreateIoOps, "ConfigCreateIo");

    printOpList(info.packetOps, "Packet");

    printOpList(info.loadKernelGroupOps, "LoadKernelGroup");

    printOpList(info.launchKernelGroupOps, "LaunchKernelGroup");

    printOpList(info.getBdIdOps, "GetBdId");

    printOpList(info.startIoOps, "StartIo");

    printOpList(info.scheduleWaitOps, "ScheduleWait");

    printOpList(info.dskernelReceiverOps, "DSKernelReceiver");

    printOpList(info.tensorEmptyOps, "TensorEmpty");

    printOpList(info.extractSliceOps, "ExtractSlice");

    printOpList(info.partitionTensorOps, "PartitionTensor");

    printOpList(info.executeRegionOps, "ExecuteRegion");

    printOpList(info.routingCreateOps, "RoutingCreate");

    printOpList(info.declareDataOps, "DeclareData");

    printOpList(info.topLevelConstantOps, "TopLevelConstant");

    

    llvm::errs() << "\n====================================\n\n";

}



// Associate packets with their target tiles

static void associatePacketsWithTiles(ModuleScheduleInfo &info) {

    // For each LoadKernelGroupOp, extract tile-to-packet/config mapping

    for (auto &opWithParent : info.loadKernelGroupOps) {

        auto loadKernel = cast<dfschedule::LoadKernelGroupOp>(opWithParent.op);

        

        auto tiles = loadKernel.getTiles();

        auto computeArgs = loadKernel.getDistributedComputeKernelArgs();

        

        // Check for distributed_args (which now contains kernel config symbols)

        auto distArgs = loadKernel.getDistributedArgs();

        

        if (distArgs) {

            // Handle distributed_args pointing to individual kernel config symbols

            auto distArgsArray = *distArgs;  // Dereference the optional

            auto moduleOp = opWithParent.parent->getParentOfType<ModuleOp>();

            

            for (size_t i = 0; i < tiles.size(); ++i) {

                Value tileVal = tiles[i];

                if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

                    TileKey key = getTileKey(declareTile);

                    

                    if (!isShimTile(key)) {

                        auto &tileInfo = info.coreTiles[key];

                        

                        // Look up the kernel config symbol for this tile

                        if (i < distArgsArray.size()) {

                            if (auto symRef = dyn_cast<SymbolRefAttr>(distArgsArray[i])) {

                                // Lookup the DeclareKernelConfigOp

                                auto configOp = moduleOp.lookupSymbol<dfschedule::DeclareKernelConfigOp>(

                                    symRef.getRootReference());

                                

                                if (configOp) {

                                    auto tileConfigsAttr = configOp.getTileConfigs();

                                    // Each config op should have exactly one tile config

                                    if (tileConfigsAttr.size() > 0) {

                                        auto configDict = dyn_cast<DictionaryAttr>(tileConfigsAttr[0]);

                                        if (configDict) {

                                            tileInfo.configDicts.push_back(configDict);

                                        }

                                    }

                                }

                                

                                // Also store the symbol reference for backward compatibility

                                tileInfo.packetSymbols.push_back(symRef);

                            }

                        }

                        

                        // Add compute kernel arg

                        if (i < computeArgs.size()) {

                            if (auto symRef = dyn_cast<SymbolRefAttr>(computeArgs[i])) {

                                tileInfo.computeKernelArgs.push_back(symRef);

                            }

                        }

                        

                        // Track packet count per tile

                        info.tilePacketCount[key]++;

                    }

                }

            }

        }

    }

}



// Associate DMA configs with shim tiles

static void associateDmaWithShimTiles(ModuleScheduleInfo &info) {

    for (auto &opWithParent : info.configDmaBdOps) {

        auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(opWithParent.op);

        Value tileVal = dmaBd.getTile();

        

        // Skip DMA BD operations inside dskernel_receiver (these are kernel-side, not host-side shim operations)

        if (opWithParent.isInDSKernelReceiver) {

            continue;

        }

        

        // Only include operations that are in dfschedule.host (the canonicalized region)

        if (!opWithParent.parent || opWithParent.parent->getName().getStringRef() != "dfschedule.host") {

            continue;

        }

        

        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            

            if (isShimTile(key)) {

                info.shimTiles[key].dmaBdOps.push_back(opWithParent.op);

                info.shimTiles[key].bdHandles.push_back(dmaBd.getBdHandle());

            }

        }

    }

    ///*

    for (auto &opWithParent : info.configCreateIoOps) {

        auto createIo = cast<dfschedule::ConfigCreateIoOp>(opWithParent.op);

        Value tileVal = createIo.getTile();

        

        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (isShimTile(key)) {

                info.shimTiles[key].createIoOps.push_back(opWithParent.op);

                info.shimTiles[key].ioHandles.push_back(createIo.getIoHandle());

            }

        }

    }

    //*/

    // Print all shim tiles and their associated DMA operations

    llvm::errs() << "\n=== Shim Tiles DMA Association ===\n";

    if (info.shimTiles.empty()) {

        llvm::errs() << "  No shim tiles found.\n";

    } else {

        for (auto &[key, shimInfo] : info.shimTiles) {

            llvm::errs() << "\nShim Tile (col=" << key.first << ", row=" << key.second << "):\n";

            

            // Print DMA BD operations

            llvm::errs() << "  DMA BD Operations: " << shimInfo.dmaBdOps.size() << "\n";

            for (size_t i = 0; i < shimInfo.dmaBdOps.size(); ++i) {

                auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(shimInfo.dmaBdOps[i]);

                llvm::errs() << "    [" << i << "] dfschedule.config.dma_bd\n";

                llvm::errs() << "        Offset: " << dmaBd.getOffset() << "\n";

                llvm::errs() << "        Length: " << dmaBd.getLen() << "\n";

                llvm::errs() << "        Enable Packet: " << dmaBd.getEnablePacket() << "\n";

                if (dmaBd.getEnablePacket()) {

                    llvm::errs() << "        Packet ID: " << dmaBd.getPacketId() << "\n";

                }

                llvm::errs() << "        BD Handle: " << shimInfo.bdHandles[i] << "\n";

            }

            

            // Print IO Config operations

            llvm::errs() << "  IO Config Operations: " << shimInfo.createIoOps.size() << "\n";

            for (size_t i = 0; i < shimInfo.createIoOps.size(); ++i) {

                auto createIo = cast<dfschedule::ConfigCreateIoOp>(shimInfo.createIoOps[i]);

                llvm::errs() << "    [" << i << "] dfschedule.config.create_io\n";

                llvm::errs() << "        Channel: " << createIo.getChannel() << "\n";

                llvm::errs() << "        Direction: " << createIo.getDirection() << "\n";

                llvm::errs() << "        IO Operation: " << createIo.getIoOperation() << "\n";

                llvm::errs() << "        IO Handle: " << shimInfo.ioHandles[i] << "\n";

            }

        }

    }

    llvm::errs() << "\n==================================\n\n";

}



// Structure to hold DMA BD parameters extracted from original ops

struct DmaBdParams {

    TileKey shimKey;

    Type bufferType;

    int64_t offset;

    int64_t len;

    bool enablePacket;  // Changed to bool for BoolAttr

    int64_t packetId;

    int64_t nextBd;

    int bdIndex;

    int64_t sliceIndex;  // Index into uniqueSliceParams/declaredMemrefs for the buffer

    int64_t acquireLockId = -1;

    int64_t acquireLockVal = -1;

    int64_t releaseLockId = -1;

    int64_t releaseLockVal = -1;

    int32_t data_id = -1; // Root-tensor identity; set by BlueprintToSchedulePass

    bool isBufferView = false;         // True when buffer came from BufferViewOp (new IR path)

    int64_t shimSourceSliceIndex = -1; // Index into sliceMap for the intermediate tensor slice

};



// Structure to hold IO config parameters

struct IoConfigParams {

    TileKey shimKey;

    int64_t channel;

    std::string direction;    // StringAttr

    std::string ioOperation;  // StringAttr

    int bdIndex; // Which BD handle to use

};



// Structure to hold core tile DMA BD parameters (with lock IDs for ping-pong)

struct CoreDmaBdParams {

    TileKey coreKey;

    Type bufferType;

    int64_t offset;

    int64_t len;

    bool enablePacket;

    int64_t packetId;

    int64_t nextBd;

    int bdIndex;

    int64_t sliceIndex;

    int64_t acquireLockId;

    int64_t acquireLockVal;

    int64_t releaseLockId;

    int64_t releaseLockVal;

    int64_t originalBdId = -1;

    bool hasLinkedBd = false;

    bool isBindCoreBuffer = false; // True when buffer came from BindCoreBufferOp (sliceIndex = L1 byte offset)

};



// Structure to hold core tile IO config parameters

struct CoreIoConfigParams {

    TileKey coreKey;

    int64_t channel;

    std::string direction;

    std::string ioOperation;

    int64_t originalBdId = 0;

};



// Create canonicalized schedule in dfschedule.host at module level

// All operations use only constants/attributes, so IsolatedFromAbove is OK

static void createCanonicalizedSchedule(

    OpBuilder &builder,

    Location loc,

    ModuleScheduleInfo &info,

    ModuleOp moduleOp,

    func::FuncOp funcOp) {

    

    // Early exit if nothing to canonicalize

    if (info.coreTiles.empty() && info.shimTiles.empty()) {

        return;

    }

    

    // ==========================================================

    // Collect DMA BD and IO parameters from original operations

    // ==========================================================

    std::vector<DmaBdParams> allDmaBdParams;

    std::vector<IoConfigParams> allIoConfigParams;

    

    // Map to track BD index per shim tile

    std::map<TileKey, int> shimBdCounter;

    

    for (auto &opWithParent : info.configDmaBdOps) {

        auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(opWithParent.op);

        Value tileVal = dmaBd.getTile();

        

        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (isShimTile(key)) {

                DmaBdParams params;

                params.shimKey = key;

                params.bufferType = dmaBd.getBuffer().getType();

                params.offset = dmaBd.getOffset();

                params.len = dmaBd.getLen();

                params.enablePacket = dmaBd.getEnablePacket();

                params.packetId = dmaBd.getPacketId();

                params.nextBd = dmaBd.getNextBd();

                params.bdIndex = shimBdCounter[key]++;

                params.sliceIndex = -1;  // Default: no slice found

                params.acquireLockId = static_cast<int32_t>(dmaBd.getAcquireLockId());

                params.acquireLockVal = static_cast<int32_t>(dmaBd.getAcquireLockVal());

                params.releaseLockId = static_cast<int32_t>(dmaBd.getReleaseLockId());

                params.releaseLockVal = static_cast<int32_t>(dmaBd.getReleaseLockVal());

                // Read data_id propagated from DmaphopTodfscheblueprintPass via typed accessor

                params.data_id = static_cast<int32_t>(dmaBd.getDataId());



                // Fast-path: buffer_view carries offset directly — no slice chain tracing needed

                Value buffer = dmaBd.getBuffer();

                if (auto bvOp = buffer.getDefiningOp<dfschedule::BufferViewOp>()) {

                    // Encode buffer_view offset as sliceIndex for dedup (guaranteed unique per view)

                    params.sliceIndex = static_cast<int64_t>(bvOp.getOffset());

                    params.isBufferView = true;



                    // Trace back through alloc_device_mem -> unrealized_cast -> extract_slice

                    // to find the source tensor slice index for reconstruction

                    Value bvBase = bvOp.getBase();

                    if (auto allocDDR = bvBase.getDefiningOp<dfschedule::AllocDeviceMemOp>()) {

                        Value allocSrc = allocDDR.getSource();

                        // Unwrap memref_mapping if present (BlueprintToSchedulePass inserts it

                        // between unrealized_cast and alloc_device_mem)

                        if (auto mappingOp = allocSrc.getDefiningOp<dfschedule::MemRefMappingOp>()) {

                            allocSrc = mappingOp.getOperand();

                        }

                        // May be unrealized_conversion_cast tensor -> memref

                        if (auto castOp = allocSrc.getDefiningOp<UnrealizedConversionCastOp>()) {

                            if (!castOp.getInputs().empty()) {

                                Value castInput = castOp.getInputs()[0];

                                if (auto sliceOp = castInput.getDefiningOp<tensor::ExtractSliceOp>()) {

                                    for (size_t i = 0; i < info.extractSliceOps.size(); ++i) {

                                        if (info.extractSliceOps[i].op == sliceOp.getOperation()) {

                                            params.shimSourceSliceIndex = static_cast<int64_t>(i);

                                            break;

                                        }

                                    }

                                }

                            }

                        }

                    }



                    allDmaBdParams.push_back(params);

                    continue;

                }



                // Legacy path removed: DeclareTensorOp deprecated (tensor path)



                allDmaBdParams.push_back(params);

            }

        }

    }

    

    // Map to track IO index per shim tile

    std::map<TileKey, int> shimIoCounter;

    

    for (auto &opWithParent : info.configCreateIoOps) {

        auto createIo = cast<dfschedule::ConfigCreateIoOp>(opWithParent.op);

        Value tileVal = createIo.getTile();

        

        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (isShimTile(key)) {

                IoConfigParams params;

                params.shimKey = key;

                params.channel = createIo.getChannel();

                params.direction = createIo.getDirection().str();

                params.ioOperation = createIo.getIoOperation().str();

                params.bdIndex = shimIoCounter[key]++;

                allIoConfigParams.push_back(params);

            }

        }

    }



    // ==========================================================

    // Deduplicate shim DMA BDs by (shimKey, data_id, sliceIndex).

    // BDs for the same root tensor AND same slice share a data_id and sliceIndex;

    // BDs for different slices (e.g. different partition rounds) must be kept

    // separate even if they share data_id, since they transfer different data.

    // Also deduplicate IO configs by (shimKey, channel, direction) for the same reason.

    // ==========================================================

    {

        using ShimMergeKey = std::tuple<TileKey, int32_t, int64_t>;

        std::set<ShimMergeKey> seenShimBdGroups;

        std::vector<DmaBdParams> deduped;

        for (auto &p : allDmaBdParams) {

            ShimMergeKey k = {p.shimKey, p.data_id, p.sliceIndex};

            if (seenShimBdGroups.find(k) == seenShimBdGroups.end()) {

                seenShimBdGroups.insert(k);

                deduped.push_back(p);

            }

        }

        allDmaBdParams = std::move(deduped);

    }

    {

        using IoMergeKey = std::tuple<TileKey, int64_t, std::string>;

        struct IoMergeKeyLess {

            bool operator()(const IoMergeKey &a, const IoMergeKey &b) const {

                if (std::get<0>(a) != std::get<0>(b))

                    return std::get<0>(a) < std::get<0>(b);

                if (std::get<1>(a) != std::get<1>(b))

                    return std::get<1>(a) < std::get<1>(b);

                return std::get<2>(a) < std::get<2>(b);

            }

        };

        std::set<IoMergeKey, IoMergeKeyLess> seenIoGroups;

        std::vector<IoConfigParams> deduped;

        for (auto &p : allIoConfigParams) {

            IoMergeKey k = {p.shimKey, p.channel, p.direction};

            if (seenIoGroups.find(k) == seenIoGroups.end()) {

                seenIoGroups.insert(k);

                deduped.push_back(p);

            }

        }

        allIoConfigParams = std::move(deduped);

    }



    // ==========================================================

    // Collect core tile DMA BD and IO parameters

    // ==========================================================

    std::vector<CoreDmaBdParams> allCoreDmaBdParams;

    std::vector<CoreIoConfigParams> allCoreIoConfigParams;



    std::map<TileKey, int> coreBdCounter;



    for (auto &opWithParent : info.configDmaBdOps) {

        auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(opWithParent.op);

        Value tileVal = dmaBd.getTile();



        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (!isShimTile(key)) {

                CoreDmaBdParams params;

                params.coreKey = key;

                params.bufferType = dmaBd.getBuffer().getType();

                params.offset = dmaBd.getOffset();

                params.len = dmaBd.getLen();

                params.enablePacket = dmaBd.getEnablePacket();

                params.packetId = dmaBd.getPacketId();

                params.nextBd = dmaBd.getNextBd();

                params.sliceIndex = -1;

                params.hasLinkedBd = (dmaBd.getLinkedBd() != Value());



                Value bdIdValue = dmaBd.getBdId();

                if (auto constOp = bdIdValue.getDefiningOp<arith::ConstantOp>()) {

                    params.originalBdId = cast<IntegerAttr>(constOp.getValue()).getInt();

                } else {

                    params.originalBdId = coreBdCounter[key];

                }

                params.bdIndex = params.originalBdId;

                coreBdCounter[key]++;



                // Read lock IDs and values directly from attributes

                params.acquireLockId = static_cast<int32_t>(dmaBd.getAcquireLockId());

                params.acquireLockVal = static_cast<int32_t>(dmaBd.getAcquireLockVal());

                params.releaseLockId = static_cast<int32_t>(dmaBd.getReleaseLockId());

                params.releaseLockVal = static_cast<int32_t>(dmaBd.getReleaseLockVal());



                // Fast-path: bind_core_buffer carries L1 offset directly

                Value buffer = dmaBd.getBuffer();

                if (auto bcbOp = buffer.getDefiningOp<dfschedule::BindCoreBufferOp>()) {

                    // Encode L1 offset as sliceIndex for buffer resolution in createCoreBd

                    params.sliceIndex = static_cast<int64_t>(bcbOp.getOffset());

                    params.isBindCoreBuffer = true;

                    allCoreDmaBdParams.push_back(params);

                    continue;

                }



                // Legacy path removed: DeclareTensorOp deprecated (tensor path)



                allCoreDmaBdParams.push_back(params);

            }

        }

    }



    for (auto &opWithParent : info.configCreateIoOps) {

        auto createIo = cast<dfschedule::ConfigCreateIoOp>(opWithParent.op);

        Value tileVal = createIo.getTile();



        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (!isShimTile(key)) {

                CoreIoConfigParams params;

                params.coreKey = key;

                params.channel = createIo.getChannel();

                params.direction = createIo.getDirection().str();

                params.ioOperation = createIo.getIoOperation().str();



                // Trace the bd_config SSA operand to find original BD ID

                Value bdConfig = createIo.getBdConfig();

                if (auto dmaBd = bdConfig.getDefiningOp<dfschedule::ConfigDmaBdOp>()) {

                    Value bdIdVal = dmaBd.getBdId();

                    if (auto constOp = bdIdVal.getDefiningOp<arith::ConstantOp>()) {

                        params.originalBdId = cast<IntegerAttr>(constOp.getValue()).getInt();

                    }

                }



                allCoreIoConfigParams.push_back(params);

            }

        }

    }



    // ==========================================================

    // PART 1: Create dfschedule.host at MODULE level (after func.func)

    // ==========================================================

    builder.setInsertionPointAfter(funcOp);

    

    auto hostOp = builder.create<dfschedule::HostBlockOp>(

        loc,

        builder.getStringAttr("host_canonicalized"));

    

    Block *hostBody = &hostOp.getBody().emplaceBlock();

    builder.setInsertionPointToStart(hostBody);

    

    // All operations below are created INSIDE dfschedule.host

    // They only use constants and values defined within this block

    

    // Helper to create type key strings

    auto makeTensorTypeKey = [](RankedTensorType t) -> std::string {

        std::string key;

        llvm::raw_string_ostream os(key);

        os << t;

        return key;

    };

    

    // Helper to create partition key - include index for uniqueness

    auto makePartitionKey = [&](const PartitionParams &p) -> std::string {

        std::string key;

        llvm::raw_string_ostream os(key);

        os << p.index << "_" << p.tensorType << "_" << p.splitnum << "_" << p.splitdim << "_" << p.hw_axis_owner;

        return key;

    };



    // ==========================================================

    // DATA FLOW: (declare_data ->) partitiontensor -> extract_slice

    // In new IR: arith.constant -> partitiontensor -> extract_slice (no declare_data)

    // ==========================================================



    // 0a. Populate sourceTensorMap by cloning each unique source tensor's init op.

    // The init tensor is captured from routing.partitiontensor operands (new IR path)

    // or from dfscheblueprint.declare_data operands (old IR path) — both stored in initTensorMap.

    std::map<std::string, Value> sourceTensorMap;

    for (auto tensorType : info.sourceTensorTypes) {

        std::string key = makeTensorTypeKey(tensorType);

        if (sourceTensorMap.find(key) == sourceTensorMap.end()) {

            // Look up the captured init tensor from the original IR

            auto initTensorIt = info.initTensorMap.find(key);

            if (initTensorIt == info.initTensorMap.end()) {

                llvm::errs() << "Warning: No init tensor found for type " << key << "\n";

                continue;

            }



            Value originalInitTensor = initTensorIt->second;



            // Clone the init tensor (arith.constant) into the new block

            Operation *initDefOp = originalInitTensor.getDefiningOp();

            if (!initDefOp) {

                llvm::errs() << "Warning: Init tensor is not defined by an operation\n";

                continue;

            }

            Operation *clonedOp = builder.clone(*initDefOp);

            // Store the cloned constant directly — routing.partitiontensor will use it.

            // (No dfscheblueprint.declare_data wrapper needed in new IR.)

            sourceTensorMap[key] = clonedOp->getResult(0);

        }

    }

    

    // 0b. Create routing.partitiontensor for each unique partition config

    std::map<std::string, Value> partitionedTensorMap;

    for (const auto &params : info.uniquePartitionParams) {

        std::string partKey = makePartitionKey(params);

        if (partitionedTensorMap.find(partKey) == partitionedTensorMap.end()) {

            // Find the source tensor

            std::string srcKey = makeTensorTypeKey(params.tensorType);

            Value sourceTensor;

            if (sourceTensorMap.find(srcKey) != sourceTensorMap.end()) {

                sourceTensor = sourceTensorMap[srcKey];

            } else {

                // Fallback: clone the init tensor directly (no declare_data wrapper)

                auto initTensorIt = info.initTensorMap.find(srcKey);

                if (initTensorIt != info.initTensorMap.end()) {

                    Value originalInitTensor = initTensorIt->second;

                    Operation *initDefOp = originalInitTensor.getDefiningOp();

                    if (initDefOp) {

                        Operation *clonedOp = builder.clone(*initDefOp);

                        sourceTensorMap[srcKey] = clonedOp->getResult(0);

                        sourceTensor = clonedOp->getResult(0);

                    }

                }

            }



            if (!sourceTensor) {

                llvm::errs() << "Warning: No source tensor for partitiontensor — skipping\n";

                continue;

            }



            // Create routing.partitiontensor

            OperationState partState(loc, "routing.partitiontensor");

            partState.addOperands({sourceTensor});

            partState.addTypes({params.tensorType});

            partState.addAttribute("splitnum", builder.getI32IntegerAttr(params.splitnum));

            partState.addAttribute("splitdim", builder.getI32IntegerAttr(params.splitdim));

            partState.addAttribute("hw_axis_owner", builder.getStringAttr(params.hw_axis_owner));

            partState.addAttribute("replicate_on", builder.getStringAttr(params.replicate_on));

            partState.addAttribute("single_tile_owner", builder.getStringAttr(params.single_tile_owner));

            Operation *partitionOp = builder.create(partState);

            partitionedTensorMap[partKey] = partitionOp->getResult(0);

        }

    }

    

    // 0c. Create extract_slice operations - maintain proper chain

    // Each slice gets a unique entry, keyed by its index

    // We process in order: first slices from partitions, then nested slices

    

    std::map<size_t, Value> sliceMap;  // Map from slice index to created Value

    

    // First pass: create slices that come directly from partitiontensors

    for (const auto &params : info.uniqueSliceParams) {

        if (params.isFromPartition) {

            Value sourceTensor;

            

            // Find the partitioned tensor by index

            for (auto &[partKey, partValue] : partitionedTensorMap) {

                std::string indexPrefix = std::to_string(params.partitionIndex) + "_";

                if (partKey.find(indexPrefix) == 0) {

                    sourceTensor = partValue;

                    break;

                }

            }

            

            if (sourceTensor) {

                SmallVector<int64_t, 4> defaultStrides(params.offsets.size(), 1);

                auto newSlice = builder.create<tensor::ExtractSliceOp>(

                    loc,

                    params.resultType,

                    sourceTensor,

                    ValueRange{}, ValueRange{}, ValueRange{},

                    params.offsets,

                    params.sizes,

                    params.strides.empty() ? defaultStrides : params.strides);

                sliceMap[params.index] = newSlice.getResult();

            }

        }

    }

    

    // Second pass: create nested slices (from other slices)

    // May need multiple passes if there are deeply nested slices

    bool progress = true;

    while (progress) {

        progress = false;

        for (const auto &params : info.uniqueSliceParams) {

            // Skip if already created or if it's directly from partition

            if (sliceMap.find(params.index) != sliceMap.end() || params.isFromPartition) {

                continue;

            }

            

            // Check if parent slice is ready

            if (params.parentSliceIndex >= 0) {

                auto parentIt = sliceMap.find(static_cast<size_t>(params.parentSliceIndex));

                if (parentIt != sliceMap.end()) {

                    // Parent is ready, create this slice

                    Value sourceTensor = parentIt->second;

                    SmallVector<int64_t, 4> defaultStrides(params.offsets.size(), 1);

                    auto newSlice = builder.create<tensor::ExtractSliceOp>(

                        loc,

                        params.resultType,

                        sourceTensor,

                        ValueRange{}, ValueRange{}, ValueRange{},

                        params.offsets,

                        params.sizes,

                        params.strides.empty() ? defaultStrides : params.strides);

                    sliceMap[params.index] = newSlice.getResult();

                    progress = true;

                }

            }

        }

    }

    

    

    // Identify leaf slices (slices with no children)

    std::set<size_t> slicesWithChildren;

    for (const auto &params : info.uniqueSliceParams) {

        if (params.parentSliceIndex >= 0) {

            slicesWithChildren.insert(static_cast<size_t>(params.parentSliceIndex));

        }

    }



    // Collect which intermediate slice indices are actually referenced by Shim BDs

    // so we only create declaretensor ops for those that are needed.

    std::set<size_t> shimReferencedIntermediates;

    for (const auto &p : allDmaBdParams) {

        if (p.sliceIndex >= 0) {

            size_t idx = static_cast<size_t>(p.sliceIndex);

            if (slicesWithChildren.count(idx) > 0)

                shimReferencedIntermediates.insert(idx);

        }

    }



    // 0d. Create dfschedule.declaretensor for leaf slices (core tile BDs)

    //     and for intermediate slices actually referenced by Shim BDs.

    std::map<size_t, Value> declaredMemrefs;     // leaf slice index -> memref

    std::map<size_t, Value> intermediateMemrefs; // intermediate slice index -> memref

    for (const auto &params : info.uniqueSliceParams) {

        auto it = sliceMap.find(params.index);

        if (it == sliceMap.end())

            continue;



        bool isIntermediate = slicesWithChildren.count(params.index) > 0;

        if (isIntermediate && shimReferencedIntermediates.count(params.index) == 0)

            continue; // intermediate slice not used by any Shim BD -- skip



        Value sliceValue = it->second;

        auto sliceType = cast<RankedTensorType>(sliceValue.getType());

        auto memrefType = MemRefType::get(sliceType.getShape(), sliceType.getElementType());



        // DeclareTensorOp removed: use UnrealizedConversionCastOp (tensor path deprecated)

        auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(loc, mlir::TypeRange{memrefType},

                                                                       mlir::ValueRange{sliceValue});

        mlir::Value memrefValue = castOp.getOutputs()[0];



        if (isIntermediate)

            intermediateMemrefs[params.index] = memrefValue;

        else

            declaredMemrefs[params.index] = memrefValue;

    }



    // 1. Create NEW deduplicated shim tile declarations

    std::map<TileKey, Value> shimTileMap;

    for (auto &[key, shimInfo] : info.shimTiles) {

        auto shimTile = builder.create<dfschedule::DeclareTileOp>(

            loc,

            dfschedule::TileType::get(builder.getContext()),

            builder.getI32IntegerAttr(key.first),

            builder.getI32IntegerAttr(key.second));

        shimTileMap[key] = shimTile.getTile();

    }

    

    // 2. Create NEW deduplicated core tile declarations

    std::map<TileKey, Value> coreTileMap;

    for (auto &[key, tileInfo] : info.coreTiles) {

        auto coreTile = builder.create<dfschedule::DeclareTileOp>(

            loc,

            dfschedule::TileType::get(builder.getContext()),

            builder.getI32IntegerAttr(key.first),

            builder.getI32IntegerAttr(key.second));

        coreTileMap[key] = coreTile.getTile();

    }

    

    // 3. Create external memory references for DMA buffers (using memref.alloc)

    // Group buffers by type to deduplicate

    std::map<std::string, Value> bufferMap;

    auto makeBufferKey = [](Type t) -> std::string {

        std::string key;

        llvm::raw_string_ostream os(key);

        os << t;

        return key;

    };

    

    // Map to store BD handles per shim tile

    std::map<TileKey, SmallVector<Value>> shimBdHandles;



    // 4. Create DMA BD configurations for shim tiles

    // Shim BDs keep the intermediate-slice buffer (the Shim receives the combined

    // data from DDR; packet routing distributes it to individual AIE core tiles).

    int bdIndexCounter = 0;

    for (const auto &params : allDmaBdParams) {

        if (shimTileMap.find(params.shimKey) == shimTileMap.end()) continue;

        Value shimTile = shimTileMap[params.shimKey];



        // Helper lambda to create one BD with a given buffer value

        auto createOneShimBd = [&](Value buffer) {

            auto bdIdConst = builder.create<arith::ConstantOp>(loc, builder.getI32Type(),

                                                               builder.getI32IntegerAttr(bdIndexCounter++));

            auto dmaBdOp = builder.create<dfschedule::ConfigDmaBdOp>(

                loc, dfschedule::BdHandleType::get(builder.getContext()), buffer, shimTile, bdIdConst.getResult(),

                builder.getI32IntegerAttr(params.offset), builder.getI32IntegerAttr(params.len),

                builder.getBoolAttr(params.enablePacket), builder.getI32IntegerAttr(params.packetId),

                builder.getI32IntegerAttr(params.nextBd), builder.getI32IntegerAttr(params.acquireLockId),

                builder.getI32IntegerAttr(params.acquireLockVal), builder.getI32IntegerAttr(params.releaseLockId),

                builder.getI32IntegerAttr(params.releaseLockVal),

                builder.getI32IntegerAttr(params.data_id), // propagate data_id

                Value());                                  // linked_bd (none for shim)

            shimBdHandles[params.shimKey].push_back(dmaBdOp.getBdHandle());

        };



        // Fast-path: isBufferView means buffer came from BufferViewOp (new IR path)

        // Reconstruct: unrealized_cast(tensor_slice) -> alloc_device_mem -> buffer_view

        if (params.isBufferView) {

            if (auto mrType = dyn_cast<MemRefType>(params.bufferType)) {

                // Find the intermediate tensor slice that feeds this DDR buffer

                Value srcTensor;

                if (params.shimSourceSliceIndex >= 0) {

                    auto it = sliceMap.find(static_cast<size_t>(params.shimSourceSliceIndex));

                    if (it != sliceMap.end()) {

                        srcTensor = it->second;

                    }

                }



                Value srcForAlloc;

                if (srcTensor) {

                    // Bridge: tensor -> flat memref (same shape, no memspace)

                    auto tensorType = cast<RankedTensorType>(srcTensor.getType());

                    auto flatMemrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

                    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, TypeRange{flatMemrefType},

                                                                             ValueRange{srcTensor});

                    srcForAlloc = castOp.getResult(0);

                } else {

                    // Fallback if we couldn't trace the source slice: use memref.alloc

                    // (DfscheduleToApiPass will emit a broken memcpy — better than crashing)

                    llvm::errs() << "Warning: shimSourceSliceIndex not found for BufferView BD; "

                                    "using alloc placeholder\n";

                    auto srcType = MemRefType::get(mrType.getShape(), mrType.getElementType());

                    srcForAlloc = builder.create<memref::AllocOp>(loc, srcType);

                }



                auto allocDDR = builder.create<dfschedule::AllocDeviceMemOp>(loc, mrType, srcForAlloc);

                Value ddrBase = allocDDR.getMemref();

                auto bvOp = builder.create<dfschedule::BufferViewOp>(

                    loc, mrType, ddrBase,

                    builder.getI64IntegerAttr(params.sliceIndex), // offset (buffer_view offset)

                    builder.getI64IntegerAttr(params.len));       // len

                createOneShimBd(bvOp.getView());

                continue;

            }

        }



        if (params.sliceIndex >= 0) {

            size_t idx = static_cast<size_t>(params.sliceIndex);



            // If this slice is an intermediate slice (has children), the Shim BD

            // should use the intermediate-slice buffer directly -- do NOT expand

            // to leaf descendants.

            auto intIt = intermediateMemrefs.find(idx);

            if (intIt != intermediateMemrefs.end()) {

                createOneShimBd(intIt->second);

                continue;

            }



            // Otherwise it is a leaf slice -- use the leaf memref.

            auto leafIt = declaredMemrefs.find(idx);

            if (leafIt != declaredMemrefs.end()) {

                createOneShimBd(leafIt->second);

                continue;

            }

        }



        // Fallback: no slice chain (sliceIndex == -1) or lookup failed.

        // Create exactly one BD using a fallback buffer (memref.alloc of full tensor).

        std::string bufKey = makeBufferKey(params.bufferType);

        Value buffer;

        if (bufferMap.find(bufKey) == bufferMap.end()) {

            auto memrefType = cast<MemRefType>(params.bufferType);

            buffer = builder.create<memref::AllocOp>(loc, memrefType);

            bufferMap[bufKey] = buffer;

        } else {

            buffer = bufferMap[bufKey];

        }

        createOneShimBd(buffer);

    }

    

    // 5. Create IO configurations for shim tiles

    std::map<TileKey, SmallVector<Value>> shimIoHandles;

    

    for (const auto &params : allIoConfigParams) {

        if (shimTileMap.find(params.shimKey) == shimTileMap.end()) continue;

        Value shimTile = shimTileMap[params.shimKey];

        

        // Get corresponding BD handle

        Value bdHandle;

        if (params.bdIndex < (int)shimBdHandles[params.shimKey].size()) {

            bdHandle = shimBdHandles[params.shimKey][params.bdIndex];

        } else if (!shimBdHandles[params.shimKey].empty()) {

            bdHandle = shimBdHandles[params.shimKey].back();

        } else {

            continue; // No BD handle available, skip this IO

        }

        

        // Create IO config

        auto createIoOp = builder.create<dfschedule::ConfigCreateIoOp>(

            loc,

            dfschedule::IoHandleType::get(builder.getContext()),

            bdHandle,

            shimTile,

            builder.getI32IntegerAttr(params.channel),

            builder.getStringAttr(params.direction),

            builder.getStringAttr(params.ioOperation));

        

        shimIoHandles[params.shimKey].push_back(createIoOp.getIoHandle());

    }



    // 5b. Create DMA BD configurations for core tiles (ping-pong)

    // Map from (tileKey, originalBdId) → new BD handle

    std::map<TileKey, std::map<int64_t, Value>> coreBdHandleMap;



    // Group core BD params by tile for ping-pong pair detection

    std::map<TileKey, SmallVector<size_t>> coreBdParamsByTile;

    for (size_t i = 0; i < allCoreDmaBdParams.size(); ++i) {

        coreBdParamsByTile[allCoreDmaBdParams[i].coreKey].push_back(i);

    }



    // Map from (tileKey, L1Offset) -> BindCoreBufferOp result (to avoid duplicate bind ops per tile/offset)

    std::map<std::pair<TileKey, int64_t>, Value> bindCoreBufferMap;



    auto resolveCoreBdBuffer = [&](const CoreDmaBdParams &params, Value coreTileValue) -> Value {

        Value buffer;



        // Fast-path: buffer came from BindCoreBufferOp; sliceIndex encodes L1 byte offset

        if (params.isBindCoreBuffer) {

            if (auto mrType = dyn_cast<MemRefType>(params.bufferType)) {

                // L1 buffer: recreate BindCoreBufferOp in the canonical host block.

                // Source type strips memspace (BindCoreBufferOp takes a plain memref input).

                auto srcType = MemRefType::get(mrType.getShape(), mrType.getElementType());

                auto key = std::make_pair(params.coreKey, params.sliceIndex);

                auto it = bindCoreBufferMap.find(key);

                if (it != bindCoreBufferMap.end()) {

                    buffer = it->second;

                } else {

                    // Alloc placeholder lowered to real L1 address by DfscheduleToApiPass

                    Value srcAlloc = builder.create<memref::AllocOp>(loc, srcType);

                    auto bindOp = builder.create<dfschedule::BindCoreBufferOp>(

                        loc, mrType, srcAlloc, coreTileValue, builder.getI64IntegerAttr(params.sliceIndex));

                    buffer = bindOp.getBuffer();

                    bindCoreBufferMap[key] = buffer;

                }

                return buffer;

            }

        }



        // Legacy path: sliceIndex is an index into declaredMemrefs

        if (params.sliceIndex >= 0) {

            auto memIt = declaredMemrefs.find(static_cast<size_t>(params.sliceIndex));

            if (memIt != declaredMemrefs.end())

                buffer = memIt->second;

        }

        if (!buffer) {

            std::string bufKey = makeBufferKey(params.bufferType);

            if (bufferMap.find(bufKey) == bufferMap.end()) {

                auto memrefType = cast<MemRefType>(params.bufferType);

                buffer = builder.create<memref::AllocOp>(loc, memrefType);

                bufferMap[bufKey] = buffer;

            } else {

                buffer = bufferMap[bufKey];

            }

        }

        return buffer;

    };



    auto createCoreBd = [&](const CoreDmaBdParams &params, Value coreTile, Value linkedBd) -> Value {

        Value buffer = resolveCoreBdBuffer(params, coreTile);

        auto bdIdConst =

            builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(params.bdIndex));

        auto dmaBdOp = builder.create<dfschedule::ConfigDmaBdOp>(

            loc, dfschedule::BdHandleType::get(builder.getContext()), buffer, coreTile, bdIdConst.getResult(),

            builder.getI32IntegerAttr(params.offset), builder.getI32IntegerAttr(params.len),

            builder.getBoolAttr(params.enablePacket), builder.getI32IntegerAttr(params.packetId),

            builder.getI32IntegerAttr(params.nextBd), builder.getI32IntegerAttr(params.acquireLockId),

            builder.getI32IntegerAttr(params.acquireLockVal), builder.getI32IntegerAttr(params.releaseLockId),

            builder.getI32IntegerAttr(params.releaseLockVal),

            builder.getI32IntegerAttr(-1), // data_id

            linkedBd);

        return dmaBdOp.getBdHandle();

    };



    for (auto &[tileKey, paramIndices] : coreBdParamsByTile) {

        if (coreTileMap.find(tileKey) == coreTileMap.end())

            continue;

        Value coreTile = coreTileMap[tileKey];



        if (paramIndices.size() == 2) {

            // Detect ping/pong by checking hasLinkedBd (SSA-based, not positional)

            size_t pingIdx, pongIdx;

            if (allCoreDmaBdParams[paramIndices[0]].hasLinkedBd) {

                pingIdx = paramIndices[0];

                pongIdx = paramIndices[1];

            } else {

                pongIdx = paramIndices[0];

                pingIdx = paramIndices[1];

            }

            Value pongHandle = createCoreBd(allCoreDmaBdParams[pongIdx], coreTile, Value());

            Value pingHandle = createCoreBd(allCoreDmaBdParams[pingIdx], coreTile, pongHandle);

            coreBdHandleMap[tileKey][allCoreDmaBdParams[pingIdx].originalBdId] = pingHandle;

            coreBdHandleMap[tileKey][allCoreDmaBdParams[pongIdx].originalBdId] = pongHandle;

        } else {

            for (size_t idx : paramIndices) {

                Value handle = createCoreBd(allCoreDmaBdParams[idx], coreTile, Value());

                coreBdHandleMap[tileKey][allCoreDmaBdParams[idx].originalBdId] = handle;

            }

        }

    }



    // 5c. Create IO configurations for core tiles (create_io + start_io immediately)

    for (const auto &params : allCoreIoConfigParams) {

        if (coreTileMap.find(params.coreKey) == coreTileMap.end())

            continue;

        Value coreTile = coreTileMap[params.coreKey];



        // Look up BD handle by originalBdId traced from SSA operand

        auto tileIt = coreBdHandleMap.find(params.coreKey);

        if (tileIt == coreBdHandleMap.end())

            continue;

        auto bdIt = tileIt->second.find(params.originalBdId);

        if (bdIt == tileIt->second.end()) {

            if (tileIt->second.empty())

                continue;

            bdIt = tileIt->second.begin();

        }

        Value bdHandle = bdIt->second;



        auto coreCreateIoOp = builder.create<dfschedule::ConfigCreateIoOp>(

            loc, dfschedule::IoHandleType::get(builder.getContext()), bdHandle, coreTile,

            builder.getI32IntegerAttr(params.channel), builder.getStringAttr(params.direction),

            builder.getStringAttr(params.ioOperation));

        // start_io immediately after create_io for core tiles (fire-and-forget, not waited on)

        auto coreBdIdOp = builder.create<dfschedule::GetBdIdOp>(loc, builder.getI32Type(), coreTile);

        builder.create<dfschedule::StartIoOp>(loc, dfschedule::EventType::get(builder.getContext()),

                                              coreCreateIoOp.getIoHandle(), coreBdIdOp.getBdId(),

                                              builder.getI32IntegerAttr(0));

    }



    // 6. Build list of core tiles and their configs for merged kernel group

    SmallVector<Value> allCoreTiles;

    SmallVector<Attribute> allTileConfigs;

    SmallVector<Attribute> allComputeKernelArgs;

    

    int tileIdx = 0;

    for (auto &[key, tileInfo] : info.coreTiles) {

        if (coreTileMap.find(key) == coreTileMap.end()) continue;

        Value coreTile = coreTileMap[key];

        allCoreTiles.push_back(coreTile);

        

        // Create config dictionary for this tile

        if (!tileInfo.configDicts.empty()) {

            // Use existing config (should already have all fields including lock IDs)

            allTileConfigs.push_back(tileInfo.configDicts[0]);

        } else {

            // Create default config (for backward compatibility)

            NamedAttrList configAttrs;

            configAttrs.append("tile_index", builder.getI32IntegerAttr(tileIdx));

            configAttrs.append("packet_id", builder.getI32IntegerAttr(tileIdx));

            configAttrs.append("dma_channel", builder.getI32IntegerAttr(0)); // default channel

            configAttrs.append("buffer_mode", builder.getI32IntegerAttr(1)); // 1 = ping-pong

            configAttrs.append("num_buffers", builder.getI32IntegerAttr(2)); // 2 buffers

            configAttrs.append("buffer_size", builder.getI32IntegerAttr(256)); // default 256 bytes

            // Add default lock IDs

            configAttrs.append("ping_acquire_lock_id", builder.getI32IntegerAttr(tileIdx * 4 + 0));

            configAttrs.append("pong_acquire_lock_id", builder.getI32IntegerAttr(tileIdx * 4 + 1));

            configAttrs.append("ping_release_lock_id", builder.getI32IntegerAttr(tileIdx * 4 + 2));

            configAttrs.append("pong_release_lock_id", builder.getI32IntegerAttr(tileIdx * 4 + 3));

            allTileConfigs.push_back(builder.getDictionaryAttr(configAttrs));

        }

        

        // Use the first compute kernel arg for this tile (or default)

        if (!tileInfo.computeKernelArgs.empty()) {

            allComputeKernelArgs.push_back(tileInfo.computeKernelArgs[0]);

        } else {

            allComputeKernelArgs.push_back(SymbolRefAttr::get(builder.getContext(), "compute0"));

        }

        

        tileIdx++;

    }

    

    // 7. Create SINGLE merged load_kernel_group (if core tiles exist)

    Value launchEvent;

    if (!allCoreTiles.empty()) {

        SmallVector<Attribute> calleeAttrs;

        calleeAttrs.push_back(SymbolRefAttr::get(builder.getContext(), info.kernelName));

        

        // Create individual kernel_config ops for each tile (e.g., @kernelconfig_merged0, @kernelconfig_merged1)

        SmallVector<Attribute> kernelConfigSymbols;

        for (size_t i = 0; i < allTileConfigs.size(); ++i) {

            std::string configName = "kernelconfig_merged" + std::to_string(i);

            

            // Create a kernel_config op with a single tile's config

            SmallVector<Attribute> singleTileConfig;

            singleTileConfig.push_back(allTileConfigs[i]);

            

            auto kernelConfigOp = builder.create<dfschedule::DeclareKernelConfigOp>(

                loc,

                dfschedule::KernelConfigType::get(builder.getContext()),

                builder.getStringAttr(configName),

                builder.getArrayAttr(singleTileConfig));

            

            // Store symbol reference

            kernelConfigSymbols.push_back(SymbolRefAttr::get(builder.getContext(), configName));

        }

        

        auto loadKernelGroupOp = builder.create<dfschedule::LoadKernelGroupOp>(

            loc,

            dfschedule::KernelGroupType::get(builder.getContext()),

            allCoreTiles,

            builder.getArrayAttr(calleeAttrs),

            builder.getArrayAttr(allComputeKernelArgs),

            nullptr,  // kernel_config = nullptr (not used)

            builder.getArrayAttr(kernelConfigSymbols));  // distributed_args = [@kernelconfig_merged0, @kernelconfig_merged1, ...]

        

        // 8. Create SINGLE launch_kernel_group

        auto launchKernelGroupOp = builder.create<dfschedule::LaunchKernelGroupOp>(

            loc,

            dfschedule::EventType::get(builder.getContext()),

            loadKernelGroupOp.getKernelGroup());

        

        launchEvent = launchKernelGroupOp.getEvent();

    }

    

    // 9. Create getBdId and start_io for each shim tile

    SmallVector<Value> allEvents;

    if (launchEvent) {

        allEvents.push_back(launchEvent);

    }

    

    for (auto &[key, ioHandles] : shimIoHandles) {

        if (shimTileMap.find(key) == shimTileMap.end()) continue;

        Value shimTile = shimTileMap[key];

        

        // Create getBdId

        auto getBdIdOp = builder.create<dfschedule::GetBdIdOp>(

            loc,

            builder.getI32Type(),

            shimTile);

        

            // Create start_io for each IO handle

        for (Value ioHandle : ioHandles) {

            auto startIoOp = builder.create<dfschedule::StartIoOp>(

                loc,

                dfschedule::EventType::get(builder.getContext()),

                ioHandle,

                getBdIdOp.getBdId(),

                builder.getI32IntegerAttr(0)

            );

            allEvents.push_back(startIoOp.getEvent());

        }

    }

    

    // 10. Create SINGLE merged schedule.wait with ALL events

    if (!allEvents.empty()) {

        builder.create<dfschedule::ScheduleWaitOp>(loc, allEvents);

    }

    

    // ==========================================================

    // PART 2: Add a call to host_canonicalized inside func.func @main()

    // ==========================================================

    Block &mainBlock = funcOp.getBody().front();

    Operation *terminator = mainBlock.getTerminator();

    if (terminator) {

        builder.setInsertionPoint(terminator);

    } else {

        builder.setInsertionPointToEnd(&mainBlock);

    }

    

    // Use dfschedule.launchhost to invoke the host schedule block

    auto hostSymbol = SymbolRefAttr::get(builder.getContext(), "host_canonicalized");

    builder.create<dfschedule::LaunchHostOp>(loc, hostSymbol);

}



// Remove old distributed schedule operations

static void removeOldScheduleOps(ModuleScheduleInfo &info) {

    // Mark operations for removal (in reverse order to handle dependencies)

    SmallVector<Operation*> opsToRemove;

    

    // Remove wait ops first

    for (auto &opWithParent : info.scheduleWaitOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove start_io ops

    for (auto &opWithParent : info.startIoOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove getBdId ops

    for (auto &opWithParent : info.getBdIdOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove launch ops

    for (auto &opWithParent : info.launchKernelGroupOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove load_kernel_group ops

    for (auto &opWithParent : info.loadKernelGroupOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove packet ops

    for (auto &opWithParent : info.packetOps) {

        opsToRemove.push_back(opWithParent.op);

    }



    auto isInHostBlock = [](const OpWithParent &owp) -> bool {

        if (owp.parent && owp.parent->getName().getStringRef() == "dfschedule.host")

            return true;

        return false;

    };



    // Remove createIo ops (but NOT those inside dskernel_receiver or dfschedule.host)

    for (auto &opWithParent : info.configCreateIoOps) {

        if (!opWithParent.isInDSKernelReceiver && !isInHostBlock(opWithParent)) {

            opsToRemove.push_back(opWithParent.op);

        }

    }



    // Remove dmaBd ops (but NOT those inside dskernel_receiver or dfschedule.host)

    for (auto &opWithParent : info.configDmaBdOps) {

        if (!opWithParent.isInDSKernelReceiver && !isInHostBlock(opWithParent)) {

            opsToRemove.push_back(opWithParent.op);

        }

    }

    

    // Remove declareTile ops

    for (auto &opWithParent : info.declareTileOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove declareTensor ops

    for (auto &opWithParent : info.declareTensorOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Erase dfschedule operations (safe, no nested structure issues)

    for (auto *op : opsToRemove) {

        if (op->use_empty()) {

            op->erase();

        }

    }

}



// Remove scf.execute_region blocks, tensor.empty, declare_data, and constants from func.func main

// This is done separately to avoid memory corruption from nested op pointer invalidation

static void removeExecuteRegionsFromMain(func::FuncOp mainFunc) {

    if (!mainFunc) return;

    

    // Collect ops to erase (fresh collection, not using old pointers)

    SmallVector<Operation*> regionsToErase;

    SmallVector<Operation*> declareDataOps;

    SmallVector<Operation*> constantOps;

    SmallVector<Operation*> emptyOps;

    

    mainFunc.walk([&](Operation *op) {

        if (isa<scf::ExecuteRegionOp>(op)) {

            regionsToErase.push_back(op);

        } else if (op->getParentOp() == mainFunc.getOperation()) {

            // Only collect ops that are direct children of main's block

            if (isa<tensor::EmptyOp>(op)) {

                emptyOps.push_back(op);

            } else if (isa<arith::ConstantOp>(op)) {

                constantOps.push_back(op);

            } else if (op->getName().getStringRef() == "dfscheblueprint.declare_data") {

                declareDataOps.push_back(op);

            }

        }

    });

    

    // Erase scf.execute_region ops first (this also erases all nested ops)

    for (auto *op : regionsToErase) {

        if (op->use_empty()) {

            op->erase();

        }

    }

    

    // Erase in dependency order:

    // 1. tensor.empty (no dependencies)

    for (auto *op : emptyOps) {

        if (op->use_empty()) {

            op->erase();

        }

    }

    

    // 2. declare_data (uses constants)

    for (auto *op : declareDataOps) {

        if (op->use_empty()) {

            op->erase();

        }

    }

    

    // 3. constants (used by declare_data, so remove after declare_data)

    for (auto *op : constantOps) {

        if (op->use_empty()) {

            op->erase();

        }

    }

}



} // namespace



namespace mlir {



void ScheduleCanonicalizePass::runOnOperation() {

    ModuleOp moduleOp = getOperation();

    

    ModuleScheduleInfo info;

    

    // Step 1: Collect all dfschedule operations

    collectScheduleOps(moduleOp, info);

    

    // Debug: Print parent information

    printParentInfo(info);

    

    // Early exit if no schedule ops found

    if (info.declareTileOps.empty() && info.loadKernelGroupOps.empty()) {

        return;

    }

    

    // Step 2: Associate packets with tiles

    associatePacketsWithTiles(info);

    

    // Step 3: Find the main function to insert canonicalized host block

    func::FuncOp mainFunc = nullptr;

    moduleOp.walk([&](func::FuncOp funcOp) {

        if (funcOp.getName() == "main") {

            mainFunc = funcOp;

        }

    });

    

    if (!mainFunc) {

        // No main function, skip

        return;

    }

    

    // Step 5: Create canonicalized schedule inside func.func @main()

    // Operations are placed in an scf.execute_region block to group them

    OpBuilder builder(moduleOp.getContext());

    Location loc = mainFunc.getLoc();

    

    createCanonicalizedSchedule(builder, loc, info, moduleOp, mainFunc);

    

    // Step 5.5: Re-collect DMA BD operations after canonicalization to get correct parent info

    // Clear the old collection and re-collect from the canonicalized IR

    info.configDmaBdOps.clear();

    info.configCreateIoOps.clear();

    moduleOp.walk([&](Operation *op) {

        Operation *parentOp = getParentOp(op);

        if (auto configDmaBd = dyn_cast<dfschedule::ConfigDmaBdOp>(op)) {

            info.configDmaBdOps.push_back(OpWithParent(op, parentOp));

        } else if (auto createIo = dyn_cast<dfschedule::ConfigCreateIoOp>(op)) {

            info.configCreateIoOps.push_back(OpWithParent(op, parentOp));

        }

    });

    

    // Step 5.6: Associate DMA configs with shim tiles (after canonicalization)

    associateDmaWithShimTiles(info);

    

    // Step 6: Remove old distributed dfschedule operations

    removeOldScheduleOps(info);

    

    // Step 7: Remove scf.execute_region blocks (contains extract_slice, routing ops)

    // and tensor.empty from func.func main

    removeExecuteRegionsFromMain(mainFunc);

}



} // namespace mlir



"

Gemini said
This code implements a very typical "Collect-and-Rebuild" style Pass. It works, but in the eyes of an MLIR expert it has several serious engineering hazards and performance bottlenecks, especially on large IR (million-line scale).

Below are the main problems with this Canonicalize logic and suggested improvements:

1. Fatal performance bottleneck: strings as Map keys
The code makes heavy use of makeTensorTypeKey and std::map<std::string, Value>.

Problem: in MLIR, Types and Attributes are essentially uniqued pointers (Unique Pointers). Comparing two Type objects by pointer is extremely fast. Converting them to strings and then doing string lookups costs orders of magnitude in performance. At a million lines, the compiler would spend minutes on string concatenation and hashing.

Improvement: use std::map<Type, Value> or llvm::DenseMap<Type, Value> directly.

2. Fragile manual SSA-chain reconstruction (Slice Tracing)
The code tries to rebuild the partitiontensor -> extract_slice chain by hand.

Problem: you maintain indices by hand with uniqueSliceParams and sliceMap. This "procedural" logic is very hardcoded. If the IR changes slightly (e.g. an extra unrealized_conversion_cast in between), your tracing breaks, sourceTensor is not found, and you end up generating a "wrong but runnable" placeholder like memref.alloc.

Improvement: use mlir::IRMapping (formerly BlockAndValueMapping). When you clone an operation, IRMapping automatically maps the old input values to the new output values for you; you don't need to maintain indices by hand.

3. The risk of "delete first, check later"
The removeOldScheduleOps function deletes operations in a loop and relies on op->use_empty().

Problem: in complex IR transformations, because of nested ExecuteRegions, simple sequential deletion easily leads to null-pointer dereferences or dangling pointers. If some deleted Op still has a User that has not been cleaned up, erase() crashes immediately.

Improvement: use rewriter.eraseOp(op), or let a PatternRewriter manage the deletion logic uniformly at the end of the Walk.

4. Higher complexity from repeated Walks
You call moduleOp.walk several times in runOnOperation.

Problem: every walk is O(N). On large models, rescanning the whole Module again and again is extremely expensive.

Improvement: collect all information in a single walk where possible, or use an indexed data structure (such as a SymbolTable) to jump directly to the target Op.

5. Non-converging logic (Non-Canonical Form)
Your Shim DMA merge logic deduplicates with a std::set.

Problem: if you have two DMA tasks whose sliceIndex differs but which are physically contiguous (and could be merged into one longer DMA), your logic cannot recognize this "spatial contiguity", because it only does a simple attribute comparison.

Improvement: add Address Analysis to the merge logic to decide whether two offset + len ranges can be joined seamlessly.

Expert refactoring advice (The "Clean" Way)
Rather than writing a function of several hundred lines that rebuilds everything by hand, adopt the following strategy:

A. Use IRMapping to simplify cloning
Don't record sliceIndex by hand; use a mapping table:

C++
mlir::IRMapping mapper;
// When you clone the base Tensor
auto newConst = builder.clone(*oldConst, mapper);
mapper.map(oldConst->getResult(0), newConst->getResult(0));

// When cloning Slices later, the mapper handles the mapping automatically
auto newSlice = builder.clone(*oldSlice, mapper); 
B. Structured "unboxing"
Don't collect ExecuteRegions by hand and then delete them. Use inlineRegionBefore as we discussed earlier:

Step 1: Walk all routing.RoutingCreate ops and inline their contents into func.func.

Step 2: now all Shim create_io ops are at the same level.

Step 3: run a local match to find create_io ops for the same Tile and merge them.

C. Fix the Shim DMA merge logic
The current dedup logic is too "hard":

C++
// Your logic
using ShimMergeKey = std::tuple<TileKey, int32_t, int64_t>;
// ... if sliceIndex differs, they are never merged
Suggestion: switch to Tile-based Bucket collection. First put all DMAs into a List per Tile, then sort the List by offset and check whether they can be merged.

Summary
Your current code is more of an "IR Translator" (translating one dialect into another) than a "Canonicalizer".

The biggest risk: you hand-build a very complex ModuleScheduleInfo struct that mirrors the state of the IR. In compiler development, the truth always lives in the IR itself. Avoid storing too much state in side-structures, because once the side information gets out of sync with the IR, you get very subtle bugs.

Would you like me to write a more robust, address-based merge example for the associateDmaWithShimTiles part?

You said
What is IRMapping for?

Gemini said
Simply put, IRMapping (formerly BlockAndValueMapping) is MLIR's **"SSA value mapping dictionary"**.

When the compiler performs transformations (especially Clone or Inlining), it solves a core problem: "How do I make the newly generated code correctly reference the newly generated operands, instead of pointing to old code that may already have been deleted?"

1. Why do you need it? (Pain points)
In your earlier code, you maintained all sorts of indices by hand, such as sliceMap and partitionedTensorMap. That is tedious and error-prone.

Suppose you have this code:

MLIR
%0 = arith.constant 1
%1 = arith.addi %0, %0
You want to move it into a new dfschedule.host block.

You clone %0 and get %0_new.

When it is %1's turn to be cloned, if you copy it directly, it will try to add %0 (the old one).

Consequence: the old %0 will soon be deleted by you, and your new code will point at a "dead" value, crashing the compiler (Segfault).

2. How IRMapping works
IRMapping is like an automatically updated mapping table. You tell it: "From now on, wherever %0 is referenced, replace it with %0_new."

Core flow:
Register a mapping: when you clone a constant %0, call mapper.map(old_val, new_val).

Automatic lookup: when you clone %1 with builder.clone(op, mapper), MLIR automatically checks all of the Op's input operands.

It sees that %1 uses %0.

It finds in the mapper that %0 has become %0_new.

It automatically points the newly generated addi at %0_new.

3. Code comparison: manual bookkeeping vs. IRMapping
Your old approach (HARDCODE risk):
C++
// Requires maintaining complicated maps and indices
std::map<size_t, Value> sliceMap;
// ... all sorts of logic to make sure the index is correct ...
auto newSlice = builder.create<tensor::ExtractSliceOp>(..., sliceMap[params.parentIndex]);
The expert approach with IRMapping:
C++
mlir::IRMapping mapper;

// 1. Clone the base operation and register it
Operation *newConst = builder.clone(*oldConst, mapper);
// The mapper now remembers: old_val -> new_val

// 2. Clone subsequent operations in bulk; no need to track who is who by hand
for (auto &oldOp : oldOps) {
    // This one line of magic automatically replaces all input operands with the "new" versions
    Operation *newOp = builder.clone(*oldOp, mapper);
}
4. Concrete uses in your code
For your very long passschedulecanonicalize.h, IRMapping can remove at least 200 lines of code:

Handling nested Slices: you no longer need a loop like while(progress) to unnest layer by layer. Just clone in order. As long as the parent Slice has been cloned and is in the mapper, the child Slice is connected automatically when cloned.

Cleaning up ModuleScheduleInfo: you no longer need to store index and partitionIndex in SliceParams. The mapper is your natural index.

Cross-block merging: when you pull out the contents of multiple routing.RoutingCreate ops, IRMapping guarantees that even if several blocks use the same %tile, after merging they all correctly point to the single DeclareTileOp.

5. When is it mandatory?
Inline: moving the contents of a function or Region to another place.

Unrolling (loop unrolling): copying a loop body multiple times.

Module reorganization: like what you are doing now, gathering Ops scattered all over into a single dfschedule.host.

Expert tip
The most common mistake with IRMapping is forgetting to initialize the mapping.
If the logic you are moving depends on values already defined outside (e.g. the arguments of the main function), you must manually map those initial values before you start cloning.

C++
mapper.map(oldFunc.getArgument(0), newFunc.getArgument(0));
Would you like me to try rewriting the Slice-reconstruction part of that complicated createCanonicalizedSchedule logic with IRMapping? It would make the code much smoother.

You said
Use %0 = arith.constant 1

%1 = arith.addi %0, %0 as an example

Gemini said
To let you fully understand the power of IRMapping, let's run a **"soul cloning"** experiment on the two lines of code you gave.

Imagine you now want to move this logic from FuncA to FuncB.

1. Original IR (Source)
This is what we want to clone:

MLIR
// The original Block
%0 = arith.constant 1 : i32
%1 = arith.addi %0, %0 : i32
2. C++ walkthrough
Without IRMapping, the cloned %1 cannot find its operand (%0 still stays in the old function), and the new function becomes a "dead end".

The standard way to use IRMapping is:

C++
mlir::IRMapping mapper;
mlir::OpBuilder builder(context);

// --- Step A: clone the first Op ---
// Assume originalOp0 is that arith.constant
Operation *clonedOp0 = builder.clone(*originalOp0, mapper);

// At this point the mapper has automatically made a "registration entry":
// { %0 (old) -> %0_new (new) }

// --- Step B: clone the second Op ---
// Assume originalOp1 is that arith.addi
Operation *clonedOp1 = builder.clone(*originalOp1, mapper);

// Here is the magic!
// builder.clone automatically asks the mapper: "Hey, what is the value %0 that addi depends on called in the new world?"
// The mapper answers: "It's called %0_new."
// So the newly generated addi is automatically connected to %0_new.
3. IR after cloning (Target)
The resulting new code looks perfectly natural, and the SSA chain is intact:

MLIR
// The new Block
%0_new = arith.constant 1 : i32
%1_new = arith.addi %0_new, %0_new : i32  // automatically points to the new constant
4. Why does this matter so much for you?
In your complex ScheduleCanonicalizePass you have many nested extract_slice ops. Without IRMapping, your logic becomes:

Find the parent of Slice_A.

Look up in your std::map whether the Value for the parent has already been created.

If it has, take it out and pass it as an argument to builder.create<Slice_A>(...).

With IRMapping, the logic simplifies to:
You just clone all Ops in root-to-leaf (PreOrder) order. IRMapping follows you like a shadow and handles all the pointer replacement automatically. You don't need to care who Slice_A's parent is at all; as long as it was cloned earlier, the mapper takes care of everything.

Expert tips
Many-to-one mapping: if you want to merge several old values into one new value (e.g. make all Shim Tile references point to the same newly declared Tile), you can call mapper.map(oldValue, sharedNewValue) by hand.

Cleanup: if your Pass runs multiple times, remember to clear the mapper with mapper.clear().

With this example, do you see that the manually maintained index logic in your earlier code can be replaced perfectly by this mapper? Shall I refactor that core cloning logic for you?

You said
Doesn't CLONE always require a MAPPER? Is there a version that doesn't take one?

Gemini said
Yes, in MLIR's C++ API clone does have a version without a Mapper, but it has to be used carefully; if you are not careful it will crash the compiler.

As an expert, let me break down the essential difference between the two kinds of clone:

1. op->clone() without a Mapper
This is an ordinary member function of the mlir::Operation class.

What it does: it copies the Op itself, its attributes, its types, and its Regions.

Its limitation: it copies the Operands verbatim.

Example:
If you clone %1 = addi %0, %0, the new Op still tries to connect to the %0 in the original code.

When can you use the version without a Mapper?
Local copy: you want to duplicate an instruction within the same Block, and you want the new instruction's inputs to be exactly the same as the old one's.

Read-only analysis: you just want to temporarily clone a piece of IR for some experimental analysis, with no plan to put it back into the real program flow.

Danger sign: if you move such a Mapper-less clone into a new function and the old function is later deleted, the clone will point to an invalid memory address.

2. builder.clone(op, mapper) with a Mapper
This is a utility function provided by mlir::OpBuilder or mlir::IRRewriter.

What it does: while copying the Op, it consults this "registry" (the Mapper).

Its advantage: if an operand has an entry in the table, it automatically replaces the new Op's input with the value from the "new world".

3. Why is the Mapper "mandatory" in your code?
What you are doing is a "structural reorganization" (moving from scf.execute_region into dfschedule.host). In that case, not using a Mapper leads to these two disasters:

Scope Violation:
The old SSA values (e.g. %subview) are defined inside scf.execute_region. Once you clone ops outside, the new Ops still reference the value "inside the box". That violates MLIR's Dominance relation, and the Verifier reports an error.

Dangling Pointer:
Your code eventually calls removeExecuteRegionsFromMain to delete the old Regions. The moment they are deleted, the old SSA values disappear with them. Without the Mapper linking old to new, your new IR is simply "disconnected".

4. Expert summary comparison
Approach	Syntax	Operand handling	When to use
Bare clone	op->clone()	Points to the old values	Copying within the same Block, no relocation.
Mapped clone	builder.clone(op, mapper)	Points to the mapped new values	Moving code across Blocks / Regions (your case).
💡 A labor-saving trick
If you still find managing IRMapping by hand tedious, there is a higher-level API:
rewriter.inlineRegionBefore(sourceRegion, targetBlock, insertionPoint).

This function automatically keeps all operands inside the Blocks consistent. It is much faster than cloning one by one and less error-prone.

Since we've come this far, would you like to see how to refactor your createOneShimBd logic concisely with rewriter.clone plus IRMapping? (It would save you a lot of manual sliceIndex lookups.)