module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 1 : i64, routing.control_plan_dedicated_shim = 1 : i64, routing.control_plan_exclusive = 1 : i64, routing.control_plan_group_reg_write = 1 : i64, routing.control_plan_mm2s_ch = 0 : i64, routing.control_plan_op_control_packet = 1 : i64, routing.control_plan_s2mm_ch = 0 : i64, routing.control_plan_shim_bd_ctrl = 1 : i64, routing.control_plan_shim_col = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 1 : i64, routing.output_tile_major = 1 : i64, routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}, routing.spatial_halo_buf_size = 4864 : i64, routing.spatial_out_rounds = 16 : i64, tensor_0.halo = {k_rounds = 4 : i32, k_slice = 256 : i32, k_step = 224 : i32, l2_rounds = 4 : i32, l2_slice = 19 : i32, l2_step = 14 : i32, ow_t = 28 : i32, row_pitch = 928 : i32, slice = 61 : i32, split_dim = 0 : i32, step = 56 : i32, w_rounds = 4 : i32, w_slice = 64 : i32, w_step = 56 : i32}, tensor_0.layout_transform = "dma_shuffle", tensor_1.layout_transform = "dma_shuffle"} {
  func.func @routing(%arg0: !emitc.ptr<!emitc.opaque<"XAie_DevInst">>, %arg1: memref<230x232x4xi8>, %arg2: memref<64x7x7x4xi8>, %arg3: memref<112x112x64xi8>) {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg1 : memref<230x232x4xi8>
    %1 = bufferization.to_tensor %arg2 : memref<64x7x7x4xi8>
    %2 = bufferization.to_tensor %arg3 : memref<112x112x64xi8>
    scf.execute_region {
      %3 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %7 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %8 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %9 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %10 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %11 = routinghw.ioshimtilecreate {IOID = 13 : i32, channelused = 0 : i32, col = 1 : i32, comments = "shim_dma_13", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %12 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %13 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %14 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.enableexttoaieshimport %11 : i32 {portdirection = "SOUTH", portidx = 3 : i32} -> i32
        %16 = routinghw.connectsinglestreamswitchport %11 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 0} : i32
        %17 = routinghw.connectsinglestreamswitchport %13 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %18 = routinghw.connectsinglestreamswitchport %14 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %19 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %20 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %21 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %22 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %23 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %24 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %25 = routinghw.connectsinglestreamswitchport %10 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        "routing.yield"() : () -> ()
      }
      %4 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %7 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %8 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %9 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %10 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %11 = routinghw.ioshimtilecreate {IOID = 14 : i32, channelused = 0 : i32, col = 2 : i32, comments = "shim_dma_14", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %12 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %13 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %14 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.enableexttoaieshimport %11 : i32 {portdirection = "SOUTH", portidx = 3 : i32} -> i32
        %16 = routinghw.connectsinglestreamswitchport %11 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 0} : i32
        %17 = routinghw.connectsinglestreamswitchport %13 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %18 = routinghw.connectsinglestreamswitchport %14 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %19 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %20 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %21 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %22 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %23 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %24 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %25 = routinghw.connectsinglestreamswitchport %10 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        "routing.yield"() : () -> ()
      }
      %5 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %7 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %8 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %9 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %10 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %11 = routinghw.ioshimtilecreate {IOID = 15 : i32, channelused = 0 : i32, col = 3 : i32, comments = "shim_dma_15", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %12 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %13 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %14 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.enableexttoaieshimport %11 : i32 {portdirection = "SOUTH", portidx = 3 : i32} -> i32
        %16 = routinghw.connectsinglestreamswitchport %11 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 0} : i32
        %17 = routinghw.connectsinglestreamswitchport %13 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %18 = routinghw.connectsinglestreamswitchport %14 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %19 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %20 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %21 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %22 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %23 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %24 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %25 = routinghw.connectsinglestreamswitchport %10 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        "routing.yield"() : () -> ()
      }
      %6 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %7 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %8 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %9 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %10 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %11 = routinghw.ioshimtilecreate {IOID = 16 : i32, channelused = 0 : i32, col = 4 : i32, comments = "shim_dma_16", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %12 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %13 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %14 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.enableexttoaieshimport %11 : i32 {portdirection = "SOUTH", portidx = 3 : i32} -> i32
        %16 = routinghw.connectsinglestreamswitchport %11 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 0} : i32
        %17 = routinghw.connectsinglestreamswitchport %13 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %18 = routinghw.connectsinglestreamswitchport %14 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %19 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %20 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %21 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %22 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %23 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %24 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        %25 = routinghw.connectsinglestreamswitchport %10 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %3 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %7 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %8 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %9 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %10 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %11 = routinghw.ioshimtilecreate {IOID = 17 : i32, channelused = 1 : i32, col = 1 : i32, comments = "shim_dma_17", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %12 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %13 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %14 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.enableexttoaieshimport %11 : i32 {portdirection = "SOUTH", portidx = 7 : i32} -> i32
        %16 = routinghw.connectsinglestreamswitchport %11 : {slaveportdirection = "SOUTH", slaveportidx = 7, masterportdirection = "NORTH", masterportidx = 1} : i32
        %17 = routinghw.connectsinglestreamswitchport %13 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %18 = routinghw.connectsinglestreamswitchport %14 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %19 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %20 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %21 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %22 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %23 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %24 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %25 = routinghw.connectsinglestreamswitchport %10 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %26 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %27 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %28 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %29 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %30 = routinghw.ioshimtilecreate {IOID = 18 : i32, channelused = 0 : i32, col = 4 : i32, comments = "shim_dma_18", dmadirection = 1 : i32, row = 0 : i32} -> i32
        %31 = routinghw.connectpktstreamswitchport %26 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 4 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %32 = routinghw.connectpktstreamswitchport %27 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 5 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %33 = routinghw.connectpktstreamswitchport %28 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 6 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %34 = routinghw.connectpktstreamswitchport %29 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "NONE", forwardmasterportidx = 0 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 7 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %35 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %36 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %37 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %38 = routinghw.enableaietoextshimport %30 : i32 {portdirection = "NORTH", portidx = 1 : i32} -> i32
        %39 = routinghw.connectpktstreamswitchport %29 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "SOUTH", forwardmasterportidx = 1 : i32, localdmaarbiter = 2 : i32, localdmadirection = "NONE", localdmapktid = 0 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %40 = routinghw.connectsinglestreamswitchport %35 : {slaveportdirection = "NORTH", slaveportidx = 1, masterportdirection = "SOUTH", masterportidx = 1} : i32
        %41 = routinghw.connectsinglestreamswitchport %36 : {slaveportdirection = "NORTH", slaveportidx = 1, masterportdirection = "SOUTH", masterportidx = 1} : i32
        %42 = routinghw.connectsinglestreamswitchport %30 : {slaveportdirection = "NORTH", slaveportidx = 1, masterportdirection = "SOUTH", masterportidx = 1} : i32
        "routing.yield"() : () -> ()
      }
      %4 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %7 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %8 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %9 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %10 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %11 = routinghw.ioshimtilecreate {IOID = 19 : i32, channelused = 1 : i32, col = 2 : i32, comments = "shim_dma_19", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %12 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %13 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %14 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %16 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %17 = routinghw.enableexttoaieshimport %11 : i32 {portdirection = "SOUTH", portidx = 7 : i32} -> i32
        %18 = routinghw.connectsinglestreamswitchport %11 : {slaveportdirection = "SOUTH", slaveportidx = 7, masterportdirection = "NORTH", masterportidx = 1} : i32
        %19 = routinghw.connectsinglestreamswitchport %13 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %20 = routinghw.connectsinglestreamswitchport %14 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %21 = routinghw.connectsinglestreamswitchport %15 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "WEST", masterportidx = 3} : i32
        %22 = routinghw.connectsinglestreamswitchport %15 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %23 = routinghw.connectsinglestreamswitchport %16 : {slaveportdirection = "EAST", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 1} : i32
        %24 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %25 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %26 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %27 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %28 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %29 = routinghw.connectsinglestreamswitchport %10 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %30 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %31 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %32 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %33 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 4 : i32} -> i32
        %34 = routinghw.ioshimtilecreate {IOID = 20 : i32, channelused = 1 : i32, col = 4 : i32, comments = "shim_dma_20", dmadirection = 1 : i32, row = 0 : i32} -> i32
        %35 = routinghw.connectpktstreamswitchport %30 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 1 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 8 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %36 = routinghw.connectpktstreamswitchport %31 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 9 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 1 : i32, receiveslaveslot = 3 : i32} -> i32
        %37 = routinghw.connectpktstreamswitchport %32 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 10 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %38 = routinghw.connectpktstreamswitchport %33 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "NONE", forwardmasterportidx = 0 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 11 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %39 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %40 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %41 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %42 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %43 = routinghw.enableaietoextshimport %34 : i32 {portdirection = "NORTH", portidx = 3 : i32} -> i32
        %44 = routinghw.connectpktstreamswitchport %33 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "SOUTH", forwardmasterportidx = 1 : i32, localdmaarbiter = 2 : i32, localdmadirection = "NONE", localdmapktid = 0 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %45 = routinghw.connectsinglestreamswitchport %39 : {slaveportdirection = "NORTH", slaveportidx = 1, masterportdirection = "SOUTH", masterportidx = 2} : i32
        %46 = routinghw.connectsinglestreamswitchport %40 : {slaveportdirection = "NORTH", slaveportidx = 2, masterportdirection = "SOUTH", masterportidx = 2} : i32
        %47 = routinghw.connectsinglestreamswitchport %41 : {slaveportdirection = "NORTH", slaveportidx = 2, masterportdirection = "SOUTH", masterportidx = 2} : i32
        %48 = routinghw.connectsinglestreamswitchport %34 : {slaveportdirection = "NORTH", slaveportidx = 2, masterportdirection = "SOUTH", masterportidx = 3} : i32
        "routing.yield"() : () -> ()
      }
      %5 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %7 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %8 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %9 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %10 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %11 = routinghw.ioshimtilecreate {IOID = 21 : i32, channelused = 1 : i32, col = 3 : i32, comments = "shim_dma_21", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %12 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %13 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %14 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %16 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %17 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 4 : i32} -> i32
        %18 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 4 : i32} -> i32
        %19 = routinghw.enableexttoaieshimport %11 : i32 {portdirection = "SOUTH", portidx = 7 : i32} -> i32
        %20 = routinghw.connectsinglestreamswitchport %11 : {slaveportdirection = "SOUTH", slaveportidx = 7, masterportdirection = "NORTH", masterportidx = 1} : i32
        %21 = routinghw.connectsinglestreamswitchport %13 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %22 = routinghw.connectsinglestreamswitchport %14 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %23 = routinghw.connectsinglestreamswitchport %15 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "WEST", masterportidx = 3} : i32
        %24 = routinghw.connectsinglestreamswitchport %16 : {slaveportdirection = "EAST", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 2} : i32
        %25 = routinghw.connectsinglestreamswitchport %17 : {slaveportdirection = "SOUTH", slaveportidx = 2, masterportdirection = "WEST", masterportidx = 2} : i32
        %26 = routinghw.connectsinglestreamswitchport %17 : {slaveportdirection = "SOUTH", slaveportidx = 2, masterportdirection = "NORTH", masterportidx = 1} : i32
        %27 = routinghw.connectsinglestreamswitchport %18 : {slaveportdirection = "EAST", slaveportidx = 2, masterportdirection = "NORTH", masterportidx = 1} : i32
        %28 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %29 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %30 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %31 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %32 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %33 = routinghw.connectsinglestreamswitchport %10 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %34 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %35 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %36 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %37 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 5 : i32} -> i32
        %38 = routinghw.ioshimtilecreate {IOID = 22 : i32, channelused = 0 : i32, col = 3 : i32, comments = "shim_dma_22", dmadirection = 1 : i32, row = 0 : i32} -> i32
        %39 = routinghw.connectpktstreamswitchport %34 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 1 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 12 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %40 = routinghw.connectpktstreamswitchport %35 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 13 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 1 : i32, receiveslaveslot = 3 : i32} -> i32
        %41 = routinghw.connectpktstreamswitchport %36 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 14 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %42 = routinghw.connectpktstreamswitchport %37 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "NONE", forwardmasterportidx = 0 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 15 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %43 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 4 : i32} -> i32
        %44 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %45 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %46 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %47 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %48 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %49 = routinghw.enableaietoextshimport %38 : i32 {portdirection = "NORTH", portidx = 1 : i32} -> i32
        %50 = routinghw.connectpktstreamswitchport %37 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "SOUTH", forwardmasterportidx = 1 : i32, localdmaarbiter = 2 : i32, localdmadirection = "NONE", localdmapktid = 0 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %51 = routinghw.connectsinglestreamswitchport %43 : {slaveportdirection = "NORTH", slaveportidx = 1, masterportdirection = "SOUTH", masterportidx = 2} : i32
        %52 = routinghw.connectsinglestreamswitchport %44 : {slaveportdirection = "NORTH", slaveportidx = 2, masterportdirection = "WEST", masterportidx = 3} : i32
        %53 = routinghw.connectsinglestreamswitchport %45 : {slaveportdirection = "EAST", slaveportidx = 3, masterportdirection = "SOUTH", masterportidx = 2} : i32
        %54 = routinghw.connectsinglestreamswitchport %46 : {slaveportdirection = "NORTH", slaveportidx = 2, masterportdirection = "SOUTH", masterportidx = 2} : i32
        %55 = routinghw.connectsinglestreamswitchport %47 : {slaveportdirection = "NORTH", slaveportidx = 2, masterportdirection = "SOUTH", masterportidx = 2} : i32
        %56 = routinghw.connectsinglestreamswitchport %38 : {slaveportdirection = "NORTH", slaveportidx = 2, masterportdirection = "SOUTH", masterportidx = 1} : i32
        "routing.yield"() : () -> ()
      }
      %6 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %7 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %8 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %9 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %10 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %11 = routinghw.ioshimtilecreate {IOID = 23 : i32, channelused = 1 : i32, col = 4 : i32, comments = "shim_dma_23", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %12 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %13 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %14 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %16 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 4 : i32} -> i32
        %17 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 4 : i32} -> i32
        %18 = routinghw.tilecreate {col = 2 : i32, comments = "tile in path", row = 4 : i32} -> i32
        %19 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 4 : i32} -> i32
        %20 = routinghw.tilecreate {col = 1 : i32, comments = "tile in path", row = 5 : i32} -> i32
        %21 = routinghw.enableexttoaieshimport %11 : i32 {portdirection = "SOUTH", portidx = 7 : i32} -> i32
        %22 = routinghw.connectsinglestreamswitchport %11 : {slaveportdirection = "SOUTH", slaveportidx = 7, masterportdirection = "NORTH", masterportidx = 3} : i32
        %23 = routinghw.connectsinglestreamswitchport %13 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 3} : i32
        %24 = routinghw.connectsinglestreamswitchport %14 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 3} : i32
        %25 = routinghw.connectsinglestreamswitchport %15 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 3} : i32
        %26 = routinghw.connectsinglestreamswitchport %16 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "WEST", masterportidx = 3} : i32
        %27 = routinghw.connectsinglestreamswitchport %17 : {slaveportdirection = "EAST", slaveportidx = 3, masterportdirection = "WEST", masterportidx = 3} : i32
        %28 = routinghw.connectsinglestreamswitchport %18 : {slaveportdirection = "EAST", slaveportidx = 3, masterportdirection = "WEST", masterportidx = 3} : i32
        %29 = routinghw.connectsinglestreamswitchport %19 : {slaveportdirection = "EAST", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 2} : i32
        %30 = routinghw.connectsinglestreamswitchport %20 : {slaveportdirection = "SOUTH", slaveportidx = 2, masterportdirection = "NORTH", masterportidx = 1} : i32
        %31 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %32 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %33 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %34 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %35 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "EAST", masterportidx = 1} : i32
        %36 = routinghw.connectsinglestreamswitchport %9 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %37 = routinghw.connectsinglestreamswitchport %10 : {slaveportdirection = "WEST", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %38 = routinghw.tilecreate {col = 1 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %39 = routinghw.tilecreate {col = 2 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %40 = routinghw.tilecreate {col = 3 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %41 = routinghw.tilecreate {col = 4 : i32, comments = "core_tile", row = 6 : i32} -> i32
        %42 = routinghw.ioshimtilecreate {IOID = 24 : i32, channelused = 1 : i32, col = 3 : i32, comments = "shim_dma_24", dmadirection = 1 : i32, row = 0 : i32} -> i32
        %43 = routinghw.connectpktstreamswitchport %38 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 17 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %44 = routinghw.connectpktstreamswitchport %39 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 18 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %45 = routinghw.connectpktstreamswitchport %40 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "EAST", forwardmasterportidx = 2 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 19 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %46 = routinghw.connectpktstreamswitchport %41 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "NONE", forwardmasterportidx = 0 : i32, localdmaarbiter = 2 : i32, localdmadirection = "DMA", localdmapktid = 20 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 2 : i32, receiveslaveslot = 3 : i32} -> i32
        %47 = routinghw.tilecreate {col = 4 : i32, comments = "tile in path", row = 5 : i32} -> i32
        %48 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 5 : i32} -> i32
        %49 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 4 : i32} -> i32
        %50 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 3 : i32} -> i32
        %51 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %52 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %53 = routinghw.tilecreate {col = 3 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %54 = routinghw.enableaietoextshimport %42 : i32 {portdirection = "NORTH", portidx = 3 : i32} -> i32
        %55 = routinghw.connectpktstreamswitchport %41 : i32 {forwardmasterarbiter = 2 : i32, forwardmasterdirection = "SOUTH", forwardmasterportidx = 1 : i32, localdmaarbiter = 2 : i32, localdmadirection = "NONE", localdmapktid = 0 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavearbiter = 2 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %56 = routinghw.connectsinglestreamswitchport %47 : {slaveportdirection = "NORTH", slaveportidx = 1, masterportdirection = "WEST", masterportidx = 3} : i32
        %57 = routinghw.connectsinglestreamswitchport %48 : {slaveportdirection = "EAST", slaveportidx = 3, masterportdirection = "SOUTH", masterportidx = 1} : i32
        %58 = routinghw.connectsinglestreamswitchport %49 : {slaveportdirection = "NORTH", slaveportidx = 1, masterportdirection = "SOUTH", masterportidx = 1} : i32
        %59 = routinghw.connectsinglestreamswitchport %50 : {slaveportdirection = "NORTH", slaveportidx = 1, masterportdirection = "SOUTH", masterportidx = 3} : i32
        %60 = routinghw.connectsinglestreamswitchport %51 : {slaveportdirection = "NORTH", slaveportidx = 3, masterportdirection = "SOUTH", masterportidx = 3} : i32
        %61 = routinghw.connectsinglestreamswitchport %52 : {slaveportdirection = "NORTH", slaveportidx = 3, masterportdirection = "SOUTH", masterportidx = 3} : i32
        %62 = routinghw.connectsinglestreamswitchport %42 : {slaveportdirection = "NORTH", slaveportidx = 3, masterportdirection = "SOUTH", masterportidx = 3} : i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
