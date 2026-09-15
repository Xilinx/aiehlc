module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.fullconnect_auto = 1 : i64, routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}} {
  func.func @main(%arg0: memref<256x256xi8>, %arg1: memref<256x256xi8>, %arg2: memref<256x256xi8>) {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg0 : memref<256x256xi8>
    %1 = routing.routingcreatescheduletensor %0 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    %2 = bufferization.to_tensor %arg1 : memref<256x256xi8>
    %3 = routing.routingcreatescheduletensor %2 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    %4 = bufferization.to_tensor %arg2 : memref<256x256xi8>
    %5 = routing.routingcreatescheduletensor %4 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    scf.execute_region {
      %6 = routing.partitiontensor %3 : tensor<256x256xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "col", replicateOn = "row", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4, slice_tiling = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 256, step = 256, rounds = 1, slice_tiling = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4>>>>
} -> tensor<256x256xi8>
      %7 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_data %6, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %12 = dmaphop.tile{TILETYPE = "core", col = 0, row = 3} -> !dmaphop.tile
        %13 = dmaphop.port @f0_corePortIn0 on %12 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %14 = dmaphop.port @f0_corePortOut0 on %12 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.tile{TILETYPE = "core", col = 0, row = 4} -> !dmaphop.tile
        %16 = dmaphop.port @f0_corePortIn1 on %15 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %17 = dmaphop.port @f0_corePortOut1 on %15 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %18 = dmaphop.tile{TILETYPE = "core", col = 0, row = 5} -> !dmaphop.tile
        %19 = dmaphop.port @f0_corePortIn2 on %18 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %20 = dmaphop.port @f0_corePortOut2 on %18 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %21 = dmaphop.tile{TILETYPE = "core", col = 0, row = 6} -> !dmaphop.tile
        %22 = dmaphop.port @f0_corePortIn3 on %21 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %23 = dmaphop.port @f0_corePortOut3 on %21 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %24 = dmaphop.consumer @f0_consumer0 {dma_port = 1 : i64, from = @f0_corePortIn0}
        %25 = dmaphop.consumer @f0_consumer1 {dma_port = 1 : i64, from = @f0_corePortIn1}
        %26 = dmaphop.consumer @f0_consumer2 {dma_port = 1 : i64, from = @f0_corePortIn2}
        %27 = dmaphop.consumer @f0_consumer3 {dma_port = 1 : i64, from = @f0_corePortIn3}
        %28 = dmaphop.tile{TILETYPE = "shim", col = 0, row = 0} -> !dmaphop.tile
        %29 = dmaphop.port @f0_shimPortOut on %28 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %30 = dmaphop.port @f0_shimPortIn on %28 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %31 = dmaphop.create_hop %29 -> %13 -> !dmaphop.hop
        %32 = dmaphop.create_hop %14 -> %16 -> !dmaphop.hop
        %33 = dmaphop.create_hop %17 -> %19 -> !dmaphop.hop
        %34 = dmaphop.create_hop %20 -> %22 -> !dmaphop.hop
        %35 = dmaphop.create_path[%31, %32, %33, %34] {producers = [[@f0_shimPortIn]], consumers = [[@f0_consumer0, @f0_consumer1, @f0_consumer2, @f0_consumer3]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %11 into %35 consumer(%11, %11, %11, %11 at %13, %16, %19, %22) : tensor<64x256xi8> !dmaphop.path tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %35
        "routing.yield"() : () -> ()
      }
      %8 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_data %6, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %12 = dmaphop.tile{TILETYPE = "core", col = 1, row = 3} -> !dmaphop.tile
        %13 = dmaphop.port @f1_corePortIn0 on %12 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %14 = dmaphop.port @f1_corePortOut0 on %12 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.tile{TILETYPE = "core", col = 1, row = 4} -> !dmaphop.tile
        %16 = dmaphop.port @f1_corePortIn1 on %15 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %17 = dmaphop.port @f1_corePortOut1 on %15 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %18 = dmaphop.tile{TILETYPE = "core", col = 1, row = 5} -> !dmaphop.tile
        %19 = dmaphop.port @f1_corePortIn2 on %18 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %20 = dmaphop.port @f1_corePortOut2 on %18 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %21 = dmaphop.tile{TILETYPE = "core", col = 1, row = 6} -> !dmaphop.tile
        %22 = dmaphop.port @f1_corePortIn3 on %21 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %23 = dmaphop.port @f1_corePortOut3 on %21 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %24 = dmaphop.consumer @f1_consumer0 {dma_port = 1 : i64, from = @f1_corePortIn0}
        %25 = dmaphop.consumer @f1_consumer1 {dma_port = 1 : i64, from = @f1_corePortIn1}
        %26 = dmaphop.consumer @f1_consumer2 {dma_port = 1 : i64, from = @f1_corePortIn2}
        %27 = dmaphop.consumer @f1_consumer3 {dma_port = 1 : i64, from = @f1_corePortIn3}
        %28 = dmaphop.tile{TILETYPE = "shim", col = 1, row = 0} -> !dmaphop.tile
        %29 = dmaphop.port @f1_shimPortOut on %28 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %30 = dmaphop.port @f1_shimPortIn on %28 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %31 = dmaphop.create_hop %29 -> %13 -> !dmaphop.hop
        %32 = dmaphop.create_hop %14 -> %16 -> !dmaphop.hop
        %33 = dmaphop.create_hop %17 -> %19 -> !dmaphop.hop
        %34 = dmaphop.create_hop %20 -> %22 -> !dmaphop.hop
        %35 = dmaphop.create_path[%31, %32, %33, %34] {producers = [[@f1_shimPortIn]], consumers = [[@f1_consumer0, @f1_consumer1, @f1_consumer2, @f1_consumer3]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %11 into %35 consumer(%11, %11, %11, %11 at %13, %16, %19, %22) : tensor<64x256xi8> !dmaphop.path tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %35
        "routing.yield"() : () -> ()
      }
      %9 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_data %6, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %12 = dmaphop.tile{TILETYPE = "core", col = 2, row = 3} -> !dmaphop.tile
        %13 = dmaphop.port @f2_corePortIn0 on %12 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %14 = dmaphop.port @f2_corePortOut0 on %12 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.tile{TILETYPE = "core", col = 2, row = 4} -> !dmaphop.tile
        %16 = dmaphop.port @f2_corePortIn1 on %15 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %17 = dmaphop.port @f2_corePortOut1 on %15 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %18 = dmaphop.tile{TILETYPE = "core", col = 2, row = 5} -> !dmaphop.tile
        %19 = dmaphop.port @f2_corePortIn2 on %18 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %20 = dmaphop.port @f2_corePortOut2 on %18 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %21 = dmaphop.tile{TILETYPE = "core", col = 2, row = 6} -> !dmaphop.tile
        %22 = dmaphop.port @f2_corePortIn3 on %21 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %23 = dmaphop.port @f2_corePortOut3 on %21 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %24 = dmaphop.consumer @f2_consumer0 {dma_port = 1 : i64, from = @f2_corePortIn0}
        %25 = dmaphop.consumer @f2_consumer1 {dma_port = 1 : i64, from = @f2_corePortIn1}
        %26 = dmaphop.consumer @f2_consumer2 {dma_port = 1 : i64, from = @f2_corePortIn2}
        %27 = dmaphop.consumer @f2_consumer3 {dma_port = 1 : i64, from = @f2_corePortIn3}
        %28 = dmaphop.tile{TILETYPE = "shim", col = 2, row = 0} -> !dmaphop.tile
        %29 = dmaphop.port @f2_shimPortOut on %28 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %30 = dmaphop.port @f2_shimPortIn on %28 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %31 = dmaphop.create_hop %29 -> %13 -> !dmaphop.hop
        %32 = dmaphop.create_hop %14 -> %16 -> !dmaphop.hop
        %33 = dmaphop.create_hop %17 -> %19 -> !dmaphop.hop
        %34 = dmaphop.create_hop %20 -> %22 -> !dmaphop.hop
        %35 = dmaphop.create_path[%31, %32, %33, %34] {producers = [[@f2_shimPortIn]], consumers = [[@f2_consumer0, @f2_consumer1, @f2_consumer2, @f2_consumer3]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %11 into %35 consumer(%11, %11, %11, %11 at %13, %16, %19, %22) : tensor<64x256xi8> !dmaphop.path tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %35
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_data %6, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %12 = dmaphop.tile{TILETYPE = "core", col = 3, row = 3} -> !dmaphop.tile
        %13 = dmaphop.port @f3_corePortIn0 on %12 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %14 = dmaphop.port @f3_corePortOut0 on %12 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.tile{TILETYPE = "core", col = 3, row = 4} -> !dmaphop.tile
        %16 = dmaphop.port @f3_corePortIn1 on %15 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %17 = dmaphop.port @f3_corePortOut1 on %15 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %18 = dmaphop.tile{TILETYPE = "core", col = 3, row = 5} -> !dmaphop.tile
        %19 = dmaphop.port @f3_corePortIn2 on %18 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %20 = dmaphop.port @f3_corePortOut2 on %18 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %21 = dmaphop.tile{TILETYPE = "core", col = 3, row = 6} -> !dmaphop.tile
        %22 = dmaphop.port @f3_corePortIn3 on %21 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %23 = dmaphop.port @f3_corePortOut3 on %21 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %24 = dmaphop.consumer @f3_consumer0 {dma_port = 1 : i64, from = @f3_corePortIn0}
        %25 = dmaphop.consumer @f3_consumer1 {dma_port = 1 : i64, from = @f3_corePortIn1}
        %26 = dmaphop.consumer @f3_consumer2 {dma_port = 1 : i64, from = @f3_corePortIn2}
        %27 = dmaphop.consumer @f3_consumer3 {dma_port = 1 : i64, from = @f3_corePortIn3}
        %28 = dmaphop.tile{TILETYPE = "shim", col = 3, row = 0} -> !dmaphop.tile
        %29 = dmaphop.port @f3_shimPortOut on %28 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %30 = dmaphop.port @f3_shimPortIn on %28 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %31 = dmaphop.create_hop %29 -> %13 -> !dmaphop.hop
        %32 = dmaphop.create_hop %14 -> %16 -> !dmaphop.hop
        %33 = dmaphop.create_hop %17 -> %19 -> !dmaphop.hop
        %34 = dmaphop.create_hop %20 -> %22 -> !dmaphop.hop
        %35 = dmaphop.create_path[%31, %32, %33, %34] {producers = [[@f3_shimPortIn]], consumers = [[@f3_consumer0, @f3_consumer1, @f3_consumer2, @f3_consumer3]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %11 into %35 consumer(%11, %11, %11, %11 at %13, %16, %19, %22) : tensor<64x256xi8> !dmaphop.path tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %35
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %6 = routing.partitiontensor %1 : tensor<256x256xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4, slice_tiling = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 256, step = 256, rounds = 1, slice_tiling = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4>>>>
} -> tensor<256x256xi8>
      %7 = routing.partitiontensor %5 : tensor<256x256xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4, slice_tiling = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4, slice_tiling = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>>>>
} -> tensor<256x256xi8>
      %8 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %12 = routing.routingextract_data %6, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %13 = dmaphop.tile{TILETYPE = "core", col = 0, row = 3} -> !dmaphop.tile
        %14 = dmaphop.port @f4_corePortIn0 on %13 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.port @f4_corePortOut0 on %13 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %16 = dmaphop.tile{TILETYPE = "core", col = 1, row = 3} -> !dmaphop.tile
        %17 = dmaphop.port @f4_corePortIn1 on %16 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %18 = dmaphop.port @f4_corePortOut1 on %16 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %19 = dmaphop.tile{TILETYPE = "core", col = 2, row = 3} -> !dmaphop.tile
        %20 = dmaphop.port @f4_corePortIn2 on %19 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %21 = dmaphop.port @f4_corePortOut2 on %19 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %22 = dmaphop.tile{TILETYPE = "core", col = 3, row = 3} -> !dmaphop.tile
        %23 = dmaphop.port @f4_corePortIn3 on %22 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %24 = dmaphop.port @f4_corePortOut3 on %22 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %25 = dmaphop.consumer @f4_consumer0 {dma_port = 0 : i64, from = @f4_corePortIn0}
        %26 = dmaphop.consumer @f4_consumer1 {dma_port = 0 : i64, from = @f4_corePortIn1}
        %27 = dmaphop.consumer @f4_consumer2 {dma_port = 0 : i64, from = @f4_corePortIn2}
        %28 = dmaphop.consumer @f4_consumer3 {dma_port = 0 : i64, from = @f4_corePortIn3}
        %29 = dmaphop.tile{TILETYPE = "shim", col = 0, row = 0} -> !dmaphop.tile
        %30 = dmaphop.port @f4_shimPortOut on %29 { direction = "Out", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %31 = dmaphop.port @f4_shimPortIn on %29 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %32 = dmaphop.create_hop %30 -> %14 -> !dmaphop.hop
        %33 = dmaphop.create_hop %15 -> %17 -> !dmaphop.hop
        %34 = dmaphop.create_hop %18 -> %20 -> !dmaphop.hop
        %35 = dmaphop.create_hop %21 -> %23 -> !dmaphop.hop
        %36 = dmaphop.create_path[%32, %33, %34, %35] {producers = [[@f4_shimPortIn]], consumers = [[@f4_consumer0, @f4_consumer1, @f4_consumer2, @f4_consumer3]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %12 into %36 consumer(%12, %12, %12, %12 at %14, %17, %20, %23) : tensor<64x256xi8> !dmaphop.path tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %36
        %37 = routing.routingextract_data %7, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %38 = dmaphop.tile{TILETYPE = "core", col = 0, row = 3} -> !dmaphop.tile
        %39 = dmaphop.port @f5_corePortIn0 on %38 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %40 = dmaphop.port @f5_corePortOut0 on %38 { direction = "Out", direction_channel = 0, dmapktid = 1 : i32 } : !dmaphop.tile -> !dmaphop.port
        %41 = dmaphop.tile{TILETYPE = "core", col = 1, row = 3} -> !dmaphop.tile
        %42 = dmaphop.port @f5_corePortIn1 on %41 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %43 = dmaphop.port @f5_corePortOut1 on %41 { direction = "Out", direction_channel = 0, dmapktid = 2 : i32 } : !dmaphop.tile -> !dmaphop.port
        %44 = dmaphop.tile{TILETYPE = "core", col = 2, row = 3} -> !dmaphop.tile
        %45 = dmaphop.port @f5_corePortIn2 on %44 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %46 = dmaphop.port @f5_corePortOut2 on %44 { direction = "Out", direction_channel = 0, dmapktid = 3 : i32 } : !dmaphop.tile -> !dmaphop.port
        %47 = dmaphop.tile{TILETYPE = "core", col = 3, row = 3} -> !dmaphop.tile
        %48 = dmaphop.port @f5_corePortIn3 on %47 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %49 = dmaphop.port @f5_corePortOut3 on %47 { direction = "Out", direction_channel = 0, dmapktid = 4 : i32 } : !dmaphop.tile -> !dmaphop.port
        %50 = dmaphop.producer @f5_producer0 {dma_port = 0 : i64, tp = @f5_corePortOut0}
        %51 = dmaphop.producer @f5_producer1 {dma_port = 0 : i64, tp = @f5_corePortOut1}
        %52 = dmaphop.producer @f5_producer2 {dma_port = 0 : i64, tp = @f5_corePortOut2}
        %53 = dmaphop.producer @f5_producer3 {dma_port = 0 : i64, tp = @f5_corePortOut3}
        %54 = dmaphop.tile{TILETYPE = "shim", col = 3, row = 0} -> !dmaphop.tile
        %55 = dmaphop.port @f5_shimPortOut on %54 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %56 = dmaphop.port @f5_shimPortIn on %54 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %57 = dmaphop.create_hop %49 -> %56 -> !dmaphop.hop
        %58 = dmaphop.create_hop %46 -> %48 -> !dmaphop.hop
        %59 = dmaphop.create_hop %43 -> %45 -> !dmaphop.hop
        %60 = dmaphop.create_hop %40 -> %42 -> !dmaphop.hop
        %61 = dmaphop.create_path[%57, %58, %59, %60] {producers = [[@f5_producer0, @f5_producer1, @f5_producer2, @f5_producer3]], consumers = [[@f5_shimPortOut]], tee_points = [[]]} -> !dmaphop.path
        %extracted_slice = tensor.extract_slice %37[0, 0] [16, 256] [1, 1] {tag = "producer0"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_0 = tensor.extract_slice %37[16, 0] [16, 256] [1, 1] {tag = "producer1"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_1 = tensor.extract_slice %37[32, 0] [16, 256] [1, 1] {tag = "producer2"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_2 = tensor.extract_slice %37[48, 0] [16, 256] [1, 1] {tag = "producer3"} : tensor<64x256xi8> to tensor<16x256xi8>
        dmaphop.pull %37 from %61 producer(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2 at %39, %42, %45, %48) : tensor<64x256xi8> !dmaphop.path tensor<16x256xi8>, tensor<16x256xi8>, tensor<16x256xi8>, tensor<16x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %61
        "routing.yield"() : () -> ()
      }
      %9 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %12 = routing.routingextract_data %6, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %13 = dmaphop.tile{TILETYPE = "core", col = 0, row = 4} -> !dmaphop.tile
        %14 = dmaphop.port @f6_corePortIn0 on %13 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.port @f6_corePortOut0 on %13 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %16 = dmaphop.tile{TILETYPE = "core", col = 1, row = 4} -> !dmaphop.tile
        %17 = dmaphop.port @f6_corePortIn1 on %16 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %18 = dmaphop.port @f6_corePortOut1 on %16 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %19 = dmaphop.tile{TILETYPE = "core", col = 2, row = 4} -> !dmaphop.tile
        %20 = dmaphop.port @f6_corePortIn2 on %19 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %21 = dmaphop.port @f6_corePortOut2 on %19 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %22 = dmaphop.tile{TILETYPE = "core", col = 3, row = 4} -> !dmaphop.tile
        %23 = dmaphop.port @f6_corePortIn3 on %22 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %24 = dmaphop.port @f6_corePortOut3 on %22 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %25 = dmaphop.consumer @f6_consumer0 {dma_port = 0 : i64, from = @f6_corePortIn0}
        %26 = dmaphop.consumer @f6_consumer1 {dma_port = 0 : i64, from = @f6_corePortIn1}
        %27 = dmaphop.consumer @f6_consumer2 {dma_port = 0 : i64, from = @f6_corePortIn2}
        %28 = dmaphop.consumer @f6_consumer3 {dma_port = 0 : i64, from = @f6_corePortIn3}
        %29 = dmaphop.tile{TILETYPE = "shim", col = 1, row = 0} -> !dmaphop.tile
        %30 = dmaphop.port @f6_shimPortOut on %29 { direction = "Out", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %31 = dmaphop.port @f6_shimPortIn on %29 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %32 = dmaphop.create_hop %30 -> %14 -> !dmaphop.hop
        %33 = dmaphop.create_hop %15 -> %17 -> !dmaphop.hop
        %34 = dmaphop.create_hop %18 -> %20 -> !dmaphop.hop
        %35 = dmaphop.create_hop %21 -> %23 -> !dmaphop.hop
        %36 = dmaphop.create_path[%32, %33, %34, %35] {producers = [[@f6_shimPortIn]], consumers = [[@f6_consumer0, @f6_consumer1, @f6_consumer2, @f6_consumer3]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %12 into %36 consumer(%12, %12, %12, %12 at %14, %17, %20, %23) : tensor<64x256xi8> !dmaphop.path tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %36
        %37 = routing.routingextract_data %7, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %38 = dmaphop.tile{TILETYPE = "core", col = 0, row = 4} -> !dmaphop.tile
        %39 = dmaphop.port @f7_corePortIn0 on %38 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %40 = dmaphop.port @f7_corePortOut0 on %38 { direction = "Out", direction_channel = 0, dmapktid = 5 : i32 } : !dmaphop.tile -> !dmaphop.port
        %41 = dmaphop.tile{TILETYPE = "core", col = 1, row = 4} -> !dmaphop.tile
        %42 = dmaphop.port @f7_corePortIn1 on %41 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %43 = dmaphop.port @f7_corePortOut1 on %41 { direction = "Out", direction_channel = 0, dmapktid = 6 : i32 } : !dmaphop.tile -> !dmaphop.port
        %44 = dmaphop.tile{TILETYPE = "core", col = 2, row = 4} -> !dmaphop.tile
        %45 = dmaphop.port @f7_corePortIn2 on %44 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %46 = dmaphop.port @f7_corePortOut2 on %44 { direction = "Out", direction_channel = 0, dmapktid = 7 : i32 } : !dmaphop.tile -> !dmaphop.port
        %47 = dmaphop.tile{TILETYPE = "core", col = 3, row = 4} -> !dmaphop.tile
        %48 = dmaphop.port @f7_corePortIn3 on %47 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %49 = dmaphop.port @f7_corePortOut3 on %47 { direction = "Out", direction_channel = 0, dmapktid = 8 : i32 } : !dmaphop.tile -> !dmaphop.port
        %50 = dmaphop.producer @f7_producer0 {dma_port = 0 : i64, tp = @f7_corePortOut0}
        %51 = dmaphop.producer @f7_producer1 {dma_port = 0 : i64, tp = @f7_corePortOut1}
        %52 = dmaphop.producer @f7_producer2 {dma_port = 0 : i64, tp = @f7_corePortOut2}
        %53 = dmaphop.producer @f7_producer3 {dma_port = 0 : i64, tp = @f7_corePortOut3}
        %54 = dmaphop.tile{TILETYPE = "shim", col = 3, row = 0} -> !dmaphop.tile
        %55 = dmaphop.port @f7_shimPortOut on %54 { direction = "Out", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %56 = dmaphop.port @f7_shimPortIn on %54 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %57 = dmaphop.create_hop %49 -> %56 -> !dmaphop.hop
        %58 = dmaphop.create_hop %46 -> %48 -> !dmaphop.hop
        %59 = dmaphop.create_hop %43 -> %45 -> !dmaphop.hop
        %60 = dmaphop.create_hop %40 -> %42 -> !dmaphop.hop
        %61 = dmaphop.create_path[%57, %58, %59, %60] {producers = [[@f7_producer0, @f7_producer1, @f7_producer2, @f7_producer3]], consumers = [[@f7_shimPortOut]], tee_points = [[]]} -> !dmaphop.path
        %extracted_slice = tensor.extract_slice %37[0, 0] [16, 256] [1, 1] {tag = "producer0"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_0 = tensor.extract_slice %37[16, 0] [16, 256] [1, 1] {tag = "producer1"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_1 = tensor.extract_slice %37[32, 0] [16, 256] [1, 1] {tag = "producer2"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_2 = tensor.extract_slice %37[48, 0] [16, 256] [1, 1] {tag = "producer3"} : tensor<64x256xi8> to tensor<16x256xi8>
        dmaphop.pull %37 from %61 producer(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2 at %39, %42, %45, %48) : tensor<64x256xi8> !dmaphop.path tensor<16x256xi8>, tensor<16x256xi8>, tensor<16x256xi8>, tensor<16x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %61
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %12 = routing.routingextract_data %6, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %13 = dmaphop.tile{TILETYPE = "core", col = 0, row = 5} -> !dmaphop.tile
        %14 = dmaphop.port @f8_corePortIn0 on %13 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.port @f8_corePortOut0 on %13 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %16 = dmaphop.tile{TILETYPE = "core", col = 1, row = 5} -> !dmaphop.tile
        %17 = dmaphop.port @f8_corePortIn1 on %16 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %18 = dmaphop.port @f8_corePortOut1 on %16 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %19 = dmaphop.tile{TILETYPE = "core", col = 2, row = 5} -> !dmaphop.tile
        %20 = dmaphop.port @f8_corePortIn2 on %19 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %21 = dmaphop.port @f8_corePortOut2 on %19 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %22 = dmaphop.tile{TILETYPE = "core", col = 3, row = 5} -> !dmaphop.tile
        %23 = dmaphop.port @f8_corePortIn3 on %22 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %24 = dmaphop.port @f8_corePortOut3 on %22 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %25 = dmaphop.consumer @f8_consumer0 {dma_port = 0 : i64, from = @f8_corePortIn0}
        %26 = dmaphop.consumer @f8_consumer1 {dma_port = 0 : i64, from = @f8_corePortIn1}
        %27 = dmaphop.consumer @f8_consumer2 {dma_port = 0 : i64, from = @f8_corePortIn2}
        %28 = dmaphop.consumer @f8_consumer3 {dma_port = 0 : i64, from = @f8_corePortIn3}
        %29 = dmaphop.tile{TILETYPE = "shim", col = 2, row = 0} -> !dmaphop.tile
        %30 = dmaphop.port @f8_shimPortOut on %29 { direction = "Out", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %31 = dmaphop.port @f8_shimPortIn on %29 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %32 = dmaphop.create_hop %30 -> %14 -> !dmaphop.hop
        %33 = dmaphop.create_hop %15 -> %17 -> !dmaphop.hop
        %34 = dmaphop.create_hop %18 -> %20 -> !dmaphop.hop
        %35 = dmaphop.create_hop %21 -> %23 -> !dmaphop.hop
        %36 = dmaphop.create_path[%32, %33, %34, %35] {producers = [[@f8_shimPortIn]], consumers = [[@f8_consumer0, @f8_consumer1, @f8_consumer2, @f8_consumer3]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %12 into %36 consumer(%12, %12, %12, %12 at %14, %17, %20, %23) : tensor<64x256xi8> !dmaphop.path tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %36
        %37 = routing.routingextract_data %7, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %38 = dmaphop.tile{TILETYPE = "core", col = 0, row = 5} -> !dmaphop.tile
        %39 = dmaphop.port @f9_corePortIn0 on %38 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %40 = dmaphop.port @f9_corePortOut0 on %38 { direction = "Out", direction_channel = 0, dmapktid = 9 : i32 } : !dmaphop.tile -> !dmaphop.port
        %41 = dmaphop.tile{TILETYPE = "core", col = 1, row = 5} -> !dmaphop.tile
        %42 = dmaphop.port @f9_corePortIn1 on %41 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %43 = dmaphop.port @f9_corePortOut1 on %41 { direction = "Out", direction_channel = 0, dmapktid = 10 : i32 } : !dmaphop.tile -> !dmaphop.port
        %44 = dmaphop.tile{TILETYPE = "core", col = 2, row = 5} -> !dmaphop.tile
        %45 = dmaphop.port @f9_corePortIn2 on %44 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %46 = dmaphop.port @f9_corePortOut2 on %44 { direction = "Out", direction_channel = 0, dmapktid = 11 : i32 } : !dmaphop.tile -> !dmaphop.port
        %47 = dmaphop.tile{TILETYPE = "core", col = 3, row = 5} -> !dmaphop.tile
        %48 = dmaphop.port @f9_corePortIn3 on %47 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %49 = dmaphop.port @f9_corePortOut3 on %47 { direction = "Out", direction_channel = 0, dmapktid = 12 : i32 } : !dmaphop.tile -> !dmaphop.port
        %50 = dmaphop.producer @f9_producer0 {dma_port = 0 : i64, tp = @f9_corePortOut0}
        %51 = dmaphop.producer @f9_producer1 {dma_port = 0 : i64, tp = @f9_corePortOut1}
        %52 = dmaphop.producer @f9_producer2 {dma_port = 0 : i64, tp = @f9_corePortOut2}
        %53 = dmaphop.producer @f9_producer3 {dma_port = 0 : i64, tp = @f9_corePortOut3}
        %54 = dmaphop.tile{TILETYPE = "shim", col = 2, row = 0} -> !dmaphop.tile
        %55 = dmaphop.port @f9_shimPortOut on %54 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %56 = dmaphop.port @f9_shimPortIn on %54 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %57 = dmaphop.create_hop %49 -> %56 -> !dmaphop.hop
        %58 = dmaphop.create_hop %46 -> %48 -> !dmaphop.hop
        %59 = dmaphop.create_hop %43 -> %45 -> !dmaphop.hop
        %60 = dmaphop.create_hop %40 -> %42 -> !dmaphop.hop
        %61 = dmaphop.create_path[%57, %58, %59, %60] {producers = [[@f9_producer0, @f9_producer1, @f9_producer2, @f9_producer3]], consumers = [[@f9_shimPortOut]], tee_points = [[]]} -> !dmaphop.path
        %extracted_slice = tensor.extract_slice %37[0, 0] [16, 256] [1, 1] {tag = "producer0"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_0 = tensor.extract_slice %37[16, 0] [16, 256] [1, 1] {tag = "producer1"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_1 = tensor.extract_slice %37[32, 0] [16, 256] [1, 1] {tag = "producer2"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_2 = tensor.extract_slice %37[48, 0] [16, 256] [1, 1] {tag = "producer3"} : tensor<64x256xi8> to tensor<16x256xi8>
        dmaphop.pull %37 from %61 producer(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2 at %39, %42, %45, %48) : tensor<64x256xi8> !dmaphop.path tensor<16x256xi8>, tensor<16x256xi8>, tensor<16x256xi8>, tensor<16x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %61
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %12 = routing.routingextract_data %6, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %13 = dmaphop.tile{TILETYPE = "core", col = 0, row = 6} -> !dmaphop.tile
        %14 = dmaphop.port @f10_corePortIn0 on %13 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.port @f10_corePortOut0 on %13 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %16 = dmaphop.tile{TILETYPE = "core", col = 1, row = 6} -> !dmaphop.tile
        %17 = dmaphop.port @f10_corePortIn1 on %16 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %18 = dmaphop.port @f10_corePortOut1 on %16 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %19 = dmaphop.tile{TILETYPE = "core", col = 2, row = 6} -> !dmaphop.tile
        %20 = dmaphop.port @f10_corePortIn2 on %19 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %21 = dmaphop.port @f10_corePortOut2 on %19 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %22 = dmaphop.tile{TILETYPE = "core", col = 3, row = 6} -> !dmaphop.tile
        %23 = dmaphop.port @f10_corePortIn3 on %22 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %24 = dmaphop.port @f10_corePortOut3 on %22 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %25 = dmaphop.consumer @f10_consumer0 {dma_port = 0 : i64, from = @f10_corePortIn0}
        %26 = dmaphop.consumer @f10_consumer1 {dma_port = 0 : i64, from = @f10_corePortIn1}
        %27 = dmaphop.consumer @f10_consumer2 {dma_port = 0 : i64, from = @f10_corePortIn2}
        %28 = dmaphop.consumer @f10_consumer3 {dma_port = 0 : i64, from = @f10_corePortIn3}
        %29 = dmaphop.tile{TILETYPE = "shim", col = 3, row = 0} -> !dmaphop.tile
        %30 = dmaphop.port @f10_shimPortOut on %29 { direction = "Out", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %31 = dmaphop.port @f10_shimPortIn on %29 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %32 = dmaphop.create_hop %30 -> %14 -> !dmaphop.hop
        %33 = dmaphop.create_hop %15 -> %17 -> !dmaphop.hop
        %34 = dmaphop.create_hop %18 -> %20 -> !dmaphop.hop
        %35 = dmaphop.create_hop %21 -> %23 -> !dmaphop.hop
        %36 = dmaphop.create_path[%32, %33, %34, %35] {producers = [[@f10_shimPortIn]], consumers = [[@f10_consumer0, @f10_consumer1, @f10_consumer2, @f10_consumer3]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %12 into %36 consumer(%12, %12, %12, %12 at %14, %17, %20, %23) : tensor<64x256xi8> !dmaphop.path tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8>, tensor<64x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %36
        %37 = routing.routingextract_data %7, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %38 = dmaphop.tile{TILETYPE = "core", col = 0, row = 6} -> !dmaphop.tile
        %39 = dmaphop.port @f11_corePortIn0 on %38 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %40 = dmaphop.port @f11_corePortOut0 on %38 { direction = "Out", direction_channel = 0, dmapktid = 13 : i32 } : !dmaphop.tile -> !dmaphop.port
        %41 = dmaphop.tile{TILETYPE = "core", col = 1, row = 6} -> !dmaphop.tile
        %42 = dmaphop.port @f11_corePortIn1 on %41 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %43 = dmaphop.port @f11_corePortOut1 on %41 { direction = "Out", direction_channel = 0, dmapktid = 14 : i32 } : !dmaphop.tile -> !dmaphop.port
        %44 = dmaphop.tile{TILETYPE = "core", col = 2, row = 6} -> !dmaphop.tile
        %45 = dmaphop.port @f11_corePortIn2 on %44 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %46 = dmaphop.port @f11_corePortOut2 on %44 { direction = "Out", direction_channel = 0, dmapktid = 15 : i32 } : !dmaphop.tile -> !dmaphop.port
        %47 = dmaphop.tile{TILETYPE = "core", col = 3, row = 6} -> !dmaphop.tile
        %48 = dmaphop.port @f11_corePortIn3 on %47 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %49 = dmaphop.port @f11_corePortOut3 on %47 { direction = "Out", direction_channel = 0, dmapktid = 16 : i32 } : !dmaphop.tile -> !dmaphop.port
        %50 = dmaphop.producer @f11_producer0 {dma_port = 0 : i64, tp = @f11_corePortOut0}
        %51 = dmaphop.producer @f11_producer1 {dma_port = 0 : i64, tp = @f11_corePortOut1}
        %52 = dmaphop.producer @f11_producer2 {dma_port = 0 : i64, tp = @f11_corePortOut2}
        %53 = dmaphop.producer @f11_producer3 {dma_port = 0 : i64, tp = @f11_corePortOut3}
        %54 = dmaphop.tile{TILETYPE = "shim", col = 2, row = 0} -> !dmaphop.tile
        %55 = dmaphop.port @f11_shimPortOut on %54 { direction = "Out", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %56 = dmaphop.port @f11_shimPortIn on %54 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %57 = dmaphop.create_hop %49 -> %56 -> !dmaphop.hop
        %58 = dmaphop.create_hop %46 -> %48 -> !dmaphop.hop
        %59 = dmaphop.create_hop %43 -> %45 -> !dmaphop.hop
        %60 = dmaphop.create_hop %40 -> %42 -> !dmaphop.hop
        %61 = dmaphop.create_path[%57, %58, %59, %60] {producers = [[@f11_producer0, @f11_producer1, @f11_producer2, @f11_producer3]], consumers = [[@f11_shimPortOut]], tee_points = [[]]} -> !dmaphop.path
        %extracted_slice = tensor.extract_slice %37[0, 0] [16, 256] [1, 1] {tag = "producer0"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_0 = tensor.extract_slice %37[16, 0] [16, 256] [1, 1] {tag = "producer1"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_1 = tensor.extract_slice %37[32, 0] [16, 256] [1, 1] {tag = "producer2"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_2 = tensor.extract_slice %37[48, 0] [16, 256] [1, 1] {tag = "producer3"} : tensor<64x256xi8> to tensor<16x256xi8>
        dmaphop.pull %37 from %61 producer(%extracted_slice, %extracted_slice_0, %extracted_slice_1, %extracted_slice_2 at %39, %42, %45, %48) : tensor<64x256xi8> !dmaphop.path tensor<16x256xi8>, tensor<16x256xi8>, tensor<16x256xi8>, tensor<16x256xi8> !dmaphop.port, !dmaphop.port, !dmaphop.port, !dmaphop.port
        dmaphop.sync %61
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
