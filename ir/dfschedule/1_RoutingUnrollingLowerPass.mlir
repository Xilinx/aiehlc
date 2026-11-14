module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 1 : i64, routing.control_plan_dedicated_shim = 1 : i64, routing.control_plan_exclusive = 1 : i64, routing.control_plan_group_reg_write = 1 : i64, routing.control_plan_mm2s_ch = 0 : i64, routing.control_plan_op_control_packet = 1 : i64, routing.control_plan_s2mm_ch = 0 : i64, routing.control_plan_shim_bd_ctrl = 1 : i64, routing.control_plan_shim_col = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 1 : i64, routing.output_tile_major = 1 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) attributes {routing.kernel_tiling = #routing.kernel_tiling<out_rounds = 16>} {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = routing.routingcreatehwmesh row = 4, col = 4 partition = 2, 6, 0, 6 -> i32
    %1 = bufferization.to_tensor %arg0 : memref<230x230x4xi8>
    %2 = routing.routingcreatescheduletensor %1 : tensor<230x230x4xi8> shape = [230, 230, 4], dim = 3 {halo = #routing.halo<slice = 61, step = 56, split_dim = 0, w_slice = 61, w_step = 56, w_rounds = 4, ow_t = 28, row_pitch = 920, l2_slice = 19, l2_step = 14, l2_rounds = 4, k_slice = 244, k_step = 224, k_rounds = 4, buf_size = 4636>, layout_transform = "dma_shuffle", pp_depth = 2 : i64} -> tensor<230x230x4xi8>
    %3 = bufferization.to_tensor %arg1 : memref<64x7x7x4xi8>
    %4 = routing.routingcreatescheduletensor %3 : tensor<64x7x7x4xi8> shape = [64, 7, 7, 4], dim = 4 {layout_transform = "dma_shuffle", pp_depth = 2 : i64} -> tensor<64x7x7x4xi8>
    %5 = bufferization.to_tensor %arg2 : memref<112x112x64xi8>
    %6 = routing.routingcreatescheduletensor %5 : tensor<112x112x64xi8> shape = [112, 112, 64], dim = 3 {pp_depth = 2 : i64} -> tensor<112x112x64xi8>
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 4, splitaxis = "col" : i32 -> i32
      %8 = routing.partitiontensor %4 : tensor<64x7x7x4xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "col", replicateOn = "row", singleTileOwner = "">
} -> tensor<64x7x7x4xi8>
      %9 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<16x7x7x4xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<16x7x7x4xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<16x7x7x4xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %12 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<16x7x7x4xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 4, splitaxis = "row" : i32 -> i32
      %8 = routing.partitiontensor %2 : tensor<230x230x4xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>
} -> tensor<230x230x4xi8>
      %9 = routing.partitiontensor %6 : tensor<112x112x64xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>
} -> tensor<112x112x64xi8>
      %10 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<230x230x4xi8>, i32 -> tensor<61x230x4xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<61x230x4xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<28x112x16xi8> -> tensor<28x112x16xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<28x112x16xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<230x230x4xi8>, i32 -> tensor<61x230x4xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<61x230x4xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<28x112x16xi8> -> tensor<28x112x16xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<28x112x16xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %12 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<230x230x4xi8>, i32 -> tensor<61x230x4xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<61x230x4xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<28x112x16xi8> -> tensor<28x112x16xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<28x112x16xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %13 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<230x230x4xi8>, i32 -> tensor<61x230x4xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<61x230x4xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<28x112x16xi8> -> tensor<28x112x16xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<28x112x16xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
