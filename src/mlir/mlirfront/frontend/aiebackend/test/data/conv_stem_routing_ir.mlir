module attributes {routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) attributes {routing.kernel_tiling = #routing.kernel_tiling<out_rounds = 16>} {
    %0 = routing.routingcreatehwmesh row = 4, col = 4 partition = 3, 6, 0, 6 -> i32
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
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %9 = arith.index_cast %arg3 : index to i32
        %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %9 : i32) -> i32{
        ^bb0(%arg4: i32):
          %11 = routing.routingextract_tiles %7, %arg4 : i32, i32 -> i32
          %12 = routing.routingextract_data %8, %arg4 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
          %13 = routing.routingcreatehwiowithtarget targettilelist = %11 : i32 {direction = "input", iotype = "mem2"} -> i32
          %14 = routing.routingmovedatabyio tensordata = %12, hwiowithtarget = %13 : tensor<16x7x7x4xi8>, i32 -> i32
          "routing.yield"() : () -> ()
        }
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
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %10 = arith.index_cast %arg3 : index to i32
        %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %10 : i32) -> i32{
        ^bb0(%arg4: i32):
          %12 = routing.routingextract_tiles %7, %arg4 : i32, i32 -> i32
          %13 = routing.routingextract_data %8, %arg4 : tensor<230x230x4xi8>, i32 -> tensor<61x230x4xi8>
          %14 = routing.routingcreatehwiowithtarget targettilelist = %12 : i32 {direction = "input", iotype = "mem2"} -> i32
          %15 = routing.routingmovedatabyio tensordata = %13, hwiowithtarget = %14 : tensor<61x230x4xi8>, i32 -> i32
          %16 = routing.routingextract_data %9, %arg4 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
          %17 = routing.routingroutinggatherout tilelist = %12, tensordata = %16 : i32, tensor<28x112x16xi8> -> tensor<28x112x16xi8>
          %18 = routing.routingcreatehwiowithtarget targettilelist = %12 : i32 {direction = "output", iotype = "mem2"} -> i32
          %19 = routing.routingmovedatabyio tensordata = %17, hwiowithtarget = %18 : tensor<28x112x16xi8>, i32 -> i32
          "routing.yield"() : () -> ()
        }
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
