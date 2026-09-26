module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.fullconnect_auto = 1 : i64, routing.kernel_config_offload = 1 : i64, routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}} {
  func.func @main(%arg0: memref<256x256xi8>, %arg1: memref<256x256xi8>, %arg2: memref<256x256xi8>) {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = routing.routingcreatehwmesh row = 4, col = 4 partition = 0, 3, 0, 6 -> i32
    %1 = bufferization.to_tensor %arg0 : memref<256x256xi8>
    %2 = routing.routingcreatescheduletensor %1 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    %3 = bufferization.to_tensor %arg1 : memref<256x256xi8>
    %4 = routing.routingcreatescheduletensor %3 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    %5 = bufferization.to_tensor %arg2 : memref<256x256xi8>
    %6 = routing.routingcreatescheduletensor %5 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 4, splitaxis = "col" : i32 -> i32
      %8 = routing.partitiontensor %4 : tensor<256x256xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "col", replicateOn = "row", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4, slice_tiling = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 256, step = 256, rounds = 1, slice_tiling = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4>>>>
} -> tensor<256x256xi8>
      %9 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %12 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 4, splitaxis = "row" : i32 -> i32
      %8 = routing.partitiontensor %2 : tensor<256x256xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4, slice_tiling = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 256, step = 256, rounds = 1, slice_tiling = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4>>>>
} -> tensor<256x256xi8>
      %9 = routing.partitiontensor %6 : tensor<256x256xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4, slice_tiling = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 256, total = 256, slice = 64, step = 64, rounds = 4, slice_tiling = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>>>>
} -> tensor<256x256xi8>
      %10 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<64x256xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<64x256xi8> -> tensor<64x256xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<64x256xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<64x256xi8> -> tensor<64x256xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %12 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<64x256xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<64x256xi8> -> tensor<64x256xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %13 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<64x256xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<64x256xi8> -> tensor<64x256xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
