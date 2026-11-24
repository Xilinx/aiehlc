module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) attributes {routing.kernel_tiling = #routing.kernel_tiling<out_rounds = 16>} {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg0 : memref<230x230x4xi8>
    %1 = dfscheblueprint.declare_data %0 : tensor<230x230x4xi8> {routing.halo = #routing.halo<slice = 61, step = 56, split_dim = 0, w_slice = 61, w_step = 56, w_rounds = 4, ow_t = 28, row_pitch = 920, l2_slice = 19, l2_step = 14, l2_rounds = 4, k_slice = 244, k_step = 224, k_rounds = 4, buf_size = 4636>, routing.kernel_block = #routing.kernel_block<tensor_shape = [230, 230, 4], shape = [19, 61, 4], rounds_i = 4, rounds_j = 4, step_i = [14, 0, 0], step_j = [0, 56, 0], resident = false, tile_r = 0, tile_c = 0>, routing.layout_transform = "dma_shuffle", routing.pp_depth = 2 : i64} -> tensor<230x230x4xi8>
    %2 = bufferization.to_tensor %arg1 : memref<64x7x7x4xi8>
    %3 = dfscheblueprint.declare_data %2 : tensor<64x7x7x4xi8> {routing.kernel_block = #routing.kernel_block<tensor_shape = [64, 7, 7, 4], shape = [16, 7, 7, 4], rounds_i = 1, rounds_j = 1, step_i = [0, 0, 0, 0], step_j = [0, 0, 0, 0], resident = false, tile_r = 0, tile_c = 0>, routing.layout_transform = "dma_shuffle", routing.pp_depth = 2 : i64, routing.window_bytes = 3136 : i64} -> tensor<64x7x7x4xi8>
    %4 = bufferization.to_tensor %arg2 : memref<112x112x64xi8>
    %5 = dfscheblueprint.declare_data %4 : tensor<112x112x64xi8> {routing.kernel_block = #routing.kernel_block<tensor_shape = [112, 112, 64], shape = [7, 28, 16], rounds_i = 4, rounds_j = 4, step_i = [7, 0, 0], step_j = [0, 28, 0], resident = false, tile_r = 0, tile_c = 0>, routing.pp_depth = 2 : i64, routing.window_bytes = 3136 : i64} -> tensor<112x112x64xi8>
    scf.execute_region {
      %6 = routing.partitiontensor %3 : tensor<64x7x7x4xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "col", replicateOn = "row", singleTileOwner = "">
} -> tensor<64x7x7x4xi8>
      %7 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_0{tiles = [[0, 0]]}
        dfscheblueprint.tile_group @group_dst_0{tiles = [[0, 3], [0, 4], [0, 5], [0, 6]]}
        %extracted_slice = tensor.extract_slice %6[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] {tag = "partitionslice0"} : tensor<64x7x7x4xi8> to tensor<16x7x7x4xi8>
        %11 = dfscheblueprint.data_slice @consumer_slice_0_0 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_0_1 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_0_2 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_0_3 wrap %extracted_slice : tensor<16x7x7x4xi8>
        dfscheblueprint.flowconfig @flow_src_0 {
          target = @group_src_0,
          view = %extracted_slice : tensor<16x7x7x4xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,type = "shim"
          ,data_id = 0 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_0 {
          target = @group_dst_0,
          view = %extracted_slice : tensor<16x7x7x4xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,slice_symbols = [@consumer_slice_0_0, @consumer_slice_0_1, @consumer_slice_0_2, @consumer_slice_0_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_0 {
          type = "one_to_many",
          from = @flow_src_0,
          to = @flow_dst_0
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 0 : i32
        }
        "routing.yield"() : () -> ()
      }
      %8 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_1{tiles = [[1, 0]]}
        dfscheblueprint.tile_group @group_dst_1{tiles = [[1, 3], [1, 4], [1, 5], [1, 6]]}
        %extracted_slice = tensor.extract_slice %6[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] {tag = "partitionslice1"} : tensor<64x7x7x4xi8> to tensor<16x7x7x4xi8>
        %11 = dfscheblueprint.data_slice @consumer_slice_1_0 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_1_1 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_1_2 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_1_3 wrap %extracted_slice : tensor<16x7x7x4xi8>
        dfscheblueprint.flowconfig @flow_src_1 {
          target = @group_src_1,
          view = %extracted_slice : tensor<16x7x7x4xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,type = "shim"
          ,data_id = 0 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_1 {
          target = @group_dst_1,
          view = %extracted_slice : tensor<16x7x7x4xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,slice_symbols = [@consumer_slice_1_0, @consumer_slice_1_1, @consumer_slice_1_2, @consumer_slice_1_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_1 {
          type = "one_to_many",
          from = @flow_src_1,
          to = @flow_dst_1
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 1 : i32
        }
        "routing.yield"() : () -> ()
      }
      %9 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_2{tiles = [[2, 0]]}
        dfscheblueprint.tile_group @group_dst_2{tiles = [[2, 3], [2, 4], [2, 5], [2, 6]]}
        %extracted_slice = tensor.extract_slice %6[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] {tag = "partitionslice2"} : tensor<64x7x7x4xi8> to tensor<16x7x7x4xi8>
        %11 = dfscheblueprint.data_slice @consumer_slice_2_0 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_2_1 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_2_2 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_2_3 wrap %extracted_slice : tensor<16x7x7x4xi8>
        dfscheblueprint.flowconfig @flow_src_2 {
          target = @group_src_2,
          view = %extracted_slice : tensor<16x7x7x4xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,type = "shim"
          ,data_id = 0 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_2 {
          target = @group_dst_2,
          view = %extracted_slice : tensor<16x7x7x4xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,slice_symbols = [@consumer_slice_2_0, @consumer_slice_2_1, @consumer_slice_2_2, @consumer_slice_2_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_2 {
          type = "one_to_many",
          from = @flow_src_2,
          to = @flow_dst_2
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 2 : i32
        }
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_3{tiles = [[3, 0]]}
        dfscheblueprint.tile_group @group_dst_3{tiles = [[3, 3], [3, 4], [3, 5], [3, 6]]}
        %extracted_slice = tensor.extract_slice %6[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] {tag = "partitionslice3"} : tensor<64x7x7x4xi8> to tensor<16x7x7x4xi8>
        %11 = dfscheblueprint.data_slice @consumer_slice_3_0 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_3_1 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_3_2 wrap %extracted_slice : tensor<16x7x7x4xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_3_3 wrap %extracted_slice : tensor<16x7x7x4xi8>
        dfscheblueprint.flowconfig @flow_src_3 {
          target = @group_src_3,
          view = %extracted_slice : tensor<16x7x7x4xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,type = "shim"
          ,data_id = 0 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_3 {
          target = @group_dst_3,
          view = %extracted_slice : tensor<16x7x7x4xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,slice_symbols = [@consumer_slice_3_0, @consumer_slice_3_1, @consumer_slice_3_2, @consumer_slice_3_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_3 {
          type = "one_to_many",
          from = @flow_src_3,
          to = @flow_dst_3
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 3 : i32
        }
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %6 = routing.partitiontensor %1 : tensor<230x230x4xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>
} -> tensor<230x230x4xi8>
      %7 = routing.partitiontensor %5 : tensor<112x112x64xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>
} -> tensor<112x112x64xi8>
      %8 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_5{tiles = [[0, 3], [1, 3], [2, 3], [3, 3]]}
        dfscheblueprint.tile_group @group_dst_5{tiles = [[3, 0]]}
        dfscheblueprint.tile_group @group_src_4{tiles = [[0, 0]]}
        dfscheblueprint.tile_group @group_dst_4{tiles = [[0, 3], [1, 3], [2, 3], [3, 3]]}
        %extracted_slice = tensor.extract_slice %6[0, 0, 0] [61, 230, 4] [1, 1, 1] {tag = "partitionslice0", tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : tensor<230x230x4xi8> to tensor<61x230x4xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_4_0 wrap %extracted_slice : tensor<61x230x4xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_4_1 wrap %extracted_slice : tensor<61x230x4xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_4_2 wrap %extracted_slice : tensor<61x230x4xi8>
        %15 = dfscheblueprint.data_slice @consumer_slice_4_3 wrap %extracted_slice : tensor<61x230x4xi8>
        dfscheblueprint.flowconfig @flow_src_4 {
          target = @group_src_4,
          view = %extracted_slice : tensor<61x230x4xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = MM2S>
          ,type = "shim"
          ,data_id = 1 : i32
          ,shim_dim_strides = [4 : i32, 920 : i32, 224 : i32]
          ,shim_dim_wraps = [61 : i32, 19 : i32, 4 : i32]
        }
        dfscheblueprint.flowconfig @flow_dst_4 {
          target = @group_dst_4,
          view = %extracted_slice : tensor<61x230x4xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,slice_symbols = [@consumer_slice_4_0, @consumer_slice_4_1, @consumer_slice_4_2, @consumer_slice_4_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_4 {
          type = "one_to_many",
          from = @flow_src_4,
          to = @flow_dst_4
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 4 : i32
        }
        %extracted_slice_0 = tensor.extract_slice %7[0, 0, 0] [28, 112, 16] [1, 1, 1] {tag = "partitionslice0", tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : tensor<112x112x64xi8> to tensor<28x112x16xi8>
        %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer0"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer1"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer2"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer3"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %16 = dfscheblueprint.data_slice @producer_slice_5_0 wrap %extracted_slice_1 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %17 = dfscheblueprint.data_slice @producer_slice_5_1 wrap %extracted_slice_2 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %18 = dfscheblueprint.data_slice @producer_slice_5_2 wrap %extracted_slice_3 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %19 = dfscheblueprint.data_slice @producer_slice_5_3 wrap %extracted_slice_4 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        dfscheblueprint.flowconfig @flow_src_5 {
          target = @group_src_5,
          view = %extracted_slice_0 : tensor<28x112x16xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,slice_symbols = [@producer_slice_5_0, @producer_slice_5_1, @producer_slice_5_2, @producer_slice_5_3]
          ,type = "core"
          ,pp_depth = 2 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_5 {
          target = @group_dst_5,
          view = %extracted_slice_0 : tensor<28x112x16xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,type = "shim"
          ,data_id = 2 : i32
          ,pp_depth = 2 : i32
        }
        dfscheblueprint.flow_transfer @transfer_5 {
          type = "many_to_one",
          from = @flow_src_5,
          to = @flow_dst_5
          ,ordering = "sequential"
          ,base_packet_id = 1 : i32,
          flow_index = 5 : i32
        }
        "routing.yield"() : () -> ()
      }
      %9 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_7{tiles = [[0, 4], [1, 4], [2, 4], [3, 4]]}
        dfscheblueprint.tile_group @group_dst_7{tiles = [[3, 0]]}
        dfscheblueprint.tile_group @group_src_6{tiles = [[1, 0]]}
        dfscheblueprint.tile_group @group_dst_6{tiles = [[0, 4], [1, 4], [2, 4], [3, 4]]}
        %extracted_slice = tensor.extract_slice %6[56, 0, 0] [61, 230, 4] [1, 1, 1] {tag = "partitionslice1", tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : tensor<230x230x4xi8> to tensor<61x230x4xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_6_0 wrap %extracted_slice : tensor<61x230x4xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_6_1 wrap %extracted_slice : tensor<61x230x4xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_6_2 wrap %extracted_slice : tensor<61x230x4xi8>
        %15 = dfscheblueprint.data_slice @consumer_slice_6_3 wrap %extracted_slice : tensor<61x230x4xi8>
        dfscheblueprint.flowconfig @flow_src_6 {
          target = @group_src_6,
          view = %extracted_slice : tensor<61x230x4xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = MM2S>
          ,type = "shim"
          ,data_id = 1 : i32
          ,shim_dim_strides = [4 : i32, 920 : i32, 224 : i32]
          ,shim_dim_wraps = [61 : i32, 19 : i32, 4 : i32]
        }
        dfscheblueprint.flowconfig @flow_dst_6 {
          target = @group_dst_6,
          view = %extracted_slice : tensor<61x230x4xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,slice_symbols = [@consumer_slice_6_0, @consumer_slice_6_1, @consumer_slice_6_2, @consumer_slice_6_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_6 {
          type = "one_to_many",
          from = @flow_src_6,
          to = @flow_dst_6
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 6 : i32
        }
        %extracted_slice_0 = tensor.extract_slice %7[28, 0, 0] [28, 112, 16] [1, 1, 1] {tag = "partitionslice1", tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : tensor<112x112x64xi8> to tensor<28x112x16xi8>
        %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer0"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer1"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer2"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer3"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %16 = dfscheblueprint.data_slice @producer_slice_7_0 wrap %extracted_slice_1 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %17 = dfscheblueprint.data_slice @producer_slice_7_1 wrap %extracted_slice_2 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %18 = dfscheblueprint.data_slice @producer_slice_7_2 wrap %extracted_slice_3 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %19 = dfscheblueprint.data_slice @producer_slice_7_3 wrap %extracted_slice_4 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        dfscheblueprint.flowconfig @flow_src_7 {
          target = @group_src_7,
          view = %extracted_slice_0 : tensor<28x112x16xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,slice_symbols = [@producer_slice_7_0, @producer_slice_7_1, @producer_slice_7_2, @producer_slice_7_3]
          ,type = "core"
          ,pp_depth = 2 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_7 {
          target = @group_dst_7,
          view = %extracted_slice_0 : tensor<28x112x16xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,type = "shim"
          ,data_id = 2 : i32
          ,pp_depth = 2 : i32
        }
        dfscheblueprint.flow_transfer @transfer_7 {
          type = "many_to_one",
          from = @flow_src_7,
          to = @flow_dst_7
          ,ordering = "sequential"
          ,base_packet_id = 5 : i32,
          flow_index = 7 : i32
        }
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_9{tiles = [[0, 5], [1, 5], [2, 5], [3, 5]]}
        dfscheblueprint.tile_group @group_dst_9{tiles = [[2, 0]]}
        dfscheblueprint.tile_group @group_src_8{tiles = [[2, 0]]}
        dfscheblueprint.tile_group @group_dst_8{tiles = [[0, 5], [1, 5], [2, 5], [3, 5]]}
        %extracted_slice = tensor.extract_slice %6[112, 0, 0] [61, 230, 4] [1, 1, 1] {tag = "partitionslice2", tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : tensor<230x230x4xi8> to tensor<61x230x4xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_8_0 wrap %extracted_slice : tensor<61x230x4xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_8_1 wrap %extracted_slice : tensor<61x230x4xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_8_2 wrap %extracted_slice : tensor<61x230x4xi8>
        %15 = dfscheblueprint.data_slice @consumer_slice_8_3 wrap %extracted_slice : tensor<61x230x4xi8>
        dfscheblueprint.flowconfig @flow_src_8 {
          target = @group_src_8,
          view = %extracted_slice : tensor<61x230x4xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = MM2S>
          ,type = "shim"
          ,data_id = 1 : i32
          ,shim_dim_strides = [4 : i32, 920 : i32, 224 : i32]
          ,shim_dim_wraps = [61 : i32, 19 : i32, 4 : i32]
        }
        dfscheblueprint.flowconfig @flow_dst_8 {
          target = @group_dst_8,
          view = %extracted_slice : tensor<61x230x4xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,slice_symbols = [@consumer_slice_8_0, @consumer_slice_8_1, @consumer_slice_8_2, @consumer_slice_8_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_8 {
          type = "one_to_many",
          from = @flow_src_8,
          to = @flow_dst_8
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 8 : i32
        }
        %extracted_slice_0 = tensor.extract_slice %7[56, 0, 0] [28, 112, 16] [1, 1, 1] {tag = "partitionslice2", tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : tensor<112x112x64xi8> to tensor<28x112x16xi8>
        %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer0"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer1"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer2"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer3"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %16 = dfscheblueprint.data_slice @producer_slice_9_0 wrap %extracted_slice_1 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %17 = dfscheblueprint.data_slice @producer_slice_9_1 wrap %extracted_slice_2 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %18 = dfscheblueprint.data_slice @producer_slice_9_2 wrap %extracted_slice_3 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %19 = dfscheblueprint.data_slice @producer_slice_9_3 wrap %extracted_slice_4 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        dfscheblueprint.flowconfig @flow_src_9 {
          target = @group_src_9,
          view = %extracted_slice_0 : tensor<28x112x16xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,slice_symbols = [@producer_slice_9_0, @producer_slice_9_1, @producer_slice_9_2, @producer_slice_9_3]
          ,type = "core"
          ,pp_depth = 2 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_9 {
          target = @group_dst_9,
          view = %extracted_slice_0 : tensor<28x112x16xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,type = "shim"
          ,data_id = 2 : i32
          ,pp_depth = 2 : i32
        }
        dfscheblueprint.flow_transfer @transfer_9 {
          type = "many_to_one",
          from = @flow_src_9,
          to = @flow_dst_9
          ,ordering = "sequential"
          ,base_packet_id = 9 : i32,
          flow_index = 9 : i32
        }
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_11{tiles = [[0, 6], [1, 6], [2, 6], [3, 6]]}
        dfscheblueprint.tile_group @group_dst_11{tiles = [[2, 0]]}
        dfscheblueprint.tile_group @group_src_10{tiles = [[3, 0]]}
        dfscheblueprint.tile_group @group_dst_10{tiles = [[0, 6], [1, 6], [2, 6], [3, 6]]}
        %extracted_slice = tensor.extract_slice %6[168, 0, 0] [61, 230, 4] [1, 1, 1] {tag = "partitionslice3", tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : tensor<230x230x4xi8> to tensor<61x230x4xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_10_0 wrap %extracted_slice : tensor<61x230x4xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_10_1 wrap %extracted_slice : tensor<61x230x4xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_10_2 wrap %extracted_slice : tensor<61x230x4xi8>
        %15 = dfscheblueprint.data_slice @consumer_slice_10_3 wrap %extracted_slice : tensor<61x230x4xi8>
        dfscheblueprint.flowconfig @flow_src_10 {
          target = @group_src_10,
          view = %extracted_slice : tensor<61x230x4xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = MM2S>
          ,type = "shim"
          ,data_id = 1 : i32
          ,shim_dim_strides = [4 : i32, 920 : i32, 224 : i32]
          ,shim_dim_wraps = [61 : i32, 19 : i32, 4 : i32]
        }
        dfscheblueprint.flowconfig @flow_dst_10 {
          target = @group_dst_10,
          view = %extracted_slice : tensor<61x230x4xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,slice_symbols = [@consumer_slice_10_0, @consumer_slice_10_1, @consumer_slice_10_2, @consumer_slice_10_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_10 {
          type = "one_to_many",
          from = @flow_src_10,
          to = @flow_dst_10
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 10 : i32
        }
        %extracted_slice_0 = tensor.extract_slice %7[84, 0, 0] [28, 112, 16] [1, 1, 1] {tag = "partitionslice3", tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : tensor<112x112x64xi8> to tensor<28x112x16xi8>
        %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer0"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer1"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer2"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[0, 0, 0] [7, 28, 16] [1, 1, 1] {tag = "producer3"} : tensor<28x112x16xi8> to tensor<7x28x16xi8>
        %16 = dfscheblueprint.data_slice @producer_slice_11_0 wrap %extracted_slice_1 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %17 = dfscheblueprint.data_slice @producer_slice_11_1 wrap %extracted_slice_2 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %18 = dfscheblueprint.data_slice @producer_slice_11_2 wrap %extracted_slice_3 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        %19 = dfscheblueprint.data_slice @producer_slice_11_3 wrap %extracted_slice_4 : tensor<7x28x16xi8> {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>}
        dfscheblueprint.flowconfig @flow_src_11 {
          target = @group_src_11,
          view = %extracted_slice_0 : tensor<28x112x16xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,slice_symbols = [@producer_slice_11_0, @producer_slice_11_1, @producer_slice_11_2, @producer_slice_11_3]
          ,type = "core"
          ,pp_depth = 2 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_11 {
          target = @group_dst_11,
          view = %extracted_slice_0 : tensor<28x112x16xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,type = "shim"
          ,data_id = 2 : i32
          ,pp_depth = 2 : i32
        }
        dfscheblueprint.flow_transfer @transfer_11 {
          type = "many_to_one",
          from = @flow_src_11,
          to = @flow_dst_11
          ,ordering = "sequential"
          ,base_packet_id = 13 : i32,
          flow_index = 11 : i32
        }
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
