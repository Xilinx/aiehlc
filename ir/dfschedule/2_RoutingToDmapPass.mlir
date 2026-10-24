module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 1 : i64, routing.control_plan_dedicated_shim = 1 : i64, routing.control_plan_exclusive = 1 : i64, routing.control_plan_group_reg_write = 1 : i64, routing.control_plan_mm2s_ch = 0 : i64, routing.control_plan_op_control_packet = 1 : i64, routing.control_plan_s2mm_ch = 0 : i64, routing.control_plan_shim_bd_ctrl = 1 : i64, routing.control_plan_shim_col = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 1 : i64, routing.output_tile_major = 1 : i64, routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}, routing.spatial_halo_buf_size = 4864 : i64, routing.spatial_out_rounds = 16 : i64, tensor_0.halo = {k_rounds = 4 : i32, k_slice = 256 : i32, k_step = 224 : i32, l2_rounds = 4 : i32, l2_slice = 19 : i32, l2_step = 14 : i32, ow_t = 28 : i32, row_pitch = 928 : i32, slice = 61 : i32, split_dim = 0 : i32, step = 56 : i32, w_rounds = 4 : i32, w_slice = 64 : i32, w_step = 56 : i32}, tensor_0.layout_transform = "dma_shuffle", tensor_1.layout_transform = "dma_shuffle"} {
  func.func @main(%arg0: memref<230x232x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg0 : memref<230x232x4xi8>
    %1 = routing.routingcreatescheduletensor %0 : tensor<230x232x4xi8> shape = [230, 232, 4], dim = 3 -> tensor<230x232x4xi8>
    %2 = bufferization.to_tensor %arg1 : memref<64x7x7x4xi8>
    %3 = routing.routingcreatescheduletensor %2 : tensor<64x7x7x4xi8> shape = [64, 7, 7, 4], dim = 4 -> tensor<64x7x7x4xi8>
    %4 = bufferization.to_tensor %arg2 : memref<112x112x64xi8>
    %5 = routing.routingcreatescheduletensor %4 : tensor<112x112x64xi8> shape = [112, 112, 64], dim = 3 -> tensor<112x112x64xi8>
    scf.execute_region {
      %6 = routing.partitiontensor %3 : tensor<64x7x7x4xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "col", replicateOn = "row", singleTileOwner = "">
} -> tensor<64x7x7x4xi8>
      %7 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_data %6, %arg3 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
        %12 = dmap.define_io_engine {io_id = 0 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %13 = dmap.define_core_group {core_count = 4 : i32, group_axis = "col", group_idx = 0 : i32} -> !dmap.dmacoreenginegroupType
        %14 = dmap.define_port_configure @receive_port_0 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %15 = dmap.create_io_engin_with_config %12 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %16 = dmap.create_core_group_with_config %13{[{0, @receive_port_0}, {1, @receive_port_0}, {2, @receive_port_0}, {3, @receive_port_0}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %17 = dmap.create_stream src = %15, dst = %16, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %11 : tensor<16x7x7x4xi8> to %17 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      %8 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_data %6, %arg3 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
        %12 = dmap.define_io_engine {io_id = 1 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %13 = dmap.define_core_group {core_count = 4 : i32, group_axis = "col", group_idx = 1 : i32} -> !dmap.dmacoreenginegroupType
        %14 = dmap.define_port_configure @receive_port_1 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %15 = dmap.create_io_engin_with_config %12 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %16 = dmap.create_core_group_with_config %13{[{0, @receive_port_1}, {1, @receive_port_1}, {2, @receive_port_1}, {3, @receive_port_1}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %17 = dmap.create_stream src = %15, dst = %16, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %11 : tensor<16x7x7x4xi8> to %17 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      %9 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_data %6, %arg3 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
        %12 = dmap.define_io_engine {io_id = 2 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %13 = dmap.define_core_group {core_count = 4 : i32, group_axis = "col", group_idx = 2 : i32} -> !dmap.dmacoreenginegroupType
        %14 = dmap.define_port_configure @receive_port_2 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %15 = dmap.create_io_engin_with_config %12 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %16 = dmap.create_core_group_with_config %13{[{0, @receive_port_2}, {1, @receive_port_2}, {2, @receive_port_2}, {3, @receive_port_2}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %17 = dmap.create_stream src = %15, dst = %16, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %11 : tensor<16x7x7x4xi8> to %17 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_data %6, %arg3 : tensor<64x7x7x4xi8>, i32 -> tensor<16x7x7x4xi8>
        %12 = dmap.define_io_engine {io_id = 3 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %13 = dmap.define_core_group {core_count = 4 : i32, group_axis = "col", group_idx = 3 : i32} -> !dmap.dmacoreenginegroupType
        %14 = dmap.define_port_configure @receive_port_3 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %15 = dmap.create_io_engin_with_config %12 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %16 = dmap.create_core_group_with_config %13{[{0, @receive_port_3}, {1, @receive_port_3}, {2, @receive_port_3}, {3, @receive_port_3}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %17 = dmap.create_stream src = %15, dst = %16, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %11 : tensor<16x7x7x4xi8> to %17 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %6 = routing.partitiontensor %1 : tensor<230x232x4xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 232, total = 232, slice = 232, step = 232, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>
} -> tensor<230x232x4xi8>
      %7 = routing.partitiontensor %5 : tensor<112x112x64xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>
} -> tensor<112x112x64xi8>
      %8 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %12 = routing.routingextract_data %6, %arg3 : tensor<230x232x4xi8>, i32 -> tensor<61x232x4xi8>
        %13 = dmap.define_io_engine {io_id = 4 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %14 = dmap.define_core_group {core_count = 4 : i32, group_axis = "row", group_idx = 0 : i32} -> !dmap.dmacoreenginegroupType
        %15 = dmap.define_port_configure @receive_port_4 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %16 = dmap.create_io_engin_with_config %13 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %17 = dmap.create_core_group_with_config %14{[{0, @receive_port_4}, {1, @receive_port_4}, {2, @receive_port_4}, {3, @receive_port_4}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %18 = dmap.create_stream src = %16, dst = %17, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %12 : tensor<61x232x4xi8> to %18 : !dmap.dmapportstream
        %19 = routing.routingextract_data %7, %arg3 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
        %20 = dmap.define_io_engine {io_id = 5 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %21 = dmap.define_core_group {core_count = 4 : i32, group_axis = "row", group_idx = 0 : i32} -> !dmap.dmacoreenginegroupType
        %22 = dmap.define_port_configure @send_port_5 : {"SEND", 16, 1, 1} -> !dmap.dmapportconfig
        %23 = dmap.create_io_engin_with_config %20 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"RECEIVE", 16, 1, 1}>} -> !dmap.dmapioconfig
        %24 = dmap.create_core_group_with_config %21{[{0, @send_port_5}, {1, @send_port_5}, {2, @send_port_5}, {3, @send_port_5}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %25 = dmap.create_stream src = %24, dst = %23, !dmap.dmacoregroupconfig !dmap.dmapioconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.pull %19 : tensor<28x112x16xi8> from %25 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      %9 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %12 = routing.routingextract_data %6, %arg3 : tensor<230x232x4xi8>, i32 -> tensor<61x232x4xi8>
        %13 = dmap.define_io_engine {io_id = 6 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %14 = dmap.define_core_group {core_count = 4 : i32, group_axis = "row", group_idx = 1 : i32} -> !dmap.dmacoreenginegroupType
        %15 = dmap.define_port_configure @receive_port_6 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %16 = dmap.create_io_engin_with_config %13 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %17 = dmap.create_core_group_with_config %14{[{0, @receive_port_6}, {1, @receive_port_6}, {2, @receive_port_6}, {3, @receive_port_6}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %18 = dmap.create_stream src = %16, dst = %17, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %12 : tensor<61x232x4xi8> to %18 : !dmap.dmapportstream
        %19 = routing.routingextract_data %7, %arg3 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
        %20 = dmap.define_io_engine {io_id = 7 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %21 = dmap.define_core_group {core_count = 4 : i32, group_axis = "row", group_idx = 1 : i32} -> !dmap.dmacoreenginegroupType
        %22 = dmap.define_port_configure @send_port_7 : {"SEND", 16, 1, 1} -> !dmap.dmapportconfig
        %23 = dmap.create_io_engin_with_config %20 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"RECEIVE", 16, 1, 1}>} -> !dmap.dmapioconfig
        %24 = dmap.create_core_group_with_config %21{[{0, @send_port_7}, {1, @send_port_7}, {2, @send_port_7}, {3, @send_port_7}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %25 = dmap.create_stream src = %24, dst = %23, !dmap.dmacoregroupconfig !dmap.dmapioconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.pull %19 : tensor<28x112x16xi8> from %25 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %12 = routing.routingextract_data %6, %arg3 : tensor<230x232x4xi8>, i32 -> tensor<61x232x4xi8>
        %13 = dmap.define_io_engine {io_id = 8 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %14 = dmap.define_core_group {core_count = 4 : i32, group_axis = "row", group_idx = 2 : i32} -> !dmap.dmacoreenginegroupType
        %15 = dmap.define_port_configure @receive_port_8 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %16 = dmap.create_io_engin_with_config %13 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %17 = dmap.create_core_group_with_config %14{[{0, @receive_port_8}, {1, @receive_port_8}, {2, @receive_port_8}, {3, @receive_port_8}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %18 = dmap.create_stream src = %16, dst = %17, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %12 : tensor<61x232x4xi8> to %18 : !dmap.dmapportstream
        %19 = routing.routingextract_data %7, %arg3 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
        %20 = dmap.define_io_engine {io_id = 9 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %21 = dmap.define_core_group {core_count = 4 : i32, group_axis = "row", group_idx = 2 : i32} -> !dmap.dmacoreenginegroupType
        %22 = dmap.define_port_configure @send_port_9 : {"SEND", 16, 1, 1} -> !dmap.dmapportconfig
        %23 = dmap.create_io_engin_with_config %20 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"RECEIVE", 16, 1, 1}>} -> !dmap.dmapioconfig
        %24 = dmap.create_core_group_with_config %21{[{0, @send_port_9}, {1, @send_port_9}, {2, @send_port_9}, {3, @send_port_9}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %25 = dmap.create_stream src = %24, dst = %23, !dmap.dmacoregroupconfig !dmap.dmapioconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.pull %19 : tensor<28x112x16xi8> from %25 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %12 = routing.routingextract_data %6, %arg3 : tensor<230x232x4xi8>, i32 -> tensor<61x232x4xi8>
        %13 = dmap.define_io_engine {io_id = 10 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %14 = dmap.define_core_group {core_count = 4 : i32, group_axis = "row", group_idx = 3 : i32} -> !dmap.dmacoreenginegroupType
        %15 = dmap.define_port_configure @receive_port_10 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %16 = dmap.create_io_engin_with_config %13 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %17 = dmap.create_core_group_with_config %14{[{0, @receive_port_10}, {1, @receive_port_10}, {2, @receive_port_10}, {3, @receive_port_10}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %18 = dmap.create_stream src = %16, dst = %17, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %12 : tensor<61x232x4xi8> to %18 : !dmap.dmapportstream
        %19 = routing.routingextract_data %7, %arg3 : tensor<112x112x64xi8>, i32 -> tensor<28x112x16xi8>
        %20 = dmap.define_io_engine {io_id = 11 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %21 = dmap.define_core_group {core_count = 4 : i32, group_axis = "row", group_idx = 3 : i32} -> !dmap.dmacoreenginegroupType
        %22 = dmap.define_port_configure @send_port_11 : {"SEND", 16, 1, 1} -> !dmap.dmapportconfig
        %23 = dmap.create_io_engin_with_config %20 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"RECEIVE", 16, 1, 1}>} -> !dmap.dmapioconfig
        %24 = dmap.create_core_group_with_config %21{[{0, @send_port_11}, {1, @send_port_11}, {2, @send_port_11}, {3, @send_port_11}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %25 = dmap.create_stream src = %24, dst = %23, !dmap.dmacoregroupconfig !dmap.dmapioconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.pull %19 : tensor<28x112x16xi8> from %25 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
