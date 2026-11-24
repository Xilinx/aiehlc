module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  dfschedule.module @kernel_driver_dskernel_receiver {
    dfschedule.kernel_config_def @config {buffer_size = 1159 : i32, element_type = i8, kernel_file = "computekernel.cc", kernel_name = "computekernel", vector_width = 4 : i32}
    dfschedule.lock_def @LOCK_window_in_0_ACQ {id = 48 : i32, init_value = 2 : i32}
    dfschedule.lock_def @LOCK_window_in_0_REL {id = 49 : i32}
    dfschedule.buffer_def @buf_in_ping_0 : memref<1159xvector<4xi8>, "LOCAL"> {address = 497792 : i64}
    dfschedule.buffer_def @buf_in_pong_0 : memref<1159xvector<4xi8>, "LOCAL"> {address = 502432 : i64}
    dfschedule.window_def @window_in_0 {acquire_lock = @LOCK_window_in_0_ACQ, async = true, buffer_size = 1159 : i32, direction = "in", dma_channel = 0 : i32, num_rounds = 16 : i32, ping_buffer = @buf_in_ping_0, pong_buffer = @buf_in_pong_0, release_lock = @LOCK_window_in_0_REL, single_buffer = false}
    dfschedule.lock_def @LOCK_window_in_1_ACQ {id = 50 : i32, init_value = 2 : i32}
    dfschedule.lock_def @LOCK_window_in_1_REL {id = 51 : i32}
    dfschedule.buffer_def @buf_in_ping_1 : memref<784xvector<4xi8>, "LOCAL"> {address = 491520 : i64}
    dfschedule.buffer_def @buf_in_pong_1 : memref<784xvector<4xi8>, "LOCAL"> {address = 494656 : i64}
    dfschedule.window_def @window_in_1 {acquire_lock = @LOCK_window_in_1_ACQ, async = true, buffer_size = 784 : i32, direction = "in", dma_channel = 1 : i32, num_rounds = 1 : i32, ping_buffer = @buf_in_ping_1, pong_buffer = @buf_in_pong_1, release_lock = @LOCK_window_in_1_REL, single_buffer = false}
    dfschedule.lock_def @LOCK_window_out_0_ACQ {id = 52 : i32, init_value = 0 : i32}
    dfschedule.lock_def @LOCK_window_out_0_REL {id = 53 : i32}
    dfschedule.buffer_def @buf_out_ping_0 : memref<784xvector<4xi8>, "LOCAL"> {address = 507072 : i64}
    dfschedule.buffer_def @buf_out_pong_0 : memref<784xvector<4xi8>, "LOCAL"> {address = 510208 : i64}
    dfschedule.window_def @window_out_0 {acquire_lock = @LOCK_window_out_0_ACQ, async = true, buffer_size = 784 : i32, direction = "out", dma_channel = 0 : i32, num_rounds = 4 : i32, ping_buffer = @buf_out_ping_0, pong_buffer = @buf_out_pong_0, release_lock = @LOCK_window_out_0_REL, single_buffer = false}
    dfschedule.kernel_decl @computekernel {inputs = [@window_in_0, @window_in_1], iteration_style = "internal", outputs = [@window_out_0]}
    dfschedule.main @main {
      %0 = dfschedule.alloc_sync_buffer {size = 8 : i32} : !dfschedule.sync_buffer
      %c0_i32 = arith.constant 0 : i32
      dfschedule.sync_buffer_write(%0, %c0_i32) {index = 0 : i32}
      %c1_i32 = arith.constant 1 : i32
      dfschedule.log(%c1_i32)
      %1 = dfschedule.window_init(@window_in_0) : !dfschedule.input_window<i8>
      %2 = dfschedule.window_init(@window_in_1) : !dfschedule.input_window<i8>
      %3 = dfschedule.window_init(@window_out_0) : !dfschedule.output_window<i8>
      dfschedule.kernel_invoke @computekernel(%1, %2, %3) : (!dfschedule.input_window<i8>, !dfschedule.input_window<i8>, !dfschedule.output_window<i8>)
      dfschedule.done
      dfschedule.kernel_return
    }
  }
}
