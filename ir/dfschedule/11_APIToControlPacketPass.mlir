module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_aot = 0 : i64, routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  func.func @main(%arg0: memref<230x230x4xi8>, %arg1: memref<64x7x7x4xi8>, %arg2: memref<112x112x64xi8>) attributes {routing.kernel_tiling = #routing.kernel_tiling<out_rounds = 16>} {
    dfschedule.launchhost @host_canonicalized
    return
  }
  dfschedule.host @host_canonicalized {
    %c15_i32 = arith.constant 15 : i32
    %c14_i32 = arith.constant 14 : i32
    %c13_i32 = arith.constant 13 : i32
    %c12_i32 = arith.constant 12 : i32
    %c11_i32 = arith.constant 11 : i32
    %c10_i32 = arith.constant 10 : i32
    %c9_i32 = arith.constant 9 : i32
    %c8_i32 = arith.constant 8 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c50176_i32 = arith.constant 50176 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %subview = memref.subview %arg1[0, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>
    %0 = dfschedule.declaretile {col = 0 : i32, row = 0 : i32} : !dfschedule.tile
    %1 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
    %2 = dfschedule.memref_mapping %subview : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
    %3 = dfschedule.bind_core_buffer(%2, %1) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %4 = dfschedule.bind_core_buffer(%2, %1) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %5 = dfschedule.schedule.getbdid(%1) : (!dfschedule.tile) -> i32
    %6 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
    %7 = dfschedule.memref_mapping %subview : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
    %8 = dfschedule.bind_core_buffer(%7, %6) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %9 = dfschedule.bind_core_buffer(%7, %6) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %10 = dfschedule.schedule.getbdid(%6) : (!dfschedule.tile) -> i32
    %11 = dfschedule.declaretile {col = 0 : i32, row = 5 : i32} : !dfschedule.tile
    %12 = dfschedule.memref_mapping %subview : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
    %13 = dfschedule.bind_core_buffer(%12, %11) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %14 = dfschedule.bind_core_buffer(%12, %11) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %15 = dfschedule.schedule.getbdid(%11) : (!dfschedule.tile) -> i32
    %16 = dfschedule.declaretile {col = 0 : i32, row = 6 : i32} : !dfschedule.tile
    %17 = dfschedule.memref_mapping %subview : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>) -> memref<16x7x7x4xi8>
    %18 = dfschedule.bind_core_buffer(%17, %16) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %19 = dfschedule.bind_core_buffer(%17, %16) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %20 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32
    %21 = dfschedule.schedule.getbdid(%0) : (!dfschedule.tile) -> i32
    %subview_0 = memref.subview %arg1[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>
    %22 = dfschedule.declaretile {col = 1 : i32, row = 0 : i32} : !dfschedule.tile
    %23 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
    %subview_1 = memref.subview %subview_0[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %24 = dfschedule.memref_mapping %subview_1 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
    %25 = dfschedule.bind_core_buffer(%24, %23) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %26 = dfschedule.bind_core_buffer(%24, %23) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %27 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
    %28 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
    %subview_2 = memref.subview %subview_0[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %29 = dfschedule.memref_mapping %subview_2 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
    %30 = dfschedule.bind_core_buffer(%29, %28) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %31 = dfschedule.bind_core_buffer(%29, %28) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %32 = dfschedule.schedule.getbdid(%28) : (!dfschedule.tile) -> i32
    %33 = dfschedule.declaretile {col = 1 : i32, row = 5 : i32} : !dfschedule.tile
    %subview_3 = memref.subview %subview_0[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %34 = dfschedule.memref_mapping %subview_3 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
    %35 = dfschedule.bind_core_buffer(%34, %33) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %36 = dfschedule.bind_core_buffer(%34, %33) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %37 = dfschedule.schedule.getbdid(%33) : (!dfschedule.tile) -> i32
    %38 = dfschedule.declaretile {col = 1 : i32, row = 6 : i32} : !dfschedule.tile
    %subview_4 = memref.subview %subview_0[16, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %39 = dfschedule.memref_mapping %subview_4 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>) -> memref<16x7x7x4xi8>
    %40 = dfschedule.bind_core_buffer(%39, %38) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %41 = dfschedule.bind_core_buffer(%39, %38) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %42 = dfschedule.schedule.getbdid(%38) : (!dfschedule.tile) -> i32
    %43 = dfschedule.schedule.getbdid(%22) : (!dfschedule.tile) -> i32
    %subview_5 = memref.subview %arg1[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>
    %44 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
    %45 = dfschedule.declaretile {col = 2 : i32, row = 3 : i32} : !dfschedule.tile
    %subview_6 = memref.subview %subview_5[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
    %46 = dfschedule.memref_mapping %subview_6 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
    %47 = dfschedule.bind_core_buffer(%46, %45) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %48 = dfschedule.bind_core_buffer(%46, %45) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %49 = dfschedule.schedule.getbdid(%45) : (!dfschedule.tile) -> i32
    %50 = dfschedule.declaretile {col = 2 : i32, row = 4 : i32} : !dfschedule.tile
    %subview_7 = memref.subview %subview_5[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
    %51 = dfschedule.memref_mapping %subview_7 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
    %52 = dfschedule.bind_core_buffer(%51, %50) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %53 = dfschedule.bind_core_buffer(%51, %50) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %54 = dfschedule.schedule.getbdid(%50) : (!dfschedule.tile) -> i32
    %55 = dfschedule.declaretile {col = 2 : i32, row = 5 : i32} : !dfschedule.tile
    %subview_8 = memref.subview %subview_5[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
    %56 = dfschedule.memref_mapping %subview_8 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
    %57 = dfschedule.bind_core_buffer(%56, %55) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %58 = dfschedule.bind_core_buffer(%56, %55) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %59 = dfschedule.schedule.getbdid(%55) : (!dfschedule.tile) -> i32
    %60 = dfschedule.declaretile {col = 2 : i32, row = 6 : i32} : !dfschedule.tile
    %subview_9 = memref.subview %subview_5[32, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>
    %61 = dfschedule.memref_mapping %subview_9 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 12544>>) -> memref<16x7x7x4xi8>
    %62 = dfschedule.bind_core_buffer(%61, %60) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %63 = dfschedule.bind_core_buffer(%61, %60) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %64 = dfschedule.schedule.getbdid(%60) : (!dfschedule.tile) -> i32
    %65 = dfschedule.schedule.getbdid(%44) : (!dfschedule.tile) -> i32
    %subview_10 = memref.subview %arg1[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<64x7x7x4xi8> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>
    %66 = dfschedule.declaretile {col = 3 : i32, row = 0 : i32} : !dfschedule.tile
    %67 = dfschedule.declaretile {col = 3 : i32, row = 3 : i32} : !dfschedule.tile
    %subview_11 = memref.subview %subview_10[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
    %68 = dfschedule.memref_mapping %subview_11 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
    %69 = dfschedule.bind_core_buffer(%68, %67) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %70 = dfschedule.bind_core_buffer(%68, %67) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %71 = dfschedule.schedule.getbdid(%67) : (!dfschedule.tile) -> i32
    %72 = dfschedule.declaretile {col = 3 : i32, row = 4 : i32} : !dfschedule.tile
    %subview_12 = memref.subview %subview_10[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
    %73 = dfschedule.memref_mapping %subview_12 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
    %74 = dfschedule.bind_core_buffer(%73, %72) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %75 = dfschedule.bind_core_buffer(%73, %72) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %76 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    %77 = dfschedule.declaretile {col = 3 : i32, row = 5 : i32} : !dfschedule.tile
    %subview_13 = memref.subview %subview_10[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
    %78 = dfschedule.memref_mapping %subview_13 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
    %79 = dfschedule.bind_core_buffer(%78, %77) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %80 = dfschedule.bind_core_buffer(%78, %77) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %81 = dfschedule.schedule.getbdid(%77) : (!dfschedule.tile) -> i32
    %82 = dfschedule.declaretile {col = 3 : i32, row = 6 : i32} : !dfschedule.tile
    %subview_14 = memref.subview %subview_10[48, 0, 0, 0] [16, 7, 7, 4] [1, 1, 1, 1] : memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>> to memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>
    %83 = dfschedule.memref_mapping %subview_14 : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 18816>>) -> memref<16x7x7x4xi8>
    %84 = dfschedule.bind_core_buffer(%83, %82) {offset = 32768 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %85 = dfschedule.bind_core_buffer(%83, %82) {offset = 35904 : i64} : (memref<16x7x7x4xi8>, !dfschedule.tile) -> memref<16x7x7x4xi8>
    %86 = dfschedule.schedule.getbdid(%82) : (!dfschedule.tile) -> i32
    %87 = dfschedule.schedule.getbdid(%66) : (!dfschedule.tile) -> i32
    %subview_15 = memref.subview %arg0[0, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1]>>
    %88 = dfschedule.memref_mapping %subview_15 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
    %89 = dfschedule.bind_core_buffer(%88, %1) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %90 = dfschedule.bind_core_buffer(%88, %1) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %91 = dfschedule.schedule.getbdid(%1) : (!dfschedule.tile) -> i32
    %92 = dfschedule.memref_mapping %subview_15 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
    %93 = dfschedule.bind_core_buffer(%92, %23) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %94 = dfschedule.bind_core_buffer(%92, %23) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %95 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
    %96 = dfschedule.memref_mapping %subview_15 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
    %97 = dfschedule.bind_core_buffer(%96, %45) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %98 = dfschedule.bind_core_buffer(%96, %45) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %99 = dfschedule.schedule.getbdid(%45) : (!dfschedule.tile) -> i32
    %100 = dfschedule.memref_mapping %subview_15 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1]>>) -> memref<61x230x4xi8>
    %101 = dfschedule.bind_core_buffer(%100, %67) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %102 = dfschedule.bind_core_buffer(%100, %67) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %103 = dfschedule.schedule.getbdid(%67) : (!dfschedule.tile) -> i32
    %104 = dfschedule.schedule.getbdid(%0) : (!dfschedule.tile) -> i32
    %subview_16 = memref.subview %arg2[0, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1]>>
    %subview_17 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %105 = dfschedule.memref_mapping %subview_17 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %106 = dfschedule.bind_core_buffer(%105, %1) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %107 = dfschedule.bind_core_buffer(%105, %1) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %108 = dfschedule.schedule.getbdid(%1) : (!dfschedule.tile) -> i32
    %subview_18 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %109 = dfschedule.memref_mapping %subview_18 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %110 = dfschedule.bind_core_buffer(%109, %23) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %111 = dfschedule.bind_core_buffer(%109, %23) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %112 = dfschedule.schedule.getbdid(%23) : (!dfschedule.tile) -> i32
    %subview_19 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %113 = dfschedule.memref_mapping %subview_19 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %114 = dfschedule.bind_core_buffer(%113, %45) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %115 = dfschedule.bind_core_buffer(%113, %45) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %116 = dfschedule.schedule.getbdid(%45) : (!dfschedule.tile) -> i32
    %subview_20 = memref.subview %subview_16[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1]>> to memref<7x28x16xi8, strided<[7168, 64, 1]>>
    %117 = dfschedule.memref_mapping %subview_20 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1]>>) -> memref<7x28x16xi8>
    %118 = dfschedule.bind_core_buffer(%117, %67) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %119 = dfschedule.bind_core_buffer(%117, %67) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %120 = dfschedule.schedule.getbdid(%67) : (!dfschedule.tile) -> i32
    %121 = dfschedule.schedule.getbdid(%66) : (!dfschedule.tile) -> i32
    %subview_21 = memref.subview %arg0[56, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>>
    %subview_22 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %122 = dfschedule.memref_mapping %subview_22 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %123 = dfschedule.bind_core_buffer(%122, %6) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %124 = dfschedule.bind_core_buffer(%122, %6) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %125 = dfschedule.schedule.getbdid(%6) : (!dfschedule.tile) -> i32
    %subview_23 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %126 = dfschedule.memref_mapping %subview_23 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %127 = dfschedule.bind_core_buffer(%126, %28) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %128 = dfschedule.bind_core_buffer(%126, %28) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %129 = dfschedule.schedule.getbdid(%28) : (!dfschedule.tile) -> i32
    %subview_24 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %130 = dfschedule.memref_mapping %subview_24 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %131 = dfschedule.bind_core_buffer(%130, %50) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %132 = dfschedule.bind_core_buffer(%130, %50) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %133 = dfschedule.schedule.getbdid(%50) : (!dfschedule.tile) -> i32
    %subview_25 = memref.subview %subview_21[56, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %134 = dfschedule.memref_mapping %subview_25 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>) -> memref<61x230x4xi8>
    %135 = dfschedule.bind_core_buffer(%134, %72) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %136 = dfschedule.bind_core_buffer(%134, %72) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %137 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    %138 = dfschedule.schedule.getbdid(%22) : (!dfschedule.tile) -> i32
    %subview_26 = memref.subview %arg2[28, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %subview_27 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %139 = dfschedule.memref_mapping %subview_27 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %140 = dfschedule.bind_core_buffer(%139, %6) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %141 = dfschedule.bind_core_buffer(%139, %6) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %142 = dfschedule.schedule.getbdid(%6) : (!dfschedule.tile) -> i32
    %subview_28 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %143 = dfschedule.memref_mapping %subview_28 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %144 = dfschedule.bind_core_buffer(%143, %28) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %145 = dfschedule.bind_core_buffer(%143, %28) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %146 = dfschedule.schedule.getbdid(%28) : (!dfschedule.tile) -> i32
    %subview_29 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %147 = dfschedule.memref_mapping %subview_29 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %148 = dfschedule.bind_core_buffer(%147, %50) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %149 = dfschedule.bind_core_buffer(%147, %50) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %150 = dfschedule.schedule.getbdid(%50) : (!dfschedule.tile) -> i32
    %subview_30 = memref.subview %subview_26[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>
    %151 = dfschedule.memref_mapping %subview_30 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 200704>>) -> memref<7x28x16xi8>
    %152 = dfschedule.bind_core_buffer(%151, %72) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %153 = dfschedule.bind_core_buffer(%151, %72) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %154 = dfschedule.schedule.getbdid(%72) : (!dfschedule.tile) -> i32
    %155 = dfschedule.schedule.getbdid(%66) : (!dfschedule.tile) -> i32
    %subview_31 = memref.subview %arg0[112, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>
    %subview_32 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %156 = dfschedule.memref_mapping %subview_32 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %157 = dfschedule.bind_core_buffer(%156, %11) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %158 = dfschedule.bind_core_buffer(%156, %11) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %159 = dfschedule.schedule.getbdid(%11) : (!dfschedule.tile) -> i32
    %subview_33 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %160 = dfschedule.memref_mapping %subview_33 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %161 = dfschedule.bind_core_buffer(%160, %33) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %162 = dfschedule.bind_core_buffer(%160, %33) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %163 = dfschedule.schedule.getbdid(%33) : (!dfschedule.tile) -> i32
    %subview_34 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %164 = dfschedule.memref_mapping %subview_34 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %165 = dfschedule.bind_core_buffer(%164, %55) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %166 = dfschedule.bind_core_buffer(%164, %55) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %167 = dfschedule.schedule.getbdid(%55) : (!dfschedule.tile) -> i32
    %subview_35 = memref.subview %subview_31[112, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>
    %168 = dfschedule.memref_mapping %subview_35 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 206080>>) -> memref<61x230x4xi8>
    %169 = dfschedule.bind_core_buffer(%168, %77) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %170 = dfschedule.bind_core_buffer(%168, %77) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %171 = dfschedule.schedule.getbdid(%77) : (!dfschedule.tile) -> i32
    %172 = dfschedule.schedule.getbdid(%44) : (!dfschedule.tile) -> i32
    %subview_36 = memref.subview %arg2[56, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %subview_37 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %173 = dfschedule.memref_mapping %subview_37 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %174 = dfschedule.bind_core_buffer(%173, %11) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %175 = dfschedule.bind_core_buffer(%173, %11) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %176 = dfschedule.schedule.getbdid(%11) : (!dfschedule.tile) -> i32
    %subview_38 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %177 = dfschedule.memref_mapping %subview_38 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %178 = dfschedule.bind_core_buffer(%177, %33) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %179 = dfschedule.bind_core_buffer(%177, %33) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %180 = dfschedule.schedule.getbdid(%33) : (!dfschedule.tile) -> i32
    %subview_39 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %181 = dfschedule.memref_mapping %subview_39 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %182 = dfschedule.bind_core_buffer(%181, %55) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %183 = dfschedule.bind_core_buffer(%181, %55) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %184 = dfschedule.schedule.getbdid(%55) : (!dfschedule.tile) -> i32
    %subview_40 = memref.subview %subview_36[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>
    %185 = dfschedule.memref_mapping %subview_40 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 401408>>) -> memref<7x28x16xi8>
    %186 = dfschedule.bind_core_buffer(%185, %77) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %187 = dfschedule.bind_core_buffer(%185, %77) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %188 = dfschedule.schedule.getbdid(%77) : (!dfschedule.tile) -> i32
    %189 = dfschedule.schedule.getbdid(%44) : (!dfschedule.tile) -> i32
    %subview_41 = memref.subview %arg0[168, 0, 0] [61, 230, 4] [1, 1, 1] : memref<230x230x4xi8> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>>
    %subview_42 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %190 = dfschedule.memref_mapping %subview_42 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %191 = dfschedule.bind_core_buffer(%190, %16) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %192 = dfschedule.bind_core_buffer(%190, %16) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %193 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32
    %subview_43 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %194 = dfschedule.memref_mapping %subview_43 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %195 = dfschedule.bind_core_buffer(%194, %38) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %196 = dfschedule.bind_core_buffer(%194, %38) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %197 = dfschedule.schedule.getbdid(%38) : (!dfschedule.tile) -> i32
    %subview_44 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %198 = dfschedule.memref_mapping %subview_44 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %199 = dfschedule.bind_core_buffer(%198, %60) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %200 = dfschedule.bind_core_buffer(%198, %60) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %201 = dfschedule.schedule.getbdid(%60) : (!dfschedule.tile) -> i32
    %subview_45 = memref.subview %subview_41[168, 0, 0] [61, 230, 4] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>> to memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>
    %202 = dfschedule.memref_mapping %subview_45 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 230, total = 230, slice = 230, step = 230, rounds = 1>>, d2 = #routing.dim<outer = #routing.level<base = 4, total = 4, slice = 4, step = 4, rounds = 1>>>} : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 309120>>) -> memref<61x230x4xi8>
    %203 = dfschedule.bind_core_buffer(%202, %82) {offset = 39040 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %204 = dfschedule.bind_core_buffer(%202, %82) {offset = 43680 : i64} : (memref<61x230x4xi8>, !dfschedule.tile) -> memref<61x230x4xi8>
    %205 = dfschedule.schedule.getbdid(%82) : (!dfschedule.tile) -> i32
    %206 = dfschedule.schedule.getbdid(%66) : (!dfschedule.tile) -> i32
    %subview_46 = memref.subview %arg2[84, 0, 0] [28, 112, 16] [1, 1, 1] : memref<112x112x64xi8> to memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %subview_47 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %207 = dfschedule.memref_mapping %subview_47 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %208 = dfschedule.bind_core_buffer(%207, %16) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %209 = dfschedule.bind_core_buffer(%207, %16) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %210 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32
    %subview_48 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %211 = dfschedule.memref_mapping %subview_48 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %212 = dfschedule.bind_core_buffer(%211, %38) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %213 = dfschedule.bind_core_buffer(%211, %38) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %214 = dfschedule.schedule.getbdid(%38) : (!dfschedule.tile) -> i32
    %subview_49 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %215 = dfschedule.memref_mapping %subview_49 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %216 = dfschedule.bind_core_buffer(%215, %60) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %217 = dfschedule.bind_core_buffer(%215, %60) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %218 = dfschedule.schedule.getbdid(%60) : (!dfschedule.tile) -> i32
    %subview_50 = memref.subview %subview_46[0, 0, 0] [7, 28, 16] [1, 1, 1] {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>> to memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>
    %219 = dfschedule.memref_mapping %subview_50 {tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4, slice_tiling = #routing.level<base = 28, total = 28, slice = 7, step = 7, rounds = 4>>, axis = 1>, d1 = #routing.dim<outer = #routing.level<base = 112, total = 112, slice = 28, step = 28, rounds = 4>>, d2 = #routing.dim<outer = #routing.level<base = 64, total = 64, slice = 16, step = 16, rounds = 4>, axis = 2>>} : (memref<7x28x16xi8, strided<[7168, 64, 1], offset: 602112>>) -> memref<7x28x16xi8>
    %220 = dfschedule.bind_core_buffer(%219, %82) {offset = 48320 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %221 = dfschedule.bind_core_buffer(%219, %82) {offset = 51456 : i64} : (memref<7x28x16xi8>, !dfschedule.tile) -> memref<7x28x16xi8>
    %222 = dfschedule.schedule.getbdid(%82) : (!dfschedule.tile) -> i32
    %223 = dfschedule.schedule.getbdid(%44) : (!dfschedule.tile) -> i32
    %224 = dfschedule.config.load_kernel_group(%1, %6, %11, %16, %23, %28, %33, %38, %45, %50, %55, %60, %67, %72, %77, %82) {
      callee = [@dskernel_receiver]
    } : (!dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
    %225 = dfschedule.schedule.launch_kernel_group(%224) : (!dfschedule.kernelgroup) -> !dfschedule.event
    %226 = dfschedule.config.dma_bd(%subview, %0, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %227 = dfschedule.config.create_io(%226, %0) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %228 = dfschedule.config.dma_bd(%4, %1, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %229 = dfschedule.config.dma_bd(%3, %1, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %228) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %230 = dfschedule.config.create_io(%229, %1) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %231 = dfschedule.config.dma_bd(%9, %6, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %232 = dfschedule.config.dma_bd(%8, %6, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %231) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %233 = dfschedule.config.create_io(%232, %6) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %234 = dfschedule.config.dma_bd(%14, %11, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %235 = dfschedule.config.dma_bd(%13, %11, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %234) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %236 = dfschedule.config.create_io(%235, %11) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %237 = dfschedule.config.dma_bd(%19, %16, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %238 = dfschedule.config.dma_bd(%18, %16, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %237) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %239 = dfschedule.config.create_io(%238, %16) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %240 = dfschedule.config.dma_bd(%subview_0, %22, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 3136>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %241 = dfschedule.config.create_io(%240, %22) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %242 = dfschedule.config.dma_bd(%26, %23, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %243 = dfschedule.config.dma_bd(%25, %23, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %242) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %244 = dfschedule.config.create_io(%243, %23) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %245 = dfschedule.config.dma_bd(%31, %28, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %246 = dfschedule.config.dma_bd(%30, %28, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %245) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %247 = dfschedule.config.create_io(%246, %28) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %248 = dfschedule.config.dma_bd(%36, %33, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %249 = dfschedule.config.dma_bd(%35, %33, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %248) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %250 = dfschedule.config.create_io(%249, %33) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %251 = dfschedule.config.dma_bd(%41, %38, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %252 = dfschedule.config.dma_bd(%40, %38, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %251) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %253 = dfschedule.config.create_io(%252, %38) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %254 = dfschedule.config.dma_bd(%subview_5, %44, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 6272>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %255 = dfschedule.config.create_io(%254, %44) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %256 = dfschedule.config.dma_bd(%48, %45, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %257 = dfschedule.config.dma_bd(%47, %45, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %256) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %258 = dfschedule.config.create_io(%257, %45) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %259 = dfschedule.config.dma_bd(%53, %50, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %260 = dfschedule.config.dma_bd(%52, %50, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %259) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %261 = dfschedule.config.create_io(%260, %50) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %262 = dfschedule.config.dma_bd(%58, %55, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %263 = dfschedule.config.dma_bd(%57, %55, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %262) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %264 = dfschedule.config.create_io(%263, %55) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %265 = dfschedule.config.dma_bd(%63, %60, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %266 = dfschedule.config.dma_bd(%62, %60, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %265) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %267 = dfschedule.config.create_io(%266, %60) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %268 = dfschedule.config.dma_bd(%subview_10, %66, %c0_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 0 : i32
    } : (memref<16x7x7x4xi8, strided<[196, 28, 4, 1], offset: 9408>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %269 = dfschedule.config.create_io(%268, %66) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %270 = dfschedule.config.dma_bd(%70, %67, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %271 = dfschedule.config.dma_bd(%69, %67, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %270) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %272 = dfschedule.config.create_io(%271, %67) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %273 = dfschedule.config.dma_bd(%75, %72, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %274 = dfschedule.config.dma_bd(%74, %72, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %273) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %275 = dfschedule.config.create_io(%274, %72) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %276 = dfschedule.config.dma_bd(%80, %77, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %277 = dfschedule.config.dma_bd(%79, %77, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %276) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %278 = dfschedule.config.create_io(%277, %77) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %279 = dfschedule.config.dma_bd(%85, %82, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 0 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %280 = dfschedule.config.dma_bd(%84, %82, %c0_i32, %c0_i32, %c0_i32, %c-1_i32, %279) {
      len = 3136 : i32,
      enable_packet = false,
      next_bd = 1 : i32,
      acquire_lock_id = 2 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 3 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<16x7x7x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %281 = dfschedule.config.create_io(%280, %82) {
      channel = 1,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %282 = dfschedule.config.dma_bd(%subview_15, %0, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 18544 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 1 : i32,
      dim_strides = [4, 920, 224],
      dim_wraps = [61, 19, 4],
      iter_step_size = 12880 : i32,
      iter_wrap = 4 : i32
    } : (memref<61x230x4xi8, strided<[920, 4, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %283 = dfschedule.config.create_io(%282, %0) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %284 = dfschedule.config.dma_bd(%90, %1, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %285 = dfschedule.config.dma_bd(%89, %1, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %284) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %286 = dfschedule.config.create_io(%285, %1) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %287 = dfschedule.config.dma_bd(%94, %23, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %288 = dfschedule.config.dma_bd(%93, %23, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %287) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %289 = dfschedule.config.create_io(%288, %23) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %290 = dfschedule.config.dma_bd(%98, %45, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %291 = dfschedule.config.dma_bd(%97, %45, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %290) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %292 = dfschedule.config.create_io(%291, %45) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %293 = dfschedule.config.dma_bd(%102, %67, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %294 = dfschedule.config.dma_bd(%101, %67, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %293) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %295 = dfschedule.config.create_io(%294, %67) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %296 = dfschedule.config.dma_bd(%107, %1, %c5_i32, %c0_i32, %c1_i32, %c2_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %297 = dfschedule.config.dma_bd(%106, %1, %c4_i32, %c0_i32, %c1_i32, %c2_i32, %296) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %298 = dfschedule.config.create_io(%297, %1) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %299 = dfschedule.config.dma_bd(%111, %23, %c5_i32, %c0_i32, %c2_i32, %c3_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %300 = dfschedule.config.dma_bd(%110, %23, %c4_i32, %c0_i32, %c2_i32, %c3_i32, %299) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %301 = dfschedule.config.create_io(%300, %23) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %302 = dfschedule.config.dma_bd(%115, %45, %c5_i32, %c0_i32, %c3_i32, %c4_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %303 = dfschedule.config.dma_bd(%114, %45, %c4_i32, %c0_i32, %c3_i32, %c4_i32, %302) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %304 = dfschedule.config.create_io(%303, %45) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %305 = dfschedule.config.dma_bd(%119, %67, %c5_i32, %c0_i32, %c4_i32, %c5_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %306 = dfschedule.config.dma_bd(%118, %67, %c4_i32, %c0_i32, %c4_i32, %c5_i32, %305) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %307 = dfschedule.config.create_io(%306, %67) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %308 = dfschedule.config.dma_bd(%subview_21, %22, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 18544 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 1 : i32,
      dim_strides = [4, 920, 224],
      dim_wraps = [61, 19, 4],
      iter_step_size = 12880 : i32,
      iter_wrap = 4 : i32
    } : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 51520>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %309 = dfschedule.config.create_io(%308, %22) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %310 = dfschedule.config.dma_bd(%124, %6, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %311 = dfschedule.config.dma_bd(%123, %6, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %310) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %312 = dfschedule.config.create_io(%311, %6) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %313 = dfschedule.config.dma_bd(%128, %28, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %314 = dfschedule.config.dma_bd(%127, %28, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %313) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %315 = dfschedule.config.create_io(%314, %28) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %316 = dfschedule.config.dma_bd(%132, %50, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %317 = dfschedule.config.dma_bd(%131, %50, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %316) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %318 = dfschedule.config.create_io(%317, %50) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %319 = dfschedule.config.dma_bd(%136, %72, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %320 = dfschedule.config.dma_bd(%135, %72, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %319) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %321 = dfschedule.config.create_io(%320, %72) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %322 = dfschedule.config.dma_bd(%141, %6, %c5_i32, %c0_i32, %c5_i32, %c7_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %323 = dfschedule.config.dma_bd(%140, %6, %c4_i32, %c0_i32, %c5_i32, %c7_i32, %322) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %324 = dfschedule.config.create_io(%323, %6) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %325 = dfschedule.config.dma_bd(%145, %28, %c5_i32, %c0_i32, %c6_i32, %c8_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %326 = dfschedule.config.dma_bd(%144, %28, %c4_i32, %c0_i32, %c6_i32, %c8_i32, %325) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %327 = dfschedule.config.create_io(%326, %28) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %328 = dfschedule.config.dma_bd(%149, %50, %c5_i32, %c0_i32, %c7_i32, %c9_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %329 = dfschedule.config.dma_bd(%148, %50, %c4_i32, %c0_i32, %c7_i32, %c9_i32, %328) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %330 = dfschedule.config.create_io(%329, %50) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %331 = dfschedule.config.dma_bd(%153, %72, %c5_i32, %c0_i32, %c8_i32, %c10_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %332 = dfschedule.config.dma_bd(%152, %72, %c4_i32, %c0_i32, %c8_i32, %c10_i32, %331) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %333 = dfschedule.config.create_io(%332, %72) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %334 = dfschedule.config.dma_bd(%subview_31, %44, %c1_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 18544 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 1 : i32,
      dim_strides = [4, 920, 224],
      dim_wraps = [61, 19, 4],
      iter_step_size = 12880 : i32,
      iter_wrap = 4 : i32
    } : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 103040>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %335 = dfschedule.config.create_io(%334, %44) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %336 = dfschedule.config.dma_bd(%158, %11, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %337 = dfschedule.config.dma_bd(%157, %11, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %336) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %338 = dfschedule.config.create_io(%337, %11) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %339 = dfschedule.config.dma_bd(%162, %33, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %340 = dfschedule.config.dma_bd(%161, %33, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %339) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %341 = dfschedule.config.create_io(%340, %33) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %342 = dfschedule.config.dma_bd(%166, %55, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %343 = dfschedule.config.dma_bd(%165, %55, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %342) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %344 = dfschedule.config.create_io(%343, %55) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %345 = dfschedule.config.dma_bd(%170, %77, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %346 = dfschedule.config.dma_bd(%169, %77, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %345) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %347 = dfschedule.config.create_io(%346, %77) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %348 = dfschedule.config.dma_bd(%175, %11, %c5_i32, %c0_i32, %c9_i32, %c3_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %349 = dfschedule.config.dma_bd(%174, %11, %c4_i32, %c0_i32, %c9_i32, %c3_i32, %348) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %350 = dfschedule.config.create_io(%349, %11) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %351 = dfschedule.config.dma_bd(%179, %33, %c5_i32, %c0_i32, %c10_i32, %c4_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %352 = dfschedule.config.dma_bd(%178, %33, %c4_i32, %c0_i32, %c10_i32, %c4_i32, %351) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %353 = dfschedule.config.create_io(%352, %33) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %354 = dfschedule.config.dma_bd(%183, %55, %c5_i32, %c0_i32, %c11_i32, %c5_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %355 = dfschedule.config.dma_bd(%182, %55, %c4_i32, %c0_i32, %c11_i32, %c5_i32, %354) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %356 = dfschedule.config.create_io(%355, %55) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %357 = dfschedule.config.dma_bd(%187, %77, %c5_i32, %c0_i32, %c12_i32, %c6_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %358 = dfschedule.config.dma_bd(%186, %77, %c4_i32, %c0_i32, %c12_i32, %c6_i32, %357) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %359 = dfschedule.config.create_io(%358, %77) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %360 = dfschedule.config.dma_bd(%subview_41, %66, %c11_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 18544 : i32,
      enable_packet = false,
      next_bd = 4294967295 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = 0 : i32,
      release_lock_id = 0 : i32,
      release_lock_val = 0 : i32,
      data_id = 1 : i32,
      dim_strides = [4, 920, 224],
      dim_wraps = [61, 19, 4],
      iter_step_size = 12880 : i32,
      iter_wrap = 4 : i32
    } : (memref<61x230x4xi8, strided<[920, 4, 1], offset: 154560>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %361 = dfschedule.config.create_io(%360, %66) {
      channel = 1,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %362 = dfschedule.config.dma_bd(%192, %16, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %363 = dfschedule.config.dma_bd(%191, %16, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %362) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %364 = dfschedule.config.create_io(%363, %16) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %365 = dfschedule.config.dma_bd(%196, %38, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %366 = dfschedule.config.dma_bd(%195, %38, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %365) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %367 = dfschedule.config.create_io(%366, %38) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %368 = dfschedule.config.dma_bd(%200, %60, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %369 = dfschedule.config.dma_bd(%199, %60, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %368) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %370 = dfschedule.config.create_io(%369, %60) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %371 = dfschedule.config.dma_bd(%204, %82, %c3_i32, %c0_i32, %c0_i32, %c-1_i32) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 2 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %372 = dfschedule.config.dma_bd(%203, %82, %c2_i32, %c0_i32, %c0_i32, %c-1_i32, %371) {
      len = 4636 : i32,
      enable_packet = false,
      next_bd = 3 : i32,
      acquire_lock_id = 0 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 1 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<61x230x4xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %373 = dfschedule.config.create_io(%372, %82) {
      channel = 0,
      direction = "S2MM",
      io_operation = "RECV",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %374 = dfschedule.config.dma_bd(%209, %16, %c5_i32, %c0_i32, %c13_i32, %c8_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %375 = dfschedule.config.dma_bd(%208, %16, %c4_i32, %c0_i32, %c13_i32, %c8_i32, %374) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %376 = dfschedule.config.create_io(%375, %16) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %377 = dfschedule.config.dma_bd(%213, %38, %c5_i32, %c0_i32, %c14_i32, %c9_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %378 = dfschedule.config.dma_bd(%212, %38, %c4_i32, %c0_i32, %c14_i32, %c9_i32, %377) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %379 = dfschedule.config.create_io(%378, %38) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %380 = dfschedule.config.dma_bd(%217, %60, %c5_i32, %c0_i32, %c15_i32, %c10_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %381 = dfschedule.config.dma_bd(%216, %60, %c4_i32, %c0_i32, %c15_i32, %c10_i32, %380) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %382 = dfschedule.config.create_io(%381, %60) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %383 = dfschedule.config.dma_bd(%221, %82, %c5_i32, %c0_i32, %c16_i32, %c11_i32) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 4 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
    %384 = dfschedule.config.dma_bd(%220, %82, %c4_i32, %c0_i32, %c16_i32, %c11_i32, %383) {
      len = 3136 : i32,
      enable_packet = true,
      next_bd = 5 : i32,
      acquire_lock_id = 5 : i32,
      acquire_lock_val = -1 : i32,
      release_lock_id = 4 : i32,
      release_lock_val = 1 : i32,
      data_id = -1 : i32
    } : (memref<7x28x16xi8>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
    %385 = dfschedule.config.create_io(%384, %82) {
      channel = 0,
      direction = "MM2S",
      io_operation = "SEND",
      enable_out_of_order = false
    } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
    %386 = dfschedule.schedule.start_io(%230, %5) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %387 = dfschedule.schedule.start_io(%233, %10) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %388 = dfschedule.schedule.start_io(%236, %15) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %389 = dfschedule.schedule.start_io(%239, %20) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %390 = dfschedule.schedule.start_io(%244, %27) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %391 = dfschedule.schedule.start_io(%247, %32) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %392 = dfschedule.schedule.start_io(%250, %37) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %393 = dfschedule.schedule.start_io(%253, %42) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %394 = dfschedule.schedule.start_io(%258, %49) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %395 = dfschedule.schedule.start_io(%261, %54) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %396 = dfschedule.schedule.start_io(%264, %59) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %397 = dfschedule.schedule.start_io(%267, %64) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %398 = dfschedule.schedule.start_io(%272, %71) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %399 = dfschedule.schedule.start_io(%275, %76) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %400 = dfschedule.schedule.start_io(%278, %81) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %401 = dfschedule.schedule.start_io(%281, %86) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %402 = dfschedule.schedule.start_io(%286, %91) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %403 = dfschedule.schedule.start_io(%289, %95) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %404 = dfschedule.schedule.start_io(%292, %99) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %405 = dfschedule.schedule.start_io(%295, %103) {flow_index = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %406 = dfschedule.schedule.start_io(%312, %125) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %407 = dfschedule.schedule.start_io(%315, %129) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %408 = dfschedule.schedule.start_io(%318, %133) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %409 = dfschedule.schedule.start_io(%321, %137) {flow_index = 6 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %410 = dfschedule.schedule.start_io(%338, %159) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %411 = dfschedule.schedule.start_io(%341, %163) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %412 = dfschedule.schedule.start_io(%344, %167) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %413 = dfschedule.schedule.start_io(%347, %171) {flow_index = 8 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %414 = dfschedule.schedule.start_io(%364, %193) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %415 = dfschedule.schedule.start_io(%367, %197) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %416 = dfschedule.schedule.start_io(%370, %201) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %417 = dfschedule.schedule.start_io(%373, %205) {flow_index = 10 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %418 = dfschedule.schedule.start_io(%298, %108) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %419 = dfschedule.schedule.start_io(%301, %112) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %420 = dfschedule.schedule.start_io(%304, %116) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %421 = dfschedule.schedule.start_io(%307, %120) {flow_index = 5 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %422 = dfschedule.schedule.start_io(%324, %142) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %423 = dfschedule.schedule.start_io(%327, %146) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %424 = dfschedule.schedule.start_io(%330, %150) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %425 = dfschedule.schedule.start_io(%333, %154) {flow_index = 7 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %426 = dfschedule.schedule.start_io(%350, %176) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %427 = dfschedule.schedule.start_io(%353, %180) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %428 = dfschedule.schedule.start_io(%356, %184) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %429 = dfschedule.schedule.start_io(%359, %188) {flow_index = 9 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %430 = dfschedule.schedule.start_io(%376, %210) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %431 = dfschedule.schedule.start_io(%379, %214) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %432 = dfschedule.schedule.start_io(%382, %218) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %433 = dfschedule.schedule.start_io(%385, %222) {flow_index = 11 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %434 = dfschedule.schedule.start_io(%227, %21) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %435 = dfschedule.schedule.start_io(%241, %43) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %436 = dfschedule.schedule.start_io(%255, %65) {flow_index = 2 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %437 = dfschedule.schedule.start_io(%269, %87) {flow_index = 3 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %438 = dfschedule.schedule.start_io(%283, %104) {flow_index = 4 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %439 = dfschedule.schedule.start_io(%309, %138) {flow_index = 6 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %440 = dfschedule.schedule.start_io(%335, %172) {flow_index = 8 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    %441 = dfschedule.schedule.start_io(%361, %206) {flow_index = 10 : i32, repeat_count = 4 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %442 = arith.index_cast %arg3 : index to i32
      %443 = arith.muli %442, %c50176_i32 : i32
      %444 = arith.addi %443, %c48_i32 : i32
      %445 = arith.addi %443, %c32_i32 : i32
      %446 = arith.addi %443, %c16_i32 : i32
      %447 = dfschedule.schedule.getbdid(%66) : (!dfschedule.tile) -> i32
      %448 = dfschedule.config.dma_bd(%subview_16, %66, %c5_i32, %444, %c4_i32, %c-1_i32) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %449 = dfschedule.config.dma_bd(%subview_16, %66, %c4_i32, %445, %c3_i32, %c-1_i32, %448) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %450 = dfschedule.config.dma_bd(%subview_16, %66, %c3_i32, %446, %c2_i32, %c-1_i32, %449) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %451 = dfschedule.config.dma_bd(%subview_16, %66, %c2_i32, %443, %c1_i32, %c-1_i32, %450) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1]>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %452 = dfschedule.config.create_io(%451, %66) {
        channel = 0,
        direction = "S2MM",
        io_operation = "RECV",
        enable_out_of_order = true
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %453 = dfschedule.schedule.start_io(%452, %447) {flow_index = 5 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
      %454 = arith.index_cast %arg3 : index to i32
      %455 = arith.muli %454, %c50176_i32 : i32
      %456 = arith.addi %455, %c48_i32 : i32
      %457 = arith.addi %455, %c32_i32 : i32
      %458 = arith.addi %455, %c16_i32 : i32
      %459 = dfschedule.schedule.getbdid(%66) : (!dfschedule.tile) -> i32
      %460 = dfschedule.config.dma_bd(%subview_26, %66, %c10_i32, %456, %c8_i32, %c-1_i32) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %461 = dfschedule.config.dma_bd(%subview_26, %66, %c9_i32, %457, %c7_i32, %c-1_i32, %460) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %462 = dfschedule.config.dma_bd(%subview_26, %66, %c8_i32, %458, %c6_i32, %c-1_i32, %461) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %463 = dfschedule.config.dma_bd(%subview_26, %66, %c7_i32, %455, %c5_i32, %c-1_i32, %462) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 200704>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %464 = dfschedule.config.create_io(%463, %66) {
        channel = 1,
        direction = "S2MM",
        io_operation = "RECV",
        enable_out_of_order = true
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %465 = dfschedule.schedule.start_io(%464, %459) {flow_index = 7 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
      %466 = arith.index_cast %arg3 : index to i32
      %467 = arith.muli %466, %c50176_i32 : i32
      %468 = arith.addi %467, %c48_i32 : i32
      %469 = arith.addi %467, %c32_i32 : i32
      %470 = arith.addi %467, %c16_i32 : i32
      %471 = dfschedule.schedule.getbdid(%44) : (!dfschedule.tile) -> i32
      %472 = dfschedule.config.dma_bd(%subview_36, %44, %c6_i32, %468, %c12_i32, %c-1_i32) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %473 = dfschedule.config.dma_bd(%subview_36, %44, %c5_i32, %469, %c11_i32, %c-1_i32, %472) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %474 = dfschedule.config.dma_bd(%subview_36, %44, %c4_i32, %470, %c10_i32, %c-1_i32, %473) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %475 = dfschedule.config.dma_bd(%subview_36, %44, %c3_i32, %467, %c9_i32, %c-1_i32, %474) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 401408>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %476 = dfschedule.config.create_io(%475, %44) {
        channel = 0,
        direction = "S2MM",
        io_operation = "RECV",
        enable_out_of_order = true
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %477 = dfschedule.schedule.start_io(%476, %471) {flow_index = 9 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
      %478 = arith.index_cast %arg3 : index to i32
      %479 = arith.muli %478, %c50176_i32 : i32
      %480 = arith.addi %479, %c48_i32 : i32
      %481 = arith.addi %479, %c32_i32 : i32
      %482 = arith.addi %479, %c16_i32 : i32
      %483 = dfschedule.schedule.getbdid(%44) : (!dfschedule.tile) -> i32
      %484 = dfschedule.config.dma_bd(%subview_46, %44, %c11_i32, %480, %c16_i32, %c-1_i32) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32) -> !dfschedule.bd_handle
      %485 = dfschedule.config.dma_bd(%subview_46, %44, %c10_i32, %481, %c15_i32, %c-1_i32, %484) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %486 = dfschedule.config.dma_bd(%subview_46, %44, %c9_i32, %482, %c14_i32, %c-1_i32, %485) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %487 = dfschedule.config.dma_bd(%subview_46, %44, %c8_i32, %479, %c13_i32, %c-1_i32, %486) {
        len = 3136 : i32,
        enable_packet = false,
        next_bd = 4294967295 : i32,
        acquire_lock_id = -1 : i32,
        acquire_lock_val = 0 : i32,
        release_lock_id = -1 : i32,
        release_lock_val = 0 : i32,
        data_id = 2 : i32,
        dim_strides = [4, 64, 7168],
        dim_wraps = [4, 28, 7],
        iter_step_size = 1792 : i32,
        iter_wrap = 4 : i32
      } : (memref<28x112x16xi8, strided<[7168, 64, 1], offset: 602112>>, !dfschedule.tile, i32, i32, i32, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
      %488 = dfschedule.config.create_io(%487, %44) {
        channel = 1,
        direction = "S2MM",
        io_operation = "RECV",
        enable_out_of_order = true
      } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
      %489 = dfschedule.schedule.start_io(%488, %483) {flow_index = 11 : i32, repeat_count = 16 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
      dfschedule.schedule.wait(%453) : (!dfschedule.event)
      dfschedule.schedule.wait(%465) : (!dfschedule.event)
      dfschedule.schedule.wait(%477) : (!dfschedule.event)
      dfschedule.schedule.wait(%489) : (!dfschedule.event)
    }
    dfschedule.schedule.wait(%225) : (!dfschedule.event)
  }
  dfschedule.dskernel_receiver @dskernel_receiver {
  }
}
