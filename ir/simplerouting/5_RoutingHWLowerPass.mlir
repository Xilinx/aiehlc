module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.control_packet_mode = 0 : i64, routing.control_plan_dedicated_shim = 0 : i64, routing.control_plan_group_reg_write = 0 : i64, routing.control_plan_op_control_packet = 0 : i64, routing.control_plan_shim_bd_ctrl = 0 : i64, routing.fullconnect_auto = 0 : i64, routing.kernel_config_offload = 0 : i64, routing.output_tile_major = 0 : i64} {
  emitc.include <"xaiengine.h">
  emitc.func private @XAie_TileLoc(i32, i32) -> !emitc.opaque<"XAie_LocType">
  emitc.func private @XAie_EnableShimDmaToAieStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_EnableAieToShimDmaStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_StrmConnCctEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
  emitc.func private @XAie_StrmPktSwSlaveSlotEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
  emitc.func private @XAie_StrmPktSwMstrPortEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
  emitc.func private @XAie_StrmPktSwSlavePortEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
  func.func @routing(%arg0: !emitc.ptr<!emitc.opaque<"XAie_DevInst">>, %arg1: memref<230x230x4xi8>, %arg2: memref<64x7x7x4xi8>, %arg3: memref<112x112x64xi8>) {
    %0 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(16, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %1 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(15, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %2 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(14, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %3 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(13, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %4 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(12, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %5 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(11, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %6 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(10, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %7 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(9, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %8 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(8, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %9 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(7, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %10 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(6, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %11 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(5, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %12 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(4, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %13 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(3, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %14 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(2, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %15 = "emitc.constant"() <{value = #emitc.opaque<"XAIE_SS_PKT_DONOT_DROP_HEADER">}> : () -> !emitc.ptr<i8>
    %16 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %17 = "emitc.constant"() <{value = #emitc.opaque<"NONE">}> : () -> !emitc.ptr<i8>
    %18 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(1, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %19 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(0, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %20 = "emitc.constant"() <{value = #emitc.opaque<"WEST">}> : () -> !emitc.ptr<i8>
    %21 = "emitc.constant"() <{value = #emitc.opaque<"EAST">}> : () -> !emitc.ptr<i8>
    %22 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %23 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %24 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %25 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %26 = "emitc.constant"() <{value = #emitc.opaque<"DMA">}> : () -> !emitc.ptr<i8>
    %27 = "emitc.constant"() <{value = #emitc.opaque<"SOUTH">}> : () -> !emitc.ptr<i8>
    %28 = "emitc.constant"() <{value = #emitc.opaque<"NORTH">}> : () -> !emitc.ptr<i8>
    %29 = "emitc.constant"() <{value = true}> : () -> i1
    %30 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %31 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %32 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %33 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "\0A//round is 0 hw split in : col -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%33, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %34, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %36 = emitc.call @XAie_TileLoc(%33, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %37 = emitc.call @XAie_StrmConnCctEnable(%arg0, %36, %27, %30, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%33, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @XAie_StrmConnCctEnable(%arg0, %38, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%33, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @XAie_StrmConnCctEnable(%arg0, %40, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %42 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %43 = emitc.call @XAie_StrmConnCctEnable(%arg0, %42, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %44 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %45 = emitc.call @XAie_StrmConnCctEnable(%arg0, %44, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%33, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @XAie_StrmConnCctEnable(%arg0, %46, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %48 = emitc.call @XAie_TileLoc(%33, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %49 = emitc.call @XAie_StrmConnCctEnable(%arg0, %48, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %50 = emitc.call @XAie_TileLoc(%33, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %51 = emitc.call @XAie_StrmConnCctEnable(%arg0, %50, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%33, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @XAie_StrmConnCctEnable(%arg0, %52, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %54 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %55 = emitc.call @XAie_StrmConnCctEnable(%arg0, %54, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 1 hw split in : col -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%32, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %34, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %36 = emitc.call @XAie_TileLoc(%32, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %37 = emitc.call @XAie_StrmConnCctEnable(%arg0, %36, %27, %30, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%32, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @XAie_StrmConnCctEnable(%arg0, %38, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%32, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @XAie_StrmConnCctEnable(%arg0, %40, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %42 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %43 = emitc.call @XAie_StrmConnCctEnable(%arg0, %42, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %44 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %45 = emitc.call @XAie_StrmConnCctEnable(%arg0, %44, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%32, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @XAie_StrmConnCctEnable(%arg0, %46, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %48 = emitc.call @XAie_TileLoc(%32, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %49 = emitc.call @XAie_StrmConnCctEnable(%arg0, %48, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %50 = emitc.call @XAie_TileLoc(%32, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %51 = emitc.call @XAie_StrmConnCctEnable(%arg0, %50, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%32, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @XAie_StrmConnCctEnable(%arg0, %52, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %54 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %55 = emitc.call @XAie_StrmConnCctEnable(%arg0, %54, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 2 hw split in : col -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %34, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %36 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %37 = emitc.call @XAie_StrmConnCctEnable(%arg0, %36, %27, %30, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%31, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @XAie_StrmConnCctEnable(%arg0, %38, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%31, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @XAie_StrmConnCctEnable(%arg0, %40, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %42 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %43 = emitc.call @XAie_StrmConnCctEnable(%arg0, %42, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %44 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %45 = emitc.call @XAie_StrmConnCctEnable(%arg0, %44, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%31, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @XAie_StrmConnCctEnable(%arg0, %46, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %48 = emitc.call @XAie_TileLoc(%31, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %49 = emitc.call @XAie_StrmConnCctEnable(%arg0, %48, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %50 = emitc.call @XAie_TileLoc(%31, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %51 = emitc.call @XAie_StrmConnCctEnable(%arg0, %50, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%31, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @XAie_StrmConnCctEnable(%arg0, %52, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %54 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %55 = emitc.call @XAie_StrmConnCctEnable(%arg0, %54, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 3 hw split in : col -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %34, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %36 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %37 = emitc.call @XAie_StrmConnCctEnable(%arg0, %36, %27, %30, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%30, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @XAie_StrmConnCctEnable(%arg0, %38, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%30, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @XAie_StrmConnCctEnable(%arg0, %40, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %42 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %43 = emitc.call @XAie_StrmConnCctEnable(%arg0, %42, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %44 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %45 = emitc.call @XAie_StrmConnCctEnable(%arg0, %44, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%30, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @XAie_StrmConnCctEnable(%arg0, %46, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %48 = emitc.call @XAie_TileLoc(%30, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %49 = emitc.call @XAie_StrmConnCctEnable(%arg0, %48, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %50 = emitc.call @XAie_TileLoc(%30, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %51 = emitc.call @XAie_StrmConnCctEnable(%arg0, %50, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%30, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @XAie_StrmConnCctEnable(%arg0, %52, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %54 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %55 = emitc.call @XAie_StrmConnCctEnable(%arg0, %54, %27, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 0 hw split in : row -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%33, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %34, %22) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %36 = emitc.call @XAie_TileLoc(%33, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %37 = emitc.call @XAie_StrmConnCctEnable(%arg0, %36, %27, %22, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%33, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @XAie_StrmConnCctEnable(%arg0, %38, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%33, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @XAie_StrmConnCctEnable(%arg0, %40, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %42 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %43 = emitc.call @XAie_StrmConnCctEnable(%arg0, %42, %27, %32, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %44 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %45 = emitc.call @XAie_StrmConnCctEnable(%arg0, %44, %27, %32, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @XAie_StrmConnCctEnable(%arg0, %46, %20, %33, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %48 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %49 = emitc.call @XAie_StrmConnCctEnable(%arg0, %48, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %50 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %51 = emitc.call @XAie_StrmConnCctEnable(%arg0, %50, %20, %33, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @XAie_StrmConnCctEnable(%arg0, %52, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %54 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %55 = emitc.call @XAie_StrmConnCctEnable(%arg0, %54, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %56 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %57 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %56, %26, %33, %33, %18, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %58 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %56, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %59 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %56, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %60 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %61 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %60, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %62 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %60, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %63 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %60, %26, %33, %33, %14, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %64 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %60, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %65 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %60, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %66 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %67 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %66, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %68 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %66, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %69 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %66, %26, %33, %33, %13, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %70 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %66, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %71 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %66, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %72 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %73 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %72, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %74 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %72, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %75 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %72, %26, %33, %33, %12, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %76 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %72, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %77 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %78 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%arg0, %77, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %79 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %80 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %79, %27, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %81 = emitc.call @XAie_TileLoc(%30, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %82 = emitc.call @XAie_StrmConnCctEnable(%arg0, %81, %28, %32, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %83 = emitc.call @XAie_TileLoc(%30, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %84 = emitc.call @XAie_StrmConnCctEnable(%arg0, %83, %28, %32, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %85 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %86 = emitc.call @XAie_StrmConnCctEnable(%arg0, %85, %28, %32, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 1 hw split in : row -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%32, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %34, %22) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %36 = emitc.call @XAie_TileLoc(%32, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %37 = emitc.call @XAie_StrmConnCctEnable(%arg0, %36, %27, %22, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%32, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @XAie_StrmConnCctEnable(%arg0, %38, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%32, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @XAie_StrmConnCctEnable(%arg0, %40, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %42 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %43 = emitc.call @XAie_StrmConnCctEnable(%arg0, %42, %27, %32, %20, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %44 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %45 = emitc.call @XAie_StrmConnCctEnable(%arg0, %44, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @XAie_StrmConnCctEnable(%arg0, %46, %21, %31, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %48 = emitc.call @XAie_TileLoc(%33, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %49 = emitc.call @XAie_StrmConnCctEnable(%arg0, %48, %27, %32, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %50 = emitc.call @XAie_TileLoc(%32, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %51 = emitc.call @XAie_StrmConnCctEnable(%arg0, %50, %27, %32, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%32, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @XAie_StrmConnCctEnable(%arg0, %52, %27, %32, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %54 = emitc.call @XAie_TileLoc(%31, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %55 = emitc.call @XAie_StrmConnCctEnable(%arg0, %54, %20, %33, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %56 = emitc.call @XAie_TileLoc(%31, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %57 = emitc.call @XAie_StrmConnCctEnable(%arg0, %56, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%30, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @XAie_StrmConnCctEnable(%arg0, %58, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %60 = emitc.call @XAie_TileLoc(%33, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %61 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %60, %26, %33, %33, %11, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %62 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %60, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %63 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %60, %21, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%32, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %64, %20, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %66 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %64, %20, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %64, %26, %33, %33, %10, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %68 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %64, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %69 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %64, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %70 = emitc.call @XAie_TileLoc(%31, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %71 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %70, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %72 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %70, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %73 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %70, %26, %33, %33, %9, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %74 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %70, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %75 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %70, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %76 = emitc.call @XAie_TileLoc(%30, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %77 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %76, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %78 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %76, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %79 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %76, %26, %33, %33, %8, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %80 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %76, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %81 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %82 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%arg0, %81, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %83 = emitc.call @XAie_TileLoc(%30, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %84 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %83, %27, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %85 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %86 = emitc.call @XAie_StrmConnCctEnable(%arg0, %85, %28, %32, %27, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %87 = emitc.call @XAie_TileLoc(%30, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %88 = emitc.call @XAie_StrmConnCctEnable(%arg0, %87, %28, %31, %27, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %89 = emitc.call @XAie_TileLoc(%30, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %90 = emitc.call @XAie_StrmConnCctEnable(%arg0, %89, %28, %31, %27, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %91 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %92 = emitc.call @XAie_StrmConnCctEnable(%arg0, %91, %28, %31, %27, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 2 hw split in : row -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %34, %22) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %36 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %37 = emitc.call @XAie_StrmConnCctEnable(%arg0, %36, %27, %22, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%31, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @XAie_StrmConnCctEnable(%arg0, %38, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%31, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @XAie_StrmConnCctEnable(%arg0, %40, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %42 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %43 = emitc.call @XAie_StrmConnCctEnable(%arg0, %42, %27, %32, %20, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %44 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %45 = emitc.call @XAie_StrmConnCctEnable(%arg0, %44, %21, %31, %20, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @XAie_StrmConnCctEnable(%arg0, %46, %21, %30, %28, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %48 = emitc.call @XAie_TileLoc(%33, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %49 = emitc.call @XAie_StrmConnCctEnable(%arg0, %48, %27, %31, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %50 = emitc.call @XAie_TileLoc(%33, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %51 = emitc.call @XAie_StrmConnCctEnable(%arg0, %50, %27, %32, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%33, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @XAie_StrmConnCctEnable(%arg0, %52, %27, %32, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %54 = emitc.call @XAie_TileLoc(%32, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %55 = emitc.call @XAie_StrmConnCctEnable(%arg0, %54, %20, %33, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %56 = emitc.call @XAie_TileLoc(%32, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %57 = emitc.call @XAie_StrmConnCctEnable(%arg0, %56, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%31, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @XAie_StrmConnCctEnable(%arg0, %58, %20, %33, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %60 = emitc.call @XAie_TileLoc(%31, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %61 = emitc.call @XAie_StrmConnCctEnable(%arg0, %60, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %62 = emitc.call @XAie_TileLoc(%30, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %63 = emitc.call @XAie_StrmConnCctEnable(%arg0, %62, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%33, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %64, %26, %33, %33, %7, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %66 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %64, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %64, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %68 = emitc.call @XAie_TileLoc(%32, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %69 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %68, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %70 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %68, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %71 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %68, %26, %33, %33, %6, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %72 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %68, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %73 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %68, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %74 = emitc.call @XAie_TileLoc(%31, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %75 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %74, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %76 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %74, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %77 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %74, %26, %33, %33, %5, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %78 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %74, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %79 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %74, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %80 = emitc.call @XAie_TileLoc(%30, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %81 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %80, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %82 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %80, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %83 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %80, %26, %33, %33, %4, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %84 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %80, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %85 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %86 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%arg0, %85, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %87 = emitc.call @XAie_TileLoc(%30, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %88 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %87, %27, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %89 = emitc.call @XAie_TileLoc(%30, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %90 = emitc.call @XAie_StrmConnCctEnable(%arg0, %89, %28, %32, %27, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %91 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %92 = emitc.call @XAie_StrmConnCctEnable(%arg0, %91, %28, %31, %20, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %93 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %94 = emitc.call @XAie_StrmConnCctEnable(%arg0, %93, %21, %31, %27, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %95 = emitc.call @XAie_TileLoc(%31, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %96 = emitc.call @XAie_StrmConnCctEnable(%arg0, %95, %28, %31, %27, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %97 = emitc.call @XAie_TileLoc(%31, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %98 = emitc.call @XAie_StrmConnCctEnable(%arg0, %97, %28, %31, %27, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %99 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %100 = emitc.call @XAie_StrmConnCctEnable(%arg0, %99, %28, %31, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 3 hw split in : row -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %34, %22) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %36 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %37 = emitc.call @XAie_StrmConnCctEnable(%arg0, %36, %27, %22, %28, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%30, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @XAie_StrmConnCctEnable(%arg0, %38, %27, %30, %28, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%30, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @XAie_StrmConnCctEnable(%arg0, %40, %27, %30, %28, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %42 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %43 = emitc.call @XAie_StrmConnCctEnable(%arg0, %42, %27, %30, %20, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %44 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %45 = emitc.call @XAie_StrmConnCctEnable(%arg0, %44, %21, %30, %20, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @XAie_StrmConnCctEnable(%arg0, %46, %21, %30, %28, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %48 = emitc.call @XAie_TileLoc(%32, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %49 = emitc.call @XAie_StrmConnCctEnable(%arg0, %48, %27, %31, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %50 = emitc.call @XAie_TileLoc(%33, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %51 = emitc.call @XAie_StrmConnCctEnable(%arg0, %50, %21, %32, %28, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%33, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @XAie_StrmConnCctEnable(%arg0, %52, %27, %31, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %54 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %55 = emitc.call @XAie_StrmConnCctEnable(%arg0, %54, %27, %32, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %56 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %57 = emitc.call @XAie_StrmConnCctEnable(%arg0, %56, %27, %32, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @XAie_StrmConnCctEnable(%arg0, %58, %20, %33, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %60 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %61 = emitc.call @XAie_StrmConnCctEnable(%arg0, %60, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %62 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %63 = emitc.call @XAie_StrmConnCctEnable(%arg0, %62, %20, %33, %21, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @XAie_StrmConnCctEnable(%arg0, %64, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %66 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %67 = emitc.call @XAie_StrmConnCctEnable(%arg0, %66, %20, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %68 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %69 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %68, %26, %33, %33, %3, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %70 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %68, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %71 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %68, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %72 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %73 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %72, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %74 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %72, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %75 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %72, %26, %33, %33, %2, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %76 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %72, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %77 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %72, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %78 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %79 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %78, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %80 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %78, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %81 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %78, %26, %33, %33, %1, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %82 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %78, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %83 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %78, %21, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %84 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %85 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %84, %20, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %86 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %84, %20, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %87 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %84, %26, %33, %33, %0, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %88 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %84, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %89 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %90 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%arg0, %89, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %91 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %92 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %91, %27, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %93 = emitc.call @XAie_TileLoc(%30, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %94 = emitc.call @XAie_StrmConnCctEnable(%arg0, %93, %28, %32, %27, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %95 = emitc.call @XAie_TileLoc(%30, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %96 = emitc.call @XAie_StrmConnCctEnable(%arg0, %95, %28, %31, %20, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %97 = emitc.call @XAie_TileLoc(%31, %25) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %98 = emitc.call @XAie_StrmConnCctEnable(%arg0, %97, %21, %31, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %99 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %100 = emitc.call @XAie_StrmConnCctEnable(%arg0, %99, %28, %32, %27, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %101 = emitc.call @XAie_TileLoc(%31, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %102 = emitc.call @XAie_StrmConnCctEnable(%arg0, %101, %28, %30, %27, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %103 = emitc.call @XAie_TileLoc(%31, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %104 = emitc.call @XAie_StrmConnCctEnable(%arg0, %103, %28, %30, %27, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %105 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %106 = emitc.call @XAie_StrmConnCctEnable(%arg0, %105, %28, %30, %27, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    return
  }
}
