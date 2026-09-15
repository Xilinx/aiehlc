/******************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
#ifndef _TYPE_dfschedule_
#define _TYPE_dfschedule_
include "dfscheduleattr.td"

//=== dfschedule Types ===//
def dfschedule_EventType:dfscheduleype<"Event"> {
  let mnemonic = "event";
  let summary = "Opaque async event type for host runtime";
}

def dfschedule_StreamType : dfscheduleype<"Stream"> {
  let mnemonic = "stream";
  let summary = "Host-visible stream handle carrying element type";
}

def dfschedule_PacketType : dfscheduleype<"Packet"> {
  let mnemonic = "packet";
  let summary = "Host-visible packet handle carrying element type";
}

//=== dfscheduleDMA Types/Attrs ===//
def dfschedule_LockType : dfscheduleype<"Lock"> {
  let mnemonic = "lock";
  let summary = "Abstract device lock handle";
}

//=== Tile Types ===//
def dfschedule_TileHandleType : dfscheduleype<"Tile"> {
  let mnemonic = "tile";
  let summary = "Tile handle for physical tile location";
}

//=== Kernel Group Types ===//
def dfschedule_KernelGroupType : dfscheduleype<"KernelGroup"> {
  let mnemonic = "kernelgroup";
  let summary = "Handle to a group of kernel instances to be launched together";
}

//=== Kernel Configuration Types ===//
def dfschedule_KernelConfigType : dfscheduleype<"KernelConfig"> {
  let mnemonic = "kernel_config";
  let summary = "Handle to kernel configuration for all tiles";
}

def dfschedule_TileConfigType : dfscheduleype<"TileConfig"> {
  let mnemonic = "tile_config";
  let summary = "Handle to tile-specific configuration in local memory";
}

//=== Compute Type (for kernel compute logic reference) ===//
def dfschedule_ComputeType : dfscheduleype<"Compute"> {
  let mnemonic = "compute";
  let summary = "Reference to a compute kernel logic block";
}

//=== DMA Configuration Types ===//
def dfschedule_BdHandleType : dfscheduleype<"BdHandle"> {
  let mnemonic = "bd_handle";
  let summary = "Buffer descriptor handle for DMA configuration";
}

def dfschedule_IoHandleType : dfscheduleype<"IoHandle"> {
  let mnemonic = "io_handle";
  let summary = "IO handle for DMA channel operations";
}

//=== DMA Channel Type ===//
def dfschedule_DmaChannelType : dfscheduleype<"DmaChannel"> {
  let mnemonic = "dma_channel";
  let summary = "DMA channel identifier";
}

//=== Sync Buffer Type ===//
def dfschedule_SyncBufferType : dfscheduleype<"SyncBuffer"> {
  let mnemonic = "sync_buffer";
  let summary = "Sync buffer for inter-iteration control signals";
}

//=== Control-plane Fabric Type ===//
// SSA handle to a configured control-packet row fabric (runtime
// __Runtime_CtrlRowFabric). Produced by dfschedule.ctrl_plan_init and threaded
// into dfschedule.group_reg_write so grouped register writes ride the fabric.
def dfschedule_CtrlFabricType : dfscheduleype<"CtrlFabric"> {
  let mnemonic = "ctrl_fabric";
  let summary = "Handle to a configured control-packet broadcast/row-multicast fabric";
}

//=== Window Types (for kernel I/O abstraction) ===//
def dfschedule_InputWindowType : dfscheduleype<"InputWindow"> {
  let mnemonic = "input_window";
  let summary = "Abstract input window type with element type parameter";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def dfschedule_OutputWindowType : dfscheduleype<"OutputWindow"> {
  let mnemonic = "output_window";
  let summary = "Abstract output window type with element type parameter";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

#endif //_TYPE_dfschedule_