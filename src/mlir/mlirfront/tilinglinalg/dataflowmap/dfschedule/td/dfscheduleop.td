/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef dfschedule_OPS
#define dfschedule_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

include "mlir/IR/BuiltinTypes.td"
include "dfscheduletype.td"

// ============================================================================
// TOP-LEVEL HOST BLOCK OPERATION
// ============================================================================

def dfschedule_HostBlockOp : dfscheduleop<"host", [
    IsolatedFromAbove,
    SymbolTable,
    Symbol,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Top-level host block containing config, kernel launch, and schedule";
  let description = [{
    This operation represents a complete host-side program block that contains:
    - Tensor/memory declarations
    - Tile declarations  
    - DMA configuration
    - Kernel group loading and launching
    - Schedule execution
    
    Example:
    ```mlir
    dfschedule.host @host_0 {
      %tensor = tensor.empty() : tensor<1024x1024xf32>
      %gmem = dfschedule.declaretensor(%tensor) : (tensor<1024x1024xf32>) -> memref<1024xf32>
      %shim0 = dfschedule.declaretile {col = 3, row = 0} : !dfschedule.tile
      // ... more ops ...
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

// ============================================================================
// CONFIG PHASE OPERATIONS
// ============================================================================

// Declare a tensor and convert to memref
def dfschedule_DeclareTensorOp : dfscheduleop<"declaretensor"> {
  let summary = "Declare a tensor and allocate corresponding device memory";
  let arguments = (ins AnyTensor:$tensor);
  let results = (outs AnyMemRef:$memref);
  let assemblyFormat = "`(` $tensor `)` attr-dict `:` `(` type($tensor) `)` `->` type($memref)";
}

// Declare a tile location
def dfschedule_DeclareTileOp : dfscheduleop<"declaretile"> {
  let summary = "Declare a physical tile by coordinates";
  let arguments = (ins
    I32Attr:$col,
    I32Attr:$row
  );
  let results = (outs dfschedule_TileHandleType:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";
}

// Create a packet handle
def dfschedule_PacketOp : dfscheduleop<"packet", [Symbol]> {
  let summary = "Create a named packet associated with data and DMA channel";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    AnyMemRef:$data,
    I32Attr:$dma_channel
  );
  let results = (outs dfschedule_PacketType:$packet);
  let assemblyFormat = "$sym_name `(` $data `)` attr-dict `:` `(` type($data) `)` `->` type($packet)";
}

// DMA Buffer Descriptor Configuration
def dfschedule_ConfigDmaBdOp : dfscheduleop<"config.dma_bd"> {
  let summary = "Configure a DMA buffer descriptor";
  let arguments = (ins
    AnyMemRef:$buffer,
    dfschedule_TileHandleType:$tile,
    I32Attr:$bd_id,
    I32Attr:$offset,
    I32Attr:$len,
    BoolAttr:$enable_packet,
    I32Attr:$packet_id,
    I32Attr:$next_bd
  );
  let results = (outs dfschedule_BdHandleType:$bd_handle);
  let hasCustomAssemblyFormat = 1;
}

// Create IO handle
def dfschedule_ConfigCreateIoOp : dfscheduleop<"config.create_io"> {
  let summary = "Create an IO handle for DMA operations";
  let arguments = (ins
    dfschedule_BdHandleType:$bd_config,
    dfschedule_TileHandleType:$tile,
    I32Attr:$channel,
    StrAttr:$direction,
    StrAttr:$io_operation  // Renamed from 'operation' to avoid conflict with getOperation()
  );
  let results = (outs dfschedule_IoHandleType:$io_handle);
  let hasCustomAssemblyFormat = 1;
}

// Load kernel group configuration
def dfschedule_LoadKernelGroupOp : dfscheduleop<"config.load_kernel_group"> {
  let summary = "Configure and load a group of kernels to be launched on tiles";
  let arguments = (ins
    Variadic<dfschedule_TileHandleType>:$tiles,
    SymbolRefArrayAttr:$callee,
    SymbolRefArrayAttr:$distributed_compute_kernel_args,
    SymbolRefArrayAttr:$distributed_args
  );
  let results = (outs dfschedule_KernelGroupType:$kernel_group);
  let hasCustomAssemblyFormat = 1;
}

// ============================================================================
// SCHEDULE PHASE OPERATIONS
// ============================================================================

// Launch kernel group
def dfschedule_LaunchKernelGroupOp : dfscheduleop<"schedule.launch_kernel_group"> {
  let summary = "Launch a pre-configured kernel group";
  let arguments = (ins dfschedule_KernelGroupType:$kernel_group);
  let results = (outs dfschedule_EventType:$event);
  let assemblyFormat = "`(` $kernel_group `)` attr-dict `:` `(` type($kernel_group) `)` `->` type($event)";
}

// Get BD ID for scheduling
def dfschedule_GetBdIdOp : dfscheduleop<"schedule.getbdid"> {
  let summary = "Get a buffer descriptor ID for runtime scheduling";
  let arguments = (ins);
  let results = (outs I32:$bd_id);
  let assemblyFormat = "attr-dict `:` type($bd_id)";
}

// Start IO operation
def dfschedule_StartIoOp : dfscheduleop<"schedule.start_io"> {
  let summary = "Start an IO operation with a given BD ID";
  let arguments = (ins
    dfschedule_IoHandleType:$io_handle,
    I32:$bd_id
  );
  let results = (outs dfschedule_EventType:$event);
  let assemblyFormat = "`(` $io_handle `,` $bd_id `)` attr-dict `:` `(` type($io_handle) `,` type($bd_id) `)` `->` type($event)";
}

// Schedule wait for multiple events
def dfschedule_ScheduleWaitOp : dfscheduleop<"schedule.wait"> {
  let summary = "Wait for multiple events to complete";
  let arguments = (ins Variadic<dfschedule_EventType>:$events);
  let results = (outs);
  let assemblyFormat = "`(` $events `)` attr-dict `:` `(` type($events) `)`";
}

// ============================================================================
// KERNEL FUNCTION OPERATIONS
// ============================================================================

// DSKernel Compute - defines a compute logic block
def dfschedule_DSKernelComputeOp : dfscheduleop<"dskernel.compute", [
    IsolatedFromAbove,
    Symbol,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Define a compute kernel logic that can be invoked from a receiver";
  let description = [{
    Defines reusable compute logic that operates on a buffer.
    The compute logic is referenced by symbol and invoked from dskernel_receiver.
    
    Example:
    ```mlir
    dfschedule.dskernel.compute @compute0 (%buf: memref<16x256xf32, "LOCAL">, %loop_count: index) -> !dfschedule.compute {
      // compute logic here
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs dfschedule_ComputeType:$compute);
  let hasCustomAssemblyFormat = 1;
}

// DSKernel Receiver - kernel function with ping-pong buffering
def dfschedule_DSKernelReceiverOp : dfscheduleop<"dskernel_receiver", [
    IsolatedFromAbove,
    Symbol,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Define a kernel receiver with ping-pong buffering pattern";
  let description = [{
    Defines a receiver kernel that:
    - Receives packets from DMA
    - Allocates ping-pong buffers
    - Initializes locks for synchronization
    - Launches DMA loop
    - Executes compute loop with ping-pong pattern
    
    Example:
    ```mlir
    dfschedule.dskernel_receiver @receiver(%arg0: !dfschedule.packet, %compute: !dfschedule.compute, %loop_count: index) {
      // receiver logic
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

// ============================================================================
// KERNEL HELPER OPERATIONS
// ============================================================================

// Get tensor from packet
def dfschedule_GetTensorOp : dfscheduleop<"gettensor"> {
  let summary = "Extract tensor reference from a packet";
  let arguments = (ins dfschedule_PacketType:$packet);
  let results = (outs AnyTensor:$tensor);
  let assemblyFormat = "`(` $packet `)` attr-dict `:` `(` type($packet) `)` `->` type($tensor)";
}

// Get DMA channel from packet
def dfschedule_GetDmaChannelOp : dfscheduleop<"getdmachannel"> {
  let summary = "Extract DMA channel from a packet";
  let arguments = (ins dfschedule_PacketType:$packet);
  let results = (outs dfschedule_DmaChannelType:$channel);
  let assemblyFormat = "`(` $packet `)` attr-dict `:` `(` type($packet) `)` `->` type($channel)";
}

// Kernel memory allocation
def dfschedule_KernelMemAllocOp : dfscheduleop<"kernel.memalloc"> {
  let summary = "Allocate local memory in kernel based on tensor shape";
  let arguments = (ins AnyTensor:$tensor);
  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = "`(` $tensor `)` attr-dict `:` `(` type($tensor) `)` `->` type($buffer)";
}

// Core compute invocation
def dfschedule_CoreComputeOp : dfscheduleop<"core.compute"> {
  let summary = "Invoke compute logic on a buffer";
  let arguments = (ins
    AnyMemRef:$buffer,
    dfschedule_ComputeType:$compute_logic
  );
  let results = (outs);
  let assemblyFormat = "`(` $buffer `,` $compute_logic `)` attr-dict `:` `(` type($buffer) `,` type($compute_logic) `)`";
}

// DSKernel Lock Init
def dfschedule_DSKernelLockInitOp : dfscheduleop<"dskernel.lock_init"> {
  let summary = "Initialize a lock with initial value and name";
  let arguments = (ins
    I64Attr:$initial_value,
    StrAttr:$name
  );
  let results = (outs dfschedule_LockType:$lock);
  let assemblyFormat = "`(` $initial_value `,` $name `)` attr-dict `->` type($lock)";
}

// DSKernel Launch DMA S2M Loop
def dfschedule_DSKernelLaunchDmaLoopOp : dfscheduleop<"dskernel.launch_dma_s2m_loop"> {
  let summary = "Launch a DMA stream-to-memory loop with ping-pong buffers";
  let arguments = (ins
    AnyMemRef:$ping,
    AnyMemRef:$pong,
    dfschedule_LockType:$ping_acq_lock,
    dfschedule_LockType:$ping_rel_lock,
    dfschedule_LockType:$pong_acq_lock,
    dfschedule_LockType:$pong_rel_lock,
    dfschedule_DmaChannelType:$channel
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
}

// DSKernel Acquire Lock
def dfschedule_DSKernelAcquireLockOp : dfscheduleop<"dskernel.acquire_lock"> {
  let summary = "Acquire a lock with specified value";
  let arguments = (ins
    dfschedule_LockType:$lock,
    I32Attr:$value
  );
  let results = (outs);
  let assemblyFormat = "`(` $lock `,` $value `)` attr-dict";
}

// DSKernel Release Lock
def dfschedule_DSKernelReleaseLockOp : dfscheduleop<"dskernel.release_lock"> {
  let summary = "Release a lock with specified value";
  let arguments = (ins
    dfschedule_LockType:$lock,
    I32Attr:$value
  );
  let results = (outs);
  let assemblyFormat = "`(` $lock `,` $value `)` attr-dict";
}

// ============================================================================
// LEGACY OPERATIONS (kept for backward compatibility)
// ============================================================================

def dfschedule_FuncOp : dfscheduleop<"func",[IsolatedFromAbove, SymbolTable, Symbol]>
{
    let summary = "Defines a function in the dfschedule Dialect";
    let description = [{This operation represents a function definition in the dfschedule dialect.
                        It contains a name, other related arguments and an optional return type.
                        It has a single region for the function body.
                        }];

  let arguments = (ins 
                    SymbolNameAttr:$sym_name,
                    TypeAttrOf<FunctionType>:$funcType
                  );

  let regions = (region AnyRegion:$body);
}

// amd.host.alloc_device_mem
def dfschedule_AllocDeviceMemOp : dfscheduleop< "alloc_device_mem", []> {
  let summary = "Allocate device global memory";
  let arguments = (ins);
  let results   = (outs AnyMemRef:$memref);
  let assemblyFormat = "attr-dict `:` type($memref)";
}

// amd.host.free_device_mem
def dfschedule_FreeDeviceMemOp : dfscheduleop< "free_device_mem", []> {
  let summary = "Free device global memory";
  let arguments = (ins AnyMemRef:$memref);
  let results   = (outs);
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

// amd.host.copy_h2d_async
def dfschedule_CopyH2DAsyncOp : dfscheduleop< "copy_h2d_async",
  [MemoryEffects<[MemWrite]>]> {
  let summary = "Async copy Host->Device";
  let arguments = (ins AnyMemRef:$host, AnyMemRef:$device);
  let results   = (outs dfschedule_EventType:$event);
  let extraClassDeclaration = [{
    static StringRef getViaAttrName() { return "via"; }  // Flow_ShimEndpointAttr
  }];
  let assemblyFormat = [{
    $host `,` $device  attr-dict `:` `(` type($host) `,` type($device) `)` `->` type($event)
  }];
}

// amd.host.copy_d2h_async
def dfschedule_CopyD2HAsyncOp : dfscheduleop< "copy_d2h_async",
  [MemoryEffects<[MemRead]>]> {
  let summary = "Async copy Device->Host";
  let arguments = (ins AnyMemRef:$device, AnyMemRef:$host);
  let results   = (outs dfschedule_EventType:$event);
  let extraClassDeclaration = [{
    static StringRef getViaAttrName() { return "via"; }
  }];
  let assemblyFormat = [{
    $device `,` $host attr-dict `:` `(` type($device) `,` type($host) `)` `->` type($event)
  }];
}

def dfschedule_GetTileHandleOp : dfscheduleop<"get_tile_handle"> {
  let summary = "Gets a runtime handle to a physical tile by coordinates.";
  let arguments = (ins
    I32Attr:$col,
    I32Attr:$row
  );
  let results = (outs dfschedule_TileHandleType:$handle);
  let assemblyFormat = "attr-dict `->` type($handle)";
}

def dfschedule_GetStreamHandleOp : dfscheduleop<"get_stream_handle"> {
  let summary = "Gets a runtime handle for a statically defined stream route contract.";
  let arguments = (ins
    SymbolRefAttr:$route // The contract: @my_routes::@route_A
  );
  let results = (outs dfschedule_StreamType:$stream);
  let assemblyFormat = "$route attr-dict `->` type($stream)";
}

def dfschedule_GetPacketIdOp : dfscheduleop<"get_packet_id"> {
  let summary = "Gets a runtime handle for a statically defined stream route contract.";
  let arguments = (ins
    SymbolRefAttr:$route // The contract: @my_routes::@route_A
  );
  let results = (outs dfschedule_PacketType:$packetid);
  let assemblyFormat = "$route attr-dict `->` type($packetid)";
}

def dfschedule_LaunchKernelAsyncOp : dfscheduleop< "launch_kernel_async"> {
  let summary = "Launch a device kernel on a tile asynchronously";
  let arguments = (ins Variadic<AnyType>:$operands);
  let results   = (outs dfschedule_EventType:$event);
  let extraClassDeclaration = [{
    static StringRef getOnTileAttrName() { return "on_tile"; } // AMD_TileAttr
    static StringRef getWaitDepsAttrName() { return "wait"; }  // Array of dfschedule_EventType
    static StringRef getCalleeAttrName() { return "callee"; }  // SymbolRefAttr
  }];
  let assemblyFormat = [{
     `(` $operands `)` attr-dict `:` `(` type($operands) `)` `->` type($event)
  }];
}

def dfschedule_WaitOp : dfscheduleop< "wait", []> {
  let summary = "Wait for one or more events to complete";
  // accept a variadic list of event operands so builders accept ValueRange
  let arguments = (ins Variadic<dfschedule_EventType>:$events);
  let results   = (outs);
  let assemblyFormat = "$events attr-dict `:` `(` type($events) `)`";
}

// ---- Ops from dskernelop.td ----

def dfschedule_LockInitOp : dfscheduleop<"lock_init"> {
  let summary = "Create/init a kernel-local lock (initial count)";
  let arguments = (ins I64Attr:$count);
  let results   = (outs dfschedule_LockType:$lock);
  let assemblyFormat = "attr-dict `(` $count `)` `->` type($lock)";
}

def dfschedule_LaunchDmaS2MLoopOp : dfscheduleop<"launch_dma_s2m_loop"> {
  let summary = "Launch a local DMA s2m ping-pong loop that listens on a packet id";
  let arguments = (ins 
    AnyMemRef:$ping, 
    AnyMemRef:$pong, 
    I32:$listen_on_packet_id, 
    dfschedule_LockType:$ping_acquire_lock, 
    dfschedule_LockType:$pong_acquire_lock,
    dfschedule_LockType:$ping_release_lock,
    dfschedule_LockType:$pong_release_lock
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$ping `:` type($ping) `,` $pong `:` type($pong) `,` $listen_on_packet_id `,` $ping_acquire_lock `,` $pong_acquire_lock `,` $ping_release_lock `,` $pong_release_lock attr-dict-with-keyword $body";
  let extraClassDeclaration = [{
    static StringRef getListenAttrName() { return "listen_on_packet_id"; }
    static StringRef getPingAcquireLockName() { return "ping_acquire_lock"; }
    static StringRef getPongAcquireLockName() { return "pong_acquire_lock"; }
    static StringRef getPingReleaseLockName() { return "ping_release_lock"; }
    static StringRef getPongReleaseLockName() { return "pong_release_lock"; }
  }];
}

def dfschedule_AcquireLockOp : dfscheduleop<"acquire_lock"> {
  let summary = "Acquire a kernel lock for a token";
  let arguments = (ins dfschedule_LockType:$lock, I32:$token);
  let results   = (outs);
  let assemblyFormat = "$lock `,` $token attr-dict";
}

def dfschedule_ReleaseLockOp : dfscheduleop<"release_lock"> {
  let summary = "Release a kernel lock for a token";
  let arguments = (ins dfschedule_LockType:$lock, I32:$token);
  let results   = (outs);
  let assemblyFormat = "$lock `,` $token attr-dict";
}

def dfschedule_ComputeOp : dfscheduleop<"compute"> {
  let summary = "Perform computation on a buffer";
  let description = [{
    This operation represents a compute kernel that processes data from a memory buffer.
  }];
  let arguments = (ins AnyMemRef:$buffer);
  let results = (outs);
  let assemblyFormat = "$buffer `:` type($buffer) attr-dict";
}

def dfschedule_L2PipelineG2sOp : dfscheduleop<"pipeline_g2s", []> {
  let summary = "Abstracts a Gmem->Smem ping-pong buffer dataflow pipeline.";
  let arguments = (ins
    AnyMemRef:$gmem_buffer,
    AnyMemRef:$ping_buffer,
    AnyMemRef:$pong_buffer
  );
  let results = (outs);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $gmem_buffer `,` $ping_buffer `,` $pong_buffer
    `yields` `(`   `)`
    $bodyRegion attr-dict
    `:` functional-type(operands, results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$gmem_buffer, "Value":$ping_buffer, "Value":$pong_buffer,"function_ref<void(OpBuilder &, Location, ValueRange)>":$bodyBuilderCallback),
    [{
      Type indexType = odsBuilder.getIndexType();
      Type smemType = ping_buffer.getType();
      Location loc = $_state.location;
      
      Region *body = $_state.addRegion();
      Block *block = new Block();
      body->push_back(block);
      
      block->addArgument(indexType, loc);
      block->addArgument(smemType, loc);
      
      OpBuilder::InsertionGuard guard(odsBuilder);
      odsBuilder.setInsertionPointToEnd(block);
      odsBuilder.create<L2YieldOp>(loc);
    }]>
  ];
}

def dfschedule_L2YieldOp : dfscheduleop<"yield", [
    Terminator,
    HasParent<"L2PipelineG2sOp">
  ]> {
  let summary = "Terminator for dfschedule.pipeline_g2s regions.";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

// ---- Block Ops for Host/Kernel Config and Schedule ----
// These operations organize the schedule into logical sections:
// - Config blocks: Setup and resource allocation (host or kernel side)
// - Schedule blocks: Execution and synchronization logic (host or kernel side)
// All four blocks use SingleBlock + NoTerminator traits, meaning:
// 1. They contain exactly one region with one block
// 2. No explicit terminator is required (operations flow through implicitly)

///*
def dfschedule_HostConfigOp : dfscheduleop<"host_config", [
//
    SymbolTable,
    Symbol,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Host-side DMA configuration block (no terminator required)";
  let description = [{
    Logical grouping for host-side resource allocation and configuration.
    Contains operations for:
    - Device memory allocation (alloc_device_mem)
    - Tile handle acquisition (get_tile_handle)
    - Stream handle lookup (get_stream_handle)
    - Packet ID retrieval (get_packet_id)
    
    This block has no explicit terminator - execution flows implicitly to the next operation.
    
    Example:
    ```mlir
    dfschedule.host_config @config_name {
      %gmem = dfschedule.alloc_device_mem : memref<1024xf32>
      %shim_handle = dfschedule.get_tile_handle {col = 2 : i32, row = 0 : i32} -> !dfschedule.tile_handle
      %streamA = dfschedule.get_stream_handle @my_routes::@route_A -> !dfschedule.stream
      %pidA = dfschedule.get_packet_id @my_routes::@route_A -> !dfschedule.packet
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

def dfschedule_KernelConfigOp : dfscheduleop<"kernel_config", [
    SingleBlock,
    NoTerminator//,
    //SymbolTable,
    //Symbol
  ]> {
  let summary = "Kernel-side buffer and lock configuration block (no terminator required)";
  let description = [{
    Logical grouping for kernel-side resource setup.
    Contains operations for:
    - Ping-pong buffer allocation (memref.alloca)
    - Lock initialization for synchronization (lock_init)
    - DMA configuration resources
    
    This block has no explicit terminator - execution flows implicitly to the next operation.
    
    Example:
    ```mlir
    dfschedule.kernel_config @config_name {
      %ping = memref.alloca() {buffer_type = "ping"} : memref<256xf32>
      %pong = memref.alloca() {buffer_type = "pong"} : memref<256xf32>
      %ping_acquire = dfschedule.lock_init(0) -> !dfschedule.lock
      %pong_acquire = dfschedule.lock_init(0) -> !dfschedule.lock
      %ping_release = dfschedule.lock_init(1) -> !dfschedule.lock
      %pong_release = dfschedule.lock_init(0) -> !dfschedule.lock
    }
    ```
  }];
  //let arguments = (ins SymbolNameAttr:$sym_name);
  let arguments = (ins StrAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

def dfschedule_HostScheduleOp : dfscheduleop<"host_schedule", [
    SingleBlock,
    NoTerminator,
    SymbolTable,
    Symbol
  ]> {
  let summary = "Host-side execution schedule block (no terminator required)";
  let description = [{
    Logical grouping for host-side execution orchestration.
    Contains operations for:
    - Asynchronous kernel launches (launch_kernel_async)
    - DMA initiation
    - Event synchronization (wait)
    
    This block has no explicit terminator - execution flows implicitly to the next operation.
    
    Example:
    ```mlir
    dfschedule.host_schedule @schedule_name {
      %evt_dma = dfschedule.launch_kernel_async(%shim_handle, %gmem, %streamA, %streamB) -> !dfschedule.event
      %evt_A = dfschedule.launch_kernel_async(%core_A_handle, %pidA) {callee = @kernel_func} -> !dfschedule.event
      %evt_B = dfschedule.launch_kernel_async(%core_B_handle, %pidB) {callee = @kernel_func} -> !dfschedule.event
      dfschedule.wait(%evt_dma, %evt_A, %evt_B) : (!dfschedule.event, !dfschedule.event, !dfschedule.event)
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

def dfschedule_KernelScheduleOp : dfscheduleop<"kernel_schedule", [
    SingleBlock,
    NoTerminator,
    SymbolTable,
    Symbol
  ]> {
  let summary = "Kernel-side execution schedule block (no terminator required)";
  let description = [{
    Logical grouping for kernel-side execution logic.
    Contains operations for:
    - DMA loops (launch_dma_s2m_loop)
    - Compute loops with ping-pong synchronization
    - Lock acquire/release operations
    
    This block has no explicit terminator - execution flows implicitly to the next operation.
    
    Example:
    ```mlir
    dfschedule.kernel_schedule @schedule_name {
      dfschedule.launch_dma_s2m_loop %ping : memref<256xf32>, %pong : memref<256xf32>, %packet_id, 
                                     %ping_acq, %pong_acq, %ping_rel, %pong_rel { ... }
      scf.for %i = %c0 to %c4 step %c1 {
        // Ping-pong buffer selection and compute logic
        dfschedule.acquire_lock %selected_lock, %lock_val
        dfschedule.compute %selected_buffer : memref<256xf32>
        dfschedule.release_lock %selected_lock, %lock_val
      }
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}
 
#endif // dfschedule_OPS
