/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
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

// Launch/invoke a host block by symbol reference
def dfschedule_LaunchHostOp : dfscheduleop<"launchhost", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "Launch/invoke a host schedule block by symbol reference";
  let description = [{
    This operation invokes a dfschedule.host block by its symbol name.
    It has memory side effects as it triggers DMA and kernel execution.
    
    Example:
    ```mlir
    func.func @main() {
      dfschedule.launchhost @host_canonicalized
      return
    }
    
    dfschedule.host @host_canonicalized {
      // schedule operations
    }
    ```
  }];
  let arguments = (ins SymbolRefAttr:$host);
  let results = (outs);
  let assemblyFormat = "$host attr-dict";
}

// ============================================================================
// CONFIG PHASE OPERATIONS
// ============================================================================

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

// Declare kernel configuration (replaces multiple packet symbols)
def dfschedule_DeclareKernelConfigOp : dfscheduleop<"declare_kernel_config", [Symbol]> {
  let summary = "Declare kernel configuration for multiple tiles";
  let description = [{
    This operation declares a unified kernel configuration that contains
    tile-specific metadata for all tiles that will run the same kernel.
    Each tile gets its configuration written to its local memory at a fixed address.
    
    Example:
    ```mlir
    dfschedule.declare_kernel_config @kernelconfig {
      tile_configs = [
        {tile_index = 0, packet_id = 0, dma_channel = 0, buffer_mode = 1, num_buffers = 2},
        {tile_index = 1, packet_id = 1, dma_channel = 0, buffer_mode = 1, num_buffers = 2}
      ]
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ArrayAttr:$tile_configs  // Array of DictionaryAttr, one per tile
  );
  let results = (outs dfschedule_KernelConfigType:$config);
  let assemblyFormat = "$sym_name attr-dict";
}


// DMA Buffer Descriptor Configuration
def dfschedule_ConfigDmaBdOp : dfscheduleop<"config.dma_bd"> {
  let summary = "Configure a DMA buffer descriptor";
  let arguments = (ins
    AnyMemRef:$buffer,
    dfschedule_TileHandleType:$tile,
    I32:$bd_id,
    I32:$offset,
    I32Attr:$len,
    BoolAttr:$enable_packet,
    I32Attr:$packet_id,
    I32Attr:$next_bd,
    DefaultValuedAttr<I32Attr, "-1">:$acquire_lock_id,
    DefaultValuedAttr<I32Attr, "-1">:$acquire_lock_val,
    DefaultValuedAttr<I32Attr, "-1">:$release_lock_id,
    DefaultValuedAttr<I32Attr, "-1">:$release_lock_val,
    DefaultValuedAttr<I32Attr, "-1">:$data_id,
    Optional<dfschedule_BdHandleType>:$linked_bd,
    // Out-of-order BD target: when >= 0, this core MM2S BD tells the shim S2MM
    // DMA engine to use this specific BD ID for the transfer (via
    // XAie_DmaSetOutofOrderBdId). Value of -1 means "not used" (backward
    // compatible).
    DefaultValuedAttr<I32Attr, "-1">:$out_of_order_bd_id,
    // Multi-dimensional addressing (optional, up to 4 dims)
    // When both present and non-empty, generates XAie_DmaSetMultiDimAddr
    // When absent, generates existing XAie_DmaSetAddrLen (no behavior change)
    OptionalAttr<I32ArrayAttr>:$dim_strides,
    OptionalAttr<I32ArrayAttr>:$dim_wraps,
    // OOO iteration parameters (optional).
    // When iter_wrap > 1, the BD re-executes iter_wrap times with address
    // advancing by iter_step_size bytes between each OOO packet arrival.
    // Uses XAie_DmaSetBdIteration(stepSize/4, iter_wrap, 0).
    // Emits __Runtime_dma_bd_config_multidim_ooo instead of _multidim.
    DefaultValuedAttr<I32Attr, "0">:$iter_step_size,
    DefaultValuedAttr<I32Attr, "0">:$iter_wrap
  );
  let results = (outs dfschedule_BdHandleType:$bd_handle);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Create IO handle
def dfschedule_ConfigCreateIoOp : dfscheduleop<"config.create_io"> {
  let summary = "Create an IO handle for DMA operations";
  let arguments = (ins
    dfschedule_BdHandleType:$bd_config,
    dfschedule_TileHandleType:$tile,
    I32Attr:$channel,
    StrAttr:$direction,
    StrAttr:$io_operation,  // Renamed from 'operation' to avoid conflict with getOperation()
    // When true, enables out-of-order BD execution on this DMA channel.
    // The shim S2MM channel will use the out_of_order_bd_id from incoming
    // packet headers to select which BD processes each transfer, bypassing
    // sequential BD chain order.
    DefaultValuedAttr<BoolAttr, "false">:$enable_out_of_order
  );
  let results = (outs dfschedule_IoHandleType:$io_handle);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Load kernel group configuration
def dfschedule_LoadKernelGroupOp : dfscheduleop<"config.load_kernel_group"> {
  let summary = "Configure and load a group of kernels to be launched on tiles";
  let description = [{
    This operation configures a kernel group with tile-specific settings.
    It can use either:
    1. NEW: kernel_config - a single unified config symbol (preferred)
    2. OLD: distributed_args - multiple packet symbols (deprecated)
    
    Example (new style):
    ```mlir
    %kg = dfschedule.config.load_kernel_group(%tile0, %tile1) {
      callee = [@dskernel_receiver],
      kernel_config = @kernelconfig,
      distributed_compute_kernel_args = [@compute0, @compute0]
    }
    ```
  }];
  let arguments = (ins
    Variadic<dfschedule_TileHandleType>:$tiles,
    SymbolRefArrayAttr:$callee,
    SymbolRefArrayAttr:$distributed_compute_kernel_args,
    OptionalAttr<FlatSymbolRefAttr>:$kernel_config,  // NEW: single unified config
    OptionalAttr<SymbolRefArrayAttr>:$distributed_args  // OLD: for backward compat
  );
  let results = (outs dfschedule_KernelGroupType:$kernel_group);
  let hasCustomAssemblyFormat = 1;
}

// ============================================================================
// CONTROL-PLANE GROUP REGISTER-WRITE OPERATIONS
// ============================================================================
// These ops coalesce identical per-tile register writes (emitted individually
// today by BD/lock lowering) into hardware control-packet group writes: one
// shim-DMA push fans out to N tiles through the stream switch. The coalescing
// decision is made by GroupRegWritePass; lowering (DfscheduleToApiPass) maps
// them onto the existing control-plane runtime.

// One-shot control fabric configuration. Lowers to __Runtime_ctrl_plan_init.
// Produces an SSA fabric handle threaded to every group_reg_write.
def dfschedule_CtrlPlanInitOp : dfscheduleop<"ctrl_plan_init"> {
  let summary = "Configure the control-packet row fabric once (broadcast/row-multicast)";
  let description = [{
    Sets up the shared control spine + per-row EAST chains used by grouped
    register writes. `rows` is an array of dictionary attrs, each carrying
    row/col_lo/col_hi describing one configured chain (ordered bottom-up).
    It lowers to __Runtime_ctrl_plan_init and yields the fabric handle.
  }];
  let arguments = (ins
    I32Attr:$shim_col,
    I32Attr:$ctrl_id,
    I32Attr:$resp_s2mm_ch,
    ArrayAttr:$rows  // array of DictionaryAttr {row, col_lo, col_hi}
  );
  let results = (outs dfschedule_CtrlFabricType:$fabric);
  let assemblyFormat = "attr-dict `:` type($fabric)";
}

// A coalesced register write delivered by control packet to a group of tiles.
// kind = "broadcast" -> __Runtime_ctrl_row_broadcast_write (whole array)
// kind = "row"       -> __Runtime_ctrl_row_whole_row_write (all cols of $row)
def dfschedule_GroupRegWriteOp : dfscheduleop<"group_reg_write"> {
  let summary = "Coalesced control-packet register write to a group of tiles";
  let description = [{
    Writes `data` (one or more 32-bit words) to tile-local byte address
    `tile_addr` on every tile of the addressed group. `kind` selects the
    delivery scope: "broadcast" (all configured tiles) or "row" (all columns of
    `row`). Fire-and-forget over the control fabric produced by
    dfschedule.ctrl_plan_init.
  }];
  let arguments = (ins
    dfschedule_CtrlFabricType:$fabric,
    StrAttr:$kind,                              // "broadcast" | "row"
    DefaultValuedAttr<I32Attr, "-1">:$row,      // target row for kind="row"
    I32Attr:$tile_addr,                         // tile-local byte address
    I32ArrayAttr:$data,                         // words to write
    I32Attr:$bd_id,
    I32Attr:$mm2s_ch,
    DefaultValuedAttr<I32Attr, "0">:$log
  );
  let results = (outs);
  let assemblyFormat = "`(` $fabric `)` attr-dict `:` `(` type($fabric) `)`";
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
  let summary = "Get a buffer descriptor ID for a given tile";
  let arguments = (ins
    dfschedule_TileHandleType:$tile
  );
  let results = (outs I32:$bd_id);
  let assemblyFormat = "`(` $tile `)` attr-dict `:` `(` type($tile) `)` `->` type($bd_id)";
}

// Start IO operation
def dfschedule_StartIoOp : dfscheduleop<"schedule.start_io"> {
  let summary = "Start an IO operation with a given BD ID";
  let arguments = (ins
    dfschedule_IoHandleType:$io_handle,
    I32:$bd_id,
    I32Attr:$flow_index,
    DefaultValuedAttr<I32Attr, "1">:$repeat_count
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

// Get DMA channel from packet
def dfschedule_GetDmaChannelOp : dfscheduleop<"getdmachannel"> {
  let summary = "Extract DMA channel from a packet";
  let arguments = (ins dfschedule_PacketType:$packet);
  let results = (outs dfschedule_DmaChannelType:$channel);
  let assemblyFormat = "`(` $packet `)` attr-dict `:` `(` type($packet) `)` `->` type($channel)";
}

// ============================================================================
// KERNEL CONFIG READ OPERATIONS (for tile local memory config)
// ============================================================================

// Read tile config from fixed local memory address
def dfschedule_KernelReadConfigOp : dfscheduleop<"kernel.read_tile_config"> {
  let summary = "Read tile configuration from fixed local memory address";
  let description = [{
    This operation reads the tile-specific configuration from a fixed address
    in the tile's local memory. The configuration was pre-written by the host
    during kernel loading.
    
    Example:
    ```mlir
    %config = dfschedule.kernel.read_tile_config : !dfschedule.tile_config
    ```
  }];
  let results = (outs dfschedule_TileConfigType:$config);
  let assemblyFormat = "attr-dict `:` type($config)";
}

// Extract packet ID from config
def dfschedule_ConfigGetPacketIdOp : dfscheduleop<"config.get_packet_id"> {
  let summary = "Get packet ID from tile config for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index  // Which flow's packet_id to get
  );
  let results = (outs I32:$packet_id);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($packet_id)";
}

// Extract DMA channel from config
def dfschedule_ConfigGetDmaChannelOp : dfscheduleop<"config.get_dma_channel"> {
  let summary = "Get DMA channel from tile config for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index  // Which flow's DMA channel to get
  );
  let results = (outs dfschedule_DmaChannelType:$channel);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($channel)";
}

// Extract buffer address from config
def dfschedule_ConfigGetBufferAddrOp : dfscheduleop<"config.get_buffer_addr"> {
  let summary = "Get ping or pong buffer address from config for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index,   // Which flow's buffer to get
    I32Attr:$buffer_index  // 0=first buffer (ping or single), 1=second buffer (pong)
  );
  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($buffer)";
}

// Get buffer mode (single vs ping-pong)
def dfschedule_ConfigGetBufferModeOp : dfscheduleop<"config.get_buffer_mode"> {
  let summary = "Get buffer mode (0=single, 1=ping-pong) for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index  // Which flow's buffer mode to get
  );
  let results = (outs I32:$mode);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($mode)";
}

// Get number of buffers
def dfschedule_ConfigGetNumBuffersOp : dfscheduleop<"config.get_num_buffers"> {
  let summary = "Get number of buffers (1 or 2) for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index  // Which flow's num buffers to get
  );
  let results = (outs I32:$num);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($num)";
}

// Get ping acquire lock ID from config
def dfschedule_ConfigGetPingAcquireLockIdOp : dfscheduleop<"config.get_ping_acquire_lock_id"> {
  let summary = "Get ping acquire lock ID from tile config for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index  // Which flow's lock ID to get
  );
  let results = (outs I32:$lock_id);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($lock_id)";
}

// Get pong acquire lock ID from config
def dfschedule_ConfigGetPongAcquireLockIdOp : dfscheduleop<"config.get_pong_acquire_lock_id"> {
  let summary = "Get pong acquire lock ID from tile config for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index  // Which flow's lock ID to get
  );
  let results = (outs I32:$lock_id);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($lock_id)";
}

// Get ping release lock ID from config
def dfschedule_ConfigGetPingReleaseLockIdOp : dfscheduleop<"config.get_ping_release_lock_id"> {
  let summary = "Get ping release lock ID from tile config for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index  // Which flow's lock ID to get
  );
  let results = (outs I32:$lock_id);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($lock_id)";
}

// Get pong release lock ID from config
def dfschedule_ConfigGetPongReleaseLockIdOp : dfscheduleop<"config.get_pong_release_lock_id"> {
  let summary = "Get pong release lock ID from tile config for a specific flow";
  let arguments = (ins 
    dfschedule_TileConfigType:$config,
    I32Attr:$flow_index  // Which flow's lock ID to get
  );
  let results = (outs I32:$lock_id);
  let assemblyFormat = "`(` $config `)` attr-dict `:` `(` type($config) `)` `->` type($lock_id)";
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
  let summary = "Initialize a lock with lock ID, initial value and name";
  let arguments = (ins
    I32:$lock_id,
    I64Attr:$initial_value,
    StrAttr:$name
  );
  let results = (outs dfschedule_LockType:$lock);
  let assemblyFormat = "`(` $lock_id `,` $initial_value `,` $name `)` attr-dict `:` type($lock_id) `->` type($lock)";
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
  let summary = "Allocate device DDR memory and copy from source memref";
  let arguments = (ins AnyMemRef:$source);
  let results   = (outs AnyMemRef:$memref);
  let assemblyFormat = "`(` $source `)` attr-dict `:` `(` type($source) `)` `->` type($memref)";
}

// amd.host.buffer_view: byte-offset view into a DDR allocation
def dfschedule_BufferViewOp : dfscheduleop< "buffer_view", []> {
  let summary = "Create a byte-offset view into a DDR buffer";
  let arguments = (ins AnyMemRef:$base, I64Attr:$offset, I64Attr:$len);
  let results   = (outs AnyMemRef:$view);
  let assemblyFormat = "$base attr-dict `:` type($base) `->` type($view)";
}

// amd.host.memref_mapping: strip storage scope from a memref to create a logical channel token
// This op is Pure (no memory effects). It serves as a zero-cost alias-cast bridge between
// a physically-scoped memref (e.g., DDR) and an unscoped logical memref token that can then
// be bound to L1 via bind_core_buffer. The SSA chain:
//   ddr_view -> memref_mapping -> bind_core_buffer -> L1 buf
// replaces the opaque unrealized_conversion_cast(tensor -> memref) bridge.
def dfschedule_MemRefMappingOp : dfscheduleop< "memref_mapping", [Pure]> {
  let summary = "Strip storage scope from memref to create a logical channel token (Pure, no-copy)";
  let arguments = (ins AnyMemRef:$source);
  let results   = (outs AnyMemRef:$mapped);
  let assemblyFormat = "$source attr-dict `:` `(` type($source) `)` `->` type($mapped)";
}

// amd.host.bind_core_buffer: bind a logical memref to pre-reserved L1 on a core tile
def dfschedule_BindCoreBufferOp : dfscheduleop< "bind_core_buffer", []> {
  let summary = "Bind a logical memref to pre-reserved L1 memory on a core tile";
  let arguments = (ins AnyMemRef:$source, dfschedule_TileHandleType:$tile, I64Attr:$offset);
  let results   = (outs AnyMemRef:$buffer);
  let assemblyFormat = "`(` $source `,` $tile `)` attr-dict `:` `(` type($source) `,` type($tile) `)` `->` type($buffer)";
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

// ============================================================================
// GENERAL KERNEL MODULE OPERATIONS
// ============================================================================
// These operations form a general-purpose kernel IR that can be lowered by
// different passes to generate either:
//   - LegacyKernelPass  -> adfkernellegacy.cc style (loop inside kernel)
//   - AdfKernelPass     -> adfkernel.cc style (loop in wrapper)
// ============================================================================

// Top-level kernel module container
def dfschedule_KernelModuleOp : dfscheduleop<"module", [
    IsolatedFromAbove,
    SymbolTable,
    Symbol,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Top-level kernel module containing config, buffers, windows, and main";
  let description = [{
    This operation represents a complete kernel module that can be lowered
    by different passes to produce either legacy or ADF-style code.
    
    Example:
    ```mlir
    dfschedule.module @kernel_driver {
      dfschedule.kernel_config_def @config {...}
      dfschedule.lock_def @LOCK_win_ping_ACQ {id = 48}
      dfschedule.buffer_def @win_ping : memref<256xvector<4xi32>, "LOCAL">
      dfschedule.window_def @window_in {...}
      dfschedule.kernel_decl @perf {...}
      dfschedule.main @main {...}
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

// Kernel configuration metadata
def dfschedule_KernelConfigDefOp : dfscheduleop<"kernel_config_def", [Symbol]> {
  let summary = "Define kernel configuration metadata";
  let description = [{
    Defines kernel metadata like name, file, buffer size, element type, etc.
    This is used by lowering passes to generate appropriate includes and constants.
    
    Example:
    ```mlir
    dfschedule.kernel_config_def @config {
      kernel_name = "perf",
      kernel_file = "perf.cc",
      buffer_size = 256 : i32,
      element_type = i32,
      vector_width = 4 : i32
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DictionaryAttr:$config_attrs
  );
  let assemblyFormat = "$sym_name $config_attrs attr-dict";
}

// Lock definition
def dfschedule_LockDefOp : dfscheduleop<"lock_def", [Symbol]> {
  let summary = "Define a named lock with ID";
  let description = [{
    Defines a named lock that can be referenced by windows and kernel operations.
    The lock ID is used for hardware lock allocation.
    
    Example:
    ```mlir
    dfschedule.lock_def @LOCK_win_ping_ACQ {id = 48 : i32}
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I32Attr:$id
  );
  let assemblyFormat = "$sym_name attr-dict";
}

// Buffer definition
def dfschedule_BufferDefOp : dfscheduleop<"buffer_def", [Symbol]> {
  let summary = "Define a named buffer with type";
  let description = [{
    Defines a named buffer (typically in local memory) that can be referenced
    by windows. Used for ping-pong buffer pairs.
    
    Example:
    ```mlir
    dfschedule.buffer_def @win_ping : memref<256xvector<4xi32>, "LOCAL">
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$buffer_type
  );
  let assemblyFormat = "$sym_name `:` $buffer_type attr-dict";
}

// Window definition (abstract ping-pong window)
def dfschedule_WindowDefOp : dfscheduleop<"window_def", [Symbol]> {
  let summary = "Define an abstract window combining ping-pong buffers with locks";
  let description = [{
    Defines a window abstraction that combines:
    - Direction (in/out)
    - Ping/pong buffer references
    - Acquire/release lock references
    - Buffer size and async mode
    
    This abstraction allows different lowering strategies based on iteration_style.
    
    Example:
    ```mlir
    dfschedule.window_def @window_in {
      direction = "in",
      ping_buffer = @win_ping,
      pong_buffer = @win_pong,
      acquire_lock = @LOCK_win_ping_ACQ,
      release_lock = @LOCK_win_pong_REL,
      buffer_size = 256 : i32,
      async = true
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DictionaryAttr:$window_attrs
  );
  let assemblyFormat = "$sym_name $window_attrs attr-dict";
}

// Kernel declaration with iteration style
def dfschedule_KernelDeclOp : dfscheduleop<"kernel_decl", [Symbol]> {
  let summary = "Declare a kernel with inputs, outputs, and iteration style";
  let description = [{
    Declares the kernel interface with:
    - Input window references
    - Output window references
    - iteration_style: "internal" (loop in kernel) or "external" (loop in wrapper)
    
    The iteration_style determines how the lowering pass generates code.
    
    Example:
    ```mlir
    dfschedule.kernel_decl @perf {
      inputs = [@window_in],
      outputs = [@window_out],
      iteration_style = "internal"
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DictionaryAttr:$decl_attrs
  );
  let assemblyFormat = "$sym_name $decl_attrs attr-dict";
}

// Main entry point
def dfschedule_KernelMainOp : dfscheduleop<"main", [
    Symbol,
    SingleBlock
  ]> {
  let summary = "Kernel main entry point";
  let description = [{
    The main entry point for the kernel driver. Contains:
    - Sync buffer allocation
    - Window initialization
    - Kernel invocation
    - Completion signaling
    - Terminated by dfschedule.kernel_return
    
    Example:
    ```mlir
    dfschedule.main @main {
      %sync = dfschedule.alloc_sync_buffer {size = 8}
      %win = dfschedule.window_init(@window_in)
      dfschedule.kernel_invoke @perf(%win)
      dfschedule.done
      dfschedule.kernel_return
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

// Allocate sync buffer
def dfschedule_AllocSyncBufferOp : dfscheduleop<"alloc_sync_buffer"> {
  let summary = "Allocate a sync buffer for iteration control";
  let arguments = (ins I32Attr:$size);
  let results = (outs dfschedule_SyncBufferType:$buffer);
  let assemblyFormat = "attr-dict `:` type($buffer)";
}

// Write to sync buffer
def dfschedule_SyncBufferWriteOp : dfscheduleop<"sync_buffer_write"> {
  let summary = "Write a value to sync buffer at specified index";
  let arguments = (ins
    dfschedule_SyncBufferType:$buffer,
    I32:$value,
    I32Attr:$index
  );
  let assemblyFormat = "`(` $buffer `,` $value `)` attr-dict";
}

// Debug logging
def dfschedule_LogOp : dfscheduleop<"log"> {
  let summary = "Log a debug value";
  let arguments = (ins I32:$value);
  let assemblyFormat = "`(` $value `)` attr-dict";
}

// Initialize window
def dfschedule_WindowInitOp : dfscheduleop<"window_init"> {
  let summary = "Initialize a window from a window definition";
  let description = [{
    Initializes a window and returns a handle for kernel invocation.
    The lowering depends on iteration_style:
    - "internal": generates window_init() + get_async_window()
    - "external": generates _io_buffer<> initialization
  }];
  let arguments = (ins SymbolRefAttr:$window_ref);
  let results = (outs AnyType:$window);
  let assemblyFormat = "`(` $window_ref `)` attr-dict `:` type($window)";
}

// Invoke kernel
def dfschedule_KernelInvokeOp : dfscheduleop<"kernel_invoke"> {
  let summary = "Invoke a kernel with window arguments";
  let description = [{
    Invokes the kernel with the given window arguments.
    The lowering depends on iteration_style:
    - "internal": single kernel call (loop is inside kernel)
    - "external": while loop with setData/acquire/release per iteration
  }];
  let arguments = (ins
    SymbolRefAttr:$kernel_ref,
    Variadic<AnyType>:$args
  );
  let assemblyFormat = "$kernel_ref `(` $args `)` attr-dict `:` `(` type($args) `)`";
}

// Signal completion
def dfschedule_DoneOp : dfscheduleop<"done"> {
  let summary = "Signal kernel completion";
  let assemblyFormat = "attr-dict";
}

// Return from kernel main
def dfschedule_KernelReturnOp : dfscheduleop<"kernel_return", [Terminator]> {
  let summary = "Return from kernel main";
  let assemblyFormat = "attr-dict";
}
 
#endif // dfschedule_OPS
