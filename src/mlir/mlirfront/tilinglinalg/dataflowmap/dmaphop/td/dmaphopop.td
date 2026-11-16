/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

/*
// --- 1. resource
%shim_engine = dataflowmap.create_engine <"shim"> {id = 0}
%shim_port_out = dataflowmap.create_port %shim_engine {direction = "out"}

%split_engine = dataflowmap.create_engine <"memtile"> {id = 1}
%split_port_in = dataflowmap.create_port %split_engine {direction = "in"}
%split_port_out_A = dataflowmap.create_port %split_engine {direction = "out", port_idx = 0}
%split_port_out_B = dataflowmap.create_port %split_engine {direction = "out", port_idx = 1}

%core_engine_A = dataflowmap.create_engine <"core"> {id = 5}
%core_port_in_A = dataflowmap.create_port %core_engine_A {direction = "in"}
%core_engine_B = dataflowmap.create_engine <"core"> {id = 6}
%core_port_in_B = dataflowmap.create_port %core_engine_B {direction = "in"}

%data_to_bcast = dataflowmap.create_data ...

// --- 2. connection
%hop_ingress = dataflowmap.create_hop %shim_port_out -> %split_port_in
%hop_egress_A = dataflowmap.create_hop %split_port_out_A -> %core_port_in_A
%hop_egress_B = dataflowmap.create_hop %split_port_out_B -> %core_port_in_B

// --- 3.path
%path_for_A = dataflowmap.create_path(%hop_ingress, %hop_egress_A) -> !dmap.path_handle
%path_for_B = dataflowmap.create_path(%hop_ingress, %hop_egress_B) -> !dmap.path_handle

// --- 4. broadcast
%broadcast_graph = dataflowmap.create_flow_graph(%path_for_A, %path_for_B) -> !dmap.graph_handle

// --- start
dataflowmap.start %data_to_bcast on %broadcast_graph

// (b) receive
dataflowmap.receive at %core_port_in_A
dataflowmap.receive at %core_port_in_B
*/
/*
builtin.module {

  // 定义一个函数来设置 "缓存并转发" 的顺序广播
  dma_hop.func @cache_and_forward_broadcast() {

    // --- 1. 定义物理 Tile 实例 (The "Where") ---
    %shim_tile = dma_hop.tile { col = 0, row = 0 } -> !dma_hop.tile
    %tile_A    = dma_hop.tile { col = 1, row = 2 } -> !dma_hop.tile
    %tile_B    = dma_hop.tile { col = 2, row = 2 } -> !dma_hop.tile
    
    // --- 2. 定义物理端口 (The "Doors") ---
    %port_shim_out = dma_hop.port on %shim_tile { ... } -> !dma_hop.port
    %port_A_in = dma_hop.port on %tile_A { ... } -> !dma_hop.port
    %port_A_out = dma_hop.port on %tile_A { ... } -> !dma_hop.port
    %port_B_in = dma_hop.port on %tile_B { ... } -> !dma_hop.port

    // --- 3. 定义 Hops (The Physical Links) ---
    %hop_1 = dma_hop.create_hop %port_shim_out -> %port_A_in -> !dma_hop.hop
    %hop_2 = dma_hop.create_hop %port_A_out -> %port_B_in -> !dma_hop.hop

    
    // --- 4. 定义 Path (The Full Route) ---
    //
    // vvvv 这里是您建议的修改 vvvv
    // 
    // "consumers" 和 "tee_points" 现在是路径的静态属性。
    // 这定义了数据流的完整拓扑。
    %serial_path = dma_hop.create_path [%hop_1, %hop_2] {
        consumers = [ %port_A_in, %port_B_in ],
        tee_points = [ %port_A_in ] // 明确指出 Tile A 需要分流
    } -> !dma_hop.path
    //
    // ^^^^ 修改结束 ^^^^
    

    // --- 5. 准备数据和 `memref` 缓冲区 (The "What") ---
    %data = memref.alloc() : memref<1024xf32, memory_space = "DDR"> 
    %buffer_A = dma_hop.alloc_buffer on %tile_A { ... } -> memref<1024xf32, ...>
    %buffer_B = dma_hop.alloc_buffer on %tile_B { ... } -> memref<1024xf32, ...>

    
    // --- 6. 执行传输 (The "How" & "When") ---
    //
    // vvvv dma_hop.push 也随之简化了 vvvv
    //
    // 它不再需要复杂的 "port -> buffer" 映射。
    // 它只需要按顺序提供缓冲区,
    // 这个顺序 *必须* 匹配 %serial_path 中 'consumers' 属性的顺序。
    // (即 %buffer_A 对应 %port_A_in, %buffer_B 对应 %port_B_in)
    dma_hop.push %data onto %serial_path
      outputs (%buffer_A, %buffer_B) 
    //
    // ^^^^ 修改结束 ^^^^
    

    // --- 7. 同步 ---
    dma_hop.sync %serial_path

    // --- 8. 释放资源 ---
    dma_hop.dealloc_buffer %buffer_A
    dma_hop.dealloc_buffer %buffer_B
    memref.dealloc %data

    return
  }
}
*/
/*
builtin.module {

  dma_hop.func @cache_and_forward_broadcast() {

    
    %shim_tile = dma_hop.tile { col = 0, row = 0 } -> !dma_hop.tile
    
    %tile_A    = dma_hop.tile { col = 1, row = 2 } -> !dma_hop.tile
    
    %tile_B    = dma_hop.tile { col = 2, row = 2 } -> !dma_hop.tile

    
    %port_shim_out = dma_hop.port on %shim_tile { 
        direction = #dma_hop<direction SEND>, 
        channel = 0 
    } -> !dma_hop.port

    %port_A_in = dma_hop.port on %tile_A { 
        direction = #dma_hop<direction RECEIVE>, 
        channel = 0 
    } -> !dma_hop.port
    
    %port_A_out = dma_hop.port on %tile_A { 
        direction = #dma_hop<direction SEND>, 
        channel = 0 
    } -> !dma_hop.port

    %port_B_in = dma_hop.port on %tile_B { 
        direction = #dma_hop<direction RECEIVE>, 
        channel = 0 
    } -> !dma_hop.port

    
    
    // Hop 1: SHIM -> Tile A
    %hop_1 = dma_hop.create_hop %port_shim_out -> %port_A_in -> !dma_hop.hop
    
    %hop_2 = dma_hop.create_hop %port_A_out -> %port_B_in -> !dma_hop.hop

    
    %serial_path = dma_hop.create_path [%hop_1, %hop_2] -> !dma_hop.path
    

    %data = memref.alloc() : memref<1024xf32, memory_space = "DDR"> 


    %buffer_A = dma_hop.alloc_buffer on %tile_A { 
        memory_space = "L1" 
    } -> memref<1024xf32, memory_space = "L1">

   
    %buffer_B = dma_hop.alloc_buffer on %tile_B { 
        memory_space = "L1" 
    } -> memref<1024xf32, memory_space = "L1">

    dma_hop.push %data onto %serial_path
      outputs(
        %port_A_in -> %buffer_A, 
        %port_B_in -> %buffer_B   
      )

      

    dma_hop.sync %serial_path


    dma_hop.dealloc_buffer %buffer_A
    dma_hop.dealloc_buffer %buffer_B
    memref.dealloc %data

    return
  }
}
*/
#ifndef DMAPHOP_OPS
#define DMAPHOP_OPS

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

include "mlir/IR/BuiltinTypes.td"  // Add this line for MemRefType
include "dmaphoptype.td"
include "dmaphopattr.td"


#ifndef AIEHLC_PUBLIC_SYMBOL_TD
#define AIEHLC_PUBLIC_SYMBOL_TD
// every aiehlc symbol is public (tilinglinalg/aiehlcsymbol.h)
def AiehlcPublicSymbol : NativeOpTrait<"PublicSymbol"> { let cppNamespace = "::aiehlc"; }
#endif
def dmaphop_FuncOp : dmaphopop<"func",[ IsolatedFromAbove,  
    SymbolTable,
    Symbol, SymbolName, AiehlcPublicSymbol,
    SingleBlock,
    NoTerminator]>
{
    let summary = "Defines a function in the Starplat Dialect";
    let description = [{This operation represents a function definiton in the Starplat dialect.
                        It contains a name, other related arguments and an optional return type.
                        It has a single region for the function body.
                        }];

  let arguments = (ins 
                    SymbolNameAttr:$sym_name,
                    TypeAttrOf<FunctionType>:$funcType
                  );

  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
}

// --- 1. Tile Definition Operation ---

def dmaphop_TileOp : dmaphopop<"tile"> {
  let summary = "Defines a physical tile instance with coordinates";
  let arguments = (ins
    StrAttr:$tiletype,
    I64Attr:$col,
    I64Attr:$row
  );
  let results = (outs dmaphoptiletype:$tile);
  let assemblyFormat = "`{` `TILETYPE` `=` $tiletype `,` `col` `=` $col `,` `row` `=` $row `}` attr-dict `->` type($tile)";
}

 
// --- 2. Port Definition Operation ---
// let hasCustomAssemblyFormat = 1; will fix the port itself generic print issue when have symbol
def dmaphop_PortOp : dmaphopop<"port",[]> {
  let summary = "Defines a logical port on a tile";
  let arguments = (ins
    dmaphoptiletype:$tile,
    StrAttr:$direction,
    SymbolNameAttr:$sym_name,
    OptionalAttr<I64Attr>:$direction_channel,
    OptionalAttr<I32Attr>:$dmapktid
  );
  let results = (outs dmaphopporttype:$port);
  let hasCustomAssemblyFormat = 1;
}
 
// --- 3. Hop Creation Operation ---
def dmaphop_create_hop : dmaphopop<"create_hop"> {
  let summary = "Creates a hop between an output port and an input port";
  let arguments = (ins
    dmaphopporttype:$source,
    dmaphopporttype:$destination
  );
  let results = (outs dmaphophoptype:$hop);
  let assemblyFormat = "$source `->` $destination attr-dict `->` type($hop)";
}

// --- 4. Path Creation Operation ---
def dmaphop_create_path : dmaphopop<"create_path"> {
  let summary = "Creates a path from a sequence of hops with producers, consumers and tee points";
  let description = [{
    Creates a complete data movement path from a sequence of hops, specifying:
    - List of hops that form the path
    - Producer ports that generate/send the data (for DMA S2MM: core -> shim)
    - Consumer ports that will receive the data (for DMA MM2S: shim -> core)
    - Tee points where data splitting occurs
    
    Semantics:
    - producers: Output ports that initiate data transfer (used for Pull/S2MM operations)
                 Example: core output ports sending data back to shim/memory
    - consumers: Input ports that receive data (used for Push/MM2S operations)
                 Example: core input ports receiving data from shim/memory
    - tee_points: Ports where data is duplicated/broadcasted to multiple destinations
  }];

  let arguments = (ins
    Variadic<dmaphophoptype>:$hops,
    ArrayAttr:$producers,              // Array of symbol refs to output ports (DMA send: S2MM)
    ArrayAttr:$consumers,              // Array of symbol refs to input ports (DMA receive: MM2S)
    ArrayAttr:$tee_points              // Array of symbol refs to ports where data splitting occurs
  );

  let results = (outs dmaphoppathtype:$path);

  let assemblyFormat = [{
    `[` $hops `]` `{`
      `producers` `=` `[` $producers `]` `,`
      `consumers` `=` `[` $consumers `]` `,`
      `tee_points` `=` `[` $tee_points `]`
    `}` attr-dict `->` type($path)
  }];
}

// --- 5. Buffer Allocation Operation ---
def dmaphop_alloc_buffer : dmaphopop<"alloc_buffer"> {
  let summary = "Allocates a buffer on a specific tile";
  let arguments = (ins
    dmaphoptiletype:$tile,
    AnyTensor:$memref_type
  );
  let results = (outs AnyTensor:$buffer);
  let assemblyFormat = "`on` $tile $memref_type `:` type($tile) type($memref_type) attr-dict `->` type($buffer)";
}

// --- 6. Push Operation ---
 
def dmaphop_push : dmaphopop<"push"> {
  let summary = "Pushes data through a path to multiple destinations";
  let arguments = (ins
    AnyTensor:$data,
    dmaphoppathtype:$path,
    Variadic<AnyTensor>:$consumer_buffers,
    Variadic<dmaphopporttype>:$consumer_ports
  );
  let traits = [SameVariadicOperandSize];
  let assemblyFormat = "$data `into` $path `consumer` `(` $consumer_buffers `at` $consumer_ports `)` `:` type($data) type($path) type($consumer_buffers) type($consumer_ports) attr-dict";
}
 
def dmaphop_pull : dmaphopop<"pull"> {
  let summary = "Pulls data through a path from multiple sources";
  let arguments = (ins
    AnyTensor:$data,
    dmaphoppathtype:$path,
    Variadic<AnyTensor>:$producer_buffers,
    Variadic<dmaphopporttype>:$producer_ports
  );
  let traits = [SameVariadicOperandSize];
  let assemblyFormat = "$data `from` $path `producer` `(` $producer_buffers `at` $producer_ports `)` `:` type($data) type($path) type($producer_buffers) type($producer_ports) attr-dict";
}

// --- 6b. Consumer Operation (DMA S2MM port mapping for receive) ---
def dmaphop_consumer : dmaphopop<"consumer", []> {
  let summary = "Declares DMA port mapping for a consumer (receive) port";
  let description = [{
    Maps a stream port (In direction) to a specific DMA S2MM channel.
    DMA direction is inferred: In port = S2MM (receive from stream to local memory).
    Has its own symbol name (e.g. @consumer0) and references the port via 'from' attr.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$from,
    I64Attr:$dma_port
  );
  let results = (outs dmaphopporttype:$result);
  let assemblyFormat = "$sym_name attr-dict";
}

// --- 6c. Producer Operation (DMA MM2S port mapping for send) ---
def dmaphop_producer : dmaphopop<"producer", []> {
  let summary = "Declares DMA port mapping for a producer (send) port";
  let description = [{
    Maps a stream port (Out direction) to a specific DMA MM2S channel.
    DMA direction is inferred: Out port = MM2S (send from local memory to stream).
    Has its own symbol name (e.g. @producer0) and references the port via 'tp' attr.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$tp,
    I64Attr:$dma_port
  );
  let results = (outs dmaphopporttype:$result);
  let assemblyFormat = "$sym_name attr-dict";
}

// --- 7. Sync Operation ---
def dmaphop_sync : dmaphopop<"sync"> {
  let summary = "Synchronizes completion of transfers on a path";
  let arguments = (ins dmaphoppathtype:$path);
  let assemblyFormat = "$path attr-dict";
}

// --- 8. Buffer Deallocation Operation ---
 
def dmaphop_dealloc_buffer : dmaphopop<"dealloc_buffer"> {
  let summary = "Deallocates a buffer";
  let arguments = (ins AnyTensor:$buffer);
  let assemblyFormat = "$buffer `:` type($buffer) attr-dict";
}
 
#endif // DMAPHOP_OPS
