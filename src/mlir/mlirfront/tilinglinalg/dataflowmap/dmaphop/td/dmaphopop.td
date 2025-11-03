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

  // 定义一个函数来设置 "缓存并转发" 的顺序广播
  dma_hop.func @cache_and_forward_broadcast() {

    // --- 1. 定义物理 Tile 实例 (The "Where") ---
    
    // 源: "mem tile" 接口, 比如 SHIM Tile (连接到 DDR)
    %shim_tile = dma_hop.tile { col = 0, row = 0 } -> !dma_hop.tile
    
    // 目的地 1: 中间 "缓存" Tile
    %tile_A    = dma_hop.tile { col = 1, row = 2 } -> !dma_hop.tile
    
    // 目的地 2: 最终 "目的地" Tile
    %tile_B    = dma_hop.tile { col = 2, row = 2 } -> !dma_hop.tile

    
    // --- 2. 定义物理端口 (The "Doors") ---
    
    // SHIM 的 *输出* 端口 (发送数据)
    %port_shim_out = dma_hop.port on %shim_tile { 
        direction = #dma_hop<direction SEND>, 
        channel = 0 
    } -> !dma_hop.port

    // Tile A 的 *输入* 端口 (接收数据)
    %port_A_in = dma_hop.port on %tile_A { 
        direction = #dma_hop<direction RECEIVE>, 
        channel = 0 
    } -> !dma_hop.port
    
    // Tile A 的 *输出* 端口 (转发数据)
    %port_A_out = dma_hop.port on %tile_A { 
        direction = #dma_hop<direction SEND>, 
        channel = 0 
    } -> !dma_hop.port

    // Tile B 的 *输入* 端口 (接收数据)
    %port_B_in = dma_hop.port on %tile_B { 
        direction = #dma_hop<direction RECEIVE>, 
        channel = 0 
    } -> !dma_hop.port

    
    // --- 3. 定义 Hops (The Physical Links) ---
    
    // Hop 1: SHIM -> Tile A
    %hop_1 = dma_hop.create_hop %port_shim_out -> %port_A_in -> !dma_hop.hop
    
    // Hop 2: Tile A -> Tile B (顺序广播的下一跳)
    %hop_2 = dma_hop.create_hop %port_A_out -> %port_B_in -> !dma_hop.hop

    
    // --- 4. 定义 Path (The Full Route) ---
    
    // 这个 path 定义了数据流的 *物理序列*
    // 路径是线性的: SHIM -> A -> B
    %serial_path = dma_hop.create_path [%hop_1, %hop_2] -> !dma_hop.path

    
    // --- 5. 准备数据和 `memref` 缓冲区 (The "What") ---
    
    // %data 是源数据, 位于 "mem tile" (DDR) 中
    %data = memref.alloc() : memref<1024xf32, memory_space = "DDR"> 

    // %buffer_A 是在 Tile A 的 L1 内存中的 "缓存" 
    %buffer_A = dma_hop.alloc_buffer on %tile_A { 
        memory_space = "L1" 
    } -> memref<1024xf32, memory_space = "L1">

    // %buffer_B 是在 Tile B 的 L1 内存中的 "目的地"
    %buffer_B = dma_hop.alloc_buffer on %tile_B { 
        memory_space = "L1" 
    } -> memref<1024xf32, memory_space = "L1">

    
    // --- 6. 执行传输 (The "How" & "When") ---
    
    // 这是核心操作：
    // 1. 将 *原始* 数据 (%data) 推送到 *整个* 路径 (%serial_path)。
    // 2. 'outputs' 映射表告诉硬件:
    //    - 当数据到达 %port_A_in 时, 存一份到 %buffer_A。
    //    - 当数据到达 %port_B_in 时, 存一份到 %buffer_B。
    //
    // 硬件（或 `dataschedule` 层）将负责配置 Tile A 上的 "Tee" (分流)
    // 以便数据流 *同时* 写入 L1 并 *转发* 到 %port_A_out。
    dma_hop.push %data onto %serial_path
      outputs(
        %port_A_in -> %buffer_A,  // 消费者 1: "缓存" 在 A
        %port_B_in -> %buffer_B   // 消费者 2: "存储" 在 B
      )

      
    // --- 7. 同步 ---
    
    // 阻塞, 直到 *所有* 消费者 (%buffer_A 和 %buffer_B) 都已接收完毕
    dma_hop.sync %serial_path

    
    // --- 8. 释放资源 ---
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

def dmaphop_FuncOp : dmaphopop<"func",[IsolatedFromAbove, SymbolTable, Symbol]>
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
}

// --- 1. Tile Definition Operation ---

def dmaphop_tile : dmaphopop<"tile"> {
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

def dmaphop_port : dmaphopop<"port",[Symbol]> {
  let summary = "Defines a logical port on a tile";
  let arguments = (ins
    dmaphoptiletype:$tile,
    StrAttr:$direction,
    SymbolNameAttr:$sym_name,
    OptionalAttr<I64Attr>:$direction_channel
  );
  let results = (outs dmaphopporttype:$port);
  let assemblyFormat = "`on` $tile `{` `direction` `=` $direction (`,` `channel` `=` $direction_channel^)? `}` `:` type($tile) attr-dict `->` type($port)";
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
  let summary = "Creates a path from a sequence of hops with consumers and tee points";
  let description = [{
    Creates a complete data movement path from a sequence of hops, specifying:
    - List of hops that form the path
    - Consumer ports that will receive the data
    - Tee points where data splitting occurs
  }];

  let arguments = (ins
    Variadic<dmaphophoptype>:$hops,
    ArrayAttr:$consumers,    // Array of ports that consume data
    ArrayAttr:$tee_points    // Array of ports where data splitting occurs
  );
  
  let results = (outs dmaphoppathtype:$path);
  
  let assemblyFormat = [{
    `[` $hops `]` `{`
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
    AnyMemRef:$memref_type
  );
  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = "`on` $tile $memref_type `:` type($tile) type($memref_type) attr-dict `->` type($buffer)";
}

// --- 6. Push Operation ---
 
def dmaphop_push : dmaphopop<"push"> {
  let summary = "Pushes data through a path to multiple destinations";
  let arguments = (ins
    AnyMemRef:$data,
    dmaphoppathtype:$path,
    Variadic<AnyMemRef>:$output_buffers,
    Variadic<dmaphopporttype>:$output_ports
  );
  let traits = [SameVariadicOperandSize];
  let assemblyFormat = "$data `onto` $path `outputs` `(` $output_buffers `at` $output_ports `)` `:` type($data) type($path) type($output_buffers) type($output_ports) attr-dict";
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
  let arguments = (ins AnyMemRef:$buffer);
  let assemblyFormat = "$buffer `:` type($buffer) attr-dict";
}
 
#endif // DMAPHOP_OPS
