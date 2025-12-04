/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef dfscheblueprint_OPS
#define dfscheblueprint_OPS

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

include "mlir/IR/BuiltinTypes.td"
include "dfscheblueprinttype.td"
//-- no SymbolTable will cause the generic print when block have Symbol
//-- no NoTerminator will cause the generic print when block is empty or there is no terminate op like yeld
//-- let hasCustomAssemblyFormat = 1; will fix the generic print issue after the said fix in
//-- the synbol op can not be put inside a func.func, if we do that the symbol func will cause generic print
//--- the reason is the func.func do isolatedfrom above but @symbol will be module visible that will cause confuse
// dfscheblueprint.config - Top-level container for schedule blueprint
def dfscheblueprint_ConfigOp : dfscheblueprintop<"config", [
    IsolatedFromAbove,  
    SymbolTable,
    Symbol,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Schedule configuration blueprint container";
  let description = [{
    Top-level operation that contains transfer manifests defining
    the complete data movement schedule blueprint.
    
    Example:
    ```mlir
    dfscheblueprint.config @my_blueprint {
      dfscheblueprint.transfer_manifest @transfer1 { ... }
      dfscheblueprint.transfer_manifest @transfer2 { ... }
    }
    ```
  }];
  
  let arguments = (ins 
    SymbolNameAttr:$sym_name
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let hasCustomAssemblyFormat = 1;
}
// Symbol will cause generic print once the partent have no Symboltalbe
// dfscheblueprint.transfer_manifest - Individual data transfer specification
def dfscheblueprint_TransferManifestOp : dfscheblueprintop<"transfer_manifest", [
    Symbol,
    HasParent<"ConfigOp">
  ]> {
  let summary = "Data transfer manifest with tiling and routing information";
  let description = [{
    Defines a single data transfer including:
    - payload_slice: What data to transfer (tiling definition)
    - packet_id: Packet identifier for routing
    - source: Single source endpoint (tile, direction, channel)
    - destinations: List of destination endpoints (for broadcast)
    
    Example:
    ```mlir
    dfscheblueprint.transfer_manifest @broadcast_upper {
      payload_slice = #dfscheblueprint.slice<
        data_type = memref<1024x1024xf32>,
        offset = [0, 0],
        size = [512, 1024],
        stride = [1024, 1]
      >,
      packet_id = 10 : i32,
      source = #dfscheblueprint.endpoint<tile=[0,2], direction="MM2S", channel=0>,
      destinations = [
        #dfscheblueprint.endpoint<tile=[2,2], direction="S2MM", channel=0>,
        #dfscheblueprint.endpoint<tile=[3,2], direction="S2MM", channel=0>
      ]
    }
    ```
  }];
  
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SliceAttr:$payload_slice,
    I32Attr:$packet_id,
    EndpointAttr:$source,
    ArrayAttr:$destinations  // Array of EndpointAttr
  );
  
  let hasCustomAssemblyFormat = 1;
  
  let extraClassDeclaration = [{
    // Helper to access destination endpoints
    ArrayAttr getDestinationEndpoints() { return getDestinations(); }
  }];
}

// dfscheblueprint.tile_group
def dfscheblueprint_TileGroupOp : dfscheblueprintop<"tile_group", [
    Symbol
  ]> {
  let summary = "Defines a group of physical tiles";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ArrayAttr:$tiles // Array of arrays/tuples e.g. [(0,2)]
  );
  let assemblyFormat = "$sym_name `{` `tiles` `=` $tiles `}` attr-dict";
}

// dfscheblueprint.declare_data - Creates a logical tensor without memory allocation
def dfscheblueprint_DeclareDataOp : dfscheblueprintop<"declare_data", [
    
  ]> {
  let summary = "Declares a logical data tensor";
  let description = [{
    Declares a logical tensor for scheduling purposes. This creates an SSA value
    representing the data shape and type without actual memory allocation.
    
    Example:
    ```mlir
    %root_data = dfscheblueprint.declare_data tensor<1024x1024xf32>
    ```
  }];
  let arguments = (ins
    TypeAttr:$data_type
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$data_type attr-dict `->` type($result)";
}

// dfscheblueprint.partition
def dfscheblueprint_PartitionOp : dfscheblueprintop<"partition", [
    
  ]> {
  let summary = "Partitions data into views";
  let arguments = (ins
    DataHandleType:$data,
    ArrayAttr:$tile_shape
  );
  let results = (outs ViewHandleType:$result);
  let assemblyFormat = "$data `{` `tile_shape` `=` $tile_shape `}` attr-dict";
}

// dfscheblueprint.extract
def dfscheblueprint_ExtractOp : dfscheblueprintop<"extract", [
    
  ]> {
  let summary = "Extracts a specific partition from a partitioned view";
  let arguments = (ins
    AnyType:$view,
    I32Attr:$index
  );
  let results = (outs ViewHandleType:$result);
  let assemblyFormat = "$view `:` type($view) `{` `index` `=` $index `}` attr-dict";
}

// dfscheblueprint.data_slice - wraps a tensor slice with a symbol name
def dfscheblueprint_DataSliceOp : dfscheblueprintop<"data_slice", [
    Symbol
  ]> {
  let summary = "Wraps a tensor slice with a symbol name for scheduling";
  let description = [{
    Wraps a tensor (typically from tensor.extract_slice) with a symbol name
    for scheduling and binding purposes.
    
    Example:
    ```mlir
    %slice = tensor.extract_slice %tensor[0, 0] [4, 256] [1, 1] : tensor<16x256xf32> to tensor<4x256xf32>
    %wrapped = dfscheblueprint.data_slice @out_slice_0 wrap %slice : tensor<4x256xf32>
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    AnyTensor:$tensor_slice
  );
  let results = (outs AnyTensor:$result);
  let hasCustomAssemblyFormat = 1;
}

// dfscheblueprint.flowconfig
def dfscheblueprint_FlowConfigOp : dfscheblueprintop<"flowconfig", [
    Symbol
  ]> {
  let summary = "Binds a resource or resource group to a data view";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SymbolRefAttr:$target,
    AnyType:$view,
    StrAttr:$distribution,
    DMAAttr:$dma,
    OptionalAttr<ArrayAttr>:$slice_symbols,
    OptionalAttr<StrAttr>:$type
  );
  let hasCustomAssemblyFormat = 1;
}

// dfscheblueprint.flow_transfer
def dfscheblueprint_FlowTransferOp : dfscheblueprintop<"flow_transfer", [
    Symbol
  ]> {
  let summary = "Defines a collective transfer between bindings";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$type,
    SymbolRefAttr:$from,
    SymbolRefAttr:$to,
    OptionalAttr<StrAttr>:$ordering,
    I32Attr:$base_packet_id
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // dfscheblueprint_OPS
