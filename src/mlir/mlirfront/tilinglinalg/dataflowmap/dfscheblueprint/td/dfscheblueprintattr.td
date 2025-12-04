/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#ifndef _ATTR_dfscheblueprint_
#define _ATTR_dfscheblueprint_

//===- dfscheblueprintDialect.td ----------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/EnumAttr.td"

// Dialect: dfscheblueprint
def dfscheblueprintdialect : Dialect {
    let name = "dfscheblueprint";
    let cppNamespace = "dfscheblueprint";
    let hasOperationAttrVerify = 0;
    let useDefaultAttributePrinterParser = 1;
    let usePropertiesForAttributes = 0;
    let useDefaultTypePrinterParser = 1;  
    dag discardableAttrs = (ins);
}

class dfscheblueprintattr<string name> : AttrDef<dfscheblueprintdialect, name>{ let mnemonic = name;}
class dfscheblueprinttype<string name, list<Trait> traits=[]> : TypeDef<dfscheblueprintdialect, name,traits>{}
class dfscheblueprintop<string name, list<Trait> traits=[]> : Op<dfscheblueprintdialect, name, traits>{}

// DMA Direction Enum
def BP_MM2S : I32EnumAttrCase<"MM2S", 0, "MM2S">;
def BP_S2MM : I32EnumAttrCase<"S2MM", 1, "S2MM">;
def BP_BOTH : I32EnumAttrCase<"BOTH", 2, "BOTH">;
def bp_direction_enum : I32EnumAttr<"bp_direction", "DMA direction", [
      BP_MM2S,
      BP_S2MM,
      BP_BOTH]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "dfscheblueprint";
}

def bp_directionAttr : EnumAttr<dfscheblueprintdialect, bp_direction_enum, "direction"> {
  let assemblyFormat = "`<` $value `>`";
}

// DMA Attribute
def DMAAttr : dfscheblueprintattr<"DMA"> {
  let summary = "DMA configuration attribute";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$channels,
    "bp_direction":$direction
  );
  let assemblyFormat = "`<` `channels` `=` `[` $channels `]` `,` `direction` `=` $direction `>`";
}

// Slice attribute - defines data tiling/slicing
def SliceAttr : dfscheblueprintattr<"Slice"> {
  let summary = "Data slice definition for tiling";
  let parameters = (ins
    "TypeAttr":$data_type,           // memref<1024x1024xf32>
    "ArrayAttr":$offset,              // [0, 0]
    "ArrayAttr":$size,                // [512, 1024]
    "ArrayAttr":$stride               // [1024, 1]
  );
  let assemblyFormat = "`<` `data_type` `=` $data_type `,` `offset` `=` $offset `,` `size` `=` $size `,` `stride` `=` $stride `>`";
}

// Endpoint attribute - defines tile location and DMA configuration
def EndpointAttr : dfscheblueprintattr<"Endpoint"> {
  let summary = "Physical tile endpoint with DMA configuration";
  let parameters = (ins
    "int64_t":$row,                   // tile row
    "int64_t":$col,                   // tile col  
    "StringAttr":$direction,          // "MM2S" or "S2MM"
    "int64_t":$channel                // DMA channel number
  );
  let assemblyFormat = "`<` `tile` `=` `(` $row `,` $col `)` `,` `direction` `=` $direction `,` `channel` `=` $channel `>`";
}

#endif //_ATTR_dfscheblueprint_