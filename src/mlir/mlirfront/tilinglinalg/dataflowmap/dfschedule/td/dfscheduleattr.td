/******************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
//===- dfscheduleDialect.td ---------------------------------------*- tablegen -*-===//
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/EnumAttr.td"

// Dialect: amd_host
def dfscheduledialect : Dialect {
    // keep the pre-LLVM-24 textual form (inherent attrs in attr-dict)
    let useStrictPropertiesInAssemblyFormat = 0;
    let name = "dfschedule";
    let cppNamespace = "dfschedule";
    //let useDefaultAttributePrinterParser = 1;
    let hasOperationAttrVerify = 0;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;  
    dag discardableAttrs = (ins);
}

class dfscheduleattr<string name> : AttrDef<dfscheduledialect, name>{ let mnemonic = name;}
class dfscheduleype<string name, list<Trait> traits=[]> : TypeDef<dfscheduledialect, name,traits>{}
class dfscheduleop<string name, list<Trait> traits=[]> : Op<dfscheduledialect, name, traits>{}


def dfschedule_TileAttr : dfscheduleattr<"Tile"> {
  let summary = "Tile location attribute (row, col)";
  let cppNamespace = "dfschedule";
  let parameters = (ins
    "int64_t":$row,
    "int64_t":$col
  );
  let assemblyFormat = "`<` `row` `=` $row `,` `col` `=` $col `>`";
}


def dfschedule_AccumAttr : dfscheduleattr<"Accum"> {
  let summary = "DMA accumulation descriptor: per-round tile [rows, cols] and "
                "per-dim round counts that reconcile a subview shape with the "
                "per-round dma_bd len.";
  let mnemonic = "accum";          // lowercase => #dfschedule.accum<...>
  let cppNamespace = "dfschedule";
  let parameters = (ins
    "int64_t":$round_rows,
    "int64_t":$round_cols,
    "int64_t":$d1_rounds,
    "int64_t":$d2_rounds
  );
  let assemblyFormat =
    "`<` `round` `=` `[` $round_rows `,` $round_cols `]` `,` "
    "`d1` `=` $d1_rounds `,` `d2` `=` $d2_rounds `>`";
}


def DMAP_MM2S : I32EnumAttrCase<"DMAP_MM2S", 0, "DMAP_MM2S">;
def DMAP_S2MM : I32EnumAttrCase<"DMAP_S2MM", 1, "DMAP_S2MM">;
def dsdirection_enum : I32EnumAttr<"dsdirection", "dsdirection direction", [
      DMAP_MM2S,
      DMAP_S2MM]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "dsdirection";
}

def dsdirectionAttr : EnumAttr<dfscheduledialect, dsdirection_enum, "io"> {
  let assemblyFormat = "`<` $value `>`";
}
 /*
def Flow_ShimEndpointAttr : dfscheduleattr<"ShimEndpoint"> {
  let summary = "Physical SHIM endpoint binding";
  let parameters = (ins
    "SymbolRefArrayAttr":$hops, // optional route, keep generic
    "int64_t":$row,
    "int64_t":$col,
    "StrAttr":$port,           // "MM2S0"/"S2MM0"/etc.
    "int64_t":$channel,
    "int64_t":$packet_id,
    "int64_t":$vc
  );
  let assemblyFormat =  "`{` $hops  `,` $row  `,` $col `,` $port `,` $channel  `,` $packet_id `,` $vc `}`";
}
 */