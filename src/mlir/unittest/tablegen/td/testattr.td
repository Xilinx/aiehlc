/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef AIE_ATTRS
#define AIE_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/OpBase.td" // This line is crucial
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"

def testdialect : Dialect {
	let name = "test";
	let cppNamespace = "test";
	dag discardableAttrs = (ins);
	let hasOperationAttrVerify = 0;
	let useDefaultTypePrinterParser = 1;
	let useDefaultAttributePrinterParser = 1;
}

class testop<string mnemonic, list<Trait> traits = []> : Op<testdialect, mnemonic, traits>;
//class testop<string name, list<Trait>  traits=[]> : Op<testdialect, name, traits> {}
class testtype<string name, list<Trait> traits=[]> : TypeDef<testdialect, name, traits> {}
class testattr<string name> : AttrDef<testdialect, name> {}

def TileRangeAttr : testattr<"TileRange"> {
  let mnemonic = "tilerangeattr";
  let cppNamespace = "test";
  let parameters = (ins
    "int64_t":$rowNum,
    "int64_t":$colNum
  );

  let assemblyFormat = "`{` $rowNum `,` $colNum `}`";
}

def TileRangeArrayAttr : testattr<"TileRangeArray"> {
  let mnemonic = "TileRangeArrayAttr";
  let summary = "a tile array";
  let cppNamespace = "test";
  let parameters = (ins
    OptionalArrayRefParameter<"TileRangeAttr">:$items
  );
  let assemblyFormat = "`{` $items `}`";
}
#endif
