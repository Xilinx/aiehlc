/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
#ifndef AIELINK_TYPES
#define AIELINK_TYPES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/OpBase.td" // This line is crucial
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"

// Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
def AieLinkDialect : Dialect {
    // keep the pre-LLVM-24 textual form (inherent attrs in attr-dict)
    let useStrictPropertiesInAssemblyFormat = 0;
  let name = "AieLink";
  let cppNamespace = "::mlir::aiel";
  dag discardableAttrs = (ins);
  let hasOperationAttrVerify = 0;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
}

class AieLinkType<string name> : TypeDef<AieLinkDialect, name> {}
class AieLinkAttr<string name> : AttrDef<AieLinkDialect, name> {}

def Aie_L1MemBankType : AieLinkType<"Aie_L1MemBank"> {
	let mnemonic = "Mem bank";
	let summary = "Mem bank";
	let description = [{the mem bank}];
	let parameters = (ins
    "IntegerAttr":$locationdirect, // 0 south, 1 west, 2 north, 3 east
		"StringAttr":$locationtext,
    "IntegerAttr":$baseaddr,
    "IntegerAttr":$memtotalsize,
    "IntegerAttr":$islocal
    
	);
	let assemblyFormat = [{`<` $locationdirect `,` $locationtext `,` $baseaddr `,` $memtotalsize `,` $islocal}];
}

def Aie_MemBufferType : AieLinkType<"Aie_MemBuffer"> {
	let mnemonic = "MemBuffer";
	let summary = "The Mem Buffer";
	let description = [{the symbol type}];
	let parameters = (ins
		"Aie_L1MemBankType": $mbanktype,
		"IntegerAttr":$addr,
    "IntegerAttr":$len
	);
	let assemblyFormat = [{`<` $mbanktype `,` $addr `,` $len `>`}];

}

def Aie_SymbolType : AieLinkType<"Aie_Symbol"> {
	let mnemonic = "SymbolType";
	let summary = "The symbol type";
	let description = [{the symbol type}];
	let parameters = (ins
		"StringAttr":$name,
    "Aie_MemBufferType":$addr
	);
	let assemblyFormat = [{`<` $name `,` $addr `>`}];

}

def Aie_SectionType : AieLinkType<"Aie_Section"> {
	let mnemonic = "SectionType";
	let summary = "the section type";
	let description = [{the sction type}];
	let parameters = (ins
		"StringAttr":$sectiontypename,
		"StringAttr":$sectionname,
		"Aie_MemBufferType":$startaddress,
		"IntegerAttr":$len
	);
	let assemblyFormat = [{`<` $sectiontypename `,` $sectionname `,` $startaddress `,` $len `>` }];
}

#endif

