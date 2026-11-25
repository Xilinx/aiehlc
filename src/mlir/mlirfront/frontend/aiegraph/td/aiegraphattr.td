/******************************************************************************
* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
#ifndef _ATTR_AIEGRAPH_
#define _ATTR_AIEGRAPH_

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/EnumAttr.td"

def aiegraphdialect : Dialect {
    // keep the pre-LLVM-24 textual form (inherent attrs in attr-dict)
    let useStrictPropertiesInAssemblyFormat = 0;
    let name = "aiegraph";
    let cppNamespace = "aiegraph";
    let hasOperationAttrVerify = 0;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
    dag discardableAttrs = (ins);
}

class aiegraphattr<string name> : AttrDef<aiegraphdialect, name>{}
class aiegraphtype<string name, list<Trait> traits=[]> : TypeDef<aiegraphdialect, name,traits>{}
class aiegraphop<string name, list<Trait> traits=[]> : Op<aiegraphdialect, name, traits>{}

// Per-tensor int8 affine-quant parameters (scale + zero-point), mirroring the
// QNN qparams recovered by the Relay importer. Kept as a structured attribute so
// the quant metadata rides as a verifiable SSA-adjacent attribute rather than a
// loose bag of dictionary entries.
def quantparam : aiegraphattr<"quantparam"> {
    let mnemonic = "quant";
    let cppNamespace = "aiegraph";
    let summary = "int8 affine quant params: input/output scale + zero-point and Q7 BN scale/bias";
    // in_scale/out_scale are floating-point affine-quant scales. MLIR's default
    // tablegen storage has no FieldParser/hash for raw `double`, so they ride as
    // builtin f64 FloatAttr (Attribute params get a FieldParser + hash_value for
    // free and round-trip in the assembly format).
    let parameters = ( ins
        "::mlir::FloatAttr":$in_scale,
        "int64_t":$in_zp,
        "::mlir::FloatAttr":$out_scale,
        "int64_t":$out_zp,
        "int64_t":$bn_scale,
        "int64_t":$bn_bias
    );
    let assemblyFormat = "`{` $in_scale `,` $in_zp `,` $out_scale `,` $out_zp `,` $bn_scale `,` $bn_bias `}`";
}

#endif//_ATTR_AIEGRAPH_
