/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

include "mlir/IR/OpBase.td"

def RoutingHWDialect : Dialect {
    // keep the pre-LLVM-24 textual form (inherent attrs in attr-dict)
    let useStrictPropertiesInAssemblyFormat = 0;
    let name = "routinghw";
    let cppNamespace = "routinghw";
    let description = "hw level routing";
    let summary = "call aie driver api";
    dag discardableAttrs = (ins);
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

class RoutingHWType<string name, list<Trait> traits=[]> : TypeDef<RoutingHWDialect, name, traits>;
class RoutingHWAttr<string name, list<Trait> traits=[]> : AttrDef<RoutingHWDialect, name, traits>;
class RoutingHWOp<string name, list<Trait> traits=[]> : Op<RoutingHWDialect, name, traits>;

def TileArrayHandleAttr :RoutingHWAttr<"tilearrayhandle"> {
    let mnemonic = "tilearrayhandle";
    let cppNamespace = "routinghw";
    let parameters = (
        ins
        "StringAttr":$comments
    );
    let assemblyFormat = " $comments";
}
