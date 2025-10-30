#ifndef _ATTR_DMAMAP_
#define _ATTR_DMAMAP_

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/EnumAttr.td"

// Direction enum for ports
def Direction_IN  : I32EnumAttrCase<"In", 0, "in">;
def Direction_OUT : I32EnumAttrCase<"Out", 1, "out">;

def DirectionAttr : I32EnumAttr<"Direction", "Port direction attribute",
                               [Direction_IN, Direction_OUT]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "dmaphop";
}

def dmaphopdialect : Dialect {
    let name = "dmaphop";
    let cppNamespace = "dmaphop";
    //let useDefaultAttributePrinterParser = 1;
    let hasOperationAttrVerify = 0;
    let useDefaultAttributePrinterParser = 1;
    let usePropertiesForAttributes = 0;
    let useDefaultTypePrinterParser = 1;  
    dag discardableAttrs = (ins);
}

class dmaphopattr<string name> : AttrDef<dmaphopdialect, name>{}
class dmaphoptype<string name, list<Trait> traits=[]> : TypeDef<dmaphopdialect, name,traits>{}
class dmaphopop<string name, list<Trait> traits=[]> : Op<dmaphopdialect, name, traits>{}

def dmaphop_portaccesspattern : dmaphopattr<"portaccesspattern"> {
    let mnemonic = "portaccesspattern";
    let cppNamespace = "dmaphop";
    let parameters = (ins
        "StringAttr":$direction,
        "int64_t":$transize,
        "int64_t":$totalsize,
        "int64_t":$pingpongbuffer,
        "int64_t":$slidingaccess
    );
    let assemblyFormat = [{
        `{` $direction `,` 
            $transize `,` 
            $totalsize `,`
            $pingpongbuffer `,` 
            $slidingaccess 
        `}`
    }];
}

#endif//_ATTR_DMAMAP_