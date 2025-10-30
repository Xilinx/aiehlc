#ifndef _ATTR_DMAMAP_
#define _ATTR_DMAMAP_

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/EnumAttr.td"

def dmapdialect : Dialect {
    let name = "dmap";
    let cppNamespace = "dmap";
    //let useDefaultAttributePrinterParser = 1;
    let hasOperationAttrVerify = 0;
    let useDefaultAttributePrinterParser = 1;
	  let usePropertiesForAttributes = 0;
    let useDefaultTypePrinterParser = 1;  
    dag discardableAttrs = (ins);
}


class dmapattr<string name> : AttrDef<dmapdialect, name>{}
class dmaptype<string name, list<Trait> traits=[]> : TypeDef<dmapdialect, name,traits>{}
class dmapop<string name, list<Trait> traits=[]> : Op<dmapdialect, name, traits>{}

///*
def DMAP_SEND : I32EnumAttrCase<"DMAP_SEND", 0, "DMAP_SEND">;
def DMAP_RECEIVE : I32EnumAttrCase<"DMAP_RECEIVE", 1, "DMAP_RECEIVE">;
def dmapdirectionattr
    : I32EnumAttr<"dmapdirection", "dmap direction", [
      DMAP_SEND,
      DMAP_RECEIVE]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "dmap";
}

//def dmapdirectionAttr : EnumAttr<dmapdialect, dmapdirection, "direction"> {
//  let assemblyFormat = "`<` $value `>`";
//}


def DMAP_SHIMIO : I32EnumAttrCase<"DMAP_SHIMIO", 0, "DMAP_SHIMIO">;
def DMAP_MEMTILEIO : I32EnumAttrCase<"DMAP_MEMTILEIO", 1, "DMAP_MEMTILEIO">;
def DMAP_CORETILEIO : I32EnumAttrCase<"DMAP_CORETILEIO", 2, "DMAP_CORETILEIO">;
def dmapio_enumattr
    : I32EnumAttr<"dmapio", "dmap direction", [
      DMAP_SHIMIO,
      DMAP_MEMTILEIO,
      DMAP_CORETILEIO]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "dmap";
}

//def dmapioAttr : EnumAttr<dmapdialect, dmapio_enum, "io"> {
// let assemblyFormat = "`<` $value `>`";
//}
//*/

def dataaccesspattern : dmapattr<"dataaccesspattern"> {
    let mnemonic = "dataaccesspattern";
    let cppNamespace = "dmap";
    let parameters = ( ins
        "StringAttr":$direction,
        "int64_t":$sizepertran,
        "int64_t":$pingpongbuffer,
        "int64_t":$slidingaccess
    );
    let assemblyFormat = "`{` $direction  `,` $sizepertran  `,` $pingpongbuffer `,` $slidingaccess `}`";

}

def dataconfigmapitem :dmapattr<"dataconfmapitem"> {
    let mnemonic = "dataconfmapitem";
    let cppNamespace = "dmap";
    let parameters = ( ins
        "int64_t":$index,
        "SymbolRefAttr":$configsymbol
    );
    let assemblyFormat = "`{` $index  `,` $configsymbol `}`";
}

def dataconfigmap : dmapattr<"dataconfigmap"> {
  let mnemonic = "dataconfigmap";
  let summary  = "configure map";
  let parameters = (ins
    ArrayRefParameter<"dataconfmapitemAttr">:$items
  );
  let assemblyFormat = "`[` $items `]`";
}



#endif//_ATTR_DMAMAP_