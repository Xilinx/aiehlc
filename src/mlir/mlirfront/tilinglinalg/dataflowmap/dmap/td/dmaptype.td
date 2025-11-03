#ifndef _TYPE_DMAMAP_
#define _TYPE_DMAMAP_
include "dmapattr.td"

def dmapcoreenginegrouptype:dmaptype<"dmacoreenginegroup"> {
    let mnemonic = "dmacoreenginegroupType";
    let summary = "this is dmap process engine group";
}

def dmapioenginetype:dmaptype<"dmapioenginetype"> {
    let mnemonic = "dmapioenginetype";
    let summary = "this is dmap IO engine";
}

def dmapporttype : dmaptype<"dmapport"> {
    let mnemonic = "dmapport";
    let summary = "this is a dmap port";
}

def dmapportgrouptype : dmaptype<"dmapportgroup"> {
    let mnemonic = "dmapportgroup";
    let summary = "this is a dmap port";
}

def dmapportstreamtype : dmaptype<"dmapportstream"> {
    let mnemonic = "dmapportstream";
    let summary = "this is a dmap stream port";
}

def dmapportchainstreamtype : dmaptype<"dmapportchainstream"> {
    let mnemonic = "dmapportchainstream";
    let summary = "this is a dmap chain/concatenated stream composed of multiple port streams";
}

def dmapdatatype : dmaptype<"dmapdata"> {
    let mnemonic = "dmapdata";
    let summary = "this is a dmap data";
    
}

def dmapioconfigtype : dmaptype<"dmapioconfig"> {
    let mnemonic = "dmapioconfig";
    let summary = "this is a dmap port config"; 
}

def dmapportconfigtype : dmaptype<"dmapportconfig"> {
    let mnemonic = "dmapportconfig";
    let summary = "this is a dmap port config"; 
}

def dmacoregroupconfigtype : dmaptype<"dmacoregroupconfig"> {
    let mnemonic = "dmacoregroupconfig";
    let summary = "this is a dmap port config map";
}

//def dmapportorgroup : TypeConstraint<
// CPred<"$_self.isa<dflow::PortType>() || $_self.isa<dflow::PortGroupType>()">,
 // "port or port group">;

#endif //_TYPE_DMAMAP_