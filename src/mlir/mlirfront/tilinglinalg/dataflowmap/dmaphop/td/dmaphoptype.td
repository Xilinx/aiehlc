#ifndef _TYPE_DMAMAPHOP_
#define _TYPE_DMAMAPHOP_
include "dmaphopattr.td"

// Physical tile type
def dmaphoptiletype : dmaphoptype<"tile"> {
    let mnemonic = "tile";
    let summary = "Represents a physical tile in the device";
}

// Port type for data movement endpoints
def dmaphopporttype : dmaphoptype<"port"> {
    let mnemonic = "port";
    let summary = "Represents a logical port on a tile for data movement";
}

// Hop type for direct connections
def dmaphophoptype : dmaphoptype<"hop"> {
    let mnemonic = "hop";
    let summary = "Represents a direct connection between two ports";
}

// Path type for sequences of hops
def dmaphoppathtype : dmaphoptype<"path"> {
    let mnemonic = "path";
    let summary = "Represents a complete data movement path composed of hops";
}

// Core engine group type (existing)
def dmaphopcoreenginegrouptype : dmaphoptype<"dmacoreenginegroup"> {
    let mnemonic = "dmacoreenginegroupType";
    let summary = "Represents a group of processing engine cores";
}

#endif //_TYPE_DMAMAPHOP_