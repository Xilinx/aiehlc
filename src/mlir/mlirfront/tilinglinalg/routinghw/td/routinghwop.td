/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

include "routinghwtype.td"
/*// the omponent 1 not implemented and it should be add then it can cooperate with data packet_flow
schedule dialect
builtin.module {

  // --- Component 1: Declarative "Contract" (L3R Database) ---
  // This is global and readable by Pass 2
  routinghw.config @my_routes {
    
    // Pass 1 *declares* its decisions here
    routinghw.packet_flow @route_A {
       dest_tile   = #dma_hop<tile {col=0, row=3}>,
       packet_id   = 1 : i32
    }
    routinghw.packet_flow @route_B {
       dest_tile   = #dma_hop<tile {col=1, row=3}>,
       packet_id   = 2 : i32
    }
  }

  // --- Component 2: Imperative "Script" (L3R Instructions) ---
  // (This is your existing @main, it *implements* the contract above)
  // Its i32 return value (%0) is only used internally or is discarded
  func.func @main() {
    ...
    // %0 = routing.RoutingCreate ... { ... }
    // ...
    // %10 = routinghw.connectpktstreamswitchport ... { localdmapktid = 1 ... }
    // %11 = routinghw.connectpktstreamswitchport ... { localdmapktid = 2 ... }
    ...
    return
  }
}
*/

def TileArrayHandleCreate:RoutingHWOp<"tilearrayhandlecreate"> {
    let cppNamespace = "routinghw";
    let arguments = (ins
        StrAttr:$name
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "attr-dict `:` type($output)";
}

def TileCreate:RoutingHWOp<"tilecreate"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32Attr:$row,
        I32Attr:$col,
        StrAttr:$comments
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "attr-dict `->` type($output) ";
}

def IOShimTileCreate:RoutingHWOp<"ioshimtilecreate"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32Attr:$row,
        I32Attr:$col,
        I32Attr:$IOID,
        StrAttr:$comments,
        I32Attr:$dmadirection, 
        I32Attr:$channelused 
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "attr-dict `->` type($output) ";
}

def ConnectIOToTile:RoutingHWOp<"connectiototile"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32:$ioshimtile,
        I32:$dsttile
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "$ioshimtile $dsttile `:` type($ioshimtile) type($dsttile) attr-dict `->` type($output)";
}

def ConnecIOToTileArray:RoutingHWOp<"connectiototilearray"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32:$ioshimtile,
        I32:$dsttilearray
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "$ioshimtile $dsttilearray `:` type($ioshimtile) type($dsttilearray) attr-dict `->` type($output)";
}

def CreateShimStreamSwitchPort:RoutingHWOp<"createshimstreamswitchport"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32:$shimile,
        StrAttr:$shimmasterport,
        I32Attr:$shimmasterportidx,
        I32Attr:$shimmasterporttype
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "$shimile `:` type($shimile) attr-dict `->` type($output)";
}

def ConnectStreamSwitchPort:RoutingHWOp<"connectstreamswitchport"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32:$pretileporthandle,
        I32:$srctile,
        I32:$dsttile,
        StrAttr:$srcmasterport,
        I32Attr:$srcmasterportidx,
        StrAttr:$srcslaveport,
        I32Attr:$srcslaveportidx,
        StrAttr:$dstmasterport,
        I32Attr:$dstmasterportidx,
        StrAttr:$dstslaveport,
        I32Attr:$dstslaveportidx
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "$pretileporthandle $srctile $dsttile `:` type($pretileporthandle) type($srctile) type($dsttile) attr-dict `->` type($output)";
}

def ConnectStreamSingleSwitchPort:RoutingHWOp<"connectsinglestreamswitchport"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32:$curtile,
        StrAttr:$slaveportdirection,
        I32Attr:$slaveportidx,
        StrAttr:$masterportdirection,
        I32Attr:$masterportidx
    );
    let results = (outs
        I32:$output
    );
    //let assemblyFormat = "$curtile `:` type($curtile) attr-dict `->` type($output)";
    //let assemblyFormat = "$curtile `:` type($curtile) $slaveportdirection `,` $slaveportidx `,` $masterportdirection `,` $masterportidx attr-dict `:` type($output)";
    let assemblyFormat =
        "$curtile `:` `{` "
        "`slaveportdirection` `=` $slaveportdirection `,` "
        "`slaveportidx` `=` $slaveportidx `,` "
        "`masterportdirection` `=` $masterportdirection `,` "
        "`masterportidx` `=` $masterportidx "
        "`}` attr-dict `:` type($output)";
}

def ConnectStreamPktSwitchPort:RoutingHWOp<"connectpktstreamswitchport"> {
     let cppNamespace = "routinghw";
     let arguments = (
        ins
        I32:$curtile,
        StrAttr:$receiveslavedirection,
        I32Attr:$receiveslaveportidx,
        I32Attr:$receiveslavepktid,
        I32Attr:$receiveslavepkttype,
        StrAttr:$localdmadirection,
        I32Attr:$localdmaportidx,
        I32Attr:$localdmapktid,
        I32Attr:$localdmapkttype,
        StrAttr:$forwardmasterdirection,
        I32Attr:$forwardmasterportidx,
        OptionalAttr<BoolAttr>:$preserveheader,
        DefaultValuedAttr<I32Attr, "0">:$receiveslavemask,
        DefaultValuedAttr<I32Attr, "0">:$receiveslavemsel,
        DefaultValuedAttr<I32Attr, "0">:$receiveslavearbiter,
        DefaultValuedAttr<I32Attr, "0">:$receiveslaveslot,
        DefaultValuedAttr<I32Attr, "31">:$localdmamask,
        DefaultValuedAttr<I32Attr, "0">:$localdmamsel,
        DefaultValuedAttr<I32Attr, "0">:$localdmaarbiter,
        DefaultValuedAttr<I32Attr, "0">:$localdmaslot,
        DefaultValuedAttr<I32Attr, "0">:$forwardmasterarbiter,
        DefaultValuedAttr<I32Attr, "1">:$forwardmastermselen,
        // Local stream-switch master port that the packet terminates into.
        // "DMA" (default) => S2MM data path; "CTRL" => in-tile register/memory
        // write via the control stream-switch port (control-packet flow).
        DefaultValuedStrAttr<StrAttr, "DMA">:$localsinkport
     );

     let results = (outs
        I32:$output
    );
    let assemblyFormat = "$curtile `:` type($curtile) attr-dict `->` type($output)";
}


def EnableExtToAieShimPort:RoutingHWOp<"enableexttoaieshimport"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32:$curtile,
        StrAttr:$portdirection,
        I32Attr:$portidx
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "$curtile `:` type($curtile) attr-dict `->` type($output)";
}

def EnableAieToExtShimPort:RoutingHWOp<"enableaietoextshimport"> {
    let cppNamespace = "routinghw";
    let arguments = (
        ins
        I32:$curtile,
        StrAttr:$portdirection,
        I32Attr:$portidx
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "$curtile `:` type($curtile) attr-dict `->` type($output)";
}