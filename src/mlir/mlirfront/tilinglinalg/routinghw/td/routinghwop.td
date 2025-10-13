/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

include "routinghwtype.td"

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
        I32:$handle,
        I32Attr:$row,
        I32Attr:$col,
        StrAttr:$comments
    );
    let results = (outs
        I32:$output
    );
    let assemblyFormat = "$handle  `:` type($handle) attr-dict `->` type($output) ";
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
        I32Attr:$forwardmasterportidx
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