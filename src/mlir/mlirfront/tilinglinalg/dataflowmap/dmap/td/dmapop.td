/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
#ifndef dmap_OPS
#define dmap_OPS
include "dmaptype.td"
include "dmapattr.td"
/*
      %data = dataflowmap.create_data {type="i32", dim1=10, dim2 20}
      %pegroup = dataflowmap.dmap_create_pe_group {group_idx = 1, pe_count=4, "row"}
      %coreenginegroup = dataflowmap.dmap_define_core_group {1, 4, "row"}
      %ioengine = dataflowmap.dmap_define_io_engine {0, "shim"}
      %portsend = dataflowmap.configure_io %ioengine {peorioidx=0, dataacces=}  
      %portreceive1 = dataflowmap.configure_io on %{peorioidx = 0, dataacces=}  
      %portreceive2 = dataflowmap.configure_io on %{peorioidx = 1, dataacces=}   
      %receive_group = dataflowmap.create_port_group(%portreceive1, %portreceive2)    
      %broadcast_stream = dataflowmap.create_stream %send_port, %receive_group         
      dataflowmap.push %data, %broadcast_stream {cache_policy = "force_memtile" }
*/
def dmap_FuncOp : dmapop<"func",[
    SymbolTable,
    Symbol,
    SingleBlock,
    NoTerminator
  ]>
{
    let summary = "Defines a function in the Starplat Dialect";
    let description = [{This operation represents a function definiton in the Starplat dialect.
                        It contains a name, other related arguments and an optional return type.
                        It has a single region for the function body.
                        }];

  let arguments = (ins 
                    SymbolNameAttr:$sym_name,
                    TypeAttrOf<FunctionType>:$funcType
                  );

  //let regions = (region AnyRegion:$body);
  let regions = (region SizedRegion<1>:$body);
  //let results = (outs I32:$output);
  let builders = [
    OpBuilder<(ins 
				   "Attribute":$Memo,
                   "TypeAttr":$type,
                   "function_ref<void(OpBuilder &, Location)>":$bodyBuilder),
    [{
      // no region creatation
	  // 2. Add operands to the state object.
      // 3. Add attributes to the state object.
      // Note that we must provide the attribute's name as a string.
      $_state.addAttribute("sym_name", Memo);
      $_state.addAttribute("funcType", type);

      // 2. Region and block
      auto *bodyRegion = $_state.addRegion();
      Block *bodyBlock = new Block();
      bodyRegion->push_back(bodyBlock);

      // 3. use the lambda to fill Block
      OpBuilder::InsertionGuard guard($_builder);
      $_builder.setInsertionPointToStart(bodyBlock);
      auto loc = mlir::UnknownLoc::get($_builder.getContext());
      bodyBuilder($_builder, loc);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
}

def dmap_YieldOp : dmapop<"yield", [Terminator]> {
  let summary = "Terminator for regions in routing ops";
  let description = [{
    The `routing.yield` operation is a terminator that yields a value from
    a region within a `routing` operation.
  }];

  //let arguments = (ins I32:$value);
  //let arguments = (ins);
  //let results = (outs); // No results
}

def dmap_create_data : dmapop<"create_data"> {
  let summary = "create data";
  let description = [{

  }];

  let arguments = (ins
    ArrayAttr:$shape,
    TypeAttr:$element_type
  );
  //let results = (outs dmapdatatype:$data );
  let results = (outs I32:$data );

  let assemblyFormat = [{
    attr-dict `->` type($data)
  }];
}

def dmap_define_core_group : dmapop<"define_core_group"> {
  let summary = "define core engine group";
  let description = [{

  }];

  let arguments = (ins
    I32Attr:$group_idx,
    I32Attr:$core_count,
    StrAttr:$group_axis // row or col
  );
  let results = (outs dmapcoreenginegrouptype:$pegroup );

  let assemblyFormat = [{
    attr-dict `->` type($pegroup)
  }];
}

def dmap_define_io_engine : dmapop<"define_io_engine"> {
  let summary = "create io group";
  let description = [{

  }];

  let arguments = (ins
    I32Attr:$io_id,
    StrAttr:$ioattr //SHIM MEM
  );
  let results = (outs dmapioenginetype:$ioengine );

  let assemblyFormat = [{
    attr-dict `->` type($ioengine)
  }];
}

def dmap_define_port_configure : dmapop<"define_port_configure", [Symbol]> {
  let summary = "this is port config.";
  let description = [{
    Binds a physical hardware handle (e.g., a tile from the routinghw dialect)
    to a logical data port, specifying its direction and a logical channel ID.
    This creates a node in the logical dataflow graph.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    dataaccesspattern:$accesspattern
  );
  let results = (outs dmapportconfigtype:$portconfig);
  //let assemblyFormat = [{
  let assemblyFormat =   "$sym_name `:` $accesspattern attr-dict-with-keyword  `->` type($portconfig)";
  //}];
}
def UseSymbolOp : Op<dmapdialect, "use_symbol", []> {
  let arguments = (ins SymbolRefAttr:$target);
  let assemblyFormat = "$target attr-dict";
}

def dmap_create_core_group_with_config : dmapop<"create_core_group_with_config"> {
  let summary = "this is port config.";
  let description = [{
    the map between core index and configure.
  }];

  let arguments = (ins
    dmapcoreenginegrouptype:$coregroup,
    StrAttr:$comments, // row or col
    dataconfigmap:$map
  );
  let results = (outs dmacoregroupconfigtype:$portconfig);
  let assemblyFormat = [{
      $coregroup `{` $map `,` $comments `}` attr-dict `:` type($coregroup) `->` type($portconfig) 
  }];
}

def dmap_create_io_engin_with_config : dmapop<"create_io_engin_with_config"> {
  let summary = "Configures a logical data port on a physical hardware entity.";
  let description = [{
    Binds a physical hardware handle (e.g., a tile from the routinghw dialect)
    to a logical data port, specifying its direction and a logical channel ID.
    This creates a node in the logical dataflow graph.
  }];

  let arguments = (ins
    dmapioenginetype :$ioengine,
    dataaccesspattern:$accesspattern
  );

  let results = (outs dmapioconfigtype:$ioconfig);

  let assemblyFormat = [{
     $ioengine `:` type($ioengine) attr-dict `->` type($ioconfig) 
  }];
}

def dmap_createport_group : dmapop<"create_port_group"> {
  let summary = "Groups multiple destination ports into a single multicast group.";
  let description = [{
    Takes a list of destination ports and bundles them into a single, addressable
    port group. This is used to enable one-to-many communication patterns like
    broadcast or multicast.
  }];

  let arguments = (ins
    Variadic<dmapporttype>:$ports
  );

  let results = (outs dmapportgrouptype:$group);

  let assemblyFormat = [{
    `(` $ports `)` attr-dict `:` type($ports) `->` type($group)
  }];
}

def dmap_createstream : dmapop<"create_stream"> {
  let summary = "Defines a logical data stream from a source to a destination.";
  let description = [{
    Creates an edge in the dataflow graph, explicitly connecting a source port
    with a destination port or port group. The resulting stream handle encapsulates
    the entire end-to-end path topology, simplifying execution ops.
  }];

  let arguments = (ins
    //dmapportorgroup:$source,
    AnyTypeOf<[dmapioconfigtype, dmacoregroupconfigtype]>:$source,
    AnyTypeOf<[dmapioconfigtype, dmacoregroupconfigtype]>:$destination,
    dmapioAttr:$streamType,
    I32Attr:$stream_group_index,
    I32Attr:$stream_id
  );

  let results = (outs dmapportstreamtype:$stream);

  let assemblyFormat = [{
   `src` `=` $source `,` `dst` `=` $destination `,` type($source) type($destination) attr-dict `->` type($stream)
  }];
}

// New: create a chain stream from an array of existing port streams.
def dmap_createchainstream : dmapop<"create_chain_stream"> {
  let summary = "Creates a chain/concatenated stream from multiple existing port streams.";
  let description = [{
    Takes a list of existing dmap port streams and returns a single chained stream handle.
  }];

  let arguments = (ins
    Variadic<dmapportstreamtype>:$streams
  );

  let results = (outs dmapportchainstreamtype:$chain);

  let assemblyFormat = [{
    `(` $streams `)` attr-dict `:` type($streams) `->` type($chain)
  }];
}

//===----------------------------------------------------------------------===//
// Execution Operators
//===----------------------------------------------------------------------===//

def dmap_push : dmapop<"push"> {
  let summary = "Pushes a block of data into a pre-defined stream.";
  let description = [{
    The primary execution-phase operation. It takes a data handle and
    a stream handle (either a single port stream or a chained stream),
    initiating the data transfer along the path defined by the stream.
  }];

  let arguments = (ins
    AnyTensor:$data,
    AnyTypeOf<[dmapportstreamtype, dmapportchainstreamtype]>:$stream
  );

  let assemblyFormat = [{
    $data `:` type($data) `to` $stream `:` type($stream) attr-dict
  }];

}

def dmap_pull : dmapop<"pull"> {
  let summary = "Pulls a block of data from a pre-defined stream.";
  let description = [{
    The primary pull execution-phase operation. It takes a data handle and
    a stream handle (either a single port stream or a chained stream),
    initiating the data transfer along the path defined by the stream.
  }];

  let arguments = (ins
    AnyTensor:$data,
    AnyTypeOf<[dmapportstreamtype, dmapportchainstreamtype]>:$stream
  );

  let assemblyFormat = [{
    $data `:` type($data) `from` $stream `:` type($stream) attr-dict
  }];

}
/*
def dmap_SyncOp : dmapop<"sync", [MemoryEffect_Interface]> {
  let summary = "Waits for all transfers in a stream to complete.";
  let description = [{
    Provides a synchronization point. Execution blocks until all previously
    pushed data in the specified stream has been fully delivered. This is
    essential for ensuring data is ready before computation begins.
  }];

  let arguments = (ins dmap_StreamType:$stream);

  let assemblyFormat = [{
    $stream attr-dict
  }];

  // A sync acts as a full memory barrier.
  let effects = [MemoryEffects::Read, MemoryEffects::Write];
}
*/
#endif // dmap_OPS