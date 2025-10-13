/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#ifndef Dmamap_OPS
#define Dmamap_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def Dmamap_Dialect : Dialect {
  let name = "Dmamap";
  let cppNamespace = "::mlir::Dmamap";
  let summary = "A dialect for describing high-level, logical dataflow maps.";
  let description = [{
    The Dataflow Map (Dmamap) dialect provides a high-level abstraction for
    describing the intent of data movement in a system with a distributed
    memory architecture. It separates the configuration of the logical dataflow
    graph (Ports, PortGroups, Streams) from the execution of data transfers
    on that graph.
  }];
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class Dmamap_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dmamap_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Custom Types
//===----------------------------------------------------------------------===//

class Dmamap_Type<string name, string typeMnemonic>
    : TypeDef<Dmamap_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Dmamap_PortType : Dmamap_Type<"Port", "port">;
def Dmamap_PortGroupType : Dmamap_Type<"PortGroup", "port_group">;
def Dmamap_StreamType : Dmamap_Type<"Stream", "stream">;

//===----------------------------------------------------------------------===//
// Custom Attributes
//===----------------------------------------------------------------------===//

def Dmamap_Direction : EnumAttr<Dmamap_Dialect, "Direction",
    [
      EnumAttrCase<"SEND", "send">,
      EnumAttrCase<"RECEIVE", "receive">
    ]>;

//===----------------------------------------------------------------------===//
// Configuration Operators
//===----------------------------------------------------------------------===//

def Dmamap_ConfigurePortOp : Dmamap_Op<"configure_port"> {
  let summary = "Configures a logical data port on a physical hardware entity.";
  let description = [{
    Binds a physical hardware handle (e.g., a tile from the routinghw dialect)
    to a logical data port, specifying its direction and a logical channel ID.
    This creates a node in the logical dataflow graph.
  }];

  let arguments = (ins
    // `on` 关键字后面的操作数，来自 routinghw dialect
    RoutingHw_TileType:$tile_handle,
    // `{...}` 字典中的属性
    Dmamap_DirectionAttr:$direction,
    I32Attr:$channel
  );

  let results = (outs Dmamap_PortType:$port);

  let assemblyFormat = [{
    `on` $tile_handle attr-dict `->` $port
  }];
}

def Dmamap_CreatePortGroupOp : Dmamap_Op<"create_port_group"> {
  let summary = "Groups multiple destination ports into a single multicast group.";
  let description = [{
    Takes a list of destination ports and bundles them into a single, addressable
    port group. This is used to enable one-to-many communication patterns like
    broadcast or multicast.
  }];

  let arguments = (ins
    Variadic<Dmamap_PortType>:$ports
  );

  let results = (outs Dmamap_PortGroupType:$group);

  let assemblyFormat = [{
    `(` $ports `)` attr-dict `:` type($ports) `->` $group
  }];
}

def Dmamap_CreateStreamOp : Dmamap_Op<"create_stream"> {
  let summary = "Defines a logical data stream from a source to a destination.";
  let description = [{
    Creates an edge in the dataflow graph, explicitly connecting a source port
    with a destination port or port group. The resulting stream handle encapsulates
    the entire end-to-end path topology, simplifying execution ops.
  }];

  let arguments = (ins
    Dmamap_PortType:$source,
    AnyTypeOf<[Dmamap_PortType, Dmamap_PortGroupType]>:$destination
  );

  let results = (outs Dmamap_StreamType:$stream);

  let assemblyFormat = [{
    $source `,` $destination attr-dict `->` $stream
  }];
}


//===----------------------------------------------------------------------===//
// Execution Operators
//===----------------------------------------------------------------------===//

def Dmamap_PushOp : Dmamap_Op<"push", [MemoryEffect_Interface]> {
  let summary = "Pushes a block of data into a pre-defined stream.";
  let description = [{
    The primary execution-phase operation. It takes a memref of data and
    a stream handle, initiating the data transfer along the path defined
    by the stream. This operation can be implemented asynchronously.
  }];

  let arguments = (ins
    AnyMemRef:$data,
    Dmamap_StreamType:$stream
  );

  let assemblyFormat = [{
    $data `to` $stream attr-dict
  }];

  let effects = [MemoryEffects::Write]; // This op causes writes to device memory.
}

def Dmamap_SyncOp : Dmamap_Op<"sync", [MemoryEffect_Interface]> {
  let summary = "Waits for all transfers in a stream to complete.";
  let description = [{
    Provides a synchronization point. Execution blocks until all previously
    pushed data in the specified stream has been fully delivered. This is
    essential for ensuring data is ready before computation begins.
  }];

  let arguments = (ins Dmamap_StreamType:$stream);

  let assemblyFormat = [{
    $stream attr-dict
  }];

  // A sync acts as a full memory barrier.
  let effects = [MemoryEffects::Read, MemoryEffects::Write];
}

#endif // Dmamap_OPS