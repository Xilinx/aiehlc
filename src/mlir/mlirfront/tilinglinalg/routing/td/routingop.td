/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

include "routingtype.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
def routing_createtilearrayOp : routingop<"routingcreatetilearray", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
	let summary = "createtilearray";
	let cppNamespace = "routing";
	let description = "create a tile array";
	let arguments = ( ins
		I32:$rownum,
		I32:$colnum
	);
	let results = (outs
		tilearray:$output
	);
	let assemblyFormat = "$rownum $colnum `:` type($rownum) type($colnum) attr-dict `->` type($output)";
}

def routing_createdataio : routingop<"routingcreatedataio", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
	let summary = "createdataio";
	let cppNamespace = "routing";
	let description = "create data io";

	let arguments = (ins
		StrAttr:$iotype,
		StrAttr:$direction
	);
	let results = (outs
		dataio:$output
	);
	let assemblyFormat = " $iotype `,` $direction attr-dict `->` type($output)";
}

def routing_createhwiowithtarget : routingop<"routingcreatehwiowithtarget"> {
	let summary = "createhwiowithtarget";
	let cppNamespace = "routing";
	let description = "create hw io with target";
	let arguments = (ins
		I32:$targettilelist,
		StrAttr:$direction,
		StrAttr:$iotype
	);
	let results = (outs
		I32:$output
	);
	let assemblyFormat = " `targettilelist` `=` $targettilelist `:` type($targettilelist) attr-dict `->` type($output)";
}

def routing_creatbroadcast : routingop<"routingcreatebroadcast"> {
	let summary = "createbroadcast";
	let cppNamespace = "routing";
	let description = "create a broadcast";

	let arguments = (ins
		dataio:$io,
		tilearray:$tilearray);

	let results = (outs
		I32:$output);
	let assemblyFormat = "$io `,` $tilearray `:` type($io) `,` type($tilearray)  attr-dict`->` type($output)";
}

def routing_routingcontrolwrite : routingop<"routingcontrolwrite"> {
	let summary = "routingcontrolwrite";
	let cppNamespace = "routing";
	let description = [{
		High-level intent op: deliver control/config register writes (e.g.
		kernel-ELF load or generic register config) to a set of target tiles
		via stream-switch control packets. Lowered to a control-marked dmap
		flow that terminates at each target tile's CTRL port. `broadcast`
		selects whether all targets receive the same writes (shared stream id)
		or are addressed individually.
	}];

	let arguments = (ins
		dataio:$io,
		tilearray:$tilearray,
		DefaultValuedAttr<BoolAttr, "true">:$broadcast);

	let results = (outs
		I32:$output);
	let assemblyFormat = "$io `,` $tilearray `:` type($io) `,` type($tilearray) attr-dict `->` type($output)";
}

def routing_movedatabyio : routingop<"routingmovedatabyio"> {
	let summary = "movedatabyio";
	let cppNamespace = "routing";
	let description = "move data through io to aie or from";

	let arguments = (ins
		AnyType:$gatherout_tensorin,
		I32:$hwiowithtarget);

	let results = (outs
		I32:$output);
	let assemblyFormat = " `tensordata` `=` $gatherout_tensorin `,` `hwiowithtarget` `=` $hwiowithtarget `:` type($gatherout_tensorin) `,` type($hwiowithtarget)  attr-dict`->` type($output)";
}

def routing_routinggatherout : routingop<"routingroutinggatherout"> {
	let summary = "routinggatherout";
	let cppNamespace = "routing";
	let description = "gather data from a tile list";

	let arguments = (ins
		I32:$tilelist,
		AnyTensor:$tensordata);

	let results = (outs
		AnyTensor:$output);
	let assemblyFormat = " `tilelist` `=` $tilelist `,` `tensordata` `=` $tensordata `:` type($tilelist) `,` type($tensordata)  attr-dict`->` type($output)";
}

def routing_createhwmesh :routingop<"routingcreatehwmesh"> {
	let summary = "createhwmesh";
	let cppNamespace = "routing";
	let description = "create an hw mesh with optional partition bounds";

	let arguments = (ins
		I64Attr:$row,
		I64Attr:$col,
		OptionalAttr<I64Attr>:$startCol,
		OptionalAttr<I64Attr>:$endCol,
		OptionalAttr<I64Attr>:$startRow,
		OptionalAttr<I64Attr>:$endRow
		);

	let results = (outs
		I32:$output);

	let hasCustomAssemblyFormat = 1;

}

def routing_createscheduletensor :routingop<"routingcreatescheduletensor"> {
	let summary = "createscheduletensor";
	let cppNamespace = "routing";
	let description = "create a schedule tensor with initial data";

	let arguments = (ins
		AnyTensor:$init_tensor,
		I64ArrayAttr:$shape,
		I64Attr:$dim
		);

	let results = (outs
		AnyTensor:$output);

	let assemblyFormat = " $init_tensor `:` type($init_tensor) `shape` `=` $shape `,` `dim` `=` $dim attr-dict `->` type($output)";	

}

def routing_partitionmesh :routingop<"partitionmesh"> {
	let summary = "partitionmesh";
	let cppNamespace = "routing";
	let description = "partitionmesh";

	let arguments = (ins
		I32:$mesh,
		I32Attr:$splitnum,
		StrAttr:$splitaxis
		);

	let results = (outs
		I32:$output);

	let assemblyFormat = " `mesh` `=` $mesh `,` `splitnum` `=` $splitnum `,` `splitaxis` `=` $splitaxis `:` type($mesh) attr-dict `->` type($output)";	

}

def routing_partitiontensor :routingop<"partitiontensor"> {
	let summary = "partitiontensor";
	let cppNamespace = "routing";
	let description = "partitiontensor";

	let arguments = (ins
		AnyTensor:$tensor,
		PartitionAttr:$partition,
		OptionalAttr<TilingAttr>:$tiling
	);

	let results = (outs
		AnyTensor:$output);

	let hasCustomAssemblyFormat = 1;

}

def routing_extract_data :routingop<"routingextract_data"> {
	let summary = "extract_data";
	let cppNamespace = "routing";
	let description = "extract_data";

	let arguments = (ins
		AnyTensor:$tensor,
		I32:$idx
	);

	let results = (outs
		AnyTensor:$output);

	let assemblyFormat = "$tensor `,`  $idx `:` type($tensor) `,` type($idx) attr-dict `->` type($output)";	

}

def routing_extract_tiles :routingop<"routingextract_tiles"> {
	let summary = "extract_tiles";
	let cppNamespace = "routing";
	let description = "extract_tiles";

	let arguments = (ins
		I32:$partitions,
		I32:$idx
	);

	let results = (outs
		I32:$output);

	let assemblyFormat = "$partitions `,`  $idx `:` type($partitions) `,` type($idx) attr-dict `->` type($output)";	

}

def Routing_YieldOp : routingop<"yield", [Terminator]> {
  let summary = "Terminator for regions in routing ops";
  let description = [{
    The `routing.yield` operation is a terminator that yields a value from
    a region within a `routing` operation.
  }];

  //let arguments = (ins I32:$value);
  //let arguments = (ins);
  //let results = (outs); // No results
}

def Routing_RoutingCreate : routingop<"RoutingCreate", [SymbolTable]> {
  let summary = "A structured operation for routing logic";
  let description = [{
    Represents a dispatch of a value through a routing pipeline,
    configured by a static device_row attribute. The operation contains
    a region that defines the routing logic.
  }];

  let arguments = (ins 
  				   I32:$scf_idx,
				   StrAttr:$Memo
				   );
  let results = (outs I32:$output);

  let regions = (region SizedRegion<1>:$body);
 
  let builders = [
    OpBuilder<(ins "Value":$scf_idx, 
				   "Attribute":$Memo,
                   "function_ref<void(OpBuilder &, Location, Value)>":$bodyBuilder),
    [{
      // no region creatation
	  // 2. Add operands to the state object.
      $_state.addTypes($_builder.getI32Type());//for output , the result
      $_state.addOperands(scf_idx);
	  
      // 3. Add attributes to the state object.
      // Note that we must provide the attribute's name as a string.
      $_state.addAttribute("Memo", Memo);

      // 2. Region 和 Block
      auto *bodyRegion = $_state.addRegion();
      Block *bodyBlock = new Block();
      bodyRegion->push_back(bodyBlock);
      bodyBlock->addArgument(scf_idx.getType(), scf_idx.getLoc());

      // 3. use the lambda to fill Block
      OpBuilder::InsertionGuard guard($_builder);
      $_builder.setInsertionPointToStart(bodyBlock);
      bodyBuilder($_builder, 
	  			  scf_idx.getLoc(),
				  bodyBlock->getArgument(0)
				  );
    }]>
  ];

  /*
  let verifier = [{
    // verify region and argument
    if (getRegion().front().getArgument(0).getType() != getInput().getType())
      return emitOpError("region argument type must match input type");

    // verify yeild
    auto yieldOp = getRegion().front().getTerminator();
    if (yieldOp->getOperand(0).getType() != getResult().getType())
      return emitOpError("yield type must match result type");
  }];
*/
  // 
  //let assemblyFormat = [{ `(` $input `)` attr-dict `->` region $body
  //}];
  let hasCustomAssemblyFormat = 1;
}

