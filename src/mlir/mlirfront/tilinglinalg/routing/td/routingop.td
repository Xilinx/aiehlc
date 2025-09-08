/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
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

def routing_createhwmesh :routingop<"routingcreatehwmesh"> {
	let summary = "createhwmesh";
	let cppNamespace = "routing";
	let description = "create an hw mesh";

	let arguments = (ins
		I64Attr:$row,
		I64Attr:$col
		);

	let results = (outs
		I32:$output);

	let assemblyFormat = " attr-dict `->` type($output)";	

}

def routing_createdummytensor :routingop<"routingcreatedummytensor"> {
	let summary = "createdummytensor";
	let cppNamespace = "routing";
	let description = "create an dummpy tensor";

	let arguments = (ins
		I64ArrayAttr:$shape,
		I64Attr:$dim
		);

	let results = (outs
		I32:$output);

	let assemblyFormat = "attr-dict `->` type($output)";	

}

def routing_partitionmesh :routingop<"routingpartitionmesh"> {
	let summary = "partitionmesh";
	let cppNamespace = "routing";
	let description = "partitionmesh";

	let arguments = (ins
		I32:$mesh,
		I32:$splitnum,
		I32:$axisidx
		);

	let results = (outs
		I32:$output);

	let assemblyFormat = [{`mesh` `=` $mesh`,` `splitnum` `=` $splitnum `,` `axisidx` `=`  $axisidx `:` type($mesh) `,` type($splitnum) `,` type($axisidx) attr-dict `->` type($output)}];	

}

def routing_partitiontensor :routingop<"routingpartitiontensor"> {
	let summary = "partitiontensor";
	let cppNamespace = "routing";
	let description = "partitionmtensor";

	let arguments = (ins
		I32:$tensor,
		I32:$splitnum,
		I32:$splitdim,
		I32:$hw_axis_owner,
		I32:$replicate_on,
		I32:$single_tile_owner
	);

	let results = (outs
		I32:$output);
	
	let assemblyFormat = [{
		`tensor` `=` $tensor `,`
		`splitnum` `=` $splitnum `,`
		`splitdim` `=` $splitdim `,`
		`hw_axis_owner` `=` $hw_axis_owner `,`
		`replicate_on` `=` $replicate_on `,`
		`single_tile_owner` `=` $single_tile_owner
		`:` type($tensor) `,` type($splitnum) `,` type($splitdim) `,`
			type($hw_axis_owner) `,` type($replicate_on) `,` type($single_tile_owner)
		attr-dict `->` type($output)
	}];	

}

def routing_extract_data :routingop<"routingextract_data"> {
	let summary = "extract_data";
	let cppNamespace = "routing";
	let description = "extract_data";

	let arguments = (ins
		I32:$tensor,
		I32:$idx
	);

	let results = (outs
		I32:$output);

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
