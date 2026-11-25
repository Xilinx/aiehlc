/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
#ifndef AIE_TYPES
#define AIE_TYPES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/OpBase.td" // This line is crucial
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"

// Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
def Aie_ADialect : Dialect {
    // keep the pre-LLVM-24 textual form (inherent attrs in attr-dict)
    let useStrictPropertiesInAssemblyFormat = 0;
  let name = "Aie";
  let cppNamespace = "::mlir::aie";
  dag discardableAttrs = (ins);
  let hasOperationAttrVerify = 0;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
}

class Aie_Type<string name> : TypeDef<Aie_ADialect, name> {}
class Aie_Attr<string name> : AttrDef<Aie_ADialect, name> {}

def Aie_KernelFuncNameType : Aie_Type<"KernelFuncName"> {
	let mnemonic = "KernelFuncName";
	let summary = "the kernel function name";
	let description = [{the kernel function name}];
	let parameters = (ins
		"StringAttr":$name
	);
	let assemblyFormat = [{`<` $name `>`}];

}

def Aie_KernelFileNameType : Aie_Type<"KernelFileName"> {
	let mnemonic = "KernelFileName";
	let summary = "the kernel path";
	let description = [{kernel file path}];
	let parameters = (ins "StringAttr":$name);
	let assemblyFormat = [{`<` $name `>`}];
}

def Aie_WindowType : Aie_Type<"Window"> {
  let mnemonic = "Window";
  let summary = "Window type with direction, size, and name";
  let description = [{
    A type encapsulating a window with a specific direction, size, and name.
    These attributes are constant and determined at the time of window creation.
  }];

  let parameters = (ins
    "int":$direction,
    "int":$size,
    "long":$pingaddr,
    "long":$pongaddr,
    "int":$pinglockid,
    "int":$ponglockid,
    "StringAttr":$name
  );

  let assemblyFormat = [{ `<` $direction `,` $size `,` $pingaddr `,` $pongaddr `,` $pinglockid 
    `,` $ponglockid `,` $name `>` }];
}

def Aie_KernelObjectType : Aie_Type<"Kernel"> {
  let mnemonic = "KernelObj";
  let description = [{
    Represents a kernel object in the AIE dialect. It encapsulates a kernel symbol reference,
    the number of input and output arguments, and a variadic array of window types for the arguments.
    Specifically this object collects all the static information about the kernel which 
    would be useful in constructing full kernel source code.
  }];

  let parameters = (ins
    "int":$numInputArgs,
    "int":$numOutputArgs,
		"KernelFuncNameType":$kname,
    "KernelFileNameType":$fname,
    OptionalArrayRefParameter<"WindowType">:$arguments
  );

  let assemblyFormat = [{ `<` $numInputArgs `,` $numOutputArgs `,` $kname `,` $fname `,` $arguments  `>` }];

}

def Aie_TileKernelObjectType : Aie_Type<"TileKernel"> {
  let mnemonic = "TileKernelObj";
  let description = [{
    Represents a kernel tile object in the AIE dialect. It encapsulates a kernel symbol reference,
    the number of input and output arguments, and a variadic array of window types for the arguments.
    Specifically this object collects all the static information about the kernel which would be useful in 
    constructing full kernel source code.
  }];

  let parameters = (ins
    "int":$col,
    "int":$row,
    "StringAttr":$kernels
  );
  
  // Investigate: It does not show !Aie.Mneumonic in the MLIR representation
  // Is it due to MLIR latest changes in default printer or parser ?
  // Or am I missing something. Works fine.
  let assemblyFormat = [{ `<` $col `,` $row `,` $kernels `>` }];
}

def Aie_LocType : Aie_Type<"Loc"> {
  let mnemonic = "Loc";
  let description = [{
    Represents a location of a tile within AIE array (col, row)
  }];
}


def Aie_CoreConstraintType : Aie_Type<"CoreConstraint"> {
  let mnemonic = "CoreConstraint";
  let description = [{
    Represents Core Constraint struct.
  }];
}

def Aie_RoutingInstanceType : Aie_Type<"RoutingInstance"> {
  let mnemonic = "RoutingInstance";
  let description = [{
    Represents Routing Instance struct.
  }];
}

def Aie_HostAieConstraintType : Aie_Type<"HostAieConstraint"> {
  let mnemonic = "HostAieConstraint";
  let description = [{
    Represents Host Aie Constraint.
  }];
}

def Aie_RoutingConstraintType : Aie_Type<"RoutingConstraint"> {
  let mnemonic = "RoutingConstraint";
  let description = [{
    Represents a routing constraint in the AIE dialect.
  }];
}

def Aie_DeviceInstanceType : Aie_Type<"DeviceInstance"> {
  let mnemonic = "DeviceInstance";
  let description = [{
    Represents The Device Instance.
  }];
}

#endif

