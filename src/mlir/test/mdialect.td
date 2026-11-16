/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
#ifndef AIE_OPS
#define AIE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
def Aie_ADialect : Dialect {
  let name = "aie";
  let cppNamespace = "::mlir";
  dag discardableAttrs = (ins);
  let hasOperationAttrVerify = 0;
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;

}

// Base class for aie dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Aie_Op<string mnemonic, list<Trait> traits = []> :
    Op<Aie_ADialect, mnemonic, traits>;

def Aie_LoadKernelOp: Aie_Op<"loadkernel"> {
	let summary = "aie op";
  	let cppNamespace = "::mlir";
	let arguments = (ins I32:$lhs, I32:$rhs);
	let results = (outs);
  /*	let builders = [
    		OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  	];
*/
	//let hasVerifier = 1;
	 let description = [{
    "This is my custom operation."
  }];

}
def ConstantOp : Aie_Op<"constant", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = aie.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 0;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 0;
}
#endif
