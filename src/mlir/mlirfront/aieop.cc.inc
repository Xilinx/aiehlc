/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: aieops.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::aie::AllocOp,
::mlir::aie::ConfigHostEdgeConstraints,
::mlir::aie::CoreWait,
::mlir::aie::CreateKernelFileNameOp,
::mlir::aie::CreateKernelFuncNameOp,
::mlir::aie::CreateKernelObjectOp,
::mlir::aie::CreateTileKernelObjectOp,
::mlir::aie::CreateWindowOp,
::mlir::aie::DeRoute,
::mlir::aie::InitRoutingHandler,
::mlir::aie::KernelCall,
::mlir::aie::KernelDefinition,
::mlir::aie::LoadKernel,
::mlir::aie::LoadKernelOp,
::mlir::aie::MoveData,
::mlir::aie::ResetHostEdgeConstraints,
::mlir::aie::Route,
::mlir::aie::RouteDmaWait,
::mlir::aie::RoutesReveal,
::mlir::aie::RoutingInstanceFree,
::mlir::aie::Run,
::mlir::aie::SetCoreExecute,
::mlir::aie::ConstantInteger32Op,
::mlir::aie::ConstantInteger64Op,
::mlir::aie::ConstantOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace aie {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(64)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 64-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return (true); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::RoutingInstanceType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::HostAieConstraintType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::LocType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::KernelFileNameType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be the kernel path, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::KernelFuncNameType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be the kernel function name, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::KernelType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::TileKernelType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::WindowType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Window type with direction, size, and name, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::DeviceInstanceType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::RoutingConstraintType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aieops16(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 64-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aieops0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aieops0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aieops0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aieops1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aieops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aieops1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aieops2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aieops2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aieops2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aieops3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DenseFPElementsAttr>(attr) &&::llvm::cast<::mlir::DenseElementsAttr>(attr).getType().getElementType().isF64())))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit float elements attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aieops3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aieops3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace aie
} // namespace mlir
namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::AllocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
AllocOpAdaptor::AllocOpAdaptor(AllocOp op) : AllocOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult AllocOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value size) {
  odsState.addOperands(size);
  odsState.addTypes(result);
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value size) {
  odsState.addOperands(size);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AllocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sizeRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sizeOperands(&sizeRawOperand, 1);  ::llvm::SMLoc sizeOperandsLoc;
  (void)sizeOperandsLoc;
  ::mlir::Type resultRawType{};
  ::llvm::ArrayRef<::mlir::Type> resultTypes(&resultRawType, 1);
  if (parser.parseLParen())
    return ::mlir::failure();

  sizeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sizeRawOperand))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawType = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(64);
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sizeOperands, odsBuildableType0, sizeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AllocOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getSize();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::MemRefType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AllocOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::AllocOp)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::ConfigHostEdgeConstraints definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ConfigHostEdgeConstraintsAdaptor::ConfigHostEdgeConstraintsAdaptor(ConfigHostEdgeConstraints op) : ConfigHostEdgeConstraintsGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ConfigHostEdgeConstraintsAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ConfigHostEdgeConstraints::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value userResCon) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(userResCon);
}

void ConfigHostEdgeConstraints::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value userResCon) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(userResCon);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConfigHostEdgeConstraints::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConfigHostEdgeConstraints::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConfigHostEdgeConstraints::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ConfigHostEdgeConstraints::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ConfigHostEdgeConstraints)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::CoreWait definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
CoreWaitAdaptor::CoreWaitAdaptor(CoreWait op) : CoreWaitGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult CoreWaitAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void CoreWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
}

void CoreWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoreWait::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoreWait::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoreWait::verifyInvariants() {
  return verifyInvariantsImpl();
}

void CoreWait::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CoreWait)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelFileNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::StringAttr CreateKernelFileNameOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CreateKernelFileNameOp::getNameAttrName(*odsOpName)));
  return attr;
}

::llvm::StringRef CreateKernelFileNameOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
CreateKernelFileNameOpAdaptor::CreateKernelFileNameOpAdaptor(CreateKernelFileNameOp op) : CreateKernelFileNameOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult CreateKernelFileNameOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_kernelfilename_op' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == CreateKernelFileNameOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'Aie.create_kernelfilename_op' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::StringRef CreateKernelFileNameOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CreateKernelFileNameOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr name) {
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addTypes(result);
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name) {
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name) {
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addTypes(result);
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateKernelFileNameOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'name'");
    if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateKernelFileNameOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateKernelFileNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;
  if (parser.parseLBrace())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.attributes.append("name", nameAttr);
  if (parser.parseRBrace())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

void CreateKernelFileNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "{";
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << "}";
  _odsPrinter << ' ' << "(";
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateKernelFileNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelFileNameOp)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelFuncNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::StringAttr CreateKernelFuncNameOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CreateKernelFuncNameOp::getNameAttrName(*odsOpName)));
  return attr;
}

::llvm::StringRef CreateKernelFuncNameOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
CreateKernelFuncNameOpAdaptor::CreateKernelFuncNameOpAdaptor(CreateKernelFuncNameOp op) : CreateKernelFuncNameOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult CreateKernelFuncNameOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_kernelfuncname_op' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == CreateKernelFuncNameOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'Aie.create_kernelfuncname_op' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::StringRef CreateKernelFuncNameOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CreateKernelFuncNameOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr name) {
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addTypes(result);
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name) {
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name) {
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addTypes(result);
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateKernelFuncNameOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'name'");
    if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateKernelFuncNameOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateKernelFuncNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;
  if (parser.parseLBrace())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.attributes.append("name", nameAttr);
  if (parser.parseRBrace())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

void CreateKernelFuncNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "{";
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << "}";
  _odsPrinter << ' ' << "(";
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateKernelFuncNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelFuncNameOp)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelObjectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> CreateKernelObjectOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::IntegerAttr CreateKernelObjectOpGenericAdaptorBase::getNumInputArgsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, CreateKernelObjectOp::getNumInputArgsAttrName(*odsOpName)));
  return attr;
}

uint32_t CreateKernelObjectOpGenericAdaptorBase::getNumInputArgs() {
  auto attr = getNumInputArgsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr CreateKernelObjectOpGenericAdaptorBase::getNumOutputArgsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, CreateKernelObjectOp::getNumOutputArgsAttrName(*odsOpName)));
  return attr;
}

uint32_t CreateKernelObjectOpGenericAdaptorBase::getNumOutputArgs() {
  auto attr = getNumOutputArgsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr CreateKernelObjectOpGenericAdaptorBase::getInParamTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, CreateKernelObjectOp::getInParamTypeAttrName(*odsOpName)));
  return attr;
}

::llvm::StringRef CreateKernelObjectOpGenericAdaptorBase::getInParamType() {
  auto attr = getInParamTypeAttr();
  return attr.getValue();
}

::mlir::StringAttr CreateKernelObjectOpGenericAdaptorBase::getOutParamTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, CreateKernelObjectOp::getOutParamTypeAttrName(*odsOpName)));
  return attr;
}

::llvm::StringRef CreateKernelObjectOpGenericAdaptorBase::getOutParamType() {
  auto attr = getOutParamTypeAttr();
  return attr.getValue();
}

} // namespace detail
CreateKernelObjectOpAdaptor::CreateKernelObjectOpAdaptor(CreateKernelObjectOp op) : CreateKernelObjectOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult CreateKernelObjectOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_in_param_type;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_kernelobject_op' op ""requires attribute 'in_param_type'");
    if (namedAttrIt->getName() == CreateKernelObjectOp::getInParamTypeAttrName(*odsOpName)) {
      tblgen_in_param_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_numInputArgs;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_kernelobject_op' op ""requires attribute 'numInputArgs'");
    if (namedAttrIt->getName() == CreateKernelObjectOp::getNumInputArgsAttrName(*odsOpName)) {
      tblgen_numInputArgs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_numOutputArgs;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_kernelobject_op' op ""requires attribute 'numOutputArgs'");
    if (namedAttrIt->getName() == CreateKernelObjectOp::getNumOutputArgsAttrName(*odsOpName)) {
      tblgen_numOutputArgs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_out_param_type;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_kernelobject_op' op ""requires attribute 'out_param_type'");
    if (namedAttrIt->getName() == CreateKernelObjectOp::getOutParamTypeAttrName(*odsOpName)) {
      tblgen_out_param_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_numInputArgs && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_numInputArgs))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_numInputArgs).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_kernelobject_op' op ""attribute 'numInputArgs' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_numOutputArgs && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_numOutputArgs))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_numOutputArgs).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_kernelobject_op' op ""attribute 'numOutputArgs' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_in_param_type && !((::llvm::isa<::mlir::StringAttr>(tblgen_in_param_type))))
    return emitError(loc, "'Aie.create_kernelobject_op' op ""attribute 'in_param_type' failed to satisfy constraint: string attribute");

  if (tblgen_out_param_type && !((::llvm::isa<::mlir::StringAttr>(tblgen_out_param_type))))
    return emitError(loc, "'Aie.create_kernelobject_op' op ""attribute 'out_param_type' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateKernelObjectOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange CreateKernelObjectOp::getArgumentsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

uint32_t CreateKernelObjectOp::getNumInputArgs() {
  auto attr = getNumInputArgsAttr();
  return attr.getValue().getZExtValue();
}

uint32_t CreateKernelObjectOp::getNumOutputArgs() {
  auto attr = getNumOutputArgsAttr();
  return attr.getValue().getZExtValue();
}

::llvm::StringRef CreateKernelObjectOp::getInParamType() {
  auto attr = getInParamTypeAttr();
  return attr.getValue();
}

::llvm::StringRef CreateKernelObjectOp::getOutParamType() {
  auto attr = getOutParamTypeAttr();
  return attr.getValue();
}

void CreateKernelObjectOp::setNumInputArgs(uint32_t attrValue) {
  (*this)->setAttr(getNumInputArgsAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CreateKernelObjectOp::setNumOutputArgs(uint32_t attrValue) {
  (*this)->setAttr(getNumOutputArgsAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CreateKernelObjectOp::setInParamType(::llvm::StringRef attrValue) {
  (*this)->setAttr(getInParamTypeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CreateKernelObjectOp::setOutParamType(::llvm::StringRef attrValue) {
  (*this)->setAttr(getOutParamTypeAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, IntegerAttr numInputArgs, IntegerAttr numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, std::string in_param_type, std::string out_param_type, ValueRange arguments) {
      ValueRange args = ValueRange(arguments);
      build(odsBuilder, odsState, result_type, numInputArgs, numOutputArgs,kname, fname, in_param_type, out_param_type, arguments);// args);
    
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments) {
  odsState.addOperands(kname);
  odsState.addOperands(fname);
  odsState.addOperands(arguments);
  odsState.addAttribute(getNumInputArgsAttrName(odsState.name), numInputArgs);
  odsState.addAttribute(getNumOutputArgsAttrName(odsState.name), numOutputArgs);
  odsState.addAttribute(getInParamTypeAttrName(odsState.name), in_param_type);
  odsState.addAttribute(getOutParamTypeAttrName(odsState.name), out_param_type);
  odsState.addTypes(result);
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments) {
  odsState.addOperands(kname);
  odsState.addOperands(fname);
  odsState.addOperands(arguments);
  odsState.addAttribute(getNumInputArgsAttrName(odsState.name), numInputArgs);
  odsState.addAttribute(getNumOutputArgsAttrName(odsState.name), numOutputArgs);
  odsState.addAttribute(getInParamTypeAttrName(odsState.name), in_param_type);
  odsState.addAttribute(getOutParamTypeAttrName(odsState.name), out_param_type);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments) {
  odsState.addOperands(kname);
  odsState.addOperands(fname);
  odsState.addOperands(arguments);
  odsState.addAttribute(getNumInputArgsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numInputArgs));
  odsState.addAttribute(getNumOutputArgsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numOutputArgs));
  odsState.addAttribute(getInParamTypeAttrName(odsState.name), odsBuilder.getStringAttr(in_param_type));
  odsState.addAttribute(getOutParamTypeAttrName(odsState.name), odsBuilder.getStringAttr(out_param_type));
  odsState.addTypes(result);
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments) {
  odsState.addOperands(kname);
  odsState.addOperands(fname);
  odsState.addOperands(arguments);
  odsState.addAttribute(getNumInputArgsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numInputArgs));
  odsState.addAttribute(getNumOutputArgsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numOutputArgs));
  odsState.addAttribute(getInParamTypeAttrName(odsState.name), odsBuilder.getStringAttr(in_param_type));
  odsState.addAttribute(getOutParamTypeAttrName(odsState.name), odsBuilder.getStringAttr(out_param_type));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateKernelObjectOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_in_param_type;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'in_param_type'");
    if (namedAttrIt->getName() == getInParamTypeAttrName()) {
      tblgen_in_param_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_numInputArgs;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'numInputArgs'");
    if (namedAttrIt->getName() == getNumInputArgsAttrName()) {
      tblgen_numInputArgs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_numOutputArgs;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'numOutputArgs'");
    if (namedAttrIt->getName() == getNumOutputArgsAttrName()) {
      tblgen_numOutputArgs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_out_param_type;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'out_param_type'");
    if (namedAttrIt->getName() == getOutParamTypeAttrName()) {
      tblgen_out_param_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_numInputArgs, "numInputArgs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_numOutputArgs, "numOutputArgs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops0(*this, tblgen_in_param_type, "in_param_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops0(*this, tblgen_out_param_type, "out_param_type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateKernelObjectOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateKernelObjectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr numInputArgsAttr;
  ::mlir::IntegerAttr numOutputArgsAttr;
  ::mlir::OpAsmParser::UnresolvedOperand knameRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> knameOperands(&knameRawOperand, 1);  ::llvm::SMLoc knameOperandsLoc;
  (void)knameOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand fnameRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> fnameOperands(&fnameRawOperand, 1);  ::llvm::SMLoc fnameOperandsLoc;
  (void)fnameOperandsLoc;
  ::mlir::StringAttr in_param_typeAttr;
  ::mlir::StringAttr out_param_typeAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argumentsOperands;
  ::llvm::SMLoc argumentsOperandsLoc;
  (void)argumentsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  if (parser.parseLBrace())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(numInputArgsAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (numInputArgsAttr) result.attributes.append("numInputArgs", numInputArgsAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(numOutputArgsAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (numOutputArgsAttr) result.attributes.append("numOutputArgs", numOutputArgsAttr);
  if (parser.parseRBrace())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  knameOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(knameRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  fnameOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(fnameRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(in_param_typeAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (in_param_typeAttr) result.attributes.append("in_param_type", in_param_typeAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(out_param_typeAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (out_param_typeAttr) result.attributes.append("out_param_type", out_param_typeAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  argumentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argumentsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(knameOperands, fnameOperands, argumentsOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateKernelObjectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "{";
  _odsPrinter.printAttributeWithoutType(getNumInputArgsAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNumOutputArgsAttr());
  _odsPrinter << "}";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("numInputArgs");
  elidedAttrs.push_back("numOutputArgs");
  elidedAttrs.push_back("in_param_type");
  elidedAttrs.push_back("out_param_type");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << "(";
  _odsPrinter << getKname();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFname();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getInParamTypeAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getOutParamTypeAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getArguments();
  _odsPrinter << ")";
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void CreateKernelObjectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelObjectOp)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateTileKernelObjectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::StringAttr CreateTileKernelObjectOpGenericAdaptorBase::getKernelsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CreateTileKernelObjectOp::getKernelsAttrName(*odsOpName)));
  return attr;
}

::llvm::StringRef CreateTileKernelObjectOpGenericAdaptorBase::getKernels() {
  auto attr = getKernelsAttr();
  return attr.getValue();
}

} // namespace detail
CreateTileKernelObjectOpAdaptor::CreateTileKernelObjectOpAdaptor(CreateTileKernelObjectOp op) : CreateTileKernelObjectOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult CreateTileKernelObjectOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kernels;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_tilekernelobject_op' op ""requires attribute 'kernels'");
    if (namedAttrIt->getName() == CreateTileKernelObjectOp::getKernelsAttrName(*odsOpName)) {
      tblgen_kernels = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_kernels && !((::llvm::isa<::mlir::StringAttr>(tblgen_kernels))))
    return emitError(loc, "'Aie.create_tilekernelobject_op' op ""attribute 'kernels' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::StringRef CreateTileKernelObjectOp::getKernels() {
  auto attr = getKernelsAttr();
  return attr.getValue();
}

void CreateTileKernelObjectOp::setKernels(::llvm::StringRef attrValue) {
  (*this)->setAttr(getKernelsAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels) {
  odsState.addOperands(row);
  odsState.addOperands(col);
  odsState.addAttribute(getKernelsAttrName(odsState.name), kernels);
  odsState.addTypes(result);
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels) {
  odsState.addOperands(row);
  odsState.addOperands(col);
  odsState.addAttribute(getKernelsAttrName(odsState.name), kernels);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels) {
  odsState.addOperands(row);
  odsState.addOperands(col);
  odsState.addAttribute(getKernelsAttrName(odsState.name), odsBuilder.getStringAttr(kernels));
  odsState.addTypes(result);
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels) {
  odsState.addOperands(row);
  odsState.addOperands(col);
  odsState.addAttribute(getKernelsAttrName(odsState.name), odsBuilder.getStringAttr(kernels));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateTileKernelObjectOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kernels;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'kernels'");
    if (namedAttrIt->getName() == getKernelsAttrName()) {
      tblgen_kernels = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops0(*this, tblgen_kernels, "kernels")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateTileKernelObjectOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateTileKernelObjectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand rowRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rowOperands(&rowRawOperand, 1);  ::llvm::SMLoc rowOperandsLoc;
  (void)rowOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand colRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> colOperands(&colRawOperand, 1);  ::llvm::SMLoc colOperandsLoc;
  (void)colOperandsLoc;
  ::mlir::StringAttr kernelsAttr;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  rowOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rowRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  colOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(colRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(kernelsAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (kernelsAttr) result.attributes.append("kernels", kernelsAttr);
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(rowOperands, odsBuildableType0, rowOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(colOperands, odsBuildableType0, colOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateTileKernelObjectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getRow();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getCol();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getKernelsAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("kernels");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateTileKernelObjectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateTileKernelObjectOp)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateWindowOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::IntegerAttr CreateWindowOpGenericAdaptorBase::getDirectionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 6, CreateWindowOp::getDirectionAttrName(*odsOpName)));
  return attr;
}

uint32_t CreateWindowOpGenericAdaptorBase::getDirection() {
  auto attr = getDirectionAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr CreateWindowOpGenericAdaptorBase::getSizeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 6, odsAttrs.end() - 0, CreateWindowOp::getSizeAttrName(*odsOpName)));
  return attr;
}

uint32_t CreateWindowOpGenericAdaptorBase::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr CreateWindowOpGenericAdaptorBase::getPingaddrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 4, CreateWindowOp::getPingaddrAttrName(*odsOpName)));
  return attr;
}

uint64_t CreateWindowOpGenericAdaptorBase::getPingaddr() {
  auto attr = getPingaddrAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr CreateWindowOpGenericAdaptorBase::getPongaddrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 2, CreateWindowOp::getPongaddrAttrName(*odsOpName)));
  return attr;
}

uint64_t CreateWindowOpGenericAdaptorBase::getPongaddr() {
  auto attr = getPongaddrAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr CreateWindowOpGenericAdaptorBase::getPinglockidAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 3, CreateWindowOp::getPinglockidAttrName(*odsOpName)));
  return attr;
}

uint32_t CreateWindowOpGenericAdaptorBase::getPinglockid() {
  auto attr = getPinglockidAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr CreateWindowOpGenericAdaptorBase::getPonglockidAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 5, odsAttrs.end() - 1, CreateWindowOp::getPonglockidAttrName(*odsOpName)));
  return attr;
}

uint32_t CreateWindowOpGenericAdaptorBase::getPonglockid() {
  auto attr = getPonglockidAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr CreateWindowOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 5, CreateWindowOp::getNameAttrName(*odsOpName)));
  return attr;
}

::llvm::StringRef CreateWindowOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
CreateWindowOpAdaptor::CreateWindowOpAdaptor(CreateWindowOp op) : CreateWindowOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult CreateWindowOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_direction;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_window' op ""requires attribute 'direction'");
    if (namedAttrIt->getName() == CreateWindowOp::getDirectionAttrName(*odsOpName)) {
      tblgen_direction = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_name;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_window' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == CreateWindowOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pingaddr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_window' op ""requires attribute 'pingaddr'");
    if (namedAttrIt->getName() == CreateWindowOp::getPingaddrAttrName(*odsOpName)) {
      tblgen_pingaddr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pinglockid;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_window' op ""requires attribute 'pinglockid'");
    if (namedAttrIt->getName() == CreateWindowOp::getPinglockidAttrName(*odsOpName)) {
      tblgen_pinglockid = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pongaddr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_window' op ""requires attribute 'pongaddr'");
    if (namedAttrIt->getName() == CreateWindowOp::getPongaddrAttrName(*odsOpName)) {
      tblgen_pongaddr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_ponglockid;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_window' op ""requires attribute 'ponglockid'");
    if (namedAttrIt->getName() == CreateWindowOp::getPonglockidAttrName(*odsOpName)) {
      tblgen_ponglockid = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_size;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.create_window' op ""requires attribute 'size'");
    if (namedAttrIt->getName() == CreateWindowOp::getSizeAttrName(*odsOpName)) {
      tblgen_size = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_direction && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_direction))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_direction).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_window' op ""attribute 'direction' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_size && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_size))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_size).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_window' op ""attribute 'size' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_pingaddr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pingaddr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pingaddr).getType().isSignlessInteger(64)))))
    return emitError(loc, "'Aie.create_window' op ""attribute 'pingaddr' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_pongaddr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pongaddr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pongaddr).getType().isSignlessInteger(64)))))
    return emitError(loc, "'Aie.create_window' op ""attribute 'pongaddr' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_pinglockid && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pinglockid))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pinglockid).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_window' op ""attribute 'pinglockid' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_ponglockid && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_ponglockid))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_ponglockid).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_window' op ""attribute 'ponglockid' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'Aie.create_window' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

uint32_t CreateWindowOp::getDirection() {
  auto attr = getDirectionAttr();
  return attr.getValue().getZExtValue();
}

uint32_t CreateWindowOp::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

uint64_t CreateWindowOp::getPingaddr() {
  auto attr = getPingaddrAttr();
  return attr.getValue().getZExtValue();
}

uint64_t CreateWindowOp::getPongaddr() {
  auto attr = getPongaddrAttr();
  return attr.getValue().getZExtValue();
}

uint32_t CreateWindowOp::getPinglockid() {
  auto attr = getPinglockidAttr();
  return attr.getValue().getZExtValue();
}

uint32_t CreateWindowOp::getPonglockid() {
  auto attr = getPonglockidAttr();
  return attr.getValue().getZExtValue();
}

::llvm::StringRef CreateWindowOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CreateWindowOp::setDirection(uint32_t attrValue) {
  (*this)->setAttr(getDirectionAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CreateWindowOp::setSize(uint32_t attrValue) {
  (*this)->setAttr(getSizeAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CreateWindowOp::setPingaddr(uint64_t attrValue) {
  (*this)->setAttr(getPingaddrAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void CreateWindowOp::setPongaddr(uint64_t attrValue) {
  (*this)->setAttr(getPongaddrAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void CreateWindowOp::setPinglockid(uint32_t attrValue) {
  (*this)->setAttr(getPinglockidAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CreateWindowOp::setPonglockid(uint32_t attrValue) {
  (*this)->setAttr(getPonglockidAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CreateWindowOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CreateWindowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name) {
  odsState.addAttribute(getDirectionAttrName(odsState.name), direction);
  odsState.addAttribute(getSizeAttrName(odsState.name), size);
  odsState.addAttribute(getPingaddrAttrName(odsState.name), pingaddr);
  odsState.addAttribute(getPongaddrAttrName(odsState.name), pongaddr);
  odsState.addAttribute(getPinglockidAttrName(odsState.name), pinglockid);
  odsState.addAttribute(getPonglockidAttrName(odsState.name), ponglockid);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addTypes(result);
}

void CreateWindowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name) {
  odsState.addAttribute(getDirectionAttrName(odsState.name), direction);
  odsState.addAttribute(getSizeAttrName(odsState.name), size);
  odsState.addAttribute(getPingaddrAttrName(odsState.name), pingaddr);
  odsState.addAttribute(getPongaddrAttrName(odsState.name), pongaddr);
  odsState.addAttribute(getPinglockidAttrName(odsState.name), pinglockid);
  odsState.addAttribute(getPonglockidAttrName(odsState.name), ponglockid);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateWindowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name) {
  odsState.addAttribute(getDirectionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), direction));
  odsState.addAttribute(getSizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size));
  odsState.addAttribute(getPingaddrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pingaddr));
  odsState.addAttribute(getPongaddrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pongaddr));
  odsState.addAttribute(getPinglockidAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pinglockid));
  odsState.addAttribute(getPonglockidAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), ponglockid));
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addTypes(result);
}

void CreateWindowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name) {
  odsState.addAttribute(getDirectionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), direction));
  odsState.addAttribute(getSizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size));
  odsState.addAttribute(getPingaddrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pingaddr));
  odsState.addAttribute(getPongaddrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pongaddr));
  odsState.addAttribute(getPinglockidAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pinglockid));
  odsState.addAttribute(getPonglockidAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), ponglockid));
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateWindowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateWindowOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_direction;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'direction'");
    if (namedAttrIt->getName() == getDirectionAttrName()) {
      tblgen_direction = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_name;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'name'");
    if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pingaddr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pingaddr'");
    if (namedAttrIt->getName() == getPingaddrAttrName()) {
      tblgen_pingaddr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pinglockid;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pinglockid'");
    if (namedAttrIt->getName() == getPinglockidAttrName()) {
      tblgen_pinglockid = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_pongaddr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'pongaddr'");
    if (namedAttrIt->getName() == getPongaddrAttrName()) {
      tblgen_pongaddr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_ponglockid;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'ponglockid'");
    if (namedAttrIt->getName() == getPonglockidAttrName()) {
      tblgen_ponglockid = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_size;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'size'");
    if (namedAttrIt->getName() == getSizeAttrName()) {
      tblgen_size = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_direction, "direction")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_size, "size")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_pingaddr, "pingaddr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_pongaddr, "pongaddr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_pinglockid, "pinglockid")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_ponglockid, "ponglockid")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops11(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateWindowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateWindowOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr directionAttr;
  ::mlir::IntegerAttr sizeAttr;
  ::mlir::IntegerAttr pingaddrAttr;
  ::mlir::IntegerAttr pongaddrAttr;
  ::mlir::IntegerAttr pinglockidAttr;
  ::mlir::IntegerAttr ponglockidAttr;
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(directionAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (directionAttr) result.attributes.append("direction", directionAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(sizeAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (sizeAttr) result.attributes.append("size", sizeAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(pingaddrAttr, parser.getBuilder().getIntegerType(64))) {
    return ::mlir::failure();
  }
  if (pingaddrAttr) result.attributes.append("pingaddr", pingaddrAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(pongaddrAttr, parser.getBuilder().getIntegerType(64))) {
    return ::mlir::failure();
  }
  if (pongaddrAttr) result.attributes.append("pongaddr", pongaddrAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(pinglockidAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (pinglockidAttr) result.attributes.append("pinglockid", pinglockidAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(ponglockidAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (ponglockidAttr) result.attributes.append("ponglockid", ponglockidAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.attributes.append("name", nameAttr);
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

void CreateWindowOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getDirectionAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getSizeAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getPingaddrAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getPongaddrAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getPinglockidAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getPonglockidAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("direction");
  elidedAttrs.push_back("size");
  elidedAttrs.push_back("pingaddr");
  elidedAttrs.push_back("pongaddr");
  elidedAttrs.push_back("pinglockid");
  elidedAttrs.push_back("ponglockid");
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateWindowOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateWindowOp)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::DeRoute definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
DeRouteAdaptor::DeRouteAdaptor(DeRoute op) : DeRouteGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult DeRouteAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void DeRoute::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(source);
  odsState.addOperands(destination);
  odsState.addOperands(shouldModifyCoreConfig);
}

void DeRoute::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(source);
  odsState.addOperands(destination);
  odsState.addOperands(shouldModifyCoreConfig);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeRoute::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DeRoute::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DeRoute::verifyInvariants() {
  return verifyInvariantsImpl();
}

void DeRoute::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::DeRoute)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::InitRoutingHandler definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
InitRoutingHandlerAdaptor::InitRoutingHandlerAdaptor(InitRoutingHandler op) : InitRoutingHandlerGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult InitRoutingHandlerAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void InitRoutingHandler::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type instance, ::mlir::Value devInst) {
  odsState.addOperands(devInst);
  odsState.addTypes(instance);
}

void InitRoutingHandler::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value devInst) {
  odsState.addOperands(devInst);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InitRoutingHandler::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InitRoutingHandler::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InitRoutingHandler::verifyInvariants() {
  return verifyInvariantsImpl();
}

void InitRoutingHandler::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::InitRoutingHandler)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::KernelCall definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> KernelCallGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
KernelCallAdaptor::KernelCallAdaptor(KernelCall op) : KernelCallGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult KernelCallAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> KernelCall::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange KernelCall::getArgumentsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

void KernelCall::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, Value col, Value row, ArrayRef<Value> arguments) {
      ValueRange args = ValueRange(arguments);
      build(odsBuilder, odsState, result_type, col, row, args);
    
}

void KernelCall::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments) {
  odsState.addOperands(col);
  odsState.addOperands(row);
  odsState.addOperands(arguments);
}

void KernelCall::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments) {
  odsState.addOperands(col);
  odsState.addOperands(row);
  odsState.addOperands(arguments);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void KernelCall::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult KernelCall::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult KernelCall::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult KernelCall::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand colRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> colOperands(&colRawOperand, 1);  ::llvm::SMLoc colOperandsLoc;
  (void)colOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rowRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rowOperands(&rowRawOperand, 1);  ::llvm::SMLoc rowOperandsLoc;
  (void)rowOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argumentsOperands;
  ::llvm::SMLoc argumentsOperandsLoc;
  (void)argumentsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argumentsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  colOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(colRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rowOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rowRawOperand))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  argumentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argumentsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType arguments__allResult_functionType;
  if (parser.parseType(arguments__allResult_functionType))
    return ::mlir::failure();
  argumentsTypes = arguments__allResult_functionType.getInputs();
  allResultTypes = arguments__allResult_functionType.getResults();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(colOperands, odsBuildableType0, colOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rowOperands, odsBuildableType0, rowOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void KernelCall::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCol();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRow();
  _odsPrinter << "(";
  _odsPrinter << getArguments();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getArguments().getTypes(), getOperation()->getResultTypes());
}

void KernelCall::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::KernelCall)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::KernelDefinition definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> KernelDefinitionGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

} // namespace detail
KernelDefinitionAdaptor::KernelDefinitionAdaptor(KernelDefinition op) : KernelDefinitionGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult KernelDefinitionAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> KernelDefinition::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange KernelDefinition::getArgumentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

void KernelDefinition::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, ArrayRef<Value> arguments) {
      ValueRange args = ValueRange(arguments);
      build(odsBuilder, odsState, result_type, args);
    
}

void KernelDefinition::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
}

void KernelDefinition::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult KernelDefinition::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult KernelDefinition::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult KernelDefinition::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argumentsOperands;
  ::llvm::SMLoc argumentsOperandsLoc;
  (void)argumentsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argumentsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  if (parser.parseLParen())
    return ::mlir::failure();

  argumentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argumentsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType arguments__allResult_functionType;
  if (parser.parseType(arguments__allResult_functionType))
    return ::mlir::failure();
  argumentsTypes = arguments__allResult_functionType.getInputs();
  allResultTypes = arguments__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void KernelDefinition::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getArguments();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getArguments().getTypes(), getOperation()->getResultTypes());
}

void KernelDefinition::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::KernelDefinition)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::LoadKernel definitions
//===----------------------------------------------------------------------===//

namespace detail {
std::pair<unsigned, unsigned> LoadKernelGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::StringAttr LoadKernelGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, LoadKernel::getNameAttrName(*odsOpName)));
  return attr;
}

::llvm::StringRef LoadKernelGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr LoadKernelGenericAdaptorBase::getColAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, LoadKernel::getColAttrName(*odsOpName)));
  return attr;
}

uint32_t LoadKernelGenericAdaptorBase::getCol() {
  auto attr = getColAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr LoadKernelGenericAdaptorBase::getRowAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, LoadKernel::getRowAttrName(*odsOpName)));
  return attr;
}

uint32_t LoadKernelGenericAdaptorBase::getRow() {
  auto attr = getRowAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
LoadKernelAdaptor::LoadKernelAdaptor(LoadKernel op) : LoadKernelGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult LoadKernelAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_col;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.load_kernel_op' op ""requires attribute 'col'");
    if (namedAttrIt->getName() == LoadKernel::getColAttrName(*odsOpName)) {
      tblgen_col = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_name;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.load_kernel_op' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == LoadKernel::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_row;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.load_kernel_op' op ""requires attribute 'row'");
    if (namedAttrIt->getName() == LoadKernel::getRowAttrName(*odsOpName)) {
      tblgen_row = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'Aie.load_kernel_op' op ""attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_col && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_col))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_col).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.load_kernel_op' op ""attribute 'col' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_row && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_row))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_row).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.load_kernel_op' op ""attribute 'row' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoadKernel::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::MutableOperandRange LoadKernel::getArgumentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::llvm::StringRef LoadKernel::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

uint32_t LoadKernel::getCol() {
  auto attr = getColAttr();
  return attr.getValue().getZExtValue();
}

uint32_t LoadKernel::getRow() {
  auto attr = getRowAttr();
  return attr.getValue().getZExtValue();
}

void LoadKernel::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void LoadKernel::setCol(uint32_t attrValue) {
  (*this)->setAttr(getColAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void LoadKernel::setRow(uint32_t attrValue) {
  (*this)->setAttr(getRowAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void LoadKernel::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getColAttrName(odsState.name), col);
  odsState.addAttribute(getRowAttrName(odsState.name), row);
}

void LoadKernel::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getColAttrName(odsState.name), col);
  odsState.addAttribute(getRowAttrName(odsState.name), row);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadKernel::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getColAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), col));
  odsState.addAttribute(getRowAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), row));
}

void LoadKernel::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getColAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), col));
  odsState.addAttribute(getRowAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), row));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadKernel::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoadKernel::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_col;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'col'");
    if (namedAttrIt->getName() == getColAttrName()) {
      tblgen_col = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_name;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'name'");
    if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_row;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'row'");
    if (namedAttrIt->getName() == getRowAttrName()) {
      tblgen_row = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops0(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_col, "col")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_row, "row")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoadKernel::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoadKernel::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::IntegerAttr colAttr;
  ::mlir::IntegerAttr rowAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argumentsOperands;
  ::llvm::SMLoc argumentsOperandsLoc;
  (void)argumentsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argumentsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.attributes.append("name", nameAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(colAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (colAttr) result.attributes.append("col", colAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(rowAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (rowAttr) result.attributes.append("row", rowAttr);
  if (parser.parseLParen())
    return ::mlir::failure();

  argumentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argumentsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType arguments__allResult_functionType;
  if (parser.parseType(arguments__allResult_functionType))
    return ::mlir::failure();
  argumentsTypes = arguments__allResult_functionType.getInputs();
  allResultTypes = arguments__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadKernel::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getColAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getRowAttr());
  _odsPrinter << "(";
  _odsPrinter << getArguments();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  elidedAttrs.push_back("col");
  elidedAttrs.push_back("row");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getArguments().getTypes(), getOperation()->getResultTypes());
}

void LoadKernel::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::LoadKernel)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::LoadKernelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
LoadKernelOpAdaptor::LoadKernelOpAdaptor(LoadKernelOp op) : LoadKernelOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult LoadKernelOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoadKernelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
}

void LoadKernelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadKernelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoadKernelOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoadKernelOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoadKernelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(&lhsRawOperand, 1);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(&rhsRawOperand, 1);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(lhsOperands, odsBuildableType0, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, odsBuildableType0, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadKernelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ",";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::LoadKernelOp)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::MoveData definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
MoveDataAdaptor::MoveDataAdaptor(MoveData op) : MoveDataGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult MoveDataAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void MoveData::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(source);
  odsState.addOperands(sourceObject);
  odsState.addOperands(dataSize);
  odsState.addOperands(destinationObject);
  odsState.addOperands(destination);
}

void MoveData::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(source);
  odsState.addOperands(sourceObject);
  odsState.addOperands(dataSize);
  odsState.addOperands(destinationObject);
  odsState.addOperands(destination);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MoveData::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MoveData::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MoveData::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MoveData::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::MoveData)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::ResetHostEdgeConstraints definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ResetHostEdgeConstraintsAdaptor::ResetHostEdgeConstraintsAdaptor(ResetHostEdgeConstraints op) : ResetHostEdgeConstraintsGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ResetHostEdgeConstraintsAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ResetHostEdgeConstraints::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
}

void ResetHostEdgeConstraints::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResetHostEdgeConstraints::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResetHostEdgeConstraints::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ResetHostEdgeConstraints::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ResetHostEdgeConstraints::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ResetHostEdgeConstraints)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::Route definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
RouteAdaptor::RouteAdaptor(Route op) : RouteGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult RouteAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Route::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(routeConstraints);
  odsState.addOperands(source);
  odsState.addOperands(destination);
}

void Route::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(routeConstraints);
  odsState.addOperands(source);
  odsState.addOperands(destination);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Route::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Route::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Route::verifyInvariants() {
  return verifyInvariantsImpl();
}

void Route::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::Route)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::RouteDmaWait definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
RouteDmaWaitAdaptor::RouteDmaWaitAdaptor(RouteDmaWait op) : RouteDmaWaitGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult RouteDmaWaitAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void RouteDmaWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  odsState.addOperands(isS2MM);
}

void RouteDmaWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  odsState.addOperands(isS2MM);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RouteDmaWait::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RouteDmaWait::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RouteDmaWait::verifyInvariants() {
  return verifyInvariantsImpl();
}

void RouteDmaWait::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::RouteDmaWait)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::RoutesReveal definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
RoutesRevealAdaptor::RoutesRevealAdaptor(RoutesReveal op) : RoutesRevealGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult RoutesRevealAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void RoutesReveal::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(source);
  odsState.addOperands(destination);
}

void RoutesReveal::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(source);
  odsState.addOperands(destination);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RoutesReveal::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RoutesReveal::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RoutesReveal::verifyInvariants() {
  return verifyInvariantsImpl();
}

void RoutesReveal::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::RoutesReveal)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::RoutingInstanceFree definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
RoutingInstanceFreeAdaptor::RoutingInstanceFreeAdaptor(RoutingInstanceFree op) : RoutingInstanceFreeGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult RoutingInstanceFreeAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void RoutingInstanceFree::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance) {
  odsState.addOperands(routingInstance);
}

void RoutingInstanceFree::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance) {
  odsState.addOperands(routingInstance);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RoutingInstanceFree::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RoutingInstanceFree::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RoutingInstanceFree::verifyInvariants() {
  return verifyInvariantsImpl();
}

void RoutingInstanceFree::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::RoutingInstanceFree)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::Run definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
RunAdaptor::RunAdaptor(Run op) : RunGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult RunAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void Run::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
}

void Run::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Run::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Run::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Run::verifyInvariants() {
  return verifyInvariantsImpl();
}

void Run::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::Run)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::SetCoreExecute definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
SetCoreExecuteAdaptor::SetCoreExecuteAdaptor(SetCoreExecute op) : SetCoreExecuteGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult SetCoreExecuteAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void SetCoreExecute::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  odsState.addOperands(isExecute);
}

void SetCoreExecute::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  odsState.addOperands(isExecute);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SetCoreExecute::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SetCoreExecute::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SetCoreExecute::verifyInvariants() {
  return verifyInvariantsImpl();
}

void SetCoreExecute::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::SetCoreExecute)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantInteger32Op definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::IntegerAttr ConstantInteger32OpGenericAdaptorBase::getInputAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantInteger32Op::getInputAttrAttrName(*odsOpName)));
  return attr;
}

uint32_t ConstantInteger32OpGenericAdaptorBase::getInputAttr() {
  auto attr = getInputAttrAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ConstantInteger32OpAdaptor::ConstantInteger32OpAdaptor(ConstantInteger32Op op) : ConstantInteger32OpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ConstantInteger32OpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inputAttr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.constant_integer32_op' op ""requires attribute 'inputAttr'");
    if (namedAttrIt->getName() == ConstantInteger32Op::getInputAttrAttrName(*odsOpName)) {
      tblgen_inputAttr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_inputAttr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_inputAttr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_inputAttr).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.constant_integer32_op' op ""attribute 'inputAttr' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

uint32_t ConstantInteger32Op::getInputAttr() {
  auto attr = getInputAttrAttr();
  return attr.getValue().getZExtValue();
}

void ConstantInteger32Op::setInputAttr(uint32_t attrValue) {
  (*this)->setAttr(getInputAttrAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, IntegerAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr) {
  odsState.addAttribute(getInputAttrAttrName(odsState.name), inputAttr);
  odsState.addTypes(outputValue);
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr) {
  odsState.addAttribute(getInputAttrAttrName(odsState.name), inputAttr);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, uint32_t inputAttr) {
  odsState.addAttribute(getInputAttrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), inputAttr));
  odsState.addTypes(outputValue);
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t inputAttr) {
  odsState.addAttribute(getInputAttrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), inputAttr));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantInteger32Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantInteger32Op::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inputAttr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'inputAttr'");
    if (namedAttrIt->getName() == getInputAttrAttrName()) {
      tblgen_inputAttr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_inputAttr, "inputAttr")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantInteger32Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConstantInteger32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr inputAttrAttr;
  ::mlir::Type outputValueRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputValueTypes(&outputValueRawType, 1);

  if (parser.parseCustomAttributeWithFallback(inputAttrAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (inputAttrAttr) result.attributes.append("inputAttr", inputAttrAttr);
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputValueRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addTypes(outputValueTypes);
  return ::mlir::success();
}

void ConstantInteger32Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getInputAttrAttr());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOutputValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("inputAttr");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ConstantInteger32Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantInteger32Op)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantInteger64Op definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::IntegerAttr ConstantInteger64OpGenericAdaptorBase::getInputAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantInteger64Op::getInputAttrAttrName(*odsOpName)));
  return attr;
}

uint64_t ConstantInteger64OpGenericAdaptorBase::getInputAttr() {
  auto attr = getInputAttrAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ConstantInteger64OpAdaptor::ConstantInteger64OpAdaptor(ConstantInteger64Op op) : ConstantInteger64OpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ConstantInteger64OpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inputAttr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.constant_integer64_op' op ""requires attribute 'inputAttr'");
    if (namedAttrIt->getName() == ConstantInteger64Op::getInputAttrAttrName(*odsOpName)) {
      tblgen_inputAttr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_inputAttr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_inputAttr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_inputAttr).getType().isSignlessInteger(64)))))
    return emitError(loc, "'Aie.constant_integer64_op' op ""attribute 'inputAttr' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

uint64_t ConstantInteger64Op::getInputAttr() {
  auto attr = getInputAttrAttr();
  return attr.getValue().getZExtValue();
}

void ConstantInteger64Op::setInputAttr(uint64_t attrValue) {
  (*this)->setAttr(getInputAttrAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, IntegerAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr) {
  odsState.addAttribute(getInputAttrAttrName(odsState.name), inputAttr);
  odsState.addTypes(outputValue);
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr) {
  odsState.addAttribute(getInputAttrAttrName(odsState.name), inputAttr);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, uint64_t inputAttr) {
  odsState.addAttribute(getInputAttrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), inputAttr));
  odsState.addTypes(outputValue);
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t inputAttr) {
  odsState.addAttribute(getInputAttrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), inputAttr));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantInteger64Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantInteger64Op::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inputAttr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'inputAttr'");
    if (namedAttrIt->getName() == getInputAttrAttrName()) {
      tblgen_inputAttr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_inputAttr, "inputAttr")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantInteger64Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConstantInteger64Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr inputAttrAttr;
  ::mlir::Type outputValueRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputValueTypes(&outputValueRawType, 1);

  if (parser.parseCustomAttributeWithFallback(inputAttrAttr, parser.getBuilder().getIntegerType(64))) {
    return ::mlir::failure();
  }
  if (inputAttrAttr) result.attributes.append("inputAttr", inputAttrAttr);
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputValueRawType = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addTypes(outputValueTypes);
  return ::mlir::success();
}

void ConstantInteger64Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getInputAttrAttr());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOutputValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("inputAttr");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ConstantInteger64Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantInteger64Op)

namespace mlir {
namespace aie {

//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::DenseElementsAttr ConstantOpGenericAdaptorBase::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::DenseElementsAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantOp::getValueAttrName(*odsOpName)));
  return attr;
}

::mlir::DenseElementsAttr ConstantOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : ConstantOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'Aie.constant' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == ConstantOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((::llvm::isa<::mlir::DenseFPElementsAttr>(tblgen_value) &&::llvm::cast<::mlir::DenseElementsAttr>(tblgen_value).getType().getElementType().isF64())))
    return emitError(loc, "'Aie.constant' op ""attribute 'value' failed to satisfy constraint: 64-bit float elements attribute");
  return ::mlir::success();
}

::mlir::DenseElementsAttr ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(resultType0);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops3(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops16(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ConstantOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace aie
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantOp)


#endif  // GET_OP_CLASSES

