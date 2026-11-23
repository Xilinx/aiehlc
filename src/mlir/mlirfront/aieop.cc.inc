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
#endif // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(64)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 64-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::MemRefType>(type))) && ([](::mlir::Type elementType) { return !((::llvm::isa<::mlir::TokenType>(elementType))); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any non-token type values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::RoutingInstanceType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::HostAieConstraintType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::LocType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::KernelFileNameType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be the kernel path, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::KernelFuncNameType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be the kernel function name, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(!((::llvm::isa<::mlir::TokenType>(type))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be variadic of any non-token type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::KernelType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::TileKernelType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::WindowType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Window type with direction, size, and name, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::DeviceInstanceType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(!((::llvm::isa<::mlir::TokenType>(type))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any non-token type, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops16(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aie::RoutingConstraintType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aieops17(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 64-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aieops1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aieops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aieops1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aieops2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aieops2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aieops2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aieops3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(64)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aieops3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aieops3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aieops4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::DenseFPElementsAttr>(attr) &&::llvm::cast<::mlir::DenseElementsAttr>(attr).getType().getElementType().isF64())))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit float elements attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aieops4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aieops4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::AllocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

AllocOpGenericAdaptorBase::AllocOpGenericAdaptorBase(AllocOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
AllocOpAdaptor::AllocOpAdaptor(AllocOp op) : AllocOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AllocOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult AllocOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute AllocOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AllocOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> AllocOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void AllocOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void AllocOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult AllocOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value size) {
  odsState.addOperands(size);
  odsState.addTypes(result);
}

AllocOp AllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value size) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(size)>(size));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocOp>(__res__);
}

AllocOp AllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value size) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(size)>(size));
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value size) {
  odsState.addOperands(size);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

AllocOp AllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value size) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(size)>(size));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocOp>(__res__);
}

AllocOp AllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value size) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(size)>(size));
}

void AllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

AllocOp AllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocOp>(__res__);
}

AllocOp AllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void AllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

AllocOp AllocOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocOp>(__res__);
}

AllocOp AllocOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult AllocOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops1(*this, v.getType(), "operand", index++)))
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

::llvm::LogicalResult AllocOp::verifyInvariants() {
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
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
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(sizeOperands, odsBuildableType0, sizeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void AllocOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getSize();
  _odsPrinter << ")";
  ::llvm::ArrayRef<::llvm::StringRef> elidedAttrs;
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
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


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::AllocOp)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ConfigHostEdgeConstraints definitions
//===----------------------------------------------------------------------===//

namespace detail {

ConfigHostEdgeConstraintsGenericAdaptorBase::ConfigHostEdgeConstraintsGenericAdaptorBase(ConfigHostEdgeConstraints op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
ConfigHostEdgeConstraintsAdaptor::ConfigHostEdgeConstraintsAdaptor(ConfigHostEdgeConstraints op) : ConfigHostEdgeConstraintsGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConfigHostEdgeConstraintsAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult ConfigHostEdgeConstraints::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute ConfigHostEdgeConstraints::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConfigHostEdgeConstraints::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> ConfigHostEdgeConstraints::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void ConfigHostEdgeConstraints::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void ConfigHostEdgeConstraints::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult ConfigHostEdgeConstraints::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void ConfigHostEdgeConstraints::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value userResCon) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, userResCon);
}

ConfigHostEdgeConstraints ConfigHostEdgeConstraints::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value userResCon) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(userResCon)>(userResCon));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConfigHostEdgeConstraints>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConfigHostEdgeConstraints>(__res__);
}

ConfigHostEdgeConstraints ConfigHostEdgeConstraints::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value userResCon) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(userResCon)>(userResCon));
}

void ConfigHostEdgeConstraints::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value userResCon) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(userResCon);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ConfigHostEdgeConstraints ConfigHostEdgeConstraints::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value userResCon) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(userResCon)>(userResCon));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConfigHostEdgeConstraints>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConfigHostEdgeConstraints>(__res__);
}

ConfigHostEdgeConstraints ConfigHostEdgeConstraints::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value userResCon) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(userResCon)>(userResCon));
}

void ConfigHostEdgeConstraints::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConfigHostEdgeConstraints ConfigHostEdgeConstraints::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConfigHostEdgeConstraints>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConfigHostEdgeConstraints>(__res__);
}

ConfigHostEdgeConstraints ConfigHostEdgeConstraints::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ConfigHostEdgeConstraints::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConfigHostEdgeConstraints ConfigHostEdgeConstraints::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConfigHostEdgeConstraints>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConfigHostEdgeConstraints>(__res__);
}

ConfigHostEdgeConstraints ConfigHostEdgeConstraints::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ConfigHostEdgeConstraints::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
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

::llvm::LogicalResult ConfigHostEdgeConstraints::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult ConfigHostEdgeConstraints::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void ConfigHostEdgeConstraints::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ConfigHostEdgeConstraints)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CoreWait definitions
//===----------------------------------------------------------------------===//

namespace detail {

CoreWaitGenericAdaptorBase::CoreWaitGenericAdaptorBase(CoreWait op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
CoreWaitAdaptor::CoreWaitAdaptor(CoreWait op) : CoreWaitGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CoreWaitAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult CoreWait::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute CoreWait::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CoreWait::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> CoreWait::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void CoreWait::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void CoreWait::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult CoreWait::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void CoreWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, tile);
}

CoreWait CoreWait::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CoreWait>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CoreWait>(__res__);
}

CoreWait CoreWait::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
}

void CoreWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CoreWait CoreWait::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CoreWait>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CoreWait>(__res__);
}

CoreWait CoreWait::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
}

void CoreWait::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CoreWait CoreWait::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CoreWait>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CoreWait>(__res__);
}

CoreWait CoreWait::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void CoreWait::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CoreWait CoreWait::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CoreWait>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CoreWait>(__res__);
}

CoreWait CoreWait::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult CoreWait::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CoreWait::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult CoreWait::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void CoreWait::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CoreWait)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelFileNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

CreateKernelFileNameOpGenericAdaptorBase::CreateKernelFileNameOpGenericAdaptorBase(CreateKernelFileNameOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef CreateKernelFileNameOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}


} // namespace detail
CreateKernelFileNameOpAdaptor::CreateKernelFileNameOpAdaptor(CreateKernelFileNameOp op) : CreateKernelFileNameOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateKernelFileNameOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'Aie.create_kernelfilename_op' op requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'Aie.create_kernelfilename_op' op attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CreateKernelFileNameOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("name");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.name, attr, "name", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute CreateKernelFileNameOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "name", prop.name);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CreateKernelFileNameOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateKernelFileNameOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void CreateKernelFileNameOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void CreateKernelFileNameOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.name) {
      ::mlir::Attribute value = prop.name;
      ::mlir::Attribute originalValue = value;
      visitor("name", value);
      if (value != originalValue)
        setInherentAttr(prop, "name", value);
    }
}

::llvm::LogicalResult CreateKernelFileNameOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(attr, "name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateKernelFileNameOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateKernelFileNameOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.name);
}

::llvm::StringRef CreateKernelFileNameOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CreateKernelFileNameOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.name = name;
  odsState.addTypes(result);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::StringAttr name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFileNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFileNameOp>(__res__);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::StringAttr name) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(name)>(name));
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.name = name;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFileNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFileNameOp>(__res__);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name));
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.name = odsBuilder.getStringAttr(name);
  odsState.addTypes(result);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::llvm::StringRef name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFileNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFileNameOp>(__res__);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::llvm::StringRef name) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(name)>(name));
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.name = odsBuilder.getStringAttr(name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFileNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFileNameOp>(__res__);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name));
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFileNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFileNameOp>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void CreateKernelFileNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFileNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFileNameOp>(__res__);
}

CreateKernelFileNameOp CreateKernelFileNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult CreateKernelFileNameOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_name, "name")))
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

::llvm::LogicalResult CreateKernelFileNameOp::verifyInvariants() {
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
  if (nameAttr) result.getOrAddProperties<CreateKernelFileNameOp::Properties>().name = nameAttr;
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void CreateKernelFileNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "{";
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << "}";
  _odsPrinter << " (";
  _odsPrinter << ")";
  static constexpr ::llvm::StringRef elidedAttrs[] = {"name"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateKernelFileNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void CreateKernelFileNameOp::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"name"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelFileNameOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateKernelFileNameOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelFuncNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

CreateKernelFuncNameOpGenericAdaptorBase::CreateKernelFuncNameOpGenericAdaptorBase(CreateKernelFuncNameOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef CreateKernelFuncNameOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}


} // namespace detail
CreateKernelFuncNameOpAdaptor::CreateKernelFuncNameOpAdaptor(CreateKernelFuncNameOp op) : CreateKernelFuncNameOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateKernelFuncNameOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'Aie.create_kernelfuncname_op' op requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'Aie.create_kernelfuncname_op' op attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CreateKernelFuncNameOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("name");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.name, attr, "name", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute CreateKernelFuncNameOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "name", prop.name);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CreateKernelFuncNameOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateKernelFuncNameOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void CreateKernelFuncNameOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void CreateKernelFuncNameOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.name) {
      ::mlir::Attribute value = prop.name;
      ::mlir::Attribute originalValue = value;
      visitor("name", value);
      if (value != originalValue)
        setInherentAttr(prop, "name", value);
    }
}

::llvm::LogicalResult CreateKernelFuncNameOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(attr, "name", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateKernelFuncNameOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateKernelFuncNameOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.name);
}

::llvm::StringRef CreateKernelFuncNameOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CreateKernelFuncNameOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.name = name;
  odsState.addTypes(result);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::StringAttr name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFuncNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFuncNameOp>(__res__);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::StringAttr name) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(name)>(name));
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.name = name;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFuncNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFuncNameOp>(__res__);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name));
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.name = odsBuilder.getStringAttr(name);
  odsState.addTypes(result);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::llvm::StringRef name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFuncNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFuncNameOp>(__res__);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::llvm::StringRef name) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(name)>(name));
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.name = odsBuilder.getStringAttr(name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFuncNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFuncNameOp>(__res__);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name));
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFuncNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFuncNameOp>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void CreateKernelFuncNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelFuncNameOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelFuncNameOp>(__res__);
}

CreateKernelFuncNameOp CreateKernelFuncNameOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult CreateKernelFuncNameOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CreateKernelFuncNameOp::verifyInvariants() {
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
  if (nameAttr) result.getOrAddProperties<CreateKernelFuncNameOp::Properties>().name = nameAttr;
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void CreateKernelFuncNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "{";
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << "}";
  _odsPrinter << " (";
  _odsPrinter << ")";
  static constexpr ::llvm::StringRef elidedAttrs[] = {"name"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateKernelFuncNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void CreateKernelFuncNameOp::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"name"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelFuncNameOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateKernelFuncNameOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelObjectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

CreateKernelObjectOpGenericAdaptorBase::CreateKernelObjectOpGenericAdaptorBase(CreateKernelObjectOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

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

uint32_t CreateKernelObjectOpGenericAdaptorBase::getNumInputArgs() {
  auto attr = getNumInputArgsAttr();
  return attr.getValue().getZExtValue();
}

uint32_t CreateKernelObjectOpGenericAdaptorBase::getNumOutputArgs() {
  auto attr = getNumOutputArgsAttr();
  return attr.getValue().getZExtValue();
}

::llvm::StringRef CreateKernelObjectOpGenericAdaptorBase::getInParamType() {
  auto attr = getInParamTypeAttr();
  return attr.getValue();
}

::llvm::StringRef CreateKernelObjectOpGenericAdaptorBase::getOutParamType() {
  auto attr = getOutParamTypeAttr();
  return attr.getValue();
}


} // namespace detail
CreateKernelObjectOpAdaptor::CreateKernelObjectOpAdaptor(CreateKernelObjectOp op) : CreateKernelObjectOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateKernelObjectOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_in_param_type = getProperties().in_param_type; (void)tblgen_in_param_type;
  if (!tblgen_in_param_type) return emitError(loc, "'Aie.create_kernelobject_op' op requires attribute 'in_param_type'");
  auto tblgen_numInputArgs = getProperties().numInputArgs; (void)tblgen_numInputArgs;
  if (!tblgen_numInputArgs) return emitError(loc, "'Aie.create_kernelobject_op' op requires attribute 'numInputArgs'");
  auto tblgen_numOutputArgs = getProperties().numOutputArgs; (void)tblgen_numOutputArgs;
  if (!tblgen_numOutputArgs) return emitError(loc, "'Aie.create_kernelobject_op' op requires attribute 'numOutputArgs'");
  auto tblgen_out_param_type = getProperties().out_param_type; (void)tblgen_out_param_type;
  if (!tblgen_out_param_type) return emitError(loc, "'Aie.create_kernelobject_op' op requires attribute 'out_param_type'");

  if (tblgen_numInputArgs && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_numInputArgs))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_numInputArgs).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_kernelobject_op' op attribute 'numInputArgs' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_numOutputArgs && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_numOutputArgs))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_numOutputArgs).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_kernelobject_op' op attribute 'numOutputArgs' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_in_param_type && !((::llvm::isa<::mlir::StringAttr>(tblgen_in_param_type))))
    return emitError(loc, "'Aie.create_kernelobject_op' op attribute 'in_param_type' failed to satisfy constraint: string attribute");

  if (tblgen_out_param_type && !((::llvm::isa<::mlir::StringAttr>(tblgen_out_param_type))))
    return emitError(loc, "'Aie.create_kernelobject_op' op attribute 'out_param_type' failed to satisfy constraint: string attribute");
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

::llvm::LogicalResult CreateKernelObjectOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("in_param_type");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.in_param_type, attr, "in_param_type", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("numInputArgs");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.numInputArgs, attr, "numInputArgs", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("numOutputArgs");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.numOutputArgs, attr, "numOutputArgs", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("out_param_type");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.out_param_type, attr, "out_param_type", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute CreateKernelObjectOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "in_param_type", prop.in_param_type);
    ::mlir::detail::appendAttributeProperty(attrs, "numInputArgs", prop.numInputArgs);
    ::mlir::detail::appendAttributeProperty(attrs, "numOutputArgs", prop.numOutputArgs);
    ::mlir::detail::appendAttributeProperty(attrs, "out_param_type", prop.out_param_type);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CreateKernelObjectOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.in_param_type.getAsOpaquePointer()), 
    llvm::hash_value(prop.numInputArgs.getAsOpaquePointer()), 
    llvm::hash_value(prop.numOutputArgs.getAsOpaquePointer()), 
    llvm::hash_value(prop.out_param_type.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateKernelObjectOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "in_param_type")
      return prop.in_param_type;

    if (name == "numInputArgs")
      return prop.numInputArgs;

    if (name == "numOutputArgs")
      return prop.numOutputArgs;

    if (name == "out_param_type")
      return prop.out_param_type;
  return std::nullopt;
}

void CreateKernelObjectOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "in_param_type") {
       prop.in_param_type = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.in_param_type)>>(value);
       return;
    }

    if (name == "numInputArgs") {
       prop.numInputArgs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.numInputArgs)>>(value);
       return;
    }

    if (name == "numOutputArgs") {
       prop.numOutputArgs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.numOutputArgs)>>(value);
       return;
    }

    if (name == "out_param_type") {
       prop.out_param_type = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.out_param_type)>>(value);
       return;
    }
}

void CreateKernelObjectOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.in_param_type) {
      ::mlir::Attribute value = prop.in_param_type;
      ::mlir::Attribute originalValue = value;
      visitor("in_param_type", value);
      if (value != originalValue)
        setInherentAttr(prop, "in_param_type", value);
    }

    if (prop.numInputArgs) {
      ::mlir::Attribute value = prop.numInputArgs;
      ::mlir::Attribute originalValue = value;
      visitor("numInputArgs", value);
      if (value != originalValue)
        setInherentAttr(prop, "numInputArgs", value);
    }

    if (prop.numOutputArgs) {
      ::mlir::Attribute value = prop.numOutputArgs;
      ::mlir::Attribute originalValue = value;
      visitor("numOutputArgs", value);
      if (value != originalValue)
        setInherentAttr(prop, "numOutputArgs", value);
    }

    if (prop.out_param_type) {
      ::mlir::Attribute value = prop.out_param_type;
      ::mlir::Attribute originalValue = value;
      visitor("out_param_type", value);
      if (value != originalValue)
        setInherentAttr(prop, "out_param_type", value);
    }
}

::llvm::LogicalResult CreateKernelObjectOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getInParamTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(attr, "in_param_type", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNumInputArgsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "numInputArgs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNumOutputArgsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "numOutputArgs", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getOutParamTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(attr, "out_param_type", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateKernelObjectOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.in_param_type)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.numInputArgs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.numOutputArgs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.out_param_type)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateKernelObjectOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.in_param_type);
  writer.writeAttribute(prop.numInputArgs);
  writer.writeAttribute(prop.numOutputArgs);
  writer.writeAttribute(prop.out_param_type);
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
  getProperties().numInputArgs = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void CreateKernelObjectOp::setNumOutputArgs(uint32_t attrValue) {
  getProperties().numOutputArgs = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void CreateKernelObjectOp::setInParamType(::llvm::StringRef attrValue) {
  getProperties().in_param_type = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void CreateKernelObjectOp::setOutParamType(::llvm::StringRef attrValue) {
  getProperties().out_param_type = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, IntegerAttr numInputArgs, IntegerAttr numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, std::string in_param_type, std::string out_param_type, ValueRange arguments) {
      ValueRange args = ValueRange(arguments);
      build(odsBuilder, odsState, result_type, numInputArgs, numOutputArgs,kname, fname, in_param_type, out_param_type, arguments);// args);
    
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, Type result_type, IntegerAttr numInputArgs, IntegerAttr numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, std::string in_param_type, std::string out_param_type, ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result_type)>(result_type), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelObjectOp>(__res__);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, Type result_type, IntegerAttr numInputArgs, IntegerAttr numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, std::string in_param_type, std::string out_param_type, ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(result_type)>(result_type), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(kname);
  odsState.addOperands(fname);
  odsState.addOperands(arguments);
  odsProperties.numInputArgs = numInputArgs;
  odsProperties.numOutputArgs = numOutputArgs;
  odsProperties.in_param_type = in_param_type;
  odsProperties.out_param_type = out_param_type;
  odsState.addTypes(result);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelObjectOp>(__res__);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(kname);
  odsState.addOperands(fname);
  odsState.addOperands(arguments);
  odsProperties.numInputArgs = numInputArgs;
  odsProperties.numOutputArgs = numOutputArgs;
  odsProperties.in_param_type = in_param_type;
  odsProperties.out_param_type = out_param_type;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelObjectOp>(__res__);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(kname);
  odsState.addOperands(fname);
  odsState.addOperands(arguments);
  odsProperties.numInputArgs = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numInputArgs);
  odsProperties.numOutputArgs = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numOutputArgs);
  odsProperties.in_param_type = odsBuilder.getStringAttr(in_param_type);
  odsProperties.out_param_type = odsBuilder.getStringAttr(out_param_type);
  odsState.addTypes(result);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelObjectOp>(__res__);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(kname);
  odsState.addOperands(fname);
  odsState.addOperands(arguments);
  odsProperties.numInputArgs = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numInputArgs);
  odsProperties.numOutputArgs = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numOutputArgs);
  odsProperties.in_param_type = odsBuilder.getStringAttr(in_param_type);
  odsProperties.out_param_type = odsBuilder.getStringAttr(out_param_type);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelObjectOp>(__res__);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(numInputArgs)>(numInputArgs), std::forward<decltype(numOutputArgs)>(numOutputArgs), std::forward<decltype(kname)>(kname), std::forward<decltype(fname)>(fname), std::forward<decltype(in_param_type)>(in_param_type), std::forward<decltype(out_param_type)>(out_param_type), std::forward<decltype(arguments)>(arguments));
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelObjectOp>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void CreateKernelObjectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateKernelObjectOp>(__res__);
}

CreateKernelObjectOp CreateKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult CreateKernelObjectOp::verifyInvariantsImpl() {
  auto tblgen_in_param_type = getProperties().in_param_type; (void)tblgen_in_param_type;
  if (!tblgen_in_param_type) return emitOpError("requires attribute 'in_param_type'");
  auto tblgen_numInputArgs = getProperties().numInputArgs; (void)tblgen_numInputArgs;
  if (!tblgen_numInputArgs) return emitOpError("requires attribute 'numInputArgs'");
  auto tblgen_numOutputArgs = getProperties().numOutputArgs; (void)tblgen_numOutputArgs;
  if (!tblgen_numOutputArgs) return emitOpError("requires attribute 'numOutputArgs'");
  auto tblgen_out_param_type = getProperties().out_param_type; (void)tblgen_out_param_type;
  if (!tblgen_out_param_type) return emitOpError("requires attribute 'out_param_type'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_numInputArgs, "numInputArgs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_numOutputArgs, "numOutputArgs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_in_param_type, "in_param_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_out_param_type, "out_param_type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
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

::llvm::LogicalResult CreateKernelObjectOp::verifyInvariants() {
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
  if (numInputArgsAttr) result.getOrAddProperties<CreateKernelObjectOp::Properties>().numInputArgs = numInputArgsAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(numOutputArgsAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (numOutputArgsAttr) result.getOrAddProperties<CreateKernelObjectOp::Properties>().numOutputArgs = numOutputArgsAttr;
  if (parser.parseRBrace())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
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
  if (in_param_typeAttr) result.getOrAddProperties<CreateKernelObjectOp::Properties>().in_param_type = in_param_typeAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(out_param_typeAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (out_param_typeAttr) result.getOrAddProperties<CreateKernelObjectOp::Properties>().out_param_type = out_param_typeAttr;
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
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(knameOperands, fnameOperands, argumentsOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void CreateKernelObjectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "{";
  _odsPrinter.printAttributeWithoutType(getNumInputArgsAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNumOutputArgsAttr());
  _odsPrinter << "}";
  static constexpr ::llvm::StringRef elidedAttrs[] = {"numInputArgs", "numOutputArgs", "in_param_type", "out_param_type"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
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
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void CreateKernelObjectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void CreateKernelObjectOp::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"numInputArgs", "numOutputArgs", "in_param_type", "out_param_type"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelObjectOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateKernelObjectOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateTileKernelObjectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

CreateTileKernelObjectOpGenericAdaptorBase::CreateTileKernelObjectOpGenericAdaptorBase(CreateTileKernelObjectOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::llvm::StringRef CreateTileKernelObjectOpGenericAdaptorBase::getKernels() {
  auto attr = getKernelsAttr();
  return attr.getValue();
}


} // namespace detail
CreateTileKernelObjectOpAdaptor::CreateTileKernelObjectOpAdaptor(CreateTileKernelObjectOp op) : CreateTileKernelObjectOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateTileKernelObjectOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_kernels = getProperties().kernels; (void)tblgen_kernels;
  if (!tblgen_kernels) return emitError(loc, "'Aie.create_tilekernelobject_op' op requires attribute 'kernels'");

  if (tblgen_kernels && !((::llvm::isa<::mlir::StringAttr>(tblgen_kernels))))
    return emitError(loc, "'Aie.create_tilekernelobject_op' op attribute 'kernels' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CreateTileKernelObjectOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("kernels");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.kernels, attr, "kernels", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute CreateTileKernelObjectOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "kernels", prop.kernels);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CreateTileKernelObjectOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.kernels.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateTileKernelObjectOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "kernels")
      return prop.kernels;
  return std::nullopt;
}

void CreateTileKernelObjectOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "kernels") {
       prop.kernels = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.kernels)>>(value);
       return;
    }
}

void CreateTileKernelObjectOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.kernels) {
      ::mlir::Attribute value = prop.kernels;
      ::mlir::Attribute originalValue = value;
      visitor("kernels", value);
      if (value != originalValue)
        setInherentAttr(prop, "kernels", value);
    }
}

::llvm::LogicalResult CreateTileKernelObjectOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getKernelsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(attr, "kernels", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateTileKernelObjectOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.kernels)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateTileKernelObjectOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.kernels);
}

::llvm::StringRef CreateTileKernelObjectOp::getKernels() {
  auto attr = getKernelsAttr();
  return attr.getValue();
}

void CreateTileKernelObjectOp::setKernels(::llvm::StringRef attrValue) {
  getProperties().kernels = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(row);
  odsState.addOperands(col);
  odsProperties.kernels = kernels;
  odsState.addTypes(result);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(row)>(row), std::forward<decltype(col)>(col), std::forward<decltype(kernels)>(kernels));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateTileKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateTileKernelObjectOp>(__res__);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(row)>(row), std::forward<decltype(col)>(col), std::forward<decltype(kernels)>(kernels));
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(row);
  odsState.addOperands(col);
  odsProperties.kernels = kernels;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(row)>(row), std::forward<decltype(col)>(col), std::forward<decltype(kernels)>(kernels));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateTileKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateTileKernelObjectOp>(__res__);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(row)>(row), std::forward<decltype(col)>(col), std::forward<decltype(kernels)>(kernels));
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(row);
  odsState.addOperands(col);
  odsProperties.kernels = odsBuilder.getStringAttr(kernels);
  odsState.addTypes(result);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(row)>(row), std::forward<decltype(col)>(col), std::forward<decltype(kernels)>(kernels));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateTileKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateTileKernelObjectOp>(__res__);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(row)>(row), std::forward<decltype(col)>(col), std::forward<decltype(kernels)>(kernels));
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(row);
  odsState.addOperands(col);
  odsProperties.kernels = odsBuilder.getStringAttr(kernels);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(row)>(row), std::forward<decltype(col)>(col), std::forward<decltype(kernels)>(kernels));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateTileKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateTileKernelObjectOp>(__res__);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(row)>(row), std::forward<decltype(col)>(col), std::forward<decltype(kernels)>(kernels));
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateTileKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateTileKernelObjectOp>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void CreateTileKernelObjectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateTileKernelObjectOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateTileKernelObjectOp>(__res__);
}

CreateTileKernelObjectOp CreateTileKernelObjectOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult CreateTileKernelObjectOp::verifyInvariantsImpl() {
  auto tblgen_kernels = getProperties().kernels; (void)tblgen_kernels;
  if (!tblgen_kernels) return emitOpError("requires attribute 'kernels'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_kernels, "kernels")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
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

::llvm::LogicalResult CreateTileKernelObjectOp::verifyInvariants() {
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
  if (kernelsAttr) result.getOrAddProperties<CreateTileKernelObjectOp::Properties>().kernels = kernelsAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(allResultTypes);
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(rowOperands, odsBuildableType0, rowOperandsLoc, result.operands))
    return ::mlir::failure();
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(colOperands, odsBuildableType0, colOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void CreateTileKernelObjectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getRow();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getCol();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getKernelsAttr());
  static constexpr ::llvm::StringRef elidedAttrs[] = {"kernels"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateTileKernelObjectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void CreateTileKernelObjectOp::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"kernels"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateTileKernelObjectOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateTileKernelObjectOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateWindowOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

CreateWindowOpGenericAdaptorBase::CreateWindowOpGenericAdaptorBase(CreateWindowOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t CreateWindowOpGenericAdaptorBase::getDirection() {
  auto attr = getDirectionAttr();
  return attr.getValue().getZExtValue();
}

uint32_t CreateWindowOpGenericAdaptorBase::getSize() {
  auto attr = getSizeAttr();
  return attr.getValue().getZExtValue();
}

uint64_t CreateWindowOpGenericAdaptorBase::getPingaddr() {
  auto attr = getPingaddrAttr();
  return attr.getValue().getZExtValue();
}

uint64_t CreateWindowOpGenericAdaptorBase::getPongaddr() {
  auto attr = getPongaddrAttr();
  return attr.getValue().getZExtValue();
}

uint32_t CreateWindowOpGenericAdaptorBase::getPinglockid() {
  auto attr = getPinglockidAttr();
  return attr.getValue().getZExtValue();
}

uint32_t CreateWindowOpGenericAdaptorBase::getPonglockid() {
  auto attr = getPonglockidAttr();
  return attr.getValue().getZExtValue();
}

::llvm::StringRef CreateWindowOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}


} // namespace detail
CreateWindowOpAdaptor::CreateWindowOpAdaptor(CreateWindowOp op) : CreateWindowOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult CreateWindowOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_direction = getProperties().direction; (void)tblgen_direction;
  if (!tblgen_direction) return emitError(loc, "'Aie.create_window' op requires attribute 'direction'");
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'Aie.create_window' op requires attribute 'name'");
  auto tblgen_pingaddr = getProperties().pingaddr; (void)tblgen_pingaddr;
  if (!tblgen_pingaddr) return emitError(loc, "'Aie.create_window' op requires attribute 'pingaddr'");
  auto tblgen_pinglockid = getProperties().pinglockid; (void)tblgen_pinglockid;
  if (!tblgen_pinglockid) return emitError(loc, "'Aie.create_window' op requires attribute 'pinglockid'");
  auto tblgen_pongaddr = getProperties().pongaddr; (void)tblgen_pongaddr;
  if (!tblgen_pongaddr) return emitError(loc, "'Aie.create_window' op requires attribute 'pongaddr'");
  auto tblgen_ponglockid = getProperties().ponglockid; (void)tblgen_ponglockid;
  if (!tblgen_ponglockid) return emitError(loc, "'Aie.create_window' op requires attribute 'ponglockid'");
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitError(loc, "'Aie.create_window' op requires attribute 'size'");

  if (tblgen_direction && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_direction))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_direction).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_window' op attribute 'direction' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_size && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_size))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_size).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_window' op attribute 'size' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_pingaddr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pingaddr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pingaddr).getType().isSignlessInteger(64)))))
    return emitError(loc, "'Aie.create_window' op attribute 'pingaddr' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_pongaddr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pongaddr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pongaddr).getType().isSignlessInteger(64)))))
    return emitError(loc, "'Aie.create_window' op attribute 'pongaddr' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_pinglockid && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_pinglockid))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_pinglockid).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_window' op attribute 'pinglockid' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_ponglockid && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_ponglockid))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_ponglockid).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.create_window' op attribute 'ponglockid' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'Aie.create_window' op attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

::llvm::LogicalResult CreateWindowOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("direction");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.direction, attr, "direction", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("name");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.name, attr, "name", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("pingaddr");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.pingaddr, attr, "pingaddr", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("pinglockid");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.pinglockid, attr, "pinglockid", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("pongaddr");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.pongaddr, attr, "pongaddr", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("ponglockid");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.ponglockid, attr, "ponglockid", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("size");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.size, attr, "size", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute CreateWindowOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "direction", prop.direction);
    ::mlir::detail::appendAttributeProperty(attrs, "name", prop.name);
    ::mlir::detail::appendAttributeProperty(attrs, "pingaddr", prop.pingaddr);
    ::mlir::detail::appendAttributeProperty(attrs, "pinglockid", prop.pinglockid);
    ::mlir::detail::appendAttributeProperty(attrs, "pongaddr", prop.pongaddr);
    ::mlir::detail::appendAttributeProperty(attrs, "ponglockid", prop.ponglockid);
    ::mlir::detail::appendAttributeProperty(attrs, "size", prop.size);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code CreateWindowOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.direction.getAsOpaquePointer()), 
    llvm::hash_value(prop.name.getAsOpaquePointer()), 
    llvm::hash_value(prop.pingaddr.getAsOpaquePointer()), 
    llvm::hash_value(prop.pinglockid.getAsOpaquePointer()), 
    llvm::hash_value(prop.pongaddr.getAsOpaquePointer()), 
    llvm::hash_value(prop.ponglockid.getAsOpaquePointer()), 
    llvm::hash_value(prop.size.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateWindowOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "direction")
      return prop.direction;

    if (name == "name")
      return prop.name;

    if (name == "pingaddr")
      return prop.pingaddr;

    if (name == "pinglockid")
      return prop.pinglockid;

    if (name == "pongaddr")
      return prop.pongaddr;

    if (name == "ponglockid")
      return prop.ponglockid;

    if (name == "size")
      return prop.size;
  return std::nullopt;
}

void CreateWindowOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "direction") {
       prop.direction = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.direction)>>(value);
       return;
    }

    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }

    if (name == "pingaddr") {
       prop.pingaddr = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pingaddr)>>(value);
       return;
    }

    if (name == "pinglockid") {
       prop.pinglockid = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pinglockid)>>(value);
       return;
    }

    if (name == "pongaddr") {
       prop.pongaddr = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.pongaddr)>>(value);
       return;
    }

    if (name == "ponglockid") {
       prop.ponglockid = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.ponglockid)>>(value);
       return;
    }

    if (name == "size") {
       prop.size = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.size)>>(value);
       return;
    }
}

void CreateWindowOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.direction) {
      ::mlir::Attribute value = prop.direction;
      ::mlir::Attribute originalValue = value;
      visitor("direction", value);
      if (value != originalValue)
        setInherentAttr(prop, "direction", value);
    }

    if (prop.name) {
      ::mlir::Attribute value = prop.name;
      ::mlir::Attribute originalValue = value;
      visitor("name", value);
      if (value != originalValue)
        setInherentAttr(prop, "name", value);
    }

    if (prop.pingaddr) {
      ::mlir::Attribute value = prop.pingaddr;
      ::mlir::Attribute originalValue = value;
      visitor("pingaddr", value);
      if (value != originalValue)
        setInherentAttr(prop, "pingaddr", value);
    }

    if (prop.pinglockid) {
      ::mlir::Attribute value = prop.pinglockid;
      ::mlir::Attribute originalValue = value;
      visitor("pinglockid", value);
      if (value != originalValue)
        setInherentAttr(prop, "pinglockid", value);
    }

    if (prop.pongaddr) {
      ::mlir::Attribute value = prop.pongaddr;
      ::mlir::Attribute originalValue = value;
      visitor("pongaddr", value);
      if (value != originalValue)
        setInherentAttr(prop, "pongaddr", value);
    }

    if (prop.ponglockid) {
      ::mlir::Attribute value = prop.ponglockid;
      ::mlir::Attribute originalValue = value;
      visitor("ponglockid", value);
      if (value != originalValue)
        setInherentAttr(prop, "ponglockid", value);
    }

    if (prop.size) {
      ::mlir::Attribute value = prop.size;
      ::mlir::Attribute originalValue = value;
      visitor("size", value);
      if (value != originalValue)
        setInherentAttr(prop, "size", value);
    }
}

::llvm::LogicalResult CreateWindowOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getDirectionAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "direction", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(attr, "name", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPingaddrAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops3(attr, "pingaddr", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPinglockidAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "pinglockid", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPongaddrAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops3(attr, "pongaddr", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getPonglockidAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "ponglockid", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "size", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult CreateWindowOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.direction)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.pingaddr)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.pinglockid)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.pongaddr)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.ponglockid)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.size)))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateWindowOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.direction);
  writer.writeAttribute(prop.name);
  writer.writeAttribute(prop.pingaddr);
  writer.writeAttribute(prop.pinglockid);
  writer.writeAttribute(prop.pongaddr);
  writer.writeAttribute(prop.ponglockid);
  writer.writeAttribute(prop.size);
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
  getProperties().direction = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void CreateWindowOp::setSize(uint32_t attrValue) {
  getProperties().size = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void CreateWindowOp::setPingaddr(uint64_t attrValue) {
  getProperties().pingaddr = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void CreateWindowOp::setPongaddr(uint64_t attrValue) {
  getProperties().pongaddr = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void CreateWindowOp::setPinglockid(uint32_t attrValue) {
  getProperties().pinglockid = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void CreateWindowOp::setPonglockid(uint32_t attrValue) {
  getProperties().ponglockid = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void CreateWindowOp::setName(::llvm::StringRef attrValue) {
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void CreateWindowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.direction = direction;
  odsProperties.size = size;
  odsProperties.pingaddr = pingaddr;
  odsProperties.pongaddr = pongaddr;
  odsProperties.pinglockid = pinglockid;
  odsProperties.ponglockid = ponglockid;
  odsProperties.name = name;
  odsState.addTypes(result);
}

CreateWindowOp CreateWindowOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(direction)>(direction), std::forward<decltype(size)>(size), std::forward<decltype(pingaddr)>(pingaddr), std::forward<decltype(pongaddr)>(pongaddr), std::forward<decltype(pinglockid)>(pinglockid), std::forward<decltype(ponglockid)>(ponglockid), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateWindowOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateWindowOp>(__res__);
}

CreateWindowOp CreateWindowOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(direction)>(direction), std::forward<decltype(size)>(size), std::forward<decltype(pingaddr)>(pingaddr), std::forward<decltype(pongaddr)>(pongaddr), std::forward<decltype(pinglockid)>(pinglockid), std::forward<decltype(ponglockid)>(ponglockid), std::forward<decltype(name)>(name));
}

void CreateWindowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.direction = direction;
  odsProperties.size = size;
  odsProperties.pingaddr = pingaddr;
  odsProperties.pongaddr = pongaddr;
  odsProperties.pinglockid = pinglockid;
  odsProperties.ponglockid = ponglockid;
  odsProperties.name = name;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateWindowOp CreateWindowOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(direction)>(direction), std::forward<decltype(size)>(size), std::forward<decltype(pingaddr)>(pingaddr), std::forward<decltype(pongaddr)>(pongaddr), std::forward<decltype(pinglockid)>(pinglockid), std::forward<decltype(ponglockid)>(ponglockid), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateWindowOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateWindowOp>(__res__);
}

CreateWindowOp CreateWindowOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(direction)>(direction), std::forward<decltype(size)>(size), std::forward<decltype(pingaddr)>(pingaddr), std::forward<decltype(pongaddr)>(pongaddr), std::forward<decltype(pinglockid)>(pinglockid), std::forward<decltype(ponglockid)>(ponglockid), std::forward<decltype(name)>(name));
}

void CreateWindowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.direction = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), direction);
  odsProperties.size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size);
  odsProperties.pingaddr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pingaddr);
  odsProperties.pongaddr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pongaddr);
  odsProperties.pinglockid = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pinglockid);
  odsProperties.ponglockid = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), ponglockid);
  odsProperties.name = odsBuilder.getStringAttr(name);
  odsState.addTypes(result);
}

CreateWindowOp CreateWindowOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(direction)>(direction), std::forward<decltype(size)>(size), std::forward<decltype(pingaddr)>(pingaddr), std::forward<decltype(pongaddr)>(pongaddr), std::forward<decltype(pinglockid)>(pinglockid), std::forward<decltype(ponglockid)>(ponglockid), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateWindowOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateWindowOp>(__res__);
}

CreateWindowOp CreateWindowOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(direction)>(direction), std::forward<decltype(size)>(size), std::forward<decltype(pingaddr)>(pingaddr), std::forward<decltype(pongaddr)>(pongaddr), std::forward<decltype(pinglockid)>(pinglockid), std::forward<decltype(ponglockid)>(ponglockid), std::forward<decltype(name)>(name));
}

void CreateWindowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.direction = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), direction);
  odsProperties.size = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size);
  odsProperties.pingaddr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pingaddr);
  odsProperties.pongaddr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), pongaddr);
  odsProperties.pinglockid = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), pinglockid);
  odsProperties.ponglockid = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), ponglockid);
  odsProperties.name = odsBuilder.getStringAttr(name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

CreateWindowOp CreateWindowOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(direction)>(direction), std::forward<decltype(size)>(size), std::forward<decltype(pingaddr)>(pingaddr), std::forward<decltype(pongaddr)>(pongaddr), std::forward<decltype(pinglockid)>(pinglockid), std::forward<decltype(ponglockid)>(ponglockid), std::forward<decltype(name)>(name));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateWindowOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateWindowOp>(__res__);
}

CreateWindowOp CreateWindowOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(direction)>(direction), std::forward<decltype(size)>(size), std::forward<decltype(pingaddr)>(pingaddr), std::forward<decltype(pongaddr)>(pongaddr), std::forward<decltype(pinglockid)>(pinglockid), std::forward<decltype(ponglockid)>(ponglockid), std::forward<decltype(name)>(name));
}

void CreateWindowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateWindowOp CreateWindowOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateWindowOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateWindowOp>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

CreateWindowOp CreateWindowOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void CreateWindowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

CreateWindowOp CreateWindowOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<CreateWindowOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<CreateWindowOp>(__res__);
}

CreateWindowOp CreateWindowOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult CreateWindowOp::verifyInvariantsImpl() {
  auto tblgen_direction = getProperties().direction; (void)tblgen_direction;
  if (!tblgen_direction) return emitOpError("requires attribute 'direction'");
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");
  auto tblgen_pingaddr = getProperties().pingaddr; (void)tblgen_pingaddr;
  if (!tblgen_pingaddr) return emitOpError("requires attribute 'pingaddr'");
  auto tblgen_pinglockid = getProperties().pinglockid; (void)tblgen_pinglockid;
  if (!tblgen_pinglockid) return emitOpError("requires attribute 'pinglockid'");
  auto tblgen_pongaddr = getProperties().pongaddr; (void)tblgen_pongaddr;
  if (!tblgen_pongaddr) return emitOpError("requires attribute 'pongaddr'");
  auto tblgen_ponglockid = getProperties().ponglockid; (void)tblgen_ponglockid;
  if (!tblgen_ponglockid) return emitOpError("requires attribute 'ponglockid'");
  auto tblgen_size = getProperties().size; (void)tblgen_size;
  if (!tblgen_size) return emitOpError("requires attribute 'size'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_direction, "direction")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_size, "size")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops3(*this, tblgen_pingaddr, "pingaddr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops3(*this, tblgen_pongaddr, "pongaddr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_pinglockid, "pinglockid")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_ponglockid, "ponglockid")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops12(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult CreateWindowOp::verifyInvariants() {
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
  if (directionAttr) result.getOrAddProperties<CreateWindowOp::Properties>().direction = directionAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(sizeAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (sizeAttr) result.getOrAddProperties<CreateWindowOp::Properties>().size = sizeAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(pingaddrAttr, parser.getBuilder().getIntegerType(64))) {
    return ::mlir::failure();
  }
  if (pingaddrAttr) result.getOrAddProperties<CreateWindowOp::Properties>().pingaddr = pingaddrAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(pongaddrAttr, parser.getBuilder().getIntegerType(64))) {
    return ::mlir::failure();
  }
  if (pongaddrAttr) result.getOrAddProperties<CreateWindowOp::Properties>().pongaddr = pongaddrAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(pinglockidAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (pinglockidAttr) result.getOrAddProperties<CreateWindowOp::Properties>().pinglockid = pinglockidAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(ponglockidAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (ponglockidAttr) result.getOrAddProperties<CreateWindowOp::Properties>().ponglockid = ponglockidAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<CreateWindowOp::Properties>().name = nameAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void CreateWindowOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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
  static constexpr ::llvm::StringRef elidedAttrs[] = {"direction", "size", "pingaddr", "pongaddr", "pinglockid", "ponglockid", "name"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateWindowOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void CreateWindowOp::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"direction", "size", "pingaddr", "pongaddr", "pinglockid", "ponglockid", "name"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CreateWindowOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateWindowOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::DeRoute definitions
//===----------------------------------------------------------------------===//

namespace detail {

DeRouteGenericAdaptorBase::DeRouteGenericAdaptorBase(DeRoute op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
DeRouteAdaptor::DeRouteAdaptor(DeRoute op) : DeRouteGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult DeRouteAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult DeRoute::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute DeRoute::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code DeRoute::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> DeRoute::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void DeRoute::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void DeRoute::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult DeRoute::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void DeRoute::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, source, destination, shouldModifyCoreConfig);
}

DeRoute DeRoute::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination), std::forward<decltype(shouldModifyCoreConfig)>(shouldModifyCoreConfig));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<DeRoute>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<DeRoute>(__res__);
}

DeRoute DeRoute::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination), std::forward<decltype(shouldModifyCoreConfig)>(shouldModifyCoreConfig));
}

void DeRoute::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(source);
  odsState.addOperands(destination);
  odsState.addOperands(shouldModifyCoreConfig);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

DeRoute DeRoute::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination), std::forward<decltype(shouldModifyCoreConfig)>(shouldModifyCoreConfig));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<DeRoute>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<DeRoute>(__res__);
}

DeRoute DeRoute::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination), std::forward<decltype(shouldModifyCoreConfig)>(shouldModifyCoreConfig));
}

void DeRoute::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

DeRoute DeRoute::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<DeRoute>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<DeRoute>(__res__);
}

DeRoute DeRoute::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void DeRoute::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

DeRoute DeRoute::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<DeRoute>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<DeRoute>(__res__);
}

DeRoute DeRoute::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult DeRoute::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult DeRoute::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult DeRoute::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void DeRoute::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::DeRoute)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::InitRoutingHandler definitions
//===----------------------------------------------------------------------===//

namespace detail {

InitRoutingHandlerGenericAdaptorBase::InitRoutingHandlerGenericAdaptorBase(InitRoutingHandler op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
InitRoutingHandlerAdaptor::InitRoutingHandlerAdaptor(InitRoutingHandler op) : InitRoutingHandlerGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult InitRoutingHandlerAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult InitRoutingHandler::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute InitRoutingHandler::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code InitRoutingHandler::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> InitRoutingHandler::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void InitRoutingHandler::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void InitRoutingHandler::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult InitRoutingHandler::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void InitRoutingHandler::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type instance, ::mlir::Value devInst) {
  odsState.addOperands(devInst);
  odsState.addTypes(instance);
}

InitRoutingHandler InitRoutingHandler::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type instance, ::mlir::Value devInst) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(instance)>(instance), std::forward<decltype(devInst)>(devInst));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<InitRoutingHandler>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<InitRoutingHandler>(__res__);
}

InitRoutingHandler InitRoutingHandler::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type instance, ::mlir::Value devInst) {
  return create(builder, builder.getLoc(), std::forward<decltype(instance)>(instance), std::forward<decltype(devInst)>(devInst));
}

void InitRoutingHandler::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value devInst) {
  odsState.addOperands(devInst);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

InitRoutingHandler InitRoutingHandler::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value devInst) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(devInst)>(devInst));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<InitRoutingHandler>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<InitRoutingHandler>(__res__);
}

InitRoutingHandler InitRoutingHandler::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value devInst) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(devInst)>(devInst));
}

void InitRoutingHandler::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

InitRoutingHandler InitRoutingHandler::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<InitRoutingHandler>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<InitRoutingHandler>(__res__);
}

InitRoutingHandler InitRoutingHandler::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void InitRoutingHandler::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

InitRoutingHandler InitRoutingHandler::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<InitRoutingHandler>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<InitRoutingHandler>(__res__);
}

InitRoutingHandler InitRoutingHandler::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult InitRoutingHandler::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult InitRoutingHandler::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult InitRoutingHandler::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void InitRoutingHandler::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::InitRoutingHandler)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::KernelCall definitions
//===----------------------------------------------------------------------===//

namespace detail {

KernelCallGenericAdaptorBase::KernelCallGenericAdaptorBase(KernelCall op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

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

::llvm::LogicalResult KernelCallAdaptor::verify(::mlir::Location loc) {
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

::llvm::LogicalResult KernelCall::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute KernelCall::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code KernelCall::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> KernelCall::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void KernelCall::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void KernelCall::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult KernelCall::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void KernelCall::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, Value col, Value row, ArrayRef<Value> arguments) {
      ValueRange args = ValueRange(arguments);
      build(odsBuilder, odsState, result_type, col, row, args);
    
}

KernelCall KernelCall::create(::mlir::OpBuilder &builder, ::mlir::Location location, Type result_type, Value col, Value row, ArrayRef<Value> arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result_type)>(result_type), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelCall>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelCall>(__res__);
}

KernelCall KernelCall::create(::mlir::ImplicitLocOpBuilder &builder, Type result_type, Value col, Value row, ArrayRef<Value> arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(result_type)>(result_type), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
}

void KernelCall::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, col, row, arguments);
}

KernelCall KernelCall::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelCall>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelCall>(__res__);
}

KernelCall KernelCall::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
}

void KernelCall::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments) {
  odsState.addOperands(col);
  odsState.addOperands(row);
  odsState.addOperands(arguments);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

KernelCall KernelCall::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelCall>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelCall>(__res__);
}

KernelCall KernelCall::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
}

void KernelCall::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

KernelCall KernelCall::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelCall>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelCall>(__res__);
}

KernelCall KernelCall::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void KernelCall::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

KernelCall KernelCall::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelCall>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelCall>(__res__);
}

KernelCall KernelCall::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult KernelCall::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult KernelCall::verifyInvariants() {
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
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
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(colOperands, odsBuildableType0, colOperandsLoc, result.operands))
    return ::mlir::failure();
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(rowOperands, odsBuildableType0, rowOperandsLoc, result.operands))
    return ::mlir::failure();
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void KernelCall::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCol();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRow();
  _odsPrinter << "(";
  _odsPrinter << getArguments();
  _odsPrinter << ")";
  ::llvm::ArrayRef<::llvm::StringRef> elidedAttrs;
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getArguments().getTypes(), getOperation()->getResultTypes());
}

void KernelCall::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::KernelCall)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::KernelDefinition definitions
//===----------------------------------------------------------------------===//

namespace detail {

KernelDefinitionGenericAdaptorBase::KernelDefinitionGenericAdaptorBase(KernelDefinition op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

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

::llvm::LogicalResult KernelDefinitionAdaptor::verify(::mlir::Location loc) {
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

::llvm::LogicalResult KernelDefinition::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute KernelDefinition::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code KernelDefinition::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> KernelDefinition::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void KernelDefinition::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void KernelDefinition::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult KernelDefinition::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void KernelDefinition::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, ArrayRef<Value> arguments) {
      ValueRange args = ValueRange(arguments);
      build(odsBuilder, odsState, result_type, args);
    
}

KernelDefinition KernelDefinition::create(::mlir::OpBuilder &builder, ::mlir::Location location, Type result_type, ArrayRef<Value> arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result_type)>(result_type), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelDefinition>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelDefinition>(__res__);
}

KernelDefinition KernelDefinition::create(::mlir::ImplicitLocOpBuilder &builder, Type result_type, ArrayRef<Value> arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(result_type)>(result_type), std::forward<decltype(arguments)>(arguments));
}

void KernelDefinition::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange arguments) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, arguments);
}

KernelDefinition KernelDefinition::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelDefinition>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelDefinition>(__res__);
}

KernelDefinition KernelDefinition::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(arguments)>(arguments));
}

void KernelDefinition::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

KernelDefinition KernelDefinition::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelDefinition>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelDefinition>(__res__);
}

KernelDefinition KernelDefinition::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void KernelDefinition::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

KernelDefinition KernelDefinition::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<KernelDefinition>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<KernelDefinition>(__res__);
}

KernelDefinition KernelDefinition::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult KernelDefinition::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult KernelDefinition::verifyInvariants() {
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
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
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void KernelDefinition::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getArguments();
  _odsPrinter << ")";
  ::llvm::ArrayRef<::llvm::StringRef> elidedAttrs;
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getArguments().getTypes(), getOperation()->getResultTypes());
}

void KernelDefinition::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::KernelDefinition)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::LoadKernel definitions
//===----------------------------------------------------------------------===//

namespace detail {

LoadKernelGenericAdaptorBase::LoadKernelGenericAdaptorBase(LoadKernel op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

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

::llvm::StringRef LoadKernelGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

uint32_t LoadKernelGenericAdaptorBase::getCol() {
  auto attr = getColAttr();
  return attr.getValue().getZExtValue();
}

uint32_t LoadKernelGenericAdaptorBase::getRow() {
  auto attr = getRowAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
LoadKernelAdaptor::LoadKernelAdaptor(LoadKernel op) : LoadKernelGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoadKernelAdaptor::verify(::mlir::Location loc) {
  auto tblgen_col = getProperties().col; (void)tblgen_col;
  if (!tblgen_col) return emitError(loc, "'Aie.load_kernel_op' op requires attribute 'col'");
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'Aie.load_kernel_op' op requires attribute 'name'");
  auto tblgen_row = getProperties().row; (void)tblgen_row;
  if (!tblgen_row) return emitError(loc, "'Aie.load_kernel_op' op requires attribute 'row'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'Aie.load_kernel_op' op attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_col && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_col))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_col).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.load_kernel_op' op attribute 'col' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_row && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_row))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_row).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.load_kernel_op' op attribute 'row' failed to satisfy constraint: 32-bit signless integer attribute");
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

::llvm::LogicalResult LoadKernel::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("col");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.col, attr, "col", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("name");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.name, attr, "name", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("row");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.row, attr, "row", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute LoadKernel::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "col", prop.col);
    ::mlir::detail::appendAttributeProperty(attrs, "name", prop.name);
    ::mlir::detail::appendAttributeProperty(attrs, "row", prop.row);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code LoadKernel::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.col.getAsOpaquePointer()), 
    llvm::hash_value(prop.name.getAsOpaquePointer()), 
    llvm::hash_value(prop.row.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> LoadKernel::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "col")
      return prop.col;

    if (name == "name")
      return prop.name;

    if (name == "row")
      return prop.row;
  return std::nullopt;
}

void LoadKernel::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "col") {
       prop.col = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.col)>>(value);
       return;
    }

    if (name == "name") {
       prop.name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }

    if (name == "row") {
       prop.row = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.row)>>(value);
       return;
    }
}

void LoadKernel::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.col) {
      ::mlir::Attribute value = prop.col;
      ::mlir::Attribute originalValue = value;
      visitor("col", value);
      if (value != originalValue)
        setInherentAttr(prop, "col", value);
    }

    if (prop.name) {
      ::mlir::Attribute value = prop.name;
      ::mlir::Attribute originalValue = value;
      visitor("name", value);
      if (value != originalValue)
        setInherentAttr(prop, "name", value);
    }

    if (prop.row) {
      ::mlir::Attribute value = prop.row;
      ::mlir::Attribute originalValue = value;
      visitor("row", value);
      if (value != originalValue)
        setInherentAttr(prop, "row", value);
    }
}

::llvm::LogicalResult LoadKernel::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getColAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "col", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(attr, "name", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRowAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "row", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult LoadKernel::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.col)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.name)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.row)))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadKernel::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.col);
  writer.writeAttribute(prop.name);
  writer.writeAttribute(prop.row);
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
  getProperties().name = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void LoadKernel::setCol(uint32_t attrValue) {
  getProperties().col = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void LoadKernel::setRow(uint32_t attrValue) {
  getProperties().row = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void LoadKernel::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, name, col, row, arguments);
}

LoadKernel LoadKernel::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(name)>(name), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernel>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernel>(__res__);
}

LoadKernel LoadKernel::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(name)>(name), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
}

void LoadKernel::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(arguments);
  odsProperties.name = name;
  odsProperties.col = col;
  odsProperties.row = row;
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

LoadKernel LoadKernel::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernel>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernel>(__res__);
}

LoadKernel LoadKernel::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
}

void LoadKernel::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, name, col, row, arguments);
}

LoadKernel LoadKernel::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(name)>(name), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernel>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernel>(__res__);
}

LoadKernel LoadKernel::create(::mlir::ImplicitLocOpBuilder &builder, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(name)>(name), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
}

void LoadKernel::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(arguments);
  odsProperties.name = odsBuilder.getStringAttr(name);
  odsProperties.col = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), col);
  odsProperties.row = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), row);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

LoadKernel LoadKernel::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernel>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernel>(__res__);
}

LoadKernel LoadKernel::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(name)>(name), std::forward<decltype(col)>(col), std::forward<decltype(row)>(row), std::forward<decltype(arguments)>(arguments));
}

void LoadKernel::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LoadKernel LoadKernel::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernel>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernel>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

LoadKernel LoadKernel::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void LoadKernel::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LoadKernel LoadKernel::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernel>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernel>(__res__);
}

LoadKernel LoadKernel::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult LoadKernel::verifyInvariantsImpl() {
  auto tblgen_col = getProperties().col; (void)tblgen_col;
  if (!tblgen_col) return emitOpError("requires attribute 'col'");
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");
  auto tblgen_row = getProperties().row; (void)tblgen_row;
  if (!tblgen_row) return emitOpError("requires attribute 'row'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops1(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_col, "col")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_row, "row")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoadKernel::verifyInvariants() {
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
  if (nameAttr) result.getOrAddProperties<LoadKernel::Properties>().name = nameAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(colAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (colAttr) result.getOrAddProperties<LoadKernel::Properties>().col = colAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(rowAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (rowAttr) result.getOrAddProperties<LoadKernel::Properties>().row = rowAttr;
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
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
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void LoadKernel::print(::mlir::OpAsmPrinter &_odsPrinter) {
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
  static constexpr ::llvm::StringRef elidedAttrs[] = {"name", "col", "row"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getArguments().getTypes(), getOperation()->getResultTypes());
}

void LoadKernel::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void LoadKernel::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"name", "col", "row"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::LoadKernel)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::LoadKernelGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::LoadKernelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

LoadKernelOpGenericAdaptorBase::LoadKernelOpGenericAdaptorBase(LoadKernelOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
LoadKernelOpAdaptor::LoadKernelOpAdaptor(LoadKernelOp op) : LoadKernelOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult LoadKernelOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult LoadKernelOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute LoadKernelOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code LoadKernelOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> LoadKernelOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void LoadKernelOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void LoadKernelOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult LoadKernelOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void LoadKernelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, lhs, rhs);
}

LoadKernelOp LoadKernelOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernelOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernelOp>(__res__);
}

LoadKernelOp LoadKernelOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs) {
  return create(builder, builder.getLoc(), std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
}

void LoadKernelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

LoadKernelOp LoadKernelOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernelOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernelOp>(__res__);
}

LoadKernelOp LoadKernelOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(lhs)>(lhs), std::forward<decltype(rhs)>(rhs));
}

void LoadKernelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LoadKernelOp LoadKernelOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernelOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernelOp>(__res__);
}

LoadKernelOp LoadKernelOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void LoadKernelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LoadKernelOp LoadKernelOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<LoadKernelOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<LoadKernelOp>(__res__);
}

LoadKernelOp LoadKernelOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult LoadKernelOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult LoadKernelOp::verifyInvariants() {
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(lhsOperands, odsBuildableType0, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(rhsOperands, odsBuildableType0, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void LoadKernelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ",";
  ::llvm::ArrayRef<::llvm::StringRef> elidedAttrs;
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::LoadKernelOp)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::MoveData definitions
//===----------------------------------------------------------------------===//

namespace detail {

MoveDataGenericAdaptorBase::MoveDataGenericAdaptorBase(MoveData op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
MoveDataAdaptor::MoveDataAdaptor(MoveData op) : MoveDataGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult MoveDataAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult MoveData::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute MoveData::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code MoveData::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> MoveData::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void MoveData::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void MoveData::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult MoveData::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void MoveData::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, source, sourceObject, dataSize, destinationObject, destination);
}

MoveData MoveData::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(sourceObject)>(sourceObject), std::forward<decltype(dataSize)>(dataSize), std::forward<decltype(destinationObject)>(destinationObject), std::forward<decltype(destination)>(destination));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<MoveData>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<MoveData>(__res__);
}

MoveData MoveData::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(sourceObject)>(sourceObject), std::forward<decltype(dataSize)>(dataSize), std::forward<decltype(destinationObject)>(destinationObject), std::forward<decltype(destination)>(destination));
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

MoveData MoveData::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(sourceObject)>(sourceObject), std::forward<decltype(dataSize)>(dataSize), std::forward<decltype(destinationObject)>(destinationObject), std::forward<decltype(destination)>(destination));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<MoveData>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<MoveData>(__res__);
}

MoveData MoveData::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(sourceObject)>(sourceObject), std::forward<decltype(dataSize)>(dataSize), std::forward<decltype(destinationObject)>(destinationObject), std::forward<decltype(destination)>(destination));
}

void MoveData::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

MoveData MoveData::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<MoveData>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<MoveData>(__res__);
}

MoveData MoveData::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void MoveData::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

MoveData MoveData::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<MoveData>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<MoveData>(__res__);
}

MoveData MoveData::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult MoveData::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult MoveData::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult MoveData::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void MoveData::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::MoveData)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ResetHostEdgeConstraints definitions
//===----------------------------------------------------------------------===//

namespace detail {

ResetHostEdgeConstraintsGenericAdaptorBase::ResetHostEdgeConstraintsGenericAdaptorBase(ResetHostEdgeConstraints op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
ResetHostEdgeConstraintsAdaptor::ResetHostEdgeConstraintsAdaptor(ResetHostEdgeConstraints op) : ResetHostEdgeConstraintsGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ResetHostEdgeConstraintsAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult ResetHostEdgeConstraints::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute ResetHostEdgeConstraints::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ResetHostEdgeConstraints::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> ResetHostEdgeConstraints::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void ResetHostEdgeConstraints::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void ResetHostEdgeConstraints::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult ResetHostEdgeConstraints::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void ResetHostEdgeConstraints::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, tile);
}

ResetHostEdgeConstraints ResetHostEdgeConstraints::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ResetHostEdgeConstraints>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ResetHostEdgeConstraints>(__res__);
}

ResetHostEdgeConstraints ResetHostEdgeConstraints::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
}

void ResetHostEdgeConstraints::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ResetHostEdgeConstraints ResetHostEdgeConstraints::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ResetHostEdgeConstraints>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ResetHostEdgeConstraints>(__res__);
}

ResetHostEdgeConstraints ResetHostEdgeConstraints::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
}

void ResetHostEdgeConstraints::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ResetHostEdgeConstraints ResetHostEdgeConstraints::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ResetHostEdgeConstraints>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ResetHostEdgeConstraints>(__res__);
}

ResetHostEdgeConstraints ResetHostEdgeConstraints::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void ResetHostEdgeConstraints::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ResetHostEdgeConstraints ResetHostEdgeConstraints::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ResetHostEdgeConstraints>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ResetHostEdgeConstraints>(__res__);
}

ResetHostEdgeConstraints ResetHostEdgeConstraints::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ResetHostEdgeConstraints::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ResetHostEdgeConstraints::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult ResetHostEdgeConstraints::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void ResetHostEdgeConstraints::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ResetHostEdgeConstraints)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::Route definitions
//===----------------------------------------------------------------------===//

namespace detail {

RouteGenericAdaptorBase::RouteGenericAdaptorBase(Route op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
RouteAdaptor::RouteAdaptor(Route op) : RouteGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RouteAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult Route::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute Route::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code Route::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> Route::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void Route::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void Route::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult Route::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void Route::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, routeConstraints, source, destination);
}

Route Route::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(routeConstraints)>(routeConstraints), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<Route>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<Route>(__res__);
}

Route Route::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(routeConstraints)>(routeConstraints), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination));
}

void Route::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(routeConstraints);
  odsState.addOperands(source);
  odsState.addOperands(destination);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

Route Route::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(routeConstraints)>(routeConstraints), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<Route>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<Route>(__res__);
}

Route Route::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(routeConstraints)>(routeConstraints), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination));
}

void Route::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

Route Route::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<Route>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<Route>(__res__);
}

Route Route::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void Route::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

Route Route::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<Route>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<Route>(__res__);
}

Route Route::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult Route::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Route::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult Route::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void Route::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::Route)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::RouteDmaWait definitions
//===----------------------------------------------------------------------===//

namespace detail {

RouteDmaWaitGenericAdaptorBase::RouteDmaWaitGenericAdaptorBase(RouteDmaWait op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
RouteDmaWaitAdaptor::RouteDmaWaitAdaptor(RouteDmaWait op) : RouteDmaWaitGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RouteDmaWaitAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult RouteDmaWait::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute RouteDmaWait::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code RouteDmaWait::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> RouteDmaWait::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void RouteDmaWait::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void RouteDmaWait::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult RouteDmaWait::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void RouteDmaWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, tile, isS2MM);
}

RouteDmaWait RouteDmaWait::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile), std::forward<decltype(isS2MM)>(isS2MM));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RouteDmaWait>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RouteDmaWait>(__res__);
}

RouteDmaWait RouteDmaWait::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile), std::forward<decltype(isS2MM)>(isS2MM));
}

void RouteDmaWait::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  odsState.addOperands(isS2MM);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

RouteDmaWait RouteDmaWait::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile), std::forward<decltype(isS2MM)>(isS2MM));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RouteDmaWait>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RouteDmaWait>(__res__);
}

RouteDmaWait RouteDmaWait::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile), std::forward<decltype(isS2MM)>(isS2MM));
}

void RouteDmaWait::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RouteDmaWait RouteDmaWait::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RouteDmaWait>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RouteDmaWait>(__res__);
}

RouteDmaWait RouteDmaWait::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void RouteDmaWait::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RouteDmaWait RouteDmaWait::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RouteDmaWait>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RouteDmaWait>(__res__);
}

RouteDmaWait RouteDmaWait::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult RouteDmaWait::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult RouteDmaWait::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult RouteDmaWait::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void RouteDmaWait::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::RouteDmaWait)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::RoutesReveal definitions
//===----------------------------------------------------------------------===//

namespace detail {

RoutesRevealGenericAdaptorBase::RoutesRevealGenericAdaptorBase(RoutesReveal op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
RoutesRevealAdaptor::RoutesRevealAdaptor(RoutesReveal op) : RoutesRevealGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RoutesRevealAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult RoutesReveal::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute RoutesReveal::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code RoutesReveal::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> RoutesReveal::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void RoutesReveal::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void RoutesReveal::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult RoutesReveal::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void RoutesReveal::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, source, destination);
}

RoutesReveal RoutesReveal::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RoutesReveal>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RoutesReveal>(__res__);
}

RoutesReveal RoutesReveal::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination));
}

void RoutesReveal::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(source);
  odsState.addOperands(destination);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

RoutesReveal RoutesReveal::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RoutesReveal>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RoutesReveal>(__res__);
}

RoutesReveal RoutesReveal::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(source)>(source), std::forward<decltype(destination)>(destination));
}

void RoutesReveal::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RoutesReveal RoutesReveal::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RoutesReveal>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RoutesReveal>(__res__);
}

RoutesReveal RoutesReveal::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void RoutesReveal::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RoutesReveal RoutesReveal::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RoutesReveal>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RoutesReveal>(__res__);
}

RoutesReveal RoutesReveal::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult RoutesReveal::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult RoutesReveal::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult RoutesReveal::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void RoutesReveal::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::RoutesReveal)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::RoutingInstanceFree definitions
//===----------------------------------------------------------------------===//

namespace detail {

RoutingInstanceFreeGenericAdaptorBase::RoutingInstanceFreeGenericAdaptorBase(RoutingInstanceFree op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
RoutingInstanceFreeAdaptor::RoutingInstanceFreeAdaptor(RoutingInstanceFree op) : RoutingInstanceFreeGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RoutingInstanceFreeAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult RoutingInstanceFree::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute RoutingInstanceFree::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code RoutingInstanceFree::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> RoutingInstanceFree::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void RoutingInstanceFree::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void RoutingInstanceFree::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult RoutingInstanceFree::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void RoutingInstanceFree::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance);
}

RoutingInstanceFree RoutingInstanceFree::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RoutingInstanceFree>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RoutingInstanceFree>(__res__);
}

RoutingInstanceFree RoutingInstanceFree::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance));
}

void RoutingInstanceFree::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance) {
  odsState.addOperands(routingInstance);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

RoutingInstanceFree RoutingInstanceFree::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RoutingInstanceFree>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RoutingInstanceFree>(__res__);
}

RoutingInstanceFree RoutingInstanceFree::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance));
}

void RoutingInstanceFree::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RoutingInstanceFree RoutingInstanceFree::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RoutingInstanceFree>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RoutingInstanceFree>(__res__);
}

RoutingInstanceFree RoutingInstanceFree::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void RoutingInstanceFree::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

RoutingInstanceFree RoutingInstanceFree::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<RoutingInstanceFree>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<RoutingInstanceFree>(__res__);
}

RoutingInstanceFree RoutingInstanceFree::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult RoutingInstanceFree::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult RoutingInstanceFree::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult RoutingInstanceFree::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void RoutingInstanceFree::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::RoutingInstanceFree)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::Run definitions
//===----------------------------------------------------------------------===//

namespace detail {

RunGenericAdaptorBase::RunGenericAdaptorBase(Run op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
RunAdaptor::RunAdaptor(Run op) : RunGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult RunAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult Run::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute Run::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code Run::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> Run::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void Run::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void Run::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult Run::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void Run::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, tile);
}

Run Run::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<Run>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<Run>(__res__);
}

Run Run::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
}

void Run::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

Run Run::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<Run>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<Run>(__res__);
}

Run Run::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile));
}

void Run::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

Run Run::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<Run>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<Run>(__res__);
}

Run Run::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void Run::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

Run Run::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<Run>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<Run>(__res__);
}

Run Run::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult Run::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult Run::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult Run::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void Run::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::Run)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::SetCoreExecute definitions
//===----------------------------------------------------------------------===//

namespace detail {

SetCoreExecuteGenericAdaptorBase::SetCoreExecuteGenericAdaptorBase(SetCoreExecute op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}


} // namespace detail
SetCoreExecuteAdaptor::SetCoreExecuteAdaptor(SetCoreExecute op) : SetCoreExecuteGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult SetCoreExecuteAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::LogicalResult SetCoreExecute::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
      return ::mlir::success();
}

::mlir::Attribute SetCoreExecute::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code SetCoreExecute::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine();
}

std::optional<mlir::Attribute> SetCoreExecute::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
  return std::nullopt;
}

void SetCoreExecute::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
}

void SetCoreExecute::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
}

::llvm::LogicalResult SetCoreExecute::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    return ::mlir::success();
}

void SetCoreExecute::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute) {
  build(odsBuilder, odsState, ::mlir::TypeRange{}, routingInstance, tile, isExecute);
}

SetCoreExecute SetCoreExecute::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile), std::forward<decltype(isExecute)>(isExecute));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<SetCoreExecute>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<SetCoreExecute>(__res__);
}

SetCoreExecute SetCoreExecute::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute) {
  return create(builder, builder.getLoc(), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile), std::forward<decltype(isExecute)>(isExecute));
}

void SetCoreExecute::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute) {
  odsState.addOperands(routingInstance);
  odsState.addOperands(tile);
  odsState.addOperands(isExecute);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

SetCoreExecute SetCoreExecute::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile), std::forward<decltype(isExecute)>(isExecute));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<SetCoreExecute>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<SetCoreExecute>(__res__);
}

SetCoreExecute SetCoreExecute::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(routingInstance)>(routingInstance), std::forward<decltype(tile)>(tile), std::forward<decltype(isExecute)>(isExecute));
}

void SetCoreExecute::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

SetCoreExecute SetCoreExecute::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<SetCoreExecute>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<SetCoreExecute>(__res__);
}

SetCoreExecute SetCoreExecute::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
}

void SetCoreExecute::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

SetCoreExecute SetCoreExecute::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<SetCoreExecute>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<SetCoreExecute>(__res__);
}

SetCoreExecute SetCoreExecute::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult SetCoreExecute::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult SetCoreExecute::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult SetCoreExecute::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void SetCoreExecute::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::SetCoreExecute)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantInteger32Op definitions
//===----------------------------------------------------------------------===//

namespace detail {

ConstantInteger32OpGenericAdaptorBase::ConstantInteger32OpGenericAdaptorBase(ConstantInteger32Op op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t ConstantInteger32OpGenericAdaptorBase::getInputAttr() {
  auto attr = getInputAttrAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
ConstantInteger32OpAdaptor::ConstantInteger32OpAdaptor(ConstantInteger32Op op) : ConstantInteger32OpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConstantInteger32OpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_inputAttr = getProperties().inputAttr; (void)tblgen_inputAttr;
  if (!tblgen_inputAttr) return emitError(loc, "'Aie.constant_integer32_op' op requires attribute 'inputAttr'");

  if (tblgen_inputAttr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_inputAttr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_inputAttr).getType().isSignlessInteger(32)))))
    return emitError(loc, "'Aie.constant_integer32_op' op attribute 'inputAttr' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ConstantInteger32Op::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("inputAttr");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.inputAttr, attr, "inputAttr", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute ConstantInteger32Op::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "inputAttr", prop.inputAttr);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConstantInteger32Op::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.inputAttr.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConstantInteger32Op::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "inputAttr")
      return prop.inputAttr;
  return std::nullopt;
}

void ConstantInteger32Op::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "inputAttr") {
       prop.inputAttr = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.inputAttr)>>(value);
       return;
    }
}

void ConstantInteger32Op::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.inputAttr) {
      ::mlir::Attribute value = prop.inputAttr;
      ::mlir::Attribute originalValue = value;
      visitor("inputAttr", value);
      if (value != originalValue)
        setInherentAttr(prop, "inputAttr", value);
    }
}

::llvm::LogicalResult ConstantInteger32Op::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getInputAttrAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(attr, "inputAttr", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ConstantInteger32Op::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.inputAttr)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConstantInteger32Op::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.inputAttr);
}

uint32_t ConstantInteger32Op::getInputAttr() {
  auto attr = getInputAttrAttr();
  return attr.getValue().getZExtValue();
}

void ConstantInteger32Op::setInputAttr(uint32_t attrValue) {
  getProperties().inputAttr = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, IntegerAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, IntegerAttr value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(value)>(value));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger32Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger32Op>(__res__);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::ImplicitLocOpBuilder &builder, IntegerAttr value) {
  return create(builder, builder.getLoc(), std::forward<decltype(value)>(value));
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.inputAttr = inputAttr;
  odsState.addTypes(outputValue);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(outputValue)>(outputValue), std::forward<decltype(inputAttr)>(inputAttr));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger32Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger32Op>(__res__);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr) {
  return create(builder, builder.getLoc(), std::forward<decltype(outputValue)>(outputValue), std::forward<decltype(inputAttr)>(inputAttr));
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.inputAttr = inputAttr;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(inputAttr)>(inputAttr));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger32Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger32Op>(__res__);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(inputAttr)>(inputAttr));
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, uint32_t inputAttr) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.inputAttr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), inputAttr);
  odsState.addTypes(outputValue);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type outputValue, uint32_t inputAttr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(outputValue)>(outputValue), std::forward<decltype(inputAttr)>(inputAttr));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger32Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger32Op>(__res__);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type outputValue, uint32_t inputAttr) {
  return create(builder, builder.getLoc(), std::forward<decltype(outputValue)>(outputValue), std::forward<decltype(inputAttr)>(inputAttr));
}

void ConstantInteger32Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t inputAttr) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.inputAttr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), inputAttr);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t inputAttr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(inputAttr)>(inputAttr));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger32Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger32Op>(__res__);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t inputAttr) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(inputAttr)>(inputAttr));
}

void ConstantInteger32Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger32Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger32Op>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void ConstantInteger32Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger32Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger32Op>(__res__);
}

ConstantInteger32Op ConstantInteger32Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ConstantInteger32Op::verifyInvariantsImpl() {
  auto tblgen_inputAttr = getProperties().inputAttr; (void)tblgen_inputAttr;
  if (!tblgen_inputAttr) return emitOpError("requires attribute 'inputAttr'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops2(*this, tblgen_inputAttr, "inputAttr")))
    return ::mlir::failure();
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

::llvm::LogicalResult ConstantInteger32Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConstantInteger32Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr inputAttrAttr;
  ::mlir::Type outputValueRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputValueTypes(&outputValueRawType, 1);

  if (parser.parseCustomAttributeWithFallback(inputAttrAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (inputAttrAttr) result.getOrAddProperties<ConstantInteger32Op::Properties>().inputAttr = inputAttrAttr;
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  result.addTypes(outputValueTypes);
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void ConstantInteger32Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getInputAttrAttr());
  _odsPrinter << " :";
  _odsPrinter << ' ';
  {
    auto type = getOutputValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  static constexpr ::llvm::StringRef elidedAttrs[] = {"inputAttr"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
}

void ConstantInteger32Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void ConstantInteger32Op::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"inputAttr"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantInteger32Op)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::ConstantInteger32OpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantInteger64Op definitions
//===----------------------------------------------------------------------===//

namespace detail {

ConstantInteger64OpGenericAdaptorBase::ConstantInteger64OpGenericAdaptorBase(ConstantInteger64Op op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint64_t ConstantInteger64OpGenericAdaptorBase::getInputAttr() {
  auto attr = getInputAttrAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
ConstantInteger64OpAdaptor::ConstantInteger64OpAdaptor(ConstantInteger64Op op) : ConstantInteger64OpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConstantInteger64OpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_inputAttr = getProperties().inputAttr; (void)tblgen_inputAttr;
  if (!tblgen_inputAttr) return emitError(loc, "'Aie.constant_integer64_op' op requires attribute 'inputAttr'");

  if (tblgen_inputAttr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_inputAttr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_inputAttr).getType().isSignlessInteger(64)))))
    return emitError(loc, "'Aie.constant_integer64_op' op attribute 'inputAttr' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ConstantInteger64Op::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("inputAttr");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.inputAttr, attr, "inputAttr", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute ConstantInteger64Op::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "inputAttr", prop.inputAttr);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConstantInteger64Op::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.inputAttr.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConstantInteger64Op::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "inputAttr")
      return prop.inputAttr;
  return std::nullopt;
}

void ConstantInteger64Op::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "inputAttr") {
       prop.inputAttr = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.inputAttr)>>(value);
       return;
    }
}

void ConstantInteger64Op::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.inputAttr) {
      ::mlir::Attribute value = prop.inputAttr;
      ::mlir::Attribute originalValue = value;
      visitor("inputAttr", value);
      if (value != originalValue)
        setInherentAttr(prop, "inputAttr", value);
    }
}

::llvm::LogicalResult ConstantInteger64Op::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getInputAttrAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops3(attr, "inputAttr", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ConstantInteger64Op::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.inputAttr)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConstantInteger64Op::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.inputAttr);
}

uint64_t ConstantInteger64Op::getInputAttr() {
  auto attr = getInputAttrAttr();
  return attr.getValue().getZExtValue();
}

void ConstantInteger64Op::setInputAttr(uint64_t attrValue) {
  getProperties().inputAttr = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue);
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, IntegerAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, IntegerAttr value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(value)>(value));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger64Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger64Op>(__res__);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::ImplicitLocOpBuilder &builder, IntegerAttr value) {
  return create(builder, builder.getLoc(), std::forward<decltype(value)>(value));
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.inputAttr = inputAttr;
  odsState.addTypes(outputValue);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(outputValue)>(outputValue), std::forward<decltype(inputAttr)>(inputAttr));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger64Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger64Op>(__res__);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr) {
  return create(builder, builder.getLoc(), std::forward<decltype(outputValue)>(outputValue), std::forward<decltype(inputAttr)>(inputAttr));
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.inputAttr = inputAttr;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(inputAttr)>(inputAttr));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger64Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger64Op>(__res__);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(inputAttr)>(inputAttr));
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, uint64_t inputAttr) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.inputAttr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), inputAttr);
  odsState.addTypes(outputValue);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type outputValue, uint64_t inputAttr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(outputValue)>(outputValue), std::forward<decltype(inputAttr)>(inputAttr));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger64Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger64Op>(__res__);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type outputValue, uint64_t inputAttr) {
  return create(builder, builder.getLoc(), std::forward<decltype(outputValue)>(outputValue), std::forward<decltype(inputAttr)>(inputAttr));
}

void ConstantInteger64Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t inputAttr) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.inputAttr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), inputAttr);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint64_t inputAttr) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(inputAttr)>(inputAttr));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger64Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger64Op>(__res__);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint64_t inputAttr) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(inputAttr)>(inputAttr));
}

void ConstantInteger64Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger64Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger64Op>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void ConstantInteger64Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantInteger64Op>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantInteger64Op>(__res__);
}

ConstantInteger64Op ConstantInteger64Op::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ConstantInteger64Op::verifyInvariantsImpl() {
  auto tblgen_inputAttr = getProperties().inputAttr; (void)tblgen_inputAttr;
  if (!tblgen_inputAttr) return emitOpError("requires attribute 'inputAttr'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops3(*this, tblgen_inputAttr, "inputAttr")))
    return ::mlir::failure();
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

::llvm::LogicalResult ConstantInteger64Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConstantInteger64Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr inputAttrAttr;
  ::mlir::Type outputValueRawType{};
  ::llvm::ArrayRef<::mlir::Type> outputValueTypes(&outputValueRawType, 1);

  if (parser.parseCustomAttributeWithFallback(inputAttrAttr, parser.getBuilder().getIntegerType(64))) {
    return ::mlir::failure();
  }
  if (inputAttrAttr) result.getOrAddProperties<ConstantInteger64Op::Properties>().inputAttr = inputAttrAttr;
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
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  result.addTypes(outputValueTypes);
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void ConstantInteger64Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getInputAttrAttr());
  _odsPrinter << " :";
  _odsPrinter << ' ';
  {
    auto type = getOutputValue().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::IntegerType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  static constexpr ::llvm::StringRef elidedAttrs[] = {"inputAttr"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
}

void ConstantInteger64Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void ConstantInteger64Op::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"inputAttr"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantInteger64Op)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::ConstantInteger64OpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

ConstantOpGenericAdaptorBase::ConstantOpGenericAdaptorBase(ConstantOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

::mlir::DenseElementsAttr ConstantOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}


} // namespace detail
ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : ConstantOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'Aie.constant' op requires attribute 'value'");

  if (tblgen_value && !((::llvm::isa<::mlir::DenseFPElementsAttr>(tblgen_value) &&::llvm::cast<::mlir::DenseElementsAttr>(tblgen_value).getType().getElementType().isF64())))
    return emitError(loc, "'Aie.constant' op attribute 'value' failed to satisfy constraint: 64-bit float elements attribute");
  return ::mlir::success();
}

::llvm::LogicalResult ConstantOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("value");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.value, attr, "value", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute ConstantOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "value", prop.value);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConstantOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConstantOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void ConstantOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void ConstantOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.value) {
      ::mlir::Attribute value = prop.value;
      ::mlir::Attribute originalValue = value;
      visitor("value", value);
      if (value != originalValue)
        setInherentAttr(prop, "value", value);
    }
}

::llvm::LogicalResult ConstantOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aieops4(attr, "value", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult ConstantOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.value)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConstantOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.value);
}

::mlir::DenseElementsAttr ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, DenseElementsAttr value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(value)>(value));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantOp>(__res__);
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, DenseElementsAttr value) {
  return create(builder, builder.getLoc(), std::forward<decltype(value)>(value));
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, double value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(value)>(value));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantOp>(__res__);
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, double value) {
  return create(builder, builder.getLoc(), std::forward<decltype(value)>(value));
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.value = value;
  odsState.addTypes(resultType0);
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultType0)>(resultType0), std::forward<decltype(value)>(value));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantOp>(__res__);
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultType0)>(resultType0), std::forward<decltype(value)>(value));
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(value)>(value));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantOp>(__res__);
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(value)>(value));
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantOp>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ConstantOp ConstantOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<ConstantOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<ConstantOp>(__res__);
}

ConstantOp ConstantOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aieops4(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aieops17(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult ConstantOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::llvm::LogicalResult ConstantOp::setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }
  // keep track of used keys in the input dictionary to be able to error out
  // if there are some unknown ones.
  ::mlir::DenseSet<::mlir::StringAttr> usedKeys;
  ::mlir::MLIRContext *ctx = dict.getContext();
  (void)ctx;
  {

    auto &propStorage = prop.value;
    auto valueAttrName = ::mlir::StringAttr::get(ctx, "value");
    auto attr = dict.get(valueAttrName);
    usedKeys.insert(valueAttrName);
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        emitError() << "expected key entry for value in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        emitError() << "Invalid attribute `value` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  for (::mlir::NamedAttribute attr : dict) {
    if (!usedKeys.contains(attr.getName()))
      return emitError() << "unknown key '" << attr.getName() <<
          "' when parsing properties dictionary";
  }
  return ::mlir::success();
}

void ConstantOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

void ConstantOp::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"value"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::detail::ConstantOpGenericAdaptorBase::Properties)


#endif // GET_OP_CLASSES

