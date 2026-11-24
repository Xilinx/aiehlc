/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: aielinkops.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::aiel::AllocMemBankOp,
::mlir::aiel::AllocMemBuffer
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace aiel {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aielinkops0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aiel::Aie_L1MemBankType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Mem bank, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_aielinkops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aiel::Aie_MemBufferType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be The Mem Buffer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aielinkops0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aielinkops0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aielinkops0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aielinkops1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_aielinkops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aielinkops1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
} // namespace aiel
} // namespace mlir
namespace mlir {
namespace aiel {

//===----------------------------------------------------------------------===//
// ::mlir::aiel::AllocMemBankOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::IntegerAttr AllocMemBankOpGenericAdaptorBase::getLocationdirectAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 2, AllocMemBankOp::getLocationdirectAttrName(*odsOpName)));
  return attr;
}

uint32_t AllocMemBankOpGenericAdaptorBase::getLocationdirect() {
  auto attr = getLocationdirectAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr AllocMemBankOpGenericAdaptorBase::getLocationtextAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 1, AllocMemBankOp::getLocationtextAttrName(*odsOpName)));
  return attr;
}

::llvm::StringRef AllocMemBankOpGenericAdaptorBase::getLocationtext() {
  auto attr = getLocationtextAttr();
  return attr.getValue();
}

::mlir::IntegerAttr AllocMemBankOpGenericAdaptorBase::getBaseaddrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 4, AllocMemBankOp::getBaseaddrAttrName(*odsOpName)));
  return attr;
}

uint32_t AllocMemBankOpGenericAdaptorBase::getBaseaddr() {
  auto attr = getBaseaddrAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr AllocMemBankOpGenericAdaptorBase::getMemtotalsizeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 0, AllocMemBankOp::getMemtotalsizeAttrName(*odsOpName)));
  return attr;
}

uint32_t AllocMemBankOpGenericAdaptorBase::getMemtotalsize() {
  auto attr = getMemtotalsizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr AllocMemBankOpGenericAdaptorBase::getIslocalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 3, AllocMemBankOp::getIslocalAttrName(*odsOpName)));
  return attr;
}

uint32_t AllocMemBankOpGenericAdaptorBase::getIslocal() {
  auto attr = getIslocalAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
AllocMemBankOpAdaptor::AllocMemBankOpAdaptor(AllocMemBankOp op) : AllocMemBankOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult AllocMemBankOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_baseaddr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'AieLink.allocmembank' op ""requires attribute 'baseaddr'");
    if (namedAttrIt->getName() == AllocMemBankOp::getBaseaddrAttrName(*odsOpName)) {
      tblgen_baseaddr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_islocal;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'AieLink.allocmembank' op ""requires attribute 'islocal'");
    if (namedAttrIt->getName() == AllocMemBankOp::getIslocalAttrName(*odsOpName)) {
      tblgen_islocal = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_locationdirect;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'AieLink.allocmembank' op ""requires attribute 'locationdirect'");
    if (namedAttrIt->getName() == AllocMemBankOp::getLocationdirectAttrName(*odsOpName)) {
      tblgen_locationdirect = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_locationtext;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'AieLink.allocmembank' op ""requires attribute 'locationtext'");
    if (namedAttrIt->getName() == AllocMemBankOp::getLocationtextAttrName(*odsOpName)) {
      tblgen_locationtext = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_memtotalsize;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'AieLink.allocmembank' op ""requires attribute 'memtotalsize'");
    if (namedAttrIt->getName() == AllocMemBankOp::getMemtotalsizeAttrName(*odsOpName)) {
      tblgen_memtotalsize = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_locationdirect && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_locationdirect))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_locationdirect).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembank' op ""attribute 'locationdirect' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_locationtext && !((::llvm::isa<::mlir::StringAttr>(tblgen_locationtext))))
    return emitError(loc, "'AieLink.allocmembank' op ""attribute 'locationtext' failed to satisfy constraint: string attribute");

  if (tblgen_baseaddr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_baseaddr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_baseaddr).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembank' op ""attribute 'baseaddr' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_memtotalsize && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_memtotalsize))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_memtotalsize).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembank' op ""attribute 'memtotalsize' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_islocal && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_islocal))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_islocal).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembank' op ""attribute 'islocal' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

uint32_t AllocMemBankOp::getLocationdirect() {
  auto attr = getLocationdirectAttr();
  return attr.getValue().getZExtValue();
}

::llvm::StringRef AllocMemBankOp::getLocationtext() {
  auto attr = getLocationtextAttr();
  return attr.getValue();
}

uint32_t AllocMemBankOp::getBaseaddr() {
  auto attr = getBaseaddrAttr();
  return attr.getValue().getZExtValue();
}

uint32_t AllocMemBankOp::getMemtotalsize() {
  auto attr = getMemtotalsizeAttr();
  return attr.getValue().getZExtValue();
}

uint32_t AllocMemBankOp::getIslocal() {
  auto attr = getIslocalAttr();
  return attr.getValue().getZExtValue();
}

void AllocMemBankOp::setLocationdirect(uint32_t attrValue) {
  (*this)->setAttr(getLocationdirectAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void AllocMemBankOp::setLocationtext(::llvm::StringRef attrValue) {
  (*this)->setAttr(getLocationtextAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void AllocMemBankOp::setBaseaddr(uint32_t attrValue) {
  (*this)->setAttr(getBaseaddrAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void AllocMemBankOp::setMemtotalsize(uint32_t attrValue) {
  (*this)->setAttr(getMemtotalsizeAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void AllocMemBankOp::setIslocal(uint32_t attrValue) {
  (*this)->setAttr(getIslocalAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void AllocMemBankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal) {
  odsState.addAttribute(getLocationdirectAttrName(odsState.name), locationdirect);
  odsState.addAttribute(getLocationtextAttrName(odsState.name), locationtext);
  odsState.addAttribute(getBaseaddrAttrName(odsState.name), baseaddr);
  odsState.addAttribute(getMemtotalsizeAttrName(odsState.name), memtotalsize);
  odsState.addAttribute(getIslocalAttrName(odsState.name), islocal);
  odsState.addTypes(result);
}

void AllocMemBankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal) {
  odsState.addAttribute(getLocationdirectAttrName(odsState.name), locationdirect);
  odsState.addAttribute(getLocationtextAttrName(odsState.name), locationtext);
  odsState.addAttribute(getBaseaddrAttrName(odsState.name), baseaddr);
  odsState.addAttribute(getMemtotalsizeAttrName(odsState.name), memtotalsize);
  odsState.addAttribute(getIslocalAttrName(odsState.name), islocal);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocMemBankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal) {
  odsState.addAttribute(getLocationdirectAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), locationdirect));
  odsState.addAttribute(getLocationtextAttrName(odsState.name), odsBuilder.getStringAttr(locationtext));
  odsState.addAttribute(getBaseaddrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), baseaddr));
  odsState.addAttribute(getMemtotalsizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memtotalsize));
  odsState.addAttribute(getIslocalAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), islocal));
  odsState.addTypes(result);
}

void AllocMemBankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal) {
  odsState.addAttribute(getLocationdirectAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), locationdirect));
  odsState.addAttribute(getLocationtextAttrName(odsState.name), odsBuilder.getStringAttr(locationtext));
  odsState.addAttribute(getBaseaddrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), baseaddr));
  odsState.addAttribute(getMemtotalsizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memtotalsize));
  odsState.addAttribute(getIslocalAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), islocal));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocMemBankOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocMemBankOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_baseaddr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'baseaddr'");
    if (namedAttrIt->getName() == getBaseaddrAttrName()) {
      tblgen_baseaddr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_islocal;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'islocal'");
    if (namedAttrIt->getName() == getIslocalAttrName()) {
      tblgen_islocal = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_locationdirect;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'locationdirect'");
    if (namedAttrIt->getName() == getLocationdirectAttrName()) {
      tblgen_locationdirect = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_locationtext;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'locationtext'");
    if (namedAttrIt->getName() == getLocationtextAttrName()) {
      tblgen_locationtext = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_memtotalsize;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'memtotalsize'");
    if (namedAttrIt->getName() == getMemtotalsizeAttrName()) {
      tblgen_memtotalsize = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops0(*this, tblgen_locationdirect, "locationdirect")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(*this, tblgen_locationtext, "locationtext")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops0(*this, tblgen_baseaddr, "baseaddr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops0(*this, tblgen_memtotalsize, "memtotalsize")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops0(*this, tblgen_islocal, "islocal")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aielinkops0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocMemBankOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AllocMemBankOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr locationdirectAttr;
  ::mlir::StringAttr locationtextAttr;
  ::mlir::IntegerAttr baseaddrAttr;
  ::mlir::IntegerAttr memtotalsizeAttr;
  ::mlir::IntegerAttr islocalAttr;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(locationdirectAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (locationdirectAttr) result.attributes.append("locationdirect", locationdirectAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(locationtextAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (locationtextAttr) result.attributes.append("locationtext", locationtextAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(baseaddrAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (baseaddrAttr) result.attributes.append("baseaddr", baseaddrAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(memtotalsizeAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (memtotalsizeAttr) result.attributes.append("memtotalsize", memtotalsizeAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(islocalAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (islocalAttr) result.attributes.append("islocal", islocalAttr);
  if (parser.parseComma())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

void AllocMemBankOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getLocationdirectAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getLocationtextAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getBaseaddrAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMemtotalsizeAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIslocalAttr());
  _odsPrinter << ",";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("locationdirect");
  elidedAttrs.push_back("locationtext");
  elidedAttrs.push_back("baseaddr");
  elidedAttrs.push_back("memtotalsize");
  elidedAttrs.push_back("islocal");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace aiel
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::AllocMemBankOp)

namespace mlir {
namespace aiel {

//===----------------------------------------------------------------------===//
// ::mlir::aiel::AllocMemBuffer definitions
//===----------------------------------------------------------------------===//

namespace detail {
::mlir::IntegerAttr AllocMemBufferGenericAdaptorBase::getMemaddrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AllocMemBuffer::getMemaddrAttrName(*odsOpName)));
  return attr;
}

uint32_t AllocMemBufferGenericAdaptorBase::getMemaddr() {
  auto attr = getMemaddrAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr AllocMemBufferGenericAdaptorBase::getMemsizeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AllocMemBuffer::getMemsizeAttrName(*odsOpName)));
  return attr;
}

uint32_t AllocMemBufferGenericAdaptorBase::getMemsize() {
  auto attr = getMemsizeAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
AllocMemBufferAdaptor::AllocMemBufferAdaptor(AllocMemBuffer op) : AllocMemBufferGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult AllocMemBufferAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_memaddr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'AieLink.allocmembuffer' op ""requires attribute 'memaddr'");
    if (namedAttrIt->getName() == AllocMemBuffer::getMemaddrAttrName(*odsOpName)) {
      tblgen_memaddr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_memsize;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'AieLink.allocmembuffer' op ""requires attribute 'memsize'");
    if (namedAttrIt->getName() == AllocMemBuffer::getMemsizeAttrName(*odsOpName)) {
      tblgen_memsize = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_memaddr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_memaddr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_memaddr).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembuffer' op ""attribute 'memaddr' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_memsize && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_memsize))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_memsize).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembuffer' op ""attribute 'memsize' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

uint32_t AllocMemBuffer::getMemaddr() {
  auto attr = getMemaddrAttr();
  return attr.getValue().getZExtValue();
}

uint32_t AllocMemBuffer::getMemsize() {
  auto attr = getMemsizeAttr();
  return attr.getValue().getZExtValue();
}

void AllocMemBuffer::setMemaddr(uint32_t attrValue) {
  (*this)->setAttr(getMemaddrAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void AllocMemBuffer::setMemsize(uint32_t attrValue) {
  (*this)->setAttr(getMemsizeAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void AllocMemBuffer::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize) {
  odsState.addOperands(banktype);
  odsState.addAttribute(getMemaddrAttrName(odsState.name), memaddr);
  odsState.addAttribute(getMemsizeAttrName(odsState.name), memsize);
  odsState.addTypes(result);
}

void AllocMemBuffer::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize) {
  odsState.addOperands(banktype);
  odsState.addAttribute(getMemaddrAttrName(odsState.name), memaddr);
  odsState.addAttribute(getMemsizeAttrName(odsState.name), memsize);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocMemBuffer::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize) {
  odsState.addOperands(banktype);
  odsState.addAttribute(getMemaddrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memaddr));
  odsState.addAttribute(getMemsizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memsize));
  odsState.addTypes(result);
}

void AllocMemBuffer::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize) {
  odsState.addOperands(banktype);
  odsState.addAttribute(getMemaddrAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memaddr));
  odsState.addAttribute(getMemsizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memsize));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocMemBuffer::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocMemBuffer::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_memaddr;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'memaddr'");
    if (namedAttrIt->getName() == getMemaddrAttrName()) {
      tblgen_memaddr = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_memsize;

  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'memsize'");
    if (namedAttrIt->getName() == getMemsizeAttrName()) {
      tblgen_memsize = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops0(*this, tblgen_memaddr, "memaddr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops0(*this, tblgen_memsize, "memsize")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aielinkops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aielinkops1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocMemBuffer::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AllocMemBuffer::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand banktypeRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> banktypeOperands(&banktypeRawOperand, 1);  ::llvm::SMLoc banktypeOperandsLoc;
  (void)banktypeOperandsLoc;
  ::mlir::IntegerAttr memaddrAttr;
  ::mlir::IntegerAttr memsizeAttr;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  banktypeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(banktypeRawOperand))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(memaddrAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (memaddrAttr) result.attributes.append("memaddr", memaddrAttr);
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(memsizeAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (memsizeAttr) result.attributes.append("memsize", memsizeAttr);
  if (parser.parseComma())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(banktypeOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AllocMemBuffer::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBanktype();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMemaddrAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMemsizeAttr());
  _odsPrinter << ",";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("memaddr");
  elidedAttrs.push_back("memsize");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace aiel
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::AllocMemBuffer)


#endif  // GET_OP_CLASSES

