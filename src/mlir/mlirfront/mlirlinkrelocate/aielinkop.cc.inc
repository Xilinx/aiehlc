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


static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aielinkops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aiel::Aie_L1MemBankType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Mem bank, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_type_constraint_aielinkops2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::aiel::Aie_MemBufferType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be The Mem Buffer, but got " << type;
  }
  return ::mlir::success();
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aielinkops1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !(((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aielinkops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aielinkops1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aielinkops2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return emitError() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::llvm::LogicalResult __mlir_ods_local_attr_constraint_aielinkops2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_aielinkops2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}
namespace mlir::aiel {


//===----------------------------------------------------------------------===//
// ::mlir::aiel::AllocMemBankOp definitions
//===----------------------------------------------------------------------===//

namespace detail {

AllocMemBankOpGenericAdaptorBase::AllocMemBankOpGenericAdaptorBase(AllocMemBankOp op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t AllocMemBankOpGenericAdaptorBase::getLocationdirect() {
  auto attr = getLocationdirectAttr();
  return attr.getValue().getZExtValue();
}

::llvm::StringRef AllocMemBankOpGenericAdaptorBase::getLocationtext() {
  auto attr = getLocationtextAttr();
  return attr.getValue();
}

uint32_t AllocMemBankOpGenericAdaptorBase::getBaseaddr() {
  auto attr = getBaseaddrAttr();
  return attr.getValue().getZExtValue();
}

uint32_t AllocMemBankOpGenericAdaptorBase::getMemtotalsize() {
  auto attr = getMemtotalsizeAttr();
  return attr.getValue().getZExtValue();
}

uint32_t AllocMemBankOpGenericAdaptorBase::getIslocal() {
  auto attr = getIslocalAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
AllocMemBankOpAdaptor::AllocMemBankOpAdaptor(AllocMemBankOp op) : AllocMemBankOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AllocMemBankOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_baseaddr = getProperties().baseaddr; (void)tblgen_baseaddr;
  if (!tblgen_baseaddr) return emitError(loc, "'AieLink.allocmembank' op requires attribute 'baseaddr'");
  auto tblgen_islocal = getProperties().islocal; (void)tblgen_islocal;
  if (!tblgen_islocal) return emitError(loc, "'AieLink.allocmembank' op requires attribute 'islocal'");
  auto tblgen_locationdirect = getProperties().locationdirect; (void)tblgen_locationdirect;
  if (!tblgen_locationdirect) return emitError(loc, "'AieLink.allocmembank' op requires attribute 'locationdirect'");
  auto tblgen_locationtext = getProperties().locationtext; (void)tblgen_locationtext;
  if (!tblgen_locationtext) return emitError(loc, "'AieLink.allocmembank' op requires attribute 'locationtext'");
  auto tblgen_memtotalsize = getProperties().memtotalsize; (void)tblgen_memtotalsize;
  if (!tblgen_memtotalsize) return emitError(loc, "'AieLink.allocmembank' op requires attribute 'memtotalsize'");

  if (tblgen_locationdirect && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_locationdirect))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_locationdirect).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembank' op attribute 'locationdirect' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_locationtext && !((::llvm::isa<::mlir::StringAttr>(tblgen_locationtext))))
    return emitError(loc, "'AieLink.allocmembank' op attribute 'locationtext' failed to satisfy constraint: string attribute");

  if (tblgen_baseaddr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_baseaddr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_baseaddr).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembank' op attribute 'baseaddr' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_memtotalsize && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_memtotalsize))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_memtotalsize).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembank' op attribute 'memtotalsize' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_islocal && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_islocal))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_islocal).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembank' op attribute 'islocal' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult AllocMemBankOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("baseaddr");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.baseaddr, attr, "baseaddr", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("islocal");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.islocal, attr, "islocal", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("locationdirect");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.locationdirect, attr, "locationdirect", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("locationtext");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.locationtext, attr, "locationtext", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("memtotalsize");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.memtotalsize, attr, "memtotalsize", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute AllocMemBankOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "baseaddr", prop.baseaddr);
    ::mlir::detail::appendAttributeProperty(attrs, "islocal", prop.islocal);
    ::mlir::detail::appendAttributeProperty(attrs, "locationdirect", prop.locationdirect);
    ::mlir::detail::appendAttributeProperty(attrs, "locationtext", prop.locationtext);
    ::mlir::detail::appendAttributeProperty(attrs, "memtotalsize", prop.memtotalsize);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AllocMemBankOp::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.baseaddr.getAsOpaquePointer()), 
    llvm::hash_value(prop.islocal.getAsOpaquePointer()), 
    llvm::hash_value(prop.locationdirect.getAsOpaquePointer()), 
    llvm::hash_value(prop.locationtext.getAsOpaquePointer()), 
    llvm::hash_value(prop.memtotalsize.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AllocMemBankOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "baseaddr")
      return prop.baseaddr;

    if (name == "islocal")
      return prop.islocal;

    if (name == "locationdirect")
      return prop.locationdirect;

    if (name == "locationtext")
      return prop.locationtext;

    if (name == "memtotalsize")
      return prop.memtotalsize;
  return std::nullopt;
}

void AllocMemBankOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "baseaddr") {
       prop.baseaddr = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.baseaddr)>>(value);
       return;
    }

    if (name == "islocal") {
       prop.islocal = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.islocal)>>(value);
       return;
    }

    if (name == "locationdirect") {
       prop.locationdirect = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.locationdirect)>>(value);
       return;
    }

    if (name == "locationtext") {
       prop.locationtext = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.locationtext)>>(value);
       return;
    }

    if (name == "memtotalsize") {
       prop.memtotalsize = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.memtotalsize)>>(value);
       return;
    }
}

void AllocMemBankOp::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.baseaddr) {
      ::mlir::Attribute value = prop.baseaddr;
      ::mlir::Attribute originalValue = value;
      visitor("baseaddr", value);
      if (value != originalValue)
        setInherentAttr(prop, "baseaddr", value);
    }

    if (prop.islocal) {
      ::mlir::Attribute value = prop.islocal;
      ::mlir::Attribute originalValue = value;
      visitor("islocal", value);
      if (value != originalValue)
        setInherentAttr(prop, "islocal", value);
    }

    if (prop.locationdirect) {
      ::mlir::Attribute value = prop.locationdirect;
      ::mlir::Attribute originalValue = value;
      visitor("locationdirect", value);
      if (value != originalValue)
        setInherentAttr(prop, "locationdirect", value);
    }

    if (prop.locationtext) {
      ::mlir::Attribute value = prop.locationtext;
      ::mlir::Attribute originalValue = value;
      visitor("locationtext", value);
      if (value != originalValue)
        setInherentAttr(prop, "locationtext", value);
    }

    if (prop.memtotalsize) {
      ::mlir::Attribute value = prop.memtotalsize;
      ::mlir::Attribute originalValue = value;
      visitor("memtotalsize", value);
      if (value != originalValue)
        setInherentAttr(prop, "memtotalsize", value);
    }
}

::llvm::LogicalResult AllocMemBankOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getBaseaddrAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(attr, "baseaddr", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getIslocalAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(attr, "islocal", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getLocationdirectAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(attr, "locationdirect", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getLocationtextAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops2(attr, "locationtext", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMemtotalsizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(attr, "memtotalsize", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AllocMemBankOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.baseaddr)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.islocal)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.locationdirect)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.locationtext)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.memtotalsize)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AllocMemBankOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.baseaddr);
  writer.writeAttribute(prop.islocal);
  writer.writeAttribute(prop.locationdirect);
  writer.writeAttribute(prop.locationtext);
  writer.writeAttribute(prop.memtotalsize);
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
  getProperties().locationdirect = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void AllocMemBankOp::setLocationtext(::llvm::StringRef attrValue) {
  getProperties().locationtext = ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue);
}

void AllocMemBankOp::setBaseaddr(uint32_t attrValue) {
  getProperties().baseaddr = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void AllocMemBankOp::setMemtotalsize(uint32_t attrValue) {
  getProperties().memtotalsize = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void AllocMemBankOp::setIslocal(uint32_t attrValue) {
  getProperties().islocal = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void AllocMemBankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.locationdirect = locationdirect;
  odsProperties.locationtext = locationtext;
  odsProperties.baseaddr = baseaddr;
  odsProperties.memtotalsize = memtotalsize;
  odsProperties.islocal = islocal;
  odsState.addTypes(result);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(locationdirect)>(locationdirect), std::forward<decltype(locationtext)>(locationtext), std::forward<decltype(baseaddr)>(baseaddr), std::forward<decltype(memtotalsize)>(memtotalsize), std::forward<decltype(islocal)>(islocal));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBankOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBankOp>(__res__);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(locationdirect)>(locationdirect), std::forward<decltype(locationtext)>(locationtext), std::forward<decltype(baseaddr)>(baseaddr), std::forward<decltype(memtotalsize)>(memtotalsize), std::forward<decltype(islocal)>(islocal));
}

void AllocMemBankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.locationdirect = locationdirect;
  odsProperties.locationtext = locationtext;
  odsProperties.baseaddr = baseaddr;
  odsProperties.memtotalsize = memtotalsize;
  odsProperties.islocal = islocal;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(locationdirect)>(locationdirect), std::forward<decltype(locationtext)>(locationtext), std::forward<decltype(baseaddr)>(baseaddr), std::forward<decltype(memtotalsize)>(memtotalsize), std::forward<decltype(islocal)>(islocal));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBankOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBankOp>(__res__);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(locationdirect)>(locationdirect), std::forward<decltype(locationtext)>(locationtext), std::forward<decltype(baseaddr)>(baseaddr), std::forward<decltype(memtotalsize)>(memtotalsize), std::forward<decltype(islocal)>(islocal));
}

void AllocMemBankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.locationdirect = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), locationdirect);
  odsProperties.locationtext = odsBuilder.getStringAttr(locationtext);
  odsProperties.baseaddr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), baseaddr);
  odsProperties.memtotalsize = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memtotalsize);
  odsProperties.islocal = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), islocal);
  odsState.addTypes(result);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(locationdirect)>(locationdirect), std::forward<decltype(locationtext)>(locationtext), std::forward<decltype(baseaddr)>(baseaddr), std::forward<decltype(memtotalsize)>(memtotalsize), std::forward<decltype(islocal)>(islocal));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBankOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBankOp>(__res__);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(locationdirect)>(locationdirect), std::forward<decltype(locationtext)>(locationtext), std::forward<decltype(baseaddr)>(baseaddr), std::forward<decltype(memtotalsize)>(memtotalsize), std::forward<decltype(islocal)>(islocal));
}

void AllocMemBankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsProperties.locationdirect = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), locationdirect);
  odsProperties.locationtext = odsBuilder.getStringAttr(locationtext);
  odsProperties.baseaddr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), baseaddr);
  odsProperties.memtotalsize = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memtotalsize);
  odsProperties.islocal = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), islocal);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(locationdirect)>(locationdirect), std::forward<decltype(locationtext)>(locationtext), std::forward<decltype(baseaddr)>(baseaddr), std::forward<decltype(memtotalsize)>(memtotalsize), std::forward<decltype(islocal)>(islocal));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBankOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBankOp>(__res__);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(locationdirect)>(locationdirect), std::forward<decltype(locationtext)>(locationtext), std::forward<decltype(baseaddr)>(baseaddr), std::forward<decltype(memtotalsize)>(memtotalsize), std::forward<decltype(islocal)>(islocal));
}

void AllocMemBankOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBankOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBankOp>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

AllocMemBankOp AllocMemBankOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void AllocMemBankOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBankOp>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBankOp>(__res__);
}

AllocMemBankOp AllocMemBankOp::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult AllocMemBankOp::verifyInvariantsImpl() {
  auto tblgen_baseaddr = getProperties().baseaddr; (void)tblgen_baseaddr;
  if (!tblgen_baseaddr) return emitOpError("requires attribute 'baseaddr'");
  auto tblgen_islocal = getProperties().islocal; (void)tblgen_islocal;
  if (!tblgen_islocal) return emitOpError("requires attribute 'islocal'");
  auto tblgen_locationdirect = getProperties().locationdirect; (void)tblgen_locationdirect;
  if (!tblgen_locationdirect) return emitOpError("requires attribute 'locationdirect'");
  auto tblgen_locationtext = getProperties().locationtext; (void)tblgen_locationtext;
  if (!tblgen_locationtext) return emitOpError("requires attribute 'locationtext'");
  auto tblgen_memtotalsize = getProperties().memtotalsize; (void)tblgen_memtotalsize;
  if (!tblgen_memtotalsize) return emitOpError("requires attribute 'memtotalsize'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(*this, tblgen_locationdirect, "locationdirect")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops2(*this, tblgen_locationtext, "locationtext")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(*this, tblgen_baseaddr, "baseaddr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(*this, tblgen_memtotalsize, "memtotalsize")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(*this, tblgen_islocal, "islocal")))
    return ::mlir::failure();
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

::llvm::LogicalResult AllocMemBankOp::verifyInvariants() {
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
  if (locationdirectAttr) result.getOrAddProperties<AllocMemBankOp::Properties>().locationdirect = locationdirectAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(locationtextAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (locationtextAttr) result.getOrAddProperties<AllocMemBankOp::Properties>().locationtext = locationtextAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(baseaddrAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (baseaddrAttr) result.getOrAddProperties<AllocMemBankOp::Properties>().baseaddr = baseaddrAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(memtotalsizeAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (memtotalsizeAttr) result.getOrAddProperties<AllocMemBankOp::Properties>().memtotalsize = memtotalsizeAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(islocalAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (islocalAttr) result.getOrAddProperties<AllocMemBankOp::Properties>().islocal = islocalAttr;
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

LLVM_ATTRIBUTE_MINSIZE void AllocMemBankOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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
  static constexpr ::llvm::StringRef elidedAttrs[] = {"locationdirect", "locationtext", "baseaddr", "memtotalsize", "islocal"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void AllocMemBankOp::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"locationdirect", "locationtext", "baseaddr", "memtotalsize", "islocal"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aiel
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::AllocMemBankOp)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::detail::AllocMemBankOpGenericAdaptorBase::Properties)

namespace mlir::aiel {


//===----------------------------------------------------------------------===//
// ::mlir::aiel::AllocMemBuffer definitions
//===----------------------------------------------------------------------===//

namespace detail {

AllocMemBufferGenericAdaptorBase::AllocMemBufferGenericAdaptorBase(AllocMemBuffer op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), properties(op.getProperties()), odsRegions(op->getRegions()) {}

uint32_t AllocMemBufferGenericAdaptorBase::getMemaddr() {
  auto attr = getMemaddrAttr();
  return attr.getValue().getZExtValue();
}

uint32_t AllocMemBufferGenericAdaptorBase::getMemsize() {
  auto attr = getMemsizeAttr();
  return attr.getValue().getZExtValue();
}


} // namespace detail
AllocMemBufferAdaptor::AllocMemBufferAdaptor(AllocMemBuffer op) : AllocMemBufferGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult AllocMemBufferAdaptor::verify(::mlir::Location loc) {
  auto tblgen_memaddr = getProperties().memaddr; (void)tblgen_memaddr;
  if (!tblgen_memaddr) return emitError(loc, "'AieLink.allocmembuffer' op requires attribute 'memaddr'");
  auto tblgen_memsize = getProperties().memsize; (void)tblgen_memsize;
  if (!tblgen_memsize) return emitError(loc, "'AieLink.allocmembuffer' op requires attribute 'memsize'");

  if (tblgen_memaddr && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_memaddr))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_memaddr).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembuffer' op attribute 'memaddr' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_memsize && !(((::llvm::isa<::mlir::IntegerAttr>(tblgen_memsize))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_memsize).getType().isSignlessInteger(32)))))
    return emitError(loc, "'AieLink.allocmembuffer' op attribute 'memsize' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

::llvm::LogicalResult AllocMemBuffer::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    emitError() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
       auto attr = dict.get("memaddr");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.memaddr, attr, "memaddr", emitError)))
      return ::mlir::failure();
  }

  {
       auto attr = dict.get("memsize");
    if (::mlir::failed(::mlir::detail::setAttributeProperty(
            prop.memsize, attr, "memsize", emitError)))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::Attribute AllocMemBuffer::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};
    ::mlir::detail::appendAttributeProperty(attrs, "memaddr", prop.memaddr);
    ::mlir::detail::appendAttributeProperty(attrs, "memsize", prop.memsize);

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code AllocMemBuffer::computePropertiesHash(const Properties &prop) {
  using llvm::hash_value;
  return llvm::hash_combine(
    llvm::hash_value(prop.memaddr.getAsOpaquePointer()), 
    llvm::hash_value(prop.memsize.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> AllocMemBuffer::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "memaddr")
      return prop.memaddr;

    if (name == "memsize")
      return prop.memsize;
  return std::nullopt;
}

void AllocMemBuffer::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "memaddr") {
       prop.memaddr = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.memaddr)>>(value);
       return;
    }

    if (name == "memsize") {
       prop.memsize = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.memsize)>>(value);
       return;
    }
}

void AllocMemBuffer::walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor) {
    if (prop.memaddr) {
      ::mlir::Attribute value = prop.memaddr;
      ::mlir::Attribute originalValue = value;
      visitor("memaddr", value);
      if (value != originalValue)
        setInherentAttr(prop, "memaddr", value);
    }

    if (prop.memsize) {
      ::mlir::Attribute value = prop.memsize;
      ::mlir::Attribute originalValue = value;
      visitor("memsize", value);
      if (value != originalValue)
        setInherentAttr(prop, "memsize", value);
    }
}

::llvm::LogicalResult AllocMemBuffer::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) {
    {
      ::mlir::Attribute attr = attrs.get(getMemaddrAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(attr, "memaddr", emitError)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getMemsizeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(attr, "memsize", emitError)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::llvm::LogicalResult AllocMemBuffer::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.memaddr)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.memsize)))
    return ::mlir::failure();
  return ::mlir::success();
}

void AllocMemBuffer::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.memaddr);
  writer.writeAttribute(prop.memsize);
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
  getProperties().memaddr = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void AllocMemBuffer::setMemsize(uint32_t attrValue) {
  getProperties().memsize = ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue);
}

void AllocMemBuffer::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(banktype);
  odsProperties.memaddr = memaddr;
  odsProperties.memsize = memsize;
  odsState.addTypes(result);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(banktype)>(banktype), std::forward<decltype(memaddr)>(memaddr), std::forward<decltype(memsize)>(memsize));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBuffer>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBuffer>(__res__);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(banktype)>(banktype), std::forward<decltype(memaddr)>(memaddr), std::forward<decltype(memsize)>(memsize));
}

void AllocMemBuffer::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(banktype);
  odsProperties.memaddr = memaddr;
  odsProperties.memsize = memsize;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(banktype)>(banktype), std::forward<decltype(memaddr)>(memaddr), std::forward<decltype(memsize)>(memsize));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBuffer>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBuffer>(__res__);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(banktype)>(banktype), std::forward<decltype(memaddr)>(memaddr), std::forward<decltype(memsize)>(memsize));
}

void AllocMemBuffer::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(banktype);
  odsProperties.memaddr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memaddr);
  odsProperties.memsize = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memsize);
  odsState.addTypes(result);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(result)>(result), std::forward<decltype(banktype)>(banktype), std::forward<decltype(memaddr)>(memaddr), std::forward<decltype(memsize)>(memsize));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBuffer>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBuffer>(__res__);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize) {
  return create(builder, builder.getLoc(), std::forward<decltype(result)>(result), std::forward<decltype(banktype)>(banktype), std::forward<decltype(memaddr)>(memaddr), std::forward<decltype(memsize)>(memsize));
}

void AllocMemBuffer::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize) {
  auto &odsProperties = odsState.getOrAddProperties<Properties>();
  odsState.addOperands(banktype);
  odsProperties.memaddr = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memaddr);
  odsProperties.memsize = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), memsize);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(banktype)>(banktype), std::forward<decltype(memaddr)>(memaddr), std::forward<decltype(memsize)>(memsize));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBuffer>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBuffer>(__res__);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(banktype)>(banktype), std::forward<decltype(memaddr)>(memaddr), std::forward<decltype(memsize)>(memsize));
}

void AllocMemBuffer::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  buildPropertiesAndDiscardableAttributes(odsState, attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBuffer>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBuffer>(__res__);
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

AllocMemBuffer AllocMemBuffer::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_PUSH

  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(attributes)>(attributes));
LLVM_SUPPRESS_DEPRECATED_DECLARATIONS_POP
}

void AllocMemBuffer::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.useProperties(const_cast<Properties&>(properties));
  odsState.addAttributes(discardableAttributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  ::mlir::OperationState __state__(location, getOperationName());
  build(builder, __state__, std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
  auto __res__ = builder.create(__state__);
  assert((::llvm::isa<AllocMemBuffer>(__res__)) && "builder didn't return the right type");
  return ::llvm::cast<AllocMemBuffer>(__res__);
}

AllocMemBuffer AllocMemBuffer::create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes) {
  return create(builder, builder.getLoc(), std::forward<decltype(resultTypes)>(resultTypes), std::forward<decltype(operands)>(operands), std::forward<decltype(properties)>(properties), std::forward<decltype(discardableAttributes)>(discardableAttributes));
}

::llvm::LogicalResult AllocMemBuffer::verifyInvariantsImpl() {
  auto tblgen_memaddr = getProperties().memaddr; (void)tblgen_memaddr;
  if (!tblgen_memaddr) return emitOpError("requires attribute 'memaddr'");
  auto tblgen_memsize = getProperties().memsize; (void)tblgen_memsize;
  if (!tblgen_memsize) return emitOpError("requires attribute 'memsize'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(*this, tblgen_memaddr, "memaddr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_aielinkops1(*this, tblgen_memsize, "memsize")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aielinkops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_aielinkops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::llvm::LogicalResult AllocMemBuffer::verifyInvariants() {
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
  if (memaddrAttr) result.getOrAddProperties<AllocMemBuffer::Properties>().memaddr = memaddrAttr;
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(memsizeAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (memsizeAttr) result.getOrAddProperties<AllocMemBuffer::Properties>().memsize = memsizeAttr;
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
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  // coverity[ARRAY_VS_SINGLETON]
  if (parser.resolveOperands(banktypeOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

LLVM_ATTRIBUTE_MINSIZE void AllocMemBuffer::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBanktype();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMemaddrAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMemsizeAttr());
  _odsPrinter << ",";
  static constexpr ::llvm::StringRef elidedAttrs[] = {"memaddr", "memsize"};
  ::mlir::NamedAttrList _odsAttrs((*this)->getRawDictionaryAttrs());
  (*this)->getName().walkInherentAttrs(*this, [&](::llvm::StringRef name, ::mlir::Attribute &attr) {
    _odsAttrs.append(name, attr);
  });
  _odsPrinter.printOptionalAttrDict(_odsAttrs.getDictionary((*this)->getContext()).getValue(), elidedAttrs);
  _odsPrinter << " :";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void AllocMemBuffer::buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  Properties &properties = odsState.getOrAddProperties<Properties>();
  populateDefaultProperties(odsState.name, properties);
  const ::llvm::StringRef inherentNames[] = {"memaddr", "memsize"};
  ::mlir::detail::splitPropertiesAndDiscardableAttributes(
      odsState, attributes, inherentNames, [&](::mlir::DictionaryAttr inherentAttrs) {
        return setPropertiesFromAttr(properties, inherentAttrs,
            [&]() { return ::mlir::emitError(odsState.location); });
      });
}


} // namespace mlir::aiel
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::AllocMemBuffer)
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::detail::AllocMemBufferGenericAdaptorBase::Properties)


#endif // GET_OP_CLASSES

