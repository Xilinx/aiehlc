/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: aielinkops.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::aiel {

/// alloc mem bank
/// "This is a operation to alloc mem bank op."
class AllocMemBankOp;

} // namespace mlir::aiel
namespace mlir::aiel {

/// alloc mem buffer
/// "This is a operation to alloc mem bank op."
class AllocMemBuffer;

} // namespace mlir::aiel
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

namespace mlir::aiel {


//===----------------------------------------------------------------------===//
// ::mlir::aiel::AllocMemBankOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AllocMemBankOpGenericAdaptorBase {
public:
  struct Properties {
    using baseaddrTy = ::mlir::IntegerAttr;
    baseaddrTy baseaddr;

    auto getBaseaddr() const {
      auto &propStorage = this->baseaddr;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setBaseaddr(const ::mlir::IntegerAttr &propValue) {
      this->baseaddr = propValue;
    }
    using islocalTy = ::mlir::IntegerAttr;
    islocalTy islocal;

    auto getIslocal() const {
      auto &propStorage = this->islocal;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setIslocal(const ::mlir::IntegerAttr &propValue) {
      this->islocal = propValue;
    }
    using locationdirectTy = ::mlir::IntegerAttr;
    locationdirectTy locationdirect;

    auto getLocationdirect() const {
      auto &propStorage = this->locationdirect;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setLocationdirect(const ::mlir::IntegerAttr &propValue) {
      this->locationdirect = propValue;
    }
    using locationtextTy = ::mlir::StringAttr;
    locationtextTy locationtext;

    auto getLocationtext() const {
      auto &propStorage = this->locationtext;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setLocationtext(const ::mlir::StringAttr &propValue) {
      this->locationtext = propValue;
    }
    using memtotalsizeTy = ::mlir::IntegerAttr;
    memtotalsizeTy memtotalsize;

    auto getMemtotalsize() const {
      auto &propStorage = this->memtotalsize;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setMemtotalsize(const ::mlir::IntegerAttr &propValue) {
      this->memtotalsize = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.baseaddr == this->baseaddr &&
        rhs.islocal == this->islocal &&
        rhs.locationdirect == this->locationdirect &&
        rhs.locationtext == this->locationtext &&
        rhs.memtotalsize == this->memtotalsize &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AllocMemBankOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("AieLink.allocmembank", odsAttrs.getContext());
  }

  AllocMemBankOpGenericAdaptorBase(AllocMemBankOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getLocationdirectAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().locationdirect);
    return attr;
  }

  uint32_t getLocationdirect();
  ::mlir::StringAttr getLocationtextAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().locationtext);
    return attr;
  }

  ::llvm::StringRef getLocationtext();
  ::mlir::IntegerAttr getBaseaddrAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().baseaddr);
    return attr;
  }

  uint32_t getBaseaddr();
  ::mlir::IntegerAttr getMemtotalsizeAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().memtotalsize);
    return attr;
  }

  uint32_t getMemtotalsize();
  ::mlir::IntegerAttr getIslocalAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().islocal);
    return attr;
  }

  uint32_t getIslocal();
};

} // namespace detail
template <typename RangeT>
class AllocMemBankOpGenericAdaptor : public detail::AllocMemBankOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocMemBankOpGenericAdaptorBase;
public:
  AllocMemBankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AllocMemBankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : AllocMemBankOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AllocMemBankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AllocMemBankOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  AllocMemBankOpGenericAdaptor(RangeT values, const AllocMemBankOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AllocMemBankOp, typename = std::enable_if_t<std::is_same_v<LateInst, AllocMemBankOp>>>
  AllocMemBankOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllocMemBankOpAdaptor : public AllocMemBankOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllocMemBankOpGenericAdaptor::AllocMemBankOpGenericAdaptor;
  AllocMemBankOpAdaptor(AllocMemBankOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AllocMemBankOp : public ::mlir::Op<AllocMemBankOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aiel::Aie_L1MemBankType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocMemBankOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocMemBankOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("baseaddr"), ::llvm::StringRef("islocal"), ::llvm::StringRef("locationdirect"), ::llvm::StringRef("locationtext"), ::llvm::StringRef("memtotalsize")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getBaseaddrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBaseaddrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIslocalAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIslocalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLocationdirectAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLocationdirectAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getLocationtextAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getLocationtextAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getMemtotalsizeAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getMemtotalsizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("AieLink.allocmembank");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::aiel::Aie_L1MemBankType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aiel::Aie_L1MemBankType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getLocationdirectAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().locationdirect);
  }

  uint32_t getLocationdirect();
  ::mlir::StringAttr getLocationtextAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().locationtext);
  }

  ::llvm::StringRef getLocationtext();
  ::mlir::IntegerAttr getBaseaddrAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().baseaddr);
  }

  uint32_t getBaseaddr();
  ::mlir::IntegerAttr getMemtotalsizeAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().memtotalsize);
  }

  uint32_t getMemtotalsize();
  ::mlir::IntegerAttr getIslocalAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().islocal);
  }

  uint32_t getIslocal();
  void setLocationdirectAttr(::mlir::IntegerAttr attr) {
    getProperties().locationdirect = attr;
  }

  void setLocationdirect(uint32_t attrValue);
  void setLocationtextAttr(::mlir::StringAttr attr) {
    getProperties().locationtext = attr;
  }

  void setLocationtext(::llvm::StringRef attrValue);
  void setBaseaddrAttr(::mlir::IntegerAttr attr) {
    getProperties().baseaddr = attr;
  }

  void setBaseaddr(uint32_t attrValue);
  void setMemtotalsizeAttr(::mlir::IntegerAttr attr) {
    getProperties().memtotalsize = attr;
  }

  void setMemtotalsize(uint32_t attrValue);
  void setIslocalAttr(::mlir::IntegerAttr attr) {
    getProperties().islocal = attr;
  }

  void setIslocal(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal);
  static AllocMemBankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal);
  static AllocMemBankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal);
  static AllocMemBankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal);
  static AllocMemBankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal);
  static AllocMemBankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal);
  static AllocMemBankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal);
  static AllocMemBankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal);
  static AllocMemBankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static AllocMemBankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static AllocMemBankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllocMemBankOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllocMemBankOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aiel
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::AllocMemBankOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::detail::AllocMemBankOpGenericAdaptorBase::Properties)

namespace mlir::aiel {


//===----------------------------------------------------------------------===//
// ::mlir::aiel::AllocMemBuffer declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AllocMemBufferGenericAdaptorBase {
public:
  struct Properties {
    using memaddrTy = ::mlir::IntegerAttr;
    memaddrTy memaddr;

    auto getMemaddr() const {
      auto &propStorage = this->memaddr;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setMemaddr(const ::mlir::IntegerAttr &propValue) {
      this->memaddr = propValue;
    }
    using memsizeTy = ::mlir::IntegerAttr;
    memsizeTy memsize;

    auto getMemsize() const {
      auto &propStorage = this->memsize;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setMemsize(const ::mlir::IntegerAttr &propValue) {
      this->memsize = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.memaddr == this->memaddr &&
        rhs.memsize == this->memsize &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AllocMemBufferGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("AieLink.allocmembuffer", odsAttrs.getContext());
  }

  AllocMemBufferGenericAdaptorBase(AllocMemBuffer op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getMemaddrAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().memaddr);
    return attr;
  }

  uint32_t getMemaddr();
  ::mlir::IntegerAttr getMemsizeAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().memsize);
    return attr;
  }

  uint32_t getMemsize();
};

} // namespace detail
template <typename RangeT>
class AllocMemBufferGenericAdaptor : public detail::AllocMemBufferGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocMemBufferGenericAdaptorBase;
  static constexpr int odsIndex_banktype = 0;
public:
  AllocMemBufferGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AllocMemBufferGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : AllocMemBufferGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AllocMemBufferGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AllocMemBufferGenericAdaptor(values, attrs, Properties{}, {}) {}

  AllocMemBufferGenericAdaptor(RangeT values, const AllocMemBufferGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AllocMemBuffer, typename = std::enable_if_t<std::is_same_v<LateInst, AllocMemBuffer>>>
  AllocMemBufferGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getBanktype() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllocMemBufferAdaptor : public AllocMemBufferGenericAdaptor<::mlir::ValueRange> {
public:
  using AllocMemBufferGenericAdaptor::AllocMemBufferGenericAdaptor;
  AllocMemBufferAdaptor(AllocMemBuffer op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AllocMemBuffer : public ::mlir::Op<AllocMemBuffer, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aiel::Aie_MemBufferType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocMemBufferAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocMemBufferGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_banktype = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("memaddr"), ::llvm::StringRef("memsize")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMemaddrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMemaddrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getMemsizeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getMemsizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("AieLink.allocmembuffer");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aiel::Aie_L1MemBankType> getBanktype() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aiel::Aie_L1MemBankType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getBanktypeMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::aiel::Aie_MemBufferType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aiel::Aie_MemBufferType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::IntegerAttr getMemaddrAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().memaddr);
  }

  uint32_t getMemaddr();
  ::mlir::IntegerAttr getMemsizeAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().memsize);
  }

  uint32_t getMemsize();
  void setMemaddrAttr(::mlir::IntegerAttr attr) {
    getProperties().memaddr = attr;
  }

  void setMemaddr(uint32_t attrValue);
  void setMemsizeAttr(::mlir::IntegerAttr attr) {
    getProperties().memsize = attr;
  }

  void setMemsize(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize);
  static AllocMemBuffer create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize);
  static AllocMemBuffer create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize);
  static AllocMemBuffer create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize);
  static AllocMemBuffer create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize);
  static AllocMemBuffer create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize);
  static AllocMemBuffer create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize);
  static AllocMemBuffer create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize);
  static AllocMemBuffer create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static AllocMemBuffer create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static AllocMemBuffer create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllocMemBuffer create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllocMemBuffer create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aiel
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::AllocMemBuffer)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::detail::AllocMemBufferGenericAdaptorBase::Properties)


#endif // GET_OP_CLASSES

