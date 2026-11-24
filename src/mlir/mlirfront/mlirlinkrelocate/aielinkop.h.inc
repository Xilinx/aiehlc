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

namespace mlir {
namespace aiel {
class AllocMemBankOp;
} // namespace aiel
} // namespace mlir
namespace mlir {
namespace aiel {
class AllocMemBuffer;
} // namespace aiel
} // namespace mlir
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace aiel {

//===----------------------------------------------------------------------===//
// ::mlir::aiel::AllocMemBankOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllocMemBankOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AllocMemBankOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("AieLink.allocmembank", odsAttrs.getContext());
  }

  AllocMemBankOpGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getLocationdirectAttr();
  uint32_t getLocationdirect();
  ::mlir::StringAttr getLocationtextAttr();
  ::llvm::StringRef getLocationtext();
  ::mlir::IntegerAttr getBaseaddrAttr();
  uint32_t getBaseaddr();
  ::mlir::IntegerAttr getMemtotalsizeAttr();
  uint32_t getMemtotalsize();
  ::mlir::IntegerAttr getIslocalAttr();
  uint32_t getIslocal();
};
} // namespace detail
template <typename RangeT>
class AllocMemBankOpGenericAdaptor : public detail::AllocMemBankOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocMemBankOpGenericAdaptorBase;
public:
  AllocMemBankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AllocMemBankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AllocMemBankOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

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

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllocMemBankOp : public ::mlir::Op<AllocMemBankOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aiel::Aie_L1MemBankType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocMemBankOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocMemBankOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
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

  ::mlir::IntegerAttr getLocationdirectAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 2, getLocationdirectAttrName()));
  }

  uint32_t getLocationdirect();
  ::mlir::StringAttr getLocationtextAttr() {
    return ::llvm::cast<::mlir::StringAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 1, getLocationtextAttrName()));
  }

  ::llvm::StringRef getLocationtext();
  ::mlir::IntegerAttr getBaseaddrAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 4, getBaseaddrAttrName()));
  }

  uint32_t getBaseaddr();
  ::mlir::IntegerAttr getMemtotalsizeAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 0, getMemtotalsizeAttrName()));
  }

  uint32_t getMemtotalsize();
  ::mlir::IntegerAttr getIslocalAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 3, getIslocalAttrName()));
  }

  uint32_t getIslocal();
  void setLocationdirectAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getLocationdirectAttrName(), attr);
  }

  void setLocationdirect(uint32_t attrValue);
  void setLocationtextAttr(::mlir::StringAttr attr) {
    (*this)->setAttr(getLocationtextAttrName(), attr);
  }

  void setLocationtext(::llvm::StringRef attrValue);
  void setBaseaddrAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getBaseaddrAttrName(), attr);
  }

  void setBaseaddr(uint32_t attrValue);
  void setMemtotalsizeAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getMemtotalsizeAttrName(), attr);
  }

  void setMemtotalsize(uint32_t attrValue);
  void setIslocalAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getIslocalAttrName(), attr);
  }

  void setIslocal(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr locationdirect, ::mlir::StringAttr locationtext, ::mlir::IntegerAttr baseaddr, ::mlir::IntegerAttr memtotalsize, ::mlir::IntegerAttr islocal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t locationdirect, ::llvm::StringRef locationtext, uint32_t baseaddr, uint32_t memtotalsize, uint32_t islocal);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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

public:
};
} // namespace aiel
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::AllocMemBankOp)

namespace mlir {
namespace aiel {

//===----------------------------------------------------------------------===//
// ::mlir::aiel::AllocMemBuffer declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllocMemBufferGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AllocMemBufferGenericAdaptorBase(::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("AieLink.allocmembuffer", odsAttrs.getContext());
  }

  AllocMemBufferGenericAdaptorBase(::mlir::Operation *op) : odsAttrs(op->getRawDictionaryAttrs()), odsOpName(op->getName()), odsRegions(op->getRegions()) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getMemaddrAttr();
  uint32_t getMemaddr();
  ::mlir::IntegerAttr getMemsizeAttr();
  uint32_t getMemsize();
};
} // namespace detail
template <typename RangeT>
class AllocMemBufferGenericAdaptor : public detail::AllocMemBufferGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocMemBufferGenericAdaptorBase;
public:
  AllocMemBufferGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = {}, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AllocMemBufferGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AllocMemBufferGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

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

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllocMemBuffer : public ::mlir::Op<AllocMemBuffer, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aiel::Aie_MemBufferType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocMemBufferAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocMemBufferGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
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

  ::mlir::IntegerAttr getMemaddrAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getMemaddrAttrName()));
  }

  uint32_t getMemaddr();
  ::mlir::IntegerAttr getMemsizeAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getMemsizeAttrName()));
  }

  uint32_t getMemsize();
  void setMemaddrAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getMemaddrAttrName(), attr);
  }

  void setMemaddr(uint32_t attrValue);
  void setMemsizeAttr(::mlir::IntegerAttr attr) {
    (*this)->setAttr(getMemsizeAttrName(), attr);
  }

  void setMemsize(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, ::mlir::IntegerAttr memaddr, ::mlir::IntegerAttr memsize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value banktype, uint32_t memaddr, uint32_t memsize);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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

public:
};
} // namespace aiel
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::AllocMemBuffer)


#endif  // GET_OP_CLASSES

