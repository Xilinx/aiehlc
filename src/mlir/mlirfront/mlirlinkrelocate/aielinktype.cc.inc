/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::aiel::Aie_L1MemBankType,
::mlir::aiel::Aie_MemBufferType,
::mlir::aiel::Aie_SymbolType,
::mlir::aiel::Aie_SectionType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::aiel::Aie_L1MemBankType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aiel::Aie_L1MemBankType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aiel::Aie_MemBufferType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aiel::Aie_MemBufferType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aiel::Aie_SymbolType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aiel::Aie_SymbolType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aiel::Aie_SectionType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aiel::Aie_SectionType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::aiel::Aie_L1MemBankType>([&](auto t) {
      printer << ::mlir::aiel::Aie_L1MemBankType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::aiel::Aie_MemBufferType>([&](auto t) {
      printer << ::mlir::aiel::Aie_MemBufferType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::aiel::Aie_SymbolType>([&](auto t) {
      printer << ::mlir::aiel::Aie_SymbolType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::aiel::Aie_SectionType>([&](auto t) {
      printer << ::mlir::aiel::Aie_SectionType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::aiel {

namespace detail {

struct Aie_L1MemBankTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<IntegerAttr, StringAttr, IntegerAttr, IntegerAttr, IntegerAttr>;
  Aie_L1MemBankTypeStorage(IntegerAttr locationdirect, StringAttr locationtext, IntegerAttr baseaddr, IntegerAttr memtotalsize, IntegerAttr islocal) : locationdirect(std::move(locationdirect)), locationtext(std::move(locationtext)), baseaddr(std::move(baseaddr)), memtotalsize(std::move(memtotalsize)), islocal(std::move(islocal)) {}

  KeyTy getAsKey() const {
    return KeyTy(locationdirect, locationtext, baseaddr, memtotalsize, islocal);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (locationdirect == std::get<0>(tblgenKey)) && (locationtext == std::get<1>(tblgenKey)) && (baseaddr == std::get<2>(tblgenKey)) && (memtotalsize == std::get<3>(tblgenKey)) && (islocal == std::get<4>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static Aie_L1MemBankTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto locationdirect = std::move(std::get<0>(tblgenKey));
    auto locationtext = std::move(std::get<1>(tblgenKey));
    auto baseaddr = std::move(std::get<2>(tblgenKey));
    auto memtotalsize = std::move(std::get<3>(tblgenKey));
    auto islocal = std::move(std::get<4>(tblgenKey));
    return new (allocator.allocate<Aie_L1MemBankTypeStorage>()) Aie_L1MemBankTypeStorage(std::move(locationdirect), std::move(locationtext), std::move(baseaddr), std::move(memtotalsize), std::move(islocal));
  }

  IntegerAttr locationdirect;
  StringAttr locationtext;
  IntegerAttr baseaddr;
  IntegerAttr memtotalsize;
  IntegerAttr islocal;
};

} // namespace detail
Aie_L1MemBankType Aie_L1MemBankType::get(::mlir::MLIRContext *context, IntegerAttr locationdirect, StringAttr locationtext, IntegerAttr baseaddr, IntegerAttr memtotalsize, IntegerAttr islocal) {
  return Base::get(context, std::move(locationdirect), std::move(locationtext), std::move(baseaddr), std::move(memtotalsize), std::move(islocal));
}

::mlir::Type Aie_L1MemBankType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<IntegerAttr> _result_locationdirect;
  ::mlir::FailureOr<StringAttr> _result_locationtext;
  ::mlir::FailureOr<IntegerAttr> _result_baseaddr;
  ::mlir::FailureOr<IntegerAttr> _result_memtotalsize;
  ::mlir::FailureOr<IntegerAttr> _result_islocal;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'locationdirect'
  _result_locationdirect = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_locationdirect)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_L1MemBankType parameter 'locationdirect' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'locationtext'
  _result_locationtext = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_locationtext)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_L1MemBankType parameter 'locationtext' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'baseaddr'
  _result_baseaddr = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_baseaddr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_L1MemBankType parameter 'baseaddr' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'memtotalsize'
  _result_memtotalsize = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_memtotalsize)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_L1MemBankType parameter 'memtotalsize' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'islocal'
  _result_islocal = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_islocal)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_L1MemBankType parameter 'islocal' which is to be a `IntegerAttr`");
    return {};
  }
  assert(::mlir::succeeded(_result_locationdirect));
  assert(::mlir::succeeded(_result_locationtext));
  assert(::mlir::succeeded(_result_baseaddr));
  assert(::mlir::succeeded(_result_memtotalsize));
  assert(::mlir::succeeded(_result_islocal));
  return Aie_L1MemBankType::get(odsParser.getContext(),
      IntegerAttr((*_result_locationdirect)),
      StringAttr((*_result_locationtext)),
      IntegerAttr((*_result_baseaddr)),
      IntegerAttr((*_result_memtotalsize)),
      IntegerAttr((*_result_islocal)));
}

void Aie_L1MemBankType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getLocationdirect());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLocationtext());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getBaseaddr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getMemtotalsize());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getIslocal());
}

IntegerAttr Aie_L1MemBankType::getLocationdirect() const {
  return getImpl()->locationdirect;
}

StringAttr Aie_L1MemBankType::getLocationtext() const {
  return getImpl()->locationtext;
}

IntegerAttr Aie_L1MemBankType::getBaseaddr() const {
  return getImpl()->baseaddr;
}

IntegerAttr Aie_L1MemBankType::getMemtotalsize() const {
  return getImpl()->memtotalsize;
}

IntegerAttr Aie_L1MemBankType::getIslocal() const {
  return getImpl()->islocal;
}


} // namespace mlir::aiel
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::Aie_L1MemBankType)
namespace mlir::aiel {

namespace detail {

struct Aie_MemBufferTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<Aie_L1MemBankType, IntegerAttr, IntegerAttr>;
  Aie_MemBufferTypeStorage(Aie_L1MemBankType mbanktype, IntegerAttr addr, IntegerAttr len) : mbanktype(std::move(mbanktype)), addr(std::move(addr)), len(std::move(len)) {}

  KeyTy getAsKey() const {
    return KeyTy(mbanktype, addr, len);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (mbanktype == std::get<0>(tblgenKey)) && (addr == std::get<1>(tblgenKey)) && (len == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static Aie_MemBufferTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto mbanktype = std::move(std::get<0>(tblgenKey));
    auto addr = std::move(std::get<1>(tblgenKey));
    auto len = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<Aie_MemBufferTypeStorage>()) Aie_MemBufferTypeStorage(std::move(mbanktype), std::move(addr), std::move(len));
  }

  Aie_L1MemBankType mbanktype;
  IntegerAttr addr;
  IntegerAttr len;
};

} // namespace detail
Aie_MemBufferType Aie_MemBufferType::get(::mlir::MLIRContext *context, Aie_L1MemBankType mbanktype, IntegerAttr addr, IntegerAttr len) {
  return Base::get(context, std::move(mbanktype), std::move(addr), std::move(len));
}

::mlir::Type Aie_MemBufferType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<Aie_L1MemBankType> _result_mbanktype;
  ::mlir::FailureOr<IntegerAttr> _result_addr;
  ::mlir::FailureOr<IntegerAttr> _result_len;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'mbanktype'
  _result_mbanktype = ::mlir::FieldParser<Aie_L1MemBankType>::parse(odsParser);
  if (::mlir::failed(_result_mbanktype)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_MemBufferType parameter 'mbanktype' which is to be a `Aie_L1MemBankType`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'addr'
  _result_addr = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_addr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_MemBufferType parameter 'addr' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'len'
  _result_len = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_len)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_MemBufferType parameter 'len' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_mbanktype));
  assert(::mlir::succeeded(_result_addr));
  assert(::mlir::succeeded(_result_len));
  return Aie_MemBufferType::get(odsParser.getContext(),
      Aie_L1MemBankType((*_result_mbanktype)),
      IntegerAttr((*_result_addr)),
      IntegerAttr((*_result_len)));
}

void Aie_MemBufferType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getMbanktype());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getAddr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLen());
  odsPrinter << ">";
}

Aie_L1MemBankType Aie_MemBufferType::getMbanktype() const {
  return getImpl()->mbanktype;
}

IntegerAttr Aie_MemBufferType::getAddr() const {
  return getImpl()->addr;
}

IntegerAttr Aie_MemBufferType::getLen() const {
  return getImpl()->len;
}


} // namespace mlir::aiel
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::Aie_MemBufferType)
namespace mlir::aiel {

namespace detail {

struct Aie_SymbolTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr, Aie_MemBufferType>;
  Aie_SymbolTypeStorage(StringAttr name, Aie_MemBufferType addr) : name(std::move(name)), addr(std::move(addr)) {}

  KeyTy getAsKey() const {
    return KeyTy(name, addr);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (name == std::get<0>(tblgenKey)) && (addr == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static Aie_SymbolTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto name = std::move(std::get<0>(tblgenKey));
    auto addr = std::move(std::get<1>(tblgenKey));
    return new (allocator.allocate<Aie_SymbolTypeStorage>()) Aie_SymbolTypeStorage(std::move(name), std::move(addr));
  }

  StringAttr name;
  Aie_MemBufferType addr;
};

} // namespace detail
Aie_SymbolType Aie_SymbolType::get(::mlir::MLIRContext *context, StringAttr name, Aie_MemBufferType addr) {
  return Base::get(context, std::move(name), std::move(addr));
}

::mlir::Type Aie_SymbolType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_name;
  ::mlir::FailureOr<Aie_MemBufferType> _result_addr;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'name'
  _result_name = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_name)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_SymbolType parameter 'name' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'addr'
  _result_addr = ::mlir::FieldParser<Aie_MemBufferType>::parse(odsParser);
  if (::mlir::failed(_result_addr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_SymbolType parameter 'addr' which is to be a `Aie_MemBufferType`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_name));
  assert(::mlir::succeeded(_result_addr));
  return Aie_SymbolType::get(odsParser.getContext(),
      StringAttr((*_result_name)),
      Aie_MemBufferType((*_result_addr)));
}

void Aie_SymbolType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getName());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getAddr());
  odsPrinter << ">";
}

StringAttr Aie_SymbolType::getName() const {
  return getImpl()->name;
}

Aie_MemBufferType Aie_SymbolType::getAddr() const {
  return getImpl()->addr;
}


} // namespace mlir::aiel
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::Aie_SymbolType)
namespace mlir::aiel {

namespace detail {

struct Aie_SectionTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr, StringAttr, Aie_MemBufferType, IntegerAttr>;
  Aie_SectionTypeStorage(StringAttr sectiontypename, StringAttr sectionname, Aie_MemBufferType startaddress, IntegerAttr len) : sectiontypename(std::move(sectiontypename)), sectionname(std::move(sectionname)), startaddress(std::move(startaddress)), len(std::move(len)) {}

  KeyTy getAsKey() const {
    return KeyTy(sectiontypename, sectionname, startaddress, len);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (sectiontypename == std::get<0>(tblgenKey)) && (sectionname == std::get<1>(tblgenKey)) && (startaddress == std::get<2>(tblgenKey)) && (len == std::get<3>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey));
  }

  static Aie_SectionTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto sectiontypename = std::move(std::get<0>(tblgenKey));
    auto sectionname = std::move(std::get<1>(tblgenKey));
    auto startaddress = std::move(std::get<2>(tblgenKey));
    auto len = std::move(std::get<3>(tblgenKey));
    return new (allocator.allocate<Aie_SectionTypeStorage>()) Aie_SectionTypeStorage(std::move(sectiontypename), std::move(sectionname), std::move(startaddress), std::move(len));
  }

  StringAttr sectiontypename;
  StringAttr sectionname;
  Aie_MemBufferType startaddress;
  IntegerAttr len;
};

} // namespace detail
Aie_SectionType Aie_SectionType::get(::mlir::MLIRContext *context, StringAttr sectiontypename, StringAttr sectionname, Aie_MemBufferType startaddress, IntegerAttr len) {
  return Base::get(context, std::move(sectiontypename), std::move(sectionname), std::move(startaddress), std::move(len));
}

::mlir::Type Aie_SectionType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_sectiontypename;
  ::mlir::FailureOr<StringAttr> _result_sectionname;
  ::mlir::FailureOr<Aie_MemBufferType> _result_startaddress;
  ::mlir::FailureOr<IntegerAttr> _result_len;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'sectiontypename'
  _result_sectiontypename = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_sectiontypename)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_SectionType parameter 'sectiontypename' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'sectionname'
  _result_sectionname = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_sectionname)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_SectionType parameter 'sectionname' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'startaddress'
  _result_startaddress = ::mlir::FieldParser<Aie_MemBufferType>::parse(odsParser);
  if (::mlir::failed(_result_startaddress)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_SectionType parameter 'startaddress' which is to be a `Aie_MemBufferType`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'len'
  _result_len = ::mlir::FieldParser<IntegerAttr>::parse(odsParser);
  if (::mlir::failed(_result_len)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_SectionType parameter 'len' which is to be a `IntegerAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_sectiontypename));
  assert(::mlir::succeeded(_result_sectionname));
  assert(::mlir::succeeded(_result_startaddress));
  assert(::mlir::succeeded(_result_len));
  return Aie_SectionType::get(odsParser.getContext(),
      StringAttr((*_result_sectiontypename)),
      StringAttr((*_result_sectionname)),
      Aie_MemBufferType((*_result_startaddress)),
      IntegerAttr((*_result_len)));
}

void Aie_SectionType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getSectiontypename());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getSectionname());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getStartaddress());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getLen());
  odsPrinter << ">";
}

StringAttr Aie_SectionType::getSectiontypename() const {
  return getImpl()->sectiontypename;
}

StringAttr Aie_SectionType::getSectionname() const {
  return getImpl()->sectionname;
}

Aie_MemBufferType Aie_SectionType::getStartaddress() const {
  return getImpl()->startaddress;
}

IntegerAttr Aie_SectionType::getLen() const {
  return getImpl()->len;
}


} // namespace mlir::aiel
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::Aie_SectionType)
namespace mlir::aiel {


/// Parse a type registered to this dialect.
::mlir::Type AieLinkDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void AieLinkDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}

} // namespace mlir::aiel

#endif // GET_TYPEDEF_CLASSES

