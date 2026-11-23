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

::mlir::aie::KernelFuncNameType,
::mlir::aie::KernelFileNameType,
::mlir::aie::WindowType,
::mlir::aie::KernelType,
::mlir::aie::TileKernelType,
::mlir::aie::LocType,
::mlir::aie::CoreConstraintType,
::mlir::aie::RoutingInstanceType,
::mlir::aie::HostAieConstraintType,
::mlir::aie::RoutingConstraintType,
::mlir::aie::DeviceInstanceType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::aie::KernelFuncNameType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::KernelFuncNameType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::KernelFileNameType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::KernelFileNameType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::WindowType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::WindowType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::KernelType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::KernelType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::TileKernelType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::TileKernelType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::LocType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::LocType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::CoreConstraintType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::CoreConstraintType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::RoutingInstanceType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::RoutingInstanceType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::HostAieConstraintType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::HostAieConstraintType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::RoutingConstraintType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::RoutingConstraintType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::mlir::aie::DeviceInstanceType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::aie::DeviceInstanceType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::aie::KernelFuncNameType>([&](auto t) {
      printer << ::mlir::aie::KernelFuncNameType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::aie::KernelFileNameType>([&](auto t) {
      printer << ::mlir::aie::KernelFileNameType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::aie::WindowType>([&](auto t) {
      printer << ::mlir::aie::WindowType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::aie::KernelType>([&](auto t) {
      printer << ::mlir::aie::KernelType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::aie::TileKernelType>([&](auto t) {
      printer << ::mlir::aie::TileKernelType::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::aie::LocType>([&](auto t) {
      printer << ::mlir::aie::LocType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::aie::CoreConstraintType>([&](auto t) {
      printer << ::mlir::aie::CoreConstraintType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::aie::RoutingInstanceType>([&](auto t) {
      printer << ::mlir::aie::RoutingInstanceType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::aie::HostAieConstraintType>([&](auto t) {
      printer << ::mlir::aie::HostAieConstraintType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::aie::RoutingConstraintType>([&](auto t) {
      printer << ::mlir::aie::RoutingConstraintType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::mlir::aie::DeviceInstanceType>([&](auto t) {
      printer << ::mlir::aie::DeviceInstanceType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir::aie {

namespace detail {

struct KernelFuncNameTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr>;
  KernelFuncNameTypeStorage(StringAttr name) : name(std::move(name)) {}

  KeyTy getAsKey() const {
    return KeyTy(name);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (name == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static KernelFuncNameTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto name = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<KernelFuncNameTypeStorage>()) KernelFuncNameTypeStorage(std::move(name));
  }

  StringAttr name;
};

} // namespace detail
KernelFuncNameType KernelFuncNameType::get(::mlir::MLIRContext *context, StringAttr name) {
  return Base::get(context, std::move(name));
}

::mlir::Type KernelFuncNameType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_name;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'name'
  _result_name = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_name)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_KernelFuncNameType parameter 'name' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_name));
  return KernelFuncNameType::get(odsParser.getContext(),
      StringAttr((*_result_name)));
}

void KernelFuncNameType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getName());
  odsPrinter << ">";
}

StringAttr KernelFuncNameType::getName() const {
  return getImpl()->name;
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::KernelFuncNameType)
namespace mlir::aie {

namespace detail {

struct KernelFileNameTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<StringAttr>;
  KernelFileNameTypeStorage(StringAttr name) : name(std::move(name)) {}

  KeyTy getAsKey() const {
    return KeyTy(name);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (name == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static KernelFileNameTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto name = std::move(std::get<0>(tblgenKey));
    return new (allocator.allocate<KernelFileNameTypeStorage>()) KernelFileNameTypeStorage(std::move(name));
  }

  StringAttr name;
};

} // namespace detail
KernelFileNameType KernelFileNameType::get(::mlir::MLIRContext *context, StringAttr name) {
  return Base::get(context, std::move(name));
}

::mlir::Type KernelFileNameType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<StringAttr> _result_name;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'name'
  _result_name = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_name)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_KernelFileNameType parameter 'name' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_name));
  return KernelFileNameType::get(odsParser.getContext(),
      StringAttr((*_result_name)));
}

void KernelFileNameType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getName());
  odsPrinter << ">";
}

StringAttr KernelFileNameType::getName() const {
  return getImpl()->name;
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::KernelFileNameType)
namespace mlir::aie {

namespace detail {

struct WindowTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<int, int, long, long, int, int, StringAttr>;
  WindowTypeStorage(int direction, int size, long pingaddr, long pongaddr, int pinglockid, int ponglockid, StringAttr name) : direction(std::move(direction)), size(std::move(size)), pingaddr(std::move(pingaddr)), pongaddr(std::move(pongaddr)), pinglockid(std::move(pinglockid)), ponglockid(std::move(ponglockid)), name(std::move(name)) {}

  KeyTy getAsKey() const {
    return KeyTy(direction, size, pingaddr, pongaddr, pinglockid, ponglockid, name);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (direction == std::get<0>(tblgenKey)) && (size == std::get<1>(tblgenKey)) && (pingaddr == std::get<2>(tblgenKey)) && (pongaddr == std::get<3>(tblgenKey)) && (pinglockid == std::get<4>(tblgenKey)) && (ponglockid == std::get<5>(tblgenKey)) && (name == std::get<6>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey), std::get<5>(tblgenKey), std::get<6>(tblgenKey));
  }

  static WindowTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto direction = std::move(std::get<0>(tblgenKey));
    auto size = std::move(std::get<1>(tblgenKey));
    auto pingaddr = std::move(std::get<2>(tblgenKey));
    auto pongaddr = std::move(std::get<3>(tblgenKey));
    auto pinglockid = std::move(std::get<4>(tblgenKey));
    auto ponglockid = std::move(std::get<5>(tblgenKey));
    auto name = std::move(std::get<6>(tblgenKey));
    return new (allocator.allocate<WindowTypeStorage>()) WindowTypeStorage(std::move(direction), std::move(size), std::move(pingaddr), std::move(pongaddr), std::move(pinglockid), std::move(ponglockid), std::move(name));
  }

  int direction;
  int size;
  long pingaddr;
  long pongaddr;
  int pinglockid;
  int ponglockid;
  StringAttr name;
};

} // namespace detail
WindowType WindowType::get(::mlir::MLIRContext *context, int direction, int size, long pingaddr, long pongaddr, int pinglockid, int ponglockid, StringAttr name) {
  return Base::get(context, std::move(direction), std::move(size), std::move(pingaddr), std::move(pongaddr), std::move(pinglockid), std::move(ponglockid), std::move(name));
}

::mlir::Type WindowType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int> _result_direction;
  ::mlir::FailureOr<int> _result_size;
  ::mlir::FailureOr<long> _result_pingaddr;
  ::mlir::FailureOr<long> _result_pongaddr;
  ::mlir::FailureOr<int> _result_pinglockid;
  ::mlir::FailureOr<int> _result_ponglockid;
  ::mlir::FailureOr<StringAttr> _result_name;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'direction'
  _result_direction = ::mlir::FieldParser<int>::parse(odsParser);
  if (::mlir::failed(_result_direction)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_WindowType parameter 'direction' which is to be a `int`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'size'
  _result_size = ::mlir::FieldParser<int>::parse(odsParser);
  if (::mlir::failed(_result_size)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_WindowType parameter 'size' which is to be a `int`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'pingaddr'
  _result_pingaddr = ::mlir::FieldParser<long>::parse(odsParser);
  if (::mlir::failed(_result_pingaddr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_WindowType parameter 'pingaddr' which is to be a `long`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'pongaddr'
  _result_pongaddr = ::mlir::FieldParser<long>::parse(odsParser);
  if (::mlir::failed(_result_pongaddr)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_WindowType parameter 'pongaddr' which is to be a `long`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'pinglockid'
  _result_pinglockid = ::mlir::FieldParser<int>::parse(odsParser);
  if (::mlir::failed(_result_pinglockid)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_WindowType parameter 'pinglockid' which is to be a `int`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'ponglockid'
  _result_ponglockid = ::mlir::FieldParser<int>::parse(odsParser);
  if (::mlir::failed(_result_ponglockid)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_WindowType parameter 'ponglockid' which is to be a `int`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'name'
  _result_name = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_name)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_WindowType parameter 'name' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_direction));
  assert(::mlir::succeeded(_result_size));
  assert(::mlir::succeeded(_result_pingaddr));
  assert(::mlir::succeeded(_result_pongaddr));
  assert(::mlir::succeeded(_result_pinglockid));
  assert(::mlir::succeeded(_result_ponglockid));
  assert(::mlir::succeeded(_result_name));
  return WindowType::get(odsParser.getContext(),
      int((*_result_direction)),
      int((*_result_size)),
      long((*_result_pingaddr)),
      long((*_result_pongaddr)),
      int((*_result_pinglockid)),
      int((*_result_ponglockid)),
      StringAttr((*_result_name)));
}

void WindowType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getDirection());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getSize());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getPingaddr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getPongaddr());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getPinglockid());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getPonglockid());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getName());
  odsPrinter << ">";
}

int WindowType::getDirection() const {
  return getImpl()->direction;
}

int WindowType::getSize() const {
  return getImpl()->size;
}

long WindowType::getPingaddr() const {
  return getImpl()->pingaddr;
}

long WindowType::getPongaddr() const {
  return getImpl()->pongaddr;
}

int WindowType::getPinglockid() const {
  return getImpl()->pinglockid;
}

int WindowType::getPonglockid() const {
  return getImpl()->ponglockid;
}

StringAttr WindowType::getName() const {
  return getImpl()->name;
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::WindowType)
namespace mlir::aie {

namespace detail {

struct KernelTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<int, int, KernelFuncNameType, KernelFileNameType, ::llvm::ArrayRef<WindowType>>;
  KernelTypeStorage(int numInputArgs, int numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, ::llvm::ArrayRef<WindowType> arguments) : numInputArgs(std::move(numInputArgs)), numOutputArgs(std::move(numOutputArgs)), kname(std::move(kname)), fname(std::move(fname)), arguments(std::move(arguments)) {}

  KeyTy getAsKey() const {
    return KeyTy(numInputArgs, numOutputArgs, kname, fname, arguments);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (numInputArgs == std::get<0>(tblgenKey)) && (numOutputArgs == std::get<1>(tblgenKey)) && (kname == std::get<2>(tblgenKey)) && (fname == std::get<3>(tblgenKey)) && (::llvm::ArrayRef<WindowType>(arguments) == ::llvm::ArrayRef<WindowType>(std::get<4>(tblgenKey)));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey), std::get<3>(tblgenKey), std::get<4>(tblgenKey));
  }

  static KernelTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto numInputArgs = std::move(std::get<0>(tblgenKey));
    auto numOutputArgs = std::move(std::get<1>(tblgenKey));
    auto kname = std::move(std::get<2>(tblgenKey));
    auto fname = std::move(std::get<3>(tblgenKey));
    auto arguments = std::move(std::get<4>(tblgenKey));
    arguments = allocator.copyInto(arguments);
    return new (allocator.allocate<KernelTypeStorage>()) KernelTypeStorage(std::move(numInputArgs), std::move(numOutputArgs), std::move(kname), std::move(fname), std::move(arguments));
  }

  int numInputArgs;
  int numOutputArgs;
  KernelFuncNameType kname;
  KernelFileNameType fname;
  ::llvm::ArrayRef<WindowType> arguments;
};

} // namespace detail
KernelType KernelType::get(::mlir::MLIRContext *context, int numInputArgs, int numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, ::llvm::ArrayRef<WindowType> arguments) {
  return Base::get(context, std::move(numInputArgs), std::move(numOutputArgs), std::move(kname), std::move(fname), std::move(arguments));
}

::mlir::Type KernelType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int> _result_numInputArgs;
  ::mlir::FailureOr<int> _result_numOutputArgs;
  ::mlir::FailureOr<KernelFuncNameType> _result_kname;
  ::mlir::FailureOr<KernelFileNameType> _result_fname;
  ::mlir::FailureOr<::llvm::SmallVector<WindowType>> _result_arguments;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'numInputArgs'
  _result_numInputArgs = ::mlir::FieldParser<int>::parse(odsParser);
  if (::mlir::failed(_result_numInputArgs)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_KernelObjectType parameter 'numInputArgs' which is to be a `int`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'numOutputArgs'
  _result_numOutputArgs = ::mlir::FieldParser<int>::parse(odsParser);
  if (::mlir::failed(_result_numOutputArgs)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_KernelObjectType parameter 'numOutputArgs' which is to be a `int`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'kname'
  _result_kname = ::mlir::FieldParser<KernelFuncNameType>::parse(odsParser);
  if (::mlir::failed(_result_kname)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_KernelObjectType parameter 'kname' which is to be a `KernelFuncNameType`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'fname'
  _result_fname = ::mlir::FieldParser<KernelFileNameType>::parse(odsParser);
  if (::mlir::failed(_result_fname)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_KernelObjectType parameter 'fname' which is to be a `KernelFileNameType`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'arguments'
  _result_arguments = ::mlir::FieldParser<::llvm::SmallVector<WindowType>>::parse(odsParser);
  if (::mlir::failed(_result_arguments)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_KernelObjectType parameter 'arguments' which is to be a `::llvm::ArrayRef<WindowType>`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_numInputArgs));
  assert(::mlir::succeeded(_result_numOutputArgs));
  assert(::mlir::succeeded(_result_kname));
  assert(::mlir::succeeded(_result_fname));
  return KernelType::get(odsParser.getContext(),
      int((*_result_numInputArgs)),
      int((*_result_numOutputArgs)),
      KernelFuncNameType((*_result_kname)),
      KernelFileNameType((*_result_fname)),
      ::llvm::ArrayRef<WindowType>((_result_arguments.value_or(::llvm::SmallVector<WindowType>()))));
}

void KernelType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getNumInputArgs());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getNumOutputArgs());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getKname());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getFname());
  odsPrinter << ",";
  if (!(::llvm::ArrayRef<WindowType>(getArguments()) == ::llvm::ArrayRef<WindowType>(::llvm::SmallVector<WindowType>()))) {
    odsPrinter << ' ';
    odsPrinter.printStrippedAttrOrType(getArguments());
  }
  odsPrinter << ">";
}

int KernelType::getNumInputArgs() const {
  return getImpl()->numInputArgs;
}

int KernelType::getNumOutputArgs() const {
  return getImpl()->numOutputArgs;
}

KernelFuncNameType KernelType::getKname() const {
  return getImpl()->kname;
}

KernelFileNameType KernelType::getFname() const {
  return getImpl()->fname;
}

::llvm::ArrayRef<WindowType> KernelType::getArguments() const {
  return getImpl()->arguments;
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::KernelType)
namespace mlir::aie {

namespace detail {

struct TileKernelTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<int, int, StringAttr>;
  TileKernelTypeStorage(int col, int row, StringAttr kernels) : col(std::move(col)), row(std::move(row)), kernels(std::move(kernels)) {}

  KeyTy getAsKey() const {
    return KeyTy(col, row, kernels);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (col == std::get<0>(tblgenKey)) && (row == std::get<1>(tblgenKey)) && (kernels == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static TileKernelTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto col = std::move(std::get<0>(tblgenKey));
    auto row = std::move(std::get<1>(tblgenKey));
    auto kernels = std::move(std::get<2>(tblgenKey));
    return new (allocator.allocate<TileKernelTypeStorage>()) TileKernelTypeStorage(std::move(col), std::move(row), std::move(kernels));
  }

  int col;
  int row;
  StringAttr kernels;
};

} // namespace detail
TileKernelType TileKernelType::get(::mlir::MLIRContext *context, int col, int row, StringAttr kernels) {
  return Base::get(context, std::move(col), std::move(row), std::move(kernels));
}

::mlir::Type TileKernelType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int> _result_col;
  ::mlir::FailureOr<int> _result_row;
  ::mlir::FailureOr<StringAttr> _result_kernels;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'col'
  _result_col = ::mlir::FieldParser<int>::parse(odsParser);
  if (::mlir::failed(_result_col)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_TileKernelObjectType parameter 'col' which is to be a `int`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'row'
  _result_row = ::mlir::FieldParser<int>::parse(odsParser);
  if (::mlir::failed(_result_row)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_TileKernelObjectType parameter 'row' which is to be a `int`");
    return {};
  }
  // Parse literal ','
  if (odsParser.parseComma()) return {};

  // Parse variable 'kernels'
  _result_kernels = ::mlir::FieldParser<StringAttr>::parse(odsParser);
  if (::mlir::failed(_result_kernels)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Aie_TileKernelObjectType parameter 'kernels' which is to be a `StringAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_col));
  assert(::mlir::succeeded(_result_row));
  assert(::mlir::succeeded(_result_kernels));
  return TileKernelType::get(odsParser.getContext(),
      int((*_result_col)),
      int((*_result_row)),
      StringAttr((*_result_kernels)));
}

void TileKernelType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getCol());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getRow());
  odsPrinter << ",";
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getKernels());
  odsPrinter << ">";
}

int TileKernelType::getCol() const {
  return getImpl()->col;
}

int TileKernelType::getRow() const {
  return getImpl()->row;
}

StringAttr TileKernelType::getKernels() const {
  return getImpl()->kernels;
}


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::TileKernelType)
namespace mlir::aie {


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::LocType)
namespace mlir::aie {


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::CoreConstraintType)
namespace mlir::aie {


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::RoutingInstanceType)
namespace mlir::aie {


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::HostAieConstraintType)
namespace mlir::aie {


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::RoutingConstraintType)
namespace mlir::aie {


} // namespace mlir::aie
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aie::DeviceInstanceType)
namespace mlir::aie {


/// Parse a type registered to this dialect.
::mlir::Type AieADialect::parseType(::mlir::DialectAsmParser &parser) const {
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
void AieADialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}

} // namespace mlir::aie

#endif // GET_TYPEDEF_CLASSES

