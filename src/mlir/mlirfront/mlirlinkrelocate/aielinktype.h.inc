/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

#include "mlir/IR/Types.h"


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir::aiel {

/// Mem bank
/// the mem bank
class Aie_L1MemBankType;
/// The Mem Buffer
/// the symbol type
class Aie_MemBufferType;
/// The symbol type
/// the symbol type
class Aie_SymbolType;
/// the section type
/// the sction type
class Aie_SectionType;
namespace detail {

struct Aie_L1MemBankTypeStorage;

} // namespace detail
class Aie_L1MemBankType : public ::mlir::Type::TypeBase<Aie_L1MemBankType, ::mlir::Type, detail::Aie_L1MemBankTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "AieLink.Mem bank";
  static constexpr ::llvm::StringLiteral dialectName = "AieLink";
  static Aie_L1MemBankType get(::mlir::MLIRContext *context, IntegerAttr locationdirect, StringAttr locationtext, IntegerAttr baseaddr, IntegerAttr memtotalsize, IntegerAttr islocal);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"Mem bank"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  IntegerAttr getLocationdirect() const;
  StringAttr getLocationtext() const;
  IntegerAttr getBaseaddr() const;
  IntegerAttr getMemtotalsize() const;
  IntegerAttr getIslocal() const;
};
namespace detail {

struct Aie_MemBufferTypeStorage;

} // namespace detail
class Aie_MemBufferType : public ::mlir::Type::TypeBase<Aie_MemBufferType, ::mlir::Type, detail::Aie_MemBufferTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "AieLink.MemBuffer";
  static constexpr ::llvm::StringLiteral dialectName = "AieLink";
  static Aie_MemBufferType get(::mlir::MLIRContext *context, Aie_L1MemBankType mbanktype, IntegerAttr addr, IntegerAttr len);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"MemBuffer"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  Aie_L1MemBankType getMbanktype() const;
  IntegerAttr getAddr() const;
  IntegerAttr getLen() const;
};
namespace detail {

struct Aie_SymbolTypeStorage;

} // namespace detail
class Aie_SymbolType : public ::mlir::Type::TypeBase<Aie_SymbolType, ::mlir::Type, detail::Aie_SymbolTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "AieLink.SymbolType";
  static constexpr ::llvm::StringLiteral dialectName = "AieLink";
  static Aie_SymbolType get(::mlir::MLIRContext *context, StringAttr name, Aie_MemBufferType addr);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"SymbolType"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getName() const;
  Aie_MemBufferType getAddr() const;
};
namespace detail {

struct Aie_SectionTypeStorage;

} // namespace detail
class Aie_SectionType : public ::mlir::Type::TypeBase<Aie_SectionType, ::mlir::Type, detail::Aie_SectionTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "AieLink.SectionType";
  static constexpr ::llvm::StringLiteral dialectName = "AieLink";
  static Aie_SectionType get(::mlir::MLIRContext *context, StringAttr sectiontypename, StringAttr sectionname, Aie_MemBufferType startaddress, IntegerAttr len);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"SectionType"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getSectiontypename() const;
  StringAttr getSectionname() const;
  Aie_MemBufferType getStartaddress() const;
  IntegerAttr getLen() const;
};

} // namespace mlir::aiel
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::Aie_L1MemBankType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::Aie_MemBufferType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::Aie_SymbolType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aiel::Aie_SectionType)

#endif // GET_TYPEDEF_CLASSES

