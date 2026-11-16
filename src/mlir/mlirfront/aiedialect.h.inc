/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: aieops.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "mlir/IR/Dialect.h"

namespace mlir::aie {


class AieADialect : public ::mlir::Dialect {
  explicit AieADialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~AieADialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("Aie");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::AieADialect)
