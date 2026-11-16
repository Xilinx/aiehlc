/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: aielinkops.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::aiel::AieLinkDialect)
namespace mlir::aiel {


AieLinkDialect::AieLinkDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<AieLinkDialect>())
    
     {
  
  initialize();
}

AieLinkDialect::~AieLinkDialect() = default;


} // namespace mlir::aiel
