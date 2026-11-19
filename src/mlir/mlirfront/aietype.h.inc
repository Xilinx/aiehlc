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
namespace mlir::aie {

/// the kernel function name
/// the kernel function name
class KernelFuncNameType;
/// the kernel path
/// kernel file path
class KernelFileNameType;
/// Window type with direction, size, and name
/// A type encapsulating a window with a specific direction, size, and name.
/// These attributes are constant and determined at the time of window creation.
class WindowType;
/// Represents a kernel object in the AIE dialect. It encapsulates a kernel symbol reference,
/// the number of input and output arguments, and a variadic array of window types for the arguments.
/// Specifically this object collects all the static information about the kernel which 
/// would be useful in constructing full kernel source code.
class KernelType;
/// Represents a kernel tile object in the AIE dialect. It encapsulates a kernel symbol reference,
/// the number of input and output arguments, and a variadic array of window types for the arguments.
/// Specifically this object collects all the static information about the kernel which would be useful in 
/// constructing full kernel source code.
class TileKernelType;
/// Represents a location of a tile within AIE array (col, row)
class LocType;
/// Represents Core Constraint struct.
class CoreConstraintType;
/// Represents Routing Instance struct.
class RoutingInstanceType;
/// Represents Host Aie Constraint.
class HostAieConstraintType;
/// Represents a routing constraint in the AIE dialect.
class RoutingConstraintType;
/// Represents The Device Instance.
class DeviceInstanceType;
namespace detail {

struct KernelFuncNameTypeStorage;

} // namespace detail
class KernelFuncNameType : public ::mlir::Type::TypeBase<KernelFuncNameType, ::mlir::Type, detail::KernelFuncNameTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.KernelFuncName";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static KernelFuncNameType get(::mlir::MLIRContext *context, StringAttr name);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"KernelFuncName"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getName() const;
};
namespace detail {

struct KernelFileNameTypeStorage;

} // namespace detail
class KernelFileNameType : public ::mlir::Type::TypeBase<KernelFileNameType, ::mlir::Type, detail::KernelFileNameTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.KernelFileName";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static KernelFileNameType get(::mlir::MLIRContext *context, StringAttr name);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"KernelFileName"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  StringAttr getName() const;
};
namespace detail {

struct WindowTypeStorage;

} // namespace detail
class WindowType : public ::mlir::Type::TypeBase<WindowType, ::mlir::Type, detail::WindowTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.Window";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static WindowType get(::mlir::MLIRContext *context, int direction, int size, long pingaddr, long pongaddr, int pinglockid, int ponglockid, StringAttr name);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"Window"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getDirection() const;
  int getSize() const;
  long getPingaddr() const;
  long getPongaddr() const;
  int getPinglockid() const;
  int getPonglockid() const;
  StringAttr getName() const;
};
namespace detail {

struct KernelTypeStorage;

} // namespace detail
class KernelType : public ::mlir::Type::TypeBase<KernelType, ::mlir::Type, detail::KernelTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.KernelObj";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static KernelType get(::mlir::MLIRContext *context, int numInputArgs, int numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, ::llvm::ArrayRef<WindowType> arguments);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"KernelObj"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getNumInputArgs() const;
  int getNumOutputArgs() const;
  KernelFuncNameType getKname() const;
  KernelFileNameType getFname() const;
  ::llvm::ArrayRef<WindowType> getArguments() const;
};
namespace detail {

struct TileKernelTypeStorage;

} // namespace detail
class TileKernelType : public ::mlir::Type::TypeBase<TileKernelType, ::mlir::Type, detail::TileKernelTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.TileKernelObj";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static TileKernelType get(::mlir::MLIRContext *context, int col, int row, StringAttr kernels);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"TileKernelObj"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getCol() const;
  int getRow() const;
  StringAttr getKernels() const;
};
class LocType : public ::mlir::Type::TypeBase<LocType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.Loc";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"Loc"};
  }

};
class CoreConstraintType : public ::mlir::Type::TypeBase<CoreConstraintType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.CoreConstraint";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"CoreConstraint"};
  }

};
class RoutingInstanceType : public ::mlir::Type::TypeBase<RoutingInstanceType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.RoutingInstance";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"RoutingInstance"};
  }

};
class HostAieConstraintType : public ::mlir::Type::TypeBase<HostAieConstraintType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.HostAieConstraint";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"HostAieConstraint"};
  }

};
class RoutingConstraintType : public ::mlir::Type::TypeBase<RoutingConstraintType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.RoutingConstraint";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"RoutingConstraint"};
  }

};
class DeviceInstanceType : public ::mlir::Type::TypeBase<DeviceInstanceType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "Aie.DeviceInstance";
  static constexpr ::llvm::StringLiteral dialectName = "Aie";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"DeviceInstance"};
  }

};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::KernelFuncNameType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::KernelFileNameType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::WindowType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::KernelType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::TileKernelType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::LocType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::CoreConstraintType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::RoutingInstanceType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::HostAieConstraintType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::RoutingConstraintType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::DeviceInstanceType)

#endif // GET_TYPEDEF_CLASSES

