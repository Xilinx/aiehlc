/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: aieops.td                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir::aie {

/// Allocates a buffer on the Host and returns a memref to the allocated memory.
/// This operation allocates a buffer of the specified size in AIE Host Memory. 
/// The result is a memref pointing to the allocated memory. The type of the 
/// memref (including its shape and element type) must be specified when 
/// creating this operation.
class AllocOp;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class ConfigHostEdgeConstraints;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class CoreWait;

} // namespace mlir::aie
namespace mlir::aie {

/// Create kernel file name object
/// create the kernel file name op
class CreateKernelFileNameOp;

} // namespace mlir::aie
namespace mlir::aie {

/// Create kernel name object
/// create the kernel name op
class CreateKernelFuncNameOp;

} // namespace mlir::aie
namespace mlir::aie {

/// Creates a kernel object
/// This operation creates a kernel object with the number of input and output arguments,
/// and a variadic list of windows for the arguments.
class CreateKernelObjectOp;

} // namespace mlir::aie
namespace mlir::aie {

/// Creates a Tile kernel object
/// This operation creates a tile kernel object with row, column and kernel source.
class CreateTileKernelObjectOp;

} // namespace mlir::aie
namespace mlir::aie {

/// Compute a custom operation that produces an Aie_WindowType
/// This operation computes the specified Aie_WindowType
class CreateWindowOp;

} // namespace mlir::aie
namespace mlir::aie {

/// Removes the Route between two tiles
class DeRoute;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class InitRoutingHandler;

} // namespace mlir::aie
namespace mlir::aie {

/// Represents a call to an AIE kernel with location and arguments
class KernelCall;

} // namespace mlir::aie
namespace mlir::aie {

/// Defines a kernel with a variable number of windows (window type).
class KernelDefinition;

} // namespace mlir::aie
namespace mlir::aie {

/// Represents a call to an AIE kernel with location and arguments
class LoadKernel;

} // namespace mlir::aie
namespace mlir::aie {

/// aie op
/// "This is my custom operation."
class LoadKernelOp;

} // namespace mlir::aie
namespace mlir::aie {

/// Moves Data from one tile to another.
class MoveData;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class ResetHostEdgeConstraints;

} // namespace mlir::aie
namespace mlir::aie {

/// Finds route between two AIE tiles.
class Route;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class RouteDmaWait;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class RoutesReveal;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class RoutingInstanceFree;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class Run;

} // namespace mlir::aie
namespace mlir::aie {

/// ...
class SetCoreExecute;

} // namespace mlir::aie
namespace mlir::aie {

/// Custom operation with int32 attribute and i32 output
/// An operation that takes an int32 attribute input and produces an
/// i32 value as a result.
class ConstantInteger32Op;

} // namespace mlir::aie
namespace mlir::aie {

/// Custom operation with int32 attribute and i32 output
/// An operation that takes an int32 attribute input and produces an
/// i32 value as a result.
class ConstantInteger64Op;

} // namespace mlir::aie
namespace mlir::aie {

/// constant
/// Constant operation turns a literal into an SSA value. The data is attached
/// to the operation as an attribute. For example:
/// 
/// ```mlir
///   %0 = aie.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
///                     : tensor<2x3xf64>
/// ```
class ConstantOp;

} // namespace mlir::aie
#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::AllocOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class AllocOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  AllocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.alloc_op", odsAttrs.getContext());
  }

  AllocOpGenericAdaptorBase(AllocOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class AllocOpGenericAdaptor : public detail::AllocOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocOpGenericAdaptorBase;
  static constexpr int odsIndex_size = 0;
public:
  AllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : AllocOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  AllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : AllocOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  AllocOpGenericAdaptor(RangeT values, const AllocOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = AllocOp, typename = std::enable_if_t<std::is_same_v<LateInst, AllocOp>>>
  AllocOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSize() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllocOpAdaptor : public AllocOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllocOpGenericAdaptor::AllocOpGenericAdaptor;
  AllocOpAdaptor(AllocOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class AllocOp : public ::mlir::Op<AllocOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_size = 0;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.alloc_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::IntegerType> getSize() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getSizeMutable() {
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

  ::mlir::TypedValue<::mlir::MemRefType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value size);
  static AllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value size);
  static AllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value size);
  static AllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value size);
  static AllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value size);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static AllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllocOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static AllocOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::AllocOp)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ConfigHostEdgeConstraints declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConfigHostEdgeConstraintsGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConfigHostEdgeConstraintsGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.config_host_edge_constraints_op", odsAttrs.getContext());
  }

  ConfigHostEdgeConstraintsGenericAdaptorBase(ConfigHostEdgeConstraints op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ConfigHostEdgeConstraintsGenericAdaptor : public detail::ConfigHostEdgeConstraintsGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConfigHostEdgeConstraintsGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_userResCon = 1;
public:
  ConfigHostEdgeConstraintsGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConfigHostEdgeConstraintsGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ConfigHostEdgeConstraintsGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConfigHostEdgeConstraintsGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConfigHostEdgeConstraintsGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConfigHostEdgeConstraintsGenericAdaptor(RangeT values, const ConfigHostEdgeConstraintsGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConfigHostEdgeConstraints, typename = std::enable_if_t<std::is_same_v<LateInst, ConfigHostEdgeConstraints>>>
  ConfigHostEdgeConstraintsGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getUserResCon() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConfigHostEdgeConstraintsAdaptor : public ConfigHostEdgeConstraintsGenericAdaptor<::mlir::ValueRange> {
public:
  using ConfigHostEdgeConstraintsGenericAdaptor::ConfigHostEdgeConstraintsGenericAdaptor;
  ConfigHostEdgeConstraintsAdaptor(ConfigHostEdgeConstraints op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConfigHostEdgeConstraints : public ::mlir::Op<ConfigHostEdgeConstraints, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConfigHostEdgeConstraintsAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConfigHostEdgeConstraintsGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_userResCon = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.config_host_edge_constraints_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::HostAieConstraintType> getUserResCon() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::HostAieConstraintType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getUserResConMutable() {
    auto range = getODSOperandIndexAndLength(1);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value userResCon);
  static ConfigHostEdgeConstraints create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value userResCon);
  static ConfigHostEdgeConstraints create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value userResCon);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value userResCon);
  static ConfigHostEdgeConstraints create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value userResCon);
  static ConfigHostEdgeConstraints create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value userResCon);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigHostEdgeConstraints create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ConfigHostEdgeConstraints create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigHostEdgeConstraints create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConfigHostEdgeConstraints create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::ConfigHostEdgeConstraints)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CoreWait declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CoreWaitGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  CoreWaitGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.core_wait_op", odsAttrs.getContext());
  }

  CoreWaitGenericAdaptorBase(CoreWait op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class CoreWaitGenericAdaptor : public detail::CoreWaitGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CoreWaitGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
public:
  CoreWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CoreWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : CoreWaitGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CoreWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CoreWaitGenericAdaptor(values, attrs, Properties{}, {}) {}

  CoreWaitGenericAdaptor(RangeT values, const CoreWaitGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CoreWait, typename = std::enable_if_t<std::is_same_v<LateInst, CoreWait>>>
  CoreWaitGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTile() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CoreWaitAdaptor : public CoreWaitGenericAdaptor<::mlir::ValueRange> {
public:
  using CoreWaitGenericAdaptor::CoreWaitGenericAdaptor;
  CoreWaitAdaptor(CoreWait op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CoreWait : public ::mlir::Op<CoreWait, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CoreWaitAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CoreWaitGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.core_wait_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getTile() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTileMutable() {
    auto range = getODSOperandIndexAndLength(1);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile);
  static CoreWait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile);
  static CoreWait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static CoreWait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static CoreWait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CoreWait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static CoreWait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CoreWait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CoreWait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::CoreWait)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelFileNameOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CreateKernelFileNameOpGenericAdaptorBase {
public:
  struct Properties {
    using nameTy = ::mlir::StringAttr;
    nameTy name;

    auto getName() const {
      auto &propStorage = this->name;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setName(const ::mlir::StringAttr &propValue) {
      this->name = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.name == this->name &&
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
  CreateKernelFileNameOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.create_kernelfilename_op", odsAttrs.getContext());
  }

  CreateKernelFileNameOpGenericAdaptorBase(CreateKernelFileNameOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getNameAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().name);
    return attr;
  }

  ::llvm::StringRef getName();
};

} // namespace detail
template <typename RangeT>
class CreateKernelFileNameOpGenericAdaptor : public detail::CreateKernelFileNameOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CreateKernelFileNameOpGenericAdaptorBase;
public:
  CreateKernelFileNameOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CreateKernelFileNameOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : CreateKernelFileNameOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CreateKernelFileNameOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CreateKernelFileNameOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CreateKernelFileNameOpGenericAdaptor(RangeT values, const CreateKernelFileNameOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CreateKernelFileNameOp, typename = std::enable_if_t<std::is_same_v<LateInst, CreateKernelFileNameOp>>>
  CreateKernelFileNameOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class CreateKernelFileNameOpAdaptor : public CreateKernelFileNameOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CreateKernelFileNameOpGenericAdaptor::CreateKernelFileNameOpGenericAdaptor;
  CreateKernelFileNameOpAdaptor(CreateKernelFileNameOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CreateKernelFileNameOp : public ::mlir::Op<CreateKernelFileNameOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aie::KernelFileNameType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateKernelFileNameOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CreateKernelFileNameOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.create_kernelfilename_op");
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

  ::mlir::TypedValue<::mlir::aie::KernelFileNameType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::KernelFileNameType>>(*getODSResults(0).begin());
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
  ::mlir::StringAttr getNameAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().name);
  }

  ::llvm::StringRef getName();
  void setNameAttr(::mlir::StringAttr attr) {
    getProperties().name = attr;
  }

  void setName(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr name);
  static CreateKernelFileNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::StringAttr name);
  static CreateKernelFileNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name);
  static CreateKernelFileNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name);
  static CreateKernelFileNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name);
  static CreateKernelFileNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::llvm::StringRef name);
  static CreateKernelFileNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name);
  static CreateKernelFileNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::StringRef name);
  static CreateKernelFileNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::StringRef name);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateKernelFileNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateKernelFileNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateKernelFileNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateKernelFileNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelFileNameOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateKernelFileNameOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelFuncNameOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CreateKernelFuncNameOpGenericAdaptorBase {
public:
  struct Properties {
    using nameTy = ::mlir::StringAttr;
    nameTy name;

    auto getName() const {
      auto &propStorage = this->name;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setName(const ::mlir::StringAttr &propValue) {
      this->name = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.name == this->name &&
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
  CreateKernelFuncNameOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.create_kernelfuncname_op", odsAttrs.getContext());
  }

  CreateKernelFuncNameOpGenericAdaptorBase(CreateKernelFuncNameOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getNameAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().name);
    return attr;
  }

  ::llvm::StringRef getName();
};

} // namespace detail
template <typename RangeT>
class CreateKernelFuncNameOpGenericAdaptor : public detail::CreateKernelFuncNameOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CreateKernelFuncNameOpGenericAdaptorBase;
public:
  CreateKernelFuncNameOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CreateKernelFuncNameOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : CreateKernelFuncNameOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CreateKernelFuncNameOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CreateKernelFuncNameOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CreateKernelFuncNameOpGenericAdaptor(RangeT values, const CreateKernelFuncNameOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CreateKernelFuncNameOp, typename = std::enable_if_t<std::is_same_v<LateInst, CreateKernelFuncNameOp>>>
  CreateKernelFuncNameOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class CreateKernelFuncNameOpAdaptor : public CreateKernelFuncNameOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CreateKernelFuncNameOpGenericAdaptor::CreateKernelFuncNameOpGenericAdaptor;
  CreateKernelFuncNameOpAdaptor(CreateKernelFuncNameOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CreateKernelFuncNameOp : public ::mlir::Op<CreateKernelFuncNameOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aie::KernelFuncNameType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateKernelFuncNameOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CreateKernelFuncNameOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.create_kernelfuncname_op");
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

  ::mlir::TypedValue<::mlir::aie::KernelFuncNameType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::KernelFuncNameType>>(*getODSResults(0).begin());
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
  ::mlir::StringAttr getNameAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().name);
  }

  ::llvm::StringRef getName();
  void setNameAttr(::mlir::StringAttr attr) {
    getProperties().name = attr;
  }

  void setName(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr name);
  static CreateKernelFuncNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::StringAttr name);
  static CreateKernelFuncNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name);
  static CreateKernelFuncNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name);
  static CreateKernelFuncNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name);
  static CreateKernelFuncNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::llvm::StringRef name);
  static CreateKernelFuncNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name);
  static CreateKernelFuncNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::StringRef name);
  static CreateKernelFuncNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::StringRef name);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateKernelFuncNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateKernelFuncNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateKernelFuncNameOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateKernelFuncNameOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelFuncNameOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateKernelFuncNameOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateKernelObjectOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CreateKernelObjectOpGenericAdaptorBase {
public:
  struct Properties {
    using in_param_typeTy = ::mlir::StringAttr;
    in_param_typeTy in_param_type;

    auto getInParamType() const {
      auto &propStorage = this->in_param_type;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setInParamType(const ::mlir::StringAttr &propValue) {
      this->in_param_type = propValue;
    }
    using numInputArgsTy = ::mlir::IntegerAttr;
    numInputArgsTy numInputArgs;

    auto getNumInputArgs() const {
      auto &propStorage = this->numInputArgs;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setNumInputArgs(const ::mlir::IntegerAttr &propValue) {
      this->numInputArgs = propValue;
    }
    using numOutputArgsTy = ::mlir::IntegerAttr;
    numOutputArgsTy numOutputArgs;

    auto getNumOutputArgs() const {
      auto &propStorage = this->numOutputArgs;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setNumOutputArgs(const ::mlir::IntegerAttr &propValue) {
      this->numOutputArgs = propValue;
    }
    using out_param_typeTy = ::mlir::StringAttr;
    out_param_typeTy out_param_type;

    auto getOutParamType() const {
      auto &propStorage = this->out_param_type;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setOutParamType(const ::mlir::StringAttr &propValue) {
      this->out_param_type = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.in_param_type == this->in_param_type &&
        rhs.numInputArgs == this->numInputArgs &&
        rhs.numOutputArgs == this->numOutputArgs &&
        rhs.out_param_type == this->out_param_type &&
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
  CreateKernelObjectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.create_kernelobject_op", odsAttrs.getContext());
  }

  CreateKernelObjectOpGenericAdaptorBase(CreateKernelObjectOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getNumInputArgsAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().numInputArgs);
    return attr;
  }

  uint32_t getNumInputArgs();
  ::mlir::IntegerAttr getNumOutputArgsAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().numOutputArgs);
    return attr;
  }

  uint32_t getNumOutputArgs();
  ::mlir::StringAttr getInParamTypeAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().in_param_type);
    return attr;
  }

  ::llvm::StringRef getInParamType();
  ::mlir::StringAttr getOutParamTypeAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().out_param_type);
    return attr;
  }

  ::llvm::StringRef getOutParamType();
};

} // namespace detail
template <typename RangeT>
class CreateKernelObjectOpGenericAdaptor : public detail::CreateKernelObjectOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CreateKernelObjectOpGenericAdaptorBase;
  static constexpr int odsIndex_kname = 0;
  static constexpr int odsIndex_fname = 1;
  static constexpr int odsIndex_arguments = 2;
public:
  CreateKernelObjectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CreateKernelObjectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : CreateKernelObjectOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CreateKernelObjectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CreateKernelObjectOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CreateKernelObjectOpGenericAdaptor(RangeT values, const CreateKernelObjectOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CreateKernelObjectOp, typename = std::enable_if_t<std::is_same_v<LateInst, CreateKernelObjectOp>>>
  CreateKernelObjectOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getKname() {
    return (*getODSOperands(0).begin());
  }

  ValueT getFname() {
    return (*getODSOperands(1).begin());
  }

  RangeT getArguments() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CreateKernelObjectOpAdaptor : public CreateKernelObjectOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CreateKernelObjectOpGenericAdaptor::CreateKernelObjectOpGenericAdaptor;
  CreateKernelObjectOpAdaptor(CreateKernelObjectOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CreateKernelObjectOp : public ::mlir::Op<CreateKernelObjectOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aie::KernelType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateKernelObjectOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CreateKernelObjectOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_kname = 0;
  static constexpr int odsIndex_fname = 1;
  static constexpr int odsIndex_arguments = 2;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("in_param_type"), ::llvm::StringRef("numInputArgs"), ::llvm::StringRef("numOutputArgs"), ::llvm::StringRef("out_param_type")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInParamTypeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInParamTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNumInputArgsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNumInputArgsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getNumOutputArgsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getNumOutputArgsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutParamTypeAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutParamTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.create_kernelobject_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::KernelFuncNameType> getKname() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::KernelFuncNameType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::KernelFileNameType> getFname() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::KernelFileNameType>>(*getODSOperands(1).begin());
  }

  ::mlir::Operation::operand_range getArguments() {
    return getODSOperands(2);
  }

  ::mlir::OpOperand &getKnameMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getFnameMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getArgumentsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::aie::KernelType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::KernelType>>(*getODSResults(0).begin());
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
  ::mlir::IntegerAttr getNumInputArgsAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().numInputArgs);
  }

  uint32_t getNumInputArgs();
  ::mlir::IntegerAttr getNumOutputArgsAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().numOutputArgs);
  }

  uint32_t getNumOutputArgs();
  ::mlir::StringAttr getInParamTypeAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().in_param_type);
  }

  ::llvm::StringRef getInParamType();
  ::mlir::StringAttr getOutParamTypeAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().out_param_type);
  }

  ::llvm::StringRef getOutParamType();
  void setNumInputArgsAttr(::mlir::IntegerAttr attr) {
    getProperties().numInputArgs = attr;
  }

  void setNumInputArgs(uint32_t attrValue);
  void setNumOutputArgsAttr(::mlir::IntegerAttr attr) {
    getProperties().numOutputArgs = attr;
  }

  void setNumOutputArgs(uint32_t attrValue);
  void setInParamTypeAttr(::mlir::StringAttr attr) {
    getProperties().in_param_type = attr;
  }

  void setInParamType(::llvm::StringRef attrValue);
  void setOutParamTypeAttr(::mlir::StringAttr attr) {
    getProperties().out_param_type = attr;
  }

  void setOutParamType(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, IntegerAttr numInputArgs, IntegerAttr numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, std::string in_param_type, std::string out_param_type, ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, Type result_type, IntegerAttr numInputArgs, IntegerAttr numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, std::string in_param_type, std::string out_param_type, ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, Type result_type, IntegerAttr numInputArgs, IntegerAttr numOutputArgs, KernelFuncNameType kname, KernelFileNameType fname, std::string in_param_type, std::string out_param_type, ValueRange arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr numInputArgs, ::mlir::IntegerAttr numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::mlir::StringAttr in_param_type, ::mlir::StringAttr out_param_type, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments);
  static CreateKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t numInputArgs, uint32_t numOutputArgs, ::mlir::Value kname, ::mlir::Value fname, ::llvm::StringRef in_param_type, ::llvm::StringRef out_param_type, ::mlir::ValueRange arguments);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::CreateKernelObjectOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateKernelObjectOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateTileKernelObjectOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CreateTileKernelObjectOpGenericAdaptorBase {
public:
  struct Properties {
    using kernelsTy = ::mlir::StringAttr;
    kernelsTy kernels;

    auto getKernels() const {
      auto &propStorage = this->kernels;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setKernels(const ::mlir::StringAttr &propValue) {
      this->kernels = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.kernels == this->kernels &&
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
  CreateTileKernelObjectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.create_tilekernelobject_op", odsAttrs.getContext());
  }

  CreateTileKernelObjectOpGenericAdaptorBase(CreateTileKernelObjectOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getKernelsAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().kernels);
    return attr;
  }

  ::llvm::StringRef getKernels();
};

} // namespace detail
template <typename RangeT>
class CreateTileKernelObjectOpGenericAdaptor : public detail::CreateTileKernelObjectOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CreateTileKernelObjectOpGenericAdaptorBase;
  static constexpr int odsIndex_row = 0;
  static constexpr int odsIndex_col = 1;
public:
  CreateTileKernelObjectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CreateTileKernelObjectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : CreateTileKernelObjectOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CreateTileKernelObjectOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CreateTileKernelObjectOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CreateTileKernelObjectOpGenericAdaptor(RangeT values, const CreateTileKernelObjectOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CreateTileKernelObjectOp, typename = std::enable_if_t<std::is_same_v<LateInst, CreateTileKernelObjectOp>>>
  CreateTileKernelObjectOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRow() {
    return (*getODSOperands(0).begin());
  }

  ValueT getCol() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CreateTileKernelObjectOpAdaptor : public CreateTileKernelObjectOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CreateTileKernelObjectOpGenericAdaptor::CreateTileKernelObjectOpGenericAdaptor;
  CreateTileKernelObjectOpAdaptor(CreateTileKernelObjectOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CreateTileKernelObjectOp : public ::mlir::Op<CreateTileKernelObjectOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aie::TileKernelType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateTileKernelObjectOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CreateTileKernelObjectOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_row = 0;
  static constexpr int odsIndex_col = 1;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kernels")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getKernelsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getKernelsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.create_tilekernelobject_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::IntegerType> getRow() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getCol() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getRowMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getColMutable() {
    auto range = getODSOperandIndexAndLength(1);
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

  ::mlir::TypedValue<::mlir::aie::TileKernelType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::TileKernelType>>(*getODSResults(0).begin());
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
  ::mlir::StringAttr getKernelsAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().kernels);
  }

  ::llvm::StringRef getKernels();
  void setKernelsAttr(::mlir::StringAttr attr) {
    getProperties().kernels = attr;
  }

  void setKernels(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels);
  static CreateTileKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels);
  static CreateTileKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels);
  static CreateTileKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels);
  static CreateTileKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::mlir::StringAttr kernels);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels);
  static CreateTileKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels);
  static CreateTileKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels);
  static CreateTileKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels);
  static CreateTileKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value row, ::mlir::Value col, ::llvm::StringRef kernels);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateTileKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateTileKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateTileKernelObjectOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateTileKernelObjectOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::CreateTileKernelObjectOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateTileKernelObjectOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::CreateWindowOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class CreateWindowOpGenericAdaptorBase {
public:
  struct Properties {
    using directionTy = ::mlir::IntegerAttr;
    directionTy direction;

    auto getDirection() const {
      auto &propStorage = this->direction;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setDirection(const ::mlir::IntegerAttr &propValue) {
      this->direction = propValue;
    }
    using nameTy = ::mlir::StringAttr;
    nameTy name;

    auto getName() const {
      auto &propStorage = this->name;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setName(const ::mlir::StringAttr &propValue) {
      this->name = propValue;
    }
    using pingaddrTy = ::mlir::IntegerAttr;
    pingaddrTy pingaddr;

    auto getPingaddr() const {
      auto &propStorage = this->pingaddr;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setPingaddr(const ::mlir::IntegerAttr &propValue) {
      this->pingaddr = propValue;
    }
    using pinglockidTy = ::mlir::IntegerAttr;
    pinglockidTy pinglockid;

    auto getPinglockid() const {
      auto &propStorage = this->pinglockid;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setPinglockid(const ::mlir::IntegerAttr &propValue) {
      this->pinglockid = propValue;
    }
    using pongaddrTy = ::mlir::IntegerAttr;
    pongaddrTy pongaddr;

    auto getPongaddr() const {
      auto &propStorage = this->pongaddr;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setPongaddr(const ::mlir::IntegerAttr &propValue) {
      this->pongaddr = propValue;
    }
    using ponglockidTy = ::mlir::IntegerAttr;
    ponglockidTy ponglockid;

    auto getPonglockid() const {
      auto &propStorage = this->ponglockid;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setPonglockid(const ::mlir::IntegerAttr &propValue) {
      this->ponglockid = propValue;
    }
    using sizeTy = ::mlir::IntegerAttr;
    sizeTy size;

    auto getSize() const {
      auto &propStorage = this->size;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setSize(const ::mlir::IntegerAttr &propValue) {
      this->size = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.direction == this->direction &&
        rhs.name == this->name &&
        rhs.pingaddr == this->pingaddr &&
        rhs.pinglockid == this->pinglockid &&
        rhs.pongaddr == this->pongaddr &&
        rhs.ponglockid == this->ponglockid &&
        rhs.size == this->size &&
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
  CreateWindowOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.create_window", odsAttrs.getContext());
  }

  CreateWindowOpGenericAdaptorBase(CreateWindowOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getDirectionAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().direction);
    return attr;
  }

  uint32_t getDirection();
  ::mlir::IntegerAttr getSizeAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().size);
    return attr;
  }

  uint32_t getSize();
  ::mlir::IntegerAttr getPingaddrAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().pingaddr);
    return attr;
  }

  uint64_t getPingaddr();
  ::mlir::IntegerAttr getPongaddrAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().pongaddr);
    return attr;
  }

  uint64_t getPongaddr();
  ::mlir::IntegerAttr getPinglockidAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().pinglockid);
    return attr;
  }

  uint32_t getPinglockid();
  ::mlir::IntegerAttr getPonglockidAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().ponglockid);
    return attr;
  }

  uint32_t getPonglockid();
  ::mlir::StringAttr getNameAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().name);
    return attr;
  }

  ::llvm::StringRef getName();
};

} // namespace detail
template <typename RangeT>
class CreateWindowOpGenericAdaptor : public detail::CreateWindowOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CreateWindowOpGenericAdaptorBase;
public:
  CreateWindowOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  CreateWindowOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : CreateWindowOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  CreateWindowOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : CreateWindowOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  CreateWindowOpGenericAdaptor(RangeT values, const CreateWindowOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = CreateWindowOp, typename = std::enable_if_t<std::is_same_v<LateInst, CreateWindowOp>>>
  CreateWindowOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class CreateWindowOpAdaptor : public CreateWindowOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CreateWindowOpGenericAdaptor::CreateWindowOpGenericAdaptor;
  CreateWindowOpAdaptor(CreateWindowOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class CreateWindowOp : public ::mlir::Op<CreateWindowOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aie::WindowType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CreateWindowOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CreateWindowOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_result = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("direction"), ::llvm::StringRef("name"), ::llvm::StringRef("pingaddr"), ::llvm::StringRef("pinglockid"), ::llvm::StringRef("pongaddr"), ::llvm::StringRef("ponglockid"), ::llvm::StringRef("size")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getDirectionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDirectionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getPingaddrAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getPingaddrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getPinglockidAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getPinglockidAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getPongaddrAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getPongaddrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getPonglockidAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getPonglockidAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getSizeAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.create_window");
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

  ::mlir::TypedValue<::mlir::aie::WindowType> getResult() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::WindowType>>(*getODSResults(0).begin());
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
  ::mlir::IntegerAttr getDirectionAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().direction);
  }

  uint32_t getDirection();
  ::mlir::IntegerAttr getSizeAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().size);
  }

  uint32_t getSize();
  ::mlir::IntegerAttr getPingaddrAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().pingaddr);
  }

  uint64_t getPingaddr();
  ::mlir::IntegerAttr getPongaddrAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().pongaddr);
  }

  uint64_t getPongaddr();
  ::mlir::IntegerAttr getPinglockidAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().pinglockid);
  }

  uint32_t getPinglockid();
  ::mlir::IntegerAttr getPonglockidAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().ponglockid);
  }

  uint32_t getPonglockid();
  ::mlir::StringAttr getNameAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().name);
  }

  ::llvm::StringRef getName();
  void setDirectionAttr(::mlir::IntegerAttr attr) {
    getProperties().direction = attr;
  }

  void setDirection(uint32_t attrValue);
  void setSizeAttr(::mlir::IntegerAttr attr) {
    getProperties().size = attr;
  }

  void setSize(uint32_t attrValue);
  void setPingaddrAttr(::mlir::IntegerAttr attr) {
    getProperties().pingaddr = attr;
  }

  void setPingaddr(uint64_t attrValue);
  void setPongaddrAttr(::mlir::IntegerAttr attr) {
    getProperties().pongaddr = attr;
  }

  void setPongaddr(uint64_t attrValue);
  void setPinglockidAttr(::mlir::IntegerAttr attr) {
    getProperties().pinglockid = attr;
  }

  void setPinglockid(uint32_t attrValue);
  void setPonglockidAttr(::mlir::IntegerAttr attr) {
    getProperties().ponglockid = attr;
  }

  void setPonglockid(uint32_t attrValue);
  void setNameAttr(::mlir::StringAttr attr) {
    getProperties().name = attr;
  }

  void setName(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name);
  static CreateWindowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name);
  static CreateWindowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name);
  static CreateWindowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name);
  static CreateWindowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr direction, ::mlir::IntegerAttr size, ::mlir::IntegerAttr pingaddr, ::mlir::IntegerAttr pongaddr, ::mlir::IntegerAttr pinglockid, ::mlir::IntegerAttr ponglockid, ::mlir::StringAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name);
  static CreateWindowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type result, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name);
  static CreateWindowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type result, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name);
  static CreateWindowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name);
  static CreateWindowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t direction, uint32_t size, uint64_t pingaddr, uint64_t pongaddr, uint32_t pinglockid, uint32_t ponglockid, ::llvm::StringRef name);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateWindowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static CreateWindowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateWindowOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static CreateWindowOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 7 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::CreateWindowOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::CreateWindowOpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::DeRoute declarations
//===----------------------------------------------------------------------===//

namespace detail {

class DeRouteGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  DeRouteGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.de_route_op", odsAttrs.getContext());
  }

  DeRouteGenericAdaptorBase(DeRoute op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class DeRouteGenericAdaptor : public detail::DeRouteGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DeRouteGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_source = 1;
  static constexpr int odsIndex_destination = 2;
  static constexpr int odsIndex_shouldModifyCoreConfig = 3;
public:
  DeRouteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  DeRouteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : DeRouteGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  DeRouteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : DeRouteGenericAdaptor(values, attrs, Properties{}, {}) {}

  DeRouteGenericAdaptor(RangeT values, const DeRouteGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = DeRoute, typename = std::enable_if_t<std::is_same_v<LateInst, DeRoute>>>
  DeRouteGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getSource() {
    return (*getODSOperands(1).begin());
  }

  ValueT getDestination() {
    return (*getODSOperands(2).begin());
  }

  ValueT getShouldModifyCoreConfig() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DeRouteAdaptor : public DeRouteGenericAdaptor<::mlir::ValueRange> {
public:
  using DeRouteGenericAdaptor::DeRouteGenericAdaptor;
  DeRouteAdaptor(DeRoute op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class DeRoute : public ::mlir::Op<DeRoute, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeRouteAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DeRouteGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_source = 1;
  static constexpr int odsIndex_destination = 2;
  static constexpr int odsIndex_shouldModifyCoreConfig = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.de_route_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getSource() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getDestination() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getShouldModifyCoreConfig() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getSourceMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDestinationMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getShouldModifyCoreConfigMutable() {
    auto range = getODSOperandIndexAndLength(3);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig);
  static DeRoute create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig);
  static DeRoute create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig);
  static DeRoute create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig);
  static DeRoute create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination, ::mlir::Value shouldModifyCoreConfig);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DeRoute create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static DeRoute create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DeRoute create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static DeRoute create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::DeRoute)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::InitRoutingHandler declarations
//===----------------------------------------------------------------------===//

namespace detail {

class InitRoutingHandlerGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  InitRoutingHandlerGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.init_routing_handler", odsAttrs.getContext());
  }

  InitRoutingHandlerGenericAdaptorBase(InitRoutingHandler op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class InitRoutingHandlerGenericAdaptor : public detail::InitRoutingHandlerGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InitRoutingHandlerGenericAdaptorBase;
  static constexpr int odsIndex_devInst = 0;
public:
  InitRoutingHandlerGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  InitRoutingHandlerGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : InitRoutingHandlerGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  InitRoutingHandlerGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : InitRoutingHandlerGenericAdaptor(values, attrs, Properties{}, {}) {}

  InitRoutingHandlerGenericAdaptor(RangeT values, const InitRoutingHandlerGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = InitRoutingHandler, typename = std::enable_if_t<std::is_same_v<LateInst, InitRoutingHandler>>>
  InitRoutingHandlerGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getDevInst() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InitRoutingHandlerAdaptor : public InitRoutingHandlerGenericAdaptor<::mlir::ValueRange> {
public:
  using InitRoutingHandlerGenericAdaptor::InitRoutingHandlerGenericAdaptor;
  InitRoutingHandlerAdaptor(InitRoutingHandler op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class InitRoutingHandler : public ::mlir::Op<InitRoutingHandler, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::aie::RoutingInstanceType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InitRoutingHandlerAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InitRoutingHandlerGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_devInst = 0;
  static constexpr int odsIndex_instance = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.init_routing_handler");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::DeviceInstanceType> getDevInst() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::DeviceInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getDevInstMutable() {
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

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSResults(0).begin());
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type instance, ::mlir::Value devInst);
  static InitRoutingHandler create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type instance, ::mlir::Value devInst);
  static InitRoutingHandler create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type instance, ::mlir::Value devInst);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value devInst);
  static InitRoutingHandler create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value devInst);
  static InitRoutingHandler create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value devInst);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static InitRoutingHandler create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static InitRoutingHandler create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static InitRoutingHandler create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static InitRoutingHandler create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::InitRoutingHandler)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::KernelCall declarations
//===----------------------------------------------------------------------===//

namespace detail {

class KernelCallGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  KernelCallGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.kernel_call_op", odsAttrs.getContext());
  }

  KernelCallGenericAdaptorBase(KernelCall op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class KernelCallGenericAdaptor : public detail::KernelCallGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::KernelCallGenericAdaptorBase;
  static constexpr int odsIndex_col = 0;
  static constexpr int odsIndex_row = 1;
  static constexpr int odsIndex_arguments = 2;
public:
  KernelCallGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  KernelCallGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : KernelCallGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  KernelCallGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : KernelCallGenericAdaptor(values, attrs, Properties{}, {}) {}

  KernelCallGenericAdaptor(RangeT values, const KernelCallGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = KernelCall, typename = std::enable_if_t<std::is_same_v<LateInst, KernelCall>>>
  KernelCallGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCol() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRow() {
    return (*getODSOperands(1).begin());
  }

  RangeT getArguments() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class KernelCallAdaptor : public KernelCallGenericAdaptor<::mlir::ValueRange> {
public:
  using KernelCallGenericAdaptor::KernelCallGenericAdaptor;
  KernelCallAdaptor(KernelCall op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class KernelCall : public ::mlir::Op<KernelCall, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = KernelCallAdaptor;
  template <typename RangeT>
  using GenericAdaptor = KernelCallGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_col = 0;
  static constexpr int odsIndex_row = 1;
  static constexpr int odsIndex_arguments = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.kernel_call_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::IntegerType> getCol() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getRow() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::Operation::operand_range getArguments() {
    return getODSOperands(2);
  }

  ::mlir::OpOperand &getColMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRowMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::MutableOperandRange getArgumentsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, Value col, Value row, ArrayRef<Value> arguments);
  static KernelCall create(::mlir::OpBuilder &builder, ::mlir::Location location, Type result_type, Value col, Value row, ArrayRef<Value> arguments);
  static KernelCall create(::mlir::ImplicitLocOpBuilder &builder, Type result_type, Value col, Value row, ArrayRef<Value> arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments);
  static KernelCall create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments);
  static KernelCall create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments);
  static KernelCall create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments);
  static KernelCall create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value col, ::mlir::Value row, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static KernelCall create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static KernelCall create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static KernelCall create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static KernelCall create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::KernelCall)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::KernelDefinition declarations
//===----------------------------------------------------------------------===//

namespace detail {

class KernelDefinitionGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  KernelDefinitionGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.kernel_definition_op", odsAttrs.getContext());
  }

  KernelDefinitionGenericAdaptorBase(KernelDefinition op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class KernelDefinitionGenericAdaptor : public detail::KernelDefinitionGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::KernelDefinitionGenericAdaptorBase;
  static constexpr int odsIndex_arguments = 0;
public:
  KernelDefinitionGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  KernelDefinitionGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : KernelDefinitionGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  KernelDefinitionGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : KernelDefinitionGenericAdaptor(values, attrs, Properties{}, {}) {}

  KernelDefinitionGenericAdaptor(RangeT values, const KernelDefinitionGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = KernelDefinition, typename = std::enable_if_t<std::is_same_v<LateInst, KernelDefinition>>>
  KernelDefinitionGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getArguments() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class KernelDefinitionAdaptor : public KernelDefinitionGenericAdaptor<::mlir::ValueRange> {
public:
  using KernelDefinitionGenericAdaptor::KernelDefinitionGenericAdaptor;
  KernelDefinitionAdaptor(KernelDefinition op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class KernelDefinition : public ::mlir::Op<KernelDefinition, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = KernelDefinitionAdaptor;
  template <typename RangeT>
  using GenericAdaptor = KernelDefinitionGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_arguments = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.kernel_definition_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::Operation::operand_range getArguments() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getArgumentsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result_type, ArrayRef<Value> arguments);
  static KernelDefinition create(::mlir::OpBuilder &builder, ::mlir::Location location, Type result_type, ArrayRef<Value> arguments);
  static KernelDefinition create(::mlir::ImplicitLocOpBuilder &builder, Type result_type, ArrayRef<Value> arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange arguments);
  static KernelDefinition create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::ValueRange arguments);
  static KernelDefinition create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static KernelDefinition create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static KernelDefinition create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static KernelDefinition create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static KernelDefinition create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::KernelDefinition)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::LoadKernel declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoadKernelGenericAdaptorBase {
public:
  struct Properties {
    using colTy = ::mlir::IntegerAttr;
    colTy col;

    auto getCol() const {
      auto &propStorage = this->col;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setCol(const ::mlir::IntegerAttr &propValue) {
      this->col = propValue;
    }
    using nameTy = ::mlir::StringAttr;
    nameTy name;

    auto getName() const {
      auto &propStorage = this->name;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setName(const ::mlir::StringAttr &propValue) {
      this->name = propValue;
    }
    using rowTy = ::mlir::IntegerAttr;
    rowTy row;

    auto getRow() const {
      auto &propStorage = this->row;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setRow(const ::mlir::IntegerAttr &propValue) {
      this->row = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.col == this->col &&
        rhs.name == this->name &&
        rhs.row == this->row &&
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
  LoadKernelGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.load_kernel_op", odsAttrs.getContext());
  }

  LoadKernelGenericAdaptorBase(LoadKernel op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::StringAttr getNameAttr() {
    auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().name);
    return attr;
  }

  ::llvm::StringRef getName();
  ::mlir::IntegerAttr getColAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().col);
    return attr;
  }

  uint32_t getCol();
  ::mlir::IntegerAttr getRowAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().row);
    return attr;
  }

  uint32_t getRow();
};

} // namespace detail
template <typename RangeT>
class LoadKernelGenericAdaptor : public detail::LoadKernelGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoadKernelGenericAdaptorBase;
  static constexpr int odsIndex_arguments = 0;
public:
  LoadKernelGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoadKernelGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : LoadKernelGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoadKernelGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoadKernelGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoadKernelGenericAdaptor(RangeT values, const LoadKernelGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoadKernel, typename = std::enable_if_t<std::is_same_v<LateInst, LoadKernel>>>
  LoadKernelGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getArguments() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoadKernelAdaptor : public LoadKernelGenericAdaptor<::mlir::ValueRange> {
public:
  using LoadKernelGenericAdaptor::LoadKernelGenericAdaptor;
  LoadKernelAdaptor(LoadKernel op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoadKernel : public ::mlir::Op<LoadKernel, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadKernelAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoadKernelGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_arguments = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("col"), ::llvm::StringRef("name"), ::llvm::StringRef("row")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getColAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getColAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getRowAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getRowAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.load_kernel_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::Operation::operand_range getArguments() {
    return getODSOperands(0);
  }

  ::mlir::MutableOperandRange getArgumentsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
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
  ::mlir::StringAttr getNameAttr() {
    return ::llvm::cast<::mlir::StringAttr>(getProperties().name);
  }

  ::llvm::StringRef getName();
  ::mlir::IntegerAttr getColAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().col);
  }

  uint32_t getCol();
  ::mlir::IntegerAttr getRowAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().row);
  }

  uint32_t getRow();
  void setNameAttr(::mlir::StringAttr attr) {
    getProperties().name = attr;
  }

  void setName(::llvm::StringRef attrValue);
  void setColAttr(::mlir::IntegerAttr attr) {
    getProperties().col = attr;
  }

  void setCol(uint32_t attrValue);
  void setRowAttr(::mlir::IntegerAttr attr) {
    getProperties().row = attr;
  }

  void setRow(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments);
  static LoadKernel create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments);
  static LoadKernel create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments);
  static LoadKernel create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments);
  static LoadKernel create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::IntegerAttr col, ::mlir::IntegerAttr row, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments);
  static LoadKernel create(::mlir::OpBuilder &builder, ::mlir::Location location, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments);
  static LoadKernel create(::mlir::ImplicitLocOpBuilder &builder, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments);
  static LoadKernel create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments);
  static LoadKernel create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, uint32_t col, uint32_t row, ::mlir::ValueRange arguments);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static LoadKernel create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static LoadKernel create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoadKernel create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoadKernel create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::LoadKernel)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::LoadKernelGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::LoadKernelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class LoadKernelOpGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  LoadKernelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.loadkernel", odsAttrs.getContext());
  }

  LoadKernelOpGenericAdaptorBase(LoadKernelOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class LoadKernelOpGenericAdaptor : public detail::LoadKernelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoadKernelOpGenericAdaptorBase;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
public:
  LoadKernelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  LoadKernelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : LoadKernelOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  LoadKernelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : LoadKernelOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  LoadKernelOpGenericAdaptor(RangeT values, const LoadKernelOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = LoadKernelOp, typename = std::enable_if_t<std::is_same_v<LateInst, LoadKernelOp>>>
  LoadKernelOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class LoadKernelOpAdaptor : public LoadKernelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoadKernelOpGenericAdaptor::LoadKernelOpGenericAdaptor;
  LoadKernelOpAdaptor(LoadKernelOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class LoadKernelOp : public ::mlir::Op<LoadKernelOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadKernelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoadKernelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_lhs = 0;
  static constexpr int odsIndex_rhs = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.loadkernel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::IntegerType> getLhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getRhs() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getLhsMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRhsMutable() {
    auto range = getODSOperandIndexAndLength(1);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs);
  static LoadKernelOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value lhs, ::mlir::Value rhs);
  static LoadKernelOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs);
  static LoadKernelOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs);
  static LoadKernelOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoadKernelOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static LoadKernelOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoadKernelOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static LoadKernelOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::LoadKernelOp)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::MoveData declarations
//===----------------------------------------------------------------------===//

namespace detail {

class MoveDataGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  MoveDataGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.move_data_op", odsAttrs.getContext());
  }

  MoveDataGenericAdaptorBase(MoveData op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class MoveDataGenericAdaptor : public detail::MoveDataGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MoveDataGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_source = 1;
  static constexpr int odsIndex_sourceObject = 2;
  static constexpr int odsIndex_dataSize = 3;
  static constexpr int odsIndex_destinationObject = 4;
  static constexpr int odsIndex_destination = 5;
public:
  MoveDataGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MoveDataGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : MoveDataGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  MoveDataGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : MoveDataGenericAdaptor(values, attrs, Properties{}, {}) {}

  MoveDataGenericAdaptor(RangeT values, const MoveDataGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = MoveData, typename = std::enable_if_t<std::is_same_v<LateInst, MoveData>>>
  MoveDataGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getSource() {
    return (*getODSOperands(1).begin());
  }

  ValueT getSourceObject() {
    return (*getODSOperands(2).begin());
  }

  ValueT getDataSize() {
    return (*getODSOperands(3).begin());
  }

  ValueT getDestinationObject() {
    return (*getODSOperands(4).begin());
  }

  ValueT getDestination() {
    return (*getODSOperands(5).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MoveDataAdaptor : public MoveDataGenericAdaptor<::mlir::ValueRange> {
public:
  using MoveDataGenericAdaptor::MoveDataGenericAdaptor;
  MoveDataAdaptor(MoveData op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class MoveData : public ::mlir::Op<MoveData, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<6>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MoveDataAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MoveDataGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_source = 1;
  static constexpr int odsIndex_sourceObject = 2;
  static constexpr int odsIndex_dataSize = 3;
  static constexpr int odsIndex_destinationObject = 4;
  static constexpr int odsIndex_destination = 5;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.move_data_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getSource() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getSourceObject() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getDataSize() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(3).begin());
  }

  ::mlir::TypedValue<::mlir::Type> getDestinationObject() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::Type>>(*getODSOperands(4).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getDestination() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(5).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getSourceMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getSourceObjectMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDataSizeMutable() {
    auto range = getODSOperandIndexAndLength(3);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDestinationObjectMutable() {
    auto range = getODSOperandIndexAndLength(4);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDestinationMutable() {
    auto range = getODSOperandIndexAndLength(5);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination);
  static MoveData create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination);
  static MoveData create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination);
  static MoveData create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination);
  static MoveData create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value sourceObject, ::mlir::Value dataSize, ::mlir::Value destinationObject, ::mlir::Value destination);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MoveData create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static MoveData create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static MoveData create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static MoveData create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::MoveData)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ResetHostEdgeConstraints declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ResetHostEdgeConstraintsGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ResetHostEdgeConstraintsGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.reset_host_edge_constraints_op", odsAttrs.getContext());
  }

  ResetHostEdgeConstraintsGenericAdaptorBase(ResetHostEdgeConstraints op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class ResetHostEdgeConstraintsGenericAdaptor : public detail::ResetHostEdgeConstraintsGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ResetHostEdgeConstraintsGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
public:
  ResetHostEdgeConstraintsGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ResetHostEdgeConstraintsGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ResetHostEdgeConstraintsGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ResetHostEdgeConstraintsGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ResetHostEdgeConstraintsGenericAdaptor(values, attrs, Properties{}, {}) {}

  ResetHostEdgeConstraintsGenericAdaptor(RangeT values, const ResetHostEdgeConstraintsGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ResetHostEdgeConstraints, typename = std::enable_if_t<std::is_same_v<LateInst, ResetHostEdgeConstraints>>>
  ResetHostEdgeConstraintsGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTile() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ResetHostEdgeConstraintsAdaptor : public ResetHostEdgeConstraintsGenericAdaptor<::mlir::ValueRange> {
public:
  using ResetHostEdgeConstraintsGenericAdaptor::ResetHostEdgeConstraintsGenericAdaptor;
  ResetHostEdgeConstraintsAdaptor(ResetHostEdgeConstraints op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ResetHostEdgeConstraints : public ::mlir::Op<ResetHostEdgeConstraints, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ResetHostEdgeConstraintsAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ResetHostEdgeConstraintsGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.reset_host_edge_constraints_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getTile() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTileMutable() {
    auto range = getODSOperandIndexAndLength(1);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile);
  static ResetHostEdgeConstraints create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile);
  static ResetHostEdgeConstraints create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static ResetHostEdgeConstraints create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static ResetHostEdgeConstraints create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ResetHostEdgeConstraints create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ResetHostEdgeConstraints create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ResetHostEdgeConstraints create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ResetHostEdgeConstraints create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::ResetHostEdgeConstraints)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::Route declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RouteGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  RouteGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.route_op", odsAttrs.getContext());
  }

  RouteGenericAdaptorBase(Route op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class RouteGenericAdaptor : public detail::RouteGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RouteGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_routeConstraints = 1;
  static constexpr int odsIndex_source = 2;
  static constexpr int odsIndex_destination = 3;
public:
  RouteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RouteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : RouteGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RouteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RouteGenericAdaptor(values, attrs, Properties{}, {}) {}

  RouteGenericAdaptor(RangeT values, const RouteGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Route, typename = std::enable_if_t<std::is_same_v<LateInst, Route>>>
  RouteGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRouteConstraints() {
    return (*getODSOperands(1).begin());
  }

  ValueT getSource() {
    return (*getODSOperands(2).begin());
  }

  ValueT getDestination() {
    return (*getODSOperands(3).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RouteAdaptor : public RouteGenericAdaptor<::mlir::ValueRange> {
public:
  using RouteGenericAdaptor::RouteGenericAdaptor;
  RouteAdaptor(Route op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Route : public ::mlir::Op<Route, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<4>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RouteAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RouteGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_routeConstraints = 1;
  static constexpr int odsIndex_source = 2;
  static constexpr int odsIndex_destination = 3;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.route_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::RoutingConstraintType> getRouteConstraints() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingConstraintType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getSource() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(2).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getDestination() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(3).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getRouteConstraintsMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getSourceMutable() {
    auto range = getODSOperandIndexAndLength(2);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDestinationMutable() {
    auto range = getODSOperandIndexAndLength(3);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination);
  static Route create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination);
  static Route create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination);
  static Route create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination);
  static Route create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value routeConstraints, ::mlir::Value source, ::mlir::Value destination);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Route create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Route create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Route create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Route create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::Route)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::RouteDmaWait declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RouteDmaWaitGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  RouteDmaWaitGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.route_dmawait_op", odsAttrs.getContext());
  }

  RouteDmaWaitGenericAdaptorBase(RouteDmaWait op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class RouteDmaWaitGenericAdaptor : public detail::RouteDmaWaitGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RouteDmaWaitGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
  static constexpr int odsIndex_isS2MM = 2;
public:
  RouteDmaWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RouteDmaWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : RouteDmaWaitGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RouteDmaWaitGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RouteDmaWaitGenericAdaptor(values, attrs, Properties{}, {}) {}

  RouteDmaWaitGenericAdaptor(RangeT values, const RouteDmaWaitGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = RouteDmaWait, typename = std::enable_if_t<std::is_same_v<LateInst, RouteDmaWait>>>
  RouteDmaWaitGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTile() {
    return (*getODSOperands(1).begin());
  }

  ValueT getIsS2MM() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RouteDmaWaitAdaptor : public RouteDmaWaitGenericAdaptor<::mlir::ValueRange> {
public:
  using RouteDmaWaitGenericAdaptor::RouteDmaWaitGenericAdaptor;
  RouteDmaWaitAdaptor(RouteDmaWait op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RouteDmaWait : public ::mlir::Op<RouteDmaWait, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RouteDmaWaitAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RouteDmaWaitGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
  static constexpr int odsIndex_isS2MM = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.route_dmawait_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getTile() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getIsS2MM() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTileMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getIsS2MMMutable() {
    auto range = getODSOperandIndexAndLength(2);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM);
  static RouteDmaWait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM);
  static RouteDmaWait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM);
  static RouteDmaWait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM);
  static RouteDmaWait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isS2MM);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RouteDmaWait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RouteDmaWait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RouteDmaWait create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RouteDmaWait create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::RouteDmaWait)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::RoutesReveal declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RoutesRevealGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  RoutesRevealGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.routes_reveal", odsAttrs.getContext());
  }

  RoutesRevealGenericAdaptorBase(RoutesReveal op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class RoutesRevealGenericAdaptor : public detail::RoutesRevealGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RoutesRevealGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_source = 1;
  static constexpr int odsIndex_destination = 2;
public:
  RoutesRevealGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RoutesRevealGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : RoutesRevealGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RoutesRevealGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RoutesRevealGenericAdaptor(values, attrs, Properties{}, {}) {}

  RoutesRevealGenericAdaptor(RangeT values, const RoutesRevealGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = RoutesReveal, typename = std::enable_if_t<std::is_same_v<LateInst, RoutesReveal>>>
  RoutesRevealGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getSource() {
    return (*getODSOperands(1).begin());
  }

  ValueT getDestination() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RoutesRevealAdaptor : public RoutesRevealGenericAdaptor<::mlir::ValueRange> {
public:
  using RoutesRevealGenericAdaptor::RoutesRevealGenericAdaptor;
  RoutesRevealAdaptor(RoutesReveal op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RoutesReveal : public ::mlir::Op<RoutesReveal, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RoutesRevealAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RoutesRevealGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_source = 1;
  static constexpr int odsIndex_destination = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.routes_reveal");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getSource() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getDestination() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getSourceMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getDestinationMutable() {
    auto range = getODSOperandIndexAndLength(2);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination);
  static RoutesReveal create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination);
  static RoutesReveal create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination);
  static RoutesReveal create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination);
  static RoutesReveal create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value source, ::mlir::Value destination);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RoutesReveal create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RoutesReveal create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RoutesReveal create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RoutesReveal create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::RoutesReveal)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::RoutingInstanceFree declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RoutingInstanceFreeGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  RoutingInstanceFreeGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.routing_instance_free_op", odsAttrs.getContext());
  }

  RoutingInstanceFreeGenericAdaptorBase(RoutingInstanceFree op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class RoutingInstanceFreeGenericAdaptor : public detail::RoutingInstanceFreeGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RoutingInstanceFreeGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
public:
  RoutingInstanceFreeGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RoutingInstanceFreeGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : RoutingInstanceFreeGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RoutingInstanceFreeGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RoutingInstanceFreeGenericAdaptor(values, attrs, Properties{}, {}) {}

  RoutingInstanceFreeGenericAdaptor(RangeT values, const RoutingInstanceFreeGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = RoutingInstanceFree, typename = std::enable_if_t<std::is_same_v<LateInst, RoutingInstanceFree>>>
  RoutingInstanceFreeGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RoutingInstanceFreeAdaptor : public RoutingInstanceFreeGenericAdaptor<::mlir::ValueRange> {
public:
  using RoutingInstanceFreeGenericAdaptor::RoutingInstanceFreeGenericAdaptor;
  RoutingInstanceFreeAdaptor(RoutingInstanceFree op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class RoutingInstanceFree : public ::mlir::Op<RoutingInstanceFree, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RoutingInstanceFreeAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RoutingInstanceFreeGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.routing_instance_free_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance);
  static RoutingInstanceFree create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance);
  static RoutingInstanceFree create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance);
  static RoutingInstanceFree create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance);
  static RoutingInstanceFree create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RoutingInstanceFree create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static RoutingInstanceFree create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RoutingInstanceFree create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static RoutingInstanceFree create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::RoutingInstanceFree)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::Run declarations
//===----------------------------------------------------------------------===//

namespace detail {

class RunGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  RunGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.run_op", odsAttrs.getContext());
  }

  RunGenericAdaptorBase(Run op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class RunGenericAdaptor : public detail::RunGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RunGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
public:
  RunGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  RunGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : RunGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  RunGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : RunGenericAdaptor(values, attrs, Properties{}, {}) {}

  RunGenericAdaptor(RangeT values, const RunGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = Run, typename = std::enable_if_t<std::is_same_v<LateInst, Run>>>
  RunGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTile() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RunAdaptor : public RunGenericAdaptor<::mlir::ValueRange> {
public:
  using RunGenericAdaptor::RunGenericAdaptor;
  RunAdaptor(Run op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class Run : public ::mlir::Op<Run, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RunAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RunGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.run_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getTile() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(1).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTileMutable() {
    auto range = getODSOperandIndexAndLength(1);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile);
  static Run create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile);
  static Run create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static Run create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static Run create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Run create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static Run create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Run create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static Run create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::Run)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::SetCoreExecute declarations
//===----------------------------------------------------------------------===//

namespace detail {

class SetCoreExecuteGenericAdaptorBase {
public:
  using Properties = ::mlir::EmptyProperties;
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  SetCoreExecuteGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.set_core_execute_op", odsAttrs.getContext());
  }

  SetCoreExecuteGenericAdaptorBase(SetCoreExecute op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

};

} // namespace detail
template <typename RangeT>
class SetCoreExecuteGenericAdaptor : public detail::SetCoreExecuteGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SetCoreExecuteGenericAdaptorBase;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
  static constexpr int odsIndex_isExecute = 2;
public:
  SetCoreExecuteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  SetCoreExecuteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : SetCoreExecuteGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  SetCoreExecuteGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : SetCoreExecuteGenericAdaptor(values, attrs, Properties{}, {}) {}

  SetCoreExecuteGenericAdaptor(RangeT values, const SetCoreExecuteGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = SetCoreExecute, typename = std::enable_if_t<std::is_same_v<LateInst, SetCoreExecute>>>
  SetCoreExecuteGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getRoutingInstance() {
    return (*getODSOperands(0).begin());
  }

  ValueT getTile() {
    return (*getODSOperands(1).begin());
  }

  ValueT getIsExecute() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SetCoreExecuteAdaptor : public SetCoreExecuteGenericAdaptor<::mlir::ValueRange> {
public:
  using SetCoreExecuteGenericAdaptor::SetCoreExecuteGenericAdaptor;
  SetCoreExecuteAdaptor(SetCoreExecute op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class SetCoreExecute : public ::mlir::Op<SetCoreExecute, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SetCoreExecuteAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SetCoreExecuteGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_routingInstance = 0;
  static constexpr int odsIndex_tile = 1;
  static constexpr int odsIndex_isExecute = 2;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.set_core_execute_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto [start, length] = getODSOperandIndexAndLength(index);
    return getOperation()->getOperands().slice(start, length);
  }

  ::mlir::TypedValue<::mlir::aie::RoutingInstanceType> getRoutingInstance() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::RoutingInstanceType>>(*getODSOperands(0).begin());
  }

  ::mlir::TypedValue<::mlir::aie::LocType> getTile() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::aie::LocType>>(*getODSOperands(1).begin());
  }

  ::mlir::TypedValue<::mlir::IntegerType> getIsExecute() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(2).begin());
  }

  ::mlir::OpOperand &getRoutingInstanceMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getTileMutable() {
    auto range = getODSOperandIndexAndLength(1);
    return getOperation()->getOpOperand(range.first);
  }

  ::mlir::OpOperand &getIsExecuteMutable() {
    auto range = getODSOperandIndexAndLength(2);
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute);
  static SetCoreExecute create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute);
  static SetCoreExecute create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute);
  static SetCoreExecute create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute);
  static SetCoreExecute create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::Value routingInstance, ::mlir::Value tile, ::mlir::Value isExecute);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SetCoreExecute create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static SetCoreExecute create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static SetCoreExecute create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static SetCoreExecute create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::SetCoreExecute)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantInteger32Op declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConstantInteger32OpGenericAdaptorBase {
public:
  struct Properties {
    using inputAttrTy = ::mlir::IntegerAttr;
    inputAttrTy inputAttr;

    auto getInputAttr() const {
      auto &propStorage = this->inputAttr;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setInputAttr(const ::mlir::IntegerAttr &propValue) {
      this->inputAttr = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.inputAttr == this->inputAttr &&
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
  ConstantInteger32OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.constant_integer32_op", odsAttrs.getContext());
  }

  ConstantInteger32OpGenericAdaptorBase(ConstantInteger32Op op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getInputAttrAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().inputAttr);
    return attr;
  }

  uint32_t getInputAttr();
};

} // namespace detail
template <typename RangeT>
class ConstantInteger32OpGenericAdaptor : public detail::ConstantInteger32OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConstantInteger32OpGenericAdaptorBase;
public:
  ConstantInteger32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConstantInteger32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ConstantInteger32OpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConstantInteger32OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConstantInteger32OpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConstantInteger32OpGenericAdaptor(RangeT values, const ConstantInteger32OpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConstantInteger32Op, typename = std::enable_if_t<std::is_same_v<LateInst, ConstantInteger32Op>>>
  ConstantInteger32OpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class ConstantInteger32OpAdaptor : public ConstantInteger32OpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConstantInteger32OpGenericAdaptor::ConstantInteger32OpGenericAdaptor;
  ConstantInteger32OpAdaptor(ConstantInteger32Op op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConstantInteger32Op : public ::mlir::Op<ConstantInteger32Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantInteger32OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantInteger32OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_outputValue = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inputAttr")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInputAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInputAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.constant_integer32_op");
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

  ::mlir::TypedValue<::mlir::IntegerType> getOutputValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
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
  ::mlir::IntegerAttr getInputAttrAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().inputAttr);
  }

  uint32_t getInputAttr();
  void setInputAttrAttr(::mlir::IntegerAttr attr) {
    getProperties().inputAttr = attr;
  }

  void setInputAttr(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, IntegerAttr value);
  static ConstantInteger32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, IntegerAttr value);
  static ConstantInteger32Op create(::mlir::ImplicitLocOpBuilder &builder, IntegerAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr);
  static ConstantInteger32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr);
  static ConstantInteger32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr);
  static ConstantInteger32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr);
  static ConstantInteger32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, uint32_t inputAttr);
  static ConstantInteger32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type outputValue, uint32_t inputAttr);
  static ConstantInteger32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type outputValue, uint32_t inputAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t inputAttr);
  static ConstantInteger32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint32_t inputAttr);
  static ConstantInteger32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint32_t inputAttr);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static ConstantInteger32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static ConstantInteger32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConstantInteger32Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConstantInteger32Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantInteger32Op)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::ConstantInteger32OpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantInteger64Op declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConstantInteger64OpGenericAdaptorBase {
public:
  struct Properties {
    using inputAttrTy = ::mlir::IntegerAttr;
    inputAttrTy inputAttr;

    auto getInputAttr() const {
      auto &propStorage = this->inputAttr;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setInputAttr(const ::mlir::IntegerAttr &propValue) {
      this->inputAttr = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.inputAttr == this->inputAttr &&
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
  ConstantInteger64OpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.constant_integer64_op", odsAttrs.getContext());
  }

  ConstantInteger64OpGenericAdaptorBase(ConstantInteger64Op op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::IntegerAttr getInputAttrAttr() {
    auto attr = ::llvm::cast<::mlir::IntegerAttr>(getProperties().inputAttr);
    return attr;
  }

  uint64_t getInputAttr();
};

} // namespace detail
template <typename RangeT>
class ConstantInteger64OpGenericAdaptor : public detail::ConstantInteger64OpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConstantInteger64OpGenericAdaptorBase;
public:
  ConstantInteger64OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConstantInteger64OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ConstantInteger64OpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConstantInteger64OpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConstantInteger64OpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConstantInteger64OpGenericAdaptor(RangeT values, const ConstantInteger64OpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConstantInteger64Op, typename = std::enable_if_t<std::is_same_v<LateInst, ConstantInteger64Op>>>
  ConstantInteger64OpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class ConstantInteger64OpAdaptor : public ConstantInteger64OpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConstantInteger64OpGenericAdaptor::ConstantInteger64OpGenericAdaptor;
  ConstantInteger64OpAdaptor(ConstantInteger64Op op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConstantInteger64Op : public ::mlir::Op<ConstantInteger64Op, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantInteger64OpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantInteger64OpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static constexpr int odsIndex_outputValue = 0;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inputAttr")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInputAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInputAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.constant_integer64_op");
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

  ::mlir::TypedValue<::mlir::IntegerType> getOutputValue() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
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
  ::mlir::IntegerAttr getInputAttrAttr() {
    return ::llvm::cast<::mlir::IntegerAttr>(getProperties().inputAttr);
  }

  uint64_t getInputAttr();
  void setInputAttrAttr(::mlir::IntegerAttr attr) {
    getProperties().inputAttr = attr;
  }

  void setInputAttr(uint64_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, IntegerAttr value);
  static ConstantInteger64Op create(::mlir::OpBuilder &builder, ::mlir::Location location, IntegerAttr value);
  static ConstantInteger64Op create(::mlir::ImplicitLocOpBuilder &builder, IntegerAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr);
  static ConstantInteger64Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr);
  static ConstantInteger64Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type outputValue, ::mlir::IntegerAttr inputAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr);
  static ConstantInteger64Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr);
  static ConstantInteger64Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr inputAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputValue, uint64_t inputAttr);
  static ConstantInteger64Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type outputValue, uint64_t inputAttr);
  static ConstantInteger64Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type outputValue, uint64_t inputAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t inputAttr);
  static ConstantInteger64Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, uint64_t inputAttr);
  static ConstantInteger64Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, uint64_t inputAttr);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static ConstantInteger64Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static ConstantInteger64Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConstantInteger64Op create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConstantInteger64Op create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  LLVM_ATTRIBUTE_MINSIZE void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantInteger64Op)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::ConstantInteger64OpGenericAdaptorBase::Properties)

namespace mlir::aie {


//===----------------------------------------------------------------------===//
// ::mlir::aie::ConstantOp declarations
//===----------------------------------------------------------------------===//

namespace detail {

class ConstantOpGenericAdaptorBase {
public:
  struct Properties {
    using valueTy = ::mlir::DenseElementsAttr;
    valueTy value;

    auto getValue() const {
      auto &propStorage = this->value;
      return ::llvm::cast<::mlir::DenseElementsAttr>(propStorage);
    }
    void setValue(const ::mlir::DenseElementsAttr &propValue) {
      this->value = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.value == this->value &&
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
  ConstantOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
      odsOpName.emplace("Aie.constant", odsAttrs.getContext());
  }

  ConstantOpGenericAdaptorBase(ConstantOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
    return {index, 1};
  }

  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes() {
    return odsAttrs;
  }

  ::mlir::DenseElementsAttr getValueAttr() {
    auto attr = ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
    return attr;
  }

  ::mlir::DenseElementsAttr getValue();
};

} // namespace detail
template <typename RangeT>
class ConstantOpGenericAdaptor : public detail::ConstantOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConstantOpGenericAdaptorBase;
public:
  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::PropertyRef properties, ::mlir::RegionRange regions = {}) : ConstantOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr) : ConstantOpGenericAdaptor(values, attrs, Properties{}, {}) {}

  ConstantOpGenericAdaptor(RangeT values, const ConstantOpGenericAdaptorBase &base) : Base(base), odsOperands(values) {}

  template <typename LateInst = ConstantOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConstantOp>>>
  ConstantOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

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
class ConstantOpAdaptor : public ConstantOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConstantOpGenericAdaptor::ConstantOpGenericAdaptor;
  ConstantOpAdaptor(ConstantOp op);

  ::llvm::LogicalResult verify(::mlir::Location loc);
};
class ConstantOp : public ::mlir::Op<ConstantOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("value")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getValueAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getValueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("Aie.constant");
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

  static ::llvm::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void walkInherentAttrs(::mlir::MLIRContext *ctx, Properties &prop, ::llvm::function_ref<void(::llvm::StringRef, ::mlir::Attribute &)> visitor);
  static ::llvm::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  static ::llvm::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::DenseElementsAttr getValueAttr() {
    return ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
  }

  ::mlir::DenseElementsAttr getValue();
  void setValueAttr(::mlir::DenseElementsAttr attr) {
    getProperties().value = attr;
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value);
  static ConstantOp create(::mlir::OpBuilder &builder, ::mlir::Location location, DenseElementsAttr value);
  static ConstantOp create(::mlir::ImplicitLocOpBuilder &builder, DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, double value);
  static ConstantOp create(::mlir::OpBuilder &builder, ::mlir::Location location, double value);
  static ConstantOp create(::mlir::ImplicitLocOpBuilder &builder, double value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value);
  static ConstantOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value);
  static ConstantOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value);
  static ConstantOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value);
  static ConstantOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value);
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static ConstantOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  [[deprecated("use the overload taking Properties and discardableAttributes instead")]]
  static ConstantOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConstantOp create(::mlir::OpBuilder &builder, ::mlir::Location location, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  static ConstantOp create(::mlir::ImplicitLocOpBuilder &builder, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, const Properties &properties, ::llvm::ArrayRef<::mlir::NamedAttribute> discardableAttributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult setPropertiesFromParsedAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    assert(name.isRegistered() && "Operation isn't registered, missing a "
          "dependent dialect loading?");
    return name.getAttributeNames()[index];
  }

  static void buildPropertiesAndDiscardableAttributes(::mlir::OperationState &odsState, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes);
public:
};

} // namespace mlir::aie
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::ConstantOp)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::aie::detail::ConstantOpGenericAdaptorBase::Properties)


#endif // GET_OP_CLASSES

