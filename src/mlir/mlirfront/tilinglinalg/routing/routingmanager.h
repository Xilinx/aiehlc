/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __ROUTING_MANAGER__
#define __ROUTING_MANAGER__
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

struct TensorSplitDesc;
struct SplitModel;

//fieldparser
#include "mlir/IR/DialectImplementation.h"

#include "routinginterface.h.inc"

#include "routingdialect.h.inc"

using namespace mlir;
using namespace routing;

#define GET_ATTRDEF_CLASSES
#include "routingattr.h.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_TYPEDEF_CLASSES
#include "routingtype.h.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_OP_CLASSES
#define GET_OP_DECLS
#include "routingop.h.inc"
#undef GET_OP_DECLS
#undef GET_OP_CLASSEST

namespace routing {
// GEMM per-dim tiling scalars sourced from routing.partitiontensor TilingAttr.
// These mirror the flat module attrs (routing.tile_m / tile_rows / tile_n /
// tile_cols / effective_k / full_k / k_rounds) but come from the IR op, making
// the TilingAttr the single source of truth for the fullconnect_auto=1 schedule.
struct GemmTilingScalars {
    int64_t tileM = 0;
    int64_t tileRows = 0;
    int64_t mRounds = 0;
    int64_t tileN = 0;
    int64_t tileCols = 0;
    int64_t nRounds = 0;
    int64_t effectiveK = 0;
    int64_t fullK = 0;
    int64_t kRounds = 0;
    bool found = false; // true when at least one usable TilingAttr was read
};

// Walk the module's routing.partitiontensor ops and extract GEMM tiling scalars
// from their #routing.tiling TilingAttr. Role is resolved by PartitionAttr's
// hwAxisOwner ("row" -> M, "col" -> N); the K dimension is the un-meshed non-split
// dim (outer.rounds==1) carrying a nested on-core slice_tiling. Only active when
// routing.fullconnect_auto==1; otherwise returns {found=false} so callers fall
// back to the flat module attrs.
GemmTilingScalars readGemmTilingScalars(mlir::ModuleOp moduleOp);

// ---------------------------------------------------------------------------
// Per-tensor routing descriptors, carried by the op that declares the tensor
// (no longer `tensor_N.*` module attributes):
//   routing.routingcreatescheduletensor  -- inherent attrs halo/shim_dma/
//                                           layout_transform/pp_depth
//   dfscheblueprint.declare_data         -- the same values, copied by
//                                           DmaphopTodfscheblueprintPass under
//                                           the `routing.`-prefixed names below
// Keyed by func-arg index, the canonical tensor index every pass resolves.
// ---------------------------------------------------------------------------
inline constexpr llvm::StringLiteral kDeclHaloAttr = "routing.halo";
inline constexpr llvm::StringLiteral kDeclShimDmaAttr = "routing.shim_dma";
inline constexpr llvm::StringLiteral kDeclKernelBlockAttr = "routing.kernel_block";
inline constexpr llvm::StringLiteral kDeclLayoutTransformAttr = "routing.layout_transform";
inline constexpr llvm::StringLiteral kDeclPpDepthAttr = "routing.pp_depth";
// Kernel round schedule on the routing func.func (see KernelTilingAttr).
inline constexpr llvm::StringLiteral kKernelTilingAttr = "routing.kernel_tiling";

struct TensorRoutingInfo {
    HaloAttr halo;                   // null: not a spatial-halo tensor
    ShimDmaAttr shimDma;             // null: no user DmaTransform
    llvm::StringRef layoutTransform; // "" when unset
    std::optional<int64_t> ppDepth;  // unset when absent
    KernelBlockAttr kernelBlock;     // null: not a kernel-only (traced) port
};
using TensorRoutingInfoMap = llvm::DenseMap<int, TensorRoutingInfo>;

// Walk the module's tensor-declaring ops and collect their descriptors. Call it
// at pass entry: conversions erase the declaring op.
TensorRoutingInfoMap collectTensorRoutingInfo(mlir::ModuleOp moduleOp);
// Descriptor of func arg `argIdx`, or nullptr.
const TensorRoutingInfo *lookupTensorRoutingInfo(const TensorRoutingInfoMap &map, int argIdx);
// The map's func-arg indices in ascending order (DenseMap order is unstable).
llvm::SmallVector<int> sortedTensorIndices(const TensorRoutingInfoMap &map);
// The halo of func arg `argIdx`, or null.
HaloAttr lookupHalo(const TensorRoutingInfoMap &map, int argIdx);
// The kernel's contiguous halo slab size (HaloAttr buf_size of the halo tensor),
// 0 when the kernel has no halo input.
int64_t spatialHaloBufSize(const TensorRoutingInfoMap &map);
// Record the kernel's contiguous halo slab size on every halo tensor's
// #routing.halo (buf_size). Returns how many tensors were updated.
int setHaloBufSize(mlir::ModuleOp moduleOp, int64_t bufSize);
// Copy a createscheduletensor's descriptors onto the op that replaces it.
void copyTensorRoutingInfo(routing::createscheduletensor from, mlir::Operation *to);

// The routing func's #routing.kernel_tiling (all-zero attr when absent, so
// callers read 0 exactly as for a missing module attr).
KernelTilingAttr readKernelTiling(mlir::ModuleOp moduleOp);
// Named fields of a #routing.kernel_tiling; 0 = leave unchanged / not set.
struct KernelTilingFields {
    int64_t tileM = 0, tileRows = 0, mRounds = 0;
    int64_t tileN = 0, tileCols = 0, nRounds = 0;
    int64_t effectiveK = 0, fullK = 0, kRounds = 0;
    int64_t outRounds = 0;
};
// Merge `fields` into the routing func's kernel_tiling (non-zero fields win).
void mergeKernelTiling(mlir::ModuleOp moduleOp, const KernelTilingFields &fields);
} // namespace routing

class routingmanager{
public:
    routingmanager(){};
    void type_interface_test(MLIRContext* ctx) ;
    ModuleOp ops_test(MLIRContext* ctx,int totalN=2) ;
    ModuleOp ops_testNew(MLIRContext *ctx, int totalN = 2, std::string routingname = "");
    mlir::func::FuncOp createroutingfunc(MLIRContext* ctx, int totalN = 16,bool purefunc=false) ;
    void createroutingfuncByDim(OpBuilder& builder, MLIRContext* ctx,  bool binput,Value mesh, Value tensor, uint32_t hwsplitnum, std::string splitAxis);
    void createroutingfuncGEMM(OpBuilder &builder, MLIRContext *ctx, Value mesh, Value tensorA, Value tensorB,
                               Value tensorC, uint32_t hwsplitnum, std::string splitAxis);
    /// Generalized routing function driven by SplitModel instead of hardcoded logic.
    /// Creates partitiontensor, extract_data, IO, and movedata ops for each tensor
    /// according to its TensorSplitDesc.
    void createroutingfuncBySplitModel(OpBuilder &builder, MLIRContext *ctx, Value mesh,
                                       const std::vector<Value> &tensors, const std::vector<bool> &isInputFlags,
                                       int meshRows, int meshCols, const struct SplitModel &splitModel);
    //void createroutingfuncByDimDmap(OpBuilder& builder, MLIRContext* ctx,  bool binput,Value mesh, Value tensor, uint32_t hwsplitnum, std::string splitAxis);
    static void loaddialect(MLIRContext* ctx);
};
#endif//__ROUTING_MANAGER__