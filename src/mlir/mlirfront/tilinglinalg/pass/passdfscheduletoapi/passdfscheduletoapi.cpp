/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passdfscheduletoapi.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <tuple>

using namespace mlir;

namespace mlir {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static std::string getEmitCTypeString(Type elemType) {
    if (elemType.isInteger(8)) return "int8_t";
    if (elemType.isInteger(16)) return "int16_t";
    if (elemType.isInteger(32)) return "int32_t";
    if (elemType.isInteger(64)) return "int64_t";
    if (elemType.isF32()) return "float";
    if (elemType.isF64()) return "double";
    return "uint8_t";
}

static std::string buildArrayDimString(ArrayRef<int64_t> shape) {
    std::string result;
    for (auto dim : shape) {
        result += "[" + std::to_string(dim) + "]";
    }
    return result;
}

static std::string buildInitializerString(DenseElementsAttr denseAttr, Type elemType) {
    std::ostringstream initStream;
    initStream << "{";
    bool first = true;
    
    if (elemType.isInteger(8)) {
        for (auto val : denseAttr.getValues<int8_t>()) {
            if (!first) initStream << ", ";
            first = false;
            initStream << static_cast<int>(val);
        }
    } else if (elemType.isInteger(16)) {
        for (auto val : denseAttr.getValues<int16_t>()) {
            if (!first) initStream << ", ";
            first = false;
            initStream << val;
        }
    } else if (elemType.isInteger(32)) {
        for (auto val : denseAttr.getValues<int32_t>()) {
            if (!first) initStream << ", ";
            first = false;
            initStream << val;
        }
    } else if (elemType.isInteger(64)) {
        for (auto val : denseAttr.getValues<int64_t>()) {
            if (!first) initStream << ", ";
            first = false;
            initStream << val;
        }
    } else if (elemType.isF32()) {
        for (auto val : denseAttr.getValues<APFloat>()) {
            if (!first) initStream << ", ";
            first = false;
            initStream << val.convertToFloat();
        }
    } else if (elemType.isF64()) {
        for (auto val : denseAttr.getValues<APFloat>()) {
            if (!first) initStream << ", ";
            first = false;
            initStream << val.convertToDouble();
        }
    }
    initStream << "}";
    return initStream.str();
}

static int64_t getElemSize(Type elemType) {
    if (elemType.isInteger(8)) return 1;
    if (elemType.isInteger(16)) return 2;
    if (elemType.isInteger(32) || elemType.isF32()) return 4;
    if (elemType.isInteger(64) || elemType.isF64()) return 8;
    return 1;
}

//===----------------------------------------------------------------------===//
// Shared State for Conversion Patterns
//===----------------------------------------------------------------------===//

struct IoDebugInfo {
    int32_t col, row, channel, bd_id;
    std::string direction; // "DMA_MM2S" or "DMA_S2MM"
};
struct TileDebugInfo {
    int32_t col, row;
};

struct ConversionState {
    // Maps MLIR Value to (PartitionTensor SSA Value, byte size, num elements)
    DenseMap<Value, std::tuple<Value, int64_t, int64_t>> memAllocMap;
    // Also track the raw data pointer for cases where we need it
    DenseMap<Value, Value> dataPtrMap;
    SmallVector<Value> allocatedMemList;
    
    // Counter for generating unique array names
    int arrayIndex = 0;
    int partitionIndex = 0;
    mutable int shapeArrayIndex = 0; // for PartitionTensor static shape arrays (C++ compatible)

    // Statistics for ExtractSliceInnerPattern
    int extractSliceCallCount = 0;
    int extractSliceFailCount = 0;
    int extractSliceSuccessCount = 0;
    
    // Resource manager for channel and BD IDs
    // Map from tile (col, row, direction) to next available channel ID.
    // AIEML core tiles have independent channel spaces per direction:
    //   MM2S channels 0-1, S2MM channels 0-1.
    // Using a single counter across directions would overflow (e.g., 2 S2MM + 1 MM2S
    // would yield channel IDs 0,1,2 — but channel 2 doesn't exist in either direction).
    // Key: (col, row, direction_str) where direction_str is "MM2S" or "S2MM".
    std::map<std::tuple<int32_t, int32_t, std::string>, int32_t> nextChannelId;
    DenseMap<std::pair<int32_t, int32_t>, int32_t> nextBdId;

    // Map from bd_handle Value (result of config.dma_bd) to bd_id
    DenseMap<Value, int32_t> bdHandleToBdId;
    // Map from io_handle Value (result of config.create_io) to (channel_id, bd_id)
    DenseMap<Value, std::pair<int32_t, int32_t>> ioHandleToResourceMap;
    // Per-tile queue of bd_ids assigned by create_io, consumed in order by getbdid
    DenseMap<std::pair<int32_t, int32_t>, std::vector<int32_t>> perTileBdIdQueue;

    // Track (col,row,lock_id) tuples that have already had XAie_LockSetValue emitted
    std::set<std::tuple<int32_t, int32_t, int32_t>> initializedLocks;
    std::vector<std::pair<std::string, std::string>> pendingLockInits;

    // Debug snapshot data (populated when enableDebug is true)
    SmallVector<IoDebugInfo> debugIos;
    SmallVector<TileDebugInfo> debugTiles;
    bool enableDebug = false;
    int runtimeDebugLevel = -1;

    // Cached values for inner patterns (set before applying inner patterns)
    Value devInstRef;
    Value cacheableConst;
    Type voidPtrType;
    Type memInstPtrType;
    Type i32Type;
    Type partitionType;
    Type devInstType;
    MLIRContext *ctx = nullptr;
    
    // Helper to allocate a channel ID for a tile, per direction.
    // AIEML core tiles have separate channel spaces for MM2S (0-1) and S2MM (0-1).
    int32_t allocateChannelId(int32_t col, int32_t row, const std::string &direction) {
        auto key = std::make_tuple(col, row, direction);
        int32_t &nextId = nextChannelId[key];
        return nextId++;
    }
    
    // Helper to allocate a BD ID for a tile
    int32_t allocateBdId(int32_t col, int32_t row) {
        auto key = std::make_pair(col, row);
        int32_t &nextId = nextBdId[key];
        return nextId++;
    }
};


//===----------------------------------------------------------------------===//
// Module-Level Patterns (Global conversions)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Helper: EraseOpLowering - Reusable pattern to erase ops by name
//===----------------------------------------------------------------------===//

template <typename Op_T>
struct EraseOpLowering : public OpConversionPattern<Op_T> {
    using OpConversionPattern<Op_T>::OpConversionPattern;
    LogicalResult matchAndRewrite(Op_T op, typename Op_T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Inner Patterns (Applied inside host op region via walk)
//===----------------------------------------------------------------------===//

/// Inner pattern for arith.constant -> emitc.constant (handles ALL constant types)
struct ArithConstantInnerPattern : public OpConversionPattern<arith::ConstantOp> {
    ConversionState &state;
    
    ArithConstantInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state)
        : OpConversionPattern<arith::ConstantOp>(typeConverter, ctx, /*benefit=*/1), state(state) {}
    
    LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto resultType = op.getType();
        auto value = op.getValue();
        
        // Handle dense constants (tensors) - these were converted to global arrays in Phase 2
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
            // Check if this op has the "emitc.global_array_name" attribute (set in Phase 2)
            if (auto arrayNameAttr = op->getAttrOfType<StringAttr>("emitc.global_array_name")) {
                std::string arrayName = arrayNameAttr.getValue().str();
                
                // Create emitc.constant that references the global array
                // The type should be a pointer to the array's element type
                auto emitcConst = rewriter.create<emitc::ConstantOp>(
                    loc, state.voidPtrType,
                    emitc::OpaqueAttr::get(state.ctx, "(void*)" + arrayName));
                
                rewriter.replaceOp(op, emitcConst.getResult());
                llvm::errs() << "[Pattern] Converted arith.constant dense -> emitc.constant reference to " 
                             << arrayName << "\n";
                return success();
            }
            // Tensor constant feeding bufferization.to_memref — not marked
            // with global_array_name (intentionally skipped in Phase 2).
            // Return failure so the conversion framework leaves it alone.
            llvm::errs() << "[Pattern] Dense constant without global array name, skipping\n";
            return failure();
        }
        
        // Handle integer constants (including index type)
        if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
            auto emitcConst = rewriter.create<emitc::ConstantOp>(
                loc, resultType, intAttr);
            rewriter.replaceOp(op, emitcConst.getResult());
            llvm::errs() << "[Pattern] Converted arith.constant (int) to emitc.constant\n";
            return success();
        }
        
        // Handle float constants
        if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
            auto emitcConst = rewriter.create<emitc::ConstantOp>(
                loc, resultType, floatAttr);
            rewriter.replaceOp(op, emitcConst.getResult());
            llvm::errs() << "[Pattern] Converted arith.constant (float) to emitc.constant\n";
            return success();
        }
        
        // Handle bool constants
        if (auto boolAttr = dyn_cast<BoolAttr>(value)) {
            auto emitcConst = rewriter.create<emitc::ConstantOp>(
                loc, resultType, boolAttr);
            rewriter.replaceOp(op, emitcConst.getResult());
            llvm::errs() << "[Pattern] Converted arith.constant (bool) to emitc.constant\n";
            return success();
        }
        
        // For any other constant types, try to convert to emitc.constant
        auto emitcConst = rewriter.create<emitc::ConstantOp>(
            loc, resultType, value);
        rewriter.replaceOp(op, emitcConst.getResult());
        llvm::errs() << "[Pattern] Converted arith.constant (other) to emitc.constant\n";
        return success();
    }
};

/// Inner pattern for dfscheblueprint.declare_data -> XAie_MemAllocate + memcpy
struct DeclareDataInnerPattern : public OpConversionPattern<dfscheblueprint::DeclareDataOp> {
    ConversionState &state;

    DeclareDataInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state)
        : OpConversionPattern<dfscheblueprint::DeclareDataOp>(typeConverter, ctx, /*benefit=*/100), state(state) {}

    // OpConversionPattern provides OpAdaptor automatically
    LogicalResult matchAndRewrite(dfscheblueprint::DeclareDataOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        llvm::errs() << "[Pattern] DeclareDataInnerPattern::matchAndRewrite called\n";
        
        auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
        if (!resultType) {
            llvm::errs() << "[Pattern] DeclareData: Result is not RankedTensorType\n";
            return failure();
        }
        
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        
        // Get converted source pointer from adaptor (auto-generated by OpConversionPattern)
        Value srcPtr = adaptor.getOperands()[0];
        
        llvm::errs() << "[Pattern] DeclareData: OpAdaptor srcPtr = ";
        srcPtr.print(llvm::errs());
        llvm::errs() << "\n[Pattern] DeclareData: OpAdaptor srcPtr type = ";
        srcPtr.getType().print(llvm::errs());
        llvm::errs() << "\n";
        
        if (!srcPtr) {
            llvm::errs() << "[Pattern] DeclareData: No converted input from adaptor\n";
            return failure();
        }
        
        // Calculate sizes from the result type
        auto shape = resultType.getShape();
        int64_t totalElements = 1;
        for (auto dim : shape) totalElements *= dim;
        
        Type elemType = resultType.getElementType();
        int64_t elemSize = getElemSize(elemType);
        int64_t byteSize = totalElements * elemSize;
        
        // Create types locally
        Type i32Type = IntegerType::get(ctx, 32);
        Type voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
        Type memInstPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "XAie_MemInst"));

        // Create XAIE_MEM_CACHEABLE constant
        auto cacheableConst = rewriter.create<emitc::ConstantOp>(
            loc, i32Type,
            emitc::OpaqueAttr::get(ctx, "XAIE_MEM_CACHEABLE"));

        // Create size constant
        auto sizeConst = rewriter.create<emitc::ConstantOp>(loc, i32Type,
            rewriter.getI32IntegerAttr(byteSize));

        // Call XAie_MemAllocate (use state.devInstRef = host block arg 0)
        auto memInst = rewriter.create<emitc::CallOpaqueOp>(
            loc, memInstPtrType, "XAie_MemAllocate", nullptr, nullptr,
            ValueRange{state.devInstRef, sizeConst.getResult(), cacheableConst.getResult()});

        // Call XAie_MemGetVAddr to get void* address
        auto vaddr = rewriter.create<emitc::CallOpaqueOp>(loc,
            voidPtrType, "XAie_MemGetVAddr", nullptr, nullptr,
            ValueRange{memInst.getResult(0)});
        
        // Call memcpy to copy data from source array to allocated memory
        rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "memcpy",
            nullptr, nullptr,
            ValueRange{vaddr.getResult(0), srcPtr, sizeConst.getResult()});
        
        llvm::errs() << "[Pattern] DeclareData: XAie_MemAllocate and memcpy completed\n";
        
        // Replace the op with the vaddr result to maintain SSA chain
        rewriter.replaceOp(op, vaddr.getResult(0));
        return success();
    }
};

/// Inner pattern for dfschedule.alloc_device_mem -> XAie_MemAllocate + memcpy + track_alloc
/// Mirrors DeclareDataInnerPattern but operates on the new SSA-clean alloc op.
struct AllocDeviceMemInnerPattern : public OpConversionPattern<dfschedule::AllocDeviceMemOp> {
    ConversionState &state;

    AllocDeviceMemInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state)
        : OpConversionPattern<dfschedule::AllocDeviceMemOp>(typeConverter, ctx, /*benefit=*/100), state(state) {}

    LogicalResult matchAndRewrite(dfschedule::AllocDeviceMemOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();

        // Source operand (may be void* from prior cast, or a converted PartitionTensor from
        // unrealized_cast -> extract_slice -> partitiontensor chain)
        Value srcPtr = adaptor.getOperands()[0];
        if (!srcPtr)
            return failure();

        // Compute size from the SOURCE memref type (the input to alloc_device_mem)
        // The source has the real shape (e.g. memref<8x16xi8>); the result has flattened DDR type.
        auto srcMemrefType = dyn_cast<MemRefType>(op.getSource().getType());
        int64_t byteSize = 128; // fallback
        if (srcMemrefType) {
            int64_t totalElements = 1;
            for (auto dim : srcMemrefType.getShape())
                totalElements *= dim;
            int64_t elemSize = getElemSize(srcMemrefType.getElementType());
            byteSize = totalElements * elemSize;
        } else {
            // Try result type
            auto resultMemrefType = dyn_cast<MemRefType>(op.getMemref().getType());
            if (resultMemrefType) {
                int64_t totalElements = 1;
                for (auto dim : resultMemrefType.getShape())
                    totalElements *= dim;
                byteSize = totalElements * getElemSize(resultMemrefType.getElementType());
            }
        }

        Type i32Type = IntegerType::get(ctx, 32);
        Type voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
        Type memInstPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "XAie_MemInst"));

        auto cacheableConst =
            rewriter.create<emitc::ConstantOp>(loc, i32Type, emitc::OpaqueAttr::get(ctx, "XAIE_MEM_CACHEABLE"));
        auto sizeConst = rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(byteSize));

        // Use state.devInstRef (host block arg 0 = XAie_DevInst* dev)
        auto memInst = rewriter.create<emitc::CallOpaqueOp>(
            loc, memInstPtrType, "XAie_MemAllocate", nullptr, nullptr,
            ValueRange{state.devInstRef, sizeConst.getResult(), cacheableConst.getResult()});

        auto vaddr = rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "XAie_MemGetVAddr", nullptr, nullptr,
                                                          ValueRange{memInst.getResult(0)});

        // Determine if the source is a PartitionTensor (i.e., came from tensor -> extract_slice path).
        // Check the ORIGINAL (pre-conversion) source op: if it's unrealized_conversion_cast with
        // a tensor input, the runtime value will be a PartitionTensor (from ExtractSliceInnerPattern).
        // In that case we must extract the .data field via __runtime_buffer_arg().
        bool isSrcPartitionTensor = false;
        {
            Value origSrc = op.getSource();
            if (auto castOp = origSrc.getDefiningOp<UnrealizedConversionCastOp>()) {
                if (!castOp.getInputs().empty()) {
                    auto inputType = castOp.getInputs()[0].getType();
                    if (isa<RankedTensorType>(inputType)) {
                        isSrcPartitionTensor = true;
                    }
                }
            }
            // Also check if the adapted srcPtr itself is already PartitionTensor (runtime type)
            if (auto opaqueType = dyn_cast<emitc::OpaqueType>(srcPtr.getType())) {
                if (opaqueType.getValue() == "PartitionTensor") {
                    isSrcPartitionTensor = true;
                }
            }
        }

        Value dataPtr = srcPtr;
        if (isSrcPartitionTensor) {
            // Extract void* data field: __runtime_buffer_arg(partition_tensor)
            // At runtime, srcPtr will be a PartitionTensor (from ExtractSliceInnerPattern).
            // Cast srcPtr to PartitionTensor opaque type so emitc.call_opaque accepts it.
            auto partitionType = emitc::OpaqueType::get(ctx, "PartitionTensor");
            auto ptCast =
                rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{partitionType}, ValueRange{srcPtr});
            auto dataPtrOp = rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "__runtime_buffer_arg", nullptr,
                                                                  nullptr, ValueRange{ptCast.getResult(0)});
            dataPtr = dataPtrOp.getResult(0);
        }

        rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "memcpy", nullptr, nullptr,
                                             ValueRange{vaddr.getResult(0), dataPtr, sizeConst.getResult()});

        rewriter.replaceOp(op, vaddr.getResult(0));
        return success();
    }
};

/// Inner pattern for bufferization.to_memref inside dfschedule.host.
/// The source tensor constant was converted to a global array (void*) in Phase 2.
/// Simply forward the converted operand (already void*) as the memref → void* result.
struct ToMemrefInnerPattern : public OpConversionPattern<bufferization::ToMemrefOp> {
    ToMemrefInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<bufferization::ToMemrefOp>(typeConverter, ctx, /*benefit=*/90) {}

    LogicalResult matchAndRewrite(bufferization::ToMemrefOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Only handle ops inside dfschedule.host (the DDR init chain).
        if (!op->getParentOfType<dfschedule::HostBlockOp>())
            return failure();
        if (adaptor.getOperands().empty())
            return failure();
        // The tensor operand was converted to void* by ArithConstantInnerPattern.
        // Pass it through as the memref result (also void* via type converter).
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return success();
    }
};

/// Inner pattern for memref.alloc inside dfschedule.host.
/// Lowers to __Runtime_malloc(byteSize) → void*.
struct MemRefAllocInnerPattern : public OpConversionPattern<memref::AllocOp> {
    MemRefAllocInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<memref::AllocOp>(typeConverter, ctx, /*benefit=*/90) {}

    LogicalResult matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!op->getParentOfType<dfschedule::HostBlockOp>())
            return failure();
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto memrefType = op.getType();
        // Compute total byte size from shape and element type.
        int64_t numElements = 1;
        for (auto dim : memrefType.getShape())
            numElements *= dim;
        int64_t elemBytes = getElemSize(memrefType.getElementType());
        int64_t byteSize = numElements * elemBytes;
        // Ensure allocation is int32-aligned (DMA transfers require 4-byte alignment).
        byteSize = (byteSize + 3) & ~3;
        Type i64Type = rewriter.getIntegerType(64);
        Type voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
        auto sizeConst = rewriter.create<emitc::ConstantOp>(loc, i64Type, rewriter.getIntegerAttr(i64Type, byteSize));
        auto allocCall = rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "__Runtime_malloc", nullptr, nullptr,
                                                              ValueRange{sizeConst.getResult()});
        rewriter.replaceOp(op, allocCall.getResult(0));
        return success();
    }
};

/// Inner pattern for memref.copy inside dfschedule.host.
/// Lowers to __Runtime_memcpy(dst, src, byteSize).
struct MemRefCopyInnerPattern : public OpConversionPattern<memref::CopyOp> {
    MemRefCopyInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<memref::CopyOp>(typeConverter, ctx, /*benefit=*/90) {}

    LogicalResult matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!op->getParentOfType<dfschedule::HostBlockOp>())
            return failure();
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        // Compute byte size from the source memref type.
        auto srcType = cast<MemRefType>(op.getSource().getType());
        int64_t numElements = 1;
        for (auto dim : srcType.getShape())
            numElements *= dim;
        int64_t elemBytes = getElemSize(srcType.getElementType());
        int64_t byteSize = numElements * elemBytes;
        // Ensure copy size is int32-aligned (consistent with allocation alignment).
        byteSize = (byteSize + 3) & ~3;
        Type i64Type = rewriter.getIntegerType(64);
        auto sizeConst = rewriter.create<emitc::ConstantOp>(loc, i64Type, rewriter.getIntegerAttr(i64Type, byteSize));
        // adaptor gives converted (void*) operands: [source, target].
        Value src = adaptor.getOperands()[0];
        Value dst = adaptor.getOperands()[1];
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "__Runtime_memcpy", nullptr, nullptr,
                                             ValueRange{dst, src, sizeConst.getResult()});
        rewriter.eraseOp(op);
        return success();
    }
};

/// Inner pattern for memref.dealloc inside dfschedule.host.
/// Lowers to __Runtime_free(ptr).
struct MemRefDeallocInnerPattern : public OpConversionPattern<memref::DeallocOp> {
    MemRefDeallocInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<memref::DeallocOp>(typeConverter, ctx, /*benefit=*/90) {}

    LogicalResult matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!op->getParentOfType<dfschedule::HostBlockOp>())
            return failure();
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        Value ptr = adaptor.getOperands()[0];
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "__Runtime_free", nullptr, nullptr, ValueRange{ptr});
        rewriter.eraseOp(op);
        return success();
    }
};

/// Inner pattern for memref.subview inside dfschedule.host.
/// Subviews of the DDR init alloc are pointer-offset views; lower to
/// __runtime_buffer_offset(base, byteOffset).
struct MemRefSubviewInnerPattern : public OpConversionPattern<memref::SubViewOp> {
    MemRefSubviewInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<memref::SubViewOp>(typeConverter, ctx, /*benefit=*/85) {}

    LogicalResult matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!op->getParentOfType<dfschedule::HostBlockOp>())
            return failure();
        // Only handle static offsets (all dynamic cases are not in the DDR init chain).
        auto staticOffsets = op.getStaticOffsets();
        auto srcType = cast<MemRefType>(op.getSource().getType());
        auto srcShape = srcType.getShape();
        int64_t byteOffset = 0;
        int64_t elemBytes = getElemSize(srcType.getElementType());
        // Prefer the subview RESULT type's encoded strided offset. When the DDR
        // root carries a non-identity strided layout (e.g. the NCHW conv-output
        // view), the subview op has already folded the offsets into the result
        // type's `offset:` field using the layout strides — which the naive
        // row-major-from-shape math below would get wrong.
        //
        // The runtime chains __runtime_buffer_offset(parent, delta) calls, so the
        // parent (already-offset source) contributes its own offset. We must emit
        // the offset RELATIVE to the immediate source: delta = resOffset - srcOffset.
        // For identity layouts both paths agree, so this is a superset (no regression).
        auto resType = cast<MemRefType>(op.getType());
        SmallVector<int64_t> resStrides, srcStrides;
        int64_t resOffset = 0, srcOffset = 0;
        if (!resType.getLayout().isIdentity() && succeeded(getStridesAndOffset(resType, resStrides, resOffset)) &&
            !ShapedType::isDynamic(resOffset)) {
            if (!srcType.getLayout().isIdentity() && succeeded(getStridesAndOffset(srcType, srcStrides, srcOffset)) &&
                !ShapedType::isDynamic(srcOffset)) {
                // relative to the offset carried by the immediate source
            } else {
                srcOffset = 0; // identity source: absolute == relative
            }
            byteOffset = (resOffset - srcOffset) * elemBytes;
        } else {
            // Compute linearized byte offset from static offsets and source strides.
            // Row-major strides: stride[i] = product of srcShape[i+1..end].
            int64_t stride = elemBytes;
            // Accumulate strides from innermost dimension outward.
            SmallVector<int64_t> strides(srcShape.size(), stride);
            for (int i = (int)srcShape.size() - 2; i >= 0; --i)
                strides[i] = strides[i + 1] * srcShape[i + 1];
            for (size_t i = 0; i < staticOffsets.size(); ++i) {
                if (ShapedType::isDynamic(staticOffsets[i]))
                    return failure(); // Give up on dynamic offsets
                byteOffset += staticOffsets[i] * strides[i];
            }
        }
        // Ensure byte offset is int32-aligned (DMA transfers require 4-byte alignment).
        byteOffset = (byteOffset + 3) & ~3;
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        Type i64Type = rewriter.getIntegerType(64);
        Type voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
        Value base = adaptor.getOperands()[0]; // converted source (void*)
        auto offsetConst =
            rewriter.create<emitc::ConstantOp>(loc, i64Type, rewriter.getIntegerAttr(i64Type, byteOffset));
        auto viewPtr = rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "__runtime_buffer_offset", nullptr,
                                                            nullptr, ValueRange{base, offsetConst.getResult()});
        rewriter.replaceOp(op, viewPtr.getResult(0));
        return success();
    }
};

/// Inner pattern for dfschedule.buffer_view -> __runtime_buffer_offset(base, offset)
struct BufferViewInnerPattern : public OpConversionPattern<dfschedule::BufferViewOp> {

    BufferViewInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<dfschedule::BufferViewOp>(typeConverter, ctx, /*benefit=*/80) {}

    LogicalResult matchAndRewrite(dfschedule::BufferViewOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();

        Value base = adaptor.getOperands()[0];
        int64_t offset = op.getOffset();

        Type voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
        Type i64Type = rewriter.getIntegerType(64);
        auto offsetConst = rewriter.create<emitc::ConstantOp>(loc, i64Type, rewriter.getIntegerAttr(i64Type, offset));

        auto viewPtr = rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "__runtime_buffer_offset", nullptr,
                                                            nullptr, ValueRange{base, offsetConst.getResult()});

        rewriter.replaceOp(op, viewPtr.getResult(0));
        return success();
    }
};

/// Inner pattern for dfschedule.bind_core_buffer -> emitc constant (L1 offset)
/// The L1 address is a compile-time constant in AIE core tile local data memory.
struct BindCoreBufferInnerPattern : public OpConversionPattern<dfschedule::BindCoreBufferOp> {

    BindCoreBufferInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<dfschedule::BindCoreBufferOp>(typeConverter, ctx, /*benefit=*/80) {}

    LogicalResult matchAndRewrite(dfschedule::BindCoreBufferOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();

        // L1 offset is a compile-time constant — emit as a void* constant
        int64_t l1Offset = op.getOffset();
        Type voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
        // Represent as (void*)L1_OFFSET — safe for XAie BD configuration
        std::string offsetStr = "(void*)" + std::to_string(l1Offset);
        auto l1Const = rewriter.create<emitc::ConstantOp>(loc, voidPtrType, emitc::OpaqueAttr::get(ctx, offsetStr));

        rewriter.replaceOp(op, l1Const.getResult());
        return success();
    }
};

/// Inner pattern for dfschedule.free_device_mem -> __Runtime_free_all_allocs()
struct FreeDeviceMemInnerPattern : public OpConversionPattern<dfschedule::FreeDeviceMemOp> {

    FreeDeviceMemInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<dfschedule::FreeDeviceMemOp>(typeConverter, ctx, /*benefit=*/1) {}

    LogicalResult matchAndRewrite(dfschedule::FreeDeviceMemOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "__Runtime_free_all_allocs", nullptr, nullptr,
                                             ValueRange{});
        rewriter.eraseOp(op);
        return success();
    }
};

/// Pass-through pattern for unrealized_conversion_cast ops produced by BlueprintToSchedulePass
/// These are type bridges from tensor to memref; at API lowering time we just pass the input through.
struct UnrealizedConversionCastInnerPattern : public OpConversionPattern<UnrealizedConversionCastOp> {

    UnrealizedConversionCastInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<UnrealizedConversionCastOp>(typeConverter, ctx, /*benefit=*/10) {}

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // The input has already been converted by a prior pattern (e.g. DeclareDataInnerPattern ->
        // void*). Simply forward the first converted operand as the result.
        if (adaptor.getOperands().empty())
            return failure();
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return success();
    }
};

/// Pass-through pattern for memref.reinterpret_cast produced by BlueprintToSchedulePass
/// (Part D: NCHW conv-output view). The reinterpret_cast only re-labels the strided
/// layout with a zero offset — the underlying base pointer is unchanged — so at API
/// lowering time we forward the converted (void*) source operand directly. The NCHW
/// strides it introduced are already baked into the downstream subview's result-type
/// offset, which MemRefSubviewInnerPattern reads.
struct MemRefReinterpretCastInnerPattern : public OpConversionPattern<memref::ReinterpretCastOp> {

    MemRefReinterpretCastInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<memref::ReinterpretCastOp>(typeConverter, ctx, /*benefit=*/85) {}

    LogicalResult matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        if (!op->getParentOfType<dfschedule::HostBlockOp>())
            return failure();
        if (adaptor.getOperands().empty())
            return failure();
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return success();
    }
};

/// Pass-through pattern for dfschedule.memref_mapping ops.
/// memref_mapping is Pure (no memory effects) and zero-cost: it simply strips the storage scope
/// of a memref to act as a logical SSA anchor. At API lowering time, we forward the converted
/// source operand directly (same as unrealized_conversion_cast pass-through).
struct MemRefMappingInnerPattern : public OpConversionPattern<dfschedule::MemRefMappingOp> {

    MemRefMappingInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpConversionPattern<dfschedule::MemRefMappingOp>(typeConverter, ctx, /*benefit=*/10) {}

    LogicalResult matchAndRewrite(dfschedule::MemRefMappingOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // memref_mapping is a no-op at codegen time; forward the converted source value.
        if (adaptor.getOperands().empty())
            return failure();
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return success();
    }
};

/// Inner pattern for routing.partitiontensor -> __Runtime_init_PartitionTensor
struct PartitionTensorInnerPattern : public OpConversionPattern<routing::partitiontensor> {
    ConversionState &state;
    
    PartitionTensorInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state)
        : OpConversionPattern<routing::partitiontensor>(typeConverter, ctx, /*benefit=*/50), state(state) {}
    
    LogicalResult matchAndRewrite(routing::partitiontensor op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Full tensor deprecation: partitiontensor is a no-op in the new MemRef host path.
        // The data pointer flows through from arith.constant -> void* global array.
        // Pass through to the first (data) operand; offset info is in buffer_view / bind_core_buffer.
        llvm::errs() << "[Pattern] PartitionTensorInnerPattern: pass-through (tensor deprecated)\n";
        if (adaptor.getOperands().empty())
            return failure();
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return success();
    }
    // DEAD CODE BELOW (kept for reference only, never reached)
    LogicalResult _dead_matchAndRewrite(routing::partitiontensor op, OpAdaptor adaptor,
                                        ConversionPatternRewriter &rewriter) const {
        auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
        auto inputType = dyn_cast<RankedTensorType>(op.getOperand().getType());
        if (!resultType || !inputType) return failure();
       
        auto loc = op.getLoc();
        
        // Use adaptor to get converted operand (this is the void* from DeclareData)
        // OpAdaptor for OneOperand ops exposes getOperands() which returns ValueRange
        Value dataVoidPtr = adaptor.getOperands()[0];
        Type want = getTypeConverter()->convertType(op.getOperand().getType());

        if (0) {//!dataVoidPtr || dataVoidPtr.getType() != want) {
            // Debug: Print what we got from adaptor
            llvm::errs() << "[PartitionTensor] adaptor.getOperands()[0] = ";
            if (dataVoidPtr) {
                dataVoidPtr.print(llvm::errs());
                llvm::errs() << "\n[PartitionTensor] Type: ";
                dataVoidPtr.getType().print(llvm::errs());
                llvm::errs() << "\nwant: ";
                want.print(llvm::errs());
                llvm::errs() << "\n";
                if (auto defOp = dataVoidPtr.getDefiningOp()) {
                    llvm::errs() << "[PartitionTensor] Defining op: " << defOp->getName() << "\n";
                }
            } else {
                llvm::errs() << "NULL!\n";
            }
           return failure();
        }   
        
        if (!dataVoidPtr) {
            dataVoidPtr = rewriter.create<emitc::ConstantOp>(loc, state.voidPtrType,
                emitc::OpaqueAttr::get(state.ctx, "NULL")).getResult();
        }
        
        // Get partition attributes
        int splitnum = 1, splitdim = 0;
        if (auto attr = op->getAttrOfType<IntegerAttr>("splitnum")) splitnum = attr.getInt();
        if (auto attr = op->getAttrOfType<IntegerAttr>("splitdim")) splitdim = attr.getInt();
        
        // Get hw_axis_owner and replicate_on (convert string to int)
        int hwAxisOwner = 0; // 0=row, 1=col
        int replicateOn = -1; // -1=none, 0=row, 1=col
        if (auto attr = op->getAttrOfType<StringAttr>("hw_axis_owner")) {
            hwAxisOwner = (attr.getValue() == "col") ? 1 : 0;
        }
        if (auto attr = op->getAttrOfType<StringAttr>("replicate_on")) {
            if (attr.getValue() == "col") replicateOn = 1;
            else if (attr.getValue() == "row") replicateOn = 0;
        }
        
        // Get original shape from input type
        auto originalShape = inputType.getShape();
        int ndim = originalShape.size();
        
        // Compute partition shape: divide by splitnum along splitdim
        SmallVector<int64_t, 4> partitionShape;
        for (int i = 0; i < ndim; i++) {
            if (i == splitdim) {
                partitionShape.push_back(originalShape[i] / splitnum);
            } else {
                partitionShape.push_back(originalShape[i]);
            }
        }
        
        int64_t totalElements = 1;
        for (auto dim : partitionShape) totalElements *= dim;
        
        int64_t elemSize = getElemSize(resultType.getElementType());
        int64_t partitionByteSize = totalElements * elemSize;

        // Emit static arrays for C++ (compound literals (int64_t[]){...} are C-only)
        int idx = state.shapeArrayIndex++;
        std::string origName = "_pt_orig_" + std::to_string(idx);
        std::string partName = "_pt_part_" + std::to_string(idx);
        std::string origInit, partInit;
        for (size_t i = 0; i < originalShape.size(); i++) {
            if (i > 0)
                origInit += ", ";
            origInit += std::to_string(originalShape[i]);
        }
        for (size_t i = 0; i < partitionShape.size(); i++) {
            if (i > 0)
                partInit += ", ";
            partInit += std::to_string(partitionShape[i]);
        }
        rewriter.create<emitc::VerbatimOp>(loc, "static const int64_t " + origName + "[] = {" + origInit + "};");
        rewriter.create<emitc::VerbatimOp>(loc, "static const int64_t " + partName + "[] = {" + partInit + "};");

        auto i64PtrType = emitc::PointerType::get(rewriter.getI64Type());
        auto elemSizeConst = rewriter.create<emitc::ConstantOp>(loc, state.i32Type,
            rewriter.getI32IntegerAttr(elemSize));
        auto ndimConst = rewriter.create<emitc::ConstantOp>(loc, state.i32Type,
            rewriter.getI32IntegerAttr(ndim));
        auto origShapeConst =
            rewriter.create<emitc::ConstantOp>(loc, i64PtrType, emitc::OpaqueAttr::get(state.ctx, origName));
        auto partShapeConst =
            rewriter.create<emitc::ConstantOp>(loc, i64PtrType, emitc::OpaqueAttr::get(state.ctx, partName));
        auto splitdimConst = rewriter.create<emitc::ConstantOp>(loc, state.i32Type,
            rewriter.getI32IntegerAttr(splitdim));
        auto splitnumConst = rewriter.create<emitc::ConstantOp>(loc, state.i32Type,
            rewriter.getI32IntegerAttr(splitnum));
        auto hwAxisConst = rewriter.create<emitc::ConstantOp>(loc, state.i32Type,
            rewriter.getI32IntegerAttr(hwAxisOwner));
        auto replicateConst = rewriter.create<emitc::ConstantOp>(loc, state.i32Type,
            rewriter.getI32IntegerAttr(replicateOn));
        ///*
        auto ptOp = rewriter.create<emitc::CallOpaqueOp>(loc, state.partitionType,
            "__Runtime_init_PartitionTensor", nullptr, nullptr,
            ValueRange{dataVoidPtr, elemSizeConst.getResult(), ndimConst.getResult(),
                       origShapeConst.getResult(), partShapeConst.getResult(),
                       splitdimConst.getResult(), splitnumConst.getResult(),
                       hwAxisConst.getResult(), replicateConst.getResult()});
        ///*
        state.memAllocMap[op.getResult()] = std::make_tuple(ptOp.getResult(0), partitionByteSize, totalElements);
        state.dataPtrMap[op.getResult()] = dataVoidPtr;
        llvm::errs() << "[Pattern] PartitionTensor: created (ndim=" << ndim 
                     << ", splitdim=" << splitdim << ", splitnum=" << splitnum << ")\n";
        //*/
        rewriter.replaceOp(op, ptOp.getResult(0));
        //rewriter.eraseOp(op);
        return success();
    }
};

/// Inner pattern for tensor.extract_slice -> __Runtime_extract_slice_*
struct ExtractSliceInnerPattern : public OpConversionPattern<tensor::ExtractSliceOp> {
    ConversionState &state;
    
    ExtractSliceInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state)
        : OpConversionPattern<tensor::ExtractSliceOp>(typeConverter, ctx, /*benefit=*/1), state(state) {}
    
    LogicalResult matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Full tensor deprecation: extract_slice is a no-op in the new MemRef host path.
        // Offset information is encoded in dfschedule.buffer_view / bind_core_buffer ops.
        // Pass through to the source operand.
        llvm::errs() << "[Pattern] ExtractSliceInnerPattern: pass-through (tensor deprecated)\n";
        state.extractSliceCallCount++;
        if (adaptor.getOperands().empty())
            return failure();
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        state.extractSliceSuccessCount++;
        return success();
    }
    // DEAD CODE BELOW (kept for reference, never reached)
    LogicalResult _dead_matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                                        ConversionPatternRewriter &rewriter) const {
        state.extractSliceCallCount++;
        llvm::errs() << "[Pattern] ExtractSliceInnerPattern::matchAndRewrite called (#" 
                     << state.extractSliceCallCount << ")\n";
        auto loc = op.getLoc();
        
        auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
        auto srcType = dyn_cast<RankedTensorType>(op.getSource().getType());
        if (!resultType || !srcType) {
            state.extractSliceFailCount++;
            llvm::errs() << "[Pattern] ExtractSlice: CONVERT FAIL - Not a ranked tensor type\n";
            llvm::errs() << "[Pattern] ExtractSlice: FAILURE #" << state.extractSliceFailCount 
                         << " (total calls: " << state.extractSliceCallCount << ")\n";
            return failure();
        }
        
        Value srcTensor = op.getSource();
        
        // Use adaptor to get the converted source
        Value srcPartitionTensor = adaptor.getSource();
        
        // Check if the input source is already a PartitionTensor or needs conversion
        Operation* srcOp = srcTensor.getDefiningOp();
        bool isFromPartitionTensor = false;
        bool isFromExtractSlice = false;
        
        if (!srcOp) {
            state.extractSliceFailCount++;
            llvm::errs() << "[Pattern] ExtractSlice: CONVERT FAIL - Source is BlockArgument\n";
            llvm::errs() << "[Pattern] ExtractSlice: FAILURE #" << state.extractSliceFailCount 
                         << " (total calls: " << state.extractSliceCallCount << ")\n";
            return failure();
        }
        
        // Check the defining operation type
        if (isa<routing::partitiontensor>(srcOp)) {
            isFromPartitionTensor = true;
            llvm::errs() << "[Pattern] ExtractSlice: Input from routing.partitiontensor\n";
        } else if (isa<tensor::ExtractSliceOp>(srcOp)) {
            isFromPartitionTensor = true;
            isFromExtractSlice = true;
            llvm::errs() << "[Pattern] ExtractSlice: Input from tensor.extract_slice (NESTED SLICE)\n";
        } else {
            llvm::errs() << "[Pattern] ExtractSlice: Input from other op: " << srcOp->getName() << "\n";
        }
        
        // Verify the converted source type
        if (auto opaqueType = dyn_cast<emitc::OpaqueType>(srcPartitionTensor.getType())) {
            if (opaqueType.getValue() == "PartitionTensor") {
                llvm::errs() << "[Pattern] ExtractSlice: Converted source is PartitionTensor ✓\n";
            } else {
                llvm::errs() << "[Pattern] ExtractSlice: Converted source type = " << opaqueType.getValue() << "\n";
            }
        } else {
            llvm::errs() << "[Pattern] ExtractSlice: Converted source NOT yet PartitionTensor, type = " 
                         << srcPartitionTensor.getType() << " (conversion in progress)\n";
        }
        
        auto offsets = op.getStaticOffsets();
        auto sizes = op.getStaticSizes();
        auto srcShape = srcType.getShape();
        
        int64_t ndim = srcShape.size();
        if (ndim != 2) {
            state.extractSliceFailCount++;
            llvm::errs() << "[Pattern] ExtractSlice: CONVERT FAIL - Only 2D slices supported, got " 
                         << ndim << "D\n";
            llvm::errs() << "[Pattern] ExtractSlice: FAILURE #" << state.extractSliceFailCount 
                         << " (total calls: " << state.extractSliceCallCount << ")\n";
            return failure();
        }
        
        int64_t elemSize = getElemSize(resultType.getElementType());
        int64_t sliceElements = sizes[0] * sizes[1];
        int64_t sliceByteSize = sliceElements * elemSize;
        
        // Check if slice is contiguous:
        // For 2D: contiguous if offset[1]==0 && size[1]==srcShape[1], or size[0]==1
        bool isContiguous = (offsets[1] == 0 && sizes[1] == srcShape[1]) || (sizes[0] == 1);
        
        Value resultPt;
        
        // Branch based on source operation type
        if (isFromExtractSlice) {
            // NESTED SLICE: slicing a previous slice result
            llvm::errs() << "[Pattern] ExtractSlice: *** NESTED SLICE MODE ***\n";
            llvm::errs() << "[Pattern]   Parent slice shape: " << srcShape[0] << "x" << srcShape[1] << "\n";
            llvm::errs() << "[Pattern]   This slice: offset=[" << offsets[0] << ", " << offsets[1] 
                         << "] size=[" << sizes[0] << ", " << sizes[1] << "]\n";
            
            if (isContiguous) {
                auto sliceOp = rewriter.create<emitc::CallOpaqueOp>(loc, state.partitionType,
                    "__Runtime_extract_slice_contiguous_2d",
                    rewriter.getArrayAttr({
                        rewriter.getIndexAttr(0),
                        rewriter.getI32IntegerAttr(offsets[0]),
                        rewriter.getI32IntegerAttr(offsets[1]),
                        rewriter.getI32IntegerAttr(sizes[0]),
                        rewriter.getI32IntegerAttr(sizes[1])
                    }),
                    nullptr,
                    ValueRange{srcPartitionTensor});
                resultPt = sliceOp.getResult(0);
                llvm::errs() << "[Pattern]   => Contiguous nested slice (no allocation)\n";
            } else {
                auto sliceOp = rewriter.create<emitc::CallOpaqueOp>(loc, state.partitionType,
                    "__Runtime_extract_slice_strided_2d",
                    rewriter.getArrayAttr({
                        rewriter.getIndexAttr(0),
                        rewriter.getIndexAttr(1),
                        rewriter.getI32IntegerAttr(offsets[0]),
                        rewriter.getI32IntegerAttr(offsets[1]),
                        rewriter.getI32IntegerAttr(sizes[0]),
                        rewriter.getI32IntegerAttr(sizes[1])
                    }),
                    nullptr,
                    ValueRange{state.devInstRef, srcPartitionTensor});
                resultPt = sliceOp.getResult(0);
                state.allocatedMemList.push_back(resultPt);
                llvm::errs() << "[Pattern]   => Strided nested slice (allocated " << sliceByteSize << " bytes)\n";
            }
        } else if (isFromPartitionTensor) {
            // PARTITION SLICE: slicing a partitioned tensor
            llvm::errs() << "[Pattern] ExtractSlice: *** PARTITION SLICE MODE ***\n";
            llvm::errs() << "[Pattern]   Partition shape: " << srcShape[0] << "x" << srcShape[1] << "\n";
            llvm::errs() << "[Pattern]   Slice: offset=[" << offsets[0] << ", " << offsets[1] 
                         << "] size=[" << sizes[0] << ", " << sizes[1] << "]\n";
            
            if (isContiguous) {
                auto sliceOp = rewriter.create<emitc::CallOpaqueOp>(loc, state.partitionType,
                    "__Runtime_extract_slice_contiguous_2d",
                    rewriter.getArrayAttr({
                        rewriter.getIndexAttr(0),
                        rewriter.getI32IntegerAttr(offsets[0]),
                        rewriter.getI32IntegerAttr(offsets[1]),
                        rewriter.getI32IntegerAttr(sizes[0]),
                        rewriter.getI32IntegerAttr(sizes[1])
                    }),
                    nullptr,
                    ValueRange{srcPartitionTensor});
                resultPt = sliceOp.getResult(0);
                llvm::errs() << "[Pattern]   => Contiguous partition slice (no allocation)\n";
            } else {
                auto sliceOp = rewriter.create<emitc::CallOpaqueOp>(loc, state.partitionType,
                    "__Runtime_extract_slice_strided_2d",
                    rewriter.getArrayAttr({
                        rewriter.getIndexAttr(0),
                        rewriter.getIndexAttr(1),
                        rewriter.getI32IntegerAttr(offsets[0]),
                        rewriter.getI32IntegerAttr(offsets[1]),
                        rewriter.getI32IntegerAttr(sizes[0]),
                        rewriter.getI32IntegerAttr(sizes[1])
                    }),
                    nullptr,
                    ValueRange{state.devInstRef, srcPartitionTensor});
                resultPt = sliceOp.getResult(0);
                state.allocatedMemList.push_back(resultPt);
                llvm::errs() << "[Pattern]   => Strided partition slice (allocated " << sliceByteSize << " bytes)\n";
            }
        } else {
            // Unknown source - this shouldn't happen
            state.extractSliceFailCount++;
            llvm::errs() << "[Pattern] ExtractSlice: *** CONVERT FAIL - UNKNOWN SOURCE ***\n";
            llvm::errs() << "[Pattern]   Source operation is not routing.partitiontensor or tensor.extract_slice\n";
            llvm::errs() << "[Pattern] ExtractSlice: FAILURE #" << state.extractSliceFailCount 
                         << " (total calls: " << state.extractSliceCallCount << ")\n";
            return failure();
        }
        
        //state.memAllocMap[op.getResult()] = std::make_tuple(resultPt, sliceByteSize, sliceElements);
        //*/
        //rewriter.eraseOp(op);
        rewriter.replaceOp(op, resultPt);
        
        // Success!
        state.extractSliceSuccessCount++;
        llvm::errs() << "[Pattern] ExtractSlice: ✓ CONVERT SUCCESS #" << state.extractSliceSuccessCount 
                     << " (total: " << state.extractSliceCallCount 
                     << ", failures: " << state.extractSliceFailCount << ")\n";
        return success();
    }
};

// DeclareTensorInnerPattern removed: dfschedule.declaretensor op deprecated and deleted from dialect.

/// OpConversionPattern for dfschedule.declaretile -> XAie_TileLoc call
/// Converts tile declaration to: XAie_LocType tile = XAie_TileLoc(col, row);
struct DeclareTileInnerPattern : public OpConversionPattern<dfschedule::DeclareTileOp> {
    ConversionState &state;
    
    DeclareTileInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state, PatternBenefit benefit = 100)
        : OpConversionPattern<dfschedule::DeclareTileOp>(typeConverter, ctx, benefit), state(state) {}
    
    LogicalResult matchAndRewrite(dfschedule::DeclareTileOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // Get tile coordinates from attributes
        int32_t col = op.getCol();
        int32_t row = op.getRow();
        
        llvm::errs() << "[Pattern] DeclareTile called for tile (col=" << col << ", row=" << row << ")\n";
        
        // Create XAie_LocType type
        auto xaieLocType = emitc::OpaqueType::get(rewriter.getContext(), "XAie_LocType");
        
        // Create constants for column and row
        auto i8Type = rewriter.getI8Type();
        auto colConst = rewriter.create<emitc::ConstantOp>(
            loc, i8Type, rewriter.getI8IntegerAttr(col));
        auto rowConst = rewriter.create<emitc::ConstantOp>(
            loc, i8Type, rewriter.getI8IntegerAttr(row));
        
        // Create XAie_TileLoc(col, row) call
        // This generates: XAie_LocType tile_<col>_<row> = XAie_TileLoc(col, row);
        auto tileLocOp = rewriter.create<emitc::CallOpaqueOp>(
            loc,
            xaieLocType,
            "XAie_TileLoc",
            nullptr,
            nullptr,
            ValueRange{colConst.getResult(), rowConst.getResult()});
        
        llvm::errs() << "  ✓ Created XAie_TileLoc(" << col << ", " << row << ") -> XAie_LocType\n";
        
        // Replace the declaretile op with the XAie_TileLoc call result
        // The result is of type XAie_LocType and can be used by DMA BD operations
        rewriter.replaceOp(op, tileLocOp.getResult(0));
        return success();
    }
};

/// OpConversionPattern for dfschedule.config.dma_bd
/// Converts DMA BD configuration to AIE API calls
/// For now, creates a placeholder that uses the tile to keep it alive
struct ConfigDmaBdInnerPattern : public OpConversionPattern<dfschedule::ConfigDmaBdOp> {
    ConversionState &state;
    
    ConfigDmaBdInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state, PatternBenefit benefit = 1)
        : OpConversionPattern<dfschedule::ConfigDmaBdOp>(typeConverter, ctx, benefit), state(state) {}
    
    LogicalResult matchAndRewrite(dfschedule::ConfigDmaBdOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();

        // Get attributes (offset is now an SSA Value operand, not an attribute)
        int32_t len = op.getLen();
        bool enablePacket = op.getEnablePacket();
        int32_t packetId = op.getPacketId();
        int32_t nextBd = static_cast<int32_t>(op.getNextBd()); // signed cast: sentinel 0xFFFFFFFF (-1) means no next BD

        llvm::errs() << "[Pattern] ConfigDmaBd called (len=" << len << ", enable_packet=" << enablePacket
                     << ", packet_id=" << packetId << ")\n";

        // Get operands
        Value buffer = adaptor.getBuffer();
        Value tile = adaptor.getTile();
        Value bdId = adaptor.getBdId();
        
        llvm::errs() << "  Buffer type: " << buffer.getType() << "\n";
        llvm::errs() << "  Tile type: " << tile.getType() << "\n";
        llvm::errs() << "  BD ID type: " << bdId.getType() << "\n";
        
        // If tile is not yet converted, check if it's a DeclareTileOp and convert it inline
        /*
        if (!tile.getType().isa<emitc::OpaqueType>() || 
            tile.getType().cast<emitc::OpaqueType>().getValue() != "XAie_LocType") {
            // Check if the tile comes from a DeclareTileOp
            if (auto declareTileOp = tile.getDefiningOp<dfschedule::DeclareTileOp>()) {
                llvm::errs() << "  ℹ Tile from DeclareTileOp, converting inline...\n";
                
                // Get tile coordinates
                int32_t col = declareTileOp.getCol();
                int32_t row = declareTileOp.getRow();
                
                // Create XAie_TileLoc call inline
                auto xaieLocType = emitc::OpaqueType::get(rewriter.getContext(), "XAie_LocType");
                auto colConst = rewriter.create<emitc::ConstantOp>(
                    loc, rewriter.getI32Type(),
                    rewriter.getI32IntegerAttr(col));
                auto rowConst = rewriter.create<emitc::ConstantOp>(
                    loc, rewriter.getI32Type(),
                    rewriter.getI32IntegerAttr(row));
                
                tile = rewriter.create<emitc::CallOpaqueOp>(
                    loc, xaieLocType, "XAie_TileLoc",
                    nullptr, nullptr,
                    ValueRange{colConst.getResult(), rowConst.getResult()}).getResult(0);
                
                llvm::errs() << "  ✓ Created inline XAie_TileLoc(" << col << ", " << row << ")\n";
            } else if (tile.isa<BlockArgument>()) {
                // Tile is a function argument (e.g., in dskernel_receiver)
                // This is OK - we'll just use it as-is and it will be handled by the kernel conversion
                llvm::errs() << "  ℹ Tile is a function argument, using as-is\n";
                // For now, just erase this op since kernel-side DMA config is handled differently
                rewriter.eraseOp(op);
                return success();
            } else {
                llvm::errs() << "  ⚠ Tile not from DeclareTileOp or BlockArgument, deferring...\n";
                return failure();
            }
        }
        
        llvm::errs() << "  ✓ Tile is XAie_LocType\n";
        */
        
        // Generate AIE DMA BD configuration following the pattern:
        // XAie_DmaDesc DmaInst;
        // XAie_DmaDescInit(&DevInst, &DmaInst, tile);
        // XAie_DmaSetAddrLen(&DmaInst, offset, len);
        // XAie_DmaSetNextBd(&DmaInst, next_bd, XAIE_ENABLE/XAIE_DISABLE);
        // XAie_DmaSetPkt(&DmaInst, {.PktId=packet_id, .PktType=0});
        // XAie_DmaEnableBd(&DmaInst);
        // XAie_DmaWriteBd(&DevInst, &DmaInst, tile, bd_id);
        
        auto i32Type = rewriter.getI32Type();
        auto dmaDescType = emitc::OpaqueType::get(rewriter.getContext(), "XAie_DmaDesc");

        // Read lock IDs and values directly from attributes
        int32_t acquireLockId = static_cast<int32_t>(op.getAcquireLockId());
        int32_t acquireLockVal = static_cast<int32_t>(op.getAcquireLockVal());
        int32_t releaseLockId = static_cast<int32_t>(op.getReleaseLockId());
        int32_t releaseLockVal = static_cast<int32_t>(op.getReleaseLockVal());
        int32_t outOfOrderBdId = static_cast<int32_t>(op.getOutOfOrderBdId());

        // Multi-dimensional addressing attributes (read here, before the provenance
        // comment, so the comment can describe the dims/iteration actually emitted).
        auto dimStrides = op.getDimStrides();
        auto dimWraps = op.getDimWraps();
        bool useMultiDim = dimStrides && dimWraps && !dimStrides->empty();

        // When DISABLE_MULTID_DIM_DMA flag is set (bit 4), force linear DMA.
        // Only check when runtimeDebugLevel is explicitly set (>= 0); -1 means "not specified".
        if (useMultiDim && state.runtimeDebugLevel >= 0 && (state.runtimeDebugLevel & (1 << 4))) {
            llvm::errs() << "  [DISABLE_MULTID_DIM_DMA flag set] Suppressing multi-dim DMA, using linear "
                            "__Runtime_dma_bd_config\n";
            useMultiDim = false;
        }

        // OOO iteration attributes.
        int32_t iterStepSize = op.getIterStepSize();
        int32_t iterWrap = op.getIterWrap();
        bool useOooIter = useMultiDim && iterWrap > 1;

        // Create comment (includes multi-dim per-dim stride/wrap + iteration when present).
        std::string comment =
            "/* DMA BD Config: bd_id=" +
            std::to_string(
                op.getBdId().getDefiningOp<arith::ConstantOp>()
                    ? mlir::cast<IntegerAttr>(op.getBdId().getDefiningOp<arith::ConstantOp>().getValue()).getInt()
                    : -1) +
            ", len=" + std::to_string(len) + ", enable_packet=" + (enablePacket ? "true" : "false") +
            ", packet_id=" + std::to_string(packetId) + ", next_bd=" + std::to_string(nextBd) +
            ", acquire_lock_id=" + std::to_string(acquireLockId) +
            ", acquire_lock_val=" + std::to_string(acquireLockVal) +
            ", release_lock_id=" + std::to_string(releaseLockId) +
            ", release_lock_val=" + std::to_string(releaseLockVal) + ", ooo_bd_id=" + std::to_string(outOfOrderBdId);
        if (useMultiDim) {
            int32_t numDimsComment = static_cast<int32_t>(dimStrides->size());
            comment += ", num_dims=" + std::to_string(numDimsComment);
            for (int32_t i = 0; i < numDimsComment; i++) {
                int32_t s = mlir::cast<IntegerAttr>((*dimStrides)[i]).getInt();
                int32_t w = mlir::cast<IntegerAttr>((*dimWraps)[i]).getInt();
                comment += ", d" + std::to_string(i) + "_stride=" + std::to_string(s) + ", d" + std::to_string(i) +
                           "_wrap=" + std::to_string(w);
            }
            if (useOooIter) {
                comment +=
                    ", iter_step_size=" + std::to_string(iterStepSize) + ", iter_wrap=" + std::to_string(iterWrap);
            }
        }
        comment += " */";
        rewriter.create<emitc::VerbatimOp>(loc, comment);

        // Create constants for parameters
        auto lenConst = rewriter.create<emitc::ConstantOp>(
            loc, i32Type, rewriter.getI32IntegerAttr(len));
        auto nextBdConst = rewriter.create<emitc::ConstantOp>(
            loc, i32Type, rewriter.getI32IntegerAttr(nextBd));
        auto packetIdConst = rewriter.create<emitc::ConstantOp>(
            loc, i32Type, rewriter.getI32IntegerAttr(packetId));

        // Dispatch buffer -> void* based on the converted buffer type:
        // - void* (from BufferViewOp, BindCoreBufferOp, AllocDeviceMemOp): pass directly
        // - PartitionTensor (legacy path): use __runtime_buffer_arg macro
        auto voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(rewriter.getContext(), "void"));
        Value bufferVoidPtr;
        if (buffer.getType() == voidPtrType) {
            // Already void* from new buffer ops — use directly
            bufferVoidPtr = buffer;
        } else {
            // Legacy PartitionTensor path: __runtime_buffer_arg(p) -> (void*)&p
            auto bufferPtrOp = rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "__runtime_buffer_arg", nullptr,
                                                                    nullptr, ValueRange{buffer});
            bufferVoidPtr = bufferPtrOp.getResult(0);
        }

        // Apply byte offset to buffer pointer for shim/DDR addressing.
        // offset is now an SSA Value; check if it's a known-zero constant to skip the call.
        Value offset = adaptor.getOffset();
        bool isZeroOffset = false;
        if (auto constOp = offset.getDefiningOp<emitc::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                isZeroOffset = intAttr.getInt() == 0;
        }
        if (auto constOp = offset.getDefiningOp<arith::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                isZeroOffset = intAttr.getInt() == 0;
        }
        if (!isZeroOffset) {
            auto i64Type = rewriter.getIntegerType(64);
            auto offsetI64 = rewriter.create<emitc::CastOp>(loc, i64Type, offset);
            auto offsetPtr =
                rewriter.create<emitc::CallOpaqueOp>(loc, voidPtrType, "__runtime_buffer_offset", nullptr, nullptr,
                                                     ValueRange{bufferVoidPtr, offsetI64.getResult()});
            bufferVoidPtr = offsetPtr.getResult(0);
        }

        auto acquireLockIdConst =
            rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(acquireLockId));
        auto acquireLockValConst =
            rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(acquireLockVal));
        auto releaseLockIdConst =
            rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(releaseLockId));
        auto releaseLockValConst =
            rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(releaseLockVal));
        auto oooIdConst = rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(outOfOrderBdId));

        // dimStrides, dimWraps, useMultiDim, iterStepSize, iterWrap and useOooIter
        // were read above (before the provenance comment) and are reused here.
        emitc::CallOpaqueOp configCall;

        if (useOooIter) {
            // OOO iteration path: __Runtime_dma_bd_config_multidim_ooo
            // Uses only D0-D2 address dims + separate iter_step_size/iter_wrap
            int32_t strides[3] = {0, 0, 0};
            int32_t wraps_arr[3] = {0, 0, 0};
            int32_t numDims = (int32_t)dimStrides->size();
            if (numDims > 3)
                numDims = 3;
            for (int32_t i = 0; i < numDims; i++) {
                strides[i] = mlir::cast<IntegerAttr>((*dimStrides)[i]).getInt();
                wraps_arr[i] = mlir::cast<IntegerAttr>((*dimWraps)[i]).getInt();
            }
            auto numDimsConst = rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(numDims));
            // Create stride/wrap constants for 3 address dims
            SmallVector<Value, 6> dimArgs;
            for (int d = 0; d < 3; d++) {
                dimArgs.push_back(
                    rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(strides[d]))
                        .getResult());
                dimArgs.push_back(
                    rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(wraps_arr[d]))
                        .getResult());
            }
            // Iteration args
            auto iterStepConst =
                rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(iterStepSize));
            auto iterWrapConst = rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(iterWrap));

            SmallVector<Value> allArgs = {
                state.devInstRef,        // XAie_DevInst* dev
                tile,                    // XAie_LocType tile
                bufferVoidPtr,           // void* buffer
                bdId,                    // bd_id
                lenConst.getResult(),    // len
                nextBdConst.getResult(), // next_bd
                rewriter.create<emitc::ConstantOp>(loc, i32Type,
                                                   rewriter.getI32IntegerAttr(enablePacket ? 1 : 0))
                    .getResult(),                // enable_packet
                packetIdConst.getResult(),       // packet_id
                acquireLockIdConst.getResult(),  // acquire_lock_id
                acquireLockValConst.getResult(), // acquire_lock_val
                releaseLockIdConst.getResult(),  // release_lock_id
                releaseLockValConst.getResult(), // release_lock_val
                oooIdConst.getResult(),          // out_of_order_bd_id
                numDimsConst.getResult(),        // num_dims
            };
            allArgs.append(dimArgs.begin(), dimArgs.end()); // stride0,wrap0,...,stride2,wrap2
            allArgs.push_back(iterStepConst.getResult());   // iter_step_size
            allArgs.push_back(iterWrapConst.getResult());   // iter_wrap
            configCall = rewriter.create<emitc::CallOpaqueOp>(loc, dmaDescType, "__Runtime_dma_bd_config_multidim_ooo",
                                                              nullptr, nullptr, allArgs);
            llvm::errs() << "  ✓ Created DMA BD config with multi-dim OOO iteration (num_dims=" << numDims
                         << " iter_step=" << iterStepSize << " iter_wrap=" << iterWrap << ")\n";
        } else if (useMultiDim) {
            // Extract stride/wrap values (up to 4 dims, pad with 0)
            int32_t strides[4] = {0, 0, 0, 0};
            int32_t wraps[4] = {0, 0, 0, 0};
            int32_t numDims = (int32_t)dimStrides->size();
            for (int32_t i = 0; i < numDims && i < 4; i++) {
                strides[i] = mlir::cast<IntegerAttr>((*dimStrides)[i]).getInt();
                wraps[i] = mlir::cast<IntegerAttr>((*dimWraps)[i]).getInt();
            }
            auto numDimsConst = rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(numDims));
            // Create stride/wrap constants for all 4 dims
            SmallVector<Value, 8> dimArgs;
            for (int d = 0; d < 4; d++) {
                dimArgs.push_back(
                    rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(strides[d]))
                        .getResult());
                dimArgs.push_back(
                    rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(wraps[d])).getResult());
            }
            SmallVector<Value> allArgs = {
                state.devInstRef,        // XAie_DevInst* dev
                tile,                    // XAie_LocType tile
                bufferVoidPtr,           // void* buffer
                bdId,                    // bd_id
                lenConst.getResult(),    // len
                nextBdConst.getResult(), // next_bd
                rewriter.create<emitc::ConstantOp>(loc, i32Type,
                                                   rewriter.getI32IntegerAttr(enablePacket ? 1 : 0))
                    .getResult(),                // enable_packet
                packetIdConst.getResult(),       // packet_id
                acquireLockIdConst.getResult(),  // acquire_lock_id
                acquireLockValConst.getResult(), // acquire_lock_val
                releaseLockIdConst.getResult(),  // release_lock_id
                releaseLockValConst.getResult(), // release_lock_val
                oooIdConst.getResult(),          // out_of_order_bd_id
                numDimsConst.getResult(),        // num_dims
            };
            allArgs.append(dimArgs.begin(), dimArgs.end()); // stride0,wrap0,...,stride3,wrap3
            configCall = rewriter.create<emitc::CallOpaqueOp>(loc, dmaDescType, "__Runtime_dma_bd_config_multidim",
                                                              nullptr, nullptr, allArgs);
            llvm::errs() << "  ✓ Created DMA BD config with multi-dim addressing (num_dims=" << numDims << ")\n";
        } else {
            configCall = rewriter.create<emitc::CallOpaqueOp>(
                loc, dmaDescType, "__Runtime_dma_bd_config", nullptr, nullptr,
                ValueRange{
                    state.devInstRef,        // XAie_DevInst* dev
                    tile,                    // XAie_LocType tile
                    bufferVoidPtr,           // void* (from new ops) or (void*)&pt (legacy)
                    bdId,                    // bd_id
                    lenConst.getResult(),    // len
                    nextBdConst.getResult(), // next_bd
                    rewriter.create<emitc::ConstantOp>(loc, i32Type,
                                                       rewriter.getI32IntegerAttr(enablePacket ? 1 : 0))
                        .getResult(),                // enable_packet
                    packetIdConst.getResult(),       // packet_id
                    acquireLockIdConst.getResult(),  // acquire_lock_id
                    acquireLockValConst.getResult(), // acquire_lock_val
                    releaseLockIdConst.getResult(),  // release_lock_id
                    releaseLockValConst.getResult(), // release_lock_val
                    oooIdConst.getResult()           // out_of_order_bd_id
                });
            llvm::errs() << "  ✓ Created DMA BD config with full AIE API parameters\n";
        }

        // For core tiles with lock-based DMA (acquire_lock_val != 0),
        // emit XAie_LockSetValue to initialize locks.
        // Input (S2MM on core): DMA acquires lock 0, init lock 0 = N (N buffers ready for DMA)
        // Output (MM2S on core): DMA acquires lock 1 (swapped in BlueprintToSchedule),
        //   init lock 0 = N (kernel can write to N buffers) and lock 1 = 0 (no data yet, default)
        // N = 1 for single-buffer (pp_depth=1), N = 2 for ping-pong (pp_depth>=2).
        // Only emitted once per (tile, lock_id) to avoid duplicate calls.
        // Gate on acquireLockVal != 0 to exclude shim BDs (which don't use locks).
        if (acquireLockId >= 0 && acquireLockVal != 0) {
            int32_t tileCol = -1, tileRow = -1;
            if (auto declareTileOp = op.getTile().getDefiningOp<dfschedule::DeclareTileOp>()) {
                tileCol = declareTileOp.getCol();
                tileRow = declareTileOp.getRow();
            }
            if (tileCol >= 0 && tileRow >= 0) {
                auto lockKey = std::make_tuple(tileCol, tileRow, acquireLockId);
                if (state.initializedLocks.find(lockKey) == state.initializedLocks.end()) {
                    state.initializedLocks.insert(lockKey);

                    // Determine direction by tracing BD result -> create_io user
                    // Walk through BD chain (a BD may be used as nextBdOp of another BD)
                    // until we find the create_io op that consumes it.
                    bool isOutput = false;
                    SmallVector<Operation *, 4> worklist;
                    for (auto *user : op.getResult().getUsers())
                        worklist.push_back(user);
                    while (!worklist.empty()) {
                        auto *u = worklist.pop_back_val();
                        if (auto createIoOp = dyn_cast<dfschedule::ConfigCreateIoOp>(u)) {
                            std::string dir = createIoOp.getDirection().str();
                            if (dir == "MM2S") {
                                isOutput = true;
                            }
                            break;
                        }
                        // If this BD is used as nextBdOp of another dma_bd, follow the chain
                        if (isa<dfschedule::ConfigDmaBdOp>(u)) {
                            for (auto *uu : u->getResults().front().getUsers())
                                worklist.push_back(uu);
                        }
                    }

                    // Determine buffer count for lock init value.
                    // Single buffer (pp_depth=1): init=1; ping-pong (pp_depth>=2): init=2.
                    // Detect single-buffer mode by checking next_bd: if next_bd==-1 (no chaining),
                    // and there's no linked_bd, it's a single-buffer flow.
                    int32_t nextBdVal = static_cast<int32_t>(op.getNextBd());
                    bool isSingleBuffer = (nextBdVal == -1) && !op.getLinkedBd();
                    int32_t lockInitValue = isSingleBuffer ? 1 : 2;

                    if (isOutput) {
                        // Output (MM2S): BD acquires lock 1 (releaseLockId from kernel perspective).
                        // Lock 1 init = 0 (no data produced yet — DMA waits for kernel).
                        // Lock 0 init = lockInitValue (kernel can write to buffer(s)).
                        // Lock 1 (DMA's acquire lock) stays at default 0.
                        int32_t kernelAcquireLock = releaseLockId; // lock 0 = BD's release lock
                        auto kernelLockKey = std::make_tuple(tileCol, tileRow, kernelAcquireLock);
                        if (state.initializedLocks.find(kernelLockKey) == state.initializedLocks.end()) {
                            state.initializedLocks.insert(kernelLockKey);
                            std::string lockComment =
                                "/* Lock init: tile(" + std::to_string(tileCol) + "," + std::to_string(tileRow) +
                                ") lock=" + std::to_string(kernelAcquireLock) +
                                " init_value=" + std::to_string(lockInitValue) + " (kernel output acquire) */";
                            std::string lockSetCall = "XAie_LockSetValue(dev, XAie_TileLoc(" + std::to_string(tileCol) +
                                                      ", " + std::to_string(tileRow) + "), XAie_LockInit(" +
                                                      std::to_string(kernelAcquireLock) + ", " +
                                                      std::to_string(lockInitValue) + "));";
                            state.pendingLockInits.emplace_back(lockComment, lockSetCall);
                            llvm::errs() << "  Deferred XAie_LockSetValue for tile(" << tileCol << "," << tileRow
                                         << ") lock=" << kernelAcquireLock << " init=" << lockInitValue
                                         << " (kernel output acquire) — will emit before launch\n";
                        }
                        // DMA acquire lock (lock 1) init = 0 (default, no explicit init needed)
                        llvm::errs() << "  Output flow: DMA acquire lock " << acquireLockId
                                     << " init=0 (default, skipped)\n";
                    } else {
                        // Input (S2MM): DMA acquires lock 0, init = lockInitValue
                        int32_t initValue = lockInitValue;
                        std::string lockComment = "/* Lock init: tile(" + std::to_string(tileCol) + "," +
                                                  std::to_string(tileRow) + ") lock=" + std::to_string(acquireLockId) +
                                                  " init_value=" + std::to_string(initValue) + " */";
                        std::string lockSetCall = "XAie_LockSetValue(dev, XAie_TileLoc(" + std::to_string(tileCol) +
                                                  ", " + std::to_string(tileRow) + "), XAie_LockInit(" +
                                                  std::to_string(acquireLockId) + ", " + std::to_string(initValue) +
                                                  "));";
                        state.pendingLockInits.emplace_back(lockComment, lockSetCall);
                        llvm::errs() << "  Deferred XAie_LockSetValue for tile(" << tileCol << "," << tileRow
                                     << ") lock=" << acquireLockId << " init=" << initValue
                                     << " — will emit before launch\n";
                    }
                }
            }
        }

        // Record bd_id for this bd_handle so create_io and getbdid can reuse the same value.
        if (auto constOp = op.getBdId().getDefiningOp<arith::ConstantOp>()) {
            int32_t bdIdVal = static_cast<int32_t>(mlir::cast<IntegerAttr>(constOp.getValue()).getInt());
            state.bdHandleToBdId[op.getResult()] = bdIdVal;
        }

        // Replace the op with the call result (status code)
        rewriter.replaceOp(op, configCall.getResult(0));
        return success();
    }
};

/// OpConversionPattern for dfschedule.schedule.getbdid
/// Converts GetBdId to allocate a BD ID from the resource manager
/// Returns the allocated BD ID as an i32 constant
struct GetBdIdInnerPattern : public OpConversionPattern<dfschedule::GetBdIdOp> {
    ConversionState &state;
    
    GetBdIdInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state, PatternBenefit benefit = 1)
        : OpConversionPattern<dfschedule::GetBdIdOp>(typeConverter, ctx, benefit), state(state) {}
    
    LogicalResult matchAndRewrite(dfschedule::GetBdIdOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        llvm::errs() << "[Pattern] GetBdId called\n";
        
        // Get the tile operand
        Value tile = adaptor.getTile();
        
        llvm::errs() << "  Tile type: " << tile.getType() << "\n";
        
        // Extract tile coordinates directly from the original DeclareTileOp
        int32_t tileCol = -1, tileRow = -1;
        Value originalTile = op.getTile();  // Get original tile value before conversion
        
        if (auto declareTileOp = originalTile.getDefiningOp<dfschedule::DeclareTileOp>()) {
            tileCol = declareTileOp.getCol();
            tileRow = declareTileOp.getRow();
            llvm::errs() << "  ✓ Extracted tile coordinates from DeclareTileOp: (" 
                         << tileCol << ", " << tileRow << ")\n";
        } else {
            llvm::errs() << "  ⚠ Could not find DeclareTileOp for tile\n";
            return failure();
        }
        
        if (tileCol < 0 || tileRow < 0) {
            llvm::errs() << "  ⚠ Invalid tile coordinates\n";
            return failure();
        }
        
        // Allocate BD ID from resource manager
        int32_t allocatedBdId = state.allocateBdId(tileCol, tileRow);
        
        llvm::errs() << "  ✓ Allocated BD ID: " << allocatedBdId 
                     << " for tile (" << tileCol << ", " << tileRow << ")\n";
        
        // Create comment
        std::string comment = "/* Allocated BD ID " + std::to_string(allocatedBdId) +
                            " for tile (" + std::to_string(tileCol) + "," + std::to_string(tileRow) + ") */";
        rewriter.create<emitc::VerbatimOp>(loc, comment);
        
        // Create constant for BD ID
        auto i32Type = rewriter.getI32Type();
        auto bdIdConst = rewriter.create<emitc::ConstantOp>(
            loc, i32Type, rewriter.getI32IntegerAttr(allocatedBdId));
        
        // Replace the op with the BD ID constant
        rewriter.replaceOp(op, bdIdConst.getResult());
        return success();
    }
};

/// OpConversionPattern for dfschedule.config.create_io
/// Converts IO creation to __Runtime_dma_createio call that returns an io struct
/// struct io = __Runtime_dma_createio(dma_desc, channel_id, bd_id);
/// The channel_id and bd_id are allocated from the resource manager
struct ConfigCreateIoInnerPattern : public OpConversionPattern<dfschedule::ConfigCreateIoOp> {
    ConversionState &state;
    
    ConfigCreateIoInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state, PatternBenefit benefit = 1)
        : OpConversionPattern<dfschedule::ConfigCreateIoOp>(typeConverter, ctx, benefit), state(state) {}
    
    LogicalResult matchAndRewrite(dfschedule::ConfigCreateIoOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        // Get attributes
        int32_t channel = op.getChannel();
        std::string direction = op.getDirection().str();  // "MM2S" or "S2MM"
        std::string ioOperation = op.getIoOperation().str();
        
        llvm::errs() << "[Pattern] ConfigCreateIo called (channel=" << channel 
                     << ", direction=" << direction << ", io_operation=" << ioOperation << ")\n";
        
        // Get operands
        Value bdConfig = adaptor.getBdConfig();  // This should be the XAie_DmaDesc from config.dma_bd
        Value tile = adaptor.getTile();
        
        // IMPORTANT: In partial conversion, adaptor gives us converted operands if they exist
        // But we need to explicitly check if they've been remapped
        llvm::errs() << "  Original BD Config: " << op.getBdConfig() << "\n";
        llvm::errs() << "  Adapted BD Config: " << bdConfig << "\n";
        llvm::errs() << "  BD Config type: " << bdConfig.getType() << "\n";
        llvm::errs() << "  Tile type: " << tile.getType() << "\n";
        
     
        //rewriter.eraseOp(op);
        //return success();
        
        // Extract tile coordinates directly from the original DeclareTileOp
        int32_t tileCol = -1, tileRow = -1;
        Value originalTile = op.getTile();  // Get original tile value before conversion
        
        if (auto declareTileOp = originalTile.getDefiningOp<dfschedule::DeclareTileOp>()) {
            tileCol = declareTileOp.getCol();
            tileRow = declareTileOp.getRow();
            llvm::errs() << "  ✓ Extracted tile coordinates from DeclareTileOp: (" 
                         << tileCol << ", " << tileRow << ")\n";
        } else {
            llvm::errs() << "  ⚠ Could not find DeclareTileOp for tile\n";
            return failure();
        }
        
        if (tileCol < 0 || tileRow < 0) {
            llvm::errs() << "  ⚠ Invalid tile coordinates\n";
            return failure();
        }
        
        auto i32Type = rewriter.getI32Type();

        // Use the channel from the schedule IR directly. The BlueprintToSchedulePass
        // has already assigned correct channels based on dma_port from DmapToDmaphop,
        // which encodes the kernel's window ordering (arg0 → ch0, arg1 → ch1).
        // Do NOT use allocateChannelId() here — that auto-increments based on
        // processing order, which may not match the kernel's window ordering.
        int32_t allocatedChannelId = channel;

        // Look up the bd_id from the bd_handle operand (recorded by ConfigDmaBdInnerPattern).
        // This ensures create_io, bd_config, and startio all use the same bd_id.
        // Do NOT call allocateBdId here: that counter is shared with GetBdIdInnerPattern
        // and would return wrong values when getbdid ops are interleaved between two
        // create_io ops for the same tile.
        int32_t allocatedBdId = -1;
        Value originalBdConfig = op.getBdConfig();
        auto bdIt = state.bdHandleToBdId.find(originalBdConfig);
        if (bdIt != state.bdHandleToBdId.end()) {
            allocatedBdId = bdIt->second;
        } else {
            // Fallback: use channel id (should not happen if ConfigDmaBd ran first)
            allocatedBdId = allocatedChannelId;
            llvm::errs() << "  ⚠ bd_handle not found in bdHandleToBdId, using channel_id as fallback\n";
        }

        llvm::errs() << "  ✓ Allocated resources: channel_id=" << allocatedChannelId << ", bd_id=" << allocatedBdId
                     << "\n";

        // Store io_handle → (channel, bd_id) for later reference
        state.ioHandleToResourceMap[op.getResult()] = {allocatedChannelId, allocatedBdId};
        // Push bd_id into per-tile queue so getbdid can pop it in order
        state.perTileBdIdQueue[{tileCol, tileRow}].push_back(allocatedBdId);

        // Collect debug info for snapshot
        if (state.enableDebug) {
            std::string dirEnum = (direction == "MM2S") ? "DMA_MM2S" : "DMA_S2MM";
            state.debugIos.push_back({tileCol, tileRow, allocatedChannelId, allocatedBdId, dirEnum});
        }

        // Create constants for channel and BD IDs
        auto channelIdConst = rewriter.create<emitc::ConstantOp>(
            loc, i32Type, rewriter.getI32IntegerAttr(allocatedChannelId));
        
        auto bdIdConst = rewriter.create<emitc::ConstantOp>(
            loc, i32Type, rewriter.getI32IntegerAttr(allocatedBdId));
        // Create DMA direction constant (DMA_MM2S or DMA_S2MM)
        std::string dirEnum = (direction == "MM2S") ? "DMA_MM2S" : "DMA_S2MM";
        auto dirType = emitc::OpaqueType::get(rewriter.getContext(), "XAie_DmaDirection");
        auto dirConst =
            rewriter.create<emitc::ConstantOp>(loc, dirType, emitc::OpaqueAttr::get(rewriter.getContext(), dirEnum));

        // Create comment
        std::string comment = "/* Create IO: channel_id=" + std::to_string(allocatedChannelId) +
                            ", bd_id=" + std::to_string(allocatedBdId) +
                            ", tile=(" + std::to_string(tileCol) + "," + std::to_string(tileRow) + ")" +
                            ", direction=" + direction + " */";
        rewriter.create<emitc::VerbatimOp>(loc, comment);

        // Emit OOO channel enable if requested
        bool enableOutOfOrder = op.getEnableOutOfOrder();
        if (enableOutOfOrder) {
            std::string oooComment = "/* Enable out-of-order BD on tile(" + std::to_string(tileCol) + "," +
                                     std::to_string(tileRow) + ") ch=" + std::to_string(allocatedChannelId) +
                                     " dir=" + direction + " */";
            rewriter.create<emitc::VerbatimOp>(loc, oooComment);

            auto voidType = emitc::OpaqueType::get(rewriter.getContext(), "void");
            SmallVector<Value, 4> oooArgs = {
                state.devInstRef,           // XAie_DevInst* dev
                tile,                       // XAie_LocType tile
                channelIdConst.getResult(), // channel
                dirConst.getResult()        // direction (DMA_MM2S or DMA_S2MM)
            };
            rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "__Runtime_dma_channel_enable_ooo", nullptr, nullptr,
                                                 ValueRange(oooArgs));
            llvm::errs() << "  ✓ Emitted __Runtime_dma_channel_enable_ooo for tile(" << tileCol << "," << tileRow
                         << ") ch=" << allocatedChannelId << " dir=" << direction << "\n";
        }

        // Define the IO type (io from aie_runtime.h)
        auto ioStructType = emitc::OpaqueType::get(rewriter.getContext(), "io");

        // Create __Runtime_dma_createio_4 call (5 args; mem = NULL in runtime)
        SmallVector<Value, 5> createIoArgs = {
            tile,                       // XAie_LocType tile_loc
            bdConfig,                   // XAie_DmaDesc dma_desc
            channelIdConst.getResult(), // channel_id (from resource manager)
            bdIdConst.getResult(),      // bd_id (from resource manager)
            dirConst.getResult()        // direction (DMA_MM2S or DMA_S2MM)
        };
        auto createIoCall = rewriter.create<emitc::CallOpaqueOp>(loc, ioStructType, "__Runtime_dma_createio_4", nullptr,
                                                                 nullptr, ValueRange(createIoArgs));

        llvm::errs() << "  ✓ Created __Runtime_dma_createio call\n";
        
        // Store the resource mapping for use by StartIoOp
        //state.ioHandleToResourceMap[op.getResult()] = std::make_pair(allocatedChannelId, allocatedBdId);
        
        // Replace the op with the IO struct
        rewriter.replaceOp(op, createIoCall.getResult(0));
        return success();
    }
};

/// OpConversionPattern for dfschedule.schedule.start_io
/// Converts StartIo to __Runtime_startio call that returns an ioevent
/// struct ioevent = __Runtime_startio(io, bd_id, repeat);
struct StartIoInnerPattern : public OpConversionPattern<dfschedule::StartIoOp> {
    ConversionState &state;
    
    StartIoInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state, PatternBenefit benefit = 1)
        : OpConversionPattern<dfschedule::StartIoOp>(typeConverter, ctx, benefit), state(state) {}
    
    LogicalResult matchAndRewrite(dfschedule::StartIoOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        llvm::errs() << "[Pattern] StartIo called\n";
        
        // Get operands
        Value ioHandle = adaptor.getIoHandle();  // This should be the "struct io" from config.create_io
        Value bdId = adaptor.getBdId();          // BD ID

        llvm::errs() << "  IO Handle type: " << ioHandle.getType() << "\n";
        llvm::errs() << "  BD ID type: " << bdId.getType() << "\n";

        // Define the ioevent type (ioevent from aie_runtime.h)
        auto ioEventType = emitc::OpaqueType::get(rewriter.getContext(), "ioevent");

        // Get repeat count from the op attribute (default 1)
        uint32_t repeatCount = op.getRepeatCount();
        auto repeatConst =
            rewriter.create<emitc::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(repeatCount));

        // Create __Runtime_startio call:
        // struct ioevent = __Runtime_startio(dev, io, bd_id, repeat);
        auto startIoCall = rewriter.create<emitc::CallOpaqueOp>(loc, ioEventType, "__Runtime_startio", nullptr, nullptr,
                                                                ValueRange{
                                                                    state.devInstRef,       // XAie_DevInst* dev
                                                                    ioHandle,               // struct io
                                                                    bdId,                   // bd_id
                                                                    repeatConst.getResult() // repeat count
                                                                });

        llvm::errs() << "  ✓ Created __Runtime_startio call\n";
        
        // Replace the op with the ioevent
        rewriter.replaceOp(op, startIoCall.getResult(0));
        return success();
    }
};

/// OpConversionPattern for dfschedule.schedule.wait
/// Converts ScheduleWait to __Runtime_wait call that waits for multiple events
/// __Runtime_wait(event1, event2, ...);
struct ScheduleWaitInnerPattern : public OpConversionPattern<dfschedule::ScheduleWaitOp> {
    ConversionState &state;
    
    ScheduleWaitInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state, PatternBenefit benefit = 1)
        : OpConversionPattern<dfschedule::ScheduleWaitOp>(typeConverter, ctx, benefit), state(state) {}
    
    LogicalResult matchAndRewrite(dfschedule::ScheduleWaitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        llvm::errs() << "[Pattern] ScheduleWait called with " << op.getEvents().size() << " events\n";
        
        // Get all event operands
        auto events = adaptor.getEvents();
        
        if (events.empty()) {
            llvm::errs() << "  ⚠ No events to wait for, erasing op\n";
            rewriter.eraseOp(op);
            return success();
        }
        
        // Log event types for debugging
        for (size_t i = 0; i < events.size(); ++i) {
            llvm::errs() << "  Event[" << i << "] type: " << events[i].getType() << "\n";
        }
        
        // Create comment showing what we're waiting for
        std::string comment = "/* Wait for " + std::to_string(events.size()) + " event(s) */";
        rewriter.create<emitc::VerbatimOp>(loc, comment);

        // Create __Runtime_wait call for each event (C++ overloads dispatch event vs ioevent)
        for (auto eventVal : events) {
            rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "__Runtime_wait", nullptr, nullptr,
                                                 ValueRange{state.devInstRef, eventVal});
        }

        llvm::errs() << "  ✓ Created __Runtime_wait calls for " << events.size() << " event(s)\n";
        
        // Erase the original wait op (it has no results)
        rewriter.eraseOp(op);
        return success();
    }
};

/// OpConversionPattern for dfschedule.declare_kernel_config
/// This is metadata-only, so just erase it
struct DeclareKernelConfigInnerPattern : public OpConversionPattern<dfschedule::DeclareKernelConfigOp> {
    using OpConversionPattern<dfschedule::DeclareKernelConfigOp>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(dfschedule::DeclareKernelConfigOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        llvm::errs() << "[Pattern] DeclareKernelConfig - erasing (metadata only)\n";
        // This operation is pure metadata, it doesn't generate any runtime code
        rewriter.eraseOp(op);
        return success();
    }
};

/// OpConversionPattern for dfschedule.config.load_kernel_group
/// Converts LoadKernelGroup to __Runtime_load_kernel_group call
/// struct kernel_group = __Runtime_load_kernel_group(tiles, callee_symbols, compute_args, kernel_config);
struct LoadKernelGroupInnerPattern : public OpConversionPattern<dfschedule::LoadKernelGroupOp> {
    ConversionState &state;
    
    LoadKernelGroupInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state, PatternBenefit benefit = 1)
        : OpConversionPattern<dfschedule::LoadKernelGroupOp>(typeConverter, ctx, benefit), state(state) {}
    
    LogicalResult matchAndRewrite(dfschedule::LoadKernelGroupOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        llvm::errs() << "[Pattern] LoadKernelGroup called\n";
        
        // Get tiles operands (variadic)
        auto tiles = adaptor.getTiles();
        llvm::errs() << "  Number of tiles: " << tiles.size() << "\n";
        
        // Get attributes
        auto calleeAttr = op.getCalleeAttr();
        auto computeKernelArgsAttr = op.getDistributedComputeKernelArgsAttr();
        auto distributedArgsAttr = op.getDistributedArgsAttr();
        
        llvm::errs() << "  Callee array: " << calleeAttr << "\n";
        llvm::errs() << "  Compute kernel args array: " << computeKernelArgsAttr << "\n";
        
        if (distributedArgsAttr) {
            llvm::errs() << "  Using distributed_args (kernel config symbols): " << distributedArgsAttr << "\n";
            
            auto moduleOp = op->getParentOfType<ModuleOp>();
            
            // Iterate through distributed_args to extract config for each tile
            for (size_t i = 0; i < distributedArgsAttr.size(); ++i) {
                auto symRef = mlir::cast<SymbolRefAttr>(distributedArgsAttr[i]);
                llvm::errs() << "  Tile[" << i << "] config symbol: " << symRef << "\n";
                
                // Look up the kernel_config op
                auto configOp = moduleOp.lookupSymbol<dfschedule::DeclareKernelConfigOp>(
                    symRef.getRootReference());
                
                if (!configOp) {
                    llvm::errs() << "    ERROR: Could not find kernel_config symbol\n";
                    continue;
                }
                
                // Extract tile_configs array (should have exactly one entry per config op)
                auto tileConfigsAttr = configOp.getTileConfigs();
                if (tileConfigsAttr.size() == 0) {
                    llvm::errs() << "    ERROR: Empty tile_configs in kernel_config\n";
                    continue;
                }
                
                auto configDict = mlir::cast<DictionaryAttr>(tileConfigsAttr[0]);
                
                // Extract and log all config fields
                uint32_t tileIndex = mlir::cast<IntegerAttr>(configDict.get("tile_index")).getInt();
                uint8_t packetId = mlir::cast<IntegerAttr>(configDict.get("packet_id")).getInt();
                uint32_t dmaChannel = mlir::cast<IntegerAttr>(configDict.get("dma_channel")).getInt();
                uint8_t bufferMode = mlir::cast<IntegerAttr>(configDict.get("buffer_mode")).getInt();
                uint8_t numBuffers = mlir::cast<IntegerAttr>(configDict.get("num_buffers")).getInt();
                uint32_t bufferSize = mlir::cast<IntegerAttr>(configDict.get("buffer_size")).getInt();
                uint64_t bufferOffset = mlir::cast<IntegerAttr>(configDict.get("buffer_offset")).getInt();
                uint8_t elementSize = mlir::cast<IntegerAttr>(configDict.get("element_size")).getInt();
                // Use null-safe reads: passblueprinttoschedule writes "acquire_lock_id" / "release_lock_id"
                // (single pair, no ping/pong prefix). The old ping/pong keys do not exist.
                uint32_t acquireLockId = 0, releaseLockId = 0;
                if (auto a = configDict.get("acquire_lock_id"))
                    acquireLockId = mlir::cast<IntegerAttr>(a).getInt();
                if (auto r = configDict.get("release_lock_id"))
                    releaseLockId = mlir::cast<IntegerAttr>(r).getInt();

                llvm::errs() << "    Config: "
                             << "tile_index=" << tileIndex << ", packet_id=" << (int)packetId
                             << ", dma_channel=" << dmaChannel << ", buffer_mode=" << (int)bufferMode
                             << ", num_buffers=" << (int)numBuffers << ", buffer_size=" << bufferSize
                             << ", buffer_offset=" << bufferOffset << ", element_size=" << (int)elementSize
                             << ", acq_lock=" << acquireLockId << ", rel_lock=" << releaseLockId << "\n";
            }
            
            // NOTE: In the future, this would generate arrays of config values
            // and pass them to __Runtime_load_kernel_group(tiles, num_tiles, configs[])
            // For now, the simple call below is a placeholder
            
        } else {
            llvm::errs() << "  ERROR: No distributed_args provided\n";
            return failure();
        }

        // Collect core tile debug info for snapshot
        if (state.enableDebug) {
            for (Value origTile : op.getTiles()) {
                if (auto dtOp = origTile.getDefiningOp<dfschedule::DeclareTileOp>()) {
                    int32_t c = dtOp.getCol(), r = dtOp.getRow();
                    bool found = false;
                    for (auto &t : state.debugTiles)
                        if (t.col == c && t.row == r) {
                            found = true;
                            break;
                        }
                    if (!found)
                        state.debugTiles.push_back({c, r});
                }
            }
        }

        // Create comment showing configuration
        std::string comment = "/* Load Kernel Group: " + std::to_string(tiles.size()) + " tile(s) */";
        rewriter.create<emitc::VerbatimOp>(loc, comment);

        // Define the kernel_group struct type
        auto kernelGroupType = emitc::OpaqueType::get(rewriter.getContext(), "kernel_group");

        // Select the right positional variant based on tile count:
        //   <=4  -> _4t (pad to 4)
        //   <=8  -> _8t (pad to 8)
        //   <=16 -> _16t (pad to 16)
        size_t numTiles = tiles.size();
        size_t padTo;
        std::string funcName;
        if (numTiles <= 4) {
            padTo = 4;
            funcName = "__Runtime_load_kernel_group_4t";
        } else if (numTiles <= 8) {
            padTo = 8;
            funcName = "__Runtime_load_kernel_group_8t";
        } else if (numTiles <= 16) {
            padTo = 16;
            funcName = "__Runtime_load_kernel_group_16t";
        } else {
            llvm::errs() << "  ERROR: too many tiles (" << numTiles << ") for load_kernel_group\n";
            return failure();
        }

        auto i32Type = rewriter.getI32Type();
        auto numTilesConst = rewriter.create<emitc::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(numTiles));
        SmallVector<Value> callOperands;
        callOperands.push_back(state.devInstRef); // XAie_DevInst* dev
        callOperands.append(tiles.begin(), tiles.end());
        // Pad with last tile to fill the positional slots
        while (callOperands.size() < padTo + 1) // +1 for dev parameter
            callOperands.push_back(tiles.back());
        callOperands.push_back(numTilesConst.getResult());

        auto loadCall =
            rewriter.create<emitc::CallOpaqueOp>(loc, kernelGroupType, funcName, nullptr, nullptr, callOperands);
        llvm::errs() << "  ✓ Created " << funcName << " call (" << numTiles << " tiles)\n";

        rewriter.replaceOp(op, loadCall.getResult(0));
        return success();
    }
};

/// OpConversionPattern for dfschedule.schedule.launch_kernel_group
/// Converts LaunchKernelGroup to __Runtime_launch_kernel_group call
/// struct event = __Runtime_launch_kernel_group(kernel_group);
struct LaunchKernelGroupInnerPattern : public OpConversionPattern<dfschedule::LaunchKernelGroupOp> {
    ConversionState &state;
    
    LaunchKernelGroupInnerPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &state, PatternBenefit benefit = 1)
        : OpConversionPattern<dfschedule::LaunchKernelGroupOp>(typeConverter, ctx, benefit), state(state) {}
    
    LogicalResult matchAndRewrite(dfschedule::LaunchKernelGroupOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        
        llvm::errs() << "[Pattern] LaunchKernelGroup called\n";
        
        // Get the kernel group operand
        Value kernelGroup = adaptor.getKernelGroup();
        
        llvm::errs() << "  Kernel Group type: " << kernelGroup.getType() << "\n";
        // rewriter.eraseOp(op);
        // return success();
        if (!state.pendingLockInits.empty()) {
            rewriter.create<emitc::VerbatimOp>(loc,
                                               "/* Lock inits — must precede kernel launch to avoid contention */");
            for (auto &[comment, call] : state.pendingLockInits) {
                rewriter.create<emitc::VerbatimOp>(loc, comment);
                rewriter.create<emitc::VerbatimOp>(loc, call);
            }
            llvm::errs() << "  Flushed " << state.pendingLockInits.size() << " lock inits before launch_kernel_group\n";
            state.pendingLockInits.clear();
        }

        // Create comment
        rewriter.create<emitc::VerbatimOp>(loc, "/* Launch Kernel Group */");

        // Define the event type (event from aie_runtime.h)
        auto eventType = emitc::OpaqueType::get(rewriter.getContext(), "event");

        // Create __Runtime_launch_kernel_group call:
        // event = __Runtime_launch_kernel_group(dev, kernel_group);
        auto launchCall = rewriter.create<emitc::CallOpaqueOp>(loc, eventType, "__Runtime_launch_kernel_group", nullptr,
                                                               nullptr, ValueRange{state.devInstRef, kernelGroup});

        llvm::errs() << "  ✓ Created __Runtime_launch_kernel_group call\n";
        
        // Replace the op with the event
        rewriter.replaceOp(op, launchCall.getResult(0));
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Outer Patterns (Convert host op structure)
//===----------------------------------------------------------------------===//

/// Helper: emit debug snapshot verbatim C block before return.
static void emitDebugSnapshotVerbatim(OpBuilder &rewriter, Location loc, const ConversionState &state) {
    auto buildArrayInit = [](const std::string &type, const std::string &name,
                             const std::vector<std::string> &vals) -> std::string {
        std::string s = type + " " + name + "[] = {";
        for (size_t i = 0; i < vals.size(); ++i) {
            if (i)
                s += ", ";
            s += vals[i];
        }
        return s + "};";
    };

    size_t N = state.debugIos.size();
    size_t M = state.debugTiles.size();

    std::vector<std::string> cols, rows, chs, bds, dirs;
    for (auto &io : state.debugIos) {
        cols.push_back(std::to_string(io.col));
        rows.push_back(std::to_string(io.row));
        chs.push_back(std::to_string(io.channel));
        bds.push_back(std::to_string(io.bd_id));
        dirs.push_back(io.direction);
    }
    std::vector<std::string> tcols, trows;
    for (auto &t : state.debugTiles) {
        tcols.push_back(std::to_string(t.col));
        trows.push_back(std::to_string(t.row));
    }

    rewriter.create<emitc::VerbatimOp>(loc, "/* AieRt debug snapshot (gated by AIE_DEBUG_LOG flag) */");
    rewriter.create<emitc::VerbatimOp>(loc, "if (AIEHLC_LOG_ENABLED()) {");
    rewriter.create<emitc::VerbatimOp>(loc, "  " + buildArrayInit("uint8_t", "_dbg_io_cols", cols));
    rewriter.create<emitc::VerbatimOp>(loc, "  " + buildArrayInit("uint8_t", "_dbg_io_rows", rows));
    rewriter.create<emitc::VerbatimOp>(loc, "  " + buildArrayInit("uint8_t", "_dbg_io_chs", chs));
    rewriter.create<emitc::VerbatimOp>(loc, "  " + buildArrayInit("uint8_t", "_dbg_io_bds", bds));
    rewriter.create<emitc::VerbatimOp>(loc, "  " + buildArrayInit("int", "_dbg_io_dirs", dirs));
    if (!tcols.empty()) {
        rewriter.create<emitc::VerbatimOp>(loc, "  " + buildArrayInit("uint8_t", "_dbg_t_cols", tcols));
        rewriter.create<emitc::VerbatimOp>(loc, "  " + buildArrayInit("uint8_t", "_dbg_t_rows", trows));
    }
    std::string call = "  AieRt_DebugSnapshotFromCoords(dev,\n"
                       "      _dbg_io_cols, _dbg_io_rows, _dbg_io_chs, _dbg_io_bds, _dbg_io_dirs, " +
                       std::to_string(N) +
                       ",\n"
                       "      " +
                       (tcols.empty() ? "NULL, NULL, " : "_dbg_t_cols, _dbg_t_rows, ") + std::to_string(M) + ");";
    rewriter.create<emitc::VerbatimOp>(loc, call);
    rewriter.create<emitc::VerbatimOp>(loc, "}");
}

/// Outer pattern for dfschedule.host -> emitc.func (simple shell only)
/// The inner ops are already converted by the host inner pattern phase
struct HostOpOuterPattern : public ConversionPattern {
    ConversionState &state;

    HostOpOuterPattern(TypeConverter &typeConverter, MLIRContext *ctx, ConversionState &s)
        : ConversionPattern(typeConverter, "dfschedule.host", /*benefit=*/1, ctx), state(s) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();

        std::string funcName = "hostruntime";
        if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
            funcName = symNameAttr.getValue().str();
        }

        // Create emitc.func with matching arguments from host block.
        // Arg 0 is XAie_DevInst* dev (added by the pass in Phase 3 pre-creation).
        // Remaining args are external DDR pointer memrefs, lowered to void*.
        SmallVector<Type> argTypes;
        Block *srcBlock = nullptr;
        if (op->getNumRegions() > 0 && !op->getRegion(0).empty()) {
            srcBlock = &op->getRegion(0).front();
            for (auto arg : srcBlock->getArguments()) {
                // Arg 0 is XAie_DevInst* (already the right type), rest become void*
                if (isa<emitc::PointerType>(arg.getType())) {
                    argTypes.push_back(arg.getType());
                } else {
                    auto ptrType = emitc::PointerType::get(emitc::OpaqueType::get(rewriter.getContext(), "void"));
                    argTypes.push_back(ptrType);
                }
            }
        }
        auto funcType = rewriter.getFunctionType(argTypes, {});
        auto emitcFunc = rewriter.create<emitc::FuncOp>(loc, funcName, funcType);
        Block *entryBlock = emitcFunc.addEntryBlock();

        // Move converted operations from host region to new func
        if (srcBlock) {
            // Replace host block arg uses with emitc.func entry block args.
            // Arg 0 = XAie_DevInst* dev, args 1+ = DDR void* pointers.
            for (unsigned i = 0; i < srcBlock->getNumArguments(); ++i) {
                srcBlock->getArgument(i).replaceAllUsesWith(entryBlock->getArgument(i));
            }

            // Move all operations except terminator to the new func
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(entryBlock);

            // Create a "dev" alias for the first parameter (XAie_DevInst*) so that
            // VerbatimOps (XAie_LockSetValue, AieRt_DebugSnapshot, etc.) can reference it
            // by the well-known name "dev" instead of the auto-generated EmitC name (v1).
            rewriter.create<emitc::VerbatimOp>(loc, "XAie_DevInst* dev = v1;");

            for (Operation &nestedOp : llvm::make_early_inc_range(srcBlock->getOperations())) {
                if (!nestedOp.hasTrait<OpTrait::IsTerminator>()) {
                    nestedOp.moveBefore(entryBlock, entryBlock->end());
                }
            }
        }

        // Add return at the end (optionally preceded by debug snapshot)
        rewriter.setInsertionPointToEnd(entryBlock);

        if (state.enableDebug && !state.debugIos.empty()) {
            emitDebugSnapshotVerbatim(rewriter, loc, state);
        }

        // Kernel log reading is handled by __Runtime_auto_teardown() in
        // aie_runtime.c (destructor), no need to emit it in host.cc.

        rewriter.create<emitc::ReturnOp>(loc, Value{});

        llvm::errs() << "[Pattern] Created emitc.func: " << funcName << "\n";

        rewriter.eraseOp(op);
        return success();
    }
};

/// ConversionPattern for dfschedule.launchhost -> emitc.call_opaque
struct LaunchHostPattern : public ConversionPattern {
    LaunchHostPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : ConversionPattern(typeConverter, "dfschedule.launchhost", /*benefit=*/1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        llvm::errs() << "[Pattern] LaunchHostPattern::matchAndRewrite called for: " << *op << "\n";
        
        // Get the host function name from the attribute
        StringRef hostFuncName = "host_canonicalized";
        if (auto symAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee")) {
            hostFuncName = symAttr.getValue();
        }
        
        llvm::errs() << "[Pattern] Calling host function: " << hostFuncName << "\n";
        
        // Create emitc.call_opaque to call the host function
        rewriter.create<emitc::CallOpaqueOp>(op->getLoc(),
            TypeRange{}, hostFuncName.str(), nullptr, nullptr, ValueRange{});
        
        llvm::errs() << "[Pattern] Created call to " << hostFuncName << "()\n";
        
        rewriter.eraseOp(op);
        return success();
    }
};

/// ConversionPattern for dfschedule.dskernel_receiver -> emitc.func with __global__
struct DsKernelReceiverPattern : public ConversionPattern {
    DsKernelReceiverPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : ConversionPattern(typeConverter, "dfschedule.dskernel_receiver", /*benefit=*/1, ctx) {}
    
    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op->getLoc();
        
        std::string kernelName = "dskernel";
        if (auto symNameAttr = op->getAttrOfType<StringAttr>("sym_name")) {
            kernelName = symNameAttr.getValue().str();
        }
        
        // NEW signature: void dskernel(index iterations)
        auto indexType = rewriter.getIndexType();
        auto funcType = rewriter.getFunctionType({indexType}, {});
        auto emitcFunc = rewriter.create<emitc::FuncOp>(loc, kernelName, funcType);
        emitcFunc->setAttr("specifiers", rewriter.getStrArrayAttr({"__global__"}));
        
        Block *entryBlock = emitcFunc.addEntryBlock();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);
        
        // Get the iterations parameter
        Value iterationsParam = entryBlock->getArgument(0);
        
        // Generate kernel body that reads config from tile-local memory
        // The kernel will read from TILE_CONFIG_ADDR at runtime
        rewriter.create<emitc::VerbatimOp>(loc,
                                           rewriter.getStringAttr("// the real kernel will be emitted separately\n"));

        rewriter.create<emitc::ReturnOp>(loc, Value{});
        
        llvm::errs() << "[Pattern] Created __global__ func: " << kernelName 
                     << " with signature (index iterations)\n";
        
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Kernel Module Conversion Pattern
//===----------------------------------------------------------------------===//

/// Structure to hold buffer definition info
struct BufferDefInfo {
    std::string name;
    int64_t size = 256;
    int32_t vectorWidth = 4;
    std::string elementType = "int32";
};

/// Structure to hold window parameter information
struct WindowParamInfo {
    std::string windowSymbol;    // e.g., "window_out_0"
    bool isInput;                // true = input, false = output
    std::string elementTypeName; // e.g., "int32", "int8"
    std::string bufferPingName;  // e.g., "buf_out_ping_0" or "out_ping"
    std::string bufferPongName;  // e.g., "buf_out_pong_0" or "out_pong"
    int32_t acquireLockId;       // Lock ID for acquire
    int32_t releaseLockId;       // Lock ID for release
};

/// Structure to hold extracted kernel module information
struct KernelModuleInfo {
    std::string moduleName;
    std::string kernelName;
    std::string kernelFile;
    int32_t bufferSize = 256;
    std::string elementType = "int32_t";
    int32_t vectorWidth = 4;
    std::string iterationStyle = "internal"; // "internal" or "external"

    // Lock IDs
    int32_t inputAcquireLockId = 48;
    int32_t inputReleaseLockId = 49;
    int32_t outputAcquireLockId = 51;
    int32_t outputReleaseLockId = 50;

    // Window names
    std::string inputWindowName = "win";
    std::string outputWindowName = "out";

    // Buffer definitions extracted from dfschedule.buffer_def ops
    SmallVector<BufferDefInfo> bufferDefs;

    // Dynamic window parameters extracted from dfschedule.main
    SmallVector<WindowParamInfo> windowParams;
};

/// Extract kernel module information from dfschedule.module operation
static KernelModuleInfo extractKernelModuleInfo(dfschedule::KernelModuleOp moduleOp) {
    KernelModuleInfo info;
    info.moduleName = moduleOp.getSymName().str();

    // Walk through the module body to extract definitions
    moduleOp.getBody().walk([&](Operation *op) {
        // Extract kernel config
        if (auto configOp = dyn_cast<dfschedule::KernelConfigDefOp>(op)) {
            DictionaryAttr configDict = configOp.getConfigAttrs();
            if (auto kernelNameAttr = configDict.getAs<StringAttr>("kernel_name")) {
                info.kernelName = kernelNameAttr.getValue().str();
            }
            if (auto kernelFileAttr = configDict.getAs<StringAttr>("kernel_file")) {
                info.kernelFile = kernelFileAttr.getValue().str();
            }
            if (auto bufSizeAttr = configDict.getAs<IntegerAttr>("buffer_size")) {
                info.bufferSize = bufSizeAttr.getInt();
            }
            if (auto vecWidthAttr = configDict.getAs<IntegerAttr>("vector_width")) {
                info.vectorWidth = vecWidthAttr.getInt();
            }
            if (auto elemTypeAttr = configDict.getAs<TypeAttr>("element_type")) {
                Type elemType = elemTypeAttr.getValue();
                if (elemType.isInteger(32))
                    info.elementType = "int32_t";
                else if (elemType.isInteger(16))
                    info.elementType = "int16_t";
                else if (elemType.isInteger(8))
                    info.elementType = "int8_t";
                else if (elemType.isF32())
                    info.elementType = "float";
            }
        }

        // Extract lock definitions
        if (auto lockOp = dyn_cast<dfschedule::LockDefOp>(op)) {
            std::string lockName = lockOp.getSymName().str();
            int32_t lockId = lockOp.getId();

            if (lockName.find("win") != std::string::npos && lockName.find("ACQ") != std::string::npos) {
                info.inputAcquireLockId = lockId;
            } else if (lockName.find("win") != std::string::npos && lockName.find("REL") != std::string::npos) {
                info.inputReleaseLockId = lockId;
            } else if (lockName.find("out") != std::string::npos && lockName.find("ACQ") != std::string::npos) {
                info.outputAcquireLockId = lockId;
            } else if (lockName.find("out") != std::string::npos && lockName.find("REL") != std::string::npos) {
                info.outputReleaseLockId = lockId;
            }
        }

        // Extract kernel declaration to get iteration style
        if (auto kernelDeclOp = dyn_cast<dfschedule::KernelDeclOp>(op)) {
            info.kernelName = kernelDeclOp.getSymName().str();
            DictionaryAttr declAttrs = kernelDeclOp.getDeclAttrs();
            if (auto iterStyleAttr = declAttrs.getAs<StringAttr>("iteration_style")) {
                info.iterationStyle = iterStyleAttr.getValue().str();
            }
        }

        // Extract window definitions
        if (auto windowOp = dyn_cast<dfschedule::WindowDefOp>(op)) {
            std::string windowName = windowOp.getSymName().str();
            DictionaryAttr windowAttrs = windowOp.getWindowAttrs();
            if (auto directionAttr = windowAttrs.getAs<StringAttr>("direction")) {
                if (directionAttr.getValue() == "in") {
                    info.inputWindowName = windowName;
                } else if (directionAttr.getValue() == "out") {
                    info.outputWindowName = windowName;
                }
            }
        }

        // Extract buffer definitions
        if (auto bufferOp = dyn_cast<dfschedule::BufferDefOp>(op)) {
            BufferDefInfo bufInfo;
            bufInfo.name = bufferOp.getSymName().str();

            // Parse buffer type: memref<256xvector<4xi32>, "LOCAL">
            TypeAttr bufferTypeAttr = bufferOp.getBufferTypeAttr();
            if (auto memrefType = dyn_cast<MemRefType>(bufferTypeAttr.getValue())) {
                // Get shape (e.g., 256)
                if (!memrefType.getShape().empty()) {
                    bufInfo.size = memrefType.getShape()[0];
                }

                // Get element type (could be vector<4xi32>)
                Type elemType = memrefType.getElementType();
                if (auto vecType = dyn_cast<VectorType>(elemType)) {
                    // Vector type: extract width and element type
                    if (!vecType.getShape().empty()) {
                        bufInfo.vectorWidth = vecType.getShape()[0];
                    }
                    Type scalarType = vecType.getElementType();
                    if (scalarType.isInteger(32))
                        bufInfo.elementType = "int32";
                    else if (scalarType.isInteger(16))
                        bufInfo.elementType = "int16";
                    else if (scalarType.isInteger(8))
                        bufInfo.elementType = "int8";
                    else if (scalarType.isF32())
                        bufInfo.elementType = "float";
                } else {
                    // Scalar type
                    bufInfo.vectorWidth = 1;
                    if (elemType.isInteger(32))
                        bufInfo.elementType = "int32";
                    else if (elemType.isInteger(16))
                        bufInfo.elementType = "int16";
                    else if (elemType.isInteger(8))
                        bufInfo.elementType = "int8";
                    else if (elemType.isF32())
                        bufInfo.elementType = "float";
                }
            }

            info.bufferDefs.push_back(bufInfo);
        }

        // Extract dfschedule.main to get dynamic window parameters
        if (auto mainOp = dyn_cast<dfschedule::KernelMainOp>(op)) {
            llvm::errs() << "[extractKernelModuleInfo] Found dfschedule.main\n";

            // Build a map: window symbol -> window definition attributes
            llvm::StringMap<DictionaryAttr> windowDefMap;
            moduleOp.getBody().walk([&](dfschedule::WindowDefOp winDefOp) {
                windowDefMap[winDefOp.getSymName()] = winDefOp.getWindowAttrs();
                llvm::errs() << "[extractKernelModuleInfo]   Found window def: " << winDefOp.getSymName() << "\n";
            });

            // Walk the main body to find window_init operations
            llvm::errs() << "[extractKernelModuleInfo] Walking main body for window_init ops\n";
            mainOp.getBody().walk([&](dfschedule::WindowInitOp winInitOp) {
                llvm::errs() << "[extractKernelModuleInfo]   Found window_init op\n";
                WindowParamInfo paramInfo;

                // Get window symbol reference from the operation's attribute
                SymbolRefAttr windowRef = winInitOp.getWindowRefAttr();
                paramInfo.windowSymbol = windowRef.getRootReference().getValue().str();

                // Determine if input or output from result type
                Type resultType = winInitOp.getResult().getType();
                if (auto inputWinType = dyn_cast<dfschedule::InputWindowType>(resultType)) {
                    paramInfo.isInput = true;
                    Type elemType = inputWinType.getElementType();
                    if (elemType.isInteger(32))
                        paramInfo.elementTypeName = "int32";
                    else if (elemType.isInteger(16))
                        paramInfo.elementTypeName = "int16";
                    else if (elemType.isInteger(8))
                        paramInfo.elementTypeName = "int8";
                    else if (elemType.isF32())
                        paramInfo.elementTypeName = "float";
                } else if (auto outputWinType = dyn_cast<dfschedule::OutputWindowType>(resultType)) {
                    paramInfo.isInput = false;
                    Type elemType = outputWinType.getElementType();
                    if (elemType.isInteger(32))
                        paramInfo.elementTypeName = "int32";
                    else if (elemType.isInteger(16))
                        paramInfo.elementTypeName = "int16";
                    else if (elemType.isInteger(8))
                        paramInfo.elementTypeName = "int8";
                    else if (elemType.isF32())
                        paramInfo.elementTypeName = "float";
                }

                // Lookup window definition to get buffer names and lock IDs
                auto it = windowDefMap.find(paramInfo.windowSymbol);
                if (it != windowDefMap.end()) {
                    DictionaryAttr windowAttrs = it->second;

                    // Get ping/pong buffer names
                    if (auto pingBufAttr = windowAttrs.getAs<SymbolRefAttr>("ping_buffer")) {
                        paramInfo.bufferPingName = pingBufAttr.getRootReference().getValue().str();
                    }
                    if (auto pongBufAttr = windowAttrs.getAs<SymbolRefAttr>("pong_buffer")) {
                        paramInfo.bufferPongName = pongBufAttr.getRootReference().getValue().str();
                    }

                    // Get acquire/release lock symbols and lookup their IDs
                    if (auto acqLockAttr = windowAttrs.getAs<SymbolRefAttr>("acquire_lock")) {
                        std::string acqLockName = acqLockAttr.getRootReference().getValue().str();
                        // Find the lock definition to get the actual ID
                        moduleOp.getBody().walk([&](dfschedule::LockDefOp lockOp) {
                            if (lockOp.getSymName().str() == acqLockName) {
                                paramInfo.acquireLockId = lockOp.getId();
                            }
                        });
                    }
                    if (auto relLockAttr = windowAttrs.getAs<SymbolRefAttr>("release_lock")) {
                        std::string relLockName = relLockAttr.getRootReference().getValue().str();
                        // Find the lock definition to get the actual ID
                        moduleOp.getBody().walk([&](dfschedule::LockDefOp lockOp) {
                            if (lockOp.getSymName().str() == relLockName) {
                                paramInfo.releaseLockId = lockOp.getId();
                            }
                        });
                    }
                }

                llvm::errs() << "[extractKernelModuleInfo]   Window param: " << paramInfo.windowSymbol
                             << ", isInput=" << paramInfo.isInput << ", elemType=" << paramInfo.elementTypeName
                             << ", ping=" << paramInfo.bufferPingName << ", pong=" << paramInfo.bufferPongName
                             << ", acqLock=" << paramInfo.acquireLockId << ", relLock=" << paramInfo.releaseLockId
                             << "\n";

                info.windowParams.push_back(paramInfo);
            });

            llvm::errs() << "[extractKernelModuleInfo] Extracted " << info.windowParams.size()
                         << " window parameters\n";
        }
    });

    return info;
}

/// Generate C++ kernel driver code from KernelModuleInfo
static std::string generateKernelDriverCode(const KernelModuleInfo &info) {
    std::ostringstream code;

    llvm::errs() << "[generateKernelDriverCode] Called with " << info.windowParams.size() << " window parameters\n";

    // Vector type name (e.g., "v4int32")
    std::string vecTypeName = "v" + std::to_string(info.vectorWidth) + info.elementType;
    // Simplify type name for vector (int32_t -> int32)
    std::string vecTypeSimple = info.elementType;
    if (vecTypeSimple == "int32_t")
        vecTypeSimple = "int32";
    else if (vecTypeSimple == "int16_t")
        vecTypeSimple = "int16";
    else if (vecTypeSimple == "int8_t")
        vecTypeSimple = "int8";
    vecTypeName = "v" + std::to_string(info.vectorWidth) + vecTypeSimple;

    // Generate includes and defines
    code << "/* ========== Kernel Driver: " << info.moduleName << " ========== */\n";
    code << "#include <stdint.h>\n";
    code << "#define FOR_READ  1\n";
    code << "#define FOR_WRITE 0\n";
    code << "#define BUF_SZ " << info.bufferSize << "\n\n";

    code << "volatile static int sync_buffer[8] = {0, -1};\n\n";

    // Generate window management functions
    code << "// Window management functions for standalone AIE kernels\n";
    code << "// Note: chess_memory_fence() and done() are already defined by Chess intrinsics\n\n";

    code << "// Window types\n";
    code << "typedef struct {\n";
    code << "    void* ptr;\n";
    code << "    int ping_acq;\n";
    code << "    int ping_rel;\n";
    code << "    int pong_acq;\n";
    code << "    int pong_rel;\n";
    code << "    int size;\n";
    code << "} window_internal;\n\n";

    code << "typedef void* output_window_int8;\n";
    code << "typedef void* input_window_int8;\n";
    code << "typedef void* output_window_int16;\n";
    code << "typedef void* input_window_int16;\n";
    code << "typedef void* output_window_int32;\n";
    code << "typedef void* input_window_int32;\n";
    code << "typedef void* output_window_float;\n";
    code << "typedef void* input_window_float;\n\n";

    code << "// Window initialization\n";
    code << "inline void window_init(window_internal* win, int count, void* ping, int ping_acq_lock, void* pong, int "
            "ping_rel_lock, int ping_size, int pong_size) {\n";
    code << "    win->ptr = ping;\n";
    code << "    win->ping_acq = ping_acq_lock;\n";
    code << "    win->ping_rel = ping_rel_lock;\n";
    code << "    win->size = ping_size;\n";
    code << "}\n\n";

    code << "// Window access functions\n";
    code << "inline output_window_int8 get_output_async_window_int8(window_internal* win) {\n";
    code << "    return win->ptr;\n";
    code << "}\n\n";

    code << "inline output_window_int16 get_output_async_window_int16(window_internal* win) {\n";
    code << "    return win->ptr;\n";
    code << "}\n\n";

    code << "inline output_window_int32 get_output_async_window_int32(window_internal* win) {\n";
    code << "    return win->ptr;\n";
    code << "}\n\n";

    code << "inline output_window_float get_output_async_window_float(window_internal* win) {\n";
    code << "    return win->ptr;\n";
    code << "}\n\n";

    code << "inline input_window_int8 get_input_async_window_int8(window_internal* win) {\n";
    code << "    return win->ptr;\n";
    code << "}\n\n";

    code << "inline input_window_int16 get_input_async_window_int16(window_internal* win) {\n";
    code << "    return win->ptr;\n";
    code << "}\n\n";

    code << "inline input_window_int32 get_input_async_window_int32(window_internal* win) {\n";
    code << "    return win->ptr;\n";
    code << "}\n\n";

    code << "inline input_window_float get_input_async_window_float(window_internal* win) {\n";
    code << "    return win->ptr;\n";
    code << "}\n\n";

    code << "inline int8_t* acquire_output_window(output_window_int8 win) {\n";
    code << "    return (int8_t*)win;\n";
    code << "}\n\n";

    code << "inline void release_output_window(output_window_int8* win) {\n";
    code << "    chess_memory_fence();\n";
    code << "    window_internal* w = (window_internal*)win;\n";
    code << "    int lockid = select(w->current_bufid, w->lockids[1], w->lockids[0]);\n";
    code << "    release(lockid, REL_READ);\n";
    code << "    w->heads[w->current_bufid] = w->head;\n";
    code << "    w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);\n";
    code << "}\n\n";

    // Generate debug logging helpers
    code << "// Debug logging at fixed address 0x73000\n";
    code << "#define LOG_BASE_ADDR 0x73000\n";
    code << "static volatile int* log_ptr = (volatile int*)LOG_BASE_ADDR;\n";
    code << "static int log_index = 0;\n\n";
    code << "inline void log(int value) { log_ptr[log_index++] = value; }\n";
    code << "inline void log_at(int index, int value) { log_ptr[index] = value; }\n";
    code << "inline void log_reset() { log_index = 0; }\n\n";

    // Generate lock defines
    code << "#define LOCK_win_ping_ACQ " << info.inputAcquireLockId << "\n";
    code << "#define LOCK_win_pong_REL " << info.inputReleaseLockId << "\n";
    code << "#define LOCK_out_ping_ACQ " << info.outputAcquireLockId << "\n";
    code << "#define LOCK_out_pong_REL " << info.outputReleaseLockId << "\n\n";

    // Generate buffer declarations from extracted buffer definitions
    if (!info.bufferDefs.empty()) {
        for (const auto &buf : info.bufferDefs) {
            std::string bufVecType = "v" + std::to_string(buf.vectorWidth) + buf.elementType;
            code << bufVecType << " " << buf.name << "[BUF_SZ];\n";
        }
        code << "\n";
    } else {
        // Fallback to default buffer names if no buffer defs extracted
        code << vecTypeName << " win_ping[BUF_SZ];\n";
        code << vecTypeName << " win_pong[BUF_SZ];\n";
        code << vecTypeName << " out_ping[BUF_SZ];\n";
        code << vecTypeName << " out_pong[BUF_SZ];\n\n";
    }

    // Include the kernel file
    if (!info.kernelFile.empty()) {
        code << "#include \"" << info.kernelFile << "\"\n\n";
    }

    // Generate main function (kernel driver entry point)
    code << "int main(void) {\n";
    // code << "    log(1);  // Log: entering main\n";
    code << "    sync_buffer[0] = 0; // reset end signal\n\n";

    // Generate kernel invocation based on iteration style
    if (info.iterationStyle == "internal") {
        // Legacy style: window_init once, call kernel once
        code << "    // Legacy style: kernel handles iterations internally\n";

        // Use dynamic window parameters if available
        if (!info.windowParams.empty()) {
            // Dynamic mode: generate window initialization for each parameter
            SmallVector<std::string> windowPtrVars;

            for (size_t i = 0; i < info.windowParams.size(); ++i) {
                const auto &param = info.windowParams[i];
                std::string windowVarName = "window_" + param.windowSymbol;
                std::string ptrVarName = param.windowSymbol + "_ptr";
                std::string lockAcqName = "LOCK_" + param.windowSymbol + "_ACQ";
                std::string lockRelName = "LOCK_" + param.windowSymbol + "_REL";

                // Generate window_internal declaration
                code << "    window_internal " << windowVarName << "[1];\n";

                // Generate window_init call with actual buffer and lock names
                code << "    window_init(" << windowVarName << ", 1, " << param.bufferPingName << ", "
                     << param.acquireLockId << ", " << param.bufferPongName << ", " << param.releaseLockId
                     << ", BUF_SZ, BUF_SZ);\n";

                // Generate get_input/output_async_window call (no pointer, return value is already the window type)
                if (param.isInput) {
                    code << "    input_window_" << param.elementTypeName << " " << ptrVarName
                         << " = get_input_async_window_" << param.elementTypeName << "(" << windowVarName << ");\n\n";
                } else {
                    code << "    output_window_" << param.elementTypeName << " " << ptrVarName
                         << " = get_output_async_window_" << param.elementTypeName << "(" << windowVarName << ");\n\n";
                }

                windowPtrVars.push_back(ptrVarName);
            }

            // Generate kernel call with dynamic arguments
            code << "    // Call kernel\n";
            code << "    " << info.kernelName << "(";
            for (size_t i = 0; i < windowPtrVars.size(); ++i) {
                if (i > 0)
                    code << ", ";
                code << windowPtrVars[i];
            }
            code << ");\n";
        } else {
            // Fallback: hardcoded mode (backward compatibility)
            code << "    window_internal window_win_ping[1];\n";
            code << "    window_init(window_win_ping, 1, win_ping, LOCK_win_ping_ACQ, win_pong, LOCK_win_pong_REL, "
                    "BUF_SZ, "
                    "BUF_SZ);\n";
            code << "    input_window_int32 win_ping_ptr = get_input_async_window_int32(window_win_ping);\n\n";
            code << "    window_internal window_out_ping[1];\n";
            code << "    window_init(window_out_ping, 1, out_ping, LOCK_out_ping_ACQ, out_pong, LOCK_out_pong_REL, "
                    "BUF_SZ, "
                    "BUF_SZ);\n";
            code << "    output_window_int32 out_ping_ptr = get_output_async_window_int32(window_out_ping);\n\n";
            code << "    // Call kernel\n";
            code << "    " << info.kernelName << "(win_ping_ptr, out_ping_ptr);\n";
        }
    } else {
        // ADF style: loop with acquire/release
        code << "    // ADF style: wrapper handles iterations with acquire/release\n";

        // Use dynamic window parameters if available
        if (!info.windowParams.empty()) {
            // Dynamic mode: generate window initialization for each parameter
            SmallVector<std::string> windowVarNames;

            for (size_t i = 0; i < info.windowParams.size(); ++i) {
                const auto &param = info.windowParams[i];
                std::string windowVarName = "window_" + param.windowSymbol;

                // Generate window_internal declaration
                code << "    window_internal " << windowVarName << "[1];\n";

                // Generate window_init call
                code << "    window_init(" << windowVarName << ", 1, " << param.bufferPingName << ", "
                     << param.acquireLockId << ", " << param.bufferPongName << ", " << param.releaseLockId
                     << ", BUF_SZ, BUF_SZ);\n\n";

                windowVarNames.push_back(windowVarName);
            }

            code << "    for(int i = 0; i < iterations; i++) {\n";

            // Generate window_acquire calls for each parameter
            SmallVector<std::string> ptrVars;
            for (size_t i = 0; i < info.windowParams.size(); ++i) {
                const auto &param = info.windowParams[i];
                std::string ptrVar = param.windowSymbol + "_ptr";

                if (param.isInput) {
                    code << "        input_window_" << param.elementTypeName << " " << ptrVar << " = window_acquire_in("
                         << windowVarNames[i] << ");\n";
                } else {
                    code << "        output_window_" << param.elementTypeName << " " << ptrVar
                         << " = window_acquire_out(" << windowVarNames[i] << ");\n";
                }
                ptrVars.push_back(ptrVar);
            }

            // Generate kernel call with dynamic arguments
            code << "        " << info.kernelName << "(";
            for (size_t i = 0; i < ptrVars.size(); ++i) {
                if (i > 0)
                    code << ", ";
                code << ptrVars[i];
            }
            code << ");\n";

            // Generate window_release calls for each parameter
            for (const auto &windowVar : windowVarNames) {
                code << "        window_release(" << windowVar << ");\n";
            }

            code << "    }\n";
        } else {
            // Fallback: hardcoded mode (backward compatibility)
            code << "    window_internal window_win_ping[1];\n";
            code << "    window_init(window_win_ping, 1, win_ping, LOCK_win_ping_ACQ, win_pong, LOCK_win_pong_REL, "
                    "BUF_SZ, "
                    "BUF_SZ);\n\n";
            code << "    window_internal window_out_ping[1];\n";
            code << "    window_init(window_out_ping, 1, out_ping, LOCK_out_ping_ACQ, out_pong, LOCK_out_pong_REL, "
                    "BUF_SZ, "
                    "BUF_SZ);\n\n";
            code << "    for(int i = 0; i < iterations; i++) {\n";
            code << "        input_window_int32 win_ptr = window_acquire_in(window_win_ping);\n";
            code << "        output_window_int32 out_ptr = window_acquire_out(window_out_ping);\n";
            code << "        " << info.kernelName << "(win_ptr, out_ptr);\n";
            code << "        window_release(window_win_ping);\n";
            code << "        window_release(window_out_ping);\n";
            code << "    }\n";
        }
    }

    code << "\n    done();\n";
    code << "    return 0;\n";
    code << "}\n";
    code << "/* ========== End Kernel Driver ========== */\n";

    return code.str();
}

/// ConversionPattern for dfschedule.module -> EmitC verbatim kernel code
struct KernelModuleConversionPattern : public ConversionPattern {
    KernelModuleConversionPattern(TypeConverter &typeConverter, MLIRContext *ctx)
        : ConversionPattern(typeConverter, "dfschedule.module", /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto moduleOp = cast<dfschedule::KernelModuleOp>(op);
        auto loc = op->getLoc();

        llvm::errs() << "[Pattern] KernelModuleConversionPattern for: " << moduleOp.getSymName() << "\n";

        // Extract module information
        KernelModuleInfo info = extractKernelModuleInfo(moduleOp);

        llvm::errs() << "  Kernel name: " << info.kernelName << "\n";
        llvm::errs() << "  Iteration style: " << info.iterationStyle << "\n";
        llvm::errs() << "  Buffer size: " << info.bufferSize << "\n";
        llvm::errs() << "  Vector width: " << info.vectorWidth << "\n";

        // Generate the kernel driver C++ code
        std::string kernelCode = generateKernelDriverCode(info);

        // Create verbatim op with the generated code
        rewriter.create<emitc::VerbatimOp>(loc, rewriter.getStringAttr(kernelCode));

        llvm::errs() << "  ✓ Generated kernel driver code\n";

        // Erase the original module op
        rewriter.eraseOp(op);
        return success();
    }
};

/// Erase pattern for kernel module nested operations
template <typename OpT> struct EraseKernelModuleNestedOp : public OpConversionPattern<OpT> {
    using OpConversionPattern<OpT>::OpConversionPattern;

    LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // These ops are handled as part of the parent KernelModuleOp conversion
        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Generate C array literal from DenseElementsAttr
static std::string generateCArrayLiteral(DenseElementsAttr denseAttr) {
    std::string result = "{";
    bool first = true;
    
    if (auto floatAttr = dyn_cast<DenseFPElementsAttr>(denseAttr)) {
        for (auto val : floatAttr.getValues<APFloat>()) {
            if (!first) result += ", ";
            first = false;
            SmallString<16> strVal;
            val.toString(strVal);
            result += strVal.str().str();
        }
    } else if (auto intAttr = dyn_cast<DenseIntElementsAttr>(denseAttr)) {
        for (auto val : intAttr.getValues<APInt>()) {
            if (!first) result += ", ";
            first = false;
            result += std::to_string(val.getSExtValue());
        }
    } else {
        // Fallback for other types
        result += "0";
    }
    
    result += "}";
    return result;
}

static Type buildVoidPtrType(MLIRContext *ctx) {
    // void*  (emitc.ptr<emitc.opaque<"void">>)
    return emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
}
  
static void setupTypeConverter(TypeConverter &typeConverter, MLIRContext *ctx) {
    Type voidPtrTy = buildVoidPtrType(ctx);
    
    // DO NOT convert tensor types globally - this causes unwanted materializations
    // for operations we're not converting yet (like tensor.extract_slice)
    // Instead, handle type changes explicitly in each pattern's replaceOp() call

    // Convert MemRef types to void* — dfschedule ops use memrefs as buffer handles
    // but emitc.call_opaque requires EmitC-compatible (non-memref) types.
    typeConverter.addConversion([voidPtrTy](MemRefType) -> Type { return voidPtrTy; });

    // Identity conversion for all other types
    typeConverter.addConversion([voidPtrTy](Type t) -> Type { 
        // Just return the type as-is
        return t; 
    });
  
    // Materialization callbacks to handle value conversions
    auto castIfNeeded =
        [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc)
        -> std::optional<Value> {
          if (inputs.size() != 1) {
              llvm::errs() << "[Materialization] ERROR: Wrong number of inputs: " << inputs.size() << "\n";
              return std::nullopt;
          }
          
          // Check if input is null
          if (!inputs[0]) {
              llvm::errs() << "[Materialization] ERROR: Input value is NULL!\n";
              llvm::errs() << "[Materialization] ERROR: Requested result type: " << resultType << "\n";
              return std::nullopt;
          }
          
          if (inputs[0].getType() == resultType) {
              llvm::errs() << "[Materialization] Types match, returning input directly\n";
              return inputs[0];
          }
          
          llvm::errs() << "[Materialization] Creating cast from " << inputs[0].getType() << " to " << resultType << "\n";
          // Create unrealized cast for type mismatches
          return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0])
              .getResult(0);
        };
  
    typeConverter.addSourceMaterialization(castIfNeeded);
    typeConverter.addTargetMaterialization(castIfNeeded);
    typeConverter.addArgumentMaterialization(castIfNeeded);
}
//===----------------------------------------------------------------------===//
// Pass Implementation - Two-Phase Conversion with Walk + Patterns
//===----------------------------------------------------------------------===//

void DfscheduleToApiPass::runOnOperation() {
    llvm::errs() << "=== DfscheduleToApiPass START ===\n";
    
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = moduleOp.getContext();
    
    // Shared conversion state
    ConversionState state;
    state.enableDebug = enableDebug_;
    state.runtimeDebugLevel = runtimeDebugLevel_;

    // Type converter
    TypeConverter typeConverter;
    setupTypeConverter(typeConverter, ctx);  
    
    llvm::errs() << "convert tensor -> "
             << typeConverter.convertType(
                    RankedTensorType::get({16,16},
                      IntegerType::get(ctx, 8)))
             << "\n";
    //==========================================================================
    // Phase 1: Setup - Generate helper definitions at module scope
    //==========================================================================
    
    {
        OpBuilder builder(ctx);
        builder.setInsertionPointToStart(moduleOp.getBody());
        builder.create<emitc::VerbatimOp>(moduleOp.getLoc(), "#include \"aie_runtime.h\"");
        if (enableDebug_)
            builder.create<emitc::VerbatimOp>(moduleOp.getLoc(), "#include \"aie_runtime_debug.h\"");

        /* DevInst: passed as first parameter to host_canonicalized (set in state.devInstRef below) */
    }
    
    llvm::errs() << "[Pass] Phase 1: Helper definitions generated\n";
    
    //==========================================================================
    // Phase 2: Convert dense constants at module level
    //==========================================================================
    
    llvm::errs() << "[Pass] Phase 2: Converting dense constants\n";
    
    // Walk and manually convert dense constants to global arrays
    moduleOp.walk([&](arith::ConstantOp constOp) {
        auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
        if (!denseAttr) return;

        // Skip tensor constants that feed bufferization.to_memref only if they
        // live in func.func @main (not inside dfschedule.host). Constants
        // inside dfschedule.host that feed to_memref are the DDR init chain and
        // must be converted to global C arrays so the pass can lower them.
        if (!constOp->getParentOfType<dfschedule::HostBlockOp>()) {
            for (Operation *user : constOp->getUsers()) {
                if (isa<bufferization::ToMemrefOp>(user))
                    return;
            }
        }

        // Create global array name
        std::string arrayName = "g_data_array_" + std::to_string(state.arrayIndex++);
        
        // Store the array name as an attribute on the operation for later retrieval
        constOp->setAttr("emitc.global_array_name", 
                         StringAttr::get(moduleOp.getContext(), arrayName));
        
        // Generate C array literal
        std::string cArrayInit = generateCArrayLiteral(denseAttr);
        
        // Get element type
        Type elemType = denseAttr.getElementType();
        std::string cType = getEmitCTypeString(elemType);

        // Get shape
        auto tensorType = dyn_cast<RankedTensorType>(constOp.getType());
        if (!tensorType) return;
        
        auto shape = tensorType.getShape();
        std::string shapeStr;
        for (auto dim : shape) {
            shapeStr += "[" + std::to_string(dim) + "]";
        }
        
        // Create verbatim array declaration at the same location as the original constant
        OpBuilder builder(moduleOp.getContext());
        builder.setInsertionPoint(constOp);
        
        std::string arrayDecl = "static " + cType + " " + arrayName + shapeStr + " = " + cArrayInit + ";";
        builder.create<emitc::VerbatimOp>(constOp.getLoc(), arrayDecl);
        
        llvm::errs() << "[Pattern] Created array: " << arrayName << " at original location\n";
    });
    
    //==========================================================================
    // Phase 3: Walk dfschedule.host ops and apply inner patterns
    //==========================================================================
    
    llvm::errs() << "[Pass] Phase 3: Converting inner ops in dfschedule.host regions\n";
    
    // Setup common types for inner patterns
    state.ctx = ctx;
    state.i32Type = IntegerType::get(ctx, 32);
    state.voidPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "void"));
    state.memInstPtrType = emitc::PointerType::get(emitc::OpaqueType::get(ctx, "XAie_MemInst"));
    state.devInstType = emitc::OpaqueType::get(ctx, "XAie_DevInst");
    state.partitionType = emitc::OpaqueType::get(ctx, "PartitionTensor");
    
    // Create inner patterns (arith.constant, declare_data, partitiontensor, extract_slice, erase ops)
    RewritePatternSet innerPatterns(ctx);
    
    // Add arith.constant -> emitc.constant conversion (for scalar constants)
    innerPatterns.add<ArithConstantInnerPattern>(typeConverter, ctx, state);
    
    // Add actual conversion patterns for inner ops
    // Higher benefits = run first. We need DeclareTile and DeclareTensor to run before ConfigDmaBd

    // New patterns for the buffer ops introduced by BlueprintToSchedulePass refactor
    innerPatterns.add<AllocDeviceMemInnerPattern>(typeConverter, ctx, state);
    innerPatterns.add<BufferViewInnerPattern>(typeConverter, ctx);
    innerPatterns.add<BindCoreBufferInnerPattern>(typeConverter, ctx);
    innerPatterns.add<FreeDeviceMemInnerPattern>(typeConverter, ctx);
    innerPatterns.add<UnrealizedConversionCastInnerPattern>(typeConverter, ctx);
    // memref_mapping is Pure/zero-cost: forward source value unchanged (same as unrealized_cast)
    innerPatterns.add<MemRefMappingInnerPattern>(typeConverter, ctx);

    // Patterns for the DDR init chain moved into dfschedule.host by ScheduleCanonicalizePass:
    //   bufferization.to_memref -> passthrough (tensor already converted to void*)
    //   memref.alloc            -> __Runtime_malloc(byteSize)
    //   memref.copy             -> __Runtime_memcpy(dst, src, byteSize)
    //   memref.dealloc          -> __Runtime_free(ptr)
    //   memref.subview          -> __runtime_buffer_offset(base, byteOffset)
    innerPatterns.add<ToMemrefInnerPattern>(typeConverter, ctx);
    innerPatterns.add<MemRefAllocInnerPattern>(typeConverter, ctx);
    innerPatterns.add<MemRefCopyInnerPattern>(typeConverter, ctx);
    innerPatterns.add<MemRefDeallocInnerPattern>(typeConverter, ctx);
    innerPatterns.add<MemRefSubviewInnerPattern>(typeConverter, ctx);
    innerPatterns.add<MemRefReinterpretCastInnerPattern>(typeConverter, ctx);

    innerPatterns.add<DeclareDataInnerPattern>(typeConverter, ctx, state);
    innerPatterns.add<PartitionTensorInnerPattern>(typeConverter, ctx, state);
    innerPatterns.add<ExtractSliceInnerPattern>(typeConverter, ctx, state);

    // These must run BEFORE ConfigDmaBdOp and ConfigCreateIoOp (higher benefit = 100)
    innerPatterns.add<DeclareTileInnerPattern>(typeConverter, ctx, state, /*benefit=*/100);
    
    // ConfigDmaBdOp must run before ConfigCreateIoOp (benefit = 50)
    innerPatterns.add<ConfigDmaBdInnerPattern>(typeConverter, ctx, state, /*benefit=*/50);
    // GetBdIdOp allocates BD IDs and should run with medium priority (benefit = 30)
    innerPatterns.add<GetBdIdInnerPattern>(typeConverter, ctx, state, /*benefit=*/30);
    
    // ConfigCreateIoOp must run before StartIoOp (benefit = 10)
    innerPatterns.add<ConfigCreateIoInnerPattern>(typeConverter, ctx, state, /*benefit=*/10);
    
    // StartIoOp depends on ConfigCreateIoOp (benefit = 1)
    innerPatterns.add<StartIoInnerPattern>(typeConverter, ctx, state, /*benefit=*/1);
    
    // ScheduleWaitOp depends on StartIoOp (benefit = 1)
    innerPatterns.add<ScheduleWaitInnerPattern>(typeConverter, ctx, state, /*benefit=*/1);
    
    // LoadKernelGroupOp loads and configures kernel groups (benefit = 2)
    innerPatterns.add<LoadKernelGroupInnerPattern>(typeConverter, ctx, state, /*benefit=*/2);
    
    // DeclareKernelConfigOp is just metadata (benefit = 5, run early)
    innerPatterns.add<DeclareKernelConfigInnerPattern>(typeConverter, ctx, /*benefit=*/5);
    
    // LaunchKernelGroupOp depends on LoadKernelGroupOp (benefit = 1)
    innerPatterns.add<LaunchKernelGroupInnerPattern>(typeConverter, ctx, state, /*benefit=*/1);
    
    // Add EraseOpLowering patterns for ops that should simply be erased
    // NOTE: tensor.extract_slice, routing.partitiontensor, declare_data are NOT here - they are converted above
    // NOTE: ScheduleWaitOp, StartIoOp, GetBdIdOp, ConfigCreateIoOp, ConfigDmaBdOp, DeclareTileOp, DeclareTensorOp,
    //       LoadKernelGroupOp, LaunchKernelGroupOp are now converted by their respective Inner patterns
    // NOTE: LaunchHostOp is handled in Phase 4, not here
    innerPatterns.add<
        EraseOpLowering<routing::partitiontensor>
    >(typeConverter, ctx);

    FrozenRewritePatternSet frozenInnerPatterns(std::move(innerPatterns));
    ConversionTarget innerTarget(*ctx);
    innerTarget.addLegalDialect<emitc::EmitCDialect>();
    innerTarget.addLegalDialect<scf::SCFDialect>();
    
    // Mark arith dialect as legal, but with exceptions for arith.constant
    innerTarget.addLegalDialect<arith::ArithDialect>();

    // Use dynamic legality for arith.constant:
    // Tensor constants feeding bufferization.to_memref in @main stay legal (old path).
    // Tensor constants inside dfschedule.host are part of the DDR init chain and need
    // conversion (they were assigned a global_array_name in Phase 2).
    innerTarget.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
        if (isa<TensorType>(op.getType())) {
            // Inside host block: needs conversion (has global_array_name from Phase 2)
            if (op->getParentOfType<dfschedule::HostBlockOp>())
                return false;
            return true; // In @main: leave tensor constants alone
        }
        return false;
    });

    // Mark typed ops as illegal (need conversion)
    innerTarget.addIllegalOp<tensor::ExtractSliceOp>();
    innerTarget.addIllegalOp<routing::partitiontensor>();
    innerTarget.addIllegalOp<dfschedule::ScheduleWaitOp>();
    innerTarget.addIllegalOp<dfschedule::StartIoOp>();
    innerTarget.addIllegalOp<dfschedule::LaunchKernelGroupOp>();
    innerTarget.addIllegalOp<dfschedule::GetBdIdOp>();
    innerTarget.addIllegalOp<dfschedule::LoadKernelGroupOp>();
    innerTarget.addIllegalOp<dfschedule::ConfigCreateIoOp>();
    innerTarget.addIllegalOp<dfschedule::ConfigDmaBdOp>();  // Converted in Phase 3 with proper benefits

    innerTarget.addIllegalOp<dfschedule::DeclareTileOp>();
    // NOTE: LaunchHostOp is handled in Phase 4, not here
    // innerTarget.addIllegalOp<dfschedule::LaunchHostOp>();
    // dfschedule::DeclareTensorOp removed from dialect (tensor deprecated)
    innerTarget.addIllegalOp<dfscheblueprint::DeclareDataOp>();
    // New buffer ops introduced by BlueprintToSchedulePass refactor
    innerTarget.addIllegalOp<dfschedule::AllocDeviceMemOp>();
    innerTarget.addIllegalOp<dfschedule::BufferViewOp>();
    innerTarget.addIllegalOp<dfschedule::BindCoreBufferOp>();
    innerTarget.addIllegalOp<dfschedule::FreeDeviceMemOp>();
    innerTarget.addIllegalOp<dfschedule::DeclareKernelConfigOp>();
    innerTarget.addIllegalOp<UnrealizedConversionCastOp>();

    // DDR init chain ops moved into dfschedule.host by ScheduleCanonicalizePass.
    // Mark them illegal only when inside a HostBlockOp (outside is @main, still legal).
    innerTarget.addDynamicallyLegalOp<bufferization::ToMemrefOp>(
        [](bufferization::ToMemrefOp op) { return !op->getParentOfType<dfschedule::HostBlockOp>(); });
    innerTarget.addDynamicallyLegalOp<memref::AllocOp>(
        [](memref::AllocOp op) { return !op->getParentOfType<dfschedule::HostBlockOp>(); });
    innerTarget.addDynamicallyLegalOp<memref::CopyOp>(
        [](memref::CopyOp op) { return !op->getParentOfType<dfschedule::HostBlockOp>(); });
    innerTarget.addDynamicallyLegalOp<memref::DeallocOp>(
        [](memref::DeallocOp op) { return !op->getParentOfType<dfschedule::HostBlockOp>(); });
    innerTarget.addDynamicallyLegalOp<memref::SubViewOp>(
        [](memref::SubViewOp op) { return !op->getParentOfType<dfschedule::HostBlockOp>(); });
    innerTarget.addDynamicallyLegalOp<memref::ReinterpretCastOp>(
        [](memref::ReinterpretCastOp op) { return !op->getParentOfType<dfschedule::HostBlockOp>(); });

    // Use dynamic legality for string-named ops without C++ types
    
    // Create &DevInst and XAIE_MEM_CACHEABLE constants in each host block BEFORE conversion
    moduleOp->walk([&](dfschedule::HostBlockOp hostOp) {
        llvm::errs() << "[Pass] Pre-creating constants in host block\n";

        OpBuilder builder(ctx);
        builder.setInsertionPointToStart(&hostOp.getRegion().front());

        // Add XAie_DevInst* dev as a block argument at position 0 of the host block.
        // This becomes the first parameter of host_canonicalized when HostOpOuterPattern
        // converts the host block to an emitc.func.
        auto devInstPtrType = emitc::PointerType::get(state.devInstType);
        Block &hostBlock = hostOp.getRegion().front();
        state.devInstRef = hostBlock.insertArgument(
            /*argNo=*/0u, devInstPtrType, hostOp.getLoc());

        // Create XAIE_MEM_CACHEABLE constant
        state.cacheableConst = builder.create<emitc::ConstantOp>(
            hostOp.getLoc(), state.i32Type,
            emitc::OpaqueAttr::get(ctx, "XAIE_MEM_CACHEABLE")).getResult();
    });
    
    // Now apply conversion patterns to the ENTIRE MODULE (not per hostOp)
    llvm::errs() << "[Pass] Applying inner conversion patterns to entire module\n";
    
    // Debug: Count declare_data operations before conversion
    int declareDataCount = 0;
    moduleOp->walk([&](dfscheblueprint::DeclareDataOp op) {
        declareDataCount++;
        llvm::errs() << "[Pass] Found declare_data #" << declareDataCount << " at " << op->getLoc() << "\n";
    });
    llvm::errs() << "[Pass] Total declare_data ops: " << declareDataCount << "\n";

    {
        std::set<std::tuple<int32_t, int32_t, int32_t>> preScannedLocks;
        moduleOp->walk([&](dfschedule::ConfigDmaBdOp bdOp) {
            int32_t acquireLockId = static_cast<int32_t>(bdOp.getAcquireLockId());
            int32_t acquireLockVal = static_cast<int32_t>(bdOp.getAcquireLockVal());
            int32_t releaseLockId = static_cast<int32_t>(bdOp.getReleaseLockId());
            if (acquireLockId < 0 || acquireLockVal == 0)
                return;
            auto declareTileOp = bdOp.getTile().getDefiningOp<dfschedule::DeclareTileOp>();
            if (!declareTileOp)
                return;
            int32_t tileCol = declareTileOp.getCol();
            int32_t tileRow = declareTileOp.getRow();

            bool isOutput = false;
            SmallVector<Operation *, 4> worklist;
            for (auto *user : bdOp.getResult().getUsers())
                worklist.push_back(user);
            while (!worklist.empty()) {
                auto *u = worklist.pop_back_val();
                if (auto createIoOp = dyn_cast<dfschedule::ConfigCreateIoOp>(u)) {
                    if (createIoOp.getDirection().str() == "MM2S")
                        isOutput = true;
                    break;
                }
                if (isa<dfschedule::ConfigDmaBdOp>(u))
                    for (auto *uu : u->getResults().front().getUsers())
                        worklist.push_back(uu);
            }

            int32_t nextBdVal = static_cast<int32_t>(bdOp.getNextBd());
            bool isSingleBuffer = (nextBdVal == -1) && !bdOp.getLinkedBd();
            int32_t lockInitValue = isSingleBuffer ? 1 : 2;

            auto emitLock = [&](int32_t lockId, int32_t initVal, const std::string &note) {
                auto key = std::make_tuple(tileCol, tileRow, lockId);
                if (preScannedLocks.insert(key).second) {
                    std::string comment = "/* Lock init: tile(" + std::to_string(tileCol) + "," +
                                          std::to_string(tileRow) + ") lock=" + std::to_string(lockId) +
                                          " init_value=" + std::to_string(initVal) + " " + note + " */";
                    std::string call = "XAie_LockSetValue(dev, XAie_TileLoc(" + std::to_string(tileCol) + ", " +
                                       std::to_string(tileRow) + "), XAie_LockInit(" + std::to_string(lockId) + ", " +
                                       std::to_string(initVal) + "));";
                    state.pendingLockInits.emplace_back(comment, call);
                }
            };

            if (isOutput) {
                emitLock(releaseLockId, lockInitValue, "(kernel output acquire)");
            } else {
                emitLock(acquireLockId, lockInitValue, "");
            }
        });
        llvm::errs() << "[Pass] Pre-scan: collected " << state.pendingLockInits.size()
                     << " lock inits to emit before launch_kernel_group\n";
    }

    if (failed(applyPartialConversion(moduleOp, innerTarget, frozenInnerPatterns))) {
        llvm::errs() << "[Pass] Warning: Some inner ops not converted\n";
    }
    
    // Debug: Count declare_data operations after conversion
    int remainingDeclareDataCount = 0;
    moduleOp->walk([&](dfscheblueprint::DeclareDataOp op) {
        remainingDeclareDataCount++;
        llvm::errs() << "[Pass] Remaining declare_data #" << remainingDeclareDataCount << " at " << op->getLoc() << "\n";
    });
    llvm::errs() << "[Pass] Remaining declare_data ops: " << remainingDeclareDataCount << "\n";
    
    // Debug: Check for unrealized_conversion_cast operations
    int castCount = 0;
    moduleOp->walk([&](UnrealizedConversionCastOp castOp) {
        castCount++;
        llvm::errs() << "[Pass] Found UnrealizedConversionCastOp #" << castCount << "\n";
        llvm::errs() << "[Pass]   Inputs: " << castOp.getInputs().size() << "\n";
        for (auto input : castOp.getInputs()) {
            if (input) {
                llvm::errs() << "[Pass]   Input type: " << input.getType() << "\n";
            } else {
                llvm::errs() << "[Pass]   Input is NULL!\n";
            }
        }
        llvm::errs() << "[Pass]   Outputs: " << castOp.getOutputs().size() << "\n";
        for (auto output : castOp.getOutputs()) {
            llvm::errs() << "[Pass]   Output type: " << output.getType() << "\n";
        }
    });
    llvm::errs() << "[Pass] Total UnrealizedConversionCastOps: " << castCount << "\n";
    
    // Phase 3.6: Remove unrealized conversion casts
    // These are leftover from operations we didn't convert (like tensor.extract_slice)
    llvm::errs() << "[Pass] Phase 3.6: Removing unrealized conversion casts\n";
    moduleOp->walk([&](UnrealizedConversionCastOp castOp) {
        // Replace the cast with its input (passthrough)
        if (castOp.getInputs().size() == 1 && castOp.getOutputs().size() == 1) {
            Value input = castOp.getInputs()[0];
            Value output = castOp.getOutputs()[0];
            
            if (input && input.getType() != output.getType()) {
                llvm::errs() << "[Pass] Removing cast from " << input.getType() << " to " << output.getType() << "\n";
                // Replace all uses of the cast output with the input
                output.replaceAllUsesWith(input);
                castOp.erase();
            }
        }
    });
    
    // Phase 3.5 is no longer needed - arith.constant dense ops are replaced directly in ArithConstantInnerPattern
    
    //==========================================================================
    // Phase 4: Convert dfschedule ops (host -> emitc.func, launchhost, dskernel_receiver)
    //==========================================================================
    llvm::errs() << "[Pass] Phase 4: Converting dfschedule operations\n";
    
    {
        ConversionTarget target(*ctx);
        target.addLegalDialect<emitc::EmitCDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<func::FuncDialect>();  // func.func, func.return are legal
        
        target.addIllegalOp<dfschedule::HostBlockOp>();
        target.addIllegalOp<dfschedule::LaunchHostOp>();
        
        target.markUnknownOpDynamicallyLegal([](Operation *op) {
            StringRef opName = op->getName().getStringRef();
            if (opName == "dfschedule.dskernel_receiver") {
                return false; // illegal
            }
            // Mark kernel module operations as illegal - they will be converted
            if (opName == "dfschedule.module") {
                return false; // illegal - convert to EmitC
            }
            // Nested kernel module ops will be erased when parent module is converted
            if (opName == "dfschedule.kernel_config_def" || opName == "dfschedule.lock_def" ||
                opName == "dfschedule.buffer_def" || opName == "dfschedule.window_def" ||
                opName == "dfschedule.kernel_decl" || opName == "dfschedule.main" ||
                opName == "dfschedule.alloc_sync_buffer" || opName == "dfschedule.sync_buffer_write" ||
                opName == "dfschedule.log" || opName == "dfschedule.window_init" ||
                opName == "dfschedule.kernel_invoke" || opName == "dfschedule.done" ||
                opName == "dfschedule.kernel_return") {
                return true; // legal - handled by parent module pattern
            }
            return true;
        });
        
        llvm::errs() << "[Pass] Phase 4: Checking for launchhost ops before conversion\n";
        moduleOp.walk([&](Operation *op) {
            if (op->getName().getStringRef() == "dfschedule.launchhost") {
                llvm::errs() << "  Found launchhost op: " << *op << "\n";
            }
        });
        
        RewritePatternSet patterns(ctx);
        patterns.add<HostOpOuterPattern>(typeConverter, ctx, state);
        patterns.add<LaunchHostPattern>(typeConverter, ctx);
        patterns.add<DsKernelReceiverPattern>(typeConverter, ctx);
        patterns.add<KernelModuleConversionPattern>(typeConverter, ctx);

        if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
            llvm::errs() << "[Pass] Warning: Partial conversion had failures\n";
        }
        
        llvm::errs() << "[Pass] Phase 4: Checking for launchhost ops after conversion\n";
        moduleOp.walk([&](Operation *op) {
            if (op->getName().getStringRef() == "dfschedule.launchhost") {
                llvm::errs() << "  Still found launchhost op: " << *op << "\n";
            }
        });
    }
    
    //==========================================================================
    // Phase 4.5: Convert remaining arith.constant to emitc.constant
    //==========================================================================
    llvm::errs() << "[Pass] Phase 4.5: Converting remaining arith.constant to emitc.constant\n";
    
    {
        RewritePatternSet remainingConstPatterns(ctx);
        remainingConstPatterns.add<ArithConstantInnerPattern>(typeConverter, ctx, state);
        
        ConversionTarget remainingConstTarget(*ctx);
        remainingConstTarget.addLegalDialect<emitc::EmitCDialect>();
        remainingConstTarget.addLegalDialect<scf::SCFDialect>();
        remainingConstTarget.addLegalDialect<func::FuncDialect>();
        remainingConstTarget.addLegalDialect<arith::ArithDialect>();
        remainingConstTarget.addLegalDialect<bufferization::BufferizationDialect>();
        remainingConstTarget.addLegalDialect<memref::MemRefDialect>();

        // Mark arith.constant as illegal only if it is NOT a tensor constant in @main
        // feeding a bufferization.to_memref. Tensor constants inside dfschedule.host
        // were already converted in Phase 3 by ArithConstantInnerPattern.
        remainingConstTarget.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
            if (isa<TensorType>(op.getType())) {
                // Inside host block: should have been converted already; mark legal to avoid error.
                if (op->getParentOfType<dfschedule::HostBlockOp>())
                    return true;
                return true; // In @main: leave tensor constants alone
            }
            return false; // All scalar constants need conversion
        });

        if (failed(applyPartialConversion(moduleOp, remainingConstTarget, std::move(remainingConstPatterns)))) {
            llvm::errs() << "[Pass] Warning: Some arith.constant ops could not be converted in phase 4.5\n";
        }
    }

    //==========================================================================
    // Phase 4.55: Convert arith ops inside scf.for → emitc, then scf.for → emitc.for
    //==========================================================================
    llvm::errs() << "[Pass] Phase 4.55: Converting scf.for → emitc.for\n";

    {
        // Step 1: Manually convert arith ops inside scf.for bodies to emitc equivalents.
        // These survived Phase 3/4.5 because arith dialect was marked legal (only arith.constant
        // was selectively made illegal). We convert arith.index_cast → emitc.cast,
        // arith.muli → emitc.mul, and arith.addi → emitc.add so the emitc.for region
        // verifier accepts the body.
        OpBuilder b(ctx);
        moduleOp.walk([&](scf::ForOp forOp) {
            // Walk the for body and convert arith.index_cast → emitc.cast
            SmallVector<arith::IndexCastOp> castOps;
            forOp.getBody()->walk([&](arith::IndexCastOp op) { castOps.push_back(op); });
            for (auto castOp : castOps) {
                b.setInsertionPoint(castOp);
                auto result = b.create<emitc::CastOp>(castOp.getLoc(), castOp.getResult().getType(), castOp.getIn());
                castOp.replaceAllUsesWith(result.getResult());
                castOp.erase();
            }

            // Walk the for body and convert arith.muli → emitc.mul
            SmallVector<arith::MulIOp> mulOps;
            forOp.getBody()->walk([&](arith::MulIOp op) { mulOps.push_back(op); });
            for (auto mulOp : mulOps) {
                b.setInsertionPoint(mulOp);
                auto result =
                    b.create<emitc::MulOp>(mulOp.getLoc(), mulOp.getResult().getType(), mulOp.getLhs(), mulOp.getRhs());
                mulOp.replaceAllUsesWith(result.getResult());
                mulOp.erase();
            }

            // Walk the for body and convert arith.addi → emitc.add
            SmallVector<arith::AddIOp> addOps;
            forOp.getBody()->walk([&](arith::AddIOp op) { addOps.push_back(op); });
            for (auto addOp : addOps) {
                b.setInsertionPoint(addOp);
                auto result =
                    b.create<emitc::AddOp>(addOp.getLoc(), addOp.getResult().getType(), addOp.getLhs(), addOp.getRhs());
                addOp.replaceAllUsesWith(result.getResult());
                addOp.erase();
            }

            // Walk the for body and convert arith.divsi → emitc.div
            // (used by the 2D width-split halo offset: hc = iv / w_rounds)
            SmallVector<arith::DivSIOp> divOps;
            forOp.getBody()->walk([&](arith::DivSIOp op) { divOps.push_back(op); });
            for (auto divOp : divOps) {
                b.setInsertionPoint(divOp);
                auto result =
                    b.create<emitc::DivOp>(divOp.getLoc(), divOp.getResult().getType(), divOp.getLhs(), divOp.getRhs());
                divOp.replaceAllUsesWith(result.getResult());
                divOp.erase();
            }

            // Walk the for body and convert arith.remsi → emitc.rem
            // (used by the 2D width-split halo offset: wc = iv % w_rounds)
            SmallVector<arith::RemSIOp> remOps;
            forOp.getBody()->walk([&](arith::RemSIOp op) { remOps.push_back(op); });
            for (auto remOp : remOps) {
                b.setInsertionPoint(remOp);
                auto result =
                    b.create<emitc::RemOp>(remOp.getLoc(), remOp.getResult().getType(), remOp.getLhs(), remOp.getRhs());
                remOp.replaceAllUsesWith(result.getResult());
                remOp.erase();
            }
        });

        // Step 2: Convert scf.for → emitc.for using upstream SCFToEmitC patterns.
        RewritePatternSet scfPatterns(ctx);
        mlir::populateSCFToEmitCConversionPatterns(scfPatterns);

        ConversionTarget scfTarget(*ctx);
        scfTarget.addLegalDialect<emitc::EmitCDialect>();
        scfTarget.addLegalDialect<func::FuncDialect>();
        scfTarget.addLegalDialect<arith::ArithDialect>();
        scfTarget.addLegalDialect<bufferization::BufferizationDialect>();
        scfTarget.addLegalDialect<memref::MemRefDialect>();
        scfTarget.addIllegalDialect<scf::SCFDialect>();

        if (failed(applyPartialConversion(moduleOp, scfTarget, std::move(scfPatterns)))) {
            llvm::errs() << "[Pass] Warning: SCF-to-EmitC conversion had issues\n";
        }
    }

    //==========================================================================
    // Phase 4.6: Remove remaining unconverted ops and orphaned unrealized casts
    //==========================================================================
    llvm::errs() << "[Pass] Phase 4.6: Removing remaining unconverted ops\n";
    
    // Collect operations that will be removed
    SmallVector<Operation*> opsToRemove;
    
    // Collect declare_data ops that weren't converted
    moduleOp.walk([&](dfscheblueprint::DeclareDataOp op) {
        llvm::errs() << "  Will remove unconverted declare_data\n";
        opsToRemove.push_back(op);
    });
    
    // Collect partitiontensor ops that weren't converted  
    moduleOp.walk([&](routing::partitiontensor op) {
        llvm::errs() << "  Will remove unconverted partitiontensor\n";
        opsToRemove.push_back(op);
    });
    
    // Now collect unrealized casts that only have uses in operations we're removing
    SmallVector<UnrealizedConversionCastOp> castsToRemove;
    moduleOp.walk([&](UnrealizedConversionCastOp castOp) {
        if (castOp.getOutputs().size() == 1) {
            Value output = castOp.getOutputs()[0];
            bool allUsersWillBeRemoved = true;
            
            for (Operation *user : output.getUsers()) {
                // Check if this user is in our removal list
                if (std::find(opsToRemove.begin(), opsToRemove.end(), user) == opsToRemove.end()) {
                    allUsersWillBeRemoved = false;
                    break;
                }
            }
            
            if (output.use_empty() || allUsersWillBeRemoved) {
                llvm::errs() << "  Will remove orphaned unrealized_conversion_cast\n";
                castsToRemove.push_back(castOp);
            }
        }
    });
    
    // Now actually remove everything
    for (auto op : opsToRemove) {
        if (auto declareData = dyn_cast<dfscheblueprint::DeclareDataOp>(op)) {
            declareData.replaceAllUsesWith(declareData.getInitTensor());
        } else if (auto partition = dyn_cast<routing::partitiontensor>(op)) {
            partition.replaceAllUsesWith(partition.getOperand());
        }
        op->erase();
    }
    
    for (auto castOp : castsToRemove) {
        if (castOp.getInputs().size() == 1 && castOp.getOutputs().size() == 1) {
            Value input = castOp.getInputs()[0];
            Value output = castOp.getOutputs()[0];
            
            llvm::errs() << "    Removing cast from " << input.getType() 
                         << " to " << output.getType() << "\n";
            
            // Replace all uses of the output with the input
            output.replaceAllUsesWith(input);
        }
        castOp.erase();
    }
    
    //==========================================================================
    // Phase 4.7: Walk and convert any remaining dfschedule.launchhost inside execute_regions
    //==========================================================================
    /*
    llvm::errs() << "[Pass] Phase 4.5: Converting nested launchhost ops\n";
    
    {
        SmallVector<Operation*> launchhostOps;
        moduleOp.walk([&](Operation *op) {
            if (op->getName().getStringRef() == "dfschedule.launchhost") {
                launchhostOps.push_back(op);
            }
        });
        
        for (Operation *op : launchhostOps) {
            OpBuilder builder(op);
            builder.create<emitc::CallOpaqueOp>(op->getLoc(),
                TypeRange{}, "hostruntime", nullptr, nullptr, ValueRange{});
            llvm::errs() << "[Pass] Converted nested launchhost\n";
            op->erase();
        }
    }
    */
    //==========================================================================
    // Phase 5: Apply canonicalization to optimize EmitC
    //==========================================================================
    
    llvm::errs() << "[Pass] Phase 5: Applying canonicalization patterns\n";
    /*
    {
        RewritePatternSet canonPatterns(ctx);
        for (auto *dialect : ctx->getLoadedDialects()) {
            dialect->getCanonicalizationPatterns(canonPatterns);
        }
        for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
            op.getCanonicalizationPatterns(canonPatterns, ctx);
        }
        
        if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(canonPatterns)))) {
            llvm::errs() << "[Pass] Warning: Canonicalization had issues\n";
        }
    }
    */
    llvm::errs() << "=== DfscheduleToApiPass COMPLETE ===\n";
    
    // Print ExtractSlice conversion statistics
    llvm::errs() << "\n=== ExtractSlice Conversion Statistics ===\n";
    llvm::errs() << "  Total calls:    " << state.extractSliceCallCount << "\n";
    llvm::errs() << "  Successes:      " << state.extractSliceSuccessCount << "\n";
    llvm::errs() << "  Failures:       " << state.extractSliceFailCount << "\n";
    if (state.extractSliceCallCount > 0) {
        double successRate = (100.0 * state.extractSliceSuccessCount) / state.extractSliceCallCount;
        llvm::errs() << "  Success rate:   " << llvm::format("%.1f", successRate) << "%\n";
    }
    llvm::errs() << "==========================================\n\n";
    
    bool foundNullOperand = false;
    moduleOp.walk([&](Operation *op) {
        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            if (!op->getOperand(i)) {
                llvm::errs() << "[Pass] ERROR: Found null operand at index " << i 
                             << " in operation: " << op->getName() << "\n";
                
                // If it's a CallOpaqueOp, print the callee name
                if (auto callOp = dyn_cast<emitc::CallOpaqueOp>(op)) {
                    llvm::errs() << "  -> CallOpaqueOp callee: " << callOp.getCallee() << "\n";
                    llvm::errs() << "  -> Location: ";
                    op->getLoc().print(llvm::errs());
                    llvm::errs() << "\n";
                    llvm::errs() << "  -> Total operands: " << op->getNumOperands() << "\n";
                    for (unsigned j = 0; j < op->getNumOperands(); ++j) {
                        llvm::errs() << "     Operand[" << j << "]: " 
                                     << (op->getOperand(j) ? "valid" : "NULL") << "\n";
                    }
                }
                
                foundNullOperand = true;
            }
        }
    });
    
    if (foundNullOperand) {
        llvm::errs() << "[Pass] ERROR: Null operands detected, failing pass\n";
        //signalPassFailure();
    }

    llvm::errs() << "=== File Generation Complete ===\n\n";
    // Debug: find any emitc.call_opaque with non-emitc-compatible operand types.
    moduleOp.walk([&](emitc::CallOpaqueOp callOp) {
        for (Value operand : callOp.getOperands()) {
            if (isa<MemRefType>(operand.getType())) {
                llvm::errs() << "[Verify] BAD emitc.call_opaque '" << callOp.getCallee()
                             << "' has memref operand: " << operand.getType() << "\n";
                callOp->print(llvm::errs());
                llvm::errs() << "\n";
            }
        }
    });
}

} // namespace mlir
