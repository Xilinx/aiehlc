/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "dmaptodmaphop.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include <sstream>
#include <iostream>

using namespace mlir;
using namespace dmap;
using namespace dmaphop;

namespace {

// This pattern converts a dmap::FuncOp into a standard func::FuncOp and
// triggers the conversion of the operations inside.
struct DmapFuncOpLowering : public OpConversionPattern<dmap::FuncOp> {
    using OpConversionPattern<dmap::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(dmap::FuncOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        auto funcType = rewriter.getFunctionType({}, {});
        auto funcOp = rewriter.create<func::FuncOp>(op.getLoc(), op.getSymName(), funcType);

        rewriter.inlineRegionBefore(op.getBody(), funcOp.getBody(), funcOp.end());

        // Convert the dmap::YieldOp terminator to a func::ReturnOp
        for (auto &block : funcOp.getBlocks()) {
            if (auto yieldOp = dyn_cast_or_null<dmap::YieldOp>(block.getTerminator())) {
                rewriter.setInsertionPoint(yieldOp);
                rewriter.replaceOpWithNewOp<func::ReturnOp>(yieldOp);
            }
        }
        
        rewriter.eraseOp(op);
        return success();
    }
};


// Enum to distinguish between push and pull dataflow directions.
enum class DataflowDirection { Push, Pull };

// Trace a Value back through the SSA chain to find the originating function
// argument index.  This walks through bufferization.to_tensor,
// routing.routingcreatescheduletensor, routing.partitiontensor,
// routing.routingextract_data, and scf.execute_region captures until it
// reaches a BlockArgument of a func::FuncOp.
// Returns the argument index (0-based), or -1 if the chain cannot be resolved.
static int traceToFuncArgIndex(Value v) {
    // Walk up the def chain, max 20 hops to avoid infinite loops
    for (int depth = 0; depth < 20; ++depth) {
        // If v is a block argument of a func op, we found it
        if (auto blockArg = dyn_cast<BlockArgument>(v)) {
            if (auto funcOp = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp()))
                return static_cast<int>(blockArg.getArgNumber());
            // It's a block argument of some other region (e.g., scf.for,
            // scf.execute_region, routing.RoutingCreate).  These inner
            // regions capture values from their parent; try to find the
            // corresponding operand.
            // scf.execute_region has no operands – values from outside are
            // simply visible inside.  Walk the uses of `v` inside the region
            // to find a defining op to continue chasing.
            break; // can't follow further through a block argument
        }
        Operation *defOp = v.getDefiningOp();
        if (!defOp)
            break;

        // routing.routingextract_data %tensor, %idx -> follow %tensor (operand 0)
        if (defOp->getName().getStringRef() == "routing.routingextract_data" ||
            defOp->getName().getStringRef() == "routing.routingcreatescheduletensor" ||
            defOp->getName().getStringRef() == "routing.partitiontensor") {
            // operand 0 is the tensor/data input
            v = defOp->getOperand(0);
            continue;
        }
        // bufferization.to_tensor %memref -> follow %memref (operand 0)
        if (defOp->getName().getStringRef() == "bufferization.to_tensor") {
            v = defOp->getOperand(0);
            continue;
        }
        // dmap ops that forward data
        if (defOp->getName().getStringRef() == "dmap.push" || defOp->getName().getStringRef() == "dmap.pull") {
            v = defOp->getOperand(0);
            continue;
        }
        // Generic single-result ops that just forward operand 0
        if (defOp->getNumOperands() > 0) {
            v = defOp->getOperand(0);
            continue;
        }
        break;
    }
    return -1; // unable to resolve
}

// Generic function to lower a data movement operation (push or pull).
static LogicalResult lowerDataMovementOp(Operation *op, ConversionPatternRewriter &rewriter,
                                         RoutingTopology &router, DataflowDirection direction) {
    auto loc = op->getLoc();
    
    // Get operand 0 and extract the data tensor
    Value dataValue = op->getOperand(0);
    Value dataId;
    
    // Try to dyn_cast to routing::extract_data
    if (auto extractOp = dataValue.getDefiningOp<routing::extract_data>()) {
        // If it's already an extract_data operation, use its result directly
        dataId = extractOp.getResult();
    } else {
        // Otherwise, use the value directly (it should be a tensor)
        dataId = dataValue;
    }

    // Determine the originating function argument index for deterministic DMA
    // port assignment (see section 2b below for the full rationale).
    int funcArgIdx = traceToFuncArgIndex(dataValue);

    // Get topology info - core tile start row and partition column offset
    int core_start_row = (int) router.getRM()->getrsc()->absTileRow(TileType::Core, 0);
    int partColOffset = router.getRM()->hasPartition() ? router.getRM()->partitionStartCol() : 0;

    // dmap.push %data, %stream -> stream is operand 1
    // dmap.pull %data from %stream -> stream is operand 1
    Value streamValue = op->getOperand(1);
    Operation *streamOp = streamValue.getDefiningOp();
    

    dmap::define_io_engine shimEngine, memEngine;
    dmap::define_core_group sourceCoreGroup, destCoreGroup;
    
    // Helper lambda to extract io_engine or core_group from a config operation
    auto extractConfig = [](Value configValue, 
                           dmap::define_io_engine &ioEngine, 
                           dmap::define_core_group &coreGroup) -> bool {
        if (auto ioConfig = configValue.getDefiningOp<dmap::create_io_engin_with_config>()) {
            ioEngine = ioConfig.getIoengine().getDefiningOp<dmap::define_io_engine>();
            return true;  // is IO engine
        } else if (auto cgConfig = configValue.getDefiningOp<dmap::create_core_group_with_config>()) {
            coreGroup = cgConfig.getCoregroup().getDefiningOp<dmap::define_core_group>();
            return false;  // is core group
        }
        return false;
    };

    // --- 1. Extract tile/engine definitions from the stream ---
    if (auto chainStreamOp = dyn_cast<dmap::createchainstream>(streamOp)) {
        if (chainStreamOp.getStreams().size() != 2) return op->emitError("only supports 2-hop chained streams");
        auto stream1 = chainStreamOp.getStreams()[0].getDefiningOp<dmap::createstream>();
        auto stream2 = chainStreamOp.getStreams()[1].getDefiningOp<dmap::createstream>();

        // Extract configurations dynamically
        dmap::define_io_engine io1, io2, io3;
        dmap::define_core_group cg1, cg2;
        
        bool s1_src_is_io = extractConfig(stream1.getSource(), io1, cg1);
        bool s1_dst_is_io = extractConfig(stream1.getDestination(), io2, cg2);
        bool s2_src_is_io = extractConfig(stream2.getSource(), shimEngine, sourceCoreGroup);
        bool s2_dst_is_io = extractConfig(stream2.getDestination(), memEngine, destCoreGroup);
        
        // Determine which is shim, mem, and core group based on the configuration
        // For chain streams, typically: SHIM -> MEM -> CORES or CORES -> MEM -> SHIM
        if (s1_src_is_io && s1_dst_is_io) {
            // stream1: IO -> IO (shim -> mem or mem -> shim)
            shimEngine = io1;
            memEngine = io2;
        }
        // stream2 configurations already extracted into shimEngine/memEngine/sourceCoreGroup/destCoreGroup

    } else if (auto createStreamOp = dyn_cast<dmap::createstream>(streamOp)) {
        // Extract source and destination dynamically
        dmap::define_io_engine srcIO, dstIO;
        dmap::define_core_group srcCG, dstCG;
        
        bool src_is_io = extractConfig(createStreamOp.getSource(), srcIO, srcCG);
        bool dst_is_io = extractConfig(createStreamOp.getDestination(), dstIO, dstCG);
        
        if (src_is_io && dst_is_io) {
            // IO -> IO: first is shim, second could be mem
            shimEngine = srcIO;
            memEngine = dstIO;
        } else if (src_is_io && !dst_is_io) {
            // IO -> CoreGroup (Push direction)
            shimEngine = srcIO;
            destCoreGroup = dstCG;
        } else if (!src_is_io && dst_is_io) {
            // CoreGroup -> IO (Pull direction)
            sourceCoreGroup = srcCG;
            shimEngine = dstIO;
        } else {
            // CoreGroup -> CoreGroup
            sourceCoreGroup = srcCG;
            destCoreGroup = dstCG;
        }
    } else {
        return op->emitError("unsupported stream type for lowering");
    }
    
    // Determine the actual core group to use based on direction
    dmap::define_core_group coreGroup = destCoreGroup ? destCoreGroup : sourceCoreGroup;
    
    // Validate we have the required components
    if (!shimEngine) {
        return op->emitError("no IO engine (shim) found in stream configuration");
    }
    if (!coreGroup) {
        return op->emitError("no core group found in stream configuration");
    }

    // --- Allocate unique flow index for port naming ---
    static int ioIdx = 0;
    int curIoIdx = ioIdx++;
    std::ostringstream dioNameStream;
    dioNameStream << "dio" << curIoIdx;
    // Prefix for making port symbol names unique across multiple data flows
    std::string flowPrefix = "f" + std::to_string(curIoIdx) + "_";

    // --- 2. Build core tile list for createDataIO ---
    SmallVector<Point, 4> coreTilePoints;
    SmallVector<dmaphop::tile, 4> coreTiles;
    SmallVector<Value, 4> corePortsInValues;
    SmallVector<dmaphop::port, 4> corePortsOutOps;
    
    for (int i = 0; i < coreGroup.getCoreCount(); ++i) {
        // Use core_start_row as the base for core tile row calculation,
        // and partColOffset to convert mesh-relative col to absolute col
        int row = (coreGroup.getGroupAxis() == "col") ? (i + core_start_row) : (coreGroup.getGroupIdx() + core_start_row);
        int col = ((coreGroup.getGroupAxis() == "row") ? i : coreGroup.getGroupIdx()) + partColOffset;

        coreTilePoints.push_back(Point{row, col});
        
        auto coreTile = rewriter.create<dmaphop::tile>(loc, rewriter.getStringAttr("core"), rewriter.getI64IntegerAttr(col), rewriter.getI64IntegerAttr(row));
        coreTiles.push_back(coreTile);

        std::string inPortName = flowPrefix + "corePortIn" + std::to_string(i);
        // direction_channel must match the DMA port number so downstream passes
        // (DmaphopTodfscheblueprint) assign the correct S2MM channel.
        // Use funcArgIdx when available; fall back to 0 for unresolvable cases.
        int64_t inDirChannel = (funcArgIdx >= 0) ? funcArgIdx : 0;
        auto portInOp = rewriter.create<dmaphop::port>(loc, coreTile, rewriter.getStringAttr("In"),
                                                       rewriter.getStringAttr(inPortName),
                                                       rewriter.getI64IntegerAttr(inDirChannel), nullptr);
        corePortsInValues.push_back(portInOp.getResult());

        std::string outPortName = flowPrefix + "corePortOut" + std::to_string(i);
        // Only allocate pkt_id for Pull (output) flows where core OUT ports
        // are producers that need packet-switched routing to shim.
        // For Push (input) flows, core OUT ports are only used for inter-core
        // hop chaining and don't need pkt_ids (saves scarce 5-bit pkt_id pool).
        mlir::IntegerAttr pktIdAttr = nullptr;
        if (direction == DataflowDirection::Pull) {
            auto pktId = router.getRM()->allocatePktId(-1);
            assert(pktId.has_value() && "pkt_id pool exhausted (max 31)");
            pktIdAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(pktId.value()));
        }
        corePortsOutOps.push_back(rewriter.create<dmaphop::port>(loc, coreTile, rewriter.getStringAttr("Out"),
                                                                 rewriter.getStringAttr(outPortName),
                                                                 rewriter.getI64IntegerAttr(0), pktIdAttr));
    }

    // --- 2b. Create explicit DMA port mapping (consumer/producer ops) ---
    // This makes DMA port allocation visible at the dmaphop IR level.
    //
    // The DMA port number must match the kernel's window ordering: the kernel
    // acquires inputs via acquire_input_window(window_in_0), ...(window_in_1),
    // which map to DMA S2MM channel 0, 1, ... respectively.  The kernel
    // signature order mirrors the function argument order (%arg0, %arg1, ...),
    // so we use the originating function argument index as the DMA port number.
    //
    // Without this, the conversion-framework processing order (which may differ
    // from the argument order) would determine the port allocation, causing A/B
    // data swaps when the execute_regions are reordered.
    auto rm = router.getRM();
    if (direction == DataflowDirection::Push) {
        // Push: core tiles receive data -> create consumer ops for each core input port
        for (int i = 0; i < coreGroup.getCoreCount(); ++i) {
            Point corePt = coreTilePoints[i];
            int64_t dmaPortNum;
            if (funcArgIdx >= 0) {
                // Use the function argument index as the specific DMA port.
                // allocate(io, portidx, dir, ioId) reserves the exact port index.
                auto dmaPort = rm->tile(corePt.r, corePt.c).allocate(IOType::Input, funcArgIdx, PortDirection::DMA, -1);
                dmaPortNum = dmaPort.has_value() ? static_cast<int64_t>(*dmaPort) : funcArgIdx;
            } else {
                // Fallback to auto-allocation when arg index cannot be resolved
                auto dmaPort = rm->tile(corePt.r, corePt.c).occupyport(IOType::Input, PortDirection::DMA, -1);
                dmaPortNum = dmaPort.has_value() ? static_cast<int64_t>(*dmaPort) : 0;
            }
            std::string portName = flowPrefix + "corePortIn" + std::to_string(i);
            std::string consumerSymName = flowPrefix + "consumer" + std::to_string(i);
            rewriter.create<dmaphop::consumer>(loc, rewriter.getStringAttr(consumerSymName),
                                               FlatSymbolRefAttr::get(rewriter.getContext(), portName),
                                               rewriter.getI64IntegerAttr(dmaPortNum));
        }
    } else {
        // Pull: core tiles send data -> create producer ops for each core output port
        for (int i = 0; i < coreGroup.getCoreCount(); ++i) {
            Point corePt = coreTilePoints[i];
            auto dmaPort = rm->tile(corePt.r, corePt.c).occupyport(IOType::Output, PortDirection::DMA, -1);
            int64_t dmaPortNum = dmaPort.has_value() ? static_cast<int64_t>(*dmaPort) : 0;
            std::string portName = flowPrefix + "corePortOut" + std::to_string(i);
            std::string producerSymName = flowPrefix + "producer" + std::to_string(i);
            rewriter.create<dmaphop::producer>(loc, rewriter.getStringAttr(producerSymName),
                                               FlatSymbolRefAttr::get(rewriter.getContext(), portName),
                                               rewriter.getI64IntegerAttr(dmaPortNum));
        }
    }

    // --- 3. Use router to allocate shim column and channel ---
    // FIXED RULES:
    // Rule #1: Shim tile In and Out ports use the SAME channel number
    // Rule #2: Shim location (column) is determined by the destination core tile location
    //          - For Push (MM2S): shim -> cores, so destination is the FIRST core tile
    //          - For Pull (S2MM): cores -> shim, so shim destination relies on LAST core tile
    // Rule #3: Mem tile (if exists) is located in the SAME column as the shim tile

    // Get the target core tile for shim allocation
    // - For Push: shim is source, so use first core tile (destination)
    // - For Pull: shim is destination, so use last core tile (source)
    Point targetCoreTile = (direction == DataflowDirection::Push) 
                           ? coreTilePoints[0]                      // first tile for push
                           : coreTilePoints[coreTilePoints.size() - 1];  // last tile for pull
    std::optional<TypeBasedTileLoc> dstcoreloc(TypeBasedTileLoc{TileType::Core, targetCoreTile});
    
    // Determine DMA direction based on dataflow direction
    DMADIRECTION dmaDirection = (direction == DataflowDirection::Push) ? DMADIRECTION::MM2S : DMADIRECTION::S2MM;
    
    // Create DataIO - this allocates the optimal shim column and channel
    auto dio = router.createDataIO(dioNameStream.str(), dstcoreloc, dmaDirection);
    int shimcol = dio->colpos();
    int dioid = dio->id();
    int channel = dio->channel();
    
    std::cout << "Allocated DataIO: shim col=" << shimcol 
              << " channel=" << channel 
              << " IOID=" << dioid << std::endl;
    
    // --- 4. Create shim tile with allocated column ---
    // Note: Both shimPortOut and shimPortIn use the SAME channel (Rule #1)
    auto shimTile = rewriter.create<dmaphop::tile>(loc, rewriter.getStringAttr("shim"), 
                                                    rewriter.getI64IntegerAttr(shimcol), 
                                                    rewriter.getI64IntegerAttr(0));
    auto shimPortOut = rewriter.create<dmaphop::port>(loc, shimTile, rewriter.getStringAttr("Out"),
                                                      rewriter.getStringAttr(flowPrefix + "shimPortOut"),
                                                      rewriter.getI64IntegerAttr(channel), // Same channel
                                                      nullptr);                            // dmapktid
    auto shimPortIn = rewriter.create<dmaphop::port>(loc, shimTile, rewriter.getStringAttr("In"),
                                                     rewriter.getStringAttr(flowPrefix + "shimPortIn"),
                                                     rewriter.getI64IntegerAttr(channel), // Same channel
                                                     nullptr);                            // dmapktid

    // --- 5. Create mem tile if needed ---
    // Note: Mem tile is in the SAME column as shim tile (Rule #3)
    dmaphop::tile memTile;
    dmaphop::port memPortIn, memPortOut;
    if (memEngine) {
        // Mem tile uses the same column as shim (shimcol), not memEngine.getIoId()
        memTile = rewriter.create<dmaphop::tile>(loc, rewriter.getStringAttr("mem"), 
                                                  rewriter.getI64IntegerAttr(shimcol),  // Same col as shim (Rule #3)
                                                  rewriter.getI64IntegerAttr(0));
        memPortIn = rewriter.create<dmaphop::port>(loc, memTile, rewriter.getStringAttr("In"),
                                                   rewriter.getStringAttr(flowPrefix + "memPortIn"),
                                                   rewriter.getI64IntegerAttr(0),
                                                   nullptr); // dmapktid
        memPortOut = rewriter.create<dmaphop::port>(loc, memTile, rewriter.getStringAttr("Out"),
                                                    rewriter.getStringAttr(flowPrefix + "memPortOut"),
                                                    rewriter.getI64IntegerAttr(0),
                                                    nullptr); // dmapktid
    }
    

    // --- 6. Create dmaphop::create_hop Ops based on direction ---
    // Producer/Consumer semantics:
    // - Producers: ports that SOURCE data (send it out)
    //   * Core/Mem tiles: Out ports
    //   * Shim tile: In port (receives from external memory to send into fabric)
    // - Consumers: ports that SINK data (receive it)
    //   * Core/Mem tiles: In ports
    //   * Shim tile: Out port (sends to external memory from fabric)
    
    SmallVector<Value, 4> hops;
    SmallVector<Attribute, 4> producerPortSymbols;
    SmallVector<Attribute, 4> consumerPortSymbols;
    
    if (direction == DataflowDirection::Push) {
        // Push: SHIM (In) -> [MEM (Out)] -> CORES (In)
        // Producers: shimPortIn (for shim), memPortOut (if mem exists)
        // Consumers: all corePortIn

        if (memTile) { // SHIM -> MEM -> CORES
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, shimPortOut, memPortIn).getResult());
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, memPortOut, corePortsInValues[0]).getResult());
            // Producers: shimPortIn (shim receives from DDR), memPortOut (mem sends to cores)
            producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), flowPrefix + "shimPortIn"));
            producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), flowPrefix + "memPortOut"));
        } else { // SHIM -> CORES
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, shimPortOut, corePortsInValues[0]).getResult());
            // Producer: shimPortIn (shim receives from DDR to send into fabric)
            producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), flowPrefix + "shimPortIn"));
        }

        // All core input ports are consumers - use consumer symbol names
        for (size_t i = 0; i < corePortsInValues.size(); ++i) {
            std::string consumerSymName = flowPrefix + "consumer" + std::to_string(i);
            consumerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), consumerSymName));
        }

        // Create hops between cores (if multiple cores)
        for (size_t i = 0; i < corePortsOutOps.size() - 1; ++i) {
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, corePortsOutOps[i], corePortsInValues[i+1]).getResult());
        }

    } else { // Pull
        // Pull: CORES (Out) -> [MEM (In)] -> SHIM (Out)
        // Producers: all corePortOut, memPortIn (if mem exists)
        // Consumers: shimPortOut (for shim sends to DDR)

        if (memTile) { // CORES -> MEM -> SHIM
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, memPortOut, shimPortIn).getResult());
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, corePortsOutOps.back(), memPortIn).getResult());
            // Producers: all core output ports (use producer symbols), memPortIn (mem stays as port symbol)
            for (size_t i = 0; i < corePortsOutOps.size(); ++i) {
                std::string producerSymName = flowPrefix + "producer" + std::to_string(i);
                producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), producerSymName));
            }
            producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), flowPrefix + "memPortIn"));
            // Consumer: shimPortOut (shim sends to DDR)
            consumerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), flowPrefix + "shimPortOut"));
        } else { // CORES -> SHIM
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, corePortsOutOps.back(), shimPortIn).getResult());
            // Producers: all core output ports (use producer symbols)
            for (size_t i = 0; i < corePortsOutOps.size(); ++i) {
                std::string producerSymName = flowPrefix + "producer" + std::to_string(i);
                producerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), producerSymName));
            }
            // Consumer: shimPortOut (shim sends to DDR)
            consumerPortSymbols.push_back(SymbolRefAttr::get(rewriter.getContext(), flowPrefix + "shimPortOut"));
        }
        
        // Create hops between cores (if multiple cores)
        for (int i = corePortsOutOps.size() - 1; i >= 1; --i) {
            hops.push_back(rewriter.create<dmaphop::create_hop>(loc, corePortsOutOps[i - 1], corePortsInValues[i]).getResult());
        }
    }

    // --- 7. Create path, buffers, and final data movement op ---
    mlir::MLIRContext *ctx = rewriter.getContext();
    ArrayAttr produceArray = rewriter.getArrayAttr(producerPortSymbols);
    mlir::ArrayAttr consumeArray = mlir::ArrayAttr::get(ctx, consumerPortSymbols);

    // For Push: shim/mem produces, cores consume
    // For Pull: cores produce, shim/mem consumes
    // pkt_id is now stored per-port on each corePortOut{i} (dmapktid attribute)
    auto path = rewriter.create<dmaphop::create_path>(loc, hops, produceArray, consumeArray, rewriter.getArrayAttr({}));

    // Propagate the control-packet marker from the source stream onto the path so
    // passdmaphoptoroutinghw can select the tile CTRL sink port (register/config
    // write) instead of the default DMA S2MM sink. Absent marker -> normal flow.
    auto hasControlMarker = [](Operation *o) -> bool {
        return o && o->hasAttrOfType<mlir::BoolAttr>("control") &&
               o->getAttrOfType<mlir::BoolAttr>("control").getValue();
    };
    bool pathIsControl = hasControlMarker(streamOp);
    // For a memtile chain (createchainstream), the marker lives on the member
    // createstream ops, so inspect them too.
    if (!pathIsControl) {
        if (auto chainStreamOp = dyn_cast_or_null<dmap::createchainstream>(streamOp)) {
            for (Value s : chainStreamOp.getStreams()) {
                if (hasControlMarker(s.getDefiningOp())) {
                    pathIsControl = true;
                    break;
                }
            }
        }
    }
    if (pathIsControl) {
        path->setAttr("control", rewriter.getBoolAttr(true));
    }

    // Replaced dmaphop.alloc_buffer with tensor.extract_slice on the data tensor
    SmallVector<Value, 4> coreBuffers;
    
    // Get shape info from dataId
    auto rankedType = dyn_cast<RankedTensorType>(dataId.getType());
    if (!rankedType) {
        return op->emitError("Data operand must be a ranked tensor");
    }
    int64_t rank = rankedType.getRank();
    auto shape = rankedType.getShape();
    
    // Determine split dimension based on core group axis
    // If axis is "row", tiles are arranged horizontally (varying columns), so we split the column dimension (1)
    // If axis is "col", tiles are arranged vertically (varying rows), so we split the row dimension (0)
    int splitDim = (coreGroup.getGroupAxis() == "row") ? 0 : 1;//axis is col 
    
    // Check if we can split along the dimension that matches core count
    // or just split the first dimension
    int numTiles = coreTiles.size();
    
    //SmallVector<OpFoldResult> offsets(rank);
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;//(rank);
    
    // Calculate slice size based on split dim
    int64_t dimSize = shape[splitDim];
    if (dimSize != ShapedType::kDynamic && numTiles > 0) {
        // If static shape, verify divisibility or just div
        // Assuming divisibility for now as per usual tiling logic
    }
    //SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < rank; ++i) {
         if (rankedType.isDynamicDim(i)) {
             sizes.push_back(rewriter.create<tensor::DimOp>(loc, dataId, i).getResult());
         } else {
             sizes.push_back(rewriter.getIndexAttr(rankedType.getDimSize(i)));
        }
    }
    // Calculate slice size and offsets for each tile
    if (direction == DataflowDirection::Pull) {
        // ----- Atomic kernel-fire slab (group2 conv OUTPUT) -----
        // Default (matmul / non-tiled): each producer is the even row split
        // <shape[splitDim]/numTiles, ...full...>. For the group2 conv OUTPUT the
        // per-core view is already channel-split (Part 1), and the on-core H/W
        // iteration lives in the partition tiling attr. In that case the atomic
        // per-fire slab is <H_slice x W_slice x C_full>:
        //   H_slice = d[splitDim].outer.slice_tiling.slice  (e.g. 7)
        //   W_slice = d[wDim].outer.slice                   (e.g. 28)
        //   C_full  = shape[cDim]                           (e.g. 16, already split)
        // Each producer sits at local offset [0,0,0]; the DDR gather strides
        // (shim S2MM BD) realise the 4x4 H/W placement, not the subslice offset.
        int64_t atomHSlice = -1; // <0 => fall back to even split
        int64_t atomWSlice = -1;
        int wDimIdx = -1;
        if (rank >= 3) {
            // Trace dataId (extract_data) -> partitiontensor -> tiling attr.
            routing::TilingAttr tiling;
            if (auto extractOp = dataValue.getDefiningOp<routing::extract_data>()) {
                if (auto part = extractOp.getTensor().getDefiningOp<routing::partitiontensor>()) {
                    tiling = part.getTilingAttr();
                }
            }
            if (tiling) {
                auto dims = tiling.getDims();
                if (splitDim < (int64_t)dims.size()) {
                    // H atomic slice = on-core slice_tiling slice under the split dim.
                    if (auto inner = dims[splitDim].getOuter().getSliceTiling()) {
                        if (inner.getSlice() > 0)
                            atomHSlice = inner.getSlice(); // e.g. 7
                    }
                }
                // W = the first non-split, non-channel dim (channel is the last dim).
                for (int64_t d = 0; d < rank; ++d) {
                    if (d == splitDim || d == rank - 1)
                        continue;
                    if (d < (int64_t)dims.size()) {
                        int64_t ws = dims[d].getOuter().getSlice();
                        if (ws > 0 && ws < shape[d]) {
                            atomWSlice = ws; // e.g. 28
                            wDimIdx = (int)d;
                        }
                    }
                    break;
                }
            }
        }
        bool atomicSlab = (atomHSlice > 0);

        for (size_t i = 0; i < coreTiles.size(); ++i) {
            for (int64_t d = 0; d < rank; ++d) {
                if (atomicSlab) {
                    // Atomic slab: local offset 0 on every dim; size is the atomic
                    // extent (H slice / W chunk / full channel).
                    offsets[d] = rewriter.getIndexAttr(0);
                    if (d == splitDim)
                        sizes[d] = rewriter.getIndexAttr(atomHSlice);
                    else if (d == wDimIdx && atomWSlice > 0)
                        sizes[d] = rewriter.getIndexAttr(atomWSlice);
                    else
                        sizes[d] = rewriter.getIndexAttr(shape[d]);
                } else if (d == splitDim) {
                    // Sliced dimension - Static calculation
                    int64_t dimSize = shape[d];
                    // Assuming static shape for now as requested
                    int64_t sliceSize = dimSize / numTiles;
                    int64_t offset = i * sliceSize;

                    offsets[d] = rewriter.getIndexAttr(offset);
                    sizes[d] = rewriter.getIndexAttr(sliceSize);
                } else {
                    // Non-sliced dimension: full size, offset 0
                    offsets[d] = rewriter.getIndexAttr(0);
                    sizes[d] = rewriter.getIndexAttr(shape[d]);
                }
            }

            // Calculate result type for this slice
            SmallVector<int64_t> sliceShape;
            for (int64_t d = 0; d < rank; ++d) {
                if (atomicSlab) {
                    if (d == splitDim)
                        sliceShape.push_back(atomHSlice);
                    else if (d == wDimIdx && atomWSlice > 0)
                        sliceShape.push_back(atomWSlice);
                    else
                        sliceShape.push_back(shape[d]);
                } else if (d == splitDim) {
                    if (shape[d] == ShapedType::kDynamic)
                        sliceShape.push_back(ShapedType::kDynamic);
                    else
                        sliceShape.push_back(shape[d] / numTiles);
                } else {
                    sliceShape.push_back(shape[d]);
                }
            }
            auto sliceType = RankedTensorType::get(sliceShape, rankedType.getElementType());

            auto slice = rewriter.create<tensor::ExtractSliceOp>(loc, sliceType, dataId, offsets, sizes, strides);

            std::string tagName = (direction == DataflowDirection::Push) ? "consumer" : "producer";
            tagName += std::to_string(i);
            slice->setAttr("tag", rewriter.getStringAttr(tagName));

            coreBuffers.push_back(slice.getResult());
        }
    } else {
        // Push direction: use routing::extract_data for each tile
        // This will later be converted to tensor.extract_slice with "partitionslice" tag
        for (size_t i = 0; i < coreTiles.size(); ++i) {
            coreBuffers.push_back(dataId);
        }
    }

    if (direction == DataflowDirection::Push) {
        rewriter.create<dmaphop::push>(loc, dataId, path, coreBuffers, corePortsInValues);
    } else {
        rewriter.create<dmaphop::pull>(loc, dataId, path, coreBuffers, corePortsInValues);
    }
    rewriter.create<dmaphop::sync>(loc, path);

    // --- 8. Deallocate buffers and erase original op ---
    // dmaphop::dealloc_buffer is not needed for tensors
    /*
    for (auto buffer : coreBuffers) {
        rewriter.create<dmaphop::dealloc_buffer>(loc, buffer);
    }
    */
    // memref::DeallocOp is not used for tensors
    // rewriter.create<memref::DeallocOp>(loc, coreBufferTemplate);
    rewriter.eraseOp(op);
    return success();
}

// Lowering for dmap::push. This is now a thin wrapper.
struct PushOpLowering : public OpConversionPattern<dmap::push> {
    explicit PushOpLowering(MLIRContext *context, RoutingTopology &router)
        : OpConversionPattern<dmap::push>(context), router_(router) {}

    LogicalResult matchAndRewrite(dmap::push op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        return lowerDataMovementOp(op, rewriter, router_, DataflowDirection::Push);
    }
private:
    RoutingTopology &router_;
};

// Lowering for dmap::pull. This is now a thin wrapper.
struct PullOpLowering : public OpConversionPattern<dmap::pull> {
    explicit PullOpLowering(MLIRContext *context, RoutingTopology &router)
        : OpConversionPattern<dmap::pull>(context), router_(router) {}

    LogicalResult matchAndRewrite(dmap::pull op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        return lowerDataMovementOp(op, rewriter, router_, DataflowDirection::Pull);
    }
private:
    RoutingTopology &router_;
};

// Generic lowering pattern to erase an op that is no longer needed.
template <typename Op_T>
struct EraseOpLowering : public OpConversionPattern<Op_T> {
    using OpConversionPattern<Op_T>::OpConversionPattern;
    LogicalResult matchAndRewrite(Op_T op, typename Op_T::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

void DmapToDmaphopPass::runOnOperation() {
    auto& ctx = getContext();
    ConversionTarget target(ctx);
    RewritePatternSet patterns(&ctx);

    // The conversion is successful when the dmap dialect is gone.
    target.addIllegalDialect<dmap::dmapdialect>();
    // These dialects are legal to have in the output.
    target.addLegalDialect<dmaphop::dmaphopdialect, func::FuncDialect, memref::MemRefDialect, tensor::TensorDialect, arith::ArithDialect>();

    // Add the primary lowering patterns.
    //patterns.add<DmapFuncOpLowering, PushOpLowering>(&ctx);
    patterns.add<PushOpLowering>(&ctx, rtopology_);
    patterns.add<PullOpLowering>(&ctx, rtopology_);
    
    // Add patterns to erase the old dmap ops that are now handled by the main patterns.
    patterns.add<
        EraseOpLowering<dmap::create_data>,
        EraseOpLowering<dmap::define_io_engine>,
        EraseOpLowering<dmap::define_core_group>,
        EraseOpLowering<dmap::define_port_configure>,
        EraseOpLowering<dmap::create_io_engin_with_config>,
        EraseOpLowering<dmap::create_core_group_with_config>,
        EraseOpLowering<dmap::createstream>,
        EraseOpLowering<dmap::createchainstream>//,
        //EraseOpLowering<dmap::push>
    >(&ctx);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

DmapToDmaphopPass::DmapToDmaphopPass(RoutingTopology& rtopology):rtopology_(rtopology) {
}