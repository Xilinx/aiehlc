/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "passschedulecanonicalize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "dfschedulemanager.h"
#include <map>
#include <set>
#include <vector>
#include <iostream>

using namespace mlir;
using namespace dfschedule;

namespace {

// Key for tile identification: (col, row)
using TileKey = std::pair<int64_t, int64_t>;

// Structure to hold collected schedule information per tile
struct TileScheduleInfo {
    TileKey key;
    SmallVector<Value> tileValues;           // All tile values for this (col, row)
    SmallVector<Operation*> packetOps;       // Packet ops targeting this tile
    SmallVector<SymbolRefAttr> packetSymbols; // Packet symbols for this tile
    SmallVector<SymbolRefAttr> computeKernelArgs; // Compute args for this tile
    bool isShimTile = false;
    bool isCoreTile = false;
};

// Structure to hold shim tile DMA info
struct ShimDmaInfo {
    TileKey key;
    Value tileValue;
    SmallVector<Operation*> dmaBdOps;
    SmallVector<Operation*> createIoOps;
    SmallVector<Value> bdHandles;
    SmallVector<Value> ioHandles;
};

// Structure to hold tensor slice parameters
struct SliceParams {
    size_t index;  // Unique index for this slice
    RankedTensorType sourceType;
    SmallVector<int64_t, 4> offsets;
    SmallVector<int64_t, 4> sizes;
    SmallVector<int64_t, 4> strides;
    RankedTensorType resultType;
    size_t partitionIndex;  // Index of the partitiontensor this slice ultimately comes from
    int64_t parentSliceIndex;  // Index of parent slice if nested, -1 if directly from partition
    bool isFromPartition;  // true if immediate source is partitiontensor, false if from another slice
};

// Structure to hold partition tensor parameters
struct PartitionParams {
    size_t index;  // Unique index to distinguish partitions with same params but different flows
    RankedTensorType tensorType;
    int64_t splitnum;
    int64_t splitdim;
    std::string hw_axis_owner;
    std::string replicate_on;
    std::string single_tile_owner;
};

// Collected module-level schedule info
struct ModuleScheduleInfo {
    // Map from (col, row) to tile info
    std::map<TileKey, TileScheduleInfo> coreTiles;
    std::map<TileKey, ShimDmaInfo> shimTiles;
    
    // All collected operations
    SmallVector<Operation*> declareTensorOps;
    SmallVector<Operation*> declareTileOps;
    SmallVector<Operation*> configDmaBdOps;
    SmallVector<Operation*> configCreateIoOps;
    SmallVector<Operation*> packetOps;
    SmallVector<Operation*> loadKernelGroupOps;
    SmallVector<Operation*> launchKernelGroupOps;
    SmallVector<Operation*> getBdIdOps;
    SmallVector<Operation*> startIoOps;
    SmallVector<Operation*> scheduleWaitOps;
    SmallVector<Operation*> dskernelReceiverOps;
    
    // Operations to move from func.func main
    SmallVector<Operation*> tensorEmptyOps;
    SmallVector<Operation*> extractSliceOps;
    SmallVector<Operation*> partitionTensorOps;
    SmallVector<Operation*> executeRegionOps;
    SmallVector<Operation*> routingCreateOps;
    SmallVector<Operation*> declareDataOps;        // dfscheblueprint.declare_data
    SmallVector<Operation*> topLevelConstantOps;   // arith.constant at top level of main
    
    // Unique source tensor types (deduplicated)
    SmallVector<RankedTensorType> sourceTensorTypes;
    
    // Map from tensor type string key to its init tensor Value
    std::map<std::string, Value> initTensorMap;
    
    // Unique partition params (deduplicated)
    SmallVector<PartitionParams> uniquePartitionParams;
    
    // Unique extract_slice params (deduplicated)
    SmallVector<SliceParams> uniqueSliceParams;
    
    // Events to wait for
    SmallVector<Value> allEvents;
    
    // Kernel info
    StringRef kernelName = "dskernel_receiver";
    RankedTensorType kernelTensorType;
    int64_t bufferLen = 0;
    uint32_t basePacketId = 0;
    
    // Track how many packet streams each tile needs
    std::map<TileKey, int> tilePacketCount;
};

// Helper: Extract (col, row) from DeclareTileOp
static TileKey getTileKey(dfschedule::DeclareTileOp op) {
    return {op.getCol(), op.getRow()};
}

// Helper: Check if a tile is a shim tile (row == 0)
static bool isShimTile(TileKey key) {
    return key.second == 0;
}

// Collect all dfschedule operations from the module
static void collectScheduleOps(ModuleOp moduleOp, ModuleScheduleInfo &info) {
    moduleOp.walk([&](Operation *op) {
        if (auto declareTensor = dyn_cast<dfschedule::DeclareTensorOp>(op)) {
            info.declareTensorOps.push_back(op);
        } else if (auto declareTile = dyn_cast<dfschedule::DeclareTileOp>(op)) {
            info.declareTileOps.push_back(op);
            TileKey key = getTileKey(declareTile);
            
            if (isShimTile(key)) {
                if (info.shimTiles.find(key) == info.shimTiles.end()) {
                    info.shimTiles[key] = ShimDmaInfo{key, declareTile.getTile(), {}, {}, {}, {}};
                }
            } else {
                if (info.coreTiles.find(key) == info.coreTiles.end()) {
                    info.coreTiles[key] = TileScheduleInfo{key, {}, {}, {}, {}, false, true};
                }
                info.coreTiles[key].tileValues.push_back(declareTile.getTile());
            }
        } else if (auto configDmaBd = dyn_cast<dfschedule::ConfigDmaBdOp>(op)) {
            info.configDmaBdOps.push_back(op);
        } else if (auto createIo = dyn_cast<dfschedule::ConfigCreateIoOp>(op)) {
            info.configCreateIoOps.push_back(op);
        } else if (auto packet = dyn_cast<dfschedule::PacketOp>(op)) {
            info.packetOps.push_back(op);
        } else if (auto loadKernel = dyn_cast<dfschedule::LoadKernelGroupOp>(op)) {
            info.loadKernelGroupOps.push_back(op);
        } else if (auto launchKernel = dyn_cast<dfschedule::LaunchKernelGroupOp>(op)) {
            info.launchKernelGroupOps.push_back(op);
            info.allEvents.push_back(launchKernel.getEvent());
        } else if (auto getBdId = dyn_cast<dfschedule::GetBdIdOp>(op)) {
            info.getBdIdOps.push_back(op);
        } else if (auto startIo = dyn_cast<dfschedule::StartIoOp>(op)) {
            info.startIoOps.push_back(op);
            info.allEvents.push_back(startIo.getEvent());
        } else if (auto wait = dyn_cast<dfschedule::ScheduleWaitOp>(op)) {
            info.scheduleWaitOps.push_back(op);
        } else if (auto receiver = dyn_cast<dfschedule::DSKernelReceiverOp>(op)) {
            info.dskernelReceiverOps.push_back(op);
        } else if (auto emptyOp = dyn_cast<tensor::EmptyOp>(op)) {
            // Collect tensor.empty ops
            info.tensorEmptyOps.push_back(op);
            auto tensorType = cast<RankedTensorType>(emptyOp.getType());
            // Track unique tensor types
            bool found = false;
            for (auto &existingType : info.sourceTensorTypes) {
                if (existingType == tensorType) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                info.sourceTensorTypes.push_back(tensorType);
            }
        } else if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(op)) {
            // Collect tensor.extract_slice ops - NO deduplication!
            // Each slice is unique even if it has same offsets/sizes
            info.extractSliceOps.push_back(op);
            
            // Create slice params
            SliceParams params;
            params.index = info.uniqueSliceParams.size();  // Unique index
            params.sourceType = cast<RankedTensorType>(extractSlice.getSource().getType());
            for (auto o : extractSlice.getStaticOffsets()) params.offsets.push_back(o);
            for (auto s : extractSlice.getStaticSizes()) params.sizes.push_back(s);
            for (auto s : extractSlice.getStaticStrides()) params.strides.push_back(s);
            params.resultType = cast<RankedTensorType>(extractSlice.getType());
            params.partitionIndex = 0;
            params.parentSliceIndex = -1;  // Default: no parent slice
            params.isFromPartition = false;
            
            // Check immediate source
            Value source = extractSlice.getSource();
            if (Operation *defOp = source.getDefiningOp()) {
                if (defOp->getName().getStringRef() == "routing.partitiontensor") {
                    // Direct source is partitiontensor
                    params.isFromPartition = true;
                    for (size_t i = 0; i < info.partitionTensorOps.size(); ++i) {
                        if (info.partitionTensorOps[i] == defOp) {
                            params.partitionIndex = i;
                            break;
                        }
                    }
                } else if (auto parentSlice = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
                    // Direct source is another slice - find its index
                    params.isFromPartition = false;
                    for (size_t i = 0; i < info.extractSliceOps.size(); ++i) {
                        if (info.extractSliceOps[i] == defOp) {
                            params.parentSliceIndex = static_cast<int64_t>(i);
                            break;
                        }
                    }
                    
                    // Walk up to find the ultimate partitiontensor
                    Value walkSource = parentSlice.getSource();
                    while (Operation *walkDefOp = walkSource.getDefiningOp()) {
                        if (walkDefOp->getName().getStringRef() == "routing.partitiontensor") {
                            for (size_t i = 0; i < info.partitionTensorOps.size(); ++i) {
                                if (info.partitionTensorOps[i] == walkDefOp) {
                                    params.partitionIndex = i;
                                    break;
                                }
                            }
                            break;
                        } else if (auto grandParentSlice = dyn_cast<tensor::ExtractSliceOp>(walkDefOp)) {
                            walkSource = grandParentSlice.getSource();
                        } else {
                            break;
                        }
                    }
                }
            }
            
            // Always add - no deduplication for slices
            info.uniqueSliceParams.push_back(params);
        } else if (auto execRegion = dyn_cast<scf::ExecuteRegionOp>(op)) {
            // Collect scf.execute_region ops
            info.executeRegionOps.push_back(op);
        }
        
        // Check by operation name for routing dialect ops
        if (op->getName().getStringRef() == "routing.partitiontensor") {
            info.partitionTensorOps.push_back(op);
            
            // Extract partition parameters - NO deduplication here!
            // Each execute_region has its own partitiontensor representing different data flows
            // (e.g., producer flow vs consumer flow may have same partition params but different purposes)
            PartitionParams params;
            params.index = info.uniquePartitionParams.size();  // Unique index for each partition
            if (op->getNumResults() > 0) {
                params.tensorType = cast<RankedTensorType>(op->getResult(0).getType());
            }
            if (auto attr = op->getAttrOfType<IntegerAttr>("splitnum")) {
                params.splitnum = attr.getInt();
            }
            if (auto attr = op->getAttrOfType<IntegerAttr>("splitdim")) {
                params.splitdim = attr.getInt();
            }
            if (auto attr = op->getAttrOfType<StringAttr>("hw_axis_owner")) {
                params.hw_axis_owner = attr.getValue().str();
            }
            if (auto attr = op->getAttrOfType<StringAttr>("replicate_on")) {
                params.replicate_on = attr.getValue().str();
            }
            if (auto attr = op->getAttrOfType<StringAttr>("single_tile_owner")) {
                params.single_tile_owner = attr.getValue().str();
            }
            
            // Always add - each partitiontensor represents a unique data flow
            info.uniquePartitionParams.push_back(params);
        } else if (op->getName().getStringRef().starts_with("routing.RoutingCreate")) {
            info.routingCreateOps.push_back(op);
        }
        
        // Collect dfscheblueprint.declare_data
        if (op->getName().getStringRef() == "dfscheblueprint.declare_data") {
            info.declareDataOps.push_back(op);
            // Extract tensor type and init tensor for source tensor creation
            if (op->getNumResults() > 0) {
                if (auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
                    // Create a key for this tensor type
                    std::string typeKey;
                    llvm::raw_string_ostream os(typeKey);
                    os << tensorType;
                    
                    bool found = false;
                    for (auto &existingType : info.sourceTensorTypes) {
                        if (existingType == tensorType) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        info.sourceTensorTypes.push_back(tensorType);
                    }
                    
                    // Capture the init tensor from the operand (new DeclareDataOp takes init_tensor as input)
                    if (op->getNumOperands() > 0 && info.initTensorMap.find(typeKey) == info.initTensorMap.end()) {
                        info.initTensorMap[typeKey] = op->getOperand(0);
                    }
                }
            }
        }
        
        // Collect top-level arith.constant in main function
        if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            // Only collect if direct child of func.func (not nested in regions)
            if (auto funcOp = dyn_cast<func::FuncOp>(op->getParentOp())) {
                if (funcOp.getName() == "main") {
                    info.topLevelConstantOps.push_back(op);
                }
            }
        }
    });
}

// Associate packets with their target tiles
static void associatePacketsWithTiles(ModuleScheduleInfo &info) {
    // For each LoadKernelGroupOp, extract tile-to-packet mapping
    for (auto *op : info.loadKernelGroupOps) {
        auto loadKernel = cast<dfschedule::LoadKernelGroupOp>(op);
        
        auto tiles = loadKernel.getTiles();
        auto distArgs = loadKernel.getDistributedArgs();
        auto computeArgs = loadKernel.getDistributedComputeKernelArgs();
        
        for (size_t i = 0; i < tiles.size(); ++i) {
            // Find the DeclareTileOp that produced this tile value
            Value tileVal = tiles[i];
            if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {
                TileKey key = getTileKey(declareTile);
                
                if (!isShimTile(key)) {
                    auto &tileInfo = info.coreTiles[key];
                    
                    // Add packet symbol
                    if (i < distArgs.size()) {
                        if (auto symRef = dyn_cast<SymbolRefAttr>(distArgs[i])) {
                            tileInfo.packetSymbols.push_back(symRef);
                        }
                    }
                    
                    // Add compute kernel arg
                    if (i < computeArgs.size()) {
                        if (auto symRef = dyn_cast<SymbolRefAttr>(computeArgs[i])) {
                            tileInfo.computeKernelArgs.push_back(symRef);
                        }
                    }
                    
                    // Track packet count per tile
                    info.tilePacketCount[key]++;
                }
            }
        }
    }
}

// Associate DMA configs with shim tiles
static void associateDmaWithShimTiles(ModuleScheduleInfo &info) {
    for (auto *op : info.configDmaBdOps) {
        auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(op);
        Value tileVal = dmaBd.getTile();
        
        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {
            TileKey key = getTileKey(declareTile);
            if (isShimTile(key)) {
                info.shimTiles[key].dmaBdOps.push_back(op);
                info.shimTiles[key].bdHandles.push_back(dmaBd.getBdHandle());
            }
        }
    }
    
    for (auto *op : info.configCreateIoOps) {
        auto createIo = cast<dfschedule::ConfigCreateIoOp>(op);
        Value tileVal = createIo.getTile();
        
        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {
            TileKey key = getTileKey(declareTile);
            if (isShimTile(key)) {
                info.shimTiles[key].createIoOps.push_back(op);
                info.shimTiles[key].ioHandles.push_back(createIo.getIoHandle());
            }
        }
    }
}

// Structure to hold DMA BD parameters extracted from original ops
struct DmaBdParams {
    TileKey shimKey;
    Type bufferType;
    int64_t offset;
    int64_t len;
    bool enablePacket;  // Changed to bool for BoolAttr
    int64_t packetId;
    int64_t nextBd;
    int bdIndex;
    int64_t sliceIndex;  // Index into uniqueSliceParams/declaredMemrefs for the buffer
};

// Structure to hold IO config parameters
struct IoConfigParams {
    TileKey shimKey;
    int64_t channel;
    std::string direction;    // StringAttr
    std::string ioOperation;  // StringAttr
    int bdIndex; // Which BD handle to use
};

// Create canonicalized schedule in dfschedule.host at module level
// All operations use only constants/attributes, so IsolatedFromAbove is OK
static void createCanonicalizedSchedule(
    OpBuilder &builder,
    Location loc,
    ModuleScheduleInfo &info,
    ModuleOp moduleOp,
    func::FuncOp funcOp) {
    
    // Early exit if nothing to canonicalize
    if (info.coreTiles.empty() && info.shimTiles.empty()) {
        return;
    }
    
    // ==========================================================
    // Collect DMA BD and IO parameters from original operations
    // ==========================================================
    std::vector<DmaBdParams> allDmaBdParams;
    std::vector<IoConfigParams> allIoConfigParams;
    
    // Map to track BD index per shim tile
    std::map<TileKey, int> shimBdCounter;
    
    for (auto *op : info.configDmaBdOps) {
        auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(op);
        Value tileVal = dmaBd.getTile();
        
        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {
            TileKey key = getTileKey(declareTile);
            if (isShimTile(key)) {
                DmaBdParams params;
                params.shimKey = key;
                params.bufferType = dmaBd.getBuffer().getType();
                params.offset = dmaBd.getOffset();
                params.len = dmaBd.getLen();
                params.enablePacket = dmaBd.getEnablePacket();
                params.packetId = dmaBd.getPacketId();
                params.nextBd = dmaBd.getNextBd();
                params.bdIndex = shimBdCounter[key]++;
                params.sliceIndex = -1;  // Default: no slice found
                
                // Trace buffer back to find the slice index
                // Buffer -> DeclareTensorOp -> ExtractSliceOp
                Value buffer = dmaBd.getBuffer();
                if (auto declareTensor = buffer.getDefiningOp<dfschedule::DeclareTensorOp>()) {
                    Value tensorVal = declareTensor.getTensor();
                    if (auto extractSlice = tensorVal.getDefiningOp<tensor::ExtractSliceOp>()) {
                        // Find which slice index this corresponds to
                        for (size_t i = 0; i < info.extractSliceOps.size(); ++i) {
                            if (info.extractSliceOps[i] == extractSlice.getOperation()) {
                                params.sliceIndex = static_cast<int64_t>(i);
                                break;
                            }
                        }
                    }
                }
                
                allDmaBdParams.push_back(params);
            }
        }
    }
    
    // Map to track IO index per shim tile
    std::map<TileKey, int> shimIoCounter;
    
    for (auto *op : info.configCreateIoOps) {
        auto createIo = cast<dfschedule::ConfigCreateIoOp>(op);
        Value tileVal = createIo.getTile();
        
        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {
            TileKey key = getTileKey(declareTile);
            if (isShimTile(key)) {
                IoConfigParams params;
                params.shimKey = key;
                params.channel = createIo.getChannel();
                params.direction = createIo.getDirection().str();
                params.ioOperation = createIo.getIoOperation().str();
                params.bdIndex = shimIoCounter[key]++;
                allIoConfigParams.push_back(params);
            }
        }
    }
    
    // ==========================================================
    // PART 1: Create dfschedule.host at MODULE level (after func.func)
    // ==========================================================
    builder.setInsertionPointAfter(funcOp);
    
    auto hostOp = builder.create<dfschedule::HostBlockOp>(
        loc,
        builder.getStringAttr("host_canonicalized"));
    
    Block *hostBody = &hostOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(hostBody);
    
    // All operations below are created INSIDE dfschedule.host
    // They only use constants and values defined within this block
    
    // Helper to create type key strings
    auto makeTensorTypeKey = [](RankedTensorType t) -> std::string {
        std::string key;
        llvm::raw_string_ostream os(key);
        os << t;
        return key;
    };
    
    // Helper to create partition key - include index for uniqueness
    auto makePartitionKey = [&](const PartitionParams &p) -> std::string {
        std::string key;
        llvm::raw_string_ostream os(key);
        os << p.index << "_" << p.tensorType << "_" << p.splitnum << "_" << p.splitdim << "_" << p.hw_axis_owner;
        return key;
    };
    
    // ==========================================================
    // DATA FLOW: declare_data -> partitiontensor -> extract_slice
    // ==========================================================
    
    // 0a. Create dfscheblueprint.declare_data for each unique source tensor type
    // Reuse the init tensor from the original IR instead of creating new ones
    std::map<std::string, Value> sourceTensorMap;
    for (auto tensorType : info.sourceTensorTypes) {
        std::string key = makeTensorTypeKey(tensorType);
        if (sourceTensorMap.find(key) == sourceTensorMap.end()) {
            // Look up the captured init tensor from the original IR
            auto initTensorIt = info.initTensorMap.find(key);
            if (initTensorIt == info.initTensorMap.end()) {
                llvm::errs() << "Warning: No init tensor found for type " << key << "\n";
                continue;
            }
            
            Value originalInitTensor = initTensorIt->second;
            
            // Clone the init tensor (arith.constant) into the new block
            Operation *initDefOp = originalInitTensor.getDefiningOp();
            Value initTensor;
            if (initDefOp) {
                Operation *clonedOp = builder.clone(*initDefOp);
                initTensor = clonedOp->getResult(0);
            } else {
                // If it's a block argument or other, we can't clone - skip
                llvm::errs() << "Warning: Init tensor is not defined by an operation\n";
                continue;
            }
            
            // Create declare_data operation using generic op builder with init_tensor
            OperationState state(loc, "dfscheblueprint.declare_data");
            state.addOperands({initTensor});
            state.addTypes({tensorType});
            Operation *declareDataOp = builder.create(state);
            sourceTensorMap[key] = declareDataOp->getResult(0);
        }
    }
    
    // 0b. Create routing.partitiontensor for each unique partition config
    std::map<std::string, Value> partitionedTensorMap;
    for (const auto &params : info.uniquePartitionParams) {
        std::string partKey = makePartitionKey(params);
        if (partitionedTensorMap.find(partKey) == partitionedTensorMap.end()) {
            // Find the source tensor
            std::string srcKey = makeTensorTypeKey(params.tensorType);
            Value sourceTensor;
            if (sourceTensorMap.find(srcKey) != sourceTensorMap.end()) {
                sourceTensor = sourceTensorMap[srcKey];
            } else {
                // Create declare_data if not found - reuse init tensor from original IR
                auto initTensorIt = info.initTensorMap.find(srcKey);
                if (initTensorIt != info.initTensorMap.end()) {
                    Value originalInitTensor = initTensorIt->second;
                    
                    // Clone the init tensor (arith.constant) into the new block
                    Operation *initDefOp = originalInitTensor.getDefiningOp();
                    Value initTensor;
                    if (initDefOp) {
                        Operation *clonedOp = builder.clone(*initDefOp);
                        initTensor = clonedOp->getResult(0);
                    } else {
                        continue;
                    }
                    
                    OperationState state(loc, "dfscheblueprint.declare_data");
                    state.addOperands({initTensor});
                    state.addTypes({params.tensorType});
                    Operation *declareDataOp = builder.create(state);
                    sourceTensorMap[srcKey] = declareDataOp->getResult(0);
                    sourceTensor = declareDataOp->getResult(0);
                }
            }
            
            // Create routing.partitiontensor
            OperationState partState(loc, "routing.partitiontensor");
            partState.addOperands({sourceTensor});
            partState.addTypes({params.tensorType});
            partState.addAttribute("splitnum", builder.getI32IntegerAttr(params.splitnum));
            partState.addAttribute("splitdim", builder.getI32IntegerAttr(params.splitdim));
            partState.addAttribute("hw_axis_owner", builder.getStringAttr(params.hw_axis_owner));
            partState.addAttribute("replicate_on", builder.getStringAttr(params.replicate_on));
            partState.addAttribute("single_tile_owner", builder.getStringAttr(params.single_tile_owner));
            Operation *partitionOp = builder.create(partState);
            partitionedTensorMap[partKey] = partitionOp->getResult(0);
        }
    }
    
    // 0c. Create extract_slice operations - maintain proper chain
    // Each slice gets a unique entry, keyed by its index
    // We process in order: first slices from partitions, then nested slices
    
    std::map<size_t, Value> sliceMap;  // Map from slice index to created Value
    
    // First pass: create slices that come directly from partitiontensors
    for (const auto &params : info.uniqueSliceParams) {
        if (params.isFromPartition) {
            Value sourceTensor;
            
            // Find the partitioned tensor by index
            for (auto &[partKey, partValue] : partitionedTensorMap) {
                std::string indexPrefix = std::to_string(params.partitionIndex) + "_";
                if (partKey.find(indexPrefix) == 0) {
                    sourceTensor = partValue;
                    break;
                }
            }
            
            if (sourceTensor) {
                SmallVector<int64_t, 4> defaultStrides(params.offsets.size(), 1);
                auto newSlice = builder.create<tensor::ExtractSliceOp>(
                    loc,
                    params.resultType,
                    sourceTensor,
                    ValueRange{}, ValueRange{}, ValueRange{},
                    params.offsets,
                    params.sizes,
                    params.strides.empty() ? defaultStrides : params.strides);
                sliceMap[params.index] = newSlice.getResult();
            }
        }
    }
    
    // Second pass: create nested slices (from other slices)
    // May need multiple passes if there are deeply nested slices
    bool progress = true;
    while (progress) {
        progress = false;
        for (const auto &params : info.uniqueSliceParams) {
            // Skip if already created or if it's directly from partition
            if (sliceMap.find(params.index) != sliceMap.end() || params.isFromPartition) {
                continue;
            }
            
            // Check if parent slice is ready
            if (params.parentSliceIndex >= 0) {
                auto parentIt = sliceMap.find(static_cast<size_t>(params.parentSliceIndex));
                if (parentIt != sliceMap.end()) {
                    // Parent is ready, create this slice
                    Value sourceTensor = parentIt->second;
                    SmallVector<int64_t, 4> defaultStrides(params.offsets.size(), 1);
                    auto newSlice = builder.create<tensor::ExtractSliceOp>(
                        loc,
                        params.resultType,
                        sourceTensor,
                        ValueRange{}, ValueRange{}, ValueRange{},
                        params.offsets,
                        params.sizes,
                        params.strides.empty() ? defaultStrides : params.strides);
                    sliceMap[params.index] = newSlice.getResult();
                    progress = true;
                }
            }
        }
    }
    
    
    // Identify leaf slices (slices with no children)
    std::set<size_t> slicesWithChildren;
    for (const auto &params : info.uniqueSliceParams) {
        if (params.parentSliceIndex >= 0) {
            slicesWithChildren.insert(static_cast<size_t>(params.parentSliceIndex));
        }
    }
    
    // Build mapping from parent slice to its child slices
    std::map<size_t, SmallVector<size_t>> sliceChildren;
    for (const auto &params : info.uniqueSliceParams) {
        if (params.parentSliceIndex >= 0) {
            sliceChildren[static_cast<size_t>(params.parentSliceIndex)].push_back(params.index);
        }
    }
    
    // 0d. Create dfschedule.declaretensor ONLY for leaf slices (no children)
    std::map<size_t, Value> declaredMemrefs;  // Map from slice index to memref
    for (const auto &params : info.uniqueSliceParams) {
        // Skip if this slice has children - only create declaretensor for leaf slices
        if (slicesWithChildren.count(params.index) > 0) {
            continue;
        }
        
        auto it = sliceMap.find(params.index);
        if (it != sliceMap.end()) {
            Value sliceValue = it->second;
            auto sliceType = cast<RankedTensorType>(sliceValue.getType());
            auto memrefType = MemRefType::get(sliceType.getShape(), sliceType.getElementType());
            
            auto declareTensor = builder.create<dfschedule::DeclareTensorOp>(
                loc, memrefType, sliceValue);
            declaredMemrefs[params.index] = declareTensor.getMemref();
        }
    }
    
    // Build mapping from intermediate slice to its leaf descendants
    // This is used to map DMA BD (which references intermediate slice) to leaf slices
    std::map<size_t, SmallVector<size_t>> sliceToLeafDescendants;
    std::function<SmallVector<size_t>(size_t)> getLeafDescendants = [&](size_t idx) -> SmallVector<size_t> {
        if (slicesWithChildren.count(idx) == 0) {
            // This is a leaf slice
            return {idx};
        }
        SmallVector<size_t> leaves;
        for (size_t childIdx : sliceChildren[idx]) {
            auto childLeaves = getLeafDescendants(childIdx);
            leaves.append(childLeaves.begin(), childLeaves.end());
        }
        return leaves;
    };
    for (const auto &params : info.uniqueSliceParams) {
        sliceToLeafDescendants[params.index] = getLeafDescendants(params.index);
    }
    
    // 1. Create NEW deduplicated shim tile declarations
    std::map<TileKey, Value> shimTileMap;
    for (auto &[key, shimInfo] : info.shimTiles) {
        auto shimTile = builder.create<dfschedule::DeclareTileOp>(
            loc,
            dfschedule::TileType::get(builder.getContext()),
            builder.getI32IntegerAttr(key.first),
            builder.getI32IntegerAttr(key.second));
        shimTileMap[key] = shimTile.getTile();
    }
    
    // 2. Create NEW deduplicated core tile declarations
    std::map<TileKey, Value> coreTileMap;
    for (auto &[key, tileInfo] : info.coreTiles) {
        auto coreTile = builder.create<dfschedule::DeclareTileOp>(
            loc,
            dfschedule::TileType::get(builder.getContext()),
            builder.getI32IntegerAttr(key.first),
            builder.getI32IntegerAttr(key.second));
        coreTileMap[key] = coreTile.getTile();
    }
    
    // 3. Create external memory references for DMA buffers (using memref.alloc)
    // Group buffers by type to deduplicate
    std::map<std::string, Value> bufferMap;
    auto makeBufferKey = [](Type t) -> std::string {
        std::string key;
        llvm::raw_string_ostream os(key);
        os << t;
        return key;
    };
    
    // Map to store BD handles per shim tile
    std::map<TileKey, SmallVector<Value>> shimBdHandles;
    
    // 4. Create DMA BD configurations for shim tiles
    // If the traced slice is an intermediate slice, create DMA BDs for all its leaf descendants
    int bdIndexCounter = 0;
    for (const auto &params : allDmaBdParams) {
        if (shimTileMap.find(params.shimKey) == shimTileMap.end()) continue;
        Value shimTile = shimTileMap[params.shimKey];
        
        // Get the leaf descendants for this slice
        SmallVector<size_t> leafIndices;
        if (params.sliceIndex >= 0) {
            auto it = sliceToLeafDescendants.find(static_cast<size_t>(params.sliceIndex));
            if (it != sliceToLeafDescendants.end()) {
                leafIndices = it->second;
            }
        }
        
        // If no leaf indices found, try direct lookup
        if (leafIndices.empty() && params.sliceIndex >= 0) {
            leafIndices.push_back(static_cast<size_t>(params.sliceIndex));
        }
        
        // Create a DMA BD for each leaf slice
        for (size_t leafIdx : leafIndices) {
            Value buffer;
            auto memIt = declaredMemrefs.find(leafIdx);
            if (memIt != declaredMemrefs.end()) {
                buffer = memIt->second;
            }
            
            // Fallback: create or reuse buffer if not found
            if (!buffer) {
                std::string bufKey = makeBufferKey(params.bufferType);
                if (bufferMap.find(bufKey) == bufferMap.end()) {
                    auto memrefType = cast<MemRefType>(params.bufferType);
                    buffer = builder.create<memref::AllocOp>(loc, memrefType);
                    bufferMap[bufKey] = buffer;
                } else {
                    buffer = bufferMap[bufKey];
                }
            }
            
            // Create BD ID constant
            auto bdIdConst = builder.create<arith::ConstantOp>(
                loc, builder.getI32Type(), builder.getI32IntegerAttr(bdIndexCounter++));
            
            // Create DMA BD config
            auto dmaBdOp = builder.create<dfschedule::ConfigDmaBdOp>(
                loc,
                dfschedule::BdHandleType::get(builder.getContext()),
                buffer,
                shimTile,
                bdIdConst.getResult(),
                builder.getI32IntegerAttr(params.offset),
                builder.getI32IntegerAttr(params.len),
                builder.getBoolAttr(params.enablePacket),
                builder.getI32IntegerAttr(params.packetId),
                builder.getI32IntegerAttr(params.nextBd));
            
            shimBdHandles[params.shimKey].push_back(dmaBdOp.getBdHandle());
        }
    }
    
    // 5. Create IO configurations for shim tiles
    std::map<TileKey, SmallVector<Value>> shimIoHandles;
    
    for (const auto &params : allIoConfigParams) {
        if (shimTileMap.find(params.shimKey) == shimTileMap.end()) continue;
        Value shimTile = shimTileMap[params.shimKey];
        
        // Get corresponding BD handle
        Value bdHandle;
        if (params.bdIndex < (int)shimBdHandles[params.shimKey].size()) {
            bdHandle = shimBdHandles[params.shimKey][params.bdIndex];
        } else if (!shimBdHandles[params.shimKey].empty()) {
            bdHandle = shimBdHandles[params.shimKey].back();
        } else {
            continue; // No BD handle available, skip this IO
        }
        
        // Create IO config
        auto createIoOp = builder.create<dfschedule::ConfigCreateIoOp>(
            loc,
            dfschedule::IoHandleType::get(builder.getContext()),
            bdHandle,
            shimTile,
            builder.getI32IntegerAttr(params.channel),
            builder.getStringAttr(params.direction),
            builder.getStringAttr(params.ioOperation));
        
        shimIoHandles[params.shimKey].push_back(createIoOp.getIoHandle());
    }
    
    // 6. Build list of core tiles and their packet symbols for merged kernel group
    SmallVector<Value> allCoreTiles;
    SmallVector<Attribute> allPacketSymbols;
    SmallVector<Attribute> allComputeKernelArgs;
    
    int packetIdx = 0;
    for (auto &[key, tileInfo] : info.coreTiles) {
        if (coreTileMap.find(key) == coreTileMap.end()) continue;
        Value coreTile = coreTileMap[key];
        allCoreTiles.push_back(coreTile);
        
        // Use the first packet symbol for this tile (or create one)
        if (!tileInfo.packetSymbols.empty()) {
            allPacketSymbols.push_back(tileInfo.packetSymbols[0]);
        } else {
            std::string pktName = "packet" + std::to_string(packetIdx);
            allPacketSymbols.push_back(SymbolRefAttr::get(builder.getContext(), pktName));
        }
        
        // Use the first compute kernel arg for this tile (or default)
        if (!tileInfo.computeKernelArgs.empty()) {
            allComputeKernelArgs.push_back(tileInfo.computeKernelArgs[0]);
        } else {
            allComputeKernelArgs.push_back(SymbolRefAttr::get(builder.getContext(), "compute0"));
        }
        
        packetIdx++;
    }
    
    // 7. Create SINGLE merged load_kernel_group (if core tiles exist)
    Value launchEvent;
    if (!allCoreTiles.empty()) {
        SmallVector<Attribute> calleeAttrs;
        calleeAttrs.push_back(SymbolRefAttr::get(builder.getContext(), info.kernelName));
        
        auto loadKernelGroupOp = builder.create<dfschedule::LoadKernelGroupOp>(
            loc,
            dfschedule::KernelGroupType::get(builder.getContext()),
            allCoreTiles,
            builder.getArrayAttr(calleeAttrs),
            builder.getArrayAttr(allComputeKernelArgs),
            builder.getArrayAttr(allPacketSymbols));
        
        // 8. Create SINGLE launch_kernel_group
        auto launchKernelGroupOp = builder.create<dfschedule::LaunchKernelGroupOp>(
            loc,
            dfschedule::EventType::get(builder.getContext()),
            loadKernelGroupOp.getKernelGroup());
        
        launchEvent = launchKernelGroupOp.getEvent();
    }
    
    // 9. Create getBdId and start_io for each shim tile
    SmallVector<Value> allEvents;
    if (launchEvent) {
        allEvents.push_back(launchEvent);
    }
    
    for (auto &[key, ioHandles] : shimIoHandles) {
        if (shimTileMap.find(key) == shimTileMap.end()) continue;
        Value shimTile = shimTileMap[key];
        
        // Create getBdId
        auto getBdIdOp = builder.create<dfschedule::GetBdIdOp>(
            loc,
            builder.getI32Type(),
            shimTile);
        
        // Create start_io for each IO handle
        for (Value ioHandle : ioHandles) {
            auto startIoOp = builder.create<dfschedule::StartIoOp>(
                loc,
                dfschedule::EventType::get(builder.getContext()),
                ioHandle,
                getBdIdOp.getBdId());
            allEvents.push_back(startIoOp.getEvent());
        }
    }
    
    // 10. Create SINGLE merged schedule.wait with ALL events
    if (!allEvents.empty()) {
        builder.create<dfschedule::ScheduleWaitOp>(loc, allEvents);
    }
    
    // ==========================================================
    // PART 2: Add a call to host_canonicalized inside func.func @main()
    // ==========================================================
    Block &mainBlock = funcOp.getBody().front();
    Operation *terminator = mainBlock.getTerminator();
    if (terminator) {
        builder.setInsertionPoint(terminator);
    } else {
        builder.setInsertionPointToEnd(&mainBlock);
    }
    
    // Use dfschedule.launchhost to invoke the host schedule block
    auto hostSymbol = SymbolRefAttr::get(builder.getContext(), "host_canonicalized");
    builder.create<dfschedule::LaunchHostOp>(loc, hostSymbol);
}

// Remove old distributed schedule operations
static void removeOldScheduleOps(ModuleScheduleInfo &info) {
    // Mark operations for removal (in reverse order to handle dependencies)
    SmallVector<Operation*> opsToRemove;
    
    // Remove wait ops first
    for (auto *op : info.scheduleWaitOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove start_io ops
    for (auto *op : info.startIoOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove getBdId ops
    for (auto *op : info.getBdIdOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove launch ops
    for (auto *op : info.launchKernelGroupOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove load_kernel_group ops
    for (auto *op : info.loadKernelGroupOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove packet ops
    for (auto *op : info.packetOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove createIo ops
    for (auto *op : info.configCreateIoOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove dmaBd ops
    for (auto *op : info.configDmaBdOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove declareTile ops
    for (auto *op : info.declareTileOps) {
        opsToRemove.push_back(op);
    }
    
    // Remove declareTensor ops
    for (auto *op : info.declareTensorOps) {
        opsToRemove.push_back(op);
    }
    
    // Erase dfschedule operations (safe, no nested structure issues)
    for (auto *op : opsToRemove) {
        if (op->use_empty()) {
            op->erase();
        }
    }
}

// Remove scf.execute_region blocks, tensor.empty, declare_data, and constants from func.func main
// This is done separately to avoid memory corruption from nested op pointer invalidation
static void removeExecuteRegionsFromMain(func::FuncOp mainFunc) {
    if (!mainFunc) return;
    
    // Collect ops to erase (fresh collection, not using old pointers)
    SmallVector<Operation*> regionsToErase;
    SmallVector<Operation*> otherOpsToErase;
    
    mainFunc.walk([&](Operation *op) {
        if (isa<scf::ExecuteRegionOp>(op)) {
            regionsToErase.push_back(op);
        } else if (op->getParentOp() == mainFunc.getOperation()) {
            // Only collect ops that are direct children of main's block
            if (isa<tensor::EmptyOp>(op)) {
                otherOpsToErase.push_back(op);
            } else if (isa<arith::ConstantOp>(op)) {
                otherOpsToErase.push_back(op);
            } else if (op->getName().getStringRef() == "dfscheblueprint.declare_data") {
                otherOpsToErase.push_back(op);
            }
        }
    });
    
    // Erase scf.execute_region ops first (this also erases all nested ops)
    for (auto *op : regionsToErase) {
        if (op->use_empty()) {
            op->erase();
        }
    }
    
    // Erase other top-level ops (tensor.empty, arith.constant, declare_data)
    for (auto *op : otherOpsToErase) {
        if (op->use_empty()) {
            op->erase();
        }
    }
}

} // namespace

namespace mlir {

void ScheduleCanonicalizePass::runOnOperation() {
    ModuleOp moduleOp = getOperation();
    
    ModuleScheduleInfo info;
    
    // Step 1: Collect all dfschedule operations
    collectScheduleOps(moduleOp, info);
    
    // Early exit if no schedule ops found
    if (info.declareTileOps.empty() && info.loadKernelGroupOps.empty()) {
        return;
    }
    
    // Step 2: Associate packets with tiles
    associatePacketsWithTiles(info);
    
    // Step 3: Associate DMA configs with shim tiles
    associateDmaWithShimTiles(info);
    
    // Step 4: Find the main function to insert canonicalized host block
    func::FuncOp mainFunc = nullptr;
    moduleOp.walk([&](func::FuncOp funcOp) {
        if (funcOp.getName() == "main") {
            mainFunc = funcOp;
        }
    });
    
    if (!mainFunc) {
        // No main function, skip
        return;
    }
    
    // Step 5: Create canonicalized schedule inside func.func @main()
    // Operations are placed in an scf.execute_region block to group them
    OpBuilder builder(moduleOp.getContext());
    Location loc = mainFunc.getLoc();
    
    createCanonicalizedSchedule(builder, loc, info, moduleOp, mainFunc);
    
    // Step 6: Remove old distributed dfschedule operations
    removeOldScheduleOps(info);
    
    // Step 7: Remove scf.execute_region blocks (contains extract_slice, routing ops)
    // and tensor.empty from func.func main
    removeExecuteRegionsFromMain(mainFunc);
}

} // namespace mlir


