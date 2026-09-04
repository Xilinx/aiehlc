/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passdmaphoptoroutinghw.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "routing/routingpath.h"
#include "routinghwmanager.h"
#include "routingmanager.h"
#include "routinghw_pkt_slot.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlir;
using namespace dmaphop;
using namespace routinghw;

namespace {

int ioIdx = 0;

// Info about a consumer/producer op, captured before erasure
struct DmaPortInfo {
    std::string portSymName; // 'from' (consumer) or 'tp' (producer) port symbol
    int dmaPort;
};

// Helper: resolve a symbol from create_path's producers/consumers array.
// If the symbol refers to a dmaphop.consumer, follows from -> port.
// If the symbol refers to a dmaphop.producer, follows tp -> port.
// Otherwise assumes it's a direct port symbol.
static dmaphop::port resolvePortFromPathSymbol(Operation *contextOp, FlatSymbolRefAttr symRef) {
    // Try direct port lookup first (shim/mem ports)
    if (auto portOp = SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(contextOp, symRef))
        return portOp;
    // Try consumer indirection
    if (auto consumerOp = SymbolTable::lookupNearestSymbolFrom<dmaphop::consumer>(contextOp, symRef)) {
        auto fromRef = consumerOp.getFromAttr();
        return SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(contextOp, fromRef);
    }
    // Try producer indirection
    if (auto producerOp = SymbolTable::lookupNearestSymbolFrom<dmaphop::producer>(contextOp, symRef)) {
        auto tpRef = producerOp.getTpAttr();
        return SymbolTable::lookupNearestSymbolFrom<dmaphop::port>(contextOp, tpRef);
    }
    return nullptr;
}

// Helper: look up dma_port from a dmaphop.consumer or dmaphop.producer op
// by matching its sym_name (or from/tp port ref) to the given symbol name.
// Returns the dma_port value, or std::nullopt if not found.
static std::optional<int64_t> lookupDmaPort(Operation *contextOp, StringRef portSymName) {
    auto parentOp = contextOp->getParentOfType<ModuleOp>();
    if (!parentOp)
        parentOp = contextOp->getParentOfType<ModuleOp>();
    Operation *searchRoot = parentOp ? parentOp.getOperation() : contextOp->getParentOp();

    std::optional<int64_t> result;
    searchRoot->walk([&](Operation *op) {
        if (result.has_value())
            return WalkResult::interrupt();
        if (auto consumerOp = dyn_cast<dmaphop::consumer>(op)) {
            // Match by own symbol name (from create_path) or by from port ref
            if (consumerOp.getSymName() == portSymName || consumerOp.getFrom() == portSymName) {
                result = consumerOp.getDmaPort();
                return WalkResult::interrupt();
            }
        } else if (auto producerOp = dyn_cast<dmaphop::producer>(op)) {
            // Match by own symbol name (from create_path) or by tp port ref
            if (producerOp.getSymName() == portSymName || producerOp.getTp() == portSymName) {
                result = producerOp.getDmaPort();
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return result;
}

/*
// Helper structures from routinglower.cpp - needed for GetSeqPath and ParseTheCCTRoutingPath
struct StreamCCTConnection {
    PortDirection SlaveReceiveForwardDirection;
    int SlaveReceiveForwardDirectionPortIdx;
    PortDirection localDMAForwardDirection;
    int localDMAForwardPortIdx;
    PortDirection MasterSendToNextTileDirection;
    int MasterSendToNextTileDirectionPortIdx;
};

struct TileListRoutingMap {
    std::unordered_map<Point, StreamCCTConnection, Point::Hash> tilemap;
    std::vector<Point> tilelist;
};

struct StreamPKTConnection {
    PortDirection SlaveReceiveForwardDirection;
    int SlaveReceiveForwardDirectionPortIdx;
    int SlaveReceivePktID;
    int SlaveReceivePktType;
    int localDMAForwardPortIdx;
    int localDMAForwardPktID;
    int localDMAForwardPktType;
    PortDirection MasterSendToNextTileDirection;
    int MasterSendToNextTileDirectionPortIdx;
};

struct TileListPktRoutingNode {
    Point tile;
    StreamPKTConnection pktconn;
    Operation* tileOp;
};
*/
// Functions from routinglower.cpp (lines 12-381)
std::optional<TileListPktRoutingNode> GatherPktRoutingPathCreate(Operation* op,
                             uint32_t dioid,
                             Point shimpoint,
                             std::shared_ptr<DataIO>  dio,
                             std::optional<std::shared_ptr<const RoutingPath>> rpath, 
                             std::vector<Point>& tilist,
                             std::unordered_map<Point, Operation*, Point::Hash> dsttiles,
                             RoutingTopology & router_,
                             ConversionPatternRewriter& rewriter) {
    auto getrowcol =  [] (routinghw::TileCreate& creatileop) -> std::vector<int> {
            std::vector<int> ret(2,0);
            if (auto rowAttr = creatileop.getRowAttr()) {
                ret[0] = rowAttr.getInt();
            } 
            if (auto colAttr = creatileop.getColAttr()) {
                ret[1] = colAttr.getInt();
            }
            return ret;
    };

    if (!rpath || !*rpath || dsttiles.empty()) {
        return std::nullopt; // Exit if no valid routing path is provided.
    }
    TileListPktRoutingNode ret;

    std::vector<Point> pktmergetile;
    for(auto x: dsttiles) {
        pktmergetile.push_back(x.first);
    }
    //sort the pktmergetile in asending order
    std::sort(pktmergetile.begin(), pktmergetile.end(), [](const Point& a, const Point& b) {
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });

    //sort tilist in asending order
    std::sort(tilist.begin(), tilist.end(), [](const Point& a, const Point& b) {
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });
    
    // For core-to-shim (S2MM) packet gather routing:
    // We need to reuse the existing shim DataIO based on shimpoint and channel
    // The passed 'dio' parameter already has the shim column and channel information
    
    int shimcol = shimpoint.c;
    int shimchannel = dio->channel();
    
    // Find the existing DataIO using the shim column and channel
    int diogetherid = dio->id();
    auto rpath2 = router_.createPath(diogetherid, pktmergetile);
    if (!rpath2) {
        return std::nullopt;
    }
    
    std::unordered_map<Point, std::vector<int>, Point::Hash> tileMasterPortMapping;
    std::unordered_map<Point, Operation*, Point::Hash> pathtiles;
    std::unordered_map<Point, StreamPKTConnection, Point::Hash> pktswitchmap;

    // parse and set dma and slave master
    // create empty structure for each dstPoint
    for (const auto& dstPoint : tilist) {
        pktswitchmap[dstPoint] = StreamPKTConnection{};
    }

    // Build indexed list of producer port symbol names from path op.
    // Each corePortOut{i} carries its own dmapktid (set by DmapToDmaphopPass).
    SmallVector<StringRef, 4> producerSymNames;
    if (auto pathOp = dyn_cast<dmaphop::create_path>(op)) {
        for (auto attr : pathOp.getProducers()) {
            if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
                producerSymNames.push_back(symRef.getValue());
            }
        }
    }

    // set the local DMA pkt connection
    //  Use dma_port from dmaphop.producer ops to determine the DMA port.
    //  This ensures consistency with host.cc which uses the same channel number.
    int tileIndex = 0;
    for (const auto &dstPoint : tilist) {
        int dmaportNum = 0; // default DMA port 0

        // Read dmapktid from the corresponding corePortOut symbol,
        // and dma_port from the dmaphop.producer op.
        int pktId = 1; // fallback: 1-based
        if (tileIndex < (int)producerSymNames.size()) {
            auto symRef = FlatSymbolRefAttr::get(op->getContext(), producerSymNames[tileIndex]);
            // Use resolvePortFromPathSymbol to handle producer symbol indirection
            if (auto portOp = resolvePortFromPathSymbol(op, symRef)) {
                if (auto pktIdOpt = portOp.getDmapktid()) {
                    pktId = static_cast<int>(*pktIdOpt);
                }
            }
            // Look up the pre-allocated DMA port from the dmaphop.producer op
            if (auto dmaPortOpt = lookupDmaPort(op, producerSymNames[tileIndex])) {
                dmaportNum = static_cast<int>(*dmaPortOpt);
            }
        }
        // set prev tile master port and dma port
        struct StreamPKTConnection &curtileconf = pktswitchmap[dstPoint];
        curtileconf.localDMAForwardPortIdx = dmaportNum;
        curtileconf.localDMAForwardPktID = pktId;
        curtileconf.localDMAForwardPktType = 0;
        tileIndex++;
    }
    //set the slave master direction
    auto prevpoint = tilist[0];
    for (const auto& dstPoint : tilist) {
        struct StreamPKTConnection& prevtileconf = pktswitchmap[prevpoint];
        struct StreamPKTConnection& curtileconf = pktswitchmap[dstPoint];
        //set the in out as default None, as the first tile slave should be None
        //and the last tile master should be None
        curtileconf.SlaveReceiveForwardDirection = PortDirection::NONE;
        curtileconf.MasterSendToNextTileDirection = PortDirection::NONE;
        if (prevpoint == dstPoint) {
            continue;// when process the first point by pass. as the occupy logic need two point
        }
        
        //get the connection port and direction
        int portNum = 0;
        PortDirection portdirectionPrevMaster, portdirectionCurSlave;
        if (!router_.occupyLink(prevpoint, dstPoint, dioid, portNum, portdirectionPrevMaster, portdirectionCurSlave)) {
            llvm::outs() << "link occupy failed " << "\n";
            assert(0);
            return std::nullopt;
        }
        
        //set prev tile master port and dma port
        prevtileconf.MasterSendToNextTileDirection = portdirectionPrevMaster;
        prevtileconf.MasterSendToNextTileDirectionPortIdx = portNum;
        //set currenttile receive/slave port
        curtileconf.SlaveReceiveForwardDirection = portdirectionCurSlave;
        curtileconf.SlaveReceiveForwardDirectionPortIdx = portNum;
        curtileconf.SlaveReceivePktID = 0;//forward all packet
        curtileconf.SlaveReceivePktType = 0;
        //
        prevpoint = dstPoint;
    }
    //create the op call
    for (const auto& dstPoint : tilist) {
        const Point& key = dstPoint;

        auto output = rewriter.getI32Type();
        auto curTileOp = dyn_cast<routinghw::TileCreate>(dsttiles[key]);
        const StreamPKTConnection& value = pktswitchmap[key];

        // Print the key
        std::cout << "\nKey: (row is " << key.r << ", col is " << key.c << ")" << std::endl;

        // Print the members of the value struct
        std::cout << "  - SlaveReceiveForwardDirection: " << PortDirectiontoString(value.SlaveReceiveForwardDirection) << std::endl;
        std::cout << "  - SlaveReceiveForwardDirectionPortIdx: " << (int)value.SlaveReceiveForwardDirectionPortIdx << std::endl;
        std::cout << "  - SlaveReceivePktID: " << value.SlaveReceivePktID << std::endl;
        std::cout << "  - SlaveReceivePktType: " << value.SlaveReceivePktType << std::endl;
        std::cout << "  - localDMAForwardPortIdx: " << value.localDMAForwardPortIdx << std::endl;
        std::cout << "  - localDMAForwardPktID: " << value.localDMAForwardPktID << std::endl;
        std::cout << "  - localDMAForwardPktType: " << value.localDMAForwardPktType << std::endl;
        std::cout << "  - MasterSendToNextTileDirection: " << PortDirectiontoString(value.MasterSendToNextTileDirection) << std::endl;
        std::cout << "  - MasterSendToNextTileDirectionPortIdx: " << (int)(value.MasterSendToNextTileDirectionPortIdx) << std::endl;

        // For output gather flows, preserve packet headers when OOO is enabled
        // so shim S2MM DMA can do OOO BD dispatch based on packet_id.
        bool preserveHdr = true;
        auto pktSlot = routinghw::pktslot::makePktSlotAttrs(rewriter);
        rewriter.create<routinghw::ConnectStreamPktSwitchPort>(
            op->getLoc(), // Operation location
            output,
            curTileOp.getResult(), // Tile to be configured
            rewriter.getStringAttr(PortDirectiontoString(
                value.SlaveReceiveForwardDirection)), // Direction of the port receiving the stream
            rewriter.getI32IntegerAttr((int)value.SlaveReceiveForwardDirectionPortIdx), // Index of the receiving port
            rewriter.getI32IntegerAttr(value.SlaveReceivePktID),                        // Packet ID to expect
            rewriter.getI32IntegerAttr(value.SlaveReceivePktType),                      // Packet Type to expect
            rewriter.getStringAttr(PortDirectiontoString(PortDirection::DMA)), // local DMA direction NONE means no DMA
            rewriter.getI32IntegerAttr(value.localDMAForwardPortIdx),          // Index of the local DMA port to send to
            rewriter.getI32IntegerAttr(value.localDMAForwardPktID),            // Packet ID for the DMA transfer
            rewriter.getI32IntegerAttr(value.localDMAForwardPktType),          // Packet Type for the DMA transfer
            rewriter.getStringAttr(
                PortDirectiontoString(value.MasterSendToNextTileDirection)), // No forwarding: empty master direction
            rewriter.getI32IntegerAttr(
                (int)(value.MasterSendToNextTileDirectionPortIdx)), // No forwarding: port index 0
            rewriter.getBoolAttr(preserveHdr), ROUTINGHW_PKT_SLOT_ATTRS(pktSlot));
    }
    ret.tile = tilist.back();
    ret.pktconn = pktswitchmap[ret.tile];
    ret.tileOp = dsttiles[ret.tile];
    // connect pkt merge/data gather into shim tile
    return std::make_optional<TileListPktRoutingNode>(ret);
}

std::optional<TileListRoutingMap>
GetSeqPath(std::optional<std::shared_ptr<const RoutingPath>> rpath, std::shared_ptr<DataIO> dio,
           std::unordered_map<Point, Operation *, Point::Hash> dsttiles,
           StreamType streamtype, // 0 no dma, 1 dma receive
           std::optional<TileListPktRoutingNode> lastPkttilemap, RoutingTopology &router_,
           ConversionPatternRewriter &rewriter,
           const std::unordered_map<std::string, Point> *portSymToTilePoint = nullptr,
           const std::unordered_map<std::string, DmaPortInfo> *consumerDmaPortInfo = nullptr,
           const SmallVector<std::string, 8> *pathConsumerNames = nullptr) {
    TileListRoutingMap troutingmap;
    std::unordered_map<Point, StreamCCTConnection, Point::Hash> & connectionData = troutingmap.tilemap;
    std::vector<Point> & orderedPathPoints = troutingmap.tilelist;
    uint32_t dioid = dio->id();
    if (!rpath || !(*rpath)) {
        return std::nullopt; // No path to process
    }

    auto outputType = rewriter.getI32Type();
    auto tree = (*rpath)->multipaths();

    // --- Phase 1: Build connection map AND an ordered list of points ---
    
    
    std::unordered_set<Point, Point::Hash> pointsInOrderedList; // Helper to avoid duplicates

    // Helper lambda to add a point to our ordered list, ensuring uniqueness
    auto addPointToOrderedList = [&](const Point& p) {
        if (pointsInOrderedList.find(p) == pointsInOrderedList.end()) {
            pointsInOrderedList.insert(p);
            orderedPathPoints.push_back(p);
        }
    };

    // 1a. Iterate over path links to populate connectionData and the ordered list
    uint8_t tree_round = 0;
    for (const auto& branch : tree.branches) {
        
        for (size_t i = 0; i < branch.size(); ++i) {
            const Point& currentPoint = branch[i];
            addPointToOrderedList(currentPoint); // Add point to maintain order
            if (0 == tree_round && 0 == i) {
                connectionData[currentPoint].SlaveReceiveForwardDirection = PortDirection::NONE;
            }
            if ( i < branch.size() - 1) {
                const Point& nextPoint = branch[i+1];

                // Skip links between tiles that are both already in the tree —
                // these were already configured by a previous branch.
                bool currentAlreadyProcessed = (pointsInOrderedList.find(currentPoint) != pointsInOrderedList.end());
                bool nextAlreadyProcessed = (pointsInOrderedList.find(nextPoint) != pointsInOrderedList.end());
                if (currentAlreadyProcessed && nextAlreadyProcessed) {
                    continue; // Both tiles already routed, skip
                }

                int portNum;
                PortDirection slaveDirOnNext, masterDirOnCurrent;
                if (!router_.occupyLink(currentPoint, nextPoint, dioid, portNum, masterDirOnCurrent, slaveDirOnNext)) {
                    llvm::report_fatal_error("Failed to occupy link in routing topology.");
                }

                // If this tile already has a master direction (from a
                // previous branch), store this one as the secondary
                // master to support fan-out at intermediate tiles.
                std::cout << "[GetSeqPath-dmap] point=(" << currentPoint.r << "," << currentPoint.c
                          << ") occupyLink->portNum=" << portNum << " masterDir=" << (int)masterDirOnCurrent
                          << " existingMasterDir=" << (int)connectionData[currentPoint].MasterSendToNextTileDirection
                          << " (NONE=" << (int)PortDirection::NONE << ")" << std::endl;
                if (connectionData[currentPoint].MasterSendToNextTileDirection != PortDirection::NONE) {
                    std::cout << "[GetSeqPath-dmap]   -> storing in SECONDARY" << std::endl;
                    connectionData[currentPoint].MasterSendToNextTileDirection2 = masterDirOnCurrent;
                    connectionData[currentPoint].MasterSendToNextTileDirectionPortIdx2 = portNum;
                } else {
                    std::cout << "[GetSeqPath-dmap]   -> storing in PRIMARY" << std::endl;
                    connectionData[currentPoint].MasterSendToNextTileDirection = masterDirOnCurrent;
                    connectionData[currentPoint].MasterSendToNextTileDirectionPortIdx = portNum;
                }

                // Only set slave/reset master on nextPoint if it's new.
                // Tiles already configured by a prior branch must keep their
                // existing master direction — resetting it destroys the prior
                // branch's routing.
                if (!nextAlreadyProcessed) {
                    connectionData[nextPoint].SlaveReceiveForwardDirection = slaveDirOnNext;
                    connectionData[nextPoint].SlaveReceiveForwardDirectionPortIdx = portNum;
                    // set next master into None
                    connectionData[nextPoint].MasterSendToNextTileDirection = PortDirection::NONE;
                }
            }
        }
        tree_round++;
    }

    //Process output dataio, when the last tile is be the shim tile of dataio

    if (dio->type() == IOType::Output) {
        auto lastilepoint = orderedPathPoints.back();
        Point shimpoint = { dio->rowpos(),dio->colpos() };
        if (lastilepoint == shimpoint) {
             if (auto shimPortInfo = dio->getshimport()) {
                connectionData[shimpoint].MasterSendToNextTileDirection = shimPortInfo->dir_;
                connectionData[shimpoint].MasterSendToNextTileDirectionPortIdx = shimPortInfo->portnum_;
            }
        }
    } else if (dio->type() == IOType::Input) {// 1c. Handle the special case for the starting SHIM tile's input
        // 1c. Handle the special case for the starting SHIM tile's input
        PortDirection shimDir = PortDirection::South;
        int shimPortNum = 3; // A reasonable default
        if (auto shimPortInfo = dio->getshimport()) {
            shimDir = shimPortInfo->dir_;
            shimPortNum = shimPortInfo->portnum_;
        }
        Point dioshimpoint = Point{dio->rowpos(), dio->colpos()};
        connectionData[dioshimpoint].SlaveReceiveForwardDirection = shimDir;
        connectionData[dioshimpoint].SlaveReceiveForwardDirectionPortIdx = shimPortNum;
    }

    // 1b. Populate DMA connection information from dmaphop.consumer ops
    // DMA port numbers are pre-allocated in DmapToDmaphopPass and stored as
    // dmaphop.consumer ops. Walk for consumer ops whose port_sym points to a
    // core tile in dsttiles.
    auto rm = router_.getRM();

    // Build a reverse map: core tile Point -> consumer dma_port
    // Use portSymToTilePoint (pre-populated before conversion) to resolve
    // consumer's 'from' port symbol to a tile Point, avoiding SymbolTable
    // lookups on port ops that may have been erased.
    std::unordered_map<Point, int, Point::Hash> consumerDmaPortMap;
    {
        if (portSymToTilePoint && consumerDmaPortInfo && pathConsumerNames) {
            // Use pre-populated maps to resolve consumer dma_port.
            // Both consumer ops and port ops may be erased by conversion patterns,
            // so we use the pre-populated maps from the pre-pass walk.
            // Only process consumers that belong to THIS path.
            for (const auto &consumerSym : *pathConsumerNames) {
                auto consumerIt = consumerDmaPortInfo->find(consumerSym);
                if (consumerIt == consumerDmaPortInfo->end())
                    continue;
                const auto &info = consumerIt->second;
                auto portIt = portSymToTilePoint->find(info.portSymName);
                if (portIt != portSymToTilePoint->end()) {
                    consumerDmaPortMap[portIt->second] = info.dmaPort;
                }
            }
        }
    }

    for (const auto& p : orderedPathPoints) {
        connectionData[p].localDMAForwardDirection = PortDirection::NONE;
        // CONTROL flows deliver to each target core like BROADCAST, but the
        // final termination is the tile's CTRL port (handled in
        // ParseTheCCTRoutingPath). Marking localDMAForwardDirection here lets the
        // per-tile termination fire; the DMA port index is unused for CTRL sinks.
        if (rm->getrsc()->tileType(p.r, p.c) == TileType::Core &&
            (StreamType::BROADCAST == streamtype || StreamType::CONTROL == streamtype) &&
            dsttiles.find(p) != dsttiles.end()) {
            auto it = consumerDmaPortMap.find(p);
            if (it != consumerDmaPortMap.end()) {
                connectionData[p].localDMAForwardDirection = PortDirection::DMA;
                connectionData[p].localDMAForwardPortIdx = it->second;
            } else {
                // Fallback to dynamic allocation if no consumer op found
                if (auto portnumptr = rm->tile(p.r, p.c).occupyport(IOType::Input, PortDirection::DMA, -1)) {
                    connectionData[p].localDMAForwardDirection = PortDirection::DMA;
                    connectionData[p].localDMAForwardPortIdx = *portnumptr;
                }
            }
        }
    }

    return std::make_optional<TileListRoutingMap>(troutingmap);
}

void ParseTheCCTRoutingPath(Operation *op, std::optional<TileListPktRoutingNode> lastPkttilemap,
                            StreamType streamtype, // 0 normal, 1 broadcast
                            uint32_t dioid, Point shimpoint, std::shared_ptr<DataIO> dio, IOShimTileCreate shimio,
                            std::optional<std::shared_ptr<const RoutingPath>> rpath,
                            std::unordered_map<Point, Operation *, Point::Hash> dsttiles, RoutingTopology &router_,
                            ConversionPatternRewriter &rewriter,
                            const std::unordered_map<std::string, Point> *portSymToTilePoint = nullptr,
                            const std::unordered_map<std::string, DmaPortInfo> *consumerDmaPortInfo = nullptr,
                            const SmallVector<std::string, 8> *pathConsumerNames = nullptr) {

    if (!rpath || !(*rpath)) {
        return; // No path to process
    }

    auto loc = op->getLoc();
    auto outputType = rewriter.getI32Type();
    // Control-packet flow: the stream carries config/register writes that must
    // terminate at each target tile's CTRL master stream-switch port (packet
    // mode, header preserved) instead of the S2MM DMA data path. Uses the same
    // path/fan-out as BROADCAST. A single control stream id is used for the flow
    // (5-bit stream-packet-header id space, bits[4:0]); broadcast targets share
    // the id so identical writes land on every tapped tile.
    bool isControl = (streamtype == StreamType::CONTROL);
    int ctrlPktId = static_cast<int>(dioid & 0x1f);
    // --- Phase 1: Build connection map AND an ordered list of points ---
    auto troutingmap =
        GetSeqPath(rpath, dio, dsttiles, streamtype /* 0 normal no dma, 1 broadcast dma receive*/, lastPkttilemap,
                   router_, rewriter, portSymToTilePoint, consumerDmaPortInfo, pathConsumerNames);
    if (!troutingmap) {
        return;
    }

    std::unordered_map<Point, StreamCCTConnection, Point::Hash> & connectionData = troutingmap->tilemap;
    std::vector<Point> & orderedPathPoints = troutingmap->tilelist;
    
    // --- Phase 2: Create all tile operations first, IN ORDER ---
    // Start with existing tiles from dsttiles (created by DmaphopTileConversionPattern)
    // Then create intermediate routing tiles that router_.createPath() inserted
    std::unordered_map<Point, Operation*, Point::Hash> allTileOps = dsttiles;
    
    for (const Point& p : orderedPathPoints) {
        if (allTileOps.find(p) == allTileOps.end()) {
            // This is an intermediate routing tile (e.g., mem tile at row 1)
            // that router_.createPath() inserted to connect shim to core tiles
            allTileOps[p] = rewriter.create<routinghw::TileCreate>(
                loc, outputType, p.r, p.c, "tile in path");
        }
    }

    // --- Phase 3: Create enable shim port operations ---
    for (const Point& point : orderedPathPoints) {
        auto it = connectionData.find(point);
        if (it == connectionData.end()) continue;
        
        const StreamCCTConnection& conn = it->second;
        
        if (conn.SlaveReceiveForwardDirection == PortDirection::NONE) {
            continue;
        }
        
        StringRef inputDirStr = PortDirectiontoString(conn.SlaveReceiveForwardDirection);
        int inputPortIdx = conn.SlaveReceiveForwardDirectionPortIdx;

        // Special handling for the SHIM tile to enable its external port
        if (point == shimpoint) {
            // Use the actual hardware port number from shimport, not the stream switch port index
            int shimHwPortNum = inputPortIdx; // Default fallback
            if (auto shimPortInfo = dio->getshimport()) {
                shimHwPortNum = shimPortInfo->portnum_;
            }

            if (dio->type() == IOType::Input) {
                rewriter.create<EnableExtToAieShimPort>(loc, outputType, shimio.getResult(), inputDirStr,
                                                        shimHwPortNum);
            } else {
                rewriter.create<EnableAieToExtShimPort>(loc, outputType, shimio.getResult(), inputDirStr,
                                                        shimHwPortNum);
            }
        }
    }
    
    // --- Phase 4: Create stream switch connections IN ORDER ---
    for (const Point& point : orderedPathPoints) {
        // Look up the connection info from our map
        auto it = connectionData.find(point);
        if (it == connectionData.end()) continue; // This point might not have connections (e.g., an un-routed destination)
        
        const StreamCCTConnection& conn = it->second;
        
        // Get the tile operation
        auto tileOpIt = allTileOps.find(point);
        if (tileOpIt == allTileOps.end()) {
            // This shouldn't happen since we created all tiles above
            continue;
        }
        
        // Ensure the tile has an input port to connect from
        if (conn.SlaveReceiveForwardDirection == PortDirection::NONE) {
            if (lastPkttilemap && lastPkttilemap->tile == point) {
                auto output = rewriter.getI32Type();
                auto tileOp = dyn_cast<routinghw::TileCreate>((Operation* )lastPkttilemap->tileOp);
                // Transition op at PKT→CCT merge point. When OOO is enabled,
                // preserve headers so shim S2MM DMA can read packet_id for
                // OOO BD dispatch.
                bool preserveHdrTransition = true;
                auto pktSlot = routinghw::pktslot::makePktSlotAttrs(rewriter);
                rewriter.create<routinghw::ConnectStreamPktSwitchPort>(
                    loc, // Operation location
                    output,
                    tileOp.getResult(), // Tile to be configured
                    rewriter.getStringAttr(
                        PortDirectiontoString(PortDirection::NONE)), // Direction of the port receiving the stream
                    rewriter.getI32IntegerAttr(0),                   // Index of the receiving port
                    rewriter.getI32IntegerAttr(0),                   // Packet ID to expect
                    rewriter.getI32IntegerAttr(0),                   // Packet Type to expect
                    rewriter.getStringAttr(PortDirectiontoString(PortDirection::NONE)),
                    rewriter.getI32IntegerAttr(0), // Index of the local DMA port to send to
                    rewriter.getI32IntegerAttr(0), // Packet ID for the DMA transfer
                    rewriter.getI32IntegerAttr(0), // Packet Type for the DMA transfer
                    rewriter.getStringAttr(PortDirectiontoString(
                        conn.MasterSendToNextTileDirection)), // No forwarding: empty master direction
                    rewriter.getI32IntegerAttr(
                        (int)(conn.MasterSendToNextTileDirectionPortIdx)), // No forwarding: port index 0
                    rewriter.getBoolAttr(preserveHdrTransition), ROUTINGHW_PKT_SLOT_ATTRS(pktSlot));
            }
            continue;
        }
        
        StringRef inputDirStr = PortDirectiontoString(conn.SlaveReceiveForwardDirection);
        int inputPortIdx = conn.SlaveReceiveForwardDirectionPortIdx;

        // Create stream switch connections
        if (point == shimpoint) {
            // Create connection from shim to the next tile in the path
            if (conn.MasterSendToNextTileDirection != PortDirection::NONE) {
                rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, shimio.getResult(),
                    inputDirStr, inputPortIdx,
                    PortDirectiontoString(conn.MasterSendToNextTileDirection), conn.MasterSendToNextTileDirectionPortIdx);
            }
            // Secondary master for fan-out at shim
            if (conn.MasterSendToNextTileDirection2 != PortDirection::NONE) {
                rewriter.create<ConnectStreamSingleSwitchPort>(
                    loc, outputType, shimio.getResult(), inputDirStr, inputPortIdx,
                    PortDirectiontoString(conn.MasterSendToNextTileDirection2),
                    conn.MasterSendToNextTileDirectionPortIdx2);
            }
        } else {
            // Handle regular tiles (core, mem, or intermediate routing tiles)
            auto currentTileOp = dyn_cast<routinghw::TileCreate>(tileOpIt->second);
            if (!currentTileOp) {
                // Not a regular tile, skip
                continue;
            }

            // Create connection to the next tile in the path
            if (conn.MasterSendToNextTileDirection != PortDirection::NONE) {
                rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, currentTileOp.getResult(),
                    inputDirStr, inputPortIdx,
                    PortDirectiontoString(conn.MasterSendToNextTileDirection), conn.MasterSendToNextTileDirectionPortIdx);
            }

            // Secondary master for fan-out at intermediate tiles
            if (conn.MasterSendToNextTileDirection2 != PortDirection::NONE) {
                rewriter.create<ConnectStreamSingleSwitchPort>(
                    loc, outputType, currentTileOp.getResult(), inputDirStr, inputPortIdx,
                    PortDirectiontoString(conn.MasterSendToNextTileDirection2),
                    conn.MasterSendToNextTileDirectionPortIdx2);
            }

            // Create connection to the local sink (if this is a destination core tile).
            if (conn.localDMAForwardDirection != PortDirection::NONE) {
                if (isControl) {
                    // Control-packet terminal: route the incoming stream into the
                    // tile's CTRL master stream-switch port (packet mode, header
                    // preserved) so the control-packet header/data drive an
                    // in-tile register/memory write. RoutingHWLowerPass forces the
                    // master port to CTRL and DONOT_DROP_HEADER when
                    // localsinkport=="CTRL"; forwardmaster is left NONE here.
                    auto pktSlot = routinghw::pktslot::makePktSlotAttrs(rewriter);
                    auto ctrlOp = rewriter.create<routinghw::ConnectStreamPktSwitchPort>(
                        loc, outputType, currentTileOp.getResult(),
                        rewriter.getStringAttr(inputDirStr),                                // receiveslavedirection
                        rewriter.getI32IntegerAttr(inputPortIdx),                           // receiveslaveportidx
                        rewriter.getI32IntegerAttr(ctrlPktId),                              // receiveslavepktid
                        rewriter.getI32IntegerAttr(0),                                      // receiveslavepkttype
                        rewriter.getStringAttr(PortDirectiontoString(PortDirection::NONE)), // localdmadirection = NONE
                        rewriter.getI32IntegerAttr(0),                                      // localdmaportidx
                        rewriter.getI32IntegerAttr(0),                                      // localdmapktid
                        rewriter.getI32IntegerAttr(0),                                      // localdmapkttype
                        rewriter.getStringAttr(PortDirectiontoString(PortDirection::NONE)), // forwardmasterdirection
                        rewriter.getI32IntegerAttr(0),                                      // forwardmasterportidx
                        rewriter.getBoolAttr(true),                                         // preserveheader
                        ROUTINGHW_PKT_SLOT_ATTRS(pktSlot));
                    ctrlOp->setAttr("localsinkport", rewriter.getStringAttr("CTRL"));
                } else {
                    rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, currentTileOp.getResult(),
                                                                   inputDirStr, inputPortIdx, "DMA",
                                                                   conn.localDMAForwardPortIdx);
                }
            }
        }
    }
}

// Helper structure to store tile information
struct TileInfo {
    Operation* tileOp;
    int col;
    int row;
    int channel;
    std::string tileType; // "shim" or "core"
};

// Helper structure to store the routing context
struct RoutingContext {
    //Operation* tileArrayHandle = nullptr;
    DenseMap<Value, TileInfo> tileMap;
    DenseMap<Value, Operation*> portToTileOpMap;
    // Map from port symbol name to tile Point — survives port erasure
    std::unordered_map<std::string, Point> portSymToTilePoint;
    // Map from consumer/producer sym_name to DmaPortInfo — survives consumer/producer erasure
    std::unordered_map<std::string, DmaPortInfo> consumerDmaPortInfo;
    std::vector<Operation*> orderedTileOps;
    bool isFirstTile = true;
    int pktId = 1;
};

// Pattern to convert dmaphop.tile and dmaphop.port to routinghw operations
struct DmaphopTileConversionPattern : public OpConversionPattern<dmaphop::tile> {
    explicit DmaphopTileConversionPattern(MLIRContext *context, RoutingTopology &router, RoutingContext &ctx)
        : OpConversionPattern<dmaphop::tile>(context), router_(router), routingCtx(ctx) {}

    LogicalResult matchAndRewrite(dmaphop::tile op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto output = rewriter.getI32Type();
        
        // Extract tile attributes
        std::string tileType = op.getTiletype().str();
        int64_t col = op.getCol();
        int64_t row = op.getRow();

        Operation* hwTileOp = nullptr;
        
        if (tileType == "shim") {
            // For shim tiles from dmaphop, we don't create IOShimTileCreate here
            // The actual shim tile will be created in DmaphopPathConversionPattern
            // based on the routing allocation (router_.createDataIO())
            // For now, just store the info and don't create any operation
            
            // Store tile info but without creating an operation
            TileInfo info;
            info.tileOp = nullptr;  // Will be set later if needed
            info.col = col;
            info.row = row;
            info.channel = 0;
            info.tileType = tileType;
            routingCtx.tileMap[op.getResult()] = info;
            
            // Erase the dmaphop.tile operation for shim tiles
            // They will be replaced by dynamically allocated shim tiles in the path conversion
            rewriter.eraseOp(op);
            return success();
        } else if (tileType == "core") {
            // Create Core tile
            hwTileOp = rewriter.create<routinghw::TileCreate>(
                loc,
                output,
                rewriter.getI32IntegerAttr(row),
                rewriter.getI32IntegerAttr(col),
                rewriter.getStringAttr("core_tile")
            );
            
            // Store tile info
            TileInfo info;
            info.tileOp = hwTileOp;
            info.col = col;
            info.row = row;
            info.channel = 0;
            info.tileType = tileType;
            routingCtx.tileMap[op.getResult()] = info;
            routingCtx.orderedTileOps.push_back(hwTileOp);
            
            // Replace the original tile operation with the hardware tile
            rewriter.replaceOp(op, hwTileOp->getResult(0));
        }
        
        return success();
    }

private:
    RoutingTopology &router_;
    RoutingContext &routingCtx;
};

// Pattern to handle dmaphop.port operations (just track them)
struct DmaphopPortConversionPattern : public OpConversionPattern<dmaphop::port> {
    explicit DmaphopPortConversionPattern(MLIRContext *context, RoutingContext &ctx)
        : OpConversionPattern<dmaphop::port>(context), routingCtx(ctx) {}

    LogicalResult matchAndRewrite(dmaphop::port op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Store port to tile mapping for later use
        Value tileValue = adaptor.getTile();
        auto it = routingCtx.tileMap.find(tileValue);
        if (it != routingCtx.tileMap.end()) {
            routingCtx.portToTileOpMap[op.getResult()] = it->second.tileOp;
            // Also store the port symbol name → tile Point mapping.
            // This survives port erasure and is used by DmaphopPathConversionPattern
            // to find producer/consumer tiles from the create_path attributes.
            routingCtx.portSymToTilePoint[op.getSymName().str()] = Point{it->second.row, it->second.col};
        }

        // Ports are implicit in routinghw, so we just erase them
        rewriter.eraseOp(op);
        return success();
    }

private:
    RoutingContext &routingCtx;
};

// Pattern to handle dmaphop.create_path - this is where we create stream switch connections
struct DmaphopPathConversionPattern : public OpConversionPattern<dmaphop::create_path> {
    explicit DmaphopPathConversionPattern(MLIRContext *context, RoutingTopology &router, RoutingContext &ctx)
        : OpConversionPattern<dmaphop::create_path>(context), router_(router), routingCtx(ctx) {}

    LogicalResult matchAndRewrite(dmaphop::create_path op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto output = rewriter.getI32Type();
        // Get producers attribute to determine which tiles should send to DMA (S2MM)
        // Producers are core tiles that send data FROM core TO shim/memory (Pull direction)
        // Use SymbolTable lookup instead of walk to find ports that may have been
        // scheduled for erasure by DmaphopPortConversionPattern.
        llvm::DenseSet<Value> producerPorts;
        llvm::DenseSet<StringRef> producerSymNames;
        if (auto producersAttr = op->getAttrOfType<ArrayAttr>("producers")) {
            for (auto symRef : producersAttr) {
                if (auto symRefAttr = dyn_cast<FlatSymbolRefAttr>(symRef)) {
                    producerSymNames.insert(symRefAttr.getValue());
                    // Use resolvePortFromPathSymbol to handle producer symbol indirection
                    if (auto portOp = resolvePortFromPathSymbol(op, symRefAttr)) {
                        producerPorts.insert(portOp.getResult());
                    }
                }
            }
        }

        // Get consumers attribute to determine which tiles should receive from DMA (MM2S)
        // Consumers are core tiles that receive data FROM shim/memory TO core (Push direction)
        llvm::DenseSet<Value> consumerPorts;
        llvm::DenseSet<StringRef> consumerSymNames;
        if (auto consumersAttr = op->getAttrOfType<ArrayAttr>("consumers")) {
            for (auto symRef : consumersAttr) {
                if (auto symRefAttr = dyn_cast<FlatSymbolRefAttr>(symRef)) {
                    consumerSymNames.insert(symRefAttr.getValue());
                    // Use resolvePortFromPathSymbol to handle consumer symbol indirection
                    if (auto portOp = resolvePortFromPathSymbol(op, symRefAttr)) {
                        consumerPorts.insert(portOp.getResult());
                    }
                }
            }
        }
        
        // Extract tiles and track which tiles have producer/consumer ports
        // Separate shim tiles from core tiles
        std::vector<Value> allTilesInPath;
        std::vector<Point> coreTileList;  // Core tiles for routing
        std::unordered_map<Point, Operation*, Point::Hash> dsttiles;  // Core tiles map
        std::vector<Point> consumerCoreTiles;  // Core tiles that consume (need DMA receive)
        std::vector<Point> producerCoreTiles;  // Core tiles that produce (need DMA send)
        
        Value shimTileValue;
        Point shimPoint{-1, -1};
        bool hasShim = false;
        bool isShimToCore = false;  // true: shim->core (MM2S), false: core->shim (S2MM)
        
        llvm::DenseSet<Value> seenTiles;
        llvm::DenseMap<Value, bool> tileHasProducer;  // Map tile -> has producer port (sends to DMA)
        llvm::DenseMap<Value, bool> tileHasConsumer;  // Map tile -> has consumer port (receives from DMA)
        int shimDmaChannel = 0;                       // direction_channel from the shim port op in dmaphop IR

        for (auto hopValue : adaptor.getHops()) {
            if (auto hopOp = hopValue.getDefiningOp<dmaphop::create_hop>()) {
                // Get source and destination ports
                Value srcPort = hopOp.getSource();
                Value dstPort = hopOp.getDestination();
                
                // Get the tiles for these ports
                auto srcPortOp = srcPort.getDefiningOp<dmaphop::port>();
                auto dstPortOp = dstPort.getDefiningOp<dmaphop::port>();
                
                if (srcPortOp) {
                    Value srcTile = srcPortOp.getTile();
                    if (!seenTiles.contains(srcTile)) {
                        allTilesInPath.push_back(srcTile);
                        seenTiles.insert(srcTile);

                        // Categorize tile
                        auto tileInfoIt = routingCtx.tileMap.find(srcTile);
                        if (tileInfoIt != routingCtx.tileMap.end()) {
                            if (tileInfoIt->second.tileType == "shim") {
                                shimTileValue = srcTile;
                                shimPoint = Point{tileInfoIt->second.row, tileInfoIt->second.col};
                                hasShim = true;
                                if (auto ch = srcPortOp.getDirectionChannel()) {
                                    shimDmaChannel = static_cast<int>(*ch);
                                }
                            } else if (tileInfoIt->second.tileType == "core") {
                                Point pt{tileInfoIt->second.row, tileInfoIt->second.col};
                                coreTileList.push_back(pt);
                                dsttiles[pt] = tileInfoIt->second.tileOp;
                            }
                        }
                    }
                    // Mark this tile as having a producer if source port is a producer
                    // Check both the Value set (from SymbolTable lookup) and the symbol name set
                    if (producerPorts.contains(srcPort) || producerSymNames.contains(srcPortOp.getSymName())) {
                        tileHasProducer[srcTile] = true;
                    }
                }

                if (dstPortOp) {
                    Value dstTile = dstPortOp.getTile();
                    if (!seenTiles.contains(dstTile)) {
                        allTilesInPath.push_back(dstTile);
                        seenTiles.insert(dstTile);

                        // Categorize tile
                        auto tileInfoIt = routingCtx.tileMap.find(dstTile);
                        if (tileInfoIt != routingCtx.tileMap.end()) {
                            if (tileInfoIt->second.tileType == "shim") {
                                shimTileValue = dstTile;
                                shimPoint = Point{tileInfoIt->second.row, tileInfoIt->second.col};
                                hasShim = true;
                                if (auto ch = dstPortOp.getDirectionChannel()) {
                                    shimDmaChannel = static_cast<int>(*ch);
                                }
                            } else if (tileInfoIt->second.tileType == "core") {
                                Point pt{tileInfoIt->second.row, tileInfoIt->second.col};
                                coreTileList.push_back(pt);
                                dsttiles[pt] = tileInfoIt->second.tileOp;
                            }
                        }
                    }
                    // Mark this tile as having a consumer if destination port is a consumer
                    // Check both the Value set (from SymbolTable lookup) and the symbol name set
                    if (consumerPorts.contains(dstPort) || consumerSymNames.contains(dstPortOp.getSymName())) {
                        tileHasConsumer[dstTile] = true;
                    }
                }
            }
        }
        //sort coreTileList in ascending order, to make sure the stream flow is from left to right, top to bottom
        std::sort(coreTileList.begin(), coreTileList.end(), [](const Point& a, const Point& b) {
            if (a.r == b.r) {
                return a.c < b.c;
            }
            return a.r < b.r;
        });
        if (allTilesInPath.empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Determine data flow direction based on shim position and producers/consumers
        if (hasShim) {
            // If shim is at the beginning and we have consumers, it's shim->core (MM2S, processing_type=0)
            // If shim is at the end and we have producers, it's core->shim (S2MM, processing_type=2)
            Value firstTile = allTilesInPath.front();
            Value lastTile = allTilesInPath.back();

            bool hasConsumers = !consumerPorts.empty() || !consumerSymNames.empty();
            bool hasProducers = !producerPorts.empty() || !producerSymNames.empty();
            if (firstTile == shimTileValue && hasConsumers) {
                isShimToCore = true;  // MM2S: shim sends to cores
            } else if (lastTile == shimTileValue && hasProducers) {
                isShimToCore = false;  // S2MM: cores send to shim
            } else if (firstTile == shimTileValue) {
                isShimToCore = true;  // Default to MM2S if shim is first
            } else {
                isShimToCore = false;  // Default to S2MM if shim is last
            }
        }

        // Build producer and consumer core tile lists directly from the
        // create_path attributes using portSymToTilePoint.
        // This is reliable because portSymToTilePoint is populated by
        // DmaphopPortConversionPattern before ports are erased, and it
        // maps port symbol names to tile coordinates.
        // The hop-based tileHasProducer/tileHasConsumer approach is unreliable
        // because hop ops may be erased before this pattern runs.
        {
            std::set<Point> producerPointSet;
            for (const auto &symName : producerSymNames) {
                // First try direct port symbol lookup
                auto it = routingCtx.portSymToTilePoint.find(symName.str());
                if (it != routingCtx.portSymToTilePoint.end()) {
                    producerPointSet.insert(it->second);
                } else {
                    // Try resolving through producer indirection (sym_name -> tp -> port)
                    auto symRef = FlatSymbolRefAttr::get(op->getContext(), symName);
                    if (auto producerOp = SymbolTable::lookupNearestSymbolFrom<dmaphop::producer>(op, symRef)) {
                        auto recvPortName = producerOp.getTp();
                        auto it2 = routingCtx.portSymToTilePoint.find(recvPortName.str());
                        if (it2 != routingCtx.portSymToTilePoint.end()) {
                            producerPointSet.insert(it2->second);
                        }
                    }
                }
            }
            for (const auto &pt : coreTileList) {
                if (producerPointSet.count(pt)) {
                    producerCoreTiles.push_back(pt);
                }
            }
        }
        {
            std::set<Point> consumerPointSet;
            for (const auto &symName : consumerSymNames) {
                // First try direct port symbol lookup
                auto it = routingCtx.portSymToTilePoint.find(symName.str());
                if (it != routingCtx.portSymToTilePoint.end()) {
                    consumerPointSet.insert(it->second);
                } else {
                    // Try resolving through consumer indirection (sym_name -> from -> port)
                    auto symRef = FlatSymbolRefAttr::get(op->getContext(), symName);
                    if (auto consumerOp = SymbolTable::lookupNearestSymbolFrom<dmaphop::consumer>(op, symRef)) {
                        auto sendPortName = consumerOp.getFrom();
                        auto it2 = routingCtx.portSymToTilePoint.find(sendPortName.str());
                        if (it2 != routingCtx.portSymToTilePoint.end()) {
                            consumerPointSet.insert(it2->second);
                        }
                    }
                }
            }
            for (const auto &pt : coreTileList) {
                if (consumerPointSet.count(pt)) {
                    consumerCoreTiles.push_back(pt);
                }
            }
        }

        // Now call the routing logic similar to RoutingmovedatabyioConvert
        if (coreTileList.empty()) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Determine which tile to use for shim allocation
        Point firstTile = isShimToCore ? coreTileList[0] : coreTileList.back();
        DMADIRECTION dmadir = isShimToCore ? DMADIRECTION::MM2S : DMADIRECTION::S2MM;
        StreamType streamtype = isShimToCore ? StreamType::BROADCAST : StreamType::FORWARDONLY;
        // A control-packet flow is a shim->core (MM2S) delivery of config/register
        // writes. When the dmaphop path is marked control, terminate at each
        // target tile's CTRL port (packet mode, header preserved) instead of DMA.
        if (isShimToCore && op->hasAttrOfType<BoolAttr>("control") &&
            op->getAttrOfType<BoolAttr>("control").getValue()) {
            streamtype = StreamType::CONTROL;
        }

        // Try to find existing DataIO for the shim location
        // If shim was specified in dmaphop, we should look it up
        // Otherwise, create a new DataIO
        std::shared_ptr<DataIO> dio;
        int shimcol, dioid, shimchannel;
        Point allocatedShimPoint;
        
        if (hasShim) {
            // Shim tile was explicitly defined in dmaphop
            // Try to find existing DataIO for this shim column+channel
            auto rm = router_.getRM();
            shimcol = shimPoint.c;

            // Look up by the channel encoded in the dmaphop port op (direction_channel attr).
            // This allows reusing an already-allocated DataIO for the same shim column+channel,
            // while correctly assigning distinct ports to different rounds (ch=0, ch=1, ...).
            dio = rm->findDataIOByShimChannel(shimcol, shimDmaChannel, dmadir);

            if (!dio) {
                // No existing DataIO found, create a new one
                std::optional<TypeBasedTileLoc> dstcoreloc(TypeBasedTileLoc{TileType::Core, firstTile});
                std::ostringstream ostr;
                ostr << "dio" << ioIdx++;
                dio = router_.createDataIO(ostr.str(), dstcoreloc, dmadir);
            }
            
            shimcol = dio->colpos();
            allocatedShimPoint = {0, shimcol};
        } else {
            // No explicit shim - let router allocate one
            std::optional<TypeBasedTileLoc> dstcoreloc(TypeBasedTileLoc{TileType::Core, firstTile});
            std::ostringstream ostr;
            ostr << "dio" << ioIdx++;
            dio = router_.createDataIO(ostr.str(), dstcoreloc, dmadir);
            shimcol = dio->colpos();
            allocatedShimPoint = {0, shimcol};
        }
        
        dioid = dio->id();
        shimchannel = dio->channel();
        
        // Create IOShimTileCreate (like line 1014 or 1055)
        std::ostringstream commentStr;
        commentStr << "shim_dma_" << dioid;
        auto shimIoOp = rewriter.create<IOShimTileCreate>(
            loc, output, 
            0,  // row
            shimcol,  // col
            dioid,  // IOID
            commentStr.str(),  // comments
            static_cast<int>(dmadir),  // dmadirection
            shimchannel  // channelused
        );
        
        // Create routing path (like line 1030 or 1058)
        auto rpath = router_.createPath(dioid, coreTileList);
        
        if (!rpath || !(*rpath)) {
            rewriter.eraseOp(op);
            return success();
        }
        
        // Check if this is core-to-shim (S2MM) with producers
        // In this case, we need packet routing for gathering data from multiple producers
        bool isCoreToShim = !isShimToCore;
        std::optional<TileListPktRoutingNode> lastPkttilemap = std::nullopt;
        
        if (isCoreToShim && !producerCoreTiles.empty()) {
            // Use GatherPktRoutingPathCreate for core-to-shim packet routing (processing_type=2)
            // This creates packet-based routing for gathering data from producer tiles
            lastPkttilemap = GatherPktRoutingPathCreate(
                op,
                dioid,
                allocatedShimPoint,
                dio,
                rpath,
                producerCoreTiles,  // Tiles that produce data
                dsttiles,
                router_,
                rewriter
            );
            
        }

        // Extract consumer names from the path op's consumers array.
        // The consumers attr may be flat [@c0, @c1] or nested [[@c0, @c1]].
        SmallVector<std::string, 8> pathConsumerNames;
        if (auto consumersAttr = op->getAttrOfType<ArrayAttr>("consumers")) {
            for (auto attr : consumersAttr) {
                if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
                    pathConsumerNames.push_back(symRef.getValue().str());
                } else if (auto innerArr = dyn_cast<ArrayAttr>(attr)) {
                    for (auto innerAttr : innerArr) {
                        if (auto symRef = dyn_cast<FlatSymbolRefAttr>(innerAttr))
                            pathConsumerNames.push_back(symRef.getValue().str());
                    }
                }
            }
        }

        // Call ParseTheCCTRoutingPath to generate the routing connections.
        // Pass portSymToTilePoint and consumerDmaPortInfo so that GetSeqPath can
        // resolve consumer dma_port without relying on erased port/consumer ops.
        ParseTheCCTRoutingPath(op, lastPkttilemap, streamtype, dioid, allocatedShimPoint, dio, shimIoOp, rpath,
                               dsttiles, router_, rewriter, &routingCtx.portSymToTilePoint,
                               &routingCtx.consumerDmaPortInfo, &pathConsumerNames);

        // Erase the path operation
        rewriter.eraseOp(op);
        return success();
    }

private:
    RoutingTopology &router_;
    RoutingContext &routingCtx;
};

// Pattern to erase dmaphop operations that don't need conversion
template <typename OpType>
struct EraseOpPattern : public OpConversionPattern<OpType> {
    using OpConversionPattern<OpType>::OpConversionPattern;
    
    LogicalResult matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

void DmaphopToRoutinghwPass::runOnOperation() {
    auto& ctx = getContext();
    auto module = getOperation();
    ConversionTarget target(ctx);
    RewritePatternSet patterns(&ctx);
    
    // Create routing context
    RoutingContext routingCtx;

    // Pre-populate routingCtx by walking dmaphop.tile and dmaphop.port ops.
    // This must happen BEFORE applyPartialConversion because the conversion
    // framework may process ops in any order — port/tile erasure patterns
    // can fire before the path conversion pattern needs the mappings.
    module->walk([&](dmaphop::tile tileOp) {
        TileInfo info;
        info.tileOp = nullptr; // Will be replaced by routinghw op during conversion
        info.col = tileOp.getCol();
        info.row = tileOp.getRow();
        info.channel = 0;
        info.tileType = tileOp.getTiletype().str();
        routingCtx.tileMap[tileOp.getResult()] = info;
    });
    module->walk([&](dmaphop::port portOp) {
        Value tileValue = portOp.getTile();
        auto it = routingCtx.tileMap.find(tileValue);
        if (it != routingCtx.tileMap.end()) {
            routingCtx.portSymToTilePoint[portOp.getSymName().str()] =
                Point{static_cast<int>(it->second.row), static_cast<int>(it->second.col)};
        }
    });
    // Pre-populate consumer/producer dma_port info before conversion erases them
    module->walk([&](dmaphop::consumer consumerOp) {
        DmaPortInfo info;
        info.portSymName = consumerOp.getFrom().str();
        info.dmaPort = static_cast<int>(consumerOp.getDmaPort());
        routingCtx.consumerDmaPortInfo[consumerOp.getSymName().str()] = info;
    });
    module->walk([&](dmaphop::producer producerOp) {
        DmaPortInfo info;
        info.portSymName = producerOp.getTp().str();
        info.dmaPort = static_cast<int>(producerOp.getDmaPort());
        routingCtx.consumerDmaPortInfo[producerOp.getSymName().str()] = info;
    });

    // Define conversion target
    //target.addIllegalDialect<dmaphop::dmaphopdialect>();
    target.addLegalDialect<routinghw::RoutingHWDialect, func::FuncDialect, memref::MemRefDialect, 
                           routing::routingdialect, scf::SCFDialect, arith::ArithDialect>();
    
    // Mark routing::extract_data as illegal so it gets erased
    target.addIllegalOp<routing::extract_data>();
    target.addIllegalOp<routing::createscheduletensor>();
    target.addIllegalOp<routing::partitiontensor>();
    // Mark tensor::ExtractSliceOp as illegal so it gets erased
    target.addIllegalOp<tensor::ExtractSliceOp>();
    
    // Explicitly mark all other routing and SCF operations as legal to preserve them
    target.addDynamicallyLegalDialect<routing::routingdialect>(
        [](Operation *op) { return !isa<routing::extract_data>(op); }
    );
    
    // Explicitly preserve scf.execute_region and other SCF operations
    target.addLegalOp<scf::ExecuteRegionOp>();
    target.addLegalOp<scf::YieldOp>();
    
    // Add conversion patterns
    patterns.add<DmaphopTileConversionPattern>(&ctx, rtopology_, routingCtx);
    patterns.add<DmaphopPortConversionPattern>(&ctx, routingCtx);
    patterns.add<DmaphopPathConversionPattern>(&ctx, rtopology_, routingCtx);
    
    // Add patterns to erase operations that don't need direct conversion
    patterns.add<EraseOpPattern<dmaphop::create_hop>>(&ctx);
    patterns.add<EraseOpPattern<dmaphop::alloc_buffer>>(&ctx);
    patterns.add<EraseOpPattern<dmaphop::dealloc_buffer>>(&ctx);
    patterns.add<EraseOpPattern<dmaphop::push>>(&ctx);
    patterns.add<EraseOpPattern<dmaphop::pull>>(&ctx);
    patterns.add<EraseOpPattern<dmaphop::sync>>(&ctx);
    patterns.add<EraseOpPattern<dmaphop::consumer>>(&ctx);
    patterns.add<EraseOpPattern<dmaphop::producer>>(&ctx);
    patterns.add<EraseOpPattern<routing::extract_data>>(&ctx);
    patterns.add<EraseOpPattern<tensor::ExtractSliceOp>>(&ctx);
    patterns.add<EraseOpPattern<routing::createscheduletensor>>(&ctx);
    patterns.add<EraseOpPattern<routing::partitiontensor>>(&ctx);

    /*
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    module->walk([&](scf::ExecuteRegionOp exec) {
        //only deal with the routing_memo executeregionop
        if (!exec->getAttrOfType<StringAttr>("routing_memo")) {
            return;
        }
        exec->walk([&](routing::RoutingCreate routingcreate) {
            if (failed(applyPartialConversion(routingcreate, target, frozenPatterns ))) {
                llvm::outs() << "routing convert failed \n";
            }
        });
    });
    */
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
    
}

DmaphopToRoutinghwPass::DmaphopToRoutinghwPass(RoutingTopology& rtopology):rtopology_(rtopology) {
}

