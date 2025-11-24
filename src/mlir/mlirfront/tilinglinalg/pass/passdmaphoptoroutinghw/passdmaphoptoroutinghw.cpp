/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "passdmaphoptoroutinghw.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "routinghwmanager.h"
#include "routingmanager.h"
#include "routing/routingpath.h"
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

using namespace mlir;
using namespace dmaphop;
using namespace routinghw;

namespace {

int ioIdx = 0;
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
    
    //parse and set dma and slave master
    //create empty structure for each dstPoint
    int pkt_idx = 0;
    for (const auto& dstPoint : tilist) {
        pktswitchmap[dstPoint] = StreamPKTConnection{};
    }
    //set the local DMA pkt connection
    for (const auto& dstPoint : tilist) {
        pkt_idx++;
        int dmaportNum;
        PortDirection dmadirection = PortDirection::DMA;
        //get DMA port index
        if (!router_.occupyPointDirection(dstPoint,dmaportNum, dmadirection, true)) {
            llvm::outs() << "DMA occupy failed " << "\n";
            assert(0);
            return std::nullopt;
        }
        //set prev tile master port and dma port
        struct StreamPKTConnection& curtileconf = pktswitchmap[dstPoint];
        curtileconf.localDMAForwardPortIdx = dmaportNum;
        curtileconf.localDMAForwardPktID = pkt_idx;//fix me
        curtileconf.localDMAForwardPktType = 0;
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

        rewriter.create<routinghw::ConnectStreamPktSwitchPort>(
            op->getLoc(),                   // Operation location
            output,
            curTileOp.getResult(),                   // Tile to be configured
            rewriter.getStringAttr(PortDirectiontoString(value.SlaveReceiveForwardDirection)), // Direction of the port receiving the stream
            rewriter.getI32IntegerAttr((int)value.SlaveReceiveForwardDirectionPortIdx),     // Index of the receiving port
            rewriter.getI32IntegerAttr(value.SlaveReceivePktID),// Packet ID to expect
            rewriter.getI32IntegerAttr(value.SlaveReceivePktType),// Packet Type to expect
            rewriter.getStringAttr(PortDirectiontoString(PortDirection::DMA)),  // local DMA direction NONE means no DMA
            rewriter.getI32IntegerAttr(value.localDMAForwardPortIdx),  // Index of the local DMA port to send to
            rewriter.getI32IntegerAttr(value.localDMAForwardPktID ),    // Packet ID for the DMA transfer
            rewriter.getI32IntegerAttr(value.localDMAForwardPktType),  // Packet Type for the DMA transfer
            rewriter.getStringAttr(PortDirectiontoString(value.MasterSendToNextTileDirection)),     // No forwarding: empty master direction
            rewriter.getI32IntegerAttr((int)(value.MasterSendToNextTileDirectionPortIdx)) // No forwarding: port index 0
        );
    }
    ret.tile = tilist.back();
    ret.pktconn = pktswitchmap[ret.tile];
    ret.tileOp = dsttiles[ret.tile];
    // connect pkt merge/data gather into shim tile
    return std::make_optional<TileListPktRoutingNode>(ret);
}

std::optional<TileListRoutingMap> GetSeqPath(
                         std::optional<std::shared_ptr<const RoutingPath>> rpath,
                         std::shared_ptr<DataIO> dio,
                         std::unordered_map<Point, Operation*, Point::Hash> dsttiles,
                         StreamType streamtype,// 0 no dma, 1 dma receive
                         std::optional<TileListPktRoutingNode> lastPkttilemap,
                         RoutingTopology& router_,
                         ConversionPatternRewriter& rewriter) {
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
                int portNum;
                PortDirection slaveDirOnNext, masterDirOnCurrent;
                if (!router_.occupyLink(currentPoint, nextPoint, dioid, portNum, masterDirOnCurrent, slaveDirOnNext)) {
                    llvm::report_fatal_error("Failed to occupy link in routing topology.");
                }
                
                connectionData[currentPoint].MasterSendToNextTileDirection = masterDirOnCurrent;
                connectionData[currentPoint].MasterSendToNextTileDirectionPortIdx = portNum;
                connectionData[nextPoint].SlaveReceiveForwardDirection = slaveDirOnNext;
                connectionData[nextPoint].SlaveReceiveForwardDirectionPortIdx = portNum;
                //set next master into None
                connectionData[nextPoint].MasterSendToNextTileDirection = PortDirection::NONE;
                
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

    // 1b. Populate DMA connection information
    auto rm = router_.getRM();
    for (const auto& p : orderedPathPoints) {
        connectionData[p].localDMAForwardDirection = PortDirection::NONE;
        if (rm->getrsc()->tileType(p.r, p.c) == TileType::Core 
            && StreamType::BROADCAST == streamtype
            && dsttiles.find(p) != dsttiles.end()) {
            if (auto portnumptr = rm->tile(p.r, p.c).occupyport(IOType::TileDMA, PortDirection::DMA, -1)) {
                connectionData[p].localDMAForwardDirection = PortDirection::DMA;
                connectionData[p].localDMAForwardPortIdx = *portnumptr;
            }
        }
    }

    return std::make_optional<TileListRoutingMap>(troutingmap);
}

void ParseTheCCTRoutingPath(Operation* op,
                         std::optional<TileListPktRoutingNode> lastPkttilemap,
                         StreamType streamtype,// 0 normal, 1 broadcast
                         uint32_t dioid,
                         Point shimpoint,
                         std::shared_ptr<DataIO> dio,
                         IOShimTileCreate shimio,
                         std::optional<std::shared_ptr<const RoutingPath>> rpath,
                         std::unordered_map<Point, Operation*, Point::Hash> dsttiles,
                         RoutingTopology& router_,
                         ConversionPatternRewriter& rewriter) {

    if (!rpath || !(*rpath)) {
        return; // No path to process
    }

    auto loc = op->getLoc();
    auto outputType = rewriter.getI32Type();
    // --- Phase 1: Build connection map AND an ordered list of points ---
    auto troutingmap = GetSeqPath(rpath,dio,dsttiles, streamtype/* 0 normal no dma, 1 broadcast dma receive*/,lastPkttilemap,router_,rewriter);
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
            if (dio->type() == IOType::Input) {
                rewriter.create<EnableExtToAieShimPort>(loc, outputType, shimio.getResult(), inputDirStr, inputPortIdx);
            } else {
                rewriter.create<EnableAieToExtShimPort>(loc, outputType, shimio.getResult(), inputDirStr, inputPortIdx);
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
                rewriter.create<routinghw::ConnectStreamPktSwitchPort>(
                        loc,                   // Operation location
                        output,
                        tileOp.getResult(),                   // Tile to be configured
                        rewriter.getStringAttr(PortDirectiontoString(PortDirection::NONE)), // Direction of the port receiving the stream
                        rewriter.getI32IntegerAttr(0),     // Index of the receiving port
                        rewriter.getI32IntegerAttr(0),// Packet ID to expect
                        rewriter.getI32IntegerAttr(0),// Packet Type to expect
                        rewriter.getStringAttr(PortDirectiontoString(PortDirection::NONE)),
                        rewriter.getI32IntegerAttr(0),  // Index of the local DMA port to send to
                        rewriter.getI32IntegerAttr(0),    // Packet ID for the DMA transfer
                        rewriter.getI32IntegerAttr(0),  // Packet Type for the DMA transfer
                        rewriter.getStringAttr(PortDirectiontoString(conn.MasterSendToNextTileDirection)),     // No forwarding: empty master direction
                        rewriter.getI32IntegerAttr((int)(conn.MasterSendToNextTileDirectionPortIdx)) // No forwarding: port index 0
                );
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

            // Create connection to the local DMA (if this is a destination core tile)
            if (conn.localDMAForwardDirection != PortDirection::NONE) {
                rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, currentTileOp.getResult(),
                    inputDirStr, inputPortIdx,
                    "DMA", conn.localDMAForwardPortIdx);
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
        llvm::DenseSet<Value> producerPorts;
        if (auto producersAttr = op->getAttrOfType<ArrayAttr>("producers")) {
            for (auto symRef : producersAttr) {
                if (auto symRefAttr = dyn_cast<FlatSymbolRefAttr>(symRef)) {
                    // Find the port with this symbol name in the parent function
                    auto parentFunc = op->getParentOfType<func::FuncOp>();
                    if (parentFunc) {
                        parentFunc.walk([&](dmaphop::port portOp) {
                            auto symName = portOp.getSymName();
                            if (!symName.empty() && symName == symRefAttr.getValue()) {
                                producerPorts.insert(portOp.getResult());
                            }
                        });
                    }
                }
            }
        }
        
        // Get consumers attribute to determine which tiles should receive from DMA (MM2S)
        // Consumers are core tiles that receive data FROM shim/memory TO core (Push direction)
        llvm::DenseSet<Value> consumerPorts;
        if (auto consumersAttr = op->getAttrOfType<ArrayAttr>("consumers")) {
            for (auto symRef : consumersAttr) {
                if (auto symRefAttr = dyn_cast<FlatSymbolRefAttr>(symRef)) {
                    // Find the port with this symbol name in the parent function
                    auto parentFunc = op->getParentOfType<func::FuncOp>();
                    if (parentFunc) {
                        parentFunc.walk([&](dmaphop::port portOp) {
                            auto symName = portOp.getSymName();
                            if (!symName.empty() && symName == symRefAttr.getValue()) {
                                consumerPorts.insert(portOp.getResult());
                            }
                        });
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
                            } else if (tileInfoIt->second.tileType == "core") {
                                Point pt{tileInfoIt->second.row, tileInfoIt->second.col};
                                coreTileList.push_back(pt);
                                dsttiles[pt] = tileInfoIt->second.tileOp;
                            }
                        }
                    }
                    // Mark this tile as having a producer if source port is a producer
                    if (producerPorts.contains(srcPort)) {
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
                            } else if (tileInfoIt->second.tileType == "core") {
                                Point pt{tileInfoIt->second.row, tileInfoIt->second.col};
                                coreTileList.push_back(pt);
                                dsttiles[pt] = tileInfoIt->second.tileOp;
                            }
                        }
                    }
                    // Mark this tile as having a consumer if destination port is a consumer
                    if (consumerPorts.contains(dstPort)) {
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
            
            if (firstTile == shimTileValue && !consumerPorts.empty()) {
                isShimToCore = true;  // MM2S: shim sends to cores
            } else if (lastTile == shimTileValue && !producerPorts.empty()) {
                isShimToCore = false;  // S2MM: cores send to shim
            } else if (firstTile == shimTileValue) {
                isShimToCore = true;  // Default to MM2S if shim is first
            } else {
                isShimToCore = false;  // Default to S2MM if shim is last
            }
        }
        
        // Build consumer and producer core tile lists
        for (const auto& pt : coreTileList) {
            // Find the tile Value for this Point
            for (const auto& tileKV : routingCtx.tileMap) {
                if (tileKV.second.row == pt.r && tileKV.second.col == pt.c) {
                    if (tileHasConsumer.count(tileKV.first) && tileHasConsumer[tileKV.first]) {
                        consumerCoreTiles.push_back(pt);
                    }
                    if (tileHasProducer.count(tileKV.first) && tileHasProducer[tileKV.first]) {
                        producerCoreTiles.push_back(pt);
                    }
                    break;
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
        
        // Try to find existing DataIO for the shim location
        // If shim was specified in dmaphop, we should look it up
        // Otherwise, create a new DataIO
        std::shared_ptr<DataIO> dio;
        int shimcol, dioid, shimchannel;
        Point allocatedShimPoint;
        
        if (hasShim) {
            // Shim tile was explicitly defined in dmaphop
            // Try to find existing DataIO for this shim column
            // For now, we assume channel 0, but this should be determined from dmaphop if available
            auto rm = router_.getRM();
            shimcol = shimPoint.c;
            
            // Try channel 0 first
            dio = rm->findDataIOByShimChannel(shimcol, 0);
            
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
        
        // Call ParseTheCCTRoutingPath to generate the routing connections
        ParseTheCCTRoutingPath(op, lastPkttilemap, streamtype, dioid, allocatedShimPoint, 
                              dio, shimIoOp, rpath, dsttiles, router_, rewriter);
        
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
    ConversionTarget target(ctx);
    RewritePatternSet patterns(&ctx);
    
    // Create routing context
    RoutingContext routingCtx;

    // Define conversion target
    //target.addIllegalDialect<dmaphop::dmaphopdialect>();
    target.addLegalDialect<routinghw::RoutingHWDialect, func::FuncDialect, memref::MemRefDialect, 
                           routing::routingdialect, scf::SCFDialect, arith::ArithDialect>();
    
    // Mark routing::extract_data as illegal so it gets erased
    target.addIllegalOp<routing::extract_data>();
    target.addIllegalOp<routing::createdummytensor>();
    target.addIllegalOp<routing::partitiontensor>();
    
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
    patterns.add<EraseOpPattern<routing::extract_data>>(&ctx);  // Erase routing::extract_data
    patterns.add<EraseOpPattern<routing::createdummytensor>>(&ctx);
    patterns.add<EraseOpPattern<routing::partitiontensor>>(&ctx);  // Erase routing::partitiontensor
    
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

DmaphopToRoutinghwPass::DmaphopToRoutinghwPass(RoutingTopology& rtopology):rtopology_(rtopology) {
}

