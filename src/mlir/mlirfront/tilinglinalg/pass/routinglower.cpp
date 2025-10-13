/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "routinglower.h"
#include "routing/routingpath.h"
#include <sstream>
int ioIdx = 0;
//connectpktstreamswitchport

std::optional<TileListPktRoutingNode> GatherPktRoutingPathCreate(Operation* op,
                             uint32_t dioid,
                             Point shimpoint,
                             std::shared_ptr<DataIO>  dio,
                             TileArrayHandleCreate tilecreatehandle, 
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
    std::optional<TypeBasedTileLoc> dstcoreloc(TypeBasedTileLoc{TileType::Core, pktmergetile[0]});
    std::ostringstream ostr;
    ostr << "dio" << ioIdx++;
    auto diogather = router_.createDataIO(ostr.str(), dstcoreloc, DMADIRECTION::MM2S);
    int diogetherid = diogather->id();
    auto rpath2 = router_.createPath(diogetherid, pktmergetile);
    if (!rpath2) {
        return std::nullopt;
    }
    
    std::unordered_map<Point, std::vector<int>, Point::Hash> tileMasterPortMapping;
    std::unordered_map<Point, Operation*, Point::Hash> pathtiles;
    /*
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
}; */
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

    return std::make_optional<TileListRoutingMap>(troutingmap);
}

void ParseTheCCTRoutingPath(Operation* op,
                         std::optional<TileListPktRoutingNode> lastPkttilemap,
                         StreamType streamtype,// 0 normal, 1 broadcast
                         uint32_t dioid,
                         Point shimpoint,
                         std::shared_ptr<DataIO> dio,
                         IOShimTileCreate shimio,
                         TileArrayHandleCreate tilecreatehandle,
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
    // --- Phase 2: Generate MLIR ops using the ordered list ---
    
    // 2a. Create all tile operations first, IN ORDER
    std::unordered_map<Point, Operation*, Point::Hash> allTileOps = dsttiles;
    for (const Point& p : orderedPathPoints) {
        if (allTileOps.find(p) == allTileOps.end()) {
            allTileOps[p] = rewriter.create<routinghw::TileCreate>(
                loc, outputType, tilecreatehandle.getResult(), p.r, p.c, "tile in path");
        }
    }

    // 2b. Create connections IN ORDER by iterating through the ordered vector
    for (const Point& point : orderedPathPoints) {
        // Look up the connection info from our map
        auto it = connectionData.find(point);
        if (it == connectionData.end()) continue; // This point might not have connections (e.g., an un-routed destination)
        
        const StreamCCTConnection& conn = it->second;
        auto currentTileOp = dyn_cast<routinghw::TileCreate>(allTileOps.at(point));
        
        // Ensure the tile has an input port to connect from
        if (conn.SlaveReceiveForwardDirection == PortDirection::NONE) {
            if (lastPkttilemap && lastPkttilemap->tile == point) {
                auto output = rewriter.getI32Type();
                auto tileOp = dyn_cast<routinghw::TileCreate>((Operation* )lastPkttilemap->tileOp);
                ///*
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
                );//*/
            }
            
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
       // /*
        // Create connection to the next tile in the path
        if (conn.MasterSendToNextTileDirection != PortDirection::NONE) {
            if (point == shimpoint) {
                 rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, shimio.getResult(),
                    inputDirStr, inputPortIdx,
                    PortDirectiontoString(conn.MasterSendToNextTileDirection), conn.MasterSendToNextTileDirectionPortIdx);
            } else {
                rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, currentTileOp.getResult(),
                    inputDirStr, inputPortIdx,
                    PortDirectiontoString(conn.MasterSendToNextTileDirection), conn.MasterSendToNextTileDirectionPortIdx);
            }
        }

        // Create connection to the local DMA
        if (conn.localDMAForwardDirection != PortDirection::NONE) {
            rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, currentTileOp.getResult(),
                    inputDirStr, inputPortIdx,
                    "DMA", conn.localDMAForwardPortIdx);
        }
           // */
    }
}

/*
void ParseTheRoutingPath(Operation* op,
                             uint32_t dioid,
                             Point shimpoint,
                             std::shared_ptr<DataIO>  dio,
                             TileArrayHandleCreate tilecreatehandle, 
                             std::optional<std::shared_ptr<const RoutingPath>> rpath, 
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
    std::unordered_map<Point, std::vector<int>, Point::Hash> tileMasterPortMapping;
    std::unordered_map<Point, Operation*, Point::Hash> pathtiles;

    if (rpath && *rpath) {
        auto output = rewriter.getI32Type();
        auto tree = (*rpath)->multipaths();
        ///* build the stream switch
        std::optional<Point> prev_optional_point = std::nullopt, prev_prev_optional_point = std::nullopt;
        for (size_t i = 0; i < tree.branches.size(); ++i) {
                std::cout << "Branch to (" << tree.dsts[i].r
                        << "," << tree.dsts[i].c << "): ";
                for (auto p : tree.branches[i]) {
                    std::cout << "(" << p.r << "," << p.c << ") ";
                    if (dsttiles.count(p) == 0 && pathtiles.count(p) == 0) {
                        auto tile1 = rewriter.create<routinghw::TileCreate>(op->getLoc(), output, tilecreatehandle.getResult(),p.r, p.c, "tile reserved in path");
                        pathtiles[p] = tile1;
                    }
                }

                //if this branch is the last branch we need to deal with the last item, then add a dump node as we only process the previous on of current
                if (i == tree.branches.size() - 1) {
                    tree.branches[i].push_back(tree.branches[i].back());
                }
                int len = tree.branches[i].size();
                //as the stream switch connect need to find the matched master (previous tile) slave (current tile) port, the current process point
                //is the previous point which already did tile occupy, then we can have the master port information
                for (int j = 0; j < len; j ++) {
                    auto currentpoint = (prev_optional_point == std::nullopt ? tree.branches[i][j] : *prev_optional_point);
                    auto nextpoint = tree.branches[i][j];
                    //if prev point is same with nextpoint at branch beginning by pass
                    if (j ==0 && prev_optional_point && *prev_optional_point == nextpoint) continue;
                    mlir::Operation* currenttile, *curtile;
                    if (dsttiles.count(currentpoint)) {
                        currenttile = dsttiles[currentpoint];
                    } else {
                        currenttile = pathtiles[currentpoint];
                    }
                    //-------get master and slave port----------------
                    int portNum=0;
                    PortDirection portdirectionPrevSlave, portdirectionCurMaster;
                    // the occupy operation will reserve pevious tile slave port and destination tile master port,
                    // we will connect the prevous tile master port into slave port, and add cur tile port into map
                    if (prev_optional_point) {
                        // when next == current, the next is dumpy point
                        if (currentpoint != nextpoint) {
                            if (!router_.occupyLink(currentpoint, nextpoint, dioid, portNum, portdirectionPrevSlave, portdirectionCurMaster)) {
                                llvm::outs() << "link occupy failed " << "\n";
                                assert(0);
                            }
                            // check if this currentpoint is the start shim port
                            if (shimpoint == currentpoint) {
                                
                            }
                            // storage cur tile infor
                            tileMasterPortMapping[nextpoint]={(int)portdirectionCurMaster, portNum, 0};
                        }
                        //get resource manager
                        auto rm = router_.getRM();
                        // get previous tile master port informaton
                        auto curop = dyn_cast<routinghw::TileCreate>(currenttile);
                        if (tileMasterPortMapping.find(currentpoint) != tileMasterPortMapping.end()) {
                            auto prevportinfo = tileMasterPortMapping[currentpoint];
                            auto portprevmaster = PortDirectiontoString((PortDirection)prevportinfo[0]);
                            auto portdirectionPrevSlaveStr = PortDirectiontoString(portdirectionPrevSlave);
                            auto portprevidx = prevportinfo[1];
                            rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),portprevmaster, portprevidx, portdirectionPrevSlaveStr, portNum);
                            //add to dma logic
                            //
                            auto rowcol = getrowcol(curop);
                            if (rm->getrsc()->tileType(rowcol[0], rowcol[1]) == TileType::Core) {
                               if (auto portnumptr = rm->tile(rowcol[0],rowcol[1]).occupyport(IOType::TileDMA, PortDirection::DMA, -1)) {
                                   rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),portprevmaster, portprevidx, "DMA", *portnumptr);                                
                               }
                            }
                            //
                        } else {
                            //no master port finding means this is the inital shim port get the master information from io
                            //io.getmasterportinfo
                            //when input shim io
                            if (dio->dmadir() == DMADIRECTION::MM2S) {
                                PortDirection shimportdir = PortDirection::South;
                                int shimportnum = 3;
                                if (auto shimportinfo = dio->getshimport()) {
                                    shimportdir=shimportinfo->dir_;
                                    shimportnum=shimportinfo->portnum_;
                                }
                                auto shimportdirstr = PortDirectiontoString(shimportdir);
                                auto portdirectionPrevSlaveStr = PortDirectiontoString(portdirectionPrevSlave);
                                if (dio->type() == IOType::Input) {
                                    rewriter.create<EnableExtToAieShimPort>(op->getLoc(), output, curop.getResult(),shimportdirstr, shimportnum);
                                } else {
                                    rewriter.create<EnableAieToExtShimPort>(op->getLoc(), output, curop.getResult(), shimportdirstr, shimportnum);
                                }
                                rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),shimportdirstr, shimportnum, portdirectionPrevSlaveStr, portNum);
                                llvm::outs() << "the logic wrong \n";
                            }
                        }
                    } else {

                    }
                    //----------get master and slave port end
                    prev_prev_optional_point = prev_optional_point;
                    prev_optional_point = std::make_optional(nextpoint);
                }
                std::cout << "\n";
        }
    }
}
*/

struct indexcastconvert : public ConversionPattern {
    explicit indexcastconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(arith::IndexCastOp::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

struct arithconstantconvert : public ConversionPattern {
    explicit arithconstantconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(arith::ConstantOp::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

struct createhwmeshconvert : public ConversionPattern {
    explicit createhwmeshconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::createhwmesh::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {   
        
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

//createdummytensor
struct createdummytensorconvert : public ConversionPattern {
    explicit createdummytensorconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::createdummytensor::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};
//partitiontensor
struct partitiontensorrconvert : public ConversionPattern {
    explicit partitiontensorrconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::partitiontensor::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

//partitionmesh
struct partitionmeshconvert : public ConversionPattern {
    explicit partitionmeshconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::partitionmesh::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {  
        //auto parent = op->getParentOfType<routing::RoutingCreate>();
        //if (!parent) {
        //    return rewriter.notifyMatchFailure(op, "not inside RoutingCreateOp");  
        //}
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

//routingcreatedataioconvert

struct routingcreatedataioconvert : public ConversionPattern {
    explicit routingcreatedataioconvert(MLIRContext * ctx, LLVMTypeConverter &converter, RoutingTopology & router):
        ConversionPattern(routing::createdataio::getOperationName(),1, ctx), typeconverter(converter), router_(router) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

//extract_data
struct extract_dataconvert : public ConversionPattern {
    explicit extract_dataconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::extract_data::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

//routinggatherout
struct routinggatheroutconvert : public ConversionPattern {
    explicit routinggatheroutconvert(MLIRContext * ctx, LLVMTypeConverter &converter, RoutingTopology & router):
        ConversionPattern(routing::routinggatherout::getOperationName(),1, ctx), typeconverter(converter), router_(router) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        //TODO create gather/pktmerge logic
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

//extract_tiles
struct extract_tilesconvert : public ConversionPattern {
    explicit extract_tilesconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::extract_tiles::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};
struct routingcreatetilearrayconvert : public ConversionPattern {
    explicit routingcreatetilearrayconvert(MLIRContext * ctx, LLVMTypeConverter &converter, RoutingTopology & router):
        ConversionPattern(routing::createtilearrayOp::getOperationName(),1, ctx), typeconverter(converter), router_(router) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {
            rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};
/*
struct routingcreatebroadcastconvert : public ConversionPattern {
    explicit routingcreatebroadcastconvert(MLIRContext * ctx, LLVMTypeConverter &converter, RoutingTopology & router):
        ConversionPattern(routing::creatbroadcast::getOperationName(),1, ctx), typeconverter(converter), router_(router) {

        }
    void createSwitchStream(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) {

    }

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {
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
       
        auto createioop = operands[0].getDefiningOp<routing::createdataio>();
        auto tilearrayOp = operands[1].getDefiningOp<routing::createtilearrayOp>();
        if (!createioop || !tilearrayOp) return success();
        llvm::outs() << operands[0].getDefiningOp()->getName().getDialectNamespace() << "\n";
        // create dataio
        auto attr = createioop.getIotypeAttrName();
        
        std::ostringstream ostr;
        ostr << "dio" << ioIdx++;
        
        auto dio = router_.createDataIO(ostr.str());
        auto ctx = getContext();
        auto output = rewriter.getI32Type();
  
        int shimcol = dio->colpos();
        int dioid = dio->id();
        Point shimpoint= {0, shimcol};
        auto routingshimio = rewriter.create<IOShimTileCreate>(op->getLoc(), output, 0, shimcol, dioid, ostr.str(), 0, 0);
        //create arraytile
        //auto output = rewriter.getI32Type();
        int cols = 0, rows=0;
        auto tilearrayoperands =  tilearrayOp.getOperands();
        if (auto indexOp = tilearrayoperands[0].getDefiningOp<mlir::arith::IndexCastOp>()) {
            Value input = indexOp.getIn();
            if (auto constOp = input.getDefiningOp<mlir::arith::ConstantOp>()) {
                if (auto intAttr = constOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
                    rows = intAttr.getInt();  // Success!
                }
            }
        }

        if (auto indexOp = tilearrayoperands[1].getDefiningOp<mlir::arith::IndexCastOp>()) {
            Value input = indexOp.getIn();
            if (auto constOp = input.getDefiningOp<mlir::arith::ConstantOp>()) {
                if (auto intAttr = constOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
                    cols = intAttr.getInt();  // Success!
                }
            }
        }
  
        //get the number of tiles
        std::vector<Point> allocatedTiles = router_.ReserveTiles(cols * rows,dio->id());
        //the current tile master port is reserved by previous tile connect occupy operation, as only the current tile
        //same mapped port is availabe then we can say the link is available
        //the vector<int> is [Master port direction, Master port index, type]
        std::unordered_map<Point, std::vector<int>, Point::Hash> tileMasterPortMapping;
        //
        std::unordered_map<Point, Operation*, Point::Hash> dsttiles, pathtiles;
        auto tilecreatehandle = rewriter.create<TileArrayHandleCreate>(op->getLoc(), output, "array handle");
        for(auto x: allocatedTiles) {
            auto tile1 = rewriter.create<routinghw::TileCreate>(op->getLoc(), output, tilecreatehandle.getResult(),x.r, x.c, "tile reserved");
            dsttiles[{x.r, x.c}] = tile1;
        }
        std::vector<Point> routerdsttiles;
        for(auto x:dsttiles) {
            routerdsttiles.push_back(x.first);
        }
        //----------start create path--------stream switch-----------
        auto rpath = router_.createPath(dioid, routerdsttiles);
        if (rpath && *rpath) {
            auto tree = (*rpath)->multipaths();
            ///  build the stream switch
            std::optional<Point> prev_optional_point = std::nullopt, prev_prev_optional_point = std::nullopt;
            for (size_t i = 0; i < tree.branches.size(); ++i) {
                std::cout << "Branch to (" << tree.dsts[i].r
                        << "," << tree.dsts[i].c << "): ";
                for (auto p : tree.branches[i]) {
                    std::cout << "(" << p.r << "," << p.c << ") ";
                    if (dsttiles.count(p) == 0 && pathtiles.count(p) == 0) {
                        auto tile1 = rewriter.create<routinghw::TileCreate>(op->getLoc(), output, tilecreatehandle.getResult(),p.r, p.c, "tile reserved in path");
                        pathtiles[p] = tile1;
                    }
                }

                //if this branch is the last branch we need to deal with the last item, then add a dump node as we only process the previous on of current
                if (i == tree.branches.size() - 1) {
                    tree.branches[i].push_back(tree.branches[i].back());
                }
                int len = tree.branches[i].size();
                //as the stream switch connect need to find the matched master (previous tile) slave (current tile) port, the current process point
                //is the previous point which already did tile occupy, then we can have the master port information
                for (int j = 0; j < len; j ++) {
                    auto currentpoint = (prev_optional_point == std::nullopt ? tree.branches[i][j] : *prev_optional_point);
                    auto nextpoint = tree.branches[i][j];
                    //if prev point is same with nextpoint at branch beginning by pass
                    if (j ==0 && prev_optional_point && *prev_optional_point == nextpoint) continue;
                    mlir::Operation* currenttile, *curtile;
                    if (dsttiles.count(currentpoint)) {
                        currenttile = dsttiles[currentpoint];
                    } else {
                        currenttile = pathtiles[currentpoint];
                    }
                    //if (dsttiles.count(nextpoint)) {
                    //    curtile = dsttiles[nextpoint];
                    //} else {
                    //    curtile = pathtiles[nextpoint];
                    //}
                    //-------get master and slave port----------------
                    int portNum=0;
                    PortDirection portdirectionPrevSlave, portdirectionCurMaster;
                    // the occupy operation will reserve pevious tile slave port and destination tile master port,
                    // we will connect the prevous tile master port into slave port, and add cur tile port into map
                    if (prev_optional_point) {
                        // when next == current, the next is dumpy point
                        if (currentpoint != nextpoint) {
                            router_.occupyLink(currentpoint, nextpoint, dioid, portNum, portdirectionPrevSlave, portdirectionCurMaster);
                            // check if this currentpoint is the start shim port
                            if (shimpoint == currentpoint) {
                                
                            }
                            // storage cur tile infor
                            tileMasterPortMapping[nextpoint]={(int)portdirectionCurMaster, portNum, 0};
                        }
                        //get resource manager
                        auto rm = router_.getRM();
                        // get previous tile master port informaton
                        auto curop = dyn_cast<routinghw::TileCreate>(currenttile);
                        if (tileMasterPortMapping.find(currentpoint) != tileMasterPortMapping.end()) {
                            auto CurMasterPortinfo = tileMasterPortMapping[currentpoint];
                            auto CurMasterPortDirection = PortDirectiontoString((PortDirection)CurMasterPortinfo[0]);
                            auto portdirectionPrevSlaveStr = PortDirectiontoString(portdirectionPrevSlave);
                            auto portCurMasteridx = CurMasterPortinfo[1];
                            rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),CurMasterPortDirection, portCurMasteridx, portdirectionPrevSlaveStr, portNum);
                            //add to dma logic
                            ///*
                            auto rowcol = getrowcol(curop);
                            if (rm->getrsc()->tileType(rowcol[0], rowcol[1]) == TileType::Core) {
                               if (auto portnumptr = rm->tile(rowcol[0],rowcol[1]).occupyport(IOType::TileDMA, PortDirection::DMA, -1)) {
                                   rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),CurMasterPortDirection, portCurMasteridx, "DMA", *portnumptr);                                
                               }
                            }
                            // 
                        } else {
                            //no master port finding means this is the inital shim port get the master information from io
                            //io.getmasterportinfo
                            PortDirection shimportdir = PortDirection::South;
                            int shimportnum = 3;
                            if (auto shimportinfo = dio->getshimport()) {
                                shimportdir=shimportinfo->dir_;
                                shimportnum=shimportinfo->portnum_;
                            }
                            auto shimportdirstr = PortDirectiontoString(shimportdir);
                            auto portdirectionPrevSlaveStr = PortDirectiontoString(portdirectionPrevSlave);
                            if (dio->type() == IOType::Input) {
                                rewriter.create<EnableExtToAieShimPort>(op->getLoc(), output, curop.getResult(),shimportdirstr, shimportnum);
                            } else {
                                rewriter.create<EnableAieToExtShimPort>(op->getLoc(), output, curop.getResult(), shimportdirstr, shimportnum);
                            }
                            rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),shimportdirstr, shimportnum, portdirectionPrevSlaveStr, portNum);
                            llvm::outs() << "the logic wrong \n";
                        }
                    } else {

                    }
                    //----------get master and slave port end
                    prev_prev_optional_point = prev_optional_point;
                    prev_optional_point = std::make_optional(nextpoint);
                }
                std::cout << "\n";
            }
        }
        //----------------end create stream switch-------------
        //rewriter.create<routinghw::ConnecIOToTileArray>(op->getLoc(), output, routingshimio.getResult(),tilecreatehandle.getResult());
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};
*/
//RoutingCreate
struct RoutingCreateConvert : public ConversionPattern {
    explicit RoutingCreateConvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::RoutingCreate::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

//RoutingCreate
struct RoutingcreatehwiowithtargetConvert : public ConversionPattern {
    explicit RoutingcreatehwiowithtargetConvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::createhwiowithtarget::getOperationName(),1, ctx), typeconverter(converter) {

        }
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

//RoutingCreate
struct RoutingmovedatabyioConvert : public ConversionPattern {
    explicit RoutingmovedatabyioConvert(MLIRContext * ctx, LLVMTypeConverter &converter, RoutingTopology & router):
        ConversionPattern(routing::movedatabyio::getOperationName(),1, ctx), typeconverter(converter), router_(router) {

        }

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        //function to get blockarg constant
        auto getRoutingCreateConsArgu = [&] (Value operand) -> int {
            if (auto barg = dyn_cast<BlockArgument>(operand)) {
                Operation *parentOp = barg.getOwner()->getParentOp();
                if (auto create = dyn_cast<routing::RoutingCreate>(parentOp)) {
                    unsigned idx = barg.getArgNumber();
                    Value incoming = create->getOperand(idx);
                    IntegerAttr intAttr;
                    if (matchPattern(incoming, m_Constant(&intAttr))) {
                        auto concrete = intAttr.getInt();        // -> 0
                        return concrete;
                    }
                }
            };
            return 0;
        };
        int round_idx = 0;
        // get all the op
        routing::createhwmesh createhwmesh;
        routing::createdummytensor createdummytensor;
        routing::partitiontensor partitiontensor;
        routing::partitionmesh partitionmesh;
        routing::extract_data extract_data;
        routing::routinggatherout gatherout_pktmerge;
        routing::extract_tiles extract_tiles;
        int processing_type = 0;// 0 broadcast //1 tbd //2 gather an packet merge
        routing::createhwiowithtarget createhwiowithtarget = operands[1].getDefiningOp<routing::createhwiowithtarget>();
        if (!(extract_data = operands[0].getDefiningOp<routing::extract_data>())) {
            if (gatherout_pktmerge = operands[0].getDefiningOp<routing::routinggatherout>()){
                extract_data = gatherout_pktmerge->getOperands()[1].getDefiningOp<routing::extract_data>();
                processing_type = 2;
            }
        }

        if (extract_data) {
             auto edata_operands = extract_data->getOperands();
            round_idx = getRoutingCreateConsArgu(edata_operands[1]);
            llvm::outs() << "slice data round_idx=" << round_idx << "\n";
            if (!(partitiontensor = extract_data->getOperands()[0].getDefiningOp<routing::partitiontensor>())) {
                llvm::outs() << " extract_tiles not found return" << "\n";
                return failure();
            }
        } else {
            return failure();
        }

        if (!(createhwiowithtarget = operands[1].getDefiningOp<routing::createhwiowithtarget>())) {
            llvm::outs() << " createhwiowithtarget not found return" << "\n";
            return failure();
        }
        if (!(extract_tiles = createhwiowithtarget->getOperands()[0].getDefiningOp<routing::extract_tiles>())) {
            llvm::outs() << " extract_tiles not found return" << "\n";
            return failure();
        }
        if (!(partitionmesh = extract_tiles->getOperands()[0].getDefiningOp<routing::partitionmesh>())) {
            llvm::outs() << " partitionmesh not found return" << "\n";
            return failure();
        }
        if (!(createhwmesh = partitionmesh->getOperands()[0].getDefiningOp<routing::createhwmesh>())) {
            llvm::outs() << " createhwmesh not found return" << "\n";
            return failure();
        }
        if (!(createdummytensor = partitiontensor->getOperands()[0].getDefiningOp<routing::createdummytensor>())) {
            llvm::outs() << " createdummytensor not found return" << "\n";
            return failure();
        }

        int col = createhwmesh.getCol();
        int row = createhwmesh.getRow();
        auto shape = createdummytensor.getShape();

        int tensor_x = shape[0].cast<mlir::IntegerAttr>().getInt();
        int tensor_y = shape[1].cast<mlir::IntegerAttr>().getInt();

        llvm::StringRef split_axis = partitionmesh.getSplitaxis();
        llvm::StringRef hw_axis_owner = partitiontensor.getHwAxisOwner();
        llvm::StringRef replicate_on = partitiontensor.getReplicateOn();
        llvm::StringRef single_tile_owner = partitiontensor.getSingleTileOwner();

        llvm::outs() << "col =" << col << " row=" << row  << " tensor_x=" << tensor_x << " tensor_y=" << tensor_y << "\n";
        llvm::outs() << "split_axis =" << split_axis << " hw_axis_owner=" << hw_axis_owner  << "\n";
        llvm::outs() << "replicate_on=" << replicate_on << " single_tile_owner=" << single_tile_owner << "\n";

        std::vector<Point> tileList;
        int tileNum = (split_axis == "row") ? row : col;
        int core_start_row = (int) router_.getRM()->getrsc()->absTileRow(TileType::Core, 0);
        for(int i = 0; i < tileNum; i++) {
            if (split_axis == "row") {
                //FIXME get the core tile base from resource manager
                tileList.push_back(Point{round_idx + core_start_row/*core row start */, i});
                llvm::outs() << " same row  list row = " << round_idx + 3 << "col = " << i << "\n";
            } else {
                tileList.push_back(Point{i + core_start_row, round_idx});
                llvm::outs() << "same col list row = " << i + core_start_row << "col = " << round_idx << "\n";
            }
        }
        auto output = rewriter.getI32Type();
        auto tilecreatehandle = rewriter.create<TileArrayHandleCreate>(op->getLoc(), output, "array handle");
        // if input choose first tile, if output use last tile for row base, and first tile for col base
        // this tile is used to connect shim
        Point firtTile = tileList[0];
        if (processing_type == 0) {
            firtTile = tileList[0];
            // start to convert
            std::optional<TypeBasedTileLoc> dstcoreloc(TypeBasedTileLoc{TileType::Core, firtTile});
            std::cout << "tile type is  TileType::Core , tile relative row is " << firtTile.r <<std::endl;
            std::ostringstream ostr;
            ostr << "dio" << ioIdx++;
            auto dio = router_.createDataIO(ostr.str(), dstcoreloc, DMADIRECTION::MM2S);
            ///*
            int shimcol = dio->colpos();
            int dioid = dio->id();
            std::cout << "get the shim tile is " << shimcol << " channel is " << dio->channel()  << " IOID is " << dio->id() << std::endl;
            Point shimpoint= {0, shimcol};
            auto shimIoOp = rewriter.create <IOShimTileCreate> ( op->getLoc(), output, 0, shimcol, dioid, ostr.str(), static_cast <int> (DMADIRECTION::MM2S), dio->channel());
            /*
            StringRef inputDirStr = PortDirectiontoString(conn.SlaveReceiveForwardDirection);
        int inputPortIdx = conn.SlaveReceiveForwardDirectionPortIdx;

        
        // Special handling for the SHIM tile to enable its external port
        if (point == shimpoint) {
            if (dio->type() == IOType::Input) {
                rewriter.create<EnableExtToAieShimPort>(loc, outputType, currentTileOp.getResult(), inputDirStr, inputPortIdx);
            } else {
                rewriter.create<EnableAieToExtShimPort>(loc, outputType, currentTileOp.getResult(), inputDirStr, inputPortIdx);
            }
        }
            */
            //----------start create path--------stream switch-----------
            auto rpath = router_.createPath(dioid, tileList);
            std::unordered_map<Point, Operation*, Point::Hash> dsttiles;
            for(auto x: tileList) {
                auto tile1 = rewriter.create<routinghw::TileCreate>(op->getLoc(), output, tilecreatehandle.getResult(),x.r, x.c, "tile reserved");
                dsttiles[{x.r , x.c}] = tile1;
            }
        
            ParseTheCCTRoutingPath(op, std::nullopt, StreamType::BROADCAST, dioid, shimpoint, dio, shimIoOp, tilecreatehandle, rpath, dsttiles, router_, rewriter);
            //ParseTheRoutingPath(op, dioid, shimpoint, dio, tilecreatehandle, rpath, dsttiles, router_, rewriter);
        }  else if (processing_type == 2) {
            if (split_axis == "row") {
                firtTile = tileList.back();
            } else if (split_axis == "col") {
                firtTile = tileList[0];
            }
            std::optional<TypeBasedTileLoc> dstcoreloc(TypeBasedTileLoc{TileType::Core, firtTile});
            std::cout << "tile type is  TileType::Core , tile relative row is " << firtTile.r <<std::endl;
            std::ostringstream ostr;
            ostr << "dio" << ioIdx++;
            auto dio = router_.createDataIO(ostr.str(), dstcoreloc, DMADIRECTION::S2MM);
            
            int shimcol = dio->colpos();
            int dioid = dio->id();
            std::cout << "get the shim tile is " << shimcol << " channel is " << dio->channel()  << " IOID is " << dio->id() << std::endl;
            Point shimpoint= {0, shimcol};
            auto shimIoCreate = rewriter.create <IOShimTileCreate> ( op->getLoc(), output, 0, shimcol, dioid, ostr.str(), static_cast <int> (DMADIRECTION::MM2S), dio->channel());
            //----------start create path--------stream switch-----------
            std::vector<Point> pktmergetile = {firtTile};
            auto rpath = router_.createPath(dioid, pktmergetile);
            std::unordered_map<Point, Operation*, Point::Hash> dsttiles;
            for(auto x: tileList) {
                auto tile1 = rewriter.create<routinghw::TileCreate>(op->getLoc(), output, tilecreatehandle.getResult(),x.r, x.c, "tile reserved");
                dsttiles[{x.r , x.c}] = tile1;
            }
            auto lastPkttilemap = GatherPktRoutingPathCreate(op, dioid, shimpoint, dio, tilecreatehandle, rpath, tileList, dsttiles, router_, rewriter);
            ParseTheCCTRoutingPath(op, lastPkttilemap, StreamType::FORWARDONLY, dioid, shimpoint, dio, shimIoCreate, tilecreatehandle, rpath, dsttiles, router_, rewriter);
            //ParseTheRoutingPath(op, dioid, shimpoint, dio, tilecreatehandle, rpath, dsttiles, router_, rewriter);
        }
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};

void RoutingLowerPass::getDependentDialects(DialectRegistry &registry) const {
        registry.insert<LLVM::LLVMDialect>();
}
RoutingLowerPass::RoutingLowerPass(RoutingTopology& rtopology):rtopology_(rtopology) {
}
void RoutingLowerPass::runOnOperation() {
    auto& ctx = getContext();
    auto module = getOperation();
    RewritePatternSet patterns(&ctx),patternsGlobal(&ctx);
    ConversionTarget target(ctx);
    target.addIllegalDialect<routing::routingdialect>();
    target.addLegalDialect<routinghw::RoutingHWDialect>();
    target.addLegalOp<routing::RoutingCreate>();
    target.addLegalOp<routing::YieldOp>();
    //target.addLegalOp<routing::createhwmesh>();
    //target.addLegalOp<routing::createdummytensor>();
    target.addIllegalOp<arith::IndexCastOp>();
    //target.addLegalOp<routinghw::TileArrayHandleCreate>();
    LLVMTypeConverter typeconverter(&ctx);

    //prepare convert parameter
    /*
    Point dst = {4, 5};
    auto res = makeResource("Gen2");          // default variant
    bool ret = ResourceMgr::init(std::move(res));
    auto rmgr = ResourceMgr::instance();
    auto free_shim = rmgr->freeShimNoc(dst);
    auto dio = rmgr->createDataIO(IOType::Input,free_shim->r, free_shim->c);
    std::vector<Point> wall ={};//{Point{2,6}};// for (int r = 0; r < 8; ++r) wall.push_back({r,3});
    RoutingPath router(rmgr, dio, wall);
    */
    //RoutingTopology rtopology("Gen2");
    
    patterns.add<partitiontensorrconvert>(&ctx, typeconverter);
    patterns.add<partitionmeshconvert>(&ctx, typeconverter);

    patterns.add<extract_dataconvert>(&ctx, typeconverter);
    patterns.add<extract_tilesconvert>(&ctx, typeconverter);
    patterns.add<routinggatheroutconvert>(&ctx, typeconverter,rtopology_);

    patterns.add<RoutingcreatehwiowithtargetConvert>(&ctx, typeconverter);
    patterns.add<RoutingmovedatabyioConvert>(&ctx, typeconverter,rtopology_);
    
    
    //patterns.add<arithconstantconvert>(&ctx, typeconverter);

    //erase hwmesh and dummytensor
    //patternsGlobal.add<routingcreatebroadcastconvert>(&ctx, typeconverter,rtopology_);
    patternsGlobal.add<routingcreatedataioconvert>(&ctx, typeconverter,rtopology_);
    patternsGlobal.add<routingcreatetilearrayconvert>(&ctx, typeconverter,rtopology_);
    patternsGlobal.add<indexcastconvert>(&ctx, typeconverter);
    //patternsGlobal.add<arithconstantconvert>(&ctx, typeconverter);
    

    patternsGlobal.add<createhwmeshconvert>(&ctx, typeconverter);
    patternsGlobal.add<createdummytensorconvert>(&ctx, typeconverter);
    //rewrite the ops inside scf::exe
    ///*
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    module->walk([&](scf::ExecuteRegionOp exec) {
        //only deal with the routing_memo executeregionop
        if (!exec->getAttrOfType<StringAttr>("routing_memo")) {
            return;
        }

        if (failed(applyPartialConversion(exec, target, frozenPatterns ))) {
            llvm::outs() << "routing convert failed \n";
        }
    });//*/
    //rewrite the hwmesh and hwdummy tensor
    if (failed(applyPartialConversion(module, target, std::move(patternsGlobal) ))) {
        llvm::outs() << "routing convert failed 2--- \n";
    }
    return;
}