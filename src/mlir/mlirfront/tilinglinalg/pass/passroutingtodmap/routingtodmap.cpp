/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "routingtodmap.h"
#include "routinghw_pkt_slot.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <sstream>
int dmapioIdx = 0;
//connectpktstreamswitchport
/*
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
        std::cout << "  - SlaveReceiveForwardDirection: " << PortDirectiontoString(value.SlaveReceiveForwardDirection)
<< std::endl; std::cout << "  - SlaveReceiveForwardDirectionPortIdx: " << (int)value.SlaveReceiveForwardDirectionPortIdx
<< std::endl; std::cout << "  - SlaveReceivePktID: " << value.SlaveReceivePktID << std::endl; std::cout << "  -
SlaveReceivePktType: " << value.SlaveReceivePktType << std::endl; std::cout << "  - localDMAForwardPortIdx: " <<
value.localDMAForwardPortIdx << std::endl; std::cout << "  - localDMAForwardPktID: " << value.localDMAForwardPktID <<
std::endl; std::cout << "  - localDMAForwardPktType: " << value.localDMAForwardPktType << std::endl; std::cout << "  -
MasterSendToNextTileDirection: " << PortDirectiontoString(value.MasterSendToNextTileDirection) << std::endl; std::cout
<< "  - MasterSendToNextTileDirectionPortIdx: " << (int)(value.MasterSendToNextTileDirectionPortIdx) << std::endl;

        auto pktSlot = routinghw::pktslot::makePktSlotAttrs(rewriter);
        rewriter.create<routinghw::ConnectStreamPktSwitchPort>(
            op->getLoc(),                   // Operation location
            output,
            curTileOp.getResult(),                   // Tile to be configured
            rewriter.getStringAttr(PortDirectiontoString(value.SlaveReceiveForwardDirection)), // Direction of the port
receiving the stream rewriter.getI32IntegerAttr((int)value.SlaveReceiveForwardDirectionPortIdx),     // Index of the
receiving port rewriter.getI32IntegerAttr(value.SlaveReceivePktID),// Packet ID to expect
            rewriter.getI32IntegerAttr(value.SlaveReceivePktType),// Packet Type to expect
            rewriter.getStringAttr(PortDirectiontoString(PortDirection::DMA)),  // local DMA direction NONE means no DMA
            rewriter.getI32IntegerAttr(value.localDMAForwardPortIdx),  // Index of the local DMA port to send to
            rewriter.getI32IntegerAttr(value.localDMAForwardPktID ),    // Packet ID for the DMA transfer
            rewriter.getI32IntegerAttr(value.localDMAForwardPktType),  // Packet Type for the DMA transfer
            rewriter.getStringAttr(PortDirectiontoString(value.MasterSendToNextTileDirection)),     // No forwarding:
empty master direction rewriter.getI32IntegerAttr((int)(value.MasterSendToNextTileDirectionPortIdx)), // No forwarding:
port index 0 mlir::BoolAttr{}, ROUTINGHW_PKT_SLOT_ATTRS(pktSlot)
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
    auto troutingmap = GetSeqPath(rpath,dio,dsttiles, streamtype,lastPkttilemap,router_,rewriter);
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
        if (it == connectionData.end()) continue; // This point might not have connections (e.g., an un-routed
destination)

        const StreamCCTConnection& conn = it->second;
        auto currentTileOp = dyn_cast<routinghw::TileCreate>(allTileOps.at(point));

        // Ensure the tile has an input port to connect from
        if (conn.SlaveReceiveForwardDirection == PortDirection::NONE) {
            if (lastPkttilemap && lastPkttilemap->tile == point) {
                auto output = rewriter.getI32Type();
                auto tileOp = dyn_cast<routinghw::TileCreate>((Operation* )lastPkttilemap->tileOp);
                auto pktSlot = routinghw::pktslot::makePktSlotAttrs(rewriter);
                ///*
                rewriter.create<routinghw::ConnectStreamPktSwitchPort>(
                        loc,                   // Operation location
                        output,
                        tileOp.getResult(),                   // Tile to be configured
                        rewriter.getStringAttr(PortDirectiontoString(PortDirection::NONE)), // Direction of the port
receiving the stream rewriter.getI32IntegerAttr(0),     // Index of the receiving port rewriter.getI32IntegerAttr(0),//
Packet ID to expect rewriter.getI32IntegerAttr(0),// Packet Type to expect
                        rewriter.getStringAttr(PortDirectiontoString(PortDirection::NONE)),
                        rewriter.getI32IntegerAttr(0),  // Index of the local DMA port to send to
                        rewriter.getI32IntegerAttr(0),    // Packet ID for the DMA transfer
                        rewriter.getI32IntegerAttr(0),  // Packet Type for the DMA transfer
                        rewriter.getStringAttr(PortDirectiontoString(conn.MasterSendToNextTileDirection)),     // No
forwarding: empty master direction rewriter.getI32IntegerAttr((int)(conn.MasterSendToNextTileDirectionPortIdx)), // No
forwarding: port index 0 mlir::BoolAttr{}, ROUTINGHW_PKT_SLOT_ATTRS(pktSlot)
                );
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

        // Create connection to the next tile in the path
        if (conn.MasterSendToNextTileDirection != PortDirection::NONE) {
            if (point == shimpoint) {
                 rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, shimio.getResult(),
                    inputDirStr, inputPortIdx,
                    PortDirectiontoString(conn.MasterSendToNextTileDirection),
conn.MasterSendToNextTileDirectionPortIdx); } else { rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType,
currentTileOp.getResult(), inputDirStr, inputPortIdx, PortDirectiontoString(conn.MasterSendToNextTileDirection),
conn.MasterSendToNextTileDirectionPortIdx);
            }
        }

        // Create connection to the local DMA
        if (conn.localDMAForwardDirection != PortDirection::NONE) {
            rewriter.create<ConnectStreamSingleSwitchPort>(loc, outputType, currentTileOp.getResult(),
                    inputDirStr, inputPortIdx,
                    "DMA", conn.localDMAForwardPortIdx);
        }

    }
}
*/
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
                        auto tile1 = rewriter.create<routinghw::TileCreate>(op->getLoc(), output,
tilecreatehandle.getResult(),p.r, p.c, "tile reserved in path"); pathtiles[p] = tile1;
                    }
                }

                //if this branch is the last branch we need to deal with the last item, then add a dump node as we only
process the previous on of current if (i == tree.branches.size() - 1) {
                    tree.branches[i].push_back(tree.branches[i].back());
                }
                int len = tree.branches[i].size();
                //as the stream switch connect need to find the matched master (previous tile) slave (current tile)
port, the current process point
                //is the previous point which already did tile occupy, then we can have the master port information
                for (int j = 0; j < len; j ++) {
                    auto currentpoint = (prev_optional_point == std::nullopt ? tree.branches[i][j] :
*prev_optional_point); auto nextpoint = tree.branches[i][j];
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
                            if (!router_.occupyLink(currentpoint, nextpoint, dioid, portNum, portdirectionPrevSlave,
portdirectionCurMaster)) { llvm::outs() << "link occupy failed " << "\n"; assert(0);
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
                            rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output,
curop.getResult(),portprevmaster, portprevidx, portdirectionPrevSlaveStr, portNum);
                            //add to dma logic
                            //
                            auto rowcol = getrowcol(curop);
                            if (rm->getrsc()->tileType(rowcol[0], rowcol[1]) == TileType::Core) {
                               if (auto portnumptr = rm->tile(rowcol[0],rowcol[1]).occupyport(IOType::TileDMA,
PortDirection::DMA, -1)) { rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output,
curop.getResult(),portprevmaster, portprevidx, "DMA", *portnumptr);
                               }
                            }
                            //
                        } else {
                            //no master port finding means this is the inital shim port get the master information from
io
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
                                    rewriter.create<EnableExtToAieShimPort>(op->getLoc(), output,
curop.getResult(),shimportdirstr, shimportnum); } else { rewriter.create<EnableAieToExtShimPort>(op->getLoc(), output,
curop.getResult(), shimportdirstr, shimportnum);
                                }
                                rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output,
curop.getResult(),shimportdirstr, shimportnum, portdirectionPrevSlaveStr, portNum); llvm::outs() << "the logic wrong
\n";
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
        // Replace all uses of this op's result with its operand.
        // This allows the user of this op to be processed correctly.
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};

//createscheduletensor
struct createscheduletensorconvert : public ConversionPattern {
    explicit createscheduletensorconvert(MLIRContext * ctx, LLVMTypeConverter &converter):
        ConversionPattern(routing::createscheduletensor::getOperationName(),1, ctx), typeconverter(converter) {

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
struct extract_dataconvertdmap : public ConversionPattern {
    explicit extract_dataconvertdmap(MLIRContext * ctx, LLVMTypeConverter &converter):
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
        //rewriter.eraseOp(op);
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
};
/*
//RoutingCreate
struct RoutingmovedatabyioConvert2 : public ConversionPattern {
    explicit RoutingmovedatabyioConvert2(MLIRContext * ctx, LLVMTypeConverter &converter, RoutingTopology & router):
        ConversionPattern(routing::movedatabyio::getOperationName(),1, ctx), typeconverter(converter), router_(router) {
            //setHasBoundedRewriteRecursion(true);
            //setBenefit(10);
        }

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        //function to get blockarg constant
        rewriter.eraseOp(op);
        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
};
*/

//RoutingCreate
struct RoutingmovedatabyioConvertdmap : public ConversionPattern {
    explicit RoutingmovedatabyioConvertdmap(MLIRContext * ctx, LLVMTypeConverter &converter, RoutingTopology & router, uint32_t oplevel):
        ConversionPattern(routing::movedatabyio::getOperationName(),1, ctx), typeconverter(converter), router_(router) {
            //setHasBoundedRewriteRecursion(true);
            //MLIRContext * context = ctx;
            moplevel = oplevel;
        }

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter ) const override {    
        //function to get blockarg constant
        ///*
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
        // Control-packet flow marker: if the abstract routing op is tagged as a
        // control write (register/config delivery), propagate the marker onto the
        // produced dmap streams so DmapToDmaphop can forward it onto create_path,
        // and passdmaphoptoroutinghw can select the tile CTRL sink port.
        bool isControl = op->hasAttrOfType<BoolAttr>("control") && op->getAttrOfType<BoolAttr>("control").getValue();
        auto tagControl = [&](Operation *streamOp) {
            if (isControl && streamOp)
                streamOp->setAttr("control", rewriter.getBoolAttr(true));
        };
        // get all the op
        routing::createhwmesh createhwmesh;
        routing::createscheduletensor createscheduletensor;
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
        if (!(createscheduletensor = partitiontensor->getOperands()[0].getDefiningOp<routing::createscheduletensor>())) {
            llvm::outs() << " createscheduletensor not found return" << "\n";
            return failure();
        }

        int col = createhwmesh.getCol();
        int row = createhwmesh.getRow();
        auto shape = createscheduletensor.getShape();

        int tensor_x = shape[0].cast<mlir::IntegerAttr>().getInt();
        int tensor_y = shape[1].cast<mlir::IntegerAttr>().getInt();

        llvm::StringRef split_axis = partitionmesh.getSplitaxis();
        llvm::StringRef hw_axis_owner = partitiontensor.getPartition().getHwAxisOwner();
        llvm::StringRef replicate_on = partitiontensor.getPartition().getReplicateOn();
        llvm::StringRef single_tile_owner = partitiontensor.getPartition().getSingleTileOwner();

        llvm::outs() << "col =" << col << " row=" << row  << " tensor_x=" << tensor_x << " tensor_y=" << tensor_y << "\n";
        llvm::outs() << "split_axis =" << split_axis << " hw_axis_owner=" << hw_axis_owner  << "\n";
        llvm::outs() << "replicate_on=" << replicate_on << " single_tile_owner=" << single_tile_owner << "\n";

        std::vector<Point> tileList;
        int tileNum = (split_axis == "row") ? row : col;
        int core_start_row = (int) router_.getRM()->getrsc()->absTileRow(TileType::Core, 0);
        for(int i = 0; i < tileNum; i++) {
            if (split_axis == "row") {
                //FIXME get the core tile base from resource manager
                tileList.push_back(Point{round_idx + core_start_row, i});
                llvm::outs() << " same row  list row = " << round_idx + 3 << "col = " << i << "\n";
            } else {
                tileList.push_back(Point{i + core_start_row, round_idx});
                llvm::outs() << "same col list row = " << i + core_start_row << "col = " << round_idx << "\n";
            }
        }
        auto output = rewriter.getI32Type();
       // */
        auto ctx = getContext();
        auto ioout = dmap::dmapioenginetypeType::get(ctx); 
        printf("before xxxx---0----\n");
        auto io = rewriter.create<dmap::define_io_engine>(rewriter.getUnknownLoc(),  ioout, dmapioIdx++, "SHIM"); 
        auto ioconfigret = dmap::dmapioconfigType::get(ctx);

        // Define patterns for SEND and RECEIVE
        auto sendpattern = dmap::dataaccesspatternAttr::get(ctx, rewriter.getStringAttr("SEND"), 16, 1, 1);
        auto receivepattern = dmap::dataaccesspatternAttr::get(ctx, rewriter.getStringAttr("RECEIVE"), 16, 1, 1);
        
        // Determine patterns based on processing_type
        dmap::dataaccesspatternAttr io_pattern;
        dmap::dataaccesspatternAttr core_group_pattern;
        std::string port_symbol_name;

        // Use dmapioIdx to make port names unique when multiple movedatabyio ops exist
        int portIdx = dmapioIdx - 1; // dmapioIdx was already incremented above
        if (processing_type == 0) { // Broadcast
            io_pattern = sendpattern;
            core_group_pattern = receivepattern;
            port_symbol_name = "receive_port_" + std::to_string(portIdx);
        } else { // Gather (processing_type == 2)
            io_pattern = receivepattern;
            core_group_pattern = sendpattern;
            port_symbol_name = "send_port_" + std::to_string(portIdx);
        }
       
        mlir::Type pgeout = dmap::dmacoreenginegroupType::get(ctx);
        auto peg = rewriter.create<define_core_group>(rewriter.getUnknownLoc(),  pgeout, round_idx, tileNum, split_axis);
        //config
        mlir::Type portconfig = dmap::dmapportconfigType::get(ctx);
        auto pf = rewriter.create<dmap::define_port_configure>(rewriter.getUnknownLoc(), portconfig, port_symbol_name, core_group_pattern);
      
        //
        //Data
        mlir::SmallVector<mlir::Attribute, 4> shapeElems;
        shapeElems.push_back(rewriter.getI64IntegerAttr(16));
        shapeElems.push_back(rewriter.getI64IntegerAttr(16));
        mlir::ArrayAttr shapettr = mlir::ArrayAttr::get(ctx, shapeElems);
        mlir::Type elementType = rewriter.getF32Type();
        //mlir::Type myDataHandleType = dmap::dmapdataType::get(ctx);
        mlir::Type myDataHandleType = rewriter.getI32Type();
        //auto data = rewriter.create<create_data>(rewriter.getUnknownLoc(),  myDataHandleType, shapettr, elementType);
        //------
        auto shimioconfig = rewriter.create<create_io_engin_with_config>(rewriter.getUnknownLoc(),  ioconfigret,  io.getResult(), io_pattern);
        //
        //config port group
        llvm::SmallVector<dmap::dataconfmapitemAttr> itemsVector;
        for (int i = 0; i < tileNum; i++) {
            dmap::dataconfmapitemAttr item = dmap::dataconfmapitemAttr::get(ctx,i, mlir::SymbolRefAttr::get(ctx, port_symbol_name));
            itemsVector.push_back(item);
        }
        
        dmap::dataconfigmapAttr configMapAttr = dmap::dataconfigmapAttr::get(ctx,itemsVector);
        auto gcret = dmap::dmacoregroupconfigType::get(ctx);
        auto gcmap = rewriter.create<create_core_group_with_config>(rewriter.getUnknownLoc(),  gcret, peg.getResult(), "row", configMapAttr);
        auto streamret = dmap::dmapportstreamType::get(ctx);
        //op level is a outside parameter, opby mem tile will pass mem tile
        if (moplevel == 1) {
            auto memio = rewriter.create<define_io_engine>(rewriter.getUnknownLoc(),  ioout, 0, "MEM"); 
            ///*
            auto memiorecvconfig = rewriter.create<create_io_engin_with_config>(rewriter.getUnknownLoc(),  ioconfigret,  memio.getResult(),receivepattern);
            auto memiosendconfig = rewriter.create<create_io_engin_with_config>(rewriter.getUnknownLoc(),  ioconfigret,  memio.getResult(),sendpattern);
            
            auto shimToMemAttr = dmapioAttr::get(ctx, dmapio::DMAP_SHIMIO);
            auto memToCoreAttr = dmapioAttr::get(ctx, dmapio::DMAP_MEMTILEIO);
            
            auto groupIndexAttr = rewriter.getI32IntegerAttr(0);
            auto streamIdAttr1 = rewriter.getI32IntegerAttr(1);
            
            Value stream_src, stream_dst;
            if (processing_type == 0) { // Broadcast: SHIM -> MEM -> CORE
                stream_src = shimioconfig.getResult();
                stream_dst = memiorecvconfig.getResult();
            } else { // Gather: CORE -> MEM -> SHIM
                stream_src = gcmap.getResult();
                stream_dst = memiorecvconfig.getResult();
            }
            auto streamhandle1 = rewriter.create<createstream>(rewriter.getUnknownLoc(),  streamret, stream_src, stream_dst, shimToMemAttr, groupIndexAttr, streamIdAttr1);
            tagControl(streamhandle1);

            if (processing_type == 0) { // Broadcast: SHIM -> MEM -> CORE
                stream_src = memiosendconfig.getResult();
                stream_dst = gcmap.getResult();
            } else { // Gather: CORE -> MEM -> SHIM
                stream_src = memiosendconfig.getResult();
                stream_dst = shimioconfig.getResult();
            }
            auto streamhandle2 = rewriter.create<createstream>(rewriter.getUnknownLoc(),  streamret, stream_src, stream_dst, memToCoreAttr, groupIndexAttr, streamIdAttr1);
            tagControl(streamhandle2);

            // Create a chained stream from streamhandle1 and streamhandle2, then push once to the chained stream.
            auto chainType = dmap::dmapportchainstreamType::get(ctx);
            // The generated op build expects (TypeRange resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attrs).
            auto chainStreamOp = rewriter.create<createchainstream>(rewriter.getUnknownLoc(),
                                                                   chainType,
                                                                   mlir::ValueRange{streamhandle1.getResult(),
                                                                                    streamhandle2.getResult()});
            auto chainStream = chainStreamOp.getResult();

            // Single push/pull that targets the chained stream
            if (processing_type == 0) {
                rewriter.create<push>(rewriter.getUnknownLoc(), extract_data.getResult(), chainStream);
            } else {
                rewriter.create<pull>(rewriter.getUnknownLoc(), extract_data.getResult(), chainStream);
            }
            //*/
        } else {
            auto shimToCoreAttr = dmapioAttr::get(ctx, dmapio::DMAP_SHIMIO);
            
            auto groupIndexAttr = rewriter.getI32IntegerAttr(0);
            auto streamIdAttr = rewriter.getI32IntegerAttr(1);
            
            Value stream_src, stream_dst;
            if (processing_type == 0) { // Broadcast: SHIM -> CORE
                stream_src = shimioconfig.getResult();
                stream_dst = gcmap.getResult();
            } else { // Gather: CORE -> SHIM
                stream_src = gcmap.getResult();
                stream_dst = shimioconfig.getResult();
            }
            auto streamhandle = rewriter.create<createstream>(rewriter.getUnknownLoc(),  streamret, stream_src, stream_dst, shimToCoreAttr, groupIndexAttr, streamIdAttr);
            tagControl(streamhandle);

            if (processing_type == 0) {
                rewriter.create<push>(rewriter.getUnknownLoc(), extract_data.getResult(),streamhandle.getResult());
            } else {
                rewriter.create<pull>(rewriter.getUnknownLoc(), extract_data.getResult(),streamhandle.getResult());
            }
        }
        //
        rewriter.eraseOp(op);

        return success();
    }
private:
    LLVMTypeConverter& typeconverter;
    RoutingTopology & router_;
    uint32_t moplevel;
    //MLIRContext * context;
};

void RoutingToDmapPass::getDependentDialects(DialectRegistry &registry) const {
        registry.insert<LLVM::LLVMDialect>();
}
RoutingToDmapPass::RoutingToDmapPass(RoutingTopology& rtopology):rtopology_(rtopology) {
}
void RoutingToDmapPass::runOnOperation() {
    auto& ctx = getContext();
    auto module = getOperation();
    RewritePatternSet patterns(&ctx),patternsGlobal(&ctx);
    ConversionTarget target(ctx);
    //target.addIllegalDialect<routing::routingdialect>();
    target.addLegalDialect<dmap::dmapdialect>();
    target.addLegalOp<routing::RoutingCreate>();
    target.addLegalOp<routing::YieldOp>();
    //target.addLegalOp<routing::createhwmesh>();
    //target.addLegalOp<routing::createscheduletensor>();
    //target.addIllegalOp<arith::IndexCastOp>();
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
    
    //patterns.add<partitiontensorrconvert>(&ctx, typeconverter);
    patterns.add<partitionmeshconvert>(&ctx, typeconverter);

    //patterns.add<extract_dataconvertdmap>(&ctx, typeconverter);
    patterns.add<extract_tilesconvert>(&ctx, typeconverter);
    patterns.add<routinggatheroutconvert>(&ctx, typeconverter,rtopology_);

    patterns.add<RoutingcreatehwiowithtargetConvert>(&ctx, typeconverter);
    patterns.add<RoutingmovedatabyioConvertdmap>(&ctx, typeconverter,rtopology_, 0/*op through memtile or not*/);
    
    
    //patterns.add<arithconstantconvert>(&ctx, typeconverter);

    //erase hwmesh and dummytensor
    //patternsGlobal.add<routingcreatebroadcastconvert>(&ctx, typeconverter,rtopology_);
    patternsGlobal.add<routingcreatedataioconvert>(&ctx, typeconverter,rtopology_);
    patternsGlobal.add<routingcreatetilearrayconvert>(&ctx, typeconverter,rtopology_);
    patternsGlobal.add<indexcastconvert>(&ctx, typeconverter);
    //patternsGlobal.add<arithconstantconvert>(&ctx, typeconverter);
    

    patternsGlobal.add<createhwmeshconvert>(&ctx, typeconverter);
    //patternsGlobal.add<createscheduletensorconvert>(&ctx, typeconverter);
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