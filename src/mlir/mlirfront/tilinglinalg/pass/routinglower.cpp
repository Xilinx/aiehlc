/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "routinglower.h"
#include "routing/routingpath.h"
#include <sstream>
int ioIdx = 0;
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
                            auto prevportinfo = tileMasterPortMapping[currentpoint];
                            auto portprevmaster = PortDirectiontoString((PortDirection)prevportinfo[0]);
                            auto portdirectionPrevSlaveStr = PortDirectiontoString(portdirectionPrevSlave);
                            auto portprevidx = prevportinfo[1];
                            rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),portprevmaster, portprevidx, portdirectionPrevSlaveStr, portNum);
                            //add to dma logic
                            ///*
                            auto rowcol = getrowcol(curop);
                            if (rm->getrsc()->tileType(rowcol[0], rowcol[1]) == TileType::Core) {
                               if (auto portnumptr = rm->tile(rowcol[0],rowcol[1]).occupyport(IOType::TileDMA, PortDirection::DMA, -1)) {
                                   rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),portprevmaster, portprevidx, "DMA", *portnumptr);                                
                               }
                            }
                            //*/
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
}

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
        ///*
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
                            auto prevportinfo = tileMasterPortMapping[currentpoint];
                            auto portprevmaster = PortDirectiontoString((PortDirection)prevportinfo[0]);
                            auto portdirectionPrevSlaveStr = PortDirectiontoString(portdirectionPrevSlave);
                            auto portprevidx = prevportinfo[1];
                            rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),portprevmaster, portprevidx, portdirectionPrevSlaveStr, portNum);
                            //add to dma logic
                            ///*
                            auto rowcol = getrowcol(curop);
                            if (rm->getrsc()->tileType(rowcol[0], rowcol[1]) == TileType::Core) {
                               if (auto portnumptr = rm->tile(rowcol[0],rowcol[1]).occupyport(IOType::TileDMA, PortDirection::DMA, -1)) {
                                   rewriter.create<ConnectStreamSingleSwitchPort>(op->getLoc(), output, curop.getResult(),portprevmaster, portprevidx, "DMA", *portnumptr);                                
                               }
                            }
                            //*/
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
        routing::createhwiowithtarget createhwiowithtarget = operands[1].getDefiningOp<routing::createhwiowithtarget>();
        if (!(extract_data = operands[0].getDefiningOp<routing::extract_data>())) {
            if (gatherout_pktmerge = operands[0].getDefiningOp<routing::routinggatherout>()){
                extract_data = gatherout_pktmerge->getOperands()[1].getDefiningOp<routing::extract_data>();
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
        int tileNum = (split_axis == "row") ? col : row;
        for(int i = 0; i < tileNum; i++) {
            if (split_axis == "row") {
                tileList.push_back(Point{round_idx + 3/*core row start */, i});
                llvm::outs() << " row = " << row + round_idx << "col = " << i << "\n";
            } else {
                 tileList.push_back(Point{i, round_idx});
                llvm::outs() << " row = " << i << "col = " << row + round_idx << "\n";
            }
        }
        // start to convert
        Point firtTile = tileList[0];
        std::optional<TypeBasedTileLoc> dstcoreloc(TypeBasedTileLoc{TileType::Core, firtTile});
        std::cout << "tile type is  TileType::Core , tile relative row is " << firtTile.r <<std::endl;
        std::ostringstream ostr;
        ostr << "dio" << ioIdx++;
        auto dio = router_.createDataIO(ostr.str(), dstcoreloc,  DMADIRECTION::MM2S);
        //auto ctx = getContext();
        //auto output = rewriter.getI32Type();
        ///*
        int shimcol = dio->colpos();
        int dioid = dio->id();
        auto output = rewriter.getI32Type();

        std::cout << "get the shim tile is " << shimcol << " channel is " << dio->channel()  << " IOID is " << dio->id() << std::endl;

        auto tilecreatehandle = rewriter.create<TileArrayHandleCreate>(op->getLoc(), output, "array handle");

        Point shimpoint= {0, shimcol};

        
        auto opCreated = rewriter.create <IOShimTileCreate> ( op->getLoc(), output, 0, shimcol, dioid, ostr.str(), static_cast <int> (DMADIRECTION::MM2S), dio->channel());

        //----------start create path--------stream switch-----------
        auto rpath = router_.createPath(dioid, tileList);
        std::unordered_map<Point, Operation*, Point::Hash> dsttiles;
        for(auto x: tileList) {
            auto tile1 = rewriter.create<routinghw::TileCreate>(op->getLoc(), output, tilecreatehandle.getResult(),x.r, x.c, "tile reserved");
            dsttiles[{x.r , x.c}] = tile1;
        }

        ParseTheRoutingPath(op, dioid, shimpoint, dio, tilecreatehandle, rpath, dsttiles, router_, rewriter);

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
    patternsGlobal.add<routingcreatebroadcastconvert>(&ctx, typeconverter,rtopology_);
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