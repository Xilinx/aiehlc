/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "../passblueprinttoschedule/passblueprinttoschedule.h"
#include "../passblueprinttoschedulekernel/passblueprinttoschedulekernel.h"
#include "../passdfscheduleprovenancemap/passdfscheduleprovenancemap.h"
#include "../passdfscheduletoapi/passdfscheduletoapi.h"
#include "../passdfscheduletokernelapi/passdfscheduletokernelapi.h"
#include "../passdmaphopprovenancemap/passdmaphopprovenancemap.h"
#include "../passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.h"
#include "../passdmaphoptoroutinghw/passdmaphoptoroutinghw.h"
#include "../passdmaptodmaphop/dmaptodmaphop.h"
#include "../passroutingtodmap/routingtodmap.h"
#include "../passschedulecanonicalize/passschedulecanonicalize.h"
#include "dfscheblueprintmanager.h"
#include "dfschedulemanager.h"
#include "dmaphopmanager.h"
#include "dmapmanager.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "routinghwlower.h"
#include "routinghwmanager.h"
#include "routinghwverify.h"
#include "routinglower.h"
#include "routingmanager.h"
#include "routingunrolling.h"
#include <iostream>
// #include "llvm/IR/IRPrintingPasses.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"


#include "routingdeadargclean.h"

#include "../kernelconfig/kernelconfig.h"
#include "../tilinglinalg_pipeline.h"
#include "hw/ResourceManager.h"
#include "mlir/Transforms/Passes.h"
#include "routingconstantfold.h"

// ---------------------------------------------------------------------------
// IR dump helpers
// ---------------------------------------------------------------------------

static std::string setupIRDir(const std::string &subdir) {
    llvm::SmallString<256> cwdPath;
    if (std::error_code EC = llvm::sys::fs::current_path(cwdPath)) {
        llvm::errs() << "Failed to get current directory: " << EC.message() << "\n";
        return "";
    }
    std::string dir = (cwdPath + "/ir/" + subdir).str();
    if (std::error_code EC = llvm::sys::fs::create_directories(dir)) {
        llvm::errs() << "Failed to create IR directory " << dir << ": " << EC.message() << "\n";
        return "";
    }
    std::cout << "IR output directory: " << dir << std::endl;
    return dir;
}

static std::string setupWorklocalDir() {
    llvm::SmallString<256> cwdPath;
    if (std::error_code EC = llvm::sys::fs::current_path(cwdPath)) {
        llvm::errs() << "Failed to get current directory: " << EC.message() << "\n";
        return "";
    }
    std::string dir = (cwdPath + "/worklocal").str();
    if (std::error_code EC = llvm::sys::fs::create_directories(dir)) {
        llvm::errs() << "Failed to create directory " << dir << ": " << EC.message() << "\n";
        return "";
    }
    return dir;
}

static void dumpIRToFile(mlir::ModuleOp module, const std::string &dir, int stage, const std::string &passName) {
    if (dir.empty())
        return;
    std::string filename = dir + "/" + std::to_string(stage) + "_" + passName + ".mlir";
    std::error_code ec;
    llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "Failed to write IR to " << filename << ": " << ec.message() << "\n";
        return;
    }
    module.print(os);
    std::cout << "  IR -> " << filename << std::endl;
}

static bool runSinglePass(MLIRContext &ctx, mlir::ModuleOp module, std::unique_ptr<mlir::Pass> pass,
                          const std::string &irDir, int &stage, const std::string &passName) {
    mlir::PassManager singlePm(&ctx);
    singlePm.addPass(std::move(pass));
    if (failed(singlePm.run(module))) {
        llvm::errs() << "ERROR: " << passName << " failed!\n";
        return false;
    }
    dumpIRToFile(module, irDir, stage, passName);
    stage++;
    return true;
}

// Unit test function to verify path contiguity for RoutingLowerPass
// Global AIE generation string, set from --gen CLI argument
static std::string g_aieGen = "Gen2";
static int g_outputPpDepth = 2; // pp_depth for output tensor (default: 2 = ping-pong)

void testRoutingLowerPassPathContiguity() {
    std::cout << "\n=== Testing RoutingLowerPass Path Contiguity ===" << std::endl;
    
    MLIRContext ctx;
    
    routingmanager mtest;
    routinghwmanager mtesthw;
    mtesthw.loaddialect(&ctx);
    mtest.loaddialect(&ctx);
    ctx.getOrLoadDialect<arith::ArithDialect>();
    
    // Create test module with row-based tiling
    auto module1 = mtest.ops_testNew(&ctx, 1);
    
    // Apply passes up to RoutingLowerPass
    mlir::PassManager pm(&ctx);
    RoutingTopology rtopology(g_aieGen);

    pm.addPass(std::make_unique<RoutingUnrollingLowerPass>());
    pm.addPass(std::make_unique<RoutingLowerPass>(rtopology));
    
    if (failed(pm.run(module1))) {
        std::cerr << "ERROR: Pass execution failed!" << std::endl;
        return;
    }
    
    // Print the module after running the passes
    std::cout << "\n--- Module IR after RoutingLowerPass ---" << std::endl;
    module1.dump();
    std::cout << "--- End of Module IR ---\n" << std::endl;
    
    // Extract and verify path information from the module
    std::cout << "\n--- Verifying Path Contiguity ---" << std::endl;
    
    struct TileLocation {
        int row;
        int col;
        TileLocation(int r = -1, int c = -1) : row(r), col(c) {}
        bool operator==(const TileLocation& other) const {
            return row == other.row && col == other.col;
        }
        bool operator<(const TileLocation& other) const {
            if (row != other.row) return row < other.row;
            return col < other.col;
        }
        bool isValid() const { return row >= 0 && col >= 0; }
        
        // Helper to print in (col, row) format
        std::string toString() const {
            return "( col:" + std::to_string(col) + ",row:" + std::to_string(row) + ")";
        }
    };
    
    struct TileInfo {
        TileLocation loc;
        Operation* op;
        std::string comments;
    };
    
    struct PortConnection {
        Operation* tileOp;
        TileLocation tileLoc;
        std::string slavePortDir;
        int slavePortIdx;
        std::string masterPortDir;
        int masterPortIdx;
    };
    
    // Map to store paths grouped by routing::RoutingCreate operations
    std::map<Operation*, std::vector<TileInfo>> tilesByRoutingCreate;
    std::map<Operation*, std::vector<PortConnection>> connectionsByRoutingCreate;
    
    // Helper function to get neighbor location based on direction
    auto getNeighbor = [](const TileLocation& loc, const std::string& dir) -> TileLocation {
        if (dir == "NORTH") return TileLocation(loc.row + 1, loc.col);
        if (dir == "SOUTH") return TileLocation(loc.row - 1, loc.col);
        if (dir == "EAST") return TileLocation(loc.row, loc.col + 1);
        if (dir == "WEST") return TileLocation(loc.row, loc.col - 1);
        return TileLocation(-1, -1);
    };
    
    // Walk through the module to find routing::RoutingCreate operations
    module1.walk([&](Operation* op) {
        if (auto routingCreateOp = dyn_cast<routing::RoutingCreate>(op)) {
            std::vector<TileInfo> tiles;
            std::vector<PortConnection> connections;
            std::map<Operation*, TileLocation> opToLocation;
            
            // First pass: collect all tiles
            routingCreateOp->walk([&](Operation* innerOp) {
                if (auto tileOp = dyn_cast<routinghw::TileCreate>(innerOp)) {
                    if (auto rowAttr = tileOp.getRowAttr()) {
                        if (auto colAttr = tileOp.getColAttr()) {
                            int row = rowAttr.getInt();
                            int col = colAttr.getInt();
                            TileLocation loc(row, col);
                            std::string comments = tileOp.getComments().str();
                            tiles.push_back({loc, tileOp, comments});
                            opToLocation[tileOp] = loc;
                        }
                    }
                } else if (auto shimTileOp = dyn_cast<routinghw::IOShimTileCreate>(innerOp)) {
                    if (auto rowAttr = shimTileOp.getRowAttr()) {
                        if (auto colAttr = shimTileOp.getColAttr()) {
                            int row = rowAttr.getInt();
                            int col = colAttr.getInt();
                            TileLocation loc(row, col);
                            tiles.push_back({loc, shimTileOp, "shim"});
                            opToLocation[shimTileOp] = loc;
                        }
                    }
                }
            });
            
            // Helper to check if a tile location exists in our tile set
            auto tileExists = [&opToLocation](const TileLocation& loc) -> bool {
                for (const auto& [op, tileLoc] : opToLocation) {
                    if (tileLoc == loc) return true;
                }
                return false;
            };
            
            // Second pass: collect all port connections
            routingCreateOp->walk([&](Operation* innerOp) {
                if (auto connectOp = dyn_cast<routinghw::ConnectStreamSingleSwitchPort>(innerOp)) {
                    // The tile is the first operand of the ConnectStreamSingleSwitchPort operation
                    if (connectOp->getNumOperands() > 0) {
                        Operation* tileOp = connectOp->getOperand(0).getDefiningOp();
                        if (opToLocation.count(tileOp)) {
                            auto slaveDir = connectOp.getSlaveportdirection().str();
                            auto masterDir = connectOp.getMasterportdirection().str();
                            int slaveIdx = connectOp.getSlaveportidx();
                            int masterIdx = connectOp.getMasterportidx();
                            
                            // Skip connections that involve DMA (source or dest)
                            // DMA connections typically use "DMA" in the port direction
                            if (slaveDir == "DMA" || masterDir == "DMA") {
                                std::cout << "  Skipping DMA connection at tile " << opToLocation[tileOp].toString() 
                                          << ": " << slaveDir << " -> " << masterDir << std::endl;
                                return;
                            }
                            
                            connections.push_back({
                                tileOp,
                                opToLocation[tileOp],
                                slaveDir,
                                slaveIdx,
                                masterDir,
                                masterIdx
                            });
                        }
                    }
                } else if (auto pktConnectOp = dyn_cast<routinghw::ConnectStreamPktSwitchPort>(innerOp)) {
                    // Handle packet switch port connections
                    if (pktConnectOp->getNumOperands() > 0) {
                        Operation* tileOp = pktConnectOp->getOperand(0).getDefiningOp();
                        if (opToLocation.count(tileOp)) {
                            auto receiveSlaveDir = pktConnectOp.getReceiveslavedirection().str();
                            auto localDmaDir = pktConnectOp.getLocaldmadirection().str();
                            auto forwardMasterDir = pktConnectOp.getForwardmasterdirection().str();
                            
                            int receiveSlaveIdx = pktConnectOp.getReceiveslaveportidx();
                            int localDmaIdx = pktConnectOp.getLocaldmaportidx();
                            int forwardMasterIdx = pktConnectOp.getForwardmasterportidx();
                            
                            // For packet connections with DMA:
                            // - If localdmadirection == "DMA", the tile has a DMA that sends data
                            // - The forwardmasterdirection shows where DMA data goes
                            // - The receiveslavedirection shows if tile receives from neighbor
                            
                            bool hasDma = (localDmaDir == "DMA");
                            bool receivesFromNeighbor = (receiveSlaveDir != "DMA" && receiveSlaveDir != "NONE");
                            bool forwardsToNeighbor = (forwardMasterDir != "DMA" && forwardMasterDir != "NONE");
                            
                            if (hasDma) {
                                // Case 1: DMA sends to a neighbor (DMA -> stream out)
                                // This creates an outgoing connection
                                if (forwardsToNeighbor) {
                                    connections.push_back({
                                        tileOp,
                                        opToLocation[tileOp],
                                        localDmaDir,  // DMA as source (slave)
                                        localDmaIdx,
                                        forwardMasterDir,  // Direction to neighbor (master)
                                        forwardMasterIdx
                                    });
                                    std::cout << "  PKT connection (DMA->stream) at tile " << opToLocation[tileOp].toString() 
                                              << ": DMA[" << localDmaIdx << "] -> "
                                              << forwardMasterDir << "[" << forwardMasterIdx << "]" << std::endl;
                                }
                                
                                // Case 2: Tile also receives from neighbor -> DMA
                                // This creates an incoming connection that feeds the DMA
                                if (receivesFromNeighbor) {
                                    connections.push_back({
                                        tileOp,
                                        opToLocation[tileOp],
                                        receiveSlaveDir,  // Receive from neighbor (slave)
                                        receiveSlaveIdx,
                                        localDmaDir,  // To DMA (master)
                                        localDmaIdx
                                    });
                                    std::cout << "  PKT connection (stream->DMA) at tile " << opToLocation[tileOp].toString()
                                              << ": " << receiveSlaveDir << "[" << receiveSlaveIdx << "] -> "
                                              << "DMA[" << localDmaIdx << "]" << std::endl;
                                }
                            } else {
                                // Case 3: Pure stream-to-stream forwarding (no DMA involved)
                                if (receivesFromNeighbor && forwardsToNeighbor) {
                                    connections.push_back({
                                        tileOp,
                                        opToLocation[tileOp],
                                        receiveSlaveDir,
                                        receiveSlaveIdx,
                                        forwardMasterDir,
                                        forwardMasterIdx
                                    });
                                    std::cout << "  PKT connection (stream->stream) at tile " << opToLocation[tileOp].toString()
                                              << ": " << receiveSlaveDir << "[" << receiveSlaveIdx << "] -> "
                                              << forwardMasterDir << "[" << forwardMasterIdx << "]" << std::endl;
                                }
                            }
                        }
                    }
                }
            });
            
            // **NEW: Third pass - infer DMA output connections for packet switches with NONE forward**
            // This handles the case where a tile has DMA but forwardmasterdirection = "NONE"
            // We infer the output direction by finding neighbors that receive from this tile
            routingCreateOp->walk([&](Operation* innerOp) {
                if (auto pktConnectOp = dyn_cast<routinghw::ConnectStreamPktSwitchPort>(innerOp)) {
                    if (pktConnectOp->getNumOperands() > 0) {
                        Operation* tileOp = pktConnectOp->getOperand(0).getDefiningOp();
                        if (opToLocation.count(tileOp)) {
                            auto localDmaDir = pktConnectOp.getLocaldmadirection().str();
                            auto forwardMasterDir = pktConnectOp.getForwardmasterdirection().str();
                            int localDmaIdx = pktConnectOp.getLocaldmaportidx();
                            
                            // Only process tiles with DMA and no explicit forward direction
                            if (localDmaDir == "DMA" && forwardMasterDir == "NONE") {
                                TileLocation dmaLoc = opToLocation[tileOp];
                                std::cout << "  Checking DMA tile at " << dmaLoc.toString() 
                                          << " for implicit output connections..." << std::endl;
                                
                                // Check all 4 directions for tiles that receive from this DMA tile
                                std::vector<std::string> directions = {"NORTH", "SOUTH", "EAST", "WEST"};
                                
                                for (const auto& dir : directions) {
                                    TileLocation neighborLoc = getNeighbor(dmaLoc, dir);
                                    if (!neighborLoc.isValid() || !tileExists(neighborLoc)) continue;
                                    
                                    // What direction does neighbor use to receive from us?
                                    std::string oppositeDir = "";
                                    if (dir == "NORTH") oppositeDir = "SOUTH";
                                    else if (dir == "SOUTH") oppositeDir = "NORTH";
                                    else if (dir == "EAST") oppositeDir = "WEST";
                                    else if (dir == "WEST") oppositeDir = "EAST";
                                    
                                    // Check if neighbor has a connection receiving from oppositeDir
                                    bool foundReceiver = false;
                                    routingCreateOp->walk([&](Operation* checkOp) {
                                        if (auto checkConnect = dyn_cast<routinghw::ConnectStreamSingleSwitchPort>(checkOp)) {
                                            if (checkConnect->getNumOperands() > 0) {
                                                Operation* checkTileOp = checkConnect->getOperand(0).getDefiningOp();
                                                if (opToLocation.count(checkTileOp) && 
                                                    opToLocation[checkTileOp] == neighborLoc) {
                                                    auto slaveDir = checkConnect.getSlaveportdirection().str();
                                                    if (slaveDir == oppositeDir) {
                                                        foundReceiver = true;
                                                    }
                                                }
                                            }
                                        }
                                    });
                                    
                                    if (foundReceiver) {
                                        // Found a neighbor that receives from this DMA tile
                                        // Create the inferred DMA output connection
                                        connections.push_back({
                                            tileOp,
                                            dmaLoc,
                                            localDmaDir,  // DMA as source
                                            localDmaIdx,
                                            dir,  // Direction to neighbor
                                            0  // Use index 0 for inferred connection
                                        });
                                        std::cout << "  ✓ Inferred DMA output: " << dmaLoc.toString()
                                                  << " DMA[" << localDmaIdx << "] -> " << dir 
                                                  << " to neighbor " << neighborLoc.toString() << std::endl;
                                        break;  // Found the output, stop checking other directions
                                    }
                                }
                            }
                        }
                    }
                }
            });
            
            if (!tiles.empty()) {
                tilesByRoutingCreate[routingCreateOp] = tiles;
                connectionsByRoutingCreate[routingCreateOp] = connections;
            }
        }
    });
    
    if (tilesByRoutingCreate.empty()) {
        std::cerr << "ERROR: No paths found in the module!" << std::endl;
        return;
    }
    
    std::cout << "Found " << tilesByRoutingCreate.size() << " routing path(s)" << std::endl;
    
    // Helper function to check if two tiles are contiguous (adjacent)
    auto isContiguousTiles = [](const TileLocation& tile1, const TileLocation& tile2) -> bool {
        int rowDiff = std::abs(tile2.row - tile1.row);
        int colDiff = std::abs(tile2.col - tile1.col);
        // Contiguous means: (same row and col distance is 1) OR (same col and row distance is 1)
        return (rowDiff == 1 && colDiff == 0) || (rowDiff == 0 && colDiff == 1);
    };
    
    // Verify each path
    int pathIndex = 0;
    bool allPathsPass = true;
    
    for (const auto& [routingOp, tiles] : tilesByRoutingCreate) {
        pathIndex++;
        std::cout << "\n--- Path " << pathIndex << " ---" << std::endl;
        
        auto& connections = connectionsByRoutingCreate[routingOp];
        
        // ========== STEP 1: Build map for all tile locations and operations ==========
        std::map<TileLocation, TileInfo> tileMap;
        // Changed from map to multimap to support multiple connections per tile
        std::multimap<TileLocation, PortConnection> connectionMap;
        
        for (const auto& tile : tiles) {
            tileMap[tile.loc] = tile;
            std::cout << "Tile at " << tile.loc.toString() << ": " << tile.comments << std::endl;
        }
        
        for (const auto& conn : connections) {
            connectionMap.insert({conn.tileLoc, conn});
            std::cout << "Connection at " << conn.tileLoc.toString() << ": "
                      << conn.slavePortDir << "[" << conn.slavePortIdx << "] -> "
                      << conn.masterPortDir << "[" << conn.masterPortIdx << "]" << std::endl;
        }
        
        // ========== STEP 2: Find the start point ==========
        // Start point: a tile that has no incoming connection from its slave port direction
        std::cout << "\n--- Step 2: Finding start point ---" << std::endl;
        
        // Build a set of tiles that receive data (have incoming connections)
        std::set<TileLocation> tilesWithIncoming;
        for (const auto& conn : connections) {
            // Master port sends data in masterPortDir direction
            // Skip if master is DMA (not a tile-to-tile connection)
            if (conn.masterPortDir == "DMA") continue;
            
            TileLocation targetTile = getNeighbor(conn.tileLoc, conn.masterPortDir);
            if (targetTile.isValid() && tileMap.count(targetTile)) {
                tilesWithIncoming.insert(targetTile);
                std::cout << "  Tile " << targetTile.toString() 
                          << " receives from " << conn.tileLoc.toString() 
                          << " via " << conn.masterPortDir << std::endl;
            }
        }
        
        // Start tiles are those with connections but no incoming data
        std::vector<TileLocation> startTiles;
        // End tiles are those with no outgoing connections
        std::vector<TileLocation> endTiles;
        
        std::set<TileLocation> tilesWithConnections;
        for (const auto& conn : connections) {
            tilesWithConnections.insert(conn.tileLoc);
        }
        
        // Find tiles with outgoing connections (excluding DMA-only)
        std::set<TileLocation> tilesWithOutgoing;
        for (const auto& conn : connections) {
            if (conn.masterPortDir != "DMA") {
                tilesWithOutgoing.insert(conn.tileLoc);
                if (tilesWithIncoming.find(conn.tileLoc) == tilesWithIncoming.end()) {
                    // This tile sends but doesn't receive - it's a start point
                    if (std::find(startTiles.begin(), startTiles.end(), conn.tileLoc) == startTiles.end()) {
                        startTiles.push_back(conn.tileLoc);
                        std::cout << "  Found start tile: " << conn.tileLoc.toString() << std::endl;
                    }
                }
            }
        }
        
        // Find end tiles (tiles that receive but have no outgoing tile-to-tile connections)
        for (const auto& [loc, tileInfo] : tileMap) {
            if (tilesWithIncoming.find(loc) != tilesWithIncoming.end() &&
                tilesWithOutgoing.find(loc) == tilesWithOutgoing.end()) {
                endTiles.push_back(loc);
                std::cout << "  Found end tile: " << loc.toString() << std::endl;
            }
        }
        
        // Validate: exactly one start point and one end point
        bool hasValidStartEnd = true;
        
        if (startTiles.size() == 0) {
            std::cerr << "✗ ERROR: No start point found for path " << pathIndex << "!" << std::endl;
            allPathsPass = false;
            hasValidStartEnd = false;
        } else if (startTiles.size() > 1) {
            std::cerr << "✗ ERROR: Path " << pathIndex << " has " << startTiles.size() 
                      << " start points (expected 1)!" << std::endl;
            std::cerr << "  Start points: ";
            for (size_t i = 0; i < startTiles.size(); ++i) {
                std::cerr << startTiles[i].toString();
                if (i < startTiles.size() - 1) std::cerr << ", ";
            }
            std::cerr << std::endl;
            allPathsPass = false;
            // Don't set hasValidStartEnd = false, we'll still trace all paths
        }
        
        if (endTiles.size() == 0) {
            std::cerr << "✗ ERROR: No end point found for path " << pathIndex << "!" << std::endl;
            allPathsPass = false;
            hasValidStartEnd = false;
        } else if (endTiles.size() > 1) {
            std::cerr << "✗ ERROR: Path " << pathIndex << " has " << endTiles.size() 
                      << " end points (expected 1)!" << std::endl;
            std::cerr << "  End points: ";
            for (size_t i = 0; i < endTiles.size(); ++i) {
                std::cerr << endTiles[i].toString();
                if (i < endTiles.size() - 1) std::cerr << ", ";
            }
            std::cerr << std::endl;
            allPathsPass = false;
            // Don't set hasValidStartEnd = false, we'll still trace all paths
        }
        
        if (!hasValidStartEnd || startTiles.empty()) {
            std::cout << "✗ FAIL: Path has invalid start/end point configuration!" << std::endl;
            continue;
        }
        
        if (startTiles.size() == 1 && endTiles.size() == 1) {
            std::cout << "✓ Valid path configuration: 1 start point, 1 end point" << std::endl;
        }
        
        // ========== STEP 3: Build path by following master port directions ==========
        // Process each start tile separately if there are multiple
        for (size_t startIdx = 0; startIdx < startTiles.size(); ++startIdx) {
            if (startTiles.size() > 1) {
                std::cout << "\n--- Step 3: Building path from start point " << (startIdx + 1) 
                          << " of " << startTiles.size() << " ---" << std::endl;
            } else {
                std::cout << "\n--- Step 3: Building path from start point ---" << std::endl;
            }
            
            std::vector<TileLocation> path;
            std::set<TileLocation> visited;
            TileLocation currentTile = startTiles[startIdx];
            
            path.push_back(currentTile);
            visited.insert(currentTile);
            std::cout << "Starting at: " << currentTile.toString() << std::endl;
            
            // Follow the path by looking at master port directions
            while (true) {
                // Find outgoing connection to next tile (not DMA)
                auto range = connectionMap.equal_range(currentTile);
                bool foundNext = false;
                
                for (auto it = range.first; it != range.second; ++it) {
                    const auto& conn = it->second;
                    
                    // Skip DMA connections - we're looking for tile-to-tile
                    if (conn.masterPortDir == "DMA") continue;
                    
                    // Find next tile based on master port direction
                    TileLocation nextTile = getNeighbor(currentTile, conn.masterPortDir);
                    
                    std::cout << "  Current tile " << currentTile.toString()
                              << " master port " << conn.masterPortDir 
                              << " points to " << nextTile.toString() << std::endl;
                    
                    // Check if next tile is valid and in our tile map
                    if (!nextTile.isValid() || !tileMap.count(nextTile)) {
                        std::cout << "  Next tile not found in tile map." << std::endl;
                        continue;
                    }
                    
                    // Check for cycles
                    if (visited.count(nextTile)) {
                        std::cout << "  Cycle detected at " << nextTile.toString() << ", skipping." << std::endl;
                        continue;
                    }
                    
                    // ========== STEP 4: Verify contiguity ==========
                    if (!isContiguousTiles(currentTile, nextTile)) {
                        std::cout << "  ✗ ERROR: Non-contiguous hop from " << currentTile.toString()
                                  << " to " << nextTile.toString() << std::endl;
                    } else {
                        std::cout << "  ✓ Contiguous hop to " << nextTile.toString() << std::endl;
                    }
                    
                    path.push_back(nextTile);
                    visited.insert(nextTile);
                    currentTile = nextTile;
                    foundNext = true;
                    break;  // Follow first valid outgoing connection
                }
                
                if (!foundNext) {
                    std::cout << "  No more outgoing connections, ending path." << std::endl;
                    break;
                }
            }
            
            // ========== Path verification for this start point ==========
            std::cout << "\n--- Path Verification";
            if (startTiles.size() > 1) {
                std::cout << " (Start Point " << (startIdx + 1) << ")";
            }
            std::cout << " ---" << std::endl;
            
            std::cout << "Reconstructed path (" << path.size() << " tiles): ";
            for (size_t i = 0; i < path.size(); ++i) {
                std::cout << path[i].toString();
                if (i < path.size() - 1) std::cout << " -> ";
            }
            std::cout << std::endl;
            
            // Print detailed path with port information
            std::cout << "\nDetailed path with port connections:" << std::endl;
            for (size_t i = 0; i < path.size(); ++i) {
                const TileLocation& tile = path[i];
                std::cout << "  [" << i << "] " << tile.toString();
                
                // Print all connection info for this tile
                auto range = connectionMap.equal_range(tile);
                if (range.first != range.second) {
                    std::cout << "\n      Connections:";
                    for (auto it = range.first; it != range.second; ++it) {
                        const auto& conn = it->second;
                        std::cout << "\n        " << conn.slavePortDir << "[" << conn.slavePortIdx << "]"
                                  << " -> " << conn.masterPortDir << "[" << conn.masterPortIdx << "]";
                    }
                }
                std::cout << std::endl;
            }
            
            // Verify all hops are contiguous
            bool isContiguous = true;
            std::vector<std::string> contiguityErrors;
            
            for (size_t i = 0; i < path.size() - 1; ++i) {
                const TileLocation& current = path[i];
                const TileLocation& next = path[i + 1];
                
                if (!isContiguousTiles(current, next)) {
                    isContiguous = false;
                    std::ostringstream error;
                    error << "Non-contiguous hop from " << current.toString()
                          << " to " << next.toString();
                    contiguityErrors.push_back(error.str());
                }
            }
            
            // Report results
            if (!path.empty()) {
                std::cout << "\nStarting location: " << path[0].toString() << std::endl;
                std::cout << "Ending location: " << path.back().toString() << std::endl;
            }
            
            if (isContiguous && !path.empty()) {
                std::cout << "✓ Contiguity check: Path is contiguous!" << std::endl;
            } else {
                std::cout << "✗ Contiguity check: ";
                if (path.empty()) {
                    std::cout << "Could not reconstruct path!" << std::endl;
                } else {
                    std::cout << "Path has non-contiguous hops!" << std::endl;
                    for (const auto& error : contiguityErrors) {
                        std::cout << "  - " << error << std::endl;
                    }
                }
                allPathsPass = false;
            }
        }
        
        // Final result for this routing path
        if (startTiles.size() == 1 && endTiles.size() == 1) {
            std::cout << "\n✓ PASS: Path is valid and contiguous!" << std::endl;
        } else {
            std::cout << "\n✗ FAIL: Path topology is invalid (multiple start/end points)" << std::endl;
            allPathsPass = false;
        }
    }
    
    // Overall summary
    std::cout << "\n--- Overall Test Results ---" << std::endl;
    std::cout << "Total paths tested: " << tilesByRoutingCreate.size() << std::endl;
    if (allPathsPass) {
        std::cout << "✓ PASS: All paths are contiguous!" << std::endl;
    } else {
        std::cout << "✗ FAIL: One or more paths have issues!" << std::endl;
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
}

// ---------------------------------------------------------------------------
// IR dialect validation
// ---------------------------------------------------------------------------

// Pipeline stage order (earliest to latest):
//   routing -> dmap -> dmaphop -> dfscheblueprint -> dfschedule
//   routing -> routinghw  (alternative path)
//
// If the user says "--parse routing <file>", the file must contain routing
// dialect ops and must NOT contain ops from a later stage.

// Detect whether a post-branch module is host or kernel IR.
// Returns "host", "kernel", or "unknown".
static std::string detectHostOrKernel(mlir::ModuleOp module) {
    bool isHost = false, isKernel = false;
    module.walk([&](Operation *op) {
        auto name = op->getName().getStringRef();
        // dfschedule host-path ops
        if (name == "dfschedule.declaretile" ||
            name == "dfschedule.config.dma_bd")
            isHost = true;
        // dfschedule kernel-path ops
        if (name == "dfschedule.kernel_config_def")
            isKernel = true;
        // emitc-level detection
        if (name == "emitc.call_opaque") {
            if (auto callee = op->getAttrOfType<StringAttr>("callee")) {
                if (callee.getValue().starts_with("__Runtime_"))
                    isHost = true;
            }
        }
        if (name == "emitc.verbatim") {
            if (auto valAttr = op->getAttrOfType<StringAttr>("value")) {
                auto text = valAttr.getValue();
                if (text.contains("aie_api/aie.hpp"))
                    isKernel = true;
                if (text.contains("__Runtime_"))
                    isHost = true;
            }
        }
    });
    if (isHost && !isKernel) return "host";
    if (isKernel && !isHost) return "kernel";
    return "unknown";
}

// Auto-detect the pipeline stage from the dialects present in the module.
// Returns the stage name (routing, dmap, dmaphop, dfscheblueprint, dfschedule, emitc)
// or "" if unrecognizable.
static std::string autoDetectStage(mlir::ModuleOp module) {
    std::set<std::string> foundDialects;
    bool hasEmitC = false;
    module.walk([&](Operation *op) {
        if (auto *dialect = op->getDialect()) {
            std::string ns = dialect->getNamespace().str();
            if (ns == "routing" || ns == "routinghw" || ns == "dmap" ||
                ns == "dmaphop" || ns == "dfscheblueprint" ||
                ns == "dfschedule") {
                foundDialects.insert(ns);
            }
            if (ns == "emitc")
                hasEmitC = true;
        }
    });

    // Detect based on the "latest" dialect present (pipeline order)
    if (foundDialects.count("dfschedule"))     return "dfschedule";
    if (foundDialects.count("dfscheblueprint")) return "dfscheblueprint";
    if (foundDialects.count("dmaphop"))         return "dmaphop";
    if (foundDialects.count("routinghw"))       return "routinghw";
    if (foundDialects.count("dmap"))            return "dmap";
    if (foundDialects.count("routing")) {
        // Check if still stage 0 (has scf.for) or stage 1 (unrolled)
        bool hasScfFor = false;
        module.walk([&](Operation *op) {
            if (op->getName().getStringRef() == "scf.for")
                hasScfFor = true;
        });
        return "routing"; // stage 0 or 1 — both map to "routing" for entry
    }
    if (hasEmitC && foundDialects.empty())      return "emitc";
    return "";
}

static bool validateIRDialect(mlir::ModuleOp module,
                              const std::string &expectedMode,
                              const std::string &filepath,
                              std::string &detectedMode) {
    detectedMode = autoDetectStage(module);

    if (detectedMode.empty()) {
        llvm::errs() << "ERROR: Could not detect any known pipeline dialect in IR.\n"
                     << "  File: " << filepath << "\n";
        return false;
    }

    // If expected mode matches detected mode, we're good
    if (expectedMode == detectedMode) {
        // Extra check for routing stage 0: scf.for must still be present
        if (expectedMode == "routing") {
            bool hasScfFor = false;
            module.walk([&](Operation *op) {
                if (op->getName().getStringRef() == "scf.for")
                    hasScfFor = true;
            });
            if (!hasScfFor) {
                // Stage 1 (post-unrolling) — still has routing but no scf.for
                // Treat as "dmap" entry point (startStage=1)
                std::cout << "Note: IR has routing dialect but scf.for already unrolled (stage 1).\n"
                          << "  Adjusting to startStage=1." << std::endl;
                detectedMode = "routing_unrolled";
            }
        }
        std::cout << "IR validation passed: file contains " << detectedMode
                  << " dialect ops." << std::endl;
        return true;
    }

    // Mismatch: auto-correct with a warning
    std::cout << "WARNING: --parse " << expectedMode
              << " specified, but IR contains " << detectedMode
              << " dialect.\n"
              << "  File: " << filepath << "\n"
              << "  Auto-correcting to --parse " << detectedMode << ".\n";
    return true;
}

// ---------------------------------------------------------------------------
// IR file parser helper
// ---------------------------------------------------------------------------

static mlir::OwningOpRef<mlir::ModuleOp> loadModuleFromFile(
    MLIRContext &ctx, const std::string &filepath) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filepath);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open file: " << filepath << ": "
                     << ec.message() << "\n";
        return nullptr;
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

    // Print diagnostics directly to stderr so they're always visible
    mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &ctx, llvm::errs());

    auto result = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
    if (!result) {
        llvm::errs() << "Failed to parse: " << filepath << "\n";
    }
    return result;
}

void routingtoroutinghw(mlir::ModuleOp module1 = nullptr) {
    MLIRContext ctx;

    routingmanager mtest;
    routinghwmanager mtesthw;
    mtesthw.loaddialect(&ctx);
    mtest.loaddialect(&ctx);

    ctx.getOrLoadDialect<arith::ArithDialect>();

    if (!module1) {
        module1 = mtest.ops_testNew(&ctx, 1, "routing");
    }

    std::string irDir = setupIRDir("simplerouting");
    RoutingTopology rtopology(g_aieGen);
    int stage = 0;

    dumpIRToFile(module1, irDir, stage++, "initial");

    if (!runSinglePass(ctx, module1, std::make_unique<RoutingUnrollingLowerPass>(), irDir, stage,
                       "RoutingUnrollingLowerPass"))
        return;
    if (!runSinglePass(ctx, module1, std::make_unique<RoutingLowerPass>(rtopology), irDir, stage, "RoutingLowerPass"))
        return;
    if (!runSinglePass(ctx, module1, std::make_unique<RoutingHWLowerPass>(rtopology), irDir, stage,
                       "RoutingHWLowerPass"))
        return;
    if (!runSinglePass(ctx, module1, std::make_unique<RoutingDeadArgPass>(), irDir, stage, "RoutingDeadArgPass"))
        return;
    if (!runSinglePass(ctx, module1, std::make_unique<RoutingConstantFoldPass>(g_aieGen), irDir, stage,
                       "RoutingConstantFoldPass"))
        return;
    if (!runSinglePass(ctx, module1, mlir::createCanonicalizerPass(), irDir, stage, "CanonicalizerPass"))
        return;

    const std::string worklocalDir = setupWorklocalDir();
    if (worklocalDir.empty()) return;

    std::string routingPath = worklocalDir + "/routing_hw.cc";
    std::error_code routingEC;
    llvm::raw_fd_ostream routingStream(routingPath, routingEC, llvm::sys::fs::OF_None);
    if (routingEC) {
        llvm::errs() << "Failed to open " << routingPath << ": " << routingEC.message() << "\n";
        return;
    }
    mlir::LogicalResult result = mlir::emitc::translateToCpp(module1, routingStream);
    routingStream.close();
    if (failed(result)) {
        llvm::errs() << "Failed to translate routing MLIR to C++.\n";
        return;
    }
    std::cout << "Routing code written to " << routingPath << std::endl;
}
void routingtodmap(const std::string &irFilepath = "", int startStage = 0) {
    MLIRContext ctx;

    routingmanager mtest;
    routinghwmanager mtesthw;
    dmapmanager mdmaptest;
    dmaphopmanager dmaphoptest;
    mtesthw.loaddialect(&ctx);
    mtest.loaddialect(&ctx);
    mdmaptest.loaddialect(&ctx);
    dmaphoptest.loaddialect(&ctx);

    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx.getOrLoadDialect<mlir::tensor::TensorDialect>();
    ctx.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    ctx.getOrLoadDialect<mlir::emitc::EmitCDialect>();
    ctx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

    // Create module: either parse from file or build programmatically
    mlir::OwningOpRef<mlir::ModuleOp> parsedModule;
    mlir::ModuleOp module1;
    if (!irFilepath.empty()) {
        parsedModule = loadModuleFromFile(ctx, irFilepath);
        if (!parsedModule) return;
        module1 = *parsedModule;
        std::cout << "Parsed module from " << irFilepath << std::endl;
    } else {
        module1 = mtest.ops_testNew(&ctx, 1, "routing");
    }

    std::string irDir = setupIRDir("dmap");
    RoutingTopology rtopology(g_aieGen);
    int stage = startStage;

    dumpIRToFile(module1, irDir, stage++, "initial");

    // Stages 0-2 are shared with dfschedule; stage 3+ diverges
    if (startStage <= 0) {
        if (!runSinglePass(ctx, module1, std::make_unique<RoutingUnrollingLowerPass>(), irDir, stage,
                           "RoutingUnrollingLowerPass"))
            return;
    }
    if (startStage <= 1) {
        if (!runSinglePass(ctx, module1, std::make_unique<RoutingToDmapPass>(rtopology), irDir, stage,
                           "RoutingToDmapPass"))
            return;
    }
    if (startStage <= 2) {
        if (!runSinglePass(ctx, module1, std::make_unique<DmapToDmaphopPass>(rtopology), irDir, stage,
                           "DmapToDmaphopPass"))
            return;
    }
    // Stage 3+: routing.cc-specific path (diverges from dfschedule here)
    if (!runSinglePass(ctx, module1, std::make_unique<DmaphopToRoutinghwPass>(rtopology), irDir, stage,
                       "DmaphopToRoutinghwPass"))
        return;
    if (!runSinglePass(ctx, module1, std::make_unique<RoutingHWVerifyPass>(), irDir, stage, "RoutingHWVerifyPass"))
        return;
    if (!runSinglePass(ctx, module1, std::make_unique<RoutingHWLowerPass>(rtopology), irDir, stage,
                       "RoutingHWLowerPass"))
        return;
    if (!runSinglePass(ctx, module1, std::make_unique<RoutingDeadArgPass>(), irDir, stage, "RoutingDeadArgPass"))
        return;
    if (!runSinglePass(ctx, module1, std::make_unique<RoutingConstantFoldPass>(g_aieGen), irDir, stage,
                       "RoutingConstantFoldPass"))
        return;
    if (!runSinglePass(ctx, module1, mlir::createCanonicalizerPass(), irDir, stage, "CanonicalizerPass"))
        return;

    const std::string worklocalDir = setupWorklocalDir();
    if (worklocalDir.empty()) return;

    std::string routingPath = worklocalDir + "/routing.cc";
    std::error_code routingEC;
    llvm::raw_fd_ostream routingStream(routingPath, routingEC, llvm::sys::fs::OF_None);
    if (routingEC) {
        llvm::errs() << "Failed to open " << routingPath << ": " << routingEC.message() << "\n";
        return;
    }
    mlir::LogicalResult result = mlir::emitc::translateToCpp(module1, routingStream);
    routingStream.close();
    if (failed(result)) {
        llvm::errs() << "Failed to translate routing MLIR to C++.\n";
        return;
    }
    std::cout << "Routing code written to " << routingPath << std::endl;
}

void routingtodfschedule(const std::string &irFilepath = "", int startStage = 0) {
    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);

    // Default tensor config for GEMM: 2 inputs + 1 output
    // Use 64x64 matrices to match simplematmul2.cc (tile_m=8 sub-tiling)
    std::vector<TensorParam> tensors = {
        {{64, 64}, 8, true},  // input A (window_in_0)
        {{64, 64}, 8, true},  // input B (window_in_1)
        {{64, 64}, 8, false}, // output C (window_out_0)
    };

    // Build SplitModel with per-tensor pp_depth from CLI
    SplitModel splitModel = SplitModel::gemm();
    // Override output tensor's pp_depth from CLI --output-pp-depth
    if (g_outputPpDepth != 2 && splitModel.tensorSplits.size() >= 3) {
        splitModel.tensorSplits[2].pingPong = g_outputPpDepth;
        std::cout << "Output tensor pp_depth overridden to " << g_outputPpDepth << std::endl;
    }

    // Create module: either parse from file or build programmatically
    mlir::OwningOpRef<mlir::ModuleOp> parsedModule;
    mlir::ModuleOp module;
    if (!irFilepath.empty()) {
        parsedModule = loadModuleFromFile(ctx, irFilepath);
        if (!parsedModule) return;
        module = *parsedModule;
        std::cout << "Parsed module from " << irFilepath << std::endl;
    } else {
        module = TilingLinalgPipeline::buildRoutingIR(ctx, 4, 4, tensors, splitModel);
    }

    // Set routing module attributes for M/N sub-tiling (matching simplematmul2.cc)
    // These are normally set by aiehlc.cc from SpatialPolicy, but buildRoutingIR
    // doesn't set them. For 64x64 matmul on 4x4 mesh with tile_m=8, tile_k=16:
    //   tileRows = 64/4 = 16, tileCols = 64/4 = 16
    //   tileM = 8 (sub-tile), tileN = 8 (sub-tile)
    //   effectiveK = 16, fullK = 64, kRounds = 4
    //   mRounds = tileRows/tileM = 2, nRounds = tileCols/tileN = 2
    if (irFilepath.empty()) {
        mlir::OpBuilder attrB(&ctx);
        module->setAttr("routing.tile_m", attrB.getI64IntegerAttr(8));
        module->setAttr("routing.tile_rows", attrB.getI64IntegerAttr(16));
        module->setAttr("routing.tile_n", attrB.getI64IntegerAttr(8));
        module->setAttr("routing.tile_cols", attrB.getI64IntegerAttr(16));
        module->setAttr("routing.effective_k", attrB.getI64IntegerAttr(16));
        module->setAttr("routing.full_k", attrB.getI64IntegerAttr(64));
        module->setAttr("routing.k_rounds", attrB.getI64IntegerAttr(4));
        module->setAttr("routing.m_rounds", attrB.getI64IntegerAttr(2));
        module->setAttr("routing.n_rounds", attrB.getI64IntegerAttr(2));
        module->setAttr("routing.iter_policy", attrB.getStringAttr("m_outer_n_inner"));
    }

    // When startStage==0 and using the default pipeline, delegate to
    // TilingLinalgPipeline::runPipeline which handles the full flow.
    if (startStage == 0) {
        std::string outputDir = setupWorklocalDir();
        if (outputDir.empty()) return;

        if (!TilingLinalgPipeline::runPipeline(ctx, module, outputDir,
                                               /*userKernelBody=*/"", /*userKernelFuncName=*/"",
                                               /*runtimeDebugLevel=*/-1, /*userRewrittenSource=*/"", tensors,
                                               /*maxPingPongBytes=*/4096, g_aieGen)) {
            llvm::errs() << "TilingLinalgPipeline::runPipeline failed!\n";
        }
        return;
    }

    // Stage-aware pipeline for --parse mode (startStage > 0)
    RoutingTopology rtopology(g_aieGen);

    // Determine which paths to run based on startStage
    // Stages 0-4: shared path (can produce host + kernel + bcf/prx)
    // Stages 5-9: host-only path (host.cc + bcf/prx)
    // Stages 10-11: kernel-only path (kernel.cc)
    bool doHostPath = (startStage <= 9);
    bool doKernelPath = (startStage <= 4 || startStage >= 10);

    // Print output summary
    std::cout << "Stage: startStage=" << startStage << std::endl;
    std::cout << "Will produce:";
    if (doHostPath) std::cout << " host.cc, aieml.bcf, aieml.prx";
    if (doHostPath && doKernelPath) std::cout << ",";
    if (doKernelPath) std::cout << " kernel.cc";
    std::cout << std::endl;

    std::string irDir = setupIRDir("dfschedule");
    int stage = startStage;

    dumpIRToFile(module, irDir, stage++, "initial");

    // Phase 1: shared path (routing -> dmap -> dmaphop -> dfscheblueprint)
    if (startStage <= 0) {
        if (!runSinglePass(ctx, module, std::make_unique<RoutingUnrollingLowerPass>(), irDir, stage,
                           "RoutingUnrollingLowerPass"))
            return;
    }
    if (startStage <= 1) {
        if (!runSinglePass(ctx, module, std::make_unique<RoutingToDmapPass>(rtopology), irDir, stage,
                           "RoutingToDmapPass"))
            return;
    }
    if (startStage <= 2) {
        if (!runSinglePass(ctx, module, std::make_unique<DmapToDmaphopPass>(rtopology), irDir, stage,
                           "DmapToDmaphopPass"))
            return;
        // Generate provenance map JSON after dmaphop IR is available
        {
            std::string worklocalDir = setupWorklocalDir();
            if (!worklocalDir.empty()) {
                auto provenancePass = std::make_unique<DmaphopProvenanceMapPass>(worklocalDir);
                runSinglePass(ctx, module, std::move(provenancePass), irDir, stage, "DmaphopProvenanceMapPass");
            }
        }
    }
    if (startStage <= 3) {
        if (!runSinglePass(ctx, module, std::make_unique<DmaphopTodfscheblueprintPass>(), irDir, stage,
                           "DmaphopTodfscheblueprintPass"))
            return;
    }

    // Branch point: clone for host and kernel paths (only when entering at shared stages)
    mlir::ModuleOp hostModule;
    mlir::ModuleOp kernelModule;

    if (startStage <= 4) {
        // Pre-branch or at branch point: clone for both paths
        if (doHostPath)
            hostModule = cast<ModuleOp>(module->clone());
        if (doKernelPath)
            kernelModule = cast<ModuleOp>(module->clone());
    } else if (startStage <= 9) {
        // Host-only stages: use module directly
        hostModule = module;
    } else {
        // Kernel-only stages: use module directly
        kernelModule = module;
    }

    // Initialize ResourceMgr singleton for CoreMemAllocator (BCF/PRX generation)
    // The singleton may already exist from a previous pass; init only creates once.
    if (doHostPath) {
        auto hwRes = makeResource(g_aieGen);
        ResourceMgr::init(std::move(hwRes));
    }

    // Pre-pipeline memory check: validate buffer requirements fit in tile data memory
    {
        auto hwResCheck = makeResource(g_aieGen);
        uint32_t usableBytes = hwResCheck->getUsableDataBytes();
        // Conservative estimate: each tensor needs ppDepth * maxPingPongBytes (4096 default)
        uint32_t maxPingPongBytes = 4096;
        uint32_t totalBufferBytes = 0;
        for (const auto &tp : tensors) {
            int ppDepth = 2; // default
            totalBufferBytes += ppDepth * maxPingPongBytes;
        }
        std::string errMsg;
        if (!hwResCheck->checkDataMemoryFits(totalBufferBytes, &errMsg)) {
            llvm::errs() << "[unitest] ERROR: Memory budget exceeded!\n"
                         << "  " << errMsg << "\n"
                         << "  numTensors=" << tensors.size() << " maxPingPongBytes=" << maxPingPongBytes << "\n";
        } else {
            std::cout << "[unitest] Memory check passed: estimated " << totalBufferBytes << " bytes per tile, limit "
                      << usableBytes << " bytes" << std::endl;
        }
    }

    // Phase 2: host path (blueprint -> schedule -> API -> EmitC)
    if (doHostPath) {
        if (startStage <= 4) {
            if (!runSinglePass(ctx, hostModule, std::make_unique<mlir::BlueprintToSchedulePass>(0.5, 4096, g_aieGen),
                               irDir, stage, "BlueprintToSchedulePass"))
                return;
        }
        if (startStage <= 5) {
            if (!runSinglePass(ctx, hostModule, std::make_unique<mlir::ScheduleCanonicalizePass>(), irDir, stage,
                               "ScheduleCanonicalizePass"))
                return;
            // Generate low-level dfschedule provenance map after ScheduleCanonicalizePass
            {
                std::string worklocalDir = setupWorklocalDir();
                if (!worklocalDir.empty()) {
                    auto dfscheProvenancePass = std::make_unique<DfscheduleProvenanceMapPass>(worklocalDir);
                    runSinglePass(ctx, hostModule, std::move(dfscheProvenancePass), irDir, stage,
                                  "DfscheduleProvenanceMapPass");
                }
            }
        }
        if (startStage <= 6) {
            if (!runSinglePass(ctx, hostModule, std::make_unique<mlir::DfscheduleToApiPass>(/*enableDebug=*/true),
                               irDir, stage, "DfscheduleToApiPass"))
                return;
        }
        if (startStage <= 7) {
            if (!runSinglePass(ctx, hostModule, mlir::createCanonicalizerPass(), irDir, stage, "CanonicalizerPass"))
                return;
        }
        if (startStage <= 8) {
            if (!runSinglePass(ctx, hostModule, std::make_unique<RoutingConstantFoldPass>(g_aieGen), irDir, stage,
                               "RoutingConstantFoldPass"))
                return;
        }
    }

    // Phase 3: kernel path (blueprint -> kernel schedule -> kernel API)
    if (doKernelPath) {
        if (startStage <= 4) {
            if (!runSinglePass(ctx, kernelModule, std::make_unique<mlir::BlueprintToScheduleKernelPass>(0.5, 4096),
                               irDir, stage, "BlueprintToScheduleKernelPass"))
                return;
        }
        if (startStage <= 10) {
            if (!runSinglePass(ctx, kernelModule, std::make_unique<mlir::DfscheduleToKernelApiPass>(), irDir, stage,
                               "DfscheduleToKernelApiPass"))
                return;
        }
    }

    // Output generated C++ code
    const std::string worklocalDir = setupWorklocalDir();
    if (worklocalDir.empty()) return;

    if (doHostPath) {
        std::string hostPath = worklocalDir + "/host.cc";
        std::error_code hostEC;
        llvm::raw_fd_ostream hostStream(hostPath, hostEC, llvm::sys::fs::OF_None);
        if (hostEC) {
            llvm::errs() << "Failed to open " << hostPath << ": " << hostEC.message() << "\n";
            return;
        }
        mlir::LogicalResult result = mlir::emitc::translateToCpp(hostModule, hostStream);
        hostStream.close();
        if (failed(result)) {
            llvm::errs() << "Failed to translate host MLIR to C++.\n";
            return;
        }
        std::cout << "Host code written to " << hostPath << std::endl;
    }

    if (doKernelPath) {
        std::string kernelPath = worklocalDir + "/kernel.cc";
        std::error_code kernelEC;
        llvm::raw_fd_ostream kernelStream(kernelPath, kernelEC, llvm::sys::fs::OF_None);
        if (kernelEC) {
            llvm::errs() << "Failed to open " << kernelPath << ": " << kernelEC.message() << "\n";
            return;
        }
        mlir::LogicalResult result2 = mlir::emitc::translateToCpp(kernelModule, kernelStream);
        kernelStream.close();
        if (failed(result2)) {
            llvm::errs() << "Failed to translate kernel MLIR to C++.\n";
            return;
        }
        std::cout << "Kernel code written to " << kernelPath << std::endl;
    }

    // Phase 4: Generate BCF/PRX for kernel compilation
    // The CoreMemAllocator was populated during BlueprintToSchedulePass (host path)
    // with buffer symbol addresses. Generate BCF/PRX files that pin those symbols
    // so the kernel linker places buffers where the host DMA BDs expect.
    if (doHostPath) {
        try {
            auto &allocator = ResourceMgr::instance()->coreMemAllocator();
            const auto &allocations = allocator.getAllocations();

            if (!allocations.empty()) {
                TilingBcf bcf;
                bcf.setStack(0x70000, 0x2800);
                bcf.addReservedDMB(0x40000, 0x10000);

                // This crashes the simulator, commenting out for now
                // Reserve the last 2KB of DM for kernel_log.h klog() region
                // (DM absolute 0x7F800-0x7FFFF = DM offset 0xF800, 0x800 bytes)
                // bcf.addReservedDMB(0x7F800, 0x800);

                for (const auto &slot : allocations) {
                    bcf.addSymbol(slot.symbolName, slot.address);
                }

                std::string bcfPath = worklocalDir + "/aieml.bcf";
                if (bcf.exportToFile(bcfPath)) {
                    std::cout << "BCF written to " << bcfPath << std::endl;
                } else {
                    llvm::errs() << "Failed to write BCF to " << bcfPath << "\n";
                }

                TilingPrx prx("kernel", 22); // AIE2PS arch=22
                prx.setBcfPath("aieml.bcf");
                prx.setKernelLLPath("./build/");

                std::string prxPath = worklocalDir + "/aieml.prx";
                if (prx.exportToFile(prxPath)) {
                    std::cout << "PRX written to " << prxPath << std::endl;
                } else {
                    llvm::errs() << "Failed to write PRX to " << prxPath << "\n";
                }

                // Print allocation summary
                std::cout << "\n=== Core Memory Allocation Summary ===" << std::endl;
                for (const auto &slot : allocations) {
                    std::cout << "  " << slot.symbolName << " @ 0x" << std::hex << slot.address << " (size="
                              << std::dec << slot.size << " bytes)" << std::endl;
                }
                std::cout << "  Free space: " << allocator.getFreeSpace() << " bytes" << std::endl;
            } else {
                std::cout << "No buffer allocations found; skipping BCF/PRX generation." << std::endl;
            }
        } catch (...) {
            std::cout << "ResourceMgr not initialized; skipping BCF/PRX generation." << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-dimensional BD addressing test
// ---------------------------------------------------------------------------
// Runs the standard dfschedule pipeline, then creates a standalone dfschedule
// module with dim_strides/dim_wraps on a config.dma_bd op and runs the
// DfscheduleToApiPass to verify __Runtime_dma_bd_config_multidim emission.

void testConv2d() {
    std::cout << "\n=== Conv2d DmaTransform Integration Test ===" << std::endl;
    std::cout << "Tests that user-specified DmaTransform (im2col) flows through\n"
              << "buildRoutingIR -> module attr -> DmaphopTodfscheblueprintPass\n"
              << "-> FlowConfigOp shim_dim_strides/wraps -> host.cc XAie_DmaSetMultiDimAddr\n"
              << std::endl;

    // Conv2d parameters: Input[8,8,1], Filter[3,3,1,1], stride=1, pad=0
    // Im2col GEMM: A[OH*OW, KH*KW*C] = [36, 9], B[9, 1], C[36, 1]
    int H = 8, W = 8, C = 1, KH = 3, KW = 3, S = 1, P = 0, F = 1;
    int OH = (H + 2 * P - KH) / S + 1; // 6
    int OW = (W + 2 * P - KW) / S + 1; // 6
    int M_dim = OH * OW;               // 36
    int K_dim = KH * KW * C;           // 9
    int N_dim = F;                     // 1

    std::cout << "Conv2d params: H=" << H << " W=" << W << " C=" << C << " KH=" << KH << " KW=" << KW << " S=" << S
              << " P=" << P << " OH=" << OH << " OW=" << OW << "\nGEMM dims: M=" << M_dim << " K=" << K_dim
              << " N=" << N_dim << std::endl;

    // Build im2col DmaAddressing (matches DmaTransform::im2col factory, C-aware)
    DmaAddressing im2col;
    im2col.dims = {{1, KW * C}, {W * C, KH}, {S * C, OW}};
    im2col.iter_step = W * C * S;
    im2col.iter_wrap = OH;

    std::cout << "DmaAddressing: dims=[";
    for (unsigned i = 0; i < im2col.dims.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << "{" << im2col.dims[i].first << "," << im2col.dims[i].second << "}";
    }
    std::cout << "] iter_step=" << im2col.iter_step << " iter_wrap=" << im2col.iter_wrap << std::endl;

    // Tensor params: A gets im2col DMA, B and C are flat
    std::vector<TensorParam> tensors = {
        {{M_dim, K_dim}, 8, true, im2col}, // input A (im2col)
        {{K_dim, N_dim}, 8, true, {}},     // input B (flat)
        {{M_dim, N_dim}, 8, false, {}},    // output C (flat)
    };

    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);

    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, 2, 2, tensors, SplitModel::gemm());

    // Verify module attribute was set
    auto shimDmaAttr = module->getAttrOfType<DictionaryAttr>("tensor_0.shim_dma");
    if (shimDmaAttr) {
        std::cout << "PASS: tensor_0.shim_dma module attribute present" << std::endl;
        auto strides = shimDmaAttr.getAs<ArrayAttr>("strides");
        auto wraps = shimDmaAttr.getAs<ArrayAttr>("wraps");
        if (strides && wraps) {
            std::cout << "  strides=[";
            for (unsigned i = 0; i < strides.size(); ++i) {
                if (i > 0)
                    std::cout << ",";
                std::cout << cast<IntegerAttr>(strides[i]).getInt();
            }
            std::cout << "] wraps=[";
            for (unsigned i = 0; i < wraps.size(); ++i) {
                if (i > 0)
                    std::cout << ",";
                std::cout << cast<IntegerAttr>(wraps[i]).getInt();
            }
            std::cout << "]" << std::endl;
        }
        auto iterStep = shimDmaAttr.getAs<IntegerAttr>("iter_step");
        auto iterWrap = shimDmaAttr.getAs<IntegerAttr>("iter_wrap");
        if (iterStep && iterWrap) {
            std::cout << "  iter_step=" << iterStep.getInt() << " iter_wrap=" << iterWrap.getInt() << std::endl;
        }
    } else {
        std::cerr << "FAIL: tensor_0.shim_dma module attribute NOT found" << std::endl;
        return;
    }

    // Verify tensor_1 (B) does NOT have shim_dma
    auto shimDmaB = module->getAttrOfType<DictionaryAttr>("tensor_1.shim_dma");
    if (!shimDmaB) {
        std::cout << "PASS: tensor_1 (flat) has no shim_dma attribute" << std::endl;
    } else {
        std::cerr << "FAIL: tensor_1 should not have shim_dma" << std::endl;
    }

    // Run full pipeline to generate host.cc and verify XAie_DmaSetMultiDimAddr
    std::string outputDir = setupWorklocalDir();
    if (outputDir.empty())
        return;

    if (!TilingLinalgPipeline::runPipeline(ctx, module, outputDir,
                                           /*userKernelBody=*/"", /*userKernelFuncName=*/"",
                                           /*runtimeDebugLevel=*/-1, /*userRewrittenSource=*/"", tensors,
                                           /*maxPingPongBytes=*/4096, g_aieGen)) {
        llvm::errs() << "TilingLinalgPipeline::runPipeline failed!\n";
        return;
    }

    std::cout << "\n=== Conv2d DmaTransform Test Complete ===" << std::endl;
    std::cout << "Check worklocal/host.cc for XAie_DmaSetMultiDimAddr with im2col strides/wraps." << std::endl;
}

void testConv2dSpatial() {
    std::cout << "\n=== Conv2d Spatial-Halo Integration Test ===" << std::endl;
    std::cout << "Tests that a spatial-halo A port (ConvTiling::spatial) flows through\n"
              << "buildRoutingIR -> tensor_N.halo module attr -> partitiontensor step attr\n"
              << "-> overlapping extract_data slices -> FLAT shim BDs (no SetMultiDimAddr).\n"
              << std::endl;

    // Conv2d params (ResNet-style 7x7 stem):
    // Input 224x224x4, Filter[7,7,4,64], stride=2. The IFM is pre-padded by
    // 3 on H and W BEFORE processing, so the test feeds the padded 230x230x4
    // tensor directly and uses P=0 in the output-size formula.
    //   H = W = 224 + 2*3 = 230, C = 4, F = 64
    //   OH = (230 - 7)/2 + 1 = 112, OW = 112
    //   oh_per_row = OH / R = 28
    //   halo_slice = (oh_per_row - 1) * S + KH = 27*2 + 7 = 61
    //   halo_step  = oh_per_row * S            = 28*2 = 56  (overlap = 5 = KH - S)
    //   raw A shape = [H, W*C] = [230, 920]; overlapping row-blocks of size 61x920
    int H = 230, W = 230, C = 4, KH = 7, KW = 7, S = 2, P = 0, F = 64;
    int R = 4;                                  // HW_ROWS (tile-rows along H)
    int OH = (H + 2 * P - KH) / S + 1;          // 112
    int OW = (W + 2 * P - KW) / S + 1;          // 112
    int oh_per_row = OH / R;                    // 28
    int halo_slice = (oh_per_row - 1) * S + KH; // 61
    int halo_step = oh_per_row * S;             // 56
    int raw_h = H;                              // 230
    int raw_wc = W * C;                         // 920
    int M_dim = OH * OW;                        // 12544 (im2col GEMM M, for C/B sizing)
    int K_dim = KH * KW * C;                    // 196
    int N_dim = F;                              // 64

    std::cout << "Conv2d params: H=" << H << " W=" << W << " C=" << C << " KH=" << KH << " KW=" << KW << " S=" << S
              << " P=" << P << " OH=" << OH << " OW=" << OW << " R=" << R << "\n"
              << "Spatial-halo: raw A=[" << raw_h << "," << raw_wc << "] halo_slice=" << halo_slice
              << " halo_step=" << halo_step << " (overlap=" << (halo_slice - halo_step) << ")" << std::endl;

    // Two-level (nested) halo: each halo_slice(61)-row L1 tile is further chunked
    // on-core into l2_rounds temporal rounds of l2_slice rows advancing by l2_step.
    // Coverage: (l2_rounds-1)*l2_step + l2_slice == halo_slice (61).
    int l2_slice = 19; // rows per on-core round
    int l2_step = 14;  // row stride between L2 rounds (overlap = 19-14 = 5)
    int l2_rounds = 4; // (4-1)*14 + 19 = 61 == halo_slice
    std::cout << "Two-level halo: l2_slice=" << l2_slice << " l2_step=" << l2_step << " l2_rounds=" << l2_rounds
              << " (coverage=" << ((l2_rounds - 1) * l2_step + l2_slice) << " == halo_slice=" << halo_slice << ")"
              << std::endl;

    // K-contraction accumulate split (independent of the H/row L2 halo above):
    // the K dim is chunked into k_rounds on-core accumulate rounds (plumbing
    // round-trip only; no kernel-round / dma_bd len wiring yet).
    int k_slice = 61 * 4, k_step = 56 * 4, k_rounds = 4; // K-accum example

    // Spatial-halo DmaAddressing on the A tensor.
    DmaAddressing halo;
    halo.mode = 1;
    halo.haloSlice = halo_slice;
    halo.haloStep = halo_step;
    halo.splitDim = 0;
    halo.l2Slice = l2_slice;
    halo.l2Step = l2_step;
    halo.l2Rounds = l2_rounds;
    halo.kSlice = k_slice;
    halo.kStep = k_step;
    halo.kRounds = k_rounds;
    // PADDED DDR row pitch (elements) — required by the K-accum single multi-dim BD
    // (D1 row stride = rowPitch, iter_step = l2_step*rowPitch). = raw_wc (= 920).
    halo.rowPitch = raw_wc;

    // Tensor params:
    //   A = raw [raw_h, raw_wc] with spatial-halo addressing (split along dim 0 by row)
    //   B = filter [K, N] col-split (flat)
    //   C = output [M, N] gather (flat)
    std::vector<TensorParam> tensors = {
        {{raw_h, raw_wc}, 8, true, halo} //, // input A (spatial halo)
        //{{K_dim, N_dim}, 8, true, {}},    // input B (flat)
        //{{M_dim, N_dim}, 8, false, {}},   // output C (flat)
    };

    // Mirror halo onto the A split desc so buildRoutingIR emits tensor_0.halo
    // and sets the partitiontensor step attr.
    SplitModel sm = SplitModel::gemm();
    if (!sm.tensorSplits.empty()) {
        sm.tensorSplits[0].haloMode = 1;
        sm.tensorSplits[0].haloSlice = halo_slice;
        sm.tensorSplits[0].haloStep = halo_step;
        sm.tensorSplits[0].splitDim = 0;
        sm.tensorSplits[0].haloL2Slice = l2_slice;
        sm.tensorSplits[0].haloL2Step = l2_step;
        sm.tensorSplits[0].haloL2Rounds = l2_rounds;
        sm.tensorSplits[0].kAccumSlice = k_slice;
        sm.tensorSplits[0].kAccumStep = k_step;
        sm.tensorSplits[0].kAccumRounds = k_rounds;
    }

    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);

    // Mesh R rows x 1 col so A is split along H into R overlapping row-blocks.
    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, R, 1, tensors, sm);
    // module.dump();
    // return;
    //  Verify tensor_0.halo module attribute was set
    auto haloAttr = module->getAttrOfType<DictionaryAttr>("tensor_0.halo");
    if (haloAttr) {
        std::cout << "PASS: tensor_0.halo module attribute present" << std::endl;
        if (auto slice = haloAttr.getAs<IntegerAttr>("slice"))
            std::cout << "  slice=" << slice.getInt() << std::endl;
        if (auto step = haloAttr.getAs<IntegerAttr>("step"))
            std::cout << "  step=" << step.getInt() << std::endl;
        if (auto sd = haloAttr.getAs<IntegerAttr>("split_dim"))
            std::cout << "  split_dim=" << sd.getInt() << std::endl;
        // Two-level (nested) halo L2 attrs.
        auto l2SliceA = haloAttr.getAs<IntegerAttr>("l2_slice");
        auto l2StepA = haloAttr.getAs<IntegerAttr>("l2_step");
        auto l2RoundsA = haloAttr.getAs<IntegerAttr>("l2_rounds");
        if (l2SliceA && l2StepA && l2RoundsA) {
            std::cout << "  l2_slice=" << l2SliceA.getInt() << " l2_step=" << l2StepA.getInt()
                      << " l2_rounds=" << l2RoundsA.getInt() << std::endl;
            if (l2SliceA.getInt() == l2_slice && l2StepA.getInt() == l2_step && l2RoundsA.getInt() == l2_rounds)
                std::cout << "PASS: tensor_0.halo carries correct L2 (nested) attrs" << std::endl;
            else
                std::cerr << "FAIL: tensor_0.halo L2 attrs mismatch (expected l2_slice=" << l2_slice
                          << " l2_step=" << l2_step << " l2_rounds=" << l2_rounds << ")" << std::endl;
        } else {
            std::cerr << "FAIL: tensor_0.halo missing L2 (l2_slice/l2_step/l2_rounds) attrs" << std::endl;
        }
        // K-contraction accumulate attrs (independent of the H/row L2 above).
        auto kSliceA = haloAttr.getAs<IntegerAttr>("k_slice");
        auto kStepA = haloAttr.getAs<IntegerAttr>("k_step");
        auto kRoundsA = haloAttr.getAs<IntegerAttr>("k_rounds");
        if (kSliceA && kStepA && kRoundsA) {
            std::cout << "  k_slice=" << kSliceA.getInt() << " k_step=" << kStepA.getInt()
                      << " k_rounds=" << kRoundsA.getInt() << std::endl;
            if (kSliceA.getInt() == k_slice && kStepA.getInt() == k_step && kRoundsA.getInt() == k_rounds)
                std::cout << "PASS: tensor_0.halo carries correct K-accum attrs" << std::endl;
            else
                std::cerr << "FAIL: tensor_0.halo K-accum attrs mismatch (expected k_slice=" << k_slice
                          << " k_step=" << k_step << " k_rounds=" << k_rounds << ")" << std::endl;
        } else {
            std::cerr << "FAIL: tensor_0.halo missing K-accum (k_slice/k_step/k_rounds) attrs" << std::endl;
        }
    } else {
        std::cerr << "FAIL: tensor_0.halo module attribute NOT found" << std::endl;
        return;
    }

    // Verify tensor_0 does NOT have a multi-dim shim_dma (must stay FLAT)
    auto shimDmaA = module->getAttrOfType<DictionaryAttr>("tensor_0.shim_dma");
    if (!shimDmaA) {
        std::cout << "PASS: tensor_0 (spatial-halo) has NO shim_dma attribute (flat BD)" << std::endl;
    } else {
        std::cerr << "FAIL: spatial-halo tensor_0 should not have shim_dma (would force multi-dim BD)" << std::endl;
    }

    // Run full pipeline to generate host.cc / kernel.cc into worklocal/.
    std::string outputDir = setupWorklocalDir();
    if (outputDir.empty())
        return;

    if (!TilingLinalgPipeline::runPipeline(ctx, module, outputDir,
                                           /*userKernelBody=*/"", /*userKernelFuncName=*/"",
                                           /*runtimeDebugLevel=*/-1, /*userRewrittenSource=*/"", tensors,
                                           /*maxPingPongBytes=*/4096, g_aieGen)) {
        llvm::errs() << "TilingLinalgPipeline::runPipeline failed!\n";
        return;
    }

    std::cout << "\n=== Conv2d Spatial-Halo Test Complete ===" << std::endl;
    std::cout << "Check worklocal/host.cc for FLAT A BDs (XAie_DmaSetAddrLen, no SetMultiDimAddr),\n"
              << "per-row DDR offsets {0, " << (halo_step * raw_wc) << "} (= i*halo_step*raw_wc), len "
              << (halo_slice * raw_wc) << " each." << std::endl;
}

void testDilatedConv2d() {
    std::cout << "\n=== Dilated Conv2d DmaTransform Test ===" << std::endl;
    std::cout << "Tests dilated_im2col DmaTransform with dilation=2.\n" << std::endl;

    // Dilated Conv2d: Input[8,8,1], Filter[3,3], stride=1, pad=0, dilation=2
    // OW = (8 + 0 - 2*(3-1) - 1) / 1 + 1 = (8 - 4 - 1)/1 + 1 = 4
    // OH = 4
    int H = 8, W = 8, C = 1, KH = 3, KW = 3, S = 1, P = 0, D = 2, F = 1;
    int OH = (H + 2 * P - D * (KH - 1) - 1) / S + 1; // 4
    int OW = (W + 2 * P - D * (KW - 1) - 1) / S + 1; // 4
    int M_dim = OH * OW;                             // 16
    int K_dim = KH * KW * C;                         // 9
    int N_dim = F;                                   // 1

    std::cout << "Dilated Conv2d params: H=" << H << " W=" << W << " C=" << C << " KH=" << KH << " KW=" << KW
              << " S=" << S << " P=" << P << " D=" << D << " OH=" << OH << " OW=" << OW << "\nGEMM dims: M=" << M_dim
              << " K=" << K_dim << " N=" << N_dim << std::endl;

    // dilated_im2col: dim0={D*C, KW}, dim1={W*C*D, KH}, dim2={S*C, OW}
    DmaAddressing dilated;
    dilated.dims = {{D * C, KW}, {W * C * D, KH}, {S * C, OW}};
    dilated.iter_step = W * C * S;
    dilated.iter_wrap = OH;

    std::cout << "DmaAddressing: dims=[";
    for (unsigned i = 0; i < dilated.dims.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << "{" << dilated.dims[i].first << "," << dilated.dims[i].second << "}";
    }
    std::cout << "] iter_step=" << dilated.iter_step << " iter_wrap=" << dilated.iter_wrap << std::endl;

    // Verify expected values
    bool pass = true;
    if (dilated.dims[0].first != 2 || dilated.dims[0].second != 3) {
        std::cerr << "FAIL: dim0 expected {2,3} got {" << dilated.dims[0].first << "," << dilated.dims[0].second << "}"
                  << std::endl;
        pass = false;
    }
    if (dilated.dims[1].first != 16 || dilated.dims[1].second != 3) {
        std::cerr << "FAIL: dim1 expected {16,3} got {" << dilated.dims[1].first << "," << dilated.dims[1].second << "}"
                  << std::endl;
        pass = false;
    }
    if (dilated.dims[2].first != 1 || dilated.dims[2].second != 4) {
        std::cerr << "FAIL: dim2 expected {1,4} got {" << dilated.dims[2].first << "," << dilated.dims[2].second << "}"
                  << std::endl;
        pass = false;
    }
    if (dilated.iter_step != 8) {
        std::cerr << "FAIL: iter_step expected 8 got " << dilated.iter_step << std::endl;
        pass = false;
    }
    if (dilated.iter_wrap != 4) {
        std::cerr << "FAIL: iter_wrap expected 4 got " << dilated.iter_wrap << std::endl;
        pass = false;
    }

    // Build module with this DmaAddressing
    std::vector<TensorParam> tensors = {
        {{M_dim, K_dim}, 8, true, dilated}, // input A (dilated im2col)
        {{K_dim, N_dim}, 8, true, {}},      // input B (flat)
        {{M_dim, N_dim}, 8, false, {}},     // output C (flat)
    };

    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, 2, 2, tensors, SplitModel::gemm());

    auto shimDmaAttr = module->getAttrOfType<DictionaryAttr>("tensor_0.shim_dma");
    if (shimDmaAttr) {
        std::cout << "PASS: tensor_0.shim_dma module attribute present" << std::endl;
        auto strides = shimDmaAttr.getAs<ArrayAttr>("strides");
        auto wraps = shimDmaAttr.getAs<ArrayAttr>("wraps");
        if (strides && wraps) {
            std::cout << "  strides=[";
            for (unsigned i = 0; i < strides.size(); ++i) {
                if (i > 0)
                    std::cout << ",";
                std::cout << cast<IntegerAttr>(strides[i]).getInt();
            }
            std::cout << "] wraps=[";
            for (unsigned i = 0; i < wraps.size(); ++i) {
                if (i > 0)
                    std::cout << ",";
                std::cout << cast<IntegerAttr>(wraps[i]).getInt();
            }
            std::cout << "]" << std::endl;
        }
    } else {
        std::cerr << "FAIL: tensor_0.shim_dma module attribute NOT found" << std::endl;
        pass = false;
    }

    // Run full pipeline
    std::string outputDir = setupWorklocalDir();
    if (!outputDir.empty()) {
        if (!TilingLinalgPipeline::runPipeline(ctx, module, outputDir,
                                               /*userKernelBody=*/"", /*userKernelFuncName=*/"",
                                               /*runtimeDebugLevel=*/-1, /*userRewrittenSource=*/"", tensors,
                                               /*maxPingPongBytes=*/4096, g_aieGen)) {
            llvm::errs() << "TilingLinalgPipeline::runPipeline failed!\n";
            pass = false;
        }
    }

    std::cout << "\n=== Dilated Conv2d Test " << (pass ? "PASSED" : "FAILED") << " ===" << std::endl;
}

void testPool() {
    std::cout << "\n=== Pool DmaTransform Test ===" << std::endl;
    std::cout << "Tests pool() factory (semantic alias for im2col).\n" << std::endl;

    // MaxPool: Input[8,8,1], PoolWindow[2,2], stride=2, pad=0
    // OW = (8 + 0 - 2) / 2 + 1 = 4
    // OH = 4
    int H = 8, W = 8, C = 1, KH = 2, KW = 2, S = 2, P = 0;
    int OH = (H + 2 * P - KH) / S + 1; // 4
    int OW = (W + 2 * P - KW) / S + 1; // 4
    int M_dim = OH * OW;               // 16
    int K_dim = KH * KW * C;           // 4

    std::cout << "Pool params: H=" << H << " W=" << W << " C=" << C << " KH=" << KH << " KW=" << KW << " S=" << S
              << " P=" << P << " OH=" << OH << " OW=" << OW << "\nGEMM dims: M=" << M_dim << " K=" << K_dim
              << std::endl;

    // pool() is alias for im2col(): dim0={1, KW*C}, dim1={W*C, KH}, dim2={S*C, OW}
    DmaAddressing pool;
    pool.dims = {{1, KW * C}, {W * C, KH}, {S * C, OW}};
    pool.iter_step = W * C * S;
    pool.iter_wrap = OH;

    std::cout << "DmaAddressing: dims=[";
    for (unsigned i = 0; i < pool.dims.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << "{" << pool.dims[i].first << "," << pool.dims[i].second << "}";
    }
    std::cout << "] iter_step=" << pool.iter_step << " iter_wrap=" << pool.iter_wrap << std::endl;

    // Verify expected values
    bool pass = true;
    if (pool.dims[0].first != 1 || pool.dims[0].second != 2) {
        std::cerr << "FAIL: dim0 expected {1,2} got {" << pool.dims[0].first << "," << pool.dims[0].second << "}"
                  << std::endl;
        pass = false;
    }
    if (pool.dims[1].first != 8 || pool.dims[1].second != 2) {
        std::cerr << "FAIL: dim1 expected {8,2} got {" << pool.dims[1].first << "," << pool.dims[1].second << "}"
                  << std::endl;
        pass = false;
    }
    if (pool.dims[2].first != 2 || pool.dims[2].second != 4) {
        std::cerr << "FAIL: dim2 expected {2,4} got {" << pool.dims[2].first << "," << pool.dims[2].second << "}"
                  << std::endl;
        pass = false;
    }
    if (pool.iter_step != 16) {
        std::cerr << "FAIL: iter_step expected 16 got " << pool.iter_step << std::endl;
        pass = false;
    }
    if (pool.iter_wrap != 4) {
        std::cerr << "FAIL: iter_wrap expected 4 got " << pool.iter_wrap << std::endl;
        pass = false;
    }

    // Build module
    std::vector<TensorParam> tensors = {
        {{M_dim, K_dim}, 8, true, pool}, // input A (pool window)
        {{K_dim, 1}, 8, true, {}},       // input B (flat, e.g. ones for avg pool)
        {{M_dim, 1}, 8, false, {}},      // output C (flat)
    };

    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, 2, 2, tensors, SplitModel::gemm());

    auto shimDmaAttr = module->getAttrOfType<DictionaryAttr>("tensor_0.shim_dma");
    if (shimDmaAttr) {
        std::cout << "PASS: tensor_0.shim_dma module attribute present" << std::endl;
    } else {
        std::cerr << "FAIL: tensor_0.shim_dma module attribute NOT found" << std::endl;
        pass = false;
    }

    std::cout << "\n=== Pool Test " << (pass ? "PASSED" : "FAILED") << " ===" << std::endl;
}

void testDepthwiseConv2d() {
    std::cout << "\n=== Depthwise Conv2d DmaTransform Test ===" << std::endl;
    std::cout << "Tests depthwise_im2col with G=C (each channel separate).\n" << std::endl;

    // Depthwise: Input[8,8,3], Filter[3,3] per channel, stride=1, pad=0, G=C=3
    // OW = (8 + 0 - 3) / 1 + 1 = 6
    // OH = 6
    // CPG = C/G = 3/3 = 1 (depthwise: each group has 1 channel)
    int H = 8, W = 8, C = 3, KH = 3, KW = 3, S = 1, P = 0, G = 3;
    int CPG = C / G;                   // 1
    int OH = (H + 2 * P - KH) / S + 1; // 6
    int OW = (W + 2 * P - KW) / S + 1; // 6
    int M_dim = OH * OW;               // 36
    int K_dim = KH * KW * CPG;         // 9 (per group)
    int N_dim = 1;

    std::cout << "Depthwise params: H=" << H << " W=" << W << " C=" << C << " KH=" << KH << " KW=" << KW << " S=" << S
              << " P=" << P << " G=" << G << " CPG=" << CPG << " OH=" << OH << " OW=" << OW
              << "\nPer-group GEMM: M=" << M_dim << " K=" << K_dim << " N=" << N_dim << std::endl;

    // depthwise_im2col: dim0={1, KW*CPG}, dim1={W*C, KH}, dim2={S*C, OW}
    DmaAddressing depthwise;
    depthwise.dims = {{1, KW * CPG}, {W * C, KH}, {S * C, OW}};
    depthwise.iter_step = W * C * S;
    depthwise.iter_wrap = OH;

    std::cout << "DmaAddressing: dims=[";
    for (unsigned i = 0; i < depthwise.dims.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << "{" << depthwise.dims[i].first << "," << depthwise.dims[i].second << "}";
    }
    std::cout << "] iter_step=" << depthwise.iter_step << " iter_wrap=" << depthwise.iter_wrap << std::endl;

    // Verify expected values for depthwise (G=C=3, CPG=1)
    bool pass = true;
    // dim0: {1, KW*CPG} = {1, 3*1} = {1, 3}
    if (depthwise.dims[0].first != 1 || depthwise.dims[0].second != 3) {
        std::cerr << "FAIL: dim0 expected {1,3} got {" << depthwise.dims[0].first << "," << depthwise.dims[0].second
                  << "}" << std::endl;
        pass = false;
    }
    // dim1: {W*C, KH} = {24, 3}
    if (depthwise.dims[1].first != 24 || depthwise.dims[1].second != 3) {
        std::cerr << "FAIL: dim1 expected {24,3} got {" << depthwise.dims[1].first << "," << depthwise.dims[1].second
                  << "}" << std::endl;
        pass = false;
    }
    // dim2: {S*C, OW} = {3, 6}
    if (depthwise.dims[2].first != 3 || depthwise.dims[2].second != 6) {
        std::cerr << "FAIL: dim2 expected {3,6} got {" << depthwise.dims[2].first << "," << depthwise.dims[2].second
                  << "}" << std::endl;
        pass = false;
    }
    // iter_step: W*C*S = 24
    if (depthwise.iter_step != 24) {
        std::cerr << "FAIL: iter_step expected 24 got " << depthwise.iter_step << std::endl;
        pass = false;
    }
    // iter_wrap: OH = 6
    if (depthwise.iter_wrap != 6) {
        std::cerr << "FAIL: iter_wrap expected 6 got " << depthwise.iter_wrap << std::endl;
        pass = false;
    }

    // Build module
    std::vector<TensorParam> tensors = {
        {{M_dim, K_dim}, 8, true, depthwise}, // input A (depthwise im2col)
        {{K_dim, N_dim}, 8, true, {}},        // input B (flat)
        {{M_dim, N_dim}, 8, false, {}},       // output C (flat)
    };

    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, 2, 2, tensors, SplitModel::gemm());

    auto shimDmaAttr = module->getAttrOfType<DictionaryAttr>("tensor_0.shim_dma");
    if (shimDmaAttr) {
        std::cout << "PASS: tensor_0.shim_dma module attribute present" << std::endl;
        auto strides = shimDmaAttr.getAs<ArrayAttr>("strides");
        auto wraps = shimDmaAttr.getAs<ArrayAttr>("wraps");
        if (strides && wraps) {
            std::cout << "  strides=[";
            for (unsigned i = 0; i < strides.size(); ++i) {
                if (i > 0)
                    std::cout << ",";
                std::cout << cast<IntegerAttr>(strides[i]).getInt();
            }
            std::cout << "] wraps=[";
            for (unsigned i = 0; i < wraps.size(); ++i) {
                if (i > 0)
                    std::cout << ",";
                std::cout << cast<IntegerAttr>(wraps[i]).getInt();
            }
            std::cout << "]" << std::endl;
        }
    } else {
        std::cerr << "FAIL: tensor_0.shim_dma module attribute NOT found" << std::endl;
        pass = false;
    }

    std::cout << "\n=== Depthwise Conv2d Test " << (pass ? "PASSED" : "FAILED") << " ===" << std::endl;
}

void testMultidimBd() {
    std::cout << "\n=== Multi-Dimensional BD Addressing Test ===" << std::endl;

    // First, run the normal pipeline to verify no regression
    std::cout << "\n--- Phase 1: Regression check (normal pipeline) ---" << std::endl;
    routingtodfschedule("");

    // Phase 2: Build a minimal dfschedule module with dim_strides/dim_wraps
    // and run DfscheduleToApiPass to check multidim call emission
    std::cout << "\n--- Phase 2: Multi-dim BD emission test ---" << std::endl;

    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);

    // Build a 2x2 mesh GEMM routing IR and run through shared stages
    std::vector<TensorParam> tensors = {
        {{16, 16}, 8, true},  // input A
        {{16, 16}, 8, true},  // input B
        {{16, 16}, 8, false}, // output C
    };
    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, 4, 4, tensors);

    RoutingTopology rtopology(g_aieGen);
    std::string irDir = setupIRDir("multidim");
    int stage = 0;

    dumpIRToFile(module, irDir, stage++, "initial");

    // Run shared stages: routing -> dmap -> dmaphop -> blueprint
    if (!runSinglePass(ctx, module, std::make_unique<RoutingUnrollingLowerPass>(), irDir, stage,
                       "RoutingUnrollingLowerPass"))
        return;
    if (!runSinglePass(ctx, module, std::make_unique<RoutingToDmapPass>(rtopology), irDir, stage, "RoutingToDmapPass"))
        return;
    if (!runSinglePass(ctx, module, std::make_unique<DmapToDmaphopPass>(rtopology), irDir, stage, "DmapToDmaphopPass"))
        return;
    if (!runSinglePass(ctx, module, std::make_unique<DmaphopTodfscheblueprintPass>(), irDir, stage,
                       "DmaphopTodfscheblueprintPass"))
        return;

    // Clone for host path
    auto hostModule = cast<ModuleOp>(module->clone());

    // Initialize ResourceMgr
    auto hwRes = makeResource(g_aieGen);
    ResourceMgr::init(std::move(hwRes));

    // Run BlueprintToSchedulePass (host path)
    if (!runSinglePass(ctx, hostModule, std::make_unique<mlir::BlueprintToSchedulePass>(0.5, 4096, g_aieGen), irDir,
                       stage, "BlueprintToSchedulePass"))
        return;
    if (!runSinglePass(ctx, hostModule, std::make_unique<mlir::ScheduleCanonicalizePass>(), irDir, stage,
                       "ScheduleCanonicalizePass"))
        return;

    // Now inject dim_strides/dim_wraps on the first Shim tile config.dma_bd
    // that corresponds to B-matrix input (data_id = 1 convention)
    std::cout << "Injecting dim_strides/dim_wraps on B-matrix BD..." << std::endl;
    bool injected = false;
    hostModule.walk([&](dfschedule::ConfigDmaBdOp bdOp) {
        // Inject on the first BD that doesn't already have multidim attrs
        // We target a shim-tile BD (data_id >= 0, no locks = shim tile pattern)
        if (!injected && !bdOp.getDimStrides()) {
            // B-matrix transpose for 16x16:
            // dim0: stride=16, wrap=16 (jump across rows for one column)
            // dim1: stride=1, wrap=8 (next column, 8 columns per tile)
            SmallVector<Attribute, 2> strides = {
                IntegerAttr::get(IntegerType::get(&ctx, 32), 16),
                IntegerAttr::get(IntegerType::get(&ctx, 32), 1),
            };
            SmallVector<Attribute, 2> wraps = {
                IntegerAttr::get(IntegerType::get(&ctx, 32), 16),
                IntegerAttr::get(IntegerType::get(&ctx, 32), 8),
            };
            bdOp.setDimStridesAttr(ArrayAttr::get(&ctx, strides));
            bdOp.setDimWrapsAttr(ArrayAttr::get(&ctx, wraps));
            injected = true;
            std::cout << "  Injected multidim attrs: dim_strides=[16,1], dim_wraps=[16,8]" << std::endl;
        }
    });

    if (!injected) {
        std::cerr << "WARNING: Could not find a BD op to inject multidim attrs" << std::endl;
    }

    dumpIRToFile(hostModule, irDir, stage++, "MultidimInjected");

    // Run DfscheduleToApiPass
    if (!runSinglePass(ctx, hostModule, std::make_unique<mlir::DfscheduleToApiPass>(true), irDir, stage,
                       "DfscheduleToApiPass"))
        return;
    if (!runSinglePass(ctx, hostModule, mlir::createCanonicalizerPass(), irDir, stage, "CanonicalizerPass"))
        return;
    if (!runSinglePass(ctx, hostModule, std::make_unique<RoutingConstantFoldPass>(g_aieGen), irDir, stage,
                       "RoutingConstantFoldPass"))
        return;

    // Emit host.cc and check for __Runtime_dma_bd_config_multidim
    const std::string worklocalDir = setupWorklocalDir();
    if (worklocalDir.empty())
        return;

    std::string hostPath = worklocalDir + "/host_multidim.cc";
    std::error_code hostEC;
    llvm::raw_fd_ostream hostStream(hostPath, hostEC, llvm::sys::fs::OF_None);
    if (hostEC) {
        llvm::errs() << "Failed to open " << hostPath << ": " << hostEC.message() << "\n";
        return;
    }
    mlir::LogicalResult result = mlir::emitc::translateToCpp(hostModule, hostStream);
    hostStream.close();
    if (failed(result)) {
        llvm::errs() << "Failed to translate host MLIR to C++.\n";
        return;
    }
    std::cout << "Host code (multidim) written to " << hostPath << std::endl;

    // Verify: read host_multidim.cc and check for __Runtime_dma_bd_config_multidim
    auto bufOrErr = llvm::MemoryBuffer::getFile(hostPath);
    if (bufOrErr) {
        auto content = (*bufOrErr)->getBuffer();
        if (content.contains("__Runtime_dma_bd_config_multidim")) {
            std::cout << "\n✓ PASS: host_multidim.cc contains __Runtime_dma_bd_config_multidim call" << std::endl;
        } else {
            std::cout << "\n✗ FAIL: host_multidim.cc does NOT contain __Runtime_dma_bd_config_multidim call"
                      << std::endl;
        }
        // Also verify regular config still present for non-multidim BDs
        if (content.contains("__Runtime_dma_bd_config(")) {
            std::cout << "✓ PASS: host_multidim.cc still contains regular __Runtime_dma_bd_config calls (no regression)"
                      << std::endl;
        }
    }

    std::cout << "\n=== Multi-Dimensional BD Test Complete ===" << std::endl;
}

// ---------------------------------------------------------------------------
// Partition test: build a 4x4 mesh on an 8x8 array with partition bounds
// and verify all generated tile coordinates fall within the partition.
// ---------------------------------------------------------------------------
void testPartition() {
    std::cout << "\n=== Partition Test ===" << std::endl;

    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);

    std::vector<TensorParam> tensors = {
        {{16, 16}, 8, true},  // input A
        {{16, 16}, 8, true},  // input B
        {{16, 16}, 8, false}, // output C
    };

    // Build IR: 4x4 mesh, partition confined to cols [2,7], rows [0,10]
    // Gen2 NoC shim columns: {2,3,6,7,...}. A 4x4 GEMM needs ~12 DataIOs
    // so the partition must include at least 4 NoC shim columns (2,3,6,7).
    PartitionDesc partition;
    partition.startCol = 2;
    partition.endCol = 7;
    partition.startRow = 0;
    partition.endRow = 10;

    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, 4, 4, tensors, SplitModel::gemm(), partition);

    // Verify createhwmesh has partition attributes in IR
    std::cout << "--- Verifying createhwmesh partition attrs in IR ---" << std::endl;
    bool foundPartition = false;
    module.walk([&](routing::createhwmesh meshOp) {
        auto sc = meshOp.getStartCol();
        auto ec = meshOp.getEndCol();
        auto sr = meshOp.getStartRow();
        auto er = meshOp.getEndRow();
        if (sc && ec && sr && er) {
            std::cout << "  createhwmesh: row=" << meshOp.getRow() << " col=" << meshOp.getCol() << " partition=["
                      << *sc << "," << *ec << "," << *sr << "," << *er << "]" << std::endl;
            foundPartition = (*sc == 2 && *ec == 7 && *sr == 0 && *er == 10);
        }
    });
    std::cout << "  Partition attrs in IR: " << (foundPartition ? "PASS" : "FAIL") << std::endl;

    // Run the pipeline
    std::string outputDir = setupWorklocalDir();
    if (outputDir.empty())
        return;

    std::cout << "--- Running pipeline with partition ---" << std::endl;
    if (!TilingLinalgPipeline::runPipeline(ctx, module, outputDir,
                                           /*userKernelBody=*/"", /*userKernelFuncName=*/"",
                                           /*runtimeDebugLevel=*/-1, /*userRewrittenSource=*/"", tensors)) {
        llvm::errs() << "Partition pipeline FAILED!\n";
        return;
    }
    std::cout << "--- Partition pipeline completed ---" << std::endl;
    std::cout << "=== Partition Test DONE ===" << std::endl;
}

// ---------------------------------------------------------------------------
// testReadGemmTilingScalars — verify readGemmTilingScalars reconstructs every
// GEMM tiling scalar (including the new mRounds/nRounds) from a #routing.tiling
// op on a routing.partitiontensor, with no reliance on the flat module attrs.
// ---------------------------------------------------------------------------
void testReadGemmTilingScalars() {
    std::cout << "\n=== readGemmTilingScalars Test ===" << std::endl;

    MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);
    mlir::OpBuilder b(&ctx);
    auto loc = b.getUnknownLoc();

    // fullconnect_auto=1 gate: reader is active only for the even-mesh GEMM path.
    auto module = mlir::ModuleOp::create(loc);
    module->setAttr("routing.fullconnect_auto", b.getI64IntegerAttr(1));
    b.setInsertionPointToStart(module.getBody());

    // Golden geometry mirrors simplematmul2.cc 256x256 on 4x4 mesh (from
    // ir/dfschedule/0_initial.mlir): tile_rows=tile_cols=64, tile_m=tile_n=16,
    // m_rounds=n_rounds=16, full_k=256, effective_k=64, k_rounds=4.
    //   split dim (M or N): outer slice=64 rounds=4 (mesh), nested slice=16
    //                       rounds=16 (on-core) → tileRows/Cols=64, tileM/N=16,
    //                       m/nRounds=16.
    //   K dim:              outer rounds=1 total=256, nested slice=64 rounds=4.
    // A helper to build a #routing.level.
    auto mkLevel = [&](int64_t base, int64_t total, int64_t slice, int64_t step, int64_t rounds,
                       routing::LevelAttr nested) {
        return routing::LevelAttr::get(&ctx, base, total, slice, step, rounds, nested);
    };
    // Mesh split dim (identical for row-M and col-N): outer 64x4, nested 16x16.
    auto meshCore = mkLevel(64, 256, 16, 16, 16, routing::LevelAttr{});
    auto meshOuter = mkLevel(256, 256, 64, 64, 4, meshCore);
    // K dim: outer rounds=1 total=256, nested slice=64 rounds=4.
    auto kCore = mkLevel(256, 256, 64, 64, 4, routing::LevelAttr{});
    auto kOuter = mkLevel(256, 256, 256, 256, 1, kCore);

    // A tensor (owner="col", splitdim=0): d0=mesh(N), d1=K.
    llvm::SmallVector<routing::DimAttr> aDims = {routing::DimAttr::get(&ctx, meshOuter, /*axis=*/0),
                                                 routing::DimAttr::get(&ctx, kOuter, /*axis=*/0)};
    auto aTiling = routing::TilingAttr::get(&ctx, aDims);
    // B/input tensor (owner="row", splitdim=0): d0=mesh(M), d1=K.
    llvm::SmallVector<routing::DimAttr> bDims = {routing::DimAttr::get(&ctx, meshOuter, /*axis=*/0),
                                                 routing::DimAttr::get(&ctx, kOuter, /*axis=*/0)};
    auto bTiling = routing::TilingAttr::get(&ctx, bDims);

    auto tensorTy = mlir::RankedTensorType::get({256, 256}, b.getI8Type());
    auto src = b.create<mlir::tensor::EmptyOp>(loc, mlir::ArrayRef<int64_t>{256, 256}, b.getI8Type());
    // Col-owner partition (N): split dim 0.
    b.create<routing::partitiontensor>(loc, tensorTy, src.getResult(),
                                       routing::PartitionAttr::get(&ctx, /*splitnum=*/4, /*splitdim=*/0,
                                                                   /*hwAxisOwner=*/"col",
                                                                   /*replicateOn=*/"row", /*singleTileOwner=*/""),
                                       aTiling);
    // Row-owner partition (M): split dim 0.
    b.create<routing::partitiontensor>(loc, tensorTy, src.getResult(),
                                       routing::PartitionAttr::get(&ctx, /*splitnum=*/4, /*splitdim=*/0,
                                                                   /*hwAxisOwner=*/"row",
                                                                   /*replicateOn=*/"col", /*singleTileOwner=*/""),
                                       bTiling);

    routing::GemmTilingScalars t = routing::readGemmTilingScalars(module);

    struct Check {
        const char *name;
        int64_t got;
        int64_t want;
    };
    Check checks[] = {
        {"found", t.found ? 1 : 0, 1}, {"tileM", t.tileM, 16},           {"tileRows", t.tileRows, 64},
        {"mRounds", t.mRounds, 16},    {"tileN", t.tileN, 16},           {"tileCols", t.tileCols, 64},
        {"nRounds", t.nRounds, 16},    {"effectiveK", t.effectiveK, 64}, {"fullK", t.fullK, 256},
        {"kRounds", t.kRounds, 4},
    };
    bool allPass = true;
    for (const auto &c : checks) {
        bool ok = (c.got == c.want);
        allPass &= ok;
        std::cout << "  " << c.name << " = " << c.got << " (want " << c.want << "): " << (ok ? "PASS" : "FAIL")
                  << std::endl;
    }
    std::cout << "=== readGemmTilingScalars Test " << (allPass ? "PASS" : "FAIL") << " ===" << std::endl;
    module->erase();
}

// ---------------------------------------------------------------------------
// Control-packet CTRL-sink lowering test
//
// Builds a minimal routinghw module with a single ConnectStreamPktSwitchPort
// whose localsinkport = "CTRL", runs RoutingHWLowerPass, and verifies the
// emitted C++ terminates the packet at the tile's CTRL master stream-switch
// port with the control header preserved (XAIE_SS_PKT_DONOT_DROP_HEADER).
// A companion DMA-sink case confirms the default path is unchanged.
// ---------------------------------------------------------------------------
static bool lowerCtrlSinkAndEmit(const std::string &localsinkport, std::string &emitted) {
    MLIRContext ctx;
    routinghwmanager mtesthw;
    mtesthw.loaddialect(&ctx);
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    ctx.getOrLoadDialect<mlir::emitc::EmitCDialect>();
    ctx.getOrLoadDialect<arith::ArithDialect>();

    // A single-tile packet route into the local sink port. The func's first
    // argument stands in for XAie_DevInst* dev (RoutingHWLowerPass reads it as
    // the device instance).
    std::string ir = R"MLIR(
    func.func @route(%dev: !emitc.ptr<!emitc.opaque<"XAie_DevInst">>) {
      %t = "routinghw.tilecreate"() {row = 2 : i32, col = 0 : i32, comments = "ctrltest"} : () -> i32
      %o = "routinghw.connectpktstreamswitchport"(%t) {
        receiveslavedirection = "SOUTH", receiveslaveportidx = 0 : i32,
        receiveslavepktid = 3 : i32, receiveslavepkttype = 0 : i32,
        localdmadirection = "NONE", localdmaportidx = 0 : i32,
        localdmapktid = 0 : i32, localdmapkttype = 0 : i32,
        forwardmasterdirection = "NONE", forwardmasterportidx = 0 : i32,
        localsinkport = ")MLIR" +
                     localsinkport + R"MLIR("
      } : (i32) -> i32
      return
    }
    )MLIR";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(ir, &ctx);
    if (!module) {
        llvm::errs() << "[ctrlwrite] failed to parse test IR\n";
        return false;
    }

    RoutingTopology rtopology(g_aieGen);
    mlir::PassManager pm(&ctx);
    pm.addPass(std::make_unique<RoutingHWLowerPass>(rtopology));
    if (failed(pm.run(*module))) {
        llvm::errs() << "[ctrlwrite] RoutingHWLowerPass failed\n";
        return false;
    }

    llvm::raw_string_ostream os(emitted);
    if (failed(mlir::emitc::translateToCpp(*module, os))) {
        llvm::errs() << "[ctrlwrite] translateToCpp failed\n";
        return false;
    }
    os.flush();
    return true;
}

static void testControlPacketCtrlSink() {
    std::cout << "=== Control-packet CTRL-sink lowering test ===" << std::endl;
    bool allPass = true;

    // CTRL sink: must emit a CTRL master enable with the header preserved.
    std::string ctrlOut;
    if (!lowerCtrlSinkAndEmit("CTRL", ctrlOut)) {
        std::cout << "  CTRL lowering: FAIL (pipeline error)" << std::endl;
        allPass = false;
    } else {
        bool hasMstr = ctrlOut.find("XAie_StrmPktSwMstrPortEnable") != std::string::npos;
        bool hasCtrl = ctrlOut.find("CTRL") != std::string::npos;
        bool preserves = ctrlOut.find("XAIE_SS_PKT_DONOT_DROP_HEADER") != std::string::npos;
        bool noDrop = ctrlOut.find("XAIE_SS_PKT_DROP_HEADER") == std::string::npos;
        std::cout << "  CTRL master enable emitted: " << (hasMstr ? "PASS" : "FAIL") << std::endl;
        std::cout << "  CTRL port targeted:         " << (hasCtrl ? "PASS" : "FAIL") << std::endl;
        std::cout << "  header preserved:           " << (preserves ? "PASS" : "FAIL") << std::endl;
        std::cout << "  header not dropped:         " << (noDrop ? "PASS" : "FAIL") << std::endl;
        allPass &= (hasMstr && hasCtrl && preserves && noDrop);
    }

    // DMA sink (default): must NOT target CTRL; backward-compatibility check.
    std::string dmaOut;
    if (!lowerCtrlSinkAndEmit("DMA", dmaOut)) {
        std::cout << "  DMA lowering: FAIL (pipeline error)" << std::endl;
        allPass = false;
    } else {
        bool noCtrl = dmaOut.find("CTRL") == std::string::npos;
        std::cout << "  DMA sink does not target CTRL: " << (noCtrl ? "PASS" : "FAIL") << std::endl;
        allPass &= noCtrl;
    }

    std::cout << "=== Control-packet CTRL-sink Test " << (allPass ? "PASS" : "FAIL") << " ===" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse --gen and --output-pp-depth arguments from anywhere in argv
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gen" && i + 1 < argc) {
            g_aieGen = argv[i + 1];
            // Shift remaining args to remove --gen <value>
            for (int j = i; j + 2 < argc; ++j)
                argv[j] = argv[j + 2];
            argc -= 2;
            --i; // re-check this index
        } else if (a == "--output-pp-depth" && i + 1 < argc) {
            g_outputPpDepth = std::atoi(argv[i + 1]);
            for (int j = i; j + 2 < argc; ++j)
                argv[j] = argv[j + 2];
            argc -= 2;
            --i;
        }
    }
    std::cout << "AIE generation: " << g_aieGen << std::endl;
    if (g_outputPpDepth != 2)
        std::cout << "Output pp_depth: " << g_outputPpDepth << std::endl;

    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--parse") {
            if (argc < 4) {
                std::cerr << "Usage: ./test --parse <stage> <mlir_file>\n"
                          << "Supported stages:\n"
                          << "  routing        - stage 0 (initial routing IR)\n"
                          << "  dmap           - stage 2 (post-RoutingToDmap)\n"
                          << "  dmaphop        - stage 3 (post-DmapToDmaphop)\n"
                          << "  dfscheblueprint - stage 4 (post-DmaphopTodfscheblueprint, branch point)\n"
                          << "  dfschedule     - stage 5/6/10 (post-branch, auto-detects host/kernel)\n"
                          << "  emitc          - stage 7/8/9/11 (pure EmitC, auto-detects host/kernel)\n"
                          << std::endl;
                return 1;
            }
            std::string mode = argv[2];
            std::string filepath = argv[3];

            std::cout << "Executing --parse " << mode << " with IR from " << filepath << std::endl;

            // Validate IR dialect and auto-detect actual stage
            std::string detectedMode;
            {
                MLIRContext valCtx;
                valCtx.allowUnregisteredDialects();
                routingmanager valRouting;
                routinghwmanager valRoutinghw;
                dmapmanager valDmap;
                dmaphopmanager valDmaphop;
                dfscheblueprintmanager valBlueprint;
                dfschedulemanager valSchedule;
                valRouting.loaddialect(&valCtx);
                valRoutinghw.loaddialect(&valCtx);
                valDmap.loaddialect(&valCtx);
                valDmaphop.loaddialect(&valCtx);
                valBlueprint.loaddialect(&valCtx);
                valSchedule.loaddialect(&valCtx);
                valCtx.getOrLoadDialect<arith::ArithDialect>();
                valCtx.getOrLoadDialect<mlir::func::FuncDialect>();
                valCtx.getOrLoadDialect<mlir::scf::SCFDialect>();
                valCtx.getOrLoadDialect<mlir::tensor::TensorDialect>();
                valCtx.getOrLoadDialect<mlir::emitc::EmitCDialect>();
                valCtx.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
                valCtx.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
                valCtx.getOrLoadDialect<mlir::memref::MemRefDialect>();

                auto valModule = loadModuleFromFile(valCtx, filepath);
                if (!valModule) return 1;
                if (!validateIRDialect(*valModule, mode, filepath, detectedMode))
                    return 1;

                // Use the detected mode (may differ from user-specified mode)
                mode = detectedMode;

                // For post-branch stages, auto-detect host vs kernel
                if (mode == "dfschedule" || mode == "emitc") {
                    std::string kind = detectHostOrKernel(*valModule);
                    if (kind == "unknown") {
                        std::cerr << "ERROR: Could not auto-detect host vs kernel for "
                                  << mode << " IR.\n"
                                  << "  File: " << filepath << "\n"
                                  << "  Hint: ensure the IR contains characteristic ops "
                                     "(dfschedule.declaretile for host, "
                                     "dfschedule.kernel_config_def for kernel).\n";
                        return 1;
                    }
                    // Store kind for startStage computation below
                    if (mode == "dfschedule")
                        mode = (kind == "host") ? "dfschedule_host" : "dfschedule_kernel";
                    else
                        mode = (kind == "host") ? "emitc_host" : "emitc_kernel";
                    std::cout << "Auto-detected " << kind << " path." << std::endl;
                }
            }

            // Map detected mode to startStage
            int startStage = -1;
            bool canProduceRouting = false;

            if (mode == "routing")                startStage = 0, canProduceRouting = true;
            else if (mode == "routing_unrolled")  startStage = 1, canProduceRouting = true;
            else if (mode == "dmap")              startStage = 2, canProduceRouting = true;
            else if (mode == "dmaphop")           startStage = 3, canProduceRouting = true;
            else if (mode == "dfscheblueprint")   startStage = 4;
            else if (mode == "dfschedule_host")   startStage = 5;
            else if (mode == "dfschedule_kernel") startStage = 10;
            else if (mode == "emitc_host")        startStage = 7;
            else if (mode == "emitc_kernel")      startStage = 11;
            else {
                std::cerr << "ERROR: Unrecognized detected mode: " << mode << "\n";
                return 1;
            }

            // Run dfschedule pipeline from the detected start stage
            routingtodfschedule(filepath, startStage);

            // Run routing.cc pipeline only if the entry stage is early enough
            if (canProduceRouting) {
                routingtodmap(filepath, startStage);
            }
        } else if (arg == "routing") {
            if (argc < 3) {
                std::cerr << "Usage: ./test routing <mlir_file>" << std::endl;
                return 1;
            }
            std::string filepath = argv[2];
            std::cout << "Executing routingtodfschedule with IR from " << filepath << std::endl;
            routingtodfschedule(filepath);
        } else if (arg == "hw") {
            std::cout << "Executing routingtoroutinghw..." << std::endl;
            routingtoroutinghw();
        } else if (arg == "test") {
            std::cout << "Executing unit test for RoutingLowerPass..." << std::endl;
            testRoutingLowerPassPathContiguity();
        } else if (arg == "dfschedule") {
            std::cout << "Executing routingtodfschedule..." << std::endl;
            routingtodfschedule();
        } else if (arg == "dmaphw") {
            std::cout << "Executing routingtodmap..." << std::endl;
            routingtodmap();
        } else if (arg == "conv2d") {
            std::cout << "Executing conv2d DmaTransform integration test..." << std::endl;
            testConv2d();
        } else if (arg == "conv2d_spatial") {
            std::cout << "Executing conv2d spatial-halo integration test..." << std::endl;
            testConv2dSpatial();
        } else if (arg == "dilated") {
            std::cout << "Executing dilated conv2d DmaTransform test..." << std::endl;
            testDilatedConv2d();
        } else if (arg == "pool") {
            std::cout << "Executing pool DmaTransform test..." << std::endl;
            testPool();
        } else if (arg == "depthwise") {
            std::cout << "Executing depthwise conv2d DmaTransform test..." << std::endl;
            testDepthwiseConv2d();
        } else if (arg == "multidim") {
            std::cout << "Executing multi-dimensional BD addressing test..." << std::endl;
            testMultidimBd();
        } else if (arg == "partition") {
            std::cout << "Executing partition test..." << std::endl;
            testPartition();
        } else if (arg == "readtiling") {
            std::cout << "Executing readGemmTilingScalars test..." << std::endl;
            testReadGemmTilingScalars();
        } else if (arg == "ctrlwrite") {
            std::cout << "Executing control-packet CTRL-sink test..." << std::endl;
            testControlPacketCtrlSink();
        } else {
            std::cout << "Invalid argument. Please use hw, test, dfschedule, dmaphw, conv2d, conv2d_spatial, "
                         "dilated, pool, "
                         "depthwise, multidim, partition, routing, --parse "
                         "<stage> <file>\n"
                      << "Stages: routing, dmap, dmaphop, dfscheblueprint, dfschedule, emitc\n"
                      << "Options: --gen <Gen1|Gen2|Gen5> (default: Gen2)" << std::endl;
        }
    } else {
        // Default behavior: routingtodfschedule generates host.cc, kernel.cc,
        // AND routing.cc (via the dmaphop->routinghw path in TilingLinalgPipeline).
        // Do NOT run routingtodmap() here — it would overwrite routing.cc with
        // independently-allocated packet IDs that don't match host.cc.
        std::cout << "Executing routingtodfschedule..." << std::endl;
        routingtodfschedule();
    }
    return 0;

}