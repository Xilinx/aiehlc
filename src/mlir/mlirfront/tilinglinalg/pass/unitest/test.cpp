/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#include <iostream>
#include "routinghwmanager.h"
#include "routinghwlower.h"
#include "routingmanager.h"
#include "routinglower.h"
#include "../passroutingtodmap/routingtodmap.h"
#include "../passdmaptodmaphop/dmaptodmaphop.h"
#include "../passdmaphoptoroutinghw/passdmaphoptoroutinghw.h"
#include "../passblueprinttoschedule/passblueprinttoschedule.h"
#include "../passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.h"
#include "../passschedulecanonicalize/passschedulecanonicalize.h"
#include "../passdfscheduletoapi/passdfscheduletoapi.h"
#include "dmapmanager.h"
#include "dmaphopmanager.h"
#include "dfschedulemanager.h"
#include "dfscheblueprintmanager.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "routingunrolling.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
//#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"


#include "routingdeadargclean.h"

#include "routingconstantfold.h"

// Unit test function to verify path contiguity for RoutingLowerPass
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
    RoutingTopology rtopology("Gen2");
    
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

void routingtoroutinghw() {
     MLIRContext ctx;
    
    routingmanager mtest;
    routinghwmanager mtesthw;
    mtesthw.loaddialect(&ctx);
    mtest.loaddialect(&ctx);

    ctx.getOrLoadDialect<arith::ArithDialect>();
    
    //auto module1 = mtest.createroutingfunc(&ctx,1);
    auto module1 = mtest.ops_testNew(&ctx,1);
    module1.dump();
    //auto module2 = mtesthw.ops_test(&ctx);
    std::cout << "main" <<std::endl;
    
    mlir::PrintIRPassOptions options;

    mlir::PassManager pm(&ctx);;
    RoutingTopology rtopology("Gen2");
    
    options.label = "Before RoutingUnrollingLowerPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    pm.addPass(std::make_unique<RoutingUnrollingLowerPass>());
    options.label = "After RoutingUnrollingLowerPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    pm.addPass(std::make_unique<RoutingLowerPass>(rtopology));
    options.label = "After RoutingLowerPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    pm.addPass(std::make_unique<RoutingHWLowerPass>(rtopology));
    options.label = "After RoutingHWLowerPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    //remove dead arg
    pm.addPass(std::make_unique<RoutingDeadArgPass>());
    options.label = "After RoutingDeadArgPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    //The constanfold change emitc.call into emic.call_opaque to convert 
    /*
    XAie_LocType v251 = XAie_TileLoc(v1, v10);
    XAie_DevInst* v252 = getOrCreateDeviceInstance();
    int32_t v253 = XAie_StrmConnCctEnable(v252, v251, v6, v13, v5, v12);
    */
    //into
    /*
    int32_t v80 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,3), WEST, 0, EAST, 0); 
    */
    pm.addPass(std::make_unique<RoutingConstantFoldPass>());

    options.label = "After RoutingConstantFoldPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    //remove the dead code
    pm.addPass(mlir::createCanonicalizerPass());

    options.label = "After createCanonicalizerPasse:";
    pm.addPass(mlir::createPrintIRPass(options));

    //remove dead arg
    //pm.addPass(mlir::createConvertSCFToEmitCPass());
    (void)pm.run(module1);

    llvm::outs() << "----------module1.dump---------\n";
    module1.dump();
/*
    mlir::PassManager pm2(&ctx);;
    pm2.addPass(std::make_unique<RoutingHWLowerPass>(rtopology));
    (void)pm2.run(module1);
    module1.dump();
  */
//conver emitc into c code  
    mlir::LogicalResult result = mlir::emitc::translateToCpp(module1, llvm::outs());
    return;
}
void routingtodmap() {
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
    
    //auto module1 = mtest.createroutingfunc(&ctx,1);
    auto module1 = mtest.ops_testNew(&ctx,1);
    module1.dump();
    //auto module2 = mtesthw.ops_test(&ctx);
    std::cout << "main" <<std::endl;
    
    mlir::PrintIRPassOptions options;

    mlir::PassManager pm(&ctx);;
    RoutingTopology rtopology("Gen2");
    
    options.label = "Before RoutingUnrollingLowerPass:";
    //pm.addPass(mlir::createPrintIRPass(options));
    pm.addPass(std::make_unique<RoutingUnrollingLowerPass>());
    options.label = "After RoutingToDmapPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    pm.addPass(std::make_unique<RoutingToDmapPass>(rtopology));
    options.label = "After DmapToDmaphopPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    pm.addPass(std::make_unique<DmapToDmaphopPass>(rtopology));
    options.label = "After DmaphopToRoutinghwPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    pm.addPass(std::make_unique<DmaphopToRoutinghwPass>(rtopology));

    pm.addPass(mlir::createPrintIRPass(options));
    pm.addPass(std::make_unique<RoutingHWLowerPass>(rtopology));
    options.label = "After RoutingHWLowerPass:";
    //pm.addPass(mlir::createPrintIRPass(options));
    //remove dead arg
    pm.addPass(std::make_unique<RoutingDeadArgPass>());
    
    //into
    /*
    int32_t v80 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,3), WEST, 0, EAST, 0); 
    */
    pm.addPass(std::make_unique<RoutingConstantFoldPass>());

    options.label = "After RoutingConstantFoldPass:";
    //pm.addPass(mlir::createPrintIRPass(options));
    //remove the dead code
    pm.addPass(mlir::createCanonicalizerPass());

    options.label = "After createCanonicalizerPasse:";
    //pm.addPass(mlir::createPrintIRPass(options));

    //remove dead arg
    //pm.addPass(mlir::createConvertSCFToEmitCPass());
    (void)pm.run(module1);

    llvm::outs() << "----------module1.dump---------\n";
    //module1.dump();
/*
    mlir::PassManager pm2(&ctx);;
    pm2.addPass(std::make_unique<RoutingHWLowerPass>(rtopology));
    (void)pm2.run(module1);
    module1.dump();
  */
//conver emitc into c code  
    mlir::LogicalResult result = mlir::emitc::translateToCpp(module1, llvm::outs());
    if (failed(result)) {
        llvm::errs() << "Failed to translate MLIR to C++.\n";
        return;
    }
    return;
}

void routingtodfschedule() {
    MLIRContext ctx;
    
    RoutingTopology rtopology("Gen2");
    
    routingmanager mtest;
    dfschedulemanager dfscheduletest;
    dfscheblueprintmanager dfscheblueprinttest;
    
    mtest.loaddialect(&ctx);
    dfscheduletest.loaddialect(&ctx);
    dfscheblueprinttest.loaddialect(&ctx);
    ctx.getOrLoadDialect<arith::ArithDialect>();
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();
    ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx.getOrLoadDialect<mlir::tensor::TensorDialect>();
    ctx.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    ctx.getOrLoadDialect<mlir::emitc::EmitCDialect>();
    
    // Create test routing module
    auto module1 = mtest.ops_testNew(&ctx, 1);
    
    std::cout << "=== Initial Routing Module ===" << std::endl;
    module1.dump();
    
    // Create pass manager
    mlir::PassManager pm(&ctx);
    mlir::PrintIRPassOptions options;
    
    // Stage 1: Unroll routing operations
    options.label = "After RoutingUnrollingLowerPass:";
    pm.addPass(std::make_unique<RoutingUnrollingLowerPass>());
    pm.addPass(mlir::createPrintIRPass(options));

    // Stage 2: Convert routing to dmap
    pm.addPass(std::make_unique<RoutingToDmapPass>(rtopology));
    options.label = "After RoutingToDmapPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    
    // Stage 3: Convert dmap to dmaphop
    pm.addPass(std::make_unique<DmapToDmaphopPass>(rtopology));
    options.label = "After DmapToDmaphopPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    
    // Stage 4: Convert dmaphop to dfscheblueprint
    pm.addPass(std::make_unique<DmaphopTodfscheblueprintPass>());
    options.label = "After DmaphopTodfscheblueprintPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    
    // Stage 5: Convert dfscheblueprint to dfschedule (final schedule IR)
    pm.addPass(std::make_unique<mlir::BlueprintToSchedulePass>());
    options.label = "After BlueprintToSchedulePass:";
    pm.addPass(mlir::createPrintIRPass(options));
    
    // Stage 6: Canonicalize schedule - merge kernel loads, deduplicate tiles, consolidate IOs
    pm.addPass(std::make_unique<mlir::ScheduleCanonicalizePass>());
    options.label = "After ScheduleCanonicalizePass:";
    pm.addPass(mlir::createPrintIRPass(options));
    
    // Stage 7: Convert dfschedule to API calls and EmitC
    pm.addPass(std::make_unique<mlir::DfscheduleToApiPass>());
    options.label = "After DfscheduleToApiPass:";
    pm.addPass(mlir::createPrintIRPass(options));
    
    // Run the pass pipeline
    if (failed(pm.run(module1))) {
        llvm::errs() << "ERROR: Pass pipeline failed!\n";
        return;
    }
    
    std::cout << "\n=== Final Module with API calls ===" << std::endl;
    module1.dump();
    
    // Convert to C++ code
    std::cout << "\n=== Generated C++ Code ===" << std::endl;
    mlir::LogicalResult result = mlir::emitc::translateToCpp(module1, llvm::outs());
    if (failed(result)) {
        llvm::errs() << "Failed to translate MLIR to C++.\n";
    }
    
    return;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "hw") {
            std::cout << "Executing routingtoroutinghw..." << std::endl;
            routingtoroutinghw();
        } else if (arg == "test") {
            std::cout << "Executing unit test for RoutingLowerPass..." << std::endl;
            testRoutingLowerPassPathContiguity();
        } else if (arg == "dfschedule") {
            std::cout << "Executing routingtodfschedule..." << std::endl;
            routingtodfschedule();
        } else {
            std::cout << "Executing routingtodmap..." << std::endl;
            routingtodmap();
        }
    } else {
        // Default behavior when no argument is provided
        std::cout << "No argument provided. Executing routingtodmap by default..." << std::endl;
        routingtodmap();
    }
    return 0;

}