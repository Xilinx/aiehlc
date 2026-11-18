/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "routinghwverify.h"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

struct Connection {
    std::string slaveDir;
    int slaveIdx;
    std::string masterDir;
    int masterIdx;
};

using TileKey = std::pair<int, int>; // (col, row)

static std::string oppositeDir(const std::string &dir) {
    if (dir == "NORTH")
        return "SOUTH";
    if (dir == "SOUTH")
        return "NORTH";
    if (dir == "EAST")
        return "WEST";
    if (dir == "WEST")
        return "EAST";
    return "";
}

static TileKey neighborTile(int col, int row, const std::string &dir) {
    if (dir == "NORTH")
        return {col, row + 1};
    if (dir == "SOUTH")
        return {col, row - 1};
    if (dir == "EAST")
        return {col + 1, row};
    if (dir == "WEST")
        return {col - 1, row};
    return {-1, -1};
}

static bool isDirectional(const std::string &dir) {
    return dir == "NORTH" || dir == "SOUTH" || dir == "EAST" || dir == "WEST";
}

// Extract (col, row) from a tile-creating op (TileCreate or IOShimTileCreate)
static bool getTileLocation(Operation *op, int &col, int &row) {
    if (auto colAttr = op->getAttrOfType<IntegerAttr>("col"))
        col = colAttr.getInt();
    else
        return false;
    if (auto rowAttr = op->getAttrOfType<IntegerAttr>("row"))
        row = rowAttr.getInt();
    else
        return false;
    return true;
}

} // anonymous namespace

void RoutingHWVerifyPass::runOnOperation() {
    auto module = getOperation();
    bool hasError = false;
    int roundIdx = 0;

    module->walk([&](routing::RoutingCreate routingCreateOp) {
        // Collect tile set and connections within this RoutingCreate region
        std::set<TileKey> tileSet;
        std::set<TileKey> shimTileSet; // IOShimTileCreate tiles (row 0, SOUTH = external NoC)
        std::map<Operation *, TileKey> opToTile;
        std::map<TileKey, std::vector<Connection>> tileConnections;

        std::string memo = routingCreateOp.getMemo().str();

        // Pass 1: collect tiles
        routingCreateOp->walk([&](Operation *innerOp) {
            if (isa<routinghw::TileCreate>(innerOp) || isa<routinghw::IOShimTileCreate>(innerOp)) {
                int col = -1, row = -1;
                if (getTileLocation(innerOp, col, row)) {
                    TileKey key = {col, row};
                    tileSet.insert(key);
                    opToTile[innerOp] = key;
                    if (isa<routinghw::IOShimTileCreate>(innerOp))
                        shimTileSet.insert(key);
                }
            }
        });

        // Pass 2: collect connections from connectsinglestreamswitchport
        routingCreateOp->walk([&](routinghw::ConnectStreamSingleSwitchPort connectOp) {
            if (connectOp->getNumOperands() == 0)
                return;
            Operation *tileOp = connectOp->getOperand(0).getDefiningOp();
            auto it = opToTile.find(tileOp);
            if (it == opToTile.end())
                return;

            Connection conn;
            conn.slaveDir = connectOp.getSlaveportdirection().str();
            conn.slaveIdx = connectOp.getSlaveportidx();
            conn.masterDir = connectOp.getMasterportdirection().str();
            conn.masterIdx = connectOp.getMasterportidx();
            tileConnections[it->second].push_back(conn);
        });

        // Pass 3: verify neighbor port consistency
        int checked = 0, errors = 0;

        // Packet switch: no tile may forward packets out of the side they come in on.
        // A tile's slave and master can sit on separate ops (the chain's last tile
        // gets its circuit-facing master on its own op), so collect per tile. A
        // same-side U-turn (slave WEST -> master WEST) loses those packets on the
        // board while the DMA slave's own packets get through (simpleeltwise_add on
        // VEK280, skill kernelblockgather).
        std::map<TileKey, std::set<std::string>> pktSlaveDirs, pktMasterDirs;
        routingCreateOp->walk([&](routinghw::ConnectStreamPktSwitchPort pktOp) {
            Operation *tileOp = pktOp->getOperand(0).getDefiningOp();
            auto it = opToTile.find(tileOp);
            if (it == opToTile.end())
                return;
            std::string slaveDir = pktOp.getReceiveslavedirection().str();
            std::string masterDir = pktOp.getForwardmasterdirection().str();
            if (isDirectional(slaveDir))
                pktSlaveDirs[it->second].insert(slaveDir);
            if (isDirectional(masterDir))
                pktMasterDirs[it->second].insert(masterDir);
        });
        for (auto &[tile, masters] : pktMasterDirs) {
            for (const std::string &dir : masters) {
                if (!pktSlaveDirs[tile].count(dir))
                    continue;
                std::cerr << "  ERROR: (" << tile.first << "," << tile.second << ") packet master " << dir
                          << " forwards packets that arrive on its " << dir
                          << " slave: a same-side U-turn, those packets are lost" << std::endl;
                errors++;
            }
        }
        std::cout << "\n=== RoutingHW Neighbor Port Verify (Round " << roundIdx << ", " << memo << ") ===" << std::endl;

        for (auto &[tile, conns] : tileConnections) {
            for (auto &conn : conns) {
                if (!isDirectional(conn.masterDir))
                    continue;

                // Skip SOUTH master on shim tiles (row 0) - SOUTH connects to external NoC
                if (conn.masterDir == "SOUTH" && shimTileSet.count(tile)) {
                    std::cout << "  SKIP: (" << tile.first << "," << tile.second << ") master SOUTH[" << conn.masterIdx
                              << "] (shim tile -> external NoC)" << std::endl;
                    continue;
                }

                checked++;
                auto [ncol, nrow] = neighborTile(tile.first, tile.second, conn.masterDir);
                TileKey neighborKey = {ncol, nrow};
                std::string oppD = oppositeDir(conn.masterDir);

                // Check 1: neighbor tile must exist
                if (tileSet.find(neighborKey) == tileSet.end()) {
                    std::cerr << "  ERROR: (" << tile.first << "," << tile.second << ") master " << conn.masterDir
                              << "[" << conn.masterIdx << "] -> expects (" << ncol << "," << nrow
                              << ") but tile NOT FOUND in region" << std::endl;
                    errors++;
                    continue;
                }

                // Check 2: neighbor must have slave with dir=oppD, idx=masterIdx
                auto neighborIt = tileConnections.find(neighborKey);
                bool foundSlave = false;
                if (neighborIt != tileConnections.end()) {
                    for (auto &nconn : neighborIt->second) {
                        if (nconn.slaveDir == oppD && nconn.slaveIdx == conn.masterIdx) {
                            foundSlave = true;
                            break;
                        }
                    }
                }

                if (!foundSlave) {
                    std::cerr << "  ERROR: (" << tile.first << "," << tile.second << ") master " << conn.masterDir
                              << "[" << conn.masterIdx << "] -> expects (" << ncol << "," << nrow << ") slave " << oppD
                              << "[" << conn.masterIdx << "] NOT FOUND" << std::endl;
                    errors++;
                } else {
                    std::cout << "  OK: (" << tile.first << "," << tile.second << ") master " << conn.masterDir << "["
                              << conn.masterIdx << "] -> (" << ncol << "," << nrow << ") slave " << oppD << "["
                              << conn.masterIdx << "]" << std::endl;
                }
            }
        }

        std::cout << "Summary: " << checked << " checked, " << errors << " error(s)" << std::endl;

        if (errors > 0)
            hasError = true;

        roundIdx++;
    });

    if (hasError) {
        llvm::errs() << "RoutingHWVerifyPass: neighbor port consistency errors detected, halting pipeline.\n";
        signalPassFailure();
    }
}
