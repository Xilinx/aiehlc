/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

//=========================================================
// routing_path.h — Incremental tree router (no IHwResource)
//=========================================================
#ifndef ROUTING_PATH_H
#define ROUTING_PATH_H

#include <vector>
#include <array>
#include <cstdint>
#include "hw/ResourceManager.h"
// 64-bit hash key (row in high 32 bits, col in low 32 bits)
static inline long long key(const Point& p) {
    return (static_cast<long long>(p.r) << 32) | static_cast<uint32_t>(p.c);
}

// Four-direction unit vectors: up, down, left, right
extern const std::array<Point, 4> kDirs;

// ──────────────────────────────────────────────────────────────
// MultiPath — result of one-to-many routing
// ──────────────────────────────────────────────────────────────
struct MultiPath {
    Point                            src;       // common source
    std::vector<Point>               dsts;      // sinks sorted by col→row
    std::vector<std::vector<Point>>  branches;  // each branch
    std::shared_ptr<DataIO> dataio;
};

// ──────────────────────────────────────────────────────────────
// RoutingPath — 2-D mesh incremental tree router
//   • row-0 forbid left right movement
//   • support addObstacle(s)  
//   • addTree(): sinks  
// ──────────────────────────────────────────────────────────────
class RoutingPath {
public:
    RoutingPath(std::shared_ptr<ResourceMgr> resmgr,
                std::shared_ptr<DataIO> dio = nullptr,
                const std::vector<Point>& initObstacles = {});

    // add obstacles
    void addObstacles(const std::vector<Point>& obs);
    void addObstacle (const Point& p);

    // single destination route
    bool addPath(Point src, const Point& dst);

    // multiple destination route
    bool addIOTree(const std::vector<Point>& sinks,
                 MultiPath& out);
    bool addIOTree(const std::vector<Point>& sinks);

    // return path
    const std::vector<std::vector<Point>>& paths() const { return paths_; }
    const MultiPath& multipaths() const { return mutipaths_; }
    void dumpGrid() const;

private:
    // tool function
    bool  inGrid(const Point& p) const;
    void  markSeg(const std::vector<Point>& seg);
    bool  isWall(int r,int c) const { return wall_[r][c]; }
    bool  connectAvailable(Point start, Point goal);

    // find the nearest point from the tree
    Point nearestTreePoint(const Point& dst) const;

    // limited BFS with priority
    bool bfsSingle(const Point& start, const Point& goal,
                   std::vector<Point>& outSeg);

    // data
    int R_, C_;
    std::vector<std::vector<bool>> wall_;   // wall
    std::vector<std::vector<bool>> tree_;   // built path
    std::vector<std::vector<Point>> paths_; // each branch
    std::vector<std::vector<Point>> parent_;// BFS backtrace

    std::shared_ptr<ResourceMgr> resmgr_;

    std::shared_ptr<DataIO> dio_;

    MultiPath mutipaths_;
    
    // Priority zone boundaries
    int M_start_;      // Memory tile start row
    int M_end_;        // Memory tile end row
    int SHIM_start_;   // SHIM tile start row
    int SHIM_end_;     // SHIM tile end row
};

#endif // ROUTING_PATH_H
