/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

//--------------------------------------------------------------------
// routing_path.cpp — Incremental tree router (col→row sorting)
//--------------------------------------------------------------------
#include "routing/routingpath.h"

#include <queue>
#include <limits>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <cmath>
#include <iomanip>


const std::array<Point,4> kDirs{{ {-1,0},{1,0},{0,-1},{0,1} }}; // up,down,left,right

//--------------------------------------------------------------------
// ctor
//--------------------------------------------------------------------
RoutingPath::RoutingPath(std::shared_ptr<ResourceMgr> resmgr,std::shared_ptr<DataIO> dio, const std::vector<Point>& initObs)
    : R_(resmgr->rows()), C_(resmgr->cols()),
      wall_(R_,std::vector<bool>(C_,false)),
      tree_(R_,std::vector<bool>(C_,false)),
      parent_(R_,std::vector<Point>(C_,{-1,-1})),
      M_start_(1),       // Memory tiles typically start at row 1
      M_end_(1),         // Memory tiles typically end at row 1 (single row)
      SHIM_start_(0),    // SHIM tiles at row 0
      SHIM_end_(0)       // SHIM tiles at row 0
{
    addObstacles(initObs);
    resmgr_= resmgr;
    dio_ = dio;
    
    // Initialize priority zones based on resource manager if available
    if (resmgr_) {
        // You can customize these based on your hardware topology
        // For AIE architectures:
        // - SHIM tiles: row 0
        // - Memory tiles: row 1 (for AIEML/AIE2)
        // - Core tiles: row 2 and above
        
        // These can be set based on device type from resmgr
        SHIM_start_ = 0;
        SHIM_end_ = 0;
        M_start_ = 1;
        M_end_ = 1;
    }
}

//--------------------------------------------------------------------
bool RoutingPath::inGrid(const Point& p) const {
    return p.r>=0 && p.r<R_ && p.c>=0 && p.c<C_;
}

void RoutingPath::addObstacles(const std::vector<Point>& obs){
    for(auto p:obs) if(inGrid(p)) wall_[p.r][p.c]=true;
}
void RoutingPath::addObstacle(const Point& p){ if(inGrid(p)) wall_[p.r][p.c]=true; }

void RoutingPath::markSeg(const std::vector<Point>& seg){
    for(auto p:seg) tree_[p.r][p.c]=true;
}

//--------------------------------------------------------------------
// find nearest existing tree point to dst (Manhattan)
//--------------------------------------------------------------------
Point RoutingPath::nearestTreePoint(const Point& dst) const {
    int best=std::numeric_limits<int>::max();
    Point cand{dst.r,dst.c};
    for(const auto& seg:paths_){
        for(auto p:seg){
            int d=std::abs(p.r-dst.r)+std::abs(p.c-dst.c);
            if(d<best){best=d;cand=p;}
        }
    }
    return cand;
}

bool RoutingPath::connectAvailable(Point start, Point goal) {
    int portNum;
    if (isWall(start.r,start.c)||isWall(goal.r,goal.c)) return false;
    return resmgr_->linkAvailable(start, goal, portNum);
}

//--------------------------------------------------------------------
// BFS with priority (prefers Memory tiles, then SHIM tiles)
// Priority order: M tiles > SHIM tiles > Core tiles
//--------------------------------------------------------------------
bool RoutingPath::bfsSingle(const Point& start,const Point& goal,
                            std::vector<Point>& out){
    const int INF=std::numeric_limits<int>::max();
    std::vector<std::vector<int>> dist(R_,std::vector<int>(C_,INF));
    
    // Priority queue: (priority, distance, Point)
    // Lower priority value = explored first
    // Priority: 0 = M tiles, 1 = SHIM tiles, 2 = Core tiles
    auto getPriority = [this](const Point& p) -> int {
        if (p.r >= M_start_ && p.r <= M_end_) return 0;      // Memory tiles highest priority
        if (p.r >= SHIM_start_ && p.r <= SHIM_end_) return 1; // SHIM tiles second priority
        return 2;                                              // Core tiles lowest priority
    };
    
    // Priority queue: (priority, distance, Point)
    using PQEntry = std::tuple<int, int, Point>;
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;

    if(!inGrid(start)||!inGrid(goal)||isWall(start.r,start.c)||isWall(goal.r,goal.c))
        return false;
    
    dist[start.r][start.c]=0;
    parent_[start.r][start.c]={-1,-1};
    pq.push({getPriority(start), 0, start});

    while(!pq.empty()){
        auto [curPriority, curDist, cur] = pq.top();
        pq.pop();
        
        // Skip if we've already found a better path
        if(curDist > dist[cur.r][cur.c]) continue;
        
        if(cur==goal) break;
        
        for(auto d:kDirs){
            if(cur.r==0 && d.c!=0) continue; // row-0 rule: no left/right movement
            Point nxt{cur.r+d.r,cur.c+d.c};
            
            //check whether the connection has enough resources
            if(!inGrid(nxt)||!connectAvailable(Point{cur.r, cur.c}, Point{nxt.r,nxt.c})) continue;
            
            int newDist = dist[cur.r][cur.c] + 1;
            
            // Only update if we found a shorter path
            if(newDist < dist[nxt.r][nxt.c]) {
                dist[nxt.r][nxt.c] = newDist;
                parent_[nxt.r][nxt.c] = cur;
                int nxtPriority = getPriority(nxt);
                pq.push({nxtPriority, newDist, nxt});
            }
        }
    }
    
    if(dist[goal.r][goal.c]==INF) return false;
    
    // rebuild path
    for(Point p=goal;p!=start;p=parent_[p.r][p.c]) out.push_back(p);
    out.push_back(start);
    std::reverse(out.begin(),out.end());
    return true;
}

//--------------------------------------------------------------------
// single sink public
//--------------------------------------------------------------------
bool RoutingPath::addPath(Point src,const Point& dst){
    std::vector<Point> seg;
    if(!bfsSingle(src,dst,seg)) return false;
    markSeg(seg);
    paths_.push_back(seg);
    return true;
}

bool RoutingPath::addIOTree(const std::vector<Point>& sinksIn) {
    return addIOTree(sinksIn, mutipaths_);
}
//--------------------------------------------------------------------
// multi sink incremental tree
//--------------------------------------------------------------------
bool RoutingPath::addIOTree(const std::vector<Point>& sinksIn,MultiPath& out) {
    if (!dio_ || sinksIn.empty()) return false;
    Point src;
    std::vector<Point> sinks;
    if(dio_->dmadir() == DMADIRECTION::MM2S) {
        src = {dio_->rowpos(), dio_->colpos()};
        sinks=sinksIn;
    }else if (dio_->dmadir() == DMADIRECTION::S2MM)  {
        src = {sinksIn.back().r, sinksIn.back().c};
        sinks= std::vector<Point>{{dio_->rowpos(), dio_->colpos()}};
    } else {
        std::cout << " dma dir not found  " << (int)dio_->dmadir() << std::endl;
        return false;
    }
    // 1. sort sinks by col then row
    std::sort(sinks.begin(),sinks.end(),[](Point a,Point b){
        if(a.c!=b.c) return a.c<b.c;
        return a.r<b.r;
    });

    // 2. route to first sink
    std::vector<Point> seg;
    if(!bfsSingle(src,sinks[0],seg)) return false;
    markSeg(seg);
    paths_.push_back(seg);

    // 3. incremental for rest
    for(size_t i=1;i<sinks.size();++i){
        Point attach=nearestTreePoint(sinks[i]);
        seg.clear();
        if(!bfsSingle(attach,sinks[i],seg)) return false;
        // avoid duplicating first node if it already tree
        // mark new nodes only
        //for(auto p:seg){ if(!tree_[p.r][p.c]) paths_.back().push_back(p); }
        markSeg(seg);
        paths_.push_back(seg);
    }

    // fill MultiPath
    out.src=src;
    out.dsts=sinks;
    out.branches=paths_; // each entry in paths_ is a branch in this simple impl
    out.dataio = dio_;
    return true;
}

//--------------------------------------------------------------------
// Add this method to your RoutingPath class
void RoutingPath::dumpGrid() const {
    for(int r=R_-1;r>=0;--r){
        std::cout << std::setw(2) << r << " | ";  // Row number
        for(int c=0;c<C_;++c){
            std::string ch=".";
            if(wall_[r][c]) ch="X";
            else if(tree_[r][c]) {
                if (resmgr_->isTileReserved(r, c)) {
                    ch="*";//first print reserve
                }
                else ch="#";
            }
            //align
            int count = ((c == 0)?1:c);
            std::cout<<ch;
            while(count) {
                std::cout << " ";
                count = count/10;
            }
        }
        std::cout<<"\n";
    }

    // Print bottom border
    std::cout << "   +";
    for (int c = 0; c < C_; ++c){
        int count = ((c == 0)?1:c);
        std::cout << "-";
        while(count) {
            std::cout << "-";
            count = count/10;
        }
    }
    std::cout << "\n";

    // Print column numbers below
    std::cout << "     ";  // Align with row labels
    for (int c = 0; c < C_; ++c)
        std::cout << c << " ";
    std::cout << "\n\n";
}

