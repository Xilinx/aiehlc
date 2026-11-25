/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#pragma once
#include <memory>
#include <unordered_map>
#include <vector>
#include <optional>
#include <mutex>
#include "routing/routingpath.h"
class RoutingTopology {
public:
  explicit RoutingTopology(std::string gen, std::string name = "", int startCol = -1, int endCol = -1,
                           int startRow = -1, int endRow = -1);
  std::shared_ptr<DataIO> createDataIO(std::string dioName);
  std::shared_ptr<DataIO> createDataIO(std::string dioName, std::optional<TypeBasedTileLoc> loc,
                                       DMADIRECTION direct = DMADIRECTION::MM2S);
  // Create a DataIO at a specific shim point without allocating a new shim channel.
  // Used for gather paths that share the output flow's shim tile.
  std::shared_ptr<DataIO> createDataIOAtPoint(std::string dioName, Point shimPoint, DMADIRECTION direction,
                                              int channel);
  // noFirstHop: an S2MM path must not leave its source tile towards this tile
  // (RoutingPath::setNoFirstHop); without such a route it routes as before.
  std::optional<std::shared_ptr<const RoutingPath>> createPath(int dioID, std::vector<Point> dsttiles,
                                                               std::optional<Point> noFirstHop = std::nullopt);
  const std::vector<std::shared_ptr<RoutingPath>> &paths() const { return paths_; }
  std::vector<Point> ReserveTiles(int nums, int dioID = -1);
  bool occupyLink(Point a, Point b, const int ioId, int &portNum, PortDirection &pda, PortDirection &pdb);
  bool occupyLinkAtPort(Point a, Point b, const int ioId, int portNum, PortDirection &pda, PortDirection &pdb);
  bool occupyPointDirection(Point a, int &portNum, PortDirection &pd, bool slave);
  std::shared_ptr<ResourceMgr> getRM() { return rm_; };

private:
    std::string gen_;
    std::string name_;
    std::shared_ptr<ResourceMgr> rm_;
    std::unordered_map<int, std::shared_ptr<DataIO>> dataios_;
    std::vector<std::shared_ptr<RoutingPath>> paths_;
    std::shared_ptr<DataIO> _createDataIO(std::string dioName, std::optional<Point> shim, DMADIRECTION direction=DMADIRECTION::MM2S, int channel=0);

};
