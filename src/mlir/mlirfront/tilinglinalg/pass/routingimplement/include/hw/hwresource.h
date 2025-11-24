/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

/*=============================================================
 *  hw_resource.hpp     (public API)
 *===========================================================*/
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <map>

// ──────────────────────────────────────────────────────────────
// Point — grid coordinate (row, col)
// ──────────────────────────────────────────────────────────────
struct Point {
    int r;  // row
    int c;  // column
    bool operator==(const Point& o) const { return r == o.r && c == o.c; }
    bool operator!=(const Point& o) const { return !(*this == o); }
    bool operator<(const Point& o) const {
        if (r != o.r) return r < o.r;
        return c < o.c;
    }
    struct Hash {
        std::size_t operator()(const Point& p) const {
            // Combine the hash of row and column
            std::size_t h1 = std::hash<int>{}(p.r);
            std::size_t h2 = std::hash<int>{}(p.c);
            
            // Combine the hashes
            return h1 ^ (h2 << 1);  // Simple but effective for two integers
        }
    };
};
//#include "routing/routingpath.h"
enum class TileType : uint8_t { Core, Mem, Shim, NocShim, PLShim, PLNocShim, Unknown };
enum class PortDirection { North, South, East, West, DMA, Control, NONE};
enum class PortRole      { Master, Slave };
struct PortTemplate {
    PortDirection dir;   // port direction
    PortRole      role;  // Master / Slave
    int           ports; // available number of ports
    std::vector<uint32_t> available_ports; //vaiable ports list, this is optinonal infor can be empty if port is 0-(ports-1)
};

struct TypeBasedTileLoc {
    TileType ttype;
    struct Point loc;
};

static const std::string& PortDirectiontoString(PortDirection dir) {
    // The map is created only on the first call to this function
    static const std::map<PortDirection, std::string> PortDirectionMap = {
        { PortDirection::North,   "NORTH"   },
        { PortDirection::South,   "SOUTH"   },
        { PortDirection::East,    "EAST"    },
        { PortDirection::West,    "WEST"    },
        { PortDirection::DMA,     "DMA"     },
        { PortDirection::Control, "CONTROL" },
        { PortDirection::NONE, "NONE" }
    };

    auto it = PortDirectionMap.find(dir);
    if (it != PortDirectionMap.end()) {
        return it->second;
    }
    
    // Handle the case where the enum value is not in the map
    static const std::string unknown = "Unknown";
    return unknown;
};


struct PortLocation {
    int row;
    int col;
    PortDirection dir;   // port direction
    PortRole      role;  // Master / Slave
    int           portidx; // port index
};

struct TileSegment          // contiguous column range → TileType
{
    uint32_t firstRow;      // inclusive
    uint32_t lastRow;       // inclusive
    TileType type;
};

struct AIEDeviceLayout
{
    uint32_t rows     = 0;
    uint32_t columns  = 0;

    uint64_t baseAddr = 0x20000000000;       // mutable at run-time
    uint8_t  colShift = 23;      // default matches Gen-1
    uint8_t  rowShift = 18;

    std::vector<TileSegment> segments;
    std::unordered_set<uint32_t> nocShimCols;
    std::unordered_set<uint32_t> shimexttoaie_mux;
    std::unordered_set<uint32_t> shimaietoext_demux;
    std::map<TileType, std::vector<PortTemplate>> portTemplates;
    
    // helpers -------------------------------------------------
    uint64_t tilePhysAddr(uint32_t r, uint32_t c) const
    {
        return baseAddr
             + (static_cast<uint64_t>(c) << colShift)
             + (static_cast<uint64_t>(r) << rowShift);
    }
    TileType tileType(uint32_t row, uint32_t /*col*/) const
    {
        for (const auto& s : segments)
            if (row >= s.firstRow && row <= s.lastRow) return s.type;
        return TileType::Unknown;
    }

    uint32_t absTileRow(TileType type, uint32_t relativeRow) const
    {
        for (const auto& s : segments)
            if (s.type == type) return relativeRow + s.firstRow;
        return relativeRow;
    }

    const std::vector<PortTemplate>& getPortsForTileType(TileType type) const {
        static const std::vector<PortTemplate> emptyList;
        auto it = portTemplates.find(type);
        return (it != portTemplates.end()) ? it->second : emptyList;
    }

    const std::unordered_set<uint32_t>& getShimNoc() const {
        return nocShimCols;
    }
    const std::unordered_set<uint32_t>& getShimExtToAieMuxList() const {
        return shimexttoaie_mux;
    }
    const std::unordered_set<uint32_t>& getShimAieToExtDemuxList() const {
        return shimaietoext_demux;
    }
};

/* ---------- abstract interface ---------------------------- */
class IHwResource
{
public:
    virtual ~IHwResource() = default;

    virtual uint32_t getRows()    const = 0;
    virtual uint32_t getColumns() const = 0;

    virtual uint64_t getBaseAddr() const = 0;
    virtual void     setBaseAddr(uint64_t addr) = 0;

    virtual uint64_t tileAddr (uint32_t r, uint32_t c) const = 0;
    virtual TileType tileType(uint32_t r, uint32_t c) const = 0;
    virtual const std::vector<PortTemplate>& getPortsForTileType(TileType type) const = 0;

    virtual const std::unordered_set<uint32_t>& getShimNoc() const = 0;
    virtual const std::unordered_set<uint32_t>& getShimExtToAieMuxList() const = 0;
    virtual const std::unordered_set<uint32_t>& getShimAieToExtDemuxList() const = 0;

    virtual uint32_t absTileRow(TileType type, uint32_t relativeRow) const = 0;

    virtual std::string name() const = 0;
};

/* ---------- factory (implementation inside .cpp) ---------- */
std::unique_ptr<IHwResource>
makeResource(const std::string& generation,
             const std::string& variant = "");
