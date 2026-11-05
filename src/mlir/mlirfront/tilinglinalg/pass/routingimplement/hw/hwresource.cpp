/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/*=============================================================
 *  hw_resource.cpp    (private implementation)
 *===========================================================*/
#include "hw/hwresource.h"
#include <unordered_map>
#include <stdexcept>

namespace detail
{
/*-------------------- generation-specific data --------------*/
template<int GEN> struct GenTraits;

/*  Gen-1  : default colShift 23, rowShift 18  ----------------*/
template<> struct GenTraits<1>
{
    static const std::string& defaultVariant()
    {
        static const std::string def = "XCVC190";
        return def;
    }

    static const std::map<TileType, std::vector<PortTemplate>>& defaultPortTemplates() {
        static const std::map<TileType, std::vector<PortTemplate>> templates = {
            { TileType::Core, {
                 { PortDirection::North, PortRole::Master, 4, {} },
                { PortDirection::South, PortRole::Master, 4 , {} },
                { PortDirection::East, PortRole::Master, 4 , {} },
                { PortDirection::West, PortRole::Master, 4 , {} },
                { PortDirection::DMA, PortRole::Master, 4 , {} },
                { PortDirection::North, PortRole::Slave, 4 , {} },
                { PortDirection::South, PortRole::Slave, 4 , {} },
                { PortDirection::East, PortRole::Slave, 4 , {} },
                { PortDirection::West, PortRole::Slave, 4 , {} },
                { PortDirection::DMA, PortRole::Slave, 4 , {} }
              }
            },
            { TileType::Shim, {
                 { PortDirection::North, PortRole::Master, 4 , {} },
                { PortDirection::South, PortRole::Master, 2 , {2, 3} },//mux
                { PortDirection::East, PortRole::Master, 4 , {} },
                { PortDirection::West, PortRole::Master, 4 , {} },
                { PortDirection::DMA, PortRole::Master, 4 , {} },
                { PortDirection::North, PortRole::Slave, 4 , {} },
                { PortDirection::South, PortRole::Slave, 2 , {3, 7} },//demux
                { PortDirection::East, PortRole::Slave, 4 , {} },
                { PortDirection::West, PortRole::Slave, 4 , {} },
                { PortDirection::DMA, PortRole::Slave, 4 , {} }
              }
            },
            { TileType::Mem, {
                { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            },
            { TileType::NocShim, {
                 { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            }
        };
        return templates;
    }
    static const std::map<TileType, TileDmaLimits> &defaultDmaLimits() {
        static const std::map<TileType, TileDmaLimits> lim = {
            {TileType::Core, {16, 16}},
            {TileType::Shim, {16, 16}},
            {TileType::Mem, {16, 16}},
        };
        return lim;
    }

    static const std::unordered_map<std::string, AIEDeviceLayout>& table()
    {
        static const std::unordered_map<std::string, AIEDeviceLayout> db = {
            {"XCVC190",
             {
                 8,
                 8,
                 0x2000'0000'000,
                 23,
                 18,
                 {{0, 0, TileType::Shim}, {1, 2, TileType::Mem}, {3, 7, TileType::Core}},
                 {2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47}, // enabled noc shim
                 {3, 7},                                                       // mux
                 {2, 3},                                                       // demux
                 defaultPortTemplates(),
                 defaultDmaLimits(),
                 65536, // tileMemoryBytes (64KB)
                 10240, // stackReserveBytes (10KB)
                 49152  // usableDataBytes (48KB)
             }}};
        return db;
    }
};

/*  Gen-2  : default colShift 25, rowShift 20  ----------------*/
template<> struct GenTraits<2>
{
    static const std::string& defaultVariant()
    {
        static const std::string def = "XCVE2802";
        return def;
    }

    static const std::map<TileType, std::vector<PortTemplate>>& defaultPortTemplates() {
        static const std::map<TileType, std::vector<PortTemplate>> templates = {
            {TileType::Core,
             {{PortDirection::North, PortRole::Master, 4},
              {PortDirection::South, PortRole::Master, 4},
              {PortDirection::East, PortRole::Master, 4},
              {PortDirection::West, PortRole::Master, 4},
              {PortDirection::DMA, PortRole::Master, 4},
              {PortDirection::North, PortRole::Slave, 4},
              {PortDirection::South, PortRole::Slave, 4},
              {PortDirection::East, PortRole::Slave, 4},
              {PortDirection::West, PortRole::Slave, 4},
              {PortDirection::DMA, PortRole::Slave, 4}}},
            {TileType::Shim,
             {{PortDirection::North, PortRole::Master, 4},
              {PortDirection::South, PortRole::Master, 2, {3, 7}}, // mux - XAie_EnableShimDmaToAieStrmPort accepts only
                                                                   // 3,7
              {PortDirection::East, PortRole::Master, 4},
              {PortDirection::West, PortRole::Master, 4},
              {PortDirection::DMA, PortRole::Master, 4},
              {PortDirection::North, PortRole::Slave, 4},
              {PortDirection::South, PortRole::Slave, 2, {2, 3}}, // demux - XAie_EnableAieToShimDmaStrmPort on AIE-ML
                                                                  // accepts 2,3 (1,3 is AIE2PS)
              {PortDirection::East, PortRole::Slave, 4},
              {PortDirection::West, PortRole::Slave, 4},
              {PortDirection::DMA, PortRole::Slave, 4}}},
            {TileType::Mem,
             {{PortDirection::North, PortRole::Master, 6},
              {PortDirection::South, PortRole::Master, 4},
              {PortDirection::East, PortRole::Master, 0},
              {PortDirection::West, PortRole::Master, 0},
              {PortDirection::DMA, PortRole::Master, 6},
              {PortDirection::North, PortRole::Slave, 4},
              {PortDirection::South, PortRole::Slave, 6},
              {PortDirection::East, PortRole::Slave, 0},
              {PortDirection::West, PortRole::Slave, 0},
              {PortDirection::DMA, PortRole::Slave, 6}}},
            {TileType::NocShim,
             {{PortDirection::North, PortRole::Master, 4},
              {PortDirection::South, PortRole::Master, 2, {3, 7}}, // mux - XAie_EnableShimDmaToAieStrmPort on AIE2PS
              {PortDirection::East, PortRole::Master, 4},
              {PortDirection::West, PortRole::Master, 4},
              {PortDirection::DMA, PortRole::Master, 4},
              {PortDirection::North, PortRole::Slave, 4},
              {PortDirection::South, PortRole::Slave, 2, {2, 3}}, // demux - XAie_EnableAieToShimDmaStrmPort on AIE-ML
              {PortDirection::East, PortRole::Slave, 4},
              {PortDirection::West, PortRole::Slave, 4},
              {PortDirection::DMA, PortRole::Slave, 4}}}};
        return templates;
    }

    static const std::map<TileType, TileDmaLimits> &defaultDmaLimits() {
        static const std::map<TileType, TileDmaLimits> lim = {
            {TileType::Core, {16, 16}}, // AIE-ML core: 16 BDs, 16 locks
            {TileType::Shim, {16, 16}}, // shim: 16 BDs, 16 locks
            {TileType::Mem, {48, 64}},  // mem tile: 48 BDs, 64 locks
        };
        return lim;
    }

    static const std::unordered_map<std::string, AIEDeviceLayout>& table()
    {
        static const std::unordered_map<std::string, AIEDeviceLayout> db = {
            {"XCVE2802",
             {
                 11,
                 38,
                 0x2000'0000000,
                 25,
                 20,
                 {
                     {0, 0, TileType::Shim},
                     {1, 2, TileType::Mem},
                     {3, 11, TileType::Core},
                 },
                 {2, 3, 6, 7, 14, 15, 22, 23, 30, 31, 34, 35}, // noc shim support
                 {3, 7},                                       // mux (ext→AIE) - AIE2PS: ports 3,7
                 {2, 3},                                       // demux (AIE→ext) - AIE-ML: ports 2,3
                 defaultPortTemplates(),
                 defaultDmaLimits(),
                 65536, // tileMemoryBytes (64KB)
                 10240, // stackReserveBytes (10KB)
                 49152  // usableDataBytes (48KB)
             }}};
        return db;
    }
};

/*  Gen-5  (AIE2PS): default colShift 25, rowShift 20  --------
 *  All shim tiles (row 0) are ShimNOC — every column has
 *  DMA engines and NOC connectivity to DDR.                    */
template <> struct GenTraits<5> {
    static const std::string &defaultVariant() {
        static const std::string def = "PALMYRA";
        return def;
    }

    static const std::map<TileType, std::vector<PortTemplate>> &defaultPortTemplates() {
        static const std::map<TileType, std::vector<PortTemplate>> templates = {
            {TileType::Core,
             {{PortDirection::North, PortRole::Master, 4},
              {PortDirection::South, PortRole::Master, 4},
              {PortDirection::East, PortRole::Master, 4},
              {PortDirection::West, PortRole::Master, 4},
              {PortDirection::DMA, PortRole::Master, 4},
              {PortDirection::North, PortRole::Slave, 4},
              {PortDirection::South, PortRole::Slave, 4},
              {PortDirection::East, PortRole::Slave, 4},
              {PortDirection::West, PortRole::Slave, 4},
              {PortDirection::DMA, PortRole::Slave, 4}}},
            {TileType::Shim,
             {{PortDirection::North, PortRole::Master, 4},
              {PortDirection::South, PortRole::Master, 2, {3, 7}}, // mux - XAie_EnableShimDmaToAieStrmPort
              {PortDirection::East, PortRole::Master, 4},
              {PortDirection::West, PortRole::Master, 4},
              {PortDirection::DMA, PortRole::Master, 4},
              {PortDirection::North, PortRole::Slave, 4},
              {PortDirection::South, PortRole::Slave, 2, {1, 3}}, // demux - XAie_EnableAieToShimDmaStrmPort
              {PortDirection::East, PortRole::Slave, 4},
              {PortDirection::West, PortRole::Slave, 4},
              {PortDirection::DMA, PortRole::Slave, 4}}},
            {TileType::Mem,
             {{PortDirection::North, PortRole::Master, 6},
              {PortDirection::South, PortRole::Master, 4},
              {PortDirection::East, PortRole::Master, 0},
              {PortDirection::West, PortRole::Master, 0},
              {PortDirection::DMA, PortRole::Master, 6},
              {PortDirection::North, PortRole::Slave, 4},
              {PortDirection::South, PortRole::Slave, 6},
              {PortDirection::East, PortRole::Slave, 0},
              {PortDirection::West, PortRole::Slave, 0},
              {PortDirection::DMA, PortRole::Slave, 6}}},
            {TileType::NocShim,
             {{PortDirection::North, PortRole::Master, 4},
              {PortDirection::South, PortRole::Master, 2, {3, 7}}, // mux
              {PortDirection::East, PortRole::Master, 4},
              {PortDirection::West, PortRole::Master, 4},
              {PortDirection::DMA, PortRole::Master, 4},
              {PortDirection::North, PortRole::Slave, 4},
              {PortDirection::South, PortRole::Slave, 2, {1, 3}}, // demux
              {PortDirection::East, PortRole::Slave, 4},
              {PortDirection::West, PortRole::Slave, 4},
              {PortDirection::DMA, PortRole::Slave, 4}}}};
        return templates;
    }

    static const std::map<TileType, TileDmaLimits> &defaultDmaLimits() {
        static const std::map<TileType, TileDmaLimits> lim = {
            {TileType::Core, {16, 16}}, // AIE2PS core: 16 BDs, 16 locks
            {TileType::Shim, {16, 16}}, // shim: 16 BDs, 16 locks
            {TileType::Mem, {48, 64}},  // mem tile: 48 BDs, 64 locks
        };
        return lim;
    }

    // Helper: generate NOC shim set with all columns (AIE2PS: every shim is ShimNOC)
    static std::unordered_set<uint32_t> allColumnsNoc(uint32_t numCols) {
        std::unordered_set<uint32_t> cols;
        for (uint32_t c = 0; c < numCols; ++c)
            cols.insert(c);
        return cols;
    }

    static const std::unordered_map<std::string, AIEDeviceLayout> &table() {
        static const std::unordered_map<std::string, AIEDeviceLayout> db = {
            {"PALMYRA",
             {
                 7,  // rows: shim(0) + mem(1-2) + core(3-6)
                 12, // columns
                 0x2000'0000'000,
                 25,
                 20,
                 {
                     {0, 0, TileType::Shim},
                     {1, 2, TileType::Mem},
                     {3, 6, TileType::Core},
                 },
                 allColumnsNoc(12), // all 12 shim tiles are ShimNOC
                 {3, 7},            // mux (ext→AIE)
                 {1, 3},            // demux (AIE→ext)
                 defaultPortTemplates(),
                 defaultDmaLimits(),
                 65536, // tileMemoryBytes (64KB)
                 10240, // stackReserveBytes (10KB)
                 49152  // usableDataBytes (48KB)
             }},
            {"VEK385",
             {
                 7,  // rows: shim(0) + mem(1-2) + core(3-6)
                 38, // columns
                 0x2000'0000'000,
                 25,
                 20,
                 {
                     {0, 0, TileType::Shim},
                     {1, 2, TileType::Mem},
                     {3, 6, TileType::Core},
                 },
                 allColumnsNoc(38), // all 38 shim tiles are ShimNOC
                 {3, 7},            // mux (ext→AIE)
                 {1, 3},            // demux (AIE→ext)
                 defaultPortTemplates(),
                 defaultDmaLimits(),
                 65536, // tileMemoryBytes (64KB)
                 10240, // stackReserveBytes (10KB)
                 49152  // usableDataBytes (48KB)
             }}};
        return db;
    }
};

static int g_partitionStartCol = 0;

/*-------------------- concrete hidden resource --------------*/
template<int GEN>
class GenResource final : public IHwResource
{
public:
    explicit GenResource(const std::string& var =
                         GenTraits<GEN>::defaultVariant())
    {
        const auto& db = GenTraits<GEN>::table();
        auto it = db.find(var);
        if (it == db.end()) throw std::runtime_error("Unknown variant: "+var);
        layout_  = it->second;
        variant_ = var;
        if (g_partitionStartCol > 0) {
            std::unordered_set<uint32_t> rel;
            for (uint32_t c : layout_.nocShimCols)
                if (c >= static_cast<uint32_t>(g_partitionStartCol))
                    rel.insert(c - g_partitionStartCol);
            layout_.nocShimCols = std::move(rel);
        }
    }

    /* interface impl. ------------------*/
    uint32_t getRows()    const override { return layout_.rows;    }
    uint32_t getColumns() const override { return layout_.columns; }

    uint64_t getBaseAddr() const override { return layout_.baseAddr; }
    void     setBaseAddr(uint64_t a) override { layout_.baseAddr = a; }

    uint64_t tileAddr(uint32_t r,uint32_t c) const override
    { return layout_.tilePhysAddr(r,c); }

    TileType tileType(uint32_t r,uint32_t c) const override
    { return layout_.tileType(r,c); }

    //

    uint32_t absTileRow(TileType type, uint32_t relativeRow) const override
    { return layout_.absTileRow(type,relativeRow);}

    const std::vector<PortTemplate>& getPortsForTileType(TileType type) const override {
        return layout_.getPortsForTileType(type);
    }

    TileDmaLimits getDmaLimits(TileType type) const override { return layout_.getDmaLimits(type); }

    uint32_t getDmaWordBytes() const override { return 4; } // 32-bit DMA word = 4 bytes

    uint32_t getTileMemoryBytes() const override { return layout_.tileMemoryBytes; }
    uint32_t getStackReserveBytes() const override { return layout_.stackReserveBytes; }
    uint32_t getUsableDataBytes() const override { return layout_.usableDataBytes; }

    const std::unordered_set<uint32_t>& getShimNoc() const override{
        return layout_.getShimNoc();
    }

    const std::unordered_set<uint32_t>& getShimExtToAieMuxList() const override {
        return layout_.getShimExtToAieMuxList();
    }
    const std::unordered_set<uint32_t>& getShimAieToExtDemuxList() const override {
        return layout_.getShimAieToExtDemuxList();
    }

    std::string name() const override
    { return "Gen"+std::to_string(GEN)+":"+variant_; }

private:
    AIEDeviceLayout layout_{};
    std::string     variant_;
    std::unordered_map<int,int> shimNoc;
};
} // namespace detail

void setResourcePartitionStartCol(int absStartCol) { detail::g_partitionStartCol = absStartCol < 0 ? 0 : absStartCol; }

/*-------------------- public factory ------------------------*/
std::unique_ptr<IHwResource>
makeResource(const std::string& gen, const std::string& var)
{
    using namespace detail;
    if (gen == "Gen1")
        return var.empty()
               ? std::make_unique<GenResource<1>>()
               : std::make_unique<GenResource<1>>(var);

    if (gen == "Gen2")
        return var.empty()
               ? std::make_unique<GenResource<2>>()
               : std::make_unique<GenResource<2>>(var);

    if (gen == "Gen5")
        return var.empty() ? std::make_unique<GenResource<5>>() : std::make_unique<GenResource<5>>(var);

    throw std::runtime_error("Unsupported generation: " + gen);
}