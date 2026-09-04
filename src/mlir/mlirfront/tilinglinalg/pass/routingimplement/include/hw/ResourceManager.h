/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//====================================================================
// routingresource.h
//====================================================================
#ifndef ROUTINGRESOURCE_H
#define ROUTINGRESOURCE_H

#include "hwresource.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <map>
#include <atomic>
#include <array>
#include <stdexcept>
#include <mutex>
inline PortDirection opposite(PortDirection d){
    return d==PortDirection::North ? PortDirection::South :
           d==PortDirection::South ? PortDirection::North :
           d==PortDirection::East  ? PortDirection::West  :
           d==PortDirection::West  ? PortDirection::East  : d;
}
inline PortDirection getDir(Point a, Point b){
    if(a.r==b.r-1 && b.c==a.c) return PortDirection::North;
    if(a.r==b.r+1 && b.c==a.c) return PortDirection::South;
    if(b.c==a.c-1 && b.r==a.r) return PortDirection::West;
    if(b.c==a.c+1 && b.r==a.r) return PortDirection::East;
    throw std::runtime_error("Points not neighbours");
}

struct StreamPKTConnection {
    PortDirection SlaveReceiveForwardDirection;
    int SlaveReceiveForwardDirectionPortIdx;
    int SlaveReceivePktID;
    int SlaveReceivePktType;
    int localDMAForwardPortIdx;
    int localDMAForwardPktID;
    int localDMAForwardPktType;
    PortDirection MasterSendToNextTileDirection;
    int MasterSendToNextTileDirectionPortIdx;
};

struct StreamCCTConnection {
    PortDirection SlaveReceiveForwardDirection = PortDirection::NONE;
    int SlaveReceiveForwardDirectionPortIdx = 0;
    PortDirection localDMAForwardDirection = PortDirection::NONE;
    int localDMAForwardPortIdx = 0;
    PortDirection MasterSendToNextTileDirection = PortDirection::NONE;
    int MasterSendToNextTileDirectionPortIdx = 0;
    // Secondary master for fan-out: when a tile forwards to two different
    // directions (e.g., WEST to continue a chain AND NORTH to reach a tile
    // above), the second direction is stored here.
    PortDirection MasterSendToNextTileDirection2 = PortDirection::NONE;
    int MasterSendToNextTileDirectionPortIdx2 = 0;
};

struct TileListRoutingMap{
    std::vector<Point> tilelist;
    std::unordered_map<Point, StreamCCTConnection, Point::Hash> tilemap;
};

struct TileListPktRoutingNode {
    Point tile;
    void * tileOp;
    StreamPKTConnection pktconn;
};

enum class  PktHeaderProcessType {
    PKT_NO_DROP,
    PKT_DROP
};

struct BdSlot {
    bool used = false;
    int ownerId = -1;
};

struct LockSlot {
    bool used = false;
    int ownerId = -1;
};

struct PktIdSlot {
    bool used = false;
    int ownerId = -1;
};

/*
some port is enabled by hw design, in here it specify to SHIM tile input/output port
for example only port 3 and port 7 enabled by HW to responsible on data movement from
DDR
*/
struct PortSlot {
public:
void setportNum(int pnum) { portNum = pnum;}
int getportNum() {return portNum;}
public: 
    bool used=false; 
    bool invalid=false; 
    int ioId=-1;
private:
    int portNum=-1; 
};
struct DirBank  { std::vector<PortSlot> master; std::vector<PortSlot> slave; };
enum class IOType {Input, Output, TileDMA};
// Added for tile reservation
enum class ReservationStrategy {
    COLUMN_FIRST,
    ROW_FIRST
};
// stream type
enum class StreamType {
    FORWARDONLY,
    BROADCAST,
    // Control-packet flow: shim->tile(s) carrying config/register writes that
    // terminate at each target tile's CTRL master stream-switch port (header
    // preserved). Uses the same packet fan-out engine as BROADCAST.
    CONTROL
};
class RoutingTile {
public:
  RoutingTile(int r, int c, ::TileType tt, const std::vector<PortTemplate> &Portinfo, int numBds = 16,
              int numLocks = 16);

  uint32_t getPortnumFromPortIdx(PortDirection dir, PortRole role, uint32_t portidx);

  std::optional<int> allocate(IOType io, int portidx, PortDirection dir, int ioId);
  std::optional<int> occupyport(IOType io, PortDirection dir, int ioId);
  bool releaseByIo(IOType io, int portidx, PortDirection dir, int ioId);

  const DirBank &bank(PortDirection d) const { return banks_.at(d); }
  ::TileType type() const { return type_; }
  int row() const { return row_; }
  int col() const { return col_; }

  // BD resource management
  std::optional<int> allocateBd(int ownerId = -1);
  bool releaseBd(int bdId, int ownerId = -1);
  bool isBdFree(int bdId) const;
  int numBds() const { return static_cast<int>(bdPool_.size()); }
  int freeBdCount() const;

  // Lock resource management
  std::optional<int> allocateLock(int ownerId = -1);
  std::optional<int> allocateLockPair(int ownerId, int &lockA, int &lockB);
  bool releaseLock(int lockId, int ownerId = -1);
  bool isLockFree(int lockId) const;
  int numLocks() const { return static_cast<int>(lockPool_.size()); }
  int freeLockCount() const;

  // Added for tile reservation
  bool isReserved() const { return reserved_; }
  void setReserved(bool reserved, int ioId) {
      reserved_ = reserved;
      if (reserved) {
          reservedByIoId_ = ioId;
      } else {
          reservedByIoId_ = -1;
      }
    }
    int getReservedByIoId() const { return reservedByIoId_; }

private:
    int row_, col_;
    ::TileType type_;
    std::unordered_map<PortDirection, DirBank> banks_;
    std::vector<BdSlot> bdPool_;
    std::vector<LockSlot> lockPool_;
    // Added for tile reservation
    bool reserved_ = false;
    int reservedByIoId_ = -1;
};

#include <vector>
#include <optional>
#include <unordered_map>
#include <string>
#include <mutex>

enum class DMADIRECTION {
    MM2S,
    S2MM
};

struct DmaAllocation {
    int row;
    int col;
    DMADIRECTION direct;
    int channel;
    int ioId;
};

struct TileCoord {
    int row;
    int col;
    bool operator==(const TileCoord& o) const { return row == o.row && col == o.col; }
};

struct TileCoordHasher {
    std::size_t operator()(const TileCoord& t) const noexcept {
        return (static_cast<std::size_t>(t.row) << 32) ^ static_cast<std::size_t>(t.col);
    }
};

struct FoundDmaSlot {
    DMADIRECTION direct;
    Point loc;
    int channel;
};

class ShimTile {
public:
    // Construct a shim tile with given row/col and number of channels for each DMA direct
    ShimTile(int r, int c, int numMm2sChannels, int numS2mmChannels)
        : row_(r), col_(c),
          mm2s_(numMm2sChannels), s2mm_(numS2mmChannels) {}

    int row() const { return row_; }
    int col() const { return col_; }

    bool isReserved() const { return reserved_; }
    int getReservedByIoId() const { return reservedByIoId_; }

    // Reserve/unreserve entire shim tile (similar to RoutingTile reservation)
    void setReserved(bool reserved, int ioId) {
        reserved_ = reserved;
        if (reserved) {
            reservedByIoId_ = ioId;
        } else {
            reservedByIoId_ = -1;
        }
    }

    // Try to allocate a specific channel or find any available if preferredChannel < 0
    // Returns std::optional<int> channel index if successful.
    std::optional<int> allocate(DMADIRECTION direct, int preferredChannel, int ioId) {
        if (isHardReservedByOther(ioId)) return std::nullopt;

        auto& bank = channels(direct);

        // If specific channel is requested
        if (preferredChannel >= 0) {
            if (!isValidChannel(direct, preferredChannel)) return std::nullopt;
            if (!bank[preferredChannel].allocated ||
                bank[preferredChannel].allocatedBy == ioId) {
                // Allow idempotent re-allocation by same ioId
                bank[preferredChannel].allocated = true;
                bank[preferredChannel].allocatedBy = ioId;
                return preferredChannel;
            }
            return std::nullopt;
        }

        // Otherwise find first free channel
        for (int ch = 0; ch < static_cast<int>(bank.size()); ++ch) {
            if (!bank[ch].allocated) {
                bank[ch].allocated = true;
                bank[ch].allocatedBy = ioId;
                return ch;
            }
        }
        return std::nullopt;
    }

    // Occupy is similar to allocate but requires the channel be currently unallocated.
    // Useful if you separate "reserve" vs "occupy" semantics.
    std::optional<int> occupy(DMADIRECTION direct, int preferredChannel, int ioId) {
        if (isHardReservedByOther(ioId)) return std::nullopt;

        auto& bank = channels(direct);
        if (preferredChannel < 0) {
            for (int ch = 0; ch < static_cast<int>(bank.size()); ++ch) {
                if (!bank[ch].allocated) {
                    bank[ch].allocated = true;
                    bank[ch].allocatedBy = ioId;
                    return ch;
                }
            }
            return std::nullopt;
        } else {
            if (!isValidChannel(direct, preferredChannel)) return std::nullopt;
            if (!bank[preferredChannel].allocated) {
                bank[preferredChannel].allocated = true;
                bank[preferredChannel].allocatedBy = ioId;
                return preferredChannel;
            }
            return std::nullopt;
        }
    }

    // Release a channel; only the same ioId can release
    bool release(DMADIRECTION direct, int channel, int ioId) {
        if (!isValidChannel(direct, channel)) return false;
        auto& ch = channels(direct)[channel];
        if (ch.allocated && ch.allocatedBy == ioId) {
            ch.allocated = false;
            ch.allocatedBy = -1;
            return true;
        }
        return false;
    }

    // Force release (administrative), regardless of ioId
    bool forceRelease(DMADIRECTION direct, int channel) {
        if (!isValidChannel(direct, channel)) return false;
        auto& ch = channels(direct)[channel];
        ch.allocated = false;
        ch.allocatedBy = -1;
        return true;
    }

    // Query functions
    int numChannels(DMADIRECTION direct) const {
        return direct == DMADIRECTION::MM2S ? static_cast<int>(mm2s_.size())
                                         : static_cast<int>(s2mm_.size());
    }

    bool isChannelFree(DMADIRECTION direct, int channel) const {
        if (!isValidChannel(direct, channel)) return false;
        return !channelsConst(direct)[channel].allocated;
    }

    std::vector<int> freeChannels(DMADIRECTION direct) const {
        const auto& bank = channelsConst(direct);
        std::vector<int> out;
        for (int ch = 0; ch < static_cast<int>(bank.size()); ++ch) {
            if (!bank[ch].allocated) out.push_back(ch);
        }
        return out;
    }

    // Release all allocations by a given ioId on this tile
    int releaseAllByIoId(int ioId) {
        int released = 0;
        for (auto& ch : mm2s_) {
            if (ch.allocated && ch.allocatedBy == ioId) {
                ch.allocated = false;
                ch.allocatedBy = -1;
                ++released;
            }
        }
        for (auto& ch : s2mm_) {
            if (ch.allocated && ch.allocatedBy == ioId) {
                ch.allocated = false;
                ch.allocatedBy = -1;
                ++released;
            }
        }
        if (reserved_ && reservedByIoId_ == ioId) {
            reserved_ = false;
            reservedByIoId_ = -1;
        }
        return released;
    }

    bool hasAnyFreeChannelForEngine(DMADIRECTION direct) {
        int n = numChannels(direct);
        for (int ch = 0; ch < n; ++ch) {
            if (isChannelFree(direct, ch)) return true;
        }
        return false;
    }

    bool isEngineCompletelyFree(DMADIRECTION direct) {
        int n = numChannels(direct);
        for (int ch = 0; ch < n; ++ch) {
            if (!isChannelFree(direct, ch)) return false; // any allocated -> not completely free
        }
        return true; // all free
    }

private:
    struct ChannelState {
        bool allocated = false;
        int  allocatedBy = -1;
    };

    int row_;
    int col_;
    bool reserved_ = false;
    int  reservedByIoId_ = -1;

    std::vector<ChannelState> mm2s_;
    std::vector<ChannelState> s2mm_;

    bool isHardReservedByOther(int ioId) const {
        return reserved_ && reservedByIoId_ != ioId;
    }

    bool isValidChannel(DMADIRECTION direct, int ch) const {
        if (ch < 0) return false;
        return ch < numChannels(direct);
    }

    std::vector<ChannelState>& channels(DMADIRECTION direct) {
        return (direct == DMADIRECTION::MM2S) ? mm2s_ : s2mm_;
    }
    const std::vector<ChannelState>& channelsConst(DMADIRECTION direct) const {
        return (direct == DMADIRECTION::MM2S) ? mm2s_ : s2mm_;
    }
};

class ShimIOPort {
public:
    ShimIOPort(IOType iotype, PortDirection dir, int portnum): iotype_(iotype), dir_(dir), portnum_(portnum){};
    IOType iotype_;
    PortDirection dir_;
    int portnum_;
};

class DataIO {
public:
    DataIO(IOType tp, int r, int c, DMADIRECTION dir, int channel, std::string nm="", std::string cmt = "");
    int                id()      const { return id_; }
    int                rowpos()    const { return rowpos_; }
    int                colpos()    const { return colpos_; }
    int                channel() const { return channel_;};
    DMADIRECTION       dmadir()  const { return dmaDirection_; };
    IOType             type()    const { return type_; }
    const std::string& name()    const { return name_; }
    const std::string& comment() const { return comment_; }

    // Added for tile reservation
    const std::vector<Point>& getReservedTiles() const { return reservedTiles_; }
    void addReservedTile(const Point& p) { reservedTiles_.push_back(p); }
    void clearReservedTiles() { reservedTiles_.clear(); }
    std::optional<ShimIOPort> getshimport() {return shimport_;};
    void setshimport(std::optional<ShimIOPort> shimport) {shimport_ = shimport;};
private:
    static std::atomic<int> next_;
    int          id_, rowpos_, colpos_;
    int channel_;
    DMADIRECTION dmaDirection_;//MM2S S2MM
    IOType       type_;
    std::string  name_, comment_;
    std::vector<Point> reservedTiles_;
    std::optional<ShimIOPort> shimport_;
};

// ──────────────────────────────────────────────────────────────
// Core Data Memory Allocator
// ──────────────────────────────────────────────────────────────
// AIE-ML/AIE2PS core tile data memory layout:
//   0x00000 - 0x6FFFF: Reserved (448KB, shared with neighboring tiles)
//   0x70000 - 0x77FFF: Stack (32KB default)
//   0x78000 - 0x7FFFF: Data memory (32KB usable for buffers)
//
// The allocator manages the 0x78000+ region, allocating buffers with alignment.
// Shared for all tiles using the same kernel binary (one BCF/PRX for all tiles).

struct CoreMemSlot {
    std::string symbolName; // e.g., "buf_out_ping_0"
    uint32_t address;       // absolute DM address, e.g., 0x78000
    uint32_t size;          // bytes
    int ownerId;            // DataIO or tile owner
};

class CoreMemAllocator {
  public:
    CoreMemAllocator(uint32_t baseAddr = 0x78000, uint32_t totalSize = 0x8000);

    // Allocate a buffer at next available aligned address
    std::optional<uint32_t> allocate(const std::string &name, uint32_t size, uint32_t alignment = 32, int ownerId = -1);

    // Get all allocated slots (for BCF generation)
    const std::vector<CoreMemSlot> &getAllocations() const;

    // Get the address for a symbol name (returns 0 if not found)
    uint32_t getAddress(const std::string &name) const;

    // Reset allocator
    void reset();

    // Get base address of the data memory region
    uint32_t getBaseAddr() const { return baseAddr_; }

    // Get remaining free space
    uint32_t getFreeSpace() const { return (baseAddr_ + totalSize_) - nextFree_; }

  private:
    uint32_t baseAddr_;  // 0x78000
    uint32_t totalSize_; // 0x8000
    uint32_t nextFree_;  // next available address
    std::vector<CoreMemSlot> allocations_;
};

class ResourceMgr {
public:
  ResourceMgr(std::unique_ptr<IHwResource> resource, ::TileType defaultType = ::TileType::Core);
  static bool init(std::unique_ptr<IHwResource> resource, ::TileType defType = ::TileType::Core);

  /// Set partition bounds to confine tile/shim allocation to a sub-region.
  /// col range [startCol, endCol], row range [startRow, endRow] (inclusive).
  /// Call before any allocation. When not set, full mesh is used.
  void setPartitionBounds(int startCol, int endCol, int startRow, int endRow);
  bool hasPartition() const { return partitionStartCol_ >= 0; }
  static std::shared_ptr<ResourceMgr> instance();
  uint32_t allocdioid();

  std::shared_ptr<DataIO> createDataIO(IOType tp, int r = 0, int c = 0, DMADIRECTION dir = DMADIRECTION::MM2S,
                                       int channel = 0, std::string nm = "", std::string cmt = "");
  bool linkAvailable(Point a, Point b, int &portIdx) const;
  bool portDirAvailable(Point a, int &portIdx, PortDirection direction, bool master) const;

  bool occupyLink(Point a, Point b, const int ioId, int &portidx, PortDirection &pda, PortDirection &pdb);
  bool occupyPointDirection(Point a, int &portidx, PortDirection &pd, bool slave);
  bool releaseLink(Point a, Point b, int ioId, int portidx);
  std::optional<Point> freeShimNoc(std::optional<Point> dst = std::nullopt) const;
  std::optional<Point> freeShimNoc(std::optional<TypeBasedTileLoc> loc) const;
  std::optional<FoundDmaSlot> freeShimNoc(std::optional<TypeBasedTileLoc> ioPaireddstTileloc, DMADIRECTION direct,
                                          int requesterIoId) const;
  RoutingTile &tile(int r, int c);
  const RoutingTile &tile(int r, int c) const;
  int rows() const;
  int cols() const;
  static std::once_flag flag_;
  static std::shared_ptr<ResourceMgr> singleton_;

  // methods for tile reservation
  bool isTileReserved(int r, int c) const;
  bool isTileReserved(const Point &p) const { return isTileReserved(p.r, p.c); }

  // Reserve a specified tile for a DataIO
  bool reserveTile(int r, int c, int ioId);
  bool reserveTile(const Point &p, int ioId) { return reserveTile(p.r, p.c, ioId); }

  // Reserve multiple tiles for a DataIO using a strategy
  bool reserveTiles(int ioId, int numTiles, ReservationStrategy strategy, std::vector<Point> &allocatedTiles,
                    std::optional<::TileType> requestedType = ::TileType::Core,
                    std::optional<Point> startPoint = std::nullopt);

  // Release all tiles reserved for a specific DataIO
  void releaseReservedTiles(int ioId);

  // Get all tiles reserved by a specific DataIO
  std::vector<Point> getReservedTilesForDataIo(int ioId) const;
  IHwResource *getrsc() { return resource_.get(); };

  // Core memory allocator (shared for all tiles using same kernel binary)
  CoreMemAllocator &coreMemAllocator() { return coreMemAllocator_; }

  // Register shim column, channel, and direction to ioId mapping
  void registerShimChannelMapping(int shimCol, int channel, DMADIRECTION direction, int ioId);

  // Find ioId based on shim column, channel number, and direction
  std::optional<int> findIoIdByShimChannel(int shimCol, int channel, DMADIRECTION direction) const;

  // Find DataIO object by shim column, channel, and direction
  std::shared_ptr<DataIO> findDataIOByShimChannel(int shimCol, int channel, DMADIRECTION direction) const;

  // Tile-level BD resource management
  std::optional<int> allocateTileBd(int row, int col, int ownerId = -1);
  bool releaseTileBd(int row, int col, int bdId, int ownerId = -1);
  int freeTileBdCount(int row, int col) const;

  // Tile-level Lock resource management
  std::optional<int> allocateTileLock(int row, int col, int ownerId = -1);
  std::optional<int> allocateTileLockPair(int row, int col, int ownerId, int &lockA, int &lockB);
  bool releaseTileLock(int row, int col, int lockId, int ownerId = -1);
  int freeTileLockCount(int row, int col) const;

  // Global pkt_id pool management (5-bit AIE field, valid 0-31)
  std::optional<int> allocatePktId(int ownerId = -1);
  bool releasePktId(int pktId, int ownerId = -1);
  bool isPktIdFree(int pktId) const;

  // Partition bounds accessors
  int partitionStartCol() const { return partitionStartCol_; }
  int partitionEndCol() const { return partitionEndCol_; }
  int partitionStartRow() const { return partitionStartRow_; }
  int partitionEndRow() const { return partitionEndRow_; }

private:
  // Partition bounds (-1 = use full mesh)
  int partitionStartCol_ = -1, partitionEndCol_ = -1;
  int partitionStartRow_ = -1, partitionEndRow_ = -1;

  // Check if a column/row is within partition bounds
  bool isColInPartition(int c) const { return !hasPartition() || (c >= partitionStartCol_ && c <= partitionEndCol_); }
  bool isRowInPartition(int r) const { return !hasPartition() || (r >= partitionStartRow_ && r <= partitionEndRow_); }
  bool isTileInPartition(int r, int c) const { return isColInPartition(c) && isRowInPartition(r); }

  static constexpr int kMaxPktId = 32; // 5-bit AIE pkt_id field
  std::array<PktIdSlot, kMaxPktId> pktIdPool_{};
  void InitSHIMNocList();

  void addShimTile(std::shared_ptr<ShimTile> shim);

  uint32_t lastdioid;
  std::vector<std::vector<RoutingTile>> tiles_;
  std::unordered_map<TileCoord, std::shared_ptr<ShimTile>, TileCoordHasher> shimTiles_;

  std::unique_ptr<IHwResource> resource_;
  std::unordered_map<int, std::shared_ptr<DataIO>> DataIOMap;
  CoreMemAllocator coreMemAllocator_; // Shared core memory allocator for BCF generation

  // Hash function for (shimCol, channel, direction) tuple
  struct ShimChannelDirHash {
      std::size_t operator()(const std::tuple<int, int, DMADIRECTION> &t) const {
          auto h1 = std::hash<int>()(std::get<0>(t));
          auto h2 = std::hash<int>()(std::get<1>(t));
          auto h3 = std::hash<int>()(static_cast<int>(std::get<2>(t)));
          return h1 ^ (h2 << 1) ^ (h3 << 2);
      }
    };
    
    // Mapping from (shimCol, channel, direction) to ioId
    std::unordered_map<std::tuple<int, int, DMADIRECTION>, int, ShimChannelDirHash> shimChannelToIoIdMap_;
};

#endif // ROUTINGRESOURCE_H
