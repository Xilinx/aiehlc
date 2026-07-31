/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passroutingprovenancemap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>
#include <vector>

using namespace mlir;
using namespace routinghw;

namespace mlir {

class JsonWriter {
    llvm::raw_ostream &os;
    int indentLevel = 0;
    bool needsComma = false;

    void writeIndent() {
        for (int i = 0; i < indentLevel; ++i)
            os << "  ";
    }

    void writeCommaIfNeeded() {
        if (needsComma)
            os << ",";
        os << "\n";
    }

  public:
    explicit JsonWriter(llvm::raw_ostream &os) : os(os) {}

    void beginObject() {
        writeCommaIfNeeded();
        writeIndent();
        os << "{";
        indentLevel++;
        needsComma = false;
    }

    void beginObjectInline() {
        if (needsComma)
            os << ",";
        os << "\n";
        writeIndent();
        os << "{";
        indentLevel++;
        needsComma = false;
    }

    void endObject() {
        os << "\n";
        indentLevel--;
        writeIndent();
        os << "}";
        needsComma = true;
    }

    void beginArray(StringRef key) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << key << "\": [";
        indentLevel++;
        needsComma = false;
    }

    void endArray() {
        os << "\n";
        indentLevel--;
        writeIndent();
        os << "]";
        needsComma = true;
    }

    void key(StringRef k) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << k << "\": ";
        needsComma = false;
    }

    void keyValue(StringRef k, int64_t v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << k << "\": " << v;
        needsComma = true;
    }

    void keyValue(StringRef k, StringRef v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << k << "\": \"" << v << "\"";
        needsComma = true;
    }

    void keyValueBool(StringRef k, bool v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << k << "\": " << (v ? "true" : "false");
        needsComma = true;
    }

    void stringInArray(StringRef v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << v << "\"";
        needsComma = true;
    }

    void beginRoot() {
        os << "{";
        indentLevel++;
        needsComma = false;
    }

    void endRoot() { os << "\n}\n"; }
};

static int64_t getIntAttr(Operation *op, StringRef name, int64_t dflt = -1) {
    if (auto a = op->getAttrOfType<IntegerAttr>(name))
        return a.getInt();
    return dflt;
}

static std::string getStrAttr(Operation *op, StringRef name, StringRef dflt = "") {
    if (auto a = op->getAttrOfType<StringAttr>(name))
        return a.getValue().str();
    return dflt.str();
}

struct TileInfo {
    int64_t col = -1;
    int64_t row = -1;
    std::string kind = "unknown";
    std::string comments;
    bool isShim = false;
    int64_t ioid = -1;
    int64_t dmaDirection = -1;
    int64_t channelUsed = -1;
};

static TileInfo resolveTile(Value tileHandle) {
    TileInfo info;
    if (!tileHandle)
        return info;
    Operation *defOp = tileHandle.getDefiningOp();
    if (!defOp)
        return info;
    if (auto shim = dyn_cast<IOShimTileCreate>(defOp)) {
        info.col = getIntAttr(shim, "col");
        info.row = getIntAttr(shim, "row");
        info.comments = getStrAttr(shim, "comments");
        info.kind = "shim";
        info.isShim = true;
        info.ioid = getIntAttr(shim, "IOID");
        info.dmaDirection = getIntAttr(shim, "dmadirection");
        info.channelUsed = getIntAttr(shim, "channelused");
    } else if (auto core = dyn_cast<TileCreate>(defOp)) {
        info.col = getIntAttr(core, "col");
        info.row = getIntAttr(core, "row");
        info.comments = getStrAttr(core, "comments");
        info.kind = "core";
    }
    return info;
}

static void writeTileRef(JsonWriter &jw, StringRef key, const TileInfo &t) {
    jw.key(key);
    jw.beginObject();
    jw.keyValue("col", t.col);
    jw.keyValue("row", t.row);
    jw.keyValue("type", StringRef(t.kind));
    jw.endObject();
}

static void writePort(JsonWriter &jw, StringRef key, StringRef dir, int64_t idx) {
    jw.key(key);
    jw.beginObject();
    jw.keyValue("dir", dir);
    jw.keyValue("idx", idx);
    jw.endObject();
}

static bool writeConnectionOp(JsonWriter &jw, Operation *op) {
    if (auto c = dyn_cast<ConnectStreamSingleSwitchPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "circuit_connect");
        writeTileRef(jw, "tile", t);
        writePort(jw, "slave", getStrAttr(op, "slaveportdirection"), getIntAttr(op, "slaveportidx"));
        writePort(jw, "master", getStrAttr(op, "masterportdirection"), getIntAttr(op, "masterportidx"));
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<ConnectStreamPktSwitchPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "packet_connect");
        writeTileRef(jw, "tile", t);
        jw.key("recv_slave");
        jw.beginObject();
        jw.keyValue("dir", getStrAttr(op, "receiveslavedirection"));
        jw.keyValue("idx", getIntAttr(op, "receiveslaveportidx"));
        jw.keyValue("pktid", getIntAttr(op, "receiveslavepktid"));
        jw.keyValue("pkttype", getIntAttr(op, "receiveslavepkttype"));
        jw.endObject();
        jw.key("local_dma");
        jw.beginObject();
        jw.keyValue("dir", getStrAttr(op, "localdmadirection"));
        jw.keyValue("idx", getIntAttr(op, "localdmaportidx"));
        jw.keyValue("pktid", getIntAttr(op, "localdmapktid"));
        jw.keyValue("pkttype", getIntAttr(op, "localdmapkttype"));
        jw.endObject();
        writePort(jw, "forward_master", getStrAttr(op, "forwardmasterdirection"),
                  getIntAttr(op, "forwardmasterportidx"));
        bool preserve = false;
        if (auto a = op->getAttrOfType<BoolAttr>("preserveheader"))
            preserve = a.getValue();
        jw.keyValueBool("preserve_header", preserve);
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<EnableExtToAieShimPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "shim_ext_to_aie");
        writeTileRef(jw, "tile", t);
        writePort(jw, "port", getStrAttr(op, "portdirection"), getIntAttr(op, "portidx"));
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<EnableAieToExtShimPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "shim_aie_to_ext");
        writeTileRef(jw, "tile", t);
        writePort(jw, "port", getStrAttr(op, "portdirection"), getIntAttr(op, "portidx"));
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<ConnectStreamSwitchPort>(op)) {
        TileInfo src = resolveTile(op->getOperand(1));
        TileInfo dst = resolveTile(op->getOperand(2));
        jw.beginObjectInline();
        jw.keyValue("kind", "circuit_connect_pair");
        writeTileRef(jw, "src_tile", src);
        writePort(jw, "src_slave", getStrAttr(op, "srcslaveport"), getIntAttr(op, "srcslaveportidx"));
        writePort(jw, "src_master", getStrAttr(op, "srcmasterport"), getIntAttr(op, "srcmasterportidx"));
        writeTileRef(jw, "dst_tile", dst);
        writePort(jw, "dst_slave", getStrAttr(op, "dstslaveport"), getIntAttr(op, "dstslaveportidx"));
        writePort(jw, "dst_master", getStrAttr(op, "dstmasterport"), getIntAttr(op, "dstmasterportidx"));
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<CreateShimStreamSwitchPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "shim_stream_switch_port");
        writeTileRef(jw, "tile", t);
        jw.key("shim_master");
        jw.beginObject();
        jw.keyValue("port", getStrAttr(op, "shimmasterport"));
        jw.keyValue("idx", getIntAttr(op, "shimmasterportidx"));
        jw.keyValue("type", getIntAttr(op, "shimmasterporttype"));
        jw.endObject();
        jw.endObject();
        return true;
    }
    return false;
}

static bool isRoutingConnectionOp(Operation *op) {
    return isa<ConnectStreamSingleSwitchPort, ConnectStreamPktSwitchPort, EnableExtToAieShimPort,
               EnableAieToExtShimPort, ConnectStreamSwitchPort, CreateShimStreamSwitchPort>(op);
}

static void writeGroupTiles(JsonWriter &jw, Region &body) {
    std::vector<std::tuple<int64_t, int64_t, bool>> seenKeys;
    auto alreadySeen = [&](int64_t c, int64_t r, bool shim) {
        for (auto &k : seenKeys)
            if (std::get<0>(k) == c && std::get<1>(k) == r && std::get<2>(k) == shim)
                return true;
        seenKeys.emplace_back(c, r, shim);
        return false;
    };

    jw.beginArray("tiles");
    body.walk([&](Operation *op) {
        bool isShim = false;
        if (isa<IOShimTileCreate>(op))
            isShim = true;
        else if (!isa<TileCreate>(op))
            return;

        int64_t col = getIntAttr(op, "col");
        int64_t row = getIntAttr(op, "row");
        if (alreadySeen(col, row, isShim))
            return;

        std::string comments = getStrAttr(op, "comments");

        jw.beginObjectInline();
        jw.keyValue("col", col);
        jw.keyValue("row", row);
        jw.keyValue("type", isShim ? StringRef("shim") : StringRef("core"));
        if (!comments.empty())
            jw.keyValue("comments", StringRef(comments));
        if (isShim) {
            jw.keyValue("ioid", getIntAttr(op, "IOID"));
            jw.keyValue("dma_direction", getIntAttr(op, "dmadirection"));
            jw.keyValue("channel_used", getIntAttr(op, "channelused"));
        }
        jw.endObject();
    });
    jw.endArray();
}

void RoutingProvenanceMapPass::runOnOperation() {
    Operation *topOp = getOperation();
    auto moduleOp = dyn_cast<ModuleOp>(topOp);
    if (!moduleOp) {
        topOp->emitError("RoutingProvenanceMapPass requires a ModuleOp");
        signalPassFailure();
        return;
    }

    std::string outPath;
    if (!outputDir.empty()) {
        if (std::error_code EC = llvm::sys::fs::create_directories(outputDir)) {
            llvm::errs() << "RoutingProvenanceMapPass: failed to create directory " << outputDir << ": " << EC.message()
                         << "\n";
            outPath = "routingprovenancemap.json";
        } else {
            outPath = outputDir + "/routingprovenancemap.json";
        }
    } else {
        outPath = "routingprovenancemap.json";
    }

    std::error_code ec;
    llvm::raw_fd_ostream fileOs(outPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "RoutingProvenanceMapPass: failed to open " << outPath << ": " << ec.message() << "\n";
        signalPassFailure();
        return;
    }

    JsonWriter jw(fileOs);
    jw.beginRoot();

    jw.keyValue("version", (int64_t)1);
    if (startCol >= 0)
        jw.keyValue("startcol", (int64_t)startCol);
    if (!aieGen.empty())
        jw.keyValue("aie_gen", StringRef(aieGen));

    auto getI64 = [&](StringRef n) -> int64_t {
        if (auto a = moduleOp->getAttrOfType<IntegerAttr>(n))
            return a.getInt();
        return 0;
    };
    jw.key("module_attrs");
    jw.beginObject();
    jw.keyValue("tile_m", getI64("routing.tile_m"));
    jw.keyValue("tile_n", getI64("routing.tile_n"));
    jw.keyValue("tile_rows", getI64("routing.tile_rows"));
    jw.keyValue("tile_cols", getI64("routing.tile_cols"));
    jw.keyValue("effective_k", getI64("routing.effective_k"));
    jw.keyValue("full_k", getI64("routing.full_k"));
    jw.keyValue("k_rounds", getI64("routing.k_rounds"));
    jw.keyValue("m_rounds", getI64("routing.m_rounds"));
    jw.keyValue("n_rounds", getI64("routing.n_rounds"));
    jw.endObject();

    int totalConnections = 0;
    int capturedConnections = 0;

    jw.beginArray("routing_groups");
    int groupCounter = 0;
    moduleOp->walk([&](routing::RoutingCreate rc) {
        Region &body = rc.getBody();

        jw.beginObjectInline();
        jw.keyValue("id", std::string("group_") + std::to_string(groupCounter++));
        jw.keyValue("memo", getStrAttr(rc, "Memo"));

        int64_t scfIdx = -1;
        if (rc->getNumOperands() > 0) {
            if (auto cst = dyn_cast_or_null<arith::ConstantOp>(rc->getOperand(0).getDefiningOp())) {
                if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
                    scfIdx = ia.getInt();
            }
        }
        jw.keyValue("scf_idx", scfIdx);

        int64_t groupIoid = -1;
        for (Block &blk : body) {
            for (Operation &innerOp : blk) {
                if (isa<IOShimTileCreate>(&innerOp)) {
                    groupIoid = getIntAttr(&innerOp, "IOID");
                    break;
                }
            }
            if (groupIoid >= 0)
                break;
        }
        jw.keyValue("ioid", groupIoid);

        writeGroupTiles(jw, body);

        jw.beginArray("connections");
        body.walk([&](Operation *op) {
            if (!isRoutingConnectionOp(op))
                return;
            totalConnections++;
            if (writeConnectionOp(jw, op))
                capturedConnections++;
        });
        jw.endArray();

        jw.endObject();
    });
    jw.endArray();

    jw.endRoot();
    fileOs.close();

    if (totalConnections != capturedConnections) {
        llvm::errs() << "RoutingProvenanceMapPass: captured " << capturedConnections << " of " << totalConnections
                     << " connection ops\n";
    }
    std::cout << "Routing provenance map written to " << outPath << " (" << groupCounter << " groups, "
              << capturedConnections << " connections)" << std::endl;
}

} // namespace mlir
