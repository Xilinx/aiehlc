/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passroutingresourcemap.h"
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

// Self-contained JSON writer (mirrors passroutingprovenancemap.cpp).
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
    bool isShim = false;
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
        info.kind = "shim";
        info.isShim = true;
    } else if (auto core = dyn_cast<TileCreate>(defOp)) {
        info.col = getIntAttr(core, "col");
        info.row = getIntAttr(core, "row");
        info.kind = "core";
    }
    return info;
}

static void writeTileRef(JsonWriter &jw, StringRef key, const TileInfo &t) {
    jw.key(key);
    jw.beginObject();
    jw.keyValue("col", t.col);
    jw.keyValue("row", t.row);
    jw.keyValue("kind", StringRef(t.kind));
    jw.endObject();
}

static void writePort(JsonWriter &jw, StringRef key, StringRef dir, int64_t idx) {
    jw.key(key);
    jw.beginObject();
    jw.keyValue("dir", dir);
    jw.keyValue("idx", idx);
    jw.endObject();
}

// Pkt mask derivation matches routinghwlower.cpp EmitC:
//   recv/forward slave slot -> mask = 0x0  (forward-all)   [routinghwlower.cpp:237]
//   local-DMA slave slot    -> mask = 0x1f (exact 5-bit)   [routinghwlower.cpp:243]
static constexpr int64_t RECV_SLOT_MASK = 0x0;
static constexpr int64_t DMA_SLOT_MASK = 0x1f;

// Pkt msel/arbiter match routinghwlower.cpp EmitC hardcoded constants:
//   slave slot (recv + local DMA): msel = 0, arbiter = 0  [routinghwlower.cpp:238-239]
//   forward master port: Arbitor = 0                      [routinghwlower.cpp:280,300]
// (XAie_StrmPktSwSlaveSlotEnable(..., Mask, MSel, Arbitor) and
//  XAie_StrmPktSwMstrPortEnable(..., DropHeader, Arbitor, MSelEn); see xaie_ss.c.)
//
// The master's MSelEn is NOT an independent constant: the packet-switch routing
// rule is that a slave slot with (Arbitor=A, MSel=M) reaches every master port
// whose Arbitor==A and whose MSelEn has bit M set. So the master port that merges
// the recv + local-DMA slave slots derives its enable mask by OR-ing in one bit
// per feeding slave slot:  MSelEn |= (1 << MSel).  With every slot at MSel=0 this
// yields MSelEn=1, identical to what EmitC programs today; it generalises when
// slots are ever assigned distinct MSel values.
static constexpr int64_t SLOT_MSEL = 0;
static constexpr int64_t SLOT_ARBITER = 0;
static constexpr int64_t MASTER_ARBITER = 0;

// MSelEn |= (1 << MSel) over the slave slots that actually feed the master.
// The PKT->CIRC transition op has neither slave present (forward-only); it still
// needs a non-zero enable, so it defaults to the MSel-0 bit (matches EmitC's
// hardcoded MSelEn=1).
static int64_t computeMasterMSelEn(bool recvPresent, int64_t recvMsel, bool dmaPresent, int64_t dmaMsel) {
    int64_t mselEn = 0;
    if (recvPresent)
        mselEn |= (1 << recvMsel);
    if (dmaPresent)
        mselEn |= (1 << dmaMsel);
    if (mselEn == 0)
        mselEn = (1 << SLOT_MSEL);
    return mselEn;
}

static bool writeConnectionOp(JsonWriter &jw, Operation *op) {
    if (auto c = dyn_cast<ConnectStreamPktSwitchPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "packet_connect");
        writeTileRef(jw, "tile", t);

        std::string recvDir = getStrAttr(op, "receiveslavedirection");
        std::string dmaDir = getStrAttr(op, "localdmadirection");
        bool recvPresent = (recvDir != "NONE");
        bool dmaPresent = (dmaDir != "NONE");

        jw.key("recv_slave");
        jw.beginObject();
        jw.keyValue("dir", StringRef(recvDir));
        jw.keyValue("idx", getIntAttr(op, "receiveslaveportidx"));
        jw.keyValue("pktid", getIntAttr(op, "receiveslavepktid"));
        jw.keyValue("pkttype", getIntAttr(op, "receiveslavepkttype"));
        jw.keyValue("mask", RECV_SLOT_MASK);
        jw.keyValue("msel", SLOT_MSEL);
        jw.keyValue("arbiter", SLOT_ARBITER);
        jw.endObject();

        jw.key("local_dma");
        jw.beginObject();
        jw.keyValue("dir", StringRef(dmaDir));
        jw.keyValue("idx", getIntAttr(op, "localdmaportidx"));
        jw.keyValue("pktid", getIntAttr(op, "localdmapktid"));
        jw.keyValue("pkttype", getIntAttr(op, "localdmapkttype"));
        jw.keyValue("mask", DMA_SLOT_MASK);
        jw.keyValue("msel", SLOT_MSEL);
        jw.keyValue("arbiter", SLOT_ARBITER);
        jw.endObject();

        // MSelEn |= (1 << MSel) over the recv + local-DMA slave slots feeding this
        // master port; see computeMasterMSelEn / xaie_ss.c packet-switch routing.
        int64_t masterMSelEn = computeMasterMSelEn(recvPresent, SLOT_MSEL, dmaPresent, SLOT_MSEL);

        jw.key("forward_master");
        jw.beginObject();
        jw.keyValue("dir", getStrAttr(op, "forwardmasterdirection"));
        jw.keyValue("idx", getIntAttr(op, "forwardmasterportidx"));
        jw.keyValue("mselen", masterMSelEn);
        jw.keyValue("arbiter", MASTER_ARBITER);
        jw.endObject();

        bool preserve = false;
        if (auto a = op->getAttrOfType<BoolAttr>("preserveheader"))
            preserve = a.getValue();
        jw.keyValueBool("preserve_header", preserve);
        jw.endObject();
        return true;
    }
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

void RoutingResourceMapPass::runOnOperation() {
    Operation *topOp = getOperation();
    auto moduleOp = dyn_cast<ModuleOp>(topOp);
    if (!moduleOp) {
        topOp->emitError("RoutingResourceMapPass requires a ModuleOp");
        signalPassFailure();
        return;
    }

    std::string outPath;
    if (!outputDir.empty()) {
        if (std::error_code EC = llvm::sys::fs::create_directories(outputDir)) {
            llvm::errs() << "RoutingResourceMapPass: failed to create directory " << outputDir << ": " << EC.message()
                         << "\n";
            outPath = "routingresourcemap.json";
        } else {
            outPath = outputDir + "/routingresourcemap.json";
        }
    } else {
        outPath = "routingresourcemap.json";
    }

    std::error_code ec;
    llvm::raw_fd_ostream fileOs(outPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "RoutingResourceMapPass: failed to open " << outPath << ": " << ec.message() << "\n";
        signalPassFailure();
        return;
    }

    JsonWriter jw(fileOs);
    jw.beginRoot();

    jw.keyValue("version", (int64_t)1);
    if (startCol >= 0)
        jw.keyValue("partition_start_col", (int64_t)startCol);
    if (!aieGen.empty())
        jw.keyValue("aie_gen", StringRef(aieGen));

    int totalConnections = 0;
    int capturedConnections = 0;

    jw.beginArray("connections");
    moduleOp->walk([&](routing::RoutingCreate rc) {
        Region &body = rc.getBody();
        body.walk([&](Operation *op) {
            if (!isRoutingConnectionOp(op))
                return;
            totalConnections++;
            if (writeConnectionOp(jw, op))
                capturedConnections++;
        });
    });
    jw.endArray();

    jw.endRoot();
    fileOs.close();

    if (totalConnections != capturedConnections) {
        llvm::errs() << "RoutingResourceMapPass: captured " << capturedConnections << " of " << totalConnections
                     << " connection ops\n";
    }
    std::cout << "Routing resource map written to " << outPath << " (" << capturedConnections << " connections)"
              << std::endl;
}

} // namespace mlir
