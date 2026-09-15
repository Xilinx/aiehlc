// Host unit tests for the control-packet transaction translator.
//
// Exercises rt_ctrl_txn_translate (the pure Write32/BlockWrite32 ->
// control-packet translation used by __Runtime_control_write_pkt_commit_transaction)
// against a hand-built XAie serialized-transaction byte buffer, so no HW and no
// XAie driver export call is needed.
//
// The translator now stamps a single broadcast/multicast stream id on every
// emitted packet and validates each op's tile against a cast target:
//   * ROW-MULTICAST (target_row >= 0): every op must land on that physical row.
//   * BROADCAST     (target_row <  0): every op must land on a CORE tile
//     (XAIEGBL_TILE_TYPE_AIETILE), classified via XAie_GetTileTypefromLoc, which
//     reads only NumCols / MemTileRowStart / AieTileRowStart etc. from the dev.
// Also covers the reject paths (MaskWrite32, row mismatch, non-core broadcast,
// capacity overflow).
//
// Build/run: script/test/build_ctrl_txn.sh
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "xaiengine.h"

// aie_runtime.c is compiled as C++ (its declarations are not extern "C"), so
// these are plain C++ declarations to match its mangled symbols.
// Pure translator under test (non-static in aie_runtime.c).
int rt_ctrl_txn_translate(XAie_DevInst *dev, const uint8_t *buf, uint8_t sid, int target_row, uint32_t *out,
                          uint32_t out_cap, uint32_t *nwords_out, int lastwriteack, uint32_t ret_sid);
// Header decoders (also in aie_runtime.c) used to check emitted packets.
int __Runtime_ctrl_parse_ctrl_hdr(uint32_t ctrl_hdr, uint32_t *addr_out, uint32_t *op_out, uint32_t *beats_out,
                                  uint32_t *ret_sid_out);
int __Runtime_ctrl_parse_pkt_hdr(uint32_t pkt_hdr, uint32_t *id_out, uint32_t *type_out, uint32_t *src_row_out,
                                 uint32_t *src_col_out);

// Host stub for the driver's tile classifier. The pure-translator test links
// only against aie_runtime.o (no XAie driver lib), so the broadcast tile-type
// check needs a definition. This mirrors the real xaie_helper.c logic: it reads
// only the simple grid fields we populate in make_dev() (NumCols, MemTile/AieTile
// row ranges), so the classification matches the on-target driver exactly.
extern "C" u8 XAie_GetTileTypefromLoc(XAie_DevInst *DevInst, XAie_LocType Loc) {
    if (Loc.Col >= DevInst->NumCols)
        return XAIEGBL_TILE_TYPE_MAX;
    if (Loc.Row == 0U) {
        u8 ColType = Loc.Col % 4U;
        return (ColType == 0U || ColType == 1U) ? XAIEGBL_TILE_TYPE_SHIMPL : XAIEGBL_TILE_TYPE_SHIMNOC;
    }
    if (Loc.Row >= DevInst->MemTileRowStart && Loc.Row < (DevInst->MemTileRowStart + DevInst->MemTileNumRows))
        return XAIEGBL_TILE_TYPE_MEMTILE;
    if (Loc.Row >= DevInst->AieTileRowStart && Loc.Row < (DevInst->AieTileRowStart + DevInst->AieTileNumRows))
        return XAIEGBL_TILE_TYPE_AIETILE;
    return XAIEGBL_TILE_TYPE_MAX;
}

// Minimal device with just the shift props the translator reads plus the tile
// grid layout XAie_GetTileTypefromLoc classifies against. Use the AIE2 gen2
// shift values so RegOff decode matches the runtime target.
static constexpr uint8_t kRowShift = 20; // tile-local addr = RegOff & ((1<<20)-1)
static constexpr uint8_t kColShift = 25; // row = (RegOff>>20)&((1<<5)-1); col = RegOff>>25

// Tile grid: row 0 = shim, row 1 = memtile, rows 2..7 = core tiles.
static constexpr uint8_t kNumCols = 8;
static constexpr uint8_t kMemTileRowStart = 1;
static constexpr uint8_t kMemTileNumRows = 1;
static constexpr uint8_t kAieTileRowStart = 2;
static constexpr uint8_t kAieTileNumRows = 6;

static XAie_DevInst make_dev() {
    XAie_DevInst dev;
    memset(&dev, 0, sizeof(dev));
    dev.DevProp.RowShift = kRowShift;
    dev.DevProp.ColShift = kColShift;
    dev.NumCols = kNumCols;
    dev.NumRows = kAieTileRowStart + kAieTileNumRows;
    dev.MemTileRowStart = kMemTileRowStart;
    dev.MemTileNumRows = kMemTileNumRows;
    dev.AieTileRowStart = kAieTileRowStart;
    dev.AieTileNumRows = kAieTileNumRows;
    return dev;
}

static uint64_t make_regoff(uint8_t col, uint8_t row, uint32_t tile_addr) {
    return ((uint64_t)col << kColShift) | ((uint64_t)row << kRowShift) | (tile_addr & ((1u << kRowShift) - 1));
}

// Append a Write32 op to a serialized-txn byte buffer.
static void push_write32(std::vector<uint8_t> &buf, uint8_t col, uint8_t row, uint32_t tile_addr, uint32_t value) {
    XAie_Write32Hdr w;
    memset(&w, 0, sizeof(w));
    w.OpHdr.Op = XAIE_IO_WRITE;
    w.OpHdr.Col = col;
    w.OpHdr.Row = row;
    w.RegOff = make_regoff(col, row, tile_addr);
    w.Value = value;
    w.Size = sizeof(XAie_Write32Hdr);
    const uint8_t *p = (const uint8_t *)&w;
    buf.insert(buf.end(), p, p + sizeof(w));
}

// Append a BlockWrite32 op (payload of nwords) to a serialized-txn byte buffer.
static void push_blockwrite32(std::vector<uint8_t> &buf, uint8_t col, uint8_t row, uint32_t tile_addr,
                              const uint32_t *data, uint32_t nwords) {
    XAie_BlockWrite32Hdr b;
    memset(&b, 0, sizeof(b));
    b.OpHdr.Op = XAIE_IO_BLOCKWRITE;
    b.OpHdr.Col = col;
    b.OpHdr.Row = row;
    b.Col = col;
    b.Row = row;
    b.RegOff = (uint32_t)make_regoff(col, row, tile_addr);
    b.Size = (uint32_t)(sizeof(XAie_BlockWrite32Hdr) + nwords * sizeof(uint32_t));
    const uint8_t *p = (const uint8_t *)&b;
    buf.insert(buf.end(), p, p + sizeof(b));
    const uint8_t *d = (const uint8_t *)data;
    buf.insert(buf.end(), d, d + nwords * sizeof(uint32_t));
}

// Append a MaskWrite32 op (rejected by the translator) to the buffer.
static void push_maskwrite32(std::vector<uint8_t> &buf, uint8_t col, uint8_t row, uint32_t tile_addr, uint32_t value,
                             uint32_t mask) {
    XAie_MaskWrite32Hdr m;
    memset(&m, 0, sizeof(m));
    m.OpHdr.Op = XAIE_IO_MASKWRITE;
    m.OpHdr.Col = col;
    m.OpHdr.Row = row;
    m.RegOff = make_regoff(col, row, tile_addr);
    m.Value = value;
    m.Mask = mask;
    m.Size = sizeof(XAie_MaskWrite32Hdr);
    const uint8_t *p = (const uint8_t *)&m;
    buf.insert(buf.end(), p, p + sizeof(m));
}

static std::vector<uint8_t> make_txn(uint32_t num_ops) {
    std::vector<uint8_t> buf;
    XAie_TxnHeader th;
    memset(&th, 0, sizeof(th));
    th.NumOps = num_ops;
    const uint8_t *p = (const uint8_t *)&th;
    buf.insert(buf.end(), p, p + sizeof(th));
    return buf;
}

// Row-multicast: a single Write32 on the target row -> one 3-word control
// packet [pkt_hdr | ctrl_hdr | value] stamped with the multicast sid.
static void test_single_write32_multicast() {
    XAie_DevInst dev = make_dev();
    const uint8_t col = 2, row = 3, sid = 7; // row 3 is a core row
    const uint32_t addr = 0x1234, val = 0xDEADBEEF;

    std::vector<uint8_t> txn = make_txn(1);
    push_write32(txn, col, row, addr, val);

    uint32_t out[16] = {0};
    uint32_t nwords = 0;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), sid, /*target_row=*/row, out, 16, &nwords, /*lastwriteack=*/0,
                                   /*ret_sid=*/0);
    assert(rc == 0);
    assert(nwords == 3);

    uint32_t id = 0, type = 0, srow = 0, scol = 0;
    assert(__Runtime_ctrl_parse_pkt_hdr(out[0], &id, &type, &srow, &scol) == 1);
    assert(id == sid);

    uint32_t got_addr = 0, op = 0, beats = 0, ret_sid = 0;
    assert(__Runtime_ctrl_parse_ctrl_hdr(out[1], &got_addr, &op, &beats, &ret_sid) == 1);
    assert(got_addr == addr);
    assert(op == 0);    // write, no return
    assert(beats == 1); // single word
    assert(out[2] == val);
    printf("[PASS] single Write32 -> row-multicast control packet\n");
}

// Row-multicast: BlockWrite32 of N words (N<=4) -> one packet with N data words.
static void test_blockwrite32_multicast() {
    XAie_DevInst dev = make_dev();
    const uint8_t col = 1, row = 2, sid = 5; // row 2 is a core row
    const uint32_t addr = 0x40;
    const uint32_t data[3] = {0x11111111, 0x22222222, 0x33333333};

    std::vector<uint8_t> txn = make_txn(1);
    push_blockwrite32(txn, col, row, addr, data, 3);

    uint32_t out[16] = {0};
    uint32_t nwords = 0;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), sid, /*target_row=*/row, out, 16, &nwords, /*lastwriteack=*/0,
                                   /*ret_sid=*/0);
    assert(rc == 0);
    assert(nwords == 2 + 3); // 2 headers + 3 data

    uint32_t id = 0, t = 0, sr = 0, sc = 0;
    assert(__Runtime_ctrl_parse_pkt_hdr(out[0], &id, &t, &sr, &sc) == 1);
    assert(id == sid);
    uint32_t got_addr = 0, op = 0, beats = 0, ret = 0;
    assert(__Runtime_ctrl_parse_ctrl_hdr(out[1], &got_addr, &op, &beats, &ret) == 1);
    assert(got_addr == addr);
    assert(op == 0);
    assert(beats == 3);
    assert(out[2] == data[0] && out[3] == data[1] && out[4] == data[2]);
    printf("[PASS] BlockWrite32 (3 words) -> row-multicast control packet\n");
}

// Multiple ops across several columns of the SAME target row -> every packet
// carries the single multicast sid (the whole-row cast reaches all columns).
static void test_multi_column_same_row() {
    XAie_DevInst dev = make_dev();
    const uint8_t row = 4, sid = 9; // row 4 is a core row
    const uint8_t colA = 4, colB = 6;
    const uint32_t d2[2] = {0xAAAA0000, 0xBBBB1111};

    std::vector<uint8_t> txn = make_txn(3);
    push_write32(txn, colA, row, 0x10, 0xCAFE);
    push_blockwrite32(txn, colB, row, 0x20, d2, 2);
    push_write32(txn, colA, row, 0x14, 0xF00D);

    uint32_t out[32] = {0};
    uint32_t nwords = 0;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), sid, /*target_row=*/row, out, 32, &nwords, /*lastwriteack=*/0,
                                   /*ret_sid=*/0);
    assert(rc == 0);
    // 3 + (2+2) + 3 = 10 words.
    assert(nwords == 10);

    uint32_t id = 0, t = 0, sr = 0, sc = 0;
    assert(__Runtime_ctrl_parse_pkt_hdr(out[0], &id, &t, &sr, &sc) == 1 && id == sid);
    assert(__Runtime_ctrl_parse_pkt_hdr(out[3], &id, &t, &sr, &sc) == 1 && id == sid);
    assert(__Runtime_ctrl_parse_pkt_hdr(out[7], &id, &t, &sr, &sc) == 1 && id == sid);
    printf("[PASS] multi-column same-row ops share the multicast stream id\n");
}

// Broadcast: every op targets a CORE tile -> success, all packets stamped with
// the broadcast id.
static void test_broadcast_core_ok() {
    XAie_DevInst dev = make_dev();
    const uint8_t sid = 0x10; // broadcast id (id[4]=1)
    const uint32_t d2[2] = {0x1, 0x2};

    std::vector<uint8_t> txn = make_txn(2);
    push_write32(txn, 2, 3, 0x10, 0xABCD);     // core tile (row 3)
    push_blockwrite32(txn, 5, 7, 0x20, d2, 2); // core tile (row 7)

    uint32_t out[16] = {0};
    uint32_t nwords = 0;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), sid, /*target_row=*/-1, out, 16, &nwords, /*lastwriteack=*/0,
                                   /*ret_sid=*/0);
    assert(rc == 0);
    assert(nwords == 3 + 4); // 3-word write + (2 hdr + 2 data)

    uint32_t id = 0, t = 0, sr = 0, sc = 0;
    assert(__Runtime_ctrl_parse_pkt_hdr(out[0], &id, &t, &sr, &sc) == 1 && id == sid);
    assert(__Runtime_ctrl_parse_pkt_hdr(out[3], &id, &t, &sr, &sc) == 1 && id == sid);
    printf("[PASS] broadcast to all-core tiles accepted\n");
}

// Broadcast with a non-core (memtile/shim) op must be rejected.
static void test_reject_broadcast_noncore() {
    XAie_DevInst dev = make_dev();
    std::vector<uint8_t> txn = make_txn(2);
    push_write32(txn, 2, 3, 0x10, 0x1); // core tile: ok
    push_write32(txn, 2, 1, 0x14, 0x2); // row 1 = memtile: not a core tile

    uint32_t out[16] = {0};
    uint32_t nwords = 0;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), 0x10, /*target_row=*/-1, out, 16, &nwords, /*lastwriteack=*/0,
                                   /*ret_sid=*/0);
    assert(rc != 0);
    printf("[PASS] broadcast to non-core tile rejected\n");
}

// MaskWrite32 must be rejected: nonzero rc.
static void test_reject_maskwrite() {
    XAie_DevInst dev = make_dev();
    const uint8_t col = 2, row = 2, sid = 3;
    std::vector<uint8_t> txn = make_txn(2);
    push_write32(txn, col, row, 0x10, 0x1);
    push_maskwrite32(txn, col, row, 0x14, 0x2, 0xFF);

    uint32_t out[16] = {0};
    uint32_t nwords = 0xDEAD;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), sid, /*target_row=*/row, out, 16, &nwords, /*lastwriteack=*/0,
                                   /*ret_sid=*/0);
    assert(rc != 0);
    printf("[PASS] MaskWrite32 rejected\n");
}

// Multicast op landing on the wrong row must be rejected.
static void test_reject_row_mismatch() {
    XAie_DevInst dev = make_dev();
    std::vector<uint8_t> txn = make_txn(1);
    push_write32(txn, 5, 5, 0x10, 0x1); // op on row 5
    uint32_t out[16] = {0};
    uint32_t nwords = 0;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), 4, /*target_row=*/3, out, 16, &nwords, /*lastwriteack=*/0,
                                   /*ret_sid=*/0);
    assert(rc != 0);
    printf("[PASS] multicast row mismatch rejected\n");
}

// Capacity overflow must be rejected (too little output capacity for the write).
static void test_reject_overflow() {
    XAie_DevInst dev = make_dev();
    std::vector<uint8_t> txn = make_txn(1);
    push_write32(txn, 1, 2, 0x10, 0x1);
    uint32_t out[2] = {0}; // need 3 words, only 2 cap
    uint32_t nwords = 0;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), 4, /*target_row=*/2, out, 2, &nwords, /*lastwriteack=*/0,
                                   /*ret_sid=*/0);
    assert(rc != 0);
    printf("[PASS] capacity overflow rejected\n");
}

// With lastwriteack=1, ONLY the final emitted packet is a WRITE-WITH-RETURN
// (control-info op = 0b10); every earlier op stays a fire-and-forget write
// (op = 0b00). Two single-word Write32s on the same row -> 2 packets (3 words
// each): the first op==0, the last op==2.
static void test_lastwriteack_only_last() {
    XAie_DevInst dev = make_dev();
    const uint8_t row = 3, sid = 7; // row 3 is a core row
    std::vector<uint8_t> txn = make_txn(2);
    push_write32(txn, 2, row, 0x10, 0x1111);
    push_write32(txn, 2, row, 0x14, 0x2222);

    uint32_t out[16] = {0};
    uint32_t nwords = 0;
    int rc = rt_ctrl_txn_translate(&dev, txn.data(), sid, /*target_row=*/row, out, 16, &nwords, /*lastwriteack=*/1,
                                   /*ret_sid=*/0);
    assert(rc == 0);
    assert(nwords == 3 + 3); // two single-word packets

    uint32_t a = 0, op = 0, beats = 0, ret = 0;
    // First packet: plain write, no return.
    assert(__Runtime_ctrl_parse_ctrl_hdr(out[1], &a, &op, &beats, &ret) == 1);
    assert(op == 0);
    // Last packet: write-with-return (write-ack).
    assert(__Runtime_ctrl_parse_ctrl_hdr(out[4], &a, &op, &beats, &ret) == 1);
    assert(op == 2);
    printf("[PASS] lastwriteack marks only the final packet write-with-return\n");
}

int main() {
    test_single_write32_multicast();
    test_lastwriteack_only_last();
    test_blockwrite32_multicast();
    test_multi_column_same_row();
    test_broadcast_core_ok();
    test_reject_broadcast_noncore();
    test_reject_maskwrite();
    test_reject_row_mismatch();
    test_reject_overflow();
    printf("ALL CTRL-TXN TESTS PASSED\n");
    return 0;
}
