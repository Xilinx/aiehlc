// Host unit test for KERNELCONFIGOFFLOAD core-tile DMA MMIO encoding.
//
// Phase 1 (this file): prove we can capture the GOLDEN register writes the
// real aie-rt gen5 driver produces for a core-tile S2MM buffer-descriptor,
// host-native, with no hardware. The whole driver is compiled for x86 with the
// DEBUG IO backend; writes are captured structurally via the XAie transaction
// export (Write32 / BlockWrite32 ops), NOT by parsing printf output.
//
// Later phases diff a standalone encoder (aie_kernel_runtime.h, same directory
// as this test's parent) against these golden words.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "xaiengine.h"
}

// The standalone encoder under test.
#include "aie_kernel_runtime.h"

// ---- AIE2PS (gen5) device geometry (matches include/aie_device_map.h) ----
#define KC_BASE_ADDR 0x20000000000ULL
#define KC_COL_SHIFT 25
#define KC_ROW_SHIFT 20
#define KC_NUM_COLS 36
#define KC_NUM_ROWS 7
#define KC_SHIM_ROW 0
#define KC_MEMTILE_ROW_ST 1
#define KC_MEMTILE_NUM_ROWS 2
#define KC_CORE_ROW_ST 3
#define KC_CORE_NUM_ROWS 4

// AIE2PS core-tile memory-module register offsets (tile-local).
#define KC_DMA_BD0_0 0x1D000u
#define KC_DMA_BD_STRIDE 0x20u
#define KC_DMA_S2MM0_START_Q 0x1DE04u

// One captured register write, decoded to tile-local coordinates.
struct CapWrite {
    uint8_t col;
    uint8_t row;
    uint32_t off; // tile-local byte offset
    uint32_t val;
};

// Walk an exported serialized-transaction buffer and collect every Write32 /
// BlockWrite32 as (col,row,tile_local_off,value). Rejects masked ops (the
// offload cannot express masked writes) by aborting the test.
static std::vector<CapWrite> parse_txn(const uint8_t *buf) {
    std::vector<CapWrite> out;
    const XAie_TxnHeader *hdr = reinterpret_cast<const XAie_TxnHeader *>(buf);
    const uint8_t *p = buf + sizeof(XAie_TxnHeader);
    for (uint32_t i = 0; i < hdr->NumOps; i++) {
        const XAie_OpHdr *op = reinterpret_cast<const XAie_OpHdr *>(p);
        switch (op->Op) {
        case XAIE_IO_WRITE: {
            const XAie_Write32Hdr *w = reinterpret_cast<const XAie_Write32Hdr *>(p);
            uint64_t regoff = w->RegOff;
            CapWrite c;
            c.col = (uint8_t)(regoff >> KC_COL_SHIFT);
            c.row = (uint8_t)((regoff >> KC_ROW_SHIFT) & ((1u << (KC_COL_SHIFT - KC_ROW_SHIFT)) - 1));
            c.off = (uint32_t)(regoff & ((1u << KC_ROW_SHIFT) - 1));
            c.val = w->Value;
            out.push_back(c);
            p += w->Size;
            break;
        }
        case XAIE_IO_BLOCKWRITE: {
            const XAie_BlockWrite32Hdr *bw = reinterpret_cast<const XAie_BlockWrite32Hdr *>(p);
            uint32_t nwords = (bw->Size - (uint32_t)sizeof(XAie_BlockWrite32Hdr)) / 4u;
            const uint32_t *data = reinterpret_cast<const uint32_t *>(p + sizeof(XAie_BlockWrite32Hdr));
            uint32_t base_off = bw->RegOff & ((1u << KC_ROW_SHIFT) - 1);
            for (uint32_t k = 0; k < nwords; k++) {
                CapWrite c;
                c.col = bw->Col;
                c.row = bw->Row;
                c.off = base_off + k * 4u;
                c.val = data[k];
                out.push_back(c);
            }
            p += bw->Size;
            break;
        }
        default:
            fprintf(stderr, "FAIL: unexpected txn op %u (masked/custom not supported by offload)\n", op->Op);
            exit(2);
        }
    }
    return out;
}

// Mirror __Runtime_dma_bd_config for a core-tile S2MM ping/pong BD: emit the
// XAie driver calls that program one buffer descriptor. Runs inside an open
// transaction, so nothing hits HW.
static void emit_golden_core_bd(XAie_DevInst *dev, XAie_LocType loc, uint8_t bd_id, uint64_t dma_addr, uint32_t len,
                                int next_bd, int acq_id, int acq_val, int rel_id, int rel_val, int en_packet,
                                int pkt_id, int pkt_type, int ooo_bd_id) {
    XAie_DmaDesc desc;
    XAie_DmaDescInit(dev, &desc, loc);
    XAie_DmaSetAddrLen(&desc, dma_addr, len);
    if (acq_id >= 0 && rel_id >= 0)
        XAie_DmaSetLock(&desc, XAie_LockInit((u16)acq_id, (s8)acq_val), XAie_LockInit((u16)rel_id, (s8)rel_val));
    if (next_bd >= 0)
        XAie_DmaSetNextBd(&desc, (uint16_t)next_bd, XAIE_ENABLE);
    if (en_packet)
        XAie_DmaSetPkt(&desc, XAie_PacketInit((u8)pkt_id, (u8)pkt_type));
    if (ooo_bd_id >= 0)
        XAie_DmaSetOutofOrderBdId(&desc, (uint8_t)ooo_bd_id);
    XAie_DmaEnableBd(&desc);
    AieRC rc = XAie_DmaWriteBd(dev, &desc, loc, bd_id);
    if (rc != XAIE_OK) {
        fprintf(stderr, "FAIL: XAie_DmaWriteBd rc=%d\n", (int)rc);
        exit(3);
    }
}

// Look up the golden captured value at a tile-local offset. Sets *found.
static uint32_t golden_val(const std::vector<CapWrite> &g, uint32_t off, bool *found) {
    for (const CapWrite &c : g) {
        if (c.off == off) {
            *found = true;
            return c.val;
        }
    }
    *found = false;
    return 0;
}

// Diff every encoder-produced (off,val) against the golden capture. Prints each
// comparison; returns the number of mismatches (0 == byte-identical).
static int diff_regs(const char *label, const AieKcReg *enc, int n, const std::vector<CapWrite> &g) {
    int bad = 0;
    printf("--- %s: encoder vs golden (%d regs) ---\n", label, n);
    for (int i = 0; i < n; i++) {
        bool found = false;
        uint32_t gv = golden_val(g, enc[i].off, &found);
        const char *tag = (found && gv == enc[i].val) ? "ok" : (found ? "MISMATCH" : "MISSING");
        printf("  off=0x%05x enc=0x%08x golden=0x%08x  %s\n", enc[i].off, enc[i].val, gv, tag);
        if (!found || gv != enc[i].val)
            bad++;
    }
    return bad;
}

int main(void) {
    XAie_SetupConfig(cfg, XAIE_DEV_GEN_AIE2PS, KC_BASE_ADDR, KC_COL_SHIFT, KC_ROW_SHIFT, KC_NUM_COLS, KC_NUM_ROWS,
                     KC_SHIM_ROW, KC_MEMTILE_ROW_ST, KC_MEMTILE_NUM_ROWS, KC_CORE_ROW_ST, KC_CORE_NUM_ROWS);

    XAie_InstDeclare(dev_inst, &cfg);
    AieRC rc = XAie_CfgInitialize(&dev_inst, &cfg);
    if (rc != XAIE_OK) {
        fprintf(stderr, "FAIL: XAie_CfgInitialize rc=%d\n", (int)rc);
        return 1;
    }

    // Capture: buffer register writes instead of executing them.
    rc = XAie_StartTransaction(&dev_inst, XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
    if (rc != XAIE_OK) {
        fprintf(stderr, "FAIL: XAie_StartTransaction rc=%d\n", (int)rc);
        return 1;
    }

    // Core tile at (col=0, row=3). BD params reused by both the golden driver
    // emission and the standalone encoder so the two can be diffed.
    XAie_LocType loc = XAie_TileLoc(0, KC_CORE_ROW_ST);
    const uint64_t ping_addr = 0x2000; // core DMA-view byte address
    const uint32_t len = 1024;         // bytes
    const uint8_t bd_id = 0, next_bd = 1;
    const int acq_id = 0, acq_val = -1, rel_id = 1, rel_val = 1;
    const uint8_t lock_id = 1;
    const int lock_val = 1;
    const uint8_t ch = 0;
    const uint32_t repeat = 1;

    // Second BD: an MM2S-style BD exercising packet + out-of-order (so BD word 1
    // is non-zero and those encoder branches are actually validated).
    const uint8_t bd2_id = 2;
    const uint64_t bd2_addr = 0x4000;
    const uint32_t bd2_len = 512;
    const int bd2_acq_id = 2, bd2_acq_val = -1, bd2_rel_id = 3, bd2_rel_val = 1;
    const int bd2_pkt_id = 5, bd2_pkt_type = 0, bd2_ooo = 7;

    // Golden #1: S2MM ping BD 0 -> pong BD 1.
    emit_golden_core_bd(&dev_inst, loc, bd_id, ping_addr, len, next_bd, acq_id, acq_val, rel_id, rel_val,
                        /*en_packet=*/0, /*pkt_id=*/0, /*pkt_type=*/0, /*ooo_bd_id=*/-1);
    // Golden #1b: MM2S BD with packet + out-of-order id, no next bd.
    emit_golden_core_bd(&dev_inst, loc, bd2_id, bd2_addr, bd2_len, /*next_bd=*/-1, bd2_acq_id, bd2_acq_val, bd2_rel_id,
                        bd2_rel_val, /*en_packet=*/1, bd2_pkt_id, bd2_pkt_type, bd2_ooo);
    // Golden #2: lock init value.
    rc = XAie_LockSetValue(&dev_inst, loc, XAie_LockInit((u8)lock_id, (s8)lock_val));
    if (rc != XAIE_OK) {
        fprintf(stderr, "FAIL: XAie_LockSetValue rc=%d\n", (int)rc);
        return 1;
    }
    // Golden #3: S2MM channel start queue.
    rc = XAie_DmaChannelSetStartQueue(&dev_inst, loc, ch, DMA_S2MM, bd_id, repeat, XAIE_DISABLE);
    if (rc != XAIE_OK) {
        fprintf(stderr, "FAIL: XAie_DmaChannelSetStartQueue rc=%d\n", (int)rc);
        return 1;
    }
    // Golden #3b: MM2S channel start queue — the core arming its own OUTGOING DMA
    // under KERNELCONFIGOFFLOAD. Deliberately a DIFFERENT channel and bd than the
    // S2MM case above: S2MM ch0 and MM2S ch0 would land on offsets 0x1DE04/0x1DE14,
    // and reusing ch/bd would let a wrong base offset or a wrong ch stride still
    // pass by aliasing onto the S2MM expectation.
    const uint8_t mm2s_ch = 1;
    const uint8_t mm2s_bd = 3;
    const uint32_t mm2s_repeat = 2;
    rc = XAie_DmaChannelSetStartQueue(&dev_inst, loc, mm2s_ch, DMA_MM2S, mm2s_bd, mm2s_repeat, XAIE_DISABLE);
    if (rc != XAIE_OK) {
        fprintf(stderr, "FAIL: XAie_DmaChannelSetStartQueue (MM2S) rc=%d\n", (int)rc);
        return 1;
    }

    uint8_t *txn = XAie_ExportSerializedTransaction(&dev_inst, 1, 0);
    if (!txn) {
        fprintf(stderr, "FAIL: XAie_ExportSerializedTransaction returned NULL\n");
        return 1;
    }

    std::vector<CapWrite> writes = parse_txn(txn);
    free(txn);

    printf("captured %zu golden register writes:\n", writes.size());
    for (const CapWrite &c : writes)
        printf("  off=0x%05x = 0x%08x\n", c.off, c.val);

    // Encode the same config via the standalone encoder and diff byte-for-byte.
    int bad = 0;
    AieKcReg bd[AIE_KC_DMA_BD_NUM_WORDS];
    int nbd = aie_kc_encode_bd(bd, bd_id, ping_addr, len, next_bd, acq_id, acq_val, rel_id, rel_val,
                               /*en_packet=*/0, /*pkt_id=*/0, /*pkt_type=*/0, /*ooo_bd_id=*/-1);
    bad += diff_regs("S2MM BD0", bd, nbd, writes);

    AieKcReg bd2[AIE_KC_DMA_BD_NUM_WORDS];
    int nbd2 = aie_kc_encode_bd(bd2, bd2_id, bd2_addr, bd2_len, /*next_bd=*/-1, bd2_acq_id, bd2_acq_val, bd2_rel_id,
                                bd2_rel_val, /*en_packet=*/1, bd2_pkt_id, bd2_pkt_type, bd2_ooo);
    bad += diff_regs("MM2S BD2 (pkt+ooo)", bd2, nbd2, writes);

    AieKcReg lk[1];
    int nlk = aie_kc_encode_lock(lk, lock_id, lock_val);
    bad += diff_regs("lock init", lk, nlk, writes);

    AieKcReg sq[1];
    int nsq = aie_kc_encode_s2mm_start(sq, ch, bd_id, repeat, /*en_token=*/0);
    bad += diff_regs("S2MM start queue", sq, nsq, writes);

    AieKcReg mq[1];
    int nmq = aie_kc_encode_mm2s_start(mq, mm2s_ch, mm2s_bd, mm2s_repeat, /*en_token=*/0);
    bad += diff_regs("MM2S start queue", mq, nmq, writes);

    if (bad != 0) {
        fprintf(stderr, "FAIL: %d encoder/golden mismatches\n", bad);
        return 1;
    }
    printf("PASS: standalone encoder is byte-identical to aie-rt golden "
           "(BD + lock + S2MM/MM2S channel-start)\n");
    return 0;
}
