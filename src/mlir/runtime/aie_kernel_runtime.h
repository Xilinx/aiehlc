/* aie_kernel_runtime.h — AIE2PS (gen5) core-tile DMA config encoder + apply.
 *
 * KERNELCONFIGOFFLOAD: let the AIE CORE self-configure its own DMA buffer
 * descriptors / locks / channel-start from inside kernel.cc via MMIO, instead
 * of the HOST programming them over the config bus.
 *
 * This header produces the SAME register (tile-local offset, value) words the
 * aie-rt gen5 driver writes via:
 *   - XAie_DmaWriteBd            -> _XAieMl_TileDmaWriteBdPrep (6 BD words)
 *   - XAie_LockSetValue          -> _XAieMl_LockSetValue        (1 lock word)
 *   - XAie_DmaChannelSetStartQueue-> start-queue                (1 channel word)
 *
 * TWO HALVES, different dependencies:
 *   - the aie_kc_encode_* ENCODERS are pure computation (only <stdint.h>), so
 *     they are includable by kernel.cc (xchesscc, AIE core) AND by the host
 *     golden-diff unit test (test_kernel_config.cpp), which validates every
 *     word against real driver output captured from an XAie transaction;
 *   - the core_reg_* APPLY helpers need the TM memory space and so only do
 *     anything on an AIE2+ core build. See kernel_tm.h, included below: on a
 *     host parse its macros compile out, which is exactly what lets this one
 *     header serve both builds.
 *
 * Field Lsb/Mask constants below are copied verbatim from
 * thirdparty/alib/aie-rt/driver/src/global/xaie2psgbl_params.h. Scope: 1D
 * contiguous core-tile S2MM/MM2S BDs (the shapes the offload emits); multi-dim
 * / compression / FIFO are not encoded (their descriptor defaults encode as 0).
 */
#ifndef AIE_KERNEL_RUNTIME_H
#define AIE_KERNEL_RUNTIME_H

#include <stdint.h>

/* TM_W / TM_R / TM_BASE_ADDR and the shared register offsets. Must be included
 * at FILE scope: the TM anchor object it declares is what gives the compiler
 * the memory space (see kernel_tm.h). */
#include "kernel_tm.h"

/* ---- Core-side address windows ----
 *
 * The register offsets below are TILE-LOCAL, i.e. what the XAie driver uses when
 * it pokes a tile over the config bus from the host. Code running ON the core
 * does not see that address space directly: the core's own memory map places the
 * tile's control/config registers in the TM space at TM_BASE_ADDR.
 *
 * Writing the bare tile-local offset from the core lands in the core's DATA
 * memory window instead of the register window — it silently scribbles on data
 * memory and the DMA is never programmed. That is why an S2MM channel could
 * appear "not started" even though the code ran: the start-queue write went to
 * the wrong window entirely.
 *
 * Adding the bias is necessary but NOT sufficient — see core_reg_write below.
 */
#define AIE_KC_CORE_PC_CONTROL_BASE_ADDR TM_BASE_ADDR /* core view of tile ctrl/config regs */
#define AIE_KC_CORE_DM_BASE_ADDR 0x70000u             /* core data memory: 0x40000-0x7FFFF (256K) */

/* ---- AIE2PS memory-module (tile-local) register offsets ----
 * Shared offsets alias kernel_tm.h so the two headers cannot drift. */
#define AIE_KC_DMA_BD0_0 TM_DMA_BD0_0          /* MEMORY_MODULE_DMA_BD0_0 */
#define AIE_KC_DMA_BD_STRIDE 0x20u             /* per-BD IdxOffset */
#define AIE_KC_DMA_BD_NUM_WORDS 6u             /* XAIEML_TILEDMA_NUM_BD_WORDS */
#define AIE_KC_S2MM0_START_Q TM_DMA_S2MM_0_START_QUEUE /* MEMORY_MODULE_DMA_S2MM_0_START_QUEUE */
#define AIE_KC_MM2S0_START_Q 0x1DE14u          /* MEMORY_MODULE_DMA_MM2S_0_START_QUEUE */
#define AIE_KC_DMA_CH_STRIDE 0x8u              /* per-channel ChIdxOffset */
#define AIE_KC_LOCK0_VALUE TM_LOCK0_VALUE      /* MEMORY_MODULE_LOCK0_VALUE */
#define AIE_KC_LOCK_STRIDE 0x10u               /* per-lock */

/* One register write: tile-local byte offset + value.
 * `off` is TILE-LOCAL — pass it to core_reg_write/core_reg_read to reach the
 * register from code running on the core. */
typedef struct {
    uint32_t off;
    uint32_t val;
} AieKcReg;

/* ---- Core-side register access ----
 *
 * Read/write a TILE-LOCAL register offset from code running on the core. ALL
 * core-side register traffic must go through these.
 *
 * These delegate to TM_W/TM_R rather than casting a biased address, because
 * THE ADDRESS DOES NOT SELECT THE INTERFACE — the compiler picks it from the
 * pointer's memory space, which a numeric cast cannot express. A raw
 * `*(volatile uint32_t *)(TM_BASE_ADDR + off) = v` assembles to a plain ST that
 * never leaves the core, yet reads back correctly on-core (store and load are
 * coherent with each other) while the host still sees 0. kernel_tm.h documents
 * the full rule; TM_W/TM_R carry the chess_storage(TM:...) anchor that makes
 * these lower to ST.TM / LDA.TM.
 *
 * On a host-side parse TM_W/TM_R compile to no-ops, so this header still builds
 * for the golden-diff unit test — which exercises the encoders, not these.
 *
 * Verify the lowering, not just the value:
 *     grep -cE '\.TM' <build>/obj/kernel.lst
 * Zero means every access degraded to plain ST/LDA and the DMA was never
 * programmed.
 */
/* The (void) casts keep -Wunused-parameter quiet on a host parse, where
 * TM_R/TM_W expand to a constant / no-op and never touch their arguments. */
static inline uint32_t core_reg_read(uint32_t reg_addr) {
    (void)reg_addr;
    return (uint32_t)TM_R(reg_addr);
}

static inline void core_reg_write(uint32_t reg_addr, uint32_t value) {
    (void)reg_addr;
    (void)value;
    TM_W(reg_addr, value);
}

/* Flush an encoded register block (the AieKcReg[] the encoders below fill). */
static inline void core_reg_write_block(const AieKcReg *regs, int n) {
    for (int i = 0; i < n; i++)
        core_reg_write(regs[i].off, regs[i].val);
}

/* Emulate the driver's XAie_SetField(v, Lsb, Mask) = (v << Lsb) & Mask.
 * Signed field values (e.g. a lock acquire value of -1) are passed already
 * cast to uint32_t; the two's-complement bit pattern masks to the field. */
static inline uint32_t aie_kc_field(uint32_t v, uint32_t lsb, uint32_t mask) { return (v << lsb) & mask; }

/* Encode one core-tile DMA buffer descriptor into 6 register writes at
 * BD0_0 + bd_id*0x20. Mirrors _XAieMl_TileDmaWriteBdPrep exactly for the 1D
 * contiguous case.
 *
 *   out          : caller array of >= AIE_KC_DMA_BD_NUM_WORDS entries
 *   bd_id        : hardware BD index (0..15)
 *   dma_addr     : buffer start byte address (core DMA view)
 *   len          : buffer length in bytes
 *   next_bd      : chained next BD id, or <0 for none
 *   acq_id/rel_id: lock ids (both >=0 to program a lock, else no lock)
 *   acq_val/rel_val: signed acquire/release lock values
 *   en_packet    : nonzero to enable packet mode
 *   pkt_id/pkt_type: packet header fields (used only when en_packet)
 *   ooo_bd_id    : out-of-order BD id, or <0 for none
 *
 * Returns the number of register writes emitted (AIE_KC_DMA_BD_NUM_WORDS). */
static inline int aie_kc_encode_bd(AieKcReg *out, uint8_t bd_id, uint64_t dma_addr, uint32_t len, int next_bd,
                                   int acq_id, int acq_val, int rel_id, int rel_val, int en_packet, int pkt_id,
                                   int pkt_type, int ooo_bd_id) {
    uint32_t addr = (uint32_t)(dma_addr >> 2); /* AddrAlignShift = 2 */
    uint32_t nwords = len >> 2;                /* 32-bit transfer length */
    uint32_t base = AIE_KC_DMA_BD0_0 + (uint32_t)bd_id * AIE_KC_DMA_BD_STRIDE;

    /* word 0: base address | buffer length */
    uint32_t w0 = aie_kc_field(addr, 14u, 0x0FFFC000u) | aie_kc_field(nwords, 0u, 0x00003FFFu);

    /* word 1: compression | packet | out-of-order bd id (compression = 0) */
    uint32_t w1 = 0u;
    if (en_packet) {
        w1 |= aie_kc_field(1u, 30u, 0x40000000u);                 /* EnPkt */
        w1 |= aie_kc_field((uint32_t)pkt_type, 16u, 0x00070000u); /* PktType */
        w1 |= aie_kc_field((uint32_t)pkt_id, 19u, 0x00F80000u);   /* PktId */
    }
    if (ooo_bd_id >= 0)
        w1 |= aie_kc_field((uint32_t)ooo_bd_id, 24u, 0x3F000000u);

    /* words 2..4: multi-dim / iteration — all zero for a 1D contiguous BD
     * (StepSize defaults 1 -> encoded (StepSize-1)=0, Wrap defaults 0). */
    uint32_t w2 = 0u, w3 = 0u, w4 = 0u;

    /* word 5: valid | lock | next-bd */
    uint32_t w5 = aie_kc_field(1u, 25u, 0x02000000u); /* ValidBd (EnableBd) */
    if (acq_id >= 0 && rel_id >= 0) {
        w5 |= aie_kc_field((uint32_t)rel_val, 18u, 0x01FC0000u); /* LckRelVal */
        w5 |= aie_kc_field((uint32_t)rel_id, 13u, 0x0001E000u);  /* LckRelId */
        w5 |= aie_kc_field(1u, 12u, 0x00001000u);                /* LckAcqEn */
        w5 |= aie_kc_field((uint32_t)acq_val, 5u, 0x00000FE0u);  /* LckAcqVal */
        w5 |= aie_kc_field((uint32_t)acq_id, 0u, 0x0000000Fu);   /* LckAcqId */
    }
    if (next_bd >= 0) {
        w5 |= aie_kc_field(1u, 26u, 0x04000000u);                /* UseNxtBd */
        w5 |= aie_kc_field((uint32_t)next_bd, 27u, 0x78000000u); /* NxtBd */
    }

    out[0].off = base + 0x00u;
    out[0].val = w0;
    out[1].off = base + 0x04u;
    out[1].val = w1;
    out[2].off = base + 0x08u;
    out[2].val = w2;
    out[3].off = base + 0x0Cu;
    out[3].val = w3;
    out[4].off = base + 0x10u;
    out[4].val = w4;
    out[5].off = base + 0x14u;
    out[5].val = w5;
    return (int)AIE_KC_DMA_BD_NUM_WORDS;
}

/* Encode a lock initial-value write (LOCK0_VALUE + id*0x10). Mirrors
 * _XAieMl_LockSetValue: 6-bit signed lock value field. Returns 1. */
static inline int aie_kc_encode_lock(AieKcReg *out, uint8_t lock_id, int value) {
    out[0].off = AIE_KC_LOCK0_VALUE + (uint32_t)lock_id * AIE_KC_LOCK_STRIDE;
    out[0].val = aie_kc_field((uint32_t)value, 0u, 0x0000003Fu);
    return 1;
}

/* Start-queue value: start bd id [3:0] | (repeat-1) [23:16] | en-token [31].
 * S2MM_0_START_QUEUE and MM2S_0_START_QUEUE have identical layouts (both
 * MASK 0x80FF000F), so both directions share this; only the base offset
 * differs. Mirrors _XAie_DmaChannelSetStartQueuePrepare. */
static inline uint32_t aie_kc_start_q_val(uint8_t start_bd, uint32_t repeat, int en_token) {
    return aie_kc_field((uint32_t)start_bd, 0u, 0x0000000Fu) | aie_kc_field(repeat - 1u, 16u, 0x00FF0000u) |
           aie_kc_field(en_token ? 1u : 0u, 31u, 0x80000000u);
}

/* Encode an S2MM channel start-queue write (S2MM_0_START_QUEUE + ch*0x8).
 * Returns 1. */
static inline int aie_kc_encode_s2mm_start(AieKcReg *out, uint8_t ch, uint8_t start_bd, uint32_t repeat, int en_token) {
    out[0].off = AIE_KC_S2MM0_START_Q + (uint32_t)ch * AIE_KC_DMA_CH_STRIDE;
    out[0].val = aie_kc_start_q_val(start_bd, repeat, en_token);
    return 1;
}

/* Encode an MM2S channel start-queue write (MM2S_0_START_QUEUE + ch*0x8).
 * Same value layout as S2MM; this is what lets the core arm its own OUTGOING
 * DMA under KERNELCONFIGOFFLOAD. Returns 1. */
static inline int aie_kc_encode_mm2s_start(AieKcReg *out, uint8_t ch, uint8_t start_bd, uint32_t repeat, int en_token) {
    out[0].off = AIE_KC_MM2S0_START_Q + (uint32_t)ch * AIE_KC_DMA_CH_STRIDE;
    out[0].val = aie_kc_start_q_val(start_bd, repeat, en_token);
    return 1;
}

#endif /* AIE_KERNEL_RUNTIME_H */
