/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef KERNEL_LOG_H
#define KERNEL_LOG_H

#include <stdint.h>

/*
 * Kernel Data Memory Log for AIE Debug
 *
 * Writes log entries (4-char tag + int32 value) to the last 2KB of AIE core
 * tile data memory. No stdlib, no printf — volatile memory writes only.
 *
 * Memory layout (512 int32 slots at offset 0xF800 of data memory):
 *   slot[0]   = write_index (number of int32s written, NOT entries)
 *   slot[1,2] = first entry: [tag_packed, value]
 *   slot[3,4] = second entry: [tag_packed, value]
 *   ...up to 255 entries (511 slots / 2 per entry)
 *
 * AIE core tile data memory: 64KB, base address 0x70000.
 * Log region: offset 0xF800 => absolute address 0x7F800.
 * Host reads via XAie_DataMemBlockRead(dev, tile, 0xF800, buf, 2048).
 */

/* Data memory offset for host-side reads */
#define KLOG_DM_OFFSET 0xF800

/* Region size in bytes (512 int32 slots) */
#define KLOG_REGION_BYTES 2048

/* Maximum number of int32 slots (slot 0 = write_index, slots 1..511 = data) */
#define KLOG_MAX_SLOTS 511

/* Magic marker written by klog_init: "KLOG" in ASCII big-endian */
#define KLOG_MAGIC 0x4B4C4F47

/* Pack 4 ASCII characters into one int32 (big-endian byte order) */
#define KLOG_TAG(a, b, c, d)                                                                                           \
    ((int32_t)(((uint8_t)(a) << 24) | ((uint8_t)(b) << 16) | ((uint8_t)(c) << 8) | ((uint8_t)(d))))

/*
 * Pack the first 4 characters of a string into an int32.
 * Shorter strings are padded with null bytes.
 */
static inline int32_t klog_pack_tag(const char *tag) {
    uint8_t c0 = tag[0];
    uint8_t c1 = c0 ? (uint8_t)tag[1] : 0;
    uint8_t c2 = c1 ? (uint8_t)tag[2] : 0;
    uint8_t c3 = c2 ? (uint8_t)tag[3] : 0;
    return (int32_t)((c0 << 24) | (c1 << 16) | (c2 << 8) | c3);
}

#ifdef __AIENGINE__
/* ---- Kernel side: volatile writes to data memory ---- */

/* Absolute address for kernel-side access (base 0x70000 + offset 0xF800) */
#define KLOG_BASE 0x7F800

/* Pointer to the log region in data memory */
#define KLOG_PTR ((volatile int32_t *)KLOG_BASE)

/*
 * Kernel logging is OPT-IN, enabled by KERNEL_LOG_ENABLED.
 *
 * script/kc.sh defines it when the user writes
 *     #pragma aie_debug_level(AIE_KERNEL_CONFIG_TRACE)
 * (see AIE_DEBUG_FLAG_KERNEL_CONFIG_TRACE in aie_runtime.h).
 *
 * Gating here rather than at each call site means ONE switch covers every klog
 * user — the compute kernel's own logging and the KERNELCONFIGOFFLOAD register
 * trace alike — and a normal build pays nothing: klog compiles to an empty
 * inline that the optimizer removes, so callers need no #ifdef of their own.
 */
#ifdef KERNEL_LOG_ENABLED

/*
 * Write a log entry: 4-char tag + int32 value.
 * Each entry occupies 2 int32 slots.
 */
static inline void klog(const char *tag, int32_t val) {
    volatile int32_t *base = KLOG_PTR;
    int32_t wi = base[0]; /* current write_index */
    if (wi + 2 > KLOG_MAX_SLOTS)
        return; /* log full */
    base[wi + 1] = klog_pack_tag(tag);
    base[wi + 2] = val;
    base[0] = wi + 2;
}

/*
 * Initialize the kernel log: clear write_index, write magic marker.
 * Call once at the start of kernel main().
 */
static inline void klog_init(void) {
    volatile int32_t *base = KLOG_PTR;
    base[0] = 0; /* write_index = 0 */
    klog("KLOG", KLOG_MAGIC);
}

#else /* !KERNEL_LOG_ENABLED — compile klog out entirely */

static inline void klog(const char *tag, int32_t val) {
    (void)tag;
    (void)val;
}
static inline void klog_init(void) {}

#endif /* KERNEL_LOG_ENABLED */

#else
/* ---- Host side: no-op stubs + readback helper ---- */

/* aiehlc.cc injects its own klog stub inside AIEHLC_STUBS_DEFINED;
   only provide ours when NOT compiled through aiehlc. */
#ifndef AIEHLC_STUBS_DEFINED
static inline void klog(const char *tag, int32_t val) {
    (void)tag;
    (void)val;
}
static inline void klog_init(void) {}
#endif

/*
 * Read and print kernel log from a tile via XAie_DataMemBlockRead.
 * Requires xaiengine.h to be included before this header.
 */
static inline void klog_read(XAie_DevInst *dev, XAie_LocType tile) {
    int32_t buf[512];
    for (int i = 0; i < 512; i++)
        buf[i] = 0;

    AieRC rc = XAie_DataMemBlockRead(dev, tile, KLOG_DM_OFFSET, (void *)buf, KLOG_REGION_BYTES);
    if (rc != XAIE_OK) {
        printf("[klog] tile(%u,%u): read failed rc=%d\n", (unsigned)tile.Col, (unsigned)tile.Row, (int)rc);
        return;
    }

    int32_t wi = buf[0];
    if (wi <= 0 || wi > KLOG_MAX_SLOTS) {
        printf("[klog] tile(%u,%u): no log (write_index=%d)\n", (unsigned)tile.Col, (unsigned)tile.Row, wi);
        return;
    }

    printf("[klog] tile(%u,%u): %d entries\n", (unsigned)tile.Col, (unsigned)tile.Row, wi / 2);
    for (int i = 0; i < wi; i += 2) {
        int32_t tag_raw = buf[i + 1];
        int32_t val = buf[i + 2];
        char tag[5];
        tag[0] = (char)((tag_raw >> 24) & 0xFF);
        tag[1] = (char)((tag_raw >> 16) & 0xFF);
        tag[2] = (char)((tag_raw >> 8) & 0xFF);
        tag[3] = (char)(tag_raw & 0xFF);
        tag[4] = '\0';
        printf("  [%d] %s = %d (0x%08x)\n", i / 2, tag, val, (unsigned)val);
    }
}

#endif /* __AIENGINE__ */

#endif /* KERNEL_LOG_H */
