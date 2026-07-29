/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "aie_runtime.h"
#include "aie_device_map.h"
#include "aie_runtime_debug.h"
#include "aie_runtime_stream_debug.h"
#ifdef __AIESIM__
#  include <unistd.h>
#  define Xil_DCacheFlushRange(addr, len)       ((void)0)
#  define Xil_DCacheInvalidateRange(addr, len)  ((void)0)
#  define Xil_SetTlbAttributes(addr, attr)      ((void)0)
#ifdef __cplusplus
extern "C" {
#endif
extern void ess_WriteGM(uint64_t addr, const void *data, uint64_t size);
extern void ess_ReadGM(uint64_t addr, void *data, uint64_t size);
#ifdef __cplusplus
}
#endif
#else
#  include "sleep.h"
#  include "xil_cache.h"
#endif
#include "xaiengine/xaie_helper.h"
#include <stdio.h>

// HW generation for device config (reference: aieml_perf.cc lines 14-20)
#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#endif

// for cache disable case we need to do that at app construction beginning
static void __Runtime_init(void) __attribute__((constructor));
extern void routing(XAie_DevInst *dev);
// Device layout declare: config and instance (reference: aieml_perf.cc lines 292-300)
static XAie_SetupConfig(g_Config, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                        XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                        XAIE_AIE_TILE_NUM_ROWS);

XAie_DevInst *g_DevInst = NULL;

XAie_DevInst *getOrCreateDeviceInstance(void) { return g_DevInst; }

// Global routing instance (kept for legacy path)
XAie_RoutingInstance *g_RoutingInst = NULL;

// Debug level: bits 0-3 = verbosity (0-15), bits 4-31 = feature flags
//   Verbosity 0: silent (default)
//   Verbosity 1: BD tracking and IO logs (no core DMA address log)
//   Verbosity 2: core DMA address log + write pattern + readback logic
//   Flag bit 4 (value 16): AIE_DEBUG_FLAG_DISABLE_MULTID_DIM_DMA
//   Flag bit 5 (value 32): AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN
//   Flag bit 6 (value 64): AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER
//     -> after routing init, arm our own MM2S BD-finished perf counters
//        (MEM module counters 0/1) across the whole partition via
//        __Runtime_perfcnt_setup_mm2s_bd_finished_partition(); teardown reads
//        them back. Same counters aiegdb.py "dma counter" reads (0x11020/24).
// Weak symbol: user source can override via #pragma aie_debug_level N
__attribute__((weak)) int g_runtime_debug_level = 0;

// Reference: aieml_perf.cc lines 111-281 for implementation patterns

/** Return 1 if tile is an AIE core tile (row >= XAIE_AIE_TILE_ROW_START), 0 for shim/res. */
static inline int __Runtime_is_aie_core_tile(XAie_LocType tile) { return tile.Row >= XAIE_AIE_TILE_ROW_START; }

#define MAX_ALLOC_BUFFERS 64
typedef struct {
    void *vaddr;
    XAie_MemInst *mem;
    size_t size;
} AllocEntry;
static AllocEntry s_alloc_map[MAX_ALLOC_BUFFERS];
static int s_alloc_count = 0;

static XAie_MemInst *__vaddr_to_mem_offset(void *vaddr, uint64_t *offset_out) {
    uintptr_t addr = (uintptr_t)vaddr;
    for (int i = 0; i < s_alloc_count; i++) {
        uintptr_t base = (uintptr_t)s_alloc_map[i].vaddr;
        if (addr >= base && addr < base + s_alloc_map[i].size) {
            uint64_t off = addr - base;
            uint64_t dev = XAie_MemGetDevAddr(s_alloc_map[i].mem) + off;
            AIEHLC_LOG(printf("[aie_runtime] vaddr_lookup: %p → alloc[%d] base=%p size=%zu off=%lu DevAddr=0x%lx\n",
                              vaddr, i, s_alloc_map[i].vaddr, s_alloc_map[i].size, (unsigned long)off,
                              (unsigned long)dev));
            if (offset_out)
                *offset_out = off;
            return s_alloc_map[i].mem;
        }
    }
    AIEHLC_LOG(printf("[aie_runtime] vaddr_lookup: %p → NO MATCH (checked %d allocs)\n", vaddr, s_alloc_count););
    return NULL;
}
static XAie_MemInst *__vaddr_to_mem(void *vaddr) { return __vaddr_to_mem_offset(vaddr, NULL); }

/** Static buffer for kernel group tiles so kg.tiles/event.tiles outlive __Runtime_load_kernel_group_*. */
#define MAX_KERNEL_TILES 32
static XAie_LocType s_kernel_tiles[MAX_KERNEL_TILES];

/* ---------------------------------------------------------------------------
 * BD tracking: records every dma_bd_config call when debug is enabled.
 * __Runtime_free prints the table and dumps the first 16 int32 values for
 * shim tile entries (those have a valid DDR buffer pointer).
 * ----------------------------------------------------------------------- */
typedef struct {
    uint8_t col;
    uint8_t row;
    uint8_t bd_id;
    uint8_t tile_type; /* XAIEGBL_TILE_TYPE_* value from XAie_GetTileTypefromLoc */
    void *buffer;      /* DDR pointer for shim tiles, NULL for core tiles */
    uint64_t dma_addr; /* physical addr programmed into the BD */
    int32_t len;       /* length in bytes */
    int32_t packet_id;
    int8_t direction;  /* -1=unknown, 0=DMA_S2MM, 1=DMA_MM2S (set by startio) */
    int8_t channel_id; /* -1=unknown (set by startio) */
    int8_t next_bd;    /* next BD in chain (-1 = none) */
} BdTrackEntry;

static const char *__bd_tile_type_str(uint8_t tt) {
    switch (tt) {
    case XAIEGBL_TILE_TYPE_SHIMNOC:
        return "SHIM_NOC";
    case XAIEGBL_TILE_TYPE_SHIMPL:
        return "SHIM_PL";
    case XAIEGBL_TILE_TYPE_AIETILE:
        return "CORE";
    case XAIEGBL_TILE_TYPE_MEMTILE:
        return "MEM";
    default:
        return "UNKNOWN";
    }
}

static int __bd_is_shim(uint8_t tt) { return (tt == XAIEGBL_TILE_TYPE_SHIMNOC || tt == XAIEGBL_TILE_TYPE_SHIMPL); }

#define BD_TRACK_MAX 64
static BdTrackEntry g_bd_track[BD_TRACK_MAX];
static int g_bd_track_count = 0;
static int g_dumped_done = 0;

/* ---------------------------------------------------------------------------
 * Kernel log reader: reads the last 2KB of core tile data memory where
 * kernel_log.h writes [tag, value] pairs via volatile stores.
 * Log region: data memory offset 0xF800, 512 int32 slots.
 *   slot[0]   = write_index (number of int32s written)
 *   slot[1,2] = first entry [tag_packed, value]
 *   slot[3,4] = second entry ...
 * Tag is 4 ASCII chars packed big-endian into int32.
 * ----------------------------------------------------------------------- */
#define KLOG_DM_OFFSET 0xF800
#define KLOG_REGION_BYTES 2048
#define KLOG_MAGIC_VAL 0x4B4C4F47 /* "KLOG" */

void __Runtime_read_kernel_log(XAie_DevInst *dev, XAie_LocType tile) {
    int32_t buf[512];
    memset(buf, 0, sizeof(buf));

    AieRC rc = XAie_DataMemBlockRead(dev, tile, KLOG_DM_OFFSET, (void *)buf, KLOG_REGION_BYTES);
    if (rc != XAIE_OK) {
        printf("[kernel_log] tile(%u,%u): read failed rc=%d\n", (unsigned)tile.Col, (unsigned)tile.Row, (int)rc);
        return;
    }

    int32_t wi = buf[0];
    if (wi <= 0 || wi > 511) {
        AIEHLC_LOG(
            printf("[kernel_log] tile(%u,%u): no log (write_index=%d)\n", (unsigned)tile.Col, (unsigned)tile.Row, wi););
        return;
    }

    int num_entries = wi / 2;
    AIEHLC_LOG(printf("[kernel_log] tile(%u,%u): %d entries\n", (unsigned)tile.Col, (unsigned)tile.Row, num_entries););

    for (int i = 0; i < wi; i += 2) {
        int32_t tag_raw = buf[i + 1];
        int32_t val = buf[i + 2];
        char tag[5];
        tag[0] = (char)((tag_raw >> 24) & 0xFF);
        tag[1] = (char)((tag_raw >> 16) & 0xFF);
        tag[2] = (char)((tag_raw >> 8) & 0xFF);
        tag[3] = (char)(tag_raw & 0xFF);
        tag[4] = '\0';
        AIEHLC_LOG(printf("  [%d] %s = %d (0x%08x)\n", i / 2, tag, val, (unsigned)val););
    }
}

/* ---------------------------------------------------------------------------
 * Plain heap wrappers for DDR buffers in the host init chain
 * ScheduleCanonicalizePass emits __Runtime_malloc / __Runtime_memcpy /
 * __Runtime_free for the arith.constant → memref.alloc → memref.copy chain
 * that initialises DDR data before DMA operations begin.
 * ----------------------------------------------------------------------- */
#include <stdlib.h>

void *__Runtime_malloc(size_t bytes) {
    void *ptr = malloc(bytes);
    AIEHLC_LOG(printf("[aie_runtime] malloc(%zu) = %p\n", bytes, ptr););
    return ptr;
}

void *__Runtime_Alloc(size_t bytes) {
    /* aligned_alloc requires the size be a multiple of the alignment. */
    size_t aligned_bytes = (bytes + 63) & ~(size_t)63;
    void *ptr = aligned_alloc(64, aligned_bytes);
    AIEHLC_LOG(printf("[aie_runtime] Alloc(%zu -> %zu, align=64) = %p\n", bytes, aligned_bytes, ptr););
    return ptr;
}

void __Runtime_free(void *ptr) {
    if (AIEHLC_LOG_ENABLED()) {
        printf("[aie_runtime] free(%p)\n", ptr);
        /* Dump key regions of buffer: offset 0-63 (round0 out), 256-319 (round1 out) */
        {
            int8_t *p8 = (int8_t *)ptr;
            printf("[aie_runtime] free buf [0..63]: ");
            for (int j = 0; j < 64; j++)
                printf("%d ", (int)p8[j]);
            printf("\n");
            printf("[aie_runtime] free buf [64..127]: ");
            for (int j = 64; j < 128; j++)
                printf("%d ", (int)p8[j]);
            printf("\n");
            printf("[aie_runtime] free buf [128..191]: ");
            for (int j = 128; j < 192; j++)
                printf("%d ", (int)p8[j]);
            printf("\n");
            printf("[aie_runtime] free buf [192..255]: ");
            for (int j = 192; j < 256; j++)
                printf("%d ", (int)p8[j]);
            printf("\n");
            printf("[aie_runtime] free buf [256..319]: ");
            for (int j = 256; j < 320; j++)
                printf("%d ", (int)p8[j]);
            printf("\n");
            printf("[aie_runtime] free buf [320..383]: ");
            for (int j = 320; j < 384; j++)
                printf("%d ", (int)p8[j]);
            printf("\n");
            printf("[aie_runtime] free buf [384..447]: ");
            for (int j = 384; j < 448; j++)
                printf("%d ", (int)p8[j]);
            printf("\n");
            printf("[aie_runtime] free buf [448..511]: ");
            for (int j = 448; j < 512; j++)
                printf("%d ", (int)p8[j]);
            printf("\n");
        }
        if (!g_dumped_done) {
            g_dumped_done = 1;
            printf("[aie_runtime] BD tracking dump (%d entries):\n", g_bd_track_count);
            for (int i = 0; i < g_bd_track_count; i++) {
                BdTrackEntry *e = &g_bd_track[i];
                const char *dir_label = (e->direction == 0) ? "S2MM" : (e->direction == 1) ? "MM2S" : "???";
                const char *type_str = __bd_tile_type_str(e->tile_type);
                int is_shim = __bd_is_shim(e->tile_type);
                printf("[aie_runtime]   [%d] tile(%u,%u) [%s] bd=%u dir=%s ch=%d pkt_id=%d addr=0x%lx len=%d\n", i,
                       (unsigned)e->col, (unsigned)e->row, type_str, (unsigned)e->bd_id, dir_label, (int)e->channel_id,
                       e->packet_id, (unsigned long)e->dma_addr, e->len);
                if (is_shim && e->buffer != NULL) {
                    int8_t *data = (int8_t *)e->buffer;
                    int byte_len = e->len;
                    printf("[aie_runtime]     shim_buf tile(%u,%u) [%s] dir=%s ch=%d @%p [0..%d] (int8):",
                           (unsigned)e->col, (unsigned)e->row, type_str, dir_label, (int)e->channel_id, e->buffer,
                           byte_len - 1);
                    for (int j = 0; j < byte_len; j++)
                        printf(" %d", (int)data[j]);
                    printf("\n");
                }
            }
        }
    }
    free(ptr);
}

void __Runtime_memcpy(void *dst, const void *src, size_t bytes) {
    memcpy(dst, src, bytes);
    if (AIEHLC_LOG_ENABLED()) {
        printf("[aie_runtime] memcpy(dst=%p, src=%p, bytes=%zu)\n", dst, src, bytes);
        /* Concise int8 summary: print first 64 bytes */
        printf("[aie_runtime] memcpy data (int8): ");
        int8_t *p = (int8_t *)dst;
        int n = (int)bytes < 64 ? (int)bytes : 64;
        for (int i = 0; i < n; i++)
            printf("%d ", (int)p[i]);
        if ((int)bytes > 64)
            printf("...");
        printf("\n");
    }
}

/* ---------------------------------------------------------------------------
 * DMA-capable buffer allocation and VAddr-based sync/free.
 * The driver tracks vaddr → MemInst mappings internally; no buffer registry
 * is needed on the runtime side.
 * ----------------------------------------------------------------------- */

void *__Runtime_alloc_buffer(XAie_DevInst *dev, size_t size_bytes) {
    if (!dev) {
        printf("[aie_runtime] ERROR: alloc_buffer called with dev=NULL. "
               "Call partition() before alloc().\n");
        return NULL;
    }
    XAie_MemInst *mem = XAie_MemAllocate(dev, size_bytes, XAIE_MEM_CACHEABLE);
    if (!mem) {
        printf("[aie_runtime] alloc_buffer: XAie_MemAllocate(%zu) failed\n", size_bytes);
        return NULL;
    }
    void *vaddr = XAie_MemGetVAddr(mem);
    uint64_t devaddr = XAie_MemGetDevAddr(mem);
    AIEHLC_LOG(printf("[aie_runtime] alloc_buffer(%zu) = %p (mem=%p devaddr=0x%lx)\n", size_bytes, vaddr, (void *)mem,
                      (unsigned long)devaddr));
    if (s_alloc_count < MAX_ALLOC_BUFFERS) {
        s_alloc_map[s_alloc_count].vaddr = vaddr;
        s_alloc_map[s_alloc_count].mem = mem;
        s_alloc_map[s_alloc_count].size = size_bytes;
        s_alloc_count++;
    }
    return vaddr;
}

void __Runtime_free_buffer(XAie_DevInst *dev, void *ptr) {
    if (!ptr)
        return;
    if (!dev) {
        printf("[aie_runtime] ERROR: free_buffer called with dev=NULL\n");
        return;
    }
    AieRC rc = XAie_MemFreeVAddr(dev, ptr);
    AIEHLC_LOG(printf("[aie_runtime] free_buffer(%p) via XAie_MemFreeVAddr rc=%d\n", ptr, rc););
}

void __Runtime_free_all_allocs(void) {
    AIEHLC_LOG(printf("[aie_runtime] free_all_allocs: no-op (VAddr mode, driver tracks allocations)\n"););
}

/* ===========================================================================
 * Performance Counter APIs
 * =========================================================================== */

/**
 * Generic perf counter setup on a core tile memory module.
 * Sets Cnt<counter_id>_Start_Event = Cnt<counter_id>_Stop_Event = event,
 * so the counter increments each time the event fires.
 * Resets the counter to 0 before arming.
 */
AieRC __Runtime_perfcnt_setup(XAie_DevInst *dev, XAie_LocType tile, uint8_t counter_id, XAie_Events event) {
    AieRC rc;

    /* Reset counter value to 0 */
    rc = XAie_PerfCounterSet(dev, tile, XAIE_MEM_MOD, counter_id, 0);
    if (rc != XAIE_OK) {
        printf("[aie_runtime] perfcnt_setup: PerfCounterSet failed tile(%u,%u) "
               "counter=%u rc=%d\n",
               (unsigned)tile.Col, (unsigned)tile.Row, (unsigned)counter_id, (int)rc);
        return rc;
    }

    /* Set start and stop events to the same event.
     * This means: counter starts counting on the event, and the same event
     * acts as the stop trigger. Each occurrence increments the counter. */
    rc = XAie_PerfCounterControlSet(dev, tile, XAIE_MEM_MOD, counter_id, event, event);
    if (rc != XAIE_OK) {
        printf("[aie_runtime] perfcnt_setup: PerfCounterControlSet failed "
               "tile(%u,%u) counter=%u event=%u rc=%d\n",
               (unsigned)tile.Col, (unsigned)tile.Row, (unsigned)counter_id, (unsigned)event, (int)rc);
        return rc;
    }

    AIEHLC_LOG(printf("[aie_runtime] perfcnt_setup OK tile(%u,%u) counter=%u event=%u\n", (unsigned)tile.Col,
                      (unsigned)tile.Row, (unsigned)counter_id, (unsigned)event));
    return XAIE_OK;
}

/**
 * Read a perf counter value from a core tile memory module.
 */
AieRC __Runtime_perfcnt_read(XAie_DevInst *dev, XAie_LocType tile, uint8_t counter_id, uint32_t *value) {
    AieRC rc = XAie_PerfCounterGet(dev, tile, XAIE_MEM_MOD, counter_id, value);
    if (rc != XAIE_OK) {
        printf("[aie_runtime] perfcnt_read: PerfCounterGet failed tile(%u,%u) "
               "counter=%u rc=%d\n",
               (unsigned)tile.Col, (unsigned)tile.Row, (unsigned)counter_id, (int)rc);
    } else {
        AIEHLC_LOG(printf("[aie_runtime] perfcnt_read tile(%u,%u) counter=%u value=%u\n", (unsigned)tile.Col,
                          (unsigned)tile.Row, (unsigned)counter_id, (unsigned)*value));
    }
    return rc;
}

/**
 * Set perf counters for core tile MM2S BD finished events:
 *   counter 0 → MM2S channel 0 BD finished
 *   counter 1 → MM2S channel 1 BD finished
 */
AieRC __Runtime_perfcnt_setup_mm2s_bd_finished(XAie_DevInst *dev, XAie_LocType tile) {
    AieRC rc;

    /* Counter 0: MM2S channel 0 BD finished */
    rc = __Runtime_perfcnt_setup(dev, tile, 0, XAIE_EVENT_DMA_MM2S_0_FINISHED_BD_MEM);
    if (rc != XAIE_OK)
        return rc;

    /* Counter 1: MM2S channel 1 BD finished */
    rc = __Runtime_perfcnt_setup(dev, tile, 1, XAIE_EVENT_DMA_MM2S_1_FINISHED_BD_MEM);
    if (rc != XAIE_OK)
        return rc;

    AIEHLC_LOG(printf("[aie_runtime] perfcnt_setup_mm2s_bd_finished OK tile(%u,%u)\n", (unsigned)tile.Col,
                      (unsigned)tile.Row););
    return XAIE_OK;
}

/**
 * Set MM2S BD finished perf counters on all core tiles in a rectangular
 * partition [start_col..end_col] x [start_row..end_row] (inclusive).
 * Skips non-core tiles (rows below XAIE_AIE_TILE_ROW_START).
 */
AieRC __Runtime_perfcnt_setup_mm2s_bd_finished_partition(XAie_DevInst *dev, uint8_t start_col, uint8_t end_col,
                                                         uint8_t start_row, uint8_t end_row) {
    AIEHLC_LOG(printf("[aie_runtime] perfcnt_setup_mm2s_bd_finished_partition "
                      "col[%u..%u] row[%u..%u]\n",
                      (unsigned)start_col, (unsigned)end_col, (unsigned)start_row, (unsigned)end_row));

    for (uint8_t col = start_col; col <= end_col; col++) {
        for (uint8_t row = start_row; row <= end_row; row++) {
            XAie_LocType tile = XAie_TileLoc(col, row);
            if (!__Runtime_is_aie_core_tile(tile)) {
                AIEHLC_LOG(printf("[aie_runtime] perfcnt_partition: skipping non-core "
                                  "tile(%u,%u)\n",
                                  (unsigned)col, (unsigned)row));
                continue;
            }
            AieRC rc = __Runtime_perfcnt_setup_mm2s_bd_finished(dev, tile);
            if (rc != XAIE_OK) {
                printf("[aie_runtime] perfcnt_partition: FAILED at tile(%u,%u) "
                       "rc=%d\n",
                       (unsigned)col, (unsigned)row, (int)rc);
                return rc;
            }
        }
    }

    AIEHLC_LOG(printf("[aie_runtime] perfcnt_setup_mm2s_bd_finished_partition OK\n"););
    return XAIE_OK;
}

/**
 * Read and print MM2S BD finished perf counters across a partition.
 */
void __Runtime_perfcnt_read_mm2s_bd_finished_partition(XAie_DevInst *dev, uint8_t start_col, uint8_t end_col,
                                                       uint8_t start_row, uint8_t end_row) {
    AIEHLC_LOG(printf("[aie_runtime] perfcnt_read_mm2s_bd_finished_partition "
                      "col[%u..%u] row[%u..%u]\n",
                      (unsigned)start_col, (unsigned)end_col, (unsigned)start_row, (unsigned)end_row));

    for (uint8_t col = start_col; col <= end_col; col++) {
        for (uint8_t row = start_row; row <= end_row; row++) {
            XAie_LocType tile = XAie_TileLoc(col, row);
            if (!__Runtime_is_aie_core_tile(tile))
                continue;
            uint32_t val0 = 0, val1 = 0;
            __Runtime_perfcnt_read(dev, tile, 0, &val0);
            __Runtime_perfcnt_read(dev, tile, 1, &val1);
            AIEHLC_LOG(printf("[aie_runtime] perfcnt tile(%u,%u) MM2S_ch0_bd_done=%u "
                              "MM2S_ch1_bd_done=%u\n",
                              (unsigned)col, (unsigned)row, val0, val1));
        }
    }
}

void __Runtime_sync_for_dev(XAie_DevInst *dev, void *ptr, size_t size) {
    if (dev) {
        AieRC rc = XAie_MemSyncForDevVAddr(dev, ptr, (uint64_t)size);
        AIEHLC_LOG(printf("[aie_runtime] sync_for_dev(%p, %zu) via VAddr rc=%d\n", ptr, size, rc););
    } else {
        Xil_DCacheFlushRange((UINTPTR)ptr, size);
        AIEHLC_LOG(printf("[aie_runtime] sync_for_dev(%p, %zu) via DCacheFlushRange\n", ptr, size));
    }
}

void __Runtime_sync_for_cpu(XAie_DevInst *dev, void *ptr, size_t size) {
    if (dev) {
        AieRC rc = XAie_MemSyncForCPUVAddr(dev, ptr, (uint64_t)size);
        AIEHLC_LOG(printf("[aie_runtime] sync_for_cpu(%p, %zu) via VAddr rc=%d\n", ptr, size, rc););
    } else {
        Xil_DCacheInvalidateRange((UINTPTR)ptr, size);
        AIEHLC_LOG(printf("[aie_runtime] sync_for_cpu(%p, %zu) via DCacheInvalidateRange\n", ptr, size));
    }
}

/* Active kernel ELF pointer — set by __Runtime_set_kernel_elf() before load_kernel_group.
 * In single-kernel mode, host.cc declares the extern and calls set_kernel_elf once.
 * In multi-kernel mode, each __aie_launch dispatch calls set_kernel_elf with the
 * appropriate _binary_kernel_<name>_start before calling host_canonicalized_<name>. */
static unsigned char *s_active_kernel_elf = NULL;

void __Runtime_set_kernel_elf(unsigned char *elf_start) { s_active_kernel_elf = elf_start; }

/**
 */
void __Runtime_platform_init(void) {
    // Flush any pre-existing dirty cache lines to DDR.
    // Cache stays ENABLED — per-buffer sync (__Runtime_sync_for_dev/cpu)
    // handles coherency at launch time instead of globally disabling DCache.
    // Xil_DCacheFlush();
    // Xil_DCacheDisable();
    // Xil_ICacheDisable();
}

void __Runtime_init(void) {
    AIEHLC_LOG(printf("[aie_runtime] __Runtime_init--\n"););
    __Runtime_platform_init();

    AIEHLC_LOG(printf("[aie_runtime] _init OK\n"));
}

/**
 * Initialize routing handler (reference: aieml_perf.cc line 128)
 * Must be called after device init.
 */
void __Runtime_routing_init(XAie_DevInst *dev) {
    AIEHLC_LOG(printf("[aie_runtime] routing_init start\n"););
    g_RoutingInst = XAie_InitRoutingHandler(dev);
    routing(dev);
    AIEHLC_LOG(printf("[aie_runtime] 2-routing_init OK----\n"));
}

/* ---------------------------------------------------------------------------
 * Partition init helper: plain default partition initialize.
 * ----------------------------------------------------------------------- */
#if AIE_GEN >= 2
static AieRC __Runtime_partition_initialize(XAie_DevInst *dev) {
#ifdef __AIESIM__
    (void)dev;
    return XAIE_OK;
#else
    return XAie_PartitionInitialize(dev, NULL);
#endif
}
#endif

/**
 * Teardown partition (reference: aieml_perf.cc lines 348-352)
 */
AieRC __Runtime_device_teardown(XAie_DevInst *dev) {
    AIEHLC_LOG(printf("[aie_runtime] device_teardown\n"););
    if (AIE_DEBUG_HAS_FLAG(g_runtime_debug_level, AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)) {
        printf("[aie_runtime] device_teardown SKIPPED (DISABLE_PARTITIONTEARDOWN flag set)\n");
        return XAIE_OK;
    }
    AieRC RC = XAie_PartitionTeardown(dev);
    AIEHLC_LOG(printf("[aie_runtime] device_teardown done rc=%d\n", (int)RC););
    return RC;
}

/* ---------------------------------------------------------------------------
 * Explicit init/teardown: heap-allocate XAie_DevInst, return pointer.
 * Caller owns the pointer and must call __Runtime_explicit_teardown().
 * ----------------------------------------------------------------------- */

XAie_DevInst *__Runtime_explicit_init(void) {
    __Runtime_platform_init();
    XAie_DevInst *dev = (XAie_DevInst *)calloc(1, sizeof(XAie_DevInst));
    if (!dev) {
        printf("[aie_runtime] explicit_init: malloc failed\n");
        return NULL;
    }

#if AIE_GEN == 5 && !defined(__AIESIM__)
    // XAie_SetXprodEnable(dev, XAIE_DISABLE);
#endif
    AieRC RC = XAie_CfgInitialize(dev, &g_Config);
    if (RC != XAIE_OK) {
        printf("[aie_runtime] explicit_init CfgInitialize failed: %d\n", (int)RC);
        free(dev);
        return NULL;
    }

#ifdef __AIESIM__
    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_SIM);
#else
    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_BAREMETAL);
#endif

#if AIE_GEN >= 2
#ifndef __AIESIM__
    if (dev->Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
#if AIE_GEN == 5
        RC = XAie_UpdateNpiAddr(dev, 0xf6d50000);
#else
        RC = XAie_UpdateNpiAddr(dev, 0xF6D10000);
#endif
        if (RC != XAIE_OK) {
            free(dev);
            return NULL;
        }
    }
#endif
    RC = __Runtime_partition_initialize(dev);
#else
    XAie_PmRequestTiles(dev, NULL, 0);
    RC = XAIE_OK;
#endif

    if (RC != XAIE_OK) {
        printf("[aie_runtime] explicit_init partition/pm failed: %d\n", (int)RC);
        free(dev);
        return NULL;
    }

    __Runtime_routing_init(dev);

    if (AIE_DEBUG_HAS_FLAG(g_runtime_debug_level, AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER)) {
        __Runtime_perfcnt_setup_mm2s_bd_finished_partition(dev, 0, XAIE_NUM_COLS - 1, XAIE_AIE_TILE_ROW_START,
                                                           XAIE_AIE_TILE_ROW_START + XAIE_AIE_TILE_NUM_ROWS - 1);
    }

    AIEHLC_LOG(printf("[aie_runtime] explicit_init OK dev=%p\n", (void *)dev););
    g_DevInst = dev;
    return dev;
}

XAie_DevInst *__Runtime_explicit_init_partition(int startCol, int numCols) {
    __Runtime_platform_init();
    // calloc zero-initializes the struct so IsReady == 0, which
    // XAie_SetupPartitionConfig requires (it rejects IsReady != 0).
    XAie_DevInst *dev = (XAie_DevInst *)calloc(1, sizeof(XAie_DevInst));
    if (!dev) {
        printf("[aie_runtime] explicit_init_partition: calloc failed\n");
        return NULL;
    }

    // XAie_SetupPartitionConfig must be called BEFORE XAie_CfgInitialize.
    // It requires IsReady == 0 (freshly zeroed struct) and sets
    // BaseAddr/StartCol/NumCols. CfgInitialize then reads those fields
    // and sets IsReady = XAIE_COMPONENT_IS_READY.
    AieRC RC =
        XAie_SetupPartitionConfig(dev, XAIE_BASE_ADDR + ((uint64_t)startCol << XAIE_COL_SHIFT), startCol, numCols);
    if (RC != XAIE_OK) {
        printf("[aie_runtime] explicit_init_partition SetupPartitionConfig failed: %d\n", (int)RC);
        free(dev);
        return NULL;
    }

#if AIE_GEN == 5 && !defined(__AIESIM__)
    // XAie_SetXprodEnable(dev, XAIE_DISABLE);
#endif
    RC = XAie_CfgInitialize(dev, &g_Config);
    if (RC != XAIE_OK) {
        printf("[aie_runtime] explicit_init_partition CfgInitialize failed: %d\n", (int)RC);
        free(dev);
        return NULL;
    }

#ifdef __AIESIM__
    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_SIM);
#else
    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_BAREMETAL);
#endif

#if AIE_GEN >= 2
#ifndef __AIESIM__
    if (dev->Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
#if AIE_GEN == 5
        RC = XAie_UpdateNpiAddr(dev, 0xf6d50000);
#else
        RC = XAie_UpdateNpiAddr(dev, 0xF6D10000);
#endif
        if (RC != XAIE_OK) {
            free(dev);
            return NULL;
        }
    }
#endif
    RC = __Runtime_partition_initialize(dev);
#else
    XAie_PmRequestTiles(dev, NULL, 0);
    RC = XAIE_OK;
#endif

    if (RC != XAIE_OK) {
        printf("[aie_runtime] explicit_init_partition partition/pm failed: %d\n", (int)RC);
        free(dev);
        return NULL;
    }

    /* Shim DMA loopback self-test: write 0xABCD at DDR offset 10MB,
     * loopback-copy to DDR offset 20MB, verify. Uses col=startCol. */
    /*
    {
        uint32_t *src = (uint32_t *)(uintptr_t)(1024U * 1024U * 10U);
        uint32_t *dst = (uint32_t *)(uintptr_t)(1024U * 1024U * 20U);
        uint32_t len = 512 * sizeof(uint32_t);
        uint32_t num_words = len / sizeof(uint32_t);

        AIEHLC_LOG(printf("[aie_runtime] shim_dma_loopback_test 3: src=%p dst=%p len=%u\n",
               (void *)src, (void *)dst, (unsigned)len));

        for (uint32_t i = 0; i < num_words; i++) {
            src[i] = 0xABCD;
            dst[i] = 0;
        }

        int lb_rc = AieRt_ShimDmaLoopback(dev, (uint8_t)0, src, dst, len);
        AIEHLC_LOG(printf("[aie_runtime] shim_dma_loopback_test: rc=%d\n", lb_rc););
    }
    */
    __Runtime_routing_init(dev);

    if (AIE_DEBUG_HAS_FLAG(g_runtime_debug_level, AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER)) {
        __Runtime_perfcnt_setup_mm2s_bd_finished_partition(dev, 0, XAIE_NUM_COLS - 1, XAIE_AIE_TILE_ROW_START,
                                                           XAIE_AIE_TILE_ROW_START + XAIE_AIE_TILE_NUM_ROWS - 1);
    }

    AIEHLC_LOG(printf("[aie_runtime] explicit_init_partition OK startCol=%d numCols=%d dev=%p\n", startCol, numCols,
                      (void *)dev););
    g_DevInst = dev;
    return dev;
}

void __Runtime_explicit_teardown(XAie_DevInst *dev) {
    if (dev == NULL)
        return;
    AIEHLC_LOG(printf("[aie_runtime] explicit_teardown dev=%p\n", (void *)dev););

    if (AIEHLC_LOG_ENABLED()) {
        AieRtSS_PrintRange(dev, 0, 3, 0, 5, /*print_all=*/0);
        /* Dump raw BD registers for shim tiles used by BD tracking */
        {
            uint8_t shim_cols[BD_TRACK_MAX];
            int shim_col_count = 0;
            for (int i = 0; i < g_bd_track_count; i++) {
                if (!__bd_is_shim(g_bd_track[i].tile_type))
                    continue;
                int dup = 0;
                for (int j = 0; j < shim_col_count; j++) {
                    if (shim_cols[j] == g_bd_track[i].col) {
                        dup = 1;
                        break;
                    }
                }
                if (!dup)
                    shim_cols[shim_col_count++] = g_bd_track[i].col;
            }
            for (int i = 0; i < shim_col_count; i++)
                AieRt_PrintShimBdRawAll(dev, shim_cols[i]);
        }
        /* Dump perf counters for all core tiles seen in BD tracking */
        {
            XAie_LocType core_tiles[BD_TRACK_MAX];
            int core_tile_count = 0;
            for (int i = 0; i < g_bd_track_count; i++) {
                if (g_bd_track[i].tile_type != XAIEGBL_TILE_TYPE_AIETILE)
                    continue;
                int dup = 0;
                for (int j = 0; j < core_tile_count; j++) {
                    if (core_tiles[j].Col == g_bd_track[i].col && core_tiles[j].Row == g_bd_track[i].row) {
                        dup = 1;
                        break;
                    }
                }
                if (!dup && core_tile_count < BD_TRACK_MAX)
                    core_tiles[core_tile_count++] = XAie_TileLoc(g_bd_track[i].col, g_bd_track[i].row);
            }
            if (core_tile_count > 0)
                AieRt_PrintCoreTilePerfCountersAll(dev, core_tiles, core_tile_count);
        }
    }

    __Runtime_free_all_allocs();
    __Runtime_device_teardown(dev);
    free(dev);
    AIEHLC_LOG(printf("[aie_runtime] explicit_teardown done\n"););
}

/* ---------------------------------------------------------------------------
 * Partition registry: init-once, get-existing, teardown-all.
 * Used by the multi-kernel/multi-partition programming model (aieArray + aieMesh).
 * Each aieMesh carries a meshId. The runtime tracks which meshIds have been
 * initialized so sequential kernel launches on the same mesh skip re-init.
 * aieArray::synchronize() calls __Runtime_teardown_all() to clean up.
 * ----------------------------------------------------------------------- */

#define PARTITION_REGISTRY_MAX 16

typedef struct {
    int meshId;
    XAie_DevInst *dev;
} PartitionRegistryEntry;

static PartitionRegistryEntry g_partition_registry[PARTITION_REGISTRY_MAX];
static int g_partition_registry_count = 0;

int __Runtime_partition_is_initialized(int meshId) {
    for (int i = 0; i < g_partition_registry_count; i++) {
        if (g_partition_registry[i].meshId == meshId)
            return 1;
    }
    return 0;
}

XAie_DevInst *__Runtime_get_partition_dev(int meshId) {
    for (int i = 0; i < g_partition_registry_count; i++) {
        if (g_partition_registry[i].meshId == meshId)
            return g_partition_registry[i].dev;
    }
    printf("[aie_runtime] ERROR: partition meshId=%d not found in registry\n", meshId);
    return NULL;
}

void __Runtime_register_partition(int meshId, XAie_DevInst *dev) {
    if (g_partition_registry_count >= PARTITION_REGISTRY_MAX) {
        printf("[aie_runtime] ERROR: partition registry full (max=%d)\n", PARTITION_REGISTRY_MAX);
        return;
    }
    g_partition_registry[g_partition_registry_count].meshId = meshId;
    g_partition_registry[g_partition_registry_count].dev = dev;
    g_partition_registry_count++;
    AIEHLC_LOG(printf("[aie_runtime] register_partition meshId=%d dev=%p (total=%d)\n", meshId, (void *)dev,
                      g_partition_registry_count));
}

XAie_DevInst *__Runtime_init_mesh_partition(int meshId, int startCol, int numCols) {
    if (__Runtime_partition_is_initialized(meshId)) {
        return __Runtime_get_partition_dev(meshId);
    }
    XAie_DevInst *dev = __Runtime_explicit_init_partition(startCol, numCols);
    if (!dev) {
        printf("[aie_runtime] ERROR: init_mesh_partition failed meshId=%d startCol=%d numCols=%d\n", meshId, startCol,
               numCols);
        return NULL;
    }
    __Runtime_register_partition(meshId, dev);
    return dev;
}

void __Runtime_teardown_all(void) {
    AIEHLC_LOG(printf("[aie_runtime] teardown_all: %d partitions registered\n", g_partition_registry_count););
    for (int i = 0; i < g_partition_registry_count; i++) {
        AIEHLC_LOG(printf("[aie_runtime] teardown_all: meshId=%d dev=%p\n", g_partition_registry[i].meshId,
                          (void *)g_partition_registry[i].dev));
        __Runtime_explicit_teardown(g_partition_registry[i].dev);
        g_partition_registry[i].dev = NULL;
    }
    g_partition_registry_count = 0;
    AIEHLC_LOG(printf("[aie_runtime] teardown_all done\n"););
}

/**
 * Configure DMA buffer descriptor.
 * @param len  Transfer length in bytes.
 */
XAie_DmaDesc __Runtime_dma_bd_config(XAie_DevInst *dev, XAie_LocType tile, void *buffer, int32_t bd_id, int32_t len,
                                     int32_t next_bd, int32_t enable_packet, int32_t packet_id, int32_t acquire_lock_id,
                                     int32_t acquire_lock_val, int32_t release_lock_id, int32_t release_lock_val,
                                     int32_t out_of_order_bd_id) {
    XAie_DmaDesc DmaInst;
    XAie_DmaDescInit(dev, &DmaInst, tile);
    uint8_t tile_type = XAie_GetTileTypefromLoc(dev, tile);
    /* Both shim and core tiles: buffer is a byte address.
     * Shim tiles: buffer IS the DDR physical address.
     * Core tiles: buffer is a DMA-view byte address (core_proc_addr - 0x70000),
     *   produced by passblueprinttoschedule after CoreMemAllocator conversion. */
    uint64_t dma_addr = (uint64_t)(uintptr_t)buffer;
    if (__bd_is_shim(tile_type)) {
        uint64_t offset = 0;
        XAie_MemInst *mem = __vaddr_to_mem_offset(buffer, &offset);
        if (mem) {
            uint64_t dev = XAie_MemGetDevAddr(mem) + offset;
            AIEHLC_LOG(printf("[aie_runtime] bd_config shim: VAddr=%p → DevAddr=0x%lx (offset=%lu)\n", buffer,
                              (unsigned long)dev, (unsigned long)offset));
            dma_addr = dev;
#ifdef __AIESIM__
            ess_WriteGM(dev, buffer, (uint64_t)len);
            const int8_t *dbg = (const int8_t *)buffer;
            AIEHLC_LOG(printf("[aie_runtime] ess_WriteGM DevAddr=0x%lx len=%d data[0..3]=%d,%d,%d,%d\n",
                              (unsigned long)dev, len, dbg[0], dbg[1], dbg[2], dbg[3]));
#endif
        }
    }
    XAie_DmaSetAddrLen(&DmaInst, dma_addr, (uint32_t)len);

    if (acquire_lock_id >= 0 && release_lock_id >= 0) {
        XAie_DmaSetLock(&DmaInst, XAie_LockInit(acquire_lock_id, acquire_lock_val),
                        XAie_LockInit(release_lock_id, release_lock_val));
    }

    if (next_bd >= 0) {
        XAie_DmaSetNextBd(&DmaInst, (uint8_t)next_bd, XAIE_ENABLE);
    }

    if (enable_packet) {
        XAie_DmaSetPkt(&DmaInst, XAie_PacketInit(packet_id, 0));
    }

    if (out_of_order_bd_id >= 0) {
        XAie_DmaSetOutofOrderBdId(&DmaInst, (uint8_t)out_of_order_bd_id);
        AIEHLC_LOG(printf("[aie_runtime] bd_config: set out_of_order_bd_id=%d\n", out_of_order_bd_id););
    }

    XAie_DmaEnableBd(&DmaInst);
    AieRC bd_rc = XAie_DmaWriteBd(dev, &DmaInst, tile, (uint8_t)bd_id);
    AIEHLC_LOG(printf(
        "[aie_runtime] bd_config tile(%u,%u) bd=%d addr=0x%lx len=%d next=%d lock_acq=%d/%d lock_rel=%d/%d "
        "pkt=%d/%d ooo_bd=%d rc=%d\n",
        (unsigned)tile.Col, (unsigned)tile.Row, bd_id, (unsigned long)dma_addr, len, next_bd, acquire_lock_id,
        acquire_lock_val, release_lock_id, release_lock_val, enable_packet, packet_id, out_of_order_bd_id, (int)bd_rc));

    /* Track this BD for debug dump in __Runtime_free */
    if (AIE_DEBUG_LEVEL(g_runtime_debug_level) >= 1 && g_bd_track_count < BD_TRACK_MAX) {
        BdTrackEntry *e = &g_bd_track[g_bd_track_count++];
        e->col = tile.Col;
        e->row = tile.Row;
        e->bd_id = (uint8_t)bd_id;
        e->tile_type = tile_type;
        e->buffer = __bd_is_shim(tile_type) ? buffer : NULL;
        e->dma_addr = dma_addr;
        e->len = len;
        e->packet_id = packet_id;
        e->direction = -1;  /* unknown until startio */
        e->channel_id = -1; /* unknown until startio */
        e->next_bd = (int8_t)next_bd;
    }

    if (AIEHLC_LOG_ENABLED()) {
        if (tile_type == XAIEGBL_TILE_TYPE_AIETILE) {
            printf("[aie_runtime] bd_config core(%u,%u) bd=%d dma_addr=0x%lx len=%d pkt_id=%d\n", (unsigned)tile.Col,
                   (unsigned)tile.Row, bd_id, (unsigned long)dma_addr, len, packet_id);
            /* Write pattern col*100+row*10+bd_id into the BD region at dma_addr.
             * len is in bytes; convert to int32 count for the pattern array.
             * ping BD (buffer=0) writes at byte 0; pong BD (buffer=64) writes at byte 256.
             * Covers the full BD transfer so shim sees consistent values (RC-1 fix). */
            int32_t dbg_pat[64];
            int32_t pat_val = (int32_t)(tile.Col * 100 + tile.Row * 10 + bd_id);
            int32_t write_words = len / (int32_t)sizeof(int32_t);
            if (write_words <= 0)
                write_words = 1;
            if (write_words > 64)
                write_words = 64;
            for (int _i = 0; _i < write_words; _i++)
                dbg_pat[_i] = pat_val;
            XAie_DataMemBlockWrite(dev, tile, (u32)dma_addr, dbg_pat, (u32)(write_words * sizeof(int32_t)));
            /* Read back and print to verify */
            int32_t dbg_read[64];
            XAie_DataMemBlockRead(dev, tile, (u32)dma_addr, dbg_read, (u32)(write_words * sizeof(int32_t)));
            AIEHLC_LOG(printf("[aie_runtime] core(%u,%u) bd=%d dma_addr=0x%lx pat=%d read:", (unsigned)tile.Col,
                              (unsigned)tile.Row, bd_id, (unsigned long)dma_addr, pat_val));
            for (int _i = 0; _i < write_words; _i++)
                printf(" %d", dbg_read[_i]);
            AIEHLC_LOG(printf("\n"););
        } else {
            printf("[aie_runtime] bd_config shim(%u,%u) bd=%d dma_addr=0x%lx len=%d pkt_id=%d buf=%p\n",
                   (unsigned)tile.Col, (unsigned)tile.Row, bd_id, (unsigned long)dma_addr, len, packet_id, buffer);
        }
    }

    return DmaInst;
}

/**
 * Configure DMA buffer descriptor with multi-dimensional addressing.
 * Uses XAie_DmaSetMultiDimAddr with stride/wrap descriptors to enable
 * DMA hardware transpose/reshape during data transfer.
 * @param len  Transfer length in bytes.
 */
XAie_DmaDesc __Runtime_dma_bd_config_multidim(XAie_DevInst *dev, XAie_LocType tile, void *buffer, int32_t bd_id,
                                              int32_t len, int32_t next_bd, int32_t enable_packet, int32_t packet_id,
                                              int32_t acquire_lock_id, int32_t acquire_lock_val,
                                              int32_t release_lock_id, int32_t release_lock_val,
                                              int32_t out_of_order_bd_id, int32_t num_dims, int32_t dim_stride0,
                                              int32_t dim_wrap0, int32_t dim_stride1, int32_t dim_wrap1,
                                              int32_t dim_stride2, int32_t dim_wrap2, int32_t dim_stride3,
                                              int32_t dim_wrap3) {

    XAie_DmaDesc DmaInst;
    XAie_DmaDescInit(dev, &DmaInst, tile);
    uint8_t tile_type = XAie_GetTileTypefromLoc(dev, tile);
    uint64_t dma_addr = (uint64_t)(uintptr_t)buffer;
    if (__bd_is_shim(tile_type)) {
        uint64_t offset = 0;
        XAie_MemInst *mem = __vaddr_to_mem_offset(buffer, &offset);
        if (mem) {
            dma_addr = XAie_MemGetDevAddr(mem) + offset;
#ifdef __AIESIM__
            ess_WriteGM(dma_addr, buffer, (uint64_t)len);
#endif
        }
    }

    /* Build dimension descriptors — split address dims vs iteration */
    int32_t strides[4] = {dim_stride0, dim_stride1, dim_stride2, dim_stride3};
    int32_t wraps[4] = {dim_wrap0, dim_wrap1, dim_wrap2, dim_wrap3};
    if (num_dims > 4)
        num_dims = 4;

    /* Address dimensions: first min(num_dims, 3) */
    int addrDims = (num_dims <= 3) ? num_dims : 3;
    XAie_DmaDimDesc dimDescs[3];
    for (int i = 0; i < addrDims; i++) {
        /* IR strides are in byte units; XAie expects 32-bit word units (÷4) */
        if (strides[i] % 4 != 0) {
            printf("[aie_runtime] ERROR: dim_stride[%d]=%d not divisible by 4 "
                   "(must be 32-bit aligned)\n",
                   i, strides[i]);
        }
        dimDescs[i].AieMlDimDesc.StepSize = (uint32_t)(strides[i] / 4);
        dimDescs[i].AieMlDimDesc.Wrap = (uint16_t)wraps[i];
    }
    XAie_DmaTensor tensor;
    tensor.NumDim = (uint8_t)addrDims;
    tensor.Dim = dimDescs;
    XAie_DmaSetMultiDimAddr(&DmaInst, &tensor, dma_addr, (uint32_t)len);

    /* Iteration dimension: 4th dim if present */
    if (num_dims == 4) {
        /* IR strides are in byte units; XAie expects 32-bit word units (÷4) */
        int32_t iterStepSize = strides[3] / 4;
        XAie_DmaSetBdIteration(&DmaInst, iterStepSize, wraps[3], 0);
        AIEHLC_LOG(printf("[aie_runtime] bd_config_multidim: iteration stride=%d (words, %d bytes) wrap=%d\n",
                          iterStepSize, strides[3], wraps[3]));
    }

    if (acquire_lock_id >= 0 && release_lock_id >= 0) {
        XAie_DmaSetLock(&DmaInst, XAie_LockInit(acquire_lock_id, acquire_lock_val),
                        XAie_LockInit(release_lock_id, release_lock_val));
    }

    if (next_bd >= 0) {
        XAie_DmaSetNextBd(&DmaInst, (uint8_t)next_bd, XAIE_ENABLE);
    }

    if (enable_packet) {
        XAie_DmaSetPkt(&DmaInst, XAie_PacketInit(packet_id, 0));
    }

    if (out_of_order_bd_id >= 0) {
        XAie_DmaSetOutofOrderBdId(&DmaInst, (uint8_t)out_of_order_bd_id);
        AIEHLC_LOG(printf("[aie_runtime] bd_config_multidim: set out_of_order_bd_id=%d\n", out_of_order_bd_id););
    }

    XAie_DmaEnableBd(&DmaInst);
    AieRC bd_rc = XAie_DmaWriteBd(dev, &DmaInst, tile, (uint8_t)bd_id);
    AIEHLC_LOG(printf("[aie_runtime] bd_config_multidim tile(%u,%u) bd=%d addr=0x%lx len=%d next=%d "
                      "lock_acq=%d/%d lock_rel=%d/%d pkt=%d/%d ooo_bd=%d num_dims=%d rc=%d\n",
                      (unsigned)tile.Col, (unsigned)tile.Row, bd_id, (unsigned long)dma_addr, len, next_bd,
                      acquire_lock_id, acquire_lock_val, release_lock_id, release_lock_val, enable_packet, packet_id,
                      out_of_order_bd_id, num_dims, (int)bd_rc));
    for (int i = 0; i < num_dims; i++) {
        AIEHLC_LOG(printf("[aie_runtime]   dim[%d] stride=%d wrap=%d\n", i, strides[i], wraps[i]););
    }

    /* Track this BD for debug dump */
    if (AIE_DEBUG_LEVEL(g_runtime_debug_level) >= 1 && g_bd_track_count < BD_TRACK_MAX) {
        BdTrackEntry *e = &g_bd_track[g_bd_track_count++];
        e->col = tile.Col;
        e->row = tile.Row;
        e->bd_id = (uint8_t)bd_id;
        e->tile_type = tile_type;
        e->buffer = __bd_is_shim(tile_type) ? buffer : NULL;
        e->dma_addr = dma_addr;
        e->len = len;
        e->packet_id = packet_id;
        e->direction = -1;
        e->channel_id = -1;
        e->next_bd = (int8_t)next_bd;
    }

    return DmaInst;
}

/**
 * Configure DMA buffer descriptor with multi-dimensional addressing (D0-D2)
 * plus a separate iteration dimension for out-of-order packet reception.
 * The iteration dimension (via XAie_DmaSetBdIteration) causes the BD to
 * re-execute iter_wrap times with the base address advancing by
 * iter_step_size bytes between each OOO packet arrival.
 * @param len            Transfer length in bytes per iteration.
 * @param num_dims       Number of address dimensions (max 3, D0-D2).
 * @param iter_step_size Iteration step size in bytes (÷4 for 32-bit words).
 * @param iter_wrap      Number of times the BD re-executes (OOO packets).
 */
XAie_DmaDesc __Runtime_dma_bd_config_multidim_ooo(XAie_DevInst *dev, XAie_LocType tile, void *buffer, int32_t bd_id,
                                                  int32_t len, int32_t next_bd, int32_t enable_packet,
                                                  int32_t packet_id, int32_t acquire_lock_id, int32_t acquire_lock_val,
                                                  int32_t release_lock_id, int32_t release_lock_val,
                                                  int32_t out_of_order_bd_id, int32_t num_dims, int32_t dim_stride0,
                                                  int32_t dim_wrap0, int32_t dim_stride1, int32_t dim_wrap1,
                                                  int32_t dim_stride2, int32_t dim_wrap2, int32_t iter_step_size,
                                                  int32_t iter_wrap) {

    XAie_DmaDesc DmaInst;
    XAie_DmaDescInit(dev, &DmaInst, tile);
    uint8_t tile_type = XAie_GetTileTypefromLoc(dev, tile);
    uint64_t dma_addr = (uint64_t)(uintptr_t)buffer;
    if (__bd_is_shim(tile_type)) {
        uint64_t offset = 0;
        XAie_MemInst *mem = __vaddr_to_mem_offset(buffer, &offset);
        if (mem) {
#ifdef __AIESIM__
            dma_addr = XAie_MemGetDevAddr(mem) + offset;
            uint64_t gm_len = (iter_wrap > 1) ? ((uint64_t)iter_step_size * (uint64_t)(iter_wrap - 1) + (uint64_t)len)
                                              : (uint64_t)len;
            ess_WriteGM(dma_addr, buffer, gm_len);
#endif
        }
    }

    /* Build address dimension descriptors (D0-D2 only, max 3) */
    int32_t strides[3] = {dim_stride0, dim_stride1, dim_stride2};
    int32_t wraps[3] = {dim_wrap0, dim_wrap1, dim_wrap2};
    if (num_dims > 3)
        num_dims = 3;

    XAie_DmaDimDesc dimDescs[3];
    for (int i = 0; i < num_dims; i++) {
        /* IR strides are in byte units; XAie expects 32-bit word units (÷4) */
        if (strides[i] % 4 != 0) {
            printf("[aie_runtime] ERROR: dim_stride[%d]=%d not divisible by 4 "
                   "(must be 32-bit aligned)\n",
                   i, strides[i]);
        }
        dimDescs[i].AieMlDimDesc.StepSize = (uint32_t)(strides[i] / 4);
        dimDescs[i].AieMlDimDesc.Wrap = (uint16_t)wraps[i];
    }
    XAie_DmaTensor tensor;
    tensor.NumDim = (uint8_t)num_dims;
    tensor.Dim = dimDescs;
    XAie_DmaSetMultiDimAddr(&DmaInst, &tensor, dma_addr, (uint32_t)len);

    /* Iteration dimension: BD re-executes iter_wrap times with address
     * advancing by iter_step_size bytes between each OOO packet trigger. */
    if (iter_wrap > 1) {
        int32_t iterStepWords = iter_step_size / 4;
        XAie_DmaSetBdIteration(&DmaInst, iterStepWords, iter_wrap, 0);
        AIEHLC_LOG(printf("[aie_runtime] bd_config_multidim_ooo: iteration step=%d words (%d bytes) wrap=%d\n",
                          iterStepWords, iter_step_size, iter_wrap));
    }

    if (acquire_lock_id >= 0 && release_lock_id >= 0) {
        XAie_DmaSetLock(&DmaInst, XAie_LockInit(acquire_lock_id, acquire_lock_val),
                        XAie_LockInit(release_lock_id, release_lock_val));
    }

    if (next_bd >= 0) {
        XAie_DmaSetNextBd(&DmaInst, (uint8_t)next_bd, XAIE_ENABLE);
    }

    if (enable_packet) {
        XAie_DmaSetPkt(&DmaInst, XAie_PacketInit(packet_id, 0));
    }

    if (out_of_order_bd_id >= 0) {
        XAie_DmaSetOutofOrderBdId(&DmaInst, (uint8_t)out_of_order_bd_id);
        AIEHLC_LOG(printf("[aie_runtime] bd_config_multidim_ooo: set out_of_order_bd_id=%d\n", out_of_order_bd_id););
    }

    XAie_DmaEnableBd(&DmaInst);
    AieRC bd_rc = XAie_DmaWriteBd(dev, &DmaInst, tile, (uint8_t)bd_id);
    AIEHLC_LOG(printf("[aie_runtime] bd_config_multidim_ooo tile(%u,%u) bd=%d addr=0x%lx len=%d next=%d "
                      "lock_acq=%d/%d lock_rel=%d/%d pkt=%d/%d ooo_bd=%d num_dims=%d "
                      "iter_step=%d iter_wrap=%d rc=%d\n",
                      (unsigned)tile.Col, (unsigned)tile.Row, bd_id, (unsigned long)dma_addr, len, next_bd,
                      acquire_lock_id, acquire_lock_val, release_lock_id, release_lock_val, enable_packet, packet_id,
                      out_of_order_bd_id, num_dims, iter_step_size, iter_wrap, (int)bd_rc));
    for (int i = 0; i < num_dims; i++) {
        AIEHLC_LOG(printf("[aie_runtime]   dim[%d] stride=%d wrap=%d\n", i, strides[i], wraps[i]););
    }

    /* Track this BD for debug dump */
    if (AIE_DEBUG_LEVEL(g_runtime_debug_level) >= 1 && g_bd_track_count < BD_TRACK_MAX) {
        BdTrackEntry *e = &g_bd_track[g_bd_track_count++];
        e->col = tile.Col;
        e->row = tile.Row;
        e->bd_id = (uint8_t)bd_id;
        e->tile_type = tile_type;
        e->buffer = __bd_is_shim(tile_type) ? buffer : NULL;
        e->dma_addr = dma_addr;
        e->len = len;
        e->packet_id = packet_id;
        e->direction = -1;
        e->channel_id = -1;
        e->next_bd = (int8_t)next_bd;
    }

    return DmaInst;
}

/**
 * Create an I/O handle bundling everything needed to launch/wait a DMA transfer.
 * Just packs the arguments into a struct_io value; no hardware is touched here.
 *
 * @param tile_loc   Location (col,row) of the tile that owns this DMA channel/BD.
 * @param dma_desc   Pre-built DMA buffer descriptor (base address, length,
 *                   multi-dim step/wrap, lock/packet config) for the transfer.
 * @param channel_id DMA channel index on the tile that runs this transfer.
 * @param bd_id      Buffer-descriptor ID holding this transfer's configuration.
 * @param direction  Transfer direction: XAIE_DMA_MM2S (tile memory -> stream) or
 *                   XAIE_DMA_S2MM (stream -> tile memory).
 * @param mem        Optional XAie_MemInst for the backing DMA buffer (e.g. a
 *                   DDR/shim-backed allocation used for cache sync); NULL if none.
 * @return struct_io populated with the above (channel_id/bd_id cast to uint8_t).
 */
struct_io __Runtime_dma_createio(XAie_LocType tile_loc, XAie_DmaDesc dma_desc, int32_t channel_id, int32_t bd_id,
                                 XAie_DmaDirection direction, XAie_MemInst *mem) {
    struct_io io;
    io.desc = dma_desc;
    io.tile_loc = tile_loc;
    io.channel_id = (uint8_t)channel_id;
    io.bd_id = (uint8_t)bd_id;
    io.direction = direction;
    io.mem = mem;
    return io;
}

/**
 * 4-argument convenience wrapper over __Runtime_dma_createio for the common case
 * of no backing XAie_MemInst (passes mem = NULL). See __Runtime_dma_createio for
 * full details.
 *
 * @param tile_loc   Location (col,row) of the tile that owns this DMA channel/BD.
 * @param dma_desc   Pre-built DMA buffer descriptor for the transfer.
 * @param channel_id DMA channel index on the tile that runs this transfer.
 * @param bd_id      Buffer-descriptor ID holding this transfer's configuration.
 * @param direction  Transfer direction: XAIE_DMA_MM2S (tile memory -> stream) or
 *                   XAIE_DMA_S2MM (stream -> tile memory).
 * @return struct_io with mem = NULL.
 */
struct_io __Runtime_dma_createio_4(XAie_LocType tile_loc, XAie_DmaDesc dma_desc, int32_t channel_id, int32_t bd_id,
                                   XAie_DmaDirection direction) {
    return __Runtime_dma_createio(tile_loc, dma_desc, channel_id, bd_id, direction, NULL);
}

/**
 * Enable out-of-order BD execution on a DMA channel.
 * When enabled, the DMA engine selects BDs based on the out_of_order_bd_id
 * field in incoming packet headers, bypassing normal sequential BD chaining.
 * Used on shim S2MM channels for gathering output from multiple core tiles.
 */
void __Runtime_dma_channel_enable_ooo(XAie_DevInst *dev, XAie_LocType tile, int32_t channel, XAie_DmaDirection dir) {
    XAie_DmaChannelDesc DmaChannelDescInst;
    XAie_DmaChannelDescInit(dev, &DmaChannelDescInst, tile);
    XAie_DmaChannelEnOutofOrder(&DmaChannelDescInst, XAIE_ENABLE);
    AieRC rc = XAie_DmaWriteChannel(dev, &DmaChannelDescInst, tile, (uint8_t)channel, dir);
    const char *dir_str = (dir == DMA_MM2S) ? "MM2S" : "S2MM";
    AIEHLC_LOG(printf("[aie_runtime] channel_enable_ooo tile(%u,%u) ch=%d dir=%s rc=%d\n", (unsigned)tile.Col,
                      (unsigned)tile.Row, channel, dir_str, (int)rc));
}

/**
 * Start an I/O operation: enqueue the DMA transfer described by @p io onto its
 * channel's start queue (triggers the DMA) and return an event handle to wait on.
 * Also back-fills the BD-tracking table with this transfer's direction/channel,
 * following next_bd chains from the starting BD.
 * Reference: aieml_perf.cc lines 173-174 (XAie_MoveDataExternal2Aie)
 *
 * @param dev    Device instance the transfer runs on.
 * @param io     I/O handle from __Runtime_dma_createio(_4): carries tile_loc,
 *               channel_id, bd_id (the BD actually queued), and direction.
 * @param bd_id  Unused; the queued BD is taken from io.bd_id. Kept for a stable
 *               call signature with the emitted host code.
 * @param repeat Repeat count passed to XAie_DmaChannelSetStartQueue: how many
 *               times the channel re-runs this BD chain before going idle.
 * @return struct_ioevent wrapping @p io with a default 10000us wait timeout;
 *         pass it to __Runtime_wait_io/__Runtime_wait to block until completion.
 */
struct_ioevent __Runtime_startio(XAie_DevInst *dev, struct_io io, int32_t bd_id, int32_t repeat) {
    struct_ioevent evt;
    evt.io = io;
    evt.timeout_us = 10000;

    uint8_t sio_tile_type = XAie_GetTileTypefromLoc(dev, io.tile_loc);
    const char *dir_str = (io.direction == DMA_MM2S) ? "MM2S" : "S2MM";
    const char *note = (__bd_is_shim(sio_tile_type) && io.direction == DMA_MM2S) ? " [shim→tile input: WriteGM done]"
                       : (__bd_is_shim(sio_tile_type) && io.direction == DMA_S2MM)
                           ? " [shim←tile output: ReadGM on wait_io]"
                           : "";
    AIEHLC_LOG(printf("[aie_runtime] startio tile(%u,%u) ch=%u dir=%s bd=%u repeat=%d%s\n", (unsigned)io.tile_loc.Col,
                      (unsigned)io.tile_loc.Row, (unsigned)io.channel_id, dir_str, (unsigned)io.bd_id, (int)repeat,
                      note));

    /* Update BD tracking entries with direction/channel from this startio.
     * Match by tile (col,row) and bd_id, then follow next_bd chains. */
    {
        int8_t target_bd = (int8_t)io.bd_id;
        int visited[BD_TRACK_MAX] = {0};
        while (target_bd >= 0) {
            int found = 0;
            for (int _t = 0; _t < g_bd_track_count; _t++) {
                BdTrackEntry *e = &g_bd_track[_t];
                if (e->col == io.tile_loc.Col && e->row == io.tile_loc.Row && e->bd_id == target_bd) {
                    e->direction = (int8_t)io.direction;
                    e->channel_id = (int8_t)io.channel_id;
                    if (!visited[_t]) {
                        visited[_t] = 1;
                        target_bd = e->next_bd;
                        found = 1;
                    }
                    break;
                }
            }
            if (!found)
                break;
            /* Avoid infinite loop on circular chains (ping-pong) */
        }
    }

    AieRC rc =
        XAie_DmaChannelSetStartQueue(dev, io.tile_loc, io.channel_id, io.direction, io.bd_id, repeat, XAIE_DISABLE);
    if (rc != XAIE_OK) {
        printf("[aie_runtime] startio FAILED rc=%d tile(%u,%u) ch=%u dir=%s bd=%u\n", (int)rc,
               (unsigned)io.tile_loc.Col, (unsigned)io.tile_loc.Row, (unsigned)io.channel_id, dir_str,
               (unsigned)io.bd_id);
    }

    return evt;
}

/**
 * Syntax sugar for OOO startio — wraps __Runtime_startio with all OOO params.
 * Enables OOO on the channel, then starts with given repeat count.
 */
struct_ioevent _Runtime_startio_ooo(XAie_DevInst *dev, struct_io io, int32_t bd_id, int32_t repeat) {
    __Runtime_dma_channel_enable_ooo(dev, io.tile_loc, io.channel_id, io.direction);
    return __Runtime_startio(dev, io, bd_id, repeat);
}

/**
 * Load kernel ELF into tiles
 * Reference: aieml_perf.cc lines 123-125
 */
/* [skip-bss] Load only PT_LOAD segments that carry file bytes (.text/.data),
 * skipping pure-.bss segments (p_filesz==0). For a DMA-fed kernel the .bss holds
 * the ping-pong window buffers, which are DMA-filled before the core reads them,
 * so the driver's default XAie_LoadElfMem() -> XAIE_LOAD_ELF_ALL block-write of
 * zeros across the whole .bss on every tile is pure kernel-load overhead. Opt-in
 * (see __Runtime_skip_bss_enabled) so it is easy to A/B.
 * WARNING: only safe when no global is read before being written. If a kernel
 * relies on zero-initialized .bss globals, leave the skip disabled. */
#if !defined(__AIESIM__)
static AieRC __Runtime_load_elf_mem_skip_bss(XAie_DevInst *dev, XAie_LocType loc, const unsigned char *elfmem) {
    const Elf32_Ehdr *ehdr = (const Elf32_Ehdr *)elfmem;
    u32 skipped = 0, loaded = 0;
    for (u32 ph = 0; ph < ehdr->e_phnum; ph++) {
        const Elf32_Phdr *phdr = (const Elf32_Phdr *)(elfmem + ehdr->e_phoff + (u64)ph * sizeof(Elf32_Phdr));
        if (phdr->p_type != (u32)PT_LOAD)
            continue;
        if (phdr->p_filesz == 0U) { /* pure .bss segment -> DMA-filled, skip zero-init */
            skipped++;
            continue;
        }
        AieRC rc = XAie_LoadElfSection(dev, loc, elfmem + phdr->p_offset, phdr);
        if (rc != XAIE_OK)
            return rc;
        loaded++;
    }
    AIEHLC_LOG(printf("[aie_runtime] skip-bss load: %u segs loaded, %u bss segs skipped\n", loaded, skipped));
    return XAIE_OK;
}
#endif

/* Returns 1 to skip .bss zero-init on kernel load, else 0.
 * Compile-time -DAIEHLC_SKIP_BSS_DEFAULT=1 wins (needed on baremetal newlib where
 * getenv() always returns NULL); otherwise fall back to the AIEHLC_SKIP_BSS env
 * var so a Linux host can A/B without a rebuild. */
#if !defined(__AIESIM__)
static int __Runtime_skip_bss_enabled(void) {
#if defined(AIEHLC_SKIP_BSS_DEFAULT)
    return (AIEHLC_SKIP_BSS_DEFAULT);
#else
    static int s_skip_bss = -1;
    if (s_skip_bss < 0) {
        const char *e = getenv("AIEHLC_SKIP_BSS");
        s_skip_bss = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return s_skip_bss;
#endif
}
#endif

/* Load an ELF into a core tile, honoring the opt-in skip-.bss fast path (HW only). */
static void __Runtime_load_kernel_elf(XAie_DevInst *dev, XAie_LocType loc, unsigned char *elfmem) {
#if !defined(__AIESIM__)
    if (__Runtime_skip_bss_enabled()) {
        __Runtime_load_elf_mem_skip_bss(dev, loc, elfmem);
        return;
    }
#endif
    XAie_LoadElfMem(dev, loc, elfmem);
}

struct_kernel_group __Runtime_load_kernel_group(XAie_DevInst *dev, XAie_LocType *tiles, int32_t num_tiles,
                                                unsigned char **elf_buffers) {
    struct_kernel_group kg;
    kg.tiles = tiles;
    kg.num_tiles = num_tiles;
    kg.elf_buffers = elf_buffers;

    if (elf_buffers) {
        for (int i = 0; i < num_tiles; i++) {
            if (!__Runtime_is_aie_core_tile(tiles[i]))
                continue;
            // uncomment if sim hangs while executing kernels
            // #ifdef __AIESIM__
            //             XAie_CoreDisable(dev, tiles[i]);
            //             XAie_CoreReset(dev, tiles[i]);
            //             XAie_LoadElfMem(dev, tiles[i], elf_buffers[i]);
            //             XAie_CoreUnreset(dev, tiles[i]);
            // #else
            XAie_CoreDisable(dev, tiles[i]);
            XAie_CoreReset(dev, tiles[i]);
            __Runtime_load_kernel_elf(dev, tiles[i], elf_buffers[i]);
            XAie_CoreUnreset(dev, tiles[i]);
            // #endif
        }
    }

    return kg;
}

/** Array-based variant: copies caller-provided tile array into the static buffer
 *  and loads the kernel ELF into each core tile. Supports up to MAX_KERNEL_TILES. */
struct_kernel_group __Runtime_load_kernel_group_nt(XAie_DevInst *dev, XAie_LocType *tiles, int n) {
    AIEHLC_LOG(printf("[aie_runtime] load_kernel_group_nt n=%d\n", n););
    if (n > MAX_KERNEL_TILES) {
        printf("[aie_runtime] WARNING: tile count %d exceeds MAX_KERNEL_TILES %d, clamping\n", n, MAX_KERNEL_TILES);
        n = MAX_KERNEL_TILES;
    }
    for (int i = 0; i < n; i++) {
        s_kernel_tiles[i] = tiles[i];
        AIEHLC_LOG(printf("[aie_runtime]   tile[%d] = (%u,%u)\n", i, (unsigned)tiles[i].Col, (unsigned)tiles[i].Row););
    }
    for (int i = 0; i < n; i++) {
        if (!__Runtime_is_aie_core_tile(s_kernel_tiles[i]))
            continue;
        AIEHLC_LOG(printf("[aie_runtime] loading kernel ELF into tile (%u,%u)\n", (unsigned)s_kernel_tiles[i].Col,
                          (unsigned)s_kernel_tiles[i].Row));
        // uncomment if sim hangs while executing kernels
        // #ifdef __AIESIM__
        //   XAie_CoreDisable(dev, s_kernel_tiles[i]);
        //   XAie_CoreReset(dev, s_kernel_tiles[i]);
        //   XAie_LoadElfMem(dev, s_kernel_tiles[i], s_active_kernel_elf);
        //   XAie_CoreUnreset(dev, s_kernel_tiles[i]);
        // #else
        XAie_CoreDisable(dev, s_kernel_tiles[i]);
        XAie_CoreReset(dev, s_kernel_tiles[i]);
        __Runtime_load_kernel_elf(dev, s_kernel_tiles[i], s_active_kernel_elf);
        XAie_CoreUnreset(dev, s_kernel_tiles[i]);
        // #endif
    }
    struct_kernel_group kg;
    kg.tiles = s_kernel_tiles;
    kg.num_tiles = n;
    kg.elf_buffers = NULL;
    return kg;
}

struct_kernel_group __Runtime_load_kernel_group_4t(XAie_DevInst *dev, XAie_LocType t0, XAie_LocType t1, XAie_LocType t2,
                                                   XAie_LocType t3, int n) {
    AIEHLC_LOG(printf("[aie_runtime] load_kernel_group_4t n=%d tiles=(%u,%u)(%u,%u)(%u,%u)(%u,%u)\n", n,
                      (unsigned)t0.Row, (unsigned)t0.Col, (unsigned)t1.Row, (unsigned)t1.Col, (unsigned)t2.Row,
                      (unsigned)t2.Col, (unsigned)t3.Row, (unsigned)t3.Col));
    XAie_LocType arr[] = {t0, t1, t2, t3};
    return __Runtime_load_kernel_group_nt(dev, arr, n);
}

struct_kernel_group __Runtime_load_kernel_group_8t(XAie_DevInst *dev, XAie_LocType t0, XAie_LocType t1, XAie_LocType t2,
                                                   XAie_LocType t3, XAie_LocType t4, XAie_LocType t5, XAie_LocType t6,
                                                   XAie_LocType t7, int n) {
    XAie_LocType arr[] = {t0, t1, t2, t3, t4, t5, t6, t7};
    return __Runtime_load_kernel_group_nt(dev, arr, n);
}

struct_kernel_group __Runtime_load_kernel_group_16t(XAie_DevInst *dev, XAie_LocType t0, XAie_LocType t1,
                                                    XAie_LocType t2, XAie_LocType t3, XAie_LocType t4, XAie_LocType t5,
                                                    XAie_LocType t6, XAie_LocType t7, XAie_LocType t8, XAie_LocType t9,
                                                    XAie_LocType t10, XAie_LocType t11, XAie_LocType t12,
                                                    XAie_LocType t13, XAie_LocType t14, XAie_LocType t15, int n) {
    XAie_LocType arr[] = {t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15};
    return __Runtime_load_kernel_group_nt(dev, arr, n);
}

/**
 * Enable AIE cores using transaction batching.
 * Reference: aeg_runtime_api.cpp graph_api::run() lines 176-181
 */
void __Runtime_core_run(XAie_DevInst *dev, XAie_LocType *tiles, uint32_t num_tiles) {
    XAie_StartTransaction(dev, XAIE_TRANSACTION_ENABLE_AUTO_FLUSH);
    for (uint32_t i = 0; i < num_tiles; i++) {
        if (__Runtime_is_aie_core_tile(tiles[i])) {
            XAie_CoreEnable(dev, tiles[i]);
        } else {
            printf("[aie_runtime] WARNING: core_run skipping non-core tile (%u,%u)\n", (unsigned)tiles[i].Col,
                   (unsigned)tiles[i].Row);
        }
    }
    XAie_SubmitTransaction(dev, NULL);
    // printf("[aie_runtime] core_run submitted %u tiles  WAIT HEAR FIXME\n", (unsigned)num_tiles);
    // while(1);
}

/**
 * Launch kernel group (start cores)
 * Reference: aeg_runtime_api.cpp graph_api::run()
 */
struct_event __Runtime_launch_kernel_group(XAie_DevInst *dev, struct_kernel_group kg) {
    struct_event evt;
    evt.tiles = kg.tiles;
    evt.num_tiles = kg.num_tiles;
    evt.timeout_us = 100000;

    AIEHLC_LOG(printf("[aie_runtime] launch_kernel_group num_tiles=%u\n", (unsigned)kg.num_tiles););
    __Runtime_core_run(dev, kg.tiles, kg.num_tiles);

    return evt;
}

/**
 * Wait for kernel completion
 * Reference: aieml_perf.cc lines 189-202 (XAie_CoreWaitForDone loop)
 */
void __Runtime_wait_event(XAie_DevInst *dev, struct_event event) {
    uint8_t allDone = 0;

    AIEHLC_LOG(printf("[aie_runtime] wait_event num_tiles=%u tiles=(%u,%u)(%u,%u)(%u,%u)(%u,%u)\n",
                      (unsigned)event.num_tiles, (unsigned)event.tiles[0].Row, (unsigned)event.tiles[0].Col,
                      (unsigned)event.tiles[1].Row, (unsigned)event.tiles[1].Col, (unsigned)event.tiles[2].Row,
                      (unsigned)event.tiles[2].Col, (unsigned)event.tiles[3].Row, (unsigned)event.tiles[3].Col));

#ifdef __AIESIM__
    {
        const uint32_t WAIT_TIMEOUT_US = 1000;
        const uint32_t timeout_iters = 120 * 1000;
        const uint32_t PROGRESS_EVERY = 50;
        uint32_t iter = 0;
        do {
            allDone = 1;
            uint32_t pending_col = 0, pending_row = 0;
            for (uint32_t i = 0; i < event.num_tiles; i++) {
                if (!__Runtime_is_aie_core_tile(event.tiles[i]))
                    continue;
                AieRC RC = XAie_CoreWaitForDone(dev, event.tiles[i], WAIT_TIMEOUT_US);
                if (RC != XAIE_OK) {
                    allDone = 0;
                    pending_col = event.tiles[i].Col;
                    pending_row = event.tiles[i].Row;
                }
            }
            iter++;
            if (!allDone && (iter % PROGRESS_EVERY) == 0)
                AIEHLC_LOG(printf("[aie_runtime] wait_event WAITING iter=%u/%u still-busy core tile(%u,%u)\n", iter,
                                  timeout_iters, pending_col, pending_row));
        } while (!allDone && iter < timeout_iters);
        if (allDone)
            AIEHLC_LOG(printf("[aie_runtime] wait_event done after %u iter(s)\n", iter););
        else
            printf("[aie_runtime] wait_event TIMEOUT after %u iters - continuing to debug snapshot\n", iter);
    }
#else
    uint32_t timeout_iters = 100;
    uint32_t iter = 0;
    do {
        allDone = 1;
        for (uint32_t i = 0; i < event.num_tiles; i++) {
            if (!__Runtime_is_aie_core_tile(event.tiles[i])) {
                continue;
            }
            AieRC RC = XAie_CoreWaitForDone(dev, event.tiles[i], 0);
            if (RC != XAIE_OK) {
                allDone = 0;
            }
        }
        iter++;
    } while (!allDone && iter < timeout_iters);
    if (allDone)
        AIEHLC_LOG(printf("[aie_runtime] wait_event done\n"););
    else
        printf("[aie_runtime] wait_event TIMEOUT after %u iters - continuing to debug snapshot\n", iter);
#endif
}

/**
 * Wait for I/O completion (DMA wait)
 * Polls XAie_DmaGetPendingBdCount until the channel has zero pending BDs,
 * meaning both the start queue is empty and no BD is currently executing.
 * Reference: aeg_runtime_api.cpp waitDMAChannelTaskQueue / waitDMAChannelDone
 */
void __Runtime_wait_io(XAie_DevInst *dev, struct_ioevent io_event) {
    XAie_LocType tile = io_event.io.tile_loc;
    uint8_t channel = io_event.io.channel_id;
    XAie_DmaDirection dir = io_event.io.direction;

    if (AIEHLC_LOG_ENABLED())
        printf("[aie_runtime] wait_io tile(%u,%u) ch=%u dir=%d\n", (unsigned)tile.Col, (unsigned)tile.Row,
               (unsigned)channel, (int)dir);

    AIEHLC_LOG(printf("aie runtime col = %d, row = %d, channel = %d, dir = %d\n", tile.Col, tile.Row, channel, dir););
    /* Read raw DMA channel status for diagnostic */
    {
        u32 ch_status = 0;
        AieRC src = XAie_DmaGetChannelStatus(dev, tile, channel, dir, &ch_status);
        AIEHLC_LOG(printf("[aie_runtime] ch_status tile(%u,%u) ch=%u dir=%d raw=0x%08x rc=%d\n", (unsigned)tile.Col,
                          (unsigned)tile.Row, (unsigned)channel, (int)dir, (unsigned)ch_status, (int)src));
    }
#ifdef __AIESIM__
    {
        const uint32_t SIM_PER_CALL_US = 5000U;
        const uint32_t max_iters = 400;
        u8 numPendingBDs = 1;
        for (uint32_t iter = 0; iter < max_iters; iter++) {
            XAie_DmaWaitForDone(dev, tile, channel, dir, SIM_PER_CALL_US);
            AieRC prc = XAie_DmaGetPendingBdCount(dev, tile, channel, dir, &numPendingBDs);
            if (prc == XAIE_OK && numPendingBDs == 0)
                break;
        }
    }
#else
    const uint32_t poll_interval_us = 1000 * 1000;
    const uint32_t max_iters = 5;
    u8 numPendingBDs = 1;
    uint32_t iter = 0;
    while (numPendingBDs > 0) {
        AieRC rc = XAie_DmaGetPendingBdCount(dev, tile, channel, dir, &numPendingBDs);
        if (rc != XAIE_OK) {
            printf("[aie_runtime] wait_io ERROR: XAie_DmaGetPendingBdCount "
                   "failed rc=%d tile(%u,%u) ch=%u dir=%d\n",
                   (int)rc, (unsigned)tile.Col, (unsigned)tile.Row, (unsigned)channel, (int)dir);
            return;
        }
        if (numPendingBDs > 0) {
            if (++iter >= max_iters) {
                printf("[aie_runtime] wait_io TIMEOUT after %u ms "
                       "tile(%u,%u) ch=%u dir=%d pending=%u\n",
                       (unsigned)(max_iters * poll_interval_us / 1000), (unsigned)tile.Col, (unsigned)tile.Row,
                       (unsigned)channel, (int)dir, (unsigned)numPendingBDs);
                return;
            } else if (AIEHLC_LOG_ENABLED()) {
                printf("[aie_runtime] wait_io pending tile(%u,%u) ch=%u dir=%d pending=%u iter=%u\n",
                       (unsigned)tile.Col, (unsigned)tile.Row, (unsigned)channel, (int)dir, (unsigned)numPendingBDs,
                       (unsigned)iter);
            }
            usleep(poll_interval_us);
        }
    }
#endif

    if (AIEHLC_LOG_ENABLED())
        printf("[aie_runtime] wait_io done tile(%u,%u) ch=%u dir=%d\n", (unsigned)tile.Col, (unsigned)tile.Row,
               (unsigned)channel, (int)dir);

#ifdef __AIESIM__
    if (__bd_is_shim(XAie_GetTileTypefromLoc(dev, tile)) && dir == DMA_S2MM) {
        uint8_t tile_type = XAie_GetTileTypefromLoc(dev, tile);
        for (int bi = 0; bi < g_bd_track_count; bi++) {
            BdTrackEntry *e = &g_bd_track[bi];
            if (e->col != tile.Col || e->row != tile.Row)
                continue;
            if (e->tile_type != tile_type)
                continue;
            if (e->direction != 0)
                continue; /* direction 0 = S2MM */
            if (!e->buffer || e->dma_addr == 0)
                continue;
            ess_ReadGM(e->dma_addr, e->buffer, (uint64_t)e->len);
            AIEHLC_LOG(printf("[aie_runtime] wait_io readGM DevAddr=0x%lx → VAddr=%p len=%d\n",
                              (unsigned long)e->dma_addr, e->buffer, e->len));
        }
    }
#endif
}

/**
 * Move data from DDR to AIE tile
 * Reference: aieml_perf.cc lines 173-174
 */
void __Runtime_move_data_to_tile(XAie_RoutingInstance *routing, XAie_LocType shim_tile, XAie_LocType dest_tile,
                                 XAie_MemInst *mem, uint32_t size, uint32_t tile_offset) {
    // Sync memory for device access
    XAie_MemSyncForDev(mem);

    // Move data using routing API
    XAie_MoveDataExternal2Aie(routing, shim_tile, mem, size, tile_offset, dest_tile);
}

/**
 * Move data from AIE tile to DDR
 * Reference: aieml_perf.cc lines 206-208
 */
void __Runtime_move_data_from_tile(XAie_RoutingInstance *routing, XAie_LocType src_tile, XAie_LocType shim_tile,
                                   XAie_MemInst *mem, uint32_t tile_offset, uint32_t size) {
    // Sync memory for CPU access
    XAie_MemSyncForCPU(mem);

    // Move data using routing API
    XAie_MoveDataAie2External(routing, src_tile, tile_offset, size, mem, shim_tile);
}
