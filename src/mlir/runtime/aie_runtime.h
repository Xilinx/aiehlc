/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef AIE_RUNTIME_H
#define AIE_RUNTIME_H

#include "xaiengine.h"
#ifndef XAIE_ROUTING_H
#ifdef __cplusplus
extern "C" {
#endif
#include <aie_codegen_inc/xaie_routing.h>
#ifdef __cplusplus
}
#endif
#endif
#include <stdio.h>
#include <string.h>
// #include <stdint.h>

// Runtime structures wrapping XAie types
typedef struct {
    XAie_DmaDesc desc;
    XAie_LocType tile_loc;
    uint8_t channel_id;
    uint8_t bd_id;
    XAie_DmaDirection direction;
    XAie_MemInst *mem;
} struct_io;

typedef struct {
    struct_io io;
    uint32_t timeout_us;
} struct_ioevent;

typedef struct {
    XAie_LocType *tiles;
    uint32_t num_tiles;
    unsigned char **elf_buffers;
} struct_kernel_group;

typedef struct {
    XAie_LocType *tiles;
    uint32_t num_tiles;
    uint32_t timeout_us;
} struct_event;

// Global routing instance (kept for legacy path)
extern XAie_RoutingInstance *g_RoutingInst;

extern XAie_DevInst *g_DevInst;
XAie_DevInst *getOrCreateDeviceInstance(void);

// Debug level: bits 0-3 = verbosity (0-15), bits 4-31 = feature flags
// Use AIE_DEBUG_LEVEL(v) to extract verbosity, AIE_DEBUG_HAS_FLAG(v, flag) to test flags
#define AIE_DEBUG_LEVEL(v) ((v) & 0xF)
#define AIE_DEBUG_FLAG_DISABLE_MULTID_DIM_DMA (1 << 4)
#define AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN (1 << 5)
// When set, device init arms our own MM2S BD-finished perf counters (MEM
// module counters 0/1) across the whole partition via
// __Runtime_perfcnt_setup_mm2s_bd_finished_partition(); teardown reads them
// back. These are the same counters aiegdb.py "dma counter" reads (0x11020/24).
#define AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER (1 << 6)
#define AIE_DEBUG_FLAG_CORE_PERF_COUNTER (1 << 8)
// bit 7 reserved
// bit 8 reserved
/* When set, enable informational runtime log output (AIEHLC_LOG).
 * Usage: #pragma aie_debug_level(AIE_DEBUG_LOG) */
#define AIE_DEBUG_LOG (1 << 9)
#define AIE_DEBUG_HAS_FLAG(v, flag) (((v) & (flag)) != 0)
extern int g_runtime_debug_level;

/* Gate for informational runtime logs.
 * Enable with: #pragma aie_debug_level(AIE_DEBUG_LOG)
 * Errors/failures print unconditionally regardless of this flag.
 * Usage:  AIEHLC_LOG(printf("...", args)); */
#define AIEHLC_LOG_ENABLED() AIE_DEBUG_HAS_FLAG(g_runtime_debug_level, AIE_DEBUG_LOG)
/* clang-format off */
#define AIEHLC_LOG(stmt) do { if (AIEHLC_LOG_ENABLED()) { stmt; } } while (0)
/* clang-format on */

// Type aliases for emitted host code (uses "io", "event", etc. without "struct" prefix)
typedef struct_io io;
typedef struct_ioevent ioevent;
typedef struct_kernel_group kernel_group;
typedef struct_event event;

// ---------------------------------------------------------------------------
// PartitionTensor: data partitioning for multi-tile tensor distribution
// ---------------------------------------------------------------------------

#define PARTITION_MAX_DIMS 8

typedef struct {
    void *data;
    size_t elem_size;
    int ndim;
    int64_t original_shape[PARTITION_MAX_DIMS];
    int64_t partition_shape[PARTITION_MAX_DIMS];
    int partition_dim;
    int num_partitions;
    int hw_axis_owner; /* 0=row, 1=col */
    int replicate_on;  /* 0=row, 1=col, -1=none */
} PartitionTensor;

/* Pass a raw void* buffer through as a void* argument (used by ScheduleCanonicalize
 * DDR init chain lowering: bind_core_buffer emits L1 offsets as raw void* values). */
#define __runtime_buffer_arg(p) ((void *)(p))

/* Compute a byte offset into a DDR buffer (used by dfschedule.buffer_view lowering) */
static inline void *__runtime_buffer_offset(void *base, int64_t offset) { return (void *)((char *)base + offset); }

/* Plain malloc/free/memcpy for DDR buffers allocated by the host init chain.
 * These wrap the standard C library so host.cc does not need to include <stdlib.h>. */
void *__Runtime_malloc(size_t bytes);
void __Runtime_free(void *ptr);
void __Runtime_memcpy(void *dst, const void *src, size_t bytes);

/* 64-byte aligned allocation for DDR buffers requiring alignment. */
void *__Runtime_Alloc(size_t bytes);

/* Legacy stub (no-op in VAddr mode; kept for backward compat) */
void __Runtime_free_all_allocs(void);

static inline void __Runtime_Partition_Print(void *data, size_t elem_size, int ndim,
        const int64_t *original_shape,
        const int64_t *partition_shape, int partition_dim,
        int num_partitions, int hw_axis_owner, int replicate_on) {
    if (AIE_DEBUG_LEVEL(g_runtime_debug_level) > 0) {
        size_t total_elems = 1;
        printf("[__Runtime_init_PartitionTensor] data=%p elem_size=%zu ndim=%d "
               "partition_dim=%d num_partitions=%d hw_axis_owner=%d replicate_on=%d\n",
               data, elem_size, ndim, partition_dim, num_partitions, hw_axis_owner, replicate_on);
        printf("  original_shape=[");
        for (int i = 0; i < ndim; i++) {
            printf("%s%lld", i ? "," : "", (long long)original_shape[i]);
            total_elems *= (size_t)original_shape[i];
        }
        printf("] partition_shape=[");
        for (int i = 0; i < ndim; i++)
            printf("%s%lld", i ? "," : "", (long long)partition_shape[i]);
        printf("]\n");
        if (data) {
            elem_size = 4;
            printf("  data[%zu elems, %zu bytes each]:", total_elems, elem_size);
            for (size_t i = 0; i < total_elems; i++) {
                if (i % 16 == 0)
                    printf("\n    [%4zu]", i);
                if (elem_size == 1)
                    printf(" %4d", (int)((int8_t *)data)[i]);
                else if (elem_size == 2)
                    printf(" %6d", (int)((int16_t *)data)[i]);
                else if (elem_size == 4)
                    printf(" %11d", ((int32_t *)data)[i]);
                else
                    printf(" 0x%llx", (unsigned long long)((uint64_t *)data)[i]);
            }
            printf("\n");
        }
    }
}

static inline PartitionTensor __Runtime_init_PartitionTensor(void *data, size_t elem_size, int ndim,
                                                             const int64_t *original_shape,
                                                             const int64_t *partition_shape, int partition_dim,
                                                             int num_partitions, int hw_axis_owner, int replicate_on) {
    PartitionTensor pt;
    pt.data = data;
    pt.elem_size = elem_size;
    pt.ndim = ndim;
    for (int i = 0; i < ndim && i < PARTITION_MAX_DIMS; i++) {
        pt.original_shape[i] = original_shape[i];
        pt.partition_shape[i] = partition_shape[i];
    }
    pt.partition_dim = partition_dim;
    pt.num_partitions = num_partitions;
    pt.hw_axis_owner = hw_axis_owner;
    pt.replicate_on = replicate_on;
    __Runtime_Partition_Print(data, elem_size, ndim, original_shape, partition_shape, partition_dim, num_partitions, hw_axis_owner, replicate_on);
    return pt;
}

static inline void *__Runtime_get_partition_slice(PartitionTensor *pt, int partition_idx) {
    if (partition_idx < 0 || partition_idx >= pt->num_partitions)
        return NULL;
    size_t slice_size = pt->elem_size;
    for (int i = 0; i < pt->ndim; i++) {
        slice_size *= pt->partition_shape[i];
    }
    return (void *)((char *)pt->data + partition_idx * slice_size);
}

static inline PartitionTensor __Runtime_extract_slice_contiguous_2d(PartitionTensor src, int off0, int off1, int size0,
                                                                    int size1) {
    PartitionTensor result;
    result.elem_size = src.elem_size;
    result.ndim = 2;
    result.partition_dim = -1;
    result.num_partitions = 1;
    result.hw_axis_owner = src.hw_axis_owner;
    result.replicate_on = src.replicate_on;
    result.original_shape[0] = size0;
    result.original_shape[1] = size1;
    result.partition_shape[0] = size0;
    result.partition_shape[1] = size1;
    size_t byte_offset = (off0 * src.original_shape[1] + off1) * src.elem_size;
    result.data = (void *)((char *)src.data + byte_offset);
    return result;
}

static inline PartitionTensor __Runtime_extract_slice_strided_2d(XAie_DevInst *dev_inst, PartitionTensor src, int off0,
                                                                 int off1, int size0, int size1) {
    PartitionTensor result;
    result.elem_size = src.elem_size;
    result.ndim = 2;
    result.partition_dim = -1;
    result.num_partitions = 1;
    result.hw_axis_owner = src.hw_axis_owner;
    result.replicate_on = src.replicate_on;
    result.original_shape[0] = size0;
    result.original_shape[1] = size1;
    result.partition_shape[0] = size0;
    result.partition_shape[1] = size1;
    size_t dst_size = (size_t)size0 * size1 * src.elem_size;
    void *dst = (void *)malloc(dst_size);
    result.data = dst;
    if (!dst)
        return result;
    char *d = (char *)dst;
    char *s = (char *)src.data;
    int elem_size = src.elem_size;
    int src_dim1 = src.original_shape[1];
    for (int i = 0; i < size0; i++) {
        int src_idx = ((off0 + i) * src_dim1 + off1) * elem_size;
        int dst_idx = (i * size1) * elem_size;
        memcpy(d + dst_idx, s + src_idx, size1 * elem_size);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Runtime API declarations
// Reference: aieml_perf.cc for XAie API usage patterns
// ---------------------------------------------------------------------------

// Routing init (takes explicit dev pointer)
void __Runtime_routing_init(XAie_DevInst *dev);
// Device teardown (takes explicit dev pointer)
AieRC __Runtime_device_teardown(XAie_DevInst *dev);

// ---------------------------------------------------------------------------
// Explicit init/teardown: heap-allocates XAie_DevInst, returns pointer.
// Caller owns the returned pointer and must call __Runtime_explicit_teardown().
// ---------------------------------------------------------------------------
XAie_DevInst *__Runtime_explicit_init(void);
XAie_DevInst *__Runtime_explicit_init_partition(int startCol, int numCols);
void __Runtime_explicit_teardown(XAie_DevInst *dev);

// ---------------------------------------------------------------------------
// Partition registry: init-once, get-existing, teardown-all.
// Used by the multi-kernel/multi-partition programming model.
// Each aieMesh carries a meshId; the runtime tracks which meshIds have been
// initialized. If already initialized, skip re-init and return the existing
// XAie_DevInst*. aieArray::synchronize() calls __Runtime_teardown_all().
// ---------------------------------------------------------------------------
int __Runtime_partition_is_initialized(int meshId);
XAie_DevInst *__Runtime_get_partition_dev(int meshId);
void __Runtime_register_partition(int meshId, XAie_DevInst *dev);
void __Runtime_teardown_all(void);

// Convenience: check-init-register in one call.
// If meshId is already initialized, returns the existing XAie_DevInst*.
// Otherwise calls __Runtime_explicit_init_partition() and registers it.
XAie_DevInst *__Runtime_init_mesh_partition(int meshId, int startCol, int numCols);

// DMA and data movement
XAie_DmaDesc __Runtime_dma_bd_config(XAie_DevInst *dev, XAie_LocType tile, void *buffer, int32_t bd_id, int32_t len,
                                     int32_t next_bd, int32_t enable_packet, int32_t packet_id, int32_t acquire_lock_id,
                                     int32_t acquire_lock_val, int32_t release_lock_id, int32_t release_lock_val,
                                     int32_t out_of_order_bd_id);

// Multi-dimensional BD addressing for DMA-driven transpose/reshape.
// Uses XAie_DmaSetMultiDimAddr with stride/wrap descriptors instead of
// simple XAie_DmaSetAddrLen. num_dims controls how many dim pairs are active
// (max 4). Unused dims should have stride=0, wrap=0.
XAie_DmaDesc __Runtime_dma_bd_config_multidim(XAie_DevInst *dev, XAie_LocType tile, void *buffer, int32_t bd_id,
                                              int32_t len, int32_t next_bd, int32_t enable_packet, int32_t packet_id,
                                              int32_t acquire_lock_id, int32_t acquire_lock_val,
                                              int32_t release_lock_id, int32_t release_lock_val,
                                              int32_t out_of_order_bd_id, int32_t num_dims, int32_t dim_stride0,
                                              int32_t dim_wrap0, int32_t dim_stride1, int32_t dim_wrap1,
                                              int32_t dim_stride2, int32_t dim_wrap2, int32_t dim_stride3,
                                              int32_t dim_wrap3);

// Multi-dimensional BD with OOO iteration support.
// Configures D0-D2 address dimensions plus a separate iteration dimension
// for out-of-order packet reception where the BD re-executes iter_wrap times
// with address advancing by iter_step_size bytes between each OOO packet.
XAie_DmaDesc __Runtime_dma_bd_config_multidim_ooo(XAie_DevInst *dev, XAie_LocType tile, void *buffer, int32_t bd_id,
                                                  int32_t len, int32_t next_bd, int32_t enable_packet,
                                                  int32_t packet_id, int32_t acquire_lock_id, int32_t acquire_lock_val,
                                                  int32_t release_lock_id, int32_t release_lock_val,
                                                  int32_t out_of_order_bd_id, int32_t num_dims, int32_t dim_stride0,
                                                  int32_t dim_wrap0, int32_t dim_stride1, int32_t dim_wrap1,
                                                  int32_t dim_stride2, int32_t dim_wrap2, int32_t iter_step_size,
                                                  int32_t iter_wrap);

// Enable out-of-order BD execution on a DMA channel.
// When enabled, the DMA engine selects BDs based on the out_of_order_bd_id
// field in incoming packet headers, bypassing normal sequential BD chaining.
// Used on shim S2MM channels for gathering output data from multiple core tiles.
void __Runtime_dma_channel_enable_ooo(XAie_DevInst *dev, XAie_LocType tile, int32_t channel, XAie_DmaDirection dir);

struct_io __Runtime_dma_createio(XAie_LocType tile_loc, XAie_DmaDesc dma_desc, int32_t channel_id, int32_t bd_id,
                                 XAie_DmaDirection direction, XAie_MemInst *mem);
/* 4-arg version for emitted host code (mem = NULL) */
struct_io __Runtime_dma_createio_4(XAie_LocType tile_loc, XAie_DmaDesc dma_desc, int32_t channel_id, int32_t bd_id,
                                   XAie_DmaDirection direction);

struct_ioevent __Runtime_startio(XAie_DevInst *dev, struct_io io, int32_t bd_id, int32_t repeat);
struct_ioevent _Runtime_startio_ooo(XAie_DevInst *dev, struct_io io, int32_t bd_id, int32_t repeat);

// Set the active kernel ELF binary for the next load_kernel_group call.
// In multi-kernel mode, host code calls this before each host_canonicalized_<name>().
void __Runtime_set_kernel_elf(unsigned char *elf_start);

// Kernel management (all take explicit XAie_DevInst*)
struct_kernel_group __Runtime_load_kernel_group(XAie_DevInst *dev, XAie_LocType *tiles, int32_t num_tiles,
                                                unsigned char **elf_buffers);
struct_kernel_group __Runtime_load_kernel_group_nt(XAie_DevInst *dev, XAie_LocType *tiles, int n);
struct_kernel_group __Runtime_load_kernel_group_4t(XAie_DevInst *dev, XAie_LocType t0, XAie_LocType t1, XAie_LocType t2,
                                                   XAie_LocType t3, int n);
struct_kernel_group __Runtime_load_kernel_group_8t(XAie_DevInst *dev, XAie_LocType t0, XAie_LocType t1, XAie_LocType t2,
                                                   XAie_LocType t3, XAie_LocType t4, XAie_LocType t5, XAie_LocType t6,
                                                   XAie_LocType t7, int n);
struct_kernel_group __Runtime_load_kernel_group_16t(XAie_DevInst *dev, XAie_LocType t0, XAie_LocType t1,
                                                    XAie_LocType t2, XAie_LocType t3, XAie_LocType t4, XAie_LocType t5,
                                                    XAie_LocType t6, XAie_LocType t7, XAie_LocType t8, XAie_LocType t9,
                                                    XAie_LocType t10, XAie_LocType t11, XAie_LocType t12,
                                                    XAie_LocType t13, XAie_LocType t14, XAie_LocType t15, int n);

struct_event __Runtime_launch_kernel_group(XAie_DevInst *dev, struct_kernel_group kg);

// Core enable (reference: aeg_runtime_api.cpp graph_api::run)
void __Runtime_core_run(XAie_DevInst *dev, XAie_LocType *tiles, uint32_t num_tiles);

// Synchronization
void __Runtime_wait_event(XAie_DevInst *dev, struct_event ev);
void __Runtime_wait_io(XAie_DevInst *dev, struct_ioevent io_ev);
/* C only: single name for emitted host. */
#ifndef __cplusplus
#define __Runtime_wait(dev, x)                                                                                         \
    _Generic((x), struct_event: __Runtime_wait_event, struct_ioevent: __Runtime_wait_io)(dev, x)
#else
/* C++ overloads so emitted host can call __Runtime_wait(dev, event_or_ioevent) */
inline void __Runtime_wait(XAie_DevInst *dev, struct_event ev) { __Runtime_wait_event(dev, ev); }
inline void __Runtime_wait(XAie_DevInst *dev, struct_ioevent ev) { __Runtime_wait_io(dev, ev); }

inline struct_kernel_group __Runtime_load_kernel_group_4t(XAie_LocType t0, XAie_LocType t1, XAie_LocType t2,
                                                          XAie_LocType t3, int n) {
    return __Runtime_load_kernel_group_4t(g_DevInst, t0, t1, t2, t3, n);
}
inline struct_kernel_group __Runtime_load_kernel_group_8t(XAie_LocType t0, XAie_LocType t1, XAie_LocType t2,
                                                          XAie_LocType t3, XAie_LocType t4, XAie_LocType t5,
                                                          XAie_LocType t6, XAie_LocType t7, int n) {
    return __Runtime_load_kernel_group_8t(g_DevInst, t0, t1, t2, t3, t4, t5, t6, t7, n);
}
inline struct_event __Runtime_launch_kernel_group(struct_kernel_group kg) {
    return __Runtime_launch_kernel_group(g_DevInst, kg);
}
inline struct_ioevent __Runtime_startio(struct_io io, int32_t repeat) {
    return __Runtime_startio(g_DevInst, io, io.bd_id, repeat);
}
inline void __Runtime_wait(struct_event ev) { __Runtime_wait_event(g_DevInst, ev); }
inline void __Runtime_wait(struct_ioevent ev) { __Runtime_wait_io(g_DevInst, ev); }
#endif

// Kernel log reader (reads log entries from core tile data memory)
void __Runtime_read_kernel_log(XAie_DevInst *dev, XAie_LocType tile);

// ---------------------------------------------------------------------------
// Performance counter APIs for core tile memory module
// ---------------------------------------------------------------------------

// Generic: configure perf counter on a core tile.
// Sets start_event = stop_event so the counter counts occurrences of that event.
// Uses memory module (XAIE_MEM_MOD) perf counter 0 by default.
AieRC __Runtime_perfcnt_setup(XAie_DevInst *dev, XAie_LocType tile, uint8_t counter_id, XAie_Events event);

// Read a perf counter value from a core tile memory module.
AieRC __Runtime_perfcnt_read(XAie_DevInst *dev, XAie_LocType tile, uint8_t counter_id, uint32_t *value);

// Read the free-running core timer of `tile` (XAIE_CORE_MOD) — the same clock
// domain as the core event trace. Lets the host anchor AIE cycles against the
// ARM host clock (XTime) so both land on one time axis. Portable wrapper only.
AieRC __Runtime_read_aie_timer(XAie_DevInst *dev, XAie_LocType tile, uint64_t *val);

// Set perf counters for MM2S channel 0 BD finished (counter 0) and
// MM2S channel 1 BD finished (counter 1) on a single core tile.
AieRC __Runtime_perfcnt_setup_mm2s_bd_finished(XAie_DevInst *dev, XAie_LocType tile);

// Set MM2S BD finished perf counters on all core tiles in a rectangular
// partition [start_col..end_col] x [start_row..end_row] (inclusive).
AieRC __Runtime_perfcnt_setup_mm2s_bd_finished_partition(XAie_DevInst *dev, uint8_t start_col, uint8_t end_col,
                                                         uint8_t start_row, uint8_t end_row);

// Read and print all MM2S BD finished perf counters across a partition.
void __Runtime_perfcnt_read_mm2s_bd_finished_partition(XAie_DevInst *dev, uint8_t start_col, uint8_t end_col,
                                                       uint8_t start_row, uint8_t end_row);

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

AieRC __Runtime_core_perf_setup(XAie_DevInst *dev, XAie_LocType tile);
AieRC __Runtime_core_perf_read(XAie_DevInst *dev, XAie_LocType tile, uint32_t *active, uint32_t *vec_instr,
                               uint32_t *stream_stall, uint32_t *lock_stall);
int __Runtime_core_perf_probe_valid(void);
void __Runtime_core_perf_read_probe(uint32_t *active, uint32_t *vec_instr, uint32_t *stream_stall,
                                    uint32_t *lock_stall);
void __Runtime_perfcnt_read_mm2s_probe(uint32_t *ch0, uint32_t *ch1);

// ---------------------------------------------------------------------------
// Core event trace: capture an ACTIVE/stall timeline from a core tile and drain
// it DOWN through the intervening core tiles into the same-column top MemTile's
// memory via a circuit-switched TRACE -> (SOUTH/NORTH hops) -> S2MM DMA path.
// The MemTile's larger memory (512 KB on AIE2PS) lets the trace run deep without
// stealing core data memory.
// ---------------------------------------------------------------------------

// DMA direction selector for the memory-module trace unit (see
// __Runtime_mem_trace_setup / the mem_dma_kind param of core_trace_setup).
enum { AIE_TRACE_DMA_NONE = 0, AIE_TRACE_DMA_S2MM = 1, AIE_TRACE_DMA_MM2S = 2 };

// Configure the core trace unit on `tile`: capture window ACTIVE_CORE..
// DISABLED_CORE, EVENT_TIME mode (delta-cycle timestamps), trace slots 0..3 =
// ACTIVE / LOCK_STALL / STREAM_STALL / MEMORY_STALL and slots 4..6 =
// PORT_IDLE_0 / PORT_RUNNING_0 / PORT_STALLED_0 (the stream-switch port-0 event
// group), then route the TRACE stream down to the top MemTile in the same column
// and land it via the MemTile's S2MM channel `s2mm_ch` into
// [buf_addr, buf_addr+buf_len) of MemTile memory.
//   strm_ch  physical stream channel (0..3) used for every SOUTH/NORTH hop from
//            the core down to the MemTile; caller must ensure it is free.
//   s2mm_ch  the MemTile's S2MM DMA channel the trace drains into.
//   buf_addr/buf_len are bytes into MemTile memory; buf_addr is the DMA-view
//            address (same value __Runtime_core_trace_read passes to
//            XAie_DataMemBlockRead for the MemTile loc).
// Call BEFORE enabling the core; the caller must reserve the MemTile buffer
// region and the strm_ch/s2mm_ch so they do not clash with data traffic.
//
// The PORT_*_0 events only fire once the core's stream-switch event port 0 is
// bound to a physical port; port_intf/port/port_num pick which one (default
// core MASTER port 0 = the core's outgoing stream). Pass XAIE_STRMSW_SLAVE /
// a different StrmSwPortType / index to watch another port.
//
// When a generated routing resource map is passed (resmap != NULL && count > 0),
// the trace route's stream channel, MemTile S2MM channel and packet id are
// re-selected to avoid the data-plane ports the map records for this column;
// strm_ch/s2mm_ch/bdnum are then treated as fallbacks. NULL/0 (raw/single-kernel
// flow) keeps the convention values the caller passed in.
//
// mem_dma_kind/mem_dma_ch additionally arm the tile's MEMORY-module trace unit
// (packet id 2) capturing a DMA channel's BD/stall/lock events; it rides the
// SAME TRACE->SOUTH->MemTile drain and lands in the SAME buffer as the core
// stream (pkt id 1), demuxed by packet id on read. AIE_TRACE_DMA_NONE disables
// it; the default is S2MM channel 0. See __Runtime_mem_trace_setup.
struct AieResourceEntry; /* generated in aie_resource_map.h; opaque here */
AieRC __Runtime_core_trace_setup(XAie_DevInst *dev, XAie_LocType tile, uint32_t buf_addr, uint32_t buf_len,
                                 uint8_t strm_ch, uint8_t s2mm_ch, uint8_t bdnum = 0,
                                 const struct AieResourceEntry *resmap = 0, int resmap_count = 0,
                                 XAie_StrmPortIntf port_intf = XAIE_STRMSW_SLAVE, StrmSwPortType port = SOUTH,
                                 uint8_t port_num = 0, int mem_dma_kind = AIE_TRACE_DMA_S2MM, uint8_t mem_dma_ch = 0);

// Arm the compute tile's MEMORY-module trace unit and merge its packet stream
// (packet id 2) onto the core trace's shared TRACE->SOUTH->MemTile drain.
//   dma_kind  AIE_TRACE_DMA_S2MM / _MM2S selects which DMA direction to trace
//             (AIE_TRACE_DMA_NONE is a no-op that returns XAIE_OK).
//   dma_ch    DMA channel index (0 or 1) whose BD/stall/lock events are traced.
//   arbiter   the packet-switch arbiter the core stream already claimed on this
//             tile's SOUTH master (both streams share ONE arbiter).
//   msel      the MASTER-select line for THIS (mem) stream; must differ from the
//             core stream's msel so the shared SOUTH master's MSelEn bitmask can
//             round-robin between the two packet ids.
// Programs the MEM_MOD trace unit (8 slots: DMA start/finish/stall/lock),
// installs packet id 2, and enables the MEM_TRACE stream-switch slave port
// (TRACE port index 1; TRACE:0 is the core AIE_TRACE port). The caller
// (__Runtime_core_trace_setup) must have already enabled the SOUTH master with
// an MSelEn covering both msels. Call BEFORE enabling the core.
AieRC __Runtime_mem_trace_setup(XAie_DevInst *dev, XAie_LocType tile, int dma_kind, uint8_t dma_ch, uint8_t arbiter,
                                uint8_t msel);

// Read raw trace words back from the MemTile buffer. Pass the MemTile loc (the
// same-column top MemTile, row XAIE_AIE_TILE_ROW_START-1) and buf_addr used in
// setup. Call AFTER the core has disabled (XAie_CoreWaitForDone +
// XAie_CoreDisable) so the trace is flushed. Loc-generic: XAie_DataMemBlockRead
// dispatches on tile type, so a MemTile loc reads MemTile memory.
AieRC __Runtime_core_trace_read(XAie_DevInst *dev, XAie_LocType tile, uint32_t buf_addr, uint32_t *dst,
                                uint32_t len_words);

// ---------------------------------------------------------------------------
// Trace profile: capture decoded intervals instead of only printing them.
// ---------------------------------------------------------------------------
// One coalesced trace interval: cycles [start_cycle, end_cycle] all carried the
// identical 8-bit event mask. Cycles are raw AIE core-timer cycles (the same
// domain as the Start-frame timer); cycle->us correlation is done off-device
// from the [TIMESYNC] anchors.
typedef struct {
    uint8_t col, row; // tile that produced it (set by attach)
    uint32_t mask;    // 8-bit event mask (ACTIVE/*_STALL/...)
    uint64_t start_cycle, end_cycle;
    const char *const *names; // 8-slot name table of the producing stream
                              // (core vs mem); set by the decoder on flush so
                              // the dump labels each pkt-id stream correctly.
} AieTraceInterval;

// One host<->AIE anchor pair for a tile: the tile's free-running AIE core-timer
// value sampled before (aie0) and after (aie1) the run loop. The matching host
// counts are shared across tiles (single host clock) and live in host0/host1.
typedef struct {
    uint8_t col, row;
    uint64_t aie0, aie1;
} AieTraceAnchor;

// One host-side phase event: the host clock count (XTime) at a named boundary
// of iteration `iter` (e.g. "dma_in_done", "run", "wait_done"). `phase` points
// to a string literal, so it is stored by pointer (no copy).
typedef struct {
    int iter;
    const char *phase;
    uint64_t host;
} AieTraceHostEvt;

#define AIE_TRACE_PROFILE_CAP 4096u
// Per-stream reservation: when the shared buffer holds several demuxed streams
// (core pkt-1 + mem pkt-2), each stream still not yet decoded is guaranteed at
// least this many interval slots so a chatty earlier stream (the core port
// toggles) cannot starve a later one (the mem DMA events). See the reserve
// bookkeeping in __Runtime_core_trace_decode / __core_trace_flush.
#define AIE_TRACE_PROFILE_MIN_PER_STREAM 1024u
#define AIE_TRACE_ANCHOR_CAP 8u
#define AIE_TRACE_EVENT_CAP 64u
// Fixed-capacity, no-malloc collector (baremetal-safe). One container for a run:
// the decoded AIE core-trace intervals AND the host<->AIE time-sync record
// (clock, anchors, phase events). __core_trace_decode appends intervals; the
// host-side helpers below record the rest; __..._dump emits everything as one
// [TIMESYNC] block so external tools parse a single artifact.
typedef struct {
    AieTraceInterval iv[AIE_TRACE_PROFILE_CAP];
    uint32_t count;           // valid intervals in iv[]
    uint32_t dropped;         // intervals discarded past CAP
    uint32_t reserve;         // slots held back for not-yet-decoded streams
                              // (set per stream by __Runtime_core_trace_decode)
    uint8_t cur_col, cur_row; // tag applied to appended intervals
    int attached;             // nonzero after attach()
    // --- host<->AIE time-sync ---
    uint64_t cps;          // host COUNTS_PER_SECOND (0 = unset)
    uint64_t host0, host1; // host anchor counts (midpoint), 0 = unset
    AieTraceAnchor anchor[AIE_TRACE_ANCHOR_CAP];
    uint32_t nanchor; // valid entries in anchor[]
    AieTraceHostEvt evt[AIE_TRACE_EVENT_CAP];
    uint32_t nevt;                // valid entries in evt[]
    uint32_t evt_dropped;         // host events discarded past CAP
    uint8_t words_col, words_row; // tile whose raw-trace word count is recorded
    uint32_t nwords;              // raw trace words read (0 = unset)
} AieTraceProfile;

// Zero the profile (no allocation).
void __Runtime_aie_trace_profile_init(AieTraceProfile *p);

// Set the tile the next decode's intervals are tagged with.
void __Runtime_aie_trace_profile_attach(AieTraceProfile *p, XAie_LocType tile);

// Record the host clock frequency (COUNTS_PER_SECOND) for the time-sync block.
void __Runtime_aie_trace_profile_set_clock(AieTraceProfile *p, uint64_t cps);

// Record one anchor sample: which=0 (before loop) or 1 (after loop), the tile's
// AIE core-timer value `aie`, and the shared host count `host`. Finds/creates
// the per-tile anchor slot and also stores host into host0 (which=0) / host1.
void __Runtime_aie_trace_profile_anchor(AieTraceProfile *p, int which, XAie_LocType tile, uint64_t aie, uint64_t host);

// Append one host phase event (iter, phase name, host count). Bounded; silently
// counts drops past AIE_TRACE_EVENT_CAP into evt_dropped.
void __Runtime_aie_trace_profile_event(AieTraceProfile *p, int iter, const char *phase, uint64_t host);

// Record how many raw trace words were read for `tile` (count only; the words
// themselves are already decoded into intervals and are not stored).
void __Runtime_aie_trace_profile_set_trace_words(AieTraceProfile *p, XAie_LocType tile, uint32_t nwords);

// Emit the whole run as one [TIMESYNC] block: an interval-count summary, then
// (when present) cps / anchor0 / anchor1 / hostevt / aiehz / trace-word count,
// then one "[TIMESYNC] trace tile=c,r <start> -- <end> EVENT (N cyc)" line per
// captured interval. Host lines are skipped when their fields are unset, so a
// decode-only profile dumps just the interval lines.
void __Runtime_aie_trace_profile_dump(AieTraceProfile *p);

// Decode+print the ACTIVE/*_STALL timeline from raw trace words (host-side).
// If prof is non-NULL, each coalesced interval is also appended into it.
void __Runtime_core_trace_decode(const uint32_t *buf, uint32_t nwords, AieTraceProfile *prof);

// ---------------------------------------------------------------------------
// Declarative core-trace session (auto-injected by #pragma aie_trace).
// ---------------------------------------------------------------------------
// A thin session layer over __Runtime_core_trace_setup/read/decode that hides
// the MemTile drain-resource bookkeeping so the compiler-injected host code is
// just two opaque calls. The fixed-reserved MemTile convention (S2MM channel,
// BD parity, high buffer offset) lives entirely inside these two functions.
//
// Arm the core trace unit on the compute tile at mesh/partition-relative
// (col,row) BEFORE the core runs. Reserves this column's trace drain resources
// by a fixed convention, routes the TRACE stream DOWN into the same-column top
// MemTile's S2MM DMA, and records the tile in a small static registry (cap
// AIE_TRACE_SESSION_CAP). Multiple tiles in the SAME column each take a distinct
// strm_ch/s2mm_ch/BD slot. Silently caps/warns past the registry or per-column
// slot limits. Idempotent-safe: a repeated (col,row) is ignored.
void __Runtime_core_trace_begin(XAie_DevInst *dev, uint8_t col, uint8_t row);

// Same as __Runtime_core_trace_begin but PINS the physical stream channel the
// core->MemTile trace route rides (every hop; 0..3). Pass AIE_TRACE_STRM_CH_AUTO
// for the default (= per-column slot, what the 3-arg form uses). Pin an explicit
// channel when a data-plane DMA shares this tile's SOUTH egress: the trace route
// is programmed directly, outside the routing engine's resource manager, so an
// auto strm_ch that collides with a data flow on the same SOUTH channel deadlocks
// that DMA (e.g. example/perf/aieml_perf.cc pins ch 1 to clear its output DMA).
// Independent of the MemTile S2MM channel/BD (still slot-derived).
#define AIE_TRACE_STRM_CH_AUTO 0xFFu
void __Runtime_core_trace_begin_ch(XAie_DevInst *dev, uint8_t col, uint8_t row, uint8_t strm_ch);

// Same as __Runtime_core_trace_begin but additionally selects which tile DMA the
// MEMORY-module trace unit watches (see __Runtime_core_trace_setup /
// __Runtime_mem_trace_setup). mem_dma_kind is AIE_TRACE_DMA_S2MM/_MM2S (or
// _NONE to disable), mem_dma_ch is the DMA channel index (0/1). The physical
// stream channel is AUTO (per-column slot). Emitted by CoreTraceInsertPass for
// `#pragma aie_trace((col,row),(STREAM,...))` / `(PARAMETER,...)`; the default
// (no second tuple) is still S2MM ch0, so plain __Runtime_core_trace_begin is
// equivalent to __Runtime_core_trace_begin_dma(dev,col,row,AIE_TRACE_DMA_S2MM,0).
void __Runtime_core_trace_begin_dma(XAie_DevInst *dev, uint8_t col, uint8_t row, int mem_dma_kind, uint8_t mem_dma_ch);

// Start host<->AIE time correlation for the tiles armed by
// __Runtime_core_trace_begin. Inits a process-global AieTraceProfile, records
// the host clock (cps) and anchor0 (host time + each armed tile's AIE core
// timer). Call AFTER all __Runtime_core_trace_begin calls and just BEFORE the
// cores run (before __Runtime_launch_kernel_group). When present, the paired
// __Runtime_core_trace_end captures anchor1 and dumps the FULL [TIMESYNC] block
// (cps/anchor0/anchor1/trace) that host_aie_timeline.correlate() needs; when
// absent, __Runtime_core_trace_end stays decode-only (trace lines only). No-op
// when no tile was armed, or (cps=0, so still decode-only) under the simulator.
void __Runtime_core_trace_sync_begin(XAie_DevInst *dev);

// Record one host phase event (iter, phase name) into the auto-injected
// core-trace session's process-global profile. Captures the host clock now.
// No-op unless __Runtime_core_trace_sync_begin has armed the correlated session
// (so it is safe to call unconditionally from generated host code).
void __Runtime_core_trace_event(XAie_DevInst *dev, int iter, const char *phase);

// Read back, decode and dump every tile armed by __Runtime_core_trace_begin.
// Owns a static AieTraceProfile; for each registered tile reads the MemTile
// trace buffer, attaches the (col,row) tag, decodes into the profile, then
// emits one [TIMESYNC] block via __Runtime_aie_trace_profile_dump. Clears the
// registry. Call AFTER the cores have finished (post kernel-group wait), before
// device teardown. No-op when no tile was armed. If __Runtime_core_trace_sync_begin
// ran this session, uses that correlated profile (adds anchor1) instead.
void __Runtime_core_trace_end(XAie_DevInst *dev);

// Same read+decode as __Runtime_core_trace_end, but decodes every armed tile
// into a CALLER-supplied profile and does NOT init or dump it. Use this when the
// core trace must be unified with other timeline data (host clock, TS_ANCHOR
// host<->AIE anchors, TS_EVT phase events) inside ONE AieTraceProfile so a single
// __Runtime_aie_trace_profile_dump emits one coherent [TIMESYNC] block (e.g.
// example/perf/aieml_perf.cc). Clears the registry. No-op when no tile was armed.
void __Runtime_core_trace_end_into(XAie_DevInst *dev, AieTraceProfile *prof);

// ---------------------------------------------------------------------------
// DMA-capable buffer allocation with cache sync support
// ---------------------------------------------------------------------------

// Allocate a DMA-capable buffer via XAie_MemAllocate.
// dev must not be NULL — call partition() before alloc().
void *__Runtime_alloc_buffer(XAie_DevInst *dev, size_t size_bytes);

// Free a buffer allocated by __Runtime_alloc_buffer.
void __Runtime_free_buffer(XAie_DevInst *dev, void *ptr);

// ---------------------------------------------------------------------------
// Control-packet register/config write helpers.
// Build a control-packet payload (two in-band headers per <=4-word chunk) and
// push it via a SHIM MM2S BD so the target tile's CTRL stream-switch port
// performs the register/memory writes. Stream-switch routes are set up
// separately (routing.cc CTRL sink). Mirrors _XAie_CtrlPktizeElfPkt /
// _XAie_LoadElfStrmSwStartDma in the aie-rt driver.
// ---------------------------------------------------------------------------

// Build WRITE control-packet words for a contiguous block of @nwords 32-bit
// values targeting tile byte address @tile_addr onward, routed by @stream_id
// (two in-band headers per <=4-word chunk). When @lastwriteack is nonzero and
// @nwords>0, the LAST written word (tile_addr + (nwords-1)*4) is re-emitted as
// its own single-word WRITE-WITH-RETURN access (control-info operation=0b10,
// return stream id @ret_stream_id); because the dest CTRL port processes
// accesses in order, that ack's response is a completion barrier for all
// preceding writes. The response is a single AIE packet-switched stream header
// word (per doc/controlpkt.txt Table 3-32), and the return route keeps the
// header, so on success *@resp_words_out (may be NULL) gets 1 when the ack is
// appended, 0 otherwise. Returns words written (0 on capacity overflow).
uint32_t __Runtime_ctrl_pktize_write(uint32_t *out, uint32_t out_cap, uint32_t stream_id, uint32_t tile_addr,
                                     const uint32_t *data, uint32_t nwords, int lastwriteack, uint32_t ret_stream_id,
                                     uint32_t *resp_words_out);

// Build READ control-packet words for a contiguous @nwords block of tile
// registers/memory at byte address @tile_addr, following the AIE2ps Control-
// Packet word format (doc/controlpkt.txt Table 3-31/3-32): per access the
// control-info word carries [19:0]=byte addr, [21:20]=beats-1, [23:22]=01
// (read with return), [28:24]=@ret_stream_id (response routing), [31]=odd
// parity. The stream packet header uses @req_stream_id (routes the request to
// the dest CTRL master port). No data payload is emitted. Accesses are split
// to <=4 words and never cross a 128-bit boundary. On success *@resp_words_out
// (may be NULL) gets the expected response length in 32-bit words (per access:
// 1 stream header + beats data words). Returns request words written (0 on
// capacity overflow).
uint32_t __Runtime_ctrl_pktize_read(uint32_t *out, uint32_t out_cap, uint32_t req_stream_id, uint32_t ret_stream_id,
                                    uint32_t tile_addr, uint32_t nwords, uint32_t *resp_words_out);

// Parse a control-info word (the word after the stream packet header) per the
// AIE2ps Control-Packet format (doc/controlpkt.txt Table 3-31/3-32):
// [19:0]=local byte address, [21:20]=beats-1, [23:22]=operation
// (00=write, 01=read w/return, 10=write w/return), [28:24]=return stream id,
// [31]=odd parity over [30:0]. Any out pointer may be NULL. @beats_out gets the
// decoded beat count (field+1, i.e. 1..4). Returns 1 if the parity bit is
// consistent, 0 otherwise.
int __Runtime_ctrl_parse_ctrl_hdr(uint32_t ctrl_hdr, uint32_t *addr_out, uint32_t *op_out, uint32_t *beats_out,
                                  uint32_t *ret_sid_out);

// Parse an AIE packet-switched stream header word (the first word of a control
// packet / response packet): [4:0]=packet id (stream id), [14:12]=packet type
// (7=SLVERR on a control-packet response), [20:16]=source row, [27:21]=source
// column, [31]=odd parity over [30:0]. Any out pointer may be NULL. Returns 1 if
// the parity bit is consistent, 0 otherwise.
int __Runtime_ctrl_parse_pkt_hdr(uint32_t pkt_hdr, uint32_t *id_out, uint32_t *type_out, uint32_t *src_row_out,
                                 uint32_t *src_col_out);

// Control-packet send context. Identifies the shim source column, the
// destination tile (same column as the shim), the packet stream id, and the
// shim DMA channels / BD used for the forward send and the response return.
// Fill the input fields (dev..s2mm_ch, resp_words), then drive it in three steps:
//   __Runtime_ctrl_setup_routing(&c) - program the shim MM2S -> dest CTRL
//       forward route and the dest CTRL slave -> shim S2MM response return
//       route, alloc the response buffer, arm the shim S2MM drain;
//   __Runtime_ctrl_push(&c, buf, nwords, block, log) - send (block!=0 also
//                                       waits for the response via ctrl_tct_poll);
//   __Runtime_ctrl_tct_poll(&c, print) - wait for the response drain, return
//       its first word.
// The return path is the control-packet response (read-with-return / write-
// with-return): the dest CTRL slave port emits the response, which is circuit-
// switched down to the shim S2MM. Completion is the S2MM drain landing
// @resp_words words. __Runtime_ctrl_read_target composes all steps around a
// register read. `token` is an internal DDR response buffer allocated by
// setup_routing; release it with __Runtime_free_buffer(c.dev, c.token).
typedef struct {
    XAie_DevInst *dev; // partitioned device instance
    uint8_t shim_col;  // shim (row 0) source column
    uint8_t dest_col;  // destination column (must equal shim_col)
    uint8_t dest_row;  // destination row (>0)
    uint8_t stream_id; // request packet id; must match the header encoded in the buffer
    int32_t bd_id; // shim MM2S BD for the send (response S2MM uses an adjacent in-range slot: bd_id+1, or bd_id-1 when
                   // bd_id is the top slot)
    int32_t mm2s_ch;     // shim MM2S channel (forward send)
    int32_t s2mm_ch;     // shim S2MM channel (response return)
    uint32_t *token;     // internal DDR response buffer (set by setup_routing)
    uint32_t resp_words; // expected response length in 32-bit words (0 => treated as 1)
} __Runtime_CtrlInstance;

// Push a control-packet buffer (from __Runtime_alloc_buffer) through the SHIM
// MM2S channel described by @inst and wait for the MM2S send to drain. The send
// context (dev, shim_col, bd_id, mm2s_ch) comes from @inst; buf/nwords are the
// packet payload for this call. If @block is nonzero, also waits for the TCT
// return token via __Runtime_ctrl_tct_poll (requires @inst routing armed by
// __Runtime_ctrl_setup_routing). @log (nonzero) prints the per-send log line and
// the polled TCT value. @inst is not mutated.
AieRC __Runtime_ctrl_push(const __Runtime_CtrlInstance *inst, uint32_t *buf, uint32_t nwords, int block, int log);

// Program the forward (shim MM2S -> dest CTRL master) and response return (dest
// CTRL slave -> shim S2MM) routes for @inst, allocate its internal response
// buffer (@inst->resp_words words, min 1), and arm the shim S2MM to drain the
// response. Same-column only (dest_col must equal shim_col). On success
// @inst->token holds a freshly allocated DDR buffer.
AieRC __Runtime_ctrl_setup_routing(__Runtime_CtrlInstance *inst, int port_evt = 0);

// Poll the shim S2MM drain until the response lands, sync it for the CPU, and
// return the first response word. Uses @inst->token armed by
// __Runtime_ctrl_setup_routing. If @print is nonzero, prints the observed word.
// @inst is not mutated.
uint32_t __Runtime_ctrl_tct_poll(const __Runtime_CtrlInstance *inst, int print);

// Self-contained control-packet send: builds a __Runtime_CtrlInstance, programs
// the forward + response return routes, pushes the packet buffer via
// __Runtime_ctrl_push, waits for the response drain, then frees the buffer.
// buf/nwords are the packetized control words (from __Runtime_ctrl_pktize_write
// / __Runtime_ctrl_pktize_read);
// stream_id must match the packet id encoded in buf so the dest CTRL slave slot
// accepts it. On success writes the first response word to *tct_value_out (may
// be NULL). dev must be partitioned first; dest_col must equal shim_col.
AieRC __Runtime_ctrl_push_target(XAie_DevInst *dev, uint8_t shim_col, uint8_t dest_col, uint8_t dest_row,
                                 uint8_t stream_id, uint32_t *buf, uint32_t nwords, int32_t bd_id, int32_t mm2s_ch,
                                 int32_t s2mm_ch, uint32_t *tct_value_out);

// Self-contained control-packet register READ: builds READ control packets for
// @nwords 32-bit words at tile byte address @tile_addr on the dest tile
// (@dest_col must equal @shim_col), programs the forward + response-return
// routes, pushes the request via the shim MM2S, waits for the CTRL response to
// drain into the shim S2MM, and copies the @nwords read values into @out_data
// (skipping the per-access response headers). @req_stream_id routes the request
// to the dest CTRL master; @ret_stream_id is encoded in the response header.
// dev must be partitioned first. Returns XAIE_OK on success.
//
// NOTE: each control-packet response is its own AXI-stream packet (TLAST at end)
// and the single shim S2MM drain BD completes on the first TLAST. This reliably
// captures a SINGLE response packet, i.e. a read whose @nwords fit one access
// (<=4 words and not crossing a 128-bit boundary from @tile_addr). Larger reads
// span multiple response packets and would need one shim S2MM BD per packet
// (not yet implemented).
AieRC __Runtime_ctrl_read_target(XAie_DevInst *dev, uint8_t shim_col, uint8_t dest_col, uint8_t dest_row,
                                 uint8_t req_stream_id, uint8_t ret_stream_id, uint32_t tile_addr, uint32_t nwords,
                                 uint32_t *out_data, int32_t bd_id, int32_t mm2s_ch, int32_t s2mm_ch);

// Flush dirty cache lines for the buffer to DDR (before DMA reads it).
void __Runtime_sync_for_dev(XAie_DevInst *dev, void *ptr, size_t size);

// Invalidate cache lines for the buffer from DDR (after DMA wrote it).
void __Runtime_sync_for_cpu(XAie_DevInst *dev, void *ptr, size_t size);

// Data movement wrappers (wraps XAie routing APIs)
void __Runtime_move_data_to_tile(XAie_RoutingInstance *routing, XAie_LocType shim_tile, XAie_LocType dest_tile,
                                 XAie_MemInst *mem, uint32_t size, uint32_t tile_offset);

void __Runtime_move_data_from_tile(XAie_RoutingInstance *routing, XAie_LocType src_tile, XAie_LocType shim_tile,
                                   XAie_MemInst *mem, uint32_t tile_offset, uint32_t size);

#endif // AIE_RUNTIME_H
