/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef AIE_RUNTIME_H
#define AIE_RUNTIME_H

#include "xaiengine.h"
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
// bit 7 reserved (AIE_DMA_ISSUE_COUNT in aiehlc flag map)
/* When set (and the runtime is built with -DAIEHLC_PROFILING=1), arm core-module
 * perf counters (active/vec/stream-stall/lock-stall) on every compute tile at
 * kernel launch so a profiling host can read the per-tile cycle budget after the
 * run via __Runtime_core_perf_read_probe(). */
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
// DMA-capable buffer allocation with cache sync support
// ---------------------------------------------------------------------------

// Allocate a DMA-capable buffer via XAie_MemAllocate.
// dev must not be NULL — call partition() before alloc().
void *__Runtime_alloc_buffer(XAie_DevInst *dev, size_t size_bytes);

// Free a buffer allocated by __Runtime_alloc_buffer.
void __Runtime_free_buffer(XAie_DevInst *dev, void *ptr);

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
