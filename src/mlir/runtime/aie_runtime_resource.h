#ifndef AIE_RUNTIME_RESOURCE_H
#define AIE_RUNTIME_RESOURCE_H
#include <stdint.h>

/* Single source of truth for the hardware stream-switch resources the control
 * plane occupies (packet slots, arbiters, msel, packet-ids/masks), keyed by
 * device generation + stream port-type. Pure C, no XAie dependency, C++-safe
 * via the extern "C" guard below.
 *
 * Consumers:
 *   1. the list API (__Runtime_res_port_reserved) enumerating reserved pkt-id
 *      resources on a given port for a gen/layout (req #1);
 *   2. aie_runtime_control_plan.{c,h}, whose ACR_* resource macros alias the
 *      RT_RES_* constants below, and whose planner may be validated against the
 *      reserved set (__Runtime_res_is_reserved, req #2);
 *   3. the MLIR compile-time ResourceMgr, which marks the reserved resources
 *      used so routing/scheduling excludes them (req #3).
 *
 * The port-type numbering mirrors aie_runtime_control_plan.h's acr_port enum
 * (WEST=0, EAST=1, NORTH=2, SOUTH=3, CTRL=4) but is duplicated here as plain
 * integer constants so this header stays self-contained (control_plan.h
 * includes THIS header, not the reverse). */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { RT_RES_GEN1 = 1, RT_RES_GEN2 = 2, RT_RES_GEN5 = 5 } rt_res_gen;

/* Port-type numbering (mirrors acr_port). */
#define RT_RES_PORT_WEST 0
#define RT_RES_PORT_EAST 1
#define RT_RES_PORT_NORTH 2
#define RT_RES_PORT_SOUTH 3
#define RT_RES_PORT_CTRL 4

/* Arbiters: forward fabric vs return fabric (disjoint by construction). */
#define RT_RES_ARB_CTRL 0
#define RT_RES_ARB_RET 1

/* Slot budget and target-row index range. */
#define RT_RES_NUM_SLOTS 4
#define RT_RES_MAX_ROW_IDX 3

/* Forward slot indices (== MSel on that tile). */
#define RT_RES_SLOT_CONSUME 0
#define RT_RES_SLOT_BCAST 1
#define RT_RES_SLOT_TRANSIT_N 2

/* Return slot indices (each on its own slave port, so slot == MSel). */
#define RT_RES_SLOT_RET_LOCAL 0
#define RT_RES_SLOT_RET_TRANSIT 1
#define RT_RES_SLOT_RET_NORTH 2

/* MSel select values (mirror the slot index). */
#define RT_RES_MSEL_CONSUME 0
#define RT_RES_MSEL_BCAST 1
#define RT_RES_MSEL_TRANSIT_N 2
#define RT_RES_MSEL_RET_LOCAL 0
#define RT_RES_MSEL_RET_TRANSIT 1
#define RT_RES_MSEL_RET_NORTH 2

/* Reserved-bit id/mask scheme (5-bit stream id): [4]=bcast marker; [3:2]=class;
 * [1:0]=target row index when [4]=0. */
#define RT_RES_ID_BCAST 0x10
#define RT_RES_MASK_BCAST 0x10
#define RT_RES_ID_TRANSIT 0x00
#define RT_RES_MASK_CLASS 0x10
#define RT_RES_MASK_CONSUME 0x17

/* Sentinel for the fields that do not apply to a given entry. */
#define RT_RES_NA 0xFF

/* One reserved stream-switch resource: a slave packet slot or a master enable. */
typedef struct {
    uint8_t port;      /* RT_RES_PORT_*: WEST/EAST/NORTH/SOUTH/CTRL */
    uint8_t is_master; /* 0 = slave slot, 1 = master enable */
    uint8_t slot;      /* slave slot index; RT_RES_NA for master */
    uint8_t arbiter;   /* RT_RES_ARB_CTRL / RT_RES_ARB_RET */
    uint8_t msel;      /* slave: slot msel; master: mselen bitmap */
    uint8_t pkt_id;    /* reserved id value (slave slot); RT_RES_NA for master */
    uint8_t mask;      /* id-match mask (slave slot); RT_RES_NA for master */
    uint8_t is_ret;    /* 0 forward fabric, 1 return fabric */
} rt_res_entry;

/* #1 list API: reserved control-plane resources on ONE stream port for gen +
 * layout. (col,row) reserved for future gen/layout divergence; ignored today.
 * Fills up to @cap entries in @out; returns the count, or -1 on bad args. */
int __Runtime_res_port_reserved(rt_res_gen gen, int col, int row, uint8_t port, uint8_t is_master, rt_res_entry *out,
                                int cap);

/* Aggregate helpers for the MLIR exclusion path (#3). */
uint32_t __Runtime_res_reserved_pktid_mask(rt_res_gen gen);                            /* bit i => id i reserved */
uint32_t __Runtime_res_reserved_arbiter_mask(rt_res_gen gen);                          /* bit i => arbiter i reserved */
int __Runtime_res_reserved_slot_mask(rt_res_gen gen, uint8_t port, uint8_t is_master); /* bit i => slot i, -1 bad */

/* Planner validation (#2): is this emitted op within the reserved set? */
int __Runtime_res_is_reserved(rt_res_gen gen, uint8_t port, uint8_t is_master, uint8_t slot, uint8_t arbiter,
                              uint8_t msel, uint8_t pkt_id);

/* Map a "Gen1"/"Gen2"/"Gen5" (or "1"/"2"/"5") string to rt_res_gen; defaults to
 * RT_RES_GEN2 for anything unrecognized. */
rt_res_gen __Runtime_res_gen_from_name(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* AIE_RUNTIME_RESOURCE_H */
