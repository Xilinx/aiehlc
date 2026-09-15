#ifndef AIE_RUNTIME_CONTROL_PLAN_H
#define AIE_RUNTIME_CONTROL_PLAN_H
#include <stdint.h>
/* The HW stream-switch resources the planner emits (slots, arbiters, msel, the
 * id/mask scheme) are owned by the reservation table. This header aliases its
 * ACR_* names to the RT_RES_* single-source-of-truth constants so the planner
 * and the table can never drift. */
#include "aie_runtime_resource.h"

/* Pure planner for row-based control connections. No XAie dependency so it can
 * be unit-tested on the host. The emit layer (aie_runtime.c) translates the
 * resulting acr_oplist into XAie stream-switch calls (packet-switch master/
 * slave enable, slave slot enable, circuit connection enable).
 * See docs/plans row-control-connection design doc. */

#define ACR_MAX_ROWS 16
#define ACR_MAX_OPS 256
#define ACR_NUM_SLOTS RT_RES_NUM_SLOTS     /* NumSlaveSlots per port (xaie_ss.c:755) */
#define ACR_MAX_ROW_IDX RT_RES_MAX_ROW_IDX /* target row index lives in id[1:0] (0..3 => up to 4 rows) */

/* Stream-id scheme (5-bit id): [4]=broadcast marker; when [4]=0 the packet is a
 * row-multicast whose target row is the add-order index in id[1:0]; id[3:2] is a
 * legacy column-subset class selector (see ACR_CLASS_* below).
 *   id = (class << 2) | rowidx  (row-multicast), or ACR_ID_BCAST (broadcast).
 * The CURRENT forward planner arms a UNIFORM per-tile slot superset so EVERY
 * column of the target row consumes a row-multicast packet (the governing
 * "all columns respond" model): a CONSUME slot (mask ACR_MASK_CONSUME 0x17,
 * matching classes 00 and 10 for row K) + a BCAST slot; spine-head tiles add a
 * TRANSIT_N climb slot. There is no longer any per-column "last tile" special
 * casing: all-but-last and whole-row both deliver to every column, and the
 * only-last class (01) is not separately routed by the planner. The ACR_CLASS_*
 * and legacy last-tile slot macros below are retained only for the runtime's
 * class-write wrappers and provenance; the planner emits the uniform superset. */
#define ACR_CLASS_ALL_BUT_LAST 0
#define ACR_CLASS_ONLY_LAST 1
#define ACR_CLASS_WHOLE_ROW 2

/* Forward slot indices (== MSel on that tile). CONSUME + BCAST are armed on every
 * tile; TRANSIT_N is armed on the spine head only. */
#define ACR_SLOT_CONSUME RT_RES_SLOT_CONSUME     /* row-multicast @row K (mask 0x17) -> CTRL + EAST */
#define ACR_SLOT_BCAST RT_RES_SLOT_BCAST         /* id[4]=1 -> CTRL + EAST (+NORTH on head) */
#define ACR_SLOT_TRANSIT_N RT_RES_SLOT_TRANSIT_N /* row-multicast/broadcast -> NORTH climb (spine head only) */
#define ACR_SLOT_TRANSIT_E 3 /* legacy: former only-last EAST forward (unused by uniform planner) */

/* Legacy last-tile slot indices — retained for the ACR_CLASS_* runtime wrappers;
 * the current uniform planner does not arm these. */
#define ACR_SLOT_ONLY_LAST 0  /* legacy: class 01 @row K -> CTRL */
#define ACR_SLOT_WHOLE 1      /* legacy: class 10 @row K -> CTRL */
#define ACR_SLOT_BCAST_LAST 2 /* legacy: id[4]=1 -> CTRL */

/* MSel select value each slot injects (mirrors the slot index above). */
#define ACR_MSEL_CONSUME RT_RES_MSEL_CONSUME
#define ACR_MSEL_BCAST RT_RES_MSEL_BCAST
#define ACR_MSEL_TRANSIT_N RT_RES_MSEL_TRANSIT_N
#define ACR_MSEL_TRANSIT_E 3
#define ACR_MSEL_ONLY_LAST 0
#define ACR_MSEL_WHOLE 1
#define ACR_MSEL_BCAST_LAST 2

#define ACR_ARB_CTRL RT_RES_ARB_CTRL /* single shared arbiter for all classes */

/* --- Return chain (write-ack / read response) ---------------------------- */
/* Every target-row column returns its CTRL-slave response; the responses merge
 * west onto a packet bus that funnels down the spine (VRET) to the shim S2MM.
 * The return arbiter is distinct from the forward one so the two packet fabrics
 * never contend on a shared arbiter. Each return slot lives on its OWN slave
 * port (RET_LOCAL on CTRL-slave, RET_TRANSIT on EAST-slave, RET_NORTH on
 * NORTH-slave), so slot index == MSel and the port carries a single slot. */
#define ACR_ARB_RET RT_RES_ARB_RET /* return arbiter, distinct from ACR_ARB_CTRL */

#define ACR_SLOT_RET_LOCAL RT_RES_SLOT_RET_LOCAL     /* this tile's CTRL-slave response */
#define ACR_SLOT_RET_TRANSIT RT_RES_SLOT_RET_TRANSIT /* east neighbor's westbound merged responses */
#define ACR_SLOT_RET_NORTH RT_RES_SLOT_RET_NORTH     /* upper spine head's descending responses */
#define ACR_SLOT_RET_EAST 2                          /* upper spine head's descending responses */
#define ACR_MSEL_RET_LOCAL RT_RES_MSEL_RET_LOCAL
#define ACR_MSEL_RET_TRANSIT RT_RES_MSEL_RET_TRANSIT
#define ACR_MSEL_RET_NORTH RT_RES_MSEL_RET_NORTH
#define ACR_MSEL_RET_EAST 2

/* Per-master MSelEn bitmaps for the return SOUTH/WEST masters. */
#define ACR_MSELEN_RET_HEAD                                                                                            \
    ((1u << ACR_MSEL_RET_LOCAL) | (1u << ACR_MSEL_RET_TRANSIT) | (1u << ACR_MSEL_RET_NORTH)) /* 0x7 */
#define ACR_MSELEN_RET_MID ((1u << ACR_MSEL_RET_LOCAL) | (1u << ACR_MSEL_RET_TRANSIT))       /* 0x3 */
#define ACR_MSELEN_RET_LOCALONLY (1u << ACR_MSEL_RET_LOCAL)                                  /* 0x1 */

/* Reserved-bit id/mask scheme (5-bit stream id):
 *   [4]=bcast marker; [3:2]=class; [1:0]=target row index when [4]=0. */
#define ACR_ID_BCAST RT_RES_ID_BCAST         /* [4]=1 */
#define ACR_MASK_BCAST RT_RES_MASK_BCAST     /* match on [4] only */
#define ACR_ID_TRANSIT RT_RES_ID_TRANSIT     /* transit-north matches any row-mcast (bit4=0) */
#define ACR_MASK_CLASS RT_RES_MASK_CLASS     /* match on [4] only: row-mcast (0) vs broadcast (1) */
#define ACR_MASK_EXACT 0x1F                  /* full 5-bit exact match */
#define ACR_MASK_CONSUME RT_RES_MASK_CONSUME /* match [4]=0,[2]=0,[1:0]=K ; ignore [3] -> classes 00 & 10 */

/* Per-master MSelEn bitmaps: which slot MSels each enabled master pulls. */
#define ACR_MSELEN_CTRL ((1u << ACR_MSEL_CONSUME) | (1u << ACR_MSEL_BCAST))                              /* 0x3 */
#define ACR_MSELEN_EAST ((1u << ACR_MSEL_CONSUME) | (1u << ACR_MSEL_BCAST) | (1u << ACR_MSEL_TRANSIT_E)) /* 0xB */
#define ACR_MSELEN_NORTH ((1u << ACR_MSEL_BCAST) | (1u << ACR_MSEL_TRANSIT_N))                           /* 0x6 */
/* Legacy last-tile CTRL master mask; unused by the uniform planner (last tiles
 * now pull the same ACR_MSELEN_CTRL 0x3 as every other column). */
#define ACR_MSELEN_CTRL_LAST                                                                                           \
    ((1u << ACR_MSEL_ONLY_LAST) | (1u << ACR_MSEL_WHOLE) | (1u << ACR_MSEL_BCAST_LAST)) /* 0x7 */

/* Direction/port tags, decoupled from XAie enums for pure testing. */
typedef enum { ACR_WEST, ACR_EAST, ACR_NORTH, ACR_SOUTH, ACR_CTRL } acr_port;

/* One emitted stream-switch operation (translated to XAie by the emit layer). */
typedef enum { ACR_OP_SLOT, ACR_OP_SLAVE_EN, ACR_OP_MASTER_EN, ACR_OP_CCT } acr_op_kind;
typedef struct {
    acr_op_kind kind;
    uint8_t col, row;
    acr_port sport;
    uint8_t sidx; /* slave side */
    acr_port mport;
    uint8_t midx; /* master side */
    uint8_t slot, pkt_id, mask, msel, arbiter, mselen;
    uint8_t keep_header; /* 1 => DONOT_DROP_HEADER */
    /* 1 => this op belongs to the RETURN (response) fabric, 0 => forward. The
     * emit layer tags its provenance-map lines dir=ret/fwd from this. It cannot
     * be inferred from the port: the return chain's slots sit on CTRL and EAST
     * slaves, the same port types the forward chain drives as masters. Slot and
     * master ops derive it from arbiter == ACR_ARB_RET (the two fabrics use
     * disjoint arbiters by construction); slave-enable and circuit ops carry no
     * arbiter, so their emitter sets it explicitly. */
    uint8_t is_ret;
} acr_op;

typedef struct {
    acr_op ops[ACR_MAX_OPS];
    int n;
} acr_oplist;

/* Per-tile used-port bitmaps for conflict rejection (requirement #7). */
typedef struct {
    uint16_t master_used[ACR_MAX_ROWS + 1][64]; /* [row][col] bit i => master idx i used */
    uint16_t slave_used[ACR_MAX_ROWS + 1][64];
} acr_portbook;

typedef enum { ACR_OK = 0, ACR_ERR_PORT_CONFLICT, ACR_ERR_SLOTS, ACR_ERR_BOUNDS, ACR_ERR_TILETYPE } acr_rc;

/* Pure mirror of the instance's shared-spine state, so the spine/row-add logic
 * can be host-unit-tested without the XAie-bearing __Runtime_CtrlRowFabric. */
typedef struct {
    uint8_t spine_top;          /* highest row with a pass-through hop (0 => none) */
    uint8_t rows[ACR_MAX_ROWS]; /* rows already configured (for idempotence) */
    uint8_t nrows;
} acr_state;

/* Pure planner API (defined in aie_runtime_control_plan.c). No XAie dependency. */

/* Book master/slave port index @idx on tile (col,row) for port @p; rejects a
 * double-book of the same (port,idx). Returns ACR_ERR_* on conflict/bounds. */
acr_rc acr_book_port(acr_portbook *b, uint8_t col, uint8_t row, acr_port p, uint8_t idx, int is_master);

/* Plain horizontal chain [col_lo..col_hi] on @row, head ingresses on WEST. Every
 * tile arms the SAME uniform forward slot superset on its ingress slave port:
 * CONSUME (row-multicast @row K, mask 0x17 -> CTRL + EAST) + BCAST (id[4]=1 ->
 * CTRL + EAST). A row-multicast therefore reaches EVERY column of the row (no
 * last-tile special casing); broadcast reaches every column too. A plain chain
 * passes row index 0 and has no NORTH spine climb or transit-north slot
 * (shim_col=-1). */
acr_rc acr_plan_chain(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t col_lo, uint8_t col_hi, uint8_t ctrl_id);

/* Ensure the shared vertical spine (column @shim_col) reaches @row, then build the
 * uniform EAST forward chain [col_lo..col_hi] AND its WEST-going return chain
 * (acr_plan_return_chain), assigning @row the next add-order index (id[1:0]).
 * Pass-through spine rows below @row get both a forward (SOUTH->NORTH) and a
 * return (NORTH->SOUTH) circuit hop. Idempotent per row; caps at
 * ACR_MAX_ROW_IDX+1 configured rows. */
acr_rc acr_plan_row_add(acr_state *s, acr_oplist *o, acr_portbook *b, uint8_t shim_col, uint8_t row, uint8_t col_lo,
                        uint8_t col_hi, uint8_t ctrl_id, int is_top);

/* Emit the WEST-going packet return chain for [col_lo..col_hi] on @row: every
 * tile injects its CTRL-slave response (RET_LOCAL slot) and forwards its east
 * neighbor's westbound merged responses (RET_TRANSIT, only when c<col_hi); the
 * head (col_lo, on the spine) drives its merged responses SOUTH->VRET and also
 * merges the descending upper-spine responses (RET_NORTH slot, harmless when no
 * head sits above). Non-head tiles drive WEST. keep_header=1, pkt_id/mask=0
 * (accept any response id, mirroring the unicast XAie_PacketInit(sid,0)/Mask=0).
 * Return ports (CTRL/EAST/NORTH slave, WEST/SOUTH master) are disjoint from the
 * forward chain's ports, so the shared port book never double-books. Invoked by
 * acr_plan_row_add after the forward chain; also usable standalone for testing.
 * @is_top marks the topmost configured row; when set, the head omits the idle
 * RET_NORTH slot and drives SOUTH->VRET with {local, transit} only (nothing
 * descends from above the top head). */
acr_rc acr_plan_return_chain(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t col_lo, uint8_t col_hi, int is_top);
#endif
