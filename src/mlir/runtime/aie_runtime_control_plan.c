#include "aie_runtime_control_plan.h"

/* Pure planner: derives per-tile stream-switch ops for a row-based control
 * connection with no XAie dependency. See the header for type docs. */

/* Opt-in cross-check: when ACR_VALIDATE_RESERVED is defined (the standalone
 * planner unit test does this), every emitted forward/return op is asserted to
 * lie within the reservation table's reserved set (aie_runtime_resource.c) so
 * the planner can never emit a resource the table does not own (requirement #2).
 * Production builds leave it off: no runtime cost, no new hard dependency at
 * HW-emit time. Gen1/2/5 share one table today, so validation uses RT_RES_GEN2;
 * acr_port and RT_RES_PORT_* share the same numbering. */
#ifdef ACR_VALIDATE_RESERVED
#include <assert.h>
#define ACR_ASSERT_SLOT(port, slot, msel, arb, pkt)                                                                    \
    assert(__Runtime_res_is_reserved(RT_RES_GEN2, (uint8_t)(port), /*is_master=*/0, (uint8_t)(slot), (uint8_t)(arb),   \
                                     (uint8_t)(msel), (uint8_t)(pkt)))
#define ACR_ASSERT_MASTER(port, mselen, arb)                                                                           \
    assert(__Runtime_res_is_reserved(RT_RES_GEN2, (uint8_t)(port), /*is_master=*/1, RT_RES_NA, (uint8_t)(arb),         \
                                     (uint8_t)(mselen), RT_RES_NA))
#else
#define ACR_ASSERT_SLOT(port, slot, msel, arb, pkt) ((void)0)
#define ACR_ASSERT_MASTER(port, mselen, arb) ((void)0)
#endif

/* Return the port-usage bitmap cell for a tile's master or slave domain. */
static uint16_t *acr_book_cell(acr_portbook *b, uint8_t col, uint8_t row, int is_master) {
    uint16_t(*t)[64] = is_master ? b->master_used : b->slave_used;
    return &t[row][col];
}

/* Book master/slave port index @idx on tile (col,row) for port @p. Distinct
 * port types (WEST/EAST/NORTH/SOUTH/CTRL) are independent physical ports, so the
 * booking bit folds the port type in: bit = p*3 + idx (idx < 3, 5 ports => 15
 * bits, fits uint16). Rejects double-book of the same (port,idx). */
acr_rc acr_book_port(acr_portbook *b, uint8_t col, uint8_t row, acr_port p, uint8_t idx, int is_master) {
    if (row > ACR_MAX_ROWS || col >= 64 || idx >= 3)
        return ACR_ERR_BOUNDS;
    uint16_t bit = (uint16_t)(1u << ((unsigned)p * 3u + idx));
    uint16_t *cell = acr_book_cell(b, col, row, is_master);
    if (*cell & bit)
        return ACR_ERR_PORT_CONFLICT;
    *cell |= bit;
    return ACR_OK;
}

/* Append one op, guarding the fixed capacity. */
static acr_rc acr_emit_op(acr_oplist *o, const acr_op *op) {
    if (o->n >= ACR_MAX_OPS)
        return ACR_ERR_BOUNDS;
    o->ops[o->n++] = *op;
    return ACR_OK;
}

/* Emit one packet-slot arm (ACR_OP_SLOT) on tile (c,row)'s ingress slave @sport.
 * A slot matches an incoming id iff (id & mask) == (pkt & mask), injecting the
 * packet into @arb with select value @msel. The ingress slave *port* is booked
 * once by the caller; the four slots are sub-resources of that one port. */
static acr_rc acr_emit_slot(acr_oplist *o, uint8_t c, uint8_t row, acr_port sport, uint8_t slot, uint8_t pkt,
                            uint8_t mask, uint8_t msel, uint8_t arb) {
    ACR_ASSERT_SLOT(sport, slot, msel, arb, pkt);
    acr_op slotop = {.kind = ACR_OP_SLOT,
                     .col = c,
                     .row = row,
                     .sport = sport,
                     .sidx = 0,
                     .slot = slot,
                     .pkt_id = pkt,
                     .mask = mask,
                     .msel = msel,
                     .arbiter = arb,
                     .is_ret = (uint8_t)(arb == ACR_ARB_RET)};
    return acr_emit_op(o, &slotop);
}

/* Emit one packet-master enable (ACR_OP_MASTER_EN) on tile (c,row)'s @mport,
 * pulling every slot whose MSel bit is set in @mselen from arbiter @arb. Books
 * the master port (idx 0) first; keep_header is always set for control packets.
 *
 * NOTE on @arb: a master pulls only from packets injected into its OWN arbiter
 * (HW rule master.Arbitor == slot.Arbitor, xaie_ss.c). Because CTRL multicasts
 * with EAST (row-mcast slot) AND with NORTH (broadcast slot) off a *single*
 * slave slot each, and one slot injects into exactly one arbiter, CTRL/EAST/NORTH
 * must currently share one arbiter (ACR_ARB_CTRL). Splitting NORTH vs EAST onto
 * different arbiters would need a duplicate slot per multicast class, exceeding
 * ACR_NUM_SLOTS (4). @arb is parameterized so a future >4-slot layout can diverge
 * them, but all call sites pass ACR_ARB_CTRL today. */
static acr_rc acr_emit_master(acr_oplist *o, acr_portbook *b, uint8_t c, uint8_t row, acr_port mport, uint8_t mselen,
                              uint8_t arb) {
    acr_rc rc = acr_book_port(b, c, row, mport, 0, /*master*/ 1);
    if (rc != ACR_OK)
        return rc;
    ACR_ASSERT_MASTER(mport, mselen, arb);
    acr_op m = {.kind = ACR_OP_MASTER_EN,
                .col = c,
                .row = row,
                .mport = mport,
                .midx = 0,
                .msel = 0,
                .arbiter = arb,
                .mselen = mselen,
                .keep_header = 1,
                .is_ret = (uint8_t)(arb == ACR_ARB_RET)};
    return acr_emit_op(o, &m);
}

/* Core chain derivation. Each tile in [col_lo..col_hi] arms the SAME uniform slot
 * table on its ingress slave port, so a row-multicast reaches EVERY column of the
 * row (the "all columns respond" model) and there is no last-tile special casing:
 *   slot CONSUME   row-multicast @rowidx (mask 0x17: classes 00 & 10) -> CTRL + EAST
 *   slot BCAST     (id[4]=1)                                          -> CTRL + EAST (+NORTH on head)
 *   slot TRANSIT_N (row-multicast/broadcast, spine head only)         -> NORTH climb
 * Masters: CTRL 0x3 on every tile, EAST 0xB when not the last column, NORTH 0x6 on
 * the spine head. @head_ingress is the slave port the leftmost tile (col_lo)
 * receives on: WEST for a plain chain, SOUTH when the head is the top of the
 * vertical spine. @shim_col < 0 disables the NORTH climb / transit-north slot
 * (plain chain). @row is the physical AIE row (op coords); @rowidx is the add-order
 * row index carried in id[1:0]. @ctrl_id is retained for the emit layer's
 * provenance logging. (The legacy only-last / all-but-last column-subset classes
 * are no longer separately routed; the runtime's class-write wrappers still exist
 * but resolve onto this uniform superset.) */
static acr_rc acr_plan_chain_ex(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t rowidx, uint8_t col_lo,
                                uint8_t col_hi, uint8_t ctrl_id, int shim_col, acr_port head_ingress) {
    if (col_hi < col_lo || col_hi >= 64)
        return ACR_ERR_BOUNDS;
    if (rowidx > ACR_MAX_ROW_IDX) /* target row index lives in id[1:0] */
        return ACR_ERR_BOUNDS;
    if (ACR_NUM_SLOTS < 4) /* spine head arms consume+bcast+transitN+transitE */
        return ACR_ERR_SLOTS;

    for (uint8_t c = col_lo; c <= col_hi; c++) {
        acr_port sport = (c == col_lo) ? head_ingress : ACR_WEST;
        int is_last = (c == col_hi);
        int has_east = !is_last;
        int is_spine_head = (shim_col >= 0) && (c == (uint8_t)shim_col) && (c == col_lo);

        acr_rc rc;
        /* Book the ingress slave *port* (idx 0) once; slots are sub-resources. */
        if ((rc = acr_book_port(b, c, row, sport, 0, /*master*/ 0)) != ACR_OK)
            return rc;

            /* Interior tile (c < col_hi): masked consume (classes 00,10) +
             * broadcast + transit-east (class 01, EAST-only) [+ transit-north on
             * the spine head]. */
        uint8_t consume_id = (uint8_t)((ACR_CLASS_ALL_BUT_LAST << 2) | rowidx); /* == rowidx */
        uint8_t te_id = (uint8_t)((ACR_CLASS_ONLY_LAST << 2) | rowidx);
        if ((rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_CONSUME, consume_id, ACR_MASK_CONSUME, ACR_MSEL_CONSUME,
                                ACR_ARB_CTRL)) != ACR_OK)
            return rc;
        if ((rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_BCAST, ACR_ID_BCAST, ACR_MASK_BCAST, ACR_MSEL_BCAST,
                                ACR_ARB_CTRL)) != ACR_OK)
            return rc;
        if (is_spine_head && (rc = acr_emit_slot(o, c, row, sport, ACR_SLOT_TRANSIT_N, ACR_ID_TRANSIT, ACR_MASK_CLASS,
                                                 ACR_MSEL_TRANSIT_N, ACR_ARB_CTRL)) != ACR_OK)
            return rc;

        acr_op slaveen = {.kind = ACR_OP_SLAVE_EN, .col = c, .row = row, .sport = sport, .sidx = 0, .pkt_id = ctrl_id};
        if ((rc = acr_emit_op(o, &slaveen)) != ACR_OK)
            return rc;

            /* Interior CTRL pulls consume + broadcast (0x3). */
        if ((rc = acr_emit_master(o, b, c, row, ACR_CTRL, (uint8_t)ACR_MSELEN_CTRL, ACR_ARB_CTRL)) != ACR_OK)
            return rc;
        /* EAST forwards consume + broadcast + only-last transit (0xB) so every
         * class reaches its columns. */
        if (has_east &&
            (rc = acr_emit_master(o, b, c, row, ACR_EAST, (uint8_t)ACR_MSELEN_EAST, ACR_ARB_CTRL)) != ACR_OK)
            return rc;
        /* NORTH climbs the spine for broadcast + any transiting row-mcast,
         * head-on-spine-column only (0x6). */
        if (is_spine_head &&
            (rc = acr_emit_master(o, b, c, row, ACR_NORTH, (uint8_t)ACR_MSELEN_NORTH, ACR_ARB_CTRL)) != ACR_OK)
            return rc;
    }
    return ACR_OK;
}

/* Emit one return SLOT on tile (c,row)'s @sport slave: accept-any (pkt=0,mask=0)
 * into arbiter ACR_ARB_RET with select value @msel, then enable that slave port.
 * Books the slave port (idx 0) first (slots do not book on their own). Each
 * return slave port carries exactly one slot, so slot index == MSel. */
static acr_rc acr_emit_ret_slot(acr_oplist *o, acr_portbook *b, uint8_t c, uint8_t row, acr_port sport, uint8_t slot,
                                uint8_t msel) {
    acr_rc rc = acr_book_port(b, c, row, sport, 0, /*master*/ 0);
    if (rc != ACR_OK)
        return rc;
    if ((rc = acr_emit_slot(o, c, row, sport, slot, /*pkt=*/0, /*mask=*/0, msel, ACR_ARB_RET)) != ACR_OK)
        return rc;
    acr_op se = {.kind = ACR_OP_SLAVE_EN, .col = c, .row = row, .sport = sport, .sidx = 0, .pkt_id = 0, .is_ret = 1};
    return acr_emit_op(o, &se);
}

/* Return chain for [col_lo..col_hi] on @row (east->west packet merge). Every tile
 * injects its own CTRL-slave response (RET_LOCAL); a tile with an east neighbor
 * (c<col_hi) also forwards that neighbor's westbound merged responses
 * (RET_TRANSIT). The head (col_lo, on the spine column) drives the merged bus
 * SOUTH->VRET and additionally merges the descending upper-spine responses on its
 * NORTH slave (RET_NORTH); arming RET_NORTH unconditionally is harmless because on
 * the topmost head that slot simply never receives traffic, and it keeps this
 * routine fully local (no dependency on later row_adds, so no re-emit / double-
 * book). Non-head tiles drive their merged bus WEST to the next tile's EAST slave.
 * All return ports are disjoint from the forward chain (forward uses WEST/SOUTH
 * slave ingress + EAST/CTRL/NORTH masters), and master/slave domains are booked
 * separately, so there is no port conflict. See the header for the slot/MSel map. */
acr_rc acr_plan_return_chain(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t col_lo, uint8_t col_hi, int is_top) {
    if (col_hi < col_lo || col_hi >= 64)
        return ACR_ERR_BOUNDS;

    for (uint8_t c = col_lo; c <= col_hi; c++) {
        int is_head = (c == col_lo);
        int has_transit = (c < col_hi);
        acr_rc rc;

        /* Local response source: this tile's CTRL slave. */
        if ((rc = acr_emit_ret_slot(o, b, c, row, ACR_CTRL, ACR_SLOT_RET_LOCAL, ACR_MSEL_RET_LOCAL)) != ACR_OK)
            return rc;
        /* East neighbor's westbound merged responses (interior + head, not last). */
        if (has_transit &&
            (rc = acr_emit_ret_slot(o, b, c, row, ACR_EAST, ACR_SLOT_RET_TRANSIT, ACR_MSEL_RET_TRANSIT)) != ACR_OK)
            return rc;

        if (is_head) {
            /* Non-top head: merge the descending upper-spine responses on its
             * NORTH slave. The top head has nothing above it, so it omits this
             * idle slot. */
            if (!is_top &&
                (rc = acr_emit_ret_slot(o, b, c, row, ACR_NORTH, ACR_SLOT_RET_NORTH, ACR_MSEL_RET_NORTH)) != ACR_OK)
                return rc;
            /* Head descent: SOUTH master -> VRET, pulling {local, transit?} plus
             * {north} only when a head can sit above (non-top). */
            uint8_t mselen = (uint8_t)((1u << ACR_MSEL_RET_LOCAL) | (has_transit ? (1u << ACR_MSEL_RET_TRANSIT) : 0u) |
                                       (is_top ? 0u : (1u << ACR_MSEL_RET_NORTH)));
            if ((rc = acr_emit_master(o, b, c, row, ACR_SOUTH, mselen, ACR_ARB_RET)) != ACR_OK)
                return rc;
        } else {
            /* Interior/last: WEST master merges {local, transit?} to the west tile. */
            uint8_t mselen = (uint8_t)((1u << ACR_MSEL_RET_LOCAL) | (has_transit ? (1u << ACR_MSEL_RET_TRANSIT) : 0u));
            if ((rc = acr_emit_master(o, b, c, row, ACR_WEST, mselen, ACR_ARB_RET)) != ACR_OK)
                return rc;
        }
    }
    return ACR_OK;
}

/* Public chain planner: plain horizontal chain, head tile ingresses on WEST, no
 * vertical spine climb (shim_col=-1). A plain chain is a single row => rowidx 0. */
acr_rc acr_plan_chain(acr_oplist *o, acr_portbook *b, uint8_t row, uint8_t col_lo, uint8_t col_hi, uint8_t ctrl_id) {
    return acr_plan_chain_ex(o, b, row, /*rowidx=*/0, col_lo, col_hi, ctrl_id, /*shim_col=*/-1, ACR_WEST);
}

/* True if @row is an already-configured chain head in @s. */
static int acr_row_is_head(const acr_state *s, uint8_t row) {
    for (uint8_t i = 0; i < s->nrows; i++)
        if (s->rows[i] == row)
            return 1;
    return 0;
}

/* Ensure the shared vertical spine (column @shim_col) reaches @row, then build
 * the EAST chain [col_lo..col_hi] on that row. Idempotent per requirement #5:
 *   - re-adding a row already in @s->rows emits nothing;
 *   - a higher row reuses the existing spine and only extends it upward.
 * @s->spine_top is the highest row that HAS a spine-up hop; to feed head @row's
 * SOUTH ingress the spine must pass through rows 1..row-1, so we extend up to
 * row-1 (NOT row itself): the head row taps SOUTH into its chain and must not
 * also drive the spine up on the same SOUTH slave port.
 *
 * A spine-up hop through a pure pass-through tile (NOT a configured head) is a
 * circuit-switched CCT, SOUTH-in -> NORTH-out (one ACR_OP_CCT). Circuit switching
 * is class-agnostic, so broadcast AND every row-multicast class climb through
 * unchanged. An already-configured head tile needs NO hop here: its own chain
 * planning already emitted a NORTH master (MSelEn 0x6) that climbs broadcast + any
 * transiting row-multicast up the spine, so we skip it in this loop. row==1 needs
 * no hop (fed by the shim's NORTH directly).
 *
 * The chain head (col_lo == shim_col) ingresses from the spine below (SOUTH) and,
 * because it sits on the spine column, emits its own NORTH climb master (broadcast
 * + transit) inside acr_plan_chain_ex, so a row-multicast targeting any row above
 * still reaches it. Each configured row is assigned its add-order index (0,1,2,..)
 * which travels in id[1:0]; the chain is built as the column-subset class superset
 * (all-but-last / only-last / whole-row + broadcast) and the per-transfer class +
 * target row index are carried by the control packet's stream id at send time. */
acr_rc acr_plan_row_add(acr_state *s, acr_oplist *o, acr_portbook *b, uint8_t shim_col, uint8_t row, uint8_t col_lo,
                        uint8_t col_hi, uint8_t ctrl_id, int is_top) {
    if (row == 0 || shim_col >= 64)
        return ACR_ERR_BOUNDS;
    if (col_lo != shim_col) /* row entry is the vertical spine's left column */
        return ACR_ERR_BOUNDS;

    /* Idempotent: already configured -> emit nothing. */
    if (acr_row_is_head(s, row))
        return ACR_OK;

    if (s->nrows >= ACR_MAX_ROWS)
        return ACR_ERR_BOUNDS;
    /* The new row's add-order index travels in id[1:0]; cap at 4 configured rows. */
    if (s->nrows > ACR_MAX_ROW_IDX)
        return ACR_ERR_BOUNDS;

    /* Extend the spine's up-hops to row-1 (the head row taps SOUTH, it is not a
     * spine-up hop). row==1 needs none (fed by the shim's NORTH). A row that is
     * already a configured head fans the spine up via its own broadcast NORTH
     * master, so it needs no CCT here; only pure pass-through rows get a CCT. */
    uint8_t need_top = (uint8_t)(row - 1);
    if (need_top > s->spine_top) {
        for (uint8_t r = (uint8_t)(s->spine_top + 1); r <= need_top; r++) {
            if (acr_row_is_head(s, r))
                continue; /* head row already climbs via its own NORTH master */
            acr_op cct = {.kind = ACR_OP_CCT,
                          .col = shim_col,
                          .row = r,
                          .sport = ACR_SOUTH,
                          .sidx = 0,
                          .mport = ACR_NORTH,
                          .midx = 0};
            acr_rc crc = acr_emit_op(o, &cct);
            if (crc != ACR_OK)
                return crc;
            /* Symmetric RETURN descent through this pass-through row: NORTH slave
             * -> SOUTH master CCT. The emit layer maps NORTH-slave and SOUTH-master
             * to VRET, so this carries the descending merged responses down toward
             * the shim. Head rows are skipped (they descend via their own return
             * SOUTH master emitted in acr_plan_return_chain). */
            acr_op rcct = {.kind = ACR_OP_CCT,
                           .col = shim_col,
                           .row = r,
                           .sport = ACR_NORTH,
                           .sidx = 0,
                           .mport = ACR_SOUTH,
                           .midx = 0,
                           .is_ret = 1};
            if ((crc = acr_emit_op(o, &rcct)) != ACR_OK)
                return crc;
        }
        s->spine_top = need_top;
    }

    /* Build the horizontal chain; head tile takes the spine input from SOUTH and
     * emits its broadcast NORTH climb master (head sits on shim_col == col_lo). */
    acr_rc rc = acr_plan_chain_ex(o, b, row, /*rowidx=*/s->nrows, col_lo, col_hi, ctrl_id, /*shim_col=*/(int)shim_col,
                                  ACR_SOUTH);
    if (rc != ACR_OK)
        return rc;

    /* Return path: every column of this row injects/forwards its CTRL response
     * west, the head drives it SOUTH->VRET (merging any upper-spine descent). */
    rc = acr_plan_return_chain(o, b, row, col_lo, col_hi, is_top);
    if (rc != ACR_OK)
        return rc;

    s->rows[s->nrows++] = row;
    return ACR_OK;
}
