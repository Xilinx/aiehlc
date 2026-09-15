// Host unit tests for the pure row-control planner.
#include <cassert>
#include <cstdio>
extern "C" {
#include "aie_runtime_control_plan.h"
acr_rc acr_book_port(acr_portbook *, uint8_t col, uint8_t row, acr_port, uint8_t idx, int is_master);
acr_rc acr_plan_chain(acr_oplist *, acr_portbook *, uint8_t row, uint8_t col_lo, uint8_t col_hi, uint8_t ctrl_id);
acr_rc acr_plan_row_add(acr_state *, acr_oplist *, acr_portbook *, uint8_t shim_col, uint8_t row, uint8_t col_lo,
                        uint8_t col_hi, uint8_t ctrl_id, int is_top);
acr_rc acr_plan_return_chain(acr_oplist *, acr_portbook *, uint8_t row, uint8_t col_lo, uint8_t col_hi, int is_top);
}

static int count_kind(const acr_oplist *o, acr_op_kind k) {
    int c = 0;
    for (int i = 0; i < o->n; i++)
        if (o->ops[i].kind == k)
            c++;
    return c;
}

// Count master-enable ops on a column for a given master port.
static int count_master(const acr_oplist *o, uint8_t col, acr_port mport) {
    int c = 0;
    for (int i = 0; i < o->n; i++)
        if (o->ops[i].kind == ACR_OP_MASTER_EN && o->ops[i].col == col && o->ops[i].mport == mport)
            c++;
    return c;
}

// Count all slot ops on a column.
static int count_slots(const acr_oplist *o, uint8_t col) {
    int c = 0;
    for (int i = 0; i < o->n; i++)
        if (o->ops[i].kind == ACR_OP_SLOT && o->ops[i].col == col)
            c++;
    return c;
}

// Count slot ops on a column that inject into a given arbiter (forward =
// ACR_ARB_CTRL, return = ACR_ARB_RET).
static int count_slots_arb(const acr_oplist *o, uint8_t col, uint8_t arb) {
    int c = 0;
    for (int i = 0; i < o->n; i++)
        if (o->ops[i].kind == ACR_OP_SLOT && o->ops[i].col == col && o->ops[i].arbiter == arb)
            c++;
    return c;
}

// MSelEn of the (single) master-enable on a column for a given master port.
static int master_mselen(const acr_oplist *o, uint8_t col, acr_port mport) {
    for (int i = 0; i < o->n; i++)
        if (o->ops[i].kind == ACR_OP_MASTER_EN && o->ops[i].col == col && o->ops[i].mport == mport)
            return o->ops[i].mselen;
    return -1;
}

// Find a slot op by (col, slave port, slot index, arbiter). Returns NULL if none.
static const acr_op *find_slot(const acr_oplist *o, uint8_t col, acr_port sport, uint8_t slot, uint8_t arb) {
    for (int i = 0; i < o->n; i++) {
        const acr_op *op = &o->ops[i];
        if (op->kind == ACR_OP_SLOT && op->col == col && op->sport == sport && op->slot == slot && op->arbiter == arb)
            return op;
    }
    return nullptr;
}

// True iff a CCT op with the given (col,row,slave-port,master-port) exists.
static int has_cct(const acr_oplist *o, uint8_t col, uint8_t row, acr_port sport, acr_port mport) {
    for (int i = 0; i < o->n; i++) {
        const acr_op *op = &o->ops[i];
        if (op->kind == ACR_OP_CCT && op->col == col && op->row == row && op->sport == sport && op->mport == mport)
            return 1;
    }
    return 0;
}

static int test_book() {
    acr_portbook b = {};
    assert(acr_book_port(&b, 3, 4, ACR_EAST, 0, /*master*/ 1) == ACR_OK);
    // same port+idx again => conflict
    assert(acr_book_port(&b, 3, 4, ACR_EAST, 0, 1) == ACR_ERR_PORT_CONFLICT);
    // different idx ok
    assert(acr_book_port(&b, 3, 4, ACR_EAST, 1, 1) == ACR_OK);
    // master and slave domains are independent
    assert(acr_book_port(&b, 3, 4, ACR_EAST, 0, /*slave*/ 0) == ACR_OK);
    return 0;
}

// Slot budget with the simplified forward chain + return chain. This row_add is
// planned as the TOP row (is_top=1), so the spine head omits the idle RET_NORTH
// merge slot. The per-column slot total is forward + return:
//   spine head (col_lo, on spine col): fwd consume+bcast+transitN (3) +
//       ret local+transit (2, no north on top head) = 5
//   interior (col_lo<c<col_hi): fwd consume+bcast (2) + ret local+transit (2) = 4
//   last (col_hi): fwd consume+bcast (2) + ret local (1) = 3
// The forward CTRL master is always 0x3 (no more last-tile 0x7). A plain chain
// (acr_plan_chain) emits forward slots only (no return chain, no NORTH, no CCT):
// every tile arms consume+bcast (2 slots), CTRL 0x3.
static int test_slot_classes() {
    acr_state s = {};
    acr_portbook b = {};
    acr_oplist o = {};
    assert(acr_plan_row_add(&s, &o, &b, 2, 3, 2, 4, /*ctrl_id*/ 0, /*is_top*/ 1) == ACR_OK); // head row 3, cols 2..4
    assert(count_slots(&o, 2) == 5); // top spine head: fwd 3 + ret 2
    assert(count_slots(&o, 3) == 4); // interior: fwd 2 + ret 2
    assert(count_slots(&o, 4) == 3); // last: fwd 2 + ret 1
    assert(count_slots_arb(&o, 2, ACR_ARB_CTRL) == 3);
    assert(count_slots_arb(&o, 2, ACR_ARB_RET) == 2); // top head: local+transit (no north)
    assert(count_slots_arb(&o, 4, ACR_ARB_CTRL) == 2);
    assert(count_slots_arb(&o, 4, ACR_ARB_RET) == 1);
    for (uint8_t c = 2; c <= 4; c++) { // forward CTRL master uniform 0x3
        assert(count_master(&o, c, ACR_CTRL) == 1);
        assert(master_mselen(&o, c, ACR_CTRL) == 0x3);
    }
    // A plain chain: forward slots only, every tile 2 slots, CTRL 0x3, no CCT,
    // no NORTH master.
    acr_oplist op = {};
    acr_portbook pb = {};
    assert(acr_plan_chain(&op, &pb, 4, 2, 3, /*ctrl_id*/ 0) == ACR_OK);
    assert(count_slots(&op, 2) == 2);
    assert(master_mselen(&op, 2, ACR_CTRL) == 0x3);
    assert(count_slots(&op, 3) == 2);
    assert(master_mselen(&op, 3, ACR_CTRL) == 0x3);
    assert(count_kind(&op, ACR_OP_CCT) == 0);
    for (int i = 0; i < op.n; i++)
        assert(!(op.ops[i].kind == ACR_OP_MASTER_EN && op.ops[i].mport == ACR_NORTH));
    return 0;
}

// EAST forwarding: interior tiles fan every class east (MSelEn 0xB = consume +
// broadcast + transit-east); the last tile does not forward EAST.
static int test_east_forward() {
    acr_oplist o = {};
    acr_portbook b = {};
    assert(acr_plan_chain(&o, &b, 4, 2, 4, /*ctrl_id*/ 0) == ACR_OK); // cols 2,3,4
    for (uint8_t c = 2; c <= 3; c++) {                                // interior
        assert(count_master(&o, c, ACR_EAST) == 1);
        assert(master_mselen(&o, c, ACR_EAST) == 0xB);
    }
    assert(count_master(&o, 4, ACR_EAST) == 0); // last tile: no further forward
    return 0;
}

// Forward slot table (arbiter ACR_ARB_CTRL). Every tile arms consume (pkt rowidx,
// mask 0x17, msel0) + broadcast (0x10/0x10/msel1); the spine head additionally
// arms transit-north (0x00/0x10/msel2). No tile arms an only-last / whole-row /
// transit-east slot anymore (the forward chain is uniform). Row index is 0.
static int test_forward_slot_table() {
    acr_state s = {};
    acr_portbook b = {};
    acr_oplist o = {};
    assert(acr_plan_row_add(&s, &o, &b, 2, 3, 2, 4, /*ctrl_id*/ 0, /*is_top*/ 1) ==
           ACR_OK); // head row 3, cols 2..4, rowidx 0
    const uint8_t rowidx = 0, head = 2, col_hi = 4;

    // Head (spine col 2, ingress SOUTH): consume + bcast + transit-north.
    const acr_op *hc = find_slot(&o, head, ACR_SOUTH, ACR_SLOT_CONSUME, ACR_ARB_CTRL);
    assert(hc && hc->pkt_id == rowidx && hc->mask == ACR_MASK_CONSUME && hc->msel == ACR_MSEL_CONSUME);
    const acr_op *hb = find_slot(&o, head, ACR_SOUTH, ACR_SLOT_BCAST, ACR_ARB_CTRL);
    assert(hb && hb->pkt_id == ACR_ID_BCAST && hb->mask == ACR_MASK_BCAST && hb->msel == ACR_MSEL_BCAST);
    const acr_op *tn = find_slot(&o, head, ACR_SOUTH, ACR_SLOT_TRANSIT_N, ACR_ARB_CTRL);
    assert(tn && tn->pkt_id == ACR_ID_TRANSIT && tn->mask == ACR_MASK_CLASS && tn->msel == ACR_MSEL_TRANSIT_N);

    // Interior (col 3, ingress WEST): consume + bcast, no transit-north/east.
    const acr_op *ic = find_slot(&o, 3, ACR_WEST, ACR_SLOT_CONSUME, ACR_ARB_CTRL);
    assert(ic && ic->pkt_id == rowidx && ic->mask == ACR_MASK_CONSUME);
    assert(find_slot(&o, 3, ACR_WEST, ACR_SLOT_BCAST, ACR_ARB_CTRL) != nullptr);
    assert(find_slot(&o, 3, ACR_WEST, ACR_SLOT_TRANSIT_N, ACR_ARB_CTRL) == nullptr);
    assert(find_slot(&o, 3, ACR_WEST, ACR_SLOT_TRANSIT_E, ACR_ARB_CTRL) == nullptr);

    // Last (col_hi, ingress WEST): consume + bcast only (no only-last/whole).
    const acr_op *lc = find_slot(&o, col_hi, ACR_WEST, ACR_SLOT_CONSUME, ACR_ARB_CTRL);
    assert(lc && lc->pkt_id == rowidx && lc->mask == ACR_MASK_CONSUME);
    assert(find_slot(&o, col_hi, ACR_WEST, ACR_SLOT_BCAST, ACR_ARB_CTRL) != nullptr);
    assert(count_slots_arb(&o, col_hi, ACR_ARB_CTRL) == 2);
    return 0;
}

// Return chain (arbiter ACR_ARB_RET): every column injects its CTRL-slave response
// (RET_LOCAL, accept-any pkt=0/mask=0/msel0); a column with an east neighbor also
// forwards the neighbor's merged responses on its EAST slave (RET_TRANSIT, msel1);
// the head merges the descending upper-spine responses on its NORTH slave
// (RET_NORTH, msel2) and drives them all SOUTH->VRET (SOUTH master mselen 0x7).
// Non-head interior drives WEST (mselen 0x3 local+transit); the last tile drives
// WEST (mselen 0x1 local only). Standalone (acr_plan_return_chain) matches the
// row_add-embedded chain.
static int test_return_chain() {
    acr_oplist o = {};
    acr_portbook b = {};
    assert(acr_plan_return_chain(&o, &b, 3, 2, 4, /*is_top*/ 0) == ACR_OK); // row 3, cols 2..4
    const uint8_t head = 2, col_hi = 4;

    // Local response slot on every column's CTRL slave.
    for (uint8_t c = 2; c <= 4; c++) {
        const acr_op *rl = find_slot(&o, c, ACR_CTRL, ACR_SLOT_RET_LOCAL, ACR_ARB_RET);
        assert(rl && rl->pkt_id == 0 && rl->mask == 0 && rl->msel == ACR_MSEL_RET_LOCAL);
    }
    // Transit slot on head + interior EAST slaves, not on the last tile.
    assert(find_slot(&o, head, ACR_EAST, ACR_SLOT_RET_TRANSIT, ACR_ARB_RET) != nullptr);
    assert(find_slot(&o, 3, ACR_EAST, ACR_SLOT_RET_TRANSIT, ACR_ARB_RET) != nullptr);
    assert(find_slot(&o, col_hi, ACR_EAST, ACR_SLOT_RET_TRANSIT, ACR_ARB_RET) == nullptr);
    // North merge slot on the head only.
    assert(find_slot(&o, head, ACR_NORTH, ACR_SLOT_RET_NORTH, ACR_ARB_RET) != nullptr);

    // Head descends SOUTH->VRET pulling {local,transit,north} = 0x7.
    assert(count_master(&o, head, ACR_SOUTH) == 1);
    assert(master_mselen(&o, head, ACR_SOUTH) == 0x7);
    // Interior merges WEST with {local,transit} = 0x3; last with {local} = 0x1.
    assert(count_master(&o, 3, ACR_WEST) == 1);
    assert(master_mselen(&o, 3, ACR_WEST) == 0x3);
    assert(count_master(&o, col_hi, ACR_WEST) == 1);
    assert(master_mselen(&o, col_hi, ACR_WEST) == 0x1);
    // The head does not drive WEST; the last tile does not drive SOUTH.
    assert(count_master(&o, head, ACR_WEST) == 0);
    assert(count_master(&o, col_hi, ACR_SOUTH) == 0);

    // Top row (is_top=1): head omits the RET_NORTH slot; SOUTH master pulls
    // {local, transit} only (0x3). Nothing descends from above a top head.
    acr_oplist ot = {};
    acr_portbook bt = {};
    assert(acr_plan_return_chain(&ot, &bt, 3, 2, 4, /*is_top*/ 1) == ACR_OK);
    assert(find_slot(&ot, head, ACR_NORTH, ACR_SLOT_RET_NORTH, ACR_ARB_RET) == nullptr);
    assert(count_master(&ot, head, ACR_SOUTH) == 1);
    assert(master_mselen(&ot, head, ACR_SOUTH) == 0x3);
    return 0;
}

static int test_spine_reuse() {
    acr_state s = {};
    acr_portbook b = {};
    acr_oplist o1 = {};
    assert(acr_plan_row_add(&s, &o1, &b, 2, 3, 2, 4, 0, /*is_top*/ 0) == ACR_OK); // rows 1..2 spine, head row 3
    // Pass-through rows 1,2 each get a forward (SOUTH->NORTH) and a return
    // (NORTH->SOUTH) CCT.
    assert(count_kind(&o1, ACR_OP_CCT) == 4);
    assert(has_cct(&o1, 2, 1, ACR_SOUTH, ACR_NORTH));
    assert(has_cct(&o1, 2, 1, ACR_NORTH, ACR_SOUTH));
    // Head row 3 sits on the spine column (2), so it emits its own NORTH climb
    // master (broadcast + transit, MSelEn 0x6).
    assert(count_master(&o1, 2, ACR_NORTH) == 1);
    assert(master_mselen(&o1, 2, ACR_NORTH) == 0x6);
    acr_oplist o2 = {};
    assert(acr_plan_row_add(&s, &o2, &b, 2, 3, 2, 4, 0, /*is_top*/ 0) == ACR_OK); // re-add same row
    assert(o2.n == 0);                                                            // idempotent: nothing new
    acr_oplist o3 = {};
    assert(acr_plan_row_add(&s, &o3, &b, 2, 5, 2, 4, 0, /*is_top*/ 1) == ACR_OK); // higher row reuses spine 1..3
    // Spine rows 3,4 are extended. Row 3 is an ALREADY-configured head: it climbs
    // via its own NORTH master (emitted in o1), so o3 emits NOTHING at row 3. Only
    // row 4 (a pure pass-through) gets CCTs: forward + return = 2.
    assert(count_kind(&o3, ACR_OP_CCT) == 2);
    assert(has_cct(&o3, 2, 4, ACR_SOUTH, ACR_NORTH));
    assert(has_cct(&o3, 2, 4, ACR_NORTH, ACR_SOUTH));
    for (int i = 0; i < o3.n; i++)
        assert(o3.ops[i].row != 3);
    return 0;
}

// A higher row reusing the shared spine must not re-emit anything at an
// already-configured head row: that head fans the spine up via the NORTH master
// (broadcast + transit) it emitted when it was first configured.
static int test_shared_head_fanout() {
    acr_state s = {};
    acr_portbook b = {};
    acr_oplist o1 = {};
    assert(acr_plan_row_add(&s, &o1, &b, 0, 3, 0, 1, 0, /*is_top*/ 0) == ACR_OK); // head row 3, spine rows 1,2
    // Head-in-shim-col emits its own NORTH master (arb0, MSelEn 0x6).
    int found1 = 0;
    for (int i = 0; i < o1.n; i++) {
        const acr_op *op = &o1.ops[i];
        if (op->kind == ACR_OP_MASTER_EN && op->mport == ACR_NORTH && op->col == 0 && op->row == 3) {
            assert(op->arbiter == 0);
            assert(op->mselen == 0x6);
            assert(op->keep_header == 1);
            found1 = 1;
        }
    }
    assert(found1);
    acr_oplist o2 = {};
    assert(acr_plan_row_add(&s, &o2, &b, 0, 5, 0, 1, 0, /*is_top*/ 1) == ACR_OK); // spine rows 3,4
    // No op at all at the shared head row 3 (neither CCT nor NORTH master).
    for (int i = 0; i < o2.n; i++)
        assert(o2.ops[i].row != 3);
    // Row 4 (not a head) stays a circuit pass-through in BOTH directions.
    assert(has_cct(&o2, 0, 4, ACR_SOUTH, ACR_NORTH));
    assert(has_cct(&o2, 0, 4, ACR_NORTH, ACR_SOUTH));
    // The new head row 5 emits its own NORTH master (broadcast + transit).
    assert(count_master(&o2, 0, ACR_NORTH) == 1);
    assert(master_mselen(&o2, 0, ACR_NORTH) == 0x6);
    return 0;
}

// Every op carries the fwd/ret fabric tag the emit layer turns into the
// CONTROLPAN-PMAP dir= field. The tag CANNOT be inferred from the port type --
// the return chain's slots live on CTRL/EAST slaves and its non-head master is
// WEST, all ports the forward chain uses too -- so the planner must carry it.
//
// The invariant the debug UI depends on: a slot and every master that pulls it
// must agree on is_ret. tile_switch_view buckets ports by dir before linking
// slots to masters, so a disagreement makes the pulling master vanish from the
// slot's section (the head's return SOUTH master is the visible case).
static int test_is_ret_tagging() {
    acr_state s = {};
    acr_portbook b = {};
    acr_oplist o = {};
    assert(acr_plan_row_add(&s, &o, &b, 2, 3, 2, 4, /*ctrl_id*/ 0, /*is_top*/ 0) == ACR_OK);

    int npairs = 0, nret = 0, nfwd = 0;
    for (int i = 0; i < o.n; i++) {
        const acr_op *op = &o.ops[i];
        assert(op->is_ret == 0 || op->is_ret == 1);
        if (op->kind == ACR_OP_SLOT || op->kind == ACR_OP_MASTER_EN) {
            // Slot/master ops derive the tag from the arbiter; the two fabrics
            // use disjoint arbiters by construction.
            assert(op->is_ret == (op->arbiter == ACR_ARB_RET));
            op->is_ret ? nret++ : nfwd++;
        }
        if (op->kind != ACR_OP_MASTER_EN)
            continue;
        // Every slot this master pulls must share its fabric tag.
        for (int j = 0; j < o.n; j++) {
            const acr_op *sl = &o.ops[j];
            if (sl->kind != ACR_OP_SLOT || sl->col != op->col || sl->row != op->row)
                continue;
            if (sl->arbiter != op->arbiter || !((op->mselen >> sl->msel) & 1u))
                continue;
            assert(sl->is_ret == op->is_ret);
            npairs++;
        }
    }
    assert(nfwd > 0 && nret > 0 && npairs > 0);

    // The head's return SOUTH master and the CTRL/EAST return slots it pulls are
    // the pair the port-based guess used to split apart: slots on CTRL/EAST (once
    // tagged fwd), master on the VRET spine (tagged ret).
    const acr_op *rl = find_slot(&o, 2, ACR_CTRL, ACR_SLOT_RET_LOCAL, ACR_ARB_RET);
    const acr_op *rt = find_slot(&o, 2, ACR_EAST, ACR_SLOT_RET_TRANSIT, ACR_ARB_RET);
    assert(rl && rl->is_ret == 1);
    assert(rt && rt->is_ret == 1);
    // Forward slots on the same tile stay fwd.
    const acr_op *fc = find_slot(&o, 2, ACR_SOUTH, ACR_SLOT_CONSUME, ACR_ARB_CTRL);
    assert(fc && fc->is_ret == 0);

    // Slave-enables ride with the slots on their port; circuit spine hops are
    // tagged by climb direction (SOUTH->NORTH forward, NORTH->SOUTH return).
    for (int i = 0; i < o.n; i++) {
        const acr_op *op = &o.ops[i];
        if (op->kind == ACR_OP_SLAVE_EN) {
            for (int j = 0; j < o.n; j++) {
                const acr_op *sl = &o.ops[j];
                if (sl->kind == ACR_OP_SLOT && sl->col == op->col && sl->row == op->row && sl->sport == op->sport)
                    assert(sl->is_ret == op->is_ret);
            }
        } else if (op->kind == ACR_OP_CCT) {
            assert(op->is_ret == (op->sport == ACR_NORTH && op->mport == ACR_SOUTH));
        }
    }
    return 0;
}

int main() {
    if (test_book() || test_slot_classes() || test_east_forward() || test_forward_slot_table() || test_return_chain() ||
        test_spine_reuse() || test_shared_head_fanout() || test_is_ret_tagging())
        return 1;
    printf("PASS\n");
    return 0;
}
