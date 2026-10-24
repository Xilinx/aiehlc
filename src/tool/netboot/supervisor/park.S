/* App-core park routine (runs on A78 core 1, EL3, with the FINISHED app's MMU on).
 *
 * The supervisor patches the app's _exit into "ldr x16,#8; br x16; .quad nb_park"
 * (elfload.c), so an app that returns from main() lands here instead of spinning
 * in `b .`. This PDI lets the supervisor neither power down nor reset core 1
 * (design section 12, item 4), so a parked core is the only way to hand it a new
 * image without a JTAG re-bootstrap.
 *
 * Mailbox protocol (struct nb_mbox, apu.c), core 1 never writes `seq`:
 *   +0  state  : 1 = parked             (written here)
 *   +4  rc     : exit status            (written here)
 *   +8  seq    : bumped by the supervisor to release the core
 *   +16 entry  : where to go when released
 * Core 1 only ever writes state/rc and cleans them to PoC, so a dirty line lost
 * to the next app's set/way invalidate cannot re-trigger a release.
 */
    .section .text.nb_park, "ax"
    .global nb_park
    .balign 64
nb_park:
    msr     daifset, #0xf                 /* nothing may interrupt us any more */
    adrp    x1, nb_mbox
    add     x1, x1, :lo12:nb_mbox
    ldr     w5, [x1, #8]                  /* release when seq differs from this */
    str     w0, [x1, #4]                  /* rc */
    mov     w2, #1
    str     w2, [x1, #0]                  /* state = parked */
    dc      civac, x1                     /* make it visible regardless of caches */
    dsb     sy
    sev
1:  wfe
    ldr     w2, [x1, #8]
    cmp     w2, w5
    b.eq    1b
    ldr     x3, [x1, #16]                 /* new entry, read while the MMU is still on */
    /* Hand the next image a reset-like core: MMU, D-cache and I-cache off.
     * Memory is identity-mapped, so execution continues at the same address. */
    mrs     x4, sctlr_el3
    bic     x4, x4, #1                    /* M */
    bic     x4, x4, #(1 << 2)             /* C */
    bic     x4, x4, #(1 << 12)            /* I */
    msr     sctlr_el3, x4
    isb
    ic      iallu
    tlbi    alle3
    dsb     sy
    isb
    br      x3
