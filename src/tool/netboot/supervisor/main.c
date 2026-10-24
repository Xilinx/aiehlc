/* netboot supervisor: stays resident on A78 core 0 (DDR_LOW1), receives ELFs
 * over TCP and runs each on core 1 with a watchdog.
 * Design: doc/design/lwip_elf_bootloader.md (mode D, section 7a). */
#include <stdio.h>
#include <string.h>

#include "lwip/init.h"
#include "lwip/ip4_frag.h"
#include "lwip/tcp.h"
#include "lwip/priv/tcp_priv.h"
#include "netboot.h"
#include "netif/etharp.h"
#include "netif/xadapter.h"
#include "netif/xemacpsif.h"
#include "xil_cache.h"
#include "xil_exception.h"
#include "xil_printf.h"
#include "xiltimer.h"
#include "xparameters.h"
#include "xtimer_config.h"

struct nb_state g_nb;
static struct netif s_netif;
static XEmacPs *s_emacps; /* polled from the main loop: the supervisor takes no interrupts */

uint64_t nb_now_ms(void) {
    XTime t;
    XTime_GetTime(&t);
    return (uint64_t)t / (COUNTS_PER_SECOND / 1000U);
}

int nb_kill(const char *why) {
    if (g_nb.state != NETBOOT_ST_RUNNING)
        return 0;
    int rc = apu_kill();
    if (rc == 0)
        g_nb.state = NETBOOT_ST_KILLED;
    xil_printf("\r\n[netboot] app core stopped (%s) after %u ms, rc %d\r\n", why,
               (unsigned)(nb_now_ms() - g_nb.run_start_ms), rc);
    return rc;
}

int nb_launch(uint32_t max_run_ms, uint32_t *place_ms, char *msg, size_t msglen) {
    if (!apu_available()) {
        snprintf(msg, msglen, "PLM/IPI unavailable: cannot drive the app core");
        return -10;
    }
    if (!apu_parked(NULL))
        nb_kill("replaced by a new run");
    if (apu_busy() && !apu_parked(NULL)) {
        snprintf(msg, msglen,
                 "app core still running and this PDI does not let the supervisor stop it "
                 "(ForcePowerDown refused, core reset NO_ACCESS) and it is not parked "
                 "(still running or hung): re-bootstrap the supervisor");
        return -5;
    }
    uint64_t t0 = nb_now_ms();
    uint64_t entry = 0;
    if (elf_place((const uint8_t *)(uintptr_t)NB_STAGING_BASE, g_nb.staged_size, &entry, msg, msglen) != 0) {
        g_nb.staged_size = 0;
        return -3;
    }
    int patched =
        elf_patch_exit((const uint8_t *)(uintptr_t)NB_STAGING_BASE, g_nb.staged_size, (uint64_t)(uintptr_t)nb_park);
    *place_ms = (uint32_t)(nb_now_ms() - t0);
    size_t n0 = strlen(msg);
    snprintf(msg + n0, msglen - n0, patched ? ", _exit -> park" : ", _exit NOT patched (core stays busy after exit)");
    g_nb.entry = entry;
    g_nb.runs++;
    g_nb.timeout_ms = max_run_ms ? max_run_ms : NB_DEFAULT_TIMEOUT_MS;
    /* Last supervisor line before the app owns the console: anything printed
     * after the wake interleaves with the app's own output on the shared UART. */
    xil_printf("[netboot] run %u: %s, start app core at 0x%llx (watchdog %u ms)\r\n", (unsigned)g_nb.runs, msg,
               (unsigned long long)entry, (unsigned)g_nb.timeout_ms);
    /* Design hazard (doc 7a): the app core's boot.S invalidates D-caches by
     * set/way; leave nothing of ours dirty when it starts. */
    Xil_DCacheFlush();
    int rc = apu_start(entry);
    if (rc != 0) {
        size_t n = strlen(msg);
        snprintf(msg + n, msglen - n, "; wake of app core FAILED (%d)", rc);
        g_nb.state = NETBOOT_ST_IDLE;
        return -4;
    }
    g_nb.state = NETBOOT_ST_RUNNING;
    g_nb.run_start_ms = nb_now_ms();
    return 0;
}

/* The app core as a client sees it: free (never started or stopped), parked
 * (returned from main through the patched _exit), or busy (running or hung). */
static uint32_t core_view(void) {
    if (apu_parked(NULL))
        return NETBOOT_CORE_PARKED;
    return apu_busy() ? NETBOOT_CORE_BUSY : NETBOOT_CORE_FREE;
}

/* How long the current (or last) app has run, its limit and what is left. */
void nb_fill_ack(struct netboot_ack *ack) {
    memset(ack, 0, sizeof(*ack));
    ack->magic = NETBOOT_ACK_MAGIC;
    ack->state = g_nb.state;
    ack->core = core_view();
    ack->max_run_ms = g_nb.timeout_ms;
    if (g_nb.runs && ack->core == NETBOOT_CORE_BUSY) {
        uint64_t now = nb_now_ms();
        ack->run_ms = now > g_nb.run_start_ms ? (uint32_t)(now - g_nb.run_start_ms) : 0;
        ack->remaining_ms = ack->run_ms < ack->max_run_ms ? ack->max_run_ms - ack->run_ms : 0;
    }
    ack->can_stop = (uint32_t)apu_can_stop();
    ack->runs = g_nb.runs;
    ack->park_addr = (uint32_t)(uintptr_t)nb_park;
}

/* Admission of a request whose header just arrived (protocol version >= 2), so
 * the client learns before sending a 15 MB ELF whether it will be run:
 *   core free / parked             -> OK
 *   app within its max_run_ms      -> BUSY, retry after the time it has left
 *   app past it, no FORCE          -> OVERTIME (resend with FORCE to preempt it)
 *   app past it, FORCE             -> terminate it (PLM) and OK, or STUCK when
 *                                     the PDI refuses (palmyra: design section 12,
 *                                     item 4) -- only a re-bootstrap frees the core
 * KILL and STATUS are always admitted. */
int nb_admit(uint32_t cmd, uint32_t flags, struct netboot_ack *ack, char *msg, size_t msglen) {
    nb_fill_ack(ack);
    msg[0] = 0;
    if (cmd != NETBOOT_CMD_LOAD_RUN && cmd != NETBOOT_CMD_RERUN)
        return ack->verdict = NETBOOT_ACK_OK;
    if (!apu_available()) {
        snprintf(msg, msglen, "PLM/IPI unavailable: cannot drive the app core");
        return ack->verdict = NETBOOT_ACK_REJECT;
    }
    if (ack->core != NETBOOT_CORE_BUSY)
        return ack->verdict = NETBOOT_ACK_OK;
    if (ack->remaining_ms > 0) {
        ack->retry_after_ms = ack->remaining_ms + 100U;
        snprintf(msg, msglen, "app core busy: run %u has run %u ms of its %u ms max, %u ms left", (unsigned)g_nb.runs,
                 (unsigned)ack->run_ms, (unsigned)ack->max_run_ms, (unsigned)ack->remaining_ms);
        return ack->verdict = NETBOOT_ACK_BUSY;
    }
    if (!(flags & NETBOOT_F_FORCE)) {
        snprintf(msg, msglen,
                 "app core busy past its max: run %u has run %u ms of its %u ms max; "
                 "resend with force to terminate it",
                 (unsigned)g_nb.runs, (unsigned)ack->run_ms, (unsigned)ack->max_run_ms);
        return ack->verdict = NETBOOT_ACK_OVERTIME;
    }
    uint32_t ran = ack->run_ms;
    g_nb.state = NETBOOT_ST_RUNNING; /* nb_kill only acts on a running app */
    if (nb_kill("forced: past its max run time") == 0) {
        nb_fill_ack(ack);
        snprintf(msg, msglen, "terminated run %u after %u ms (past its max)", (unsigned)g_nb.runs, (unsigned)ran);
        return ack->verdict = NETBOOT_ACK_OK;
    }
    g_nb.state = NETBOOT_ST_TIMEOUT;
    nb_fill_ack(ack);
    snprintf(msg, msglen,
             "run %u is past its max (%u ms) but this PDI does not let the supervisor stop the app core "
             "(PLM refuses power-down and reset): free it over JTAG (core 1: stop, rst -proc, pc=park_addr, con; "
             "apppaltest --netload does it) or re-bootstrap the supervisor",
             (unsigned)g_nb.runs, (unsigned)ran);
    return ack->verdict = NETBOOT_ACK_STUCK;
}

/* NETBOOT_IP etc. arrive as "a,b,c,d" from build.sh; the variadic hop expands
 * them before IP4_ADDR counts its arguments. */
#define NB_IP4(dst, ...) IP4_ADDR(dst, __VA_ARGS__)

static int net_init(void) {
    ip4_addr_t ip, mask, gw;
    NB_IP4(&ip, NETBOOT_IP);
    NB_IP4(&mask, NETBOOT_MASK);
    NB_IP4(&gw, NETBOOT_GW);
    /* Xilinx OUI; one board per lab segment. */
    unsigned char mac[6] = {0x00, 0x0a, 0x35, 0x00, 0x1b, 0x55};
    lwip_init();
    if (!xemac_add(&s_netif, &ip, &mask, &gw, mac, XPAR_XEMACPS_0_BASEADDR)) {
        xil_printf("[netboot] xemac_add failed\r\n");
        return -1;
    }
    netif_set_default(&s_netif);
    /* POLLED, no interrupts (design 7a): the GEM's own interrupt status is
     * serviced from the main loop (XEmacPs_IntrHandler), and IRQs stay masked
     * at the CPU. The GIC is shared with the app core and with debuggers: an
     * xsdb `rst -proc` of the app core was seen to leave GICD_CTLR with both
     * groups disabled and the GEM interrupt off, which silenced an
     * interrupt-driven supervisor for good (RX ring full, no ARP reply). */
    s_emacps = &((xemacpsif_s *)((struct xemac_s *)s_netif.state)->state)->emacps;
    Xil_ExceptionDisable();
    netif_set_up(&s_netif);
    if (server_init() != 0) {
        xil_printf("[netboot] TCP server init failed\r\n");
        return -2;
    }
    xil_printf("[netboot] ready: %d.%d.%d.%d port %d\r\n", ip4_addr1(&ip), ip4_addr2(&ip), ip4_addr3(&ip),
               ip4_addr4(&ip), NETBOOT_PORT);
    return 0;
}

/* A fault must say where it happened: the BSP default handler hangs silently,
 * which from the network looks exactly like a dead board. The label is fixed per
 * handler (no data pointer: the BSP's argument is not reliably ours). */
static void dump_fault(const char *what) {
    uint64_t esr, far, elr, sp;
    __asm__ volatile("mrs %0, esr_el3" : "=r"(esr));
    __asm__ volatile("mrs %0, far_el3" : "=r"(far));
    __asm__ volatile("mrs %0, elr_el3" : "=r"(elr));
    __asm__ volatile("mov %0, sp" : "=r"(sp));
    xil_printf("\r\n[netboot] FATAL %s: ESR_EL3 0x%llx (EC 0x%x) FAR_EL3 0x%llx ELR_EL3 0x%llx SP 0x%llx\r\n", what,
               (unsigned long long)esr, (unsigned)((esr >> 26) & 0x3f), (unsigned long long)far,
               (unsigned long long)elr, (unsigned long long)sp);
    for (;;)
        ;
}

static void on_sync_abort(void *unused) { dump_fault("synchronous abort"); }
static void on_serror(void *unused) { dump_fault("SError"); }

static void install_fault_handlers(void) {
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_SYNC_INT, (Xil_ExceptionHandler)on_sync_abort, NULL);
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_SERROR_ABORT_INT, (Xil_ExceptionHandler)on_serror, NULL);
}

static void timers(uint64_t now) {
    static uint64_t fast, slow, arp, rxchk;
    if (now - fast >= TCP_FAST_INTERVAL) {
        tcp_fasttmr();
        fast = now;
    }
    if (now - slow >= TCP_SLOW_INTERVAL) {
        tcp_slowtmr();
        slow = now;
    }
    if (now - arp >= ARP_TMR_INTERVAL) {
        etharp_tmr();
        ip_reass_tmr();
        arp = now;
    }
    if (now - rxchk >= 1000U) { /* GEM RX-hang workaround, as in the reference platform */
        xemacpsif_resetrx_on_no_rxdata(&s_netif);
        rxchk = now;
    }
}

/* Reads its own clock: the loop's `now` is taken before server_poll() may launch
 * a run, and `now - run_start_ms` on that stale value underflowed and fired the
 * watchdog a few ms into every run. */
static void watchdog(void) {
    int32_t rc;
    /* also after TIMEOUT: an app that could not be stopped may still finish */
    if ((g_nb.state == NETBOOT_ST_RUNNING || g_nb.state == NETBOOT_ST_TIMEOUT) && apu_parked(&rc)) {
        if (g_nb.state == NETBOOT_ST_TIMEOUT)
            xil_printf("\r\n[netboot] run %u exited after its max run time (%u ms)\r\n", (unsigned)g_nb.runs,
                       (unsigned)(nb_now_ms() - g_nb.run_start_ms));
        g_nb.state = NETBOOT_ST_EXITED; /* app returned from main and parked */
        return;
    }
    uint64_t now = nb_now_ms();
    if (g_nb.state != NETBOOT_ST_RUNNING || now < g_nb.run_start_ms || now - g_nb.run_start_ms <= g_nb.timeout_ms)
        return;
    if (nb_kill("watchdog: past its max run time") != 0) {
        xil_printf("[netboot] watchdog: run %u exceeded its max %u ms; app core cannot be stopped on this PDI, "
                   "a forced run will ask for a re-bootstrap\r\n",
                   (unsigned)g_nb.runs, (unsigned)g_nb.timeout_ms);
        g_nb.state = NETBOOT_ST_TIMEOUT;
    }
}

int main(void) {
    xil_printf("\r\n[netboot] supervisor (doc/design/lwip_elf_bootloader.md), staging 0x%llx\r\n",
               (unsigned long long)NB_STAGING_BASE);
    install_fault_handlers();
    int pm = apu_init();
    if (pm != 0)
        xil_printf("[netboot] WARNING: PLM/IPI init failed (%d): LOAD_RUN will be refused\r\n", pm);
    if (net_init() != 0)
        return -1;
    for (;;) {
        uint64_t now = nb_now_ms();
        timers(now);
        XEmacPs_IntrHandler(s_emacps); /* RX/TX completions and errors, polled */
        xemacif_input(&s_netif);
        server_poll();
        watchdog();
    }
}
