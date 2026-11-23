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
#include "xil_mmu.h"
#include "xil_printf.h"
#include "xiltimer.h"
#include "xparameters.h"
#include "xtimer_config.h"

struct nb_state g_nb;
static struct netif s_netif;
static XEmacPs *s_emacps; /* polled from the main loop: the supervisor takes no interrupts */
static xemacpsif_s *s_emacpsif;
static ip4_addr_t s_ip;

uint64_t nb_now_ms(void) {
    XTime t;
    XTime_GetTime(&t);
    return (uint64_t)t / (COUNTS_PER_SECOND / 1000U);
}

/* NETBOOT_IP etc. arrive as "a,b,c,d" from build.sh; the variadic hop expands
 * them before IP4_ADDR counts its arguments. */
#define NB_IP4(dst, ...) IP4_ADDR(dst, __VA_ARGS__)

/* NETBOOT_DMA_BASE (build.sh, VEK385): the GEM sits in the LPD, and on the
 * vek385 base platform LPD_AXI_NOC0 only reaches DDR_CH1 (0xA_0000_0000) and
 * DDR_CH2, not DDR_LOW0, so every buffer the GEM DMAs (lwIP heap and pools,
 * BD ring) is linked there. The BSP table has no entry for that 1 GB block,
 * so map it before lwIP touches its memory. Attribute = the BSP's own normal
 * write-back, outer-shareable DDR descriptor (0x605). The BD block one GB up
 * is mapped non-cacheable by the lwIP port itself. */
static void dma_map(void) {
#ifdef NETBOOT_DMA_BASE
    Xil_SetTlbAttributes((UINTPTR)NETBOOT_DMA_BASE, 0x605ULL);
    xil_printf("[netboot] GEM DMA memory at 0x%llx (LPD-reachable DDR)\r\n",
               (unsigned long long)NETBOOT_DMA_BASE);
#endif
}

/* Ethernet link (2026-11-10). The port's TI PHY setup gives up when
 * autonegotiation takes over 5 s (PHY_AUTONEG_TIMEOUT_DEFAULT), and right after
 * a power cycle the lab switch sometimes takes longer (1 of 3 bootstraps on 213:
 * "Auto negotiation timeout", "Phy setup failure init_emacps"). The port then
 * leaves the link DOWN for good -- its recovery, eth_link_detect(), runs from
 * link_detect_thread, which NO_SYS does not have -- and "ready" was printed on a
 * dead link. So the PHY setup is retried here, "ready" is printed only once the
 * link is up, and the main loop polls the link. The port's own recovery is not
 * reused: it applies phy_setup_emacps()'s result without checking XST_FAILURE. */
#define NB_PHY_TRIES 6 /* PHY setups in net_init, up to ~5 s each */
#define NB_BMSR 1      /* IEEE basic mode status register (xemac_ieee_reg.h is private to the port) */
#define NB_BMSR_LINK 0x0004U
#define NB_BMSR_ANEG_DONE 0x0020U
extern u32_t phyaddrforemac; /* xadapter.c: the PHY address init_emacps found */

static int link_up(void) {
    return s_emacpsif->eth_link_status == ETH_LINK_UP || s_emacpsif->eth_link_status == ETH_LINK_UNDEFINED;
}

static int phy_setup(void) {
    u32_t speed = phy_setup_emacps(s_emacps, phyaddrforemac);
    if (speed == (u32_t)XST_FAILURE)
        return 0;
    XEmacPs_SetOperatingSpeed(s_emacps, (u16)speed);
    s_emacpsif->eth_link_status = ETH_LINK_UP;
    netif_set_link_up(&s_netif);
    xil_printf("[netboot] Ethernet link up, %u Mb/s\r\n", (unsigned)speed);
    return 1;
}

static void say_ready(void) {
    static int said;
    if (said++)
        return;
    xil_printf("[netboot] ready: %d.%d.%d.%d port %d\r\n", ip4_addr1(&s_ip), ip4_addr2(&s_ip), ip4_addr3(&s_ip),
               ip4_addr4(&s_ip), NETBOOT_PORT);
}

/* Every second from the main loop: notice a lost link, and set the PHY up again
 * once it reports link + autonegotiation complete. */
static void link_poll(void) {
    u16 st;
    if (s_emacpsif->eth_link_status == ETH_LINK_UNDEFINED)
        return;
    XEmacPs_PhyRead(s_emacps, phyaddrforemac, NB_BMSR, &st);
    XEmacPs_PhyRead(s_emacps, phyaddrforemac, NB_BMSR, &st); /* the link bit latches low: read twice */
    if (link_up()) {
        if (!(st & NB_BMSR_LINK)) {
            s_emacpsif->eth_link_status = ETH_LINK_DOWN;
            netif_set_link_down(&s_netif);
            xil_printf("[netboot] Ethernet link lost; waiting for it to come back\r\n");
        }
        return;
    }
    if ((st & NB_BMSR_LINK) && (st & NB_BMSR_ANEG_DONE) && phy_setup())
        say_ready();
}

static int net_init(void) {
    ip4_addr_t ip, mask, gw;
    NB_IP4(&ip, NETBOOT_IP);
    NB_IP4(&mask, NETBOOT_MASK);
    NB_IP4(&gw, NETBOOT_GW);
    /* Xilinx OUI; one board per lab segment. */
    unsigned char mac[6] = {0x00, 0x0a, 0x35, 0x00, 0x1b, 0x55};
    dma_map();
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
    s_emacpsif = (xemacpsif_s *)((struct xemac_s *)s_netif.state)->state;
    s_emacps = &s_emacpsif->emacps;
    Xil_ExceptionDisable();
    netif_set_up(&s_netif);
    if (server_init() != 0) {
        xil_printf("[netboot] TCP server init failed\r\n");
        return -2;
    }
    s_ip = ip;
    int tries = NB_PHY_TRIES;
#ifdef NB_TEST_LINK_DOWN /* build.sh NETBOOT_TEST_LINK_DOWN: exercise the recovery on a good link */
    s_emacpsif->eth_link_status = ETH_LINK_DOWN;
    netif_set_link_down(&s_netif);
    tries = NB_TEST_LINK_DOWN == 2 ? 0 : tries; /* 2: leave it to link_poll */
    xil_printf("[netboot] TEST: Ethernet link forced down (NB_TEST_LINK_DOWN=%d)\r\n", NB_TEST_LINK_DOWN);
#endif
    for (int i = 1; !link_up() && i <= tries; i++) {
        xil_printf("[netboot] Ethernet link down: PHY setup retry %d/%d\r\n", i, NB_PHY_TRIES);
        phy_setup();
    }
    if (link_up())
        say_ready();
    else
        xil_printf("[netboot] NOT ready: Ethernet link down after %d PHY setup retries; "
                   "still polling it every second\r\n",
                   tries);
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
        link_poll();
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
