/* The app core (A78 core 1), driven through the PLM over IPI (xilpm_ng).
 * RequestWakeUp starts it at an address out of a real reset; ForcePowerDown
 * stops it whatever it is doing -- the watchdog's kill switch. */
#include "netboot.h"

#include "pm_api_sys.h"
#include "xil_printf.h"
#include "xipipsu.h"
#include "xparameters.h"
#include "xpm_defs.h"
#include "xpm_nodeid.h"

#ifndef NETBOOT_APP_CORE
#define NETBOOT_APP_CORE PM_DEV_ACPU_0_1
#endif

#ifndef NETBOOT_APP_CORE_RST
#define NETBOOT_APP_CORE_RST PM_RST_APU0_CORE1_WARM /* cluster 0, core 1 == PM_DEV_ACPU_0_1 */
#endif
#ifndef NETBOOT_APP_CORE_RST_COLD
#define NETBOOT_APP_CORE_RST_COLD PM_RST_APU0_CORE1_COLD
#endif

enum stop_how { STOP_NONE, STOP_POWER, STOP_RST_WARM, STOP_RST_COLD };

static XIpiPsu s_ipi;
static int s_ready;
static int s_powered;        /* woken by us and not stopped since */
static enum stop_how s_stop; /* how the last kill stopped it */
static int s_can_stop = 2;   /* 1 a PLM stop worked, 0 the PLM refused one, 2 none tried yet */

int apu_init(void) {
    XIpiPsu_Config *cfg = XIpiPsu_LookupConfig(XPAR_XIPIPSU_0_BASEADDR);
    if (!cfg)
        return -1;
    if (XIpiPsu_CfgInitialize(&s_ipi, cfg, cfg->BaseAddress) != XST_SUCCESS)
        return -2;
    if (XPm_InitXilpm(&s_ipi) != XST_SUCCESS)
        return -3;
    /* Own the core: without it the PLM answers XPM_PM_NO_ACCESS to its resets. */
    XStatus st = XPm_RequestNode(NETBOOT_APP_CORE, PM_CAP_ACCESS, XPM_DEF_QOS, REQUEST_ACK_BLOCKING);
    xil_printf("[netboot] RequestNode(app core) rc %d\r\n", (int)st);
    s_ready = 1;
    return 0;
}

int apu_available(void) { return s_ready; }

int apu_can_stop(void) { return s_can_stop; }

/* Shared with park.S (offsets 0/4/8/16 are hard-coded there). */
struct nb_mbox {
    volatile uint32_t state; /* 1 = parked */
    volatile int32_t rc;
    volatile uint32_t seq; /* written only by the supervisor */
    uint32_t pad;
    volatile uint64_t entry;
} __attribute__((aligned(64)));
struct nb_mbox nb_mbox;

int apu_parked(int32_t *rc) {
    if (!s_powered || nb_mbox.state != 1U)
        return 0;
    if (rc)
        *rc = nb_mbox.rc;
    return 1;
}

/* Send a parked app core to `entry`. No PLM involved: the core is already ours. */
static void apu_release(uint64_t entry) {
    nb_mbox.state = 0;
    nb_mbox.entry = entry;
    __asm__ volatile("dmb ish" ::: "memory");
    nb_mbox.seq = nb_mbox.seq + 1U;
    __asm__ volatile("dsb sy\n\tsev" ::: "memory");
}

int apu_busy(void) { return s_powered; }

/* Stop the app core whatever it is doing. Each PLM path is tried in turn and
 * logged, because which one is permitted is a property of the PDI's subsystem
 * definition. On the palmyra BOOT.BIN NONE is: ForcePowerDown -> 0x2, both core
 * resets -> XPM_PM_NO_ACCESS. (XPm_ReleaseNode is deliberately not used: it
 * returns success but leaves the core running.) */
int apu_kill(void) {
    if (!s_ready)
        return -1;
    if (!s_powered)
        return 0;
    XStatus fp = XPm_ForcePowerDown(NETBOOT_APP_CORE, REQUEST_ACK_BLOCKING);
    XStatus rw = XST_FAILURE, rc = XST_FAILURE;
    if (fp == XST_SUCCESS)
        s_stop = STOP_POWER;
    else if ((rw = XPm_ResetAssert(NETBOOT_APP_CORE_RST, PM_RESET_ACTION_ASSERT)) == XST_SUCCESS)
        s_stop = STOP_RST_WARM;
    else if ((rc = XPm_ResetAssert(NETBOOT_APP_CORE_RST_COLD, PM_RESET_ACTION_ASSERT)) == XST_SUCCESS)
        s_stop = STOP_RST_COLD;
    else {
        xil_printf("[netboot] app core stop REFUSED by PLM: powerdown %d, warm rst %d, cold rst %d\r\n", (int)fp,
                   (int)rw, (int)rc);
        s_can_stop = 0;
        return -1;
    }
    s_can_stop = 1;
    xil_printf("[netboot] app core stopped via %s\r\n", s_stop == STOP_POWER      ? "ForcePowerDown"
                                                        : s_stop == STOP_RST_WARM ? "warm reset"
                                                                                  : "cold reset");
    s_powered = 0;
    return 0;
}

int apu_start(uint64_t entry) {
    if (!s_ready)
        return -1;
    if (apu_parked(NULL)) {
        apu_release(entry);
        xil_printf("[netboot] app core was parked: released to 0x%llx (no PLM call)\r\n", (unsigned long long)entry);
        return 0;
    }
    if (s_powered && apu_kill() != 0)
        return -2;
    XStatus st = XPm_RequestWakeUp(NETBOOT_APP_CORE, 1, entry, REQUEST_ACK_BLOCKING);
    if (s_stop == STOP_RST_WARM || s_stop == STOP_RST_COLD) {
        /* The wake set the start address; let the core out of the reset we hold. */
        XStatus rel = XPm_ResetAssert(s_stop == STOP_RST_WARM ? NETBOOT_APP_CORE_RST : NETBOOT_APP_CORE_RST_COLD,
                                      PM_RESET_ACTION_RELEASE);
        xil_printf("[netboot] app core: wake %d, reset release %d\r\n", (int)st, (int)rel);
        if (rel != XST_SUCCESS)
            return (int)rel;
        st = XST_SUCCESS;
    }
    if (st != XST_SUCCESS) {
        xil_printf("[netboot] RequestWakeUp(app core, 0x%llx) failed: %d\r\n", (unsigned long long)entry, (int)st);
        return (int)st;
    }
    s_stop = STOP_NONE;
    s_powered = 1;
    return 0;
}
