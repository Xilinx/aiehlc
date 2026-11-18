/* The app core's life cycle as the server drives it: free it, start a placed
 * image (staged for RERUN, streamed for LOAD_RUN), kill it, admit requests.
 * Split from main.c so test/streamload_x86.sh can build it on the host. */
#include <stdio.h>
#include <string.h>

#include "netboot.h"
#include "xil_cache.h"
#include "xil_printf.h"

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

/* The app core must be free before anything is placed over the old image. */
int nb_stream_prepare(char *msg, size_t msglen) {
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
    return 0;
}

/* Common tail of a launch: the image is placed, caches are clean. */
static int nb_start(uint64_t entry, uint32_t max_run_ms, int stream, uint64_t elf_size, char *msg, size_t msglen) {
    g_nb.entry = entry;
    g_nb.runs++;
    g_nb.timeout_ms = max_run_ms ? max_run_ms : NB_DEFAULT_TIMEOUT_MS;
    g_nb.app_valid = 1;
    /* Last supervisor line before the app owns the console: anything printed
     * after the wake interleaves with the app's own output on the shared UART. */
    xil_printf("[netboot] run %u: %s, start app core at 0x%llx (watchdog %u ms) stream=%d elf_size=%llu\r\n",
               (unsigned)g_nb.runs, msg, (unsigned long long)entry, (unsigned)g_nb.timeout_ms, stream,
               (unsigned long long)elf_size);
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

/* RERUN: re-place the ELF kept in staging. */
int nb_launch(uint32_t max_run_ms, uint32_t *place_ms, char *msg, size_t msglen) {
    int rc = nb_stream_prepare(msg, msglen);
    if (rc != 0)
        return rc;
    uint64_t t0 = nb_now_ms();
    uint64_t entry = 0;
    if (elf_place(NB_PHYS(NB_STAGING_BASE), g_nb.staged_size, &entry, msg, msglen) != 0) {
        g_nb.staged_size = 0;
        return -3;
    }
    int patched = elf_patch_exit(NB_PHYS(NB_STAGING_BASE), g_nb.staged_size, (uint64_t)(uintptr_t)nb_park);
    *place_ms = (uint32_t)(nb_now_ms() - t0);
    size_t n0 = strlen(msg);
    snprintf(msg + n0, msglen - n0, patched ? ", _exit -> park" : ", _exit NOT patched (core stays busy after exit)");
    return nb_start(entry, max_run_ms, 0, g_nb.staged_size, msg, msglen);
}

/* LOAD_RUN: the stream placed every PT_LOAD as it arrived and elf_stream_finish()
 * checked sha256 and zeroed .bss; patch _exit, clean the caches, jump. */
int nb_launch_streamed(struct nb_stream *s, uint64_t entry, uint32_t max_run_ms, uint32_t *place_ms, char *msg, size_t msglen) {
    uint64_t t0 = nb_now_ms();
    int patched = elf_stream_patch_exit(s, (uint64_t)(uintptr_t)nb_park);
    elf_stream_sync(s); /* Xil_DCacheFlushRange over every PT_LOAD, then I-cache invalidate */
    *place_ms += (uint32_t)(nb_now_ms() - t0);
    size_t n0 = strlen(msg);
    snprintf(msg + n0, msglen - n0, patched ? ", _exit -> park" : ", _exit NOT patched (core stays busy after exit)");
    return nb_start(entry, max_run_ms, 1, s->size, msg, msglen);
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

