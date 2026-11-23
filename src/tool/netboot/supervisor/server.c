/* TCP protocol (netboot_proto.h) on lwIP RAW. One request per connection.
 * The callbacks only move bytes; the work (SHA, placement, launch) runs from
 * server_poll() in the main loop, outside lwIP's call stack.
 *
 * Version 2: as soon as the 64 B header is in, server_poll() answers with a
 * netboot_ack (admitted / busy + time left / past its max / stuck), before any
 * payload, and closes the connection unless admitted. A second connection
 * while one is in progress gets a SERVING ack instead of a reset.
 *
 * LOAD_RUN streams (elfload.c): once the request is admitted (version 2) or its
 * header is in (version 1) the app core is freed, and every payload segment is
 * hashed and placed at its PT_LOAD address as it arrives -- an ELF need not fit
 * in staging. One up to NB_STAGING_SIZE is also copied there for RERUN. A
 * sha256 mismatch answers rc -2 "sha256 mismatch ... NEED_RELOAD" and marks the
 * app area invalid. The wire format (netboot_proto.h) is unchanged. */
#include <stdio.h>
#include <string.h>

#include "lwip/tcp.h"
#include "netboot.h"
#include "xil_printf.h"

struct conn {
    struct tcp_pcb *pcb;
    struct netboot_req req;
    uint32_t hdr_got;
    uint64_t payload_got;
    uint64_t t_first_ms;
    int ready; /* request complete, waiting for server_poll */
    int acked; /* version 2: admission ack sent */
    int streaming; /* LOAD_RUN: s_stream begun, payload is being placed */
    int hdr_bad;   /* header_ok() failed */
    int prep_rc;   /* nb_stream_prepare() failed: answer with this rc */
    uint64_t dropped; /* payload bytes that arrived before the stream could begin */
};

static struct conn s_c;
static int s_busy;
static struct nb_stream s_stream;
static char s_prep_msg[256];

static void conn_reset(void) {
    memset(&s_c, 0, sizeof(s_c));
    s_busy = 0;
}

static void send_ack(struct tcp_pcb *pcb, const struct netboot_ack *ack0, const char *msg) {
    struct netboot_ack ack = *ack0;
    ack.msg_len = msg ? (uint32_t)strlen(msg) : 0;
    tcp_write(pcb, &ack, sizeof(ack), TCP_WRITE_FLAG_COPY);
    if (ack.msg_len)
        tcp_write(pcb, msg, (u16_t)ack.msg_len, TCP_WRITE_FLAG_COPY);
    tcp_output(pcb);
}

static void close_pcb(struct tcp_pcb *pcb) {
    tcp_arg(pcb, NULL);
    tcp_recv(pcb, NULL);
    tcp_err(pcb, NULL);
    if (tcp_close(pcb) != ERR_OK)
        tcp_abort(pcb);
}

static void reply(int32_t rc, uint32_t rx_ms, uint32_t place_ms, const char *msg) {
    struct netboot_rep rep;
    memset(&rep, 0, sizeof(rep));
    rep.magic = NETBOOT_REP_MAGIC;
    rep.rc = rc;
    rep.bytes_rx = s_c.payload_got;
    rep.rx_ms = rx_ms;
    rep.place_ms = place_ms;
    rep.entry = g_nb.entry;
    rep.state = g_nb.state;
    rep.run_ms = g_nb.state == NETBOOT_ST_RUNNING ? (uint32_t)(nb_now_ms() - g_nb.run_start_ms) : 0;
    rep.runs = g_nb.runs;
    rep.msg_len = msg ? (uint32_t)strlen(msg) : 0;
    struct tcp_pcb *pcb = s_c.pcb;
    if (pcb) {
        tcp_write(pcb, &rep, sizeof(rep), TCP_WRITE_FLAG_COPY);
        if (rep.msg_len)
            tcp_write(pcb, msg, (u16_t)rep.msg_len, TCP_WRITE_FLAG_COPY);
        tcp_output(pcb);
        close_pcb(pcb);
    }
    conn_reset();
}

static void on_err(void *arg, err_t err) {
    (void)arg;
    xil_printf("[netboot] connection error %d, request dropped\r\n", (int)err);
    conn_reset(); /* lwIP already freed the pcb */
}

static int header_ok(char *msg, size_t n) {
    const struct netboot_req *r = &s_c.req;
    if (r->magic != NETBOOT_REQ_MAGIC || r->version < 1 || r->version > NETBOOT_VERSION) {
        snprintf(msg, n, "bad magic/version");
        return 0;
    }
    if (r->cmd == NETBOOT_CMD_LOAD_RUN && (r->elf_size == 0 || r->elf_size > NB_ELF_MAX)) {
        snprintf(msg, n, "elf_size %llu outside 1..%llu", (unsigned long long)r->elf_size,
                 (unsigned long long)NB_ELF_MAX);
        return 0;
    }
    return 1;
}

/* LOAD_RUN admitted and the app core free: from now on payload is placed. */
static void stream_start(void) {
    uint64_t n = s_c.req.elf_size;
    int keep = n <= NB_STAGING_SIZE; /* RERUN needs the copy */
    g_nb.staged_size = 0;            /* staging is about to be overwritten */
    g_nb.last_elf_size = n;
    elf_stream_begin(&s_stream, n, keep ? NB_PHYS(NB_STAGING_BASE) : NULL, keep ? NULL : NB_PHYS(NB_STAGING_BASE),
                     NB_STAGING_SIZE);
    s_c.streaming = 1;
}

/* Version 1 has no admission step: free the core and begin as soon as the
 * header is in, before the first payload byte (it may share the segment). */
static void on_header(void) {
    static char msg[160];
    s_c.hdr_bad = !header_ok(msg, sizeof(msg));
    if (s_c.hdr_bad || s_c.req.version >= 2 || s_c.req.cmd != NETBOOT_CMD_LOAD_RUN)
        return;
    s_c.prep_rc = nb_stream_prepare(s_prep_msg, sizeof(s_prep_msg));
    if (s_c.prep_rc == 0)
        stream_start();
}

static err_t on_recv(void *arg, struct tcp_pcb *pcb, struct pbuf *p, err_t err) {
    if (!p) { /* peer closed before the request completed */
        if (!s_c.ready) {
            tcp_close(pcb);
            conn_reset();
        }
        return ERR_OK;
    }
    if (s_c.t_first_ms == 0)
        s_c.t_first_ms = nb_now_ms();
    for (struct pbuf *q = p; q; q = q->next) {
        const uint8_t *src = (const uint8_t *)q->payload;
        uint32_t len = q->len;
        if (s_c.hdr_got < sizeof(s_c.req)) {
            uint32_t take = sizeof(s_c.req) - s_c.hdr_got;
            take = take < len ? take : len;
            memcpy((uint8_t *)&s_c.req + s_c.hdr_got, src, take);
            s_c.hdr_got += take, src += take, len -= take;
            if (s_c.hdr_got == sizeof(s_c.req))
                on_header();
        }
        if (len && s_c.hdr_got == sizeof(s_c.req) && s_c.req.cmd == NETBOOT_CMD_LOAD_RUN) {
            uint64_t room = s_c.req.elf_size - s_c.payload_got;
            uint32_t take = (uint32_t)(len < room ? len : room);
            if (s_c.streaming) {
                if (elf_stream_feed(&s_stream, src, take) != 0)
                    s_c.ready = 1; /* rejected (headers, bounds): answer now */
            } else {
                s_c.dropped += take;
            }
            s_c.payload_got += take;
        }
    }
    tcp_recved(pcb, p->tot_len);
    pbuf_free(p);
    if (s_c.hdr_got == sizeof(s_c.req) && (s_c.req.cmd != NETBOOT_CMD_LOAD_RUN || s_c.payload_got >= s_c.req.elf_size ||
                                           s_c.hdr_bad || s_c.prep_rc))
        s_c.ready = 1;
    return ERR_OK;
}

/* Another client's request is still in progress (one at a time): tell this one
 * when to come back instead of resetting it. A version-1 client reads this as a
 * bad reply magic, which it already treated as an error. */
static void refuse_serving(struct tcp_pcb *pcb) {
    static char msg[160];
    struct netboot_ack ack;
    nb_fill_ack(&ack);
    ack.verdict = NETBOOT_ACK_SERVING;
    uint64_t age = s_c.t_first_ms ? nb_now_ms() - s_c.t_first_ms : 0;
    ack.retry_after_ms = 500U;
    snprintf(msg, sizeof(msg), "supervisor busy with another client's request (%llu ms in): retry shortly",
             (unsigned long long)age);
    send_ack(pcb, &ack, msg);
    close_pcb(pcb);
}

static err_t on_accept(void *arg, struct tcp_pcb *pcb, err_t err) {
    if (err != ERR_OK || !pcb)
        return ERR_VAL;
    if (s_busy) { /* one request at a time */
        refuse_serving(pcb);
        return ERR_OK;
    }
    conn_reset();
    s_busy = 1;
    s_c.pcb = pcb;
    tcp_arg(pcb, &s_c);
    tcp_recv(pcb, on_recv);
    tcp_err(pcb, on_err);
    return ERR_OK;
}

int server_init(void) {
    struct tcp_pcb *pcb = tcp_new_ip_type(IPADDR_TYPE_ANY);
    if (!pcb)
        return -1;
    if (tcp_bind(pcb, IP_ANY_TYPE, NETBOOT_PORT) != ERR_OK)
        return -2;
    pcb = tcp_listen(pcb);
    if (!pcb)
        return -3;
    tcp_accept(pcb, on_accept);
    return 0;
}

static void handle_load_run(uint32_t rx_ms) {
    static char msg[256];
    if (s_c.prep_rc) {
        reply(s_c.prep_rc, rx_ms, 0, s_prep_msg);
        return;
    }
    if (!s_c.streaming || s_c.dropped) {
        reply(-1, rx_ms, 0, "payload arrived before the request was admitted (version 2: wait for the ack)");
        return;
    }
    uint64_t t0 = nb_now_ms(), entry = 0;
    int rc = elf_stream_finish(&s_stream, s_c.req.sha256, &entry, msg, sizeof(msg));
    if (rc != 0) {
        g_nb.staged_size = 0;
        if (s_stream.wrote)
            g_nb.app_valid = 0; /* the old image is partly overwritten */
        xil_printf("[netboot] LOAD_RUN %llu B rejected, rc %d: %s\r\n", (unsigned long long)s_c.req.elf_size,
                   rc == -2 ? -2 : -3, msg);
        reply(rc == -2 ? -2 : -3, rx_ms, (uint32_t)(nb_now_ms() - t0), msg);
        return;
    }
    g_nb.staged_size = s_stream.stage ? s_c.req.elf_size : 0;
    xil_printf("[netboot] LOAD_RUN %llu B received and placed in %u ms (stream=1, staging copy %s), sha256 ok\r\n",
               (unsigned long long)s_c.req.elf_size, rx_ms, s_stream.stage ? "kept" : "none");
    uint32_t place_ms = (uint32_t)(nb_now_ms() - t0);
    rc = nb_launch_streamed(&s_stream, entry, s_c.req.max_run_ms, &place_ms, msg, sizeof(msg));
    reply(rc, rx_ms, place_ms, msg);
}

/* Version 2: answer the header before the payload. 1 = admitted, go on. */
static int admit(void) {
    static char msg[256];
    struct netboot_ack ack;
    if (!header_ok(msg, sizeof(msg))) {
        nb_fill_ack(&ack);
        ack.verdict = NETBOOT_ACK_REJECT;
    } else {
        nb_admit(s_c.req.cmd, s_c.req.flags, &ack, msg, sizeof(msg));
        if (ack.verdict == NETBOOT_ACK_OK && s_c.req.cmd == NETBOOT_CMD_LOAD_RUN) {
            int rc = nb_stream_prepare(msg, sizeof(msg)); /* before the first byte is placed */
            if (rc != 0) {
                nb_fill_ack(&ack);
                ack.verdict = rc == -5 ? NETBOOT_ACK_STUCK : NETBOOT_ACK_REJECT;
            } else {
                stream_start();
            }
        }
    }
    send_ack(s_c.pcb, &ack, msg);
    s_c.acked = 1;
    if (ack.verdict == NETBOOT_ACK_OK) {
        s_c.t_first_ms = nb_now_ms(); /* rx_ms: the payload, not the admission */
        return 1;
    }
    xil_printf("[netboot] request (cmd %u) not admitted, verdict %d: %s\r\n", (unsigned)s_c.req.cmd, (int)ack.verdict,
               msg);
    close_pcb(s_c.pcb);
    conn_reset();
    return 0;
}

void server_poll(void) {
    static char msg[256];
    if (!s_busy)
        return;
    if (!s_c.acked && s_c.hdr_got == sizeof(s_c.req) && s_c.req.version >= 2 && !admit())
        return;
    if (!s_c.ready)
        return;
    if (!header_ok(msg, sizeof(msg))) {
        reply(-1, 0, 0, msg);
        return;
    }
    uint32_t rx_ms = (uint32_t)(nb_now_ms() - s_c.t_first_ms);
    switch (s_c.req.cmd) {
    case NETBOOT_CMD_LOAD_RUN:
        handle_load_run(rx_ms);
        break;
    case NETBOOT_CMD_RERUN: {
        uint32_t place_ms = 0;
        if (g_nb.staged_size) {
            int rc = nb_launch(s_c.req.max_run_ms, &place_ms, msg, sizeof(msg));
            reply(rc, 0, place_ms, msg);
        } else if (g_nb.last_elf_size > NB_STAGING_SIZE) {
            snprintf(msg, sizeof(msg),
                     "NEED_RELOAD: the last ELF (%llu B) is larger than staging (%llu B); it was placed as it "
                     "streamed in and no copy was kept: send LOAD_RUN again",
                     (unsigned long long)g_nb.last_elf_size, (unsigned long long)NB_STAGING_SIZE);
            xil_printf("[netboot] RERUN refused: %s\r\n", msg);
            reply(-3, 0, 0, msg);
        } else {
            reply(-3, 0, 0, "nothing staged; send LOAD_RUN first (NEED_RELOAD)");
        }
        break;
    }
    case NETBOOT_CMD_KILL:
        reply(nb_kill("KILL command"), 0, 0, "app core powered down");
        break;
    case NETBOOT_CMD_STATUS: {
        int32_t xrc = 0;
        int parked = apu_parked(&xrc);
        snprintf(msg, sizeof(msg), "pm %s, staged %llu B, app core %s", apu_available() ? "ok" : "UNAVAILABLE",
                 (unsigned long long)g_nb.staged_size,
                 parked       ? "parked"
                 : apu_busy() ? "busy"
                              : "free");
        size_t n = strlen(msg);
        if (parked)
            snprintf(msg + n, sizeof(msg) - n, " (exit rc %d)", (int)xrc);
        else if (apu_busy() && g_nb.runs)
            snprintf(msg + n, sizeof(msg) - n, " (run %u: %llu ms of its %u ms max)", (unsigned)g_nb.runs,
                     (unsigned long long)(nb_now_ms() - g_nb.run_start_ms), (unsigned)g_nb.timeout_ms);
        n = strlen(msg);
        if (g_nb.last_elf_size && !g_nb.app_valid && (parked || !apu_busy()))
            snprintf(msg + n, sizeof(msg) - n, "; app area invalid (NEED_RELOAD)");
    }
        reply(0, 0, 0, msg);
        break;
    default:
        reply(-1, 0, 0, "unknown command");
    }
}
