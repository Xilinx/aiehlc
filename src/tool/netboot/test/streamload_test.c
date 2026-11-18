/* x86 unit test of the streaming netboot loader (run by streamload_x86.sh).
 *
 * DDR_LOW0 (2 GB) is a malloc'd block on the host: elfload.c built with
 * NB_HOST_TEST maps physical addresses onto nb_host_mem. old_elf_place /
 * old_elf_patch_exit are elfload.c from ed72954 (the pre-streaming supervisor),
 * built by the script with its address casts rewritten to nb_host_mem.
 *
 *   streamload_test ident ELF...       old placement vs streamed placement (staging copy
 *                                      and gap-cache modes, random segment sizes), whole
 *                                      2 GB memcmp each
 *   streamload_test serve K=V...       requests captured from netload.py replayed through
 *                                      the real server.c + launch.c (lwIP stubbed)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "lwip/tcp.h"
#include "netboot.h"
#include "xil_cache.h"

#define AREA 0x80000000ULL /* DDR_LOW0 */
#define PARK 0x78001234ULL /* stand-in nb_park address for the direct comparisons */

int old_elf_place(const uint8_t *img, uint64_t size, uint64_t *entry, char *msg, size_t msglen);
int old_elf_patch_exit(const uint8_t *img, uint64_t size, uint64_t park_addr);

uint8_t *nb_host_mem;
struct nb_state g_nb;

/* ------------------------------------------------------------ board stubs */
static uint64_t s_ms;
uint64_t nb_now_ms(void) { return ++s_ms; }
static int s_busy_core, s_parked;
static uint64_t s_started;
static unsigned s_icache, s_dflush;
int apu_init(void) { return 0; }
int apu_start(uint64_t entry) {
    s_started = entry;
    s_busy_core = 1;
    s_parked = 0;
    return 0;
}
int apu_kill(void) {
    s_busy_core = 0;
    s_parked = 0;
    return 0;
}
int apu_available(void) { return 1; }
int apu_busy(void) { return s_busy_core; }
int apu_parked(int32_t *rc) {
    if (rc)
        *rc = 0;
    return s_parked;
}
int apu_can_stop(void) { return 1; }
void nb_park(void) {}
void Xil_DCacheFlushRange(INTPTR adr, INTPTR len) { (void)adr, (void)len, s_dflush++; }
void Xil_DCacheFlush(void) {}
void Xil_ICacheInvalidate(void) { s_icache++; }

/* ------------------------------------------------------------ lwIP stubs */
static tcp_accept_fn s_accept;
static tcp_recv_fn s_recv;
static void *s_arg;
static uint8_t s_out[1 << 16];
static size_t s_outlen;
static struct tcp_pcb s_listen;
struct tcp_pcb *tcp_new_ip_type(u8_t type) { return (void)type, &s_listen; }
err_t tcp_bind(struct tcp_pcb *pcb, const void *ip, u16_t port) { return (void)pcb, (void)ip, (void)port, ERR_OK; }
struct tcp_pcb *tcp_listen(struct tcp_pcb *pcb) { return pcb; }
void tcp_accept(struct tcp_pcb *pcb, tcp_accept_fn fn) { (void)pcb, s_accept = fn; }
void tcp_arg(struct tcp_pcb *pcb, void *arg) { (void)pcb, s_arg = arg; }
void tcp_recv(struct tcp_pcb *pcb, tcp_recv_fn fn) { (void)pcb, s_recv = fn; }
void tcp_err(struct tcp_pcb *pcb, tcp_err_fn fn) { (void)pcb, (void)fn; }
err_t tcp_write(struct tcp_pcb *pcb, const void *p, u16_t len, u8_t fl) {
    (void)pcb, (void)fl;
    if (s_outlen + len > sizeof(s_out))
        return ERR_MEM;
    memcpy(s_out + s_outlen, p, len);
    s_outlen += len;
    return ERR_OK;
}
err_t tcp_output(struct tcp_pcb *pcb) { return (void)pcb, ERR_OK; }
err_t tcp_close(struct tcp_pcb *pcb) { return pcb->closed = 1, ERR_OK; }
void tcp_abort(struct tcp_pcb *pcb) { pcb->closed = 1; }
void tcp_recved(struct tcp_pcb *pcb, u16_t len) { (void)pcb, (void)len; }
u8_t pbuf_free(struct pbuf *p) { return (void)p, 1; }

/* ------------------------------------------------------------ helpers */
static uint64_t s_rng = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void) {
    s_rng ^= s_rng << 13, s_rng ^= s_rng >> 7, s_rng ^= s_rng << 17;
    return (uint32_t)(s_rng >> 16);
}
/* a TCP-segment-like size: mostly MSS-sized, sometimes tiny, sometimes a big pbuf */
static uint32_t seg_len(void) {
    uint32_t r = rnd() % 16;
    if (r == 0)
        return 1 + rnd() % 64;
    if (r < 12)
        return 1460;
    return 1 + rnd() % 65535;
}

static uint8_t *load_file(const char *path, uint64_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    *size = (uint64_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*size ? *size : 1);
    if (!b || fread(b, 1, *size, f) != *size) {
        fprintf(stderr, "read %s failed\n", path);
        exit(2);
    }
    fclose(f);
    return b;
}

static uint8_t *area(void) {
    void *p = mmap(NULL, AREA, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        perror("mmap 2 GB");
        exit(2);
    }
    return p;
}
/* a non-zero background, so a missed .bss clear or a stray write shows up */
static void fill(uint8_t *a) { memset(a, 0xA5, AREA); }

static void stream_all(struct nb_stream *s, const uint8_t *img, uint64_t size) {
    for (uint64_t off = 0; off < size;) {
        uint64_t n = seg_len();
        if (n > size - off)
            n = size - off;
        if (elf_stream_feed(s, img + off, n) != 0)
            return;
        off += n;
    }
}

/* ------------------------------------------------------------ ident */
static int cmd_ident(int argc, char **argv) {
    uint8_t *A = area(), *B = area();
    uint8_t *stage = malloc(NB_STAGING_SIZE), *cache = malloc(NB_STAGING_SIZE);
    static struct nb_stream st;
    int same = 0, total = 0;
    for (int i = 0; i < argc; i++) {
        uint64_t size, e_old = 0, e_new = 0;
        uint8_t *img = load_file(argv[i], &size);
        char m_old[256], m_new[256];
        uint8_t sha[32];
        nb_sha256(img, size, sha);
        fill(A);
        nb_host_mem = A;
        int rc_old = old_elf_place(img, size, &e_old, m_old, sizeof(m_old));
        int p_old = rc_old == 0 ? old_elf_patch_exit(img, size, PARK) : -1;
        int ok = 1;
        const char *modes[2] = {"server", "gapcache"};
        for (int mode = 0; mode < 2; mode++) {
            fill(B);
            nb_host_mem = B;
            /* mode 0 as the server runs it (staging copy when it fits), mode 1 forces the gap cache */
            int keep = mode == 0 && size <= NB_STAGING_SIZE;
            elf_stream_begin(&st, size, keep ? stage : NULL, keep ? NULL : cache, NB_STAGING_SIZE);
            stream_all(&st, img, size);
            int rc_new = elf_stream_finish(&st, sha, &e_new, m_new, sizeof(m_new));
            int p_new = rc_new == 0 ? elf_stream_patch_exit(&st, PARK) : -1;
            unsigned ic = s_icache;
            if (rc_new == 0)
                elf_stream_sync(&st);
            int cmp = memcmp(A, B, AREA);
            int good = rc_old == 0 && rc_new == 0 && e_old == e_new && p_old == p_new && cmp == 0 && s_icache == ic + 1;
            printf("IDENT %s mode=%s size=%llu rc_old=%d rc_new=%d entry=0x%llx exit_patched=%d/%d memcmp(2GB)=%s %s\n",
                   argv[i], modes[mode], (unsigned long long)size, rc_old, rc_new, (unsigned long long)e_new, p_old,
                   p_new, cmp == 0 ? "equal" : "DIFFER", good ? "OK" : "FAIL");
            if (!good)
                printf("  old: %s\n  new: %s\n", m_old, m_new);
            ok &= good;
        }
        total++;
        same += ok;
        free(img);
    }
    printf("IDENT RESULT identical=%d/%d mismatched=%d\n", same, total, total - same);
    return same == total ? 0 : 1;
}

/* ------------------------------------------------------------ serve */
struct result {
    int version, acked, verdict, rc;
    char msg[512];
};

static void deliver(struct tcp_pcb *pcb, const uint8_t *p, uint64_t n) {
    /* 1-3 pbufs chained, as the GEM driver hands them over */
    struct pbuf q[3];
    uint8_t *copy = malloc(n);
    memcpy(copy, p, n);
    int parts = n >= 3 ? 1 + (int)(rnd() % 3) : 1;
    uint64_t off = 0;
    for (int k = 0; k < parts; k++) {
        uint64_t len = k == parts - 1 ? n - off : (n - off) / (uint64_t)(parts - k);
        q[k].payload = copy + off;
        q[k].len = (u16_t)len;
        q[k].next = k + 1 < parts ? &q[k + 1] : NULL;
        off += len;
    }
    q[0].tot_len = (u16_t)n;
    s_recv(s_arg, pcb, &q[0], ERR_OK);
    free(copy);
}

/* one connection: the request bytes as netload.py sent them */
static void run_request(const uint8_t *req, uint64_t n, struct result *r) {
    struct tcp_pcb pcb = {0};
    memset(r, 0, sizeof(*r));
    s_outlen = 0;
    s_accept(NULL, &pcb, ERR_OK);
    struct netboot_req h;
    memcpy(&h, req, sizeof(h));
    r->version = (int)h.version;
    uint64_t off = 0;
    if (h.version >= 2) { /* the client waits for the ack before the payload */
        while (off < sizeof(h)) {
            uint64_t k = 1 + rnd() % 64;
            k = k > sizeof(h) - off ? sizeof(h) - off : k;
            deliver(&pcb, req + off, k);
            off += k;
        }
        server_poll();
        struct netboot_ack ack;
        if (s_outlen < sizeof(ack)) {
            snprintf(r->msg, sizeof(r->msg), "no ack");
            r->rc = 999;
            return;
        }
        memcpy(&ack, s_out, sizeof(ack));
        r->acked = ack.magic == NETBOOT_ACK_MAGIC;
        r->verdict = ack.verdict;
        if (ack.verdict != NETBOOT_ACK_OK) {
            snprintf(r->msg, sizeof(r->msg), "%.*s", (int)ack.msg_len, (const char *)s_out + sizeof(ack));
            r->rc = 998;
            return;
        }
        memmove(s_out, s_out + sizeof(ack) + ack.msg_len, s_outlen - sizeof(ack) - ack.msg_len);
        s_outlen -= sizeof(ack) + ack.msg_len;
    }
    while (off < n && !pcb.closed) {
        uint64_t k = seg_len();
        k = k > n - off ? n - off : k;
        deliver(&pcb, req + off, k);
        off += k;
        server_poll(); /* the main loop polls between segments */
    }
    server_poll();
    struct netboot_rep rep;
    if (s_outlen < sizeof(rep)) {
        snprintf(r->msg, sizeof(r->msg), "no reply");
        r->rc = 997;
        return;
    }
    memcpy(&rep, s_out, sizeof(rep));
    if (rep.magic != NETBOOT_REP_MAGIC) {
        snprintf(r->msg, sizeof(r->msg), "bad reply magic 0x%x", rep.magic);
        r->rc = 996;
        return;
    }
    r->rc = rep.rc;
    snprintf(r->msg, sizeof(r->msg), "%.*s", (int)rep.msg_len, (const char *)s_out + sizeof(rep));
}

static const char *arg(int argc, char **argv, const char *key) {
    size_t k = strlen(key);
    for (int i = 0; i < argc; i++)
        if (strncmp(argv[i], key, k) == 0 && argv[i][k] == '=')
            return argv[i] + k + 1;
    fprintf(stderr, "missing %s=\n", key);
    exit(2);
}

/* the app region [0, NB_APP_LIMIT) after the server placed an ELF, vs the old loader */
static int same_as_old(uint8_t *A, const char *elf) {
    uint64_t size, e;
    char m[256];
    uint8_t *img = load_file(elf, &size);
    uint8_t *B = nb_host_mem;
    fill(A);
    nb_host_mem = A;
    int rc = old_elf_place(img, size, &e, m, sizeof(m));
    if (rc == 0)
        old_elf_patch_exit(img, size, (uint64_t)(uintptr_t)nb_park);
    nb_host_mem = B;
    free(img);
    return rc == 0 && memcmp(A, B, NB_APP_LIMIT) == 0;
}

static void clear_app(void) { memset(nb_host_mem, 0xA5, NB_APP_LIMIT); }

static void app_exits(void) { /* the app returned through the patched _exit */
    s_parked = 1;
    g_nb.state = NETBOOT_ST_EXITED;
}

#define CHECK(name, cond)                                                                                              \
    do {                                                                                                               \
        int c_ = (cond);                                                                                               \
        printf("SERVE %-34s %s  (v%d rc=%d) %s\n", name, c_ ? "OK" : "FAIL", r.version, r.rc, r.msg);                \
        fails += !c_;                                                                                                  \
    } while (0)

static int cmd_serve(int argc, char **argv) {
    uint8_t *A = area();
    nb_host_mem = area();
    fill(nb_host_mem);
    server_init();
    struct result r;
    int fails = 0;
    uint64_t n;
    uint8_t *q;
    const char *elf = arg(argc, argv, "elf"), *big = arg(argc, argv, "bigelf");

    q = load_file(arg(argc, argv, "v1"), &n);
    clear_app();
    run_request(q, n, &r);
    free(q);
    int v1 = r.rc == 0 && s_started == g_nb.entry && same_as_old(A, elf);
    CHECK("v1 LOAD_RUN placed == old", v1);
    app_exits();

    q = load_file(arg(argc, argv, "rerun2"), &n);
    clear_app();
    run_request(q, n, &r);
    CHECK("v2 RERUN of a staged ELF", r.rc == 0 && r.acked && same_as_old(A, elf));
    app_exits();
    free(q);

    q = load_file(arg(argc, argv, "v2"), &n);
    clear_app();
    run_request(q, n, &r);
    free(q);
    int v2 = r.rc == 0 && r.acked && r.verdict == 0 && same_as_old(A, elf);
    CHECK("v2 LOAD_RUN placed == old", v2);
    app_exits();

    q = load_file(arg(argc, argv, "corrupt"), &n);
    clear_app();
    run_request(q, n, &r);
    free(q);
    int corrupt = r.rc == -2 && strstr(r.msg, "sha256 mismatch") && strstr(r.msg, "NEED_RELOAD") &&
                  g_nb.app_valid == 0 && g_nb.staged_size == 0;
    CHECK("NETLOAD_CORRUPT_AT upload rejected", corrupt);
    q = load_file(arg(argc, argv, "status"), &n);
    clear_app();
    run_request(q, n, &r);
    free(q);
    corrupt &= r.rc == 0 && strstr(r.msg, "NEED_RELOAD") != NULL;
    CHECK("STATUS after it: app area invalid", r.rc == 0 && strstr(r.msg, "NEED_RELOAD"));
    q = load_file(arg(argc, argv, "rerun2"), &n);
    clear_app();
    run_request(q, n, &r);
    free(q);
    corrupt &= r.rc != 0 && strstr(r.msg, "NEED_RELOAD") != NULL;
    CHECK("RERUN after it refused", r.rc != 0 && strstr(r.msg, "NEED_RELOAD"));

    q = load_file(arg(argc, argv, "big"), &n);
    clear_app();
    run_request(q, n, &r);
    free(q);
    int bigok = r.rc == 0 && n - 64 > NB_STAGING_SIZE && g_nb.staged_size == 0 && same_as_old(A, big);
    CHECK("v2 LOAD_RUN > staging placed == old", bigok);
    app_exits();
    int rerun_big = bigok;
    const char *rr[2] = {"rerun2", "rerun1"};
    for (int k = 0; k < 2; k++) {
        q = load_file(arg(argc, argv, rr[k]), &n);
        clear_app();
    run_request(q, n, &r);
        free(q);
        int ok = r.rc != 0 && strstr(r.msg, "NEED_RELOAD") != NULL;
        CHECK(k ? "v1 RERUN after it: NEED_RELOAD" : "v2 RERUN after it: NEED_RELOAD", ok);
        rerun_big &= ok;
    }
    app_exits();

    /* bad requests: unknown magic (v1 path) and out-of-range version */
    uint8_t bad[64];
    memset(bad, 0, sizeof(bad));
    struct netboot_req *b = (struct netboot_req *)bad;
    b->magic = 0x12345678;
    b->version = 1;
    b->cmd = NETBOOT_CMD_STATUS;
    run_request(bad, sizeof(bad), &r);
    CHECK("bad magic rejected", r.rc == -1);

    printf("SERVE RESULT v1=%s v2=%s corrupt=%s rerun_big=%s fails=%d\n", v1 ? "ok" : "FAIL", v2 ? "ok" : "FAIL",
           corrupt ? "REJECT" : "ACCEPTED", rerun_big ? "NEED_RELOAD" : "FAIL", fails);
    return fails ? 1 : 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc >= 2 && strcmp(argv[1], "ident") == 0)
        return cmd_ident(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "serve") == 0)
        return cmd_serve(argc - 2, argv + 2);
    fprintf(stderr, "usage: %s ident ELF... | serve k=v...\n", argv[0]);
    return 2;
}
