/* netboot supervisor internals (runs on A78 core 0 from DDR_LOW1). */
#ifndef NETBOOT_H
#define NETBOOT_H

#include <stddef.h>
#include <stdint.h>

#include "netboot_proto.h"

/* Memory map, doc/design/lwip_elf_bootloader.md section 3 (as measured on the
 * board): the design's "unused DDR_LOW1" at 0x8_0000_0000 is a MIRROR of the 2 GB
 * DDR_LOW0 on this board, so the supervisor lives at the top of DDR_LOW0 and apps
 * must stay below NB_APP_LIMIT. */
#define NB_APP_BASE 0x0ULL
#define NB_APP_LIMIT 0x78000000ULL    /* supervisor image starts here (build.sh) */
#define NB_STAGING_BASE 0x7A200000ULL /* after the 2 MB BD-ring block at 0x7A000000 */
#define NB_STAGING_SIZE (0x80000000ULL - NB_STAGING_BASE)

/* Largest LOAD_RUN accepted. An ELF up to NB_STAGING_SIZE is also kept in
 * staging (RERUN re-places it); a larger one is placed as it streams in and
 * RERUN answers NEED_RELOAD. */
#define NB_ELF_MAX 0x100000000ULL

/* Physical address -> pointer. The x86 unit test (test/streamload_x86.sh)
 * builds with NB_HOST_TEST and maps the 2 GB of DDR_LOW0 onto a malloc'd block. */
#ifdef NB_HOST_TEST
extern uint8_t *nb_host_mem;
#define NB_PHYS(a) ((uint8_t *)(nb_host_mem + (uint64_t)(a)))
#else
#define NB_PHYS(a) ((uint8_t *)(uintptr_t)(a))
#endif

#define NB_DEFAULT_TIMEOUT_MS NETBOOT_DEFAULT_MAX_RUN_MS /* per-run limit when the request says 0 */

/* time.c-free helpers in main.c */
uint64_t nb_now_ms(void);

/* apu.c: the app core, driven through the PLM (xilpm). */
int apu_init(void);
int apu_start(uint64_t entry);
int apu_kill(void);
int apu_available(void);
int apu_busy(void); /* app core woken and not stopped since */
int apu_parked(int32_t *rc); /* finished app sits in park.S, reusable without the PLM */
int apu_can_stop(void);      /* 1 a stop succeeded, 0 the PLM refused one, 2 none tried yet */

/* park.S: the patched _exit lands here */
void nb_park(void);

/* elfload.c */
int elf_place(const uint8_t *img, uint64_t size, uint64_t *entry, char *msg, size_t msglen);
int elf_patch_exit(const uint8_t *img, uint64_t size, uint64_t park_addr);

/* Streaming placement (elfload.c): LOAD_RUN bytes go straight to their
 * PT_LOAD destination as they arrive. The ELF and program headers are buffered
 * first and checked (every PT_LOAD inside NB_APP_LIMIT) before any byte is
 * written; bytes outside every PT_LOAD (section headers, symtab, strtab: the
 * .bss zeroing and the _exit patch need them) are cached; sha256 is updated
 * per segment and checked at the end. */
#define NB_STREAM_HDR_MAX 65536U /* ELF + program headers must lie in the first 64 KiB */
#define NB_STREAM_GAP_MAX 64U    /* file ranges outside every PT_LOAD */

struct nb_sha256_ctx {
    uint32_t h[8];
    uint8_t buf[64];
    uint32_t n;
    uint64_t len;
};

struct nb_gap { /* a file range outside every PT_LOAD; its cached tail is [coff, off+len) */
    uint64_t off, len;
    uint64_t coff, cpos; /* file offset of the first cached byte, its place in the cache */
};

struct nb_stream {
    uint64_t size; /* elf_size from the request */
    uint64_t got;
    uint8_t *stage; /* whole-file copy (elf_size <= NB_STAGING_SIZE) or NULL */
    uint8_t *cache; /* gap cache (used when stage is NULL) */
    uint64_t cache_size;
    int phase;    /* 0 headers, 1 placing, -1 rejected */
    int wrote;    /* something was written to the app's DDR (the old image is gone) */
    uint32_t hdr_len;
    uint64_t placed;
    unsigned ngap;
    struct nb_gap gap[NB_STREAM_GAP_MAX];
    struct nb_sha256_ctx sha;
    char err[200];
    uint8_t hdr[NB_STREAM_HDR_MAX] __attribute__((aligned(16)));
};

void elf_stream_begin(struct nb_stream *s, uint64_t size, uint8_t *stage, uint8_t *cache, uint64_t cache_size);
int elf_stream_feed(struct nb_stream *s, const uint8_t *data, uint64_t len); /* 0, -1 rejected (s->err) */
/* All bytes in: 0 placed (+ .bss zeroed), -1 malformed/short, -2 sha256 mismatch
 * (msg says "sha256 mismatch ... NEED_RELOAD"; check s->wrote: app image invalid). */
int elf_stream_finish(struct nb_stream *s, const uint8_t want_sha[32], uint64_t *entry, char *msg, size_t msglen);
int elf_stream_patch_exit(struct nb_stream *s, uint64_t park_addr);
/* Before the jump: Xil_DCacheFlushRange over every PT_LOAD, then I-cache invalidate. */
void elf_stream_sync(struct nb_stream *s);

/* sha256.c */
void nb_sha256(const uint8_t *data, uint64_t len, uint8_t out[32]);
void nb_sha256_init(struct nb_sha256_ctx *c);
void nb_sha256_update(struct nb_sha256_ctx *c, const uint8_t *data, uint64_t len);
void nb_sha256_final(struct nb_sha256_ctx *c, uint8_t out[32]);

/* supervisor state (main.c), read by server.c */
struct nb_state {
    uint32_t state;       /* enum netboot_state */
    uint64_t staged_size; /* bytes of a validated ELF in staging, 0 = none */
    uint64_t last_elf_size; /* last LOAD_RUN's ELF (> NB_STAGING_SIZE: streamed, no staging copy) */
    int app_valid;          /* 0: a streamed upload overwrote the app area and failed (NEED_RELOAD) */
    uint64_t entry;
    uint64_t run_start_ms;
    uint32_t timeout_ms;
    uint32_t runs;
};
extern struct nb_state g_nb;

/* Start (or restart) the staged image on the app core. 0 on success. */
int nb_launch(uint32_t timeout_ms, uint32_t *place_ms, char *msg, size_t msglen);
/* Streaming LOAD_RUN: free the app core before the first byte is placed (0 ok),
 * then start the image elf_stream_finish() completed. */
int nb_stream_prepare(char *msg, size_t msglen);
int nb_launch_streamed(struct nb_stream *s, uint64_t entry, uint32_t timeout_ms, uint32_t *place_ms, char *msg, size_t msglen);
int nb_kill(const char *why);

/* Admission (version >= 2): fill an ack for a request whose header just
 * arrived; for a forced run past its limit this is where the old app is
 * terminated. Returns the verdict. */
int nb_admit(uint32_t cmd, uint32_t flags, struct netboot_ack *ack, char *msg, size_t msglen);
void nb_fill_ack(struct netboot_ack *ack);

/* server.c */
int server_init(void);
void server_poll(void);

#endif
