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

/* sha256.c */
void nb_sha256(const uint8_t *data, uint64_t len, uint8_t out[32]);

/* supervisor state (main.c), read by server.c */
struct nb_state {
    uint32_t state;       /* enum netboot_state */
    uint64_t staged_size; /* bytes of a validated ELF in staging, 0 = none */
    uint64_t entry;
    uint64_t run_start_ms;
    uint32_t timeout_ms;
    uint32_t runs;
};
extern struct nb_state g_nb;

/* Start (or restart) the staged image on the app core. 0 on success. */
int nb_launch(uint32_t timeout_ms, uint32_t *place_ms, char *msg, size_t msglen);
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
