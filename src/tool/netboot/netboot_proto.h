/* Wire protocol shared by the netboot supervisor (board) and netload.py (host).
 * Little endian, packed. Keep in step with netload.py's struct formats.
 *
 * One request per TCP connection:
 *   host  -> board  struct netboot_req (64 B)
 *   board -> host   struct netboot_ack (48 B + msg)    version >= 2 only, sent as
 *                   soon as the header is in, BEFORE any payload: is the request
 *                   admitted, and if not, how long until it would be
 *   host  -> board  payload (LOAD_RUN, only when the ack verdict is OK)
 *   board -> host   struct netboot_rep (48 B + msg)    only when admitted
 * A version-1 request gets no ack (the old exchange, unchanged). */
#ifndef NETBOOT_PROTO_H
#define NETBOOT_PROTO_H

#include <stdint.h>

#define NETBOOT_PORT 5155
#define NETBOOT_REQ_MAGIC 0x3154424EU /* 'NBT1' */
#define NETBOOT_REP_MAGIC 0x3152424EU /* 'NBR1' */
#define NETBOOT_ACK_MAGIC 0x3141424EU /* 'NBA1' */
#define NETBOOT_VERSION 2             /* the supervisor also accepts version 1 */

#define NETBOOT_DEFAULT_MAX_RUN_MS 60000U /* a run's limit when the request says 0 */

enum netboot_cmd {
    NETBOOT_CMD_LOAD_RUN = 1, /* payload = ELF; place, start on the app core */
    NETBOOT_CMD_RERUN = 2,    /* re-place the staged ELF and start again */
    NETBOOT_CMD_KILL = 3,     /* force the app core down */
    NETBOOT_CMD_STATUS = 4,
};

/* netboot_req.flags */
#define NETBOOT_F_FORCE 0x1U /* terminate an app that ran past its max_run_ms to take the core */

enum netboot_state {
    NETBOOT_ST_IDLE = 0,
    NETBOOT_ST_RUNNING = 1,
    NETBOOT_ST_KILLED = 2,  /* by KILL or by a forced run */
    NETBOOT_ST_TIMEOUT = 3, /* ran past its max_run_ms (the core may still be running) */
    NETBOOT_ST_EXITED = 4,  /* app returned from main; core parked and reusable */
};

/* netboot_ack.verdict */
enum netboot_verdict {
    NETBOOT_ACK_OK = 0,       /* admitted: send the payload (LOAD_RUN), then read the reply */
    NETBOOT_ACK_BUSY = 1,     /* an app is running within its max: retry after retry_after_ms */
    NETBOOT_ACK_OVERTIME = 2, /* the app ran past its max: resend with NETBOOT_F_FORCE to terminate it */
    NETBOOT_ACK_STUCK = 3,    /* forced, but this PDI cannot stop the app core: park it over JTAG
                               * (stop, rst -proc, pc = park_addr, con) or re-bootstrap the board */
    NETBOOT_ACK_SERVING = 4,  /* another client's request is in progress: retry after retry_after_ms */
    NETBOOT_ACK_REJECT = -1,  /* malformed request */
};

/* netboot_ack.core */
enum netboot_core { NETBOOT_CORE_FREE = 0, NETBOOT_CORE_PARKED = 1, NETBOOT_CORE_BUSY = 2 };

struct netboot_req { /* 64 bytes */
    uint32_t magic;
    uint32_t version;
    uint32_t cmd;
    uint32_t flags; /* NETBOOT_F_* */
    uint64_t elf_size;
    uint32_t max_run_ms; /* this run's limit; 0 = NETBOOT_DEFAULT_MAX_RUN_MS (version 1: "timeout_ms") */
    uint32_t reserved;
    uint8_t sha256[32];
} __attribute__((packed));

struct netboot_ack { /* 48 bytes, then msg_len bytes of text */
    uint32_t magic;
    int32_t verdict;         /* enum netboot_verdict */
    uint32_t state;          /* enum netboot_state */
    uint32_t core;           /* enum netboot_core */
    uint32_t run_ms;         /* how long the current/last app has run */
    uint32_t max_run_ms;     /* its limit */
    uint32_t remaining_ms;   /* until it reaches the limit (0 when past it) */
    uint32_t retry_after_ms; /* suggested wait before retrying (BUSY / SERVING), 0 otherwise */
    uint32_t can_stop;       /* PLM lets the supervisor stop the app core: 1 yes, 0 refused, 2 not tried yet */
    uint32_t runs;
    uint32_t park_addr; /* this supervisor's nb_park: where a debugger may send a reset app core
                         * so that it parks (stuck-core recovery without a re-bootstrap) */
    uint32_t msg_len;
} __attribute__((packed));

struct netboot_rep { /* 48 bytes, then msg_len bytes of text */
    uint32_t magic;
    int32_t rc;
    uint64_t bytes_rx;
    uint32_t rx_ms;
    uint32_t place_ms;
    uint64_t entry;
    uint32_t state;
    uint32_t run_ms;
    uint32_t runs;
    uint32_t msg_len;
} __attribute__((packed));

#endif
