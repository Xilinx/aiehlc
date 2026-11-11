---
name: netbootsupervisor
description: Working on src/tool/netboot (lwIP ELF supervisor on the baremetal A78, `apppaltest.py --netload`). Use when the supervisor dies or goes silent during LOAD_RUN, uploads crawl, the app core will not relaunch ("app core busy", PLM 0x7DB / 0x7D2 / "ERR XPm_ForcePowerdown: 0x2"), the watchdog fires right after launch, a client wants to know whether the board is busy / how long to wait ("board busy", "overtime", "stuck", "serving", netload exit 3/4/5), setting a run's max run time or forcing an overdue app off, a bootstrap prints `[netboot] ready` but every load says `No route to host` ("Auto negotiation timeout", "Phy setup failure init_emacps", `NOT ready: Ethernet link down`), or when choosing memory for any second resident program on palmyra.
---

# netboot supervisor: board facts that are not in the design's assumptions

Design and status: `doc/design/lwip_elf_bootloader.md` (section 12). Usage:
`src/tool/netboot/README.md`.

## 1. `0x8_0000_0000` is NOT free memory on palmyra

"DDR_LOW1" mirrors the 2 GB DDR_LOW0. Proven with xsdb physical writes: a value
written at `0x8_0000_0000` reads back at `0x0`. A program linked there physically
sits under the app at `0x0`. Placing the app then overwrites that program's MMU
tables and code:
- `ESR_EL3 0x9600004A`: data abort, write, access-flag fault at level 2;
- then an `EC 0` (undefined instruction) loop inside the fault handler.

The supervisor therefore lives at `0x7800_0000` (top of DDR_LOW0), and the
loader refuses app `PT_LOAD`s that reach it. To check physical aliasing, use
xsdb on target 1/19/37 with `mwr -force`/`mrd -force`. The core target (20) goes
through that core's MMU.

## 2. `Xil_SetTlbAttributes` above 4 GB rewrites a whole 1 GB block

The GEM lwIP port makes `emac_bd_space` non-cacheable. In `.bss` above 4 GB that
turned the entire GB uncached, and uploads crawled until the client timed out.
Keep the BD ring alone in its own MMU block. Below 4 GB that block is 2 MB:
`.netboot_bd` at `0x7A00_0000`.

## 3. `p_memsz` includes the 1 GB `.heap` and `.stack`

The BSP `lscript` folds them into the one `PT_LOAD`. Copy `p_filesz` only, and
zero exactly the allocated `SHT_NOBITS` sections except `.heap`/`.stack`. Do not
rely on `crt0`: it skips BSS clearing when `APU_PWRCTL` reports a warm boot.

## 4. This PDI gives no way to stop a running app core

| call | PLM answer |
|---|---|
| `XPm_ForcePowerDown(PM_DEV_ACPU_0_1)` | `ERR XPm_ForcePowerdown: 0x2` |
| `XPm_ResetAssert(PM_RST_APU0_CORE1_WARM/COLD)` | `0x7D2` `XPM_PM_NO_ACCESS` (even after `XPm_RequestNode`) |
| `XPm_ReleaseNode` | returns success, core keeps running (do not use) |
| `XPm_RequestWakeUp` on a running core | `0x7DB` `XPM_ERR_WAKEUP` |

So `STATUS` says `app core busy` after any run, and `apppaltest --netload`
re-bootstraps the supervisor (`device program` frees core 1). Only a BOOT.BIN
subsystem-permission change lifts this.

**This applies to palmyra only.** On VEK385 the supervisor patches the app's `_exit` to park the core
(`_exit -> park`), so load and rerun can be repeated with no re-bootstrap (15 loads in a row on vek385-11).
See `.cursor/skills/vek385netboot/SKILL.md`.

## 5. Watchdog firing a few ms into every run was OUR bug

The main loop sampled `now` before `server_poll()` launched the run, so
`now - run_start_ms` underflowed. `watchdog()` reads its own clock. It is not
the system counter: switching to `PMCCNTR_EL0` changed nothing.

## Debugging rules that paid off

- A silent death needs a fault handler that prints `ESR/FAR/ELR/SP` with fixed
  label strings. The BSP handler's data argument is not reliably yours.
- Trace with prints first, then remove them: anything the supervisor prints
  after the wake interleaves with the app's output on the shared UART.
- The board is shared. Another user's `device program` / `rst -proc` kills the
  supervisor, so get exclusive use before chasing a "hang".

## Protocol v2: ack before payload, max run time, forced runs

Design section 13. The supervisor answers the 64 B header with a `netboot_ack`
**before** the ELF is sent: `ok` / `busy` (time left, retry-after) / `overtime`
(past `max_run_ms`, resend with `--force`) / `stuck` (forced, PLM cannot stop
the core: re-bootstrap) / `serving` (another client mid-request).
`netload.py` exit 3/4/5 = busy / overtime / stuck.

- The default max run time is **60 s** (`NETBOOT_DEFAULT_MAX_RUN_MS`). An app that
  legitimately runs longer needs `--max-run-ms` / `--netload-max-run-ms`, or the
  next forced run takes the core.
- `retry_after_ms` is the time to the run's *limit*, not to its exit: a waiting
  client must poll (netload polls every <= 2 s), or it sleeps through an early
  exit.
- On palmyra a forced run is always `stuck` (can_stop = 0): only
  `apppaltest --netload` (re-bootstrap) frees the core. An app *within* its limit
  is never preempted, so a shared board is never power-cycled under someone
  else's live run unless it is overdue.
- After rebuilding `netboot.elf`, `--netload-bootstrap` once: an old v1
  supervisor answers a v2 header with a v1 error reply, which netload reports as
  "speaks protocol version 1 ... re-bootstrap".
- Test app with a known run time: `src/tool/netboot/test/build.sh <s>` ->
  `build/nbspin<s>.elf`.

## Stuck core: JTAG park instead of a re-bootstrap

- `apppaltest --netload` answers `stuck` with `netboot_park_app_core()`: xsdb
  `stop; rst -proc; rwr pc <park_addr>; con` on `Cortex-A78AE #0.1` only. The
  core comes out of reset into `nb_park`; `status` then says `app core parked`.
  ~16-19 s per run vs ~60 s for power cycle + `device program`.
- `park_addr` must come from the **running** supervisor (the ack field), never
  from a local `nm build/netboot.elf`: a different build parks the core in
  garbage. A supervisor that reports 0 (pre-park_addr) gets the re-bootstrap.
- The supervisor MUST stay polled (`Xil_ExceptionDisable`, `XEmacPs_IntrHandler()`
  in the main loop). `rst -proc` on core 1 leaves `GICD_CTLR = 0x30` (groups
  off), GEM IRQ 71 disabled/pending, core 0 IRQs masked: an interrupt-driven
  supervisor keeps looping but never takes a packet again (no ping/ARP,
  GEM `rx_status` 0x3 = buffer not available). Polling costs nothing measurable
  (board rx 839 vs 837 ms for 26.4 MB).
- xsdb reads over pexpect wrap at 80 columns and rlwrap mangles lines: open
  the spawn with `dimensions=(200, 2000)`, or write results with
  `set f [open /tmp/x a]; puts $f ...` on the lab host and `cat` it.

## Ethernet link down after a bootstrap ("ready" but `No route to host`)

- Symptom (213, 2026-11-10, 1 of 3 bootstraps): console shows `Start TI PHY
  autonegotiation`, `Auto negotiation timeout`, `Phy setup error : link_speed
  invalid`, `Phy setup failure init_emacps`, and the old supervisor still printed
  `[netboot] ready`; every `load` then failed `[Errno 113] No route to host` and
  the NEXT run had to bootstrap again -- it looked like "every run redoes
  everything", and like an hw_server problem. It is neither: hws, power cycle,
  `device program` and `dow` all succeeded.
- Cause: the lwIP port's TI PHY setup waits only 5 s for autonegotiation
  (`PHY_AUTONEG_TIMEOUT_DEFAULT` in `xemacpsif_physpeed.c`); after a power cycle
  the lab switch sometimes takes longer. The port's recovery
  (`eth_link_detect()`) runs from `link_detect_thread`, which **NO_SYS does not
  have**, so the link stayed DOWN for good. The port's recovery also applies
  `phy_setup_emacps()`'s result without checking `XST_FAILURE` -- do not call it.
- Fix (`supervisor/main.c`): `net_init` retries `phy_setup_emacps()` up to
  `NB_PHY_TRIES` (6) times; `[netboot] ready` is printed **only with the link
  up** (else `NOT ready: Ethernet link down ...`); `link_poll()` reads BMSR (twice:
  the link bit latches low) every second from `timers()` and sets the PHY up again
  when it reports link + autonegotiation complete, or marks a lost link down.
  apppaltest `netboot_bootstrap()` re-bootstraps once (`NETBOOT_BOOTSTRAP_TRIES`)
  when `ready` never comes or a `ready` supervisor still does not answer.
- Testing it on a board whose link always comes up:
  `NETBOOT_TEST_LINK_DOWN=1|2 NETBOOT_OUT=/tmp/nbt src/tool/netboot/build.sh`
  forces the link down after init (1 = the init retry, 2 = the main-loop poll);
  copy it over `build/netboot.elf`, `touch` it, run `apppaltest --netload
  --netload-bootstrap`, then put the normal build back and bootstrap again.
  Both verified on 213: `Ethernet link up, 1000 Mb/s` then `ready` and a load.
