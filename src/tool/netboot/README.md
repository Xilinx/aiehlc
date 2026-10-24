# netboot: resident lwIP supervisor for the baremetal A78

Implements mode D of [doc/design/lwip_elf_bootloader.md](../../../doc/design/lwip_elf_bootloader.md).
A supervisor stays resident on **A78 core 0** at the top of DDR (`0x7800_0000`;
`0x8_0000_0000` mirrors `0x0` on this board, see design section 12). It receives an ELF
over TCP and runs it on **core 1** through the PLM, with a watchdog. The app's
console output stays on the board UART (com0) and is collected by
`apppaltest.py --netload` exactly as in the JTAG flow.

```
src/tool/netboot/
  build.sh              builds lwIP 2.2 + Xilinx GEM port for the A78 SDT BSP, then netboot.elf
  netboot_proto.h       wire protocol v2: 64 B request, 48 B ack (before the payload), 48 B reply + text
  netload.py            host client: load | rerun | kill | status (stdlib only); direct TCP on PALIP,
                        or --via user@PALIP from the dev box through netboot_proxy.py
  netboot_proxy.py      stdio<->TCP relay that lives on PALIP; netload.py streams through it over ssh
  lwip/                 lwipopts.h, xlwipconfig.h, xtopology_g.c (what the Vitis BSP flow would generate)
  supervisor/
    main.c              lwIP init (static IP), poll loop, timers, watchdog, launch/kill
    server.c            TCP protocol on lwIP RAW; work runs from the main loop, not callbacks
    elfload.c           ELF64/AArch64 checks, PT_LOAD placement into DDR_LOW0, cache clean
    apu.c               app core via xilpm: XPm_RequestWakeUp / XPm_ForcePowerDown on PM_DEV_ACPU_0_1
    sha256.c            upload integrity check
```

lwIP follows the aiebaremetal `example_lwip_echo` reference: RAW API, `NO_SYS`,
polled `xemacif_input`. It is compiled from the Vitis embeddedsw sources
against this repo's A78 BSP, because the reference's `libxtflwipall.a` is an
xsct-generated A72 / Versal-Gen1 build.

## Build

```bash
src/tool/netboot/build.sh                       # -> src/tool/netboot/build/netboot.elf
NETBOOT_IP=10.10.70.50 NETBOOT_GW=10.10.70.101 src/tool/netboot/build.sh
```

The supervisor occupies `0x7800_0000..0x7A20_0000` (image, then the GEM BD ring
alone in its 2 MB MMU block at `0x7A00_0000`) and stages uploads at
`0x7A20_0000..0x7FFF_FFFF`. It refuses an app `PT_LOAD` reaching `0x7800_0000`.

The default IP sits on the lab board segment that `PALIP` serves on `eth1`
(10.10.70.0/24, dhcpd range .1-.5), outside the DHCP range. The build refuses a
supervisor image outside `0x7800_0000..0x7A20_0000`.

## Run

```bash
python3 script/test/apppaltest.py -y -nonreboot --netload aout/main.elf
```

1. stages `netboot_proxy.py` on `PALIP`, only if its SHA-256 changed. The
   **target ELF is never copied**;
2. `netload.py status --via`: from the dev box, through ssh, into the proxy, to the
   supervisor. If it answers with core 1 `free` or `parked`, there is no JTAG and no power command;
   otherwise it stages `netboot.elf` (if changed), loads it over JTAG and waits for `[netboot] ready`;
3. opens com0, then `netload.py load <local elf> --via`: the ELF **streams from the
   dev box through the ssh pipe** into the proxy and on to the supervisor (SHA-256
   check, place, start or release core 1);
4. collects the console until `device_teardown done` / `Not tearing down partition`,
   or until the supervisor reports the app exited;
5. leaves the board up with the supervisor resident.

**Reuse without re-bootstrap (park).** The palmyra PDI does not let the supervisor
power down or reset core 1 (design section 12, item 4). To reuse it anyway, the
supervisor patches each app's `_exit` (the BSP's `b .`, padded to 16 bytes) into
`ldr x16,#8; br x16; .quad nb_park`. When the app returns from `main`, core 1
lands in `supervisor/park.S`: it records the exit code, parks in WFE, and on
release turns MMU/caches off and jumps to the next entry. `status` reports
`app core parked (exit rc N)`, and `apppaltest --netload` then sends the next
ELF directly, with no JTAG and no power commands. It still re-bootstraps when
nothing answers, when core 1 is `busy` (app running or hung, or `_exit` not
patchable), or with `--netload-bootstrap`.
Verified on palmyra: after one bootstrap, a ResNet run and a conv2dstem run were
each sent directly (`released to 0x0 (no PLM call)`). They took **11 s** and
**15 s** wall, against ~50 s with a bootstrap. `--netload` also ends the console
wait when the supervisor reports `state=exited`, so apps that print no
`device_teardown done` finish on their real exit instead of the 300 s timeout.
The reported exit rc is whatever `x0` held at `_exit`; treat it as informational.

**Power.** Netload never powers the board off when an app finishes. The only power
commands are the `power 0`/`power 1` before a bootstrap's `device program`,
because programming a live system aborts part-way (measured: ~18%). The bootstrap
uses the non-owning `systest-client` session (no `become`), so closing it cannot
release the board either.

**When `device program` happens.** A bootstrap is the full JTAG sequence:
`device program BOOT.BIN` (waited on until it reports success, then a 5 s PLM
settle), then `tar 20`, `rst -proc`, `dow -force netboot.elf`, `con`. Each step is
confirmed by its own completion token, not by an `xsdb%` prompt, which rlwrap
echoes twice. When a resident supervisor is **reused** (it answers with
`app core free`), nothing is reprogrammed and PL/AIE state carries over from the
previous run. Pass `--netload-bootstrap` to force a fresh PDI program.

**Build on demand.** `apppaltest.py --netload` checks that `netboot_proxy.py`
exists and that `build/netboot.elf` exists and is newer than every supervisor and
lwIP source (`ensure_netboot_built`). Otherwise it runs `build.sh` first, which
needs the A78 cross compiler on `PATH`. With lwIP cached a rebuild takes ~2 s; a
first build ~1 min.

**Debug web UI.** The debug server's Run button for the pal board probes the
supervisor and passes `--netload` when it acks. See `src/tool/debug/README.md`,
"Run deploys through netboot when a supervisor answers".

Flags: `--netload-bootstrap` forces a supervisor reload (use it after rebuilding
`netboot.elf`); `--netload-ip IP`; `--netload-max-run-ms N` (alias
`--netload-timeout-ms`) sets this app's max run time, 0 = the supervisor default
60 s; `--netload-wait S` (default 120) waits for a board busy with a run still
within its max; `--netload-no-force` refuses to terminate a previous app past its
max (by default it is forced off, which on palmyra means a re-bootstrap).

## Busy board, max run time, forced runs (protocol v2)

The supervisor answers every request's header with an **ack** before the ELF is
sent (design section 13), so a client learns in one round trip whether it will
run and, if not, how long to wait:

```
[netload] load not run: board busy (core busy, state=running, run 7: 11.2 s of its 60.0 s max, 48.8 s left); retry in 48.9 s (or --wait)
[netload] load not run: board overtime (... run 2: 11.4 s of its 8.0 s max, past it); resend with --force to terminate it
[netload] load not run: board stuck (...); re-bootstrap the supervisor to free it
[netload] status not run: board serving -- another client's request is in progress: retry shortly
```

| `netload.py` option | meaning |
|---|---|
| `--max-run-ms N` (alias `--timeout-ms`) | this run's limit; 0 = board default 60 s |
| `--force` | terminate an app that is past its limit and run this one |
| `--wait S` | while busy, retry for up to S s (polls every <= 2 s) |
| `--proto 1` | talk to a supervisor built before version 2 |

**A stuck app no longer needs a reboot.** For `stuck`, `apppaltest --netload`
frees core 1 over JTAG instead of re-bootstrapping: xsdb on the lab host runs
`stop; rst -proc; rwr pc <park_addr>; con` on `Cortex-A78AE #0.1` only, so the
core restarts into the supervisor's park routine (`park_addr` comes in every
ack), then the ELF is sent again: ~16-19 s for the whole run instead of ~60 s.
By hand:

```tcl
connect -url TCP:<PALIP>:3121
targets -set -nocase -filter {name =~ "*A78*#0.1*"}
stop; rst -proc; rwr pc 0x7801a780; con      ;# park_addr from `netload.py status`/the stuck ack
```

The supervisor is **polled** (no interrupts): that reset leaves the GIC
disabled, which used to silence it (design section 12 item 9).

Exit codes: 0 ok, 1 refused/failed, 2 no answer, 3 still busy, 4 past its max
(needs `--force`), 5 stuck (needs a re-bootstrap). An app *within* its limit is
never preempted. Test app: `src/tool/netboot/test/build.sh <seconds>` builds
`build/nbspin<seconds>.elf`, which busy-waits that long.

Data path, no scp of the target ELF:

```
dev box: netload.py --via huaj@PALIP ==ssh stdin/stdout==> PALIP: netboot_proxy.py ==TCP==> board 10.10.70.50:5155
```

**Run time.** `apppaltest.py -y -nonreboot --netload <ResNet main.elf>` takes
**~2.35 s end to end** (measured 2.35-2.37 s, three consecutive runs). About
1.4 s of that is the app itself (`inference 1270 ms` + boot). How:
- **one multiplexed ssh connection** (`ControlMaster`, `ControlPersist=4h`,
  `ControlPath=/tmp/apppal-$USER-%C`) for every ssh/scp. A fresh connection to
  PALIP costs 0.55-0.85 s; a multiplexed one 0.05 s;
- **console via `ssh -tt host systest-client`**, so there is no login-shell prompt to wait for;
- **optimistic `load`**: no `status` round trip first. A refusal ("busy" / no answer)
  falls back to the bootstrap;
- **console attach overlaps the transfer**: `netload.send_gate` holds back the
  last 4 KB until com0 is attached. The supervisor cannot start the app before
  the last byte, so nothing is missed (`resnet18: init` is captured);
- netload runs in-process (no second Python); poll interval 0.05 s, console read
  0.1 s, trailing drain 0.05 s.

The first run after the shared ssh connection expires (4 h idle) pays one
handshake: measured **3.29 s**. The bootstrap path (no supervisor, or core 1 busy)
still takes ~50 s.

From the dev box by hand:

```bash
python3 src/tool/netboot/netload.py status --via huaj@10.23.224.213 --proxy /home/huaj/aiehlc/netboot/netboot_proxy.py
python3 src/tool/netboot/netload.py load aout/main.elf --via huaj@10.23.224.213 --proxy /home/huaj/aiehlc/netboot/netboot_proxy.py
```

Or on `PALIP` itself (direct TCP):

```bash
python3 netload.py status
python3 netload.py load main.elf --max-run-ms 120000      # this run may take 2 min
python3 netload.py load main.elf --wait 60 --force        # wait for a busy board, preempt an overdue app
python3 ~/aiehlc/netboot/netload.py rerun     # same image again, no transfer   (needs a free app core)
python3 ~/aiehlc/netboot/netload.py kill      # stop the app core               (refused by the palmyra PDI)
```

## Console lines from the supervisor

The supervisor prints nothing after it wakes the app core (except watchdog or
fault events), so the collected log is the app's own output:

```
[netboot] ready: 10.10.70.50 port 5155
[netboot] LOAD_RUN 15411872 B received in 200 ms, sha256 ok
[netboot] run 1: placed 13234248 bytes, zeroed 31436417 bss bytes, entry 0x0, start app core at 0x0 (watchdog 600000 ms)
resnet18: init                       <- from here on: the app on core 1
...
resnet18: top1 class=258 logit=12.017338
  inference    1270.868 ms   <- graph_run()
device_teardown done
```

On a fault the supervisor prints `[netboot] FATAL ...: ESR_EL3 .. FAR_EL3 .. ELR_EL3 ..`
instead of hanging silently.
