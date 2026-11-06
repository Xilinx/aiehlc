---
name: vek385netboot
description: Runs an ELF on a VEK385 farm board with script/test/apppaltest.py (-board vek385 -host <farm host>), by full JTAG boot or the netboot supervisor (--netload). Use when a VEK385 run prints nothing, nbspin never ends, the supervisor never answers ping/ARP, "wake of app core FAILED", A78 #0.1 sits in "Reset Catch", "APU core power request denied", systest-client dies with "User signal 1", or you need the vek385-11 / vek385-3 netload procedure (bootstrap, load/rerun loop, __rt_done pass check).
---

# VEK385 netboot / JTAG boot with apppaltest

```bash
python3 script/test/apppaltest.py -board vek385 -host gomphus3 -y --netload aout/main.elf   # netboot
python3 script/test/apppaltest.py -board vek385 -host gomphus3 -y aout/main.elf             # JTAG boot
```

`-board vek385` accepts any `vek385-N` on that host (`-board vek385-3` only that
one); `-board-rev b` programs the boot + PLD PDI pair. Run board tests under the
farm lock: `BOARD_OWNER=huaj board.py lock vek385-3 -- <cmd>`
(`/scratch/staff/huaj/aiengineers/skills/board-mgmt/board.py`). Do **not** nest
`board.py power-cycle` inside `board.py lock` -- both take the same flock and it
deadlocks.

## Procedure (verified on vek385-11 @ crimini1, HEAD ccfcaa2, 2026-11-05)

**Re-verified on vek385-3 @ gomphus3, aiehlc main e570563, 2026-11-06:** the exact
`python3 script/test/apppaltest.py -board vek385 -host gomphus3 -y --netload ./aout/main.elf`
(conv2dstem ELF, md5 c06b32dd) passed 5/5 under one lock on the fast path (supervisor already up,
no bootstrap): 4.28 MB sent in 0.29-0.79 s, board rx 195-397 ms, 2.2-3.6 s wall per run.
Logs: `aiengineers/jane/tasks/2026-11-06_vek385-3_main_netload/s2/`. With `-board/-host` you do not
need `BOARDNAME/PALIP` or `--netload-bootstrap`: `--netload` bootstraps by itself if the board is off.

**Prerequisites**
- The board must be held by huaj's job: `/proj/systest/bin/cluster-ping | grep vek385-11`
  shows `huaj`. Pairs: vek385-11 = crimini1 (hw_server `crimini1:3121`), vek385-3 = gomphus3.
  Both hosts have eth1 = 10.10.70.101/24, and the supervisor is at 10.10.70.50.
- `BOARD_OWNER=huaj board.py probe vek385-11` must show `[vek385-11] Systest#`. Never use `become`
  (palmyra only). apppaltest uses the host's systest-client by itself.
- `source /scratch/staff/huaj/aiengineers/skills/vitis-2026.2/vitis_env.sh` (xsdb, A78 gcc).
  The supervisor was verified on the board with gcc 13.4 (2026.1). gcc 15.3 is not verified on the board.

**PMC system reset before programming.** With `-board vek385*` the bootstrap runs
`targets -set -nocase -filter {name =~ "PMC"}; rst -system` right before `device program`
(after the power cycle and `connect`), so the PDI programs a clean system. Verified on
vek385-11 (2026-11-04): reset OK, BIN programmed, SCNTRS on, supervisor up (`status`: idle, pm ok).

**Build (automatic).** With `-board vek385*` (or `BOARDNAME=vek385-*`), apppaltest builds
`src/tool/netboot/build/vek385/netboot.elf` with `NETBOOT_DMA_BASE=0xA00000000` and stages it as
`~/aiehlc/netboot/netboot_vek385.elf`. This is needed because GEM0 sits on LPD_AXI_NOC0, which reaches
DDR_CH1 0xA_0000_0000 but not DDR_LOW0. The default image is `~/aiehlc/netboot/vek385_netboot.BIN`
(aiebaremetal boot.pdi + pld.pdi, md5 a3c1ce75). Override it with `VEK385PDI=` or `BOOT_IMAGE=`.

**Run the whole experiment under ONE lock.** Put every state-changing step (bootstrap, loads, reruns,
JTAG writes) in one script. Run it as `BOARD_OWNER=huaj board.py lock vek385-11 -- ./full.sh`.
The template is `aiengineers/jane/tasks/2026-11-05_vek385-11_netload/full.sh`.
```bash
export USERNAME=huaj PALIP=crimini1 BOARDNAME=vek385-11 BOARD_OWNER=huaj   # == -board vek385-11 -host crimini1
# 1) bootstrap (~100 s): power cycle once, program the BIN once, SCNTRS on, `rst -processor` on A78 #0.0 only,
#    download the supervisor, then run the ELF
timeout 400 python3 script/test/apppaltest.py -y -nonreboot --netload --netload-bootstrap aout/main.elf
#    expect: "[netboot] GEM DMA memory at 0xA00000000", "[netboot] ready: 10.10.70.50 port 5155"
ssh crimini1 ping -c 3 10.10.70.50                     # the supervisor answers
# 2) fast path, no JTAG: a 2.8 MB ELF takes 0.19-0.7 s
python3 script/test/apppaltest.py -y -nonreboot --netload aout/main.elf
#    or the bare client loop:
NL=src/tool/netboot/netload.py
python3 $NL load  aout/main.elf --via huaj@crimini1 --max-run-ms 30000
python3 $NL rerun --via huaj@crimini1 --max-run-ms 30000    # same ELF, no transfer
python3 $NL status --via huaj@crimini1                      # "pm ok, ... app core parked (exit rc N)"
```
On vek385 the app's `_exit` parks the core, so you can repeat load and rerun without a new bootstrap.
try3 ran bootstrap, then 3 loads, 1 rerun, and the fast path, all PASS. Before ccfcaa2, 15 loads in a row passed.
try3 also had an uncommitted apppaltest hunk (worktree diff in jane tasks `apppaltest_park_worktree.diff`) that
parks A78 #0.1 only if it is in Reset Catch. With ccfcaa2 #0.1 came up in `Power On Reset`, so the hunk printed "no park needed"
and did nothing. If a load still fails with 0x7D2, park #0.1 by hand (see the table).

**Pass/fail: do not trust "exit rc N".** xil-crt0 runs `bl main; bl __libc_fini_array; bl exit`, which
clobbers w0, so the supervisor reports `exit rc 1` even when main returned 0. Until the loader patch
(`netboot_main_rc.diff`, not merged) lands, check the app's own result block over JTAG. For rt_result apps
that block is `__rt_done` (find it with `nm main.elf | grep __rt_done`; it is 0x73100 in rt hello).
Pass = word 0 is `0xD0E5A1E5` (3504710117) and status (word 1) is 0. Zero the block before each repeated
run, or a stale PASS will look like a new one:
```tcl
connect -url TCP:crimini1:3121; targets -set -filter {name =~ "Versal*"}
mwr -force 0x73100 {0 0 0 0}          ;# before the run
mrd -force -value 0x73100 4           ;# after: 3504710117 0 0 0
```
(`rtdone.tcl [clear]` in the same tasks dir.)

**Apps without `__rt_done`** (e.g. the conv2dstem `aout/main.elf`, `test_conv2d_new.cc`): judge by the
console line `[conv2dstem] GOLDEN PASS 802816/802816` (+ `conv2dstem done (rc=0)`), but count it only if
it comes AFTER this run's own `[netboot] run N:` line. When com3 attaches it replays the previous run's
output, so a plain `grep GOLDEN` sees a stale PASS (seen in 3 of 5 runs on vek385-3).

**Agent rules**
- One board, one session. Another session's `device program` on the live board caused PLM 0x302 and
  spoiled try1 and try2.
- Don't nest `board.py power-cycle` inside `board.py lock`. apppaltest's bootstrap power-cycles through
  systest itself.
- Run the bootstrap in the background with `timeout`. Poll it and write a progress line at least every 3 min.

## Symptoms -> cause -> fix (all in apppaltest / netboot now)

| Symptom | Cause | Fix |
|---|---|---|
| app prints nothing / nbspin never ends after a JTAG boot | A78 system counter off: SCNTRS `0xEC920000` CNTCR=0, freq 0; the BSP's enable does not take effect | apppaltest writes `0xEC920020`=99999001, `0xEC920000`=1 over JTAG after `device program` |
| JTAG park fails at `rst -proc` | PLM: "APU core power request denied" | `catch {rst -proc}`, then `pc = nb_park`, `con` (still parks) |
| `[netload] ... console did not attach`, `User signal 1` / `Shared connection closed` | a long-lived systest session on the farm host re-opened its com telnets | transient; `open_console_fast` retries 3x |
| `wake of app core FAILED` = PLM `XPm_RequestWakeUp` 0x7D2 `XPM_PM_NO_ACCESS` on every load | A78 #0.1 held in Reset Catch (pre-ccfcaa2 bootstrap, or a manual `rst -cores`) | re-bootstrap with ccfcaa2 (resets #0.0 only), or JTAG-park #0.1: `stop; catch {rst -proc}; rwr pc <nb_park>; con` (tasks `park_core1.tcl`, nb_park 0x7801ab40 in that build) |
| GEM TXSR 0x11 (HRESP), RXRES 0x103, PLM EAM NOCTYPE1_NCR, no ARP | GEM DMA buffers in DDR_LOW0 (supervisor built without `NETBOOT_DMA_BASE`) | let apppaltest build `build/vek385/`; check the `GEM DMA memory at 0xA00000000` line |
| `ERR XPmDevice_Request: 0x7F4` right after the supervisor banner | harmless PLM message at init (also in run1_bootstrap_1752.log, where the network worked) | ignore |
| `PLM Error Major 0x302 Minor 0x1` during `device program` | device programmed twice without a power cycle (often a second session) | power-cycle, then program once; keep everything under one lock |
| `connect com3` -> `Unrecognized device`, supervisor silent | board powered off | automatic now: `--netload` sees no console AND no supervisor and bootstraps (power cycle + PMC reset + program); verified vek385-3 2026-11-05 (109 s, GOLDEN PASS) |
| supervisor says `exit rc 1` but the app passed | xil-crt0 clobbers w0 (see Procedure) | judge by `__rt_done`, not the rc |
| `select PMC` fails: `more than one targets found with "name =~ "*PMC*""` | the ASU target's name carries its status (`ASU (Memory write error ... PMC LPD domain isolation)`) | apppaltest selects `{name =~ "PMC"}` (exact) for the PMC `rst -system` it runs before every VEK385 `device program` (`vek385_reset_system`) |
| `connect com3` prints `Versal PS UART0`, then a `telnet>` prompt and a burst of `^@` (NUL), then nothing -- yet `netload status` answers and the app runs | a telnet to the console server port (`telnet 10.10.71.1 400N`, child of the farm host's long-lived `systest`) left over from an earlier session still holds the port | apppaltest now leaves the port the systest way (`leave_console`: Ctrl-\\, `exit`) instead of force-closing the ssh, so its runs add no telnets (verified gomphus3). Older leftovers: `ssh <host> 'ps -eo pid,etime,args \| grep "telnet 10.10.71"'`, end the stale ones (yours only), reconnect; Enter at `telnet>` resumes but brings no data while the stale one lives |
| `device program` "PLM stalled" at 85% | `~/aiehlc/vek385.pdi` (= `tutorial/hw/vek385-reva/vek385.pdi`) on vek385-3 | default image `~/aiehlc/netboot/vek385_netboot.BIN` (aiebaremetal boot + pld PDIs) |

## Verify

- JTAG: `A78 system counter enabled`, then the app's output on com3.
- netload bootstrap: `[netboot] GEM DMA memory at 0xA00000000`, `[netboot] ready`,
  `LOAD_RUN ... sha256 ok`, app output, `[Program completed after Ns]`.
- fast path: `app core was parked: released to 0x0 (no PLM call)`.
- Console lines appear twice on JTAG runs (logfile_read + queue echo); not two runs.
- netload: com3 replays the previous run's tail on attach; only lines after `[netboot] run N:` are this run.
- An app without `device_teardown done` ends a JTAG run on the 300 s no-output
  timeout; netload ends on the supervisor's `exited`.

Details: `src/tool/netboot/README.md` "VEK385", `doc/design/lwip_elf_bootloader.md` §14.
Logs for every number above: `/scratch/staff/huaj/aiengineers/jane/tasks/` `2026-11-05_vek385-11_netload/`
(README, try3/full.out, full_bootstrap.log, exp2_steps.log), `2026-11-04_vek385_netload/dma_ch1/`,
`2026-11-05_vek385_netboot_bin/`, `2026-11-05_netboot_exit_rc/`.
