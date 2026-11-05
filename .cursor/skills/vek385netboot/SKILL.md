---
name: vek385netboot
description: Runs an ELF on a VEK385 farm board with script/test/apppaltest.py (-board vek385 -host <farm host>), by full JTAG boot or the netboot supervisor (--netload). Use when a VEK385 run prints nothing, nbspin never ends, the supervisor never answers ping/ARP, "wake of app core FAILED", A78 #0.1 sits in "Reset Catch", "APU core power request denied", or systest-client dies with "User signal 1".
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

## Symptoms -> cause -> fix (all in apppaltest / netboot now)

| Symptom | Cause | Fix |
|---|---|---|
| supervisor banner, PHY up, but no ARP/ping; GEM TXSR HRESP, PLM NOCTYPE1_NCR | GEM is an LPD master; LPD_AXI_NOC0 cannot reach DDR_LOW0 | VEK385 supervisor built with `NETBOOT_DMA_BASE=0xA00000000` (`build/vek385/`, staged `netboot_vek385.elf`) |
| app prints nothing / nbspin never ends after a JTAG boot | A78 system counter off: SCNTRS `0xEC920000` CNTCR=0, freq 0; the BSP's enable does not take effect | apppaltest writes `0xEC920020`=99999001, `0xEC920000`=1 over JTAG after `device program` |
| `wake of app core FAILED (0x...7D2)`, or woken app silent until the 60 s watchdog; xsdb shows A78 #0.1 `Stopped: (Reset Catch)` | APU `rst -cores` (appvek385.py style) suspends BOTH cores | reset only #0.0: `rst -processor` (falls back to `rst -cores` if refused) |
| JTAG park fails at `rst -proc` | PLM: "APU core power request denied" | `catch {rst -proc}`, then `pc = nb_park`, `con` (still parks) |
| `[netload] ... console did not attach`, `User signal 1` / `Shared connection closed` | a long-lived systest session on the farm host re-opened its com telnets | transient; `open_console_fast` retries 3x |
| `device program` "PLM stalled" at 85% | `~/aiehlc/vek385.pdi` (= `tutorial/hw/vek385-reva/vek385.pdi`) on vek385-3 | default image `~/aiehlc/netboot/vek385_netboot.BIN` (aiebaremetal boot + pld PDIs) |

## Verify

- JTAG: `A78 system counter enabled`, then the app's output on com3.
- netload bootstrap: `[netboot] GEM DMA memory at 0xA00000000`, `[netboot] ready`,
  `LOAD_RUN ... sha256 ok`, app output, `[Program completed after Ns]`.
- fast path: `app core was parked: released to 0x0 (no PLM call)`.
- Console lines appear twice on JTAG runs (logfile_read + queue echo); not two runs.
- An app without `device_teardown done` ends a JTAG run on the 300 s no-output
  timeout; netload ends on the supervisor's `exited`.

Details: `src/tool/netboot/README.md` "VEK385", `doc/design/lwip_elf_bootloader.md` §14.
