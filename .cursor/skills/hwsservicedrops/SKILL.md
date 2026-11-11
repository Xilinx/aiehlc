---
name: hwsservicedrops
description: The palmyra hw_server service (hws, `script/test/hws*.py`, tmux session `hws` on 213) is gone or DEGRADED again -- apppaltest prints `[hws] status DOWN` / `3121 open but ...` / `session alive but 3121 closed` and spends ~40 s on `hws up`, the hws session vanished ~60 min after it started with no `hws_down` log, or hw_server died when some other systest session ended. Also: can hws tell "service up but board gone" (213 reboot, board taken, powered off) and what it does then.
---

# hws keeps disappearing on 213

Evidence lives on 213: `journalctl --since <t> | grep sysconfig.pl` (systest start =
`clean_tmp` + `ssh_auth add`; a systest JVM shutting down runs `shutdown.cmd` =
`port_forward kill_rpc_nfsd kill_tftpd ptpv2d ssh_auth`, then `clean_tmp` + `ssh_auth
remove`), `last -F huaj`, and `/scratch/staff/huaj/tmp/hws/<ip>/hws_*.log` (every
`up` / `down` / `restart` we did). A session end with no matching `hws_down` log was
not us. The remote login shell is **csh**: wrap checks in `bash -c '...'`, or
`2>/dev/null &&` silently reads as "gone".

## Causes found (2026-11-10)

1. **systest autologout.** `/opt/systest/common/etc/default.cmd`:
   `$SYSTEST_AUTOLOGOUT = 3600`. hws is a `/bin/systest`; nobody types into it after
   `hw_server`, so it logged out 60 min 20 s after each start, three times in a row.
   Fix: `hws_palmyra._start_once` sends `$SYSTEST_DISABLE_AUTOLOGOUT = 1` (systest.jar
   `Session$3` checks it when the timer fires). Verified on 213: a session started
   18:56 with it was still up, 3121 open, 62 min later with no input in between.
   Only sessions started by the new code
   have it: `apppaltest.py --hws down` then `--hws up` to apply it to a running one.
2. **Any `/bin/systest` exit kills hw_server.** `/bin/systest` `cleanup_session()` runs
   `killall -9 hw_server` (+ cs_server, atftpd, rpc.nfsd) on single-session hosts, and with
   `lsf=on` in /proc/cmdline terminates every process of the user (213: `lsf=off`). A
   full-mode apppaltest run, `hws restart`, or another agent on the same unix user (e.g.
   a Maxwell `claude -p` in docker) does it. hws then reads DEGRADED and `up` repairs it.
3. **Not the cause** (reproduced each, hws survived): a systest-client console leaving
   cleanly or force-killed, a client that did `power 0/1` then exit or force-kill, a
   client with xsdb attached to hw_server force-killed, a client started while hws was
   down (it waits for the server; `open_console_fast` does not close it on TIMEOUT).
   One drop 2.5 min after start (18:40:52) stayed unexplained.

## "Service up but board gone" -- what status sees

`status` = session + `become` banner + port 3121 + xsdb `targets` (8 A78 + 10 R52 + PMC
on xc2ve3858). Board powered off / JTAG lost -> DEGRADED -> one restart, then FAIL with
the lab pointer. Other unix user's hw_server on 3121 -> FOREIGN, nothing touched. 213
rebooting -> UNREACHABLE: `ensure_service` fails at once (does not wait; `up` would).
Another agent on the **same** user looks like ours: only the board lock
(`/scratch/staff/huaj/tmp/board_locks/<ip>_3121.lock`) keeps a restart from `become`-ing
the board away from it. Netload is lazy: none of this runs unless JTAG is needed.
