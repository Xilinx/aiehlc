---
name: alibflocklease
description: aiehlc.sh hangs at "aiehlc.sh: waiting for alib lock .../thirdparty/alib/.aiehlc_alib.lock" with no other build running -- a shell that SOURCED an older aiehlc.sh still holds the flock fd. How to find the holder (lslocks, /proc/*/fd), release it, and why main now runs in a subshell so the kernel drops the lock.
---

# alib flock held by a shell after `source aiehlc.sh`

## Symptom

    aiehlc.sh: AIEHLC_BIN=...
    aiehlc.sh: waiting for alib lock .../thirdparty/alib/.aiehlc_alib.lock

and nothing more, forever -- for any source file (looks like "aiehlc is stuck
compiling this file"). Scripts calling aiehlc.sh (e.g. a board/merge script) sit
in `flock 10`.

## Cause

GAP-ALIB-RACE (`script/aiehlc.sh`, before `build_hw_lib`) takes an exclusive
`flock` on `thirdparty/alib/.aiehlc_alib.lock` through
`exec {AIEHLC_ALIB_LOCK_FD}>lockfile`. aiehlc.sh is **sourced**, so that fd is
opened in the *caller's shell*. Before the fix nothing closed it: after the run
returned, the interactive shell kept the lock, and every later run in that tree
-- from any shell, including the same one (a new `exec` is a new open file
description, so it blocks on its own old lock) -- waited forever. A `bash -c
'source ...'` / subshell caller was fine: the process exit closed the fd.

## Fix (2026-11-23): main runs in a subshell

The tail of `aiehlc.sh` is `( main "$@" ) && rc=0 || rc=$?` then
`return $rc 2>/dev/null || exit $rc`. `main` -- and so the `exec {FD}>lockfile`
+ `flock` -- runs in its own process whether the file is sourced or executed,
so the **kernel** releases the lock when that process ends: normal return,
error, Ctrl-C, `kill`. No trap, no explicit unlock. Same subshell also stops
`set -eo pipefail` (which main turns on) and main's variables/exports from
leaking into a sourcing caller -- before, a caller without `set -e` (e.g.
`conv2dstem/buildtest.sh`'s `rc=$?`) was killed by the leaked errexit before it
could report the failure. Callers only consume the return code and `aout/`
(checked: every in-tree `source` caller; `piplinerun.sh` gets its env from
`setup.sh`). Behaviour change: `AIEHLC_DIR`, `LLVM_AIE_PATH`,
`XILINX_VITIS_AIETOOLS` are no longer left in the caller's shell.

`aiehlc_release_alib_lock` stays, called once in the *caller* before the
subshell: only a shell that sourced an older aiehlc.sh still holds an fd, and
the subshell would inherit it and wait on it forever. The `&& ||` form keeps a
failing main from tripping a caller's own `set -e` before rc is captured; a
caller that has `set -e` still exits on a non-zero `source` (plain bash).

Verified: early failure sourced -> rc=1, errexit off, 0 fds; subtree of the
subshell killed with TERM mid alib-make -> lock FREE at once, sourcing parent
continues with rc=143 and 0 fds; full simplematmul2 build sourced -> rc=0,
`flowcheck: OK`, 0 fds, errexit off.

Still true: a process started by main that outlives it (none today: no `&`,
`nohup`, `setsid` in the script) would inherit the fd and keep the lock. Killing
only the subshell pid leaves its `make` child holding the lock until it ends --
Ctrl-C hits the whole foreground group, so it is not an issue there.
The lock still serializes whole runs on one tree (from `build_hw_lib` to exit),
because `build_hw_lib` wipes alib unconditionally; use
`worktree.sh link-deps --private-alib` for real parallel builds.

## Releasing a lock held right now

    lslocks -o PID,TYPE,MODE,BLOCKER,PATH | grep alib          # waiters + blocker pid
    for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q aiehlc_alib.lock \
      && echo "$(basename $p) $(tr '\0' ' ' < $p/cmdline)"; done   # who has it open
    ls -l /proc/<shell pid>/fd | grep alib                     # which fd numbers

The BLOCKER pid is often already dead (the `flock` child); the lock lives on in
the shell's fd. In that shell: `exec 10>&- 11>&-` (its fd numbers), or close the
terminal. Never delete the lock file -- holders keep the old inode and the race
protection is lost.

## Note

Worktrees made with `script/worktree.sh link-deps --private-alib` have their own
`thirdparty/alib` and so their own lock file; they are not blocked by the main
checkout's holder (that is why a worktree reproduction "doesn't hang").

## The lock is opt-in now (2026-11-23): `-serialize`

Default `aiehlc.sh` takes **no** alib lock, so a run never waits on another
(or on a leaked holder). `-serialize` / `--serialize` / `AIEHLC_SERIALIZE=1`
restores the exclusive lock for the whole run (`waiting for alib lock` /
`alib lock held` lines). Verified: with the lock held elsewhere, a default run
finishes (rc 0) and a `-serialize` run waits. Without the lock, two runs on one
tree at once can wipe each other's `thirdparty/alib` (GAP-ALIB-RACE:
`xaie_feature_config.h: No such file`) -- run them in separate worktrees
(`script/worktree.sh link-deps --private-alib`) or pass `-serialize` to all of
them. `AIEHLC_LIBCACHE` keeps its own shared/exclusive locking.
