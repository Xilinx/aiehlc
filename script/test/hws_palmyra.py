"""palmyra (213) backend for hws (contract v1 §2, plan-1109-b1597a s3).

Service = tmux session ``hws`` on huaj@<ip>: /bin/systest -> become "palmyra" -> plain
``hw_server`` (binds 0.0.0.0:3121; ``-stcp:...`` is a syntax error at Systest#).

Rules learned on 213 (plan-1109-cd847e):
  * systest is only ever closed through ``tmux send-keys`` (hw_server stop, power 0, exit).
    An unclean end (kill, ssh drop) of a /bin/systest makes 213 reboot itself (~70 s).
  * ``clean`` kills only huaj-owned xsdb / hw_server / systest / JTAG holders that are
    outside our ``hws`` tmux, older than STALE_SECS and not under the current lock owner.
    Every kill is logged. Afterwards it waits out a possible self-reboot.
  * systest logs a session out after ``$SYSTEST_AUTOLOGOUT`` s without input
    (default.cmd: 3600), taking hw_server with it -- the hws session ended 60 min 20 s
    after every start (2026-11-10 journal). ``_start_once`` sets
    ``$SYSTEST_DISABLE_AUTOLOGOUT = 1`` (read by the timer when it fires).
  * ANY ``/bin/systest`` of this user that exits runs its ``cleanup_session``:
    ``killall -9 hw_server`` (and cs_server, tftpd, nfsd; with ``lsf=on`` in
    /proc/cmdline also every process of the user -- 213 is ``lsf=off``). So a full-mode
    apppaltest run, another agent's systest or our own restart kills the hws hw_server;
    ``status`` then says DEGRADED (session alive, 3121 closed) and ``up`` restarts it.
    systest-client exits (console, JTAG park, bootstrap) do not -- checked: clean exit,
    force-kill, after power 0/1, with xsdb attached, started before hws.
"""
import os
import re
import shlex

import hws

SESSION = "hws"
TARGETS_EXPECT = {"Cortex-A78AE": 8, "Cortex-R52": 10}
DEVICE = "xc2ve3858"
PROMPT_RE = re.compile(r"Systest#\s*$")
BECOME_CMD = 'become "palmyra"'
# systest logs a session out after $SYSTEST_AUTOLOGOUT seconds without input
# (/opt/systest/common/etc/default.cmd: 3600), and hw_server goes with it: on 213
# the hws session ended 60 min 20 s after each start (journal, 2026-11-10). The
# autologout timer (systest.jar Session$3) skips when this variable is set, and
# reads it when it fires, so setting it in the running session is enough.
NO_AUTOLOGOUT_CMD = "$SYSTEST_DISABLE_AUTOLOGOUT = 1"
# On 213 the prompt is "[cpg-lab-06] Systest#" before AND after become, so the
# proof that become worked is the setup banner it prints (seen 2026-11-10 22:57).
BECAME_RE = re.compile(r"com0 \(PS\)")
# Program names (argv[0] basename; argv[1] for `perl /bin/systest`) that hold JTAG/the board.
HOLDER_NAMES = {"xsdb", "hw_server", "systest-client", "systty", "telnet", "systest"}
LOCK_SKEW = 30  # s of clock skew allowed between the container and 213
STALE_SECS = 60
SYSTEST_WAIT, BECOME_WAIT, PORT_WAIT, TARGETS_WAIT = 30, 60, 30, 60
STOP_WAIT, EXIT_WAIT = 20, 60
REBOOT_GRACE, REBOOT_WAIT, SETTLE_SECS = 60, 180, 20
POLL = 2
START_ATTEMPTS = 2


class Unreachable(Exception):
    pass


def parse_ps(text):
    """``ps -eo pid=,ppid=,user=,etimes=,args=`` -> list of dicts."""
    procs = []
    for line in text.splitlines():
        f = line.split(None, 4)
        if len(f) == 5 and f[0].isdigit() and f[1].isdigit() and f[3].isdigit():
            procs.append({"pid": int(f[0]), "ppid": int(f[1]), "user": f[2], "age": int(f[3]), "args": f[4]})
    return procs


def holder_name(args):
    """Holder program name of a ps args line, or None (wrappers, editors, tail, grep...)."""
    argv = args.split()
    if not argv:
        return None
    name = os.path.basename(argv[0])
    if name == "java" and "com.xilinx.systest" in args:
        return "systest-java"
    if name == "perl" and len(argv) > 1:
        name = os.path.basename(argv[1])
    return name if name in HOLDER_NAMES else None


def descendants(procs, roots):
    kids = {}
    for p in procs:
        kids.setdefault(p["ppid"], []).append(p["pid"])
    seen, todo = set(), [r for r in roots if r]
    while todo:
        pid = todo.pop()
        if pid not in seen:
            seen.add(pid)
            todo.extend(kids.get(pid, []))
    return seen


def count_targets(out):
    """Unique cores per family in an xsdb `targets` listing."""
    return {fam: len(set(re.findall(rf"{fam}\s+#(\d+\.\d+)", out))) for fam in TARGETS_EXPECT}


class Backend:
    def __init__(self, board, io, log, lock_owner=None):
        self.host = hws.host_of(board)
        self.target = board if "@" in board else f"huaj@{self.host}"
        self.user = self.target.split("@")[0]
        self.board = board
        self.io = io
        self.log = log
        self.lock_owner = lock_owner or {}

    # ---- remote primitives -------------------------------------------------
    def sh(self, cmd, timeout=30):
        rc, out = self.io.ssh(self.target, cmd, timeout)
        if rc in (255, hws.RC_STALLED):
            raise Unreachable(("ssh stalled: " if rc != 255 else "") + out.strip()[-200:])
        return rc, out

    def has_session(self):
        return self.sh(f"tmux has-session -t {SESSION}")[0] == 0

    def pane(self):
        rc, out = self.sh(f"tmux capture-pane -p -t {SESSION} -S -3000")
        return out if rc == 0 else ""

    def last_line(self):
        lines = [l for l in self.pane().splitlines() if l.strip()]
        return lines[-1] if lines else ""

    def send(self, text, session=SESSION):
        self.log(f"send-keys {session}: {text}")
        self.sh(f"tmux send-keys -t {session} -l -- {shlex.quote(text)} && tmux send-keys -t {session} Enter")

    @staticmethod
    def became(pane):
        """True if the pane shows a become "palmyra" that printed the setup banner."""
        i = pane.rfind(BECOME_CMD)
        return i >= 0 and bool(BECAME_RE.search(pane[i:]))

    def at_prompt(self):
        return bool(PROMPT_RE.search(self.last_line()))

    def wait_for(self, pred, secs):
        end = self.io.now() + secs
        while True:
            if pred():
                return True
            if self.io.now() >= end:
                return False
            self.io.sleep(POLL)

    def boot_id(self):
        rc, out = self.io.ssh(self.target, "cat /proc/sys/kernel/random/boot_id", 20)
        return out.strip() if rc == 0 else None

    def ps(self):
        rc, out = self.sh("ps -eo pid=,ppid=,user=,etimes=,args=")
        procs = parse_ps(out)
        if rc != 0 or not procs:
            raise hws.HwsError(hws.RC_DEGRADED, f"HWS FAIL {self.host}: ps failed (rc {rc}); nothing killed")
        return procs

    def tmux_panes(self):
        """{session_name: [pane_pid, ...]} for all of our tmux sessions on the host."""
        rc, out = self.sh("tmux list-panes -a -F '#{session_name} #{pane_pid}'")
        panes = {}
        for line in out.splitlines() if rc == 0 else []:
            f = line.split()
            if len(f) == 2 and f[1].isdigit():
                panes.setdefault(f[0], []).append(int(f[1]))
        return panes

    def our_pids(self, procs):
        """Process tree of our hws session. Fails closed: never guess when the pane is unknown."""
        if not self.has_session():
            return set()
        roots = self.tmux_panes().get(SESSION, [])
        if not roots:
            raise hws.HwsError(hws.RC_DEGRADED, f"HWS FAIL {self.host}: hws session exists but its pane pid "
                               "is unknown; refusing to clean (nothing killed)")
        return descendants(procs, roots)

    # ---- health ------------------------------------------------------------
    def checks(self):
        c = {"ssh": True, "session": self.has_session()}
        c["palmyra_prompt"] = c["session"] and self.became(self.pane())
        c["port"] = self.io.port_open(self.host)
        c["targets"] = False
        if c["port"]:
            rc, out = self.io.xsdb_targets(self.host)
            n = count_targets(out)
            c["cores"] = n
            c["targets"] = (rc == 0 and DEVICE in out and "PMC" in out
                            and all(n[f] >= k for f, k in TARGETS_EXPECT.items()))
        return c

    def foreign_hw_server(self):
        """Users other than ours running hw_server on the host (port owner unknown otherwise)."""
        return sorted({p["user"] for p in self.ps() if "hw_server" in p["args"] and p["user"] != self.user})

    def status(self):
        try:
            return self._status()
        except Unreachable as e:
            return self._health(hws.UNREACHABLE, {"ssh": False}, f"ssh to {self.target} failed: {e}",
                                hws.RC_UNREACHABLE)

    def _status(self):
        c = self.checks()
        if c["session"] and c["palmyra_prompt"] and c["port"] and c["targets"]:
            return self._health(hws.UP, c, f"hw_server {self.host}:{hws.PORT} serves {c['cores']}")
        if not c["session"] and not c["port"]:
            return self._health(hws.DOWN, c, "no hws session, 3121 closed", hws.RC_OK)
        if not c["session"] and c["port"]:
            users = self.foreign_hw_server()
            if users:
                return self._health(hws.FOREIGN, c, f"3121 served by hw_server of {users}; not touched",
                                    hws.RC_FOREIGN)
        return self._health(hws.DEGRADED, c, self._why(c), hws.RC_DEGRADED)

    def _why(self, c):
        if not c["session"]:
            return "3121 open but no hws session of ours (stale hw_server? run clean)"
        if not c["palmyra_prompt"]:
            return "hws session has not done become palmyra (no setup banner)"
        if not c["port"]:
            return "hws session alive but 3121 closed (hw_server not running)"
        return f"3121 open but targets incomplete {c.get('cores')}, want {TARGETS_EXPECT} on {DEVICE}"

    def _health(self, state, checks, msg, rc=hws.RC_OK):
        return hws.Health(self.board, "palmyra", state, checks, msg, rc)

    # ---- reboot survival ---------------------------------------------------
    def wait_ssh(self, secs=REBOOT_WAIT):
        ok = self.wait_for(lambda: self.boot_id() is not None, secs)
        if not ok:
            raise hws.HwsError(hws.RC_UNREACHABLE, f"HWS FAIL {self.host}: no ssh for {secs}s (after reboot?)")

    def settle_after_kill(self, boot_before):
        """213 may reboot itself after a holder dies. Wait it out, never more than once."""
        self.log(f"waiting up to {REBOOT_GRACE}s for a possible 213 self-reboot")
        rebooted = self.wait_for(lambda: self.boot_id() != boot_before, REBOOT_GRACE)
        if not rebooted:
            self.log("no reboot")
            return False
        self.log(f"213 is rebooting; waiting up to {REBOOT_WAIT}s for ssh")
        self.wait_ssh()
        self.io.sleep(SETTLE_SECS)  # let the farm re-enumerate USB/JTAG
        self.log(f"213 back, boot_id {self.boot_id()}")
        return True

    # ---- clean -------------------------------------------------------------
    def stale_holders(self):
        """huaj holders outside our hws tree, older than STALE_SECS and older than the current lock.

        Anything started after the current lock owner took the lock belongs to that owner
        (the lock is exclusive), so it is never stale -- this protects apppaltest's own
        xsdb/systest when it calls hws with locked=True."""
        procs = self.ps()
        keep = self.our_pids(procs)
        min_age = STALE_SECS
        if self.lock_owner.get("start"):
            min_age = max(min_age, self.io.now() - self.lock_owner["start"] + LOCK_SKEW)
        return procs, [p for p in procs if p["user"] == self.user and p["pid"] not in keep
                       and p["age"] >= min_age and holder_name(p["args"])]

    def close_other_sessions(self, procs, stale):
        """Other tmux sessions of ours running a stale systest: close them via send-keys, not kill."""
        stale_systest = {p["pid"] for p in stale if holder_name(p["args"]) == "systest"}
        names = [n for n, roots in self.tmux_panes().items()
                 if n != SESSION and descendants(procs, roots) & stale_systest]
        for name in names:
            self.log(f"clean: closing stale systest in tmux session {name!r} via send-keys")
            for cmd in ('hw_server "stop"', "power 0", "exit"):
                self.send(cmd, session=name)
                self.io.sleep(POLL)
        if names:
            self.wait_for(lambda: not set(names) & set(self.tmux_panes()), EXIT_WAIT)
        return bool(names)

    def clean(self):
        try:
            boot = self.boot_id()
            if boot is None:
                self.wait_ssh()
                boot = self.boot_id()
            procs, stale = self.stale_holders()
            if stale and self.close_other_sessions(procs, stale):
                procs, stale = self.stale_holders()
        except Unreachable as e:
            raise hws.HwsError(hws.RC_UNREACHABLE, f"HWS FAIL {self.host}: ssh failed: {e}")
        if not stale:
            self.log("clean: no stale holders")
            return self.status()
        # Children before parents: tools first, the /bin/systest perl last.
        stale.sort(key=lambda p: (holder_name(p["args"]) == "systest", -p["pid"]))
        for p in stale:
            self.log(f"KILL pid {p['pid']} user {p['user']} age {p['age']}s: {p['args'][:160]}")
        pids = " ".join(str(p["pid"]) for p in stale)
        try:
            self.sh(f"kill {pids} 2>/dev/null; sleep 3; kill -9 {pids} 2>/dev/null; true")
        except Unreachable:
            self.log("ssh dropped during kill (host going down)")
        self.settle_after_kill(boot)
        return self.status()

    # ---- up ----------------------------------------------------------------
    def up(self):
        try:
            self.sh("true")
        except Unreachable:
            self.log("ssh refused; maybe a 213 self-reboot, waiting")
            self.wait_ssh()
        last = None
        for attempt in range(1, START_ATTEMPTS + 1):
            h = self.status()
            if h.state == hws.UP:
                self.log(f"up: UP ({h.msg})")
                return h
            if h.state in (hws.FOREIGN, hws.UNREACHABLE):
                raise hws.HwsError(h.rc, f"HWS {h.state} {self.host}: {h.msg}", h)
            if attempt > 1:
                self.down()  # attempt 2 = restart of our own session
            if attempt > 1 or not h.checks.get("session"):
                self.clean()  # a second palmyra session would conflict with any stale one
            self.log(f"up: start attempt {attempt}/{START_ATTEMPTS} ({h.state}: {h.msg})")
            try:
                last = self._start_once()
            except Unreachable as e:
                self.wait_ssh()
                last = f"ssh dropped during start: {e}"
        h = self.status()
        if h.state == hws.UP:
            return h
        msg = (f"HWS FAIL {self.host}: {last or h.msg} after {START_ATTEMPTS} attempts "
               f"(port {'open' if h.checks.get('port') else 'closed'}, session "
               f"{'alive' if h.checks.get('session') else 'absent'}). {hws.LAB_POINTER} Log: {self.log.path}")
        raise hws.HwsError(hws.RC_DEGRADED, msg, h)

    def _start_once(self):
        """Bring session -> palmyra -> hw_server -> targets. Returns a failure reason or None."""
        if not self.has_session():
            self.sh(f"tmux new-session -d -s {SESSION} -x 200 -y 50 /bin/systest")
            if not self.wait_for(self.at_prompt, SYSTEST_WAIT):
                return f"no Systest# prompt in {SYSTEST_WAIT}s"
        if not self.became(self.pane()):
            for _ in range(2):  # become, 1 retry
                self.send(BECOME_CMD)
                self.wait_for(self.at_prompt, BECOME_WAIT)
                if self.became(self.pane()):
                    break
            else:
                return f"become palmyra printed no setup banner (com0 (PS)) in 2x{BECOME_WAIT}s"
        self.send(NO_AUTOLOGOUT_CMD)
        self.wait_for(self.at_prompt, BECOME_WAIT)
        if not self.io.port_open(self.host):
            self.send("hw_server")
            if not self.wait_for(lambda: self.io.port_open(self.host), PORT_WAIT):
                return f"3121 not open {PORT_WAIT}s after hw_server"
        if not self.wait_for(lambda: self.checks()["targets"], TARGETS_WAIT):
            return f"H3 targets incomplete, want {TARGETS_EXPECT}"
        return None

    # ---- down --------------------------------------------------------------
    def down(self):
        try:
            if not self.has_session():
                h = self.status()
                return h if h.state == hws.DOWN else self._health(
                    h.state, h.checks, f"no hws session of ours; {h.msg}", h.rc)
            self._stop_session()
            still = self.has_session()
        except Unreachable as e:
            raise hws.HwsError(hws.RC_UNREACHABLE, f"HWS FAIL {self.host}: ssh failed during down: {e}")
        h = self.status()
        if still:
            raise hws.HwsError(hws.RC_DEGRADED, f"HWS FAIL {self.host}: hws session did not exit "
                               f"(left alone, never killed). Log: {self.log.path}", h)
        return self._health(hws.DOWN, h.checks, h.msg if h.state != hws.DOWN else "hws closed cleanly",
                            hws.RC_OK if h.state == hws.DOWN else h.rc)

    def _stop_session(self):
        closed = lambda: not self.io.port_open(self.host)  # noqa: E731
        for cmd in ('hw_server "stop"', "hw_server stop"):
            if closed():
                break
            self.send(cmd)
            self.wait_for(closed, STOP_WAIT)
        self.send("power 0")
        self.wait_for(self.at_prompt, SYSTEST_WAIT)
        for _ in range(2):  # exit leaves systest; a second exit if a sub-prompt remains
            self.send("exit")
            if self.wait_for(lambda: not self.has_session(), EXIT_WAIT // 2):
                self.log("hws session exited cleanly")
                return
