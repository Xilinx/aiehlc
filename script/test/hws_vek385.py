"""VEK385 backend of hws (contract section 2, "VEK385"; plan-1109-b1597a s4).

On VEK385 hw_server runs on the farm host the board is cabled to
(``<farmhost>:3121``) and only exists as a device of a systest session that
holds the board. So hws owns a *hold session*: ``script``-wrapped
``/proj/systest/bin/systest vek385-N`` fed through a FIFO (the hold.sh
pattern), state in ``$HWS_LOG_DIR/<farmhost>/``:

    in.fifo     commands for the [vek385-N] Systest# prompt
    hold.log    the session console
    owner.json  pids (+pidns, start time) of script and FIFO writer, board, LSF job

start   = ``hw_server`` at Systest# (the alias forks it as a session device)
stop    = ``hw_server "stop"`` (unquoted ``stop`` is a syntax error)
release = ``power 0`` then ``exit`` -> "Reboot is disabled!", no host reboot.
Never kill the systest client or the LSF job (an unclean end can reboot the host).
Locking is done by hws._run; the backend never takes the board lock.
"""
import base64
import json
import os
import re
import shlex
import subprocess

import devlock
import hws

TYPE = "vek385"
CLUSTER_PING = "/proj/systest/bin/cluster-ping"
SYSTEST = "/proj/systest/bin/systest"
# Pass bar for `targets` (contract: >=1 A78AE and >=1 R52 plus Versal/PMC).
TARGETS_MIN = {"Cortex-A78AE": 1, "Cortex-R52": 1}
# What healthy vek385-12 (crimini2, rev A1) listed in the s4 run 2026-11-10: Versal Gen 2
# xc2ve3858, RPU 5 clusters x2 R52, APU 4 clusters x2 A78AE, PPU, ASU, PMC, PL, DPC.
# Same silicon as palmyra; informational, the pass bar stays TARGETS_MIN.
EXPECTED_SEEN = {"Cortex-A78AE": 8, "Cortex-R52": 10}

PROMPT_RE = r"\[(vek385-\d+)\] Systest#"
DEAD_RE = r"Systest server exiting|Script done|AUTOLOGOUT!"
STARTED_RE = r"Starting Vivado hardware server|hardware server is running"
SESSION_SECS = 900      # LSF dispatch + system controller boot
PROMPT_SECS = 60
PORT_UP_SECS = 30
TARGETS_SECS = 60
PORT_DOWN_SECS = 20
EXIT_SECS = 90
RELEASE_SECS = 180     # cluster-ping keeps showing our ended job for a while
POLL = 2
START_ATTEMPTS = 2
AUTOLOGOUT_SECS = 14400  # systest default 3600 s idle killed earlier holds (and their hw_server)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\r")
# own leftovers on the farm host: (pkill args, only when we have no live session)
# systest-client is handled separately: a live hold has its own client (child of the perl systest).
# -f patterns use the [x] trick so they never match the clean script's own `bash -c` / login shell.
STALE = (("-x xsdb", False), ("-f 'telnet 10.10.71.1 40[0]'", True))


def parse_cluster_ping(text):
    """{farmhost: (vboard, holder, job)}; holder is 'ok' (free), 'closed' or a user."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(vek385-\d+)\s+(\S+)\s+\[[^\]]*\]\s+(\S+)(?:\s+(\d+))?", line)
        if m:
            out[m.group(2)] = (m.group(1), m.group(3), m.group(4))
    return out


def count_targets(out):
    """Unique cores per family in an xsdb `targets` listing."""
    return {fam: len(set(re.findall(rf"{fam}\s+#(\d+\.\d+)", out))) for fam in TARGETS_MIN}


def targets_good(rc, out):
    n = count_targets(out)
    ok = (rc == 0 and "Versal" in out and "PMC" in out
          and all(n[f] >= k for f, k in TARGETS_MIN.items()))
    return ok, n


class Backend:
    """One VEK385 farm host. Remote effects go through ``io``; local session effects
    (spawn, FIFO, console file, cluster-ping) are methods tests can replace."""

    def __init__(self, board, io, log, lock_owner=None):
        self.board = board
        self.host = hws.host_of(board)
        self.me = devlock.whoami()
        self.io = io
        self.log = log
        self.lock_owner = lock_owner or {}
        self.dir = os.path.join(hws.LOG_ROOT, self.host)
        os.makedirs(self.dir, exist_ok=True)
        self.fifo = os.path.join(self.dir, "in.fifo")
        self.logf = os.path.join(self.dir, "hold.log")
        self.ownerf = os.path.join(self.dir, "owner.json")

    # ---- local effects -------------------------------------------------------
    def cluster_ping(self):
        r = subprocess.run([CLUSTER_PING], capture_output=True, text=True, timeout=120)
        return parse_cluster_ping(r.stdout)

    def spawn_hold(self, vboard):
        """Start the hold in its own session group so it outlives this process."""
        sh = (f"cd {shlex.quote(self.dir)}; rm -f in.fifo; mkfifo in.fifo; : > hold.log\n"
              f"export USER={self.me} LOGNAME={self.me}\n"
              "( exec sleep 172800 > in.fifo 2>/dev/null < /dev/null ) & echo writer $!\n"
              f"setsid nohup script -qfc '{SYSTEST} {vboard}' hold.log < in.fifo > /dev/null 2>&1 &\n"
              "echo script $!\n")  # children must not keep our capture pipes open
        out = subprocess.run(["bash", "-c", sh], capture_output=True, text=True,
                             timeout=30, start_new_session=True).stdout
        return {k: int(v) for k, v in re.findall(r"(writer|script) (\d+)", out)}

    def send(self, cmd):
        """One line into the Systest# prompt; False if nothing reads the FIFO."""
        try:
            fd = os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            return False
        try:
            os.write(fd, (cmd + "\n").encode())
        finally:
            os.close(fd)
        return True

    def console(self):
        try:
            with open(self.logf, errors="replace") as f:
                return ANSI.sub("", f.read())
        except OSError:
            return ""

    def kill_own(self, pid):
        os.kill(pid, 15)

    # ---- owner / session -------------------------------------------------------
    def owner(self):
        try:
            with open(self.ownerf) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def write_owner(self, owner):
        with open(self.ownerf, "w") as f:
            json.dump(owner, f)

    @staticmethod
    def pid_alive(owner, key):
        """True/False for a pid recorded in owner.json; None if not visible from here."""
        pid = owner.get(key)
        if not pid or owner.get("pidns") != devlock.pid_ns():
            return None
        return devlock._alive(pid) and devlock.proc_start(pid) == owner.get(key + "_pstart")

    def session(self):
        """(alive, job, why) of OUR hold session."""
        owner = self.owner()
        if not owner:
            return False, None, "no hold session of ours"
        text = self.console()
        job = owner.get("job")
        dead = re.findall(DEAD_RE, text)
        if dead:
            return False, job, f"hold session ended ({dead[-1]})"
        if self.pid_alive(owner, "script") is False:
            return False, job, f"hold pid {owner.get('script')} is gone"
        if not re.search(PROMPT_RE, text):
            return False, job, "hold session has no Systest# prompt"
        return True, job, f"hold {owner.get('vboard')} job {job} alive"

    def wait_console(self, pattern, secs, since=0):
        end = self.io.now() + secs
        while True:
            m = re.search(pattern, self.console()[since:])
            if m or self.io.now() >= end:
                return m
            self.io.sleep(POLL)

    def cmd(self, line, wait=PROMPT_RE, secs=PROMPT_SECS):
        since = len(self.console())
        self.log(f"systest> {line}")
        if not self.send(line):
            raise hws.HwsError(hws.RC_DEGRADED, f"HWS FAIL {self.host}: hold FIFO has no reader "
                                                f"(session died), log {self.logf}")
        return self.wait_console(wait, secs, since) if wait else True

    def wait_port(self, want_open, secs):
        end = self.io.now() + secs
        while True:
            if self.io.port_open(self.host) == want_open:
                return True
            if self.io.now() >= end:
                return False
            self.io.sleep(POLL)

    def create_session(self, vboard):
        for attempt in (1, 2):
            pids = self.spawn_hold(vboard)
            owner = {"vboard": vboard, "host": self.host, "user": self.me, "pidns": devlock.pid_ns(),
                     "start": self.io.now(), "script": pids.get("script"), "writer": pids.get("writer"),
                     "script_pstart": devlock.proc_start(pids.get("script", 0)),
                     "writer_pstart": devlock.proc_start(pids.get("writer", 0))}
            self.write_owner(owner)
            self.log(f"hold {vboard} attempt {attempt}: script pid {owner['script']}, log {self.logf}")
            if self.wait_console(PROMPT_RE, SESSION_SECS):
                owner["job"] = (re.findall(r"Job <(\d+)>", self.console()) or [None])[-1]
                self.write_owner(owner)
                self.cmd(f"$SYSTEST_AUTOLOGOUT = {AUTOLOGOUT_SECS}")
                return
            self.log(f"no Systest# prompt after {SESSION_SECS}s; queueing `exit` for a late dispatch")
            self.send("exit")
            self.release_writer()
        raise hws.HwsError(hws.RC_DEGRADED, f"HWS FAIL {self.host}: no systest hold on {vboard} after "
                                            f"2 attempts (log {self.logf}). {hws.LAB_POINTER}")

    def wait_released(self):
        """After our hold ended, wait (bounded) until cluster-ping stops showing our job.
        False at once if the holder is not our own ended job."""
        job = (self.owner() or {}).get("job")
        end = self.io.now() + RELEASE_SECS
        while True:
            vboard, holder, cp_job = self.cluster_ping().get(self.host, (None, None, None))
            if holder == "ok":
                return True
            if holder != self.me or not job or cp_job != job or self.session()[0]:
                return False
            if self.io.now() >= end:
                self.log(f"{vboard} still shows our ended job {job} after {RELEASE_SECS}s")
                return False
            self.io.sleep(10)

    def release_writer(self):
        """Stop our FIFO-writer `sleep` (never the systest client)."""
        owner = self.owner() or {}
        if self.pid_alive(owner, "writer"):
            self.kill_own(owner["writer"])

    # ---- health ----------------------------------------------------------------
    def _health(self, state, checks, msg, rc):
        return hws.Health(self.board, TYPE, state, checks, msg, rc)

    def checks(self):
        entry = self.cluster_ping().get(self.host)
        if not entry:
            raise hws.HwsError(hws.RC_USAGE, f"{self.host} is not a VEK385 farm host in cluster-ping")
        vboard, holder, cp_job = entry
        alive, job, why = self.session()
        c = {"vboard": vboard, "holder": holder, "holder_job": cp_job, "session": alive, "session_why": why}
        c["port"] = self.io.port_open(self.host)
        c["targets"] = False
        if c["port"]:
            rc, out = self.io.xsdb_targets(self.host)
            c["targets"], c["cores"] = targets_good(rc, out)
        return c

    def status(self):
        c = self.checks()
        holder = c["holder"]
        if holder not in ("ok", self.me) or (holder == self.me and not c["session"] and c["port"]):
            who = f"{holder} job {c['holder_job']}" + (" outside hws" if holder == self.me else "")
            return self._health(hws.FOREIGN, c, f"{c['vboard']} held by {who}; not touched", hws.RC_FOREIGN)
        if c["session"] and c["port"] and c["targets"]:
            return self._health(hws.UP, c, f"hw_server {self.host}:{hws.PORT} serves {c['cores']}", hws.RC_OK)
        if not c["session"] and not c["port"]:
            return self._health(hws.DOWN, c, f"{c['vboard']}: {c['session_why']}, 3121 closed", hws.RC_OK)
        return self._health(hws.DEGRADED, c, self._why(c), hws.RC_DEGRADED)

    @staticmethod
    def _why(c):
        if not c["session"]:
            return f"3121 open but {c['session_why']} (stale/orphan hw_server? run clean)"
        if not c["port"]:
            return "hold session alive but 3121 closed (hw_server not running)"
        return f"3121 open but targets incomplete {c.get('cores')}, want >= {TARGETS_MIN} + Versal/PMC"

    # ---- hw_server inside the session ---------------------------------------------
    def start_hw_server(self):
        """Returns None on success, else why it failed."""
        # Right after `hw_server "stop"` the prompt can eat the next line as a stray
        # token ("token recognition error at: ''", seen on vek385-12): retype once.
        m = self.cmd("hw_server", STARTED_RE + "|Syntax error", 30)
        if m and m.group(0) == "Syntax error":
            self.cmd("hw_server", STARTED_RE, 30)
        if not self.wait_port(True, PORT_UP_SECS):
            return f"port 3121 still closed {PORT_UP_SECS}s after `hw_server`"
        end = self.io.now() + TARGETS_SECS
        while True:
            rc, out = self.io.xsdb_targets(self.host)
            ok, n = targets_good(rc, out)
            if ok:
                self.log(f"targets: {n}")
                return None
            if self.io.now() >= end:
                return f"H3 targets incomplete {n} after {TARGETS_SECS}s"
            self.io.sleep(5)

    def stop_hw_server(self):
        self.cmd('hw_server "stop"', r"exited with status|Syntax error|" + PROMPT_RE, 20)
        if self.wait_port(False, PORT_DOWN_SECS):
            return True
        self.log('hw_server "stop" left 3121 open; pkill of our own hw_server device')
        self.io.ssh(self.host, f"pkill -u {self.me} -x hw_server", 30)
        return self.wait_port(False, PORT_DOWN_SECS)

    def bring_up(self, h):
        """Hold (create if needed) + at most START_ATTEMPTS starts."""
        if not self.session()[0]:
            if h.checks.get("holder") != "ok" and self.wait_released():
                h = self.status()
            if h.checks.get("holder") != "ok":
                raise hws.HwsError(hws.RC_FOREIGN, f"{h.checks.get('vboard')} not free: {h.msg}", h)
            self.create_session(h.checks["vboard"])
        why = None
        for attempt in range(1, START_ATTEMPTS + 1):
            if attempt > 1 or self.io.port_open(self.host):
                self.stop_hw_server()
            why = self.start_hw_server()
            self.log(f"start attempt {attempt}: {why or 'ok'}")
            if why is None:
                break
        h = self.status()
        if h.state != hws.UP:
            msg = (f"HWS FAIL {self.host}: {why or h.msg} after {START_ATTEMPTS} attempts "
                   f"({h.checks.get('session_why')}). {hws.LAB_POINTER} Log: {self.logf}")
            raise hws.HwsError(hws.RC_DEGRADED, msg, h)
        return h

    # ---- contract operations ---------------------------------------------------------
    def up(self):
        h = self.status()
        if h.state == hws.UP:
            return h
        if h.state == hws.FOREIGN:
            raise hws.HwsError(hws.RC_FOREIGN, h.msg, h)
        return self.bring_up(h)

    def down(self):
        h = self.status()
        if h.state == hws.FOREIGN:
            raise hws.HwsError(hws.RC_FOREIGN, h.msg, h)
        if self.session()[0]:
            if not self.stop_hw_server():
                self.log("WARN 3121 still open after stop; ending the session anyway")
            self.cmd("power 0")
            self.cmd("exit", wait=DEAD_RE, secs=EXIT_SECS)
        self.release_writer()
        self.wait_released()
        if self.io.port_open(self.host):
            h = self.status()
            raise hws.HwsError(hws.RC_DEGRADED, f"down: 3121 still open on {self.host}: {h.msg}", h)
        return self._health(hws.DOWN, dict(h.checks, session=False, session_why="hold released", port=False, targets=False),
                            f"{self.host}: hold released, 3121 closed", hws.RC_OK)

    def clean(self):
        h = self.status()
        if h.state == hws.FOREIGN:
            raise hws.HwsError(hws.RC_FOREIGN, h.msg, h)
        alive = self.session()[0]
        sh = ""
        for args, stale_only in STALE:
            if stale_only and alive:
                continue
            sh += f"pgrep -a -u {self.me} {args}; pkill -u {self.me} {args}\n"
        # stray systest-clients only: the hold's own client is a child of the perl systest, and
        # killing it would end our live session uncleanly (s6, gomphus3)
        sh += (f"for p in $(pgrep -u {self.me} -f '[s]ystest-client'); do "
               "pp=$(ps -o ppid= -p $p | tr -d ' '); "
               "case \"$(ps -o args= -p $pp)\" in *perl*/bin/systest|*perl*/bin/systest\\ *) continue;; esac; "
               "ps -o pid=,args= -p $p && K=\"$K $p\"; done; "
               "[ -n \"$K\" ] && { kill $K; sleep 3; kill -9 $K; } 2>/dev/null; true\n")
        # our own orphan `hw_server -d` (parent init); a session device has the systest parent
        sh += (f"for p in $(pgrep -u {self.me} -x hw_server); do "
               "[ \"$(ps -o ppid= -p $p | tr -d ' ')\" = 1 ] && ps -o pid=,args= -p $p && kill $p; done; true\n")
        # The farm login shell is tcsh, which cannot parse a quoted `bash -c` script
        # ("Unmatched '"); ship it base64-encoded and let bash read it from stdin.
        b64 = base64.b64encode(sh.encode()).decode()
        rc, out = self.io.ssh(self.host, f"echo {b64} | base64 -d | bash", 60)
        if rc == 255:
            msg = "clean: no ssh to the farm host (no hold of ours), nothing to clean remotely"
        else:
            msg = "clean: killed " + (" | ".join(out.split("\n")).strip(" |") or "nothing (no stale holders)")
        self.log(msg)
        return self._health(h.state, h.checks, msg, hws.RC_OK)

    def restart(self):
        """Soft restart inside our hold (keeps the board): stop, clean, start.
        Without a hold of ours this is just `up`."""
        h = self.status()
        if h.state == hws.FOREIGN:
            raise hws.HwsError(hws.RC_FOREIGN, h.msg, h)
        if self.session()[0] and h.checks.get("port"):
            self.stop_hw_server()
        self.clean()
        return self.bring_up(self.status())
