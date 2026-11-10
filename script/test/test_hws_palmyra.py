"""Unit tests for hws + the palmyra backend, with a fake 213 (mocked ssh, tmux, nc, xsdb)."""
import os
import re
import shlex
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import hws  # noqa: E402
import hws_palmyra as hp  # noqa: E402

BOARD = "10.23.224.213"
GOOD_TARGETS = "\n".join(
    ["  1  Versal Gen2 xc2ve3858", "     2  PMC", "     3  PSM"]
    + [f"        {10 + i}  Cortex-A78AE #{i // 2}.{i % 2} (Running)" for i in range(8)]
    + [f"        {30 + i}  Cortex-R52 #{i // 2}.{i % 2} (Running)" for i in range(10)])


P = "[cpg-lab-06] Systest# "


class Fake213(hws.Io):
    """Scripted 213: one tmux server, a process table, a systest that obeys send-keys."""

    def __init__(self):
        self.t = 0.0
        self.session = False
        self.pane_lines = []  # 213 prompt is "[cpg-lab-06] Systest# " before and after become
        self.palmyra = False
        self.hw_server = False
        self.targets = GOOD_TARGETS
        self.procs = []  # (pid, ppid, user, age, args)
        self.reachable = True
        self.boot = "boot-1"
        self.reboot_at = None  # time when ssh goes down; back REBOOT_LEN later
        self.reboot_on_kill_systest = True
        self.killed = []
        self.other = {}  # other tmux sessions of huaj: name -> pane pid
        self.sent = []
        self.cmds = []
        self.broken_become = False
        self.hw_server_ignores_stop = False

    # clock
    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def _tick_reboot(self):
        if self.reboot_at is not None and self.t >= self.reboot_at:
            self.reachable = self.t >= self.reboot_at + 70
            if self.reachable:
                self.boot, self.reboot_at = "boot-2", None
                self.session = self.hw_server = self.palmyra = False
                self.procs = []

    # Io
    def port_open(self, host, port=3121, timeout=3):
        self._tick_reboot()
        return self.reachable and self.hw_server

    def xsdb_targets(self, host, port=3121, timeout=60):
        return (0, self.targets) if self.port_open(host) else (1, "HWS_CONNECT_ERR refused")

    def ssh(self, target, cmd, timeout=30):
        self._tick_reboot()
        self.cmds.append(cmd)
        if not self.reachable:
            return 255, "ssh: connect to host 10.23.224.213 port 22: Connection refused"
        if cmd == "true":
            return 0, ""
        if cmd == "hostname":
            return 0, "xsjpalmyra213\n"
        if cmd.startswith("cat /proc/sys/kernel/random/boot_id"):
            return 0, self.boot + "\n"
        if cmd.startswith("tmux has-session"):
            return (0 if self.session else 1), ""
        if cmd.startswith("tmux capture-pane"):
            return (0, "\n".join(self.pane_lines) + "\n") if self.session else (1, "no session")
        if cmd.startswith("tmux list-panes -a"):
            lines = (["hws 500"] if self.session else []) + [f"{n} {pp}" for n, pp in self.other.items()]
            return (0, "".join(l + "\n" for l in lines)) if lines else (1, "no server running")
        if cmd.startswith("tmux new-session"):
            assert "/bin/systest" in cmd and not self.session
            self.session = True
            self.pane_lines = ["Welcome to Systest.", P]
            self.procs.append((500, 1, "huaj", 0, "/usr/bin/perl /bin/systest"))
            return 0, ""
        if cmd.startswith("tmux send-keys"):
            argv = shlex.split(cmd.split("&&")[0])
            text, sess = argv[-1], argv[argv.index("-t") + 1]
            if sess != "hws":
                self.sent.append(f"{sess}:{text}")
                if text == "exit":
                    root = self.other.pop(sess)
                    self.procs = [p for p in self.procs if p[0] != root and p[1] != root]
                return 0, ""
            self._typed(text)
            return 0, ""
        if cmd.startswith("ps -eo"):
            init = "1 0 root 9999 /sbin/init\n"
            return 0, init + "".join(f"{p} {pp} {u} {a} {args}\n" for p, pp, u, a, args in self.procs)
        if cmd.startswith("kill "):
            pids = [int(x) for x in cmd.split(";")[0].split()[1:] if x.isdigit()]
            self.killed.extend(pids)
            dead = [p for p in self.procs if p[0] in pids]
            self.procs = [p for p in self.procs if p[0] not in pids]
            if any("hw_server" in p[4] for p in dead):
                self.hw_server = False
            if self.reboot_on_kill_systest and any("systest" in p[4] for p in dead):
                self.reboot_at = self.t + 15
            return 0, ""
        raise AssertionError(f"unexpected ssh command: {cmd}")

    def _typed(self, text):
        assert self.session
        self.sent.append(text)
        prompt = P
        self.pane_lines[-1] += text
        if text == 'become "palmyra"' and not self.broken_become:
            self.palmyra = True
            self.pane_lines += [" com0 (PS), com1 (PL), com2 (System Controller)",
                                "System control operation failed.", prompt]
        elif text == "hw_server":
            assert not text.startswith("hw_server -"), "-stcp is a syntax error at Systest#"
            self.hw_server = True
            self.procs.append((600, 500, "huaj", 0, "hw_server"))
            self.pane_lines += ["Starting hw_server ... running", prompt]
        elif text == 'become "palmyra"':
            self.pane_lines += ["System control operation failed.", prompt]
        elif text.startswith("hw_server") and "stop" in text:
            if not self.hw_server_ignores_stop:
                self.hw_server = False
                self.procs = [p for p in self.procs if p[0] != 600]
            self.pane_lines.append(prompt)
        elif text == "power 0":
            self.pane_lines.append(prompt)
        elif text == "exit":
            self.session = self.palmyra = False
            self.procs = [p for p in self.procs if p[0] not in (500, 600)]
        else:
            self.pane_lines.append(prompt)


@pytest.fixture
def fake():
    return Fake213()


@pytest.fixture(autouse=True)
def fast_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(hws.devlock, "LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(hws, "LOG_ROOT", str(tmp_path / "logs"))


def run(cmd, fake, **kw):
    return getattr(hws, cmd)(BOARD, io=fake, log=hws.Log(BOARD, cmd, quiet=True), **kw)


def make_up(fake):
    fake.session = fake.palmyra = fake.hw_server = True
    fake.pane_lines = [P + 'become "palmyra"', " com0 (PS), com1 (PL)", P + "hw_server",
                       "Starting hw_server ... running", P]
    fake.procs = [(500, 1, "huaj", 900, "/usr/bin/perl /bin/systest"), (600, 500, "huaj", 800, "hw_server")]


# ---- type / usage ------------------------------------------------------------
def test_board_type_detection():
    assert hws.board_type("huaj@10.23.224.213") == "palmyra"
    assert hws.board_type("crimini1") == "vek385"
    with pytest.raises(hws.HwsError) as e:
        hws.board_type("example.com")
    assert e.value.rc == hws.RC_USAGE


def test_cli_unknown_board_is_rc2(capsys):
    assert hws.main(["--board", "example.com", "status"]) == hws.RC_USAGE


# ---- status ------------------------------------------------------------------
def test_status_down_when_nothing_runs(fake):
    h = run("status", fake)
    assert h.state == hws.DOWN and not h.checks["port"]


def test_status_up_needs_session_port_and_all_cores(fake):
    make_up(fake)
    h = run("status", fake)
    assert h.state == hws.UP and h.rc == 0
    assert h.checks["cores"] == {"Cortex-A78AE": 8, "Cortex-R52": 10}


def test_status_degraded_when_rpu_missing(fake):
    make_up(fake)
    fake.targets = "\n".join(l for l in GOOD_TARGETS.splitlines() if "R52" not in l)
    h = run("status", fake)
    assert h.state == hws.DEGRADED and h.rc == hws.RC_DEGRADED and "targets incomplete" in h.msg


def test_status_foreign_hw_server(fake):
    fake.hw_server = True
    fake.procs = [(700, 1, "sswxtf", 999, "hw_server -d")]
    h = run("status", fake)
    assert h.state == hws.FOREIGN and h.rc == hws.RC_FOREIGN


def test_status_unreachable(fake):
    fake.reachable = False
    h = run("status", fake)
    assert h.state == hws.UNREACHABLE and h.rc == hws.RC_UNREACHABLE


def test_status_is_read_only_and_takes_no_lock(fake, tmp_path):
    make_up(fake)
    import devlock
    with devlock.DeviceLock(BOARD, wait_secs=0):
        assert run("status", fake).state == hws.UP
    assert not any(c.startswith(("tmux send-keys", "tmux new-session", "kill")) for c in fake.cmds)


def test_cli_status_down_is_rc1(fake, monkeypatch, capsys):
    monkeypatch.setattr(hws, "Io", lambda: fake)
    assert hws.main(["--board", BOARD, "--json", "status"]) == hws.RC_DOWN
    assert '"state": "DOWN"' in capsys.readouterr().out.splitlines()[-1]


# ---- up ----------------------------------------------------------------------
def test_up_creates_session_becomes_palmyra_and_starts_plain_hw_server(fake):
    h = run("up", fake)
    assert h.state == hws.UP
    assert fake.sent == ['become "palmyra"', "hw_server"]
    assert not any("-stcp" in c for c in fake.cmds)


def test_up_is_idempotent(fake):
    make_up(fake)
    assert run("up", fake).state == hws.UP
    assert fake.sent == [] and not any(c.startswith("tmux new-session") for c in fake.cmds)


def test_up_restarts_hw_server_in_existing_session(fake):
    make_up(fake)
    fake.hw_server = False
    fake.procs = fake.procs[:1]
    assert run("up", fake).state == hws.UP
    assert fake.sent == ["hw_server"]


def test_up_fails_degraded_with_lab_pointer_after_two_attempts(fake):
    fake.targets = "Versal Gen2 xc2ve3858\nPMC\n"  # JTAG chain without cores
    with pytest.raises(hws.HwsError) as e:
        run("up", fake)
    assert e.value.rc == hws.RC_DEGRADED and "ask Amit" in e.value.msg and "2 attempts" in e.value.msg
    assert fake.sent.count("hw_server") == 2  # exactly 2 start attempts
    assert "exit" in fake.sent  # attempt 2 closed our session cleanly first


def test_up_become_retried_once_then_fails(fake):
    fake.broken_become = True
    with pytest.raises(hws.HwsError) as e:
        run("up", fake)
    assert "setup banner" in e.value.msg
    assert fake.sent.count('become "palmyra"') == 4  # 2 attempts x (become + 1 retry)


def test_up_refuses_foreign(fake):
    fake.hw_server = True
    fake.procs = [(700, 1, "sswxtf", 999, "hw_server -d")]
    with pytest.raises(hws.HwsError) as e:
        run("up", fake)
    assert e.value.rc == hws.RC_FOREIGN and fake.killed == [] and fake.sent == []


def test_up_lock_busy_is_rc4(fake):
    import devlock
    with devlock.DeviceLock(BOARD, wait_secs=0):
        pid = os.fork()
        if pid == 0:  # child: a second process cannot get the flock
            try:
                run("up", fake, wait_lock=0)
                os._exit(0)
            except hws.HwsError as e:
                os._exit(e.rc)
        _, st = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(st) == hws.RC_BUSY


def test_up_locked_true_does_not_take_lock(fake):
    import devlock
    with devlock.DeviceLock(BOARD, wait_secs=0):
        assert run("up", fake, locked=True).state == hws.UP


def test_up_waits_for_ssh_after_reboot(fake):
    fake.reachable = False
    fake.reboot_at = -10  # rebooting: ssh back at t=60
    h = run("up", fake)
    assert h.state == hws.UP and fake.boot == "boot-2"


def test_up_unreachable_after_180s_is_rc6(fake):
    fake.reachable = False
    with pytest.raises(hws.HwsError) as e:
        run("up", fake)
    assert e.value.rc == hws.RC_UNREACHABLE


# ---- down --------------------------------------------------------------------
def test_down_only_via_send_keys_in_order(fake):
    make_up(fake)
    h = run("down", fake)
    assert h.state == hws.DOWN and h.rc == 0
    assert fake.sent == ['hw_server "stop"', "power 0", "exit"]
    assert not any(c.startswith(("kill", "tmux kill")) for c in fake.cmds)


def test_down_falls_back_to_unquoted_stop(fake):
    make_up(fake)
    fake.hw_server_ignores_stop = True
    run("down", fake)
    assert fake.sent[:2] == ['hw_server "stop"', "hw_server stop"]


def test_down_is_idempotent(fake):
    assert run("down", fake).state == hws.DOWN
    assert fake.sent == []


# ---- clean -------------------------------------------------------------------
STALE = [(3278, 1, "huaj", 900, "/usr/bin/perl /bin/systest"),
         (3311, 3278, "huaj", 900, "/tmp/systest.temp.x/systest-client"),
         (4297, 4259, "huaj", 800, "/proj/.../lnx64.o/xsdb"),
         (4389, 1, "huaj", 800, "/proj/.../hw_server -S -d -I10 -s tcp:127.0.0.1:3121")]


def test_clean_kills_stale_holders_logs_them_and_survives_reboot(fake):
    fake.procs = list(STALE) + [(9000, 1, "gvaradar", 900, "xsdb"), (9100, 1, "huaj", 900, "bash")]
    log = hws.Log(BOARD, "clean", quiet=True)
    h = hws.clean(BOARD, io=fake, log=log)
    assert sorted(fake.killed) == [3278, 3311, 4297, 4389]
    assert fake.cmds.index(next(c for c in fake.cmds if c.startswith("kill"))) >= 0
    kill_cmd = next(c for c in fake.cmds if c.startswith("kill"))
    assert kill_cmd.split(" 2>")[0].split()[-1] == "3278"  # /bin/systest last
    text = open(log.path).read()
    for pid in (3278, 3311, 4297, 4389):
        assert f"KILL pid {pid} user huaj" in text
    assert fake.boot == "boot-2" and "rebooting" in text  # waited out the self-reboot
    assert h.state == hws.DOWN


def test_clean_spares_our_session_young_procs_and_other_users(fake):
    make_up(fake)
    fake.procs += [(7000, 1, "huaj", 5, "xsdb"), (9000, 1, "gvaradar", 900, "xsdb")]
    run("clean", fake)
    assert fake.killed == []


def test_clean_spares_processes_started_under_the_current_lock(fake):
    """apppaltest holds the lock (started 300 s ago) and calls hws locked=True: its xsdb survives."""
    fake.t = 10_000
    fake.procs = [(4001, 1, "huaj", 200, "/proj/x/lnx64.o/xsdb"), (4002, 1, "huaj", 5000, "/proj/x/lnx64.o/xsdb")]
    be = hp.Backend(BOARD, fake, hws.Log(BOARD, "clean", quiet=True), lock_owner={"start": 10_000 - 300})
    be.clean()
    assert fake.killed == [4002]


def test_clean_closes_other_tmux_systest_with_exit_not_kill(fake):
    fake.reboot_on_kill_systest = False
    fake.other = {"old": 800}
    fake.procs = [(800, 1, "huaj", 900, "-csh -c /bin/systest"),
                  (801, 800, "huaj", 900, "/usr/bin/perl /bin/systest")]
    run("clean", fake)
    assert fake.killed == [] and "old:exit" in fake.sent and "old" not in fake.other


def test_clean_fails_closed_when_hws_pane_unknown(fake):
    make_up(fake)
    real = fake.ssh
    fake.ssh = lambda t, c, timeout=30: (1, "") if c.startswith("tmux list-panes") else real(t, c, timeout)
    fake.procs.append((7000, 1, "huaj", 900, "xsdb"))
    with pytest.raises(hws.HwsError) as e:
        run("clean", fake)
    assert "refusing to clean" in e.value.msg and fake.killed == []


def test_holder_name_ignores_editors_and_wrappers():
    assert hp.holder_name("/usr/bin/perl /bin/systest") == "systest"
    assert hp.holder_name("/proj/x/unwrapped/lnx64.o/hw_server -S -d") == "hw_server"
    assert hp.holder_name("/opt/jdk/bin/java -classpath s.jar com.xilinx.systest.Command") == "systest-java"
    for a in ("tail -f systest.log", "vim xsdb.tcl", "grep hw_server", "/bin/bash /x/bin/xsdb",
              "tmux new-session -d -s hws /bin/systest", "-csh -c /bin/systest"):
        assert hp.holder_name(a) is None, a


def test_ssh_stall_counts_as_unreachable(fake):
    fake.ssh = lambda t, c, timeout=30: (hws.RC_STALLED, "")
    assert run("status", fake).state == hws.UNREACHABLE


def test_clean_no_reboot_returns_after_grace(fake):
    fake.reboot_on_kill_systest = False
    fake.procs = [(4297, 1, "huaj", 800, "xsdb")]
    run("clean", fake)
    assert fake.killed == [4297] and fake.boot == "boot-1"


# ---- restart / ensure ----------------------------------------------------------
def test_restart_is_down_clean_up(fake):
    make_up(fake)
    h = run("restart", fake)
    assert h.state == hws.UP
    assert fake.sent == ['hw_server "stop"', "power 0", "exit", 'become "palmyra"', "hw_server"]


def test_ensure_noop_when_up(fake):
    make_up(fake)
    assert run("ensure", fake, locked=True).state == hws.UP and fake.sent == []


def test_ensure_heals_down_board(fake):
    assert run("ensure", fake).state == hws.UP


def test_ensure_raises_on_foreign(fake):
    fake.hw_server = True
    fake.procs = [(700, 1, "sswxtf", 999, "hw_server -d")]
    with pytest.raises(hws.HwsError) as e:
        run("ensure", fake)
    assert e.value.rc == hws.RC_FOREIGN


def test_count_targets_parses_real_names():
    assert hp.count_targets(GOOD_TARGETS) == {"Cortex-A78AE": 8, "Cortex-R52": 10}
    assert hp.count_targets("Info: Cortex-A78AE #0.0 (target 20) Running") == {"Cortex-A78AE": 1, "Cortex-R52": 0}


def test_cli_script_maps_backend_errors_to_rc(tmp_path):
    """Run as a script: a backend HwsError must become the exit code, not a traceback."""
    import subprocess
    code = ("import sys, hws, hws_palmyra\n"
            "def boom(self): raise hws.HwsError(hws.RC_DEGRADED, 'HWS FAIL x')\n"
            "hws_palmyra.Backend.up = boom\n")
    p = subprocess.run([sys.executable, "-c", code + "import runpy; sys.argv=['hws.py','--board','10.23.224.213',"
                        "'--wait-lock','0','up']; runpy.run_path('hws.py', run_name='__main__')"],
                       cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True,
                       env={**os.environ, "BOARD_LOCK_DIR": str(tmp_path), "HWS_LOG_DIR": str(tmp_path)})
    assert "Traceback" not in p.stderr and "HWS ERROR" in p.stdout and "rc=3" in p.stdout
