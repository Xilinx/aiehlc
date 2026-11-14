"""Unit tests for the VEK385 hws backend with a fake farm host (mocked cluster-ping,
systest hold session, nc, xsdb, ssh)."""
import base64
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import hws  # noqa: E402
import hws_vek385 as hv  # noqa: E402

HOST = "crimini2"
GOOD = "\n".join(
    ["  1  Versal xc2ve3858", "     2  PMC", "     3  PSM", "     4  APU"]
    + [f"        {10 + i}  Cortex-A78AE #{i // 2}.{i % 2} (Running)" for i in range(8)]
    + ["     20  RPU"] + [f"        {30 + i}  Cortex-R52 #{i // 2}.{i % 2} (Running)" for i in range(10)])
CP = (" vek385-12      crimini2       [B-revA1,reserved-ssw_aie,usb3] {h}\n"
      " vek385-5       gomphus5       [B-revA1,usb2] sswxtf 13395452 57m:9s   PM_APPS\n")


class FakeFarm(hws.Io):
    def __init__(self):
        self.t = 0.0
        self.holder = "ok"
        self.dispatch = True       # LSF gives us the board
        self.hold = False          # our systest session is at Systest#
        self.hw_server = False
        self.targets = GOOD
        self.stop_works = True     # `hw_server "stop"` closes 3121
        self.start_works = True
        self.ssh_ok = True
        self.console = ""
        self.eat_next = False      # prompt swallows the line after a stop (seen on vek385-12)
        self.sent, self.ssh_cmds, self.killed = [], [], []
        self.raw_ssh_cmds = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def port_open(self, host, port=3121, timeout=3):
        return self.hw_server

    def xsdb_targets(self, host, port=3121, timeout=60):
        return (0, self.targets) if self.hw_server else (1, "HWS_CONNECT_ERR refused")

    def ssh(self, target, cmd, timeout=30):
        self.raw_ssh_cmds.append(cmd)
        m = re.fullmatch(r"echo ([A-Za-z0-9+/=]+) \| base64 -d \| bash", cmd)
        if m:  # a clean script shipped csh-safe: record the bash it runs
            cmd = base64.b64decode(m.group(1)).decode()
        self.ssh_cmds.append(cmd)
        if not self.ssh_ok:
            return 255, "Permission denied (publickey)"
        if "pkill -u huaj -x hw_server" in cmd:
            self.hw_server = False
        return 0, "4242 xsdb\n" if "xsdb" in cmd else ""

    # systest prompt behaviour
    def type(self, line):
        self.sent.append(line)
        if not self.hold:
            return False
        out = ""
        if self.eat_next:
            self.eat_next = False
            self.console += "Error: line 1: token recognition error at: ''\nSyntax error\n[vek385-12] Systest# "
            return True
        if line == "hw_server":
            if self.hw_server:
                out = "Vivado hardware server is running on: crimini2:3121\n"
            else:
                out = "Starting Vivado hardware server.\n"
                self.hw_server = self.start_works
        elif line == 'hw_server "stop"':
            if self.stop_works:
                self.hw_server = False
                out = 'Device hw_server command: "hw_server" (pid 9) exited with status 137.\n'
        elif line == "exit":
            self.hold = False
            self.release_at = self.t + 40  # cluster-ping lags behind the exit
            self.hw_server = False
            self.console += "Reboot is disabled!\n------ Systest server exiting ------\n"
            return True
        self.console += out + "[vek385-12] Systest# "
        return True


class FakeBackend(hv.Backend):
    def __init__(self, farm, tmp_path, board=HOST):
        self.farm = farm
        hws_root, hws.LOG_ROOT = hws.LOG_ROOT, str(tmp_path)
        try:
            super().__init__(board, farm, lambda m: None)
        finally:
            hws.LOG_ROOT = hws_root
        self.me = "huaj"

    def cluster_ping(self):
        f = self.farm
        if getattr(f, "release_at", None) is not None and f.t >= f.release_at:
            f.holder, f.release_at = "ok", None
        h = f.holder
        return hv.parse_cluster_ping(CP.format(h="ok" if h == "ok" else f"{h} {13396000 if h == 'huaj' else 13395452} 1m:0s"))

    def spawn_hold(self, vboard):
        f = self.farm
        f.spawns = getattr(f, "spawns", 0) + 1
        open(self.fifo, "w").close()
        if f.dispatch:
            f.hold, f.holder = True, "huaj"
            f.console = "Job <13396000> is submitted to queue <hwboard>.\nPower: ON\n[vek385-12] Systest# "
        else:
            f.console = "Job <13396000> is submitted\n<<Waiting for dispatch ...>>\n"
        return {"script": 0, "writer": 0}

    def send(self, cmd):
        return self.farm.type(cmd)

    def console(self):
        return self.farm.console

    def kill_own(self, pid):
        self.farm.killed.append(pid)


@pytest.fixture
def farm():
    return FakeFarm()


@pytest.fixture
def be(farm, tmp_path):
    return FakeBackend(farm, tmp_path)


def make_up(be, farm):
    h = be.up()
    assert h.state == hws.UP
    return h


# ---- parsing ---------------------------------------------------------------------
def test_cluster_ping_parses_free_and_held_boards():
    cp = hv.parse_cluster_ping(CP.format(h="ok"))
    assert cp["crimini2"] == ("vek385-12", "ok", None)
    assert cp["gomphus5"] == ("vek385-5", "sswxtf", "13395452")


def test_targets_counts_unique_cores_and_needs_versal_pmc():
    ok, n = hv.targets_good(0, GOOD)
    assert ok and n == {"Cortex-A78AE": 8, "Cortex-R52": 10}
    assert not hv.targets_good(0, GOOD.replace("Cortex-R52", "Cortex-R5x"))[0]
    assert not hv.targets_good(0, GOOD.replace("PMC", "XXX"))[0]
    assert not hv.targets_good(1, GOOD)[0]


# ---- status ------------------------------------------------------------------------
def test_status_down_when_board_free_and_no_session(be):
    h = be.status()
    assert (h.state, h.rc) == (hws.DOWN, hws.RC_OK)


def test_status_foreign_when_other_user_holds_board(be, farm):
    farm.holder, farm.hw_server = "sswxtf", True
    h = be.status()
    assert (h.state, h.rc) == (hws.FOREIGN, hws.RC_FOREIGN)


def test_status_foreign_for_huaj_hold_outside_hws(be, farm):
    farm.holder, farm.hw_server = "huaj", True
    assert be.status().state == hws.FOREIGN


def test_status_degraded_when_port_open_without_session(be, farm):
    farm.hw_server = True  # e.g. an orphan `hw_server -d`
    assert be.status().state == hws.DEGRADED


def test_unknown_farm_host_is_usage_error(farm, tmp_path):
    with pytest.raises(hws.HwsError) as e:
        FakeBackend(farm, tmp_path, board="portobello99").status()
    assert e.value.rc == hws.RC_USAGE


# ---- up ------------------------------------------------------------------------------
def test_up_creates_hold_starts_hw_server_and_is_up(be, farm):
    h = make_up(be, farm)
    assert farm.sent[:2] == [f"$SYSTEST_AUTOLOGOUT = {hv.AUTOLOGOUT_SECS}", "hw_server"]
    assert h.checks["cores"] == {"Cortex-A78AE": 8, "Cortex-R52": 10}
    assert be.owner()["job"] == "13396000"


def test_up_is_idempotent(be, farm):
    make_up(be, farm)
    n = len(farm.sent)
    assert be.up().state == hws.UP
    assert len(farm.sent) == n  # nothing typed


def test_up_never_touches_foreign_board(be, farm):
    farm.holder = "sswxtf"
    with pytest.raises(hws.HwsError) as e:
        be.up()
    assert e.value.rc == hws.RC_FOREIGN
    assert farm.sent == [] and farm.ssh_cmds == []


def test_up_gives_up_after_two_starts_with_lab_pointer(be, farm):
    farm.start_works = False
    with pytest.raises(hws.HwsError) as e:
        be.up()
    assert e.value.rc == hws.RC_DEGRADED and "ask Amit" in e.value.msg
    assert farm.sent.count("hw_server") == hv.START_ATTEMPTS


def test_up_fails_when_targets_miss_r52(be, farm):
    farm.targets = GOOD.replace("Cortex-R52", "Cortex-Rxx")
    with pytest.raises(hws.HwsError) as e:
        be.up()
    assert e.value.rc == hws.RC_DEGRADED and "H3 targets" in e.value.msg


def test_up_without_dispatch_retries_once_then_fails(be, farm):
    farm.dispatch = False
    with pytest.raises(hws.HwsError) as e:
        be.up()
    assert e.value.rc == hws.RC_DEGRADED
    assert farm.spawns == 2 and farm.sent.count("exit") == 2  # late dispatch exits cleanly


def test_up_restarts_hw_server_in_existing_hold(be, farm):
    make_up(be, farm)
    farm.hw_server = False  # device died, hold alive
    assert be.up().state == hws.UP
    assert farm.spawns == 1


# ---- down ------------------------------------------------------------------------------
def test_down_stops_powers_off_and_exits_cleanly(be, farm):
    make_up(be, farm)
    h = be.down()
    assert (h.state, h.rc) == (hws.DOWN, hws.RC_OK)
    assert farm.sent[-3:] == ['hw_server "stop"', "power 0", "exit"]
    assert not any("kill" in c and "systest" in c and "client" not in c for c in farm.ssh_cmds)


def test_down_falls_back_to_own_pkill_when_stop_is_ignored(be, farm):
    make_up(be, farm)
    farm.stop_works = False
    assert be.down().state == hws.DOWN
    assert any("pkill -u huaj -x hw_server" in c for c in farm.ssh_cmds)


def test_down_is_idempotent_when_already_down(be, farm):
    assert be.down().state == hws.DOWN
    assert farm.sent == []


def test_down_refuses_foreign(be, farm):
    farm.holder = "sswxtf"
    with pytest.raises(hws.HwsError) as e:
        be.down()
    assert e.value.rc == hws.RC_FOREIGN


# ---- clean / restart ----------------------------------------------------------------
def test_clean_keeps_session_telnets_while_hold_alive(be, farm):
    make_up(be, farm)
    be.clean()
    script = farm.ssh_cmds[-1]
    assert "xsdb" in script and "ystest-client" in script and "telnet" not in script
    assert "pkill -u huaj" in script


def test_clean_spares_the_hold_sessions_own_systest_client(be, farm):
    # s6 on gomphus3: `pkill -f systest-client` also hit the hold's own client (child of the
    # perl /bin/systest), which would end our live session uncleanly.
    make_up(be, farm)
    be.clean()
    script = farm.ssh_cmds[-1]
    assert "pkill -u huaj -f systest-client" not in script
    assert "ystest-client" in script and "bin/systest" in script


def test_clean_ssh_command_is_csh_safe(be, farm):
    # s6 on gomphus3: the farm login shell is tcsh; `bash -c '<quoted>'` died with
    # "Unmatched '" / "Illegal variable name" and clean killed nothing (but said rc 0).
    make_up(be, farm)
    be.clean()
    raw = farm.raw_ssh_cmds[-1]
    assert re.fullmatch(r"echo [A-Za-z0-9+/=]+ \| base64 -d \| bash", raw)


def test_clean_patterns_do_not_match_the_clean_script_itself(be, farm):
    import re
    be.clean()
    script = farm.ssh_cmds[-1]
    for pat in ("[s]ystest-client", "telnet 10.10.71.1 40[0]"):
        assert pat in script and not re.search(pat, script)


def test_clean_systest_client_guard_in_a_real_shell(be, farm):
    import subprocess
    make_up(be, farm)
    be.clean()
    loop = [l for l in farm.ssh_cmds[-1].split("\n") if "ystest-client" in l and "for p" in l][0]
    # stub pgrep/ps/kill: pid 11 = the session's client (parent perl systest), 12 = a stray one
    stub = """pgrep() { echo 11; echo 12; }
ps() { case "$*" in *ppid=*11) echo 10;; *ppid=*12) echo 1;;
       *args=*10) echo /usr/bin/perl /proj/systest/x/bin/systest;; *args=*1) echo init;;
       *) echo "$4 systest-client";; esac; }
kill() { [ "$1" = -9 ] && shift; echo KILLED "$@"; }
sleep() { :; }
"""
    out = subprocess.run(["bash", "-c", stub + loop], capture_output=True, text=True).stdout
    assert "KILLED 12" in out and "11" not in out.replace("KILLED", "")


def test_clean_without_session_also_kills_stale_telnets(be, farm):
    be.clean()
    assert "telnet 10.10.71.1 40[0]" in farm.ssh_cmds[-1]


def test_clean_without_ssh_is_not_an_error(be, farm):
    farm.ssh_ok = False
    h = be.clean()
    assert h.rc == hws.RC_OK and "no ssh" in h.msg


def test_restart_keeps_the_hold(be, farm):
    make_up(be, farm)
    h = be.restart()
    assert h.state == hws.UP and farm.spawns == 1
    assert 'hw_server "stop"' in farm.sent and "exit" not in farm.sent


def test_restart_from_down_brings_it_up(be, farm):
    assert be.restart().state == hws.UP


def test_dead_session_by_autologout_is_not_alive(be, farm):
    make_up(be, farm)
    farm.console += "AUTOLOGOUT!\n"
    farm.hold, farm.hw_server = False, False
    assert be.session()[0] is False


def test_restart_retypes_hw_server_when_prompt_eats_it(be, farm):
    make_up(be, farm)
    orig = farm.type

    def typ(line):
        r = orig(line)
        if line == 'hw_server "stop"':
            farm.eat_next = True
        return r
    farm.type = typ
    h = be.restart()
    assert h.state == hws.UP
    assert farm.sent.count('hw_server "stop"') == 1  # no second start attempt needed


def test_down_waits_until_cluster_ping_frees_the_board(be, farm):
    make_up(be, farm)
    be.down()
    assert farm.holder == "ok"
    assert be.up().state == hws.UP and farm.spawns == 2


def test_full_restart_down_clean_up_takes_a_new_hold(be, farm):
    make_up(be, farm)
    be.down(); be.clean()
    assert be.up().state == hws.UP


def test_huaj_hold_with_other_job_is_not_waited_for(be, farm):
    make_up(be, farm)
    be.down()
    farm.holder = "huaj"
    farm.release_at = None
    be.write_owner(dict(be.owner(), job="111"))  # our ended job differs from the holder's
    with pytest.raises(hws.HwsError) as e:
        be.up()
    assert e.value.rc == hws.RC_FOREIGN
