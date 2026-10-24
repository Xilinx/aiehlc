#!/usr/bin/env python3
"""Host client for the netboot supervisor (doc/design/lwip_elf_bootloader.md).

Standard library only. Two ways to reach the board:

  * on the lab host (PALIP), which shares a segment with the board's GEM0: direct TCP;
  * from anywhere that can ssh to PALIP: --via user@PALIP streams the request
    through ssh into netboot_proxy.py on PALIP (copied there once), so the ELF is
    never scp'd and nothing lands on PALIP's disk.

    python3 netload.py load  main.elf [--max-run-ms N]   # send, place, start on the app core
    python3 netload.py load  main.elf --via huaj@10.23.224.213   # same, from the dev box
    python3 netload.py rerun          [--max-run-ms N]   # start the staged ELF again, no transfer
    python3 netload.py kill                              # power the app core down
    python3 netload.py status

Protocol version 2: the board acknowledges the request header before the ELF is
sent, so a busy board costs one round trip, not a 15 MB upload. The ack says
whether the request is admitted and, if not, why and for how long:
  busy      an app is running within its max run time: retry after the time it has left
            (--wait N retries for up to N seconds)
  overtime  the app ran past its max run time: --force terminates it and runs this one
  stuck     forced, but this PDI does not let the supervisor stop the app core:
            park it over JTAG (core 1: stop, rst -proc, pc=park_addr, con) or re-bootstrap;
            apppaltest.py --netload does the JTAG park, falling back to the re-bootstrap
  serving   another client's request is in progress: retry shortly

--max-run-ms sets this run's limit (0 = the board default, 60 s). Past it the
board terminates the app if the PDI allows, and a later --force request may
take the core over.

Exit status: 0 board replied rc == 0; 1 refused or failed; 2 no answer;
3 still busy (after --wait); 4 app past its max, needs --force; 5 stuck, needs
a re-bootstrap. The app's console output is NOT on this socket: it stays on
the board UART (com0), which apppaltest.py --netload reads exactly as for a
JTAG download.
"""
import argparse
import hashlib
import select
import socket
import struct
import subprocess
import sys
import time

PORT = 5155
REQ_MAGIC = 0x3154424E  # 'NBT1'
REP_MAGIC = 0x3152424E  # 'NBR1'
ACK_MAGIC = 0x3141424E  # 'NBA1'
VERSION = 2             # 1 = no ack (the supervisor still serves it)
F_FORCE = 0x1
CMDS = {"load": 1, "rerun": 2, "kill": 3, "status": 4}
STATES = {0: "idle", 1: "running", 2: "killed", 3: "watchdog-timeout", 4: "exited"}
VERDICTS = {0: "ok", 1: "busy", 2: "overtime", 3: "stuck", 4: "serving", -1: "rejected"}
CORES = {0: "free", 1: "parked", 2: "busy"}
EXIT_FOR = {"busy": 3, "serving": 3, "overtime": 4, "stuck": 5, "rejected": 1}
REQ_FMT = "<IIIIQII32s"            # struct netboot_req, 64 B
REP_FMT = "<IiQIIQIIII"            # struct netboot_rep, 48 B
ACK_FMT = "<IiIIIIIIIIII"          # struct netboot_ack, 48 B
assert struct.calcsize(REQ_FMT) == 64 and struct.calcsize(REP_FMT) == 48 and struct.calcsize(ACK_FMT) == 48


class TcpLink:
    """Direct connection (run on the lab host)."""

    def __init__(self, host, connect_timeout, reply_timeout):
        self.s = socket.create_connection((host, PORT), timeout=connect_timeout)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.settimeout(reply_timeout)

    def send_part(self, data):
        self.s.sendall(data)

    def send(self, data):
        self.s.sendall(data)

    def recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.s.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(f"board closed the connection after {len(buf)}/{n} reply bytes")
            buf += chunk
        return buf

    def close(self):
        self.s.close()


class SshProxyLink:
    """ssh <via> python3 <proxy>: the bytes ride ssh stdin/stdout through netboot_proxy.py."""

    def __init__(self, via, proxy, host, connect_timeout, reply_timeout, ssh_opts=()):
        self.reply_timeout = reply_timeout
        self.p = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", *ssh_opts, via, "python3", proxy, "--host", host, "--port", str(PORT),
             "--connect-timeout", str(connect_timeout)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def send_part(self, data):
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def send(self, data):
        self.send_part(data)
        self.p.stdin.close()  # one request per connection: EOF = TCP half-close at the proxy

    def recv_exact(self, n):
        buf = b""
        fd = self.p.stdout.fileno()
        while len(buf) < n:
            if not select.select([fd], [], [], self.reply_timeout)[0]:
                raise ConnectionError(f"no reply within {self.reply_timeout:.0f} s")
            chunk = self.p.stdout.read1(n - len(buf))
            if not chunk:
                err = self.p.stderr.read().decode(errors="replace").strip()
                raise ConnectionError(err or f"proxy closed after {len(buf)}/{n} reply bytes")
            buf += chunk
        return buf

    def close(self):
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()


# Optional callable run before the LAST bytes of a request go out. The supervisor
# cannot start the app before it has every byte, so a caller can overlap its own
# setup (apppaltest: attaching the console) with the transfer and still be ready
# before the app prints anything.
send_gate = None
GATE_TAIL = 4096
WAIT_POLL_S = 2.0  # --wait: longest pause between retries while the board is busy


def _read_ack(link):
    f = struct.unpack(ACK_FMT, link.recv_exact(48))
    magic, verdict, state, core, run_ms, max_ms, left_ms, retry_ms, can_stop, runs, park_addr, msg_len = f
    if magic == REP_MAGIC:  # a version-1 supervisor answered the header with its reply
        raise ConnectionError("the supervisor speaks protocol version 1 (no ack): rebuild/re-bootstrap it, "
                              "or pass --proto 1")
    if magic != ACK_MAGIC:
        raise ConnectionError(f"bad ack magic 0x{magic:08x}")
    msg = link.recv_exact(msg_len).decode(errors="replace") if msg_len else ""
    return {"verdict": VERDICTS.get(verdict, str(verdict)), "state": STATES.get(state, str(state)),
            "core": CORES.get(core, str(core)), "run_ms": run_ms, "max_run_ms": max_ms, "remaining_ms": left_ms,
            "retry_after_ms": retry_ms, "can_stop": {0: "no", 1: "yes"}.get(can_stop, "untried"), "runs": runs,
            "park_addr": park_addr, "msg": msg}


def _exchange(link, hdr, payload, version):
    """One connection: header, (v2) ack, payload if admitted, reply."""
    if version < 2:
        body = hdr + payload
        if send_gate and len(body) > GATE_TAIL:
            link.send_part(body[:-GATE_TAIL])
            send_gate()
            link.send(body[-GATE_TAIL:])
        else:
            link.send(body)
        return None, time.monotonic()
    link.send_part(hdr)
    ack = _read_ack(link)
    if ack["verdict"] != "ok":
        link.send(b"")
        return ack, time.monotonic()
    if send_gate and len(payload) > GATE_TAIL:
        link.send_part(payload[:-GATE_TAIL])
        send_gate()
        link.send(payload[-GATE_TAIL:])
    else:
        link.send(payload)
    return ack, time.monotonic()


def request(host, cmd, payload=b"", timeout_ms=0, connect_timeout=5.0, reply_timeout=60.0, via=None, proxy=None,
            ssh_opts=(), force=False, wait_s=0.0, version=VERSION, on_wait=None):
    """Send one request; with version 2, retry while the board is busy for up to wait_s seconds.

    timeout_ms is the run's max run time (0 = board default). Returns a dict: `admitted`, the ack fields
    (`verdict`, `run_ms`, `max_run_ms`, `remaining_ms`, `retry_after_ms`, `core`, `can_stop`) and, when
    admitted, the reply (`rc`, `bytes_rx`, `rx_ms`, `place_ms`, `entry`, `state`, `runs`, `msg`)."""
    digest = hashlib.sha256(payload).digest() if payload else bytes(32)
    flags = F_FORCE if force else 0
    hdr = struct.pack(REQ_FMT, REQ_MAGIC, version, CMDS[cmd], flags, len(payload), timeout_ms, 0, digest)
    t0 = time.monotonic()
    deadline = t0 + wait_s
    while True:
        t_try = time.monotonic()
        link = (SshProxyLink(via, proxy, host, connect_timeout, reply_timeout, ssh_opts) if via
                else TcpLink(host, connect_timeout, reply_timeout))
        try:
            ack, t_sent = _exchange(link, hdr, payload, version)
            if ack is not None and ack["verdict"] != "ok":
                left = deadline - time.monotonic()
                if ack["verdict"] in ("busy", "serving") and left > 0:
                    # retry_after is the time to the run's LIMIT; it may exit sooner, so look again at least
                    # every WAIT_POLL_S
                    pause = min(max(ack["retry_after_ms"] / 1000.0, 0.2), left, WAIT_POLL_S)
                    if on_wait:
                        on_wait(ack, pause)
                    time.sleep(pause)
                    continue
                return {**ack, "admitted": False, "rc": None, "send_s": t_sent - t_try,
                        "total_s": time.monotonic() - t0}
            rep = struct.unpack(REP_FMT, link.recv_exact(48))
            magic, rc, bytes_rx, rx_ms, place_ms, entry, state, run_ms, runs, msg_len = rep
            if magic != REP_MAGIC:
                raise ConnectionError(f"bad reply magic 0x{magic:08x}")
            msg = link.recv_exact(msg_len).decode(errors="replace") if msg_len else ""
        finally:
            link.close()
        t_done = time.monotonic()
        out = dict(ack or {})
        out.update({
            "admitted": True, "ack_msg": (ack or {}).get("msg", ""), "rc": rc, "bytes_rx": bytes_rx, "rx_ms": rx_ms,
            "place_ms": place_ms, "entry": entry, "state": STATES.get(state, str(state)), "run_ms": run_ms,
            "runs": runs, "msg": msg, "send_s": t_sent - t_try, "total_s": t_done - t0, "waited_s": t_try - t0,
        })
        return out


def _ack_line(cmd, r):
    s = f"[netload] {cmd} not run: board {r['verdict']} (core {r['core']}, state={r['state']}, run {r['runs']}"
    if r["core"] == "busy":
        s += f": {r['run_ms'] / 1000:.1f} s of its {r['max_run_ms'] / 1000:.1f} s max"
        s += f", {r['remaining_ms'] / 1000:.1f} s left" if r["remaining_ms"] else ", past it"
    s += ")"
    hint = {"busy": f"retry in {r['retry_after_ms'] / 1000:.1f} s (or --wait)",
            "serving": f"retry in {r['retry_after_ms'] / 1000:.1f} s (or --wait)",
            "overtime": "resend with --force to terminate it",
            "stuck": "free the app core over JTAG (apppaltest --netload does it) or re-bootstrap the supervisor",
            "rejected": "malformed request"}.get(r["verdict"], "")
    return s + (f"; {hint}" if hint else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=sorted(CMDS))
    ap.add_argument("elf", nargs="?", help="ELF to send (load only)")
    ap.add_argument("--host", default="10.10.70.50", help="board IP (netboot build.sh NETBOOT_IP)")
    ap.add_argument("--max-run-ms", "--timeout-ms", dest="max_run_ms", type=int, default=0,
                    help="this run's max run time; 0 = the board default (60 s). Past it the board stops the app "
                         "where the PDI allows, and a --force request may take the core over")
    ap.add_argument("--force", action="store_true",
                    help="terminate an app that ran past its max run time and run this one instead")
    ap.add_argument("--wait", type=float, default=0.0, metavar="SEC",
                    help="while the board is busy (app within its max, or another client), retry for up to SEC "
                         "seconds, sleeping for the time the board says is left")
    ap.add_argument("--proto", type=int, default=VERSION, choices=(1, 2),
                    help="protocol version (1: no ack, for a supervisor built before version 2)")
    ap.add_argument("--connect-timeout", type=float, default=5.0)
    ap.add_argument("--via", metavar="USER@HOST", help="reach the board through netboot_proxy.py on this host")
    ap.add_argument("--proxy", default="~/aiehlc/netboot/netboot_proxy.py", help="proxy path on the --via host")
    ap.add_argument("--ssh-opt", action="append", default=[], metavar="OPT",
                    help="extra `ssh -o` option for --via (repeatable), e.g. ControlPath=... to reuse a connection")
    a = ap.parse_args(argv)

    payload = b""
    if a.cmd == "load":
        if not a.elf:
            ap.error("load needs an ELF path")
        with open(a.elf, "rb") as f:
            payload = f.read()
    try:
        ssh_opts = [x for o in a.ssh_opt for x in ("-o", o)]
        def on_wait(ack, pause):
            print(_ack_line(a.cmd, ack) + f" -- waiting {pause:.1f} s", flush=True)
        r = request(a.host, a.cmd, payload, a.max_run_ms, a.connect_timeout, via=a.via, proxy=a.proxy,
                    ssh_opts=ssh_opts, force=a.force, wait_s=a.wait, version=a.proto, on_wait=on_wait)
    except (OSError, ConnectionError) as e:
        print(f"[netload] {a.cmd}: no answer from {a.host}:{PORT}: {e}")
        return 2
    if not r["admitted"]:
        print(_ack_line(a.cmd, r))
        if r["msg"]:
            print(f"[netload] board: {r['msg']}")
        if r["verdict"] == "stuck" and r.get("park_addr"):
            # a debugger can free the app core without a re-bootstrap: core 1 stop, rst -proc, pc=<this>, con
            print(f"[netload] park_addr=0x{r['park_addr']:x}")
        return EXIT_FOR.get(r["verdict"], 1)
    if r.get("ack_msg"):
        print(f"[netload] board: {r['ack_msg']}")
    mb = len(payload) / 1e6
    line = f"[netload] {a.cmd} rc={r['rc']} state={r['state']} runs={r['runs']}"
    if a.cmd == "load":
        rate = mb / r["send_s"] if r["send_s"] > 0 else 0.0
        how = f" via {a.via}" if a.via else ""
        line += (f" | {mb:.2f} MB sent{how} in {r['send_s']:.3f} s ({rate:.1f} MB/s), board rx {r['rx_ms']} ms,"
                 f" place {r['place_ms']} ms, entry 0x{r['entry']:x}")
        if r.get("waited_s", 0) > 0.05:
            line += f", after waiting {r['waited_s']:.1f} s for the board"
    elif r["state"] == "running":
        line += f" run_ms={r['run_ms']}"
    print(line)
    if r["msg"]:
        print(f"[netload] board: {r['msg']}")
    return 0 if r["rc"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
