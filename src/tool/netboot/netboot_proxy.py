#!/usr/bin/env python3
"""stdio <-> TCP relay to the netboot supervisor. Lives on the lab host (PALIP).

The dev box cannot reach the board's network (10.10.70.0/24 hangs off PALIP's
eth1), so netload.py runs this over ssh and speaks the netboot protocol through
the ssh pipe:

    dev box: netload.py --via huaj@PALIP  ==ssh stdin/stdout==>  this relay  ==TCP==>  board :5155

The ELF streams straight through; nothing is written to PALIP's disk.
stdin EOF is forwarded as a TCP half-close, and the relay exits when the
supervisor closes the connection (after its reply). Standard library only.
Exit status: 0 relayed, 2 could not reach the supervisor.
"""
import argparse
import os
import select
import socket
import sys


def write_all(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="10.10.70.50")
    ap.add_argument("--port", type=int, default=5155)
    ap.add_argument("--connect-timeout", type=float, default=5.0)
    a = ap.parse_args(argv)
    try:
        sock = socket.create_connection((a.host, a.port), timeout=a.connect_timeout)
    except OSError as e:
        sys.stderr.write(f"[netboot-proxy] cannot reach {a.host}:{a.port}: {e}\n")
        return 2
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    stdin_open = True
    while True:
        rlist = [sock] + ([0] if stdin_open else [])
        ready, _, _ = select.select(rlist, [], [])
        if 0 in ready:
            data = os.read(0, 1 << 20)
            if data:
                sock.sendall(data)
            else:
                stdin_open = False
                sock.shutdown(socket.SHUT_WR)
        if sock in ready:
            data = sock.recv(1 << 20)
            if not data:
                break
            write_all(1, data)
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
