#!/usr/bin/env python3
"""Helpers for streamload_x86.sh (no board involved).

  capture OUT netload-args...   run netload.py's main() against a local fake board that
                                acks (version 2) and replies rc 0, and save the request
                                bytes (header + payload) exactly as netload sent them
  mkbig SRC OUT MB ADDR         SRC plus one extra PT_LOAD of MB MiB of pseudo-random
                                "weights" at ADDR, so the ELF is larger than staging
"""
import importlib.util
import os
import random
import socket
import struct
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))


def _netload():
    spec = importlib.util.spec_from_file_location("netload", os.path.join(HERE, "..", "netload.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _recv_exact(c, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = c.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError(f"client closed after {len(buf)}/{n}")
        buf += chunk
    return bytes(buf)


def capture(out, argv):
    nl = _netload()
    ls = socket.socket()
    ls.bind(("127.0.0.1", 0))
    ls.listen(1)
    nl.PORT = ls.getsockname()[1]
    box = {}

    def board():
        c, _ = ls.accept()
        hdr = _recv_exact(c, 64)
        magic, ver, cmd, flags, size, max_ms, _res, _sha = struct.unpack(nl.REQ_FMT, hdr)
        if ver >= 2:
            c.sendall(struct.pack(nl.ACK_FMT, nl.ACK_MAGIC, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0))
        payload = _recv_exact(c, size) if cmd == nl.CMDS["load"] else b""
        c.sendall(struct.pack(nl.REP_FMT, nl.REP_MAGIC, 0, len(payload), 0, 0, 0, 0, 0, 0, 0))
        c.close()
        with open(out, "wb") as f:
            f.write(hdr + payload)
        box["v"] = ver

    t = threading.Thread(target=board, daemon=True)
    t.start()
    rc = nl.main(argv + ["--host", "127.0.0.1", "--connect-timeout", "5"])
    t.join(60)
    print(f"[capture] {out}: version {box.get('v')} netload rc {rc}")
    return 0 if rc == 0 and "v" in box else 1


def mkbig(src, out, mb, addr):
    b = bytearray(open(src, "rb").read())
    (phoff,) = struct.unpack_from("<Q", b, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", b, 0x36)
    assert phentsize == 56
    slot = phoff + phnum * 56
    # the new program header goes in the free bytes after the table (before the first segment/section)
    first = min(struct.unpack_from("<Q", b, phoff + i * 56 + 8)[0] for i in range(phnum)
                if struct.unpack_from("<I", b, phoff + i * 56)[0] == 1)
    assert slot + 56 <= first and not any(b[slot:slot + 56]), "no room for one more program header"
    off = (len(b) + 0xFFFF) & ~0xFFFF
    n = mb << 20
    rng = random.Random(1113)
    weights = rng.randbytes(n)
    b += bytes(off - len(b))
    b += weights
    struct.pack_into("<IIQQQQQQ", b, slot, 1, 6, off, addr, addr, n, n, 0x10000)  # PT_LOAD RW
    struct.pack_into("<H", b, 0x38, phnum + 1)
    with open(out, "wb") as f:
        f.write(b)
    print(f"[mkbig] {out}: {len(b)} B, PT_LOAD {n} B at 0x{addr:x} from file offset 0x{off:x}")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        sys.exit(capture(sys.argv[2], sys.argv[3:]))
    if sys.argv[1] == "mkbig":
        sys.exit(mkbig(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5], 0)))
    sys.exit(__doc__)
