#!/usr/bin/env python3
"""Faithful simulator of __Runtime_core_trace_decode (src/mlir/runtime/aie_runtime.c).

Decodes AIE2ps Event-Time trace frames (Architecture Spec, Figure 4-14). The C
function's signature is:

    void __Runtime_core_trace_decode(const uint32_t *buf, uint32_t nwords,
                                     AieTraceProfile *prof);

and the caller (example/perf/aieml_perf.cc) passes nwords = TRC_LEN/4, i.e. the
FULL buffer capacity, so the all-zero packet check terminates decoding. The
optional prof sink (C-side interval capture) does not affect decode output and
is not modelled here; this simulator mirrors the printed timeline only.

TRANSPORT FRAMING (hardware-defined): the trace unit output is a packet-switched
stream hardened to 8-word packets:
    word[0]    = stream-packet routing header (packet ID + packet type)
    word[1..7] = 7 x 32-bit trace payload
__Runtime_core_trace_setup routes with XAIE_SS_PKT_DONOT_DROP_HEADER, so the
header word IS present in the buffer and is skipped. The 7 payload words of every
packet are concatenated into ONE contiguous frame stream. An all-zero 8-word
packet marks the untouched buffer tail and stops decoding.

PAYLOAD ENCODING -- Event-Time trace frames (Figure 4-14). Variable-width frames
packed MSB-first (bit 31 first); the trace unit aligns them to its 32-bit output
words with 8-bit Filler frames, so no non-Start frame straddles a word.
Mode bits [25:24] = 0b00 (Event-Time).

    Frame      Width  Prefix (from bit31)   Fields
    Single0     8b    0                     event(3)  cycles(4)
    Single1    16b    100                   event(3)  cycles(10)
    Single2    24b    101                   event(3)  cycles(18)
    Multiple0  16b    1100                  mask(8)   cycles(4)
    Multiple1  24b    110100                mask(8)   cycles(10)
    Multiple2  32b    110101                mask(8)   cycles(18)
    Repeat0     8b    1110                  repeats(4)
    Repeat1    16b    110110                repeats(10)
    Start      64b    11110 <OR> 00         timer(56, absolute cycle base)
    Stop       32b    110111  x(8)          cycles(18)
    Filler      8b    11111110 (0xFE)       32-bit alignment pad (skipped)
    Sync        8b    11111111 (0xFF)       0x3FFFF idle cycles

"cycles" is the count since the previous frame -> cycle += cycles, then event(s)
are emitted at the new cycle. Single carries a 3-bit event index; Multiple an
8-bit event bitmap. Event index -> slot name follows __Runtime_core_trace_setup
(0..3 = ACTIVE, LOCK_STALL, STREAM_STALL, MEMORY_STALL;
4..6 = PORT_IDLE_0, PORT_RUNNING_0, PORT_STALLED_0). Repeat re-emits the
previous frame's event set for `repeats` more consecutive cycles (or, after a
Sync, adds that many 0x3FFFF idle periods). Frame bit layouts are high confidence
(Figure 4-14); Repeat/Sync cycle accumulation is the documented compression.

OUTPUT (faithful mode) is run-length compressed: contiguous cycles carrying the
identical event mask collapse into one interval line
"<start> -- <end>  <events>  (<N> cyc)"; a single cycle keeps the compact form
"<cycle>  <events>". Multi-bit masks join slot names with '|', e.g.
"142848 -- 143000  STREAM_STALL|EVENT5|EVENT6  (153 cyc)".

Usage:
    core_trace_decode.py 0x00000001 0xF0000000 0x000003E8 ...   # 8-word packets
    core_trace_decode.py -v ...     # step-by-step frame-by-frame explanation
    core_trace_decode.py            # runs a built-in demo (one 8-word packet)
    echo "0x1 0xF0000000 ..." | core_trace_decode.py -   # words from stdin
"""

import sys

PACKET_WORDS = 8          # 1 header + 7 payload
PAYLOAD_WORDS = 7
SYNC_CYCLES = 0x3FFFF     # Sync frame = 18-bit count wrap

# s_core_trace_slot_name[8] from aie_runtime.c. Slots 0..3 are core-state
# events; 4..6 are the stream-switch port-0 event group; slot 7 is unused.
SLOT_NAMES = ("ACTIVE", "LOCK_STALL", "STREAM_STALL", "MEMORY_STALL",
              "PORT_IDLE_0", "PORT_RUNNING_0", "PORT_STALLED_0", "EVENT7")

# s_mem_trace_slot_name[8] from aie_runtime.c: the memory-module DMA trace unit
# (pkt id 2). Slots 0..4 are the DMA BD/stall events, 5..7 are the lock group /
# acquire (SEL0 ACQ_GE) / release (LOCK_0_REL, event 46 -- valid on the
# compute-tile mem module for both AIE-ML and AIE2PS). The channel+direction
# (S2MM/MM2S) is carried in the [TRACESTREAMCONFIG] line, not the slot names, so
# these are generic.
MEM_SLOT_NAMES = ("DMA_START", "DMA_FINISH", "DMA_STALL_LOCK", "STREAM_STALL",
                  "MEM_BP", "LOCK_GRP", "LOCK_ACQ", "LOCK_REL")


def table_for_id(pkt_id):
    """Map a stream packet id to its 8-slot event name table: pkt 2 = mem-module
    DMA trace, everything else (pkt 1 core, or legacy id 0 no-header captures) =
    core trace. Mirrors __core_trace_table_for_id in aie_runtime.c."""
    return MEM_SLOT_NAMES if pkt_id == 2 else SLOT_NAMES


def slot_name(s, names=SLOT_NAMES):
    """Event index -> slot name using the active table (0..7 configured)."""
    return names[s] if s < 8 else f"EVENT{s}"


def event_category(name, stream="core"):
    """Trace-slot event -> category. The mem-module DMA stream is its own 'mem'
    category (distinct timeline lane). Within the core stream, the stream-switch
    PORT_* events describe port activity ('event' category); ACTIVE and the
    *_STALL / LOCK states describe the core itself ('core' category)."""
    if stream == "mem":
        return "mem"
    return "event" if name.startswith("PORT_") else "core"


class CoreTraceDecoder:
    """Stateful Event-Time frame decoder mirroring __Runtime_core_trace_decode.

    Output modes:
        "faithful" (default) -- reproduce the C printf output verbatim.
        "verbose"            -- frame-by-frame explanation.
        "quiet"              -- no printing; just return the event timeline.
    """

    def __init__(self, mode="faithful"):
        self.mode = mode
        self.cycle = 0
        self.last_kind = None   # None | "single" | "multiple" | "sync"
        self.last_events = 0    # single: slot index; multiple: 8-bit mask
        self.payload = []       # header-stripped payload words
        self.total_bits = 0
        self.bitpos = 0
        self.timeline = []
        self.run = None         # open run-length interval: (start, end, mask)
        self.names = SLOT_NAMES  # active slot-name table (set per pkt-id group)
        self.stream = "core"     # active stream tag ("core" | "mem")

    # -- bit reader (MSB-first over the header-stripped payload stream) ------
    def _read(self, nbits, advance=True):
        v = 0
        for i in range(nbits):
            bp = self.bitpos + i
            w = self.payload[bp >> 5]
            off = bp & 31
            v = (v << 1) | ((w >> (31 - off)) & 1)
        if advance:
            self.bitpos += nbits
        return v

    def _peek8(self):
        return self._read(8, advance=False)

    # -- run-length interval tracker (mirrors __core_trace_run in the C code) -
    def _names(self, mask):
        """'|'-joined slot names of an 8-bit event mask (slot 0..7 order), using
        the active pkt-id group's slot-name table."""
        return "|".join(slot_name(s, self.names) for s in range(8) if mask & (1 << s))

    def _run_flush(self):
        """Print the open run and close it: compact for a single cycle, else
        '<start> -- <end>  <events>  (<N> cyc)'. No-op when no run is open (so
        it is safe to call unconditionally, and inert in verbose/quiet mode)."""
        if self.run is None:
            return
        start, end, mask = self.run
        names = self._names(mask)
        if start == end:
            print(f"{start}  {names}")
        else:
            print(f"{start} -- {end}  {names}  ({end - start + 1} cyc)")
        self.run = None

    def _run_add(self, cycle, mask):
        """Extend the open run when the cycle is contiguous and the mask
        identical, else flush and open a new one."""
        if self.run is not None:
            start, end, rmask = self.run
            if cycle == end + 1 and mask == rmask:
                self.run = (start, cycle, mask)
                return
        self._run_flush()
        self.run = (cycle, cycle, mask)

    def _mark(self, cycle, mask):
        """Route one per-cycle event mask. Always records the per-bit
        (cycle, name, category) timeline (the semantic return value); category
        is 'event' for PORT_* and 'core' otherwise. In faithful mode drives the
        run tracker, in verbose mode keeps the per-slot '-> cycle name' lines.
        mask==0 is a no-op."""
        if mask == 0:
            return
        for s in range(8):
            if mask & (1 << s):
                name = slot_name(s, self.names)
                self.timeline.append((cycle, name, event_category(name, self.stream)))
        if self.mode == "faithful":
            self._run_add(cycle, mask)
        elif self.mode == "verbose":
            for s in range(8):
                if mask & (1 << s):
                    name = slot_name(s, self.names)
                    print(f"        -> {cycle}  {name}  [{event_category(name, self.stream)}]")

    def _repeat(self, rep):
        if self.mode == "verbose":
            print(f"      repeat x{rep} of last {self.last_kind}")
        if self.last_kind == "sync":
            self.cycle += rep * SYNC_CYCLES
        elif self.last_kind == "single":
            for _ in range(rep):
                self.cycle += 1
                self._mark(self.cycle, 1 << self.last_events)
        elif self.last_kind == "multiple":
            for _ in range(rep):
                self.cycle += 1
                self._mark(self.cycle, self.last_events)

    # -- one frame ----------------------------------------------------------
    def step_frame(self):
        """Decode ONE frame at the current bit position. Returns False if the
        remaining stream cannot hold the frame (stop)."""
        b0 = self._peek8()

        if (b0 & 0x80) == 0:                       # Single0 8b: 0 eee cccc
            v = self._read(8)
            self.cycle += v & 0xF
            self.last_events, self.last_kind = (v >> 4) & 0x7, "single"
            if self.mode == "verbose":
                print(f"    Single0  0x{v:02X}  event={self.last_events} +{v & 0xF} -> {self.cycle}")
            self._mark(self.cycle, 1 << self.last_events)
        elif (b0 & 0x40) == 0:                      # Single1/2: 10x
            nb = 24 if (b0 & 0x20) else 16
            if self.bitpos + nb > self.total_bits:
                return False
            v = self._read(nb)
            sh = 18 if nb == 24 else 10
            self.cycle += v & ((1 << sh) - 1)
            self.last_events, self.last_kind = (v >> sh) & 0x7, "single"
            if self.mode == "verbose":
                name = "Single2" if nb == 24 else "Single1"
                print(f"    {name}  0x{v:0{nb // 4}X}  event={self.last_events} "
                      f"+{v & ((1 << sh) - 1)} -> {self.cycle}")
            self._mark(self.cycle, 1 << self.last_events)
        elif (b0 & 0x20) == 0:                      # 110...
            if (b0 & 0x10) == 0:                    # Multiple0 16b: 1100 m(8) c(4)
                if self.bitpos + 16 > self.total_bits:
                    return False
                v = self._read(16)
                self.cycle += v & 0xF
                self.last_events, self.last_kind = (v >> 4) & 0xFF, "multiple"
                if self.mode == "verbose":
                    print(f"    Multiple0  0x{v:04X}  mask=0b{self.last_events:08b} "
                          f"+{v & 0xF} -> {self.cycle}")
                self._mark(self.cycle, self.last_events)
            else:
                sel = (b0 >> 2) & 0x3               # 1101 xx
                if sel == 0:                        # Multiple1 24b
                    if self.bitpos + 24 > self.total_bits:
                        return False
                    v = self._read(24)
                    self.cycle += v & 0x3FF
                    self.last_events, self.last_kind = (v >> 10) & 0xFF, "multiple"
                    if self.mode == "verbose":
                        print(f"    Multiple1  0x{v:06X}  mask=0b{self.last_events:08b} "
                              f"+{v & 0x3FF} -> {self.cycle}")
                    self._mark(self.cycle, self.last_events)
                elif sel == 1:                      # Multiple2 32b
                    if self.bitpos + 32 > self.total_bits:
                        return False
                    v = self._read(32)
                    self.cycle += v & 0x3FFFF
                    self.last_events, self.last_kind = (v >> 18) & 0xFF, "multiple"
                    if self.mode == "verbose":
                        print(f"    Multiple2  0x{v:08X}  mask=0b{self.last_events:08b} "
                              f"+{v & 0x3FFFF} -> {self.cycle}")
                    self._mark(self.cycle, self.last_events)
                elif sel == 2:                      # Repeat1 16b
                    if self.bitpos + 16 > self.total_bits:
                        return False
                    v = self._read(16)
                    self._repeat(v & 0x3FF)
                else:                               # Stop 32b: 110111 x(8) c(18)
                    if self.bitpos + 32 > self.total_bits:
                        return False
                    v = self._read(32)
                    self.cycle += v & 0x3FFFF
                    self.last_kind = None
                    self._run_flush()
                    if self.mode == "faithful":
                        print(f"[aie_runtime] core_trace_decode: STOP @ {self.cycle}")
                    elif self.mode == "verbose":
                        print(f"    Stop  0x{v:08X}  +{v & 0x3FFFF} -> STOP @ {self.cycle}")
        elif (b0 & 0x10) == 0:                      # Repeat0 8b: 1110 rrrr
            v = self._read(8)
            self._repeat(v & 0xF)
        elif (b0 & 0x08) == 0:                      # Start 64b: 11110 O 00 + timer56
            if self.bitpos + 64 > self.total_bits:
                return False
            w0 = self._read(32)
            w1 = self._read(32)
            self.cycle = ((w0 & 0x00FFFFFF) << 32) | w1
            overrun = (w0 >> 26) & 1
            self.last_kind = None
            self._run_flush()
            if self.mode == "faithful":
                print(f"[aie_runtime] core_trace_decode: START timer={self.cycle} overrun={overrun}")
            elif self.mode == "verbose":
                print(f"    Start  timer={self.cycle}  overrun={overrun}")
        else:                                       # Filler (0xFE) / Sync (0xFF)
            v = self._read(8)
            if v == 0xFF:
                self.cycle += SYNC_CYCLES
                self.last_kind = "sync"
                if self.mode == "verbose":
                    print(f"    Sync  +0x{SYNC_CYCLES:X} -> {self.cycle}")
            elif self.mode == "verbose":
                print("    Filler  (alignment pad)")
        return True

    def _decode_group(self, payload, names, stream):
        """Decode ONE pkt-id stream's concatenated payload words with its own
        slot-name table. Resets the per-stream frame state (cycle/bitpos/run/
        last_*), runs the frame loop, and appends to self.timeline. Mirrors
        __core_trace_decode_group in aie_runtime.c."""
        self.names = names
        self.stream = stream
        self.payload = payload
        self.total_bits = len(payload) * 32
        self.bitpos = 0
        self.cycle = 0
        self.run = None
        self.last_kind = None
        self.last_events = 0
        while self.bitpos + 8 <= self.total_bits:
            if not self.step_frame():
                break
        self._run_flush()

    # -- the whole call: __Runtime_core_trace_decode(buf, nwords) -----------
    def decode(self, words, nwords=None):
        """Faithful port of __Runtime_core_trace_decode(buf, nwords).

        Collects the 8-word packets up to the first all-zero (untouched-tail)
        packet, then DEMUXES them by header packet id (low 5 bits): each id's 7
        payload words are concatenated into its own frame stream and decoded with
        the matching slot-name table (pkt 2 = mem-module DMA, else core). Ids are
        decoded ascending so core (1) precedes mem (2); a lone id (incl. legacy
        id 0) yields a single group == the old behavior. Returns the combined
        [(cycle, slot_name, category), ...] timeline (category 'mem' for the
        mem stream, else 'event' for PORT_* / 'core')."""
        words = [w & 0xFFFFFFFF for w in words]
        if nwords is None:
            nwords = len(words)

        if self.mode == "faithful":
            print(f"[aie_runtime] core_trace_decode: buf=<sim> nwords={nwords}")

        # Outer transport: gather each valid packet (header + payload) up to the
        # untouched all-zero tail, grouped by header pkt id (header & 0x1F).
        whole = nwords - (nwords % PACKET_WORDS)
        groups = {}          # pkt_id -> concatenated payload words
        for p in range(0, whole, PACKET_WORDS):
            if p + PACKET_WORDS > len(words):
                break
            pkt = words[p:p + PACKET_WORDS]
            if all(w == 0 for w in pkt):
                if self.mode == "verbose":
                    print(f"packet {p // PACKET_WORDS}: all-zero -> untouched tail: STOP")
                break
            pid = pkt[0] & 0x1F
            if self.mode == "verbose":
                print(f"packet {p // PACKET_WORDS}: hdr=0x{pkt[0]:08X} "
                      f"id={pid} type={(pkt[0] >> 12) & 0x7}")
            groups.setdefault(pid, []).extend(pkt[1:])   # drop word[0] header

        # Demux: decode each pkt id ascending, each with its own name table.
        self.timeline = []
        for pid in sorted(groups):
            names = table_for_id(pid)
            stream = "mem" if pid == 2 else "core"
            self._decode_group(groups[pid], names, stream)
        return self.timeline

    # Back-compat alias for callers that used the old name.
    decode_stream = decode


def _parse_word(tok):
    tok = tok.strip().rstrip(",")
    if not tok:
        return None
    return int(tok, 0) & 0xFFFFFFFF  # int(...,0) honours 0x / 0b / decimal


def _demo_words():
    """One 8-word packet: header + Start@1000, then Single0 ACTIVE(+10),
    Single1 STREAM_STALL(+100), Filler, Multiple0 ACTIVE|LOCK_STALL(+5), Sync,
    then filler padding."""
    #  hdr        Start w0    Start w1    S0 S1..     M0 Sync..    filler x3
    return [0x00000001, 0xF0000000, 0x000003E8, 0x0A8864FE,
            0xC035FFFE, 0xFEFEFEFE, 0xFEFEFEFE, 0xFEFEFEFE]


def main(argv):
    args = argv[1:]

    verbose = False
    if args and args[0] in ("-v", "--verbose"):
        verbose = True
        args = args[1:]

    if args == ["-"]:
        toks = sys.stdin.read().split()
        words = [w for w in (_parse_word(t) for t in toks) if w is not None]
    elif args:
        words = [w for w in (_parse_word(t) for t in args) if w is not None]
    else:
        words = _demo_words()
        print("(no words given; running built-in demo: one 8-word packet)\n")

    dec = CoreTraceDecoder(mode="verbose" if verbose else "faithful")
    timeline = dec.decode(words)

    print("\n=== decoded timeline (cycle  slot  category) ===")
    if not timeline:
        print("(no events)")
    for cyc, name, category in timeline:
        print(f"{cyc}  {name}  [{category}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
