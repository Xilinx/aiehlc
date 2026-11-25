#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""Make an aiehlc build persistent: load + launch the kernel once, then every further
launch only re-arms the shim BDs (aiehlc.sh --persistent; productized from
proto/schedule-api-prototype/spike/m0b/persist_transform.py).

  kernel.cc : the core keeps its one-time setup (KCO DMA/locks, window_init) and loops
              the kernel call forever. Core-tile channels run circular ping-pong BD
              chains and the window structs stay alive, so nothing is re-armed on the
              core. A megakernel re-reads its mode descriptor on every call.
  host.cc   : host_first = host_canonicalized without the wait for the cores (they never
              finish); host_tile = shim BD programming + startio + output waits only
              (no ELF load, no launch, no BD-template prepare). Every __aie_launch calls
              host_first on its first use of a partition, host_tile afterwards.

Needs #pragma KERNELCONFIGOFFLOAD: without it host_canonicalized also programs the
core-tile BDs/locks, which must not be redone under a running core (refused below).

usage: persist_transform.py <worklocal_dir>   (rewrites host.cc / kernel.cc, keeps .orig)
"""
import re
import shutil
import sys
from pathlib import Path


def func_span(text, name):
    m = re.search(r"^void %s\(XAie_DevInst\* v1(?:, void\* v\d+)*\) \{\n" % name, text, re.M)
    if not m:
        sys.exit(f"persist_transform: {name}() not found")
    depth, i = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return m.start(), i


def transform_host(src):
    s, e = func_span(src, "host_canonicalized")
    body = src[s:e]
    core = {m.group(1) for m in re.finditer(r"XAie_LocType (v\d+) = XAie_TileLoc\(\d+, (\d+)\);", body)
            if int(m.group(2)) >= 3}                       # core rows (VEK280: 3..10)
    for m in re.finditer(r"__Runtime_(dma_bd_config\w*|lock\w*|dma_channel\w*)\(v1, (v\d+)", body):
        if m.group(2) in core:
            sys.exit(f"persist_transform: host programs core tile {m.group(2)} ({m.group(1)}) -- "
                     "add #pragma KERNELCONFIGOFFLOAD")
    launch = re.search(r"^\s*event (v\d+) = __Runtime_launch_kernel_group\w*\(v1, (?:&__ctrl_fabric_\d+, )?v\d+.*\);\n",
                       body, re.M)
    if not launch:
        sys.exit("persist_transform: no launch_kernel_group")
    ev = launch.group(1)
    wait_core = re.compile(r"^\s*__Runtime_wait\(v1, %s\);\n" % ev, re.M)
    if not wait_core.search(body):
        sys.exit(f"persist_transform: no wait on the core event {ev}")

    first = wait_core.sub("", body).replace("void host_canonicalized(", "void host_first(", 1)

    tile = wait_core.sub("", body)
    tile = re.sub(r"^\s*/\* Load Kernel Group.*\*/\n", "", tile, flags=re.M)
    tile = re.sub(r"^\s*kernel_group v\d+ = __Runtime_load_kernel_group_\w+\(.*\);\n", "", tile, flags=re.M)
    tile = re.sub(r"^\s*/\* Launch Kernel Group \*/\n", "", tile, flags=re.M)
    tile = re.sub(r"^\s*__Runtime_bd_template_prepare_begin\(\);\n.*?__Runtime_bd_template_prepare_end\(\);\n",
                  "", tile, flags=re.M | re.S)
    tile = re.sub(r"^\s*event %s = __Runtime_launch_kernel_group\w*\(v1, .*\);\n" % ev, "", tile, flags=re.M)
    # control-packet plane (#pragma control_plan_op_control_packet): the fabric is planned once, in
    # host_first; it only carries the ELF load + core launch, which host_tile no longer does
    tile = re.sub(r"^\s*/\* control-packet group-write fabric setup \*/\n", "", tile, flags=re.M)
    tile = re.sub(r"^\s*static __Runtime_CtrlRow\w+ __ctrl_fabric_\d+\w*(?:\[\])?(?: = .*)?;\n", "", tile, flags=re.M)
    tile = re.sub(r"^\s*__Runtime_ctrl_plan_\w+\(&__ctrl_fabric_\d+, .*\);\n", "", tile, flags=re.M)
    tile = tile.replace("void host_canonicalized(", "void host_tile(", 1)
    for bad in ("load_kernel_group", "launch_kernel_group", "bd_template_prepare", "__ctrl_fabric_"):
        if bad in tile:
            sys.exit(f"persist_transform: {bad} left in host_tile")

    split = split_tile(tile)
    out = (src[:e] + "\nstatic int s_persist_started = 0; /* --persistent: the array is configured */\n" +
           "/* several megakernels (aiednn_megakernel_select): the core ELF the next launch needs; switching it\n"
           " * makes that launch load + start it (host_first) instead of re-arming the shim BDs */\n"
           "static unsigned char *s_persist_elf = 0;\n"
           "static inline void __aie_persist_select_elf(unsigned char *elf) {\n"
           "    if (elf && elf != s_persist_elf) { s_persist_elf = elf; s_persist_started = 0;\n"
           "        __Runtime_memtile_weights_invalidate(); } /* host_first reconfigures the memtiles */\n"
           "}\n" + first +
           "\n" + tile + "\n" + split + src[e:])
    # every launch wrapper: first call of the process configures + launches, the rest re-arm
    call = re.compile(r"^(\s*)host_canonicalized\(dev, ([^;]*)\);\n", re.M)
    n = len(call.findall(out))
    if not n:
        sys.exit("persist_transform: no host_canonicalized call in a launch wrapper")
    out = call.sub(lambda m: (f"{m.group(1)}{{ /* --persistent */\n"
                              f"{m.group(1)}  if (!s_persist_started) {{ host_first(dev, {m.group(2)}); "
                              f"s_persist_started = 1; }}\n"
                              f"{m.group(1)}  else host_tile(dev, {m.group(2)}); }}\n"), out)
    # the active ELF: the selected megakernel, else the one the wrapper names
    out = re.sub(r"__Runtime_set_kernel_elf\((_binary_kernel_\w+_start)\);",
                 r"__Runtime_set_kernel_elf(s_persist_elf ? s_persist_elf : \1);", out)
    return ntile_wrappers(async_wrappers(out), split) if split else out


def _buffer_roots(body, args=("v2", "v3", "v4")):
    """SSA name -> the launch argument (v2 A, v3 B, v4 C) its pointer derives from."""
    root = {a: a for a in args}
    for m in re.finditer(r"void\* (v\d+) = __runtime_buffer_(?:offset|arg)\((v\d+)", body):
        if m.group(2) in root:
            root[m.group(1)] = root[m.group(2)]
    return root


def ntile_start(start):
    """host_tile_n_start(..., n, a_step, b_step, b_period, c_step): host_tile_start streaming n GEMM
    tiles in one launch. A (v2) BDs fold their k-window iteration into d2 and step a_step per tile;
    B (v3): b_step 0 re-emits the same block n times from the memtile, else tile t reads the block
    at (t % b_period) * b_step (a K chain: m outer, k inner); C (v4) out-of-order S2MM BDs step
    c_step per tile;
    every start-queue repeat scales with n. The persistent core needs no change: it consumes one
    tile's windows per kernel call and re-reads the descriptor from each re-streamed B. Returns
    "" when any BD or start is not one of those shapes (no n-tile variant)."""
    root = _buffer_roots(start)
    kind, bad = {}, []

    def bd(m):
        d, fn, args, buf = m.group(1), m.group(2), m.group(3), m.group(4)
        r, ooo = root.get(buf), fn.endswith("_ooo")
        if ooo and r == "v2":
            kind[d] = "A"
            return f"XAie_DmaDesc {d} = __Runtime_ntile_bd_a({args}, n, a_step);"
        if not ooo and r == "v4":
            kind[d] = "C"
            return f"XAie_DmaDesc {d} = __Runtime_ntile_bd_c({args}, n, c_step);"
        bad.append(f"BD {d} on {buf} ({r})")
        return m.group(0)

    out = re.sub(r"XAie_DmaDesc (v\d+) = (__Runtime_dma_bd_config_multidim(?:_ooo)?)\((v1, v\d+, (v\d+), [^;]*)\);",
                 bd, start)
    ios = {m.group(1): kind.get(m.group(2))
           for m in re.finditer(r"io (v\d+) = __Runtime_dma_createio_4\(v\d+, (v\d+),", out)}

    def sio(m):
        k = ios.get(m.group(2))
        if k == "A":
            return f"{m.group(1)} = __Runtime_startio(v1, {m.group(2)}, {m.group(3)}, n);"
        if k == "C":
            return f"{m.group(1)} = __Runtime_startio(v1, {m.group(2)}, {m.group(3)}, {m.group(4)} * n);"
        bad.append(f"startio {m.group(2)}")
        return m.group(0)

    out = re.sub(r"(s_tile_v\d+) = __Runtime_startio\(v1, (v\d+), ([^,]+), ([^)]+)\);", sio, out)
    for m in re.finditer(r"__Runtime_memtile_weights_start\(v1, v\d+, (v\d+),", out):
        if root.get(m.group(1)) != "v3":
            bad.append(f"memtile weights on {m.group(1)}")
    if bad or "A" not in kind.values() or "C" not in kind.values():
        print("persist_transform: no n-tile variant (" + (", ".join(bad) or "no A / C BDs") + ")")
        return ""
    out = re.sub(r"__Runtime_memtile_weights_start\(([^;]*)\);", r"__Runtime_memtile_weights_start_n(\1, n, b_step, b_period);", out)
    return re.sub(r"^void host_tile_start\((XAie_DevInst\* v1(?:, void\* v\d+)*)\) \{\n",
                  r"void host_tile_n_start(\1, int32_t n, int64_t a_step, int64_t b_step, int32_t b_period, int64_t c_step) {\n", out, count=1, flags=re.M)


def ntile_wrappers(out, split):
    """__aie_launch_n_start(..., n, a_step, c_step): __aie_launch_start for n tiles (sizes cover all
    n), finished by the ordinary __aie_launch_finish. Defines AIEHLC_NTILE_LAUNCH for the user source."""
    s = split.index("void host_tile_start(")
    e = split.index("void host_tile_finish(")
    n = ntile_start(split[s:e])
    if not n:
        return out
    anchor = "inline void __aie_launch_finish("
    i = out.find(anchor)
    if i < 0:
        return out
    j = out.index("\n}\n", out.index("inline void __aie_launch_start(")) + 3
    start_wrap = out[out.index("inline void __aie_launch_start("):j]
    nwrap = start_wrap.replace("inline void __aie_launch_start(", "inline void __aie_launch_n_start(", 1)
    nwrap = nwrap.replace("size_t _s2) {", "size_t _s2, int32_t n, int64_t a_step, int64_t b_step, int32_t b_period, "
                                           "int64_t c_step) {", 1)
    nwrap = nwrap.replace("host_tile_start(dev, _t0, _t1, _t2);",
                          "host_tile_n_start(dev, _t0, _t1, _t2, n, a_step, b_step, b_period, c_step);")
    nwrap = nwrap.replace("if (!s_persist_started) { host_first(dev, _t0, _t1, _t2); s_persist_started = 1; s_async_pending = 0; }\n    else ",
                          "if (!s_persist_started) return; /* host_first configures the array: run one tile first */\n    ")
    # host_tile_n_start goes right before the wrapper (after host_tile_finish, which it shares)
    k = out.index(split) + len(split)
    out = out[:k] + n + "\n" + out[k:]
    j = out.index("\n}\n", out.index("inline void __aie_launch_start(")) + 3
    return out[:j] + "#define AIEHLC_NTILE_LAUNCH 1 /* --persistent: n GEMM tiles per launch */\n" + nwrap + out[j:]


def split_tile(tile):
    """host_tile_start (shim BDs + startio) and host_tile_finish (the output waits): the host may
    work between them while the array runs. The startio events become statics. "" if host_tile
    has no wait to split at."""
    w = re.search(r"^\s*__Runtime_wait\(v1, v\d+\);\n", tile, re.M)
    if not w:
        return ""
    head = tile[:tile.index("{\n") + 2]
    start, finish = tile[len(head):w.start()], tile[w.start():]
    evs = re.findall(r"ioevent (v\d+) = __Runtime_startio\(", start)
    for v in evs:
        start = start.replace(f"ioevent {v} = ", f"s_tile_{v} = ")
        finish = re.sub(r"\b%s\b" % v, f"s_tile_{v}", finish)
    decls = "".join(f"static ioevent s_tile_{v};\n" for v in evs)
    # the debug snapshot after the waits reads `dev` (a local the start half declares)
    dev = "  XAie_DevInst* dev = v1;\n  (void)dev;\n" if re.search(r"\bdev\b", finish) else ""
    return (decls + head.replace("void host_tile(", "void host_tile_start(", 1) + start + "}\n\n" +
            head.replace("void host_tile(", "void host_tile_finish(", 1) + dev + finish + "\n")


def async_wrappers(out):
    """__aie_launch_start / __aie_launch_finish next to the 3-buffer aieMesh __aie_launch: the same
    prelude (cache syncs) and postlude (drain, sync_for_cpu) around host_tile_start / _finish.
    Defines AIEHLC_ASYNC_LAUNCH for the code below (the user source; the frontend parsed it with the
    macro undefined, so such a branch is text it never lowered)."""
    sig = ("inline void __aie_launch(const char* kernel, aieMesh mesh, void* _t0, size_t _s0, void* _t1, "
           "size_t _s1, void* _t2, size_t _s2, ...) {\n")
    i = out.find(sig)
    if i < 0:
        return out
    j = out.index("\n}\n", i) + 3
    body = out[i + len(sig):j - 2]
    blk = re.search(r"^\s*\{ /\* --persistent \*/\n.*?else host_tile\(dev, [^;]*\); \}\n", body, re.M | re.S)
    if not blk:
        return out
    pre, post = body[:blk.start()], body[blk.end():]
    params = "const char* kernel, aieMesh mesh, void* _t0, size_t _s0, void* _t1, size_t _s1, void* _t2, size_t _s2"
    wrappers = ("#define AIEHLC_ASYNC_LAUNCH 1 /* --persistent: host work between a launch's start and finish */\n"
                "static int s_async_pending = 0;\n"
                f"inline void __aie_launch_start({params}) {{\n{pre}"
                "    if (!s_persist_started) { host_first(dev, _t0, _t1, _t2); s_persist_started = 1; s_async_pending = 0; }\n"
                "    else { host_tile_start(dev, _t0, _t1, _t2); s_async_pending = 1; }\n}\n"
                f"inline void __aie_launch_finish({params}) {{\n"
                "    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);\n"
                "    if (s_async_pending) { host_tile_finish(dev, _t0, _t1, _t2); s_async_pending = 0; }\n"
                f"{post}}}\n")
    return out[:j] + wrappers + out[j:]


def transform_kernel(src):
    m = re.search(r"^(\s*)// kernel_invoke (\w+)\n(\s*)(\2\(.*\);)\n", src, re.M)
    if not m:
        sys.exit("persist_transform: kernel_invoke not found")
    indent, call = m.group(3), m.group(4)
    loop = f"{m.group(1)}// kernel_invoke {m.group(2)} (persistent)\n{indent}for (;;) {{\n{indent}  {call}\n{indent}}}\n"
    return src[:m.start()] + loop + src[m.end():]


def main():
    wl = Path(sys.argv[1])
    for name, fn in (("host.cc", transform_host), ("kernel.cc", transform_kernel)):
        p = wl / name
        orig = p.with_suffix(p.suffix + ".orig")
        shutil.copy(p, orig)
        p.write_text(fn(orig.read_text()))
        print(f"persist_transform: rewrote {p}")


if __name__ == "__main__":
    main()
