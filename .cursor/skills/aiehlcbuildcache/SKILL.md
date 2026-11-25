---
name: aiehlcbuildcache
description: aiehlc.sh rebuild caches (default ON) -- alib skip (`alib up to date` / `alib needs a rebuild`, --rebuild-alib), kernel .o cache (`KOBJ cache hit/store`, AIEHLC_KOBJ_CACHE), host object cache (`object cache ... N of M objects reused`, HOSTCOMPILE_OBJ_CACHE). Suspect a stale build / force a rebuild / turn a cache off; why the kernel key lists kernel.cc's #includes; why "the files exist" is never a cache key.
---

# aiehlc.sh rebuild caches

A plain rerun of an unchanged source used to cost ~94 s (simplematmul2, aie2ps): aie-rt
rebuilt from scratch (~23 s, serial `make` + an x86 build nothing links), the kernel
recompiled (xchesscc ~15 s even for a one-line file + xchessmk ~30 s), 8 runtime `.c`
recompiled. Now ~5.5 s, `main.elf` byte-identical (sha256 `f3135a8b054c...`).

| Layer | Where | Key | Log line | Force / off |
|---|---|---|---|---|
| alib (aie-rt lib + headers) | `aiehlc.sh` `aiehlc_alib_ensure` | `aiehlc_libcache_key`: aie-rt `git ls-files -s` + diff + untracked, BSP `*.h`/`*.a`/`lscript.ld`, `gcc -v`, flags, `AIE_GEN`, platform, `declare -f build_hw_lib` | `alib up to date (key K)` / `alib rebuilt` | `--rebuild-alib`, `AIEHLC_ALIB_REBUILD=1` |
| kernel .o + ELF + lst/map/linemap | `hostcompile.sh` `compile_one_kernel` (tiling flow, single + multi-kernel) | `kernel_stamp` + bcf + ctrl-pkt sites/manifest (+ host.cc if sites) + **every file `kernel.cc` includes** (`kobj_kernel_deps`) + worklocal `*.h`, runtime/include headers, `hostcompile.sh`, tool paths, `AIEHLC_KERNEL_DEFINES`/`_LOG` | `KOBJ cache hit/store <fn> (key)` | `HOSTCOMPILE_KERNEL_REBUILD=1`; `AIEHLC_KOBJ_CACHE=off` |
| host.cc / runtime / routing objects | `hostcompile.sh` parallel block `_hc_one` | `g++ --version` + command (build paths masked) + `-E -P` text | `object cache D: N of M objects reused` | `HOSTCOMPILE_OBJ_CACHE=off`; disabled for `-g` |

Dirs: `~/.cache/aiehlc/{kobj,hobj}` (`XDG_CACHE_HOME` honoured); kobj pruned to
`AIEHLC_KOBJ_CACHE_MAX_MB` (4096), hobj to `HOSTCOMPILE_OBJ_CACHE_KEEP` (2000) objects.

## Rules that are easy to break

1. **Presence is not a key.** alib holds ONE build: `--aie-version 2` installs
   `libxaienginea72.a` + the a72 BSP headers, `5` the a78 ones; an edited aie-rt must
   rebuild. "skip if `lib/*.a` exists" links the wrong library silently. Verified matrix:
   5->2 rebuilt, 2 again skipped, 2->5 rebuilt, aie-rt edit rebuilt, revert rebuilt, next skipped.
2. **The kernel key must follow `kernel.cc`'s includes.** The generated `kernel.cc` is a
   wrapper that `#include "matmul.cc"` (the user's `__global__` body). Editing only the body
   leaves `kernel.cc` byte-identical; the pre-2026-11-24 -fast key (kernel.cc + `*.h`) then
   HIT and shipped the old kernel (reproduced: one added line in the body -> every
   old-key input identical). `kobj_kernel_deps` runs host `g++ -std=c++20 -M -MG -nostdinc
   -D__chess__ -D__AIENGINE__ -D__AIE_ARCH__=<20|22>` with kc.sh's `-I` list + `-include
   kernel_log.h` and hashes every listed file (missing ones by name). If it cannot list
   them the cache is skipped for that kernel ("KOBJ cache off ... could not list"), never
   guessed.
3. **Hash the recipe, not just the inputs.** The alib key includes `declare -f
   build_hw_lib`, and the kobj key `hostcompile.sh` + `kc.sh`: editing the build steps
   invalidates old entries (expect one miss after such an edit).
4. **A restore must recreate what the build would have.** kobj entries carry all
   `build/kernel*` artifacts + `xchessmk.log` (aiperf reads `kernel.lst`, the `.TM` check
   greps it, aiediag reads the linemap) and the bcf after kc.sh's stack shrink. The restore
   `mkdir -p build/` first -- a fresh `aout/` has none (the original get failed there and
   silently fell back to compiling).
5. **Locking (alib).** Shared while used, exclusive to rebuild (skill **alibflocklease**).
   Two same-alib runs build in parallel (~5 s each); a v2 run started meanwhile printed
   `waiting for the exclusive alib lock` and rebuilt after both exited.

## Suspect a stale result?

- Compare the key line across runs; rerun with the force knob of that layer; or set all
  three off: `AIEHLC_KOBJ_CACHE=off HOSTCOMPILE_OBJ_CACHE=off source script/aiehlc.sh --rebuild-alib ...`.
- `main.elf` of a cached and an uncached build must be byte-identical (they are for
  simplematmul2, tpl_two_instances, multi_kernel, simpleconv2d).

## Not covered

The raw (non-tiling) single-kernel flow in `aiehlc.sh` still compiles its kernels and
runtime every run (it gets the alib skip only).
