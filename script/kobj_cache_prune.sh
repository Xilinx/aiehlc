#!/bin/bash
# kobj_cache_prune.sh [cache_dir] [max_mb]
# Prune the F2 kernel-object cache (AIEHLC_KOBJ_CACHE, see hostcompile.sh) to max_mb (default
# AIEHLC_KOBJ_CACHE_MAX_MB or 4096), oldest entries first. Age = mtime of the entry's DONE file,
# which kobj_cache_get touches on every hit (LRU). Also drops .tmp.* store dirs older than 60 min
# (a store that died mid-copy). Safe while builds run: an entry is removed by renaming it to
# .del.* first, so a concurrent get sees either the whole entry or none.
set -u
C="${1:-${AIEHLC_KOBJ_CACHE:-}}"
MAX_MB="${2:-${AIEHLC_KOBJ_CACHE_MAX_MB:-4096}}"
[ -n "$C" ] && [ -d "$C" ] || { echo "kobj_cache_prune: no cache dir" >&2; exit 0; }
find "$C" -maxdepth 1 -name '.tmp.*' -type d -mmin +60 -exec rm -rf {} + 2>/dev/null
find "$C" -maxdepth 1 -name '.del.*' -type d -exec rm -rf {} + 2>/dev/null
total_kb=$(du -sk "$C" | cut -f1)
max_kb=$(( MAX_MB * 1024 ))
n=0
if [ "$total_kb" -gt "$max_kb" ]; then
    # oldest DONE first
    while read -r _ done_f; do
        [ "$total_kb" -le "$max_kb" ] && break
        e="$(dirname "$done_f")"
        kb=$(du -sk "$e" | cut -f1)
        d="$C/.del.$(basename "$e").$$"
        mv -T "$e" "$d" 2>/dev/null && rm -rf "$d" && { total_kb=$(( total_kb - kb )); n=$(( n + 1 )); }
    done < <(find "$C" -mindepth 2 -maxdepth 2 -name DONE -printf '%T@ %p\n' | sort -n)
fi
echo "[kobj_cache_prune] $C: removed $n entries, now $(( total_kb / 1024 )) MB (max ${MAX_MB} MB)"
