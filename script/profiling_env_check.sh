# Sourced by aiehlc.sh. Warns when the caller exported AIEHLC_PROFILING but did
# not pass --profiling: aiehlc.sh resets PROFILING=0 on every run and passes
# that value to hostcompile.sh, so the env var alone never reaches the build.
# Behaviour is unchanged (the flag stays the only switch); this only makes the
# silent drop visible.
#   $1 = PROFILING value after argument parsing (0|1)
#   reads AIEHLC_PROFILING from the environment
# Prints the warning on stderr; always returns 0.
aiehlc_profiling_env_check() {
    local flag="${1:-0}"
    local env_val="${AIEHLC_PROFILING:-}"
    if [ "$flag" = "1" ]; then
        return 0
    fi
    case "$env_val" in
        ""|0) return 0 ;;
    esac
    echo "[aiehlc] WARNING: AIEHLC_PROFILING=${env_val} is set in the environment but --profiling was not given;" \
         "the host is built WITHOUT profiling. Pass --profiling to enable it." >&2
    return 0
}
