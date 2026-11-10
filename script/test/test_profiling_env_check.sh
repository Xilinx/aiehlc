#!/usr/bin/env bash
# Unit test for script/profiling_env_check.sh. Run: bash script/test/test_profiling_env_check.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../profiling_env_check.sh"
fail=0; n=0
check() {  # name flag env expect_warn(0|1)
    local name="$1" flag="$2" env="$3" want="$4" err
    n=$((n+1))
    if [ "$env" = "<unset>" ]; then
        err="$(unset AIEHLC_PROFILING; aiehlc_profiling_env_check "$flag" 2>&1 >/dev/null)"
    else
        err="$(AIEHLC_PROFILING="$env" aiehlc_profiling_env_check "$flag" 2>&1 >/dev/null)"
    fi
    local rc=$?
    local got=0; [ -n "$err" ] && got=1
    if [ "$got" != "$want" ] || [ "$rc" != 0 ]; then
        echo "FAIL $name: warn=$got want=$want rc=$rc"; fail=$((fail+1))
    else
        echo "ok   $name"
    fi
}
check "env unset, no flag"      0 "<unset>" 0
check "env empty, no flag"      0 ""        0
check "env 0, no flag"          0 0         0
check "env 1, no flag -> warn"  0 1         1
check "env yes, no flag -> warn" 0 yes      1
check "env 1, --profiling"      1 1         0
check "env unset, --profiling"  1 "<unset>" 0
# the warning must name the flag so the user knows the fix
msg="$(AIEHLC_PROFILING=1 aiehlc_profiling_env_check 0 2>&1)"
n=$((n+1)); case "$msg" in *--profiling*) echo "ok   message names --profiling";; *) echo "FAIL message"; fail=$((fail+1));; esac
# aiehlc.sh must source the helper after the arg loop (so PROFILING is final)
n=$((n+1))
A="${HERE}/../aiehlc.sh"
src=$(grep -n 'aiehlc_profiling_env_check "${PROFILING}"' "$A" | cut -d: -f1)
reset=$(grep -n '^PROFILING=0' "$A" | cut -d: -f1)
flag=$(grep -n 'PROFILING=1$' "$A" | head -1 | cut -d: -f1)
if [ -n "$src" ] && [ "$src" -gt "$reset" ] && [ "$src" -gt "$flag" ]; then echo "ok   wired after arg parsing"; else echo "FAIL wiring"; fail=$((fail+1)); fi
echo "$((n-fail))/$n passed"
[ "$fail" -eq 0 ]
