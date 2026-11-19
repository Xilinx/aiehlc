#!/usr/bin/env bash
# Self-test for the host entry-point detection in script/hostcompile.sh
# (user-main vs mlir-main vs library). Uses the regex as written in
# hostcompile.sh, so the test follows the script.
# Usage: bash script/test/test_hostcompile_main_detect.sh
# Prints: HC_MAIN_DETECT cases=<n> fail=<m> PASS|FAIL
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HC="${SCRIPT_DIR}/../hostcompile.sh"
RE_LINE="$(grep -m1 "^HOST_USER_MAIN_RE=" "${HC}")" || {
    echo "HC_MAIN_DETECT cases=0 fail=1 FAIL (no HOST_USER_MAIN_RE in hostcompile.sh)"; exit 1; }
eval "${RE_LINE}"
kind() {  # same order as hostcompile.sh
    if grep -Eq "${HOST_USER_MAIN_RE}" "$1"; then echo user-main
    elif grep -q 'void main()' "$1"; then echo mlir-main
    else echo library; fi
}
TMP="$(mktemp -d /tmp/hc_main_detect.XXXXXX)"
cleanup() { rm -rf -- "${TMP}"; }
trap cleanup EXIT
n=0; fail=0
check() {  # <expected> <name> <source text>
    n=$((n+1)); printf '%s\n' "$3" > "${TMP}/$2.cc"
    got="$(kind "${TMP}/$2.cc")"
    if [ "${got}" = "$1" ]; then echo "ok   $2 -> ${got}"
    else echo "FAIL $2 -> ${got} (want $1)"; fail=$((fail+1)); fi
}
check user-main  empty_args      'int main() { return 0; }'
check user-main  void_args       'int main(void) { return 0; }'
check user-main  argc_argv       'int main(int argc, char **argv) { return 0; }'
check user-main  unnamed_args    'int main(int, char**) { return 0; }'
check user-main  space_paren     'int main (int argc, char *argv[])'
check user-main  indented        '  int  main(){'
check mlir-main  mlir_void_main  'void main() {
  return;
}'
check library    no_main         'int test_matmul(int n) { return n; }'
check library    main_helper     'int main_helper(int x) { return x; }'
check library    comment_only    '// call int main() from your app
int test_matmul(int n) { return n; }'
echo "HC_MAIN_DETECT cases=${n} fail=${fail} $([ ${fail} -eq 0 ] && echo PASS || echo FAIL)"
[ ${fail} -eq 0 ]
