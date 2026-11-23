#!/bin/bash
# x86 unit test of the streaming netboot supervisor (no board):
#   1. identity: real app ELFs placed by the old loader (elfload.c at ed72954) and
#      streamed by the new one into a malloc'd 2 GB stand-in for DDR_LOW0, whole
#      block memcmp'd (server mode with the staging copy, and gap-cache mode);
#   2. server.c + launch.c on the host (lwIP stubbed), fed the exact bytes
#      netload.py sends: v1 and v2 LOAD_RUN, a NETLOAD_CORRUPT_AT upload (must be
#      rejected with sha256 mismatch / NEED_RELOAD), an ELF larger than staging
#      (placed while streaming; RERUN must answer NEED_RELOAD).
#
#   bash src/tool/netboot/test/streamload_x86.sh [ELF...]
#   -> STREAMLOAD X86 PASS identical=K/K mismatched=0 corrupt=REJECT rerun_big=NEED_RELOAD proto=v1,v2 machine=x86_64
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB="$(cd "${HERE}/.." && pwd)"
REPO="$(cd "${NB}/../../.." && pwd)"
OLD_REV="${STREAMLOAD_OLD_REV:-ed72954}"
CC="${HOST_CC:-gcc}"
PY="${PYTHON:-python3}"

DEFAULT_ELFS=(
    /scratch/staff/huaj/tmp/sprint1113_unet/p3/conv/main.elf
    /scratch/staff/huaj/aiengineers/steve/work/p3_board/elf/main.elf
    /scratch/staff/huaj/tmp/sprint1113_unet/p2/main.elf
    /scratch/staff/huaj/tmp/sprint1113_unet/p1/main.elf
    /scratch/staff/huaj/tmp/sprint1113_unet/aiehlchj/aout/main.elf
    /scratch/staff/huaj/aiengineers/jane/tasks/2026-11-13_rt-device-lost/board/ctrl_lost/health/main.elf
)
ELFS=()
for e in "${@:-${DEFAULT_ELFS[@]}}"; do
    [ -f "$e" ] && ELFS+=("$e") || echo "[streamload] skip missing $e"
done
fail() { echo "STREAMLOAD X86 FAIL $* machine=$(uname -m)"; exit 1; }
[ "${#ELFS[@]}" -ge 3 ] || fail "need at least 3 real ELFs, have ${#ELFS[@]}"

W="$(mktemp -d "${TMPDIR:-/tmp}/streamload_x86.XXXXXX")"
trap 'rm -rf "${W}"' EXIT
echo "[streamload] work dir ${W}, old loader from ${OLD_REV}"

# ---------------------------------------------------------------- build
mkdir -p "${W}/old"
# Old loader: take it from git when this user may read the repo; otherwise (e.g. the
# checker runs as another uid -> "dubious ownership") use the vendored copy of ed72954,
# whose content is pinned by SHA256SUMS so it cannot silently drift.
REF="${HERE}/ref_ed72954"
OLD_SRC=git
for f in elfload.c netboot.h; do
    if ! git -c safe.directory='*' -C "${REPO}" show "${OLD_REV}:src/tool/netboot/supervisor/${f}" >"${W}/old/${f}" 2>/dev/null; then
        OLD_SRC=vendored
        cp "${REF}/${f}" "${W}/old/${f}" || fail "no old ${f} (git and vendored both failed)"
    fi
done
( cd "${W}/old" && sha256sum -c --quiet "${REF}/SHA256SUMS" ) || fail "old loader does not match ed72954 SHA256SUMS"
echo "[streamload] old loader source: ${OLD_SRC}"
# the old loader writes physical addresses directly: point them into the 2 GB block
sed -i -e 's/(uint8_t \*)(uintptr_t)ph\[i\]\.p_paddr/(nb_host_mem + ph[i].p_paddr)/' \
    -e 's/(uint8_t \*)(uintptr_t)lo;/(nb_host_mem + lo);/' \
    -e 's/(volatile uint32_t \*)(uintptr_t)a;/(volatile uint32_t *)(nb_host_mem + a);/' \
    -e '1i #include <stdint.h>\nextern uint8_t *nb_host_mem;' "${W}/old/elfload.c"
[ "$(grep -c 'nb_host_mem +' "${W}/old/elfload.c")" = 3 ] || fail "old elfload.c rewrite did not match 3 sites"
grep -q '(uintptr_t)' "${W}/old/elfload.c" && fail "old elfload.c still has a raw address cast"

CF=(-O2 -g -std=gnu11 -Wall -Wextra -Wno-unused-parameter -fno-delete-null-pointer-checks -DNB_HOST_TEST)
"${CC}" "${CF[@]}" -I"${W}/old" -I"${NB}/test/host" -I"${NB}" -Delf_place=old_elf_place \
    -Delf_patch_exit=old_elf_patch_exit -c "${W}/old/elfload.c" -o "${W}/old_elfload.o" || fail "build old elfload"
"${CC}" "${CF[@]}" -Werror -I"${NB}/test/host" -I"${NB}/supervisor" -I"${NB}" \
    "${NB}/test/streamload_test.c" "${NB}/supervisor/elfload.c" "${NB}/supervisor/sha256.c" \
    "${NB}/supervisor/server.c" "${NB}/supervisor/launch.c" "${W}/old_elfload.o" -o "${W}/streamload_test" ||
    fail "build streamload_test"

# ---------------------------------------------------------------- 1. identity
"${W}/streamload_test" ident "${ELFS[@]}" | tee "${W}/ident.log"
IDENT="$(grep '^IDENT RESULT' "${W}/ident.log")" || fail "identity run crashed"
K="$(sed -n 's/.*identical=\([0-9]*\)\/.*/\1/p' <<<"${IDENT}")"
N="${#ELFS[@]}"
MIS="$(sed -n 's/.*mismatched=\([0-9]*\).*/\1/p' <<<"${IDENT}")"

# ---------------------------------------------------------------- 2. server
BASE="${ELFS[0]}"
BIG="${W}/big.elf"
"${PY}" "${HERE}/streamload_capture.py" mkbig "${BASE}" "${BIG}" 112 0x50000000 || fail "mkbig"
cap() { "${PY}" "${HERE}/streamload_capture.py" capture "$@" || fail "capture $1"; }
SZ="$(stat -c %s "${BASE}")"
cap "${W}/v1.req" load "${BASE}" --proto 1
cap "${W}/v2.req" load "${BASE}"
NETLOAD_CORRUPT_AT=$((SZ / 2)) cap "${W}/corrupt.req" load "${BASE}"
cap "${W}/big.req" load "${BIG}" --reply-timeout 300
cap "${W}/rerun2.req" rerun
cap "${W}/rerun1.req" rerun --proto 1
cap "${W}/status.req" status
cmp -s <(tail -c +65 "${W}/v2.req") "${BASE}" || fail "v2 capture payload != ELF"
cmp -s <(tail -c +65 "${W}/corrupt.req") "${BASE}" && fail "NETLOAD_CORRUPT_AT did not change the payload"
cmp -s <(head -c 64 "${W}/corrupt.req") <(head -c 64 "${W}/v2.req") || fail "NETLOAD_CORRUPT_AT changed the header/sha"

"${W}/streamload_test" serve elf="${BASE}" bigelf="${BIG}" v1="${W}/v1.req" v2="${W}/v2.req" \
    corrupt="${W}/corrupt.req" big="${W}/big.req" rerun2="${W}/rerun2.req" rerun1="${W}/rerun1.req" \
    status="${W}/status.req" | tee "${W}/serve.log"
SERVE="$(grep '^SERVE RESULT' "${W}/serve.log")" || fail "server run crashed"
RUNLINES="$(grep -c '^\[netboot\] run .* stream=1 elf_size=' "${W}/serve.log")"
echo "[streamload] ${RUNLINES} '[netboot] run ... stream=1 elf_size=' lines"

PROTO=""
grep -q 'v1=ok' <<<"${SERVE}" && PROTO="v1"
grep -q 'v2=ok' <<<"${SERVE}" && PROTO="${PROTO:+${PROTO},}v2"
CORRUPT="$(sed -n 's/.*corrupt=\([A-Z_]*\).*/\1/p' <<<"${SERVE}")"
RERUN="$(sed -n 's/.*rerun_big=\([A-Z_]*\).*/\1/p' <<<"${SERVE}")"
SFAILS="$(sed -n 's/.*fails=\([0-9]*\).*/\1/p' <<<"${SERVE}")"

SUMMARY="identical=${K}/${N} mismatched=${MIS} corrupt=${CORRUPT} rerun_big=${RERUN} proto=${PROTO} machine=$(uname -m)"
if [ "${K}" = "${N}" ] && [ "${MIS}" = 0 ] && [ "${CORRUPT}" = REJECT ] && [ "${RERUN}" = NEED_RELOAD ] &&
    [ "${PROTO}" = "v1,v2" ] && [ "${SFAILS}" = 0 ] && [ "${RUNLINES}" -ge 3 ]; then
    echo "STREAMLOAD X86 PASS ${SUMMARY}"
    exit 0
fi
echo "STREAMLOAD X86 FAIL ${SUMMARY} serve_fails=${SFAILS} run_lines=${RUNLINES}"
exit 1
