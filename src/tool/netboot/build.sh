#!/bin/bash
# Build the netboot supervisor (doc/design/lwip_elf_bootloader.md).
#
#   src/tool/netboot/build.sh            # -> src/tool/netboot/build/netboot.elf
#   NETBOOT_IP=10.10.70.50 NETBOOT_GW=10.10.70.101 src/tool/netboot/build.sh
#   NETBOOT_DMA_BASE=0xA00000000 NETBOOT_OUT=src/tool/netboot/build/vek385 src/tool/netboot/build.sh  # VEK385
#
# lwIP 2.2 and its Xilinx port are compiled from the Vitis embeddedsw tree, the
# same sources the aiebaremetal example_lwip_echo links (via libxtflwipall.a),
# but against THIS repo's A78 SDT BSP. That BSP has no liblwip, and the XTF
# library is an A72 / Versal-Gen1 build.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
# NETBOOT_OUT: output dir (apppaltest builds the VEK385 variant into build/vek385).
OUT="${NETBOOT_OUT:-${HERE}/build}"
VITIS="${VITIS:-/proj/custbuilds/SWIP/2026.1_0616_1700/installs/lin64/2026.1/Vitis}"
LWIP="${LWIP_SRC:-${VITIS}/data/embeddedsw/ThirdParty/sw_services/lwip220_v1_4/src/lwip-2.2.0}"
PORT="${LWIP}/contrib/ports/xilinx"
BSP="${REPO}/thirdparty/arch/cortexa78_0/bsp"
CROSS="${CROSS:-aarch64-none-elf-}"
CC="${CROSS}gcc"
AR="${CROSS}ar"

NETBOOT_IP="${NETBOOT_IP:-10.10.70.50}"
NETBOOT_MASK="${NETBOOT_MASK:-255.255.255.0}"
NETBOOT_GW="${NETBOOT_GW:-10.10.70.101}"
# VEK385: GEM (LPD) cannot reach DDR_LOW0 there; put its DMA memory in DDR_CH1.
#   NETBOOT_DMA_BASE=0xA00000000 src/tool/netboot/build.sh
# lwIP heap+pools go at NETBOOT_DMA_BASE, the BD ring 1 GB above it. Empty = palmyra layout.
NETBOOT_DMA_BASE="${NETBOOT_DMA_BASE:-}"

die() { echo "[netboot] ERROR: $*" >&2; exit 1; }
command -v "${CC}" >/dev/null || die "${CC} not on PATH (source Vitis settings64.sh)"
[ -d "${PORT}" ] || die "lwIP Xilinx port not found at ${PORT} (set LWIP_SRC)"

ipdef() { # 10.10.70.50 -> 10,10,70,50
    echo "$1" | tr '.' ','
}

mkdir -p "${OUT}/lwipobj" "${OUT}/obj"

CFLAGS=(-O2 -g -mcpu=cortex-a78 -std=gnu11 -ffunction-sections -fdata-sections
    -DSDT -DPROCESSOR_LITTLE_ENDIAN
    -I"${HERE}" -I"${HERE}/lwip" -I"${PORT}/include" -I"${LWIP}/src/include"
    -I"${BSP}/include")
# -mcmodel=large: ADRP reaches only +-4 GB, the DMA memory is ~38 GB above the image.
[ -n "${NETBOOT_DMA_BASE}" ] && CFLAGS+=(-mcmodel=large -DNETBOOT_DMA_BASE="${NETBOOT_DMA_BASE}ULL")

# ---------------------------------------------------------------- liblwip.a
LWIP_SRCS=()
for f in "${LWIP}"/src/core/*.c "${LWIP}"/src/core/ipv4/*.c "${LWIP}"/src/netif/ethernet.c; do
    LWIP_SRCS+=("$f")
done
LWIP_SRCS+=("${PORT}/sys_arch_raw.c" "${PORT}/netif/xpqueue.c" "${PORT}/netif/xadapter.c"
    "${PORT}/netif/xemacpsif.c" "${PORT}/netif/xemacpsif_dma.c" "${PORT}/netif/xemacpsif_hw.c"
    "${PORT}/netif/xemacpsif_physpeed.c" "${HERE}/lwip/xtopology_g.c")

LIB="${OUT}/liblwip_a78.a"
stamp="${OUT}/lwipobj/.cflags"
want="$(printf '%s\n' "${CFLAGS[@]}" "${LWIP}")"
if [ ! -f "${LIB}" ] || [ "$(cat "${stamp}" 2>/dev/null)" != "${want}" ] || [ "${HERE}/lwip/lwipopts.h" -nt "${LIB}" ]; then
    echo "[netboot] building liblwip_a78.a (${#LWIP_SRCS[@]} sources)"
    rm -f "${OUT}"/lwipobj/*.o "${LIB}"
    for s in "${LWIP_SRCS[@]}"; do
        o="${OUT}/lwipobj/$(basename "${s%.c}").o"
        "${CC}" "${CFLAGS[@]}" -c "$s" -o "$o" || die "compile failed: $s"
    done
    "${AR}" rcs "${LIB}" "${OUT}"/lwipobj/*.o
    echo "${want}" >"${stamp}"
fi

[ "${1:-}" = "--lwip-only" ] && { echo "[netboot] ${LIB}"; exit 0; }

# ---------------------------------------------------------------- netboot.elf
# Placement (measured on the board): 0x8_0000_0000 "DDR_LOW1" mirrors the 2 GB
# DDR_LOW0, so it is NOT free memory. The supervisor takes the top of DDR_LOW0:
#   0x7800_0000 +32 MB   image, heap, stacks          (apps must stay below this)
#   0x7A00_0000 + 2 MB   GEM BD ring, alone in its 2 MB MMU block (the port makes
#                        that block non-cacheable with Xil_SetTlbAttributes)
#   0x7A20_0000 .. 2 GB  staging for received ELFs    (netboot.h)
LDS="${OUT}/lscript_netboot.ld"
python3 - "${REPO}/thirdparty/arch/cortexa78_0/lscript.ld" "${LDS}" "${NETBOOT_DMA_BASE}" <<'PYEOF'
import re, sys
dma = int(sys.argv[3], 0) if len(sys.argv) > 3 and sys.argv[3] else 0
s = open(sys.argv[1]).read()
s = re.sub(r"^_HEAP_SIZE = .*$", "_HEAP_SIZE = DEFINED(_HEAP_SIZE) ? _HEAP_SIZE : 0x1000000;", s, flags=re.M)
s = re.sub(r"^_STACK_SIZE = .*$", "_STACK_SIZE = DEFINED(_STACK_SIZE) ? _STACK_SIZE : 0x100000;", s, flags=re.M)
s = s.replace("MEMORY\n{", "MEMORY\n{\n   netboot_ram : ORIGIN = 0x78000000, LENGTH = 0x2000000\n"
              "   netboot_bd_ram : ORIGIN = 0x7A000000, LENGTH = 0x200000", 1)
s = s.replace("> noc_ddr4_C0_DDR_LOW0", "> netboot_ram")
s = s.replace("SECTIONS\n{", "SECTIONS\n{\n.netboot_bd (NOLOAD) : {\n   KEEP (*(.bss.emac_bd_space))\n} > netboot_bd_ram\n", 1)
if dma:
    s = s.replace("ORIGIN = 0x7A000000, LENGTH = 0x200000",
                  "ORIGIN = 0x%X, LENGTH = 0x200000\n   netboot_dma_ram : ORIGIN = 0x%X, LENGTH = 0x40000000"
                  % (dma + 0x40000000, dma), 1)
    s = s.replace("SECTIONS\n{", "SECTIONS\n{\n.netboot_dma (NOLOAD) : ALIGN(64) {\n   *(.bss.netboot_dma)\n} > netboot_dma_ram\n", 1)
open(sys.argv[2], "w").write(s)
PYEOF
grep -q '> noc_ddr4_C0_DDR_LOW0' "${LDS}" && die "linker script still places sections in the app's DDR"
grep -q 'netboot_bd_ram' "${LDS}" || die "could not add the BD-ring region to ${LDS}"
if [ -n "${NETBOOT_DMA_BASE}" ]; then
    grep -q 'netboot_dma_ram' "${LDS}" || die "could not add the DMA region to ${LDS}"
fi

APP_SRCS=("${HERE}"/supervisor/*.c "${HERE}"/supervisor/*.S)
DEFS=(-DNETBOOT_IP="$(ipdef "${NETBOOT_IP}")" -DNETBOOT_MASK="$(ipdef "${NETBOOT_MASK}")"
    -DNETBOOT_GW="$(ipdef "${NETBOOT_GW}")")
# NETBOOT_TEST_LINK_DOWN=1|2: test build that forces the Ethernet link down after
# init, to exercise the PHY-setup retry (1) or the main-loop link poll (2) on a
# board whose link always comes up. Never stage one for normal use.
[ -n "${NETBOOT_TEST_LINK_DOWN:-}" ] && DEFS+=(-DNB_TEST_LINK_DOWN="${NETBOOT_TEST_LINK_DOWN}")
OBJS=()
for s in "${APP_SRCS[@]}"; do
    b="$(basename "$s")"; o="${OUT}/obj/${b%.*}.o"
    # -fno-delete-null-pointer-checks: the app's first PT_LOAD is at address 0.
    "${CC}" "${CFLAGS[@]}" -Wall -Wextra -Wno-unused-parameter -fno-delete-null-pointer-checks \
        "${DEFS[@]}" -c "$s" -o "$o" || die "compile failed: $s"
    OBJS+=("$o")
done

"${CC}" -mcpu=cortex-a78 -o "${OUT}/netboot.elf" "${OBJS[@]}" "${LIB}" \
    --specs=nosys.specs -Wl,--gc-sections -Wl,--defsym,end=__bss_end__ \
    -Wl,-T -Wl,"${LDS}" -L"${BSP}/lib" -Wl,-Map,"${OUT}/netboot.map" \
    -Wl,--start-group,-lxil,-lxilstandalone,-lxiltimer,-lxilpm_ng,-lgcc,-lc,--end-group ||
    die "link failed"

# Refuse an image that strays outside [0x7800_0000, 0x7A20_0000): below is the
# app's, above is staging.
"${CROSS}readelf" -lW "${OUT}/netboot.elf" |
    awk -v dma="${NETBOOT_DMA_BASE:-0}" '$1=="LOAD"{ lo=strtonum($3); hi=lo+strtonum($6)
        d=strtonum(dma); if (d && strtonum($5)==0 && lo >= d && hi <= d + 0x80000000) next   # NOLOAD DMA/BD
        if (lo < 0x78000000 || hi > 0x7A200000) bad=1 } END{ exit bad }' ||
    die "netboot.elf has a PT_LOAD outside 0x7800_0000..0x7A20_0000 (would collide with the app or staging)"
"${CROSS}size" "${OUT}/netboot.elf"
echo "[netboot] built ${OUT}/netboot.elf  (IP ${NETBOOT_IP}, port 5155${NETBOOT_DMA_BASE:+, GEM DMA at ${NETBOOT_DMA_BASE}})"
