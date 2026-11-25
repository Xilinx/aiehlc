#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Install the latest public AMD Vitis (FPGAs & Adaptive SoCs Unified Installer)
# for aiehlc on a machine without the AMD-internal /proj xbuilds. Called by
# script/setup.sh when no Vitis is found; can also be run on its own:
#
#   bash script/install_vitis.sh [--dest DIR] [--installer FILE] [--version VER]
#
# Steps (each one is skipped when already done, so a rerun resumes):
#   1. Pick the version: --version / AIEHLC_VITIS_VERSION, else the newest
#      release listed on the AMD download page.
#   2. Get the web installer FPGAs_AdaptiveSoCs_Unified_SDI_<ver>_*_Lin64.bin:
#      --installer / AIEHLC_VITIS_INSTALLER, else look in aiehlc/download/
#      (AIEHLC_VITIS_DOWNLOAD_DIR), else try to download it from the link on the
#      AMD download page. AMD usually requires an account login for that, so if
#      it fails the script prints the link, asks you to save the file into
#      aiehlc/download/, waits for a key press and looks there again.
#   3. Authenticate the web installer: `xsetup -b AuthTokenGen` asks for the
#      AMD account e-mail and password (token cached in ~/.Xilinx for 7 days).
#      Non-interactive: export AMD_USER / AMD_PASSWORD.
#   4. Batch-install Vitis (includes Vivado + aietools) with only the Versal
#      device families, into DEST/<ver>/ (DEST default: the folder next to
#      this checkout, ../Xilinx; override --dest / AIEHLC_VITIS_INSTALL_DIR).
#
# On success prints the settings64.sh path on the last line:
#   VITIS_SETTINGS=<DEST>/<ver>/Vitis/settings64.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEHLC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="${AIEHLC_VITIS_INSTALL_DIR:-$(cd "$AIEHLC_DIR/.." && pwd)/Xilinx}"
INSTALLER="${AIEHLC_VITIS_INSTALLER:-}"
VERSION="${AIEHLC_VITIS_VERSION:-}"
DOWNLOAD_DIR="${AIEHLC_VITIS_DOWNLOAD_DIR:-$AIEHLC_DIR/download}"
DOWNLOAD_INDEX="https://www.xilinx.com/support/download.html"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --installer) INSTALLER="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        -h|--help) sed -n '6,32p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

log() { echo "[install_vitis] $*" >&2; }

# The AMD-internal Vitis (same default as script/setup.sh) wins: never download
# when it is there. AIEHLC_VITIS_FORCE_INSTALL=1 installs anyway.
SITE_VITIS="${VITIS_SETTINGS_PATH:-/proj/xbuilds/HEAD_qualified_latest/installs/lin64/HEAD/Vitis/settings64.sh}"
if [ -f "$SITE_VITIS" ] && [ "${AIEHLC_VITIS_FORCE_INSTALL:-0}" != "1" ]; then
    log "Vitis found at $SITE_VITIS; nothing to download"
    echo "VITIS_SETTINGS=$SITE_VITIS"
    exit 0
fi

# ---------------------------------------------------------------- 1. version
# Installer names look like FPGAs_AdaptiveSoCs_Unified_SDI_2026.1.1_0904_0832_Lin64.bin;
# list them from the download page (it redirects to the newest release).
PAGE="$(curl -sSL --http1.1 --max-time 60 "$DOWNLOAD_INDEX" 2>/dev/null || true)"
list_installers() {
    printf '%s' "$PAGE" | grep -oE 'FPGAs_AdaptiveSoCs_Unified_SDI_[0-9.]+_[0-9]+_[0-9]+_Lin64\.bin' | sort -u
}
INSTALLER_NAME=""
if [ -n "$INSTALLER" ]; then
    INSTALLER_NAME="$(basename "$INSTALLER")"
else
    names="$(list_installers || true)"
    if [ -n "$VERSION" ]; then
        names="$(printf '%s\n' "$names" | grep "_SDI_${VERSION}_" || true)"
    fi
    # newest version first (2026.1.1 > 2026.1)
    INSTALLER_NAME="$(printf '%s\n' "$names" | sed -n 's/.*_SDI_\([0-9.]*\)_.*/\1 &/p' \
                      | sort -k1,1Vr | head -1 | cut -d' ' -f2)"
fi
if [ -z "$VERSION" ]; then
    VERSION="$(printf '%s' "$INSTALLER_NAME" | sed -n 's/.*_SDI_\([0-9.]*\)_.*/\1/p')"
fi
if [ -z "$VERSION" ]; then
    log "ERROR: could not determine the Vitis version from $DOWNLOAD_INDEX."
    log "       Pass --version <e.g. 2026.1> and --installer <path to the _Lin64.bin>."
    exit 1
fi
log "Vitis version: $VERSION"

# Already installed?
for s in "$DEST/$VERSION/Vitis/settings64.sh" "$DEST/Vitis/$VERSION/settings64.sh"; do
    if [ -f "$s" ]; then
        log "Vitis $VERSION already installed"
        echo "VITIS_SETTINGS=$s"
        exit 0
    fi
done

# ---------------------------------------------------------------- 2. installer
# is_installer <file>: a complete self-extracting installer (not an HTML login
# page or a partial download).
is_installer() {
    local f="$1"
    [ -s "$f" ] || return 1
    head -c 2 "$f" | grep -q '^#!' || return 1                 # a shell script, not HTML
    [ "$(stat -c %s "$f")" -gt 10000000 ] || return 1          # > 10 MB
    if head -c 20000 "$f" | grep -qa 'makeself'; then            # makeself: verify checksums
        bash "$f" --check >/dev/null 2>&1 || return 1
    fi
    return 0
}
# find_installer: newest web installer in DOWNLOAD_DIR (then the older spots),
# preferring the wanted VERSION. Sets INSTALLER and VERSION.
find_installer() {
    local d f
    for d in "$DOWNLOAD_DIR" "$DEST/downloads" "$HOME/Downloads"; do
        f="$(ls -1 "$d"/FPGAs_AdaptiveSoCs_Unified_SDI_"${VERSION}"_*_Lin64.bin 2>/dev/null | sort | tail -1 || true)"
        [ -z "$f" ] && f="$(ls -1 "$d"/FPGAs_AdaptiveSoCs_Unified_SDI_*_Lin64.bin 2>/dev/null | sort -V | tail -1 || true)"
        if [ -n "$f" ]; then
            if is_installer "$f"; then
                INSTALLER="$f"
                VERSION="$(basename "$f" | sed -n 's/.*_SDI_\([0-9.]*\)_.*/\1/p')"
                return 0
            fi
            log "WARNING: $f is not a complete installer (still downloading, or a login page?); ignoring it"
        fi
    done
    return 1
}

if [ -z "$INSTALLER" ]; then
    mkdir -p "$DOWNLOAD_DIR"
    find_installer || true
fi
# (a) Try to download it directly from the link on the AMD download page.
if [ -z "$INSTALLER" ] && [ -n "$INSTALLER_NAME" ]; then
    url="$(printf '%s' "$PAGE" | grep -oE "https://download\.amd\.com/[^\"' <>]*/${INSTALLER_NAME}" | head -1 || true)"
    if [ -n "$url" ]; then
        log "Trying direct download: $url"
        if curl -fSL --http1.1 --retry 2 --max-time 1800 -o "$DOWNLOAD_DIR/$INSTALLER_NAME.part" "$url" \
           && is_installer "$DOWNLOAD_DIR/$INSTALLER_NAME.part"; then
            mv "$DOWNLOAD_DIR/$INSTALLER_NAME.part" "$DOWNLOAD_DIR/$INSTALLER_NAME"
            INSTALLER="$DOWNLOAD_DIR/$INSTALLER_NAME"
        else
            log "Direct download failed (AMD requires an account login for this file)."
        fi
        rm -f "$DOWNLOAD_DIR/$INSTALLER_NAME.part"
    fi
fi
# (b) Ask the user to download it in a browser, wait, then look again.
if [ -z "$INSTALLER" ] || [ ! -f "$INSTALLER" ]; then
    name="${INSTALLER_NAME:-FPGAs_AdaptiveSoCs_Unified_SDI_${VERSION}_<build>_Lin64.bin}"
    cat >&2 <<EOF

==============================================================================
 Please download the Vitis $VERSION web installer (AMD account login needed):
   https://account.amd.com/en/forms/downloads/xef.html?filename=$name
   (or "Linux Self Extracting Web Installer" for $VERSION on
    $DOWNLOAD_INDEX)
 and save it as:
   $DOWNLOAD_DIR/$name
==============================================================================
EOF
    if [ ! -t 0 ]; then
        log "No terminal to wait on: save the file there and rerun (or pass --installer <file>)."
        exit 2
    fi
    while :; do
        read -r -n 1 -s -p "Press any key once the download has finished (q to quit) ... " key </dev/tty
        echo >&2
        [ "$key" = "q" ] && { log "Stopped by user."; exit 2; }
        find_installer && break
        log "No complete FPGAs_AdaptiveSoCs_Unified_SDI_*_Lin64.bin in $DOWNLOAD_DIR yet."
    done
fi
log "Installer: $INSTALLER (Vitis $VERSION)"

# Extract once (a self-extracting makeself archive) to get xsetup.
EXTRACT="$DEST/.installer/$(basename "$INSTALLER" .bin)"
if [ ! -x "$EXTRACT/xsetup" ]; then
    mkdir -p "$EXTRACT"
    log "Extracting installer to $EXTRACT"
    bash "$INSTALLER" --keep --noexec --target "$EXTRACT" >&2
fi
XSETUP="$EXTRACT/xsetup"

# ---------------------------------------------------------------- 3. auth token
TOKEN="$HOME/.Xilinx/wi_authentication_key"
if [ -s "$TOKEN" ] && [ -n "$(find "$TOKEN" -mtime -7 2>/dev/null)" ]; then
    log "Using the AMD auth token in $TOKEN"
else
    log "Generating the AMD web-installer auth token (AMD account e-mail + password)"
    if [ -n "${AMD_USER:-}" ] && [ -n "${AMD_PASSWORD:-}" ]; then
        command -v expect >/dev/null || { log "ERROR: expect not installed (needed with AMD_USER/AMD_PASSWORD)"; exit 1; }
        expect -c "
            set timeout 180
            spawn $XSETUP -b AuthTokenGen
            expect -re {User ID:?}  { send -- \"\$env(AMD_USER)\r\" }
            expect -re {Password:?} { send -- \"\$env(AMD_PASSWORD)\r\" }
            expect eof
            catch wait r
            exit [lindex \$r 3]" >&2
    elif [ -t 0 ]; then
        "$XSETUP" -b AuthTokenGen >&2
    else
        log "ERROR: no terminal to ask for the AMD login. Run interactively:"
        log "         $XSETUP -b AuthTokenGen"
        log "       or export AMD_USER and AMD_PASSWORD, then rerun."
        exit 3
    fi
    [ -s "$TOKEN" ] || { log "ERROR: AuthTokenGen did not create $TOKEN (wrong login?)"; exit 3; }
fi

# ---------------------------------------------------------------- 4. install
CFG="$DEST/.installer/install_config_${VERSION}.txt"
if [ ! -f "$CFG" ]; then
    log "Generating the install config"
    # ConfigGen writes ~/.Xilinx/install_config.txt for the chosen product/edition.
    rm -f "$HOME/.Xilinx/install_config.txt"
    "$XSETUP" -b ConfigGen -p "Vitis" -e "Vitis Unified Software Platform" >&2 </dev/null || true
    [ -f "$HOME/.Xilinx/install_config.txt" ] || { log "ERROR: xsetup -b ConfigGen produced no config"; exit 4; }
    # Destination and devices. The generated config has every device off and
    # lists Versal parts by part number; turn on only the aiehlc boards
    # (AIEHLC_VITIS_DEVICES): VCK190 xcvc1902 (AIE), VEK280 xcve2802 (AIE-ML),
    # VEK385 xc2ve3858 (AIE2PS). No desktop shortcuts (headless/docker).
    awk -v dest="$DEST" -v devs="${AIEHLC_VITIS_DEVICES:-xcvc1902 xcve2802 xc2ve3858}" '
        BEGIN { n = split(devs, d, " "); for (i = 1; i <= n; i++) want[d[i]] = 1 }
        /^Destination=/ { print "Destination=" dest; next }
        /^(CreateProgramGroupShortcuts|CreateDesktopShortcuts|CreateFileAssociation)=/ {
            sub(/=.*/, "=0"); print; next }
        /^Modules=/ {
            sub(/^Modules=/, ""); m = split($0, mod, ","); out = ""
            for (i = 1; i <= m; i++) {
                k = mod[i]; sub(/:[01]$/, "", k); v = substr(mod[i], length(mod[i]))
                if (k in want) { v = 1; found[k] = 1 }
                out = out (out == "" ? "" : ",") k ":" v
            }
            for (k in want) if (!(k in found)) printf "[install_vitis] WARNING: device %s not offered by this installer\n", k > "/dev/stderr"
            print "Modules=" out; next }
        { print }' "$HOME/.Xilinx/install_config.txt" > "$CFG"
fi
log "Install config: $CFG"
grep -E '^(Edition|Product|Destination|Modules)=' "$CFG" >&2 || true
mkdir -p "$DEST"
log "Installing Vitis $VERSION into $DEST (downloads ~tens of GB; takes a while)"
"$XSETUP" -a XilinxEULA,3rdPartyEULA -b Install -c "$CFG" >&2

for s in "$DEST/$VERSION/Vitis/settings64.sh" "$DEST/Vitis/$VERSION/settings64.sh"; do
    if [ -f "$s" ]; then
        log "Vitis $VERSION installed"
        echo "VITIS_SETTINGS=$s"
        exit 0
    fi
done
log "ERROR: install finished but no Vitis/settings64.sh under $DEST"
exit 5
