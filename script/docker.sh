#!/bin/bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Build (once) and enter the aiehlc dev container.
#   bash script/docker.sh                # interactive shell, --rm
#   AIEHLC_DOCKER_NAME=aiehlc-dev AIEHLC_DOCKER_DETACH=1 bash script/docker.sh   # long-lived container
# Host /proj (Vitis / PetaLinux installs) and /scratch are mapped 1:1 so the
# absolute paths in setup.sh (VITIS_SETTINGS_PATH, LLVM_INSTALL_DIR) work as-is.
# Extra mounts: AIEHLC_DOCKER_MOUNTS="/a:/a /b:/b"
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${AIEHLC_DOCKER_IMAGE:-aiehlc-dev:latest}"
NAME="${AIEHLC_DOCKER_NAME:-}"
# ~/.docker is often on a small-quota home; keep buildx state elsewhere if asked.
export DOCKER_CONFIG="${DOCKER_CONFIG:-$HOME/.docker}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building $IMAGE from $SCRIPT_DIR/docker/Dockerfile"
    DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-0} docker build -t "$IMAGE" \
        --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" --build-arg USERNAME="$(id -un)" \
        "$SCRIPT_DIR/docker"
fi

MOUNTS=()
for d in /proj /scratch ${AIEHLC_DOCKER_MOUNTS:-}; do
    case "$d" in
        *:*) MOUNTS+=(-v "$d") ;;
        # rslave: /proj (xbuilds) is autofs; without propagation a submount
        # that expires on the host turns into "Too many levels of symbolic
        # links" inside the container and settings64.sh silently fails.
        *) [ -d "$d" ] && MOUNTS+=(-v "$d:$d:rslave") ;;
    esac
done

# ssh keys for script/test/apppaltest.py (board host) and git
[ -d "$HOME/.ssh" ] && MOUNTS+=(-v "$HOME/.ssh:/home/$(id -un)/.ssh:ro")
# Vitis/aietools license files: mount every hidden dir under $HOME that holds
# *.lic (node-locked licenses also need --network host for the NIC hostid);
# without them xchesscc fails with "AIEBuild license not found". Any
# *_LICENSE_FILE variable set on the host (license server) is forwarded.
for d in "$HOME"/.[!.]*/; do
    d="${d%/}"
    compgen -G "$d/*.lic" >/dev/null && MOUNTS+=(-v "$d:/home/$(id -un)/$(basename "$d")")
done
ENVS=()
for v in $(compgen -v | grep '_LICENSE_FILE$'); do
    [ -n "${!v:-}" ] && ENVS+=(-e "$v=${!v}")
done

OPTS=(--network host -e TZ="${TZ:-America/Los_Angeles}" "${ENVS[@]}" -w "$(pwd)")
[ -n "$NAME" ] && OPTS+=(--name "$NAME")
if [ "${AIEHLC_DOCKER_DETACH:-0}" = "1" ]; then
    docker run -d "${OPTS[@]}" "${MOUNTS[@]}" "$IMAGE" sleep infinity
else
    docker run -it --rm "${OPTS[@]}" "${MOUNTS[@]}" "$IMAGE" /bin/bash
fi
