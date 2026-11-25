#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Install the repo's git hooks (script/hooks/*) into this clone, as symlinks, so a `git pull` that
# updates a hook takes effect without reinstalling. Worktrees share the hooks of their main clone.
#   bash script/install_git_hooks.sh
# Never overwrites a different existing hook; it reports it instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)/hooks"
custom="$(git -C "$ROOT" config core.hooksPath || true)"
if [ -n "$custom" ]; then
    echo "install_git_hooks: core.hooksPath=${custom} is set; git ignores ${HOOKS_DIR}."
    echo "                   Link script/hooks/* into ${custom} yourself."
    exit 1
fi
mkdir -p "$HOOKS_DIR"
for src in "$ROOT"/script/hooks/*; do
    name="$(basename "$src")"
    dst="$HOOKS_DIR/$name"
    chmod +x "$src"
    if [ -L "$dst" ] && [ "$(readlink -f "$dst")" = "$(readlink -f "$src")" ]; then
        echo "install_git_hooks: $name already installed"
    elif [ -e "$dst" ] || [ -L "$dst" ]; then
        echo "install_git_hooks: $dst exists and is not ours -- left alone (remove it to install $name)"
    else
        ln -s "$src" "$dst"
        echo "install_git_hooks: installed $name -> $src"
    fi
done
