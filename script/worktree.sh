#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
###############################################################################
# Git Worktree Helper for AIEHLC
#
# Manages git worktrees under .worktrees/ for parallel development.
# Each worktree gets its own build/ directory so you can compile and test
# independently on different branches.
#
# Usage: bash script/worktree.sh <command> [args]
#
# Commands:
#   create <branch> [base]    Create worktree + new branch from base (default: main)
#   create-from <branch>      Create worktree from existing local/remote branch
#   list                      List all worktrees
#   remove <branch>           Remove worktree (prompts to delete branch)
#   build <branch>            Run cmake + make in the worktree's build/
#   build-unitest <branch>    Build the unitest binary in the worktree
#   cd <branch>               Print worktree path (use: cd $(bash script/worktree.sh cd <name>))
#   link-deps <wt-path> [--from <ref>] [--private-alib]
#                             Link untracked deps (thirdparty/alib, thirdparty/tvm-0.16,
#                             build/aiehlc, A78 BSP workspace) from a reference worktree
#                             (default: repo root). Idempotent; never overwrites an
#                             existing file/link. --private-alib: give the worktree its
#                             own COPY of thirdparty/alib (aiehlc.sh rebuilds aie-rt into
#                             alib in place, so parallel builds on a shared alib clobber
#                             each other); an existing real alib dir is kept.
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKTREES_DIR="${REPO_ROOT}/.worktrees"
LLVM_INSTALL_DIR="/scratch/staff/huaj/llvm-project/build/"

usage() {
    cat <<'EOF'
Usage: bash script/worktree.sh <command> [args]

Commands:
  create <branch> [base]    Create worktree + new branch from base (default: main)
  create-from <branch>      Create worktree from existing local/remote branch
  list                      List all worktrees
  remove <branch>           Remove worktree (prompts to delete branch)
  build <branch>            Run cmake + make in the worktree's build/
  build-unitest <branch>    Build the unitest binary in the worktree
  cd <branch>               Print worktree path
  link-deps <wt-path> [--from <ref-worktree>] [--private-alib]
                            Symlink thirdparty/alib, thirdparty/tvm-0.16,
                            build/aiehlc and the A78 BSP workspace from
                            <ref-worktree> (default: repo root) into <wt-path>.
                            Safe to re-run; existing files/links are left untouched.
                            --private-alib: copy thirdparty/alib (cp -a) instead of
                            linking it, so this worktree's aiehlc.sh rebuild of
                            aie-rt cannot clobber other worktrees. A symlinked alib
                            is replaced by a copy of its target; a real dir is kept.

Examples:
  bash script/worktree.sh create fix-routing main
  bash script/worktree.sh create-from routingbrokenbug
  bash script/worktree.sh build fix-routing
  bash script/worktree.sh remove fix-routing
  cd $(bash script/worktree.sh cd fix-routing)
  bash script/worktree.sh link-deps .worktrees/fix-routing --from .worktrees/unet_m2
  bash script/worktree.sh link-deps .worktrees/fix-routing --from .worktrees/unet_m2 --private-alib
EOF
}

ensure_worktrees_dir() {
    if [ ! -d "${WORKTREES_DIR}" ]; then
        mkdir -p "${WORKTREES_DIR}"
    fi
}

worktree_path() {
    local branch="$1"
    echo "${WORKTREES_DIR}/${branch}"
}

cmd_create() {
    local branch="${1:?Error: branch name required}"
    local base="${2:-main}"

    ensure_worktrees_dir

    local wt_path
    wt_path="$(worktree_path "${branch}")"

    if [ -d "${wt_path}" ]; then
        echo "Error: worktree already exists at ${wt_path}"
        exit 1
    fi

    echo "Creating worktree for new branch '${branch}' based on '${base}'..."
    git -C "${REPO_ROOT}" worktree add -b "${branch}" "${wt_path}" "${base}"

    echo ""
    echo "Worktree created at: ${wt_path}"
    echo "Branch: ${branch} (based on ${base})"
    echo ""
    cmd_link_deps "${wt_path}" || echo "Warning: link-deps failed; run: bash script/worktree.sh link-deps ${wt_path}"

    echo "Next steps:"
    echo "  cd ${wt_path}"
    echo "  bash script/worktree.sh build ${branch}"
}

cmd_create_from() {
    local branch="${1:?Error: branch name required}"

    ensure_worktrees_dir

    local wt_path
    wt_path="$(worktree_path "${branch}")"

    if [ -d "${wt_path}" ]; then
        echo "Error: worktree already exists at ${wt_path}"
        exit 1
    fi

    # Check if branch exists locally
    if git -C "${REPO_ROOT}" show-ref --verify --quiet "refs/heads/${branch}" 2>/dev/null; then
        echo "Creating worktree from existing local branch '${branch}'..."
        git -C "${REPO_ROOT}" worktree add "${wt_path}" "${branch}"
    # Check if branch exists on remote
    elif git -C "${REPO_ROOT}" show-ref --verify --quiet "refs/remotes/origin/${branch}" 2>/dev/null; then
        echo "Creating worktree from remote branch 'origin/${branch}'..."
        git -C "${REPO_ROOT}" worktree add --track -b "${branch}" "${wt_path}" "origin/${branch}"
    else
        echo "Error: branch '${branch}' not found locally or on origin"
        echo "Available branches:"
        git -C "${REPO_ROOT}" branch -a
        exit 1
    fi

    echo ""
    echo "Worktree created at: ${wt_path}"
    echo "Branch: ${branch}"
    echo ""
    cmd_link_deps "${wt_path}" || echo "Warning: link-deps failed; run: bash script/worktree.sh link-deps ${wt_path}"

    echo "Next steps:"
    echo "  cd ${wt_path}"
    echo "  bash script/worktree.sh build ${branch}"
}

cmd_list() {
    echo "Git worktrees:"
    echo ""
    git -C "${REPO_ROOT}" worktree list
}

cmd_remove() {
    local branch="${1:?Error: branch name required}"

    local wt_path
    wt_path="$(worktree_path "${branch}")"

    if [ ! -d "${wt_path}" ]; then
        echo "Error: no worktree found at ${wt_path}"
        exit 1
    fi

    echo "Removing worktree at ${wt_path}..."
    git -C "${REPO_ROOT}" worktree remove "${wt_path}" --force

    echo "Worktree removed."

    # Ask about branch deletion (skip in non-interactive mode)
    if git -C "${REPO_ROOT}" show-ref --verify --quiet "refs/heads/${branch}" 2>/dev/null; then
        if [ -t 0 ]; then
            read -r -p "Also delete branch '${branch}'? [y/N] " answer
            if [[ "${answer}" =~ ^[Yy]$ ]]; then
                git -C "${REPO_ROOT}" branch -D "${branch}"
                echo "Branch '${branch}' deleted."
            else
                echo "Branch '${branch}' kept."
            fi
        else
            echo "Branch '${branch}' kept (non-interactive mode; delete manually with: git branch -D ${branch})"
        fi
    fi
}

cmd_build() {
    local branch="${1:?Error: branch name required}"

    local wt_path
    wt_path="$(worktree_path "${branch}")"

    # Allow building in main repo too
    if [ "${branch}" = "main" ] && [ ! -d "${wt_path}" ]; then
        wt_path="${REPO_ROOT}"
    fi

    if [ ! -d "${wt_path}" ]; then
        echo "Error: no worktree found at ${wt_path}"
        echo "Create one first: bash script/worktree.sh create ${branch}"
        exit 1
    fi

    local build_dir="${wt_path}/build"
    mkdir -p "${build_dir}"

    echo "Building in ${build_dir}..."
    echo "LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}"

    cd "${build_dir}"
    cmake .. -DLLVM_INSTALL_DIR="${LLVM_INSTALL_DIR}"
    make -j"$(nproc)"

    echo ""
    echo "Build complete: ${build_dir}"
}

cmd_build_unitest() {
    local branch="${1:?Error: branch name required}"

    local wt_path
    wt_path="$(worktree_path "${branch}")"

    # Allow building in main repo too
    if [ "${branch}" = "main" ] && [ ! -d "${wt_path}" ]; then
        wt_path="${REPO_ROOT}"
    fi

    if [ ! -d "${wt_path}" ]; then
        echo "Error: no worktree found at ${wt_path}"
        echo "Create one first: bash script/worktree.sh create ${branch}"
        exit 1
    fi

    local unitest_dir="${wt_path}/src/mlir/mlirfront/tilinglinalg/pass/unitest"
    local build_dir="${unitest_dir}/build"

    if [ ! -d "${unitest_dir}" ]; then
        echo "Error: unitest directory not found at ${unitest_dir}"
        exit 1
    fi

    mkdir -p "${build_dir}"

    echo "Building unitest in ${build_dir}..."

    cd "${build_dir}"
    cmake ..
    make -j"$(nproc)"

    echo ""
    echo "Unitest build complete: ${build_dir}"
    echo "Run: ${build_dir}/test"
}

# Untracked dependencies a fresh worktree needs (relative to the worktree root).
# thirdparty/arch/cortexa78_0/workspace = the generated A78 BSP/platform (setup.sh); untracked,
# so a fresh worktree has none and aiehlc.sh stops with "AIE BSP not found".
LINK_DEPS=(thirdparty/alib thirdparty/tvm-0.16 build/aiehlc thirdparty/arch/cortexa78_0/workspace)

# link-deps <wt-path> [--from <ref-worktree>]
# For each dep: if missing in <wt-path>, symlink it to the ref's resolved
# target. Existing files/links (including dangling links) are never touched,
# so a worktree with its own build/aiehlc keeps it. Re-running is a no-op.
cmd_link_deps() {
    local wt="${1:?Error: worktree path required}"
    shift
    local ref="${REPO_ROOT}" private_alib=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --from) ref="${2:?Error: --from needs a path}"; shift 2 ;;
            --private-alib) private_alib=1; shift ;;
            *) echo "Error: unknown link-deps option '$1'" >&2; return 1 ;;
        esac
    done
    [ -d "${wt}" ] || { echo "Error: worktree not found: ${wt}" >&2; return 1; }
    [ -d "${ref}" ] || { echo "Error: reference worktree not found: ${ref}" >&2; return 1; }
    wt="$(cd "${wt}" && pwd)"
    ref="$(cd "${ref}" && pwd)"

    local dep src dst missing=0
    if [ "${private_alib}" -eq 1 ]; then
        private_alib_copy "${wt}" "${ref}" || return 1
    fi
    for dep in "${LINK_DEPS[@]}"; do
        dst="${wt}/${dep}"
        if [ -e "${dst}" ] || [ -L "${dst}" ]; then
            echo "link-deps: keep ${dep} (exists)"
            continue
        fi
        if [ ! -e "${ref}/${dep}" ]; then
            echo "link-deps: skip ${dep} (not in ref ${ref})" >&2
            missing=$((missing + 1))
            continue
        fi
        src="$(readlink -f "${ref}/${dep}")"
        if [ "${src}" = "${dst}" ]; then
            echo "link-deps: skip ${dep} (ref is the worktree itself)"
            continue
        fi
        mkdir -p "$(dirname "${dst}")"
        ln -s "${src}" "${dst}"
        echo "link-deps: ${dep} -> ${src}"
    done
    [ "${missing}" -eq 0 ] || echo "link-deps: ${missing} dep(s) missing in ref" >&2
    return 0
}

# private_alib_copy <wt> <ref>: make <wt>/thirdparty/alib a real directory (cp -a of the
# ref's resolved alib). aiehlc.sh compiles aie-rt into alib in place (build_hw_lib), so a
# shared symlinked alib is rewritten by every concurrent build ("xaiegbl.h: No such file").
# Keeps an existing real dir (idempotent); replaces a symlink by a copy of its target.
# The copy goes to a temp dir next to the destination and is moved in atomically; the
# source is only read.
private_alib_copy() {
    local wt="$1" ref="$2" dst="$1/thirdparty/alib" src tmp
    if [ -d "${dst}" ] && [ ! -L "${dst}" ]; then
        echo "link-deps: keep thirdparty/alib (private dir)"
        return 0
    fi
    if [ -L "${dst}" ]; then
        src="$(readlink -f "${dst}")"
    else
        [ -e "${ref}/thirdparty/alib" ] || { echo "Error: --private-alib: no thirdparty/alib in ref ${ref}" >&2; return 1; }
        src="$(readlink -f "${ref}/thirdparty/alib")"
    fi
    [ -d "${src}" ] || { echo "Error: --private-alib: alib source is not a directory: ${src}" >&2; return 1; }
    [ "${src}" != "${dst}" ] || { echo "Error: --private-alib: source is the destination ${dst}" >&2; return 1; }
    mkdir -p "$(dirname "${dst}")"
    tmp="$(mktemp -d "$(dirname "${dst}")/.alib_private.XXXXXX")" || return 1
    if ! cp -a "${src}/." "${tmp}/"; then
        rm -rf "${tmp}"; echo "Error: --private-alib: copy from ${src} failed" >&2; return 1
    fi
    chmod --reference="${src}" "${tmp}"
    if [ -L "${dst}" ]; then rm -f "${dst}"; fi
    mv -T "${tmp}" "${dst}" || { rm -rf "${tmp}"; return 1; }
    echo "link-deps: thirdparty/alib private copy of ${src} ($(du -sh "${dst}" | cut -f1))"
}

cmd_cd() {
    local branch="${1:?Error: branch name required}"

    local wt_path
    wt_path="$(worktree_path "${branch}")"

    if [ ! -d "${wt_path}" ]; then
        echo "Error: no worktree found at ${wt_path}" >&2
        exit 1
    fi

    # Print just the path (stdout) so it can be used with: cd $(bash script/worktree.sh cd <name>)
    echo "${wt_path}"
}

# --- Main dispatch ---

command="${1:-}"
shift || true

case "${command}" in
    create)
        cmd_create "$@"
        ;;
    create-from)
        cmd_create_from "$@"
        ;;
    list)
        cmd_list
        ;;
    remove)
        cmd_remove "$@"
        ;;
    build)
        cmd_build "$@"
        ;;
    build-unitest)
        cmd_build_unitest "$@"
        ;;
    cd)
        cmd_cd "$@"
        ;;
    link-deps)
        cmd_link_deps "$@"
        ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        echo "Error: unknown command '${command}'"
        echo ""
        usage
        exit 1
        ;;
esac
