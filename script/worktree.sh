#!/usr/bin/env bash
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
#   link-deps <wt-path> [--from <ref>]
#                             Link untracked deps (thirdparty/alib, thirdparty/tvm-0.16,
#                             build/aiehlc) from a reference worktree (default: repo root).
#                             Idempotent; never overwrites an existing file/link.
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
  link-deps <wt-path> [--from <ref-worktree>]
                            Symlink thirdparty/alib, thirdparty/tvm-0.16 and
                            build/aiehlc from <ref-worktree> (default: repo root)
                            into <wt-path>. Safe to re-run; existing files/links
                            are left untouched.

Examples:
  bash script/worktree.sh create fix-routing main
  bash script/worktree.sh create-from routingbrokenbug
  bash script/worktree.sh build fix-routing
  bash script/worktree.sh remove fix-routing
  cd $(bash script/worktree.sh cd fix-routing)
  bash script/worktree.sh link-deps .worktrees/fix-routing --from .worktrees/unet_m2
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
LINK_DEPS=(thirdparty/alib thirdparty/tvm-0.16 build/aiehlc)

# link-deps <wt-path> [--from <ref-worktree>]
# For each dep: if missing in <wt-path>, symlink it to the ref's resolved
# target. Existing files/links (including dangling links) are never touched,
# so a worktree with its own build/aiehlc keeps it. Re-running is a no-op.
cmd_link_deps() {
    local wt="${1:?Error: worktree path required}"
    shift
    local ref="${REPO_ROOT}"
    while [ $# -gt 0 ]; do
        case "$1" in
            --from) ref="${2:?Error: --from needs a path}"; shift 2 ;;
            *) echo "Error: unknown link-deps option '$1'" >&2; return 1 ;;
        esac
    done
    [ -d "${wt}" ] || { echo "Error: worktree not found: ${wt}" >&2; return 1; }
    [ -d "${ref}" ] || { echo "Error: reference worktree not found: ${ref}" >&2; return 1; }
    wt="$(cd "${wt}" && pwd)"
    ref="$(cd "${ref}" && pwd)"

    local dep src dst missing=0
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
