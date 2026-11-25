#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEHLC_DIR="${SCRIPT_DIR}/../"

setup_git_hooks() {
    echo "Setting up git pre-commit hooks..."
    # Use git rev-parse to find hooks dir (works in both normal repos and worktrees)
    local HOOKS_DIR
    HOOKS_DIR="$(git -C "$AIEHLC_DIR" rev-parse --git-path hooks 2>/dev/null)" || true
    if [ -z "$HOOKS_DIR" ] || [ ! -d "$HOOKS_DIR" ]; then
        # Fallback: try creating hooks dir if we can resolve git dir
        local GIT_DIR
        GIT_DIR="$(git -C "$AIEHLC_DIR" rev-parse --git-common-dir 2>/dev/null)" || true
        if [ -n "$GIT_DIR" ] && [ -d "$GIT_DIR" ]; then
            HOOKS_DIR="$GIT_DIR/hooks"
            mkdir -p "$HOOKS_DIR"
        else
            echo "Warning: Git hooks directory not found. Make sure you're in a git repository."
            return 0
        fi
    fi
    local PRE_COMMIT_HOOK="$HOOKS_DIR/pre-commit"
    
    if [ -f "$PRE_COMMIT_HOOK" ]; then
        echo "Pre-commit hook already exists at $PRE_COMMIT_HOOK. Skipping installation."
        return 0
    fi
    
    cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

# This hook runs git-clang-format to check and format code before commits

if ! command -v git-clang-format &> /dev/null; then
    echo "Warning: git-clang-format not found in PATH"
    echo "Clang-format checking will be skipped"
    echo "Please install clang-format tools for automatic code formatting"
    exit 0
fi

staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(c|cc|cpp|cxx|h|hpp|hxx)$' || true)

if [ -z "$staged_files" ]; then
    exit 0
fi

echo "Auto-formatting staged C/C++ files with clang-format..."

for file in $staged_files; do
    git clang-format --staged "$file"
    git add "$file"
done

echo "Code formatting complete. Files have been automatically formatted and re-staged."
exit 0
EOF
    
    chmod +x "$PRE_COMMIT_HOOK"
    echo "Git pre-commit hook installed successfully at $PRE_COMMIT_HOOK"
}

setup_git_hooks
# pre-push: block a push that fails script/ci/check_opensource.py
bash "${SCRIPT_DIR}/install_git_hooks.sh" || echo "Warning: git hook install failed (see above)."
