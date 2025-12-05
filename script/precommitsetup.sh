#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEHLC_DIR="${SCRIPT_DIR}/../"

setup_git_hooks() {
    echo "Setting up git pre-commit hooks..."
    local HOOKS_DIR="$AIEHLC_DIR/.git/hooks"
    local PRE_COMMIT_HOOK="$HOOKS_DIR/pre-commit"
    
    if [ ! -d "$HOOKS_DIR" ]; then
        echo "Warning: Git hooks directory not found. Make sure you're in a git repository."
        return 1
    fi
    
    if [ -f "$PRE_COMMIT_HOOK" ]; then
        echo "Pre-commit hook already exists at $PRE_COMMIT_HOOK. Skipping installation."
        return 0
    fi
    
    cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
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