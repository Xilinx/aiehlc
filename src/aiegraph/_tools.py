###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""The ``aiehlc`` console command: ``script/aiehlc.sh`` without ``source``.

    aiehlc --aie-version 5 --runtime-source-file kernel.cc

``aiehlc.sh`` is written to be SOURCED and writes ``aout/`` relative to the
current directory (CLAUDE.md); this runs it in a bash child with the caller's
cwd, so it behaves exactly like ``source script/aiehlc.sh ...`` in that
directory, without leaking its environment into the caller's shell.
"""

from __future__ import annotations

import subprocess
import sys


def aiehlc_main(argv=None) -> int:
    from ._env import repo_root
    from .errors import AiegraphError

    try:
        script = repo_root() / "script" / "aiehlc.sh"
    except AiegraphError as exc:
        print(exc, file=sys.stderr)
        return 2
    args = sys.argv[1:] if argv is None else list(argv)
    return subprocess.run(["bash", "-c", 'source "$0" "$@"', str(script), *args]).returncode


if __name__ == "__main__":
    sys.exit(aiehlc_main())
