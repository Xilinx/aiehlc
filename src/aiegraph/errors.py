###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""One exception type, always with the stage, the reason and what to do next."""

from __future__ import annotations

__all__ = ["AiegraphError"]


class AiegraphError(RuntimeError):
    """``stage`` (install / load / convert / layer / check / offload / build),
    ``reason`` (what happened), ``hint`` (the next thing to try)."""

    def __init__(self, stage: str, reason: str, hint: str = ""):
        self.stage, self.reason, self.hint = stage, reason, hint
        msg = f"[{stage}] {reason}"
        if hint:
            msg += f"\n  hint: {hint}"
        super().__init__(msg)
