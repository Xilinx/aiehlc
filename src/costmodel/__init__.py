###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""AIEHLC local cost layer (AIE side): parsers, parameters, discrete-event model."""

from .estimate import estimate  # noqa: E402,F401  (public API: costmodel.estimate)

__all__ = ["estimate"]
