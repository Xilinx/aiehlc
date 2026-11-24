###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""python -m costmodel <subcommand> -- see cli.py"""
import sys

from .cli import main

sys.exit(main())
