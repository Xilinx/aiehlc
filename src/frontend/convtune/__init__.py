###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""convtune: tune the aiehlc conv2d auto-tiling knobs with TVM AutoTVM (XGBoost).

    from frontend.convtune import convtile            # model only (no TVM needed)
    from frontend.convtune import autotvm_bridge      # AutoTVM task / builder / runner / tune()

See src/frontend/convtune/README.md.
"""
