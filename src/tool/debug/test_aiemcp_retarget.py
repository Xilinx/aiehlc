###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
import os
import sys
import types
import unittest
from unittest.mock import patch


sys.path.insert(0, os.path.dirname(__file__))

import aiemcp


class AieMcpRetargetTest(unittest.TestCase):
    def setUp(self):
        self.old_gdb = aiemcp._gdb
        self.old_config = aiemcp._current_config
        keys = (
            "AIEDBG_TARGET", "AIEMCP_BACKEND", "AIEMCP_DEVICE",
            "AIEMCP_STARTCOL", "AIEMCP_AIE_VERSION",
        )
        self.old_env = {key: os.environ.get(key) for key in keys}

    def tearDown(self):
        aiemcp._gdb = self.old_gdb
        aiemcp._current_config = self.old_config
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_hardware_retarget_preserves_scope_and_updates_config(self):
        current = types.SimpleNamespace(
            target="xsdb://old:3121",
            device="pal",
            startcol=0,
            aie_version="2ps",
            tile={"col": 3, "row": 1},
            channel={"direction": "mm2s", "channel": 0},
            _reg_read="old-read",
            _passthrough="old-passthrough",
        )
        fresh = types.SimpleNamespace(
            target="xsdb://new:3121",
            device="vek385",
            startcol=4,
            aie_version="2ps",
            _reg_read="new-read",
            _passthrough="new-passthrough",
        )
        status = {
            "backend": "hardware",
            "target": "xsdb://new:3121",
            "device": "vek385",
            "startcol": 4,
            "aie_version": "2ps",
            "dbg_socket": "",
        }
        aiemcp._gdb = current
        aiemcp._current_config = (
            "hardware", "xsdb://old:3121", "pal", "0", "2ps", "",
        )

        with patch.object(aiemcp, "_read_backend_status", return_value=status), \
                patch.object(aiemcp, "_build_gdb", return_value=fresh):
            aiemcp._ensure_backend_current()

        self.assertIs(aiemcp._gdb, current)
        self.assertEqual(current.tile, {"col": 3, "row": 1})
        self.assertEqual(current.channel, {"direction": "mm2s", "channel": 0})
        self.assertEqual(current.target, "xsdb://new:3121")
        self.assertEqual(current.device, "vek385")
        self.assertEqual(current.startcol, 4)
        self.assertEqual(current._reg_read, "new-read")
        self.assertEqual(current._passthrough, "new-passthrough")

    def test_unchanged_config_does_not_rebuild(self):
        status = {
            "backend": "hardware",
            "target": "xsdb://board:3121",
            "device": "vek385",
            "startcol": 0,
            "aie_version": "2ps",
            "dbg_socket": "",
        }
        aiemcp._current_config = (
            "hardware", "xsdb://board:3121", "vek385", "0", "2ps", "",
        )

        with patch.object(aiemcp, "_read_backend_status", return_value=status), \
                patch.object(aiemcp, "_build_gdb") as build:
            aiemcp._ensure_backend_current()

        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
