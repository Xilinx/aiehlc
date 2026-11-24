###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
import os
import sys
import threading
import time
import unittest
from unittest.mock import Mock


sys.path.insert(0, os.path.dirname(__file__))

from schedule_debug_server import DebugState


class LlmRetargetTest(unittest.TestCase):
    def make_state(self):
        state = DebugState.__new__(DebugState)
        state.target = "xsdb://board:3121"
        state.device = "vek385"
        state._gdb_lock = threading.Lock()
        state._gdb_proc = None
        state._llm_generation = 4
        state._llm_buf = "existing conversation"
        state.llm_enabled = True
        state._invalidate_mcp_config = Mock()
        state._write_backend_status = Mock()
        state.llm_reset = Mock(return_value={
            "ok": True,
            "llm_generation": 5,
            "llm_reset_reason": "debug target changed to xsdb://other:3121",
        })
        return state

    def test_same_target_does_not_reset_llm(self):
        state = self.make_state()

        result = state.retarget("xsdb://board:3121", "vek385")

        self.assertFalse(result["llm_reset"])
        self.assertEqual(result["llm_generation"], 4)
        state.llm_reset.assert_not_called()
        state._invalidate_mcp_config.assert_not_called()

    def test_changed_target_preserves_llm_generation(self):
        state = self.make_state()

        result = state.retarget("xsdb://other:3121", "vek385")

        self.assertFalse(result["llm_reset"])
        self.assertEqual(result["llm_generation"], 4)
        self.assertEqual(state.target, "xsdb://other:3121")
        self.assertEqual(state._llm_buf, "existing conversation")
        state.llm_reset.assert_not_called()
        state._invalidate_mcp_config.assert_called_once_with()
        state._write_backend_status.assert_called_once_with()

    def test_poll_exposes_process_generation(self):
        state = DebugState.__new__(DebugState)
        state._llm_lock = threading.Lock()
        state._llm_buf = "partial answer"
        state._llm_active = False
        state._llm_last_output = None
        state._llm_generation = 7
        state._llm_reset_reason = "debug target changed"

        result = state.llm_poll(0)

        self.assertEqual(result["llm_generation"], 7)
        self.assertEqual(result["llm_reset_reason"], "debug target changed")
        self.assertEqual(result["data"], "partial answer")

    def _make_poll_state(self, *, active, age_s, proc_alive=True):
        state = DebugState.__new__(DebugState)
        state._llm_lock = threading.Lock()
        state._llm_buf = "partial answer"
        state._llm_active = active
        state._llm_last_output = time.monotonic() - age_s
        state._llm_generation = 1
        state._llm_reset_reason = ""
        state._llm_log_write = Mock()
        proc = Mock()
        proc.poll.return_value = None if proc_alive else 0
        state._llm_proc = proc
        return state

    def test_poll_soft_stuck_keeps_active_when_proc_alive(self):
        state = self._make_poll_state(active=True, age_s=150, proc_alive=True)

        result = state.llm_poll(0)

        self.assertTrue(result["active"])
        self.assertNotIn("stuck", result)
        state._llm_log_write.assert_not_called()

    def test_poll_hard_stuck_clears_active_when_proc_dead(self):
        state = self._make_poll_state(active=True, age_s=150, proc_alive=False)

        result = state.llm_poll(0)

        self.assertFalse(result["active"])
        self.assertTrue(result["stuck"])
        state._llm_log_write.assert_called_once()

    def test_poll_hard_stuck_after_long_silence_even_if_proc_alive(self):
        from schedule_debug_server import _LLM_STUCK_HARD_S
        state = self._make_poll_state(
            active=True, age_s=_LLM_STUCK_HARD_S + 5, proc_alive=True)

        result = state.llm_poll(0)

        self.assertFalse(result["active"])
        self.assertTrue(result["stuck"])
        state._llm_log_write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
