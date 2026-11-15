"""A deadline fallback must answer once, even if the worker finishes later."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))

from harness import core  # noqa: E402


class ClockTests(unittest.TestCase):
    def test_watchdog_uses_original_receipt_time_and_emits_once(self) -> None:
        lines: list[dict] = []
        response = core.Responder()
        with patch.object(core, "emit", side_effect=lines.append):
            response.arm(20.0, {"type": "action", "action": "A1", "chat": "Your move."},
                         received_at=time.monotonic() - 19.0)
            time.sleep(0.25)
            self.assertFalse(response.send({"type": "action", "action": "B2", "chat": "Later."}))
        self.assertEqual(lines, [{"type": "action", "action": "A1", "chat": "Your move."}])


if __name__ == "__main__":
    unittest.main()
