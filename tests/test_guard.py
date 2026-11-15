"""Speech-ban checks for lookalike text used in opponent bait."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))

from harness.guard import SpeechGuard


class SpeechGuardTests(unittest.TestCase):
    def test_ban_catches_cyrillic_lookalikes(self) -> None:
        guard = SpeechGuard(["pass"])
        self.assertTrue(guard.hits("\u0440\u0430\u0455\u0455"))


if __name__ == "__main__":
    unittest.main()
