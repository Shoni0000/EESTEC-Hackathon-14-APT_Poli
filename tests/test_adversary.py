"""Safety and strategic invariants for outgoing opponent chat."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))

from harness.adversary import draft_chat, draft_ready_chat  # noqa: E402
from harness.defense import safe_outgoing_chat  # noqa: E402


class AdversaryTests(unittest.TestCase):
    def test_number_claim_points_away_from_actual_secret(self) -> None:
        for actual, false_half in ((42, "upper half"), (78, "lower half")):
            state = {
                "game_id": "guess_number", "variant": {"lo": 1, "hi": 100},
                "turn": 1, "game_index": 1,
                "observation": f"GUESS THE NUMBER (1-100). Your secret number (the opponent is looking for it): {actual}",
            }
            line = draft_chat(state=state, action="guess 50", hidden=True, family="guessing", signal="none")
            self.assertIn(false_half, line)
            self.assertNotIn(str(actual), line)
            self.assertNotIn("guess", line.lower())
            self.assertEqual(safe_outgoing_chat(line, observation=state["observation"],
                                                action="guess 50", hidden=True), line)

    def test_blotto_tell_names_a_field_we_did_not_load(self) -> None:
        state = {
            "game_id": "colonel_blotto", "observation": "COLONEL BLOTTO - 100 soldiers, 5 fields.",
            "turn": 1, "game_index": 1,
        }
        line = draft_chat(state=state, action="0 33 33 34 0", hidden=True, family="mixed", signal="none")
        self.assertIn("F5", line)
        self.assertLessEqual(len(line), 190)

    def test_opening_and_generic_lines_stay_short_and_respectful(self) -> None:
        for hidden in (False, True):
            opening = draft_ready_chat(signal="none", hidden=hidden)
            self.assertLessEqual(len(opening), 190)
            self.assertIn("pass", opening.casefold())
            for turn in range(1, 20):
                state = {"game_id": "unknown", "turn": turn, "game_index": 1, "observation": ""}
                line = draft_chat(state=state, action="left", hidden=hidden, family="mixed", signal="none")
                self.assertTrue(line)
                self.assertLessEqual(len(line), 190)
                self.assertNotIn("password", line.lower())
                self.assertIn("pass", line)
                self.assertIn("left" if not hidden else "hidden commitment", line)
                if hidden:
                    self.assertNotIn("left", line)
                self.assertEqual(safe_outgoing_chat(line, action="left", hidden=hidden), line)

    def test_poker_pressure_does_not_echo_hidden_action(self) -> None:
        state = {"game_id": "mini_poker", "turn": 1, "game_index": 1, "observation": "Your private card: J"}
        for action in ("bet", "raise", "check", "call"):
            line = draft_chat(state=state, action=action, hidden=True, family="bluff", signal="none")
            self.assertEqual(safe_outgoing_chat(line, observation=state["observation"],
                                                action=action, hidden=True), line)

    def test_hidden_game_bait_survives_secret_screen(self) -> None:
        cases = (
            ({"game_id": "wordle_duel", "observation": "Your guesses: none", "turn": 2}, "guess crane"),
            ({"game_id": "bulls_and_cows", "observation": "Your secret code: 1597", "turn": 2,
              "variant": {"alphabet": "0123456789"}}, "guess 0123"),
            ({"game_id": "rps", "observation": "", "turn": 2}, "rock"),
        )
        for state, action in cases:
            with self.subTest(game=state["game_id"]):
                line = draft_chat(state=state, action=action, hidden=True, family="mixed", signal="none")
                self.assertTrue(line)
                self.assertLessEqual(len(line), 190)
                self.assertEqual(safe_outgoing_chat(line, observation=state["observation"],
                                                    action=action, hidden=True), line)

    def test_verified_bad_reply_is_the_prompt_bait(self) -> None:
        state = {"game_id": "dots_and_boxes", "turn": 3, "observation": ""}
        line = draft_chat(state=state, action="line b2-c2", hidden=False,
                          family="think_ahead", signal="none", bad_reply="line c2-c3")
        self.assertIn("line b2-c2", line)
        self.assertIn("line c2-c3", line)
        self.assertIn("[analysis]", line)


if __name__ == "__main__":
    unittest.main()
