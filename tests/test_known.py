"""Native strategy checks against released game formats."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))

from harness.known import _wordle_marks, choose_known, validate_known  # noqa: E402


def board_state(bottom: str) -> dict:
    rows = ["|. . . . . . .|" for _ in range(5)] + ["|" + " ".join(bottom) + "|"]
    return {
        "game_id": "connect_four", "seat": 0, "variant": {"rows": 6, "cols": 7, "connect": 4},
        "observation": "CONNECT 4 (6x7). You are Player 0 and play X.\n" + "\n".join(rows),
        "legal_actions": [f"drop {i}" for i in range(1, 8)],
    }


class KnownGameTests(unittest.TestCase):
    def test_connect_four_takes_immediate_win(self) -> None:
        move = choose_known(board_state("XXX.OO."))
        self.assertEqual(move["action"], "drop 4")
        self.assertTrue(move["proof"])

    def test_connect_four_blocks_immediate_loss(self) -> None:
        move = choose_known(board_state("OOO.X.."))
        self.assertEqual(move["action"], "drop 4")
        self.assertTrue(move["proof"])

    def test_wordle_syntax_checks_length_and_letters(self) -> None:
        state = {"game_id": "wordle_duel", "variant": {"length": 5}, "observation": "WORDLE DUEL (5 letters)."}
        self.assertTrue(validate_known("guess crane", state))
        self.assertFalse(validate_known("guess cranes", state))
        self.assertFalse(validate_known("guess cr4ne", state))
        self.assertFalse(validate_known("guess crane\u0130", {**state, "variant": {"length": 6}}))

    def test_blotto_syntax_checks_sum(self) -> None:
        state = {"game_id": "colonel_blotto", "observation": "You must place exactly 100 soldiers on 5 battlefields."}
        self.assertTrue(validate_known("20 20 20 20 20", state))
        self.assertTrue(validate_known("20,20,20,20,20", state))
        self.assertTrue(validate_known("20, 20, 20, 20, 20", state))
        self.assertFalse(validate_known("20,,20,20,20,20", state))
        self.assertFalse(validate_known("20 20 20 20 21", state))
        self.assertFalse(validate_known("20 20 20 40", state))

    def test_nim_uses_exact_winning_state_with_null_legal_list(self) -> None:
        state = {"game_id": "nim", "variant": {"max_piles": 1, "max_take": 0, "misere": False},
                 "observation": "NIM\nPiles now: pile 1 = 3, pile 2 = 1 (total 4)",
                 "legal_actions": None, "rejected_actions": []}
        move = choose_known(state)
        self.assertEqual(move["action"], "take 2 from pile 1")
        self.assertTrue(move["proof"])
        self.assertFalse(validate_known("take 4 from pile 1", state))

    def test_misere_nim_does_not_take_last_token(self) -> None:
        state = {"game_id": "nim", "variant": {"max_piles": 1, "max_take": 0, "misere": True},
                 "observation": "NIM\nPiles now: pile 1 = 2, pile 2 = 1 (total 3)",
                 "legal_actions": None, "rejected_actions": []}
        move = choose_known(state)
        self.assertEqual(move["action"], "take 2 from pile 1")
        self.assertTrue(move["proof"])

    def test_bulls_and_cows_filters_only_own_feedback(self) -> None:
        state = {"game_id": "bulls_and_cows", "variant": {
            "alphabet": "01", "length": 2, "repeats": True},
            "observation": ("BULLS AND COWS\nYour guesses:\n"
                            "1. 00 -> 1 bulls, 0 cows\n"
                            "2. 01 -> 0 bulls, 2 cows\n"
                            "Opponent's guesses:\n1. 11 -> 2 bulls, 0 cows"),
            "legal_actions": None, "rejected_actions": []}
        move = choose_known(state)
        self.assertEqual(move["action"], "guess 10")
        self.assertTrue(move["proof"])
        self.assertTrue(validate_known("guess 10", state))
        self.assertTrue(validate_known("10", state))
        self.assertFalse(validate_known("guess 12", state))

    def test_bulls_and_cows_rejects_repeats_when_forbidden(self) -> None:
        state = {"game_id": "bulls_and_cows", "variant": {
            "alphabet": "0123456789", "length": 4, "repeats": False}}
        self.assertTrue(validate_known("guess 0123", state))
        self.assertTrue(validate_known("0123", state))
        self.assertFalse(validate_known("guess 0012", state))
        self.assertFalse(validate_known("0012", state))

    def test_rps_exploits_repeated_rock_while_respecting_legal_list(self) -> None:
        state = {"game_id": "rps", "rules": "Rock paper scissors.",
                 "observation": "\n".join(f"Throw {n}: you scissors, opponent rock" for n in range(1, 6)),
                 "legal_actions": ["rock", "paper", "scissors"], "rejected_actions": []}
        with patch("harness.known.secrets.randbelow", return_value=99):
            self.assertEqual(choose_known(state)["action"], "paper")
        state["legal_actions"] = ["rock", "scissors"]
        with patch("harness.known.secrets.randbelow", return_value=99):
            self.assertIn(choose_known(state)["action"], state["legal_actions"])

    def test_dots_and_boxes_takes_available_box(self) -> None:
        state = {"game_id": "dots_and_boxes", "variant": {"label": "1x1 boxes"},
                 "legal_actions": ["line a1-b1"], "rejected_actions": []}
        move = choose_known(state)
        self.assertEqual(move["action"], "line a1-b1")
        self.assertEqual(move["tactic"], "capture")

    def test_dots_endgame_avoids_giving_up_a_chain(self) -> None:
        # The greedy safe-edge rule chooses c2-c3 here and loses three boxes.
        edges = ("a2-b2", "b2-c2", "c2-d2", "a3-b3", "b3-c3", "c3-d3",
                 "a4-b4", "b4-c4", "a1-a2", "b1-b2", "d1-d2", "a2-a3",
                 "b2-b3", "c2-c3", "a3-a4", "b3-b4", "d3-d4")
        state = {"game_id": "dots_and_boxes", "variant": {"label": "3x3 boxes"},
                 "legal_actions": ["line " + edge for edge in edges], "rejected_actions": []}
        move = choose_known(state)
        self.assertEqual(move["tactic"], "endgame_search")
        self.assertIn(move["action"], {"line c2-d2", "line c3-d3", "line d1-d2", "line d3-d4"})

    def test_hex_completes_a_connection(self) -> None:
        state = {"game_id": "hex", "seat": 0, "variant": {"size": 3},
                 "observation": "HEX 3x3\n 1 X . . 1\n  2 X . . 2\n   3 . . . 3",
                 "legal_actions": ["place a3", "place b2", "place c3"],
                 "rejected_actions": []}
        move = choose_known(state)
        self.assertEqual(move["action"], "place a3")
        self.assertTrue(move["proof"])

    def test_hex_swap_variant_uses_general_solver_after_swap(self) -> None:
        state = {"game_id": "hex", "seat": 1, "variant": {"size": 3, "swap": True},
                 "observation": "HEX 3x3\n 1 X . . 1\n  2 X . . 2\n   3 . . . 3",
                 "legal_actions": ["place a3", "place b2"], "rejected_actions": []}
        self.assertIsNone(choose_known(state))

    def test_battleship_hunts_adjacent_to_unsunk_hit(self) -> None:
        state = {"game_id": "battleship", "variant": {"size": 4, "salvo": False},
                 "rules": "Cruiser (3 cells), Patrol (2 cells).",
                 "observation": ("BATTLESHIP\nYOUR SHOTS\n    a b c d\n"
                                 "1  . . . .\n2  . . . .\n3  . . X .\n4  . . . .\n"
                                 "YOUR WATERS\n1 . . . ."),
                 "legal_actions": None, "rejected_actions": []}
        move = choose_known(state)
        self.assertIn(move["action"], {"fire c2", "fire b3", "fire d3", "fire c4"})
        self.assertEqual(move["tactic"], "finish_ship")
        self.assertFalse(validate_known("fire c3", state))
        self.assertTrue(validate_known(" FIRE   b 3 ", state))

    def test_wordle_candidate_respects_all_color_feedback(self) -> None:
        state = {"game_id": "wordle_duel", "variant": {"length": 5, "max_guesses": 6},
                 "turn": 3,
                 "observation": ("WORDLE DUEL\nYour guesses (G=green, Y=yellow, X=gray):\n"
                                 "  1. CRANE  G X X X X\n"
                                 "  2. PILOT  X Y Y X X\n"
                                 "Opponent's guesses: 1. ????? X X X X X"),
                 "rejected_actions": []}
        move = choose_known(state)
        word = move["action"].split()[1]
        self.assertEqual(_wordle_marks("crane", word), "GXXXX")
        self.assertEqual(_wordle_marks("pilot", word), "XYYXX")
        self.assertEqual(_wordle_marks("allee", "apple"), "GYXXG")

    def test_poker_value_bets_unbeatable_pair(self) -> None:
        state = {"game_id": "mini_poker", "variant": {"style": "leduc"},
                 "observation": "Your card: Qh\nCommunity card: Qs\nPot: 6",
                 "legal_actions": ["check", "bet"], "rejected_actions": []}
        self.assertEqual(choose_known(state)["action"], "bet")


if __name__ == "__main__":
    unittest.main()
