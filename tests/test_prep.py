"""Prep, solver validation, and artifact round-trip tests."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path


STARTER = Path(__file__).resolve().parents[1] / "starter"
sys.path.insert(0, str(STARTER))

from harness.memory import MemoryStore  # noqa: E402
from harness.prep import Specialist, SpecialistMaker, _valid_pattern  # noqa: E402
from harness.sandbox import run_function, validate_source  # noqa: E402


SOLVER = """\
def validate_action(action, state):
    legal = state.get("legal_actions")
    if legal:
        return action in legal
    return action in ("left", "right")

def choose_move(state):
    legal = state.get("legal_actions") or ["left", "right"]
    rejected = state.get("rejected_actions") or []
    for action in legal:
        if action not in rejected:
            return {"action": action, "confidence": 0.98, "chat": "Let's test that line."}
    return None
"""


def variants() -> list[dict]:
    return [{
        "label": "default",
        "rules": "Choose left or right. The exact actions are left and right.",
        "sample_turn": {
            "type": "turn", "turn": 1, "observation": "You are player 0. Choose left or right.",
            "legal_actions": ["left", "right"], "chat": [],
        },
    }]


def live_state(*, legal: list[str] | None = None) -> dict:
    return {
        "observation": "You are player 0. Choose left or right.",
        "rules": variants()[0]["rules"], "variant": {"label": "default"},
        "variant_label": "default", "legal_actions": legal,
        "seat": 0, "turn": 1, "chat": [], "opponent_profile": {},
        "rejected_actions": [], "time_budget_s": 1.0,
    }


class FakeGateway:
    def __init__(self, *, spec: dict | None = None, solver: str = SOLVER) -> None:
        self.spec = spec
        self.solver = solver
        self.spec_calls = 0
        self.solver_calls = 0

    @property
    def available(self) -> bool:
        return True

    def complete_json(self, *args: object, **kwargs: object) -> dict | None:
        self.spec_calls += 1
        return self.spec

    def complete(self, *args: object, **kwargs: object) -> str:
        self.solver_calls += 1
        return f"```python\n{self.solver}\n```"


class PrepTests(unittest.TestCase):
    def test_audit_does_not_promote_unverified_best_move_claim(self) -> None:
        game_variants = variants()
        game_variants[0]["rules"] = (
            "Choose left or right. The opponent threatens one lane; choosing the "
            "threatened lane blocks them and wins. Exact actions: left or right."
        )
        game_variants[0]["sample_turn"]["observation"] = "Opponent threatens left. Choose a block."
        spec_reply = {
            "family": "think_ahead", "hidden_information": False, "simultaneous": False,
            "forbidden_terms": [], "playbook": "Block the threatened lane.",
            "variants": [{"label": "default", "action_regex": "(?:left|right)",
                          "action_examples": ["left", "right"], "forbidden_terms": []}],
        }
        audit_reply = {
            "issues": ["The solver always chooses left, even when right is threatened."],
            "cases": [{
                "variant_label": "default", "observation": "Opponent threatens right. Choose a block.",
                "legal_actions": ["left", "right"], "best_actions": ["right"],
            }],
        }
        improved = """\
def validate_action(action, state):
    legal = state.get("legal_actions") or ["left", "right"]
    return action in legal

def choose_move(state):
    legal = state.get("legal_actions") or ["left", "right"]
    rejected = state.get("rejected_actions") or []
    wanted = "right" if "threatens right" in state.get("observation", "").lower() else "left"
    for action in [wanted] + legal:
        if action in legal and action not in rejected:
            return {"action": action, "confidence": 0.98, "chat": "Let's block that lane."}
    return None
"""
        invalid = "def choose_move(state):\n    return {'action': 'right'}\n"

        class AuditGateway(FakeGateway):
            def __init__(self, revision: str) -> None:
                super().__init__(spec=spec_reply, solver=SOLVER)
                self.revision = revision

            def complete_json(self, *args: object, **kwargs: object) -> dict:
                self.spec_calls += 1
                return spec_reply if self.spec_calls == 1 else audit_reply

            def complete(self, *args: object, **kwargs: object) -> str:
                self.solver_calls += 1
                source = SOLVER if self.solver_calls == 1 else self.revision
                return f"```python\n{source}\n```"

        for revision in (improved, invalid, SOLVER):
            with self.subTest(revision="candidate", invalid=revision == invalid):
                with tempfile.TemporaryDirectory() as directory:
                    gateway = AuditGateway(revision)
                    specialist = SpecialistMaker(gateway, MemoryStore(directory)).prepare(
                        "block", "Block", game_variants, time.monotonic() + 300
                    )
                    self.assertEqual(gateway.spec_calls, 2)
                    self.assertEqual(gateway.solver_calls, 2)
                    self.assertEqual(specialist.solver_code, SOLVER.strip())
                    self.assertIsNone(specialist.metadata.get("audit_improved_cases"))

    def test_audit_case_score_checks_legal_move_and_rejected_retry(self) -> None:
        weak = """\
def validate_action(action, state):
    return action in (state.get('legal_actions') or [])
def choose_move(state):
    rejected = state.get('rejected_actions') or []
    return {'action': 'right' if 'left' in rejected else 'left'}
"""
        cases = [{"variant_label": "default", "observation": "Choose up or down.",
                  "legal_actions": ["up", "down"], "best_actions": ["up"]}]
        deadline = time.monotonic() + 15
        self.assertEqual(SpecialistMaker._case_score(weak, cases, variants(), deadline), 0)
        self.assertEqual(SpecialistMaker._case_score(SOLVER, cases, variants(), deadline), 2)

    def test_nullable_spec_terms_builds_and_reloads_solver(self) -> None:
        spec_reply = {
            "family": "think_ahead", "hidden_information": False,
            "simultaneous": False, "forbidden_terms": None,
            "playbook": "Choose a legal action.",
            "variants": [{
                "label": "default", "action_regex": "(?:left|right)",
                "action_examples": ["left", "right"], "forbidden_terms": None,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            gateway = FakeGateway(spec=spec_reply)
            maker = SpecialistMaker(gateway, MemoryStore(directory))
            specialist = maker.prepare("choice", "Choice", variants(), time.monotonic() + 30)

            self.assertEqual(gateway.spec_calls, 1)
            self.assertEqual(gateway.solver_calls, 1)
            self.assertEqual(specialist.solver_code, SOLVER.strip())
            self.assertIsNone(validate_source(specialist.solver_code))
            self.assertEqual(specialist.action_quality("left", live_state(legal=["left", "right"])), 2)

            reloaded = maker.load("choice", "default", variants()[0]["rules"])
            self.assertIsNotNone(reloaded)
            assert reloaded is not None
            self.assertEqual(reloaded.solver_code, specialist.solver_code)
            self.assertEqual(reloaded.solve(live_state(legal=["left", "right"]), 2.0)["action"], "left")

    def test_empty_cached_solver_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = MemoryStore(directory)
            metadata = SpecialistMaker._base_metadata(variants())
            memory.save_artifact("choice", variants(), metadata, "Cached analysis.", "")
            gateway = FakeGateway()
            specialist = SpecialistMaker(gateway, memory).prepare(
                "choice", "Choice", variants(), time.monotonic() + 30
            )
            self.assertEqual(gateway.spec_calls, 0)
            self.assertEqual(gateway.solver_calls, 1)
            self.assertTrue(specialist.solver_code)

    def test_validator_takes_precedence_over_advisory_regex(self) -> None:
        metadata = {"family": "mixed", "variants": {"default": {
            "action_regex": "left", "action_examples": ["left", "right"],
        }}}
        specialist = Specialist("choice", metadata, solver_code=SOLVER)
        self.assertEqual(specialist.action_quality("right", live_state(), strict=True), 2)
        self.assertEqual(specialist.action_quality("left", live_state(), strict=True), 2)

        rejecting_solver = """\
def validate_action(action, state):
    return action == "right"
def choose_move(state):
    return {"action": "right", "confidence": 1.0, "chat": "Ready."}
"""
        specialist.solver_code = rejecting_solver
        self.assertEqual(specialist.action_quality("left", live_state(), strict=True), 0)

    def test_null_list_sample_probe_rejects_solver_that_returns_bogus(self) -> None:
        bad = """\
def validate_action(action, state):
    legal = state.get('legal_actions')
    return action in legal if legal else action in ('left', 'right', 'bogus')
def choose_move(state):
    legal = state.get('legal_actions')
    if not legal:
        return {'action': 'bogus'}
    rejected = state.get('rejected_actions') or []
    return {'action': next(action for action in legal if action not in rejected)}
"""
        error = SpecialistMaker._validate_solver(
            bad, variants(), SpecialistMaker._base_metadata(variants()), time.monotonic() + 15,
        )
        self.assertIn("null-list choose_move", error)

    def test_between_numeric_bounds_reject_out_of_range_action(self) -> None:
        spec = Specialist("number", {"family": "guessing", "variants": {}})
        state = {"rules": "Guess a number between 1 and 100.", "observation": "",
                 "legal_actions": None, "rejected_actions": []}
        self.assertEqual(spec.action_quality("guess 999", state), 0)
        self.assertEqual(spec.action_quality("guess 37", state), 2)

    def test_pathological_regex_and_unsafe_source_are_rejected(self) -> None:
        self.assertEqual(_valid_pattern("(a|aa)+", ["aaaa"]), "")
        self.assertEqual(_valid_pattern("(a+)+", ["aaaa"]), "")
        self.assertIn("not allowed", validate_source("import os\ndef choose_move(state): return os.getcwd()"))
        self.assertIn("not allowed", validate_source("def choose_move(state): return state.__class__"))
        result, error = run_function("def choose_move(state): return open('secret')", "choose_move", ({},), 1.0)
        self.assertIsNone(result)
        self.assertIn("not allowed", error or "")


if __name__ == "__main__":
    unittest.main()
