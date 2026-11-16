#!/usr/bin/env python3
"""Dependency-light checks for the fair ablation runner; no paid model calls."""
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

# Test the control flow without downloading embedding models or hitting APIs.
requests_stub = types.ModuleType("requests")
requests_stub.post = lambda *a, **kw: (_ for _ in ()).throw(
    RuntimeError("Network use forbidden in unit tests")
)
sys.modules.setdefault("requests", requests_stub)
pkg = types.ModuleType("benchmark")
pkg.__path__ = []
subpkg = types.ModuleType("benchmark.evaluation")
subpkg.__path__ = []
loop = types.ModuleType("benchmark.evaluation.mycelic_loop")
loop.ANSWERER_SYSTEM = "ANSWERER"
loop.VERIFIER_SYSTEM = "VERIFIER"
loop.MycelicLoopAdapter = object
loop._extract_json = json.loads
sys.modules["benchmark"] = pkg
sys.modules["benchmark.evaluation"] = subpkg
sys.modules["benchmark.evaluation.mycelic_loop"] = loop

path = Path(__file__).resolve().parent / "atlas_fair_ablation.py"
spec = importlib.util.spec_from_file_location("atlas_fair_ablation", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeRetriever:
    sessions = {"s1": {"t": 1}, "s2": {"t": 2}}

    def _retrieve_sessions(self, queries):
        return ["s2"] if len(queries) > 1 else ["s1"]

    def _render(self, session_ids):
        return "|".join(session_ids)


class FakeModel:
    def __init__(self):
        self.calls = []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        if system == "ANSWERER":
            raw = '{"answer":"Paris","sessions":["s1"],"quote":"Paris"}'
        elif system == "VERIFIER":
            raw = '{"verdict":"retrieve","final_answer":"Paris","search_query":"followup"}'
        else:
            raise AssertionError("Unexpected system prompt")
        return raw, {"latency_s": 0.01, "input_chars": len(prompt),
                     "output_chars": len(raw), "usage": {}}


class FairStudyTests(unittest.TestCase):
    def test_unknown_is_included_and_requires_abstention(self):
        self.assertEqual(mod.judge_prediction(None, "q", "Unknown", "Unknown")[0], True)
        self.assertEqual(mod.judge_prediction(None, "q", "Unknown", "Paris")[0], False)
        self.assertTrue(mod.is_unknown("Not known."))
        self.assertEqual(mod.normalize("The  Answer."), "the answer")

    def test_five_arms_share_one_answerer_prompt(self):
        counts = {
            "full_context": 1, "full_context_verify": 2,
            "retrieve_answer": 1, "retrieve_verify": 2,
            "retrieve_verify_retry": 4,
        }
        for arm, count in counts.items():
            with self.subTest(arm=arm):
                model = FakeModel()
                result = mod.run_arm(FakeRetriever(), "What city?", arm, model)
                self.assertEqual(len(model.calls), count)
                self.assertEqual(model.calls[0][0], "ANSWERER")
                self.assertEqual(result["prediction"], "Paris")
                self.assertGreaterEqual(result["end_to_end_latency_s"], 0)
                if arm == "full_context":
                    self.assertEqual(result["rounds"][0]["sessions_shown"], ["s1", "s2"])
                if arm == "retrieve_verify_retry":
                    self.assertEqual(len(result["rounds"]), 2)

    def test_stress_worlds_are_explicitly_unrelated(self):
        target = {"sessions": [{"id": "target", "timestamp": 4,
                                "turns": [{"speaker": "user", "text": "I live in Paris"}]}]}
        distractor = {"sessions": [{"id": "other", "timestamp": 4,
                                    "turns": [{"speaker": "user", "text": "I live in Berlin"}]}]}
        out = mod.expanded_sessions(target, [distractor], 1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["id"], "target")
        self.assertTrue(out[1]["id"].startswith("distractor-world-"))
        self.assertIn("UNRELATED ARCHIVED WORLD", out[1]["turns"][0]["text"])
        self.assertEqual(target["sessions"][0]["turns"][0]["text"], "I live in Paris")

    def test_bootstrap_with_small_sample_does_not_claim_confidence(self):
        data = [{"world": "w1", "results": {
            "retrieve_answer": {"semantic_correct": True},
            "full_context": {"semantic_correct": False},
        }}]
        r = mod.bootstrap_diff(data, "retrieve_answer", "full_context", 0)
        self.assertEqual(r["mean_diff"], 1.0)
        self.assertNotIn("ci95_world_bootstrap", r)


if __name__ == "__main__":
    unittest.main()
