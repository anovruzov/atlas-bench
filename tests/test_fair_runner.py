"""Offline, credential-free structural tests of the prospective ATLAS live runner."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import atlas_head_to_head as live


class FakeMycelic:
    def __init__(self):
        self.traces = []
    def ingest_sessions(self, sessions):
        self.sessions = sessions
    def generate_answer(self, question, unused_context):
        self.traces.append({"rounds": [{"queries": [question], "sessions_shown": ["s1"]}]})
        return "component A"


class FakeNebula:
    def collection(self, test_set, world, sessions):
        return "test-collection"
    def search(self, collection, question):
        return [{"summary": "Component A was replaced", "excerpt": "confirmed by maintenance"}]


class FairRunnerTests(unittest.TestCase):
    def test_completion_logs_usage_without_token_fabrication(self):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 4, "total_tokens": 25},
        }
        http = Mock()
        http.post.return_value = response
        live.MODEL_CALL_EVENTS.clear()
        result = live.completion(http, "model-A", "provider-A", "system", "question", 50)
        self.assertEqual(result, "answer")
        self.assertEqual(live.MODEL_CALL_EVENTS[-1]["total_tokens"], 25)
        self.assertGreaterEqual(live.MODEL_CALL_EVENTS[-1]["elapsed_ms"], 0)

    def test_no_usage_is_reported_as_missing(self):
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]
        }
        http = Mock()
        http.post.return_value = response
        live.MODEL_CALL_EVENTS.clear()
        live.completion(http, "model-A", "provider-A", "", "question", 50)
        self.assertIsNone(live.MODEL_CALL_EVENTS[-1]["total_tokens"])

    def test_world_contains_three_paired_arms_and_resource_traces(self):
        world = {
            "sessions": [
                {"id": "s1", "timestamp": 1, "turns": [
                    {"speaker": "user", "text": "Component A was replaced."}]}
            ],
            "probes": [
                {"id": "p1", "question": "Which component?", "answer_type": "short_answer",
                 "gold_answer": {"text": "component A"}}
            ],
        }
        def fake_answer(system, prompt):
            live.MODEL_CALL_EVENTS.append({
                "model": "fixed-model", "provider": "fixed-provider",
                "elapsed_ms": 1.0, "prompt_tokens": 30,
                "completion_tokens": 4, "total_tokens": 34,
            })
            return "component A"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "world.json"
            path.write_text(json.dumps(world))
            rows = live.world_answers(FakeMycelic(), FakeNebula(), fake_answer, "simple", path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["answers"]), {"mycelic_loop", "nebula", "full_context"})
        self.assertEqual(rows[0]["resources"]["nebula"]["search_calls"], 1)
        self.assertEqual(rows[0]["resources"]["full_context"]["model_call_count"], 1)
        self.assertEqual(rows[0]["resources"]["mycelic_loop"]["search_calls"], 1)


if __name__ == "__main__":
    unittest.main()
