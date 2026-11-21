"""Offline tests for truthful token/cost/latency scorecards."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from atlas_efficiency import percentile, summarize_rows, summarize_directory


def row(score, total, prompt, cached, dollars, duration):
    return {
        "scores": {"mycelic_loop": score, "full_context": 1},
        "resources": {
            "mycelic_loop": {
                "end_to_end_ms": duration, "search_calls": 2,
                "model_calls": [{
                    "prompt_tokens": prompt, "completion_tokens": total - prompt,
                    "total_tokens": total, "cached_prompt_tokens": cached,
                    "cache_write_tokens": 0, "cost_usd": dollars,
                }],
            },
            "full_context": {
                "end_to_end_ms": duration / 2, "search_calls": 0,
                "model_calls": [{
                    "prompt_tokens": 500, "completion_tokens": 10,
                    "total_tokens": 510, "cached_prompt_tokens": 0,
                    "cache_write_tokens": 0, "cost_usd": 0.025,
                }],
            },
        },
    }


class EfficiencyTests(unittest.TestCase):
    def test_interpolated_percentiles(self):
        self.assertEqual(percentile([10, 20, 30], 0.5), 20)
        self.assertEqual(percentile([], 0.95), None)

    def test_tokens_cost_and_latency_are_aggregated(self):
        data = summarize_rows([
            row(1, 100, 80, 40, 0.01, 40),
            row(0, 140, 100, 30, 0.02, 60),
        ])
        arm = data["mycelic_loop"]
        self.assertEqual(arm["n_correct"], 1)
        self.assertEqual(arm["usage"]["total_tokens"], 240)
        self.assertEqual(arm["usage"]["cached_prompt_tokens"], 70)
        self.assertAlmostEqual(arm["cost_usd"], 0.03)
        self.assertEqual(arm["latency_ms"]["p50"], 50)
        self.assertEqual(arm["latency_ms"]["p95"], 59)
        self.assertEqual(arm["model_calls"], 2)
        self.assertAlmostEqual(arm["total_token_reduction_vs_full_context_pct"], 76.471, places=2)

    def test_missing_provider_cost_and_cache_are_not_zero(self):
        probe = row(1, 100, 80, None, None, 70)
        result = summarize_rows([probe])["mycelic_loop"]
        self.assertIsNone(result["cost_usd"])
        self.assertIsNone(result["usage"]["cached_prompt_tokens"])
        self.assertIsNone(result["cached_share_of_input"])
        self.assertIsNone(result["usd_per_correct"])
        self.assertEqual(result["missing_usage_model_calls"]["cost_usd"], 1)

    def test_reads_saved_worlds_without_inventing_other_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "simple_world.json").write_text(json.dumps([row(1, 40, 30, 0, 0.004, 10)]))
            report = summarize_directory(folder)
        self.assertEqual(report["sets"]["simple"]["mycelic_loop"]["n_probes"], 1)
        self.assertEqual(report["sets"]["complex"]["status"], "no completed world artifacts")


if __name__ == "__main__":
    unittest.main()
