"""ATLAS efficiency reporting, with explicit missing-data semantics.

The primary latency is serial per-query wall time (retrieval + all answer/verifier
calls, excluding one-time indexing). When provider usage metadata is missing,
token or cost totals are null rather than guessed. No historical benchmark
results are modified by this module.
"""
from __future__ import annotations
import math
from pathlib import Path
import json

FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens",
          "cached_prompt_tokens", "cache_write_tokens", "cost_usd")


def percentile(values, p):
    if not values:
        return None
    values = sorted(float(v) for v in values)
    pos = (len(values) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    return round(values[low] + (values[high] - values[low]) * (pos - low), 3)


def total_if_complete(events, key):
    values = [e.get(key) for e in events]
    if not values or any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
        return None
    return round(sum(values), 7) if key == "cost_usd" else int(sum(values))


def summarize_rows(rows):
    arms = sorted({a for row in rows for a in row.get("resources", {})})
    report = {}
    for arm in arms:
        selected = [row for row in rows if arm in row.get("resources", {})]
        durations = [row["resources"][arm].get("end_to_end_ms") for row in selected]
        durations = [v for v in durations if isinstance(v, (int, float))]
        all_calls = [
            c for row in selected for c in row["resources"][arm].get("model_calls", [])
        ]
        measured = {k: total_if_complete(all_calls, k) for k in FIELDS}
        correct = sum(row.get("scores", {}).get(arm) == 1 for row in selected)
        calls = sum(len(row["resources"][arm].get("model_calls", [])) for row in selected)
        prompts = measured["prompt_tokens"]
        total = measured["total_tokens"]
        dollars = measured["cost_usd"]
        report[arm] = {
            "n_probes": len(selected),
            "n_correct": correct,
            "accuracy": round(correct / len(selected), 6) if selected else None,
            "latency_ms": {
                "p50": percentile(durations, 0.50),
                "p95": percentile(durations, 0.95),
                "measured_probes": len(durations),
                "definition": "serial retrieval + model calls, excludes once-per-world ingestion and indexing",
            },
            "model_calls": calls,
            "mean_model_calls_per_probe": round(calls / len(selected), 4) if selected else None,
            "search_calls": sum(row["resources"][arm].get("search_calls", 0) for row in selected),
            "usage": measured,
            "cached_share_of_input": round(measured["cached_prompt_tokens"] / prompts, 6)
                if prompts and measured["cached_prompt_tokens"] is not None else None,
            "correct_per_million_total_tokens": round(correct / total * 1000000, 5) if total else None,
            "usd_per_correct": round(dollars / correct, 7) if dollars is not None and correct else None,
            "cost_usd": dollars,
            "missing_usage_model_calls": {
                key: sum(e.get(key) is None for e in all_calls) for key in FIELDS
            },
        }
    base = report.get("full_context", {}).get("usage", {})
    base_tokens = base.get("total_tokens")
    base_cost = base.get("cost_usd")
    for arm, entry in report.items():
        value = entry["usage"]["total_tokens"]
        price = entry["usage"]["cost_usd"]
        entry["total_token_reduction_vs_full_context_pct"] = (
            round((1 - value / base_tokens) * 100, 3)
            if arm != "full_context" and value is not None and base_tokens else None
        )
        entry["inference_cost_reduction_vs_full_context_pct"] = (
            round((1 - price / base_cost) * 100, 3)
            if arm != "full_context" and price is not None and base_cost else None
        )
    return report


def summarize_directory(directory: Path, sets=("simple", "complex")):
    result = {}
    for name in sets:
        rows = []
        for p in sorted(directory.glob(name + "_*.json")):
            doc = json.loads(p.read_text())
            if isinstance(doc, list):
                rows.extend(doc)
        result[name] = summarize_rows(rows) if rows else {"status": "no completed world artifacts"}
    return {
        "schema_version": 1,
        "status": "measured only for saved completed worlds; no projection or fabricated numbers",
        "sets": result,
        "interpretation": (
            "Prompt tokens include cached tokens when the provider counts them that way. "
            "Cached-token values are null if absent, not zero. Provider-reported USD costs "
            "are null when unreported. Indexing/ingestion time is excluded from per-query "
            "latency and must be reported separately. Sequential execution means figures "
            "describe this runner, not unconstrained production p95."
        ),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = summarize_directory(args.run_directory)
    body = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(body)
    print(body)
