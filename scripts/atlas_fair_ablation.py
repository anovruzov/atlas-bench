#!/usr/bin/env python3
"""Matched-model ATLAS ablations. New runs only; never rewrites published scores.

All arms get the same answer model, the same Answerer prompt, and a fresh
per-question API call. A different judge model is optional; without it, report
strict normalized-exact accuracy only (not semantic accuracy).

The optional unrelated-world expansion is a noise/stress diagnostic, NOT an
independently collected longitudinal benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.evaluation.mycelic_loop import (  # noqa: E402
    ANSWERER_SYSTEM, VERIFIER_SYSTEM, MycelicLoopAdapter, _extract_json
)

ARMS = (
    "full_context",
    "full_context_verify",
    "retrieve_answer",
    "retrieve_verify",
    "retrieve_verify_retry",
)
JUDGE_SYSTEM = """You are an independent, blind evaluator. Score the candidate answer
against the reference for the given question, without knowing which experimental
arm generated it. Accept factually equivalent paraphrases. Reject unsupported
extra alternatives, contradictions, wrong entities, and wrong times. If the
reference is Unknown, ONLY a clear abstention is correct. Return exactly JSON:
{"correct": true or false, "reason": "brief explanation"}"""


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value).casefold())).strip()


def is_unknown(value: str) -> bool:
    return normalize(value) in {"unknown", "not known", "information unavailable", "not provided"}


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Model:
    def __init__(self, *, base_url: str, model: str, key: str, max_tokens: int):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = key
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str) -> tuple[str, dict[str, Any]]:
        start = time.monotonic()
        res = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            },
            timeout=240,
        )
        res.raise_for_status()   # no silent model/provider fallback
        data = res.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No model choices returned for {self.model}")
        answer = choices[0]["message"].get("content") or ""
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(f"Empty completion from {self.model}")
        return answer, {
            "model_requested": self.model,
            "model_reported": data.get("model"),
            "latency_s": round(time.monotonic() - start, 4),
            "input_chars": len(system) + len(prompt),
            "output_chars": len(answer),
            "usage": data.get("usage") or {},
        }


def call_answer(model: Model, question: str, evidence: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, metrics = model.complete(
        ANSWERER_SYSTEM, f"Conversations:\n{evidence}\n\nQuestion: {question}"
    )
    parsed = _extract_json(raw)
    # Malformed answers are never silently promoted to "Unknown".
    if not isinstance(parsed, dict) or not str(parsed.get("answer", "")).strip():
        return {"answer": "[INVALID ANSWER FORMAT]", "sessions": [], "quote": ""}, metrics
    return parsed, metrics


def call_verifier(model: Model, question: str, evidence: str,
                  draft: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, metrics = model.complete(
        VERIFIER_SYSTEM,
        f"Conversations:\n{evidence}\n\nQuestion: {question}\n\n"
        f"Draft answer: {draft.get('answer', '')}\n"
        f"Cited sessions: {draft.get('sessions', [])}\n"
        f"Cited quote: {draft.get('quote', '')}"
    )
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict) or parsed.get("verdict") not in {"accept", "revise", "retrieve"}:
        return {"verdict": "invalid", "final_answer": "[INVALID VERIFIER FORMAT]",
                "search_query": ""}, metrics
    return parsed, metrics


def run_arm(retriever: MycelicLoopAdapter, question: str, arm: str,
            answer_model: Model) -> dict[str, Any]:
    wall_start = time.monotonic()
    full = arm.startswith("full_context")
    verify = arm in {"full_context_verify", "retrieve_verify", "retrieve_verify_retry"}
    retry = arm == "retrieve_verify_retry"
    queries = [question]
    shown: list[str] = []
    calls: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    answer = "[NO ANSWER]"
    max_rounds = 2 if retry else 1
    for turn in range(max_rounds):
        chosen = (list(retriever.sessions) if full
                  else retriever._retrieve_sessions(queries))
        for sid in chosen:
            if sid not in shown:
                shown.append(sid)
        evidence = retriever._render(shown)
        draft, meta = call_answer(answer_model, question, evidence)
        calls.append({"role": "answerer", **meta})
        answer = str(draft["answer"]).strip()
        step: dict[str, Any] = {
            "round": turn + 1, "queries": list(queries),
            "sessions_shown": list(shown), "evidence_chars": len(evidence),
            "draft": answer,
        }
        if verify:
            verdict, meta = call_verifier(answer_model, question, evidence, draft)
            calls.append({"role": "verifier", **meta})
            answer = str(verdict.get("final_answer") or "[INVALID VERIFIER FORMAT]").strip()
            step["verdict"] = verdict.get("verdict")
            new_query = str(verdict.get("search_query") or "").strip()
            rounds.append(step)
            if retry and turn == 0 and verdict["verdict"] == "retrieve" and \
                    new_query and new_query not in queries:
                queries.append(new_query)
                continue
        else:
            rounds.append(step)
        break
    return {"prediction": answer, "rounds": rounds, "calls": calls,
            "end_to_end_latency_s": round(time.monotonic() - wall_start, 4),
            "model_latency_s": round(sum(c["latency_s"] for c in calls), 4),
            "total_input_chars": sum(c["input_chars"] for c in calls),
            "total_output_chars": sum(c["output_chars"] for c in calls),
            "usage": {
                field: sum(int(c["usage"].get(field) or 0) for c in calls)
                for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            }}


def judge_prediction(judge: Model | None, question: str, gold: str,
                     prediction: str) -> tuple[bool | None, dict[str, Any] | None]:
    # Always include Unknowns in the denominator. This avoids the original
    # benchmark's historical Unknown/Unknown LLM-judge inconsistency.
    if is_unknown(gold):
        return is_unknown(prediction), {"method": "deterministic_abstention"}
    if normalize(gold) == normalize(prediction):
        return True, {"method": "normalized_exact"}
    if judge is None:
        return None, None
    raw, meta = judge.complete(
        JUDGE_SYSTEM,
        f"Question: {question}\nReference answer: {gold}\n"
        f"Candidate answer: {prediction}\nReturn JSON only."
    )
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("correct"), bool):
        raise RuntimeError(f"Invalid judge JSON: {raw[:200]}")
    return parsed["correct"], {"method": "independent_model",
                                "reason": parsed.get("reason"), **meta}


def expanded_sessions(target: dict[str, Any], other_worlds: list[dict[str, Any]],
                      n: int) -> list[dict[str, Any]]:
    """Append clearly labeled unrelated histories; gold answers stay untouched."""
    sessions = [dict(s) for s in target["sessions"]]
    if not n:
        return sessions
    for j, world in enumerate(other_worlds[:n]):
        for k, s in enumerate(world["sessions"]):
            copy = dict(s)
            copy["id"] = f"distractor-world-{j}-session-{k}"
            copy["timestamp"] = -10000000 + j * 1000 + k
            copy["turns"] = [
                {**t, "text": f"[UNRELATED ARCHIVED WORLD {j}; NOT THE TARGET USER] "
                              + str(t.get("text", ""))}
                for t in s.get("turns", [])
            ]
            sessions.append(copy)
    return sessions


def bootstrap_diff(rows: list[dict[str, Any]], left: str, right: str,
                   seed: int) -> dict[str, Any]:
    by_world: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        if left in item["results"] and right in item["results"]:
            a = item["results"][left].get("semantic_correct")
            b = item["results"][right].get("semantic_correct")
            if a is not None and b is not None:
                by_world[item["world"]].append(float(a) - float(b))
    if not by_world:
        return {"error": "No paired independently judged results"}
    all_vals = [x for group in by_world.values() for x in group]
    result = {"mean_diff": sum(all_vals) / len(all_vals),
              "n_probes": len(all_vals), "n_worlds": len(by_world)}
    if len(by_world) < 5:
        result["note"] = "Exploratory: fewer than five independent worlds; CI withheld"
        return result
    rng = random.Random(seed)
    worlds = sorted(by_world)
    estimates = []
    for _ in range(2000):
        draw = [by_world[rng.choice(worlds)] for _ in worlds]
        vals = [x for group in draw for x in group]
        estimates.append(sum(vals) / len(vals))
    estimates.sort()
    result["ci95_world_bootstrap"] = [estimates[50], estimates[1949]]
    return result


def run(args: argparse.Namespace) -> None:
    if args.judge_model and args.judge_model == args.answer_model:
        raise SystemExit("Judge model must differ from answer model.")
    if args.max_worlds < 0 or args.max_probes < 0 or args.distractor_worlds < 0:
        raise SystemExit("Counts cannot be negative.")
    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"Set {args.key_env} for the model endpoint. No calls made.")
    answer_model = Model(base_url=args.base_url, model=args.answer_model,
                         key=key, max_tokens=args.max_tokens)
    judge = (Model(base_url=args.base_url, model=args.judge_model, key=key,
                   max_tokens=180) if args.judge_model else None)
    world_dirs = {"simple": ROOT / "simple_test_set",
                  "complex": ROOT / "complex_test_set"}
    selected_sets = list(world_dirs) if args.test_set == "both" else [args.test_set]
    output = ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "answer_model": args.answer_model, "judge_model": args.judge_model,
        "base_url": args.base_url, "arms": ARMS,
        "scoring": "normalized exact plus independent judge when configured",
        "split": args.split, "max_worlds": args.max_worlds,
        "max_probes": args.max_probes,
        "distractor_worlds": args.distractor_worlds, "seed": args.seed,
        "source_hashes": {},
        "note": "Stress expansion uses labeled unrelated ATLAS worlds, not real longitudinal histories.",
    }
    rows: list[dict[str, Any]] = []
    out_path = output / "predictions.jsonl"
    with out_path.open("w") as sink:
        for test_set in selected_sets:
            files = sorted(world_dirs[test_set].glob("*.json"))
            selection = (files[:10] if args.split == "dev" else
                         files[10:] if args.split == "heldout" else files)
            if args.max_worlds:
                selection = selection[:args.max_worlds]
            if not selection:
                raise RuntimeError(f"No worlds in {test_set}/{args.split}")
            for path in selection:
                world = json.loads(path.read_text())
                distractors: list[dict[str, Any]] = []
                if args.distractor_worlds:
                    pool = [p for p in files if p != path]
                    for p in pool[:args.distractor_worlds]:
                        distractors.append(json.loads(p.read_text()))
                        manifest["source_hashes"][str(p.relative_to(ROOT))] = hash_file(p)
                sessions = expanded_sessions(world, distractors, args.distractor_worlds)
                manifest["source_hashes"][str(path.relative_to(ROOT))] = hash_file(path)
                retriever = MycelicLoopAdapter()
                retriever.ingest_sessions(sessions)
                probes = world["probes"]
                if args.max_probes:
                    probes = probes[:args.max_probes]
                for index, probe in enumerate(probes):
                    question = str(probe["question"])
                    gold = str(probe["gold_answer"]["text"])
                    answers: dict[str, Any] = {}
                    ordering = list(ARMS)
                    random.Random(args.seed + index).shuffle(ordering)
                    for arm in ordering:
                        answer = run_arm(retriever, question, arm, answer_model)
                        exact = normalize(gold) == normalize(answer["prediction"])
                        semantic, judge_meta = (
                            judge_prediction(judge, question, gold, answer["prediction"])
                            if judge is not None else (None, None)
                        )
                        answer["strict_exact_correct"] = exact
                        answer["semantic_correct"] = semantic
                        answer["judge"] = judge_meta
                        answers[arm] = answer
                    row = {
                        "set": test_set, "world": path.stem,
                        "probe_id": probe.get("id"), "pillar": probe.get("pillar"),
                        "answer_type": probe.get("answer_type"),
                        "question": question, "gold": gold,
                        "unknown_gold": is_unknown(gold),
                        "original_sessions": len(world["sessions"]),
                        "total_sessions": len(sessions), "results": answers,
                    }
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    rows.append(row)
                    print(f"{test_set}/{path.stem} probe {index+1}/{len(probes)}",
                          flush=True)
    summary: dict[str, Any] = {
        "n_probes": len(rows), "n_worlds": len(set((r["set"], r["world"]) for r in rows)),
        "n_unknown_probes": sum(r["unknown_gold"] for r in rows), "arms": {},
        "paired_vs_full_context": {},
    }
    for arm in ARMS:
        sub = [r["results"][arm] for r in rows]
        valid = [r["semantic_correct"] for r in sub if r["semantic_correct"] is not None]
        summary["arms"][arm] = {
            "strict_exact_accuracy": sum(r["strict_exact_correct"] for r in sub) / len(sub),
            "semantic_accuracy": sum(valid) / len(valid) if len(valid) == len(sub) else None,
            "semantic_scored": len(valid),
            "mean_end_to_end_latency_s": statistics.mean(r["end_to_end_latency_s"] for r in sub),
            "median_end_to_end_latency_s": statistics.median(r["end_to_end_latency_s"] for r in sub),
            "p95_end_to_end_latency_s": sorted(r["end_to_end_latency_s"] for r in sub)[max(0, int(0.95 * len(sub)) - 1)],
            "mean_model_latency_s": statistics.mean(r["model_latency_s"] for r in sub),
            "mean_input_chars": statistics.mean(r["total_input_chars"] for r in sub),
            "usage_totals": {
                k: sum(r["usage"][k] for r in sub)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
        }
    for arm in ARMS:
        if arm != "full_context":
            summary["paired_vs_full_context"][arm] = bootstrap_diff(
                rows, arm, "full_context", args.seed
            )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(output), **summary}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answer-model", required=True,
                   help="Explicit OpenAI-compatible model ID; same for every arm")
    p.add_argument("--judge-model", default=None,
                   help="Different model; omit for strict-exact scoring only")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--test-set", choices=["simple", "complex", "both"], default="both")
    p.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    p.add_argument("--max-worlds", type=int, default=1,
                   help="0 means every world in selected split")
    p.add_argument("--max-probes", type=int, default=2,
                   help="0 means every probe")
    p.add_argument("--distractor-worlds", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=datetime.now(timezone.utc).strftime(
        "fair_ablation_%Y%m%dT%H%M%SZ"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
