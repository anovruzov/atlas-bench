#!/usr/bin/env python3
"""
Re-judge ATLAS answers with one judge (Claude Opus) for every system.

The published runs do not record which judge scored them, and their judge was
inconsistent (gold "Unknown" + answer "Unknown" was marked wrong 31/65 times).
This script re-scores every system's saved answers with the SAME judge, blind
to which system produced them.

  prepare   Collect (answer_type, gold, prediction) triples from result files,
            dedupe, shuffle (seed 0), and build each judge prompt by calling the
            repo's own metrics.llm_judge_binary with the LLM call intercepted.
            The prompt text is therefore byte-identical to the repo judge.
            Writes judge/items.jsonl and judge/batches/batch_XX.jsonl.
  api       Judge all items through the Anthropic API (reproduction path):
            ANTHROPIC_API_KEY, model ATLAS_JUDGE_MODEL (default claude-opus-5-5),
            temperature 0, max_tokens 10. Writes judge/verdicts_api.json.
  apply     Merge verdict files into judge/verdicts.json, re-score every result
            file, and write judge/RESCORED.md + judge/rescored.json.

Verdict parsing is the repo's: "YES" in reply -> 1, elif "NO" -> 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
J = ROOT / "judge"


def key(answer_type: str, gold: str, pred: str) -> str:
    return hashlib.sha256(json.dumps([answer_type, gold.strip(), pred.strip()]).encode()).hexdigest()[:16]


def build_prompt(answer_type: str, gold: str, pred: str) -> str:
    from benchmark.evaluation import metrics

    captured: List[str] = []
    orig = metrics.call_llm_judge
    metrics.call_llm_judge = lambda prompt: captured.append(prompt) or "YES"
    try:
        metrics.llm_judge_binary(pred, gold, answer_type)
    finally:
        metrics.call_llm_judge = orig
    return captured[0]


def result_files(extra: List[str]) -> List[Path]:
    files = sorted((ROOT / "results").glob("results_real_*/*.json"))
    files += [Path(e) for e in extra]
    return files


def cmd_prepare(a: argparse.Namespace) -> None:
    items: Dict[str, Dict[str, Any]] = {}
    for f in result_files(a.extra):
        d = json.loads(f.read_text())
        for b in d["detailed_results"]:
            for p in b["probe_results"]:
                k = key(p["answer_type"], p["gold_answer"], p["predicted_answer"])
                if k not in items:
                    items[k] = {
                        "id": k,
                        "answer_type": p["answer_type"],
                        "prompt": build_prompt(p["answer_type"], p["gold_answer"].strip(), p["predicted_answer"].strip()),
                    }
    done = set()
    if (J / "verdicts.json").exists():
        done = set(json.loads((J / "verdicts.json").read_text()))
    todo = [v for k, v in items.items() if k not in done]
    random.Random(0).shuffle(todo)
    (J / "batches").mkdir(parents=True, exist_ok=True)
    with open(J / "items.jsonl", "a" if done else "w") as fh:
        for it in todo:
            fh.write(json.dumps(it) + "\n")
    n = a.batches
    for i in range(n):
        chunk = todo[i::n]
        if chunk:
            (J / "batches" / f"batch_{a.prefix}{i:02d}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunk))
    print(f"{len(items)} unique items, {len(done)} already judged, {len(todo)} to judge in {n} batches")


def cmd_api(a: argparse.Namespace) -> None:
    import requests

    model = os.environ.get("ATLAS_JUDGE_MODEL", "claude-opus-5-5")
    out = {}
    for line in open(J / "items.jsonl"):
        it = json.loads(line)
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 10, "temperature": 0, "messages": [{"role": "user", "content": it["prompt"]}]},
            timeout=60,
        )
        r.raise_for_status()
        out[it["id"]] = "".join(c.get("text", "") for c in r.json()["content"]).strip()
    (J / "verdicts_api.json").write_text(json.dumps(out, indent=1))


def parse(reply: str) -> int:
    r = reply.strip().upper()
    if "YES" in r:
        return 1
    if "NO" in r:
        return 0
    raise ValueError(f"unparseable verdict: {reply!r}")


def cmd_apply(a: argparse.Namespace) -> None:
    import atlas_repro as ar

    verdicts: Dict[str, str] = {}
    if (J / "verdicts.json").exists():
        verdicts = json.loads((J / "verdicts.json").read_text())
    for vf in sorted((J / "out").glob("*.json")):
        for k, v in json.loads(vf.read_text()).items():
            verdicts.setdefault(k, v)
    (J / "verdicts.json").write_text(json.dumps(dict(sorted(verdicts.items())), indent=1))
    rows, summary = [], {}
    rescored: Dict[str, Dict] = {}
    for f in result_files(a.extra):
        d = json.loads(f.read_text())
        name = f.stem
        ts = "simple" if "simple" in name else "complex"
        old = new = n = missing = 0
        scores = {}
        flips = {"0->1": 0, "1->0": 0}
        for b in d["detailed_results"]:
            stem = re.split(r"[\\/]", b["benchmark_file"])[-1].rsplit(".", 1)[0]
            for i, p in enumerate(b["probe_results"]):
                k = key(p["answer_type"], p["gold_answer"], p["predicted_answer"])
                if k not in verdicts:
                    missing += 1
                    continue
                s = parse(verdicts[k])
                n += 1
                old += p["score"]
                new += s
                if s != p["score"]:
                    flips["0->1" if s else "1->0"] += 1
                scores[(stem, f"{i}:{p['probe_id']}")] = float(s)
        rescored[name] = {"set": ts, "scores": scores}
        summary[name] = {"set": ts, "n": n, "missing": missing, "original": old, "opus": new, "original_acc": old / max(n, 1), "opus_acc": new / max(n, 1), **flips}
        if not missing:
            _, lo, hi = ar.acc_ci(scores, ts, "all")
            summary[name]["opus_ci95"] = [lo, hi]
    lines = ["# ATLAS re-judged with one judge (Claude Opus), blind to system", "",
             "Same probes, same saved answers, the repo's own judge prompts; only the judge model changes.", "",
             "| run | set | n | original judge | Opus judge | 95% CI (file-clustered) | flips 0→1 | flips 1→0 |", "|---|---|---:|---:|---:|---|---:|---:|"]
    for name, s in sorted(summary.items(), key=lambda kv: (kv[1]["set"], -kv[1]["opus_acc"])):
        ci = s.get("opus_ci95")
        lines.append(f"| {name} | {s['set']} | {s['n']} | {s['original_acc']:.1%} | {s['opus_acc']:.1%} | {f'{ci[0]:.1%}–{ci[1]:.1%}' if ci else 'incomplete'} | {s['0->1']} | {s['1->0']} |")
    (J / "rescored.json").write_text(json.dumps(summary, indent=2))
    (J / "RESCORED.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nverdicts: {len(verdicts)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--prefix", default="")
    p.add_argument("--extra", nargs="*", default=[], help="additional result JSON files (e.g. runs/<id>/results_*.json)")
    sub.add_parser("api")
    x = sub.add_parser("apply")
    x.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    {"prepare": cmd_prepare, "api": cmd_api, "apply": cmd_apply}[a.cmd](a)


if __name__ == "__main__":
    main()
