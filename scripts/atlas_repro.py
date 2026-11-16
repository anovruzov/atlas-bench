#!/usr/bin/env python3
"""
Reproducible ATLAS runs and honest comparison against the published results.

  run         Evaluate adapters with the repo's own evaluator and judge, and
              write results plus a manifest (git SHA, data hashes, models,
              prompts, package versions, config) to runs/<run_id>/.
  compare     Paired comparison of a run against the published results in
              results/ on identical probes, with a calibration gate.
  retrieval   Key-free diagnostic: gold-session recall@k of the Mycelic
              retriever vs the naive_rag retrieval baseline (no LLM calls).

Rules this script enforces:
  * Files are split by sorted name: the first 10 files of each set are DEV
    (allowed for prompt/config development), the remaining 40 are HELD-OUT.
    Both are always reported separately.
  * A "beats" verdict is printed only if (a) the run's naive_rag reproduces the
    published naive_rag within its 95% CI (same answer model + judge setup),
    and (b) the lower bound of the paired 95% CI of (ours - published) is > 0
    on the held-out files. Otherwise the verdict says why not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SETS = {"simple": ROOT / "simple_test_set", "complex": ROOT / "complex_test_set"}
PUBLISHED = {"simple": ROOT / "results" / "results_real_simple", "complex": ROOT / "results" / "results_real_complex"}
DEV_FILES = 10
BOOT = 10_000
SEED = 0


# ----------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> Dict[str, Any]:
    def g(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"sha": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def files_for(test_set: str) -> List[Path]:
    return sorted(SETS[test_set].glob("*.json"))


def split_of(stem: str, test_set: str) -> str:
    names = [p.stem for p in files_for(test_set)]
    return "dev" if names.index(stem) < DEV_FILES else "heldout"


def probe_scores(report: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    out = {}
    for b in report["detailed_results"]:
        # Published runs were made on Windows; split on both separators.
        stem = re.split(r"[\\/]", b["benchmark_file"])[-1].rsplit(".", 1)[0]
        for i, p in enumerate(b["probe_results"]):
            out[(stem, f"{i}:{p['probe_id']}")] = float(p["score"])
    return out


def cluster_boot_ci(diffs_by_file: Dict[str, List[float]], seed: int = SEED) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    files = list(diffs_by_file)
    tot = sum(sum(v) for v in diffs_by_file.values())
    n = sum(len(v) for v in diffs_by_file.values())
    stats = []
    for _ in range(BOOT):
        s = c = 0.0
        for _ in files:
            v = diffs_by_file[files[rng.randrange(len(files))]]
            s += sum(v)
            c += len(v)
        stats.append(s / c)
    stats.sort()
    return tot / n, stats[int(0.025 * BOOT)], stats[int(0.975 * BOOT) - 1]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def published_report(test_set: str, adapter: str) -> Dict[str, Any]:
    hits = sorted(PUBLISHED[test_set].glob(f"results_{adapter}_*.json"))
    if not hits:
        raise FileNotFoundError(f"no published {adapter} run for {test_set}")
    return json.loads(hits[-1].read_text())


# --------------------------------------------------------------------- run
def judge_route() -> str:
    """Best-effort description of which judge path metrics.py will take."""
    from benchmark.evaluation import metrics

    if metrics.GOOGLE_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
        return f"google:{metrics.GEMINI_MODEL} (falls back to OpenRouter on error)"
    if os.environ.get("OPENROUTER_API_KEY"):
        return f"openrouter:{metrics.OPENROUTER_MODEL}"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    return "none (heuristic fallback)"


def cmd_run(a: argparse.Namespace) -> None:
    from benchmark.evaluation.evaluator import BenchmarkEvaluator
    from benchmark.evaluation.memory_adapters import create_adapter, OPENROUTER_MODEL, GEMINI_MODEL
    from benchmark.evaluation import mycelic_loop

    run_id = a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    pkgs = {}
    for name in ["openai", "requests", "google-genai", "rank_bm25", "fastembed", "chromadb", "sentence-transformers", "numpy"]:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    manifest = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": pkgs,
        "llm_judge_enabled": not a.no_llm_eval,
        "judge_route": judge_route() if not a.no_llm_eval else "heuristic",
        "baseline_answer_model": f"repo default: openrouter {OPENROUTER_MODEL} (gemini {GEMINI_MODEL} if Google creds present)",
        "mycelic_loop": {
            "backend": os.environ.get("ATLAS_LLM_BACKEND", "openrouter"),
            "model": os.environ.get("ATLAS_LLM_MODEL", OPENROUTER_MODEL),
            "config": mycelic_loop.DEFAULT_CONFIG,
            "prompts_sha256": hashlib.sha256((mycelic_loop.ANSWERER_SYSTEM + mycelic_loop.VERIFIER_SYSTEM).encode()).hexdigest(),
        },
        "env_present": {k: bool(os.environ.get(k)) for k in ["OPENROUTER_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]},
        "data": {},
        "runs": {},
    }
    for ts in a.test_sets:
        files = files_for(ts)[: a.max_files] if a.max_files else files_for(ts)
        manifest["data"][ts] = {p.name: sha256(p) for p in files}
        for ad in a.adapters:
            adapter = create_adapter(ad, user_id=f"repro_{ad}_{run_id}")
            ev = BenchmarkEvaluator(adapter=adapter, verbose=not a.quiet, use_llm_eval=not a.no_llm_eval, search_top_k=10)
            results = []
            for p in files:
                adapter.user_id = p.stem
                results.append(ev.evaluate_benchmark(p))
            path = out / f"results_{ad}_{ts}.json"
            rep = BenchmarkEvaluator.generate_report(results, path)
            if hasattr(adapter, "traces"):
                (out / f"traces_{ad}_{ts}.json").write_text(json.dumps(adapter.traces, indent=1))
            s = rep["overall_summary"]
            manifest["runs"][f"{ad}:{ts}"] = {"file": path.name, "sha256": sha256(path), "score": s["total_score"], "count": s["count"]}
            print(f"[repro] {ad} {ts}: {s['total_score']:.0f}/{s['count']} = {s['overall_accuracy']:.1%}")
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[repro] wrote {out}")


# ----------------------------------------------------------------- compare
def paired(ours: Dict, ref: Dict, test_set: str, split: str) -> Dict[str, Any]:
    keys = [k for k in ours if k in ref and (split == "all" or split_of(k[0], test_set) == split)]
    by_file: Dict[str, List[float]] = {}
    b = c = 0
    for k in keys:
        d = ours[k] - ref[k]
        by_file.setdefault(k[0], []).append(d)
        b += d > 0
        c += d < 0
    if not keys:
        return {"n": 0}
    mean, lo, hi = cluster_boot_ci(by_file)
    return {
        "n": len(keys),
        "ours": sum(ours[k] for k in keys) / len(keys),
        "ref": sum(ref[k] for k in keys) / len(keys),
        "diff": mean,
        "ci95": [lo, hi],
        "ours_wins": b,
        "ref_wins": c,
        "mcnemar_p": mcnemar_exact(b, c),
    }


def acc_ci(scores: Dict, test_set: str, split: str) -> Tuple[float, float, float]:
    by_file: Dict[str, List[float]] = {}
    for k, v in scores.items():
        if split == "all" or split_of(k[0], test_set) == split:
            by_file.setdefault(k[0], []).append(v)
    return cluster_boot_ci(by_file)


def cmd_compare(a: argparse.Namespace) -> None:
    run = ROOT / "runs" / a.run_id
    manifest = json.loads((run / "manifest.json").read_text())
    report: Dict[str, Any] = {"run_id": a.run_id, "judge_route": manifest.get("judge_route"), "sets": {}}
    lines = [f"# ATLAS comparison: run {a.run_id}", "", f"Judge route: {manifest.get('judge_route')}", ""]
    for ts in a.test_sets:
        ours_path = run / f"results_{a.adapter}_{ts}.json"
        if not ours_path.exists():
            continue
        ours = probe_scores(json.loads(ours_path.read_text()))
        entry: Dict[str, Any] = {}
        # Calibration: does our naive_rag reproduce the published naive_rag?
        calib_ok = None
        nr = run / f"results_naive_rag_{ts}.json"
        if nr.exists():
            our_nr = probe_scores(json.loads(nr.read_text()))
            pub_nr = probe_scores(published_report(ts, "naive_rag"))
            cal = paired(our_nr, pub_nr, ts, "all")
            calib_ok = cal["ci95"][0] <= 0 <= cal["ci95"][1]
            entry["calibration_naive_rag"] = {**cal, "reproduces_published": calib_ok}
        for ref_name in ["nebula", "supermemory", "mem0", "naive_rag"]:
            ref = probe_scores(published_report(ts, ref_name))
            entry[ref_name] = {sp: paired(ours, ref, ts, sp) for sp in ["dev", "heldout", "all"]}
        neb = entry["nebula"]["heldout"]
        if calib_ok is None:
            verdict = "NO CLAIM: naive_rag was not re-run, so the setup is not calibrated against the published runs."
        elif not calib_ok:
            verdict = "NO CLAIM: our naive_rag does not reproduce the published naive_rag; answer model/judge setup differs."
        elif neb["n"] and neb["ci95"][0] > 0:
            verdict = f"BEATS published Nebula on held-out files: +{neb['diff']*100:.1f} pts (95% CI {neb['ci95'][0]*100:+.1f} to {neb['ci95'][1]*100:+.1f}), McNemar p={neb['mcnemar_p']:.3g}."
        else:
            verdict = f"NOT SIGNIFICANT vs published Nebula on held-out files: {neb['diff']*100:+.1f} pts (95% CI {neb['ci95'][0]*100:+.1f} to {neb['ci95'][1]*100:+.1f})."
        entry["verdict"] = verdict
        report["sets"][ts] = entry
        m, lo, hi = acc_ci(ours, ts, "all")
        lines += [f"## {ts}", "", f"{a.adapter}: {m:.1%} (95% CI {lo:.1%}-{hi:.1%}) on {len(ours)} probes", ""]
        if "calibration_naive_rag" in entry:
            c = entry["calibration_naive_rag"]
            lines.append(f"Calibration, naive_rag re-run vs published: {c['ours']:.1%} vs {c['ref']:.1%} (diff CI {c['ci95'][0]*100:+.1f} to {c['ci95'][1]*100:+.1f} pts) -> {'reproduces' if calib_ok else 'DOES NOT reproduce'}")
            lines.append("")
        lines += ["| vs published | split | n | ours | theirs | diff (95% CI) | wins/losses | McNemar p |", "|---|---|---:|---:|---:|---|---|---:|"]
        for ref_name in ["nebula", "supermemory", "mem0", "naive_rag"]:
            for sp in ["dev", "heldout", "all"]:
                r = entry[ref_name][sp]
                if r.get("n"):
                    lines.append(f"| {ref_name} | {sp} | {r['n']} | {r['ours']:.1%} | {r['ref']:.1%} | {r['diff']*100:+.1f} ({r['ci95'][0]*100:+.1f}, {r['ci95'][1]*100:+.1f}) | {r['ours_wins']}/{r['ref_wins']} | {r['mcnemar_p']:.3g} |")
        lines += ["", f"**Verdict:** {verdict}", ""]
    (run / f"comparison_{a.adapter}.json").write_text(json.dumps(report, indent=2))
    (run / f"COMPARISON_{a.adapter}.md").write_text("\n".join(lines))
    print("\n".join(lines))


# --------------------------------------------------------------- retrieval
SESSION_REF = re.compile(r"llm[-_:\s]?(\d+)", re.I)


def gold_sessions(probe: Dict[str, Any], session_ids: List[str]) -> List[str]:
    out = []
    for item in probe["gold_answer"].get("supporting_items", []):
        for m in SESSION_REF.finditer(str(item)):
            sid = f"llm-{m.group(1)}"
            if sid in session_ids and sid not in out:
                out.append(sid)
    return out


def cmd_retrieval(a: argparse.Namespace) -> None:
    import numpy as np
    from fastembed import TextEmbedding
    from benchmark.evaluation.mycelic_loop import MycelicLoopAdapter

    os.environ.setdefault("ATLAS_LLM_BACKEND", "openai_compat")  # no LLM call is made here
    ours = MycelicLoopAdapter(user_id="retrieval")
    mini = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    ks = [1, 3, 5, 8]
    result: Dict[str, Any] = {"definition": "share of probes with >=1 parseable gold session whose gold sessions are ALL among the top-k retrieved sessions", "sets": {}}
    lines = ["# Retrieval diagnostic (no LLM)", "", result["definition"] + ".", "Baseline = repo naive_rag retrieval (one doc per turn, all-MiniLM-L6-v2, top-10 turns -> sessions in rank order).", ""]
    for ts in a.test_sets:
        hits = {"mycelic": {k: 0 for k in ks}, "naive_rag": {k: 0 for k in ks}}
        n = 0
        for p in files_for(ts):
            bench = json.loads(p.read_text())
            sids = [s["id"] for s in bench["sessions"]]
            ours.ingest_sessions(bench["sessions"])
            docs, owner = [], []
            for s in bench["sessions"]:
                for i, t in enumerate(s["turns"]):
                    docs.append(f"[Session {s['id']}, Turn {i}] {t['speaker']}: {t['text']}")
                    owner.append(s["id"])
            E = np.array(list(mini.embed(docs)), dtype=np.float32)
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            for probe in bench["probes"]:
                g = gold_sessions(probe, sids)
                if not g:
                    continue
                n += 1
                mine = ours._retrieve_sessions([probe["question"]])
                q = np.array(list(mini.embed([probe["question"]]))[0], dtype=np.float32)
                q /= np.linalg.norm(q)
                order = np.argsort(-(E @ q))[:10]
                base: List[str] = []
                for idx in order:
                    if owner[idx] not in base:
                        base.append(owner[idx])
                for k in ks:
                    hits["mycelic"][k] += all(x in mine[:k] for x in g)
                    hits["naive_rag"][k] += all(x in base[:k] for x in g)
            print(f"[retrieval] {ts} {p.name}: probes so far {n}", flush=True)
        result["sets"][ts] = {"n_probes": n, **{sys_: {f"recall@{k}": v[k] / n for k in ks} for sys_, v in hits.items()}}
        lines += [f"## {ts} ({n} probes with gold session refs)", "", "| system | " + " | ".join(f"R@{k}" for k in ks) + " |", "|---|" + "---:|" * len(ks)]
        for sys_ in hits:
            lines.append(f"| {sys_} | " + " | ".join(f"{hits[sys_][k]/n:.1%}" for k in ks) + " |")
        lines.append("")
    out = ROOT / "runs" / "retrieval_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "retrieval.json").write_text(json.dumps(result, indent=2))
    (out / "RETRIEVAL.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapters", nargs="+", default=["mycelic_loop", "naive_rag", "no_rag"])
    r.add_argument("--test-sets", nargs="+", default=["simple", "complex"], choices=list(SETS))
    r.add_argument("--max-files", type=int)
    r.add_argument("--run-id")
    r.add_argument("--no-llm-eval", action="store_true")
    r.add_argument("--quiet", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("run_id")
    c.add_argument("--adapter", default="mycelic_loop")
    c.add_argument("--test-sets", nargs="+", default=["simple", "complex"], choices=list(SETS))
    t = sub.add_parser("retrieval")
    t.add_argument("--test-sets", nargs="+", default=["simple", "complex"], choices=list(SETS))
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "retrieval": cmd_retrieval}[a.cmd](a)


if __name__ == "__main__":
    main()
