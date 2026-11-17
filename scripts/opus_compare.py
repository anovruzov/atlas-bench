#!/usr/bin/env python3
"""Paired comparison of all systems under the single blind Opus judge (judge/verdicts.json)."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import atlas_repro as ar  # noqa: E402
from opus_judge import key, parse  # noqa: E402

V = json.loads((ROOT / "judge" / "verdicts.json").read_text())


def read_batches(directory, pattern):
    merged = {}
    for path in sorted(directory.glob(pattern)):
        for item, value in json.loads(path.read_text()).items():
            if item in merged:
                raise ValueError(f"{path.relative_to(ROOT)}: duplicate item {item}")
            merged[item] = value
    return merged


def verify_inputs():
    verdicts = read_batches(ROOT / "judge" / "out", "*.json")
    if verdicts != V or any(value not in {"YES", "NO"} for value in V.values()):
        raise ValueError("Merged judge verdicts differ from the saved YES/NO batches")

    used_verdicts = set()
    for system in ("mycelic_loop", "no_rag", "naive_rag"):
        answers = read_batches(ROOT / "opus_run" / "out", f"{system}_*.json")
        expected_ids = set()
        for test_set in ar.SETS:
            report = ROOT / "opus_run" / "results" / f"results_opus_{system}_{test_set}.json"
            saved = json.loads(report.read_text())
            worlds = ar.files_for(test_set)
            blocks = saved["detailed_results"]
            if saved["missing_answers"] or len(blocks) != len(worlds):
                raise ValueError(f"{report.name}: incomplete world or answer coverage")
            for world, block in zip(worlds, blocks):
                probes = json.loads(world.read_text())["probes"]
                rows = block["probe_results"]
                if Path(block["benchmark_file"]) != world.relative_to(ROOT) or len(rows) != len(probes):
                    raise ValueError(f"{report.name}: mismatched world or probe count for {world.name}")
                for index, (probe, row) in enumerate(zip(probes, rows)):
                    item = f"{test_set}/{world.stem}|{index}"
                    expected_ids.add(item)
                    expected = (probe["id"], probe["question"], probe["answer_type"], probe["gold_answer"]["text"])
                    actual = (row["probe_id"], row["question"], row["answer_type"], row["gold_answer"])
                    answer = row["predicted_answer"]
                    if actual != expected or not answer.strip() or answers.get(item) != answer:
                        raise ValueError(f"{report.name}: probe or raw answer mismatch for {item}")
                    used_verdicts.add(key(row["answer_type"], row["gold_answer"], answer))
        if expected_ids != answers.keys():
            raise ValueError(f"{system}: raw answer batches do not cover exactly the dataset")
        print(f"{system}: {len(expected_ids)} answers match the dataset and saved batches")
    if not used_verdicts.issubset(V):
        raise ValueError("Missing judge verdicts for the matched-model runs")
    print(f"{len(V)} saved verdicts match the judge batches")


def scores(path):
    d = json.loads(Path(path).read_text())
    out = {}
    for b in d["detailed_results"]:
        stem = re.split(r"[\\/]", b["benchmark_file"])[-1].rsplit(".", 1)[0]
        for i, p in enumerate(b["probe_results"]):
            out[(stem, f"{i}:{p['probe_id']}")] = float(parse(V[key(p["answer_type"], p["gold_answer"], p["predicted_answer"])]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify inputs and compare regenerated reports without writing files")
    args = parser.parse_args()
    if args.check:
        verify_inputs()

    lines = ["# ATLAS under one blind Opus judge", ""]
    res = {}
    for ts in ["simple", "complex"]:
        S = {
            "mycelic_loop (Opus)": scores(ROOT / f"opus_run/results/results_opus_mycelic_loop_{ts}.json"),
            "no_rag full context (Opus)": scores(ROOT / f"opus_run/results/results_opus_no_rag_{ts}.json"),
            "naive_rag (Opus)": scores(ROOT / f"opus_run/results/results_opus_naive_rag_{ts}.json"),
        }
        for name in ["nebula", "supermemory", "naive_rag", "mem0"]:
            S[f"{name} (published answers)"] = scores(sorted(ar.PUBLISHED[ts].glob(f"results_{name}_*.json"))[-1])
        lines += [f"## {ts}", "", "| system | all | dev | held-out | 95% CI (all) |", "|---|---:|---:|---:|---|"]
        for n, s in S.items():
            m, lo, hi = ar.acc_ci(s, ts, "all")
            dv = ar.acc_ci(s, ts, "dev")[0]
            ho = ar.acc_ci(s, ts, "heldout")[0]
            lines.append(f"| {n} | {m:.1%} | {dv:.1%} | {ho:.1%} | {lo:.1%}–{hi:.1%} |")
        me = S["mycelic_loop (Opus)"]
        lines += ["", "Paired, mycelic_loop minus X (file-clustered bootstrap, McNemar exact):", "",
                  "| vs | split | n | diff (95% CI) | wins/losses | p |", "|---|---|---:|---|---|---:|"]
        res[ts] = {}
        for n, s in S.items():
            if n.startswith("mycelic"):
                continue
            for sp in ["heldout", "all"]:
                r = ar.paired(me, s, ts, sp)
                res[ts][f"{n}|{sp}"] = r
                lines.append(f"| {n} | {sp} | {r['n']} | {r['diff']*100:+.1f} ({r['ci95'][0]*100:+.1f}, {r['ci95'][1]*100:+.1f}) | {r['ours_wins']}/{r['ref_wins']} | {r['mcnemar_p']:.3g} |")
        lines.append("")
    reports = {
        "COMPARISON.md": "\n".join(lines),
        "comparison.json": json.dumps(res, indent=1),
    }
    for name, content in reports.items():
        path = ROOT / "opus_run" / name
        if args.check:
            if path.read_text() != content:
                raise ValueError(f"{path.relative_to(ROOT)} differs from the recomputed report")
        else:
            path.write_text(content)
    if args.check:
        print("Paired statistics and confidence intervals match both saved reports")
    else:
        print(reports["COMPARISON.md"])


if __name__ == "__main__":
    main()
