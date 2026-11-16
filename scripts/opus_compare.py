#!/usr/bin/env python3
"""Paired comparison of all systems under the single blind Opus judge (judge/verdicts.json)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import atlas_repro as ar  # noqa: E402
from opus_judge import key, parse  # noqa: E402

V = json.loads((ROOT / "judge" / "verdicts.json").read_text())


def scores(path):
    d = json.loads(Path(path).read_text())
    out = {}
    for b in d["detailed_results"]:
        stem = re.split(r"[\\/]", b["benchmark_file"])[-1].rsplit(".", 1)[0]
        for i, p in enumerate(b["probe_results"]):
            out[(stem, f"{i}:{p['probe_id']}")] = float(parse(V[key(p["answer_type"], p["gold_answer"], p["predicted_answer"])]))
    return out


def main():
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
    out = ROOT / "opus_run" / "COMPARISON.md"
    out.write_text("\n".join(lines))
    (ROOT / "opus_run" / "comparison.json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
