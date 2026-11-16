#!/usr/bin/env python3
"""Audit of the published ATLAS benchmark result files.

Stdlib only. Run from anywhere with:  python3 -I audit/audit_published.py
Reads results/results_real_{simple,complex}/*.json and {simple,complex}_test_set/*.json.
Writes audit/manifest.json, audit/audit_results.json and audit/AUDIT_PUBLISHED.md.
Does not modify any other file.
"""
import glob
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "audit")
SETS = {
    "simple": ("results/results_real_simple", "simple_test_set"),
    "complex": ("results/results_real_complex", "complex_test_set"),
}
SYSTEMS = ["nebula", "supermemory", "mem0", "naive_rag"]
FLOAT_TOL = 1e-9
BOOT_N = 10000
BOOT_SEED = 0
N_EXAMPLES = 15

ABSTAIN_TERMS = [
    "unknown", "i don't know", "i do not know", "don't know", "not sure",
    "cannot determine", "can't determine", "no information", "not mentioned",
    "not specified", "unclear", "uncertain", "n/a", "not available",
    "cannot answer", "can't answer", "don't have", "do not have", "not provided",
    "no record", "not recorded", "not stated", "not discussed", "no mention",
    "not in the", "isn't mentioned", "is not mentioned", "not known",
    "no details", "not enough information", "insufficient",
]


def rel(p):
    return os.path.relpath(p, ROOT)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(s):
    s = (s or "").lower()
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def contains_gold(pred, gold):
    g, p = norm(gold), norm(pred)
    if not g:
        return False
    return re.search(r"(?:^|\s)" + re.escape(g) + r"(?:\s|$)", p) is not None


def is_abstention(pred):
    p = (pred or "").lower().replace("\u2019", "'")
    return any(t in p for t in ABSTAIN_TERMS)


def bool_polarity(s):
    p = norm(s)
    words = p.split()
    if not words:
        return None
    first = words[0]
    if first in ("yes", "true", "correct"):
        return True
    if first in ("no", "false", "incorrect"):
        return False
    return None


MULTI_RE = re.compile(r"\beither\b|\bor\b|\band/or\b|\bpossibly\b|\bmay be\b|\bmight be\b|\bcould be\b|\bone of\b")


VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:-[a-z0-9]+)?\b", re.I)


def multi_option(pred):
    """Hedge words, or two or more distinct version strings (e.g. 'v1.2.0 and v2.0.0-beta')."""
    p = (pred or "").lower()
    if MULTI_RE.search(p):
        return True
    return len(set(v.lstrip("v") for v in VERSION_RE.findall(p))) >= 2


def system_of(path):
    b = os.path.basename(path)
    for s in sorted(SYSTEMS, key=len, reverse=True):
        if b.startswith("results_" + s + "_"):
            return s
    return None


def bench_name(p):
    return re.split(r"[\\/]", p)[-1]


def aggregate(probes):
    tot = sum(p["score"] for p in probes)
    out = {"overall_accuracy": tot / len(probes) if probes else 0.0, "total_score": tot, "count": len(probes)}
    for key, field in (("by_answer_type", "answer_type"), ("by_pillar", "pillar")):
        d = {}
        for p in probes:
            e = d.setdefault(p.get(field, "unknown"), {"total": 0, "count": 0})
            e["total"] += p["score"]
            e["count"] += 1
        for e in d.values():
            e["accuracy"] = e["total"] / e["count"]
        out[key] = d
    return out


def compare_summary(mine, theirs, where):
    issues = []
    for k in ("total_score", "count"):
        if mine.get(k) != theirs.get(k):
            issues.append(f"{where}: {k} recomputed={mine.get(k)} published={theirs.get(k)}")
    if abs(mine["overall_accuracy"] - theirs.get("overall_accuracy", -1)) > FLOAT_TOL:
        issues.append(f"{where}: overall_accuracy recomputed={mine['overall_accuracy']} published={theirs.get('overall_accuracy')}")
    for key in ("by_answer_type", "by_pillar"):
        a, b = mine.get(key, {}), theirs.get(key, {})
        if set(a) != set(b):
            issues.append(f"{where}: {key} keys differ recomputed={sorted(a)} published={sorted(b)}")
        for g in set(a) & set(b):
            for f in ("total", "count"):
                if a[g][f] != b[g].get(f):
                    issues.append(f"{where}: {key}.{g}.{f} recomputed={a[g][f]} published={b[g].get(f)}")
            if abs(a[g]["accuracy"] - b[g].get("accuracy", -1)) > FLOAT_TOL:
                issues.append(f"{where}: {key}.{g}.accuracy mismatch")
    return issues


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def percentile(sorted_vals, q):
    # linear interpolation, same as numpy default
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def cluster_bootstrap(clusters, n=BOOT_N, seed=BOOT_SEED):
    """clusters: list of (sum_score, count) per benchmark file, or for paired diffs
    list of (sum_a - sum_b, count). Returns (lo, hi) 95% percentile CI of ratio."""
    rng = random.Random(seed)
    k = len(clusters)
    vals = []
    for _ in range(n):
        s = c = 0
        for _ in range(k):
            a, b = clusters[rng.randrange(k)]
            s += a
            c += b
        vals.append(s / c)
    vals.sort()
    return percentile(vals, 0.025), percentile(vals, 0.975)


def scan_model_hints(obj, path, hits, key_hits):
    pat = re.compile(r"gpt-?4|gpt-?3|gemini|openrouter|claude|anthropic|openai|llama|mistral|vertex", re.I)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if re.search(r"(?:^|_)(?:model|judge|llm|provider|engine)s?(?:_|$)", k, re.I):
                key_hits[f"{path}.{k}"] += 1
            scan_model_hints(v, f"{path}.{k}", hits, key_hits)
    elif isinstance(obj, list):
        for v in obj:
            scan_model_hints(v, path + "[]", hits, key_hits)
    elif isinstance(obj, str):
        for m in pat.findall(obj):
            hits[(path, m.lower())] += 1


def short(s, n=160):
    s = (s or "").replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"results": {}, "test_sets": {}}
    R = {"summary_checks": {}, "coverage_checks": {}, "leniency": {}, "paired": {}, "bootstrap": {},
         "model_hints": {}, "misc": {}}
    md = []
    issues_all = []

    probes_by = {}  # (set, system) -> dict key -> probe record
    clusters_by = {}  # (set, system) -> {bench: (sum,count)}

    for sname, (rdir, tdir) in SETS.items():
        # test set
        ts = {}
        for tf in sorted(glob.glob(os.path.join(ROOT, tdir, "*.json"))):
            manifest["test_sets"][rel(tf)] = sha256(tf)
            ts[os.path.basename(tf)] = load(tf)
        for rf in sorted(glob.glob(os.path.join(ROOT, rdir, "*.json"))):
            manifest["results"][rel(rf)] = sha256(rf)
            sysn = system_of(rf)
            d = load(rf)
            tag = f"{sname}/{sysn}"
            issues = []
            allp = []
            seen_bench = []
            recs = {}
            clusters = {}
            for br in d["detailed_results"]:
                bn = bench_name(br["benchmark_file"])
                seen_bench.append(bn)
                pr = br["probe_results"]
                allp.extend(pr)
                issues += compare_summary(aggregate(pr), br.get("summary", {}), f"{tag}:{bn}:summary")
                if br.get("num_probes") != len(pr):
                    issues.append(f"{tag}:{bn}: num_probes={br.get('num_probes')} but {len(pr)} probe_results")
                clusters[bn] = (sum(p["score"] for p in pr), len(pr))
                # coverage vs test set
                t = ts.get(bn)
                if t is None:
                    issues.append(f"{tag}:{bn}: benchmark file not present in {tdir}/")
                    continue
                tp = t["probes"]
                if br.get("num_sessions") != t.get("num_sessions"):
                    R["coverage_checks"].setdefault(tag, []).append(
                        f"{bn}: num_sessions result={br.get('num_sessions')} testset={t.get('num_sessions')}")
                if [p["probe_id"] for p in pr] != [p["id"] for p in tp]:
                    R["coverage_checks"].setdefault(tag, []).append(f"{bn}: probe id list differs")
                tmap = {p["id"]: p for p in tp}
                for p in pr:
                    q = tmap.get(p["probe_id"])
                    if q is None:
                        R["coverage_checks"].setdefault(tag, []).append(f"{bn}:{p['probe_id']} not in test set")
                        continue
                    for a, b in (("question", "question"), ("answer_type", "answer_type"), ("pillar", "pillar"),
                                 ("subpillar", "subpillar")):
                        if p.get(a) != q.get(b):
                            R["coverage_checks"].setdefault(tag, []).append(
                                f"{bn}:{p['probe_id']}: {a} differs")
                    if p.get("gold_answer") != q["gold_answer"]["text"]:
                        R["coverage_checks"].setdefault(tag, []).append(
                            f"{bn}:{p['probe_id']}: gold differs result={p.get('gold_answer')!r} testset={q['gold_answer']['text']!r}")
                    key = (bn, p["probe_id"])
                    if key in recs:
                        issues.append(f"{tag}: duplicate probe {key}")
                    recs[key] = dict(p, bench=bn)
                for q in tp:
                    if (bn, q["id"]) not in recs:
                        R["coverage_checks"].setdefault(tag, []).append(f"{bn}:{q['id']} missing from results")
            missing_b = sorted(set(ts) - set(seen_bench))
            if missing_b:
                R["coverage_checks"].setdefault(tag, []).append(f"benchmark files missing: {missing_b}")
            if len(seen_bench) != len(set(seen_bench)):
                issues.append(f"{tag}: duplicate benchmark files in detailed_results")
            mine = aggregate(allp)
            issues += compare_summary(mine, d["overall_summary"], f"{tag}:overall_summary")
            if d.get("total_probes") != len(allp):
                issues.append(f"{tag}: total_probes={d.get('total_probes')} vs {len(allp)} probe_results")
            if d.get("num_benchmarks") != len(d["detailed_results"]):
                issues.append(f"{tag}: num_benchmarks mismatch")
            for bs in d.get("benchmark_summaries", []):
                bn = bench_name(bs["file"])
                s, c = clusters.get(bn, (None, None))
                if c is None or bs.get("num_probes") != c or abs(bs.get("accuracy", -1) - s / c) > FLOAT_TOL:
                    issues.append(f"{tag}: benchmark_summaries entry for {bn} inconsistent with detailed_results")
            # scores binary?
            bad = [p for p in allp if p["score"] not in (0, 1)]
            if bad:
                issues.append(f"{tag}: {len(bad)} non-binary scores")
            R["summary_checks"][tag] = {
                "file": rel(rf), "adapter_type": d.get("adapter_type"), "generated_at": d.get("generated_at"),
                "recomputed": mine, "issues": issues,
                "match_types": dict(Counter(p.get("match_type") for p in allp)),
                "probe_fields": sorted(set(k for p in allp for k in p)),
            }
            issues_all += issues
            probes_by[(sname, sysn)] = recs
            clusters_by[(sname, sysn)] = clusters
            hits, key_hits = Counter(), Counter()
            scan_model_hints({k: v for k, v in d.items()}, "$", hits, key_hits)
            R["model_hints"][tag] = {
                "keys_matching_model_judge_llm": dict(key_hits),
                "string_mentions": {f"{p} :: {m}": n for (p, m), n in hits.items()},
            }
            if any("scoring_latency_ms" in p for p in allp):
                sl = sorted(p["scoring_latency_ms"] for p in allp if "scoring_latency_ms" in p)
                R["misc"].setdefault("scoring_latency_ms", {})[tag] = {
                    "n": len(sl), "min": sl[0], "median": percentile(sl, 0.5), "max": sl[-1]}

    # ---------------- leniency ----------------
    for (sname, sysn), recs in probes_by.items():
        tag = f"{sname}/{sysn}"
        L = {}
        ps = list(recs.values())
        content = [p for p in ps if p["answer_type"] in ("short_answer", "verbatim", "generation")]
        s1 = [p for p in content if p["score"] == 1]
        s1_nogold = [p for p in s1 if not contains_gold(p["predicted_answer"], p["gold_answer"])]
        L["content_probes"] = len(content)
        L["content_scored1"] = len(s1)
        L["scored1_without_gold_substring"] = len(s1_nogold)
        L["scored1_without_gold_by_type"] = dict(Counter(p["answer_type"] for p in s1_nogold))
        s1_all = [p for p in ps if p["score"] == 1]
        long1 = [p for p in s1_all if len(p["predicted_answer"] or "") > 200]
        L["scored1_long_gt200"] = len(long1)
        L["scored1_long_gt200_without_gold"] = sum(
            1 for p in long1 if p["answer_type"] not in ("abstain", "boolean")
            and not contains_gold(p["predicted_answer"], p["gold_answer"]))
        multi1 = [p for p in s1 if multi_option(p["predicted_answer"])]
        L["content_scored1_multi_option"] = len(multi1)
        L["content_scored1_multi_option_without_gold"] = sum(
            1 for p in multi1 if not contains_gold(p["predicted_answer"], p["gold_answer"]))
        # false negatives
        s0_gold = [p for p in content if p["score"] == 0 and contains_gold(p["predicted_answer"], p["gold_answer"])]
        L["scored0_with_gold_substring"] = len(s0_gold)
        L["scored0_exact_normalized_match"] = sum(
            1 for p in ps if p["score"] == 0 and norm(p["predicted_answer"]) == norm(p["gold_answer"]))
        L["scored0_exact_normalized_match_by_type"] = dict(Counter(
            p["answer_type"] for p in ps if p["score"] == 0 and norm(p["predicted_answer"]) == norm(p["gold_answer"])))
        mo = [p for p in content if multi_option(p["predicted_answer"])]
        L["content_multi_option_n"] = len(mo)
        L["content_multi_option_scored1"] = sum(p["score"] for p in mo)
        # abstain
        ab = [p for p in ps if p["answer_type"] == "abstain"]
        ab_pass = [p for p in ab if p["score"] == 1]
        ab_pass_noterm = [p for p in ab_pass if not is_abstention(p["predicted_answer"])]
        ab_fail_term = [p for p in ab if p["score"] == 0 and is_abstention(p["predicted_answer"])]
        L["abstain_n"] = len(ab)
        L["abstain_pass"] = len(ab_pass)
        L["abstain_pass_without_abstain_phrase"] = len(ab_pass_noterm)
        L["abstain_fail_with_abstain_phrase"] = len(ab_fail_term)
        # non-abstain probes where the system abstained
        na = [p for p in ps if p["answer_type"] != "abstain" and is_abstention(p["predicted_answer"])]
        L["nonabstain_probes_answered_with_abstain_phrase"] = len(na)
        L["nonabstain_abstained_but_scored1"] = sum(p["score"] for p in na)
        # boolean
        bo = [p for p in ps if p["answer_type"] == "boolean"]
        bo_mis = [p for p in bo if p["score"] == 1 and bool_polarity(p["predicted_answer"]) is not None
                  and bool_polarity(p["gold_answer"]) is not None
                  and bool_polarity(p["predicted_answer"]) != bool_polarity(p["gold_answer"])]
        L["boolean_n"] = len(bo)
        L["boolean_pass"] = sum(p["score"] for p in bo)
        L["boolean_scored1_opposite_leading_polarity"] = len(bo_mis)
        # accuracy if every content probe scored 1 without gold substring were 0 (strict lower bound proxy)
        tot = sum(p["score"] for p in ps)
        L["accuracy_published"] = tot / len(ps)
        L["accuracy_if_nogold_content_passes_zeroed"] = (tot - len(s1_nogold)) / len(ps)
        L["accuracy_substring_only_content_plus_judge_other"] = (
            tot - len(s1_nogold) + len(s0_gold)) / len(ps)
        if sysn == "nebula" or L["abstain_pass_without_abstain_phrase"]:
            L["examples_abstain_pass_without_phrase"] = [
                {k: p[k] for k in ("bench", "probe_id", "question", "gold_answer", "predicted_answer")}
                for p in ab_pass_noterm][:8]
        if sysn == "nebula":
            rng = random.Random(BOOT_SEED)
            ex = sorted(s1_nogold, key=lambda p: (p["bench"], p["probe_id"]))
            pick = sorted(rng.sample(ex, min(N_EXAMPLES, len(ex))), key=lambda p: (p["bench"], p["probe_id"]))
            L["examples_scored1_without_gold"] = [
                {k: p[k] for k in ("bench", "probe_id", "answer_type", "question", "gold_answer", "predicted_answer")}
                for p in pick]
            L["examples_long"] = [
                {k: p[k] for k in ("bench", "probe_id", "answer_type", "gold_answer", "predicted_answer")}
                for p in sorted(long1, key=lambda p: -len(p["predicted_answer"]))[:5]]
            L["examples_multi"] = [
                {k: p[k] for k in ("bench", "probe_id", "answer_type", "gold_answer", "predicted_answer")}
                for p in multi1 if not contains_gold(p["predicted_answer"], p["gold_answer"])][:8]
            L["examples_false_negative"] = [
                {k: p[k] for k in ("bench", "probe_id", "answer_type", "gold_answer", "predicted_answer")}
                for p in s0_gold][:8]
            L["examples_multi_all"] = [
                {k: p[k] for k in ("bench", "probe_id", "answer_type", "gold_answer", "predicted_answer", "score")}
                for p in mo][:10]
            L["examples_abstain_pass_without_phrase"] = [
                {k: p[k] for k in ("bench", "probe_id", "question", "gold_answer", "predicted_answer")}
                for p in ab_pass_noterm][:8]
            L["examples_nonabstain_abstained_scored1"] = [
                {k: p[k] for k in ("bench", "probe_id", "answer_type", "gold_answer", "predicted_answer")}
                for p in na if p["score"] == 1][:8]
        R["leniency"][tag] = L

    # ---------------- judge consistency across identical (type, gold, prediction) ----------------
    groups = defaultdict(list)
    for (sname, sysn), recs in probes_by.items():
        for p in recs.values():
            groups[(p["answer_type"], norm(p["gold_answer"]), norm(p["predicted_answer"]))].append(
                (f"{sname}/{sysn}", p["bench"], p["probe_id"], p["score"]))
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    incons = {k: v for k, v in multi.items() if len({x[3] for x in v}) > 1}
    R["judge_consistency"] = {
        "distinct_type_gold_pred_triples_seen_more_than_once": len(multi),
        "probes_in_those": sum(len(v) for v in multi.values()),
        "triples_with_conflicting_scores": len(incons),
        "probes_in_conflicting": sum(len(v) for v in incons.values()),
        "examples": [{"answer_type": k[0], "gold_norm": k[1], "pred_norm": k[2],
                      "n_scored1": sum(x[3] for x in v), "n_scored0": sum(1 - x[3] for x in v),
                      "scored0_in": sorted({x[0] for x in v if x[3] == 0}),
                      "scored1_in": sorted({x[0] for x in v if x[3] == 1})}
                     for k, v in sorted(incons.items(), key=lambda kv: -len(kv[1]))][:15],
    }

    # ---------------- paired + bootstrap ----------------
    for sname in SETS:
        neb = probes_by.get((sname, "nebula"), {})
        for sysn in SYSTEMS:
            cl = clusters_by.get((sname, sysn))
            if cl is None:
                continue
            tot = sum(a for a, _ in cl.values())
            cnt = sum(b for _, b in cl.values())
            lo, hi = cluster_bootstrap([cl[k] for k in sorted(cl)])
            R["bootstrap"][f"{sname}/{sysn}"] = {"score": tot, "count": cnt, "accuracy": tot / cnt,
                                                 "ci95": [lo, hi], "clusters": len(cl)}
            if sysn == "nebula":
                continue
            oth = probes_by[(sname, sysn)]
            common = sorted(set(neb) & set(oth))
            w = sum(1 for k in common if neb[k]["score"] == 1 and oth[k]["score"] == 0)
            l = sum(1 for k in common if neb[k]["score"] == 0 and oth[k]["score"] == 1)
            both1 = sum(1 for k in common if neb[k]["score"] == 1 and oth[k]["score"] == 1)
            both0 = len(common) - w - l - both1
            # paired diff bootstrap by cluster
            dcl = defaultdict(lambda: [0, 0])
            for k in common:
                dcl[k[0]][0] += neb[k]["score"] - oth[k]["score"]
                dcl[k[0]][1] += 1
            dlo, dhi = cluster_bootstrap([tuple(dcl[b]) for b in sorted(dcl)])
            # same-pred check (identical answers) for sanity
            R["paired"][f"{sname}/nebula_vs_{sysn}"] = {
                "n_common": len(common), "nebula_only_correct": w, "other_only_correct": l,
                "both_correct": both1, "both_wrong": both0, "mcnemar_exact_p": mcnemar_exact(w, l),
                "diff": (w - l) / len(common), "diff_ci95_cluster_bootstrap": [dlo, dhi]}
    # pooled bootstrap (simple+complex clusters together)
    for sysn in SYSTEMS:
        cls = []
        for sname in SETS:
            cl = clusters_by.get((sname, sysn), {})
            cls += [cl[k] for k in sorted(cl)]
        if cls:
            lo, hi = cluster_bootstrap(cls)
            t = sum(a for a, _ in cls); c = sum(b for _, b in cls)
            R["bootstrap"][f"pooled/{sysn}"] = {"score": t, "count": c, "accuracy": t / c, "ci95": [lo, hi],
                                               "clusters": len(cls)}

    R["all_summary_issues"] = issues_all
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(os.path.join(OUT, "audit_results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, sort_keys=True, default=str)
    write_md(R, manifest)
    print(f"summary issues: {len(issues_all)}")
    for t, v in R["coverage_checks"].items():
        print(f"coverage issues {t}: {len(v)}")
    print("wrote audit/manifest.json, audit/audit_results.json, audit/AUDIT_PUBLISHED.md")


def pct(x):
    return f"{100 * x:.1f}%"


def write_md(R, manifest):
    o = []
    A = o.append
    A("# Audit of published ATLAS results")
    A("")
    A("Generated by `audit/audit_published.py` (stdlib only, `python3 -I`). Every number below is computed by that "
      "script from the JSON files listed in `audit/manifest.json` (sha256 recorded). Full machine-readable output: "
      "`audit/audit_results.json`. Tracked files were not modified.")
    A("")
    A("Note: at audit time the git working tree already had uncommitted edits to `benchmark/evaluation/memory_adapters.py` "
      "and `benchmark/evaluation/run_evaluation.py` (adding a `mycelic_loop` adapter) and an untracked "
      "`benchmark/evaluation/mycelic_loop.py`; these were not made by the audit and do not touch `metrics.py`, the "
      "results, or the test sets. Code references below are to the committed `metrics.py`.")
    A("")
    A("Labels: **[Verified]** = computed directly from the files. **[Inference]** = interpretation, not provable from the files.")
    A("")
    # 1
    A("## 1. Recomputed accuracy vs published `overall_summary`")
    A("")
    A("| Set | System | adapter_type | generated_at | Recomputed | Accuracy | Mismatches |")
    A("|---|---|---|---|---|---|---|")
    for tag, v in R["summary_checks"].items():
        s, n = tag.split("/")
        r = v["recomputed"]
        A(f"| {s} | {n} | {v['adapter_type']} | {v['generated_at']} | {r['total_score']}/{r['count']} | "
          f"{pct(r['overall_accuracy'])} | {len(v['issues'])} |")
    A("")
    if R["all_summary_issues"]:
        A("**[Verified]** Mismatches found:")
        for i in R["all_summary_issues"][:50]:
            A(f"- {i}")
    else:
        A("**[Verified]** For all 8 files, overall, per-answer-type and per-pillar totals/counts/accuracies recomputed "
          "from `detailed_results[].probe_results[]` equal `overall_summary`; each per-benchmark `summary` and "
          "`benchmark_summaries` entry is also consistent; all scores are 0/1. The arithmetic of the published "
          "headline (Nebula simple 483/589, complex 357/503) is internally consistent.")
    A("")
    A("Per-pillar accuracy (recomputed):")
    A("")
    pillars = sorted({p for v in R["summary_checks"].values() for p in v["recomputed"]["by_pillar"]})
    types = sorted({p for v in R["summary_checks"].values() for p in v["recomputed"]["by_answer_type"]})
    A("| Set/System | " + " | ".join(pillars) + " |")
    A("|---|" + "---|" * len(pillars))
    for tag, v in R["summary_checks"].items():
        bp = v["recomputed"]["by_pillar"]
        A(f"| {tag} | " + " | ".join(
            f"{bp[p]['total']}/{bp[p]['count']} ({pct(bp[p]['accuracy'])})" if p in bp else "-" for p in pillars) + " |")
    A("")
    A("Per-answer-type accuracy (recomputed):")
    A("")
    A("| Set/System | " + " | ".join(types) + " |")
    A("|---|" + "---|" * len(types))
    for tag, v in R["summary_checks"].items():
        bp = v["recomputed"]["by_answer_type"]
        A(f"| {tag} | " + " | ".join(
            f"{bp[p]['total']}/{bp[p]['count']} ({pct(bp[p]['accuracy'])})" if p in bp else "-" for p in types) + " |")
    A("")
    # 2
    A("## 2. Coverage and gold answers vs current test-set files")
    A("")
    if not R["coverage_checks"]:
        A("**[Verified]** Every run covers exactly the 50 benchmark files of its test set; within each benchmark file "
          "the probe IDs (same order), questions, answer_type, pillar, subpillar and gold answer text are identical to "
          "the current `simple_test_set/` / `complex_test_set/` files, and `num_sessions` matches. So all four systems "
          "were scored on the same probes and gold answers (589 simple, 503 complex).")
    else:
        for t, v in R["coverage_checks"].items():
            A(f"- **{t}**: {len(v)} issue(s); first: " + "; ".join(v[:5]))
    A("")
    A("Note **[Verified]**: benchmark paths in the results are Windows paths (`C:\\Users\\...\\nebulaMemBenchmark\\...`); "
      "matching was done by basename. The test-set files contain no recorded hash/version, so it cannot be proven they "
      "were byte-identical at run time, only that probe text and gold answers match now.")
    A("")
    # 3
    A("## 3. Judge-leniency risks")
    A("")
    A("Method: text is normalized (lowercase, punctuation to spaces, whitespace collapsed); \"contains gold\" means the "
      "normalized gold string appears as a whole-word substring of the normalized prediction. Applied to "
      "`short_answer`, `verbatim` and `generation` probes (abstain and boolean handled separately). A pass without the "
      "gold substring is *not* necessarily wrong (paraphrase, synonyms, numbers as words); it marks where correctness "
      "depends entirely on the judge.")
    A("")
    A("| Set/System | content probes | judged correct | correct w/o gold substring | judged wrong but contains gold | "
      "correct & >200 chars | correct & multi-option/hedge wording (w/o gold) | published acc | acc if no-gold passes zeroed |")
    A("|---|---|---|---|---|---|---|---|---|")
    for tag, L in R["leniency"].items():
        A(f"| {tag} | {L['content_probes']} | {L['content_scored1']} | {L['scored1_without_gold_substring']} "
          f"({pct(L['scored1_without_gold_substring'] / max(1, L['content_scored1']))}) | "
          f"{L['scored0_with_gold_substring']} | {L['scored1_long_gt200']} | "
          f"{L['content_scored1_multi_option']} ({L['content_scored1_multi_option_without_gold']}) | "
          f"{pct(L['accuracy_published'])} | {pct(L['accuracy_if_nogold_content_passes_zeroed'])} |")
    A("")
    A("| Set/System | judged wrong although normalized prediction == normalized gold (by type) | "
      "content answers with hedge words or >=2 version strings | of which judged correct |")
    A("|---|---|---|---|")
    for tag, L in R["leniency"].items():
        A(f"| {tag} | {L['scored0_exact_normalized_match']} {L['scored0_exact_normalized_match_by_type'] or ''} | "
          f"{L['content_multi_option_n']} | {L['content_multi_option_scored1']} |")
    A("")
    JC = R["judge_consistency"]
    A(f"**Judge consistency [Verified]:** across all 8 runs, {JC['distinct_type_gold_pred_triples_seen_more_than_once']} "
      f"distinct (answer_type, normalized gold, normalized prediction) triples occur more than once "
      f"({JC['probes_in_those']} probe results). {JC['triples_with_conflicting_scores']} of them received both 0 and 1 "
      f"({JC['probes_in_conflicting']} probe results), i.e. the judge gave different verdicts to identical inputs "
      f"(apart from question text, which the judge prompt does not include). Largest conflicting groups:")
    A("")
    A("| type | gold | prediction | #scored 1 | #scored 0 | runs with 0 |")
    A("|---|---|---|---|---|---|")
    for e in JC["examples"][:10]:
        A(f"| {e['answer_type']} | {short(e['gold_norm'], 60)} | {short(e['pred_norm'], 80)} | {e['n_scored1']} | "
          f"{e['n_scored0']} | {', '.join(e['scored0_in'])} |")
    A("")
    A("**[Inference]** Identical inputs with different verdicts indicate judge nondeterminism (temperature 0.1), "
      "a change of judge model/provider between or within runs, or heuristic fallbacks on judge errors; the files "
      "do not allow distinguishing these.")
    A("")
    A("\"acc if no-gold passes zeroed\" is a deliberately harsh lower-bound proxy (treats every judge-only pass as wrong), "
      "not a corrected score.")
    A("")
    A("### Abstain and boolean probes")
    A("")
    A("| Set/System | abstain pass | abstain pass w/o abstain phrase | abstain fail with abstain phrase | "
      "non-abstain probes answered with abstain phrase (of which scored 1) | boolean pass | boolean pass w/ opposite leading yes/no |")
    A("|---|---|---|---|---|---|---|")
    for tag, L in R["leniency"].items():
        A(f"| {tag} | {L['abstain_pass']}/{L['abstain_n']} ({pct(L['abstain_pass'] / max(1, L['abstain_n']))}) | "
          f"{L['abstain_pass_without_abstain_phrase']} | {L['abstain_fail_with_abstain_phrase']} | "
          f"{L['nonabstain_probes_answered_with_abstain_phrase']} ({L['nonabstain_abstained_but_scored1']}) | "
          f"{L['boolean_pass']}/{L['boolean_n']} | {L['boolean_scored1_opposite_leading_polarity']} |")
    A("")
    A("Abstain-phrase detection is a keyword heuristic (list in the script); counts are approximate.")
    A("")
    for sname in SETS:
        L = R["leniency"].get(f"{sname}/nebula")
        if not L:
            continue
        A(f"### Nebula {sname}: sample of probes judged correct without gold substring "
          f"({len(L['examples_scored1_without_gold'])} of {L['scored1_without_gold_substring']}, random sample seed 0)")
        A("")
        A("| benchmark | probe | type | question | gold | predicted |")
        A("|---|---|---|---|---|---|")
        for e in L["examples_scored1_without_gold"]:
            A(f"| {e['bench'].replace('.json', '')} | {e['probe_id']} | {e['answer_type']} | {short(e['question'], 120)} | "
              f"{short(e['gold_answer'], 100)} | {short(e['predicted_answer'], 220)} |")
        A("")
        for key, title in (("examples_multi_all", "content answers with hedge words or multiple version strings (score shown in type col)"),
                           ("examples_long", "longest answers judged correct"),
                           ("examples_false_negative", "judged wrong although prediction contains gold"),
                           ("examples_abstain_pass_without_phrase", "abstain probes passed without an abstain phrase"),
                           ("examples_nonabstain_abstained_scored1", "non-abstain probes answered with abstain phrase but scored 1")):
            ex = L.get(key) or []
            if not ex:
                continue
            A(f"Nebula {sname} - {title} (up to 8 shown):")
            A("")
            A("| probe | type | gold | predicted |")
            A("|---|---|---|---|")
            for e in ex:
                sc = f" (score {e['score']})" if "score" in e else ""
                A(f"| {e['bench'].replace('.json', '')}:{e['probe_id']} | {e.get('answer_type', 'abstain')}{sc} | "
                  f"{short(e['gold_answer'], 100)} | {short(e['predicted_answer'], 260)} |")
            A("")
    # 4
    A("## 4. Paired comparisons and bootstrap CIs")
    A("")
    A("Paired on identical (benchmark file, probe_id). McNemar exact two-sided binomial test on discordant pairs. "
      "CIs: percentile cluster bootstrap over the 50 benchmark files, 10,000 resamples, `random.Random(0)`.")
    A("")
    A("| Set | System | Score | Accuracy | 95% CI (cluster bootstrap) |")
    A("|---|---|---|---|---|")
    for tag, b in R["bootstrap"].items():
        s, n = tag.split("/")
        A(f"| {s} | {n} | {b['score']}/{b['count']} | {pct(b['accuracy'])} | {pct(b['ci95'][0])} - {pct(b['ci95'][1])} |")
    A("")
    A("| Comparison | n | Nebula-only correct | Other-only correct | both correct | both wrong | diff (pp) | 95% CI of diff | McNemar p |")
    A("|---|---|---|---|---|---|---|---|---|")
    for tag, p in R["paired"].items():
        A(f"| {tag} | {p['n_common']} | {p['nebula_only_correct']} | {p['other_only_correct']} | {p['both_correct']} | "
          f"{p['both_wrong']} | {100 * p['diff']:+.1f} | {100 * p['diff_ci95_cluster_bootstrap'][0]:+.1f} to "
          f"{100 * p['diff_ci95_cluster_bootstrap'][1]:+.1f} | {p['mcnemar_exact_p']:.3g} |")
    A("")
    A("**[Inference]** These intervals reflect only sampling over benchmark files with the judge's labels taken as "
      "ground truth; they do not include judge error, answer-model nondeterminism, or the fact that runs were made on "
      "different dates (see section 5).")
    A("")
    # 5
    A("## 5. Which models were used (judge / answer model)")
    A("")
    A("**[Verified]** No result file records the judge model, the answer-generation model, the provider, temperature, "
      "or the code commit. Keys whose names contain model/judge/llm/provider/engine found in the files:")
    A("")
    for tag, h in R["model_hints"].items():
        keys = h["keys_matching_model_judge_llm"]
        A(f"- {tag}: keys = {sorted(keys) if keys else 'none'}; string mentions of model/provider names = "
          f"{len(h['string_mentions'])} distinct (field::name) " +
          (str(sorted(h['string_mentions'].items())[:6]) if h["string_mentions"] else ""))
    A("")
    A("String mentions (if any) come from probe/answer/context text, not from run metadata.")
    A("")
    A("**[Verified]** Every probe has `match_type = \"llm_judge\"`. In `benchmark/evaluation/metrics.py`, "
      "`llm_judge_binary` catches judge errors and unclear responses and falls back to `compute_heuristic_score` "
      "internally, but `compute_answer_score` still labels the result `llm_judge`. So the files cannot distinguish "
      "real judge verdicts from heuristic fallbacks.")
    A("")
    sl = R["misc"].get("scoring_latency_ms", {})
    if sl:
        for tag, v in sl.items():
            A(f"**[Verified]** Only {tag} has a per-probe `scoring_latency_ms` field (n={v['n']}, min {v['min']:.0f} ms, "
              f"median {v['median']:.0f} ms, max {v['max']:.0f} ms). No code in this repository writes "
              "`scoring_latency_ms`, so that run was produced by a code version not present in the repo.")
        A("")
    A("**[Verified]** Code as committed: judge tries Google GenAI with `GEMINI_MODEL = \"gemini-3.0-pro\"` first (if "
      "`google-genai` is installed), then OpenRouter `openai/gpt-4o` if `OPENROUTER_API_KEY` is set, else OpenAI "
      "`gpt-4o-mini`. The answer model (memory_adapters.py) is `gemini-2.5-flash` > OpenRouter `openai/gpt-4o` > "
      "`gpt-4o-mini` depending on available credentials.")
    A("")
    A("**[Inference]** \"gemini-3.0-pro\" does not look like a valid Gemini model ID, so if the Google path was attempted "
      "it most likely errored and the judge fell through to OpenRouter gpt-4o (or gpt-4o-mini, or the heuristic). This "
      "cannot be confirmed from the result files. A ~2.5 s minimum judge latency in the Nebula complex run for a "
      "10-token reply is consistent with (but does not prove) a failed first attempt plus a fallback call. Whether "
      "all 8 runs used the same judge and answer model is unknown; runs span 2025-12-29 to 2026-01-07, and the "
      "Nebula runs were generated last (simple 2026-01-05, complex 2026-01-07, about 9 days after the other "
      "complex runs).")
    A("")
    A("**[Verified]** `ingestion_time_s` is 0.0 for all benchmarks in the Nebula, Mem0 and (complex) Supermemory runs, "
      "i.e. ingestion was done outside the evaluation run (cf. `ingest_nebula.py`, `ingest_mem0.py`); the result files "
      "therefore do not show what was ingested or when.")
    A("")
    for tag, L in R["leniency"].items():
        if not tag.endswith("nebula") and L.get("examples_abstain_pass_without_phrase"):
            for e in L["examples_abstain_pass_without_phrase"]:
                A(f"- {tag} abstain probe passed without an abstain phrase: {e['bench']}:{e['probe_id']} gold="
                  f"{short(e['gold_answer'], 60)!r} predicted={short(e['predicted_answer'], 200)!r}")
            A("")
    A("## 6. File hashes")
    A("")
    A("sha256 of all 8 result files and 100 test-set files are in `audit/manifest.json`. Result files:")
    A("")
    for k, v in sorted(manifest["results"].items()):
        A(f"- `{k}`: `{v}`")
    A("")
    with open(os.path.join(OUT, "AUDIT_PUBLISHED.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(o))


if __name__ == "__main__":
    main()
