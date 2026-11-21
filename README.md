# ATLAS — Mycelic reproducible evaluation

**Status: historical matched-model research is archived; the live Nebula v2 head-to-head has not been run.**

This repository extends [Zeroset's ATLAS benchmark](https://github.com/zeroset-inc/atlas) with a Mycelic retrieval–answer–verification research adapter, pinned comparison artifacts, audit tools, and a live cross-system runner. The ATLAS test data originated with Zeroset; Mycelic's adapter is a Python research reimplementation, **not** the production Rust NeuralGraph binary.

## What is included

- **Archived matched-Opus study**: three-role Mycelic loop, naive RAG, and full-context controls answered with the same model and were evaluated by one system-blind judge. The results and limitations remain in the linked paper and immutable experiment artifacts, not in this README.
- **Archived-answer comparison**: one blind judge rescored saved Mycelic and published Nebula answers. The original answer models differed; this is **not** a controlled demonstration that one memory system outperforms another.
- **Live paired runner**: ingest both systems against the same pinned ATLAS worlds, answer with one fixed model/provider, use an independent blind judge, and report paired scores and resource usage. **Live results are pending.**
- **Reproducibility and audit tools**: check dataset identity, saved verdicts, manifests, provenance, and limitations. A research run is not evidence of enterprise deployments or production Rust performance.

## Independent-review materials

- [Fair evaluation protocol](docs/FAIR_ATLAS_PROTOCOL.md) — preregistered comparisons, budgets, treatment boundaries, scoring rules, and publication requirements.
- [Research paper](paper/atlas_mycelic.pdf) — archived experimental findings and analysis.
- [Research-method details](MYCELIC_LOOP.md) — precise adapter behavior, model settings, and known confounds.
- [Paired archived analysis](opus_run/COMPARISON.md) — paired estimates and uncertainty.
- [Saved model traces](opus_run/traces/) and [saved answers](opus_run/results/).
- [Judge audit](judge/RESCORED.md) and [published-results audit](audit/AUDIT_PUBLISHED.md).
- [Original ATLAS source](https://github.com/zeroset-inc/atlas).

The historical findings remain available for audit and citation. They are intentionally not used as a competitive headline or summary table here.

## Validate archived artifacts without API keys

```bash
git clone https://github.com/anovruzov/atlas-bench.git
cd atlas-bench
python -B scripts/opus_compare.py --check
python3 -I audit/audit_published.py
python -m unittest discover -s tests -p 'test_fair_*.py' -v
```

The archived checker recomputes its recorded comparisons; it does not regenerate LLM answers or confirm an unrun live comparison.

## Run the matched live experiment

Install dependencies, then provide your own authorized `NEBULA_API_KEY`, `NEBULA_WORKSPACE_ID`, and `OPENROUTER_API_KEY` locally. Never commit credentials or customer data.

```bash
python -m pip install requests rank_bm25 fastembed
python scripts/atlas_head_to_head.py atlas_nebula_smoke --smoke \
  --answer-model openai/gpt-4.1 --answer-provider openai \
  --judge-model anthropic/claude-sonnet-4.5 --judge-provider anthropic \
  --context-char-cap 12000 --mycelic-max-rounds 2
```

This command is an illustrative **development smoke test**, not a published score. Confirm availability of these exact model/provider routes before use. Run the full held-out experiment with a fresh run ID and no `--smoke` after freezing the protocol and validating the smoke test. Do not tune on those held-out outputs.

The runner records per-probe evidence context, model call counts, reported token usage where available, request and answer timings, a pinned data manifest, and the same judge route for all arms. The context cap is measured in **characters**, not equivalent tokenizer tokens; resource parity must be evaluated from actual usage and call counts.

## Interpreting evidence

A fair result must keep **three claims separate**: same-model answer quality; matched-resource efficiency; and unrestricted full-system performance. Do not attribute gains to production NeuralGraph, verified superiority over Nebula, or a single component without the appropriate live test and ablation. If usage data is absent, report it as missing rather than assuming equal compute.

MIT License.