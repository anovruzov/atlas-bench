# Mycelic Labs on ATLAS

**The Mycelic memory loop gained 19.0 percentage points on Simple and 21.1 on Complex over naive RAG on held-out ATLAS worlds.** Both systems answered with Claude Opus and were scored by the same system-blind Opus judge. The comparison measures the complete research loop, including retrieval, prompting, and verification.

[Paper](paper/atlas_mycelic.pdf) · [Paired results](opus_run/COMPARISON.md) · [Saved answers](opus_run/results/) · [Judge verdicts](judge/verdicts.json) · [Methodology](MYCELIC_LOOP.md)

## Held-out results

The evaluation pairs answers to the same probes. Each set has 40 held-out synthetic conversation worlds. The first 10 worlds by filename in each set form the development partition.

| System | Simple · 469 probes | Complex · 384 probes |
| --- | ---: | ---: |
| **Mycelic retrieve → answer → verify** | **452/469 (96.4%)** | **357/384 (93.0%)** |
| Full conversation in context | 417/469 (88.9%) | 341/384 (88.8%) |
| Naive RAG | 363/469 (77.4%) | 276/384 (71.9%) |

| Paired lift for Mycelic | Simple · 95% CI | Complex · 95% CI |
| --- | ---: | ---: |
| vs. naive RAG | **+19.0 pp** [+15.7, +22.3] | **+21.1 pp** [+17.3, +24.7] |
| vs. full context | **+7.5 pp** [+1.5, +14.8] | +4.2 pp [−1.6, +11.6] |

The paired bootstrap resamples whole worlds (10,000 draws, seed 0) to account for questions that share a conversation history. The Complex interval against full context includes zero.

Across all 100 worlds, including development, the loop scored **566/589 (96.1%)** on Simple and **472/503 (93.8%)** on Complex. Full context scored 90.5% and 89.3%. Naive RAG scored 78.1% and 70.8%.

## Verify the published numbers

Run from a checkout with Python 3.10 or newer. This check uses the standard library and saved artifacts. It requires no API key or package installation.

```bash
git clone https://github.com/anovruzov/atlas-bench.git
cd atlas-bench
python -B scripts/opus_compare.py --check
```

The check matches each of the **3,276 answers** in the three Opus runs to its [raw batch](opus_run/out/) and the corresponding dataset probe. It checks **2,287 verdicts** against the [saved judge batches](judge/out/) and recomputes the tables, paired differences, and confidence intervals. A mismatch exits with an error. The command leaves the saved reports untouched.

The [original results commit](https://github.com/anovruzov/atlas-bench/tree/2513121a509076470bc5f0074bafbec7f3872ece) pins the data and experiment artifacts. Inspect the [per-probe verifier traces](opus_run/traces/), [machine-readable comparisons](opus_run/comparison.json), or [published-results audit](audit/AUDIT_PUBLISHED.md).

## What the experiment measures

- **Three roles.** The retriever fuses BM25 and BGE-small rankings and returns up to eight dated conversation sessions in round one. The answerer cites evidence. The verifier checks support, recency, and entity scope and can request one additional retrieval round. This run used Claude Code batches in which one subagent executed the answerer and verifier roles.
- **Controls.** Naive RAG retrieves ten turns with MiniLM. Full context supplies the entire world history. The arms share the answer model and judge but differ in prompts, evidence and batching. The experiment measures their combined effect. It has no verifier-only ablation or equal-compute control.
- **Blind scoring.** ATLAS evaluates 1,092 probes across six memory dimensions. The judge receives the benchmark's rubric, reference, and predicted answer without system identity. Opus supplied both answers and verdicts, so shared model-family bias remains possible. The paper documents an execution caveat involving an agent that reported briefly seeing two scratch files from unrelated worlds.
- **Published systems.** Saved Nebula, Supermemory, Mem0 and naive RAG answers were [re-scored with the Opus judge](judge/RESCORED.md). Their answer models and run settings cannot be verified from the published artifacts, so these scores support a rescoring comparison only.
- **Implementation scope.** The benchmark used a NeuralGraph-inspired Python prototype. Production NeuralGraph, retrieval latency, and end-to-end latency are outside the measured scope.

## Run new evaluations

```bash
python -m pip install -r requirements.txt rank_bm25 fastembed
cp env.example .env
```

Set credentials for your chosen provider in `.env` or export them. The provider options and adapter setup are documented in [the methodology](MYCELIC_LOOP.md) and [environment template](env.example).

```bash
# Standard ATLAS baseline and heuristic smoke check
python run_eval.py complex --adapter naive_rag
python run_eval.py simple --adapter no_rag --max-files 3 --no-llm-eval

# Retrieval-only diagnostic with no LLM calls (downloads an embedding model if needed)
python scripts/atlas_repro.py retrieval

# Fresh controlled run with the OpenRouter GPT-4o route
export OPENROUTER_API_KEY=YOUR_KEY
python scripts/atlas_repro.py run --run-id atlas_gpt4o_v1
python scripts/atlas_repro.py compare atlas_gpt4o_v1
```

The GPT-4o harness is a separate experiment. To regenerate the Opus experiment, see the [packet runner](scripts/opus_packets.py), [judge runner](scripts/opus_judge.py), and [paper appendix](paper/atlas_mycelic.pdf). A fresh run requires model access and executing the answer batches. Model outputs can vary across runs. Standard ATLAS evaluations write to `evaluation_results/` by default.

[MIT license](LICENSE) · [Mycelic Labs](https://myceliclabs.com) · [NeuralGraph](https://github.com/anovruzov/NeuralGraph)
