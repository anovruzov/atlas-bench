# NeuralGraph Research × ATLAS

**96.1% on ATLAS Simple. 93.8% on ATLAS Complex.**

**+19.0 and +21.1 percentage points over naive RAG on held-out questions — with the same Claude Opus answer model and the same blind judge.**

A [Mycelic Labs](https://myceliclabs.com) research evaluation of **retrieve → answer → verify** on [ATLAS](#about-atlas), a cognitive memory benchmark. The result is a three-role loop that retrieves source evidence, answers from it, then checks whether the answer deserves to survive.

[**Read the paper**](paper/atlas_mycelic.pdf) · [**Inspect every comparison**](opus_run/COMPARISON.md) · [**Review the methodology**](MYCELIC_LOOP.md) · [**Explore NeuralGraph**](https://github.com/anovruzov/NeuralGraph)

> **Scope:** These numbers belong to the **Mycelic three-agent research prototype**, which uses a Python reimplementation of retrieval ideas from NeuralGraph. The benchmark did **not** execute the production NeuralGraph Rust binary. The Opus results do **not** establish a same-answer-model victory over published Nebula, Mem0, or Supermemory runs. No comparable end-to-end ATLAS latency has been measured.

---

## Results that matter

### Matched-model comparison: Claude Opus answers, one blind Opus judge

| System | Simple (589 probes) | Complex (503 probes) |
|:--|--:|--:|
| **Mycelic retrieve → answer → verify** | **96.1%** | **93.8%** |
| Full conversation in context, no retrieval | 90.5% | 89.3% |
| Naive RAG | 78.1% | 70.8% |

Results are across **100 synthetic conversation worlds** (50 simple, 50 complex). All three rows use the same answer model and judge. Full result files, per-probe traces, and comparisons are in [`opus_run/`](opus_run/) and [`judge/`](judge/).

**Held-out results** (40 worlds per set; 469 simple and 384 complex probes):

| Comparison | Simple advantage | Complex advantage |
|:--|--:|--:|
| **Mycelic loop vs naive RAG** | **+19.0 pp** (95% CI +15.7 to +22.3) | **+21.1 pp** (95% CI +17.3 to +24.7) |
| Mycelic loop vs full context | **+7.5 pp** (95% CI +1.5 to +14.8) | +4.2 pp (95% CI −1.6 to +11.6) |

The improvement over naive RAG is supported by the held-out confidence intervals in **both** sets. The improvement over full context is supported by the bootstrap interval on Simple; on Complex the interval crosses zero. On Complex, the loop used approximately **31% of the full-history characters**.

**What changed?** Instead of treating the first retrieved passage and generated answer as final, the system checks provenance, time, entity, and missing information — then accepts, revises, or retrieves again.

### Re-judged published answers (not a controlled system ranking)

We independently re-scored the existing published answers with the same blind Opus judge, keeping ATLAS's judge prompts. These answers came from **different answer-model setups** than the Mycelic Opus experiment:

| Published system | Simple | Complex |
|:--|--:|--:|
| Nebula | 87.3% | 76.9% |
| Supermemory | 79.5% | 67.6% |
| Naive RAG | 73.2% | 61.4% |
| Mem0 | 57.7% | 53.9% |

The scores above are **judge-normalized, not answer-model-normalized**. They should not be used to claim NeuralGraph outperformed those memory systems under equal conditions. Full-context Opus alone also exceeds Nebula's published re-judged scores, illustrating why the distinction matters. See [`judge/RESCORED.md`](judge/RESCORED.md), [`audit/`](audit/), and the [paper](paper/atlas_mycelic.pdf).

---

## The architecture

```text
                     ATLAS QUESTION
                            |
                            v
       +-----------------------------------------+
       | 1. RETRIEVER                            |
       | BM25 + dense embeddings + rank fusion   |
       | Retrieve up to 8 whole, dated sessions  |
       +-----------------------------------------+
                            |
                            v
       +-----------------------------------------+
       | 2. ANSWERER                             |
       | Answer from evidence; cite sessions     |
       | Return Unknown if unsupported           |
       +-----------------------------------------+
                            |
                            v
       +-----------------------------------------+
       | 3. VERIFIER                             |
       | Check evidence, recency, entity, scope  |
       +-----------------------------------------+
             |               |              |
           ACCEPT          REVISE        RETRIEVE
             |               |              |
             +---------------+        one more round
                     |
                 FINAL ANSWER
```

The retriever uses BM25 and `BAAI/bge-small-en-v1.5`, reciprocal rank fusion (`k=60`), and session-level ranking. The verifier can request a second retrieval round. The loop is capped at two rounds, and its decisions are traceable.

**Important implementation detail:** `benchmark/evaluation/mycelic_loop.py` is a standalone **Python** implementation, inspired by NeuralGraph's retrieval path. It is **not** a wrapper around the [NeuralGraph](https://github.com/anovruzov/NeuralGraph) engine, nor does this run test NeuralGraph's SQLite persistence, MCP service, or production retrieval latency.

---

## About ATLAS

ATLAS (Cognitive-Based Benchmarking for Agentic Memory Systems) evaluates memory behavior across six dimensions: world modeling, declarative reasoning, temporal and episodic memory, preference learning, knowledge boundaries, and procedural knowledge.

The test collection comprises 50 simple and 50 complex synthetic multi-session worlds, containing **1,092 questions**. Adapters receive conversation sessions and a question; the evaluator scores their answers against held-out references. The Mycelic adapter is not given gold answers, answer types, cognitive-pillar labels, or supporting items.

## Run it

**Requirements:** Python 3.10+.

```bash
git clone https://github.com/anovruzov/atlas-bench.git
cd atlas-bench
pip install -r requirements.txt rank_bm25 fastembed
cp env.example .env
```

Configure the credentials you actually use in `.env` (or export them): `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`. Third-party memory adapters additionally require their own keys (`MEM0_API_KEY`, `SUPERMEMORY_API_KEY`, `NEBULA_API_KEY`).

### Standard ATLAS evaluations

```bash
# Simple set, all available adapters
python run_eval.py simple

# Complex set, one existing baseline
python run_eval.py complex --adapter naive_rag

# Fast wiring/heuristic smoke check, NOT a leaderboard score
python run_eval.py simple --adapter no_rag --max-files 3 --no-llm-eval
```

`run_eval.py` writes into **`evaluation_results/` by default** (override with `--output-dir`). The repository's `results/` directory contains archived benchmark results.

### Mycelic reproducible harness

```bash
# No LLM key needed: direct retriever comparison
python scripts/atlas_repro.py retrieval

# Controlled evaluation using the repo's OpenRouter GPT-4o route
export OPENROUTER_API_KEY=YOUR_KEY
python scripts/atlas_repro.py run --run-id atlas_gpt4o_v1
python scripts/atlas_repro.py compare atlas_gpt4o_v1

# Audit previously published scores
python3 -I audit/audit_published.py
```

**Do not confuse this GPT-4o reproduction path with the already-reported Opus experiment.** The Opus run used Claude Code answer agents; its artifacts and specific reproduction procedure are documented in [`MYCELIC_LOOP.md`](MYCELIC_LOOP.md), [`opus_run/`](opus_run/), and [`scripts/`](scripts/). A scored, answer-model-matched comparison to the published GPT-4o setup has **not** yet been completed.

### Optional pre-ingestion

```bash
python ingest_mem0.py simple
python run_eval.py simple --adapter mem0 --skip-ingest

python ingest_nebula.py simple
python run_eval.py simple --adapter nebula --skip-ingest
```

---

## Evidence, not just a percentage

- **Blind judge audit:** `judge/` contains the re-judging procedure and verdicts. The published answers were re-scored with one judge; the published ordering was preserved.
- **Independent audit:** `audit/` checks published arithmetic, coverage, and judging inconsistencies. In the released records, 31 of 65 `Unknown`/`Unknown` matches were originally marked incorrect.
- **Paired tests:** `opus_run/COMPARISON.md` reports file-clustered bootstrap confidence intervals and exact McNemar tests, with dev and held-out partitions separated.
- **Reproducibility:** `scripts/atlas_repro.py` records dataset and result hashes, model and prompt configuration, package versions, and git state in each run manifest.
- **Traceability:** Saved answers, retrieved sessions, verifier verdicts, and per-probe traces are provided in the research artifacts.

### Latency status

**No verified end-to-end ATLAS latency or production NeuralGraph latency is established by these results.** Retrieval, Opus answer generation, verification, and total wall-clock time must be measured on the same run before publishing an accuracy–latency claim. The separately reported **17 ms** NeuralGraph recall figure is **not** the latency of this ATLAS loop.

### Next experiment

Wire the real NeuralGraph Rust engine into the ATLAS adapter, ingest identical worlds, use the same answer model and judge, and report **retrieval p50/p95, total p50/p95, end-to-end accuracy, and confidence intervals**. Keep the existing Python loop and full-context baseline as controls. Until that run exists, the headline remains **a strong result for the NeuralGraph-inspired research loop**, not a production NeuralGraph benchmark.

---

## Repository layout

| Path | Contents |
|:--|:--|
| [`benchmark/`](benchmark/) | ATLAS generator, adapters, evaluation, metrics |
| [`simple_test_set/`](simple_test_set/) / [`complex_test_set/`](complex_test_set/) | Synthetic worlds and question sets |
| [`benchmark/evaluation/mycelic_loop.py`](benchmark/evaluation/mycelic_loop.py) | Retrieve → answer → verify implementation |
| [`opus_run/`](opus_run/) | Opus experiment results, controls, comparisons, and traces |
| [`judge/`](judge/) | Blind judge records and rescoring |
| [`audit/`](audit/) | Audits of original benchmark results |
| [`runs/`](runs/) / [`scripts/`](scripts/) | Reproduction tools, manifests, diagnostics |
| [`paper/atlas_mycelic.pdf`](paper/atlas_mycelic.pdf) | Research paper |
| [`MYCELIC_LOOP.md`](MYCELIC_LOOP.md) | Full loop methodology, limitations, reproduction |

## License

MIT — see [LICENSE](LICENSE).

**Mycelic Labs** · [NeuralGraph](https://github.com/anovruzov/NeuralGraph) · *See what others miss.*
