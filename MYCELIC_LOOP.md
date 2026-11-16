# Mycelic three-agent loop on ATLAS

This branch adds a memory adapter (`mycelic_loop`) and a reproducible harness
(`scripts/atlas_repro.py`). They run Mycelic's three-agent loop on ATLAS and
compare it with the published Nebula, Supermemory, Mem0 and naive RAG results,
without changing the benchmark, its test sets, or its judge.

## Status

| Item | State |
|---|---|
| Three-agent adapter (`benchmark/evaluation/mycelic_loop.py`) | **Built**; smoke-tested end to end with a small local model |
| Reproducible harness: manifest, dev/held-out split, calibration gate, paired stats | **Built**; paired stats cross-checked against an independent audit script |
| Audit of the published results (`audit/`) | **Done**; see `audit/AUDIT_PUBLISHED.md` |
| Key-free retrieval diagnostic | **Run**; see `runs/retrieval_diagnostic/RETRIEVAL.md` |
| Scored run with the published answer model and judge (GPT-4o via OpenRouter) | **Not run.** It needs an `OPENROUTER_API_KEY` |
| Any claim that Mycelic beats Nebula on ATLAS | **None yet.** It needs the scored run, and the harness must print a BEATS verdict |

## The loop

1. **Retriever.** BM25 and dense retrieval (`BAAI/bge-small-en-v1.5`) run over
   turn-pair units. Their rankings are fused with Reciprocal Rank Fusion
   (k = 60), and the best-matching whole sessions (up to 8) are returned in
   chronological order with their timestamps. This is the same design as
   NeuralGraph's recall path: ranks are fused, raw scores are not. It is
   reimplemented in Python. It does not call the NeuralGraph binary.
2. **Answerer.** Reads those sessions and returns JSON: a short answer, the
   cited session ids, and the supporting quote.
3. **Verifier.** Checks the draft against the shown sessions:
   - Is the draft supported by the text?
   - Is it the latest value?
   - Does it answer the right entity and attribute?
   - If the information is missing, the answer should be "Unknown".

   The Verifier then accepts the draft, corrects it, or sends the Retriever a
   new query. The loop runs at most 2 rounds.

The whole trace for every probe is saved to `runs/<id>/traces_*.json`. That
covers the queries, sessions shown, context size, draft, and verdict.

## Honesty rules (enforced in code)

- **Same inputs as every other adapter.** The adapter receives only the sessions
  and the question. It never sees the gold answer, answer type, pillar, or
  supporting items.
- **Judge untouched.** Scoring uses the repo's own `metrics.py` judge prompt and
  routing, unchanged.
- **Dev/held-out split.** Files are sorted by name; the first 10 per set are DEV
  and may be used to tune prompts or config. The other 40 are HELD-OUT. Results
  are always reported for both.
- **Calibration gate.** `run` re-runs `naive_rag` under the same keys. `compare`
  claims nothing unless that re-run reproduces the published naive_rag within
  its 95% CI, which is evidence that our answer model and judge match the
  published setup.
- **Claim threshold.** A BEATS verdict needs the lower bound of the paired,
  file-clustered bootstrap 95% CI of (ours − published Nebula) to be above 0 on
  the held-out files. McNemar's exact p-value is reported next to it.
- **Full-context control is always run.** One complex file is about 9k tokens,
  so the whole history fits in GPT-4o's context window. `no_rag` puts every
  session in the prompt and is part of the default run. Any memory-system
  result must be read next to it. The published comparison has no such control.
- **Mycelic's extra context is disclosed.** The loop hands the Answerer whole
  sessions, which is more context than the 10 retrieved snippets in the
  published Nebula run. Context size per probe is in the traces.

## Reproduce

```bash
pip install -r requirements.txt rank_bm25 fastembed
export OPENROUTER_API_KEY=...        # answer model and judge: openai/gpt-4o, the repo default
export HF_HUB_DISABLE_XET=1          # only needed behind proxies that block the Xet CDN

python scripts/atlas_repro.py run --run-id atlas_gpt4o_v1            # mycelic_loop + naive_rag + no_rag, both sets
python scripts/atlas_repro.py compare atlas_gpt4o_v1                 # writes runs/atlas_gpt4o_v1/COMPARISON_mycelic_loop.md
python scripts/atlas_repro.py retrieval                              # key-free retrieval diagnostic
python3 -I audit/audit_published.py                                  # re-audit the published results
```

Do not set `GOOGLE_API_KEY` for a comparison run. In the repo's code it switches
the answer model to Gemini 2.5 Flash and the judge to the `gemini-3.0-pro` path.

`runs/<id>/manifest.json` records:
- the git SHA and dirty flag
- sha256 hashes of every test file and result file
- package versions
- the judge route
- the loop's model, config and prompt hash

### Smoke test without any key

This only checks that the code runs end to end. Its scores are not comparable to
anything.

```bash
python -m llama_cpp.server --model qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8080 &
ATLAS_LLM_BACKEND=openai_compat ATLAS_LLM_BASE_URL=http://127.0.0.1:8080/v1 \
  python scripts/atlas_repro.py run --adapters mycelic_loop --test-sets simple --max-files 1 --no-llm-eval
```

## Known issues in the benchmark (from `audit/`)

- **Run setup not recorded.** The published result files don't record the answer
  model, the judge model, or the code version. The Nebula complex file contains
  a field (`scoring_latency_ms`) that no code in this repo writes.
- **The judge is noisy.** Of 65 probes where the gold answer is "Unknown" and the
  model also answered "Unknown", the judge scored 31 as wrong, across all
  systems.
- **`gemini-3.0-pro` is configured as the judge model.** The Google path falls
  back to OpenRouter GPT-4o on error, and every probe is labelled `llm_judge`
  even when the heuristic fallback ran.
- **Silent provider switch.** From reading the code (not tested here): if
  google-genai is installed, `get_llm_provider()` picks Google whenever
  `genai.Client(vertexai=True)` constructs without raising, even with no
  `GOOGLE_API_KEY`. `call_google_llm` returns "" on errors, so a misconfigured
  run can score near zero without failing. Check the manifest and the first
  answers before trusting a run.
