# ATLAS controlled ablation: what each component adds

This is a **new experiment**, not a replacement for the November 2026 paper or
its 96.1% / 93.8% scores. It does not claim NeuralGraph superiority. The
repository's published result files remain untouched.

## Questions

1. Holding the **answer model and Answerer prompt fixed**, how much does
   retrieval improve or hurt relative to receiving the full history?
2. With the **same retrieved sessions**, does a verifier help?
3. Does allowing the verifier to **request a second retrieval** add value?
4. Are gains robust to unrelated-history expansion and to an independent judge?

Five arms, one fresh model call per question and step:

| Arm | Context | Answerer | Verifier | Extra retrieval |
| --- | --- | --- | --- | --- |
| full_context | All sessions | Yes | No | No |
| full_context_verify | All sessions | Yes | Yes | No |
| retrieve_answer | Hybrid top sessions | Yes | No | No |
| retrieve_verify | Hybrid top sessions | Yes | Yes | No |
| retrieve_verify_retry | Hybrid top sessions | Yes | Yes | Up to one |

The retrieve arms use the existing **Python research retriever**, not the
production NeuralGraph Rust binary. Each arm shares the same model ID, answer
prompt, temperature zero, output cap, test worlds and gold answers. Retrieval
amounts and model calls legitimately differ and are measured. Ingestion is
performed once per world, shared by all arms; ingestion time is not included in
the per-question latency. Response usage is recorded when the endpoint returns
it; empty usage is *not* evidence that calls were free.

## Run on a Mac (local inference; no cloud inference required)

Use any local OpenAI-compatible chat server, for example a local inference
server listening on port 8080. Install retrieval dependencies:

~~~bash
pip install -r requirements.txt rank_bm25 fastembed
python scripts/test_atlas_fair_ablation.py
export LOCAL_API_KEY=local
python scripts/atlas_fair_ablation.py \
  --base-url http://127.0.0.1:8080/v1 \
  --key-env LOCAL_API_KEY \
  --answer-model YOUR_LOCALLY_SERVED_MODEL \
  --test-set simple --split dev \
  --max-worlds 1 --max-probes 2 \
  --run-id fair_local_smoke_001
~~~

This is **strict exact-match scoring only**, because no independent judge was
configured. The summary's semantic accuracy is null by design, not zero.
The endpoint must support the OpenAI-compatible chat completions contract.

For stronger semantic scoring, use a **different judge model** at the same
server URL (if your server exposes both models). Alternatively, use an
OpenAI-compatible hosted endpoint if allowed by your compute/data constraints.
The script refuses to use the answer model as its judge.

~~~bash
python scripts/atlas_fair_ablation.py \
  --base-url http://127.0.0.1:8080/v1 \
  --key-env LOCAL_API_KEY \
  --answer-model YOUR_ANSWER_MODEL \
  --judge-model YOUR_DISTINCT_JUDGE_MODEL \
  --test-set both --split dev \
  --max-worlds 1 --max-probes 2 \
  --run-id fair_independent_smoke_001
~~~

For a full run after auditing the dev results, use --split heldout
--max-worlds 0 --max-probes 0 with a fresh --run-id. This runs **all five
arms over every probe** in the selected split and makes many model calls.
The same existing "held-out" ATLAS worlds were already examined in prior
research; they are not an entirely fresh external test set.

## Long-history noise stress (secondary diagnostic, not a new benchmark)

~~~bash
python scripts/atlas_fair_ablation.py \
  --base-url http://127.0.0.1:8080/v1 \
  --key-env LOCAL_API_KEY \
  --answer-model YOUR_LOCALLY_SERVED_MODEL \
  --test-set complex --split dev \
  --max-worlds 1 --max-probes 2 \
  --distractor-worlds 5 \
  --run-id fair_noise_5_001
~~~

This appends **clearly labeled, unrelated ATLAS worlds** to each target
conversation while preserving the target question and gold answer. It tests
resistance to extraneous history, **not** genuine years-long user memory.
Increase the distractor count only within the model's context capacity. This
stress setting must be reported separately from unchanged ATLAS scores.

## Scoring and outputs

Run files are under runs/<run-id>/:

- predictions.jsonl: every question and gold answer, every arm's prediction,
  sessions shown, verifier verdicts, input/output character counts, model
  usage when reported, and measured per-arm latency.
- manifest.json: model IDs, test split, probe limits, source SHA-256s and
  stress parameters; no API keys.
- summary.json: strict-exact accuracy for **all probes including Unknowns**,
  and semantic accuracy only if the different judge model is configured.
  Semantic scoring uses deterministic exact and abstention gates before the
  independent LLM judge for other answers.
- Paired differences against full context, with world-clustered bootstrap
  95% CIs only for at least five worlds; smaller smoke runs are descriptive.

**Interpretation rules:** A better loop score does not by itself prove a new
memory primitive, an improvement attributable exclusively to verification,
or superiority over Nebula / Mem0 / Supermemory. A production NeuralGraph
claim requires actually swapping in the Rust engine and re-running.

**Further checks before publication:** independent human review of a random
sample of judge disagreements; at least one distinct answer model and judge
pair; fresh real long histories rather than synthetic filler; cost/pricing
computed from documented rates; and an ablation distinguishing better
retrieval from larger retrieved context.
