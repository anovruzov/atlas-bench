# ATLAS fair live comparison — prospective protocol
Version 1.0 · 2026-11-21 · No live Nebula-vs-Mycelic result exists under this protocol yet.

## Research question
On the identical ATLAS questions, using a *matched answer model/provider* and a *common independent blind judge*, does the Mycelic research loop produce more correct answers than a live Nebula v2 retrieval-and-answer workflow?

**Do not call this a matched-compute comparison.** Mycelic adds a verifier and potentially a second retrieval pass, while Nebula uses one retrieval and answer. Record costs, requests, context and latency; never treat the same model ID as the same total reasoning budget.

## Study populations
Data: zeroset-inc/atlas at dbe076344d191671939fd5e3671da291f0114607. Verify tree SHAs pinned in scripts/atlas_head_to_head.py. Simple and Complex have 50 public synthetic worlds apiece. First ten lexicographic files per set are development; other 40 are study-held-out. This split was already public and exposed to prior iterations; it is NOT an independently generated never-seen holdout. A truly independent prospective dataset is required for broad generalization claims.

## Arms
- A: Mycelic research adapter, BM25 + BGE embeddings, session context, answer, verifier, optional second retrieval round; not the production Rust NeuralGraph.
- B: Nebula v2 collection per world; wait for index readiness, query live service, feed returned context to the SAME answer model/provider.
- C: Full-context answer-only control: give the same model complete world history (or deterministic capped portion), with no memory engine.
- D: For ablation, an A variant with verifier disabled should be run under a separately locked configuration. Do not infer component causality without this ablation.
- Optional naive RAG only if implemented and live-rerun with identical answer model. Do not substitute archived answers.

## Fairness and accounting
* The same ATLAS probes, gold references, answer model ID/provider, temperature and per-call max output tokens apply to A/B/C. No model/provider fallbacks. Identical judge for all arms from a different model family.
* An identical **rendered context character cap**, if used, does not establish equal tokens, relevance, requests or model compute. Publish actual input/output token accounting, model-call counts, search calls and latency when the provider exposes them. Missing cost/usage must be reported as missing.
* Prompt templates are not identical because systems play different roles; archive all templates and hashes. Report this as a *system-level* comparison, not a controlled isolated-memory comparison.
* Report ingestion time separately from query time, since Nebula ingestion is asynchronous.
* Question/reference metadata must never enter the answer generation path except for the question itself. Verify that all model prompts contain no gold labels.
* Fail closed on malformed responses, missing coverage, credential issues, judge failures and incomplete results. Report coverage and all exclusions.

## Token, dollar and latency reporting (mandatory)
For every answer (and every revision or verification call), capture provider-reported:
- Prompt/input tokens and completion/output tokens, summed across the whole investigation loop.
- Cached/read input tokens and cache-write tokens where provided; if metadata is absent, record **null**, never zero.
- Provider-reported billed dollar cost where available, otherwise **null**. Do not invent prices or infer billing from an undisclosed model rate.
- Per-query end-to-end latency in milliseconds: retrieval + all answer/verifier calls, from query initiation to final answer.
- Retrieval-only latency, number of retrieval requests, total model calls, and question-level timeout/failure status.
- p50 and p95 end-to-end latency over every completed probe, separately for each arm and test set.
- Cost per correct answer and correct answers per million *total* model tokens, only when usage coverage is complete.
- Explicit inference token and billed-cost reduction relative to the full-context arm (which can be negative).

**Cold vs warm cache:** Run and label cold/warm measurements separately when the provider supports an explicit stable prompt-cache configuration. The initial prospective runner does not guarantee a cache hit, so observed cache-token accounting is diagnostic, not an assertion of matched caching. Avoid calling a low context-char count "token savings" until actual provider usage confirms it.

**Ingestion/indexing:** Report wall time for every world's indexing and ingestion separately; the Rust NeuralGraph lane additionally performs local extraction during ingestion. Do not hide extraction or embedding time inside "free memory." These workloads can have unmeasured model usage, which is a known accounting gap until separately instrumented.

**Sequential execution:** The reported p50/p95 quantify serial synthetic-world runs including HTTP or CLI startup for the relevant lane; they are not throughput, tail-latency SLOs, or warm production-service measurements. Measure warm daemon mode independently for deployment claims.

## Real Rust NeuralGraph lane (optional, separate)
If \`--neuralgraph-bin\` is supplied, the runner executes the **actual** \`neuralgraph-memory\` CLI from an externally built and configured Rust binary:
- For each world, create an isolated SQLite database and ingest the original conversation messages without gold answers; wait for \`ingest --process\` to complete.
- Query with \`context\` and send the returned memory block to the same downstream answer model as Nebula; log retrieval latency and end-to-end latency.
- Record the Rust binary SHA-256, configured local extraction model name, embedding model name, storage path, ingestion duration, and context size.
- Synthetic ATLAS timestamps are integer session positions, not real dates; the first Rust lane does not convert them to dates. Temporal questions may therefore require separate adaptation and must be reported as a limitation.
- This is a **distinct retrieval system and ingestion pipeline** from Mycelic's Python research loop; do not pool their metrics or call the Python baseline a benchmark of shipping Rust NeuralGraph.

The Rust lane requires installed and configured local extraction and embedding models. The test runner does **not** download or install them, and it must fail closed if they are unavailable. It has not been executed with external models as part of this protocol update.

## Scoring and inference
* One blind judge prompt per unique (answer type, gold, prediction), with no arm labels. Strict YES/NO, no silent fallback. Judging may still be biased by rubric wording and model family; audit disagreements and abstentions separately.
* Report Simple and Complex separately with numerators/denominators; paired world-clustered 10,000-replicate bootstrap CIs; the same worlds and probes for each arm; signs/ties by world.
* Primary pairwise endpoint: A minus live B. Show A vs C as a separate control. Do not claim a win if either primary CI lower bound is nonpositive; do not claim superiority based only on an archived Nebula score.
* Include usage per arm, p50/p95 request latency, aggregate dollars if provider pricing is reproducibly logged, and a coverage/timeout table. No claims of equal budget without enforced and recorded total-budget constraints.
* An end-to-end result, even when positive, cannot be called production NeuralGraph superiority. The adapter is Python and NeuralGraph-inspired.

## Reproducibility release
Freeze the exact source commit, dataset tree/hash, prompts, judge and model IDs, providers, dependency versions, settings, evidence outputs, verifier traces, blinded judge prompts/verdicts, individual-world outcomes, random seeds, bootstrap implementation, and all deviations. Do not commit keys, customer data, or sensitive service request/response metadata.

## Known limitations carried forward
Original Zeroset archived results do not supply trustworthy per-answer model provenance. Mycelic's published archived-answer comparison uses different answer models. The current matched-Opus vs naive-RAG study measures the complete loop, not isolated effects. The current ATLAS world set is public and only study-held-out, with the known opportunity for prior tuning and answer contamination. Full-context's Complex CI overlaps zero. No live head-to-head may be inferred from the existence of a runnable script.

## Publication decision
The README must stay empty at the owner's request. Publish dated results in separate immutable study reports only after all required runs finish and auditable traces are archived. A failed/inconclusive experiment remains a reportable outcome.
