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
