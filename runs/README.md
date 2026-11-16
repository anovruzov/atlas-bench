# runs/

Outputs of `scripts/atlas_repro.py`. Each run directory has a `manifest.json` that
records the git SHA, data hashes, models, judge route and config.

| Directory | What it is | Comparable to published results? |
|---|---|---|
| `smoke_local_qwen1.5b/` | Wiring check. One simple file (13 probes), local Qwen2.5-1.5B (Q4) as both Answerer and Verifier, heuristic scoring (`--no-llm-eval`). | **No.** It only shows the loop runs end to end. The 7/13 is not a benchmark score. |
| `retrieval_diagnostic/` | Key-free retrieval recall against parseable gold session refs: Mycelic Retriever vs the repo's naive_rag retrieval. | Only to naive_rag's retrieval step. It says nothing about end-to-end accuracy and nothing about Nebula. |
