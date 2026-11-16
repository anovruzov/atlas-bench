# Retrieval diagnostic (no LLM)

share of probes with >=1 parseable gold session whose gold sessions are ALL among the top-k retrieved sessions.
Baseline = repo naive_rag retrieval (one doc per turn, all-MiniLM-L6-v2, top-10 turns -> sessions in rank order).

## simple (129 probes with gold session refs)

| system | R@1 | R@3 | R@5 | R@8 |
|---|---:|---:|---:|---:|
| mycelic | 31.0% | 52.7% | 68.2% | 87.6% |
| naive_rag | 24.8% | 46.5% | 68.2% | 82.2% |

## complex (226 probes with gold session refs)

| system | R@1 | R@3 | R@5 | R@8 |
|---|---:|---:|---:|---:|
| mycelic | 34.1% | 52.7% | 61.5% | 71.2% |
| naive_rag | 31.4% | 47.3% | 52.2% | 61.5% |
