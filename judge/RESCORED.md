# ATLAS re-judged with one judge (Claude Opus), blind to system

Same probes, same saved answers, the repo's own judge prompts; only the judge model changes.

| run | set | n | original judge | Opus judge | 95% CI (file-clustered) | flips 0→1 | flips 1→0 |
|---|---|---:|---:|---:|---|---:|---:|
| results_opus_mycelic_loop_complex | complex | 503 | 0.0% | 93.8% | 91.6%–95.7% | 472 | 0 |
| results_opus_no_rag_complex | complex | 503 | 0.0% | 89.3% | 82.4%–95.0% | 449 | 0 |
| results_nebula_complex_test_set_20260107_111020 | complex | 503 | 71.0% | 76.9% | 73.3%–80.6% | 31 | 1 |
| results_opus_naive_rag_complex | complex | 503 | 0.0% | 70.8% | 67.7%–73.8% | 356 | 0 |
| results_supermemory_complex_test_set_20251229_192151 | complex | 503 | 61.0% | 67.6% | 62.5%–72.5% | 33 | 0 |
| results_naive_rag_complex_test_set_20251229_201613 | complex | 503 | 54.9% | 61.4% | 57.8%–65.1% | 34 | 1 |
| results_mem0_complex_test_set_20251229_211903 | complex | 503 | 47.5% | 53.9% | 50.2%–57.8% | 34 | 2 |
| results_opus_mycelic_loop_simple | simple | 589 | 0.0% | 96.1% | 94.9%–97.3% | 566 | 0 |
| results_opus_no_rag_simple | simple | 589 | 0.0% | 90.5% | 84.5%–95.3% | 533 | 0 |
| results_nebula_simple_test_set_20260105_094750 | simple | 589 | 82.0% | 87.3% | 84.6%–89.9% | 33 | 2 |
| results_supermemory_simple_test_set_20260104_181156 | simple | 589 | 75.4% | 79.5% | 76.2%–82.6% | 25 | 1 |
| results_opus_naive_rag_simple | simple | 589 | 0.0% | 78.1% | 75.0%–81.1% | 460 | 0 |
| results_naive_rag_simple_test_set_20260104_205320 | simple | 589 | 68.8% | 73.2% | 70.4%–76.0% | 27 | 1 |
| results_mem0_simple_test_set_20260104_194736 | simple | 589 | 55.3% | 57.7% | 54.0%–61.4% | 18 | 4 |
