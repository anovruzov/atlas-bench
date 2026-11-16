# ATLAS under one blind Opus judge

## simple

| system | all | dev | held-out | 95% CI (all) |
|---|---:|---:|---:|---|
| mycelic_loop (Opus) | 96.1% | 95.0% | 96.4% | 94.9%–97.3% |
| no_rag full context (Opus) | 90.5% | 96.7% | 88.9% | 84.5%–95.3% |
| naive_rag (Opus) | 78.1% | 80.8% | 77.4% | 75.0%–81.1% |
| nebula (published answers) | 87.3% | 87.5% | 87.2% | 84.6%–89.9% |
| supermemory (published answers) | 79.5% | 84.2% | 78.3% | 76.2%–82.6% |
| naive_rag (published answers) | 73.2% | 77.5% | 72.1% | 70.4%–76.0% |
| mem0 (published answers) | 57.7% | 63.3% | 56.3% | 54.0%–61.4% |

Paired, mycelic_loop minus X (file-clustered bootstrap, McNemar exact):

| vs | split | n | diff (95% CI) | wins/losses | p |
|---|---|---:|---|---|---:|
| no_rag full context (Opus) | heldout | 469 | +7.5 (+1.5, +14.8) | 36/1 | 5.53e-10 |
| no_rag full context (Opus) | all | 589 | +5.6 (+0.8, +11.7) | 37/4 | 1.03e-07 |
| naive_rag (Opus) | heldout | 469 | +19.0 (+15.7, +22.3) | 92/3 | 7.22e-24 |
| naive_rag (Opus) | all | 589 | +18.0 (+15.1, +21.0) | 109/3 | 9.02e-29 |
| nebula (published answers) | heldout | 469 | +9.2 (+6.1, +12.4) | 47/4 | 2.42e-10 |
| nebula (published answers) | all | 589 | +8.8 (+6.2, +11.7) | 58/6 | 9.03e-12 |
| supermemory (published answers) | heldout | 469 | +18.1 (+14.4, +21.9) | 91/6 | 1.33e-20 |
| supermemory (published answers) | all | 589 | +16.6 (+13.5, +19.9) | 105/7 | 1.49e-23 |
| naive_rag (published answers) | heldout | 469 | +24.3 (+21.1, +27.4) | 118/4 | 3.42e-30 |
| naive_rag (published answers) | all | 589 | +22.9 (+19.9, +25.9) | 140/5 | 2.32e-35 |
| mem0 (published answers) | heldout | 469 | +40.1 (+36.0, +44.1) | 193/5 | 1.23e-50 |
| mem0 (published answers) | all | 589 | +38.4 (+34.6, +42.2) | 233/7 | 9.72e-60 |

## complex

| system | all | dev | held-out | 95% CI (all) |
|---|---:|---:|---:|---|
| mycelic_loop (Opus) | 93.8% | 96.6% | 93.0% | 91.6%–95.7% |
| no_rag full context (Opus) | 89.3% | 90.8% | 88.8% | 82.4%–95.0% |
| naive_rag (Opus) | 70.8% | 67.2% | 71.9% | 67.7%–73.8% |
| nebula (published answers) | 76.9% | 76.5% | 77.1% | 73.3%–80.6% |
| supermemory (published answers) | 67.6% | 69.7% | 66.9% | 62.5%–72.5% |
| naive_rag (published answers) | 61.4% | 58.0% | 62.5% | 57.8%–65.1% |
| mem0 (published answers) | 53.9% | 53.8% | 53.9% | 50.2%–57.8% |

Paired, mycelic_loop minus X (file-clustered bootstrap, McNemar exact):

| vs | split | n | diff (95% CI) | wins/losses | p |
|---|---|---:|---|---|---:|
| no_rag full context (Opus) | heldout | 384 | +4.2 (-1.6, +11.6) | 27/11 | 0.0139 |
| no_rag full context (Opus) | all | 503 | +4.6 (-1.0, +11.2) | 36/13 | 0.0014 |
| naive_rag (Opus) | heldout | 384 | +21.1 (+17.3, +24.7) | 87/6 | 1.65e-19 |
| naive_rag (Opus) | all | 503 | +23.1 (+19.6, +26.3) | 123/7 | 1.64e-28 |
| nebula (published answers) | heldout | 384 | +15.9 (+10.6, +20.8) | 74/13 | 1.59e-11 |
| nebula (published answers) | all | 503 | +16.9 (+12.4, +21.0) | 99/14 | 6.13e-17 |
| supermemory (published answers) | heldout | 384 | +26.0 (+19.5, +32.6) | 108/8 | 1.65e-23 |
| supermemory (published answers) | all | 503 | +26.2 (+21.0, +31.3) | 141/9 | 1.24e-31 |
| naive_rag (published answers) | heldout | 384 | +30.5 (+25.3, +35.2) | 128/11 | 1.96e-26 |
| naive_rag (published answers) | all | 503 | +32.4 (+28.1, +36.5) | 175/12 | 2.91e-38 |
| mem0 (published answers) | heldout | 384 | +39.1 (+33.2, +44.3) | 165/15 | 4.06e-33 |
| mem0 (published answers) | all | 503 | +40.0 (+35.2, +44.2) | 218/17 | 1.24e-45 |
