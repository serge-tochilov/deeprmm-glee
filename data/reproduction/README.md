# Deidentified metric-level tables

These 8 Parquet tables retain only the targets, frozen predictions, anonymous cluster structure, call-cost fields, and execution measurements consumed by `tools/reproduce_paper_metrics.py`. Together with the separate KI/HI table, they reproduce the paper's principal quantitative results without the private event journals or training corpora.

| File | Rows | Retained sufficient information |
|---|---:|---|
| `conditional-twin-test.parquet` | 2,928 | Family, anonymous game cluster, target action, and 3 frozen arm distributions |
| `persuasion-continuation-test.parquet` | 3,021 | Anonymous game cluster, target action, reversed-head distribution, and Markov-baseline distribution |
| `self-mirror-test.parquet` | 12,414 | Family, target type, anonymous game cluster, targets, and 2 frozen-seed outputs |
| `rating-model-heldout.parquet` | 1,438 | Family, authenticated rating delta, and frozen v3 estimate |
| `dossier-comparison.parquet` | 146 | Anonymous opponent and game clusters, low-dimensional outcomes and predictions, and successful-call resource fields |
| `execution-decisions.parquet` | 97,581 | Final-v108 membership, local or cloud-assisted route, fallback flag, and elapsed seconds |
| `execution-submissions.parquet` | 97,473 | Final-v108 membership and validity flag |
| `execution-terminal-games.parquet` | 13,125 | One row per distinct journal-terminal game and final-v108 membership |

Group labels are sequential within one table and have no meaning or linkage outside that table. Execution and rating rows have no chronology field. The tables contain no participant name, stable participant or game identifier, message, prompt, exact timestamp, account linkage, credential, private path, or model vocabulary.

Conditional-twin and self-mirror rows were projected from preserved held-out prediction files. The Persuasion rows were regenerated once from the SHA-256-bound frozen weights and corpus, then stripped to predictions and targets because the original per-row output was not retained. GPU floating-point differences change the unrounded Persuasion NLL by less than `6e-7`; the recomputed values and confidence interval equal the frozen private-source metrics at every precision reported in the paper. The rating, dossier, and execution tables are direct deidentified projections of their frozen source records.

Run `UV_CACHE_DIR=/tmp/deeprmm-glee-lab-uv-cache uv run --project artifact/nommd-arena/opponent-sequence-lab --locked --no-sync python tools/reproduce_paper_metrics.py --output /tmp/deeprmm-glee-paper-metrics.json --expected receipts/analysis/reproduced-paper-metrics.json` from the repository root. The command validates each public schema and anonymous label namespace before recomputing metrics. On the documented AMD WSL host it completed in 1.66 seconds with 135,664 KiB peak resident memory.

The tables and frozen expected receipt are licensed under `CC-BY-4.0` to the extent copyright or database rights subsist. They support evaluation reproduction at the reported precision; they do not support retraining, participant deanonymization, or replay of the nonstationary competition.
